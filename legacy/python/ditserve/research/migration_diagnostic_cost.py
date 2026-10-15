"""Plan the complete cache-off diagnostic from sealed measured costs.

This reads small journals only. Historical timings are planning inputs, never
quality/performance qualifications, and no budget allocation is changed here.
"""

from pathlib import Path

from .common import digest, finite, read_json
from .dual_evidence import read_records
from .dual_matrix import seal
from .migration_diagnostic_protocol import FROZEN_SCHEMA, ROOT, validate_protocol


MATH_ROOT = ROOT / "results/research-two-gpu/run02"
MATH_PROTOCOL_SHA256 = "0dc7c96f5ced633f25a8b498a393de6926a0488059edbff3cf67970922782d4e"
MATH_RECORD_SHA256 = {
    "a100": "54689ab02fcd5eccf0270a5cb0f2573cae18eeab3cf738d5dfe0cacb15220f42",
    "v100": "778427b859a20d4e9d2e8e9b839cf79b0cafa1183be02ce68466b7023d00ceb0",
    "evaluation_proxy": "f0c95f36e15737806afd93e1e931d2d9c512e199340622acee92818f5eae5369",
}
SHAPE = {"height": 256, "width": 448, "frames": 17, "steps": 32}
SAFETY = 1.3
SETUP_RESERVE = 35.
INSTRUMENTATION_RESERVE = 60.
EVALUATIONS_PER_UNIT = 8
UNITS_PER_COMPARISON = 4
GPU_COUNT = 2


def _sealed(value, label):
    if (not isinstance(value, dict) or value.get("sha256") !=
            digest({key: child for key, child in value.items() if key != "sha256"})):
        raise ValueError(label + " checksum differs")
    return value


def _history(root, frozen):
    """Validate the recorded specification without rehashing weights/arrays."""
    root = Path(root).resolve()
    history = _sealed(read_json(root / "protocol.json"), "historical protocol")
    config = history.get("config", {})
    for name in ("model_id", "model_path", "revision", "precisions", "guidance_scale", "steps"):
        if config.get(name) != frozen["config"].get(name):
            raise ValueError("historical trajectory specification differs: " + name)
    if (config.get("steps") != 32 or config.get("guidance_scale") != 5.
            or config.get("precisions") != dict(transformer="float16", vae="float32", text_encoder="float32")
            or history.get("model_manifest", {}).get("sha256") != frozen["model_manifest"]["sha256"]
            or SHAPE not in config.get("shapes", {}).values()):
        raise ValueError("historical model, precision or small-shape specification differs")
    recipes = [recipe for recipe in config.get("recipes", []) if recipe.get("id") == "off"]
    if len(recipes) != 1 or recipes[0].get("cache") is not False or recipes[0].get("threshold") != 0:
        raise ValueError("historical off recipe is not cache off")
    return history, read_records(root, history)


def _evidence(root, history, row, field):
    if row.get("performance_valid") is not True:
        raise ValueError("cost evidence belongs to an invalidated/incomplete unit")
    unit = _sealed(read_json(Path(root) / "units" / (row["unit_id"] + ".json")), "historical unit")
    wall = finite(unit["ended_at"] - unit["started_at"], "measured complete unit wall", positive=True)
    return dict(run_root=str(Path(root).resolve()), protocol_sha256=history["sha256"],
        source_sha256=history["source_sha256"], record_sha256=row["sha256"], case_id=row["case_id"],
        record_type=row["type"], unit_id=row["unit_id"], unit_sha256=unit["sha256"],
        unit_wall_seconds=wall, field=field)


def _trajectory(row, run, role, backend):
    identity = run.get("identity", {})
    steps = [event for event in run.get("events", []) if event.get("kind") == "step"]
    decodes = [event for event in run.get("events", []) if event.get("kind") == "decode"]
    if (row.get("status") != "completed" or run.get("status") != "completed"
            or run.get("recipe_id") != "off" or run.get("device_role") != role
            or run.get("all_requests_decoded") is not True or run.get("diagnostic", False)
            or identity.get("recipe") != "off" or identity.get("path") != [role]
            or identity.get("action") != "stay" or identity.get("attention_path") != [backend]
            or identity.get("implementation", "native") != "native"
            or identity.get("context_sha256") != row.get("context_sha256")
            or len(steps) != 32 or len(decodes) != 1
            or decodes[0].get("shape") != [1, 17, 256, 448, 3]
            or decodes[0].get("finite") is not True):
        raise ValueError("cost evidence is not the requested complete cache-off trajectory")
    for index, event in enumerate(steps):
        if (event.get("step_index") != index or event.get("completed_steps") != index + 1
                or event.get("finite") is not True or event.get("path") != "compute"
                or event.get("reused_forwards") != 0 or event.get("threshold") != 0.):
            raise ValueError("cache-off cost evidence lacks 32 ordered full-compute steps")
    policy = run.get("policy")
    if policy is not None:
        _sealed(policy, "historical off policy")
        if policy.get("cache") is not False or identity.get("policy_sha256") != policy["sha256"]:
            raise ValueError("historical off identity does not bind its cache policy")
    return finite(run.get("elapsed_ms"), "measured off trajectory", positive=True) / 1000.


def _original_costs(frozen, ledger):
    anchor = _sealed(frozen.get("parent_failure"), "failure anchor")
    root = Path(anchor["parent_run"])
    history, rows = _history(root, frozen)
    if (history["sha256"] != anchor.get("parent_protocol_sha256")
            or history["source_sha256"] != anchor.get("parent_source_sha256")):
        raise ValueError("original timings are not from the bound failure run")
    costs, evidence = {}, {}
    for role, backend in (("a100", "flash"), ("v100", "efficient")):
        selected = [row for row in rows if row.get("type") == "cfg_calibration"
                    and row.get("device_role") == role and row.get("phase") in ("dev", "holdout")]
        if (len(selected) != 20 or len({row["unit_id"] for row in selected}) != 20
                or sum(row["phase"] == "dev" for row in selected) != 16
                or any(row.get("fresh_pair") is not True or row.get("timing_eligible") is not True
                       for row in selected)):
            raise ValueError("original cost maximum requires all 16 development and four held-out pairs per card")
        measured = []
        for row in selected:
            ref = _evidence(root, history, row, "reference")
            seconds = _trajectory(row, row.get("reference", {}), role, backend)
            measured.append((seconds, row["sha256"], ref))
        maximum = max(measured, key=lambda item: (item[0], item[1]))
        costs[role] = maximum[0]
        evidence[role] = dict(maximum[2], elapsed_seconds=maximum[0],
            selection="maximum of every sealed development/holdout off reference on this device",
            considered_record_sha256=sorted(item[1] for item in measured))
    sessions = [item for item in ledger.get("sessions", [])
                if item.get("session_id") == anchor.get("parent_session_id")]
    if (len(sessions) != 1 or digest(sessions[0]) != anchor.get("parent_session_sha256")
            or sessions[0].get("status") != "completed" or sessions[0].get("gpu_count") != 2
            or sessions[0].get("stage") != "migration"):
        raise ValueError("setup evidence requires the settled bound dual-GPU migration session")
    units = {}
    for row in rows:
        if row["unit_id"].startswith("migration-"):
            ref = _evidence(root, history, row, None)
            units[ref["unit_id"]] = ref
    if not units:
        raise ValueError("setup evidence has no complete migration units")
    residual = finite(finite(sessions[0].get("wall_seconds"), "settled session wall", positive=True)
        - sum(unit["unit_wall_seconds"] for unit in units.values()), "session non-unit residual")
    if residual > SETUP_RESERVE:
        raise ValueError("measured setup residual exceeds the explicit 35-second planning reserve")
    evidence["setup"] = dict(parent_session_id=sessions[0]["session_id"],
        parent_session_sha256=digest(sessions[0]), measured_non_unit_wall_seconds=residual,
        estimated_setup_wall_seconds=SETUP_RESERVE, kind="rounded planning estimate from complete-session residual",
        units=list(units.values()))
    return costs, evidence


def _math_costs(frozen):
    history, rows = _history(MATH_ROOT, frozen)
    if history["sha256"] != MATH_PROTOCOL_SHA256:
        raise ValueError("MATH cost protocol differs from the pinned historical measurement")
    pinned = {}
    for name, expected in MATH_RECORD_SHA256.items():
        matches = [row for row in rows if row["sha256"] == expected]
        if len(matches) != 1:
            raise ValueError("missing pinned MATH cost record: " + name)
        pinned[name] = matches[0]
    costs, evidence = {}, {}
    for role, kind, case, field in (("a100", "backend_control", "fallback", "a100_math"),
                                     ("v100", "probe", "v100-fallback", "run")):
        row = pinned[role]
        if row.get("type") != kind or row.get("case_id") != case:
            raise ValueError("pinned MATH measurement schema differs")
        evidence[role] = _evidence(MATH_ROOT, history, row, field)
        costs[role] = _trajectory(row, row.get(field, {}), role, "math")
        evidence[role]["elapsed_seconds"] = costs[role]
    row = pinned["evaluation_proxy"]
    if row.get("type") != "backend_pair" or row.get("case_id") != "fallback-0" or row.get("shape") != SHAPE:
        raise ValueError("pinned evaluation proxy schema/shape differs")
    ref = _evidence(MATH_ROOT, history, row, "runs")
    generation = sum(_trajectory(row, row.get("runs", {}).get(backend, {}), "v100", backend)
                     for backend in ("math", "efficient"))
    proxy = finite(ref["unit_wall_seconds"] - generation, "evaluation/flush/backend-switch proxy", positive=True)
    evidence["evaluation_proxy"] = dict(ref, measured_generation_wall_seconds=generation,
        observed_residual_wall_seconds=proxy,
        kind="complete pair residual: one quality evaluation, artifact flush, backend switch and other overhead; not isolated LPIPS time")
    return costs, proxy, evidence


def cost_plan(frozen, ledger):
    """Return a sealed whole-matrix estimate; raise for missing cost evidence.

    ``unit_wall_seconds`` already includes safety. ``phase_startup_wall_seconds``
    includes setup and all four units but excludes separately charged cleanup.
    Both cards are charged throughout each comparison, including idle holding.
    """
    _sealed(frozen, "diagnostic protocol")
    if frozen.get("schema_version") != FROZEN_SCHEMA:
        raise ValueError("diagnostic cost planning requires its independent frozen protocol")
    diagnostic = validate_protocol(frozen.get("diagnostic"))
    if ledger.get("active"):
        raise ValueError("diagnostic cost planning requires a settled ledger")
    binding = frozen.get("budget_binding", {})
    stage_limit = finite(ledger.get("stage_limits_gpu_seconds", {}).get("migration"), "migration stage cap", positive=True)
    if (ledger.get("limit_gpu_seconds") != 28800. or binding.get("stage") != "migration"
            or binding.get("limit_gpu_seconds") != 28800. or binding.get("stage_limit_gpu_seconds") != stage_limit
            or binding.get("reallocation_allowed") is not False or binding.get("borrowing_allowed") is not False):
        raise ValueError("diagnostic must retain the bound lifetime and migration caps without borrowing")
    remaining = {name: max(0., finite(limit, "stage limit")
        - finite(ledger.get("stage_used_gpu_seconds", {}).get(name, 0.), "stage usage"))
        for name, limit in ledger["stage_limits_gpu_seconds"].items()}
    used = finite(ledger.get("used_gpu_seconds"), "cumulative GPU seconds")
    lifetime = max(0., ledger["limit_gpu_seconds"] - used)
    original, original_evidence = _original_costs(frozen, ledger)
    math, proxy, math_evidence = _math_costs(frozen)
    cleanup = finite(diagnostic["cleanup_wall_seconds"], "cleanup reserve", positive=True)
    comparisons = {}
    for name, costs in (("original", original), ("math", math)):
        generation = 4 * costs["a100"] + 5 * costs["v100"]
        raw_unit = generation + INSTRUMENTATION_RESERVE + EVALUATIONS_PER_UNIT * proxy
        unit = raw_unit * SAFETY
        startup = SETUP_RESERVE * SAFETY + UNITS_PER_COMPARISON * unit
        comparisons[name] = dict(unit_wall_seconds=unit, phase_startup_wall_seconds=startup,
            total_gpu_seconds=GPU_COUNT * (startup + cleanup), gpu_count=GPU_COUNT,
            complete_units=UNITS_PER_COMPARISON, cases=2, orders=2,
            measured_off_trajectory_wall_seconds=costs, trajectory_bound_multipliers=dict(a100=4, v100=5),
            generation_bound_wall_seconds=generation, unsafed_unit_wall_seconds=raw_unit,
            estimated_instrumentation_wall_seconds=INSTRUMENTATION_RESERVE,
            evaluation_count_bound=EVALUATIONS_PER_UNIT, evaluation_proxy_wall_seconds=proxy,
            estimated_setup_wall_seconds=SETUP_RESERVE, cleanup_wall_seconds=cleanup,
            per_unit_formula="1.3 * (4 * A100_off + 5 * V100_off + 60 + 8 * evaluation_proxy)")
    total = sum(value["total_gpu_seconds"] for value in comparisons.values())
    fits = total < remaining["migration"] and total < lifetime
    return seal(dict(schema_version="ditserve.research.migration_diagnostic.cost_plan.v1",
        status="fits" if fits else "budget_stopped", protocol_sha256=frozen["sha256"],
        source_sha256=frozen["source_sha256"], parent_failure_sha256=frozen["parent_failure"]["sha256"],
        budget_stage="migration", safety_factor=SAFETY, comparisons=comparisons,
        total_gpu_seconds=total, estimates_gpu_seconds=dict(migration=total),
        remaining_stage_gpu_seconds=remaining, remaining_lifetime_gpu_seconds=lifetime,
        migration_margin_gpu_seconds=remaining["migration"] - total,
        ledger_used_gpu_seconds=used, ledger_snapshot_sha256=digest(ledger),
        evidence=dict(original=original_evidence, math=math_evidence),
        quality_qualified=False, performance_qualified=False, inherit_formal_evidence=False,
        borrowing_gpu_seconds=0., optimization_savings_assumed_seconds=0.,
        instrumentation_reserve_kind="estimate: new state instrumentation, codec and array I/O; not measured",
        evaluation_proxy_limitation="observed pair residual is a planning proxy, not an isolated evaluator benchmark",
        scope="two cases by two opposite diagnostic orders for each of original and MATH; both physical GPUs held throughout",
        execution_rule="admit complete units only; update upward from observed unit cost; MATH requires original correctness"))
