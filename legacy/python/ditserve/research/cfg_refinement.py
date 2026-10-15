"""A closed failed implementation may authorize one fresh paired-only retry.

The bound predecessor supplies planning measurements, never quality or cost
qualification. Its original oracle authorization remains separately verifiable.
"""

import copy
from pathlib import Path

from .common import digest, finite, read_json
from .cfg_continuation import _recipe, _sealed, verify_continuation
from .dual_evidence import read_records
from .dual_matrix import seal


SCHEMA = "ditserve.research.cfg.implementation_refinement.v1"
PURPOSE = "paired_implementation_refinement_only"


def build_refinement(parent_run, *, config, ledger):
    from .cfg_matrix import CANDIDATES, select_candidate
    from .dual_protocol import verify_protocol

    root = Path(parent_run).resolve()
    parent = verify_protocol(root / "protocol.json", verify_current=False)
    if (parent.get("schema_version") != "ditserve.research.cfg.frozen.v1"
            or not parent.get("prototype_continuation") or parent.get("implementation_refinement")):
        raise ValueError("refinement requires a direct completed split-prefix continuation")
    if _recipe(parent["config"]) != _recipe(config) or Path(parent["ledger"]).resolve() != Path(ledger).resolve():
        raise ValueError("refinement cannot change the scientific recipe or lifetime ledger")
    prototype = verify_continuation(parent["prototype_continuation"], config=config, ledger=ledger)
    budget = read_json(ledger)
    active = budget.get("active") or {}
    if active.get("protocol_sha256") == parent["sha256"]:
        raise ValueError("refinement predecessor has an unsettled session")
    if (root / "active-unit.json").exists() and read_json(root / "active-unit.json").get("status") == "running":
        raise ValueError("refinement predecessor has an active unit")
    pending = root / "pending-references.json"
    if pending.exists() and read_json(pending).get("pending") is True:
        raise ValueError("refinement predecessor still has pending work references")

    def bound(name, statuses):
        value = read_json(root / (name + ".json"))
        if (not _sealed(value) or value.get("status") not in statuses
                or value.get("protocol_sha256") != parent["sha256"]
                or value.get("source_sha256") != parent["source_sha256"]):
            raise ValueError("refinement requires a closed bound predecessor record: " + name)
        return value

    selection = bound("selection", ("direction_stopped",))
    decision = bound("calibrate", ("direction_stopped",))
    rows = read_records(root, parent)
    context = selection.get("context_sha256")
    recomputed = select_candidate(rows, config, context, frozen=parent)
    if (not isinstance(context, str) or not context
            or any(selection.get(k) != v for k, v in recomputed.items() if k != "sha256")
            or selection.get("selected") is not None or selection.get("missing")
            or decision.get("eligible") is not False
            or decision.get("selection") != recomputed
            or any(recomputed["development"].get(name, {}).get("status") != "failed"
                   or recomputed["development"][name].get("observed") != 32
                   or recomputed["development"][name].get("quality_statuses") != ["passed"] * 32
                   for name in CANDIDATES)):
        raise ValueError("refinement requires both complete quality-passing failed development selections")
    if any(r.get("type") == "cfg_calibration" and r.get("phase") == "holdout" for r in rows):
        raise ValueError("failed development refinement must not consume held-out evidence")
    sessions, phases = [], []
    for role in ("a100", "v100"):
        for kind, stage in (("correctness", "correctness"), ("calibrate", "calibration")):
            phase = kind + "-" + role
            summary = bound(phase, ("completed",))
            attempt = read_json(root / (phase + "-attempt.json"))
            job = read_json(root / (phase + "-job.json"))
            if (not _sealed(attempt) or attempt.get("status") != "completed" or attempt.get("returncode") != 0
                    or attempt.get("protocol_sha256") != parent["sha256"]
                    or job.get("protocol_sha256") != parent["sha256"]
                    or job.get("source_sha256") != parent["source_sha256"] or job.get("phase") != phase
                    or len(job.get("physical_devices", [])) != 1):
                raise ValueError("refinement predecessor supervisor/job is incomplete")
            uuid = job["physical_devices"][0]["uuid"]
            matches = [s for s in budget.get("sessions", []) if s.get("session_id") == job.get("session_id")]
            if (len(matches) != 1 or matches[0].get("status") != "completed"
                    or matches[0].get("protocol_sha256") != parent["sha256"]
                    or matches[0].get("stage") != stage or matches[0].get("devices") != [uuid]
                    or matches[0].get("gpu_count") != 1):
                raise ValueError("refinement predecessor session was not fully settled")
            finite(matches[0].get("wall_seconds"), "settled predecessor wall time", positive=True)
            from .cfg_cost_plan import _completed
            if not _completed(root, parent, phase, rows, list(CANDIDATES)):
                raise ValueError("refinement predecessor complete phase evidence is missing")
            phases.append(dict(phase=phase, summary_sha256=summary["sha256"], attempt_sha256=attempt["sha256"],
                               job_sha256=digest(job)))
            sessions.append(dict(phase=phase, session_id=matches[0]["session_id"], session_sha256=digest(matches[0])))
    references = []
    unit_seals = []
    for path in sorted((root / "units").glob("*.json")):
        unit = read_json(path)
        if (not _sealed(unit) or unit.get("status") != "completed" or unit.get("performance_valid") is not True
                or unit.get("protocol_sha256") != parent["sha256"] or unit.get("source_sha256") != parent["source_sha256"]):
            raise ValueError("refinement predecessor contains incomplete or invalid units")
        unit_seals.append(dict(unit_id=unit["unit_id"], sha256=unit["sha256"]))
    for path in root.glob("*-attempt.json"):
        attempt = read_json(path)
        if not _sealed(attempt) or attempt.get("status") != "completed" or attempt.get("returncode") != 0:
            raise ValueError("refinement predecessor still has a nonterminal or failed supervisor")
    for row in rows:
        unit = read_json(root / "units" / (row["unit_id"] + ".json"))
        if row.get("performance_valid") is not True or unit.get("status") != "completed":
            raise ValueError("refinement predecessor contains incomplete or invalid units")
        references.append(dict(record_sha256=row["sha256"], unit_id=row["unit_id"], unit_sha256=unit["sha256"]))
    return seal(dict(schema_version=SCHEMA, purpose=PURPOSE, parent_run=str(root),
        parent_protocol_sha256=parent["sha256"], parent_source_sha256=parent["source_sha256"],
        parent_context_sha256=context, parent_selection_sha256=selection["sha256"],
        parent_calibrate_sha256=decision["sha256"], prototype_continuation_sha256=prototype["sha256"],
        records=sorted(references, key=lambda x: (x["unit_id"], x["record_sha256"])), unit_seals=unit_seals, phases=phases,
        settled_sessions=sessions, active_candidates=["paired_008"],
        stopped_candidates={"independent_004": "complete predecessor development failed; no repeat"},
        target_implementation="split_prefix", expected_cache_context_mode="noop_guarded_v1",
        quality_qualified=False, performance_qualified=False, inherit_formal_evidence=False,
        correctness_mode="refinement_three_trajectories",
        evidence_scope="closed predecessor metrics for bounded implementation refinement and planning only"))


def verify_refinement(value, *, config, ledger):
    if (not _sealed(value) or value.get("schema_version") != SCHEMA
            or value.get("purpose") != PURPOSE or value.get("quality_qualified") is not False
            or value.get("performance_qualified") is not False or value.get("inherit_formal_evidence") is not False
            or not isinstance(value.get("parent_run"), str) or not Path(value["parent_run"]).is_absolute()):
        raise ValueError("invalid implementation refinement proof")
    expected = build_refinement(value["parent_run"], config=config, ledger=ledger)
    if value != expected:
        raise ValueError("implementation refinement differs from its closed predecessor evidence")
    return copy.deepcopy(expected)


def development_candidates(frozen, screens=None):
    """Do not rewrite the original oracle gates to represent stopped policies."""
    if screens is None:
        from .cfg_continuation import screening_decisions
        screens = screening_decisions(frozen, [], None)
    proof = frozen.get("implementation_refinement")
    if proof is not None:
        verified = verify_refinement(proof, config=frozen["config"], ledger=frozen["ledger"])
        if verified["prototype_continuation_sha256"] != frozen.get("prototype_continuation", {}).get("sha256"):
            raise ValueError("refinement must retain its original prototype authorization")
        return [name for name in verified["active_candidates"] if screens[name]["eligible"]]
    return [name for name, gate in screens.items() if gate["eligible"]]


def correctness_complete(rows, summary, role):
    """Recheck the three fresh trajectories, not just a summary's success bit."""
    kinds = {"cfg_correctness", "cfg_implementation_equivalence", "cfg_diagnostic_neutrality"}
    if (summary.get("checks_passed") is not True or summary.get("diagnostic_gpu_validated") is not True
            or summary.get("validation_mode") != "refinement_three_trajectories" or len(rows) != 3
            or {r.get("type") for r in rows} != kinds or len({r.get("unit_id") for r in rows}) != 1
            or any(r.get("equal") is not True or r.get("timing_eligible") is not False
                   or r.get("performance_valid") is not True or r.get("status") != "completed"
                   or r.get("device_role") != role or r.get("recipe_id") != "paired_008" for r in rows)):
        return False
    by_type = {r["type"]: r for r in rows}
    control, equivalent, neutral = (by_type[k] for k in
        ("cfg_correctness", "cfg_implementation_equivalence", "cfg_diagnostic_neutrality"))
    diagnostic, restored, oracle = control.get("continuous", {}), control.get("restored", {}), equivalent.get("continuous", {})
    runs = [diagnostic, restored, oracle]
    if (neutral.get("instrumented") != diagnostic or neutral.get("clean") != restored
            or equivalent.get("restored") != diagnostic or neutral.get("off_trajectory_latent_error_measured") is not False
            or diagnostic.get("diagnostic") is not True or restored.get("diagnostic") is not False
            or oracle.get("diagnostic") is not False
            or diagnostic.get("diagnostic_scope") != "same_input_prediction_only_correctness"
            or [r.get("identity", {}).get("implementation") for r in runs] != ["split_prefix", "split_prefix", "oracle"]
            or [r.get("action") for r in runs] != ["stay", "local_full_migrate", "stay"]):
        return False
    if (not all(guarded_split_trace(run) for run in (diagnostic, restored))
            or not any(e.get("cfg_research", {}).get("same_input_prediction_error")
                       for e in diagnostic.get("events", []) if e.get("kind") == "step")):
        return False
    for field in ("latent_sha256", "output_sha256"):
        values = [r.get(field) for r in runs]
        if not isinstance(values[0], str) or not values[0] or len(set(values)) != 1:
            return False
    traces = [r.get("traces", []) for r in runs]
    return (all(len(t) == 32 and t == traces[0] for t in traces)
            and [t.get("step") for t in traces[0]] == list(range(1, 33))
            and all(isinstance(t.get("state_sha256"), str) and t["state_sha256"] for t in traces[0]))


def guarded_split_trace(run):
    details = [e.get("cfg_research", {}).get("split_prefix", {}) for e in run.get("events", [])
               if e.get("kind") == "step"]
    if len(details) != 32 or any(d.get("cache_context_mode") != "noop_guarded_v1" for d in details):
        return False
    try:
        return all(finite(d.get("context_guard_ms"), "context guard time") >= 0 for d in details)
    except (TypeError, ValueError):
        return False


def verify_cpu_binding(binding, *, config, model_manifest, verify_arrays=True):
    """Historical reports check the manifest seal; launches also verify arrays."""
    if (not isinstance(binding, dict) or set(binding) != {"root", "manifest_sha256", "model_manifest_sha256"}
            or not isinstance(binding["root"], str) or not Path(binding["root"]).is_absolute()
            or binding["model_manifest_sha256"] != model_manifest["sha256"]):
        raise ValueError("refinement requires an explicitly bound CPU embedding cache")
    if verify_arrays:
        from .cfg_embeddings import verify_cache
        return verify_cache(binding["root"], config, model_manifest, expected_sha256=binding["manifest_sha256"])
    manifest = read_json(Path(binding["root"]) / "manifest.json")
    if (not _sealed(manifest) or manifest.get("sha256") != binding["manifest_sha256"]
            or manifest.get("schema_version") != "ditserve.research.cfg.cpu_embeddings.v1"
            or manifest.get("model_manifest_sha256") != model_manifest["sha256"]
            or manifest.get("model_path") != config["model_path"] or manifest.get("revision") != config["revision"]):
        raise ValueError("CPU embedding manifest changed or belongs to another model")
    return manifest
