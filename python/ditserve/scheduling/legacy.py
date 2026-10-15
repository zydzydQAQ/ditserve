"""Strict CPU provenance checks for completing the interrupted historical D1 matrix.

Historical files are read only. The last pair is replayed in full using the
archived inputs and executable, then combined with eleven existing full pairs.
"""

import copy
import os
from pathlib import Path
import subprocess

from ..directions import direction1 as d1
from ..research.common import atomic_json, digest, file_digest, read_json
from .reporting import complete_replay


TARGET = (103, .9, "burst")
GPU_SOURCE_FILES = (
    "python/ditserve/directions/direction1.py",
    "python/ditserve/directions/block_executor.py",
    "python/ditserve/directions/sana.py",
    "python/ditserve/gpu/worker.py",
    "python/ditserve/research/adapter.py",
)


def _require(condition, message):
    if not condition:
        raise ValueError("legacy provenance: " + message)


def _identity(record):
    return record.get("seed"), record.get("load"), record.get("pattern")


def _request_inputs(rows):
    fields = ("request_id", "kind", "arrival_ms", "deadline_ms", "service_ms", "seed")
    return [{field: row.get(field) for field in fields} for row in rows]


def _uuid(identity):
    if isinstance(identity, str):
        return identity
    if isinstance(identity, dict):
        return identity.get("uuid", identity.get("gpu_uuid", identity.get("device_uuid")))
    return None


def build_legacy_plan(legacy_run):
    root = Path(legacy_run).resolve()
    archive = root / "sources" / "direction1-online"
    manifest = read_json(archive / "manifest.json")
    _require(manifest.get("sha256") == digest({key: value for key, value in manifest.items() if key != "sha256"}),
             "archive manifest digest mismatch")
    _require(manifest.get("source_sha256") == digest(manifest.get("source_files", {})), "source list digest mismatch")
    for relative, expected in manifest["source_files"].items():
        path = archive / relative
        _require(not Path(relative).is_absolute() and archive in path.resolve().parents,
                 "archive path escapes its root")
        _require(path.is_file() and file_digest(path) == expected, "archived source changed: " + relative)
    for name, value in manifest.get("asset_manifests", {}).items():
        path = archive / "assets" / (name + ".json")
        _require(path.is_file() and file_digest(path) == value["archived_sha256"], "archived asset changed: " + name)
    files = {name: root / name for name in (
        "protocol.json", "direction1/worker-config.json", "direction1/direction1-online.json",
        "direction1/direction1-profile.json", "direction1/supervisor-online.json", "direction1/environment-online.json",
        "direction1/supervisor-correctness.json", "sources/direction1-correctness/manifest.json")}
    protocol, config, online, profile, supervisor, environment, correctness, correctness_source = (
        read_json(path) for path in files.values())
    _require(protocol.get("configuration_sha256") == digest(config) == manifest["config_sha256"],
             "frozen configuration does not match archive")
    _require(protocol.get("config") == config, "worker configuration differs from scientific protocol")
    _require(online.get("config_sha256") == profile.get("config_sha256") == d1._digest(config),
             "phase configuration digests disagree")
    _require(supervisor.get("source_sha256") == manifest["source_sha256"]
             and supervisor.get("config_sha256") == manifest["config_sha256"], "supervisor binding differs")
    _require(supervisor.get("budget_stopped") is True and not supervisor.get("supervisor_failure"),
             "historical interruption is not a clean budget stop")
    _require(supervisor.get("final_occupancy", {}).get("status") == "exclusive_observed"
             and not supervisor.get("final_occupancy", {}).get("foreign"), "historical isolation is not verified")
    _require(profile.get("eligible") is True and profile.get("selected_shape") == dict(height=480, width=832, frames=81),
             "historical full shape is not qualified")
    _require(correctness.get("status") == "passed" and correctness.get("eligible") is True
             and correctness.get("returncode") == 0 and not correctness.get("supervisor_failure")
             and correctness.get("config_sha256") == manifest["config_sha256"], "historical correctness gate is unverified")
    _require(correctness_source.get("sha256") == digest({key: value for key, value in correctness_source.items() if key != "sha256"})
             and correctness_source.get("source_sha256") == digest(correctness_source.get("source_files", {}))
             and correctness_source.get("source_sha256") == correctness.get("source_sha256"),
             "historical correctness executable binding differs")
    _require(all(correctness_source.get("source_files", {}).get(name) == manifest["source_files"].get(name)
                 for name in GPU_SOURCE_FILES), "online GPU source differs from the historical correctness gate")
    opts = d1.settings(config)
    _require(tuple(opts["policies"]) == d1.POLICIES and opts["online_count"] == 24,
             "historical policy matrix changed")
    service = online["frozen_service_ms"]
    expected_groups = {(seed, load, pattern) for seed in opts["online_seeds"]
                       for load in opts["online_loads"] for pattern in opts["patterns"]}
    _require(len(expected_groups) == 12 and TARGET in expected_groups, "historical matrix shape changed")
    grouped = {}
    for record in online.get("rows", []):
        key, policy = _identity(record), record.get("policy")
        _require(key in expected_groups and policy in opts["policies"], "unknown historical replay")
        _require(policy not in grouped.setdefault(key, {}), "duplicate historical replay")
        _require(complete_replay(record, require_quality=False), "unfinished historical replay in completed rows")
        requests = d1.frozen_requests(count=24, seed=key[0], load=key[1], pattern=key[2],
                                      video_ms=service["video"], image_ms=service["image"])
        _require(record.get("trace_sha256") == d1._digest(requests)
                 and _request_inputs(record["rows"]) == _request_inputs(requests), "historical trace inputs changed")
        grouped[key][policy] = record
    _require(len(online.get("rows", [])) == 57 and set(grouped) == expected_groups,
             "expected exactly eleven full groups and two final-group records")
    _require(set(grouped[TARGET]) == {"step", "cfg"}, "target partial group differs")
    _require(all(set(group) == set(opts["policies"]) for key, group in grouped.items() if key != TARGET),
             "an inherited group is incomplete")
    inherited = [copy.deepcopy(row) for row in online["rows"] if _identity(row) != TARGET]
    requests = d1.frozen_requests(count=24, seed=TARGET[0], load=TARGET[1], pattern=TARGET[2],
                                  video_ms=service["video"], image_ms=service["image"])
    repository = Path(__file__).resolve().parents[3]
    audit = []
    for relative in GPU_SOURCE_FILES:
        current = repository / relative
        expected = manifest["source_files"].get(relative)
        actual = file_digest(current) if current.is_file() else None
        audit.append(dict(path=relative, archived_sha256=expected, current_sha256=actual, matches=actual == expected))
    devices = supervisor["final_occupancy"]["devices"]
    _require(len(devices) == 1, "historical hardware identity must contain one A100")
    source = dict(legacy_run=str(root), archived_execution_root=str(archive),
                  manifest_sha256=file_digest(archive / "manifest.json"), source_sha256=manifest["source_sha256"],
                  config_sha256=manifest["config_sha256"], historical_files={name: file_digest(path) for name, path in files.items()},
                  dependencies=online["execution"]["dependencies"], environment=environment,
                  device_uuid=devices[0], current_gpu_source_audit=audit,
                  execution_requirement="run archived Python package; same A100, dependencies and frozen configuration")
    plan = dict(schema_version="ditserve.scheduling.legacy-plan.v1", source=source,
                inherited=inherited, inherited_count=55, excluded_partial_group=copy.deepcopy(list(grouped[TARGET].values())),
                excluded_active_trace=copy.deepcopy(online.get("active_trace")),
                target=dict(seed=TARGET[0], load=TARGET[1], pattern=TARGET[2]), target_requests=requests,
                trace_sha256=d1._digest(requests), policies=list(opts["policies"]), frozen_service_ms=service,
                config=config, profile=profile, options=opts)
    plan["sha256"] = digest(plan)
    return plan


def merge_legacy(plan, new_records):
    """Require all five new policies; never fill the last group with old arms."""
    _require(plan.get("sha256") == digest({key: value for key, value in plan.items() if key != "sha256"}),
             "completion plan mutated")
    source = plan["source"]
    root = Path(source["legacy_run"])
    for relative, expected in source["historical_files"].items():
        _require(file_digest(root / relative) == expected, "historical evidence changed: " + relative)
    _require(file_digest(Path(source["archived_execution_root"]) / "manifest.json") == source["manifest_sha256"],
             "archived manifest changed after planning")
    archive = Path(source["archived_execution_root"])
    manifest = read_json(archive / "manifest.json")
    for relative, expected in manifest["source_files"].items():
        _require(file_digest(archive / relative) == expected, "archived source changed after planning: " + relative)
    _require(len(new_records) == 5 and {record.get("policy") for record in new_records} == set(plan["policies"]),
             "a new complete five-policy group is required")
    rows = copy.deepcopy(plan["inherited"])
    for record in new_records:
        _require(_identity(record) == TARGET and record.get("trace_sha256") == plan["trace_sha256"],
                 "new replay identity differs from the frozen target")
        _require(complete_replay(record, require_quality=False), "new replay is failed or incomplete")
        _require(_request_inputs(record["rows"]) == _request_inputs(plan["target_requests"]), "new replay inputs changed")
        _require(record.get("source_sha256") == source["source_sha256"]
                 and record.get("config_sha256") == source["config_sha256"], "new replay executable/config binding differs")
        _require(_uuid(record.get("device_identity")) == source["device_uuid"], "new replay is not on the historical A100")
        dependencies = record.get("dependencies", record.get("execution", {}).get("dependencies"))
        _require(dependencies == source["dependencies"], "new replay dependency versions differ")
        item = copy.deepcopy(record)
        # Cached metrics do not control the merged gate.
        item["metrics"] = d1.request_metrics(item["rows"], item["elapsed_ms"])
        item["image_metrics"] = d1.request_metrics([r for r in item["rows"] if r["kind"] == "image"], item["elapsed_ms"])
        item["video_metrics"] = d1.request_metrics([r for r in item["rows"] if r["kind"] == "video"], item["elapsed_ms"])
        rows.append(item)
    for item in rows[:55]:
        for key, kind in (("metrics", None), ("image_metrics", "image"), ("video_metrics", "video")):
            item[key] = d1.request_metrics([row for row in item["rows"] if kind is None or row["kind"] == kind], item["elapsed_ms"])
    summaries, simple = d1.online_decision(rows, plan["options"])
    useful = any(value["meets_screen"] for value in summaries.values())
    return dict(schema_version="ditserve.scheduling.legacy-merge.v1", status="complete", rows=rows,
                inherited_replays=55, new_replays=5, paired_groups=12, summaries=summaries,
                best_simple_sufficient=simple, mechanism_benefit=useful, eligible=useful and not simple,
                decision="simple_policy_sufficient" if simple else "potential_quantum_tradeoff" if useful else "stop_no_online_benefit",
                source=copy.deepcopy(source), plan_sha256=plan["sha256"],
                excluded_partial_replays=2, excluded_active_trace=True,
                scope="historical fixed D1 gate; new session supplies one full paired group; historical evidence remains unchanged")


def legacy_admission_seconds(plan):
    """Historical measured replay maxima plus measured process overhead, at 1.3x."""
    root = Path(plan["source"]["legacy_run"])
    online = read_json(root / "direction1/direction1-online.json")
    supervisor = read_json(root / "direction1/supervisor-online.json")
    matched = [row for row in online["rows"] if row["load"] == TARGET[1] and row["pattern"] == TARGET[2]]
    policy_seconds = {policy: max(row["elapsed_ms"] for row in matched if row["policy"] == policy) / 1000.
                      for policy in plan["policies"]}
    active_seconds = (online.get("active_trace") or {}).get("elapsed_ms", 0.) / 1000.
    measured_overhead = max(0., supervisor.get("child_wall_seconds", 0.)
                            - sum(row["elapsed_ms"] for row in online["rows"]) / 1000. - active_seconds)
    # A historical full reference is the observed minimum warmup. Keeping it
    # prevents missing/short historical wall metadata from omitting warmup.
    overhead = max(measured_overhead, plan["frozen_service_ms"]["video"] / 1000.)
    return dict(policy_seconds=policy_seconds, measured_overhead_seconds=measured_overhead,
                admitted_overhead_seconds=overhead, safety_factor=1.3, cleanup_seconds=45.,
                required_seconds=1.3 * (sum(policy_seconds.values()) + overhead) + 45.,
                scope="all five arms and one archived-runtime full reference admitted before model loading")


def _invoke_bridge(config, plan, root, check_budget):
    """Use the already charged process group; the supervisor owns descendants."""
    archive = Path(plan["source"]["archived_execution_root"])
    deadline = getattr(check_budget, "deadline", None)
    if deadline is None:
        raise ValueError("legacy bridge requires the charged cooperative deadline")
    script = Path(__file__).with_name("legacy_bridge.py").resolve()
    command = [config["python"], "-u", str(script), "--plan", str(root / "plan.json"),
               "--output", str(root / "new-replays.json"), "--deadline", str(deadline),
               "--parent-pid", str(os.getpid()), "--process-group", str(os.getpgrp())]
    env = dict(os.environ, PYTHONPATH=str(archive / "python"), CUDA_VISIBLE_DEVICES=config["gpu_uuid"],
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    process = None
    try:
        with (root / "bridge.log").open("x") as log:
            process = subprocess.Popen(command, cwd=str(archive), env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=False)
            while True:
                check_budget()
                try:
                    code = process.wait(timeout=1.)
                    break
                except subprocess.TimeoutExpired:
                    pass
        raw_path = root / "new-replays.json"
        raw = read_json(raw_path) if raw_path.exists() else dict(status="failed", reason="legacy bridge wrote no evidence", records=[])
        if code != 0 or raw.get("status") != "complete":
            if raw.get("budget_stopped"):
                from .budget import BudgetStop
                raise BudgetStop(raw.get("reason", "legacy bridge allowance reached"))
            raise RuntimeError(raw.get("reason", "legacy bridge failed with exit code %s" % code))
        return raw
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.)


def run_legacy(config, run_root, check_budget, result, save):
    """Complete the old matrix without loading any current model implementation."""
    from .budget import BudgetStop
    root = Path(run_root).resolve() / "legacy"
    root.mkdir(parents=True, exist_ok=True)
    check_budget()
    if "A100" not in config.get("device_preference", ["A100", "V100"]):
        result.update(status="blocked", eligible=False, deferred=True, reason_code="device_constraint",
                      reason="legacy completion deferred: current device constraint excludes its historical A100")
        save()
        return result
    plan = build_legacy_plan(config["legacy_run"])
    if config.get("gpu_uuid") != plan["source"]["device_uuid"] or config.get("sana_profile") != "a100_bf16":
        raise ValueError("legacy completion must run on the original A100 with its original Sana recipe")
    for name in ("plan.json", "new-replays.json", "merged.json", "bridge.log"):
        if (root / name).exists():
            raise FileExistsError("preserve legacy completion evidence; use a fresh attempt: " + name)
    atomic_json(root / "plan.json", plan)
    admission = legacy_admission_seconds(plan)
    result.update(source=plan["source"], plan_sha256=plan["sha256"], inherited_replays=55, new_replays=0,
                  paired_groups=11, missing_policies=plan["policies"], admission=admission,
                  quality_scope="inherits historical D1 correctness gate; historical online replay stores metrics only; no new per-output equality claim")
    save()
    try:
        if not hasattr(check_budget, "require_seconds"):
            raise ValueError("legacy requires whole-group budget admission")
        check_budget.require_seconds(admission["required_seconds"], "all five legacy replays, archived model warmup and cleanup")
    except BudgetStop as error:
        result.update(status="incomplete", eligible=False, budget_stopped=True, reason=str(error))
        save()
        return result
    try:
        raw = _invoke_bridge(config, plan, root, check_budget)
        merged = merge_legacy(plan, raw["records"])
        atomic_json(root / "merged.json", merged)
        result.update({key: merged[key] for key in (
            "eligible", "inherited_replays", "new_replays", "paired_groups", "summaries", "best_simple_sufficient",
            "mechanism_benefit", "decision", "source", "plan_sha256", "scope")})
        result.update(status="passed", missing_policies=[], merged_path=str(root / "merged.json"),
                      merged_sha256=file_digest(root / "merged.json"),
                      new_replays_path=str(root / "new-replays.json"),
                      new_replays_sha256=file_digest(root / "new-replays.json"))
        save()
        return result
    except BaseException as error:
        raw_path = root / "new-replays.json"
        raw = read_json(raw_path) if raw_path.exists() else {}
        completed = {row.get("policy") for row in raw.get("records", [])}
        result.update(status="incomplete" if isinstance(error, BudgetStop) else "failed", eligible=False,
                      reason=str(error), new_replays=len(completed), active_trace=raw.get("active_trace"),
                      missing_policies=[policy for policy in plan["policies"] if policy not in completed],
                      new_replays_path=str(raw_path) if raw_path.exists() else None)
        save()
        raise
