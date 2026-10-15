"""Reuse audited, complete Wan local evidence for a newly charged online replay.

This module reads only JSON and source files. It does not load models, arrays, or
compiled executables, and never carries capacity/warmup state into a new process.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

from ..research.common import atomic_json, digest, file_digest
from .protocol import ROOT


_PREFIX = "python/ditserve/optimizations/"
# Explicitly reviewed orchestration/observation changes. The two other-phase
# modules are not called by the Wan local pilot or its phase-3 online replay.
_OBSERVATION = {_PREFIX + name for name in (
    "online.py", "worker.py", "runner.py", "reporting.py", "protocol.py",
    "stage_phases.py", "cache_online.py", "retention.py", "replay.py",
)}
_REQUIRED = {_PREFIX + name for name in (
    "wan.py", "wan_phases.py", "sana.py", "sana_phases.py", "cache_executor.py",
)} | {"python/ditserve/" + name for name in (
    "directions/block_executor.py", "directions/sana.py", "directions/direction1.py",
    "research/cfg_split_prefix.py", "research/cfg_adapter.py", "research/prompt_bank.py",
    "gpu/worker.py", "gpu/profiling.py", "gpu/phase_conflict.py", "gpu/quality.py",
)}
_SHAPE = [1, 81, 480, 832, 3]


def _require(condition, reason):
    if not condition:
        raise ValueError("local replay refused: " + reason)


def _read(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    value = json.loads(raw, parse_constant=invalid)
    _require(isinstance(value, dict), "evidence must be a JSON object")
    return value, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _allowed(name):
    path = Path(name)
    return (name in _OBSERVATION or (path.parent.as_posix() == "tests/python"
            and path.name.startswith("test_optimizations") and path.suffix == ".py"))


def _sources(manifest, snapshot):
    old = manifest.get("source_files", {})
    _require(isinstance(old, dict) and _REQUIRED <= old.keys(), "missing mathematical source files")
    _require(digest(old) == manifest.get("source_sha256"), "old source manifest digest differs")
    for name, expected in old.items():
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts,
                 "unsafe source path")
        source = snapshot / relative
        _require(source.is_file() and not source.is_symlink()
                 and file_digest(source) == expected, "old snapshot source differs: " + name)
    files = list((ROOT / "python/ditserve").rglob("*.py"))
    files += list((ROOT / "tests/python").glob("test_optimizations*.py"))
    files += list((ROOT / "configs").rglob("*.json"))
    current = {str(p.relative_to(ROOT)): file_digest(p) for p in files}
    _require(_REQUIRED <= current.keys(), "current mathematical source files missing")
    unchanged, changed = {}, []
    for name in sorted(old.keys() | current.keys()):
        before, after = old.get(name), current.get(name)
        if before != after:
            _require(_allowed(name), "mathematical or unclassified source changed: " + name)
            changed.append(dict(path=name, previous_sha256=before, current_sha256=after,
                                scope="explicit observation/orchestration/other-phase or test allowlist"))
        elif not _allowed(name):
            unchanged[name] = after
    return unchanged, changed, digest(current)


def _artifact(value):
    # Provenance metadata only: loading or hashing the multi-GB arrays is neither
    # necessary for replay nor permitted during another model's loading window.
    _require(isinstance(value, dict) and isinstance(value.get("path"), str)
             and Path(value["path"]).is_absolute() and value.get("shape") == _SHAPE
             and value.get("raw_available") is True
             and all(isinstance(value.get(k), str) and len(value[k]) == 64
                     for k in ("sha256", "array_sha256")), "incomplete local artifact metadata")


def _local(local, config):
    _require(local.get("schema_version") == "ditserve.optimizations.wan.v1"
             and local.get("phase") == "wan-compile" and local.get("status") == "passed"
             and local.get("local_qualified") is True and local.get("active_unit") is None
             and local.get("selected_backend") == "inductor-block1", "local pilot is not complete and qualified")
    q = config["quality"]
    _require(local.get("quality_policy") == dict(video_lpips=q["video_lpips"],
                 same_backend_resume_rtol=q["rtol"], same_backend_resume_atol=q["atol"]),
             "local numerical policy differs")
    expected_inputs = [dict(input_id="p%d-s%d" % (index, seed), prompt=prompt,
                            seed=seed, split="dev")
                       for index, prompt in enumerate(config["direction1"]["prompts"][:2])
                       for seed in (101, 102)]
    _require(len(expected_inputs) == 4 and local.get("inputs") == expected_inputs,
             "local prompt/seed inputs differ")
    inputs = {item["input_id"] for item in expected_inputs}
    rows = local.get("rows", [])
    recovery = [row for row in rows if row.get("kind") == "same_backend_resume"]
    performance = [row for row in rows if row.get("kind") == "performance"]
    _require(len(rows) == 24 and len(recovery) == 8 and len(performance) == 16,
             "local matrix must contain 8 recovery and 16 performance rows")
    _require({(row.get("backend"), row.get("input_id")) for row in recovery}
             == {(backend, item) for backend in ("eager", "inductor-block1") for item in inputs},
             "recovery pairs are missing or duplicated")
    for row in recovery:
        result = row.get("result", {})
        steps = result.get("steps", [])
        _require(result.get("passed") is True and len(steps) == 32
                 and [step.get("step") for step in steps] == list(range(1, 33))
                 and all(step.get(key) is True for step in steps
                         for key in ("latent", "solver", "cache_state", "rng", "passed")),
                 "incomplete or failed recovery trajectory")
        interruptions = result.get("interruptions", [])
        _require(len(interruptions) == 3 and [x.get("step") for x in interruptions] == [0, 8, 16]
                 and all(x.get("image_service", {}).get("finite") is True for x in interruptions),
                 "real image interruptions missing")
        final = result.get("final", {})
        _require(final.get("passed") is True and final.get("finite") is True
                 and final.get("shape") == _SHAPE and final.get("rtol") == q["rtol"]
                 and final.get("atol") == q["atol"], "recovery final numerical gate failed")
        _require(len(result.get("artifacts", [])) == 2, "recovery artifacts missing")
        for artifact in result["artifacts"]:
            _artifact(artifact)
    _require({(r.get("order"), r.get("input_id"), r.get("arm"), r.get("repeat")) for r in performance}
             == {(order, item, arm, 0) for order in range(2) for item in inputs
                 for arm in ("eager", "candidate")}, "performance pairs are missing or duplicated")
    for row in performance:
        service, quality = row.get("service", {}), row.get("quality", {})
        metrics = quality.get("metrics", {})
        values = metrics.get("lpips_per_frame", [])
        _require(row.get("complete") is True and row.get("candidate_backend") == "inductor-block1"
                 and service.get("generation_completed") is True and service.get("completed_steps") == 32
                 and service.get("cache") is False and service.get("native_cache") is False
                 and service.get("backend") == ("eager" if row["arm"] == "eager" else "inductor-block1")
                 and _number(service.get("ms"), positive=True), "incomplete performance trajectory")
        _require(quality.get("passed") is True and metrics.get("status") == "ok"
                 and metrics.get("finite") is True and metrics.get("range_valid") is True
                 and metrics.get("lpips_status") == "ok" and metrics.get("shape") == _SHAPE
                 and metrics.get("frame_count") == 81 and len(values) == 81
                 and all(_number(v) for v in values) and _number(metrics.get("lpips_mean"))
                 and metrics["lpips_mean"] <= q["video_lpips"], "performance quality gate failed")
        _artifact(row.get("artifact"))
        _require(row["artifact"]["array_sha256"] == service.get("output_sha256"),
                 "performance output/artifact identity differs")
    summary = local.get("single_block", {})
    _require(all(summary.get(k) == v for k, v in dict(complete=True, distinct_inputs=4,
                 repeats=1, orders=2, quality_passed=True, quality_checked=16,
                 performance_passed=True, n_per_arm=8, baseline_backend="eager",
                 candidate_backend="inductor-block1").items()), "local performance summary incomplete")
    gains = []
    for order in range(2):
        means = {arm: statistics.mean(r["service"]["ms"] for r in performance
                 if r["order"] == order and r["arm"] == arm) for arm in ("eager", "candidate")}
        gains.append(1 - means["candidate"] / means["eager"])
    recorded = summary.get("gains", [])
    _require(len(recorded) == 2 and all(_number(r) and math.isclose(g, r, rel_tol=1e-9, abs_tol=1e-12)
                 and g >= config["metrics"]["minimum_local_gain"] for g, r in zip(gains, recorded)),
             "local performance gain not supported by rows")
    compiled = local.get("compilation", {}).get("inductor-block1", {})
    _require(all(compiled.get(k) == v for k, v in dict(compiled=True, group_size=1,
                 fullgraph=True, dynamic=False, cuda_graphs=False, external_callable=True,
                 model_forward_replaced=False, test_compiler=False).items())
             and _number(compiled.get("compilations"), positive=True)
             and _number(compiled.get("graph_invocations"), positive=True)
             and compiled.get("compile_attempts")
             and all(x.get("status") == "completed" for x in compiled["compile_attempts"]),
             "real compiled execution evidence missing")
    _require(local.get("eager_baseline", {}).get("original_advance") is True,
             "original eager reference missing")
    return dict(recovery_rows=8, performance_rows=16, quality_checked=16,
                recomputed_gains=gains, arrays_read=False, quality_remeasured=False)


def inspect_local_replay(supervisor_path, config, output_root, *, allow_active_output=None):
    """Validate immutable local provenance; only an admitted loader allows activity."""
    run = Path(output_root).resolve(strict=True)
    path = Path(supervisor_path).resolve(strict=True)
    attempt = path.parent
    _require(path.name == "supervisor.json" and attempt.parent == run / "wan_compile"
             and attempt.name.startswith("attempt") and attempt.name[7:].isdigit(),
             "source must be a Wan attempt in the same run")
    supervisor, sb = _read(path)
    local, lb = _read(attempt / "wan-compile.json")
    audit, ab = _read(attempt / "online-measurement-audit.json")
    snapshot = run / "sources" / ("wan_compile-" + attempt.name)
    manifest, mb = _read(snapshot / "manifest.json")
    ledger, db = _read(run.parent / "gpu-budget.json")
    config_sha = digest(config)
    _require(supervisor.get("phase") == "wan_compile" and supervisor.get("output") == str(attempt)
             and supervisor.get("config_sha256") == config_sha and not supervisor.get("supervisor_failure"),
             "supervisor identity/config/failure differs")
    occupancy = supervisor.get("final_occupancy", {})
    _require(occupancy.get("status") == "exclusive_observed" and occupancy.get("owned") == []
             and occupancy.get("foreign") == [], "exclusive cleanup is unproven")
    _require(audit.get("invalidates_online_qualification") is True
             and audit.get("invalidates_local_qualification") is not True,
             "audit does not restrict invalidation to online evidence")
    charge = supervisor.get("charge", {})
    session_id = charge.get("session_id")
    _require(isinstance(session_id, str) and session_id, "charged session missing")
    settled = [s for s in ledger.get("sessions", []) if s.get("session_id") == session_id]
    _require(len(settled) == 1 and settled[0] == charge and _number(charge.get("gpu_seconds"), positive=True)
             and _number(charge.get("elapsed_seconds"), positive=True)
             and _number(charge.get("ended_at"), positive=True), "original session is not identically settled")
    identity = charge.get("identity", {})
    _require(charge.get("stage") in ("wan_compile", "reserve")
             and identity.get("phase") == "wan_compile" and identity.get("run_root") == str(run)
             and identity.get("output") == str(attempt) and identity.get("config_sha256") == config_sha
             and identity.get("source_sha256") == supervisor.get("source_sha256"), "settled session identity differs")
    active = ledger.get("active")
    if allow_active_output is None:
        _require(active is None, "ledger is active")
    else:
        target = Path(allow_active_output).resolve()
        admitted = (active or {}).get("identity", {})
        replay = admitted.get("online_replay_from", {})
        _require(target != attempt and target.parent == run / "wan_compile"
                 and active is not None and active.get("session_id") != session_id
                 and admitted.get("output") == str(target) and admitted.get("run_root") == str(run)
                 and admitted.get("phase") == "wan_compile" and admitted.get("config_sha256") == config_sha
                 and replay.get("path") == str(path) and replay.get("sha256") == sb["sha256"],
                 "active session is not the newly admitted online replay")
    controlled = False
    if supervisor.get("returncode") != 0:
        stop = audit.get("controlled_stop", {})
        controlled = (supervisor.get("returncode") == 1
            and str(supervisor.get("reason", "")).startswith("KeyboardInterrupt:")
            and stop.get("signal") == "SIGINT" and stop.get("reason") == "history-dependent online observer"
            and stop.get("local_evidence_sha256") == lb["sha256"] and stop.get("session_id") == session_id)
        _require(controlled, "nonzero exit was not the audited controlled measurement stop")
    else:
        _require(supervisor.get("status") in ("passed", "completed", "stopped"),
                 "failed supervisor is not reusable")
    for key, binding in (("local_evidence", lb), ("source_manifest", mb)):
        if key in audit:
            _require(all(audit[key].get(k) == binding[k] for k in ("path", "sha256")),
                     "audit " + key + " binding differs")
    _require(manifest.get("phase") == "wan_compile" and manifest.get("attempt") == attempt.name
             and manifest.get("execution_root") == str(snapshot)
             and manifest.get("config_sha256") == config_sha
             and manifest.get("source_sha256") == supervisor.get("source_sha256"),
             "source manifest identity differs")
    validation = _local(local, config)
    unchanged, changed, current_sha = _sources(manifest, snapshot)
    return dict(schema_version="ditserve.optimizations.local_replay.v1", path=str(path), sha256=sb["sha256"],
                phase="wan_compile", source_attempt=str(attempt), source_run_root=str(run),
                config_sha256=config_sha, session_id=session_id, settled_session_sha256=digest(charge),
                bindings=dict(supervisor=sb, local=lb, manifest=mb, audit=ab, ledger=db),
                controlled_measurement_stop=controlled, local_validation=validation,
                mathematical_source_files=unchanged, observation_only_changed_files=changed,
                current_source_sha256=current_sha)


def _stable_binding(value):
    value = copy.deepcopy(value)
    # Admission changes the ledger file, but never the old settled session.
    value.get("bindings", {}).pop("ledger", None)
    return value


def load_local_evidence(context, supervisor_path):
    """Copy qualified metadata to a new attempt, retaining original artifact paths."""
    expected = getattr(context, "replay_binding", None)
    _require(isinstance(expected, dict), "charged replay binding missing")
    actual = inspect_local_replay(supervisor_path, context.config, context.run_root,
                                  allow_active_output=context.output)
    _require(_stable_binding(actual) == _stable_binding(expected), "replay binding changed after admission")
    result, local_binding = _read(actual["bindings"]["local"]["path"])
    _require(local_binding == actual["bindings"]["local"], "local evidence changed while loading")
    destination = Path(context.output) / "wan-compile.json"
    _require(not destination.exists(), "new attempt already has local evidence")
    old_metadata = result.get("metadata", {})
    context.metadata.pop("wan_warmed", None)
    context.metadata.pop("wan_warm_evidence", None)
    result["metadata"] = copy.deepcopy(context.metadata)
    for key in ("online_path", "online_decision", "online_summary", "online", "decision"):
        result.pop(key, None)
    result.update(online_qualified=False, online_evaluation_complete=False,
                  scientific_qualified=False, system_qualified=False,
                  reason="complete local evidence reused; fresh capacity, warmup and online evaluation required",
                  local_evidence_reuse=dict(binding=copy.deepcopy(expected), remeasured=False,
                      source_metadata=old_metadata, original_local_path=local_binding["path"],
                      original_local_sha256=local_binding["sha256"],
                      scope="metadata-validated 8 recovery and 16 performance rows; arrays/LPIPS not rerun; no inherited online qualification"))
    atomic_json(destination, result)
    return result
