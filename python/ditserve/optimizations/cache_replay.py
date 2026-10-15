"""Read-only admission of completed cache evidence into a fresh online run.

Only JSON/source bytes are read. No model, array, executable, warm state, charge,
or selected serving default is inherited from the archived experiment.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

from ..research.common import atomic_json, digest, file_digest
from .protocol import ROOT


_SHAPE = [1, 81, 480, 832, 3]
_REQUIRED = {"python/ditserve/" + name for name in (
    "optimizations/cache_executor.py", "optimizations/wan.py", "optimizations/wan_phases.py",
    "optimizations/sana.py", "optimizations/sana_phases.py", "optimizations/stage_phases.py",
    "directions/block_executor.py", "directions/sana.py", "directions/direction1.py",
    "directions/direction2.py", "research/cfg_split_prefix.py", "research/cfg_adapter.py",
    "research/cfg_empty_context.py", "research/cfg_policy.py", "research/prompt_bank.py",
    "research/dual_state.py", "research/state.py", "research/storage.py", "research/common.py",
    "gpu/worker.py", "gpu/profiling.py", "gpu/phase_conflict.py", "gpu/quality.py", "util.py",
)} | {"configs/research_cfg.json", "configs/serving_optimizations.json"}


def _require(condition, reason):
    if not condition:
        raise ValueError("cache replay refused: " + reason)


def _within(path, root):
    return path == root or root in path.parents


def _read(path):
    path = Path(path)
    _require(not path.is_symlink(), "symlink evidence path")
    path = path.resolve(strict=True)
    _require(path.suffix == ".json" and path.stat().st_size <= 16 * 1024 * 1024,
             "evidence is not a bounded JSON file")
    raw = path.read_bytes()
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    data = json.loads(raw, parse_constant=invalid)
    _require(isinstance(data, dict), "evidence must be an object")
    return data, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def _number(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _equal_number(left, right):
    return _number(left) and _number(right) and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _artifact(value):
    _require(isinstance(value, dict) and isinstance(value.get("path"), str)
             and Path(value["path"]).is_absolute() and value.get("shape") == _SHAPE
             and value.get("raw_available") is True
             and all(isinstance(value.get(key), str) and len(value[key]) == 64
                     and all(char in "0123456789abcdef" for char in value[key])
                     for key in ("sha256", "array_sha256")), "incomplete artifact metadata")


def _numerical(value, quality):
    _require(value.get("passed") is True and value.get("finite") is True
             and value.get("shape") == _SHAPE and value.get("rtol") == quality["rtol"]
             and value.get("atol") == quality["atol"] and _number(value.get("max_abs")),
             "strict numerical comparison failed or incomplete")


def _quality(value, threshold):
    metrics = value.get("metrics", {})
    frames = metrics.get("lpips_per_frame", [])
    _require(value.get("passed") is True and metrics.get("status") == "ok"
             and metrics.get("finite") is True and metrics.get("range_valid") is True
             and metrics.get("lpips_status") == "ok" and metrics.get("shape") == _SHAPE
             and metrics.get("frame_count") == 81 and len(frames) == 81
             and all(_number(v) for v in frames) and _number(metrics.get("lpips_mean"))
             and metrics["lpips_mean"] <= threshold
             and _equal_number(statistics.mean(frames), metrics["lpips_mean"]),
             "quality failed, incomplete, or inconsistent with 81 frame values")
    return metrics["lpips_mean"]


def _paths(value):
    sequence = value.get("path_sequence", [])
    _require(len(sequence) == 32 and value.get("required_steps") == 32
             and value.get("observed_steps") == 32 and value.get("valid") is True,
             "32 committed cache paths missing")
    counts = dict(full=0, reuse=0, mixed=0, unknown=0)
    for index, row in enumerate(sequence):
        _require(row.get("step_index") == index and row.get("completed_steps") == index+1,
                 "cache path step order differs")
        outcome = row.get("outcome")
        _require((outcome == "full" and row.get("path") == "compute" and row.get("reused_forwards") == 0)
                 or (outcome == "reuse" and row.get("path") == "reuse" and row.get("reused_forwards") == 2),
                 "mixed/unknown/inconsistent paired cache path")
        counts[outcome] += 1
    _require(value.get("counts") == counts and value.get("full_reuse_observed") ==
             (counts["full"] > 0 and counts["reuse"] > 0), "cache path counts differ")
    return sequence, counts


def _generation(row, *, cache, native=False):
    service = row.get("service", {})
    _require(service.get("generation_completed") is True and service.get("completed_steps") == 32
             and _number(service.get("ms"), True) and service.get("cache") is cache
             and service.get("native_cache") is native
             and service.get("backend") == ("paired_008_original" if native else "eager"),
             "incomplete or different-backend generation")
    _artifact(row.get("artifact"))
    _require(row["artifact"]["array_sha256"] == service.get("output_sha256"),
             "generation output identity differs")
    if cache:
        sequence, _ = _paths(service.get("cache_paths", {}))
        _require(service.get("reuse_steps") == [r["completed_steps"] for r in sequence if r["outcome"] == "reuse"],
                 "reuse-step evidence differs")
    return service


def _catalog(snapshot):
    catalog, _ = _read(snapshot / "configs/research_cfg.json")
    result = {}
    for split, seeds in (("dev", (101, 102)), ("holdout", (201, 202))):
        result[split] = [dict(prompt_id=p["id"], prompt=p["text"], seed=seed, split=split,
                              input_id="%s-%s" % (p["id"], seed))
                         for p in catalog["prompts"] if p["split"] == split for seed in seeds]
    _require(len(result["dev"]) == 8 and len(result["holdout"]) == 4
             and len({item["input_id"] for group in result.values() for item in group}) == 12,
             "frozen 8-development/4-heldout catalog differs")
    bank = {item["input_id"]: item for item in result["dev"]}
    _require({"origami-101", "kettle-101", "kettle-102"} <= bank.keys(), "sensitive inputs missing")
    result["screen"] = [bank[key] for key in ("origami-101", "kettle-101", "kettle-102")]
    return result


def _local(local, config, catalog):
    q = config["quality"]
    _require(q == dict(image_lpips=.01, video_lpips=.05, rtol=1e-5, atol=1e-6), "frozen quality policy differs")
    _require(local.get("schema_version") == "ditserve.optimizations.wan.v1"
             and local.get("phase") == "cache-k4" and local.get("status") == "passed"
             and local.get("active_unit") is None and local.get("backend") == "eager"
             and local.get("compiled") is False and local.get("selected_backend") == "eager-paired-k4"
             and all(local.get(key) is True for key in ("local_qualified", "screen_local_qualified",
                       "cache_quality_qualified", "full_qualification_completed")),
             "local cache qualification incomplete or failed")
    _require(local.get("quality_policy") == dict(video_lpips=.05, same_backend_resume_rtol=1e-5,
                 same_backend_resume_atol=1e-6) and local.get("inputs") == catalog["screen"]
             and local.get("eager_baseline", {}).get("original_advance") is True,
             "input/numerical/native-eager identity differs")
    rows = local.get("rows", [])
    _require(len(rows) == 84, "required 84 original rows missing or duplicated")
    offset = 0
    def take(**identity):
        nonlocal offset
        _require(offset < len(rows), "local matrix truncated")
        row = rows[offset]
        offset += 1
        _require(all(row.get(key) == value for key, value in identity.items()), "frozen row order/identity differs")
        return row
    for item in catalog["screen"]:
        name = item["input_id"]
        original = []
        for native in (True, False):
            row = take(kind="original_split_equivalence", input_id=name, native=native)
            original.append(_generation(row, cache=True, native=native)["cache_paths"]["path_sequence"])
        comparison = take(kind="original_split_comparison", input_id=name)
        _numerical(comparison.get("numerical", {}), q)
        _require(all(comparison.get(key) is True for key in
                     ("passed", "cache_paths_equal", "path_records_valid", "full_paths_equal"))
                 and original[0] == original[1], "original paired/K4 parity failed")
        result = take(kind="same_backend_resume", input_id=name).get("result", {})
        steps = result.get("steps", [])
        _require(result.get("passed") is True and len(steps) == 32
                 and [s.get("step") for s in steps] == list(range(1, 33))
                 and all(s.get(key) is True for s in steps for key in
                         ("latent", "solver", "cache_state", "rng", "passed", "cache_paths_equal")),
                 "recovery state check failed or truncated")
        paths = result.get("cache_paths", {})
        left, _ = _paths(paths.get("continuous", {}))
        right, _ = _paths(paths.get("k4", {}))
        _require(paths.get("valid") is True and paths.get("paths_equal") is True and left == right
                 and all(s.get("continuous_path") == a and s.get("k4_path") == b
                         for s, a, b in zip(steps, left, right)), "recovery path evidence differs")
        interrupts = result.get("interruptions", [])
        _require(len(interrupts) == 3 and [r.get("step") for r in interrupts] == [0, 8, 16]
                 and all(r.get("image_service", {}).get("finite") is True
                         and _number(r["image_service"].get("ms"), True)
                         and r.get("resolved_step_path") == right[r["step"]]["outcome"]
                         and r.get("phase") == "prefix" and r.get("branch") == "cond"
                         and r.get("next_block") == 4 for r in interrupts), "real fixed Sana interruptions missing")
        _numerical(result.get("final", {}), q)
        _require(len(result.get("artifacts", [])) == 2, "recovery output pair missing")
        for artifact in result["artifacts"]:
            _artifact(artifact)
    # This pure observer is itself source-pinned; no tensor/GPU import occurs.
    from .wan_phases import _cache_coverage
    coverage = _cache_coverage(rows[:offset], [item["input_id"] for item in catalog["screen"]])
    _require(coverage["passed"] is True and coverage == local.get("cache_path_coverage"),
             "full/reuse recovery aggregate differs or is incomplete")
    screen_times = []
    for order, arms in enumerate((("off", "paired"), ("paired", "off"))):
        for item in catalog["screen"]:
            for arm in arms:
                row = take(kind="cache_screen", order=order, input_id=item["input_id"], arm=arm)
                _require(row.get("complete") is True, "screen generation not evaluated")
                service = _generation(row, cache=arm == "paired")
                _quality(row.get("quality", {}), q["video_lpips"])
                screen_times.append(dict(order=order, arm=arm, ms=service["ms"]))
    summary = local.get("screen", {})
    _require(all(summary.get(k) == v for k, v in dict(complete=True, distinct_development_inputs=3,
                 heldout_inputs=0, quality_passed=True, quality_checked=12, n_per_arm=6,
                 performance_passed=True).items()), "sensitive quality/speed screen incomplete")
    def gains(times, recorded):
        values = [1 - sum(r["ms"] for r in times if r["order"] == order and r["arm"] == "paired") /
                      sum(r["ms"] for r in times if r["order"] == order and r["arm"] == "off")
                  for order in (0, 1)]
        _require(len(recorded) == 2 and all(_equal_number(a, b) and a >= config["metrics"]["minimum_local_gain"]
                     for a, b in zip(values, recorded)), "both-order speed gate unsupported")
        return values
    screen_gains = gains(screen_times, summary.get("gains", []))
    times, quality_rows = [], []
    for split, order in (("dev", 0), ("dev", 1), ("holdout", 0)):
        for item in catalog[split]:
            services = {}
            for arm in (("off", "paired") if order == 0 else ("paired", "off")):
                row = take(kind="cache_qualification", split=split, order=order, input_id=item["input_id"], arm=arm)
                _require(row.get("complete") is True, "qualification generation unevaluated")
                services[arm] = _generation(row, cache=arm == "paired")
                if split == "dev":
                    times.append(dict(order=order, arm=arm, ms=services[arm]["ms"]))
            pair = take(kind="cache_qualification_pair", split=split, order=order, input_id=item["input_id"])
            _require(pair.get("complete") is True and all(_equal_number(pair.get(arm+"_ms"), services[arm]["ms"])
                     for arm in ("off", "paired")), "qualification pair is incomplete/unbound")
            quality_rows.append(dict(split=split, lpips_mean=_quality(pair.get("quality", {}), q["video_lpips"])))
    full = local.get("full_qualification", {})
    _require(offset == 84 and all(full.get(k) == v for k, v in dict(status="passed", complete=True,
                 qualified=True, quality_passed=True, performance_passed=True, pairs=20, required_pairs=20,
                 dev_pairs=16, heldout_pairs=4, remaining_pairs=0, remaining_units=[], heldout_selection=False).items()),
             "complete frozen 20-pair qualification missing")
    full_gains = gains(times, full.get("gains", []))
    return dict(original_rows=84, new_local_executions=0, recovery_inputs=3, recovery_steps=96,
        real_sana_interruptions=9, screen_quality_checked=12, qualification_pairs=20, dev_pairs=16,
        heldout_pairs=4, screen_gains=screen_gains, qualification_gains=full_gains,
        max_mean_lpips={split: max(r["lpips_mean"] for r in quality_rows if r["split"] == split)
                        for split in ("dev", "holdout")}, cache_path_coverage=coverage,
        arrays_read=False, quality_remeasured=False)


def inspect_cache_replay(supervisor_path, config, source_run_root):
    """Reject incomplete local work; bind only the archived attempt00 evidence."""
    run = Path(source_run_root).resolve(strict=True)
    path = Path(supervisor_path)
    _require(not path.is_symlink(), "symlink supervisor")
    path = path.resolve(strict=True)
    attempt = run / "cache_k4/attempt00"
    _require(path == attempt / "supervisor.json", "source must be archived cache_k4/attempt00")
    supervisor, sb = _read(path)
    local, lb = _read(attempt / "cache-k4.json")
    old_config, cb = _read(attempt / "config.json")
    protocol, pb = _read(run / "protocol.json")
    snapshot = run / "sources/cache_k4-attempt00"
    manifest, mb = _read(snapshot / "manifest.json")
    ledger, db = _read(run.parent / "gpu-budget.json")
    config_sha = digest(config)
    _require(old_config == config and protocol.get("config") == config and protocol.get("sha256") == config_sha,
             "scientific config differs")
    _require(supervisor.get("status") == "passed" and supervisor.get("returncode") == 0
             and supervisor.get("phase") == "cache_k4" and supervisor.get("output") == str(attempt)
             and supervisor.get("config_sha256") == config_sha and not supervisor.get("supervisor_failure")
             and all(supervisor.get(key) is True for key in
                     ("local_qualified", "cache_quality_qualified", "full_qualification_completed")),
             "failed/incomplete supervisor cannot authorize replay")
    _require(all(supervisor.get(key) == value for key, value in local.items() if key != "phase"),
             "supervisor and original local evidence disagree")
    occupancy = supervisor.get("final_occupancy", {})
    _require(occupancy.get("status") == "exclusive_observed" and occupancy.get("owned") == []
             and occupancy.get("foreign") == [], "exclusive cleanup is unproven")
    charge = supervisor.get("charge", {})
    session = charge.get("session_id")
    settled = [row for row in ledger.get("sessions", []) if row.get("session_id") == session]
    _require(isinstance(session, str) and session and ledger.get("active") is None
             and len(settled) == 1 and settled[0] == charge and charge.get("status") == "completed"
             and charge.get("stage") == "cache_k4" and charge.get("gpu_count") == 1
             and _number(charge.get("gpu_seconds"), True) and _number(charge.get("elapsed_seconds"), True)
             and _number(charge.get("ended_at"), True)
             and _equal_number(charge["gpu_seconds"], charge["elapsed_seconds"]), "source charge not identically settled")
    identity = charge.get("identity", {})
    _require(identity.get("phase") == "cache_k4" and identity.get("output") == str(attempt)
             and identity.get("run_root") == str(run) and identity.get("config_sha256") == config_sha
             and identity.get("source_sha256") == supervisor.get("source_sha256")
             and identity.get("gpu_uuids") == [config["gpu_uuid"]]
             and identity.get("auxiliary_encoder") is False, "source charge identity differs")
    _require(manifest.get("phase") == "cache_k4" and manifest.get("attempt") == "attempt00"
             and manifest.get("execution_root") == str(snapshot) and manifest.get("config_sha256") == config_sha
             and manifest.get("source_sha256") == supervisor.get("source_sha256"), "frozen source identity differs")
    old = manifest.get("source_files", {})
    _require(isinstance(old, dict) and _REQUIRED <= old.keys() and digest(old) == manifest["source_sha256"],
             "mathematical source manifest missing/inconsistent")
    for name, expected in old.items():
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts, "unsafe snapshot source path")
        source = snapshot / relative
        _require(_within(source.resolve(), snapshot.resolve()) and source.is_file()
                 and not source.is_symlink() and file_digest(source) == expected, "frozen source changed: " + name)
    math_hashes = {}
    for name in sorted(_REQUIRED):
        source = ROOT / name
        _require(source.is_file() and not source.is_symlink() and file_digest(source) == old[name],
                 "mathematical source changed: " + name)
        math_hashes[name] = old[name]
    validation = _local(local, config, _catalog(snapshot))
    return dict(schema_version="ditserve.optimizations.cache_replay_binding.v1", path=str(path), sha256=sb["sha256"],
        source_run_root=str(run), source_attempt=str(attempt), config_sha256=config_sha, session_id=session,
        settled_session_sha256=digest(charge), source_sha256=manifest["source_sha256"],
        mathematical_source_files=math_hashes, mathematical_source_sha256=digest(math_hashes),
        bindings=dict(supervisor=sb, local=lb, config=cb, protocol=pb, manifest=mb, ledger=db),
        local_validation=validation, scope="completed local evidence only; fresh online execution and budget required")


def load_cache_evidence(context, supervisor_path):
    """Write a new reference record, never pretend the 84 local rows ran again."""
    expected = getattr(context, "cache_replay_binding", None)
    _require(isinstance(expected, dict), "admitted cache replay binding missing")
    actual = inspect_cache_replay(supervisor_path, context.config, expected["source_run_root"])
    _require(actual == expected, "cache replay binding changed after admission")
    output = Path(context.output).resolve()
    _require(_within(output, Path(context.run_root).resolve())
             and not _within(output, Path(actual["source_run_root"])), "new output overlaps archived source run")
    destination = output / "cache-replay.json"
    _require(not destination.exists(), "new output already contains cache replay evidence")
    local, binding = _read(actual["bindings"]["local"]["path"])
    _require(binding == actual["bindings"]["local"], "local source changed while loading")
    result = dict(schema_version="ditserve.optimizations.cache_replay.v1", phase="cache_online", status="passed",
        local_qualified=True, cache_quality_qualified=True, full_qualification_completed=True,
        online_qualified=False, online_evaluation_complete=False, scientific_qualified=False, system_qualified=False,
        decision="未验证", rows=[], active_unit=None, new_local_executions=0, local_evidence_remeasured=False,
        inputs=copy.deepcopy(local["inputs"]), quality_policy=copy.deepcopy(local["quality_policy"]),
        screen=copy.deepcopy(local["screen"]), full_qualification=copy.deepcopy(local["full_qualification"]),
        cache_path_coverage=copy.deepcopy(local["cache_path_coverage"]),
        local_evidence_reuse=dict(binding=copy.deepcopy(actual), original_rows=84, new_rows=0,
            source_metadata=copy.deepcopy(local.get("metadata", {})),
            scope="source-bound reference only; no inherited charge, warm state or serving default"),
        reason="complete archived cache local qualification reused; four-arm online serving remains unverified")
    atomic_json(destination, result)
    return result
