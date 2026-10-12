#!/usr/bin/env python3
"""Read phase-conflict evidence and render reports without importing GPU libraries."""

from ..gpu.media import video_frames
from ..util import ROOT, atomic_json, canonical_gpu_uuid as canonical_uuid, archive_run

import argparse
from collections import Counter
import csv
import datetime
import hashlib
import html
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics


SCHEMA = "ditserve.phase-conflict-summary.v1"
PROFILE_SCHEMA = "ditserve.phase-conflict.v1"
STRATEGIES = ("immediate", "deferred", "blocking", "overlap")
LABELS = {"immediate": "先解码 B", "deferred": "A 完成后解码 B",
          "blocking": "步边界同步插入 B", "overlap": "步边界异步插入 B"}
REQUIRED_CONFIG = ("height", "width", "frames", "steps", "seed", "cache", "dtype", "vae_dtype")
NON_IDENTITY = {"id", "mode", "strategy", "insert_after", "b_latent_path", "b_metadata_path",
                "output", "output_dir", "artifact_dir", "local_files_only", "warmup", "repeats",
                "repeat", "diagnostic", "b_frames"}
METRICS = ("remaining_ms", "a_completed_ms", "b_completed_ms")
MEMORY = ("peak_allocated_bytes", "peak_reserved_bytes", "initial_allocated_bytes",
          "before_a_vae_peak_allocated_bytes", "before_a_vae_peak_reserved_bytes")


def number(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0))


def _reject_constant(value):
    raise ValueError("non-finite JSON number: " + value)


def read_json(path, issues):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        issues.append("%s: %s" % (path, exc))
        return None


def resolve_path(value, root, owner=None):
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = [root / path]
    if owner is not None:
        candidates.append(owner / path)
    candidates.append(ROOT / path)
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def output_valid(output):
    if not isinstance(output, dict) or output.get("finite") is not True:
        return False
    digest, shape = output.get("sha256"), output.get("shape")
    return (isinstance(digest, str) and len(digest) == 64 and
            all(char in "0123456789abcdefABCDEF" for char in digest) and
            isinstance(shape, list) and bool(shape) and
            all(type(dim) is int and dim > 0 for dim in shape) and
            isinstance(output.get("dtype"), str) and bool(output["dtype"]))


def same_output(left, right):
    return (output_valid(left) and output_valid(right) and
            all(left.get(key) == right.get(key) for key in ("sha256", "shape", "dtype")))


def video_shape_matches(output, config):
    return (isinstance(output, dict) and isinstance(config, dict) and
            output.get("shape") == [1, config.get("frames"), config.get("height"), config.get("width"), 3])


def generation_identity(profile):
    config = profile.get("config")
    if not isinstance(config, dict) or any(key not in config for key in REQUIRED_CONFIG):
        return None
    if any(type(config[key]) is not int or config[key] <= 0 for key in ("height", "width", "frames", "steps")):
        return None
    if type(config["seed"]) is not int or not isinstance(config["cache"], str):
        return None
    identity = {key: value for key, value in config.items() if key not in NON_IDENTITY and not key.startswith("b_")}
    # Preserve all generation parameters, including cache policy and input prompt.
    # Only execution order, file locations and run bookkeeping are excluded.
    for key in ("model_id", "revision"):
        observed = profile.get(key)
        if key == "revision" and isinstance(observed, dict):
            observed = observed.get("requested")
        value = config.get(key, observed)
        if not isinstance(value, str) or not value:
            return None
        if observed is not None and observed != value:
            return None
        identity[key] = value
    return identity


def environment_issues(profile):
    issues = []
    if profile.get("kind") != "real_gpu_phase_conflict" or profile.get("measured") is not True:
        issues.append("not a measured real_gpu_phase_conflict record")
    config = profile.get("config", {})
    env = profile.get("environment") or {}
    target = canonical_uuid(config.get("gpu_uuid"))
    if not (isinstance(env, dict) and env.get("ready") is True and env.get("device") == "cuda:0" and
            env.get("cuda_device_order") == "PCI_BUS_ID" and target and
            canonical_uuid(env.get("cuda_visible_devices")) == target and
            canonical_uuid((env.get("gpu") or {}).get("uuid")) == target):
        issues.append("actual GPU environment/UUID does not match normalized target")
    return issues


def _key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _metric(values):
    values = [value for value in values if number(value)]
    return {"median": statistics.median(values) if values else None, "n": len(values),
            "min": min(values) if values else None, "max": max(values) if values else None}


def _events(raw, issues):
    raw = raw if isinstance(raw, dict) else {}
    forwards, steps = [], []
    for item in raw.get("dit_forwards", []) if isinstance(raw.get("dit_forwards"), list) else []:
        if (isinstance(item, dict) and number(item.get("cuda_start_ms")) and
                number(item.get("cuda_end_ms")) and item["cuda_end_ms"] >= item["cuda_start_ms"]):
            forwards.append(dict(item, duration_ms=item["cuda_end_ms"] - item["cuda_start_ms"]))
        else:
            issues.append("invalid DiT CUDA event; interval omitted from timeline")
    for item in raw.get("steps", []) if isinstance(raw.get("steps"), list) else []:
        if isinstance(item, dict):
            steps.append(item)
    decode = raw.get("b_decode") if isinstance(raw.get("b_decode"), dict) else {}
    interval = None
    if (number(decode.get("cuda_start_ms")) and number(decode.get("cuda_finished_ms")) and
            decode["cuda_finished_ms"] >= decode["cuda_start_ms"]):
        interval = [decode["cuda_start_ms"], decode["cuda_finished_ms"]]
    intersections = []
    if interval:
        for item in forwards:
            overlap = max(0.0, min(interval[1], item["cuda_end_ms"]) - max(interval[0], item["cuda_start_ms"]))
            if overlap:
                intersections.append({"forward_index": item.get("forward_index"), "interval_intersection_ms": overlap})
    step_rows = []
    previous = 0
    ordered = all(item.get("forward_index") == index for index, item in enumerate(forwards))
    for index, step in enumerate(steps):
        count = step.get("forward_count")
        mapped = (ordered and step.get("step_index") == index and step.get("completed_step") == index + 1 and
                  type(count) is int and previous < count <= len(forwards))
        selected = forwards[previous:count] if mapped else []
        step_rows.append(dict(step, mapping_status="mapped" if mapped else "unknown",
                              first_forward_index=previous if mapped else None,
                              last_forward_index=count - 1 if mapped else None,
                              transformer_event_interval_sum_ms=sum(item["duration_ms"] for item in selected) if mapped else None))
        if mapped:
            previous = count
        else:
            ordered = False
    return {"dit_forwards": forwards, "steps": steps, "step_rows": step_rows, "b_decode": decode,
            "residency_snapshots": raw.get("residency_snapshots", []),
            "before_a_vae": raw.get("before_a_vae", {}),
            "transformer_event_interval_sum_ms": sum(item["duration_ms"] for item in forwards) if forwards else None,
            "a_stream": raw.get("a_stream"), "b_stream": raw.get("b_stream"),
            "forward_count_reported": raw.get("forward_count"), "step_count_reported": raw.get("step_count"),
            "b_decode_cuda_interval_ms": interval, "intersections": intersections,
            "kernel_concurrency_verified": False,
            "clock_note": "CUDA events use the case-local common origin; host monotonic timestamps are a separate clock."}


def load_case(record, index, root):
    issues = []
    if not isinstance(record, dict):
        record = {"status": "invalid_record", "original_record": record}
        issues.append("manifest record is not an object")
    job = record.get("job") if isinstance(record.get("job"), dict) else {}
    path = resolve_path(record.get("profile_path"), root)
    profile = read_json(path, issues) if path else None
    profile = profile or {}
    config = profile.get("config") if isinstance(profile.get("config"), dict) else {}
    strategy = config.get("strategy", job.get("strategy"))
    mode = config.get("mode", job.get("mode"))
    result = {"record_index": index, "id": record.get("id", "record-%d" % index),
              "job": job, "config": config, "strategy": strategy, "mode": mode,
              "manifest_status": record.get("status", "unknown"), "profile_status": profile.get("status", "missing"),
              "profile_path": str(path) if path else None,
              "elapsed_seconds": record.get("elapsed_seconds") if number(record.get("elapsed_seconds")) else None,
              "reason": record.get("reason"), "failure_category": record.get("failure_category"),
              "issues": issues, "failure": profile.get("failure", record.get("failure")),
              "output_a": profile.get("output_a"), "output_b": profile.get("output_b"),
              "generation_identity": generation_identity(profile), "valid_execution": False,
              "valid_pair": False, "selected": False, "selection_reason": "not evaluated",
              "a_validation": {"status": "unknown", "reference_ids": []},
              "b_validation": {"status": "unknown"}, "profile": profile}
    if not path and record.get("status") == "completed":
        issues.append("completed record has no profile_path")
    if profile:
        if profile.get("schema_version") != PROFILE_SCHEMA:
            issues.append("unsupported worker schema")
        if profile.get("job") != job:
            issues.append("manifest job differs from profile job")
        if any(config.get(key) != value for key, value in job.items() if key != "output_dir"):
            issues.append("normalized profile config differs from requested job")
        if config.get("id") is not None and config["id"] != result["id"]:
            issues.append("manifest id differs from profile config id")
    timings = profile.get("timings") if isinstance(profile.get("timings"), dict) else {}
    memory = profile.get("memory") if isinstance(profile.get("memory"), dict) else {}
    result["timings"] = {key: timings.get(key) if number(timings.get(key)) else None for key in METRICS}
    for key in ("cpu_prompt_encode_ms", "prompt_embeddings_h2d_ms", "started_at_unix_ns", "finished_at_unix_ns"):
        result["timings"][key] = timings.get(key) if number(timings.get(key)) else None
    result["memory"] = {key: memory.get(key) if number(memory.get(key)) else None for key in MEMORY}
    observed = memory.get("observed_at_failure")
    result["failure_memory_observation"] = None
    if isinstance(observed, dict):
        measured = isinstance(result["failure"], dict) and result["failure"].get("measured_at_failure") is True
        result["failure_memory_observation"] = {
            "scope": "measured_case" if measured and mode == "case" else
                     "measured_prepare" if measured and mode == "prepare" else "outside_measured_window",
            "is_completed_peak": False, "is_observed_lower_bound": True,
            "measured_at_failure": measured,
            "values": {key: observed.get(key) if number(observed.get(key)) else None for key in
                       ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")}}
    result["events"] = _events(profile.get("events"), issues)
    decode_start = result["events"]["b_decode"].get("cuda_start_ms")
    result["timings"]["b_wait_to_gpu_decode_start_ms"] = decode_start if mode == "case" and number(decode_start) else None
    completed = record.get("status") == "completed" and profile.get("status") == "completed"
    if completed:
        issues.extend(environment_issues(profile))
    result["phase_profiles"] = profile.get("phase_profiles", {})
    if mode == "prepare":
        if completed and not video_shape_matches(result["output_b"], config):
            issues.append("prepared B video shape differs from prepare request specification")
        result["valid_execution"] = completed and not issues and output_valid(result["output_b"])
        return result
    if completed:
        if strategy not in STRATEGIES:
            issues.append("unknown execution strategy")
        if result["generation_identity"] is None:
            issues.append("generation identity missing or invalid")
        for name in ("output_a", "output_b"):
            if not output_valid(result[name]):
                issues.append(name + " missing, non-finite or invalid fingerprint")
        if not video_shape_matches(result["output_a"], config):
            issues.append("A video shape differs from A request specification")
        if any(result["timings"][key] is None for key in METRICS):
            issues.append("remaining-stage completion timing missing or invalid")
        elif max(result["timings"][key] for key in ("a_completed_ms", "b_completed_ms")) > result["timings"]["remaining_ms"] + 1.0:
            issues.append("request completion exceeds remaining-stage measurement window")
        if result["memory"]["peak_allocated_bytes"] is None or result["memory"]["peak_reserved_bytes"] is None:
            issues.append("allocator peaks missing or invalid")
    result["valid_execution"] = completed and not issues
    return result


def select_attempts(cases):
    """Keep retries visible; select the latest valid result for an unchanged job.

    Changing a generation/execution field while reusing an id is not a retry.
    Output directories may differ because the runner preserves each attempt.
    """
    by_id = {}
    paths = Counter(case["profile_path"] for case in cases if case["profile_path"])
    for case in cases:
        by_id.setdefault(_key(case["id"]), []).append(case)
        if case["profile_path"] and paths[case["profile_path"]] > 1:
            case["issues"].append("profile path reused across manifest attempts")
            case["valid_execution"] = False
    for attempts in by_id.values():
        jobs = {_key({key: value for key, value in case["job"].items() if key != "output_dir"})
                for case in attempts}
        if len(jobs) != 1:
            for case in attempts:
                case["issues"].append("job identity changes across attempts with the same id")
                case["selection_reason"] = "inconsistent retry identity"
            continue
        valid = [case for case in attempts if case["valid_execution"]]
        selected = valid[-1] if valid else None
        for case in attempts:
            case["selected"] = case is selected
            case["selection_reason"] = ("latest valid execution of unchanged job" if case is selected else
                                        "superseded or unsuccessful attempt; retained for audit")


def validate_references(cases, root):
    for case in cases:
        if case["mode"] != "case" or not case["selected"]:
            continue
        owner = Path(case["profile_path"]).parent
        path = resolve_path(case["config"].get("b_metadata_path", case["job"].get("b_metadata_path")), root, owner)
        reference_issues = []
        meta = read_json(path, reference_issues) if path else None
        if not path:
            reference_issues.append("B metadata path missing")
        identity = generation_identity(meta or {})
        if meta and identity is None:
            reference_issues.append("B metadata generation identity missing or invalid")
        if identity and case["generation_identity"]:
            for name, output in (("decoded B", case["output_b"]), ("reference B", meta.get("output_b"))):
                if not video_shape_matches(output, identity):
                    reference_issues.append(name + " video shape differs from B request specification")
            for field in ("model_id", "revision", "dtype", "vae_dtype", "text_encoder_dtype"):
                if identity.get(field) != case["generation_identity"].get(field):
                    reference_issues.append("B metadata model/precision differs: " + field)
            latent_path = resolve_path(case["config"].get("b_latent_path"), root, owner)
            reference_latent = resolve_path(meta.get("b_latent_path"), root, path.parent)
            if latent_path is None or reference_latent is None or latent_path != reference_latent:
                reference_issues.append("B metadata latent path differs from requested B")
            for field in ("height", "width", "frames", "steps", "seed", "cache", "attention", "text_encoder_device"):
                wanted = case["config"].get("b_" + field, case["config"].get(field))
                if identity.get(field) != wanted:
                    reference_issues.append("B metadata identity differs from requested B: " + field)
            source = case["profile"].get("b_source")
            if not isinstance(source, dict):
                reference_issues.append("loaded B provenance evidence missing")
            else:
                if source.get("config") != meta.get("config"):
                    reference_issues.append("loaded B config differs from reference metadata")
                if not same_output(source.get("latent"), meta.get("latent")):
                    reference_issues.append("loaded B latent fingerprint differs from reference metadata")
                if not same_output(source.get("output_b"), meta.get("output_b")):
                    reference_issues.append("loaded B reference output differs from metadata")
        match = bool(meta and same_output(case["output_b"], meta.get("output_b")))
        if meta and not match:
            reference_issues.append("B output differs from metadata output_b or reference is invalid")
        case["b_validation"] = {"status": "matched" if match and not reference_issues else "invalid",
                                "metadata_path": str(path) if path else None,
                                "generation_identity": identity, "reference_output": meta.get("output_b") if meta else None,
                                "issues": reference_issues}
    for case in cases:
        if case["mode"] != "case" or not case["selected"]:
            continue
        # Immediate/deferred are no-insertion references for A. A reference is
        # never chosen from blocking/overlap, nor from a differently seeded arm.
        references = [other for other in cases if other is not case and other["selected"] and
                      other["mode"] == "case" and
                      other["strategy"] in ("immediate", "deferred") and
                      other["generation_identity"] == case["generation_identity"]]
        if not references:
            case["a_validation"] = {"status": "unknown", "reference_ids": [],
                                    "issues": ["no independent immediate/deferred reference with identical A generation identity"]}
        else:
            matched = all(same_output(case["output_a"], other["output_a"]) for other in references)
            case["a_validation"] = {"status": "matched" if matched else "mismatch",
                                    "reference_ids": [other["id"] for other in references],
                                    "issues": [] if matched else ["A output or no-insertion references disagree"]}
        case["valid_pair"] = case["a_validation"]["status"] == "matched" and case["b_validation"]["status"] == "matched"


def _groups(cases, root):
    grouped = {}
    for case in cases:
        if case["mode"] != "case" or case["strategy"] not in STRATEGIES or case["generation_identity"] is None:
            continue
        metadata = resolve_path(case["config"].get("b_metadata_path"), root,
                                Path(case["profile_path"]).parent if case["profile_path"] else None)
        identity = {"a": case["generation_identity"],
                    "b_metadata_path": str(metadata) if metadata else None,
                    "diagnostic": case["config"].get("diagnostic", False),
                    "b_spec": {key: case["config"].get("b_" + key, case["config"].get(key))
                               for key in ("height", "width", "frames", "steps")}}
        grouped.setdefault(_key(identity), {"identity": identity, "cases": []})["cases"].append(case)
    groups = []
    for group in grouped.values():
        offsets = {case["config"].get("insert_after") for case in group["cases"]
                   if case["strategy"] in ("blocking", "overlap")}
        if not offsets:
            offsets = {case["config"].get("insert_after") for case in group["cases"]}
        for offset in sorted(offsets, key=lambda value: (value is None, str(value))):
            # No-insertion baselines can be reused across insertion offsets;
            # repetitions remain independent, retries are counted only once.
            members = [case for case in group["cases"] if
                       case["strategy"] in ("immediate", "deferred") or
                       case["config"].get("insert_after") == offset]
            rows = {}
            for strategy in STRATEGIES:
                selected = [case for case in members if case["strategy"] == strategy]
                valid = [case for case in selected if case["valid_pair"]]
                rows[strategy] = {"case_ids": [case["id"] for case in selected],
                                  "record_indices": [case["record_index"] for case in selected],
                                  "attempted": len(selected), "valid_pairs": len(valid),
                                  "metrics": {name: _metric([case["timings"][name] for case in valid]) for name in METRICS},
                                  "peak_allocated_bytes": _metric([case["memory"]["peak_allocated_bytes"] for case in valid])}
            complete = all(rows[strategy]["valid_pairs"] for strategy in STRATEGIES)
            comparisons = {}
            if complete:
                blocking = rows["blocking"]["metrics"]["remaining_ms"]["median"]
                overlap = rows["overlap"]["metrics"]["remaining_ms"]["median"]
                serial = min(rows[name]["metrics"]["remaining_ms"]["median"] for name in ("immediate", "deferred"))
                comparisons = {
                    "blocking_over_overlap_remaining_ratio": blocking / overlap if number(overlap, True) else None,
                    "overlap_over_blocking_remaining_ratio": overlap / blocking if number(blocking, True) else None,
                    "overlap_over_best_serial_remaining_ratio": overlap / serial if number(serial, True) else None,
                    "overlap_remaining_reduction_fraction": 1 - overlap / blocking if number(blocking, True) else None}
            groups.append({"identity": dict(group["identity"], insert_after=offset), "strategies": rows,
                           "complete_four_strategies": complete, "comparisons": comparisons,
                           "baseline_reuse_note": "Immediate/deferred have no insertion; matching baselines may be reused across offsets."})
    return groups


def conclusion_for(groups, cases):
    # A diagnostic smoke run cannot by itself establish the formal signal.
    formal = [group for group in groups if not group["identity"]["diagnostic"]]
    complete = [group for group in formal if group["complete_four_strategies"]]
    repeated = [group for group in complete if all(group["strategies"][name]["valid_pairs"] >= 2
                                                  for name in ("blocking", "overlap"))]
    time_signals = [group for group in repeated if
                    abs((group["comparisons"].get("overlap_over_blocking_remaining_ratio") or 1.0) - 1.0) >= 0.10]
    oom_signals = []
    for group in formal:
        if not all(group["strategies"][name]["valid_pairs"] for name in ("immediate", "deferred")):
            continue
        indices = {index for name in ("blocking", "overlap")
                   for index in group["strategies"][name]["record_indices"]}
        latest_indices = {case["id"]: case["record_index"] for case in cases}
        oom_ids = {case["id"] for case in cases if case["record_index"] in indices and
                   latest_indices[case["id"]] == case["record_index"] and
                   isinstance(case.get("failure"), dict) and case["failure"].get("measured_at_failure") is True and
                   not environment_issues(case["profile"]) and
                   (case["manifest_status"] == "oom" or "oom" in str(case["failure"].get("category", "")))}
        if len(oom_ids) >= 2:
            oom_signals.append(group)
    phenomenon = {
        "status": "observed" if time_signals or oom_signals else ("not_observed_in_sample" if repeated else "insufficient_evidence"),
        "label_zh": "阶段冲突/干扰现象成立（当前协议）" if time_signals or oom_signals else
                    ("当前重复样本未发现明确干扰" if repeated else "阶段干扰证据不足"),
        "repeated_time_signal_groups": len(time_signals), "repeated_oom_signal_groups": len(oom_signals),
        "note_zh": "重复插入 OOM 需有效串行参考；时间信号需同步/异步各至少两次且中位数相差至少 10%。不证明缓存因果、kernel 并行或在线收益。"}
    stable = []
    for group in complete:
        rows = group["strategies"]
        if not all(rows[name]["valid_pairs"] >= 2 for name in STRATEGIES):
            continue
        ordered = sorted(STRATEGIES, key=lambda name: rows[name]["metrics"]["remaining_ms"]["median"])
        best = rows[ordered[0]]["metrics"]["remaining_ms"]
        others = [rows[name]["metrics"]["remaining_ms"] for name in ordered[1:]]
        if best["max"] < min(item["min"] for item in others) and best["median"] <= 0.9 * min(item["median"] for item in others):
            identity = group["identity"]
            workload = {"a": {key: identity["a"][key] for key in ("height", "width", "frames", "steps")},
                        "b": identity["b_spec"], "insert_after": identity["insert_after"]}
            stable.append({"workload": workload, "strategy": ordered[0]})
    different_loads = len({_key(item["workload"]) for item in stable}) >= 2
    choices = {item["strategy"] for item in stable}
    if different_loads and len(choices) > 1:
        status, label, reason = ("continue_validation", "可继续验证在线调度选题",
            "不同规格/插入相位出现不同最快安排，四策略均有重复，最快安排的观测范围与其他策略分离且中位数优势至少 10%。这支持继续验证调度需求，尚非在线 goodput 结果或统计显著性结论。")
    elif different_loads and len(choices) == 1 and choices <= {"immediate", "deferred"}:
        status, label, reason = ("stop_complex_scheduling", "当前停止复杂调度方向",
            "已充分重复的不同规格/相位均由同一固定串行策略胜出；优先采用该简单策略。本结论只覆盖已测负载，不等于不存在阶段干扰。")
    elif phenomenon["status"] == "not_observed_in_sample":
        status, label, reason = ("not_observed_in_sample", "当前样本未发现明确继续信号",
            "已完成的重复四策略样本未显示明确干扰；不能据此断言其他负载不存在问题。")
    else:
        status, label, reason = ("insufficient_evidence", "在线调度研究证据不足",
            "阶段干扰现象与调度选题价值分开判断。继续需不同规格/相位有不同最优安排且差异超过重复波动；当前缺少该证据，单次串行基线和 smoke 不足以建立稳健最优策略。")
    return {"status": status, "label_zh": label, "reason_zh": reason,
            "phenomenon": phenomenon, "stable_best_arrangements": stable,
            "decision_rule_zh": "选题继续/停止规则采用四策略各至少两次、观测范围不重叠且最快中位数至少优 10%；仅是保守 pilot 规则，不是显著性检验。"}


def summarize(root):
    root = Path(root).resolve()
    issues = []
    manifest = read_json(root / "manifest.json", issues) or {}
    records = manifest.get("records", [])
    if not isinstance(records, list):
        issues.append("manifest.records is not a list")
        records = []
    cases = [load_case(record, index, root) for index, record in enumerate(records)]
    expected_gpu = canonical_uuid((manifest.get("protocol_config") or {}).get("gpu_uuid"))
    if expected_gpu:
        for case in cases:
            if case["valid_execution"] and canonical_uuid(case["config"].get("gpu_uuid")) != expected_gpu:
                case["issues"].append("normalized GPU target differs from suite protocol")
                case["valid_execution"] = False
    select_attempts(cases)
    validate_references(cases, root)
    groups = _groups(cases, root)
    complete = [group for group in groups if group["complete_four_strategies"]]
    conclusion = conclusion_for(groups, cases)
    summary = {"schema_version": SCHEMA, "root": str(root),
               "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "manifest": manifest, "issues": issues, "cases": cases, "groups": groups,
               "counts": {"manifest_records": len(cases), "valid_execution": sum(case["valid_execution"] for case in cases),
                          "selected_executions": sum(case["selected"] for case in cases),
                          "valid_pairs": sum(case["valid_pair"] for case in cases), "complete_four_strategy_groups": len(complete),
                          "manifest_statuses": dict(Counter(case["manifest_status"] for case in cases)),
                          "failure_categories": dict(Counter((case.get("failure") or {}).get("category", "unknown")
                                                             for case in cases if isinstance(case.get("failure"), dict)))},
               "conclusion": conclusion, "perceptual_quality_validated": False, "online_goodput_measured": False,
               "research_gate_eligible": False, "kernel_concurrency_verified": False,
               "fingerprint_validation": "Compare recorded finite/shape/dtype/SHA256 fields; artifacts are not independently rehashed."}
    return summary


def fmt(value, divisor=1):
    if not number(value):
        return "未知"
    return ("%.4f" % (value / divisor)).rstrip("0").rstrip(".")


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(lines, columns, rows):
    lines.extend(["", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
    lines.extend("| " + " | ".join(cell(item) for item in row) + " |" for row in rows)
    lines.append("")


def _link(path, root, label):
    if not path:
        return "未知"
    path = Path(path)
    try:
        value = path.relative_to(root).as_posix()
    except ValueError:
        value = path.as_posix()
    return "[%s](<%s>)" % (label, value)


def group_label(identity):
    a, b = identity["a"], identity["b_spec"]
    label = "%s×%s，A/B %s/%s 帧，%s 步，cache=%s，seed=%s，插入步=%s" % (
        a.get("height"), a.get("width"), a.get("frames"), b.get("frames"),
        a.get("steps"), a.get("cache"), a.get("seed"), identity.get("insert_after"))
    if (b.get("height"), b.get("width")) != (a.get("height"), a.get("width")):
        label += "，B %s×%s" % (b.get("height"), b.get("width"))
    if b.get("steps") != a.get("steps"):
        label += "，B %s 步" % b.get("steps")
    if identity.get("diagnostic"):
        label += "（诊断模式）"
    return label


def timeline_svg(cases):
    plotted = [case for case in cases if case["events"]["dit_forwards"] or case["events"]["b_decode_cuda_interval_ms"]]
    height = 100 + 105 * len(plotted)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="%d" viewBox="0 0 1200 %d">' % (height, height),
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="13" fill="#18212f">',
             '<text x="20" y="27">阶段诊断时间线：每个 case 使用自己的 CUDA event 相对时间原点</text>',
             '<text x="20" y="49">蓝色：DiT forward 事件区间；橙色：B decode 事件区间。区间重叠不证明 kernel 同时执行。</text>']
    if not plotted:
        parts.append('<text x="20" y="80">没有可绘制的有效事件区间；失败与缺失见报告。</text>')
    for index, case in enumerate(plotted):
        y = 90 + index * 105
        events = case["events"]
        ends = [item["cuda_end_ms"] for item in events["dit_forwards"]]
        if events["b_decode_cuda_interval_ms"]:
            ends.append(events["b_decode_cuda_interval_ms"][1])
        maximum = max(ends + [1.0])
        scale = 940.0 / maximum
        label = "%s / %s / %s" % (case["id"], case["strategy"], case["profile_status"])
        parts.append('<text x="20" y="%d">%s</text>' % (y, html.escape(label)))
        for item in events["dit_forwards"]:
            parts.append('<rect x="%.3f" y="%d" width="%.3f" height="15" fill="#3478ae"><title>DiT %s: %.3f–%.3f ms</title></rect>' %
                         (200 + item["cuda_start_ms"] * scale, y + 10, max(0.4, item["duration_ms"] * scale),
                          html.escape(str(item.get("forward_index"))), item["cuda_start_ms"], item["cuda_end_ms"]))
        if events["b_decode_cuda_interval_ms"]:
            start, end = events["b_decode_cuda_interval_ms"]
            parts.append('<rect x="%.3f" y="%d" width="%.3f" height="15" fill="#d77b25"><title>B decode: %.3f–%.3f ms</title></rect>' %
                         (200 + start * scale, y + 32, max(0.4, (end - start) * scale), start, end))
        parts.append('<text x="200" y="%d">0 ms</text><text x="1010" y="%d">%s ms</text>' % (y + 68, y + 68, fmt(maximum)))
    parts.append('</g></svg>')
    return "\n".join(parts)


def write_report(root, summary):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "summary.json", summary)
    (root / "timeline.svg").write_text(timeline_svg(summary["cases"]), encoding="utf-8")
    with (root / "steps.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("record_index", "id", "strategy", "manifest_status", "selected", "valid_pair", "step_index",
                  "completed_step", "forward_count", "host_monotonic_ns", "mapping_status", "first_forward_index",
                  "last_forward_index", "transformer_event_interval_sum_ms")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in summary["cases"]:
            for step in case["events"]["step_rows"]:
                row = dict(step, **{key: case[key] for key in fields[:6]})
                writer.writerow({key: row.get(key) for key in fields})
    conclusion = summary["conclusion"]
    running = summary["manifest"].get("status") == "running"
    lines = ["# 单 V100 阶段共存诊断", ""]
    if running:
        lines.extend(["**实验进行中。** 以下是当前记录的中间快照，未完成或尚未启动的策略不构成实验失败；暂时判读不能视为整套实验结论。", ""])
    lines.extend([("当前已完成记录的暂时判读：" if running else "") + "**%s。** %s" % (conclusion["label_zh"], conclusion["reason_zh"]), "",
             "现象层：**%s**。%s" % (conclusion["phenomenon"]["label_zh"], conclusion["phenomenon"]["note_zh"]),
             conclusion["decision_rule_zh"], "",
             "这是阶段诊断。remaining_ms、a_completed_ms、b_completed_ms 以剩余阶段窗口为起点，"
             "不是从真实在线到达开始的完整 E2E，也不是 goodput。B 的预生成成本不在该窗口内；进程 elapsed_seconds 另列，不能替代这些指标。", "",
             "仅比较记录中的有限值、shape、dtype 与 SHA256；未重新读取并校验全部输出数组。"
             "组内参考一致不表示感知等质量。perceptual_quality_validated=false，research_gate_eligible=false。", "",
             "共 %d 条 manifest 记录；执行有效 %d 条，A/B 参考均通过 %d 条，完整四策略组 %d。" %
             tuple(summary["counts"][key] for key in ("manifest_records", "valid_execution", "valid_pairs", "complete_four_strategy_groups")),
             "manifest 当前状态：`%s`。失败、预算跳过、缺失文件均保留。" % cell(summary["manifest"].get("status", "unknown")),
             "同 id 重试只选身份一致的最新有效执行；旧尝试保留。若最新尝试失败而采用旧有效结果，会在下表与审计段同时显示失败。",
             "预算已用 / 上限秒：%s / %s（包含加载、失败与保留 GPU 的进程时间）。" %
             tuple(fmt(summary["manifest"].get("budget", {}).get(key)) for key in ("used_seconds", "limit_seconds")), "",
             "## 所有记录与参考校验"])
    table(lines, ["序号 / id / 策略", "manifest / profile", "统计选中", "A / B 校验", "剩余时间 s", "A 完成 s", "B 完成 s", "进程耗时 s", "原始记录"],
          [("%s / %s / %s" % (case["record_index"], case["id"],
                              "准备 B / 规格筛查" if case["mode"] == "prepare" else LABELS.get(case["strategy"], case["mode"])),
            "%s / %s" % (case["manifest_status"], case["profile_status"]),
            "是" if case["selected"] else "否",
            "%s / %s" % (case["a_validation"]["status"], case["b_validation"]["status"]),
            *(fmt(case["timings"][key], 1000) for key in METRICS), fmt(case["elapsed_seconds"]),
            _link(case["profile_path"], root, "JSON")) for case in summary["cases"]])
    lines.extend(["## 四策略比较", "", "只在身份相同且 A/B 参考均通过的记录间比较。每个中位数都列有效分母；"
                  "缺失策略不补零。分组记录数只含已有 profile、能确定规范化身份的记录；尚无 profile 的运行中任务仍保留在总表。"
                  "这里只是描述统计，不给小样本 p99 或显著性结论。完整生成身份与原始配置保留在 summary.json。",
                  "剩余时间包含 A 的 CPU prompt 编码；编码波动也会改变策略排名，几秒的差异不能直接归因于 GPU 阶段干扰。"
                  "编码分项见下表，不能把简单扣除编码后的差值称为完整去噪墙钟。"])
    for index, group in enumerate(summary["groups"]):
        lines.extend(["", "### 组 %d：%s（%s）" % (index + 1, cell(group_label(group["identity"])),
                       "完整" if group["complete_four_strategies"] else "不完整")])
        table(lines, ["策略", "有效 / 记录", "剩余时间中位数 s", "A 完成中位数 s", "B 完成中位数 s", "allocated 峰值 MiB"],
              [(LABELS[strategy], "%d / %d" % (row["valid_pairs"], row["attempted"]),
                *(fmt(row["metrics"][key]["median"], 1000) for key in METRICS),
                fmt(row["peak_allocated_bytes"]["median"], 1024 ** 2)) for strategy, row in group["strategies"].items()])
        lines.append("同步插入 ÷ 异步插入的剩余时间比：%s。" % fmt(group["comparisons"].get("blocking_over_overlap_remaining_ratio")))
        lines.append("异步插入 ÷ 较快串行基线的剩余时间比：%s；比值大于 1 表示异步更慢。无插入基线可跨相同负载的插入步边界复用，不能据此增加独立样本数。" %
                     fmt(group["comparisons"].get("overlap_over_best_serial_remaining_ratio")))
    lines.extend(["", "## 阶段与逐步记录", "", "下列编码与搬运时间按 worker 的独立字段列出。Transformer CUDA event 区间累计是部分去噪诊断，"
                  "包含区间内等待/资源竞争，不是 kernel 时间总和或完整去噪墙钟。B decode CUDA event 区间也不是纯 VAE kernel 时间。",
                  "B 等待至 GPU 解码开始时间直接取 case 的 b_decode.cuda_start_ms：从 B-ready CUDA event 原点到 B GPU 解码阶段起点，"
                  "可能包含 A 编码和此前 DiT 执行的等待；它与解码自身 cuda_interval_ms 分开列出，不等于纯 decoder 队列等待或真实在线到达延迟。缺失时保持未知。",
                  "[逐步 CSV](steps.csv) 使用累计 forward_count 映射 DiT 事件；无法可靠映射时留空，host_monotonic_ns 不与 CUDA 毫秒混算。"])
    table(lines, ["序号 / id", "CPU 编码 s", "embedding H2D ms", "Transformer 事件累计 s", "B 等待至 GPU 解码开始 s", "B decode 事件区间 s", "逐步映射 / 记录"],
          [("%d / %s" % (case["record_index"], case["id"]), fmt(case["timings"].get("cpu_prompt_encode_ms"), 1000),
            fmt(case["timings"].get("prompt_embeddings_h2d_ms")),
            fmt(case["events"].get("transformer_event_interval_sum_ms"), 1000),
            fmt(case["timings"].get("b_wait_to_gpu_decode_start_ms"), 1000),
            fmt(case["events"]["b_decode"].get("cuda_interval_ms"), 1000),
            "%d / %d" % (sum(row["mapping_status"] == "mapped" for row in case["events"]["step_rows"]),
                           len(case["events"]["step_rows"]))) for case in summary["cases"]])
    lines.extend(["", "独立 B 准备阶段如下：denoise 为 worker 定义的 latent-only pipeline 同步墙钟；decode 包含逆归一化、专用线程/stream VAE 和 CPU 输出物化。"
                  "两个阶段各自重置 allocator 峰值，保留初始常驻量，不进行因果差分；失败记录的部分阶段值不计为完整执行。"])
    phase_rows = []
    for case in summary["cases"]:
        if case["mode"] != "prepare":
            continue
        profiles = case["phase_profiles"] if isinstance(case["phase_profiles"], dict) else {}
        for phase in ("denoise", "decode"):
            values = profiles.get(phase) if isinstance(profiles.get(phase), dict) else {}
            phase_rows.append(("%d / %s" % (case["record_index"], case["id"]), case["profile_status"], phase,
                               fmt(values.get("wall_ms"), 1000),
                               *(fmt(values.get(key), 1024 ** 2) for key in
                                 ("initial_allocated_bytes", "initial_reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"))))
    table(lines, ["序号 / id", "执行状态", "独立阶段", "同步墙钟 s", "初始 allocated MiB", "初始 reserved MiB", "峰值 allocated MiB", "峰值 reserved MiB"], phase_rows)
    lines.extend(["", "## 显存与时间线", "", "显存为 PyTorch allocator 的实测峰值，不能把策略峰值差分解释为缓存状态大小；"
                  "不等于设备全部显存。CUDA event 区间交叠不等于内核同执行，需 kernel 级 profiler 才能验证。", "",
                  "[查看时间线](timeline.svg)（覆盖全部有 CUDA 事件区间的记录，每条使用自己的时间轴；完整事件保留在 summary.json）。"])
    table(lines, ["id", "初始 allocated MiB", "峰值 allocated MiB", "峰值 reserved MiB", "A VAE 前 allocated 峰值 MiB", "DiT steps / forwards"],
          [(case["id"], *(fmt(case["memory"].get(key), 1024 ** 2) for key in
                          ("initial_allocated_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "before_a_vae_peak_allocated_bytes")),
            "%s / %s" % (case["events"]["step_count_reported"], case["events"]["forward_count_reported"])) for case in summary["cases"]])
    lines.extend(["", "失败显存观测单列如下。失败前已观测到的 allocator 峰值只是该失败执行的下界，"
                  "不含未成功分配的张量，也不是请求成功完成所需的精确峰值。异常展开可能释放张量，因此捕获时当前 allocated 可明显低于峰值。"
                  "这些数值不填入成功策略的峰值统计，也不与 OOM 消息中的申请量相加。测量窗口外的 warmup OOM 不计为跨请求阶段冲突。"])
    failed_rows = []
    for case in summary["cases"]:
        observation = case["failure_memory_observation"]
        if not observation:
            continue
        scope = {"measured_case": "正式 case 测量期间", "measured_prepare": "独立 B 准备测量期间",
                 "outside_measured_window": "测量窗口外（如 warmup）"}[observation["scope"]]
        progress = (case.get("failure") or {}).get("last_progress") or {}
        failed_rows.append(("%d / %s" % (case["record_index"], case["id"]), scope, progress.get("stage", "未知"),
                            *(fmt(observation["values"].get(key), 1024 ** 2) for key in
                              ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"))))
    table(lines, ["序号 / id", "观测范围", "最后进度（非精确故障算子）", "捕获时 allocated MiB", "捕获时 reserved MiB",
                  "已观测 allocated 峰值下界 MiB", "已观测 reserved 峰值下界 MiB"], failed_rows)
    lines.extend(["## 失败、缺失与审计事项", ""])
    problems = list(summary["issues"])
    for case in summary["cases"]:
        entries = list(case["issues"])
        entries.extend(case["a_validation"].get("issues", []))
        entries.extend(case["b_validation"].get("issues", []))
        if case.get("failure"):
            entries.append("failure=" + _key(case["failure"]))
        if case["manifest_status"] != "completed":
            entries.append("manifest status=" + str(case["manifest_status"]))
        for key in ("reason", "failure_category"):
            if case.get(key):
                entries.append(key + "=" + str(case[key]))
        problems.extend("%s: %s" % (case["id"], entry) for entry in entries)
    lines.extend("- " + cell(problem) for problem in problems)
    if not problems:
        lines.append("没有检测到记录完整性问题；这不构成在线性能或感知质量验证。")
    lines.extend(["", "## 视频导出", "", "视频导出在测量结束后单独执行，原始 NPY 保留；"
                  "[media-export.json](media-export.json) 记录输出 SHA256 到视频路径的映射、复用及全部导出失败。"
                  "该文件不存在时表示尚未导出。视频可用于人工检查，不产生质量通过结论。"])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_media(root, summary):
    """Optional CPU-only export; imports NumPy/imageio only when called.

    The measured arrays and raw profiles are never changed. Rehash the float
    pixels before export; sharing MP4 files does not change the statistical n.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "media-export.json"
    previous_issues = []
    previous = read_json(destination, previous_issues) if destination.exists() else {}
    result = {"schema_version": "ditserve.phase-conflict-media.v1", "status": "completed",
              "fps": 16, "codec": "libx264", "quality": 8, "macro_block_size": 1,
              "timing_scope": "CPU export outside GPU measurements", "perceptual_quality_validated": False,
              "videos": [], "hash_to_video": {}, "errors": [{"message": message} for message in previous_issues],
              "previous_export_errors": (previous or {}).get("previous_export_errors", []) + (previous or {}).get("errors", [])}
    reusable = (previous or {}).get("hash_to_video", {})
    try:
        np = importlib.import_module("numpy")
        imageio = importlib.import_module("imageio.v2")
    except ImportError as exc:
        result["errors"].append({"type": type(exc).__name__, "message": str(exc),
                                  "stage": "optional CPU media dependencies"})
        np = imageio = None
    for case in summary["cases"] if np is not None else []:
        for name in ("output_a", "output_b"):
            output = case.get(name)
            if not output:
                continue
            item = {"record_index": case["record_index"], "id": case["id"], "output": name,
                    "manifest_status": case["manifest_status"],
                    "sha256": output.get("sha256") if isinstance(output, dict) else None}
            temporary = None
            try:
                if not output_valid(output):
                    raise ValueError("output fingerprint missing or non-finite; no export")
                directory = Path(case["profile_path"]).parent
                directory.relative_to(root)
                source = resolve_path(output.get("artifact", output.get("path")), root, directory)
                if source is None or source.suffix.lower() != ".npy":
                    raise ValueError("missing NPY artifact")
                raw = np.load(str(source), allow_pickle=False, mmap_mode="r")
                frames = video_frames(raw, np, expected=output)
                video = directory / "artifacts" / (name.replace("_", "-") + ".mp4")
                video.parent.mkdir(parents=True, exist_ok=True)
                known = result["hash_to_video"].get(output["sha256"], reusable.get(output["sha256"]))
                old = Path(known["video"]) if isinstance(known, dict) and known.get("video") else None
                reuse = bool(old and old.is_file() and known.get("video_sha256") == hashlib.sha256(old.read_bytes()).hexdigest())
                temporary = video.with_name(video.stem + ".partial.mp4")
                if reuse:
                    if old.resolve() != video.resolve():
                        try:
                            os.link(str(old), str(temporary))
                        except OSError:
                            shutil.copyfile(str(old), str(temporary))
                        temporary.replace(video)
                else:
                    imageio.mimwrite(str(temporary), frames, fps=16, codec="libx264", quality=8, macro_block_size=1)
                    temporary.replace(video)
                item.update(status="completed", artifact=str(source), video=str(video),
                            reused_video=reuse, frame_count=len(frames), pixel_sha256_verified=True)
                result["videos"].append(item)
                result["hash_to_video"].setdefault(output["sha256"],
                    {"video": str(video), "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest()})
            except Exception as exc:
                item.update(status="failed", type=type(exc).__name__, message=str(exc))
                result["errors"].append(item)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    if result["errors"]:
        result["status"] = "partial_failure" if result["videos"] else "failed"
    atomic_json(destination, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing manifest.json")
    parser.add_argument("--export-media", action="store_true", help="export finite NPY videos using optional CPU dependencies")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    summary = summarize(args.root)
    write_report(args.root, summary)
    media = export_media(args.root, summary) if args.export_media else None
    archived = archive_run(args.root, "phase-conflict", args.results_root)
    print("阶段诊断报告已写入 %s" % archived["report"])
    return 2 if media and media["status"] in ("failed", "partial_failure") else 0


if __name__ == "__main__":
    raise SystemExit(main())
