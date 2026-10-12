#!/usr/bin/env python3
"""Read saved V100 profiles and write a Chinese report; never load models or GPUs."""

from ..util import ROOT, atomic_json, archive_run

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import statistics


IDENTITY = ("dtype", "stage", "prompt_id", "seed", "arm")
PAIR_CONFIG = ("model", "model_id", "revision", "dtype", "height", "width", "frames", "steps",
               "seed", "prompt", "negative_prompt", "guidance_scale", "batch_size", "attention",
               "device", "vae_dtype", "text_encoder_device", "text_encoder_dtype", "warmup", "repeats")


def finite_number(value, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(value) and (value > 0 if positive else value >= 0))


def read_json(path, issues, required=True):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if required or Path(path).exists():
            issues.append("%s: %s" % (path, exc))
        return None


def read_text(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def source_path(value, run_dir, fallback=None):
    if not value:
        return fallback
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (run_dir / path, ROOT / path):
        if candidate.exists():
            return candidate.resolve()
    return run_dir / path


def identity(record):
    return tuple(record.get(field) for field in IDENTITY)


def public_cache_timing_groups(diagnostic):
    """Group observed step durations only when every public scope agrees."""
    mapping = diagnostic.get("cache_step_mapping") or {}
    result = {"status": "unknown", "groups": None,
              "labels_are_whole_transformer_paths": False,
              "reason": "requires complete mapped scopes with identical per-step cached/CFG labels"}
    if mapping.get("status") == "cache_disabled":
        result.update(status="cache_disabled", reason="cache is disabled; no public reuse labels")
        return result
    entries, steps = mapping.get("entries") or [], diagnostic.get("steps") or []
    count = diagnostic.get("step_count")
    if (not entries or not steps or type(count) is not int or count != len(steps) or
            any(not isinstance(entry, dict) or entry.get("status") != "mapped" for entry in entries)):
        return result
    indices = [entry.get("summary_entry_index") for entry in entries]
    if len(set(indices)) != len(indices) or any(type(index) is not int for index in indices):
        return result
    groups = {"reported_partial_reuse": [], "no_cached_report": []}
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("step_index") != index or not finite_number(step.get("transformer_cuda_ms")):
            return result
        evidence = step.get("cache_evidence") or []
        if (len(evidence) != len(entries) or any(not isinstance(item, dict) for item in evidence) or
                {item.get("summary_entry_index") for item in evidence} != set(indices)):
            return result
        labels = []
        for entry in entries:
            matches = [item for item in evidence if item.get("summary_entry_index") == entry["summary_entry_index"]]
            if len(matches) != 1:
                return result
            item = matches[0]
            pair = (item.get("reported_cached"), item.get("reported_cfg_cached"))
            if any(type(flag) is not bool for flag in pair):
                return result
            if (not isinstance(entry.get("cached_steps"), list) or not isinstance(entry.get("cfg_cached_steps"), list) or
                    pair != (index in entry["cached_steps"], index in entry["cfg_cached_steps"])):
                return result
            labels.append(pair)
        if len(set(labels)) != 1:
            result["reason"] = "public scopes disagree on at least one step; no combined-label timing distribution"
            return result
        group = "reported_partial_reuse" if any(labels[0]) else "no_cached_report"
        groups[group].append((index, step["transformer_cuda_ms"]))
    result.update(status="mapped_consistent_public_labels", groups={}, summary_entry_indices=indices,
                  reason="single mapped scope or identical cached/CFG flags across all mapped scopes at every step; Fn/Bn computation remains")
    for name, samples in groups.items():
        values = sorted(value for index, value in samples)
        percentile = None
        if values:
            position = (len(values) - 1) * 0.95
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            percentile = values[lower] + (position - lower) * (values[upper] - values[lower])
        result["groups"][name] = {"n_steps": len(values), "step_indices": [index for index, value in samples],
                                  "min_ms": min(values) if values else None,
                                  "median_ms": statistics.median(values) if values else None,
                                  "p95_descriptive_ms": percentile, "max_ms": max(values) if values else None}
    return result


def extract_repeat(run, index):
    issues = []
    timing = run.get("uninstrumented") or {}
    diagnostic = run.get("diagnostic") or {}
    result = {"index": run.get("index", index), "seed": run.get("seed"),
              "status": run.get("status"), "measured": run.get("measured"), "issues": issues}
    for field in ("e2e_ms", "cpu_prompt_encode_ms", "prompt_embeddings_h2d_ms",
                  "initial_allocated_bytes", "peak_allocated_bytes", "peak_reserved_bytes"):
        value = timing.get(field)
        result[field] = value if finite_number(value, positive=field == "e2e_ms") else None
        if result[field] is None:
            issues.append("uninstrumented.%s missing/invalid" % field)
    result["timed_window_unix_ns"] = {key: timing.get(key) for key in ("started_at_unix_ns", "finished_at_unix_ns")}
    result["diagnostic_window_unix_ns"] = {key: diagnostic.get(key) for key in ("started_at_unix_ns", "finished_at_unix_ns")}
    result["cache_refresh"] = timing.get("cache_refresh")
    if result["cache_refresh"] is None:
        issues.append("cache_refresh observability record missing")
    forwards = diagnostic.get("forwards")
    forward_count = diagnostic.get("forward_count")
    complete_forwards = (isinstance(forwards, list) and len(forwards) > 0 and
                         type(forward_count) is int and forward_count == len(forwards) and
                         all(isinstance(forward, dict) and forward.get("forward_index") == i and
                             finite_number(forward.get("cuda_ms")) for i, forward in enumerate(forwards)))
    result["diagnostic_transformer_cuda_sum_ms"] = sum(forward["cuda_ms"] for forward in forwards) if complete_forwards else None
    result["diagnostic_forward_count"] = forward_count
    result["diagnostic_step_count"] = diagnostic.get("step_count")
    if not complete_forwards:
        issues.append("diagnostic transformer forward events missing/incomplete; total is unknown")
    result["cache_statistics_timed"] = run.get("cache_statistics")
    result["cache_statistics_diagnostic"] = diagnostic.get("cache_statistics")
    mapping = diagnostic.get("cache_step_mapping") or {}
    result["cache_mapping_status"] = mapping.get("status", "missing")
    result["cache_scopes"] = []
    for entry in mapping.get("entries", []):
        scope = dict(entry)
        valid = (entry.get("status") == "mapped" and type(diagnostic.get("step_count")) is int and
                 all(isinstance(entry.get(key), list) and
                     all(type(value) is int and 0 <= value < diagnostic["step_count"] for value in entry[key])
                     for key in ("cached_steps", "cfg_cached_steps")))
        for field, key in (("reported_cached_step_count", "cached_steps"), ("reported_cfg_cached_step_count", "cfg_cached_steps")):
            scope[field] = len(set(entry[key])) if valid else None
        scope["reported_either_step_count"] = len(set(entry["cached_steps"]) | set(entry["cfg_cached_steps"])) if valid else None
        result["cache_scopes"].append(scope)
    result["step_evidence"] = [{key: step.get(key) for key in
                                ("step_index", "forward_indices", "transformer_cuda_ms", "latents_finite", "cache_evidence")}
                               for step in diagnostic.get("steps", []) if isinstance(step, dict)]
    result["compute_reuse_path_timing"] = {
        "status": "unknown", "full_compute_transformer_cuda_ms": None,
        "reuse_transformer_cuda_ms": None, "full_denoising_loop_wall_ms": None,
        "source_classification": mapping.get("whole_transformer_compute_or_reuse", "not_classified"),
        "reason": "Public entries can overlap and DBCache Fn/Bn blocks still execute; reported_cached=false is not proof of whole-transformer full compute. Primary paired profiles do not independently time the full denoising loop; optional phase supplements are reported separately.",
    }
    result["public_cache_step_timing_groups"] = public_cache_timing_groups(diagnostic)
    if result["cache_mapping_status"] not in ("mapped_public_statistics", "cache_disabled"):
        issues.append("per-step cache mapping missing/unsupported")
    timed_output = run.get("output") or {}
    diagnostic_output = diagnostic.get("output") or {}
    result["outputs"] = {"timed": timed_output, "diagnostic": diagnostic_output}
    hashes = [output.get("sha256") for output in (timed_output, diagnostic_output)]
    hashes_valid = all(isinstance(value, str) and len(value) == 64 and
                       all(character in "0123456789abcdefABCDEF" for character in value) for value in hashes)
    result["timed_diagnostic_sha256_equal"] = hashes[0].lower() == hashes[1].lower() if hashes_valid else None
    result["reported_instrumentation_output_exact_match"] = run.get("instrumentation_output_exact_match")
    if not hashes_valid:
        issues.append("timed/diagnostic output SHA256 missing/invalid")
    elif run.get("instrumentation_output_exact_match") is not result["timed_diagnostic_sha256_equal"]:
        issues.append("reported instrumentation hash equality missing or inconsistent with saved SHA256")
    if any(output.get("finite") is not True for output in (timed_output, diagnostic_output)):
        issues.append("timed/diagnostic finite output evidence missing or false")
    return result


def summarize_case(record, run_dir):
    case = {field: record.get(field) for field in IDENTITY}
    case.update(status=record.get("status", "unknown"), failure_category=record.get("failure_category"),
                failure=record.get("failure"), telemetry=record.get("telemetry"),
                raw_log=record.get("raw_log"), issues=[], repeats=[], valid_completion=False)
    directory = source_path(record.get("directory"), run_dir)
    path = source_path(record.get("profile_file"), run_dir, directory / "profile.json" if directory else None)
    case["profile_file"] = str(path) if path else None
    if case["status"] == "not_attempted":
        case["issues"].append("planned arm not attempted; no measurement")
        return case
    profile = read_json(path, case["issues"]) if path else None
    if not isinstance(profile, dict):
        case["issues"].append("profile JSON unavailable; cached runner aggregate is not substituted")
        return case
    case.update(profile_status=profile.get("status"), profile_measured=profile.get("measured"),
                profile_kind=profile.get("kind"), config=profile.get("config") or {},
                environment=profile.get("environment") or {}, revision=profile.get("revision"),
                scheduler=profile.get("scheduler"), cache=profile.get("cache"),
                measurement_protocol=profile.get("measurement_protocol"))
    case["failure"] = profile.get("failure") or case["failure"]
    runs = profile.get("runs") or []
    for index, run in enumerate(runs):
        if isinstance(run, dict):
            case["repeats"].append(extract_repeat(run, index))
        else:
            case["issues"].append("malformed repeat %s" % index)
    completed = [run for run in case["repeats"] if run["status"] == "completed" and run["measured"] is True and run["e2e_ms"] is not None]
    case["valid_completion"] = (case["status"] == "completed" and profile.get("status") == "completed" and
                                profile.get("kind") == "gpu_profile" and profile.get("measured") is True and
                                len(completed) == len(runs) == 1 and
                                all(output.get("finite") is True for run in completed for output in run["outputs"].values()))
    if case["status"] == "completed" and not case["valid_completion"]:
        case["issues"].append("claimed completion lacks the fixed one-repeat completed/finite measured profile")
    config = case["config"]
    expected = dict(record.get("shape") or {}, dtype=record.get("dtype"), seed=record.get("seed"), cache=record.get("arm"))
    for field, value in expected.items():
        if config.get(field) != value:
            case["issues"].append("profile config differs from planned %s" % field)
            case["valid_completion"] = False
    case["metrics"] = {}
    for field in ("e2e_ms", "cpu_prompt_encode_ms", "prompt_embeddings_h2d_ms", "diagnostic_transformer_cuda_sum_ms",
                  "peak_allocated_bytes", "peak_reserved_bytes"):
        values = [run[field] for run in completed if run[field] is not None]
        # Missing supporting fields remain unknown rather than silently changing their denominator.
        case["metrics"][field] = statistics.median(values) if completed and len(values) == len(completed) else None
    for field in ("cpu_prompt_encode_ms", "prompt_embeddings_h2d_ms"):
        value, total = case["metrics"][field], case["metrics"]["e2e_ms"]
        case["metrics"][field + "_e2e_fraction"] = value / total if value is not None and total else None
    return case


def scheduler_for_comparison(scheduler):
    """Normalize only Diffusers' unordered string-list bookkeeping metadata.

    Diffusers 0.36.0 configuration_utils.py:695/743 creates this field using
    list(set(...)); lines 469-470 inspect membership. Other lists stay ordered.
    Return shallow copies so the original scheduler evidence remains intact.
    """
    if not isinstance(scheduler, dict) or not isinstance(scheduler.get("config"), dict):
        return scheduler
    defaults = scheduler["config"].get("_use_default_values")
    if not isinstance(defaults, list) or not all(isinstance(value, str) for value in defaults):
        return scheduler
    result = dict(scheduler)
    result["config"] = dict(scheduler["config"], _use_default_values=sorted(defaults))
    return result


def make_pairs(cases, selected_dtype):
    groups = {}
    for case in cases:
        if case["dtype"] == selected_dtype:
            key = case["stage"], case["prompt_id"], case["seed"]
            groups.setdefault(key, {}).setdefault(case["arm"], []).append(case)
    pairs = []
    for (stage, prompt_id, seed), arms in groups.items():
        pair = {"stage": stage, "prompt_id": prompt_id, "seed": seed, "dtype": selected_dtype,
                "status": "incomplete", "issues": [], "quality_status": "manual_review_pending",
                "research_gate_eligible": False}
        baseline = arms.get("none", [])
        cached = arms.get("cache-dit", [])
        if len(baseline) != 1 or len(cached) != 1:
            pair["issues"].append("missing or duplicate arm; no pairing performed")
        else:
            baseline, cached = baseline[0], cached[0]
            pair["baseline_profile"] = baseline["profile_file"]
            pair["cache_profile"] = cached["profile_file"]
            pair["baseline_status"], pair["cache_status"] = baseline["status"], cached["status"]
            if baseline["valid_completion"] and cached["valid_completion"]:
                for field in PAIR_CONFIG:
                    if baseline["config"].get(field) != cached["config"].get(field):
                        pair["issues"].append("paired config differs: " + field)
                if scheduler_for_comparison(baseline.get("scheduler")) != scheduler_for_comparison(cached.get("scheduler")):
                    pair["issues"].append("paired scheduler differs")
                if baseline.get("revision") != cached.get("revision"):
                    pair["issues"].append("paired revision resolution differs")
                if not pair["issues"]:
                    pair["status"] = "completed"
                    for label, case in (("baseline", baseline), ("cache", cached)):
                        for field, value in case["metrics"].items():
                            pair[label + "_" + field] = value
                    base_e2e, cache_e2e = baseline["metrics"]["e2e_ms"], cached["metrics"]["e2e_ms"]
                    pair["e2e_baseline_over_cache"] = base_e2e / cache_e2e
                    pair["e2e_latency_reduction_fraction"] = 1 - cache_e2e / base_e2e
                    base_dit = baseline["metrics"].get("diagnostic_transformer_cuda_sum_ms")
                    cache_dit = cached["metrics"].get("diagnostic_transformer_cuda_sum_ms")
                    pair["diagnostic_transformer_cuda_baseline_over_cache"] = (
                        base_dit / cache_dit if finite_number(base_dit, positive=True) and finite_number(cache_dit, positive=True) else None)
                    for metric in ("peak_allocated_bytes", "peak_reserved_bytes"):
                        base_value, cache_value = baseline["metrics"][metric], cached["metrics"][metric]
                        pair[metric + "_cache_minus_baseline"] = cache_value - base_value if base_value is not None and cache_value is not None else None
            else:
                pair["issues"].append("one or both arms lack a valid completed measurement")
        pairs.append(pair)
    return pairs


def formal_snapshot(pairs):
    """Descriptive numbers for completed formal pairs, with per-metric n."""
    rows = []
    for pair in pairs:
        if pair.get("stage") != "formal":
            continue
        row = {field: pair.get(field) for field in ("prompt_id", "seed", "status")}
        complete = pair.get("status") == "completed"
        for label in ("baseline", "cache"):
            for source, target in (("e2e_ms", "e2e_seconds"), ("diagnostic_transformer_cuda_sum_ms", "diagnostic_transformer_cuda_seconds")):
                value = pair.get(label + "_" + source)
                row[label + "_" + target] = value / 1000 if complete and finite_number(value) else None
        for field in ("e2e_baseline_over_cache", "diagnostic_transformer_cuda_baseline_over_cache"):
            row[field] = pair.get(field) if complete else None
        delta = pair.get("peak_allocated_bytes_cache_minus_baseline")
        row["allocated_peak_cache_minus_baseline_mib"] = delta / (1024 * 1024) if complete and isinstance(delta, (int, float)) and not isinstance(delta, bool) and math.isfinite(delta) else None
        rows.append(row)
    metric_names = ("baseline_e2e_seconds", "cache_e2e_seconds", "baseline_diagnostic_transformer_cuda_seconds",
                    "cache_diagnostic_transformer_cuda_seconds", "e2e_baseline_over_cache",
                    "diagnostic_transformer_cuda_baseline_over_cache", "allocated_peak_cache_minus_baseline_mib")
    medians = {}
    for field in metric_names:
        values = [row[field] for row in rows if row[field] is not None]
        medians[field] = {"value": statistics.median(values) if values else None, "n": len(values)}
    return {"planned_pairs": len(rows), "completed_pairs": sum(row["status"] == "completed" for row in rows),
            "rows": rows, "descriptive_medians": medians}


def build_summary(run_dir, environment_dir):
    run_dir, environment_dir = Path(run_dir).resolve(), Path(environment_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError("run directory does not exist: " + str(run_dir))
    issues = []
    manifest = read_json(run_dir / "manifest.json", issues) or {}
    records = read_json(run_dir / "records.json", issues)
    if not isinstance(records, list):
        issues.append("records.json unavailable; reading surviving per-case record.json files")
        records = []
        for path in sorted(run_dir.rglob("record.json")):
            record = read_json(path, issues)
            if isinstance(record, dict):
                records.append(record)
    records = [record for record in records if isinstance(record, dict)]
    # The runner may be interrupted before its current case reaches records.json.
    # Preserve that surviving attempt instead of marking it never attempted.
    known = {identity(record) for record in records}
    for path in sorted(run_dir.rglob("record.json")):
        record = read_json(path, issues)
        if isinstance(record, dict) and identity(record) not in known:
            records.append(record)
            known.add(identity(record))
    attempted = {identity(record) for record in records}
    for job in (manifest.get("initial_plan") or []) + (manifest.get("fp32_plan") or []):
        if identity(job) not in attempted:
            records.append(dict(job, status="not_attempted"))
            attempted.add(identity(job))
    cases = [summarize_case(record, run_dir) for record in records]
    dtype = manifest.get("selected_dtype")
    if dtype not in ("float16", "float32"):
        issues.append("selected_dtype missing/invalid; no cross-dtype pairing is inferred")
    pairs = make_pairs(cases, dtype)
    environment = {"directory": str(environment_dir)}
    for name in ("preflight.json", "model-revision.json", "model-files.json"):
        environment[name] = read_json(environment_dir / name, issues)
    for name in ("requirements-lock.txt", "dependency-check.txt", "model-path.txt"):
        environment[name] = read_text(environment_dir / name)
        if environment[name] is None:
            issues.append("environment evidence missing: " + name)
    completed = [pair for pair in pairs if pair["status"] == "completed"]
    formal = [pair for pair in completed if pair["stage"] == "formal"]
    return {"schema_version": "ditserve.v100-detailed-summary.v1",
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_directory": str(run_dir), "manifest": manifest, "environment": environment,
            "selected_dtype": dtype, "issues": issues, "cases": cases, "pairs": pairs,
            "scheduler_comparison": "exact except sorted config._use_default_values when it is a string list; original scheduler evidence is retained",
            "counts": {"attempted": sum(case["status"] != "not_attempted" for case in cases),
                       "not_attempted": sum(case["status"] == "not_attempted" for case in cases),
                       "valid_completed_arms_all_dtypes": sum(case["valid_completion"] for case in cases),
                       "completed_pairs_selected_dtype": len(completed), "formal_completed_pairs": len(formal)},
            "formal_e2e_ratio_median": statistics.median(pair["e2e_baseline_over_cache"] for pair in formal) if formal else None,
            "scope": "single-V100 offline paired generation; diagnostic CUDA-event sums are not full denoising wall time",
            "quality_status": "manual_review_pending", "perceptual_quality_validated": False,
            "research_gate_eligible": False}


def display(value):
    if value is None:
        return "未知"
    if value is True:
        return "一致"
    if value is False:
        return "不一致"
    if isinstance(value, float):
        return format(value, ",.4f").rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(lines, headings, rows):
    lines.extend(["| " + " | ".join(headings) + " |", "| " + " | ".join("---" for _ in headings) + " |"])
    lines.extend("| " + " | ".join(display(value) for value in row) + " |" for row in rows)
    lines.append("")


def link(path, output, label="原始 JSON"):
    return "[%s](%s)" % (label, os.path.relpath(str(path), str(output))) if path else "缺失"


def case_label(case):
    return "%s/%s/%s seed=%s %s" % tuple(case.get(field) for field in IDENTITY)


def attach_manual_review(summary, output):
    """Transcribe an optional human record; it cannot satisfy a quality gate."""
    path = output / "review/formal-review.json"
    review = read_json(path, summary["issues"], required=False)
    def text_list(value):
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    valid = (isinstance(review, dict) and review.get("status") == "completed" and
             all(isinstance(review.get(key), str) for key in ("method_zh", "scope_zh")) and
             text_list(review.get("overall_observations_zh")) and isinstance(review.get("pairs"), list) and
             all(isinstance(pair, dict) and isinstance(pair.get("prompt_id"), str) and type(pair.get("seed")) is int and
                 isinstance(pair.get("frames_reviewed"), list) and
                 all(type(index) is int and index >= 0 for index in pair["frames_reviewed"]) and
                 isinstance(pair.get("artifact"), str) and not Path(pair["artifact"]).is_absolute() and
                 ".." not in Path(pair["artifact"]).parts and text_list(pair.get("observations_zh")) and
                 pair.get("obvious_additional_degradation") in ("not_observed", "observed", "inconclusive")
                 for pair in review["pairs"]))
    summary["manual_review"] = {"source": str(path), "record": review, "completed_record_available": valid}
    summary["quality_status"] = "limited_manual_review_recorded" if valid else "manual_review_pending"
    summary["perceptual_quality_validated"] = False
    summary["research_gate_eligible"] = False
    reviewed_pairs = {(pair["prompt_id"], pair["seed"]) for pair in review["pairs"]} if valid else set()
    for pair in summary["pairs"]:
        reviewed = pair.get("stage") == "formal" and (pair.get("prompt_id"), pair.get("seed")) in reviewed_pairs
        pair["quality_status"] = "limited_manual_review_recorded" if reviewed else "manual_review_pending"
        pair["perceptual_quality_validated"] = False
        pair["research_gate_eligible"] = False
    if review is not None and not valid:
        summary["issues"].append("manual review is incomplete or does not match the expected schema: " + str(path))
    return review if valid else None


def phase_selection(output):
    """Resolve explicit retry choices and retain the declared failed attempts."""
    output = Path(output)
    path = output / "phase-selection.json"
    selection = {"source": str(path) if path.exists() else None, "status": "default",
                 "reason": None, "issues": [], "record": None, "previous_failed_attempts": []}
    paths = {arm: output / "phases" / arm / "profile.json" for arm in ("none", "cache-dit")}
    if not path.exists():
        return paths, selection
    record = read_json(path, selection["issues"])
    selection["record"] = record
    profiles = record.get("selected_profiles") if isinstance(record, dict) else None
    valid = (isinstance(profiles, dict) and set(profiles) == {"none", "cache-dit"} and
             all(isinstance(value, str) and Path(value).is_absolute() for value in profiles.values()) and
             isinstance(record.get("reason"), str) and bool(record["reason"].strip()) and
             isinstance(record.get("previous_failed_profiles"), list) and
             all(isinstance(value, str) and bool(value) for value in record["previous_failed_profiles"]))
    selection["status"] = "explicit" if valid else "invalid"
    if isinstance(record, dict):
        selection["reason"] = record.get("reason")
        for value in record.get("previous_failed_profiles", []) if isinstance(record.get("previous_failed_profiles"), list) else []:
            if not isinstance(value, str) or not value:
                continue
            failed_path = Path(value) if Path(value).is_absolute() else output / value
            attempt = {"source": str(failed_path), "issues": []}
            previous = read_json(failed_path, attempt["issues"])
            attempt.update(status=previous.get("status", "unknown") if isinstance(previous, dict) else "missing",
                           failure=previous.get("failure") if isinstance(previous, dict) else None, record=previous)
            if attempt["status"] not in ("failed", "missing"):
                attempt["issues"].append("declared previous failed profile does not report status=failed")
            selection["previous_failed_attempts"].append(attempt)
    if not valid:
        selection["issues"].append("invalid phase-selection.json: require both arms with absolute profile paths, reason, and previous_failed_profiles; default profiles are not substituted")
        return {}, selection
    return {arm: Path(value) for arm, value in profiles.items()}, selection


def read_phase_supplement(output):
    """Keep optional instrumented phase measurements outside primary statistics."""
    paths, selection = phase_selection(output)
    rows = []
    for arm in ("none", "cache-dit"):
        path = paths.get(arm)
        row = {"arm": arm, "source": str(path) if path else None, "status": "missing", "issues": [],
               "denoising_loop_seconds": None, "reset_call_wall_ms": None,
               "reset_synchronized_wall_ms": None, "reset_post_call_synchronize_ms": None,
               "reference_hash_equal": None, "reset_status": "unknown"}
        record = read_json(path, row["issues"], required=selection["status"] == "explicit") if path else None
        if selection["status"] == "invalid":
            row["status"] = "incomplete"
            row["issues"].append("explicit phase selection is invalid; no profile selected")
        row["record"] = record
        if isinstance(record, dict):
            row["status"] = "incomplete"
            phase = record.get("phase_run") or {}
            loop = phase.get("denoising_loop") or {}
            config = record.get("config") or {}
            reset = phase.get("internal_cache_reset") or {}
            complete = (record.get("schema_version") == "ditserve.v100-phases.v1" and
                        record.get("kind") == "gpu_phase_supplement" and record.get("arm") == arm and
                        record.get("status") == "completed" and record.get("measured") is True and
                        record.get("instrumented") is True and record.get("telemetry_complete") is True and
                        bool(record.get("finished_at_utc")) and
                        phase.get("status") == "completed" and phase.get("seed") == 0 and
                        (phase.get("output") or {}).get("finite") is True and
                        all(config.get(key) == value for key, value in
                            {"seed": 0, "steps": 50, "height": 480, "width": 832, "frames": 17}.items()) and
                        loop.get("status") == "completed" and loop.get("instrumented") is True and
                        loop.get("progress_updates") == 50 and finite_number(loop.get("wall_ms"), positive=True))
            calls = reset.get("calls") or []
            if arm == "cache-dit":
                complete = (complete and reset.get("status") == "measured_populated_pipeline_reset" and
                            reset.get("instrumented") is True and reset.get("all_resets_precede_denoising_loop") is True and
                            bool(calls) and reset.get("context_count") == len(calls) and
                            all(isinstance(call, dict) and call.get("status") == "completed" and
                                all(finite_number(call.get(key)) for key in
                                    ("call_wall_ms", "synchronized_wall_ms", "post_call_synchronize_ms")) and
                                (call.get("before") or {}).get("executed_steps") == 50 and
                                (call.get("before") or {}).get("transformer_executed_steps") == 100 and
                                any(tensor.get("device") == "cuda:0" and finite_number(tensor.get("bytes"), positive=True)
                                    for tensor in ((call.get("before") or {}).get("buffer_inventory") or {}).get("tensors", []))
                                for call in calls))
            else:
                complete = complete and reset.get("status") == "not_applicable"
            if complete:
                row.update(status="completed", denoising_loop_seconds=loop["wall_ms"] / 1000,
                           reset_status=reset["status"])
                if arm == "cache-dit":
                    row["reset_call_wall_ms"] = sum(call["call_wall_ms"] for call in calls)
                    row["reset_synchronized_wall_ms"] = sum(call["synchronized_wall_ms"] for call in calls)
                    row["reset_post_call_synchronize_ms"] = sum(call["post_call_synchronize_ms"] for call in calls)
                comparison = phase.get("output_vs_existing_same_arm") or {}
                hashes = [comparison.get("reference_sha256"), comparison.get("phase_sha256")]
                hashes_valid = all(isinstance(value, str) and len(value) == 64 and
                                   all(character in "0123456789abcdefABCDEF" for character in value) for value in hashes)
                if hashes_valid and hashes[1] == phase["output"].get("sha256"):
                    equal = hashes[0].lower() == hashes[1].lower()
                    if comparison.get("exact_match") is equal:
                        row["reference_hash_equal"] = equal
                if row["reference_hash_equal"] is None:
                    row["issues"].append("reference hash comparison missing/inconsistent; equality is unknown")
            else:
                row["issues"].append("phase record is failed, partial, or lacks complete fixed-scope evidence; phase timings remain unknown")
        rows.append(row)
    complete_count = sum(row["status"] == "completed" for row in rows)
    return {"status": "completed" if complete_count == 2 else "partial" if complete_count else "unavailable",
            "completed_arms": complete_count, "rows": rows, "selection": selection, "instrumented": True,
            "scope": "object seed0 only; separate full warmup and instrumented full request per arm",
            "included_in_primary_e2e_statistics": False, "perceptual_quality_validated": False,
            "research_gate_eligible": False}


def write_report(summary, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    review = attach_manual_review(summary, output)
    summary["phase_supplement"] = read_phase_supplement(output)
    summary["formal_readable_summary"] = formal_snapshot(summary["pairs"])
    atomic_json(output / "v100-summary.json", summary)
    fields = ["stage", "prompt_id", "seed", "dtype", "status"] + sorted({key for pair in summary["pairs"] for key in pair} - {"stage", "prompt_id", "seed", "dtype", "status"})
    with (output / "v100-pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in pair.items()} for pair in summary["pairs"])
    step_fields = list(IDENTITY) + ["repeat", "step_index", "transformer_cuda_ms", "forward_indices",
                                   "latents_finite", "cache_mapping_status", "public_scope_evidence",
                                   "whole_transformer_path"]
    with (output / "v100-steps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=step_fields)
        writer.writeheader()
        for case in summary["cases"]:
            for repeat in case["repeats"]:
                for step in repeat["step_evidence"]:
                    row = {field: case.get(field) for field in IDENTITY}
                    row.update(repeat=repeat["index"], step_index=step["step_index"],
                               transformer_cuda_ms=step["transformer_cuda_ms"],
                               forward_indices=json.dumps(step["forward_indices"]), latents_finite=step["latents_finite"],
                               cache_mapping_status=repeat["cache_mapping_status"],
                               public_scope_evidence=json.dumps(step["cache_evidence"], ensure_ascii=False),
                               whole_transformer_path="unknown")
                    writer.writerow(row)
    counts, manifest, environment = summary["counts"], summary["manifest"], summary["environment"]
    preflight = environment.get("preflight.json") or {}
    gpu = manifest.get("gpu") or preflight.get("gpu") or {}
    revision = environment.get("model-revision.json") or {}
    quality_intro = ("已完成有限范围人工画面检查，见下文；未作等质量统计验证。" if review else "感知质量仍待人工评价。")
    lines = ["# V100 Wan 缓存对照结果", "",
             "本报告只汇总已保存的真实 profiler 记录。单卡离线配对测量不包含在线排队、SLO goodput 或 4 GPU 调度；小样本不能支持稳定 p99 或研究方案达到 20% 门槛的结论。" + quality_intro, "",
             "运行目录：`%s`；runner 状态：`%s`；选定精度：`%s`。" % (summary["run_directory"], manifest.get("status", "unknown"), summary["selected_dtype"]), "",
             "尝试 %s 个 arm，未执行 %s 个；选定精度完成 %s 对，其中正式阶段 %s 对。失败、未执行、缺失数据及被 FP32 替代的 FP16 记录均保留；不同精度不混合计算。" % (counts["attempted"], counts["not_attempted"], counts["completed_pairs_selected_dtype"], counts["formal_completed_pairs"]), "",
             "## 正式配对速览", ""]
    snapshot = summary["formal_readable_summary"]
    lines.extend(["当前完整正式配对 n=%s / %s。下表保留尚未完成或不可配对的项目；空白测项显示未知，不填零。两种比值均为无缓存 ÷ 缓存，超过 1 表示该项缓存组耗时较低。" % (snapshot["completed_pairs"], snapshot["planned_pairs"]), ""])
    def two_values(row, suffix):
        return "%s / %s" % (display(row.get("baseline_" + suffix)), display(row.get("cache_" + suffix)))
    table(lines, ["prompt / seed", "配对状态", "E2E 秒：无/缓存", "DiT 诊断秒：无/缓存", "E2E 比值", "DiT 比值", "allocated 峰值增量 MiB"],
          (["%s / %s" % (row["prompt_id"], row["seed"]), "完成" if row["status"] == "completed" else "未完整配对",
            two_values(row, "e2e_seconds"), two_values(row, "diagnostic_transformer_cuda_seconds"),
            row["e2e_baseline_over_cache"], row["diagnostic_transformer_cuda_baseline_over_cache"],
            row["allocated_peak_cache_minus_baseline_mib"]] for row in snapshot["rows"]))
    table(lines, ["正式配对描述性中位数", "数值", "有效配对 n"],
          ([label, snapshot["descriptive_medians"][field]["value"], snapshot["descriptive_medians"][field]["n"]]
           for field, label in (("baseline_e2e_seconds", "无缓存 E2E 秒"), ("cache_e2e_seconds", "缓存 E2E 秒"),
                               ("baseline_diagnostic_transformer_cuda_seconds", "无缓存 DiT 诊断秒"),
                               ("cache_diagnostic_transformer_cuda_seconds", "缓存 DiT 诊断秒"),
                               ("e2e_baseline_over_cache", "E2E 比值"),
                               ("diagnostic_transformer_cuda_baseline_over_cache", "DiT 比值"),
                               ("allocated_peak_cache_minus_baseline_mib", "allocated 峰值增量 MiB"))))
    lines.extend(["DiT 诊断秒是另一次同 seed 运行中 transformer forward CUDA event 的总和，不是完整去噪墙钟。显存增量为 cache − baseline 的 PyTorch allocator allocated 峰值差，不是完整缓存状态大小。这些中位数仅描述已有配对；perceptual_quality_validated=false、research_gate_eligible=false，不能据此验证在线 20% 门槛。", "",
             "## 环境与计时口径", "",
             "GPU：%s；目标 UUID：`%s`。模型：`%s`；固定 revision：`%s`。" % (gpu.get("name", "未知"), gpu.get("uuid", "未知"), revision.get("id", "未知"), revision.get("sha", "未知")), "",
             "依赖版本：%s。CPU 线程设置：`%s`。完整软件、权重和设备证据保存在机器可读汇总的 environment、manifest 及每个 case.environment。" % (json.dumps(preflight.get("packages", {}), ensure_ascii=False), json.dumps(manifest.get("process_environment", {}), ensure_ascii=False)), "",
             "缓存组固定使用上游 `max_continuous_cached_steps=%s`；参数原样传递给 cache-dit，不额外声称实现了严格连续三步的自定义限制。" % (manifest.get("config", {}).get("cache", {}).get("max_continuous", "未知")), "",
             "E2E 使用无诊断 hooks 的 `uninstrumented.e2e_ms`，包括每请求 CPU 文本编码、embedding H2D、相关有限值检查、去噪、VAE 解码和 CPU NumPy 输出。CPU 编码字段含 CPU 有限值检查；H2D 字段含类型转换、同步及 GPU embedding 有限值检查，不是纯 PCIe 带宽测试。下载、加载、warmup、hash 和文件写入不在 E2E 内。", "",
             "完整 pipeline 在调用内部自动重置 cache-dit 上下文，该成本已含在 E2E 中。主配对记录未单独测量内部 reset；如有补测见专节。`cache_refresh.wall_ms=0` 表示未执行外部刷新，不表示重置成本为零。`pipeline_plus_refresh_ms` 是兼容 E2E 别名，不重复相加。", "",
             "诊断 transformer CUDA 总和来自另一次同 seed 运行的各次 forward CUDA event，包含 CFG 的多次 forward，仅覆盖去噪工作的一部分。主配对记录未单独测量完整去噪 loop 墙钟；如有补测见专节。该 CUDA 总和也不包含完整 CPU、scheduler、VAE 或队列成本，不能从 E2E 减去它推断其他阶段耗时。", "",
             "配对时只规范化 scheduler.config._use_default_values 字符串列表的顺序：Diffusers 0.36.0 从集合生成该元数据。其他 scheduler 字段和列表仍严格比较，原始 JSON 证据保留。", "",
             "## 完成配对：E2E 与阶段指标", ""])
    completed = [pair for pair in summary["pairs"] if pair["status"] == "completed"]
    table(lines, ["阶段/prompt/seed", "组", "E2E ms", "CPU 编码 ms", "embedding H2D ms", "诊断 transformer CUDA 总和 ms"],
          (["%s/%s/%s" % (pair["stage"], pair["prompt_id"], pair["seed"]), label] +
           [pair.get(prefix + "_" + metric) for metric in ("e2e_ms", "cpu_prompt_encode_ms", "prompt_embeddings_h2d_ms", "diagnostic_transformer_cuda_sum_ms")]
           for pair in completed for prefix, label in (("baseline", "无缓存"), ("cache", "cache-dit"))))
    table(lines, ["阶段/prompt/seed", "E2E baseline/cache", "延迟降低比例"],
          (["%s/%s/%s" % (pair["stage"], pair["prompt_id"], pair["seed"]), pair["e2e_baseline_over_cache"], pair["e2e_latency_reduction_fraction"]] for pair in completed))
    lines.extend(["正式阶段完整配对的 E2E baseline/cache 中位数：%s（分母 %s 对）。只描述这些配对，不是在线吞吐或质量约束下的收益。" % (display(summary["formal_e2e_ratio_median"]), counts["formal_completed_pairs"]), "", "## Allocator 峰值", "",
                  "以下均为无诊断运行的 PyTorch allocator 峰值，单位 MiB；差值定义为 cache − baseline，可正可负。它们不是设备全部显存占用，差值也不是完整缓存状态大小；完整缓存快照、恢复和迁移成本仍未实测。", ""])
    def mib(value):
        return value / (1024 * 1024) if value is not None else None
    table(lines, ["阶段/prompt/seed", "allocated baseline", "allocated cache", "allocated 差值", "reserved baseline", "reserved cache", "reserved 差值"],
          (["%s/%s/%s" % (pair["stage"], pair["prompt_id"], pair["seed"])] +
           [mib(pair.get(field)) for field in ("baseline_peak_allocated_bytes", "cache_peak_allocated_bytes", "peak_allocated_bytes_cache_minus_baseline", "baseline_peak_reserved_bytes", "cache_peak_reserved_bytes", "peak_reserved_bytes_cache_minus_baseline")]
           for pair in completed))
    supplement = summary["phase_supplement"]
    lines.extend(["## 补充阶段诊断（带观测，仅 object seed 0）", "",
                  "补测有效完成 %s / 2 个 arm，状态 `%s`。这是独立的 instrumented 完整请求，只覆盖 object seed 0 一对；不替代原六对无观测 E2E 比值，也不并入上面的任何中位数或样本量。未存在、未完成或失败的补测保持未知。" % (supplement["completed_arms"], supplement["status"]), ""])
    selection = supplement["selection"]
    if selection["source"]:
        lines.extend(["补测选择来源：%s；选择状态：`%s`；原因：%s。选定记录仍通过相同完整性检查，之前失败的尝试保留如下。" %
                      (link(selection["source"], output, "phase-selection.json"), selection["status"], selection["reason"]), ""])
    table(lines, ["arm", "补测状态", "完整 loop 同步墙钟秒", "reset CPU 函数 ms", "reset 含后同步 ms", "其中后同步 ms", "与同组 timed reference hash", "原始记录"],
          ([row["arm"], row["status"], row["denoising_loop_seconds"],
            "不适用" if row["reset_status"] == "not_applicable" else row["reset_call_wall_ms"],
            "不适用" if row["reset_status"] == "not_applicable" else row["reset_synchronized_wall_ms"],
            "不适用" if row["reset_status"] == "not_applicable" else row["reset_post_call_synchronize_ms"],
            row["reference_hash_equal"], link(row["source"], output)] for row in supplement["rows"]))
    lines.extend(["完整 loop 使用 progress_bar 上下文的同步墙钟边界，含去噪迭代、CFG、scheduler 和进度更新，排除文本编码、latent/timestep 准备、VAE、外层统计及入口 reset。边界同步本身属于观测干扰。reset 只计上一轮真实 warmup 填充过的 context 在下一次完整请求入口执行的 clear_buffers/重建函数，并单列返回后的同步时间；它不是步内缓存特征重算或迁移耗时，也不保证把 allocator reserved 显存归还驱动。hash 相同只作为同组带观测复跑的一致性证据，不能证明感知质量等价。", ""])
    if selection["previous_failed_attempts"]:
        lines.extend(["先前补测失败尝试（保留原文件，不计入选定阶段测量）：", ""])
        table(lines, ["原始记录", "原记录状态", "failure 类型/类别", "failure 消息"],
              ([link(attempt["source"], output), attempt["status"],
                (attempt["failure"] or {}).get("type") or (attempt["failure"] or {}).get("category"),
                (attempt["failure"] or {}).get("message")] for attempt in selection["previous_failed_attempts"]))
    for issue in selection["issues"]:
        lines.append("- 补测选择：" + issue)
    for attempt in selection["previous_failed_attempts"]:
        for issue in attempt["issues"]:
            lines.append("- 先前补测 %s：%s" % (attempt["source"], issue))
    for row in supplement["rows"]:
        for issue in row["issues"]:
            lines.append("- 补测 %s：%s" % (row["arm"], issue))
    lines.append("")
    lines.extend(["## 逐步缓存证据与同组输出一致性", "",
                  "每个 public summary entry 单独统计，scope 可能重叠，不跨 entry 相加。cached 与 CFG cached 计数来自映射成功的零起始步号；某 scope 报告缓存不等于整个 transformer 跳过。完整逐步列表、原始 summary 和 cache_refresh 可观测性记录保存在 v100-summary.json。", ""])
    rows, scopes, grouped = [], [], []
    for case in summary["cases"]:
        for repeat in case["repeats"]:
            rows.append([case_label(case), repeat["index"], repeat["diagnostic_forward_count"], repeat["diagnostic_step_count"], repeat["cache_mapping_status"], repeat["timed_diagnostic_sha256_equal"], (repeat.get("cache_refresh") or {}).get("status")])
            for scope in repeat["cache_scopes"]:
                scopes.append([case_label(case), repeat["index"], scope.get("summary_entry_index"), scope.get("status"), scope.get("reported_cached_step_count"), scope.get("reported_cfg_cached_step_count"), scope.get("reported_either_step_count")])
            timing_groups = repeat["public_cache_step_timing_groups"]
            if timing_groups["groups"] is not None:
                for key, label in (("reported_partial_reuse", "DBCache 报告部分复用"), ("no_cached_report", "未报告缓存")):
                    group = timing_groups["groups"][key]
                    grouped.append([case_label(case), repeat["index"], label, group["n_steps"], group["median_ms"], group["p95_descriptive_ms"], group["min_ms"], group["max_ms"]])
            elif case["arm"] == "cache-dit":
                grouped.append([case_label(case), repeat["index"], "分组 unknown", None, None, None, None, None])
    table(lines, ["arm", "repeat", "forward 数", "步数", "cache mapping", "timed/diagnostic SHA256", "refresh 状态"], rows)
    table(lines, ["arm", "repeat", "entry", "映射状态", "cached 步数", "CFG cached 步数", "两者并集步数"], scopes)
    lines.extend(["仅当所有 summary scope 都成功映射，且每一步的 cached/CFG 标签一致时，按“DBCache 报告部分复用”与“未报告缓存”分组描述整步 transformer CUDA 时间。单 scope 同样需通过完整性检查。步骤不是独立请求样本；p95 仅为这些步骤的描述统计。Fn/Bn 仍有计算，两组不能重命名为整个 transformer 跳过/完整重算；空组 n=0、耗时未知。", ""])
    table(lines, ["arm", "repeat", "public 标签组", "步数", "中位数 ms", "描述 p95 ms", "最小 ms", "最大 ms"], grouped)
    lines.extend(["[逐步时间及缓存证据表](v100-steps.csv) 列出每步的 transformer CUDA 时间、forward 索引，以及各 scope 的 reported_cached / reported_cfg_cached。scope 未报告缓存并不能证明整步完整重算，scope 报告缓存也不表示 Fn/Bn 块停止执行。因此，整步完整重算/复用路径的独立时间分布目前为 unknown，不能用上下文 reset 的兼容零值填充。", "",
                  "同组 timed/diagnostic SHA256 一致只检查诊断观测是否改变该组输出，不能证明缓存组与无缓存组感知质量等价。不一致和缺失分别保留，均不自动解释为质量通过。", "", "## 失败、OOM、缺失与原始证据", ""])
    table(lines, ["arm", "状态", "失败类别", "有效完成", "GPU 采样", "原始 profile", "进程日志"],
          ([case_label(case), case["status"], case.get("failure_category") or (case.get("failure") or {}).get("category"), "是" if case["valid_completion"] else "否", (case.get("telemetry") or {}).get("status", "missing"), link(case["profile_file"], output), link(case.get("raw_log"), output, "raw.log")] for case in summary["cases"]))
    for issue in summary["issues"]:
        lines.append("- " + issue)
    for case in summary["cases"]:
        case_issues = case["issues"] + ["repeat %s: %s" % (repeat["index"], issue) for repeat in case["repeats"] for issue in repeat["issues"]]
        if case.get("failure"):
            case_issues.append("failure: " + json.dumps(case["failure"], ensure_ascii=False))
        for issue in case_issues:
            lines.append("- %s：%s" % (case_label(case), issue))
    for pair in summary["pairs"]:
        for issue in pair["issues"]:
            lines.append("- 配对 %s/%s/seed=%s：%s" % (pair["stage"], pair["prompt_id"], pair["seed"], issue))
    lines.extend(["", "GPU 采样原始路径和统计保留于各 case.telemetry；采样覆盖模型加载、warmup、测量与诊断整段进程，不直接用整段平均值代表性能窗口。可通过 Unix 纳秒时间戳对齐各窗口。", ""])
    if review:
        lines.extend(["## 已记录的有限范围人工画面检查", "",
                      "以下仅转述 " + link(summary["manual_review"]["source"], output, "人工检查记录") + "，不根据 completed 状态、hash 一致或这些观察推导等质量、质量门槛通过或自动 passed。", "",
                      "检查方法：" + review["method_zh"], "", "检查范围：" + review["scope_zh"], "", "总体观察：", ""])
        lines.extend("- " + observation for observation in review["overall_observations_zh"])
        lines.append("")
        for pair in review["pairs"]:
            artifact = Path(summary["manual_review"]["source"]).parent / pair["artifact"]
            lines.extend(["%s / seed=%s；检查帧：%s；记录标签：`%s`；%s。" %
                          (pair["prompt_id"], pair["seed"], pair["frames_reviewed"], pair["obvious_additional_degradation"], link(artifact, output, "检查画面")), ""])
            lines.extend("- " + observation for observation in pair["observations_zh"])
            lines.append("")
        lines.extend(["quality_status=limited_manual_review_recorded；perceptual_quality_validated=false；research_gate_eligible=false。记录只覆盖上述人工检查范围。", ""])
    else:
        lines.extend(["## 人工质量评价（待填写）", "",
                      "评价人：____；日期：____；盲评/配对检视方法：____；物体外观、运动连续性、水面细节及明显伪影：____；结论及不确定性：____。当前 quality_status=manual_review_pending，perceptual_quality_validated=false，研究门槛判定始终为 false。", ""])
    media = manifest.get("media_export") or {}
    for item in media.get("contact_sheets", []):
        lines.append("- " + link(item.get("path"), output, "%s seed=%s 配对帧" % (item.get("prompt_id"), item.get("seed"))))
    for item in media.get("videos", []):
        lines.append("- " + link(item.get("video"), output, "%s/%s seed=%s %s 视频" % (item.get("stage"), item.get("prompt_id"), item.get("seed"), item.get("arm"))))
    lines.extend(["", "机器可读文件：[完整汇总](v100-summary.json)、[配对指标 CSV](v100-pairs.csv)。原始 profile、runner 记录、日志和数组均只读，未被修改。", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / ".cache/runs/v100/paired")
    parser.add_argument("--environment-dir", type=Path, default=ROOT / ".cache/runs/v100/environment")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/runs/v100")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    summary = build_summary(args.run_dir, args.environment_dir)
    write_report(summary, args.output)
    archived = archive_run(args.output, "wan-paired", args.results_root)
    print("已写入报告：" + archived["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
