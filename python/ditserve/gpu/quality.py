"""Per-video quality and development-only calibration for PreViS pilots.

LPIPS is a real, lazily loaded AlexNet LPIPS model. Missing dependencies,
weights, or failed evaluation produce unknown quality, never a passing proxy.
All reported confidence intervals resample independent trace IDs, not frames.
"""

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EPSILON_GRID = (0.01, 0.03, 0.05, 0.10)
CALIBRATION_SCHEMA = "previs-calibration.v1"
_SHAPE_FIELDS = ("height", "width", "frames", "steps")
_ACTION_FIELDS = ("mode", "threshold", "max_continuous")


def _finite_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


@lru_cache(maxsize=4)
def _lpips_model(device):
    import lpips

    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    model.requires_grad_(False)
    return model


def _lpips_frames(reference, candidate, device):
    import torch

    model = _lpips_model(str(device))
    values = []
    with torch.inference_mode():
        for ref_frame, candidate_frame in zip(reference[0], candidate[0]):
            tensors = []
            for frame in (ref_frame, candidate_frame):
                tensor = torch.from_numpy(np.array(frame, dtype=np.float32, copy=True))
                tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                tensors.append(tensor.mul(2).sub(1))
            value = float(model(*tensors).detach().cpu().reshape(-1)[0])
            if not math.isfinite(value) or value < 0:
                raise ValueError("LPIPS returned a non-finite or negative distance")
            values.append(value)
    return values


def evaluate_quality(reference, candidate, with_lpips=True, lpips_device="cpu"):
    """Evaluate [1,T,H,W,3] floating videos in [0,1], without resizing.

    PSNR uses unit peak and global pixel MSE. Exact PSNR is represented by
    ``psnr_db=None, psnr_is_infinite=True`` so JSON never contains Infinity.
    Temporal delta RMSE compares consecutive RGB frame differences to the
    reference; it is a diagnostic, not a calibrated perceptual guarantee.
    ``status=ok`` requires successful real LPIPS evaluation for every frame.
    """
    result = {"psnr_db": None, "psnr_is_infinite": False,
              "lpips_mean": None, "lpips_p95": None, "lpips_per_frame": [],
              "temporal_delta_rmse": None, "exact_match": None,
              "status": "invalid", "finite": False, "range_valid": False,
              "shape": None, "frame_count": 0, "lpips_status": "not_evaluated",
              "lpips_network": "alex", "lpips_device": str(lpips_device),
              "temporal_semantics": "rmse_of_reference_relative_consecutive_rgb_differences",
              "quality_scope": "reference_relative_lpips_not_vbench_or_quality_guarantee"}
    if not isinstance(reference, np.ndarray) or not isinstance(candidate, np.ndarray):
        result["error"] = "reference and candidate must be numpy arrays"
        return result
    if reference.shape != candidate.shape:
        result["error"] = "reference and candidate shapes differ"
        return result
    if reference.ndim != 5 or reference.shape[0] != 1 or reference.shape[-1] != 3 or any(size < 1 for size in reference.shape):
        result["error"] = "expected nonempty [1,T,H,W,3] videos"
        return result
    result["shape"] = list(reference.shape)
    result["frame_count"] = int(reference.shape[1])
    if not np.issubdtype(reference.dtype, np.floating) or not np.issubdtype(candidate.dtype, np.floating):
        result["error"] = "videos must have floating dtype and range [0,1]"
        return result
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        result["error"] = "videos contain non-finite values"
        return result
    result["finite"] = True
    if reference.min() < 0 or reference.max() > 1 or candidate.min() < 0 or candidate.max() > 1:
        result["error"] = "videos must be in [0,1]; inputs are never clipped"
        return result
    result["range_valid"] = True
    squared_error = temporal_error = 0.0
    previous = None
    for ref_frame, candidate_frame in zip(reference[0], candidate[0]):
        difference = candidate_frame.astype(np.float64) - ref_frame.astype(np.float64)
        squared_error += float(np.sum(difference * difference))
        if previous is not None:
            change = difference - previous
            temporal_error += float(np.sum(change * change))
        previous = difference
    mse = squared_error / reference.size
    result["exact_match"] = mse == 0
    result["psnr_is_infinite"] = mse == 0
    result["psnr_db"] = None if mse == 0 else -10.0 * math.log10(mse)
    if reference.shape[1] > 1:
        denominator = (reference.shape[1] - 1) * reference[0, 0].size
        result["temporal_delta_rmse"] = math.sqrt(temporal_error / denominator)
    result["status"] = "unknown"
    if not with_lpips:
        result["lpips_status"] = "disabled"
        return result
    try:
        values = _lpips_frames(reference, candidate, lpips_device)
        if len(values) != reference.shape[1] or not all(_finite_number(value) and value >= 0 for value in values):
            raise ValueError("LPIPS did not produce a valid value for every frame")
    except Exception as error:
        result["lpips_status"] = "unavailable"
        result["lpips_error"] = "%s: %s" % (type(error).__name__, str(error))
        return result
    result.update({"status": "ok", "lpips_status": "ok", "lpips_per_frame": values,
                   "lpips_mean": float(np.mean(values)), "lpips_p95": float(np.percentile(values, 95))})
    return result


def quality_pass(metrics, epsilon):
    """A known mean-LPIPS threshold check, not a video-quality guarantee."""
    if not _finite_number(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if not isinstance(metrics, dict) or metrics.get("status") != "ok" or metrics.get("finite") is not True:
        return False
    if metrics.get("range_valid", True) is not True:
        return False
    for name in ("psnr_db", "temporal_delta_rmse"):
        if metrics.get(name) is not None and not _finite_number(metrics[name]):
            return False
    for name in ("lpips_mean", "lpips_p95"):
        if not _finite_number(metrics.get(name)) or metrics[name] < 0:
            return False
    return bool(metrics["lpips_mean"] <= epsilon)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(record, include_action=True):
    """No shape interpolation, device fallback, or omitted config defaults."""
    if not isinstance(record.get("device"), str) or not record["device"]:
        raise ValueError("a nonempty device identity is required")
    shape = record.get("shape")
    if not isinstance(shape, dict) or any(not isinstance(shape.get(key), int) or isinstance(shape.get(key), bool) or shape[key] < 1 for key in _SHAPE_FIELDS):
        raise ValueError("shape requires positive integer height,width,frames,steps")
    if "progress_bin" not in record or record["progress_bin"] is None:
        raise ValueError("progress_bin is required")
    config = record.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    identity = {"device": record["device"], "shape": {key: shape[key] for key in _SHAPE_FIELDS},
                "progress_bin": record["progress_bin"], "config": config}
    if include_action:
        if not isinstance(record.get("mode"), str) or not record["mode"]:
            raise ValueError("mode is required")
        threshold, continuous = record.get("threshold"), record.get("max_continuous")
        if not _finite_number(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        if not isinstance(continuous, int) or isinstance(continuous, bool) or continuous < 0:
            raise ValueError("max_continuous must be a nonnegative integer")
        identity.update({"mode": record["mode"], "threshold": float(threshold), "max_continuous": continuous})
    # Validate nested config/progress and detach caller-owned dictionaries.
    return json.loads(_canonical(identity))


def build_calibration(records):
    """Freeze empirical action eligibility using only records with split='dev'.

    An action passes epsilon only if every development observation has known
    passing quality. No confidence-bound or out-of-distribution guarantee is
    implied. Put model revision/dtype/cache implementation in ``config``;
    the complete object participates in subsequent exact matching.
    """
    groups = defaultdict(list)
    source = []
    excluded = 0
    for record in records:
        if record.get("split") != "dev":
            excluded += 1
            continue
        identity = _identity(record)
        groups[_canonical(identity)].append(record)
        source.append({"identity": identity, "quality": record.get("quality"),
                       "elapsed_ms": record.get("elapsed_ms"), "trace_id": record.get("trace_id"),
                       "id": record.get("id")})
    entries = []
    for key, group in sorted(groups.items()):
        known = [row["quality"]["lpips_mean"] for row in group if quality_pass(row.get("quality"), float("1e300"))]
        elapsed = [float(row["elapsed_ms"]) for row in group if _finite_number(row.get("elapsed_ms")) and row["elapsed_ms"] >= 0]
        entries.append({"identity": json.loads(key), "samples": len(group), "known_quality_samples": len(known),
                        "max_lpips_mean": max(known) if known else None,
                        "median_elapsed_ms": float(np.median(elapsed)) if elapsed else None,
                        "eligible_epsilons": [value for value in EPSILON_GRID if all(quality_pass(row.get("quality"), value) for row in group)]})
    source_digest = hashlib.sha256(_canonical(sorted(source, key=_canonical)).encode("utf-8")).hexdigest()
    return {"schema_version": CALIBRATION_SCHEMA, "split": "dev", "epsilon_grid": list(EPSILON_GRID),
            "development_records": len(source), "excluded_nondev_records": excluded,
            "source_sha256": source_digest, "entries": entries,
            "scope": "empirical_development_observations_only_not_a_quality_guarantee",
            "fallback": "no_cache"}


def freeze_calibration(records, path):
    """Write a new calibration file; an existing frozen file is never replaced."""
    table = build_calibration(records)
    encoded = json.dumps(table, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return table


def select_calibrated_action(table, request, epsilon):
    """Approve one proposed, exactly matched action or return no_cache.

    Call separately for each candidate action, then choose using development
    timing only. Unseen epsilon values, missing metadata, and unknown quality
    cannot silently borrow calibration from another context.
    """
    fallback = {"mode": "no_cache", "threshold": 0.0, "max_continuous": 0,
                "status": "fallback", "reason": "no_exact_calibration_match"}
    if table.get("schema_version") != CALIBRATION_SCHEMA or table.get("split") != "dev":
        return dict(fallback, reason="invalid_calibration_schema")
    if not _finite_number(epsilon) or float(epsilon) not in EPSILON_GRID:
        return dict(fallback, reason="epsilon_not_calibrated")
    try:
        identity = _identity(request)
    except (TypeError, ValueError):
        return dict(fallback, reason="incomplete_or_invalid_action_identity")
    matches = [entry for entry in table.get("entries", []) if entry.get("identity") == identity]
    if len(matches) != 1:
        return fallback
    entry = matches[0]
    if entry.get("samples", 0) < 1 or entry.get("known_quality_samples") != entry.get("samples") or epsilon not in entry.get("eligible_epsilons", []):
        return dict(fallback, reason="development_quality_not_eligible")
    return {**{key: identity[key] for key in _ACTION_FIELDS}, "status": "selected",
            "identity": identity, "calibration_source_sha256": table.get("source_sha256"),
            "development_samples": entry["samples"], "median_elapsed_ms": entry.get("median_elapsed_ms"),
            "scope": "empirical_calibration_not_a_quality_guarantee"}


def summarize_groups(records):
    """Summarize repeated videos without treating their frames as replicates."""
    groups = defaultdict(list)
    for record in records:
        identity = _identity(record)
        identity["split"] = record.get("split", "unknown")
        groups[_canonical(identity)].append(record)
    result = []
    for key, group in sorted(groups.items()):
        known = [row["quality"] for row in group if quality_pass(row.get("quality"), 1e300)]
        times = [float(row["elapsed_ms"]) for row in group if _finite_number(row.get("elapsed_ms")) and row["elapsed_ms"] >= 0]
        result.append({"identity": json.loads(key), "records": len(group),
                       "known_quality_records": len(known), "unknown_quality_records": len(group) - len(known),
                       "trace_count": len({row["trace_id"] for row in group if row.get("trace_id") is not None}),
                       "lpips_mean": float(np.mean([item["lpips_mean"] for item in known])) if known else None,
                       "lpips_max": max(item["lpips_mean"] for item in known) if known else None,
                       "median_elapsed_ms": float(np.median(times)) if times else None,
                       "elapsed_samples": len(times),
                       "passes_by_epsilon": {str(value): sum(quality_pass(row.get("quality"), value) for row in group) for value in EPSILON_GRID},
                       "denominator": "all_records_in_group", "independent_unit": "trace"})
    return result


def quality_frontier(records):
    """Observed latency/LPIPS Pareto points, grouped by identical context.

    Points summarize all repetitions of an action. Any unknown quality or
    missing latency excludes that action, avoiding successful-only frontiers.
    The empirical mean LPIPS frontier is not an epsilon guarantee.
    """
    groups = defaultdict(list)
    for summary in summarize_groups(records):
        context = {key: value for key, value in summary["identity"].items() if key not in _ACTION_FIELDS}
        groups[_canonical(context)].append(summary)
    result = []
    for key, summaries in sorted(groups.items()):
        valid = [item for item in summaries if item["records"] == item["known_quality_records"] == item["elapsed_samples"]]
        points = []
        for item in valid:
            dominated = any(other["median_elapsed_ms"] <= item["median_elapsed_ms"] and other["lpips_mean"] <= item["lpips_mean"] and
                            (other["median_elapsed_ms"] < item["median_elapsed_ms"] or other["lpips_mean"] < item["lpips_mean"])
                            for other in valid)
            if not dominated:
                points.append(item)
        result.append({"context": json.loads(key), "points": points,
                       "excluded_incomplete_groups": len(summaries) - len(valid),
                       "scope": "observed_latency_quality_frontier_not_serving_goodput_or_quality_guarantee"})
    return result


def paired_difference_bootstrap(records, baseline, candidate, metric="elapsed_ms", group_field="policy",
                                trace_field="trace_id", n_resamples=2000, seed=0, confidence=0.95):
    """Cluster bootstrap of candidate-minus-baseline mean across paired traces.

    Multiple observations within a trace/strategy are averaged first. The
    caller must supply one comparable experimental stratum; frames are never
    accepted as independent samples. A dotted metric (e.g. quality.lpips_mean)
    is supported. Incomplete trace pairs are excluded and counted explicitly.
    """
    if baseline == candidate or not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise ValueError("distinct strategies and positive integer n_resamples are required")
    if not _finite_number(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    observations = {baseline: defaultdict(list), candidate: defaultdict(list)}
    excluded = 0
    invalid_traces = set()
    for record in records:
        group = record.get(group_field)
        if group not in observations:
            continue
        value = record
        for part in metric.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        trace = record.get(trace_field)
        if trace is None or not _finite_number(value):
            excluded += 1
            if trace is not None:
                invalid_traces.add(_canonical(trace))
            continue
        observations[group][_canonical(trace)].append(float(value))
    left, right = observations[baseline], observations[candidate]
    paired = sorted((set(left) & set(right)) - invalid_traces)
    differences = [float(np.mean(right[trace]) - np.mean(left[trace])) for trace in paired]
    result = {"status": "ok" if len(paired) >= 2 else "insufficient_evidence", "baseline": baseline, "candidate": candidate,
              "metric": metric, "difference_direction": "candidate_minus_baseline", "independent_unit": "trace",
              "paired_traces": len(paired), "unpaired_traces": len(set(left) ^ set(right)),
              "excluded_records": excluded, "invalid_traces": len(invalid_traces),
              "estimate": float(np.mean(differences)) if differences else None,
              "ci_low": None, "ci_high": None, "confidence": confidence,
              "n_resamples": n_resamples, "seed": seed, "trace_differences": differences,
              "scope": "paired_trace_cluster_bootstrap_not_a_quality_guarantee"}
    if len(paired) >= 2:
        rng = np.random.default_rng(seed)
        values = np.asarray(differences, dtype=np.float64)
        # Bounded memory even when the number of traces is large.
        samples = [float(np.mean(values[rng.integers(0, len(values), size=len(values))])) for _ in range(n_resamples)]
        alpha = (1 - confidence) / 2
        result["ci_low"], result["ci_high"] = [float(value) for value in np.quantile(samples, [alpha, 1 - alpha])]
    return result
