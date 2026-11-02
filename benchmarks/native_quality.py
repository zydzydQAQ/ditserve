#!/usr/bin/env python3
"""Compare hash-bound native outputs with PSNR, SSIM, LPIPS and failure counts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ditserve.evaluation.quality import evaluate_quality
from native_exact import validate_pair_identity, validate_run
from native_metrics import distribution, percentile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ssim(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim != 5 or reference.shape[0] != 1 or reference.shape[-1] != 3:
        raise ValueError("SSIM expects matching [1,T,H,W,3] videos")
    values = []
    for ref_frame, candidate_frame in zip(reference[0], candidate[0]):
        for channel in range(3):
            ref = ref_frame[..., channel].astype(np.float64)
            other = candidate_frame[..., channel].astype(np.float64)
            mu_ref = gaussian_filter(ref, 1.5, truncate=3.0)
            mu_other = gaussian_filter(other, 1.5, truncate=3.0)
            var_ref = gaussian_filter(ref * ref, 1.5, truncate=3.0) - mu_ref * mu_ref
            var_other = gaussian_filter(other * other, 1.5, truncate=3.0) - mu_other * mu_other
            covariance = gaussian_filter(ref * other, 1.5, truncate=3.0) - mu_ref * mu_other
            c1, c2 = 0.01**2, 0.03**2
            score = ((2 * mu_ref * mu_other + c1) * (2 * covariance + c2)
                     / ((mu_ref * mu_ref + mu_other * mu_other + c1) * (var_ref + var_other + c2)))
            values.append(float(np.mean(score)))
    return float(np.mean(values))


def summary(values):
    if not values:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values), "mean": float(np.mean(values)), "min": min(values),
        "p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "max": max(values),
    }


def output_index(run):
    rows = {}
    for row in run["requests"]:
        if row["kind"] != "video":
            continue
        key = (row.get("dataset_sample_id"), row["spec"].get("seed"), row["spec"].get("prompt"),
               tuple(row["spec"].get(k) for k in ("frames", "height", "width", "steps")))
        if key in rows:
            raise ValueError("duplicate video identity: " + repr(key))
        rows[key] = row
    return rows


def checked_array(row):
    if row["status"] != "completed" or not row.get("outputs"):
        raise ValueError("video output is incomplete")
    produced = row["outputs"][-1]
    artifact = produced["output"]
    path = Path(artifact["path"])
    if sha256(path) != artifact["sha256"]:
        raise ValueError("output file hash mismatch: " + str(path))
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != artifact["shape"] or str(array.dtype) != artifact["dtype"]:
        raise ValueError("output shape or dtype mismatch: " + str(path))
    if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != artifact["array_sha256"]:
        raise ValueError("output array hash mismatch: " + str(path))
    if not np.isfinite(array).all():
        raise ValueError("non-finite output: " + str(path))
    return array


def fold_count(row, expected=None):
    produced = row["outputs"][-1]
    generation = produced["generation"]
    count = generation.get("cfg_folded_steps", 0)
    if type(count) is not int or count < 0:
        raise ValueError("actual CFG fold count is missing or invalid")
    statistics = row.get("cfg_fold")
    if statistics is None:
        if count != 0 or generation.get("approximate") is not False:
            raise ValueError("approximate output is missing worker CFG-fold statistics")
    else:
        if statistics.get("folded_cfg_steps") != count:
            raise ValueError("generation and worker fold counts differ")
        full = statistics.get("full_cfg_steps")
        if type(full) is not int or full + count != row["spec"]["steps"]:
            raise ValueError("worker full and folded steps do not cover the request")
    history = produced.get("path_history")
    steps = row["spec"]["steps"]
    if not isinstance(history, list) or len(history) != steps:
        raise ValueError("complete solver-step history is required for fold accounting")
    if any(event.get("step_index") != index for index, event in enumerate(history)):
        raise ValueError("solver-step history is incomplete or out of order")
    folded_indices = [index for index, event in enumerate(history) if event.get("cfg_folded", False)]
    if len(folded_indices) != count:
        raise ValueError("step history and actual fold count differ")
    if statistics is not None and statistics.get("folded_step_indices") != folded_indices:
        raise ValueError("worker and step-history folded indices differ")
    if history[-1].get("cfg_folded"):
        raise ValueError("final solver step is not an exact anchor")
    if expected is not None and count != expected:
        raise ValueError(f"actual fold count {count} does not match required fixed count {expected}")
    return count


def paired_bootstrap(candidate, control, *, resamples=2000):
    if type(resamples) is not int or resamples < 100:
        raise ValueError("paired bootstrap requires at least 100 resamples")
    for result in (candidate, control):
        if result.get("status") != "completed" or not result.get("pairs"):
            raise ValueError("matched-fold comparison requires complete retained pairs")
    if (candidate.get("source", {}).get("reference_run_sha256")
            != control.get("source", {}).get("reference_run_sha256")):
        raise ValueError("candidate arms do not share the same exact reference run")
    def index(result):
        rows = {}
        for row in result["pairs"]:
            key = quality_pair_key(row)
            if key in rows:
                raise ValueError("duplicate quality pair")
            rows[key] = row
        return rows
    left, right = index(candidate), index(control)
    if left.keys() != right.keys():
        raise ValueError("candidate arms have different prompt or seed identities")
    keys = sorted(left, key=repr)
    mismatched = [key for key in keys
                  if left[key]["candidate_cfg_folded_steps"] != right[key]["candidate_cfg_folded_steps"]]
    if mismatched:
        raise ValueError("adaptive and fixed arms differ in actual fold counts; compare within each count stratum")
    rng = np.random.default_rng(20261102)
    metrics = {}
    for name in ("psnr_db", "ssim", "lpips_mean"):
        if name == "psnr_db":
            # JSON represents exact-output PSNR as null plus explicit flags.
            # Never subtract infinities or turn them into an arbitrary ceiling.
            # A flag without exact-match evidence remains missing evidence.
            def psnr_kind(row):
                value = row.get("psnr_db")
                if (value is None and row.get("psnr_is_infinite") is True
                        and row.get("exact_match") is True):
                    return "exact"
                if (type(value) in (int, float) and np.isfinite(value)
                        and not row.get("psnr_is_infinite", False)):
                    return "finite"
                return "invalid"
            kinds = [(psnr_kind(left[key]), psnr_kind(right[key])) for key in keys]
            if any("invalid" in pair for pair in kinds):
                metrics[name] = {"status": "unverified", "reason": "missing or inconsistent PSNR/exact-match evidence"}
                continue
            if any("exact" in pair for pair in kinds):
                metrics[name] = {
                    "status": "exact_relation", "mean_difference": None, "ci95": None,
                    "relations": {
                        "both_exact": kinds.count(("exact", "exact")),
                        "candidate_exact_control_finite": kinds.count(("exact", "finite")),
                        "candidate_finite_control_exact": kinds.count(("finite", "exact")),
                        "both_finite": kinds.count(("finite", "finite")),
                    },
                    "scope": "non-numeric PSNR relation; no mean delta or confidence interval involving infinity",
                }
                continue
        a = np.asarray([left[key][name] for key in keys], dtype=np.float64)
        b = np.asarray([right[key][name] for key in keys], dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            metrics[name] = {"status": "unverified", "reason": "non-finite or missing paired metric"}
            continue
        delta = a - b
        draws = delta[rng.integers(0, len(keys), size=(resamples, len(keys)))].mean(axis=1)
        metrics[name] = {
            "status": "measured", "mean_candidate": float(a.mean()), "mean_control": float(b.mean()),
            "mean_difference": float(delta.mean()),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        }
    zero_self_control = all(left[key]["candidate_cfg_folded_steps"] == 0
                            and left[key].get("exact_match") is True
                            and right[key].get("exact_match") is True for key in keys)
    return {
        "method": "paired prompt/seed bootstrap; candidate minus fixed; positive PSNR/SSIM and negative LPIPS favor candidate",
        "matched_pairs": len(keys), "actual_fold_count_per_pair": "identical", "resamples": resamples,
        "interpretation": ("zero-fold/no-compute-reduction self-control; not adaptive superiority"
                           if zero_self_control else "matched actual-fold quality evidence; not a superiority decision"),
        "zero_fold_exact_self_control": zero_self_control,
        "metrics": metrics,
    }


def quality_pair_key(row):
    return (row["dataset_sample_id"], row["seed"], row.get("prompt"), tuple(row.get("shape", [])))


def matched_fold_bootstrap(candidate, controls, *, resamples=2000):
    if not isinstance(controls, dict) or any(type(count) is not int or count < 0 for count in controls):
        raise ValueError("matched-fold controls must be indexed by nonnegative actual fold counts")
    counts = {row["candidate_cfg_folded_steps"] for row in candidate.get("pairs", [])}
    strata = {}
    matched = 0
    for count in sorted(counts):
        pairs = [row for row in candidate["pairs"] if row["candidate_cfg_folded_steps"] == count]
        if count not in controls:
            strata[str(count)] = {"status": "unverified", "reason": "no fixed-N control", "matched_pairs": 0}
            continue
        # A fixed-N run normally contains the entire test set. Select the same
        # prompt/seed/shape members, not only the candidate's fold-count stratum.
        # Missing members and duplicates still fail in paired_bootstrap.
        keys = {quality_pair_key(row) for row in pairs}
        control = controls[count]
        matched_control = [row for row in control.get("pairs", []) if quality_pair_key(row) in keys]
        result = paired_bootstrap(dict(candidate, pairs=pairs), dict(control, pairs=matched_control), resamples=resamples)
        strata[str(count)] = result
        matched += result["matched_pairs"]
    return {
        "status": "measured" if candidate.get("pairs") and matched == len(candidate["pairs"])
                  and all(metric.get("status") in ("measured", "exact_relation") for stratum in strata.values()
                          for metric in stratum.get("metrics", {}).values()) else "unverified",
        "scope": "pair adaptive and fixed arms only within identical per-video actual fold-count strata",
        "zero_fold_scope": "N=0 exact-output pairs are no-compute-reduction self-controls, never evidence of adaptive gain",
        "superiority_qualified": False,
        "matched_pairs": matched, "candidate_pairs": len(candidate.get("pairs", [])),
        "strata": strata,
    }


def compare_runs(reference_run, candidate_run, *, with_lpips=True, lpips_device="cpu",
                 min_psnr=None, min_ssim=None, max_lpips=None, expected_fold_count=None,
                 allow_legacy_runtime=False):
    for run in (reference_run, candidate_run):
        validate_run(run)
    if reference_run["source"] != candidate_run["source"]:
        raise ValueError("runtime source identity differs")
    for name, threshold in (("min_psnr", min_psnr), ("min_ssim", min_ssim), ("max_lpips", max_lpips)):
        if threshold is not None and not np.isfinite(threshold):
            raise ValueError(name + " must be finite")
    if max_lpips is not None and not with_lpips:
        raise ValueError("an LPIPS upper bound cannot be checked with --no-lpips")
    if expected_fold_count is not None and (type(expected_fold_count) is not int or expected_fold_count < 0):
        raise ValueError("expected_fold_count must be a nonnegative integer")
    reference, candidate = output_index(reference_run), output_index(candidate_run)
    missing = sorted(set(reference) ^ set(candidate), key=repr)
    pairs, failures = [], []
    for key in sorted(set(reference) & set(candidate), key=repr):
        try:
            identity = validate_pair_identity(reference[key], candidate[key], allow_approximate=True,
                                              allow_legacy_runtime=allow_legacy_runtime)
            if reference[key]["outputs"][-1]["generation"].get("approximate") is not False:
                raise ValueError("quality reference must be an exact output")
            if reference[key].get("forms") != candidate[key].get("forms") or len(reference[key].get("forms", [])) != 1:
                raise ValueError("controlled quality comparison requires the same single parallel form")
            actual_folds = fold_count(candidate[key], expected_fold_count)
            ref = checked_array(reference[key])
            other = checked_array(candidate[key])
            metrics = evaluate_quality(ref, other, with_lpips=with_lpips, lpips_device=lpips_device)
            if with_lpips and metrics["lpips_mean"] is None:
                raise ValueError("LPIPS is required but unavailable")
            score = ssim(ref, other)
            pairs.append({
                "dataset_sample_id": key[0], "seed": key[1], "prompt": key[2],
                "shape": list(key[3]),
                "psnr_db": metrics["psnr_db"], "psnr_is_infinite": metrics["psnr_is_infinite"],
                "ssim": score, "lpips_mean": metrics["lpips_mean"],
                "temporal_delta_rmse": metrics["temporal_delta_rmse"],
                "exact_match": metrics["exact_match"], "quality_status": metrics["status"],
                "candidate_cfg_folded_steps": actual_folds,
                "candidate_cfg_fold": candidate[key].get("cfg_fold"),
                **identity,
            })
        except Exception as error:
            failures.append({"dataset_sample_id": key[0], "seed": key[1], "error": str(error)})
    finite_psnr = [row["psnr_db"] for row in pairs if row["psnr_db"] is not None]
    result = {
        "schema": "ditserve.native-quality.v1",
        "status": "completed" if pairs and not missing and not failures else "incomplete",
        "scope": "reference-relative decoded-video comparison; not an absolute quality guarantee",
        "reference": {"experiment": reference_run["experiment"], "run_id": reference_run["run_id"]},
        "candidate": {"experiment": candidate_run["experiment"], "run_id": candidate_run["run_id"]},
        "matched_pairs": len(pairs),
        "identity_qualification": "verified" if pairs and all(row["runtime_verified"] for row in pairs) else "unverified",
        "weights_verified": bool(pairs) and all(row["weights_verified"] for row in pairs),
        "missing_identities": [{"dataset_sample_id": key[0], "seed": key[1]} for key in missing],
        "failures": failures,
        "metrics": {
            "psnr_db": summary(finite_psnr),
            "ssim": summary([row["ssim"] for row in pairs]),
            "lpips_mean": summary([row["lpips_mean"] for row in pairs if row["lpips_mean"] is not None]),
            "exact_match_count": sum(bool(row["exact_match"]) for row in pairs),
            "candidate_cfg_folded_steps": summary([row["candidate_cfg_folded_steps"] for row in pairs]),
        },
        "fold_count_histogram": {str(count): sum(row["candidate_cfg_folded_steps"] == count for row in pairs)
                                 for count in sorted({row["candidate_cfg_folded_steps"] for row in pairs})},
        "pairs": pairs,
    }
    if min_psnr is not None:
        result["psnr_threshold"] = {"value": min_psnr, "passed": sum(
            row["psnr_is_infinite"] or row["psnr_db"] is not None and row["psnr_db"] >= min_psnr for row in pairs)}
    if min_ssim is not None:
        result["ssim_threshold"] = {"value": min_ssim, "passed": sum(row["ssim"] >= min_ssim for row in pairs)}
    if max_lpips is not None:
        result["lpips_threshold"] = {"value": max_lpips, "passed": sum(
            row["lpips_mean"] is not None and row["lpips_mean"] <= max_lpips for row in pairs)}
    if expected_fold_count is not None:
        result["fold_count_requirement"] = {"value": expected_fold_count, "matched": len(pairs)}
    configured = [result[key] for key in ("psnr_threshold", "ssim_threshold", "lpips_threshold") if key in result]
    passed = result["status"] == "completed" and all(gate["passed"] == len(pairs) for gate in configured)
    result["numeric_qualification"] = "unverified" if not configured else "passed" if passed else "failed"
    result["semantic_qualification"] = "unverified"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--lpips-device", default="cpu")
    parser.add_argument("--min-psnr", type=float)
    parser.add_argument("--min-ssim", type=float)
    parser.add_argument("--max-lpips", type=float)
    parser.add_argument("--expected-fold-count", type=int)
    parser.add_argument("--control-quality", type=Path, action="append",
                        help="fixed-N quality JSON with retained pairs and fold_count_requirement; repeat for every available N")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--allow-legacy-runtime", action="store_true", help="diagnostic only when historical runtime metadata is absent")
    parser.add_argument("--keep-pairs", action="store_true", help="retain per-video prompts and quality metrics; default keeps aggregate and failure evidence")
    args = parser.parse_args()
    reference_raw, candidate_raw = args.reference_run.read_bytes(), args.candidate_run.read_bytes()
    reference, candidate = json.loads(reference_raw), json.loads(candidate_raw)
    result = compare_runs(reference, candidate, with_lpips=not args.no_lpips,
                          lpips_device=args.lpips_device, min_psnr=args.min_psnr, min_ssim=args.min_ssim,
                          max_lpips=args.max_lpips, expected_fold_count=args.expected_fold_count,
                          allow_legacy_runtime=args.allow_legacy_runtime)
    result["source"] = {
        "reference_run_sha256": hashlib.sha256(reference_raw).hexdigest(),
        "candidate_run_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "runtime_source": reference.get("source"),
    }
    if args.control_quality:
        controls = {}
        for path in args.control_quality:
            control = json.loads(path.read_text(encoding="utf-8"))
            requirement = control.get("fold_count_requirement", {})
            count = requirement.get("value")
            if type(count) is not int or count < 0 or requirement.get("matched") != control.get("matched_pairs"):
                raise ValueError("fixed-N control lacks a complete actual fold-count requirement: " + str(path))
            if count in controls:
                raise ValueError("duplicate fixed-N control: " + str(count))
            controls[count] = control
        result["matched_fold_comparison"] = matched_fold_bootstrap(result, controls, resamples=args.bootstrap_samples)
    result["preemption_stop_ms"] = {name: distribution([sample["stop_ms"] for row in run["requests"]
        for sample in row.get("preemption_stop_samples", [])]) for name, run in (("reference", reference), ("candidate", candidate))}
    if not args.keep_pairs:
        result["threshold_failures"] = [{key: value for key, value in row.items() if key != "prompt"}
            for row in result.pop("pairs") if
            (args.min_psnr is not None and not row["psnr_is_infinite"] and
             (row["psnr_db"] is None or row["psnr_db"] < args.min_psnr)) or
            (args.min_ssim is not None and row["ssim"] < args.min_ssim) or
            (args.max_lpips is not None and (row["lpips_mean"] is None or row["lpips_mean"] > args.max_lpips)) or
            (args.expected_fold_count is not None and row["candidate_cfg_folded_steps"] != args.expected_fold_count)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "status": result["status"], "metrics": result["metrics"]}, ensure_ascii=False, indent=2))
    if (result["status"] != "completed" or result["numeric_qualification"] == "failed"
            or result.get("matched_fold_comparison", {}).get("status") == "unverified"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
