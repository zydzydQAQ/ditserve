#!/usr/bin/env python3
"""Compare hash-bound native outputs with PSNR, SSIM, LPIPS and failure counts."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ditserve.evaluation.quality import evaluate_quality


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ssim(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim != 5 or reference.shape[-1] != 3:
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


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


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
        key = (row.get("dataset_sample_id"), row["spec"].get("seed"), row["spec"].get("prompt"))
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
    return array


def compare_runs(reference_run, candidate_run, *, with_lpips=True, lpips_device="cpu",
                 min_psnr=None, min_ssim=None):
    reference, candidate = output_index(reference_run), output_index(candidate_run)
    missing = sorted(set(reference) ^ set(candidate), key=repr)
    pairs, failures = [], []
    for key in sorted(set(reference) & set(candidate), key=repr):
        try:
            ref = checked_array(reference[key])
            other = checked_array(candidate[key])
            metrics = evaluate_quality(ref, other, with_lpips=with_lpips, lpips_device=lpips_device)
            score = ssim(ref, other)
            pairs.append({
                "dataset_sample_id": key[0], "seed": key[1], "prompt": key[2],
                "psnr_db": metrics["psnr_db"], "psnr_is_infinite": metrics["psnr_is_infinite"],
                "ssim": score, "lpips_mean": metrics["lpips_mean"],
                "temporal_delta_rmse": metrics["temporal_delta_rmse"],
                "exact_match": metrics["exact_match"], "quality_status": metrics["status"],
                "candidate_cfg_folded_steps": candidate[key]["outputs"][-1]["generation"].get("cfg_folded_steps", 0),
                "candidate_cfg_fold": candidate[key].get("cfg_fold"),
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
        "missing_identities": [{"dataset_sample_id": key[0], "seed": key[1]} for key in missing],
        "failures": failures,
        "metrics": {
            "psnr_db": summary(finite_psnr),
            "ssim": summary([row["ssim"] for row in pairs]),
            "lpips_mean": summary([row["lpips_mean"] for row in pairs if row["lpips_mean"] is not None]),
            "exact_match_count": sum(bool(row["exact_match"]) for row in pairs),
        },
        "pairs": pairs,
    }
    if min_psnr is not None:
        result["psnr_threshold"] = {"value": min_psnr, "passed": sum(value >= min_psnr for value in finite_psnr)}
    if min_ssim is not None:
        result["ssim_threshold"] = {"value": min_ssim, "passed": sum(row["ssim"] >= min_ssim for row in pairs)}
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
    args = parser.parse_args()
    reference = json.loads(args.reference_run.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_run.read_text(encoding="utf-8"))
    result = compare_runs(reference, candidate, with_lpips=not args.no_lpips,
                          lpips_device=args.lpips_device, min_psnr=args.min_psnr, min_ssim=args.min_ssim)
    result["source"] = {
        "reference_run_sha256": sha256(args.reference_run),
        "candidate_run_sha256": sha256(args.candidate_run),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "status": result["status"], "metrics": result["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
