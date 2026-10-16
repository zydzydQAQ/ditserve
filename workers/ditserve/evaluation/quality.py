"""Pinned reference-relative LPIPS evaluation; missing evidence stays unknown."""
from functools import lru_cache
import math
import numpy as np


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
