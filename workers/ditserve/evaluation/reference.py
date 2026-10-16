"""Reference comparisons with optional perceptual evaluation."""

import json
from pathlib import Path


def evaluate(request):
    if request.get("evaluator"):
        from . import evaluate as evaluate_external
        return evaluate_external(request)
    if str(request["reference"]).endswith(".txt") and str(request["output"]).endswith(".txt"):
        exact = Path(request["reference"]).read_bytes() == Path(request["output"]).read_bytes()
        return dict(status="ok", exact=exact, allclose=exact, passed=exact,
                    scope="exact_text_only", perceptual_quality="unverified")
    if str(request["reference"]).endswith(".json") and str(request["output"]).endswith(".json"):
        with Path(request["reference"]).open() as stream:
            left = json.load(stream)
        with Path(request["output"]).open() as stream:
            right = json.load(stream)
        return dict(status="ok", exact=left == right, allclose=left == right, passed=left == right,
                    measured=False, perceptual_quality="fixture_only")
    import numpy as np
    left = np.load(request["reference"], allow_pickle=False)
    right = np.load(request["output"], allow_pickle=False)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return {"status": "failed", "passed": False, "finite": bool(np.isfinite(left).all() and np.isfinite(right).all()),
                "shape_equal": left.shape == right.shape}
    delta = left.astype(np.float64) - right.astype(np.float64)
    result = {"status": "ok", "shape_equal": True, "finite": True,
              "exact": bool(np.array_equal(left, right)),
              "max_abs": float(np.abs(delta).max()) if delta.size else 0.,
              "rmse": float(np.sqrt(np.square(delta).mean())) if delta.size else 0.,
              "allclose": bool(np.allclose(left, right, atol=request.get("atol", 1e-6),
                                           rtol=request.get("rtol", 1e-5))),
              "perceptual_quality": "unverified"}
    result.update(passed=result["allclose"], scope="numerical_reference_only_not_perceptual_or_blind_quality")
    if request.get("lpips"):
        from .quality import evaluate_quality
        # The pinned evaluator expects [batch, frames, height, width, channels].
        if left.ndim == 4:
            left, right = left[:, None], right[:, None]
        result["quality"] = evaluate_quality(left, right, with_lpips=True,
                                            lpips_device=request.get("device", "cpu"))
        result["perceptual_quality"] = result["quality"]["status"]
    return result
