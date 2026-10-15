"""Explicit cross-attention reference gate, separate from scheduling parity."""

from pathlib import Path

from .common import file_hash, positive, read_json, seal, verify_seal
from .evaluation import generation_identity, resolve_artifact_path, _run_path


def check_backend_quality(references, reference_run_path, output_dir, epsilon=.05, budget=None,
                          candidate_identity=None):
    import numpy as np
    from ..gpu.quality import evaluate_quality

    positive(epsilon, "epsilon", allow_zero=True)
    path = _run_path(reference_run_path)
    baseline = verify_seal(read_json(path))
    if (baseline.get("backend") != "wan" or baseline.get("measured") is not True
            or baseline.get("status") != "completed"):
        raise ValueError("cross-attention gate requires a completed measured Wan reference run")
    identity = baseline.get("execution_identity", {})
    if identity.get("attention", {}).get("sdpa_backend") != "MATH":
        raise ValueError("cross-attention reference must use the frozen MATH backend")
    from .optimization import IDENTITY_FIELDS
    if not candidate_identity or any(candidate_identity.get(key) != identity.get(key) for key in IDENTITY_FIELDS):
        raise ValueError("cross-attention reference must match the candidate GPU and numerical recipe")
    rows = {}
    for rid, candidate in references.items():
        if budget is not None:
            budget.check()
        reference = baseline.get("references", {}).get(rid)
        if reference is None or generation_identity(candidate["generation"]) != generation_identity(reference["generation"]):
            raise ValueError("cross-attention reference request mismatch: " + rid)
        actual = resolve_artifact_path(candidate["output_path"], candidate["output_sha256"], Path(output_dir))
        expected = resolve_artifact_path(reference["output_path"], reference["output_sha256"], path.parent)
        metrics = evaluate_quality(np.load(expected, allow_pickle=False), np.load(actual, allow_pickle=False),
                                   with_lpips=True, lpips_device="cpu")
        lpips = metrics.get("lpips_mean")
        passed = (metrics.get("status") == "ok" and metrics.get("lpips_status") == "ok"
                  and metrics.get("finite") is True and metrics.get("range_valid") is True
                  and isinstance(lpips, (int, float)) and lpips <= epsilon)
        rows[rid] = dict(status="passed" if passed else "failed", metrics=metrics,
            candidate_sha256=candidate["output_sha256"], reference_sha256=reference["output_sha256"])
    return seal(dict(schema_version="ditserve.benchmark.backend_quality.v1",
        status="passed" if rows and all(row["status"] == "passed" for row in rows.values()) else "failed",
        reference_run_path=str(path), reference_run_sha256=baseline["sha256"],
        reference_run_file_sha256=file_hash(path), epsilon=epsilon, requests=rows,
        scope="held-out same-input cross-attention mean LPIPS; not absolute video quality or VBench"))
