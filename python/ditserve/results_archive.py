"""Lightweight results archiving wrapper and CLI entry point for DitServe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _rel in ("workers", "python"):
    _d = str(_REPO_ROOT / _rel)
    if _d not in sys.path:
        sys.path.insert(0, _d)

import ditserve as _ditserve_pkg  # noqa: E402

for _pkg_dir in (_REPO_ROOT / "workers" / "ditserve", Path(__file__).resolve().parent):
    if _pkg_dir.is_dir() and hasattr(_ditserve_pkg, "__path__") and str(_pkg_dir) not in _ditserve_pkg.__path__:
        _ditserve_pkg.__path__.append(str(_pkg_dir))

from ditserve.results import (  # noqa: E402, F401
    APPROXIMATE_OPTION_KEYS, APPROXIMATE_POLICY_TOKENS, TIER1_LOSSLESS_GATE_NAME,
    UNVERIFIED, VERIFIED_NUMERIC, VERIFIED_PERFORMANCE, VERIFIED_SEMANTIC,
    _summary_report, _summary_run, apply_tier1_qualification_gate, archive_run,
    compute_on_time_goodput_rps, evaluate_tier1_lossless_qualification,
    finalize, has_quality_certificate, has_real_gpu_execution,
    is_lossless_exact_policy, load, verify,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="Path to results-work/<experiment>/<run_id>")
    parser.add_argument("--output", type=Path, default=None, help="Path to results/<experiment>/<run_id>")
    parser.add_argument("--qualification", type=Path, default=None, help="Optional explicit qualification JSON")
    parser.add_argument("--prune", action="store_true", default=False)
    args = parser.parse_args(argv)
    receipt = archive_run(
        args.work_dir, args.output, explicit_qualification=args.qualification, prune=args.prune,
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
