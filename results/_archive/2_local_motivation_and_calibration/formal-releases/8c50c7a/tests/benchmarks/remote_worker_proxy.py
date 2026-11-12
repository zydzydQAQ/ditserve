#!/usr/bin/env python3
"""Node 0 TCP proxy script for remote GPU workers (`index >= 8`) in 16-GPU DitServe runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

WORKERS_ROOT = Path(__file__).resolve().parents[1] / "workers"
if str(WORKERS_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKERS_ROOT))

from ditserve.runtime.remote_bridge import (  # noqa: E402
    DOORBELL_STRUCT,
    HEADER_STRUCT,
    parse_endpoint,
    ping_daemon,
    recv_frame,
    run_proxy,
    send_frame,
)

__all__ = [
    "DOORBELL_STRUCT",
    "HEADER_STRUCT",
    "main",
    "parse_endpoint",
    "ping_daemon",
    "recv_frame",
    "run_proxy",
    "send_frame",
]


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--target-script", default=None)
    known, remaining = parser.parse_known_args(raw_argv)

    target_script = known.target_script or os.environ.get("DITSERVE_REMOTE_TARGET_SCRIPT", "")
    target_argv = raw_argv if target_script else remaining
    return run_proxy(
        backend=known.backend,
        config_path=known.config,
        target_script=target_script,
        target_argv=target_argv,
        endpoint=known.endpoint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
