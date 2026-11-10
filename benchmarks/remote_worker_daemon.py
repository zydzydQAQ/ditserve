#!/usr/bin/env python3
"""Node 1 TCP daemon for spawning and managing remote GPU workers (`index 8..15`) in 16-GPU runs."""

from __future__ import annotations

from pathlib import Path
import sys

WORKERS_ROOT = Path(__file__).resolve().parents[1] / "workers"
if str(WORKERS_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKERS_ROOT))

from ditserve.runtime.remote_bridge import (  # noqa: E402
    RemoteWorkerDaemon,
    main_daemon,
    ping_daemon,
    query_remote_occupancy,
    release_daemon,
)

__all__ = [
    "RemoteWorkerDaemon",
    "main_daemon",
    "ping_daemon",
    "query_remote_occupancy",
    "release_daemon",
]


if __name__ == "__main__":
    raise SystemExit(main_daemon())
