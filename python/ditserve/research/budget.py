"""Exclusive, persistent GPU-seconds ledger for the 8 GPU-hour pilot.

The parent owns this ledger from before loading a model until every worker has
exited. Charging includes warmup, failed work, evaluation while holding GPUs,
and cleanup. Reservations estimate a *complete paired unit*, not one arm.
"""

import contextlib
import fcntl
import os
from pathlib import Path
import time
import uuid

from .common import atomic_json, finite, read_json, process_group_alive


STAGE_LIMITS = {"correctness": 1800., "probe": 5400., "calibrate": 9000.,
                "sp": 9000., "reserve": 3600.}
MAX_GPU_SECONDS = 28800.


class BudgetStop(RuntimeError):
    pass


class WorkerCleanupError(RuntimeError):
    """Some GPU-owning processes may still be alive; keep charging on recovery."""
    pass


class ResearchBudget:
    def __init__(self, path, limit_gpu_seconds=MAX_GPU_SECONDS, clock=time.time,
                 monotonic=time.monotonic):
        self.path = Path(path)
        self.limit = finite(limit_gpu_seconds, "limit_gpu_seconds", positive=True)
        if self.limit > MAX_GPU_SECONDS:
            raise ValueError("research pilot is authorized for at most 28,800 GPU-seconds")
        self.clock, self.monotonic = clock, monotonic
        self._lock = self._data = self._started = None
        self._outcome = None

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._close()
            raise RuntimeError("research budget already in use")
        try:
            self._data = read_json(self.path) if self.path.exists() else {
                "schema_version": "ditserve.research.budget.v1", "limit_gpu_seconds": self.limit,
                "used_gpu_seconds": 0., "stage_used_gpu_seconds": dict.fromkeys(STAGE_LIMITS, 0.),
                "stage_limits_gpu_seconds": STAGE_LIMITS, "active": None, "sessions": []}
            if (self._data.get("schema_version") != "ditserve.research.budget.v1"
                    or self._data.get("limit_gpu_seconds") != self.limit
                    or self._data.get("stage_limits_gpu_seconds") != STAGE_LIMITS):
                raise ValueError("existing research ledger identity/limits cannot be changed")
            finite(self._data["used_gpu_seconds"], "used_gpu_seconds")
            for stage in STAGE_LIMITS:
                finite(self._data["stage_used_gpu_seconds"][stage], "stage usage")
            if self._data["active"]:
                active = self._data["active"]
                if active.get("workers_expected") and process_group_alive(active.get("worker_pgid")):
                    raise WorkerCleanupError("prior GPU worker cleanup is unconfirmed; existing reservation stays active")
                # A crash cannot become a free GPU session. Recovery charges
                # until now; clock reversal consumes all remaining allowance.
                wall = self.clock() - active["started_at"]
                wall = wall if wall >= 0 else self.limit / active["gpu_count"]
                self._charge(wall, "interrupted_conservatively_charged")
            atomic_json(self.path, self._data)
        except BaseException:
            self._close()
            raise

    def _close(self):
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
        self._lock = None

    def initialize(self):
        """Create/read the ledger without allocating or charging GPUs."""
        if self._lock is not None:
            raise RuntimeError("ledger already open")
        self._open()
        try:
            return read_json(self.path)
        finally:
            self._close()

    def _charge(self, wall_seconds, status):
        active = self._data["active"]
        gpu_seconds = wall_seconds * active["gpu_count"]
        self._data["used_gpu_seconds"] += gpu_seconds
        self._data["stage_used_gpu_seconds"][active["stage"]] += gpu_seconds
        self._data["sessions"].append(dict(active, wall_seconds=wall_seconds,
                                           gpu_seconds=gpu_seconds, status=status))
        self._data["active"] = None
        atomic_json(self.path, self._data)

    def check(self, estimate_wall_seconds=0., cleanup_wall_seconds=0.):
        if self._started is None:
            raise RuntimeError("GPU work requires an active research budget session")
        estimate = finite(estimate_wall_seconds, "complete unit estimate")
        cleanup = finite(cleanup_wall_seconds, "cleanup estimate")
        active = self._data["active"]
        running = max(0., self.monotonic() - self._started) * active["gpu_count"]
        remaining = min(self.limit - self._data["used_gpu_seconds"] - running,
                        STAGE_LIMITS[active["stage"]]
                        - self._data["stage_used_gpu_seconds"][active["stage"]] - running)
        if (estimate + cleanup) * active["gpu_count"] >= remaining:
            raise BudgetStop("insufficient GPU-seconds for next complete paired unit and cleanup")
        return remaining / active["gpu_count"]

    def expect_workers(self):
        if self._started is None:
            raise RuntimeError("worker launch requires an active budget session")
        self._data["active"]["workers_expected"] = True
        atomic_json(self.path, self._data)

    def register_worker_group(self, pgid):
        if self._started is None or type(pgid) is not int or pgid < 1:
            raise ValueError("a live session and positive worker process group are required")
        self._data["active"].update(workers_expected=True, worker_pgid=pgid)
        atomic_json(self.path, self._data)

    def set_outcome(self, status):
        if self._started is None or status not in ("completed", "failed", "budget_stopped", "blocked_contention"):
            raise ValueError("a live budget session and valid worker outcome are required")
        self._outcome = status

    @contextlib.contextmanager
    def session(self, stage, devices, estimate_wall_seconds, cleanup_wall_seconds=30.):
        devices = list(devices)
        if stage not in STAGE_LIMITS or not devices or len(devices) > 4 or len(set(devices)) != len(devices):
            raise ValueError("valid stage and 1..4 unique devices are required")
        if self._lock is not None:
            raise RuntimeError("budget session already active")
        estimate = finite(estimate_wall_seconds, "complete unit estimate", positive=True)
        cleanup = finite(cleanup_wall_seconds, "cleanup estimate")
        self._open()
        status = "failed"
        self._outcome = None
        cleanup_failed = False
        try:
            needed = (estimate + cleanup) * len(devices)
            if (needed >= self.limit - self._data["used_gpu_seconds"] or
                    needed >= STAGE_LIMITS[stage] - self._data["stage_used_gpu_seconds"][stage]):
                raise BudgetStop("complete unit cannot fit remaining stage/total budget")
            self._started = self.monotonic()
            self._data["active"] = dict(session_id=uuid.uuid4().hex, pid=os.getpid(), stage=stage,
                                        devices=devices, gpu_count=len(devices), started_at=self.clock(),
                                        complete_unit_estimate_wall_seconds=estimate,
                                        cleanup_estimate_wall_seconds=cleanup)
            atomic_json(self.path, self._data)
            yield self
            status = self._outcome or "completed"
        except BudgetStop:
            status = "budget_stopped"
            raise
        except WorkerCleanupError:
            cleanup_failed = True
            if self._data.get("active"):
                self._data["active"]["cleanup_status"] = "unreaped_workers"
                atomic_json(self.path, self._data)
            raise
        finally:
            try:
                if self._started is not None and not cleanup_failed:
                    self._charge(max(0., self.monotonic() - self._started), status)
            finally:
                self._started = None
                self._close()
