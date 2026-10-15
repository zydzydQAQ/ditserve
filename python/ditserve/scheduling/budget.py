"""One independent, cumulative 12 GPU-hour allocation for scheduling research."""

import fcntl
import os
import time
import uuid
from pathlib import Path

from ..research.common import atomic_json, file_digest, finite, read_json


TOTAL = 43200.0
SCHEMA = "ditserve.scheduling.budget.v1"
AUTHORIZATION = "user approved an independent 12 GPU-hour scheduling allocation"
HISTORICAL_LEDGER = Path(__file__).resolve().parents[3] / "results/research-directions/gpu-budget.json"


class BudgetStop(RuntimeError):
    pass


class Allocation:
    """Hold the single-ledger lock until the owned process group is cleaned up."""

    def __init__(self, path, *, clock=time.time, monotonic=time.monotonic,
                 historical_path=HISTORICAL_LEDGER):
        self.path = Path(path).resolve()
        self.clock, self.monotonic = clock, monotonic
        self.historical_path = Path(historical_path).resolve()
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                self.data = read_json(self.path)
                if (self.data.get("schema_version") != SCHEMA or
                        self.data.get("limit_gpu_seconds") != TOTAL or
                        self.data.get("authorization") != AUTHORIZATION):
                    raise ValueError("scheduling allocation identity or authorized limit differs")
                finite(self.data["used_gpu_seconds"], "used_gpu_seconds")
                charges = sum(finite(row["gpu_seconds"], "session gpu_seconds")
                              for row in self.data["sessions"])
                if abs(charges - self.data["used_gpu_seconds"]) > 1e-6:
                    raise ValueError("scheduling allocation charge history is inconsistent")
            else:
                historical = read_json(self.historical_path)
                self.data = dict(schema_version=SCHEMA, limit_gpu_seconds=TOTAL,
                                 authorization=AUTHORIZATION, used_gpu_seconds=0.0,
                                 allocation_id=str(uuid.uuid4()), active=None, sessions=[],
                                 historical_source=dict(path=str(self.historical_path),
                                     sha256=file_digest(self.historical_path),
                                     used_gpu_seconds=historical["used_gpu_seconds"],
                                     limit_gpu_seconds=historical["limit_gpu_seconds"],
                                     inherited_budget=False))
                self.save()
            if self.data.get("active") is not None:
                raise BudgetStop("unsettled scheduling session; verify cleanup before recovery")
            return self
        except BaseException:
            self.handle.close()
            self.handle = None
            raise

    def __exit__(self, *unused):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def save(self):
        atomic_json(self.path, self.data)

    def remaining(self):
        return max(0.0, TOTAL - self.data["used_gpu_seconds"])

    def begin(self, *, phase, identity, gpu_count=1, minimum_seconds=120.0):
        if self.data["active"] is not None:
            raise BudgetStop("a charged scheduling session is already active")
        if type(gpu_count) is not int or gpu_count != 1:
            raise ValueError("scheduling phases require one GPU")
        minimum_seconds = finite(minimum_seconds, "minimum_seconds", positive=True)
        if self.remaining() < minimum_seconds * gpu_count:
            raise BudgetStop("insufficient allowance for a complete unit and cleanup")
        self.data["active"] = dict(phase=phase, gpu_count=gpu_count,
                                   session_id=str(uuid.uuid4()), supervisor_pid=os.getpid(),
                                   child_pgid=None, started_at=self.clock(),
                                   started_monotonic=self.monotonic(), identity=identity)
        self.save()
        return self.data["active"]

    def child_started(self, pid):
        if self.data["active"] is None or type(pid) is not int or pid <= 0:
            raise ValueError("invalid scheduling child process")
        self.data["active"]["child_pgid"] = pid
        self.save()

    def finish(self, status):
        active = self.data["active"]
        if active is None:
            raise ValueError("no charged scheduling session")
        ended = self.monotonic()
        elapsed = ended - active["started_monotonic"]
        if elapsed < 0:
            raise BudgetStop("monotonic clock reversed; leave scheduling session unsettled")
        charge = elapsed * active["gpu_count"]
        record = dict(active, ended_at=self.clock(), ended_monotonic=ended,
                      elapsed_seconds=elapsed, gpu_seconds=charge, status=status)
        self.data["sessions"].append(record)
        self.data["used_gpu_seconds"] += charge
        self.data["active"] = None
        self.save()
        return record


class DeadlineCheck:
    """Cooperative deadline plus an explicit admission check for whole groups."""

    def __init__(self, deadline, *, monotonic=time.monotonic, parent_pid=None,
                 getppid=os.getppid):
        self.deadline = finite(deadline, "deadline", positive=True)
        self.monotonic, self.parent_pid, self.getppid = monotonic, parent_pid, getppid

    def remaining_seconds(self):
        return max(0.0, self.deadline - self.monotonic())

    def __call__(self):
        if self.parent_pid is not None and self.getppid() != self.parent_pid:
            raise BudgetStop("charged supervisor exited; stop orphaned worker")
        if self.remaining_seconds() <= 0:
            raise BudgetStop("scheduling allowance reached")

    def require_seconds(self, seconds, label="complete paired group"):
        self()
        if self.remaining_seconds() < finite(seconds, "required seconds", positive=True):
            raise BudgetStop("insufficient time for " + label)
