"""Additional allocation with forward-only carry and independently charged retries."""
import fcntl
import os
import time
import uuid
from pathlib import Path

from ..directions.budget import BudgetStop
from ..research.common import atomic_json, read_json, file_digest
from .protocol import PHASES, LIMITS, TOTAL


class Allocation:
    def __init__(self, path, parent, *, monotonic=time.monotonic, clock=time.time):
        self.path, self.parent = Path(path), Path(parent)
        self.monotonic, self.clock, self.handle = monotonic, clock, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.with_suffix(".lock").open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            parent = read_json(self.parent)
            if parent.get("active"):
                raise BudgetStop("historical allocation is still active")
            if self.path.exists():
                self.data = read_json(self.path)
                if (self.data["limit_gpu_seconds"] != TOTAL or self.data["stage_limits_gpu_seconds"] != LIMITS
                        or self.data["parent"]["sha256"] != file_digest(self.parent)
                        or self.data["parent"]["path"] != str(self.parent.resolve())):
                    raise ValueError("allocation or immutable parent differs")
            else:
                self.data = dict(schema_version="ditserve.optimizations.budget.v1", limit_gpu_seconds=TOTAL,
                    stage_limits_gpu_seconds=LIMITS.copy(), used_gpu_seconds=0.,
                    stage_used_gpu_seconds={key: 0. for key in LIMITS}, active=None, sessions=[],
                    authorization="user approved additional 8 GPU hours and forward-only unused phase carry",
                    parent=dict(path=str(self.parent.resolve()), sha256=file_digest(self.parent),
                                used_gpu_seconds=parent["used_gpu_seconds"], limit_gpu_seconds=parent["limit_gpu_seconds"]))
                self.save()
            if self.data["active"]:
                raise BudgetStop("unsettled GPU session requires ownership and cleanup audit")
            return self
        except BaseException:
            self.handle.close()
            self.handle = None
            raise

    def __exit__(self, *unused):
        if self.handle:
            self.handle.close()
            self.handle = None

    def save(self):
        atomic_json(self.path, self.data)

    def remaining(self, stage):
        used = self.data["stage_used_gpu_seconds"]
        if stage == "reserve":
            remaining = LIMITS[stage] - used[stage]
        else:
            allowed = PHASES[:PHASES.index(stage) + 1]
            remaining = sum(LIMITS[key] for key in allowed) - sum(used[key] for key in PHASES)
        return max(0., min(remaining, TOTAL - self.data["used_gpu_seconds"]))

    def begin(self, stage, *, gpu_count, identity, minimum_seconds=120.):
        if self.data["active"] or type(gpu_count) is not int or gpu_count < 1:
            raise ValueError("invalid charged admission")
        if self.remaining(stage) < minimum_seconds * gpu_count:
            raise BudgetStop("insufficient allowance for a complete unit and cleanup")
        self.data["active"] = dict(stage=stage, gpu_count=gpu_count, identity=identity,
            started_at=self.clock(), started_monotonic=self.monotonic(), session_id=str(uuid.uuid4()),
            supervisor_pid=os.getpid(), child_pgid=None)
        self.save()

    def child_started(self, pid):
        if not self.data["active"] or type(pid) is not int or pid <= 0:
            raise ValueError("invalid child process")
        self.data["active"]["child_pgid"] = pid
        self.save()

    def finish(self, status):
        active = self.data["active"]
        if not active:
            raise ValueError("no active session")
        elapsed = self.monotonic() - active["started_monotonic"]
        if elapsed < 0:
            raise BudgetStop("monotonic clock reversed")
        record = dict(active, ended_at=self.clock(), elapsed_seconds=elapsed,
                      gpu_seconds=elapsed * active["gpu_count"], status=status)
        self.data["used_gpu_seconds"] += record["gpu_seconds"]
        self.data["stage_used_gpu_seconds"][active["stage"]] += record["gpu_seconds"]
        self.data["sessions"].append(record)
        self.data["active"] = None
        self.save()
        return record


class BudgetGuard:
    def __init__(self, deadline, supervisor_pid=None):
        self.deadline, self.supervisor_pid = deadline, supervisor_pid

    def __call__(self):
        if self.supervisor_pid is not None and os.getppid() != self.supervisor_pid:
            raise BudgetStop("supervisor exited")
        if time.monotonic() >= self.deadline:
            raise BudgetStop("phase allowance reached")

    def remaining(self):
        self()
        return self.deadline - time.monotonic()

    def ensure_unit(self, seconds):
        if self.remaining() < seconds:
            raise BudgetStop("remaining budget cannot admit complete paired unit (%.1fs)" % seconds)
