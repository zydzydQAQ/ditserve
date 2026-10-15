"""Independent 24 GPU-hour allocation; historical ledgers remain immutable."""

import contextlib
import fcntl
import os
import time
import uuid
from pathlib import Path

from ..research.common import atomic_json, file_digest, read_json


LIMITS = {"direction1": 28800., "direction2": 21600.,
          "direction3": 28800., "reserve": 7200.}
TOTAL = 86400.


class BudgetStop(RuntimeError):
    pass


class Allocation:
    """One supervisor owns the lock from admission through child cleanup."""

    def __init__(self, path, parent, *, clock=time.time, monotonic=time.monotonic):
        self.path, self.parent, self.clock = Path(path), Path(parent), clock
        self.monotonic = monotonic
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.with_suffix(".lock").open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                self.data = read_json(self.path)
                if (self.data["limit_gpu_seconds"] != TOTAL or
                        self.data["stage_limits_gpu_seconds"] != LIMITS or
                        self.data["parent"]["path"] != str(self.parent.resolve())):
                    raise ValueError("allocation identity or limits changed")
            else:
                historical = read_json(self.parent)
                self.data = dict(schema_version="ditserve.directions.budget.v1",
                    limit_gpu_seconds=TOTAL, stage_limits_gpu_seconds=LIMITS.copy(),
                    used_gpu_seconds=0., stage_used_gpu_seconds={k: 0. for k in LIMITS},
                    parent=dict(path=str(self.parent.resolve()), sha256=file_digest(self.parent),
                                used_gpu_seconds=historical["used_gpu_seconds"],
                                limit_gpu_seconds=historical["limit_gpu_seconds"]),
                    authorization="user approved an additional 24 GPU hours for directions 1,2,3",
                    active=None, sessions=[])
                self.save()
            if self.data["active"] is not None:
                raise BudgetStop("unsettled session: verify recorded child group is gone before recovery")
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
        if stage not in LIMITS:
            raise ValueError("unknown budget stage")
        return max(0., min(TOTAL - self.data["used_gpu_seconds"],
                          LIMITS[stage] - self.data["stage_used_gpu_seconds"][stage]))

    def begin(self, stage, *, gpu_count, identity, minimum_seconds=120.):
        if self.data["active"] is not None:
            raise BudgetStop("a charged session is already active")
        if type(gpu_count) is not int or gpu_count < 1:
            raise ValueError("gpu_count must be a positive integer")
        if self.remaining(stage) < minimum_seconds * gpu_count:
            raise BudgetStop("insufficient allowance for a complete unit and cleanup")
        self.data["active"] = dict(stage=stage, gpu_count=gpu_count,
            started_at=self.clock(), started_monotonic=self.monotonic(),
            session_id=str(uuid.uuid4()), supervisor_pid=os.getpid(),
            identity=identity, child_pgid=None)
        self.save()

    def child_started(self, pid):
        if self.data["active"] is None or type(pid) is not int or pid <= 0:
            raise ValueError("invalid active child")
        self.data["active"]["child_pgid"] = pid
        self.save()

    def finish(self, status):
        active = self.data["active"]
        if active is None:
            raise ValueError("no charged session")
        ended_monotonic = self.monotonic()
        elapsed = ended_monotonic - active["started_monotonic"]
        if elapsed < 0:
            raise BudgetStop("monotonic clock reversed; leave session unsettled for recovery")
        # Wall time is audit metadata. Clock corrections must not change the
        # charge for a live supervisor's entire admitted process lifetime.
        charge = elapsed * active["gpu_count"]
        record = dict(active, ended_at=self.clock(), ended_monotonic=ended_monotonic,
                      elapsed_seconds=elapsed, gpu_seconds=charge, status=status)
        self.data["used_gpu_seconds"] += charge
        self.data["stage_used_gpu_seconds"][active["stage"]] += charge
        self.data["sessions"].append(record)
        self.data["active"] = None
        self.save()
        return record


def deadline_check(deadline):
    def check():
        if time.monotonic() >= deadline:
            raise BudgetStop("stage allowance reached; stop before starting another unit")
    return check
