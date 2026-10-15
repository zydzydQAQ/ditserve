"""Unlimited cumulative accounting with finite, per-phase watchdogs."""

import fcntl
import os
import time
import uuid
from pathlib import Path

from ..research.common import atomic_json, file_digest, finite, read_json


SCHEMA = "ditserve.scheduling.next.budget.v1"
AUTHORIZATION = "user_2026_10_15_unlimited_v100"
HISTORICAL_LEDGER = Path(__file__).resolve().parents[3] / "results/research-scheduling/gpu-budget.json"


class AccountingStop(RuntimeError):
    """An unsettled or invalid charged session cannot admit another worker."""


class WatchdogStop(AccountingStop):
    """A finite phase watchdog expired; cumulative accounting has no limit."""


class Allocation:
    """Lock one ledger through loading, execution, failures and GPU cleanup."""

    def __init__(self, path, *, clock=time.time, monotonic=time.monotonic,
                 historical_path=HISTORICAL_LEDGER):
        self.path = Path(path).resolve()
        self.historical_path = Path(historical_path).resolve()
        if self.path in (self.historical_path, HISTORICAL_LEDGER.resolve()):
            raise ValueError("the historical 12-hour ledger is read-only")
        self.clock, self.monotonic, self.handle = clock, monotonic, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                self.data = read_json(self.path)
                if (self.data.get("schema_version") != SCHEMA or
                        "limit_gpu_seconds" not in self.data or self.data["limit_gpu_seconds"] is not None or
                        self.data.get("authorization") != AUTHORIZATION):
                    raise ValueError("unlimited accounting identity or authorization differs")
                used = finite(self.data["used_gpu_seconds"], "used_gpu_seconds")
                charges = sum(finite(row["gpu_seconds"], "session gpu_seconds")
                              for row in self.data["sessions"])
                if abs(charges - used) > 1e-6:
                    raise ValueError("unlimited accounting charge history is inconsistent")
            else:
                historical = read_json(self.historical_path)
                self.data = dict(schema_version=SCHEMA, limit_gpu_seconds=None,
                    authorization=AUTHORIZATION, used_gpu_seconds=0.0, allocation_id=str(uuid.uuid4()),
                    active=None, sessions=[], historical_source=dict(path=str(self.historical_path),
                        sha256=file_digest(self.historical_path), used_gpu_seconds=historical["used_gpu_seconds"],
                        limit_gpu_seconds=historical["limit_gpu_seconds"], inherited_budget=False))
                self.save()
            if self.data.get("active") is not None:
                raise AccountingStop("unsettled session; verify owned GPU cleanup before recovery")
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
        return None

    def begin(self, *, phase, identity, gpu_count=1):
        if self.data["active"] is not None:
            raise AccountingStop("a charged session is already active")
        if type(gpu_count) is not int or gpu_count != 1:
            raise ValueError("next scheduling phases require one GPU")
        self.data["active"] = dict(phase=phase, gpu_count=gpu_count, session_id=str(uuid.uuid4()),
            supervisor_pid=os.getpid(), child_pgid=None, started_at=self.clock(),
            started_monotonic=self.monotonic(), identity=identity)
        self.save()
        return self.data["active"]

    def child_started(self, pid):
        if self.data["active"] is None or type(pid) is not int or pid <= 0:
            raise ValueError("invalid next scheduling child process")
        self.data["active"]["child_pgid"] = pid
        self.save()

    def finish(self, status):
        active = self.data["active"]
        if active is None:
            raise ValueError("no charged next scheduling session")
        ended = self.monotonic()
        elapsed = ended - active["started_monotonic"]
        if elapsed < 0:
            raise AccountingStop("monotonic clock reversed; leave accounting unsettled")
        charge = elapsed * active["gpu_count"]
        record = dict(active, ended_at=self.clock(), ended_monotonic=ended,
                      elapsed_seconds=elapsed, gpu_seconds=charge, status=status)
        self.data["sessions"].append(record)
        self.data["used_gpu_seconds"] += charge
        self.data["active"] = None
        self.save()
        return record


class DeadlineCheck:
    """Cooperative phase watchdog and whole-work admission; never a GPU budget."""

    def __init__(self, deadline, *, monotonic=time.monotonic, parent_pid=None, getppid=os.getppid):
        self.deadline = finite(deadline, "watchdog deadline", positive=True)
        self.monotonic, self.parent_pid, self.getppid = monotonic, parent_pid, getppid

    def remaining_seconds(self):
        return max(0.0, self.deadline - self.monotonic())

    def __call__(self):
        if self.parent_pid is not None and self.getppid() != self.parent_pid:
            raise WatchdogStop("charged supervisor exited; stop orphaned worker")
        if self.remaining_seconds() <= 0:
            raise WatchdogStop("finite phase watchdog expired; cumulative GPU accounting remains unlimited")

    def require_seconds(self, seconds, label="complete paired group"):
        self()
        if self.remaining_seconds() < finite(seconds, "required seconds", positive=True):
            raise WatchdogStop("phase watchdog cannot admit " + label)
