"""Persistent GPU wall-time ledger, including failed and interrupted attempts.

A session owns an exclusive ledger lock for its lifetime. Interrupted sessions
are charged through the next recovery attempt (conservatively), never reset.
This ledger is independent of PreViS experiment budgets.
"""

import contextlib
import fcntl
import os
import time
import uuid
from pathlib import Path

from .common import atomic_json, positive, read_json


class BudgetExceeded(RuntimeError):
    pass


class GpuBudget:
    def __init__(self, path, hours=2.0, clock=time.time, monotonic=time.monotonic):
        self.path = Path(path)
        self.limit_seconds = positive(hours, "gpu_hours") * 3600
        if self.limit_seconds > 7200:
            raise ValueError("this benchmark's authorized GPU budget is at most 2 GPU hours")
        self.clock, self.monotonic = clock, monotonic
        self._active = None
        self._lock = None
        self._ledger = None

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            self._lock = None
            raise RuntimeError("GPU budget is already in use")
        try:
            if self.path.exists():
                ledger = read_json(self.path)
                if ledger.get("schema_version") != "ditserve.benchmark.budget.v1":
                    raise ValueError("unsupported budget ledger")
                positive(ledger["used_seconds"], "used_seconds", allow_zero=True)
                if ledger["limit_seconds"] != self.limit_seconds:
                    raise ValueError("existing budget limit cannot be changed")
            else:
                ledger = {"schema_version": "ditserve.benchmark.budget.v1",
                          "limit_seconds": self.limit_seconds, "used_seconds": 0.,
                          "sessions": [], "active": None}
            if ledger.get("active"):
                active = ledger["active"]
                elapsed = max(0., self.clock() - active["started_at"])
                ledger["used_seconds"] += elapsed
                ledger["sessions"].append(dict(active, elapsed_seconds=elapsed,
                                                status="interrupted_conservatively_charged"))
                ledger["active"] = None
                atomic_json(self.path, ledger)
            self._ledger = ledger
        except BaseException:
            self._close()
            raise

    def _close(self):
        if self._lock:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
        self._lock = None

    def check(self, seconds=0):
        positive(seconds, "reservation_seconds", allow_zero=True)
        if self._ledger is None:
            raise RuntimeError("GPU work must run inside a budget session")
        running = 0. if self._active is None else max(0., self.monotonic() - self._active)
        if self._ledger["used_seconds"] + running + seconds >= self.limit_seconds:
            raise BudgetExceeded("2-hour GPU budget insufficient for the next complete work group")
        return self.limit_seconds - self._ledger["used_seconds"] - running

    def start(self):
        if self._lock is not None:
            raise RuntimeError("budget session is already active")
        self._open()
        try:
            self.check()
            self._active = self.monotonic()
            self._ledger["active"] = {"session_id": uuid.uuid4().hex, "pid": os.getpid(),
                                      "started_at": self.clock()}
            atomic_json(self.path, self._ledger)
        except BaseException:
            self._active = None
            self._close()
            raise
        return self

    def finish(self, status="completed"):
        if self._active is None:
            return
        try:
            elapsed = max(0., self.monotonic() - self._active)
            self._ledger["used_seconds"] += elapsed
            self._ledger["sessions"].append(dict(self._ledger["active"],
                                                elapsed_seconds=elapsed, status=status))
            self._ledger["active"] = None
            atomic_json(self.path, self._ledger)
        finally:
            self._active = None
            self._close()

    @contextlib.contextmanager
    def session(self):
        self.start()
        try:
            yield self
        except BaseException:
            self.finish("failed")
            raise
        else:
            self.finish()
