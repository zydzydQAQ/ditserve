"""Bounded CPU persistence for the separate asynchronous-artifact experiment.

The caller retains ownership of arrays and must not mutate them until ``drain``
or ``close`` returns.  This writer neither copies nor changes arrays, and never
accepts device tensors.  Its capacity includes the running job and queued jobs.
"""

from collections import deque
from pathlib import Path
import re
import threading
import time

import numpy as np


class AsyncArtifacts:
    """Save at most two request batches in flight on one CPU worker thread."""

    capacity = 2

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self._condition = threading.Condition()
        self._close_lock = threading.Lock()
        self._jobs = deque()
        self._seen = set()
        self._in_flight = 0
        self._accepting = True
        self._closed = False
        self._error = None
        self._stats = dict(submitted=0, completed=0, failed=0, skipped=0,
                           files=0, bytes=0, write_ms=0., queue_wait_ms=0.,
                           drain_wait_ms=0., max_in_flight=0, capacity=self.capacity)
        self._thread = threading.Thread(target=self._work, name="scheduling-artifacts", daemon=False)
        self._thread.start()

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError("asynchronous artifact persistence failed") from self._error

    def save(self, rid, output, latent):
        """Submit a request batch and return its absolute artifact paths.

        The replay must create the directory before submitting.  Returning
        means accepted, not persisted.  ``drain`` is required before
        reading files or reporting a successful replay.  A full two-batch window
        blocks this call and records that foreground delay in ``queue_wait_ms``.
        """
        with self._condition:
            self._raise_error()
            if not self._accepting:
                raise RuntimeError("artifact writer is closed")
            if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", rid):
                raise ValueError("request id must be a simple nonempty filename component")
            arrays = []
            paths = {}
            for kind, array in (("output", output), ("latent", latent)):
                if array is None:
                    continue
                if not isinstance(array, np.ndarray):
                    raise TypeError("artifact arrays must be CPU NumPy arrays")
                if array.dtype.hasobject:
                    raise TypeError("object arrays cannot be persisted without pickle")
                path = self.directory / (rid + "-" + kind + ".npy")
                arrays.append((path, array))
                paths[kind] = str(path)
            if rid in self._seen or any(path.exists() for path, _ in arrays):
                raise ValueError("request artifact already exists: " + rid)
            if not arrays:
                return paths
            waited_at = time.perf_counter() if self._in_flight >= self.capacity else None
            try:
                while self._in_flight >= self.capacity:
                    self._condition.wait()
                    self._raise_error()
                    if not self._accepting:
                        raise RuntimeError("artifact writer is closed")
            finally:
                if waited_at is not None:
                    self._stats["queue_wait_ms"] += (time.perf_counter() - waited_at) * 1000
            # Another submitting thread may have claimed this id while we waited.
            if rid in self._seen:
                raise ValueError("request artifact already submitted: " + rid)
            self._seen.add(rid)
            self._jobs.append(arrays)
            self._in_flight += 1
            self._stats["submitted"] += 1
            self._stats["max_in_flight"] = max(self._stats["max_in_flight"], self._in_flight)
            self._condition.notify_all()
            return paths

    def _work(self):
        while True:
            with self._condition:
                while not self._jobs and self._accepting:
                    self._condition.wait()
                if not self._jobs:
                    return
                arrays = self._jobs.popleft()
                skip = self._error is not None
            started = time.perf_counter()
            files, size, error = 0, 0, None
            if not skip:
                try:
                    for path, array in arrays:
                        np.save(str(path), array, allow_pickle=False)
                        files += 1
                        size += path.stat().st_size
                except BaseException as exc:
                    error = exc.with_traceback(None)
            # Release all input references before making the capacity available.
            arrays = None
            array = None
            with self._condition:
                self._stats["files"] += files
                self._stats["bytes"] += size
                self._stats["write_ms"] += 0. if skip else (time.perf_counter() - started) * 1000
                if skip:
                    self._stats["skipped"] += 1
                elif error is not None:
                    self._stats["failed"] += 1
                    if self._error is None:
                        self._error = error
                else:
                    self._stats["completed"] += 1
                self._in_flight -= 1
                self._condition.notify_all()

    def drain(self):
        """Wait for every accepted batch, then return accounting or raise failure."""
        with self._condition:
            started = time.perf_counter() if self._in_flight else None
            while self._in_flight:
                self._condition.wait()
            if started is not None:
                self._stats["drain_wait_ms"] += (time.perf_counter() - started) * 1000
            self._raise_error()
            return dict(self._stats, pending=self._in_flight, closed=self._closed)

    def close(self):
        """Stop submissions and join the worker even if a write or drain failed."""
        with self._close_lock:
            with self._condition:
                self._accepting = False
                self._condition.notify_all()
            try:
                return self.drain()
            finally:
                self._thread.join()
                with self._condition:
                    self._closed = True
