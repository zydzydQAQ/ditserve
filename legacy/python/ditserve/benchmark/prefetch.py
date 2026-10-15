"""Bounded CPU encoding for already-arrived requests; never owns CUDA state."""

from concurrent.futures import ThreadPoolExecutor, wait
import threading


class PromptPrefetcher:
    def __init__(self, adapter, clock, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("text_prefetch must be a positive integer")
        if clock.virtual:
            raise ValueError("CPU text prefetch requires a real clock")
        self.adapter, self.clock, self.capacity = adapter, clock, capacity
        self.jobs = {}
        self.starts = {}
        self.events = []
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="benchmark-text")
        self.closed = False
        self.peak_pending = 0

    def submit(self, request):
        rid = request["request_id"]
        if self.closed or rid in self.jobs or len(self.jobs) >= self.capacity:
            return False
        if request["arrival_ms"] > self.clock.now():
            raise ValueError("cannot encode a request before its scheduled arrival")
        self.jobs[rid] = self.executor.submit(self._encode, request)
        self.peak_pending = max(self.peak_pending, len(self.jobs))
        return True

    def _encode(self, request):
        rid = request["request_id"]
        start = self.clock.now()
        with self.lock:
            self.starts[rid] = start
        status = "completed"
        try:
            return self.adapter.encode(request)
        except BaseException:
            status = "failed"
            raise
        finally:
            end = self.clock.now()
            with self.lock:
                self.events.append((rid, dict(kind="text_encode", start_ms=start,
                    end_ms=end, ms=end - start, status=status, resource="cpu")))

    def ready(self, rid):
        return rid in self.jobs and self.jobs[rid].done()

    def take(self, rid):
        if not self.ready(rid):
            raise ValueError("encoded request is not ready")
        return self.jobs.pop(rid).result()

    def discard(self, rid):
        future = self.jobs.get(rid)
        if future is not None and (future.cancel() or future.done()):
            self.jobs.pop(rid, None)

    def collect(self):
        with self.lock:
            events, self.events = self.events, []
            return dict(self.starts), events

    def close(self):
        """Defer Python KeyboardInterrupt until the CPU futures are terminal."""
        self.closed = True
        futures = tuple(self.jobs.values())
        interrupted = None
        # Do not release the model while a CPU encoder still uses it. Cleanup
        # remains inside the enclosing GPU budget session, outside E2E timing.
        while True:
            try:
                for future in futures:
                    future.cancel()
                self.executor.shutdown(wait=True)
                # An interrupted Thread.join can mark a thread stopped while
                # its callable is still running. Retrying shutdown alone can
                # therefore return early; Future state is the completion guard.
                if all(future.done() for future in futures):
                    break
                wait(futures, timeout=.05)
            except KeyboardInterrupt as error:
                if interrupted is None:
                    interrupted = error
        self.jobs.clear()
        if interrupted is not None:
            raise interrupted
