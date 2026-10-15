"""Independent arrival clock and measured single-thread CPU encoding."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time


class EncodedArrivals:
    def __init__(self, jobs, encode, origin):
        self.jobs, self.encode, self.origin = jobs, encode, origin
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.records, self.error = {}, None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scheduling-encode")
        self.thread = threading.Thread(target=self._arrive, name="scheduling-arrivals")
        self.thread.start()

    def now(self):
        return (time.perf_counter() - self.origin) * 1000

    def _encode(self, rid, spec):
        with self.lock:
            self.records[rid]["start_ms"] = self.now()
        try:
            return self.encode(spec)
        finally:
            with self.lock:
                self.records[rid]["end_ms"] = self.now()

    def _arrive(self):
        try:
            for rid, arrival, spec in sorted(self.jobs, key=lambda x: (x[1], x[0])):
                if self.stop.wait(max(0., self.origin + arrival / 1000 - time.perf_counter())):
                    return
                with self.lock:
                    self.records[rid] = dict(submit_ms=self.now(), start_ms=None, end_ms=None, future=None)
                future = self.pool.submit(self._encode, rid, spec)
                with self.lock:
                    self.records[rid]["future"] = future
        except BaseException as error:
            with self.lock:
                self.error = error

    def snapshot(self):
        with self.lock:
            if self.error is not None:
                raise self.error
            return {rid: dict(record) for rid, record in self.records.items()}

    def close(self):
        self.stop.set()
        self.thread.join()
        self.pool.shutdown(wait=True, cancel_futures=True)
