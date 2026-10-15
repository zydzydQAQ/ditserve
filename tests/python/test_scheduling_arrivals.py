"""Real CPU arrival threads must preserve causality, timing and cleanup."""

import threading
import time
import unittest

from ditserve.scheduling.arrivals import EncodedArrivals


class SchedulingArrivalsTests(unittest.TestCase):
    def wait_snapshot(self, arrivals, predicate, timeout=2.):
        until = time.perf_counter() + timeout
        while time.perf_counter() < until:
            snapshot = arrivals.snapshot()
            if predicate(snapshot):
                return snapshot
            time.sleep(.002)
        self.fail("arrival thread did not reach the expected state")

    def test_future_request_is_not_visible_or_encoded_and_close_interrupts_wait(self):
        calls = []
        arrivals = EncodedArrivals([("future", 60000., {"payload": 1})], calls.append, time.perf_counter())
        try:
            self.assertEqual(arrivals.snapshot(), {})
        finally:
            started = time.perf_counter()
            arrivals.close()
        self.assertLess(time.perf_counter() - started, 1.)
        self.assertFalse(arrivals.thread.is_alive())
        self.assertEqual(calls, [])
        self.assertEqual(arrivals.snapshot(), {})

    def test_second_arrival_is_submitted_while_first_encoding_blocks(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def encode(spec):
            calls.append(spec)
            if spec == "first":
                entered.set()
                if not release.wait(2.):
                    raise TimeoutError("test encoder was not released")
            return "encoded-" + spec
        arrivals = EncodedArrivals([("a", 0., "first"), ("b", 30., "second")], encode, time.perf_counter())
        try:
            self.assertTrue(entered.wait(1.))
            queued = self.wait_snapshot(arrivals, lambda rows: "b" in rows and rows["b"]["future"] is not None)
            self.assertGreaterEqual(queued["b"]["submit_ms"], 30.)
            self.assertIsNone(queued["b"]["start_ms"])
            self.assertIsNone(queued["b"]["end_ms"])
            self.assertFalse(queued["a"]["future"].done())
            self.assertEqual(calls, ["first"])
            release.set()
            self.assertEqual(queued["a"]["future"].result(timeout=1.), "encoded-first")
            self.assertEqual(queued["b"]["future"].result(timeout=1.), "encoded-second")
            finished = arrivals.snapshot()
            for ticket in finished.values():
                self.assertGreaterEqual(ticket["submit_ms"], 0.)
                self.assertGreaterEqual(ticket["start_ms"], ticket["submit_ms"])
                self.assertGreaterEqual(ticket["end_ms"], ticket["start_ms"])
            self.assertGreaterEqual(finished["b"]["start_ms"], finished["a"]["end_ms"])
            self.assertGreater(finished["b"]["start_ms"] - finished["b"]["submit_ms"], 0.)
            # A caller cannot overwrite timing state through a snapshot.
            finished["a"]["start_ms"] = -10.
            self.assertGreaterEqual(arrivals.snapshot()["a"]["start_ms"], 0.)
        finally:
            release.set()
            arrivals.close()
        self.assertFalse(arrivals.thread.is_alive())

    def test_encoder_exception_retains_end_time_and_close_cancels_future_arrival(self):
        calls = []
        class EncodingFailure(RuntimeError):
            pass
        def encode(spec):
            calls.append(spec)
            raise EncodingFailure("bad prompt")
        arrivals = EncodedArrivals([("failed", 0., "bad"), ("future", 60000., "future")],
                                   encode, time.perf_counter())
        try:
            ticket = self.wait_snapshot(arrivals, lambda rows: rows.get("failed", {}).get("future") is not None)["failed"]
            with self.assertRaisesRegex(EncodingFailure, "bad prompt"):
                ticket["future"].result(timeout=1.)
            failed = arrivals.snapshot()["failed"]
            self.assertGreaterEqual(failed["end_ms"], failed["start_ms"])
            self.assertNotIn("future", arrivals.snapshot())
        finally:
            arrivals.close()
        self.assertFalse(arrivals.thread.is_alive())
        self.assertEqual(calls, ["bad"])

    def test_close_waits_for_running_encoder_but_cancels_queued_encoder(self):
        entered, release, close_done = threading.Event(), threading.Event(), threading.Event()
        calls = []
        def encode(spec):
            calls.append(spec)
            entered.set()
            if not release.wait(2.):
                raise TimeoutError("test encoder was not released")
            return spec
        arrivals = EncodedArrivals([("a", 0., "first"), ("b", 0., "second")], encode, time.perf_counter())
        closer = None
        try:
            self.assertTrue(entered.wait(1.))
            queued = self.wait_snapshot(arrivals, lambda rows: rows.get("b", {}).get("future") is not None)
            def close():
                try:
                    arrivals.close()
                finally:
                    close_done.set()
            closer = threading.Thread(target=close)
            closer.start()
            # shutdown cancels pending work before joining the active worker.
            until = time.perf_counter() + 1.
            while not queued["b"]["future"].cancelled() and time.perf_counter() < until:
                time.sleep(.002)
            self.assertTrue(queued["b"]["future"].cancelled())
            self.assertFalse(close_done.is_set())
            release.set()
            self.assertTrue(close_done.wait(1.))
        finally:
            release.set()
            if closer is not None:
                closer.join(2.)
            else:
                arrivals.close()
        self.assertFalse(arrivals.thread.is_alive())
        self.assertEqual(calls, ["first"])


if __name__ == "__main__":
    unittest.main()
