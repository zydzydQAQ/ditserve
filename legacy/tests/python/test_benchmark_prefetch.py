import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ditserve.benchmark.backends import RealClock, VirtualClock
from ditserve.benchmark.prefetch import PromptPrefetcher
from ditserve.benchmark.runner import _collect_encoding, _run_policy, build_profile, select_ready
from test_benchmark_runner import config, dataset, request, trace


class EncodingTests(unittest.TestCase):
    def test_capacity_includes_running_and_completed_payloads_and_rejects_future(self):
        release, started = threading.Event(), threading.Event()
        class Encoder:
            def encode(self, value):
                started.set()
                release.wait(2)
                return value["request_id"]
        worker = PromptPrefetcher(Encoder(), RealClock(), 1)
        try:
            self.assertTrue(worker.submit(dict(request_id="one", arrival_ms=0)))
            self.assertTrue(started.wait(1))
            self.assertFalse(worker.submit(dict(request_id="two", arrival_ms=0)))
            release.set()
            deadline = time.monotonic() + 1
            while not worker.ready("one") and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertTrue(worker.ready("one"))
            self.assertFalse(worker.submit(dict(request_id="two", arrival_ms=0)))
            self.assertEqual(worker.take("one"), "one")
            with self.assertRaisesRegex(ValueError, "before its scheduled arrival"):
                worker.submit(dict(request_id="future", arrival_ms=1e9))
        finally:
            release.set()
            worker.close()
        with self.assertRaisesRegex(ValueError, "real clock"):
            PromptPrefetcher(Encoder(), VirtualClock(), 1)

    def test_close_waits_for_active_encoder_and_cancels_queue(self):
        release, started, closed = threading.Event(), threading.Event(), threading.Event()
        called = []
        class Encoder:
            def encode(self, value):
                called.append(value["request_id"])
                started.set()
                release.wait(2)
                return 1
        worker = PromptPrefetcher(Encoder(), RealClock(), 2)
        worker.submit(dict(request_id="one", arrival_ms=0))
        self.assertTrue(started.wait(1))
        worker.submit(dict(request_id="two", arrival_ms=0))
        closer = threading.Thread(target=lambda: (worker.close(), closed.set()))
        closer.start()
        self.assertFalse(closed.wait(.02))
        release.set()
        closer.join(2)
        self.assertTrue(closed.is_set())
        self.assertEqual(called, ["one"])


    def test_interrupted_shutdown_waits_for_future_even_if_retry_returns_early(self):
        release, started = threading.Event(), threading.Event()
        retried_shutdown, closed = threading.Event(), threading.Event()
        errors = []

        class Encoder:
            def encode(self, value):
                started.set()
                if not release.wait(2):
                    raise AssertionError("test did not release the encoder")
                return value["request_id"]

        worker = PromptPrefetcher(Encoder(), RealClock(), 1)
        original_shutdown = worker.executor.shutdown
        closer = None
        interruption = KeyboardInterrupt("interrupted executor join")
        calls = 0

        def shutdown(wait=True):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise interruption
            # Model the falsely stopped Thread.join path: shutdown can return
            # on its second call even though the real encode Future is RUNNING.
            retried_shutdown.set()

        def close():
            try:
                worker.close()
            except BaseException as error:
                errors.append(error)
            finally:
                closed.set()

        try:
            worker.submit(dict(request_id="one", arrival_ms=0))
            self.assertTrue(started.wait(1))
            future = worker.jobs["one"]
            with patch.object(worker.executor, "shutdown", side_effect=shutdown):
                closer = threading.Thread(target=close)
                closer.start()
                self.assertTrue(retried_shutdown.wait(1))
                self.assertTrue(future.running())
                self.assertFalse(closed.wait(.02))
                release.set()
                closer.join(2)
                self.assertFalse(closer.is_alive())
                self.assertTrue(future.done())
                self.assertEqual(future.result(), "one")
                self.assertEqual(errors, [interruption])
                self.assertFalse(worker.jobs)
        finally:
            release.set()
            if closer is not None:
                closer.join(2)
            # The mocked shutdown deliberately never signals the real pool.
            original_shutdown(wait=True)


class TimedAdapter:
    """Real-clock test double; no GPU performance claims are made from it."""
    virtual = False

    def __init__(self, fail_encode=False):
        self.clock = RealClock()
        self.fail_encode = fail_encode
        self.states = {}

    def encode(self, value):
        time.sleep(.015)
        if self.fail_encode:
            raise RuntimeError("encoder failed")
        return value["seed"]

    def prepare_encoded(self, value, encoded):
        state = SimpleNamespace(spec=value, step_index=0, residency="gpu", events=[], encoded=encoded)
        self.states[value["request_id"]] = state
        return state

    def step(self, state):
        time.sleep(.025)
        state.step_index += 1
        event = dict(kind="step", ms=25., path="compute")
        state.events.append(event)
        return event

    def latent(self, state):
        import numpy as np
        return np.array([state.encoded], dtype=np.float32)

    def finish(self, state):
        return self.latent(state)

    def release(self, state):
        del self.states[state.spec["request_id"]]

    def available_mb(self):
        return 1e6

    def state_mb(self, state):
        return 1.

    def memory(self):
        return {"scope": "test_double"}


class PipelineTests(unittest.TestCase):
    def test_late_encoding_receipt_corrects_gpu_start_but_not_after_cutoff(self):
        class Receipts:
            def collect(self):
                return {"started": 10., "cancelled": 120.}, [
                    ("started", dict(kind="text_encode", start_ms=10., end_ms=30., ms=20.)),
                    ("cancelled", dict(kind="text_encode", start_ms=120., end_ms=140., ms=20.))]
        rows = {"started": dict(start_ms=31., events=[]), "cancelled": dict(start_ms=None, events=[])}
        _collect_encoding(Receipts(), rows, cutoff=100.)
        self.assertEqual(rows["started"]["start_ms"], 10.)
        self.assertIsNone(rows["cancelled"]["start_ms"])
        self.assertTrue(rows["cancelled"]["events"][0]["outside_policy_horizon"])

    def test_srtf_does_not_charge_already_finished_cpu_encoding_again(self):
        current = dict(request_id="current", shape_id="current", arrival_ms=0)
        encoded = dict(request_id="encoded", shape_id="encoded", arrival_ms=1)
        profile = {"shapes": {
            "current": dict(service_ms=120., cpu_prompt_encode_ms=100., step_ms=[10., 10.], decode_ms=0.),
            "encoded": dict(service_ms=103., cpu_prompt_encode_ms=100., step_ms=[3.], decode_ms=0.)}}
        states = {"current": SimpleNamespace(step_index=0)}
        self.assertEqual(select_ready("srtf", [current, encoded], "current", states, profile)["request_id"], "current")
        self.assertEqual(select_ready("srtf", [current, encoded], "current", states, profile,
                                      encoded_ids={"encoded"})["request_id"], "encoded")

    def test_cpu_encoding_overlaps_execution_without_changing_output_or_arrivals(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = build_profile(config(), dataset(), "cpu-test", root / "profile")
            frozen = trace(profile, [request("one", "long", 0, 1000), request("two", "long", 0, 1000)])
            frozen["config"]["timeout_scale"] = 1000
            result = _run_policy(frozen, "fcfs", TimedAdapter(), root / "run", None, text_prefetch=2)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["text_prefetch_peak_pending"], 2)
            first, second = result["requests"]
            encoding = next(e for e in second["events"] if e["kind"] == "text_encode")
            steps = [e for e in first["events"] if e["kind"] == "step"]
            self.assertTrue(any(encoding["start_ms"] < e["end_ms"] and encoding["end_ms"] > e["start_ms"] for e in steps))
            for row in result["requests"]:
                self.assertEqual(row["status"], "completed")
                self.assertGreaterEqual(row["start_ms"], row["arrival_ms"])
                self.assertEqual(float(np.load(row["artifact"]["output_path"])[0]), row["generation"]["seed"])
            self.assertEqual(result["observations"]["step_preemptions"], 0)

    def test_encoding_failure_preserves_arrived_denominator_and_closes_thread(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = build_profile(config(), dataset(), "cpu-test", root / "profile")
            frozen = trace(profile, [request("bad", "long", 0, 1000), request("later", "long", 10000, 11000)])
            frozen["config"]["timeout_scale"] = 1000
            result = _run_policy(frozen, "fcfs", TimedAdapter(fail_encode=True), root / "run", None, text_prefetch=2)
            self.assertEqual(result["status"], "partial")
            self.assertIn("encoder failed", result["error"])
            self.assertEqual([r["status"] for r in result["requests"]], ["arrived_aborted", "not_arrived"])
            self.assertEqual(result["cleanup_errors"], [])


if __name__ == "__main__":
    unittest.main()
