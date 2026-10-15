"""D1 protocol/metric checks are CPU-only and never fabricate GPU results."""

import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from ditserve.directions import direction1 as d1
from ditserve.directions.sana import SanaAdapter, SanaPrompt, MODEL_ID, RECIPE


class Direction1Tests(unittest.TestCase):
    def test_edf_shared_across_modalities_and_never_sees_future_arrivals(self):
        rows = [dict(request_id="video", kind="video", arrival_ms=0, deadline_ms=5, status="pending"),
                dict(request_id="image", kind="image", arrival_ms=0, deadline_ms=9, status="pending"),
                dict(request_id="future", kind="image", arrival_ms=10, deadline_ms=1, status="pending")]
        self.assertEqual(d1.select_edf(rows, 0)["request_id"], "video")
        rows[0]["status"] = "completed"
        self.assertEqual(d1.select_edf(rows, 5)["request_id"], "image")
        self.assertEqual(d1.select_edf(rows, 10)["request_id"], "future")

    def test_metrics_do_not_drop_failed_missed_or_not_yet_arrived_requests(self):
        rows = [dict(arrival_ms=0, start_ms=1, finish_ms=5, deadline_ms=6, status="completed"),
                dict(arrival_ms=0, start_ms=2, finish_ms=9, deadline_ms=6, status="completed"),
                dict(arrival_ms=0, status="failed"), dict(arrived=False, status="pending")]
        result = d1.request_metrics(rows, 1000)
        self.assertEqual(result["arrived"], 3)
        self.assertEqual(result["on_time"], 1)
        self.assertEqual(result["on_time_fraction"], 1 / 3)
        self.assertEqual(result["failed_or_incomplete"], 1)
        self.assertEqual(result["goodput_rps"], 1)
        self.assertEqual(result["sample_count"], 2)
        self.assertIsNone(d1.request_metrics([], 0)["latency_p95_ms"])

    def test_frozen_trace_reproducible_across_policy_and_contains_real_modalities(self):
        kwargs = dict(count=24, seed=101, load=.5, pattern="ordinary", video_ms=1000, image_ms=10)
        original = d1.frozen_requests(**kwargs)
        self.assertEqual(original, d1.frozen_requests(**kwargs))
        self.assertEqual(sum(r["kind"] == "video" for r in original), 2)
        self.assertEqual(len(original), 24)
        copy_for_policy = copy.deepcopy(original)
        copy_for_policy[0]["status"] = "completed"
        self.assertEqual(original[0]["status"], "pending")
        burst = d1.frozen_requests(**dict(kwargs, pattern="burst"))
        self.assertEqual([r["seed"] for r in burst], [r["seed"] for r in original])
        self.assertEqual([r["service_ms"] for r in burst], [r["service_ms"] for r in original])
        for row in original:
            self.assertEqual(row["deadline_ms"], row["arrival_ms"] + row["service_ms"] * 2.5)

    def test_protocol_uses_current_config_names_and_requires_strong_baselines(self):
        result = d1.settings(dict(direction1=dict(online_patterns=["burst"], steps=4)))
        self.assertEqual(result["patterns"], ["burst"])
        self.assertEqual(result["profile_steps"], 3)
        for policies in (["step", "block-1"], ["cfg", "block-1"], ["step", "cfg", "fake"]):
            with self.assertRaises(ValueError):
                d1.settings(dict(direction1=dict(policies=policies)))

    def test_partial_evidence_atomic_immutable_and_config_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, config = Path(temporary), {"direction1": {"steps": 32}}
            evidence = d1.Evidence(config, root, "profile")
            evidence.append(dict(status="oom", shape=dict(frames=81)))
            self.assertEqual(json.loads(evidence.path.read_text())["rows"][0]["status"], "oom")
            self.assertFalse(evidence.path.with_name(evidence.path.name + ".tmp").exists())
            with self.assertRaises(ValueError):
                d1._previous(config, root, "profile")
            evidence.finish(eligible=True, selected_shape=dict(frames=49))
            self.assertEqual(d1._previous(config, root, "profile")["status"], "passed")
            with self.assertRaises(FileExistsError):
                d1.Evidence(config, root, "profile")
            with self.assertRaises(ValueError):
                d1._previous({}, root, "profile")

    def test_failure_or_stop_never_qualifies_as_a_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            failed = d1.Evidence({}, Path(temporary), "correctness").finish(passed=False, eligible=False)
            stopped = d1.Evidence({}, Path(temporary), "preemption").finish(passed=True, eligible=False)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(stopped["status"], "stopped")

    def test_budget_exhaustion_before_loading_leaves_an_interrupted_report(self):
        for error_type in (TimeoutError, d1.BudgetStop):
            def exhausted():
                raise error_type("GPU budget exhausted")
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(error_type):
                    d1.run_phase({}, Path(temporary), exhausted, "profile")
                report = json.loads((Path(temporary) / "direction1-profile.json").read_text())
                self.assertEqual(report["status"], "incomplete")
                self.assertFalse(report["eligible"])
                self.assertNotIn("execution", report)

    def test_sana_fixed_official_recipe_and_encoded_owner_guard(self):
        self.assertEqual(MODEL_ID, "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")
        self.assertEqual((RECIPE["height"], RECIPE["width"], RECIPE["num_inference_steps"]), (1024, 1024, 2))
        adapter = SanaAdapter.__new__(SanaAdapter)
        adapter._owner = object()
        payload = SanaPrompt("hello", None, None, 0., object())
        with self.assertRaisesRegex(ValueError, "another"):
            adapter.generate("hello", encoded=payload)

    def test_numerical_gate_catches_nonfinite_and_shape_errors(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy needed for numerical gate")
        self.assertTrue(d1.numerical_error(np.ones(3), np.ones(3))["exact"])
        self.assertFalse(d1.numerical_error(np.array([float("nan")]), np.ones(1))["passed"])
        self.assertFalse(d1.numerical_error(np.ones(3), np.ones(2))["passed"])

    def test_solver_history_tolerance_preserves_exact_schedule_checks(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU torch required")
        reference = SimpleNamespace(model_outputs=[torch.ones(2)], last_sample=torch.ones(2),
                                    timestep_list=[torch.tensor(10.)], _step_index=1)
        actual = copy.deepcopy(reference)
        actual.model_outputs[0][0] += 1e-6
        self.assertTrue(d1.solver_matches(torch, actual, reference))
        self.assertFalse(d1.solver_matches(torch, actual, reference, exact=True))
        actual.timestep_list[0] += 1e-6
        self.assertFalse(d1.solver_matches(torch, actual, reference))
        actual = copy.deepcopy(reference)
        actual.last_sample[0] = float("nan")
        self.assertFalse(d1.solver_matches(torch, actual, reference))

    def test_cpu_encoding_arrives_while_dispatcher_is_blocked(self):
        # No scheduler polling occurs between construction and the event: the
        # second submission must happen even while the first CPU encode is busy.
        release, entered = threading.Event(), threading.Event()
        def encode(spec):
            entered.set()
            release.wait(2.)
            return spec
        origin = time.perf_counter()
        arrivals = d1.VideoArrivals([("first", 0., {}), ("second", 30., {})], encode, origin)
        try:
            self.assertTrue(entered.wait(1.))
            time.sleep(.15)
            tickets = arrivals.snapshot()
            self.assertEqual(set(tickets), {"first", "second"})
            self.assertGreaterEqual(tickets["second"][2], 30.)
            self.assertFalse(tickets["first"][1].done())
        finally:
            release.set()
            arrivals.close()

    def test_future_arrival_cancellation_never_encodes_early(self):
        calls = []
        arrivals = d1.VideoArrivals([("future", 10000., {})], calls.append, time.perf_counter())
        arrivals.close()
        self.assertEqual(calls, [])
        self.assertEqual(arrivals.snapshot(), {})

    def test_online_decision_requires_complete_policy_pairs_and_preserves_video_slo(self):
        opts = d1.settings({})
        with self.assertRaises(ValueError):
            d1.online_decision([], opts)
        records = []
        for policy in opts["policies"]:
            wait = 10 if policy == "step" else 2
            requests = [dict(request_id="image", kind="image", arrival_ms=0, start_ms=wait,
                             finish_ms=wait + 1, deadline_ms=100, status="completed"),
                        dict(request_id="video", kind="video", arrival_ms=0, start_ms=0,
                             finish_ms=80, deadline_ms=100, status="completed")]
            records.append(dict(policy=policy, trace_sha256="one", rows=requests, elapsed_ms=100,
                metrics=d1.request_metrics(requests, 100),
                image_metrics=d1.request_metrics(requests[:1], 100),
                video_metrics=d1.request_metrics(requests[1:], 100)))
        summaries, sufficient = d1.online_decision(records, opts)
        self.assertTrue(sufficient)
        self.assertTrue(summaries["block-1"]["meets_screen"])
        # A faster image does not excuse moving the video beyond its deadline.
        for record in records:
            if record["policy"] != "step":
                record["rows"][1]["finish_ms"] = 120
                record["metrics"] = d1.request_metrics(record["rows"], 100)
                record["video_metrics"] = d1.request_metrics(record["rows"][1:], 100)
        summaries, sufficient = d1.online_decision(records, opts)
        self.assertFalse(sufficient)
        self.assertFalse(summaries["block-1"]["meets_screen"])


if __name__ == "__main__":
    unittest.main()
