"""CPU-only numerical and protocol tests for the single-seed follow-up."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

from ditserve.optimizations import followup_study as study
from ditserve.optimizations.online import request_identity
from ditserve.research.common import read_json


@unittest.skipUnless(np is not None, "requires CPU NumPy")
class CompareArraysTests(unittest.TestCase):
    def test_strided_video_matches_contiguous_reference_exactly(self):
        data = np.linspace(0., 1., 3 * 5 * 4 * 6, dtype=np.float32).reshape(3, 5, 4, 6)
        candidate = np.expand_dims(data.transpose(1, 2, 3, 0), 0)
        self.assertFalse(candidate.flags.c_contiguous)
        reference = np.stack([candidate[0]])
        result = study.compare_arrays(reference, candidate)
        self.assertTrue(result["passed"])
        self.assertTrue(result["exact"])
        self.assertTrue(result["finite"])
        self.assertEqual(result["lpips"], 0.)
        self.assertEqual(result["max_abs"], 0.)
        self.assertTrue(result["psnr_infinite"])
        json.dumps(result, allow_nan=False)

    def test_nan_and_inf_fail_and_remain_json_serializable(self):
        reference = np.full((1, 2, 3, 4, 3), .5, dtype=np.float32)
        for bad in (float("nan"), float("inf"), float("-inf")):
            candidate = reference.copy()
            candidate[0, 0, 0, 0, 0] = bad
            with np.errstate(invalid="ignore", divide="ignore"):
                result = study.compare_arrays(reference, candidate)
            self.assertFalse(result["passed"])
            self.assertFalse(result["finite"])
            self.assertIsNone(result["lpips"])
            json.dumps(result, allow_nan=False)

    def test_identical_out_of_range_arrays_are_not_valid_outputs(self):
        for value in (-.01, 1.01):
            data = np.full((1, 2, 3, 4, 3), value, dtype=np.float32)
            result = study.compare_arrays(data, data.copy())
            self.assertTrue(result["exact"])
            self.assertFalse(result["passed"])
            self.assertFalse(result["range_valid"])
            self.assertIsNone(result["lpips"])

    def test_same_backend_numeric_tolerance_is_unchanged(self):
        reference = np.full((1, 2, 3, 4, 3), .5, dtype=np.float32)
        within, outside = reference.copy(), reference.copy()
        within[0, 0, 0, 0, 0] += 5e-6
        outside[0, 0, 0, 0, 0] += 2e-5
        good, bad = study.compare_arrays(reference, within), study.compare_arrays(reference, outside)
        self.assertTrue(good["passed"])
        self.assertFalse(good["exact"])
        self.assertIsNone(good["lpips"])
        self.assertEqual(good["rtol"], 1e-5)
        self.assertEqual(good["atol"], 1e-6)
        self.assertFalse(bad["passed"])

    def test_shape_and_dtype_mismatch_fail_before_comparison(self):
        reference = np.zeros((1, 2, 3, 4, 3), dtype=np.float32)
        for candidate in (reference[:, :, :, :3], reference.astype(np.float64)):
            result = study.compare_arrays(reference, candidate)
            self.assertFalse(result["passed"])
            self.assertEqual(result["reason"], "shape_or_dtype_mismatch")


class EventSummaryTests(unittest.TestCase):
    def test_overlap_is_unioned_within_kind_and_nested_kinds_are_not_added(self):
        row = dict(events=[
            dict(kind="video_finish", start_ms=0., end_ms=10.),
            dict(kind="video_decode", start_ms=0., end_ms=7.),
            dict(kind="video_convert", start_ms=7., end_ms=10.),
            dict(kind="video_quantum", start_ms=0., end_ms=4.),
            dict(kind="video_quantum", start_ms=2., end_ms=6.),
        ], requests=[dict(kind="image", request_id="i0", arrival_ms=2., start_ms=9.),
                     dict(kind="video", request_id="v0", arrival_ms=0., start_ms=0.),
                     dict(kind="image", request_id="failed", arrival_ms=2., start_ms=None)])
        raw = copy.deepcopy(row["events"])
        result = study.event_summary(row)
        self.assertIs(result, row)
        self.assertEqual(row["events"], raw)
        self.assertEqual(row["event_summary"]["total_events"], 5)
        quantum = row["event_summary"]["by_kind"]["video_quantum"]
        self.assertEqual(quantum["sum_ms"], 8.)
        self.assertEqual(quantum["mean_ms"], 4.)
        self.assertEqual(quantum["p95_ms"], 4.)
        wait = row["requests"][0]["wait_attribution"]
        self.assertEqual(wait["wait_ms"], 7.)
        self.assertEqual(wait["by_event_kind_ms"], dict(video_finish=7., video_decode=5.,
                                                       video_convert=2., video_quantum=4.))
        self.assertIn("must not be summed", wait["scope"])
        self.assertNotIn("total_overlap_ms", wait)
        self.assertNotIn("wait_attribution", row["requests"][1])
        self.assertNotIn("wait_attribution", row["requests"][2])

    def test_missing_durations_are_counted_without_fabricated_zero_samples(self):
        row = dict(events=[dict(kind="marker", start_ms=2.),
            dict(kind="marker", start_ms=5., end_ms=4.),
            dict(kind="valid", start_ms=1., end_ms=3.)], requests=[])
        study.event_summary(row)
        marker = row["event_summary"]["by_kind"]["marker"]
        self.assertEqual(marker["count"], 2)
        self.assertEqual(marker["n"], 0)
        self.assertEqual(marker["missing_duration"], 2)
        self.assertIsNone(marker["mean_ms"])
        self.assertIsNone(marker["p95_ms"])
        self.assertIsNone(marker["max_ms"])


class FrozenProtocolTests(unittest.TestCase):
    def test_real_readonly_quick_traces_have_one_seed_eight_replays_192_requests(self):
        path = study.ROOT / "configs/serving_next_validation.json"
        if not path.is_file():
            self.skipTest("historical fixed workload config unavailable")
        config = read_json(path)
        if not Path(config["old_online"]).is_file():
            self.skipTest("historical fixed workload raw source unavailable")
        traces = study.quick_traces(config)
        self.assertEqual(len(traces), 2)
        self.assertEqual({row["seed"] for row in traces}, {101})
        self.assertEqual({row["load"] for row in traces}, {.5})
        self.assertEqual({row["pattern"] for row in traces}, {"ordinary", "burst"})
        self.assertEqual(len(traces) * 2 * 2, 8)
        self.assertEqual(sum(len(row["requests"]) for row in traces) * 2 * 2, 192)
        for trace in traces:
            self.assertEqual(len(trace["requests"]), 24)
            self.assertEqual(request_identity(trace["requests"]), trace["trace_sha256"])

    def test_quick_trace_missing_or_duplicate_pattern_is_rejected(self):
        for traces in ([dict(seed=101, pattern="ordinary")],
                       [dict(seed=101, pattern="ordinary"), dict(seed=101, pattern="ordinary")]):
            with patch("ditserve.optimizations.next_workloads.build_workloads",
                       return_value={"highload": {"0.5": traces}}):
                with self.assertRaisesRegex(ValueError, "pair absent"):
                    study.quick_traces({})

    def test_fixed_k1_is_routed_through_the_actual_online_dispatch_helper(self):
        from ditserve.directions.direction1 import _advance_policy
        executor_type = getattr(study, "FollowupPairedExecutor", None)
        self.assertIsNotNone(executor_type, "K1 online requires a follow-up executor quantum override")
        executor = executor_type.__new__(executor_type)
        calls = []
        executor.advance = lambda state, blocks, **kwargs: calls.append(blocks)
        for policy in ("block-1", "block-4"):
            _advance_policy(executor, object(), policy, lambda: None)
        self.assertEqual(calls, [1, 4])
        with self.assertRaises(ValueError):
            executor.quantum(object(), "block-8")


if __name__ == "__main__":
    unittest.main()
