"""Telemetry arithmetic must not overstate overlap or substitute busy for SM."""

import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.evaluation.workflow_metrics import (
    PhaseRecorder, interval_intersection, interval_union, overlap_summary,
    process_snapshot, summarize_nvml_samples, summarize_phases, torch_snapshot,
)


class WorkflowIntervalTests(unittest.TestCase):
    def test_union_and_intersection_do_not_double_count_concurrent_events(self):
        compute = [(0, 10), (5, 15), (0, 10), (20, 25)]
        communication = [(8, 12), (10, 22)]
        self.assertEqual(interval_union(compute), [(0, 15), (20, 25)])
        self.assertEqual(interval_intersection(compute, communication), [(8, 15), (20, 22)])
        summary = overlap_summary(compute, communication)
        self.assertEqual(summary["compute_ns"], 20)
        self.assertEqual(summary["communication_ns"], 14)
        self.assertEqual(summary["overlap_ns"], 9)
        self.assertEqual(summary["exposed_communication_ns"], 5)
        self.assertAlmostEqual(summary["communication_coverage"], 9 / 14)

    def test_empty_and_touching_intervals(self):
        self.assertEqual(interval_union([(5, 5), (0, 5), (5, 7)]), [(0, 7)])
        self.assertEqual(interval_intersection([(0, 5)], [(5, 7)]), [])
        summary = overlap_summary([], [])
        self.assertIsNone(summary["communication_coverage"])
        self.assertEqual(summary["overlap_ns"], 0)
        self.assertEqual(overlap_summary([], [(0, 10)])["communication_coverage"], 0)

    def test_invalid_or_nonfinite_intervals_are_rejected(self):
        for interval in ((2, 1), (0, float("nan")), (0, float("inf")), (True, 2), ("0", 1), (0, 1, 2)):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                interval_union([interval])

    def test_large_integer_nanosecond_timestamps_remain_exact(self):
        start = 2 ** 60
        summary = overlap_summary([(start, start + 7)], [(start + 3, start + 8)])
        self.assertEqual(summary["overlap_ns"], 4)
        self.assertEqual(summary["communication_ns"], 5)


class WorkflowSampleTests(unittest.TestCase):
    def samples(self):
        return [dict(timestamp_ns=t, uuid="GPU-test", gpu_busy_percent=value,
                     memory_used_bytes=value * 10, memory_total_bytes=1000,
                     compute_processes=[dict(pid=42)], graphics_processes=[])
                for t, value in ((0, 10), (10, 50), (30, 90))]

    def test_weighted_mean_clips_window_and_does_not_extrapolate(self):
        summary = summarize_nvml_samples(self.samples(), 5, 40)["GPU-test"]
        self.assertEqual(summary["covered_ns"], 25)
        self.assertEqual(summary["window_ns"], 35)
        self.assertEqual(summary["gpu_busy_percent_observed_ns"], 25)
        self.assertEqual(summary["gpu_busy_percent_time_mean"], 42)
        self.assertEqual(summary["memory_used_bytes_observed_peak"], 900)
        self.assertEqual(summary["compute_pids_seen"], [42])
        self.assertFalse(any("sm" in key or "occupancy" in key for key in summary))

    def test_single_sample_does_not_invent_a_time_average(self):
        summary = summarize_nvml_samples(self.samples()[:1], 0, 100)["GPU-test"]
        self.assertIsNone(summary["gpu_busy_percent_time_mean"])
        self.assertEqual(summary["covered_ns"], 0)
        self.assertEqual(summary["memory_used_bytes_observed_peak"], 100)

    def test_missing_fields_are_not_zeroes(self):
        samples = self.samples()
        del samples[1]["gpu_busy_percent"]
        summary = summarize_nvml_samples(samples)["GPU-test"]
        self.assertEqual(summary["gpu_busy_percent_time_mean"], 10)
        self.assertEqual(summary["gpu_busy_percent_observed_ns"], 10)
        self.assertIsNone(summary["memory_io_busy_percent_time_mean"])


class WorkflowPhaseTests(unittest.TestCase):
    def test_cpu_phase_records_exception_without_masking_it(self):
        recorder = PhaseRecorder(device="cpu", nvtx=False)
        with self.assertRaisesRegex(RuntimeError, "workload failed"):
            with recorder.phase("encode", request_id="r0"):
                raise RuntimeError("workload failed")
        row = recorder.records[0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["metadata"], dict(request_id="r0"))
        self.assertGreaterEqual(row["wall_ms"], 0)
        self.assertGreaterEqual(row["process_cpu_ms"], 0)
        self.assertEqual(row["timing_scope"], "host_interval")
        self.assertFalse(row["torch_after"]["cuda_initialized"])
        json.dumps(row, allow_nan=False)
        self.assertEqual(summarize_phases(recorder.records)["encode"]["failed"], 1)

    def test_peak_resets_require_isolation_and_disallow_nested_ranges(self):
        with self.assertRaises(ValueError):
            PhaseRecorder(reset_peaks=True)
        recorder = PhaseRecorder(device="cpu", synchronize=True, reset_peaks=True, nvtx=False)
        with recorder.phase("outer"):
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with recorder.phase("inner"):
                    pass
        self.assertEqual(len(recorder.records), 1)

    def test_nvtx_cpu_ranges_do_not_require_cuda_initialization(self):
        calls = []
        fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False,
                               nvtx=SimpleNamespace(range_push=lambda value: calls.append(value),
                                                    range_pop=lambda: calls.append("pop"))))
        with patch.dict(sys.modules, torch=fake):
            with PhaseRecorder(device="cpu").phase("cpu_encode", variant="中文", purpose="warmup", repeat=0):
                pass
            self.assertEqual(torch_snapshot(), dict(cuda_initialized=False))
        self.assertEqual(calls, ['ditserve.workflow/cpu_encode|{"purpose":"warmup","repeat":0,"variant":"\\u4e2d\\u6587"}', "pop"])

    def test_nonprimitive_or_nonfinite_phase_metadata_is_rejected(self):
        recorder = PhaseRecorder(device="cpu", nvtx=False)
        for metadata in (dict(variant=["a", "b"]), dict(repeat=float("nan"))):
            with self.subTest(metadata=metadata), self.assertRaises((ValueError, TypeError)):
                with recorder.phase("invalid", **metadata):
                    pass
        self.assertEqual(recorder.records, [])

    def test_process_snapshot_contains_current_process(self):
        snapshot = process_snapshot()
        self.assertIn(snapshot["pid"], snapshot["pids"])
        self.assertGreater(snapshot["rss_bytes"], 0)
        self.assertGreaterEqual(snapshot["tree_cpu_s"], snapshot["process_cpu_s"])
        json.dumps(snapshot, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
