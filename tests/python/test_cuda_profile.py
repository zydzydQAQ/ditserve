"""CPU fixtures for trustworthy kernel-only Chrome trace aggregation."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ditserve.evaluation.cuda_profile import summarize_trace, summarize_trace_events


def event(cat, name, start, duration, *, pid=100, tid=1, **args):
    return {"cat": cat, "name": name, "ph": "X", "ts": start, "dur": duration,
            "pid": pid, "tid": tid, "args": args}


def scope(name, start, duration, **kwargs):
    return event("user_annotation", "wan_profile/" + name, start, duration, **kwargs)


def cpu(start, external_id, duration=1):
    return event("cpu_op", "aten::linear", start, duration, **{"External id": external_id})


def runtime(start, correlation, external_id=None, **kwargs):
    args = {"correlation": correlation}
    if external_id is not None:
        args["External id"] = external_id
    return event("cuda_runtime", "cudaLaunchKernel", start, 0.2, **kwargs, **args)


def kernel(start, duration, *, name="gemm", correlation=11, external_id=101, stream=7, device=0):
    args = {"correlation": correlation, "device": device, "stream": stream}
    if external_id is not None:
        args["External id"] = external_id
    return event("kernel", name, start, duration, pid=device, tid=stream, **args)


def fixture():
    return [scope("request", 0, 200), scope("denoise", 5, 145),
            scope("transformer", 10, 90), scope("self_attention", 20, 20),
            scope("ffn", 50, 30), cpu(22, 101), runtime(22.1, 11, 101),
            cpu(52, 102), runtime(52.1, 12, 102)]


def category(summary, name, key="kernel_categories"):
    return next(row for row in summary[key] if row["name"] == name)


class CudaProfileTests(unittest.TestCase):
    def test_gpu_annotations_and_parent_cpu_ranges_never_become_kernels(self):
        rows = fixture() + [kernel(100, 20), kernel(130, 10, correlation=12, external_id=102),
            event("gpu_user_annotation", "wan_profile/request", 0, 200, pid=0, tid=7),
            event("gpu_user_annotation", "wan_profile/transformer", 90, 80, pid=0, tid=7),
            event("gpu_user_annotation", "wan_profile/self_attention", 100, 20, pid=0, tid=7)]
        result = summarize_trace_events(rows)
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["diagnostics"]["kernel_count"], 2)
        self.assertEqual(result["diagnostics"]["excluded_gpu_annotation_count"], 3)
        self.assertAlmostEqual(result["kernel_duration_ms"], 0.030)
        self.assertAlmostEqual(result["request"]["kernel_active_pct"], 15.0)
        self.assertAlmostEqual(result["gpu_window"]["kernel_active_pct"], 75.0)
        self.assertAlmostEqual(category(result, "self_attention")["share_pct"], 200 / 3)
        self.assertAlmostEqual(category(result, "ffn")["share_pct"], 100 / 3)
        self.assertEqual(result["kernel_duration_share_sum_pct"], 100.0)
        self.assertFalse(any(row["name"].startswith("wan_profile/") for row in result["top_kernels"]))

    def test_async_launch_is_assigned_by_link_not_gpu_timestamp(self):
        rows = fixture() + [kernel(60, 10)]  # GPU runs during CPU's FFN scope.
        result = summarize_trace_events(rows)
        self.assertEqual(category(result, "self_attention")["kernel_count"], 1)
        self.assertEqual(category(result, "ffn")["kernel_count"], 0)

    def test_duplicate_links_and_inclusive_ancestors_do_not_duplicate_work(self):
        rows = fixture()
        rows.append(copy.deepcopy(rows[6]))  # Duplicate CUDA-runtime link.
        rows.append(copy.deepcopy(rows[5]))  # Duplicate CPU External id link.
        rows.append(kernel(100, 20))
        result = summarize_trace_events(rows)
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["diagnostics"]["kernel_count"], 1)
        self.assertAlmostEqual(result["kernel_duration_ms"], 0.02)
        self.assertEqual(category(result, "self_attention")["share_pct"], 100.0)
        self.assertEqual(category(result, "transformer_other")["share_pct"], 0.0)
        stages = {row["name"]: row for row in result["stages"]}
        self.assertEqual(stages["transformer"]["kernel_union_ms"], 0.02)
        self.assertEqual(stages["self_attention"]["kernel_union_ms"], 0.02)

    def test_cross_stream_overlap_has_separate_sum_and_union_denominators(self):
        rows = fixture() + [kernel(100, 20), kernel(110, 20, correlation=12, external_id=102, stream=8)]
        result = summarize_trace_events(rows)
        self.assertEqual(result["status"], "measured")
        self.assertTrue(result["diagnostics"]["multistream"])
        self.assertEqual(result["request"]["kernel_union_ms"], 0.03)
        self.assertEqual(result["kernel_duration_ms"], 0.04)
        self.assertEqual(result["kernel_union_partition"]["cross_category_overlap_ms"], 0.01)
        self.assertEqual(category(result, "self_attention")["exclusive_union_ms"], 0.01)
        self.assertEqual(category(result, "ffn")["exclusive_union_ms"], 0.01)
        partition = result["kernel_union_partition"]
        self.assertAlmostEqual(partition["exclusive_category_share_sum_pct"] +
                               partition["cross_category_overlap_share_pct"], 100.0)
        self.assertEqual(category(result, "ffn")["share_pct"], 50.0)

    def test_unattributed_kernel_stays_explicit_and_exact_share_is_unverified(self):
        result = summarize_trace_events(fixture() + [kernel(100, 20, external_id=None, correlation=999)])
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["timeline_status"], "measured")
        self.assertEqual(result["attribution_status"], "unverified")
        self.assertEqual(result["request"]["kernel_active_pct"], 10.0)
        self.assertEqual(category(result, "unattributed")["share_pct"], 100.0)
        self.assertEqual(result["diagnostics"]["unattributed_kernel_count"], 1)
        self.assertIn("unattributed_kernels", result["reasons"])
        self.assertIsNone(next(row for row in result["stages"] if row["name"] == "transformer")["kernel_active_pct"])

    def test_conflicting_external_and_runtime_links_are_ambiguous(self):
        result = summarize_trace_events(fixture() + [kernel(100, 20, external_id=101, correlation=12)])
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["diagnostics"]["ambiguous_kernel_count"], 1)
        self.assertEqual(category(result, "unattributed")["kernel_count"], 1)
        self.assertEqual(category(result, "self_attention")["kernel_count"], 0)

    def test_crossing_or_equal_annotation_scopes_are_ambiguous(self):
        for extra in (scope("ffn", 21, 30), scope("cross_attention", 20, 20)):
            result = summarize_trace_events(fixture() + [extra, kernel(100, 20)])
            self.assertEqual(result["diagnostics"]["ambiguous_kernel_count"], 1)
            self.assertEqual(result["attribution_status"], "unverified")

    def test_runtime_only_link_is_valid_and_uses_same_cpu_thread(self):
        result = summarize_trace_events(fixture() + [kernel(100, 20, external_id=None)])
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["diagnostics"]["kernel_link_methods"], {"runtime": 1})
        rows = fixture() + [runtime(22.1, 15, None, tid=999),
                            kernel(100, 20, correlation=15, external_id=None)]
        result = summarize_trace_events(rows)
        self.assertEqual(result["attribution_status"], "unverified")
        self.assertEqual(category(result, "unattributed")["kernel_count"], 1)

    def test_memcpy_and_memset_are_activity_but_not_kernel_or_ffn_work(self):
        rows = fixture() + [kernel(100, 20),
            event("gpu_memcpy", "Memcpy HtoD", 125, 5, pid=0, tid=7,
                  device=0, stream=7, correlation=12, **{"External id": 102}),
            event("gpu_memset", "Memset", 140, 5, pid=0, tid=7,
                  device=0, stream=7, correlation=12, **{"External id": 102})]
        result = summarize_trace_events(rows)
        self.assertEqual(result["diagnostics"]["kernel_count"], 1)
        self.assertEqual(result["diagnostics"]["memcpy_count"], 1)
        self.assertEqual(result["diagnostics"]["memset_count"], 1)
        self.assertEqual(result["request"]["kernel_active_pct"], 10.0)
        self.assertEqual(result["request"]["activity_active_pct"], 15.0)
        self.assertEqual(category(result, "ffn")["kernel_duration_ms"], 0.0)

    def test_unattributed_memory_invalidates_stage_activity_but_not_kernel_attribution(self):
        for memory_type in ("gpu_memcpy", "gpu_memset"):
            with self.subTest(memory_type=memory_type):
                rows = fixture() + [kernel(100, 20),
                    event(memory_type, "unlinked memory operation", 125, 5, pid=0, tid=7,
                          device=0, stream=7, correlation=999)]
                result = summarize_trace_events(rows)
                self.assertEqual(result["status"], "unverified")
                self.assertEqual(result["timeline_status"], "measured")
                self.assertEqual(result["kernel_attribution_status"], "measured")
                self.assertEqual(result["activity_attribution_status"], "unverified")
                self.assertEqual(result["diagnostics"]["unattributed_kernel_count"], 0)
                self.assertEqual(result["diagnostics"]["unattributed_gpu_event_count"], 1)
                self.assertEqual(result["request"]["activity_active_pct"], 12.5)
                self.assertEqual(category(result, "self_attention")["share_pct"], 100.0)
                transformer = next(row for row in result["stages"] if row["name"] == "transformer")
                self.assertEqual(transformer["status"], "unverified")
                self.assertEqual(transformer["kernel_attribution_status"], "measured")
                # Its GPU window also depends on the unknown memory operation;
                # even the kernel/window ratio therefore lacks a valid denominator.
                self.assertEqual(transformer["window_status"], "unverified")
                self.assertIsNone(transformer["kernel_active_pct"])
                self.assertIsNone(transformer["activity_active_pct"])
                self.assertTrue(all(row["activity_active_pct"] is None for row in result["stages"]))

    def test_missing_or_malformed_timeline_does_not_become_zero_utilization(self):
        fixtures = [fixture(), [kernel(100, 20)], fixture() + [{"cat": "kernel", "name": "bad", "ph": "X"}]]
        for rows in fixtures:
            result = summarize_trace_events(rows)
            self.assertEqual(result["status"], "unverified")
            self.assertIsNone(result["request"]["kernel_active_pct"])
        result = summarize_trace_events(fixture() + [kernel(100, 20), kernel(130, float("nan"))])
        self.assertEqual(result["diagnostics"]["invalid_gpu_timeline_count"], 1)
        self.assertIsNone(result["request"]["kernel_active_pct"])

    def test_request_boundary_and_multiple_gpu_are_unverified(self):
        for extra in (kernel(190, 20), kernel(100, 20, device=1)):
            result = summarize_trace_events(fixture() + [kernel(100, 20), extra])
            self.assertEqual(result["timeline_status"], "unverified")
            self.assertIsNone(result["request"]["kernel_active_pct"])
        result = summarize_trace_events(fixture() + [kernel(100, 20), kernel(300, 20)])
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["diagnostics"]["gpu_events_outside_request"], 1)
        self.assertEqual(result["diagnostics"]["kernel_count"], 1)

    def test_transformer_denominator_excludes_vae(self):
        rows = fixture() + [scope("vae", 160, 30), cpu(162, 103), runtime(162.1, 13, 103),
            kernel(100, 20), kernel(130, 10, correlation=12, external_id=102),
            kernel(170, 30, correlation=13, external_id=103)]
        result = summarize_trace_events(rows)
        self.assertEqual(result["status"], "measured")
        self.assertEqual(category(result, "vae")["share_pct"], 50.0)
        self.assertAlmostEqual(category(result, "ffn", "transformer_kernel_categories")["share_pct"], 100 / 3)
        self.assertEqual(result["transformer_kernel_duration_ms"], 0.03)

    def test_bounded_top_kernels_preserve_omitted_totals(self):
        rows = fixture() + [kernel(100, 2, name="a"), kernel(110, 8, name="b"), kernel(120, 4, name="c")]
        result = summarize_trace_events(rows, top_kernel_limit=1)
        self.assertEqual([row["name"] for row in result["top_kernels"]], ["b"])
        self.assertEqual(result["top_kernels_retention"]["omitted_name_count"], 2)
        self.assertEqual(result["top_kernels_retention"]["omitted_kernel_count"], 2)
        self.assertEqual(result["top_kernels_retention"]["omitted_kernel_duration_ms"], 0.006)
        self.assertAlmostEqual(result["kernel_duration_ms"], 0.014)

    def test_file_loader_keeps_units_and_hash_and_does_not_modify_source(self):
        payload = {"displayTimeUnit": "ms", "traceEvents": fixture() + [kernel(100, 20)]}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.json"
            encoded = json.dumps(payload).encode()
            path.write_bytes(encoded)
            result = summarize_trace(path)
            self.assertEqual(result["request"]["wall_ms"], 0.2)
            self.assertEqual(result["trace_source"]["sha256"], hashlib.sha256(encoded).hexdigest())
            self.assertEqual(result["trace_source"]["bytes"], len(encoded))
            self.assertEqual(path.read_bytes(), encoded)
            self.assertEqual(summarize_trace(payload)["request"], result["request"])

    def test_unknown_gpu_category_is_diagnostic_not_assumed_kernel(self):
        rows = fixture() + [kernel(100, 20), event("gpu_future_activity", "unknown", 120, 20, pid=0, tid=7)]
        result = summarize_trace_events(rows)
        self.assertEqual(result["diagnostics"]["kernel_count"], 1)
        self.assertEqual(result["timeline_status"], "unverified")
        self.assertEqual(result["diagnostics"]["unknown_gpu_categories"], {"gpu_future_activity": 1})


if __name__ == "__main__":
    unittest.main()
