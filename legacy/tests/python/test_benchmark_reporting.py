"""Denominators, complete paired attempts, immutable evidence, and exports."""

import copy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

from ditserve.benchmark.common import atomic_json, file_hash, read_json, seal
from ditserve.benchmark.reporting import paired_improvements, summarize_requests, write_report


def request(identity="r", **updates):
    value = {"request_id": identity, "shape_id": "small", "status": "completed", "arrival_ms": 10.,
             "submit_ms": 12., "start_ms": 15., "finish_ms": 30., "deadline_ms": 35.,
             "generation": {"prompt": identity, "seed": 0, "steps": 2}, "events": [], "artifact": None}
    value.update(updates)
    return value


class ServingStatisticsTests(unittest.TestCase):
    def test_denominator_keeps_failed_pending_late_and_excludes_future(self):
        rows = [request("ok"), request("late", finish_ms=50), request("fail", status="failed", finish_ms=None),
                request("pending", status="queued", start_ms=None, finish_ms=None),
                request("future", status="not_arrived", arrival_ms=200, submit_ms=None, start_ms=None, finish_ms=None)]
        qualities = {"ok": {"status": "pass", "qualified": True}, "late": {"status": "unknown", "qualified": False}}
        result = summarize_requests(rows, 100., qualities)
        self.assertEqual(result["arrivals"], 4)
        self.assertEqual(result["not_arrived"], 1)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["completion_rate"], .5)
        self.assertEqual(result["deadline_completion_rate"], .25)
        self.assertEqual(result["completed_latency_ms"]["samples"], 2)
        self.assertEqual(result["completed_latency_ms"]["mean"], 30.)
        self.assertEqual(result["deadline_goodput_per_s"], 10.)
        self.assertEqual(result["quality_qualified_goodput_per_s"], 10.)
        self.assertEqual(result["submit_delay_ms"]["samples"], 4)
        self.assertEqual(result["queue_ms"]["samples"], 3)
        self.assertEqual(result["failure_rate"], .25)
        self.assertEqual(result["rejection_rate"], 0.)
        self.assertEqual(result["timeout_rate"], 0.)

    def test_rejected_timed_out_and_aborted_rates_use_all_arrivals(self):
        rows = [request("done"), request("rejected", status="resource_rejected"),
                request("timeout", status="timed_out"), request("abort", status="arrived_aborted"),
                request("future", status="not_arrived")]
        result = summarize_requests(rows, 100)
        self.assertEqual(result["arrivals"], 4)
        self.assertEqual(result["rejection_rate"], .25)
        self.assertEqual(result["timeout_rate"], .25)
        self.assertEqual(result["failure_rate"], .25)
        self.assertEqual(result["failure_rate_statuses"], ["failed", "arrived_aborted"])

    def test_zero_horizon_and_missing_latency_are_unknown_not_zero(self):
        result = summarize_requests([request(finish_ms=None)], 0.)
        self.assertIsNone(result["throughput_per_s"])
        self.assertEqual(result["completed_missing_latency"], 1)
        self.assertEqual(result["completed_latency_ms"]["samples"], 0)
        self.assertIsNone(result["completed_latency_ms"]["p99"])

    def test_partial_attempt_cannot_claim_paired_improvement(self):
        left = {"status": "completed", "horizon_ms": 40, "requests": [request()]}
        right = {"status": "completed", "horizon_ms": 30, "requests": [request(finish_ms=20)]}
        run = {"status": "completed", "trace_sha256": "same", "policies": {"fcfs": left, "candidate": right}}
        summaries = {name: summarize_requests(value["requests"], value["horizon_ms"]) for name, value in run["policies"].items()}
        result = paired_improvements(run, summaries)[0]
        self.assertEqual(result["status"], "included")
        self.assertEqual(result["completed_mean_latency_improvement_fraction"], .5)
        self.assertEqual(paired_improvements(run, summaries, [request(), request("missing")])[0]["reason"], "incomplete_trace_coverage")
        run["status"] = "partial"
        result = paired_improvements(run, summaries)[0]
        self.assertEqual(result["status"], "excluded")
        self.assertIsNone(result["completed_mean_latency_improvement_fraction"])
        run["status"] = "completed"
        right["requests"][0]["generation"]["seed"] = 5
        self.assertEqual(paired_improvements(run, summaries)[0]["reason"], "generation_identity_missing_or_mismatch")

    def test_known_synthetic_evidence_is_excluded_with_specific_reason_precedence(self):
        policy = {"status": "completed", "horizon_ms": 40, "requests": [request()]}
        base = {"status": "completed", "trace_sha256": "same", "purpose": "online_serving", "serving_performance_evidence": True,
                "policies": {"fcfs": policy, "candidate": copy.deepcopy(policy)}}
        summaries = {name: summarize_requests(value["requests"], value["horizon_ms"]) for name, value in base["policies"].items()}
        for marker in ({"backend": "cpu-test"}, {"backend": "core"}, {"measurement_kind": "synthetic_fixture"}, {"measured": False}):
            run = dict(base, **marker)
            with self.subTest(marker=marker):
                self.assertEqual(paired_improvements(run, summaries)[0]["reason"], "synthetic_backend")
                run.update(purpose="independent_quality_generation", serving_performance_evidence=False)
                self.assertEqual(paired_improvements(run, summaries)[0]["reason"], "independent_quality_generation")
                run["purpose"] = "offline_diagnostic"
                self.assertEqual(paired_improvements(run, summaries)[0]["reason"], "offline_diagnostic")
        self.assertEqual(paired_improvements(base, summaries)[0]["status"], "included")


class ReportExportTests(unittest.TestCase):
    def make_run(self, root):
        trace = seal({"schema_version": "ditserve.benchmark.trace.v1", "requests": [request()]})
        atomic_json(root / "trace.json", trace)
        run = seal({"schema_version": "ditserve.benchmark.run.v1", "trace_sha256": trace["sha256"],
                    "backend": "cpu-test", "measurement_kind": "synthetic_fixture", "status": "completed",
                    "policies": {"fcfs": {"status": "completed", "horizon_ms": 100, "requests": [request(), request("failed", status="failed", finish_ms=None)]}}})
        atomic_json(root / "run.json", run)
        return run

    def test_report_keeps_failures_and_builds_real_excel_without_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_run(root)
            before = file_hash(root / "run.json")
            report = write_report(root)
            self.assertEqual(file_hash(root / "run.json"), before)
            self.assertEqual(report["summary"]["policies"]["fcfs"]["arrivals"], 2)
            self.assertEqual(report["summary"]["policies"]["fcfs"]["quality_status_counts"], {"unknown": 2})
            self.assertIs(report["summary"]["serving_performance_evidence"], False)
            output = Path(report["output_dir"])
            for filename in ("summary.json", "summary.csv", "requests.csv", "by_shape.csv", "results.xlsx", "REPORT.md", "completion.svg", "deadline_goodput.svg"):
                self.assertTrue((output / filename).is_file(), filename)
            with zipfile.ZipFile(output / "results.xlsx") as archive:
                self.assertIn("xl/workbook.xml", archive.namelist())
                for filename in archive.namelist():
                    ET.fromstring(archive.read(filename))
            self.assertEqual(file_hash(output / "evidence" / "run.json"), before)
            self.assertIn("未完成", (output / "REPORT.md").read_text())
            ET.parse(output / "completion.svg")
            rebuilt = write_report(root)
            self.assertEqual(rebuilt["summary"]["policies"], report["summary"]["policies"])
            self.assertTrue((output / "report_history" / report["summary"]["sha256"] / "summary.json").is_file())

    def test_stale_evaluation_and_trace_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root)
            atomic_json(root / "evaluation.json", seal({"run_sha256": "wrong", "run_file_sha256": file_hash(root / "run.json")}))
            with self.assertRaises(ValueError):
                write_report(root)
            (root / "evaluation.json").unlink()
            trace = read_json(root / "trace.json")
            trace["requests"][0]["seed"] = 3
            atomic_json(root / "trace.json", seal(trace))
            with self.assertRaises(ValueError):
                write_report(root)

    def test_independent_quality_generation_excludes_all_performance_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root)
            trace = read_json(root / "trace.json")
            trace.update(purpose="independent_quality_generation", serving_performance_evidence=False)
            trace = seal(trace)
            atomic_json(root / "trace.json", trace)
            run.update(trace_sha256=trace["sha256"], purpose=trace["purpose"], serving_performance_evidence=False)
            run["policies"]["candidate"] = copy.deepcopy(run["policies"]["fcfs"])
            atomic_json(root / "run.json", seal(run))
            result = write_report(root)
            self.assertEqual(result["summary"]["purpose"], "independent_quality_generation")
            self.assertIs(result["summary"]["serving_performance_evidence"], False)
            pair = result["summary"]["paired_improvements"][0]
            self.assertEqual(pair["status"], "excluded")
            self.assertEqual(pair["reason"], "independent_quality_generation")
            self.assertIsNone(pair["completed_mean_latency_improvement_fraction"])
            self.assertIn("不是在线服务性能证据", Path(result["report_path"]).read_text())

    def test_offline_diagnostic_has_its_own_exclusion_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root)
            trace = read_json(root / "trace.json")
            trace.update(purpose="offline_diagnostic", serving_performance_evidence=False)
            trace = seal(trace)
            atomic_json(root / "trace.json", trace)
            run.update(trace_sha256=trace["sha256"], purpose="offline_diagnostic", serving_performance_evidence=False)
            run["policies"]["candidate"] = copy.deepcopy(run["policies"]["fcfs"])
            atomic_json(root / "run.json", seal(run))
            result = write_report(root)
            pair = result["summary"]["paired_improvements"][0]
            self.assertEqual(pair["status"], "excluded")
            self.assertEqual(pair["reason"], "offline_diagnostic")
            self.assertIsNone(pair["completed_mean_latency_improvement_fraction"])
            text = Path(result["report_path"]).read_text()
            self.assertIn("仅用于离线诊断", text)
            self.assertIn("不用于声明在线服务性能提升", text)
            self.assertNotIn("本次为独立质量评价", text)


if __name__ == "__main__":
    unittest.main()
