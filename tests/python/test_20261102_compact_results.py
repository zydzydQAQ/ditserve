"""2026-11-02: compact publication preserves evidence and cleanup boundaries."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve import results


class CompactResultsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.work, self.output = self.root / "work", self.root / "results"
        self.work.mkdir()
        self.run = dict(schema="ditserve.run.v1", experiment="serving", run_id="policy-study",
            status="completed", config={"workers": {"video": {"backend": "wan", "config": {
                "model_path": "wan21", "revision": "frozen-revision", "dtype": "float16"}}}},
            source={"core_sha256": "a" * 64, "files": {"large-source-manifest": "b" * 64}},
            metrics={"requests": 2, "completed": 1, "failed": 1, "qualified_goodput_rps": None},
            phases=[dict(id="cluster", status="completed", attempt=0,
                qualification={"scopes": {"performance": {"status": "failed"}}},
                cells=[dict(cell=0, policy="ditserve", status="completed", gpus=8,
                    metrics={"requests": 2, "latency_p95_ms": 1.2345678901234567},
                    doorbell={"rings": 3}, by_kind={"image": {"attainment": .5}},
                    requests=[{"request_id": "r1", "events": ["huge" * 1000]}])])],
            requests=[{"request_id": "r1", "status": "failed", "error": "out of memory"},
                      {"request_id": "r2", "events": ["huge" * 1000] * 30}],
            budget={"active": None, "used_gpu_seconds": 5.0,
                "sessions": [{"ended_at": 1793577600, "identity": {"gpu_uuids": ["GPU-1"]}}]},
            failures=[], artifacts=[], retention={"pending_dependencies": []})

    def write(self):
        (self.work / "run.json").write_text(json.dumps(self.run))

    def raw(self, name, dependencies=None, kind="raw"):
        path = self.work / name
        path.write_bytes(b"raw evidence")
        self.run["artifacts"].append(dict(path=name, bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), kind=kind,
            dependencies=dependencies or []))
        return path

    def finalize(self, **kwargs):
        self.write()
        return results.finalize(self.work, self.output, **kwargs)

    def test_default_has_two_small_files_and_never_loads_excel(self):
        self.write()
        original = (self.work / "run.json").read_bytes()
        with patch.object(results, "_excel", side_effect=AssertionError("no Excel dependency")):
            outcome = results.finalize(self.work, self.output)
            self.assertEqual(outcome["status"], "completed", outcome)
            saved = results.load(self.output)
            self.assertEqual(results.verify(self.output)["format"], results.SUMMARY_SCHEMA)
        self.assertEqual({p.name for p in self.output.iterdir()}, {"summary.json", "20261102-policy-study.md"})
        self.assertEqual((self.work / "run.json").read_bytes(), original)
        self.assertLess((self.output / "summary.json").stat().st_size, len(original) // 10)
        self.assertNotIn("requests", saved["run"])
        self.assertNotIn("config", saved["run"])
        self.assertNotIn("files", saved["run"]["source"])
        self.assertEqual(saved["run"]["config_content_sha256"], results._digest(self.run["config"]))
        self.assertEqual(saved["run"]["models"][0]["config"]["revision"], "frozen-revision")
        cell = saved["run"]["phases"][0]["cells"][0]
        self.assertNotIn("requests", cell)
        self.assertEqual(cell["metrics"]["latency_p95_ms"], 1.2345678901234567)
        self.assertEqual(cell["doorbell"]["rings"], 3)
        self.assertEqual(saved["run"]["failures"][0]["error"], "out of memory")
        self.assertFalse(saved["run"]["summary_limits"]["raw_re_evaluable"])
        self.assertIsNone(saved["run"]["metrics"]["qualified_goodput_rps"])
        report = Path(outcome["report"]).read_text()
        self.assertIn("performance=failed", report)
        self.assertIn("不能用于续跑或原始重评", report)

    def test_partial_cleanup_only_allows_explicit_dependency_closure(self):
        closed = self.raw("closed.npy")
        pending = self.raw("pending.npy", ["quality-evaluation"])
        self.assertEqual(self.finalize()["status"], "deferred")
        self.assertFalse(closed.exists())
        self.assertTrue(pending.exists())
        self.run["metrics"]["completed"] = 999
        self.assertEqual(self.finalize()["status"], "deferred")
        self.assertTrue(pending.exists())
        self.run["metrics"]["completed"] = 1
        self.run["artifacts"][1]["dependencies"] = [{"closed": True, "reason": "quality evaluation saved"}]
        self.assertEqual(self.finalize()["status"], "completed")
        self.assertFalse(pending.exists())
        saved = results.load(self.output)
        self.assertEqual(len(saved["finalization"]["deleted"]), 2)
        self.assertEqual(saved["finalization"]["input_revisions"][0]["dependency_closures"][0]["previous"],
                         ["quality-evaluation"])

    def test_crash_after_unlink_recovers_from_summary_bound_intent(self):
        raw = self.raw("sample.npy")
        self.write()
        original = results._save
        def crash(work, journal):
            if journal.get("deleted"):
                raise KeyboardInterrupt("crash after unlink")
            original(work, journal)
        with patch.object(results, "_save", crash), self.assertRaises(KeyboardInterrupt):
            results.finalize(self.work, self.output)
        self.assertFalse(raw.exists())
        self.assertEqual(results.finalize(self.work, self.output)["status"], "completed")
        self.assertEqual(results.load(self.output)["finalization"]["deleted"][0]["status"],
                         "absent_after_recorded_delete_intent")

    def test_raw_survives_failed_summary_publication_and_open_global_dependency(self):
        raw = self.raw("sample.npy")
        with patch.object(results, "_publish", side_effect=OSError("disk full")):
            self.assertEqual(self.finalize()["status"], "deferred")
        self.assertTrue(raw.exists())
        self.run["retention"]["pending_dependencies"] = ["semantic-review"]
        self.assertEqual(self.finalize()["status"], "deferred")
        self.assertTrue(raw.exists())
        self.run["retention"]["pending_dependencies"] = []
        self.assertEqual(self.finalize()["status"], "completed")
        self.assertFalse(raw.exists())

    def test_completed_history_is_immutable_and_tampering_is_detected(self):
        self.assertEqual(self.finalize()["status"], "completed")
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        self.assertEqual(results.finalize(self.work, self.output)["status"], "completed")
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, before)
        self.assertEqual(results.finalize(self.work, self.output, format="full")["status"], "deferred")
        self.run["metrics"]["completed"] = 500
        self.assertEqual(self.finalize()["status"], "deferred")
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, before)
        self.run["metrics"]["completed"] = 1
        self.write()
        path = self.output / "summary.json"
        value = json.loads(path.read_text())
        value["run"]["metrics"]["completed"] = 100
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "checksum"):
            results.verify(self.output)
        path.write_bytes(before["summary.json"])
        report = self.output / "20261102-policy-study.md"
        report.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "report differs"):
            results.verify(self.output)
        self.assertEqual(results.finalize(self.work, self.output)["status"], "deferred")
        self.assertEqual(report.read_text(), "tampered")

    def test_log_cleanup_keeps_hash_bound_scalar_aggregation(self):
        raw = self.raw("events.jsonl", kind="log")
        artifact = self.run["artifacts"][0]
        artifact["summary"] = dict(source_sha256=artifact["sha256"], source_bytes=artifact["bytes"],
            total_records=3, aggregation="all records", omitted_fields=["body"], metrics={"event_count": 3})
        self.assertEqual(self.finalize()["status"], "completed")
        self.assertFalse(raw.exists())
        summary = results.load(self.output)["run"]["log_summaries"][0]
        self.assertEqual(summary["metrics"], {"event_count": 3})
        self.assertEqual(summary["source_sha256"], artifact["sha256"])

    def test_active_run_is_protected_and_dispatch_defaults_to_summary(self):
        self.run["status"] = "running"
        self.assertEqual(self.finalize()["status"], "protected")
        self.assertFalse(self.output.exists())
        self.run["status"] = "completed"
        self.write()
        outcome = results.dispatch(dict(op="finalize", work_dir=str(self.work), output=str(self.output)), {})
        self.assertEqual(outcome["status"], "completed", outcome)
        self.assertEqual(outcome["format"], results.SUMMARY_SCHEMA)

    def test_simulated_phase_result_keeps_aggregates_but_drops_events(self):
        self.run["notice"] = "Cost-model simulation; not GPU measurements."
        self.run["phases"][0].update(policy="ditserve", cell=3, reason="diagnostic")
        self.run["phases"][0]["result"] = dict(policy="ditserve", simulated=True,
            metrics={"deadline_attainment": .75, "latency_p95_ms": 100.0},
            by_kind={"image": {"requests": 4, "ontime": 3}},
            requests=[{"request_id": "r", "events": [1, 2]}], events=["not retained"])
        self.assertEqual(self.finalize()["status"], "completed")
        phase = results.load(self.output)["run"]["phases"][0]
        self.assertEqual((phase["policy"], phase["cell"], phase["reason"]), ("ditserve", 3, "diagnostic"))
        result = phase["result"]
        self.assertTrue(result["simulated"])
        self.assertEqual(result["metrics"]["deadline_attainment"], .75)
        self.assertEqual(result["by_kind"]["image"]["ontime"], 3)
        self.assertNotIn("requests", result)
        self.assertNotIn("events", result)
        report = (self.output / "20261102-policy-study.md").read_text()
        self.assertIn("not GPU measurements", report)
        self.assertIn("| cluster / ditserve | 全部 | completed | 未记录 | 0.75 | 100 |", report)

    def test_mechanism_samples_become_bounded_aggregates(self):
        cell = self.run["phases"][0]["cells"][0]
        cell.update(elapsed_ms=2000, gpus=8,
            guard_telemetry={"status": "exclusive_observed", "dropped_events": 0, "events": [{"large": "log"}]},
            requests=[dict(status="completed", finish_ms=10, deadline_ms=20, transfers=2,
                           preemption_reason="block_preempt", preemption_stop_samples=[{"stop_ms": 1}, {"stop_ms": 5}]),
                      dict(status="failed", finish_ms=10, deadline_ms=20, transfers=1)])
        self.assertEqual(self.finalize()["status"], "completed")
        selected = results.load(self.output)["run"]["phases"][0]["cells"][0]
        self.assertEqual(selected["guard_telemetry"], {"status": "exclusive_observed", "dropped_events": 0})
        summary = selected["execution_summary"]
        self.assertEqual(summary["observed_transfers"], 3)
        self.assertEqual(summary["preempting_request_count"], 1)
        self.assertEqual(summary["preemption_stop_ms"]["p50"], 3)
        self.assertEqual(summary["preemption_stop_ms"]["count"], 2)
        self.assertEqual(summary["gpu_seconds_per_completed_request"], 16)
        self.assertEqual(summary["gpu_seconds_per_ontime_request"], 16)

    def test_report_shows_modalities_and_both_goodput_scopes_without_json_dump(self):
        self.run["phases"][0]["cells"] = []
        self.run["phases"][0]["policy"] = "sim-policy"
        self.run["phases"][0]["result"] = dict(makespan_ms=2000,
            metrics={"requests": 4, "slo_attainment": .75, "latency_ms": {"p95": 300},
                     "goodput_rps": 1.5, "qualified_goodput_rps": .5},
            by_kind={"image": {"requests": 2, "attainment": 1., "ontime": 2, "latency_p95_ms": 20},
                     "video": {"requests": 2, "deadline_attainment": .5, "ontime": 1,
                               "latency_ms": {"p95": 400}, "qualified_goodput_rps": 0.0}})
        outcome = self.finalize()
        self.assertEqual(outcome["status"], "completed", outcome)
        report = Path(outcome["report"]).read_text()
        self.assertIn("| cluster / sim-policy | 全部 | completed | 4 | 0.75 | 300 | 1.5 | 0.5 |", report)
        self.assertIn("| cluster / sim-policy | image | completed | 2 | 1 | 20 | 1 | 未验证 |", report)
        self.assertIn("| cluster / sim-policy | video | completed | 2 | 0.5 | 400 | 0.5 | 0 |", report)
        self.assertIn("资格缺失不能用按时 goodput 代替", report)
        self.assertNotIn("```json", report)

    def test_already_dated_run_id_does_not_repeat_date(self):
        self.run["run_id"] = "20261101-policy-study"
        outcome = self.finalize()
        self.assertEqual(outcome["status"], "completed", outcome)
        self.assertEqual(Path(outcome["report"]).name, "20261101-policy-study.md")
        self.assertEqual(results.verify(self.output)["status"], "verified")

    def test_atomic_batch_histogram_deduplicates_members_and_marks_unknowns(self):
        cell = self.run["phases"][0]["cells"][0]
        cell.update(image_batch=8, requests=[
            dict(kind="image", request_id="a", batch_size=2, batch_request_ids=["a", "b"]),
            dict(kind="image", request_id="b", batch_size=2, batch_request_ids=["b", "a"]),
            dict(kind="image", request_id="c", batch_size=1, batch_request_ids=["c"]),
            dict(kind="image", request_id="d"),
            dict(kind="image", request_id="e", batch_size=2, batch_request_ids=["e", "missing"]),
            dict(kind="image", request_id="f", batch_size=2, batch_request_ids=["f", "f"])])
        self.assertEqual(self.finalize()["status"], "completed")
        saved = results.load(self.output)["run"]["phases"][0]["cells"][0]
        self.assertEqual(saved["image_batch"], 8)
        evidence = saved["execution_summary"]["atomic_batch"]
        self.assertEqual(evidence["batch_size_histogram"], {"1": 1, "2": 1})
        self.assertEqual(evidence["request_size_histogram"], {"1": 1, "2": 4})
        self.assertEqual(evidence["verified_batches"], 2)
        self.assertEqual(evidence["covered_requests"], 3)
        self.assertEqual(evidence["unknown_requests"], 3)
        self.assertEqual(evidence["status"], "partial")
        self.assertIsNone(evidence["total_batches"])
        self.assertNotIn("8", evidence["batch_size_histogram"])

    def test_missing_mechanism_evidence_is_unknown_not_zero(self):
        summary = results._execution_summary({"requests": [
            dict(kind="image", request_id="a"), dict(kind="video", request_id="v")]})
        self.assertEqual(summary["atomic_batch"]["status"], "unknown")
        self.assertIsNone(summary["atomic_batch"]["total_batches"])
        for field in ("resumes", "resume_extra_ms"):
            self.assertEqual(summary[field]["count"], 0)
            self.assertEqual(summary[field]["missing_count"], 1)
            self.assertIsNone(summary[field]["total"])
            self.assertIsNone(summary[field]["p50"])

    def test_resume_distributions_keep_zero_negative_and_missing_distinct(self):
        summary = results._execution_summary({"requests": [
            dict(kind="video", resumes=0),
            dict(kind="video", resumes=2, resume_extra_ms=-2.0),
            dict(kind="video", resumes=4, resume_extra_ms=6.0),
            dict(kind="video")]})
        self.assertEqual(summary["resumes"]["count"], 3)
        self.assertEqual(summary["resumes"]["total"], 6)
        self.assertEqual(summary["resumes"]["p50"], 2)
        self.assertEqual(summary["resumes"]["missing_count"], 1)
        self.assertEqual(summary["resume_extra_ms"]["count"], 2)
        self.assertEqual(summary["resume_extra_ms"]["min"], -2)
        self.assertEqual(summary["resume_extra_ms"]["total"], 4)
        self.assertEqual(summary["resume_extra_ms"]["p50"], 2)
        self.assertEqual(summary["resume_extra_ms"]["missing_count"], 2)


if __name__ == "__main__":
    unittest.main()
