"""CPU fixtures for small, exact follow-up archives and retained dependencies."""

import json
from pathlib import Path
import tempfile
import unittest

from ditserve.optimizations import followup_report as report
from ditserve.optimizations.online import paired_gate, request_identity


class FollowupReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "work"
        self.output = Path(self.tmp.name) / "output"
        self.root.mkdir()
        self.ledger = self.root / "ledger.json"
        self.save(self.root / "config.json", dict(seeds=[101], loads=[.7], text="=do_not_execute", long_id=100000000000000000001))
        self.save(self.root / "study.json", dict(status="running", ledger=str(self.ledger), dependencies_closed=False))
        self.save(self.ledger, dict(active=None, sessions=[], used_gpu_seconds=0., budget_unlimited=True))

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def install(self, *, failed=False, attempt="attempt000"):
        directory = self.root / "sana_combo" / attempt
        rows = []
        for pattern in ("ordinary", "burst"):
            for order in (0, 1):
                for arm in ("baseline", "candidate"):
                    requests = []
                    for index, kind in enumerate(("image", "video")):
                        cost = (20. if kind == "image" else 80.) * (1. if arm == "baseline" else .8)
                        arrival = 0. if pattern == "ordinary" else 1.
                        requests.append(dict(request_id="r%d" % index, kind=kind, arrival_ms=arrival,
                            deadline_ms=100., service_ms=25., seed=101, arrived=True,
                            status="completed", start_ms=arrival, finish_ms=arrival + cost,
                            service=dict(ms=cost, stages=[dict(stage="raw", secret="NO_RAW_STAGE_BODY")]),
                            encode_start_ms=arrival, encode_finish_ms=arrival + 1.))
                    if failed and order == 1 and arm == "candidate" and pattern == "burst":
                        requests[0].update(status="failed", finish_ms=None, error="OOM")
                    rows.append(dict(trace_sha256=request_identity(requests), order_index=order, arm=arm,
                        seed=101, load=.7, pattern=pattern, elapsed_ms=100.,
                        status="failed" if any(r["status"] == "failed" for r in requests) else "completed",
                        complete=all(r["status"] == "completed" for r in requests), requests=requests,
                        events=[dict(kind="video_finish", start_ms=10., end_ms=15., secret="NO_RAW_EVENT_BODY")]))
        expected = sorted({r["trace_sha256"] for r in rows})
        data = dict(schema_version=1, phase="sana_combo", status="completed", verdict="值得继续",
            quality=dict(passed=True, rows=[dict(input_id="p0-s201", lpips=.00001)]),
            local=[dict(input_id="p0-s201", arm="candidate", service_ms=15., output=dict(shape=[1, 2, 2, 3]))],
            online=dict(status="completed", replays=rows, comparison_gate=paired_gate(rows, expected_trace_ids=expected),
                expected_trace_ids=expected, expected_replays=8, expected_requests=16),
            costs=dict(cold_compile_ms=30., warmup_ms=10.), failures=[dict(error="historical fixture failure")] if failed else [])
        self.save(directory / "phase-result.json", data)
        charge = dict(session_id=attempt, stage="sana_combo", gpu_seconds=20., elapsed_seconds=20., gpu_count=1, status="completed")
        supervisor = dict(status="completed", returncode=0, charge=charge,
            final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[]))
        self.save(directory / "supervisor.json", supervisor)
        budget = json.loads(self.ledger.read_text())
        budget["sessions"].append(charge)
        budget["used_gpu_seconds"] += 20.
        self.save(self.ledger, budget)
        return directory, data

    def test_collect_preserves_requests_pairs_failures_and_event_aggregates(self):
        directory, data = self.install(failed=True)
        value = report.collect(self.root)
        self.assertEqual(len(value["tables"]["Requests"]) - 1, 16)
        self.assertEqual(len(value["tables"]["PairedRequests"]) - 1, 8)
        self.assertEqual(value["phases"][0]["aggregates"]["candidate"]["image"]["arrived"], 4)
        self.assertEqual(value["phases"][0]["aggregates"]["candidate"]["image"]["completed"], 3)
        self.assertEqual(value["phases"][0]["aggregates"]["candidate"]["image"]["failed_or_incomplete"], 1)
        self.assertFalse(value["phases"][0]["recomputed_gate_passed"])
        encoded = json.dumps(value)
        self.assertNotIn("NO_RAW_EVENT_BODY", encoded)
        self.assertNotIn("NO_RAW_STAGE_BODY", encoded)
        self.assertIn("event_summary", encoded)
        self.assertIn("historical fixture failure", encoded)
        self.assertTrue(value["phases"][0]["accounting_settled"])
        self.assertFalse(value["dependencies_closed"])

    def test_write_only_two_files_and_exact_literal_readback(self):
        self.install()
        raw = self.root / "sana_combo/attempt000/reference.npy"
        raw.write_bytes(b"RAW_ARRAY_DO_NOT_LOAD")
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = report.write(self.root, self.output)
        self.assertEqual({p.name for p in self.output.iterdir()}, {"results.xlsx", "REPORT.md"})
        self.assertEqual(result["excel"]["status"], "verified")
        self.assertEqual(result["requests"], 16)
        self.assertEqual(result["originals_deleted"], 0)
        self.assertFalse(result["baseline_promotion"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        from openpyxl import load_workbook
        workbook = load_workbook(self.output / "results.xlsx", read_only=True, data_only=False)
        try:
            values = [cell for row in workbook["Config"] for cell in row]
            formula_text = next(cell for cell in values if cell.value == "=do_not_execute")
            self.assertEqual(formula_text.data_type, "s")
            self.assertIn("100000000000000000001", [cell.value for cell in values])
        finally:
            workbook.close()
        self.assertIn("原始文件全部保留", (self.output / "REPORT.md").read_text())
        report.write(self.root, self.output)

    def test_active_budget_and_missing_usage_are_not_filled_with_zero(self):
        self.install()
        self.save(self.ledger, dict(active=dict(stage="sana_combo"), sessions=[]))
        value = report.collect(self.root)
        self.assertTrue(value["active_unsettled"])
        self.assertIsNone(value["used_gpu_seconds"])
        self.assertFalse(value["phases"][0]["accounting_settled"])
        self.assertFalse(value["dependencies_closed"])

    def test_closed_inventory_hashes_raw_but_never_deletes_it(self):
        self.install()
        self.save(self.root / "study.json", dict(status="completed", ledger=str(self.ledger), dependencies_closed=True))
        raw = self.root / "sana_combo/attempt000/reference.npy"
        raw.write_bytes(b"still retained")
        value = report.collect(self.root)
        self.assertTrue(value["dependencies_closed"])
        retained = next(r for r in value["retention"] if r["path"] == str(raw))
        self.assertEqual(len(retained["sha256"]), 64)
        self.assertTrue(retained["dependencies_closed"])
        self.assertFalse(retained["cleanup_authorized"])
        self.assertTrue(raw.exists())

    def test_retries_stay_separate_and_are_not_pooled_as_new_evidence(self):
        self.install(failed=True)
        self.install(attempt="attempt001")
        value = report.collect(self.root)
        self.assertEqual(len(value["phases"]), 2)
        self.assertEqual([r["recorded_requests"] for r in value["phases"]], [16, 16])
        self.assertEqual(value["used_gpu_seconds"], 40.)
        self.assertFalse(value["baseline_promotion"])

    def test_missing_supervisor_unknown_schema_and_missing_fields_remain_reportable(self):
        directory, data = self.install()
        (directory / "supervisor.json").unlink()
        self.save(directory / "phase-result.json", dict(schema_version=99, phase="sana_combo", status="running", online=None))
        value = report.collect(self.root)
        self.assertEqual(value["phases"][0]["recorded_requests"], 0)
        self.assertFalse(value["phases"][0]["accounting_settled"])
        self.assertIn("schema_version", " ".join(value["phases"][0]["errors"]))
        self.assertFalse(value["dependencies_closed"])

    def test_bad_input_pair_and_duplicate_arm_are_retained_and_flagged(self):
        directory, data = self.install()
        data["online"]["replays"][1]["requests"][0]["deadline_ms"] = 99.
        data["online"]["replays"].append(data["online"]["replays"][0])
        self.save(directory / "phase-result.json", data)
        value = report.collect(self.root)
        self.assertEqual(value["phases"][0]["recorded_requests"], 18)
        self.assertIn("missing_or_duplicate_arm", " ".join(value["phases"][0]["errors"]))
        self.assertIn("expected replay count", " ".join(value["phases"][0]["errors"]))

    def test_provided_gate_cannot_override_failed_recomputed_gate(self):
        directory, data = self.install(failed=True)
        data["online"]["comparison_gate"]["passed"] = True
        self.save(directory / "phase-result.json", data)
        value = report.collect(self.root)
        self.assertTrue(value["phases"][0]["reported_gate_passed"])
        self.assertFalse(value["phases"][0]["recomputed_gate_passed"])
        self.assertIn("disagree", " ".join(value["phases"][0]["errors"]))

    def test_missing_plan_does_not_claim_complete_planned_matrix(self):
        directory, data = self.install()
        data["online"].pop("expected_trace_ids")
        data["online"].pop("expected_replays")
        self.save(directory / "phase-result.json", data)
        value = report.collect(self.root)
        self.assertIn("complete planned matrix not established", value["phases"][0]["matrix_scope"])

    def test_long_error_text_is_reversibly_split(self):
        directory, data = self.install()
        message = "failure text " * 5000
        data["failures"] = [dict(error=message)]
        self.save(directory / "phase-result.json", data)
        value = report.collect(self.root)
        parts = [row for row in value["tables"]["Failures"][1:] if row[2] == "recorded failures" and row[3] == "/0/error"]
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(row[4] for row in sorted(parts, key=lambda r: r[5])), message)
        report.write(self.root, self.output)

    def test_compacted_events_keep_recorded_attribution_and_cleanup_receipts(self):
        directory, data = self.install()
        for replay in data["online"]["replays"]:
            replay.pop("events")
            replay["event_summary"] = dict(recorded=1000, counts=dict(video_finish=2))
            replay["attribution"] = dict(video_finish_wait_ms=1250.)
            replay["requests"][0]["attribution"] = dict(video_finish_overlap_ms=1000.)
        self.save(directory / "phase-result.json", data)
        self.save(self.root / "cleanup.json", dict(files=[dict(path="removed.npy", bytes=123, sha256="a" * 64, status="deleted", deleted=True)],
            rewrites=[dict(path=str(directory / "phase-result.json"), before_sha256="b" * 64, after_sha256="c" * 64,
                          before_bytes=25000, after_bytes=4000, status="aggregated")]))
        value = report.collect(self.root)
        encoded = json.dumps(value["tables"])
        self.assertIn("video_finish_overlap_ms", encoded)
        self.assertIn("video_finish_wait_ms", encoded)
        self.assertIn("/event_summary/recorded", encoded)
        self.assertIn("1000", encoded)
        self.assertIn("b" * 64, encoded)
        self.assertIn("c" * 64, encoded)
        self.assertFalse(value["raw_retained"])
        self.assertEqual(value["prior_cleanup_deleted"], 1)
        text = report.render(value)
        self.assertNotIn("原始文件全部保留", text)
        self.assertIn("无法据本报告重新评测", text)
        result = report.write(self.root, self.output)
        self.assertEqual(result["prior_cleanup_rewrites"], 1)

    def test_long_numeric_arrays_are_explicitly_aggregated_and_raw_remains(self):
        directory, data = self.install()
        data["quality"]["per_frame_metric"] = list(range(300))
        self.save(directory / "phase-result.json", data)
        value = report.collect(self.root)
        self.assertIn("long numeric list aggregated", json.dumps(value["omissions"]))
        self.assertTrue((directory / "phase-result.json").exists())

    def test_resume_scalar_paths_are_preserved(self):
        directory, data = self.install()
        data["local"].append(dict(kind="resume", steps=32, interruptions=3, finish=2,
            final=dict(strict_metrics=dict(rtol=1e-5, atol=1e-6, passed=True, maximum_error=0.))))
        self.save(directory / "phase-result.json", data)
        value = report.collect(self.root)
        encoded = json.dumps(value["tables"]["Metrics"])
        self.assertIn("/1/final/strict_metrics/rtol", encoded)
        self.assertIn("/1/interruptions", encoded)

    def test_frozen_study_uses_live_queue_status_and_phase_dependency_closure(self):
        self.install()
        self.save(self.root / "study.json", dict(ledger=str(self.ledger), phases=["sana_combo", "granularity"]))
        self.save(self.root / "queue.json", dict(status="running", phases=["sana_combo", "granularity"]))
        value = report.collect(self.root)
        self.assertEqual(value["status"], "running")
        self.assertFalse(value["dependencies_closed"])
        self.assertTrue(value["phases"][0]["dependencies_closed"])
        retained = next(r for r in value["retention"] if r["relative_path"].endswith("phase-result.json"))
        self.assertTrue(retained["dependencies_closed"])
        self.assertIn(str(self.root / "queue.json"), {r["path"] for r in value["sources"]})
        self.save(self.root / "queue.json", dict(status="completed", phases=["sana_combo"]))
        self.assertTrue(report.collect(self.root)["dependencies_closed"])

    def test_active_failed_partial_retains_denominator_and_its_actual_source(self):
        directory, data = self.install(failed=True)
        partial = data["online"]["replays"].pop()
        partial_path = directory / "online-partials/replay-07/partial.json"
        self.save(partial_path, partial)
        data["status"] = "failed"
        data["online"].update(status="running", comparison_gate={}, active_unit=dict(partial_path=str(partial_path)))
        self.save(directory / "phase-result.json", data)
        value = report.collect(self.root)
        self.assertEqual(value["phases"][0]["recorded_requests"], 16)
        self.assertEqual(value["phases"][0]["aggregates"]["candidate"]["image"]["failed_or_incomplete"], 1)
        rows = [r for r in value["tables"]["Requests"][1:] if r[-2] == str(partial_path)]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][-1], "/requests/0")
        self.assertFalse(value["phases"][0]["dependencies_closed"])

    def test_waiter_without_study_or_ledger_is_reportable_before_gpu_work(self):
        (self.root / "study.json").unlink()
        self.ledger.unlink()
        self.save(self.root / "queue.json", dict(status="waiting", ledger=str(self.ledger), phases=["sana_combo"]))
        value = report.collect(self.root)
        self.assertEqual(value["status"], "waiting")
        self.assertFalse(value["errors"])
        self.assertIsNone(value["used_gpu_seconds"])

    def test_planned_cleanup_is_not_reported_as_already_performed(self):
        self.install()
        self.save(self.root / "cleanup.json", dict(status="planned",
            files=[dict(path="raw.npy", status="planned")], rewrites=[dict(path="phase-result.json", status="planned")]))
        value = report.collect(self.root)
        self.assertTrue(value["raw_retained"])
        self.assertEqual(value["prior_cleanup_rewrites"], 0)

    def test_symlink_input_and_existing_foreign_output_are_protected(self):
        self.install()
        link = Path(self.tmp.name) / "linked_work"
        link.symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "symlink"):
            report.collect(link)
        self.output.mkdir()
        (self.output / "REPORT.md").write_text("historical report")
        with self.assertRaisesRegex(ValueError, "another experiment"):
            report.write(self.root, self.output)
        self.assertEqual((self.output / "REPORT.md").read_text(), "historical report")

    def test_work_output_overlap_is_rejected(self):
        self.install()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            report.write(self.root, self.root / "exports")


if __name__ == "__main__":
    unittest.main()
