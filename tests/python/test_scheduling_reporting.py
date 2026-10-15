import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json
from ditserve.scheduling.reporting import (
    BASELINES, CANDIDATES, SCENARIOS, aggregate, build_report, complete_replay,
    evaluate_screen, evaluate_validate, report,
)


def replay(scenario, arm, seed=301, elapsed=1000.):
    videos = 3 if scenario == "video_contention" else 2
    rows = [dict(request_id="r%03d" % i, kind="video" if i < videos else "image", seed=seed+i,
                 arrival_ms=0., deadline_ms=200., start_ms=10., finish_ms=100.,
                 status="completed", arrived=True, pause_max_ms=float(i),
                 timing=dict(encoding_queue_ms=1., gpu_queue_ms=2.)) for i in range(24)]
    return dict(scenario=scenario, seed=seed, arm=arm, trace_sha256="%s-%s" % (scenario, seed),
                device_identity=dict(uuid="A100"), cost_model_sha256="cost", rows=rows,
                elapsed_ms=elapsed, quality=dict(passed=True), policy_triggers=dict(edf=24))


def matrix(seeds=(301,), arms=(*BASELINES, *CANDIDATES)):
    return [replay(scenario, arm, seed, 900. if arm in CANDIDATES and scenario in
                   ("tight_deadline", "video_contention") else 1000.)
            for seed in seeds for scenario in SCENARIOS for arm in arms]


class SchedulingReportingTests(unittest.TestCase):
    def test_equal_candidates_choose_fixed_quantum_after_complete_screen(self):
        result = evaluate_screen(matrix())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["selected_arm"], "guarded_b4")
        self.assertTrue(all(value["passed"] for value in result["candidates"].values()))

    def test_missing_cell_or_unverified_quality_cannot_pass(self):
        for records in (matrix()[:-1], matrix()):
            if len(records) == 16:
                records[-1]["quality"] = dict(passed=False)
            result = evaluate_screen(records)
            self.assertEqual(result["status"], "incomplete")
            self.assertIsNone(result["selected_arm"])

    def test_trace_hash_cannot_hide_changed_inputs_or_device(self):
        for field, value in (("device_identity", dict(uuid="V100")), ("cost_model_sha256", "other"),
                             ("trace_sha256", "other")):
            records = matrix()
            records[-1][field] = value
            self.assertEqual(evaluate_screen(records)["status"], "incomplete")
        records = matrix()
        records[-1]["rows"][0]["deadline_ms"] += 1
        self.assertEqual(evaluate_screen(records)["status"], "incomplete")

    def test_even_consistent_pairs_must_not_pool_devices_across_scenarios(self):
        records = matrix()
        for record in records:
            if record["scenario"] == "video_contention":
                record["device_identity"] = dict(uuid="V100")
        self.assertEqual(evaluate_screen(records)["status"], "incomplete")

    def test_duplicate_attempt_not_silently_selected(self):
        records = matrix()
        records.append(copy.deepcopy(records[-1]))
        result = evaluate_screen(records)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["candidates"]["guarded_adaptive"]["issues"][0]["reason"], "duplicate")

    def test_faster_overall_does_not_excuse_original_or_video_regression(self):
        records = matrix()
        for row in records:
            if row["arm"] in CANDIDATES and row["scenario"] == "original_ordinary":
                row["elapsed_ms"] = 1030.
        self.assertEqual(evaluate_screen(records)["status"], "failed")
        records = matrix()
        for row in records:
            if row["arm"] in CANDIDATES and row["scenario"] == "video_contention":
                row["rows"][0]["finish_ms"] = 201.
        result = evaluate_screen(records)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any(check["name"] == "video_attainment" and not check["passed"]
                            for check in result["candidates"]["guarded_b4"]["checks"]))

    def test_holdout_cannot_reselect_better_candidate(self):
        records = matrix((401, 402))
        for record in records:
            if record["arm"] == "guarded_b4":
                record["elapsed_ms"] = 1050.
        result = evaluate_validate(records, "guarded_b4")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["selected_arm"], "guarded_b4")
        self.assertFalse(result["reselection_allowed"])
        self.assertEqual(evaluate_validate(records, None)["status"], "not_eligible")

    def test_holdout_requires_positive_pressure_gain_in_each_seed(self):
        records = matrix((401, 402))
        for record in records:
            if record["arm"] == "guarded_b4" and record["scenario"] in ("tight_deadline", "video_contention"):
                record["elapsed_ms"] = 600. if record["seed"] == 401 else 1100.
        result = evaluate_validate(records, "guarded_b4")
        self.assertFalse(result["passed"])
        self.assertTrue(any(check["name"] == "positive_seed_pressure_gain" and not check["passed"]
                            for check in result["checks"]))

    def test_failures_and_not_arrived_rows_keep_correct_denominators(self):
        record = replay("original_burst", "edf_b1")
        record["rows"][0].update(status="failed", finish_ms=None)
        record["rows"][1].update(status="pending", finish_ms=None, start_ms=None)
        record["rows"][2].update(arrived=False, status="pending", finish_ms=None)
        metrics = aggregate([record])["overall"]
        self.assertEqual((metrics["arrived"], metrics["completed"], metrics["failed_or_incomplete"]), (23, 21, 2))
        self.assertAlmostEqual(metrics["on_time_fraction"], 21/23)
        self.assertEqual(metrics["goodput_rps"], 21.)
        self.assertFalse(complete_replay(record))

    def test_future_finish_or_duplicate_request_ids_is_not_complete(self):
        record = replay("original_burst", "edf_b1")
        record["rows"][0]["finish_ms"] = 1001.
        self.assertFalse(complete_replay(record))
        record["rows"][0]["finish_ms"] = 100.
        record["rows"][0]["request_id"] = record["rows"][1]["request_id"]
        self.assertFalse(complete_replay(record))

    def test_report_keeps_partial_evidence_and_deduplicates_exact_inherited_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = matrix()
            pilot = [record for record in records if record["scenario"] in ("original_burst", "tight_deadline")
                     and record["arm"] != "guarded_adaptive"]
            atomic_json(root / "pilot/result.json", dict(status="complete", records=pilot))
            atomic_json(root / "screen/result.json", dict(status="complete", records=records))
            for phase in ("calibrate", "pilot", "screen"):
                atomic_json(root / phase / "supervisor.json", dict(status="passed", returncode=0, eligible=True))
            before = (root / "pilot/result.json").read_bytes()
            with patch("ditserve.scheduling.reporting.export_report", return_value=dict(status="exported")) as export:
                value = report(root)
            export.assert_called_once()
            self.assertEqual(value["pilot"]["status"], "complete")
            self.assertTrue(value["pilot"]["descriptive_only"])
            self.assertEqual(value["screen"]["selected_arm"], "guarded_b4")
            self.assertEqual((root / "pilot/result.json").read_bytes(), before)
            self.assertEqual(value["budget_summary"]["status"], "unavailable")
            self.assertIn("p95", (root / "REPORT.md").read_text())
            self.assertEqual(json.loads((root / "report.json").read_text())["screen"]["status"], "passed")

    def test_local_success_cannot_override_failed_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "screen/result.json", dict(status="passed", records=matrix()))
            atomic_json(root / "calibrate/supervisor.json", dict(status="passed", returncode=0, eligible=True))
            atomic_json(root / "screen/supervisor.json", dict(status="incomplete", returncode=1,
                        supervisor_failure=dict(reason="GPU isolation lost")))
            result = build_report(root)
            self.assertEqual(result["screen"]["status"], "incomplete")
            self.assertEqual(result["phases"]["screen"]["status"], "incomplete")
            self.assertEqual(len(result["phases"]["screen"]["summaries"]), 16)

    def test_compact_report_preserves_request_metrics_without_action_event_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = replay("original_burst", "edf_b1")
            record["rows"][0].update(action_spans=[dict(action="denoise", dispatch_ms=10.)] * 20,
                                      stage_ms=dict(prepare=1., denoise=20.), gpu_queue_ms=4.)
            atomic_json(root / "pilot/result.json", dict(status="running", records=[record]))
            value = build_report(root)
            row = value["phases"]["pilot"]["summaries"][0]["requests"][0]
            self.assertEqual(row["action_spans_count"], 20)
            self.assertNotIn("action_spans", row)
            self.assertEqual(row["stage_ms"], dict(prepare=1., denoise=20.))
            self.assertEqual(row["gpu_queue_ms"], 4.)
            self.assertNotIn("records", value["phases"]["pilot"])

    def test_broken_phase_json_remains_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pilot").mkdir()
            (root / "pilot/result.json").write_text("{")
            value = build_report(root)
            self.assertEqual(value["pilot"]["status"], "incomplete")
            self.assertEqual(value["phases"]["pilot"]["status"], "unreadable")
            self.assertTrue(value["evidence_errors"])


if __name__ == "__main__":
    unittest.main()
