import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, file_digest
from ditserve.scheduling.next_reporting import (
    COMPLETION_ARMS, GRANULARITY_ARMS, SCENARIOS, build_report, collect_event_summaries,
    evaluate_completion, evaluate_granularity, evaluate_replication, overhead_metrics, report,
)


COSTS = dict(profile="frozen empirical costs")
DEVICE = dict(gpu_uuid="GPU-V100", sana_profile="v100_fp32")


def replay(scenario, arm, seed=301, elapsed=None):
    videos = 3 if scenario == "video_contention" else 2
    rows = [dict(request_id="r%03d" % i, kind="video" if i < videos else "image", seed=seed+i,
                 arrival_ms=0., deadline_ms=200., start_ms=10., finish_ms=100.,
                 status="completed", arrived=True, pause_max_ms=float(i),
                 stage_ms=dict(denoise=2.), action_counts={}) for i in range(24)]
    candidate = arm in ("edf_b2", "edf_b1_async")
    return dict(scenario=scenario, seed=seed, arm=arm, trace_sha256="%s-%s" % (scenario, seed),
                device_identity=DEVICE.copy(), cost_model_sha256=digest(COSTS), rows=rows,
                elapsed_ms=elapsed or (900. if candidate and scenario in ("tight_deadline", "video_contention") else 1000.),
                quality=dict(passed=True), policy_triggers=dict(edf=24),
                scheduling_ms=2., checkpoint_ms=3., output_persistence_ms=10.)


def development():
    return [replay(scenario, arm) for scenario in SCENARIOS for arm in GRANULARITY_ARMS]


def replication(candidate=None):
    arms = ("edf_b1", "edf_b4", "edf_step", *((candidate,) if candidate else ()))
    return [replay(scenario, arm, seed) for seed in (401, 402) for scenario in SCENARIOS for arm in arms]


def source_fixture(root):
    source = root / "source-run"
    inherited = [replay(scenario, arm) for scenario in SCENARIOS[:3]
                 for arm in (COMPLETION_ARMS if scenario == "tight_deadline" else COMPLETION_ARMS[:4])]
    screen_path, calibration_path, supervisor_path = (source / suffix for suffix in (
        "screen/result.json", "calibrate/result.json", "screen/supervisor.json"))
    atomic_json(screen_path, dict(status="incomplete", error_type="BudgetStop", records=inherited))
    atomic_json(calibration_path, dict(status="passed", eligible=True, cost_model=COSTS,
                cost_model_sha256=digest(COSTS), device_identity=DEVICE))
    atomic_json(supervisor_path, dict(status="incomplete", returncode=2, budget_stopped=True,
                final_occupancy=dict(status="exclusive_observed", foreign=[], owned=[])))
    evidence = dict(source_run=str(source), screen_result_path=str(screen_path), screen_result_sha256=file_digest(screen_path),
                    calibration_path=str(calibration_path), calibration_sha256=file_digest(calibration_path),
                    source_supervisor_path=str(supervisor_path), source_supervisor_sha256=file_digest(supervisor_path))
    return dict(status="passed", eligible=True, source_evidence=evidence, inherited_records=inherited,
                records=[replay("video_contention", arm) for arm in COMPLETION_ARMS])


def phase(root, name, value, *, supervisor=None):
    atomic_json(root / name / "result.json", value)
    atomic_json(root / name / "supervisor.json", dict(value, returncode=0) if supervisor is None else supervisor)


class SchedulingNextReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_complete_four_arm_screen_tie_selects_synchronous_two_block(self):
        value = evaluate_granularity(development())
        self.assertEqual(value["status"], "passed")
        self.assertEqual(value["selected_candidate"], "edf_b2")
        self.assertTrue(all(result["passed"] for result in value["candidates"].values()))

    def test_async_candidate_can_win_without_relaxing_quality_or_video_gates(self):
        rows = development()
        for row in rows:
            if row["arm"] == "edf_b1_async":
                row["elapsed_ms"] = 800.
        self.assertEqual(evaluate_granularity(rows)["selected_candidate"], "edf_b1_async")
        rows[-1]["quality"]["passed"] = False
        self.assertEqual(evaluate_granularity(rows)["status"], "incomplete")

    def test_missing_candidate_or_extra_undeclared_record_blocks_selection(self):
        for rows in (development()[:-1], development()+[replay("original_burst", "other")]):
            value = evaluate_granularity(rows)
            self.assertEqual(value["status"], "incomplete")
            self.assertIsNone(value["selected_candidate"])

    def test_original_regression_cannot_be_offset_by_pressure_gain(self):
        rows = development()
        for row in rows:
            if row["arm"] in ("edf_b2", "edf_b1_async"):
                row["elapsed_ms"] = 1030. if row["scenario"] == "original_burst" else 500.
        value = evaluate_granularity(rows)
        self.assertEqual(value["status"], "failed")
        self.assertIsNone(value["selected_candidate"])

    def test_video_regression_and_cross_device_pairs_fail(self):
        rows = development()
        for row in rows:
            if row["arm"] in ("edf_b2", "edf_b1_async"):
                row["rows"][0]["finish_ms"] = 201.
        self.assertEqual(evaluate_granularity(rows)["status"], "failed")
        rows = development()
        rows[0]["device_identity"] = dict(gpu_uuid="other")
        self.assertEqual(evaluate_granularity(rows)["status"], "incomplete")

    def test_completion_closes_new_matrix_without_modifying_historical_report(self):
        value = source_fixture(self.root)
        originals = {path: path.read_bytes() for path in (self.root / "source-run").rglob("*.json")}
        result = evaluate_completion(value)
        self.assertTrue(result["technical_complete"])
        self.assertEqual((result["inherited_replays"], result["new_replays"]), (14, 6))
        self.assertFalse(result["source"]["historical_qualification_rewritten"])
        self.assertEqual(result["recomputed_original_gate"]["status"], "failed")
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})

    def test_completion_requires_exact_old_records_and_new_six_arm_group(self):
        value = source_fixture(self.root)
        value["inherited_records"][0]["elapsed_ms"] += 1.
        self.assertFalse(evaluate_completion(value)["technical_complete"])
        value = source_fixture(self.root)
        value["records"].pop()
        self.assertFalse(evaluate_completion(value)["technical_complete"])

    def test_completion_rejects_changed_source_digest_or_non_budget_interruption(self):
        value = source_fixture(self.root)
        path = Path(value["source_evidence"]["screen_result_path"])
        path.write_text("{}")
        self.assertFalse(evaluate_completion(value)["technical_complete"])
        value = source_fixture(self.root)
        path = Path(value["source_evidence"]["source_supervisor_path"])
        old = json.loads(path.read_text())
        old["supervisor_failure"] = dict(reason="foreign GPU process")
        atomic_json(path, old)
        value["source_evidence"]["source_supervisor_sha256"] = file_digest(path)
        self.assertFalse(evaluate_completion(value)["technical_complete"])

    def test_replication_always_runs_fixed_mechanisms_without_candidate_reselection(self):
        value = evaluate_replication(replication(), None)
        self.assertTrue(value["technical_complete"])
        self.assertEqual(value["candidate_validation"]["status"], "not_selected")
        self.assertEqual(len(value["mechanism_comparisons"]), 16)
        self.assertFalse(value["reselection_allowed"])
        self.assertFalse(evaluate_replication(replication("edf_b2"), None)["technical_complete"])

    def test_replication_validates_only_the_frozen_candidate(self):
        rows = replication("edf_b1_async")
        self.assertTrue(evaluate_replication(rows, "edf_b1_async")["candidate_validation"]["passed"])
        for row in rows:
            if row["arm"] == "edf_b1_async" and row["seed"] == 402 and row["scenario"] in ("tight_deadline", "video_contention"):
                row["elapsed_ms"] = 1100.
        result = evaluate_replication(rows, "edf_b1_async")
        self.assertFalse(result["candidate_validation"]["passed"])
        self.assertEqual(result["frozen_candidate"], "edf_b1_async")

    def test_overhead_shares_preserve_missing_data_and_async_overlap(self):
        row = replay("original_burst", "edf_b1")
        result = overhead_metrics([row])
        self.assertEqual(result["tracked_overhead_share"], .015)
        row.pop("checkpoint_ms")
        self.assertIsNone(overhead_metrics([row])["tracked_overhead_share"])
        row = replay("original_burst", "edf_b1_async")
        result = overhead_metrics([row])
        self.assertIsNone(result["tracked_overhead_share"])
        self.assertEqual(result["measurements"]["output_persistence_ms"]["share_of_elapsed"], .01)
        self.assertTrue(result["background_output_overlap"])
        row.update(artifact_writer=dict(write_ms=500., queue_wait_ms=1., drain_wait_ms=2.), output_drain_ms=2.)
        result = overhead_metrics([row])
        self.assertEqual(result["tracked_overhead_share"], .017)
        self.assertEqual(result["background_writer"]["write_ms"]["mean"], 500.)

    def _study(self, *, candidate_passed=True):
        completion = source_fixture(self.root)
        root = self.root / "new-run"
        phase(root, "completion", completion)
        rows = development()
        if not candidate_passed:
            for row in rows:
                row["elapsed_ms"] = 1000.
        selection = evaluate_granularity(rows)
        phase(root, "granularity", dict(status="passed", eligible=True, records=rows, selection=selection))
        candidate = selection["selected_candidate"]
        phase(root, "replicate", dict(status="passed", eligible=True, records=replication(candidate), frozen_candidate=candidate))
        return root

    def test_failed_new_candidates_still_allow_complete_prespecified_mechanism_replication(self):
        root = self._study(candidate_passed=False)
        result = build_report(root)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["granularity"]["status"], "failed")
        self.assertEqual(result["replication"]["candidate_validation"]["status"], "not_selected")
        self.assertTrue(result["replication"]["frozen_selection_verified"])

    def test_supervisor_failure_or_changed_local_selection_prevents_qualification(self):
        root = self._study()
        path = root / "granularity/supervisor.json"
        supervisor = json.loads(path.read_text())
        supervisor["supervisor_failure"] = dict(reason="isolation lost")
        atomic_json(path, supervisor)
        self.assertEqual(build_report(root)["granularity"]["status"], "incomplete")
        root = self._study()
        path = root / "granularity/result.json"
        local = json.loads(path.read_text())
        local["selection"]["selected_candidate"] = "edf_b1_async"
        atomic_json(path, local)
        result = build_report(root)
        self.assertEqual(result["granularity"]["status"], "incomplete")
        self.assertTrue(result["evidence_errors"])

    def test_report_exports_lightweight_per_request_metrics_and_independent_budget(self):
        root = self._study()
        ledger = self.root / "gpu-budget.json"
        atomic_json(ledger, dict(used_gpu_seconds=3600., limit_gpu_seconds=None, active=None, sessions=[]))
        with patch("ditserve.scheduling.next_reporting.export_report", return_value=dict(status="verified")):
            value = report(root, ledger)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(value["budget_summary"]["including_active_gpu_seconds"], 3600.)
        self.assertTrue(value["budget_summary"]["unlimited"])
        self.assertEqual(len(value["phases"]["granularity"]["summaries"][0]["requests"]), 24)
        self.assertIn("旧资格没有回写", (root / "REPORT.md").read_text())
        self.assertTrue((root / "report.json").is_file())

    def test_event_summary_streams_counts_durations_and_rejects_external_or_changed_files(self):
        root = self.root / "new-run"
        row = replay("original_burst", "edf_b1")
        path = root / "granularity/arrays/replay/execution-events.jsonl"
        path.parent.mkdir(parents=True)
        events = [dict(request_id="r000", action="denoise", quantum=1, reason="edf", dispatch_ms=2., end_ms=5.),
                  dict(request_id="r000", action="denoise", quantum=1, reason="edf", dispatch_ms=8., end_ms=12.)]
        path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        row["rows"][0]["action_counts"] = dict(denoise=2)
        row["execution_events"] = dict(path=str(path), count=2, sha256=file_digest(path))
        phase(root, "granularity", dict(status="passed", eligible=True, records=[row]))
        result = collect_event_summaries(root)
        self.assertEqual(result["cleanup_summarized_paths"], ["granularity/arrays/replay/execution-events.jsonl"])
        summary = next(iter(result["summaries"].values()))
        self.assertEqual(summary["groups"][0]["duration_sum_ms"], 7.)
        self.assertEqual(summary["groups"][0]["duration_mean_ms"], 3.5)
        path.write_text(path.read_text() + "{}\n")
        result = collect_event_summaries(root)
        self.assertEqual(result["cleanup_summarized_paths"], [])
        self.assertTrue(result["errors"])
        row["execution_events"]["path"] = str(self.root / "source-run/events.jsonl")
        phase(root, "granularity", dict(status="passed", eligible=True, records=[row]))
        self.assertEqual(collect_event_summaries(root)["cleanup_summarized_paths"], [])

    def test_real_final_excel_roundtrip_authenticates_explicit_event_cleanup_only(self):
        from ditserve.compact_results import verify
        root = self._study()
        source_before = {path: path.read_bytes() for path in (self.root / "source-run").rglob("*.json")}
        phase_path = root / "granularity/result.json"
        value = json.loads(phase_path.read_text())
        record = value["records"][0]
        event_path = root / "granularity/arrays/record/execution-events.jsonl"
        event_path.parent.mkdir(parents=True)
        event_path.write_text(json.dumps(dict(request_id="r000", action="denoise", quantum=1,
                                             reason="edf", dispatch_ms=10., end_ms=12.)) + "\n")
        record["rows"][0]["action_counts"] = dict(denoise=1)
        record["execution_events"] = dict(path=str(event_path), count=1, sha256=file_digest(event_path))
        phase(root, "granularity", value)
        result = report(root)
        self.assertEqual(result["excel"]["status"], "verified")
        self.assertEqual(verify(root / "excel")["status"], "verified")
        self.assertEqual(result["cleanup_summarized_paths"], ["granularity/arrays/record/execution-events.jsonl"])
        with gzip.open(root / "excel/data.json.gz", "rt") as stream:
            archive = json.load(stream)
        entry = next(item for item in archive["files"] if item["path"] == result["cleanup_summarized_paths"][0])
        self.assertEqual(entry["kind"], "summarized_text")
        self.assertEqual(entry["summary"]["events"], 1)
        self.assertTrue(event_path.exists())  # Reporting exports; controller owns explicit deletion.
        self.assertEqual(source_before, {path: path.read_bytes() for path in source_before})


if __name__ == "__main__":
    unittest.main()
