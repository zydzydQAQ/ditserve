from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, read_json
from ditserve.scheduling import study
from ditserve.scheduling.budget import TOTAL


class SchedulingStudyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output, self.ledger = self.root / "run", self.root / "budget.json"
        self.config = dict(budget_gpu_seconds=TOTAL, device_preference=["V100"], seed=301)
        self.events = []
        self.now = 0.
        atomic_json(self.ledger, dict(active=None, used_gpu_seconds=1.0, limit_gpu_seconds=TOTAL))

    def evidence(self, phase, **changes):
        value = dict(status="passed", eligible=True, phase=phase, returncode=0, gpu_work_started=True)
        value.update(changes)
        atomic_json(self.output / phase / "supervisor.json", value)
        return value

    def export(self, output, ledger):
        self.assertEqual((output, ledger), (self.output, self.ledger))
        state = read_json(self.output / "controller.json")
        self.events.append(("report", state["phase"]))
        return dict(status="descriptive", excel={"status": "verified"})

    def execute(self, config, output, ledger, phase, **kwargs):
        self.assertEqual((config, output, ledger), (self.config, self.output, self.ledger))
        self.assertGreater(kwargs["timeout"], 0)
        self.events.append(("execute", phase))
        value = self.evidence(phase, eligible=phase != "pilot")
        if phase == "validate":
            value = self.evidence(phase, validation={"status": "passed"})
        return value

    def run_controller(self, *, execute=None, sleep=None, wait_seconds=1000., export=None, plan=None, prune=None):
        with patch.object(study.runner, "execute", side_effect=execute or self.execute) as executor, \
             patch.object(study.runner, "report", side_effect=export or self.export), \
             patch.object(study.compact_results, "plan", side_effect=plan,
                          return_value=dict(status="planned", files=[])) as planner, \
             patch.object(study.compact_results, "prune", side_effect=prune,
                          return_value=dict(status="pruned", deleted=[])) as pruner:
            result = study.continue_study(self.config, self.output, self.ledger,
                wait_seconds=wait_seconds, poll_seconds=2., monotonic=lambda: self.now,
                clock=lambda: 1000. + self.now, sleep=sleep or self.sleep)
        self.planner, self.pruner = planner, pruner
        return result, executor

    def sleep(self, duration):
        self.assertLessEqual(duration, 5.)
        self.now += duration

    def test_phase_order_reports_pilot_before_starting_screen_and_defers_legacy(self):
        self.evidence("calibrate")
        result, executor = self.run_controller()
        self.assertEqual(self.events, [("report", "calibrate"), ("execute", "pilot"),
            ("report", "pilot"), ("execute", "screen"), ("report", "screen"),
            ("execute", "validate"), ("report", "validate"), ("report", "validate")])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["scientific_complete"])
        self.assertEqual(result["cleanup"]["status"], "planned_verified_prune")
        self.assertEqual(result["actual_cleanup"]["status"], "no_raw_arrays")
        self.assertEqual(executor.call_count, 3)
        legacy = read_json(self.output / "legacy/admission.json")
        self.assertTrue(legacy["deferred"])
        self.assertFalse(legacy["gpu_work_started"])

    def test_waits_for_externally_running_calibration_without_starting_a_new_one(self):
        def wake(duration):
            self.sleep(duration)
            self.evidence("calibrate")
        result, executor = self.run_controller(sleep=wake)
        self.assertEqual(self.now, 2.)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([c.args[3] for c in executor.call_args_list], ["pilot", "screen", "validate"])

    def test_failed_or_ineligible_calibration_stops_and_reports_without_running_pilot(self):
        for status, eligible in (("failed", False), ("passed", False), ("incomplete", False)):
            with self.subTest(status=status):
                self.events.clear()
                self.evidence("calibrate", status=status, eligible=eligible, reason="qualification gate")
                result, executor = self.run_controller()
                executor.assert_not_called()
                self.assertEqual(result["status"], "stopped")
                self.assertFalse(result["scientific_complete"])
                self.assertEqual(self.events, [("report", "calibrate")])
                self.assertEqual(result["cleanup"]["status"], "retained")

    def test_budget_failure_is_reported_and_no_later_phase_is_started(self):
        self.evidence("calibrate")
        def exhausted(config, output, ledger, phase, **kwargs):
            self.events.append(("execute", phase))
            return dict(status="stopped", eligible=False, reason_code="budget_insufficient", gpu_work_started=False)
        result, executor = self.run_controller(execute=exhausted)
        self.assertEqual(executor.call_count, 1)
        self.assertEqual(self.events, [("report", "calibrate"), ("execute", "pilot"), ("report", "pilot")])
        self.assertFalse(result["scientific_complete"])
        self.assertEqual(result["cleanup"]["status"], "retained")

    def test_completed_screen_without_candidate_stops_before_holdout(self):
        self.evidence("calibrate")
        self.evidence("pilot", eligible=False)
        self.evidence("screen", status="stopped", eligible=False, selection={"status": "failed"})
        result, executor = self.run_controller()
        executor.assert_not_called()
        self.assertEqual(self.events, [("report", "calibrate"), ("report", "pilot"),
                                       ("report", "screen"), ("report", "screen")])
        self.assertTrue(result["scientific_complete"])
        self.assertEqual(result["cleanup"]["status"], "planned_verified_prune")

    def test_existing_completed_phases_are_consumed_without_rerun(self):
        for phase in ("calibrate", "pilot", "screen"):
            self.evidence(phase, eligible=phase != "pilot")
        self.evidence("validate", status="stopped", eligible=False, validation={"status": "failed"})
        result, executor = self.run_controller()
        executor.assert_not_called()
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["scientific_complete"])
        self.assertEqual(self.events, [("report", phase) for phase in
                                       ("calibrate", "pilot", "screen", "validate", "validate")])

    def test_existing_unsettled_attempt_is_not_retried(self):
        self.evidence("calibrate")
        atomic_json(self.output / "pilot/worker.json", {"status": "incomplete"})
        result, executor = self.run_controller()
        executor.assert_not_called()
        self.assertEqual(result["status"], "stopped")
        self.assertIn("refusing to rerun", result["reason"])
        self.assertEqual(result["cleanup"]["status"], "retained")

    def test_existing_active_phase_is_waited_on_not_executed_again(self):
        self.evidence("calibrate")
        atomic_json(self.output / "pilot/runtime-config.json", {})
        atomic_json(self.ledger, dict(active=dict(phase="pilot", identity={"output": str(self.output)})))
        def finish_pilot(duration):
            self.sleep(duration)
            self.evidence("pilot", eligible=False)
            atomic_json(self.ledger, dict(active=None))
        result, executor = self.run_controller(sleep=finish_pilot)
        self.assertEqual([c.args[3] for c in executor.call_args_list], ["screen", "validate"])
        self.assertEqual(result["status"], "completed")

    def test_bounded_wait_times_out_with_evidence_retained(self):
        result, executor = self.run_controller(wait_seconds=3.)
        executor.assert_not_called()
        self.assertEqual(self.now, 3.)
        self.assertEqual(result["status"], "stopped")
        self.assertIn("wait horizon exhausted", result["reason"])
        self.assertEqual(result["cleanup"]["status"], "retained")

    def test_reporting_error_prevents_starting_next_phase(self):
        self.evidence("calibrate")
        with patch.object(study.runner, "report", side_effect=RuntimeError("export failed")), \
             patch.object(study.runner, "execute") as executor:
            result = study.continue_study(self.config, self.output, self.ledger, wait_seconds=1000.)
        executor.assert_not_called()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["report_error"], "export failed")

    def test_limits_reject_excessive_wait_and_polling(self):
        for kwargs in (dict(wait_seconds=TOTAL + 1), dict(poll_seconds=6)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                study.continue_study(self.config, self.output, self.ledger, **kwargs)

    def test_terminal_save_precedes_final_export_then_only_raw_npy_is_pruned(self):
        self.evidence("calibrate")
        final = {}

        def export(output, ledger):
            report = self.export(output, ledger)
            state = read_json(output / "controller.json")
            if state.get("scientific_complete"):
                self.assertEqual(state["status"], "completed")
                self.assertEqual(state["cleanup"]["status"], "planned_verified_prune")
                final["controller"] = (output / "controller.json").read_bytes()
                self.events.append(("final_export", "validate"))
            return report

        def plan(output, *, destination):
            self.assertEqual(destination, output / "excel")
            self.assertEqual((output / "controller.json").read_bytes(), final["controller"])
            self.events.append(("plan", "validate"))
            return dict(status="planned", files=[
                dict(path="validate/raw/output.npy", kind="raw"),
                dict(path="screen/raw/reference.npy", kind="raw"),
                dict(path="screen/raw/matrix.npz", kind="raw"),
                dict(path="validate/result.json", kind="text"),
                dict(path="validate/worker.log", kind="text"),
                dict(path="calibrate/sources/worker.py", kind="retained"),
                dict(path="retained.npy", kind="retained")])

        def prune(destination, paths):
            self.assertEqual((self.output / "controller.json").read_bytes(), final["controller"])
            self.assertFalse((self.output / "cleanup-summary.json").exists())
            self.assertEqual(destination, self.output / "excel")
            self.assertEqual(paths, ["screen/raw/reference.npy", "validate/raw/output.npy"])
            self.events.append(("prune", "validate"))
            return dict(status="pruned", deleted=paths, bytes=100)

        result, unused = self.run_controller(export=export, plan=plan, prune=prune)
        self.assertEqual(self.events[-3:], [("final_export", "validate"), ("plan", "validate"), ("prune", "validate")])
        self.assertEqual((self.output / "controller.json").read_bytes(), final["controller"])
        self.assertNotIn("actual_cleanup", read_json(self.output / "controller.json"))
        self.assertEqual(read_json(self.output / "cleanup-summary.json"), result["actual_cleanup"])
        self.assertEqual(result["actual_cleanup"]["status"], "pruned")

    def test_unverified_final_excel_retains_raw_without_planning(self):
        self.evidence("calibrate")
        def export(output, ledger):
            value = self.export(output, ledger)
            if read_json(output / "controller.json").get("scientific_complete"):
                value["excel"]["status"] = "deferred"
            return value
        result, unused = self.run_controller(export=export)
        self.assertEqual(result["actual_cleanup"]["status"], "retained")
        self.assertIn("not verified", result["actual_cleanup"]["reason"])
        self.planner.assert_not_called()
        self.pruner.assert_not_called()

    def test_active_ledger_after_final_export_retains_all_files(self):
        self.evidence("calibrate")
        def export(output, ledger):
            value = self.export(output, ledger)
            if read_json(output / "controller.json").get("scientific_complete"):
                atomic_json(ledger, dict(active={"phase": "other"}))
            return value
        result, unused = self.run_controller(export=export)
        self.assertEqual(result["actual_cleanup"]["status"], "retained")
        self.planner.assert_not_called()
        self.pruner.assert_not_called()

    def test_ledger_is_rechecked_between_plan_and_prune(self):
        self.evidence("calibrate")
        def plan(output, *, destination):
            atomic_json(self.ledger, dict(active={"phase": "other"}))
            return dict(status="planned", files=[dict(path="raw.npy", kind="raw")])
        result, unused = self.run_controller(plan=plan)
        self.assertEqual(result["actual_cleanup"]["status"], "retained")
        self.assertIn("became active", result["actual_cleanup"]["reason"])
        self.pruner.assert_not_called()

    def test_cleanup_failure_does_not_rewrite_terminal_controller_or_report(self):
        self.evidence("calibrate")
        final = {}
        def plan(output, *, destination):
            final["controller"] = (output / "controller.json").read_bytes()
            final["events"] = list(self.events)
            raise RuntimeError("protected inventory")
        result, unused = self.run_controller(plan=plan)
        self.assertEqual(result["actual_cleanup"]["status"], "retained")
        self.assertEqual(result["actual_cleanup"]["reason"], "protected inventory")
        self.assertEqual((self.output / "controller.json").read_bytes(), final["controller"])
        self.assertEqual(self.events, final["events"])
        self.pruner.assert_not_called()

    def test_completed_controller_restart_is_read_only(self):
        self.evidence("calibrate")
        first, unused = self.run_controller()
        before = (self.output / "controller.json").read_bytes()
        self.events.clear()
        result, executor = self.run_controller()
        self.assertEqual(result, first)
        self.assertEqual((self.output / "controller.json").read_bytes(), before)
        self.assertEqual(self.events, [])
        executor.assert_not_called()
        self.planner.assert_not_called()
        self.pruner.assert_not_called()

    def test_new_summary_sidecar_never_overwrites_existing_metadata(self):
        self.evidence("calibrate")
        summary = self.output / "cleanup-summary.json"
        summary.write_text("original metadata\n", encoding="utf-8")
        result, unused = self.run_controller()
        self.assertEqual(summary.read_text(encoding="utf-8"), "original metadata\n")
        self.assertNotEqual(result["cleanup"]["summary_path"], str(summary))
        self.assertEqual(read_json(Path(result["cleanup"]["summary_path"])), result["actual_cleanup"])

    def test_real_cpu_export_and_prune_accept_terminal_metadata_and_keep_logs(self):
        self.evidence("calibrate")
        raw, log = self.output / "raw.npy", self.output / "worker.log"
        raw.write_bytes(b"tiny array payload")
        log.write_text("complete execution log\n", encoding="utf-8")
        def export(output, ledger):
            self.export(output, ledger)
            return dict(status="descriptive", excel=study.compact_results.export(
                output, output / "excel", replace_existing=True))
        with patch.object(study.runner, "execute", side_effect=self.execute), \
             patch.object(study.runner, "report", side_effect=export):
            result = study.continue_study(self.config, self.output, self.ledger,
                wait_seconds=1000., monotonic=lambda: self.now, clock=lambda: 1000. + self.now, sleep=self.sleep)
        self.assertEqual(result["actual_cleanup"]["status"], "pruned")
        self.assertEqual(result["actual_cleanup"]["result"]["deleted"], ["raw.npy"])
        self.assertFalse(raw.exists())
        self.assertEqual(log.read_text(encoding="utf-8"), "complete execution log\n")
        self.assertEqual(read_json(self.output / "excel/pruned.json")["status"], "completed")
        self.assertEqual(study.compact_results.verify(self.output / "excel")["status"], "verified")


if __name__ == "__main__":
    unittest.main()
