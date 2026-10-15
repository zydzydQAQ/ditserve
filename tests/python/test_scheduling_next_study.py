import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.scheduling import next_study as study
from ditserve.scheduling.next_budget import AUTHORIZATION


class NextStudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output, self.source, self.ledger = self.root / "run", self.root / "source", self.root / "gpu-budget.json"
        self.source.mkdir()
        self.source_raw = self.source / "old.npy"
        self.source_raw.write_bytes(b"historical evidence")
        self.config = dict(budget_gpu_seconds=None, authorization=AUTHORIZATION, device_preference=["V100"],
            models={"wan": {}, "sana": {}}, phases=list(study.runner.PHASES), source_run=str(self.source))
        atomic_json(self.ledger, dict(active=None, limit_gpu_seconds=None, used_gpu_seconds=10.))
        self.events, self.now = [], 0.

    def evidence(self, phase, **changes):
        value = dict(status="passed", eligible=False, technical_complete=True, returncode=0,
                     phase=phase, main_config_sha256=digest(self.config), gpu_work_started=True)
        value.update(changes)
        atomic_json(self.output / phase / "supervisor.json", value)
        return value

    def execute(self, config, output, ledger, phase, **kwargs):
        self.assertEqual((config, output, ledger), (self.config, self.output, self.ledger))
        self.assertEqual(read_json(output / "controller.json")["status"], "running")
        self.events.append(("execute", phase))
        return self.evidence(phase)

    def report(self, output, ledger):
        self.assertEqual((output, ledger), (self.output, self.ledger))
        state = read_json(output / "controller.json")
        self.events.append(("final_report" if "technical_complete" in state else "report", state["phase"]))
        return dict(status="complete", excel=dict(status="verified"), cleanup_summarized_paths=[])

    def sleep(self, seconds):
        self.assertLessEqual(seconds, 5.)
        self.now += seconds

    def run_controller(self, *, execute=None, report=None, sleep=None, plan=None, prune=None):
        with patch.object(study.runner, "execute", side_effect=execute or self.execute) as executor, \
             patch.object(study.runner, "report", side_effect=report or self.report) as reporter, \
             patch.object(study.compact_results, "plan", side_effect=plan,
                          return_value=dict(status="planned", files=[])) as planner, \
             patch.object(study.compact_results, "prune", side_effect=prune,
                          return_value=dict(status="pruned", deleted=[])) as pruner:
            result = study.continue_study(self.config, self.output, self.ledger, poll_seconds=2.,
                                         monotonic=lambda: self.now, sleep=sleep or self.sleep,
                                         clock=lambda: 1000. + self.now)
        self.reporter, self.planner, self.pruner = reporter, planner, pruner
        return result, executor

    def test_all_phases_report_before_next_even_when_scientifically_unqualified(self):
        result, executor = self.run_controller()
        self.assertEqual(self.events, [("execute", "completion"), ("report", "completion"),
            ("execute", "granularity"), ("report", "granularity"),
            ("execute", "replicate"), ("report", "replicate"), ("final_report", "replicate")])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["technical_complete"])
        self.assertEqual(executor.call_count, 3)
        self.assertEqual(result["actual_cleanup"]["status"], "no_eligible_files")
        self.assertTrue(all(not row["eligible"] for row in result["phases"].values()))
        self.assertEqual(self.source_raw.read_bytes(), b"historical evidence")

    def test_first_technical_failure_stops_and_still_exports_terminal_report(self):
        def fail(*args, **kwargs):
            self.events.append(("execute", args[3]))
            return self.evidence(args[3], status="incomplete", technical_complete=False)
        result, executor = self.run_controller(execute=fail)
        self.assertEqual(executor.call_count, 1)
        self.assertEqual(self.events, [("execute", "completion"), ("report", "completion"),
                                       ("final_report", "completion")])
        self.assertFalse(result["technical_complete"])
        self.assertEqual(result["cleanup"]["status"], "retained")
        self.planner.assert_not_called()
        self.pruner.assert_not_called()

    def test_busy_v100_can_wait_without_global_budget_horizon(self):
        attempts = []
        def execute(*args, **kwargs):
            attempts.append(args[3])
            if len(attempts) == 1:
                return dict(status="blocked", gpu_work_started=False, reason_code="device_unavailable")
            return self.execute(*args, **kwargs)
        def sleep(seconds):
            self.sleep(seconds)
            self.now += 200000.
        result, unused = self.run_controller(execute=execute, sleep=sleep)
        self.assertEqual(attempts, ["completion", "completion", "granularity", "replicate"])
        self.assertGreater(self.now, 172800.)
        self.assertEqual(result["status"], "completed")

    def test_existing_successful_phases_are_read_without_rerun(self):
        for phase in study.runner.PHASES:
            self.evidence(phase, status="stopped", technical_complete=True)
        result, executor = self.run_controller()
        executor.assert_not_called()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.events, [("report", phase) for phase in study.runner.PHASES] + [("final_report", "replicate")])

    def test_existing_active_attempt_is_waited_on_not_restarted(self):
        atomic_json(self.output / "completion/runtime-config.json", {})
        atomic_json(self.ledger, dict(active=dict(phase="completion", identity=dict(
            output=str(self.output), deadline_monotonic=500.))))
        def finish_existing(seconds):
            self.sleep(seconds)
            self.evidence("completion")
            atomic_json(self.ledger, dict(active=None))
        result, executor = self.run_controller(sleep=finish_existing)
        self.assertEqual([call.args[3] for call in executor.call_args_list], ["granularity", "replicate"])
        self.assertEqual(result["status"], "completed")

    def test_unsettled_attempt_is_preserved_and_never_restarted(self):
        atomic_json(self.output / "completion/worker.json", dict(status="incomplete"))
        result, executor = self.run_controller()
        executor.assert_not_called()
        self.assertEqual(result["status"], "stopped")
        self.assertIn("refusing to rerun", result["reason"])
        self.assertEqual(self.now, 10.)

    def test_exception_still_attempts_a_final_report(self):
        result, unused = self.run_controller(execute=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fixture error")))
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(self.events, [("final_report", "completion")])
        self.pruner.assert_not_called()

    def test_single_controller_lock_rejects_concurrent_execution(self):
        study.runner.bind_protocol(self.config, self.output, self.ledger)
        with (self.output / "controller.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_controller()

    def test_final_export_precedes_explicit_raw_and_summarized_event_prune(self):
        final = {}
        def report(output, ledger):
            value = self.report(output, ledger)
            if read_json(output / "controller.json").get("technical_complete"):
                final["controller"] = (output / "controller.json").read_bytes()
                value["cleanup_summarized_paths"] = ["replicate/replay/execution-events.jsonl"]
            return value
        def plan(output, *, destination):
            self.assertEqual((output / "controller.json").read_bytes(), final["controller"])
            self.events.append(("plan", "replicate"))
            return dict(status="planned", files=[dict(path="replicate/raw.npy", kind="raw"),
                dict(path="replicate/replay/execution-events.jsonl", kind="text"),
                dict(path="replicate/isolation.jsonl", kind="text"), dict(path="raw.npz", kind="raw"),
                dict(path="worker.log", kind="text"), dict(path="report.json", kind="text")])
        def prune(destination, paths):
            self.assertFalse((self.output / "cleanup-summary.json").exists())
            self.assertEqual(paths, ["replicate/raw.npy", "replicate/replay/execution-events.jsonl"])
            self.assertEqual((self.output / "controller.json").read_bytes(), final["controller"])
            self.events.append(("prune", "replicate"))
            return dict(status="pruned", deleted=paths)
        result, unused = self.run_controller(report=report, plan=plan, prune=prune)
        self.assertEqual(self.events[-3:], [("final_report", "replicate"), ("plan", "replicate"), ("prune", "replicate")])
        self.assertEqual((self.output / "controller.json").read_bytes(), final["controller"])
        self.assertEqual(read_json(self.output / "cleanup-summary.json"), result["actual_cleanup"])
        self.assertEqual(self.source_raw.read_bytes(), b"historical evidence")

    def test_source_run_or_unsafe_summary_is_never_selected(self):
        def report(output, ledger):
            value = self.report(output, ledger)
            value["cleanup_summarized_paths"] = ["../source/execution-events.jsonl"]
            return value
        result, unused = self.run_controller(report=report)
        self.assertEqual(result["actual_cleanup"]["status"], "retained")
        self.pruner.assert_not_called()
        self.assertEqual(self.source_raw.read_bytes(), b"historical evidence")

    def test_unverified_export_retains_files_and_completed_restart_is_read_only(self):
        def report(output, ledger):
            value = self.report(output, ledger)
            value["excel"]["status"] = "deferred"
            return value
        first, unused = self.run_controller(report=report)
        self.assertEqual(first["actual_cleanup"]["status"], "retained")
        self.pruner.assert_not_called()
        original = (self.output / "controller.json").read_bytes()
        self.events.clear()
        result, executor = self.run_controller()
        self.assertEqual(result, first)
        self.assertEqual((self.output / "controller.json").read_bytes(), original)
        executor.assert_not_called()
        self.reporter.assert_not_called()

    def test_source_tree_overlap_is_rejected_before_writing(self):
        for output in (self.source, self.source / "child", self.root):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "separate directory"):
                study.continue_study(self.config, output, self.ledger)

    def test_real_cpu_archive_prunes_only_new_arrays_and_validated_summarized_events(self):
        self.output.mkdir()
        raw, events = self.output / "raw.npy", self.output / "execution-events.jsonl"
        raw.write_bytes(b"new raw array")
        events.write_text('{"action":"advance"}\n', encoding="utf-8")
        log = self.output / "worker.log"
        log.write_text("keep log\n", encoding="utf-8")
        def report(output, ledger):
            self.report(output, ledger)
            summaries = {"execution-events.jsonl": dict(valid=True, records=1,
                source_metadata=dict(path=str(events), sha256=file_digest(events), size=events.stat().st_size))}
            return dict(status="complete", cleanup_summarized_paths=["execution-events.jsonl"],
                excel=study.compact_results.export(output, output / "excel", replace_existing=True,
                                                  summarized_text=summaries))
        with patch.object(study.runner, "execute", side_effect=self.execute), \
             patch.object(study.runner, "report", side_effect=report):
            result = study.continue_study(self.config, self.output, self.ledger,
                monotonic=lambda: self.now, sleep=self.sleep, clock=lambda: 1000. + self.now)
        self.assertEqual(result["actual_cleanup"]["status"], "pruned")
        self.assertFalse(raw.exists())
        self.assertFalse(events.exists())
        self.assertEqual(log.read_text(encoding="utf-8"), "keep log\n")
        self.assertEqual(self.source_raw.read_bytes(), b"historical evidence")
        receipt = read_json(self.output / "excel/pruned.json")
        self.assertEqual(receipt["deleted"]["execution-events.jsonl"]["evidence_status"], "aggregated_only")
        self.assertEqual(study.compact_results.verify(self.output / "excel")["status"], "verified")


if __name__ == "__main__":
    unittest.main()
