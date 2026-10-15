import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.sessions import study
from ditserve.sessions.budget import AUTHORIZATION


class SessionStudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output, self.source, self.ledger = self.root / "run", self.root / "old", self.root / "gpu-budget.json"
        self.source.mkdir()
        self.source_raw = self.source / "old.npy"
        self.source_raw.write_bytes(b"historical evidence")
        self.config = dict(budget_gpu_seconds=None, authorization=AUTHORIZATION, device_preference=["V100"],
            models={"llm": {}, "sana": {}}, phases=list(study.runner.PHASES), prerequisite_run=str(self.source))
        atomic_json(self.ledger, dict(active=None, limit_gpu_seconds=None, used_gpu_seconds=10.))
        prior = dict(status="waiting", phase="completion", controller_id="original")
        atomic_json(self.source / "controller.json", prior)
        atomic_json(self.source / "snapshot.json", prior)
        (self.source / "controller.lock").touch()
        self.amendment = dict(schema_version="ditserve.scheduling.queue_amendment.v1",
            authorization="user_2026_10_15_start_ordered_session_experiments", replacement_run=str(self.output),
            preserved_active_phase="completion", cancelled_unstarted_phases=["granularity", "replicate"],
            stopped_controller_pid=1020, preserved_supervisor_pid=2010, preserved_worker_pid=2011,
            stopped_controller_args=["python", "-m", "ditserve", "research-scheduling-next", "continue"],
            prior_controller_snapshot=str(self.source / "snapshot.json"))
        atomic_json(self.source / "queue-amendment.json", self.amendment)
        self.events, self.now, self.files = [], 0., []

    def evidence(self, phase, **changes):
        value = dict(status="passed", eligible=phase == "qualify", technical_complete=True, returncode=0,
                     phase=phase, main_config_sha256=digest(self.config), gpu_work_started=True)
        value.update(changes)
        atomic_json(self.output / phase / "supervisor.json", value)
        return value

    def execute(self, config, output, ledger, phase, **kwargs):
        self.assertEqual((config, output, ledger), (self.config, self.output, self.ledger))
        self.events.append(("execute", phase))
        return self.evidence(phase)

    def report(self, output, ledger, retention=None):
        self.assertEqual((output, ledger), (self.output, self.ledger))
        state = read_json(output / "controller.json")
        event = "retention_report" if retention is not None else "final_report" if "technical_complete" in state else "report"
        self.events.append((event, state["phase"]))
        return dict(status="complete", archive=dict(status="verified", files=self.files))

    def prior_report(self, root, ledger):
        self.assertEqual(root, self.source)
        self.events.append(("prior_report", "completion"))
        return dict(completion={"technical_complete": True}, excel={"status": "verified"})

    def upstream(self, config):
        return dict(status="passed", path=str(self.source / "completion/supervisor.json"), sha256="abc")

    def sleep(self, seconds):
        self.assertLessEqual(seconds, 5.)
        self.now += seconds

    def run_controller(self, *, execute=None, report=None, upstream=None, sleep=None):
        verify = study.verify_queue_amendment
        with patch.object(study, "verify_queue_amendment", side_effect=lambda config, output: verify(config, output, alive=lambda pid: False)), \
             patch.object(study.runner, "upstream_completion", side_effect=upstream or self.upstream), \
             patch.object(study.runner, "execute", side_effect=execute or self.execute) as executor, \
             patch.object(study.runner, "report", side_effect=report or self.report) as reporter, \
             patch("ditserve.scheduling.next_reporting.report", side_effect=self.prior_report), patch("builtins.print"):
            result = study.continue_study(self.config, self.output, self.ledger, poll_seconds=2., finalize=False,
                monotonic=lambda: self.now, clock=lambda: 1000. + self.now, sleep=sleep or self.sleep)
        self.reporter = reporter
        return result, executor

    def test_amendment_binds_replacement_and_never_accepts_live_controller(self):
        original = (self.source / "controller.json").read_bytes()
        value = study.verify_queue_amendment(self.config, self.output, alive=lambda pid: False)
        self.assertEqual(value["stopped_controller_pid"], 1020)
        with self.assertRaisesRegex(ValueError, "still active"):
            study.verify_queue_amendment(self.config, self.output, alive=lambda pid: True)
        for changes in ({"replacement_run": str(self.root / "wrong")},
                        {"cancelled_unstarted_phases": ["granularity"]}, {"stopped_controller_pid": 2010}):
            atomic_json(self.source / "queue-amendment.json", dict(self.amendment, **changes))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                study.verify_queue_amendment(self.config, self.output, alive=lambda pid: False)
        self.assertEqual((self.source / "controller.json").read_bytes(), original)

    def test_snapshot_formatting_may_differ_but_semantics_and_both_hashes_are_bound(self):
        snapshot = self.source / "snapshot.json"
        snapshot.write_text(json.dumps(read_json(snapshot), separators=(",", ":")), encoding="utf-8")
        value = study.verify_queue_amendment(self.config, self.output, alive=lambda pid: False)
        self.assertNotEqual(value["snapshot_sha256"], value["original_controller_sha256"])
        self.assertEqual(value["snapshot_sha256"], file_digest(snapshot))

    def test_amendment_cannot_cancel_an_existing_attempt_or_ignore_old_controller_lock(self):
        with (self.source / "controller.lock").open("r") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                study.verify_queue_amendment(self.config, self.output, alive=lambda pid: False)
        atomic_json(self.source / "granularity/runtime-config.json", {})
        with self.assertRaisesRegex(ValueError, "already attempted"):
            study.verify_queue_amendment(self.config, self.output, alive=lambda pid: False)

    def test_waits_and_exports_prior_completion_before_any_session_gpu_work(self):
        calls = []
        def upstream(config):
            calls.append(1)
            return dict(status="waiting") if len(calls) == 1 else self.upstream(config)
        result, executor = self.run_controller(upstream=upstream)
        self.assertEqual(self.now, 2.)
        self.assertEqual(self.events[:3], [("prior_report", "completion"), ("execute", "qualify"), ("report", "qualify")])
        self.assertEqual([call.args[3] for call in executor.call_args_list], ["qualify", "pilot", "state"])
        self.assertTrue(result["technical_complete"])
        self.assertEqual(self.events[-2:], [("final_report", "state"), ("retention_report", "state")])
        self.assertEqual(self.source_raw.read_bytes(), b"historical evidence")

    def test_failed_prior_completion_is_reported_and_blocks_all_session_work(self):
        result, executor = self.run_controller(upstream=lambda config: dict(self.upstream(config), status="failed"))
        executor.assert_not_called()
        self.assertFalse(result["technical_complete"])
        self.assertEqual(self.events, [("prior_report", "completion"), ("final_report", "upstream_completion")])

    def test_failed_qualification_stops_before_pilot_and_retains_evidence(self):
        result, executor = self.run_controller(execute=lambda *args, **kwargs: self.evidence(
            args[3], status="stopped", eligible=False, technical_complete=True))
        self.assertEqual(executor.call_count, 1)
        self.assertFalse(result["technical_complete"])
        self.assertEqual(result["cleanup"]["status"], "retained")
        self.assertFalse((self.output / "cleanup-summary.json").exists())

    def test_existing_active_qualification_is_waited_on_without_restart(self):
        atomic_json(self.output / "qualify/runtime-config.json", {})
        atomic_json(self.ledger, dict(active=dict(phase="qualify", identity=dict(output=str(self.output), deadline_monotonic=500.))))
        def finish(seconds):
            self.sleep(seconds)
            self.evidence("qualify")
            atomic_json(self.ledger, dict(active=None))
        result, executor = self.run_controller(sleep=finish)
        self.assertEqual([call.args[3] for call in executor.call_args_list], ["pilot", "state"])
        self.assertTrue(result["technical_complete"])

    def test_existing_phases_and_not_applicable_state_are_not_rerun(self):
        for phase in study.runner.PHASES:
            self.evidence(phase, conclusion="not_applicable" if phase == "state" else "measured")
        result, executor = self.run_controller()
        executor.assert_not_called()
        self.assertTrue(result["technical_complete"])
        original = (self.output / "controller.json").read_bytes()
        self.events.clear()
        repeated, executor = self.run_controller()
        executor.assert_not_called()
        self.assertEqual(repeated, result)
        self.assertEqual((self.output / "controller.json").read_bytes(), original)
        self.assertEqual(self.events, [])

    def test_unsettled_attempt_cannot_restart(self):
        atomic_json(self.output / "qualify/worker.json", {"status": "incomplete"})
        result, executor = self.run_controller()
        executor.assert_not_called()
        self.assertIn("refusing to rerun", result["reason"])

    def test_exception_preserves_evidence_and_attempts_terminal_export(self):
        def fail(*args, **kwargs):
            raise RuntimeError("fixture failure")
        result, unused = self.run_controller(execute=fail)
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(result["cleanup"]["status"], "retained")
        self.assertEqual(self.events[-1], ("final_report", "qualify"))

    def add_file(self, name, data=b"payload", *, eligible=True):
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.files.append(dict(path=name, sha256=file_digest(path), bytes=len(data),
                               coverage="verified measurement", cleanup_eligible=eligible))
        return path

    def test_verified_cleanup_only_removes_explicit_arrays_and_covered_events(self):
        raw = self.add_file("pilot/replay/output.npy")
        events = self.add_file("pilot/replay/events.jsonl", b'{"action":"decode"}\n')
        log = self.add_file("worker.log")
        source = self.add_file("source-snapshots/fixture.py")
        unresolved = self.add_file("state/unverified.npy", eligible=False)
        original_controller = {}
        def report(output, ledger, retention=None):
            value = self.report(output, ledger, retention)
            state = read_json(output / "controller.json")
            if state.get("technical_complete") and retention is None:
                original_controller["bytes"] = (output / "controller.json").read_bytes()
            if retention is not None:
                self.assertEqual({row["path"] for row in retention["files"]},
                                 {"pilot/replay/output.npy", "pilot/replay/events.jsonl"})
                self.assertTrue(all(row["state"] == "deleted" for row in retention["files"]))
            return value
        result, unused = self.run_controller(report=report)
        self.assertFalse(raw.exists())
        self.assertFalse(events.exists())
        self.assertTrue(log.exists() and source.exists() and unresolved.exists())
        self.assertEqual(self.source_raw.read_bytes(), b"historical evidence")
        self.assertEqual(result["actual_cleanup"]["status"], "completed")
        self.assertFalse((self.output / "retention-working.json").exists())
        self.assertEqual(read_json(self.output / "cleanup-summary.json"), result["actual_cleanup"])
        self.assertEqual((self.output / "controller.json").read_bytes(), original_controller["bytes"])

    def test_source_hash_change_or_active_account_prevents_any_deletion(self):
        raw = self.add_file("output.npy")
        def report(output, ledger, retention=None):
            value = self.report(output, ledger, retention)
            if read_json(output / "controller.json").get("technical_complete") and retention is None:
                raw.write_bytes(b"changed")
            return value
        result, unused = self.run_controller(report=report)
        self.assertTrue(raw.exists())
        self.assertEqual(result["actual_cleanup"]["status"], "retained")
        self.assertIn("changed", result["actual_cleanup"]["reason"])

    def test_receipt_export_failure_keeps_retryable_journal(self):
        raw = self.add_file("output.npy")
        def report(output, ledger, retention=None):
            value = self.report(output, ledger, retention)
            if retention is not None:
                raise RuntimeError("receipt export failed")
            return value
        result, unused = self.run_controller(report=report)
        self.assertFalse(raw.exists())
        self.assertEqual(result["actual_cleanup"]["final_report_error"], "receipt export failed")
        self.assertTrue((self.output / "retention-working.json").exists())
        self.assertEqual(read_json(self.output / "cleanup-summary.json")["final_report_error"], "receipt export failed")

    def test_unsafe_inventory_cannot_select_prior_sources(self):
        self.files = [dict(path="../old/old.npy", cleanup_eligible=True)]
        result, unused = self.run_controller()
        self.assertEqual(result["actual_cleanup"]["status"], "retained")
        self.assertEqual(self.source_raw.read_bytes(), b"historical evidence")

    def final_fixture(self):
        from test_sessions_reporting import matrix
        self.config["session_count"] = 2
        study.runner.bind_protocol(self.config, self.output, self.ledger)
        (self.output / "controller.lock").touch()
        atomic_json(self.output / "controller.json", dict(technical_complete=True,
                    config_sha256=digest(self.config), status="completed"))
        for phase in study.runner.PHASES:
            atomic_json(self.output / phase / "result.json", dict(status="passed", records=matrix() if phase == "pilot" else [],
                        conclusion="not_applicable" if phase == "state" else "measured"))
        self.add_file("pilot/events.jsonl", b'{"kind":"llm_decode","start_ms":1}\n')
        self.add_file("pilot/output.npy")
        self.add_file("qualify/sources/fixture.py", b"# archived source identity\n")
        self.add_file("qualify/worker.log", b"complete log\n")

    def test_real_finalizer_retires_control_readers_and_leaves_two_verified_files(self):
        from ditserve.sessions.reporting import archived, load_archive, report
        self.final_fixture()
        before = report(self.output, self.ledger)
        ledger_before = self.ledger.read_bytes()
        result = study.finalize_closed_run(self.config, self.output, self.ledger)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual({path.name for path in self.output.iterdir()}, {"results.xlsx", "REPORT.md"})
        self.assertEqual(self.ledger.read_bytes(), ledger_before)
        self.assertEqual(self.source_raw.read_bytes(), b"historical evidence")
        self.assertTrue(archived(self.output))
        restored = load_archive(self.output)
        self.assertEqual(restored["pilot"], before["pilot"])
        self.assertEqual(restored["phases"], before["phases"])
        self.assertEqual(restored["config"], self.config)
        self.assertTrue(restored["retention"]["final_archive"])
        with patch.object(study.runner, "execute") as execute:
            resumed = study.continue_study(self.config, self.output, self.ledger)
        execute.assert_not_called()
        self.assertTrue(resumed["archive_only"])
        with self.assertRaisesRegex(ValueError, "archived"):
            study.runner.prepare(self.config, self.output, self.ledger)

    def test_finalizer_waits_for_controller_lock_release(self):
        self.final_fixture()
        with (self.output / "controller.lock").open("r") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = study.finalize_closed_run(self.config, self.output, self.ledger)
        self.assertEqual(result["status"], "retained")
        self.assertTrue((self.output / "controller.json").exists())
        self.assertFalse((self.output / "finalizing.json").exists())

    def test_partial_control_cleanup_preserves_complete_archive_and_blocks_reexecution(self):
        from ditserve.sessions.reporting import load_archive, report
        self.final_fixture()
        before = report(self.output, self.ledger)
        unlink = Path.unlink
        def fail_one(path, *args, **kwargs):
            if path == self.output / "qualify/result.json":
                raise OSError("fixture finalization failure")
            return unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", fail_one):
            result = study.finalize_closed_run(self.config, self.output, self.ledger)
        self.assertEqual(result["status"], "retained")
        self.assertTrue((self.output / "finalizing.json").exists())
        self.assertEqual(load_archive(self.output)["pilot"], before["pilot"])
        self.assertEqual(load_archive(self.output)["phases"], before["phases"])
        with self.assertRaisesRegex(ValueError, "finalizing"):
            study.runner.prepare(self.config, self.output, self.ledger)


if __name__ == "__main__":
    unittest.main()
