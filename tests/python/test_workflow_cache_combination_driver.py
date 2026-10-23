"""A screening gate, not an unlimited budget, controls new confirmation runs."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_cache_combination_report as report
from ditserve.evaluation import workflow_cache_combination_study as study
from ditserve.evaluation import workflow_next_study as api

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cache_combination_driver", REPO / "docker/run-next-study.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class CacheCombinationDriverTests(unittest.TestCase):
    def run_matrix(self, technical, performance, *, results=None, affinity={58}):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        current = dict(root=str(root), source=str(REPO), source_digest="fixture", budget_seconds=None,
            source_revisions=[dict(source=str(REPO))], active_job=None, decisions=[], selections={})
        rows = [dict(id=f"screen-{i}", metadata=dict(stage="screening" if i == 0 else "confirmation"))
                for i in range(4)]
        api.atomic_json(root / "budget.json", dict(charged_seconds=90000., active=None))
        recovery = SimpleNamespace(recover_study=lambda *a, **kw: dict(status="clean"))
        with patch.object(driver, "verify_revision"), patch.object(driver, "helper", return_value=recovery), \
             patch.object(driver.os, "sched_getaffinity", return_value=affinity), \
             patch.object(study, "register", return_value=rows) as register, \
             patch.object(driver, "run_job", return_value=technical, side_effect=results) as run, \
             patch.object(driver.report, "evidence", return_value={}), \
             patch.object(report, "screening_gate", return_value=dict(passed=performance)) as screen, \
             patch.object(report, "summarize", return_value={}), \
             patch.object(driver.report, "build_report"), patch.object(driver, "archive") as archive, \
             redirect_stdout(io.StringIO()):
            error = None
            try:
                driver.execute(current, optimize_cache_combination=True)
            except RuntimeError as exc:
                error = exc
        return current, rows, run, screen, archive, register, error

    def test_failed_technical_screen_never_runs_confirmation(self):
        current, rows, run, screen, archive, _, error = self.run_matrix(False, True)
        self.assertIsNone(error)
        self.assertEqual(run.call_count, 1)
        screen.assert_not_called()
        self.assertTrue(all(r["decision"] == "cache_combination_screening_not_promising" for r in rows[1:]))
        self.assertEqual(current["status"], "settled")
        archive.assert_called_once()

    def test_failed_performance_screen_never_runs_confirmation(self):
        _, _, run, screen, archive, _, error = self.run_matrix(True, False)
        self.assertIsNone(error)
        self.assertEqual(run.call_count, 1)
        screen.assert_called_once()
        archive.assert_called_once()

    def test_passing_screen_runs_three_fresh_confirmations(self):
        current, rows, run, _, archive, _, error = self.run_matrix(True, True)
        self.assertIsNone(error)
        self.assertEqual([call.args[1]["id"] for call in run.call_args_list], [r["id"] for r in rows])
        self.assertEqual(current["status"], "settled")
        archive.assert_called_once()

    def test_technical_confirmation_failure_stops_later_jobs(self):
        _, _, run, _, archive, _, error = self.run_matrix(True, True, results=[True, False])
        self.assertRegex(str(error), "confirmation did not qualify")
        self.assertEqual(run.call_count, 2)
        archive.assert_not_called()

    def test_wrong_observer_cpu_stops_before_registration(self):
        _, _, run, _, archive, register, error = self.run_matrix(True, True, affinity={0, 58})
        self.assertRegex(str(error), "CPU 58")
        register.assert_not_called()
        run.assert_not_called()
        archive.assert_not_called()

    def test_execution_modes_are_mutually_exclusive(self):
        for flag in ("retry", "only", "diagnose_sp", "optimize_followup", "accept_sp2", "optimize_runtime",
                     "optimize_image_cache", "optimize_encoding", "optimize_dispatch_refresh", "optimize_occupancy"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "cannot be combined"):
                driver.execute({}, optimize_cache_combination=True, **{flag: True})

    def test_cli_rejects_missing_run_and_combined_modes(self):
        for arguments in (["--optimize-cache-combination"],
                          ["--run", "--optimize-cache-combination", "--optimize-occupancy"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                driver.main(arguments)
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
