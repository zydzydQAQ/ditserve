"""Screen failures must prevent confirmation even with unlimited budget."""

import copy
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_occupancy_report as report
from ditserve.evaluation import workflow_occupancy_study as study
from ditserve.evaluation import workflow_next_study as api

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("occupancy_driver", REPO / "docker/run-next-study.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class OccupancyDriverTests(unittest.TestCase):
    def run_matrix(self, technical, performance):
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
             patch.object(driver.os, "sched_getaffinity", return_value={58}), \
             patch.object(study, "register", return_value=rows), \
             patch.object(driver, "run_job", return_value=technical) as run, \
             patch.object(driver.report, "evidence", return_value={}), \
             patch.object(report, "screening_gate", return_value=dict(passed=performance)) as screen, \
             patch.object(report, "summarize", return_value={}), \
             patch.object(driver.report, "build_report"), patch.object(driver, "archive") as archive, \
             redirect_stdout(io.StringIO()):
            driver.execute(current, optimize_occupancy=True)
        return current, rows, run, screen, archive

    def test_failed_technical_qualification_never_runs_confirmation(self):
        current, rows, run, screen, archive = self.run_matrix(False, True)
        self.assertEqual(run.call_count, 1)
        screen.assert_not_called()
        self.assertTrue(all(r["decision"] == "occupancy_screening_not_promising" for r in rows[1:]))
        archive.assert_called_once()

    def test_failed_performance_screen_never_runs_confirmation(self):
        _, _, run, screen, _ = self.run_matrix(True, False)
        self.assertEqual(run.call_count, 1)
        screen.assert_called_once()

    def test_passing_screen_runs_three_independent_confirmations(self):
        current, rows, run, _, _ = self.run_matrix(True, True)
        self.assertEqual([call.args[1]["id"] for call in run.call_args_list], [r["id"] for r in rows])
        self.assertEqual(current["status"], "settled")


if __name__ == "__main__":
    unittest.main()
