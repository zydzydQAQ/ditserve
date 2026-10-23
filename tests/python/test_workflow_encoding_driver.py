"""Explicit unlimited authority retains accounting and prerequisite gates."""

import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_next_study as api

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("encoding_driver", REPO / "docker/run-next-study.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class EncodingDriverTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.study = dict(root=str(self.root), run_id="run03", budget_seconds=28800,
            source=str(REPO), source_digest="fixture", image="fixture", active_job=None,
            family_caps_seconds=copy.deepcopy(api.CAPS), jobs=[], decisions=[], selections={})
        self.budget = dict(schema="ditserve.workflow.budget.v1", limit_seconds=28800,
            charged_seconds=27403., attempts=[dict(job_id="old", charged_seconds=27403., status="completed")], active=None)
        api.save(self.study)
        api.atomic_json(self.root / "budget.json", self.budget)

    def test_explicit_authorization_preserves_old_attempts_and_family_caps(self):
        before = copy.deepcopy(self.study)
        driver.authorize_unlimited(self.study, "取消预算，GPU预算无限，继续运行")
        current = api.read(self.root / "budget.json")
        self.assertIsNone(self.study["budget_seconds"])
        self.assertEqual(self.study["budget_mode"], "unlimited")
        self.assertIsNone(current["limit_seconds"])
        self.assertEqual(current["charged_seconds"], self.budget["charged_seconds"])
        self.assertEqual(current["attempts"], self.budget["attempts"])
        self.assertEqual(self.study["family_caps_seconds"], before["family_caps_seconds"])
        self.assertTrue(self.study["budget_authorizations"])
        self.assertEqual(api.read(self.root / "study.json"), self.study)
        self.assertIsNone(driver.remaining(self.study))
        self.assertIsNone(driver.family_available(self.study, "extension"))

    def test_active_work_rejects_budget_change_without_mutation(self):
        self.budget["active"] = dict(job_id="active")
        api.atomic_json(self.root / "budget.json", self.budget)
        before = (self.root / "budget.json").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "active"):
            driver.authorize_unlimited(self.study, "unlimited authorized")
        self.assertEqual((self.root / "budget.json").read_bytes(), before)
        self.assertEqual(self.study["budget_seconds"], 28800)

    def test_resume_after_ledger_conversion_repairs_study_without_duplicate_authorization(self):
        original = copy.deepcopy(self.study)
        driver.authorize_unlimited(self.study, "unlimited authorized")
        converted = api.read(self.root / "budget.json")
        api.save(original)  # Simulate interruption between the two atomic writes.
        driver.authorize_unlimited(original, "unlimited authorized")
        self.assertIsNone(original["budget_seconds"])
        self.assertEqual(api.read(self.root / "budget.json"), converted)
        self.assertEqual(len(original["budget_authorizations"]), 1)

    def job(self):
        job = dict(id="encoding-test", family="extension", requires=[], estimated_seconds=1800,
                   source=str(REPO), source_digest="fixture", timeout_seconds=3600)
        self.study["jobs"].append(job)
        return job

    def test_unlimited_admission_keeps_accounted_bundle_and_finite_task_timeout(self):
        self.study.update(budget_seconds=None, budget_mode="unlimited")
        self.budget.update(limit_seconds=None, charged_seconds=100000.)
        api.atomic_json(self.root / "budget.json", self.budget)
        job = self.job()

        def launched(command, **kwargs):
            bundle = api.read(self.root / "jobs/encoding-test.json")
            self.assertIn("budget_seconds", bundle)
            self.assertIsNone(bundle["budget_seconds"])
            self.assertEqual(bundle["jobs"][0]["timeout_seconds"], 3600)
            self.assertNotIn("budget_ceiling_seconds", bundle["jobs"][0])
            api.atomic_json(self.root / "launches/encoding-test/launch.json", dict(status="completed"))
            return SimpleNamespace(returncode=0)

        with patch.object(driver.subprocess, "run", side_effect=launched), \
             patch.object(driver.report, "evidence", return_value=dict(qualified=True)), \
             patch.object(driver.report, "build_report"), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(driver.run_job(self.study, job))
        self.assertEqual(api.read(self.root / "budget.json"), self.budget)

    def test_unlimited_does_not_bypass_numeric_prerequisite(self):
        self.study["budget_seconds"] = None
        job = self.job()
        job["requires"] = ["numeric"]
        with patch.object(driver.report, "current_job", return_value={}), \
             patch.object(driver.report, "evidence", return_value=dict(qualified=False)), \
             patch.object(driver.subprocess, "run") as launch:
            self.assertFalse(driver.run_job(self.study, job))
            launch.assert_not_called()
        self.assertEqual(job["decision"], "prerequisite_not_qualified")


if __name__ == "__main__":
    unittest.main()
