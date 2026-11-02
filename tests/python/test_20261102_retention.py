"""Local retention planning contracts; apply tests prohibit physical deletion."""
from contextlib import redirect_stdout
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve import results


SCRIPT = Path(os.environ.get("DITSERVE_KAIROS_RUNTIME_ROOT",
    Path(__file__).resolve().parents[3] / ".local/ditserve-kairos-rewrite-runtime")) / "scripts/ditserve_output_retention.py"
if not SCRIPT.is_file():
    raise unittest.SkipTest("local operator retention script is not distributed with the package")
SPEC = importlib.util.spec_from_file_location("retention_20261102", SCRIPT)
retention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retention)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "output/DITSERVE"
        self.work = self.root / ".cache/results-work/test/run01"
        self.work.mkdir(parents=True)
        self.final = self.root / "results/test/run01"
        self.destination = self.root / "reports/output-retention-test"
        self.destination.mkdir(parents=True)
        self.manifest = self.destination / "verified-reports.json"
        self.run = dict(schema="ditserve.run.v1", experiment="test", run_id="run01", status="completed",
                        config={}, phases=[], requests=[], failures=[], metrics={},
                        budget=dict(active=None, charges=[]), artifacts=[])

    def publish(self, format="summary"):
        (self.work / "run.json").write_text(json.dumps(self.run), encoding="utf-8")
        outcome = results.finalize(self.work, self.final, format=format, prune=False)
        self.assertEqual(outcome["status"], "completed", outcome)
        return outcome

    def manifest_report(self):
        with redirect_stdout(io.StringIO()):
            retention.command_manifest(SimpleNamespace(root=self.root, source_root=None, experiment=None,
                                                       run_id=None, output=self.manifest))

    def plan(self):
        return retention.build_plan(self.root, self.manifest, False, "run01")

    def test_summary_plan_binds_both_files_and_source(self):
        self.publish()
        self.manifest_report()
        plan = self.plan()
        self.assertEqual(len(plan["work"]), 1)
        self.assertEqual(plan["reports"][0]["format"], retention.RESULT_SUMMARY_SCHEMA)
        self.assertIn("summary_sha256", plan["reports"][0])
        self.assertEqual(plan["work"][0]["run_json"], plan["reports"][0]["run_json"])
        self.assertFalse(plan["protected"])
        self.assertAlmostEqual(retention.numeric_summary([0, 10])["p99"], 9.9)

    def test_legacy_workbook_and_manifest_still_validate(self):
        self.publish(format="full")
        self.manifest_report()
        manifest = json.loads(self.manifest.read_text())
        manifest["reports"][0].pop("format")
        manifest["reports"][0].pop("run_json")
        self.manifest.write_text(json.dumps(manifest))
        self.assertEqual(len(self.plan()["work"]), 1)

    def test_pending_consumers_cannot_be_forged_closed_in_manifest(self):
        self.run["retention"] = {"pending_dependencies": ["quality"]}
        self.publish()
        self.manifest_report()
        manifest = json.loads(self.manifest.read_text())
        manifest["reports"][0]["pending_dependencies"] = []
        self.manifest.write_text(json.dumps(manifest))
        plan = self.plan()
        self.assertFalse(plan["work"])
        self.assertIn("pending dependencies", plan["protected"][0]["reason"])
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            retention.build_plan(self.root, self.manifest, True, "run01")

    def test_legitimate_closure_requires_refinalization_and_new_manifest(self):
        self.run["retention"] = {"pending_dependencies": ["quality"]}
        self.publish()
        self.manifest_report()
        self.run["retention"]["pending_dependencies"] = [{"name": "quality", "closed": True, "reason": "comparison complete"}]
        (self.work / "run.json").write_text(json.dumps(self.run))
        self.assertFalse(self.plan()["work"])
        self.publish()
        with self.assertRaisesRegex(ValueError, "changed"):
            self.plan()
        self.manifest = self.destination / "renewed-verified-reports.json"
        self.manifest_report()
        self.assertEqual(len(self.plan()["work"]), 1)

    def test_same_size_mtime_source_mutation_is_protected(self):
        self.publish()
        self.manifest_report()
        source = self.work / "run.json"
        before = source.stat()
        source.write_bytes(source.read_bytes().replace(b"completed", b"cancelled"))
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        plan = self.plan()
        self.assertFalse(plan["work"])
        self.assertIn("run.json differs", plan["protected"][0]["reason"])

    def test_invalid_report_is_never_a_cleanup_authority(self):
        self.publish()
        report = next(self.final.glob("*.md"))
        report.write_text("changed")
        self.manifest_report()
        self.assertFalse(self.plan()["work"])
        self.assertIn("not verified", self.plan()["protected"][0]["reason"])

    def test_orphan_acceptance_and_active_work_are_protected(self):
        self.publish()
        self.manifest_report()
        (self.work / "ACTIVE").touch()
        (self.root / "acceptance/run01").mkdir(parents=True)
        (self.root / "acceptance/orphan").mkdir()
        plan = retention.build_plan(self.root, self.manifest, False, None)
        self.assertFalse(plan["work"])
        self.assertFalse(plan["acceptance"])
        self.assertEqual(len(plan["protected"]), 3)

    def test_missing_artifact_dependency_closure_protects_acceptance_without_work(self):
        raw = self.work / "raw.npy"
        raw.write_bytes(b"fixture")
        self.run["artifacts"] = [dict(path="raw.npy", kind="raw", bytes=7,
                                     sha256=hashlib.sha256(raw.read_bytes()).hexdigest())]
        self.publish()
        self.manifest_report()
        self.assertFalse(self.plan()["work"])
        self.work.rename(self.root / "held-work")
        (self.root / "acceptance/run01").mkdir(parents=True)
        self.assertFalse(self.plan()["acceptance"])

    def test_changed_source_after_plan_cannot_reach_deletion(self):
        self.publish()
        self.manifest_report()
        args = SimpleNamespace(root=self.root, report_dir=self.destination, manifest=self.manifest,
                               discard_pending=False, run_id="run01")
        with redirect_stdout(io.StringIO()):
            retention.command_plan(args)
        (self.work / "run.json").write_text("{}")
        with patch.object(retention.shutil, "rmtree") as remove:
            with self.assertRaisesRegex(RuntimeError, "changed after planning"):
                retention.command_apply(args)
            remove.assert_not_called()
        self.assertFalse((self.destination / "retention-index.json").exists())

    def test_apply_keeps_one_dated_report_and_never_overwrites(self):
        self.publish()
        self.manifest_report()
        args = SimpleNamespace(root=self.root, report_dir=self.destination, manifest=self.manifest,
                               discard_pending=False, run_id="run01")
        with redirect_stdout(io.StringIO()):
            retention.command_plan(args)
            with patch.object(retention.shutil, "rmtree", side_effect=OSError("test prohibits deletion")) as remove:
                retention.command_apply(args)
                remove.assert_called_once_with(self.work)
        report = next(self.destination.glob("*.md"))
        before = report.read_bytes()
        self.assertRegex(report.name, r"^\d{8}-output-retention\.md$")
        self.assertEqual(len(list(self.destination.glob("*.md"))), 1)
        self.assertTrue(self.work.exists())
        self.assertEqual(retention.verify_retention(self.root, self.destination)["status"], "completed_with_errors")
        with self.assertRaisesRegex(RuntimeError, "receipt already exists"):
            retention.command_apply(args)
        self.assertEqual(report.read_bytes(), before)
        report.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "report changed"):
            retention.verify_retention(self.root, self.destination)

    def test_legacy_cleanup_receipt_still_verifies(self):
        self.publish(format="full")
        self.manifest_report()
        plan = self.plan()
        retention.write_json(self.destination / "plan.json", plan)
        (self.destination / "REPORT.md").write_text("historical report")
        index = dict(schema=retention.INDEX_SCHEMA, root=str(self.root), status="completed",
                     plan_sha256=retention.file_hash(self.destination / "plan.json"),
                     retained_reports=plan["reports"], failure_summaries=[], acceptance_summaries=[],
                     deleted_work=[], deleted_acceptance=[], protected=[], errors=[],
                     reclaimed_bytes=0, reclaimed_allocated_bytes=0)
        retention.write_json(self.destination / "retention-index.json", index)
        self.assertEqual(retention.verify_retention(self.root, self.destination)["report"], str(self.destination / "REPORT.md"))


if __name__ == "__main__":
    unittest.main()
