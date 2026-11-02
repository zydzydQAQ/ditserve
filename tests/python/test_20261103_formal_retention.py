"""Formal gate sources remain re-readable after ordinary run finalization."""
from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_20261102_retention as retention_fixture

retention = retention_fixture.retention


class FormalRetentionTests(unittest.TestCase):
    setUp = retention_fixture.RetentionTests.setUp
    publish = retention_fixture.RetentionTests.publish
    manifest_report = retention_fixture.RetentionTests.manifest_report
    plan = retention_fixture.RetentionTests.plan

    def acceptance(self, name, data):
        source = self.root / "acceptance/run01" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(data), encoding="utf-8")
        return source

    def test_completed_formal_run_with_empty_or_claimed_closed_consumers_is_protected(self):
        self.run["config"] = {"benchmark": {"formal_protocol": {
            "schema": "ditserve.formal-protocol.v1", "role": "calibration"}}}
        for dependencies in ([], [{"name": "stage_gates", "closed": True, "reason": "job completed"}]):
            with self.subTest(dependencies=dependencies):
                self.run["retention"] = {"pending_dependencies": dependencies}
                self.publish()
                if self.manifest.exists():
                    self.manifest.unlink()
                self.manifest_report()
                plan = self.plan()
                self.assertFalse(plan["work"])
                self.assertIn("live verification dependency", plan["protected"][0]["reason"])

    def test_nested_contract_documents_protect_acceptance_even_without_raw_work(self):
        self.publish()
        self.manifest_report()
        self.work.rename(self.root / "held-work")
        for name, data in (
            ("nested/contract-evidence.json", {}),
            ("nested/contract-receipt.json", {}),
            ("nested/unusual-name.json", {"schema": "ditserve.stage-evidence.v1"}),
            ("nested/gates.json", {"schema": "ditserve.stage-gates.v1"}),
            ("nested/runtime-smoke.json", {"schema": "ditserve.runtime-smoke.v1", "image_digest": "pending"}),
        ):
            with self.subTest(name=name):
                source = self.acceptance(name, data)
                plan = self.plan()
                self.assertFalse(plan["acceptance"])
                self.assertIn("live verification dependency", plan["protected"][0]["reason"])
                source.unlink()

    def test_generated_configs_profile_bundles_and_nested_runs_are_protected(self):
        self.publish()
        self.manifest_report()
        for data in (
            {"benchmark": {"formal_protocol": {}}},
            {"schema": "ditserve.shape-profiles.v1"},
            {"schema": "ditserve.run.v1", "config": {"benchmark": {"formal_protocol": None}}},
        ):
            with self.subTest(data=data):
                self.acceptance("child/source.json", data)
                plan = self.plan()
                self.assertFalse(plan["acceptance"])
                self.assertIn("live verification dependency", plan["protected"][0]["reason"])

    def test_truncated_contract_still_protects_parent(self):
        self.publish()
        self.manifest_report()
        self.acceptance("child/contract-evidence.json", {}).write_text('{"schema":')
        plan = self.plan()
        self.assertFalse(plan["acceptance"])
        self.assertIn("live verification dependency", plan["protected"][0]["reason"])

    def test_old_plan_cannot_delete_work_or_acceptance_after_guard_upgrade(self):
        self.run["config"] = {"benchmark": {"formal_protocol": {"role": "preemption"}}}
        self.publish()
        self.manifest_report()
        self.acceptance("child/contract-receipt.json", {"schema": "ditserve.stage-gates.v1"})
        args = SimpleNamespace(root=self.root, report_dir=self.destination, manifest=self.manifest,
                               discard_pending=False, run_id="run01")
        # Reproduce a plan from the old cleaner, which trusted empty pending lists.
        with patch.object(retention, "protected_formal_source", return_value=None), redirect_stdout(io.StringIO()):
            retention.command_plan(args)
        previous = json.loads((self.destination / "plan.json").read_text())
        self.assertEqual(len(previous["work"]), 1)
        self.assertEqual(len(previous["acceptance"]), 1)
        with patch.object(retention.shutil, "rmtree") as remove:
            with self.assertRaisesRegex(RuntimeError, "changed after planning"):
                retention.command_apply(args)
            remove.assert_not_called()
        self.assertTrue((self.work / "run.json").exists())
        self.assertTrue((self.root / "acceptance/run01/child/contract-receipt.json").exists())
        self.assertFalse((self.destination / "retention-index.json").exists())


if __name__ == "__main__":
    unittest.main()
