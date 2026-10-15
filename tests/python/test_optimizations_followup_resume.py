"""CPU-only checks for immutable-source continuation and honest reporting."""

import json
from pathlib import Path
import tempfile
import unittest

from ditserve.optimizations import followup_resume as resume
from ditserve.research.common import atomic_json, digest, file_digest


class FrozenContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "snapshot"
        self.source.mkdir()
        self.work = self.base / "work"
        self.work.mkdir()
        self.config = {"model": "same-model", "steps": 32, "seed": 101}
        for relative, body in {
            "python/ditserve/example.py": b"ORIGINAL = 1\n",
            "configs/example.json": b'{"scientific_recipe": "frozen"}\n',
        }.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        self.source_files = {
            str(path.relative_to(self.source)): file_digest(path)
            for path in sorted(self.source.rglob("*")) if path.is_file()
        }
        self.manifest = self.source / "manifest.json"
        self.manifest_value = dict(execution_root=str(self.source),
            config_sha256=digest(self.config), source_files=self.source_files,
            source_sha256=digest(self.source_files), phase="finish_copy", attempt="attempt000")
        atomic_json(self.manifest, self.manifest_value)
        self.controller = self.base / "resume_controller.py"
        self.controller.write_text("# fixture controller\n")
        atomic_json(self.work / "protocol.json", dict(config=self.config, sha256=digest(self.config)))
        self.factory = resume.SnapshotFreezer(resume.binding(self.manifest), resume.binding(self.controller))

    def derive(self, **changes):
        arguments = dict(config=self.config, run_root=self.work, phase="sana_combo", attempt="attempt000")
        arguments.update(changes)
        return self.factory(**arguments)

    def replace_manifest(self, **changes):
        self.manifest_value.update(changes)
        atomic_json(self.manifest, self.manifest_value)
        self.factory = resume.SnapshotFreezer(resume.binding(self.manifest), resume.binding(self.controller))

    def test_derivation_uses_only_frozen_bytes_and_keeps_source_immutable(self):
        mutable_workspace = self.base / "workspace/python/ditserve"
        mutable_workspace.mkdir(parents=True)
        (mutable_workspace / "example.py").write_text("MUTABLE = 'must not enter worker'\n")
        before = {str(path): path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        result = self.derive()
        target = Path(result["execution_root"])
        self.assertEqual(target, self.work / "sources/sana_combo-attempt000")
        self.assertEqual(result["source_files"], self.source_files)
        self.assertEqual(result["source_sha256"], digest(self.source_files))
        self.assertEqual(result["derived_from_manifest"], resume.binding(self.manifest))
        self.assertEqual(result["continuation_controller"], resume.binding(self.controller))
        for relative, sha in self.source_files.items():
            self.assertEqual(file_digest(target / relative), sha)
            self.assertEqual((target / relative).read_bytes(), (self.source / relative).read_bytes())
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.source.rglob("*") if path.is_file()})
        self.assertNotIn("MUTABLE", (target / "python/ditserve/example.py").read_text())

    def test_changed_manifest_binding_fails_before_creating_target(self):
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaises(ValueError):
            self.derive()
        self.assertFalse((self.work / "sources").exists())

    def test_changed_controller_binding_fails(self):
        self.controller.write_text("# new executable\n")
        with self.assertRaises(ValueError):
            self.derive()

    def test_changed_frozen_source_fails(self):
        (self.source / "python/ditserve/example.py").write_text("CORRUPT = 1\n")
        with self.assertRaises(ValueError):
            self.derive()

    def test_changed_manifest_source_digest_fails(self):
        self.replace_manifest(source_sha256="0" * 64)
        with self.assertRaises(ValueError):
            self.derive()

    def test_configuration_and_stage_are_not_editable(self):
        for change in ({"config": dict(self.config, steps=16)}, {"phase": "finish_copy"},
                       {"phase": "../outside"}, {"attempt": "attempt001"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.derive(**change)
        self.assertFalse((self.work / "sources").exists())

    def test_protocol_content_and_hash_must_both_match(self):
        for value in (dict(config=dict(self.config, seed=102), sha256=digest(self.config)),
                      dict(config=self.config, sha256="0" * 64)):
            atomic_json(self.work / "protocol.json", value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.derive()

    def test_existing_destination_is_never_overwritten(self):
        target = self.work / "sources/sana_combo-attempt000"
        target.mkdir(parents=True)
        sentinel = target / "keep.txt"
        sentinel.write_text("retain existing evidence")
        with self.assertRaises(FileExistsError):
            self.derive()
        self.assertEqual(sentinel.read_text(), "retain existing evidence")

    def test_manifest_relative_traversal_is_rejected(self):
        outside = self.base / "outside.py"
        outside.write_text("outside source")
        source_files = {"../outside.py": file_digest(outside)}
        self.replace_manifest(source_files=source_files, source_sha256=digest(source_files))
        with self.assertRaises(ValueError):
            self.derive()

    def test_manifest_absolute_source_is_rejected(self):
        outside = self.base / "outside.py"
        outside.write_text("outside source")
        source_files = {str(outside): file_digest(outside)}
        self.replace_manifest(source_files=source_files, source_sha256=digest(source_files))
        with self.assertRaises(ValueError):
            self.derive()

    def test_source_file_symlink_is_rejected_even_with_same_bytes(self):
        original = self.source / "python/ditserve/example.py"
        outside = self.base / "same_bytes.py"
        outside.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.derive()

    def test_source_root_symlink_is_rejected(self):
        alias = self.base / "snapshot-alias"
        alias.symlink_to(self.source, target_is_directory=True)
        self.replace_manifest(execution_root=str(alias))
        with self.assertRaises(ValueError):
            self.derive()

    def test_destination_parent_symlink_is_rejected_without_external_write(self):
        outside = self.base / "outside-destination"
        outside.mkdir()
        (self.work / "sources").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.derive()
        self.assertEqual(list(outside.iterdir()), [])

    def test_work_root_symlink_is_rejected(self):
        alias = self.base / "work-alias"
        alias.symlink_to(self.work, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.derive(run_root=alias)

    def test_frozen_factory_restores_supervisor_on_exception(self):
        original = resume.supervisor.freeze
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with resume.frozen_source_factory(self.factory):
                self.assertIs(resume.supervisor.freeze, self.factory)
                raise RuntimeError("interrupted")
        self.assertIs(resume.supervisor.freeze, original)


class ContinuationReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.work, self.output = self.base / "work", self.base / "output"
        self.work.mkdir()
        self.ledger = self.work / "ledger.json"
        atomic_json(self.ledger, dict(active=None, sessions=[], used_gpu_seconds=0., budget_unlimited=True))
        atomic_json(self.work / "config.json", dict(seed=101, fixed_arrivals=True))
        atomic_json(self.work / "study.json", dict(status="running", ledger=str(self.ledger), dependencies_closed=False))
        self.queue = self.work / "queue.json"
        atomic_json(self.queue, dict(status="failed", reason="ValueError: queued executable changed: example.py"))
        self.receipt = self.work / "resume01.json"
        atomic_json(self.receipt, dict(schema_version="ditserve.followup_resume.v1", status="running",
            original_queue=resume.binding(self.queue), current_phase="sana_combo",
            remaining=["sana_combo", "granularity"], completed_phases=[]))

    def write(self):
        return resume.write_report(self.work, self.output, self.ledger, self.receipt)

    def test_report_shows_resume_and_preserves_failed_queue_in_verified_excel(self):
        before = {str(path): path.read_bytes() for path in self.work.rglob("*") if path.is_file()}
        result = self.write()
        self.assertEqual(result["status"], "resume01:running")
        self.assertEqual(result["excel"]["status"], "verified")
        self.assertFalse(result["original_queue_modified"])
        self.assertEqual(result["originals_deleted"], 0)
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.work.rglob("*") if path.is_file()})
        self.assertEqual({path.name for path in self.output.iterdir()}, {"results.xlsx", "REPORT.md"})
        text = (self.output / "REPORT.md").read_text()
        self.assertIn("resume01:running", text)
        self.assertIn("失败记录保持原样", text)
        from openpyxl import load_workbook
        workbook = load_workbook(self.output / "results.xlsx", read_only=True, data_only=False)
        try:
            rows = list(workbook["Config"].values)
            self.assertTrue(any(row[2] == "queue" and row[3] == "/status" and row[4] == "failed" for row in rows))
            self.assertTrue(any(row[3] == "/status" and row[4] == "running" and row[7] == str(self.receipt) for row in rows))
            self.assertTrue(any(row[0] == str(self.receipt) and row[1] == file_digest(self.receipt)
                for row in workbook["Sources"].values))
        finally:
            workbook.close()

    def test_changed_failed_queue_prevents_new_report(self):
        self.queue.write_text(self.queue.read_text() + "\n")
        with self.assertRaises(ValueError):
            self.write()
        self.assertFalse(self.output.exists())

    def test_unknown_continuation_schema_fails(self):
        value = json.loads(self.receipt.read_text())
        value["schema_version"] = "unknown"
        atomic_json(self.receipt, value)
        with self.assertRaises(ValueError):
            self.write()

    def test_unrelated_report_is_not_overwritten(self):
        self.output.mkdir()
        report = self.output / "REPORT.md"
        report.write_text("user-owned unrelated document")
        with self.assertRaises(ValueError):
            self.write()
        self.assertEqual(report.read_text(), "user-owned unrelated document")


if __name__ == "__main__":
    unittest.main()
