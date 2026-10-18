"""Small CPU contracts for precise archives and recoverable cleanup."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve import results


LEGACY_READERS = importlib.util.find_spec("ditserve.results.legacy") is not None

class ResultsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.work = self.base / "work"
        self.work.mkdir()
        self.output = self.base / "published"
        self.run = dict(schema="ditserve.run.v1", experiment="cpu", run_id="run01", status="completed",
            config={}, phases=[dict(id="pair", status="completed", attempt=0, qualification="unverified")],
            metrics={}, requests=[], failures=[], budget=dict(active=None, charges=[]), artifacts=[])

    def write(self):
        (self.work / "run.json").write_text(json.dumps(self.run, ensure_ascii=True), encoding="utf-8")

    def artifact(self, name="sample.npy", dependencies=None, kind="raw"):
        path = self.work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"indispensable raw evidence")
        row = dict(path=name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   kind=kind, dependencies=dependencies or [])
        self.run["artifacts"].append(row)
        return path

    def final(self, **options):
        self.write()
        return results.finalize(self.work, self.output, **options)

    def test_optional_report_tools_do_not_shadow_the_model_environment(self):
        cached = self.base / "report-tools"
        (cached / "openpyxl").mkdir(parents=True)
        (cached / "openpyxl/__init__.py").write_text("# fixture")
        expected = object()
        with patch.dict(results.os.environ, {"DITSERVE_REPORT_TOOLS": str(cached)}), \
                patch.object(results.sys, "path", ["existing-model-environment"]), \
                patch.object(results.importlib, "import_module", side_effect=[ModuleNotFoundError("openpyxl"), expected]):
            self.assertIs(results._excel(), expected)
            self.assertEqual(results.sys.path, ["existing-model-environment", str(cached)])

    def test_roundtrip_precision_types_escaping_and_split_sheets(self):
        self.run["metrics"] = dict(identifier=123456789012345678901234567890, precise=1.2345678901234567,
            negative_zero=-0.0, missing=None, yes=True, numeric_dict={"01": [], "1": {}},
            literal="=SUM(A1:A3)\u0000\r\n\ud800", long="x\U0001f642" * 24000,
            sequence=list(range(15)))
        self.run["config"] = {"/~": "@formula", "": ""}
        with patch.object(results, "MAX_SHEET_ROWS", 12):
            value = self.final()
        self.assertEqual(value["status"], "completed", value)
        restored = results.load(self.output)
        self.assertEqual(results._json(restored["run"]), results._json(self.run))
        self.assertEqual({p.name for p in self.output.iterdir()}, {"results.xlsx", "REPORT.md"})
        book = results._excel().load_workbook(self.output / "results.xlsx", read_only=True)
        try:
            self.assertTrue(any(name.startswith("metrics_") for name in book.sheetnames))
            for sheet in book:
                self.assertLessEqual(sheet.max_row or len(list(sheet.values)), 12)
                self.assertFalse(any(cell.data_type == "f" for row in sheet for cell in row))
        finally:
            book.close()
        self.assertEqual(results.verify(self.output)["status"], "verified")

    def test_verified_cleanup_and_idempotent_finalization(self):
        raw = self.artifact()
        value = self.final()
        self.assertEqual(value["status"], "completed", value)
        self.assertFalse(raw.exists())
        self.assertEqual(results.load(self.output)["run"], self.run)
        self.assertEqual(results.load(self.output)["finalization"]["deleted"][0]["status"], "deleted")
        before = (self.output / "results.xlsx").read_bytes()
        self.assertEqual(results.finalize(self.work, self.output)["status"], "completed")
        self.assertEqual((self.output / "results.xlsx").read_bytes(), before)
        self.run["metrics"]["new"] = 1
        self.write()
        self.assertEqual(results.finalize(self.work, self.output)["status"], "deferred")
        self.assertEqual((self.output / "results.xlsx").read_bytes(), before)

    def test_report_distinguishes_completion_from_scoped_qualification_and_errors(self):
        self.run["phases"][0]["qualification"] = dict(qualified=False, scopes={
            "numeric": {"status": "passed"}, "performance": {"status": "failed"}},
            checks=[dict(scope="performance", metric="/speedup", status="failed", actual=1.05, op="ge", expected=1.2)])
        self.run["phases"].append(dict(id="decode", status="failed", attempt=1, error="OOM|device\nallocation failed"))
        self.run["failures"] = [dict(phase="decode", attempt=1, error="OOM|device\nallocation failed")]
        reason = "invalid output\n" + "detail " * 80 + "FULL_ERROR_TAIL"
        self.run["requests"] = [dict(request_id="r", status="failed", error=reason)]
        self.assertEqual(self.final()["status"], "completed")
        report = (self.output / "REPORT.md").read_text()
        self.assertIn("completed 仅表示执行完成", report)
        self.assertIn("| pair | completed | 0 | passed | unverified | failed |", report)
        self.assertIn("| decode | failed | 1 | unverified | unverified | unverified |", report)
        self.assertIn("performance 门槛 /speedup：failed；实际 1.05，要求 ge 1.2", report)
        self.assertEqual(report.count("OOM\\|device allocation failed"), 1)
        self.assertIn("每项最多 400 字", report)
        self.assertNotIn("FULL_ERROR_TAIL", report)
        restored = results.load(self.output)
        self.assertEqual(restored["run"], self.run)
        self.assertEqual(restored["finalization"]["report_version"], 2)

    def test_previous_report_template_stays_verifiable_and_immutable(self):
        self.assertEqual(self.final()["status"], "completed")
        previous = results.load(self.output)
        previous["finalization"].pop("report_version")
        results._write_book(self.output / "results.xlsx", previous)
        report = results._report(previous)
        self.assertIn("| 阶段 | 状态 | 尝试 |", report)
        self.assertNotIn("数值 numeric", report)
        (self.output / "REPORT.md").write_text(report)
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        self.assertEqual(results.verify(self.output)["status"], "verified")
        self.assertEqual(results.finalize(self.work, self.output)["status"], "completed")
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, before)

    def test_active_budget_and_active_external_ledger_protect_inputs(self):
        raw = self.artifact()
        self.run["budget"]["active"] = {"session_id": "live"}
        self.assertEqual(self.final()["status"], "protected")
        self.run["budget"]["active"] = None
        ledger = self.base / "ledger.json"
        ledger.write_text(json.dumps({"active": {"session_id": "live"}}))
        self.run["budget_ledger"] = str(ledger)
        self.assertEqual(self.final()["status"], "protected")
        self.assertTrue(raw.exists())
        self.assertFalse(self.output.exists())

    def test_open_dependencies_export_and_retry_after_explicit_closure(self):
        raw = self.artifact(dependencies=["blind-review"])
        value = self.final()
        self.assertEqual(value["status"], "deferred", value)
        self.assertTrue(raw.exists())
        self.assertIn("blind-review", (self.output / "REPORT.md").read_text())
        self.run["artifacts"][0]["dependencies"] = [{"closed": True, "reason": "review completed"}]
        self.assertEqual(self.final()["status"], "completed")
        self.assertFalse(raw.exists())

    def test_partial_cleanup_allows_dependency_closure_without_changing_metrics(self):
        first = self.artifact("closed.npy")
        later = self.artifact("pending.npy", dependencies=["evaluation"])
        self.assertEqual(self.final()["status"], "deferred")
        self.assertFalse(first.exists())
        self.assertTrue(later.exists())
        self.run["artifacts"][1]["dependencies"] = []
        self.assertEqual(self.final()["status"], "completed")
        self.assertFalse(later.exists())
        saved = results.load(self.output)["finalization"]
        self.assertEqual(len(saved["deleted"]), 2)
        self.assertEqual(saved["input_revisions"][0]["dependency_closures"][0]["previous"], ["evaluation"])

    def test_control_files_survive_successful_finalization(self):
        source = self.artifact("sources/model.py", kind="source")
        config = self.artifact("config.json", kind="raw")
        self.assertEqual(self.final()["status"], "completed")
        self.assertTrue(source.exists())
        self.assertTrue(config.exists())

    def test_unlink_failure_is_recorded_and_retryable(self):
        raw = self.artifact()
        unlink = Path.unlink
        def fail(path, *args, **kwargs):
            if path == raw:
                raise PermissionError("test deletion failure")
            return unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", fail):
            failed = self.final()
        self.assertEqual(failed["status"], "deferred")
        self.assertTrue(raw.exists())
        self.assertIn("test deletion failure", results.load(self.output)["finalization"]["error"])
        self.assertIn("test deletion failure", (self.output / "REPORT.md").read_text())
        self.assertEqual(results.finalize(self.work, self.output)["status"], "completed")
        self.assertFalse(raw.exists())

    def test_recovery_after_unlink_before_receipt_write(self):
        raw = self.artifact()
        self.write()
        original_save = results._save
        def crash(work, journal):
            if journal.get("deleted"):
                raise KeyboardInterrupt("crash after unlink")
            return original_save(work, journal)
        with patch.object(results, "_save", crash), self.assertRaises(KeyboardInterrupt):
            results.finalize(self.work, self.output)
        self.assertFalse(raw.exists())
        self.assertIn("pending_delete", json.loads((self.work / results.JOURNAL).read_text()))
        self.assertEqual(results.finalize(self.work, self.output)["status"], "completed")
        receipt = results.load(self.output)["finalization"]["deleted"][0]
        self.assertEqual(receipt["status"], "absent_after_recorded_delete_intent")

    @unittest.skipUnless(LEGACY_READERS, "legacy readers ship only in a repository checkout")
    def test_hash_mismatch_escape_symlinks_and_legacy_destination_never_prune(self):
        for mistake in ("hash", "escape", "symlink", "legacy"):
            with self.subTest(mistake=mistake):
                # Each case uses an independent journal/destination.
                work = self.base / mistake
                work.mkdir()
                output = self.base / (mistake + "-out")
                raw = work / "raw.npy"
                raw.write_bytes(b"raw")
                row = dict(path="raw.npy", bytes=3, sha256=hashlib.sha256(b"raw").hexdigest(), kind="raw", dependencies=[])
                data = copy.deepcopy(self.run)
                data["artifacts"] = [row]
                if mistake == "hash":
                    row["sha256"] = "0" * 64
                elif mistake == "escape":
                    row["path"] = "../elsewhere.npy"
                elif mistake == "symlink":
                    (work / "link.npy").symlink_to(raw)
                    row["path"] = "link.npy"
                else:
                    output.mkdir()
                    (output / "results.xlsx").write_bytes(b"immutable historical archive")
                    (output / "pruned.json").write_text("historical receipt")
                (work / "run.json").write_text(json.dumps(data))
                value = results.finalize(work, output)
                self.assertEqual(value["status"], "deferred", value)
                self.assertTrue(raw.exists())
                if mistake == "legacy":
                    self.assertEqual((output / "results.xlsx").read_bytes(), b"immutable historical archive")

    def test_export_failure_or_cell_limit_never_deletes(self):
        raw = self.artifact()
        with patch.object(results, "_write_book", side_effect=OSError("disk full")):
            self.assertEqual(self.final()["status"], "deferred")
        self.assertTrue(raw.exists())
        self.run["config"] = {"x" * 40000: 1}
        self.assertEqual(self.final()["status"], "deferred")
        self.assertTrue(raw.exists())

    def test_readback_tampering_fails(self):
        self.assertEqual(self.final()["status"], "completed")
        (self.output / "REPORT.md").write_text("altered report")
        with self.assertRaisesRegex(ValueError, "report differs"):
            results.verify(self.output)
        before = (self.output / "results.xlsx").read_bytes()
        self.assertEqual(results.finalize(self.work, self.output)["status"], "deferred")
        self.assertEqual((self.output / "REPORT.md").read_text(), "altered report")
        self.assertEqual((self.output / "results.xlsx").read_bytes(), before)

    def test_log_requires_bound_auditable_summary_and_keep_raw_can_later_prune(self):
        raw = self.artifact("events.jsonl", kind="log")
        self.assertEqual(self.final()["status"], "deferred")
        self.assertTrue(raw.exists())
        artifact = self.run["artifacts"][0]
        artifact["summary"] = dict(source_sha256=artifact["sha256"], source_bytes=artifact["bytes"],
            total_records=2, aggregation="exact event count", omitted_fields=["event_body"], metrics={"count": 2})
        self.assertEqual(self.final(prune=False)["status"], "completed")
        self.assertTrue(raw.exists())
        self.assertEqual(results.finalize(self.work, self.output)["status"], "completed")
        self.assertFalse(raw.exists())

    @unittest.skipUnless(LEGACY_READERS, "legacy readers ship only in a repository checkout")
    def test_compact_archive_is_read_only(self):
        from ditserve.results import compact as compact_results
        (self.work / "summary.json").write_text(json.dumps({"loss": 1.25, "status": "failed"}))
        compact_results.export(self.work, self.output)
        before = {p.name: p.read_bytes() for p in self.output.iterdir() if p.is_file()}
        self.assertEqual(results.verify(self.output)["status"], "verified")
        self.assertEqual(results.load(self.output)["schema"], compact_results.SCHEMA)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir() if p.is_file()})

    @unittest.skipUnless(LEGACY_READERS, "legacy readers ship only in a repository checkout")
    def test_legacy_sessions_two_file_read_preserves_typed_values(self):
        self.output.mkdir()
        book = results._excel().Workbook()
        book.remove(book.active)
        def sheet(name, rows):
            target = book.create_sheet(name)
            target.append(["path", "type", "part", "value"])
            for row in rows:
                target.append(row)
                for cell in target[target.max_row]:
                    if isinstance(cell.value, str):
                        cell.data_type = "s"
        sheet("summary", [["", "dict", 0, ""], ["/schema_version", "str", 0, "ditserve.sessions.report.v1"],
                          ["/id", "int", 0, "12345678901234567890"], ["/valid", "bool", 0, False]])
        for name in ("config", "budget", "retention"):
            sheet(name, [["", "empty_dict", 0, ""]])
        sheet("provenance", [["", "empty_list", 0, ""]])
        sheet("evidence", [])
        book.save(self.output / "results.xlsx")
        (self.output / "REPORT.md").write_text("Original report; not rewritten.")
        before = (self.output / "results.xlsx").read_bytes()
        read = results.load(self.output)
        self.assertEqual(read["summary"]["id"], 12345678901234567890)
        self.assertIs(read["summary"]["valid"], False)
        self.assertEqual(results.verify(self.output)["status"], "verified")
        self.assertEqual((self.output / "results.xlsx").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
