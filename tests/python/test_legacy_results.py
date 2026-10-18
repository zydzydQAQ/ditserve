"""Historical evidence stays readable; cleanup operates on temporary fixtures."""

import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch
import zipfile

import importlib.util
import unittest

if importlib.util.find_spec("ditserve.results.legacy") is None:
    raise unittest.SkipTest("legacy result readers ship only in a repository checkout")

from ditserve import results
from ditserve.results import compact
from ditserve.runtime.client import CoreBridge, native_environment


class LegacyResultsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)

    def fixture(self, name="run"):
        root = self.base / name
        root.mkdir()
        (root / "output.npy").write_bytes(b"raw fixture")
        (root / "summary.json").write_text('{"status":"failed","latency_ms":1.25}\n')
        return root, root / "excel"

    def snapshot(self, root):
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def command(self, arguments):
        bridge = CoreBridge()
        if not bridge.available():
            self.skipTest("build the native CLI first")
        result = subprocess.run([str(bridge.binary), "--json", *arguments],
            env=native_environment(), capture_output=True, text=True, check=False)
        self.assertTrue(result.stdout.strip(), result.stderr)
        return result.returncode, json.loads(result.stdout)

    def reference(self, root):
        return compact.verify_compacted_reference(root, "output.npy", hashlib.sha256(b"raw fixture").hexdigest())

    def test_archive_gzip_and_index_are_read_only_without_old_report_package(self):
        try:
            results._excel()
        except ImportError:
            self.skipTest("openpyxl optional")
        destination = self.base / "archive"
        shutil.copytree(Path(__file__).resolve().parents[1] / "fixtures/legacy-archive", destination)
        step = destination / "series/step"
        before = self.snapshot(destination)
        self.assertEqual(self.command(["results", "verify", str(destination)])[1]["steps"], 1)
        code, payload = self.command(["results", "inspect", str(step)])
        self.assertEqual(code, 0)
        self.assertEqual(results.load(step), {key: value for key, value in payload.items()
                                             if key not in {"load_ms", "worker_operation_ms"}})
        evidence = json.loads(payload["files"]["summary.json"]["text"])
        self.assertEqual(evidence["runs"]["edf"], dict(status="failed", completed=0,
            parity_failures=1, latency_ms=1.2345678901234567))
        self.assertEqual(evidence["config"], dict(enabled=False, identifier=12345678901234567890,
            literal="=SUM(A1:A2)", missing=None))
        self.assertIs(evidence["config"]["enabled"], False)
        self.assertIsNone(evidence["config"]["missing"])
        self.assertFalse(results.verify(step)["raw_re_evaluated"])
        checked = subprocess.run([sys.executable, "-m", "ditserve", "results", "verify", str(destination)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "workers")),
            capture_output=True, text=True, check=False)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(json.loads(checked.stdout)["status"], "verified")
        self.assertEqual(self.snapshot(destination), before)
        for change, name in (("report", "REPORT.md"), ("payload", "data.json.gz"),
                             ("title", "results.xlsx"), ("boolean", "results.xlsx"),
                             ("integer", "results.xlsx"), ("literal", "results.xlsx")):
            with self.subTest(change=change):
                target = step / name
                saved = target.read_bytes()
                if name == "data.json.gz":
                    data = results.load(step)
                    data["files"]["summary.json"]["text"] = "{}"
                    with gzip.open(target, "wt", encoding="utf-8") as stream:
                        json.dump(data, stream)
                elif name == "results.xlsx":
                    book = results._excel().load_workbook(target)
                    if change == "title":
                        book["Overview"]["B1"] = "tampered"
                    else:
                        parameter, value = {"boolean": ("enabled", 0),
                            "integer": ("identifier", 12345678901234567890),
                            "literal": ("literal", "=SUM(A1:A2)")}[change]
                        row = next(row for row in book["Config"].iter_rows(min_row=2)
                                   if row[1].value == parameter)
                        row[2].value = value
                    book.save(target)
                    book.close()
                else:
                    target.write_text("tampered")
                with self.assertRaises(ValueError):
                    results.verify(step)
                target.write_bytes(saved)

    def test_compact_keeps_complete_gzip_text_and_literal_chunked_excel(self):
        root, destination = self.fixture()
        value = '=HYPERLINK("https://invalid")' + "文😀" * 22000
        (root / "summary.json").write_text(json.dumps({"value": value, "control": "\x00\r"}))
        text = '{"a":1,"b":2}\n{"a":3,"b":4}\n'
        (root / "events.jsonl").write_text(text)
        original = self.snapshot(root)
        compact.export(root, destination, metrics_limit=1)
        before = self.snapshot(destination)
        payload = results.load(destination)
        entries = {row["path"]: row for row in payload["files"]}
        self.assertEqual(entries["summary.json"]["text"].encode(), original["summary.json"])
        self.assertEqual(entries["events.jsonl"]["text"], text)
        self.assertNotIn("text", entries["output.npy"])
        chunks = [row[-1] for row in compact._tables(payload)[2][1][1:] if row[2] == "/value"]
        self.assertEqual("".join(chunks), value)
        self.assertGreater(len(chunks), 1)
        with zipfile.ZipFile(destination / "results.xlsx") as book:
            for name in book.namelist():
                if name.startswith("xl/worksheets/"):
                    self.assertNotIn(b"<f>", book.read(name))
        self.assertEqual(results.verify(destination)["status"], "verified")
        self.assertEqual(self.snapshot(destination), before)

    def test_both_cli_names_leave_hash_bound_completed_cleanup_receipts(self):
        for index, command in enumerate((["compact-results"], ["results", "compact"])):
            root, destination = self.fixture(str(index))
            (root / "run.json").write_text('{"schema_version":"ditserve.benchmark.run.v1","status":"completed"}')
            source = (root / "summary.json").read_bytes()
            code, outcome = self.command(command + [str(root), "--prune-raw"])
            self.assertEqual(code, 0)
            self.assertEqual(outcome["cleanup"]["status"], "pruned")
            self.assertEqual((root / "summary.json").read_bytes(), source)
            evidence = self.reference(root)
            self.assertEqual(evidence["status"], "retained_metrics_only")
            self.assertFalse(evidence["re_evaluable"])
            self.assertIsNone(evidence["path"])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                compact.verify_compacted_reference(root, "output.npy", "0" * 64)
            with self.assertRaisesRegex(ValueError, "pruned evidence"):
                compact.export(root, destination, replace_existing=True)
            receipt = destination / compact.RECEIPT_NAME
            value = json.loads(receipt.read_text())
            value["deleted"]["output.npy"]["bytes"] += 1
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "receipt"):
                self.reference(root)
            receipt.unlink()
            with self.assertRaises(FileNotFoundError):
                self.reference(root)

    def test_missing_raw_requires_receipt_and_unlink_crash_recovers(self):
        root, destination = self.fixture()
        compact.export(root, destination)
        raw = root / "output.npy"
        raw.unlink()
        with self.assertRaises(FileNotFoundError):
            self.reference(root)
        raw.write_bytes(b"raw fixture")
        unlink = Path.unlink
        def crash(path, *args, **kwargs):
            unlink(path, *args, **kwargs)
            if path == raw:
                raise OSError("fixture crash after unlink")
        with patch.object(Path, "unlink", crash), self.assertRaisesRegex(OSError, "fixture crash"):
            compact.prune(destination, ["output.npy"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.reference(root)
        compact.prune(destination, ["output.npy"])
        self.assertFalse(self.reference(root)["raw_available"])

    def test_preflight_changes_block_entire_deletion_batch(self):
        for change in ("array", "metadata", "workbook", "payload", "active_child"):
            with self.subTest(change=change):
                root, destination = self.fixture(change)
                child = root / "child"
                child.mkdir()
                (child / "second.npy").write_bytes(b"other raw")
                (child / "gpu-budget.json").write_text('{"active":null}')
                compact.export(root, destination)
                targets = {"array": child / "second.npy", "metadata": root / "summary.json",
                           "workbook": destination / "results.xlsx", "payload": destination / "data.json.gz",
                           "active_child": child / "gpu-budget.json"}
                targets[change].write_bytes(b'{"active":{"id":"running"}}' if change == "active_child" else b"changed")
                with self.assertRaises(ValueError):
                    compact.prune(destination, ["output.npy", "child/second.npy"])
                self.assertTrue((root / "output.npy").exists())
                self.assertTrue((child / "second.npy").exists())

    def test_locks_active_ancestors_and_links_preserve_working_evidence(self):
        root, destination = self.fixture()
        ledger = root / "gpu-budget.json"
        ledger.write_text('{"active":null}')
        for name in ("gpu-budget.lock", "gpu-budget.json.lock", ".compact-results.lock"):
            with self.subTest(lock=name), (root / name).open("a+") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if name.startswith(".compact"):
                    with self.assertRaises(RuntimeError):
                        compact.export(root, destination)
                else:
                    self.assertEqual(compact.export(root, destination)["status"], "protected")
                self.assertFalse(destination.exists())
        ledger.write_text('{"active":{"id":"running"}}')
        code, outcome = self.command(["results", "compact", str(root), "--prune-raw"])
        self.assertEqual((code, outcome["status"]), (1, "protected"))
        child = root / "child"
        child.mkdir()
        self.assertEqual(compact.export(child, child / "excel")["status"], "protected")
        ledger.write_text('{"active":null}')
        outside = self.base / "outside.npy"
        outside.write_bytes(b"external")
        (root / "linked.npy").symlink_to(outside)
        snapshot = root / "source-snapshots"
        snapshot.mkdir()
        (snapshot / "frozen.npy").write_bytes(b"source")
        compact.export(root, destination)
        names = {row["path"] for row in results.load(destination)["files"]}
        self.assertNotIn("linked.npy", names)
        self.assertNotIn("source-snapshots/frozen.npy", names)

    def test_only_explicit_hash_bound_aggregation_authorizes_log_cleanup(self):
        root, destination = self.fixture()
        log = root / "events.jsonl"
        log.write_text('{"ms":1}\n{"ms":2}\n')
        summary = dict(schema="dit-field-summary-v1", valid=True, documents=2,
            fields=[dict(path="$.ms", numeric_count=2, numeric_mean=1.5)],
            source_metadata=dict(source=str(log), size=log.stat().st_size,
                                 sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        invalid = dict(summary, source_metadata=dict(summary["source_metadata"], sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            compact.export(root, destination, summarized_text={log.name: invalid})
        compact.export(root, destination, max_text_bytes=1, summarized_text={log.name: summary})
        with self.assertRaises(ValueError):
            compact.prune(destination, ["summary.json"])
        compact.prune(destination, [log.name])
        reference = compact.verify_compacted_reference(root, log.name, summary["source_metadata"]["sha256"])
        self.assertEqual(reference["evidence_status"], "aggregated_only")
        self.assertFalse(reference["original_recoverable"])
        self.assertTrue((root / "output.npy").exists())

    def test_report_bridge_preserves_old_run_schema_and_defers_failures(self):
        root, destination = self.fixture()
        config = self.base / "configs"
        config.mkdir()
        (config / "result_storage.json").write_text('{"auto_export_reports":true,"export_directory":"excel"}')
        (root / "run.json").write_text('{"schema_version":"ditserve.benchmark.run.v1","status":"completed"}')
        before = (root / "run.json").read_bytes()
        with patch.object(compact, "CONFIGS", config):
            self.assertEqual(compact.export_report(root)["status"], "verified")
            self.assertEqual(compact.export_report(root)["files"], 3)
            self.assertEqual(compact.verify(destination, check_sources=True)["status"], "verified")
            ledger = self.base / "external-ledger.json"
            ledger.write_text('{"active":{"id":"running"}}')
            self.assertEqual(compact.export_report(root, ledger=ledger)["status"], "protected")
            with patch.object(compact, "export", side_effect=OSError("fixture disk failure")), self.assertWarns(RuntimeWarning):
                self.assertEqual(compact.export_report(root)["status"], "deferred")
        self.assertEqual((root / "run.json").read_bytes(), before)
        self.assertTrue((root / "output.npy").exists())
        (root / "run.json").write_text('{"schema":"ditserve.run.v1"}')
        with patch.object(results, "finalize", return_value={"status": "completed"}) as finalizer:
            self.assertEqual(compact.export_report(root)["status"], "completed")
            finalizer.assert_called_once_with(root, prune=False)


if __name__ == "__main__":
    unittest.main()
