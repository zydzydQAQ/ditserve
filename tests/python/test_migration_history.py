import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpyxl import load_workbook

from ditserve.migration import history


class HistoryExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "source"
        self.root.mkdir()
        self.destination = Path(temporary.name) / "export"

    def write(self, path, data):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data if isinstance(data, bytes) else data.encode())
        return target

    def test_exact_archives_metadata_sources_and_formula_safe_provenance(self):
        selected = {
            "results/done/run/REPORT.md": "historical incomplete evidence",
            "results/done/run/results.xlsx": b"old-workbook-bytes",
            "results/done/run/excel/data.json.gz": b"original-complete-archive",
            "results/done/run/excel/pruned.json": '{"deleted":["old.npy"]}',
            "results/done/run/.compact-results.json": '{"root":"/original/absolute/path"}',
            "results/done/run/sources/python/a.py": "# original source\n",
            "results/done/run/sources-final/python/a.py": "# original source\n",
            "results/=literal/run/config.json": '{"integer":12345678901234567890}',
        }
        for path, data in selected.items():
            self.write(path, data)
        raw = self.write("results/done/run/not-evaluated.npy", b"raw-input")
        self.write("results/done/run/worker.log", "log")
        result = history.export_history(self.root, self.destination)
        self.assertFalse(result["resumable"])
        self.assertEqual(result["source_deleted_files"], 0)
        self.assertEqual(result["counts"]["copied_files"], len(selected))
        self.assertEqual(raw.read_bytes(), b"raw-input")
        self.assertFalse((self.destination / raw.relative_to(self.root)).exists())
        for path in selected:
            self.assertEqual((self.root / path).read_bytes(), (self.destination / path).read_bytes())
            self.assertNotEqual((self.root / path).stat().st_ino, (self.destination / path).stat().st_ino)
        first = self.destination / "results/done/run/sources/python/a.py"
        second = self.destination / "results/done/run/sources-final/python/a.py"
        self.assertEqual(first.stat().st_ino, second.stat().st_ino)
        workbook = load_workbook(self.destination / "results.xlsx", read_only=True, data_only=False)
        try:
            rows = list(workbook["provenance"].values)
            self.assertEqual(len(rows), len(selected) + 1)
            row = next(row for row in rows if row[0] == "results/=literal/run/config.json")
            self.assertEqual(row[2], hashlib.sha256((self.root / row[0]).read_bytes()).hexdigest())
            self.assertTrue(all(cell.data_type != "f" for sheet in workbook for row in sheet for cell in row))
        finally:
            workbook.close()

    def test_active_cache_group_protects_result_reports_without_rewriting_source(self):
        report = self.write("results/active/run/REPORT.md", "still running")
        ledger = self.write(".cache/results-work/active/gpu-budget.json", '{"active":{"pid":42}}')
        self.write(".cache/results-work/active/run/queue.json", '{"status":"running"}')
        self.write("results/done/REPORT.md", "complete")
        result = history.export_history(self.root, self.destination)
        self.assertIn("active", result["protected_groups"])
        self.assertFalse((self.destination / report.relative_to(self.root)).exists())
        self.assertTrue((self.destination / "results/done/REPORT.md").is_file())
        self.assertEqual(ledger.read_text(), '{"active":{"pid":42}}')

    def test_pending_queue_and_unreadable_ledger_fail_closed(self):
        self.write("results/pending/run/queue.json", '{"status":"waiting"}')
        self.write("results/pending/run/REPORT.md", "pending")
        self.write("results/broken/gpu-budget.json", "not json")
        self.write("results/broken/run/REPORT.md", "unknown")
        result = history.export_history(self.root, self.destination)
        self.assertEqual(set(result["protected_groups"]), {"pending", "broken"})
        self.assertEqual(result["counts"]["copied_files"], 0)

    def test_symlinks_never_traversed_or_copied(self):
        outside = self.write("outside/private.json", '{"private":true}')
        target = self.root / "results/run"
        target.mkdir(parents=True)
        (target / "linked.json").symlink_to(outside)
        (target / "linked-directory").symlink_to(outside.parent, target_is_directory=True)
        result = history.export_history(self.root, self.destination)
        self.assertEqual(result["counts"]["copied_files"], 0)
        self.assertEqual(len(result["pending_paths"]), 2)
        self.assertFalse(any(p.is_symlink() for p in self.destination.rglob("*")))

    def test_source_changed_after_copy_is_skipped(self):
        source = self.write("results/run/config.json", '{"state":1}')
        original = history._read_file
        def changing(path, **kwargs):
            result = original(path, **kwargs)
            if Path(path) == source and kwargs.get("target") is not None:
                source.write_text('{"state":2}')
            return result
        with patch.object(history, "_read_file", side_effect=changing):
            result = history.export_history(self.root, self.destination)
        self.assertEqual(result["counts"]["copied_files"], 0)
        self.assertIn("unstable", result["pending_paths"][0]["reason"])
        self.assertFalse((self.destination / "results/run/config.json").exists())

    def test_linked_cache_group_protects_matching_results_without_following(self):
        self.write("results/unknown/REPORT.md", "activity unknown")
        work = self.root / ".cache/results-work"
        work.mkdir(parents=True)
        outside = self.root / "external"
        outside.mkdir()
        (work / "unknown").symlink_to(outside, target_is_directory=True)
        result = history.export_history(self.root, self.destination)
        self.assertIn("unknown", result["protected_groups"])
        self.assertEqual(result["counts"]["copied_files"], 0)

    def test_ledger_becoming_active_discards_staged_group_only(self):
        ledger = self.write("results/run/gpu-budget.json", '{"active":null}')
        self.write("results/run/REPORT.md", "copy this before ledger changes")
        original = history._protection
        calls = []
        def changing(project, rows):
            calls.append(1)
            if len(calls) == 2:
                ledger.write_text('{"active":{"pid":42}}')
            return original(project, rows)
        with patch.object(history, "_protection", side_effect=changing):
            result = history.export_history(self.root, self.destination)
        self.assertEqual(result["counts"]["copied_files"], 0)
        self.assertIn("run", result["protected_groups"])
        self.assertFalse((self.destination / "results/run/REPORT.md").exists())

    def test_reject_existing_or_source_overlapping_destination(self):
        self.write("results/run/REPORT.md", "report")
        for destination in (self.root, self.root / "results/export", self.root / ".cache/results-work/export"):
            with self.assertRaises(ValueError):
                history.export_history(self.root, destination)
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            history.export_history(self.root, self.destination)


if __name__ == "__main__":
    unittest.main()
