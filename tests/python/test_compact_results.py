"""Evidence preservation and deletion-gate tests for compact Excel archives."""

import gzip
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from ditserve import compact_results as compact


class CompactResultsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "results"
        self.root.mkdir()
        self.destination = self.root / "excel"
        self.raw = self.root / "output.npy"
        self.raw.write_bytes(b"binary-array-content")
        self.digest = hashlib.sha256(self.raw.read_bytes()).hexdigest()
        (self.root / "summary.json").write_text('{"status":"passed","latency_ms":1.25}\n', encoding="utf-8")

    def export(self, **kwargs):
        return compact.export(self.root, self.destination, **kwargs)

    def payload(self):
        with gzip.open(self.destination / "data.json.gz", "rt", encoding="utf-8") as stream:
            return json.load(stream)

    def summary_for(self, path, **overrides):
        value = dict(schema="dit-field-summary-v1", valid=True, documents=2,
                     fields=[{"path": "$.latency_ms", "numeric_count": 2, "numeric_mean": 1.5}],
                     source_metadata=dict(source=str(path), size=path.stat().st_size,
                                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        value.update(overrides)
        return value

    def test_roundtrip_keeps_exact_metadata_and_indexes_raw(self):
        original = (self.root / "summary.json").read_bytes()
        result = self.export()
        self.assertEqual(result["status"], "verified")
        compact.verify(self.destination, check_sources=True)
        entries = {item["path"]: item for item in self.payload()["files"]}
        self.assertEqual(entries["summary.json"]["text"].encode("utf-8"), original)
        self.assertEqual(entries["output.npy"]["sha256"], self.digest)
        self.assertNotIn("text", entries["output.npy"])
        self.assertTrue(self.raw.exists())
        self.assertIn("metrics-only", self.payload()["notice"])

    def test_formula_injection_and_long_cells_are_literal_and_complete(self):
        value = "=HYPERLINK(\"https://invalid\")" + "文😀" * 22000
        (self.root / "summary.json").write_text(json.dumps({"value": value, "control": "\x00\r"}), encoding="utf-8")
        self.export()
        with zipfile.ZipFile(self.destination / "results.xlsx") as archive:
            xml = archive.read("xl/worksheets/sheet3.xml")
            self.assertNotIn(b"<f>", xml)
            self.assertIn(b"HYPERLINK", xml)
        rows = compact._tables(self.payload())[2][1][1:]
        chunks = [row[-1] for row in rows if row[2] == "/value"]
        self.assertEqual("".join(chunks), value)
        self.assertGreater(len(chunks), 1)
        compact.verify(self.destination)

    def test_large_json_is_hashed_without_embedding_or_deletion(self):
        big = self.root / "large.json"
        big.write_bytes(b" " * 256)
        self.export(max_text_bytes=100)
        entry = next(item for item in self.payload()["files"] if item["path"] == "large.json")
        self.assertEqual(entry["kind"], "retained")
        self.assertNotIn("text", entry)
        with self.assertRaisesRegex(ValueError, "protected|raw array"):
            compact.prune(self.destination, ["large.json"])
        self.assertTrue(big.exists())

    def test_explicit_valid_aggregation_roundtrips_and_authorizes_only_selected_log(self):
        source = self.root / "large.jsonl"
        source.write_text('{"latency_ms":1}\n{"latency_ms":2}\n')
        summary = self.summary_for(source)
        self.export(summarized_text={"large.jsonl": summary}, max_text_bytes=1)
        entry = next(item for item in self.payload()["files"] if item["path"] == "large.jsonl")
        self.assertEqual(entry["summary"], summary)
        self.assertEqual(entry["kind"], "summarized_text")
        self.assertEqual(entry["evidence_status"], "aggregated_only")
        self.assertFalse(entry["original_recoverable"])
        self.assertNotIn("text", entry)
        self.assertIn("aggregated_only", str(compact._tables(self.payload())))
        self.assertIn("numeric_mean", str(compact._tables(self.payload())))
        compact.verify(self.destination, check_sources=True)
        result = compact.prune(self.destination, ["large.jsonl"])
        self.assertEqual(result["bytes"], summary["source_metadata"]["size"])
        self.assertFalse(source.exists())
        self.assertTrue(self.raw.exists())
        self.assertTrue((self.root / "summary.json").exists())
        receipt = json.loads((self.destination / compact.RECEIPT_NAME).read_text())
        self.assertEqual(receipt["deleted"]["large.jsonl"]["evidence_status"], "aggregated_only")
        reference = compact.verify_compacted_reference(self.root, "large.jsonl", entry["sha256"])
        self.assertFalse(reference["original_recoverable"])

    def test_invalid_or_wrong_source_aggregation_is_rejected(self):
        source = self.root / "summary.json"
        valid = self.summary_for(source)
        variants = [dict(valid, valid=False),
                    dict(valid, source_metadata=dict(valid["source_metadata"], sha256="0" * 64)),
                    dict(valid, source_metadata=dict(valid["source_metadata"], size=123456)),
                    dict(valid, source_metadata=dict(valid["source_metadata"], source="/different/source.json"))]
        for summary in variants:
            with self.subTest(summary=summary), self.assertRaisesRegex(ValueError, "aggregation|mismatch"):
                self.export(summarized_text={"summary.json": summary})
            self.assertFalse(self.destination.exists())
        self.assertTrue(source.exists())

    def test_changed_aggregated_source_blocks_deletion(self):
        source = self.root / "summary.json"
        self.export(summarized_text={"summary.json": self.summary_for(source)})
        source.write_text('{"status":"changed"}')
        with self.assertRaisesRegex(ValueError, "changed since export"):
            compact.prune(self.destination, ["summary.json", "output.npy"])
        self.assertTrue(source.exists())
        self.assertTrue(self.raw.exists())

    def test_aggregation_cannot_bypass_active_subtree(self):
        active = self.root / "active"
        active.mkdir()
        (active / "gpu-budget.json").write_text('{"active":{"phase":"running"}}')
        source = active / "records.json"
        source.write_text('[{"latency_ms":1}]')
        with self.assertRaisesRegex(ValueError, "protected"):
            self.export(summarized_text={"active/records.json": self.summary_for(source)})
        self.assertTrue(source.exists())

    def test_active_ledger_protects_entire_subtree_but_exports_others(self):
        active = self.root / "active-experiment"
        active.mkdir()
        (active / "gpu-budget.json").write_text('{"active":{"phase":"run"}}')
        (active / "output.npy").write_bytes(b"do not touch")
        self.export()
        self.assertFalse(any(item["path"].startswith("active-experiment/") for item in self.payload()["files"]))
        self.assertIn(str(active), self.payload()["protected"])
        child = active / "run01"
        child.mkdir()
        result = compact.export(child, child / "excel")
        self.assertEqual(result["status"], "protected")
        self.assertFalse((child / "excel").exists())

    def test_symlinks_and_source_snapshots_are_not_followed(self):
        outside = Path(self.directory.name) / "outside"
        outside.mkdir()
        (outside / "secret.npy").write_bytes(b"external")
        (self.root / "linked").symlink_to(outside, target_is_directory=True)
        (self.root / "linked.npy").symlink_to(outside / "secret.npy")
        snapshot = self.root / "source-snapshots"
        snapshot.mkdir()
        (snapshot / "copy.npy").write_bytes(b"frozen source fixture")
        self.export()
        names = {item["path"] for item in self.payload()["files"]}
        self.assertEqual(names, {"summary.json", "output.npy"})

    def test_both_budget_lock_conventions_protect_experiment(self):
        (self.root / "gpu-budget.json").write_text('{"active":null}')
        for name in ("gpu-budget.lock", "gpu-budget.json.lock"):
            with self.subTest(name=name), (self.root / name).open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(self.export()["status"], "protected")
                self.assertFalse(self.destination.exists())

    def test_compact_lock_prevents_concurrent_export_without_ledger(self):
        with (self.root / ".compact-results.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "another compact"):
                self.export()
        self.assertFalse(self.destination.exists())

    def test_repeat_export_excludes_itself(self):
        self.export()
        self.export(replace_existing=True)
        self.assertEqual({item["path"] for item in self.payload()["files"]}, {"summary.json", "output.npy"})

    def test_archive_tamper_blocks_pruning(self):
        self.export()
        with (self.destination / "results.xlsx").open("ab") as stream:
            stream.write(b"tampering")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            compact.prune(self.destination, ["output.npy"])
        self.assertTrue(self.raw.exists())

    def test_changed_source_blocks_entire_selected_set(self):
        second = self.root / "second.npy"
        second.write_bytes(b"second")
        self.export()
        second.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed since export"):
            compact.prune(self.destination, ["output.npy", "second.npy"])
        self.assertTrue(self.raw.exists())

    def test_new_activity_blocks_pruning(self):
        ledger = self.root / "gpu-budget.json"
        ledger.write_text('{"active":null}')
        self.export()
        ledger.write_text('{"active":{"phase":"continued"}}')
        with self.assertRaisesRegex(ValueError, "protected"):
            compact.prune(self.destination, ["output.npy"])
        self.assertTrue(self.raw.exists())

    def test_new_activity_in_child_blocks_whole_requested_deletion_batch(self):
        child = self.root / "child"
        child.mkdir()
        (child / "gpu-budget.json").write_text('{"active":null}')
        child_raw = child / "child.npy"
        child_raw.write_bytes(b"child array")
        self.export()
        (child / "gpu-budget.json").write_text('{"active":{"phase":"started"}}')
        with self.assertRaisesRegex(ValueError, "protected"):
            compact.prune(self.destination, ["output.npy", "child/child.npy"])
        self.assertTrue(self.raw.exists())
        self.assertTrue(child_raw.exists())

    def test_new_metadata_requires_fresh_export_before_pruning(self):
        self.export()
        (self.root / "summary.json").write_text('{"status":"updated"}')
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            compact.prune(self.destination, ["output.npy"])
        self.assertTrue(self.raw.exists())

    def test_pruned_reference_requires_verified_completed_receipt(self):
        self.export()
        result = compact.prune(self.destination, ["output.npy"])
        self.assertEqual(result["bytes"], len(b"binary-array-content"))
        self.assertFalse(self.raw.exists())
        reference = compact.verify_compacted_reference(self.root, "output.npy", self.digest)
        self.assertEqual(reference["status"], "retained_metrics_only")
        self.assertFalse(reference["raw_available"])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            compact.verify_compacted_reference(self.root, "output.npy", "0" * 64)
        with self.assertRaisesRegex(ValueError, "pruned evidence"):
            self.export(replace_existing=True)
        (self.destination / compact.RECEIPT_NAME).unlink()
        with self.assertRaises(FileNotFoundError):
            compact.verify_compacted_reference(self.root, "output.npy", self.digest)

    def test_undeclared_missing_raw_is_rejected(self):
        self.export()
        self.raw.unlink()
        with self.assertRaises(FileNotFoundError):
            compact.verify_compacted_reference(self.root, "output.npy", self.digest)

    def test_crash_after_unlink_is_recoverable_from_durable_intent(self):
        self.export()
        original_unlink = Path.unlink

        def interrupted_unlink(path, *args, **kwargs):
            original_unlink(path, *args, **kwargs)
            if path == self.raw:
                raise OSError("simulated crash after unlink")

        with mock.patch.object(Path, "unlink", interrupted_unlink):
            with self.assertRaisesRegex(OSError, "simulated crash"):
                compact.prune(self.destination, ["output.npy"])
        self.assertFalse(self.raw.exists())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            compact.verify_compacted_reference(self.root, "output.npy", self.digest)
        compact.prune(self.destination, ["output.npy"])
        result = compact.verify_compacted_reference(self.root, "output.npy", self.digest)
        self.assertEqual(result["status"], "retained_metrics_only")
        self.assertIsNone(result["path"])
        self.assertFalse(result["re_evaluable"])

    def test_metrics_view_limit_preserves_complete_jsonl(self):
        text = '{"a":1,"b":2}\n{"a":3,"b":4}\n'
        (self.root / "records.jsonl").write_text(text)
        self.export(metrics_limit=1)
        entry = next(item for item in self.payload()["files"] if item["path"] == "records.jsonl")
        self.assertEqual(entry["text"], text)
        self.assertIn("Excel view limit", str(compact._tables(self.payload())))


if __name__ == "__main__":
    unittest.main()
