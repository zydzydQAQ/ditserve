"""Archive lifecycle tests: finalize -> verify -> rebuild, plus tamper detection."""

import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from ditserve import archive
from ditserve.report import write_report
from ditserve.util import atomic_json

OPENPYXL = importlib.util.find_spec("openpyxl") is not None


def _make_raw(root):
    """A minimal but realistic raw run directory."""
    raw = Path(root) / "raw"
    raw.mkdir(parents=True)
    specs = [{"request_id": "a", "arrival_ms": 0.0, "deadline_ms": 5000.0, "model": "reference",
              "height": 512, "width": 512, "frames": 1, "steps": 2, "seed": 0,
              "cache_threshold": 0.15, "cache_enabled": True}]
    atomic_json(raw / "workload.json", specs)
    run = {"schema_version": 2, "config": {"policy": "edf", "num_gpus": 1},
           "cost_model": {"provenance": "synthetic", "compute_ms": 12.0},
           "summary": {"policy": "edf", "total_arrivals": 1, "completed": 1, "parity_failures": 0,
                       "slo_attainment": 1.0},
           "records": [], "events": []}
    atomic_json(raw / "edf.json", run)
    atomic_json(raw / "cost_model.json", run["cost_model"])
    write_report(raw, {"edf": run})
    return raw


@unittest.skipUnless(OPENPYXL, "openpyxl not installed")
class ArchiveLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.results = self.root / "results"
        self.raw = _make_raw(self.root)
        self.archived = archive.finalize_run(self.raw, "synthetic", "step-a", results_root=self.results)

    def test_finalize_layout(self):
        step = self.results / "synthetic" / "step-a"
        for name in ("data.json.gz", "results.xlsx", "REPORT.md"):
            self.assertTrue((step / name).is_file(), name)
        self.assertTrue((self.results / "archive.json").is_file())
        self.assertTrue((self.results / "INDEX.md").is_file())
        self.assertTrue(self.archived["verified"])

    def test_verify_passes(self):
        result = archive.verify(self.results)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["steps"], 1)
        self.assertGreaterEqual(result["source_files"], 5)

    def test_rebuild_restores_outputs(self):
        step = self.results / "synthetic" / "step-a"
        report = (step / "REPORT.md").read_text(encoding="utf-8")
        (step / "results.xlsx").unlink()
        (step / "REPORT.md").unlink()
        archive.rebuild(step)
        self.assertTrue((step / "results.xlsx").is_file())
        self.assertEqual((step / "REPORT.md").read_text(encoding="utf-8"), report)
        archive.verify(self.results)

    def test_tampered_report_detected(self):
        step = self.results / "synthetic" / "step-a"
        (step / "REPORT.md").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "report differs"):
            archive.verify(self.results)

    def test_tampered_evidence_detected(self):
        step = self.results / "synthetic" / "step-a"
        with gzip.open(step / "data.json.gz", "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        data["files"]["workload.json"]["text"] = "[]"
        encoded = (json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        with gzip.GzipFile(step / "data.json.gz", "wb", mtime=0) as stream:
            stream.write(encoded)
        with self.assertRaisesRegex(ValueError, "evidence changed|cell differs|report differs"):
            archive.verify(self.results)

    def test_foreign_raw_source_rejected(self):
        other = _make_raw(self.root / "elsewhere")
        with self.assertRaisesRegex(ValueError, "another or unknown raw source"):
            archive.finalize_run(other, "synthetic", "step-a", results_root=self.results)

    def test_unsafe_step_names_rejected(self):
        with self.assertRaises(ValueError):
            archive.finalize_run(self.raw, "synthetic", "../escape", results_root=self.results)


if __name__ == "__main__":
    unittest.main()
