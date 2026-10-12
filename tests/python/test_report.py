"""Report rendering tests: summary.json / summary.csv / REPORT.md."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ditserve.report import write_report


def _runs():
    return {
        "edf": {"summary": {"provenance": "synthetic", "total_arrivals": 4, "completed": 4,
                            "slo_attainment": 1.0, "p99_latency_ms": 42.5,
                            "gpu_seconds_per_completed_request": None}},
        "cache_only": {"summary": {"provenance": "synthetic", "total_arrivals": 4, "completed": 3,
                                   "slo_attainment": 0.75, "p99_latency_ms": 50.0,
                                   "gpu_seconds_per_completed_request": 0.12}},
    }


class WriteReportTest(unittest.TestCase):
    def test_three_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_report(directory, _runs())
            for key in ("json", "csv", "markdown"):
                self.assertTrue(Path(paths[key]).is_file(), key)
            metadata = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertIn("not measured GPU performance", metadata["notice"])
            self.assertEqual(set(metadata["runs"]), {"edf", "cache_only"})
            with Path(paths["csv"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["policy"] for row in rows], ["edf", "cache_only"])
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("not measured GPU performance", markdown)
            self.assertIn("N/A", markdown)  # None rendered, never NaN

    def test_empty_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_report(directory, {})
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("No experiment runs supplied.", markdown)


if __name__ == "__main__":
    unittest.main()
