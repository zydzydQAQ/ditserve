"""Report exports must preserve working evidence and never interrupt reporting."""

import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.compact_results import verify
from ditserve.result_storage import export_report


class ResultStorageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.root = self.base / "run"
        self.root.mkdir()
        self.ledger = self.base / "external" / "accounting.json"
        self.ledger.parent.mkdir()
        self.ledger.write_text(json.dumps(dict(active=None)))

    def test_refresh_has_no_recursive_export_and_preserves_raw_working_data(self):
        raw = self.root / "working.npy"
        raw.write_bytes(b"temporary experiment data")
        report = self.root / "report.json"
        report.write_text(json.dumps(dict(status="completed", latency_ms=4.5)))
        first = export_report(self.root)
        self.assertEqual(first["status"], "verified")
        report.write_text(json.dumps(dict(status="completed", latency_ms=3.5)))
        second = export_report(self.root)
        self.assertEqual(second["status"], "verified")
        self.assertEqual(first["files"], second["files"])
        self.assertEqual(verify(self.root / "excel", check_sources=True)["status"], "verified")
        self.assertEqual(raw.read_bytes(), b"temporary experiment data")
        self.assertEqual(len(list(self.root.glob("excel*"))), 1)

    def test_external_active_protocol_ledger_prevents_export(self):
        self.ledger.write_text(json.dumps(dict(active=dict(session_id="live"))))
        (self.root / "protocol.json").write_text(json.dumps(dict(ledger=str(self.ledger))))
        with patch("ditserve.compact_results.export") as exporter:
            result = export_report(self.root)
        self.assertEqual(result["status"], "protected")
        exporter.assert_not_called()
        self.assertFalse((self.root / "excel").exists())

    def test_directions_report_survives_export_failure_without_changing_evidence(self):
        from ditserve.directions.reporting import write_report
        from ditserve.directions.runner import PHASES

        evidence = self.root / "direction1" / "supervisor-profile.json"
        evidence.parent.mkdir()
        evidence.write_text(json.dumps(dict(status="passed", eligible=True, rows=[])))
        before = evidence.read_bytes()
        with patch("ditserve.compact_results.export", side_effect=OSError("disk unavailable")):
            with self.assertWarnsRegex(RuntimeWarning, "Excel export deferred"):
                result = write_report(self.root, self.ledger, PHASES)
        self.assertEqual(evidence.read_bytes(), before)
        self.assertTrue((self.root / "REPORT.md").is_file())
        self.assertEqual(json.loads((self.root / "report.json").read_text()), result)

    def test_mainline_report_commands_export_after_publishing(self):
        for name, report_name, export_target in (
            ("directions.runner", "directions.reporting", "ditserve.result_storage.export_report"),
            ("scheduling.runner", "scheduling.reporting", "ditserve.scheduling.reporting.export_report"),
            ("scheduling.next_runner", "scheduling.next_reporting", "ditserve.scheduling.next_reporting.export_report"),
        ):
            with self.subTest(runner=name), contextlib.ExitStack() as stack:
                module = importlib.import_module("ditserve." + name)
                reporting = importlib.import_module("ditserve." + report_name)
                output = self.root / name
                output.mkdir()
                evidence = output / "working.npy"
                evidence.write_bytes(b"retained working evidence")
                value = dict(status="completed", budget_path=str(self.ledger), source=name)
                calls = []
                stack.enter_context(patch.object(reporting, "build_report",
                    side_effect=lambda *args, **kwargs: calls.append("report") or value))
                stack.enter_context(patch.object(reporting, "render_markdown",
                    side_effect=lambda *args: calls.append("render") or "Published report\n"))
                def export_after_publication(root, *, ledger):
                    self.assertEqual(Path(root), output)
                    self.assertEqual(Path(ledger), self.ledger)
                    self.assertEqual(json.loads((output / "report.json").read_text()), value)
                    self.assertEqual((output / "REPORT.md").read_text(), "Published report\n")
                    calls.append("export")
                    return dict(status="verified")
                exporter = stack.enter_context(patch(export_target, side_effect=export_after_publication))
                stack.enter_context(patch.object(module, "load_config", return_value={}))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                self.assertEqual(module.main(["report", "--output", str(output), "--ledger", str(self.ledger)]), 0)
                self.assertEqual(calls, ["report", "render", "export"])
                exporter.assert_called_once()
                self.assertEqual(evidence.read_bytes(), b"retained working evidence")


if __name__ == "__main__":
    unittest.main()
