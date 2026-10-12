"""CLI tests: argument validation, doctor, archive commands, end-to-end smoke."""

import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from ditserve import cli
from ditserve.core import CoreBridge, POLICIES

OPENPYXL = importlib.util.find_spec("openpyxl") is not None
CORE = CoreBridge()


class ArgumentTest(unittest.TestCase):
    def test_policies_all(self):
        self.assertEqual(cli._policies("all"), list(POLICIES))

    def test_policies_reject_duplicates(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli._policies("edf,edf")

    def test_shapes(self):
        self.assertEqual(cli._shapes("512x512x1,768x768x2"), [(512, 512, 1), (768, 768, 2)])
        with self.assertRaises(argparse.ArgumentTypeError):
            cli._shapes("512x512")

    def test_csv(self):
        self.assertEqual(cli._csv("2,12", float), [2.0, 12.0])
        with self.assertRaises(argparse.ArgumentTypeError):
            cli._csv("")


class DoctorTest(unittest.TestCase):
    def test_doctor_json(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(["doctor"]), 0)
        report = json.loads(output.getvalue())
        self.assertIn("core_binary", report)
        self.assertIn("reference_simulation_ready", report)


class ArchiveCommandTest(unittest.TestCase):
    def test_verify_missing_root_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    cli.main(["archive", "verify", "--root", directory])


@unittest.skipUnless(OPENPYXL and CORE.available(), "needs openpyxl and the built C++ core")
class EndToEndTest(unittest.TestCase):
    def test_simulate_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = cli.main(["simulate", "--count", "4", "--policies", "edf,cache_only",
                             "--output", str(root / "raw"), "--results-root", str(root / "results")])
            self.assertEqual(code, 0)
            self.assertTrue((root / "results" / "archive.json").is_file())
            verified = io.StringIO()
            with contextlib.redirect_stdout(verified):
                cli.main(["archive", "verify", "--root", str(root / "results")])
            self.assertEqual(json.loads(verified.getvalue())["status"], "verified")

    def test_trace_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.json"
            trace.write_text(json.dumps([
                {"request_id": "long", "arrival_ms": 0, "deadline_ms": 1500, "steps": 4, "seed": 0},
                {"request_id": "urgent", "arrival_ms": 8, "deadline_ms": 200, "steps": 2, "seed": 1}]),
                encoding="utf-8")
            code = cli.main(["simulate", "--trace", str(trace), "--policies", "state_aware",
                             "--memory-mb", "72", "--max-batch", "1",
                             "--output", str(root / "raw"), "--results-root", str(root / "results")])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
