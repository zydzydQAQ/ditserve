"""Exercise the user-facing archive command and historical evidence lookup."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from ditserve.cli import main
from ditserve.research.retention import verify_reference


class CompactCommandTests(unittest.TestCase):
    def test_verified_cleanup_remains_readable_by_historical_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results" / "experiment"
            root.mkdir(parents=True)
            original = b"recorded raw tensor"
            raw = root / "sample.npy"
            raw.write_bytes(original)
            metadata = root / "report.json"
            metadata.write_text(json.dumps({"status": "completed", "latency_ms": 1.5}))
            before = metadata.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["compact-results", str(root), "--prune-raw"]), 0)
            self.assertFalse(raw.exists())
            self.assertEqual(metadata.read_bytes(), before)
            result = verify_reference(root, raw, hashlib.sha256(original).hexdigest())
            self.assertEqual(result["status"], "retained_metrics_only")
            self.assertFalse(result["re_evaluable"])
            self.assertIsNone(result["path"])

    def test_active_command_preserves_working_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            root.mkdir()
            (root / "gpu-budget.json").write_text(json.dumps({"active": {"id": "running"}}))
            raw = root / "working.npy"
            raw.write_bytes(b"working input")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["compact-results", str(root), "--prune-raw"]), 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "protected")
            self.assertTrue(raw.exists())


if __name__ == "__main__":
    unittest.main()
