import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import read_json
from ditserve.research.protocol import DEFAULT_CONFIG, freeze_protocol, validate_protocol, verify_protocol
from ditserve.research.preflight import discover_gpus, select_devices


class ResearchProtocolTests(unittest.TestCase):
    def test_scientific_protocol_and_approximation_cannot_silently_change(self):
        config = read_json(DEFAULT_CONFIG)
        validate_protocol(config)
        for key, value in (("attention", "math"), ("transition_step", 15), ("epsilon", .1),
                           ("budget_gpu_seconds", 28801), ("frame_candidates", [17])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(dict(config, **{key: value}))

    def test_freeze_archives_and_detects_current_and_archive_drift(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "ditserve.research.protocol.dependency_identity", return_value={}):
            root = Path(directory)
            source = root / "python/ditserve/a.py"
            source.parent.mkdir(parents=True)
            source.write_text("x = 1\n")
            output = root / "run"
            frozen = freeze_protocol(DEFAULT_CONFIG, output, root / "ledger", root=root)
            self.assertEqual(verify_protocol(output / "protocol.json", root=root), frozen)
            source.write_text("x = 2\n")
            with self.assertRaisesRegex(ValueError, "current implementation"):
                verify_protocol(output / "protocol.json", root=root)
            self.assertEqual(verify_protocol(output / "protocol.json", root=root, verify_current=False), frozen)
            with self.assertRaisesRegex(ValueError, "changed"):
                freeze_protocol(DEFAULT_CONFIG, output, root / "ledger", root=root)
            (output / "sources/python/ditserve/a.py").write_text("altered")
            with self.assertRaisesRegex(ValueError, "archive mismatch"):
                verify_protocol(output / "protocol.json", root=root, verify_current=False)

    def test_driver_failure_and_homogeneous_requirement(self):
        failed = discover_gpus(lambda *a, **kw: subprocess.CompletedProcess(a, 9, "", "driver unavailable"))
        self.assertEqual(select_devices(failed, ["0"])["status"], "blocked")
        devices = [{"index": i, "uuid": "GPU-" + str(i), "name": "A100", "memory_mib": 40960,
                    "compute_capability": "8.0"} for i in range(4)]
        discovered = {"status": "available", "devices": devices}
        self.assertEqual(select_devices(discovered, ["0", "1", "2", "3"])["status"], "ready")
        devices[3]["name"] = "V100"
        self.assertEqual(select_devices(discovered, ["0", "1", "2", "3"])["status"], "blocked")
        with self.assertRaises(ValueError):
            select_devices(discovered, ["0", "GPU-0", "1", "2"])


if __name__ == "__main__":
    unittest.main()
