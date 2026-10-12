"""CoreBridge protocol tests using a fake core binary (a Python script)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from ditserve.core import CoreBridge, CoreError

FAKE = """\
#!%s
import json, sys
payload = json.load(sys.stdin)
mode = %r
if mode == "fail":
    print("boom", file=sys.stderr)
    sys.exit(2)
if mode == "badjson":
    print("not json")
    sys.exit(0)
if mode == "nan":
    print('{"value": NaN}')
    sys.exit(0)
json.dump({"command": sys.argv[1], "echo": payload}, sys.stdout)
"""


class FakeCoreTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def bridge(self, mode="ok"):
        binary = Path(self.directory.name) / ("fake-" + mode)
        binary.write_text(FAKE % (sys.executable, mode), encoding="utf-8")
        binary.chmod(0o755)
        return CoreBridge(binary)

    def test_roundtrip_payload(self):
        reply = self.bridge().call("simulate", {"schema_version": 2, "workload": []})
        self.assertEqual(reply["command"], "simulate")
        self.assertEqual(reply["echo"]["schema_version"], 2)

    def test_generate_and_simulate_wrap_schema(self):
        bridge = self.bridge()
        self.assertEqual(bridge.generate({"count": 1})["echo"]["generator"], {"count": 1})
        reply = bridge.simulate({"policy": "edf"}, [], {"compute_ms": 1})
        self.assertEqual(reply["echo"]["cost_model"], {"compute_ms": 1})
        self.assertEqual(reply["echo"]["workload"], [])

    def test_core_failure_raises_core_error(self):
        with self.assertRaisesRegex(CoreError, "boom"):
            self.bridge("fail").call("generate", {})

    def test_invalid_json_raises_core_error(self):
        with self.assertRaises(CoreError):
            self.bridge("badjson").call("generate", {})

    def test_non_finite_output_raises_core_error(self):
        with self.assertRaises(CoreError):
            self.bridge("nan").call("generate", {})

    def test_missing_binary_raises_core_error(self):
        with self.assertRaisesRegex(CoreError, "not found"):
            CoreBridge(Path(self.directory.name) / "absent").call("generate", {})

    def test_non_finite_input_rejected(self):
        with self.assertRaises(ValueError):
            self.bridge().call("generate", {"rate": float("nan")})

    def test_environment_variable_selects_binary(self):
        bridge = self.bridge()
        os.environ["DITSERVE_CORE"] = str(bridge.binary)
        try:
            self.assertTrue(CoreBridge().available())
        finally:
            del os.environ["DITSERVE_CORE"]


if __name__ == "__main__":
    unittest.main()
