"""CoreBridge protocol tests using a fake core binary (a Python script)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ditserve import api
from ditserve.runtime.client import CoreBridge, CoreError, native_binary
from ditserve.runtime import PACKAGE_ROOT, ROOT

FAKE = """\
#!%s
import json, os, sys
mode = %r
environment = {key: os.environ.get(key) for key in ('DITSERVE_PYTHON', 'DITSERVE_PACKAGE_ROOT', 'DITSERVE_ROOT')}
if mode == "forward":
    json.dump({"argv": sys.argv[1:], "environment": environment}, sys.stdout)
    sys.exit(7)
assert sys.argv[1] == "rpc", sys.argv
payload = json.load(sys.stdin)
if mode == "fail":
    print("boom", file=sys.stderr)
    sys.exit(2)
if mode == "badjson":
    print("not json")
    sys.exit(0)
if mode == "nan":
    print('{"value": NaN}')
    sys.exit(0)
if mode == "business":
    json.dump({"status": "deferred"}, sys.stdout)
    sys.exit(1)
json.dump({"command": sys.argv[2], "echo": payload, "environment": environment}, sys.stdout)
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
        self.assertEqual(reply["echo"], {"schema_version": 2, "workload": []})
        self.assertEqual(reply["environment"]["DITSERVE_PYTHON"], os.environ.get("DITSERVE_PYTHON", sys.executable))
        self.assertEqual(reply["environment"]["DITSERVE_PACKAGE_ROOT"], os.environ.get("DITSERVE_PACKAGE_ROOT", str(PACKAGE_ROOT)))

    def test_generate_and_simulate_wrap_schema(self):
        bridge = self.bridge()
        self.assertEqual(bridge.generate({"count": 1})["echo"]["generator"], {"count": 1})
        reply = bridge.simulate({"policy": "edf"}, [], {"compute_ms": 1})
        self.assertEqual(reply["echo"]["cost_model"], {"compute_ms": 1})
        self.assertEqual(reply["echo"]["workload"], [])

    def test_core_failure_raises_core_error(self):
        with self.assertRaisesRegex(CoreError, "boom"):
            self.bridge("fail").call("generate", {})
        for command in ("run", "doctor", "report", "results"):
            with self.subTest(command=command):
                self.assertEqual(self.bridge("business").call(command, {}), {"status": "deferred"})

    def test_invalid_json_output_raises_core_error(self):
        for mode in ("badjson", "nan"):
            with self.subTest(mode=mode), self.assertRaises(CoreError):
                self.bridge(mode).call("generate", {})

    def test_missing_binary_raises_core_error(self):
        with self.assertRaisesRegex(CoreError, "not found"):
            CoreBridge(Path(self.directory.name) / "absent").call("generate", {})

    def test_non_finite_input_rejected(self):
        with self.assertRaises(ValueError):
            self.bridge().call("generate", {"rate": float("nan")})

    def test_environment_variable_selects_binary(self):
        bridge = self.bridge()
        self.assertEqual(native_binary(os.path.relpath(bridge.binary)), bridge.binary)
        with patch.dict(os.environ, {"DITSERVE_CORE": str(bridge.binary), "DITSERVE_PYTHON": "custom-python",
                                     "DITSERVE_PACKAGE_ROOT": "custom-package", "DITSERVE_ROOT": "custom-root"}):
            self.assertTrue(CoreBridge().available())
            self.assertEqual(CoreBridge().call("generate", {})["environment"], {
                "DITSERVE_PYTHON": "custom-python", "DITSERVE_PACKAGE_ROOT": "custom-package", "DITSERVE_ROOT": "custom-root"})
        root = Path(self.directory.name)
        candidates = [root / ".cache/build/ditserve", root / "prefix/bin/ditserve", root / "path/bin/ditserve"]
        for candidate in candidates:
            candidate.parent.mkdir(parents=True)
            candidate.write_text("#!/bin/sh\nexit 0\n")
            candidate.chmod(0o755)
        environment = {key: value for key, value in os.environ.items() if key != "DITSERVE_CORE"}
        environment["PATH"] = str(candidates[-1].parent)
        with patch.dict(os.environ, environment, clear=True), patch("ditserve.runtime.client.ROOT", root), \
                patch("ditserve.runtime.client.sys.prefix", str(root / "prefix")):
            for expected in candidates:
                self.assertEqual(native_binary(), expected)
                if expected == candidates[-1]:
                    with patch.dict(os.environ, {"DITSERVE_CORE": "ditserve"}):
                        self.assertEqual(native_binary(), expected)
                expected.unlink()

    def test_public_client_only_forwards_explicit_options(self):
        bridge = self.bridge()
        config = dict(experiment="example", budget=dict(limit_gpu_seconds=None), model=Path("weights/local"))
        result = api.run(config, core=bridge, work_dir=Path("work"), results_root=None, allow_gpu=False)
        self.assertEqual(result["command"], "run")
        self.assertEqual(result["echo"], dict(config=dict(config, model="weights/local"), work_dir="work", allow_gpu=False))
        result = api.simulate(core=bridge, count=3, seed=None)
        self.assertEqual(result["command"], "simulate-study")
        self.assertEqual(result["echo"], dict(count=3))
        self.assertEqual(api.simulate(core=bridge)["echo"], {})
        self.assertEqual(api.doctor(core=bridge)["command"], "doctor")
        self.assertEqual(api.doctor(core=bridge)["echo"], {})

    def test_module_entry_execs_native_arguments_and_preserves_exit_status(self):
        binary = self.bridge("forward").binary
        environment = dict(os.environ, DITSERVE_CORE=str(binary), PYTHONPATH=str(PACKAGE_ROOT))
        arguments = ["unknown-native-command", "--opaque-option", "a value"]
        result = subprocess.run([sys.executable, "-m", "ditserve", *arguments],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 7, result.stderr)
        reply = json.loads(result.stdout)
        self.assertEqual(reply["argv"], arguments)
        self.assertEqual(reply["environment"]["DITSERVE_ROOT"], os.environ.get("DITSERVE_ROOT", str(ROOT)))


if __name__ == "__main__":
    unittest.main()
