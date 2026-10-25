"""CPU-only native-controller admission tests; no real nvidia-smi/CUDA use."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from ditserve import api
from ditserve.runtime.client import CoreBridge, CoreError
from ditserve.runtime import PACKAGE_ROOT


class _FixtureBridge(CoreBridge):
    def __init__(self, package_root):
        super().__init__()
        self.package_root = package_root

    def call(self, command, payload):
        return super().call(command, dict(payload, package_root=str(self.package_root)))


@unittest.skipUnless(CoreBridge().available(), "build the C++ core first")
class ProfileAdmissionTests(unittest.TestCase):
    gpu = "GPU-00000000-0000-0000-0000-000000000000"

    @classmethod
    def setUpClass(cls):
        cls.package_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.package_directory.cleanup)
        cls.package_root = Path(cls.package_directory.name) / "python"
        for package in ("ditserve", "xfuser"):
            source = PACKAGE_ROOT / package
            for path in source.rglob("*.py"):
                target = cls.package_root / package / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bridge = _FixtureBridge(self.package_root)
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self.observations = self.root / "observations.jsonl"
        environment = patch.dict(os.environ, {"PATH": str(self.fake_bin) + os.pathsep + os.environ.get("PATH", "")})
        environment.start()
        self.addCleanup(environment.stop)

    def fake_smi(self, states, delay=0):
        """Every command is a local fake; busy PID belongs to this test."""
        script = """#!%s
import json, pathlib, sys, time
states = %r
path = pathlib.Path(%r)
if '--query-gpu=uuid,name' in sys.argv:
    print(%r + ', CPU fixture GPU')
elif '--query-compute-apps=pid,gpu_uuid,used_memory' in sys.argv:
    prior = path.read_text().splitlines() if path.exists() else []
    state = states[min(len(prior), len(states)-1)]
    time.sleep(%r)
    with path.open('a') as out:
        out.write(json.dumps({'state':state,'monotonic':time.monotonic()})+'\\n')
    if state == 'busy':
        print(str(%d) + ', ' + %r + ', 10')
    elif state == 'unknown':
        sys.exit(19)
else:
    raise RuntimeError('unexpected nvidia-smi invocation: ' + repr(sys.argv))
""" % (sys.executable, states, str(self.observations), self.gpu, delay, os.getpid(), self.gpu)
        path = self.fake_bin / "nvidia-smi"
        path.write_text(script)
        path.chmod(0o700)

    def config(self, *, gpu=False, wait=1, poll=0.02):
        worker = {"backend": "wan_profile" if gpu else "mock", "config": {}}
        if gpu:
            worker.update(device="cuda:0", gpu_uuid=self.gpu)
        return dict(schema="ditserve.study.v1", experiment="profile_admission",
            workers={"model": worker}, budget=dict(limit_gpu_seconds=5, watchdog_seconds=5),
            phases=[dict(id="probe", kind="profile", worker="model", actions=[dict(op="hello")],
                         admission_wait_seconds=wait, admission_poll_seconds=poll)])

    def run_study(self, config, name="run", *, allow_gpu=False):
        return api.run(config, run_id=name, work_dir=self.root / name,
                       results_root=self.root / "results", ledger=self.root / (name + "-budget.json"),
                       core=self.bridge, allow_gpu=allow_gpu)

    def assert_uncharged(self, result):
        self.assertIsNone(result["budget"]["active"])
        self.assertEqual(result["budget"]["used_gpu_seconds"], 0)
        self.assertEqual(result["budget"]["sessions"], [])

    def test_mock_wait_observes_idle_twice_and_has_no_gpu_charge(self):
        self.fake_smi(["unknown"])  # A CPU fixture must not probe any GPU.
        result = self.run_study(self.config(wait=0.2, poll=0.06))
        self.assertEqual(result["status"], "completed", result)
        self.assertGreaterEqual(result["admission_wait_seconds"], 0.055)
        self.assertEqual(result["budget"]["used_gpu_seconds"], 0)
        self.assertIsNone(result["budget"]["active"])
        self.assertEqual(len(result["budget"]["sessions"]), 1)
        self.assertFalse(self.observations.exists())
        hello = result["phases"][0]["cells"][0]["operations"][0]["ranks"][0]
        self.assertFalse(hello["loaded"])

    def test_invalid_wait_or_poll_is_rejected_before_budget_admission(self):
        cases = [("admission_wait_seconds", value) for value in (-1, 86400.1, True, "2", None)]
        cases += [("admission_poll_seconds", value) for value in (0, -1, 60.1, False, "1", None)]
        self.fake_smi(["unknown"])
        for index, (key, value) in enumerate(cases):
            with self.subTest(key=key, value=value):
                config = self.config()
                config["phases"][0][key] = value
                name = "invalid-%d" % index
                with self.assertRaises(CoreError):
                    self.run_study(config, name)
                ledger = self.root / (name + "-budget.json")
                if ledger.exists():
                    self.assertIsNone(json.loads(ledger.read_text())["active"])
                    self.assertEqual(json.loads(ledger.read_text())["sessions"], [])
        self.assertFalse(self.observations.exists())

    def test_busy_gpu_timeout_never_starts_worker_or_charges_budget(self):
        self.fake_smi(["busy"])
        # Long enough for a cold probe worker (~0.3 s without bytecode caches) plus a Python fake.
        result = self.run_study(self.config(gpu=True, wait=2), allow_gpu=True)
        self.assertEqual(result["status"], "blocked", result)
        self.assertIn("timed out", result["admission"]["error"])
        # The final bounded probe may itself hit the total wait deadline and
        # become unknown.  Every successfully completed observation was busy.
        self.assertIn(result["admission"]["observation"]["status"], ("busy", "unknown"))
        observations = [json.loads(line) for line in self.observations.read_text().splitlines()]
        self.assertTrue(observations)
        self.assertTrue(all(row["state"] == "busy" for row in observations))
        self.assertEqual(result["phases"], [])
        self.assert_uncharged(result)
        self.assertFalse(list((self.root / "run" / "attempts").rglob("model.json")))

    def test_unknown_gpu_observation_is_not_admitted_as_idle(self):
        self.fake_smi(["unknown"])
        result = self.run_study(self.config(gpu=True, wait=0.12), allow_gpu=True)
        self.assertEqual(result["status"], "blocked", result)
        self.assertEqual(result["admission"]["observation"]["status"], "unknown")
        self.assert_uncharged(result)

    def test_busy_between_idle_observations_resets_the_consecutive_count(self):
        self.fake_smi(["idle", "busy", "idle", "idle"])
        # Four observations with cold probe starts can exceed 3 s on a loaded or slow-filesystem host.
        result = self.run_study(self.config(gpu=True, wait=10), allow_gpu=True)
        self.assertEqual(result["status"], "completed", result)
        observations = [json.loads(line) for line in self.observations.read_text().splitlines()]
        self.assertEqual([row["state"] for row in observations[:4]], ["idle", "busy", "idle", "idle"])
        self.assertGreaterEqual(len(observations), 4)
        hello = result["phases"][0]["cells"][0]["operations"][0]["ranks"][0]
        self.assertEqual(hello["backend"], "wan_profile")
        self.assertFalse(hello["loaded"])  # No CUDA/model load is permitted by these tests.
        session = result["budget"]["sessions"][0]
        self.assertGreaterEqual(session["started_monotonic"], observations[3]["monotonic"])
        self.assertAlmostEqual(result["budget"]["used_gpu_seconds"], session["gpu_seconds"])
        self.assertAlmostEqual(session["gpu_seconds"], session["elapsed_seconds"])
        self.assertGreater(result["admission_wait_seconds"], 0.04)

    def test_wait_deadline_cannot_admit_a_second_idle_observation_after_expiry(self):
        self.fake_smi(["idle"])
        result = self.run_study(self.config(gpu=True, wait=0.05, poll=0.2), allow_gpu=True)
        self.assertEqual(result["status"], "blocked", result)
        self.assert_uncharged(result)


if __name__ == "__main__":
    unittest.main()
