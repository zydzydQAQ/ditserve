"""API orchestration tests with an injected fake bridge (no real core needed)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from ditserve import api

OPENPYXL = importlib.util.find_spec("openpyxl") is not None


class FakeBridge:
    binary = "fake-core"

    def __init__(self):
        self.calls = []

    def available(self):
        return True

    def generate(self, generator, cost_model=None):
        self.calls.append(("generate", generator))
        return [self._spec(index) for index in range(generator["count"])]

    def simulate(self, config, workload, cost_model=None):
        self.calls.append(("simulate", config))
        return {"schema_version": 2, "config": config,
                "cost_model": cost_model or {"provenance": "synthetic", "compute_ms": 12.0},
                "summary": {"policy": config["policy"], "provenance": "synthetic",
                            "total_arrivals": len(workload), "completed": len(workload),
                            "parity_failures": 0, "slo_attainment": 1.0},
                "records": [], "events": []}

    @staticmethod
    def _spec(index):
        return {"request_id": "r%04d" % index, "arrival_ms": float(index), "deadline_ms": 5000.0,
                "model": "reference", "height": 512, "width": 512, "frames": 1, "steps": 2,
                "seed": index, "cache_threshold": 0.15, "cache_enabled": True}


@unittest.skipUnless(OPENPYXL, "openpyxl not installed")
class SimulateFlowTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bridge = FakeBridge()

    def _kwargs(self):
        return {"count": 4, "policies": ("edf", "cache_only"), "output": self.root / "raw",
                "results_root": self.root / "results", "core": self.bridge}

    def test_simulate_runs_each_policy_and_archives(self):
        result = api.simulate(**self._kwargs())
        simulate_calls = [call for call in self.bridge.calls if call[0] == "simulate"]
        self.assertEqual([call[1]["policy"] for call in simulate_calls], ["edf", "cache_only"])
        self.assertTrue(result["archive"]["verified"])
        self.assertTrue(Path(result["archive"]["workbook"]).is_file())
        self.assertFalse((self.root / "raw").exists())  # cleaned after verified archival

    def test_keep_raw_retains_working_dir(self):
        kwargs = self._kwargs()
        kwargs["keep_raw"] = True
        api.simulate(**kwargs)
        self.assertTrue((self.root / "raw" / "workload.json").is_file())

    def test_trace_skips_generation(self):
        trace = self.root / "trace.json"
        trace.write_text('[{"request_id": "t", "arrival_ms": 0, "deadline_ms": 100}]', encoding="utf-8")
        kwargs = self._kwargs()
        kwargs["trace"] = trace
        api.simulate(**kwargs)
        self.assertFalse(any(call[0] == "generate" for call in self.bridge.calls))

    def test_nonempty_output_rejected(self):
        (self.root / "raw").mkdir()
        (self.root / "raw" / "junk").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            api.simulate(**self._kwargs())

    def test_empty_policies_rejected(self):
        kwargs = self._kwargs()
        kwargs["policies"] = []
        with self.assertRaisesRegex(ValueError, "at least one policy"):
            api.simulate(**kwargs)


@unittest.skipUnless(OPENPYXL, "openpyxl not installed")
class SweepFlowTest(unittest.TestCase):
    def test_grid_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bridge = FakeBridge()
            result = api.sweep(rates=(2.0,), memories=(128.0,), gpu_counts=(1, 2), patterns=("steady", "bursty"),
                               count=2, policies=("edf",), output=root / "raw",
                               results_root=root / "results", core=bridge)
            self.assertTrue(result["archive"]["verified"])
            generates = [call for call in bridge.calls if call[0] == "generate"]
            self.assertEqual(len(generates), 4)  # 2 patterns x 2 gpu counts
            configs = [call[1] for call in bridge.calls if call[0] == "simulate"]
            self.assertEqual({config["num_gpus"] for config in configs}, {1, 2})
            self.assertEqual({config["memory_mb"] for config in configs}, {128.0})


if __name__ == "__main__":
    unittest.main()
