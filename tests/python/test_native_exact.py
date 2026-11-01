import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
SPEC = importlib.util.spec_from_file_location("native_exact", ROOT / "benchmarks" / "native_exact.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(candidate_hash="a" * 64, stops=True):
    def row(cell, digest):
        return {
            "cell": cell, "request_id": "v0", "kind": "video", "status": "completed",
            "forms": ["cfg2"], "spec": {"prompt": "a kettle", "seed": 7},
            "outputs": [{"generation": {"approximate": False}, "output": {"output_mode": "digest", "array_sha256": digest}}],
            "preemption_stop_samples": [{"stop_ms": 11.0}] if cell and stops else [],
        }
    return {
        "run_id": "r1",
        "phases": [{"cells": [
            {"cell": 0, "policy": "static_cfg2", "status": "completed"},
            {"cell": 1, "policy": "block_cfg2", "status": "completed"},
        ]}],
        "requests": [row(0, "a" * 64), row(1, candidate_hash)],
    }


class NativeExactTest(unittest.TestCase):
    def test_fixed_form_digest_and_actual_stop_are_required(self):
        result = MODULE.compare(record())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["matched_videos"], 1)
        self.assertEqual(result["preemption_stop_ms"]["p50"], 11.0)
        self.assertEqual(MODULE.compare(record(stops=False))["status"], "incomplete")
        changed = MODULE.compare(record(candidate_hash="b" * 64))
        self.assertEqual(changed["status"], "incomplete")
        self.assertEqual(changed["mismatches"], ["v0"])


if __name__ == "__main__":
    unittest.main()
