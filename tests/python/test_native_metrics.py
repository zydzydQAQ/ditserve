import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_metrics", ROOT / "benchmarks" / "native_metrics.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NativeMetricsTest(unittest.TestCase):
    def test_per_kind_slo_goodput_and_gpu_cost(self):
        record = {
            "experiment": "fixture", "run_id": "r1", "status": "completed",
            "metrics": {}, "budget": {"used_gpu_seconds": 40.0},
            "phases": [{"cells": [{
                "cell": 0, "policy": "block_elastic", "order": "forward", "status": "completed",
                "elapsed_ms": 2000.0, "gpus": 2,
            }]}],
            "requests": [
                {"cell": 0, "kind": "video", "status": "completed", "arrival_ms": 0.0,
                 "deadline_ms": 100.0, "finish_ms": 120.0, "advance_samples": [{"steady": True, "elapsed_ms": 5.0}],
                 "preemption_stop_samples": [{"stop_ms": 7.5}]},
                {"cell": 0, "kind": "image", "status": "completed", "arrival_ms": 0.0, "start_ms": 50.0,
                 "deadline_ms": 100.0, "finish_ms": 80.0, "preemption_reason": "block_preempt",
                 "advance_samples": [{"steady": True, "elapsed_ms": 7.0}]},
            ],
        }
        result = MODULE.summarize_run(record)
        cell = result["cells"][0]
        self.assertEqual(cell["by_kind"]["video"]["deadline_attainment"], 0.0)
        self.assertEqual(cell["by_kind"]["image"]["deadline_attainment"], 1.0)
        self.assertEqual(cell["by_kind"]["image"]["goodput_rps"], 0.5)
        self.assertEqual(cell["gpu_seconds_per_ontime_request"], 4.0)
        self.assertEqual(cell["block_quantum_elapsed_ms"]["count"], 2)
        self.assertEqual(cell["preemption_response_ms"]["p50"], 50.0)
        self.assertEqual(cell["preempting_request_count"], 1)
        self.assertEqual(cell["preemption_stop_ms"]["p50"], 7.5)
        self.assertEqual(result["used_gpu_seconds"], 40.0)

    def test_fission_records_capacity_and_actual_image_overlap(self):
        video = {
            "cell": 0, "kind": "video", "status": "completed", "arrival_ms": 0, "start_ms": 0,
            "deadline_ms": 100, "finish_ms": 40, "spec": {"steps": 20},
            "cfg_fold": {"full_cfg_steps": 19, "folded_cfg_steps": 1, "folded_step_indices": [12],
                         "final_step_exact_anchor": True, "curvature_gate_ms": 2.5, "cfg_state_bytes": 64},
            "advance_samples": [
                {"form": "cfg2sp3", "group": [0, 1, 2, 3, 4, 5], "cfg_folded": False,
                 "start_ms": 0, "finish_ms": 10, "elapsed_ms": 10},
                {"form": "sp3", "group": [0, 1, 2], "cfg_folded": True,
                 "start_ms": 10, "finish_ms": 30, "elapsed_ms": 20},
            ],
        }
        image = {"cell": 0, "request_id": "i0", "kind": "image", "status": "completed", "gpu_group": [4],
                 "arrival_ms": 0, "start_ms": 15, "deadline_ms": 100, "finish_ms": 25}
        result = MODULE.approximation_summary([video, image])
        self.assertEqual(result["released_gpu_ms_capacity"], 60)
        self.assertEqual(result["image_overlap_on_released_gpu_ms"], 10)
        self.assertEqual(result["image_requests_on_released_gpus"], 1)
        self.assertEqual(result["curvature_gate_ms"]["p50"], 2.5)
        video["advance_samples"][1].pop("start_ms")
        with self.assertRaisesRegex(ValueError, "timed execution"):
            MODULE.approximation_summary([video, image])


if __name__ == "__main__":
    unittest.main()
