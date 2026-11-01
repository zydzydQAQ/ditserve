import importlib.util
import sys
import unittest
from pathlib import Path

from ditserve.evaluation.common import seal


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
import native_serving


def pool(kind):
    return seal({
        "kind": kind,
        "source": {"revision": "a" * 40},
        "selection": {"method": "test"},
        "samples": [{"sample_id": f"{kind}-{i}", "prompt": f"{kind} prompt {i}"} for i in range(100)],
    })


def profile():
    return {
        "provenance": "mixed",
        "classes": {
            "wan22-preflight": {
                "kind": "video", "steps": 20, "blocks_per_step": 60, "prepare_ms": 0.0,
                "encode_ms": 20.0, "step_compute_ms": 1000.0, "finish_ms": 100.0,
                "comm_mb_per_step": 100.0, "collectives_per_step": 240, "cfg_exchange_mb": 1.0,
                "decode_chunks": 13,
            },
            "sd35-preflight": {"kind": "image", "service_ms": 250.0},
        },
        "links": {str(d): {"gbps": 5.0, "latency_ms": 0.1} for d in (2, 3, 4, 6, 8)},
        "source": {"shape": {
            "frames": 49, "steps": 20, "height": 512, "width": 896,
            "guidance_scale": 5.0, "cache": False, "threshold": 0.0, "max_continuous": 1,
            "negative_prompt": "", "prompt": "fixture", "seed": 201,
        }},
    }


class NativeServingTest(unittest.TestCase):
    def test_exact_trace_is_frozen_and_has_no_approximation(self):
        pools = {"image": pool("diffusiondb"), "video": pool("vidprom")}
        config = native_serving.build_serving_config(
            "wan22", [f"GPU-{i}" for i in range(8)], profile(), pools,
            seed=201, load=1.0, pattern="poisson", video_fraction=0.2, count=40,
        )
        phase = config["phases"][0]
        self.assertEqual(len(phase["matrix"]["/workload"][0]), 40)
        counts = config["benchmark"]["trace"]["manifest"]["counts"]
        self.assertEqual(counts, {"video": 8, "image": 32})
        self.assertTrue(config["benchmark"]["exact_mode"])
        self.assertNotIn("cfg_fold", config["workers"]["wan0"]["config"]["options"])
        self.assertFalse(phase["options"]["async_rank0_decode"])
        self.assertEqual(config["benchmark"]["trace"]["sha256"], phase["matrix"]["/trace_provenance"][0]["sha256"])

    def test_approximation_is_explicit_and_separate(self):
        pools = {"image": pool("diffusiondb"), "video": pool("vidprom")}
        with self.assertRaises(ValueError):
            native_serving.build_serving_config(
                "wan22", [f"GPU-{i}" for i in range(8)], profile(), pools,
                seed=201, load=1.0, pattern="poisson", video_fraction=0.2, count=40,
                policies=["heterodit_v5"],
            )
        config = native_serving.build_serving_config(
            "wan22", [f"GPU-{i}" for i in range(8)], profile(), pools,
            seed=201, load=1.0, pattern="poisson", video_fraction=0.2, count=40,
            policies=["heterodit_v5"], cfg_fold={"folded_steps": [12, 13]},
        )
        self.assertFalse(config["benchmark"]["exact_mode"])
        self.assertTrue(config["phases"][0]["options"]["async_rank0_decode"])
        self.assertEqual(config["workers"]["wan0"]["config"]["options"]["cfg_fold"]["folded_steps"], [12, 13])


if __name__ == "__main__":
    unittest.main()
