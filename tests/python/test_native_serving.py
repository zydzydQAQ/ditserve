import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
        self.assertFalse(phase["options"]["doorbell"])
        self.assertEqual(config["benchmark"]["trace"]["sha256"], phase["matrix"]["/trace_provenance"][0]["sha256"])
        continuous = native_serving.build_serving_config(
            "wan22", [f"GPU-{i}" for i in range(8)], profile(), pools,
            seed=201, load=1.0, pattern="poisson", video_fraction=0.2, count=40, doorbell=True,
        )
        self.assertTrue(continuous["phases"][0]["options"]["doorbell"])
        with self.assertRaisesRegex(ValueError, "GPU UUID"):
            native_serving.build_serving_config(
                "wan22", [f"GPU-{i}" for i in range(8)], profile(), pools,
                seed=201, load=1.0, pattern="poisson", video_fraction=0.2, count=40,
                require_profile_match=True,
            )
        bound = profile()
        bound["source"]["gpu_uuids"] = [f"GPU-{i}" for i in range(8)]
        matched = native_serving.build_serving_config(
            "wan22", [f"GPU-{i}" for i in range(8)], bound, pools,
            seed=201, load=1.0, pattern="poisson", video_fraction=0.2, count=40,
            require_profile_match=True,
        )
        self.assertTrue(matched["benchmark"]["profile_placement"]["matches"])
        sub_alloc = native_serving.build_serving_config(
            "wan22", [f"GPU-{i}" for i in range(4)], bound, pools,
            seed=201, load=1.0, pattern="gamma", cv=2.5, video_fraction=0.5, count=20,
            policies=native_serving.SUITES["sweep"], require_profile_match=True,
        )
        self.assertTrue(sub_alloc["benchmark"]["profile_placement"]["matches"])
        self.assertEqual(sub_alloc["benchmark"]["gpus"], 4)
        self.assertEqual(sub_alloc["benchmark"]["cv"], 2.5)
        self.assertNotIn("cfg2sp3", sub_alloc["phases"][0]["options"]["forms"])
        self.assertIn("cfg2sp2", sub_alloc["phases"][0]["options"]["forms"])

    def test_suites_name_registered_policies(self):
        self.assertEqual(native_serving.DEFAULT_POLICIES, native_serving.SUITES["main"])
        self.assertIn("ditserve", native_serving.SUITES["main"])
        for retired in ("heterodit_exact", "step_elastic", "vanilla_xdit", "genserve_40gb"):
            for suite in native_serving.SUITES.values():
                self.assertNotIn(retired, suite)
        self.assertEqual(native_serving.SUITES["preempt"], ["edf-cfg2", "block-cfg2"])
        merged = native_serving.SUITES["main+ablation"]
        self.assertEqual(len(merged), len(set(merged)))
        for name in merged:
            self.assertIn(name, native_serving.POLICY_DISPLAY)
        pools = {"image": pool("diffusiondb"), "video": pool("vidprom")}
        config = native_serving.build_serving_config(
            "wan22", [f"GPU-{i}" for i in range(8)], profile(), pools,
            seed=201, load=1.0, pattern="poisson", video_fraction=0.2, count=40,
            policies=native_serving.SUITES["main+ablation"],
        )
        self.assertEqual(config["phases"][0]["matrix"]["/policy"], merged)
        self.assertEqual(config["benchmark"]["policy_display"]["sbp-esp"], "SBP-ESP")
        self.assertTrue(config["benchmark"]["exact_mode"])

    def test_partition_share_follows_the_mix(self):
        cost = {"steps": 20, "step_compute_ms": 1147.0, "finish_ms": 1840.0}
        self.assertEqual(native_serving.partition_image_gpus(8, 0.5, cost, 6450.0), 2)
        self.assertEqual(native_serving.partition_image_gpus(8, 0.2, cost, 6450.0), 4)
        for fraction in (0.0, 0.2, 0.5, 0.8, 1.0):
            images = native_serving.partition_image_gpus(8, fraction, cost, 6450.0)
            self.assertTrue(1 <= images <= 6 and (8 - images) % 2 == 0)

    def test_approximation_is_explicit_and_separate(self):
        pools = {"image": pool("diffusiondb"), "video": pool("vidprom")}
        for policies in (["heterodit_v5"], ["ditserve-approx"], ["heterodit_approx"], ["heterodit_v3"]):
            with self.assertRaises(ValueError):
                native_serving.build_serving_config(
                    "wan22", [f"GPU-{i}" for i in range(8)], profile(), pools,
                    seed=201, load=1.0, pattern="poisson", video_fraction=0.2, count=40,
                    policies=policies,
                )
        config = native_serving.build_serving_config(
            "wan22", [f"GPU-{i}" for i in range(8)], profile(), pools,
            seed=201, load=1.0, pattern="poisson", video_fraction=0.2, count=40,
            policies=["heterodit_v5"], cfg_fold={"folded_steps": [12, 13]},
        )
        self.assertFalse(config["benchmark"]["exact_mode"])
        self.assertFalse(config["phases"][0]["options"]["async_rank0_decode"])
        self.assertEqual(config["workers"]["wan0"]["config"]["options"]["cfg_fold"]["folded_steps"], [12, 13])

    def test_quality_outputs_remain_pending_for_evaluation(self):
        pools = {"image": pool("diffusiondb"), "video": pool("vidprom")}
        config = native_serving.build_serving_config(
            "wan22", [f"GPU-{i}" for i in range(8)], profile(), pools,
            seed=201, load=1.0, pattern="poisson", video_fraction=0.2, count=40,
            policies=["heterodit_v5"], cfg_fold={"folded_steps": [12, 13]}, quality_per_kind=1,
        )
        self.assertEqual(config["phases"][0]["artifact_dependencies"], [{"consumer": "native_quality", "closed": False}])
        rows = config["phases"][0]["matrix"]["/workload"][0]
        self.assertEqual(sum(row["output_mode"] == "file" for row in rows), 2)
        self.assertTrue(all(row["output_mode"] == "file" for row in rows if row["request_id"] in ("w0-v0000", "w0-i0000")))

    def test_digest_trace_is_hash_bound_without_quality_files(self):
        pools = {"image": pool("diffusiondb"), "video": pool("vidprom")}
        config = native_serving.build_serving_config(
            "wan22", [f"GPU-{i}" for i in range(8)], profile(), pools,
            seed=201, load=1.0, pattern="bursty", video_fraction=0.5, count=16,
            policies=["static_cfg2", "block_cfg2"], forms=["cfg2"], doorbell=True, output_mode="digest",
        )
        rows = config["phases"][0]["matrix"]["/workload"][0]
        self.assertTrue(all(row["output_mode"] == "digest" for row in rows))
        self.assertEqual(config["phases"][0]["artifact_dependencies"], [])
        self.assertEqual(config["benchmark"]["output_mode"], "digest")

    def test_curvature_and_fixed_fold_share_predictor_settings(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile.json").write_text(json.dumps(profile()), encoding="utf-8")
            for kind, name in (("image", "diffusiondb"), ("video", "vidprom")):
                (root / (name + ".json")).write_text(json.dumps(pool(kind)), encoding="utf-8")
            common = [
                "native_serving.py", "--model", "wan22", "--profile", str(root / "profile.json"),
                "--datasets", str(root), "--gpu-uuids", ",".join(f"GPU-{i}" for i in range(8)),
                "--policies", "heterodit_v5", "--count", "32", "--video-fraction", "1.0",
            ]
            with patch.object(sys, "argv", common + ["--curvature", "--output", str(root / "curve.json")]):
                native_serving.main()
            with patch.object(sys, "argv", common + ["--fold-steps", "12,13,14,15,16,17,18", "--output", str(root / "fixed.json")]):
                native_serving.main()
            curve = json.loads((root / "curve.json").read_text())["workers"]["wan0"]["config"]["options"]["cfg_fold"]
            fixed = json.loads((root / "fixed.json").read_text())["workers"]["wan0"]["config"]["options"]["cfg_fold"]
            self.assertEqual(curve["folded_steps"], fixed["folded_steps"])
            self.assertEqual(curve["cfg_vel_alpha"], fixed["cfg_vel_alpha"])
            self.assertEqual(curve["cfg_damping"], fixed["cfg_damping"])
            self.assertEqual(curve["max_folded_steps"], len(fixed["folded_steps"]))
            self.assertEqual(fixed["max_folded_steps"], 7)

    def test_quality_arms_hold_form_and_resources_fixed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile.json").write_text(json.dumps(profile()), encoding="utf-8")
            for kind, name in (("image", "diffusiondb"), ("video", "vidprom")):
                (root / (name + ".json")).write_text(json.dumps(pool(kind)), encoding="utf-8")
            for arm in ("exact", "fixed", "curvature"):
                argv = ["native_serving.py", "--model", "wan22", "--profile", str(root / "profile.json"),
                        "--datasets", str(root), "--gpu-uuids", ",".join(f"GPU-{i}" for i in range(8)),
                        "--quality-arm", arm, "--count", "2", "--output", str(root / (arm + ".json"))]
                with patch.object(sys, "argv", argv):
                    native_serving.main()
                config = json.loads((root / (arm + ".json")).read_text())
                expected = "block-cfg2sp3" if arm == "exact" else "fold-cfg2sp3"
                self.assertEqual(config["phases"][0]["matrix"]["/policy"], [expected])
                self.assertEqual(config["phases"][0]["options"]["forms"], ["cfg2sp3"])
                self.assertEqual(config["benchmark"]["quality_arm"], arm)


if __name__ == "__main__":
    unittest.main()
