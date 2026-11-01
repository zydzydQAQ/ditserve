import importlib.util
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = os.environ.get("DITSERVE_PACKAGE_ROOT")
BENCHMARKS = Path(PACKAGE_ROOT).parent / "benchmarks" if PACKAGE_ROOT else ROOT / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location("cluster_config", BENCHMARKS / "cluster_config.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
import native_study


class ClusterConfigTest(unittest.TestCase):
    def test_native_preflight_uses_allocated_hardware_and_current_models(self):
        uuids = [f"GPU-{index}" for index in range(8)]
        config = MODULE.build_config("wan21", uuids)
        self.assertEqual(len(config["workers"]), 16)
        self.assertEqual([slot["video"] for slot in config["phases"][0]["gpus"]], [f"wan{i}" for i in range(8)])
        self.assertEqual([config["workers"][f"wan{i}"]["gpu_uuid"] for i in range(8)], uuids)
        phase = config["phases"][0]
        self.assertEqual(phase["profile"]["provenance"], "synthetic")
        self.assertEqual(phase["options"]["forms"], ["cfg2sp3"])
        self.assertTrue(phase["options"]["async_rank0_decode"])
        self.assertFalse(config["benchmark"]["performance_qualified"])
        self.assertFalse(config["benchmark"]["quality_qualified"])
        self.assertEqual(config["workers"]["wan0"]["config"]["model_path"], "/shared/data/DITSERVE/models/wan")
        self.assertEqual(config["workers"]["image0"]["config"]["model_path"], "/shared/data/DITSERVE/models/sd3.5-medium")
        self.assertEqual(config["workers"]["wan0"]["config"]["options"]["cfg_fold"]["folded_steps"], [2])
        self.assertNotIn("48403", str(config))
        self.assertNotIn("Sana", str(config))

    def test_forms_cover_every_supported_native_world_size(self):
        for count, form in ((2, "cfg2"), (4, "cfg2sp2"), (8, "cfg2sp3")):
            with self.subTest(count=count):
                config = MODULE.build_config("wan22", [f"GPU-{i}" for i in range(count)])
                self.assertEqual(config["phases"][0]["options"]["forms"], [form])
                self.assertEqual(config["workers"]["wan0"]["config"]["model_path"], "/shared/data/DITSERVE/models/wan2.2-ti2v-5b")

    def test_native_calibration_has_disjoint_fit_and_held_out_forms(self):
        config = native_study.build_calibration_config("wan22", [f"GPU-{i}" for i in range(8)])
        phase = config["phases"][0]
        self.assertEqual(len(phase["matrix"]["/policy"]), 10)
        self.assertTrue(set(native_study.FIT_POLICIES).isdisjoint(native_study.HELD_OUT_POLICIES))
        self.assertEqual(set(native_study.FIT_POLICIES + native_study.HELD_OUT_POLICIES), set(native_study.CALIBRATION_POLICIES))
        self.assertTrue(all(item["spec"]["steps"] == 1 for item in phase["warmup"]))
        self.assertNotIn("cfg_fold", config["workers"]["wan0"]["config"]["options"])
        self.assertEqual(config["benchmark"]["shape"], {"frames": 49, "steps": 20, "image_steps": 28})
        self.assertEqual(phase["profile"]["provenance"], "synthetic")
        self.assertFalse(config["benchmark"]["performance_qualified"])

    def test_invalid_allocations_fail_before_a_job_is_created(self):
        with self.assertRaises(ValueError):
            MODULE.build_config("wan21", ["GPU-0"])
        with self.assertRaises(ValueError):
            MODULE.build_config("wan21", ["GPU-0", "GPU-0"])
        with self.assertRaises(ValueError):
            MODULE.build_config("unknown", ["GPU-0", "GPU-1"])


if __name__ == "__main__":
    unittest.main()
