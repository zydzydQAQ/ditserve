"""Unit tests for S4 runnable memory strategy, capability table, and per-cell /dev/shm isolation (2026-11-05)."""

from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "benchmarks", ROOT / "workers", ROOT / "tests" / "python"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cluster_config  # noqa: E402
import cross_modal_memory_study as cmms  # noqa: E402
import native_serving  # noqa: E402
from test_native_serving import pool as serving_pool, profile as serving_profile  # noqa: E402
from ditserve.models.encoder_pool import DEFAULT_SHM_DIR, ProactiveEncoderPool, resolve_shm_dir  # noqa: E402


UUIDS = [f"GPU-{i}" for i in range(4)]


class S4RunnableConfigTests(unittest.TestCase):
    def test_worker_memory_strategies_and_s4_descriptor_truthfulness(self):
        self.assertEqual(
            cluster_config.WORKER_MEMORY_STRATEGIES,
            ("s2_wan_cpu_sd35_gpu", "s3_both_encoders_cpu", "s4_disaggregated_pool"),
        )
        s4 = cluster_config.memory_strategy_descriptor("s4_disaggregated_pool")
        self.assertEqual(s4["code"], "S4")
        self.assertEqual(s4["wan_text_encoder_device"], "cpu")
        self.assertEqual(s4["sd35_text_encoder_device"], "cpu")
        self.assertTrue(s4["offload_text_encoders_after_warmup"])
        self.assertFalse(s4["weight_swap_on_switch"])
        self.assertTrue(s4["release_cache_on_stop"])
        # D1 Option C: export-mode cluster configs do not run C++ pinned-SP1 tiered_memory.
        self.assertFalse(s4["tiered_memory"])
        self.assertTrue(s4["encoder_pool"]["enabled"])
        self.assertTrue(s4["encoder_pool"]["write_shm"])
        self.assertTrue(s4["encoder_pool"]["pin_memory"])

    def test_capability_table_fails_closed_on_unknown_or_unsupported_knobs(self):
        base = cluster_config.memory_strategy_descriptor("s4_disaggregated_pool")
        self.assertEqual(cluster_config._unimplemented_memory_knobs(base), [])

        # Unknown top-level descriptor key
        unknown_top = copy.deepcopy(base)
        unknown_top["mystery_knob"] = True
        missing_top = cluster_config._unimplemented_memory_knobs(unknown_top)
        self.assertTrue(any("unknown descriptor knob mystery_knob" in item for item in missing_top))

        # Unknown encoder_pool key
        unknown_pool = copy.deepcopy(base)
        unknown_pool["encoder_pool"]["mystery_pool_knob"] = 123
        missing_pool = cluster_config._unimplemented_memory_knobs(unknown_pool)
        self.assertTrue(any("unknown encoder_pool knob encoder_pool.mystery_pool_knob" in item for item in missing_pool))

        # Unsupported tiered_memory=True
        tiered = copy.deepcopy(base)
        tiered["tiered_memory"] = True
        self.assertIn("tiered_memory=True", cluster_config._unimplemented_memory_knobs(tiered))

        # Unsupported weight_swap_on_switch=True (S1)
        s1 = cluster_config.memory_strategy_descriptor("s1_weight_swap")
        self.assertIn("weight_swap_on_switch=True", cluster_config._unimplemented_memory_knobs(s1))

        # Unsupported wan_text_encoder_device="cuda" (S0)
        s0 = cluster_config.memory_strategy_descriptor("s0_naive_gpu")
        self.assertIn("wan_text_encoder_device='cuda'", cluster_config._unimplemented_memory_knobs(s0))

        # D2 Option (a): offload_text_encoders_after_warmup must match (sd35_text_encoder_device == "cpu")
        mismatch_cpu = copy.deepcopy(base)
        mismatch_cpu["offload_text_encoders_after_warmup"] = False
        self.assertTrue(
            any("offload_text_encoders_after_warmup" in item for item in cluster_config._unimplemented_memory_knobs(mismatch_cpu))
        )
        mismatch_cuda = cluster_config.memory_strategy_descriptor("s2_wan_cpu_sd35_gpu")
        mismatch_cuda["offload_text_encoders_after_warmup"] = True
        self.assertTrue(
            any("offload_text_encoders_after_warmup" in item for item in cluster_config._unimplemented_memory_knobs(mismatch_cuda))
        )

    def test_make_workers_places_text_encoder_device_in_config_not_options(self):
        for strategy, expected_dev in (
            ("s2_wan_cpu_sd35_gpu", "cuda"),
            ("s3_both_encoders_cpu", "cpu"),
            ("s4_disaggregated_pool", "cpu"),
        ):
            with self.subTest(strategy=strategy):
                _, workers, _, _ = cluster_config.make_workers(
                    "wan21", UUIDS, video_spec={}, image_spec={}, memory_strategy=strategy
                )
                for idx in range(len(UUIDS)):
                    img_cfg = workers[f"image{idx}"]["config"]
                    self.assertEqual(img_cfg["text_encoder_device"], expected_dev)
                    self.assertNotIn("text_encoder_device", img_cfg["options"])
                    self.assertEqual(
                        img_cfg["options"]["offload_text_encoders_after_warmup"],
                        expected_dev == "cpu",
                    )

    def test_build_config_and_native_serving_enable_image_text_prefetch_for_cpu_t5(self):
        pools = {"image": serving_pool("diffusiondb"), "video": serving_pool("vidprom")}
        for strategy in ("s3_both_encoders_cpu", "s4_disaggregated_pool"):
            with self.subTest(strategy=strategy):
                preflight = cluster_config.build_config("wan21", UUIDS, memory_strategy=strategy, continuous_batch=2)
                self.assertTrue(preflight["phases"][0]["options"]["image_text_prefetch"])
                self.assertFalse(preflight["phases"][0]["options"]["tiered_memory"])
                recorded = preflight["benchmark"]["memory_strategy"]
                self.assertEqual(recorded["not_applied"], [])
                self.assertEqual(recorded["applied_worker_config"]["image"]["text_encoder_device"], "cpu")
                flat = json.dumps(recorded)
                for forbidden in ("tiered_memory", "wan_text_encoder_device", "weight_swap_on_switch"):
                    self.assertNotIn(forbidden, flat)

                served = native_serving.build_serving_config(
                    "wan22",
                    [f"GPU-{i}" for i in range(8)],
                    serving_profile(),
                    pools,
                    seed=201,
                    load=1.0,
                    pattern="poisson",
                    video_fraction=0.2,
                    count=20,
                    policies=["ditserve"],
                    memory_strategy=strategy,
                )
                self.assertTrue(served["phases"][0]["options"]["image_text_prefetch"])
                self.assertEqual(served["workers"]["image0"]["config"]["text_encoder_device"], "cpu")
                self.assertNotIn("text_encoder_device", served["workers"]["image0"]["config"]["options"])


class PerCellShmIsolationTests(unittest.TestCase):
    def test_resolve_shm_dir_scopes_by_cell_and_shares_across_workers_in_same_cell(self):
        cell0_wan0 = "/shared/output/DITSERVE/run1/phases/00-cluster/cells/00-edf/artifacts/wan0"
        cell0_img3 = "/shared/output/DITSERVE/run1/phases/00-cluster/cells/00-edf/artifacts/image3"
        cell1_img0 = "/shared/output/DITSERVE/run1/phases/00-cluster/cells/01-ditserve/artifacts/image0"

        dir_c0_w0, scoped_c0_w0 = resolve_shm_dir({"shm_dir": DEFAULT_SHM_DIR}, work_dir=cell0_wan0)
        dir_c0_i3, scoped_c0_i3 = resolve_shm_dir({"shm_dir": "auto"}, work_dir=cell0_img3)
        dir_c1_i0, scoped_c1_i0 = resolve_shm_dir({}, work_dir=cell1_img0)

        self.assertTrue(scoped_c0_w0)
        self.assertTrue(scoped_c0_i3)
        self.assertTrue(scoped_c1_i0)
        # Same cell -> identical shm_dir so Wan/SD3.5 workers in cell 0 share slabs
        self.assertEqual(dir_c0_w0, dir_c0_i3)
        # Different cell -> distinct shm_dir so cell 0 slabs never leak into cell 1
        self.assertNotEqual(dir_c0_w0, dir_c1_i0)

        # Custom explicit shm_dir is preserved without scoping
        custom_dir, custom_scoped = resolve_shm_dir({"shm_dir": "/tmp/custom-shm-pool"}, work_dir=cell0_wan0)
        self.assertEqual(custom_dir, Path("/tmp/custom-shm-pool"))
        self.assertFalse(custom_scoped)

    def test_cross_cell_pools_do_not_leak_slabs_and_close_cleans_up(self):
        had_torch = "torch" in sys.modules
        torch_mod, _ = cmms._ensure_torch_or_shim()
        try:
            with tempfile.TemporaryDirectory() as root:
                base_shm = Path(root) / "shm"
                cell0_w0 = Path(root) / "cells" / "cell-0" / "artifacts" / "image0"
                cell0_w1 = Path(root) / "cells" / "cell-0" / "artifacts" / "image1"
                cell1_w0 = Path(root) / "cells" / "cell-1" / "artifacts" / "image0"

                with unittest.mock.patch("ditserve.models.encoder_pool.DEFAULT_SHM_DIR", str(base_shm)):
                    pool_c0_a = ProactiveEncoderPool(
                        {"shm_dir": "auto", "write_shm": True, "pin_memory": False},
                        work_dir=cell0_w0,
                    )
                    pool_c0_b = ProactiveEncoderPool(
                        {"shm_dir": "auto", "write_shm": True, "pin_memory": False},
                        work_dir=cell0_w1,
                    )
                    pool_c1 = ProactiveEncoderPool(
                        {"shm_dir": "auto", "write_shm": True, "pin_memory": False},
                        work_dir=cell1_w0,
                    )

                    key = ProactiveEncoderPool.prompt_key(
                        "t2i", "sd35", "a mountain lake", "blurry", 256,
                        text_encoder_device="cpu", text_encoder_dtype="bfloat16",
                    )
                    pool_c0_a.stage_embeddings(
                        key,
                        {"prompt_embeds": torch_mod.empty((1, 16, 32), dtype=torch_mod.bfloat16, device="cpu")},
                        pin_memory=False,
                        write_shm=True,
                    )
                    self.assertEqual(pool_c0_a.stats()["slab_writes"], 1)
                    # Peer worker in cell 0 adopts the slab
                    self.assertTrue(pool_c0_b.ensure_resident(key))
                    self.assertEqual(pool_c0_b.stats()["shm_adoptions"], 1)
                    # Worker in cell 1 is completely isolated and sees a miss
                    self.assertFalse(pool_c1.ensure_resident(key))

                    # Closing cell 0 pools removes owned slabs and empty cell directory
                    cell0_shm_dir = pool_c0_a.shm_dir
                    self.assertTrue(cell0_shm_dir.exists())
                    pool_c0_b.close()
                    pool_c0_a.close()
                    self.assertEqual(pool_c0_a.stats()["slab_unlinks"], 1)
                    self.assertFalse(cell0_shm_dir.exists())
        finally:
            if not had_torch and isinstance(sys.modules.get("torch"), cmms._CpuTorchModuleShim):
                sys.modules.pop("torch", None)


if __name__ == "__main__":
    unittest.main()
