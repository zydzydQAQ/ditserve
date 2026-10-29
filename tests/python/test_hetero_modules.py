"""Unit tests for ditserve-hetero Python modules (Encoder Pool, Tiered State, Gamma Workloads, Hetero Policies)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ditserve.evaluation.workloads import build_trace
from ditserve.models.encoder_pool import ProactiveEncoderPool
from ditserve.models.wan.tiered_state import PinnedSlabPool

try:
    import torch as _real_torch
except ImportError:
    _real_torch = None


class _FakeSlab:
    def __init__(self, nbytes: int, pinned: bool):
        self._nbytes = nbytes
        self._pinned = pinned

    def numel(self) -> int:
        return self._nbytes

    def element_size(self) -> int:
        return 1

    def is_pinned(self) -> bool:
        return self._pinned


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True


class _FakeTorch:
    uint8 = "uint8"
    cuda = _FakeCuda()

    @staticmethod
    def empty(nbytes: int, dtype=None, device: str = "cpu", pin_memory: bool = False):
        return _FakeSlab(int(nbytes), bool(pin_memory))


class HeteroModulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="ditserve-hetero-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_proactive_encoder_pool_keys_and_lru(self) -> None:
        pool = ProactiveEncoderPool({"shm_dir": str(self.tmpdir / "shm"), "max_entries": 2, "max_bytes": 1024})
        k1 = pool.prompt_key("image", "sd35", "a mountain lake", negative_prompt="blurry")
        k2 = pool.prompt_key("image", "sd35", "a mountain lake", negative_prompt="blurry")
        k3 = pool.prompt_key("video", "wan21", "a mountain lake", negative_prompt="blurry")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertEqual(pool.stats()["entries"], 0)
        pool.clear()

    def test_pinned_slab_pool_reuse_and_trim(self) -> None:
        torch_mod = _real_torch if _real_torch is not None else _FakeTorch()
        pool = PinnedSlabPool(torch_mod, max_bytes=1024)
        s1 = pool.acquire(256, pin_memory=True)
        self.assertEqual(pool.stats()["allocations"], 1)
        self.assertEqual(pool.stats()["active_bytes"], 256)
        pool.release(s1)
        self.assertEqual(pool.stats()["cached_free_bytes"], 256)
        s2 = pool.acquire(256, pin_memory=True)
        self.assertIs(s1, s2)
        self.assertEqual(pool.stats()["reuses"], 1)
        pool.release(s2)
        # Acquiring 896 bytes trims the 256-byte cached slab to fit inside 1024 B budget
        s3 = pool.acquire(896, pin_memory=True)
        self.assertIsNotNone(s3)
        self.assertLessEqual(pool.stats()["allocated_bytes"], 1024)

    def test_gamma_workloads_and_dual_tier_slos(self) -> None:
        classes = {
            "wan21": {
                "class": "wan21",
                "kind": "video",
                "service_ms": 4000.0,
                "gpu_service_ms": 4000.0,
                "slo_scale": 3.0,
                "spec": {"frames": 33, "steps": 20},
            },
            "sd35": {
                "class": "sd35",
                "kind": "image",
                "service_ms": 120.0,
                "gpu_service_ms": 120.0,
                "slo_scale": 3.0,
                "spec": {"height": 1024, "width": 1024, "steps": 20},
            },
        }
        pools = {
            "video": {"kind": "video", "source": "test", "selection": "fixed", "samples": [{"sample_id": f"v-{i}", "prompt": f"v prompt {i}"} for i in range(40)]},
            "image": {"kind": "image", "source": "test", "selection": "fixed", "samples": [{"sample_id": f"i-{i}", "prompt": f"i prompt {i}"} for i in range(40)]},
        }
        windows = [
            {
                "count": 20,
                "video_fraction": 0.5,
                "pattern": "gamma",
                "cv": 2.5,
                "load": 0.85,
                "slo_scale": 3.0,
                "fast_video_fraction": 0.4,
                "fast_slo_scale": 1.2,
            }
        ]
        t1 = build_trace(classes, pools, windows, slots=8, seed=11)
        t2 = build_trace(classes, pools, windows, slots=8, seed=11)
        self.assertEqual(t1["manifest"]["workload_sha256"], t2["manifest"]["workload_sha256"])
        video_rows = [r for r in t1["workload"] if r["kind"] == "video"]
        tiers = {r.get("tier") for r in video_rows}
        self.assertEqual(tiers, {"fast", "standard"})
        fast_slos = {r["slo_ms"] for r in video_rows if r["tier"] == "fast"}
        std_slos = {r["slo_ms"] for r in video_rows if r["tier"] == "standard"}
        self.assertEqual(fast_slos, {4800.0})
        self.assertEqual(std_slos, {12000.0})

    def test_native_cli_heterodit_policies(self) -> None:
        which_bin = shutil.which("ditserve")
        binary = Path(which_bin) if which_bin else Path(__file__).resolve().parents[2] / ".cache" / "build" / "ditserve"
        if not binary.is_file():
            self.skipTest("native binary not built")
        config = {
            "gpus": 8,
            "policy": "heterodit_approx",
            "profile": {
                "provenance": "synthetic",
                "classes": {
                    "video": {
                        "kind": "video",
                        "steps": 4,
                        "blocks_per_step": 10,
                        "prepare_ms": 0,
                        "encode_ms": 0,
                        "step_compute_ms": 1000,
                        "comm_mb_per_step": 800,
                        "collectives_per_step": 10,
                        "cfg_exchange_mb": 20,
                        "finish_ms": 100,
                    },
                    "image": {
                        "kind": "image",
                        "service_ms": 50,
                        "batch_ms": {"2": 70, "4": 110},
                    },
                },
                "links": {
                    "2": {"gbps": 20, "latency_ms": 0.1},
                    "4": {"gbps": 10, "latency_ms": 0.1},
                    "8": {"gbps": 5, "latency_ms": 0.1},
                },
                "sp_efficiency": 1.0,
                "resize_ms": 10.0,
            },
            "workload": [
                {"request_id": "v1", "kind": "video", "class": "video", "arrival_ms": 0, "deadline_ms": 1500},
                {"request_id": "v2", "kind": "video", "class": "video", "arrival_ms": 10, "deadline_ms": 2200},
                {"request_id": "i1", "kind": "image", "class": "image", "arrival_ms": 100, "deadline_ms": 280},
                {"request_id": "i2", "kind": "image", "class": "image", "arrival_ms": 105, "deadline_ms": 290},
            ],
        }
        proc = subprocess.run(
            [str(binary), "rpc", "simulate"],
            input=json.dumps(config),
            capture_output=True,
            text=True,
            check=True,
        )
        out = json.loads(proc.stdout)
        self.assertEqual(out["metrics"]["completed"], 4)
        self.assertEqual(out["by_kind"]["image"]["ontime"], 2)

    def test_2d_hybrid_usp_fixed_layouts_and_strided_ring_groups(self) -> None:
        from ditserve.models.wan.parallel import FixedLayout, LayoutHandle, ParallelContractError, strided_ring_groups

        rings_4 = strided_ring_groups(4)
        self.assertIn((0, 2), rings_4)
        self.assertIn((1, 3), rings_4)

        rings_8 = strided_ring_groups(8)
        for pair in ((0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
            self.assertIn(pair, rings_8)

        layout_u2r2 = FixedLayout((0, 1, 2, 3), epoch=1, cfg=1, ring=2)
        self.assertEqual(layout_u2r2.sp_degree, 4)
        self.assertEqual(layout_u2r2.ring, 2)

        layout_u4r2 = FixedLayout((0, 1, 2, 3, 4, 5, 6, 7), epoch=2, cfg=1, ring=2)
        self.assertEqual(layout_u4r2.sp_degree, 8)
        self.assertEqual(layout_u4r2.ring, 2)

        layout_cfg2u2r2 = FixedLayout((0, 1, 2, 3, 4, 5, 6, 7), epoch=3, cfg=2, ring=2)
        self.assertEqual(layout_cfg2u2r2.sp_degree, 4)
        self.assertEqual(layout_cfg2u2r2.ring, 2)

        with self.assertRaises(ParallelContractError):
            FixedLayout((0, 1), epoch=1, cfg=1, ring=2)

        handle = LayoutHandle(0, (0, 1, 2, 3), cfg=1, ring=2)
        self.assertEqual(handle.current.ring, 2)

    def test_asymmetric_ring_int8_quantizer_roundtrip(self) -> None:
        if _real_torch is None:
            self.skipTest("torch required for tensor roundtrip test")
        from ditserve.models.wan.ring_quant import AsymmetricRingQuantizer

        q = AsymmetricRingQuantizer(_real_torch, warmup_steps=2, tail_anchor_steps=2, enabled=True)
        self.assertFalse(q.should_quantize(0, 20))
        self.assertFalse(q.should_quantize(1, 20))
        self.assertTrue(q.should_quantize(2, 20))
        self.assertTrue(q.should_quantize(17, 20))
        self.assertFalse(q.should_quantize(18, 20))

        _real_torch.manual_seed(42)
        kv = _real_torch.randn(2, 1, 3, 256, 128, dtype=_real_torch.float16)
        packed = q.pack_packet(kv)
        self.assertEqual(packed.dtype, _real_torch.uint8)
        self.assertEqual(tuple(packed.shape), (2, 1, 3, 256, 130))
        recon = q.unpack_packet(packed, target_dtype=_real_torch.float16)
        self.assertEqual(tuple(recon.shape), tuple(kv.shape))
        cos = float(
            _real_torch.nn.functional.cosine_similarity(
                kv.flatten().float().unsqueeze(0),
                recon.flatten().float().unsqueeze(0),
            ).item()
        )
        self.assertGreater(cos, 0.9999)
        summary = q.summary()
        self.assertAlmostEqual(summary["compression_ratio"], 256.0 / 130.0, places=3)

    def test_taylor_residual_compensator_lifecycle(self) -> None:
        if _real_torch is None:
            self.skipTest("torch required for Taylor compensator test")
        from types import SimpleNamespace
        from ditserve.models.wan.taylor_cache import TaylorResidualCompensator

        comp = TaylorResidualCompensator(_real_torch, alpha=0.30, max_norm_ratio=0.12, enabled=True)
        b0 = _real_torch.ones(1, 16, 64, dtype=_real_torch.float16) * 2.0
        b1 = _real_torch.ones(1, 16, 64, dtype=_real_torch.float16) * 2.4
        ctx = SimpleNamespace(
            buffers={
                "blocks_Bn_residual_buffer": b0.clone(),
                "blocks_Bn_residual_buffer_cfg": b0.clone(),
            }
        )
        state = SimpleNamespace(cache_enabled=True, step_index=0, contexts={"default": ctx})

        # Step 0: computed anchor
        comp.pre_step(state)
        state.step_index = 1
        comp.post_step(state, {"step_index": 0, "cache_contexts": [{"cond_reused": False, "uncond_reused": False}]})
        self.assertEqual(comp.stats["anchor_updates"], 2)

        # Step 1: second computed anchor -> velocity established
        ctx.buffers["blocks_Bn_residual_buffer"] = b1.clone()
        ctx.buffers["blocks_Bn_residual_buffer_cfg"] = b1.clone()
        comp.pre_step(state)
        state.step_index = 2
        comp.post_step(state, {"step_index": 1, "cache_contexts": [{"cond_reused": False, "uncond_reused": False}]})
        self.assertEqual(comp.stats["anchor_updates"], 4)

        # Step 2: cached step -> pre_step extrapolates above 2.4, post_step restores clean 2.4 anchor
        comp.pre_step(state)
        extrap_val = float(ctx.buffers["blocks_Bn_residual_buffer"].mean().item())
        self.assertGreater(extrap_val, 2.405)
        state.step_index = 3
        rec = comp.post_step(state, {"step_index": 2, "cache_contexts": [{"cond_reused": True, "uncond_reused": True}]})
        self.assertEqual(rec["taylor_compensated_forwards"], 2)
        restored_val = float(ctx.buffers["blocks_Bn_residual_buffer"].mean().item())
        self.assertAlmostEqual(restored_val, float(b1.mean().item()), places=3)


if __name__ == "__main__":
    unittest.main()
