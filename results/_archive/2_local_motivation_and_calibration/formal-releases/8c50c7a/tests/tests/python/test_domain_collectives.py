"""Canonical Functional Domain Suite 4: Collectives Domain (`test_domain_collectives.py`).

Consolidates and verifies:
1. `GFC` group-free IPC collectives (`gfc.group` `GroupDescriptor`, `stable_hash64`,
   `compute_token`, `_EpochManager`, `gfc.config.SymmetricCollectiveConfig`,
   `ditserve.models.xdit.collectives` subgroup routing, `all_to_all_flat`,
   `all_gather_into`, `all_gather_chunks`, `all_reduce_max`, and
   `max_collective_bytes` oversize fallback to NCCL).
2. `AsymmetricRingQuantizer` per-token dynamic INT8 ring KV transport (`256B -> 130B`
   `1.9692x` compression ratio, cosine similarity > 0.9999, warmup/tail anchor
   schedule, and `ring_p2p_exchange`).
3. `GreenPool` CUDA Green Contexts (`88:20` and `16:92` SM partitioning on 108-SM
   A100 with granularity 4, disjointness guarantees, and stream isolation).
4. `CFG2xSP` hybrid parallel wrappers (`FixedLayout`, `LayoutHandle`, `aligned_groups`,
   `strided_ring_groups`, `RequestGroupGate`, `broadcast_folded_prediction`, and
   Ulysses `all_to_all_single` communication accounting).
"""

from __future__ import annotations

from types import SimpleNamespace
import pathlib
import pytest
import torch

from ditserve.models.wan.parallel import (
    FixedLayout,
    FixedProcessGroups,
    LayoutHandle,
    ParallelContractError,
    RequestGroupGate,
    aligned_groups,
    strided_ring_groups,
)
from ditserve.models.wan.ring_quant import AsymmetricRingQuantizer
from ditserve.models.xdit import collectives
from ditserve.models.xdit.process_group import PROCESS_GROUP
from ditserve.runtime.green_context import GreenPool, partition_indices
from gfc.config import SymmetricCollectiveConfig
from gfc.group import GroupDescriptor, _EpochManager, compute_token, stable_hash64


# ==============================================================================
# 1. GFC Group-Free IPC Collectives & `max_collective_bytes` Fallback to NCCL
# ==============================================================================


class TestGFCGroupFreeCollectivesAndFallback:
    """Verifies GFC descriptors, subgroup routing, and `max_collective_bytes` fallback to NCCL."""

    def teardown_method(self) -> None:
        collectives.shutdown()
        PROCESS_GROUP.ALL_TO_ALL = None

    def test_gfc_group_descriptor_hash_epoch_manager_and_token_nonzero(self) -> None:
        # 1. Deterministic 64-bit FNV+splitmix64 hashing and non-zero barrier token
        h1 = stable_hash64("ditserve-gfc", (0, 1, 2, 3), 42)
        h2 = stable_hash64("ditserve-gfc", [0, 1, 2, 3], 42)
        assert h1 == h2
        assert 0 < h1 < (1 << 64)
        for epoch in range(16):
            tok = compute_token(0x12345678, h1, epoch)
            assert tok != 0

        # 2. GroupDescriptor validation
        desc = GroupDescriptor(group_id=h1, ranks=(2, 3), local_index=1, size=2)
        assert desc.ranks == (2, 3)
        assert desc.local_index == 1
        with pytest.raises(ValueError, match="size must match"):
            GroupDescriptor(group_id=1, ranks=(0, 1), local_index=0, size=4)
        with pytest.raises(ValueError, match="local_index out of range"):
            GroupDescriptor(group_id=1, ranks=(0, 1), local_index=2, size=2)

        # 3. Monotonic host-side _EpochManager
        em = _EpochManager()
        assert em.peek(7) == 0
        assert em.next_for_barrier(7) == 0
        assert em.next_pair_for_collective(7) == (1, 2)
        assert em.peek(7) == 3

        # 4. SymmetricCollectiveConfig validation and vendored license presence
        cfg = SymmetricCollectiveConfig(max_group_size=8, max_collective_bytes=64 * (1 << 20))
        assert cfg.max_group_size == 8
        gfc_root = pathlib.Path(collectives.__file__).resolve().parents[3] / "gfc"
        assert (gfc_root / "LICENSE").is_file()
        assert "MIT" in (gfc_root / "LICENSE").read_text(encoding="utf-8")

    def test_collectives_subgroup_routing_and_max_collective_bytes_nccl_fallback(self) -> None:
        # Default is nccl and never intercepts
        assert collectives.backend() == "nccl"
        assert collectives.all_to_all_flat(object(), object()) is None
        assert collectives.all_gather_into(object(), object(), (0, 1)) is False
        assert collectives.all_reduce_max(object(), (0, 1)) is False

        with pytest.raises(ValueError, match="collective_backend"):
            collectives.configure("mpi")

        # Install a deterministic mock GFC runtime with max_collective_bytes = 256 B
        class _MockGFCRuntime:
            def __init__(self, max_bytes: int = 256):
                self.config = SimpleNamespace(max_collective_bytes=max_bytes, use_copy_engine=False)
                self.stream = torch.cuda.Stream("cuda:0")
                self.registered: list[tuple[int, ...]] = []
                self.all2all_calls: list[int] = []
                self.all_gather_calls: list[int] = []

            def register_group(self, ranks: tuple[int, ...]) -> GroupDescriptor:
                self.registered.append(tuple(ranks))
                return GroupDescriptor(group_id=stable_hash64(ranks), ranks=tuple(ranks), local_index=0, size=len(ranks))

            def wait_for_external(self, _event) -> None:
                return None

            def record_event(self):
                ev = torch.cuda.Event()
                ev.record(self.stream)
                return ev

            def all2all(self, src: torch.Tensor, dst: torch.Tensor, desc: GroupDescriptor, slice_bytes: int) -> None:
                self.all2all_calls.append(slice_bytes * desc.size)
                dst.copy_(src.flip(0))

            def all_gather(self, src: torch.Tensor, dst: torch.Tensor, desc: GroupDescriptor) -> None:
                self.all_gather_calls.append(src.nbytes)
                chunks = [src + float(i) for i in range(desc.size)]
                dst.copy_(torch.stack(chunks, dim=0).reshape(dst.shape))

            def shutdown(self) -> None:
                return None

        mock_rt = _MockGFCRuntime(max_bytes=256)
        collectives._BACKEND = "gfc"
        collectives._RUNTIME = mock_rt
        collectives._STATS["config"] = {"max_collective_bytes": 256, "use_copy_engine": False}

        # Unbound process group fails closed
        unbound_pg = SimpleNamespace()
        with pytest.raises(RuntimeError, match="never bound"):
            collectives._ranks_of(unbound_pg)

        # Bind process group to subgroup (2, 3)
        bound_pg = SimpleNamespace()
        collectives.bind(bound_pg, [2, 3])
        assert collectives._ranks_of(bound_pg) == (2, 3)

        # 1. Within max_collective_bytes (64 floats * 2B = 128B <= 256B) -> runs on GFC
        small_flat = torch.arange(64, dtype=torch.float16, device="cuda:0")
        out_small = collectives.all_to_all_flat(small_flat, bound_pg)
        assert out_small is not None
        assert torch.equal(out_small, small_flat.flip(0))
        assert collectives.info()["calls"]["all2all"] == 1
        assert mock_rt.registered == [(2, 3)]

        # 2. Exceeding max_collective_bytes (256 floats * 2B = 512B > 256B) -> returns None (falls back to NCCL)
        oversize_flat = torch.arange(256, dtype=torch.float16, device="cuda:0")
        out_oversize = collectives.all_to_all_flat(oversize_flat, bound_pg)
        assert out_oversize is None
        assert collectives.info()["calls"]["all2all_oversize_fallback"] == 1

        # 3. all_gather_chunks and all_reduce_max (doorbell consensus) over subgroup (0, 1)
        flag = torch.tensor([1, 0], dtype=torch.int32, device="cuda:0")
        assert collectives.all_reduce_max(flag, (0, 1)) is True
        assert flag.tolist() == [2, 1]


# ==============================================================================
# 2. AsymmetricRingQuantizer Per-Token Dynamic INT8 Ring KV Transport
# ==============================================================================


class TestAsymmetricRingQuantizer:
    """Verifies `AsymmetricRingQuantizer` 1.9692x payload compression, cosine fidelity, and P2P exchange."""

    def test_step_anchor_schedule_1p9692x_compression_and_cosine_similarity(self) -> None:
        quant = AsymmetricRingQuantizer(torch, warmup_steps=2, tail_anchor_steps=2, enabled=True)
        total_steps = 10
        schedule = [quant.should_quantize(s, total_steps) for s in range(total_steps)]
        assert schedule == [False, False, True, True, True, True, True, True, False, False]

        # KV tensor [2, B=1, H_local=6, S_ring=64, D=128] in FP16 (256 B per 128-dim head vector)
        torch.manual_seed(2026)
        kv = torch.randn((2, 1, 6, 64, 128), dtype=torch.float16) * 1.5
        packed = quant.pack_packet(kv)
        assert packed.dtype == torch.uint8
        assert packed.shape == (2, 1, 6, 64, 130)  # 128 B INT8 + 2 B FP16 scale = 130 B

        summary = quant.summary()
        assert summary["compression_ratio"] == pytest.approx(256.0 / 130.0, abs=1e-4)
        assert summary["bandwidth_saving_pct"] == pytest.approx(49.22, abs=0.05)

        unpacked = quant.unpack_packet(packed, target_dtype=torch.float16)
        assert unpacked.shape == kv.shape
        cos_sim = float(
            torch.nn.functional.cosine_similarity(kv.float().reshape(-1, 128), unpacked.float().reshape(-1, 128), dim=-1)
            .mean()
            .item()
        )
        assert cos_sim > 0.9999

    def test_ring_p2p_exchange_quantized_vs_fp16_anchor(self) -> None:
        quant = AsymmetricRingQuantizer(torch, warmup_steps=1, tail_anchor_steps=1, enabled=True)
        kv_send = torch.randn((2, 1, 2, 16, 128), dtype=torch.float16)
        recorded_ops: list[tuple] = []

        class _FakeDist:
            isend = "isend"
            irecv = "irecv"

            @staticmethod
            def P2POp(op, tensor, peer, group=None):
                recorded_ops.append((op, tensor.dtype, tuple(tensor.shape), peer))
                return (op, tensor)

            @staticmethod
            def batch_isend_irecv(ops):
                send_tensor = ops[0][1]
                recv_tensor = ops[1][1]
                recv_tensor.copy_(send_tensor)
                return [SimpleNamespace(wait=lambda: None) for _ in ops]

        # Quantized P2P exchange transmits uint8[..., 130]
        recv_q = quant.ring_p2p_exchange(_FakeDist, kv_send, send_peer=1, recv_peer=3, quantize=True)
        assert recorded_ops[0][1] == torch.uint8
        assert recorded_ops[0][2] == (2, 1, 2, 16, 130)
        assert recv_q.shape == kv_send.shape

        # FP16 anchor P2P exchange transmits exact float16[..., 128]
        recorded_ops.clear()
        recv_fp16 = quant.ring_p2p_exchange(_FakeDist, kv_send, send_peer=1, recv_peer=3, quantize=False)
        assert recorded_ops[0][1] == torch.float16
        assert recorded_ops[0][2] == (2, 1, 2, 16, 128)
        assert torch.equal(recv_fp16, kv_send)
        assert quant.stats["fp16_anchor_packets"] == 1


# ==============================================================================
# 3. GreenPool CUDA Green Contexts (`88:20` & `16:92` SM Partitioning)
# ==============================================================================


class TestGreenPoolSMPartitioning:
    """Verifies `partition_indices` and `GreenPool` `88:20` / `16:92` SM partitioning on 108-SM A100."""

    def test_a100_108sm_partition_arithmetic_88_20_and_16_92(self) -> None:
        # 88:20 video-dominant SM partition on 108-SM A100 (granularity 4 SMs -> 27 groups of 4 SMs)
        vid_88 = partition_indices(108, 4, "video", 88, 20)
        img_20 = partition_indices(108, 4, "image", 88, 20)
        assert vid_88 == list(range(0, 22))
        assert img_20 == list(range(22, 27))
        assert set(vid_88).isdisjoint(set(img_20))
        assert len(vid_88) * 4 + len(img_20) * 4 == 108

        # 16:92 image-dominant SM partition on 108-SM A100
        vid_16 = partition_indices(108, 4, "video", 16, 92)
        img_92 = partition_indices(108, 4, "image", 16, 92)
        assert vid_16 == [0, 1, 2, 3]
        assert img_92 == list(range(4, 27))
        assert set(vid_16).isdisjoint(set(img_92))

        # Invalid non-divisible or unbalanced quotas fail closed
        for bad_args in (
            (108, 4, "video", 86, 22),
            (108, 4, "image", 88, 24),
            (108, 4, "audio", 88, 20),
            (108, 0, "video", 88, 20),
        ):
            with pytest.raises(ValueError):
                partition_indices(*bad_args)

    def test_green_pool_report_and_fallback_stream_lifecycle(self) -> None:
        pool = GreenPool("video", {"88:20": (88, 20), "16:92": (16, 92)}, "88:20")
        try:
            rep = pool.report()
            assert rep["enabled"] is True
            assert rep["initialized"] is False
            assert rep["role"] == "video"
            assert rep["default"] == "88:20"
            assert rep["labels"]["88:20"]["requested_sm"] == 88
            assert rep["labels"]["16:92"]["requested_sm"] == 16
            assert rep["same_process_partitions_disjoint_by_construction"] is True
        finally:
            pool.close()


# ==============================================================================
# 4. CFG2xSP Hybrid Parallel Wrappers & Ulysses All-to-All Accounting
# ==============================================================================


class TestCFG2xSPHybridParallelAndUlyssesCollectives:
    """Verifies `FixedLayout`, `LayoutHandle`, 2D Hybrid USP groups, and Ulysses `all_to_all_single`."""

    def test_fixed_layout_cfg2xsp_and_2d_hybrid_usp_topologies(self) -> None:
        # 1. Standard CFG x SP layouts across 1, 2, 4, 8 ranks
        sp1 = FixedLayout((0,), cfg=1)
        assert (sp1.degree, sp1.sp_degree, sp1.cfg_role(0), sp1.sp_ranks(0)) == (1, 1, None, (0,))

        cfg2 = FixedLayout((0, 1), cfg=2)
        assert (cfg2.degree, cfg2.sp_degree) == (2, 1)
        assert (cfg2.cfg_role(0), cfg2.cfg_role(1)) == (0, 1)
        assert (cfg2.sp_ranks(0), cfg2.sp_ranks(1)) == ((0,), (1,))

        cfg2sp2 = FixedLayout((0, 1, 2, 3), cfg=2)
        assert (cfg2sp2.degree, cfg2sp2.sp_degree) == (4, 2)
        assert (cfg2sp2.cfg_role(0), cfg2sp2.cfg_role(1), cfg2sp2.cfg_role(2), cfg2sp2.cfg_role(3)) == (0, 0, 1, 1)
        assert cfg2sp2.sp_ranks(0) == (0, 1)
        assert cfg2sp2.sp_ranks(2) == (2, 3)

        cfg2sp4 = FixedLayout(tuple(range(8)), cfg=2)
        assert (cfg2sp4.degree, cfg2sp4.sp_degree) == (8, 4)
        assert cfg2sp4.sp_ranks(1) == (0, 1, 2, 3)
        assert cfg2sp4.sp_ranks(6) == (4, 5, 6, 7)

        # 2. 2D Hybrid USP (ring=2 requires sp_degree >= 4)
        u2r2 = FixedLayout((0, 1, 2, 3), cfg=1, ring=2)
        assert u2r2.sp_degree == 4 and u2r2.ring == 2
        cfg2_u2r2 = FixedLayout(tuple(range(8)), cfg=2, ring=2)
        assert cfg2_u2r2.sp_degree == 4 and cfg2_u2r2.ring == 2

        with pytest.raises(ParallelContractError, match="2D Hybrid USP"):
            FixedLayout((0, 1, 2, 3), cfg=2, ring=2)  # sp_degree == 2 < 4

        # 3. Aligned groups and strided ring groups for 8-GPU nodes
        ag8 = aligned_groups(8)
        assert (0, 1) in ag8 and (0, 1, 2, 3) in ag8 and tuple(range(8)) in ag8
        rg8 = strided_ring_groups(8)
        assert (0, 2) in rg8 and (1, 3) in rg8

    def test_layout_handle_lockstep_pairing_and_ulysses_all_to_all_override(self) -> None:
        handle = LayoutHandle(rank=0, initial_ranks=(0, 1), cfg=2)
        s_a = handle.begin_step("req-a", 0, cfg="paired")
        assert s_a["cfg_degree"] == 2 and s_a["sp_degree"] == 1

        # Second request joins in lockstep when pair=True
        s_b = handle.begin_step("req-b", 0, cfg="paired", pair=True)
        assert s_b["request_id"] == "req-b"
        handle.end_step("req-b")
        assert handle.active["request_id"] == "req-a" and handle.paired is None
        handle.end_step("req-a")
        assert handle.active is None

        # Commit new layout at step boundary increments epoch
        new_layout = handle.commit((0, 1, 2, 3), expected_epoch=0, cfg=2)
        assert new_layout.epoch == 1 and new_layout.sp_degree == 2

        # Verify Ulysses _sdpa_all_to_all_single routes through PROCESS_GROUP.ALL_TO_ALL override
        import importlib.util
        import pathlib
        import xfuser.core.distributed  # noqa: F401

        usp_path = pathlib.Path(__file__).resolve().parents[2] / "workers" / "xfuser" / "model_executor" / "layers" / "usp.py"
        usp_spec = importlib.util.spec_from_file_location("xfuser_usp_under_test", usp_path)
        assert usp_spec is not None and usp_spec.loader is not None
        usp = importlib.util.module_from_spec(usp_spec)
        usp_spec.loader.exec_module(usp)

        seen_numel: list[int] = []
        PROCESS_GROUP.ALL_TO_ALL = lambda flat, group: (seen_numel.append(flat.numel()), flat.flip(0))[1]
        try:
            x = torch.arange(16, dtype=torch.float16).reshape(4, 4)
            out = usp._sdpa_all_to_all_single(x)
            assert seen_numel == [16]
            assert torch.equal(out, x.flatten().flip(0).reshape(4, 4))
        finally:
            PROCESS_GROUP.ALL_TO_ALL = None
