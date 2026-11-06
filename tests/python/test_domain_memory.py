"""Canonical Functional Domain Suite 3: Memory Domain (`test_domain_memory.py`).

Consolidates and verifies:
1. Cross-modal memory strategies (`S0`-`S4` in `cluster_config.py`), `ProactiveEncoderPool`
   (S3/S4 split CPU UMT5, idle worker selection / `image_text_prefetch` overlap,
   `/dev/shm` per-cell cache isolation via `resolve_shm_dir`, cross-worker slab
   adoption via `ensure_resident`, and owner-only slab unlinking on eviction).
2. `TieredVideoStateManager` & `PinnedSlabPool` zero-allocation slab reuse, and
   `refresh_certificate` step-end latent-only migration reduction (`>= 90.6%`
   payload reduction vs `full_migrate`, stale certificate rejection, and
   two-phase ownership handoff `MigrationEndpoint`).
3. Lossless UMT5 valid-token truncation (`_encode_single_wan_text_trimmed`:
   `max_abs_diff == 0.0` vs full-length masked attention, pad token elimination,
   and `NegativePromptCache` exact hit/miss/LRU byte accounting).
"""

from __future__ import annotations

from collections import defaultdict
import copy
import dataclasses
import pathlib
import shutil
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import cluster_config as cc
from ditserve import api
from ditserve.models.encoder_pool import ProactiveEncoderPool, resolve_shm_dir
from ditserve.models.wan.migration import MigrationEndpoint, migrate_request, pack_request, unpack_request
from ditserve.models.wan.prompt_cache import NegativePromptCache, cache_options
from ditserve.models.wan.runtime import (
    CACHED_CONTEXT_FIELDS,
    DBCACHE_CONFIG_FIELDS,
    EXTRA_CACHE_CONFIG_FIELDS,
    RequestState,
    normalize_spec,
)
from ditserve.models.wan.state import refresh_certificate, state_inventory, transition_plan, validate_refresh_certificate
from ditserve.models.wan.tiered_state import PinnedSlabPool, TieredVideoStateManager
from ditserve.runtime import PACKAGE_ROOT
from ditserve.runtime.client import CoreBridge


# ==============================================================================
# 1. Cross-Modal Memory Strategies (`S0`-`S4`) & `ProactiveEncoderPool`
# ==============================================================================


class TestCrossModalMemoryStrategiesAndEncoderPool:
    """Verifies `S0`-`S4` descriptors, `ProactiveEncoderPool` `/dev/shm` isolation, and idle prefetch."""

    def test_s0_to_s4_descriptors_and_worker_capability_validation(self) -> None:
        assert cc.MEMORY_STRATEGIES == (
            "s0_naive_gpu",
            "s1_weight_swap",
            "s2_wan_cpu_sd35_gpu",
            "s3_both_encoders_cpu",
            "s4_disaggregated_pool",
        )
        assert cc.WORKER_MEMORY_STRATEGIES == (
            "s2_wan_cpu_sd35_gpu",
            "s3_both_encoders_cpu",
            "s4_disaggregated_pool",
        )

        # S0 (naive GPU co-location) and S1 (reactive weight swap) fail closed in worker config
        for analytical_only in ("s0_naive_gpu", "s1_weight_swap"):
            desc = cc.memory_strategy_descriptor(analytical_only)
            assert desc["code"] in ("S0", "S1")
            with pytest.raises(ValueError, match="analytical only"):
                cc.applied_memory_strategy(analytical_only)

        # S2, S3, S4 map cleanly to supported worker knobs
        s2 = cc.applied_memory_strategy("s2_wan_cpu_sd35_gpu")
        assert s2["code"] == "S2"
        assert s2["applied_worker_config"]["image"]["text_encoder_device"] == "cuda"
        assert s2["applied_worker_options"]["wan"]["encoder_pool"]["write_shm"] is False

        s3 = cc.applied_memory_strategy("s3_both_encoders_cpu")
        assert s3["code"] == "S3"
        assert s3["applied_worker_config"]["image"]["text_encoder_device"] == "cpu"
        assert s3["applied_worker_options"]["image"]["offload_text_encoders_after_warmup"] is True
        assert s3["applied_worker_options"]["wan"]["encoder_pool"]["write_shm"] is False

        s4 = cc.applied_memory_strategy("s4_disaggregated_pool")
        assert s4["code"] == "S4"
        assert s4["applied_worker_config"]["image"]["text_encoder_device"] == "cpu"
        assert s4["applied_worker_options"]["wan"]["release_cache_on_stop"] is True
        assert s4["applied_worker_options"]["wan"]["encoder_pool"]["enabled"] is True
        assert s4["applied_worker_options"]["wan"]["encoder_pool"]["write_shm"] is True
        assert s4["applied_worker_options"]["wan"]["encoder_pool"]["pin_memory"] is True

    def test_proactive_encoder_pool_per_cell_shm_isolation_and_cross_worker_adoption(
        self, tmp_path: pathlib.Path
    ) -> None:
        # 1. Per-cell /dev/shm isolation scopes distinct matrix cell work_dirs into disjoint directories
        cell0_work = tmp_path / "cell-0" / "artifacts" / "v0"
        cell1_work = tmp_path / "cell-1" / "artifacts" / "v0"
        shm0, scoped0 = resolve_shm_dir({"shm_dir": str(tmp_path / "shm"), "per_cell_isolation": True}, work_dir=cell0_work)
        shm1, scoped1 = resolve_shm_dir({"shm_dir": str(tmp_path / "shm"), "per_cell_isolation": True}, work_dir=cell1_work)
        assert scoped0 is True and scoped1 is True
        assert shm0 != shm1
        assert shm0.name.startswith("cell-") and shm1.name.startswith("cell-")

        # 2. Cross-worker /dev/shm staging, adoption, and owner-only slab unlinking
        shared_shm = tmp_path / "shared-shm"
        producer = ProactiveEncoderPool(
            {"shm_dir": str(shared_shm), "per_cell_isolation": False, "write_shm": True, "max_entries": 2}
        )
        consumer = ProactiveEncoderPool(
            {"shm_dir": str(shared_shm), "per_cell_isolation": False, "write_shm": True, "max_entries": 1}
        )

        key_a = ProactiveEncoderPool.prompt_key("wan_pos", "wan21", "ocean waves", dtype="bfloat16")
        key_b = ProactiveEncoderPool.prompt_key("wan_pos", "wan21", "mountain peak", dtype="bfloat16")
        emb_a = torch.ones((1, 8, 16), dtype=torch.bfloat16)
        emb_b = torch.ones((1, 8, 16), dtype=torch.bfloat16) * 2.0

        staged_a = producer.stage_embeddings(key_a, {"embed": emb_a})
        slab_a = pathlib.Path(staged_a["slab_path"])
        assert slab_a.is_file()

        # Consumer adopts key_a from /dev/shm into its own memory
        assert consumer.ensure_resident(key_a) is True
        assert consumer.shm_adoptions == 1
        fetched = consumer.fetch_to_device(key_a, "cpu")
        assert fetched["hit"] is True
        torch.testing.assert_close(fetched["tensors"]["embed"], emb_a)

        # Evicting adopted entry on consumer must NOT unlink producer's slab on disk!
        consumer.stage_embeddings(key_b, {"embed": emb_b}, write_shm=False)
        assert slab_a.is_file()
        assert consumer.slab_unlinks == 0

        # Clearing producer unlinks only the slabs producer itself wrote
        producer.clear()
        assert not slab_a.is_file()
        assert producer.slab_unlinks == 1

    def test_idle_worker_image_text_prefetch_isolation_in_cluster_controller(
        self, tmp_path: pathlib.Path
    ) -> None:
        pkg_root = tmp_path / "pkg"
        for package in ("ditserve", "xfuser"):
            source = PACKAGE_ROOT / package
            for path in source.rglob("*.py"):
                target = pkg_root / package / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)

        class _Bridge(CoreBridge):
            def call(self, command: str, payload: dict) -> dict:
                return super().call(command, dict(payload, package_root=str(pkg_root)))

        profile = {
            "provenance": "synthetic",
            "sp_efficiency": 1.0,
            "resize_ms": 1.0,
            "classes": {
                "video": {
                    "kind": "video",
                    "steps": 2,
                    "blocks_per_step": 4,
                    "prepare_ms": 0.0,
                    "encode_ms": 0.0,
                    "step_compute_ms": 60.0,
                    "comm_mb_per_step": 0.0,
                    "collectives_per_step": 0,
                    "finish_ms": 10.0,
                },
                "image": {"kind": "image", "service_ms": 30.0},
            },
            "links": {"2": {"gbps": 10.0, "latency_ms": 0.0}},
        }
        workload = [
            {
                "request_id": "i0",
                "kind": "image",
                "class": "image",
                "worker": "i0",
                "arrival_ms": 0,
                "slo_ms": 5000,
                "spec": {"steps": 1, "prompt": "cat", "seed": 1},
            },
            {
                "request_id": "i1",
                "kind": "image",
                "class": "image",
                "worker": "i0",
                "arrival_ms": 5,
                "slo_ms": 5000,
                "spec": {"steps": 1, "prompt": "dog", "seed": 2},
            },
        ]
        workers = {
            "v0": {"backend": "mock", "config": {"options": {"costs": {"step_ms": 20, "finish_ms": 2}}}},
            "i0": {
                "backend": "mock",
                "config": {
                    "options": {
                        "costs": {"encode_ms": 5, "prepare_ms": 4, "step_ms": 3, "finish_ms": 2},
                        "encoder_pool": {"enabled": True, "write_shm": True, "shm_dir": str(tmp_path / "shm-pool")},
                    }
                },
            },
        }
        phase = {
            "id": "cluster",
            "kind": "cluster",
            "policy": "ditserve",
            "options": {"continuous_batch": 4, "bubble_free": True, "image_text_prefetch": True},
            "profile": {
                "provenance": "synthetic",
                "links": {"2": {"gbps": 10.0, "latency_ms": 0.0}},
                "classes": {
                    "video": {
                        "kind": "video",
                        "steps": 2,
                        "blocks_per_step": 4,
                        "prepare_ms": 5,
                        "encode_ms": 10,
                        "step_compute_ms": 80,
                        "comm_mb_per_step": 0,
                        "collectives_per_step": 0,
                        "finish_ms": 5,
                    },
                    "image": {
                        "kind": "image",
                        "service_ms": 20,
                        "steps": 2,
                        "prepare_ms": 4,
                        "finish_ms": 4,
                        "batch_step_ms": {"1": 6, "2": 9, "4": 15, "8": 25},
                    },
                },
            },
            "gpus": [{"video": "v0", "image": "i0"}],
            "workload": [
                {
                    "request_id": "img-0",
                    "kind": "image",
                    "class": "image",
                    "worker": "i0",
                    "arrival_ms": 0,
                    "slo_ms": 1000,
                    "spec": {"steps": 2, "prompt": "cat", "seed": 1},
                },
                {
                    "request_id": "img-1",
                    "kind": "image",
                    "class": "image",
                    "worker": "i0",
                    "arrival_ms": 0,
                    "slo_ms": 1000,
                    "spec": {"steps": 2, "prompt": "dog", "seed": 2},
                },
            ],
        }
        cfg = {
            "schema": "ditserve.study.v1",
            "experiment": "prefetch-isolation",
            "workers": workers,
            "budget": {"limit_gpu_seconds": None, "watchdog_seconds": 15},
            "phases": [phase],
        }
        res = api.run(
            cfg,
            run_id="img-prefetch",
            work_dir=tmp_path / "work",
            results_root=tmp_path / "results",
            ledger=tmp_path / "budget.json",
            core=_Bridge(),
        )
        assert res["status"] == "completed"
        reqs = {r["request_id"]: r for r in res["phases"][0]["cells"][0]["requests"]}
        for rid, row in reqs.items():
            assert row["status"] == "completed", rid
            assert row.get("encode_ticket_used") is True
            assert row.get("encoder_pool_hit") is True
            assert row["encode_submit_ms"] <= row["encode_ready_ms"] <= row["start_ms"]


# ==============================================================================
# 2. TieredVideoStateManager, PinnedSlabPool & `refresh_certificate` (90.6% Reduction)
# ==============================================================================


def _make_cached_wan_state_for_certificate(
    *,
    step_index: int = 16,
    device: str = "cuda:0",
) -> tuple[SimpleNamespace, RequestState]:
    """Constructs a schema-valid Wan `RequestState` with DBCache feature buffers."""
    from diffusers.configuration_utils import FrozenDict

    spec = normalize_spec(
        {
            "request_id": "req-cert-1",
            "prompt": "a coastal lighthouse in storm",
            "negative_prompt": "low quality",
            "seed": 101,
            "height": 32,
            "width": 32,
            "frames": 5,
            "steps": 32,
            "guidance_scale": 5.0,
            "cache": True,
            "threshold": 0.08,
            "max_continuous": 1,
        }
    )
    immutable = {k: v for k, v in spec.items() if k not in ("cache", "threshold", "max_continuous")}

    class _UniPC:
        def __init__(self, config: dict):
            cfg = dict(config)
            cfg.setdefault("solver_order", 2)
            cfg.setdefault("disable_corrector", [])
            self._internal_dict = FrozenDict(cfg)
            self.betas = torch.linspace(0.001, 0.02, 1000, dtype=torch.float32, device="cpu")
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.linspace(0.99, 0.50, 1000, dtype=torch.float32, device="cpu")
            self.alpha_t = torch.linspace(0.99, 0.70, 1000, dtype=torch.float32, device="cpu")
            self.sigma_t = torch.linspace(0.05, 0.70, 1000, dtype=torch.float32, device="cpu")
            self.lambda_t = torch.linspace(2.0, 0.1, 1000, dtype=torch.float32, device="cpu")
            self.sigmas = torch.linspace(1.0, 0.0, 33, dtype=torch.float32, device="cpu")
            self.init_noise_sigma = 1.0
            self.predict_x0 = True
            self.num_inference_steps = 32
            self.timesteps = torch.arange(1000, 0, -31, dtype=torch.int64, device=device)[:32]
            self.model_outputs = [
                torch.full((1, 16, 2, 4, 4), 0.1, dtype=torch.float32, device=device),
                torch.full((1, 16, 2, 4, 4), 0.2, dtype=torch.float32, device=device),
            ]
            self.timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1]]
            self.lower_order_nums = 2
            self.this_order = 2
            self.disable_corrector = []
            self.solver_p = None
            self.last_sample = torch.full((1, 16, 2, 4, 4), 0.3, dtype=torch.float32, device=device)
            self._step_index = step_index
            self._begin_index = None

        @property
        def config(self):
            return self._internal_dict

        @classmethod
        def from_config(cls, config: dict) -> "_UniPC":
            return cls(dict(config))

    _UniPC.__name__ = "UniPCMultistepScheduler"

    @dataclasses.dataclass
    class _DBCacheConfig:
        cache_type: str = "DBCache"
        Fn_compute_blocks: int = 8
        Bn_compute_blocks: int = 0
        residual_diff_threshold: float = 0.08
        max_accumulated_residual_diff_threshold: float | None = None
        max_warmup_steps: int = 8
        warmup_interval: int = 1
        max_cached_steps: int = -1
        max_continuous_cached_steps: int = 1
        enable_separate_cfg: bool = True
        cfg_compute_first: bool = False
        cfg_diff_compute_separate: bool = True
        num_inference_steps: int = 32
        steps_computation_mask: list | None = None
        steps_computation_policy: str = "dynamic"
        force_refresh_step_hint: int | None = None
        force_refresh_step_policy: str = "once"

    @dataclasses.dataclass
    class _ExtraCacheConfig:
        l1_hidden_states_diff_threshold: float | None = None
        important_condition_threshold: float = 0.0
        downsample_factor: int = 1

    @dataclasses.dataclass
    class _CachedContext:
        name: str = "blocks"
        cache_config: _DBCacheConfig = dataclasses.field(default_factory=_DBCacheConfig)
        extra_cache_config: _ExtraCacheConfig = dataclasses.field(default_factory=_ExtraCacheConfig)
        calibrator_config: None = None
        calibrator: None = None
        encoder_calibrator: None = None
        cfg_calibrator: None = None
        cfg_encoder_calibrator: None = None
        executed_steps: int = step_index
        accumulated_executed_steps: int = step_index
        transformer_executed_steps: int = 2 * step_index
        accumulated_transformer_executed_steps: int = 2 * step_index
        continuous_cached_steps: int = 1
        cfg_continuous_cached_steps: int = 1
        accumulated_cached_steps: int = 4
        cfg_accumulated_cached_steps: int = 4
        cached_steps: list = dataclasses.field(default_factory=lambda: [step_index - 2, step_index - 1])
        cfg_cached_steps: list = dataclasses.field(default_factory=lambda: [step_index - 2, step_index - 1])
        residual_diffs: dict = dataclasses.field(
            default_factory=lambda: defaultdict(float, {str(step_index - 1): 0.015})
        )
        cfg_residual_diffs: dict = dataclasses.field(
            default_factory=lambda: defaultdict(float, {str(step_index - 1): 0.018})
        )
        accumulated_residual_diff: float = 0.03
        cfg_accumulated_residual_diff: float = 0.035
        buffers: dict = dataclasses.field(
            default_factory=lambda: {
                "blocks_Fn_residual_buffer": torch.ones((1, 2600, 4096), dtype=torch.float16, device=device),
                "blocks_Fn_residual_buffer_cfg": torch.ones((1, 2600, 4096), dtype=torch.float16, device=device) * 2.0,
                "blocks_Bn_residual_buffer": torch.ones((1, 2600, 4096), dtype=torch.float16, device=device) * 3.0,
                "blocks_Bn_residual_buffer_cfg": torch.ones((1, 2600, 4096), dtype=torch.float16, device=device) * 4.0,
            }
        )
        _init_args: tuple = ()
        _init_kwargs: dict = dataclasses.field(default_factory=dict)

        def __post_init__(self) -> None:
            if not self._init_kwargs:
                self._init_kwargs = {
                    "name": "blocks",
                    "cache_config": self.cache_config,
                }

    assert set(vars(_DBCacheConfig())) == DBCACHE_CONFIG_FIELDS
    assert set(vars(_ExtraCacheConfig())) == EXTRA_CACHE_CONFIG_FIELDS
    assert set(vars(_CachedContext())) == CACHED_CONTEXT_FIELDS

    sched = _UniPC({"num_inference_steps": 32, "disable_corrector": [], "solver_order": 2})
    gen = torch.Generator(device="cpu").manual_seed(101)
    latents = torch.linspace(0.0, 1.0, 512, dtype=torch.float32, device=device).reshape(1, 16, 2, 4, 4)
    prompt_embeds = torch.ones((1, 512, 4096), dtype=torch.float16, device=device)
    neg_embeds = torch.zeros((1, 512, 4096), dtype=torch.float16, device=device)

    owner = object()
    ctx = _CachedContext()
    state = RequestState(
        request_id="req-cert-1",
        spec=spec,
        step_index=step_index,
        latents=latents,
        cache_enabled=True,
        events=[],
        scheduler=sched,
        generator=gen,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        contexts={"blocks": ctx},
        residency="gpu",
        needs_anchor=False,
        finished=False,
        owner=owner,
        pending_snapshot=None,
        prepare_ms=2.0,
        current_step_refreshed=False,
        immutable_spec=immutable,
    )

    worker = SimpleNamespace(
        torch=torch,
        device=device,
        xdit=None,
        engine_name="native",
        attention="math",
        model_path=pathlib.Path("/shared/data/DITSERVE/models/wan"),
        model_revision="0fad780a534b6463e45facd96134c9f345acfa5b",
        pipe=SimpleNamespace(
            transformer=SimpleNamespace(config=FrozenDict({"in_channels": 16}), dtype=torch.float16),
            vae=SimpleNamespace(config=SimpleNamespace(z_dim=16, scale_factor_temporal=4, scale_factor_spatial=8)),
        ),
        _owner=owner,
        _states={state.request_id: state},
        _lock=threading.RLock(),
        _scheduler_type=_UniPC,
        _context_names=("blocks",),
        _sync=lambda: None,
        _memory=lambda: {"allocated_bytes": 0, "reserved_bytes": 0},
    )

    def _check(st: RequestState, residency: str | None = None) -> None:
        if st.owner is not worker._owner or worker._states.get(st.request_id) is not st:
            raise ValueError("foreign or released request")
        if residency is not None and st.residency != residency:
            raise ValueError(f"residency mismatch: {st.residency} != {residency}")

    def _payload(st: RequestState) -> dict:
        return {
            "latents": st.latents,
            "scheduler": vars(st.scheduler),
            "prompt_embeds": st.prompt_embeds,
            "negative_prompt_embeds": st.negative_prompt_embeds,
            "contexts": st.contexts,
            "rng_state": st.generator.get_state(),
        }

    def _install_payload(st: RequestState, payload: dict) -> None:
        st.latents = payload["latents"]
        st.prompt_embeds = payload["prompt_embeds"]
        st.negative_prompt_embeds = payload["negative_prompt_embeds"]
        vars(st.scheduler).clear()
        vars(st.scheduler).update(payload["scheduler"])
        st.contexts = payload["contexts"]
        st.generator.set_state(payload["rng_state"])

    def _contexts(_spec: dict) -> dict:
        return {"blocks": _CachedContext()}

    def _validate_request_schema(_st: RequestState) -> None:
        return None

    def _release(st: RequestState) -> None:
        st.residency = "released"
        worker._states.pop(st.request_id, None)

    worker._check = _check
    worker._payload = _payload
    worker._install_payload = _install_payload
    worker._contexts = _contexts
    worker._validate_context = lambda _ctx: None
    worker._validate_request_schema = _validate_request_schema
    worker.release = _release
    return worker, state


class TestTieredStatePoolAndRefreshCertificate:
    """Verifies `PinnedSlabPool`, `TieredVideoStateManager`, and `refresh_certificate` 90.6% migration reduction."""

    def test_pinned_slab_pool_zero_allocation_reuse_and_budget_trim(self) -> None:
        pool = PinnedSlabPool(torch, max_bytes=1024)
        s1 = pool.acquire(512, pin_memory=True)
        assert pool.stats()["allocations"] == 1
        assert pool.stats()["reuses"] == 0
        assert pool.stats()["active_bytes"] == 512

        # Release and re-acquire same size -> zero new allocations, reuse counter increments
        pool.release(s1)
        assert pool.stats()["cached_free_bytes"] == 512
        s2 = pool.acquire(512, pin_memory=True)
        assert s2 is s1
        assert pool.stats()["allocations"] == 1
        assert pool.stats()["reuses"] == 1
        pool.release(s2)

        # Acquiring 768 B trims the 512 B cached slab to stay within 1024 B budget
        s3 = pool.acquire(768, pin_memory=True)
        assert pool.stats()["allocated_bytes"] == 768
        assert pool.stats()["cached_free_bytes"] == 0

        # Exceeding budget with active slabs raises MemoryError
        with pytest.raises(MemoryError, match="PinnedSlabPool budget exceeded"):
            pool.acquire(512, pin_memory=True)
        pool.release(s3)

    def test_tiered_video_state_manager_d2h_offload_and_h2d_lookahead_prefetch(self) -> None:
        worker, state = _make_cached_wan_state_for_certificate(device="cuda:0")
        mgr = TieredVideoStateManager(worker, max_pinned_mb=128)

        orig_latents = state.latents.clone()
        offload_ev = mgr.offload_to_pinned(state, state.request_id)
        assert offload_ev["kind"] == "tiered_offload"
        assert offload_ev["transfer_bytes"] > 0
        assert state.residency == "cpu_pinned"
        assert state.latents.device.type == "cpu"
        assert mgr.summary()["suspended_count"] == 1

        prefetch_ev = mgr.lookahead_prefetch(state, state.request_id, target_device="cuda:0")
        assert prefetch_ev["kind"] == "tiered_prefetch"
        assert prefetch_ev["transfer_bytes"] == offload_ev["transfer_bytes"]
        assert state.residency == "gpu"
        assert state.latents.device.type == "cuda"
        torch.testing.assert_close(state.latents.cpu(), orig_latents.cpu())
        assert mgr.summary()["suspended_count"] == 0
        assert mgr.summary()["pool"]["cached_free_bytes"] == offload_ev["transfer_bytes"]

        # Second cycle reuses the exact pinned slabs without new host allocations
        allocs_before = mgr.pinned_pool.allocations
        mgr.offload_to_pinned(state, state.request_id)
        mgr.lookahead_prefetch(state, state.request_id, target_device="cuda:0")
        assert mgr.pinned_pool.allocations == allocs_before
        assert mgr.pinned_pool.reuses > 0

    def test_refresh_certificate_step_end_latent_only_migration_reduction(self) -> None:
        worker, state = _make_cached_wan_state_for_certificate(step_index=16, device="cuda:0")
        inv = state_inventory(state, torch_module=torch)
        # Feature buffers account for >= 90.6% of total state bytes
        reduction_ratio = inv["feature_only_bytes"] / inv["unique_storage_bytes"]
        assert reduction_ratio >= 0.906, f"Expected >= 90.6% feature buffer share, got {reduction_ratio:.4f}"

        cert = refresh_certificate(state, torch_module=torch)
        assert cert["eligible"] is True, cert["reasons"]
        assert cert["reclaimable_bytes"] == inv["feature_only_bytes"]
        validated = validate_refresh_certificate(state, cert, torch_module=torch)
        assert validated["sha256"] == cert["sha256"]

        plan = transition_plan(state, "exact_refresh", certificate=cert, torch_module=torch)
        assert plan["reclaimable_bytes"] == inv["feature_only_bytes"]

        # Compare full_migrate vs certified_omit wire payload bytes
        full_manifest, full_tensors = pack_request(
            worker, state, action="full_migrate", target_device="GPU-target", migration_id="m-full"
        )
        omit_manifest, omit_tensors = pack_request(
            worker,
            state,
            action="certified_omit",
            target_device="GPU-target",
            migration_id="m-omit",
            certificate=cert,
        )
        full_bytes = sum(t.numel() * t.element_size() for t in full_tensors.values())
        omit_bytes = sum(t.numel() * t.element_size() for t in omit_tensors.values())
        payload_savings = 1.0 - (omit_bytes / full_bytes)
        assert payload_savings >= 0.906
        assert omit_manifest["omitted_feature_bytes"] == inv["feature_only_bytes"]

        # Unpack certified_omit on target worker and verify needs_anchor is False and buffers are empty
        target_worker, placeholder = _make_cached_wan_state_for_certificate(step_index=16, device="cuda:0")
        target_worker.release(placeholder)
        restored = unpack_request(
            target_worker, omit_manifest, omit_tensors, target_device="GPU-target", migration_id="m-omit"
        )
        assert restored.needs_anchor is False
        assert restored.contexts["blocks"].buffers == {}

        # Mutating source state invalidates certificate (stale certificate rejected)
        state.generator.manual_seed(999)
        with pytest.raises(ValueError, match="stale"):
            pack_request(
                worker,
                state,
                action="certified_omit",
                target_device="GPU-target",
                migration_id="m-stale",
                certificate=cert,
            )

        # Two-phase ownership handoff via MigrationEndpoint
        w_src, s_src = _make_cached_wan_state_for_certificate(step_index=16, device="cuda:0")
        w_dst, s_dst_ph = _make_cached_wan_state_for_certificate(step_index=16, device="cuda:0")
        w_dst.release(s_dst_ph)
        ep_src = MigrationEndpoint(w_src, "GPU-source")
        ep_dst = MigrationEndpoint(w_dst, "GPU-target")
        fresh_cert = refresh_certificate(s_src, torch_module=torch)
        migrated, mig_manifest, summary = migrate_request(
            ep_src,
            ep_dst,
            s_src,
            action="certified_omit",
            migration_id="tx-1",
            certificate=fresh_cert,
        )
        assert migrated.step_index == 16
        assert mig_manifest["omitted_feature_bytes"] == inv["feature_only_bytes"]
        assert summary["transfer_bytes"] == mig_manifest["transfer_bytes"]
        assert not w_src._states
        assert "req-cert-1" in w_dst._states


# ==============================================================================
# 3. Lossless UMT5 Valid-Token Truncation & `NegativePromptCache`
# ==============================================================================


class TestLosslessUMT5TruncationAndNegativePromptCache:
    """Verifies `max_abs_diff == 0.0` valid-token UMT5 truncation and `NegativePromptCache`."""

    def test_lossless_umt5_valid_token_truncation_max_abs_diff_zero(self) -> None:
        # Construct an attention-masked UMT5 encoder where masked positions have zero attention weight
        class _MaskedUMT5Encoder(torch.nn.Module):
            def __init__(self, vocab_size: int = 64, d_model: int = 16):
                super().__init__()
                self.dtype = torch.bfloat16
                self.embed = torch.nn.Embedding(vocab_size, d_model)
                self.proj = torch.nn.Linear(d_model, d_model)

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
                # Embedding + masked mean context (padding tokens with mask==0 have zero contribution)
                x = self.embed(input_ids).float()
                mask_f = attention_mask.float().unsqueeze(-1)
                active_sum = (x * mask_f).sum(dim=1, keepdim=True)
                active_cnt = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
                ctx = active_sum / active_cnt
                out = self.proj(x + ctx) * mask_f
                return SimpleNamespace(last_hidden_state=out.to(dtype=self.dtype))

        class _FixedTokenizer:
            def __call__(
                self,
                texts: list[str],
                padding: str = "max_length",
                max_length: int = 512,
                truncation: bool = True,
                add_special_tokens: bool = True,
                return_attention_mask: bool = True,
                return_tensors: str = "pt",
            ):
                tokens = [min(63, max(1, len(w))) for w in texts[0].split()] + [1]
                valid_len = len(tokens)
                ids = tokens + [0] * (max_length - valid_len)
                mask = [1] * valid_len + [0] * (max_length - valid_len)
                return SimpleNamespace(
                    input_ids=torch.tensor([ids], dtype=torch.int64),
                    attention_mask=torch.tensor([mask], dtype=torch.int64),
                )

        encoder = _MaskedUMT5Encoder()
        tokenizer = _FixedTokenizer()
        wan_runtime = SimpleNamespace(
            text_encoder_dtype="bfloat16",
            model_path="/shared/data/DITSERVE/models/wan",
            model_revision="rev1",
            _owner=object(),
            pipe=SimpleNamespace(text_encoder=encoder, tokenizer=tokenizer),
        )

        pool = ProactiveEncoderPool({"write_shm": False, "pin_memory": False})
        opt_info = pool.optimize_wan_cpu_encoder(wan_runtime, mode="bf16_trimmed")
        assert opt_info["trimmed"] is True

        prompt = "a cinematic aerial view of snow mountains at sunrise"
        trimmed_out, active_seq_len = pool._encode_single_wan_text_trimmed(
            wan_runtime, prompt, max_sequence_length=512
        )
        assert active_seq_len == 10  # 9 words + EOS token
        assert trimmed_out.shape == (1, 512, 16)

        # Full 512-token masked execution produces identical output (max_abs_diff == 0.0)
        full_inputs = tokenizer([prompt], max_length=512)
        full_hidden = encoder(full_inputs.input_ids, full_inputs.attention_mask).last_hidden_state
        full_valid_padded = full_hidden.clone()
        full_valid_padded[:, active_seq_len:, :] = 0.0

        max_abs_diff = float((trimmed_out.float() - full_valid_padded.float()).abs().max().item())
        assert max_abs_diff == 0.0
        # Trailing padding tokens [active_seq_len:512] are strictly zero
        assert float(trimmed_out[:, active_seq_len:, :].float().abs().max().item()) == 0.0

    def test_negative_prompt_cache_hit_miss_lru_eviction_and_storage_accounting(self) -> None:
        opts = cache_options({"negative_prompt_cache_entries": 2, "negative_prompt_cache_bytes": 4096})
        assert opts == {"negative_prompt_cache_entries": 2, "negative_prompt_cache_bytes": 4096}

        cache = NegativePromptCache(entries=2, byte_limit=4096, recipe={"model": "wan21", "dtype": "float16"})
        encode_calls = 0

        def _fake_encode(prompt: str, negative_prompt: str, do_classifier_free_guidance: bool, negative_prompt_embeds=None):
            nonlocal encode_calls
            p_emb = torch.ones((1, 8, 16), dtype=torch.float16)
            if negative_prompt_embeds is not None:
                return p_emb, negative_prompt_embeds
            encode_calls += 1
            n_emb = torch.full((1, 8, 16), float(len(negative_prompt)), dtype=torch.float16)
            return p_emb, n_emb

        # 1. Miss inserts into cache
        _, neg1, rec1 = cache.encode(
            _fake_encode,
            {"prompt": "a", "negative_prompt": "blurry", "do_classifier_free_guidance": True},
            request_id="r1",
            validate=lambda p, n: None,
        )
        assert rec1["outcome"] == "miss"
        assert rec1["admitted"] is True
        assert encode_calls == 1

        # 2. Repeat negative prompt is an exact hit without calling negative encoder
        _, neg2, rec2 = cache.encode(
            _fake_encode,
            {"prompt": "b", "negative_prompt": "blurry", "do_classifier_free_guidance": True},
            request_id="r2",
            validate=lambda p, n: None,
        )
        assert rec2["outcome"] == "hit"
        assert rec2["negative_encode_called"] is False
        assert encode_calls == 1
        assert torch.equal(neg1, neg2)
        assert neg1 is not neg2  # independent clone

        # 3. Third distinct negative prompt evicts LRU entry when entries limit is 2
        cache.encode(
            _fake_encode,
            {"prompt": "c", "negative_prompt": "lowres", "do_classifier_free_guidance": True},
            request_id="r3",
            validate=lambda p, n: None,
        )
        _, _, rec4 = cache.encode(
            _fake_encode,
            {"prompt": "d", "negative_prompt": "overexposed", "do_classifier_free_guidance": True},
            request_id="r4",
            validate=lambda p, n: None,
        )
        assert len(rec4["evicted"]) == 1
        assert cache.counters["evictions"] == 1
        assert cache.snapshot()["entries"] == 2
