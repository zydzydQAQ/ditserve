"""Canonical Functional Domain Suite 2: BlockExecutor Domain (`test_domain_block_executor.py`).

Consolidates and verifies:
1. `WanBlockExecutor` sub-step block state machine, `WanContinuationState` (`BlockState`)
   packet serialization & SHA256 digest verification (`export_block_state` /
   `import_block_state`), and one-tick delayed doorbell stop & resume parity.
2. `AsymmetricCFGStepper` late-step curvature-gated CFG folding &
   `TaylorResidualCompensator` second-order finite-difference residual forecast.
3. `SD35AsyncPipelineManager` continuous batching, `cu_seqlens` variable-resolution
   ragged packing ($512^2 + 768^2 + 1024^2$), and `AsyncDecoder` / `ChunkedDecode`.
"""

from __future__ import annotations

import contextlib
import copy
import io
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ditserve.models.pipeline import ChunkedDecode, decode_final_latent
from ditserve.models.sd35_batch import (
    SD35AsyncPipelineManager,
    build_cu_seqlens,
    calculate_shift,
    forward_packed_sd35,
    packing_stats,
)
from ditserve.models.wan.async_decode import AsyncDecoder
from ditserve.models.wan.block import BlockState, WanBlockExecutor, export_block_state, import_block_state
from ditserve.models.wan.cfg_reuse import AsymmetricCFGStepper
from ditserve.models.wan.parallel import FixedLayout
from ditserve.models.wan.runtime import RequestState, normalize_spec
from ditserve.models.wan.taylor_cache import TaylorResidualCompensator


# ==============================================================================
# Deterministic CPU Fixtures for WanBlockExecutor & ChunkedDecode
# ==============================================================================


class _TinyUniPCScheduler:
    """Deterministic UniPC scheduler exposing exact `UNIPC_REQUIRED_FIELDS`."""

    def __init__(self, config: dict):
        from diffusers.configuration_utils import FrozenDict

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
        self.num_inference_steps = int(cfg.get("num_inference_steps", 4))
        self.sigmas = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, dtype=torch.float32, device="cpu")
        self.init_noise_sigma = 1.0
        self.predict_x0 = True
        self.timesteps = torch.tensor([900, 700, 400, 100][: self.num_inference_steps], dtype=torch.int64, device="cpu")
        self.model_outputs = [None, None]
        self.timestep_list = [None, None]
        self.lower_order_nums = 0
        self.disable_corrector = list(cfg.get("disable_corrector", []))
        self.solver_p = None
        self.last_sample = None
        self._step_index = None
        self._begin_index = None

    @property
    def config(self):
        return self._internal_dict

    @classmethod
    def from_config(cls, config: dict) -> "_TinyUniPCScheduler":
        return cls(dict(config))

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, return_dict: bool = False):
        idx = int(self._step_index or 0)
        self.model_outputs = [self.model_outputs[-1], model_output.to(dtype=torch.float32).clone()]
        self.timestep_list = [self.timestep_list[-1], self.timesteps[min(idx, len(self.timesteps) - 1)]]
        self.last_sample = sample.clone()
        self.lower_order_nums = min(2, self.lower_order_nums + 1)
        self.this_order = min(2, self.lower_order_nums)
        self._step_index = idx + 1
        next_sample = sample.to(dtype=torch.float32) - 0.125 * model_output.to(dtype=torch.float32)
        return (next_sample,)


_TinyUniPCScheduler.__name__ = "UniPCMultistepScheduler"


class _TinyWanBlock(torch.nn.Module):
    def __init__(self, dim: int, idx: int):
        super().__init__()
        self.scale = 0.02 * (idx + 1)
        self.linear = torch.nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor, encoder: torch.Tensor, projected: torch.Tensor, rotary: tuple) -> torch.Tensor:
        enc_bias = encoder.mean(dim=1, keepdim=True)
        proj_bias = projected.mean(dim=-2)
        cos, sin = rotary
        delta = self.linear(hidden.float() + enc_bias.float() + proj_bias.float() + cos.float() * 0.05 + sin.float() * 0.03)
        return (hidden.float() + self.scale * delta).to(dtype=hidden.dtype)


class _TinyWanTransformer(torch.nn.Module):
    def __init__(self, num_layers: int = 8, dim: int = 12, in_channels: int = 16, out_channels: int = 16):
        super().__init__()
        from diffusers.configuration_utils import FrozenDict

        self.dtype = torch.float16
        self.config = FrozenDict(
            {
                "patch_size": (1, 2, 2),
                "in_channels": in_channels,
                "out_channels": out_channels,
                "added_kv_proj_dim": None,
                "image_dim": None,
            }
        )
        self.gradient_checkpointing = False
        self.dim = dim
        self.blocks = torch.nn.ModuleList([_TinyWanBlock(dim, i) for i in range(num_layers)])
        self.scale_shift_table = torch.ones((1, 2, dim), dtype=torch.float32) * 0.1
        self.norm_out = torch.nn.LayerNorm(dim)
        self.proj_out = torch.nn.Linear(dim, 1 * 2 * 2 * out_channels)
        self.text_proj = torch.nn.Linear(16, dim)

    def rope(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, f, h, w = latent.shape
        tokens = f * (h // 2) * (w // 2)
        base = torch.linspace(0.1, 0.9, tokens * self.dim, dtype=torch.float16, device=latent.device).reshape(
            b, tokens, self.dim
        )
        return (base, base * 0.5)

    def patch_embedding(self, latent: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = latent.shape
        oh, ow = h // 2, w // 2
        pooled = latent.float().reshape(b, c, f, oh, 2, ow, 2).mean(dim=(4, 6))
        out = pooled[:, : self.dim, :, :, :]
        return out.to(dtype=self.dtype)

    def condition_embedder(
        self,
        timestep: torch.Tensor,
        embedding: torch.Tensor,
        _image: None,
        timestep_seq_len: int | None = None,
    ):
        b = embedding.shape[0]
        enc = self.text_proj(embedding[:, :4, :16].float()).to(dtype=self.dtype)
        t_val = float(timestep.float().mean().item()) / 1000.0
        if timestep_seq_len is not None:
            temb = torch.full((b, timestep_seq_len, self.dim), t_val, dtype=self.dtype, device=embedding.device)
            projected = torch.full(
                (b, timestep_seq_len, 6 * self.dim), t_val * 0.5, dtype=self.dtype, device=embedding.device
            )
        else:
            temb = torch.full((b, self.dim), t_val, dtype=self.dtype, device=embedding.device)
            projected = torch.full((b, 6 * self.dim), t_val * 0.5, dtype=self.dtype, device=embedding.device)
        return temb, projected, enc, None


def _make_wan_worker_and_state(
    *,
    request_id: str = "req-0",
    steps: int = 3,
    guidance_scale: float = 5.0,
    num_layers: int = 8,
    seed: int = 42,
    model: _TinyWanTransformer | None = None,
) -> tuple[SimpleNamespace, RequestState, _TinyWanTransformer]:
    transformer = model if model is not None else _TinyWanTransformer(num_layers=num_layers)
    spec = normalize_spec(
        {
            "request_id": request_id,
            "prompt": "a scenic mountain river",
            "negative_prompt": "blurry",
            "seed": seed,
            "height": 32,
            "width": 32,
            "frames": 5,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "cache": False,
            "threshold": 0.08,
            "max_continuous": 3,
        }
    )
    immutable = {k: v for k, v in spec.items() if k not in ("cache", "threshold", "max_continuous")}
    sched = _TinyUniPCScheduler({"num_inference_steps": steps, "disable_corrector": [], "solver_order": 2})
    gen = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn((1, 16, 2, 4, 4), generator=gen, dtype=torch.float32, device="cpu")
    prompt_embeds = torch.full((1, 512, 4096), 0.25, dtype=torch.float16, device="cpu")
    neg_embeds = torch.full((1, 512, 4096), -0.15, dtype=torch.float16, device="cpu") if guidance_scale > 1.0 else None

    owner = object()
    state = RequestState(
        request_id=request_id,
        spec=spec,
        step_index=0,
        latents=latents,
        cache_enabled=False,
        events=[],
        scheduler=sched,
        generator=gen,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        contexts={},
        residency="gpu",
        needs_anchor=False,
        finished=False,
        owner=owner,
        pending_snapshot=None,
        prepare_ms=1.5,
        current_step_refreshed=False,
        immutable_spec=immutable,
    )

    worker = SimpleNamespace(
        torch=torch,
        device="cpu",
        pipe=SimpleNamespace(
            transformer=transformer,
            vae=SimpleNamespace(config=SimpleNamespace(z_dim=16, scale_factor_temporal=4, scale_factor_spatial=8)),
        ),
        attention="math",
        expand_timesteps=False,
        _lock=threading.RLock(),
        _owner=owner,
        _states={request_id: state},
        _scheduler_type=_TinyUniPCScheduler,
        p=SimpleNamespace(_sdpa_math_context=lambda _t: contextlib.nullcontext()),
        _dit_attention_context=lambda: contextlib.nullcontext(),
        _sync=lambda: None,
    )

    def _check(st: RequestState, residency: str | None = None) -> None:
        if st.owner is not worker._owner or worker._states.get(st.request_id) is not st:
            raise ValueError("foreign or released request state")
        if residency is not None and st.residency != residency:
            raise ValueError(f"expected residency {residency}, got {st.residency}")

    def _update_policy(st: RequestState) -> None:
        norm = normalize_spec(st.spec)
        ident = {k: v for k, v in norm.items() if k not in ("cache", "threshold", "max_continuous")}
        if ident != st.immutable_spec:
            raise ValueError("immutable spec changed")

    def _validate_request_schema(st: RequestState) -> None:
        assert st.request_id == st.spec["request_id"]

    def _payload(st: RequestState) -> dict:
        return {
            "latents": st.latents,
            "scheduler": vars(st.scheduler),
            "prompt_embeds": st.prompt_embeds,
            "negative_prompt_embeds": st.negative_prompt_embeds,
            "contexts": st.contexts,
            "rng_state": st.generator.get_state(),
        }

    def _release(st: RequestState) -> None:
        st.residency = "released"
        worker._states.pop(st.request_id, None)

    worker._check = _check
    worker._update_policy = _update_policy
    worker._validate_request_schema = _validate_request_schema
    worker._payload = _payload
    worker.release = _release
    return worker, state, transformer


def _attach_executor(
    worker: SimpleNamespace,
    state: RequestState,
    *,
    parallel=None,
    interleave: bool = False,
    cfg_fold_options: dict | None = None,
) -> tuple[WanBlockExecutor, BlockState]:
    with patch.object(WanBlockExecutor, "_validate_model"):
        executor = WanBlockExecutor(
            worker,
            parallel=parallel,
            interleave=interleave,
            cfg_fold_options=cfg_fold_options,
        )
    return executor, executor.attach(state)


# ==============================================================================
# 1. WanBlockExecutor State Machine, Continuation Serialization & Doorbell Stop
# ==============================================================================


class TestWanBlockExecutorAndContinuationState:
    """Verifies `WanBlockExecutor` sub-step quanta, `BlockState` export/import, and doorbell stop."""

    def test_substep_quanta_match_full_step_and_interleaved_branches(self) -> None:
        model = _TinyWanTransformer(num_layers=8)
        # 1. Full step reference (16 block forwards per step: 8 cond + 8 uncond)
        w_ref, s_ref, _ = _make_wan_worker_and_state(steps=2, num_layers=8, model=model)
        ex_ref, cont_ref = _attach_executor(w_ref, s_ref)
        for _ in range(2):
            q = ex_ref.quantum(cont_ref, "step")
            assert q == 16
            rec = ex_ref.advance(cont_ref, blocks=q)
            assert rec["step_completed"] is True
            assert rec["extra_state_bytes"] == 0

        # 2. Sub-step block-3 quanta stop at branch and solver commit boundaries
        w_blk, s_blk, _ = _make_wan_worker_and_state(steps=2, num_layers=8, model=model)
        ex_blk, cont_blk = _attach_executor(w_blk, s_blk)
        while s_blk.step_index < 2:
            rec = ex_blk.advance(cont_blk, blocks=ex_blk.quantum(cont_blk, "block-3"))
            if not rec["step_completed"]:
                assert rec["extra_state_bytes"] > 0
            else:
                assert rec["extra_state_bytes"] == 0
        assert torch.equal(s_ref.latents, s_blk.latents)

        # 3. Interleaved cond/uncond execution produces identical latents
        w_int, s_int, _ = _make_wan_worker_and_state(steps=2, num_layers=8, model=model)
        ex_int, cont_int = _attach_executor(w_int, s_int, interleave=True)
        while s_int.step_index < 2:
            ex_int.advance(cont_int, blocks=3)
        assert torch.equal(s_ref.latents, s_int.latents)

        # 4. Wan2.2 per-token expand_timesteps & advance_pair lockstep execution
        w_p1, s_p1, _ = _make_wan_worker_and_state(request_id="pair-1", steps=1, num_layers=6, seed=11, model=model)
        _, s_p2, _ = _make_wan_worker_and_state(request_id="pair-2", steps=1, num_layers=6, seed=19, model=model)
        s_p2.owner = w_p1._owner
        w_p1._states["pair-2"] = s_p2
        w_p1.expand_timesteps = True
        ex_pair, c1 = _attach_executor(w_p1, s_p1)
        c2 = ex_pair.attach(s_p2)
        rows = ex_pair.advance_pair(c1, c2, blocks=3)
        assert len(rows) == 2
        assert all(r["paired"] is True and r["blocks"] == 3 for r in rows)

    def test_continuation_packet_export_import_and_sha256_tamper_rejection(self) -> None:
        model = _TinyWanTransformer(num_layers=8)
        w_ref, s_ref, _ = _make_wan_worker_and_state(steps=2, num_layers=8, model=model)
        ex_ref, cont_ref = _attach_executor(w_ref, s_ref)
        ex_ref.advance(cont_ref, blocks=16)
        ex_ref.advance(cont_ref, blocks=16)

        for stop_blocks, expected_branch, expected_has_cond in ((3, "cond", False), (11, "uncond", True)):
            w_src, s_src, _ = _make_wan_worker_and_state(steps=2, num_layers=8, model=model)
            ex_src, cont_src = _attach_executor(w_src, s_src)
            ex_src.advance(cont_src, blocks=16)
            ex_src.advance(cont_src, blocks=stop_blocks)
            payload, timing = export_block_state(ex_src, cont_src)
            assert payload["manifest"]["layout"]["branch"] == expected_branch
            assert payload["manifest"]["layout"]["conditional"] is expected_has_cond
            assert timing["packet_bytes"] > 0

            buf = io.BytesIO()
            torch.save(payload, buf)
            buf.seek(0)
            loaded = torch.load(buf, map_location="cpu", weights_only=True)

            w_dst, placeholder, _ = _make_wan_worker_and_state(
                request_id="placeholder", steps=2, num_layers=8, model=copy.deepcopy(model)
            )
            w_dst.release(placeholder)
            with patch.object(WanBlockExecutor, "_validate_model"):
                ex_dst = WanBlockExecutor(w_dst)
            resumed_base, resumed_cont, _ = import_block_state(ex_dst, loaded)
            assert (resumed_cont.branch, resumed_cont.next_block) == (expected_branch, stop_blocks % 8)
            while resumed_base.step_index < s_ref.step_index:
                ex_dst.advance(resumed_cont, blocks=5)
            assert torch.equal(s_ref.latents, resumed_base.latents)

            # Tampering with packet tensor triggers SHA256 verification failure before registration
            tampered = copy.deepcopy(loaded)
            tampered["tensors"]["packet.hidden"] = tampered["tensors"]["packet.hidden"] + 1.0
            with pytest.raises(ValueError, match="checksum"):
                import_block_state(ex_dst, tampered)

    def test_cfg_fold_state_crosses_worker_boundary_via_export_import(self) -> None:
        model = _TinyWanTransformer(num_layers=8)
        fold_opts = {"folded_steps": {1}, "final_step_exact_anchor": False, "cfg_vel_alpha": 0.0}
        w_src, s_src, _ = _make_wan_worker_and_state(steps=2, num_layers=8, model=model)
        ex_src, cont_src = _attach_executor(w_src, s_src, cfg_fold_options=fold_opts)
        ex_src.advance(cont_src, blocks=16, cfg_folded=False)

        payload, _ = export_block_state(ex_src, cont_src)
        assert "cfg_fold" in payload["manifest"]["layout"]
        assert "cfg_fold.delta" in payload["tensors"]

        w_dst, placeholder, _ = _make_wan_worker_and_state(
            request_id="placeholder", steps=2, num_layers=8, model=copy.deepcopy(model)
        )
        w_dst.release(placeholder)
        with patch.object(WanBlockExecutor, "_validate_model"):
            ex_dst = WanBlockExecutor(w_dst, cfg_fold_options=fold_opts)
        resumed_base, resumed_cont, _ = import_block_state(ex_dst, payload)

        ex_src.advance(cont_src, blocks=8, cfg_folded=True)
        ex_dst.advance(resumed_cont, blocks=8, cfg_folded=True)
        assert torch.equal(s_src.latents, resumed_base.latents)
        assert resumed_cont.stepper.stats["folded_cfg_steps"] == 1
        assert resumed_cont.stepper.stats["full_cfg_steps"] == 1

    def test_one_tick_delayed_doorbell_stop_and_resume_parity(self) -> None:
        model = _TinyWanTransformer(num_layers=8)
        w_ref, s_ref, _ = _make_wan_worker_and_state(steps=1, num_layers=8, model=model)
        ex_ref, cont_ref = _attach_executor(w_ref, s_ref)
        ex_ref.advance(cont_ref, blocks=16)

        # 1. Local doorbell stops at check_every=2 boundary and resumes bit-exactly
        w_loc, s_loc, _ = _make_wan_worker_and_state(steps=1, num_layers=8, model=model)
        ex_loc, cont_loc = _attach_executor(w_loc, s_loc)
        checks = 0

        def _local_bell() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 2

        first = ex_loc.advance(cont_loc, blocks=16, doorbell=_local_bell, check_every=2)
        assert first["doorbell_stop"] is True
        assert first["blocks"] == 4
        assert checks == 2
        rest = ex_loc.advance(cont_loc, blocks=16)
        assert rest["step_completed"] is True
        assert torch.equal(s_ref.latents, s_loc.latents)

        # 2. Multi-rank group doorbell uses one-tick delayed consensus (stops at check point 2)
        w_grp, s_grp, _ = _make_wan_worker_and_state(steps=1, num_layers=8, model=model)
        reduced_flags: list[bool] = []

        class _FakeParallel:
            def __init__(self):
                self.handle = SimpleNamespace(
                    rank=0,
                    current=FixedLayout((0, 1), cfg=2),
                    begin_step=lambda *a, **k: None,
                    end_step=lambda *a, **k: None,
                    fail=lambda: None,
                )

            def begin_step(self, *a, **k):
                return None

            def end_step(self, *a, **k):
                return None

            def quantum_point(self, *a, **k):
                return None

            def reduce_flag(self, flag) -> bool:
                val = bool(int(flag[0])) if hasattr(flag, "__getitem__") else bool(flag)
                reduced_flags.append(val)
                return val

            def exchange_predictions(self, pred):
                return pred, pred * 0.9

        ex_grp, cont_grp = _attach_executor(w_grp, s_grp, parallel=_FakeParallel())
        grp_rec = ex_grp.advance(cont_grp, blocks=8, doorbell=lambda: True, check_every=2)
        assert grp_rec["doorbell_stop"] is True
        # Check point 1 at block 2 reduces True; check point 2 at block 4 acts on that previous True!
        assert grp_rec["blocks"] == 4
        assert reduced_flags == [True]

    def test_doorbell_rung_reuses_two_slot_ring_buffer_without_per_checkpoint_allocations(self) -> None:
        w_grp, s_grp, _ = _make_wan_worker_and_state(steps=1, num_layers=8)
        reduced_tensors: list[torch.Tensor] = []

        class _FakeParallel:
            def __init__(self):
                self.handle = SimpleNamespace(
                    rank=0,
                    current=FixedLayout((0, 1), cfg=2),
                )

            def reduce_flag(self, flag: torch.Tensor) -> None:
                reduced_tensors.append(flag)

        ex_grp, _ = _attach_executor(w_grp, s_grp, parallel=_FakeParallel())
        assert ex_grp._doorbell_ring == {}
        assert ex_grp._doorbell_slot == 0

        empty_calls = 0
        event_calls = 0
        sync_calls = 0

        class _FakeCudaEvent:
            def __init__(self):
                nonlocal event_calls
                event_calls += 1
                self.recorded = 0

            def record(self) -> None:
                self.recorded += 1

            def synchronize(self) -> None:
                nonlocal sync_calls
                sync_calls += 1

        class _FakeTorch:
            int32 = torch.int32
            cuda = SimpleNamespace(Event=_FakeCudaEvent)

            @staticmethod
            def empty(size, *, dtype=torch.int32, device=None, pin_memory: bool = False):
                nonlocal empty_calls
                empty_calls += 1
                return torch.empty(size, dtype=dtype)

        fake_cuda_device = torch.device("cuda:0")
        pending: list = []
        bell_state = False

        # Checkpoint 1: initializes the 2-slot ring buffer (2 dev_flag + 4 host tensors = 6 empty calls, 2 Event calls)
        assert ex_grp._doorbell_rung(
            lambda: bell_state, pending, group=True, torch=_FakeTorch, device=fake_cuda_device
        ) is False
        assert empty_calls == 6
        assert event_calls == 2
        assert ex_grp._doorbell_slot == 1
        assert "cuda:0" in ex_grp._doorbell_ring
        slot0, slot1 = ex_grp._doorbell_ring["cuda:0"]
        assert len(pending) == 1 and pending[0][0] is slot0[3] and pending[0][1] is slot0[2]

        # Checkpoints 2..5: alternate slots 1 -> 0 -> 1 -> 0 with ZERO new tensor or Event allocations
        for expected_slot_after in (0, 1, 0, 1):
            stopped = ex_grp._doorbell_rung(
                lambda: bell_state, pending, group=True, torch=_FakeTorch, device=fake_cuda_device
            )
            assert stopped is False
            assert empty_calls == 6
            assert event_calls == 2
            assert ex_grp._doorbell_slot == expected_slot_after

        # Checkpoint 6: doorbell rings True -> staged into current slot without new allocations
        bell_state = True
        assert ex_grp._doorbell_rung(
            lambda: bell_state, pending, group=True, torch=_FakeTorch, device=fake_cuda_device
        ) is False
        assert empty_calls == 6
        assert event_calls == 2

        # Checkpoint 7: synchronizes previous slot's event, observes reduced True, and stops!
        assert ex_grp._doorbell_rung(
            lambda: False, pending, group=True, torch=_FakeTorch, device=fake_cuda_device
        ) is True
        assert empty_calls == 6
        assert event_calls == 2
        assert sync_calls == 6
        assert pending == []
        # Every reduce_flag call reused one of the two pre-allocated device flag tensors
        assert {id(t) for t in reduced_tensors} == {id(slot0[0]), id(slot1[0])}


# ==============================================================================
# 2. AsymmetricCFGStepper & TaylorResidualCompensator
# ==============================================================================


class TestAsymmetricCFGAndTaylorResidualCompensator:
    """Verifies curvature-gated late-step CFG folding and Taylor residual compensation."""

    def test_asymmetric_cfg_stepper_curvature_gating_anchor_and_momentum(self) -> None:
        stepper = AsymmetricCFGStepper(
            torch,
            folded_steps={3, 4, 5},
            final_step_exact_anchor=True,
            max_folded_steps=2,
            min_full_cfg_steps=2,
            curvature_threshold=0.15,
            cfg_vel_alpha=0.25,
        )
        total_steps = 6
        # Step 0..2 are full CFG steps; feed low-curvature delta_cfg trajectory
        base_cond = torch.ones((1, 4, 2, 2, 2), dtype=torch.float32) * 2.0
        base_uncond = torch.ones((1, 4, 2, 2, 2), dtype=torch.float32) * 1.0
        for step_idx in (0, 1, 2):
            assert stepper.should_fold(step_idx, total_steps) is False
            stepper.observe_cfg_pair(step_idx, base_cond + 0.005 * step_idx, base_uncond)

        assert stepper._last_curvature is not None and stepper._last_curvature < 0.15
        # Step 3 is in folded_steps and curvature < 0.15 -> eligible to fold
        assert stepper.should_fold(3, total_steps) is True
        folded_pred = stepper.fold_prediction(3, base_cond + 0.015, guidance_scale=5.0)
        assert folded_pred.shape == base_cond.shape
        assert stepper.stats["folded_cfg_steps"] == 1

        # Step 4 is also eligible (within max_folded_steps=2)
        assert stepper.should_fold(4, total_steps) is True
        stepper.fold_prediction(4, base_cond + 0.02, guidance_scale=5.0)
        assert stepper.stats["folded_cfg_steps"] == 2

        # Final step 5 is protected by final_step_exact_anchor=True AND max_folded_steps=2
        assert stepper.should_fold(5, total_steps) is False

        # High curvature blocks folding when curvature exceeds curvature_threshold
        stepper_high_curv = AsymmetricCFGStepper(
            torch,
            folded_steps={2},
            final_step_exact_anchor=False,
            min_full_cfg_steps=2,
            curvature_threshold=0.05,
        )
        stepper_high_curv.observe_cfg_pair(0, base_cond, base_uncond)
        stepper_high_curv.observe_cfg_pair(1, -base_cond * 3.0, base_uncond)
        assert stepper_high_curv._last_curvature is not None and stepper_high_curv._last_curvature > 0.05
        assert stepper_high_curv.should_fold(2, 5) is False

    def test_taylor_residual_compensator_second_order_forecast_and_anchor_restore(self) -> None:
        comp = TaylorResidualCompensator(
            torch,
            alpha=0.28,
            damping=0.70,
            max_norm_ratio=0.12,
            adaptive_lsq=True,
        )
        base_buf = torch.ones((1, 16, 8), dtype=torch.float32) * 2.0
        ctx = SimpleNamespace(
            buffers={
                "blocks_Bn_residual_buffer": base_buf.clone(),
                "blocks_Bn_residual_buffer_cfg": (base_buf * 0.8).clone(),
            }
        )
        state = SimpleNamespace(step_index=0, cache_enabled=True, contexts={"blocks": ctx})

        # Steps 0, 1, 2: anchor steps build finite-difference velocities and adaptive_lsq ratios
        for step_idx in range(3):
            state.step_index = step_idx
            comp.pre_step(state)
            ctx.buffers["blocks_Bn_residual_buffer"] = base_buf + 0.15 * (step_idx + 1)
            ctx.buffers["blocks_Bn_residual_buffer_cfg"] = base_buf * 0.8 + 0.10 * (step_idx + 1)
            state.step_index = step_idx + 1
            rec = comp.post_step(
                state,
                {
                    "step_index": step_idx,
                    "cache_contexts": [{"cond_reused": False, "uncond_reused": False}],
                },
            )
            assert rec["taylor_compensated_forwards"] == 0

        assert comp.stats["anchor_updates"] == 6
        clean_anchor = ctx.buffers["blocks_Bn_residual_buffer"].clone()

        # Step 3: DBCache reuse step -> pre_step extrapolates Bn buffer, post_step restores clean anchor
        state.step_index = 3
        comp.pre_step(state)
        extrapolated = ctx.buffers["blocks_Bn_residual_buffer"].clone()
        assert not torch.equal(extrapolated, clean_anchor)

        state.step_index = 4
        rec_reused = comp.post_step(
            state,
            {
                "step_index": 3,
                "cache_contexts": [{"cond_reused": True, "uncond_reused": True}],
            },
        )
        assert rec_reused["taylor_compensated_forwards"] == 2
        assert comp.stats["taylor_compensated_forwards"] == 2
        torch.testing.assert_close(ctx.buffers["blocks_Bn_residual_buffer"], clean_anchor, rtol=1e-6, atol=1e-6)


# ==============================================================================
# 3. SD35 Continuous Batching, Ragged Packing (`cu_seqlens`), & ChunkedDecode
# ==============================================================================


class TestSD35RaggedBatchingAndChunkedDecode:
    """Verifies SD3.5 ragged packing (`512^2 + 768^2 + 1024^2`), continuous batching, and ChunkedDecode."""

    def test_ragged_packing_512_768_1024_cu_seqlens_and_forward_equivalence(self) -> None:
        # Mixed resolutions: 512x512 (32x32 patches -> 1024 tokens),
        # 768x768 (48x48 patches -> 2304 tokens), 1024x1024 (64x64 patches -> 4096 tokens)
        stats = packing_stats([512, 768, 1024], [512, 768, 1024], vae_scale_factor=8, patch_size=2)
        assert stats["image_token_counts"] == [1024, 2304, 4096]
        assert stats["cu_seqlens"] == [0, 1024, 3328, 7424]
        assert build_cu_seqlens([1024, 2304, 4096]) == [0, 1024, 3328, 7424]
        assert stats["packed_image_tokens"] == 7424
        assert stats["max_seqlen"] == 4096
        assert stats["padded_image_tokens"] == 3 * 4096
        assert stats["padding_saved_tokens"] == 4864
        assert stats["padding_waste_ratio"] == pytest.approx(4864 / 12288, abs=1e-4)
        assert calculate_shift(1024) < calculate_shift(2304) < calculate_shift(4096)

        # Verify forward_packed_sd35 matches solo execution across variable-resolution latents
        lat_list = [
            torch.randn((1, 4, 4, 4), dtype=torch.float32),
            torch.randn((1, 4, 6, 6), dtype=torch.float32),
            torch.randn((1, 4, 8, 8), dtype=torch.float32),
        ]

        class _AdaLN(torch.nn.Module):
            def __init__(self, hidden: int, mult: int):
                super().__init__()
                self.silu = torch.nn.SiLU()
                self.linear = torch.nn.Linear(hidden, mult * hidden)
                self.norm = torch.nn.LayerNorm(hidden)

        class _PatchEmbed(torch.nn.Module):
            def __init__(self, in_ch: int, hidden: int):
                super().__init__()
                self.proj = torch.nn.Conv2d(in_ch, hidden, kernel_size=2, stride=2)

            def forward(self, lat: torch.Tensor) -> torch.Tensor:
                return self.proj(lat).flatten(2).transpose(1, 2)

        class _JointAttn(torch.nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.heads = 2
                self.context_pre_only = False
                self.to_q = torch.nn.Linear(hidden, hidden)
                self.to_k = torch.nn.Linear(hidden, hidden)
                self.to_v = torch.nn.Linear(hidden, hidden)
                self.add_q_proj = torch.nn.Linear(hidden, hidden)
                self.add_k_proj = torch.nn.Linear(hidden, hidden)
                self.add_v_proj = torch.nn.Linear(hidden, hidden)
                self.to_out = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden), torch.nn.Identity()])
                self.to_add_out = torch.nn.Linear(hidden, hidden)

        class _ToySD35Transformer(torch.nn.Module):
            def __init__(self, in_ch: int = 4, hidden: int = 8):
                super().__init__()
                self.out_channels = in_ch
                self.config = SimpleNamespace(patch_size=2, in_channels=in_ch, out_channels=in_ch)
                self.pos_embed = _PatchEmbed(in_ch, hidden)
                self.time_text_embed = lambda t, p: t.unsqueeze(-1).expand(t.shape[0], hidden) + p[:, :hidden]
                self.context_embedder = torch.nn.Linear(12, hidden)

                class _Block(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.use_dual_attention = False
                        self.context_pre_only = False
                        self.norm1 = _AdaLN(hidden, 6)
                        self.norm1_context = _AdaLN(hidden, 6)
                        self.attn = _JointAttn(hidden)
                        self.norm2 = torch.nn.LayerNorm(hidden)
                        self.ff = torch.nn.Linear(hidden, hidden)
                        self.norm2_context = torch.nn.LayerNorm(hidden)
                        self.ff_context = torch.nn.Linear(hidden, hidden)

                self.transformer_blocks = [_Block(), _Block()]
                self.norm_out = _AdaLN(hidden, 2)
                self.proj_out = torch.nn.Linear(hidden, in_ch * 4)

        toy = _ToySD35Transformer()
        prompts = [torch.randn((1, 5, 12)) for _ in range(3)]
        pooled = [torch.randn((1, 8)) for _ in range(3)]
        t_steps = [torch.tensor([500.0]), torch.tensor([300.0]), torch.tensor([100.0])]

        batched_outs, meta = forward_packed_sd35(torch, toy, lat_list, t_steps, prompts, pooled)
        assert meta["sequence_packed"] is True
        assert meta["cu_seqlens_img"] == [0, 4, 13, 29]
        for i in range(3):
            solo_outs, _ = forward_packed_sd35(torch, toy, [lat_list[i]], [t_steps[i]], [prompts[i]], [pooled[i]])
            torch.testing.assert_close(batched_outs[i], solo_outs[0], rtol=1e-5, atol=1e-5)

    def test_sd35_async_pipeline_manager_encode_decode_overlap_and_ticket_validation(self) -> None:
        encoded_store: dict[str, dict] = {}

        class _FakeSD35Backend:
            def __init__(self):
                self.torch = torch
                self.device = "cpu"
                self.config = {"options": {}}
                self.defaults = {"steps": 4, "guidance_scale": 4.5, "max_sequence_length": 256}
                self.states: dict[str, dict] = {}

            def _spec(self, request: dict) -> dict:
                raw = request.get("spec", request)
                return {
                    "prompt": raw.get("prompt", ""),
                    "negative_prompt": raw.get("negative_prompt", ""),
                    "max_sequence_length": int(raw.get("max_sequence_length", 256)),
                    "guidance_scale": float(raw.get("guidance_scale", 4.5)),
                }

            def _ensure_staged(self, prompt: str, negative_prompt: str, max_seq_len: int, do_cfg: bool) -> dict:
                key = f"{prompt}|{negative_prompt}|{max_seq_len}|{int(do_cfg)}"
                encoded_store[key] = {
                    "prompt_embeds": torch.ones((1, 4, 8)),
                    "negative_prompt_embeds": torch.zeros((1, 4, 8)) if do_cfg else None,
                    "pooled_prompt_embeds": torch.ones((1, 8)),
                    "negative_pooled_prompt_embeds": torch.zeros((1, 8)) if do_cfg else None,
                }
                return {"key": key, "staged": True, "cached": False, "encode_ms": 1.25}

            def _fetch_staged(self, prompt: str, negative_prompt: str, max_seq_len: int, do_cfg: bool, stream=None):
                stage_info = self._ensure_staged(prompt, negative_prompt, max_seq_len, do_cfg)
                return stage_info, {"tensors": encoded_store[stage_info["key"]], "push_ms": 0.12, "hit": True}

            def _decode_latents_to_np(self, latents: torch.Tensor) -> np.ndarray:
                b, _, h, w = latents.shape
                return np.ones((b, h * 8, w * 8, 3), dtype=np.float32)

            def finish(self, request: dict) -> dict:
                sid = request["state_id"]
                arr = self.states[sid]["output"]
                return {"state_id": sid, "shape": list(arr.shape)}

        backend = _FakeSD35Backend()
        mgr = SD35AsyncPipelineManager(backend)
        try:
            # Submit async encode and verify ticket consumption and spec-mismatch protection
            sub = mgr.submit_encode({"prompt": "cyberpunk skyline", "negative_prompt": "blur", "guidance_scale": 4.5})
            ticket = sub["ticket"]
            polled = mgr.poll_encode({"ticket": ticket, "wait": True})
            assert polled["ready"] is True

            with pytest.raises(ValueError, match="different spec"):
                mgr.consume_encode_ticket(
                    ticket,
                    {"prompt": "wrong prompt", "negative_prompt": "blur", "max_sequence_length": 256, "guidance_scale": 4.5},
                )

            consumed = mgr.consume_encode_ticket(
                ticket,
                {"prompt": "cyberpunk skyline", "negative_prompt": "blur", "max_sequence_length": 256, "guidance_scale": 4.5},
            )
            assert "prompt_embeds" in consumed["tensors"]

            # Submit async decode on finished state and verify readiness + output shape
            fake_state = {
                "id": "sd35-state-1",
                "step": 4,
                "steps": 4,
                "latents": torch.zeros((1, 16, 64, 64), dtype=torch.float32),
                "spec": {"height": 512, "width": 512, "steps": 4},
            }
            backend.states["sd35-state-1"] = fake_state
            dec_sub = mgr.submit_decode({"state_id": "sd35-state-1", "output": "summary"}, fake_state)
            dec_polled = mgr.poll_decode({"ticket": dec_sub["ticket"], "wait": True})
            assert dec_polled["ready"] is True
            assert dec_polled["shape"] == [1, 512, 512, 3]
            assert mgr.has_active_decode("sd35-state-1") is False
        finally:
            mgr.close()

    def test_chunked_decode_parity_interleaved_isolation_and_async_decoder(self) -> None:
        class _ToyWanVAE(torch.nn.Module):
            def __init__(self, z_dim: int = 4, patch_size: int | None = None):
                super().__init__()
                self.dtype = torch.float32
                self.use_tiling = False
                self._conv_num = 1
                self._feat_map = [None]
                self._enc_feat_map = [None]
                self.config = SimpleNamespace(
                    z_dim=z_dim,
                    latents_mean=[0.1] * z_dim,
                    latents_std=[0.5] * z_dim,
                    patch_size=patch_size,
                )
                out_ch = 3 * (patch_size * patch_size if patch_size else 1)
                self.proj = torch.nn.Linear(z_dim, out_ch)

            def clear_cache(self) -> None:
                self._feat_map = [None]
                self._enc_feat_map = [None]

            def post_quant_conv(self, x: torch.Tensor) -> torch.Tensor:
                return x * 1.1

            def decoder(self, chunk: torch.Tensor, feat_cache: list, feat_idx: list, first_chunk: bool = False) -> torch.Tensor:
                prev = feat_cache[0]
                feat_cache[0] = chunk.clone()
                mixed = chunk if (first_chunk or prev is None) else (chunk + 0.25 * prev[:, :, -1:, :, :])
                b, c, f, h, w = mixed.shape
                flat = mixed.permute(0, 2, 3, 4, 1).reshape(-1, c)
                out = self.proj(flat).reshape(b, f, h, w, -1).permute(0, 4, 1, 2, 3)
                if not first_chunk:
                    out = out.repeat_interleave(2, dim=2)
                return out

            def decode(self, z: torch.Tensor, return_dict: bool = False):
                x = self.post_quant_conv(z)
                cache = [None] * self._conv_num
                chunks = []
                for i in range(x.shape[2]):
                    ch = x[:, :, i : i + 1, :, :]
                    chunks.append(self.decoder(ch, feat_cache=cache, feat_idx=[0], first_chunk=(i == 0)))
                video = torch.cat(chunks, dim=2)
                if self.config.patch_size is not None:
                    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

                    video = unpatchify(video, patch_size=self.config.patch_size)
                video = torch.clamp(video, min=-1.0, max=1.0)
                self.clear_cache()
                return (video,)

        class _VideoProcessor:
            @staticmethod
            def postprocess_video(video: torch.Tensor, output_type: str = "np") -> np.ndarray:
                return video.permute(0, 2, 3, 4, 1).numpy()

        for patch_size in (None, 2):
            pipe = SimpleNamespace(vae=_ToyWanVAE(patch_size=patch_size), video_processor=_VideoProcessor())
            latent = torch.randn((1, 4, 4, 4, 4), dtype=torch.float32)
            expected = decode_final_latent(torch, pipe, latent)

            for grouping in ([1, 1, 1, 1], [2, 2], [4]):
                dec = ChunkedDecode(torch, pipe, latent)
                for cnt in grouping:
                    dec.advance(cnt)
                assert dec.done is True
                actual = dec.finish()
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

        # Interleaved ChunkedDecode instances maintain independent temporal caches
        pipe_shared = SimpleNamespace(vae=_ToyWanVAE(patch_size=None), video_processor=_VideoProcessor())
        lat_a = torch.randn((1, 4, 3, 4, 4), dtype=torch.float32)
        lat_b = torch.randn((1, 4, 3, 4, 4), dtype=torch.float32) * 0.7
        ref_a = decode_final_latent(torch, pipe_shared, lat_a)
        ref_b = decode_final_latent(torch, pipe_shared, lat_b)
        dec_a = ChunkedDecode(torch, pipe_shared, lat_a)
        dec_b = ChunkedDecode(torch, pipe_shared, lat_b)
        for _ in range(3):
            dec_a.advance(1)
            dec_b.advance(1)
        np.testing.assert_allclose(dec_a.finish(), ref_a, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(dec_b.finish(), ref_b, rtol=1e-6, atol=1e-6)

        # AsyncDecoder lifecycle on worker stream
        w_vae, s_vae, _ = _make_wan_worker_and_state(steps=1)
        w_vae.pipe = pipe_shared
        s_vae.step_index = 1
        s_vae.latents = lat_a
        async_dec = AsyncDecoder(w_vae)
        try:
            async_dec.submit(s_vae)
            out_video, event = async_dec.collect(wait=True)
            np.testing.assert_allclose(out_video, ref_a, rtol=1e-6, atol=1e-6)
            assert event["kind"] == "decode"
            assert event["asynchronous"] is True
        finally:
            async_dec.close()
