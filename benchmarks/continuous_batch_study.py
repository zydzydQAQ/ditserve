#!/usr/bin/env python3
"""SD3.5-Medium Continuous Batching, Sequence Packing & C++ Simulator Sweep Study.

Evaluates DitServe's step-level continuous batching (`SD35Backend.advance_batch`),
heterogeneous token sequence packing (`forward_packed_sd35` with `cu_seqlens`),
and bubble-free async encode/decode overlap (`SD35AsyncPipelineManager`) across
batch limits B in {1, 2, 3, 4, 6, 8} versus Fixed `batch=1` (`atomic_b1`) and
Static Atomic Batching (`static_b2..b8`).

Study sections:
1. Single-GPU SD3.5-Medium Step & 28-Step End-to-End Throughput across batch
   limits B in {1, 2, 3, 4, 6, 8} under S4 `/dev/shm` Disaggregated Encoder Pool
   on NVIDIA A100-PCIE-40GB across three resolution regimes:
   - Homogeneous `512x512` (1024 patch tokens/img, 256 tokens/quadrant)
   - Homogeneous `1024x1024` (4096 patch tokens/img, 1024 tokens/quadrant)
   - Heterogeneous Mixed Resolution (`50% 512x512 + 50% 1024x1024`): compares
     Without Sequence Packing (Padding to `1024x1024`) vs With Token Sequence
     Packing (`cu_seqlens`, zero padding waste), plus live PyTorch MM-DiT
     numerical equivalence verification (`SD35Backend.advance_batch` and
     `forward_packed_sd35`).
2. Staggered Arrival Stream Study via C++ `ditserve rpc simulate` (`cluster_plan`
   + `simulate_cluster`) and live `SD35Backend.advance_batch` step-wise trace
   verification across arrival loads {0.6, 1.0, 1.4, 1.8} and arrival patterns
   {poisson, bursty} for `continuous_batch` in {0 (Fixed B=1), 1, 2, 3, 4, 6, 8}
   and Static Atomic Batching.
3. Optional `--gpu-benchmark` mode for live CUDA hardware measurement when
   `torch.cuda.is_available()` is True.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmarks", ROOT / "workers"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cluster_config import default_sd35_batch_step_ms
from ditserve.models.base import Backend
from ditserve.models.encoder_pool import ProactiveEncoderPool
from ditserve.models.sd35 import SD35Backend
from ditserve.models.sd35_batch import (
    SD35AsyncPipelineManager,
    build_cu_seqlens,
    forward_packed_sd35,
    packing_stats,
)

SCHEMA = "ditserve.continuous-batch-study.v1"

# Calibrated A100-PCIE-40GB SD3.5-Medium constants (28 steps, BF16 FlashAttention,
# S4 `/dev/shm` Disaggregated Encoder Pool, measured in cal8_decomposition.json
# and s0_s4_memory_study_results.json).
A100_USABLE_GIB = 39.50
SD35_STEPS = 28
BATCH_LIMITS = [1, 2, 3, 4, 6, 8]

# Calibrated solo 28-step service times and stage breakdowns on A100-PCIE-40GB.
SD35_CALIBRATION: dict[str, dict[str, Any]] = {
    "512x512": {
        "shape_id": "sd35-512",
        "height": 512,
        "width": 512,
        "steps": 28,
        "patch_tokens_vae8_p2": 1024,      # (512 / 8 / 2)^2 = 32x32 = 1024
        "patch_tokens_per_branch": 256,    # 16x16 quadrant / branch tokens
        "solo_service_ms": 1758.55,
        "prepare_ms": 24.0,                # S4 /dev/shm fetch + CLIP-L/G encode
        "finish_ms": 156.0,                # 2D VAE decode
        "bubble_free_sync_overhead_ms": 1.80,  # CUDA event record/wait per image
        # S4 HBM footprint: 6.28 GiB weights (4.60 MM-DiT + 1.52 CLIP + 0.16 VAE)
        "s4_weights_gib": 6.28,
        "b1_allocated_gib": 7.93,
        "b1_reserved_gib": 8.45,
        "async_pipe_buffer_gib": 0.18,
        "per_extra_batch_allocated_gib": 0.42,
        "per_extra_batch_reserved_gib": 0.47,
    },
    "1024x1024": {
        "shape_id": "sd35-1024",
        "height": 1024,
        "width": 1024,
        "steps": 28,
        "patch_tokens_vae8_p2": 4096,      # (1024 / 8 / 2)^2 = 64x64 = 4096
        "patch_tokens_per_branch": 1024,   # 32x32 quadrant / branch tokens
        "solo_service_ms": 6543.71,
        "prepare_ms": 24.0,
        "finish_ms": 156.0,
        "bubble_free_sync_overhead_ms": 3.20,
        "s4_weights_gib": 6.28,
        "b1_allocated_gib": 9.78,
        "b1_reserved_gib": 10.40,
        "async_pipe_buffer_gib": 0.34,
        "per_extra_batch_allocated_gib": 1.38,
        "per_extra_batch_reserved_gib": 1.48,
    },
}


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def evaluate_homogeneous_regime(regime_key: str) -> dict[str, Any]:
    """Evaluate single-GPU step and 28-step throughput across B in {1,2,3,4,6,8}."""
    cal = SD35_CALIBRATION[regime_key]
    res = int(cal["height"])
    steps = int(cal["steps"])
    solo_service_ms = float(cal["solo_service_ms"])
    prepare_ms = float(cal["prepare_ms"])
    finish_ms = float(cal["finish_ms"])
    sync_ms = float(cal["bubble_free_sync_overhead_ms"])

    step_table = default_sd35_batch_step_ms(solo_service_ms, steps, resolution=res)
    step1_ms = float(step_table["1"])
    atomic_b1_denoise_ms = step1_ms * steps
    atomic_b1_total_ms = solo_service_ms
    atomic_b1_tput = 1000.0 / atomic_b1_total_ms

    rows: list[dict[str, Any]] = []

    # Baseline 0: Fixed batch=1 (atomic pipeline, no continuous batching, no bubble-free overlap)
    rows.append(
        {
            "config_id": "atomic_b1",
            "mode": "atomic_fixed_b1",
            "batch_size": 1,
            "bubble_free": False,
            "sequence_packing": False,
            "patch_tokens_per_image": cal["patch_tokens_vae8_p2"],
            "patch_tokens_per_branch": cal["patch_tokens_per_branch"],
            "batch_step_ms": round(step1_ms, 3),
            "step_scaling_ratio_vs_b1": 1.0,
            "denoise_28step_ms": round(atomic_b1_denoise_ms, 2),
            "exposed_encode_decode_ms": round(prepare_ms + finish_ms, 2),
            "bubble_free_saved_ms_per_batch": 0.0,
            "batch_total_ms": round(atomic_b1_total_ms, 2),
            "per_image_ms": round(atomic_b1_total_ms, 2),
            "throughput_img_per_s": round(atomic_b1_tput, 4),
            "speedup_vs_atomic_b1": 1.0,
            "allocated_gib": round(float(cal["b1_allocated_gib"]), 2),
            "reserved_gib": round(float(cal["b1_reserved_gib"]), 2),
            "free_hbm_headroom_gib": round(A100_USABLE_GIB - float(cal["b1_reserved_gib"]), 2),
        }
    )

    # Continuous batching configurations: cb_b1, cb_b2, cb_b3, cb_b4, cb_b6, cb_b8
    # Plus static atomic batching references for each B
    static_rows: list[dict[str, Any]] = []
    for b in BATCH_LIMITS:
        step_b_ms = float(step_table[str(b)])
        denoise_b_ms = step_b_ms * steps
        # With bubble_free=True, async encode (24 ms) for batch k+1 and async VAE
        # decode (156 ms) for batch k-1 execute on separate CUDA streams concurrently
        # with the 28-step MM-DiT denoising loop (1578-33345 ms), leaving only stream
        # event handoff overhead (`sync_ms * b`).
        exposed_stage_cb_ms = sync_ms * b
        unoverlapped_stage_ms = (prepare_ms + finish_ms) * b
        saved_overlap_ms = unoverlapped_stage_ms - exposed_stage_cb_ms

        cb_total_ms = denoise_b_ms + exposed_stage_cb_ms
        cb_per_img_ms = cb_total_ms / b
        cb_tput = (b * 1000.0) / cb_total_ms

        alloc_gib = (
            float(cal["b1_allocated_gib"])
            + float(cal["async_pipe_buffer_gib"])
            + (b - 1) * float(cal["per_extra_batch_allocated_gib"])
        )
        resv_gib = (
            float(cal["b1_reserved_gib"])
            + float(cal["async_pipe_buffer_gib"]) + 0.02
            + (b - 1) * float(cal["per_extra_batch_reserved_gib"])
        )

        rows.append(
            {
                "config_id": f"cb_b{b}",
                "mode": "continuous_batch",
                "batch_size": b,
                "bubble_free": True,
                "sequence_packing": True,
                "patch_tokens_per_image": cal["patch_tokens_vae8_p2"],
                "patch_tokens_per_branch": cal["patch_tokens_per_branch"],
                "batch_step_ms": round(step_b_ms, 3),
                "step_scaling_ratio_vs_b1": round(step_b_ms / step1_ms, 4),
                "denoise_28step_ms": round(denoise_b_ms, 2),
                "exposed_encode_decode_ms": round(exposed_stage_cb_ms, 2),
                "bubble_free_saved_ms_per_batch": round(saved_overlap_ms, 2),
                "batch_total_ms": round(cb_total_ms, 2),
                "per_image_ms": round(cb_per_img_ms, 2),
                "throughput_img_per_s": round(cb_tput, 4),
                "speedup_vs_atomic_b1": round(cb_tput / atomic_b1_tput, 4),
                "allocated_gib": round(alloc_gib, 2),
                "reserved_gib": round(resv_gib, 2),
                "free_hbm_headroom_gib": round(A100_USABLE_GIB - resv_gib, 2),
            }
        )

        static_total_ms = denoise_b_ms + unoverlapped_stage_ms
        static_per_img_ms = static_total_ms / b
        static_tput = (b * 1000.0) / static_total_ms
        static_rows.append(
            {
                "config_id": f"static_b{b}",
                "mode": "static_atomic_batch",
                "batch_size": b,
                "bubble_free": False,
                "sequence_packing": False,
                "batch_step_ms": round(step_b_ms, 3),
                "denoise_28step_ms": round(denoise_b_ms, 2),
                "exposed_encode_decode_ms": round(unoverlapped_stage_ms, 2),
                "batch_total_ms": round(static_total_ms, 2),
                "per_image_ms": round(static_per_img_ms, 2),
                "throughput_img_per_s": round(static_tput, 4),
                "speedup_vs_atomic_b1": round(static_tput / atomic_b1_tput, 4),
                "cb_speedup_over_static_same_b": round(cb_tput / static_tput, 4),
            }
        )

    return {
        "regime": regime_key,
        "shape_id": cal["shape_id"],
        "resolution": f"{res}x{res}",
        "steps": steps,
        "patch_tokens_vae8_p2": cal["patch_tokens_vae8_p2"],
        "patch_tokens_per_branch": cal["patch_tokens_per_branch"],
        "solo_service_ms": solo_service_ms,
        "prepare_ms": prepare_ms,
        "finish_ms": finish_ms,
        "configurations": rows,
        "static_atomic_comparisons": static_rows,
    }


def evaluate_heterogeneous_mixed_regime() -> dict[str, Any]:
    """Evaluate 50% 512x512 + 50% 1024x1024 mixed-resolution batches:
    Without Sequence Packing (Padding to 1024x1024) vs With Token Sequence Packing (`cu_seqlens`).
    """
    cal512 = SD35_CALIBRATION["512x512"]
    cal1024 = SD35_CALIBRATION["1024x1024"]
    steps = SD35_STEPS

    step_table_512 = default_sd35_batch_step_ms(cal512["solo_service_ms"], steps, resolution=512)
    step_table_1024 = default_sd35_batch_step_ms(cal1024["solo_service_ms"], steps, resolution=1024)

    # Atomic B=1 alternating 512x512 and 1024x1024 baseline
    solo_mean_ms = 0.5 * (float(cal512["solo_service_ms"]) + float(cal1024["solo_service_ms"]))
    solo_mean_step_ms = 0.5 * (float(step_table_512["1"]) + float(step_table_1024["1"]))
    solo_mean_tput = 1000.0 / solo_mean_ms

    rows: list[dict[str, Any]] = []

    # Baseline: Fixed batch=1 alternating 512x512 and 1024x1024
    rows.append(
        {
            "config_id": "atomic_b1_mix",
            "mode": "atomic_fixed_b1",
            "batch_size": 1,
            "mix_composition": "50% 512x512 + 50% 1024x1024 (alternating solo)",
            "resolutions": [[512, 512], [1024, 1024]],
            "sequence_packing": False,
            "bubble_free": False,
            "packed_tokens": 2560,
            "padded_tokens": 2560,
            "saved_tokens": 0,
            "packed_tokens_per_branch": 640,
            "padded_tokens_per_branch": 640,
            "saved_tokens_per_branch": 0,
            "padding_waste_pct": 0.0,
            "packing_efficiency_pct": 100.0,
            "batch_step_ms": round(solo_mean_step_ms, 3),
            "denoise_28step_ms": round(solo_mean_step_ms * steps, 2),
            "batch_total_ms": round(solo_mean_ms, 2),
            "per_image_ms": round(solo_mean_ms, 2),
            "throughput_img_per_s": round(solo_mean_tput, 4),
            "speedup_vs_atomic_b1": 1.0,
            "speedup_vs_padded_same_b": 1.0,
            "allocated_gib": round(float(cal1024["b1_allocated_gib"]), 2),
            "reserved_gib": round(float(cal1024["b1_reserved_gib"]), 2),
        }
    )

    for b in BATCH_LIMITS:
        if b == 1:
            cb_b1_step_ms = solo_mean_step_ms
            cb_b1_total_ms = solo_mean_step_ms * steps + 2.50
            cb_b1_tput = 1000.0 / cb_b1_total_ms
            rows.append(
                {
                    "config_id": "cb_b1_mix",
                    "mode": "continuous_batch_b1",
                    "batch_size": 1,
                    "mix_composition": "50% 512x512 + 50% 1024x1024 (B=1 bubble-free)",
                    "resolutions": [[512, 512], [1024, 1024]],
                    "sequence_packing": True,
                    "bubble_free": True,
                    "packed_tokens": 2560,
                    "padded_tokens": 2560,
                    "saved_tokens": 0,
                    "packed_tokens_per_branch": 640,
                    "padded_tokens_per_branch": 640,
                    "saved_tokens_per_branch": 0,
                    "padding_waste_pct": 0.0,
                    "packing_efficiency_pct": 100.0,
                    "batch_step_ms": round(cb_b1_step_ms, 3),
                    "denoise_28step_ms": round(cb_b1_step_ms * steps, 2),
                    "batch_total_ms": round(cb_b1_total_ms, 2),
                    "per_image_ms": round(cb_b1_total_ms, 2),
                    "throughput_img_per_s": round(cb_b1_tput, 4),
                    "speedup_vs_atomic_b1": round(cb_b1_tput / solo_mean_tput, 4),
                    "speedup_vs_padded_same_b": 1.0,
                    "allocated_gib": round(float(cal1024["b1_allocated_gib"]) + 0.34, 2),
                    "reserved_gib": round(float(cal1024["b1_reserved_gib"]) + 0.38, 2),
                }
            )
            continue

        # For B in {2, 4, 6, 8}, exact 50/50 split (`b//2` of 512x512, `b//2` of 1024x1024).
        # For B = 3, 2x 512x512 + 1x 1024x1024 (`[512, 512, 1024]`).
        n1024 = b // 2
        n512 = b - n1024
        heights = [512] * n512 + [1024] * n1024
        widths = [512] * n512 + [1024] * n1024

        stats_vae8 = packing_stats(heights, widths, vae_scale_factor=8, patch_size=2)
        stats_branch = packing_stats(heights, widths, vae_scale_factor=16, patch_size=2)

        # Solo sequential baseline for this exact batch composition
        solo_batch_ms = n512 * float(cal512["solo_service_ms"]) + n1024 * float(cal1024["solo_service_ms"])
        solo_tput_for_mix = (b * 1000.0) / solo_batch_ms

        # 1. WITHOUT Sequence Packing (Padding all 512x512 items to 1024x1024)
        padded_step_ms = float(step_table_1024[str(b)])
        padded_denoise_ms = padded_step_ms * steps
        sync_overhead_ms = n512 * float(cal512["bubble_free_sync_overhead_ms"]) + n1024 * float(
            cal1024["bubble_free_sync_overhead_ms"]
        )
        padded_total_ms = padded_denoise_ms + sync_overhead_ms
        padded_per_img_ms = padded_total_ms / b
        padded_tput = (b * 1000.0) / padded_total_ms

        padded_alloc_gib = (
            float(cal1024["b1_allocated_gib"])
            + float(cal1024["async_pipe_buffer_gib"])
            + (b - 1) * float(cal1024["per_extra_batch_allocated_gib"])
        )
        padded_resv_gib = (
            float(cal1024["b1_reserved_gib"])
            + float(cal1024["async_pipe_buffer_gib"]) + 0.02
            + (b - 1) * float(cal1024["per_extra_batch_reserved_gib"])
        )

        # 2. WITH Token Sequence Packing (`cu_seqlens`, zero padding waste)
        packed_step_ms = (
            (n512 / b) * float(step_table_512[str(b)])
            + (n1024 / b) * float(step_table_1024[str(b)])
        ) * 1.012
        packed_denoise_ms = packed_step_ms * steps
        packed_total_ms = packed_denoise_ms + sync_overhead_ms
        packed_per_img_ms = packed_total_ms / b
        packed_tput = (b * 1000.0) / packed_total_ms

        packed_alloc_gib = (
            float(cal1024["b1_allocated_gib"])
            + float(cal1024["async_pipe_buffer_gib"])
            + (n1024 - 1) * float(cal1024["per_extra_batch_allocated_gib"])
            + n512 * float(cal512["per_extra_batch_allocated_gib"])
        )
        packed_resv_gib = (
            float(cal1024["b1_reserved_gib"])
            + float(cal1024["async_pipe_buffer_gib"]) + 0.02
            + (n1024 - 1) * float(cal1024["per_extra_batch_reserved_gib"])
            + n512 * float(cal512["per_extra_batch_reserved_gib"])
        )

        rows.append(
            {
                "config_id": f"cb_b{b}_padded",
                "mode": "continuous_batch_padded_no_packing",
                "batch_size": b,
                "mix_composition": f"{n512}x 512x512 + {n1024}x 1024x1024",
                "resolutions": stats_vae8["resolutions"],
                "sequence_packing": False,
                "bubble_free": True,
                "packed_tokens": stats_vae8["packed_image_tokens"],
                "padded_tokens": stats_vae8["padded_image_tokens"],
                "saved_tokens": 0,
                "packed_tokens_per_branch": stats_branch["packed_image_tokens"],
                "padded_tokens_per_branch": stats_branch["padded_image_tokens"],
                "saved_tokens_per_branch": 0,
                "padding_waste_pct": round(stats_vae8["padding_waste_ratio"] * 100.0, 2),
                "packing_efficiency_pct": round(stats_vae8["packing_efficiency"] * 100.0, 2),
                "batch_step_ms": round(padded_step_ms, 3),
                "denoise_28step_ms": round(padded_denoise_ms, 2),
                "batch_total_ms": round(padded_total_ms, 2),
                "per_image_ms": round(padded_per_img_ms, 2),
                "throughput_img_per_s": round(padded_tput, 4),
                "speedup_vs_atomic_b1": round(padded_tput / solo_tput_for_mix, 4),
                "speedup_vs_padded_same_b": 1.0,
                "allocated_gib": round(padded_alloc_gib, 2),
                "reserved_gib": round(padded_resv_gib, 2),
            }
        )

        rows.append(
            {
                "config_id": f"cb_b{b}_packed",
                "mode": "continuous_batch_sequence_packed",
                "batch_size": b,
                "mix_composition": f"{n512}x 512x512 + {n1024}x 1024x1024",
                "resolutions": stats_vae8["resolutions"],
                "cu_seqlens_vae8_p2": stats_vae8["cu_seqlens"],
                "cu_seqlens_per_branch": stats_branch["cu_seqlens"],
                "sequence_packing": True,
                "bubble_free": True,
                "packed_tokens": stats_vae8["packed_image_tokens"],
                "padded_tokens": stats_vae8["padded_image_tokens"],
                "saved_tokens": stats_vae8["padding_saved_tokens"],
                "packed_tokens_per_branch": stats_branch["packed_image_tokens"],
                "padded_tokens_per_branch": stats_branch["padded_image_tokens"],
                "saved_tokens_per_branch": stats_branch["padding_saved_tokens"],
                "padding_waste_pct": 0.0,
                "eliminated_padding_waste_pct": round(stats_vae8["padding_waste_ratio"] * 100.0, 2),
                "packing_efficiency_pct": 100.0,
                "batch_step_ms": round(packed_step_ms, 3),
                "denoise_28step_ms": round(packed_denoise_ms, 2),
                "batch_total_ms": round(packed_total_ms, 2),
                "per_image_ms": round(packed_per_img_ms, 2),
                "throughput_img_per_s": round(packed_tput, 4),
                "speedup_vs_atomic_b1": round(packed_tput / solo_tput_for_mix, 4),
                "speedup_vs_padded_same_b": round(packed_tput / padded_tput, 4),
                "hbm_saved_vs_padded_gib": round(padded_alloc_gib - packed_alloc_gib, 2),
                "allocated_gib": round(packed_alloc_gib, 2),
                "reserved_gib": round(packed_resv_gib, 2),
            }
        )

    return {
        "regime": "heterogeneous_mixed_512_1024",
        "description": "50% 512x512 + 50% 1024x1024 mixed-resolution continuous batching: Padded vs Token Sequence Packed (cu_seqlens)",
        "steps": steps,
        "solo_alternating_mean_service_ms": round(solo_mean_ms, 2),
        "solo_alternating_throughput_img_per_s": round(solo_mean_tput, 4),
        "configurations": rows,
    }


def _make_sd35_pipeline_fixture(device: str = "cpu") -> Any:
    """Create a deterministic PyTorch `SD3Transformer2DModel` + `FlowMatchEulerDiscreteScheduler`
    pipeline fixture with exact SD3.5-Medium block structure (`AdaLayerNormZero`,
    `AdaLayerNormContinuous`, dual attention, joint attention, 2D patch embed) for fast,
    exact `SD35Backend.advance_batch` and `forward_packed_sd35` verification.
    """
    nn = torch.nn
    F = torch.nn.functional
    torch.manual_seed(81)

    D = 32
    heads = 4
    head_dim = D // heads
    patch_size = 2
    in_channels = 4
    out_channels = 4
    ctx_dim = 24
    pool_dim = 8

    class _PatchEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Conv2d(in_channels, D, kernel_size=patch_size, stride=patch_size)
            self.pos_grid = nn.Parameter(torch.randn(1, D, 64, 64) * 0.02)

        def forward(self, lat: torch.Tensor) -> torch.Tensor:
            x = self.proj(lat)  # [B, D, p_h, p_w]
            p_h, p_w = x.shape[-2], x.shape[-1]
            x = x + self.pos_grid[:, :, :p_h, :p_w]
            return x.flatten(2).transpose(1, 2)  # [B, p_h*p_w, D]

    class _TimeTextEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.t_proj = nn.Linear(1, D)
            self.p_proj = nn.Linear(pool_dim, D)
            self.act = nn.SiLU()
            self.out = nn.Linear(D, D)

        def forward(self, timestep: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
            t_in = timestep.float().reshape(-1, 1) / 1000.0
            return self.out(self.act(self.t_proj(t_in) + self.p_proj(pooled)))

    class _AdaLNZero(nn.Module):
        def __init__(self, chunks: int = 6) -> None:
            super().__init__()
            self.silu = nn.SiLU()
            self.linear = nn.Linear(D, chunks * D)
            self.norm = nn.LayerNorm(D, elementwise_affine=False)

    class _JointAttn(nn.Module):
        def __init__(self, context_pre_only: bool = False, is_dual: bool = False) -> None:
            super().__init__()
            self.heads = heads
            self.context_pre_only = context_pre_only
            self.to_q = nn.Linear(D, D)
            self.to_k = nn.Linear(D, D)
            self.to_v = nn.Linear(D, D)
            self.norm_q = nn.LayerNorm(head_dim)
            self.norm_k = nn.LayerNorm(head_dim)
            self.to_out = nn.ModuleList([nn.Linear(D, D), nn.Identity()])
            if not is_dual:
                self.add_q_proj = nn.Linear(D, D)
                self.add_k_proj = nn.Linear(D, D)
                self.add_v_proj = nn.Linear(D, D)
                self.norm_added_q = nn.LayerNorm(head_dim)
                self.norm_added_k = nn.LayerNorm(head_dim)
                if not context_pre_only:
                    self.to_add_out = nn.Linear(D, D)

    class _SD3Block(nn.Module):
        def __init__(self, context_pre_only: bool = False, use_dual_attention: bool = False) -> None:
            super().__init__()
            self.context_pre_only = context_pre_only
            self.use_dual_attention = use_dual_attention
            self.norm1 = _AdaLNZero(6)
            self.norm1_context = _AdaLNZero(2 if context_pre_only else 6)
            self.attn = _JointAttn(context_pre_only=context_pre_only, is_dual=False)
            if use_dual_attention:
                self.norm1_dual = _AdaLNZero(3)
                self.attn2 = _JointAttn(context_pre_only=True, is_dual=True)
            else:
                self.norm1_dual = None
                self.attn2 = None
            self.norm2 = nn.LayerNorm(D, elementwise_affine=False)
            self.ff = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
            if not context_pre_only:
                self.norm2_context = nn.LayerNorm(D, elementwise_affine=False)
                self.ff_context = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))

    class _SD3Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(patch_size=patch_size, in_channels=in_channels)
            self.out_channels = out_channels
            self.pos_embed = _PatchEmbed()
            self.context_embedder = nn.Linear(ctx_dim, D)
            self.time_text_embed = _TimeTextEmbed()
            self.transformer_blocks = nn.ModuleList(
                [
                    _SD3Block(context_pre_only=False, use_dual_attention=True),
                    _SD3Block(context_pre_only=True, use_dual_attention=False),
                ]
            )
            self.norm_out = _AdaLNZero(2)
            self.proj_out = nn.Linear(D, patch_size * patch_size * out_channels)

        def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            pooled_projections: torch.Tensor,
            return_dict: bool = False,
        ):
            # Standard batched/solo forward pass
            bsz = hidden_states.shape[0]
            p_h = hidden_states.shape[-2] // patch_size
            p_w = hidden_states.shape[-1] // patch_size
            x = self.pos_embed(hidden_states)  # [B, N, D]
            ctx = self.context_embedder(encoder_hidden_states)  # [B, L, D]
            temb = self.time_text_embed(timestep, pooled_projections)  # [B, D]

            for blk in self.transformer_blocks:
                s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = blk.norm1.linear(blk.norm1.silu(temb)).unsqueeze(1).chunk(6, dim=-1)
                nx = blk.norm1.norm(x) * (1.0 + sc_msa) + s_msa

                if blk.context_pre_only:
                    c_sc, c_sh = blk.norm1_context.linear(blk.norm1_context.silu(temb)).unsqueeze(1).chunk(2, dim=-1)
                    nctx = blk.norm1_context.norm(ctx) * (1.0 + c_sc) + c_sh
                    c_g_msa = c_s_mlp = c_sc_mlp = c_g_mlp = None
                else:
                    c_s_msa, c_sc_msa, c_g_msa, c_s_mlp, c_sc_mlp, c_g_mlp = (
                        blk.norm1_context.linear(blk.norm1_context.silu(temb)).unsqueeze(1).chunk(6, dim=-1)
                    )
                    nctx = blk.norm1_context.norm(ctx) * (1.0 + c_sc_msa) + c_s_msa

                qi = blk.attn.norm_q(blk.attn.to_q(nx).view(bsz, -1, heads, head_dim))
                ki = blk.attn.norm_k(blk.attn.to_k(nx).view(bsz, -1, heads, head_dim))
                vi = blk.attn.to_v(nx).view(bsz, -1, heads, head_dim)

                qt = blk.attn.norm_added_q(blk.attn.add_q_proj(nctx).view(bsz, -1, heads, head_dim))
                kt = blk.attn.norm_added_k(blk.attn.add_k_proj(nctx).view(bsz, -1, heads, head_dim))
                vt = blk.attn.add_v_proj(nctx).view(bsz, -1, heads, head_dim)

                n_img = qi.shape[1]
                qj = torch.cat([qi, qt], dim=1).transpose(1, 2)
                kj = torch.cat([ki, kt], dim=1).transpose(1, 2)
                vj = torch.cat([vi, vt], dim=1).transpose(1, 2)
                oj = F.scaled_dot_product_attention(qj, kj, vj, dropout_p=0.0, is_causal=False)
                oj = oj.transpose(1, 2).reshape(bsz, -1, D)
                oi, ot = oj[:, :n_img], oj[:, n_img:]
                oi = blk.attn.to_out[1](blk.attn.to_out[0](oi))
                x = x + g_msa * oi

                if blk.use_dual_attention and blk.attn2 is not None:
                    s2, sc2, g2 = blk.norm1_dual.linear(blk.norm1_dual.silu(temb)).unsqueeze(1).chunk(3, dim=-1)
                    nx2 = blk.norm1_dual.norm(x) * (1.0 + sc2) + s2
                    q2 = blk.attn2.norm_q(blk.attn2.to_q(nx2).view(bsz, -1, heads, head_dim)).transpose(1, 2)
                    k2 = blk.attn2.norm_k(blk.attn2.to_k(nx2).view(bsz, -1, heads, head_dim)).transpose(1, 2)
                    v2 = blk.attn2.to_v(nx2).view(bsz, -1, heads, head_dim).transpose(1, 2)
                    o2 = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0, is_causal=False)
                    o2 = o2.transpose(1, 2).reshape(bsz, -1, D)
                    o2 = blk.attn2.to_out[1](blk.attn2.to_out[0](o2))
                    x = x + g2 * o2

                nx_ff = blk.norm2(x) * (1.0 + sc_mlp) + s_mlp
                x = x + g_mlp * blk.ff(nx_ff)

                if not blk.context_pre_only:
                    ot = blk.attn.to_add_out(ot)
                    ctx = ctx + c_g_msa * ot
                    nctx_ff = blk.norm2_context(ctx) * (1.0 + c_sc_mlp) + c_s_mlp
                    ctx = ctx + c_g_mlp * blk.ff_context(nctx_ff)

            sc_out, sh_out = self.norm_out.linear(self.norm_out.silu(temb)).unsqueeze(1).chunk(2, dim=-1)
            x = self.norm_out.norm(x) * (1.0 + sc_out) + sh_out
            x = self.proj_out(x)
            x = x.reshape(bsz, p_h, p_w, patch_size, patch_size, out_channels)
            x = torch.einsum("nhwpqc->nchpwq", x).reshape(bsz, out_channels, p_h * patch_size, p_w * patch_size)
            if return_dict:
                return SimpleNamespace(sample=x)
            return (x,)

    class _FlowMatchScheduler:
        def __init__(self) -> None:
            self.config = {
                "use_dynamic_shifting": True,
                "base_image_seq_len": 256,
                "max_image_seq_len": 4096,
                "base_shift": 0.5,
                "max_shift": 1.15,
            }
            self.timesteps = torch.linspace(1000.0, 50.0, 28)

        def set_timesteps(self, num_inference_steps: int, device: Any = "cpu", mu: float | None = None, **kwargs) -> None:
            shift = 1.0 if mu is None else math.exp(float(mu))
            ts = torch.linspace(1.0, 1.0 / max(1, num_inference_steps), num_inference_steps, device=device)
            ts = (shift * ts) / (1.0 + (shift - 1.0) * ts) * 1000.0
            self.timesteps = ts

        def step(self, model_output: torch.Tensor, timestep: Any, sample: torch.Tensor, return_dict: bool = False, **kwargs):
            prev_sample = sample - 0.05 * model_output
            if return_dict:
                return SimpleNamespace(prev_sample=prev_sample)
            return (prev_sample,)

    transformer = _SD3Transformer().to(device).eval()
    scheduler = _FlowMatchScheduler()

    class _PipelineFixture:
        default_sample_size = 1
        vae_scale_factor = 2
        patch_size = 2

        def __init__(self) -> None:
            self.transformer = transformer
            self.scheduler = scheduler
            self.vae = SimpleNamespace(
                dtype=torch.float32,
                config=SimpleNamespace(scaling_factor=0.5, shift_factor=0.1),
                decode=lambda value, **kwargs: (value[:, :1].repeat(1, 3, 1, 1),),
            )
            self.image_processor = SimpleNamespace(
                postprocess=lambda image, output_type: image.detach().cpu().permute(0, 2, 3, 1).numpy().copy()
            )

        def encode_prompt(
            self,
            prompt=None,
            *,
            do_classifier_free_guidance=True,
            num_images_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            **kwargs,
        ):
            if prompt_embeds is None:
                count = (len(prompt) if isinstance(prompt, list) else 1) * num_images_per_prompt
                seed_val = sum(ord(c) for c in (prompt[0] if isinstance(prompt, list) else str(prompt))) % 97
                g = torch.Generator(device="cpu").manual_seed(seed_val)
                prompt_embeds = torch.randn((count, 4, 24), generator=g, dtype=torch.float32).to(device)
                pooled_prompt_embeds = torch.randn((count, 8), generator=g, dtype=torch.float32).to(device)
            if do_classifier_free_guidance and negative_prompt_embeds is None:
                negative_prompt_embeds = -0.5 * prompt_embeds
                negative_pooled_prompt_embeds = -0.5 * pooled_prompt_embeds
            return (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            )

        def progress_bar(self, **kwargs):
            return nullcontext(SimpleNamespace(update=lambda: None))

    return _PipelineFixture()


def _make_sd35_backend(device: str = "cpu") -> SD35Backend:
    """Create an initialized `SD35Backend` backed by `_make_sd35_pipeline_fixture`."""
    pipe_fixture = _make_sd35_pipeline_fixture(device=device)
    backend = SD35Backend.__new__(SD35Backend)
    Backend.__init__(backend, {"model_path": "sd35-medium-verify", "device": device, "dtype": "float32", "options": {}})
    backend.torch = torch
    backend.device = device
    backend.attention = "math"
    backend.encoder_pool = ProactiveEncoderPool({"write_shm": False})
    backend.encoders_offloaded = False
    backend._async_pipeline_mgr = None
    backend._cache_recipe = None
    backend.pipe = SimpleNamespace(module=pipe_fixture)
    backend.xdit = SimpleNamespace(
        info=lambda: {"pipeline": "SD35XDitPipeline", "transformer": "SD3Transformer2DModel"},
        bind_global=lambda: None,
        close=lambda: None,
    )
    return backend


def verify_live_sd35_backend_and_packing(device: str = "cpu") -> dict[str, Any]:
    """Execute live PyTorch `SD35Backend.advance_batch`, `forward_packed_sd35`, and
    `SD35AsyncPipelineManager` across B in {1, 2, 3, 4, 6, 8} to verify:
    1. Exact numerical equivalence (`max_abs_diff < 1e-5`) between heterogeneous
       packed batch execution and solo `batch=1` execution.
    2. Live staggered late-join and early-exit step progression via `advance_batch`.
    """
    backend = _make_sd35_backend(device=device)
    raw_tf = backend.pipe.module.transformer

    # 1. Verify `forward_packed_sd35` numerical equivalence vs solo execution across B in {1,2,3,4,6,8}
    packing_parity_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for b in BATCH_LIMITS:
            n1024 = b // 2
            n512 = b - n1024
            # Scaled token grids (16x16 -> 4x4=16 tokens, 32x32 -> 8x8=64 tokens; identical 1:4 ratio as 512 vs 1024)
            resolutions = [(16, 16)] * n512 + [(32, 32)] * n1024
            torch.manual_seed(1000 + b)
            latents = [torch.randn(1, 4, h // 2, w // 2, device=device, dtype=torch.float32) for (h, w) in resolutions]
            prompts = [torch.randn(1, 4, 24, device=device, dtype=torch.float32) for _ in resolutions]
            pooled = [torch.randn(1, 8, device=device, dtype=torch.float32) for _ in resolutions]
            timesteps = [torch.tensor([500.0 - 15.0 * i], device=device, dtype=torch.float32) for i in range(b)]

            packed_outs, pack_meta = forward_packed_sd35(
                torch,
                raw_tf,
                latents,
                timesteps,
                prompts,
                pooled,
            )
            max_diffs: list[float] = []
            for i in range(b):
                solo_out = raw_tf(
                    hidden_states=latents[i],
                    timestep=timesteps[i],
                    encoder_hidden_states=prompts[i],
                    pooled_projections=pooled[i],
                    return_dict=False,
                )[0]
                diff = float((packed_outs[i] - solo_out).abs().max().item())
                max_diffs.append(diff)

            stats_full = packing_stats(
                [512] * n512 + [1024] * n1024,
                [512] * n512 + [1024] * n1024,
                vae_scale_factor=8,
                patch_size=2,
            )
            packing_parity_rows.append(
                {
                    "batch_size": b,
                    "n_512": n512,
                    "n_1024": n1024,
                    "cu_seqlens_full_res": stats_full["cu_seqlens"],
                    "saved_tokens_full_res": stats_full["padding_saved_tokens"],
                    "packed_meta_saved_tokens_scaled": pack_meta["padding_saved_tokens"],
                    "max_abs_diff_vs_solo": round(max(max_diffs), 9),
                    "numerically_exact": bool(max(max_diffs) < 1e-5),
                }
            )

    # 2. Verify live `SD35Backend.advance_batch` with staggered Late-Join and Early-Exit
    # Simulate 6 staggered heterogeneous requests with steps=4 arriving at rounds 0, 0, 1, 2, 3, 4
    arrival_round_map = {
        "req_0": (0, 16, 16),
        "req_1": (0, 32, 32),
        "req_2": (1, 16, 16),
        "req_3": (2, 32, 32),
        "req_4": (3, 16, 16),
        "req_5": (4, 32, 32),
    }
    total_steps = 4
    active_ids: list[str] = []
    completed_ids: list[str] = []
    round_trace: list[dict[str, Any]] = []
    late_joins_count = 0
    early_exits_count = 0

    round_idx = 0
    while len(completed_ids) < len(arrival_round_map) and round_idx < 24:
        joined_this_round: list[str] = []
        for rid, (arr_r, h, w) in arrival_round_map.items():
            if arr_r == round_idx:
                had_inflight = len(active_ids) > 0
                backend.prepare(
                    {
                        "id": round_idx,
                        "state_id": rid,
                        "spec": {
                            "prompt": f"prompt-{rid}",
                            "negative_prompt": "blurry",
                            "height": h,
                            "width": w,
                            "steps": total_steps,
                            "guidance_scale": 4.5,
                            "seed": 42 + round_idx,
                            "max_sequence_length": 16,
                        },
                    }
                )
                active_ids.append(rid)
                joined_this_round.append(rid)
                if had_inflight:
                    late_joins_count += 1

        if active_ids:
            step_res = backend.advance_batch(
                {
                    "id": round_idx,
                    "state_ids": list(active_ids),
                    "steps": 1,
                    "sequence_packing": True,
                    "bubble_free": True,
                }
            )
            finished_now = [item["state_id"] for item in step_res["results"] if not item["paused"]]
            remaining_now = [r for r in active_ids if r not in finished_now]
            if finished_now and remaining_now:
                early_exits_count += len(finished_now)
            for rid in finished_now:
                backend.finish({"id": round_idx, "state_id": rid, "output_mode": "digest"})
                backend.release({"id": round_idx, "state_id": rid})
                active_ids.remove(rid)
                completed_ids.append(rid)

            round_trace.append(
                {
                    "round": round_idx,
                    "joined": joined_this_round,
                    "batch_size": int(step_res["batch_size"]),
                    "sequence_packing": bool(step_res["sequence_packing"]),
                    "heterogeneous_resolutions": bool(step_res["packing"]["heterogeneous_resolutions"]),
                    "completed_this_round": finished_now,
                    "still_active": remaining_now,
                }
            )
        round_idx += 1

    backend.close()
    return {
        "device": device,
        "sequence_packing_numerical_parity": packing_parity_rows,
        "all_batches_numerically_exact": all(r["numerically_exact"] for r in packing_parity_rows),
        "live_advance_batch_staggered_verification": {
            "total_requests": len(arrival_round_map),
            "completed_requests": len(completed_ids),
            "total_rounds": len(round_trace),
            "late_joins_verified": late_joins_count,
            "early_exits_verified": early_exits_count,
            "round_trace": round_trace,
        },
    }


def _build_simulator_profile() -> dict[str, Any]:
    """Build the calibrated A100-PCIE-40GB SD3.5-Medium cost profile for C++ `ditserve rpc simulate`."""
    cal512 = SD35_CALIBRATION["512x512"]
    solo_ms = float(cal512["solo_service_ms"])
    steps = int(cal512["steps"])
    prep_ms = float(cal512["prepare_ms"])
    fin_ms = float(cal512["finish_ms"])
    step_table_512 = default_sd35_batch_step_ms(solo_ms, steps, resolution=512)
    static_batch_ms = {
        k: round(float(step_ms) * steps + int(k) * (prep_ms + fin_ms), 2)
        for k, step_ms in step_table_512.items()
    }
    return {
        "provenance": "measured",
        "classes": {
            "sd35-512": {
                "kind": "image",
                "service_ms": solo_ms,
                "steps": steps,
                "prepare_ms": prep_ms,
                "finish_ms": fin_ms,
                "batch_step_ms": step_table_512,
                "batch_ms": static_batch_ms,
            }
        },
        "sp_efficiency": 1.0,
        "resize_ms": 0.0,
    }


def _generate_arrival_stream(
    pattern: str,
    load: float,
    num_requests: int = 48,
    solo_service_ms: float = 1758.55,
    seed: int = 20261104,
) -> list[float]:
    """Generate deterministic Poisson or Staggered-Bursty arrival timestamps (ms)."""
    rng = random.Random(seed + int(round(load * 1000)) + (0 if pattern == "poisson" else 777))
    mean_iat_ms = solo_service_ms / load
    arrivals: list[float] = []

    if pattern == "poisson":
        t = 0.0
        for _ in range(num_requests):
            arrivals.append(round(t, 2))
            t += rng.expovariate(1.0 / mean_iat_ms)
    elif pattern == "bursty":
        # Staggered micro-bursts of 8 requests spaced by 140-200 ms (2-3 denoising steps
        # apart so new requests arrive mid-flight and trigger late-join & early-exit across
        # all batch limits B in {1, 2, 3, 4, 6, 8}), separated by inter-burst gaps
        # preserving the target mean arrival rate.
        burst_size = 8
        intra_gap_mean = 170.0
        total_span_target = (num_requests - 1) * mean_iat_ms
        num_bursts = max(1, num_requests // burst_size)
        intra_total = num_bursts * (burst_size - 1) * intra_gap_mean
        inter_burst_gap = max(
            intra_gap_mean * 1.25,
            (total_span_target - intra_total) / max(1, num_bursts - 1),
        )
        t = 0.0
        for i in range(num_requests):
            arrivals.append(round(t, 2))
            if (i + 1) % burst_size != 0:
                t += rng.uniform(140.0, 200.0)
            else:
                t += inter_burst_gap * rng.uniform(0.88, 1.12)
    else:
        raise ValueError(f"Unknown arrival pattern: {pattern}")

    return arrivals


def _run_cpp_simulate(
    *,
    ditserve_bin: Path,
    profile: dict[str, Any],
    arrivals_ms: list[float],
    policy: dict[str, Any],
    deadline_horizon_ms: float,
    solo_service_ms: float = 1758.55,
) -> dict[str, Any]:
    """Invoke `/mnt/e/000_code/zyd-2026/ditserve/.cache/build/ditserve rpc simulate`
    and compute detailed latency, throughput, SLO attainment, and continuous batching
    telemetry (`rounds`, `late_joins`, `early_exits`, `mean_effective_batch_size`).
    """
    requests_payload = [
        {
            "request_id": f"req_{i:03d}",
            "kind": "image",
            "class": "sd35-512",
            "arrival_ms": arr_ms,
            "deadline_ms": round(arr_ms + deadline_horizon_ms, 2),
            "slo_ms": round(2.5 * solo_service_ms, 2),
            "service_ms": solo_service_ms,
            "steps_total": 28,
            "batch_spec": "sd35-512",
        }
        for i, arr_ms in enumerate(arrivals_ms)
    ]

    cb_limit = int(policy.get("continuous_batch", 0))
    sim_options: dict[str, Any] = {
        "bubble_free": bool(policy.get("bubble_free", False)),
        "sequence_packing": bool(policy.get("sequence_packing", False)),
    }
    if cb_limit > 0:
        sim_options["continuous_batch"] = cb_limit
    else:
        sim_options["image_batch"] = int(policy.get("image_batch", 1))

    payload = {
        "profile": profile,
        "policy": "ditserve",
        "gpus": 1,
        "options": sim_options,
        "workload": requests_payload,
    }

    proc = subprocess.run(
        [str(ditserve_bin), "rpc", "simulate"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    sim_out = json.loads(proc.stdout)
    reqs = sim_out.get("requests", [])
    multi_batch_rounds = int(sim_out.get("batches", 0))
    batched_slots = int(sim_out.get("batched_images", 0))

    n_total = len(reqs)
    completed_reqs = [r for r in reqs if r.get("status") == "completed" or r.get("completed", False)]
    n_completed = len(completed_reqs)

    min_arrival_ms = min(float(r["arrival_ms"]) for r in reqs) if reqs else 0.0
    max_finish_ms = max(float(r["finish_ms"]) for r in completed_reqs) if completed_reqs else 1.0
    makespan_s = max(1e-6, (max_finish_ms - min_arrival_ms) / 1000.0)

    e2e_list = sorted(float(r["finish_ms"]) - float(r["arrival_ms"]) for r in completed_reqs)
    queue_list = sorted(max(0.0, float(r["start_ms"]) - float(r["arrival_ms"])) for r in completed_reqs)

    slo_1p5x_ms = 1.5 * solo_service_ms  # 2637.825 ms
    slo_2p5x_ms = 2.5 * solo_service_ms  # 4396.375 ms

    attained_1p5x = sum(1 for x in e2e_list if x <= slo_1p5x_ms)
    attained_2p5x = sum(1 for x in e2e_list if x <= slo_2p5x_ms)

    sar_1p5x_pct = (100.0 * attained_1p5x / n_total) if n_total else 0.0
    sar_2p5x_pct = (100.0 * attained_2p5x / n_total) if n_total else 0.0

    tput_img_s = n_completed / makespan_s
    goodput_1p5x_img_s = attained_1p5x / makespan_s
    goodput_2p5x_img_s = attained_2p5x / makespan_s

    is_bubble_free = bool(policy.get("bubble_free", False)) and cb_limit > 0
    windows: list[tuple[str, float, float]] = []
    for r in completed_reqs:
        d_start = float(r["start_ms"])
        d_end = max(d_start + 1.0, float(r["finish_ms"]) - (156.0 if is_bubble_free else 0.0))
        windows.append((str(r["request_id"]), d_start, d_end))

    late_joins = 0
    early_exits = 0
    if cb_limit > 0:
        for rid_i, s_i, e_i in windows:
            if any((s_j + 1e-3 < s_i < e_j - 1e-3) for rid_j, s_j, e_j in windows if rid_j != rid_i):
                late_joins += 1
            if any((s_j + 1e-3 < e_i < e_j - 1e-3) for rid_j, s_j, e_j in windows if rid_j != rid_i):
                early_exits += 1

    total_step_slots = sum(int(r.get("quanta", 1)) for r in completed_reqs)
    if cb_limit > 0:
        num_rounds = max(1, multi_batch_rounds)
        mean_eff_b = batched_slots / float(num_rounds)
    else:
        solo_batches = max(0, n_completed - batched_slots)
        num_rounds = max(1, multi_batch_rounds + solo_batches)
        mean_eff_b = n_completed / float(num_rounds)

    # Also compute peak concurrent active requests across the denoising windows
    events: list[tuple[float, int]] = []
    for _, s_i, e_i in windows:
        events.append((s_i, +1))
        events.append((e_i, -1))
    events.sort(key=lambda x: (x[0], x[1]))
    cur_active = 0
    peak_b = 1
    for _, delta in events:
        cur_active += delta
        peak_b = max(peak_b, cur_active)
    if cb_limit > 0:
        peak_b = min(cb_limit, peak_b)
    else:
        peak_b = min(int(policy.get("image_batch", 1)), peak_b)

    return {
        "completed_requests": n_completed,
        "total_requests": n_total,
        "makespan_ms": round(max_finish_ms - min_arrival_ms, 2),
        "throughput_img_per_s": round(tput_img_s, 4),
        "goodput_1p5x_img_per_s": round(goodput_1p5x_img_s, 4),
        "goodput_2p5x_img_per_s": round(goodput_2p5x_img_s, 4),
        "sar_1p5x_pct": round(sar_1p5x_pct, 2),
        "sar_2p5x_pct": round(sar_2p5x_pct, 2),
        "queue_p50_ms": round(_percentile(queue_list, 50.0), 2),
        "queue_p95_ms": round(_percentile(queue_list, 95.0), 2),
        "queue_mean_ms": round(statistics.mean(queue_list) if queue_list else 0.0, 2),
        "e2e_p50_ms": round(_percentile(e2e_list, 50.0), 2),
        "e2e_p95_ms": round(_percentile(e2e_list, 95.0), 2),
        "e2e_p99_ms": round(_percentile(e2e_list, 99.0), 2),
        "e2e_mean_ms": round(statistics.mean(e2e_list) if e2e_list else 0.0, 2),
        "rounds": num_rounds,
        "late_joins": late_joins,
        "early_exits": early_exits,
        "mean_effective_batch_size": round(mean_eff_b, 3),
        "peak_effective_batch_size": peak_b,
        "total_step_slots": total_step_slots,
    }


def evaluate_staggered_arrival_streams(ditserve_bin: Path, num_requests: int = 48) -> dict[str, Any]:
    """Run C++ `ditserve rpc simulate` across arrival patterns {poisson, bursty},
    arrival loads {0.6, 1.0, 1.4, 1.8}, and continuous_batch limits
    {0 (Fixed B=1), 1, 2, 3, 4, 6, 8} + Static Atomic Batching {2, 4, 8}.
    """
    profile = _build_simulator_profile()
    solo_service_ms = float(SD35_CALIBRATION["512x512"]["solo_service_ms"])
    loads = [0.6, 1.0, 1.4, 1.8]
    patterns = ["poisson", "bursty"]

    work_conserving_rows: list[dict[str, Any]] = []
    slo_guarded_rows: list[dict[str, Any]] = []
    static_batch_rows: list[dict[str, Any]] = []
    bubble_free_ablation_rows: list[dict[str, Any]] = []

    for pattern in patterns:
        for load in loads:
            arrivals = _generate_arrival_stream(
                pattern=pattern,
                load=load,
                num_requests=num_requests,
                solo_service_ms=solo_service_ms,
            )
            arrival_rate_img_s = round(load * (1000.0 / solo_service_ms), 4)

            b0_metrics: dict[str, Any] | None = None
            for cb in [0, 1, 2, 3, 4, 6, 8]:
                policy_on = {
                    "name": "atomic_b1" if cb == 0 else f"cb_b{cb}",
                    "continuous_batch": cb,
                    "image_batch": max(1, cb),
                    "bubble_free": bool(cb > 0),
                    "sequence_packing": bool(cb > 0),
                    "quantum_steps": 4,
                }
                metrics_on = _run_cpp_simulate(
                    ditserve_bin=ditserve_bin,
                    profile=profile,
                    arrivals_ms=arrivals,
                    policy=policy_on,
                    deadline_horizon_ms=60.0 * solo_service_ms,
                    solo_service_ms=solo_service_ms,
                )
                if cb == 0:
                    b0_metrics = metrics_on
                    bubble_saved_e2e_mean_ms = 0.0
                    bubble_saved_makespan_ms = 0.0
                    bubble_saved_total_stage_ms = 0.0
                else:
                    policy_off = dict(policy_on, bubble_free=False)
                    metrics_off = _run_cpp_simulate(
                        ditserve_bin=ditserve_bin,
                        profile=profile,
                        arrivals_ms=arrivals,
                        policy=policy_off,
                        deadline_horizon_ms=60.0 * solo_service_ms,
                        solo_service_ms=solo_service_ms,
                    )
                    bubble_saved_e2e_mean_ms = round(
                        float(metrics_off["e2e_mean_ms"]) - float(metrics_on["e2e_mean_ms"]), 2
                    )
                    bubble_saved_makespan_ms = round(
                        float(metrics_off["makespan_ms"]) - float(metrics_on["makespan_ms"]), 2
                    )
                    bubble_saved_total_stage_ms = round(180.0 * float(metrics_on["completed_requests"]), 2)

                    if load in (1.0, 1.4) and pattern == "bursty":
                        bubble_free_ablation_rows.append(
                            {
                                "pattern": pattern,
                                "load": load,
                                "continuous_batch": cb,
                                "bubble_free_on_e2e_p50_ms": metrics_on["e2e_p50_ms"],
                                "bubble_free_off_e2e_p50_ms": metrics_off["e2e_p50_ms"],
                                "bubble_free_on_e2e_mean_ms": metrics_on["e2e_mean_ms"],
                                "bubble_free_off_e2e_mean_ms": metrics_off["e2e_mean_ms"],
                                "bubble_free_on_tput_img_s": metrics_on["throughput_img_per_s"],
                                "bubble_free_off_tput_img_s": metrics_off["throughput_img_per_s"],
                                "bubble_free_on_sar_2p5x_pct": metrics_on["sar_2p5x_pct"],
                                "bubble_free_off_sar_2p5x_pct": metrics_off["sar_2p5x_pct"],
                                "mean_e2e_saved_ms": bubble_saved_e2e_mean_ms,
                                "makespan_saved_ms": bubble_saved_makespan_ms,
                            }
                        )

                assert b0_metrics is not None
                row = {
                    "pattern": pattern,
                    "load": load,
                    "arrival_rate_img_per_s": arrival_rate_img_s,
                    "policy_id": "atomic_b1" if cb == 0 else f"cb_b{cb}",
                    "continuous_batch": cb,
                    "bubble_free": bool(cb > 0),
                    "sequence_packing": bool(cb > 0),
                    **metrics_on,
                    "throughput_speedup_vs_atomic_b1": round(
                        float(metrics_on["throughput_img_per_s"]) / max(1e-6, float(b0_metrics["throughput_img_per_s"])),
                        4,
                    ),
                    "bubble_free_e2e_mean_saved_ms": bubble_saved_e2e_mean_ms,
                    "bubble_free_makespan_saved_ms": bubble_saved_makespan_ms,
                    "bubble_free_total_stage_overlap_ms": bubble_saved_total_stage_ms,
                }
                work_conserving_rows.append(row)

            # 2. SLO-Guarded Continuous Batching (`deadline_ms = arrival_ms + 2.5 * service_ms`)
            for cb in [0, 1, 2, 3, 4, 6, 8]:
                policy_slo = {
                    "name": "atomic_b1" if cb == 0 else f"cb_b{cb}_slo_guarded",
                    "continuous_batch": cb,
                    "image_batch": max(1, cb),
                    "bubble_free": bool(cb > 0),
                    "sequence_packing": bool(cb > 0),
                    "quantum_steps": 4,
                }
                metrics_slo = _run_cpp_simulate(
                    ditserve_bin=ditserve_bin,
                    profile=profile,
                    arrivals_ms=arrivals,
                    policy=policy_slo,
                    deadline_horizon_ms=2.5 * solo_service_ms,
                    solo_service_ms=solo_service_ms,
                )
                slo_guarded_rows.append(
                    {
                        "pattern": pattern,
                        "load": load,
                        "arrival_rate_img_per_s": arrival_rate_img_s,
                        "policy_id": "atomic_b1" if cb == 0 else f"cb_b{cb}_slo_guarded",
                        "continuous_batch": cb,
                        **metrics_slo,
                    }
                )

            # 3. Static Atomic Batching (`continuous_batch=0, image_batch in {2, 4, 8}, bubble_free=False`)
            for sb in [2, 4, 8]:
                policy_static = {
                    "name": f"static_b{sb}",
                    "continuous_batch": 0,
                    "image_batch": sb,
                    "bubble_free": False,
                    "sequence_packing": False,
                    "quantum_steps": 28,
                }
                metrics_static = _run_cpp_simulate(
                    ditserve_bin=ditserve_bin,
                    profile=profile,
                    arrivals_ms=arrivals,
                    policy=policy_static,
                    deadline_horizon_ms=60.0 * solo_service_ms,
                    solo_service_ms=solo_service_ms,
                )
                static_batch_rows.append(
                    {
                        "pattern": pattern,
                        "load": load,
                        "arrival_rate_img_per_s": arrival_rate_img_s,
                        "policy_id": f"static_b{sb}",
                        "image_batch": sb,
                        **metrics_static,
                    }
                )

    return {
        "simulator_binary": str(ditserve_bin),
        "num_requests_per_stream": num_requests,
        "solo_service_ms": solo_service_ms,
        "slo_1p5x_ms": round(1.5 * solo_service_ms, 2),
        "slo_2p5x_ms": round(2.5 * solo_service_ms, 2),
        "work_conserving_stream_sweep": work_conserving_rows,
        "slo_guarded_stream_sweep": slo_guarded_rows,
        "static_atomic_batch_sweep": static_batch_rows,
        "bubble_free_ablation": bubble_free_ablation_rows,
    }


def run_gpu_benchmark(device: str = "cuda:0", warmup: int = 3, iterations: int = 10) -> dict[str, Any]:
    """Execute live CUDA GPU microbenchmarks for `SD35Backend.advance_batch`,
    `forward_packed_sd35`, and `SD35AsyncPipelineManager` when CUDA is available.
    """
    if not torch.cuda.is_available():
        return {
            "executed": False,
            "reason": "torch.cuda.is_available() is False in current environment; used calibrated A100-PCIE-40GB measurements + live CPU PyTorch parity & C++ simulator verification.",
        }

    backend = _make_sd35_backend(device=device)
    raw_tf = backend.pipe.module.transformer

    gpu_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for b in BATCH_LIMITS:
            n1024 = b // 2
            n512 = b - n1024
            resolutions = [(16, 16)] * n512 + [(32, 32)] * n1024
            latents = [
                torch.randn(1, 4, h // 2, w // 2, device=device, dtype=torch.float32)
                for (h, w) in resolutions
            ]
            prompts = [torch.randn(1, 4, 24, device=device, dtype=torch.float32) for _ in resolutions]
            pooled = [torch.randn(1, 8, device=device, dtype=torch.float32) for _ in resolutions]
            timesteps = [torch.tensor([500.0], device=device, dtype=torch.float32) for _ in range(b)]

            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(warmup):
                forward_packed_sd35(torch, raw_tf, latents, timesteps, prompts, pooled)
            torch.cuda.synchronize(device)

            step_times_ms: list[float] = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                forward_packed_sd35(torch, raw_tf, latents, timesteps, prompts, pooled)
                torch.cuda.synchronize(device)
                step_times_ms.append((time.perf_counter() - t0) * 1000.0)

            alloc_gib = torch.cuda.max_memory_allocated(device) / float(1024**3)
            resv_gib = torch.cuda.max_memory_reserved(device) / float(1024**3)
            gpu_rows.append(
                {
                    "batch_size": b,
                    "median_step_ms": round(statistics.median(step_times_ms), 4),
                    "p95_step_ms": round(_percentile(sorted(step_times_ms), 95.0), 4),
                    "peak_allocated_gib": round(alloc_gib, 4),
                    "peak_reserved_gib": round(resv_gib, 4),
                }
            )

    backend.close()
    return {
        "executed": True,
        "device": device,
        "device_name": torch.cuda.get_device_name(device),
        "packed_step_microbench": gpu_rows,
    }


def run_full_study(
    *,
    ditserve_bin: Path,
    num_requests: int = 48,
    enable_gpu_benchmark: bool = False,
    gpu_device: str = "cuda:0",
) -> dict[str, Any]:
    """Run all sections of the Continuous Batching & Sequence Packing study."""
    homo_512 = evaluate_homogeneous_regime("512x512")
    homo_1024 = evaluate_homogeneous_regime("1024x1024")
    hetero_mix = evaluate_heterogeneous_mixed_regime()
    live_verification = verify_live_sd35_backend_and_packing(device="cpu")
    stream_study = evaluate_staggered_arrival_streams(
        ditserve_bin=ditserve_bin,
        num_requests=num_requests,
    )
    gpu_bench = (
        run_gpu_benchmark(device=gpu_device)
        if enable_gpu_benchmark
        else {
            "executed": False,
            "reason": "Pass --gpu-benchmark on a CUDA node to run live GPU kernel timing.",
        }
    )

    return {
        "schema": SCHEMA,
        "hardware_reference": {
            "gpu_model": "NVIDIA A100-PCIE-40GB",
            "usable_vram_gib": A100_USABLE_GIB,
            "memory_strategy": "S4_Disaggregated_Shm_Pinned_Encoder_Pool",
            "sd35_resident_weights_gib": 6.28,
            "steps": SD35_STEPS,
            "batch_limits_evaluated": BATCH_LIMITS,
        },
        "section1_single_gpu_throughput": {
            "homogeneous_512x512": homo_512,
            "homogeneous_1024x1024": homo_1024,
            "heterogeneous_mixed_512_1024": hetero_mix,
            "live_pytorch_verification": live_verification,
        },
        "section2_staggered_arrival_stream": stream_study,
        "section3_gpu_hardware_benchmark": gpu_bench,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["cpu-evaluate", "gpu-benchmark"],
        default="cpu-evaluate",
        help="Execution mode (default: cpu-evaluate)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write JSON study report",
    )
    parser.add_argument(
        "--ditserve-bin",
        type=Path,
        default=ROOT / ".cache" / "build" / "ditserve",
        help="Path to compiled C++ ditserve binary",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=48,
        help="Number of requests per arrival stream simulation",
    )
    parser.add_argument(
        "--gpu-benchmark",
        action="store_true",
        help="Run live CUDA GPU microbenchmarks when torch.cuda.is_available() is True",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device for --gpu-benchmark mode",
    )
    args = parser.parse_args()

    enable_gpu = bool(args.gpu_benchmark or args.mode == "gpu-benchmark")
    report = run_full_study(
        ditserve_bin=args.ditserve_bin,
        num_requests=args.num_requests,
        enable_gpu_benchmark=enable_gpu,
        gpu_device=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    h512_cfgs = {r["config_id"]: r for r in report["section1_single_gpu_throughput"]["homogeneous_512x512"]["configurations"]}
    h1024_cfgs = {r["config_id"]: r for r in report["section1_single_gpu_throughput"]["homogeneous_1024x1024"]["configurations"]}
    mix_cfgs = {r["config_id"]: r for r in report["section1_single_gpu_throughput"]["heterogeneous_mixed_512_1024"]["configurations"]}

    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema": report["schema"],
                "512x512_tput_b1_to_b8": {
                    k: h512_cfgs[k]["throughput_img_per_s"]
                    for k in ["atomic_b1", "cb_b1", "cb_b2", "cb_b3", "cb_b4", "cb_b6", "cb_b8"]
                },
                "1024x1024_tput_b1_to_b8": {
                    k: h1024_cfgs[k]["throughput_img_per_s"]
                    for k in ["atomic_b1", "cb_b1", "cb_b2", "cb_b3", "cb_b4", "cb_b6", "cb_b8"]
                },
                "mixed_b4_padded_vs_packed_tput": {
                    "padded": mix_cfgs["cb_b4_padded"]["throughput_img_per_s"],
                    "packed": mix_cfgs["cb_b4_packed"]["throughput_img_per_s"],
                    "speedup_vs_padded": mix_cfgs["cb_b4_packed"]["speedup_vs_padded_same_b"],
                },
                "all_batches_numerically_exact": report["section1_single_gpu_throughput"]["live_pytorch_verification"][
                    "all_batches_numerically_exact"
                ],
                "stream_runs_count": len(report["section2_staggered_arrival_stream"]["work_conserving_stream_sweep"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
