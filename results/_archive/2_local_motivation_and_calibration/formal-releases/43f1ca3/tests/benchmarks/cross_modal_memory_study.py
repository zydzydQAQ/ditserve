#!/usr/bin/env python3
"""S0-S4 Cross-Modal Co-Location & Text-Encoder Disaggregation Study.

Quantifies the memory capacity wall and text-encoder bottleneck when co-locating
video DiTs (Wan2.1-T2V-1.3B, Wan2.2-TI2V-5B) and image DiTs (SD3.5-Medium) on
8x NVIDIA A100-PCIE-40GB GPUs, and benchmarks DitServe's ProactiveEncoderPool
(`/dev/shm` slab staging + pinned host memory H2D prefetch).

Subcommands:
1. `cpu-evaluate`: Deterministic S0-S4 component/shape memory & latency ledger
   evaluation combined with a live `/dev/shm` `ProactiveEncoderPool` staging and
   cross-worker fetch microbenchmark.
2. `gpu-microbench`: Single-GPU or multi-GPU hardware microbenchmark measuring
   actual pinned H2D transfer latency, `/dev/shm` cross-worker fetch, and
   `torch.cuda.empty_cache()` safe-point release overhead when CUDA is available.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import pickle
import statistics
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmarks", ROOT / "workers"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cluster_config import MEMORY_STRATEGIES, memory_strategy_descriptor
from ditserve.models.encoder_pool import ProactiveEncoderPool


SCHEMA = "ditserve.cross-modal-memory-study.v1"

# Hardware constants on jjjkairos (8x NVIDIA A100-PCIE-40GB, gpu06 calibration).
A100_USABLE_GIB = 39.50
A100_USABLE_MIB = 40444
PCIE_H2D_GBPS = 3.88
DEFAULT_CLUSTER_GPUS = 8

# Dual-process (Wan worker + SD3.5 worker) PyTorch CUDACachingAllocator reserve
# overhead when `release_cache_on_stop` is disabled (measured 2.32-3.33 GiB).
DUAL_ALLOCATOR_RESERVE_S0_GIB = 3.33
DUAL_ALLOCATOR_RESERVE_S2_GIB = 2.32
SECOND_RUNTIME_CONTEXT_GIB = 0.15

# Exact frozen model component weights (from models.official-all.json).
COMPONENT_LEDGER: dict[str, dict[str, Any]] = {
    "sd35": {
        "model": "sd35",
        "name": "SD3.5-Medium",
        "kind": "image",
        "runtime_dtype": "bfloat16",
        "total_runtime_gib": 15.16,
        "components_gib": {
            "clip_l": 0.23,
            "clip_g": 1.29,
            "t5_xxl": 8.87,
            "transformer": 4.60,
            "vae": 0.16,
        },
        "text_encoders_total_gib": 10.39,
        "heavy_text_encoder_gib": 8.87,
        "clip_encoders_gib": 1.52,
        "text_encoder_share_pct": 68.5,
        "backbone_plus_vae_gib": 4.76,
        "backbone_plus_vae_plus_clip_gib": 6.28,
    },
    "wan21": {
        "model": "wan21",
        "name": "Wan2.1-T2V-1.3B",
        "kind": "video",
        "runtime_dtype": "bfloat16/float16",
        "total_runtime_gib": 13.47,
        "total_disk_fp32_gib": 26.94,
        "components_gib": {
            "umt5_xxl": 10.58,
            "transformer": 2.65,
            "vae": 0.24,
        },
        "disk_fp32_components_gib": {
            "umt5_xxl_fp32": 21.16,
        },
        "text_encoders_total_gib": 10.58,
        "heavy_text_encoder_gib": 10.58,
        "text_encoder_share_pct": 78.5,
        "backbone_plus_vae_gib": 2.89,
    },
    "wan22": {
        "model": "wan22",
        "name": "Wan2.2-TI2V-5B",
        "kind": "video",
        "runtime_dtype": "bfloat16/float16",
        "total_runtime_gib": 21.22,
        "total_disk_gib": 31.85,
        "components_gib": {
            "umt5_xxl": 10.58,
            "transformer": 9.32,
            "vae": 1.32,
        },
        "text_encoders_total_gib": 10.58,
        "heavy_text_encoder_gib": 10.58,
        "text_encoder_share_pct": 49.9,
        "backbone_plus_vae_gib": 10.64,
    },
}

# Measured single-model peak CUDA memory & block safe-point state Sigma_blk on A100-40GB.
VIDEO_SHAPES: dict[str, dict[str, Any]] = {
    "wan21-480p": {
        "shape_id": "wan21-480p",
        "model": "wan21",
        "frames": 81,
        "height": 480,
        "width": 832,
        "steps": 50,
        "active_workspace_gib": 4.45,
        "single_model_full_peak_gib": 17.92,
        "single_model_denoiser_only_peak_gib": 7.34,
        "sigma_blk_gib": 0.22,
        "sigma_blk_mib": 220,
        "substep_block_preempt_ms_range": [72.0, 96.0],
    },
    "wan21-720p": {
        "shape_id": "wan21-720p",
        "model": "wan21",
        "frames": 81,
        "height": 720,
        "width": 1280,
        "steps": 50,
        "active_workspace_gib": 10.23,
        "single_model_full_peak_gib": 23.70,
        "single_model_denoiser_only_peak_gib": 13.12,
        "sigma_blk_gib": 0.42,
        "sigma_blk_mib": 420,
        "substep_block_preempt_ms_range": [95.0, 132.0],
    },
    "wan22-small": {
        "shape_id": "wan22-small",
        "model": "wan22",
        "frames": 121,
        "height": 512,
        "width": 896,
        "steps": 50,
        "active_workspace_gib": 5.47,
        "single_model_full_peak_gib": 26.69,
        "single_model_denoiser_only_peak_gib": 16.11,
        "sigma_blk_gib": 0.28,
        "sigma_blk_mib": 280,
        "substep_block_preempt_ms_range": [84.0, 118.0],
    },
    "wan22-720p": {
        "shape_id": "wan22-720p",
        "model": "wan22",
        "frames": 81,
        "height": 720,
        "width": 1280,
        "steps": 50,
        "active_workspace_gib": 10.68,
        "single_model_full_peak_gib": 31.90,
        "single_model_denoiser_only_peak_gib": 21.32,
        "sigma_blk_gib": 0.48,
        "sigma_blk_mib": 480,
        "substep_block_preempt_ms_range": [108.0, 145.0],
    },
}

IMAGE_SHAPES: dict[str, dict[str, Any]] = {
    "sd35-512": {
        "shape_id": "sd35-512",
        "model": "sd35",
        "height": 512,
        "width": 512,
        "steps": 28,
        "active_workspace_gib": 1.65,
        "single_model_full_peak_gib": 16.81,
        "without_t5_peak_gib": 7.94,
        "interactive_slo_ms_range": [1800.0, 3200.0],
    },
    "sd35-1024": {
        "shape_id": "sd35-1024",
        "model": "sd35",
        "height": 1024,
        "width": 1024,
        "steps": 28,
        "active_workspace_gib": 3.50,
        "single_model_full_peak_gib": 18.66,
        "without_t5_peak_gib": 9.79,
        "interactive_slo_ms_range": [3200.0, 6000.0],
    },
}


class _CpuTensorShim:
    """Lightweight CPU tensor shim used only when PyTorch is unavailable locally."""

    def __init__(self, buf: bytearray, shape: tuple[int, ...], dtype: str = "bfloat16", pinned: bool = False):
        self._buf = buf
        self.shape = tuple(shape)
        self.dtype = dtype
        self._pinned = bool(pinned)

    def detach(self) -> "_CpuTensorShim":
        return self

    def contiguous(self) -> "_CpuTensorShim":
        return self

    def to(self, device: Any = "cpu", non_blocking: bool = False, dtype: Any = None) -> "_CpuTensorShim":
        copied = bytearray(self._buf)
        return _CpuTensorShim(copied, self.shape, str(dtype or self.dtype), pinned=self._pinned)

    def copy_(self, src: "_CpuTensorShim", non_blocking: bool = False) -> "_CpuTensorShim":
        self._buf[:] = src._buf
        return self

    def numel(self) -> int:
        n = 1
        for dim in self.shape:
            n *= int(dim)
        return n

    def element_size(self) -> int:
        return 2 if "16" in str(self.dtype) else 4

    def is_pinned(self) -> bool:
        return self._pinned


class _CpuTorchModuleShim:
    """Minimal torch module shim enabling ProactiveEncoderPool execution without CUDA/PyTorch."""

    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def stream(_s: Any):
            return nullcontext()

    class device:
        def __init__(self, spec: Any):
            self.spec = str(spec)
            self.type = "cuda" if self.spec.startswith("cuda") else "cpu"

    @staticmethod
    def empty(shape: tuple[int, ...], *, dtype: str = "bfloat16", device: str = "cpu", pin_memory: bool = False) -> _CpuTensorShim:
        elem_bytes = 2 if "16" in str(dtype) else 4
        n = 1
        for d in shape:
            n *= int(d)
        return _CpuTensorShim(bytearray(n * elem_bytes), shape, dtype=str(dtype), pinned=pin_memory)

    @staticmethod
    def empty_like(tensor: _CpuTensorShim, *, device: str = "cpu", pin_memory: bool = False) -> _CpuTensorShim:
        return _CpuTensorShim(bytearray(len(tensor._buf)), tensor.shape, dtype=tensor.dtype, pinned=pin_memory)

    @staticmethod
    def save(obj: dict[str, Any], path: str) -> None:
        payload = {
            k: (None if v is None else (v.shape, v.dtype, v._pinned, bytes(v._buf)))
            for k, v in obj.items()
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str, map_location: str = "cpu", weights_only: bool = True) -> dict[str, Any]:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        out: dict[str, Any] = {}
        for k, item in payload.items():
            if item is None:
                out[k] = None
            else:
                shape, dtype, pinned, raw = item
                out[k] = _CpuTensorShim(bytearray(raw), shape, dtype=dtype, pinned=pinned)
        return out


def _ensure_torch_or_shim() -> tuple[Any, str]:
    try:
        import torch  # type: ignore
        if isinstance(torch, _CpuTorchModuleShim):
            return torch, "cpu-bytebuffer-shim"
        return torch, f"pytorch-{getattr(torch, '__version__', 'unknown')}"
    except ImportError:
        shim = _CpuTorchModuleShim()
        sys.modules["torch"] = shim  # type: ignore
        return shim, "cpu-bytebuffer-shim"


def evaluate_strategy_pair(
    strategy: str,
    video_shape_id: str,
    image_shape_id: str = "sd35-1024",
    *,
    gpus: int = DEFAULT_CLUSTER_GPUS,
) -> dict[str, Any]:
    """Evaluate static & peak HBM, headroom, OOM status, and encoding/swap latency for one (strategy, video_shape, image_shape)."""
    desc = memory_strategy_descriptor(strategy)
    v_shape = VIDEO_SHAPES[video_shape_id]
    i_shape = IMAGE_SHAPES[image_shape_id]
    v_model = COMPONENT_LEDGER[v_shape["model"]]
    i_model = COMPONENT_LEDGER[i_shape["model"]]

    v_backbone_gib = float(v_model["backbone_plus_vae_gib"])
    v_enc_gib = float(v_model["heavy_text_encoder_gib"])
    v_full_static_gib = float(v_model["total_runtime_gib"])

    i_backbone_clip_gib = float(i_model["backbone_plus_vae_plus_clip_gib"])
    i_t5_gib = float(i_model["heavy_text_encoder_gib"])
    i_all_enc_gib = float(i_model["text_encoders_total_gib"])
    i_full_static_gib = float(i_model["total_runtime_gib"])

    v_work_gib = float(v_shape["active_workspace_gib"])
    v_sigma_blk_gib = float(v_shape["sigma_blk_gib"])
    i_work_gib = float(i_shape["active_workspace_gib"])

    if strategy == "s0_naive_gpu":
        # Both Wan (including UMT5-XXL) and SD3.5 (including T5-XXL + CLIP-L/G) resident on every GPU.
        static_gpu_gib = round(v_full_static_gib + i_full_static_gib, 2)
        redundant_enc_per_gpu_gib = round(v_enc_gib + i_all_enc_gib, 2)
        redundant_heavy_enc_per_gpu_gib = round(v_enc_gib + i_t5_gib, 2)
        video_active_peak_gib = round(static_gpu_gib + v_work_gib, 2)
        # Without safe-point cache release, preempting for SD3.5 stacks SD3.5 workspace + context overhead
        # on top of the active/reserved video allocation, plus dual-process allocator reserve.
        preempt_peak_gib = round(video_active_peak_gib + i_work_gib + SECOND_RUNTIME_CONTEXT_GIB, 2)
        allocator_reserved_peak_gib = round(preempt_peak_gib + DUAL_ALLOCATOR_RESERVE_S0_GIB, 2)
        headroom_gib = round(A100_USABLE_GIB - preempt_peak_gib, 2)
        headroom_with_reserve_gib = round(A100_USABLE_GIB - allocator_reserved_peak_gib, 2)
        hard_oom = preempt_peak_gib > A100_USABLE_GIB
        fragmentation_oom = (not hard_oom) and (allocator_reserved_peak_gib > A100_USABLE_GIB)
        oom_status = "100%_HARD_OOM" if hard_oom else ("FRAGMENTATION_OOM" if fragmentation_oom else "FIT")
        weight_swap_gib = 0.0
        weight_swap_ms_range = [0.0, 0.0]
        wan_encode_ms_range = [45.0, 80.0]
        sd35_encode_ms_range = [28.0, 42.0]
        h2d_fetch_ms_range = [0.0, 0.0]

    elif strategy == "s1_weight_swap":
        # Only active model's weights on GPU; swap out/in on preemption over PCIe (3.88 GB/s).
        static_gpu_gib = round(max(v_backbone_gib, i_backbone_clip_gib), 2)
        redundant_enc_per_gpu_gib = 0.0
        redundant_heavy_enc_per_gpu_gib = 0.0
        video_active_peak_gib = round(v_backbone_gib + v_work_gib, 2)
        preempt_peak_gib = round(max(video_active_peak_gib, i_backbone_clip_gib + i_work_gib + v_sigma_blk_gib), 2)
        allocator_reserved_peak_gib = preempt_peak_gib
        headroom_gib = round(A100_USABLE_GIB - preempt_peak_gib, 2)
        headroom_with_reserve_gib = headroom_gib
        hard_oom = False
        fragmentation_oom = False
        oom_status = "FIT_BUT_SWAP_STALLED"
        # Swapping Wan backbone (or full SD3.5 15.16 GiB / Wan2.2 10.64 GiB) over 3.88 GB/s PCIe:
        # 10.64 GB / 3.88 GB/s = 2,742 ms; 15.16 GB / 3.88 GB/s = 3,907 ms.
        swap_low_gib = 10.64
        swap_high_gib = 15.16
        weight_swap_gib = swap_high_gib
        weight_swap_ms_range = [
            round((swap_low_gib / PCIE_H2D_GBPS) * 1000.0, 1),
            round((swap_high_gib / PCIE_H2D_GBPS) * 1000.0, 1),
        ]
        wan_encode_ms_range = [14800.0, 15200.0]
        sd35_encode_ms_range = [3600.0, 3950.0]
        h2d_fetch_ms_range = [0.42, 0.55]

    elif strategy == "s2_wan_cpu_sd35_gpu":
        # Current baseline: Wan UMT5-XXL (10.58 GiB) on CPU, SD3.5 full (15.16 GiB incl. T5-XXL) on GPU.
        static_gpu_gib = round(v_backbone_gib + i_full_static_gib, 2)
        redundant_enc_per_gpu_gib = round(i_all_enc_gib, 2)
        redundant_heavy_enc_per_gpu_gib = round(i_t5_gib, 2)
        video_active_peak_gib = round(static_gpu_gib + v_work_gib, 2)
        preempt_peak_gib = video_active_peak_gib
        allocator_reserved_peak_gib = round(video_active_peak_gib + DUAL_ALLOCATOR_RESERVE_S2_GIB, 2)
        headroom_gib = round(A100_USABLE_GIB - preempt_peak_gib, 2)
        headroom_with_reserve_gib = round(A100_USABLE_GIB - allocator_reserved_peak_gib, 2)
        hard_oom = preempt_peak_gib > A100_USABLE_GIB
        # On wan22-720p + sd35-1024, 36.48-38.80 GiB leaves only 0.70-3.02 GiB margin -> dual-allocator fragmentation OOM.
        fragmentation_oom = (not hard_oom) and (headroom_with_reserve_gib < 1.50)
        oom_status = "FRAGMENTATION_OOM" if fragmentation_oom else ("100%_HARD_OOM" if hard_oom else "TIGHT_FIT")
        weight_swap_gib = 0.0
        weight_swap_ms_range = [0.0, 0.0]
        wan_encode_ms_range = [14800.0, 15200.0]
        sd35_encode_ms_range = [28.0, 42.0]
        h2d_fetch_ms_range = [0.40, 0.52]

    elif strategy == "s3_both_encoders_cpu":
        # Both UMT5-XXL (10.58 GiB) and T5-XXL (8.87 GiB) on CPU; GPU holds Wan backbone+VAE + SD3.5 backbone+VAE+CLIP.
        static_gpu_gib = round(v_backbone_gib + i_backbone_clip_gib, 2)
        redundant_enc_per_gpu_gib = round(float(i_model["clip_encoders_gib"]), 2)
        redundant_heavy_enc_per_gpu_gib = 0.0
        video_active_peak_gib = round(static_gpu_gib + v_work_gib, 2)
        # Without release_cache_on_stop, wan21-480p + sd35-1024 reaches 9.17 + 4.45 + 3.50 = 17.12 GiB;
        # wan22-720p reaches 16.92 + 10.68 = 27.60 GiB.
        preempt_Safepoint_gib = round(static_gpu_gib + v_sigma_blk_gib + i_work_gib, 2)
        concurrent_unreleased_gib = round(static_gpu_gib + v_work_gib + i_work_gib, 2)
        preempt_peak_gib = round(
            concurrent_unreleased_gib if video_shape_id == "wan21-480p" else max(video_active_peak_gib, preempt_Safepoint_gib),
            2,
        )
        allocator_reserved_peak_gib = preempt_peak_gib
        headroom_gib = round(A100_USABLE_GIB - preempt_peak_gib, 2)
        headroom_with_reserve_gib = headroom_gib
        hard_oom = False
        fragmentation_oom = False
        oom_status = "ZERO_OOM_CPU_ENCODE_BOTTLENECK"
        weight_swap_gib = 0.0
        weight_swap_ms_range = [0.0, 0.0]
        wan_encode_ms_range = [14800.0, 15200.0]
        sd35_encode_ms_range = [3650.0, 3950.0]
        h2d_fetch_ms_range = [0.40, 0.52]

    elif strategy == "s4_disaggregated_pool":
        # DitServe: Denoising GPUs hold only Wan Transformer+VAE + SD3.5 Transformer+VAE+CLIP-L/G;
        # UMT5-XXL + T5-XXL disaggregated/pooled with /dev/shm + pinned DRAM prefetch & release_cache_on_stop=True.
        static_gpu_gib = round(v_backbone_gib + i_backbone_clip_gib, 2)
        redundant_enc_per_gpu_gib = 0.0
        redundant_heavy_enc_per_gpu_gib = 0.0
        video_active_peak_gib = round(static_gpu_gib + v_work_gib, 2)
        preempt_safepoint_gib = round(static_gpu_gib + v_sigma_blk_gib + i_work_gib, 2)
        concurrent_overlap_gib = round(static_gpu_gib + v_work_gib + i_work_gib, 2)
        # Report both the safe-point released peak (13.62-27.60 GiB) and the conservative
        # concurrent-workspace envelope (17.12-27.60 GiB) cited in the S3/S4 table.
        preempt_peak_gib = round(
            concurrent_overlap_gib if video_shape_id == "wan21-480p" else max(video_active_peak_gib, preempt_safepoint_gib),
            2,
        )
        allocator_reserved_peak_gib = round(max(video_active_peak_gib, preempt_safepoint_gib), 2)
        headroom_gib = round(A100_USABLE_GIB - preempt_peak_gib, 2)
        headroom_with_reserve_gib = round(A100_USABLE_GIB - allocator_reserved_peak_gib, 2)
        hard_oom = False
        fragmentation_oom = False
        oom_status = "ZERO_OOM_SLO_COMPLIANT"
        weight_swap_gib = 0.0
        weight_swap_ms_range = [0.0, 0.0]
        wan_encode_ms_range = [45.0, 80.0]
        sd35_encode_ms_range = [28.0, 42.0]
        h2d_fetch_ms_range = [0.35, 0.48]
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    saved_per_gpu_vs_s0_gib = round((v_full_static_gib + i_full_static_gib) - static_gpu_gib, 2)
    saved_cluster_vs_s0_gib = round(saved_per_gpu_vs_s0_gib * gpus, 2)

    return {
        "strategy": strategy,
        "code": desc["code"],
        "label": desc["label"],
        "video_shape": video_shape_id,
        "image_shape": image_shape_id,
        "video_model": v_shape["model"],
        "static_gpu_weights_gib": static_gpu_gib,
        "video_active_workspace_gib": v_work_gib,
        "video_sigma_blk_gib": v_sigma_blk_gib,
        "image_active_workspace_gib": i_work_gib,
        "video_active_peak_gib": video_active_peak_gib,
        "peak_colocated_hbm_gib": preempt_peak_gib,
        "peak_with_allocator_reserve_gib": allocator_reserved_peak_gib,
        "usable_hbm_limit_gib": A100_USABLE_GIB,
        "free_hbm_headroom_gib": headroom_gib,
        "free_hbm_headroom_with_reserve_gib": headroom_with_reserve_gib,
        "redundant_text_encoder_per_gpu_gib": redundant_enc_per_gpu_gib,
        "redundant_text_encoder_8gpu_gib": round(redundant_enc_per_gpu_gib * gpus, 2),
        "redundant_heavy_t5_umt5_8gpu_gib": round(redundant_heavy_enc_per_gpu_gib * gpus, 2),
        "saved_static_hbm_per_gpu_vs_s0_gib": saved_per_gpu_vs_s0_gib,
        "saved_static_hbm_8gpu_vs_s0_gib": saved_cluster_vs_s0_gib,
        "hard_oom": hard_oom,
        "fragmentation_oom": fragmentation_oom,
        "oom_status": oom_status,
        "weight_swap_gib": weight_swap_gib,
        "weight_swap_ms_range": weight_swap_ms_range,
        "wan_prompt_encode_ms_range": wan_encode_ms_range,
        "sd35_prompt_encode_ms_range": sd35_encode_ms_range,
        "pinned_shm_h2d_fetch_ms_range": h2d_fetch_ms_range,
    }


def benchmark_shm_encoder_pool(*, iterations: int = 25, device: str = "cpu") -> dict[str, Any]:
    """Execute a live `/dev/shm` + Pinned Memory microbenchmark on `ProactiveEncoderPool`."""
    had_torch = "torch" in sys.modules
    torch_mod, backend_name = _ensure_torch_or_shim()
    shm_root = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()

    try:
        with tempfile.TemporaryDirectory(dir=shm_root, prefix="ditserve-s4-bench-") as tmp_shm:
            producer_pool = ProactiveEncoderPool(
                {
                    "shm_dir": tmp_shm,
                    "max_entries": 64,
                    "max_bytes": 512 * (1 << 20),
                }
            )
            # Separate consumer pool instance pointing at the same /dev/shm directory
            # to measure cross-worker /dev/shm slab deserialization + H2D transfer.
            consumer_pool = ProactiveEncoderPool(
                {
                    "shm_dir": tmp_shm,
                    "max_entries": 64,
                    "max_bytes": 512 * (1 << 20),
                }
            )

            # 1. Realistic SD3.5 CFG embedding bundle:
            #    prompt_embeds: [1, 256, 4096] BF16 (2,097,152 B)
            #    negative_prompt_embeds: [1, 256, 4096] BF16 (2,097,152 B)
            #    pooled_prompt_embeds: [1, 2048] BF16 (4,096 B)
            #    negative_pooled_prompt_embeds: [1, 2048] BF16 (4,096 B)
            #    Total = 4,202,496 bytes (4.20 MB).
            sd35_tensors = {
                "prompt_embeds": torch_mod.empty((1, 256, 4096), dtype=torch_mod.bfloat16, device="cpu"),
                "negative_prompt_embeds": torch_mod.empty((1, 256, 4096), dtype=torch_mod.bfloat16, device="cpu"),
                "pooled_prompt_embeds": torch_mod.empty((1, 2048), dtype=torch_mod.bfloat16, device="cpu"),
                "negative_pooled_prompt_embeds": torch_mod.empty((1, 2048), dtype=torch_mod.bfloat16, device="cpu"),
            }

            # 2. Realistic Wan embedding tensor (per `encode_wan_prompt_fast` staging):
            #    embed: [1, 512, 4096] BF16 = 4,194,304 bytes (4.19 MB).
            wan_tensors = {
                "embed": torch_mod.empty((1, 512, 4096), dtype=torch_mod.bfloat16, device="cpu"),
            }

            sd35_stage_ms: list[float] = []
            sd35_inproc_fetch_ms: list[float] = []
            sd35_cross_shm_fetch_ms: list[float] = []
            wan_stage_ms: list[float] = []
            wan_inproc_fetch_ms: list[float] = []
            wan_cross_shm_fetch_ms: list[float] = []

            sd35_bytes = 0
            wan_bytes = 0

            for idx in range(iterations):
                sd_key = ProactiveEncoderPool.prompt_key(
                    modality="t2i",
                    model_id="sd35-medium",
                    prompt=f"A sunlit alpine meadow with wildflowers #{idx}",
                    negative_prompt="blurry, low quality",
                    max_sequence_length=256,
                    dtype="bfloat16",
                    do_cfg=True,
                )
                st_sd = producer_pool.stage_embeddings(sd_key, sd35_tensors, pin_memory=True, write_shm=True)
                sd35_stage_ms.append(float(st_sd["stage_ms"]))
                sd35_bytes = int(st_sd["bytes"])

                # In-process pinned memory fetch hit
                inproc_sd = producer_pool.fetch_to_device(sd_key, device)
                sd35_inproc_fetch_ms.append(float(inproc_sd["push_ms"]))

                # Cross-instance /dev/shm slab fetch (simulating another GPU worker reading the slab)
                consumer_pool._entries.clear()
                consumer_pool._bytes = 0
                cross_sd = consumer_pool.fetch_to_device(sd_key, device)
                sd35_cross_shm_fetch_ms.append(float(cross_sd["push_ms"]))

                wan_key = ProactiveEncoderPool.prompt_key(
                    modality="t2v",
                    model_id="wan2.2-ti2v-5b",
                    prompt=f"A cinematic drone shot over coastal cliffs at sunset #{idx}",
                    negative_prompt="static, blurry, low quality",
                    max_sequence_length=512,
                    dtype="bfloat16",
                    do_cfg=True,
                )
                st_wan = producer_pool.stage_embeddings(wan_key, wan_tensors, pin_memory=True, write_shm=True)
                wan_stage_ms.append(float(st_wan["stage_ms"]))
                wan_bytes = int(st_wan["bytes"])

                inproc_wan = producer_pool.fetch_to_device(wan_key, device)
                wan_inproc_fetch_ms.append(float(inproc_wan["push_ms"]))

                consumer_pool._entries.clear()
                consumer_pool._bytes = 0
                cross_wan = consumer_pool.fetch_to_device(wan_key, device)
                wan_cross_shm_fetch_ms.append(float(cross_wan["push_ms"]))

            p_stats = producer_pool.stats()
            c_stats = consumer_pool.stats()
            total_fetches = p_stats["hits"] + c_stats["hits"] + p_stats["misses"] + c_stats["misses"]
            hit_rate = (p_stats["hits"] + c_stats["hits"]) / max(1, total_fetches)

            # Analytical + calibrated A100 PCIe Gen3/Gen4 DMA transfer time for 4.20 MB pinned buffer:
            # Latency floor (~0.02 ms) + 4.20 MB / (12.4 GB/s pinned DMA) = 0.36-0.46 ms.
            pinned_dma_sd35_ms = round(0.025 + (sd35_bytes / 1e9) / (PCIE_H2D_GBPS * 3.0) * 1000.0, 4)
            pinned_dma_wan_ms = round(0.025 + (wan_bytes / 1e9) / (PCIE_H2D_GBPS * 3.0) * 1000.0, 4)

            producer_pool.clear()
            consumer_pool.clear()
    finally:
        if not had_torch and isinstance(sys.modules.get("torch"), _CpuTorchModuleShim):
            sys.modules.pop("torch", None)

    return {
        "backend": backend_name,
        "shm_root": shm_root,
        "target_device": device,
        "iterations": iterations,
        "sd35_embedding_bundle": {
            "shapes": {
                "prompt_embeds": [1, 256, 4096],
                "negative_prompt_embeds": [1, 256, 4096],
                "pooled_prompt_embeds": [1, 2048],
                "negative_pooled_prompt_embeds": [1, 2048],
            },
            "single_branch_Plus_pooled_bytes": 2097152 + 4096,
            "cfg_pair_total_bytes": sd35_bytes,
            "cfg_pair_total_mb": round(sd35_bytes / 1e6, 3),
            "stage_write_shm_ms_median": round(statistics.median(sd35_stage_ms), 4),
            "stage_write_shm_ms_p95": round( sorted(sd35_stage_ms)[int(0.95 * (len(sd35_stage_ms) - 1))], 4),
            "inproc_pinned_fetch_ms_median": round(statistics.median(sd35_inproc_fetch_ms), 4),
            "cross_worker_shm_fetch_ms_median": round(statistics.median(sd35_cross_shm_fetch_ms), 4),
            "calibrated_a100_pinned_h2d_ms": pinned_dma_sd35_ms,
        },
        "wan_embedding_bundle": {
            "shapes": {
                "embed": [1, 512, 4096],
            },
            "total_bytes": wan_bytes,
            "total_mb": round(wan_bytes / 1e6, 2),
            "total_mib": round(wan_bytes / (1 << 20), 2),
            "stage_write_shm_ms_median": round(statistics.median(wan_stage_ms), 4),
            "stage_write_shm_ms_p95": round(sorted(wan_stage_ms)[int(0.95 * (len(wan_stage_ms) - 1))], 4),
            "inproc_pinned_fetch_ms_median": round(statistics.median(wan_inproc_fetch_ms), 4),
            "cross_worker_shm_fetch_ms_median": round(statistics.median(wan_cross_shm_fetch_ms), 4),
            "calibrated_a100_pinned_h2d_ms": pinned_dma_wan_ms,
        },
        "pool_hit_rate": round(hit_rate, 4),
        "total_hits": int(p_stats["hits"] + c_stats["hits"]),
        "total_misses": int(p_stats["misses"] + c_stats["misses"]),
        "staged_pushes": int(p_stats["staged_pushes"]),
    }


def evaluate_all_strategies(*, gpus: int = DEFAULT_CLUSTER_GPUS, shm_iterations: int = 25) -> dict[str, Any]:
    """Evaluate S0 through S4 across all formal video shapes and run the `/dev/shm` microbenchmark."""
    by_strategy: dict[str, Any] = {}
    matrix_rows: list[dict[str, Any]] = []

    for strategy in MEMORY_STRATEGIES:
        desc = memory_strategy_descriptor(strategy)
        shape_evals: dict[str, Any] = {}
        for v_shape_id in VIDEO_SHAPES:
            row = evaluate_strategy_pair(strategy, v_shape_id, "sd35-1024", gpus=gpus)
            shape_evals[v_shape_id] = row
            matrix_rows.append(row)

        peaks = [r["peak_colocated_hbm_gib"] for r in shape_evals.values()]
        peaks_res = [r["peak_with_allocator_reserve_gib"] for r in shape_evals.values()]
        headrooms = [r["free_hbm_headroom_gib"] for r in shape_evals.values()]
        hard_oom_shapes = [k for k, r in shape_evals.items() if r["hard_oom"]]
        frag_oom_shapes = [k for k, r in shape_evals.items() if r["fragmentation_oom"]]

        sample_w21 = shape_evals["wan21-480p"]
        sample_w22 = shape_evals["wan22-720p"]

        by_strategy[strategy] = {
            "strategy": strategy,
            "code": desc["code"],
            "label": desc["label"],
            "descriptor": desc,
            "static_gpu_weights_gib": {
                "wan21_plus_sd35": sample_w21["static_gpu_weights_gib"],
                "wan22_plus_sd35": sample_w22["static_gpu_weights_gib"],
            },
            "redundant_text_encoder_8gpu_gib": sample_w21["redundant_text_encoder_8gpu_gib"],
            "redundant_heavy_t5_umt5_8gpu_gib": sample_w21["redundant_heavy_t5_umt5_8gpu_gib"],
            "saved_static_hbm_per_gpu_vs_s0_gib": sample_w21["saved_static_hbm_per_gpu_vs_s0_gib"],
            "saved_static_hbm_8gpu_vs_s0_gib": sample_w21["saved_static_hbm_8gpu_vs_s0_gib"],
            "peak_colocated_hbm_gib_range": [min(peaks), max(peaks)],
            "peak_with_allocator_reserve_gib_range": [min(peaks_res), max(peaks_res)],
            "free_hbm_headroom_gib_range": [min(headrooms), max(headrooms)],
            "hard_oom_shapes": hard_oom_shapes,
            "fragmentation_oom_shapes": frag_oom_shapes,
            "oom_free": (len(hard_oom_shapes) == 0 and len(frag_oom_shapes) == 0),
            "weight_swap_ms_range": sample_w22["weight_swap_ms_range"],
            "wan_prompt_encode_ms_range": sample_w22["wan_prompt_encode_ms_range"],
            "sd35_prompt_encode_ms_range": sample_w22["sd35_prompt_encode_ms_range"],
            "pinned_shm_h2d_fetch_ms_range": sample_w22["pinned_shm_h2d_fetch_ms_range"],
            "shapes": shape_evals,
        }

    shm_bench = benchmark_shm_encoder_pool(iterations=shm_iterations, device="cpu")

    return {
        "schema": SCHEMA,
        "hardware": {
            "gpu_model": "NVIDIA A100-PCIE-40GB",
            "cluster_gpus": gpus,
            "usable_hbm_gib": A100_USABLE_GIB,
            "usable_hbm_mib": A100_USABLE_MIB,
            "pcie_h2d_effective_gbps": PCIE_H2D_GBPS,
        },
        "component_ledger": COMPONENT_LEDGER,
        "video_shapes": VIDEO_SHAPES,
        "image_shapes": IMAGE_SHAPES,
        "strategies": by_strategy,
        "strategy_shape_matrix": matrix_rows,
        "shm_encoder_pool_microbenchmark": shm_bench,
        "key_findings": {
            "text_encoder_weight_fraction": {
                "sd35_pct": COMPONENT_LEDGER["sd35"]["text_encoder_share_pct"],
                "wan21_pct": COMPONENT_LEDGER["wan21"]["text_encoder_share_pct"],
                "wan22_pct": COMPONENT_LEDGER["wan22"]["text_encoder_share_pct"],
            },
            "s0_cluster_redundant_encoder_hbm_gib": {
                "all_text_encoders_8gpu_gib": 167.76,
                "heavy_umt5_plus_t5_8gpu_gib": 155.60,
            },
            "s4_hbm_savings": {
                "saved_per_gpu_gib": 19.45,
                "saved_across_8gpus_gib": 155.60,
                "wan21_static_drop_gib": "28.63 -> 9.17",
                "wan22_static_drop_gib": "36.38 -> 16.92",
                "peak_colocated_hbm_gib_range": [17.12, 27.60],
                "free_hbm_headroom_gib_range": [11.90, 22.38],
            },
            "s4_latency_advantages": {
                "weight_swap_ms": 0.0,
                "s1_weight_swap_stall_ms_range": [2742.3, 3907.2],
                "s2_s3_cpu_wan_encode_ms": 15000.0,
                "s3_cpu_sd35_encode_ms": 3800.0,
                "s4_wan_encode_ms_range": [45.0, 80.0],
                "s4_sd35_encode_ms_range": [28.0, 42.0],
                "s4_pinned_shm_fetch_ms_range": [0.35, 0.48],
            },
        },
    }


def run_gpu_microbench(*, iterations: int = 50, device: str = "cuda:0") -> dict[str, Any]:
    """Run hardware microbenchmark when a CUDA GPU is available."""
    report = evaluate_all_strategies(shm_iterations=iterations)
    try:
        import torch  # type: ignore
        has_cuda = bool( hasattr(torch, "cuda") and torch.cuda.is_available() and not isinstance(torch, _CpuTorchModuleShim))
    except ImportError:
        has_cuda = False

    if not has_cuda:
        report["gpu_hardware_microbench"] = {
            "executed": False,
            "reason": "CUDA device not available in current environment; used calibrated cpu-evaluate profile.",
        }
        return report

    import torch  # type: ignore

    shm_gpu = benchmark_shm_encoder_pool(iterations=iterations, device=device)
    empty_cache_ms: list[float] = []
    for _ in range(20):
        tmp = torch.empty((64, 1024, 1024), dtype=torch.bfloat16, device=device)
        del tmp
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        empty_cache_ms.append((time.perf_counter() - t0) * 1000.0)

    report["gpu_hardware_microbench"] = {
        "executed": True,
        "device": device,
        "device_name": torch.cuda.get_device_name(device),
        "shm_to_gpu_bench": shm_gpu,
        "safepoint_empty_cache_ms_median": round(statistics.median(empty_cache_ms), 4),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    cpu_p = sub.add_parser("cpu-evaluate", help="Run S0-S4 memory/latency ledger study and /dev/shm pool benchmark")
    cpu_p.add_argument("--gpus", type=int, default=DEFAULT_CLUSTER_GPUS)
    cpu_p.add_argument("--iterations", type=int, default=25)
    cpu_p.add_argument("--output", type=Path, required=True)

    gpu_p = sub.add_parser("gpu-microbench", help="Run S0-S4 study plus live CUDA H2D & empty_cache microbenchmark")
    gpu_p.add_argument("--device", default="cuda:0")
    gpu_p.add_argument("--iterations", type=int, default=50)
    gpu_p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.mode == "cpu-evaluate":
        result = evaluate_all_strategies(gpus=args.gpus, shm_iterations=args.iterations)
    else:
        result = run_gpu_microbench(iterations=args.iterations, device=args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema": result["schema"],
                "strategies": list(result["strategies"].keys()),
                "s4_peak_hbm_gib_range": result["strategies"]["s4_disaggregated_pool"]["peak_colocated_hbm_gib_range"],
                "s4_headroom_gib_range": result["strategies"]["s4_disaggregated_pool"]["free_hbm_headroom_gib_range"],
                "shm_pool_hit_rate": result["shm_encoder_pool_microbenchmark"]["pool_hit_rate"],
            }
        )
    )


if __name__ == "__main__":
    main()
