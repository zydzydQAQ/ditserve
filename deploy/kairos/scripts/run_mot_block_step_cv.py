#!/usr/bin/env python3
"""Motivation Step- & Block-Level Predictability (CV%) Collector on 8x A100-PCIE-40GB.

Collects zero-perturbation CUDA-event step and per-block latency distributions across
real DiffusionDB (SD3.5-Medium) and VidProM (Wan2.2-TI2V-5B) prompts over 36 parallel
configurations split into two runtime-balanced 8-GPU shards (`s0` and `s1`).

Key invariants:
- Pre-allocates `torch.cuda.Event(enable_timing=True)` pools for `[S]` steps and
  `[S][P][L]` block executions before each timed prompt.
- Attaches `register_forward_pre_hook(..., prepend=True)` and
  `register_forward_hook(..., prepend=False)` on every Transformer block.
- NEVER calls `torch.cuda.synchronize()` inside step/block hooks; calls `synchronize()`
  EXACTLY ONCE after each prompt finishes.
- Gathers `[S][P][L]` block ms and `[S]` step ms across ranks via `dist.all_gather_object`,
  reduces by subgroup/active-rank critical-path maximum, and writes standardized JSON to
  `$RUN_DIR/mot_block_step_cv_shard.json`.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SD35_PATH = os.environ.get("DITSERVE_SD35_MODEL_PATH", "/shared/data/DITSERVE/models/sd3.5-medium")
WAN22_PATH = os.environ.get("DITSERVE_WAN22_MODEL_PATH", "/shared/data/DITSERVE/models/wan2.2-ti2v-5b")

# Verified test-split prompts from frozen formal-mixed-v2 pools (DiffusionDB & VidProM),
# spanning short, medium, and long token counts while excluding known exploration prompts.
DIFFUSIONDB_PROMPTS: List[Dict[str, str]] = [
    {
        "sample_id": "diffusiondb-80e644a1-bc4c-4fb9-98bb-c0e9ddace880.png",
        "prompt": "a photo of a snow owl! and grizzly bear hybrid creature concept ",
    },
    {
        "sample_id": "diffusiondb-399a7e35-6644-409b-a248-a6967da44d15.png",
        "prompt": (
            "lovers being pulled apart across space & time, photorealistic by michael komarck, "
            "greg rutkowski, victo ngai, artgerm, willem claesz heda and j. dickenson "
        ),
    },
    {
        "sample_id": "diffusiondb-07c5ee65-7ba0-4363-b508-380736dee0ca.png",
        "prompt": (
            "photo illustration comics painting of small scifi machine shop in a shipping container, "
            "scifi, cyberpunk, neon lighting, noir, digital photography, intricate detail, 8 k, "
            "volumetric lighting, artstation "
        ),
    },
]

VIDPROM_PROMPTS: List[Dict[str, str]] = [
    {
        "sample_id": "vidprom-58f6cd3d-5443-5b3b-a8b5-225e64b0581a",
        "prompt": "Nature Peony Garden Japan  ",
    },
    {
        "sample_id": "vidprom-6aedf182-06b7-57dc-bf15-40162b2d041c",
        "prompt": "girl lo-fi studyng at desk, warm lights, rainy outside  ",
    },
    {
        "sample_id": "vidprom-89be7e11-6b97-5e7d-8b84-cc12b6fed891",
        "prompt": (
            "wood cabin living room with a christmas tree and a working fireplace with big fire. "
            "Snow falling inside the room. Looped animation -ar 16:9  "
        ),
    },
]

SD35_NEG_PROMPT = "blurry, low quality, distorted, watermark, oversaturated, static"
WAN22_NEG_PROMPT = ""

SD35_RESOLUTIONS: List[Tuple[str, int, int]] = [
    ("512x512", 512, 512),
    ("1024x1024", 1024, 1024),
    ("1440x1440", 1440, 1440),
]

SD35_FORMS: List[Tuple[str, int, int]] = [
    ("sp1", 1, 1),
    ("sp2", 1, 2),
    ("sp4", 1, 4),
    ("sp8", 1, 8),
    ("cfg2", 2, 1),
]

WAN22_RESOLUTIONS: List[Tuple[str, int, int, int]] = [
    ("256p", 256, 448, 81),
    ("480p", 512, 896, 81),
    ("720p", 704, 1280, 81),
]

WAN22_FORMS: List[Tuple[str, int, int]] = [
    ("sp1", 1, 1),
    ("sp2", 1, 2),
    ("sp4", 1, 4),
    ("sp8", 1, 8),
    ("cfg2", 2, 1),
    ("cfg2sp2", 2, 2),
    ("cfg2sp4", 2, 4),
]

# Module-scoped active valid_kv_len for SD3.5 token-sharded SP execution when
# latent height is not divisible by (2 * sp_degree) (specifically 1440x1440 SP8 -> 8100).
_ACTIVE_SD35_VALID_KV_LEN: Optional[int] = None


def build_full_matrix() -> List[Dict[str, Any]]:
    """Return all 36 configs (15 SD3.5-Medium + 21 Wan2.2-TI2V-5B)."""
    items: List[Dict[str, Any]] = []
    for res_label, h, w in SD35_RESOLUTIONS:
        for form, cfg_deg, sp_deg in SD35_FORMS:
            items.append(
                {
                    "modality": "image",
                    "model": "SD3.5-Medium",
                    "resolution": res_label,
                    "height": h,
                    "width": w,
                    "frames": 1,
                    "form": form,
                    "gpus": cfg_deg * sp_deg,
                    "sp_degree": sp_deg,
                    "cfg_degree": cfg_deg,
                    "num_blocks": 24,
                    "total_steps": 28,
                    "warmup_steps": 3,
                    "steady_steps_per_prompt": 25,
                }
            )
    for res_label, h, w, frames in WAN22_RESOLUTIONS:
        for form, cfg_deg, sp_deg in WAN22_FORMS:
            items.append(
                {
                    "modality": "video",
                    "model": "Wan2.2-TI2V-5B",
                    "resolution": res_label,
                    "height": h,
                    "width": w,
                    "frames": frames,
                    "form": form,
                    "gpus": cfg_deg * sp_deg,
                    "sp_degree": sp_deg,
                    "cfg_degree": cfg_deg,
                    "num_blocks": 30,
                    "total_steps": 23,
                    "warmup_steps": 3,
                    "steady_steps_per_prompt": 20,
                }
            )
    return items


def build_shard_matrix(shard: str) -> List[Dict[str, Any]]:
    """Partition the 36 configs across balanced 8-GPU pods (`s0` ~22.6 min, `s1` ~23.3 min):
    - `s0`: Wan2.2-TI2V-5B 720p x 7 forms (7 heavy configs)
    - `s1`: Wan2.2-TI2V-5B {256p, 480p} x 7 forms (14 configs) + SD3.5-Medium {512, 1024, 1440} x 5 forms (15 configs)
    - `all`: all 36 configs
    """
    full = build_full_matrix()
    if shard == "all":
        return full
    if shard == "s0":
        return [c for c in full if c["model"] == "Wan2.2-TI2V-5B" and c["resolution"] == "720p"]
    if shard == "s1":
        return [c for c in full if not (c["model"] == "Wan2.2-TI2V-5B" and c["resolution"] == "720p")]
    if shard == "s2":
        return [
            c
            for c in full
            if c["model"] == "SD3.5-Medium" and c["resolution"] == "1440x1440" and c["form"] in ("sp4", "sp8", "cfg2")
        ]
    raise ValueError(f"Unsupported shard {shard!r}; expected 's0', 's1', 's2', or 'all'")


def check_storage(mount_path: str = "/shared") -> Optional[Dict[str, int]]:
    """Enforce shared-storage safety gate (<=805G and <81%, strictly inside AGENTS.md 810G/82%)."""
    if not os.path.exists(mount_path):
        return None
    out = subprocess.check_output(["df", "-BG", mount_path], text=True, timeout=15).splitlines()[-1].split()
    used_g = int(out[2].rstrip("G"))
    pct = int(out[4].rstrip("%"))
    if not (used_g <= 805 and pct < 81):
        raise RuntimeError(f"shared storage stop line reached on {mount_path}: used={used_g}G pct={pct}%")
    return {"used_gib": used_g, "percent": pct}


def percentile(values: Sequence[float], q: float) -> float:
    """Linearly interpolated percentile matching numpy.percentile(..., method='linear')."""
    if not values:
        return 0.0
    s = sorted(float(x) for x in values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def cv_pct(values: Sequence[float]) -> float:
    """Sample Coefficient of Variation in percent: 100 * sample_std / mean."""
    vals = [float(x) for x in values]
    if len(vals) < 2:
        return 0.0
    mu = statistics.fmean(vals)
    if mu <= 0.0:
        return 0.0
    return 100.0 * (statistics.stdev(vals) / mu)


class ZeroSyncBlockStepProfiler:
    """Pre-allocated CUDA-event profiler for `[S]` steps and `[S][P][L]` blocks with zero in-loop sync."""

    def __init__(
        self,
        blocks: Sequence[Any],
        total_steps: int,
        passes_per_step: int,
        *,
        event_factory: Optional[Callable[[], Any]] = None,
        sync_fn: Optional[Callable[[], None]] = None,
    ):
        if total_steps <= 0 or passes_per_step <= 0 or len(blocks) == 0:
            raise ValueError("total_steps, passes_per_step, and len(blocks) must all be positive")
        if event_factory is None:
            import torch

            event_factory = lambda: torch.cuda.Event(enable_timing=True)
        if sync_fn is None:
            import torch

            sync_fn = torch.cuda.synchronize

        self.blocks = list(blocks)
        self.total_steps = int(total_steps)
        self.passes_per_step = int(passes_per_step)
        self.num_blocks = len(self.blocks)
        self._sync_fn = sync_fn

        # Pre-allocate all step and block timing events before execution starts.
        self.step_start_events = [event_factory() for _ in range(self.total_steps)]
        self.step_end_events = [event_factory() for _ in range(self.total_steps)]
        self.block_start_events = [
            [[event_factory() for _ in range(self.num_blocks)] for _ in range(self.passes_per_step)]
            for _ in range(self.total_steps)
        ]
        self.block_end_events = [
            [[event_factory() for _ in range(self.num_blocks)] for _ in range(self.passes_per_step)]
            for _ in range(self.total_steps)
        ]
        self._recorded_pre = [
            [[False] * self.num_blocks for _ in range(self.passes_per_step)]
            for _ in range(self.total_steps)
        ]
        self._recorded_post = [
            [[False] * self.num_blocks for _ in range(self.passes_per_step)]
            for _ in range(self.total_steps)
        ]
        self._step_started = [False] * self.total_steps
        self._step_ended = [False] * self.total_steps

        self._active = False
        self._cur_step = -1
        self._cur_pass = -1
        self._handles: List[Any] = []
        self._finalized = False

    def attach(self) -> "ZeroSyncBlockStepProfiler":
        if self._handles:
            raise RuntimeError("Hooks are already attached")
        for b_idx, blk in enumerate(self.blocks):

            def _pre_hook(_mod: Any, _inputs: Any, _b: int = b_idx) -> None:
                if not self._active:
                    return None
                s = self._cur_step
                p = self._cur_pass
                if not (0 <= s < self.total_steps and 0 <= p < self.passes_per_step):
                    raise RuntimeError(f"Block {_b} pre-hook fired outside valid (step={s}, pass={p})")
                if self._recorded_pre[s][p][_b]:
                    raise RuntimeError(f"Duplicate pre-hook recording at (step={s}, pass={p}, block={_b})")
                self.block_start_events[s][p][_b].record()
                self._recorded_pre[s][p][_b] = True
                return None

            def _post_hook(_mod: Any, _inputs: Any, _output: Any, _b: int = b_idx) -> None:
                if not self._active:
                    return None
                s = self._cur_step
                p = self._cur_pass
                if not (0 <= s < self.total_steps and 0 <= p < self.passes_per_step):
                    raise RuntimeError(f"Block {_b} post-hook fired outside valid (step={s}, pass={p})")
                if self._recorded_post[s][p][_b]:
                    raise RuntimeError(f"Duplicate post-hook recording at (step={s}, pass={p}, block={_b})")
                self.block_end_events[s][p][_b].record()
                self._recorded_post[s][p][_b] = True
                return None

            self._handles.append(blk.register_forward_pre_hook(_pre_hook, prepend=True))
            self._handles.append(blk.register_forward_hook(_post_hook, prepend=False))
        return self

    def detach(self) -> None:
        self._active = False
        while self._handles:
            h = self._handles.pop()
            h.remove()

    def __enter__(self) -> "ZeroSyncBlockStepProfiler":
        return self.attach()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.detach()

    def begin_step(self, step_idx: int) -> None:
        if not (0 <= step_idx < self.total_steps):
            raise IndexError(f"step_idx {step_idx} out of [0, {self.total_steps})")
        if self._step_started[step_idx]:
            raise RuntimeError(f"Step {step_idx} already started")
        self._cur_step = step_idx
        self._cur_pass = 0
        self._active = True
        self._step_started[step_idx] = True
        self.step_start_events[step_idx].record()

    def begin_pass(self, pass_idx: int) -> None:
        if not self._active:
            raise RuntimeError("begin_pass called while no step is active")
        if not (0 <= pass_idx < self.passes_per_step):
            raise IndexError(f"pass_idx {pass_idx} out of [0, {self.passes_per_step})")
        self._cur_pass = pass_idx

    def end_step(self, step_idx: int) -> None:
        if step_idx != self._cur_step or not self._active:
            raise RuntimeError(f"end_step({step_idx}) mismatched active step {self._cur_step}")
        self.step_end_events[step_idx].record()
        self._step_ended[step_idx] = True
        self._active = False
        self._cur_step = -1
        self._cur_pass = -1

    def finalize(self, warmup_steps: int = 3) -> Tuple[List[float], List[List[List[float]]]]:
        """Synchronize ONCE after the prompt finishes and return steady `([S_steady], [S_steady][P][L])` ms."""
        if self._finalized:
            raise RuntimeError("Profiler already finalized")
        if warmup_steps < 0 or warmup_steps >= self.total_steps:
            raise ValueError(f"Invalid warmup_steps={warmup_steps} for total_steps={self.total_steps}")
        if not all(self._step_started) or not all(self._step_ended):
            raise RuntimeError("Not all pre-allocated steps were started and ended")
        for s in range(self.total_steps):
            for p in range(self.passes_per_step):
                for b in range(self.num_blocks):
                    if not (self._recorded_pre[s][p][b] and self._recorded_post[s][p][b]):
                        raise RuntimeError(f"Missing block event recording at (step={s}, pass={p}, block={b})")

        # Single device synchronization for the entire prompt trajectory.
        self._sync_fn()
        self._finalized = True

        steady_step_ms: List[float] = []
        steady_block_ms: List[List[List[float]]] = []
        for s in range(warmup_steps, self.total_steps):
            steady_step_ms.append(float(self.step_start_events[s].elapsed_time(self.step_end_events[s])))
            passes_ms: List[List[float]] = []
            for p in range(self.passes_per_step):
                blk_ms = [
                    float(self.block_start_events[s][p][b].elapsed_time(self.block_end_events[s][p][b]))
                    for b in range(self.num_blocks)
                ]
                passes_ms.append(blk_ms)
            steady_block_ms.append(passes_ms)
        return steady_step_ms, steady_block_ms


def reduce_critical_path_timings(active_payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Take elementwise max critical path across active subgroup ranks (`0 .. gpus-1`)."""
    if not active_payloads:
        raise ValueError("active_payloads must be non-empty")
    ref_steps = active_payloads[0]["step_ms"]
    ref_blocks = active_payloads[0]["block_ms"]
    num_steps = len(ref_steps)
    num_passes = len(ref_blocks[0])
    num_blocks = len(ref_blocks[0][0])

    crit_steps = [
        max(float(payload["step_ms"][s]) for payload in active_payloads)
        for s in range(num_steps)
    ]
    crit_blocks = [
        [
            [
                max(float(payload["block_ms"][s][p][b]) for payload in active_payloads)
                for b in range(num_blocks)
            ]
            for p in range(num_passes)
        ]
        for s in range(num_steps)
    ]
    peak_hbm_gib = max(float(payload.get("peak_hbm_gib", 0.0)) for payload in active_payloads)
    return {
        "step_ms": crit_steps,
        "block_ms": crit_blocks,
        "peak_hbm_gib": peak_hbm_gib,
    }


def compute_config_cv_metrics(
    cfg: Dict[str, Any],
    prompt_step_ms: Sequence[Sequence[float]],
    prompt_block_ms: Sequence[Sequence[Sequence[Sequence[float]]]],
    peak_hbm_gib: float,
    prompt_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute standardized step & block CV% statistics across K prompts x S steady steps."""
    prompts_count = len(prompt_step_ms)
    if prompts_count == 0 or len(prompt_block_ms) != prompts_count:
        raise ValueError("prompt_step_ms and prompt_block_ms must have matching positive prompt counts")

    steady_steps = len(prompt_step_ms[0])
    passes_per_step = len(prompt_block_ms[0][0])
    num_blocks = int(cfg["num_blocks"])
    for k in range(prompts_count):
        if len(prompt_step_ms[k]) != steady_steps or len(prompt_block_ms[k]) != steady_steps:
            raise ValueError("Inconsistent steady step count across prompts")
        for s in range(steady_steps):
            if len(prompt_block_ms[k][s]) != passes_per_step:
                raise ValueError("Inconsistent pass count across steps")
            for p in range(passes_per_step):
                if len(prompt_block_ms[k][s][p]) != num_blocks:
                    raise ValueError(f"Expected {num_blocks} blocks, got {len(prompt_block_ms[k][s][p])}")

    # Step-level statistics
    within_prompt_step_cvs = [cv_pct(prompt_step_ms[k]) for k in range(prompts_count)]
    pooled_steps = [float(x) for k in range(prompts_count) for x in prompt_step_ms[k]]
    step_mean = statistics.fmean(pooled_steps)
    step_p50 = percentile(pooled_steps, 50.0)
    step_p99 = percentile(pooled_steps, 99.0)
    step_p99_p50 = (step_p99 / step_p50) if step_p50 > 0.0 else 0.0
    step_temporal_cv = statistics.fmean(within_prompt_step_cvs)
    cross_prompt_step_cv = cv_pct(pooled_steps)
    per_prompt_step_means = [round(statistics.fmean(prompt_step_ms[k]), 4) for k in range(prompts_count)]

    # Block-level pooled distribution
    pooled_blocks = [
        float(prompt_block_ms[k][s][p][b])
        for k in range(prompts_count)
        for s in range(steady_steps)
        for p in range(passes_per_step)
        for b in range(num_blocks)
    ]
    block_mean = statistics.fmean(pooled_blocks)
    block_p50 = percentile(pooled_blocks, 50.0)
    block_p99 = percentile(pooled_blocks, 99.0)
    block_p99_p50 = (block_p99 / block_p50) if block_p50 > 0.0 else 0.0

    # Temporal CV% per (pass, block): mean across prompts of within-prompt across-step CV%
    pb_temporal_cvs: List[float] = []
    for p in range(passes_per_step):
        for b in range(num_blocks):
            per_prompt_cvs = [
                cv_pct([prompt_block_ms[k][s][p][b] for s in range(steady_steps)])
                for k in range(prompts_count)
            ]
            pb_temporal_cvs.append(statistics.fmean(per_prompt_cvs))

    block_temporal_cv_mean = statistics.fmean(pb_temporal_cvs)
    block_temporal_cv_p95 = percentile(pb_temporal_cvs, 95.0)
    block_temporal_cv_max = max(pb_temporal_cvs)

    # Spatial mean latency per block index b in 0..L-1
    per_block_means: List[float] = []
    for b in range(num_blocks):
        b_samples = [
            float(prompt_block_ms[k][s][p][b])
            for k in range(prompts_count)
            for s in range(steady_steps)
            for p in range(passes_per_step)
        ]
        per_block_means.append(statistics.fmean(b_samples))

    block_spatial_cv_all = cv_pct(per_block_means)
    if cfg["model"] == "SD3.5-Medium":
        if num_blocks != 24:
            raise ValueError(f"SD3.5-Medium expects 24 blocks, got {num_blocks}")
        # Dual-attention blocks 0..12 (13 blocks) and single-attention blocks 13..22 (10 blocks),
        # excluding context_pre_only final block 23.
        dual_cv = cv_pct(per_block_means[0:13])
        single_cv = cv_pct(per_block_means[13:23])
        block_spatial_cv_hom = 0.5 * (dual_cv + single_cv)
    else:
        dual_cv = None
        single_cv = None
        block_spatial_cv_hom = block_spatial_cv_all

    record: Dict[str, Any] = {
        "modality": cfg["modality"],
        "model": cfg["model"],
        "resolution": cfg["resolution"],
        "height": int(cfg["height"]),
        "width": int(cfg["width"]),
        "frames": int(cfg["frames"]),
        "form": cfg["form"],
        "gpus": int(cfg["gpus"]),
        "sp_degree": int(cfg["sp_degree"]),
        "cfg_degree": int(cfg["cfg_degree"]),
        "num_blocks": num_blocks,
        "passes_per_step": passes_per_step,
        "prompts_count": prompts_count,
        "steady_steps_per_prompt": steady_steps,
        "step_mean_ms": round(step_mean, 4),
        "step_p50_ms": round(step_p50, 4),
        "step_p99_ms": round(step_p99, 4),
        "step_p99_p50_ratio": round(step_p99_p50, 5),
        "step_temporal_cv_pct": round(step_temporal_cv, 4),
        "cross_prompt_step_cv_pct": round(cross_prompt_step_cv, 4),
        "block_mean_ms": round(block_mean, 4),
        "block_p50_ms": round(block_p50, 4),
        "block_p99_ms": round(block_p99, 4),
        "block_p99_p50_ratio": round(block_p99_p50, 5),
        "block_temporal_cv_mean_pct": round(block_temporal_cv_mean, 4),
        "block_temporal_cv_p95_pct": round(block_temporal_cv_p95, 4),
        "block_temporal_cv_max_pct": round(block_temporal_cv_max, 4),
        "block_spatial_cv_all_pct": round(block_spatial_cv_all, 4),
        "block_spatial_cv_homogeneous_pct": round(block_spatial_cv_hom, 4),
        "per_block_mean_ms": [round(x, 4) for x in per_block_means],
        "peak_hbm_gib": round(float(peak_hbm_gib), 3),
        "per_prompt_step_mean_ms": per_prompt_step_means,
    }
    if prompt_ids is not None:
        record["prompt_sample_ids"] = list(prompt_ids)
    if dual_cv is not None and single_cv is not None:
        record["sd35_dual_attn_spatial_cv_pct"] = round(dual_cv, 4)
        record["sd35_single_attn_spatial_cv_pct"] = round(single_cv, 4)
    return record


# ---------------------------------------------------------------------------
# SD3.5 Patch-Strip Helpers for Arbitrary Resolution SP Sharding (e.g. 1440^2 SP4/SP8)
# ---------------------------------------------------------------------------


def pack_sd35_latents_to_patch_strip(latents: Any, patch_size: int = 2, pad_tokens: int = 0) -> Any:
    """Pack `[B, C, H_lat, W_lat]` into `[B, C, p, p * (N_valid + pad_tokens)]` so non-overlapping
    `Conv2d(kernel_size=p, stride=p)` produces the exact flattened patch sequence `[B, N_padded, D]`."""
    import torch

    b, c, h_lat, w_lat = latents.shape
    p = int(patch_size)
    if h_lat % p != 0 or w_lat % p != 0:
        raise ValueError(f"Latent shape ({h_lat}, {w_lat}) not divisible by patch_size={p}")
    hp, wp = h_lat // p, w_lat // p
    n_valid = hp * wp
    strip = latents.reshape(b, c, hp, p, wp, p).permute(0, 1, 3, 2, 4, 5).reshape(b, c, p, p * n_valid)
    if pad_tokens > 0:
        pad = torch.zeros((b, c, p, p * int(pad_tokens)), dtype=latents.dtype, device=latents.device)
        strip = torch.cat([strip, pad], dim=-1)
    return strip.contiguous()


def unpatchify_sd35_tokens_to_patch_strip(tokens: Any, out_channels: int = 16, patch_size: int = 2) -> Any:
    """Invert SD3 `proj_out` `[B, N_local, p*p*C]` back into `[B, C, p, p * N_local]`."""
    import torch

    b, n_local, _ = tokens.shape
    p = int(patch_size)
    c = int(out_channels)
    reshaped = tokens.reshape(b, 1, n_local, p, p, c)
    return torch.einsum("nhwpqc->nchpwq", reshaped).reshape(b, c, p, p * n_local).contiguous()


def unpack_sd35_patch_strip_to_latents(
    strip: Any, h_lat: int, w_lat: int, patch_size: int = 2
) -> Any:
    """Unpack `[B, C, p, p * N_padded]` (slicing to `N_valid`) back to `[B, C, H_lat, W_lat]`."""
    b, c, p, total_w = strip.shape
    if p != int(patch_size):
        raise ValueError(f"Expected strip height {patch_size}, got {p}")
    hp, wp = h_lat // p, w_lat // p
    n_valid = hp * wp
    valid_strip = strip[:, :, :, : p * n_valid]
    return valid_strip.reshape(b, c, p, hp, wp, p).permute(0, 1, 3, 2, 4, 5).reshape(b, c, h_lat, w_lat).contiguous()


# ---------------------------------------------------------------------------
# xFuser Patching & Dynamic 8-GPU Subgroup Manager
# ---------------------------------------------------------------------------


def patch_xfuser_sd3_processor() -> None:
    """Ensure xFuserJointAttnProcessor2_0 initializes joint_strategy=None when encoder_hidden_states is None
    (required by SD3.5-Medium dual-attention blocks 0..12 attn2 under SP > 1), skips PipeFusion KV cache
    when pp_degree == 1, and honours `_ACTIVE_SD35_VALID_KV_LEN` when token-sharded SP pads trailing tokens."""
    import torch
    import xfuser.model_executor.layers.attention_processor as ap_mod
    from xfuser.core.cache_manager.cache_manager import get_cache_manager
    from xfuser.core.distributed import (
        get_pipeline_parallel_world_size,
        get_runtime_state,
        get_sequence_parallel_world_size,
    )
    from xfuser.model_executor.layers.usp import USP

    cls = ap_mod.xFuserJointAttnProcessor2_0
    HAS_LONG_CTX_ATTN = ap_mod.HAS_LONG_CTX_ATTN
    _joint_sp_padding_attention_kwargs = ap_mod._joint_sp_padding_attention_kwargs

    def patched_call(
        self: Any,
        attn: Any,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        if encoder_hidden_states is not None:
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states).view(
                batch_size, -1, attn.heads, head_dim
            )
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states).view(
                batch_size, -1, attn.heads, head_dim
            )
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states).view(
                batch_size, -1, attn.heads, head_dim
            )
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
        else:
            encoder_hidden_states_query_proj = None
            encoder_hidden_states_key_proj = None
            encoder_hidden_states_value_proj = None

        if not self.use_long_ctx_attn_kvcache and get_pipeline_parallel_world_size() > 1:
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=1,
                layer_type="attn",
            )

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            attention_kwargs = (
                {"valid_kv_len": _ACTIVE_SD35_VALID_KV_LEN}
                if _ACTIVE_SD35_VALID_KV_LEN is not None
                else None
            )
            joint_strategy = None
            if encoder_hidden_states is not None:
                if get_runtime_state().split_text_embed_in_sp:
                    attention_kwargs = _joint_sp_padding_attention_kwargs(query, encoder_hidden_states_query_proj)
                    query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
                    key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
                    value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)
                    encoder_hidden_states_query_proj = None
                    encoder_hidden_states_key_proj = None
                    encoder_hidden_states_value_proj = None
                    joint_strategy = None
                else:
                    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.transpose(1, 2)
                    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.transpose(1, 2)
                    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.transpose(1, 2)
                    joint_strategy = "rear"

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            hidden_states = USP(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                joint_query=encoder_hidden_states_query_proj,
                joint_key=encoder_hidden_states_key_proj,
                joint_value=encoder_hidden_states_value_proj,
                joint_strategy=joint_strategy,
                attn_layer=attn,
                attention_kwargs=attention_kwargs,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        else:
            if encoder_hidden_states is not None:
                query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states = USP(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                head_balance_layer=attn,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if encoder_hidden_states is not None:
            if context_input_ndim == 4:
                encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )
            return hidden_states, encoder_hidden_states
        return hidden_states

    cls.__call__ = patched_call


class SubgroupCoordinatorManager:
    """Pre-creates all NCCL groups across 8 ranks and switches `(cfg_degree, sp_degree)` dynamically."""

    def __init__(self, rank: int, world_size: int, device: Any):
        import torch.distributed as dist

        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.nccl_groups: Dict[Tuple[int, ...], Any] = {}
        self.gloo_groups: Dict[Tuple[int, ...], Any] = {}

        all_tuples = set()
        for r in range(world_size):
            all_tuples.add((r,))
        for deg in (2, 4, 8):
            if deg <= world_size:
                for start in range(0, world_size - deg + 1, deg):
                    all_tuples.add(tuple(range(start, start + deg)))
        for sp_deg in (1, 2, 4):
            if 2 * sp_deg <= world_size:
                for off in range(sp_deg):
                    all_tuples.add((off, off + sp_deg))

        for ranks in sorted(all_tuples, key=lambda x: (len(x), x)):
            if len(ranks) == 1:
                self.nccl_groups[ranks] = None
            else:
                self.nccl_groups[ranks] = dist.new_group(list(ranks), backend="nccl")
            self.gloo_groups[ranks] = None

    def _make_coord(
        self,
        ranks: Sequence[int],
        *,
        sequence: bool = False,
        pipeline: bool = False,
        ulysses_degree: Optional[int] = None,
        ring_degree: int = 1,
    ) -> Any:
        from xfuser.core.distributed.group_coordinator import (
            GroupCoordinator,
            PipelineGroupCoordinator,
            SequenceParallelGroupCoordinator,
        )

        ranks_t = tuple(ranks)
        cls = (
            SequenceParallelGroupCoordinator
            if sequence
            else PipelineGroupCoordinator
            if pipeline
            else GroupCoordinator
        )
        coord = cls.__new__(cls)
        coord.rank = self.rank
        coord.local_rank = self.rank
        coord.ranks = ranks_t
        coord.world_size = len(ranks_t)
        coord.rank_in_group = ranks_t.index(self.rank)
        coord.device_group = self.nccl_groups[ranks_t]
        coord.cpu_group = self.gloo_groups[ranks_t]
        coord.shm_broadcaster = None
        coord.device = self.device
        if sequence:
            deg = len(ranks_t)
            r_deg = int(ring_degree or 1)
            u_deg = int(ulysses_degree or (deg // r_deg))
            coord.ulysses_group = self.nccl_groups[ranks_t]
            coord.ring_group = self.nccl_groups[(self.rank,)]
            coord.ulysses_world_size = u_deg
            coord.ulysses_rank = coord.rank_in_group
            coord.ring_world_size = 1
            coord.ring_rank = 0
        if pipeline:
            coord.cpu_groups = []
            coord.device_groups = []
            coord.recv_buffer_set = False
            coord.dtype = None
            coord.num_pipefusion_patches = 1
            coord.skip_tensor_recv_buffer_set = False
            coord.skip_device_group = coord.device_group
            coord.reset_buffer()
        return coord

    def activate_layout(self, cfg_degree: int, sp_degree: int, runtime_state: Any = None) -> None:
        from ditserve.models.xdit.process_group import PROCESS_GROUP
        from xfuser.core.distributed import parallel_state as ps

        total_gpus = cfg_degree * sp_degree
        if total_gpus == 1:
            world_ranks = (self.rank,)
            cfg_ranks = (self.rank,)
            sp_ranks = (self.rank,)
        else:
            assert self.rank < total_gpus, f"rank {self.rank} >= total_gpus {total_gpus}"
            world_ranks = tuple(range(total_gpus))
            if cfg_degree == 1:
                cfg_ranks = (self.rank,)
                sp_ranks = tuple(range(sp_degree))
            else:
                cfg_role = self.rank // sp_degree
                sp_off = self.rank % sp_degree
                cfg_ranks = (sp_off, sp_off + sp_degree)
                sp_ranks = tuple(range(cfg_role * sp_degree, (cfg_role + 1) * sp_degree))

        ps._WORLD = self._make_coord(world_ranks)
        ps._DP = self._make_coord((self.rank,))
        ps._CFG = self._make_coord(cfg_ranks)
        ps._SP = self._make_coord(sp_ranks, sequence=True, ulysses_degree=sp_degree, ring_degree=1)
        ps._PP = self._make_coord((self.rank,), pipeline=True)
        ps._TP = self._make_coord((self.rank,))
        ps._FS = self._make_coord((self.rank,))
        ps._MODEL_REPLICA = self._make_coord(sp_ranks)
        ps._DIT = self.nccl_groups[world_ranks]
        ps._VAE = None

        PROCESS_GROUP.ULYSSES_PG = ps._SP.ulysses_group
        PROCESS_GROUP.RING_PG = ps._SP.ring_group
        PROCESS_GROUP.ALL_TO_ALL = None

        if runtime_state is not None:
            cfg = runtime_state.parallel_config
            cfg.world_size = total_gpus
            cfg.dit_parallel_size = total_gpus
            cfg.dp_degree = 1
            cfg.dp_config.dp_degree = 1
            cfg.dp_config.dit_parallel_size = total_gpus
            cfg.cfg_degree = cfg_degree
            cfg.dp_config.cfg_degree = cfg_degree
            cfg.sp_degree = sp_degree
            cfg.ulysses_degree = sp_degree
            cfg.ring_degree = 1
            cfg.sp_config.sp_degree = sp_degree
            cfg.sp_config.ulysses_degree = sp_degree
            cfg.sp_config.ring_degree = 1
            cfg.sp_config.dit_parallel_size = total_gpus
            runtime_state.ready = False


# ---------------------------------------------------------------------------
# GPU Execution: SD3.5-Medium (15 configs) & Wan2.2-TI2V-5B (21 configs)
# ---------------------------------------------------------------------------


def _broadcast_tensor_dict_list(
    items: Optional[List[Dict[str, Any]]],
    keys: Sequence[str],
    rank: int,
    device: Any,
) -> List[Dict[str, Any]]:
    import torch
    import torch.distributed as dist

    meta: List[Any] = [None]
    if rank == 0:
        assert items is not None
        meta[0] = [
            {
                "sample_id": it["sample_id"],
                "shapes": {k: list(it[k].shape) for k in keys},
                "dtypes": {k: str(it[k].dtype).replace("torch.", "") for k in keys},
            }
            for it in items
        ]
    dist.broadcast_object_list(meta, src=0)
    spec_list = meta[0]
    out: List[Dict[str, Any]] = []
    for idx, spec in enumerate(spec_list):
        entry: Dict[str, Any] = {"sample_id": spec["sample_id"]}
        for k in keys:
            dt = getattr(torch, spec["dtypes"][k])
            if rank == 0:
                assert items is not None
                buf = items[idx][k].to(device=device, dtype=dt).contiguous()
            else:
                buf = torch.empty(spec["shapes"][k], device=device, dtype=dt)
            dist.broadcast(buf, src=0)
            entry[k] = buf
        out.append(entry)
    return out


def run_sd35_configs(
    configs: Sequence[Dict[str, Any]],
    rank: int,
    world_size: int,
    device: Any,
    mgr: SubgroupCoordinatorManager,
    on_config_done: Callable[[Dict[str, Any]], None],
) -> List[Dict[str, Any]]:
    global _ACTIVE_SD35_VALID_KV_LEN
    if not configs:
        return []

    import torch
    import torch.distributed as dist
    from diffusers import StableDiffusion3Pipeline
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
        calculate_shift,
        retrieve_timesteps,
    )
    from transformers import T5EncoderModel
    from ditserve.models.xdit.runtime import engine_config
    from ditserve.models.xdit.sd35 import pipeline_class
    from xfuser.core.distributed.runtime_state import get_runtime_state

    patch_xfuser_sd3_processor()
    torch.cuda.empty_cache()

    # Load T5-XXL ONLY on rank 0 to encode the 3 real DiffusionDB prompts rapidly on GPU,
    # then release T5-XXL before multi-rank denoising so 8 ranks do not thrash CPU/NFS.
    t0 = time.perf_counter()
    t5 = (
        T5EncoderModel.from_pretrained(
            SD35_PATH, subfolder="text_encoder_3", torch_dtype=torch.bfloat16, local_files_only=True
        )
        if rank == 0
        else None
    )
    raw_pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_PATH,
        text_encoder_3=t5,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    raw_pipe.text_encoder.to(device=device, dtype=torch.bfloat16)
    raw_pipe.text_encoder_2.to(device=device, dtype=torch.bfloat16)
    raw_pipe.set_progress_bar_config(disable=True)

    encoded_local: Optional[List[Dict[str, Any]]] = None
    if rank == 0:
        assert raw_pipe.text_encoder_3 is not None
        raw_pipe.text_encoder_3.to(device=device, dtype=torch.bfloat16)
        encoded_local = []
        with torch.inference_mode():
            t5_neg = raw_pipe._get_t5_prompt_embeds(
                prompt=SD35_NEG_PROMPT, num_images_per_prompt=1, max_sequence_length=256, device=device
            ).to(dtype=torch.bfloat16)
            clip_l_neg, pooled_l_neg = raw_pipe._get_clip_prompt_embeds(
                prompt=SD35_NEG_PROMPT, device=device, num_images_per_prompt=1, clip_skip=None, clip_model_index=0
            )
            clip_g_neg, pooled_g_neg = raw_pipe._get_clip_prompt_embeds(
                prompt=SD35_NEG_PROMPT, device=device, num_images_per_prompt=1, clip_skip=None, clip_model_index=1
            )
            clip_neg = torch.cat([clip_l_neg, clip_g_neg], dim=-1)
            clip_neg = torch.nn.functional.pad(clip_neg, (0, int(t5_neg.shape[-1]) - int(clip_neg.shape[-1])))
            neg_pe = torch.cat([clip_neg.to(dtype=torch.bfloat16), t5_neg], dim=-2).contiguous()
            neg_ppe = torch.cat([pooled_l_neg, pooled_g_neg], dim=-1).to(dtype=torch.bfloat16).contiguous()

            for row in DIFFUSIONDB_PROMPTS:
                p_str = row["prompt"]
                t5_pos = raw_pipe._get_t5_prompt_embeds(
                    prompt=p_str, num_images_per_prompt=1, max_sequence_length=256, device=device
                ).to(dtype=torch.bfloat16)
                clip_l_pos, pooled_l_pos = raw_pipe._get_clip_prompt_embeds(
                    prompt=p_str, device=device, num_images_per_prompt=1, clip_skip=None, clip_model_index=0
                )
                clip_g_pos, pooled_g_pos = raw_pipe._get_clip_prompt_embeds(
                    prompt=p_str, device=device, num_images_per_prompt=1, clip_skip=None, clip_model_index=1
                )
                clip_pos = torch.cat([clip_l_pos, clip_g_pos], dim=-1)
                clip_pos = torch.nn.functional.pad(clip_pos, (0, int(t5_pos.shape[-1]) - int(clip_pos.shape[-1])))
                pos_pe = torch.cat([clip_pos.to(dtype=torch.bfloat16), t5_pos], dim=-2).contiguous()
                pos_ppe = torch.cat([pooled_l_pos, pooled_g_pos], dim=-1).to(dtype=torch.bfloat16).contiguous()
                encoded_local.append(
                    {
                        "sample_id": row["sample_id"],
                        "pos_pe": pos_pe.cpu(),
                        "neg_pe": neg_pe.cpu(),
                        "pos_ppe": pos_ppe.cpu(),
                        "neg_ppe": neg_ppe.cpu(),
                    }
                )
        raw_pipe.text_encoder_3 = None
        del t5
        gc.collect()
        torch.cuda.empty_cache()

    raw_pipe.transformer.to(device=device, dtype=torch.bfloat16)
    raw_pipe.vae.to(device=device, dtype=torch.bfloat16)
    encoded_prompts = _broadcast_tensor_dict_list(
        encoded_local, ("pos_pe", "neg_pe", "pos_ppe", "neg_ppe"), rank, device
    )

    mgr.activate_layout(1, world_size)
    eng_cfg = engine_config(
        "stable-diffusion-3.5-medium", world_size=world_size, attention="flash", dtype=torch.bfloat16
    )
    eng_cfg.parallel_config.sp_degree = world_size
    eng_cfg.parallel_config.ulysses_degree = world_size
    eng_cfg.parallel_config.sp_config.sp_degree = world_size
    eng_cfg.parallel_config.sp_config.ulysses_degree = world_size

    sd35_cls = pipeline_class()
    xpipe = sd35_cls(raw_pipe, eng_cfg)
    for mod in xpipe.transformer.modules():
        if hasattr(mod, "use_long_ctx_attn_kvcache"):
            mod.use_long_ctx_attn_kvcache = True

    raw_tf = getattr(xpipe.transformer, "module", xpipe.transformer)
    raw_pos_embed = getattr(raw_tf.pos_embed, "module", raw_tf.pos_embed)
    raw_proj_conv = getattr(raw_pos_embed.proj, "module", raw_pos_embed.proj)
    raw_sched = getattr(xpipe.scheduler, "module", xpipe.scheduler)
    rstate = get_runtime_state()
    mgr.activate_layout(1, 1, rstate)
    dist.barrier()
    if rank == 0:
        print(
            f"[Progress][SD3.5] Ready in {(time.perf_counter() - t0):.1f}s "
            f"(blocks={len(raw_tf.transformer_blocks)}, configs={len(configs)})",
            flush=True,
        )

    results: List[Dict[str, Any]] = []
    for cfg in configs:
        check_storage()
        h, w = int(cfg["height"]), int(cfg["width"])
        cfg_deg, sp_deg = int(cfg["cfg_degree"]), int(cfg["sp_degree"])
        total_gpus = int(cfg["gpus"])
        total_steps = int(cfg["total_steps"])
        warmup_steps = int(cfg["warmup_steps"])
        h_lat, w_lat = h // 8, w // 8
        tokens_valid = (h_lat // 2) * (w_lat // 2)
        mu = calculate_shift(tokens_valid, 256, 4096, 0.5, 1.15)
        use_row_split = (h_lat % (2 * sp_deg)) == 0

        dist.barrier()
        t_cfg0 = time.perf_counter()
        prompt_crit_steps: List[List[float]] = []
        prompt_crit_blocks: List[List[List[List[float]]]] = []
        max_peak_gib = 0.0

        if rank < total_gpus:
            mgr.activate_layout(cfg_deg, sp_deg, rstate)
            if use_row_split:
                rstate.set_input_parameters(
                    height=h,
                    width=w,
                    batch_size=1,
                    num_inference_steps=total_steps,
                    split_text_embed_in_sp=False,
                    split_latents_by_rows=True,
                )
            else:
                rstate.input_config.num_inference_steps = total_steps
                rstate.split_text_embed_in_sp = False
                rstate.split_latents_by_rows = False
                rstate.ready = False
            xpipe._guidance_scale = 4.5
            xpipe._clip_skip = None
            xpipe._joint_attention_kwargs = None
            xpipe._interrupt = False
            torch.cuda.reset_peak_memory_stats(device)

            def _run_sd35_prompt(p_entry: Dict[str, Any], steps_to_run: int, profile: bool) -> Optional[Tuple[List[float], List[List[List[float]]]]]:
                global _ACTIVE_SD35_VALID_KV_LEN
                timesteps, _ = retrieve_timesteps(raw_sched, 28, device, None, None, mu=mu)
                timesteps = timesteps[:steps_to_run]
                pe, ppe = xpipe._process_cfg_split_batch(
                    p_entry["neg_pe"], p_entry["pos_pe"], p_entry["neg_ppe"], p_entry["pos_ppe"]
                )
                gen = torch.Generator(device=device).manual_seed(1234)
                init_lat = torch.randn((1, 16, h_lat, w_lat), generator=gen, device=device, dtype=torch.bfloat16)

                if use_row_split:
                    _ACTIVE_SD35_VALID_KV_LEN = None
                    lat, pe_run = xpipe._init_sync_pipeline(init_lat, pe)
                    if not profile:
                        for t in timesteps:
                            last_lat = lat
                            noise_pred, _ = xpipe._backbone_forward(
                                latents=lat, encoder_hidden_states=pe_run, pooled_prompt_embeds=ppe, t=t
                            )
                            lat = xpipe._scheduler_step(noise_pred, last_lat, t)
                        torch.cuda.synchronize(device)
                        return None

                    with ZeroSyncBlockStepProfiler(
                        raw_tf.transformer_blocks,
                        total_steps=steps_to_run,
                        passes_per_step=1,
                        sync_fn=lambda: torch.cuda.synchronize(device),
                    ) as prof:
                        for s_idx, t in enumerate(timesteps):
                            prof.begin_step(s_idx)
                            prof.begin_pass(0)
                            last_lat = lat
                            noise_pred, _ = xpipe._backbone_forward(
                                latents=lat, encoder_hidden_states=pe_run, pooled_prompt_embeds=ppe, t=t
                            )
                            lat = xpipe._scheduler_step(noise_pred, last_lat, t)
                            prof.end_step(s_idx)
                        return prof.finalize(warmup_steps=warmup_steps)
                else:
                    # Token-strip SP sharding for non-row-divisible shapes (1440x1440 SP4 & SP8).
                    assert cfg_deg == 1, "Token-strip fallback only needed when cfg_deg == 1 and sp_deg in {4, 8}"
                    sp_rank = rank % sp_deg
                    pad_tokens = (sp_deg - (tokens_valid % sp_deg)) % sp_deg
                    n_padded = tokens_valid + pad_tokens
                    n_local = n_padded // sp_deg
                    local_pad_rows = max(0, (sp_rank + 1) * n_local - tokens_valid)
                    _ACTIVE_SD35_VALID_KV_LEN = tokens_valid if pad_tokens > 0 else None

                    full_strip = pack_sd35_latents_to_patch_strip(init_lat, patch_size=2, pad_tokens=pad_tokens)
                    lat_local = full_strip[:, :, :, 2 * sp_rank * n_local : 2 * (sp_rank + 1) * n_local].contiguous()
                    full_pos = raw_pos_embed.cropped_pos_embed(h_lat, w_lat)
                    if pad_tokens > 0:
                        pad_pe = torch.zeros(
                            (1, pad_tokens, full_pos.shape[-1]), dtype=full_pos.dtype, device=full_pos.device
                        )
                        full_pos = torch.cat([full_pos, pad_pe], dim=1)
                    pos_local = full_pos[:, sp_rank * n_local : (sp_rank + 1) * n_local, :].to(
                        device=device, dtype=torch.bfloat16
                    ).contiguous()

                    try:
                        def _token_step(lat_cur: Any, t_cur: Any) -> Any:
                            lat_in = torch.cat([lat_cur, lat_cur], dim=0)
                            t_in = t_cur.expand(2)
                            hs = (raw_proj_conv(lat_in).flatten(2).transpose(1, 2) + pos_local).to(lat_in.dtype)
                            temb = raw_tf.time_text_embed(t_in, ppe)
                            enc_hs = raw_tf.context_embedder(pe)
                            for blk in raw_tf.transformer_blocks:
                                enc_hs, hs = blk(hidden_states=hs, encoder_hidden_states=enc_hs, temb=temb)
                                if local_pad_rows > 0:
                                    hs[:, -local_pad_rows:, :].zero_()
                            hs = raw_tf.proj_out(raw_tf.norm_out(hs, temb))
                            out_strip = unpatchify_sd35_tokens_to_patch_strip(
                                hs, out_channels=int(raw_tf.out_channels), patch_size=2
                            )
                            nu, nc = out_strip.chunk(2, dim=0)
                            npred = nu + 4.5 * (nc - nu)
                            if local_pad_rows > 0:
                                npred[:, :, :, -2 * local_pad_rows :].zero_()
                            return raw_sched.step(npred, t_cur, lat_cur, return_dict=False)[0]

                        if not profile:
                            for t in timesteps:
                                lat_local = _token_step(lat_local, t)
                            torch.cuda.synchronize(device)
                            return None

                        with ZeroSyncBlockStepProfiler(
                            raw_tf.transformer_blocks,
                            total_steps=steps_to_run,
                            passes_per_step=1,
                            sync_fn=lambda: torch.cuda.synchronize(device),
                        ) as prof:
                            for s_idx, t in enumerate(timesteps):
                                prof.begin_step(s_idx)
                                prof.begin_pass(0)
                                lat_local = _token_step(lat_local, t)
                                prof.end_step(s_idx)
                            return prof.finalize(warmup_steps=warmup_steps)
                    finally:
                        _ACTIVE_SD35_VALID_KV_LEN = None

            with torch.inference_mode():
                # 1-step communicator warmup before timed prompt loop
                _ = _run_sd35_prompt(encoded_prompts[0], 1, False)

        for p_idx, p_entry in enumerate(encoded_prompts):
            local_payload: Optional[Dict[str, Any]] = None
            if rank < total_gpus:
                with torch.inference_mode():
                    res_pair = _run_sd35_prompt(p_entry, total_steps, True)
                    assert res_pair is not None
                    s_ms, b_ms = res_pair
                    local_payload = {
                        "step_ms": s_ms,
                        "block_ms": b_ms,
                        "peak_hbm_gib": torch.cuda.max_memory_allocated(device) / (1024.0**3),
                    }
            gathered: List[Optional[Dict[str, Any]]] = [None] * world_size
            dist.all_gather_object(gathered, local_payload)
            if rank == 0:
                active = [gathered[r] for r in range(total_gpus)]
                assert all(x is not None for x in active)
                reduced = reduce_critical_path_timings(active)  # type: ignore[arg-type]
                prompt_crit_steps.append(reduced["step_ms"])
                prompt_crit_blocks.append(reduced["block_ms"])
                max_peak_gib = max(max_peak_gib, reduced["peak_hbm_gib"])

        if rank == 0:
            rec = compute_config_cv_metrics(
                cfg,
                prompt_crit_steps,
                prompt_crit_blocks,
                max_peak_gib,
                prompt_ids=[x["sample_id"] for x in encoded_prompts],
            )
            results.append(rec)
            on_config_done(rec)
            print(
                f"[Progress][SD3.5] {cfg['resolution']} {cfg['form']} (gpus={total_gpus}) in "
                f"{(time.perf_counter() - t_cfg0):.1f}s | step_mean={rec['step_mean_ms']:.2f}ms "
                f"step_tCV={rec['step_temporal_cv_pct']:.2f}% xPrompt_CV={rec['cross_prompt_step_cv_pct']:.2f}% "
                f"blk_tCV={rec['block_temporal_cv_mean_pct']:.2f}% blk_hom_sCV={rec['block_spatial_cv_homogeneous_pct']:.2f}% "
                f"hbm={rec['peak_hbm_gib']:.2f}GiB",
                flush=True,
            )
        dist.barrier()

    del xpipe, raw_pipe
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return results


def run_wan22_configs(
    configs: Sequence[Dict[str, Any]],
    rank: int,
    world_size: int,
    device: Any,
    mgr: SubgroupCoordinatorManager,
    on_config_done: Callable[[Dict[str, Any]], None],
) -> List[Dict[str, Any]]:
    if not configs:
        return []

    import torch
    import torch.distributed as dist
    from diffusers import UniPCMultistepScheduler
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, UMT5EncoderModel
    from ditserve.models.pipeline import _sdpa_context
    from ditserve.models.wan.runtime import model_timestep
    from ditserve.models.xdit.runtime import engine_config
    from ditserve.models.xdit.wan import load_wan_transformer
    from xfuser.core.distributed.runtime_state import get_runtime_state, initialize_runtime_state

    torch.cuda.empty_cache()
    t0 = time.perf_counter()

    # Encode the 3 real VidProM prompts + negative prompt on Rank 0 using UMT5-XXL before
    # loading the 5B DiT transformer so UMT5 takes ~0.1s on GPU and 0 B during denoising.
    encoded_local: Optional[List[Dict[str, Any]]] = None
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(WAN22_PATH, subfolder="tokenizer", local_files_only=True)
        umt5 = UMT5EncoderModel.from_pretrained(
            WAN22_PATH, subfolder="text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
        ).to(device=device, dtype=torch.bfloat16)
        umt5.eval()

        def _encode_umt5(text: str, max_seq_len: int = 512) -> torch.Tensor:
            cleaned = prompt_clean(text)
            inputs = tokenizer(
                [cleaned],
                padding="max_length",
                max_length=max_seq_len,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids = inputs.input_ids.to(device)
            mask = inputs.attention_mask.to(device)
            seq_len = max(1, int(mask.gt(0).sum(dim=1).max().item()))
            with torch.inference_mode():
                hidden = umt5(ids[:, :seq_len], mask[:, :seq_len]).last_hidden_state
                valid = hidden[0, :seq_len]
                if seq_len < max_seq_len:
                    pad = torch.zeros(
                        (max_seq_len - seq_len, valid.shape[-1]), dtype=hidden.dtype, device=device
                    )
                    out = torch.cat([valid, pad], dim=0).unsqueeze(0)
                else:
                    out = valid.unsqueeze(0)
            return out.to(dtype=torch.float16).cpu().contiguous()

        neg_emb = _encode_umt5(WAN22_NEG_PROMPT)
        encoded_local = [
            {
                "sample_id": row["sample_id"],
                "pos_emb": _encode_umt5(row["prompt"]),
                "neg_emb": neg_emb,
            }
            for row in VIDPROM_PROMPTS
        ]
        del umt5, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    encoded_prompts = _broadcast_tensor_dict_list(encoded_local, ("pos_emb", "neg_emb"), rank, device)

    model_idx_path = Path(WAN22_PATH) / "model_index.json"
    expand_timesteps = bool(json.loads(model_idx_path.read_text()).get("expand_timesteps", True))
    vae_cfg_path = Path(WAN22_PATH) / "vae" / "config.json"
    vae_cfg = json.loads(vae_cfg_path.read_text()) if vae_cfg_path.is_file() else {}
    spatial_stride = int(vae_cfg.get("scale_factor_spatial", 16))
    temporal_stride = int(vae_cfg.get("scale_factor_temporal", 4))

    sched_proto = UniPCMultistepScheduler.from_pretrained(WAN22_PATH, subfolder="scheduler", local_files_only=True)
    sched_config = dict(sched_proto.config)

    wan22_tf = load_wan_transformer(WAN22_PATH, torch.float16, local_files_only=True).to(
        device=device, dtype=torch.float16
    )
    wan22_tf.eval()
    in_ch = int(wan22_tf.config.in_channels)

    class WanDummyPipeline:
        def __init__(self, transformer: Any):
            self.transformer = transformer
            self.vae_scale_factor_spatial = 8
            self.vae_scale_factor_temporal = 4

    mgr.activate_layout(1, 1)
    eng_cfg = engine_config("wan", world_size=1, attention="flash", dtype=torch.float16)
    initialize_runtime_state(pipeline=WanDummyPipeline(wan22_tf), engine_config=eng_cfg)
    rstate = get_runtime_state()
    dist.barrier()
    if rank == 0:
        print(
            f"[Progress][Wan2.2] Ready in {(time.perf_counter() - t0):.1f}s "
            f"(blocks={len(wan22_tf.blocks)}, in_ch={in_ch}, spatial_stride={spatial_stride}, "
            f"expand_ts={expand_timesteps}, configs={len(configs)})",
            flush=True,
        )

    results: List[Dict[str, Any]] = []
    for cfg in configs:
        check_storage()
        h, w, frames = int(cfg["height"]), int(cfg["width"]), int(cfg["frames"])
        cfg_deg, sp_deg = int(cfg["cfg_degree"]), int(cfg["sp_degree"])
        total_gpus = int(cfg["gpus"])
        total_steps = int(cfg["total_steps"])
        warmup_steps = int(cfg["warmup_steps"])
        passes_per_step = 2 if cfg_deg == 1 else 1

        f_lat = (frames - 1) // temporal_stride + 1
        h_lat = h // spatial_stride
        w_lat = w // spatial_stride

        dist.barrier()
        t_cfg0 = time.perf_counter()
        prompt_crit_steps: List[List[float]] = []
        prompt_crit_blocks: List[List[List[List[float]]]] = []
        max_peak_gib = 0.0

        if rank < total_gpus:
            mgr.activate_layout(cfg_deg, sp_deg, rstate)
            rstate.set_video_input_parameters(
                height=h,
                width=w,
                num_frames=frames,
                batch_size=1,
                num_inference_steps=50,
                split_text_embed_in_sp=False,
            )
            cfg_group = mgr.nccl_groups[(rank % sp_deg, (rank % sp_deg) + sp_deg)] if cfg_deg == 2 else None
            torch.cuda.reset_peak_memory_stats(device)

            def _run_wan_prompt(
                p_entry: Dict[str, Any], steps_to_run: int, profile: bool
            ) -> Optional[Tuple[List[float], List[List[List[float]]]]]:
                sched = UniPCMultistepScheduler.from_config(sched_config)
                sched.set_timesteps(50, device=device)
                timesteps = sched.timesteps[:steps_to_run]

                gen_cpu = torch.Generator(device="cpu").manual_seed(301)
                lat = torch.randn(
                    (1, in_ch, f_lat, h_lat, w_lat), generator=gen_cpu, dtype=torch.float32
                ).to(device=device)
                pos_emb = p_entry["pos_emb"]
                neg_emb = p_entry["neg_emb"]

                if not profile:
                    with _sdpa_context(torch, "flash"):
                        for t_scalar in timesteps:
                            ts_in = model_timestep(torch, lat, t_scalar, expand_timesteps)
                            lat_fp16 = lat.to(torch.float16)
                            if cfg_deg == 1:
                                oc = wan22_tf(
                                    hidden_states=lat_fp16, timestep=ts_in, encoder_hidden_states=pos_emb, return_dict=False
                                )[0]
                                ou = wan22_tf(
                                    hidden_states=lat_fp16, timestep=ts_in, encoder_hidden_states=neg_emb, return_dict=False
                                )[0]
                            else:
                                role = rank // sp_deg
                                enc = pos_emb if role == 0 else neg_emb
                                pred = wan22_tf(
                                    hidden_states=lat_fp16, timestep=ts_in, encoder_hidden_states=enc, return_dict=False
                                )[0].contiguous()
                                gathered_cfg = torch.empty((2, *pred.shape), dtype=pred.dtype, device=device)
                                dist.all_gather_into_tensor(gathered_cfg, pred, group=cfg_group)
                                oc, ou = gathered_cfg[0], gathered_cfg[1]
                            guided = ou + 5.0 * (oc - ou)
                            lat = sched.step(guided, t_scalar, lat, return_dict=False)[0]
                    torch.cuda.synchronize(device)
                    return None

                with ZeroSyncBlockStepProfiler(
                    wan22_tf.blocks,
                    total_steps=steps_to_run,
                    passes_per_step=passes_per_step,
                    sync_fn=lambda: torch.cuda.synchronize(device),
                ) as prof, _sdpa_context(torch, "flash"):
                    for s_idx, t_scalar in enumerate(timesteps):
                        prof.begin_step(s_idx)
                        ts_in = model_timestep(torch, lat, t_scalar, expand_timesteps)
                        lat_fp16 = lat.to(torch.float16)
                        if cfg_deg == 1:
                            prof.begin_pass(0)
                            oc = wan22_tf(
                                hidden_states=lat_fp16, timestep=ts_in, encoder_hidden_states=pos_emb, return_dict=False
                            )[0]
                            prof.begin_pass(1)
                            ou = wan22_tf(
                                hidden_states=lat_fp16, timestep=ts_in, encoder_hidden_states=neg_emb, return_dict=False
                            )[0]
                        else:
                            prof.begin_pass(0)
                            role = rank // sp_deg
                            enc = pos_emb if role == 0 else neg_emb
                            pred = wan22_tf(
                                hidden_states=lat_fp16, timestep=ts_in, encoder_hidden_states=enc, return_dict=False
                            )[0].contiguous()
                            gathered_cfg = torch.empty((2, *pred.shape), dtype=pred.dtype, device=device)
                            dist.all_gather_into_tensor(gathered_cfg, pred, group=cfg_group)
                            oc, ou = gathered_cfg[0], gathered_cfg[1]
                        guided = ou + 5.0 * (oc - ou)
                        lat = sched.step(guided, t_scalar, lat, return_dict=False)[0]
                        prof.end_step(s_idx)
                    return prof.finalize(warmup_steps=warmup_steps)

            with torch.inference_mode():
                # 1-step communicator warmup before timed prompt loop
                _ = _run_wan_prompt(encoded_prompts[0], 1, False)

        for p_entry in encoded_prompts:
            local_payload: Optional[Dict[str, Any]] = None
            if rank < total_gpus:
                with torch.inference_mode():
                    res_pair = _run_wan_prompt(p_entry, total_steps, True)
                    assert res_pair is not None
                    s_ms, b_ms = res_pair
                    local_payload = {
                        "step_ms": s_ms,
                        "block_ms": b_ms,
                        "peak_hbm_gib": torch.cuda.max_memory_allocated(device) / (1024.0**3),
                    }
            gathered: List[Optional[Dict[str, Any]]] = [None] * world_size
            dist.all_gather_object(gathered, local_payload)
            if rank == 0:
                active = [gathered[r] for r in range(total_gpus)]
                assert all(x is not None for x in active)
                reduced = reduce_critical_path_timings(active)  # type: ignore[arg-type]
                prompt_crit_steps.append(reduced["step_ms"])
                prompt_crit_blocks.append(reduced["block_ms"])
                max_peak_gib = max(max_peak_gib, reduced["peak_hbm_gib"])

        if rank == 0:
            rec = compute_config_cv_metrics(
                cfg,
                prompt_crit_steps,
                prompt_crit_blocks,
                max_peak_gib,
                prompt_ids=[x["sample_id"] for x in encoded_prompts],
            )
            results.append(rec)
            on_config_done(rec)
            print(
                f"[Progress][Wan2.2] {cfg['resolution']} {cfg['form']} (gpus={total_gpus}) in "
                f"{(time.perf_counter() - t_cfg0):.1f}s | step_mean={rec['step_mean_ms']:.1f}ms "
                f"step_tCV={rec['step_temporal_cv_pct']:.2f}% xPrompt_CV={rec['cross_prompt_step_cv_pct']:.2f}% "
                f"blk_tCV={rec['block_temporal_cv_mean_pct']:.2f}% blk_sCV={rec['block_spatial_cv_all_pct']:.2f}% "
                f"hbm={rec['peak_hbm_gib']:.2f}GiB",
                flush=True,
            )
        dist.barrier()

    del wan22_tf
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return results


def format_markdown_report(payload: Dict[str, Any]) -> str:
    lines = [
        f"# Motivation Step & Block Predictability (CV%) Report — Shard `{payload['shard']}` (`{payload['run_id']}`)",
        "",
        f"- **Timestamp (UTC)**: `{payload['timestamp_utc']}`",
        f"- **Host / Hardware**: `{payload['hostname']}` — `{payload['world_size']}x {payload['gpu_name']}`",
        f"- **Build**: `{payload['build']}`",
        f"- **Completed Configs**: `{len(payload['configs'])}`",
        "",
        "| Modality | Model | Resolution | Form | GPUs | Step Mean (ms) | Step P99/P50 | Step Temporal CV (%) | Cross-Prompt Step CV (%) | Block Mean (ms) | Block Temporal CV Mean / P95 / Max (%) | Block Spatial CV All / Homog (%) | Peak HBM (GiB) |",
        "| :--- | :--- | :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: |",
    ]
    for r in payload["configs"]:
        lines.append(
            f"| {r['modality']} | `{r['model']}` | `{r['resolution']}` | **{r['form']}** | {r['gpus']} | "
            f"{r['step_mean_ms']:.2f} | {r['step_p99_p50_ratio']:.4f} | **{r['step_temporal_cv_pct']:.2f}%** | "
            f"**{r['cross_prompt_step_cv_pct']:.2f}%** | {r['block_mean_ms']:.2f} | "
            f"{r['block_temporal_cv_mean_pct']:.2f}% / {r['block_temporal_cv_p95_pct']:.2f}% / {r['block_temporal_cv_max_pct']:.2f}% | "
            f"{r['block_spatial_cv_all_pct']:.2f}% / **{r['block_spatial_cv_homogeneous_pct']:.2f}%** | "
            f"{r['peak_hbm_gib']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard", choices=("s0", "s1", "s2", "all"), default=os.environ.get("DITSERVE_MOT_CV_SHARD", "all"))
    parser.add_argument("--report-name", default="")
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    check_storage()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    shard_configs = build_shard_matrix(args.shard)
    sd35_cfgs = [c for c in shard_configs if c["model"] == "SD3.5-Medium"]
    wan22_cfgs = [c for c in shard_configs if c["model"] == "Wan2.2-TI2V-5B"]

    out_dir = Path(args.output_dir)
    shard_json_path = out_dir / "mot_block_step_cv_shard.json"
    summary_json_path = out_dir / "summary.json"
    report_name = args.report_name or f"20261116-mot-block-step-cv-{args.shard}.md"

    hostname = subprocess.check_output(["hostname"], text=True).strip()
    gpu_name = torch.cuda.get_device_name(device)
    collected: List[Dict[str, Any]] = []

    def _flush_checkpoint(_latest: Dict[str, Any]) -> None:
        if local_rank != 0:
            return
        collected.append(_latest)
        out_dir.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema": "ditserve.mot-block-step-cv.v1",
            "run_id": os.environ.get("DITSERVE_RUN_ID", "local"),
            "build": os.environ.get("DITSERVE_BUILD", "unknown"),
            "shard": args.shard,
            "expected_configs": len(shard_configs),
            "completed_configs": len(collected),
            "hostname": hostname,
            "gpu_name": gpu_name,
            "world_size": world_size,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "prompts": {
                "image_diffusiondb": DIFFUSIONDB_PROMPTS,
                "video_vidprom": VIDPROM_PROMPTS,
            },
            "configs": collected,
        }
        tmp_path = out_dir / "mot_block_step_cv_shard.json.tmp"
        tmp_path.write_text(json.dumps(doc, indent=2) + "\n")
        os.replace(tmp_path, shard_json_path)

    if local_rank == 0:
        print(
            f"[Progress] Starting Motivation CV collector shard={args.shard} "
            f"(sd35={len(sd35_cfgs)}, wan22={len(wan22_cfgs)}, total={len(shard_configs)})",
            flush=True,
        )

    mgr = SubgroupCoordinatorManager(local_rank, world_size, device)
    if wan22_cfgs:
        run_wan22_configs(wan22_cfgs, local_rank, world_size, device, mgr, _flush_checkpoint)
    if sd35_cfgs:
        run_sd35_configs(sd35_cfgs, local_rank, world_size, device, mgr, _flush_checkpoint)

    if local_rank == 0:
        final_doc = json.loads(shard_json_path.read_text())
        summary_json_path.write_text(json.dumps(final_doc, indent=2) + "\n")
        md_text = format_markdown_report(final_doc)
        (out_dir / report_name).write_text(md_text)
        print(md_text, flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
