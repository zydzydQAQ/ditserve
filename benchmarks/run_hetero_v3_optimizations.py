#!/usr/bin/env python3
"""GPU v3 component microbenchmarks/replay and a separate 8-A100 simulation.

The preencoded, manually phased replay does not evaluate online scheduling.
The C++ study simulates eight devices; it is not an eight-GPU measurement.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ditserve.evaluation.hetero_metrics import (
    compute_psnr,
    compute_ssim_frame,
    compute_video_ssim,
    fast_prepare_wan as _fast_prepare_wan,
)
from ditserve.models.sd35 import SD35Backend
from ditserve.models.wan.backend import WanBackend
from ditserve.models.wan.cfg_reuse import AsymmetricCFGStepper
from ditserve.models.wan.taylor_cache import TaylorResidualCompensator
from ditserve.models.wan.vae_overlap import ConcurrentVAEDecodeStreamer
from run_hetero_8a100_study import evaluate_suite as evaluate_8a100_v3_suite
from run_hetero_quality_and_replay import (
    DENOISE_SCOPE, IMAGE_SCOPE, REPLAY_SCOPE, _completion_records, _decode_video,
    _image_requests, _replay_summary, _run_image_batch, _wait_until_arrival,
)


def _format_quality(value, digits=2):
    return "未评测" if value is None else f"{value:.{digits}f}"


WAN_PATH = os.environ.get(
    "WAN21_PATH",
    os.environ.get("WAN_MODEL_PATH", "/shared/data/DITSERVE/models/wan"),
)
SD35_PATH = os.environ.get(
    "SD35_PATH",
    os.environ.get("SD35_MODEL_PATH", "/shared/data/DITSERVE/models/sd3.5-medium"),
)
SD35_REVISION = os.environ.get(
    "SD35_REVISION",
    os.environ.get("SD35_MODEL_REVISION", "b940f670f0eda2d07fbb75229e779da1ad11eb80"),
)


def benchmark_opt9_asymmetric_cfg_and_taylor(
    wan: WanBackend,
    cached_wan_payload,
) -> Dict[str, Any]:
    """Benchmark Opt #9: Late-Step Asymmetric CFG Reuse + 1st-Order Taylor DBCache."""
    print("[Opt #9] Running Asymmetric Late-Step CFG Reuse + Taylor-DBCache benchmark...", flush=True)
    prompt = "A cinematic drone shot over a misty alpine lake at sunrise, pine forest, crystal water"
    neg_prompt = "blurry, low quality, distorted, watermark"
    base_spec = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "height": 480,
        "width": 832,
        "frames": 17,
        "steps": 20,
        "guidance_scale": 5.0,
        "seed": 42,
    }

    # Warmup
    w_state = _fast_prepare_wan(wan, dict(base_spec, steps=2, cache=False), "wan-opt9-warm", cached_wan_payload)
    wan.advance({"id": "w", "state_id": "wan-opt9-warm", "steps": 2})
    wan.release({"id": "w", "state_id": "wan-opt9-warm"})
    torch.cuda.synchronize()

    # 1. Exact Reference (tau=0.00, 20 steps * 2 CFG branches = 40 full passes)
    ref_id = "wan-opt9-ref"
    ref_state = _fast_prepare_wan(wan, dict(base_spec, cache=False, threshold=0.0), ref_id, cached_wan_payload)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wan.advance({"id": "r", "state_id": ref_id, "steps": 20})
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) * 1000.0
    ref_latents = ref_state.latents.detach().cpu().float().clone()
    ref_video = _decode_video(wan, ref_state)
    wan.release({"id": "r", "state_id": ref_id})

    rows: List[Dict[str, Any]] = [
        {
            "mode": "Exact-FullCFG (Baseline)",
            "threshold": 0.0,
            "folded_cfg_steps": 0,
            "reused_cache_steps": 0,
            "effective_branch_forwards": 40,
            "denoise_ms": round(ref_ms, 2),
            "speedup": 1.0,
            "latent_cos_sim": 1.0,
            "latent_rmse": 0.0,
            "video_psnr_db": 99.99,
            "video_ssim": 1.0,
        }
    ]

    # 2. Asymmetric CFG Reuse Only (No DBCache, fold uncond on steps {14, 15, 16, 17, 18}, step 19 anchor)
    cfg_only_id = "wan-opt9-cfgonly"
    cfg_state = _fast_prepare_wan(wan, dict(base_spec, cache=False, threshold=0.0), cfg_only_id, cached_wan_payload)
    cfg_stepper = AsymmetricCFGStepper(
        torch,
        folded_steps={14, 15, 16, 17, 18},
        cfg_damping=0.96,
        cfg_vel_alpha=0.15,
        final_step_exact_anchor=True,
        enabled=True,
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        cfg_stepper.step(wan.runtime, cfg_state, taylor_compensator=None)
    torch.cuda.synchronize()
    cfg_only_ms = (time.perf_counter() - t0) * 1000.0
    cfg_latents = cfg_state.latents.detach().cpu().float().clone()
    cfg_video = _decode_video(wan, cfg_state)
    wan.release({"id": "c", "state_id": cfg_only_id})

    cfg_cos = float(F.cosine_similarity(ref_latents.flatten().unsqueeze(0), cfg_latents.flatten().unsqueeze(0)).item())
    cfg_rmse = float(torch.sqrt(torch.mean((ref_latents - cfg_latents) ** 2)).item())
    cfg_psnr = compute_psnr(ref_video, cfg_video)
    cfg_ssim = compute_video_ssim(ref_video, cfg_video)
    rows.append(
        {
            "mode": "AsymmetricCFG-Only (Fold Steps 14..18)",
            "threshold": 0.0,
            "folded_cfg_steps": cfg_stepper.stats["folded_cfg_steps"],
            "reused_cache_steps": 0,
            "effective_branch_forwards": 40 - cfg_stepper.stats["saved_uncond_forwards"],
            "denoise_ms": round(cfg_only_ms, 2),
            "speedup": round(ref_ms / cfg_only_ms, 3),
            "latent_cos_sim": round(cfg_cos, 6),
            "latent_rmse": round(cfg_rmse, 6),
            "video_psnr_db": cfg_psnr,
            "video_ssim": cfg_ssim,
        }
    )
    print(
        f"  [Opt #9] AsymmetricCFG-Only: {cfg_only_ms:.1f} ms ({ref_ms/cfg_only_ms:.2f}x), "
        f"folded={cfg_stepper.stats['folded_cfg_steps']} steps, PSNR={cfg_psnr:.2f} dB, SSIM={cfg_ssim:.4f}",
        flush=True,
    )

    # 3. Co-Execution: Taylor1stOrder-DBCache + AsymmetricCFGStepper across tau in {0.12, 0.16, 0.24}
    for tau, folded_set, exact_cond in (
        (0.12, {17, 18}, True),
        (0.16, {17, 18}, True),
        (0.16, {14, 15, 16, 17, 18}, False),
        (0.24, {16, 17, 18}, True),
    ):
        co_id = f"wan-opt9-co-{int(tau*100)}-{len(folded_set)}-{int(exact_cond)}"
        co_state = _fast_prepare_wan(
            wan,
            dict(base_spec, cache=True, threshold=tau, max_continuous=2),
            co_id,
            cached_wan_payload,
        )
        comp = TaylorResidualCompensator(
            torch,
            alpha=-0.22,
            damping=0.70,
            max_norm_ratio=0.10,
            adaptive_lsq=False,
            enabled=True,
        )
        comp.monotonic_gate = False
        stepper = AsymmetricCFGStepper(
            torch,
            folded_steps=folded_set,
            cfg_damping=0.97,
            cfg_vel_alpha=0.0,
            final_step_exact_anchor=True,
            enabled=True,
        )
        stepper.force_exact_cond_on_fold = exact_cond
        if not exact_cond:
            stepper.pre_fold_anchor_step = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        events = []
        for _ in range(20):
            events.append(stepper.step(wan.runtime, co_state, taylor_compensator=comp))
        torch.cuda.synchronize()
        co_ms = (time.perf_counter() - t0) * 1000.0
        co_latents = co_state.latents.detach().cpu().float().clone()
        co_video = _decode_video(wan, co_state)
        wan.release({"id": "co", "state_id": co_id})

        reused_steps = sum(1 for ev in events if ev.get("reused_forwards", 0) > 0)
        folded_steps_cnt = stepper.stats["folded_cfg_steps"]
        co_cos = float(F.cosine_similarity(ref_latents.flatten().unsqueeze(0), co_latents.flatten().unsqueeze(0)).item())
        co_rmse = float(torch.sqrt(torch.mean((ref_latents - co_latents) ** 2)).item())
        co_psnr = compute_psnr(ref_video, co_video)
        co_ssim = compute_video_ssim(ref_video, co_video)

        row = {
            "mode": f"Taylor1stOrder + AsymmetricCFG (Fold {min(folded_set)}..{max(folded_set)})",
            "threshold": tau,
            "folded_cfg_steps": folded_steps_cnt,
            "reused_cache_steps": reused_steps,
            "effective_branch_forwards": max(1, 40 - reused_steps * 2 - folded_steps_cnt),
            "denoise_ms": round(co_ms, 2),
            "speedup": round(ref_ms / co_ms, 3),
            "latent_cos_sim": round(co_cos, 6),
            "latent_rmse": round(co_rmse, 6),
            "video_psnr_db": co_psnr,
            "video_ssim": co_ssim,
        }
        rows.append(row)
        print(
            f"  [Opt #9] Taylor+AsymmetricCFG (tau={tau:.2f}): {co_ms:.1f} ms ({ref_ms/co_ms:.2f}x speedup), "
            f"cache_reused={reused_steps}/20, cfg_folded={folded_steps_cnt}/20, "
            f"PSNR={co_psnr:.2f} dB, SSIM={co_ssim:.4f}, cos_sim={co_cos:.6f}",
            flush=True,
        )

    for row in rows:
        row.update(evidence_type="measured_microbenchmark", timing_scope=DENOISE_SCOPE,
                   reused_steps=row["reused_cache_steps"], workload_spec=base_spec,
                   quality_status="reference" if row is rows[0] else "measured_comparison")
    return {"evidence_type": "measured_microbenchmark", "timing_scope": DENOISE_SCOPE, "modes": rows}


def benchmark_opt10_bf16_3dvae_decode(
    wan: WanBackend,
    sd35: SD35Backend,
    cached_wan_payload,
) -> Dict[str, Any]:
    """Benchmark Opt #10: BF16 Tensor-Core Chunked 3D-VAE Decode vs FP32 3D-VAE Decode."""
    print("[Opt #10] Running Wan2.1 BF16 Tensor-Core vs FP32 Chunked 3D-VAE Decode benchmark...", flush=True)
    wan_spec = {
        "prompt": "A cinematic drone shot over a misty alpine lake at sunrise, pine forest, crystal water",
        "negative_prompt": "blurry, low quality, distorted, watermark",
        "height": 480,
        "width": 832,
        "frames": 17,
        "steps": 4,
        "guidance_scale": 5.0,
        "seed": 42,
        "cache": False,
    }

    # Prepare a reference latent state after 4 denoising steps
    s_fp32 = _fast_prepare_wan(wan, wan_spec, "wan-vae-fp32", cached_wan_payload)
    wan.advance({"id": "f32", "state_id": "wan-vae-fp32", "steps": 4})
    saved_latents = s_fp32.latents.detach().clone()

    # 1. FP32 Chunked 3D-VAE Decode (Baseline)
    wan.runtime.pipe.vae.to(dtype=torch.float32)
    # Warmup FP32 VAE
    _decode_video(wan, s_fp32, vae_dtype="float32")
    s_fp32.finished = False
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    video_fp32 = _decode_video(wan, s_fp32, vae_dtype="float32")
    torch.cuda.synchronize()
    fp32_vae_ms = (time.perf_counter() - t0) * 1000.0
    fp32_peak_hbm = round(torch.cuda.max_memory_allocated() / (1 << 20), 2)
    wan.release({"id": "f32", "state_id": "wan-vae-fp32"})

    # 2. BF16 Tensor-Core Chunked 3D-VAE Decode (Opt #10)
    wan.runtime.pipe.vae.to(dtype=torch.bfloat16)
    s_bf16 = _fast_prepare_wan(wan, wan_spec, "wan-vae-bf16", cached_wan_payload)
    s_bf16.latents = saved_latents.clone()
    s_bf16.step_index = 4
    # Warmup BF16 VAE
    _decode_video(wan, s_bf16, vae_dtype="bfloat16")
    s_bf16.finished = False
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    video_bf16 = _decode_video(wan, s_bf16, vae_dtype="bfloat16")
    torch.cuda.synchronize()
    bf16_vae_ms = (time.perf_counter() - t0) * 1000.0
    bf16_peak_hbm = round(torch.cuda.max_memory_allocated() / (1 << 20), 2)
    wan.release({"id": "bf16", "state_id": "wan-vae-bf16"})

    bf16_psnr = compute_psnr(video_fp32, video_bf16)
    bf16_ssim = compute_video_ssim(video_fp32, video_bf16)
    bf16_max_diff = float(np.max(np.abs(video_fp32.astype(np.float32) - video_bf16.astype(np.float32))))
    vae_speedup = fp32_vae_ms / bf16_vae_ms

    # 3. BF16 3D-VAE + Stream-Isolated Overlap with SD3.5 Denoising
    img_prompt = "A futuristic neon city skyline at dusk, cinematic lighting, 8k"
    sd35._ensure_staged(img_prompt, "", 256, True)
    sd35.offload_encoders({})

    s_ovl = _fast_prepare_wan(wan, wan_spec, "wan-vae-bf16-ovl", cached_wan_payload)
    s_ovl.latents = saved_latents.clone()
    s_ovl.step_index = 4
    streamer = ConcurrentVAEDecodeStreamer(wan.runtime, chunk_frames=1, vae_dtype="bfloat16")
    try:
        ovl_res = streamer.run_overlapped(
            s_ovl,
            lambda: sd35.run_micro_batch({
                "prompts": [img_prompt],
                "seeds": [101],
                "height": 512,
                "width": 512,
                "steps": 12,
                "attention": "flash",
            }),
            chunk_frames=1,
        )
        ovl_res.pop("video_np", None)
    finally:
        streamer.close()
        wan.release({"id": "ovl", "state_id": "wan-vae-bf16-ovl"})

    print(
        f"  [Opt #10] 3D-VAE Chunked Decode (5 chunks, 17f): FP32={fp32_vae_ms:.1f} ms -> "
        f"BF16 TensorCore={bf16_vae_ms:.1f} ms ({vae_speedup:.2f}x speedup, "
        f"PSNR={bf16_psnr:.2f} dB, SSIM={bf16_ssim:.5f}, Peak HBM={bf16_peak_hbm:.1f} MB) | "
        f"BF16-VAE + SD3.5 Overlapped Wall={ovl_res['wall_ms']:.1f} ms",
        flush=True,
    )

    return {
        "evidence_type": "measured_microbenchmark",
        "timing_scope": "complete decode of precomputed latents; excludes encoding and denoise",
        "overlap_timing_scope": "preencoded: excludes encoding and video denoise; video decode + image pipeline to NumPy",
        "quality_status": "measured_comparison; FP32 self row is reference only",
        "fp32_vae_decode_ms": round(fp32_vae_ms, 2),
        "fp32_peak_hbm_mb": fp32_peak_hbm,
        "bf16_vae_decode_ms": round(bf16_vae_ms, 2),
        "bf16_peak_hbm_mb": bf16_peak_hbm,
        "vae_speedup": round(vae_speedup, 3),
        "saved_vae_ms": round(fp32_vae_ms - bf16_vae_ms, 2),
        "bf16_vs_fp32_psnr_db": bf16_psnr,
        "bf16_vs_fp32_ssim": bf16_ssim,
        "bf16_vs_fp32_max_abs_diff": round(bf16_max_diff, 6),
        "bf16_stream_overlapped_with_sd35_wall_ms": ovl_res["wall_ms"],
        "bf16_stream_overlapped_peak_hbm_mb": ovl_res["peak_hbm_mb"],
    }


def benchmark_opt11_sd35_flash_channels_last_and_cache(sd35: SD35Backend) -> Dict[str, Any]:
    """Benchmark Opt #11: SD3.5 SDPA_FLASH + Channels-Last 2D-VAE + MM-DiT Block Cache + MicroBatch."""
    print("[Opt #11] Running SD3.5 SDPA_FLASH + Channels-Last 2D-VAE + MM-DiT Cache benchmark...", flush=True)
    prompts = [
        "A futuristic neon city skyline at dusk, cinematic lighting, 8k",
        "An astronaut riding a white horse on Mars, photorealistic",
        "A serene Japanese garden in autumn with red maple leaves",
        "A vintage steam locomotive crossing a snowy mountain viaduct",
    ]
    seeds = [101, 202, 303, 404]
    for p in prompts:
        sd35._ensure_staged(p, "", 256, True)
    sd35.offload_encoders({})
    torch.cuda.empty_cache()

    # Warmup
    sd35.run_micro_batch({"prompts": [prompts[0]], "seeds": [101], "height": 512, "width": 512, "steps": 4, "attention": "flash"})
    torch.cuda.synchronize()

    requests = _image_requests(prompts, seeds)

    def run(members, **options):
        torch.cuda.synchronize()
        return _run_image_batch(sd35, members, time.perf_counter(), **options)

    if getattr(sd35.pipe, "vae", None) is not None:
        sd35.pipe.vae.to(memory_format=torch.contiguous_format)
    res_math = run(requests[:1], attention="math", cache=False)
    img_ref = res_math.pop("images")
    torch.cuda.synchronize()
    seq_start = time.perf_counter()
    seq_completions = []
    for request in requests:
        batch = _run_image_batch(sd35, [request], seq_start, attention="math", cache=False)
        batch.pop("images")
        seq_completions.extend(batch["requests"])
    seq4_math_ms = max(r["finish_ms"] for r in seq_completions)

    if getattr(sd35.pipe, "vae", None) is not None:
        sd35.pipe.vae.to(memory_format=torch.channels_last)
    res_flash_cl = run(requests[:1], attention="flash", cache=False)
    img_flash_cl = res_flash_cl.pop("images")
    cache_options = dict(attention="flash", cache=True, threshold=0.25,
                         cache_warmup=4, cache_max_continuous=1)
    res_flash_cache = run(requests[:1], **cache_options)
    img_flash_cache = res_flash_cache.pop("images")
    res_mb4_exact = run(requests, attention="flash", cache=False)
    imgs_mb4_exact = res_mb4_exact.pop("images")
    res_mb4_cache = run(requests, **cache_options)
    imgs_mb4_cache = res_mb4_cache.pop("images")

    modes = []
    for name, result, cache, reference, output, quality_status, quality_reference, matched in (
        ("SDPA_MATH + Contiguous-VAE (Baseline BS=1)", res_math, False,
         None, None, "reference", "self; not an independent quality pass", res_math),
        ("SDPA_FLASH + ChannelsLast-2DVAE (BS=1)", res_flash_cl, False,
         img_ref, img_flash_cl, "measured_comparison", "SDPA_MATH BS1", res_math),
        ("SDPA_FLASH + ChannelsLast-2DVAE + MM-DiT-DBCache (BS=1, tau=0.25)", res_flash_cache, True,
         img_flash_cl, img_flash_cache, "measured_comparison", "SDPA_FLASH exact BS1", res_flash_cl),
        ("MicroBatch-BS4 + SDPA_FLASH + ChannelsLast-2DVAE (4 Prompts)", res_mb4_exact, False,
         None, None, "not_evaluated", "not compared with sequential outputs", res_mb4_exact),
        ("MicroBatch-BS4 + SDPA_FLASH + ChannelsLast-2DVAE + MM-DiT-DBCache (tau=0.25)", res_mb4_cache, True,
         imgs_mb4_exact, imgs_mb4_cache, "measured_comparison", "SDPA_FLASH exact BS4", res_mb4_exact),
    ):
        count = len(result["requests"])
        latency = result["batch_latency_ms"]
        baseline_ms = res_math["batch_latency_ms"] if count == 1 else seq4_math_ms
        reused = result.get("reused_steps")
        reuse_observed = reused is not None and reused > 0
        row = dict(
            evidence_type="measured_microbenchmark", timing_scope=IMAGE_SCOPE,
            mode=name, batch_size=count, cache=cache, reused_steps=reused,
            cache_reuse_observed=reuse_observed,
            cache_config=cache_options if cache else None,
            requests=result["requests"], output_count=sum(r["output_count"] for r in result["requests"]),
            batch_latency_ms=latency, amortized_ms_per_image=latency / count,
            throughput_img_s=1000.0 * count / latency,
            speedup_vs_baseline=baseline_ms / latency,
            baseline_scope="measured same-workload sequential SDPA_MATH; includes non-cache changes",
            cache_speedup_vs_matched_exact=matched["batch_latency_ms"] / latency if cache and reuse_observed else None,
            quality_status=quality_status, quality_reference=quality_reference,
            independent_quality_pass=None,
            psnr_db=compute_psnr(reference, output) if reference is not None else None,
            ssim=round(float(np.mean([compute_ssim_frame(reference[i], output[i])
                                    for i in range(count)])), 5) if reference is not None else None,
        )
        modes.append(row)
        print(f"  [Opt #11] {name}: batch latency={latency:.1f} ms, "
              f"amortized={row['amortized_ms_per_image']:.1f} ms/img, reused={reused}/12, "
              f"PSNR={_format_quality(row['psnr_db'])}, SSIM={_format_quality(row['ssim'], 4)} "
              f"({quality_status})", flush=True)
    return {"evidence_type": "measured_microbenchmark", "timing_scope": IMAGE_SCOPE,
            "modes": modes, "sequential_four_image_baseline": {
                "total_ms": seq4_math_ms, "requests": seq_completions,
                "output_count": sum(r["output_count"] for r in seq_completions)}}


def benchmark_e2e_v3_trace_replay(
    wan: WanBackend,
    sd35: SD35Backend,
    cached_wan_payload,
) -> Dict[str, Any]:
    """Preencoded component replay with manual phases, not an online scheduler."""
    print("[Component Replay] Running v3 manual mixed-modality phases...", flush=True)
    # Workload:
    #   t = 0 ms:     Wan2.1 video request (832x480, 17 frames, 20 steps, SLO = 17,000 ms)
    #   t = 1,200 ms: Burst of 2 SD3.5 image requests (img-1 @ 1200ms, img-2 @ 1350ms; SLO = 4,500 ms)
    #   t = 3,500 ms: SD3.5 image request (img-3 @ 3500ms; SLO = 4,500 ms)
    wan_spec = {
        "prompt": "A cinematic drone shot over a misty alpine lake at sunrise, pine forest, crystal water",
        "negative_prompt": "blurry, low quality, distorted, watermark",
        "height": 480,
        "width": 832,
        "frames": 17,
        "steps": 20,
        "guidance_scale": 5.0,
        "seed": 42,
        "cache": True,
        "threshold": 0.16,
        "max_continuous": 2,
    }
    img_prompts = [
        "A futuristic neon city skyline at dusk, cinematic lighting, 8k",
        "An astronaut riding a white horse on Mars, photorealistic",
        "A serene Japanese garden in autumn with red maple leaves",
    ]
    for p in img_prompts:
        sd35._ensure_staged(p, "", 256, True)
    sd35.offload_encoders({})
    wan.runtime.pipe.vae.to(dtype=torch.bfloat16)
    if getattr(sd35.pipe, "vae", None) is not None:
        sd35.pipe.vae.to(memory_format=torch.channels_last)
    torch.cuda.empty_cache()

    images = _image_requests(img_prompts, [101, 202, 303], [1200.0, 1350.0, 3500.0])
    video_request = dict(request_id="v0", modality="video", arrival_ms=0.0, spec=wan_spec)
    v_id = "v3-replay-wan"
    comp = TaylorResidualCompensator(
        torch,
        alpha=-0.22,
        damping=0.70,
        max_norm_ratio=0.10,
        adaptive_lsq=False,
        enabled=True,
    )
    comp.monotonic_gate = False
    stepper = AsymmetricCFGStepper(
        torch,
        folded_steps={14, 15, 16, 17, 18},
        cfg_damping=0.97,
        cfg_vel_alpha=0.0,
        final_step_exact_anchor=True,
        enabled=True,
    )
    stepper.force_exact_cond_on_fold = False
    stepper.pre_fold_anchor_step = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    v_start_ms = _wait_until_arrival(t_start, video_request["arrival_ms"])
    v_state = _fast_prepare_wan(wan, wan_spec, v_id, cached_wan_payload)
    completions, events, image_reused_steps = [], [], []
    try:
        for steps, members in ((3, images[:2]), (4, images[2:])):
            for _ in range(steps):
                events.append(stepper.step(wan.runtime, v_state, taylor_compensator=comp))
            batch = _run_image_batch(sd35, members, t_start, attention="flash", cache=True,
                                     threshold=0.25, cache_warmup=4, cache_max_continuous=1)
            batch.pop("images")
            completions.extend(batch["requests"])
            image_reused_steps.append(batch.get("reused_steps"))
        for _ in range(13):
            events.append(stepper.step(wan.runtime, v_state, taylor_compensator=comp))
        video = _decode_video(wan, v_state, vae_dtype="bfloat16")
        torch.cuda.synchronize()
        finish_ms = (time.perf_counter() - t_start) * 1000.0
        completions.extend(_completion_records([video_request], video, v_start_ms, finish_ms))
        peak_hbm_mb = round(torch.cuda.max_memory_allocated() / (1 << 20), 2)
    finally:
        wan.release({"id": "v3", "state_id": v_id})

    res = _replay_summary(completions, 4500.0, 17000.0)
    res.update(policy="v3 component replay: Taylor + CFGReuse + BF16 decode + Flash/Cache/MicroBatch",
               peak_hbm_mb=peak_hbm_mb, quality_status="not_evaluated_in_replay",
               reused_steps=sum(ev.get("reused_forwards", 0) > 0 for ev in events),
               image_batch_reused_steps=image_reused_steps,
               wan_folded_cfg_steps=stepper.stats["folded_cfg_steps"],
               wan_saved_uncond_forwards=stepper.stats["saved_uncond_forwards"])
    print(f"  [Component Replay] v3: Image max (n=3)={res['image_max_latency_ms']:.1f} ms, "
          f"Video={res['video_latency_ms']:.1f} ms", flush=True)
    return res


def write_markdown_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# HeteroDiT v3 GPU Component Microbenchmarks/Replay & Separate 8-A100 Simulation",
        "", f"Component device: `{payload['device_name']}`. The eight-device study is simulation, not real eight-GPU evidence.",
        "", "## 1. CFG reuse / Taylor: measured_microbenchmark", "", DENOISE_SCOPE,
        "Reference/self rows are not independent quality passes.",
        "",
        "| Mode | Tau | Reused steps | Folded CFG steps | Denoise (ms) | Speedup | PSNR (dB) | SSIM | Quality status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in payload["opt9_asymmetric_cfg_taylor"]["modes"]:
        lines.append(f"| {r['mode']} | {r['threshold']:.2f} | {r['reused_steps']}/20 | "
                     f"{r['folded_cfg_steps']}/20 | {r['denoise_ms']:.1f} | {r['speedup']:.2f}x | "
                     f"{_format_quality(r['video_psnr_db'])} | {_format_quality(r['video_ssim'], 4)} | {r['quality_status']} |")
    v10 = payload["opt10_bf16_3dvae"]
    lines.extend([
        "", "## 2. VAE decode: measured_microbenchmark", "", v10["timing_scope"],
        "| Mode | Decode (ms) | Peak HBM (MB) | PSNR vs FP32 | SSIM vs FP32 |",
        "| --- | ---: | ---: | --- | --- |",
        f"| FP32 reference | {v10['fp32_vae_decode_ms']:.1f} | {v10['fp32_peak_hbm_mb']:.1f} | reference | reference |",
        f"| BF16 | {v10['bf16_vae_decode_ms']:.1f} | {v10['bf16_peak_hbm_mb']:.1f} | "
        f"{_format_quality(v10['bf16_vs_fp32_psnr_db'])} | {_format_quality(v10['bf16_vs_fp32_ssim'], 5)} |",
        "", v10["overlap_timing_scope"],
        f"BF16 decode + SD3.5 overlap wall: {v10['bf16_stream_overlapped_with_sd35_wall_ms']:.1f} ms.",
        "", "## 3. SD3.5: measured_microbenchmark", "", IMAGE_SCOPE,
        "Amortized cost is not latency; batch completion latency and per-request timestamps are retained.",
        "BS4 speedup uses an actual four-prompt sequential baseline, not four times one image.",
        "Quality 未评测/null is not a pass. Cache benefit requires observed reused_steps and a matched exact baseline.",
        "",
        "| Mode | Batch size | Batch latency (ms) | Amortized ms/image | Reused steps | Speedup vs sequential math | Cache speedup vs matched exact | PSNR | SSIM | Quality status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ])
    for r in payload["opt11_sd35_flash_channels_last_cache"]["modes"]:
        lines.append(f"| {r['mode']} | {r['batch_size']} | {r['batch_latency_ms']:.1f} | "
                     f"{r['amortized_ms_per_image']:.1f} | {r['reused_steps']} | {r['speedup_vs_baseline']:.2f}x | "
                     f"{_format_quality(r['cache_speedup_vs_matched_exact'])} | {_format_quality(r['psnr_db'])} | "
                     f"{_format_quality(r['ssim'], 4)} | {r['quality_status']} |")
    replay = payload["component_replay_v3"]
    lines.extend([
        "", "## 4. Manually Ordered Component Replay: component_replay", "", REPLAY_SCOPE,
        "Not online scheduling evidence. Image max uses n=3, not a P99 estimate; replay quality was not evaluated.",
        f"{replay['policy']}: image mean {replay['image_mean_latency_ms']:.1f} ms, "
        f"image max (n=3) {replay['image_max_latency_ms']:.1f} ms, video {replay['video_latency_ms']:.1f} ms, "
        f"video reused steps {replay['reused_steps']}/20, image batch reused steps {replay['image_batch_reused_steps']}, "
        f"total SLO {replay['total_slo_attainment_pct']:.1f}%.",
        "", "## 5. Eight-A100 control-plane simulation: simulation (not measured eight-GPU execution)", "",
        "| Simulated ablation stage | Total SLO (%) | Image SLO (%) | Video SLO (%) | Simulated image P99 (ms) | Simulated video P99 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in payload["opt12_8a100_study"]["ablation"]["policies"].values():
        lines.append(f"| {row['label']} | {row['total_slo_pct']:.2f} | {row['image_slo_pct']:.2f} | "
                     f"{row['video_slo_pct']:.2f} | {row['image_p99_ms']:.1f} | {row['video_p99_ms']:.1f} |")
    (out_dir / "V3_OPTIMIZATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("RUN_DIR", "/mnt/e/000_code/zyd-2026/ditserve-hetero/results/v3-opt")),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"=== Running HeteroDiT v3 Optimizations (Opts #9-#12) on {device_name} ===", flush=True)

    # 1. Load Wan2.1 and SD3.5 on GPU 0
    print("[Init] Loading Wan2.1-T2V-1.3B and SD3.5-Medium on cuda:0...", flush=True)
    wan = WanBackend({
        "device": device,
        "model_path": WAN_PATH,
        "attention": "flash",
        "vae_dtype": "float32",
        "options": {"engine": "xdit", "execution": "step", "max_pinned_mb": 4096},
    })
    from ditserve.models.wan.runtime import normalize_spec

    base_wan_spec = normalize_spec({
        "request_id": "preencode-wan-v3",
        "prompt": "A cinematic drone shot over a misty alpine lake at sunrise, pine forest, crystal water",
        "negative_prompt": "blurry, low quality, distorted, watermark",
        "height": 480,
        "width": 832,
        "frames": 17,
        "steps": 20,
        "guidance_scale": 5.0,
        "seed": 42,
        "cache": False,
    })
    cached_wan_payload = wan.runtime.encode_prompt(base_wan_spec)

    # 2. Opt #9: Asymmetric Late-Step CFG Reuse + 1st-Order Taylor DBCache Co-Execution
    opt9_res = benchmark_opt9_asymmetric_cfg_and_taylor(wan, cached_wan_payload)

    # 3. Load SD3.5-Medium alongside Wan2.1
    sd35 = SD35Backend({
        "device": device,
        "model_path": SD35_PATH,
        "revision": SD35_REVISION,
        "dtype": "bfloat16",
        "attention": "flash",
        "vae_channels_last": True,
        "options": {
            "offload_text_encoders_after_warmup": False,
            "encoder_pool": {"max_entries": 64, "max_bytes": 2 * (1024 ** 3)},
        },
    })

    # 4. Opt #10: BF16 Tensor-Core Chunked 3D-VAE Decode vs FP32 3D-VAE Decode + Stream Overlap
    opt10_res = benchmark_opt10_bf16_3dvae_decode(wan, sd35, cached_wan_payload)

    # 5. Opt #11: SD3.5 SDPA_FLASH + Channels-Last 2D-VAE + MM-DiT Block Cache + Micro-Batching
    opt11_res = benchmark_opt11_sd35_flash_channels_last_and_cache(sd35)

    # 6. E2E Mixed-Modality Trace Replay (HeteroDiT-v3)
    e2e_v3_res = benchmark_e2e_v3_trace_replay(wan, sd35, cached_wan_payload)

    # This calls the control-plane simulation, not an eight-GPU job.
    print("[Opt #12] Running simulated eight-A100 control-plane ablation...", flush=True)
    opt12_res = dict(evaluate_8a100_v3_suite(), evidence_type="simulation",
                     timing_scope="simulated request service times, not GPU wall-clock measurements")

    payload = {
        "device_name": device_name,
        "evidence_types": ["measured_microbenchmark", "component_replay", "simulation"],
        "opt9_asymmetric_cfg_taylor": opt9_res,
        "opt10_bf16_3dvae": opt10_res,
        "opt11_sd35_flash_channels_last_cache": opt11_res,
        "component_replay_v3": e2e_v3_res,
        "opt12_8a100_study": opt12_res,
    }
    out_json = args.out_dir / "V3_OPTIMIZATIONS.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(args.out_dir, payload)

    sd35.close()
    wan.close()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"=== Completed HeteroDiT v3 Optimizations -> {out_json} ===", flush=True)


if __name__ == "__main__":
    main()
