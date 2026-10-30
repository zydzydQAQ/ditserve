#!/usr/bin/env python3
"""GPU v2 component microbenchmarks, estimated communication and component replay.

The manually phased, preencoded replay is not an online scheduling evaluation.
INT8 codec measurements use synthetic tensors on one device, not network traffic.
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
from ditserve.models.wan.ring_quant import AsymmetricRingQuantizer
from ditserve.models.wan.taylor_cache import TaylorResidualCompensator
from ditserve.models.wan.vae_overlap import ConcurrentVAEDecodeStreamer
from run_hetero_quality_and_replay import (
    DENOISE_SCOPE, IMAGE_SCOPE, REPLAY_SCOPE, _completion_records, _decode_video,
    _image_requests, _replay_summary, _run_image_batch, _wait_until_arrival,
)

TAYLOR_CONFIG = dict(alpha=0.22, damping=0.70, max_norm_ratio=0.10, adaptive_lsq=False)


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


def benchmark_opt1_ring_int8_quant(device: str = "cuda:0") -> Dict[str, Any]:
    """Benchmark Asymmetric INT8 Ring KV Quantizer on realistic Wan2.1 2D-USP shapes."""
    print("[Opt #1] Running Asymmetric INT8 Ring KV Compression benchmark...", flush=True)
    quantizer = AsymmetricRingQuantizer(torch, warmup_steps=2, tail_anchor_steps=2, enabled=True)
    results = []

    # Shape 1: 8-GPU u4r2 (H_local = 12/4 = 3, S_ring = 3120/2 = 1560, D = 128)
    # Shape 2: 8-GPU u4r2 33-frame HD (H_local = 3, S_ring = 6240/2 = 3120, D = 128)
    # Shape 3: 4-GPU u2r2 33-frame HD (H_local = 6, S_ring = 3120, D = 128)
    configs = [
        {"name": "8GPU_u4r2_17f", "h_local": 3, "s_ring": 1560, "d": 128},
        {"name": "8GPU_u4r2_33f", "h_local": 3, "s_ring": 3120, "d": 128},
        {"name": "4GPU_u2r2_33f", "h_local": 6, "s_ring": 3120, "d": 128},
    ]

    for cfg in configs:
        torch.manual_seed(101)
        h_loc, s_ring, d = cfg["h_local"], cfg["s_ring"], cfg["d"]
        q = torch.randn(1, h_loc, s_ring, d, device=device, dtype=torch.float16)
        kv = torch.randn(2, 1, h_loc, s_ring, d, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(5):
            pkt = quantizer.pack_packet(kv)
            _ = quantizer.unpack_packet(pkt, target_dtype=torch.float16)
        torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        iters = 40
        for _ in range(iters):
            pkt = quantizer.pack_packet(kv)
            kv_deq = quantizer.unpack_packet(pkt, target_dtype=torch.float16)
        torch.cuda.synchronize(device)
        codec_ms = ((time.perf_counter() - t0) * 1000.0) / iters

        orig_bytes = kv.numel() * kv.element_size()
        wire_bytes = pkt.numel() * pkt.element_size()
        comp_ratio = orig_bytes / wire_bytes

        # Measure attention fidelity (SDPA output with exact FP16 KV vs INT8-dequantized KV)
        out_ref = F.scaled_dot_product_attention(q, kv[0], kv[1])
        out_int8 = F.scaled_dot_product_attention(q, kv_deq[0], kv_deq[1])
        cos_attn = float(
            F.cosine_similarity(
                out_ref.flatten().float().unsqueeze(0),
                out_int8.flatten().float().unsqueeze(0),
            ).item()
        )
        rmse_attn = float(torch.sqrt(torch.mean((out_ref.float() - out_int8.float()) ** 2)).item())
        signal_power = float(torch.mean(out_ref.float() ** 2).item())
        sqnr_db = 10.0 * math.log10(signal_power / max(rmse_attn ** 2, 1e-12))

        assumptions = dict(cross_node_bandwidth_gib_s=1.05, cross_socket_bandwidth_gib_s=12.1,
                           codec_cost_fraction=0.25, calls_per_step=60,
                           network_transfer_measured=False,
                           model="bytes/bandwidth + codec_roundtrip_ms * codec_cost_fraction")
        eth_bw_bytes_per_ms = assumptions["cross_node_bandwidth_gib_s"] * (1024 ** 3) / 1000.0
        pcie_bw_bytes_per_ms = assumptions["cross_socket_bandwidth_gib_s"] * (1024 ** 3) / 1000.0
        calls_per_step = assumptions["calls_per_step"]
        codec_cost = codec_ms * assumptions["codec_cost_fraction"]
        fp16_eth_step_ms = calls_per_step * (orig_bytes / eth_bw_bytes_per_ms)
        int8_eth_step_ms = calls_per_step * (wire_bytes / eth_bw_bytes_per_ms + codec_cost)
        fp16_pcie_step_ms = calls_per_step * (orig_bytes / pcie_bw_bytes_per_ms)
        int8_pcie_step_ms = calls_per_step * (wire_bytes / pcie_bw_bytes_per_ms + codec_cost)

        results.append(
            {
                "config": cfg["name"],
                "kv_shape": [2, 1, h_loc, s_ring, d],
                "orig_kb": round(orig_bytes / 1024.0, 2),
                "wire_kb": round(wire_bytes / 1024.0, 2),
                "compression_ratio": round(comp_ratio, 4),
                "bandwidth_saving_pct": round(100.0 * (1.0 - wire_bytes / orig_bytes), 2),
                "codec_roundtrip_ms": round(codec_ms, 4),
                "attn_output_cos_sim": round(cos_attn, 6),
                "attn_output_rmse": round(rmse_attn, 6),
                "attn_output_sqnr_db": round(sqnr_db, 2),
                "evidence_type": "measured_microbenchmark_with_estimates",
                "timing_scope": "single-device synthetic tensor pack+unpack, mean of 40 iterations",
                "communication_evidence_type": "estimated",
                "communication_assumptions": assumptions,
                "estimated_cross_node_ring_comm_step_fp16_ms": round(fp16_eth_step_ms, 2),
                "estimated_cross_node_ring_comm_step_int8_ms": round(int8_eth_step_ms, 2),
                "estimated_cross_node_comm_speedup": round(fp16_eth_step_ms / int8_eth_step_ms, 3),
                "estimated_cross_socket_ring_comm_step_fp16_ms": round(fp16_pcie_step_ms, 2),
                "estimated_cross_socket_ring_comm_step_int8_ms": round(int8_pcie_step_ms, 2),
                "estimated_cross_socket_comm_speedup": round(fp16_pcie_step_ms / int8_pcie_step_ms, 3),
            }
        )
        print(
            f"  [Opt #1] {cfg['name']}: {orig_bytes/1024:.0f} KB -> {wire_bytes/1024:.0f} KB "
            f"({comp_ratio:.2f}x), codec={codec_ms:.3f} ms, attn_cos={cos_attn:.6f}, "
            f"SQNR={sqnr_db:.2f} dB, Estimated CrossNode Comm Speedup={fp16_eth_step_ms/int8_eth_step_ms:.2f}x (no transfer)",
            flush=True,
        )

    return {"evidence_type": "measured_microbenchmark_with_estimates",
            "timing_scope": "single-device synthetic codec; communication is estimated, no transfer",
            "configs": results, "summary": quantizer.summary()}


def benchmark_opt3_taylor_cache(wan: WanBackend, cached_wan_payload) -> List[Dict[str, Any]]:
    """Compare Zero-Order DBCache vs First-Order Taylor-Compensated DBCache across thresholds."""
    print("[Opt #3] Running First-Order Taylor-Compensated Residual Extrapolation benchmark...", flush=True)
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

    # Warmup GPU
    w_state = _fast_prepare_wan(wan, dict(base_spec, steps=2, cache=False), "wan-opt3-warm", cached_wan_payload)
    wan.advance({"id": "w", "state_id": "wan-opt3-warm", "steps": 2})
    wan.release({"id": "w", "state_id": "wan-opt3-warm"})
    torch.cuda.synchronize()

    # 1. Exact Reference (tau = 0.00)
    ref_id = "wan-opt3-ref"
    ref_state = _fast_prepare_wan(wan, dict(base_spec, cache=False, threshold=0.0), ref_id, cached_wan_payload)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wan.advance({"id": "r", "state_id": ref_id, "steps": 20})
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) * 1000.0
    ref_latents = ref_state.latents.detach().cpu().float().clone()
    ref_video = _decode_video(wan, ref_state)
    wan.release({"id": "r", "state_id": ref_id})

    results = [
        {
            "mode": "Exact-NoCache",
            "threshold": 0.0,
            "taylor_alpha": 0.0,
            "denoise_ms": round(ref_ms, 2),
            "speedup": 1.0,
            "reused_steps": 0,
            "skip_ratio_pct": 0.0,
            "latent_cos_sim": 1.0,
            "latent_rmse": 0.0,
            "video_psnr_db": 99.99,
            "video_ssim": 1.0,
        }
    ]

    for tau in (0.12, 0.16, 0.24):
        # A. Standard Zero-Order DBCache
        z_id = f"wan-zero-{int(tau*100)}"
        z_state = _fast_prepare_wan(
            wan,
            dict(base_spec, cache=True, threshold=tau, max_continuous=2),
            z_id,
            cached_wan_payload,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        z_adv = wan.advance({"id": "z", "state_id": z_id, "steps": 20})
        torch.cuda.synchronize()
        z_ms = (time.perf_counter() - t0) * 1000.0
        z_latents = z_state.latents.detach().cpu().float().clone()
        z_video = _decode_video(wan, z_state)
        wan.release({"id": "z", "state_id": z_id})

        z_reused = sum(1 for ev in z_adv["events"] if ev.get("reused_forwards", 0) > 0)
        z_cos = float(
            F.cosine_similarity(
                ref_latents.flatten().unsqueeze(0),
                z_latents.flatten().unsqueeze(0),
            ).item()
        )
        z_rmse = float(torch.sqrt(torch.mean((ref_latents - z_latents) ** 2)).item())
        z_psnr = compute_psnr(ref_video, z_video)
        z_ssim = compute_video_ssim(ref_video, z_video)

        results.append(
            {
                "mode": "ZeroOrder-DBCache",
                "threshold": tau,
                "taylor_alpha": 0.0,
                "denoise_ms": round(z_ms, 2),
                "speedup": round(ref_ms / z_ms, 3),
                "reused_steps": z_reused,
                "skip_ratio_pct": round(100.0 * z_reused / 20.0, 1),
                "latent_cos_sim": round(z_cos, 6),
                "latent_rmse": round(z_rmse, 6),
                "video_psnr_db": z_psnr,
                "video_ssim": z_ssim,
            }
        )
        print(
            f"  [Opt #3] ZeroOrder tau={tau:.2f}: {z_ms:.1f} ms ({ref_ms/z_ms:.2f}x), "
            f"skip={z_reused}/20, PSNR={z_psnr:.2f} dB, SSIM={z_ssim:.4f}",
            flush=True,
        )

        # Fixed before evaluation and shared by all thresholds, never selected by quality.
        t_id = f"wan-taylor-{int(tau*100)}"
        t_state = _fast_prepare_wan(
            wan, dict(base_spec, cache=True, threshold=tau, max_continuous=2),
            t_id, cached_wan_payload,
        )
        comp = TaylorResidualCompensator(torch, **TAYLOR_CONFIG, enabled=True)
        comp.monotonic_gate = False
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_events = [comp.step(wan.runtime, t_state) for _ in range(20)]
        torch.cuda.synchronize()
        t_ms = (time.perf_counter() - t0) * 1000.0
        t_latents = t_state.latents.detach().cpu().float().clone()
        t_video = _decode_video(wan, t_state)
        wan.release({"id": "t", "state_id": t_id})

        t_reused = sum(1 for ev in t_events if ev.get("reused_forwards", 0) > 0)
        t_cos = float(F.cosine_similarity(
            ref_latents.flatten().unsqueeze(0), t_latents.flatten().unsqueeze(0),
        ).item())
        t_rmse = float(torch.sqrt(torch.mean((ref_latents - t_latents) ** 2)).item())
        t_psnr = compute_psnr(ref_video, t_video)
        t_ssim = compute_video_ssim(ref_video, t_video)
        row = {
            "mode": "Taylor1stOrder-DBCache",
            "threshold": tau,
            "taylor_alpha": TAYLOR_CONFIG["alpha"],
            "adaptive_lsq": TAYLOR_CONFIG["adaptive_lsq"],
            "taylor_config": dict(TAYLOR_CONFIG, monotonic_gate=False),
            "parameter_selection": "fixed_before_evaluation; no test-quality selection",
            "denoise_ms": round(t_ms, 2),
            "speedup": round(ref_ms / t_ms, 3),
            "reused_steps": t_reused,
            "skip_ratio_pct": round(100.0 * t_reused / 20.0, 1),
            "latent_cos_sim": round(t_cos, 6),
            "latent_rmse": round(t_rmse, 6),
            "video_psnr_db": t_psnr,
            "video_ssim": t_ssim,
            "psnr_gain_db": round(t_psnr - z_psnr, 4),
            "ssim_gain": round(t_ssim - z_ssim, 5),
        }
        results.append(row)
        print(
            f"  [Opt #3] Taylor1stOrder tau={tau:.2f} (fixed alpha={row['taylor_alpha']}): "
            f"{t_ms:.1f} ms, reused={t_reused}/20, PSNR={t_psnr:.2f} dB, SSIM={t_ssim:.4f}",
            flush=True,
        )

    for row in results:
        row.update(evidence_type="measured_microbenchmark", timing_scope=DENOISE_SCOPE,
                   quality_status="reference" if row["threshold"] == 0 else "measured_comparison",
                   workload_spec=base_spec)
    return results


def benchmark_opt2_sd35_flash_and_microbatch(sd35: SD35Backend) -> Dict[str, Any]:
    """Benchmark SD3.5-Medium SDPA_FLASH vs SDPA_MATH and Multi-Prompt Micro-Batching."""
    print("[Opt #2] Running SD3.5-Medium SDPA_FLASH + Multi-Prompt Micro-Batching benchmark...", flush=True)
    prompts = [
        "A futuristic neon city skyline at dusk, cinematic lighting, 8k",
        "An astronaut riding a white horse on Mars, photorealistic",
        "A serene Japanese garden in autumn with red maple leaves",
        "A vintage steam locomotive crossing a snowy mountain viaduct",
    ]
    seeds = [101, 202, 303, 404]

    # Pre-stage all 4 prompts in ProactiveEncoderPool and offload T5-XXL
    for p in prompts:
        sd35._ensure_staged(p, "", 256, True)
    sd35.offload_encoders({})
    torch.cuda.empty_cache()

    # Warmup both math and flash
    sd35.run_micro_batch({"prompts": [prompts[0]], "seeds": [101], "height": 512, "width": 512, "steps": 4, "attention": "math"})
    sd35.run_micro_batch({"prompts": [prompts[0]], "seeds": [101], "height": 512, "width": 512, "steps": 4, "attention": "flash"})
    torch.cuda.synchronize()

    requests = _image_requests(prompts, seeds)
    res_math_1 = _run_image_batch(sd35, requests[:1], time.perf_counter(), attention="math", cache=False)
    img_math = res_math_1.pop("images")
    res_flash_1 = _run_image_batch(sd35, requests[:1], time.perf_counter(), attention="flash", cache=False)
    img_flash = res_flash_1.pop("images")
    flash_vs_math_psnr = compute_psnr(img_math, img_flash)
    flash_vs_math_ssim = compute_ssim_frame(img_math[0], img_flash[0])

    rows = []
    for name, batch_size, attention in (
        ("Sequential-4xB1-SDPA_MATH", 1, "math"),
        ("Sequential-4xB1-SDPA_FLASH", 1, "flash"),
        ("MicroBatch-2xB2-SDPA_FLASH", 2, "flash"),
        ("MicroBatch-1xB4-SDPA_FLASH", 4, "flash"),
    ):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        completions, batch_latencies = [], []
        for offset in range(0, len(requests), batch_size):
            batch = _run_image_batch(sd35, requests[offset:offset + batch_size], t0,
                                     attention=attention, cache=False)
            batch.pop("images")
            completions.extend(batch["requests"])
            batch_latencies.append(batch["batch_latency_ms"])
        total_ms = max(r["finish_ms"] for r in completions)
        baseline_ms = rows[0]["total_4img_ms"] if rows else total_ms
        row = dict(
            evidence_type="measured_microbenchmark", timing_scope=IMAGE_SCOPE,
            mode=name, batch_size=batch_size, attention=attention,
            requests=completions, output_count=sum(r["output_count"] for r in completions),
            total_4img_ms=total_ms, batch_latencies_ms=batch_latencies,
            amortized_ms_per_image=total_ms / len(completions),
            throughput_img_s=1000.0 * len(completions) / total_ms,
            speedup_vs_seq_math=baseline_ms / total_ms,
            peak_hbm_mb=round(torch.cuda.max_memory_allocated() / (1 << 20), 2),
        )
        rows.append(row)
        print(f"  [Opt #2] {name}: completion={total_ms:.1f} ms, "
              f"amortized={row['amortized_ms_per_image']:.1f} ms/img", flush=True)

    return {
        "evidence_type": "measured_microbenchmark",
        "timing_scope": IMAGE_SCOPE,
        "single_image_math_vs_flash": {
            "math_ms": res_math_1["batch_latency_ms"],
            "flash_ms": res_flash_1["batch_latency_ms"],
            "speedup": res_math_1["batch_latency_ms"] / res_flash_1["batch_latency_ms"],
            "math_requests": res_math_1["requests"],
            "flash_requests": res_flash_1["requests"],
            "quality_status": "measured_comparison",
            "flash_vs_math_psnr_db": flash_vs_math_psnr,
            "flash_vs_math_ssim": flash_vs_math_ssim,
        },
        "four_distinct_prompts_burst": rows,
    }


def benchmark_opt4_vae_stream_overlap(
    wan: WanBackend,
    sd35: SD35Backend,
    cached_wan_payload,
) -> Dict[str, Any]:
    """Benchmark Stream-Isolated 3D-VAE Chunked Decode overlapping with SD3.5 Denoising."""
    print("[Opt #4] Running Stream-Isolated 3D-VAE Decode + SD3.5 Denoising Overlap benchmark...", flush=True)
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
    img_prompt = "A futuristic neon city skyline at dusk, cinematic lighting, 8k"
    sd35._ensure_staged(img_prompt, "", 256, True)
    sd35.offload_encoders({})

    # 1. Baseline Sequential: Wan 3D-VAE Chunked Decode (5 latent frames) followed by SD3.5 (12 steps, Flash)
    s_seq = _fast_prepare_wan(wan, wan_spec, "wan-vae-seq", cached_wan_payload)
    wan.advance({"id": "s", "state_id": "wan-vae-seq", "steps": 4})
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    t_vae0 = time.perf_counter()
    seq_video = _decode_video(wan, s_seq)
    torch.cuda.synchronize()
    seq_vae_ms = (time.perf_counter() - t_vae0) * 1000.0

    t_sd0 = time.perf_counter()
    seq_sd_res = sd35.run_micro_batch({
        "prompts": [img_prompt],
        "seeds": [101],
        "height": 512,
        "width": 512,
        "steps": 12,
        "attention": "flash",
    })
    torch.cuda.synchronize()
    seq_sd_ms = (time.perf_counter() - t_sd0) * 1000.0
    seq_wall_ms = (time.perf_counter() - t0) * 1000.0
    seq_peak_hbm = round(torch.cuda.max_memory_allocated() / (1 << 20), 2)
    wan.release({"id": "s", "state_id": "wan-vae-seq"})

    # 2. Stream-Isolated Overlap: ConcurrentVAEDecodeStreamer on secondary CUDA stream + SD3.5 on primary stream
    s_ovl = _fast_prepare_wan(wan, wan_spec, "wan-vae-ovl", cached_wan_payload)
    wan.advance({"id": "o", "state_id": "wan-vae-ovl", "steps": 4})
    streamer = ConcurrentVAEDecodeStreamer(wan.runtime, chunk_frames=1)
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
        ovl_video = ovl_res.pop("video_np")
    finally:
        streamer.close()
        wan.release({"id": "o", "state_id": "wan-vae-ovl"})

    # Verify exact numerical equivalence between sequential decode and stream-overlapped decode
    max_abs_diff = float(np.max(np.abs(seq_video.astype(np.float32) - ovl_video.astype(np.float32))))
    overlap_saving_ms = seq_wall_ms - ovl_res["wall_ms"]
    overlap_speedup = seq_wall_ms / ovl_res["wall_ms"]

    print(
        f"  [Opt #4] Sequential (VAE={seq_vae_ms:.1f} ms + SD3.5={seq_sd_ms:.1f} ms) = {seq_wall_ms:.1f} ms | "
        f"Stream-Overlapped Wall = {ovl_res['wall_ms']:.1f} ms "
        f"(Saved {overlap_saving_ms:.1f} ms, {overlap_speedup:.2f}x speedup, "
        f"Peak HBM={ovl_res['peak_hbm_mb']:.1f} MB, max_diff={max_abs_diff:.2e})",
        flush=True,
    )

    return {
        "evidence_type": "measured_microbenchmark",
        "timing_scope": "preencoded: excludes encoding and video denoise; complete video decode + image pipeline to NumPy",
        "quality_status": "measured_video_comparison",
        "sequential_vae_ms": round(seq_vae_ms, 2),
        "sequential_sd35_ms": round(seq_sd_ms, 2),
        "sequential_wall_ms": round(seq_wall_ms, 2),
        "sequential_peak_hbm_mb": seq_peak_hbm,
        "overlapped_wall_ms": ovl_res["wall_ms"],
        "overlapped_vae_stream_ms": ovl_res["vae_decode"]["vae_stream_ms"],
        "overlapped_sd35_primary_ms": ovl_res["primary_task"]["model_ms"],
        "overlapped_peak_hbm_mb": ovl_res["peak_hbm_mb"],
        "saved_ms": round(overlap_saving_ms, 2),
        "speedup": round(overlap_speedup, 3),
        "output_max_abs_diff": max_abs_diff,
    }


def benchmark_e2e_v2_trace_replay(
    wan: WanBackend,
    sd35: SD35Backend,
    cached_wan_payload,
) -> Dict[str, Any]:
    """Preencoded component replay with manual phases, not an online scheduler."""
    print("[Component Replay] Running v2 manual mixed-modality phases...", flush=True)
    # Workload:
    #   t = 0 ms:     Wan2.1 video request (832x480, 17 frames, 20 steps, SLO = 17,000 ms)
    #   t = 1,200 ms: Burst of 2 SD3.5 image requests (img-1, img-2; arrived within micro-batch window, SLO = 4,500 ms)
    #   t = 3,500 ms: SD3.5 image request (img-3; SLO = 4,500 ms)
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
    torch.cuda.empty_cache()

    images = _image_requests(img_prompts, [101, 202, 303], [1200.0, 1350.0, 3500.0])
    video_request = dict(request_id="v0", modality="video", arrival_ms=0.0, spec=wan_spec)
    v_id = "v2-replay-wan"
    comp = TaylorResidualCompensator(torch, **TAYLOR_CONFIG, enabled=True)
    comp.monotonic_gate = False
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    v_start_ms = _wait_until_arrival(t_start, video_request["arrival_ms"])
    v_state = _fast_prepare_wan(wan, wan_spec, v_id, cached_wan_payload)
    completions, events = [], []
    try:
        for steps, members in ((3, images[:2]), (4, images[2:])):
            for _ in range(steps):
                events.append(comp.step(wan.runtime, v_state))
            batch = _run_image_batch(sd35, members, t_start, attention="flash", cache=False)
            batch.pop("images")
            completions.extend(batch["requests"])
        for _ in range(13):
            events.append(comp.step(wan.runtime, v_state))
        video = _decode_video(wan, v_state)
        torch.cuda.synchronize()
        finish_ms = (time.perf_counter() - t_start) * 1000.0
        completions.extend(_completion_records([video_request], video, v_start_ms, finish_ms))
        peak_hbm_mb = round(torch.cuda.max_memory_allocated() / (1 << 20), 2)
    finally:
        wan.release({"id": "v2", "state_id": v_id})

    res = _replay_summary(completions, 4500.0, 17000.0)
    res.update(policy="v2 component replay: Flash + MicroBatch + fixed TaylorCache + sequential decode",
               peak_hbm_mb=peak_hbm_mb, quality_status="not_evaluated_in_replay",
               taylor_config=dict(TAYLOR_CONFIG, monotonic_gate=False),
               reused_steps=sum(ev.get("reused_forwards", 0) > 0 for ev in events))
    print(f"  [Component Replay] v2: Image max (n=3)={res['image_max_latency_ms']:.1f} ms, "
          f"Video={res['video_latency_ms']:.1f} ms", flush=True)
    return res


def write_markdown_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# HeteroDiT v2 Component Microbenchmarks, Estimates & Replay",
        "",
        f"Device: `{payload['device_name']}`. GPU component evidence is not online scheduling evidence.",
        "",
        "## 1. INT8 codec: measured_microbenchmark_with_estimates",
        "",
        "Codec and attention fidelity use single-device synthetic tensors. No network transfer was measured.",
        "Communication is estimated: assumed bandwidth 1.05/12.1 GiB/s (node/socket), 60 calls/step,",
        "with only 0.25 of measured codec cost charged per call. These are assumptions, not link measurements.",
        "",
        "| Config | Compression | Codec roundtrip mean, n=40 (ms) | Attention cosine | Estimated cross-node speedup |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["opt1_ring_int8"]["configs"]:
        lines.append(f"| {r['config']} | {r['compression_ratio']:.2f}x | {r['codec_roundtrip_ms']:.3f} | "
                     f"{r['attn_output_cos_sim']:.6f} | {r['estimated_cross_node_comm_speedup']:.2f}x |")
    lines.extend([
        "", "## 2. SD3.5 image microbenchmarks: measured_microbenchmark", "",
        IMAGE_SCOPE,
        "Amortized cost is not request latency. JSON retains each batch completion latency and request timestamps.",
        "",
        "| Mode | Batch size | Total 4-image completion (ms) | Amortized ms/image | Throughput (img/s) | Speedup vs Seq-Math |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in payload["opt2_sd35_flash_microbatch"]["four_distinct_prompts_burst"]:
        lines.append(f"| {r['mode']} | {r['batch_size']} | {r['total_4img_ms']:.1f} | "
                     f"{r['amortized_ms_per_image']:.1f} | {r['throughput_img_s']:.2f} | {r['speedup_vs_seq_math']:.2f}x |")
    lines.extend([
        "", "## 3. Taylor cache: measured_microbenchmark", "", DENOISE_SCOPE,
        f"Taylor parameters fixed before evaluation across all three thresholds: {TAYLOR_CONFIG}; monotonic_gate=False.",
        "No selection by test quality. Reference/self rows are not independent quality passes.",
        "",
        "| Mode | Tau | Reused steps | Denoise (ms) | Speedup | PSNR (dB) | SSIM | Quality status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for r in payload["opt3_taylor_cache"]:
        lines.append(f"| {r['mode']} | {r['threshold']:.2f} | {r['reused_steps']}/20 | "
                     f"{r['denoise_ms']:.1f} | {r['speedup']:.2f}x | {r['video_psnr_db']:.2f} | "
                     f"{r['video_ssim']:.4f} | {r['quality_status']} |")
    ovl = payload["opt4_vae_stream_overlap"]
    lines.extend([
        "", "## 4. VAE overlap: measured_microbenchmark", "", ovl["timing_scope"],
        f"Sequential wall: {ovl['sequential_wall_ms']:.1f} ms; overlapped wall: {ovl['overlapped_wall_ms']:.1f} ms; "
        f"speedup: {ovl['speedup']:.2f}x; video output max diff: {ovl['output_max_abs_diff']:.2e}.",
        "", "## 5. Manually Ordered Component Replay: component_replay", "", REPLAY_SCOPE,
        "Known prompts preencoded before the trace; no online scheduler or stream-overlapped VAE in this replay.",
        "Image max uses n=3, not a P99 estimate. Replay quality was not evaluated.",
    ])
    replay = payload["component_replay_v2"]
    lines.append(f"{replay['policy']}: image mean {replay['image_mean_latency_ms']:.1f} ms, "
                 f"image max (n=3) {replay['image_max_latency_ms']:.1f} ms, "
                 f"video {replay['video_latency_ms']:.1f} ms, reused steps {replay['reused_steps']}/20, "
                 f"total SLO {replay['total_slo_attainment_pct']:.1f}%.")
    (out_dir / "V2_OPTIMIZATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("RUN_DIR", "/mnt/e/000_code/zyd-2026/ditserve-hetero/results/v2-opt")),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"=== Running HeteroDiT v2 Optimizations on {device_name} ===", flush=True)

    # 1. Opt #1: Ring INT8 Quantization
    opt1_res = benchmark_opt1_ring_int8_quant(device=device)

    # 2. Load Wan2.1 and SD3.5 on GPU 0
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
        "request_id": "preencode-wan-v2",
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

    # 3. Opt #3: Taylor Residual Extrapolation vs Zero-Order DBCache
    opt3_res = benchmark_opt3_taylor_cache(wan, cached_wan_payload)

    # 4. Load SD3.5-Medium alongside Wan2.1
    sd35 = SD35Backend({
        "device": device,
        "model_path": SD35_PATH,
        "revision": SD35_REVISION,
        "dtype": "bfloat16",
        "attention": "flash",
        "options": {
            "offload_text_encoders_after_warmup": False,
            "encoder_pool": {"max_entries": 64, "max_bytes": 2 * (1024 ** 3)},
        },
    })

    # 5. Opt #2: SD3.5 SDPA_FLASH + True Multi-Prompt Heterogeneous Micro-Batching
    opt2_res = benchmark_opt2_sd35_flash_and_microbatch(sd35)

    # 6. Opt #4: Stream-Isolated 3D-VAE Chunked Decode Overlapping with SD3.5 Denoising
    opt4_res = benchmark_opt4_vae_stream_overlap(wan, sd35, cached_wan_payload)

    # 7. E2E Mixed-Modality Trace Replay (HeteroDiT-v2)
    e2e_res = benchmark_e2e_v2_trace_replay(wan, sd35, cached_wan_payload)

    payload = {
        "device_name": device_name,
        "opt1_ring_int8": opt1_res,
        "opt2_sd35_flash_microbatch": opt2_res,
        "opt3_taylor_cache": opt3_res,
        "opt4_vae_stream_overlap": opt4_res,
        "component_replay_v2": e2e_res,
    }
    out_json = args.out_dir / "V2_OPTIMIZATIONS.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(args.out_dir, payload)

    sd35.close()
    wan.close()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"=== Completed HeteroDiT v2 Optimizations -> {out_json} ===", flush=True)


if __name__ == "__main__":
    main()
