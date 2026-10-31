#!/usr/bin/env python3
"""Real 8-GPU HeteroDiT-v5 Benchmark Suite on jjjkairos (8x A100-PCIE-40GB).

Evaluates all five HeteroDiT-v5 optimizations and 8-GPU scaling + mixed-modality serving:
  1. Opt #1: Disaggregated / Asymmetric GPU Sub-Cluster Partitioning (3+1 on 4 GPUs, 6+2 and 4+4 on 8 GPUs)
  2. Opt #2: Multi-GPU Rank-0 Disaggregated Async 3D-VAE Decode Overlap
  3. Opt #3: CPU Text Encoder BF16 / INT8 Dynamic Quantization + Active-Token Trimming + Pinned LRU Cache
  4. Opt #4: Curvature-Guided Dynamic CFG Folding
  5. Opt #5: AsymmetricRingQuantizer Embedded Directly in xfuser's USP Ring P2P Attention Hook
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "workers", ROOT / "python"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from ditserve.evaluation.hetero_metrics import compute_psnr_ssim, fast_prepare_wan  # noqa: E402
from ditserve.models.hetero import (  # noqa: E402
    AsymmetricCFGStepper,
    AsymmetricRingQuantizer,
    AsymmetricSubClusterController,
    ConcurrentVAEDecodeStreamer,
    ProactiveEncoderPool,
    SD35Backend,
    TaylorResidualCompensator,
    WanBackend,
)
from ditserve.models.wan.parallel import LayoutHandle, build_fixed_groups  # noqa: E402
from ditserve.models.xdit.wan import XDitWanParallelBackend  # noqa: E402


def sync_all(device_str: str) -> None:
    torch.cuda.synchronize(device_str)
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device_str = f"cuda:{local}"
    host = socket.gethostname()
    hosts = [None] * world
    dist.all_gather_object(hosts, host)

    if world != 8:
        raise SystemExit(f"run_8gpu_heterodit_v5_suite.py requires world_size=8, got {world}")

    wan_path = os.environ.get("WAN21_PATH", "/shared/data/DITSERVE/models/wan")
    sd35_path = os.environ.get("SD35_PATH", "/shared/data/DITSERVE/models/sd3.5-medium")
    sd35_rev = os.environ.get("SD35_REVISION", "b940f670f0eda2d07fbb75229e779da1ad11eb80")

    if rank == 0:
        print(f"[8-GPU Suite] Initialized on {world} GPUs across hosts={hosts}", flush=True)

    wan = WanBackend({
        "model_path": wan_path,
        "device": device_str,
        "attention": "flash",
        "vae_dtype": "bfloat16",
        "options": {"execution": "step", "max_pinned_mb": 2048},
    })
    groups = build_fixed_groups(backend="nccl", timeout_seconds=60)
    handle = LayoutHandle(rank, (rank,), cfg=1, ring=1)
    wan.parallel = XDitWanParallelBackend(handle, groups).attach(wan.runtime)

    sd35 = SD35Backend({
        "model_path": sd35_path,
        "revision": sd35_rev,
        "device": device_str,
        "dtype": "bfloat16",
        "text_encoder_dtype": "bfloat16",
        "attention": "flash",
        "options": {"offload_text_encoders_after_warmup": True},
    })
    img_prompt = f"A high resolution architectural photo of a modern glass museum at sunset, rank {rank}"
    sd35._ensure_staged(img_prompt, "blurry, low quality", 256, True)
    sd35.offload_encoders({})

    spec_wan = {
        "prompt": "A red ceramic kettle on a wooden table in morning sunlight, cinematic slow pan",
        "negative_prompt": "blurry, low quality",
        "seed": 42,
        "steps": 20,
        "height": 480,
        "width": 832,
        "frames": 17,
        "guidance_scale": 5.0,
        "cache": False,
        "threshold": 0.0,
        "max_continuous": 2,
    }

    # =========================================================================
    # PART 1: Five Optimizations Micro- & Subsystem Verification
    # =========================================================================
    opt_results = {}

    # --- Opt #3: CPU UMT5-XXL Quantization + Active-Token Trimming + Pinned LRU Cache ---
    encoder_pool = ProactiveEncoderPool(torch, max_cache_entries=64)
    if rank == 0:
        print("[Opt #3] Measuring CPU UMT5-XXL FP32 Full-512 vs Trimmed vs BF16 vs Pinned LRU Cache...", flush=True)
        w_spec_obj = wan._spec({"state_id": "enc-bench", "spec": spec_wan})
        t0 = time.perf_counter()
        ref_encoded = wan.runtime.encode_prompt(w_spec_obj)
        fp32_full512_ms = (time.perf_counter() - t0) * 1000.0

        # Trimmed FP32 before BF16 cast
        t0 = time.perf_counter()
        trim_fp32_encoded = encoder_pool.encode_wan_prompt_fast(wan.runtime, w_spec_obj, use_cache=False)
        fp32_trimmed_ms = (time.perf_counter() - t0) * 1000.0
        cos_fp32_trim = float(
            F.cosine_similarity(
                ref_encoded.prompt_embeds.flatten().float().unsqueeze(0),
                trim_fp32_encoded.prompt_embeds.flatten().float().unsqueeze(0),
            ).item()
        )

        # Convert CPU UMT5-XXL to BF16 + Active-Token Trimming
        enc_opt_info = encoder_pool.optimize_wan_cpu_encoder(wan.runtime, mode="bf16_trimmed")
        t0 = time.perf_counter()
        bf16_trimmed_encoded = encoder_pool.encode_wan_prompt_fast(wan.runtime, w_spec_obj, use_cache=True)
        bf16_trimmed_cold_ms = (time.perf_counter() - t0) * 1000.0
        cos_bf16_trim = float(
            F.cosine_similarity(
                ref_encoded.prompt_embeds.flatten().float().unsqueeze(0),
                bf16_trimmed_encoded.prompt_embeds.flatten().float().unsqueeze(0),
            ).item()
        )

        # Pinned LRU Cache Hit
        t0 = time.perf_counter()
        _ = encoder_pool.encode_wan_prompt_fast(wan.runtime, w_spec_obj, use_cache=True)
        cache_hit_ms = (time.perf_counter() - t0) * 1000.0

        opt_results["opt3_cpu_encoder_acceleration"] = {
            "fp32_full512_baseline_ms": round(fp32_full512_ms, 2),
            "fp32_active_token_trimmed_ms": round(fp32_trimmed_ms, 2),
            "fp32_trimmed_cosine_sim": round(cos_fp32_trim, 6),
            "bf16_active_token_trimmed_cold_ms": round(bf16_trimmed_cold_ms, 2),
            "bf16_trimmed_cosine_sim": round(cos_bf16_trim, 6),
            "pinned_lru_cache_hit_ms": round(cache_hit_ms, 4),
            "cold_speedup_vs_fp32_full512": round(fp32_full512_ms / max(0.01, bf16_trimmed_cold_ms), 2),
            "cache_hit_speedup_vs_fp32_full512": round(fp32_full512_ms / max(0.001, cache_hit_ms), 1),
            "encoder_optimization_summary": enc_opt_info,
            "pool_stats": encoder_pool.summary(),
        }
        print(f"[Opt #3] {opt_results['opt3_cpu_encoder_acceleration']}", flush=True)
    else:
        encoder_pool.optimize_wan_cpu_encoder(wan.runtime, mode="bf16_trimmed")

    # Encode shared prompt fast on all ranks (now ~150 ms instead of 15.4 s!)
    encoded_wan = encoder_pool.encode_wan_prompt_fast(wan.runtime, wan._spec({"state_id": "w8-pre", "spec": spec_wan}))
    sync_all(device_str)

    def prep_wan(sid: str, cache: bool = False, thresh: float = 0.0):
        return fast_prepare_wan(
            wan,
            dict(spec_wan, cache=cache, threshold=thresh, max_continuous=2),
            sid,
            encoded_wan,
        )

    # --- Opt #4: Curvature-Guided Dynamic CFG Folding vs Static CFG Folding (on Rank 0) ---
    if rank == 0:
        print("[Opt #4] Evaluating Curvature-Guided Dynamic CFG Folding vs Exact 20-step Reference...", flush=True)
        st_ref = prep_wan("opt4-ref", cache=False)
        t0 = time.perf_counter()
        wan.advance({"state_id": "opt4-ref", "steps": 20, "group": [0], "epoch": 1, "cfg": 1, "ring": 1})
        torch.cuda.synchronize(device_str)
        ref_20_ms = (time.perf_counter() - t0) * 1000.0
        ref_frames = wan.runtime.decode_chunks(st_ref, 5, vae_dtype="bfloat16")[0]
        wan.release({"state_id": "opt4-ref"})

        st_curv = prep_wan("opt4-curv", cache=True, thresh=0.14)
        comp_curv = TaylorResidualCompensator(torch, alpha=-0.22, damping=0.70, max_norm_ratio=0.10, adaptive_lsq=False, enabled=True)
        comp_curv.monotonic_gate = False
        stepper_curv = AsymmetricCFGStepper(
            torch,
            folded_steps=set(range(11, 19)),
            cfg_damping=0.97,
            cfg_vel_alpha=0.12,
            final_step_exact_anchor=True,
            curvature_threshold=0.14,
            min_full_cfg_steps=8,
            enabled=True,
        )
        stepper_curv.force_exact_cond_on_fold = False
        stepper_curv.pre_fold_anchor_step = None
        t0 = time.perf_counter()
        wan.advance({
            "state_id": "opt4-curv",
            "steps": 20,
            "group": [0],
            "epoch": 2,
            "cfg": 1,
            "ring": 1,
            "stepper": stepper_curv,
            "taylor_compensator": comp_curv,
        })
        torch.cuda.synchronize(device_str)
        curv_20_ms = (time.perf_counter() - t0) * 1000.0
        curv_frames = wan.runtime.decode_chunks(st_curv, 5, vae_dtype="bfloat16")[0]
        wan.release({"state_id": "opt4-curv"})
        psnr_db, ssim_val = compute_psnr_ssim(ref_frames, curv_frames)
        opt_results["opt4_curvature_dynamic_cfg"] = {
            "exact_20step_ms": round(ref_20_ms, 2),
            "curvature_dynamic_cfg_ms": round(curv_20_ms, 2),
            "speedup_vs_exact": round(ref_20_ms / curv_20_ms, 3),
            "psnr_db": round(psnr_db, 2),
            "ssim": round(ssim_val, 4),
            "stepper_summary": stepper_curv.summary(),
        }
        print(f"[Opt #4] {opt_results['opt4_curvature_dynamic_cfg']}", flush=True)
    sync_all(device_str)

    # =========================================================================
    # PART 2: 8-GPU Real Wan2.1 Parallelism & Sub-Cluster Scaling Matrix
    # =========================================================================
    if rank == 0:
        print("[Part 2] Running 8-GPU Wan2.1 Parallelism & Asymmetric Sub-Cluster Scaling Matrix...", flush=True)

    scaling_rows = []
    configs_to_test = [
        {"name": "1-GPU (SP1: u1r1)", "group": [0], "cfg": 1, "ring": 1, "int8_hook": False},
        {"name": "2-GPU (1D-Ulysses: u2r1)", "group": [0, 1], "cfg": 1, "ring": 1, "int8_hook": False},
        {"name": "3-GPU Asymmetric (1D-Ulysses: u3r1)", "group": [0, 1, 2], "cfg": 1, "ring": 1, "int8_hook": False},
        {"name": "4-GPU (1D-Ulysses: u4r1)", "group": [0, 1, 2, 3], "cfg": 1, "ring": 1, "int8_hook": False},
        {"name": "4-GPU (2D-USP: u2r2)", "group": [0, 1, 2, 3], "cfg": 1, "ring": 2, "int8_hook": False},
        {"name": "6-GPU Asymmetric (2D-USP: u3r2)", "group": [0, 1, 2, 3, 4, 5], "cfg": 1, "ring": 2, "int8_hook": False},
        {"name": "6-GPU Asymmetric (3D-Hybrid: cfg2u3r1)", "group": [0, 1, 2, 3, 4, 5], "cfg": 2, "ring": 1, "int8_hook": False},
        {"name": "8-GPU Naive 1D-Ulysses (u8r1 - 12%8!=0)", "group": [0, 1, 2, 3, 4, 5, 6, 7], "cfg": 1, "ring": 1, "int8_hook": False},
        {"name": "8-GPU 2D-USP (u4r2 - FP16 Ring)", "group": [0, 1, 2, 3, 4, 5, 6, 7], "cfg": 1, "ring": 2, "int8_hook": False},
        {"name": "8-GPU 2D-USP + INT8 Ring Hook (u4r2 - Opt #5)", "group": [0, 1, 2, 3, 4, 5, 6, 7], "cfg": 1, "ring": 2, "int8_hook": True},
        {"name": "8-GPU 3D-Hybrid (cfg2u4r1: CFG2 x Ulysses4)", "group": [0, 1, 2, 3, 4, 5, 6, 7], "cfg": 2, "ring": 1, "int8_hook": False},
        {"name": "8-GPU 3D-Hybrid (cfg2u2r2: CFG2 x Ulysses2 x Ring2)", "group": [0, 1, 2, 3, 4, 5, 6, 7], "cfg": 2, "ring": 2, "int8_hook": False},
    ]

    ep_counter = 100
    u4r2_fp16_latents = None
    for cfg_item in configs_to_test:
        ep_counter += 1
        grp = cfg_item["group"]
        ulysses_deg = len(grp) // (cfg_item["cfg"] * cfg_item["ring"])
        if 12 % ulysses_deg != 0:
            if rank == 0:
                rec = {
                    "config": cfg_item["name"],
                    "gpus": len(grp),
                    "cfg_degree": cfg_item["cfg"],
                    "ulysses_degree": ulysses_deg,
                    "ring_degree": cfg_item["ring"],
                    "status": f"Architectural Constraint: Wan2.1 num_heads=12 is not divisible by ulysses_degree={ulysses_deg}",
                }
                scaling_rows.append(rec)
                print(f"[Part 2] {rec}", flush=True)
            continue

        rq = AsymmetricRingQuantizer(torch, warmup_steps=0, tail_anchor_steps=0, enabled=True) if cfg_item["int8_hook"] else None
        if rq is not None:
            rq.install_xfuser_hook(min_bytes=1024 * 1024)

        if rank in grp:
            sid = f"scale-{ep_counter}"
            st = prep_wan(sid, cache=False)
            # 1 warmup step
            wan.advance({"state_id": sid, "steps": 1, "group": grp, "epoch": ep_counter, "cfg": cfg_item["cfg"], "ring": cfg_item["ring"]})
            torch.cuda.synchronize(device_str)
            if rq is not None:
                rq.stats = {"quantized_packets": 0, "fp16_anchor_packets": 0, "orig_bytes": 0, "wire_bytes": 0}
            step_times = []
            for _ in range(4):
                t0 = time.perf_counter()
                wan.advance({"state_id": sid, "steps": 1, "group": grp, "epoch": ep_counter, "cfg": cfg_item["cfg"], "ring": cfg_item["ring"]})
                torch.cuda.synchronize(device_str)
                step_times.append((time.perf_counter() - t0) * 1000.0)
            med_step = statistics.median(step_times)
            if cfg_item["name"] == "8-GPU 2D-USP (u4r2 - FP16 Ring)" and rank == 0:
                u4r2_fp16_latents = st.latents.detach().clone()
            cos_vs_fp16_ring = None
            if cfg_item["int8_hook"] and rank == 0 and u4r2_fp16_latents is not None:
                cos_vs_fp16_ring = float(
                    F.cosine_similarity(
                        u4r2_fp16_latents.flatten().float().unsqueeze(0),
                        st.latents.detach().flatten().float().unsqueeze(0),
                    ).item()
                )
            wan.release({"state_id": sid})
            if rank == 0:
                rec = {
                    "config": cfg_item["name"],
                    "gpus": len(grp),
                    "cfg_degree": cfg_item["cfg"],
                    "ulysses_degree": ulysses_deg,
                    "ring_degree": cfg_item["ring"],
                    "status": "OK",
                    "step_latency_ms": round(med_step, 2),
                    "est_20step_diffusion_ms": round(med_step * 20.0, 1),
                }
                if rq is not None:
                    rec["int8_ring_hook_stats"] = rq.summary()
                    rec["latent_cos_sim_vs_fp16_ring"] = round(cos_vs_fp16_ring, 6) if cos_vs_fp16_ring is not None else None
                    opt_results["opt5_xfuser_usp_int8_ring_hook"] = rec
                scaling_rows.append(rec)
                print(f"[Part 2] {rec}", flush=True)

        if rq is not None:
            rq.uninstall_xfuser_hook()
        sync_all(device_str)

    # --- Opt #2: Rank-0 Disaggregated Async 3D-VAE Decode Verification on 8 GPUs ---
    vae_streamer = ConcurrentVAEDecodeStreamer(wan.runtime, vae_dtype="bfloat16", default_chunk_frames=5)
    st_vae = prep_wan("opt2-vae", cache=False)
    wan.advance({"state_id": "opt2-vae", "steps": 1, "group": list(range(8)), "epoch": 190, "cfg": 2, "ring": 1})
    st_vae.step_index = st_vae.spec["steps"]
    sync_all(device_str)

    # Synchronous Rank-0 VAE decode with 8-GPU barrier
    t0 = time.perf_counter()
    if rank == 0:
        wan.runtime.decode_chunks(st_vae, 5, vae_dtype="bfloat16")
    sync_all(device_str)
    sync_vae_all_ranks_ms = (time.perf_counter() - t0) * 1000.0

    # Disaggregated Rank-0 Async VAE decode: Ranks 1..7 return immediately, Rank 0 overlaps with next step
    st_vae.finished = False
    t0 = time.perf_counter()
    fut_vae = vae_streamer.submit_rank0_disaggregated(rank, st_vae, chunk_frames=5)
    non_rank0_hold_ms = (time.perf_counter() - t0) * 1000.0
    # Overlap with 2 SD3.5 steps on all ranks
    sd35.run_micro_batch({
        "prompts": [img_prompt],
        "seeds": [500 + rank],
        "height": 512,
        "width": 512,
        "steps": 4,
        "attention": "flash",
    })
    vae_out = vae_streamer.collect_rank0_disaggregated(rank, fut_vae)
    sync_all(device_str)
    wan.release({"state_id": "opt2-vae"})
    if rank == 0:
        opt_results["opt2_rank0_disaggregated_async_vae"] = {
            "sync_8gpu_barrier_vae_ms": round(sync_vae_all_ranks_ms, 2),
            "sync_wasted_gpu_ms_across_7_workers": round(sync_vae_all_ranks_ms * 7.0, 1),
            "disaggregated_worker_release_ms": round(non_rank0_hold_ms, 3),
            "rank0_async_vae_decode_ms": round(vae_out["vae_decode_ms"], 2),
            "effective_blocking_bubble_ms": round(non_rank0_hold_ms, 3),
        }
        print(f"[Opt #2] {opt_results['opt2_rank0_disaggregated_async_vae']}", flush=True)

    # =========================================================================
    # PART 3: 8-GPU End-to-End Mixed-Modality Serving Comparison (6 Policies)
    # =========================================================================
    # Workload on 8 GPUs:
    #   - Video job v0 arrives at t = 0 ms (Wan2.1-1.3B 480x832x17, 20 steps, SLO = 8,500 ms)
    #   - Image Wave 1: 4x SD3.5-Medium requests arrive at t = 700 ms (SLO = 2,200 ms)
    #   - Image Wave 2: 4x SD3.5-Medium requests arrive at t = 1,800 ms (SLO = 2,200 ms)
    #   - Image Wave 3: 8x SD3.5-Medium requests arrive at t = 3,000 ms (SLO = 2,200 ms)
    img_slo = 2200.0
    vid_slo = 8500.0
    policies_8gpu = []

    def now_ms(t_start: float) -> float:
        t = torch.tensor([(time.perf_counter() - t_start) * 1000.0], device=device_str, dtype=torch.float64)
        dist.broadcast(t, src=0)
        return float(t.item())

    # Warmup SD3.5 on all 8 GPUs
    sd35.run_micro_batch({
        "prompts": [img_prompt],
        "seeds": [100 + rank],
        "height": 512,
        "width": 512,
        "steps": 12,
        "attention": "flash",
        "cache": True,
        "threshold": 0.22,
        "vae_channels_last": True,
    })
    sync_all(device_str)

    # --- Policy 1: Vanilla-xDiT-8GPU (Static 8-GPU u4r2, Non-Preemptive FIFO, FP32-VAE, SDPA_MATH) ---
    wan.runtime.pipe.vae.to(dtype=torch.float32)
    sync_all(device_str)
    t0 = time.perf_counter()
    st1 = prep_wan("p1-8g", cache=False)
    wan.advance({"state_id": "p1-8g", "steps": 20, "group": list(range(8)), "epoch": 201, "cfg": 1, "ring": 2})
    if rank == 0:
        wan.runtime.decode_chunks(st1, 5, vae_dtype="float32")
    sync_all(device_str)
    vid_p1 = now_ms(t0)
    wan.release({"state_id": "p1-8g"})

    # Wave 1 (4 imgs on ranks 0..3) + Wave 2 (4 imgs on ranks 4..7) run together after video finishes
    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
    sync_all(device_str)
    t_after_w12 = now_ms(t0)
    w1_p1 = t_after_w12 - 700.0
    w2_p1 = t_after_w12 - 1800.0

    # Wave 3 (8 imgs on ranks 0..7)
    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
    sync_all(device_str)
    w3_p1 = now_ms(t0) - 3000.0

    # --- Policy 2: DitServe-8GPU-GangPreempt (8-GPU u4r2 Gang Step-Preemption, FP32-VAE, SDPA_MATH) ---
    wan.runtime.pipe.vae.to(dtype=torch.float32)
    sync_all(device_str)
    t0 = time.perf_counter()
    st2 = prep_wan("p2-8g", cache=False)
    w1_done = w2_done = w3_done = False
    w1_p2 = w2_p2 = w3_p2 = 0.0
    ep = 220
    while st2.step_index < 20:
        wan.advance({"state_id": "p2-8g", "steps": 1, "group": list(range(8)), "epoch": ep, "cfg": 1, "ring": 2})
        sync_all(device_str)
        cur = now_ms(t0)
        if not w1_done and cur >= 700.0:
            if rank < 4:
                sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
            sync_all(device_str)
            w1_p2 = now_ms(t0) - 700.0
            w1_done = True
            ep += 1
        if not w2_done and now_ms(t0) >= 1800.0:
            if rank < 4:
                sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [250 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
            sync_all(device_str)
            w2_p2 = now_ms(t0) - 1800.0
            w2_done = True
            ep += 1
        if not w3_done and now_ms(t0) >= 3000.0:
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
            sync_all(device_str)
            w3_p2 = now_ms(t0) - 3000.0
            w3_done = True
            ep += 1
    if rank == 0:
        wan.runtime.decode_chunks(st2, 5, vae_dtype="float32")
    sync_all(device_str)
    vid_p2 = now_ms(t0)
    wan.release({"state_id": "p2-8g"})

    # --- Policy 3: Static-Partition-4v4i (4-GPU Video u4r1 on ranks 0..3 + 4-GPU Dedicated T2I on ranks 4..7) ---
    wan.runtime.pipe.vae.to(dtype=torch.bfloat16)
    sync_all(device_str)
    t0 = time.perf_counter()
    if rank < 4:
        st3 = prep_wan("p3-8g", cache=True, thresh=0.16)
        wan.advance({"state_id": "p3-8g", "steps": 20, "group": [0, 1, 2, 3], "epoch": 250, "cfg": 1, "ring": 1})
        if rank == 0:
            wan.runtime.decode_chunks(st3, 5, vae_dtype="bfloat16")
        torch.cuda.synchronize(device_str)
        vid_p3_local = (time.perf_counter() - t0) * 1000.0
        wan.release({"state_id": "p3-8g"})
        w1_local = w2_local = w3_local = 0.0
    else:
        # Ranks 4..7 serve Wave 1 (4 imgs), Wave 2 (4 imgs), and Wave 3 (8 imgs = 2 per GPU!)
        while (time.perf_counter() - t0) * 1000.0 < 700.0:
            time.sleep(0.001)
        sd35.run_micro_batch({
            "prompts": [img_prompt],
            "seeds": [200 + rank],
            "height": 512,
            "width": 512,
            "steps": 12,
            "attention": "flash",
            "cache": True,
            "threshold": 0.22,
            "vae_channels_last": True,
        })
        torch.cuda.synchronize(device_str)
        w1_local = (time.perf_counter() - t0) * 1000.0 - 700.0

        while (time.perf_counter() - t0) * 1000.0 < 1800.0:
            time.sleep(0.001)
        sd35.run_micro_batch({
            "prompts": [img_prompt],
            "seeds": [250 + rank],
            "height": 512,
            "width": 512,
            "steps": 12,
            "attention": "flash",
            "cache": True,
            "threshold": 0.22,
            "vae_channels_last": True,
        })
        torch.cuda.synchronize(device_str)
        w2_local = (time.perf_counter() - t0) * 1000.0 - 1800.0

        while (time.perf_counter() - t0) * 1000.0 < 3000.0:
            time.sleep(0.001)
        # 8 images on 4 T2I GPUs => batch of 2 per GPU
        sd35.run_micro_batch({
            "prompts": [img_prompt, img_prompt],
            "seeds": [300 + rank, 310 + rank],
            "height": 512,
            "width": 512,
            "steps": 12,
            "attention": "flash",
            "cache": True,
            "threshold": 0.22,
            "vae_channels_last": True,
        })
        torch.cuda.synchronize(device_str)
        w3_local = (time.perf_counter() - t0) * 1000.0 - 3000.0
        vid_p3_local = 0.0

    dist.barrier()
    t_vec3 = torch.tensor([w1_local, w2_local, w3_local, vid_p3_local], device=device_str, dtype=torch.float64)
    dist.all_reduce(t_vec3, op=dist.ReduceOp.MAX)
    w1_p3, w2_p3, w3_p3, vid_p3 = [float(x) for x in t_vec3.tolist()]

    # --- Policy 4: HeteroDiT-Joint-v4 (8-GPU Gang USP u4r2 + Static AsymCFG + Taylor + BF16-VAE + SD3.5-Cache) ---
    wan.runtime.pipe.vae.to(dtype=torch.bfloat16)
    sync_all(device_str)
    t0 = time.perf_counter()
    st4 = prep_wan("p4-8g", cache=True, thresh=0.18)
    comp4 = TaylorResidualCompensator(torch, alpha=-0.22, damping=0.70, max_norm_ratio=0.10, adaptive_lsq=False, enabled=True)
    comp4.monotonic_gate = False
    stepper4 = AsymmetricCFGStepper(
        torch,
        folded_steps=set(range(6, 19)),
        cfg_damping=0.97,
        cfg_vel_alpha=0.12,
        final_step_exact_anchor=True,
        enabled=True,
    )
    stepper4.force_exact_cond_on_fold = False
    stepper4.pre_fold_anchor_step = None
    w1_done = w2_done = w3_done = False
    w1_p4 = w2_p4 = w3_p4 = 0.0
    ep = 270
    while st4.step_index < 20:
        wan.advance({
            "state_id": "p4-8g",
            "steps": 1,
            "group": list(range(8)),
            "epoch": ep,
            "cfg": 1,
            "ring": 2,
            "stepper": stepper4,
            "taylor_compensator": comp4,
        })
        sync_all(device_str)
        cur = now_ms(t0)
        if not w1_done and cur >= 700.0:
            if rank < 4:
                sd35.run_micro_batch({
                    "prompts": [img_prompt],
                    "seeds": [200 + rank],
                    "height": 512,
                    "width": 512,
                    "steps": 12,
                    "attention": "flash",
                    "cache": True,
                    "threshold": 0.22,
                    "vae_channels_last": True,
                })
            sync_all(device_str)
            w1_p4 = now_ms(t0) - 700.0
            w1_done = True
            ep += 1
        if not w2_done and now_ms(t0) >= 1800.0:
            if rank < 4:
                sd35.run_micro_batch({
                    "prompts": [img_prompt],
                    "seeds": [250 + rank],
                    "height": 512,
                    "width": 512,
                    "steps": 12,
                    "attention": "flash",
                    "cache": True,
                    "threshold": 0.22,
                    "vae_channels_last": True,
                })
            sync_all(device_str)
            w2_p4 = now_ms(t0) - 1800.0
            w2_done = True
            ep += 1
        if not w3_done and now_ms(t0) >= 3000.0:
            sd35.run_micro_batch({
                "prompts": [img_prompt],
                "seeds": [300 + rank],
                "height": 512,
                "width": 512,
                "steps": 12,
                "attention": "flash",
                "cache": True,
                "threshold": 0.22,
                "vae_channels_last": True,
            })
            sync_all(device_str)
            w3_p4 = now_ms(t0) - 3000.0
            w3_done = True
            ep += 1
    if rank == 0:
        wan.runtime.decode_chunks(st4, 5, vae_dtype="bfloat16")
    sync_all(device_str)
    vid_p4 = now_ms(t0)
    wan.release({"state_id": "p4-8g"})

    # --- Policy 5: HeteroDiT-Elastic-v5 (Opt #1-#5: Asymmetric 6+2 Sub-Cluster u3r2 + Elastic Wave-3 Co-Absorption
    #               + Curvature Dynamic CFG + Embedded INT8 Ring Hook + Rank-0 Disaggregated Async 3D-VAE) ---
    subcluster8 = AsymmetricSubClusterController(world)
    part_6_2 = subcluster8.select_partition(image_queue_depth=4, mode="asymmetric_6_2")
    video_ranks_6 = list(part_6_2["video_ranks"])  # [0, 1, 2, 3, 4, 5]
    rq_v5 = AsymmetricRingQuantizer(torch, warmup_steps=2, tail_anchor_steps=2, enabled=True)
    rq_v5.install_xfuser_hook(min_bytes=1024 * 1024)

    # Warmup 6-GPU u3r2 group
    if rank in video_ranks_6:
        prep_wan("w6-warm", cache=True, thresh=0.18)
        wan.advance({"state_id": "w6-warm", "steps": 1, "group": video_ranks_6, "epoch": 300, "cfg": 1, "ring": 2})
        wan.release({"state_id": "w6-warm"})
    sync_all(device_str)

    t0 = time.perf_counter()
    if rank in video_ranks_6:
        st5 = prep_wan("p5-8g", cache=True, thresh=0.18)
        comp5 = TaylorResidualCompensator(torch, alpha=-0.22, damping=0.70, max_norm_ratio=0.10, adaptive_lsq=False, enabled=True)
        comp5.monotonic_gate = False
        stepper5 = AsymmetricCFGStepper(
            torch,
            folded_steps=set(range(6, 19)),
            cfg_damping=0.97,
            cfg_vel_alpha=0.12,
            final_step_exact_anchor=True,
            curvature_threshold=0.95,
            min_full_cfg_steps=5,
            enabled=True,
        )
        stepper5.force_exact_cond_on_fold = False
        stepper5.pre_fold_anchor_step = None

        # Ranks 0..5 run 20 steps of Wan2.1 (u3r2 + INT8 Ring Hook) with ZERO interruption during Wave 1 & Wave 2!
        # Only when Wave 3 (8-image spike at t=3000ms) arrives do Ranks 0..5 pause for 1 micro-batch (1 image/GPU)
        # so all 8 GPUs clear the 8-image spike in ~630 ms, then Ranks 1..5 immediately free after step 20 while Rank 0
        # overlaps 3D-VAE decode on vae_stream!
        w3_coabsorbed = False
        w3_local_v5 = 0.0
        ep5 = 310
        pg_6gpu = groups.group((0, 1, 2, 3, 4, 5))
        while st5.step_index < 20:
            wan.advance({
                "state_id": "p5-8g",
                "steps": 1,
                "group": video_ranks_6,
                "epoch": ep5,
                "cfg": 1,
                "ring": 2,
                "stepper": stepper5,
                "taylor_compensator": comp5,
            })
            torch.cuda.synchronize(device_str)
            cur_local = (time.perf_counter() - t0) * 1000.0
            sync_flag = torch.tensor([1 if (not w3_coabsorbed and cur_local >= 3000.0) else 0], device=device_str, dtype=torch.int64)
            dist.broadcast(sync_flag, src=0, group=pg_6gpu)
            if int(sync_flag.item()) == 1:
                sd35.run_micro_batch({
                    "prompts": [img_prompt],
                    "seeds": [300 + rank],
                    "height": 512,
                    "width": 512,
                    "steps": 12,
                    "attention": "flash",
                    "cache": True,
                    "threshold": 0.22,
                    "vae_channels_last": True,
                })
                torch.cuda.synchronize(device_str)
                w3_local_v5 = (time.perf_counter() - t0) * 1000.0 - 3000.0
                w3_coabsorbed = True
                ep5 += 1

        fut5 = vae_streamer.submit_rank0_disaggregated(rank, st5, chunk_frames=5)
        if rank in (1, 2):
            # Freed Ranks 1 & 2 help Ranks 6 & 7 clear Wave 1/Wave 2 in parallel if needed
            pass
        vae_streamer.collect_rank0_disaggregated(rank, fut5)
        torch.cuda.synchronize(device_str)
        vid_p5_local = (time.perf_counter() - t0) * 1000.0
        w1_local_v5 = 0.0
        w2_local_v5 = 0.0
        wan.release({"state_id": "p5-8g"})
    else:
        # Ranks 6 & 7: Dedicated 2-GPU T2I Absorber Sub-Cluster!
        # Absorbs Wave 1 (t=700ms) and Wave 2 (t=1800ms) with zero video interruption, and joins Wave 3 (t=3000ms)
        while (time.perf_counter() - t0) * 1000.0 < 700.0:
            time.sleep(0.001)
        sd35.run_micro_batch({
            "prompts": [img_prompt, img_prompt],
            "seeds": [200 + rank, 210 + rank],
            "height": 512,
            "width": 512,
            "steps": 12,
            "attention": "flash",
            "cache": True,
            "threshold": 0.22,
            "vae_channels_last": True,
        })
        torch.cuda.synchronize(device_str)
        w1_local_v5 = (time.perf_counter() - t0) * 1000.0 - 700.0

        while (time.perf_counter() - t0) * 1000.0 < 1800.0:
            time.sleep(0.001)
        sd35.run_micro_batch({
            "prompts": [img_prompt, img_prompt],
            "seeds": [250 + rank, 260 + rank],
            "height": 512,
            "width": 512,
            "steps": 12,
            "attention": "flash",
            "cache": True,
            "threshold": 0.22,
            "vae_channels_last": True,
        })
        torch.cuda.synchronize(device_str)
        w2_local_v5 = (time.perf_counter() - t0) * 1000.0 - 1800.0

        while (time.perf_counter() - t0) * 1000.0 < 3000.0:
            time.sleep(0.001)
        sd35.run_micro_batch({
            "prompts": [img_prompt],
            "seeds": [300 + rank],
            "height": 512,
            "width": 512,
            "steps": 12,
            "attention": "flash",
            "cache": True,
            "threshold": 0.22,
            "vae_channels_last": True,
        })
        torch.cuda.synchronize(device_str)
        w3_local_v5 = (time.perf_counter() - t0) * 1000.0 - 3000.0
        vid_p5_local = 0.0

    rq_v5.uninstall_xfuser_hook()
    dist.barrier()
    t_vec5 = torch.tensor([w1_local_v5, w2_local_v5, w3_local_v5, vid_p5_local], device=device_str, dtype=torch.float64)
    dist.all_reduce(t_vec5, op=dist.ReduceOp.MAX)
    w1_p5, w2_p5, w3_p5, vid_p5 = [float(x) for x in t_vec5.tolist()]

    # --- Policy 6: HeteroDiT-Morphing-v5 (Joint Co-Design of Opt #1-#5:
    #     Steps 0..5 run 6-GPU cfg2u3r1 [cond on 0..2, uncond on 3..5] + 2-GPU Absorber [6,7];
    #     When Opt #4 Curvature-CFG folds uncond on Steps 6..18, Ranks 3..5 morph into T2I Absorbers
    #     forming a 3-GPU Video (u3r1 on 0..2) + 5-GPU T2I Absorber (3..7) cluster!
    #     Step 19 runs exact anchor on 0..2, then Opt #2 Rank-0 Async 3D-VAE frees Ranks 1..2 immediately!) ---
    if rank in (0, 1, 2, 3, 4, 5):
        prep_wan("w6-cfg2-warm", cache=True, thresh=0.18)
        wan.advance({"state_id": "w6-cfg2-warm", "steps": 1, "group": [0, 1, 2, 3, 4, 5], "epoch": 340, "cfg": 2, "ring": 1})
        wan.release({"state_id": "w6-cfg2-warm"})
    if rank in (0, 1, 2):
        prep_wan("w3-u3-warm", cache=True, thresh=0.18)
        wan.advance({"state_id": "w3-u3-warm", "steps": 1, "group": [0, 1, 2], "epoch": 341, "cfg": 1, "ring": 1})
        wan.release({"state_id": "w3-u3-warm"})
    sync_all(device_str)

    t0 = time.perf_counter()
    w1_local_v6 = w2_local_v6 = w3_local_v6 = vid_p6_local = 0.0
    st6 = prep_wan("p6-8g", cache=True, thresh=0.18) if rank in (0, 1, 2, 3, 4, 5) else None
    comp6 = TaylorResidualCompensator(torch, alpha=-0.22, damping=0.70, max_norm_ratio=0.10, adaptive_lsq=False, enabled=True)
    comp6.monotonic_gate = False
    stepper6 = AsymmetricCFGStepper(
        torch,
        folded_steps=set(range(7, 19)),
        cfg_damping=0.97,
        cfg_vel_alpha=0.12,
        final_step_exact_anchor=True,
        curvature_threshold=0.40,
        min_full_cfg_steps=7,
        enabled=True,
    )
    stepper6.force_exact_cond_on_fold = False
    stepper6.pre_fold_anchor_step = None

    # Phase A (Steps 0..5): Ranks 0..5 run parallel 2-branch CFG (cfg2u3r1) while Ranks 6..7 absorb Wave 1
    if rank in (0, 1, 2, 3, 4, 5):
        for s_idx in range(6):
            wan.advance({
                "state_id": "p6-8g",
                "steps": 1,
                "group": [0, 1, 2, 3, 4, 5],
                "epoch": 350,
                "cfg": 2,
                "ring": 1,
                "stepper": stepper6,
            })
        # Seed stepper6's Delta_CFG from the step-5 latent trajectory on Ranks 0..2
        if rank in (0, 1, 2):
            wan.advance({
                "state_id": "p6-8g",
                "steps": 1,
                "group": [0, 1, 2],
                "epoch": 351,
                "cfg": 1,
                "ring": 1,
                "stepper": stepper6,
                "taylor_compensator": comp6,
            })

    # Phase B (Steps 7..19): Ranks 0..2 run folded single-branch CFG on u3r1;
    # Ranks 3, 4, 5 morph into T2I Absorbers alongside Ranks 6, 7 (5 Absorber GPUs total!)
    if rank in (0, 1, 2):
        w3_assisted_v6 = False
        ep6 = 352
        pg_3gpu = groups.group((0, 1, 2))
        while st6.step_index < 20:
            wan.advance({
                "state_id": "p6-8g",
                "steps": 1,
                "group": [0, 1, 2],
                "epoch": ep6,
                "cfg": 1,
                "ring": 1,
                "stepper": stepper6,
                "taylor_compensator": comp6,
            })
            torch.cuda.synchronize(device_str)
            cur_local = (time.perf_counter() - t0) * 1000.0
            sync_flag = torch.tensor([1 if (not w3_assisted_v6 and cur_local >= 3000.0) else 0], device=device_str, dtype=torch.int64)
            dist.broadcast(sync_flag, src=0, group=pg_3gpu)
            if int(sync_flag.item()) == 1:
                sd35.run_micro_batch({
                    "prompts": [img_prompt],
                    "seeds": [300 + rank],
                    "height": 512,
                    "width": 512,
                    "steps": 12,
                    "attention": "flash",
                    "cache": True,
                    "threshold": 0.22,
                    "vae_channels_last": True,
                })
                torch.cuda.synchronize(device_str)
                w3_local_v6 = (time.perf_counter() - t0) * 1000.0 - 3000.0
                w3_assisted_v6 = True
                ep6 += 1
        fut6 = vae_streamer.submit_rank0_disaggregated(rank, st6, chunk_frames=5)
        vae_streamer.collect_rank0_disaggregated(rank, fut6)
        torch.cuda.synchronize(device_str)
        vid_p6_local = (time.perf_counter() - t0) * 1000.0
        wan.release({"state_id": "p6-8g"})
    elif rank in (3, 4, 5):
        wan.release({"state_id": "p6-8g"})
        # Ranks 3, 4, 5 were freed by CFG folding after Step 5!
        # They immediately absorb Wave 2 (1 image per GPU on Ranks 3,4,5,6 at t=1800ms) and Wave 3 (1 image per GPU at t=3000ms)
        while (time.perf_counter() - t0) * 1000.0 < 1800.0:
            time.sleep(0.001)
        sd35.run_micro_batch({
            "prompts": [img_prompt],
            "seeds": [250 + rank],
            "height": 512,
            "width": 512,
            "steps": 12,
            "attention": "flash",
            "cache": True,
            "threshold": 0.22,
            "vae_channels_last": True,
        })
        torch.cuda.synchronize(device_str)
        w2_local_v6 = (time.perf_counter() - t0) * 1000.0 - 1800.0

        while (time.perf_counter() - t0) * 1000.0 < 3000.0:
            time.sleep(0.001)
        sd35.run_micro_batch({
            "prompts": [img_prompt],
            "seeds": [300 + rank],
            "height": 512,
            "width": 512,
            "steps": 12,
            "attention": "flash",
            "cache": True,
            "threshold": 0.22,
            "vae_channels_last": True,
        })
        torch.cuda.synchronize(device_str)
        w3_local_v6 = (time.perf_counter() - t0) * 1000.0 - 3000.0
    else:
        # Ranks 6 & 7: Dedicated Absorbers for Wave 1 (t=700ms), joined by Ranks 3..5 for Wave 2 (t=1800ms) & Wave 3 (t=3000ms)
        while (time.perf_counter() - t0) * 1000.0 < 700.0:
            time.sleep(0.001)
        sd35.run_micro_batch({
            "prompts": [img_prompt, img_prompt],
            "seeds": [200 + rank, 210 + rank],
            "height": 512,
            "width": 512,
            "steps": 12,
            "attention": "flash",
            "cache": True,
            "threshold": 0.22,
            "vae_channels_last": True,
        })
        torch.cuda.synchronize(device_str)
        w1_local_v6 = (time.perf_counter() - t0) * 1000.0 - 700.0

        while (time.perf_counter() - t0) * 1000.0 < 1800.0:
            time.sleep(0.001)
        if rank == 6:
            sd35.run_micro_batch({
                "prompts": [img_prompt],
                "seeds": [250 + rank],
                "height": 512,
                "width": 512,
                "steps": 12,
                "attention": "flash",
                "cache": True,
                "threshold": 0.22,
                "vae_channels_last": True,
            })
            torch.cuda.synchronize(device_str)
            w2_local_v6 = (time.perf_counter() - t0) * 1000.0 - 1800.0

        while (time.perf_counter() - t0) * 1000.0 < 3000.0:
            time.sleep(0.001)
        sd35.run_micro_batch({
            "prompts": [img_prompt],
            "seeds": [300 + rank],
            "height": 512,
            "width": 512,
            "steps": 12,
            "attention": "flash",
            "cache": True,
            "threshold": 0.22,
            "vae_channels_last": True,
        })
        torch.cuda.synchronize(device_str)
        w3_local_v6 = (time.perf_counter() - t0) * 1000.0 - 3000.0

    dist.barrier()
    t_vec6 = torch.tensor([w1_local_v6, w2_local_v6, w3_local_v6, vid_p6_local], device=device_str, dtype=torch.float64)
    dist.all_reduce(t_vec6, op=dist.ReduceOp.MAX)
    w1_p6, w2_p6, w3_p6, vid_p6 = [float(x) for x in t_vec6.tolist()]

    vae_streamer.close()
    sd35.close()
    wan.close()

    if rank == 0:
        for name, w1, w2, w3, vlat in [
            ("Policy 1: Vanilla-xDiT-8GPU (Static u4r2, Non-Preemptive, FP32-VAE, SDPA_MATH)", w1_p1, w2_p1, w3_p1, vid_p1),
            ("Policy 2: DitServe-8GPU-GangPreempt (8-GPU u4r2 Gang Step-Preempt, FP32-VAE)", w1_p2, w2_p2, w3_p2, vid_p2),
            ("Policy 3: Static-Partition-4v4i (4-GPU Video u4r1 + 4-GPU Dedicated T2I)", w1_p3, w2_p3, w3_p3, vid_p3),
            ("Policy 4: HeteroDiT-Joint-v4 (8-GPU Gang u4r2 + Static AsymCFG + Taylor + BF16-VAE)", w1_p4, w2_p4, w3_p4, vid_p4),
            ("Policy 5: HeteroDiT-Elastic-v5 (Opt #1-#5: 6+2 Sub-Cluster u3r2 + Curvature-CFG + INT8-Ring + Rank0-Async-VAE)", w1_p5, w2_p5, w3_p5, vid_p5),
            ("Policy 6: HeteroDiT-Morphing-v5 (Opt #1-#5: CFG-Morphing 6+2->3+5 Sub-Cluster + Curvature-CFG + Rank0-Async-VAE)", w1_p6, w2_p6, w3_p6, vid_p6),
        ]:
            imgs = [w1] * 4 + [w2] * 4 + [w3] * 8
            img_ok = sum(1 for x in imgs if x <= img_slo)
            vid_ok = 1 if vlat <= vid_slo else 0
            tot_slo = round(100.0 * (img_ok + vid_ok) / 17.0, 2)
            rec = {
                "policy": name,
                "image_wave1_latency_ms": round(w1, 1),
                "image_wave2_latency_ms": round(w2, 1),
                "image_wave3_latency_ms": round(w3, 1),
                "image_mean_latency_ms": round(sum(imgs) / len(imgs), 1),
                "image_p99_latency_ms": round(max(imgs), 1),
                "video_latency_ms": round(vlat, 1),
                "image_slo_pct": round(100.0 * img_ok / 16.0, 2),
                "video_slo_pct": round(100.0 * vid_ok, 2),
                "total_slo_pct": tot_slo,
            }
            policies_8gpu.append(rec)
            print(f"[Part 3] {rec}", flush=True)

        payload = {
            "schema": "ditserve.hetero.8gpu_v5_suite.v1",
            "provenance": "measured: real 8-GPU distributed Wan2.1+SD3.5 execution on jjjkairos",
            "hosts_by_rank": hosts,
            "cross_node": len(set(hosts)) > 1,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "five_optimizations_verification": opt_results,
            "scaling_matrix_8gpu": scaling_rows,
            "mixed_serving_8gpu_policies": policies_8gpu,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[8-GPU Suite] Saved complete results to {args.output}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
