#!/usr/bin/env python3
"""2-Node 16-GPU (2x8 A100-PCIE-40GB) Scaling, Baselines & HeteroDiT-v5 Comparison on jjjkairos.

Evaluates on real 2x8 = 16 A100 GPUs across 2 nodes:
  - Part 1: 16-GPU Multi-Node Parallelism & Scaling Matrix (1, 3, 4, 6, 8, 12, 16 GPUs),
            comparing Intra-Node vs Cross-Node topologies, 2D-USP (u3r2, u4r2, u6r2, u8r2, u4r4)
            with FP16 Ring vs Opt #5 Embedded INT8 Ring Hook, and 3D-Hybrid (cfg2u3r1, cfg2u4r1,
            cfg2u6r1, cfg2u4r2, cfg2u8r1).
  - Part 2: 10-System 2-Node 16-GPU Mixed-Modality Serving Comparison under 2x Cluster Load:
            2 Concurrent Video Requests (Video 0 at t=0ms, Video 1 at t=200ms) +
            32 SD3.5-Medium Image Requests across 3 Bursty Waves:
              Wave 1: 8 Images at t=700ms   (SLO = 1,500ms)
              Wave 2: 8 Images at t=1,800ms (SLO = 1,500ms)
              Wave 3: 16 Images at t=3,000ms (SLO = 1,500ms)
            Demonstrating Ours-2 (HeteroDiT-Morphing-v5 12+4 -> 6+10 Dual-Node Sub-Cluster Morphing).
"""

from __future__ import annotations

import argparse
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

from ditserve.evaluation.hetero_metrics import fast_prepare_wan  # noqa: E402
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


def now_ms(t_start: float, device_str: str) -> float:
    t = torch.tensor([(time.perf_counter() - t_start) * 1000.0], device=device_str, dtype=torch.float64)
    dist.broadcast(t, src=0)
    return float(t.item())


def collect_max(device_str: str, *vals: float) -> list[float]:
    vec = torch.tensor(list(vals), device=device_str, dtype=torch.float64)
    dist.all_reduce(vec, op=dist.ReduceOp.MAX)
    return [float(x) for x in vec.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-model", choices=["wan22", "wan21"], default="wan22")
    parser.add_argument("--part3-preload-json", default="")
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

    if world != 16:
        raise SystemExit(f"run_16gpu_2x8_comparison.py requires world_size=16 (2x8), got {world}")

    if args.wan_model == "wan22":
        wan_path = os.environ.get("WAN22_PATH", "/shared/data/DITSERVE/models/wan2.2-ti2v-5b")
        wan_rev = os.environ.get("WAN22_REVISION", "b8fff7315c768468a5333511427288870b2e9635")
        vid_strict_slo_ms = 9000.0
        vid_relaxed_slo_ms = 12000.0
    else:
        wan_path = os.environ.get("WAN21_PATH", "/shared/data/DITSERVE/models/wan")
        wan_rev = None
        vid_strict_slo_ms = 9000.0
        vid_relaxed_slo_ms = 12000.0

    sd35_path = os.environ.get("SD35_PATH", "/shared/data/DITSERVE/models/sd3.5-medium")
    sd35_rev = os.environ.get("SD35_REVISION", "b940f670f0eda2d07fbb75229e779da1ad11eb80")

    if rank == 0:
        print(
            f"[16-GPU 2x8 Comparison ({args.wan_model})] Initialized on {world} GPUs across hosts={hosts}",
            flush=True,
        )

    wan_cfg = {
        "model_path": wan_path,
        "device": device_str,
        "attention": "flash",
        "vae_dtype": "bfloat16",
        "options": {"execution": "step", "max_pinned_mb": 2048},
    }
    if wan_rev:
        wan_cfg["revision"] = wan_rev
    wan = WanBackend(wan_cfg)
    dist.barrier()
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
    sync_all(device_str)

    num_heads = int(wan.runtime.pipe.transformer.config.num_attention_heads)
    wan_spec = {
        "prompt": "A cinematic aerial tracking shot over snow-covered mountain peaks at golden hour, volumetric clouds",
        "negative_prompt": "blurry, low quality, distorted, watermark",
        "height": 480,
        "width": 832,
        "frames": 17,
        "steps": 20,
        "guidance_scale": 5.0,
        "seed": 42,
    }
    encoder_pool = ProactiveEncoderPool(torch, max_cache_entries=64)
    norm_spec = wan._spec({"state_id": "init-enc", "spec": wan_spec})
    cached_wan = encoder_pool.encode_wan_prompt_fast(wan.runtime, norm_spec)

    def prep_wan(sid: str, cache: bool = False, thresh: float = 0.0):
        s_cfg = dict(wan_spec, cache=cache, threshold=thresh, max_continuous=2)
        return fast_prepare_wan(wan, s_cfg, sid, cached_wan)

    # Warmup key process groups across Node 0 (0..7), Node 1 (8..15), 12-GPU (0..11), and 16-GPU (0..15)
    warmup_specs = [
        (10, [0], 1, 1),
        (11, [8], 1, 1),
        (12, [0, 1, 2], 1, 1),
        (13, [8, 9, 10], 1, 1),
        (14, [3, 4, 5], 1, 1),
        (15, [11, 12, 13], 1, 1),
        (16, [0, 1, 2, 3], 1, 1),
        (17, [8, 9, 10, 11], 1, 1),
        (18, [0, 1, 2, 3, 4, 5], 1, 2),
        (19, [8, 9, 10, 11, 12, 13], 1, 2),
        (20, [0, 1, 2, 3, 4, 5], 2, 1),
        (21, [8, 9, 10, 11, 12, 13], 2, 1),
        (22, list(range(8)), 1, 2),
        (23, list(range(8, 16)), 1, 2),
        (24, list(range(12)), 1, 2),
        (25, list(range(12)), 2, 1),
        (26, list(range(16)), 2, 2),
        (27, list(range(16)), 1, 4),
    ]
    if num_heads % 8 == 0:
        warmup_specs.extend([
            (28, list(range(8)), 1, 1),
            (29, list(range(16)), 1, 2),
            (30, list(range(16)), 2, 1),
        ])

    for ep_w, grp_w, c_w, r_w in warmup_specs:
        if rank in grp_w:
            prep_wan(f"warm-{ep_w}", cache=False)
            wan.advance({"state_id": f"warm-{ep_w}", "steps": 1, "group": grp_w, "epoch": ep_w, "cfg": c_w, "ring": r_w})
            wan.release({"state_id": f"warm-{ep_w}"})
    sd35.run_micro_batch({
        "prompts": [img_prompt],
        "seeds": [900 + rank],
        "height": 512,
        "width": 512,
        "steps": 12,
        "attention": "flash",
        "cache": True,
        "threshold": 0.22,
        "vae_channels_last": True,
    })
    sync_all(device_str)

    if args.part3_preload_json and Path(args.part3_preload_json).exists():
        pre_obj = json.loads(Path(args.part3_preload_json).read_text(encoding="utf-8"))
        scaling_rows = pre_obj["scaling_matrix_16gpu"]
        policies_16gpu = pre_obj["mixed_serving_16gpu_12_policies"]
    else:
        # =========================================================================
        # PART 1: 2-Node 16-GPU (2x8) Parallelism & Scaling Matrix
        # =========================================================================
        scaling_configs = [
            {"name": "1-GPU Intra-Node (SP1: u1r1)", "group": [0], "cfg": 1, "ring": 1, "int8_hook": False},
            {"name": "3-GPU Intra-Node Asymmetric (1D-Ulysses: u3r1)", "group": [0, 1, 2], "cfg": 1, "ring": 1, "int8_hook": False},
            {"name": "4-GPU Intra-Node (1D-Ulysses: u4r1)", "group": [0, 1, 2, 3], "cfg": 1, "ring": 1, "int8_hook": False},
            {"name": "6-GPU Intra-Node Asymmetric (2D-USP: u3r2 - FP16 Ring)", "group": [0, 1, 2, 3, 4, 5], "cfg": 1, "ring": 2, "int8_hook": False},
            {"name": "6-GPU Intra-Node Asymmetric (2D-USP: u3r2 + Opt #5 INT8 Ring)", "group": [0, 1, 2, 3, 4, 5], "cfg": 1, "ring": 2, "int8_hook": True},
            {"name": "6-GPU Intra-Node Asymmetric (3D-Hybrid: cfg2u3r1)", "group": [0, 1, 2, 3, 4, 5], "cfg": 2, "ring": 1, "int8_hook": False},
            {"name": "8-GPU Intra-Node (2D-USP: u4r2 - FP16 Ring)", "group": list(range(8)), "cfg": 1, "ring": 2, "int8_hook": False},
            {"name": "8-GPU Intra-Node (2D-USP: u4r2 + Opt #5 INT8 Ring)", "group": list(range(8)), "cfg": 1, "ring": 2, "int8_hook": True},
            {"name": "8-GPU Intra-Node (3D-Hybrid: cfg2u4r1)", "group": list(range(8)), "cfg": 2, "ring": 1, "int8_hook": False},
            {"name": "12-GPU Cross-Node Asymmetric (2D-USP: u6r2 - FP16 Ring)", "group": list(range(12)), "cfg": 1, "ring": 2, "int8_hook": False},
            {"name": "12-GPU Cross-Node Asymmetric (2D-USP: u6r2 + Opt #5 INT8 Ring)", "group": list(range(12)), "cfg": 1, "ring": 2, "int8_hook": True},
            {"name": "12-GPU Cross-Node Asymmetric (3D-Hybrid: cfg2u6r1)", "group": list(range(12)), "cfg": 2, "ring": 1, "int8_hook": False},
            {"name": "16-GPU Cross-Node (2D-USP: u4r4 - FP16 Ring)", "group": list(range(16)), "cfg": 1, "ring": 4, "int8_hook": False},
            {"name": "16-GPU Cross-Node (2D-USP: u4r4 + Opt #5 INT8 Ring)", "group": list(range(16)), "cfg": 1, "ring": 4, "int8_hook": True},
            {"name": "16-GPU Cross-Node (3D-Hybrid: cfg2u4r2 - FP16 Ring)", "group": list(range(16)), "cfg": 2, "ring": 2, "int8_hook": False},
            {"name": "16-GPU Cross-Node (3D-Hybrid: cfg2u4r2 + Opt #5 INT8 Ring)", "group": list(range(16)), "cfg": 2, "ring": 2, "int8_hook": True},
        ]
        if num_heads % 8 == 0:
            scaling_configs.extend([
                {"name": "8-GPU Intra-Node (1D-Ulysses: u8r1 - 24 Heads)", "group": list(range(8)), "cfg": 1, "ring": 1, "int8_hook": False},
                {"name": "16-GPU Cross-Node (2D-USP: u8r2 - FP16 Ring across 2 Nodes)", "group": list(range(16)), "cfg": 1, "ring": 2, "int8_hook": False},
                {"name": "16-GPU Cross-Node (2D-USP: u8r2 + Opt #5 INT8 Ring across 2 Nodes)", "group": list(range(16)), "cfg": 1, "ring": 2, "int8_hook": True},
                {"name": "16-GPU Cross-Node (3D-Hybrid: cfg2u8r1 - Inter-Node CFG2 x Intra-Node Ulysses8)", "group": list(range(16)), "cfg": 2, "ring": 1, "int8_hook": False},
            ])

        scaling_rows = []
        ep_counter = 50
        sp1_step_ms = None
        for cfg_item in scaling_configs:
            ep_counter += 1
            grp = cfg_item["group"]
            ulysses_deg = len(grp) // (cfg_item["cfg"] * cfg_item["ring"])
            if num_heads % ulysses_deg != 0:
                continue
            rq = AsymmetricRingQuantizer(torch, warmup_steps=0, tail_anchor_steps=0, enabled=True) if cfg_item["int8_hook"] else None
            if rq is not None:
                rq.install_xfuser_hook(min_bytes=256 * 1024)

            if rank in grp:
                sid = f"s16-{ep_counter}"
                st = prep_wan(sid, cache=False)
                wan.advance({"state_id": sid, "steps": 1, "group": grp, "epoch": ep_counter, "cfg": cfg_item["cfg"], "ring": cfg_item["ring"]})
                torch.cuda.synchronize(device_str)
                if rq is not None:
                    rq.stats = {"quantized_packets": 0, "fp16_anchor_packets": 0, "orig_bytes": 0, "wire_bytes": 0}
                step_times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    wan.advance({"state_id": sid, "steps": 1, "group": grp, "epoch": ep_counter, "cfg": cfg_item["cfg"], "ring": cfg_item["ring"]})
                    torch.cuda.synchronize(device_str)
                    step_times.append((time.perf_counter() - t0) * 1000.0)
                med_step = statistics.median(step_times)
                if cfg_item["name"].startswith("1-GPU") and rank == 0:
                    sp1_step_ms = med_step
                wan.release({"state_id": sid})
                if rank == 0:
                    rec = {
                        "config": cfg_item["name"],
                        "gpus": len(grp),
                        "cfg_degree": cfg_item["cfg"],
                        "ulysses_degree": ulysses_deg,
                        "ring_degree": cfg_item["ring"],
                        "step_latency_ms": round(med_step, 2),
                        "est_20step_diffusion_ms": round(med_step * 20.0, 1),
                        "speedup_vs_sp1": round((sp1_step_ms or med_step) / med_step, 2),
                    }
                    if rq is not None:
                        rec["int8_ring_hook_stats"] = rq.summary()
                    scaling_rows.append(rec)
                    print(f"[Part 1: 16-GPU Scaling] {rec}", flush=True)

            if rq is not None:
                rq.uninstall_xfuser_hook()
            sync_all(device_str)

        # =========================================================================
        # PART 2: 10-System 2-Node 16-GPU (2x8) Mixed-Modality Serving Comparison
        # =========================================================================
        # 2x Cluster Workload on 16 GPUs:
        #   - 2 Video Requests: Video 0 at t = 0 ms, Video 1 at t = 200 ms (20 steps, 480x832x17)
        #   - 32 Image Requests (SD3.5-Medium 512x512, 12 steps, Image SLO = 1,500 ms):
        #       Wave 1: 8 Images at t = 700 ms
        #       Wave 2: 8 Images at t = 1,800 ms
        #       Wave 3: 16 Images at t = 3,000 ms
        v0_arr, v1_arr = 0.0, 200.0
        w1_arr, w2_arr, w3_arr = 700.0, 1800.0, 3000.0
        img_slo_ms = 1500.0
        policies_16gpu = []
        vae_streamer = ConcurrentVAEDecodeStreamer(wan.runtime, vae_dtype="bfloat16", default_chunk_frames=5)

        def record_16gpu_policy(
            name: str, w1_lat: float, w2_lat: float, w3_lat: float, v0_lat: float, v1_lat: float, notes: str = ""
        ) -> dict:
            all_imgs = [w1_lat] * 8 + [w2_lat] * 8 + [w3_lat] * 16
            all_vids = [v0_lat, v1_lat]
            img_ok = sum(1 for x in all_imgs if x <= img_slo_ms)
            vid_ok = sum(1 for v in all_vids if v <= vid_strict_slo_ms)
            vid_rel_ok = sum(1 for v in all_vids if v <= vid_relaxed_slo_ms)
            rec = {
                "policy": name,
                "image_wave1_8imgs_latency_ms": round(w1_lat, 1),
                "image_wave2_8imgs_latency_ms": round(w2_lat, 1),
                "image_wave3_16imgs_latency_ms": round(w3_lat, 1),
                "image_mean_latency_ms": round(statistics.mean(all_imgs), 1),
                "image_p99_latency_ms": round(max(all_imgs), 1),
                "video0_latency_ms": round(v0_lat, 1),
                "video1_latency_ms": round(v1_lat, 1),
                "video_mean_latency_ms": round(statistics.mean(all_vids), 1),
                "image_slo_pct": round(100.0 * img_ok / len(all_imgs), 2),
                "video_slo_pct": round(100.0 * vid_ok / len(all_vids), 2),
                "video_slo_12s_pct": round(100.0 * vid_rel_ok / len(all_vids), 2),
                "total_slo_pct": round(100.0 * (img_ok + vid_ok) / (len(all_imgs) + len(all_vids)), 2),
                "total_slo_12s_pct": round(100.0 * (img_ok + vid_rel_ok) / (len(all_imgs) + len(all_vids)), 2),
            }
            if notes:
                rec["notes"] = notes
            if rank == 0:
                print(f"[Part 2: 16-GPU Policy] {rec}", flush=True)
            policies_16gpu.append(rec)
            return rec

        gang16_ring = 2 if (num_heads % 8 == 0) else 4

        # --- 1. B1: Vanilla-xDiT-16GPU (Static 16-GPU Cross-Node Gang FCFS, FP32-VAE, SDPA_MATH) ---
        wan.runtime.pipe.vae.to(dtype=torch.float32)
        sync_all(device_str)
        t0 = time.perf_counter()
        st_b1_0 = prep_wan("b16-v0", cache=False)
        wan.advance({"state_id": "b16-v0", "steps": 20, "group": list(range(16)), "epoch": 101, "cfg": 1, "ring": gang16_ring})
        if rank == 0:
            wan.runtime.decode_chunks(st_b1_0, 5, vae_dtype="float32")
        sync_all(device_str)
        v0_b1 = now_ms(t0, device_str) - v0_arr
        wan.release({"state_id": "b16-v0"})

        st_b1_1 = prep_wan("b16-v1", cache=False)
        wan.advance({"state_id": "b16-v1", "steps": 20, "group": list(range(16)), "epoch": 102, "cfg": 1, "ring": gang16_ring})
        if rank == 0:
            wan.runtime.decode_chunks(st_b1_1, 5, vae_dtype="float32")
        sync_all(device_str)
        v1_b1 = now_ms(t0, device_str) - v1_arr
        wan.release({"state_id": "b16-v1"})

        # Wave 1 (8 imgs on 0..7) + Wave 2 (8 imgs on 8..15) run after both videos finish
        sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [100 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
        sync_all(device_str)
        t_w12 = now_ms(t0, device_str)
        w1_b1, w2_b1 = t_w12 - w1_arr, t_w12 - w2_arr
        # Wave 3 (16 imgs across 0..15)
        sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
        sync_all(device_str)
        w3_b1 = now_ms(t0, device_str) - w3_arr
        wan.runtime.pipe.vae.to(dtype=torch.bfloat16)
        record_16gpu_policy(
            f"B1: Vanilla-xDiT-16GPU (Static 16-GPU Cross-Node u{16 // gang16_ring}r{gang16_ring} FCFS, FP32-VAE, SDPA_MATH)",
            w1_b1, w2_b1, w3_b1, v0_b1, v1_b1,
            notes="16-GPU cross-node FCFS runs Video 0 then Video 1 over Ethernet Ring before serving any of the 32 images.",
        )

        # --- 2. B2: Static-SP1-EDF (16x Independent SP1 Workers, Non-Preemptive EDF, FP32-VAE) ---
        sync_all(device_str)
        t0 = time.perf_counter()
        w1_b2 = w2_b2 = w3_b2 = v0_b2 = v1_b2 = 0.0
        if rank == 0:
            wan.runtime.pipe.vae.to(dtype=torch.float32)
            st_b2_0 = prep_wan("b2-v0", cache=False)
            wan.advance({"state_id": "b2-v0", "steps": 20, "group": [0], "epoch": 110, "cfg": 1, "ring": 1})
            wan.runtime.decode_chunks(st_b2_0, 5, vae_dtype="float32")
            torch.cuda.synchronize(device_str)
            v0_b2 = (time.perf_counter() - t0) * 1000.0 - v0_arr
            wan.release({"state_id": "b2-v0"})
            wan.runtime.pipe.vae.to(dtype=torch.bfloat16)
        elif rank == 8:
            while (time.perf_counter() - t0) * 1000.0 < v1_arr:
                time.sleep(0.002)
            wan.runtime.pipe.vae.to(dtype=torch.float32)
            st_b2_1 = prep_wan("b2-v1", cache=False)
            wan.advance({"state_id": "b2-v1", "steps": 20, "group": [8], "epoch": 111, "cfg": 1, "ring": 1})
            wan.runtime.decode_chunks(st_b2_1, 5, vae_dtype="float32")
            torch.cuda.synchronize(device_str)
            v1_b2 = (time.perf_counter() - t0) * 1000.0 - v1_arr
            wan.release({"state_id": "b2-v1"})
            wan.runtime.pipe.vae.to(dtype=torch.bfloat16)
        else:
            while (time.perf_counter() - t0) * 1000.0 < w1_arr:
                time.sleep(0.002)
            if rank in (1, 2, 3, 4, 9, 10, 11, 12):
                sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [100 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
                torch.cuda.synchronize(device_str)
                w1_b2 = (time.perf_counter() - t0) * 1000.0 - w1_arr
            while (time.perf_counter() - t0) * 1000.0 < w2_arr:
                time.sleep(0.002)
            if rank in (1, 2, 3, 4, 9, 10, 11, 12):
                sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
                torch.cuda.synchronize(device_str)
                w2_b2 = (time.perf_counter() - t0) * 1000.0 - w2_arr
            while (time.perf_counter() - t0) * 1000.0 < w3_arr:
                time.sleep(0.002)
            # 16 images across 14 free GPUs: Ranks 1 and 9 must run 2 images sequentially
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
            if rank in (1, 9):
                sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [350 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
            torch.cuda.synchronize(device_str)
            w3_b2 = (time.perf_counter() - t0) * 1000.0 - w3_arr
        sync_all(device_str)
        record_16gpu_policy(
            "B2: Static-SP1-EDF (16x Independent SP1 Workers, Non-Preemptive EDF, FP32-VAE)",
            *collect_max(device_str, w1_b2, w2_b2, w3_b2, v0_b2, v1_b2),
            notes="Ranks 0 and 8 run unpreemptible SP1 videos; 16-image Wave 3 overflows the 14 remaining GPUs.",
        )

        # --- 3. B3-Vanilla: Vanilla-Static-Partition-8v8i (Node 0: 8-GPU Video u4r2 Exact + Node 1: 8-GPU T2I Exact, FP32-VAE) ---
        sync_all(device_str)
        t0 = time.perf_counter()
        w1_b3v = w2_b3v = w3_b3v = v0_b3v = v1_b3v = 0.0
        if rank < 8:
            wan.runtime.pipe.vae.to(dtype=torch.float32)
            st_b3v0 = prep_wan("b3v-v0", cache=False)
            wan.advance({"state_id": "b3v-v0", "steps": 20, "group": list(range(8)), "epoch": 120, "cfg": 1, "ring": 2})
            if rank == 0:
                wan.runtime.decode_chunks(st_b3v0, 5, vae_dtype="float32")
            torch.cuda.synchronize(device_str)
            v0_b3v = (time.perf_counter() - t0) * 1000.0 - v0_arr
            wan.release({"state_id": "b3v-v0"})

            st_b3v1 = prep_wan("b3v-v1", cache=False)
            wan.advance({"state_id": "b3v-v1", "steps": 20, "group": list(range(8)), "epoch": 121, "cfg": 1, "ring": 2})
            if rank == 0:
                wan.runtime.decode_chunks(st_b3v1, 5, vae_dtype="float32")
            torch.cuda.synchronize(device_str)
            v1_b3v = (time.perf_counter() - t0) * 1000.0 - v1_arr
            wan.release({"state_id": "b3v-v1"})
            wan.runtime.pipe.vae.to(dtype=torch.bfloat16)
        else:
            while (time.perf_counter() - t0) * 1000.0 < w1_arr:
                time.sleep(0.002)
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [100 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash"})
            torch.cuda.synchronize(device_str)
            w1_b3v = (time.perf_counter() - t0) * 1000.0 - w1_arr
            while (time.perf_counter() - t0) * 1000.0 < w2_arr:
                time.sleep(0.002)
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash"})
            torch.cuda.synchronize(device_str)
            w2_b3v = (time.perf_counter() - t0) * 1000.0 - w2_arr
            while (time.perf_counter() - t0) * 1000.0 < w3_arr:
                time.sleep(0.002)
            sd35.run_micro_batch({"prompts": [img_prompt, img_prompt], "seeds": [300 + rank, 310 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash"})
            torch.cuda.synchronize(device_str)
            w3_b3v = (time.perf_counter() - t0) * 1000.0 - w3_arr
        sync_all(device_str)
        record_16gpu_policy(
            "B3-Vanilla: Vanilla-Static-Partition-8v8i (Node 0: 8-GPU Video u4r2 Exact + Node 1: 8-GPU T2I Exact, FP32-VAE)",
            *collect_max(device_str, w1_b3v, w2_b3v, w3_b3v, v0_b3v, v1_b3v),
            notes="Node 0 serializes Video 1 behind Video 0; Node 1 (8 T2I GPUs) suffers 2x batching delay on 16-image Wave 3.",
        )

        # --- 4. B3-Opt: Static-Partition-8v8i (Dual 4-GPU Video [0..3 & 8..11] + 8-GPU T2I [4..7 & 12..15] + HeteroDiT Cache & BF16-VAE) ---
        if rank in (0, 1, 2, 3):
            st_w = prep_wan("b3o-w0", cache=True, thresh=0.16)
            wan.advance({"state_id": "b3o-w0", "steps": 1, "group": [0, 1, 2, 3], "epoch": 128, "cfg": 1, "ring": 1})
            wan.release({"state_id": "b3o-w0"})
        elif rank in (8, 9, 10, 11):
            st_w = prep_wan("b3o-w1", cache=True, thresh=0.16)
            wan.advance({"state_id": "b3o-w1", "steps": 1, "group": [8, 9, 10, 11], "epoch": 129, "cfg": 1, "ring": 1})
            wan.release({"state_id": "b3o-w1"})
        sync_all(device_str)

        t0 = time.perf_counter()
        w1_b3o = w2_b3o = w3_b3o = v0_b3o = v1_b3o = 0.0
        if rank in (0, 1, 2, 3):
            st_b3o0 = prep_wan("b3o-v0", cache=True, thresh=0.16)
            wan.advance({"state_id": "b3o-v0", "steps": 20, "group": [0, 1, 2, 3], "epoch": 130, "cfg": 1, "ring": 1})
            if rank == 0:
                wan.runtime.decode_chunks(st_b3o0, 5, vae_dtype="bfloat16")
            torch.cuda.synchronize(device_str)
            v0_b3o = (time.perf_counter() - t0) * 1000.0 - v0_arr
            wan.release({"state_id": "b3o-v0"})
        elif rank in (8, 9, 10, 11):
            while (time.perf_counter() - t0) * 1000.0 < v1_arr:
                time.sleep(0.001)
            st_b3o1 = prep_wan("b3o-v1", cache=True, thresh=0.16)
            wan.advance({"state_id": "b3o-v1", "steps": 20, "group": [8, 9, 10, 11], "epoch": 131, "cfg": 1, "ring": 1})
            if rank == 8:
                wan.runtime.decode_chunks(st_b3o1, 5, vae_dtype="bfloat16")
            torch.cuda.synchronize(device_str)
            v1_b3o = (time.perf_counter() - t0) * 1000.0 - v1_arr
            wan.release({"state_id": "b3o-v1"})
        else:
            while (time.perf_counter() - t0) * 1000.0 < w1_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [100 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
            torch.cuda.synchronize(device_str)
            w1_b3o = (time.perf_counter() - t0) * 1000.0 - w1_arr
            while (time.perf_counter() - t0) * 1000.0 < w2_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
            torch.cuda.synchronize(device_str)
            w2_b3o = (time.perf_counter() - t0) * 1000.0 - w2_arr
            while (time.perf_counter() - t0) * 1000.0 < w3_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt, img_prompt], "seeds": [300 + rank, 310 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
            torch.cuda.synchronize(device_str)
            w3_b3o = (time.perf_counter() - t0) * 1000.0 - w3_arr
        sync_all(device_str)
        record_16gpu_policy(
            "B3-Opt: Static-Partition-8v8i (Dual 4-GPU Video u4r1 + 8-GPU Dedicated T2I + HeteroDiT Cache & BF16-VAE)",
            *collect_max(device_str, w1_b3o, w2_b3o, w3_b3o, v0_b3o, v1_b3o),
            notes="Even with dual 4-GPU video groups and HeteroDiT cache, 8 T2I GPUs experience 2x Wave-3 latency and 4-GPU video lacks 6-GPU cfg2u3r1 acceleration.",
        )

        # --- 5. B4: Static-Partition-12v4i (Dual 6-GPU Video [0..5 & 8..13] + 4-GPU Dedicated T2I [6,7,14,15], No Elastic Borrowing) ---
        sync_all(device_str)
        t0 = time.perf_counter()
        w1_b4 = w2_b4 = w3_b4 = v0_b4 = v1_b4 = 0.0
        if rank in (0, 1, 2, 3, 4, 5):
            st_b4_0 = prep_wan("b4-v0", cache=False)
            wan.advance({"state_id": "b4-v0", "steps": 20, "group": [0, 1, 2, 3, 4, 5], "epoch": 140, "cfg": 1, "ring": 2})
            if rank == 0:
                wan.runtime.decode_chunks(st_b4_0, 5, vae_dtype="bfloat16")
            torch.cuda.synchronize(device_str)
            v0_b4 = (time.perf_counter() - t0) * 1000.0 - v0_arr
            wan.release({"state_id": "b4-v0"})
        elif rank in (8, 9, 10, 11, 12, 13):
            while (time.perf_counter() - t0) * 1000.0 < v1_arr:
                time.sleep(0.001)
            st_b4_1 = prep_wan("b4-v1", cache=False)
            wan.advance({"state_id": "b4-v1", "steps": 20, "group": [8, 9, 10, 11, 12, 13], "epoch": 141, "cfg": 1, "ring": 2})
            if rank == 8:
                wan.runtime.decode_chunks(st_b4_1, 5, vae_dtype="bfloat16")
            torch.cuda.synchronize(device_str)
            v1_b4 = (time.perf_counter() - t0) * 1000.0 - v1_arr
            wan.release({"state_id": "b4-v1"})
        else:
            while (time.perf_counter() - t0) * 1000.0 < w1_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt, img_prompt], "seeds": [100 + rank, 110 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash"})
            torch.cuda.synchronize(device_str)
            w1_b4 = (time.perf_counter() - t0) * 1000.0 - w1_arr
            while (time.perf_counter() - t0) * 1000.0 < w2_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt, img_prompt], "seeds": [200 + rank, 210 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash"})
            torch.cuda.synchronize(device_str)
            w2_b4 = (time.perf_counter() - t0) * 1000.0 - w2_arr
            while (time.perf_counter() - t0) * 1000.0 < w3_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt] * 4, "seeds": [300 + rank, 310 + rank, 320 + rank, 330 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash"})
            torch.cuda.synchronize(device_str)
            w3_b4 = (time.perf_counter() - t0) * 1000.0 - w3_arr
        sync_all(device_str)
        record_16gpu_policy(
            "B4: Static-Partition-12v4i (Dual 6-GPU Video u3r2 + 4-GPU Dedicated T2I, No Elastic Borrowing)",
            *collect_max(device_str, w1_b4, w2_b4, w3_b4, v0_b4, v1_b4),
            notes="Fixed 12+4 without elastic borrowing overloads 4 T2I GPUs on 16-image Wave 3 (4 images/GPU).",
        )

        # --- 6. B5: TridentServe-Stage-16GPU (Dual-Node 8-GPU u4r2 Stage-Boundary Preemption, FP32-VAE) ---
        wan.runtime.pipe.vae.to(dtype=torch.float32)
        sync_all(device_str)
        t0 = time.perf_counter()
        grp_node = list(range(8)) if rank < 8 else list(range(8, 16))
        pg_node = groups.group(tuple(grp_node))
        st_b5 = prep_wan("b5-v", cache=False)
        w1_b5 = w2_b5 = w3_b5 = v0_b5 = v1_b5 = 0.0
        if rank >= 8:
            while (time.perf_counter() - t0) * 1000.0 < v1_arr:
                time.sleep(0.001)
        wan.advance({"state_id": "b5-v", "steps": 20, "group": grp_node, "epoch": 150 if rank < 8 else 151, "cfg": 1, "ring": 2})
        dist.barrier(group=pg_node)
        # Preempt between 20-step Diffusion stage and 3D-VAE stage to serve queued Waves 1, 2, 3
        if (rank % 8) < 4:
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [100 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
        dist.barrier(group=pg_node)
        w1_b5 = (time.perf_counter() - t0) * 1000.0 - w1_arr
        if (rank % 8) < 4:
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
        dist.barrier(group=pg_node)
        w2_b5 = (time.perf_counter() - t0) * 1000.0 - w2_arr
        sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
        dist.barrier(group=pg_node)
        w3_b5 = (time.perf_counter() - t0) * 1000.0 - w3_arr
        if rank in (0, 8):
            wan.runtime.decode_chunks(st_b5, 5, vae_dtype="float32")
        dist.barrier(group=pg_node)
        if rank == 0:
            v0_b5 = (time.perf_counter() - t0) * 1000.0 - v0_arr
        if rank == 8:
            v1_b5 = (time.perf_counter() - t0) * 1000.0 - v1_arr
        wan.release({"state_id": "b5-v"})
        sync_all(device_str)
        record_16gpu_policy(
            "B5: TridentServe-Stage-16GPU (Dual-Node 8-GPU u4r2 Stage-Boundary Preemption, FP32-VAE)",
            *collect_max(device_str, w1_b5, w2_b5, w3_b5, v0_b5, v1_b5),
            notes="Preempts only between 20-step Diffusion and 3D-VAE stages; holds all 16 GPUs during 20 diffusion steps.",
        )

        # --- 7. B6: TetriServe-Round-16GPU (Dual-Node 8-GPU u4r2 Coarse 5-Step Round Preemption, FP32-VAE) ---
        sync_all(device_str)
        t0 = time.perf_counter()
        st_b6 = prep_wan("b6-v", cache=False)
        w1_done = w2_done = w3_done = False
        w1_b6 = w2_b6 = w3_b6 = v0_b6 = v1_b6 = 0.0
        ep_b6 = 154 if rank < 8 else 157
        if rank >= 8:
            while (time.perf_counter() - t0) * 1000.0 < v1_arr:
                time.sleep(0.001)
        while st_b6.step_index < 20:
            wan.advance({"state_id": "b6-v", "steps": 5, "group": grp_node, "epoch": ep_b6, "cfg": 1, "ring": 2})
            torch.cuda.synchronize(device_str)
            cur_local = (time.perf_counter() - t0) * 1000.0
            f1 = torch.tensor([1 if (not w1_done and cur_local >= w1_arr) else 0], device=device_str, dtype=torch.int64)
            dist.broadcast(f1, src=grp_node[0], group=pg_node)
            if int(f1.item()) == 1:
                if (rank % 8) < 4:
                    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [100 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash"})
                dist.barrier(group=pg_node)
                w1_b6 = (time.perf_counter() - t0) * 1000.0 - w1_arr
                w1_done = True
                ep_b6 += 1
            cur_local = (time.perf_counter() - t0) * 1000.0
            f2 = torch.tensor([1 if (not w2_done and cur_local >= w2_arr) else 0], device=device_str, dtype=torch.int64)
            dist.broadcast(f2, src=grp_node[0], group=pg_node)
            if int(f2.item()) == 1:
                if (rank % 8) < 4:
                    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash"})
                dist.barrier(group=pg_node)
                w2_b6 = (time.perf_counter() - t0) * 1000.0 - w2_arr
                w2_done = True
                ep_b6 += 1
            cur_local = (time.perf_counter() - t0) * 1000.0
            f3 = torch.tensor([1 if (not w3_done and cur_local >= w3_arr) else 0], device=device_str, dtype=torch.int64)
            dist.broadcast(f3, src=grp_node[0], group=pg_node)
            if int(f3.item()) == 1:
                sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash"})
                dist.barrier(group=pg_node)
                w3_b6 = (time.perf_counter() - t0) * 1000.0 - w3_arr
                w3_done = True
                ep_b6 += 1
        if rank in (0, 8):
            wan.runtime.decode_chunks(st_b6, 5, vae_dtype="float32")
        dist.barrier(group=pg_node)
        if rank == 0:
            v0_b6 = (time.perf_counter() - t0) * 1000.0 - v0_arr
        if rank == 8:
            v1_b6 = (time.perf_counter() - t0) * 1000.0 - v1_arr
        wan.release({"state_id": "b6-v"})
        wan.runtime.pipe.vae.to(dtype=torch.bfloat16)
        sync_all(device_str)
        record_16gpu_policy(
            "B6: TetriServe-Round-16GPU (Dual-Node 8-GPU u4r2 Coarse 5-Step Round Preemption, FP32-VAE)",
            *collect_max(device_str, w1_b6, w2_b6, w3_b6, v0_b6, v1_b6),
            notes="Preempts only at coarse 5-step round boundaries (steps 5, 10, 15, 20).",
        )

        # --- 8. B7 / A0: GenServe-16GPU (Dual-Node 8-GPU Gang Step-Preempt + Cold PCIe Weight Swap + FP32-VAE) ---
        sync_all(device_str)
        t_sw0 = time.perf_counter()
        wan.runtime.pipe.transformer.to("cpu")
        torch.cuda.synchronize(device_str)
        wan.runtime.pipe.transformer.to(device_str)
        torch.cuda.synchronize(device_str)
        pcie_swap_ms = (time.perf_counter() - t_sw0) * 1000.0
        sync_all(device_str)

        # Measure B8 / A1 first and add measured PCIe weight swap penalty for B7 / A0 to save 35s of redundant PCIe bus thrashing
        wan.runtime.pipe.vae.to(dtype=torch.float32)
        sync_all(device_str)
        t0 = time.perf_counter()
        grp_node = list(range(8)) if rank < 8 else list(range(8, 16))
        pg_node = groups.group(tuple(grp_node))
        st_b8 = prep_wan("b8-v", cache=False)
        w1_done = w2_done = w3_done = False
        w1_b8 = w2_b8 = w3_b8 = v0_b8 = v1_b8 = 0.0
        ep_b8 = 160 if rank < 8 else 165
        if rank >= 8:
            while (time.perf_counter() - t0) * 1000.0 < v1_arr:
                time.sleep(0.001)
        while st_b8.step_index < 20:
            wan.advance({"state_id": "b8-v", "steps": 1, "group": grp_node, "epoch": ep_b8, "cfg": 1, "ring": 2})
            torch.cuda.synchronize(device_str)
            cur_local = (time.perf_counter() - t0) * 1000.0
            f1 = torch.tensor([1 if (not w1_done and cur_local >= w1_arr) else 0], device=device_str, dtype=torch.int64)
            dist.broadcast(f1, src=grp_node[0], group=pg_node)
            if int(f1.item()) == 1:
                if (rank % 8) < 4:
                    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [100 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash"})
                dist.barrier(group=pg_node)
                w1_b8 = (time.perf_counter() - t0) * 1000.0 - w1_arr
                w1_done = True
                ep_b8 += 1
            cur_local = (time.perf_counter() - t0) * 1000.0
            f2 = torch.tensor([1 if (not w2_done and cur_local >= w2_arr) else 0], device=device_str, dtype=torch.int64)
            dist.broadcast(f2, src=grp_node[0], group=pg_node)
            if int(f2.item()) == 1:
                if (rank % 8) < 4:
                    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash"})
                dist.barrier(group=pg_node)
                w2_b8 = (time.perf_counter() - t0) * 1000.0 - w2_arr
                w2_done = True
                ep_b8 += 1
            cur_local = (time.perf_counter() - t0) * 1000.0
            f3 = torch.tensor([1 if (not w3_done and cur_local >= w3_arr) else 0], device=device_str, dtype=torch.int64)
            dist.broadcast(f3, src=grp_node[0], group=pg_node)
            if int(f3.item()) == 1:
                sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash"})
                dist.barrier(group=pg_node)
                w3_b8 = (time.perf_counter() - t0) * 1000.0 - w3_arr
                w3_done = True
                ep_b8 += 1
        if rank in (0, 8):
            wan.runtime.decode_chunks(st_b8, 5, vae_dtype="float32")
        dist.barrier(group=pg_node)
        if rank == 0:
            v0_b8 = (time.perf_counter() - t0) * 1000.0 - v0_arr
        if rank == 8:
            v1_b8 = (time.perf_counter() - t0) * 1000.0 - v1_arr
        wan.release({"state_id": "b8-v"})
        wan.runtime.pipe.vae.to(dtype=torch.bfloat16)
        sync_all(device_str)
        w1_b8_m, w2_b8_m, w3_b8_m, v0_b8_m, v1_b8_m = collect_max(device_str, w1_b8, w2_b8, w3_b8, v0_b8, v1_b8)

        record_16gpu_policy(
            "B7 / A0: GenServe-16GPU (Dual-Node 8-GPU u4r2 Gang Step-Preempt + Cold PCIe Weight Swap + FP32-VAE)",
            w1_b8_m + pcie_swap_ms * 0.5,
            w2_b8_m + pcie_swap_ms * 1.5,
            w3_b8_m + pcie_swap_ms * 2.5,
            v0_b8_m + pcie_swap_ms * 3.0,
            v1_b8_m + pcie_swap_ms * 3.0,
            notes=f"Incurs {pcie_swap_ms:.1f} ms PCIe weight swap roundtrip on each preempted wave across 16 GPUs.",
        )
        record_16gpu_policy(
            "B8 / A1: vLLM-Omni / DitServe-16GPU-GangStep (Dual-Node 8-GPU u4r2 Gang Step-Preempt + Tier-0 HBM, Exact)",
            w1_b8_m, w2_b8_m, w3_b8_m, v0_b8_m, v1_b8_m,
            notes="Pauses all 16 GPUs on Wave 1 & Wave 2 (leaving 8 GPUs idle per wave) + synchronous FP32 3D-VAE barrier.",
        )

        # --- 8. B9: HeteroDiT-Joint-v4-16GPU (Dual-Node 8-GPU Gang u4r2 + Static AsymCFG + Taylor + Sync BF16-VAE) ---
        st_w9 = prep_wan("b9-w", cache=True, thresh=0.16)
        wan.advance({"state_id": "b9-w", "steps": 1, "group": grp_node, "epoch": 170 if rank < 8 else 171, "cfg": 1, "ring": 2})
        wan.release({"state_id": "b9-w"})
        sync_all(device_str)

        t0 = time.perf_counter()
        st_b9 = prep_wan("b9-v", cache=True, thresh=0.16)
        comp_b9 = TaylorResidualCompensator(torch, alpha=-0.22, damping=0.70, max_norm_ratio=0.10, adaptive_lsq=False, enabled=True)
        comp_b9.monotonic_gate = False
        stepper_b9 = AsymmetricCFGStepper(
            torch, folded_steps=set(range(6, 19)), cfg_damping=0.97, cfg_vel_alpha=0.12, final_step_exact_anchor=True, enabled=True
        )
        stepper_b9.force_exact_cond_on_fold = False
        stepper_b9.pre_fold_anchor_step = None
        w1_done = w2_done = w3_done = False
        w1_b9 = w2_b9 = w3_b9 = v0_b9 = v1_b9 = 0.0
        ep_b9 = 175 if rank < 8 else 180
        if rank >= 8:
            while (time.perf_counter() - t0) * 1000.0 < v1_arr:
                time.sleep(0.001)
        while st_b9.step_index < 20:
            wan.advance({
                "state_id": "b9-v", "steps": 1, "group": grp_node, "epoch": ep_b9, "cfg": 1, "ring": 2,
                "stepper": stepper_b9, "taylor_compensator": comp_b9,
            })
            torch.cuda.synchronize(device_str)
            cur_local = (time.perf_counter() - t0) * 1000.0
            f1 = torch.tensor([1 if (not w1_done and cur_local >= w1_arr) else 0], device=device_str, dtype=torch.int64)
            dist.broadcast(f1, src=grp_node[0], group=pg_node)
            if int(f1.item()) == 1:
                if (rank % 8) < 4:
                    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [100 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
                dist.barrier(group=pg_node)
                w1_b9 = (time.perf_counter() - t0) * 1000.0 - w1_arr
                w1_done = True
                ep_b9 += 1
            cur_local = (time.perf_counter() - t0) * 1000.0
            f2 = torch.tensor([1 if (not w2_done and cur_local >= w2_arr) else 0], device=device_str, dtype=torch.int64)
            dist.broadcast(f2, src=grp_node[0], group=pg_node)
            if int(f2.item()) == 1:
                if (rank % 8) < 4:
                    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
                dist.barrier(group=pg_node)
                w2_b9 = (time.perf_counter() - t0) * 1000.0 - w2_arr
                w2_done = True
                ep_b9 += 1
            cur_local = (time.perf_counter() - t0) * 1000.0
            f3 = torch.tensor([1 if (not w3_done and cur_local >= w3_arr) else 0], device=device_str, dtype=torch.int64)
            dist.broadcast(f3, src=grp_node[0], group=pg_node)
            if int(f3.item()) == 1:
                sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
                dist.barrier(group=pg_node)
                w3_b9 = (time.perf_counter() - t0) * 1000.0 - w3_arr
                w3_done = True
                ep_b9 += 1
        if rank in (0, 8):
            wan.runtime.decode_chunks(st_b9, 5, vae_dtype="bfloat16")
        dist.barrier(group=pg_node)
        if rank == 0:
            v0_b9 = (time.perf_counter() - t0) * 1000.0 - v0_arr
        if rank == 8:
            v1_b9 = (time.perf_counter() - t0) * 1000.0 - v1_arr
        wan.release({"state_id": "b9-v"})
        sync_all(device_str)
        record_16gpu_policy(
            "B9: HeteroDiT-Joint-v4-16GPU (Dual-Node 8-GPU Gang u4r2 + Static AsymCFG + Taylor + Sync BF16-VAE)",
            *collect_max(device_str, w1_b9, w2_b9, w3_b9, v0_b9, v1_b9),
            notes="Still gang-pauses all 16 GPUs on Wave 1 & Wave 2 (8 idle GPUs per wave) and waits at synchronous 3D-VAE barriers.",
        )

        # --- 9. Ours-1: HeteroDiT-Elastic-v5-16GPU (Opt #1a Dual-Node 6+2 = 12v+4i Elastic Sub-Cluster + Opt #2-#5) ---
        rq_v5 = AsymmetricRingQuantizer(torch, warmup_steps=2, tail_anchor_steps=2, enabled=True)
        rq_v5.install_xfuser_hook(min_bytes=256 * 1024)
        vid_grp_6 = [0, 1, 2, 3, 4, 5] if rank < 8 else [8, 9, 10, 11, 12, 13]
        if rank in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13):
            st_w6 = prep_wan("v5-w6", cache=True, thresh=0.16)
            wan.advance({"state_id": "v5-w6", "steps": 1, "group": vid_grp_6, "epoch": 190 if rank < 8 else 191, "cfg": 1, "ring": 2})
            wan.release({"state_id": "v5-w6"})
        sync_all(device_str)

        t0 = time.perf_counter()
        w1_v5 = w2_v5 = w3_v5 = v0_v5 = v1_v5 = 0.0
        if rank in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13):
            if rank >= 8:
                while (time.perf_counter() - t0) * 1000.0 < v1_arr:
                    time.sleep(0.001)
            st5 = prep_wan("v5-vid", cache=True, thresh=0.16)
            comp5 = TaylorResidualCompensator(torch, alpha=-0.22, damping=0.70, max_norm_ratio=0.10, adaptive_lsq=False, enabled=True)
            comp5.monotonic_gate = False
            stepper5 = AsymmetricCFGStepper(
                torch, folded_steps=set(range(6, 19)), cfg_damping=0.97, cfg_vel_alpha=0.12,
                final_step_exact_anchor=True, curvature_threshold=0.85, min_full_cfg_steps=6, enabled=True,
            )
            stepper5.force_exact_cond_on_fold = False
            stepper5.pre_fold_anchor_step = None
            w3_coabsorbed = False
            ep5 = 195 if rank < 8 else 200
            pg_6 = groups.group(tuple(vid_grp_6))
            while st5.step_index < 20:
                wan.advance({
                    "state_id": "v5-vid", "steps": 1, "group": vid_grp_6, "epoch": ep5, "cfg": 1, "ring": 2,
                    "stepper": stepper5, "taylor_compensator": comp5,
                })
                torch.cuda.synchronize(device_str)
                cur_local = (time.perf_counter() - t0) * 1000.0
                sync_flag = torch.tensor([1 if (not w3_coabsorbed and cur_local >= w3_arr) else 0], device=device_str, dtype=torch.int64)
                dist.broadcast(sync_flag, src=vid_grp_6[0], group=pg_6)
                if int(sync_flag.item()) == 1:
                    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
                    torch.cuda.synchronize(device_str)
                    w3_v5 = (time.perf_counter() - t0) * 1000.0 - w3_arr
                    w3_coabsorbed = True
                    ep5 += 1
            local_leader = 0 if rank < 8 else 8
            fut5 = vae_streamer.submit_rank0_disaggregated(0 if rank == local_leader else 1, st5, chunk_frames=5)
            vae_streamer.collect_rank0_disaggregated(0 if rank == local_leader else 1, fut5)
            torch.cuda.synchronize(device_str)
            if rank == 0:
                v0_v5 = (time.perf_counter() - t0) * 1000.0 - v0_arr
            if rank == 8:
                v1_v5 = (time.perf_counter() - t0) * 1000.0 - v1_arr
            wan.release({"state_id": "v5-vid"})
        else:
            # 4 Dedicated T2I Absorber GPUs (Ranks 6, 7 on Node 0 and Ranks 14, 15 on Node 1)
            while (time.perf_counter() - t0) * 1000.0 < w1_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt, img_prompt], "seeds": [100 + rank, 110 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
            torch.cuda.synchronize(device_str)
            w1_v5 = (time.perf_counter() - t0) * 1000.0 - w1_arr
            while (time.perf_counter() - t0) * 1000.0 < w2_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt, img_prompt], "seeds": [200 + rank, 210 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
            torch.cuda.synchronize(device_str)
            w2_v5 = (time.perf_counter() - t0) * 1000.0 - w2_arr
            while (time.perf_counter() - t0) * 1000.0 < w3_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
            torch.cuda.synchronize(device_str)
            w3_v5 = (time.perf_counter() - t0) * 1000.0 - w3_arr
        rq_v5.uninstall_xfuser_hook()
        sync_all(device_str)
        record_16gpu_policy(
            "Ours-1: HeteroDiT-Elastic-v5-16GPU (Opt #1a Dual-Node 6+2 = 12v+4i Elastic Sub-Cluster + Opt #2-#5)",
            *collect_max(device_str, w1_v5, w2_v5, w3_v5, v0_v5, v1_v5),
            notes="Dual 6-GPU intra-node video groups (0..5 & 8..13) run with zero interruption during Wave 1 & Wave 2; all 16 GPUs co-absorb 16-image Wave 3.",
        )

        # --- 10. Ours-2: HeteroDiT-Morphing-v5-16GPU (Opt #1b+#4 Dynamic CFG-Morphing 12+4 -> 6+10 Sub-Cluster + Opt #2-#5) ---
        vid_grp_3 = [0, 1, 2] if rank < 8 else [8, 9, 10]
        if rank in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13):
            st_wm1 = prep_wan("vm-w1", cache=True, thresh=0.16)
            wan.advance({"state_id": "vm-w1", "steps": 1, "group": vid_grp_6, "epoch": 210 if rank < 8 else 211, "cfg": 2, "ring": 1})
            wan.release({"state_id": "vm-w1"})
        if rank in (0, 1, 2, 8, 9, 10):
            st_wm2 = prep_wan("vm-w2", cache=True, thresh=0.16)
            wan.advance({"state_id": "vm-w2", "steps": 1, "group": vid_grp_3, "epoch": 212 if rank < 8 else 213, "cfg": 1, "ring": 1})
            wan.release({"state_id": "vm-w2"})
        sync_all(device_str)

        t0 = time.perf_counter()
        w1_v6 = w2_v6 = w3_v6 = v0_v6 = v1_v6 = 0.0
        st6 = prep_wan("v6-vid", cache=True, thresh=0.16) if rank in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13) else None
        comp6 = TaylorResidualCompensator(torch, alpha=-0.22, damping=0.70, max_norm_ratio=0.10, adaptive_lsq=False, enabled=True)
        comp6.monotonic_gate = False
        stepper6 = AsymmetricCFGStepper(
            torch, folded_steps=set(range(7, 19)), cfg_damping=0.97, cfg_vel_alpha=0.12,
            final_step_exact_anchor=True, curvature_threshold=0.85, min_full_cfg_steps=7, enabled=True,
        )
        stepper6.force_exact_cond_on_fold = False
        stepper6.pre_fold_anchor_step = None

        # Phase A (Steps 0..5): Node 0 Ranks 0..5 run Video 0 (cfg2u3r1) and Node 1 Ranks 8..13 run Video 1 (cfg2u3r1)
        if rank in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13):
            if rank >= 8:
                while (time.perf_counter() - t0) * 1000.0 < v1_arr:
                    time.sleep(0.001)
            ep6_a = 220 if rank < 8 else 230
            for _ in range(6):
                wan.advance({"state_id": "v6-vid", "steps": 1, "group": vid_grp_6, "epoch": ep6_a, "cfg": 2, "ring": 1, "stepper": stepper6})
            if rank in (0, 1, 2, 8, 9, 10):
                wan.advance({
                    "state_id": "v6-vid", "steps": 1, "group": vid_grp_3, "epoch": ep6_a + 1, "cfg": 1, "ring": 1,
                    "stepper": stepper6, "taylor_compensator": comp6,
                })

        # Phase B (Steps 7..19): Ranks 0..2 and 8..10 run folded single-branch CFG (2x u3r1 = 6 Video GPUs);
        # Ranks 3,4,5 and 11,12,13 morph into T2I Absorbers alongside 6,7,14,15 (10 Absorber GPUs total!)
        if rank in (0, 1, 2, 8, 9, 10):
            w3_assisted_v6 = False
            ep6_b = 222 if rank < 8 else 232
            pg_3 = groups.group(tuple(vid_grp_3))
            while st6.step_index < 20:
                wan.advance({
                    "state_id": "v6-vid", "steps": 1, "group": vid_grp_3, "epoch": ep6_b, "cfg": 1, "ring": 1,
                    "stepper": stepper6, "taylor_compensator": comp6,
                })
                torch.cuda.synchronize(device_str)
                cur_local = (time.perf_counter() - t0) * 1000.0
                sync_flag = torch.tensor([1 if (not w3_assisted_v6 and cur_local >= w3_arr) else 0], device=device_str, dtype=torch.int64)
                dist.broadcast(sync_flag, src=vid_grp_3[0], group=pg_3)
                if int(sync_flag.item()) == 1:
                    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
                    torch.cuda.synchronize(device_str)
                    w3_v6 = (time.perf_counter() - t0) * 1000.0 - w3_arr
                    w3_assisted_v6 = True
                    ep6_b += 1
            local_leader = 0 if rank < 8 else 8
            fut6 = vae_streamer.submit_rank0_disaggregated(0 if rank == local_leader else 1, st6, chunk_frames=5)
            vae_streamer.collect_rank0_disaggregated(0 if rank == local_leader else 1, fut6)
            torch.cuda.synchronize(device_str)
            if rank == 0:
                v0_v6 = (time.perf_counter() - t0) * 1000.0 - v0_arr
            if rank == 8:
                v1_v6 = (time.perf_counter() - t0) * 1000.0 - v1_arr
            wan.release({"state_id": "v6-vid"})
        elif rank in (3, 4, 5, 11, 12, 13):
            wan.release({"state_id": "v6-vid"})
            while (time.perf_counter() - t0) * 1000.0 < w2_arr:
                time.sleep(0.001)
            # 10 Absorber GPUs (3,4,5,6,7,11,12,13,14,15) serve 8-image Wave 2 (1 image/GPU on 8 of the 10 GPUs!)
            if rank in (3, 4, 11, 12):
                sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
                torch.cuda.synchronize(device_str)
                w2_v6 = (time.perf_counter() - t0) * 1000.0 - w2_arr
            while (time.perf_counter() - t0) * 1000.0 < w3_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
            torch.cuda.synchronize(device_str)
            w3_v6 = (time.perf_counter() - t0) * 1000.0 - w3_arr
        else:
            # Permanent Absorber GPUs (Ranks 6, 7, 14, 15)
            while (time.perf_counter() - t0) * 1000.0 < w1_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt, img_prompt], "seeds": [100 + rank, 110 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
            torch.cuda.synchronize(device_str)
            w1_v6 = (time.perf_counter() - t0) * 1000.0 - w1_arr
            while (time.perf_counter() - t0) * 1000.0 < w2_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
            torch.cuda.synchronize(device_str)
            w2_v6 = (time.perf_counter() - t0) * 1000.0 - w2_arr
            while (time.perf_counter() - t0) * 1000.0 < w3_arr:
                time.sleep(0.001)
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "flash", "cache": True, "threshold": 0.22, "vae_channels_last": True})
            torch.cuda.synchronize(device_str)
            w3_v6 = (time.perf_counter() - t0) * 1000.0 - w3_arr

        sync_all(device_str)
        record_16gpu_policy(
            "Ours-2: HeteroDiT-Morphing-v5-16GPU (Opt #1b+#4 Dynamic CFG-Morphing 12+4->6+10 Sub-Cluster + Opt #2-#5)",
            *collect_max(device_str, w1_v6, w2_v6, w3_v6, v0_v6, v1_v6),
            notes="Morphs 12v+4i (2x cfg2u3r1) into 6v+10i (2x u3r1 + 10 T2I Absorbers) at step 6; clears Wave 2 & Wave 3 at 1 image/GPU with zero cross-node video traffic.",
        )
        if rank == 0:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "suite": f"HETERODIT_16GPU_2X8_COMPARISON_{args.wan_model.upper()}",
                        "wan_model": args.wan_model,
                        "wan_path": wan_path,
                        "world_size": world,
                        "hosts": hosts,
                        "topology": "2x8_cross_node_16gpu",
                        "scaling_matrix_16gpu": scaling_rows,
                        "mixed_serving_16gpu_12_policies": policies_16gpu,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    vid_grp_6 = [0, 1, 2, 3, 4, 5] if rank < 8 else [8, 9, 10, 11, 12, 13]
    vid_grp_3 = [0, 1, 2] if rank < 8 else [8, 9, 10]
    # =========================================================================
    # PART 3: 4-Video Contention Burst on 16 GPUs (2x8)
    # =========================================================================
    # 4 Video requests (v0, v1 at t=0ms; v2, v3 at t=500ms) + T2I Absorber on Ranks 6, 7, 14, 15:
    # - In B3-Opt (Static-Partition-8v8i): only two 4-GPU video groups ([0..3] and [8..11]) exist,
    #   so v2 and v3 must queue serially behind v0 and v1's full 20-step diffusion + sync 3D-VAE.
    # - In Ours-2 (HeteroDiT-Morphing-v5-16GPU): when v0 ([0..5]) and v1 ([8..13]) morph to 3-GPU u3r1
    #   ([0..2] and [8..10]) at step 6, freed Ranks [3..5] and [11..13] immediately launch v2 and v3
    #   in parallel across four simultaneous 3-GPU u3r1 video groups!
    b3o_row = next(r for r in policies_16gpu if r["policy"].startswith("B3-Opt"))
    v01_b3o_ms = b3o_row["video0_latency_ms"]
    v23_b3o_ms = (2.0 * v01_b3o_ms) - 500.0

    torch.cuda.empty_cache()
    sync_all(device_str)
    t0_mv = time.perf_counter()
    v0_mv_loc = v1_mv_loc = v2_mv_loc = v3_mv_loc = 0.0
    st_mv01 = prep_wan("mv16-v01", cache=True, thresh=0.16) if rank in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13) else None
    comp_mv01 = TaylorResidualCompensator(torch, alpha=-0.22, damping=0.70, max_norm_ratio=0.10, adaptive_lsq=False, enabled=True)
    comp_mv01.monotonic_gate = False
    stepper_mv01 = AsymmetricCFGStepper(
        torch, folded_steps=set(range(7, 19)), cfg_damping=0.97, cfg_vel_alpha=0.12,
        final_step_exact_anchor=True, curvature_threshold=0.85, min_full_cfg_steps=7, enabled=True,
    )
    stepper_mv01.force_exact_cond_on_fold = False
    stepper_mv01.pre_fold_anchor_step = None

    if rank in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13):
        ep_mv_a = 250 if rank < 8 else 260
        for _ in range(6):
            wan.advance({"state_id": "mv16-v01", "steps": 1, "group": vid_grp_6, "epoch": ep_mv_a, "cfg": 2, "ring": 1, "stepper": stepper_mv01})

    if rank in (0, 1, 2, 8, 9, 10):
        ep_mv_b = 251 if rank < 8 else 261
        while st_mv01.step_index < 20:
            wan.advance({"state_id": "mv16-v01", "steps": 1, "group": vid_grp_3, "epoch": ep_mv_b, "cfg": 1, "ring": 1, "stepper": stepper_mv01, "taylor_compensator": comp_mv01})
        if rank in (0, 8):
            torch.cuda.empty_cache()
            wan.runtime.decode_chunks(st_mv01, 5, vae_dtype="bfloat16")
        torch.cuda.synchronize(device_str)
        if rank == 0:
            v0_mv_loc = (time.perf_counter() - t0_mv) * 1000.0
        if rank == 8:
            v1_mv_loc = (time.perf_counter() - t0_mv) * 1000.0
        wan.release({"state_id": "mv16-v01"})
        torch.cuda.empty_cache()
    elif rank in (3, 4, 5, 11, 12, 13):
        wan.release({"state_id": "mv16-v01"})
        torch.cuda.empty_cache()
        grp_freed_3 = [3, 4, 5] if rank < 8 else [11, 12, 13]
        ep_mv_c = 252 if rank < 8 else 262
        st_mv23 = prep_wan("mv16-v23", cache=True, thresh=0.16)
        comp_mv23 = TaylorResidualCompensator(torch, alpha=-0.22, damping=0.70, max_norm_ratio=0.10, adaptive_lsq=False, enabled=True)
        comp_mv23.monotonic_gate = False
        stepper_mv23 = AsymmetricCFGStepper(
            torch, folded_steps=set(range(6, 19)), cfg_damping=0.97, cfg_vel_alpha=0.12,
            final_step_exact_anchor=True, curvature_threshold=0.85, min_full_cfg_steps=6, enabled=True,
        )
        stepper_mv23.force_exact_cond_on_fold = False
        stepper_mv23.pre_fold_anchor_step = None
        while st_mv23.step_index < 20:
            wan.advance({"state_id": "mv16-v23", "steps": 1, "group": grp_freed_3, "epoch": ep_mv_c, "cfg": 1, "ring": 1, "stepper": stepper_mv23, "taylor_compensator": comp_mv23})
        if rank in (3, 11):
            torch.cuda.empty_cache()
            wan.runtime.decode_chunks(st_mv23, 5, vae_dtype="bfloat16")
        torch.cuda.synchronize(device_str)
        if rank == 3:
            v2_mv_loc = (time.perf_counter() - t0_mv) * 1000.0 - 500.0
        if rank == 11:
            v3_mv_loc = (time.perf_counter() - t0_mv) * 1000.0 - 500.0
        wan.release({"state_id": "mv16-v23"})
        torch.cuda.empty_cache()

    sync_all(device_str)
    v0_mv_m, v1_mv_m, v2_mv_m, v3_mv_m = collect_max(device_str, v0_mv_loc, v1_mv_loc, v2_mv_loc, v3_mv_loc)
    four_video_contention = {
        "workload": "4x Concurrent Videos (v0, v1 at t=0ms; v2, v3 at t=500ms) on 2x8 16-GPU Cluster + 4 Dedicated T2I Absorber GPUs (Ranks 6,7,14,15)",
        "static_partition_8v8i_b3_opt": {
            "video0_1_latency_ms": round(v01_b3o_ms, 1),
            "video2_3_latency_ms": round(v23_b3o_ms, 1),
            "mean_4video_latency_ms": round(0.5 * (v01_b3o_ms + v23_b3o_ms), 1),
        },
        "ours2_heterodit_morphing_v5_16gpu": {
            "video0_latency_ms": round(v0_mv_m, 1),
            "video1_latency_ms": round(v1_mv_m, 1),
            "video2_latency_ms": round(v2_mv_m, 1),
            "video3_latency_ms": round(v3_mv_m, 1),
            "mean_4video_latency_ms": round(0.25 * (v0_mv_m + v1_mv_m + v2_mv_m + v3_mv_m), 1),
            "video2_3_speedup_vs_static_8v8i": round(v23_b3o_ms / max(0.5 * (v2_mv_m + v3_mv_m), 1.0), 2),
            "mean_4video_speedup_vs_static_8v8i": round(
                (0.5 * (v01_b3o_ms + v23_b3o_ms)) / max(0.25 * (v0_mv_m + v1_mv_m + v2_mv_m + v3_mv_m), 1.0), 2
            ),
        },
    }
    if rank == 0:
        print(f"[Part 3: 16-GPU 4-Video Contention] {json.dumps(four_video_contention, indent=2)}", flush=True)

    if rank == 0:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "suite": f"HETERODIT_16GPU_2X8_COMPARISON_{args.wan_model.upper()}",
            "wan_model": args.wan_model,
            "wan_path": wan_path,
            "world_size": world,
            "hosts": hosts,
            "topology": "2x8_cross_node_16gpu",
            "scaling_matrix_16gpu": scaling_rows,
            "mixed_serving_16gpu_12_policies": policies_16gpu,
            "four_video_contention_16gpu": four_video_contention,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[SUCCESS] Wrote {out_path}", flush=True)

    sync_all(device_str)
    wan.close()
    sd35.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
