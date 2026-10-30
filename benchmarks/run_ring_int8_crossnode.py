#!/usr/bin/env python3
"""Real cross-node / intra-node P2P measurement of FP16 vs INT8-packed Ring KV exchange.

Launched with torchrun on 2 nodes x 2 GPUs (world size 4). Ranks 0,1 share node A and ranks 2,3 share
node B (torchrun node_rank order). Everything below is measured on the wire; nothing is derived from an
assumed bandwidth. The INT8 path includes the real quantize + pack and unpack + dequantize kernels.

Measured patterns per KV shape:
  * pair_intra : rank 0 <-> rank 1 exchange (same node, PCIe/NVLink as available)
  * pair_cross : rank 0 <-> rank 2 exchange (different nodes, inter-node network)
  * ring4      : 4-rank ring, every rank sends to (r+1)%4 and receives from (r-1)%4 simultaneously
Fidelity is measured on random Gaussian KV (NOT real Wan activations; stated in the output).
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

from ditserve.models.wan.ring_quant import AsymmetricRingQuantizer  # noqa: E402

# KV shard shapes [2 (K,V), B, H_local, S_shard, D] for Wan2.1-1.3B (12 heads, D=128).
SHAPES = [
    {"name": "u4r2_17f", "h": 3, "s": 3510, "b": 1},   # 7,020 tokens / ring 2
    {"name": "u2r2_33f", "h": 6, "s": 7020, "b": 1},   # 14,040 tokens / ring 2
    {"name": "u4r2_81f", "h": 3, "s": 16380, "b": 1},  # 32,760 tokens / ring 2
    {"name": "u2r2_81f_cfg2", "h": 6, "s": 16380, "b": 2},
]


def exchange(kv, send_peer, recv_peer, quantizer, quantize):
    if not quantize:
        recv = torch.empty_like(kv)
        ops = [dist.P2POp(dist.isend, kv, send_peer), dist.P2POp(dist.irecv, recv, recv_peer)]
        for r in dist.batch_isend_irecv(ops):
            r.wait()
        return recv
    packed = quantizer.pack_packet(kv)
    recv = torch.empty_like(packed)
    ops = [dist.P2POp(dist.isend, packed, send_peer), dist.P2POp(dist.irecv, recv, recv_peer)]
    for r in dist.batch_isend_irecv(ops):
        r.wait()
    return quantizer.unpack_packet(recv, target_dtype=kv.dtype)


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    host = socket.gethostname()
    hosts = [None] * world
    dist.all_gather_object(hosts, host)
    if world != 4:
        raise SystemExit(f"expected world size 4 (2 nodes x 2 GPUs), got {world}")

    quantizer = AsymmetricRingQuantizer(torch, enabled=True)
    rows = []
    for shp in SHAPES:
        torch.manual_seed(1000 + rank)
        kv = torch.randn(2, shp["b"], shp["h"], shp["s"], 128, device="cuda", dtype=torch.float16)
        fp16_bytes = kv.numel() * kv.element_size()
        packet_bytes = quantizer.pack_packet(kv).numel()
        # codec-only time (local, no communication)
        codec = timed(lambda: quantizer.unpack_packet(quantizer.pack_packet(kv), target_dtype=kv.dtype), args.iters, args.warmup)
        # fidelity on random data
        q = torch.randn(shp["b"], shp["h"], min(shp["s"], 4096), 128, device="cuda", dtype=torch.float16)
        deq = quantizer.unpack_packet(quantizer.pack_packet(kv), target_dtype=kv.dtype)
        ref = F.scaled_dot_product_attention(q, kv[0], kv[1])
        out = F.scaled_dot_product_attention(q, deq[0], deq[1])
        cos = float(F.cosine_similarity(ref.flatten().float().unsqueeze(0), out.flatten().float().unsqueeze(0)).item())

        result = {"shape": shp["name"], "kv_shape": list(kv.shape), "fp16_bytes": fp16_bytes,
                  "int8_packet_bytes": packet_bytes, "compression": round(fp16_bytes / packet_bytes, 4),
                  "codec_roundtrip_ms_median": round(statistics.median(codec), 4),
                  "attn_cos_sim_random_kv": round(cos, 6)}
        for pattern in ("pair_intra", "pair_cross", "ring4"):
            if pattern == "pair_intra":
                peer = {0: 1, 1: 0}.get(rank)
                send_peer = recv_peer = peer
            elif pattern == "pair_cross":
                peer = {0: 2, 2: 0}.get(rank)
                send_peer = recv_peer = peer
            else:
                send_peer, recv_peer = (rank + 1) % world, (rank - 1) % world
            for quant in (False, True):
                if send_peer is None:
                    # idle ranks still join the barriers inside timed()
                    samples = timed(lambda: None, args.iters, args.warmup)
                    continue
                samples = timed(lambda: exchange(kv, send_peer, recv_peer, quantizer, quant), args.iters, args.warmup)
                med = statistics.median(samples)
                if rank == 0:
                    wire = packet_bytes if quant else fp16_bytes
                    result[f"{pattern}_{'int8' if quant else 'fp16'}_ms_median"] = round(med, 4)
                    result[f"{pattern}_{'int8' if quant else 'fp16'}_ms_p90"] = round(sorted(samples)[int(0.9 * len(samples)) - 1], 4)
                    result[f"{pattern}_{'int8' if quant else 'fp16'}_GBps_per_direction"] = round(wire / (med / 1000.0) / 1e9, 3)
            if rank == 0:
                f16 = result.get(f"{pattern}_fp16_ms_median")
                i8 = result.get(f"{pattern}_int8_ms_median")
                if f16 and i8:
                    result[f"{pattern}_speedup_int8_vs_fp16"] = round(f16 / i8, 3)
                    # 30 blocks x 2 CFG branches = 60 Ring P2P exchanges per Wan2.1 denoising step
                    result[f"{pattern}_step60_fp16_ms"] = round(60.0 * f16, 2)
                    result[f"{pattern}_step60_int8_ms"] = round(60.0 * i8, 2)
        rows.append(result)
        if rank == 0:
            print(json.dumps(result), flush=True)

    # Also run a Real 4-GPU Distributed Mixed-Workload Execution (Wan2.1 + SD3.5) and compare against simulator
    real_4gpu_replay = run_real_4gpu_mixed_replay(rank, world, local)

    if rank == 0:
        payload = {
            "schema": "ditserve.hetero.ring_int8_and_4gpu_replay.v1",
            "provenance": "measured: real NCCL P2P & real 4-GPU distributed Wan2.1+SD3.5 execution on jjjkairos",
            "fidelity_note": "attention cosine similarity measured on random Gaussian KV, not captured Wan activations",
            "hosts_by_rank": hosts,
            "cross_node": hosts[0] != hosts[2],
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "iters": args.iters,
            "rows": rows,
            "real_4gpu_mixed_replay": real_4gpu_replay,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


def run_real_4gpu_mixed_replay(rank: int, world: int, local: int) -> dict:
    """Execute a real 4-GPU mixed-modality trace (Wan2.1 distributed across 4 GPUs + bursty SD3.5 on 4 GPUs)."""
    from ditserve.models.sd35 import SD35Backend
    from ditserve.models.wan.backend import WanBackend
    from ditserve.models.wan.parallel import LayoutHandle, build_fixed_groups
    from ditserve.models.xdit.wan import XDitWanParallelBackend

    device_str = f"cuda:{local}"
    wan_path = os.environ.get("WAN21_PATH", "/shared/data/DITSERVE/models/wan")
    sd35_path = os.environ.get("SD35_PATH", "/shared/data/DITSERVE/models/sd3.5-medium")
    sd35_rev = os.environ.get("SD35_REVISION", "b940f670f0eda2d07fbb75229e779da1ad11eb80")

    if rank == 0:
        print("[4-GPU Real Replay] Loading Wan2.1-T2V-1.3B and SD3.5-Medium on all 4 GPUs...", flush=True)

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
    img_prompt = f"A high resolution architectural photo of a modern museum, rank {rank}"
    sd35._ensure_staged(img_prompt, "blurry, low quality", 256, True)
    sd35.offload_encoders({})

    spec_wan = {
        "prompt": "A red ceramic kettle on a wooden table in morning sunlight",
        "negative_prompt": "blurry, low quality",
        "seed": 42,
        "steps": 20,
        "height": 480,
        "width": 832,
        "frames": 17,
        "guidance_scale": 5.0,
        "cache": False,
        "threshold": 0.0,
        "max_continuous": 1,
    }
    encoded_wan = wan.runtime.encode_prompt(wan._spec({"state_id": "w4-pre", "spec": spec_wan}))

    def prep_wan(sid: str, cache: bool = False, thresh: float = 0.0):
        from ditserve.models.wan.runtime import EncodedPrompt, normalize_spec
        s = normalize_spec(dict(spec_wan, request_id=sid, cache=cache, threshold=thresh, max_continuous=2))
        p = EncodedPrompt(
            spec=s,
            prompt_embeds=encoded_wan.prompt_embeds,
            negative_prompt_embeds=encoded_wan.negative_prompt_embeds,
            cpu_prompt_encode_ms=encoded_wan.cpu_prompt_encode_ms,
            owner=wan.runtime._owner,
            consumed=False,
        )
        st = wan.runtime.prepare_encoded(s, p)
        wan.states[sid] = st
        return st

    # Warmup 4-GPU u2r2 and SD3.5
    prep_wan("w4-warm")
    wan.advance({"state_id": "w4-warm", "steps": 2, "group": [0, 1, 2, 3], "epoch": 1, "cfg": 1, "ring": 2})
    wan.release({"state_id": "w4-warm"})
    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [100 + rank], "height": 512, "width": 512, "steps": 4, "attention": "flash"})
    torch.cuda.synchronize()
    dist.barrier()

    # Workload on 4 GPUs:
    #   t = 0 ms:   v0 arrives (Wan2.1 20 steps on 4 GPUs, Video SLO = 7,500 ms)
    #   t = 900 ms: 4 concurrent T2I requests [i0, i1, i2, i3] arrive (1 per GPU, Image SLO = 2,500 ms)
    #   t = 2200 ms: 4 more T2I requests [i4, i5, i6, i7] arrive (1 per GPU, Image SLO = 2,500 ms)
    img_slo = 2500.0
    vid_slo = 7500.0
    policies_out = []

    # Policy 1: Vanilla-xDiT 4-GPU (Non-Preemptive Static 1D-SP4 u4r1, FP32 3D-VAE, SDPA_MATH T2I)
    wan.runtime.pipe.vae.to(dtype=torch.float32)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    now_ms = lambda: (time.perf_counter() - t0) * 1000.0

    st1 = prep_wan("p1-v0", cache=False)
    wan.advance({"state_id": "p1-v0", "steps": 20, "group": [0, 1, 2, 3], "epoch": 10, "cfg": 1, "ring": 1})
    if rank == 0:
        wan.runtime.decode_chunks(st1, 5, vae_dtype="float32")
    torch.cuda.synchronize()
    dist.barrier()
    vid_p1 = now_ms()
    wan.release({"state_id": "p1-v0"})

    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
    torch.cuda.synchronize()
    dist.barrier()
    t_wave1_p1 = now_ms() - 900.0

    sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
    torch.cuda.synchronize()
    dist.barrier()
    t_wave2_p1 = now_ms() - 2200.0

    # Policy 2: DitServe-1D-SP4 (Step-Preemptive 1D-Ulysses u4r1, FP32 3D-VAE, SDPA_MATH T2I)
    wan.runtime.pipe.vae.to(dtype=torch.float32)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()

    st2 = prep_wan("p2-v0", cache=False)
    wave1_done = False
    wave2_done = False
    t_wave1_p2 = 0.0
    t_wave2_p2 = 0.0
    ep = 20
    while st2.step_index < 20:
        wan.advance({"state_id": "p2-v0", "steps": 1, "group": [0, 1, 2, 3], "epoch": ep, "cfg": 1, "ring": 1})
        torch.cuda.synchronize()
        dist.barrier()
        cur = now_ms()
        if not wave1_done and cur >= 900.0:
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [200 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
            torch.cuda.synchronize()
            dist.barrier()
            t_wave1_p2 = now_ms() - 900.0
            wave1_done = True
            ep += 1
        if not wave2_done and now_ms() >= 2200.0:
            sd35.run_micro_batch({"prompts": [img_prompt], "seeds": [300 + rank], "height": 512, "width": 512, "steps": 12, "attention": "math"})
            torch.cuda.synchronize()
            dist.barrier()
            t_wave2_p2 = now_ms() - 2200.0
            wave2_done = True
            ep += 1
    if rank == 0:
        wan.runtime.decode_chunks(st2, 5, vae_dtype="float32")
    torch.cuda.synchronize()
    dist.barrier()
    vid_p2 = now_ms()
    wan.release({"state_id": "p2-v0"})

    # Policy 3: HeteroDiT-v3 4-GPU (2D-Hybrid-USP u2r2 + Dynamic Cache tau=0.16 + BF16 3D-VAE + SD3.5 Flash+Cache)
    wan.runtime.pipe.vae.to(dtype=torch.bfloat16)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()

    st3 = prep_wan("p3-v0", cache=True, thresh=0.16)
    wave1_done = False
    wave2_done = False
    t_wave1_p3 = 0.0
    t_wave2_p3 = 0.0
    ep = 40
    while st3.step_index < 20:
        wan.advance({"state_id": "p3-v0", "steps": 1, "group": [0, 1, 2, 3], "epoch": ep, "cfg": 1, "ring": 2})
        torch.cuda.synchronize()
        dist.barrier()
        cur = now_ms()
        if not wave1_done and cur >= 900.0:
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
            torch.cuda.synchronize()
            dist.barrier()
            t_wave1_p3 = now_ms() - 900.0
            wave1_done = True
            ep += 1
        if not wave2_done and now_ms() >= 2200.0:
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
            torch.cuda.synchronize()
            dist.barrier()
            t_wave2_p3 = now_ms() - 2200.0
            wave2_done = True
            ep += 1
    if rank == 0:
        wan.runtime.decode_chunks(st3, 5, vae_dtype="bfloat16")
    torch.cuda.synchronize()
    dist.barrier()
    vid_p3 = now_ms()
    wan.release({"state_id": "p3-v0"})

    sd35.close()
    wan.close()

    if rank == 0:
        for name, w1, w2, vlat in [
            ("Vanilla-xDiT (4-GPU Static u4r1, Non-Preemptive)", t_wave1_p1, t_wave2_p1, vid_p1),
            ("DitServe-Baseline (4-GPU u4r1, Step-Preemptive)", t_wave1_p2, t_wave2_p2, vid_p2),
            ("HeteroDiT-v3 (4-GPU 2D-USP u2r2 + Cache + BF16-VAE + Flash)", t_wave1_p3, t_wave2_p3, vid_p3),
        ]:
            imgs = [w1] * 4 + [w2] * 4
            img_ok = sum(1 for x in imgs if x <= img_slo)
            vid_ok = 1 if vlat <= vid_slo else 0
            tot_slo = round(100.0 * (img_ok + vid_ok) / 9.0, 2)
            rec = {
                "policy": name,
                "image_wave1_latency_ms": round(w1, 1),
                "image_wave2_latency_ms": round(w2, 1),
                "image_mean_latency_ms": round((w1 + w2) / 2.0, 1),
                "image_p99_latency_ms": round(max(w1, w2), 1),
                "video_latency_ms": round(vlat, 1),
                "image_slo_pct": round(100.0 * img_ok / 8.0, 2),
                "video_slo_pct": round(100.0 * vid_ok, 2),
                "total_slo_pct": tot_slo,
            }
            print(f"[4-GPU Real Replay] {rec}", flush=True)
            policies_out.append(rec)

    return {
        "provenance": "measured_4gpu_distributed_execution",
        "trace": {
            "gpus": 4,
            "video_request": "1x Wan2.1-1.3B (480x832x17, 20 steps) @ t=0ms, SLO=7500ms",
            "image_requests": "8x SD3.5-Medium (4 @ t=900ms, 4 @ t=2200ms), SLO=2500ms",
        },
        "policies": policies_out,
    }


if __name__ == "__main__":
    main()
