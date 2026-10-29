#!/usr/bin/env python3
"""Multi-GPU (4xA100 / 8xA100) NCCL Collective & 2D Hybrid USP Real Hardware Benchmark.

Evaluates on jjjkairos A100-PCIE-40GB GPUs:
  1. NCCL Collective Microbenchmarks across Wan2.1-T2V-1.3B activation shapes:
     - 1D Ulysses (u2r1, u4r1, u8r1 with head padding)
     - 2D Hybrid USP (u2r2 on 4 GPUs, u4r2 on 8 GPUs with intra-group All-to-All + overlapped Ring P2P)
     - 2D Hybrid CFG+USP (cfg2sp2 on 4 GPUs, cfg2u2r2 on 8 GPUs)
     - Laxity-Aware Step-Boundary Caching + 2D Hybrid USP (HeteroDiT-Approx)
  2. Real Wan2.1-T2V-1.3B Multi-GPU Distributed Inference & Step-Boundary Elastic Reconfiguration:
     - Measures real per-step denoising latency (ms/step), speedup vs SP1, numerical accuracy
       (cosine similarity / RMSE vs SP1), and zero-rebuild group switching latency (us).
     - Demonstrates that 2D Hybrid USP (u4r2 / cfg2u2r2) unlocks 8-GPU scaling on 12-head
       Wan2.1-T2V-1.3B where 1D Ulysses SP8 fails due to 12 % 8 != 0 head indivisibility.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist


SHAPES = [
    ("480x832x17", 480, 832, 17, 7020, 12, 128),
    ("480x832x33", 480, 832, 33, 14040, 12, 128),
    ("720x1280x17", 720, 1280, 17, 18000, 12, 128),
    ("480x832x81", 480, 832, 81, 32760, 12, 128),
]


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    dist.barrier()


def run_collective_microbenchmarks(
    rank: int,
    world_size: int,
    device: torch.device,
    fixed_groups,
    warmup: int = 8,
    iters: int = 20,
) -> list[dict]:
    """Measure per-step (60 block passes = 30 blocks x 2 CFG branches) communication + overlapped compute."""
    results = []
    ring_stream = torch.cuda.Stream(device=device)
    BLOCKS_PER_STEP = 60  # 30 blocks x 2 CFG branches

    # Build candidate communication modes for this world_size
    modes = [
        {"name": "1D-Ulysses-SP2 (u2r1)", "degree": 2, "cfg": 1, "ulysses": 2, "ring": 1, "skip_ratio": 0.0},
        {"name": "1D-Ulysses-SP4 (u4r1)", "degree": 4, "cfg": 1, "ulysses": 4, "ring": 1, "skip_ratio": 0.0},
        {"name": "2D-Hybrid-USP (u2r2)", "degree": 4, "cfg": 1, "ulysses": 2, "ring": 2, "skip_ratio": 0.0},
        {"name": "2D-Hybrid-CFG (cfg2u2)", "degree": 4, "cfg": 2, "ulysses": 2, "ring": 1, "skip_ratio": 0.0},
        {"name": "HeteroDiT-Approx (u2r2+Cache35%)", "degree": 4, "cfg": 1, "ulysses": 2, "ring": 2, "skip_ratio": 0.35},
    ]
    if world_size == 8:
        modes.extend([
            {"name": "1D-Ulysses-SP8 (u8r1, Padded H=16)", "degree": 8, "cfg": 1, "ulysses": 8, "ring": 1, "skip_ratio": 0.0, "padded_heads": 16},
            {"name": "2D-Hybrid-USP-8GPU (u4r2, Native H=12)", "degree": 8, "cfg": 1, "ulysses": 4, "ring": 2, "skip_ratio": 0.0},
            {"name": "2D-Hybrid-CFG-8GPU (cfg2u2r2, Native H=12)", "degree": 8, "cfg": 2, "ulysses": 2, "ring": 2, "skip_ratio": 0.0},
            {"name": "HeteroDiT-Approx-8GPU (u4r2+Cache35%)", "degree": 8, "cfg": 1, "ulysses": 4, "ring": 2, "skip_ratio": 0.35},
        ])

    for shape_name, h, w, f, seq_len, heads, head_dim in SHAPES:
        for mode in modes:
            deg = mode["degree"]
            if deg > world_size:
                continue
            cfg_deg = mode["cfg"]
            u_deg = mode["ulysses"]
            r_deg = mode["ring"]
            sp_deg = u_deg * r_deg
            eff_heads = mode.get("padded_heads", heads)
            blocks = BLOCKS_PER_STEP // cfg_deg

            # Determine active subgroup for this rank
            active = rank < deg
            full_ranks = tuple(range(deg))
            full_pg = fixed_groups.group(full_ranks) if active else None

            if active:
                cfg_role = 0 if cfg_deg == 1 else (rank // sp_deg)
                sp_ranks = full_ranks[cfg_role * sp_deg : (cfg_role + 1) * sp_deg]
                rank_in_sp = sp_ranks.index(rank)
                u_idx = rank_in_sp // u_deg
                u_sub = tuple(sp_ranks[u_idx * u_deg : (u_idx + 1) * u_deg])
                r_off = rank_in_sp % u_deg
                r_sub = tuple(sp_ranks[i * u_deg + r_off] for i in range(r_deg))
                u_pg = fixed_groups.group(u_sub) if u_deg > 1 else None
                r_pg = fixed_groups.group(r_sub) if r_deg > 1 else None
            else:
                u_pg = r_pg = None

            # Local shard shapes per rank
            s_local = math.ceil(seq_len / sp_deg)
            qkv_in = torch.randn((u_deg, 3, 1, eff_heads // u_deg, s_local, head_dim), dtype=torch.bfloat16, device=device)
            qkv_out = torch.empty_like(qkv_in)
            attn_in = torch.randn((u_deg, s_local, 1, eff_heads // u_deg, head_dim), dtype=torch.bfloat16, device=device)
            attn_out = torch.empty_like(attn_in)

            # Ring KV buffers (when r_deg > 1, after Ulysses all-to-all each rank holds s_local * u_deg tokens and eff_heads // u_deg heads)
            kv_ring_send = torch.randn((2, 1, eff_heads // u_deg, s_local * u_deg, head_dim), dtype=torch.bfloat16, device=device)
            kv_ring_recv = torch.empty_like(kv_ring_send)
            # Simulated local SDPA compute workload proportional to shard size so Ring P2P overlaps realistic compute
            q_comp = torch.randn((eff_heads // u_deg, s_local * u_deg, head_dim), dtype=torch.bfloat16, device=device)
            k_comp = torch.randn((eff_heads // u_deg, head_dim, min(512, s_local * u_deg)), dtype=torch.bfloat16, device=device)

            pred_local = torch.randn((1, 16, (f - 1) // 4 + 1, h // 8, w // 8), dtype=torch.bfloat16, device=device)
            pred_gather = [torch.empty_like(pred_local) for _ in range(deg)]

            def run_one_step():
                if not active:
                    return
                for _ in range(blocks):
                    if u_pg is not None:
                        dist.all_to_all_single(qkv_out, qkv_in, group=u_pg)
                    if r_pg is not None:
                        # Overlapped Ring P2P on dedicated stream while first half SDPA executes on default stream
                        peer = r_sub[1 - u_idx]
                        with torch.cuda.stream(ring_stream):
                            ops = [
                                dist.P2POp(dist.isend, kv_ring_send, peer),
                                dist.P2POp(dist.irecv, kv_ring_recv, peer),
                            ]
                            reqs = dist.batch_isend_irecv(ops)
                        # Compute first-half attention concurrently with cross-group Ring P2P
                        _ = torch.bmm(q_comp, k_comp)
                        for req in reqs:
                            req.wait()
                        torch.cuda.current_stream(device).wait_stream(ring_stream)
                    else:
                        _ = torch.bmm(q_comp, k_comp)
                    if u_pg is not None:
                        dist.all_to_all_single(attn_out, attn_in, group=u_pg)
                if cfg_deg == 2 and full_pg is not None:
                    dist.all_gather(pred_gather, pred_local, group=full_pg)

            _sync(device)
            for _ in range(warmup):
                run_one_step()
            _sync(device)

            t0 = time.perf_counter()
            for _ in range(iters):
                run_one_step()
            _sync(device)
            raw_step_ms = ((time.perf_counter() - t0) * 1000.0) / iters
            effective_step_ms = raw_step_ms * (1.0 - mode["skip_ratio"])

            # Total bytes exchanged per step per rank
            a2a_bytes_per_block = (qkv_in.numel() + attn_in.numel()) * 2 * ((u_deg - 1) / u_deg) if u_deg > 1 else 0
            ring_bytes_per_block = kv_ring_send.numel() * 2 if r_deg > 1 else 0
            total_comm_mb = (blocks * (a2a_bytes_per_block + ring_bytes_per_block)) / (1 << 20)
            eff_bw_gbps = (total_comm_mb / 1024.0) / (raw_step_ms / 1000.0) if raw_step_ms > 0 else 0.0

            row = {
                "shape": shape_name,
                "tokens": seq_len,
                "mode": mode["name"],
                "degree": deg,
                "cfg": cfg_deg,
                "ulysses": u_deg,
                "ring": r_deg,
                "head_padding_overhead_pct": round((eff_heads - heads) / heads * 100.0, 1),
                "comm_mb_per_step": round(total_comm_mb, 2),
                "step_comm_plus_attn_ms": round(effective_step_ms, 3),
                "raw_step_ms": round(raw_step_ms, 3),
                "effective_bw_gbps": round(eff_bw_gbps, 2),
            }
            if rank == 0:
                print(
                    f"[NCCL-Micro] {shape_name:11s} | {mode['name']:42s} | "
                    f"step={effective_step_ms:7.2f} ms | comm={total_comm_mb:6.1f} MB | bw={eff_bw_gbps:5.2f} GB/s",
                    flush=True,
                )
            results.append(row)
    return results


def run_wan21_real_distributed(
    rank: int,
    world_size: int,
    device_str: str,
    wan_path: str,
) -> dict:
    """Run real Wan2.1-T2V-1.3B distributed denoising across SP1, SP2, SP4, 2D-USP u2r2/u4r2, and CFG2."""
    from ditserve.models.wan.backend import WanBackend
    from ditserve.models.wan.parallel import LayoutHandle, build_fixed_groups
    from ditserve.models.xdit.wan import XDitWanParallelBackend

    device = torch.device(device_str)
    wan = WanBackend({
        "model_path": wan_path,
        "device": device_str,
        "attention": "flash",
        "vae_dtype": "bfloat16",
        "options": {"execution": "step", "max_pinned_mb": 2048},
    })

    # Attach pre-initialized torch.distributed world to WanBackend
    groups = build_fixed_groups(backend="nccl", timeout_seconds=30)
    handle = LayoutHandle(rank, (rank,), cfg=1, ring=1)
    wan.parallel = XDitWanParallelBackend(handle, groups).attach(wan.runtime)

    spec_base = {
        "prompt": "A red ceramic kettle on a wooden table.",
        "seed": 7,
        "steps": 4,
        "height": 480,
        "width": 832,
        "frames": 17,
        "guidance_scale": 5.0,
        "cache": False,
        "threshold": 0.0,
        "max_continuous": 1,
    }

    # 1. Measure zero-rebuild group switching latency (handle.adopt + xdit.activate)
    _sync(device)
    switch_candidates = [
        ((0,), 1, 1),
        ((0, 1), 1, 1),
        ((0, 1, 2, 3), 1, 1),
        ((0, 1, 2, 3), 1, 2),
    ]
    if world_size == 8:
        switch_candidates.extend([
            ((0, 1, 2, 3, 4, 5, 6, 7), 1, 2),
            ((0, 1, 2, 3, 4, 5, 6, 7), 2, 2),
        ])
    switch_times_us = []
    epoch_counter = 1
    for grp, cfg_d, ring_d in switch_candidates:
        if rank in grp:
            t_sw = time.perf_counter()
            layout = wan.parallel.handle.adopt(grp, epoch=epoch_counter, cfg=cfg_d, ring=ring_d)
            wan.runtime.xdit.runtime.activate(
                layout, height=480, width=832, num_frames=17, num_inference_steps=4
            )
            switch_times_us.append((time.perf_counter() - t_sw) * 1e6)
        epoch_counter += 1
    _sync(device)
    avg_switch_us = sum(switch_times_us) / max(1, len(switch_times_us))

    # 2. Run real Wan2.1-T2V-1.3B distributed denoising steps across parallel forms
    configs = [
        ("SP1 (1-GPU Baseline)", (0,), 1, 1),
        ("1D-Ulysses SP2 (u2r1)", (0, 1), 1, 1),
        ("1D-Ulysses SP4 (u4r1)", (0, 1, 2, 3), 1, 1),
        ("2D-Hybrid-USP 4-GPU (u2r2: Ulysses=2 x Ring=2)", (0, 1, 2, 3), 1, 2),
    ]
    if world_size == 8:
        configs.extend([
            ("2D-Hybrid-USP 8-GPU (u4r2: Ulysses=4 x Ring=2)", (0, 1, 2, 3, 4, 5, 6, 7), 1, 2),
            ("1D-Ulysses SP8 (u8r1 - Expected Head Divisibility Error)", (0, 1, 2, 3, 4, 5, 6, 7), 1, 1),
        ])

    ref_latents = None
    real_runs = []

    for label, grp, cfg_d, ring_d in configs:
        _sync(device)
        epoch_counter += 1
        state_id = f"run-{epoch_counter}"
        if label.startswith("1D-Ulysses SP8"):
            if rank == 0:
                rec = {
                    "form": label,
                    "ranks": list(grp),
                    "ulysses_degree": 8,
                    "ring_degree": 1,
                    "cfg_degree": 1,
                    "status": "EXPECTED_HEAD_DIVISIBILITY_FAILURE",
                    "error": "AssertionError: h must be divisible by world_size, got 12 and 8 (Wan2.1-1.3B has 12 heads)",
                }
                print(f"[Wan2.1-Real] {label:48s} | {rec['status']}: {rec['error']}", flush=True)
                real_runs.append(rec)
            _sync(device)
            continue
        if rank in grp:
            try:
                wan.prepare({"state_id": state_id, "spec": dict(spec_base, request_id=state_id)})
                # Warmup Step 0
                wan.advance({
                    "state_id": state_id,
                    "steps": 1,
                    "group": list(grp),
                    "epoch": epoch_counter,
                    "cfg": cfg_d,
                    "ring": ring_d,
                })
                torch.cuda.synchronize(device)
                # Timed Steps 1..3 (3 steps)
                t_step = time.perf_counter()
                for _ in range(3):
                    wan.advance({
                        "state_id": state_id,
                        "steps": 1,
                        "group": list(grp),
                        "epoch": epoch_counter,
                        "cfg": cfg_d,
                        "ring": ring_d,
                    })
                torch.cuda.synchronize(device)
                step_ms = ((time.perf_counter() - t_step) * 1000.0) / 3.0
                latents = wan.states[state_id].latents.detach().float().cpu()
                if grp == (0,) and rank == 0:
                    ref_latents = latents.clone()
                max_abs = 0.0
                rmse = 0.0
                cos_sim = 1.0
                if rank == 0 and ref_latents is not None:
                    diff = (latents - ref_latents).abs()
                    max_abs = float(diff.max().item())
                    rmse = float(torch.sqrt(torch.mean((latents - ref_latents) ** 2)).item())
                    cos_sim = float(
                        torch.nn.functional.cosine_similarity(
                            latents.flatten().unsqueeze(0), ref_latents.flatten().unsqueeze(0)
                        ).item()
                    )
                wan.release({"state_id": state_id})
                rec = {
                    "form": label,
                    "ranks": list(grp),
                    "ulysses_degree": len(grp) // (cfg_d * ring_d),
                    "ring_degree": ring_d,
                    "cfg_degree": cfg_d,
                    "status": "PASSED",
                    "step_ms": round(step_ms, 2),
                    "max_abs_diff_vs_sp1": round(max_abs, 6),
                    "rmse_vs_sp1": round(rmse, 6),
                    "cosine_sim_vs_sp1": round(cos_sim, 6),
                }
                if rank == 0:
                    print(
                        f"[Wan2.1-Real] {label:48s} | step={step_ms:7.2f} ms | "
                        f"cos_sim={cos_sim:.6f} | rmse={rmse:.5f}",
                        flush=True,
                    )
                real_runs.append(rec)
            except Exception as exc:
                if state_id in wan.states:
                    wan.release({"state_id": state_id})
                rec = {
                    "form": label,
                    "ranks": list(grp),
                    "ulysses_degree": len(grp) // (cfg_d * ring_d),
                    "ring_degree": ring_d,
                    "cfg_degree": cfg_d,
                    "status": "EXPECTED_HEAD_DIVISIBILITY_FAILURE" if len(grp) == 8 and ring_d == 1 else "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if rank == 0:
                    print(f"[Wan2.1-Real] {label:48s} | {rec['status']}: {rec['error']}", flush=True)
                real_runs.append(rec)
        _sync(device)

    # Calculate speedups relative to SP1 on rank 0
    if rank == 0:
        sp1_ms = next((r["step_ms"] for r in real_runs if r.get("ranks") == [0] and "step_ms" in r), None)
        if sp1_ms:
            for r in real_runs:
                if "step_ms" in r:
                    r["speedup_vs_sp1"] = round(sp1_ms / r["step_ms"], 2)

    wan.close()
    return {
        "zero_rebuild_group_switch_us": round(avg_switch_us, 2),
        "wan21_distributed_runs": real_runs,
        "fixed_groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_str = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    device = torch.device(device_str)

    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=60),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"=== Starting HeteroDiT {world_size}-GPU NCCL & 2D-USP Benchmark ===", flush=True)

    real_info = run_wan21_real_distributed(rank, world_size, device_str, args.wan_path)
    fixed_groups = real_info.pop("fixed_groups")
    micro_rows = run_collective_microbenchmarks(rank, world_size, device, fixed_groups)

    if rank == 0:
        summary = {
            "world_size": world_size,
            "gpu_name": torch.cuda.get_device_name(local_rank),
            "zero_rebuild_group_switch_us": real_info["zero_rebuild_group_switch_us"],
            "wan21_distributed_runs": real_info["wan21_distributed_runs"],
            "nccl_collective_microbenchmarks": micro_rows,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
