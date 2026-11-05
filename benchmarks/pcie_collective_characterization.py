#!/usr/bin/env python3
"""PCIe collective communication wall characterization and step-latency decomposition.

Supports two execution modes:
1. `decompose`: Deterministic CPU analysis joining verified 8-GPU calibration
   bundles (`ditserve.shape-profiles.v1`) with optional hardware microbenchmark
   measurements to decompose every legal Form(c, d) across Wan2.1, Wan2.2, and
   SD3.5-Medium into ideal compute, All-to-All volume, collective count,
   effective PCIe bandwidth, parallel efficiency, and dominance relations.
2. `run`: Single-node multi-GPU (`torchrun --nproc_per_node {2,4,8}`) hardware
   microbenchmark measuring:
   - GPU UUIDs, PCIe Gen/Width, NUMA mapping, and `nvidia-smi topo -m`;
   - Hockney latency floor (4 KB) and asymptotic All-to-All bandwidth across
     group sizes d in {2, 3, 4, 6, 8};
   - Step-equivalent collective replay comparing single-group Pure SP
     (240 collectives/step for Wan; 192 for SD3.5) against concurrent dual-subgroup
     Asymmetric CFG2 x SP (120 collectives/step per subgroup + 1 FP16 latent exchange).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any


SCHEMA = "ditserve.pcie-collective-characterization.v1"
STORAGE_MAX_GIB = 800
STORAGE_MAX_PCT = 80

# Exact architectural parameters matching fit_native_profile.py and 02-background.tex.
SHAPES: dict[str, dict[str, Any]] = {
    "wan21-dev": {
        "model": "wan21",
        "kind": "video",
        "scope": "development",
        "frames": 33,
        "height": 480,
        "width": 832,
        "steps": 20,
        "tokens": 14040,
        "dim": 1536,
        "heads": 12,
        "layers": 30,
        "comm_mb_per_step": 10351.4112,
        "collectives_per_step": 240,
        "cfg_exchange_mb": 1.79712,
        "legal_sp": (1, 2, 3, 4, 6),
    },
    "wan22-dev": {
        "model": "wan22",
        "kind": "video",
        "scope": "development",
        "frames": 49,
        "height": 512,
        "width": 896,
        "steps": 20,
        "tokens": 5824,
        "dim": 3072,
        "heads": 24,
        "layers": 30,
        "comm_mb_per_step": 8587.823104,
        "collectives_per_step": 240,
        "cfg_exchange_mb": 2.236416,
        "legal_sp": (1, 2, 3, 4, 6, 8),
    },
    "wan21-480p": {
        "model": "wan21",
        "kind": "video",
        "scope": "formal",
        "frames": 81,
        "height": 480,
        "width": 832,
        "steps": 50,
        "tokens": 32760,
        "dim": 1536,
        "heads": 12,
        "layers": 30,
        "comm_mb_per_step": 24153.2928,
        "collectives_per_step": 240,
        "cfg_exchange_mb": 4.19328,
        "legal_sp": (1, 2, 3, 4, 6),
    },
    "wan21-720p": {
        "model": "wan21",
        "kind": "video",
        "scope": "formal",
        "frames": 81,
        "height": 720,
        "width": 1280,
        "steps": 50,
        "tokens": 75600,
        "dim": 1536,
        "heads": 12,
        "layers": 30,
        "comm_mb_per_step": 55738.368,
        "collectives_per_step": 240,
        "cfg_exchange_mb": 9.6768,
        "legal_sp": (1, 2, 3, 4, 6),
    },
    "wan22-small": {
        "model": "wan22",
        "kind": "video",
        "scope": "formal",
        "frames": 121,
        "height": 512,
        "width": 896,
        "steps": 50,
        "tokens": 13888,
        "dim": 3072,
        "heads": 24,
        "layers": 30,
        "comm_mb_per_step": 20478.68928,
        "collectives_per_step": 240,
        "cfg_exchange_mb": 5.332992,
        "legal_sp": (1, 2, 3, 4, 6, 8),
    },
    "wan22-720p": {
        "model": "wan22",
        "kind": "video",
        "scope": "formal",
        "frames": 121,
        "height": 704,
        "width": 1280,
        "steps": 50,
        "tokens": 27280,
        "dim": 3072,
        "heads": 30,  # 24 heads in Wan2.2-5B; divisible by 1,2,3,4,6,8
        "num_heads": 24,
        "layers": 30,
        "comm_mb_per_step": 40225.9968,
        "collectives_per_step": 240,
        "cfg_exchange_mb": 10.47552,
        "legal_sp": (1, 2, 3, 4, 6, 8),
    },
    "sd35-512": {
        "model": "sd35",
        "kind": "image",
        "scope": "formal",
        "frames": 1,
        "height": 512,
        "width": 512,
        "steps": 28,
        "tokens": 1024,
        "dim": 1536,
        "heads": 24,
        "layers": 24,
        "comm_mb_per_step": 603.979776,
        "collectives_per_step": 192,
        "cfg_exchange_mb": 0.131072,
        "legal_sp": (1, 2, 3, 4, 6, 8),
    },
    "sd35-1024": {
        "model": "sd35",
        "kind": "image",
        "scope": "formal",
        "frames": 1,
        "height": 1024,
        "width": 1024,
        "steps": 28,
        "tokens": 4096,
        "dim": 1536,
        "heads": 24,
        "layers": 24,
        "comm_mb_per_step": 2415.919104,
        "collectives_per_step": 192,
        "cfg_exchange_mb": 0.524288,
        "legal_sp": (1, 2, 3, 4, 6, 8),
    },
}

# Historical development steady-step measurements from 03-motivation.tex (8x A100-PCIE-40GB).
HISTORICAL_DEV_STEP_MS: dict[str, dict[str, float]] = {
    "wan21-dev": {
        "sp1": 1149.62,
        "sp2": 1203.74,
        "sp4": 1096.01,
        "cfg2": 579.23,
        "cfg2sp2": 803.42,
        "cfg2sp3": 624.72,
        "cfg2sp4": 558.60,
    },
    "wan22-dev": {
        "sp1": 1061.41,
        "sp2": 1039.43,
        "sp4": 944.68,
        "cfg2": 538.29,
        "cfg2sp2": 672.00,
        "cfg2sp3": 557.88,
        "cfg2sp4": 482.91,
    },
}


def form_tuple(name: str) -> tuple[int, int]:
    clean = name.removeprefix("static_")
    if clean.startswith("cfg2"):
        rest = clean[4:]
        return 2, (1 if rest == "" else int(rest.removeprefix("sp")))
    if clean.startswith("sp"):
        return 1, int(clean.removeprefix("sp"))
    raise ValueError(f"unsupported parallel form name: {name}")


def form_name(cfg: int, sp: int) -> str:
    if cfg == 1:
        return f"sp{sp}"
    if cfg == 2:
        return "cfg2" if sp == 1 else f"cfg2sp{sp}"
    raise ValueError(f"unsupported (cfg, sp) tuple: {(cfg, sp)}")


def legal_forms_for_shape(shape_id: str, max_gpus: int = 8) -> list[str]:
    info = SHAPES[shape_id]
    forms: list[str] = []
    for cfg in (1, 2):
        for sp in info["legal_sp"]:
            if cfg * sp <= max_gpus:
                forms.append(form_name(cfg, sp))
    return forms


def per_step_comm_stats(shape_id: str, form: str) -> dict[str, Any]:
    """Compute exact per-step communication traffic and collective counts for Form(c, d)."""
    info = SHAPES[shape_id]
    cfg, sp = form_tuple(form)
    gpus = cfg * sp
    if sp not in info["legal_sp"]:
        raise ValueError(f"form {form} is illegal for {shape_id} (legal SP={info['legal_sp']})")
    total_comm_mb = info["comm_mb_per_step"]
    total_colls = info["collectives_per_step"]
    if sp == 1:
        a2a_colls_per_rank = 0
        a2a_tensor_bytes_per_call = 0
        a2a_bus_mb_per_rank_per_step = 0.0
    else:
        # Each CFG branch executes L layers * 4 All-to-Alls (120 for Wan, 96 for SD3.5).
        # With CFG=1, both branches run sequentially on the same group (240 or 192 calls).
        # With CFG=2, two disjoint subgroups each execute 1 branch (120 or 96 calls) concurrently.
        a2a_colls_per_rank = total_colls // cfg
        branch_comm_mb = total_comm_mb / cfg
        # In torch.distributed.all_to_all_single, each rank holds an input tensor of size
        # (branch_comm_bytes / a2a_colls_per_rank / sp), of which (sp - 1)/sp traverses PCIe.
        a2a_tensor_bytes_per_call = int(round((branch_comm_mb * 1e6) / a2a_colls_per_rank / sp))
        # Ensure tensor element count (FP16 = 2 bytes) is divisible by sp.
        elem_per_slice = max(1, (a2a_tensor_bytes_per_call // 2) // sp)
        a2a_tensor_bytes_per_call = elem_per_slice * sp * 2
        a2a_bus_mb_per_rank_per_step = branch_comm_mb * (sp - 1) / (sp * sp)
    cfg_exchange_mb = info["cfg_exchange_mb"] if cfg == 2 else 0.0
    cfg_exchange_calls = 1 if cfg == 2 else 0
    return {
        "form": form,
        "cfg_degree": cfg,
        "sp_degree": sp,
        "gpus": gpus,
        "concurrent_sp_subgroups": cfg if sp > 1 else 0,
        "a2a_collectives_per_rank": a2a_colls_per_rank,
        "a2a_tensor_bytes_per_call": a2a_tensor_bytes_per_call,
        "a2a_bus_mb_per_rank_per_step": round(a2a_bus_mb_per_rank_per_step, 6),
        "cfg_exchange_calls_per_step": cfg_exchange_calls,
        "cfg_exchange_mb_per_step": round(cfg_exchange_mb, 6),
        "total_pcie_bus_mb_per_rank": round(a2a_bus_mb_per_rank_per_step + cfg_exchange_mb, 6),
    }


def decompose_shape_measurements(
    shape_id: str,
    measured_forms_ms: dict[str, float],
    microbench_comm_ms: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Decompose measured steady-state step latencies for one shape across all measured forms."""
    if "sp1" not in measured_forms_ms or measured_forms_ms["sp1"] <= 0:
        raise ValueError(f"shape {shape_id} requires positive sp1 step_ms")
    sp1_ms = float(measured_forms_ms["sp1"])
    rows: list[dict[str, Any]] = []
    for fname, step_ms in sorted(
        measured_forms_ms.items(),
        key=lambda kv: (form_tuple(kv[0])[0] * form_tuple(kv[0])[1], form_tuple(kv[0])[0]),
    ):
        step_ms = float(step_ms)
        comm = per_step_comm_stats(shape_id, fname)
        gpus = comm["gpus"]
        ideal_compute_ms = sp1_ms / gpus
        inferred_overhead_ms = step_ms - ideal_compute_ms
        speedup = sp1_ms / step_ms
        efficiency_pct = (speedup / gpus) * 100.0
        bus_mb = comm["total_pcie_bus_mb_per_rank"]
        inferred_bw_gbps = (bus_mb / inferred_overhead_ms) if (bus_mb > 0 and inferred_overhead_ms > 0) else None
        row = {
            **comm,
            "measured_step_ms": round(step_ms, 4),
            "ideal_compute_ms": round(ideal_compute_ms, 4),
            "inferred_comm_and_contention_ms": round(inferred_overhead_ms, 4),
            "inferred_comm_fraction_pct": round((inferred_overhead_ms / step_ms) * 100.0, 2),
            "speedup_vs_sp1": round(speedup, 4),
            "parallel_efficiency_pct": round(efficiency_pct, 2),
            "gpu_seconds_per_step": round((step_ms / 1000.0) * gpus, 4),
            "inferred_effective_busbw_gbps": None if inferred_bw_gbps is None else round(inferred_bw_gbps, 4),
        }
        if microbench_comm_ms and fname in microbench_comm_ms:
            direct_comm_ms = float(microbench_comm_ms[fname])
            row["direct_measured_comm_ms"] = round(direct_comm_ms, 4)
            row["unexplained_residual_ms"] = round(step_ms - (ideal_compute_ms + direct_comm_ms), 4)
        rows.append(row)

    # Identify dominated forms under DitServe's CFG-aware dominance rule (min_degree >= 2):
    # Form A is dominated by Form B if B uses <= GPUs and is faster (or < GPUs and <= latency).
    for row in rows:
        dominators = []
        if row["gpus"] >= 2:
            for other in rows:
                if other["form"] == row["form"] or other["gpus"] < 2:
                    continue
                fewer = other["gpus"] < row["gpus"]
                faster = other["measured_step_ms"] < row["measured_step_ms"] - 1e-9
                if (
                    other["gpus"] <= row["gpus"]
                    and other["measured_step_ms"] <= row["measured_step_ms"] + 1e-9
                    and (fewer or faster)
                ):
                    dominators.append(other["form"])
        row["dominated"] = bool(dominators)
        row["dominated_by"] = dominators

    # Compute same-GPU-count CFG2xSP vs Pure-SP comparisons (2, 4, 6, 8 GPUs).
    by_form = {r["form"]: r for r in rows}
    same_gpu_contrasts = []
    for gpus, pure_f, cfg_f in ((2, "sp2", "cfg2"), (4, "sp4", "cfg2sp2"), (6, "sp6", "cfg2sp3"), (8, "sp8", "cfg2sp4")):
        if pure_f in by_form and cfg_f in by_form:
            p_ms = by_form[pure_f]["measured_step_ms"]
            c_ms = by_form[cfg_f]["measured_step_ms"]
            same_gpu_contrasts.append({
                "gpus": gpus,
                "pure_sp_form": pure_f,
                "pure_sp_ms": p_ms,
                "cfg_sp_form": cfg_f,
                "cfg_sp_ms": c_ms,
                "cfg_sp_speedup_over_pure_sp": round(p_ms / c_ms, 4),
                "latency_reduction_pct": round((p_ms - c_ms) / p_ms * 100.0, 2),
            })

    return {
        "shape_id": shape_id,
        "architecture": SHAPES[shape_id],
        "sp1_step_ms": round(sp1_ms, 4),
        "forms": rows,
        "pruned_dominated_forms": [r["form"] for r in rows if r["dominated"]],
        "retained_pareto_forms": [r["form"] for r in rows if r["gpus"] >= 2 and not r["dominated"]],
        "same_gpu_cfg_vs_pure_sp": same_gpu_contrasts,
    }


def load_bundle_forms(bundle_path: Path) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
    """Extract measured video form latencies and image service latencies from a shape-profiles bundle."""
    raw = Path(bundle_path).read_bytes()
    data = json.loads(raw)
    if data.get("schema") != "ditserve.shape-profiles.v1":
        raise ValueError(f"unexpected bundle schema in {bundle_path}")
    video_forms: dict[str, dict[str, float]] = {}
    image_service: dict[str, float] = {}
    for cname, cval in data.get("classes", {}).items():
        cost = cval.get("cost", {})
        if cost.get("kind") == "video":
            forms = {}
            for fname, fval in cost.get("forms", {}).items():
                placements = fval.get("placements", {})
                if placements:
                    first_key = sorted(placements)[0]
                    forms[fname] = float(placements[first_key]["step_ms"])
            video_forms[cname] = forms
        elif cost.get("kind") == "image":
            image_service[cname] = float(cost["service_ms"])
    meta = {
        "path": str(bundle_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "model": data.get("model"),
        "protocol_sha256": data.get("protocol_sha256"),
        "gpu_uuids": data.get("gpu_uuids", []),
    }
    return video_forms, image_service, meta


def build_decomposition_report(
    bundle_paths: list[Path],
    include_historical_dev: bool = True,
    microbench_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete Challenge-1 decomposition report from verified bundles and optional microbench."""
    bundles_meta = []
    shapes_out: dict[str, Any] = {}
    image_out: dict[str, Any] = {}
    micro_by_shape = {}
    if microbench_report:
        for sid, sdata in microbench_report.get("step_equivalent_replay", {}).items():
            micro_by_shape[sid] = {
                k: v["total_comm_median_ms"]
                for k, v in sdata.get("forms", {}).items()
                if "total_comm_median_ms" in v
            }

    if include_historical_dev:
        for sid, forms_ms in HISTORICAL_DEV_STEP_MS.items():
            shapes_out[sid] = decompose_shape_measurements(sid, forms_ms, micro_by_shape.get(sid))

    for path in bundle_paths:
        v_forms, img_srv, meta = load_bundle_forms(path)
        bundles_meta.append(meta)
        for sid, forms_ms in v_forms.items():
            shapes_out[sid] = decompose_shape_measurements(sid, forms_ms, micro_by_shape.get(sid))
        for img_id, srv_ms in img_srv.items():
            steps = SHAPES[img_id]["steps"] if img_id in SHAPES else 28
            est_step_ms = srv_ms / steps
            img_comm = {
                f: per_step_comm_stats(img_id, f)
                for f in ("sp1", "sp2", "sp4", "sp8", "cfg2", "cfg2sp2", "cfg2sp4")
                if img_id in SHAPES
            }
            image_out[img_id] = {
                "shape_id": img_id,
                "single_gpu_batch2_service_ms": round(srv_ms, 4),
                "steps": steps,
                "upper_bound_step_ms": round(est_step_ms, 4),
                "forms_comm_profile": img_comm,
                "note": (
                    "SD3.5-Medium fuses conditional and unconditional CFG branches as batch_size=2 on 1 GPU; "
                    "per-step time (~62 ms at 512x512, ~232 ms at 1024x1024) is dominated by 192 PCIe "
                    "collective launches if distributed across SP2/SP4/SP8."
                ),
            }

    return {
        "schema": SCHEMA,
        "mode": "decomposition",
        "scope": "Challenge-1 PCIe communication decomposition across development and formal shapes",
        "bundles": bundles_meta,
        "video_shapes": shapes_out,
        "image_shapes": image_out,
        "microbenchmark_attached": microbench_report is not None,
    }


def check_storage(mount: str = "/shared") -> dict[str, int]:
    """Enforce project shared-storage stop threshold (>800 GiB or >=80%)."""
    target = mount if Path(mount).exists() else "/data"
    if not Path(target).exists():
        return {"used_gib": 0, "percent": 0}
    proc = subprocess.run(["df", "-BG", target], capture_output=True, text=True, timeout=20)
    try:
        row = proc.stdout.splitlines()[-1].split()
        used, pct = int(row[2].rstrip("G")), int(row[4].rstrip("%"))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"failed to parse df output for {target}") from exc
    if proc.returncode != 0 or used > STORAGE_MAX_GIB or pct >= STORAGE_MAX_PCT:
        raise ValueError(f"shared-storage stop threshold reached: {used}G ({pct}%)")
    return {"used_gib": used, "percent": pct}


def query_hardware_topology() -> dict[str, Any]:
    """Query nvidia-smi topology matrix and per-GPU PCIe link Gen/Width/UUID."""
    info: dict[str, Any] = {"topo_matrix": None, "gpus": []}
    try:
        topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=15)
        if topo.returncode == 0:
            info["topo_matrix"] = topo.stdout.strip()
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if q.returncode == 0:
            for line in q.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 8:
                    info["gpus"].append({
                        "index": int(parts[0]),
                        "uuid": parts[1],
                        "name": parts[2],
                        "pci_bus_id": parts[3],
                        "pcie_gen_current": parts[4],
                        "pcie_gen_max": parts[5],
                        "pcie_width_current": parts[6],
                        "pcie_width_max": parts[7],
                    })
    except Exception as exc:
        info["query_error"] = repr(exc)
    return info


def run_microbenchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Execute multi-GPU NCCL All-to-All Hockney curve and Step-Equivalent Collective Replay."""
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Do NOT pass device_id=device to init_process_group when using dist.new_group() subgroups:
    # PyTorch NCCL device_id rendezvous breaks subgroup creation (see xfuser parallel_state.py L267).
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world not in (2, 4, 8):
        raise ValueError(f"pcie_collective_characterization expects world_size in (2, 4, 8), got {world}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.parent / f"rank-{rank}-progress.json"

    def write_progress(stage: str, extra: dict[str, Any] | None = None) -> None:
        if rank == 0:
            payload = {"rank": rank, "stage": stage, "timestamp": time.time(), **(extra or {})}
            progress_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Eagerly initialize the default WORLD NCCL communicator on all ranks in lockstep.
    world_init = torch.zeros(1, dtype=torch.float32, device=device)
    dist.all_reduce(world_init)
    torch.cuda.synchronize(device)
    write_progress("world_initialized")

    initial_storage = check_storage() if rank == 0 else {}
    hw_topo = query_hardware_topology() if rank == 0 else {}

    def timed_cuda(fn, warmup: int, iters: int) -> dict[str, float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            samples.append(float(start.elapsed_time(end)))
        return {
            "median_ms": round(statistics.median(samples), 4),
            "p95_ms": round(sorted(samples)[min(len(samples) - 1, int(math.ceil(0.95 * len(samples))) - 1)], 4),
            "min_ms": round(min(samples), 4),
            "max_ms": round(max(samples), 4),
            "iters": iters,
        }

    pg_cache: dict[tuple[int, ...], Any] = {}

    def get_subgroup(ranks: list[int]) -> Any:
        key = tuple(ranks)
        if key not in pg_cache:
            pg = dist.new_group(list(key), backend="nccl")
            if rank in key:
                dummy = torch.zeros(1, dtype=torch.float32, device=device)
                dist.all_reduce(dummy, group=pg)
                torch.cuda.synchronize(device)
            pg_cache[key] = pg
        return pg_cache[key]

    # 1. Hockney latency floor and saturation bandwidth curve across degrees d <= world.
    hockney: dict[str, Any] = {}
    degrees = [d for d in (2, 3, 4, 6, 8) if d <= world]
    probe_sizes_bytes = [
        4 * 1024,          # 4 KB (latency floor alpha)
        64 * 1024,         # 64 KB
        1 * 1024 * 1024,   # 1 MB
        4 * 1024 * 1024,   # 4 MB
        16 * 1024 * 1024,  # 16 MB
        32 * 1024 * 1024,  # 32 MB
        64 * 1024 * 1024,  # 64 MB
    ]
    for d in degrees:
        write_progress(f"hockney_sp{d}_start")
        active_ranks = list(range(d))
        pg = get_subgroup(active_ranks)
        d_rows = []
        for nbytes in probe_sizes_bytes:
            elems = max(d, (nbytes // 2 // d) * d)
            actual_bytes = elems * 2
            if rank in active_ranks:
                inp = torch.ones(elems, dtype=torch.float16, device=device)
                out = torch.empty_like(inp)
                stats = timed_cuda(lambda: dist.all_to_all_single(out, inp, group=pg), args.warmup, args.iters)
                bus_mb = (actual_bytes / 1e6) * (d - 1) / d
                stats["bytes_per_rank"] = actual_bytes
                stats["bus_mb_per_rank"] = round(bus_mb, 6)
                stats["busbw_gbps"] = round(bus_mb / stats["median_ms"], 4) if stats["median_ms"] > 0 else 0.0
                d_rows.append(stats)
            dist.all_reduce(world_init)
            torch.cuda.synchronize(device)
        if rank == 0:
            hockney[f"sp{d}"] = {
                "degree": d,
                "latency_floor_4kb_ms": d_rows[0]["median_ms"] if d_rows else None,
                "peak_busbw_gbps": max((r["busbw_gbps"] for r in d_rows), default=0.0),
                "measurements": d_rows,
            }

    # 2. Step-equivalent collective replay for each shape and legal form <= world.
    step_replay: dict[str, Any] = {}
    selected_shapes = args.shapes or list(SHAPES)
    for sid in selected_shapes:
        write_progress(f"step_replay_{sid}_start")
        sforms = legal_forms_for_shape(sid, max_gpus=world)
        form_results: dict[str, Any] = {}
        for fname in sforms:
            cstats = per_step_comm_stats(sid, fname)
            cfg, sp = cstats["cfg_degree"], cstats["sp_degree"]
            if cfg == 1 and sp == 1:
                form_results[fname] = {**cstats, "total_comm_median_ms": 0.0}
                continue

            # Create or reuse subgroup(s) for All-to-All and CFG exchange pairs.
            subgroups = []
            if sp > 1:
                for branch in range(cfg):
                    ranks = list(range(branch * sp, (branch + 1) * sp))
                    subgroups.append((ranks, get_subgroup(ranks)))
            cfg_pg = None
            if cfg == 2:
                # Rank 0 of branch 0 (rank 0) and rank 0 of branch 1 (rank sp) exchange latent velocity prediction.
                cfg_ranks = [0, sp]
                cfg_pg = get_subgroup(cfg_ranks)

            my_sp_pg = None
            for ranks, pg in subgroups:
                if rank in ranks:
                    my_sp_pg = pg

            a2a_calls = cstats["a2a_collectives_per_rank"]
            a2a_bytes = cstats["a2a_tensor_bytes_per_call"]
            cfg_bytes = int(round(cstats["cfg_exchange_mb_per_step"] * 1e6))

            a2a_inp = a2a_out = None
            if my_sp_pg is not None and a2a_calls > 0 and a2a_bytes > 0:
                elems = max(sp, (a2a_bytes // 2 // sp) * sp)
                a2a_inp = torch.ones(elems, dtype=torch.float16, device=device)
                a2a_out = torch.empty_like(a2a_inp)

            cfg_inp = cfg_out = None
            if cfg == 2 and rank in (0, sp) and cfg_bytes > 0:
                celems = max(2, cfg_bytes // 2)
                cfg_inp = torch.ones(celems, dtype=torch.float16, device=device)
                cfg_out = torch.empty(2 * celems, dtype=torch.float16, device=device)

            def run_one_step_collectives():
                if my_sp_pg is not None and a2a_inp is not None:
                    for _ in range(a2a_calls):
                        dist.all_to_all_single(a2a_out, a2a_inp, group=my_sp_pg)
                if cfg_pg is not None and cfg_inp is not None:
                    dist.all_gather_into_tensor(cfg_out, cfg_inp, group=cfg_pg)

            tstats = timed_cuda(run_one_step_collectives, warmup=args.step_warmup, iters=args.step_iters)

            if rank == 0:
                bus_mb = cstats["total_pcie_bus_mb_per_rank"]
                med_ms = tstats["median_ms"]
                form_results[fname] = {
                    **cstats,
                    "step_collective_timing": tstats,
                    "total_comm_median_ms": med_ms,
                    "measured_step_busbw_gbps": round(bus_mb / med_ms, 4) if med_ms > 0 else 0.0,
                }
            dist.all_reduce(world_init)
            torch.cuda.synchronize(device)
        if rank == 0:
            step_replay[sid] = {"shape_id": sid, "forms": form_results}

    report: dict[str, Any] = {}
    if rank == 0:
        write_progress("writing_summary")
        report = {
            "schema": SCHEMA,
            "mode": "hardware_microbenchmark",
            "world_size": world,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "initial_storage": initial_storage,
            "hardware_topology": hw_topo,
            "hockney_all2all_curve": hockney,
            "step_equivalent_replay": step_replay,
        }
        if args.bundle:
            report["calibration_decomposition"] = build_decomposition_report(
                args.bundle,
                include_historical_dev=True,
                microbench_report=report,
            )
        check_storage()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        write_progress("completed")
        print(json.dumps({"status": "completed", "output": str(args.output), "world_size": world}))

    dist.all_reduce(world_init)
    torch.cuda.synchronize(device)
    dist.destroy_process_group()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dec = sub.add_parser("decompose", help="CPU decomposition of calibration bundles")
    dec.add_argument("--bundle", type=Path, action="append", default=[], help="shape-profiles.v1 bundle path(s)")
    dec.add_argument("--microbench", type=Path, help="optional hardware microbenchmark JSON to join")
    dec.add_argument("--no-historical-dev", action="store_true")
    dec.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run", help="Multi-GPU torchrun NCCL All-to-All & CFG2xSP replay")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--bundle", type=Path, action="append", default=[])
    run.add_argument("--shapes", nargs="*", choices=sorted(SHAPES))
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--iters", type=int, default=10)
    run.add_argument("--step-warmup", type=int, default=1)
    run.add_argument("--step-iters", type=int, default=3)

    args = parser.parse_args(argv)
    if args.command == "decompose":
        micro = json.loads(args.microbench.read_bytes()) if args.microbench else None
        report = build_decomposition_report(
            args.bundle,
            include_historical_dev=not args.no_historical_dev,
            microbench_report=micro,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({"status": "completed", "output": str(args.output), "shapes": sorted(report["video_shapes"])}))
        return 0
    run_microbenchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
