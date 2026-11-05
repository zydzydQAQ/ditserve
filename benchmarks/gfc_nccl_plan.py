#!/usr/bin/env python3
"""Offline GFC-vs-NCCL experiment plan for DitServe on PCIe A100 (no GPU, no submission).

What this does (CPU only, deterministic):
  1. Derives the per-rank collective message sizes and per-step counts that DitServe's xDiT
     data path issues for the frozen formal shapes (formal-mixed-v1) under CFGxSP forms with
     SP in {2, 4, 8}, from model geometry (transformer/VAE configs recorded in the cal8
     runtime-facts bundle) and the code paths in ``workers/xfuser/model_executor/layers/usp.py``
     (separate Q/K/V/O all-to-all, ``combine_qkv_a2a`` default False),
     ``workers/ditserve/models/xdit/wan.py`` (CFG prediction all-gather) and
     ``workers/ditserve/models/xdit/collectives.py`` (GFC adapter).
  2. Reports how much of that traffic the existing GFC smoke (256 B / 64 KiB payloads,
     ``max_collective_bytes`` = 1 MiB) covers.
  3. Emits a GFC-vs-NCCL microbenchmark matrix (steady-state latency/bandwidth sweep,
     group setup and dynamic switching, step-equivalent replay) with a GPU-time estimate.

It never imports torch, never touches the cluster and never submits a job. Every number is
labelled ``derived`` (exact arithmetic from configs), ``inferred`` (from code reading or public
defaults, not measured) or ``assumption`` (planning constant). Run:

    python benchmarks/gfc_nccl_plan.py --output-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "ditserve.gfc-nccl-plan.v1"
MiB = 1024 * 1024
BF16 = 2

# Source of model facts: cal8 runtime-facts bundle (read-only evidence).
RUNTIME_FACTS = {
    "path": ".local/ditserve-kairos-rewrite-runtime/evidence/20261103-cal8-51740c0/"
            "calibration-local/wan21/runtime-facts/source.json",
    "sha256": "a635044022b1b9c8816ef40d1822f572b1246be135961db386bf40140facfecf",
}

# transformer_config / vae_config values copied from RUNTIME_FACTS (verified by a test).
# Wan2.1 VAE has no scale_factor_* keys; fit_native_profile.tokens_and_dim defaults (8, 4) apply.
MODELS: dict[str, dict[str, Any]] = {
    "wan21": {
        "model_id": "Wan2.1-T2V-1.3B", "kind": "video",
        "heads": 12, "head_dim": 128, "layers": 30, "out_channels": 16,
        "patch": (1, 2, 2), "vae_spatial": 8, "vae_temporal": 4,
        # diffusers WanPipeline runs cond and uncond as two separate forwards (batch 1 each).
        "cfg_batched": False, "text_tokens": 0, "dual_attention_layers": 0,
    },
    "wan22": {
        "model_id": "Wan2.2-TI2V-5B", "kind": "video",
        "heads": 24, "head_dim": 128, "layers": 30, "out_channels": 48,
        "patch": (1, 2, 2), "vae_spatial": 16, "vae_temporal": 4,
        "cfg_batched": False, "text_tokens": 0, "dual_attention_layers": 0,
    },
    "sd35": {
        "model_id": "SD3.5-Medium", "kind": "image",
        "heads": 24, "head_dim": 64, "layers": 24, "out_channels": 16,
        "patch": (1, 2, 2), "vae_spatial": 8, "vae_temporal": 1,
        # diffusers/xFuser SD3 pipelines concatenate [uncond, cond] into batch 2 (inferred).
        # 77 CLIP + 256 T5 tokens (diffusers default max_sequence_length=256; assumption).
        "cfg_batched": True, "text_tokens": 333, "dual_attention_layers": 13,
    },
}

# Frozen formal shapes: configs/benchmarks/formal-mixed-v1.json (committed in 12b0918).
SHAPES: dict[str, dict[str, Any]] = {
    "wan21-480p": {"model": "wan21", "frames": 81, "height": 480, "width": 832, "steps": 50},
    "wan21-720p": {"model": "wan21", "frames": 81, "height": 720, "width": 1280, "steps": 50},
    "wan22-small": {"model": "wan22", "frames": 121, "height": 512, "width": 896, "steps": 50},
    "wan22-720p": {"model": "wan22", "frames": 121, "height": 704, "width": 1280, "steps": 50},
    "sd35-512": {"model": "sd35", "frames": 1, "height": 512, "width": 512, "steps": 28},
    "sd35-1024": {"model": "sd35", "frames": 1, "height": 1024, "width": 1024, "steps": 28},
}

FORMS = ((1, 2), (1, 4), (1, 8), (2, 2), (2, 4))  # (cfg, sp) with cfg*sp <= 8, sp in {2,4,8}
SMOKE_TESTED_BYTES = (256, 65536)       # external_gfc_smoke.py SIZES
SMOKE_MAX_COLLECTIVE_BYTES = 1 * MiB    # smoke runtime config (rank-0.json evidence)
ADAPTER_MAX_COLLECTIVE_BYTES = 128 * MiB  # collectives.configure default
ALL_GATHER_GRID_CAP = 32 * MiB
WORLD = 8

# Planning constants (assumptions, not measurements).
PLAN = {
    "warmup": 10,
    "iters": 50,
    "repeats": 3,                       # independent jobs; one job is never reported as a gain
    "alpha_s": 30e-6,                   # assumed per-collective latency floor
    "beta_pessimistic_Bps": 3.75e9,     # fitted cal8 link proxy (not hardware bandwidth)
    "beta_optimistic_Bps": 20e9,        # ~PCIe Gen4 x16 practical unidirectional (assumption)
    "case_overhead_s": 0.5,
    "job_overhead_s": 900.0,            # container start, imports, Triton JIT, runtime init
    "nccl_create_s": 0.8,               # GF-DiT reports ~778 ms on its hardware; planning only
    "gpus_per_job": 8,                  # full node reserved so PCIe contention is controlled
}

PLACEMENTS = {
    2: {"pair-aligned": (0, 1), "pair-unaligned": (1, 2), "pair-half-split": (0, 4), "pair-far": (0, 7)},
    4: {"quad-aligned": (0, 1, 2, 3), "quad-straddle": (2, 3, 4, 5), "quad-interleaved": (0, 2, 4, 6)},
    8: {"octet": tuple(range(8))},
}

GFC_BACKENDS = {
    "gfc_vec": {"config": {"use_copy_engine": False, "enable_fused_path": False}},
    "gfc_pipelined4": {"config": {"use_copy_engine": False, "pipeline_chunks": 4}},
    "gfc_fused": {"config": {"enable_fused_path": True, "fused_num_channels": 24}},
    "gfc_copy_engine": {"config": {"use_copy_engine": True}},
}
EXCLUDED_PATHS = {
    "gfc_tma": "A100 is sm_80: no TMA (Hopper feature); runtime raises TMAUnsupportedError",
}


def tokens(shape: dict[str, Any]) -> int:
    """Main (image/video) sequence length, as fit_native_profile.tokens_and_dim."""
    model = MODELS[shape["model"]]
    pt, ph, pw = model["patch"]
    frames = (shape["frames"] - 1) // model["vae_temporal"] + 1 if model["kind"] == "video" else 1
    return ((frames // pt) * (shape["height"] // (model["vae_spatial"] * ph))
            * (shape["width"] // (model["vae_spatial"] * pw)))


def dim(model: dict[str, Any]) -> int:
    return model["heads"] * model["head_dim"]


def legal(model_name: str, cfg: int, sp: int) -> tuple[bool, str]:
    model = MODELS[model_name]
    if cfg * sp > WORLD:
        return False, "cfg*sp exceeds one 8-GPU node"
    if model["heads"] % sp:
        return False, f"pure Ulysses needs heads ({model['heads']}) divisible by sp ({sp})"
    return True, ""


def messages(shape_id: str, cfg: int, sp: int) -> list[dict[str, Any]]:
    """Per-rank collective messages per denoising step for one (shape, cfg, sp) form."""
    shape = SHAPES[shape_id]
    model = MODELS[shape["model"]]
    s = tokens(shape)
    d = dim(model)
    patch_vol = math.prod(model["patch"])
    batch = 2 if (model["cfg_batched"] and cfg == 1) else 1
    forwards = 2 if (not model["cfg_batched"] and cfg == 1) else 1
    s_local = -(-s // sp)  # uniform SP suffix padding
    out: list[dict[str, Any]] = []
    if sp > 1:
        a2a = batch * s_local * d * BF16
        upper = batch * (s_local + model["text_tokens"]) * d * BF16 if model["text_tokens"] else a2a
        out.append({
            "name": "ulysses_a2a", "collective": "all2all", "group_size": sp,
            "bytes_per_rank": a2a, "bytes_per_rank_upper": upper,
            "count_per_step": 4 * model["layers"] * forwards,
            "basis": "derived: 4 separate Q/K/V/O all-to-all per attention layer (usp.py "
                     "combine_qkv_a2a=False); bytes = batch*ceil(S/sp)*dim*2",
        })
        if model["dual_attention_layers"]:
            out.append({
                "name": "ulysses_a2a_dual_attn2", "collective": "all2all", "group_size": sp,
                "bytes_per_rank": a2a, "bytes_per_rank_upper": a2a,
                "count_per_step": 4 * model["dual_attention_layers"] * forwards,
                "basis": "inferred: SD3.5-M dual_attention_layers (13) add an image-only attn2; "
                         "routing through xDiT USP not verified (architecture() omits it)",
            })
        gather = batch * s_local * patch_vol * model["out_channels"] * BF16
        out.append({
            "name": "sp_output_all_gather", "collective": "all_gather", "group_size": sp,
            "bytes_per_rank": gather, "bytes_per_rank_upper": gather,
            "count_per_step": forwards,
            "basis": "inferred: one SP output gather per forward after proj_out (bf16)",
        })
    if cfg == 2:
        exchange = s * patch_vol * model["out_channels"] * BF16
        out.append({
            "name": "cfg_prediction_all_gather", "collective": "all_gather", "group_size": cfg * sp,
            "bytes_per_rank": exchange, "bytes_per_rank_upper": 2 * exchange,
            "count_per_step": 1,
            "basis": "derived for Wan (xdit/wan.py exchange_predictions over all cfg*sp ranks, "
                     "bf16; upper = fp32); SD3.5 CFG-parallel path not verified",
        })
    out.append({
        "name": "doorbell_flag_all_gather", "collective": "all_gather", "group_size": cfg * sp,
        "bytes_per_rank": None, "bytes_per_rank_upper": None, "count_per_step": None,
        "basis": "unknown: reduce_flag payload and frequency not derived (tiny, <=1 MiB)",
    })
    return out


def per_step_totals(shape_id: str, cfg: int, sp: int) -> dict[str, Any]:
    msgs = [m for m in messages(shape_id, cfg, sp) if m["bytes_per_rank"] is not None]
    count = sum(m["count_per_step"] for m in msgs)
    total = sum(m["count_per_step"] * m["bytes_per_rank"] for m in msgs)
    small = [m for m in msgs if m["bytes_per_rank"] <= SMOKE_MAX_COLLECTIVE_BYTES]
    return {
        "collectives_per_step": count,
        "bytes_per_step_per_rank": total,
        "max_message_bytes": max((m["bytes_per_rank"] for m in msgs), default=0),
        "max_message_bytes_upper": max((m["bytes_per_rank_upper"] for m in msgs), default=0),
        "count_le_1MiB": sum(m["count_per_step"] for m in small),
        "bytes_le_1MiB": sum(m["count_per_step"] * m["bytes_per_rank"] for m in small),
        "fits_adapter_128MiB": all(_gfc_fits(m, ADAPTER_MAX_COLLECTIVE_BYTES) for m in msgs),
        "fits_smoke_1MiB": all(_gfc_fits(m, SMOKE_MAX_COLLECTIVE_BYTES) for m in msgs),
    }


def _gfc_fits(message: dict[str, Any], limit: int) -> bool:
    """runtime.all2all: slice*group == input <= max; all_gather: input <= max (staged slice)."""
    return message["bytes_per_rank_upper"] <= limit


def message_table() -> list[dict[str, Any]]:
    rows = []
    for shape_id, shape in SHAPES.items():
        for cfg, sp in FORMS:
            ok, reason = legal(shape["model"], cfg, sp)
            row = {"shape": shape_id, "model": shape["model"], "cfg": cfg, "sp": sp,
                   "tokens": tokens(shape), "legal": ok, "illegal_reason": reason,
                   "messages": messages(shape_id, cfg, sp)}
            row.update(per_step_totals(shape_id, cfg, sp))
            rows.append(row)
    return rows


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    legal_rows = [r for r in rows if r["legal"]]
    count = sum(r["collectives_per_step"] for r in legal_rows)
    total = sum(r["bytes_per_step_per_rank"] for r in legal_rows)
    small_count = sum(r["count_le_1MiB"] for r in legal_rows)
    small_bytes = sum(r["bytes_le_1MiB"] for r in legal_rows)
    sizes = sorted({m["bytes_per_rank"] for r in legal_rows for m in r["messages"]
                    if m["bytes_per_rank"] is not None})
    return {
        "weighting": "every legal (shape, cfg, sp) form weighted equally, per step per rank; "
                     "doorbell flags excluded (size/count unknown)",
        "legal_forms": len(legal_rows),
        "collectives_per_step_sum": count,
        "count_fraction_le_1MiB": round(small_count / count, 6) if count else None,
        "bytes_fraction_le_1MiB": round(small_bytes / total, 6) if total else None,
        "smoke_tested_bytes": list(SMOKE_TESTED_BYTES),
        "exact_sizes_tested_by_smoke": sum(1 for s in sizes if s in SMOKE_TESTED_BYTES),
        "distinct_real_sizes": len(sizes),
        "min_real_bytes": sizes[0] if sizes else None,
        "max_real_bytes": sizes[-1] if sizes else None,
        "forms_fitting_smoke_1MiB": sum(1 for r in legal_rows if r["fits_smoke_1MiB"]),
        "forms_fitting_adapter_128MiB": sum(1 for r in legal_rows if r["fits_adapter_128MiB"]),
    }


def real_sizes(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Distinct real per-rank sizes keyed by 'collective/group_size' (legal forms only)."""
    found: dict[str, set[int]] = {}
    for r in rows:
        if not r["legal"]:
            continue
        for m in r["messages"]:
            if m["bytes_per_rank"] is None:
                continue
            found.setdefault(f"{m['collective']}/{m['group_size']}", set()).add(m["bytes_per_rank"])
    return {k: sorted(v) for k, v in sorted(found.items())}


def grid_sizes() -> list[int]:
    return [1 << e for e in range(12, 28)]  # 4 KiB .. 128 MiB


def collective_time_s(collective: str, group: int, nbytes: int, beta: float) -> float:
    moved = nbytes * (group - 1) / group if collective == "all2all" else nbytes * (group - 1)
    return PLAN["alpha_s"] + moved / beta


def phase1_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reals = real_sizes(rows)
    cases = []
    for collective in ("all2all", "all_gather"):
        for group, placements in PLACEMENTS.items():
            sizes = sorted(set(grid_sizes()) | set(reals.get(f"{collective}/{group}", [])))
            for pname, ranks in placements.items():
                for nbytes in sizes:
                    if collective == "all2all" and nbytes % group:
                        continue
                    if nbytes > ADAPTER_MAX_COLLECTIVE_BYTES:
                        continue
                    if collective == "all_gather" and nbytes > ALL_GATHER_GRID_CAP:
                        continue  # real all-gathers are <= ~10.5 MiB (fp32 upper ~21 MiB)
                    for backend in ("nccl", *GFC_BACKENDS):
                        cases.append({
                            "phase": "p1_steady", "collective": collective, "group_size": group,
                            "placement": pname, "ranks": list(ranks), "bytes_per_rank": nbytes,
                            "size_origin": "real" if nbytes in reals.get(f"{collective}/{group}", [])
                            else "grid",
                            "backend": backend,
                        })
    return cases


def phase2_cases() -> list[dict[str, Any]]:
    cases = []
    group_sets = {
        "aligned": [(0, 1), (2, 3), (0, 1, 2, 3), (4, 5, 6, 7), tuple(range(8))],
        "unaligned": [(1, 2), (3, 4), (1, 2, 3, 4), (0, 2, 4, 6), (3, 4, 5, 6)],
    }
    for set_name, groups in group_sets.items():
        for ranks in groups:
            for backend in ("nccl_lazy_new_group", "nccl_eager_split_new_group", "gfc_register"):
                cases.append({"phase": "p2_cold_setup", "group_set": set_name, "ranks": list(ranks),
                              "backend": backend, "reps": 5})
    sizes = {"small-64KiB": 65536, "wan22-720p-sp4-a2a": next(
        m["bytes_per_rank"] for m in messages("wan22-720p", 1, 4) if m["name"] == "ulysses_a2a")}
    for set_name in group_sets:
        for size_name, nbytes in sizes.items():
            for every in (1, 4, 120, 240, 0):  # 0 = never switch (steady reference)
                for backend in ("nccl_precreated", "nccl_on_demand", "gfc_vec"):
                    if backend == "nccl_on_demand" and every and every < 120:
                        continue  # ~1 s per create; only per-forward or slower switching
                    cases.append({"phase": "p2_switch", "group_set": set_name, "size": size_name,
                                  "bytes_per_rank": nbytes, "switch_every_ops": every,
                                  "ops": 960, "backend": backend})
    for backend in ("nccl_precreated_aligned_15", "nccl_precreated_all_pairs_quads", "gfc_runtime"):
        cases.append({"phase": "p2_memory", "backend": backend})
    return cases


def phase3_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for r in rows:
        if not r["legal"]:
            continue
        concurrency = ("single",) if r["cfg"] * r["sp"] > 4 else ("single", "two_disjoint_groups")
        for conc in concurrency:
            for backend in ("nccl", "gfc_best_from_p1"):
                cases.append({"phase": "p3_step_replay", "shape": r["shape"], "cfg": r["cfg"],
                              "sp": r["sp"], "concurrency": conc, "steps": 3, "backend": backend})
    return cases


def estimate(rows: list[dict[str, Any]], p1, p2, p3) -> dict[str, Any]:
    iters = PLAN["warmup"] + PLAN["iters"]
    out = {}
    for label, beta in (("pessimistic", PLAN["beta_pessimistic_Bps"]),
                        ("optimistic", PLAN["beta_optimistic_Bps"])):
        t1 = sum(iters * collective_time_s(c["collective"], c["group_size"], c["bytes_per_rank"], beta)
                 + PLAN["case_overhead_s"] for c in p1)
        t2 = 0.0
        for c in p2:
            if c["phase"] == "p2_cold_setup":
                per = PLAN["nccl_create_s"] if c["backend"].startswith("nccl") else 0.01
                t2 += c["reps"] * (per + PLAN["case_overhead_s"])
            elif c["phase"] == "p2_switch":
                base = c["ops"] * collective_time_s("all2all", 4, c["bytes_per_rank"], beta)
                switches = c["ops"] // c["switch_every_ops"] if c["switch_every_ops"] else 0
                extra = switches * PLAN["nccl_create_s"] if c["backend"] == "nccl_on_demand" else 0.0
                t2 += base + extra + PLAN["case_overhead_s"]
            else:
                t2 += 120.0
        t3 = 0.0
        by_key = {(r["shape"], r["cfg"], r["sp"]): r for r in rows}
        for c in p3:
            r = by_key[(c["shape"], c["cfg"], c["sp"])]
            step = sum(m["count_per_step"] * collective_time_s(m["collective"], m["group_size"],
                                                               m["bytes_per_rank"], beta)
                       for m in r["messages"] if m["bytes_per_rank"] is not None)
            t3 += (c["steps"] + 1) * step + PLAN["case_overhead_s"]
        wall = t1 + t2 + t3 + PLAN["job_overhead_s"]
        out[label] = {
            "p1_s": round(t1, 1), "p2_s": round(t2, 1), "p3_s": round(t3, 1),
            "wall_per_repeat_h": round(wall / 3600, 3),
            "gpu_hours_all_repeats": round(wall / 3600 * PLAN["gpus_per_job"] * PLAN["repeats"], 2),
        }
    return out


def minimal_plan(rows) -> dict[str, Any]:
    """Smallest set that can support (or refute) a paper claim; estimated, not submitted."""
    p0 = 0.25  # h per node: topology + pairwise P2P read/write matrix + NCCL transport log
    return {
        "jobs": [
            {"name": "gfc-p0-topology-probe", "gpus": 8, "est_wall_h": p0, "repeats": 1,
             "content": "nvidia-smi topo -m / -p2p r,w; pairwise cudaMemcpyPeer and SM-pull "
                        "read bandwidth 8x8; NCCL_DEBUG=INFO transport (P2P vs SHM) per pair"},
            {"name": "gfc-p1p2p3-core", "gpus": 8, "repeats": 3,
             "est_wall_h": "see estimate.pessimistic.wall_per_repeat_h (upper) / optimistic",
             "content": "steady sweep (real sizes + 4 KiB..128 MiB), setup/switch, step replay"},
        ],
        "optional": [{"name": "gfc-p4-e2e-wan", "gpus": 8, "est_wall_h": 1.0, "repeats": 3,
                      "content": "Wan2.2-small cfg1sp4 and cfg2sp4, collective_backend=gfc vs nccl, "
                                 "3 requests each; step-time delta only if p1/p3 show a gap"}],
        "p0_gpu_hours": p0 * 8,
    }


def build_plan() -> dict[str, Any]:
    rows = message_table()
    p1, p2, p3 = phase1_cases(rows), phase2_cases(), phase3_cases(rows)
    plan = {
        "schema": SCHEMA,
        "scope": "offline plan; nothing measured, nothing submitted",
        "sources": {
            "runtime_facts": RUNTIME_FACTS,
            "shapes": "configs/benchmarks/formal-mixed-v1.json (commit 12b0918)",
            "gfc_upstream_commit": "9aadcdc6d4abc30af13feac908fbb4be98313398",
            "topology_evidence": "native-serving-*/acceptance/topology.txt and cal8 runtime-facts: "
                                 "all GPU pairs PHB, CPU affinity 0-79, NUMA affinity 0-1, "
                                 "GPU NUMA ID N/A -> switch/NUMA placement classes unknown",
        },
        "models": {k: {**v, "patch": list(v["patch"]), "dim": dim(v)} for k, v in MODELS.items()},
        "message_table": rows,
        "coverage": coverage(rows),
        "real_sizes": real_sizes(rows),
        "matrix": {
            "metrics": ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                        "algbw_GBps", "busbw_GBps (nccl-tests convention; not HW utilisation)",
                        "setup_ms", "switch_overhead_ms", "nccl_comm_gpu_mem_MiB",
                        "bitwise_equal_to_nccl"],
            "timing": "CUDA events around each op incl. GFC stream sync; host wall for setup",
            "placements": {str(k): {n: list(r) for n, r in v.items()} for k, v in PLACEMENTS.items()},
            "placement_classes": "unknown-pending-p0; hypothesis 0-3 / 4-7 per socket (unverified)",
            "gfc_backends": GFC_BACKENDS,
            "excluded_paths": EXCLUDED_PATHS,
            "gfc_runtime_max_collective_bytes": ADAPTER_MAX_COLLECTIVE_BYTES,
            "p1_cases": len(p1), "p2_cases": len(p2), "p3_cases": len(p3),
            "p1": p1, "p2": p2, "p3": p3,
        },
        "planning_constants": PLAN,
        "estimate": estimate(rows, p1, p2, p3),
        "minimal_plan": minimal_plan(rows),
    }
    body = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    plan["plan_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return plan


def dumps(plan: dict[str, Any]) -> str:
    return json.dumps(plan, sort_keys=True, indent=1) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = build_plan()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gfc_nccl_plan.json").write_text(dumps(plan), encoding="utf-8")
    summary = {k: plan[k] for k in ("schema", "scope", "coverage", "estimate", "plan_sha256")}
    summary["cases"] = {k: plan["matrix"][k] for k in ("p1_cases", "p2_cases", "p3_cases")}
    summary["forms"] = [{k: r[k] for k in ("shape", "cfg", "sp", "legal", "tokens",
                                           "collectives_per_step", "bytes_per_step_per_rank",
                                           "max_message_bytes", "max_message_bytes_upper",
                                           "fits_smoke_1MiB", "fits_adapter_128MiB")}
                        for r in plan["message_table"]]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=1) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "plan_sha256": plan["plan_sha256"],
                      **summary["cases"], "estimate": plan["estimate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
