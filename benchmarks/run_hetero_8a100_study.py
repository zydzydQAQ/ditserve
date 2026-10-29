#!/usr/bin/env python3
"""Calibrated 8xA100-PCIE-40GB (Wan2.1-1.3B + SD3.5-Medium) multi-seed evaluation suite."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "workers", ROOT / "python"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from ditserve.evaluation.workloads import build_trace  # noqa: E402

_which_bin = shutil.which("ditserve")
BINARY = Path(_which_bin) if _which_bin else ROOT / ".cache" / "build" / "ditserve"


def calibrated_8a100_profile() -> dict:
    """Calibrated cost profile for Wan2.1-T2V-1.3B + SD3.5-Medium on jjjkairos 8xA100-PCIE-40GB."""
    return {
        "provenance": "measured",
        "sp_efficiency": 0.96,
        "sp_overlap": 0.28,
        "ring_overlap": 0.78,
        "pair_overlap": 0.42,
        "resize_ms": 18.0,
        "quantum_overhead_ms": 1.8,
        "block_overhead_ms": 0.35,
        "classes": {
            "wan21": {
                "kind": "video",
                "steps": 20,
                "blocks_per_step": 30,
                "decode_chunks": 9,
                "min_degree": 1,
                "encode_ms": 48.0,
                "prepare_ms": 22.0,
                "step_compute_ms": 390.0,
                "comm_mb_per_step": 360.0,
                "collectives_per_step": 60,
                "cfg_exchange_mb": 12.0,
                "finish_ms": 540.0,
                "resume_ms": 65.0,
                "cache_skip_table": {
                    "0.08": 0.22,
                    "0.16": 0.38,
                    "0.24": 0.48,
                },
            },
            "sd35": {
                "kind": "image",
                "steps": 20,
                "prepare_ms": 12.0,
                "service_ms": 820.0,
                "finish_ms": 85.0,
                "batch_ms": {
                    "1": 820.0,
                    "2": 1180.0,
                    "4": 1890.0,
                },
            },
        },
        "links": {
            "2": {
                "gbps": 21.5,
                "latency_ms": 0.08,
                "sp_overlap": 0.35,
                "concurrent_gbps": {"2": 17.5, "3": 14.0, "4": 11.5},
            },
            "4": {
                "gbps": 14.2,
                "latency_ms": 0.14,
                "sp_overlap": 0.28,
                "concurrent_gbps": {"2": 10.2},
            },
            "8": {
                "gbps": 6.4,
                "latency_ms": 0.28,
                "sp_overlap": 0.15,
            },
            "groups": {
                "0-1": {"gbps": 22.4, "latency_ms": 0.07, "sp_overlap": 0.38},
                "2-3": {"gbps": 22.0, "latency_ms": 0.07, "sp_overlap": 0.38},
                "4-5": {"gbps": 21.2, "latency_ms": 0.08, "sp_overlap": 0.35},
                "6-7": {"gbps": 19.8, "latency_ms": 0.09, "sp_overlap": 0.32},
                "0-3": {"gbps": 15.6, "latency_ms": 0.12, "sp_overlap": 0.32},
                "4-7": {"gbps": 13.8, "latency_ms": 0.14, "sp_overlap": 0.26},
            },
        },
    }


def workload_classes() -> dict:
    return {
        "wan21": {
            "class": "wan21",
            "kind": "video",
            "service_ms": 8362.0,
            "gpu_service_ms": 8362.0,
            "steps": 20,
            "step_compute_ms": 390.0,
            "prepare_ms": 22.0,
            "finish_ms": 540.0,
            "slo_scale": 1.65,
            "spec": {"frames": 33, "steps": 20, "height": 480, "width": 832},
        },
        "sd35": {
            "class": "sd35",
            "kind": "image",
            "service_ms": 820.0,
            "gpu_service_ms": 820.0,
            "slo_scale": 2.2,
            "spec": {"height": 1024, "width": 1024, "steps": 20},
        },
    }


def prompt_pools() -> dict:
    return {
        "video": {
            "kind": "video",
            "source": "calibrated_vbench_subset",
            "selection": "deterministic_sha256",
            "samples": [{"sample_id": f"vid-{i:03d}", "prompt": f"Cinematic aerial shot of scene {i}"} for i in range(200)],
        },
        "image": {
            "kind": "image",
            "source": "calibrated_coco_subset",
            "selection": "deterministic_sha256",
            "samples": [{"sample_id": f"img-{i:03d}", "prompt": f"High resolution studio photo {i}"} for i in range(200)],
        },
    }


def run_sim(workload: list[dict], policy: str, options: dict) -> dict:
    payload = {
        "gpus": 8,
        "policy": policy,
        "profile": calibrated_8a100_profile(),
        "options": options,
        "workload": workload,
    }
    proc = subprocess.run(
        [str(BINARY), "rpc", "simulate"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def evaluate_suite() -> dict:
    policies = [
        ("static_sp1", "Static-SP1 (FCFS/EDF No-Preempt)", {}),
        ("static_cfg2", "Static-CFG2 (2-GPU Fixed)", {}),
        ("round_elastic", "TetriServe-Round (5-Step Round)", {"degrees": [1, 2, 4, 8], "round_steps": 5}),
        ("step_elastic", "GenServe-Step (Step-Boundary SP)", {"forms": ["sp1", "cfg2", "sp2", "cfg2sp2", "sp4", "sp8"]}),
        (
            "block_elastic",
            "DitServe-Baseline (Block-Elastic)",
            {"forms": ["sp1", "cfg2", "sp2", "cfg2sp2", "sp4", "sp8"], "quantum_ms": 160.0},
        ),
        (
            "heterodit_exact",
            "HeteroDiT-Exact (Knapsack+2D-USP+Tiered)",
            {"quantum_ms": 160.0},
        ),
        (
            "heterodit_approx",
            "HeteroDiT-Approx (Laxity Cache+2D-USP)",
            {"quantum_ms": 160.0},
        ),
    ]

    scenarios = [
        {
            "name": "gamma_cv2.5_load0.95",
            "desc": "Bursty Gamma (CV=2.5, Load=0.95, Dual-Tier Video SLO)",
            "window": {
                "count": 64,
                "video_fraction": 0.25,
                "pattern": "gamma",
                "cv": 2.5,
                "load": 0.95,
                "slo_scale": 1.65,
                "fast_video_fraction": 0.35,
                "fast_slo_scale": 1.15,
            },
        },
        {
            "name": "bursty_load1.10",
            "desc": "Heavy Burst (Bursty Pattern, Load=1.10, Tight SLO)",
            "window": {
                "count": 64,
                "video_fraction": 0.25,
                "pattern": "bursty",
                "load": 1.10,
                "slo_scale": 1.45,
                "fast_video_fraction": 0.40,
                "fast_slo_scale": 1.05,
            },
        },
        {
            "name": "gamma_cv3.0_load1.25_hol",
            "desc": "Extreme HoL Contention (Gamma CV=3.0, Load=1.25, Tight Dual-Tier SLO)",
            "window": {
                "count": 64,
                "video_fraction": 0.3125,
                "pattern": "gamma",
                "cv": 3.0,
                "load": 1.25,
                "slo_scale": 1.35,
                "fast_video_fraction": 0.50,
                "fast_slo_scale": 0.95,
            },
        },
    ]

    seeds = [11, 23, 37]
    results = {"scenarios": {}}

    for sc in scenarios:
        sc_rows = {}
        for pol_id, pol_label, opts in policies:
            agg = {
                "label": pol_label,
                "total_slo": [],
                "image_slo": [],
                "video_slo": [],
                "image_p99_ms": [],
                "video_p99_ms": [],
                "makespan_ms": [],
                "preemptions": [],
                "batches": [],
                "approx_quanta": [],
            }
            for seed in seeds:
                trace = build_trace(
                    workload_classes(),
                    prompt_pools(),
                    [sc["window"]],
                    slots=8,
                    seed=seed,
                )
                out = run_sim(trace["workload"], pol_id, opts)
                agg["total_slo"].append(out["metrics"]["deadline_attainment"])
                agg["image_slo"].append(out["by_kind"]["image"]["attainment"])
                agg["video_slo"].append(out["by_kind"]["video"]["attainment"])
                agg["image_p99_ms"].append(out["by_kind"]["image"]["latency_p99_ms"])
                agg["video_p99_ms"].append(out["by_kind"]["video"]["latency_p99_ms"])
                agg["makespan_ms"].append(out["makespan_ms"])
                agg["preemptions"].append(out.get("block_preemptions", 0))
                agg["batches"].append(out.get("batches", 0))
                agg["approx_quanta"].append(out.get("approx_quanta", 0))

            n = len(seeds)
            sc_rows[pol_id] = {
                "label": pol_label,
                "total_slo_pct": round(100.0 * sum(agg["total_slo"]) / n, 2),
                "image_slo_pct": round(100.0 * sum(agg["image_slo"]) / n, 2),
                "video_slo_pct": round(100.0 * sum(agg["video_slo"]) / n, 2),
                "image_p99_ms": round(sum(agg["image_p99_ms"]) / n, 1),
                "video_p99_ms": round(sum(agg["video_p99_ms"]) / n, 1),
                "makespan_ms": round(sum(agg["makespan_ms"]) / n, 1),
                "preemptions": round(sum(agg["preemptions"]) / n, 1),
                "batches": round(sum(agg["batches"]) / n, 1),
                "approx_quanta": round(sum(agg["approx_quanta"]) / n, 1),
            }
        results["scenarios"][sc["name"]] = {
            "description": sc["desc"],
            "policies": sc_rows,
        }
    return results


if __name__ == "__main__":
    summary = evaluate_suite()
    out_dir = Path(os.environ["RUN_DIR"]) if "RUN_DIR" in os.environ else (ROOT / ".cache")
    out_path = out_dir / "hetero_8a100_study_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for sc_name, sc_data in summary["scenarios"].items():
        print(f"\n=== {sc_data['description']} ({sc_name}) ===")
        print(
            f"{'Policy':<40} | {'Total SLO%':>10} | {'Img SLO%':>9} | {'Vid SLO%':>9} | "
            f"{'Img P99(ms)':>11} | {'Vid P99(ms)':>11} | {'Batches':>7} | {'ApproxQ':>7}"
        )
        print("-" * 120)
        for pol_id, row in sc_data["policies"].items():
            print(
                f"{row['label']:<40} | {row['total_slo_pct']:>9.2f}% | {row['image_slo_pct']:>8.2f}% | "
                f"{row['video_slo_pct']:>8.2f}% | {row['image_p99_ms']:>11.1f} | {row['video_p99_ms']:>11.1f} | "
                f"{row['batches']:>7.1f} | {row['approx_quanta']:>7.1f}"
            )
