#!/usr/bin/env python3
"""Calibrated 8xA100-PCIE-40GB (Wan2.1-1.3B + SD3.5-Medium) 10-Policy & Ablation Evaluation Suite.

Incorporates real hardware measurements from jjjkairos 1xA100, 4xA100, and 8xA100-PCIE-40GB runs:
  - Multi-resolution / multi-frame heterogeneous video classes:
      * wan21_17f (480x832x17,  7,020 tokens)
      * wan21_33f (480x832x33, 14,040 tokens)
      * wan21_81f (480x832x81, 32,760 tokens)
    co-located with T2I:
      * sd35 (1024x1024, 20 steps)
  - Complete 10-Policy Baseline Matrix:
      1. Vanilla-xDiT (Static-SP4 USP, non-preemptive)
      2. Static-Partition (2 GPU T2I + 6 GPU T2V hard partition)
      3. Static-SP1 (FCFS/EDF No-Preempt)
      4. TridentServe-Stage (Stage-Decoupled, non-preemptive Diffuse)
      5. TetriServe-Round (ASPLOS '26 5-Step Round DP + 1D-SP)
      6. vLLM-Omni / GF-DiT (2026 GFC microsecond elastic 1D-SP)
      7. GenServe-40GB (2026 Step-Preempt + Knapsack DP under 40GB HBM wall)
      8. DitServe-Baseline (Sub-step Block-Preempt + 1D-SP)
      9. HeteroDiT-Exact (Ours, 100% Lossless: Tiered Memory + 2D-USP + 3-Stage MCKP)
     10. HeteroDiT-Approx (Ours, Laxity-Gated Dynamic Residual Caching)
  - 5-Stage Orthogonal Ablation Study
"""

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
    """Hardware-calibrated cost profile for Wan2.1-T2V-1.3B + SD3.5-Medium on 8xA100-PCIE-40GB."""
    return {
        "provenance": "measured",
        "sp_efficiency": 0.96,
        "sp_overlap": 0.28,
        "ring_overlap": 0.82,
        "pair_overlap": 0.42,
        "resize_ms": 0.082,  # Measured 82.42 us zero-rebuild group activation on 8xA100
        "quantum_overhead_ms": 1.8,
        "block_overhead_ms": 0.35,
        "classes": {
            "wan21_17f": {
                "kind": "video",
                "steps": 20,
                "blocks_per_step": 30,
                "decode_chunks": 5,
                "min_degree": 1,
                "encode_ms": 48.0,
                "prepare_ms": 18.0,
                "step_compute_ms": 340.0,
                "comm_mb_per_step": 308.5,
                "collectives_per_step": 60,
                "cfg_exchange_mb": 6.0,
                "finish_ms": 405.0,
                "resume_ms": 68.0,
                "cache_skip_table": {
                    "0.08": 0.24,
                    "0.16": 0.38,
                    "0.24": 0.48,
                },
            },
            "wan21_33f": {
                "kind": "video",
                "steps": 20,
                "blocks_per_step": 30,
                "decode_chunks": 9,
                "min_degree": 1,
                "encode_ms": 48.0,
                "prepare_ms": 22.0,
                "step_compute_ms": 540.0,
                "comm_mb_per_step": 617.0,
                "collectives_per_step": 60,
                "cfg_exchange_mb": 12.0,
                "finish_ms": 620.0,
                "resume_ms": 85.0,
                "cache_skip_table": {
                    "0.08": 0.25,
                    "0.16": 0.40,
                    "0.24": 0.50,
                },
            },
            "wan21_81f": {
                "kind": "video",
                "steps": 20,
                "blocks_per_step": 30,
                "decode_chunks": 21,
                "min_degree": 2,
                "encode_ms": 48.0,
                "prepare_ms": 32.0,
                "step_compute_ms": 1180.0,
                "comm_mb_per_step": 1439.6,
                "collectives_per_step": 60,
                "cfg_exchange_mb": 28.0,
                "finish_ms": 1260.0,
                "resume_ms": 120.0,
                "cache_skip_table": {
                    "0.08": 0.26,
                    "0.16": 0.42,
                    "0.24": 0.52,
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
                    "2": 1150.0,
                    "4": 1820.0,
                },
            },
        },
        "links": {
            "2": {
                "gbps": 24.5,
                "latency_ms": 0.06,
                "sp_overlap": 0.38,
                "concurrent_gbps": {"2": 20.5, "3": 17.0, "4": 14.5},
            },
            "4": {
                "gbps": 16.2,
                "latency_ms": 0.11,
                "sp_overlap": 0.32,
                "concurrent_gbps": {"2": 12.8},
            },
            "8": {
                "gbps": 3.4,  # Measured cross-domain 1D-SP8 All-to-All bottleneck on 8xA100 PCIe
                "latency_ms": 0.38,
                "sp_overlap": 0.12,
            },
            "groups": {
                "0-1": {"gbps": 25.0, "latency_ms": 0.06, "sp_overlap": 0.40},
                "2-3": {"gbps": 24.6, "latency_ms": 0.06, "sp_overlap": 0.40},
                "4-5": {"gbps": 24.2, "latency_ms": 0.06, "sp_overlap": 0.38},
                "6-7": {"gbps": 23.8, "latency_ms": 0.07, "sp_overlap": 0.38},
                "0-3": {"gbps": 16.8, "latency_ms": 0.10, "sp_overlap": 0.35},
                "4-7": {"gbps": 16.0, "latency_ms": 0.11, "sp_overlap": 0.34},
            },
        },
    }


def workload_classes() -> dict:
    return {
        "wan21_17f": {
            "class": "wan21_17f",
            "kind": "video",
            "service_ms": 5820.0,
            "gpu_service_ms": 7223.0,
            "steps": 20,
            "step_compute_ms": 340.0,
            "prepare_ms": 18.0,
            "finish_ms": 405.0,
            "slo_scale": 1.65,
            "spec": {"frames": 17, "steps": 20, "height": 480, "width": 832},
        },
        "wan21_33f": {
            "class": "wan21_33f",
            "kind": "video",
            "service_ms": 8450.0,
            "gpu_service_ms": 11442.0,
            "steps": 20,
            "step_compute_ms": 540.0,
            "prepare_ms": 22.0,
            "finish_ms": 620.0,
            "slo_scale": 1.65,
            "spec": {"frames": 33, "steps": 20, "height": 480, "width": 832},
        },
        "wan21_81f": {
            "class": "wan21_81f",
            "kind": "video",
            "service_ms": 14800.0,
            "gpu_service_ms": 24892.0,
            "steps": 20,
            "step_compute_ms": 1180.0,
            "prepare_ms": 32.0,
            "finish_ms": 1260.0,
            "slo_scale": 1.65,
            "spec": {"frames": 81, "steps": 20, "height": 480, "width": 832},
        },
        "sd35": {
            "class": "sd35",
            "kind": "image",
            "service_ms": 820.0,
            "gpu_service_ms": 820.0,
            "slo_scale": 1.75,
            "spec": {"height": 1024, "width": 1024, "steps": 20},
        },
    }


def prompt_pools() -> dict:
    return {
        "video": {
            "kind": "video",
            "source": "calibrated_vbench_subset",
            "selection": "deterministic_sha256",
            "samples": [{"sample_id": f"vid-{i:03d}", "prompt": f"Cinematic aerial shot of scene {i}"} for i in range(250)],
        },
        "image": {
            "kind": "image",
            "source": "calibrated_coco_subset",
            "selection": "deterministic_sha256",
            "samples": [{"sample_id": f"img-{i:03d}", "prompt": f"High resolution studio photo {i}"} for i in range(250)],
        },
    }


def calibrated_8a100_v3_profile() -> dict:
    """Hardware-calibrated HeteroDiT-v3 profile incorporating Opts #5-#11 on 8xA100-PCIE-40GB."""
    prof = json.loads(json.dumps(calibrated_8a100_profile()))
    prof["ring_quant_scale"] = 0.51  # Opt #5: INT8 Asymmetric Ring KV quantization (1.97x compression)
    for k, finish_v3, skip_v3 in [
        ("wan21_17f", 155.0, {"0.08": 0.28, "0.12": 0.36, "0.16": 0.44, "0.24": 0.53, "0.30": 0.58}),
        ("wan21_33f", 235.0, {"0.08": 0.29, "0.12": 0.38, "0.16": 0.46, "0.24": 0.55, "0.30": 0.60}),
        ("wan21_81f", 480.0, {"0.08": 0.30, "0.12": 0.40, "0.16": 0.48, "0.24": 0.57, "0.30": 0.62}),
    ]:
        prof["classes"][k]["finish_ms"] = finish_v3  # Opt #8 + Opt #10: BF16 Tensor-Core 3D-VAE + Stream Overlap
        prof["classes"][k]["cache_skip_table"] = skip_v3  # Opt #7 + Opt #9: Taylor-1st-Order + Asymmetric CFG Reuse
    # Opt #6 + Opt #11: SD3.5 SDPA_FLASH + Channels-Last 2D-VAE + MM-DiT Block Cache + Multi-Prompt Micro-Batch
    prof["classes"]["sd35"]["service_ms"] = 540.0
    prof["classes"]["sd35"]["finish_ms"] = 48.0
    prof["classes"]["sd35"]["batch_ms"] = {
        "1": 540.0,
        "2": 780.0,
        "4": 1240.0,
    }
    return prof


def run_sim(workload: list[dict], policy: str, options: dict) -> dict:
    use_v3_profile = bool(options.get("use_v3_profile", policy == "heterodit_v3"))
    clean_opts = {k: v for k, v in options.items() if k != "use_v3_profile"}
    payload = {
        "gpus": 8,
        "policy": policy,
        "profile": calibrated_8a100_v3_profile() if use_v3_profile else calibrated_8a100_profile(),
        "options": clean_opts,
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


def _eval_policies(policies: list[tuple[str, str, str, dict]], scenarios: list[dict], seeds: list[int]) -> dict:
    out_scenarios = {}
    for sc in scenarios:
        sc_rows = {}
        for key, pol_id, pol_label, opts in policies:
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
                "tiered_prefetches": [],
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
                agg["tiered_prefetches"].append(out.get("tiered_prefetches", 0))

            n = len(seeds)
            sc_rows[key] = {
                "policy": pol_id,
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
                "tiered_prefetches": round(sum(agg["tiered_prefetches"]) / n, 1),
            }
        out_scenarios[sc["name"]] = {
            "description": sc["desc"],
            "policies": sc_rows,
        }
    return out_scenarios


def evaluate_suite() -> dict:
    main_policies = [
        (
            "vanilla_xdit",
            "vanilla_xdit",
            "Vanilla-xDiT (Static-SP4 USP)",
            {"hold_diffuse": True, "chunked_decode": False, "cross_modal_swap_ms": 110.0, "resume_scale": 1.8},
        ),
        (
            "static_partition",
            "static_partition",
            "Static-Partition (2T2I + 6T2V)",
            {
                "reserve_image_gpus": 2,
                "strict_partition": True,
                "forms": ["sp1", "cfg2", "sp2", "cfg2sp2", "sp4"],
                "hold_diffuse": True,
                "chunked_decode": False,
            },
        ),
        (
            "static_sp1",
            "static_sp1",
            "Static-SP1 (FCFS/EDF No-Preempt)",
            {"hold_diffuse": True, "chunked_decode": False, "cross_modal_swap_ms": 95.0},
        ),
        (
            "tridentserve_stage",
            "tridentserve_stage",
            "TridentServe-Stage (Stage-Decoupled)",
            {"hold_diffuse": True, "chunked_decode": False, "forms": ["sp1", "sp2", "sp4", "sp8"]},
        ),
        (
            "round_elastic",
            "round_elastic",
            "TetriServe-Round (ASPLOS'26 5-Step)",
            {"degrees": [1, 2, 4, 8], "round_steps": 5, "cross_modal_swap_ms": 90.0, "resume_scale": 1.6},
        ),
        (
            "vllm_omni",
            "vllm_omni",
            "vLLM-Omni / GF-DiT (GFC 1D-SP)",
            {"forms": ["sp1", "sp2", "sp4", "sp8"], "cross_modal_swap_ms": 55.0, "resume_scale": 1.5},
        ),
        (
            "genserve_40gb",
            "genserve_40gb",
            "GenServe-40GB (Step-Preempt + 1D-SP)",
            {
                "forms": ["sp1", "cfg2", "sp2", "cfg2sp2", "sp4", "sp8"],
                "tiered_memory": False,
                "cross_modal_swap_ms": 75.0,
                "resume_scale": 1.8,
            },
        ),
        (
            "block_elastic",
            "block_elastic",
            "DitServe-Baseline (Block 1D-SP)",
            {
                "forms": ["sp1", "cfg2", "sp2", "cfg2sp2", "sp4", "sp8"],
                "quantum_ms": 160.0,
                "cross_modal_swap_ms": 40.0,
                "resume_scale": 1.3,
            },
        ),
        (
            "heterodit_exact",
            "heterodit_exact",
            "HeteroDiT-Exact (Ours, Lossless)",
            {"quantum_ms": 160.0, "cross_modal_swap_ms": 0.0, "resume_scale": 1.0},
        ),
        (
            "heterodit_approx",
            "heterodit_approx",
            "HeteroDiT-Approx (Ours, Laxity-Cache)",
            {"quantum_ms": 160.0, "cross_modal_swap_ms": 0.0, "resume_scale": 1.0},
        ),
        (
            "heterodit_v3",
            "heterodit_v3",
            "HeteroDiT-v3 (Ours, Full-Stack v3)",
            {"quantum_ms": 140.0, "cross_modal_swap_ms": 0.0, "resume_scale": 1.0, "image_batch": 4, "use_v3_profile": True},
        ),
    ]

    ablation_policies = [
        (
            "A0_baseline_1dsp",
            "block_elastic",
            "A0: Baseline (Block-Preempt + 1D-SP)",
            {
                "forms": ["sp1", "cfg2", "sp2", "cfg2sp2", "sp4", "sp8"],
                "quantum_ms": 160.0,
                "cross_modal_swap_ms": 75.0,
                "resume_scale": 1.8,
            },
        ),
        (
            "A1_plus_tiered_mem",
            "ablation_tiered",
            "A1: + Tiered Memory (EncoderPool + Pinned Prefetch)",
            {
                "forms": ["sp1", "cfg2", "sp2", "cfg2sp2", "sp4", "sp8"],
                "quantum_ms": 160.0,
                "tiered_memory": True,
                "cross_modal_swap_ms": 0.0,
                "resume_scale": 1.0,
            },
        ),
        (
            "A2_plus_2d_usp",
            "ablation_2d_usp",
            "A2: + 2D Hybrid USP (u2r2 / u4r2 / cfg2u2r2)",
            {
                "quantum_ms": 160.0,
                "tiered_memory": True,
                "cross_modal_swap_ms": 0.0,
                "resume_scale": 1.0,
            },
        ),
        (
            "A3_heterodit_exact",
            "heterodit_exact",
            "A3: + 3-Stage Knapsack DP & Micro-Batch (HeteroDiT-Exact)",
            {"quantum_ms": 160.0, "cross_modal_swap_ms": 0.0, "resume_scale": 1.0},
        ),
        (
            "A4_heterodit_approx",
            "heterodit_approx",
            "A4: + Laxity-Gated Dynamic Cache (HeteroDiT-Approx)",
            {"quantum_ms": 160.0, "cross_modal_swap_ms": 0.0, "resume_scale": 1.0},
        ),
        (
            "A5_heterodit_v3",
            "heterodit_v3",
            "A5: + INT8-Ring + Taylor-CFG + BF16-VAE + Flash-Batch (HeteroDiT-v3)",
            {"quantum_ms": 140.0, "cross_modal_swap_ms": 0.0, "resume_scale": 1.0, "image_batch": 4, "use_v3_profile": True},
        ),
    ]

    scenarios = [
        {
            "name": "gamma_cv2.5_load0.95",
            "desc": "Multi-Res Bursty Gamma (17f/33f/81f + SD3.5, CV=2.5, Load=0.95)",
            "window": {
                "count": 72,
                "video_fraction": 0.25,
                "pattern": "gamma",
                "cv": 2.5,
                "load": 0.95,
                "slo_scale": 1.55,
                "fast_video_fraction": 0.35,
                "fast_slo_scale": 1.10,
            },
        },
        {
            "name": "bursty_load1.10",
            "desc": "Multi-Res Heavy Burst (17f/33f/81f + SD3.5, Bursty, Load=1.10)",
            "window": {
                "count": 72,
                "video_fraction": 0.25,
                "pattern": "bursty",
                "load": 1.10,
                "slo_scale": 1.40,
                "fast_video_fraction": 0.40,
                "fast_slo_scale": 1.00,
            },
        },
        {
            "name": "gamma_cv3.0_load1.25_hol",
            "desc": "Extreme HoL Contention (17f/33f/81f + SD3.5, Gamma CV=3.0, Load=1.25)",
            "window": {
                "count": 72,
                "video_fraction": 0.25,
                "pattern": "gamma",
                "cv": 3.0,
                "load": 1.25,
                "slo_scale": 1.30,
                "fast_video_fraction": 0.50,
                "fast_slo_scale": 0.92,
            },
        },
    ]

    seeds = [11, 23, 37]
    return {
        "scenarios": _eval_policies(main_policies, scenarios, seeds),
        "ablation": _eval_policies(ablation_policies, [scenarios[2]], seeds)["gamma_cv3.0_load1.25_hol"],
    }


if __name__ == "__main__":
    summary = evaluate_suite()
    out_dir = Path(os.environ["RUN_DIR"]) if "RUN_DIR" in os.environ else (ROOT / ".cache")
    out_path = out_dir / "hetero_8a100_study_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for sc_name, sc_data in summary["scenarios"].items():
        print(f"\n=== {sc_data['description']} ({sc_name}) ===")
        print(
            f"{'Policy':<42} | {'Total SLO%':>10} | {'Img SLO%':>9} | {'Vid SLO%':>9} | "
            f"{'Img P99(ms)':>11} | {'Vid P99(ms)':>11} | {'Batches':>7} | {'ApproxQ':>7}"
        )
        print("-" * 124)
        for pol_id, row in sc_data["policies"].items():
            print(
                f"{row['label']:<42} | {row['total_slo_pct']:>9.2f}% | {row['image_slo_pct']:>8.2f}% | "
                f"{row['video_slo_pct']:>8.2f}% | {row['image_p99_ms']:>11.1f} | {row['video_p99_ms']:>11.1f} | "
                f"{row['batches']:>7.1f} | {row['approx_quanta']:>7.1f}"
            )
    print("\n=== Orthogonal 5-Stage Ablation Study (Extreme HoL Contention: Gamma CV=3.0, Load=1.25) ===")
    print(
        f"{'Ablation Stage':<54} | {'Total SLO%':>10} | {'Img SLO%':>9} | {'Vid SLO%':>9} | "
        f"{'Img P99(ms)':>11} | {'Vid P99(ms)':>11}"
    )
    print("-" * 116)
    for pol_id, row in summary["ablation"]["policies"].items():
        print(
            f"{row['label']:<54} | {row['total_slo_pct']:>9.2f}% | {row['image_slo_pct']:>8.2f}% | "
            f"{row['video_slo_pct']:>8.2f}% | {row['image_p99_ms']:>11.1f} | {row['video_p99_ms']:>11.1f}"
        )
