#!/usr/bin/env python3
"""Generate publication-ready figures (.pdf + .png) and LaTeX booktabs tables for HeteroDiT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
})


def plot_fig1_10policy_and_ablation(study: Dict[str, Any], out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.7), gridspec_kw={"width_ratios": [1.55, 1.0]})

    balanced = study["scenarios"]["gamma_cv2.5_load0.95"]["policies"]
    policies = [
        "static_partition",
        "vanilla_xdit",
        "static_sp1",
        "tridentserve_stage",
        "round_elastic",
        "vllm_omni",
        "genserve_40gb",
        "block_elastic",
        "heterodit_exact",
        "heterodit_approx",
    ]
    labels = [
        "Static\n(2I+6V)",
        "Vanilla\nxDiT",
        "Static\nSP1",
        "Trident\nServe",
        "TetriServe\n(Round)",
        "vLLM-Omni\n(GF-DiT)",
        "GenServe\n(40GB)",
        "DitServe\n(1D-SP)",
        "HeteroDiT\n(Exact)",
        "HeteroDiT\n(Approx)",
    ]
    total_slo = [balanced[p]["total_slo_pct"] for p in policies]
    vid_slo = [balanced[p]["video_slo_pct"] for p in policies]
    img_slo = [balanced[p]["image_slo_pct"] for p in policies]

    x = np.arange(len(policies))
    w = 0.26
    ax1.bar(x - w, total_slo, width=w, label="Total SLO (%)", color="#1f4e78", edgecolor="black", linewidth=0.5)
    ax1.bar(x, vid_slo, width=w, label="Video SLO (%)", color="#2e75b6", edgecolor="black", linewidth=0.5)
    ax1.bar(x + w, img_slo, width=w, label="Image SLO (%)", color="#70ad47", edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=22, ha="right")
    ax1.set_ylabel("SLO Attainment (%)")
    ax1.set_ylim(0, 112)
    ax1.set_title("(a) 8xA100 10-Policy Baseline Comparison (Multi-Res Gamma CV=2.5)")
    ax1.legend(loc="upper left", ncol=3)

    # Panel B: 5-Stage Orthogonal Ablation
    ablation = study["ablation"]["policies"]
    stage_keys = [
        "A0_baseline_1dsp",
        "A1_plus_tiered_mem",
        "A2_plus_2d_usp",
        "A3_heterodit_exact",
        "A4_heterodit_approx",
    ]
    stage_labels = [
        "A0: Baseline\n(1D-SP)",
        "A1: +EncoderPool\n& Tiered RAM",
        "A2: +2D Hybrid\nUSP (u4r2/cfg)",
        "A3: +Knapsack DP\n& uBatch (Exact)",
        "A4: +Laxity Cache\n(HeteroDiT-Approx)",
    ]
    s_total = [ablation[k]["total_slo_pct"] for k in stage_keys]
    s_vid_p99 = [ablation[k]["video_p99_ms"] / 1000.0 for k in stage_keys]

    sx = np.arange(len(stage_keys))
    bars = ax2.bar(
        sx,
        s_total,
        width=0.50,
        color=["#8c8c8c", "#5b9bd5", "#4472c4", "#2f5597", "#c55a11"],
        edgecolor="black",
        linewidth=0.6,
    )
    for b, val in zip(bars, s_total):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            val + 1.4,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_xticks(sx)
    ax2.set_xticklabels(stage_labels, rotation=20, ha="right")
    ax2.set_ylabel("Total SLO Attainment (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("(b) 5-Stage Orthogonal Ablation (Extreme Load=1.25)")

    ax2_r = ax2.twinx()
    ax2_r.plot(sx, s_vid_p99, color="#d9534f", marker="o", linewidth=2.0, label="Video P99 (s)")
    ax2_r.set_ylabel("Video P99 Latency (s)", color="#d9534f")
    ax2_r.tick_params(axis="y", labelcolor="#d9534f")
    ax2_r.set_ylim(10, 42)
    ax2_r.grid(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_10policy_slo_pareto.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_nccl_and_int8_ring(nccl8: Dict[str, Any], v2: Dict[str, Any], out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.5))

    # Panel A: 8-GPU 1D-Ulysses (u8r1 padded H=16) vs 2D-Hybrid-USP (u4r2) vs 2D-Hybrid-CFG (cfg2u2r2)
    micro = nccl8["nccl_collective_microbenchmarks"]
    shapes = ["480x832x17", "480x832x33", "720x1280x17", "480x832x81"]
    shape_labels = ["480p-17f\n(7,020 tok)", "480p-33f\n(14,040 tok)", "720p-17f\n(18,000 tok)", "480p-81f\n(32,760 tok)"]

    u8r1_ms = [
        next(r["step_comm_plus_attn_ms"] for r in micro if r["shape"] == s and "u8r1" in r["mode"])
        for s in shapes
    ]
    u4r2_ms = [
        next(r["step_comm_plus_attn_ms"] for r in micro if r["shape"] == s and "u4r2, Native" in r["mode"])
        for s in shapes
    ]
    cfg_ms = [
        next(r["step_comm_plus_attn_ms"] for r in micro if r["shape"] == s and "cfg2u2r2" in r["mode"])
        for s in shapes
    ]

    x = np.arange(len(shapes))
    w = 0.26
    ax1.bar(x - w, u8r1_ms, width=w, label="1D-Ulysses-SP8 (u8r1, Padded H=16)", color="#d9534f", edgecolor="black", linewidth=0.5)
    ax1.bar(x, u4r2_ms, width=w, label="2D-Hybrid-USP-8GPU (u4r2, Native H=12)", color="#2e75b6", edgecolor="black", linewidth=0.5)
    ax1.bar(x + w, cfg_ms, width=w, label="2D-Hybrid-CFG-8GPU (cfg2u2r2, Native H=12)", color="#70ad47", edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(shape_labels)
    ax1.set_ylabel("8-GPU Step Comm + Attention Latency (ms)")
    ax1.set_title("(a) 8-GPU 2D-USP vs 1D-Ulysses (H=12 Head-Divisor Trap)")
    ax1.legend(loc="upper left")

    # Panel B: Opt #1 Asymmetric INT8 Ring KV Compression
    ring_cfgs = v2["opt1_ring_int8"]["configs"]
    cfg_names = [c["config"].replace("_", "\n") for c in ring_cfgs]
    fp16_comm = [c["cross_node_ring_comm_step_fp16_ms"] for c in ring_cfgs]
    int8_comm = [c["cross_node_ring_comm_step_int8_ms"] for c in ring_cfgs]

    rx = np.arange(len(ring_cfgs))
    rw = 0.34
    ax2.bar(rx - rw / 2, fp16_comm, width=rw, label="Standard FP16 Ring P2P", color="#8c8c8c", edgecolor="black", linewidth=0.5)
    b2 = ax2.bar(
        rx + rw / 2,
        int8_comm,
        width=rw,
        label="Opt #1: INT8 Ring Packet (1.97x Comp)",
        color="#c55a11",
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, c in zip(b2, ring_cfgs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 8.0,
            f"{c['cross_node_comm_speedup']:.2f}x\ncos={c['attn_output_cos_sim']:.5f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax2.set_xticks(rx)
    ax2.set_xticklabels(cfg_names)
    ax2.set_ylabel("Cross-Node Ring P2P Comm per Step (ms)")
    ax2.set_ylim(0, max(fp16_comm) * 1.22)
    ax2.set_title("(b) Opt #1: Asymmetric INT8 Ring KV Compression")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_nccl_2d_usp_and_int8_ring.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_taylor_cache_and_sd35_microbatch(v2: Dict[str, Any], out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.5))

    # Panel A: Zero-Order DBCache vs First-Order Taylor-Compensated DBCache
    rows = v2["opt3_taylor_cache"]
    zero_rows = [r for r in rows if r["mode"] == "ZeroOrder-DBCache"]
    taylor_rows = [r for r in rows if r["mode"] == "Taylor1stOrder-DBCache"]

    taus = [f"tau={r['threshold']:.2f}\n({r['skip_ratio_pct']:.0f}% skip)" for r in zero_rows]
    z_psnr = [r["video_psnr_db"] for r in zero_rows]
    t_psnr = [r["video_psnr_db"] for r in taylor_rows]

    x = np.arange(len(taus))
    w = 0.32
    ax1.bar(x - w / 2, z_psnr, width=w, label="Zero-Order DBCache (Standard)", color="#5b9bd5", edgecolor="black", linewidth=0.5)
    bars_t = ax1.bar(
        x + w / 2,
        t_psnr,
        width=w,
        label="Opt #3: 1st-Order Taylor Extrapolation",
        color="#2e75b6",
        edgecolor="black",
        linewidth=0.5,
    )
    for b, zr, tr in zip(bars_t, zero_rows, taylor_rows):
        gain = tr["video_psnr_db"] - zr["video_psnr_db"]
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 0.12,
            f"{tr['video_psnr_db']:.2f} dB\n({gain:+.2f} dB)",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(taus)
    ax1.set_ylabel("Decoded Video PSNR vs Exact (dB)")
    ax1.set_ylim(30.0, max(max(z_psnr), max(t_psnr)) + 1.4)
    ax1.set_title("(a) Opt #3: Video Quality (PSNR) across Cache Thresholds")
    ax1.legend(loc="lower left")

    # Panel B: Opt #2 SD3.5-Medium SDPA_FLASH + True Multi-Prompt Heterogeneous Micro-Batching
    mb_rows = v2["opt2_sd35_flash_microbatch"]["four_distinct_prompts_burst"]
    mb_labels = [
        "Seq 4xB1\n(SDPA_MATH)",
        "Seq 4xB1\n(SDPA_FLASH)",
        "uBatch 2xB2\n(SDPA_FLASH)",
        "uBatch 1xB4\n(SDPA_FLASH)",
    ]
    per_img_ms = [r["per_image_ms"] for r in mb_rows]
    speedups = [r["speedup_vs_seq_math"] for r in mb_rows]

    mx = np.arange(len(mb_rows))
    mb_bars = ax2.bar(
        mx,
        per_img_ms,
        width=0.5,
        color=["#d9534f", "#5b9bd5", "#2e75b6", "#70ad47"],
        edgecolor="black",
        linewidth=0.6,
    )
    for b, ms, sp in zip(mb_bars, per_img_ms, speedups):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            ms + 45.0,
            f"{ms:.1f} ms\n({sp:.2f}x)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_xticks(mx)
    ax2.set_xticklabels(mb_labels)
    ax2.set_ylabel("Amortized Latency per Image (ms)")
    ax2.set_ylim(0, max(per_img_ms) * 1.22)
    ax2.set_title("(b) Opt #2: SD3.5 FlashAttention + Multi-Prompt uBatching")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_quality_pareto_taylor_vs_zero_order.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_fig4_e2e_replay_and_vae_overlap(q_replay: Dict[str, Any], v2: Dict[str, Any], out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.38, 1.0]})

    # Panel A: Real 1xA100 Mixed-Modality Trace Replay (5 Policies including HeteroDiT-v2)
    base_rows = q_replay["real_mixed_trace_replay"]
    e2e_v2 = v2["e2e_v2_replay"]
    p_labels = [
        "Vanilla-xDiT\n(Run-to-Comp)",
        "GenServe-40GB\n(Cold Offload)",
        "HeteroDiT-v1\n(Exact)",
        "HeteroDiT-v1\n(Approx)",
        "HeteroDiT-v2\n(All Opts)",
    ]
    img_p99 = [r["image_p99_ms"] / 1000.0 for r in base_rows] + [e2e_v2["image_p99_latency_ms"] / 1000.0]
    vid_lat = [r["video_latency_ms"] / 1000.0 for r in base_rows] + [e2e_v2["video_latency_ms"] / 1000.0]
    slo_pct = [r["total_slo_pct"] for r in base_rows] + [e2e_v2["total_slo_attainment_pct"]]

    x = np.arange(len(p_labels))
    w = 0.34
    ax1.bar(x - w / 2, img_p99, width=w, label="Image P99 Latency (s) [SLO <= 4.5s]", color="#70ad47", edgecolor="black", linewidth=0.5)
    ax1.bar(x + w / 2, vid_lat, width=w, label="Video Latency (s) [SLO <= 17.0s]", color="#2e75b6", edgecolor="black", linewidth=0.5)
    ax1.axhline(4.5, color="#70ad47", linestyle=":", linewidth=1.2)
    ax1.axhline(17.0, color="#2e75b6", linestyle="--", linewidth=1.2)
    for i, sp in enumerate(slo_pct):
        top = max(img_p99[i], vid_lat[i])
        ax1.text(x[i], top + 0.45, f"SLO: {sp:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(p_labels, rotation=15, ha="right")
    ax1.set_ylabel("End-to-End Request Latency (s)")
    ax1.set_ylim(0, max(vid_lat) * 1.18)
    ax1.set_title("(a) Real 1xA100 Mixed-Workload Trace Replay")
    ax1.legend(loc="upper right")

    # Panel B: Opt #4 Stream-Isolated 3D-VAE Chunked Decode Overlap with SD3.5 Denoising
    ovl = v2["opt4_vae_stream_overlap"]
    modes = ["Sequential\n(3D-VAE + SD3.5)", "Opt #4: Stream-Isolated\nConcurrent Overlap"]
    wall_vals = [ovl["sequential_wall_ms"], ovl["overlapped_wall_ms"]]
    bars_o = ax2.bar([0, 1], wall_vals, width=0.46, color=["#8c8c8c", "#c55a11"], edgecolor="black", linewidth=0.6)
    for b, v in zip(bars_o, wall_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            v + 35.0,
            f"{v:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax2.annotate(
        f"Saved {ovl['saved_ms']:.1f} ms\n({ovl['speedup']:.2f}x speedup)\nPeak HBM: {ovl['overlapped_peak_hbm_mb']:.0f} MB",
        xy=(1, ovl["overlapped_wall_ms"]),
        xytext=(0.12, ovl["sequential_wall_ms"] * 0.72),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
        fontsize=8.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff2cc", ec="#d6b656"),
    )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(modes)
    ax2.set_ylabel("Combined Wall-Clock Latency (ms)")
    ax2.set_ylim(0, max(wall_vals) * 1.24)
    ax2.set_title("(b) Opt #4: 3D-VAE Decode + SD3.5 Stream Overlap")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig4_e2e_trace_replay_and_vae_overlap.{ext}", bbox_inches="tight")
    plt.close(fig)


def generate_latex_tables(
    study: Dict[str, Any],
    q_replay: Dict[str, Any],
    v2: Dict[str, Any],
    out_dir: Path,
) -> None:
    bal = study["scenarios"]["gamma_cv2.5_load0.95"]["policies"]
    tex = [
        "% Auto-generated LaTeX booktabs tables from real A100 hardware measurements",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{8$\\times$A100-40GB Multi-Tenant Mixed T2I+T2V Serving Comparison Across 10 Policies (Gamma $CV=2.5$, Load $=0.95$).}",
        "\\label{tab:main_10policies}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Policy} & \\textbf{Total SLO (\\%)} & \\textbf{Video SLO (\\%)} & \\textbf{Image SLO (\\%)} & \\textbf{Image P99 (ms)} & \\textbf{Video P99 (ms)} & \\textbf{Makespan (ms)} \\\\",
        "\\midrule",
    ]
    for key in (
        "static_partition",
        "vanilla_xdit",
        "static_sp1",
        "tridentserve_stage",
        "round_elastic",
        "vllm_omni",
        "genserve_40gb",
        "block_elastic",
        "heterodit_exact",
        "heterodit_approx",
    ):
        r = bal[key]
        label = r["label"].replace("_", "\\_")
        if key.startswith("heterodit"):
            label = f"\\textbf{{{label}}}"
        tex.append(
            f"{label} & {r['total_slo_pct']:.2f} & {r['video_slo_pct']:.2f} & {r['image_slo_pct']:.2f} & "
            f"{r['image_p99_ms']:.1f} & {r['video_p99_ms']:.1f} & {r['makespan_ms']:.1f} \\\\"
        )
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Real 1$\\times$A100 Mixed-Modality Trace Replay (Wan2.1 + 3 Bursty SD3.5 Requests).}",
        "\\label{tab:e2e_replay}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "\\textbf{Serving System} & \\textbf{Img Mean (ms)} & \\textbf{Img P99 (ms)} & \\textbf{Video (ms)} & \\textbf{Total SLO (\\%)} \\\\",
        "\\midrule",
    ])
    for r in q_replay["real_mixed_trace_replay"]:
        pol = r["system"].replace("_", "\\_")
        tex.append(
            f"{pol} & {r['image_mean_ms']:.1f} & {r['image_p99_ms']:.1f} & "
            f"{r['video_latency_ms']:.1f} & {r['total_slo_pct']:.1f} \\\\"
        )
    e2e = v2["e2e_v2_replay"]
    tex.append(
        f"\\textbf{{{e2e['policy'].replace('_', '\\_')}}} & \\textbf{{{e2e['image_mean_latency_ms']:.1f}}} & "
        f"\\textbf{{{e2e['image_p99_latency_ms']:.1f}}} & \\textbf{{{e2e['video_latency_ms']:.1f}}} & "
        f"\\textbf{{{e2e['total_slo_attainment_pct']:.1f}}} \\\\"
    )
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    (out_dir / "PAPER_TABLES.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-json",
        type=Path,
        default=Path("/mnt/e/000_code/zyd-2026/ditserve-hetero/results/heterodit-quality-replay-20261029-r4/hetero_8a100_study_summary.json"),
    )
    parser.add_argument(
        "--quality-json",
        type=Path,
        default=Path("/mnt/e/000_code/zyd-2026/ditserve-hetero/results/heterodit-quality-replay-20261029-r4/QUALITY_AND_REPLAY.json"),
    )
    parser.add_argument(
        "--nccl8-json",
        type=Path,
        default=Path("/mnt/e/000_code/zyd-2026/ditserve-hetero/results/heterodit-nccl-8gpu-20261029-r1/NCCL_BENCHMARK.json"),
    )
    parser.add_argument(
        "--v2-json",
        type=Path,
        default=Path("/mnt/e/000_code/zyd-2026/ditserve-hetero/results/heterodit-v2-opt-20261029-r1/V2_OPTIMIZATIONS.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/e/000_code/zyd-2026/ditserve-hetero/results/figures_and_tables"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    study = json.loads(args.study_json.read_text(encoding="utf-8"))
    q_replay = json.loads(args.quality_json.read_text(encoding="utf-8"))
    nccl8 = json.loads(args.nccl8_json.read_text(encoding="utf-8"))
    v2 = json.loads(args.v2_json.read_text(encoding="utf-8"))

    plot_fig1_10policy_and_ablation(study, args.out_dir)
    plot_fig2_nccl_and_int8_ring(nccl8, v2, args.out_dir)
    plot_fig3_taylor_cache_and_sd35_microbatch(v2, args.out_dir)
    plot_fig4_e2e_replay_and_vae_overlap(q_replay, v2, args.out_dir)
    generate_latex_tables(study, q_replay, v2, args.out_dir)
    print(f"Generated 4 figures (.png + .pdf) and PAPER_TABLES.tex in {args.out_dir}")


if __name__ == "__main__":
    main()
