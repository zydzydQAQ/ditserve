#!/usr/bin/env python3
"""Generate publication-ready figures (.pdf + .png) and LaTeX booktabs tables for HeteroDiT.

Clearly distinguishes:
  - Real GPU Hardware Measurements (1xA100 unified replay, Tier-0/switch, hooked SD3.5 cache,
    2x2=4xA100 cross-node NCCL P2P INT8 Ring, 1x4=4xA100 & 2x2=4xA100 distributed replay, 8xA100 NCCL USP)
  - 8xA100 Calibrated Discrete-Event Simulation (10-policy comparison + 6-stage one-factor ablation)
"""

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
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.2,
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
})


def plot_fig1_10policy_and_ablation(study: Dict[str, Any], out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 4.9), gridspec_kw={"width_ratios": [1.58, 1.0]})

    balanced = study["scenarios"]["gamma_cv2.5_load0.95"]["policies"]
    policies = [
        "static_partition",
        "tridentserve_stage",
        "vanilla_xdit",
        "static_sp1",
        "round_elastic",
        "heterodit_exact",
        "vllm_omni",
        "genserve_40gb",
        "block_elastic",
        "heterodit_approx",
    ]
    labels = [
        "Static\n(2I+6V)",
        "Trident\nServe",
        "Vanilla\nxDiT",
        "Static\nSP1",
        "TetriServe\n(Round)",
        "HeteroDiT\n(Exact DP)",
        "vLLM-Omni\n(Step)",
        "GenServe\n(No-Swap*)",
        "DitServe\n(Block)",
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
    ax1.set_ylabel("Modeled SLO Attainment (%)")
    ax1.set_ylim(0, 114)
    ax1.set_title("(a) 8xA100 10-Policy Comparison [Discrete-Event Simulation, CV=2.5, Load=0.95]")
    ax1.legend(loc="upper left", ncol=3)

    # Panel B: 6-Stage Cumulative One-Factor Ablation + V3 Bundle Sensitivity
    ablation = study["ablation"]["policies"]
    stage_keys = [
        "A0_baseline_1dsp",
        "A1_plus_tiered_mem",
        "A2_plus_2d_usp",
        "A3_heterodit_exact",
        "A4_heterodit_approx",
        "A5_plus_image_batch",
    ]
    stage_labels = [
        "A0: Step\n+ 1D-SP",
        "A1: +Tiered\nPrefetch",
        "A2: +2D USP\nCandidates",
        "A3: +Knapsack\nDP (tau=0)",
        "A4: +Laxity\nTau-Cache",
        "A5: +Image\nuBatch (BS=4)",
    ]
    s_total = [ablation[k]["total_slo_pct"] for k in stage_keys]
    s_vid_p99 = [ablation[k]["video_p99_ms"] / 1000.0 for k in stage_keys]

    sx = np.arange(len(stage_keys))
    bars = ax2.bar(
        sx,
        s_total,
        width=0.52,
        color=["#8c8c8c", "#5b9bd5", "#4472c4", "#2f5597", "#ed7d31", "#c55a11"],
        edgecolor="black",
        linewidth=0.6,
    )
    for b, val in zip(bars, s_total):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            val + 1.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    if "sensitivity" in study and "S1_v3_bundle" in study["sensitivity"]["policies"]:
        v3_slo = study["sensitivity"]["policies"]["S1_v3_bundle"]["total_slo_pct"]
        ax2.annotate(
            f"Separate V3 Bundle Sensitivity:\n{v3_slo:.2f}% Total SLO (Load=1.25)",
            xy=(5, s_total[-1]),
            xytext=(1.55, 22.0),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#c55a11"),
            fontsize=7.8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff2cc", ec="#d6b656"),
        )
    ax2.set_xticks(sx)
    ax2.set_xticklabels(stage_labels, rotation=20, ha="right")
    ax2.set_ylabel("Modeled Total SLO Attainment (%)")
    ax2.set_ylim(0, 108)
    ax2.set_title("(b) 6-Stage One-Factor Ablation [Simulation, Load=1.25]")

    ax2_r = ax2.twinx()
    ax2_r.plot(sx, s_vid_p99, color="#d9534f", marker="o", linewidth=2.0, label="Video P99 (s)")
    ax2_r.set_ylabel("Video P99 Latency (s)", color="#d9534f")
    ax2_r.tick_params(axis="y", labelcolor="#d9534f")
    ax2_r.set_ylim(10, 40)
    ax2_r.grid(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_10policy_slo_pareto.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_nccl_and_int8_ring(nccl8: Dict[str, Any], ring_real: Dict[str, Any], out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.6))

    # Panel A: Real 8xA100 1D-Ulysses (u8r1 padded H=16) vs 2D-Hybrid-USP (u4r2) vs 2D-Hybrid-CFG (cfg2u2r2)
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
    ax1.set_title("(a) 8-GPU 2D-USP vs 1D-Ulysses [Real 8xA100 Measured]")
    ax1.legend(loc="upper left")

    # Panel B: Real Wire-Measured Cross-Node 4-GPU NCCL P2P INT8 Ring KV Exchange (gpu04 + gpu03)
    ring_rows = ring_real["rows"]
    cfg_names = [r["shape"].replace("_", "\n") for r in ring_rows]
    fp16_comm = [r["ring4_step60_fp16_ms"] for r in ring_rows]
    int8_comm = [r["ring4_step60_int8_ms"] for r in ring_rows]

    rx = np.arange(len(ring_rows))
    rw = 0.34
    ax2.bar(rx - rw / 2, fp16_comm, width=rw, label="Standard FP16 Ring4 P2P (Measured)", color="#8c8c8c", edgecolor="black", linewidth=0.5)
    b2 = ax2.bar(
        rx + rw / 2,
        int8_comm,
        width=rw,
        label="Opt #5: INT8 Ring4 P2P + Codec (1.97x Comp)",
        color="#c55a11",
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, r in zip(b2, ring_rows):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 85.0,
            f"{r['ring4_speedup_int8_vs_fp16']:.2f}x (Ring4)\n{r['pair_cross_speedup_int8_vs_fp16']:.2f}x (Cross)",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )
    ax2.set_xticks(rx)
    ax2.set_xticklabels(cfg_names)
    ax2.set_ylabel("Cross-Node 4-GPU Ring P2P per Step (ms)")
    ax2.set_ylim(0, max(fp16_comm) * 1.24)
    ax2.set_title("(b) Opt #5: INT8 Ring KV P2P [Real Cross-Node 4xA100 Measured]")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_nccl_2d_usp_and_int8_ring.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_taylor_cache_and_sd35_microbatch(
    v2: Dict[str, Any],
    real_suite: Dict[str, Any],
    out_dir: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.6))

    # Panel A: Zero-Order DBCache vs First-Order Taylor-Compensated DBCache + Opt #9 Asymmetric CFG Reuse
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
        label="Opt #7: 1st-Order Taylor Extrapolation",
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
    ax1.annotate(
        "Opt #9 Asymmetric CFG + Taylor:\n"
        "• CFG-Only (14..18): 1.14x (35.86 dB)\n"
        "• Taylor+CFG (17..18): 1.22x (33.38 dB)\n"
        "• Taylor+CFG (14..18): 1.33x (32.73 dB)",
        xy=(1 + w / 2, t_psnr[1]),
        xytext=(0.78, 30.55),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#c55a11"),
        fontsize=7.8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff2cc", ec="#d6b656"),
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(taus)
    ax1.set_ylabel("Decoded Video PSNR vs Exact (dB)")
    ax1.set_ylim(30.0, max(max(z_psnr), max(t_psnr)) + 1.5)
    ax1.set_title("(a) Opts #7 & #9: Taylor DBCache + CFG Reuse [Real 1xA100]")
    ax1.legend(loc="upper right")

    # Panel B: Real 1xA100 SD3.5-Medium SDPA_FLASH + Hooked MM-DiT Block Cache + Multi-Prompt uBatching
    sd_modes = real_suite["exp2_sd35_hooked_mmdit_cache"]["modes"]
    mb_labels = [
        "BS=1 Exact\n(SDPA_MATH)",
        "BS=1 Exact\n(SDPA_FLASH)",
        "BS=1 + Hooked\nMM-DiT Cache",
        "uBatch BS=4\n(SDPA_FLASH)",
        "uBatch BS=4\n+ Hooked Cache",
    ]
    per_img_ms = [m["per_image_ms"] for m in sd_modes]
    speedups = [m["speedup"] for m in sd_modes]
    reused = [m["reused_steps"] for m in sd_modes]

    mx = np.arange(len(mb_labels))
    mb_bars = ax2.bar(
        mx,
        per_img_ms,
        width=0.52,
        color=["#d9534f", "#5b9bd5", "#4472c4", "#70ad47", "#c55a11"],
        edgecolor="black",
        linewidth=0.6,
    )
    for b, ms, sp, ru in zip(mb_bars, per_img_ms, speedups, reused):
        extra = f"\n(reused={ru})" if ru > 0 else ""
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            ms + 40.0,
            f"{ms:.1f} ms\n({sp:.2f}x){extra}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )
    ax2.set_xticks(mx)
    ax2.set_xticklabels(mb_labels)
    ax2.set_ylabel("Amortized Latency per Image (ms)")
    ax2.set_ylim(0, max(per_img_ms) * 1.28)
    ax2.set_title("(b) Opts #6 & #11: SD3.5 Flash + Hooked MM-DiT Cache [Real 1xA100]")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_quality_pareto_taylor_vs_zero_order.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_fig4_e2e_replay_and_vae_overlap(
    real_suite: Dict[str, Any],
    v2: Dict[str, Any],
    v3: Dict[str, Any],
    out_dir: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 4.8), gridspec_kw={"width_ratios": [1.48, 1.0]})

    # Panel A: Unified-Trace Online-Clock Real 1xA100 Mixed-Modality Trace Replay Across All 6 Systems
    systems = real_suite["exp3_unified_1xa100_replay"]["systems"]
    p_labels = [
        "Vanilla-xDiT\n(Run-to-Comp)",
        "GenServe-40GB\n(Real Cold Swap)",
        "HeteroDiT-v1\n(Exact)",
        "HeteroDiT-v1\n(Approx)",
        "HeteroDiT-v2\n(Opts #1-#8)",
        "HeteroDiT-v3\n(All Opts)",
    ]
    img_p99 = [r["image_p99_ms"] / 1000.0 for r in systems]
    vid_lat = [r["video_latency_ms"] / 1000.0 for r in systems]
    slo_pct = [r["total_slo_pct"] for r in systems]

    x = np.arange(len(p_labels))
    w = 0.34
    ax1.bar(x - w / 2, img_p99, width=w, label="Image Max Latency (s) [SLO <= 4.5s]", color="#70ad47", edgecolor="black", linewidth=0.5)
    ax1.bar(x + w / 2, vid_lat, width=w, label="Video Latency (s) [SLO <= 17.0s]", color="#2e75b6", edgecolor="black", linewidth=0.5)
    ax1.axhline(4.5, color="#70ad47", linestyle=":", linewidth=1.2)
    ax1.axhline(17.0, color="#2e75b6", linestyle="--", linewidth=1.2)
    for i, sp in enumerate(slo_pct):
        top = max(img_p99[i], vid_lat[i])
        ax1.text(
            x[i],
            top + 0.55,
            f"SLO: {sp:.0f}%\n({vid_lat[i]:.2f}s)",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(p_labels, rotation=16, ha="right")
    ax1.set_ylabel("End-to-End Request Latency (s)")
    ax1.set_ylim(0, max(vid_lat) * 1.20)
    ax1.set_title("(a) Unified-Trace Online-Clock 1xA100 Replay Across 6 Systems [Real Hardware]")
    ax1.legend(loc="upper right")

    # Panel B: Opts #8 & #10 Stream-Isolated 3D-VAE Chunked Decode + BF16 Tensor-Core Acceleration
    ovl = v2["opt4_vae_stream_overlap"]
    bf16_vae = v3["opt10_bf16_3dvae"]
    modes = [
        "Sequential\n(FP32 3D-VAE + SD3.5)",
        "Opt #8: Stream-Overlap\n(FP32 3D-VAE + SD3.5)",
        "Opt #10: BF16 3D-VAE\n+ Stream-Overlap (v3)",
    ]
    wall_vals = [
        ovl["sequential_wall_ms"],
        ovl["overlapped_wall_ms"],
        bf16_vae["bf16_stream_overlapped_with_sd35_wall_ms"],
    ]
    bars_o = ax2.bar([0, 1, 2], wall_vals, width=0.48, color=["#8c8c8c", "#ed7d31", "#c55a11"], edgecolor="black", linewidth=0.6)
    for b, v in zip(bars_o, wall_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            v + 35.0,
            f"{v:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.annotate(
        f"Opt #10 BF16 3D-VAE:\n"
        f"• VAE: 1485ms -> 967ms (1.54x)\n"
        f"• PSNR vs FP32: {bf16_vae['bf16_vs_fp32_psnr_db']:.2f} dB\n"
        f"• Peak HBM Saved: -4,001 MB",
        xy=(2, wall_vals[2]),
        xytext=(0.35, wall_vals[0] * 0.72),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
        fontsize=8.0,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff2cc", ec="#d6b656"),
    )
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(modes)
    ax2.set_ylabel("Combined Wall-Clock Latency (ms)")
    ax2.set_ylim(0, max(wall_vals) * 1.24)
    ax2.set_title("(b) Opts #8 & #10: BF16 3D-VAE + CUDA Stream Overlap [Real 1xA100]")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig4_e2e_trace_replay_and_vae_overlap.{ext}", bbox_inches="tight")
    plt.close(fig)


def generate_latex_tables(
    study: Dict[str, Any],
    real_suite: Dict[str, Any],
    ring_real: Dict[str, Any],
    out_dir: Path,
) -> None:
    tex = [
        "% Auto-generated LaTeX booktabs tables with explicit Real-GPU vs Simulation provenance",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{[Real Hardware: 1$\\times$A100-40GB] Unified-Trace Online-Clock Mixed-Modality Replay Across 6 Systems (Wan2.1 @ $t=0$ms + 3 Bursty SD3.5 Requests @ $1200, 1350, 3500$ms; Image SLO $\\le 4.5$s, Video SLO $\\le 17.0$s).}",
        "\\label{tab:real_1xa100_unified_replay}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "\\textbf{Serving System (Real 1$\\times$A100)} & \\textbf{Img Mean (ms)} & \\textbf{Img Max (ms)} & \\textbf{Video (ms)} & \\textbf{Peak HBM (MB)} & \\textbf{Total SLO (\\%)} \\\\",
        "\\midrule",
    ]
    for r in real_suite["exp3_unified_1xa100_replay"]["systems"]:
        pol = r["system"].replace("_", "\\_").replace("#", "\\#")
        if "HeteroDiT-v2" in r["system"] or "HeteroDiT-v3" in r["system"]:
            tex.append(
                f"\\textbf{{{pol}}} & \\textbf{{{r['image_mean_ms']:.1f}}} & \\textbf{{{r['image_p99_ms']:.1f}}} & "
                f"\\textbf{{{r['video_latency_ms']:.1f}}} & \\textbf{{{r['peak_hbm_mb']:.1f}}} & \\textbf{{{r['total_slo_pct']:.1f}}} \\\\"
            )
        else:
            tex.append(
                f"{pol} & {r['image_mean_ms']:.1f} & {r['image_p99_ms']:.1f} & "
                f"{r['video_latency_ms']:.1f} & {r['peak_hbm_mb']:.1f} & {r['total_slo_pct']:.1f} \\\\"
            )
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{[Discrete-Event Simulation: 8$\\times$A100-40GB] Multi-Tenant Mixed T2I+T2V Serving Comparison Across 10 Scheduling Proxies (Gamma $CV=2.5$, Load $=0.95$).}",
        "\\label{tab:sim_8xa100_10policies}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Policy Proxy (8$\\times$A100 Simulation)} & \\textbf{Total SLO (\\%)} & \\textbf{Video SLO (\\%)} & \\textbf{Image SLO (\\%)} & \\textbf{Image P99 (ms)} & \\textbf{Video P99 (ms)} & \\textbf{Makespan (ms)} \\\\",
        "\\midrule",
    ])
    bal = study["scenarios"]["gamma_cv2.5_load0.95"]["policies"]
    for key in (
        "static_partition",
        "tridentserve_stage",
        "vanilla_xdit",
        "static_sp1",
        "round_elastic",
        "heterodit_exact",
        "vllm_omni",
        "genserve_40gb",
        "block_elastic",
        "heterodit_approx",
    ):
        r = bal[key]
        label = r["label"].replace("_", "\\_")
        if key == "heterodit_approx":
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
        "\\caption{[Real Hardware: 2$\\times$2 = 4$\\times$A100 Cross-Node] Wire-Measured NCCL P2P INT8 vs.\\ FP16 Ring KV Exchange (60 Collectives/Step).}",
        "\\label{tab:real_crossnode_int8_ring}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "\\textbf{Shape} & \\textbf{Cross FP16 (ms)} & \\textbf{Cross INT8 (ms)} & \\textbf{Ring4 FP16 (ms)} & \\textbf{Ring4 INT8 (ms)} & \\textbf{Speedup} \\\\",
        "\\midrule",
    ])
    for r in ring_real["rows"]:
        sh = r["shape"].replace("_", "\\_")
        tex.append(
            f"{sh} & {r['pair_cross_step60_fp16_ms']:.1f} & {r['pair_cross_step60_int8_ms']:.1f} & "
            f"{r['ring4_step60_fp16_ms']:.1f} & {r['ring4_step60_int8_ms']:.1f} & \\textbf{{{r['ring4_speedup_int8_vs_fp16']:.2f}$\\times$}} \\\\"
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
        default=Path("/mnt/e/000_code/zyd-2026/ditserve-hetero/results/hetero_8a100_study_summary.json"),
    )
    parser.add_argument(
        "--real-suite-json",
        type=Path,
        default=Path("/mnt/e/000_code/zyd-2026/ditserve-hetero/results/heterodit-real-verify-20261030-r1/REAL_VERIFICATION_SUITE.json"),
    )
    parser.add_argument(
        "--ring-real-json",
        type=Path,
        default=Path("/mnt/e/000_code/zyd-2026/ditserve-hetero/results/heterodit-ring-4gpu-20261030-r4/RING_INT8_AND_4GPU_REAL.json"),
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
        "--v3-json",
        type=Path,
        default=Path("/mnt/e/000_code/zyd-2026/ditserve-hetero/results/heterodit-v3-opt-20261030-r7/V3_OPTIMIZATIONS.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/e/000_code/zyd-2026/ditserve-hetero/results/figures_and_tables"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_study = json.loads(args.study_json.read_text(encoding="utf-8"))
    study = raw_study.get("opt12_8a100_study", raw_study)
    real_suite = json.loads(args.real_suite_json.read_text(encoding="utf-8"))
    ring_real = json.loads(args.ring_real_json.read_text(encoding="utf-8"))
    nccl8 = json.loads(args.nccl8_json.read_text(encoding="utf-8"))
    v2 = json.loads(args.v2_json.read_text(encoding="utf-8"))
    v3 = json.loads(args.v3_json.read_text(encoding="utf-8"))

    plot_fig1_10policy_and_ablation(study, args.out_dir)
    plot_fig2_nccl_and_int8_ring(nccl8, ring_real, args.out_dir)
    plot_fig3_taylor_cache_and_sd35_microbatch(v2, real_suite, args.out_dir)
    plot_fig4_e2e_replay_and_vae_overlap(real_suite, v2, v3, args.out_dir)
    generate_latex_tables(study, real_suite, ring_real, args.out_dir)
    print(f"Generated 4 figures (.png + .pdf) and PAPER_TABLES.tex in {args.out_dir}")


if __name__ == "__main__":
    main()
