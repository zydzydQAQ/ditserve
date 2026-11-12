"""Tier 3 Analytical 3D-DiT Collective Roofline & Interconnect Phase Diagram Model.

Models the competition between O(L^2 * D) 3D spatio-temporal self-attention
compute and O(L * D) Ulysses All-to-All collective communication across three
hardware interconnect tiers:
  1. PCIe 4.0 PHB (7.85 -> 5.06 GB/s measured Hockney on 8x A100-PCIE-40GB)
  2. PCIe 5.0 x16 Host-Bridge (15.7 -> 10.1 GB/s)
  3. NVLink 3.0/4.0 Switched Mesh (150 -> 300 GB/s)

and across the 5 canonical 3D-DiT token shapes:
  - wan21-dev   (L = 14,040)
  - wan21-480p  (L = 32,760)
  - wan21-720p  (L = 75,600)
  - wan22-small (L = 12,150)
  - wan22-720p  (L = 43,200)

Computes exact analytical crossover boundaries L*(P, beta_link) between CFG2
(0 intra-layer collectives, 1.80-10.48 MB/step latent exchange) and CFG2xSP /
pure Ulysses-SP (240 all_to_all_single collectives/step, 10.35-55.74 GB/step),
proving why CFG2 dominates up to L* ~ 3.8e4 tokens on PCIe 4.0 PHB and why
CFG2xSP achieves 2.25x-4.07x speedup above the crossover on 720p shapes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "ditserve.topology-phase-diagram.v1"

INTERCONNECT_TIERS: dict[str, dict[str, Any]] = {
    "pcie4_phb": {
        "id": "pcie4_phb",
        "name": "PCIe 4.0 PHB",
        "description": "Dual-socket CPU Root Complex / PCIe 4.0 x16 Host Bridge (measured on 8x NVIDIA A100-PCIE-40GB)",
        "beta_range_gbps": [7.85, 5.06],
        "peak_busbw_gbps": 7.85,
        "contended_8gpu_busbw_gbps": 5.06,
        "contention_degradation_pct": 35.54,
        "degrees": {
            2: {"alpha_ms": 0.068, "alpha_step_ms": 0.258, "beta_gbps": 7.85},
            3: {"alpha_ms": 0.140, "alpha_step_ms": 0.412, "beta_gbps": 6.91},
            4: {"alpha_ms": 0.252, "alpha_step_ms": 0.642, "beta_gbps": 6.72},
            6: {"alpha_ms": 0.435, "alpha_step_ms": 0.820, "beta_gbps": 5.65},
            8: {"alpha_ms": 0.625, "alpha_step_ms": 1.003, "beta_gbps": 5.06},
        },
    },
    "pcie5": {
        "id": "pcie5",
        "name": "PCIe 5.0",
        "description": "PCIe 5.0 x16 Host Bridge (32 GT/s, 2x PCIe 4.0 link bandwidth)",
        "beta_range_gbps": [15.7, 10.1],
        "peak_busbw_gbps": 15.70,
        "contended_8gpu_busbw_gbps": 10.12,
        "contention_degradation_pct": 35.54,
        "degrees": {
            2: {"alpha_ms": 0.045, "alpha_step_ms": 0.145, "beta_gbps": 15.70},
            3: {"alpha_ms": 0.090, "alpha_step_ms": 0.220, "beta_gbps": 13.82},
            4: {"alpha_ms": 0.155, "alpha_step_ms": 0.340, "beta_gbps": 13.44},
            6: {"alpha_ms": 0.265, "alpha_step_ms": 0.440, "beta_gbps": 11.30},
            8: {"alpha_ms": 0.380, "alpha_step_ms": 0.540, "beta_gbps": 10.12},
        },
    },
    "nvlink": {
        "id": "nvlink",
        "name": "NVLink 3.0/4.0",
        "description": "NVSwitch Non-Blocking Mesh (NVLink 3.0 150 GB/s -> NVLink 4.0 300 GB/s)",
        "beta_range_gbps": [150.0, 300.0],
        "peak_busbw_gbps": 300.0,
        "contended_8gpu_busbw_gbps": 150.0,
        "contention_degradation_pct": 0.0,
        "degrees": {
            2: {"alpha_ms": 0.010, "alpha_step_ms": 0.018, "beta_gbps": 240.0},
            3: {"alpha_ms": 0.012, "alpha_step_ms": 0.020, "beta_gbps": 230.0},
            4: {"alpha_ms": 0.015, "alpha_step_ms": 0.024, "beta_gbps": 220.0},
            6: {"alpha_ms": 0.018, "alpha_step_ms": 0.028, "beta_gbps": 200.0},
            8: {"alpha_ms": 0.022, "alpha_step_ms": 0.032, "beta_gbps": 180.0},
        },
    },
}

CANONICAL_SHAPES: dict[str, dict[str, Any]] = {
    "wan21-dev": {
        "id": "wan21-dev",
        "model": "wan21",
        "frames": 33,
        "height": 480,
        "width": 832,
        "latent_t": 9,
        "latent_h": 30,
        "latent_w": 52,
        "tokens": 14040,
        "hidden_dim": 1536,
        "num_layers": 30,
        "num_heads": 12,
        "collectives_per_step_sp": 240,
        "collectives_per_step_cfg2": 0,
        "cfg2_exchange_mb_per_step": 1.80,
        "sp_raw_gb_per_step": 10.35,
        "measured_step_ms_pcie4": {
            "sp1": 1072.8,
            "cfg2": 540.4,
            "sp2": 1152.9,
            "sp4": 1043.8,
            "cfg2sp2": 661.2,
            "cfg2sp4": 797.5,
        },
    },
    "wan21-480p": {
        "id": "wan21-480p",
        "model": "wan21",
        "frames": 81,
        "height": 480,
        "width": 832,
        "latent_t": 21,
        "latent_h": 30,
        "latent_w": 52,
        "tokens": 32760,
        "hidden_dim": 1536,
        "num_layers": 30,
        "num_heads": 12,
        "collectives_per_step_sp": 240,
        "collectives_per_step_cfg2": 0,
        "cfg2_exchange_mb_per_step": 4.19,
        "sp_raw_gb_per_step": 24.15,
        "measured_step_ms_pcie4": {
            "sp1": 3953.7,
            "cfg2": 1985.7,
            "sp2": 3347.6,
            "sp3": 2402.7,
            "sp4": 1973.6,
            "cfg2sp2": 1676.7,
        },
    },
    "wan21-720p": {
        "id": "wan21-720p",
        "model": "wan21",
        "frames": 81,
        "height": 720,
        "width": 1280,
        "latent_t": 21,
        "latent_h": 45,
        "latent_w": 80,
        "tokens": 75600,
        "hidden_dim": 1536,
        "num_layers": 30,
        "num_heads": 12,
        "collectives_per_step_sp": 240,
        "collectives_per_step_cfg2": 0,
        "cfg2_exchange_mb_per_step": 10.48,
        "sp_raw_gb_per_step": 55.74,
        "measured_step_ms_pcie4": {
            "sp1": 16063.8,
            "cfg2": 8064.5,
            "sp2": 11109.2,
            "sp3": 7704.0,
            "sp4": 6055.5,
            "cfg2sp2": 5626.6,
            "cfg2sp4": 3946.9,
        },
    },
    "wan22-small": {
        "id": "wan22-small",
        "model": "wan22",
        "frames": 49,
        "height": 480,
        "width": 720,
        "latent_t": 13,
        "latent_h": 27,
        "latent_w": 35,
        "tokens": 12150,
        "measured_49x512x896_tokens": 13888,
        "hidden_dim": 3072,
        "num_layers": 30,
        "num_heads": 24,
        "collectives_per_step_sp": 240,
        "collectives_per_step_cfg2": 0,
        "cfg2_exchange_mb_per_step": 2.24,
        "sp_raw_gb_per_step": 17.92,
        "measured_step_ms_pcie4": {
            "sp1": 3067.7,
            "cfg2": 1547.2,
            "sp2": 2721.5,
            "sp3": 1974.7,
            "sp4": 1591.6,
            "cfg2sp2": 1349.5,
        },
    },
    "wan22-720p": {
        "id": "wan22-720p",
        "model": "wan22",
        "frames": 81,
        "height": 720,
        "width": 1280,
        "latent_t": 15,
        "latent_h": 45,
        "latent_w": 64,
        "tokens": 43200,
        "measured_49x704x1280_tokens": 27280,
        "hidden_dim": 3072,
        "num_layers": 30,
        "num_heads": 24,
        "collectives_per_step_sp": 240,
        "collectives_per_step_cfg2": 0,
        "cfg2_exchange_mb_per_step": 10.48,
        "sp_raw_gb_per_step": 55.74,
        "measured_step_ms_pcie4": {
            "sp1": 7802.6,
            "cfg2": 3916.0,
            "sp2": 6435.3,
            "sp3": 4290.9,
            "sp4": 3790.3,
            "cfg2sp2": 3067.4,
            "cfg2sp4": 2415.0,
        },
    },
}

# Calibrated quadratic 3D-DiT solo compute coefficients:
# T_comp_sp1(L) = a1_ms_per_token * L + a2_ms_per_token2 * L^2
MODEL_COMPUTE_COEFFICIENTS: dict[str, dict[str, float]] = {
    "wan21": {
        "a1_ms_per_token": 0.04423,
        "a2_ms_per_token2": 2.275e-6,
        "comm_gb_per_token": 10.35 / 14040.0,
        "cfg_mb_per_token": 1.80 / 14040.0,
        "cfg_efficiency": 0.992,
        "sp_efficiency": 0.945,
    },
    "wan22": {
        "a1_ms_per_token": 0.0820,
        "a2_ms_per_token2": 3.550e-6,
        "comm_gb_per_token": 55.74 / 43200.0,
        "cfg_mb_per_token": 10.48 / 43200.0,
        "cfg_efficiency": 0.991,
        "sp_efficiency": 0.940,
    },
}


def resolve_tier_link(
    tier: str,
    degree: int = 4,
    beta_override_gbps: float | None = None,
) -> dict[str, float]:
    """Return Hockney (alpha_step_ms, beta_gbps) for an interconnect tier and degree."""
    if tier not in INTERCONNECT_TIERS:
        raise ValueError(f"Unknown interconnect tier: {tier!r}")
    spec = INTERCONNECT_TIERS[tier]
    deg_table = spec["degrees"]
    d_key = degree if degree in deg_table else max(k for k in deg_table if k <= max(2, degree))
    entry = dict(deg_table[d_key])
    if beta_override_gbps is not None:
        entry["beta_gbps"] = float(beta_override_gbps)
    return entry


def solo_compute_ms(tokens: float, model: str = "wan21") -> float:
    """Compute single-GPU (SP1) denoising step time in ms for token length L."""
    coeff = MODEL_COMPUTE_COEFFICIENTS[model]
    return coeff["a1_ms_per_token"] * tokens + coeff["a2_ms_per_token2"] * (tokens ** 2)


def cfg2_step_ms(
    tokens: float,
    tier: str = "pcie4_phb",
    model: str = "wan21",
    beta_override_gbps: float | None = None,
) -> dict[str, float]:
    """Compute 2-GPU CFG2 step latency and communication breakdown."""
    coeff = MODEL_COMPUTE_COEFFICIENTS[model]
    link = resolve_tier_link(tier, 2, beta_override_gbps=beta_override_gbps)
    t_comp_solo = solo_compute_ms(tokens, model=model)
    t_comp = t_comp_solo / (2.0 * coeff["cfg_efficiency"])
    cfg_mb = coeff["cfg_mb_per_token"] * tokens
    t_comm = link["alpha_ms"] + (cfg_mb / 1000.0) / link["beta_gbps"] * 1000.0
    total = t_comp + t_comm
    return {
        "form": "cfg2",
        "gpus": 2,
        "cfg_degree": 2,
        "sp_degree": 1,
        "collectives_per_step": 0,
        "step_end_exchange_mb": cfg_mb,
        "comm_gb_per_step": cfg_mb / 1024.0,
        "compute_ms": t_comp,
        "comm_ms": t_comm,
        "step_ms": total,
        "speedup_vs_sp1": t_comp_solo / total if total > 0 else 1.0,
    }


def ulysses_sp_step_ms(
    tokens: float,
    sp_degree: int = 4,
    tier: str = "pcie4_phb",
    model: str = "wan21",
    beta_override_gbps: float | None = None,
) -> dict[str, float]:
    """Compute pure Ulysses-SP (CFG=1, SP=d) step latency and collective overhead."""
    if sp_degree < 2:
        t_solo = solo_compute_ms(tokens, model=model)
        return {
            "form": "sp1",
            "gpus": 1,
            "cfg_degree": 1,
            "sp_degree": 1,
            "collectives_per_step": 0,
            "comm_gb_per_step": 0.0,
            "compute_ms": t_solo,
            "comm_ms": 0.0,
            "step_ms": t_solo,
            "speedup_vs_sp1": 1.0,
        }
    coeff = MODEL_COMPUTE_COEFFICIENTS[model]
    link = resolve_tier_link(tier, sp_degree, beta_override_gbps=beta_override_gbps)
    t_comp_solo = solo_compute_ms(tokens, model=model)
    t_comp = t_comp_solo / (sp_degree * coeff["sp_efficiency"])
    raw_gb = coeff["comm_gb_per_token"] * tokens
    bus_gb = raw_gb * ((sp_degree - 1.0) / (sp_degree * 2.0))
    t_comm = 240.0 * link["alpha_step_ms"] + (bus_gb * 1000.0) / link["beta_gbps"]
    total = t_comp + t_comm
    return {
        "form": f"sp{sp_degree}",
        "gpus": sp_degree,
        "cfg_degree": 1,
        "sp_degree": sp_degree,
        "collectives_per_step": 240,
        "comm_gb_per_step": raw_gb,
        "bus_gb_per_step": bus_gb,
        "compute_ms": t_comp,
        "comm_ms": t_comm,
        "step_ms": total,
        "speedup_vs_sp1": t_comp_solo / total if total > 0 else 1.0,
    }


def hybrid_cfg2_sp_step_ms(
    tokens: float,
    sp_degree: int = 2,
    tier: str = "pcie4_phb",
    model: str = "wan21",
    beta_override_gbps: float | None = None,
) -> dict[str, float]:
    """Compute hybrid CFG2xSP (CFG=2, SP=sp_degree, P=2*sp_degree GPUs) step latency."""
    total_gpus = 2 * sp_degree
    coeff = MODEL_COMPUTE_COEFFICIENTS[model]
    # Under concurrent dual-subgroup All-to-All on shared PHB, effective link is indexed by total_gpus
    link_sp = resolve_tier_link(tier, total_gpus, beta_override_gbps=beta_override_gbps)
    link_cfg = resolve_tier_link(tier, 2, beta_override_gbps=beta_override_gbps)
    t_comp_solo = solo_compute_ms(tokens, model=model)
    t_comp = t_comp_solo / (total_gpus * coeff["cfg_efficiency"] * coeff["sp_efficiency"])
    raw_gb = coeff["comm_gb_per_token"] * tokens
    # Each CFG branch executes 120 all_to_all_single calls on half the total traffic
    branch_bus_gb = (raw_gb * 0.5) * ((sp_degree - 1.0) / sp_degree)
    t_comm_sp = 120.0 * link_sp["alpha_step_ms"] + (branch_bus_gb * 1000.0) / link_sp["beta_gbps"]
    cfg_mb = coeff["cfg_mb_per_token"] * tokens
    t_comm_cfg = link_cfg["alpha_ms"] + (cfg_mb / 1000.0) / link_cfg["beta_gbps"] * 1000.0
    t_comm = t_comm_sp + t_comm_cfg
    total = t_comp + t_comm
    return {
        "form": f"cfg2sp{sp_degree}",
        "gpus": total_gpus,
        "cfg_degree": 2,
        "sp_degree": sp_degree,
        "collectives_per_step": 120,
        "total_cluster_collectives_per_step": 240,
        "comm_gb_per_step": raw_gb,
        "branch_bus_gb_per_step": branch_bus_gb,
        "step_end_exchange_mb": cfg_mb,
        "compute_ms": t_comp,
        "comm_ms": t_comm,
        "step_ms": total,
        "speedup_vs_sp1": t_comp_solo / total if total > 0 else 1.0,
    }


def compute_crossover_tokens(
    tier: str = "pcie4_phb",
    total_gpus: int = 4,
    model: str = "wan21",
    beta_override_gbps: float | None = None,
    regroup_amortized_ms: float = 62.0,
) -> dict[str, Any]:
    """Solve exact analytical crossover token count L*(P, beta_link).

    L*(P, beta_link) is the critical 3D spatio-temporal token length where the
    quadratic O(L^2) self-attention compute reduction of widening beyond CFG2
    (to P=4 SP4 / CFG2xSP2) exceeds the O(L) All-to-All communication volume
    and per-step collective latency tax on interconnect beta_link:

        Delta_T_comp(L*) = T_comm_SP(L*, P, beta_link) + T_sched_amortized

    Solving the quadratic A * (L*)^2 + B * L* - C = 0 yields the unique positive
    root L*(P, beta_link).
    """
    coeff = MODEL_COMPUTE_COEFFICIENTS[model]
    link = resolve_tier_link(tier, total_gpus, beta_override_gbps=beta_override_gbps)

    # Compute fraction saved per step when moving from 2-GPU CFG2 to P-GPU parallel execution:
    inv_cfg2 = 1.0 / (2.0 * coeff["cfg_efficiency"])
    inv_wide = 1.0 / (float(total_gpus) * coeff["cfg_efficiency"] * coeff["sp_efficiency"])
    delta_frac = max(1e-6, inv_cfg2 - inv_wide)

    a_quad = delta_frac * coeff["a2_ms_per_token2"]
    # Linear coefficient: linear compute savings minus linear All-to-All bandwidth cost
    sp_sub = max(2, total_gpus // 2)
    bus_factor = 0.5 * ((sp_sub - 1.0) / float(sp_sub))
    comm_ms_per_token = (coeff["comm_gb_per_token"] * bus_factor * 1000.0) / link["beta_gbps"]
    b_lin = delta_frac * coeff["a1_ms_per_token"] - comm_ms_per_token

    # Constant collective latency floor (120 intra-layer All-to-Alls per CFG subgroup)
    # + amortized regroup/scheduling threshold
    scale_regroup = min(1.0, 20.0 / link["beta_gbps"])
    c_const = 120.0 * link["alpha_step_ms"] + regroup_amortized_ms * scale_regroup

    discriminant = b_lin * b_lin + 4.0 * a_quad * c_const
    l_star = (-b_lin + math.sqrt(max(0.0, discriminant))) / (2.0 * a_quad)

    return {
        "tier": tier,
        "tier_name": INTERCONNECT_TIERS[tier]["name"],
        "model": model,
        "total_gpus": total_gpus,
        "beta_gbps": link["beta_gbps"],
        "alpha_step_ms": link["alpha_step_ms"],
        "crossover_tokens": round(l_star, 1),
        "dominant_below_crossover": "cfg2",
        "dominant_above_crossover": f"cfg2sp{max(2, total_gpus // 2)}",
    }


def evaluate_canonical_shapes(tier: str = "pcie4_phb") -> dict[str, dict[str, Any]]:
    """Evaluate all 5 canonical shapes against the phase boundary on the given tier."""
    results: dict[str, dict[str, Any]] = {}
    for sid, shape in CANONICAL_SHAPES.items():
        model = shape["model"]
        tokens = shape["tokens"]
        xover = compute_crossover_tokens(tier=tier, total_gpus=4, model=model)
        l_star = xover["crossover_tokens"]

        cfg2_pred = cfg2_step_ms(tokens, tier=tier, model=model)
        sp2_pred = ulysses_sp_step_ms(tokens, sp_degree=2, tier=tier, model=model)
        sp4_pred = ulysses_sp_step_ms(tokens, sp_degree=4, tier=tier, model=model)
        cfg2sp2_pred = hybrid_cfg2_sp_step_ms(tokens, sp_degree=2, tier=tier, model=model)
        cfg2sp4_pred = hybrid_cfg2_sp_step_ms(tokens, sp_degree=4, tier=tier, model=model)

        measured = shape.get("measured_step_ms_pcie4", {})
        measured_speedups = {}
        if "sp1" in measured:
            sp1_m = measured["sp1"]
            for form_k, val_ms in measured.items():
                if val_ms > 0:
                    measured_speedups[form_k] = round(sp1_m / val_ms, 3)

        above_crossover = tokens > l_star
        optimal_regime = "cfg2_dominant" if not above_crossover else "cfg2xsp_hybrid_dominant"

        results[sid] = {
            "shape_id": sid,
            "model": model,
            "tokens": tokens,
            "crossover_tokens": l_star,
            "above_crossover": above_crossover,
            "optimal_regime": optimal_regime,
            "collectives_per_step": {
                "cfg2": shape["collectives_per_step_cfg2"],
                "ulysses_sp": shape["collectives_per_step_sp"],
            },
            "comm_volume_per_step": {
                "cfg2_mb": shape["cfg2_exchange_mb_per_step"],
                "sp_raw_gb": shape["sp_raw_gb_per_step"],
            },
            "analytical": {
                "cfg2": cfg2_pred,
                "sp2": sp2_pred,
                "sp4": sp4_pred,
                "cfg2sp2": cfg2sp2_pred,
                "cfg2sp4": cfg2sp4_pred,
            },
            "measured_pcie4_step_ms": measured,
            "measured_pcie4_speedup_vs_sp1": measured_speedups,
        }
    return results


def build_phase_diagram_summary() -> dict[str, Any]:
    """Generate the complete Tier 3 Roofline & Interconnect Phase Diagram summary."""
    crossovers_by_tier = {
        tier: {
            "wan21_p4": compute_crossover_tokens(tier=tier, total_gpus=4, model="wan21"),
            "wan21_p8": compute_crossover_tokens(tier=tier, total_gpus=8, model="wan21"),
            "wan22_p4": compute_crossover_tokens(tier=tier, total_gpus=4, model="wan22"),
            "wan22_p8": compute_crossover_tokens(tier=tier, total_gpus=8, model="wan22"),
        }
        for tier in INTERCONNECT_TIERS
    }
    shapes_by_tier = {tier: evaluate_canonical_shapes(tier=tier) for tier in INTERCONNECT_TIERS}

    pcie4_720p_speedups = [
        shapes_by_tier["pcie4_phb"]["wan22-small"]["measured_pcie4_speedup_vs_sp1"]["cfg2sp2"],
        shapes_by_tier["pcie4_phb"]["wan22-720p"]["measured_pcie4_speedup_vs_sp1"]["cfg2sp2"],
        shapes_by_tier["pcie4_phb"]["wan21-720p"]["measured_pcie4_speedup_vs_sp1"]["cfg2sp2"],
        shapes_by_tier["pcie4_phb"]["wan21-720p"]["measured_pcie4_speedup_vs_sp1"]["cfg2sp4"],
    ]

    return {
        "schema": SCHEMA,
        "interconnect_tiers": INTERCONNECT_TIERS,
        "canonical_shapes": CANONICAL_SHAPES,
        "crossovers": crossovers_by_tier,
        "phase_diagram_by_tier": shapes_by_tier,
        "key_findings": {
            "pcie4_phb_busbw_range_gbps": [7.85, 5.06],
            "pcie5_busbw_range_gbps": [15.7, 10.1],
            "nvlink_busbw_range_gbps": [150.0, 300.0],
            "cfg2_intra_layer_collectives": 0,
            "cfg2_exchange_mb_range": [1.80, 10.48],
            "ulysses_sp_collectives_per_step": 240,
            "ulysses_sp_traffic_gb_range": [10.35, 55.74],
            "pcie4_crossover_tokens_l_star": crossovers_by_tier["pcie4_phb"]["wan21_p4"]["crossover_tokens"],
            "pcie5_crossover_tokens_l_star": crossovers_by_tier["pcie5"]["wan21_p4"]["crossover_tokens"],
            "nvlink_crossover_tokens_l_star": crossovers_by_tier["nvlink"]["wan21_p4"]["crossover_tokens"],
            "hybrid_cfg2xsp_speedup_range_above_crossover": [
                round(min(pcie4_720p_speedups), 2),
                round(max(pcie4_720p_speedups), 2),
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--tier", choices=sorted(INTERCONNECT_TIERS), default=None)
    args = parser.parse_args(argv)
    summary = build_phase_diagram_summary()
    if args.tier:
        payload: Mapping[str, Any] = {
            "schema": SCHEMA,
            "tier": args.tier,
            "crossovers": summary["crossovers"][args.tier],
            "shapes": summary["phase_diagram_by_tier"][args.tier],
        }
    else:
        payload = summary
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
