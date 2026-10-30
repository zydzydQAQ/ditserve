#!/usr/bin/env python3
"""8-slot A100-shaped cost-model simulation, not an 8-GPU hardware evaluation.

All named strategies are simulator proxies, not reproductions of their systems.
Main scheduling comparisons share cost inputs and execution options. Exact means
cache tau=0, not validated lossless output; approximate rows use assumed savings.
The six-stage cumulative ablation changes one configured factor per transition.
Unverified V3 execution discounts are a separate, non-orthogonal bundle sensitivity.
"""

from __future__ import annotations

import copy
import hashlib
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


COST_SOURCE_DETAILS = {
    "legacy_timing_claims": "resize_ms and links.8.gbps carried historical hardware timing claims; artifacts not revalidated here",
    "uncalibrated_inputs": "all other hard-coded service, resume, batch, bandwidth, overhead and overlap values are model assumptions",
    "scope": "mixed provenance is not an end-to-end hardware calibration",
}
MODEL_ASSUMPTIONS = [
    "All policy names denote simulator proxies, not reproductions of published systems.",
    "Exact denotes cache tau=0 only; output equivalence and quality have not been validated.",
    "Tau-to-cache-skip tables and batch timings are assumed savings, not quality/performance evidence.",
    "Tiered prefetch, when enabled, uses the simulator's hard-coded 0.25 * resume_ms discount; transfer and residency are not validated.",
]
FORMS_1D = ["sp1", "cfg2", "sp2", "cfg2sp2", "sp4", "sp8"]
FORMS_2D = ["sp1", "cfg2", "sp2", "cfg2sp2", "u2r2", "sp4", "u4r2", "cfg2u2r2", "sp8"]
APPROX_THRESHOLDS = [0.0, 0.08, 0.16, 0.24]
# Retain the old A0 swap/resume inputs for every main row, not result-tuned per policy.
SHARED_EXECUTION_OPTIONS = {
    "cross_modal_swap_ms": 75.0, "resume_scale": 1.8, "quantum_ms": 160.0,
    "chunked_decode": True, "decode_chunk_frames": 0, "tiered_memory": False,
    "image_batch": 1, "continuous_batch": 0, "pairing": False,
}


def shared_8a100_profile() -> dict:
    """Legacy mixed-source inputs shared by all main scheduling comparisons."""
    return {
        "provenance": "mixed",
        "cost_source_details": copy.deepcopy(COST_SOURCE_DETAILS),
        "model_assumptions": list(MODEL_ASSUMPTIONS),
        "sp_efficiency": 0.96,
        "sp_overlap": 0.28,
        "ring_overlap": 0.82,
        "pair_overlap": 0.42,
        "resize_ms": 0.082,  # Historical 82.42 us claim; no calibration artifact verified here.
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
                "gbps": 3.4,  # Historical cross-domain SP8 bottleneck claim; unverified here.
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
            "source": "synthetic_video_prompt_templates",
            "selection": "deterministic_sha256",
            "samples": [{"sample_id": f"vid-{i:03d}", "prompt": f"Cinematic aerial shot of scene {i}"} for i in range(250)],
        },
        "image": {
            "kind": "image",
            "source": "synthetic_image_prompt_templates",
            "selection": "deterministic_sha256",
            "samples": [{"sample_id": f"img-{i:03d}", "prompt": f"High resolution studio photo {i}"} for i in range(250)],
        },
    }


def v3_bundle_sensitivity_profile() -> dict:
    """Unverified historical V3 discounts; never selected implicitly by policy."""
    prof = shared_8a100_profile()
    prof["model_assumptions"].append(
        "V3 bundle sensitivity assumes ring_quant_scale=0.51, reduced decode/image times and larger cache skips; none are validated speedups."
    )
    prof["ring_quant_scale"] = 0.51  # Assumed Opt #5 ring quantization savings.
    for k, finish_v3, skip_v3 in [
        ("wan21_17f", 155.0, {"0.08": 0.28, "0.12": 0.36, "0.16": 0.44, "0.24": 0.53, "0.30": 0.58}),
        ("wan21_33f", 235.0, {"0.08": 0.29, "0.12": 0.38, "0.16": 0.46, "0.24": 0.55, "0.30": 0.60}),
        ("wan21_81f", 480.0, {"0.08": 0.30, "0.12": 0.40, "0.16": 0.48, "0.24": 0.57, "0.30": 0.62}),
    ]:
        prof["classes"][k]["finish_ms"] = finish_v3  # Assumed Opt #8/#10 decode and overlap savings.
        prof["classes"][k]["cache_skip_table"] = skip_v3  # Assumed Opt #7/#9 Taylor/CFG reuse savings.
    # Assumed Opt #6/#11 image kernel/cache/batching savings, not isolated effects.
    prof["classes"]["sd35"]["service_ms"] = 540.0
    prof["classes"]["sd35"]["finish_ms"] = 48.0
    prof["classes"]["sd35"]["batch_ms"] = {
        "1": 540.0,
        "2": 780.0,
        "4": 1240.0,
    }
    return prof


def _sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _simulation_options(policy: str, options: dict) -> dict:
    # Explicit execution/candidate defaults avoid policy-dependent batching,
    # tiered-memory discounts, placement and widening leaking into DP contrasts.
    defaults = {
        **SHARED_EXECUTION_OPTIONS, "forms": FORMS_2D,
        "widen_efficiency": 0.0, "image_placement": "first", "prefer_exact": 0.0,
        "lazy_preemption": True, "feasibility_degree": "free", "order": "edf",
        "demote_late": True, "late_short_ms": 2000.0,
        "pin_started": policy in ("vanilla_xdit", "tridentserve_stage"),
        "hold_diffuse": policy in ("vanilla_xdit", "tridentserve_stage"),
        "reserve_image_gpus": 2 if policy == "static_partition" else 0,
        "strict_partition": policy == "static_partition",
        "round_steps": 1000 if policy == "tridentserve_stage" else 5,
        "cache_thresholds": APPROX_THRESHOLDS if policy in ("heterodit_approx", "heterodit_v3") else [0.0],
    }
    if "degrees" in options:
        defaults.pop("forms")
    return copy.deepcopy({**defaults, **options})


def run_sim(workload: list[dict], policy: str, options: dict) -> dict:
    """Run a proxy simulation, retaining existing metrics plus auditable evidence.

    Options may explicitly select profile_scenario='v3_bundle_sensitivity'.
    Otherwise every policy, including V3, receives exactly the shared profile.
    """
    clean_opts = copy.deepcopy(options)
    if clean_opts.pop("use_v3_profile", False):
        raise ValueError("use_v3_profile is retired; explicitly select profile_scenario='v3_bundle_sensitivity'")
    profile_scenario = clean_opts.pop("profile_scenario", "shared")
    if profile_scenario not in ("shared", "v3_bundle_sensitivity"):
        raise ValueError("unknown profile_scenario: " + str(profile_scenario))
    profile = v3_bundle_sensitivity_profile() if profile_scenario == "v3_bundle_sensitivity" else shared_8a100_profile()
    clean_opts = _simulation_options(policy, clean_opts)
    approximate = any(tau > 0 for tau in clean_opts["cache_thresholds"])
    comparison_group = "approximate_model" if approximate else "scheduling_exact"
    if profile_scenario != "shared" or policy == "heterodit_v3":
        comparison_group = "bundle_sensitivity"
    payload = {
        "gpus": 8,
        "policy": policy,
        "profile": profile,
        "options": clean_opts,
        "workload": workload,
    }
    proc = subprocess.run(
        [str(BINARY), "rpc", "simulate"],
        input=json.dumps(payload, allow_nan=False),
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(proc.stdout)
    out.update(
        evidence_type="simulation", implementation_type="proxy", policy_label=f"{policy} [proxy]",
        cost_source="mixed", cost_source_details=copy.deepcopy(profile["cost_source_details"]),
        model_assumptions=list(profile["model_assumptions"]), comparison_group=comparison_group,
        profile_scenario=profile_scenario, profile_sha256=_sha256(profile), trace_sha256=_sha256(workload),
        simulation_options=clean_opts,
    )
    return out


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
            evidence_runs = []
            for seed in seeds:
                trace = build_trace(
                    workload_classes(),
                    prompt_pools(),
                    [sc["window"]],
                    slots=8,
                    seed=seed,
                )
                out = run_sim(trace["workload"], pol_id, opts)
                evidence_runs.append({"seed": seed, "profile_sha256": out["profile_sha256"],
                                      "trace_sha256": out["trace_sha256"], "manifest": trace["manifest"]})
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
                "label": pol_label if "proxy" in pol_label.lower() else pol_label + " [proxy]",
                **{field: out[field] for field in (
                    "evidence_type", "implementation_type", "cost_source", "cost_source_details",
                    "model_assumptions", "comparison_group", "profile_scenario", "profile_sha256", "simulation_options",
                )},
                "trace_sha256": [run["trace_sha256"] for run in evidence_runs],
                "runs": evidence_runs,
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
            "evidence_type": "simulation",
            "policies": sc_rows,
        }
    return out_scenarios


def main_policy_specs() -> list[tuple[str, str, str, dict]]:
    # Only scheduling constraints differ. All elastic policies see the same
    # candidate forms; fixed-policy IDs still enforce their fixed placement.
    rows = [
        ("vanilla_xdit", "Vanilla-xDiT (fixed SP4)", {}),
        ("static_partition", "Static-Partition (2 image + 6 video slots)", {"hold_diffuse": True}),
        ("static_sp1", "Static-SP1 (EDF, no preemption)", {"hold_diffuse": True, "pin_started": True}),
        ("tridentserve_stage", "TridentServe-Stage (stage scheduling)", {}),
        ("round_elastic", "TetriServe-Round (5-step rounds)", {}),
        ("vllm_omni", "vLLM-Omni / GF-DiT (step scheduling)", {}),
        ("genserve_40gb", "GenServe (step DP; no HBM validation)", {}),
        ("block_elastic", "DitServe (block scheduling)", {}),
        ("heterodit_exact", "HeteroDiT-Exact (DP, tau=0)", {}),
        ("heterodit_approx", "HeteroDiT-Approx (assumed tau-cache savings)", {}),
    ]
    return [(policy, policy, label + " [proxy]", _simulation_options(policy, opts))
            for policy, label, opts in rows]


def ablation_policy_specs() -> list[tuple[str, str, str, dict]]:
    options = _simulation_options("step_elastic", {"forms": FORMS_1D})
    rows = [("A0_baseline_1dsp", "step_elastic", "A0: Step + 1D-SP [proxy]", copy.deepcopy(options))]
    # A3 uses the Approx entrypoint with tau=0, which disables both cache choice
    # and its laxity fallback. A3->A4 thus changes only cache_thresholds, not DP,
    # placement, widening, batching, costs or any other policy default.
    changes = [
        ("A1_plus_tiered_mem", "step_elastic", "A1: + Tiered prefetch assumption [proxy]", {"tiered_memory": True}),
        ("A2_plus_2d_usp", "step_elastic", "A2: + 2D USP candidates [proxy]", {"forms": FORMS_2D}),
        ("A3_heterodit_exact", "heterodit_approx", "A3: + DP (tau=0) [proxy]", {}),
        ("A4_heterodit_approx", "heterodit_approx", "A4: + Tau-cache assumption [proxy]", {"cache_thresholds": APPROX_THRESHOLDS}),
        ("A5_plus_image_batch", "heterodit_approx", "A5: + Image batching assumption [proxy]", {"image_batch": 4}),
    ]
    for key, policy, label, change in changes:
        options.update(copy.deepcopy(change))
        rows.append((key, policy, label, copy.deepcopy(options)))
    return rows


def sensitivity_policy_specs() -> list[tuple[str, str, str, dict]]:
    # Historical multi-component V3 inputs retained only as an explicit bundle.
    baseline = ablation_policy_specs()[-1][3]
    return [
        ("S0_shared_approx", "heterodit_approx", "Shared-cost batched Approx [proxy]", baseline),
        ("S1_v3_bundle", "heterodit_v3", "Unverified V3 bundle sensitivity [proxy]", {
            **copy.deepcopy(baseline), "profile_scenario": "v3_bundle_sensitivity", "quantum_ms": 140.0,
            "cache_thresholds": [0.0, 0.12, 0.16, 0.24, 0.30],
        }),
    ]


def evaluate_suite() -> dict:
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
    ablation = _eval_policies(ablation_policy_specs(), [scenarios[2]], seeds)[scenarios[2]["name"]]
    ablation.update(design="cumulative_one_factor", factors=["tiered_memory", "forms", "DP", "cache_thresholds", "image_batch"])
    sensitivity = _eval_policies(sensitivity_policy_specs(), [scenarios[2]], seeds)[scenarios[2]["name"]]
    sensitivity.update(design="bundle_sensitivity", orthogonal=False,
                       note="Joint unverified V3 discounts and scheduler settings; no component attribution.")
    profiles = {"shared": shared_8a100_profile(), "v3_bundle_sensitivity": v3_bundle_sensitivity_profile()}
    return {
        "evidence_type": "simulation", "implementation_type": "proxy", "cost_source": "mixed",
        "cost_source_details": copy.deepcopy(COST_SOURCE_DETAILS), "model_assumptions": list(MODEL_ASSUMPTIONS),
        "claim_scope": "Modeled latency/SLO only; no hardware throughput, quality, HBM or losslessness validation.",
        "profiles": {name: {"profile_sha256": _sha256(profile), "profile": profile} for name, profile in profiles.items()},
        "seeds": seeds,
        "scenarios": _eval_policies(main_policy_specs(), scenarios, seeds),
        "ablation": ablation,
        "sensitivity": sensitivity,
    }


if __name__ == "__main__":
    summary = evaluate_suite()
    out_dir = Path(os.environ["RUN_DIR"]) if "RUN_DIR" in os.environ else (ROOT / ".cache")
    out_path = out_dir / "hetero_8a100_study_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("evidence_type=simulation; all strategies are proxies; cost_source=mixed (uncalibrated assumptions included)")
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
    print("\n=== Simulation: 6-stage cumulative one-factor ablation (Gamma CV=3.0, Load=1.25) ===")
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
    print("\n=== Simulation: unverified V3 bundle sensitivity (NOT orthogonal) ===")
    for row in summary["sensitivity"]["policies"].values():
        print(f"{row['label']}: modeled SLO={row['total_slo_pct']:.2f}%; profile_sha256={row['profile_sha256']}")
