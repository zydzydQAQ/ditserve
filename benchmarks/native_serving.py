#!/usr/bin/env python3
"""Generate frozen native serving studies from a fitted profile and sealed prompts."""

import argparse
import json
from pathlib import Path

from cluster_config import MODELS, allocated_uuids, make_workers
from ditserve.evaluation.common import read_json, verify_seal
from ditserve.evaluation.workloads import build_trace, validate_trace


# Paper tables (see docs/experiments.md and src/ditserve/control/policy.cpp, the authoritative
# registry). Preprint-only methods are named by method; TetriServe (ASPLOS'26) and xDiT by system.
POLICY_DISPLAY = {
    "xdit": "xDiT", "tetriserve": "TetriServe (re-impl.)", "sbp-esp": "SBP-ESP", "edf-bestfit": "EDF-BF",
    "edf-cfg2": "EDF-CFG2", "cfg2-first": "CFG2-First", "partition": "Static-Partition",
    "ditserve": "DitServe", "ditserve-approx": "DitServe-Approx",
    "abl-no-block": "DitServe w/o block preemption", "abl-no-prune": "DitServe w/o form pruning",
    "abl-no-mindeg": "DitServe w/o CFG-pair admission", "abl-first-fit": "DitServe w/o pair-fit placement",
    "ditserve-v1": "DitServe-v1 (knapsack)", "block-cfg2": "Block-cfg2",
}
SUITES = {
    # External methods and simple heuristics against DitServe.
    "main": ["xdit", "tetriserve", "sbp-esp", "edf-bestfit", "edf-cfg2", "cfg2-first", "ditserve"],
    # DitServe with one mechanism removed; EDF-CFG2 has all of them removed.
    "ablation": ["ditserve", "abl-no-block", "abl-no-prune", "abl-no-mindeg", "abl-first-fit",
                 "block-cfg2", "edf-cfg2", "ditserve-v1"],
    # Same fixed form with and without sub-step preemption (digest-checked pair).
    "preempt": ["edf-cfg2", "block-cfg2"],
}
SUITES["main+ablation"] = list(dict.fromkeys(SUITES["main"] + SUITES["ablation"]))
DEFAULT_POLICIES = SUITES["main"]
APPROXIMATE_POLICIES = ("ditserve-approx", "ditserve-v1-approx", "heterodit_v5")
FORMS = ["sp1", "cfg2", "sp2", "cfg2sp2", "sp3", "cfg2sp3", "sp4", "cfg2sp4", "sp6", "sp8"]


def partition_image_gpus(gpus, video_fraction, video_cost, image_service_ms):
    """GPUs the partitioned policies (xDiT, Static-Partition) reserve for images.

    Proportional to the trace's expected image share of GPU time (videos priced at their SP1
    GPU time, which a CFG pair spends as two halves), at least one, leaving an even number of
    GPUs for CFG pairs. Fixed before any run, from the profile and the mix only.
    """
    video_gpu_ms = video_cost["steps"] * video_cost["step_compute_ms"] + video_cost.get("finish_ms", 0.0)
    image_demand = (1.0 - video_fraction) * image_service_ms
    total = image_demand + video_fraction * video_gpu_ms
    images = max(1, min(gpus - 2, round(gpus * image_demand / total))) if total > 0 else 1
    if (gpus - images) % 2:
        images = images + 1 if images + 1 <= gpus - 2 else images - 1
    return max(1, images)


def build_serving_config(model, gpu_uuids, profile, prompt_pools, *, seed, load, pattern,
                         video_fraction, count, policies=None, video_slo_scale=1.5,
                         image_slo_scale=2.5, quality_per_kind=0, cfg_fold=None, forms=None, doorbell=False,
                         output_mode="check", require_profile_match=False):
    policies = policies or DEFAULT_POLICIES
    forms = forms or FORMS
    if model == "wan21":
        forms = [name for name in forms if name != "sp8"]
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("policies must be nonempty and unique")
    if not forms or len(set(forms)) != len(forms):
        raise ValueError("forms must be nonempty and unique")
    if output_mode not in ("check", "digest") or quality_per_kind and output_mode != "check":
        raise ValueError("digest trace mode requires quality_per_kind=0")
    profile_uuids = profile["source"].get("gpu_uuids")
    profile_match = profile_uuids == list(gpu_uuids)
    if require_profile_match and not profile_match:
        raise ValueError("calibration GPU UUIDs are absent or differ from this allocation")
    if any(name in ("heterodit_approx", "heterodit_v3", "diag-feature-cache", "diag-v3") for name in policies):
        raise ValueError("native serving's CFG-fold evaluation supports only ditserve-approx")
    approximate = any(name in APPROXIMATE_POLICIES for name in policies)
    if approximate and cfg_fold is None:
        raise ValueError("approximate serving requires an explicit cfg_fold configuration")
    if not approximate and cfg_fold is not None:
        raise ValueError("exact serving must not carry cfg_fold configuration")
    selected, workers, slots, warmup = make_workers(
        model, gpu_uuids, video_spec=profile["source"]["shape"], image_spec={
            "height": 1024, "width": 1024, "steps": 28, "frames": 1, "batch_size": 1,
            "guidance_scale": 4.5, "negative_prompt": "blurry, low quality",
            "max_sequence_length": 256, "prompt": "A lighthouse at dusk.", "seed": 202,
        }, cfg_fold=cfg_fold,
    )
    for item in warmup:
        item["spec"]["steps"] = 1
    model_key = model + "-preflight"
    video_cost = profile["classes"][model_key]
    service_ms = (video_cost.get("prepare_ms", 0.0)
                  + video_cost["steps"] * video_cost["step_compute_ms"] + video_cost.get("finish_ms", 0.0))
    classes = {
        model_key: dict(video_cost, **{
            "class": model_key, "kind": "video", "service_ms": service_ms,
            "slo_scale": video_slo_scale, "spec": profile["source"]["shape"],
        }),
        "sd35-preflight": {
            "class": "sd35-preflight", "kind": "image",
            "service_ms": profile["classes"]["sd35-preflight"]["service_ms"],
            "gpu_service_ms": profile["classes"]["sd35-preflight"]["service_ms"],
            "slo_scale": image_slo_scale,
            "spec": {
                "height": 1024, "width": 1024, "steps": 28, "frames": 1, "batch_size": 1,
                "guidance_scale": 4.5, "negative_prompt": "blurry, low quality",
                "max_sequence_length": 256,
            },
        },
    }
    image_share = partition_image_gpus(len(gpu_uuids), video_fraction, video_cost,
                                       profile["classes"]["sd35-preflight"]["service_ms"])
    window = {
        "count": count, "video_fraction": video_fraction, "pattern": pattern, "load": load,
    }
    if pattern == "bursty":
        window.update(bursts=2, burst_active_fraction=0.25)
    trace = build_trace(classes, prompt_pools, [window], slots=len(gpu_uuids), seed=seed,
                        quality_per_kind=quality_per_kind, output_mode=output_mode)
    validate_trace(trace)
    provenance = {
        "path": "generated:diffusiondb+vidprom",
        "sha256": trace["manifest"]["workload_sha256"],
        "manifest": trace["manifest"],
    }
    gpus = len(gpu_uuids)
    return {
        "schema": "ditserve.study.v1",
        "experiment": f"kairos-{model}-native-serving-s{seed}-l{str(load).replace('.', 'p')}",
        "workers": workers,
        "budget": {
            "limit_gpu_seconds": len(policies) * gpus * 7200,
            "watchdog_seconds": len(policies) * 7200,
            "repair_reserve_gpu_seconds": 0,
            "carry": "none",
        },
        "phases": [{
            "id": "cluster",
            "kind": "cluster",
            "policy": policies[0],
            "profile": profile,
            "occupancy_backend": "nvml",
            "occupancy_thread": True,
            "video_worker_options": {
                "negative_prompt_cache_entries": 2,
                "negative_prompt_cache_bytes": 67108864,
            },
            "gpus": slots,
            "text_prefetch": 1,
            "output_mode": "check",
            "warmup": warmup,
            "options": {
                "sp_transport": "export",
                "forms": forms,
                "quantum_blocks": 4,
                "async_rank0_decode": approximate,
                "image_batch": 1,
                "tiered_memory": False,
                "doorbell": doorbell,
                "partition_image_gpus": image_share,
            },
            "matrix": {
                "/policy": policies,
                "/workload": [trace["workload"]],
                "/trace_provenance": [provenance],
            },
            "artifact_dependencies": ([{"consumer": "native_quality", "closed": False}] if quality_per_kind else []),
        }],
        "benchmark": {
            "native_serving": True,
            "performance_qualified": False,
            "quality_qualified": False,
            "exact_mode": not approximate,
            "models": [selected["name"], "SD3.5-Medium"],
            "seed": seed,
            "load": load,
            "pattern": pattern,
            "video_fraction": video_fraction,
            "count": count,
            "output_mode": output_mode,
            "profile_placement": {
                "required": require_profile_match, "matches": profile_match,
                "gpu_uuids": profile_uuids or [],
            },
            "trace": provenance,
            "policies": policies,
            "policy_display": {name: POLICY_DISPLAY.get(name, name) for name in policies},
            "partition_image_gpus": {"value": image_share, "rule": "expected image share of GPU time, even video remainder"},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--gpu-uuids")
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--load", type=float, default=1.0)
    parser.add_argument("--pattern", choices=["poisson", "gamma", "bursty", "offline_all"], default="poisson")
    parser.add_argument("--video-fraction", type=float, default=0.2)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--suite", choices=sorted(SUITES), help="named policy list; --policies overrides it")
    parser.add_argument("--policies")
    parser.add_argument("--forms", default=",".join(FORMS))
    parser.add_argument("--video-slo-scale", type=float, default=1.5)
    parser.add_argument("--image-slo-scale", type=float, default=2.5)
    parser.add_argument("--quality-per-kind", type=int, default=0)
    parser.add_argument("--curvature", action="store_true")
    parser.add_argument("--fold-steps", help="comma-separated fixed CFG folding steps")
    parser.add_argument("--doorbell", action="store_true")
    parser.add_argument("--output-mode", choices=["check", "digest"], default="check")
    parser.add_argument("--require-profile-match", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.curvature and args.fold_steps:
        parser.error("--curvature and --fold-steps are mutually exclusive")
    uuids = args.gpu_uuids.split(",") if args.gpu_uuids else allocated_uuids()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    pools = {
        kind: verify_seal(read_json(args.datasets / f"{name}.json"))
        for kind, name in (("image", "diffusiondb"), ("video", "vidprom"))
    }
    policies = (args.policies.split(",") if args.policies
                else SUITES[args.suite] if args.suite else DEFAULT_POLICIES)
    cfg_fold = None
    if args.curvature:
        cfg_fold = {
            "curvature_threshold": 0.30,
            "min_full_cfg_steps": 12,
            "max_curvature_guard": 1.65,
            "cfg_vel_alpha": 0.0,
            "cfg_damping": 0.96,
            "folded_steps": list(range(12, 19)),
            "max_folded_steps": 7,
            "final_step_exact_anchor": True,
            "force_exact_cond_on_fold": True,
        }
    elif args.fold_steps:
        cfg_fold = {
            "folded_steps": [int(value) for value in args.fold_steps.split(",")],
            "final_step_exact_anchor": True,
            "force_exact_cond_on_fold": True,
            "cfg_vel_alpha": 0.0,
            "cfg_damping": 0.96,
        }
    config = build_serving_config(
        args.model, uuids, profile, pools, seed=args.seed, load=args.load,
        pattern=args.pattern, video_fraction=args.video_fraction, count=args.count,
        policies=policies, video_slo_scale=args.video_slo_scale,
        image_slo_scale=args.image_slo_scale, quality_per_kind=args.quality_per_kind,
        cfg_fold=cfg_fold, forms=args.forms.split(","), doorbell=args.doorbell, output_mode=args.output_mode,
        require_profile_match=args.require_profile_match,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output), "experiment": config["experiment"],
        "policies": policies, "trace_sha256": config["benchmark"]["trace"]["sha256"],
        "exact_mode": config["benchmark"]["exact_mode"],
    }))


if __name__ == "__main__":
    main()
