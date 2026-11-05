#!/usr/bin/env python3
"""Generate frozen native serving studies from a fitted profile and sealed prompts."""

import argparse
import copy
import json
import math
import random
import re
from pathlib import Path

from cluster_config import (  # DEFAULT_CONTINUOUS_BATCH (0, off) is re-exported
    DEFAULT_CONTINUOUS_BATCH,
    MEMORY_STRATEGIES,
    MODELS,
    allocated_uuids,
    applied_memory_strategy,
    continuous_batch_settings,
    default_sd35_batch_step_ms,
    make_workers,
)
from ditserve.evaluation.common import read_json, verify_seal
from ditserve.evaluation.workloads import build_trace, validate_trace
# Display tables live in a stdlib-only module so paper_results.py does not need
# workers/; they stay importable from here.
from policy_display import IN_FRAMEWORK_MECHANISM_DISPLAY, POLICY_DISPLAY


STATIC_POLICY = re.compile(r"^static-(?:sp1|cfg2|sp2|cfg2sp2|sp3|cfg2sp3|sp4|cfg2sp4|sp6|sp8)$")
IN_FRAMEWORK_COMPARISON_DECLARATION = {
    "enabled": True,
    "comparison_mode": "same_executor_mechanism_comparison",
    "statement": (
        "All policies execute on the same verified DitServe native runtime, worker pool, "
        "SP/CFG collective transport, and calibrated latency profile to isolate scheduling "
        "mechanisms (optimal Static-<Form>, TetriServe-policy, SBP-ESP, EDF-BF vs. DitServe) "
        "on the Wan + SD3.5 mixed-workload main table; the in-house heuristics EDF-CFG2 and "
        "CFG2-First run only in the ablation suite."
    ),
    "canonical_mechanisms": [
        "static-cfg2sp4",
        "tetriserve",
        "sbp-esp",
        "edf-bestfit",
        "ditserve",
    ],
}
# In-house heuristics that isolate DitServe mechanisms; never part of the full
# baseline matrix (main/sweep), only of the ablation suite.
ABLATION_ONLY_POLICIES = ("edf-cfg2", "cfg2-first")


def in_framework_main_policies(best_static_form="static-cfg2sp4"):
    """Return the 5-policy in-framework main table mechanism comparison list."""
    if not isinstance(best_static_form, str) or not STATIC_POLICY.fullmatch(best_static_form):
        raise ValueError(f"best_static_form must match STATIC_POLICY, got {best_static_form!r}")
    return [best_static_form, "tetriserve", "sbp-esp", "edf-bestfit", "ditserve"]


SUITES = {
    # Paper baselines against DitServe (full matrix).
    "main": ["xdit", "tetriserve", "sbp-esp", "edf-bestfit", "ditserve"],
    # Same-executor in-framework mechanism comparison on the shared native runtime.
    "in-framework-main": ["static-cfg2sp4", "tetriserve", "sbp-esp", "edf-bestfit", "ditserve"],
    # Combined main table with both xDiT (static partition, CFG=1) and optimal Static-<Form>.
    "main-all": ["xdit", "static-cfg2sp4", "tetriserve", "sbp-esp", "edf-bestfit", "ditserve"],
    # DitServe with one mechanism removed; CFG2-First drops sub-step preemption and
    # pair-fit placement, EDF-CFG2 has all of them removed.
    "ablation": ["ditserve", "abl-no-block", "abl-no-prune", "abl-no-mindeg", "abl-first-fit",
                 "block-cfg2", "cfg2-first", "edf-cfg2", "ditserve-v1"],
    # Same fixed form with and without sub-step preemption (digest-checked pair).
    "preempt": ["edf-cfg2", "block-cfg2"],
    # Paper baselines + DitServe for Rate/SLO/CV/GPU sweeps.
    "sweep": ["xdit", "tetriserve", "sbp-esp", "edf-bestfit", "ditserve"],
}
SUITES["main+ablation"] = list(dict.fromkeys(SUITES["main"] + SUITES["ablation"]))
DEFAULT_POLICIES = SUITES["main"]
APPROXIMATE_POLICIES = ("ditserve-approx", "ditserve-v1-approx", "heterodit_v5",
                        "fold-cfg2sp3", "approx-cfg2sp3")
FORMS = ["sp1", "cfg2", "sp2", "cfg2sp2", "sp3", "cfg2sp3", "sp4", "cfg2sp4", "sp6", "sp8"]
FORM_DEGREES = {"sp1": 1, "cfg2": 2, "sp2": 2, "sp3": 3, "cfg2sp2": 4,
                "sp4": 4, "cfg2sp3": 6, "sp6": 6, "cfg2sp4": 8, "sp8": 8}


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
                         output_mode="check", require_profile_match=False, collective_backend="nccl", cv=1.0,
                         image_batch=1, policy_order="forward", order_seed=None, async_rank0_decode=False,
                         continuous_batch=None, sequence_packing=None, bubble_free=None, memory_strategy=None):
    """Build a frozen native serving study.

    SD3.5 continuous batching and worker memory strategies are explicit opt-in:
    continuous_batch None or 0 keeps atomic image execution for every policy and
    workers with continuous_batch/sequence_packing/bubble_free off, whatever
    the profile contains.
    """
    policies = list(DEFAULT_POLICIES if policies is None else policies)
    requested_policies = list(policies)
    if policy_order not in ("forward", "reverse", "random"):
        raise ValueError("policy_order must be forward, reverse or random")
    if policy_order == "reverse":
        policies.reverse()
    elif policy_order == "random":
        random.Random(seed if order_seed is None else order_seed).shuffle(policies)
    if not isinstance(image_batch, int) or isinstance(image_batch, bool) or not 1 <= image_batch <= 8:
        raise ValueError("image_batch must be an integer in [1, 8]")
    eff_cb, packing, bubble = continuous_batch_settings(continuous_batch, sequence_packing, bubble_free)
    if eff_cb > 0 and image_batch > 1:
        raise ValueError("continuous_batch and atomic image_batch are mutually exclusive")
    applied_memory = None if memory_strategy is None else applied_memory_strategy(memory_strategy)
    cb_cost_provenance = None
    if eff_cb > 0:
        sd_cls = profile["classes"]["sd35-preflight"]
        table = sd_cls.get("batch_step_ms")
        if table is not None and (not isinstance(table, dict) or "1" not in table
                                  or max(int(key) for key in table) < eff_cb):
            raise ValueError(f"continuous_batch={eff_cb} needs sd35-preflight.batch_step_ms with batch 1 "
                             f"and a largest size >= {eff_cb}; no synthetic extension of a measured table")
        cb_cost_provenance = {key: "profile" for key in ("steps", "prepare_ms", "finish_ms", "batch_step_ms")}
        missing = [key for key in cb_cost_provenance if key not in sd_cls]
        if missing:
            # Explicit opt-in only: placeholders are labelled synthetic, never measured.
            profile = copy.deepcopy(profile)
            sd_cls = profile["classes"]["sd35-preflight"]
            sd_steps = int(sd_cls.get("steps", 28))
            synthetic = {
                "steps": sd_steps, "prepare_ms": 15.0, "finish_ms": 65.0,
                "batch_step_ms": default_sd35_batch_step_ms(float(sd_cls["service_ms"]), steps=sd_steps,
                                                            resolution=1024),
            }
            for key in missing:
                sd_cls[key] = synthetic[key]
                cb_cost_provenance[key] = ("synthetic_default_table" if key == "batch_step_ms"
                                           else "synthetic_placeholder")
            source = profile.setdefault("source", {})
            source["image_cost_provenance"] = dict(source.get("image_cost_provenance", {}),
                                                   **{key: cb_cost_provenance[key] for key in missing})
    if image_batch > 1:
        table = profile["classes"]["sd35-preflight"].get("batch_ms", {})
        for size in range(2, image_batch + 1):
            value = table.get(str(size))
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"image_batch={image_batch} requires measured sd35-preflight.batch_ms[{size!r}]")
    if collective_backend not in ("nccl", "gfc"):
        raise ValueError("collective_backend must be nccl or gfc")
    forms = list(FORMS if forms is None else forms)
    if any(name not in FORM_DEGREES for name in forms):
        raise ValueError("unknown parallel form")
    if model == "wan21":
        forms = [name for name in forms if name != "sp8"]
    forms = [name for name in forms if FORM_DEGREES.get(name, 99) <= len(gpu_uuids)]
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("policies must be nonempty and unique")
    if not forms or len(set(forms)) != len(forms):
        raise ValueError("forms must be nonempty and unique")
    if output_mode not in ("check", "digest") or quality_per_kind and output_mode != "check":
        raise ValueError("digest trace mode requires quality_per_kind=0")
    profile_uuids = profile["source"].get("gpu_uuids")
    profile_match = bool(profile_uuids) and profile_uuids[:len(gpu_uuids)] == list(gpu_uuids)
    if require_profile_match and not profile_match:
        raise ValueError("calibration GPU UUIDs are absent or differ from this allocation")
    if any(name in ("heterodit_approx", "heterodit_v3", "diag-feature-cache", "diag-v3") for name in policies):
        raise ValueError("native serving's CFG-fold evaluation supports only ditserve-approx")
    approximate = any(name in APPROXIMATE_POLICIES for name in policies)
    if approximate and any(name not in APPROXIMATE_POLICIES for name in policies):
        raise ValueError("exact and approximate arms must use separate controlled runs")
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
        memory_strategy=memory_strategy,
        continuous_batch=eff_cb,
        sequence_packing=packing,
        bubble_free=bubble,
    )
    for item in warmup:
        item["spec"]["steps"] = 1
    model_key = model + "-preflight"
    video_cost = profile["classes"][model_key]
    gpu_service_ms = (video_cost.get("prepare_ms", 0.0)
                      + video_cost["steps"] * video_cost["step_compute_ms"] + video_cost.get("finish_ms", 0.0))
    service_ms = video_cost.get("encode_ms", 0.0) + gpu_service_ms
    classes = {
        model_key: dict(video_cost, **{
            "class": model_key, "kind": "video", "service_ms": service_ms, "gpu_service_ms": gpu_service_ms,
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
    elif pattern == "gamma":
        window["cv"] = float(cv)
    trace = build_trace(classes, prompt_pools, [window], slots=len(gpu_uuids), seed=seed,
                        quality_per_kind=quality_per_kind, output_mode=output_mode)
    validate_trace(trace)
    provenance = {
        "path": "generated:diffusiondb+vidprom",
        "sha256": trace["manifest"]["workload_sha256"],
        "manifest": trace["manifest"],
    }
    gpus = len(gpu_uuids)
    phase_options = {
        "sp_transport": "export",
        "forms": forms,
        "quantum_blocks": 4,
        "async_rank0_decode": async_rank0_decode,
        "image_batch": image_batch,
        "tiered_memory": False,
        "doorbell": doorbell,
        "partition_image_gpus": image_share,
        "collective_backend": collective_backend,
    }
    if eff_cb > 0:
        phase_options["continuous_batch"] = eff_cb
        phase_options["sequence_packing"] = packing
        phase_options["bubble_free"] = bubble
    benchmark_meta = {
        "native_serving": True,
        "performance_qualified": False,
        "quality_qualified": False,
        "exact_mode": not approximate,
        "models": [selected["name"], "SD3.5-Medium"],
        "gpus": gpus,
        "seed": seed,
        "load": load,
        "pattern": pattern,
        "cv": float(cv) if pattern == "gamma" else None,
        "video_fraction": video_fraction,
        "count": count,
        "video_slo_scale": video_slo_scale,
        "image_slo_scale": image_slo_scale,
        "output_mode": output_mode,
        "profile_placement": {
            "required": require_profile_match, "matches": profile_match,
            "gpu_uuids": profile_uuids or [],
        },
        "trace": provenance,
        "policies": policies,
        "requested_policies": requested_policies,
        "policy_order": {"mode": policy_order, "seed": seed if order_seed is None else order_seed},
        "image_batch": {"max_requests": image_batch, "scope": "same atomic request batching limit for every policy; each request generates one image"},
        "policy_display": {name: POLICY_DISPLAY.get(name, name) for name in policies},
        "in_framework_policy_comparison": dict(
            copy.deepcopy(IN_FRAMEWORK_COMPARISON_DECLARATION),
            mechanism_display={
                name: IN_FRAMEWORK_MECHANISM_DISPLAY.get(name, POLICY_DISPLAY.get(name, name))
                for name in policies
            },
        ),
        "collective_backend": collective_backend,
        "partition_image_gpus": {"value": image_share, "rule": "expected image share of GPU time, even video remainder"},
    }
    if eff_cb > 0:
        benchmark_meta["continuous_batch"] = {
            "max_requests": eff_cb,
            "sequence_packing": packing,
            "bubble_free": bubble,
            "cost_provenance": cb_cost_provenance,
        }
    if applied_memory is not None:
        benchmark_meta["memory_strategy"] = applied_memory
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
            "options": phase_options,
            "matrix": {
                "/policy": policies,
                "/workload": [trace["workload"]],
                "/trace_provenance": [provenance],
            },
            "artifact_dependencies": ([{"consumer": "native_quality", "closed": False}] if quality_per_kind else []),
        }],
        "benchmark": benchmark_meta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--gpu-uuids")
    parser.add_argument("--num-gpus", type=int, choices=[2, 4, 6, 8],
                        help="use the first N allocated GPUs for intra-node GPU scaling studies")
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--load", type=float, default=1.0)
    parser.add_argument("--pattern", choices=["poisson", "gamma", "bursty", "offline_all"], default="poisson")
    parser.add_argument("--cv", type=float, default=1.0, help="inter-arrival coefficient of variation for pattern=gamma")
    parser.add_argument("--video-fraction", type=float, default=0.2)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--suite", choices=sorted(SUITES), help="named policy list; --policies overrides it")
    parser.add_argument("--policy-order", choices=["forward", "reverse", "random"], default="random")
    parser.add_argument("--order-seed", type=int, help="independent reproducible policy order seed; defaults to trace seed")
    parser.add_argument("--image-batch", type=int, default=1,
                        help="maximum atomic image requests per batch, shared by all policies; >1 requires measured batch_ms")
    parser.add_argument("--continuous-batch", type=int, default=None,
                        help="opt-in step-level continuous image batch size in [0, 8]; default off (never "
                             "enabled from the profile); mutually exclusive with --image-batch > 1")
    parser.add_argument("--sequence-packing", action=argparse.BooleanOptionalAction, default=None,
                        help="token sequence packing for SD3.5 continuous batches (default: on iff continuous batching)")
    parser.add_argument("--bubble-free", action=argparse.BooleanOptionalAction, default=None,
                        help="overlap T5 encode and VAE decode with continuous rounds (default: on iff continuous batching)")
    parser.add_argument("--memory-strategy", choices=MEMORY_STRATEGIES,
                        help="opt-in worker memory strategy; only worker-implemented strategies are accepted")
    parser.add_argument("--quality-arm", choices=["exact", "fixed", "curvature"],
                        help="controlled all-video cfg2sp3 quality arm, with identical runtime settings")
    parser.add_argument("--policies")
    parser.add_argument("--forms")
    parser.add_argument("--video-slo-scale", type=float, default=1.5)
    parser.add_argument("--image-slo-scale", type=float, default=2.5)
    parser.add_argument("--quality-per-kind", type=int, default=0)
    parser.add_argument("--curvature", action="store_true")
    parser.add_argument("--curvature-threshold", type=float, default=0.30)
    parser.add_argument("--min-full-cfg-steps", type=int, default=12)
    parser.add_argument("--max-folded-steps", type=int, default=7)
    parser.add_argument("--cfg-vel-alpha", type=float, default=0.0)
    parser.add_argument("--fold-steps", help="comma-separated fixed CFG folding steps")
    parser.add_argument("--doorbell", action="store_true")
    parser.add_argument("--no-preemption-diagnostic", action="store_true",
                        help="preempt suite with video-only low load; verify zero actual stops using native_exact --no-preemption")
    parser.add_argument("--output-mode", choices=["check", "digest"], default="check")
    parser.add_argument("--require-profile-match", action="store_true")
    parser.add_argument("--collective-backend", choices=["nccl", "gfc"], default="nccl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", help="result and ledger identity; default derives from model, seed and load")
    args = parser.parse_args()
    if args.experiment is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.experiment):
        parser.error("--experiment must be a safe nonempty result identity")
    if args.curvature and args.fold_steps:
        parser.error("--curvature and --fold-steps are mutually exclusive")
    uuids = args.gpu_uuids.split(",") if args.gpu_uuids else allocated_uuids()
    if args.num_gpus is not None:
        if args.num_gpus > len(uuids):
            parser.error(f"--num-gpus={args.num_gpus} exceeds allocated GPUs ({len(uuids)})")
        uuids = uuids[:args.num_gpus]
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    pools = {
        kind: verify_seal(read_json(args.datasets / f"{name}.json"))
        for kind, name in (("image", "diffusiondb"), ("video", "vidprom"))
    }
    policies = (args.policies.split(",") if args.policies
                else SUITES[args.suite] if args.suite else DEFAULT_POLICIES)
    forms = args.forms.split(",") if args.forms else list(FORMS)
    if args.suite == "preempt" or args.no_preemption_diagnostic:
        if args.policies and set(policies) != set(SUITES["preempt"]):
            parser.error("controlled preemption requires exactly edf-cfg2 and block-cfg2")
        if args.forms and forms != ["cfg2"]:
            parser.error("controlled preemption requires --forms cfg2")
        policies, forms, args.doorbell, args.output_mode = SUITES["preempt"], ["cfg2"], True, "digest"
        if args.no_preemption_diagnostic:
            args.video_fraction, args.load, args.pattern = 1.0, 0.1, "poisson"
    if args.quality_arm:
        if args.suite or args.no_preemption_diagnostic or args.policies or args.curvature:
            parser.error("--quality-arm selects its own policy and CFG-fold configuration")
        if args.fold_steps is not None:
            if args.quality_arm != "fixed":
                parser.error("only --quality-arm fixed accepts --fold-steps")
            try:
                quality_steps = [int(value) for value in args.fold_steps.split(",")]
            except ValueError:
                parser.error("fixed quality --fold-steps must be an integer prefix of 12..18")
            if not 1 <= len(quality_steps) <= 7 or quality_steps != list(range(12, 12 + len(quality_steps))):
                parser.error("fixed quality --fold-steps must be a contiguous prefix of 12..18")
        if args.forms and forms != ["cfg2sp3"]:
            parser.error("controlled quality arms require --forms cfg2sp3")
        if len(uuids) < 6:
            parser.error("controlled cfg2sp3 quality arms require at least 6 GPUs")
        policies = ["block-cfg2sp3" if args.quality_arm == "exact" else "fold-cfg2sp3"]
        forms, args.video_fraction, args.quality_per_kind = ["cfg2sp3"], 1.0, args.count
        args.doorbell, args.output_mode = True, "check"
        args.curvature = args.quality_arm == "curvature"
        if args.quality_arm == "fixed" and args.fold_steps is None:
            args.fold_steps = "12,13,14,15,16,17,18"
    cfg_fold = None
    if args.curvature:
        cfg_fold = {
            "curvature_threshold": args.curvature_threshold,
            "min_full_cfg_steps": args.min_full_cfg_steps,
            "max_curvature_guard": 1.65,
            "cfg_vel_alpha": args.cfg_vel_alpha,
            "cfg_damping": 0.96,
            "folded_steps": list(range(args.min_full_cfg_steps, 19)),
            "max_folded_steps": args.max_folded_steps,
            "final_step_exact_anchor": True,
            "force_exact_cond_on_fold": True,
        }
    elif args.fold_steps:
        fixed_steps = [int(value) for value in args.fold_steps.split(",")]
        if len(set(fixed_steps)) != len(fixed_steps):
            parser.error("--fold-steps must not contain duplicates")
        cfg_fold = {
            "folded_steps": fixed_steps,
            "max_folded_steps": len(fixed_steps),
            "final_step_exact_anchor": True,
            "force_exact_cond_on_fold": True,
            "cfg_vel_alpha": args.cfg_vel_alpha,
            "cfg_damping": 0.96,
        }
    config = build_serving_config(
        args.model, uuids, profile, pools, seed=args.seed, load=args.load,
        pattern=args.pattern, video_fraction=args.video_fraction, count=args.count,
        policies=policies, video_slo_scale=args.video_slo_scale,
        image_slo_scale=args.image_slo_scale, quality_per_kind=args.quality_per_kind,
        cfg_fold=cfg_fold, forms=forms, doorbell=args.doorbell, output_mode=args.output_mode,
        require_profile_match=args.require_profile_match, collective_backend=args.collective_backend, cv=args.cv,
        image_batch=args.image_batch, policy_order=args.policy_order, order_seed=args.order_seed,
        continuous_batch=args.continuous_batch, sequence_packing=args.sequence_packing, bubble_free=args.bubble_free,
        memory_strategy=args.memory_strategy,
    )
    config["benchmark"]["quality_arm"] = args.quality_arm
    if args.experiment is not None:
        config["experiment"] = args.experiment
    config["benchmark"]["preemption_diagnostic"] = ("no_preemption" if args.no_preemption_diagnostic
                                                     else "contention" if args.suite == "preempt" else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output), "experiment": config["experiment"],
        "policies": config["benchmark"]["policies"], "trace_sha256": config["benchmark"]["trace"]["sha256"],
        "exact_mode": config["benchmark"]["exact_mode"],
    }))


if __name__ == "__main__":
    main()
