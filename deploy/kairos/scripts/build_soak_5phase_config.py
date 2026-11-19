#!/usr/bin/env python3
"""Build a sealed 5-phase 500-request non-stationary mixed-modality soak config
reusing the verified 8x A100 calibration bundle and 720da72 serving stack."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys


FIVE_PHASE_SPEC = [
    {
        "phase_id": "P1_Steady_Baseline",
        "count": 100,
        "rate_rpm": 10.0,
        "video_fraction": 0.5,
        "pattern": "poisson",
        "cv": 1.0,
    },
    {
        "phase_id": "P2_Bursty_Overload_Storm",
        "count": 120,
        "rate_rpm": 28.0,
        "video_fraction": 0.5,
        "pattern": "gamma",
        "cv": 2.0,
    },
    {
        "phase_id": "P3_PostStorm_Image_Tide",
        "count": 100,
        "rate_rpm": 14.0,
        "video_fraction": 0.2,
        "pattern": "poisson",
        "cv": 1.0,
    },
    {
        "phase_id": "P4_Video_Heavy_Tide",
        "count": 80,
        "rate_rpm": 12.0,
        "video_fraction": 0.7,
        "pattern": "gamma",
        "cv": 2.0,
    },
    {
        "phase_id": "P5_Recovery_Endurance",
        "count": 100,
        "rate_rpm": 10.0,
        "video_fraction": 0.5,
        "pattern": "poisson",
        "cv": 1.0,
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks-dir", type=pathlib.Path, required=True)
    ap.add_argument("--protocol", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="wan22")
    ap.add_argument("--bundle", type=pathlib.Path, required=True)
    ap.add_argument("--datasets-dir", type=pathlib.Path, required=True)
    ap.add_argument("--manifests", type=pathlib.Path, required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--slo-scale", type=float, default=1.5)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.benchmarks_dir))
    import formal_benchmark as fb
    from ditserve.evaluation.common import canonical_hash, seal
    from ditserve.evaluation.workloads import build_trace, validate_trace

    p = fb.protocol(args.protocol)
    bundle = fb.read_json(args.bundle)
    active_uuids = list(bundle.get("active_gpu_uuids") or fb.allocated_uuids())[:8]
    origin_uuids = list(bundle.get("origin_gpu_uuids") or bundle.get("gpu_uuids", active_uuids))
    match_mode = bundle.get("calibration_match_mode", "exact_uuid")

    bundle_for_serving = copy.deepcopy({k: v for k, v in bundle.items() if k != "sha256"})
    bundle_for_serving["protocol_sha256"] = canonical_hash(p)
    bundle_for_serving["gpu_uuids"] = active_uuids
    for entry in bundle_for_serving.get("classes", {}).values():
        if isinstance(entry.get("source"), dict):
            entry["source"]["gpu_uuids"] = active_uuids
            if isinstance(entry["source"].get("fit"), dict):
                entry["source"]["fit"]["gpu_uuids"] = active_uuids
    bundle_for_serving = seal(bundle_for_serving)

    pools = {
        kind: fb.read_json(args.datasets_dir / (p["datasets"][kind]["kind"] + ".json"))
        for kind in ("image", "video")
    }
    manifests = fb.read_json(args.manifests)

    # Start from a compliant formal serving config for the target policy
    config = fb.serving(
        p,
        args.model,
        active_uuids,
        bundle_for_serving,
        pools,
        manifests,
        split="test",
        seed=args.seed,
        rate_rpm=12.0,
        video_fraction=0.5,
        pattern="poisson",
        suite="cfg2",
        policies=args.policy,
        slo_scale=args.slo_scale,
        resolution_distribution="uniform",
    )

    # Reconstruct exact calibrated classes dict and test split pools to build the 5-window 500-req trace
    selected_pools = fb.split_pools(pools, p, "test")
    ids = [
        sid
        for sid, shape in p["shapes"].items()
        if shape["scope"] == "formal" and shape["model"] in (args.model, "sd35")
    ]
    classes = {}
    for sid in ids:
        spec = fb.shape_spec(p, sid)
        cost = copy.deepcopy(bundle_for_serving["classes"][sid]["cost"])
        if cost["kind"] == "video":
            gpu_ms = float(cost.get("prepare_ms", 0.0) + spec["steps"] * cost["step_compute_ms"] + cost["finish_ms"])
            service = float(gpu_ms + cost.get("encode_ms", 0.0))
        else:
            gpu_ms = service = float(cost["service_ms"])
            cost, _, _ = fb.sanitize_image_cost(
                cost,
                resolution=int(spec.get("height") or 1024),
                eff_cb=fb.DEFAULT_CONTINUOUS_BATCH,
            )
        classes[sid] = dict(
            cost,
            **{
                "class": sid,
                "spec": spec,
                "service_ms": service,
                "gpu_service_ms": gpu_ms,
                "slo_scale": float(args.slo_scale),
            },
        )

    dist_uniform = {"name": "uniform", "alpha": 0.0}
    windows = []
    for ph in FIVE_PHASE_SPEC:
        w = {
            "count": int(ph["count"]),
            "video_fraction": float(ph["video_fraction"]),
            "pattern": str(ph["pattern"]),
            "load": 1.0,
            "slo_scale": float(args.slo_scale),
            "image_slo_scale": float(args.slo_scale),
            "resolution_distribution": copy.deepcopy(dist_uniform),
        }
        if ph["pattern"] == "gamma":
            w["cv"] = float(ph["cv"])
        probe = build_trace(
            classes, selected_pools, [w], slots=len(active_uuids), seed=args.seed, output_mode="check"
        )
        r0_rpm = float(probe["manifest"]["r0_rpm"])
        w["load"] = float(ph["rate_rpm"]) / r0_rpm
        w["requested_rate_rpm"] = float(ph["rate_rpm"])
        windows.append(w)

    trace = build_trace(
        classes, selected_pools, windows, slots=len(active_uuids), seed=args.seed, output_mode="check"
    )
    validate_trace(trace)
    if len(trace["workload"]) != 500:
        raise RuntimeError(f"Expected 500 requests in 5-phase soak trace, got {len(trace['workload'])}")

    provenance = {
        "path": "generated:soak-5phase-500req",
        "sha256": trace["manifest"]["workload_sha256"],
        "manifest": trace["manifest"],
        "five_phase_spec": FIVE_PHASE_SPEC,
    }
    phase = config["phases"][0]
    phase["matrix"]["/workload"] = [trace["workload"]]
    phase["matrix"]["/trace_provenance"] = [provenance]

    videos = [sid for sid in ids if classes[sid]["kind"] == "video"]
    images = [sid for sid in ids if classes[sid]["kind"] == "image"]
    class_counts = trace["manifest"]["class_counts"]
    vid_total = sum(class_counts.get(sid, 0) for sid in videos)
    img_total = sum(class_counts.get(sid, 0) for sid in images)
    overall_vf = vid_total / float(vid_total + img_total)
    video_mean_ms = sum(classes[sid]["gpu_service_ms"] * class_counts.get(sid, 0) for sid in videos) / vid_total
    image_mean_ms = sum(classes[sid]["service_ms"] * class_counts.get(sid, 0) for sid in images) / img_total
    image_share = fb.partition_image_gpus(
        len(active_uuids), overall_vf, {"steps": 1, "step_compute_ms": video_mean_ms}, image_mean_ms
    )
    phase["options"]["partition_image_gpus"] = image_share

    pol_slug = args.policy.replace("_", "-")
    config["experiment"] = f"formal-{args.model}-test-soak5phase-s{args.seed}-n500-slo1p5-{pol_slug}"
    config["benchmark"]["trace"] = provenance
    config["benchmark"]["scope"] = (
        "5-phase 500-request non-stationary mixed-modality sustained SLA and HBM/SHM stability soak"
    )
    config.setdefault("benchmark", {}).setdefault("formal_protocol", {}).update(
        origin_gpu_uuids=origin_uuids,
        active_gpu_uuids=active_uuids,
        calibration_match_mode=match_mode,
        num_gpus=len(active_uuids),
    )
    fb.bind_manifests(config, args.model, manifests)
    fb.apply_trace_budget(config)
    fb.write_new(args.output, config)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "experiment": config["experiment"],
                "requests": len(trace["workload"]),
                "workload_sha256": trace["manifest"]["workload_sha256"],
                "watchdog_seconds": config["budget"]["watchdog_seconds"],
                "sha256": canonical_hash(config),
            }
        )
    )


if __name__ == "__main__":
    main()
