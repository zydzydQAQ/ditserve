#!/usr/bin/env python3
"""Build a 121-frame Wan2.2-720p (27.3K tokens > L*_4 = 25.0K) + SD3.5-1024
mixed-modality online CFG2 -> CFG2xSP elasticity study config on 8x A100-PCIE-40GB."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks-dir", type=pathlib.Path, required=True)
    ap.add_argument("--protocol", type=pathlib.Path, required=True)
    ap.add_argument("--bundle-121f", type=pathlib.Path, required=True)
    ap.add_argument("--bundle-81f", type=pathlib.Path, required=True)
    ap.add_argument("--datasets-dir", type=pathlib.Path, required=True)
    ap.add_argument("--manifests", type=pathlib.Path, required=True)
    ap.add_argument("--policies", default="ditserve,block-cfg2,sbp-esp-cfg2")
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--rate-rpm", type=float, default=1.5)
    ap.add_argument("--video-slo-scale", type=float, default=0.50)
    ap.add_argument("--image-slo-scale", type=float, default=1.50)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.benchmarks_dir))
    import formal_benchmark as fb
    from native_serving import build_serving_config
    from ditserve.evaluation.common import canonical_hash
    from ditserve.evaluation.workloads import build_trace

    p = fb.protocol(args.protocol)
    b121 = fb.read_json(args.bundle_121f)
    b81 = fb.read_json(args.bundle_81f)
    active_uuids = list(b81.get("active_gpu_uuids") or fb.allocated_uuids())[:8]

    v_item = copy.deepcopy(b121["classes"]["wan22-720p"])
    i_item = copy.deepcopy(b81["classes"]["sd35-1024"])
    v_spec = copy.deepcopy(v_item["spec"])
    v_cost = copy.deepcopy(v_item["cost"])
    i_cost = copy.deepcopy(i_item["cost"])

    # ProactiveEncoderPool prefetched UMT5 encode latency on 720da72 wan22-720p (~15.3 ms)
    v_cost["encode_ms"] = float(b81["classes"]["wan22-720p"]["cost"]["encode_ms"])

    resize_ms = max(
        float(v_item.get("source", {}).get("resize_ms", 15.0)),
        float(i_item.get("source", {}).get("resize_ms", 15.0)),
    )
    quantum_overhead_ms = max(
        float(v_item.get("source", {}).get("quantum_overhead_ms", 1.5)),
        float(i_item.get("source", {}).get("quantum_overhead_ms", 1.5)),
    )

    profile = {
        "provenance": "measured",
        "classes": {
            "wan22-preflight": v_cost,
            "sd35-preflight": i_cost,
        },
        "links": copy.deepcopy(v_cost["links"]),
        "resize_ms": resize_ms,
        "quantum_overhead_ms": quantum_overhead_ms,
        "block_overhead_ms": 0.0,
        "source": {
            "gpu_uuids": active_uuids,
            "origin_121f_bundle_sha256": b121.get("sha256", ""),
            "origin_81f_bundle_sha256": b81.get("sha256", ""),
            "shape": v_spec,
        },
    }

    pools = {
        kind: fb.read_json(args.datasets_dir / (p["datasets"][kind]["kind"] + ".json"))
        for kind in ("image", "video")
    }
    selected_pools = fb.split_pools(pools, p, "test")
    manifests = fb.read_json(args.manifests)

    gpu_service_ms = float(
        v_cost.get("prepare_ms", 0.0) + v_cost["steps"] * v_cost["step_compute_ms"] + v_cost.get("finish_ms", 0.0)
    )
    service_ms = float(v_cost.get("encode_ms", 0.0)) + gpu_service_ms
    probe_classes = {
        "wan22-preflight": dict(
            v_cost,
            **{
                "class": "wan22-preflight",
                "kind": "video",
                "service_ms": service_ms,
                "gpu_service_ms": gpu_service_ms,
                "slo_scale": args.video_slo_scale,
                "spec": v_spec,
            },
        ),
        "sd35-preflight": {
            "class": "sd35-preflight",
            "kind": "image",
            "service_ms": float(i_cost["service_ms"]),
            "gpu_service_ms": float(i_cost["service_ms"]),
            "slo_scale": args.image_slo_scale,
            "spec": copy.deepcopy(i_item["spec"]),
        },
    }
    probe = build_trace(
        probe_classes,
        selected_pools,
        [{"count": args.count, "video_fraction": 0.5, "pattern": "poisson", "load": 1.0, "resolution_distribution": {"name": "uniform", "alpha": 0.0}}],
        slots=len(active_uuids),
        seed=args.seed,
        output_mode="check",
    )
    r0_rpm = float(probe["manifest"]["r0_rpm"])
    resolved_load = float(args.rate_rpm) / r0_rpm

    pol_list = [x.strip() for x in args.policies.split(",") if x.strip()]
    config = build_serving_config(
        "wan22",
        active_uuids,
        profile,
        selected_pools,
        seed=args.seed,
        load=resolved_load,
        pattern="poisson",
        video_fraction=0.5,
        count=args.count,
        policies=pol_list,
        video_slo_scale=args.video_slo_scale,
        image_slo_scale=args.image_slo_scale,
        doorbell=True,
        output_mode="check",
        require_profile_match=True,
        policy_order="forward",
    )
    config["experiment"] = (
        f"formal-wan22-121f-720p-elastic-s{args.seed}-n{args.count}"
        f"-r{str(args.rate_rpm).replace('.', 'p')}rpm-vslo{str(args.video_slo_scale).replace('.', 'p')}"
    )
    config["benchmark"]["scope"] = (
        "121-frame Wan2.2-720p (27,280 tokens > L*_4 = 25,000) + SD3.5-1024 online CFG2->CFG2xSP elasticity study"
    )
    config["benchmark"]["requested_rate_rpm"] = float(args.rate_rpm)
    config["benchmark"]["r0_rpm"] = r0_rpm
    fb.bind_manifests(config, "wan22", manifests)
    fb.apply_trace_budget(config)
    fb.write_new(args.output, config)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "experiment": config["experiment"],
                "requests": args.count,
                "r0_rpm": r0_rpm,
                "resolved_load": resolved_load,
                "policies": pol_list,
                "sha256": canonical_hash(config),
            }
        )
    )


if __name__ == "__main__":
    main()
