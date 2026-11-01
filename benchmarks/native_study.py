#!/usr/bin/env python3
"""Generate native-cluster calibration and serving studies without historical measurements."""

import argparse
import json
from pathlib import Path

from cluster_config import MODELS, allocated_uuids, make_workers


CALIBRATION_POLICIES = [
    "static_sp1",
    "static_cfg2",
    "static_sp2",
    "static_cfg2sp2",
    "static_sp3",
    "static_cfg2sp3",
    "static_sp4",
    "static_cfg2sp4",
    "static_sp6",
    "static_sp8",
]
FIT_POLICIES = ["static_sp1", "static_cfg2", "static_sp2", "static_cfg2sp2", "static_sp4"]
HELD_OUT_POLICIES = [name for name in CALIBRATION_POLICIES if name not in FIT_POLICIES]


def build_calibration_config(model, gpu_uuids):
    if model == "wan21":
        frames = 33
    elif model == "wan22":
        frames = 49
    else:
        raise ValueError("unknown model: " + model)
    policies = [name for name in CALIBRATION_POLICIES if model != "wan21" or name != "static_sp8"]
    held_out = [name for name in policies if name not in FIT_POLICIES]
    steps, image_steps = 20, 28
    video_spec = {
        "steps": steps,
        "height": 480 if model == "wan21" else 512,
        "width": 832 if model == "wan21" else 896,
        "frames": frames,
        "guidance_scale": 5.0,
        "cache": False,
        "threshold": 0.0,
        "max_continuous": 1,
        "negative_prompt": "",
        "prompt": "A red ceramic kettle on a wooden table.",
        "seed": 201,
    }
    image_spec = {
        "height": 1024,
        "width": 1024,
        "steps": image_steps,
        "frames": 1,
        "batch_size": 1,
        "guidance_scale": 4.5,
        "negative_prompt": "blurry, low quality",
        "max_sequence_length": 256,
        "prompt": "A lighthouse at dusk.",
        "seed": 202,
    }
    selected, workers, slots, warmup = make_workers(
        model, gpu_uuids, video_spec=video_spec, image_spec=image_spec,
    )
    for item in warmup:
        item["spec"]["steps"] = 1
    gpus = len(gpu_uuids)
    video_row = {
        "request_id": "video-calibration",
        "kind": "video",
        "class": selected["class"],
        "arrival_ms": 0,
        "deadline_ms": 3600000,
        "service_ms": 60000,
        "spec": video_spec,
        "output_mode": "check",
    }
    image_row = {
        "request_id": "image-calibration",
        "kind": "image",
        "class": "sd35-preflight",
        "arrival_ms": 1000,
        "deadline_ms": 3600000,
        "service_ms": 10000,
        "spec": image_spec,
        "output_mode": "check",
    }
    return {
        "schema": "ditserve.study.v1",
        "experiment": f"kairos-{model}-native-calibration",
        "note": "Native per-form calibration. Synthetic fields are scheduling placeholders, not measured results.",
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
            "policy": "static_sp1",
            "profile": {
                "provenance": "synthetic",
                "sp_efficiency": 1.0,
                "resize_ms": 100.0,
                "quantum_overhead_ms": 0.0,
                "block_overhead_ms": 0.0,
                "classes": {
                    selected["class"]: {
                        "kind": "video",
                        "blocks_per_step": 60,
                        "prepare_ms": 100.0,
                        "encode_ms": 1000.0,
                        "step_compute_ms": 1000.0,
                        "comm_mb_per_step": 1000.0,
                        "collectives_per_step": 240,
                        "cfg_exchange_mb": 4.0,
                        "finish_ms": 1000.0,
                        "steps": steps,
                        "decode_chunks": (frames - 1) // 4 + 1,
                    },
                    "sd35-preflight": {"kind": "image", "service_ms": 1000.0},
                },
                "links": {str(degree): {"gbps": 3.0, "latency_ms": 0.1} for degree in (2, 3, 4, 6, 8)},
            },
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
            "workload": [video_row, image_row],
            "options": {
                "sp_transport": "export",
                "async_rank0_decode": False,
                "image_batch": 1,
                "tiered_memory": False,
            },
            "matrix": {"/policy": policies},
            "artifact_dependencies": [],
        }],
        "benchmark": {
            "native_calibration": True,
            "performance_qualified": False,
            "quality_qualified": False,
            "models": [selected["name"], "SD3.5-Medium"],
            "fit_policies": FIT_POLICIES,
            "held_out_policies": held_out,
            "shape": {"frames": frames, "steps": steps, "image_steps": image_steps},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--gpu-uuids", help="comma-separated UUIDs; defaults to nvidia-smi in this job")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    uuids = args.gpu_uuids.split(",") if args.gpu_uuids else allocated_uuids()
    config = build_calibration_config(args.model, uuids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "experiment": config["experiment"],
        "cells": len(config["phases"][0]["matrix"]["/policy"]),
        "held_out": config["benchmark"]["held_out_policies"],
    }))


if __name__ == "__main__":
    main()
