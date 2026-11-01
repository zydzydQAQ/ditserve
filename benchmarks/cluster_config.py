#!/usr/bin/env python3
"""Generate a functional native-cluster preflight on the allocated Kairos GPUs."""

import argparse
import json
import subprocess
from pathlib import Path


MODELS = {
    "wan21": {
        "name": "Wan2.1-T2V-1.3B",
        "path": "/shared/data/DITSERVE/models/wan",
        "class": "wan21-preflight",
    },
    "wan22": {
        "name": "Wan2.2-TI2V-5B",
        "path": "/shared/data/DITSERVE/models/wan2.2-ti2v-5b",
        "class": "wan22-preflight",
    },
}
SD35_PATH = "/shared/data/DITSERVE/models/sd3.5-medium"
SD35_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"


def allocated_uuids():
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [value.strip() for value in output.splitlines() if value.strip()]


def build_config(model, gpu_uuids, *, frames=17, steps=4, image_steps=4):
    if model not in MODELS:
        raise ValueError("unknown model: " + model)
    if len(gpu_uuids) not in (2, 4, 8) or len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError("preflight requires 2, 4 or 8 distinct GPU UUIDs")
    if frames != 17 or steps != 4 or image_steps != 4:
        raise ValueError("the functional preflight is fixed at Wan 17f/4 steps and SD3.5 4 steps")
    selected = MODELS[model]
    gpus = len(gpu_uuids)
    form = {2: "cfg2", 4: "cfg2sp2", 8: "cfg2sp3"}[gpus]
    workers = {}
    slots = []
    warmup = []
    for index, uuid in enumerate(gpu_uuids):
        environment = {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        video_spec = {
            "steps": steps,
            "height": 480,
            "width": 832,
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
            "height": 512,
            "width": 512,
            "steps": image_steps,
            "frames": 1,
            "batch_size": 1,
            "guidance_scale": 4.5,
            "negative_prompt": "blurry, low quality",
            "max_sequence_length": 256,
            "prompt": "A lighthouse at dusk.",
            "seed": 202,
        }
        workers[f"wan{index}"] = {
            "backend": "wan",
            "python": "/opt/venv/bin/python",
            "gpu_uuid": uuid,
            "environment": environment,
            "config": {
                "model_path": selected["path"],
                "device": "cuda:0",
                "attention": "flash",
                "vae_dtype": "bfloat16",
                "options": {
                    "execution": "block",
                    "cfg_fold": {
                        "folded_steps": [2],
                        "final_step_exact_anchor": True,
                        "force_exact_cond_on_fold": True,
                        "cfg_vel_alpha": 0.0,
                    },
                },
            },
        }
        workers[f"image{index}"] = {
            "backend": "sd35",
            "python": "/opt/venv/bin/python",
            "gpu_uuid": uuid,
            "environment": environment,
            "config": {
                "model_path": SD35_PATH,
                "device": "cuda:0",
                "revision": SD35_REVISION,
                "dtype": "bfloat16",
                "text_encoder_dtype": "bfloat16",
                "attention": "flash",
                "options": {"profile": False, "cache": False},
            },
        }
        slots.append({"video": f"wan{index}", "image": f"image{index}"})
        warmup.extend([
            {"worker": f"wan{index}", "spec": dict(video_spec, prompt="warmup", seed=7), "quantum": "step"},
            {"worker": f"image{index}", "spec": dict(image_spec, prompt="warmup", seed=7)},
        ])
    video_row = {
        "request_id": "video-preflight",
        "kind": "video",
        "class": selected["class"],
        "arrival_ms": 0,
        "deadline_ms": 600000,
        "service_ms": 60000,
        "spec": video_spec,
        "output_mode": "check",
    }
    image_rows = [
        {
            "request_id": f"image-preflight-{index}",
            "kind": "image",
            "class": "sd35-preflight",
            "arrival_ms": 1000 * (index + 1),
            "deadline_ms": 600000,
            "service_ms": 5000,
            "spec": dict(image_spec, seed=202 + index),
            "output_mode": "check",
        }
        for index in range(2)
    ]
    return {
        "schema": "ditserve.study.v1",
        "experiment": f"kairos-{model}-native-preflight",
        "note": "Functional preflight only. Synthetic planning costs are not calibrated performance data.",
        "workers": workers,
        "budget": {
            "limit_gpu_seconds": gpus * 7200,
            "watchdog_seconds": 7200,
            "repair_reserve_gpu_seconds": 0,
            "carry": "none",
        },
        "phases": [{
            "id": "cluster",
            "kind": "cluster",
            "policy": "heterodit_v5",
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
                "links": {
                    str(degree): {"gbps": 3.0, "latency_ms": 0.1}
                    for degree in (2, 3, 4, 6, 8)
                },
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
            "workload": [video_row, *image_rows],
            "options": {
                "sp_transport": "export",
                "forms": [form],
                "quantum_blocks": 4,
                "async_rank0_decode": True,
                "image_batch": 1,
                "tiered_memory": False,
            },
            "artifact_dependencies": [],
        }],
        "benchmark": {
            "functional_preflight": True,
            "performance_qualified": False,
            "quality_qualified": False,
            "models": [selected["name"], "SD3.5-Medium"],
            "hardware_source": "nvidia-smi in this job",
            "profile_role": "synthetic scheduling placeholders only",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--gpu-uuids", help="comma-separated UUIDs; defaults to nvidia-smi in this job")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    uuids = args.gpu_uuids.split(",") if args.gpu_uuids else allocated_uuids()
    config = build_config(args.model, uuids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "experiment": config["experiment"],
        "gpus": len(uuids),
        "workers": len(config["workers"]),
        "performance_qualified": False,
    }))


if __name__ == "__main__":
    main()
