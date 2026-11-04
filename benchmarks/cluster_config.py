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

MEMORY_STRATEGIES = (
    "s0_naive_gpu",
    "s1_weight_swap",
    "s2_wan_cpu_sd35_gpu",
    "s3_both_encoders_cpu",
    "s4_disaggregated_pool",
)
DEFAULT_MEMORY_STRATEGY = "s4_disaggregated_pool"
CONTINUOUS_BATCH_SIZES = (1, 2, 3, 4, 6, 8)
DEFAULT_CONTINUOUS_BATCH = 4

_MEMORY_STRATEGY_DESCRIPTORS = {
    "s0_naive_gpu": {
        "strategy": "s0_naive_gpu",
        "code": "S0",
        "label": "Naive GPU Co-Location (All Components on GPU)",
        "wan_text_encoder_device": "cuda",
        "sd35_text_encoder_device": "cuda",
        "offload_text_encoders_after_warmup": False,
        "weight_swap_on_switch": False,
        "release_cache_on_stop": False,
        "tiered_memory": False,
        "encoder_pool": {
            "enabled": False,
            "write_shm": False,
            "pin_memory": False,
            "shm_dir": "/dev/shm/ditserve-hetero-embeds",
        },
    },
    "s1_weight_swap": {
        "strategy": "s1_weight_swap",
        "code": "S1",
        "label": "Reactive On-Demand CPU-GPU Weight Swapping",
        "wan_text_encoder_device": "cpu",
        "sd35_text_encoder_device": "cpu",
        "offload_text_encoders_after_warmup": True,
        "weight_swap_on_switch": True,
        "release_cache_on_stop": False,
        "tiered_memory": False,
        "encoder_pool": {
            "enabled": False,
            "write_shm": False,
            "pin_memory": True,
            "shm_dir": "/dev/shm/ditserve-hetero-embeds",
        },
    },
    "s2_wan_cpu_sd35_gpu": {
        "strategy": "s2_wan_cpu_sd35_gpu",
        "code": "S2",
        "label": "Wan UMT5 on CPU + SD3.5 T5-XXL on GPU",
        "wan_text_encoder_device": "cpu",
        "sd35_text_encoder_device": "cuda",
        "offload_text_encoders_after_warmup": False,
        "weight_swap_on_switch": False,
        "release_cache_on_stop": False,
        "tiered_memory": False,
        "encoder_pool": {
            "enabled": True,
            "write_shm": False,
            "pin_memory": True,
            "shm_dir": "/dev/shm/ditserve-hetero-embeds",
        },
    },
    "s3_both_encoders_cpu": {
        "strategy": "s3_both_encoders_cpu",
        "code": "S3",
        "label": "Both Heavy T5 Encoders on CPU (No Cross-Worker Pool)",
        "wan_text_encoder_device": "cpu",
        "sd35_text_encoder_device": "cpu",
        "offload_text_encoders_after_warmup": True,
        "weight_swap_on_switch": False,
        "release_cache_on_stop": False,
        "tiered_memory": False,
        "encoder_pool": {
            "enabled": True,
            "write_shm": False,
            "pin_memory": True,
            "shm_dir": "/dev/shm/ditserve-hetero-embeds",
        },
    },
    "s4_disaggregated_pool": {
        "strategy": "s4_disaggregated_pool",
        "code": "S4",
        "label": "DitServe Cross-Modal Disaggregated Encoder Pool + Safe-Point Cache Release",
        "wan_text_encoder_device": "cpu",
        "sd35_text_encoder_device": "cpu",
        "offload_text_encoders_after_warmup": True,
        "weight_swap_on_switch": False,
        "release_cache_on_stop": True,
        "tiered_memory": True,
        "encoder_pool": {
            "enabled": True,
            "write_shm": True,
            "pin_memory": True,
            "shm_dir": "/dev/shm/ditserve-hetero-embeds",
            "max_entries": 64,
            "max_bytes": 512 * (1 << 20),
        },
    },
}


def memory_strategy_Descriptor(strategy: str) -> dict:
    if strategy not in _MEMORY_STRATEGY_DESCRIPTORS:
        raise ValueError("unknown memory_strategy: " + str(strategy))
    base = dict(_MEMORY_STRATEGY_DESCRIPTORS[strategy])
    base["encoder_pool"] = dict(base["encoder_pool"])
    return base


memory_strategy_descriptor = memory_strategy_Descriptor


def allocated_uuids():
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [value.strip() for value in output.splitlines() if value.strip()]


def default_sd35_batch_step_ms(service_ms: float = 1000.0, steps: int = 28, *, resolution: int = 1024) -> dict:
    """Calibrated SD3.5-Medium MM-DiT 1-step forward scaling table on A100-PCIE-40GB (BF16 FlashAttention).

    For 512x512 (256 patch tokens per branch), B=1..8 scales sub-linearly because B=1 under-utilizes SMs.
    For 1024x1024 (1024 patch tokens per branch), B=1..8 still amortizes weight reads, kernel launch, and CFG overhead.
    """
    steps = max(1, int(steps))
    decode_and_prep = min(service_ms * 0.12, 180.0)
    step1_ms = max(4.0, (float(service_ms) - decode_and_prep) / steps)
    if int(resolution) <= 512:
        # 512x512 sub-linear scaling ratios on A100-PCIE-40GB (d20261104-sd35-cb-sweep-r3)
        ratios = {1: 1.00, 2: 1.838, 3: 2.669, 4: 3.567, 6: 5.194, 8: 6.941}
    else:
        # 1024x1024 sub-linear scaling ratios on A100-PCIE-40GB (d20261104-sd35-cb-sweep-r3)
        ratios = {1: 1.00, 2: 1.940, 3: 2.834, 4: 3.788, 6: 5.636, 8: 7.514}
    return {str(k): round(step1_ms * mult, 3) for k, mult in ratios.items()}


def make_workers(
    model,
    gpu_uuids,
    *,
    video_spec,
    image_spec,
    cfg_fold=None,
    memory_strategy=DEFAULT_MEMORY_STRATEGY,
    continuous_batch=DEFAULT_CONTINUOUS_BATCH,
    sequence_packing=True,
    bubble_free=True,
):
    if model not in MODELS:
        raise ValueError("unknown model: " + model)
    if len(gpu_uuids) not in (2, 4, 6, 8) or len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError("native studies require 2, 4, 6 or 8 distinct GPU UUIDs")
    eff_strategy = DEFAULT_MEMORY_STRATEGY if memory_strategy is None else memory_strategy
    strategy_desc = memory_strategy_Descriptor(eff_strategy)
    cb_enabled = bool(continuous_batch and int(continuous_batch) > 0)
    selected = MODELS[model]
    workers, slots, warmup = {}, [], []
    for index, uuid in enumerate(gpu_uuids):
        environment = {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        options = {"execution": "block"}
        if cfg_fold is not None:
            options["cfg_fold"] = cfg_fold
        image_options = {
            "profile": False,
            "cache": False,
            "continuous_batch": cb_enabled,
            "sequence_packing": bool(sequence_packing),
            "bubble_free": bool(bubble_free),
        }
        if strategy_desc is not None:
            options["memory_strategy"] = eff_strategy
            options["release_cache_on_stop"] = bool(strategy_desc["release_cache_on_stop"])
            options["encoder_pool"] = dict(strategy_desc["encoder_pool"])
            image_options["memory_strategy"] = eff_strategy
            image_options["offload_text_encoders_after_warmup"] = bool(
                strategy_desc["offload_text_encoders_after_warmup"]
            )
            image_options["encoder_pool"] = dict(strategy_desc["encoder_pool"])
            if strategy_desc.get("weight_swap_on_switch"):
                image_options["weight_swap_on_switch"] = True
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
                "text_encoder_dtype": "bfloat16",
                "options": options,
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
                "options": image_options,
            },
        }
        slots.append({"video": f"wan{index}", "image": f"image{index}"})
        warmup.extend([
            {"worker": f"wan{index}", "spec": dict(video_spec, prompt="warmup", seed=7), "quantum": "step"},
            {"worker": f"image{index}", "spec": dict(image_spec, prompt="warmup", seed=7)},
        ])
    return selected, workers, slots, warmup


def build_config(
    model,
    gpu_uuids,
    *,
    frames=17,
    steps=4,
    image_steps=4,
    memory_strategy=DEFAULT_MEMORY_STRATEGY,
    continuous_batch=DEFAULT_CONTINUOUS_BATCH,
    sequence_packing=True,
    bubble_free=True,
):
    if frames != 17 or steps != 4 or image_steps != 4:
        raise ValueError("the functional preflight is fixed at Wan 17f/4 steps and SD3.5 4 steps")
    if len(gpu_uuids) not in (2, 4, 6, 8) or len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError("preflight requires 2, 4, 6 or 8 distinct GPU UUIDs")
    if continuous_batch is not None and (
        not isinstance(continuous_batch, int) or isinstance(continuous_batch, bool) or not 0 <= continuous_batch <= 8
    ):
        raise ValueError("continuous_batch must be an integer in [0, 8]")
    eff_cb = DEFAULT_CONTINUOUS_BATCH if continuous_batch is None else int(continuous_batch)
    eff_strategy = DEFAULT_MEMORY_STRATEGY if memory_strategy is None else memory_strategy
    strategy_desc = memory_strategy_Descriptor(eff_strategy)
    gpus = len(gpu_uuids)
    form = {2: "cfg2", 4: "cfg2sp2", 6: "cfg2sp3", 8: "cfg2sp3"}[gpus]
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
    cfg_fold = {
        "folded_steps": [2],
        "final_step_exact_anchor": True,
        "force_exact_cond_on_fold": True,
        "cfg_vel_alpha": 0.0,
    }
    selected, workers, slots, warmup = make_workers(
        model,
        gpu_uuids,
        video_spec=video_spec,
        image_spec=image_spec,
        cfg_fold=cfg_fold,
        memory_strategy=eff_strategy,
        continuous_batch=eff_cb,
        sequence_packing=sequence_packing,
        bubble_free=bubble_free,
    )
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
    phase_options = {
        "sp_transport": "export",
        "forms": [form],
        "quantum_blocks": 4,
        "async_rank0_decode": True,
        "image_batch": 1,
        "continuous_batch": eff_cb,
        "sequence_packing": bool(sequence_packing),
        "bubble_free": bool(bubble_free),
        "tiered_memory": False,
        "memory_strategy": eff_strategy,
    }
    benchmark_meta = {
        "functional_preflight": True,
        "performance_qualified": False,
        "quality_qualified": False,
        "models": [selected["name"], "SD3.5-Medium"],
        "hardware_source": "nvidia-smi in this job",
        "profile_role": "synthetic scheduling placeholders only",
        "memory_strategy": strategy_desc,
        "continuous_batch": {
            "max_requests": eff_cb,
            "sequence_packing": bool(sequence_packing),
            "bubble_free": bool(bubble_free),
        },
    }
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
                    "sd35-preflight": {
                        "kind": "image",
                        "service_ms": 1000.0,
                        "steps": image_steps,
                        "prepare_ms": 20.0,
                        "finish_ms": 80.0,
                        "batch_step_ms": default_sd35_batch_step_ms(1000.0, image_steps, resolution=512),
                    },
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
            "options": phase_options,
            "artifact_dependencies": [],
        }],
        "benchmark": benchmark_meta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--gpu-uuids", help="comma-separated UUIDs; defaults to nvidia-smi in this job")
    parser.add_argument(
        "--continuous-batch",
        type=int,
        default=DEFAULT_CONTINUOUS_BATCH,
        help="maximum continuous step-level image batch size in [0, 8] (0 disables continuous batching)",
    )
    parser.add_argument(
        "--sequence-packing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable multi-resolution token sequence packing for SD3.5 continuous batches",
    )
    parser.add_argument(
        "--bubble-free",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="overlap T5 encode and VAE decode with MM-DiT denoising rounds",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    uuids = args.gpu_uuids.split(",") if args.gpu_uuids else allocated_uuids()
    config = build_config(
        args.model,
        uuids,
        continuous_batch=args.continuous_batch,
        sequence_packing=args.sequence_packing,
        bubble_free=args.bubble_free,
    )
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
