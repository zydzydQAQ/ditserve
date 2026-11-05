#!/usr/bin/env python3
"""Generate a functional native-cluster preflight on the allocated Kairos GPUs."""

import argparse
import copy
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
# Worker memory strategies and SD3.5 continuous batching are explicit opt-in.
# None / 0 keep the legacy worker options (Wan {"execution": "block"}, SD3.5
# {"profile": False, "cache": False}) and atomic SD3.5 execution.
DEFAULT_MEMORY_STRATEGY = None
CONTINUOUS_BATCH_SIZES = (1, 2, 3, 4, 6, 8)
DEFAULT_CONTINUOUS_BATCH = 0

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


# Behaviour the workers have regardless of options (workers/ditserve/models):
# WanStepWorker always loads UMT5 on the CPU; AtomicPipelineBackend rejects CPU
# text encoding for SD3.5, so T5-XXL always encodes on the GPU
# (offload_text_encoders_after_warmup only parks its weights on the CPU and
# restores them for every uncached prompt); no worker reads
# weight_swap_on_switch; native configs keep the phase option tiered_memory
# False (export transport, while the C++ tiered memory requires pinned SP1).
_FIXED_WORKER_MEMORY = {
    "wan_text_encoder_device": "cpu",
    "sd35_text_encoder_device": "cuda",
    "weight_swap_on_switch": False,
    "tiered_memory": False,
}


def _unimplemented_memory_knobs(descriptor):
    missing = [f"{key}={descriptor[key]!r}" for key, value in _FIXED_WORKER_MEMORY.items()
               if descriptor[key] != value]
    if not descriptor["encoder_pool"].get("enabled", True):
        # Wan honours encoder_pool.enabled; SD35Backend always uses its pool.
        missing.append("encoder_pool.enabled=False (SD35Backend always uses its encoder pool)")
    return missing


# Strategies whose every descriptor knob the workers implement; the others are
# analytical only (benchmarks/cross_modal_memory_study.py).
WORKER_MEMORY_STRATEGIES = tuple(
    name for name in MEMORY_STRATEGIES
    if not _unimplemented_memory_knobs(_MEMORY_STRATEGY_DESCRIPTORS[name])
)


def applied_memory_strategy(strategy):
    """Return the worker options a memory strategy actually applies.

    Raises ValueError for strategies relying on knobs no worker implements, so
    a GPU config never carries a label its workers do not execute. The record
    lists only applied worker options (never tiered_memory or device claims).
    """
    descriptor = memory_strategy_descriptor(strategy)
    missing = _unimplemented_memory_knobs(descriptor)
    if missing:
        raise ValueError(
            f"memory_strategy {strategy!r} is analytical only; the workers do not implement "
            + "; ".join(missing)
            + ". Use benchmarks/cross_modal_memory_study.py for S0-S4 analysis; "
            + "worker configs accept only " + ", ".join(WORKER_MEMORY_STRATEGIES))
    pool = dict(descriptor["encoder_pool"])
    return {
        "strategy": strategy,
        "code": descriptor["code"],
        "label": descriptor["label"],
        "applied_worker_options": {
            "wan": {"release_cache_on_stop": bool(descriptor["release_cache_on_stop"]), "encoder_pool": pool},
            "image": {
                "offload_text_encoders_after_warmup": bool(descriptor["offload_text_encoders_after_warmup"]),
                "encoder_pool": dict(pool),
            },
        },
    }


def continuous_batch_settings(continuous_batch=0, sequence_packing=None, bubble_free=None):
    """Resolve an explicit SD3.5 continuous-batch request; None or 0 means off.

    sequence_packing/bubble_free are continuous-batch sub-options: None means
    on with continuous batching and off without it; an explicit True without
    continuous batching is rejected rather than reaching atomic workers.
    """
    value = 0 if continuous_batch is None else continuous_batch
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 8:
        raise ValueError("continuous_batch must be an integer in [0, 8]")
    if value == 0:
        if sequence_packing or bubble_free:
            raise ValueError("sequence_packing and bubble_free require continuous_batch > 0")
        return 0, False, False
    return (value, True if sequence_packing is None else bool(sequence_packing),
            True if bubble_free is None else bool(bubble_free))


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
    sequence_packing=None,
    bubble_free=None,
):
    """Build per-GPU Wan + SD3.5 workers.

    With no memory_strategy and continuous_batch 0 the legacy options are
    produced. sequence_packing/bubble_free are continuous-batch sub-options
    (see continuous_batch_settings) and are emitted only when continuous_batch > 0.
    """
    if model not in MODELS:
        raise ValueError("unknown model: " + model)
    if len(gpu_uuids) not in (2, 4, 6, 8) or len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError("native studies require 2, 4, 6 or 8 distinct GPU UUIDs")
    cb, packing, bubble = continuous_batch_settings(continuous_batch, sequence_packing, bubble_free)
    applied = None if memory_strategy is None else applied_memory_strategy(memory_strategy)
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
        image_options = {"profile": False, "cache": False}
        if cb > 0:
            image_options.update(continuous_batch=True, sequence_packing=packing, bubble_free=bubble)
        if applied is not None:
            options.update(copy.deepcopy(applied["applied_worker_options"]["wan"]))
            image_options.update(copy.deepcopy(applied["applied_worker_options"]["image"]))
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
    sequence_packing=None,
    bubble_free=None,
):
    if frames != 17 or steps != 4 or image_steps != 4:
        raise ValueError("the functional preflight is fixed at Wan 17f/4 steps and SD3.5 4 steps")
    if len(gpu_uuids) not in (2, 4, 6, 8) or len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError("preflight requires 2, 4, 6 or 8 distinct GPU UUIDs")
    eff_cb, packing, bubble = continuous_batch_settings(continuous_batch, sequence_packing, bubble_free)
    applied = None if memory_strategy is None else applied_memory_strategy(memory_strategy)
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
        memory_strategy=memory_strategy,
        continuous_batch=eff_cb,
        sequence_packing=packing,
        bubble_free=bubble,
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
        "tiered_memory": False,
    }
    benchmark_meta = {
        "functional_preflight": True,
        "performance_qualified": False,
        "quality_qualified": False,
        "models": [selected["name"], "SD3.5-Medium"],
        "hardware_source": "nvidia-smi in this job",
        "profile_role": "synthetic scheduling placeholders only",
    }
    image_cost = {"kind": "image", "service_ms": 1000.0}
    if eff_cb > 0:
        phase_options.update(continuous_batch=eff_cb, sequence_packing=packing, bubble_free=bubble)
        benchmark_meta["continuous_batch"] = {
            "max_requests": eff_cb, "sequence_packing": packing, "bubble_free": bubble,
        }
        # Synthetic like the rest of this functional-preflight profile.
        image_cost.update(steps=image_steps, prepare_ms=20.0, finish_ms=80.0,
                          batch_step_ms=default_sd35_batch_step_ms(1000.0, image_steps, resolution=512))
    if applied is not None:
        benchmark_meta["memory_strategy"] = applied
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
                    "sd35-preflight": image_cost,
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
        help="opt-in maximum continuous step-level image batch size in [0, 8]; default 0 (off)",
    )
    parser.add_argument(
        "--sequence-packing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="token sequence packing for SD3.5 continuous batches (default: on iff --continuous-batch > 0)",
    )
    parser.add_argument(
        "--bubble-free",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="overlap T5 encode and VAE decode with continuous MM-DiT rounds (default: on iff --continuous-batch > 0)",
    )
    parser.add_argument(
        "--memory-strategy",
        choices=MEMORY_STRATEGIES,
        help="opt-in worker memory strategy; only strategies the workers implement are accepted "
             "(default: none, legacy worker options)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    uuids = args.gpu_uuids.split(",") if args.gpu_uuids else allocated_uuids()
    try:
        config = build_config(
            args.model,
            uuids,
            memory_strategy=args.memory_strategy,
            continuous_batch=args.continuous_batch,
            sequence_packing=args.sequence_packing,
            bubble_free=args.bubble_free,
        )
    except ValueError as error:
        parser.error(str(error))
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
