#!/usr/bin/env python3
"""Official xDiT 1..8 GPU steady-step scaling across SP and CFG2xSP forms (Wan2.1, Wan2.2, SD3.5).

Extends official xDiT diagnostics without modifying `external_xdit_distributed.py`
or any pinned upstream xDiT file:
- Supports `world_size in (1, 2, 4, 8)` with `--cfg-degree {1,2}` and
  `--ulysses-degree {1,2,3,4,6,8}` (`cfg_degree * ulysses_degree == world_size`).
- Supports both video models (`wan21`, `wan22`) and the image model (`sd35` at
  `sd35-512` and `sd35-1024`) to measure why 1-GPU atomic Batch-2 execution
  outperforms multi-GPU SP/CFG for images on PCIe.
- Installs a non-invasive CUDA-Event step timer around `pipeline.transformer.forward`
  to extract the exact 49-step (Wan) or 27-step (SD3.5) post-warmup steady-state
  median step latency (`median_steady_step_ms`), matching DitServe's calibration scope.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

import external_xdit as source
import external_xdit_distributed as dist_base
import external_xdit_native as native
import pcie_collective_characterization as pcie


SCHEMA = "ditserve.external-xdit-scaling.v1"
ALLOWED_WORLDS = (1, 2, 4, 8)
HEAD_COUNTS = {"wan21": 12, "wan22": 24, "sd35": 24}
ALL_CLASSES = {
    **native.CLASSES,
    "sd35": "xFuserStableDiffusionModel",
}
ALL_REVISIONS = {
    **native.REVISIONS,
    "sd35": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
}


def scaling_topology(world: int, environ: dict[str, str]) -> int:
    """Validate single-node torchrun topology for 1, 2, 4, or 8 GPUs."""
    if type(world) is not int or world not in ALLOWED_WORLDS:
        raise ValueError(f"world_size must be one of {ALLOWED_WORLDS}, got {world!r}")
    if world == 1 and "RANK" not in environ and "WORLD_SIZE" not in environ:
        return 0
    values = {}
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        val = environ.get(key, "")
        if not isinstance(val, str) or not val.isdecimal():
            raise ValueError("missing or invalid torchrun topology: " + key)
        values[key] = int(val)
    rank = values["RANK"]
    if (
        values["WORLD_SIZE"] != world
        or values["LOCAL_WORLD_SIZE"] != world
        or not 0 <= rank < world
        or values["LOCAL_RANK"] != rank
    ):
        raise ValueError("requires one node with unique rank/local-rank in the declared world")
    return rank


def validate_parallel_degrees(model: str, world: int, cfg_degree: int, ulysses_degree: int) -> str:
    """Validate (world, cfg_degree, ulysses_degree) and head divisibility, returning canonical form name."""
    if model not in HEAD_COUNTS:
        raise ValueError(f"unsupported model: {model}")
    if type(world) is not int or world not in ALLOWED_WORLDS:
        raise ValueError(f"unsupported world_size: {world}")
    if type(cfg_degree) is not int or cfg_degree not in (1, 2):
        raise ValueError(f"cfg_degree must be 1 or 2, got {cfg_degree!r}")
    if type(ulysses_degree) is not int or ulysses_degree not in (1, 2, 3, 4, 6, 8):
        raise ValueError(f"unsupported ulysses_degree: {ulysses_degree!r}")
    if cfg_degree * ulysses_degree != world:
        raise ValueError(
            f"cfg_degree ({cfg_degree}) * ulysses_degree ({ulysses_degree}) != world_size ({world})"
        )
    heads = HEAD_COUNTS[model]
    if heads % ulysses_degree != 0:
        raise ValueError(
            f"model {model} has {heads} attention heads, which is not divisible by ulysses_degree={ulysses_degree}"
        )
    return pcie.form_name(cfg_degree, ulysses_degree)


def request_plan_any_modality(
    protocol: dict[str, Any],
    model: str,
    shape_id: str,
    mode: str,
    prompt: str,
    seed: int,
) -> dict[str, Any]:
    """Build a request plan for either video (Wan2.1/Wan2.2) or image (SD3.5-Medium)."""
    if protocol.get("schema") != "ditserve.formal-protocol.v1" or protocol.get("id") != "formal-mixed-v1":
        raise ValueError("formal protocol identity differs")
    if model not in ALL_CLASSES or mode not in ("quick", "formal"):
        raise ValueError("unsupported model or execution mode")
    if shape_id not in protocol["shapes"]:
        raise ValueError(f"unknown shape_id: {shape_id}")
    shape = protocol["shapes"][shape_id]
    if shape["model"] != model:
        raise ValueError(f"shape {shape_id} does not belong to model {model}")
    if mode == "formal" and shape["scope"] != "formal":
        raise ValueError("formal mode requires a formal protocol shape")
    if not isinstance(prompt, str) or not prompt.strip() or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("nonempty prompt and integer seed in [0, 2^63) required")
    return {
        "mode": mode,
        "kind": shape["kind"],
        "shape_id": shape_id,
        "shape_scope": shape["scope"],
        "height": shape["height"],
        "width": shape["width"],
        "num_frames": shape["frames"],
        "num_inference_steps": 1 if mode == "quick" else shape["steps"],
        "protocol_steps": shape["steps"],
        "guidance_scale": shape["guidance_scale"],
        "negative_prompt": "blurry, low quality" if shape["kind"] == "image" else "",
        "prompt": prompt,
        "seed": seed,
        "changed_protocol_parameters": (
            {"steps": {"protocol": shape["steps"], "executed": 1}} if mode == "quick" else {}
        ),
        "num_outputs": 1,
    }


def scaling_runner_config(
    model: str,
    request: dict[str, Any],
    identity: dict[str, Any],
    output_dir: Path,
    world: int,
    cfg_degree: int,
    ulysses_degree: int,
    cpu_offload: bool = False,
) -> dict[str, Any]:
    """Construct official xDiT runner configuration for any legal (cfg_degree, ulysses_degree) form."""
    form = validate_parallel_degrees(model, world, cfg_degree, ulysses_degree)
    config = {
        k: request[k]
        for k in (
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "negative_prompt",
            "prompt",
            "seed",
        )
    }
    config.update(
        model=source.MODEL_IDS[model],
        task="t2v" if model == "wan22" else None,
        input_images=[],
        flow_shift=identity.get("flow_shift"),
        output_directory=str(output_dir),
        ulysses_degree=ulysses_degree,
        ring_degree=1,
        data_parallel_degree=1,
        tensor_parallel_degree=1,
        pipefusion_parallel_degree=1,
        fully_shard_degree=1,
        use_cfg_parallel=(cfg_degree == 2),
        use_torch_compile=False,
        cache_method=None,
        use_teacache=False,
        use_fbcache=False,
        use_fp8_gemms=False,
        use_fp8_comms=False,
        enable_model_cpu_offload=bool(cpu_offload),
        enable_tiling=False,
        enable_slicing=False,
        num_iterations=1,
        warmup_calls=0,
        batch_size=1 if model == "sd35" else 0,
        determinism_check=0,
        use_ray=False,
        parallel_form=form,
    )
    return config


def expected_forward_calls_per_step(model: str, cfg_degree: int, guidance_scale: float) -> int:
    """Return how many transformer.forward calls occur per denoising solver step."""
    if guidance_scale <= 1.0:
        return 1
    if model in ("wan21", "wan22"):
        # WanPipeline evaluates conditional and unconditional in 2 separate transformer calls
        # when CFG=1, and 1 transformer call per rank when CFG=2 (use_cfg_parallel=True).
        return 1 if cfg_degree == 2 else 2
    # SD3.5 concatenates [uncond, cond] along batch=2 when CFG=1, and splits 1 per rank when CFG=2.
    return 1


def summarize_step_timings(
    forward_durations_ms: list[float],
    expected_steps: int,
    calls_per_step: int = 1,
) -> dict[str, Any]:
    """Aggregate per-forward durations into solver-step latencies and post-warmup steady statistics."""
    if calls_per_step < 1:
        raise ValueError("calls_per_step must be >= 1")
    if len(forward_durations_ms) != expected_steps * calls_per_step:
        raise ValueError(
            f"expected {expected_steps * calls_per_step} transformer forwards "
            f"({expected_steps} steps x {calls_per_step}), observed {len(forward_durations_ms)}"
        )
    step_ms = [
        round(math.fsum(forward_durations_ms[i * calls_per_step : (i + 1) * calls_per_step]), 4)
        for i in range(expected_steps)
    ]
    steady = step_ms[1:] if len(step_ms) > 1 else step_ms
    sorted_steady = sorted(steady)
    p95_idx = min(len(sorted_steady) - 1, max(0, int(math.ceil(0.95 * len(sorted_steady))) - 1))
    return {
        "total_steps": expected_steps,
        "calls_per_step": calls_per_step,
        "warmup_step0_ms": step_ms[0],
        "steady_sample_count": len(steady),
        "median_steady_step_ms": round(statistics.median(steady), 4),
        "mean_steady_step_ms": round(math.fsum(steady) / len(steady), 4),
        "p95_steady_step_ms": sorted_steady[p95_idx],
        "min_steady_step_ms": min(steady),
        "max_steady_step_ms": max(steady),
        "all_steps_ms": step_ms,
        "timing_scope": "median_steady_advance" if len(step_ms) > 1 else "single_step_quick",
    }


@contextmanager
def cuda_step_timing_hook(transformer: Any, torch_mod: Any):
    """Non-invasively record CUDA Event durations for every `transformer.forward` invocation."""
    events: list[tuple[Any, Any]] = []
    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def timed_forward(*args, **kwargs):
        start_ev = torch_mod.cuda.Event(enable_timing=True)
        end_ev = torch_mod.cuda.Event(enable_timing=True)
        start_ev.record()
        out = original_forward(*args, **kwargs)
        end_ev.record()
        events.append((start_ev, end_ev))
        return out

    transformer.forward = timed_forward
    durations_ms: list[float] = []
    try:
        yield durations_ms
    finally:
        transformer.forward = original_forward
        torch_mod.cuda.synchronize()
        for start_ev, end_ev in events:
            durations_ms.append(float(start_ev.elapsed_time(end_ev)))


def compare_xdit_and_ditserve_scaling(
    xdit_records: list[dict[str, Any]],
    ditserve_decomposition: dict[str, Any],
) -> dict[str, Any]:
    """Join official xDiT steady-step measurements with DitServe's 8-GPU calibration decomposition."""
    by_shape: dict[str, dict[str, Any]] = {}
    for rec in xdit_records:
        req = rec.get("request", {})
        sid = req.get("shape_id")
        form = rec.get("parallel_form") or rec.get("runner_config", {}).get("parallel_form")
        st = rec.get("step_timings", {})
        if not sid or not form or "median_steady_step_ms" not in st:
            continue
        by_shape.setdefault(sid, {})[form] = {
            "xdit_median_steady_step_ms": st["median_steady_step_ms"],
            "xdit_p95_steady_step_ms": st.get("p95_steady_step_ms"),
            "xdit_steady_samples": st.get("steady_sample_count"),
            "world_size": rec.get("world_size", 1),
            "cfg_degree": 2 if rec.get("runner_config", {}).get("use_cfg_parallel") else 1,
            "ulysses_degree": rec.get("runner_config", {}).get("ulysses_degree", 1),
        }

    comparison: dict[str, Any] = {}
    video_shapes = ditserve_decomposition.get("video_shapes", {})
    for sid, xforms in sorted(by_shape.items()):
        x_sp1 = xforms.get("sp1", {}).get("xdit_median_steady_step_ms")
        dit_forms = {
            r["form"]: r
            for r in video_shapes.get(sid, {}).get("forms", [])
        }
        rows = []
        for fname, xrow in sorted(xforms.items()):
            x_ms = xrow["xdit_median_steady_step_ms"]
            entry = {
                "form": fname,
                "gpus": xrow["world_size"],
                "xdit_steady_step_ms": x_ms,
                "xdit_speedup_vs_sp1": round(x_sp1 / x_ms, 4) if x_sp1 and x_ms > 0 else None,
            }
            if fname in dit_forms:
                d_ms = dit_forms[fname]["measured_step_ms"]
                entry["ditserve_steady_step_ms"] = d_ms
                entry["ditserve_speedup_vs_sp1"] = dit_forms[fname]["speedup_vs_sp1"]
                entry["delta_pct_vs_xdit"] = round((d_ms - x_ms) / x_ms * 100.0, 2)
            rows.append(entry)
        comparison[sid] = {"shape_id": sid, "forms": rows}

    return {
        "schema": SCHEMA,
        "scope": "Steady-state per-step scaling comparison between official xDiT and DitServe",
        "shapes": comparison,
    }


def verify_any_model(
    model: str,
    model_root: Path,
    bindings_path: Path,
    bindings_sha256: str,
) -> dict[str, Any]:
    """Verify either Wan2.1/Wan2.2 or SD3.5-Medium checkpoint against frozen manifest."""
    if model in native.CLASSES:
        return native.verify_model(model, model_root, bindings_path, bindings_sha256)
    if model != "sd35":
        raise ValueError(f"unsupported model: {model}")
    binding = native.bound_json(bindings_path, bindings_sha256)["sd35"]
    root = Path(model_root).resolve(strict=True)
    if binding.get("model_path") and root != Path(binding["model_path"]).resolve(strict=True):
        raise ValueError("SD3.5 checkpoint root differs from frozen model binding")
    manifest_path = Path(binding["path"])
    manifest = native.bound_json(manifest_path, binding["sha256"])
    if manifest.get("schema") != "ditserve.model-manifest.v1":
        raise ValueError("unexpected SD3.5 manifest schema")
    return {
        "model_id": source.MODEL_IDS["sd35"],
        "root": str(root),
        "binding_sha256": bindings_sha256,
        "manifest_sha256": binding["sha256"],
        "manifest_path": str(manifest_path),
        "revision": manifest.get("revision", ALL_REVISIONS["sd35"]),
        "verified_files": len(manifest.get("files", [])),
    }


def run_scaling(args: argparse.Namespace) -> int:
    """Execute a 1..8 GPU official xDiT SP or CFG2xSP scaling run with CUDA Event step timing."""
    import contextlib
    import sys
    import threading
    import traceback

    rank = scaling_topology(args.world_size, os.environ)
    output_dir = (args.output_dir or args.output).resolve()
    if not output_dir.is_relative_to(Path("/shared/output/DITSERVE")):
        raise ValueError("GPU evidence must be below /shared/output/DITSERVE")
    rank_dir = output_dir / f"rank-{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout_seconds
    stop = threading.Event()

    def watch():
        while not stop.wait(5):
            try:
                native.check_storage()
                if time.monotonic() >= deadline:
                    raise TimeoutError("xDiT scaling rank deadline exceeded")
            except Exception as exc:
                print("xDiT scaling watchdog stopped execution: " + repr(exc), file=sys.stderr, flush=True)
                os._exit(75)

    threading.Thread(target=watch, daemon=True).start()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "native_scaling_failed",
        "rank": rank,
        "world_size": args.world_size,
        "cfg_degree": args.cfg_degree,
        "ulysses_degree": args.ulysses_degree,
        "parallel_form": pcie.form_name(args.cfg_degree, args.ulysses_degree),
        "planned_returncode": 1,
        "qualification_eligible": False,
        "allows": [],
    }
    try:
        result["initial_storage"] = native.check_storage()
        manifest = source.verify_source(args.source_root, args.source_manifest, args.source_manifest_sha256)
        protocol = native.bound_json(args.protocol, args.protocol_sha256)
        request = request_plan_any_modality(protocol, args.model, args.shape, args.mode, args.prompt, args.seed)
        identity = verify_any_model(args.model, args.model_root, args.models, args.models_sha256)
        config = scaling_runner_config(
            args.model, request, identity, rank_dir, args.world_size, args.cfg_degree, args.ulysses_degree
        )
        result.update(request=request, model_identity=identity, runner_config=config)

        sys.path.insert(0, str(Path(args.source_root).resolve()))
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        import torch

        torch.cuda.set_device(rank)
        from xfuser.runner import xFuserModelRunner
        from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

        if args.model == "sd35":
            from xfuser.model_executor.models.runner_models.stable_diffusion import xFuserStableDiffusionModel

            MODEL_REGISTRY[source.MODEL_IDS["sd35"]] = xFuserStableDiffusionModel

        class ScalingCheckpointRunner(xFuserModelRunner):
            def _select_model(self, model_name, upstream_config):
                instance = super()._select_model(model_name, upstream_config)
                if args.model == "sd35":
                    instance.attention_heads = 24
                instance.settings.model_name = identity["root"]
                return instance

        runner = None
        initialized = False
        try:
            with contextlib.redirect_stdout(sys.stderr):
                t0 = time.monotonic()
                runner = ScalingCheckpointRunner(config)
                effective = runner.preprocess_args(dict(vars(runner.config)))
                runner.initialize(effective)
                initialized = True
                torch.cuda.synchronize()
                result["load_seconds"] = round(time.monotonic() - t0, 4)
                pipeline = runner.model.pipe
                calls_per_step = expected_forward_calls_per_step(
                    args.model, args.cfg_degree, request["guidance_scale"]
                )
                torch.cuda.reset_peak_memory_stats(rank)
                t_run = time.monotonic()
                with cuda_step_timing_hook(pipeline.transformer, torch) as raw_call_ms:
                    output, timings = runner.run(effective)
                torch.cuda.synchronize()
                result["cold_rank_request_wall_seconds"] = round(time.monotonic() - t_run, 4)
                result["official_runner_timings_seconds"] = [float(v) for v in timings]
                result["step_timings"] = summarize_step_timings(
                    raw_call_ms, int(request["num_inference_steps"]), calls_per_step
                )
                result["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(rank))
                result["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved(rank))
                result["gpu"] = torch.cuda.get_device_name(rank)
                result["gpu_uuid"] = str(torch.cuda.get_device_properties(rank).uuid)
                del output
        finally:
            if initialized and runner is not None:
                runner.cleanup()
            elif torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        result.update(status="native_scaling_completed", planned_returncode=0)
    except Exception as exc:
        result.update(
            status="native_scaling_failed",
            planned_returncode=1,
            error=repr(exc),
            traceback=traceback.format_exc()[-10000:],
        )

    dist_base.publish_json(rank_dir / "summary.json", result)
    if rank == 0:
        paths = [output_dir / f"rank-{i}" / "summary.json" for i in range(args.world_size)]
        while not all(p.is_file() for p in paths) and time.monotonic() < deadline:
            time.sleep(0.25)
        rank_rows = [json.loads(p.read_bytes()) for p in paths if p.is_file()]
        all_ok = len(rank_rows) == args.world_size and all(r.get("planned_returncode") == 0 for r in rank_rows)
        summary = {
            "schema": SCHEMA,
            "status": "native_scaling_completed" if all_ok else "native_scaling_failed",
            "model": args.model,
            "shape_id": args.shape,
            "world_size": args.world_size,
            "cfg_degree": args.cfg_degree,
            "ulysses_degree": args.ulysses_degree,
            "parallel_form": pcie.form_name(args.cfg_degree, args.ulysses_degree),
            "request": result.get("request"),
            "runner_config": result.get("runner_config"),
            "step_timings": result.get("step_timings"),
            "official_runner_timings_seconds": result.get("official_runner_timings_seconds"),
            "cold_rank_request_wall_seconds": result.get("cold_rank_request_wall_seconds"),
            "rank_receipts": rank_rows,
            "qualification_eligible": False,
            "allows": [],
        }
        if not all_ok:
            summary["error"] = result.get("error", "one or more ranks failed")
            result["planned_returncode"] = 1
        dist_base.publish_json(output_dir / "summary.json", summary)
        print(json.dumps({"rank": 0, "status": summary["status"], "summary": str(output_dir / "summary.json")}), flush=True)
    stop.set()
    return int(result["planned_returncode"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "compare"))
    parser.add_argument("--world-size", type=int, choices=ALLOWED_WORLDS, default=2)
    parser.add_argument("--cfg-degree", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ulysses-degree", type=int, choices=(1, 2, 3, 4, 6, 8), default=2)
    parser.add_argument("--model", choices=tuple(ALL_CLASSES), default="wan21")
    parser.add_argument("--shape", default="wan21-480p")
    parser.add_argument("--mode", choices=("quick", "formal"), default="formal")
    parser.add_argument("--prompt", default="A dog walking in a park.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", type=Path, default=source.DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-sha256", default="86aae608c7f86d194e77a5e2e9645b192b921d8137174cbcdb2825ed7d94b392")
    parser.add_argument("--source-root", type=Path, default=Path("/shared/data/DITSERVE/external-systems/20261103-source-v1/xDiT"))
    parser.add_argument("--source-manifest", type=Path, default=Path("/shared/data/DITSERVE/external-systems/20261103-source-v1/xDiT.source.json"))
    parser.add_argument("--source-manifest-sha256", default="49595ace72ba61efd7c7ec9d28f11ed1c430a3848903031dcf9e0913925125e5")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--models", type=Path, default=native.DEFAULT_MODELS)
    parser.add_argument("--models-sha256", default=native.DEFAULT_MODELS_SHA256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--xdit-summary", type=Path, action="append", default=[])
    parser.add_argument("--decomposition", type=Path)
    args = parser.parse_args(argv)

    if args.command == "plan":
        proto = source.read_json(args.protocol.read_bytes())
        req = request_plan_any_modality(proto, args.model, args.shape, args.mode, args.prompt, args.seed)
        cfg = scaling_runner_config(
            args.model,
            req,
            {"flow_shift": 3.0 if args.model == "wan21" else 5.0},
            args.output or Path("/tmp/xdit-plan"),
            args.world_size,
            args.cfg_degree,
            args.ulysses_degree,
        )
        plan = {
            "schema": SCHEMA,
            "status": "planned",
            "model": args.model,
            "world_size": args.world_size,
            "cfg_degree": args.cfg_degree,
            "ulysses_degree": args.ulysses_degree,
            "parallel_form": cfg["parallel_form"],
            "expected_forward_calls_per_step": expected_forward_calls_per_step(
                args.model, args.cfg_degree, req["guidance_scale"]
            ),
            "request": req,
            "runner_config": cfg,
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        if not args.model_root:
            parser.error("--model-root is required for run")
        if not (args.output_dir or args.output):
            parser.error("--output-dir is required for run")
        return run_scaling(args)

    if not args.decomposition:
        parser.error("--decomposition is required for compare")
    decomp = json.loads(args.decomposition.read_bytes())
    xdit_recs = [json.loads(p.read_bytes()) for p in args.xdit_summary]
    comp = compare_xdit_and_ditserve_scaling(xdit_recs, decomp)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(comp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comp, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
