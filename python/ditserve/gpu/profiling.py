"""Optional, real CUDA profiling. Importing this module needs only Python's stdlib.

No simulation fallback exists here. The reference runtime's synthetic profiles
must never be confused with the measured records emitted by this module.
"""

from ..util import atomic_json

import argparse
import copy
import dataclasses
import datetime
import enum
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path


SCHEMA_VERSION = "ditserve.gpu-profile.v1"
MODEL_IDS = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}
DEFAULT_PROMPT = "A red bicycle leaning against a blue wall in warm afternoon light."


class ProfilingError(RuntimeError):
    """A missing dependency or unsupported runtime prevents real measurement."""

    def __init__(self, message, category="profiling_error", details=None, partial_report=None):
        super().__init__(message)
        self.category = category
        self.details = details or {}
        self.partial_report = partial_report


def _positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def _finite_nonnegative(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="check dependencies/CUDA without loading weights")
    parser.add_argument("--preset", choices=["v100-wan"], help="Wan on Volta: FP16 DiT, FP32 VAE, CPU FP32 encoder")
    parser.add_argument("--model", choices=sorted(MODEL_IDS))
    parser.add_argument("--model-id", help="override with a compatible local checkpoint or Hub ID")
    parser.add_argument("--revision", help="Hub commit hash recommended; omitted means repository default")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache", choices=["none", "cache-dit"], default="none")
    parser.add_argument("--attention", choices=["default", "cache-dit-native"],
                        help="cache-dit-native uses the same official attention setup in both cache arms")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--text-encoder-device", choices=["cpu", "cuda"])
    parser.add_argument("--text-encoder-dtype", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vae-dtype", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=_positive)
    parser.add_argument("--width", type=_positive)
    parser.add_argument("--frames", type=_positive)
    parser.add_argument("--steps", type=_positive)
    parser.add_argument("--guidance-scale", type=_finite_nonnegative)
    parser.add_argument("--batch-size", type=_positive, default=1,
                        help="offline batch of identical prompts; not online continuous batching")
    parser.add_argument("--warmup", type=_nonnegative, default=1, help="full untimed pipeline runs")
    parser.add_argument("--repeats", type=_positive, default=3)
    parser.add_argument("--seed", type=_nonnegative, default=0)
    parser.add_argument("--cache-fn", type=_positive, default=8)
    parser.add_argument("--cache-bn", type=_nonnegative, default=0)
    parser.add_argument("--cache-warmup", type=_nonnegative, default=8)
    parser.add_argument("--cache-threshold", type=_finite_nonnegative, default=0.08)
    parser.add_argument("--cache-max-continuous", type=_positive, default=3)
    parser.add_argument("--output", type=Path, help="write measured JSON; required except for --preflight")
    parser.add_argument("--artifact-dir", type=Path, help="also save decoded outputs as NumPy arrays outside timing")
    return parser


def normalize_args(args, parser):
    if args.preset == "v100-wan" and args.model not in (None, "wan"):
        parser.error("--preset v100-wan requires --model wan")
    args.model = args.model or ("wan" if args.preset == "v100-wan" else "flux")
    args.dtype = args.dtype or ("float16" if args.preset == "v100-wan" else "bfloat16")
    args.attention = args.attention or ("cache-dit-native" if args.preset == "v100-wan" else "default")
    args.text_encoder_device = args.text_encoder_device or ("cpu" if args.preset == "v100-wan" else "cuda")
    args.text_encoder_dtype = args.text_encoder_dtype or ("float32" if args.text_encoder_device == "cpu" else args.dtype)
    args.vae_dtype = args.vae_dtype or ("float32" if args.model == "wan" else args.dtype)
    if args.text_encoder_device == "cpu" and (args.model != "wan" or args.text_encoder_dtype != "float32"):
        parser.error("explicit CPU prompt encoding supports Wan with --text-encoder-dtype float32 only")
    defaults = {"flux": (1024, 1024, 1, 28, 3.5), "wan": (480, 832, 81, 50, 5.0)}[args.model]
    for name, value in zip(("height", "width", "frames", "steps", "guidance_scale"), defaults):
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.model == "flux" and args.frames != 1:
        parser.error("FLUX profiling requires --frames 1")
    if args.model == "flux" and args.negative_prompt:
        parser.error("negative prompts are not enabled in this FLUX profile recipe")
    if args.model == "wan" and (args.frames - 1) % 4:
        parser.error("Wan requires --frames = 4*k+1")
    if args.height % 16 or args.width % 16:
        parser.error("height and width must be multiples of 16")
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        parser.error("device must be an explicit CUDA device such as cuda:0")
    if args.seed + args.repeats + args.warmup >= 2 ** 63:
        parser.error("seed and run counts exceed the supported seed range")
    if not args.preflight and args.output is None:
        parser.error("--output is required for real profiling")
    if args.cache == "cache-dit" and args.attention != "cache-dit-native":
        parser.error("use --attention cache-dit-native with --cache cache-dit; use the same attention option for its baseline")
    args.model_id = args.model_id or MODEL_IDS[args.model]
    return args


def package_versions():
    result = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "cache-dit", "numpy", "huggingface-hub"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def native_bf16_supported(torch, device):
    """Never count PyTorch's emulated BF16 support as native device support."""
    with torch.cuda.device(device):
        if torch.cuda.get_device_capability(device)[0] < 8:
            return False
        try:
            return bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:
            # Older CUDA PyTorch releases lack the keyword; Ampere+ is still
            # required, so their permissive check cannot admit Volta emulation.
            return bool(torch.cuda.is_bf16_supported())


def _sdpa_math_context(torch):
    """Pin every GPU execution arm to SDPA MATH, including warmup/preflight."""
    attention = getattr(getattr(torch, "nn", None), "attention", None)
    if attention is not None and hasattr(attention, "sdpa_kernel"):
        return attention.sdpa_kernel([attention.SDPBackend.MATH])
    options = dict(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    try:
        return torch.backends.cuda.sdp_kernel(enable_cudnn=False, **options)
    except TypeError:
        return torch.backends.cuda.sdp_kernel(**options)


def _kernel_smoke_tests(torch, args):
    """Actually execute tiny CUDA matrix and MATH-SDPA kernels, no weights."""
    records = []
    gpu_dtypes = {args.dtype, args.vae_dtype}
    if args.text_encoder_device == "cuda":
        gpu_dtypes.add(args.text_encoder_dtype)
    for dtype_name in sorted(gpu_dtypes):
        for operation in ("matmul", "sdpa_math"):
            record = {"operation": operation, "dtype": dtype_name, "device": args.device}
            try:
                with torch.cuda.device(args.device), torch.inference_mode(), _sdpa_math_context(torch):
                    dtype = getattr(torch, dtype_name)
                    started = time.perf_counter()
                    if operation == "matmul":
                        values = torch.ones((16, 16), device=args.device, dtype=dtype)
                        output = values @ values
                    else:
                        values = torch.ones((1, 2, 16, 32), device=args.device, dtype=dtype)
                        output = torch.nn.functional.scaled_dot_product_attention(values, values, values)
                    torch.cuda.synchronize(args.device)
                    record["wall_ms"] = (time.perf_counter() - started) * 1000.0
                    record["finite"] = bool(torch.isfinite(output).all().item())
                    record["status"] = "passed" if record["finite"] else "failed"
                    if not record["finite"]:
                        record["error"] = "NONFINITE_OUTPUT in CUDA preflight"
                    del output, values
            except Exception as exc:
                record.update(status="failed", finite=None, error="{}: {}".format(type(exc).__name__, exc))
            records.append(record)
    return records


def preflight(args):
    """Return a truthful environment report even on a stdlib-only machine."""
    result = {"schema_version": SCHEMA_VERSION, "kind": "preflight", "measured": False,
              "python": platform.python_version(), "packages": package_versions(), "errors": [],
              "cuda_available": False, "device": args.device, "weights_loaded": False,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
              "device_numbering": "logical CUDA index after CUDA_VISIBLE_DEVICES filtering",
              "sdpa_backend": "MATH", "kernel_checks": []}
    required = ["torch", "diffusers", "transformers", "accelerate", "numpy"]
    if args.cache == "cache-dit" or args.attention == "cache-dit-native":
        required.append("cache-dit")
    for name in required:
        if result["packages"][name] is None:
            result["errors"].append("Missing optional GPU dependency: " + name)
    try:
        torch = importlib.import_module("torch")
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_runtime"] = torch.version.cuda
        if not result["cuda_available"]:
            result["errors"].append("CUDA is unavailable; CPU or synthetic profiling is not supported")
        elif int(args.device[5:]) >= torch.cuda.device_count():
            result["errors"].append("Requested CUDA device is not visible")
        else:
            props = torch.cuda.get_device_properties(args.device)
            result["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                             "compute_capability": [props.major, props.minor],
                             "uuid": str(getattr(props, "uuid", "unknown"))}
            architectures = list(torch.cuda.get_arch_list())
            result["torch_cuda_architectures"] = architectures
            if (props.major, props.minor) == (7, 0) and not any(arch in architectures for arch in ("sm_70", "compute_70")):
                result["errors"].append("PyTorch wheel does not contain sm_70/compute_70 support for Volta")
            if args.preset == "v100-wan" and ((props.major, props.minor) != (7, 0) or "V100" not in props.name.upper()):
                result["errors"].append("v100-wan preset requires an actual V100 with compute capability sm_70")
            with torch.cuda.device(args.device):
                result["native_bfloat16_supported"] = native_bf16_supported(torch, args.device)
                gpu_dtypes = [args.dtype, args.vae_dtype]
                if args.text_encoder_device == "cuda":
                    gpu_dtypes.append(args.text_encoder_dtype)
                if "bfloat16" in gpu_dtypes and not result["native_bfloat16_supported"]:
                    result["errors"].append("Selected device does not support native bfloat16; emulation is excluded")
            if not result["errors"]:
                result["kernel_checks"] = _kernel_smoke_tests(torch, args)
                for check in result["kernel_checks"]:
                    if check["status"] != "passed":
                        result["errors"].append("CUDA {} {} check failed: {}".format(check["operation"], check["dtype"], check.get("error")))
    except Exception as exc:
        result["errors"].append("Cannot initialize torch/CUDA: {}: {}".format(type(exc).__name__, exc))
    if not result["errors"]:
        try:
            diffusers = importlib.import_module("diffusers")
            required_classes = ["FluxPipeline"] if args.model == "flux" else ["WanPipeline", "AutoencoderKLWan"]
            for name in required_classes:
                if not hasattr(diffusers, name):
                    result["errors"].append("Installed diffusers lacks " + name)
            if args.cache == "cache-dit" or args.attention == "cache-dit-native":
                cache_dit = importlib.import_module("cache_dit")
                for name in ("DBCacheConfig", "enable_cache", "summary", "set_attn_backend"):
                    if not callable(getattr(cache_dit, name, None)):
                        result["errors"].append("Installed cache-dit lacks public API " + name)
        except Exception as exc:
            result["errors"].append("GPU dependency import failed: {}: {}".format(type(exc).__name__, exc))
    result["ready"] = not result["errors"]
    return result


def json_safe(value):
    """Serialize public stats/configs without pretending opaque objects are tensors."""
    if isinstance(value, enum.Enum):
        return json_safe(value.value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if dataclasses.is_dataclass(value):
        return {field.name: json_safe(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return {"unsupported_type": type(value).__module__ + "." + type(value).__name__}


def tensor_inventory(value, is_tensor):
    """Count logical tensor payload, not allocated storage or a restorable snapshot.

    Repeated references are deduplicated by identity. Distinct tensor views may
    overlap; their logical payloads are counted separately and never called VRAM.
    Only containers are traversed, never arbitrary model/module object graphs.
    """
    seen = set()
    tensors = []
    opaque = []

    def visit(item, path):
        if id(item) in seen:
            return
        if is_tensor(item):
            seen.add(id(item))
            tensors.append({"path": path, "shape": list(item.shape), "dtype": str(item.dtype),
                            "device": str(item.device), "bytes": int(item.numel() * item.element_size())})
        elif isinstance(item, dict):
            seen.add(id(item))
            for key, child in item.items():
                visit(child, path + "." + str(key))
        elif isinstance(item, (list, tuple)):
            seen.add(id(item))
            for index, child in enumerate(item):
                visit(child, path + "[{}]".format(index))
        elif item is not None and not isinstance(item, (str, int, float, bool, enum.Enum)):
            opaque.append({"path": path, "type": type(item).__name__})

    visit(value, "state")
    return {"status": "measured_tensor_payload_only", "bytes": sum(t["bytes"] for t in tensors),
            "tensors": tensors, "opaque_fields_not_traversed": opaque,
            "is_complete_checkpoint": False, "counts_allocated_storage": False}


def cache_state_support(cache_name):
    if cache_name == "none":
        return {"status": "not_applicable", "bytes": 0, "restorable_snapshot": False}
    return {"status": "unsupported", "bytes": None, "restorable_snapshot": False,
            "reason": "cache_dit.summary exposes statistics, not a supported complete per-request cache tensor export/import API; migration is not measured"}


def _cache_reset_record(enabled):
    if not enabled:
        return {"status": "not_applicable", "wall_ms": 0.0, "separate_operation_executed": False}
    return {"status": "auto_inside_pipeline", "wall_ms": 0.0,
            "separate_operation_executed": False, "reset_latency_separately_measured": False,
            "reason": "cache-dit full-pipeline contexts reset inside each pipeline call; its internal reset and statistics hooks are included in e2e_ms. Zero denotes no external operation, not a measured zero-cost reset"}


class CacheBackend:
    def __init__(self, pipe, args):
        self.pipe, self.args, self.module, self.config = pipe, args, None, None
        if args.cache == "cache-dit" or args.attention == "cache-dit-native":
            self.module = importlib.import_module("cache_dit")
        if args.cache == "cache-dit":
            self.config = self.module.DBCacheConfig(
                Fn_compute_blocks=args.cache_fn, Bn_compute_blocks=args.cache_bn,
                max_warmup_steps=args.cache_warmup, warmup_interval=1,
                residual_diff_threshold=args.cache_threshold, max_cached_steps=-1,
                max_continuous_cached_steps=args.cache_max_continuous,
                enable_separate_cfg=args.model == "wan" and args.guidance_scale > 1,
                cfg_compute_first=False, cfg_diff_compute_separate=True,
                num_inference_steps=args.steps)
            self.module.enable_cache(pipe, cache_config=copy.deepcopy(self.config), attention_backend="native")
        elif args.attention == "cache-dit-native":
            self.module.set_attn_backend(pipe, attention_backend="native")

    def reset(self):
        # cache-dit 1.5 full-pipeline contexts are nonpersistent. The wrapped
        # pipeline call resets them automatically; public refresh_context only
        # supports persistent transformer contexts and would warn/no-op here.
        return _cache_reset_record(self.config is not None)

    def stats(self):
        if self.config is None:
            return {"status": "disabled", "entries": []}
        entries = self.module.summary(self.pipe, details=True, logging=False)
        if not isinstance(entries, (list, tuple)) or not entries:
            raise ProfilingError("cache_dit.summary did not return the supported structured statistics")
        records = json_safe(entries)
        if not any(isinstance(item, dict) and isinstance(item.get("accumulated_executed_steps"), (int, float))
                   and item["accumulated_executed_steps"] > 0 for item in records):
            raise ProfilingError("cache_dit.summary reports no executed cache context; refusing to label an inactive/unsupported cache measured")
        return {"status": "measured", "api": "cache_dit.summary", "entries": records}


def annotate_cache_steps(diagnostic, stats):
    """Attach public cache histories, without inferring whole-transformer paths.

    cache-dit's CachedContext.add_cached_step records get_current_step(), which
    is executed_steps-1 (a denoising index, not a transformer-forward index).
    Require matching counters and in-range indices before aligning versions.
    Summary entries are opaque scopes and may overlap; never add their counts.
    """
    mapping = {"source": "cache_dit.summary cached_steps/cfg_cached_steps",
               "index_semantics": "zero-based denoising step",
               "whole_transformer_compute_or_reuse": "not_classified",
               "scope": "public summary entries; potentially overlapping block and transformer scopes",
               "entries": []}
    if stats["status"] == "disabled":
        mapping["status"] = "cache_disabled"
        diagnostic["cache_step_mapping"] = mapping
        return
    count = diagnostic["step_count"]
    for index, entry in enumerate(stats["entries"]):
        if not isinstance(entry, dict):
            continue
        cached = entry.get("cached_steps")
        cfg = entry.get("cfg_cached_steps", [])
        valid_lists = all(isinstance(values, list) and all(type(i) is int and 0 <= i < count for i in values)
                          for values in (cached, cfg))
        aligned = (valid_lists and entry.get("accumulated_executed_steps") == count
                   and entry.get("accumulated_transformer_executed_steps") == diagnostic["forward_count"])
        mapping["entries"].append({"summary_entry_index": index,
                                   "status": "mapped" if aligned else "unsupported_counter_or_index_contract",
                                   "cached_steps": cached, "cfg_cached_steps": cfg})
    mapped = [entry for entry in mapping["entries"] if entry["status"] == "mapped"]
    mapping["status"] = "mapped_public_statistics" if mapped else "unsupported"
    for step in diagnostic["steps"]:
        step["cache_evidence"] = [
            {"summary_entry_index": entry["summary_entry_index"],
             "reported_cached": step["step_index"] in entry["cached_steps"],
             "reported_cfg_cached": step["step_index"] in entry["cfg_cached_steps"]}
            for entry in mapped]
    diagnostic["cache_step_mapping"] = mapping


class StepObserver:
    """CUDA event timing on the transformer plus metadata-only callback sampling."""
    def __init__(self, torch, pipe):
        self.torch, self.pipe = torch, pipe
        self.calls, self.steps, self.handles = [], [], []
        self.pending = []
        self.step_start = 0
        self.finite_checks = []

    def __enter__(self):
        try:
            self.handles.append(self.pipe.transformer.register_forward_pre_hook(self._before))
            self.handles.append(self.pipe.transformer.register_forward_hook(self._after))
        except Exception:
            for handle in self.handles:
                handle.remove()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()

    def _before(self, module, inputs):
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        self.pending.append((start, end))

    def _after(self, module, inputs, output):
        start, end = self.pending.pop()
        end.record()
        self.calls.append((start, end))

    def callback(self, pipe, index, timestep, kwargs):
        if "latents" not in kwargs:
            raise ProfilingError("The pipeline callback did not expose latents")
        sampler = {name: value for name, value in vars(pipe.scheduler).items()
                   if name != "_internal_dict"}
        self.steps.append({"step_index": int(index), "forward_indices": list(range(self.step_start, len(self.calls))),
                           "latents": tensor_inventory(kwargs["latents"], self.torch.is_tensor),
                           "sampler": tensor_inventory(sampler, self.torch.is_tensor)})
        # Retain one boolean per step, not a full latent copy or a per-step CPU
        # synchronization. The timed pipeline synchronizes before report().
        self.finite_checks.append(self.torch.isfinite(kwargs["latents"]).all())
        self.step_start = len(self.calls)
        return kwargs

    def report(self, expected_steps):
        if ([step["step_index"] for step in self.steps] != list(range(expected_steps))
                or not self.calls or self.step_start != len(self.calls)):
            raise ProfilingError("Incomplete transformer/callback instrumentation; unsupported pipeline version")
        forwards = [{"forward_index": i, "cuda_ms": float(start.elapsed_time(end))}
                    for i, (start, end) in enumerate(self.calls)]
        for step, finite in zip(self.steps, self.finite_checks):
            step["transformer_cuda_ms"] = sum(forwards[i]["cuda_ms"] for i in step["forward_indices"])
            step["latents_finite"] = bool(finite.item())
        report = {"status": "measured", "timing": "CUDA events on current stream; no per-step synchronization; deferred finite reductions",
                "forwards": forwards, "steps": self.steps,
                "forward_count": len(forwards), "step_count": len(self.steps)}
        if any(not step["latents_finite"] for step in self.steps):
            raise ProfilingError("NONFINITE_LATENTS detected in diagnostic denoising steps",
                                 category="nonfinite_latents", details={"diagnostic": report})
        return report


def timed_call(torch, pipe, kwargs, device, clock=time.perf_counter, prepare_kwargs=None):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    initial = int(torch.cuda.memory_allocated(device))
    started_at_unix_ns = time.time_ns()
    start = clock()
    stages = {}
    if prepare_kwargs is not None:
        kwargs, stages = prepare_kwargs(kwargs)
    output = pipe(**kwargs)
    torch.cuda.synchronize(device)
    duration = (clock() - start) * 1000.0
    return output, dict(stages, **{"e2e_ms": duration, "initial_allocated_bytes": initial,
                    "started_at_unix_ns": started_at_unix_ns, "finished_at_unix_ns": time.time_ns(),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))})


def timed_cache_reset(torch, backend, device, clock=time.perf_counter):
    """Compatibility field for full-pipeline runs; no separate reset exists.

    Keep the historical signature/output field for consumers, while making no
    clock or CUDA calls and never pretending an external no-op measures reset.
    """
    return _cache_reset_record(backend.config is not None)


def _fingerprint(output, model):
    field = "images" if model == "flux" else "frames"
    array = getattr(output, field, None)
    if not all(hasattr(array, name) for name in ("tobytes", "shape", "dtype")):
        raise ProfilingError("Expected a NumPy array in pipeline output.{} for output_type=np; unsupported pipeline output version".format(field))
    expected_rank = 4 if model == "flux" else 5
    if len(array.shape) != expected_rank or array.shape[-1] != 3 or any(int(size) < 1 for size in array.shape):
        raise ProfilingError("Unexpected decoded {} output shape: {}".format(model, array.shape))
    result = {"sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
              "shape": list(array.shape), "dtype": str(array.dtype)}
    numpy = importlib.import_module("numpy")
    result["finite"] = bool(numpy.isfinite(array).all())
    if not result["finite"]:
        raise ProfilingError("NONFINITE_OUTPUT detected in decoded {} array".format(model),
                             category="nonfinite_output", details={"output": result})
    return result


def _wan_pipeline_class(base_class):
    """Keep denoising on the DiT device with an explicitly resident CPU encoder.

    DiffusionPipeline.device chooses the first module in a signature-derived
    set; with mixed placement that may be the CPU encoder. Only these placement
    properties change. The inherited Wan call/encoding/sampling are unchanged.
    The Wan prefix also preserves cache-dit's documented adapter discovery.
    """
    class WanExplicitExecutionPipeline(base_class):
        @property
        def device(self):
            return self.transformer.device

        @property
        def _execution_device(self):
            return self.transformer.device

    return WanExplicitExecutionPipeline


def _pipeline(torch, diffusers, args):
    load = {"torch_dtype": getattr(torch, args.dtype), "local_files_only": args.local_files_only}
    if args.revision:
        load["revision"] = args.revision
    if args.model == "flux":
        pipe = diffusers.FluxPipeline.from_pretrained(args.model_id, **load)
    else:
        vae_args = dict(load, torch_dtype=getattr(torch, args.vae_dtype))
        vae = diffusers.AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", **vae_args)
        # Load directly in the intended precision: FP16 loading followed by a
        # cast back to FP32 would irreversibly round the encoder weights.
        transformers = importlib.import_module("transformers")
        encoder_args = dict(load, torch_dtype=getattr(torch, args.text_encoder_dtype))
        encoder = transformers.UMT5EncoderModel.from_pretrained(args.model_id, subfolder="text_encoder", **encoder_args)
        pipeline_class = (_wan_pipeline_class(diffusers.WanPipeline)
                          if args.text_encoder_device == "cpu" else diffusers.WanPipeline)
        pipe = pipeline_class.from_pretrained(args.model_id, vae=vae, text_encoder=encoder, **load)
    if args.model == "wan":
        # Never pipe.to(cuda) here: it transiently puts the large CPU encoder on
        # the GPU and can OOM before it is moved back.
        pipe.transformer.to(device=args.device, dtype=getattr(torch, args.dtype))
        if getattr(pipe, "transformer_2", None) is not None:
            pipe.transformer_2.to(device=args.device, dtype=getattr(torch, args.dtype))
        pipe.vae.to(device=args.device, dtype=getattr(torch, args.vae_dtype))
        encoder_device = "cpu" if args.text_encoder_device == "cpu" else args.device
        pipe.text_encoder.to(device=encoder_device, dtype=getattr(torch, args.text_encoder_dtype))
    else:
        pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _failure_artifact_dir(args):
    if args.artifact_dir is not None:
        directory = args.artifact_dir
    elif args.output is not None:
        directory = args.output.parent / (args.output.stem + "-failure-artifacts")
    else:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _require_finite_tensor(torch, tensor, name, args):
    if bool(torch.isfinite(tensor).all().item()):
        return
    details = {"name": name, "finite": False, "shape": list(tensor.shape),
               "dtype": str(tensor.dtype), "device": str(tensor.device)}
    try:
        directory = _failure_artifact_dir(args)
        if directory is not None:
            path = directory / ("failure-" + name + ".npy")
            array = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
            importlib.import_module("numpy").save(str(path), array)
            details["artifact"] = str(path.resolve())
    except Exception as exc:
        details["artifact_save_error"] = "{}: {}".format(type(exc).__name__, exc)
    raise ProfilingError("NONFINITE_EMBEDDINGS detected in " + name,
                         category="nonfinite_embeddings", details=details)


def _prepare_prompt_kwargs(torch, pipe, args, kwargs, clock=time.perf_counter):
    """Per-request CPU FP32 prompt encoding and explicit embedding transfer.

    Called inside the E2E timing window on every warmup and measured run. The
    public Wan encode_prompt interface is used; embeddings are never cached
    between requests or arms.
    """
    if args.text_encoder_device != "cpu":
        return dict(kwargs), {}
    result = dict(kwargs)
    start = clock()
    prompt_embeds, negative_embeds = pipe.encode_prompt(
        prompt=result.pop("prompt"), negative_prompt=result.pop("negative_prompt", None),
        do_classifier_free_guidance=args.guidance_scale > 1.0, num_videos_per_prompt=1,
        max_sequence_length=512, device="cpu", dtype=torch.float32)
    _require_finite_tensor(torch, prompt_embeds, "cpu-prompt-embeds", args)
    if negative_embeds is not None:
        _require_finite_tensor(torch, negative_embeds, "cpu-negative-prompt-embeds", args)
    encoded = clock()
    target_dtype = pipe.transformer.dtype
    result["prompt_embeds"] = prompt_embeds.to(device=args.device, dtype=target_dtype)
    result["negative_prompt_embeds"] = (
        negative_embeds.to(device=args.device, dtype=target_dtype) if negative_embeds is not None else None)
    torch.cuda.synchronize(args.device)
    _require_finite_tensor(torch, result["prompt_embeds"], "gpu-prompt-embeds", args)
    if result["negative_prompt_embeds"] is not None:
        _require_finite_tensor(torch, result["negative_prompt_embeds"], "gpu-negative-prompt-embeds", args)
    transferred = clock()
    return result, {"cpu_prompt_encode_ms": (encoded - start) * 1000.0,
                    "prompt_embeddings_h2d_ms": (transferred - encoded) * 1000.0}


def _call_kwargs(torch, args, seed):
    result = {"prompt": [args.prompt] * args.batch_size, "height": args.height, "width": args.width,
              "num_inference_steps": args.steps, "guidance_scale": args.guidance_scale,
              "generator": torch.Generator(device="cpu").manual_seed(seed), "output_type": "np"}
    if args.model == "wan":
        result.update(num_frames=args.frames, negative_prompt=[args.negative_prompt] * args.batch_size)
    return result


def _component_metadata(pipe):
    result = {}
    for name in ("transformer", "transformer_2", "text_encoder", "text_encoder_2", "vae"):
        module = getattr(pipe, name, None)
        if module is not None:
            config = getattr(module, "config", {})
            config = config.to_dict() if hasattr(config, "to_dict") else dict(config)
            dtypes, devices = set(), set()
            for iterator_name in ("parameters", "buffers"):
                iterator = getattr(module, iterator_name, None)
                if callable(iterator):
                    for tensor in iterator():
                        dtypes.add(str(tensor.dtype))
                        devices.add(str(tensor.device))
            result[name] = {"class": type(module).__name__, "dtype": str(getattr(module, "dtype", None)),
                            "device": str(getattr(module, "device", None)),
                            "actual_tensor_dtypes": sorted(dtypes), "actual_tensor_devices": sorted(devices),
                            "revision_resolved": config.get("_commit_hash", getattr(module.config, "_commit_hash", None)),
                            "config": json_safe(config)}
    processors = getattr(pipe.transformer, "attn_processors", {})
    result["attention_processor_classes"] = sorted({type(value).__module__ + "." + type(value).__name__
                                                     for value in processors.values()})
    return result


def _base_report(args):
    return {"schema_version": SCHEMA_VERSION, "kind": "gpu_profile", "measured": False,
              "status": "running",
              "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "model_id": args.model_id, "revision_requested": args.revision,
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "measurement_protocol": {"e2e_includes": "per-request text encoding, embedding H2D and finite checks when CPU encoder is selected, denoising, VAE decode, CPU numpy conversion; cache-dit internal context reset and statistics hooks when enabled",
                                       "e2e_excludes": "loading, warmup, external public summary extraction, output hashing/saving",
                                       "cache_context_lifecycle": "full-pipeline nonpersistent contexts automatically reset inside each pipeline call; no public refresh_context call or separate reset measurement",
                                       "pipeline_plus_refresh": "compatibility alias of e2e_ms because cache reset is already inside the pipeline call; cache_refresh.wall_ms=0 denotes no external operation, not measured reset latency",
                                       "diagnostic_run": "separate same-seed run with transformer hooks, per-step tensor metadata and deferred latent finite reductions",
                                       "batching": "offline fixed batch only", "offload": False, "compile": False,
                                       "sdpa_backend": "MATH for preflight, warmup, measured and diagnostic runs",
                                       "text_encoder_placement": args.text_encoder_device,
                                       "prompt_embeddings_reused_between_requests": False,
                                       "wan_max_sequence_length": 512 if args.model == "wan" else None,
                                       "memory": "PyTorch allocator peaks, not total process/device memory"},
              "warmups": [], "runs": []}


def _failure_report(exc, args):
    report = getattr(exc, "partial_report", None)
    if report is None:
        report = _base_report(args)
    category = getattr(exc, "category", "profiling_error")
    if category == "profiling_error" and ("out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"):
        category = "cuda_oom"
    report["status"] = "failed"
    report["failure_type"] = "nonfinite" if category.startswith("nonfinite") else category
    report["failure"] = {"category": category, "type": type(exc).__name__, "message": str(exc),
                         "details": json_safe(getattr(exc, "details", {}))}
    report["completed_runs"] = sum(run.get("status") == "completed" for run in report.get("runs", []))
    return report


def _save_decoded_output(output, args, record, label, force=False):
    directory = _failure_artifact_dir(args) if force else args.artifact_dir
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (label + ".npy")
    array = output.images if args.model == "flux" else output.frames
    importlib.import_module("numpy").save(str(path), array)
    record["artifact"] = str(path.resolve())


def _record_output(output, args, stage, label):
    try:
        stage["output"] = _fingerprint(output, args.model)
    except ProfilingError as exc:
        if exc.category == "nonfinite_output":
            stage["output"] = exc.details["output"]
            try:
                _save_decoded_output(output, args, stage["output"], label, force=True)
            except Exception as save_exc:
                stage["output"]["artifact_save_error"] = "{}: {}".format(type(save_exc).__name__, save_exc)
        raise
    _save_decoded_output(output, args, stage["output"], label)


def _revision_metadata(args, components):
    commits = {entry.get("revision_resolved") for entry in components.values() if isinstance(entry, dict)}
    commits.discard(None)
    # ModelMixin config may omit the Hub commit; resolving an existing cache
    # entry is read-only and never fetches a newer repository revision.
    try:
        hub = importlib.import_module("huggingface_hub")
        cached = hub.try_to_load_from_cache(args.model_id, "model_index.json", revision=args.revision or "main")
        if isinstance(cached, str):
            parts = Path(cached).parts
            if "snapshots" in parts:
                commits.add(parts[parts.index("snapshots") + 1])
    except Exception:
        pass
    return {"requested": args.revision, "resolved_commits": sorted(commits),
            "status": "resolved" if commits else "unresolved_or_local_checkpoint"}


def run_profile(args):
    report = _base_report(args)
    try:
        environment = preflight(args)
        report["environment"] = environment
        if not environment["ready"]:
            category = ("nonfinite_preflight" if any(check.get("finite") is False for check in environment.get("kernel_checks", []))
                        else "preflight")
            raise ProfilingError("; ".join(environment["errors"]), category=category)
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        torch.cuda.set_device(args.device)
        pipe = _pipeline(torch, diffusers, args)
        report["components"] = _component_metadata(pipe)
        report["revision"] = _revision_metadata(args, report["components"])
        report["pipeline"] = {"class": type(pipe).__name__, "device": str(pipe.device),
                              "execution_device": str(pipe._execution_device)}
        if str(pipe._execution_device) != args.device:
            raise ProfilingError("Pipeline execution device does not match requested CUDA device")
        report["scheduler"] = {"class": type(pipe.scheduler).__name__, "config": json_safe(dict(pipe.scheduler.config))}
        backend = CacheBackend(pipe, args)
        # Attention processors may change when the cache backend is installed.
        report["components"] = _component_metadata(pipe)
        report["cache"] = {"implementation": "cache-dit DBCache" if backend.config else "none",
                           "config": json_safe(backend.config), "state": cache_state_support(args.cache)}
        prepare = lambda kwargs: _prepare_prompt_kwargs(torch, pipe, args, kwargs)
        with torch.inference_mode(), _sdpa_math_context(torch):
            for index in range(args.warmup):
                seed = args.seed + args.repeats + index
                warmup = {"index": index, "seed": seed, "status": "running"}
                report["warmups"].append(warmup)
                backend.reset()
                print("GPU profile: warmup {} / {}, seed={}".format(index + 1, args.warmup, seed), file=sys.stderr, flush=True)
                output, timing = timed_call(torch, pipe, _call_kwargs(torch, args, seed), args.device, prepare_kwargs=prepare)
                warmup["timing"] = timing
                _record_output(output, args, warmup, "warmup-{:03d}".format(index))
                warmup["status"] = "completed"
                del output
            for index in range(args.repeats):
                seed = args.seed + index
                run = {"index": index, "seed": seed, "status": "running", "measured": False}
                report["runs"].append(run)
                refresh = timed_cache_reset(torch, backend, args.device)
                run["uninstrumented"] = {"cache_refresh": refresh}
                print("GPU profile: timed run {} / {}, seed={}".format(index + 1, args.repeats, seed), file=sys.stderr, flush=True)
                output, timing = timed_call(torch, pipe, _call_kwargs(torch, args, seed), args.device, prepare_kwargs=prepare)
                timing["cache_refresh"] = refresh
                timing["pipeline_plus_refresh_ms"] = timing["e2e_ms"]
                run["uninstrumented"] = timing
                run["measured"] = report["measured"] = True
                _record_output(output, args, run, "run-{:03d}".format(index))
                run["cache_statistics"] = backend.stats()
                del output
                refresh = timed_cache_reset(torch, backend, args.device)
                kwargs = _call_kwargs(torch, args, seed)
                run["diagnostic"] = {"cache_refresh": refresh}
                print("GPU profile: diagnostic run {} / {}, seed={}".format(index + 1, args.repeats, seed), file=sys.stderr, flush=True)
                with StepObserver(torch, pipe) as observer:
                    kwargs.update(callback_on_step_end=observer.callback, callback_on_step_end_tensor_inputs=["latents"])
                    output, diagnostic_timing = timed_call(torch, pipe, kwargs, args.device, prepare_kwargs=prepare)
                diagnostic_timing["cache_refresh"] = refresh
                diagnostic_timing["pipeline_plus_refresh_ms"] = diagnostic_timing["e2e_ms"]
                run["diagnostic"].update(diagnostic_timing)
                _record_output(output, args, run["diagnostic"], "diagnostic-{:03d}".format(index))
                try:
                    run["diagnostic"].update(observer.report(args.steps))
                except ProfilingError as exc:
                    run["diagnostic"].update(exc.details.get("diagnostic", {}))
                    if exc.category.startswith("nonfinite"):
                        try:
                            _save_decoded_output(output, args, run["diagnostic"]["output"], "diagnostic-{:03d}".format(index), force=True)
                        except Exception as save_exc:
                            exc.details["artifact_save_error"] = str(save_exc)
                    raise
                run["diagnostic"]["cache_statistics"] = backend.stats()
                annotate_cache_steps(run["diagnostic"], run["diagnostic"]["cache_statistics"])
                run["instrumentation_output_exact_match"] = run["output"]["sha256"] == run["diagnostic"]["output"]["sha256"]
                del output
                run["status"] = "completed"
        report["aggregate"] = {"e2e_ms_median": statistics.median(r["uninstrumented"]["e2e_ms"] for r in report["runs"]),
                               "pipeline_plus_refresh_ms_median": statistics.median(r["uninstrumented"]["pipeline_plus_refresh_ms"] for r in report["runs"]),
                               "peak_allocated_bytes_max": max(r["uninstrumented"]["peak_allocated_bytes"] for r in report["runs"])}
        report["status"] = "completed"
        report["completed_runs"] = len(report["runs"])
    except Exception as exc:
        if not isinstance(exc, ProfilingError):
            exc = ProfilingError("{}: {}".format(type(exc).__name__, exc))
        exc.partial_report = report
        raise exc
    return report


def main(argv=None):
    parser = make_parser()
    args = normalize_args(parser.parse_args(argv), parser)
    if args.preflight:
        report = preflight(args)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if report["ready"] else 2
    try:
        # Keep library chatter separate from machine-readable files.
        report = run_profile(args)
        atomic_json(args.output, report)
    except Exception as exc:
        report = _failure_report(exc, args)
        try:
            atomic_json(args.output, json_safe(report))
        except Exception as write_exc:
            print("Could not preserve failure report: {}".format(write_exc), file=sys.stderr)
        print("GPU profiling failed: {}: {}. No synthetic measurements were produced.".format(type(exc).__name__, exc), file=sys.stderr)
        return 2
    print("Measured GPU profile written to {}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
