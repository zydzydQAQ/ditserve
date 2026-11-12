"""Run a single request through the real official xDiT Wan model runner.

The only backend adaptation changes the model instance's checkpoint location
from its registered HF identifier to a fully hashed offline local snapshot.
This is an atomic native diagnostic, not the DitServe scheduler or a serving
benchmark. Quick mode changes only the step count to one, explicitly recorded.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import subprocess
import threading
import time
import traceback

class source:
    __file__ = __file__
    COMMIT = "16c34e7f070de65d9d2c67a6d74e0007dd8b426a"
    REPOSITORY = "xdit-project/xDiT"
    ARCHIVE_SHA256 = "c09e48fa4792290b299ab3cfc0afc9c11ce60ac24affee933b1a0ce146e52f5b"
    MODEL_IDS = {
        "wan21": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "wan22": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "sd35": "stabilityai/stable-diffusion-3.5-medium",
    }
    DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / "configs/benchmarks/formal-mixed-v1.json"

    @staticmethod
    def digest(raw):
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def canonical(value):
        return source.digest(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":"), allow_nan=False).encode())

    @staticmethod
    def read_json(raw):
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate JSON key: " + key)
                value[key] = item
            return value
        def nonfinite(value):
            raise ValueError("nonfinite JSON value: " + value)
        return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)

    @staticmethod
    def verify_source(source_root, manifest_path, manifest_sha256):
        root = Path(source_root)
        if root.is_symlink():
            raise ValueError("symlink source root")
        root = root.resolve(strict=True)
        raw = Path(manifest_path).read_bytes()
        if not re.fullmatch("[0-9a-f]{64}", manifest_sha256) or source.digest(raw) != manifest_sha256:
            raise ValueError("official source manifest SHA256 differs")
        manifest = source.read_json(raw)
        if (manifest.get("schema") != "ditserve.official-source-archive.v1"
                or manifest.get("repository") != source.REPOSITORY
                or manifest.get("commit") != source.COMMIT
                or manifest.get("archive_sha256") != source.ARCHIVE_SHA256):
            raise ValueError("official source identity differs")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("official source inventory is empty")
        for relative, expected in files.items():
            parts = PurePosixPath(relative)
            if (parts.is_absolute() or ".." in parts.parts or str(parts) != relative
                    or not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected)):
                raise ValueError("invalid source inventory entry: " + relative)
            path = root / relative
            if any(root.joinpath(*parts.parts[:n]).is_symlink() for n in range(1, len(parts.parts) + 1)):
                raise ValueError("symlink in source: " + relative)
            if not path.is_file() or source.digest(path.read_bytes()) != expected:
                raise ValueError("official source file SHA256 differs: " + relative)
        actual = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("symlink in source: " + str(path))
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
        if actual != set(files):
            raise ValueError("official source inventory differs (extra or missing files)")
        return manifest

DEFAULT_MODELS = Path("/shared/data/DITSERVE/models/formal-manifests/20261103-inputs-v1/models.official-all.json")
DEFAULT_MODELS_SHA256 = "3e612738984ae56ac369f48b5a207c94d31cd1071d6506c633692ed9ccce37be"
CLASSES = {"wan21": "xFuserWan21T2V1_3BModel", "wan22": "xFuserWan22TI2VModel"}
REVISIONS = {"wan21": "0fad780a534b6463e45facd96134c9f345acfa5b", "wan22": "b8fff7315c768468a5333511427288870b2e9635"}


def check_storage():
    """Stop only this native process at the project's shared-storage hard line."""
    process = subprocess.run(["df", "-BG", "/shared"], capture_output=True, text=True, timeout=20)
    try:
        row = process.stdout.splitlines()[-1].split()
        used, percent = int(row[2].rstrip("G")), int(row[4].rstrip("%"))
        if process.returncode or used > 800 or percent >= 80:
            raise ValueError("shared-storage stop threshold reached")
    except (IndexError, ValueError) as exc:
        raise ValueError("shared-storage check failed or crossed stop threshold") from exc
    return {"used_gib": used, "percent": percent}


def start_storage_guard():
    initial = check_storage()
    stop = threading.Event()
    def watch():
        while not stop.wait(30):
            try:
                check_storage()
            except Exception as exc:
                print("native xDiT storage guard stopped this process: " + str(exc), file=sys.stderr, flush=True)
                os._exit(75)
    threading.Thread(target=watch, name="native-xdit-storage-guard", daemon=True).start()
    return stop, initial


def bound_json(path, expected):
    raw = Path(path).read_bytes()
    if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected) or source.digest(raw) != expected:
        raise ValueError("bound JSON SHA256 differs: " + str(path))
    return source.read_json(raw)


def fingerprint(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def verify_model(model, model_root, bindings_path, bindings_sha256):
    if model not in CLASSES:
        raise ValueError("only the two registered Wan models are supported")
    binding = bound_json(bindings_path, bindings_sha256)[model]
    root = Path(model_root).resolve(strict=True)
    if binding.get("model_path") and root != Path(binding["model_path"]).resolve(strict=True):
        raise ValueError("checkpoint root differs from frozen model binding")
    manifest_path = Path(binding["path"])
    manifest = bound_json(manifest_path, binding["sha256"])
    if (manifest.get("schema") != "ditserve.model-manifest.v1"
            or manifest.get("model") != source.MODEL_IDS[model]
            or manifest.get("revision") != REVISIONS[model]
            or manifest.get("component_paths") != {"*": model}):
        raise ValueError("official checkpoint identity or component mapping differs")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty model file inventory")
    seen, total = {}, 0
    for row in rows:
        name = row.get("path", "")
        rel = PurePosixPath(name)
        if (not name or rel.is_absolute() or ".." in rel.parts or str(rel) != name
                or "\\" in name or name in seen):
            raise ValueError("invalid or duplicate model file path")
        if (type(row.get("bytes")) is not int or row["bytes"] < 0
                or not isinstance(row.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
            raise ValueError("invalid model size or SHA256")
        path = root / name
        if not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("model file escapes snapshot")
        before = fingerprint(path)
        if before[2] != row["bytes"]:
            raise ValueError("model file size differs: " + name)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != row["sha256"] or fingerprint(path) != before:
            raise ValueError("model file changed or SHA256 differs: " + name)
        seen[name] = before
        total += row["bytes"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(seen) or any(fingerprint(root/n) != f for n, f in seen.items()):
        raise ValueError("model inventory incomplete or changed during verification")
    index = source.read_json((root/"model_index.json").read_bytes())
    config = source.read_json((root/"transformer/config.json").read_bytes())
    scheduler = source.read_json((root/"scheduler/scheduler_config.json").read_bytes())
    if index.get("_class_name") != "WanPipeline" or index.get("transformer_2") not in (None, [None, None]):
        raise ValueError("expected a single-transformer WanPipeline checkpoint")
    if config.get("num_attention_heads") != (12 if model == "wan21" else 24):
        raise ValueError("checkpoint transformer head count differs")
    flow = scheduler.get("flow_shift")
    if isinstance(flow, bool) or not isinstance(flow, (float, int)) or not math.isfinite(flow) or flow <= 0:
        raise ValueError("checkpoint scheduler flow_shift is missing or invalid")
    return {"model_id": source.MODEL_IDS[model], "root": str(root),
            "binding_sha256": bindings_sha256, "manifest_sha256": binding["sha256"],
            "manifest_path": str(manifest_path), "revision": manifest["revision"],
            "verified_files": len(seen), "verified_bytes": total,
            "model_index_sha256": source.digest((root/"model_index.json").read_bytes()),
            "flow_shift": flow, "snapshot_fingerprint_sha256": source.canonical(seen)}


def request_plan(protocol, model, shape_id, mode, prompt, seed):
    if protocol.get("schema") != "ditserve.formal-protocol.v1" or protocol.get("id") != "formal-mixed-v1":
        raise ValueError("formal protocol identity differs")
    if model not in CLASSES or mode not in ("quick", "formal"):
        raise ValueError("unsupported model or execution mode")
    shape = protocol["shapes"][shape_id]
    if shape["model"] != model or shape["kind"] != "video":
        raise ValueError("shape does not belong to selected model")
    if mode == "formal" and shape["scope"] != "formal":
        raise ValueError("formal mode requires a formal protocol shape")
    if not isinstance(prompt, str) or not prompt.strip() or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("nonempty prompt and integer seed in [0, 2^63) required")
    return {"mode": mode, "shape_id": shape_id, "shape_scope": shape["scope"],
            "height": shape["height"], "width": shape["width"], "num_frames": shape["frames"],
            "num_inference_steps": 1 if mode == "quick" else shape["steps"],
            "protocol_steps": shape["steps"], "guidance_scale": shape["guidance_scale"],
            "negative_prompt": "", "prompt": prompt, "seed": seed,
            "changed_protocol_parameters": {"steps": {"protocol": shape["steps"], "executed": 1}} if mode == "quick" else {},
            "num_outputs": 1}


def runner_config(model, request, identity, output_dir, cpu_offload=False):
    config = {k: request[k] for k in ("height", "width", "num_frames", "num_inference_steps",
                                    "guidance_scale", "negative_prompt", "prompt", "seed")}
    # Match the official CLI default: the dataclass alone defaults to None,
    # but both native Wan preprocessors require an iterable image collection.
    config.update(model=source.MODEL_IDS[model], task="t2v" if model == "wan22" else None,
                  input_images=[], flow_shift=identity["flow_shift"], output_directory=str(output_dir),
                  ulysses_degree=1, ring_degree=1, data_parallel_degree=1,
                  tensor_parallel_degree=1, pipefusion_parallel_degree=1, fully_shard_degree=1,
                  use_cfg_parallel=False, use_torch_compile=False, cache_method=None,
                  use_teacache=False, use_fbcache=False, use_fp8_gemms=False,
                  use_fp8_comms=False, enable_model_cpu_offload=bool(cpu_offload),
                  enable_tiling=False, enable_slicing=False, num_iterations=1,
                  warmup_calls=0, batch_size=0, determinism_check=0)
    return config


def check_effective_request(request, effective):
    for key in ("height", "width", "num_frames", "num_inference_steps", "guidance_scale", "negative_prompt", "seed"):
        if effective.get(key) != request[key]:
            raise ValueError("official preprocessing changed requested parameter: " + key)
    # Official preprocessing normalizes a scalar prompt to a one-element list.
    if effective.get("prompt") not in (request["prompt"], [request["prompt"]]):
        raise ValueError("official preprocessing changed requested prompt")


def summarize_video(videos, request):
    import numpy as np
    # Diffusers produces one video, as a frame array or a list of PIL frames.
    if len(videos) != 1:
        raise ValueError("native pipeline did not produce exactly one video")
    frames = videos[0]
    if len(frames) != request["num_frames"]:
        raise ValueError("native pipeline output frame count differs")
    h, minimum, maximum, dtype = hashlib.sha256(), math.inf, -math.inf, None
    shape = [request["num_frames"], request["height"], request["width"], 3]
    for frame in frames:
        array = np.asarray(frame)
        if list(array.shape) != shape[1:] or not np.issubdtype(array.dtype, np.number):
            raise ValueError("native pipeline output geometry or dtype differs")
        if not np.isfinite(array).all():
            raise ValueError("native pipeline output has nonfinite values")
        current_dtype = str(array.dtype)
        if dtype is None:
            dtype = current_dtype
            h.update(json.dumps({"shape": shape, "dtype": dtype}, sort_keys=True).encode())
        elif dtype != current_dtype:
            raise ValueError("native output has inconsistent frame dtypes")
        h.update(np.ascontiguousarray(array).tobytes())
        minimum, maximum = min(minimum, float(array.min())), max(maximum, float(array.max()))
    return {"outputs": 1, "shape": shape, "dtype": dtype, "sha256": h.hexdigest(),
            "finite": True, "minimum": minimum, "maximum": maximum,
            "finite_scope": "returned decoded frames after pipeline postprocessing",
            "floating_point_frames": bool(np.issubdtype(np.dtype(dtype), np.floating)),
            "latent_finite_checked": False, "pre_postprocess_finite_checked": False,
            "pre_quantization_finite_checked": False,
            "retained_arrays": False, "quality_evaluated": False}



def attention_observation(pipeline, requested):
    """Inspect existing objects only; never import/select/resolve an attention kernel.

    The pinned runtime selector can route individual calls to a fallback. Its
    post-request value, processor classes and cached callable are evidence of
    runtime state, not a trace proving which GPU kernel served every call.
    """
    import functools

    def label(value):
        if value is None:
            return "unknown"
        name = getattr(value, "name", None)
        text = name if isinstance(name, str) else str(value)
        return text if text.strip() else "unknown"

    def symbol(value):
        if value is None:
            return None
        partial = isinstance(value, functools.partial)
        function = value.func if partial else value
        module_name = getattr(function, "__module__", type(function).__module__)
        module = sys.modules.get(module_name)
        return {"module": module_name,
                "qualname": getattr(function, "__qualname__", type(function).__qualname__),
                "module_file": getattr(module, "__file__", None), "partial": partial}

    result = {"requested_attention_backend": None if requested is None else label(requested),
        "runtime_selector_after_request": "unknown", "cross_attention_selector_after_request": "unknown",
        "runtime_state_class": None, "hybrid_schedule_enabled": None,
        "selected_registry_entries": {}, "transformer_processors": [], "observation_errors": [],
        "kernel_dispatch_verified": False,
        "scope": "post-request runtime selector and existing processor/cached callable inspection; loaded modules or cached fallback functions do not prove per-call kernel dispatch",
        "source_evidence": "loaded_source_sha256 records verified loaded xfuser files"}
    runtime_module = sys.modules.get("xfuser.core.distributed.runtime_state")
    registry = sys.modules.get("xfuser.core.attention.registry")
    try:
        # The pinned getter only returns _RUNTIME and asserts it exists.
        state = runtime_module.get_runtime_state() if runtime_module is not None else None
        if state is not None:
            result["runtime_state_class"] = symbol(type(state))
            selected = getattr(state, "attention_backend", None)
            cross = getattr(state, "cross_attention_backend", None)
            cross = selected if cross is None else cross
            result["runtime_selector_after_request"] = label(selected)
            result["cross_attention_selector_after_request"] = label(cross)
            result["hybrid_schedule_enabled"] = getattr(getattr(state, "runtime_config", None),
                                                         "use_hybrid_attn_schedule", None)
            for role, backend in (("self", selected), ("cross", cross)):
                entry = registry.find(backend) if registry is not None and backend is not None else None
                if entry is not None:
                    impl = getattr(entry, "impl", None)
                    result["selected_registry_entries"][role] = {
                        "backend": label(backend), "registered_impl_class": symbol(type(impl)),
                        "registered_target": getattr(impl, "target", None),
                        "registered_package": getattr(entry, "package", None),
                        "cached_callable": symbol(getattr(entry, "_resolved", None)),
                        "declared_fallback": None if getattr(entry, "fallback", None) is None else label(entry.fallback)}
    except Exception as exc:
        result["observation_errors"].append("runtime selector inspection: " + type(exc).__name__ + ": " + str(exc))
    try:
        processors = {}
        for _, layer in pipeline.transformer.named_modules():
            processor = getattr(layer, "processor", None)
            if processor is None:
                continue
            row = {"class": symbol(type(processor)),
                   "attention_function": symbol(getattr(processor, "attention_function", None)),
                   "is_cross_attention": getattr(processor, "is_cross_attention", None)}
            key = json.dumps(row, sort_keys=True)
            if key not in processors:
                processors[key] = dict(row, count=0)
            processors[key]["count"] += 1
        result["transformer_processors"] = [processors[key] for key in sorted(processors)]
    except Exception as exc:
        result["observation_errors"].append("processor inspection: " + type(exc).__name__ + ": " + str(exc))
    return result


def loaded_sources(source_root, manifest):
    root = Path(source_root).resolve()
    files = {}
    for name, module in sorted(sys.modules.items()):
        if name != "xfuser" and not name.startswith("xfuser."):
            continue
        origin = getattr(module, "__file__", None)
        if origin:
            path = Path(origin).resolve()
            if not path.is_relative_to(root):
                raise ValueError("xfuser module escaped official tree: " + name)
            relative = path.relative_to(root).as_posix()
            actual = source.digest(path.read_bytes())
            if manifest["files"].get(relative) != actual:
                raise ValueError("loaded xfuser module is not in verified inventory: " + name)
            files[relative] = actual
        else:
            locations = list(getattr(module, "__path__", []))
            if not locations or any(not Path(p).resolve().is_relative_to(root/"xfuser") for p in locations):
                raise ValueError("xfuser namespace escaped official tree: " + name)
    return files


def execute_native(model, request, identity, config, source_root, manifest, capture_context=None):
    if any(n == "xfuser" or n.startswith("xfuser.") for n in sys.modules):
        raise ValueError("native run requires a fresh interpreter without loaded xfuser")
    sys.path.insert(0, str(Path(source_root).resolve()))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    import torch
    from xfuser.runner import xFuserModelRunner
    if torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("native initial diagnostic requires exactly one allocated GPU and rank")
    loaded_sources(source_root, manifest)
    expected_class = CLASSES[model]

    class LocalCheckpointRunner(xFuserModelRunner):
        def _select_model(self, model_name, upstream_config):
            instance = super()._select_model(model_name, upstream_config)
            if type(instance).__name__ != expected_class or instance.settings.model_name != identity["model_id"]:
                raise ValueError("official runner selected an unexpected model or checkpoint")
            instance.settings.model_name = identity["root"]
            return instance

    runner, initialized = None, False
    result, execution_error, phase = None, None, "construct_runner"
    start = time.monotonic()
    try:
        runner = LocalCheckpointRunner(config)
        phase = "preprocess_request"
        effective = runner.preprocess_args(dict(vars(runner.config)))
        check_effective_request(request, effective)
        phase = "load_model"
        runner.initialize(effective)
        initialized = True
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - start
        pipeline = runner.model.pipe
        if type(pipeline).__name__ != "WanPipeline" or getattr(pipeline, "transformer_2", None) is not None:
            raise ValueError("expected native single-transformer WanPipeline")
        components = {name: {"class": type(component).__name__, "dtype": str(getattr(component, "dtype", None)),
                              "device": str(getattr(component, "device", None))}
                      for name, component in pipeline.components.items()}
        if str(pipeline.transformer.dtype) != "torch.bfloat16":
            raise ValueError("official transformer dtype differs from declared bfloat16")
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        phase = "native_inference"
        output, timings = runner.run(effective)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        phase = "validate_output"
        if len(timings) != 1 or any(not math.isfinite(float(t)) or float(t) < 0 for t in timings):
            raise ValueError("native runner returned invalid single-request timing")
        summary = summarize_video(output.videos, request)
        artifact = None
        if capture_context is not None:
            phase = "retain_decoded_output"
            artifact = capture_context.get("capture", lambda *a: None)(output.videos, summary, capture_context)
            summary["retained_arrays"] = True
        attention = attention_observation(pipeline, runner.config.attention_backend)
        result = {"status": "native_request_completed", "completed_requests": 1, "requested_requests": 1,
            "output": summary, "load_seconds": load_seconds, "cold_request_wall_seconds": elapsed,
            "official_runner_timings_seconds": [float(t) for t in timings],
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "pipeline_class": type(pipeline).__name__, "components": components,
            "scheduler_config": dict(pipeline.scheduler.config),
            "effective_request": {k: effective[k] for k in request if k in effective},
            "requested_attention_backend": attention["requested_attention_backend"],
            "attention_backend_observation": attention,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "loaded_source_sha256": loaded_sources(source_root, manifest)}
        if artifact is not None:
            result["decoded_artifact"] = artifact
            result["performance_eligible"] = False
        del output
        return result
    except Exception as exc:
        execution_error = exc
        exc.native_stage = phase
        raise
    finally:
        try:
            if initialized:
                runner.cleanup()
            elif torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as cleanup_error:
            if execution_error is not None:
                execution_error.native_cleanup_error = str(cleanup_error)
            else:
                cleanup_error.native_stage = "cleanup"
                cleanup_error.native_completed_result = result
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--models-sha256", default=DEFAULT_MODELS_SHA256)
    parser.add_argument("--model", choices=list(CLASSES), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=source.DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--mode", choices=["quick", "formal"], required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-offload", action="store_true", help="explicitly enable upstream model CPU offload; changes performance conditions")
    parser.add_argument("--numerics-contract", type=Path)
    parser.add_argument("--numerics-contract-sha256")
    args = parser.parse_args(argv)
    if bool(args.numerics_contract) != bool(args.numerics_contract_sha256):
        parser.error("capture requires both a contract and its SHA256")
    if args.numerics_contract and args.output_dir.resolve() != args.output_dir:
        parser.error("capture output path must be absolute without symlink traversal")
    result = {"schema": "ditserve.external-xdit-native.v1", "status": "not_started",
              "qualification_eligible": False, "allows": [], "serving_executed": False,
              "quality_evaluated": False, "formal_benchmark_executed": False,
              "requested_requests": 1, "completed_requests": 0}
    output = args.output_dir.resolve()
    created_output = False
    storage_guard = None
    phase = "verify_inputs"
    try:
        if args.command == "run":
            storage_guard, result["initial_storage"] = start_storage_guard()
        manifest = source.verify_source(args.source_root, args.source_manifest, args.source_manifest_sha256)
        protocol = bound_json(args.protocol, args.protocol_sha256)
        request = request_plan(protocol, args.model, args.shape, args.mode, args.prompt, args.seed)
        phase = "verify_model_files"
        identity = verify_model(args.model, args.model_root, args.models, args.models_sha256)
        config = runner_config(args.model, request, identity, output, args.cpu_offload)
        result.update(status="planned", request=request, model_identity=identity, runner_config=config,
            source={"commit": source.COMMIT, "manifest_sha256": args.source_manifest_sha256,
                    "archive_sha256": source.ARCHIVE_SHA256, "files": len(manifest["files"])},
            adapter={"sha256": source.digest(Path(__file__).read_bytes()),
                     "source_verifier_sha256": source.digest(Path(source.__file__).read_bytes()),
                     "override": "LocalCheckpointRunner._select_model changes only instance.settings.model_name to verified local root"},
            protocol_sha256=args.protocol_sha256, native_dtype="bfloat16", shared_backend_comparison=False,
            measurement_scope="one cold native request; no arrival replay, batching, baseline comparison, or quality qualification")
        capture_context = None
        if args.numerics_contract:
            raise ValueError("standalone numerics capture script has been retired")
        if args.command == "run":
            if not output.is_relative_to(Path("/shared/output/DITSERVE")):
                raise ValueError("GPU outputs must be under /shared/output/DITSERVE")
            output.mkdir(parents=True, exist_ok=False)
            created_output = True
            with (output/"started.json").open("x") as stream:
                json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            # Actual native logs go to stderr; stdout stays machine-readable JSON.
            phase = "execute_native"
            with contextlib.redirect_stdout(sys.stderr):
                result.update(execute_native(args.model, request, identity, config, args.source_root, manifest, capture_context))
            phase = "reverify_source"
            source.verify_source(args.source_root, args.source_manifest, args.source_manifest_sha256)
            result["runtime_packages"] = {}
            for name in ("torch", "diffusers", "transformers", "cache-dit", "numpy"):
                try: result["runtime_packages"][name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError: result["runtime_packages"][name] = None
    except Exception as exc:
        if getattr(exc, "native_completed_result", None):
            result.update(exc.native_completed_result)
        result["failed_stage"] = getattr(exc, "native_stage", phase)
        if getattr(exc, "native_cleanup_error", None):
            result["cleanup_error"] = exc.native_cleanup_error
        result.update(status="native_request_failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-10000:])
    if storage_guard is not None:
        storage_guard.set()
    if created_output:
        check_storage()
        with (output/"summary.json").open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["status"] in ("planned", "native_request_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
