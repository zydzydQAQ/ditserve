"""Whole-request Wan profiling adapter; scheduling and ownership stay in C++.

Only the trace operation adds CPU annotations.  GPU kernels are identified by
the exported Kineto activity category, never by ``FunctionEvent.device_type``.
Importing this module does not import torch or initialize CUDA.
"""

from contextlib import contextmanager, nullcontext
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from ..base import Backend, file_hash, positive_int
from ..pipeline import _pipeline, _sdpa_math_context


PINNED_VERSIONS = {"torch": "2.10.0+cu126", "diffusers": "0.36.0", "transformers": "4.57.6"}


def _versions():
    actual = {name: importlib.metadata.version(name) for name in PINNED_VERSIONS}
    if actual != PINNED_VERSIONS:
        raise RuntimeError("Wan profiling requires pinned packages: " + json.dumps(actual, sort_keys=True))
    return actual


def _output_summary(value):
    """Hash only finite output, without saving media or duplicating its bytes."""
    import numpy as np
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not array.flags.c_contiguous:
        if array.dtype.kind not in "fiu":
            raise ValueError("Wan output must be a numeric array")
        array = np.ascontiguousarray(array)
    flat = array.reshape(-1)
    for offset in range(0, flat.size, 1024 * 1024):
        if not np.isfinite(flat[offset:offset + 1024 * 1024]).all():
            raise ValueError("Wan output contains nonfinite values")
    digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    return dict(sha256=digest, finite=True, shape=list(array.shape), dtype=str(array.dtype),
                scope="exact_output_only_not_perceptual_quality")


def _spec(config, request):
    spec = dict(config.get("spec", {}), **request.get("spec", {}))
    normalized = {name: positive_int(spec.get(name, default), name) for name, default in (
        ("frames", 17), ("height", 480), ("width", 832), ("steps", 32))}
    if normalized["frames"] % 4 != 1:
        raise ValueError("frames must be 1 modulo 4; implicit frame rounding is forbidden")
    if normalized["height"] % 16 or normalized["width"] % 16:
        raise ValueError("height and width must be divisible by 16")
    seed = spec.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    scale = spec.get("guidance_scale", 5.)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 0:
        raise ValueError("guidance_scale must be finite and nonnegative")
    for name, default in (("prompt", ""), ("negative_prompt", "")):
        value = spec.get(name, default)
        if not isinstance(value, str) or len(value) > 16384:
            raise ValueError(name + " must be a string of at most 16384 characters")
        normalized[name] = value
    normalized.update(seed=seed, guidance_scale=float(scale))
    return normalized


@contextmanager
def _patch_methods(pipe, torch, *, trace, hardware_frames=None):
    """Restore even inherited bound methods exactly when any forward fails."""
    patches = []

    def replace(owner, name, wrapper):
        own = name in vars(owner)
        previous = vars(owner).get(name)
        original = getattr(owner, name)
        setattr(owner, name, wrapper(original))
        patches.append((owner, name, own, previous))

    def scope(name):
        if trace:
            return torch.profiler.record_function("wan_profile/" + name)
        if hardware_frames is not None and name == "denoise":
            return torch.cuda.nvtx.range("wan_profile/denoise_%d" % hardware_frames)
        return nullcontext()

    def annotated(name, attention=False):
        def wrap(original):
            def invoke(*args, **kwargs):
                context = (torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION])
                           if attention else nullcontext())
                with scope(name), context:
                    return original(*args, **kwargs)
            return invoke
        return wrap

    try:
        replace(pipe.transformer, "forward", annotated("transformer", attention=True))
        if trace:
            for owner, method, label in ((pipe, "prepare_latents", "latent_init"),
                    (pipe.scheduler, "set_timesteps", "scheduler"),
                    (pipe.scheduler, "step", "scheduler"),
                    (pipe.vae, "decode", "vae"),
                    (pipe.video_processor, "postprocess_video", "postprocess")):
                replace(owner, method, annotated(label))
            for block in pipe.transformer.blocks:
                for name, label in (("attn1", "self_attention"), ("attn2", "cross_attention"),
                                    ("ffn", "ffn")):
                    replace(getattr(block, name), "forward", annotated(label))

        if trace or hardware_frames is not None:
            def progress_wrapper(original):
                def progress(*args, **kwargs):
                    bar = original(*args, **kwargs)
                    if "total" not in kwargs:
                        return bar

                    @contextmanager
                    def denoise():
                        with scope("denoise"), bar as entered:
                            yield entered
                    return denoise()
                return progress
            replace(pipe, "progress_bar", progress_wrapper)
        yield
    finally:
        for owner, name, own, previous in reversed(patches):
            if own:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)


class WanProfileBackend(Backend):
    operations = ("profile",)

    def __init__(self, config):
        super().__init__(config)
        self.pipe = None
        self.baselines = {}
        required_uid = config.get("required_uid")
        if required_uid is not None:
            if type(required_uid) is not int or required_uid <= 0:
                raise ValueError("required_uid must identify a non-root user")
            if os.geteuid() != required_uid or os.geteuid() == 0:
                raise PermissionError("profiling model must load as required_uid=%d, actual=%d" %
                                      (required_uid, os.geteuid()))
        self.versions = _versions()
        self.device = config.get("device", "cuda:0")
        if not str(self.device).startswith("cuda"):
            raise ValueError("Wan profiling requires the controller-owned CUDA device")
        model_path = Path(config["model_path"]).resolve()
        if not (model_path / "model_index.json").is_file():
            raise ValueError("local Wan snapshot missing")
        self.torch = importlib.import_module("torch")
        self.torch.set_num_threads(positive_int(config.get("cpu_threads", 8), "cpu_threads"))
        started = time.perf_counter()
        args = SimpleNamespace(model="wan", model_id=str(model_path), revision=config.get("revision"),
            dtype="float16", vae_dtype="float32", text_encoder_dtype="float32",
            text_encoder_device="cpu", device=self.device, local_files_only=True)
        self.pipe = _pipeline(self.torch, importlib.import_module("diffusers"), args)
        if type(self.pipe.transformer).__name__ != "WanTransformer3DModel":
            self.close()
            raise ValueError("unexpected Wan transformer")
        model_config = self.pipe.transformer.config
        expected = dict(num_layers=30, num_attention_heads=12, attention_head_dim=128,
                        in_channels=16, out_channels=16, ffn_dim=8960)
        if (any(getattr(model_config, key, None) != value for key, value in expected.items())
                or tuple(getattr(model_config, "patch_size", ())) != (1, 2, 2)
                or type(self.pipe.scheduler).__name__ != "UniPCMultistepScheduler"):
            self.close()
            raise ValueError("profiling protocol requires the Wan2.1-T2V-1.3B architecture and UniPC sampler")
        if getattr(self.pipe, "transformer_2", None) is not None:
            self.close()
            raise ValueError("this protocol requires the single-transformer Wan2.1 pipeline")
        blocks = self.pipe.transformer.blocks
        if len(blocks) != 30 or not all(callable(getattr(getattr(block, name, None), "forward", None))
                                        for block in blocks for name in ("attn1", "attn2", "ffn")):
            self.close()
            raise ValueError("Wan leaf instrumentation requires 30 blocks with attn1, attn2, and ffn")
        for component in (self.pipe.transformer, self.pipe.vae, self.pipe.text_encoder):
            component.eval()
        self.scheduler_class = type(self.pipe.scheduler)
        self.scheduler_config = dict(self.pipe.scheduler.config)
        self._reset_scheduler()
        self.torch.cuda.synchronize(self.device)
        self.model_load_ms = (time.perf_counter() - started) * 1000

    def _reset_scheduler(self):
        self.pipe.scheduler = self.scheduler_class.from_config(self.scheduler_config)

    def _memory(self):
        cuda = self.torch.cuda
        return dict(allocated_bytes=int(cuda.memory_allocated(self.device)),
                    reserved_bytes=int(cuda.memory_reserved(self.device)),
                    peak_allocated_bytes=int(cuda.max_memory_allocated(self.device)),
                    peak_reserved_bytes=int(cuda.max_memory_reserved(self.device)),
                    scope="whole_request_including_resident_model_weights")

    def _run(self, spec, *, trace, hardware=False):
        torch, pipe = self.torch, self.pipe
        def scope(name):
            if trace:
                return torch.profiler.record_function("wan_profile/" + name)
            if hardware and name == "request":
                return torch.cuda.nvtx.range("wan_profile/request_%d" % spec["frames"])
            return nullcontext()
        with torch.inference_mode(), _sdpa_math_context(torch), _patch_methods(pipe, torch, trace=trace,
                hardware_frames=spec["frames"] if hardware else None):
            with scope("request"):
                try:
                    generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
                    with scope("text_encode"):
                        positive, negative = pipe.encode_prompt(prompt=spec["prompt"],
                            negative_prompt=spec["negative_prompt"],
                            do_classifier_free_guidance=spec["guidance_scale"] > 1,
                            num_videos_per_prompt=1, device="cpu", dtype=torch.float32)
                    with scope("embedding_transfer"):
                        positive = positive.to(device=self.device, dtype=pipe.transformer.dtype)
                        if negative is not None:
                            negative = negative.to(device=self.device, dtype=pipe.transformer.dtype)
                    output = pipe(prompt_embeds=positive, negative_prompt_embeds=negative,
                        height=spec["height"], width=spec["width"], num_frames=spec["frames"],
                        num_inference_steps=spec["steps"], guidance_scale=spec["guidance_scale"],
                        num_videos_per_prompt=1, generator=generator, output_type="np", return_dict=False)[0]
                finally:
                    # Drain every stream before ending the CPU annotation,
                    # including a partial request. Never sync between steps.
                    torch.cuda.synchronize(self.device)
        return output

    def profile(self, request):
        if self.pipe is None:
            raise RuntimeError("Wan profiling backend is closed")
        mode = request.get("mode", request.get("spec", {}).get("mode", "clean"))
        if mode not in ("warmup", "clean", "trace", "hardware"):
            raise ValueError("profile mode must be warmup, clean, trace, or hardware")
        spec = _spec(self.config, request)
        key = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        if mode in ("trace", "hardware") and key not in self.baselines:
            raise ValueError(mode + " requires a successful clean request with identical inputs")
        result = dict(status="ok", mode=mode, spec=spec, artifacts=[], versions=self.versions,
                      model_load_ms=self.model_load_ms, hardware_counters="not_measured_by_torch_profiler")
        if mode == "hardware":
            result["nvtx_ranges"] = dict(request="wan_profile/request_%d" % spec["frames"],
                                         denoise="wan_profile/denoise_%d" % spec["frames"])
            result["hardware_counters"] = "external_collector_required_nvtx_annotations_only"
        torch = self.torch
        profiler = None
        output = None
        self._reset_scheduler()
        try:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            if mode == "trace":
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False,
                    with_stack=False, with_flops=False)
            try:
                with profiler if profiler is not None else nullcontext():
                    started = time.perf_counter()
                    output = self._run(spec, trace=mode == "trace", hardware=mode == "hardware")
                    result["e2e_ms"] = (time.perf_counter() - started) * 1000
                    result["memory"] = self._memory()
                result["output"] = _output_summary(output)
                expected_shape = [1, spec["frames"], spec["height"], spec["width"], 3]
                if result["output"]["shape"] != expected_shape:
                    raise ValueError("Wan output shape differs from the requested batch/frame/resolution")
                previous = self.baselines.get(key)
                if mode != "warmup":
                    if previous is None:
                        self.baselines[key] = result["output"]
                        result["numerical_validation"] = dict(status="baseline_recorded", passed=True)
                    else:
                        exact = previous["sha256"] == result["output"]["sha256"]
                        result["numerical_validation"] = dict(status="matched" if exact else "mismatch",
                            passed=exact, reference_sha256=previous["sha256"],
                            scope="identical_input_clean_output_hash_not_perceptual_quality")
                        if not exact:
                            result["status"] = "failed"
                else:
                    result["numerical_validation"] = dict(status="warmup_only", passed=None)
            except Exception as error:
                result.update(status="failed", error=dict(type=type(error).__name__, message=str(error)[:2000]))
            finally:
                # Export a failed request too: downstream validation owns this
                # evidence until the controller explicitly closes its dependency.
                if profiler is not None:
                    self._export_trace(profiler, request, result)
        finally:
            output = None
            self._reset_scheduler()
            gc.collect()
        return result

    def _export_trace(self, profiler, request, result):
        work = Path(self.config["work_dir"]).resolve()
        work.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix="wan-profile-", suffix=".json", dir=work)
        os.close(descriptor)
        path = Path(name)
        artifact = dict(kind="trace", path=str(path), summarized=False,
                        dependencies=["profile_validation"])
        result["artifacts"].append(artifact)
        try:
            profiler.export_chrome_trace(str(path))
            artifact.update(bytes=path.stat().st_size, sha256=file_hash(path))
            from ...evaluation.cuda_profile import summarize_trace
            result["profiler"] = summarize_trace(path)
            artifact["summarized"] = True
            # Bind the bounded aggregation to the exact temporary trace for
            # the common finalizer. Its dependency stays open until review.
            summary = result["profiler"]
            artifact["summary"] = dict(source_sha256=artifact["sha256"],
                source_bytes=artifact["bytes"],
                total_records=summary.get("diagnostics", {}).get("trace_event_count", 0),
                aggregation="Kineto category filter; linked CPU ancestry; interval unions; each real GPU activity once",
                omitted_fields=["per-event timestamps, arguments and CPU operator details",
                                "kernel names beyond top_kernels_retention.limit"],
                metrics=summary)
            forwards = result["spec"]["steps"] * (2 if result["spec"]["guidance_scale"] > 1 else 1)
            expected = {"transformer": forwards, **{name: forwards * len(self.pipe.transformer.blocks)
                        for name in ("self_attention", "cross_attention", "ffn")}}
            observed = {row["name"]: row.get("annotation_count", 0)
                        for row in result["profiler"].get("stages", []) if row.get("name") in expected}
            valid = expected == observed
            result["instrumentation_validation"] = dict(status="verified" if valid else "failed",
                passed=valid, expected_annotation_counts=expected, observed_annotation_counts=observed)
            if not valid:
                result["status"] = "failed"
            if result["profiler"].get("status") not in ("ok", "measured"):
                result["status"] = "failed"
        except Exception as error:
            result["status"] = "failed"
            result["profiler"] = dict(status="failed", error_type=type(error).__name__,
                                      error=str(error)[:2000])
            artifact.update(bytes=path.stat().st_size, sha256=file_hash(path))

    def close(self):
        torch = getattr(self, "torch", None)
        if self.pipe is not None:
            if torch is not None:
                torch.cuda.synchronize(self.device)
            self.pipe = None
            self.baselines.clear()
            gc.collect()
            if torch is not None:
                torch.cuda.empty_cache()
