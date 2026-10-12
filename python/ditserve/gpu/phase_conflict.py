"""Real, isolated V100 phase-residency experiments (stdlib-only at import time).

This is a mechanism experiment, not an online server or a goodput benchmark.
Call ``run_worker`` once in a fresh process.  The caller must enforce a hard
process timeout and must never reuse a failed worker, especially after CUDA OOM.
"""

from ..util import CONFIGS, atomic_json, utc_now as _utc_now, canonical_gpu_uuid

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace

from . import profiling as p


SCHEMA_VERSION = "ditserve.phase-conflict.v1"
STRATEGIES = ("immediate", "deferred", "blocking", "overlap")
EXPECTED_PACKAGES = {
    "torch": "2.10.0+cu126", "diffusers": "0.36.0", "transformers": "4.57.6",
    "accelerate": "1.12.0", "cache-dit": "1.5.0", "numpy": "2.2.6",
    "huggingface-hub": "0.36.2",
}


def _write_json(path, value):
    atomic_json(path, p.json_safe(value))


class Progress:
    """One shared writer; callbacks and the decoder thread only store metadata."""

    def __init__(self, directory, job_id):
        self.path = Path(directory) / "progress.json"
        self.lock = threading.Lock()
        self.state = {"id": job_id, "status": "running", "sequence": 0}

    def mark(self, stage, **fields):
        with self.lock:
            self.state.update(fields)
            self.state.update(stage=stage, updated_at_utc=_utc_now(),
                              updated_monotonic_ns=time.monotonic_ns(),
                              thread=threading.current_thread().name,
                              sequence=self.state["sequence"] + 1)
            _write_json(self.path, self.state)


def normalize_job(job):
    """Reject unsupported recipes before importing torch or touching CUDA."""
    if not isinstance(job, dict):
        raise ValueError("job must be a dictionary")
    result = dict(job)
    for key in ("id", "mode", "height", "width", "frames", "steps", "seed",
                "model_id", "revision", "output_dir"):
        if key not in result:
            raise ValueError("missing job field: " + key)
    if result["mode"] not in ("prepare", "case"):
        raise ValueError("mode must be prepare or case")
    if not isinstance(result["id"], str) or not result["id"]:
        raise ValueError("id must be a nonempty string")
    for key in ("height", "width", "frames", "steps", "seed"):
        if type(result[key]) is not int or result[key] < (0 if key == "seed" else 1):
            raise ValueError("invalid integer field: " + key)
    if result["height"] % 16 or result["width"] % 16 or (result["frames"] - 1) % 4:
        raise ValueError("Wan requires H/W multiples of 16 and frames=4*k+1")
    if result["seed"] >= 2 ** 63 - 2:
        raise ValueError("seed is out of range")
    protocol = json.loads((CONFIGS / "v100.json").read_text(encoding="utf-8"))
    fixed = {"preset": "v100-wan", "dtype": "float16", "vae_dtype": "float32",
             "text_encoder_dtype": "float32", "text_encoder_device": "cpu",
             "device": "cuda:0", "attention": "cache-dit-native",
             "guidance_scale": 5.0, "batch_size": 1, "warmup_steps": 4}
    for key, expected in fixed.items():
        if key in result and result[key] != expected:
            raise ValueError("fixed protocol mismatch: " + key)
        result[key] = expected
    if result["revision"] != protocol["revision"]:
        raise ValueError("checkpoint revision differs from the verified V100 recipe")
    for key in ("model_id", "output_dir"):
        if not Path(result[key]).is_absolute():
            raise ValueError(key + " must be an absolute local path")
        result[key] = str(Path(result[key]).resolve())
    result.setdefault("cache", "none")
    if result["cache"] not in ("none", "cache-dit"):
        raise ValueError("unsupported cache backend")
    result.setdefault("diagnostic", False)
    if type(result["diagnostic"]) is not bool:
        raise ValueError("diagnostic must be boolean")
    result.setdefault("strategy", "immediate")
    result.setdefault("insert_after", 25)
    if result["strategy"] not in STRATEGIES:
        raise ValueError("unsupported strategy")
    if type(result["insert_after"]) is not int or result["insert_after"] < 1:
        raise ValueError("insert_after is a positive one-based step")
    if (result["mode"] == "case" and result["strategy"] in ("blocking", "overlap")
            and result["insert_after"] >= result["steps"]):
        raise ValueError("insertion must precede the final step")
    if result["mode"] == "case":
        if not result.get("b_latent_path") or not Path(result["b_latent_path"]).is_absolute():
            raise ValueError("case requires an absolute b_latent_path")
        result["b_latent_path"] = str(Path(result["b_latent_path"]).resolve())
    if result.get("b_metadata_path"):
        result["b_metadata_path"] = str(Path(result["b_metadata_path"]).resolve())
    result["gpu_uuid"] = protocol["gpu_uuid"]
    result["cache_parameters"] = dict(protocol["cache"])
    result["prompt_id"] = "water" if result["mode"] == "prepare" else "object"
    result["prompt"] = next(x["text"] for x in protocol["prompts"] if x["id"] == result["prompt_id"])
    return result


def _profile_args(config, **changes):
    cache = config["cache_parameters"]
    values = dict(config)
    values.update(model="wan", local_files_only=True, preflight=False,
                  negative_prompt="", cache_fn=cache["fn"], cache_bn=cache["bn"],
                  cache_warmup=cache["warmup"], cache_threshold=cache["threshold"],
                  cache_max_continuous=cache["max_continuous"], warmup=1, repeats=1,
                  output=Path(config["output_dir"]) / "profile.json",
                  artifact_dir=Path(config["output_dir"]) / "artifacts")
    values.update(changes)
    return argparse.Namespace(**values)


def storage_inventory(groups, is_tensor):
    """Deduplicate actual storages across groups; never retain their tensors.

    Storage bytes include unused portions of views and are not allocator peaks.
    First-owner grouping is attribution bookkeeping, not additive causal cost.
    """
    seen_objects, seen_storages = set(), {}
    tensors, opaque = [], []

    def visit(value, path, owner):
        if id(value) in seen_objects:
            return
        if is_tensor(value):
            seen_objects.add(id(value))
            storage = value.untyped_storage()
            key = (str(value.device), int(storage.data_ptr()), int(storage.nbytes()))
            index = seen_storages.get(key)
            if index is None:
                index = len(seen_storages)
                seen_storages[key] = index
                stores.append({"index": index, "device": key[0], "data_ptr": key[1],
                               "bytes": key[2], "first_owner": owner})
            tensors.append({"path": path, "shape": list(value.shape), "dtype": str(value.dtype),
                            "device": str(value.device), "storage_index": index,
                            "storage_offset": int(value.storage_offset()),
                            "logical_bytes": int(value.numel() * value.element_size())})
        elif isinstance(value, dict):
            seen_objects.add(id(value))
            for key, child in value.items():
                visit(child, path + "." + str(key), owner)
        elif isinstance(value, (tuple, list)):
            seen_objects.add(id(value))
            for index, child in enumerate(value):
                visit(child, path + "[" + str(index) + "]", owner)
        elif value is not None and not isinstance(value, (str, int, bool, float)):
            opaque.append({"path": path, "type": type(value).__name__})

    stores = []
    for owner, value in groups.items():
        visit(value, owner, owner)
    return {"storages": stores, "tensors": tensors, "opaque_fields": opaque,
            "unique_storage_bytes": sum(x["bytes"] for x in stores),
            "cuda_storage_bytes": sum(x["bytes"] for x in stores if x["device"].startswith("cuda:")),
            "is_complete_checkpoint": False, "is_allocator_peak": False,
            "contains_live_tensor_references": False}


def _active_snapshot(torch, pipe, latent, callback_kwargs):
    manager = getattr(pipe, "_context_manager", None)
    contexts = getattr(manager, "_cached_context_manager", {})
    histories = []
    cache_buffers = {}
    for name, context in contexts.items():
        cache_buffers[name] = context.buffers
        histories.append({"name": name, "executed_steps": int(context.executed_steps),
                          "transformer_executed_steps": int(context.transformer_executed_steps),
                          "cached_steps": list(context.cached_steps),
                          "cfg_cached_steps": list(context.cfg_cached_steps)})
    scheduler = {key: value for key, value in vars(pipe.scheduler).items() if key != "_internal_dict"}
    groups = {"latent": latent, "scheduler": scheduler, "cache": cache_buffers,
              "embeddings": {key: callback_kwargs.get(key) for key in ("prompt_embeds", "negative_prompt_embeds")}}
    inventory = storage_inventory(groups, torch.is_tensor)
    return {"inventory": inventory, "cache_contexts": histories,
            "callback_stack_inventory_status": "unobserved; reading CPython frame locals can retain old tensors across steps",
            "stack_caveat": "callback suspends the actual Wan Python frame; its temporaries are not a minimal request checkpoint"}


def release_completed_denoiser_state(pipe):
    """Release only dead request state, never a suspended pipeline invocation."""
    if getattr(pipe, "_phase_conflict_active", False):
        raise p.ProfilingError("cannot release state of an active pipeline", category="active_state_cleanup")
    manager = getattr(pipe, "_context_manager", None)
    if manager is not None and manager.get_context() is not None:
        raise p.ProfilingError("cache context is still active", category="active_state_cleanup")
    names = list(getattr(manager, "_cached_context_manager", {}))
    if manager is not None:
        manager.clear_contexts()
    scheduler_type = type(pipe.scheduler)
    scheduler_config = dict(pipe.scheduler.config)
    pipe.scheduler = scheduler_type.from_config(scheduler_config)
    return {"status": "released_completed_denoiser_state", "cache_context_names": names,
            "scheduler_class": scheduler_type.__name__, "weights_reloaded": False}


def _vae_cache_empty(pipe):
    return all(value is None for name in ("_feat_map", "_enc_feat_map")
               for value in getattr(pipe.vae, name, []))


def decode_final_latent(torch, pipe, latent):
    """Exact arithmetic/order from diffusers 0.36 WanPipeline's decode tail."""
    latent = latent.to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
    inverse_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
    normalized = latent / inverse_std + mean
    video = pipe.vae.decode(normalized, return_dict=False)[0]
    array = pipe.video_processor.postprocess_video(video, output_type="np")
    del video, normalized, mean, inverse_std
    if not _vae_cache_empty(pipe):
        raise p.ProfilingError("VAE retained temporal feature caches after decode", category="vae_state")
    return array


class VaeWorker:
    """A single warmed VAE owner; main-thread A VAE runs only after join."""

    def __init__(self, torch, pipe, progress):
        self.torch, self.pipe, self.progress = torch, pipe, progress
        self.stream = torch.cuda.Stream(device="cuda:0")
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream("cuda:0"))
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="phase-vae")
        self.future = None
        self.cuda_events = {}

    def submit(self, latent, boundary, record, label):
        if self.future is not None and not self.future.done():
            raise p.ProfilingError("VAE already has an active decode", category="vae_concurrency")
        latent.record_stream(self.stream)
        record.update(label=label, host_submitted_monotonic_ns=time.monotonic_ns(),
                      stream=int(self.stream.cuda_stream), status="submitted")
        self.future = self.executor.submit(self._execute, latent, boundary, record, label)
        return self.future

    def _execute(self, latent, boundary, record, label):
        torch = self.torch
        try:
            with torch.cuda.device("cuda:0"), torch.inference_mode(), torch.cuda.stream(self.stream):
                # SDPA settings are process-global in this pinned torch release.
                # The owner holds MATH until this thread has joined, including errors.
                if (not torch.backends.cuda.math_sdp_enabled() or torch.backends.cuda.flash_sdp_enabled()
                        or torch.backends.cuda.mem_efficient_sdp_enabled() or torch.backends.cuda.cudnn_sdp_enabled()):
                    raise p.ProfilingError("VAE thread escaped global SDPA MATH policy", category="attention_policy")
                self.stream.wait_event(self.ready)
                self.stream.wait_event(boundary)
                start, finish = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record(self.stream)
                record.update(status="running", host_started_monotonic_ns=time.monotonic_ns())
                self.progress.mark("b_decode_started", decode_label=label)
                array = decode_final_latent(torch, self.pipe, latent)
                finish.record(self.stream)
                finish.synchronize()  # never synchronize unrelated A work
                del latent
                if not bool(importlib.import_module("numpy").isfinite(array).all()):
                    _save_output(array, self.progress.path.parent, "failure-" + label, record)
                self.cuda_events[label] = (start, finish)
                record.update(status="completed", host_completed_monotonic_ns=time.monotonic_ns(),
                              completed_perf_ns=time.perf_counter_ns(),
                              cuda_interval_ms=float(start.elapsed_time(finish)),
                              vae_cache_empty=True)
                self.progress.mark("b_decode_completed", decode_label=label)
                return array
        except BaseException as exc:
            record.update(status="failed", error=str(exc), host_failed_monotonic_ns=time.monotonic_ns())
            self.progress.mark("b_decode_failed", decode_label=label, error=str(exc))
            raise

    def close(self):
        # CUDA work cannot be safely cancelled by Future.cancel(). A hard process
        # watchdog is the caller's responsibility if a CUDA operation never returns.
        self.executor.shutdown(wait=True)

    def add_relative_events(self, origin, label, record):
        if label in self.cuda_events:
            start, finish = self.cuda_events[label]
            record.update(cuda_start_ms=float(origin.elapsed_time(start)),
                          cuda_finished_ms=float(origin.elapsed_time(finish)))


class ForwardObserver:
    def __init__(self, torch, pipe):
        self.torch, self.pipe = torch, pipe
        self.events, self.pending, self.handles = [], [], []

    def __enter__(self):
        try:
            self.handles.append(self.pipe.transformer.register_forward_pre_hook(self._before))
            self.handles.append(self.pipe.transformer.register_forward_hook(self._after))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _before(self, module, inputs):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        self.pending.append(event)

    def _after(self, module, inputs, output):
        end = self.torch.cuda.Event(enable_timing=True)
        end.record()
        self.events.append((self.pending.pop(), end))

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def report(self, origin):
        return [{"forward_index": index, "cuda_start_ms": float(origin.elapsed_time(start)),
                 "cuda_end_ms": float(origin.elapsed_time(end)), "cuda_ms": float(start.elapsed_time(end))}
                for index, (start, end) in enumerate(self.events)]


class PendingDecode:
    """Single ownership avoids retaining B's GPU latent after output is ready."""

    def __init__(self, latent, worker, record):
        self.latent, self.worker, self.record = latent, worker, record
        self.future = None

    def launch(self, boundary, label="measured-b"):
        if self.future is not None or self.latent is None:
            raise p.ProfilingError("B decode must be launched exactly once", category="duplicate_decode")
        self.future = self.worker.submit(self.latent, boundary, self.record, label)
        self.latent = None

    def check_failure(self):
        if self.future is not None and self.future.done():
            error = self.future.exception()
            if error is not None:
                raise error

    def join(self):
        if self.future is None:
            raise p.ProfilingError("B decode has not been launched", category="missing_decode")
        return self.future.result()


def _memory(torch):
    return {"allocated_bytes": int(torch.cuda.memory_allocated("cuda:0")),
            "reserved_bytes": int(torch.cuda.memory_reserved("cuda:0")),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda:0")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda:0"))}


class InsertDecodeCallback:
    def __init__(self, torch, config, pipe, pending, forwards, events, progress):
        self.torch, self.config, self.pipe = torch, config, pipe
        self.pending, self.forwards, self.events, self.progress = pending, forwards, events, progress
        self.finite_checks = []
        self.boundaries = []

    def __call__(self, pipe, index, timestep, kwargs):
        completed = int(index) + 1
        if int(index) != len(self.events["steps"]):
            raise p.ProfilingError("non-sequential Wan callback", category="step_contract")
        self.pending.check_failure()
        step = {"step_index": int(index), "completed_step": completed,
                "forward_count": len(self.forwards.events), "host_monotonic_ns": time.monotonic_ns()}
        self.events["steps"].append(step)
        self.events.update(step_count=len(self.events["steps"]), forward_count=len(self.forwards.events))
        self.progress.mark("a_step_completed", a_completed_steps=completed)
        if self.config["diagnostic"] or completed == self.config["steps"]:
            self.finite_checks.append((completed, self.torch.isfinite(kwargs["latents"]).all()))
        if completed in (self.config["insert_after"], self.config["steps"]):
            snapshot = _active_snapshot(self.torch, pipe, kwargs["latents"], kwargs)
            snapshot["completed_step"] = completed
            snapshot["allocator"] = _memory(self.torch)
            self.events.setdefault("residency_snapshots", []).append(snapshot)
        if completed == self.config["insert_after"] and self.config["strategy"] in ("blocking", "overlap"):
            boundary = self.torch.cuda.Event(enable_timing=True)
            boundary.record(self.torch.cuda.current_stream("cuda:0"))
            self.boundaries.append((completed, boundary))
            self.pending.record["boundary_step"] = completed
            self.pending.launch(boundary)
            if self.config["strategy"] == "blocking":
                self.pending.join()
        if completed == self.config["steps"]:
            if self.config["strategy"] != "deferred":
                self.pending.join()
            for number, finite in self.finite_checks:
                if not bool(finite.item()):
                    raise p.ProfilingError("NONFINITE_LATENTS at A step " + str(number), category="nonfinite_latents")
            self.events["before_a_vae"] = dict(_memory(self.torch), host_monotonic_ns=time.monotonic_ns())
            self.progress.mark("a_before_own_vae", a_completed_steps=completed)
        return kwargs


def _pipeline_call(torch, pipe, args, output_type="np", callback=None):
    kwargs = p._call_kwargs(torch, args, args.seed)
    kwargs["output_type"] = output_type
    if callback is not None:
        kwargs.update(callback_on_step_end=callback,
                      callback_on_step_end_tensor_inputs=["latents", "prompt_embeds", "negative_prompt_embeds"])
    kwargs, encode = p._prepare_prompt_kwargs(torch, pipe, args, kwargs)
    pipe._phase_conflict_active = True
    try:
        output = pipe(**kwargs)
    finally:
        pipe._phase_conflict_active = False
    return output, encode


def _save_output(array, directory, label, record, save=True):
    """Record even a nonfinite decoded array, then fail the current worker."""
    output = SimpleNamespace(frames=array)
    try:
        record.update(p._fingerprint(output, "wan"))
    except p.ProfilingError as exc:
        record.update(exc.details.get("output", {}))
        save = True
        error = exc
    else:
        error = None
    if save:
        path = Path(directory) / "artifacts" / (label + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        importlib.import_module("numpy").save(str(path), array)
        record.update(artifact=str(path), path=str(path))
    if error is not None:
        raise error


def _latent_fingerprint(torch, latent):
    if not torch.is_tensor(latent) or latent.dtype != torch.float32 or latent.ndim != 5:
        raise p.ProfilingError("expected a rank-5 FP32 final Wan latent", category="latent_contract")
    cpu = latent.detach().to(device="cpu", dtype=torch.float32)
    array = cpu.numpy()
    finite = bool(importlib.import_module("numpy").isfinite(array).all())
    return {"shape": list(array.shape), "dtype": str(array.dtype), "finite": finite,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}, cpu


def _warmup(torch, pipe, args, decoder, report, progress):
    progress.mark("warmup_4_steps")
    warm_args = argparse.Namespace(**dict(vars(args), steps=4, seed=args.seed + 1))
    capture = {}

    def last_latent(_pipe, index, timestep, kwargs):
        if index == 3:
            capture["latent"] = kwargs["latents"].detach().clone()
        return kwargs

    started = time.perf_counter_ns()
    output, encode = _pipeline_call(torch, pipe, warm_args, callback=last_latent)
    official = {}
    _save_output(output.frames, args.output.parent, "warmup-official", official, save=False)
    release_completed_denoiser_state(pipe)
    boundary = torch.cuda.Event()
    boundary.record(torch.cuda.current_stream("cuda:0"))
    record = {}
    future = decoder.submit(capture.pop("latent"), boundary, record, "warmup")
    array = future.result()
    direct = {}
    _save_output(array, args.output.parent, "warmup-direct", direct, save=False)
    report["warmup"] = {"steps": 4, "seed": warm_args.seed, "encode": encode,
                        "official": official, "direct": direct, "decoder": record,
                        "wall_ms": (time.perf_counter_ns() - started) / 1e6,
                        "exact_match": official["sha256"] == direct["sha256"]}
    if not report["warmup"]["exact_match"]:
        _save_output(output.frames, args.output.parent, "warmup-official", official)
        _save_output(array, args.output.parent, "warmup-direct", direct)
        raise p.ProfilingError("warmup direct VAE differs from official Wan tail", category="output_mismatch")
    del output, array, future
    torch.cuda.synchronize("cuda:0")
    progress.mark("warmup_completed")


def _prepare(torch, pipe, args, config, decoder, backend, report, progress):
    progress.mark("prepare_b_denoise")
    start = time.perf_counter_ns()
    kwargs = p._call_kwargs(torch, args, args.seed)
    kwargs["output_type"] = "latent"
    kwargs, encode = p._prepare_prompt_kwargs(torch, pipe, args, kwargs)
    torch.cuda.synchronize("cuda:0")
    torch.cuda.reset_peak_memory_stats("cuda:0")
    denoise = {"initial_allocated_bytes": int(torch.cuda.memory_allocated("cuda:0")),
               "initial_reserved_bytes": int(torch.cuda.memory_reserved("cuda:0")),
               "scope": "prepared-embedding pipeline call with output_type=latent; includes sampler/latent preparation, excludes encoder and VAE"}
    report["phase_profiles"] = {"denoise": denoise}
    report["measured"] = True
    denoise_started = time.perf_counter_ns()
    pipe._phase_conflict_active = True
    try:
        output = pipe(**kwargs)
    finally:
        pipe._phase_conflict_active = False
    latent = output.frames
    del output
    torch.cuda.synchronize("cuda:0")
    denoise.update(_memory(torch), wall_ms=(time.perf_counter_ns() - denoise_started) / 1e6)
    del kwargs  # release prepared embeddings before the standalone decode stage
    report["b_cache_statistics"] = backend.stats()
    report["b_terminal_cleanup"] = release_completed_denoiser_state(pipe)
    latent_record, cpu = _latent_fingerprint(torch, latent)
    latent_path = Path(config["output_dir"]) / "latent.pt"
    torch.save(cpu, str(latent_path))
    del cpu
    report.update(latent=latent_record, b_latent_path=str(latent_path))
    if not latent_record["finite"]:
        raise p.ProfilingError("NONFINITE_LATENTS in prepared B", category="nonfinite_latents")
    boundary = torch.cuda.Event()
    boundary.record(torch.cuda.current_stream("cuda:0"))
    torch.cuda.synchronize("cuda:0")
    torch.cuda.reset_peak_memory_stats("cuda:0")
    decode = report["phase_profiles"]["decode"] = {
        "initial_allocated_bytes": int(torch.cuda.memory_allocated("cuda:0")),
        "initial_reserved_bytes": int(torch.cuda.memory_reserved("cuda:0")),
        "scope": "standalone B VAE on dedicated thread/stream, including inverse normalization and CPU output materialization"}
    decode_started = time.perf_counter_ns()
    decode_record = report["events"]["b_decode"]
    future = decoder.submit(latent, boundary, decode_record, "prepare-b")
    del latent
    array = future.result()
    decode.update(_memory(torch), wall_ms=(time.perf_counter_ns() - decode_started) / 1e6)
    report["memory"].update({field: max(denoise[field], decode[field])
                             for field in ("peak_allocated_bytes", "peak_reserved_bytes")})
    report["memory"]["scope"] = "max of measured denoise/decode segment peaks; warmup and subsequent equivalence run excluded"
    report["prepare_ms"] = (time.perf_counter_ns() - start) / 1e6
    report["timings"].update(remaining_ms=report["prepare_ms"], a_completed_ms=None,
                             b_completed_ms=report["prepare_ms"], **encode)
    _save_output(array, config["output_dir"], "output-b", report["output_b"])
    if config["diagnostic"]:
        progress.mark("prepare_official_pipeline_equivalence", verification_steps=args.steps)
        official, unused = _pipeline_call(torch, pipe, args)
        record = report["official_pipeline_verification"] = {"steps": args.steps, "seed": args.seed}
        _save_output(official.frames, config["output_dir"], "b-official-reference", record)
        record["exact_match"] = record["sha256"] == report["output_b"]["sha256"]
        release_completed_denoiser_state(pipe)
        if not record["exact_match"]:
            raise p.ProfilingError("latent-only plus direct decode differs from full Wan pipeline", category="output_mismatch")
        del official
    provenance = {"schema_version": SCHEMA_VERSION, "job": report["job"], "config": config,
                  "model_id": config["model_id"], "revision": config["revision"],
                  "latent": latent_record, "b_latent_path": str(latent_path),
                  "output_b": report["output_b"], "prepare_ms": report["prepare_ms"],
                  "phase_profiles": report["phase_profiles"],
                  "official_pipeline_verification": report.get("official_pipeline_verification"),
                  "warmup_direct_decode_exact_match": report["warmup"]["exact_match"]}
    metadata_path = Path(config["output_dir"]) / "provenance.json"
    _write_json(metadata_path, provenance)
    report["b_metadata_path"] = str(metadata_path)


def _load_b(torch, config, report):
    path = Path(config["b_latent_path"])
    cpu = torch.load(str(path), map_location="cpu", weights_only=True)
    latent_record, cpu = _latent_fingerprint(torch, cpu)
    if not latent_record["finite"]:
        raise p.ProfilingError("NONFINITE_LATENTS in B fixture", category="nonfinite_latents")
    source = {"latent_path": str(path), "latent": latent_record, "metadata_path": config.get("b_metadata_path")}
    if config.get("b_metadata_path"):
        metadata = json.loads(Path(config["b_metadata_path"]).read_text(encoding="utf-8"))
        expected = metadata.get("latent", {})
        if any(expected.get(key) != latent_record[key] for key in ("sha256", "shape", "dtype", "finite")):
            raise p.ProfilingError("B latent differs from its provenance", category="provenance")
        for key in ("model_id", "revision", "dtype", "vae_dtype", "text_encoder_dtype"):
            if metadata.get("config", {}).get(key) != config[key]:
                raise p.ProfilingError("B provenance recipe mismatch: " + key, category="provenance")
        for key in ("height", "width", "frames", "steps", "seed", "cache", "attention", "text_encoder_device"):
            wanted = config.get("b_" + key, config[key])
            if metadata.get("config", {}).get(key) != wanted:
                raise p.ProfilingError("B provenance identity mismatch: " + key, category="provenance")
        b_config = metadata["config"]
        shape = [1, 16, (b_config["frames"] - 1) // 4 + 1,
                 b_config["height"] // 8, b_config["width"] // 8]
        if latent_record["shape"] != shape:
            raise p.ProfilingError("B latent shape does not match its request identity", category="provenance")
        source.update(config=metadata["config"], output_b=metadata["output_b"], prepare_ms=metadata.get("prepare_ms"))
    else:
        shape = [1, 16, (config.get("b_frames", config["frames"]) - 1) // 4 + 1,
                 config.get("b_height", config["height"]) // 8,
                 config.get("b_width", config["width"]) // 8]
        if latent_record["shape"] != shape:
            raise p.ProfilingError("B latent shape differs from expected B shape", category="latent_contract")
    report["b_source"] = source
    return cpu.to(device="cuda:0", dtype=torch.float32)


def _case(torch, pipe, args, config, decoder, backend, report, progress):
    progress.mark("load_b_fixture")
    pending = PendingDecode(_load_b(torch, config, report), decoder, report["events"]["b_decode"])
    torch.cuda.synchronize("cuda:0")
    ready = torch.cuda.Event(enable_timing=True)
    ready.record(torch.cuda.current_stream("cuda:0"))
    torch.cuda.reset_peak_memory_stats("cuda:0")  # exactly once in this measured strategy
    report["memory"]["initial_allocated_bytes"] = int(torch.cuda.memory_allocated("cuda:0"))
    origin = torch.cuda.Event(enable_timing=True)
    origin.record(torch.cuda.current_stream("cuda:0"))
    started_ns = time.perf_counter_ns()
    report["timings"]["started_at_unix_ns"] = time.time_ns()
    report["events"]["a_stream"] = int(torch.cuda.current_stream("cuda:0").cuda_stream)
    report["events"]["b_stream"] = int(decoder.stream.cuda_stream)
    report["measured"] = True
    progress.mark("b_ready_strategy_started", strategy=config["strategy"])
    if config["strategy"] == "immediate":
        pending.launch(origin)
        pending.join()
    forwards = ForwardObserver(torch, pipe)
    callback = InsertDecodeCallback(torch, config, pipe, pending, forwards, report["events"], progress)
    with forwards:
        output, encode = _pipeline_call(torch, pipe, args, callback=callback)
    a_completed_ns = time.perf_counter_ns()
    report["a_cache_statistics"] = backend.stats()
    report["a_terminal_cleanup"] = release_completed_denoiser_state(pipe)
    if config["strategy"] == "deferred":
        boundary = torch.cuda.Event(enable_timing=True)
        boundary.record(torch.cuda.current_stream("cuda:0"))
        pending.launch(boundary)
    array_b = pending.join()
    torch.cuda.synchronize("cuda:0")
    finished_ns = time.perf_counter_ns()
    report["timings"].update(remaining_ms=(finished_ns - started_ns) / 1e6,
                             a_completed_ms=(a_completed_ns - started_ns) / 1e6,
                             b_completed_ms=(pending.record["completed_perf_ns"] - started_ns) / 1e6,
                             finished_at_unix_ns=time.time_ns(), **encode)
    report["memory"].update(_memory(torch))
    before = report["events"].get("before_a_vae", {})
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        report["memory"]["before_a_vae_" + field] = before.get(field)
    report["events"].update(forward_count=len(forwards.events), step_count=len(report["events"]["steps"]),
                            dit_forwards=forwards.report(origin))
    decoder.add_relative_events(origin, "measured-b", pending.record)
    report["events"]["insertion_boundaries"] = [
        {"completed_step": number, "cuda_ms": float(origin.elapsed_time(event))}
        for number, event in callback.boundaries]
    _save_output(output.frames, config["output_dir"], "output-a", report["output_a"])
    _save_output(array_b, config["output_dir"], "output-b", report["output_b"])
    if len(forwards.events) != 2 * config["steps"] or len(report["events"]["steps"]) != config["steps"]:
        raise p.ProfilingError("A did not execute its complete CFG/step sequence", category="step_contract")
    reference = report.get("b_source", {}).get("output_b")
    if reference:
        report["b_output_exact_match"] = report["output_b"]["sha256"] == reference["sha256"]
        if not report["b_output_exact_match"]:
            raise p.ProfilingError("B decode differs from standalone prepared output", category="output_mismatch")


def _failure(report, exc, progress_state=None):
    category = getattr(exc, "category", "worker_error")
    if "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError":
        category = "cuda_oom"
    original_progress = (report.get("failure") or {}).get("last_progress")
    if original_progress is None:
        original_progress = {key: value for key, value in (progress_state or {}).items() if key != "failure"}
    report.update(status="failed", failure={"category": category, "type": type(exc).__name__,
                  "message": str(exc), "details": p.json_safe(getattr(exc, "details", {})),
                  "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                  "last_progress": p.json_safe(original_progress),
                  "measured_at_failure": bool(report.get("measured"))})


def run_worker(job):
    """Run one real job; failures are persisted and require process termination."""
    directory = Path(job.get("output_dir", ""))
    if not directory.is_absolute():
        raise ValueError("output_dir must be absolute")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "profile.json").exists():
        raise ValueError("refusing to overwrite an existing worker profile")
    report = {"schema_version": SCHEMA_VERSION, "kind": "real_gpu_phase_conflict",
              "status": "running", "measured": False, "job": p.json_safe(job),
              "config": {}, "created_at_utc": _utc_now(), "failure": None,
              "output_a": {}, "output_b": {}, "timings": {}, "memory": {},
              "events": {"steps": [], "b_decode": {}},
              "scope": {"online_goodput_measured": False, "perceptual_quality_validated": False,
                        "timing_origin": "case B-ready before A prompt encoding; prepare reported separately",
                        "memory": "joint process allocator peaks; no per-stage causal subtraction",
                        "cuda_intervals_prove_kernel_concurrency": False,
                        "state": "actual suspended Wan call frame, not a minimal checkpoint",
                        "process_reuse_after_failure": "forbidden"}}
    progress = Progress(directory, job.get("id"))
    decoder = None
    worker_start = time.perf_counter_ns()
    try:
        config = normalize_job(job)
        report["config"] = config
        progress.mark("preflight")
        for name, expected in {"CUDA_VISIBLE_DEVICES": config["gpu_uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                               "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}.items():
            if os.environ.get(name) != expected:
                raise p.ProfilingError("worker environment mismatch: " + name, category="environment")
        args = _profile_args(config)
        environment = report["environment"] = p.preflight(args)
        if not environment["ready"]:
            raise p.ProfilingError("; ".join(environment["errors"]), category="preflight")
        uuid = canonical_gpu_uuid(environment.get("gpu", {}).get("uuid"))
        if uuid is None or uuid != canonical_gpu_uuid(config["gpu_uuid"]):
            raise p.ProfilingError("actual GPU UUID differs from target", category="environment")
        if environment["packages"] != EXPECTED_PACKAGES:
            raise p.ProfilingError("dependencies differ from pinned V100 suite", category="environment")
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        torch.cuda.set_device("cuda:0")
        progress.mark("load_models")
        pipe = p._pipeline(torch, diffusers, args)
        if str(pipe.device) != "cuda:0" or str(pipe._execution_device) != "cuda:0":
            raise p.ProfilingError("pipeline execution device mismatch", category="environment")
        if getattr(pipe.vae, "_hf_hook", None) is not None or pipe.vae.use_tiling or pipe.vae.use_slicing:
            raise p.ProfilingError("VAE hooks/tiling/slicing differ from fixed recipe", category="vae_state")
        backend = p.CacheBackend(pipe, args)
        report["components"] = p._component_metadata(pipe)
        report["revision"] = p._revision_metadata(args, report["components"])
        report["cache_config"] = p.json_safe(backend.config)
        report["scheduler"] = {"class": type(pipe.scheduler).__name__, "config": dict(pipe.scheduler.config)}
        # This global MATH guard exits only after the decoder thread stops.
        with torch.inference_mode(), p._sdpa_math_context(torch):
            decoder = VaeWorker(torch, pipe, progress)
            try:
                _warmup(torch, pipe, args, decoder, report, progress)
                if config["mode"] == "prepare":
                    _prepare(torch, pipe, args, config, decoder, backend, report, progress)
                else:
                    _case(torch, pipe, args, config, decoder, backend, report, progress)
            except BaseException as exc:
                _failure(report, exc, progress.state)
                try:
                    report["memory"]["observed_at_failure"] = _memory(torch)
                except Exception as memory_exc:
                    report["memory"]["failure_observation_error"] = str(memory_exc)
                _write_json(directory / "profile.json", report)
                progress.mark("failed_waiting_for_worker_exit", status="failed", failure=report["failure"])
                raise
            finally:
                decoder.close()
        report["status"] = "completed"
        progress.mark("completed", status="completed")
    except BaseException as exc:
        _failure(report, exc, progress.state)
        if "torch" in locals() and "observed_at_failure" not in report["memory"]:
            try:
                report["memory"]["observed_at_failure"] = _memory(torch)
            except Exception as memory_exc:
                report["memory"]["failure_observation_error"] = str(memory_exc)
        progress.mark("failed", status="failed", failure=report["failure"])
    finally:
        report["finished_at_utc"] = _utc_now()
        report["worker_total_ms"] = (time.perf_counter_ns() - worker_start) / 1e6
        _write_json(directory / "profile.json", report)
    return report
