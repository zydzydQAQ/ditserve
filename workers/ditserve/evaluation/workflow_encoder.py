"""Isolated CPU/GPU UMT5 placement experiments for the pinned Wan worker.

This module changes no production defaults.  Run it with the runtime image's
Python and the checkout on PYTHONPATH.  Each selected variant runs in a fresh
process, so an OOM cannot poison the next variant.  T0 must precede other
variants to produce numerical references; missing references are reported,
never silently treated as numerical passes.

Example::

    python -m ditserve.evaluation.workflow_encoder --model-path /models/wan2.1-t2v-1.3b \
      --output /data/workflow/encoder --phases T0,T1,T2 --device cuda:1

The GPU FP32 variant loads *only* the tokenizer and text encoder.  It does not
attempt to co-reside the 21 GiB encoder with the DiT/VAE.  T3/T4 intentionally
exclude Sana.  T5 explicitly stages embeddings through host memory, matching
this machine's disabled GPU peer access.  Prompt embeddings and video outputs
are never reused across requests to manufacture a throughput improvement.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import MethodType, SimpleNamespace


PHASES = {
    "T0": {"encoder_dtype": "float32", "placement": "cpu", "video": True},
    "T1": {"encoder_dtype": "float32", "placement": "gpu_only", "video": False},
    "T2": {"encoder_dtype": "bfloat16", "placement": "gpu_only", "video": False},
    "T3": {"encoder_dtype": "bfloat16", "placement": "resident", "video": True},
    "T4": {"encoder_dtype": "bfloat16", "placement": "staged", "video": True},
    "T5": {"encoder_dtype": "bfloat16", "placement": "dedicated", "video": True},
}
PROMPTS = (
    {"id": "english", "prompt": "A small red boat moves slowly across a calm lake, mountains in the background.",
     "negative_prompt": ""},
    {"id": "chinese", "prompt": "一艘红色小船缓缓驶过平静的湖面，远处是群山，镜头平稳移动。",
     "negative_prompt": ""},
    {"id": "truncation", "prompt": "A red boat on a calm lake. " * 600,
     "negative_prompt": ""},
    {"id": "negative", "prompt": "A small red boat moves slowly across a calm lake, mountains in the background.",
     "negative_prompt": "blurry, distorted, low quality, oversaturated"},
)


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def compare_arrays(actual, expected):
    """Bounded-memory numerical comparison, including a useful zero case."""
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    result = {"shape": list(actual.shape), "reference_shape": list(expected.shape),
              "shape_match": actual.shape == expected.shape,
              "finite": bool(np.isfinite(actual).all()),
              "reference_finite": bool(np.isfinite(expected).all())}
    if not result["shape_match"] or not result["finite"] or not result["reference_finite"]:
        result.update(max_abs=None, rmse=None, cosine=None, exact=False)
        return result
    flat_a, flat_b = actual.reshape(-1), expected.reshape(-1)
    maximum, squares, dot, norm_a, norm_b = 0.0, 0.0, 0.0, 0.0, 0.0
    for start in range(0, flat_a.size, 262144):
        a = flat_a[start:start + 262144].astype(np.float64)
        b = flat_b[start:start + 262144].astype(np.float64)
        difference = a - b
        maximum = max(maximum, float(np.abs(difference).max(initial=0)))
        squares += float(np.dot(difference, difference))
        dot += float(np.dot(a, b))
        norm_a += float(np.dot(a, a))
        norm_b += float(np.dot(b, b))
    denominator = math.sqrt(norm_a * norm_b)
    cosine = dot / denominator if denominator else (1.0 if norm_a == norm_b else 0.0)
    result.update(max_abs=maximum, rmse=math.sqrt(squares / max(1, flat_a.size)),
                  cosine=min(1.0, max(-1.0, cosine)), exact=maximum == 0)
    return result


def _array_evidence(value):
    """Small numeric evidence independent of raw-array retention."""
    import numpy as np
    value = np.ascontiguousarray(value)
    raw = memoryview(value).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(raw), 8 * 1024 * 1024):
        digest.update(raw[start:start + 8 * 1024 * 1024])
    finite = bool(np.isfinite(value).all())
    return dict(shape=list(value.shape), dtype=str(value.dtype), finite=finite, array_sha256=digest.hexdigest(),
                statistics=dict(min=float(value.min()), max=float(value.max()), mean=float(value.mean(dtype=np.float64)))
                if finite and value.size else None)


def _save_encoder_outputs(directory, latent, video, *, save_videos=False):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "latent.npy", latent, allow_pickle=False)
    artifacts = dict(latent=str(directory / "latent.npy"))
    evidence = dict(_array_evidence(video), retained=save_videos)
    if save_videos:
        np.save(directory / "video.npy", video, allow_pickle=False)
        artifacts["video"] = str(directory / "video.npy")
    return artifacts, evidence


def _video_reference_metadata(row):
    """Legacy records imply FP32 from the fixed Wan NumPy output recipe."""
    shape = [1, row["frames"], 480, 832, 3]
    existing = row.get("video", {})
    digest = row.get("sha256", {}).get("video")
    if (row.get("finite") is not True or row.get("shape") != shape
            or existing.get("dtype", "float32") != "float32"
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError("invalid original T0 video shape/dtype/hash evidence")
    if existing and (existing.get("shape") != shape or existing.get("array_sha256") != digest
                     or existing.get("finite") is not True):
        raise ValueError("conflicting original T0 video evidence")
    return dict(shape=shape, dtype="float32", finite=True, array_sha256=digest,
                dtype_source="record" if existing else "fixed_Wan_numpy_recipe")


def _verify_video_reference(video, expected):
    actual = _array_evidence(video)
    if any(actual[name] != expected[name] for name in ("shape", "dtype", "finite", "array_sha256")):
        raise ValueError("T0 video reference reconstruction differs in shape/dtype/hash; numerical comparison refused")
    return actual


def _reference_video(worker, row, latent, path, recorder, context):
    """Recover a reference only after exact binding to its original evidence."""
    import numpy as np
    expected = _video_reference_metadata(row)
    if path.is_file():
        video = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        if (str(latent.dtype) != "float32" or not np.isfinite(latent).all()
                or list(latent.shape) != [1, 16, (row["frames"] - 1) // 4 + 1, 60, 104]
                or _array_evidence(latent)["array_sha256"] != row["sha256"]["latent"]):
            raise ValueError("T0 latent reference differs from recorded shape/dtype/hash")
        metadata = dict(context, purpose="numerical_reference_reconstruction", include_in_performance=False)
        # This runs after the measured request's timer and uses no request state.
        with recorder.phase("reference_latent_h2d", **metadata):
            state = SimpleNamespace(latents=worker.torch.from_numpy(np.array(latent, copy=True)).to(worker.device))
        class ReferenceRecorder:
            def phase(self, name, **fields):
                return recorder.phase("numerical_reference_" + name, **fields)
        try:
            video = _decode_separate(worker, state, ReferenceRecorder(), metadata)
        finally:
            del state
    _verify_video_reference(video, expected)
    return video


def _tensor_array(value):
    return None if value is None else value.detach().float().cpu().numpy()


def _embedding_arrays(pair):
    return {name: _tensor_array(value) for name, value in zip(("prompt", "negative"), pair)
            if value is not None}


def _compare_embeddings(arrays, reference):
    import numpy as np
    if not Path(reference).is_file():
        return {"status": "reference_missing", "reference": str(reference)}
    with np.load(reference, allow_pickle=False) as expected:
        if set(arrays) != set(expected.files):
            return {"status": "key_mismatch", "actual": sorted(arrays), "reference": sorted(expected.files)}
        result = {key: compare_arrays(value, expected[key]) for key, value in arrays.items()}
        # DiT consumes FP16 embeddings even when the encoder is FP32/BF16.
        downstream = {key: compare_arrays(value.astype(np.float16), expected[key].astype(np.float16))
                      for key, value in arrays.items()}
    return {"status": "compared", "encoder_output": result, "dit_fp16_input": downstream}


def _instrument_prompt_helper(pipe, recorder, metadata):
    """Pinned diffusers 0.36 helper arithmetic with separate stage ranges.

    Source: diffusers/pipelines/wan/pipeline_wan.py::_get_t5_prompt_embeds.
    Caption cleaning, padding, attention mask and repeat order are unchanged.
    """
    import torch
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    def helper(self, prompt=None, num_videos_per_prompt=1, max_sequence_length=226,
               device=None, dtype=None):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        with recorder.phase("tokenize", **metadata):
            prompt = [prompt_clean(value) for value in prompt]
            batch_size = len(prompt)
            inputs = self.tokenizer(prompt, padding="max_length", max_length=max_sequence_length,
                                    truncation=True, add_special_tokens=True,
                                    return_attention_mask=True, return_tensors="pt")
            ids, mask = inputs.input_ids, inputs.attention_mask
            lengths = mask.gt(0).sum(dim=1).long()
        with recorder.phase("text_token_h2d" if str(device).startswith("cuda") else "text_token_cpu", **metadata):
            ids_device, mask_device = ids.to(device), mask.to(device)
        with recorder.phase("text_encoder", **metadata):
            embeddings = self.text_encoder(ids_device, mask_device).last_hidden_state
        with recorder.phase("text_embedding_pad", **metadata):
            embeddings = embeddings.to(dtype=dtype, device=device)
            embeddings = [value[:length] for value, length in zip(embeddings, lengths)]
            embeddings = torch.stack([torch.cat([value, value.new_zeros(max_sequence_length - value.size(0),
                                                                        value.size(1))])
                                      for value in embeddings], dim=0)
            _, sequence_length, _ = embeddings.shape
            embeddings = embeddings.repeat(1, num_videos_per_prompt, 1)
            embeddings = embeddings.view(batch_size * num_videos_per_prompt, sequence_length, -1)
        return embeddings

    pipe._get_t5_prompt_embeds = MethodType(helper, pipe)


def _experimental_worker(device, model_path, attention):
    """Relax only experiment payload placement, retaining state ownership checks."""
    from ..models.wan.runtime import WanStepWorker

    class ExperimentalWanWorker(WanStepWorker):
        def _validate_cpu_embeddings(self, prompt, negative, spec):
            if prompt is None or (spec["guidance_scale"] > 1 and negative is None):
                raise ValueError("missing experiment prompt embeddings")
            for tensor in (prompt, negative):
                if tensor is None:
                    continue
                if not self.torch.is_tensor(tensor) or tensor.dtype not in (self.torch.float32, self.torch.bfloat16):
                    raise ValueError("experiment payload must be an FP32/BF16 tensor")
                if not bool(self.torch.isfinite(tensor).all()):
                    raise RuntimeError("nonfinite experiment prompt embeddings")

    return ExperimentalWanWorker(device=device, model_path=model_path, attention=attention)


def _standalone_encoder(model_path, dtype, device):
    import torch
    import diffusers
    import transformers
    from ..models.wan.runtime import REQUIRED_VERSIONS
    import importlib.metadata
    for package, expected in REQUIRED_VERSIONS.items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError("unsupported pinned runtime version: " + package)
    encoder = transformers.UMT5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=getattr(torch, dtype), local_files_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
    # Wan supports absent VAE/transformers for prompt encoding.  No DiT/VAE
    # weights are loaded here, avoiding accidental GPU FP32 capacity failures.
    pipe = diffusers.WanPipeline(tokenizer=tokenizer, text_encoder=encoder,
                                 vae=None, scheduler=None, transformer=None)
    pipe.text_encoder.to(device=device, dtype=getattr(torch, dtype)).eval()
    return pipe


def _spec(case, frames, steps, request_id):
    return {"request_id": request_id, "prompt": case["prompt"],
            "negative_prompt": case["negative_prompt"], "seed": 20261019,
            "height": 480, "width": 832, "frames": frames, "steps": steps, "guidance_scale": 5.0}


def _decode_separate(worker, state, recorder, metadata):
    """Same arithmetic as decode_final_latent, with D2H separated from VAE."""
    torch, pipe = worker.torch, worker.pipe
    with torch.inference_mode(), worker.p._sdpa_math_context(torch):
        with recorder.phase("vae_prepare", **metadata):
            latent = state.latents.to(pipe.vae.dtype)
            mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
            inverse_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
            normalized = latent / inverse_std + mean
        with recorder.phase("vae_decode", **metadata):
            video = pipe.vae.decode(normalized, return_dict=False)[0]
        # The pinned processor combines device-side denormalize and D2H.  Keep
        # this label honest; profiler memcpy intervals provide the separation.
        with recorder.phase("video_postprocess_d2h", **metadata):
            array = pipe.video_processor.postprocess_video(video, output_type="np")
        state.finished = True
        if any(value is not None for name in ("_feat_map", "_enc_feat_map")
               for value in getattr(pipe.vae, name, [])):
            raise RuntimeError("VAE retained temporal caches after decode")
    return array


def _dedicated_queue(worker, encode, recorder, args):
    """Paired same-hardware T5 serial/prefetch queues, after stage warmup.

    The encoder thread synchronizes only GPU 0; the Wan thread synchronizes
    only GPU 1.  The result hashes are compared across both schedules.  This
    does not claim that a two-GPU dedicated topology beats two independent Wan
    workers: that whole-machine comparison belongs to the E2 experiment.
    """
    from ..models.wan.runtime import EncodedPrompt
    rows = []
    hashes = {}
    for frames in args.frames:
        for repeat in range(args.queue_repeats):
            order = ("serial", "prefetch") if repeat % 2 == 0 else ("prefetch", "serial")
            for mode in order:
                specs = [_spec(PROMPTS[0], frames, args.steps,
                               "T5-q-%s-f%d-r%d-i%d" % (mode, frames, repeat, index))
                         for index in range(args.pipeline_requests)]
                for index, spec in enumerate(specs):
                    spec["seed"] += index
                started = time.perf_counter_ns()
                completed, outputs, latent_outputs = [], [], []
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-text-encoder") as pool:
                    future = (pool.submit(encode, PROMPTS[0], specs[0]["request_id"], "dedicated_queue", repeat)
                              if mode == "prefetch" else None)
                    for index, spec in enumerate(specs):
                        pair, encode_ms = (future.result() if future is not None else
                                           encode(PROMPTS[0], spec["request_id"], "dedicated_queue", repeat))
                        if mode == "prefetch" and index + 1 < len(specs):
                            future = pool.submit(encode, PROMPTS[0], specs[index + 1]["request_id"],
                                                 "dedicated_queue", repeat)
                        else:
                            future = None
                        payload = EncodedPrompt(spec=spec, prompt_embeds=pair[0], negative_prompt_embeds=pair[1],
                                                cpu_prompt_encode_ms=0.0, owner=worker._owner)
                        context = {"variant": "T5", "request_id": spec["request_id"], "purpose": "dedicated_queue",
                                   "mode": mode, "frames": frames, "repeat": repeat}
                        with recorder.phase("wan_prepare_input_h2d_latent", **context):
                            state = worker.prepare_encoded(spec, payload)
                        del payload, pair
                        for step in range(args.steps):
                            with recorder.phase("dit_step", step=step, **context):
                                worker.step(state)
                        video = _decode_separate(worker, state, recorder, context)
                        finished_ns = time.perf_counter_ns()
                        completed.append({"index": index, "encode_ms": encode_ms,
                                          "completed_after_ms": (finished_ns - started) / 1e6})
                        outputs.append(hashlib.sha256(video.tobytes()).hexdigest())
                        # Numerical checks follow the request completion stamp;
                        # their transfer/hash cost remains in total queue wall time.
                        latent = _tensor_array(state.latents)
                        latent_outputs.append(hashlib.sha256(latent.tobytes()).hexdigest())
                        del latent
                        worker.release(state)
                        del state, video
                wall_ms = (time.perf_counter_ns() - started) / 1e6
                key = (frames, repeat)
                previous = hashes.get(key)
                current_hashes={"video":outputs,"latent":latent_outputs}
                if previous is not None and previous != current_hashes:
                    raise RuntimeError("T5 dedicated encoder prefetch changed exact video or latent outputs")
                hashes[key] = current_hashes
                rows.append({"kind": "dedicated_queue", "mode": mode, "frames": frames, "repeat": repeat,
                             "requests": args.pipeline_requests, "wall_ms": wall_ms,
                             "videos_per_second": args.pipeline_requests * 1000 / wall_ms,
                             "completed": completed, "video_sha256": outputs,"latent_sha256":latent_outputs,
                             "paired_exact": True if previous is not None else None})
    return rows


def _run_child(args):
    # Bind before importing numerical runtimes so lazily created OpenMP/BLAS
    # worker threads inherit the intended physical CPU set.
    if args.cpu_affinity == "auto":
        from .workflow_overlap import bind_cpu_to_gpu
        affinity = bind_cpu_to_gpu(args.device, args.cpu_threads)
    else:
        affinity = {"policy": "none", "cpus": sorted(os.sched_getaffinity(0))}
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    import numpy as np
    import torch
    from .workflow_metrics import PhaseRecorder, NvmlSampler, summarize_phases
    from ..models.wan.runtime import EncodedPrompt

    phase = args.child
    config = PHASES[phase]
    out = Path(args.output).resolve()
    phase_dir = out / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    cpu_threads = args.cpu_threads
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Preserve the pinned runtime's FP32 VAE convolution policy (normally True).
    # Only UMT5's matmul TF32 is disabled for a fair FP32 encoder comparison.
    torch.cuda.set_device(args.device)
    dtype = getattr(torch, config["encoder_dtype"])
    encoder_device = ("cpu" if phase == "T0" else args.encoder_device if phase == "T5" else args.device)
    if phase == "T5" and encoder_device == args.device:
        raise ValueError("T5 requires distinct encoder and denoising CUDA devices")
    recorder = PhaseRecorder(device=args.device, synchronize=True, reset_peaks=True, nvtx=True)
    # T0 is an isolated stage run: sample the idle Wan GPU as well, showing
    # resident weights without misattributing an earlier VAE allocator peak.
    encoder_recorder = (recorder if encoder_device in (args.device, "cpu") else
                        PhaseRecorder(device=encoder_device, synchronize=True, reset_peaks=True, nvtx=True))
    unique_devices = tuple(sorted({int(args.device.split(":")[-1])} |
                                 ({int(encoder_device.split(":")[-1])} if encoder_device != "cpu" else set())))
    sampler = NvmlSampler(devices=unique_devices, interval_s=args.sample_interval)
    result = {"schema": "ditserve.workflow.encoder.v1", "variant": phase, "configuration": config,
              "model_path": str(Path(args.model_path).resolve()), "encoder_device": encoder_device,
              "wan_device": args.device if config["video"] else None, "frames": args.frames,
              "steps": args.steps, "warmups": args.warmups, "repeats": args.repeats,
              "save_videos": args.save_videos,
              "cpu_threads": cpu_threads, "cpu_affinity": affinity,
              "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
              "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32), "pid": os.getpid(),
              "numeric": [], "requests": [], "status": "running", "started_unix_ns": time.time_ns(),
              "pipeline_requests": args.pipeline_requests if phase == "T5" else 0,
              "warnings": ["Phase synchronization is intentional for isolated resource attribution; use overlap experiments for throughput.",
                           "SM hardware counters are supplied by profiler traces, not NVML GPU busy samples."]}
    worker = None
    metadata = {"variant": phase}
    sampler.start()
    try:
        with recorder.phase("cold_model_load", variant=phase):
            if config["video"]:
                worker = _experimental_worker(args.device, args.model_path, args.attention)
                pipe = worker.pipe
                pipe.text_encoder.to(dtype=dtype).eval()
            else:
                pipe = _standalone_encoder(args.model_path, config["encoder_dtype"], encoder_device)
        if config["video"] and phase in ("T3", "T5"):
            with encoder_recorder.phase("cold_encoder_weight_h2d", variant=phase):
                pipe.text_encoder.to(device=encoder_device, dtype=dtype)
        _instrument_prompt_helper(pipe, encoder_recorder, metadata)

        def encode(case, request_id, purpose, repeat=None):
            metadata.clear()
            metadata.update(variant=phase, request_id=request_id, purpose=purpose, repeat=repeat,
                            prompt_case=case["id"])
            started = time.perf_counter_ns()
            if phase == "T4":
                with encoder_recorder.phase("encoder_weight_h2d", **metadata):
                    pipe.text_encoder.to(device=encoder_device)
            with torch.inference_mode():
                pair = pipe.encode_prompt(prompt=case["prompt"], negative_prompt=case["negative_prompt"],
                                          do_classifier_free_guidance=True, num_videos_per_prompt=1,
                                          max_sequence_length=512, device=encoder_device, dtype=dtype)
            if phase == "T4":
                with encoder_recorder.phase("encoder_weight_d2h", **metadata):
                    pipe.text_encoder.to(device="cpu")
                    torch.cuda.empty_cache()
            if phase == "T5":
                with encoder_recorder.phase("embedding_d2h", **metadata):
                    pair = tuple(None if value is None else value.to("cpu") for value in pair)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if any(value is not None and not bool(torch.isfinite(value).all()) for value in pair):
                raise RuntimeError("nonfinite encoded prompt")
            return pair, elapsed

        for repeat in range(args.warmups):
            pair, _ = encode(PROMPTS[0], "encoder-warmup-%d" % repeat, "warmup", repeat)
            del pair
        # Timing-only encoder samples apply to every variant, and are distinct
        # from whole video samples, numerical validation and model cold load.
        for repeat in range(args.repeats):
            pair, ms = encode(PROMPTS[0], "encoder-measured-%d" % repeat, "encoder_measurement", repeat)
            result["requests"].append({"kind": "encoder", "repeat": repeat, "encode_ms": ms})
            del pair
        for case in PROMPTS:
            pair, ms = encode(case, "numeric-" + case["id"], "numerical_validation")
            arrays = _embedding_arrays(pair)
            numeric_path = phase_dir / "numeric" / (case["id"] + ".npz")
            numeric_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(numeric_path, **arrays)
            comparison = (_compare_embeddings(arrays, out / "T0" / "numeric" / (case["id"] + ".npz"))
                          if phase != "T0" else {"status": "reference"})
            result["numeric"].append({"case": case["id"], "encode_ms": ms,
                                      "artifact": str(numeric_path), "comparison": comparison,
                                      "finite": all(bool(np.isfinite(value).all()) for value in arrays.values())})
            del pair, arrays

        if config["video"] and not args.encoder_only:
            for frames in args.frames:
                for repeat in range(-args.warmups, args.repeats):
                    purpose = "warmup" if repeat < 0 else "video_measurement"
                    request_id = "%s-f%d-r%d" % (phase, frames, repeat)
                    spec = _spec(PROMPTS[0], frames, args.steps, request_id)
                    started = time.perf_counter_ns()
                    pair, encode_ms = encode(PROMPTS[0], request_id, purpose, repeat)
                    payload = EncodedPrompt(spec=spec, prompt_embeds=pair[0], negative_prompt_embeds=pair[1],
                                            cpu_prompt_encode_ms=0.0, owner=worker._owner)
                    context = {"variant": phase, "request_id": request_id, "purpose": purpose,
                               "frames": frames, "repeat": repeat}
                    with recorder.phase("wan_prepare_input_h2d_latent", **context):
                        state = worker.prepare_encoded(spec, payload)
                    del payload, pair
                    for step in range(args.steps):
                        with recorder.phase("dit_step", step=step, **context):
                            worker.step(state)
                    # Decode leaves state.latents unchanged.  Copy comparison
                    # artifacts only after timing, avoiding an extra allocation
                    # in the VAE's peak memory measurement.
                    video = _decode_separate(worker, state, recorder, context)
                    torch.cuda.synchronize(args.device)
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    row = {"kind": "video", "request_id": request_id, "frames": frames, "repeat": repeat,
                           "purpose": purpose, "encode_ms": encode_ms, "inference_wall_ms": elapsed,
                           "finite": bool(np.isfinite(video).all()), "shape": list(video.shape)}
                    if not row["finite"]:
                        raise RuntimeError("nonfinite decoded video")
                    if repeat == 0:
                        latent = _tensor_array(state.latents)
                        directory = phase_dir / "outputs" / str(frames)
                        row["artifacts"], row["video"] = _save_encoder_outputs(
                            directory, latent, video, save_videos=args.save_videos)
                        row["sha256"] = {"latent": hashlib.sha256(latent.tobytes()).hexdigest(),
                                         "video": row["video"]["array_sha256"]}
                        reference_dir = out / "T0" / "outputs" / str(frames)
                        row["comparison"] = {}
                        for name, value in (("latent", latent), ("video", video)):
                            reference = reference_dir / (name + ".npy")
                            if phase == "T0":
                                row["comparison"][name] = {"status": "reference"}
                            elif name == "video" and (out / "T0/result.json").is_file():
                                source = json.loads((out / "T0/result.json").read_text())
                                rows = [item for item in source["requests"] if item.get("kind") == "video"
                                        and item.get("frames") == frames and item.get("repeat") == 0]
                                if source.get("status") != "completed" or len(rows) != 1:
                                    raise ValueError("missing unique completed T0 video reference")
                                source_latent = np.load(reference_dir / "latent.npy", mmap_mode="r", allow_pickle=False)
                                expected = _reference_video(worker, rows[0], source_latent, reference, recorder, context)
                                row["comparison"][name] = compare_arrays(value, expected)
                                row["comparison"][name]["reference_source"] = (
                                    "retained_video" if reference.is_file() else "verified_latent_reconstruction")
                                del expected, source_latent
                            else:
                                row["comparison"][name] = (compare_arrays(value, np.load(reference, mmap_mode="r", allow_pickle=False))
                                    if reference.is_file() else {"status": "reference_missing", "reference": str(reference)})
                        del latent
                    result["requests"].append(row)
                    worker.release(state)
                    del state, video
                    _json_write(phase_dir / "progress.json", result)
            if phase == "T5" and args.pipeline_requests:
                result["requests"].extend(_dedicated_queue(worker, encode, recorder, args))
        result["status"] = "completed"
    except BaseException as error:
        result["status"] = ("interrupted" if isinstance(error, KeyboardInterrupt) else
                            "oom" if isinstance(error, torch.OutOfMemoryError) else "failed")
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        sampler.stop()
        records = list(recorder.records)
        if encoder_recorder is not recorder:
            records.extend(encoder_recorder.records)
        records.sort(key=lambda row: row.get("start_ns", row.get("started_ns", 0)))
        result["phase_records"] = records
        result["phase_summary"] = summarize_phases(records)
        result["samples"] = sampler.samples
        result["sampling_errors"] = sampler.errors
        result["sampling_summary"] = sampler.summary()
        result["finished_unix_ns"] = time.time_ns()
        result["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                             "devices": [{"logical_index": device, "name": torch.cuda.get_device_name(device),
                                          "total_memory": torch.cuda.get_device_properties(device).total_memory}
                                         for device in unique_devices]}
        _json_write(phase_dir / "result.json", result)
    print(json.dumps({"variant": phase, "status": result["status"], "result": str(phase_dir / "result.json")}), flush=True)
    return 0 if result["status"] == "completed" else 1


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phases", default="T0,T1,T2,T3,T4,T5")
    parser.add_argument("--device", default="cuda:1", help="Wan GPU, and encoding GPU except T5")
    parser.add_argument("--encoder-device", default="cuda:0", help="Dedicated T5 encoding GPU")
    parser.add_argument("--frames", default="17,33")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--cpu-affinity", choices=("auto", "none"), default="auto",
                        help="Bind to GPU-local physical CPU cores before initializing numerical runtimes")
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--pipeline-requests", type=int, default=4,
                        help="T5 paired serial/prefetch queue size; 0 skips queue measurements")
    parser.add_argument("--queue-repeats", type=int, default=3)
    parser.add_argument("--attention", choices=("math", "flash", "efficient"), default="flash")
    parser.add_argument("--encoder-only", action="store_true", help="Skip videos for compatibility/numerical preflight")
    parser.add_argument("--save-videos", action="store_true", help="Retain raw video arrays for manual quality review")
    parser.add_argument("--child", choices=tuple(PHASES), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        args.frames = [int(value) for value in args.frames.split(",")]
    except ValueError:
        parser.error("--frames must be comma-separated integers")
    if not args.frames or any(value < 1 or (value - 1) % 4 for value in args.frames):
        parser.error("Wan frame counts must be positive 4*k+1 values")
    if args.steps < 1 or args.repeats < 1 or args.warmups < 0 or args.cpu_threads < 1 or args.sample_interval <= 0:
        parser.error("steps/repeats/threads/sample-interval must be positive; warmups must be nonnegative")
    if args.pipeline_requests < 0 or args.queue_repeats < 1:
        parser.error("pipeline-requests must be nonnegative and queue-repeats positive")
    if any(not value.startswith("cuda:") or not value[5:].isdigit() for value in (args.device, args.encoder_device)):
        parser.error("devices must be explicit cuda:N indices")
    args.phases = [value.strip() for value in args.phases.split(",")]
    if not args.phases or any(value not in PHASES for value in args.phases) or len(set(args.phases)) != len(args.phases):
        parser.error("--phases must contain unique names T0 through T5")
    return args


def main(argv=None):
    args = _parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("experiment interrupted by signal %d" % signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.child:
        return _run_child(args)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    # No torch import or CUDA initialization in the orchestrating process.
    for phase in args.phases:
        command = [sys.executable, "-m", __name__, "--child", phase, "--model-path", args.model_path,
                   "--output", str(root), "--device", args.device, "--encoder-device", args.encoder_device,
                   "--frames", ",".join(map(str, args.frames)), "--steps", str(args.steps),
                   "--repeats", str(args.repeats), "--warmups", str(args.warmups),
                   "--cpu-threads", str(args.cpu_threads), "--sample-interval", str(args.sample_interval),
                   "--cpu-affinity", args.cpu_affinity,
                   "--pipeline-requests", str(args.pipeline_requests), "--queue-repeats", str(args.queue_repeats),
                   "--attention", args.attention]
        # __name__ is __main__ for python -m invocation; use the stable module.
        command[2] = "ditserve.evaluation.workflow_encoder"
        if args.encoder_only:
            command.append("--encoder-only")
        if args.save_videos:
            command.append("--save-videos")
        phase_dir = root / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        was_interrupted = False
        with (phase_dir / "stdout.log").open("w") as stdout, (phase_dir / "stderr.log").open("w") as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                returncode = child.wait()
            except KeyboardInterrupt:
                was_interrupted = True
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    returncode = child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    returncode = child.wait()
        result_path = phase_dir / "result.json"
        row = {"variant": phase, "returncode": returncode, "result": str(result_path)}
        if result_path.is_file():
            row["status"] = json.loads(result_path.read_text())["status"]
        else:
            row["status"] = "interrupted" if was_interrupted else "process_failed_before_result"
        results.append(row)
        _json_write(root / "summary.json", {"schema": "ditserve.workflow.encoder.v1", "variants": results})
        print(json.dumps(row), flush=True)
        if was_interrupted:
            break
    return 0 if all(row["status"] == "completed" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
