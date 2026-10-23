"""Explicit request state for the pinned Wan 1.3B serving worker.

Only the standard library is imported until a worker is constructed. GPU work is
serialized on one device; CPU prompt encoding may overlap it on another thread.
The denoising arithmetic follows diffusers v0.36.0 WanPipeline (Apache-2.0):
https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/pipelines/wan/pipeline_wan.py
Every step computes the full model; there is no feature caching.
"""

import contextlib
from copy import deepcopy
import dataclasses
import importlib
import importlib.metadata
import math
import threading
import time
from pathlib import Path

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
REQUIRED_VERSIONS = {"torch": "2.10.0+cu126", "diffusers": "0.36.0", "cache-dit": "1.5.0"}
ATTENTION_BACKENDS = {"math": "MATH", "flash": "FLASH_ATTENTION", "efficient": "EFFICIENT_ATTENTION"}
# Explicit schemas for the pinned worker and diffusers 0.36.0 UniPC. Do not
# infer these sets from a live instance: that would silently bless an
# unexpected CUDA-bearing attribute. A deliberate state extension must update
# this contract together with the wire codec (parallel.py) and its tests.
REQUEST_STATE_FIELDS = frozenset((
    "request_id", "spec", "step_index", "latents", "events", "scheduler", "generator",
    "prompt_embeds", "negative_prompt_embeds", "residency", "finished", "owner", "prepare_ms",
))
UNIPC_REQUIRED_FIELDS = frozenset((
    "_internal_dict", "betas", "alphas", "alphas_cumprod", "alpha_t", "sigma_t",
    "lambda_t", "sigmas", "init_noise_sigma", "predict_x0", "num_inference_steps",
    "timesteps", "model_outputs", "timestep_list", "lower_order_nums",
    "disable_corrector", "solver_p", "last_sample", "_step_index", "_begin_index",
))
# UniPC creates this_order inside its first step, not in __init__/set_timesteps.
UNIPC_OPTIONAL_FIELDS = frozenset(("this_order",))
SPEC_FIELDS = frozenset((
    "request_id", "prompt", "negative_prompt", "seed", "height", "width", "frames",
    "steps", "guidance_scale",
))


def _require_fields(value, required, optional=(), label="state"):
    try:
        actual = set(vars(value))
    except TypeError as error:
        raise TypeError(label + " must expose the fixed instance-state schema") from error
    unknown = actual - required - set(optional)
    missing = required - actual
    if unknown or missing:
        raise TypeError("unsupported %s schema: unknown=%s missing=%s" % (
            label, sorted(unknown), sorted(missing)))


def _validate_metadata(value, label, active=None):
    """Metadata outside the moved payload must contain no tensors/opaque state."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise ValueError("nonfinite metadata at " + label)
    if type(value) not in (dict, list, tuple):
        raise TypeError("unsupported metadata object at %s: %s" % (label, type(value).__name__))
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError("cyclic metadata at " + label)
    active.add(id(value))
    try:
        children = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in children:
            if isinstance(value, dict) and type(key) is not str:
                raise TypeError("metadata dictionary keys must be strings at " + label)
            _validate_metadata(child, label + "." + str(key), active)
    finally:
        active.remove(id(value))


def normalize_spec(spec):
    # Check before deepcopy: copying an unexpected tensor can itself allocate
    # GPU memory, before a rejected request has acquired any tracked state.
    result = dict(spec)
    if any(type(key) is not str for key in result):
        raise ValueError("spec keys must be strings")
    unknown = set(result) - SPEC_FIELDS
    if unknown:
        raise ValueError("unknown spec fields: " + ", ".join(sorted(unknown)))
    _validate_metadata(result, "spec")
    defaults = {"negative_prompt": "", "guidance_scale": 5.0}
    for key, value in defaults.items():
        result.setdefault(key, value)
    if not isinstance(result.get("request_id"), str) or not result["request_id"]:
        raise ValueError("request_id must be a nonempty string")
    if not isinstance(result.get("prompt"), str) or not isinstance(result["negative_prompt"], str):
        raise ValueError("prompt and negative_prompt must be strings")
    for key in ("height", "width", "frames", "steps", "seed"):
        if type(result.get(key)) is not int or result[key] < (0 if key == "seed" else 1):
            raise ValueError("invalid integer spec field: " + key)
    if result["seed"] >= 2 ** 63:
        raise ValueError("seed must be less than 2**63")
    if result["height"] % 16 or result["width"] % 16 or (result["frames"] - 1) % 4:
        raise ValueError("Wan requires H/W multiples of 16 and frames=4*k+1")
    value = result["guidance_scale"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid finite nonnegative spec field: guidance_scale")
    return result


@dataclasses.dataclass
class EncodedPrompt:
    """CPU-only, single-use prompt payload bound to one worker and exact request."""

    spec: dict
    prompt_embeds: object
    negative_prompt_embeds: object
    cpu_prompt_encode_ms: float
    owner: object = dataclasses.field(repr=False)
    consumed: bool = False
    negative_prompt_cache: object = None


@dataclasses.dataclass
class RequestState:
    request_id: str
    spec: dict
    step_index: int = 0  # Number of completed solver updates, not a timestep value.
    latents: object = None
    events: list = dataclasses.field(default_factory=list)
    scheduler: object = None
    generator: object = None
    prompt_embeds: object = None
    negative_prompt_embeds: object = None
    residency: str = "gpu"
    finished: bool = False
    owner: object = None
    prepare_ms: float = 0.0


class WanStepWorker:
    """One resident model, independent request schedulers and RNG state."""

    def __init__(self, device="cuda:0", model_path=None, attention="math", *,
                 negative_prompt_cache_entries=None, negative_prompt_cache_bytes=None):
        from .prompt_cache import cache_options
        cache_config = cache_options({name: value for name, value in (
            ("negative_prompt_cache_entries", negative_prompt_cache_entries),
            ("negative_prompt_cache_bytes", negative_prompt_cache_bytes)) if value is not None})
        if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("device must be an explicit cuda:N device")
        if attention not in ATTENTION_BACKENDS:
            raise ValueError("attention must be math, flash, or efficient")
        self.attention = attention
        for package, expected in REQUIRED_VERSIONS.items():
            actual = importlib.metadata.version(package)
            if actual != expected:
                raise RuntimeError("unsupported %s version: %s (expected %s)" % (package, actual, expected))
        self.torch = importlib.import_module("torch")
        self.diffusers = importlib.import_module("diffusers")
        self.cache_dit = importlib.import_module("cache_dit")
        from .. import pipeline as p
        self.p = p
        self.device = device
        if not model_path:
            raise ValueError("Wan requires config.model_path (resolved from the environment profile)")
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir() or not (self.model_path / "model_index.json").is_file():
            raise ValueError("fixed local Wan snapshot is missing: " + str(self.model_path))
        self.torch.cuda.set_device(device)
        if not self.torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        from types import SimpleNamespace
        args = SimpleNamespace(model="wan", model_id=str(self.model_path), revision=REVISION,
                               dtype="float16", vae_dtype="float32", text_encoder_dtype="float32",
                               text_encoder_device="cpu", local_files_only=True, device=device)
        self.pipe = p._pipeline(self.torch, self.diffusers, args)
        if (type(self.pipe.scheduler).__name__ != "UniPCMultistepScheduler" or
                self.pipe.config.boundary_ratio is not None or self.pipe.config.expand_timesteps or
                getattr(self.pipe, "transformer_2", None) is not None):
            raise RuntimeError("worker supports only fixed Wan2.1 1.3B and UniPC recipe")
        if self.pipe.scheduler.config.solver_p is not None:
            raise RuntimeError("external UniPC predictor state is unsupported")
        self._scheduler_config = dict(self.pipe.scheduler.config)
        self._scheduler_type = type(self.pipe.scheduler)
        # _pipeline creates a distinct subclass per worker; no wrapper is shared.
        self._official_call = type(self.pipe).__call__
        # The native attention processor is also what the Ulysses SP attachment wraps.
        self.cache_dit.set_attn_backend(self.pipe, attention_backend="native")
        self._original_forward = self.pipe.transformer.forward
        self._lock = threading.RLock()
        self._encoder_lock = threading.Lock()
        self._owner = object()
        self._states = {}
        self._negative_prompt_cache = None
        if cache_config:
            from .prompt_cache import NegativePromptCache
            self._negative_prompt_cache = NegativePromptCache(
                entries=cache_config["negative_prompt_cache_entries"],
                byte_limit=cache_config["negative_prompt_cache_bytes"],
                recipe=dict(model_id=MODEL_ID, revision=REVISION, model_path=str(self.model_path),
                            diffusers=REQUIRED_VERSIONS["diffusers"], text_encoder="UMT5",
                            device="cpu", dtype="torch.float32", max_sequence_length=512,
                            num_videos_per_prompt=1))

    @staticmethod
    def _validate_scheduler(scheduler):
        _require_fields(scheduler, UNIPC_REQUIRED_FIELDS, UNIPC_OPTIONAL_FIELDS, label="UniPC scheduler")
        if scheduler.solver_p is not None:
            raise RuntimeError("external UniPC predictor state is unsupported")
        if scheduler._step_index is not None and scheduler._step_index > 0 and "this_order" not in vars(scheduler):
            raise TypeError("active UniPC scheduler is missing this_order")

    def _validate_request_schema(self, state):
        _require_fields(state, REQUEST_STATE_FIELDS, label="RequestState")
        if type(state.spec) is not dict or set(state.spec) != SPEC_FIELDS:
            raise ValueError("RequestState.spec must contain exactly the normalized spec fields")
        if normalize_spec(state.spec) != state.spec:
            raise ValueError("RequestState.spec is not normalized")
        # A released state's payload was deliberately removed. Ownership checks
        # still reject its further use, while the top-level schema remains exact.
        if state.residency == "released":
            return
        self._validate_scheduler(state.scheduler)

    def _check(self, state, resident=None):
        self._validate_request_schema(state)
        if state.owner is not self._owner or self._states.get(state.request_id) is not state:
            raise ValueError("request is not owned by this worker or was released")
        if state.residency == "released":
            raise ValueError("request was released")
        if resident is not None and state.residency != resident:
            raise ValueError("request must reside in " + resident)

    def _sync(self):
        self.torch.cuda.synchronize(self.device)

    def _dit_attention_context(self):
        """Force one native SDPA kernel; unsupported shapes/devices fail loudly."""
        attention = self.torch.nn.attention
        return attention.sdpa_kernel([getattr(attention.SDPBackend, ATTENTION_BACKENDS[self.attention])])

    @contextlib.contextmanager
    def _activate(self, state):
        self._check(state, "gpu")
        forward = self._original_forward
        def selected_forward(*args, **kwargs):
            # Scope the override to DiT calls so the official reference loop's
            # FP32 VAE still executes under its surrounding MATH context.
            with self._dit_attention_context():
                return forward(*args, **kwargs)
        self.pipe.transformer.forward = selected_forward
        try:
            yield
        finally:
            self.pipe.transformer.forward = forward

    def prepare(self, spec):
        """Synchronous compatibility path using the same per-request encoder."""
        return self.prepare_encoded(spec, self.encode_prompt(spec))

    def encode_prompt(self, spec):
        """Encode on CPU without acquiring the GPU lock or changing SDPA flags.

        The pinned UMT5 encoder uses explicit matmul/softmax attention. Changing
        global SDPA settings here would race a concurrently executing DiT/VAE.
        Optional negative embeddings are reused only within this worker. Each
        request retains a separate payload and its own CPU tensors.
        """
        spec = normalize_spec(spec)
        with self._encoder_lock, self.torch.inference_mode():
            started = time.perf_counter()
            arguments = dict(
                prompt=spec["prompt"], negative_prompt=spec["negative_prompt"],
                do_classifier_free_guidance=spec["guidance_scale"] > 1,
                num_videos_per_prompt=1, max_sequence_length=512, device="cpu", dtype=self.torch.float32)
            cache = getattr(self, "_negative_prompt_cache", None)
            receipt = None
            if cache is None:
                prompt, negative = self.pipe.encode_prompt(**arguments)
                self._validate_cpu_embeddings(prompt, negative, spec)
            else:
                prompt, negative, receipt = cache.encode(self.pipe.encode_prompt, arguments,
                    request_id=spec["request_id"],
                    validate=lambda prompt, negative: self._validate_cpu_embeddings(prompt, negative, spec))
            elapsed_ms = (time.perf_counter() - started) * 1000
            if receipt is not None:
                receipt["cpu_prompt_encode_ms"] = elapsed_ms
            return EncodedPrompt(spec=spec, prompt_embeds=prompt, negative_prompt_embeds=negative,
                                 cpu_prompt_encode_ms=elapsed_ms, owner=self._owner,
                                 negative_prompt_cache=receipt)

    def clear_negative_prompt_cache(self):
        """Release retained CPU tensors after encoder shutdown, including failures."""
        cache = getattr(self, "_negative_prompt_cache", None)
        if cache is None:
            return
        with self._encoder_lock:
            cache.clear()

    def _validate_cpu_embeddings(self, prompt, negative, spec):
        if prompt is None or (spec["guidance_scale"] > 1 and negative is None):
            raise ValueError("missing CPU prompt embeddings")
        for tensor in (prompt, negative):
            if tensor is None:
                continue
            if not self.torch.is_tensor(tensor) or tensor.device.type != "cpu" or tensor.dtype != self.torch.float32:
                raise ValueError("encoded prompt payload requires CPU float32 tensors")
            if not bool(self.torch.isfinite(tensor).all()):
                raise RuntimeError("nonfinite CPU prompt embeddings")

    def prepare_encoded(self, spec, payload):
        """Install an encoded request on GPU; account for encoding service work.

        prepare_ms is CPU encoding plus GPU preparation work, even when encoding
        overlapped another request. It is not the caller's wall-clock interval.
        """
        spec = normalize_spec(spec)
        with self._lock, self.torch.inference_mode(), self.p._sdpa_math_context(self.torch):
            if not isinstance(payload, EncodedPrompt) or payload.owner is not self._owner:
                raise ValueError("encoded prompt belongs to another worker")
            if payload.consumed:
                raise ValueError("encoded prompt was already consumed")
            if payload.spec != spec:
                raise ValueError("encoded prompt does not match request spec")
            if (not isinstance(payload.cpu_prompt_encode_ms, (int, float)) or
                    not math.isfinite(payload.cpu_prompt_encode_ms) or payload.cpu_prompt_encode_ms < 0):
                raise ValueError("invalid CPU prompt encoding duration")
            if spec["request_id"] in self._states:
                raise ValueError("duplicate live request_id")
            self._validate_cpu_embeddings(payload.prompt_embeds, payload.negative_prompt_embeds, spec)
            self._sync()
            started = time.perf_counter()
            prompt = payload.prompt_embeds.to(device=self.device, dtype=self.torch.float16)
            negative = payload.negative_prompt_embeds
            negative = None if negative is None else negative.to(device=self.device, dtype=self.torch.float16)
            if not bool(self.torch.isfinite(prompt).all()) or (negative is not None and not bool(self.torch.isfinite(negative).all())):
                raise RuntimeError("nonfinite FP16 prompt embeddings")
            scheduler = self._scheduler_type.from_config(self._scheduler_config)
            scheduler.set_timesteps(spec["steps"], device=self.device)
            if len(scheduler.timesteps) != spec["steps"]:
                raise RuntimeError("unexpected sampler timestep count")
            generator = self.torch.Generator(device="cpu").manual_seed(spec["seed"])
            latents = self.pipe.prepare_latents(
                1, self.pipe.transformer.config.in_channels, spec["height"], spec["width"],
                spec["frames"], self.torch.float32, self.device, generator, None)
            state = RequestState(spec["request_id"], spec, latents=latents, scheduler=scheduler,
                                 generator=generator, prompt_embeds=prompt, negative_prompt_embeds=negative,
                                 owner=self._owner)
            self._validate_request_schema(state)
            self._sync()
            gpu_prepare_ms = (time.perf_counter() - started) * 1000
            state.prepare_ms = payload.cpu_prompt_encode_ms + gpu_prepare_ms
            state.events.append({"kind": "prepare", "ms": state.prepare_ms,
                                 "cpu_prompt_encode_ms": payload.cpu_prompt_encode_ms,
                                 "gpu_prepare_ms": gpu_prepare_ms})
            if payload.negative_prompt_cache is not None:
                state.events[-1]["negative_prompt_cache"] = deepcopy(payload.negative_prompt_cache)
            self._states[state.request_id] = state
            payload.consumed = True
            return state

    def step(self, state):
        with self._lock, self.torch.inference_mode(), self.p._sdpa_math_context(self.torch):
            self._check(state, "gpu")
            if state.finished or state.step_index >= state.spec["steps"]:
                raise ValueError("request has no remaining denoising steps")
            self._sync()
            started = time.perf_counter()
            index = state.step_index
            with self._activate(state):
                timestep_scalar = state.scheduler.timesteps[index]
                timestep = timestep_scalar.expand(state.latents.shape[0])
                model_input = state.latents.to(self.pipe.transformer.dtype)
                with self.pipe.transformer.cache_context("cond"):
                    prediction = self.pipe.transformer(hidden_states=model_input, timestep=timestep,
                        encoder_hidden_states=state.prompt_embeds, attention_kwargs=None, return_dict=False)[0]
                if state.spec["guidance_scale"] > 1:
                    with self.pipe.transformer.cache_context("uncond"):
                        unconditional = self.pipe.transformer(hidden_states=model_input, timestep=timestep,
                            encoder_hidden_states=state.negative_prompt_embeds, attention_kwargs=None, return_dict=False)[0]
                    prediction = unconditional + state.spec["guidance_scale"] * (prediction - unconditional)
                state.latents = state.scheduler.step(prediction, timestep_scalar, state.latents, return_dict=False)[0]
                state.step_index += 1
            self._sync()
            finite = bool(self.torch.isfinite(state.latents).all().item())
            ms = (time.perf_counter() - started) * 1000
            if not finite:
                raise RuntimeError("nonfinite latent after solver step %d" % index)
            record = {"kind": "step", "step_index": index, "completed_steps": state.step_index,
                      "ms": ms, "finite": finite, "path": "compute"}
            state.events.append(record)
            return record

    def _payload(self, state):
        """The request's own tensors; block packets count only storage beyond this."""
        self._validate_request_schema(state)
        return {"latents": state.latents, "scheduler": vars(state.scheduler),
                "prompt_embeds": state.prompt_embeds, "negative_prompt_embeds": state.negative_prompt_embeds,
                "rng_state": state.generator.get_state()}

    def finish(self, state):
        with self._lock, self.torch.inference_mode(), self.p._sdpa_math_context(self.torch):
            self._check(state, "gpu")
            if state.step_index != state.spec["steps"]:
                raise ValueError("finish requires all solver steps to be complete")
            if state.finished:
                raise ValueError("request was already decoded")
            self._sync()
            started = time.perf_counter()
            from ..pipeline import decode_final_latent
            array = decode_final_latent(self.torch, self.pipe, state.latents)
            self._sync()
            ms = (time.perf_counter() - started) * 1000
            numpy = importlib.import_module("numpy")
            if not numpy.isfinite(array).all():
                raise RuntimeError("nonfinite decoded output")
            state.finished = True
            state.events.append({"kind": "decode", "ms": ms, "shape": list(array.shape), "finite": True})
            return array

    def decode_chunks(self, state, count):
        """Advance a resumable decode by ``count`` latent frames.

        Returns (array or None, info); the array appears with the last chunk.
        Arithmetic and order match ``finish``; other work may run in between.
        """
        with self._lock, self.torch.inference_mode(), self.p._sdpa_math_context(self.torch):
            self._check(state, "gpu")
            if state.step_index != state.spec["steps"] or state.finished:
                raise ValueError("chunked decode requires a complete undecoded request")
            decodes = self.__dict__.setdefault("_chunked_decodes", {})
            self._sync()
            started = time.perf_counter()
            from ..pipeline import ChunkedDecode
            decode = decodes.get(state.request_id) or decodes.setdefault(
                state.request_id, ChunkedDecode(self.torch, self.pipe, state.latents))
            done = decode.advance(count)
            info = dict(frames_done=decode.index, frames=decode.frames, done=done)
            array = None
            if done:
                array = decode.finish()
                del decodes[state.request_id]
            self._sync()
            info["ms"] = (time.perf_counter() - started) * 1000
            if done:
                numpy = importlib.import_module("numpy")
                if not numpy.isfinite(array).all():
                    raise RuntimeError("nonfinite decoded output")
                state.finished = True
            state.events.append(dict(kind="decode_chunk", shape=list(array.shape) if done else None, **info))
            return array, info

    def release(self, state):
        with self._lock:
            self._check(state)
            self.__dict__.get("_chunked_decodes", {}).pop(state.request_id, None)
            self._sync()
            state.latents = state.prompt_embeds = state.negative_prompt_embeds = None
            state.scheduler = state.generator = None
            state.residency = "released"
            del self._states[state.request_id]

    def reference(self, spec):
        """Run the unmodified official denoising/decode loop with the same inputs.

        Inputs are prepared by the same official encode_prompt/prepare_latents
        methods, so this checks the step loop's arithmetic and order against
        diffusers itself.
        """
        with self._lock:
            state = self.prepare(spec)
            old_scheduler = self.pipe.scheduler
            captured = {}
            def callback(pipe, index, timestep, kwargs):
                if int(index) + 1 == state.spec["steps"]:
                    captured["latents"] = kwargs["latents"].detach().cpu().clone()
                return kwargs
            try:
                with self.torch.inference_mode(), self.p._sdpa_math_context(self.torch), self._activate(state):
                    self.pipe.scheduler = state.scheduler
                    self._sync()
                    started = time.perf_counter()
                    output = self._official_call(self.pipe, prompt_embeds=state.prompt_embeds,
                        negative_prompt_embeds=state.negative_prompt_embeds, latents=state.latents,
                        generator=state.generator, height=state.spec["height"], width=state.spec["width"],
                        num_frames=state.spec["frames"], num_inference_steps=state.spec["steps"],
                        guidance_scale=state.spec["guidance_scale"], output_type="np",
                        callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
                    self._sync()
                    ms = (time.perf_counter() - started) * 1000
                numpy = importlib.import_module("numpy")
                if "latents" not in captured or not numpy.isfinite(output.frames).all():
                    raise RuntimeError("reference output missing or nonfinite")
                return {"output": output.frames, "latents": captured["latents"],
                        "elapsed_ms": state.prepare_ms + ms, "prepare_ms": state.prepare_ms,
                        "pipeline_ms": ms, "instrumented": True,
                        "reference_scope": "official Wan loop with official prepared inputs; final latent D2H capture included"}
            finally:
                self.pipe.scheduler = old_scheduler
                self.pipe._current_timestep = None
                self.release(state)
