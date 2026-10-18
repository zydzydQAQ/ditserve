"""Explicit request state for the pinned Wan 1.3B mechanism experiments.

Only the standard library is imported until a worker is constructed. GPU work is
serialized on one device; CPU prompt encoding may overlap it on another thread.
The denoising arithmetic follows diffusers v0.36.0 WanPipeline (Apache-2.0):
https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/pipelines/wan/pipeline_wan.py
DBCache math is left to cache-dit 1.5.0; this adapter owns its request contexts.
"""

import contextlib
import copy
import dataclasses
import enum
import importlib
import importlib.metadata
import math
import threading
import time
from pathlib import Path

from ...runtime import ROOT

REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
DEFAULT_MODEL_PATH = (ROOT / ".cache/huggingface/hub" /
                      "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots" / REVISION)
REQUIRED_VERSIONS = {"torch": "2.10.0+cu126", "diffusers": "0.36.0", "cache-dit": "1.5.0"}
ATTENTION_BACKENDS = {"math": "MATH", "flash": "FLASH_ATTENTION", "efficient": "EFFICIENT_ATTENTION"}
CONTEXT_COUNTERS = (
    "executed_steps", "transformer_executed_steps", "accumulated_executed_steps",
    "accumulated_transformer_executed_steps", "accumulated_cached_steps",
    "cfg_accumulated_cached_steps", "continuous_cached_steps", "cfg_continuous_cached_steps",
    "accumulated_residual_diff", "cfg_accumulated_residual_diff",
)
CALIBRATORS = ("calibrator", "encoder_calibrator", "cfg_calibrator", "cfg_encoder_calibrator")

# Explicit schemas for the pinned worker, diffusers 0.36.0 UniPC, and cache-dit
# 1.5.0. Do not infer these sets from a live instance: that would silently bless
# an unexpected CUDA-bearing attribute. A deliberate state extension must update
# this contract together with its storage/transfer implementation and tests.
REQUEST_STATE_FIELDS = frozenset((
    "request_id", "spec", "step_index", "latents", "cache_enabled", "events",
    "scheduler", "generator", "prompt_embeds", "negative_prompt_embeds", "contexts",
    "residency", "needs_anchor", "finished", "owner", "pending_snapshot",
    "prepare_ms", "current_step_refreshed", "immutable_spec",
))
UNIPC_REQUIRED_FIELDS = frozenset((
    "_internal_dict", "betas", "alphas", "alphas_cumprod", "alpha_t", "sigma_t",
    "lambda_t", "sigmas", "init_noise_sigma", "predict_x0", "num_inference_steps",
    "timesteps", "model_outputs", "timestep_list", "lower_order_nums",
    "disable_corrector", "solver_p", "last_sample", "_step_index", "_begin_index",
))
# UniPC creates this_order inside its first step, not in __init__/set_timesteps.
UNIPC_OPTIONAL_FIELDS = frozenset(("this_order",))
CACHED_CONTEXT_FIELDS = frozenset(CONTEXT_COUNTERS + CALIBRATORS + (
    "name", "buffers", "cache_config", "extra_cache_config", "calibrator_config",
    "cached_steps", "cfg_cached_steps", "residual_diffs", "cfg_residual_diffs",
    "_init_args", "_init_kwargs",
))
DBCACHE_CONFIG_FIELDS = frozenset((
    "cache_type", "Fn_compute_blocks", "Bn_compute_blocks", "residual_diff_threshold",
    "max_accumulated_residual_diff_threshold", "max_warmup_steps", "warmup_interval",
    "max_cached_steps", "max_continuous_cached_steps", "enable_separate_cfg",
    "cfg_compute_first", "cfg_diff_compute_separate", "num_inference_steps",
    "steps_computation_mask", "steps_computation_policy", "force_refresh_step_hint",
    "force_refresh_step_policy",
))
EXTRA_CACHE_CONFIG_FIELDS = frozenset((
    "l1_hidden_states_diff_threshold", "important_condition_threshold", "downsample_factor",
))
SPEC_FIELDS = frozenset((
    "request_id", "prompt", "negative_prompt", "seed", "height", "width", "frames",
    "steps", "guidance_scale", "cache", "threshold", "max_continuous",
))
IMMUTABLE_SPEC_FIELDS = SPEC_FIELDS - {"cache", "threshold", "max_continuous"}


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
    defaults = {"negative_prompt": "", "guidance_scale": 5.0, "cache": False,
                "threshold": 0.08, "max_continuous": 3}
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
    if type(result["cache"]) is not bool:
        raise ValueError("cache must be boolean")
    for key in ("guidance_scale", "threshold"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid finite nonnegative spec field: " + key)
    if type(result["max_continuous"]) is not int or result["max_continuous"] < 1:
        raise ValueError("max_continuous must be a positive integer")
    return result


def _tree_map(value, is_tensor, transform, memo=None):
    """Map the complete supported state graph while preserving object aliases.

    Unknown opaque objects are rejected rather than silently retaining a hidden
    CUDA tensor. Config dataclasses and scheduler FrozenDicts are supported.
    """
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if is_tensor(value):
        result = transform(value)
        memo[id(value)] = result
        return result
    if value is None or isinstance(value, (str, int, float, bool, enum.Enum)):
        return value
    if isinstance(value, dict):
        # Fixed UniPC/cache-dit state dictionaries use plain string keys.
        # Keys are retained rather than moved, so accepting tensor/opaque keys
        # (including a str subclass carrying state) would evade the inventory.
        if any(type(key) is not str for key in value):
            raise TypeError("unsupported dictionary key in request state: expected plain str")
        result = copy.copy(value)
        dict.clear(result)
        memo[id(value)] = result
        for key, child in value.items():
            dict.__setitem__(result, key, _tree_map(child, is_tensor, transform, memo))
        return result
    if isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(_tree_map(child, is_tensor, transform, memo) for child in value)
        return result
    if isinstance(value, tuple):
        result = tuple(_tree_map(child, is_tensor, transform, memo) for child in value)
        memo[id(value)] = result
        return result
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = copy.copy(value)
        memo[id(value)] = result
        for key, child in vars(value).items():
            setattr(result, key, _tree_map(child, is_tensor, transform, memo))
        return result
    # These immutable torch descriptors contain no tensor storage.
    if type(value).__module__ == "torch" and type(value).__name__ in ("device", "dtype", "layout", "memory_format"):
        return value
    raise TypeError("unsupported object in request state: " + type(value).__module__ + "." + type(value).__name__)


@dataclasses.dataclass
class EncodedPrompt:
    """CPU-only, single-use prompt payload bound to one worker and exact request."""

    spec: dict
    prompt_embeds: object
    negative_prompt_embeds: object
    cpu_prompt_encode_ms: float
    owner: object = dataclasses.field(repr=False)
    consumed: bool = False


@dataclasses.dataclass
class RequestState:
    request_id: str
    spec: dict
    step_index: int = 0  # Number of completed solver updates, not a timestep value.
    latents: object = None
    cache_enabled: bool = False
    events: list = dataclasses.field(default_factory=list)
    scheduler: object = None
    generator: object = None
    prompt_embeds: object = None
    negative_prompt_embeds: object = None
    contexts: dict = dataclasses.field(default_factory=dict)
    residency: str = "gpu"
    needs_anchor: bool = False
    finished: bool = False
    owner: object = None
    pending_snapshot: object = None
    prepare_ms: float = 0.0
    current_step_refreshed: bool = False
    immutable_spec: dict = dataclasses.field(default_factory=dict)


class WanStepWorker:
    """One resident model, independent request schedulers/RNG/cache state."""

    def __init__(self, device="cuda:0", model_path=None, attention="math"):
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
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH).resolve()
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
        self.cache_dit.set_attn_backend(self.pipe, attention_backend="native")
        self._original_forward = self.pipe.transformer.forward
        self.cache_dit.enable_cache(self.pipe, cache_config=self._cache_config({
            "threshold": 0.08, "max_continuous": 3, "steps": 50, "guidance_scale": 5.0}),
            attention_backend="native")
        self._cached_forward = self.pipe.transformer.forward
        self.manager = self.pipe._context_manager
        if self.manager is not self.pipe.transformer._context_manager or self.manager.persistent_context:
            raise RuntimeError("unsupported DBCache context manager contract")
        self._context_names = tuple(self.pipe.transformer._context_names)
        if len(self._context_names) != 1:
            raise RuntimeError("expected one Wan1.3B cached blocks context")
        self._lock = threading.RLock()
        self._encoder_lock = threading.Lock()
        self._owner = object()
        self._states = {}
        self._detach_contexts()

    def _cache_config(self, spec):
        return self.cache_dit.DBCacheConfig(
            Fn_compute_blocks=8, Bn_compute_blocks=0, max_warmup_steps=8, warmup_interval=1,
            residual_diff_threshold=spec["threshold"], max_cached_steps=-1,
            max_continuous_cached_steps=spec["max_continuous"],
            enable_separate_cfg=spec["guidance_scale"] > 1, cfg_compute_first=False,
            cfg_diff_compute_separate=True, num_inference_steps=spec["steps"])

    def _contexts(self, spec):
        result = {}
        for name in self._context_names:
            context = self.manager.new_context(name=name, cache_config=self._cache_config(spec))
            self._validate_context(context)
            result[name] = context
        self._detach_contexts()
        return result

    @staticmethod
    def _validate_context(context):
        _require_fields(context, CACHED_CONTEXT_FIELDS, label="CachedContext")
        _require_fields(context.cache_config, DBCACHE_CONFIG_FIELDS, label="DBCacheConfig")
        _require_fields(context.extra_cache_config, EXTRA_CACHE_CONFIG_FIELDS, label="ExtraCacheConfig")
        if any(getattr(context, key, None) is not None for key in CALIBRATORS) or context.calibrator_config is not None:
            raise RuntimeError("calibrators are outside this checkpoint adapter contract")
        if context.cache_config.steps_computation_mask is not None or context.cache_config.force_refresh_step_hint is not None:
            raise RuntimeError("forced masks and context restart policies are unsupported")
        for key in CONTEXT_COUNTERS + ("buffers", "cached_steps", "cfg_cached_steps", "residual_diffs", "cfg_residual_diffs", "_init_args", "_init_kwargs"):
            if not hasattr(context, key):
                raise RuntimeError("incomplete DBCache context: " + key)

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
        normalized = normalize_spec(state.spec)
        if type(state.immutable_spec) is not dict or set(state.immutable_spec) != IMMUTABLE_SPEC_FIELDS:
            raise ValueError("RequestState.immutable_spec has an unsupported field schema")
        _validate_metadata(state.immutable_spec, "immutable_spec")
        if state.immutable_spec != {key: normalized[key] for key in IMMUTABLE_SPEC_FIELDS}:
            raise ValueError("immutable request metadata differs from prepared spec")
        # A released state's payload was deliberately removed. Ownership checks
        # still reject its further use, while the top-level schema remains exact.
        if state.residency == "released":
            return
        self._validate_scheduler(state.scheduler)
        if not isinstance(state.contexts, dict):
            raise TypeError("RequestState.contexts must be a context dictionary")
        for context in state.contexts.values():
            self._validate_context(context)

    def _detach_contexts(self):
        # Do not call clear_contexts: it clears the request-owned buffer dicts.
        self.manager._current_context = None
        self.manager._cached_context_manager = {}
        self.manager._current_step_refreshed = False

    def _check(self, state, resident=None):
        self._validate_request_schema(state)
        if state.owner is not self._owner or self._states.get(state.request_id) is not state:
            raise ValueError("request is not owned by this worker or was released")
        if state.residency == "released":
            raise ValueError("request was released")
        if resident is not None and state.residency != resident:
            raise ValueError("request must reside in " + resident + "; restore it first")

    def _sync(self):
        self.torch.cuda.synchronize(self.device)

    def _dit_attention_context(self):
        """Force one native SDPA kernel; unsupported shapes/devices fail loudly."""
        attention = self.torch.nn.attention
        return attention.sdpa_kernel([getattr(attention.SDPBackend, ATTENTION_BACKENDS[self.attention])])

    def _memory(self):
        allocated = int(self.torch.cuda.memory_allocated(self.device))
        reserved = int(self.torch.cuda.memory_reserved(self.device))
        free, total = self.torch.cuda.mem_get_info(self.device)
        return {"allocated_bytes": allocated, "reserved_bytes": reserved,
                "driver_free_bytes": int(free), "device_total_bytes": int(total),
                "allocator_unused_reserved_bytes": max(0, reserved - allocated),
                "allocation_headroom_upper_bound_bytes": int(free) + max(0, reserved - allocated),
                "headroom_semantics": "driver free plus allocator unused reservation; fragmentation and concurrent allocations may reduce usability"}

    @contextlib.contextmanager
    def _activate(self, state):
        self._check(state, "gpu")
        self.manager._cached_context_manager = dict(state.contexts)
        self.manager._current_context = None
        self.manager._current_step_refreshed = state.current_step_refreshed
        original_configs = []
        for context in state.contexts.values():
            self._validate_context(context)
            context.cache_config.residual_diff_threshold = state.spec["threshold"]
            context.cache_config.max_continuous_cached_steps = state.spec["max_continuous"]
            # Config is request-local; implicit refresh must use the same policy.
            context._init_kwargs["cache_config"] = context.cache_config
            if state.needs_anchor or state.spec["threshold"] == 0:
                original_configs.append((context.cache_config, context.cache_config.max_cached_steps))
                context.cache_config.max_cached_steps = 0
        forward = self._cached_forward if state.cache_enabled else self._original_forward
        def selected_forward(*args, **kwargs):
            # Scope the override to DiT calls so the official reference loop's
            # FP32 VAE still executes under its surrounding MATH context.
            with self._dit_attention_context():
                return forward(*args, **kwargs)
        self.pipe.transformer.forward = selected_forward
        try:
            yield
        finally:
            state.current_step_refreshed = self.manager._current_step_refreshed
            for config, limit in original_configs:
                config.max_cached_steps = limit
            self.pipe.transformer.forward = self._cached_forward
            self._detach_contexts()

    def prepare(self, spec):
        """Synchronous compatibility path using the same per-request encoder."""
        return self.prepare_encoded(spec, self.encode_prompt(spec))

    def encode_prompt(self, spec):
        """Encode on CPU without acquiring the GPU lock or changing SDPA flags.

        The pinned UMT5 encoder uses explicit matmul/softmax attention. Changing
        global SDPA settings here would race a concurrently executing DiT/VAE.
        No embedding is reused across requests, including identical prompts.
        """
        spec = normalize_spec(spec)
        with self._encoder_lock, self.torch.inference_mode():
            started = time.perf_counter()
            prompt, negative = self.pipe.encode_prompt(
                prompt=spec["prompt"], negative_prompt=spec["negative_prompt"],
                do_classifier_free_guidance=spec["guidance_scale"] > 1,
                num_videos_per_prompt=1, max_sequence_length=512, device="cpu", dtype=self.torch.float32)
            self._validate_cpu_embeddings(prompt, negative, spec)
            return EncodedPrompt(spec=spec, prompt_embeds=prompt, negative_prompt_embeds=negative,
                                 cpu_prompt_encode_ms=(time.perf_counter() - started) * 1000,
                                 owner=self._owner)

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
            state = RequestState(spec["request_id"], spec, latents=latents, cache_enabled=spec["cache"],
                                 scheduler=scheduler, generator=generator, prompt_embeds=prompt,
                                 negative_prompt_embeds=negative,
                                 contexts=self._contexts(spec) if spec["cache"] else {}, owner=self._owner)
            state.immutable_spec = {key: value for key, value in spec.items()
                                    if key not in ("cache", "threshold", "max_continuous")}
            self._validate_request_schema(state)
            self._sync()
            gpu_prepare_ms = (time.perf_counter() - started) * 1000
            state.prepare_ms = payload.cpu_prompt_encode_ms + gpu_prepare_ms
            state.events.append({"kind": "prepare", "ms": state.prepare_ms,
                                 "cpu_prompt_encode_ms": payload.cpu_prompt_encode_ms,
                                 "gpu_prepare_ms": gpu_prepare_ms})
            self._states[state.request_id] = state
            payload.consumed = True
            return state

    def _path(self, state, index):
        contexts = []
        for context in state.contexts.values():
            contexts.append({"name": context.name, "cond_reused": index in context.cached_steps,
                             "uncond_reused": index in context.cfg_cached_steps,
                             "cond_drift": context.residual_diffs.get(str(index)),
                             "uncond_drift": context.cfg_residual_diffs.get(str(index)),
                             "executed_steps": context.executed_steps,
                             "transformer_executed_steps": context.transformer_executed_steps})
        reused = sum(int(item["cond_reused"]) + int(item["uncond_reused"]) for item in contexts)
        return {"cache_contexts": contexts, "reused_forwards": reused,
                "path_scope": "DBCache middle blocks; first/last configured blocks still execute",
                "path": "reuse" if reused == (2 if state.spec["guidance_scale"] > 1 else 1) else "mixed" if reused else "compute"}

    def step(self, state):
        with self._lock, self.torch.inference_mode(), self.p._sdpa_math_context(self.torch):
            self._check(state, "gpu")
            if state.finished or state.step_index >= state.spec["steps"]:
                raise ValueError("request has no remaining denoising steps")
            self._update_policy(state)
            self._sync()
            started = time.perf_counter()
            index = state.step_index
            anchor = state.needs_anchor
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
                state.needs_anchor = False
            self._sync()
            finite = bool(self.torch.isfinite(state.latents).all().item())
            ms = (time.perf_counter() - started) * 1000
            if not finite:
                raise RuntimeError("nonfinite latent after solver step %d" % index)
            record = {"kind": "step", "step_index": index, "completed_steps": state.step_index,
                      "ms": ms, "forced_cold_anchor": anchor, "finite": finite,
                      "threshold": state.spec["threshold"], "max_continuous": state.spec["max_continuous"],
                      **self._path(state, index)}
            if anchor and record["reused_forwards"]:
                raise RuntimeError("cold restore unexpectedly reused a missing cache")
            for context in state.contexts.values():
                expected = state.step_index * (2 if state.spec["guidance_scale"] > 1 else 1)
                if context.executed_steps != state.step_index or context.transformer_executed_steps != expected:
                    raise RuntimeError("cache step counters diverged from solver")
            state.events.append(record)
            return record

    def _update_policy(self, state):
        normalized = normalize_spec(state.spec)
        identity = {key: value for key, value in normalized.items()
                    if key not in ("cache", "threshold", "max_continuous")}
        if identity != state.immutable_spec:
            raise ValueError("only cache, threshold, and max_continuous may change after prepare")
        if normalized["cache"] == state.cache_enabled:
            return
        state.cache_enabled = normalized["cache"]
        state.contexts = self._contexts(normalized) if state.cache_enabled else {}
        state.needs_anchor = state.cache_enabled
        for context in state.contexts.values():
            context.executed_steps = context.accumulated_executed_steps = state.step_index
            forwards = state.step_index * (2 if normalized["guidance_scale"] > 1 else 1)
            context.transformer_executed_steps = context.accumulated_transformer_executed_steps = forwards
        state.events.append({"kind": "cache_policy_switch", "completed_steps": state.step_index,
                             "cache_enabled": state.cache_enabled, "old_features_discarded": True})

    def _payload(self, state):
        self._validate_request_schema(state)
        return {"latents": state.latents, "scheduler": vars(state.scheduler),
                "prompt_embeds": state.prompt_embeds, "negative_prompt_embeds": state.negative_prompt_embeds,
                "contexts": state.contexts, "rng_state": state.generator.get_state()}

    def _install_payload(self, state, payload):
        state.latents = payload["latents"]
        state.prompt_embeds = payload["prompt_embeds"]
        state.negative_prompt_embeds = payload["negative_prompt_embeds"]
        vars(state.scheduler).clear()
        vars(state.scheduler).update(payload["scheduler"])
        state.contexts = payload["contexts"]
        state.generator.set_state(payload["rng_state"])

    def _inventory(self, state):
        seen, tensors, totals = set(), set(), {"l0_bytes": 0, "l1_bytes": 0, "cuda_bytes": 0, "tensor_count": 0,
                                               "l0_logical_bytes": 0, "l1_logical_bytes": 0, "cuda_logical_bytes": 0,
                                               "logical_tensor_count": 0}
        def count(tensor, owner):
            if id(tensor) not in tensors:
                tensors.add(id(tensor))
                logical = int(tensor.numel() * tensor.element_size())
                totals[owner.replace("_bytes", "_logical_bytes")] += logical
                totals["cuda_logical_bytes"] += logical if tensor.device.type == "cuda" else 0
                totals["logical_tensor_count"] += 1
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                totals[owner] += int(storage.nbytes())
                totals["cuda_bytes"] += int(storage.nbytes()) if tensor.device.type == "cuda" else 0
                totals["tensor_count"] += 1
            return tensor
        payload = self._payload(state)
        contexts = payload.pop("contexts")
        _tree_map(payload, self.torch.is_tensor, lambda tensor: count(tensor, "l0_bytes"))
        _tree_map(contexts, self.torch.is_tensor, lambda tensor: count(tensor, "l1_bytes"))
        totals["total_bytes"] = totals["l0_bytes"] + totals["l1_bytes"]
        totals["storage_bytes"] = totals["total_bytes"]
        totals["logical_bytes"] = totals["l0_logical_bytes"] + totals["l1_logical_bytes"]
        totals["unique_storage_count"] = totals["tensor_count"]
        return totals

    def snapshot(self, state, mode="keep"):
        if mode not in ("keep", "hot", "cold"):
            raise ValueError("snapshot mode must be keep, hot, or cold")
        with self._lock, self.torch.inference_mode():
            self._check(state, "gpu")
            if state.pending_snapshot is not None:
                raise ValueError("request already has a pending snapshot")
            self._sync()
            started = time.perf_counter()
            before = self._memory()
            inventory = self._inventory(state)
            transfer_bytes = 0
            placements = {}
            if mode != "keep":
                # Clone metadata first; cold mode drops only optional features.
                payload = self._payload(state)
                if mode == "cold":
                    payload["contexts"] = {name: copy.copy(context) for name, context in state.contexts.items()}
                    for context in payload["contexts"].values():
                        context.buffers = {}
                def offload(tensor):
                    nonlocal transfer_bytes
                    if tensor.device.type != "cuda":
                        return tensor.detach().clone()
                    host = self.torch.empty_like(tensor, device="cpu", pin_memory=True)
                    host.copy_(tensor, non_blocking=True)
                    placements[id(host)] = str(tensor.device)
                    transfer_bytes += int(tensor.numel() * tensor.element_size())
                    return host
                host_payload = _tree_map(payload, self.torch.is_tensor, offload)
                self._sync()  # Source storage may be freed only after all D2H copies finish.
                self._install_payload(state, host_payload)
                state.pending_snapshot = {"mode": mode, "placements": placements,
                                          "rng_state": host_payload["rng_state"]}
                state.residency = "cpu"
                state.needs_anchor = state.needs_anchor or (mode == "cold" and state.cache_enabled)
                # payload retained old GPU references until this point.
                del payload, host_payload
            self._sync()
            record = {"kind": "snapshot", "mode": mode, "ms": (time.perf_counter() - started) * 1000,
                      "transfer_bytes": transfer_bytes, "bytes": transfer_bytes, "state_bytes": inventory,
                      "memory_before": before, "memory_after": self._memory(), "residency": state.residency}
            state.events.append(record)
            return record

    def restore(self, state):
        with self._lock, self.torch.inference_mode():
            self._check(state)
            self._sync()
            started = time.perf_counter()
            before = self._memory()
            transfer_bytes = 0
            mode = "keep"
            if state.residency == "cpu":
                saved = state.pending_snapshot
                if saved is None:
                    raise RuntimeError("CPU request has no snapshot placement metadata")
                mode = saved["mode"]
                placements = saved["placements"]
                def onload(tensor):
                    nonlocal transfer_bytes
                    target = placements.get(id(tensor))
                    if target is None:
                        return tensor
                    transfer_bytes += int(tensor.numel() * tensor.element_size())
                    return tensor.to(device=target, non_blocking=True)
                payload = self._payload(state)
                payload["rng_state"] = saved["rng_state"]
                gpu_payload = _tree_map(payload, self.torch.is_tensor, onload)
                self._sync()
                self._install_payload(state, gpu_payload)
                state.pending_snapshot = None
                state.residency = "gpu"
            record = {"kind": "restore", "mode": mode, "ms": (time.perf_counter() - started) * 1000,
                      "transfer_bytes": transfer_bytes, "bytes": transfer_bytes,
                      "memory_before": before, "memory_after": self._memory(),
                      "needs_anchor": state.needs_anchor}
            state.events.append(record)
            return record

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
            self._detach_contexts()
            state.latents = state.prompt_embeds = state.negative_prompt_embeds = None
            state.scheduler = state.generator = state.pending_snapshot = None
            state.contexts = {}
            state.residency = "released"
            del self._states[state.request_id]

    def reference(self, spec):
        """Run the unmodified official denoising/decode loop with the same inputs.

        Inputs are prepared by the same official encode_prompt/prepare_latents
        methods. The official __call__ is invoked below its automatic cache reset
        wrapper; request-local contexts are supplied by this adapter. This tests
        arithmetic/step parity, not an independent implementation of DBCache.
        """
        with self._lock:
            state = self.prepare(spec)
            old_scheduler = self.pipe.scheduler
            captured = {}
            paths = []
            def callback(pipe, index, timestep, kwargs):
                paths.append(dict(step_index=int(index), **self._path(state, int(index))))
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
                return {"output": output.frames, "latents": captured["latents"], "path_history": paths,
                        "elapsed_ms": state.prepare_ms + ms, "prepare_ms": state.prepare_ms,
                        "pipeline_ms": ms, "instrumented": True,
                        "reference_scope": "official Wan loop with official prepared inputs; final latent D2H capture included"}
            finally:
                self.pipe.scheduler = old_scheduler
                self.pipe._current_timestep = None
                self.release(state)
