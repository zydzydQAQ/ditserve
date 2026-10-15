"""Exact model adapters with explicit V100 and A100 session recipes.

Imports do not load models or initialize CUDA.  Mistral uses greedy FP16 SDPA
with the MATH backend; Sana keeps the existing full-FP32 MATH recipe.  KV is an
exact autoregressive state, not diffusion approximation caching.  State-only
offload is explicit and is not used automatically by either adapter. Selecting
the A100 recipe does not carry over V100 study qualification or authorization.
"""

from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import importlib
from pathlib import Path
import threading
import time

from ..directions.sana import RECIPE


def _torch():
    return importlib.import_module("torch")


def _cpu_copy(value):
    return None if value is None else value.detach().cpu().clone()


def tensor_comparison(actual, expected, *, atol=0., rtol=0.):
    """CPU comparison; callers keep copies and reductions outside serving timing."""
    torch = _torch()
    if actual is None or expected is None:
        return dict(passed=actual is expected, exact=actual is expected, finite=True,
                    shape_equal=actual is expected, dtype_equal=actual is expected, max_abs=None, rmse=None)
    a, b = torch.as_tensor(actual).detach().cpu(), torch.as_tensor(expected).detach().cpu()
    shape, dtype = tuple(a.shape) == tuple(b.shape), a.dtype == b.dtype
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not shape:
        return dict(passed=False, exact=False, finite=finite, shape_equal=False,
                    dtype_equal=dtype, max_abs=None, rmse=None)
    a, b = a.double(), b.double()
    delta = a - b
    exact = dtype and bool(torch.equal(a, b))
    return dict(passed=shape and dtype and finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
                exact=exact, finite=finite, shape_equal=shape, dtype_equal=dtype,
                max_abs=float(delta.abs().max()) if delta.numel() else 0.,
                rmse=float(delta.square().mean().sqrt()) if delta.numel() else 0.)


def _legacy(cache):
    if cache is None:
        return ()
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else tuple(cache)


def cache_snapshot(cache):
    return tuple(tuple(_cpu_copy(value) for value in pair) for pair in _legacy(cache))


def cache_comparison(actual, expected, *, atol=0., rtol=0.):
    a, b = _legacy(actual), _legacy(expected)
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return dict(passed=False, exact=False, layers=len(a), expected_layers=len(b), tensors=[])
    checks = [dict(layer=i, component=j, **tensor_comparison(x, y, atol=atol, rtol=rtol))
              for i, (left, right) in enumerate(zip(a, b)) for j, (x, y) in enumerate(zip(left, right))]
    return dict(passed=all(row["passed"] for row in checks), exact=all(row["exact"] for row in checks),
                layers=len(a), expected_layers=len(b), tensors=checks)


def _inventory(tensors, **metadata):
    rows, seen, unique_bytes = [], set(), 0
    for name, tensor in tensors:
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if identity not in seen:
            unique_bytes += storage.nbytes()
            seen.add(identity)
        rows.append(dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                         device=str(tensor.device), bytes=tensor.numel() * tensor.element_size()))
    return dict(metadata, tensors=rows, bytes=unique_bytes, device=metadata.get("location"),
                tensor_bytes=sum(row["bytes"] for row in rows),
                unique_storage_bytes=unique_bytes,
                gpu_bytes=sum(row["bytes"] for row in rows if row["device"].startswith("cuda")),
                cpu_bytes=sum(row["bytes"] for row in rows if row["device"] == "cpu"))


@dataclass(eq=False)
class LLMState:
    owner: object = field(repr=False)
    request_id: str
    prompt_ids: list
    max_new_tokens: int
    cache: object = field(repr=False)
    messages: list = field(default_factory=list)
    cached_ids: list = field(default_factory=list)
    generated_ids: list = field(default_factory=list)
    action: str = "prefill"
    done: bool = False
    failed: bool = False
    closed: bool = False
    location: str = "cpu"
    reused_tokens: int = 0
    prepare_ms: float = 0.
    last_ms: float = 0.

    @property
    def cached_tokens(self):
        return len(self.cached_ids)


@dataclass(eq=False)
class SanaState:
    owner: object = field(repr=False)
    request_id: str
    prompt: str
    seed: int
    scheduler: object = field(repr=False)
    generator: object = field(repr=False)
    prompt_embeds: object = field(repr=False)
    prompt_attention_mask: object = field(repr=False)
    latents: object = field(repr=False)
    guidance: object = field(repr=False)
    denoised: object = field(default=None, repr=False)
    step_index: int = 0
    num_steps: int = 2
    failed: bool = False
    closed: bool = False
    finished: bool = False
    location: str = "cpu"
    prepare_ms: float = 0.
    decode_ms: float = 0.
    last_ms: float = 0.
    output: object = field(default=None, repr=False)

    @property
    def done(self):
        return self.step_index == self.num_steps


class _Runtime:
    def _initialize_runtime(self, device):
        self.torch = _torch()
        self.device = str(self.torch.device(device))
        self._owner, self._states = object(), {}
        self._lock = threading.RLock()

    def _sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def _attention(self):
        return self.torch.nn.attention.sdpa_kernel([self.torch.nn.attention.SDPBackend.MATH])

    @contextmanager
    def _action(self, state=None):
        with self._lock:
            self._sync()
            started = time.perf_counter()
            try:
                with self.torch.inference_mode(), self._attention():
                    yield
                self._sync()
            except BaseException:
                if state is not None:
                    state.failed = True
                raise
            finally:
                self._last_ms = (time.perf_counter() - started) * 1000
                if state is not None:
                    state.last_ms = self._last_ms

    def _check(self, state, *, resident=True, allow_failed=False):
        if (state.owner is not self._owner or self._states.get(id(state)) is not state
                or state.closed or (state.failed and not allow_failed)):
            raise ValueError("inactive, failed, closed or foreign session state")
        if resident and state.location != self.device:
            raise ValueError("state is offloaded; restore explicitly before execution")

    def _finite(self, *tensors):
        if any(not bool(self.torch.isfinite(value).all()) for value in tensors if value is not None):
            raise RuntimeError("nonfinite session model output")

    def _validate_profile(self, profile):
        capabilities = {"v100_fp32": (7, 0), "a100_fp32": (8, 0)}
        if profile not in capabilities:
            raise ValueError("unknown explicit session hardware profile")
        if (not self.device.startswith("cuda")
                or self.torch.cuda.get_device_capability(self.device) != capabilities[profile]):
            raise ValueError(f"session {profile} requires CUDA sm{capabilities[profile][0]}0")
        self.profile = profile


class MistralRuntime(_Runtime):
    """One shared Mistral model and independent, single-owner greedy KV states."""

    def __init__(self, model_path, device="cuda:0", *, profile="v100_fp32"):
        self._initialize_runtime(device)
        self._validate_profile(profile)
        path = Path(model_path).resolve()
        transformers = importlib.import_module("transformers")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, torch_dtype=self.torch.float16,
            attn_implementation="sdpa", use_safetensors=True).eval().to(self.device)
        if type(self.model).__name__ != "MistralForCausalLM" or self.model.config.sliding_window is not None:
            raise ValueError("pilot requires full-context MistralForCausalLM without sliding KV")
        if getattr(self.model, "is_quantized", False) or getattr(self.model.config, "quantization_config", None):
            raise ValueError("quantized Mistral execution is outside this pilot")
        if any(p.is_floating_point() and p.dtype != self.torch.float16 for p in self.model.parameters()):
            raise ValueError("Mistral must remain unquantized FP16")
        self.model_path = str(path)
        self._configure()

    def _configure(self):
        self.DynamicCache = importlib.import_module("transformers.cache_utils").DynamicCache
        eos = self.model.generation_config.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, (list, tuple)) else [] if eos is None else [eos])

    def _tokens(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids):
            raise ValueError("chat template must produce one nonempty unbatched token sequence")
        return ids

    def prepare(self, messages, *, max_new_tokens=64, previous=None, request_id="llm"):
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        started = time.perf_counter()
        ids = self._tokens(messages)
        if len(ids) + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("session exceeds the qualified context window")
        with self._lock:
            cache, prefix = self.DynamicCache(config=self.model.config), []
            if previous is not None:
                self._check(previous)
                if not previous.done:
                    raise ValueError("previous turn must finish before KV ownership transfers")
                if previous.cache.get_seq_length() != previous.cached_tokens:
                    raise ValueError("previous turn KV differs from its committed token prefix")
                common = 0
                # Keep one input token for the first forward even for an identical prompt.
                for old, new in zip(previous.cached_ids, ids[:-1]):
                    if old != new:
                        break
                    common += 1
                if common:
                    cache = previous.cache
                    cache.crop(common)
                    prefix = previous.cached_ids[:common]
                previous.cache = None
                previous.closed = True
                self._states.pop(id(previous))
            state = LLMState(self._owner, request_id, ids, max_new_tokens, cache,
                             messages=copy.deepcopy(messages), cached_ids=prefix,
                             location=self.device, reused_tokens=len(prefix),
                             prepare_ms=(time.perf_counter() - started) * 1000)
            self._states[id(state)] = state
            return state

    def advance(self, state):
        with self._lock:
            self._check(state)
            if state.done:
                raise ValueError("LLM turn is already complete")
            action = state.action
            incoming = (state.prompt_ids[state.cached_tokens:] if action == "prefill"
                        else state.generated_ids[-1:])
            if not incoming or state.cache.get_seq_length() != state.cached_tokens:
                raise ValueError("KV sequence length does not match the committed token prefix")
            with self._action(state):
                torch = self.torch
                positions = torch.arange(state.cached_tokens, state.cached_tokens + len(incoming),
                                         device=self.device, dtype=torch.long)
                result = self.model(
                    input_ids=torch.tensor([incoming], device=self.device, dtype=torch.long),
                    attention_mask=torch.ones((1, state.cached_tokens + len(incoming)),
                                              device=self.device, dtype=torch.long),
                    position_ids=positions.unsqueeze(0), cache_position=positions,
                    past_key_values=state.cache, use_cache=True, logits_to_keep=1, return_dict=True)
                self._finite(result.logits)
                token = int(result.logits[0, -1].argmax())
                state.cache = result.past_key_values
                state.cached_ids.extend(incoming)
                if state.cache.get_seq_length() != state.cached_tokens:
                    raise RuntimeError("Mistral returned an inconsistent KV sequence length")
                state.generated_ids.append(token)
                state.done = token in self.eos_token_ids or len(state.generated_ids) >= state.max_new_tokens
                state.action = "decode"
            return dict(action=action, ms=state.last_ms, token_id=token, done=state.done,
                        generated_tokens=len(state.generated_ids), cached_tokens=state.cached_tokens)

    def finish(self, state):
        self._check(state, resident=False)
        if not state.done:
            raise ValueError("LLM turn has not reached its stop condition")
        return self.tokenizer.decode(state.generated_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)

    def reference(self, messages, *, max_new_tokens=64):
        ids = self._tokens(messages)
        if type(max_new_tokens) is not int or max_new_tokens <= 0 or len(ids) + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("invalid reference generation length")
        with self._action():
            inputs = self.torch.tensor([ids], device=self.device, dtype=self.torch.long)
            config = copy.deepcopy(self.model.generation_config)
            config.do_sample, config.num_beams, config.use_cache = False, 1, True
            config.max_new_tokens = max_new_tokens
            config.pad_token_id = config.pad_token_id if config.pad_token_id is not None else self.tokenizer.eos_token_id
            result = self.model.generate(input_ids=inputs, attention_mask=self.torch.ones_like(inputs),
                generation_config=config, return_dict_in_generate=True, output_scores=False,
                logits_to_keep=1)
            tokens = result.sequences[0, len(ids):].tolist()
        ms = self._last_ms
        return dict(text=self.tokenizer.decode(tokens, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False),
                    token_ids=tokens, input_ids=ids, cache=cache_snapshot(result.past_key_values), ms=ms)

    def snapshot(self, state):
        self._check(state, resident=False)
        return dict(prompt_ids=list(state.prompt_ids), generated_ids=list(state.generated_ids),
                    cached_ids=list(state.cached_ids), done=state.done, cache=cache_snapshot(state.cache))

    def inventory(self, state):
        self._check(state, resident=False, allow_failed=True)
        tensors = [("kv.%d.%s" % (i, name), tensor) for i, pair in enumerate(_legacy(state.cache))
                   for name, tensor in zip(("key", "value"), pair)]
        return _inventory(tensors, kind="llm", location=state.location, done=state.done,
                          cached_tokens=state.cached_tokens, generated_tokens=len(state.generated_ids),
                          reused_tokens=state.reused_tokens, action=state.action)

    def _move(self, state, destination):
        self._check(state, resident=False)
        if state.location == destination:
            return dict(ms=0., bytes=0, location=destination)
        amount = self.inventory(state)["tensor_bytes"]
        with self._action(state):
            if state.cached_tokens:
                pairs = tuple(tuple(tensor.to(destination) for tensor in pair) for pair in _legacy(state.cache))
                state.cache = self.DynamicCache.from_legacy_cache(pairs)
            else:
                # Configured DynamicCache has lazy layers containing None before
                # prefill; passing those to from_legacy_cache is not valid.
                state.cache = self.DynamicCache(config=self.model.config)
            state.location = destination
        return dict(ms=state.last_ms, bytes=amount, location=destination)

    def offload_to_cpu(self, state):
        with self._lock:
            return self._move(state, "cpu")

    def restore(self, state):
        with self._lock:
            return self._move(state, self.device)

    def release(self, state):
        with self._lock:
            if state.owner is self._owner and state.closed:
                return
            self._check(state, resident=False, allow_failed=True)
            state.cache = None
            state.closed = True
            self._states.pop(id(state))

    def close(self):
        with self._lock:
            for state in list(self._states.values()):
                self.release(state)
            self._sync()
            self.model = None


class SanaRuntime(_Runtime):
    """Official Sana-Sprint arithmetic, with per-request SCM and CPU RNG state.

    The two step body below follows local diffusers SanaSprintPipeline.__call__.
    An independent official-pipeline reference is retained for GPU qualification.
    """

    _tensor_fields = ("prompt_embeds", "prompt_attention_mask", "latents", "denoised", "guidance")

    def __init__(self, model_path, device="cuda:0", *, profile="v100_fp32"):
        self._initialize_runtime(device)
        self._validate_profile(profile)
        from ..scheduling.runtime import SchedulingSana
        self.adapter = SchedulingSana(model_path, device, profile=profile)
        self.pipe = self.adapter.pipe
        self.model_path = str(Path(model_path).resolve())
        self._instruction = self.adapter._instruction
        self._configure()

    def _configure(self):
        module = importlib.import_module("diffusers.pipelines.sana.pipeline_sana_sprint")
        self._retrieve_timesteps = module.retrieve_timesteps
        self._aspect_ratios = module.ASPECT_RATIO_1024_BIN
        if type(self.pipe.scheduler).__name__ != "SCMScheduler" or self.pipe.transformer.config.sample_size != 32:
            raise ValueError("session Sana requires the pinned 1024px SCM recipe")

    def prepare(self, prompt, seed=0, *, request_id="sana"):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Sana prompt must be nonempty")
        with self._action():
            pipe, torch = self.pipe, self.torch
            height, width = pipe.image_processor.classify_height_width_bin(
                RECIPE["height"], RECIPE["width"], ratios=self._aspect_ratios)
            pipe.check_inputs(prompt=prompt, height=height, width=width,
                num_inference_steps=2, timesteps=None, max_timesteps=RECIPE["max_timesteps"],
                intermediate_timesteps=RECIPE["intermediate_timesteps"],
                callback_on_step_end_tensor_inputs=["latents"], prompt_embeds=None, prompt_attention_mask=None)
            embeddings, mask = pipe.encode_prompt(prompt=prompt, device=self.device,
                num_images_per_prompt=1, clean_caption=False, max_sequence_length=300,
                complex_human_instruction=self._instruction)
            scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
            timesteps, steps = self._retrieve_timesteps(scheduler, 2, self.device, None,
                sigmas=None, max_timesteps=RECIPE["max_timesteps"],
                intermediate_timesteps=RECIPE["intermediate_timesteps"])
            scheduler.set_begin_index(0)
            if steps != 2 or len(timesteps) != 3:
                raise RuntimeError("Sana two-step schedule changed")
            generator = torch.Generator(device="cpu").manual_seed(seed)
            latents = pipe.prepare_latents(1, pipe.transformer.config.in_channels, height, width,
                                           torch.float32, self.device, generator, None)
            latents = latents * scheduler.config.sigma_data
            guidance = torch.full([1], RECIPE["guidance_scale"], device=self.device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0]).to(embeddings.dtype)
            guidance = guidance * pipe.transformer.config.guidance_embeds_scale
            self._finite(embeddings, latents, guidance)
            state = SanaState(self._owner, request_id, prompt, seed, scheduler, generator,
                              embeddings, mask, latents, guidance, location=self.device)
        state.prepare_ms = self._last_ms
        with self._lock:
            self._states[id(state)] = state
        return state

    def advance(self, state):
        with self._lock:
            self._check(state)
            if state.done or state.finished:
                raise ValueError("Sana denoising is already complete")
            if state.scheduler.step_index not in (None, state.step_index):
                raise ValueError("Sana sampler index differs from request continuation")
            with self._action(state):
                torch, pipe = self.torch, self.pipe
                timestep = state.scheduler.timesteps[state.step_index].expand(state.latents.shape[0])
                latents_model_input = state.latents / state.scheduler.config.sigma_data
                scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
                expanded = scm_timestep.view(-1, 1, 1, 1)
                latent_model_input = latents_model_input * torch.sqrt(expanded**2 + (1 - expanded)**2)
                noise_pred = pipe.transformer(latent_model_input.to(dtype=pipe.transformer.dtype),
                    encoder_hidden_states=state.prompt_embeds.to(dtype=pipe.transformer.dtype),
                    encoder_attention_mask=state.prompt_attention_mask, guidance=state.guidance,
                    timestep=scm_timestep, return_dict=False, attention_kwargs=None)[0]
                noise_pred = ((1 - 2 * expanded) * latent_model_input
                    + (1 - 2 * expanded + 2 * expanded**2) * noise_pred) / torch.sqrt(expanded**2 + (1 - expanded)**2)
                noise_pred = noise_pred.float() * state.scheduler.config.sigma_data
                state.latents, state.denoised = state.scheduler.step(noise_pred, timestep,
                    state.latents, generator=state.generator, return_dict=False)
                self._finite(state.latents, state.denoised)
                state.step_index += 1
            return dict(action="denoise", ms=state.last_ms, step=state.step_index, done=state.done)

    def finish(self, state):
        with self._lock:
            self._check(state)
            if not state.done:
                raise ValueError("Sana must complete both denoise steps before decode")
            if state.finished:
                return state.output
            with self._action(state):
                pipe = self.pipe
                latents = (state.denoised / state.scheduler.config.sigma_data).to(pipe.vae.dtype)
                image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                self._finite(image)
                image = pipe.image_processor.resize_and_crop_tensor(image, RECIPE["width"], RECIPE["height"])
                state.output = pipe.image_processor.postprocess(image, output_type="np")
                if not importlib.import_module("numpy").isfinite(state.output).all():
                    raise RuntimeError("nonfinite Sana postprocessed image")
                state.finished = True
            state.decode_ms = state.last_ms
            return state.output

    def reference(self, prompt, seed=0):
        steps = []

        def callback(pipe, index, timestep, values):
            steps.append(_cpu_copy(values["latents"]))
            return values

        with self._action():
            output = self.pipe(prompt=prompt, generator=self.torch.Generator(device="cpu").manual_seed(seed),
                **RECIPE, output_type="np", callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"]).images
            if not importlib.import_module("numpy").isfinite(output).all() or len(steps) != 2:
                raise RuntimeError("invalid official Sana reference")
        return dict(output=output, step_latents=steps, ms=self._last_ms)

    def snapshot(self, state):
        self._check(state, resident=False)
        return dict(step_index=state.step_index, sampler_step=state.scheduler.step_index,
                    sampler_begin=state.scheduler.begin_index, done=state.done,
                    generator_state=state.generator.get_state().clone(),
                    timesteps=_cpu_copy(state.scheduler.timesteps),
                    **{name: _cpu_copy(getattr(state, name)) for name in self._tensor_fields})

    def inventory(self, state):
        self._check(state, resident=False, allow_failed=True)
        tensors = [(name, getattr(state, name)) for name in self._tensor_fields]
        tensors += [("sampler.timesteps", state.scheduler.timesteps), ("rng", state.generator.get_state())]
        return _inventory(tensors, kind="sana", location=state.location, step=state.step_index,
                          sampler_step=state.scheduler.step_index, done=state.done, finished=state.finished)

    def _move(self, state, destination):
        self._check(state, resident=False)
        if state.location == destination:
            return dict(ms=0., bytes=0, location=destination)
        amount = sum(row["bytes"] for row in self.inventory(state)["tensors"] if row["name"] != "rng")
        with self._action(state):
            for name in self._tensor_fields:
                value = getattr(state, name)
                if value is not None:
                    setattr(state, name, value.to(destination))
            state.scheduler.timesteps = state.scheduler.timesteps.to(destination)
            state.location = destination
        return dict(ms=state.last_ms, bytes=amount, location=destination)

    def offload_to_cpu(self, state):
        with self._lock:
            return self._move(state, "cpu")

    def restore(self, state):
        with self._lock:
            return self._move(state, self.device)

    def release(self, state):
        with self._lock:
            if state.owner is self._owner and state.closed:
                return
            self._check(state, resident=False, allow_failed=True)
            for name in self._tensor_fields:
                setattr(state, name, None)
            state.output = state.scheduler = state.generator = None
            state.closed = True
            self._states.pop(id(state))

    def close(self):
        with self._lock:
            for state in list(self._states.values()):
                self.release(state)
            self._sync()
            self.pipe = None
            if getattr(self, "adapter", None) is not None:
                self.adapter.pipe = None
