"""SD3.5 round execution over the frozen common xfuser pipeline.

Implements TetriServe's prologue/denoising five-tensor protocol using the actual
shared pipeline's _sync_pipeline, not a DitServe scheduling policy. The official
runner's communicator activation and model registry still require an integration
patch; this module alone is not a serving reproduction or a qualified result.

Five tensors carry embeddings (including negative conditioning), latents and the
complete Euler time/sigma schedule. Guidance and text-length are immutable worker
profile values; dimensions are explicit on every round. Deterministic first-order
FlowMatch Euler is the only supported solver. No solver history is discarded.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
from typing import Any

FROZEN_WORKER_SHA256 = 'fc493414a9ff589850ec25303c9166fb86e138af340f40dfd4caa62ed1b6995a'
SD35_MODEL = 'stabilityai/stable-diffusion-3.5-medium'
SD35_REVISION = 'b940f670f0eda2d07fbb75229e779da1ad11eb80'


def positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def round_window(start, end, total):
    positive_int('total steps', total)
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= total:
        raise ValueError('round must be a nonempty contiguous interval within the complete schedule')
    return slice(start, end), slice(start, end + 1)


def validate_schedule(timesteps, sigmas):
    """Pure protocol check, also used on real tensor values before execution."""
    if not timesteps or len(sigmas) != len(timesteps) + 1:
        raise ValueError('a round requires N timesteps and N+1 sigmas')
    for values in (timesteps, sigmas):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError('schedule values must be finite numbers')
        if any(a <= b for a, b in zip(values, values[1:])):
            raise ValueError('Euler schedule must be strictly decreasing')
    if sigmas[-1] < 0:
        raise ValueError('negative sigma')


def validate_solver(order, config):
    if type(order) is not int or order != 1 or config.get('stochastic_sampling', False):
        raise ValueError('five tensors require deterministic first-order Euler; solver/RNG history would be lost')


def cfg_order(batch_size):
    """Wire format [u0,c0,u1,c1] -> xfuser format [u0,u1,c0,c1]."""
    positive_int('batch size', batch_size)
    return tuple(range(0, 2 * batch_size, 2)) + tuple(range(1, 2 * batch_size, 2))


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    negative_prompt: str = ''
    height: int = 512
    width: int = 512
    steps: int = 28
    seed: int = 101
    guidance_scale: float = 4.5
    max_sequence_length: int = 256

    def __post_init__(self):
        if not isinstance(self.prompt, str) or not isinstance(self.negative_prompt, str):
            raise ValueError('prompt and negative prompt must be strings')
        for name in ('height', 'width', 'steps', 'max_sequence_length'):
            positive_int(name, getattr(self, name))
        if self.height % 16 or self.width % 16:
            raise ValueError('SD3.5 dimensions must be multiples of 16')
        if self.max_sequence_length > 512:
            raise ValueError('text length exceeds SD3.5 T5 limit')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        if (isinstance(self.guidance_scale, bool) or not isinstance(self.guidance_scale, (int, float))
                or not math.isfinite(self.guidance_scale) or self.guidance_scale <= 1):
            raise ValueError('this common SD3.5 CFG pipeline requires finite guidance > 1')


@dataclass(frozen=True)
class RoundState:
    request: ImageRequest
    prompt_embeds: Any
    pooled_prompt_embeds: Any
    latents: Any
    timesteps: Any
    sigmas: Any
    cursor: int = 0

    def __post_init__(self):
        if type(self.cursor) is not int or not 0 <= self.cursor <= self.request.steps:
            raise ValueError('invalid continuation cursor')
        if len(self.timesteps) != self.request.steps or len(self.sigmas) != self.request.steps + 1:
            raise ValueError('continuation must retain the complete original schedule')

    def window(self, end):
        a, b = round_window(self.cursor, end, self.request.steps)
        return self.timesteps[a], self.sigmas[b]

    def transferred_tensors(self):
        return (self.prompt_embeds, self.pooled_prompt_embeds, self.latents, self.timesteps, self.sigmas)


def verify_shared_sources(root):
    root = Path(root).resolve(strict=True)
    inventory = {}
    for name in ('ditserve', 'xfuser'):
        for path in sorted((root / name).rglob('*.py')):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('linked shared backend source')
            inventory[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    sha = hashlib.sha256(json.dumps(inventory, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if sha != FROZEN_WORKER_SHA256:
        raise ValueError('shared backend differs from frozen 51740c0 worker sources')
    return root


class SD35RoundPipeline:
    """Actual inference adapter; scheduler/worker dispatch remains external.

    The runtime argument is the frozen common XDitRuntime (model plumbing only).
    Its process groups must already correspond to the official assigned ranks.
    This class never creates a serving scheduler or decides rank allocation.
    """
    def __init__(self, shared_runtime, *, guidance_scale=4.5, max_sequence_length=256, provenance=None):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from xfuser.model_executor.schedulers.base_scheduler import xFuserSchedulerBaseWrapper
        from xfuser.model_executor.pipelines.pipeline_stable_diffusion_3 import xFuserStableDiffusion3Pipeline

        ImageRequest(prompt='', guidance_scale=guidance_scale, max_sequence_length=max_sequence_length)
        if not isinstance(shared_runtime.pipe, xFuserStableDiffusion3Pipeline):
            raise ValueError('a real shared xFuserStableDiffusion3Pipeline is required')
        self.runtime = shared_runtime
        self.pipe = shared_runtime.pipe
        self.raw = self.pipe.module
        scheduler = self.raw.scheduler
        if isinstance(scheduler, xFuserSchedulerBaseWrapper):
            scheduler = scheduler.module
        if type(scheduler) is not FlowMatchEulerDiscreteScheduler:
            raise ValueError('only the unmodified FlowMatchEulerDiscreteScheduler is supported')
        validate_solver(scheduler.order, scheduler.config)
        self.torch = torch
        self.scheduler_class = FlowMatchEulerDiscreteScheduler
        self.scheduler_config = dict(scheduler.config)
        self.guidance_scale = float(guidance_scale)
        self.max_sequence_length = max_sequence_length
        self.provenance = dict(provenance or {})
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(cls, model_path, *, shared_python_root, model_manifest_path,
                        model_manifest_sha256, device='cuda', groups=None, attention='math',
                        guidance_scale=4.5, max_sequence_length=256):
        """Load verified local weights into the real frozen shared xfuser backend."""
        root = verify_shared_sources(shared_python_root)
        sys.path.insert(0, str(root))
        for name, module in list(sys.modules.items()):
            if name == 'xfuser' or name.startswith('xfuser.') or name == 'ditserve' or name.startswith('ditserve.'):
                origin = getattr(module, '__file__', None)
                if origin and not Path(origin).resolve().is_relative_to(root):
                    raise ValueError('another worker/backend package is already imported: ' + name)
        import torch
        from diffusers import StableDiffusion3Pipeline
        from ditserve.models.xdit.sd35 import attach
        from ditserve.runtime.model_integrity import verify_model_manifest
        model_path = Path(model_path).resolve(strict=True)
        raw_manifest = Path(model_manifest_path).read_bytes()
        if hashlib.sha256(raw_manifest).hexdigest() != model_manifest_sha256:
            raise ValueError('model manifest SHA256 differs')
        manifest = json.loads(raw_manifest)
        if manifest.get('model') != SD35_MODEL or manifest.get('revision') != SD35_REVISION:
            raise ValueError('model must be the frozen official SD3.5 medium revision')
        integrity = verify_model_manifest(str(model_path), str(model_manifest_path), model_manifest_sha256)
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, local_files_only=True).to(device)
        runtime = attach(pipeline, device, attention=attention, groups=groups)
        return cls(runtime, guidance_scale=guidance_scale, max_sequence_length=max_sequence_length,
                   provenance={'shared_worker_sha256': FROZEN_WORKER_SHA256,
                               'model_integrity': integrity, 'model_revision': SD35_REVISION})

    @contextmanager
    def _execution(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('concurrent calls cannot share mutable xfuser pipeline runtime')
        try:
            with self.torch.inference_mode():
                yield
        finally:
            self._lock.release()

    def _profile(self, guidance_scale=None, max_sequence_length=None):
        if guidance_scale is not None and guidance_scale != self.guidance_scale:
            raise ValueError('guidance differs from immutable worker continuation profile')
        if max_sequence_length is not None and max_sequence_length != self.max_sequence_length:
            raise ValueError('text length differs from immutable worker continuation profile')

    def dummy_inference_prologue(self, *, guidance_scale=None, max_sequence_length=None):
        # No hidden request state is restored here; metadata is explicit/immutable.
        self._profile(guidance_scale, max_sequence_length)

    def inference_prologue(self, prompt, height, width, num_inference_steps, generator,
                           *, negative_prompt='', guidance_scale=None, max_sequence_length=None):
        self._profile(guidance_scale, max_sequence_length)
        positive_int('steps', num_inference_steps)
        if not isinstance(generator, self.torch.Generator) or str(generator.device) != 'cpu':
            raise ValueError('formal paired execution requires an explicit CPU torch.Generator')
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        negatives = [negative_prompt] * len(prompts) if isinstance(negative_prompt, str) else list(negative_prompt)
        if not prompts or len(prompts) != len(negatives):
            raise ValueError('prompt/negative prompt batch mismatch')
        for p, n in zip(prompts, negatives):
            ImageRequest(p, n, height, width, num_inference_steps,
                         guidance_scale=self.guidance_scale, max_sequence_length=self.max_sequence_length)
        with self._execution():
            device = self.raw._execution_device
            pe, ne, pp, np = self.raw.encode_prompt(
                prompt=prompts, prompt_2=None, prompt_3=None,
                negative_prompt=negatives, negative_prompt_2=None, negative_prompt_3=None,
                do_classifier_free_guidance=True, device=device, num_images_per_prompt=1,
                max_sequence_length=self.max_sequence_length)
            # Per-request adjacent pairs survive the official runner's chunk/cat transfer.
            packed_pe = self.torch.stack([ne, pe], dim=1).flatten(0, 1)
            packed_pp = self.torch.stack([np, pp], dim=1).flatten(0, 1)
            latents = self.raw.prepare_latents(len(prompts), self.pipe.transformer.config.in_channels,
                height, width, pe.dtype, device, generator, None)
            scheduler = self.scheduler_class.from_config(self.scheduler_config)
            kwargs = {}
            if self.scheduler_config.get('use_dynamic_shifting', False):
                # Use the actual diffusers 0.36 implementation and model configuration.
                from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift
                patch = self.pipe.transformer.config.patch_size
                tokens = (latents.shape[-2] // patch) * (latents.shape[-1] // patch)
                kwargs['mu'] = calculate_shift(tokens,
                    self.scheduler_config.get('base_image_seq_len', 256),
                    self.scheduler_config.get('max_image_seq_len', 4096),
                    self.scheduler_config.get('base_shift', .5),
                    self.scheduler_config.get('max_shift', 1.16))
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # NCCL migration requires CUDA tensors, including the schedule.
            timesteps = scheduler.timesteps.to(device)
            sigmas = scheduler.sigmas.to(device)
            validate_schedule(timesteps.detach().cpu().tolist(), sigmas.detach().cpu().tolist())
            return packed_pe, packed_pp, latents, timesteps, sigmas

    def inference_denoising(self, *, height, width, batch_size, num_inference_steps,
                            timesteps, sigmas, latents, prompt_embeds, pooled_prompt_embeds,
                            is_last_steps=None, output_type='pil', guidance_scale=None):
        self._profile(guidance_scale)
        positive_int('batch size', batch_size)
        positive_int('steps', num_inference_steps)
        ImageRequest('', height=height, width=width, steps=num_inference_steps,
                     guidance_scale=self.guidance_scale, max_sequence_length=self.max_sequence_length)
        if output_type not in ('pil', 'pt', 'latent'):
            raise ValueError('unsupported output type')
        if is_last_steps is None:
            is_last_steps = [True] * batch_size
        if len(is_last_steps) != batch_size or any(type(v) is not bool for v in is_last_steps):
            raise ValueError('final-step flags must bind every batch item')
        if timesteps.ndim != 1 or sigmas.ndim != 1 or not timesteps.is_floating_point() or not sigmas.is_floating_point():
            raise ValueError('floating one-dimensional timestep/sigma tensors are required')
        validate_schedule(timesteps.detach().cpu().tolist(), sigmas.detach().cpu().tolist())
        if len(timesteps) > num_inference_steps:
            raise ValueError('round exceeds its declared remaining steps')
        expected_latent = (batch_size, self.pipe.transformer.config.in_channels,
                           height // self.raw.vae_scale_factor, width // self.raw.vae_scale_factor)
        if tuple(latents.shape) != expected_latent:
            raise ValueError('latent shape differs from the current request')
        if prompt_embeds.shape[0] != batch_size * 2 or pooled_prompt_embeds.shape[0] != batch_size * 2:
            raise ValueError('each request must retain its adjacent unconditional/conditional pair')
        for tensor in (latents, prompt_embeds, pooled_prompt_embeds):
            if tensor.device != latents.device or not self.torch.isfinite(tensor).all().item():
                raise ValueError('nonfinite or misplaced continuation tensor')
        with self._execution():
            scheduler = self.scheduler_class.from_config(self.scheduler_config)
            # Restore transferred intervals directly. Calling set_timesteps here would
            # apply the resolution shift again and change the denoising trajectory.
            scheduler.timesteps = timesteps.to(latents.device)
            scheduler.sigmas = sigmas.to(latents.device)
            scheduler.num_inference_steps = len(timesteps)
            scheduler._step_index = 0
            scheduler.set_begin_index(0)
            order = list(cfg_order(batch_size))
            pe = prompt_embeds[order]
            pp = pooled_prompt_embeds[order]
            self.runtime.bind_global()
            state = self.runtime.state
            if state.parallel_config.pp_degree != 1 or state.parallel_config.cfg_degree != 1:
                raise ValueError('TetriServe round adapter supports SP only (PP=CFG=1)')
            state.set_input_parameters(height=height, width=width, batch_size=batch_size,
                num_inference_steps=num_inference_steps, split_text_embed_in_sp=True,
                split_latents_by_rows=True)
            previous_scheduler = self.pipe.scheduler
            self.pipe.scheduler = scheduler
            self.pipe._guidance_scale = self.guidance_scale
            self.pipe._joint_attention_kwargs = None
            self.pipe._interrupt = False
            self.pipe._num_timesteps = len(timesteps)
            self.pipe._num_inference_steps = num_inference_steps
            try:
                with self.pipe.progress_bar(total=len(timesteps)) as progress:
                    result_latents = self.pipe._sync_pipeline(
                        latents=latents, prompt_embeds=pe, pooled_prompt_embeds=pp,
                        timesteps=scheduler.timesteps, num_warmup_steps=0,
                        progress_bar=progress, sync_only=True)
                if not self.torch.isfinite(result_latents).all().item():
                    raise ValueError('nonfinite denoising output')
                images = []
                for i, final in enumerate(is_last_steps):
                    item = result_latents[i:i + 1]
                    if final and output_type != 'latent':
                        image = self.raw.vae.decode(
                            item / self.raw.vae.config.scaling_factor + self.raw.vae.config.shift_factor,
                            return_dict=False)[0]
                        if not self.torch.isfinite(image).all().item():
                            raise ValueError('nonfinite decoded output')
                        images.append(self.raw.image_processor.postprocess(image, output_type=output_type)[0])
                    else:
                        images.append(item[0])
                from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
                return StableDiffusion3PipelineOutput(images=images)
            finally:
                self.pipe.scheduler = previous_scheduler

    def prepare(self, request: ImageRequest):
        self._profile(request.guidance_scale, request.max_sequence_length)
        # Formal shared-backend baseline uses the same CPU seeded generator.
        generator = self.torch.Generator(device='cpu').manual_seed(request.seed)
        tensors = self.inference_prologue(request.prompt, request.height, request.width, request.steps,
            generator, negative_prompt=request.negative_prompt)
        return RoundState(request, *tensors)

    def advance(self, state: RoundState, end_step):
        self._profile(state.request.guidance_scale, state.request.max_sequence_length)
        timesteps, sigmas = state.window(end_step)
        output = self.inference_denoising(height=state.request.height, width=state.request.width,
            batch_size=1, num_inference_steps=state.request.steps - state.cursor,
            timesteps=timesteps, sigmas=sigmas, latents=state.latents,
            prompt_embeds=state.prompt_embeds, pooled_prompt_embeds=state.pooled_prompt_embeds,
            is_last_steps=[False], output_type='latent')
        return replace(state, latents=output.images[0].unsqueeze(0), cursor=end_step)
