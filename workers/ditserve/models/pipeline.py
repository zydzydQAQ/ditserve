"""Pinned model loading and decoding; no experiment CLI or orchestration."""
from contextlib import nullcontext
import importlib
import math
from pathlib import Path
import time
from types import SimpleNamespace

from .base import Backend, positive_int, progress
from .wan.precision import TEXT_ENCODER_DTYPE, text_encoder_dtype

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


def _sdpa_context(torch, mode="math"):
    """Select SDPA kernel backend ('math' for strict bit-exact or 'flash' for Ampere Flash/Efficient SDPA)."""
    if mode == "math":
        return _sdpa_math_context(torch)
    attention = getattr(getattr(torch, "nn", None), "attention", None)
    if attention is not None and hasattr(attention, "sdpa_kernel"):
        return attention.sdpa_kernel([
            attention.SDPBackend.FLASH_ATTENTION,
            attention.SDPBackend.EFFICIENT_ATTENTION,
            attention.SDPBackend.MATH,
        ])
    options = dict(enable_flash=True, enable_math=True, enable_mem_efficient=True)
    try:
        return torch.backends.cuda.sdp_kernel(enable_cudnn=True, **options)
    except TypeError:
        return torch.backends.cuda.sdp_kernel(**options)

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


def _sd35_pipeline_class(base_class):
    """Keep SD3.5 denoising on the DiT/CLIP device when T5-XXL (`text_encoder_3`) lives on CPU."""
    if not isinstance(base_class, type):
        return base_class

    class SD35ExplicitExecutionPipeline(base_class):
        @property
        def device(self):
            return self.transformer.device

        @property
        def _execution_device(self):
            return self.transformer.device

    return SD35ExplicitExecutionPipeline


def _pipeline(torch, diffusers, args):
    load = {"torch_dtype": getattr(torch, args.dtype), "local_files_only": args.local_files_only}
    if args.revision:
        load["revision"] = args.revision
    if args.model == "sd35":
        te_device = getattr(args, "text_encoder_device", "cuda")
        te_dtype_name = getattr(args, "text_encoder_dtype", args.dtype)
        if te_device == "cpu":
            components = {}
            if te_dtype_name != args.dtype:
                transformers = importlib.import_module("transformers")
                t5_cls = getattr(transformers, "T5EncoderModel", None)
                if t5_cls is not None:
                    encoder_args = dict(load, torch_dtype=getattr(torch, te_dtype_name))
                    components["text_encoder_3"] = t5_cls.from_pretrained(
                        args.model_id, subfolder="text_encoder_3", **encoder_args
                    )
            pipeline_class = _sd35_pipeline_class(diffusers.StableDiffusion3Pipeline)
            pipe = pipeline_class.from_pretrained(args.model_id, **components, **load)
        else:
            pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(args.model_id, **load)
    else:
        vae_args = dict(load, torch_dtype=getattr(torch, args.vae_dtype))
        vae = diffusers.AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", **vae_args)
        # Load directly in the intended encoder precision, independently of the
        # FP16 DiT. An intermediate FP16 load would irreversibly round weights.
        transformers = importlib.import_module("transformers")
        encoder_args = dict(load, torch_dtype=getattr(torch, args.text_encoder_dtype))
        encoder = transformers.UMT5EncoderModel.from_pretrained(args.model_id, subfolder="text_encoder", **encoder_args)
        pipeline_class = (_wan_pipeline_class(diffusers.WanPipeline)
                          if args.text_encoder_device == "cpu" else diffusers.WanPipeline)
        components = {}
        if getattr(args, "engine", "legacy") == "xdit":
            from .xdit.wan import load_wan_transformer
            components["transformer"] = load_wan_transformer(
                args.model_id, getattr(torch, args.dtype), local_files_only=args.local_files_only)
        pipe = pipeline_class.from_pretrained(args.model_id, vae=vae, text_encoder=encoder, **components, **load)
    if args.model == "wan":
        # Never pipe.to(cuda) here: it transiently puts the large CPU encoder on
        # the GPU and can OOM before it is moved back.
        pipe.transformer.to(device=args.device, dtype=getattr(torch, args.dtype))
        if getattr(pipe, "transformer_2", None) is not None:
            pipe.transformer_2.to(device=args.device, dtype=getattr(torch, args.dtype))
        pipe.vae.to(device=args.device, dtype=getattr(torch, args.vae_dtype))
        encoder_device = "cpu" if args.text_encoder_device == "cpu" else args.device
        pipe.text_encoder.to(device=encoder_device, dtype=getattr(torch, args.text_encoder_dtype))
    elif args.model == "sd35" and getattr(args, "text_encoder_device", "cuda") == "cpu":
        # Never pipe.to(cuda) when T5-XXL is configured for CPU: keep text_encoder_3
        # on CPU in text_encoder_dtype and move only transformer, vae, CLIP-L, CLIP-G to device.
        gpu_dtype = getattr(torch, args.dtype)
        te_dtype = getattr(torch, getattr(args, "text_encoder_dtype", args.dtype))
        pipe.transformer.to(device=args.device, dtype=gpu_dtype)
        pipe.vae.to(device=args.device, dtype=gpu_dtype)
        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder.to(device=args.device, dtype=gpu_dtype)
        if getattr(pipe, "text_encoder_2", None) is not None:
            pipe.text_encoder_2.to(device=args.device, dtype=gpu_dtype)
        if getattr(pipe, "text_encoder_3", None) is not None:
            pipe.text_encoder_3.to(device="cpu", dtype=te_dtype)
        if not isinstance(getattr(type(pipe), "_execution_device", None), property):
            pipe._execution_device = getattr(pipe.transformer, "device", args.device)
    else:
        pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe



def decode_final_latent(torch, pipe, latent, *, vae_dtype=None):
    """Exact arithmetic/order from diffusers 0.36 WanPipeline's decode tail."""
    if vae_dtype is not None:
        target_dtype = getattr(torch, vae_dtype) if isinstance(vae_dtype, str) else vae_dtype
        if pipe.vae.dtype != target_dtype:
            pipe.vae.to(dtype=target_dtype)
    latent_f32 = latent.to(torch.float32)
    mean = torch.tensor(pipe.vae.config.latents_mean, dtype=torch.float32, device=latent.device).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    inverse_std = 1.0 / torch.tensor(pipe.vae.config.latents_std, dtype=torch.float32, device=latent.device).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    normalized = (latent_f32 / inverse_std + mean).to(pipe.vae.dtype)
    video = pipe.vae.decode(normalized, return_dict=False)[0].to(torch.float32)
    array = pipe.video_processor.postprocess_video(video, output_type="np")
    del video, normalized, mean, inverse_std, latent_f32
    if any(value is not None for name in ("_feat_map", "_enc_feat_map")
           for value in getattr(pipe.vae, name, [])):
        raise RuntimeError("VAE retained temporal feature caches after decode")
    return array


class ChunkedDecode:
    """Wan VAE decode suspendable at latent-frame boundaries.

    AutoencoderKLWan decodes causally one latent frame at a time and carries
    its temporal feature cache as an explicit argument. This replays diffusers
    0.36 ``_decode`` in the same order with the cache held per request, so the
    GPU can run other work between chunks; the result matches
    ``decode_final_latent``. The Wan2.2 VAE's pixel patches are folded back after
    the last chunk, as ``_decode`` does. Tiled decoding is not supported.
    """

    def __init__(self, torch, pipe, latent, *, vae_dtype=None):
        vae = pipe.vae
        if getattr(vae, "use_tiling", False) or latent.shape[0] != 1:
            raise ValueError("chunked decode supports the untiled single-video Wan VAE")
        if vae_dtype is not None:
            target_dtype = getattr(torch, vae_dtype) if isinstance(vae_dtype, str) else vae_dtype
            if vae.dtype != target_dtype:
                vae.to(dtype=target_dtype)
        self.torch, self.pipe = torch, pipe
        vae.clear_cache()
        latent_f32 = latent.to(torch.float32)
        mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=latent.device).view(1, vae.config.z_dim, 1, 1, 1)
        inverse_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32, device=latent.device).view(1, vae.config.z_dim, 1, 1, 1)
        normalized = (latent_f32 / inverse_std + mean).to(vae.dtype)
        self.x = vae.post_quant_conv(normalized)
        self.frames, self.index, self.chunks = self.x.shape[2], 0, []
        self.cache = [None] * vae._conv_num

    @property
    def done(self):
        return self.index == self.frames

    def advance(self, count=1):
        """Decode up to ``count`` latent frames; returns True after the last one."""
        if type(count) is not int or count < 1:
            raise ValueError("decode chunk count must be a positive integer")
        decoder = self.pipe.vae.decoder
        for _ in range(min(count, self.frames - self.index)):
            i = self.index
            chunk = self.x[:, :, i:i + 1, :, :]
            if i == 0:
                out = decoder(chunk, feat_cache=self.cache, feat_idx=[0], first_chunk=True)
            else:
                out = decoder(chunk, feat_cache=self.cache, feat_idx=[0])
            self.chunks.append(out)
            self.index += 1
        return self.done

    def finish(self):
        if not self.done:
            raise ValueError("decode has remaining latent frames")
        video = self.torch.cat(self.chunks, 2).to(self.torch.float32)
        patch = self.pipe.vae.config.patch_size
        if patch is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
            video = unpatchify(video, patch_size=patch)
        video = self.torch.clamp(video, min=-1.0, max=1.0)
        array = self.pipe.video_processor.postprocess_video(video, output_type="np")
        self.x = self.cache = self.chunks = None
        return array


class AtomicPipelineBackend(Backend):
    """Official full-pipeline actions and fixed offline batches; no scheduling."""

    model = "sd35"
    output_field = "images"
    text_encoder_dtype = TEXT_ENCODER_DTYPE
    defaults = dict(height=512, width=512, frames=1, steps=20, guidance_scale=3.5,
                    batch_size=1, seed=0, negative_prompt="")

    def __init__(self, config):
        super().__init__(config)
        import torch
        import diffusers
        path = Path(config["model_path"]).resolve()
        if not (path / "model_index.json").is_file():
            raise ValueError(self.model + " requires a local pipeline snapshot")
        self.torch, self.device = torch, config.get("device", "cuda:0")
        args = SimpleNamespace(model=self.model, model_id=str(path), device=self.device,
            revision=config.get("revision"), local_files_only=True,
            dtype=config.get("dtype", "float16" if self.model == "wan" else "bfloat16"),
            vae_dtype=config.get("vae_dtype", "float32"),
            text_encoder_dtype=config.get("text_encoder_dtype", TEXT_ENCODER_DTYPE),
            text_encoder_device=config.get("text_encoder_device", "cpu" if self.model == "wan" else "cuda"),
            engine="xdit")
        for value in (args.dtype, args.vae_dtype, args.text_encoder_dtype):
            if value not in ("float16", "float32", "bfloat16"):
                raise ValueError("unsupported model/component dtype: " + str(value))
        if args.text_encoder_device not in ("cpu", "cuda"):
            raise ValueError("text_encoder_device must be cpu or cuda")
        if self.model in ("wan", "sd35"):
            text_encoder_dtype(args.text_encoder_dtype)
        if args.text_encoder_device == "cpu" and self.model not in ("wan", "sd35"):
            raise ValueError("CPU text encoding requires Wan or SD3.5")
        self.text_encoder_dtype = args.text_encoder_dtype
        self.text_encoder_device = args.text_encoder_device
        gpu_dtypes = [args.dtype] + ([args.vae_dtype] if self.model == "wan" else [])
        if args.text_encoder_device == "cuda":
            gpu_dtypes.append(args.text_encoder_dtype)
        if (self.device.startswith("cuda") and "bfloat16" in gpu_dtypes
                and torch.cuda.get_device_capability(self.device)[0] < 8):
            raise ValueError("native bfloat16 requires Ampere or later; emulation is unsupported")
        self.pipe = _pipeline(torch, diffusers, args)
        if self.model == "wan":
            from .xdit.wan import attach_pipeline
            self.xdit = attach_pipeline(self.pipe, self.device)
        else:
            from .xdit.sd35 import attach
            attn_cfg = config.get("attention", config.get("options", {}).get("attention", "math"))
            self.xdit = attach(self.pipe, self.device, attention=attn_cfg)
        self.pipe = self.xdit.pipe
        exec_dev = getattr(getattr(self.pipe, "module", self.pipe), "_execution_device", None)
        if exec_dev is not None and str(self.torch.device(exec_dev)) != str(self.torch.device(self.device)):
            raise RuntimeError(
                f"{self.model} pipeline _execution_device ({exec_dev}) does not match worker device ({self.device})"
            )
        self._cache_recipe = None

    def _cache_target(self):
        return self.pipe.module

    def _enable_cache(self, spec):
        options = self.config.get("options", {})
        if options.get("cache"):
            recipe = (spec["steps"], self.model == "wan" and spec["guidance_scale"] > 1)
            if getattr(self, "_cache_recipe", None) is not None:
                if recipe != self._cache_recipe:
                    raise ValueError("cached full-pipeline workers require fixed steps and CFG mode")
                return
            import importlib.metadata
            import cache_dit
            if importlib.metadata.version("cache-dit") != "1.5.0":
                raise ValueError("full-pipeline cache adapter requires cache-dit 1.5.0")
            cache_dit.enable_cache(self._cache_target(), cache_config=cache_dit.DBCacheConfig(
                Fn_compute_blocks=options.get("cache_fn", 8), Bn_compute_blocks=options.get("cache_bn", 0),
                max_warmup_steps=options.get("cache_warmup", 8), warmup_interval=1, max_cached_steps=-1,
                residual_diff_threshold=options.get("threshold", .08),
                max_continuous_cached_steps=options.get("cache_max_continuous", 3),
                enable_separate_cfg=recipe[1], cfg_compute_first=False, cfg_diff_compute_separate=True,
                num_inference_steps=recipe[0]), attention_backend="native")
            self._cache_recipe = recipe

    def _spec(self, request):
        spec = dict(self.defaults, **self.config.get("spec", {}))
        spec.update(request.get("spec", {}))
        if not isinstance(spec.get("prompt"), str) or not isinstance(spec["negative_prompt"], str):
            raise ValueError("fixed offline batch requires string prompt and negative_prompt")
        for name in ("height", "width", "steps", "frames", "batch_size"):
            positive_int(spec[name], name)
        if spec["height"] % 16 or spec["width"] % 16:
            raise ValueError("height and width must be divisible by 16")
        if type(spec["seed"]) is not int or not -(1 << 63) <= spec["seed"] < (1 << 64):
            raise ValueError("seed must be an integer accepted by torch.Generator.manual_seed")
        guidance = spec["guidance_scale"]
        if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not math.isfinite(guidance) or guidance < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if self.model == "sd35" and spec["frames"] != 1:
            raise ValueError("sd35 requires frames=1")
        if self.model == "wan" and (spec["frames"] - 1) % 4:
            raise ValueError("Wan frames must be 4*k+1")
        return spec

    def prepare(self, request):
        key = self.new_id(request)
        self.states[key] = dict(spec=self._spec(request), output=None)
        return dict(state_id=key, progress=progress(0, 1), atomic_pipeline=True,
                    batch_size=self.states[key]["spec"]["batch_size"])

    def _arguments(self, spec):
        # One generator for the complete batch is the original offline protocol.
        return dict(prompt=[spec["prompt"]] * spec["batch_size"], height=spec["height"],
            width=spec["width"], num_inference_steps=spec["steps"], guidance_scale=spec["guidance_scale"],
            output_type="np", generator=self.torch.Generator(device="cpu").manual_seed(spec["seed"]))

    def _encode(self, arguments):
        return {}

    def advance(self, request):
        state = self.state(request)
        if state["output"] is not None:
            raise ValueError("pipeline has already completed")
        spec = state["spec"]
        self._enable_cache(spec)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        observer = None
        if self.config.get("options", {}).get("profile", False):
            from .profiling import PipelineProfiler
            observer = PipelineProfiler(self.torch, self.pipe, self.device)
        with self.torch.inference_mode(), _sdpa_math_context(self.torch):
            self.xdit.bind_global()
            arguments = self._arguments(spec)
            encoding = self._encode(arguments)
            with observer or nullcontext():
                if observer:
                    arguments.update(callback_on_step_end=observer.callback, callback_on_step_end_tensor_inputs=["latents"])
                state["output"] = getattr(self.pipe(**arguments), self.output_field)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter() - started) * 1000
        return dict(state_id=request["state_id"], progress=progress(1, 1), model_ms=elapsed,
                    atomic_pipeline=True, batch_size=spec["batch_size"], seed_protocol="single_cpu_generator_for_batch",
                    engine=self.xdit.info(),
                    **encoding, **(observer.report(spec["steps"]) if observer else {}))

    def finish(self, request):
        state = self.state(request)
        if state["output"] is None:
            raise ValueError("pipeline has not completed")
        result = dict(output=self.persist(state["output"], request), progress=progress(1, 1),
                      generation=dict(state["spec"], model=self.config.get("model_path"),
                          revision=self.config.get("revision"), batch_protocol="fixed_repeated_prompt",
                          seed_protocol="single_cpu_generator_for_batch", engine=self.xdit.info(),
                          approximate=bool(self.config.get("options", {}).get("cache"))))
        if self.model == "wan":
            result["generation"].update(
                dtype=self.config.get("dtype", "float16"), vae_dtype=self.config.get("vae_dtype", "float32"),
                text_encoder_dtype=self.text_encoder_dtype,
                text_encoder_device=self.config.get("text_encoder_device", "cpu"))
        elif self.model == "sd35":
            te_dev = getattr(self, "text_encoder_device", self.config.get("text_encoder_device", "cuda"))
            clip_dev = str(getattr(self, "device", self.config.get("device", "cuda")))
            t5_dev = "cpu" if te_dev == "cpu" else clip_dev
            result["generation"].update(
                dtype=self.config.get("dtype", "bfloat16"),
                text_encoder_dtype=getattr(self, "text_encoder_dtype", self.config.get("text_encoder_dtype", TEXT_ENCODER_DTYPE)),
                text_encoder_device=te_dev,
                text_encoder_devices={"clip_l": clip_dev, "clip_g": clip_dev, "t5_xxl": t5_dev},
            )
        self.release_cache()
        return result

    def close(self):
        super().close()
        self.xdit.close()

