"""Official full-pipeline actions and fixed offline batches; no scheduling."""

import math
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from .base import Backend, artifact, positive_int, progress


class FluxBackend(Backend):
    model = "flux"
    output_field = "images"
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
            text_encoder_dtype=config.get("text_encoder_dtype", "float32"),
            text_encoder_device=config.get("text_encoder_device", "cpu" if self.model == "wan" else "cuda"),
            engine="xdit")
        for value in (args.dtype, args.vae_dtype, args.text_encoder_dtype):
            if value not in ("float16", "float32", "bfloat16"):
                raise ValueError("unsupported model/component dtype: " + str(value))
        if args.text_encoder_device not in ("cpu", "cuda"):
            raise ValueError("text_encoder_device must be cpu or cuda")
        if args.text_encoder_device == "cpu" and (self.model != "wan" or args.text_encoder_dtype != "float32"):
            raise ValueError("CPU text encoding requires Wan and float32 encoder weights")
        gpu_dtypes = [args.dtype] + ([args.vae_dtype] if self.model == "wan" else [])
        if args.text_encoder_device == "cuda":
            gpu_dtypes.append(args.text_encoder_dtype)
        if (self.device.startswith("cuda") and "bfloat16" in gpu_dtypes
                and torch.cuda.get_device_capability(self.device)[0] < 8):
            raise ValueError("native bfloat16 requires Ampere or later; emulation is unsupported")
        from .pipeline import _pipeline
        self.pipe = _pipeline(torch, diffusers, args)
        if self.model == "wan":
            from .xdit.wan import attach_pipeline
            self.xdit = attach_pipeline(self.pipe, self.device)
        else:
            module = ".xdit.sd35" if self.model == "sd35" else ".xdit.flux"
            from importlib import import_module
            self.xdit = import_module(module, __package__).attach(self.pipe, self.device)
        self.pipe = self.xdit.pipe
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
        if self.model in ("flux", "sd35") and spec["frames"] != 1:
            raise ValueError(self.model + " requires frames=1")
        if self.model == "flux" and spec["negative_prompt"]:
            raise ValueError("FLUX does not accept a negative prompt")
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
        from .pipeline import _sdpa_math_context
        observer = None
        if self.config.get("options", {}).get("profile", False):
            from .profiling import PipelineProfiler
            observer = PipelineProfiler(self.torch, self.pipe, self.device)
        with self.torch.inference_mode(), _sdpa_math_context(self.torch):
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
        self.release_cache()
        return result

    def close(self):
        super().close()
        self.xdit.close()
