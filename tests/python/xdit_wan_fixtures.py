"""Tiny real Wan components shared by xDiT runtime contract tests."""

from contextlib import nullcontext
from types import SimpleNamespace


def transformer(torch, layers=3):
    from diffusers import WanTransformer3DModel
    torch.manual_seed(907)
    return WanTransformer3DModel(
        num_attention_heads=4, attention_head_dim=12, in_channels=16, out_channels=16,
        num_layers=layers, ffn_dim=24, freq_dim=4, text_dim=4096,
        patch_size=(1, 2, 2), rope_max_seq_len=16).eval().to(torch.float16)


class WanTestVAE:
    dtype = None
    use_tiling = False

    def __init__(self, torch):
        self.torch = torch
        self.dtype = torch.float32
        self.config = SimpleNamespace(
            z_dim=16, latents_mean=[0.] * 16, latents_std=[1.] * 16, patch_size=None)
        self._cached_conv_counts = {"decoder": 1, "encoder": 1}

    def clear_cache(self):
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def post_quant_conv(self, value):
        return value

    def decoder(self, chunk, feat_cache=None, feat_idx=None, first_chunk=False):
        if feat_cache is not None:
            feat_cache[0] = chunk.detach().clone()
        frames = 1 if first_chunk else 4
        return self.torch.nn.functional.interpolate(
            chunk[:, :3], size=(frames, chunk.shape[-2] * 8, chunk.shape[-1] * 8), mode="nearest")

    def decode(self, latents, return_dict=False):
        self.clear_cache()
        cache = [None] * self._conv_num
        chunks = [self.decoder(latents[:, :, index:index + 1], cache, [0], index == 0)
                  for index in range(latents.shape[2])]
        output = self.torch.cat(chunks, dim=2).clamp(-1, 1)
        self.clear_cache()
        return (output,)


def native_pipeline(torch, layers=3):
    from diffusers import UniPCMultistepScheduler, WanPipeline
    from diffusers.configuration_utils import FrozenDict

    class WanRuntimePipeline(WanPipeline):
        vae_scale_factor = 8
        vae_scale_factor_spatial = 8
        vae_scale_factor_temporal = 4

        def __init__(self):
            self._internal_dict = FrozenDict(expand_timesteps=False, boundary_ratio=None)
            self.transformer = transformer(torch, layers)
            self.scheduler = UniPCMultistepScheduler(num_train_timesteps=1000)
            self.vae = WanTestVAE(torch)
            self.video_processor = SimpleNamespace(
                postprocess_video=lambda video, output_type: video.detach().cpu().permute(0, 2, 3, 4, 1).numpy().copy())

        @property
        def _execution_device(self):
            return torch.device("cpu")

        def encode_prompt(self, prompt, negative_prompt, do_classifier_free_guidance,
                          num_videos_per_prompt, prompt_embeds=None, negative_prompt_embeds=None, **kwargs):
            if prompt_embeds is None:
                count = len(prompt) if isinstance(prompt, list) else 1
                prompt_embeds = torch.full((count, 512, 4096), .25,
                    dtype=kwargs.get("dtype", torch.float16), device=kwargs.get("device", "cpu"))
            if not do_classifier_free_guidance:
                return prompt_embeds, None
            if negative_prompt_embeds is None:
                negative_prompt_embeds = torch.full_like(prompt_embeds, -.125)
            return prompt_embeds, negative_prompt_embeds

        def progress_bar(self, **kwargs):
            return nullcontext(SimpleNamespace(update=lambda: None))

        def maybe_free_model_hooks(self):
            pass

    return WanRuntimePipeline()


def runtime_worker(torch, *, cached=False, xdit=True, layers=3, steps=None):
    try:
        from test_sp_block import plain_executor
    except ImportError:
        from tests.python.test_sp_block import plain_executor
    from ditserve.models.xdit.wan import XDitWanEngine

    worker, state, unused = plain_executor(torch)
    worker.pipe = native_pipeline(torch, layers)
    worker.pipe.scheduler = state.scheduler
    import threading
    from ditserve.models import pipeline as pipeline_module
    worker.p = pipeline_module
    worker._encoder_lock = threading.Lock()
    worker._scheduler_config = dict(state.scheduler.config)
    worker.spatial_multiple = worker.pipe.vae_scale_factor_spatial * worker.pipe.transformer.config.patch_size[1]
    worker._memory = lambda: {}
    worker.engine_name = "xdit" if xdit else "legacy"
    worker.xdit = XDitWanEngine(worker) if xdit else None
    worker._original_forward = worker.pipe.transformer.forward
    worker.cache_dit.enable_cache(worker.pipe, cache_config=worker._cache_config(state.spec),
                                  attention_backend="native")
    worker._cached_forward = worker.pipe.transformer.forward
    worker.manager = worker.pipe._context_manager
    worker._context_names = tuple(worker.pipe.transformer._context_names)
    worker._detach_contexts()
    worker._official_call = getattr(type(worker.pipe), "_original_call", type(worker.pipe).__call__)
    state.spec.update(cache=cached, threshold=1. if cached else 0., max_continuous=1)
    state.cache_enabled = cached
    state.contexts = worker._contexts(state.spec) if cached else {}
    state.prompt_embeds = torch.full_like(state.prompt_embeds, .25)
    state.negative_prompt_embeds = torch.full_like(state.negative_prompt_embeds, -.125)
    if steps is not None and steps != state.spec["steps"]:
        state.scheduler = type(state.scheduler)(num_train_timesteps=1000)
        state.scheduler.set_timesteps(steps, device="cpu")
        state.step_index = 0
        state.spec["steps"] = state.immutable_spec["steps"] = steps
        state.latents = worker.pipe.prepare_latents(
            1, worker.pipe.transformer.config.in_channels, state.spec["height"], state.spec["width"],
            state.spec["frames"], torch.float32, "cpu", state.generator, None)
        worker.pipe.scheduler = state.scheduler
    for context in state.contexts.values():
        context.executed_steps = context.accumulated_executed_steps = state.step_index
        forwards = state.step_index * (2 if state.spec["guidance_scale"] > 1 else 1)
        context.transformer_executed_steps = forwards
        context.accumulated_transformer_executed_steps = forwards
    return worker, state


def engine_info():
    return dict(name="test_double", revision="unverified")


class EngineDouble:
    def info(self):
        return engine_info()

    def close(self):
        pass
