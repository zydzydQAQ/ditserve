"""Complete xFuser FLUX pipeline and fixed-batch cache contracts on CPU."""

from contextlib import nullcontext
import importlib.util
from types import SimpleNamespace
import unittest


PINNED = all(importlib.util.find_spec(name) is not None
             for name in ("torch", "diffusers", "cache_dit", "xfuser", "yunchang"))


def tiny_pipeline(torch):
    from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel

    class FluxPipelineStub(FluxPipeline):
        default_sample_size = 1
        vae_scale_factor = 2

        @property
        def _execution_device(self):
            return torch.device("cpu")

        def __init__(self):
            self._internal_dict = {}
            torch.manual_seed(81)
            self.transformer = FluxTransformer2DModel(
                in_channels=4, out_channels=4, num_layers=2, num_single_layers=2,
                attention_head_dim=8, num_attention_heads=2, joint_attention_dim=8,
                pooled_projection_dim=4, guidance_embeds=True, axes_dims_rope=(2, 2, 4)).eval()
            self.scheduler = FlowMatchEulerDiscreteScheduler()
            self.vae = SimpleNamespace(
                dtype=torch.float32, config=SimpleNamespace(scaling_factor=.5, shift_factor=.1),
                decode=lambda value, **kwargs: (value[:, :1].repeat(1, 3, 1, 1),))
            self.image_processor = SimpleNamespace(
                postprocess=lambda image, output_type: image.detach().cpu().permute(0, 2, 3, 1).numpy().copy())

        def encode_prompt(self, prompt, **kwargs):
            count = len(prompt) if isinstance(prompt, list) else 1
            embeds = torch.arange(count * 24, dtype=torch.float32).reshape(count, 3, 8) / 256
            return embeds, torch.full((count, 4), .25), torch.zeros(3, 3)

        def progress_bar(self, **kwargs):
            return nullcontext(SimpleNamespace(update=lambda: None))

        def maybe_free_model_hooks(self):
            pass

    return FluxPipelineStub()


def arguments(torch):
    return dict(prompt=["kettle", "kettle"], height=8, width=8, num_inference_steps=2,
                guidance_scale=3.5, output_type="np",
                generator=torch.Generator(device="cpu").manual_seed(17))


def generate(torch, pipeline):
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        return pipeline(**arguments(torch)).images


@unittest.skipUnless(PINNED, "pinned xDiT model environment is required")
class XDitFluxTests(unittest.TestCase):
    def test_complete_pipeline_matches_native_fixed_batch_protocol(self):
        import numpy as np
        import torch
        from ditserve.models.xdit.flux import attach

        expected = generate(torch, tiny_pipeline(torch))
        engine = attach(tiny_pipeline(torch), "cpu")
        self.addCleanup(engine.close)
        actual = generate(torch, engine.pipe)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape[0], 2)
        self.assertEqual(engine.info()["pipeline"], "FluxXDitPipeline")
        self.assertEqual(engine.info()["transformer"], "xFuserFluxTransformer2DWrapper")

    def test_cache_dit_recognizes_and_runs_the_wrapped_pipeline(self):
        import cache_dit
        import numpy as np
        import torch
        from ditserve.models.xdit.flux import attach

        expected = generate(torch, tiny_pipeline(torch))
        engine = attach(tiny_pipeline(torch), "cpu")
        self.addCleanup(engine.close)
        from cache_dit.caching.block_adapters.block_registers import BlockAdapterRegister
        adapter = BlockAdapterRegister.get_adapter(engine.pipe.module)
        self.assertIsNotNone(adapter)
        cache_dit.enable_cache(adapter, cache_config=cache_dit.DBCacheConfig(
            Fn_compute_blocks=1, Bn_compute_blocks=0, max_warmup_steps=1, warmup_interval=1,
            max_cached_steps=-1, residual_diff_threshold=0., max_continuous_cached_steps=1,
            num_inference_steps=2), attention_backend="native")
        actual = generate(torch, engine.pipe)
        np.testing.assert_array_equal(actual, expected)
        contexts = engine.pipe.module._context_manager._cached_context_manager
        self.assertTrue(any(context.buffers for context in contexts.values()))

    def test_pipeline_profiler_observes_native_scheduler_and_all_stages(self):
        from unittest.mock import Mock
        from ditserve.models.base import Backend
        from ditserve.models.flux import FluxBackend
        from ditserve.models.xdit.flux import attach
        import torch

        class Event:
            tick = 0

            def __init__(self, **unused):
                self.at = None

            def record(self):
                Event.tick += 1
                self.at = Event.tick

            def elapsed_time(self, end):
                return (end.at - self.at) * .25

        cuda = SimpleNamespace(Event=Event, synchronize=Mock(), reset_peak_memory_stats=Mock(),
            memory_allocated=lambda device: 16, max_memory_allocated=lambda device: 64,
            max_memory_reserved=lambda device: 128)
        proxy = SimpleNamespace(cuda=cuda, is_tensor=torch.is_tensor, isfinite=torch.isfinite,
            Generator=torch.Generator, inference_mode=torch.inference_mode,
            nn=SimpleNamespace(attention=SimpleNamespace(sdpa_kernel=lambda modes: nullcontext(),
                                                          SDPBackend=SimpleNamespace(MATH="math"))))
        backend = FluxBackend.__new__(FluxBackend)
        Backend.__init__(backend, dict(options={"profile": True}))
        backend.torch, backend.device = proxy, "cuda:0"
        backend.xdit = attach(tiny_pipeline(torch), "cpu")
        backend.pipe, backend._cache_recipe = backend.xdit.pipe, None
        self.addCleanup(backend.close)
        backend.prepare(dict(state_id="profile", spec=dict(prompt="kettle", steps=2, height=16, width=16)))
        result = backend.advance(dict(state_id="profile"))
        self.assertEqual(set(result["phase_ms"]),
                         {"text_encode", "latent_init", "transformer", "scheduler", "vae_decode", "postprocess"})
        self.assertEqual(result["profile"]["forward_count"], 2)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
