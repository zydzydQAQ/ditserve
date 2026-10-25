"""Complete xDiT SD3.5 pipeline and fixed-batch serving contracts on CPU."""

from contextlib import nullcontext
import importlib.util
from types import SimpleNamespace
import unittest

PINNED = all(importlib.util.find_spec(name) is not None
             for name in ("torch", "diffusers", "cache_dit", "xfuser", "yunchang"))


def tiny_pipeline(torch):
    from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline

    class StableDiffusion3PipelineStub(StableDiffusion3Pipeline):
        default_sample_size = 1
        vae_scale_factor = 2
        patch_size = 2

        @property
        def _execution_device(self):
            return torch.device("cpu")

        def __init__(self):
            self._internal_dict = {}
            self._callback_tensor_inputs = ["latents"]
            torch.manual_seed(81)
            self.transformer = SD3Transformer2DModel(
                sample_size=16, patch_size=2, in_channels=4, out_channels=4,
                num_layers=2, attention_head_dim=8, num_attention_heads=2,
                joint_attention_dim=24, caption_projection_dim=16,
                pooled_projection_dim=8).eval()
            self.scheduler = FlowMatchEulerDiscreteScheduler()
            self.vae = SimpleNamespace(
                dtype=torch.float32, config=SimpleNamespace(scaling_factor=.5, shift_factor=.1),
                decode=lambda value, **kwargs: (value[:, :1].repeat(1, 3, 1, 1),))
            self.image_processor = SimpleNamespace(
                postprocess=lambda image, output_type: image.detach().cpu().permute(0, 2, 3, 1).numpy().copy())

        def encode_prompt(self, prompt, **kwargs):
            count = len(prompt) if isinstance(prompt, list) else 1
            embeds = torch.arange(count * 72, dtype=torch.float32).reshape(count, 3, 24) / 256
            pooled = torch.full((count, 8), .25)
            if self.do_classifier_free_guidance:
                return embeds, -embeds, pooled, -pooled
            return embeds, None, pooled, None

        def progress_bar(self, **kwargs):
            return nullcontext(SimpleNamespace(update=lambda: None))

        def maybe_free_model_hooks(self):
            pass

    return StableDiffusion3PipelineStub()


def arguments(torch):
    return dict(prompt=["kettle", "kettle"], negative_prompt="blurry",
                height=16, width=16, num_inference_steps=2, guidance_scale=4.5,
                max_sequence_length=8, output_type="np",
                generator=torch.Generator(device="cpu").manual_seed(17))


def generate(torch, pipeline):
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        return pipeline(**arguments(torch)).images


@unittest.skipUnless(PINNED, "pinned xDiT model environment is required")
class XDitSD35Tests(unittest.TestCase):
    def test_complete_pipeline_matches_native_fixed_batch_protocol(self):
        import numpy as np
        import torch
        from ditserve.models.xdit.sd35 import attach

        expected = generate(torch, tiny_pipeline(torch))
        engine = attach(tiny_pipeline(torch), "cpu")
        self.addCleanup(engine.close)
        actual = generate(torch, engine.pipe)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape[0], 2)
        self.assertEqual(engine.info()["pipeline"], "SD35XDitPipeline")
        self.assertEqual(engine.info()["transformer"], "xFuserSD3Transformer2DWrapper")

    def test_transformer_return_contract_matches_native(self):
        import torch
        from ditserve.models.xdit.sd35 import attach

        native = tiny_pipeline(torch)
        engine = attach(tiny_pipeline(torch), "cpu")
        self.addCleanup(engine.close)
        engine.activate_image(height=16, width=16, num_inference_steps=2)
        torch.manual_seed(121)
        arguments = dict(hidden_states=torch.randn(1, 4, 8, 8),
                         encoder_hidden_states=torch.randn(1, 3, 24),
                         pooled_projections=torch.randn(1, 8), timestep=torch.tensor([1]))
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            expected = native.transformer(**arguments, return_dict=False)[0]
            actual = engine.pipe.transformer(**arguments, return_dict=False)[0]
            output = engine.pipe.transformer(**arguments, return_dict=True)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.is_tensor(output.sample))
        self.assertTrue(torch.equal(output.sample, expected))

    def test_cache_dit_recognizes_and_runs_the_wrapped_pipeline(self):
        import numpy as np
        import torch
        from ditserve.models.base import Backend
        from ditserve.models.sd35 import SD35Backend
        from ditserve.models.xdit.sd35 import attach

        expected = generate(torch, tiny_pipeline(torch))
        backend = SD35Backend.__new__(SD35Backend)
        Backend.__init__(backend, dict(options=dict(cache=True, cache_fn=1, cache_warmup=1,
                                                   threshold=0., cache_max_continuous=1)))
        backend.xdit = attach(tiny_pipeline(torch), "cpu")
        backend.pipe = backend.xdit.pipe
        self.addCleanup(backend.close)
        self.assertEqual(backend._cache_target().blocks_name, "transformer_blocks")
        backend._enable_cache(dict(steps=2, guidance_scale=4.5))
        actual = generate(torch, backend.pipe)
        np.testing.assert_array_equal(actual, expected)
        contexts = backend.pipe.module._context_manager._cached_context_manager
        self.assertTrue(any(context.buffers for context in contexts.values()))

    def test_backend_preserves_negative_prompt_and_profiles_all_stages(self):
        from unittest.mock import Mock
        from ditserve.models.base import Backend
        from ditserve.models.sd35 import SD35Backend
        from ditserve.models.xdit.sd35 import attach
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
        backend = SD35Backend.__new__(SD35Backend)
        Backend.__init__(backend, dict(options={"profile": True}))
        backend.torch, backend.device = proxy, "cuda:0"
        backend.xdit = attach(tiny_pipeline(torch), "cpu")
        backend.pipe, backend._cache_recipe = backend.xdit.pipe, None
        self.addCleanup(backend.close)
        backend.prepare(dict(state_id="profile", spec=dict(
            prompt="kettle", negative_prompt="blurry", steps=2, height=16, width=16,
            batch_size=2, max_sequence_length=8)))
        result = backend.advance(dict(state_id="profile"))
        self.assertEqual(backend.states["profile"]["output"].shape[0], 2)
        self.assertEqual(set(result["phase_ms"]),
                         {"text_encode", "latent_init", "transformer", "scheduler", "vae_decode", "postprocess"})
        self.assertEqual(result["profile"]["forward_count"], 2)
        self.assertEqual(result["engine"]["transformer"], "xFuserSD3Transformer2DWrapper")
        self.assertFalse(torch.cuda.is_initialized())

    def test_worker_loads_and_executes_sd35_backend(self):
        import diffusers
        import numpy as np
        import tempfile
        import torch
        from pathlib import Path
        from unittest.mock import patch
        from ditserve.runtime.worker import Worker

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "model_index.json").write_text("{}")
            with patch.object(diffusers.StableDiffusion3Pipeline, "from_pretrained",
                              return_value=tiny_pipeline(torch)) as loader:
                worker = Worker("sd35", dict(model_path=directory, revision="pinned",
                                             device="cpu", dtype="float32", work_dir=directory))
                self.addCleanup(worker.close)

                def call(op, ident, **fields):
                    return worker.dispatch(dict(version=1, id=ident, op=op, **fields))

                self.assertFalse(call("hello", 1)["loaded"])
                call("prepare", 2, state_id="image", spec=dict(
                    prompt="kettle", negative_prompt="blurry", seed=17, steps=2,
                    height=16, width=16, batch_size=2, max_sequence_length=8))
                answer = call("advance", 3, state_id="image")
                self.assertEqual(answer["engine"]["transformer"], "xFuserSD3Transformer2DWrapper")
                answer = call("finish", 4, state_id="image", output_path=str(Path(directory, "actual.npy")))
                expected = generate(torch, tiny_pipeline(torch))
                np.testing.assert_array_equal(np.load(answer["output"]["path"], allow_pickle=False), expected)
                self.assertEqual(answer["generation"]["batch_size"], 2)
                self.assertEqual(answer["generation"]["revision"], "pinned")
                self.assertTrue(loader.call_args.kwargs["local_files_only"])
                self.assertEqual(loader.call_args.kwargs["torch_dtype"], torch.float32)
                call("release", 5, state_id="image")
                self.assertTrue(call("shutdown", 6)["shutdown"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_model_specific_spec_validation(self):
        from ditserve.models.base import Backend
        from ditserve.models.sd35 import SD35Backend

        backend = SD35Backend.__new__(SD35Backend)
        Backend.__init__(backend, {})
        with self.assertRaisesRegex(ValueError, "frames=1"):
            backend._spec(dict(spec=dict(prompt="kettle", frames=2)))
        with self.assertRaisesRegex(ValueError, "T5 tokenizer"):
            backend._spec(dict(spec=dict(prompt="kettle", max_sequence_length=513)))


if __name__ == "__main__":
    unittest.main()
