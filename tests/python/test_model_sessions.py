"""CPU numerical references for Sana session ownership and resumable arithmetic.

Sana uses the installed official pipeline loop and SCM with small deterministic
model components. No test loads production weights or initializes CUDA.
"""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
import tempfile
from pathlib import Path

import importlib.util
if any(importlib.util.find_spec(name) is None for name in ("torch", "numpy", "diffusers", "transformers")):
    raise unittest.SkipTest("optional pinned CPU model dependencies unavailable")

import numpy as np
import torch
from diffusers import SCMScheduler
from diffusers.pipelines.sana.pipeline_sana_sprint import SanaSprintPipeline

from ditserve.models.session.runtime import SanaRuntime, tensor_comparison


class Transformer(torch.nn.Module):
    dtype = torch.float32
    config = SimpleNamespace(sample_size=32, in_channels=2, guidance_embeds_scale=.1)

    def forward(self, value, *, encoder_hidden_states, guidance, timestep, **kwargs):
        prompt = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)
        return (value * .125 + prompt * .01
                + guidance.view(-1, 1, 1, 1) * .02 + timestep.view(-1, 1, 1, 1) * .03,)


class VAE:
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=.5)

    def decode(self, value, **kwargs):
        return (value[:, :1].repeat(1, 3, 1, 1),)


class Processor:
    def classify_height_width_bin(self, height, width, **kwargs):
        return height, width

    def resize_and_crop_tensor(self, image, *args):
        return image

    def postprocess(self, image, output_type):
        return image.detach().cpu().permute(0, 2, 3, 1).numpy().copy()


class Pipeline:
    __call__ = SanaSprintPipeline.__call__
    check_inputs = SanaSprintPipeline.check_inputs
    prepare_latents = SanaSprintPipeline.prepare_latents
    prepare_extra_step_kwargs = SanaSprintPipeline.prepare_extra_step_kwargs
    guidance_scale = SanaSprintPipeline.guidance_scale
    attention_kwargs = SanaSprintPipeline.attention_kwargs
    interrupt = SanaSprintPipeline.interrupt
    _callback_tensor_inputs = SanaSprintPipeline._callback_tensor_inputs
    _execution_device = torch.device("cpu")
    vae_scale_factor = 32

    def __init__(self):
        self.scheduler = SCMScheduler()
        self.transformer = Transformer()
        self.vae = VAE()
        self.image_processor = Processor()

    def encode_prompt(self, prompt, **kwargs):
        offset = sum(map(ord, prompt)) / 1000.
        return torch.arange(8, dtype=torch.float32).reshape(1, 4, 2) / 10 + offset, torch.ones(1, 4)

    def progress_bar(self, **kwargs):
        return nullcontext(SimpleNamespace(update=lambda: None))

    def maybe_free_model_hooks(self):
        pass


class TransformerEngineDouble:
    def __init__(self, pipe):
        self.pipe = pipe

    def forward(self, hidden_states, **kwargs):
        kwargs.pop("num_inference_steps")
        return self.pipe.transformer(hidden_states, **kwargs)

    def close(self):
        pass


def sana():
    adapter = SanaRuntime.__new__(SanaRuntime)
    adapter._initialize_runtime("cpu")
    adapter.pipe = Pipeline()
    adapter.xdit = TransformerEngineDouble(adapter.pipe)
    adapter._instruction = []
    adapter._configure()
    return adapter


class SanaRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = sana()

    def tearDown(self):
        self.adapter.close()

    def test_each_step_and_decoded_output_match_official_pipeline_loop(self):
        expected = self.adapter.reference("silver kettle", 301)
        state = self.adapter.prepare("silver kettle", 301)
        self.assertFalse(state.done)
        for index, latent in enumerate(expected["step_latents"]):
            action = self.adapter.advance(state)
            self.assertEqual(action["step"], index + 1)
            self.assertTrue(tensor_comparison(state.latents, latent)["exact"])
        self.assertTrue(state.done)
        np.testing.assert_array_equal(self.adapter.finish(state), expected["output"])
        self.assertGreaterEqual(state.prepare_ms, 0.)
        self.assertGreaterEqual(state.decode_ms, 0.)

    def test_interleaved_sana_requests_keep_sampler_and_rng_independent(self):
        references = [self.adapter.reference(prompt, seed) for prompt, seed in (("kettle", 301), ("clock", 302))]
        a, b = self.adapter.prepare("kettle", 301), self.adapter.prepare("clock", 302)
        self.assertIsNot(a.scheduler, b.scheduler)
        self.adapter.advance(a)
        snapshot = self.adapter.snapshot(a)
        self.adapter.advance(b)
        self.adapter.advance(b)
        self.assertTrue(tensor_comparison(a.generator.get_state(), snapshot["generator_state"])["exact"])
        self.assertEqual(a.scheduler.step_index, snapshot["sampler_step"])
        self.adapter.advance(a)
        for state, expected in zip((a, b), references):
            np.testing.assert_array_equal(self.adapter.finish(state), expected["output"])





    def test_continuous_batch_late_join_and_early_exit_match_independent_references(self):
        references = {prompt: self.adapter.reference(prompt, seed)
                      for prompt, seed in (("kettle", 301), ("clock", 302))}
        a = self.adapter.prepare("kettle", 301, request_id="a")
        self.adapter.advance(a)
        b = self.adapter.prepare("clock", 302, request_id="b")

        batch = self.adapter.advance_batch([a, b])
        self.assertEqual(batch["batch_size"], 2)
        self.assertEqual([(row["step"], row["done"]) for row in batch["events"]], [(2, True), (1, False)])
        self.assertEqual([row["ms"] for row in batch["events"]], [batch["ms"], batch["ms"]])
        self.assertIsNone(batch["memory_before"])
        self.assertEqual((a.scheduler.step_index, b.scheduler.step_index), (2, 1))
        self.assertTrue(tensor_comparison(a.latents, references["kettle"]["step_latents"][1])["exact"])
        self.assertTrue(tensor_comparison(b.latents, references["clock"]["step_latents"][0])["exact"])

        np.testing.assert_array_equal(self.adapter.finish(a), references["kettle"]["output"])
        self.adapter.release(a)
        with self.assertRaisesRegex(ValueError, "closed"):
            self.adapter.advance(a)
        remaining = self.adapter.advance_batch([b])
        self.assertEqual(remaining["events"][0]["step"], 2)
        self.assertTrue(remaining["events"][0]["done"])
        np.testing.assert_array_equal(self.adapter.finish(b), references["clock"]["output"])

    def test_continuous_batch_validates_all_states_before_advancing_any(self):
        with self.assertRaisesRegex(ValueError, "1 to 8"):
            self.adapter.advance_batch([])
        a, b = self.adapter.prepare("kettle", 301), self.adapter.prepare("clock", 302)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.adapter.advance_batch([a, a])
        other = sana()
        try:
            foreign = other.prepare("foreign", 303)
            with self.assertRaisesRegex(ValueError, "foreign"):
                self.adapter.advance_batch([a, foreign])
        finally:
            other.close()
        self.adapter._move(b, "cpu:0")
        with self.assertRaisesRegex(ValueError, "offloaded"):
            self.adapter.advance_batch([a, b])
        self.adapter.restore(b)
        self.assertEqual((a.step_index, b.step_index), (0, 0))

        embeddings = b.prompt_embeds
        b.prompt_embeds = torch.ones((1, embeddings.shape[1] + 1, embeddings.shape[2]),
                                     dtype=embeddings.dtype, device=embeddings.device)
        with self.assertRaisesRegex(ValueError, "compatible"):
            self.adapter.advance_batch([a, b])
        b.prompt_embeds = embeddings
        self.adapter.advance(a)
        self.adapter.advance(a)
        before = self.adapter.snapshot(b)
        with self.assertRaisesRegex(ValueError, "already complete"):
            self.adapter.advance_batch([a, b])
        after = self.adapter.snapshot(b)
        self.assertEqual(after["sampler_step"], before["sampler_step"])
        for name in ("latents", "denoised", "prompt_embeds", "timesteps", "generator_state"):
            self.assertTrue(tensor_comparison(after[name], before[name])["exact"])

    def test_state_tensor_reconstruction_preserves_rng_and_sampler_index(self):
        state = self.adapter.prepare("kettle", 301)
        self.adapter.advance(state)
        before = self.adapter.snapshot(state)
        self.adapter._move(state, "cpu:0")
        with self.assertRaisesRegex(ValueError, "offloaded"):
            self.adapter.advance(state)
        self.adapter.restore(state)
        after = self.adapter.snapshot(state)
        self.assertEqual(after["sampler_step"], before["sampler_step"])
        for name in ("latents", "denoised", "prompt_embeds", "timesteps", "generator_state"):
            self.assertTrue(tensor_comparison(after[name], before[name])["exact"])
        self.adapter.advance(state)
        expected = self.adapter.reference("kettle", 301)
        np.testing.assert_array_equal(self.adapter.finish(state), expected["output"])


class ComparisonTests(unittest.TestCase):
    def test_shared_cleanup_rejects_foreign_owner_and_closes_failed_states(self):
        runtime, other = sana(), sana()
        state, foreign = runtime.prepare("kettle"), other.prepare("kettle")
        try:
            with self.assertRaisesRegex(ValueError, "foreign"):
                runtime.release(foreign)
            self.assertFalse(foreign.closed)
            state.failed = True
            runtime.release(state)
            runtime.release(state)
            self.assertTrue(state.closed)
            self.assertFalse(runtime._states)
            for name in ("latents", "prompt_embeds", "scheduler", "generator", "output"):
                self.assertIsNone(getattr(state, name))
        finally:
            runtime.close()
            other.close()
        self.assertIsNone(runtime.pipe)
        self.assertTrue(foreign.closed)

    def test_tensor_comparison_rejects_shape_dtype_and_nonfinite(self):
        self.assertFalse(tensor_comparison(torch.ones(2), torch.ones(1))["passed"])
        self.assertFalse(tensor_comparison(torch.ones(2), torch.ones(2, dtype=torch.float64))["passed"])
        self.assertFalse(tensor_comparison(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["passed"])
        self.assertTrue(tensor_comparison(torch.ones(1), torch.ones(1) + 1e-5, atol=1e-4)["passed"])

    def test_cpu_test_suite_has_not_initialized_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_json_adapter_progress_and_output(self):
        from ditserve.models.base import Backend
        from ditserve.models.session.backend import SessionBackend
        with tempfile.TemporaryDirectory() as directory:
            backend = SessionBackend.__new__(SessionBackend)
            Backend.__init__(backend, dict(work_dir=directory))
            backend.pipeline_mode, backend.runtime = False, sana()
            spec = dict(prompt="1 2", seed=11)
            result = backend.prepare(dict(id=1, state_id="sana", spec=spec))
            while not result["progress"]["done"]:
                result = backend.advance(dict(id=2, state_id="sana"))
            finished = backend.finish(dict(id=3, state_id="sana"))
            self.assertTrue(Path(finished["output"]["path"]).is_file())
            self.assertTrue(finished["progress"]["done"])
            backend.close()

    def test_json_adapter_continuous_batch_has_shared_cost_and_independent_progress(self):
        from ditserve.models.base import Backend
        from ditserve.models.session.backend import SessionBackend
        self.assertIn("advance_batch", SessionBackend.operations)
        backend = SessionBackend.__new__(SessionBackend)
        Backend.__init__(backend, dict(work_dir=tempfile.gettempdir()))
        backend.pipeline_mode, backend.runtime = False, sana()
        try:
            for key, prompt, seed in (("a", "kettle", 301), ("b", "clock", 302)):
                backend.prepare(dict(id=key, state_id=key, spec=dict(prompt=prompt, seed=seed)))
            backend.advance(dict(id="a", state_id="a"))
            result = backend.advance_batch(dict(id="join", state_ids=["a", "b"]))
            self.assertEqual(result["batch_size"], 2)
            self.assertEqual([row["state_id"] for row in result["results"]], ["a", "b"])
            self.assertEqual([row["progress"]["step"] for row in result["results"]], [2, 1])
            self.assertEqual([row["progress"]["done"] for row in result["results"]], [True, False])
            self.assertEqual([row["event"]["ms"] for row in result["results"]],
                             [result["model_ms"], result["model_ms"]])
            finished = backend.finish(dict(id="a", state_id="a", output_mode="check"))
            self.assertEqual(finished["output"]["output_mode"], "check")
            backend.release(dict(state_id="a"))
            self.assertNotIn("a", backend.states)
            advanced = backend.advance(dict(id="b", state_id="b"))
            self.assertTrue(advanced["progress"]["done"])
            self.assertTrue(backend.finish(dict(id="b", state_id="b", output_mode="check"))["progress"]["done"])
            backend.pipeline_mode = True
            with self.assertRaisesRegex(ValueError, "explicit-step Sana"):
                backend.advance_batch(dict(state_ids=[]))
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
