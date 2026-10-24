"""CPU numerical references for session ownership and resumable model arithmetic.

Mistral uses a real tiny randomly initialized model; Sana uses the installed
official pipeline loop and SCM with small deterministic model components.  No
test loads production weights or initializes CUDA.
"""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import importlib.util
if any(importlib.util.find_spec(name) is None for name in ("torch", "numpy", "diffusers", "transformers")):
    raise unittest.SkipTest("optional pinned CPU model dependencies unavailable")

import numpy as np
import torch
from transformers import MistralConfig, MistralForCausalLM
from diffusers import SCMScheduler
from diffusers.pipelines.sana.pipeline_sana_sprint import SanaSprintPipeline

from ditserve.models.session.runtime import (
    MistralRuntime, SanaRuntime, cache_comparison, tensor_comparison,
)


class Tokenizer:
    eos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        ids = [1]
        for message in messages:
            tokens = [int(value) for value in message["content"].split()]
            ids.extend([3, *tokens, 4] if message["role"] == "user" else [*tokens, 2])
        return ids

    def decode(self, ids, **kwargs):
        return " ".join(str(value) for value in ids)


def llm():
    torch.manual_seed(71)
    config = MistralConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=128, sliding_window=None,
                          attention_dropout=0., eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "sdpa"
    adapter = MistralRuntime.__new__(MistralRuntime)
    adapter._initialize_runtime("cpu")
    adapter.model = MistralForCausalLM(config).eval()
    adapter.tokenizer = Tokenizer()
    adapter._configure()
    return adapter


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


def sana():
    adapter = SanaRuntime.__new__(SanaRuntime)
    adapter._initialize_runtime("cpu")
    adapter.pipe = Pipeline()
    adapter._instruction = []
    adapter._configure()
    return adapter


class MistralRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = llm()
        self.messages = [dict(role="user", content="6 7 8")]

    def tearDown(self):
        self.adapter.close()

    def complete(self, state):
        actions = []
        while not state.done:
            actions.append(self.adapter.advance(state))
        return actions

    def test_tokens_and_kv_match_official_greedy_generate(self):
        expected = self.adapter.reference(self.messages, max_new_tokens=5)
        state = self.adapter.prepare(self.messages, max_new_tokens=5, request_id="a")
        self.assertEqual(state.cached_tokens, 0)
        actions = self.complete(state)
        self.assertEqual([row["action"] for row in actions], ["prefill", *(["decode"] * 4)])
        self.assertEqual(state.generated_ids, expected["token_ids"])
        self.assertEqual(self.adapter.finish(state), expected["text"])
        self.assertTrue(cache_comparison(state.cache, expected["cache"])["exact"])
        self.assertEqual(state.cached_tokens, len(state.prompt_ids) + len(state.generated_ids) - 1)
        self.assertTrue(all(row["ms"] >= 0 for row in actions))

    def test_interleaved_sessions_do_not_share_kv_or_change_output(self):
        expected = self.adapter.reference(self.messages, max_new_tokens=4)
        a = self.adapter.prepare(self.messages, max_new_tokens=4)
        self.adapter.advance(a)
        before = self.adapter.snapshot(a)
        b = self.adapter.prepare([dict(role="user", content="9 10")], max_new_tokens=3)
        self.complete(b)
        self.assertTrue(cache_comparison(a.cache, before["cache"])["exact"])
        self.complete(a)
        self.assertEqual(a.generated_ids, expected["token_ids"])
        self.assertTrue(cache_comparison(a.cache, expected["cache"])["exact"])

    def test_second_turn_reuses_exact_prefix_and_transfers_ownership_once(self):
        first = self.adapter.prepare(self.messages, max_new_tokens=3)
        self.complete(first)
        messages = [*self.messages, dict(role="assistant", content=self.adapter.finish(first)),
                    dict(role="user", content="11 12")]
        before = self.adapter.snapshot(first)
        second = self.adapter.prepare(messages, max_new_tokens=3, previous=first)
        self.assertGreater(second.reused_tokens, 0)
        self.assertTrue(first.closed)
        self.assertIsNone(first.cache)
        cropped = tuple(tuple(t[:, :, :second.reused_tokens] for t in pair) for pair in before["cache"])
        self.assertTrue(cache_comparison(second.cache, cropped)["exact"])
        expected = self.adapter.reference(messages, max_new_tokens=3)
        self.complete(second)
        self.assertEqual(second.generated_ids, expected["token_ids"])
        # A full-length prefill is a distinct reduction shape; this comparison is numerical.
        self.assertTrue(cache_comparison(second.cache, expected["cache"], atol=1e-6, rtol=1e-6)["passed"])
        self.adapter.release(first)
        with self.assertRaises(ValueError):
            self.adapter.advance(first)
        self.assertEqual(second.messages, messages)


    def test_nonfinite_forward_fails_state_and_cleanup_still_works(self):
        state = self.adapter.prepare(self.messages, max_new_tokens=2)
        original = self.adapter.model.forward

        def poisoned(*args, **kwargs):
            result = original(*args, **kwargs)
            result.logits.fill_(float("nan"))
            return result

        with mock.patch.object(self.adapter.model, "forward", side_effect=poisoned):
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                self.adapter.advance(state)
        self.assertTrue(state.failed)
        with self.assertRaises(ValueError):
            self.adapter.advance(state)
        self.adapter.release(state)





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
        for kind, factory in (("mistral", llm), ("sana", sana)):
            with self.subTest(kind=kind):
                runtime, other = factory(), factory()
                prompt = [dict(role="user", content="1 2")] if kind == "mistral" else "kettle"
                state, foreign = runtime.prepare(prompt), other.prepare(prompt)
                try:
                    with self.assertRaisesRegex(ValueError, "foreign"):
                        runtime.release(foreign)
                    self.assertFalse(foreign.closed)
                    state.failed = True
                    runtime.release(state)
                    runtime.release(state)
                    self.assertTrue(state.closed)
                    self.assertFalse(runtime._states)
                    if kind == "mistral":
                        self.assertIsNone(state.cache)
                    else:
                        for name in ("latents", "prompt_embeds", "scheduler", "generator", "output"):
                            self.assertIsNone(getattr(state, name))
                finally:
                    runtime.close()
                    other.close()
                self.assertIsNone(getattr(runtime, "model" if kind == "mistral" else "pipe"))
                self.assertTrue(foreign.closed)

    def test_comparison_rejects_shape_dtype_nonfinite_and_missing_cache_layers(self):
        self.assertFalse(tensor_comparison(torch.ones(2), torch.ones(1))["passed"])
        self.assertFalse(tensor_comparison(torch.ones(2), torch.ones(2, dtype=torch.float64))["passed"])
        self.assertFalse(tensor_comparison(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["passed"])
        self.assertFalse(cache_comparison((), ((torch.ones(1), torch.ones(1)),))["passed"])
        self.assertTrue(tensor_comparison(torch.ones(1), torch.ones(1) + 1e-5, atol=1e-4)["passed"])

    def test_cpu_test_suite_has_not_initialized_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_json_adapter_progress_and_output_cover_both_runtimes(self):
        from ditserve.models.base import Backend
        from ditserve.models.session.backend import SessionBackend
        with tempfile.TemporaryDirectory() as directory:
            for kind, runtime in (("mistral", llm()), ("sana", sana())):
                with self.subTest(kind=kind):
                    backend = SessionBackend.__new__(SessionBackend)
                    Backend.__init__(backend, dict(work_dir=directory))
                    backend.kind, backend.pipeline_mode, backend.runtime = kind, False, runtime
                    spec = dict(prompt="1 2", max_new_tokens=3, seed=11)
                    result = backend.prepare(dict(id=1, state_id=kind, spec=spec))
                    while not result["progress"]["done"]:
                        result = backend.advance(dict(id=2, state_id=kind))
                    finished = backend.finish(dict(id=3, state_id=kind))
                    self.assertTrue(Path(finished["output"]["path"]).is_file())
                    self.assertTrue(finished["progress"]["done"])
                    if kind == "mistral":
                        backend.prepare(dict(id=4, state_id="round2", previous_state_id=kind, spec=spec))
                        self.assertNotIn(kind, backend.states)
                    backend.close()

    def test_json_adapter_continuous_batch_has_shared_cost_and_independent_progress(self):
        from ditserve.models.base import Backend
        from ditserve.models.session.backend import SessionBackend
        self.assertIn("advance_batch", SessionBackend.operations)
        backend = SessionBackend.__new__(SessionBackend)
        Backend.__init__(backend, dict(work_dir=tempfile.gettempdir()))
        backend.kind, backend.pipeline_mode, backend.runtime = "sana", False, sana()
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
            backend.kind = "mistral"
            with self.assertRaisesRegex(ValueError, "explicit-step Sana"):
                backend.advance_batch(dict(state_ids=[]))
            backend.kind, backend.pipeline_mode = "sana", True
            with self.assertRaisesRegex(ValueError, "explicit-step Sana"):
                backend.advance_batch(dict(state_ids=[]))
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
