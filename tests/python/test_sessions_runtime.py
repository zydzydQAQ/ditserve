"""CPU numerical references for session ownership and resumable model arithmetic.

Mistral uses a real tiny randomly initialized model; Sana uses the installed
official pipeline loop and SCM with small deterministic model components.  No
test loads production weights or initializes CUDA.
"""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from transformers import MistralConfig, MistralForCausalLM
from diffusers import SCMScheduler
from diffusers.pipelines.sana.pipeline_sana_sprint import SanaSprintPipeline

from ditserve.sessions.runtime import (
    MistralRuntime, SanaRuntime, cache_comparison, tensor_comparison,
)
from ditserve.sessions import runtime


class HardwareProfileTests(unittest.TestCase):
    def test_a100_is_explicit_and_does_not_reuse_v100_default(self):
        for constructor in (MistralRuntime, SanaRuntime):
            with self.subTest(runtime=constructor.__name__), \
                    mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
                with self.assertRaisesRegex(ValueError, "v100_fp32 requires CUDA sm70"):
                    constructor("missing-model")

    def test_profiles_reject_cpu_other_architectures_and_unknown_recipe_before_loading(self):
        for constructor in (MistralRuntime, SanaRuntime):
            for capability in ((7, 0), (8, 6), (9, 0)):
                with mock.patch.object(torch.cuda, "get_device_capability", return_value=capability):
                    with self.assertRaisesRegex(ValueError, "a100_fp32 requires CUDA sm80"):
                        constructor("missing-model", profile="a100_fp32")
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                constructor("missing-model", device="cpu", profile="a100_fp32")
            with self.assertRaisesRegex(ValueError, "unknown explicit"):
                constructor("missing-model", profile="auto")

    def test_a100_mistral_loads_local_unquantized_fp16_sdpa_model(self):
        model = type("MistralForCausalLM", (), {})()
        model.config = SimpleNamespace(sliding_window=None)
        model.eval = mock.Mock(return_value=model)
        model.to = mock.Mock(return_value=model)
        model.parameters = lambda: iter([torch.ones(1, dtype=torch.float16)])
        loader = mock.Mock(return_value=model)
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=mock.Mock()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader))
        with mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)), \
                mock.patch.object(runtime.importlib, "import_module",
                                  side_effect={"torch": torch, "transformers": transformers}.__getitem__), \
                mock.patch.object(MistralRuntime, "_configure"):
            adapter = MistralRuntime("missing-model", profile="a100_fp32")
        self.assertEqual(adapter.profile, "a100_fp32")
        self.assertEqual(loader.call_args.kwargs,
                         dict(local_files_only=True, torch_dtype=torch.float16,
                              attn_implementation="sdpa", use_safetensors=True))
        model.to.assert_called_once_with("cuda:0")


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
        return (value * .125 + encoder_hidden_states.mean() * .01
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

    def test_invalid_lifecycles_do_not_consume_live_states(self):
        state = self.adapter.prepare(self.messages, max_new_tokens=1)
        with self.assertRaises(ValueError):
            self.adapter.finish(state)
        with self.assertRaises(ValueError):
            self.adapter.prepare(self.messages, previous=state)
        self.assertFalse(state.closed)
        self.adapter.advance(state)
        with self.assertRaises(ValueError):
            self.adapter.advance(state)
        self.adapter.release(state)
        self.adapter.release(state)
        for value in (0, -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.adapter.prepare(self.messages, max_new_tokens=value)

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

    def test_snapshot_inventory_and_explicit_cpu_move_preserve_state(self):
        state = self.adapter.prepare(self.messages, max_new_tokens=3)
        self.adapter.advance(state)
        snapshot = self.adapter.snapshot(state)
        inventory = self.adapter.inventory(state)
        self.assertGreater(inventory["bytes"], 0)
        self.assertEqual(inventory["device"], "cpu")
        self.assertEqual(inventory["gpu_bytes"], 0)
        self.adapter.offload_to_cpu(state)
        self.adapter.restore(state)
        self.assertTrue(cache_comparison(state.cache, snapshot["cache"])["exact"])
        snapshot["cache"][0][0].zero_()
        self.assertFalse(cache_comparison(state.cache, snapshot["cache"])["exact"])

    def test_lazy_and_populated_cache_reconstruction_without_cuda(self):
        state = self.adapter.prepare(self.messages, max_new_tokens=3)
        # cpu:0 exercises the transfer/reconstruction branch without a GPU.
        moved = self.adapter._move(state, "cpu:0")
        self.assertEqual(moved["bytes"], 0)
        self.adapter.restore(state)
        self.adapter.advance(state)
        before = self.adapter.snapshot(state)
        self.assertGreater(self.adapter._move(state, "cpu:0")["bytes"], 0)
        with self.assertRaisesRegex(ValueError, "offloaded"):
            self.adapter.advance(state)
        self.adapter.restore(state)
        self.assertTrue(cache_comparison(state.cache, before["cache"])["exact"])
        self.complete(state)

    def test_changed_chat_template_prefix_never_reuses_incompatible_tokens(self):
        first = self.adapter.prepare(self.messages, max_new_tokens=2)
        self.complete(first)
        next_turn = self.adapter.prepare([dict(role="user", content="20 21")], previous=first)
        self.assertEqual(next_turn.reused_tokens, 2)  # BOS and user delimiter only.
        self.assertEqual(next_turn.cached_ids, next_turn.prompt_ids[:2])


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

    def test_sana_does_not_advance_process_global_rng(self):
        torch.manual_seed(741)
        before = torch.get_rng_state().clone()
        state = self.adapter.prepare("clock", 301)
        self.adapter.advance(state)
        self.adapter.advance(state)
        self.adapter.finish(state)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_safe_point_validation_and_idempotent_release(self):
        state = self.adapter.prepare("clock", 302)
        with self.assertRaises(ValueError):
            self.adapter.finish(state)
        self.adapter.advance(state)
        self.adapter.advance(state)
        with self.assertRaises(ValueError):
            self.adapter.advance(state)
        output = self.adapter.finish(state)
        self.assertIs(self.adapter.finish(state), output)
        self.adapter.release(state)
        self.adapter.release(state)
        with self.assertRaises(ValueError):
            self.adapter.finish(state)

    def test_nonfinite_step_invalidates_state_and_releases_partial_buffers(self):
        state = self.adapter.prepare("clock", 301)
        with mock.patch.object(self.adapter.pipe.transformer, "forward", side_effect=lambda value, **kwargs: (value * float("nan"),)):
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                self.adapter.advance(state)
        self.assertTrue(state.failed)
        self.adapter.release(state)
        self.assertIsNone(state.latents)
        self.assertIsNone(state.scheduler)

    def test_snapshot_is_owned_and_inventory_includes_sampler_and_rng(self):
        state = self.adapter.prepare("clock", 301)
        self.adapter.advance(state)
        snapshot = self.adapter.snapshot(state)
        inventory = self.adapter.inventory(state)
        names = {row["name"] for row in inventory["tensors"]}
        self.assertIn("sampler.timesteps", names)
        self.assertIn("rng", names)
        self.assertGreater(inventory["bytes"], 0)
        self.assertEqual(inventory["device"], "cpu")
        self.adapter.offload_to_cpu(state)
        self.adapter.restore(state)
        self.assertTrue(tensor_comparison(state.latents, snapshot["latents"])["exact"])
        snapshot["latents"].zero_()
        self.assertFalse(tensor_comparison(state.latents, snapshot["latents"])["exact"])

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
    def test_comparison_rejects_shape_dtype_nonfinite_and_missing_cache_layers(self):
        self.assertFalse(tensor_comparison(torch.ones(2), torch.ones(1))["passed"])
        self.assertFalse(tensor_comparison(torch.ones(2), torch.ones(2, dtype=torch.float64))["passed"])
        self.assertFalse(tensor_comparison(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["passed"])
        self.assertFalse(cache_comparison((), ((torch.ones(1), torch.ones(1)),))["passed"])
        self.assertTrue(tensor_comparison(torch.ones(1), torch.ones(1) + 1e-5, atol=1e-4)["passed"])

    def test_cpu_test_suite_has_not_initialized_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
