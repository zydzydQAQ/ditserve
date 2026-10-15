"""CPU-only adapter tests, including SCM-style repeated random draws."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ditserve.optimizations.sana import OptimizedSanaAdapter


class Event:
    def __init__(self, **kwargs):
        pass

    def record(self):
        pass

    def elapsed_time(self, other):
        return 1.


class Pipeline:
    def __init__(self):
        self.vae = SimpleNamespace(decode=lambda array, **kwargs: array)
        self.transformer = object()
        self.generators = None

    def encode_prompt(self, **kwargs):
        return None

    def __call__(self, **kwargs):
        self.encode_prompt(prompt=kwargs.get("prompt"))
        generators = kwargs["generator"]
        self.generators = generators if isinstance(generators, list) else [generators]
        # Initial latent and two independent SCM step draws must use the same
        # generator belonging to each request, regardless of batch position.
        values = [torch.randn((1, 2, 2, 3), generator=generator)
                  for generator in self.generators]
        for step in range(2):
            values = [value + torch.randn(value.shape, generator=generator)
                      for value, generator in zip(values, self.generators)]
            kwargs["callback_on_step_end"](self, step, step, {})
        array = torch.sigmoid(torch.cat(values)).numpy()
        return SimpleNamespace(images=self.vae.decode(array))


def adapter():
    instance = OptimizedSanaAdapter.__new__(OptimizedSanaAdapter)
    instance.pipe, instance.device = Pipeline(), "cpu"
    instance.torch = SimpleNamespace(
        Generator=torch.Generator, cat=torch.cat, inference_mode=torch.inference_mode,
        cuda=SimpleNamespace(Event=Event, synchronize=lambda device: None,
                             memory_allocated=lambda device: 10,
                             memory_reserved=lambda device: 20,
                             max_memory_allocated=lambda device: 30,
                             reset_peak_memory_stats=lambda device: None))
    instance.backend, instance._owner, instance._busy = "eager", object(), False
    instance.compile_records, instance.compile_setup_ms = [], 0.
    instance._eager_transformer = instance.pipe.transformer
    instance._compiled_transformer = None
    return instance


class SanaAdapterTests(unittest.TestCase):
    def test_batch_rng_matches_single_at_all_three_draws_and_permutation(self):
        instance = adapter()
        a, _ = instance.generate("a", seed=101)
        b, _ = instance.generate("b", seed=202)
        batch, metrics = instance.generate_batch(["a", "b"], seeds=[101, 202])
        self.assertIsNot(instance.pipe.generators[0], instance.pipe.generators[1])
        np.testing.assert_array_equal(batch[0:1], a)
        np.testing.assert_array_equal(batch[1:2], b)
        reverse, _ = instance.generate_batch(["b", "a"], seeds=[202, 101])
        np.testing.assert_array_equal(reverse[0:1], b)
        np.testing.assert_array_equal(reverse[1:2], a)
        self.assertEqual(metrics["batch_size"], 2)
        self.assertEqual(metrics["compile_delta"], 0)

    def test_equal_seeds_still_have_distinct_streams(self):
        instance = adapter()
        result, _ = instance.generate_batch(["a", "a"], seeds=[5, 5])
        self.assertIsNot(instance.pipe.generators[0], instance.pipe.generators[1])
        np.testing.assert_array_equal(result[0], result[1])

    def test_stage_timestamps_have_explicit_origin_and_host_scope(self):
        _, result = adapter().generate("a", seed=4)
        self.assertGreater(result["origin_perf_counter_s"], 0)
        stages = {row["stage"]: row for row in result["stages"]}
        for name in ("text_encode", "vae_decode"):
            self.assertGreaterEqual(stages[name]["end_ms"], stages[name]["start_ms"])
            self.assertIn("cuda_elapsed_ms", stages[name])
        self.assertEqual(stages["denoise_step_submitted"]["index"], 1)

    def test_budget_exception_restores_wrapped_methods_and_busy_guard(self):
        instance = adapter()
        encode, decode = instance.pipe.encode_prompt, instance.pipe.vae.decode
        calls = []
        def budget():
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("budget")
        with self.assertRaisesRegex(RuntimeError, "budget"):
            instance.generate_batch(["a", "b"], seeds=[1, 2], check_budget=budget)
        self.assertEqual(instance.pipe.encode_prompt, encode)
        self.assertEqual(instance.pipe.vae.decode, decode)
        self.assertFalse(instance._busy)
        instance.generate("a", seed=1)

    def test_invalid_arity_seed_and_foreign_encoding_rejected_before_pipeline(self):
        instance = adapter()
        for prompts, seeds in [([], []), (["a", "b"], [1]), (["a"] * 3, [1, 2, 3])]:
            with self.assertRaises(ValueError):
                instance.generate_batch(prompts, seeds=seeds)
        with self.assertRaises(TypeError):
            instance.generate("a", seed=True)
        with self.assertRaisesRegex(ValueError, "another"):
            instance.generate("a", encoded=SimpleNamespace(owner=object(), prompt="a"))

    def test_compile_passes_config_patches_and_counts_backend_work(self):
        instance = adapter()
        instance.torch = SimpleNamespace(compile=lambda module, **kwargs: kwargs["backend"]("graph", []))
        observed = []
        def backend(graph, inputs, **kwargs):
            observed.append(kwargs)
            return "compiled"
        with patch("torch._dynamo.backends.registry.lookup_backend", return_value=backend):
            result = instance.compile_transformer()
        self.assertEqual(observed, [{"config_patches": {"triton.cudagraphs": False}}])
        self.assertEqual(result["graph_compile_count"], 1)
        self.assertEqual(result["records"][0]["status"], "completed")

    def test_real_cpu_inductor_compiles_and_reuses_graph(self):
        # This executes only CPU tensors. A real backend invocation catches
        # option-forwarding defects which a mocked torch.compile cannot catch.
        class Arithmetic(torch.nn.Module):
            def forward(self, value):
                return torch.sin(value) + value.square()
        instance = adapter()
        instance.torch = torch
        instance._eager_transformer = Arithmetic()
        instance.compile_transformer()
        value = torch.arange(8, dtype=torch.float32)
        with torch.inference_mode():
            actual = instance._compiled_transformer(value)
            with torch.compiler.set_stance("fail_on_recompile"):
                repeated = instance._compiled_transformer(value)
        torch.testing.assert_close(actual, instance._eager_transformer(value))
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
        self.assertGreaterEqual(len(instance.compile_records), 1)
        self.assertTrue(all(row["status"] == "completed" for row in instance.compile_records))

    def test_official_scm_cpu_rng_is_per_request_through_initial_and_step_noise(self):
        from diffusers import SCMScheduler, SanaSprintPipeline
        def run(seeds):
            generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]
            scheduler = SCMScheduler()
            scheduler.set_timesteps(2, device="cpu", max_timesteps=1.57080, intermediate_timesteps=1.3)
            # SanaSprintPipeline resets this before each invocation. Without
            # it SCM's scalar timestep lookup cannot accept a batch vector.
            scheduler.set_begin_index(0)
            latents = SanaSprintPipeline.prepare_latents(
                SimpleNamespace(vae_scale_factor=32), len(seeds), 4, 64, 64,
                torch.float32, torch.device("cpu"), generators)
            observations = [latents.clone()]
            latents = latents * scheduler.config.sigma_data
            for timestep in scheduler.timesteps[:-1]:
                latents, _ = scheduler.step(torch.zeros_like(latents), timestep.expand(len(seeds)),
                                           latents, generator=generators, return_dict=False)
                observations.append(latents.clone())
            return observations
        single_a, single_b, batch, reverse = run([101]), run([202]), run([101, 202]), run([202, 101])
        for a, b, both, swapped in zip(single_a, single_b, batch, reverse):
            torch.testing.assert_close(both[0:1], a, rtol=0, atol=0)
            torch.testing.assert_close(both[1:2], b, rtol=0, atol=0)
            torch.testing.assert_close(swapped[0:1], b, rtol=0, atol=0)
            torch.testing.assert_close(swapped[1:2], a, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
