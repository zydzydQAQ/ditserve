"""CPU-only adapter checks: submission order, timing scopes, and cleanup."""

from contextlib import contextmanager
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.models.sana.runtime import RECIPE, SanaAdapter


class SanaSyncTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.active_stream = None
        self.stream = object()
        self.embeddings, self.mask = object(), object()
        self.images = [[[1.]]]
        self.finite = True
        self.budget_calls = 0
        self.fail_budget = None
        self.fail_encode = self.fail_generate = False
        self.adapter = SanaAdapter.__new__(SanaAdapter)
        self.adapter.device = "cuda:1"
        self.adapter._owner = object()
        self.adapter._instruction = "unchanged instruction"
        self.adapter.torch = SimpleNamespace(
            inference_mode=lambda: self.context("inference"), Generator=self.generator,
            cuda=SimpleNamespace(synchronize=self.synchronize, current_stream=self.current_stream,
                                 stream=lambda stream: self.context("stream", stream)))
        outer = self

        class Pipe:
            def encode_prompt(self, **kwargs):
                outer.events.append("encode")
                outer.encode_arguments = kwargs
                outer.encode_stream = outer.active_stream
                if outer.fail_encode:
                    raise RuntimeError("encode failed")
                return outer.embeddings, outer.mask

            def __call__(self, **kwargs):
                outer.events.append("generate")
                outer.generate_arguments = kwargs
                outer.generate_stream = outer.active_stream
                if outer.fail_generate:
                    raise RuntimeError("generate failed")
                for index in range(RECIPE["num_inference_steps"]):
                    payload = dict(index=index)
                    outer.assertIs(kwargs["callback_on_step_end"](self, index, 0, payload), payload)
                return SimpleNamespace(images=outer.images)

        self.adapter.pipe = Pipe()

        def finite(images):
            self.assertIs(images, self.images)
            self.events.append("finite")
            return SimpleNamespace(all=lambda: self.finite)

        original_import = importlib.import_module
        patcher = patch("ditserve.models.sana.runtime.importlib.import_module",
                        side_effect=lambda name: SimpleNamespace(isfinite=finite) if name == "numpy" else original_import(name))
        patcher.start()
        self.addCleanup(patcher.stop)

    @contextmanager
    def context(self, name, stream=None):
        self.events.append(name + "_enter")
        previous = self.active_stream
        if name == "stream":
            self.active_stream = stream
        try:
            yield
        finally:
            self.active_stream = previous
            self.events.append(name + "_exit")

    def generator(self, *, device):
        self.assertEqual(device, "cpu")
        self.events.append("generator")
        def seed(value):
            self.seed = value
            return "seeded_generator"
        return SimpleNamespace(manual_seed=seed)

    def synchronize(self, device):
        self.assertEqual(device, "cuda:1")
        self.events.append("sync")

    def current_stream(self, device):
        self.assertEqual(device, "cuda:1")
        self.events.append("current_stream")
        return self.stream

    def budget(self):
        self.budget_calls += 1
        self.events.append("budget")
        if self.budget_calls == self.fail_budget:
            raise TimeoutError("budget expired")

    def admit(self, packet):
        self.events.append("admit")
        self.assertIs(packet.owner, self.adapter._owner)
        self.assertIs(packet.embeddings, self.embeddings)
        self.assertIs(self.active_stream, self.stream)

    def test_fused_same_stream_has_two_endpoint_syncs_and_full_wall_accounting(self):
        with patch("ditserve.models.sana.runtime.time.perf_counter", side_effect=[10., 10.002, 10.010]):
            packet, output, service = self.adapter.encode_and_generate(
                "cat", seed=19, on_encoded=self.admit, check_budget=self.budget)
        self.assertEqual(self.events.count("sync"), 2)
        self.assertEqual(self.events[:3], ["budget", "sync", "current_stream"])
        self.assertEqual(self.events[-3:], ["sync", "finite", "budget"])
        encode, generate = self.events.index("encode"), self.events.index("generate")
        self.assertNotIn("sync", self.events[encode:generate])
        self.assertIn("admit", self.events[encode:generate])
        self.assertIs(self.encode_stream, self.stream)
        self.assertIs(self.generate_stream, self.stream)
        self.assertIsNone(self.active_stream)
        self.assertIs(output, self.images)
        self.assertEqual(self.seed, 19)
        self.assertAlmostEqual(packet.encode_ms, 2.)
        self.assertAlmostEqual(service["generation_ms"], 8.)
        self.assertAlmostEqual(service["ms"], 10.)
        self.assertEqual(service["ms"], service["encode_ms"] + service["generation_ms"])
        self.assertEqual(service["timing_scope"], "fused_miss_host_wall")
        self.assertIn("GPU encoding may still be running", service["encode_timing_scope"])
        self.assertIn("finite and budget checks", service["generation_timing_scope"])
        self.assertEqual(self.encode_arguments, dict(prompt="cat", device="cuda:1", num_images_per_prompt=1,
            clean_caption=False, max_sequence_length=300, complex_human_instruction="unchanged instruction"))
        self.assertNotIn("prompt", self.generate_arguments)
        self.assertIs(self.generate_arguments["prompt_embeds"], self.embeddings)
        self.assertIs(self.generate_arguments["prompt_attention_mask"], self.mask)
        self.assertEqual(self.generate_arguments["output_type"], "np")
        self.assertEqual(self.generate_arguments["generator"], "seeded_generator")
        for key, value in RECIPE.items():
            self.assertEqual(self.generate_arguments[key], value)
        self.assertEqual(self.budget_calls, 6)

    def test_default_staged_path_keeps_four_syncs_and_the_same_pipeline_arguments(self):
        packet = self.adapter.encode("cat", check_budget=self.budget)
        images, service = self.adapter.generate("cat", seed=23, encoded=packet, check_budget=self.budget)
        self.assertEqual(self.events.count("sync"), 4)
        middle = self.events[self.events.index("encode"):self.events.index("generate")]
        self.assertEqual(middle.count("sync"), 2)
        self.assertIs(images, self.images)
        self.assertEqual(self.seed, 23)
        self.assertTrue(service["prompt_preencoded"])
        self.assertNotIn("timing_scope", service)
        self.assertNotIn("prompt", self.generate_arguments)
        self.assertIs(self.generate_arguments["prompt_embeds"], packet.embeddings)

    def test_default_uncached_generate_passes_original_prompt(self):
        self.adapter.generate("cat", seed=29, check_budget=self.budget)
        self.assertEqual(self.events.count("sync"), 2)
        self.assertNotIn("encode", self.events)
        self.assertEqual(self.generate_arguments["prompt"], "cat")
        self.assertNotIn("prompt_embeds", self.generate_arguments)
        self.assertEqual(self.seed, 29)

    def test_fused_drains_stream_if_encode_admission_or_generate_fails(self):
        for failure in ("encode", "admit", "generate"):
            with self.subTest(failure=failure):
                self.events.clear()
                self.fail_encode, self.fail_generate = failure == "encode", failure == "generate"
                def admit(packet):
                    self.admit(packet)
                    if failure == "admit":
                        raise ValueError("admission failed")
                with self.assertRaisesRegex((RuntimeError, ValueError), "failed"):
                    self.adapter.encode_and_generate("cat", on_encoded=admit, check_budget=self.budget)
                self.assertEqual(self.events.count("sync"), 2)
                self.assertEqual(self.events[-1], "sync")
                self.assertNotIn("finite", self.events)
                self.assertIsNone(self.active_stream)

    def test_fused_budget_checks_before_and_after_admission_and_at_each_step(self):
        for failure_call in range(1, 7):
            with self.subTest(failure_call=failure_call):
                self.events.clear()
                self.budget_calls, self.fail_budget = 0, failure_call
                with self.assertRaisesRegex(TimeoutError, "expired"):
                    self.adapter.encode_and_generate("cat", on_encoded=self.admit, check_budget=self.budget)
                self.assertEqual(self.events.count("sync"), 0 if failure_call == 1 else 2)
                if failure_call == 2:
                    self.assertNotIn("admit", self.events)
                if failure_call <= 3:
                    self.assertNotIn("generate", self.events)
                self.assertIsNone(self.active_stream)

    def test_fused_rejects_nonfinite_output_after_final_sync(self):
        self.finite = False
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            self.adapter.encode_and_generate("cat", check_budget=self.budget)
        self.assertEqual(self.events.count("sync"), 2)
        self.assertEqual(self.events[-2:], ["sync", "finite"])


if __name__ == "__main__":
    unittest.main()
