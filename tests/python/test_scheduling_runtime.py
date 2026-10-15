from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from ditserve.directions.sana import SanaAdapter
from ditserve.scheduling import runtime


class Parameter:
    def __init__(self, dtype="float32", floating=True):
        self.dtype, self.floating = dtype, floating

    def is_floating_point(self):
        return self.floating


class Component:
    def __init__(self, dtype="float32"):
        self.values = [Parameter(dtype), Parameter("int64", floating=False)]

    def parameters(self):
        return iter(self.values)


class FakePipe:
    def __init__(self):
        self.scheduler = type("SCMScheduler", (), {})()
        self.transformer = type("SanaTransformer2DModel", (Component,), {})()
        self.text_encoder, self.vae = Component(), Component()
        self.to = Mock(return_value=self)
        self.set_progress_bar_config = Mock()

    def __call__(self, complex_human_instruction="instruction", **kwargs):
        raise AssertionError("a mocked construction test must not generate an image")


class SchedulingRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model = Path(temporary.name)
        (self.model / "model_index.json").write_text("{}")
        self.pipe = FakePipe()
        self.loader = Mock(return_value=self.pipe)
        self.attention_events = []

        @contextmanager
        def sdpa_kernel(backends):
            self.attention_events.append(("enter", backends))
            try:
                yield
            finally:
                self.attention_events.append(("exit", backends))

        self.torch = SimpleNamespace(float32="float32", bfloat16="bfloat16",
            cuda=SimpleNamespace(get_device_capability=Mock(return_value=(7, 0))),
            nn=SimpleNamespace(attention=SimpleNamespace(sdpa_kernel=sdpa_kernel,
                                                        SDPBackend=SimpleNamespace(MATH="math"))))
        self.diffusers = SimpleNamespace(SanaSprintPipeline=SimpleNamespace(from_pretrained=self.loader))
        modules = {"torch": self.torch, "diffusers": self.diffusers}
        importer = patch.object(runtime.importlib, "import_module", side_effect=modules.__getitem__)
        importer.start()
        self.addCleanup(importer.stop)

    def test_original_and_a100_profiles_retain_bf16_architecture_guard(self):
        for adapter in (lambda: SanaAdapter(self.model), lambda: runtime.SchedulingSana(self.model)):
            with self.assertRaisesRegex(ValueError, "BF16 requires Ampere"):
                adapter()
        self.loader.assert_not_called()

    def test_a100_profile_delegates_to_original_bf16_resident_loader(self):
        self.torch.cuda.get_device_capability.return_value = (8, 0)
        adapter = runtime.SchedulingSana(self.model, profile="a100_bf16")
        self.loader.assert_called_once_with(str(self.model), torch_dtype="bfloat16", local_files_only=True)
        self.pipe.to.assert_called_once_with("cuda:0")
        with adapter._attention():
            pass
        self.assertEqual(self.attention_events, [])

    def test_v100_profile_explicitly_loads_all_components_fp32_without_offload(self):
        adapter = runtime.SchedulingSana(self.model, profile="v100_fp32")
        self.loader.assert_called_once_with(str(self.model), torch_dtype="float32", local_files_only=True)
        self.pipe.to.assert_called_once_with("cuda:0")
        self.pipe.set_progress_bar_config.assert_called_once_with(disable=True)
        self.assertEqual(adapter._instruction, "instruction")
        self.assertEqual(adapter.model_path, str(self.model))

    def test_a100_session_recipe_keeps_fp32_and_math_attention(self):
        self.torch.cuda.get_device_capability.return_value = (8, 0)
        adapter = runtime.SchedulingSana(self.model, profile="a100_fp32")
        self.loader.assert_called_once_with(str(self.model), torch_dtype="float32", local_files_only=True)
        self.pipe.to.assert_called_once_with("cuda:0")
        with adapter._attention():
            self.assertEqual(self.attention_events[-1], ("enter", ["math"]))
        self.assertEqual(self.attention_events[-1], ("exit", ["math"]))

    def test_a100_session_recipe_rejects_other_architecture_before_loading(self):
        for capability in ((7, 0), (8, 6), (9, 0)):
            self.torch.cuda.get_device_capability.return_value = capability
            with self.assertRaisesRegex(ValueError, "a100_fp32 requires an Ampere sm80"):
                runtime.SchedulingSana(self.model, profile="a100_fp32")
        self.loader.assert_not_called()

    def test_any_floating_component_in_lower_precision_rejects_v100_recipe(self):
        for name in ("text_encoder", "transformer", "vae"):
            with self.subTest(component=name):
                component = getattr(self.pipe, name)
                component.values[0].dtype = "float16"
                with self.assertRaisesRegex(ValueError, "did not remain FP32"):
                    runtime.SchedulingSana(self.model, profile="v100_fp32")
                component.values[0].dtype = "float32"

    def test_v100_profile_rejects_other_architecture_and_unknown_profile_before_loading(self):
        for capability in ((8, 0), (7, 5)):
            self.torch.cuda.get_device_capability.return_value = capability
            with self.assertRaisesRegex(ValueError, "requires a Volta"):
                runtime.SchedulingSana(self.model, profile="v100_fp32")
        with self.assertRaisesRegex(ValueError, "unknown explicit"):
            runtime.SchedulingSana(self.model, profile="automatic_fp16")
        self.loader.assert_not_called()

    def test_v100_encoding_and_full_generation_are_both_inside_math_attention_context(self):
        adapter = runtime.SchedulingSana(self.model, profile="v100_fp32")
        check = Mock()
        encoded_prompt = object()
        calls = []
        def encode(instance, prompt, check_budget):
            self.assertIs(instance, adapter)
            self.assertEqual(self.attention_events[-1], ("enter", ["math"]))
            self.assertIs(check_budget, check)
            calls.append(("encode", prompt))
            return encoded_prompt
        def generate(instance, prompt, *, seed, encoded, check_budget):
            self.assertIs(instance, adapter)
            self.assertEqual(self.attention_events[-1], ("enter", ["math"]))
            self.assertEqual((prompt, seed), ("live prompt", 301))
            self.assertIs(encoded, encoded_prompt)
            self.assertIs(check_budget, check)
            calls.append(("generate", prompt))
            return "image", dict(ms=8., prompt_preencoded=True)
        with patch.object(SanaAdapter, "encode", new=encode), patch.object(SanaAdapter, "generate", new=generate), \
             patch.object(runtime.time, "perf_counter", side_effect=[10., 10.002, 10.010]):
            self.assertIs(adapter.encode("live prompt", check), encoded_prompt)
            image, service = adapter.generate("live prompt", seed=301, check_budget=check)
        self.assertEqual(image, "image")
        self.assertEqual(calls, [("encode", "live prompt"), ("encode", "live prompt"), ("generate", "live prompt")])
        self.assertAlmostEqual(service["ms"], 10.)
        self.assertAlmostEqual(service["encode_ms"], 2.)
        self.assertEqual(service["encode_start_offset_ms"], 0.)
        self.assertAlmostEqual(service["encode_end_offset_ms"], 2.)
        self.assertEqual(service["encoding_device"], "cuda:0")
        self.assertFalse(service["prompt_preencoded"])
        self.assertEqual(self.attention_events,
                         [("enter", ["math"]), ("exit", ["math"])] * 2)

    def test_explicit_preencoded_path_is_distinguished_and_does_not_encode_twice(self):
        adapter = runtime.SchedulingSana(self.model, profile="v100_fp32")
        encoded = object()
        expected = ("image", dict(ms=8., prompt_preencoded=True))
        with patch.object(SanaAdapter, "encode") as encoder, \
             patch.object(SanaAdapter, "generate", return_value=expected) as generator:
            self.assertIs(adapter.generate("prompt", encoded=encoded), expected)
        encoder.assert_not_called()
        self.assertIs(generator.call_args.kwargs["encoded"], encoded)
        self.assertTrue(expected[1]["prompt_preencoded"])

    def test_math_context_is_restored_after_generation_error(self):
        adapter = runtime.SchedulingSana(self.model, profile="v100_fp32")
        with patch.object(SanaAdapter, "encode", return_value=object()), \
             patch.object(SanaAdapter, "generate", side_effect=RuntimeError("failed model")):
            with self.assertRaisesRegex(RuntimeError, "failed model"):
                adapter.generate("live prompt")
        self.assertEqual(self.attention_events, [("enter", ["math"]), ("exit", ["math"])])


if __name__ == "__main__":
    unittest.main()
