"""CPU pipeline doubles validate offline batch contracts, not GPU performance."""

import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ditserve.models.base import Backend
from ditserve.models.flux import FluxBackend
from ditserve.models.wan.pipeline import WanPipelineBackend


class PipelineBatchTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            import numpy
        except ImportError:
            self.skipTest("CPU torch/numpy required")
        self.torch, self.np = torch, numpy

    def backend(self, cls, config):
        backend = cls.__new__(cls)
        Backend.__init__(backend, config)
        backend.torch, backend.device = self.torch, "cpu"
        return backend

    def test_offline_batches_repeat_prompts_share_seed_and_reencode_each_call(self):
        torch, np = self.torch, self.np
        for cls, encoder in ((FluxBackend, "cuda"), (WanPipelineBackend, "cuda"), (WanPipelineBackend, "cpu")):
            with self.subTest(backend=cls.__name__, encoder=encoder), tempfile.TemporaryDirectory() as directory:
                backend = self.backend(cls, dict(text_encoder_device=encoder, work_dir=directory))
                calls, encodings, outputs = [], [], {}

                class Pipeline:
                    transformer = SimpleNamespace(dtype=torch.float16)

                    def encode_prompt(self, **kwargs):
                        encodings.append(kwargs)
                        count = len(kwargs["prompt"])
                        return torch.ones(count, 3, dtype=torch.float32), torch.zeros(count, 3, dtype=torch.float32)

                    def __call__(self, **kwargs):
                        calls.append(kwargs)
                        count = len(kwargs["prompt"] if "prompt" in kwargs else kwargs["prompt_embeds"])
                        value = torch.randn(count, 4, generator=kwargs["generator"]).numpy()
                        return SimpleNamespace(images=value, frames=value)

                backend.pipe = Pipeline()
                for index, (size, seed) in enumerate(((1, 7), (3, 7), (3, 7), (3, 8))):
                    key = str(index)
                    negative = "noise" if cls is WanPipelineBackend else ""
                    prepared = backend.prepare(dict(state_id=key, spec=dict(prompt="kettle", negative_prompt=negative,
                        seed=seed, batch_size=size, steps=2)))
                    self.assertEqual(prepared["batch_size"], size)
                    result = backend.advance(dict(state_id=key))
                    output = backend.finish(dict(id=index, state_id=key))
                    array = np.load(output["output"]["path"], allow_pickle=False)
                    outputs[index] = array
                    np.testing.assert_array_equal(array, torch.randn(size, 4,
                        generator=torch.Generator(device="cpu").manual_seed(seed)).numpy())
                    self.assertEqual(result["batch_size"], size)
                    self.assertEqual(output["generation"]["batch_size"], size)
                    kwargs = calls[-1]
                    self.assertIsInstance(kwargs["generator"], torch.Generator)
                    self.assertEqual(kwargs["generator"].initial_seed(), seed)
                    if cls is WanPipelineBackend:
                        self.assertEqual(kwargs["num_frames"], 81)
                    if encoder == "cpu":
                        self.assertNotIn("prompt", kwargs)
                        self.assertEqual(encodings[-1]["prompt"], ["kettle"] * size)
                        self.assertEqual(encodings[-1]["negative_prompt"], ["noise"] * size)
                        self.assertEqual(encodings[-1]["device"], "cpu")
                        self.assertEqual(encodings[-1]["dtype"], torch.float32)
                        self.assertEqual(encodings[-1]["max_sequence_length"], 512)
                        self.assertEqual(kwargs["prompt_embeds"].dtype, torch.float16)
                        self.assertFalse(result["prompt_embeddings_reused"])
                    else:
                        self.assertEqual(kwargs["prompt"], ["kettle"] * size)
                        if cls is WanPipelineBackend:
                            self.assertEqual(kwargs["negative_prompt"], ["noise"] * size)
                    with self.assertRaisesRegex(ValueError, "already completed"):
                        backend.advance(dict(state_id=key))
                    backend.release(dict(state_id=key))
                np.testing.assert_array_equal(outputs[1], outputs[2])
                self.assertFalse(np.array_equal(outputs[1], outputs[3]))
                if encoder == "cpu":
                    self.assertEqual(len(encodings), 4)
        self.assertFalse(torch.cuda.is_initialized())

    def test_invalid_batch_shapes_fail_before_state_creation(self):
        for cls in (FluxBackend, WanPipelineBackend):
            backend = self.backend(cls, {})
            bad = [dict(batch_size=0), dict(batch_size=True), dict(batch_size=1.5), dict(prompt=["kettle"]),
                   dict(seed=True), dict(seed=1 << 64), dict(guidance_scale=float("nan")), dict(height=17)]
            bad.append(dict(frames=2))
            if cls is FluxBackend:
                bad.append(dict(negative_prompt="noise"))
            for spec in bad:
                with self.subTest(backend=cls.__name__, spec=spec), self.assertRaises(ValueError):
                    backend.prepare(dict(state_id="invalid", spec=dict(dict(prompt="kettle"), **spec)))
                self.assertFalse(backend.states)

    def test_wan_loader_keeps_cpu_encoder_and_component_precisions(self):
        from ditserve.models.pipeline import _pipeline
        torch = self.torch
        calls = {}

        class Module:
            def __init__(self):
                self.to = Mock(return_value=self)

        class Wan:
            def __init__(self, vae, encoder):
                self.transformer, self.vae, self.text_encoder = Module(), vae, encoder
                self.transformer.device = "cuda:0"
                self.to = Mock(side_effect=AssertionError("must never move CPU encoder through CUDA"))
                self.set_progress_bar_config = Mock()

            @classmethod
            def from_pretrained(cls, model, **kwargs):
                calls["pipeline"] = kwargs
                return cls(kwargs["vae"], kwargs["text_encoder"])

        vae_loader, encoder_loader = Mock(return_value=Module()), Mock(return_value=Module())
        diffusers = SimpleNamespace(WanPipeline=Wan, AutoencoderKLWan=SimpleNamespace(from_pretrained=vae_loader))
        args = SimpleNamespace(model="wan", model_id="snapshot", revision="pinned", local_files_only=True,
            dtype="float16", vae_dtype="float32", text_encoder_dtype="float32", text_encoder_device="cpu", device="cuda:0")
        with patch("ditserve.models.pipeline.importlib.import_module", return_value=SimpleNamespace(
                UMT5EncoderModel=SimpleNamespace(from_pretrained=encoder_loader))):
            pipe = _pipeline(torch, diffusers, args)
        self.assertEqual(vae_loader.call_args.kwargs["torch_dtype"], torch.float32)
        self.assertEqual(encoder_loader.call_args.kwargs["torch_dtype"], torch.float32)
        self.assertEqual(calls["pipeline"]["torch_dtype"], torch.float16)
        self.assertTrue(calls["pipeline"]["local_files_only"])
        self.assertEqual(calls["pipeline"]["revision"], "pinned")
        pipe.transformer.to.assert_called_once_with(device="cuda:0", dtype=torch.float16)
        pipe.vae.to.assert_called_once_with(device="cuda:0", dtype=torch.float32)
        pipe.text_encoder.to.assert_called_once_with(device="cpu", dtype=torch.float32)
        self.assertEqual(pipe._execution_device, "cuda:0")
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
