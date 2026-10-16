"""Real CPU module hooks with fake CUDA events; never GPU measurement evidence."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ditserve.models.base import Backend
from ditserve.models.flux import FluxBackend


class ProfilingTests(unittest.TestCase):
    def test_flux_profile_preserves_output_and_restores_hooks_after_forward_failure(self):
        try:
            import torch
            import numpy as np
        except ImportError:
            self.skipTest("CPU torch/numpy required")

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

        class Pipeline:
            def __init__(self):
                self.transformer = torch.nn.Linear(2, 2, bias=False).eval()
                with torch.no_grad():
                    self.transformer.weight.fill_(.25)
                self.scheduler = SimpleNamespace(step=lambda value: value + 1, timesteps=torch.tensor([2, 1]))
                self.vae = SimpleNamespace(decode=lambda value: value.sigmoid())
                self.image_processor = SimpleNamespace(postprocess=lambda value: value.detach().numpy())

            def encode_prompt(self, prompt):
                return prompt

            def prepare_latents(self):
                return torch.ones(1, 2)

            def __call__(self, **kwargs):
                self.encode_prompt(kwargs["prompt"])
                value = self.prepare_latents()
                for step in range(kwargs["num_inference_steps"]):
                    value = self.scheduler.step(self.transformer(value))
                    if "callback_on_step_end" in kwargs:
                        value = kwargs["callback_on_step_end"](self, step, step, {"latents": value})["latents"]
                return SimpleNamespace(images=self.image_processor.postprocess(self.vae.decode(value)))

        backend = FluxBackend.__new__(FluxBackend)
        Backend.__init__(backend, dict(options={}))
        backend.pipe, backend.torch, backend.device = Pipeline(), proxy, "cuda:0"
        outputs = []
        for enabled in (False, True):
            backend.config["options"]["profile"] = enabled
            key = "profiled" if enabled else "baseline"
            backend.prepare(dict(state_id=key, spec=dict(prompt="kettle", steps=2)))
            result = backend.advance(dict(state_id=key))
            outputs.append(backend.states[key]["output"])
            if enabled:
                self.assertEqual(set(result["phase_ms"]), {"text_encode", "latent_init", "transformer", "scheduler", "vae_decode", "postprocess"})
                self.assertEqual(result["profile"]["forward_count"], 2)
                self.assertTrue(all(step["latents_finite"] for step in result["profile"]["steps"]))
                self.assertEqual(result["profile"]["memory"]["peak_allocated_bytes"], 64)
        self.assertTrue(np.array_equal(*outputs))
        self.assertNotIn("encode_prompt", vars(backend.pipe))
        self.assertNotIn("prepare_latents", vars(backend.pipe))
        original_decode = backend.pipe.vae.decode
        backend.pipe.transformer.forward = Mock(side_effect=RuntimeError("failed forward"))
        backend.prepare(dict(state_id="failure", spec=dict(prompt="kettle", steps=2)))
        with self.assertRaisesRegex(RuntimeError, "failed forward"):
            backend.advance(dict(state_id="failure"))
        self.assertFalse(backend.pipe.transformer._forward_pre_hooks)
        self.assertFalse(backend.pipe.transformer._forward_hooks)
        self.assertNotIn("encode_prompt", vars(backend.pipe))
        self.assertIs(backend.pipe.vae.decode, original_decode)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
