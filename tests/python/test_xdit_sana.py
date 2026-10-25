"""Real xFuser Sana wrappers against the native tiny CPU pipeline."""

import importlib.util
import unittest


PINNED = all(importlib.util.find_spec(name) is not None
             for name in ("torch", "diffusers", "cache_dit", "xfuser", "yunchang", "transformers"))


def tiny_pipeline(torch):
    from diffusers import SanaTransformer2DModel
    try:
        from test_model_sessions import Pipeline
    except ImportError:
        from tests.python.test_model_sessions import Pipeline
    torch.manual_seed(51)
    pipeline = Pipeline()
    pipeline.transformer = SanaTransformer2DModel(
        in_channels=2, out_channels=2, num_attention_heads=2, attention_head_dim=4,
        num_layers=1, num_cross_attention_heads=1, cross_attention_head_dim=8,
        cross_attention_dim=8, caption_channels=4, sample_size=32, guidance_embeds=True).eval()

    def encode_prompt(prompt, **kwargs):
        embeddings = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 16
        return embeddings + sum(map(ord, prompt)) / 1000, torch.ones(1, 4)

    pipeline.encode_prompt = encode_prompt
    return pipeline


def reference(torch, pipeline, prompt, seed):
    from ditserve.models.sana.runtime import RECIPE
    steps = []

    def callback(pipe, index, timestep, values):
        steps.append(values["latents"].detach().clone())
        return values

    arguments = dict(RECIPE, output_type="np", complex_human_instruction=[])
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        output = pipeline(prompt=prompt, generator=torch.Generator(device="cpu").manual_seed(seed),
                          callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"],
                          **arguments)
    return output.images, steps


def session_runtime(pipeline):
    from ditserve.models.session.runtime import SanaRuntime
    from ditserve.models.xdit.sana import XDitSanaEngine
    engine = XDitSanaEngine(pipeline, "cpu")
    worker = SanaRuntime.__new__(SanaRuntime)
    worker._initialize_runtime("cpu")
    worker.pipe, worker.xdit, worker._instruction = engine.pipe, engine, []
    worker._configure()
    return worker


@unittest.skipUnless(PINNED, "pinned xDiT model environment is required")
class XDitSanaTests(unittest.TestCase):
    def test_forward_matches_native_transformer(self):
        import torch
        from ditserve.models.xdit.sana import XDitSanaEngine

        pipeline = tiny_pipeline(torch)
        native = pipeline.transformer
        engine = XDitSanaEngine(pipeline, "cpu")
        self.addCleanup(engine.close)
        hidden = torch.randn(2, 2, 4, 4)
        kwargs = dict(encoder_hidden_states=torch.randn(2, 3, 4),
                      encoder_attention_mask=torch.ones(2, 3), guidance=torch.randn(2),
                      timestep=torch.randn(2), return_dict=False, attention_kwargs=None)
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            expected = native(hidden, **kwargs)[0]
            actual = engine.forward(hidden, num_inference_steps=2, **kwargs)[0]
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(engine.runtime.state.input_config.batch_size, 2)
        self.assertEqual(engine.info()["transformer"], "xFuserSanaTransformer2DWrapper")
        self.assertEqual(engine.info()["name"], "xdit")

    def test_ragged_continuous_batch_matches_official_pipeline(self):
        import numpy as np
        import torch

        references = {prompt: reference(torch, tiny_pipeline(torch), prompt, seed)
                      for prompt, seed in (("kettle", 301), ("clock", 302))}
        worker = session_runtime(tiny_pipeline(torch))
        self.addCleanup(worker.close)
        a = worker.prepare("kettle", 301, request_id="a")
        worker.advance(a)
        self.assertTrue(torch.equal(a.latents, references["kettle"][1][0]))
        b = worker.prepare("clock", 302, request_id="b")
        batch = worker.advance_batch([a, b])
        self.assertEqual(batch["batch_size"], 2)
        # Batch GEMM/convolution has a different reduction shape from independent calls.
        torch.testing.assert_close(a.latents, references["kettle"][1][1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(b.latents, references["clock"][1][0], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(worker.finish(a), references["kettle"][0], atol=1e-6, rtol=1e-6)
        worker.release(a)
        worker.advance(b)
        torch.testing.assert_close(b.latents, references["clock"][1][1], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(worker.finish(b), references["clock"][0], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
