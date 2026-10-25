"""Actual xDiT Sana candidate execution; no CUDA graph qualification implied."""

import importlib.util
import unittest


PINNED = all(importlib.util.find_spec(name) is not None
             for name in ("torch", "diffusers", "cache_dit", "xfuser", "yunchang", "transformers"))


@unittest.skipUnless(PINNED, "pinned xDiT model environment and compiler are required")
class XDitSanaCandidateTests(unittest.TestCase):
    def test_inductor_compiles_and_executes_the_xfuser_transformer(self):
        import torch
        from ditserve.models.sana.compile import OptimizedSanaAdapter
        from ditserve.models.xdit.sana import XDitSanaEngine
        from test_xdit_sana import tiny_pipeline

        torch.set_num_threads(1)
        engine = XDitSanaEngine(tiny_pipeline(torch), "cpu")
        self.addCleanup(engine.close)
        adapter = OptimizedSanaAdapter.__new__(OptimizedSanaAdapter)
        adapter.torch, adapter.device, adapter.pipe = torch, "cpu", engine.pipe
        adapter._eager_transformer = engine.pipe.transformer
        adapter._compiled_transformer = None
        adapter.backend = "eager"
        adapter.compile_records, adapter.compile_setup_ms, adapter._busy = [], 0., False
        adapter.compile_transformer()
        kwargs = dict(encoder_hidden_states=torch.randn(1, 4, 4),
                      encoder_attention_mask=torch.ones(1, 4), guidance=torch.tensor([.45]),
                      timestep=torch.tensor([.5]), return_dict=False, attention_kwargs=None)
        hidden = torch.randn(1, 2, 32, 32)
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            expected = engine.forward(hidden, num_inference_steps=2, **kwargs)[0].clone()
            adapter.set_backend("compiled")
            actual = engine.forward(hidden, num_inference_steps=2, **kwargs)[0]
        self.assertEqual(adapter.backend, "compiled")
        self.assertEqual(len(adapter.compile_records), 1)
        self.assertEqual(adapter.compile_records[0]["status"], "completed")
        self.assertGreater(adapter.compile_records[0]["backend_compile_ms"], 0.)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
