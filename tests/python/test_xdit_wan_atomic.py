"""Atomic xDiT Wan pipeline with CPU encoding, fixed batch and decode evidence."""

import importlib.util
import tempfile
import unittest


PINNED = all(importlib.util.find_spec(name) is not None
             for name in ("torch", "diffusers", "cache_dit", "xfuser", "yunchang"))


@unittest.skipUnless(PINNED, "pinned xDiT model environment is required")
class XDitWanAtomicTests(unittest.TestCase):
    def test_atomic_pipeline_preserves_batch_rng_cache_and_engine_evidence(self):
        import cache_dit
        import numpy as np
        import torch
        from ditserve.models.base import Backend
        from ditserve.models.pipeline import _sdpa_math_context
        from ditserve.models.wan.pipeline import WanPipelineBackend
        from ditserve.models.xdit.wan import attach_pipeline
        from xdit_wan_fixtures import native_pipeline

        with tempfile.TemporaryDirectory() as directory:
            backend = WanPipelineBackend.__new__(WanPipelineBackend)
            Backend.__init__(backend, dict(work_dir=directory, text_encoder_device="cpu",
                options=dict(cache=True, cache_fn=1, cache_warmup=1, cache_max_continuous=1, threshold=0.)))
            backend.torch, backend.device = torch, "cpu"
            backend.xdit = attach_pipeline(native_pipeline(torch), "cpu")
            backend.pipe = backend.xdit.pipe
            backend._cache_recipe = None
            self.addCleanup(backend.close)
            spec = dict(prompt="kettle", negative_prompt="", batch_size=2, steps=2,
                        height=32, width=32, frames=5, guidance_scale=5.)
            backend.prepare(dict(state_id="atomic", spec=spec))
            result = backend.advance(dict(id=1, state_id="atomic"))
            output = backend.finish(dict(id=2, state_id="atomic"))
            actual = np.load(output["output"]["path"], allow_pickle=False)

            expected = native_pipeline(torch)
            cache_dit.enable_cache(expected, cache_config=cache_dit.DBCacheConfig(
                Fn_compute_blocks=1, Bn_compute_blocks=0, max_warmup_steps=1, warmup_interval=1,
                max_cached_steps=-1, residual_diff_threshold=0., max_continuous_cached_steps=1,
                enable_separate_cfg=True, cfg_compute_first=False, cfg_diff_compute_separate=True,
                num_inference_steps=2), attention_backend="native")
            with torch.inference_mode(), _sdpa_math_context(torch):
                wanted = expected(prompt=["kettle"] * 2, negative_prompt=[""] * 2,
                    height=32, width=32, num_frames=5, num_inference_steps=2, guidance_scale=5.,
                    generator=torch.Generator(device="cpu").manual_seed(0), output_type="np").frames
            np.testing.assert_array_equal(actual, wanted)
            self.assertEqual(actual.shape, (2, 5, 32, 32, 3))
            self.assertEqual(result["engine"]["transformer"], "xFuserWanTransformer3DWrapper")
            self.assertEqual(output["generation"]["engine"]["revision"], backend.xdit.revision)
            self.assertGreater(result["cpu_prompt_encode_ms"], 0.)


if __name__ == "__main__":
    unittest.main()
