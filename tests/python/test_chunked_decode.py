"""Chunked Wan VAE decode against the pinned whole decode on a tiny CPU VAE."""

import importlib.util
import unittest
from types import SimpleNamespace


@unittest.skipIf(any(importlib.util.find_spec(name) is None for name in ("torch", "diffusers")), "torch/diffusers required")
class ChunkedDecodeTests(unittest.TestCase):
    def setUp(self):
        import torch
        from diffusers import AutoencoderKLWan
        from diffusers.video_processor import VideoProcessor
        torch.set_num_threads(1)
        torch.manual_seed(5)
        vae = AutoencoderKLWan(base_dim=8, z_dim=16, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
                               temperal_downsample=[False, True, True]).eval()
        self.torch = torch
        self.pipe = SimpleNamespace(vae=vae, video_processor=VideoProcessor(vae_scale_factor=8))
        self.latent = torch.randn(1, 16, 4, 4, 4)

    def test_chunks_in_any_grouping_match_the_whole_decode_bit_exactly(self):
        from ditserve.models.pipeline import ChunkedDecode, decode_final_latent
        torch = self.torch
        with torch.inference_mode():
            reference = decode_final_latent(torch, self.pipe, self.latent.clone())
            for grouping in ([1, 1, 1, 1], [3, 5], [4]):
                decode = ChunkedDecode(torch, self.pipe, self.latent.clone())
                for count in grouping:
                    decode.advance(count)
                self.assertTrue(decode.done)
                self.assertTrue((decode.finish() == reference).all())

    def test_interleaved_requests_keep_separate_temporal_caches(self):
        from ditserve.models.pipeline import ChunkedDecode, decode_final_latent
        torch = self.torch
        other = torch.randn(1, 16, 4, 4, 4)
        with torch.inference_mode():
            expected = [decode_final_latent(torch, self.pipe, value.clone()) for value in (self.latent, other)]
            first, second = ChunkedDecode(torch, self.pipe, self.latent.clone()), ChunkedDecode(torch, self.pipe, other.clone())
            while not (first.done and second.done):
                first.advance(1)
                second.advance(2)
            self.assertTrue((first.finish() == expected[0]).all())
            self.assertTrue((second.finish() == expected[1]).all())
            with self.assertRaises(ValueError):
                ChunkedDecode(torch, self.pipe, self.latent.clone()).finish()


if __name__ == "__main__":
    unittest.main()
