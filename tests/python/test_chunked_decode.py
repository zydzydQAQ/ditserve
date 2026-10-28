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

    def test_fresh_vae_chunked_decode_initializes_decoder_cache(self):
        from ditserve.models.pipeline import ChunkedDecode, decode_final_latent
        with self.torch.inference_mode():
            decode = ChunkedDecode(self.torch, self.pipe, self.latent.clone())
            decode.advance(1)
            decode.advance(3)
            reference = decode_final_latent(self.torch, self.pipe, self.latent.clone())
            self.assertTrue((decode.finish() == reference).all())

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

    def test_patched_wan22_vae_folds_pixels_back_like_the_whole_decode(self):
        from diffusers import AutoencoderKLWan
        from diffusers.video_processor import VideoProcessor
        from ditserve.models.pipeline import ChunkedDecode, decode_final_latent
        torch = self.torch
        # Wan2.2 TI2V layout: residual blocks and 2x2 pixel patches (3*2*2 channels).
        vae = AutoencoderKLWan(base_dim=8, decoder_base_dim=16, z_dim=16, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
                               temperal_downsample=[False, True, True], is_residual=True, patch_size=2,
                               in_channels=12, out_channels=12).eval()
        pipe = SimpleNamespace(vae=vae, video_processor=VideoProcessor(vae_scale_factor=16))
        latent = torch.randn(1, 16, 3, 2, 2)
        with torch.inference_mode():
            reference = decode_final_latent(torch, pipe, latent.clone())
            for grouping in ([1, 1, 1], [2, 1], [3]):
                decode = ChunkedDecode(torch, pipe, latent.clone())
                for count in grouping:
                    decode.advance(count)
                video = decode.finish()
                self.assertEqual(video.shape, reference.shape)
                self.assertTrue((video == reference).all())

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
