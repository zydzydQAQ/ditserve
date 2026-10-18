"""Block-boundary export/import on a tiny CPU Wan; exactness here is a codec contract, not GPU evidence."""

import copy
import importlib.util
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_model_state import fixture


def tiny_model(torch):
    from diffusers import WanTransformer3DModel
    torch.manual_seed(907)
    return WanTransformer3DModel(num_attention_heads=1, attention_head_dim=12, in_channels=16, out_channels=16,
                                 num_layers=30, ffn_dim=24, freq_dim=4, text_dim=4096, patch_size=(1, 2, 2),
                                 rope_max_seq_len=16).eval().to(torch.float16)


def attach(model):
    from ditserve.models.wan.block import WanBlockExecutor
    worker, state = fixture(cached=False, boundary=2)
    worker.pipe, worker.attention = SimpleNamespace(transformer=model), "math"
    with patch.object(WanBlockExecutor, "_validate_model"):
        executor = WanBlockExecutor(worker)
    return worker, state, executor, executor.attach(state)


class BlockTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ("torch", "diffusers", "cache_dit")):
            raise unittest.SkipTest("pinned CPU dependencies required")
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def transfer(self, blocks):
        from ditserve.models.wan.block import WanBlockExecutor, export_block_state, import_block_state
        torch = self.torch
        model = tiny_model(torch)
        _, reference, executor, continuation = attach(model)
        for _ in range(2):
            executor.advance(continuation, blocks=60)
        worker, state, executor, continuation = attach(model)
        executor.advance(continuation, blocks=60)
        executor.advance(continuation, blocks=blocks)
        payload, timing = export_block_state(executor, continuation)
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=True)
        executor.close(continuation)
        worker.release(state)
        # A separate, empty worker with an independent copy of the same weights.
        target, placeholder = fixture(cached=False, boundary=0)
        target.release(placeholder)
        target.pipe, target.attention = SimpleNamespace(transformer=copy.deepcopy(model)), "math"
        with patch.object(WanBlockExecutor, "_validate_model"):
            target_executor = WanBlockExecutor(target)
        resumed, moved, imported = import_block_state(target_executor, loaded)
        self.assertEqual((moved.branch, moved.next_block), (payload["manifest"]["layout"]["branch"], blocks % 30))
        while resumed.step_index < reference.step_index:
            target_executor.advance(moved, blocks=7)
        return reference, resumed, payload, timing

    def test_mid_conditional_branch_resumes_bit_exactly_in_a_new_worker(self):
        reference, resumed, payload, timing = self.transfer(13)
        self.assertEqual(payload["manifest"]["layout"]["branch"], "cond")
        self.assertFalse(payload["manifest"]["layout"]["conditional"])
        self.assertGreater(timing["packet_bytes"], 0)
        self.assertTrue(self.torch.equal(reference.latents, resumed.latents))

    def test_mid_unconditional_branch_carries_the_conditional_prediction(self):
        reference, resumed, payload, _ = self.transfer(45)
        self.assertEqual(payload["manifest"]["layout"]["branch"], "uncond")
        self.assertTrue(payload["manifest"]["layout"]["conditional"])
        self.assertTrue(self.torch.equal(reference.latents, resumed.latents))

    def test_tampered_packet_is_rejected_before_registration(self):
        from ditserve.models.wan.block import WanBlockExecutor, export_block_state, import_block_state
        model = tiny_model(self.torch)
        worker, state, executor, continuation = attach(model)
        executor.advance(continuation, blocks=5)
        payload, _ = export_block_state(executor, continuation)
        payload["tensors"]["packet.hidden"] = payload["tensors"]["packet.hidden"] + 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            import_block_state(executor, payload)


if __name__ == "__main__":
    unittest.main()
