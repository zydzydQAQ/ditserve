"""Real xFuser Wan wrapper and block continuation contracts on CPU."""

import importlib.util
from types import SimpleNamespace
import unittest

try:
    from test_sp_block import WanPipelineStub, plain_executor, run_quanta, tiny_model
except ImportError:
    from tests.python.test_sp_block import WanPipelineStub, plain_executor, run_quanta, tiny_model


PINNED = all(importlib.util.find_spec(name) for name in ("torch", "diffusers", "cache_dit", "xfuser", "yunchang"))


@unittest.skipUnless(PINNED, "pinned xDiT model environment is required")
class XDitWanTests(unittest.TestCase):
    def test_whole_forward_and_block_packet_match_native(self):
        import torch
        from ditserve.models.xdit.runtime import XDitRuntime
        from ditserve.models.xdit.wan import convert_wan_transformer
        from ditserve.models.wan.parallel import FixedLayout

        native = tiny_model(torch)
        wrapped = convert_wan_transformer(native)
        runtime = XDitRuntime(WanPipelineStub(wrapped), model="wan", device="cpu", attention="math")
        self.addCleanup(runtime.close)
        runtime.activate(FixedLayout((0,)), height=32, width=32, num_frames=5, num_inference_steps=2)
        torch.manual_seed(19)
        latents = torch.randn(1, 16, 2, 4, 4, dtype=torch.float16)
        prompt = torch.randn(1, 8, 4096, dtype=torch.float16)
        timestep = torch.tensor([3])
        attention = torch.nn.attention
        with torch.inference_mode(), attention.sdpa_kernel([attention.SDPBackend.MATH]):
            expected = native(latents, timestep, prompt, return_dict=False)[0]
            actual = wrapped(latents, timestep, prompt, return_dict=False)[0]
            packet = wrapped.begin_packet(latents, timestep, prompt)
            for index in range(len(wrapped.blocks)):
                wrapped.forward_block(packet, index)
            blocked = wrapped.prediction(packet)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(blocked, expected))
        self.assertEqual(runtime.info()["revision"], "16c34e7f070de65d9d2c67a6d74e0007dd8b426a")

    def test_ragged_block_quanta_match_native_executor(self):
        import torch
        from ditserve.models.wan.block import WanBlockExecutor
        from ditserve.models.xdit.wan import XDitWanEngine

        native_worker, native_state, native_executor = plain_executor(torch)
        native_continuation = native_executor.attach(native_state)
        run_quanta(native_executor, native_continuation, [1, 2, 3, 1, 2], until=4)

        worker, state, unused = plain_executor(torch)
        worker.pipe = WanPipelineStub(worker.pipe.transformer)
        engine = XDitWanEngine(worker)
        self.addCleanup(engine.close)
        worker.xdit = engine
        worker.engine_name = "xdit"
        executor = WanBlockExecutor(worker)
        continuation = executor.attach(state)
        run_quanta(executor, continuation, [1, 2, 3, 1, 2], until=4)
        self.assertTrue(torch.equal(state.latents, native_state.latents))

    def test_packet_roundtrip_preserves_layout_and_rejects_other_engines(self):
        import torch
        from ditserve.models.wan.block import (
            XDIT_PACKET_SCHEMA,
            WanBlockExecutor,
            export_block_state,
            import_block_state,
        )
        from ditserve.models.xdit.wan import XDitWanEngine

        native_worker, native_state, native_executor = plain_executor(torch)
        run_quanta(native_executor, native_executor.attach(native_state), [6], until=4)

        source, state, unused = plain_executor(torch)
        source.pipe = WanPipelineStub(source.pipe.transformer)
        source_engine = XDitWanEngine(source)
        self.addCleanup(source_engine.close)
        source.xdit, source.engine_name = source_engine, "xdit"
        source_executor = WanBlockExecutor(source)
        continuation = source_executor.attach(state)
        source_executor.advance(continuation, blocks=2)
        payload, unused = export_block_state(source_executor, continuation)
        self.assertEqual(payload["manifest"]["schema"], XDIT_PACKET_SCHEMA)
        self.assertIn("valid_kv_len", payload["manifest"]["layout"]["xdit"])

        target, unused, unused_executor = plain_executor(torch, empty=True)
        target.pipe = WanPipelineStub(target.pipe.transformer)
        target_engine = XDitWanEngine(target)
        self.addCleanup(target_engine.close)
        target.xdit, target.engine_name = target_engine, "xdit"
        target_executor = WanBlockExecutor(target)
        target_executor.parallel = SimpleNamespace(handle=SimpleNamespace(active=None))
        with self.assertRaisesRegex(ValueError, "SP block import.*solver-step"):
            import_block_state(target_executor, payload)
        self.assertNotIn(restored_id := state.request_id, target._states)
        target_executor.parallel.handle.active = {"request_id": "other"}
        with self.assertRaisesRegex(ValueError, "another step is active"):
            import_block_state(target_executor, payload)
        self.assertNotIn(restored_id, target._states)
        target_executor.parallel = None

        restored, continuation, unused = import_block_state(target_executor, payload)
        run_quanta(target_executor, continuation, [1, 2, 1], until=4)
        self.assertTrue(torch.equal(restored.latents, native_state.latents))

        legacy, unused, legacy_executor = plain_executor(torch, empty=True)
        with self.assertRaisesRegex(ValueError, "engine mismatch"):
            import_block_state(legacy_executor, payload)


if __name__ == "__main__":
    unittest.main()
