"""Tiny native Wan blocks on CPU; no weights, CUDA kernels or GPU evidence."""

import contextlib
import copy
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ditserve.models.wan.block import WanBlockExecutor


def fixture(steps=4, expand=False, interleave=False):
    import torch
    from diffusers import WanTransformer3DModel, UniPCMultistepScheduler
    torch.manual_seed(731)
    model = WanTransformer3DModel(num_attention_heads=1, attention_head_dim=12,
        in_channels=2, out_channels=2, num_layers=30, ffn_dim=24, freq_dim=4,
        text_dim=4, patch_size=(1, 2, 2), rope_max_seq_len=16).eval()
    # expand: Wan2.2 TI2V per-token timesteps.
    worker = SimpleNamespace(torch=torch, pipe=SimpleNamespace(transformer=model), expand_timesteps=expand,
                             _lock=threading.RLock(), _sync=Mock(), _states={})
    worker._dit_attention_context = contextlib.nullcontext
    worker._update_policy = Mock()
    def check(state, residency):
        if worker._states.get(state.spec["request_id"]) is not state or state.residency != residency:
            raise ValueError("invalid state")
    worker._check = check
    worker._payload = lambda state: dict(latents=state.latents, prompt=state.prompt_embeds,
                                         negative=state.negative_prompt_embeds,
                                         scheduler=vars(state.scheduler))
    def worker_step(state):
        native_step(worker, state)
        return dict(kind="step", ms=1., step_index=state.step_index - 1)
    worker.step = worker_step
    def prepare(name, guidance=5., seed=0):
        scheduler = UniPCMultistepScheduler()
        scheduler.set_timesteps(steps, device="cpu")
        state = SimpleNamespace(spec=dict(request_id=name, steps=steps, guidance_scale=guidance, cache=False),
            latents=torch.linspace(-.1, .1, 32).reshape(1, 2, 1, 4, 4) * (1 + .1 * seed),
            prompt_embeds=torch.ones(1, 3, 4), negative_prompt_embeds=torch.zeros(1, 3, 4),
            scheduler=scheduler, step_index=0, finished=False, residency="gpu", cache_enabled=False,
            contexts={}, needs_anchor=False, events=[])
        worker._states[name] = state
        return state
    with patch.object(WanBlockExecutor, "_validate_model"):
        executor = WanBlockExecutor(worker, interleave=interleave)
    return worker, executor, prepare


def native_step(worker, state):
    from ditserve.models.wan.runtime import model_timestep
    torch, model = worker.torch, worker.pipe.transformer
    with torch.inference_mode():
        timestep = state.scheduler.timesteps[state.step_index]
        model_input = model_timestep(torch, state.latents, timestep, worker.expand_timesteps)
        conditional = model(state.latents, model_input, state.prompt_embeds, return_dict=False)[0]
        if state.spec["guidance_scale"] > 1:
            negative = model(state.latents, model_input, state.negative_prompt_embeds, return_dict=False)[0]
            conditional = negative + state.spec["guidance_scale"] * (conditional - negative)
        state.latents = state.scheduler.step(conditional, timestep, state.latents, return_dict=False)[0]
        state.step_index += 1


class BlockExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import diffusers
        except ImportError:
            raise unittest.SkipTest("CPU torch and diffusers required")
        torch.set_num_threads(1)
        cls.torch = torch

    def test_worker_step_quantum_commits_solver_before_allowing_transfer(self):
        from ditserve.models.base import Backend
        from ditserve.models.wan.backend import WanBackend
        worker, executor, prepare = fixture(steps=2)
        state = prepare("a")
        adapter = WanBackend.__new__(WanBackend)
        Backend.__init__(adapter, {})
        adapter.runtime, adapter.executor, adapter.decode_state = worker, executor, None
        adapter.states["a"] = state
        adapter.continuations = {"a": executor.attach(state)}
        conditional = adapter.advance(dict(state_id="a", blocks=30))
        self.assertEqual(conditional["progress"]["step"], 0)
        self.assertFalse(conditional["progress"]["transfer_ready"])
        with self.assertRaisesRegex(ValueError, "solver-step boundary"):
            adapter._boundary(dict(state_id="a"))
        committed = adapter.advance(dict(state_id="a", quantum="step"))
        self.assertEqual(committed["events"][0]["blocks"], 30)
        self.assertEqual(committed["progress"]["step"], 1)
        self.assertTrue(committed["progress"]["transfer_ready"])
        self.assertIs(adapter._boundary(dict(state_id="a")), state)
        done = adapter.advance(dict(state_id="a", quantum="step"))
        self.assertEqual(done["events"][0]["blocks"], 60)
        self.assertTrue(done["progress"]["done"])
        executor.close(adapter.continuations["a"])

    def test_async_decoder_joins_only_its_stream_and_restores_sync_on_error(self):
        import numpy as np
        from ditserve.models.wan.async_decode import AsyncDecoder
        for fails in (False, True):
            with self.subTest(fails=fails):
                entered, release = threading.Event(), threading.Event()
                stream = SimpleNamespace(cuda_stream=17, synchronize=Mock(), wait_event=Mock())
                event = lambda **unused: SimpleNamespace(record=Mock(), synchronize=Mock(), elapsed_time=lambda other: 2.)
                cuda = SimpleNamespace(Stream=lambda **unused: stream, Event=event,
                    current_stream=lambda device: stream, device=lambda device: contextlib.nullcontext(),
                    stream=lambda value: contextlib.nullcontext())
                flags = SimpleNamespace(math_sdp_enabled=lambda: True, flash_sdp_enabled=lambda: False,
                    mem_efficient_sdp_enabled=lambda: False, cudnn_sdp_enabled=lambda: False)
                torch = SimpleNamespace(cuda=cuda, backends=SimpleNamespace(cuda=flags),
                                        inference_mode=contextlib.nullcontext)
                original = Mock()
                worker = SimpleNamespace(attention="math", torch=torch, device="cuda:0", pipe=object(),
                    _check=Mock(), _sync=original, p=SimpleNamespace(_sdpa_math_context=lambda _: contextlib.nullcontext()))
                state = SimpleNamespace(latents=SimpleNamespace(record_stream=Mock()), finished=False,
                                        step_index=2, spec=dict(steps=2))
                def decode(*unused):
                    entered.set()
                    if not release.wait(2):
                        raise RuntimeError("test decoder did not receive release")
                    if fails:
                        raise RuntimeError("decode failed")
                    return np.zeros((1, 1, 2, 2, 3), dtype=np.float32)
                decoder = AsyncDecoder(worker)
                try:
                    with patch("ditserve.models.wan.async_decode.decode_final_latent", side_effect=decode):
                        decoder.submit(state)
                        self.assertTrue(entered.wait(2))
                        worker._sync()
                        original.assert_not_called()
                        stream.synchronize.assert_called_once()
                        self.assertIsNone(decoder.collect())
                        release.set()
                        if fails:
                            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                                decoder.collect(wait=True)
                        else:
                            output, timing = decoder.collect(wait=True)
                            self.assertTrue(timing["asynchronous"])
                    self.assertIs(worker._sync, original)
                finally:
                    release.set()
                    decoder.close()


    def test_interleaving_different_quantum_preserves_every_native_solver_state(self):
        worker, executor, prepare = fixture()
        left, right, reference = prepare("left"), prepare("right"), prepare("reference")
        a, b = executor.attach(left), executor.attach(right)
        for step in range(4):
            native_step(worker, reference)
            while left.step_index <= step or right.step_index <= step:
                if left.step_index <= step:
                    executor.advance(a, 4)
                if right.step_index <= step:
                    executor.advance(b, 7)
            self.assertTrue(self.torch.equal(left.latents, reference.latents))
            self.assertTrue(self.torch.equal(right.latents, reference.latents))
            self.assertEqual(left.scheduler._step_index, reference.scheduler._step_index)
            for actual, expected in zip(left.scheduler.model_outputs, reference.scheduler.model_outputs):
                self.assertTrue(actual is expected if actual is None or expected is None
                                else self.torch.equal(actual, expected))
        self.assertEqual(len(left.events), 4)

    def test_per_token_timesteps_match_the_official_forward_bit_exactly(self):
        from ditserve.models.wan.runtime import model_timestep
        worker, executor, prepare = fixture(expand=True)
        state, reference = prepare("a"), prepare("reference")
        timestep = model_timestep(self.torch, state.latents, state.scheduler.timesteps[0], True)
        self.assertEqual(tuple(timestep.shape), (1, 4))   # 1 frame x (4/2) x (4/2) patch tokens
        continuation = executor.attach(state)
        for step in range(4):
            native_step(worker, reference)
            executor.advance(continuation, 7)
            # Mid-step the packet carries per-token conditioning.
            self.assertEqual(continuation.packet["temb"].ndim, 3)
            self.assertEqual(tuple(continuation.packet["projected"].shape[:3]), (1, 4, 6))
            while state.step_index <= step:
                executor.advance(continuation, 11)
            self.assertTrue(self.torch.equal(state.latents, reference.latents))

    def test_doorbell_stops_a_continuous_quantum_at_a_check_point_bit_exactly(self):
        worker, executor, prepare = fixture()
        state, reference = prepare("a"), prepare("reference")
        continuation = executor.attach(state)
        native_step(worker, reference)
        checks = []
        def bell():
            checks.append(True)
            return len(checks) >= 3
        # Checked every 4 blocks: the third check (after 12 blocks) stops the quantum mid-step.
        row = executor.advance(continuation, 60, doorbell=bell, check_every=4)
        self.assertTrue(row["doorbell_stop"])
        self.assertEqual((row["blocks"], row["step_completed"], state.step_index), (12, False, 0))
        # Without a ring the quantum runs to the step end, and the result equals the native step.
        row = executor.advance(continuation, 60, doorbell=lambda: False, check_every=4)
        self.assertEqual((row["blocks"], row["step_completed"], row["doorbell_stop"]), (48, True, False))
        self.assertTrue(self.torch.equal(state.latents, reference.latents))
        with self.assertRaises(ValueError):
            executor.advance(continuation, 60, doorbell=lambda: False, check_every=0)

    def test_interleaved_branches_match_sequential_execution_bit_exactly(self):
        worker, executor, prepare = fixture(interleave=True)
        state, reference = prepare("a"), prepare("reference")
        continuation = executor.attach(state)
        for step, quanta in enumerate(([60], [7, 9, 44], [1] * 30, [13, 13, 13, 13, 13])):
            native_step(worker, reference)
            for blocks in quanta:
                if state.step_index <= step:
                    row = executor.advance(continuation, blocks)
                    self.assertEqual(row["blocks"] % 2, 0)   # a pair per block index
            self.assertEqual(state.step_index, step + 1)
            self.assertTrue(self.torch.equal(state.latents, reference.latents))
        self.assertIsNone(continuation.packet_uncond)
        # Without CFG there is no second branch to interleave.
        plain, plain_reference = prepare("p", guidance=1.), prepare("pr", guidance=1.)
        native_step(worker, plain_reference)
        executor.advance(executor.attach(plain), 30)
        self.assertTrue(self.torch.equal(plain.latents, plain_reference.latents))

    def test_paired_requests_match_sequential_execution_bit_exactly(self):
        # Two requests advanced in lockstep (block i of each back to back) equal their own
        # sequential runs, through ragged quanta and across step commits.
        worker, executor, prepare = fixture()
        a, b = prepare("a"), prepare("b", seed=5)
        ra, rb = prepare("ra"), prepare("rb", seed=5)
        first, second = executor.attach(a), executor.attach(b)
        # A pair quantum counts block indices; with sequential CFG branches a step is 60 of them.
        for step, quanta in enumerate(([60], [7, 9, 44], [1] * 60, [13, 13, 34])):
            native_step(worker, ra)
            native_step(worker, rb)
            for blocks in quanta:
                if a.step_index <= step:
                    rows = executor.advance_pair(first, second, blocks)
                    self.assertEqual([row["paired"] for row in rows], [True, True])
                    self.assertEqual(rows[0]["blocks"], rows[1]["blocks"])
            self.assertEqual((a.step_index, b.step_index), (step + 1, step + 1))
            self.assertTrue(self.torch.equal(a.latents, ra.latents))
            self.assertTrue(self.torch.equal(b.latents, rb.latents))
        self.assertFalse(self.torch.equal(a.latents, b.latents))
        with self.assertRaisesRegex(ValueError, "two different requests"):
            executor.advance_pair(first, first, 1)
        # A pair forms only at matching positions.
        c, d = executor.attach(prepare("c")), executor.attach(prepare("d", seed=2))
        executor.advance(c, 3)
        with self.assertRaisesRegex(ValueError, "same block"):
            executor.advance_pair(c, d, 1)

    def test_no_cfg_completes_after_one_forward(self):
        worker, executor, prepare = fixture()
        state, reference = prepare("a", guidance=1.), prepare("reference", guidance=1.)
        continuation = executor.attach(state)
        native_step(worker, reference)
        executor.advance(continuation, 30)
        self.assertTrue(self.torch.equal(state.latents, reference.latents))



    def test_cancel_discards_partial_state_without_solver_commit(self):
        worker, executor, prepare = fixture()
        state = prepare("a")
        continuation = executor.attach(state)
        original = state.latents.clone()
        executor.advance(continuation, 33)
        executor.cancel(continuation)
        self.assertEqual(state.step_index, 0)
        self.assertTrue(self.torch.equal(original, state.latents))
        self.assertEqual(executor.extra_state_bytes(continuation), 0)
        with self.assertRaisesRegex(ValueError, "cancelled"):
            executor.advance(continuation)
        executor.close(continuation)
        self.assertFalse(executor._states)


    def test_duplicate_attach_external_step_cache_and_invalid_quantum_rejected(self):
        worker, executor, prepare = fixture()
        state = prepare("a")
        continuation = executor.attach(state)
        with self.assertRaisesRegex(ValueError, "already"):
            executor.attach(state)
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                executor.advance(continuation, count)
        state.spec["cache"] = True
        with self.assertRaisesRegex(ValueError, "cache"):
            executor.advance(continuation)
        state.spec["cache"] = False
        native_step(worker, state)
        with self.assertRaisesRegex(ValueError, "outside"):
            executor.advance(continuation)

    def test_budget_interrupt_fails_closed_and_can_release_packet(self):
        worker, executor, prepare = fixture()
        state = prepare("a")
        continuation = executor.attach(state)
        calls = [0]
        def budget():
            calls[0] += 1
            if calls[0] == 3:
                raise TimeoutError("budget")
        with self.assertRaises(TimeoutError):
            executor.advance(continuation, 8, check_budget=budget)
        self.assertTrue(continuation.failed)
        self.assertEqual(state.step_index, 0)
        self.assertFalse(executor._advancing)
        executor.close(continuation)



if __name__ == "__main__":
    unittest.main()
