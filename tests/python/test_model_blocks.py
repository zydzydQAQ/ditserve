"""Tiny native Wan blocks on CPU; no weights, CUDA kernels or GPU evidence."""

import contextlib
import copy
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ditserve.models.wan.block import WanBlockExecutor


def fixture(steps=4):
    import torch
    from diffusers import WanTransformer3DModel, UniPCMultistepScheduler
    torch.manual_seed(731)
    model = WanTransformer3DModel(num_attention_heads=1, attention_head_dim=12,
        in_channels=2, out_channels=2, num_layers=30, ffn_dim=24, freq_dim=4,
        text_dim=4, patch_size=(1, 2, 2), rope_max_seq_len=16).eval()
    worker = SimpleNamespace(torch=torch, pipe=SimpleNamespace(transformer=model),
                             _lock=threading.RLock(), _sync=Mock(), _states={})
    worker._dit_attention_context = contextlib.nullcontext
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
    def prepare(name, guidance=5.):
        scheduler = UniPCMultistepScheduler()
        scheduler.set_timesteps(steps, device="cpu")
        state = SimpleNamespace(spec=dict(request_id=name, steps=steps, guidance_scale=guidance),
            latents=torch.linspace(-.1, .1, 32).reshape(1, 2, 1, 4, 4),
            prompt_embeds=torch.ones(1, 3, 4), negative_prompt_embeds=torch.zeros(1, 3, 4),
            scheduler=scheduler, step_index=0, finished=False, residency="gpu", events=[])
        worker._states[name] = state
        return state
    with patch.object(WanBlockExecutor, "_validate_model"):
        executor = WanBlockExecutor(worker)
    return worker, executor, prepare


def native_step(worker, state):
    torch, model = worker.torch, worker.pipe.transformer
    with torch.inference_mode():
        timestep = state.scheduler.timesteps[state.step_index]
        conditional = model(state.latents, timestep.expand(1), state.prompt_embeds, return_dict=False)[0]
        if state.spec["guidance_scale"] > 1:
            negative = model(state.latents, timestep.expand(1), state.negative_prompt_embeds, return_dict=False)[0]
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
        adapter.runtime, adapter.executor = worker, executor
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

    def test_no_cfg_completes_after_one_forward(self):
        worker, executor, prepare = fixture()
        state, reference = prepare("a", guidance=1.), prepare("reference", guidance=1.)
        continuation = executor.attach(state)
        native_step(worker, reference)
        executor.advance(continuation, 30)
        self.assertTrue(self.torch.equal(state.latents, reference.latents))


    def test_duplicate_attach_external_step_and_invalid_quantum_rejected(self):
        worker, executor, prepare = fixture()
        state = prepare("a")
        continuation = executor.attach(state)
        with self.assertRaisesRegex(ValueError, "already"):
            executor.attach(state)
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                executor.advance(continuation, count)
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
