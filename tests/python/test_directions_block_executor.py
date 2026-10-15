"""Tiny native Wan blocks on CPU; no weights, CUDA kernels or GPU evidence."""

import contextlib
import copy
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ditserve.directions.block_executor import WanBlockExecutor
from ditserve.directions.direction1 import correctness_injections


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
    def prepare(name, guidance=5.):
        scheduler = UniPCMultistepScheduler()
        scheduler.set_timesteps(steps, device="cpu")
        state = SimpleNamespace(spec=dict(request_id=name, steps=steps, guidance_scale=guidance, cache=False),
            latents=torch.linspace(-.1, .1, 32).reshape(1, 2, 1, 4, 4),
            prompt_embeds=torch.ones(1, 3, 4), negative_prompt_embeds=torch.zeros(1, 3, 4),
            scheduler=scheduler, step_index=0, finished=False, residency="gpu", cache_enabled=False,
            contexts={}, needs_anchor=False, events=[])
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

    def test_cfg_step_commits_once_after_both_forward_branches(self):
        worker, executor, prepare = fixture()
        state = prepare("a")
        initial = state.latents.clone()
        continuation = executor.attach(state)
        executor.advance(continuation, 1)
        self.assertEqual(state.step_index, 0)
        self.assertIsNone(state.scheduler._step_index)
        self.assertTrue(self.torch.equal(initial, state.latents))
        self.assertGreater(executor.extra_state_bytes(continuation), 0)
        executor.advance(continuation, 29)
        self.assertEqual(continuation.branch, "uncond")
        self.assertIsNotNone(continuation.conditional)
        self.assertIsNone(state.scheduler._step_index)
        row = executor.advance(continuation, 30)
        self.assertTrue(row["step_completed"])
        self.assertEqual(state.step_index, 1)
        self.assertEqual(state.scheduler._step_index, 1)
        self.assertEqual(executor.extra_state_bytes(continuation), 0)

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

    def test_full_mixed_quantum_trajectory_covers_both_branches_and_cfg_gap(self):
        worker, executor, prepare = fixture()
        state, reference, other = prepare("a"), prepare("reference"), prepare("other")
        continuation = executor.attach(state)
        foreign = executor.attach(other)
        covered = set()
        for step, policy in correctness_injections(4).items():
            native_step(worker, reference)
            while state.step_index <= step:
                executor.advance(continuation, executor.quantum(continuation, policy))
                branch = "cfg" if policy == "cfg" and continuation.branch == "uncond" else continuation.branch
                marker = (policy, branch)
                eligible = (policy == "cfg" and branch == "cfg") or continuation.packet is not None
                if eligible and marker not in covered:
                    covered.add(marker)
                    # A second request executes while the first keeps its packet.
                    executor.advance(foreign, 1)
            self.assertTrue(self.torch.equal(state.latents, reference.latents))
        expected = {(policy, branch) for policy in ("block-1", "block-4", "block-8")
                    for branch in ("cond", "uncond")} | {("cfg", "cfg")}
        self.assertEqual(covered, expected)

    def test_native_baseline_runs_original_worker_and_refuses_mid_step(self):
        worker, executor, prepare = fixture()
        state, reference = prepare("a"), prepare("reference")
        continuation = executor.attach(state)
        native_step(worker, reference)
        executor.native_step(continuation)
        self.assertTrue(self.torch.equal(state.latents, reference.latents))
        self.assertEqual(continuation.committed_step, 1)
        executor.advance(continuation, 1)
        with self.assertRaisesRegex(ValueError, "boundary"):
            executor.native_step(continuation)

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

    def test_reentrant_overlap_is_rejected_and_foreign_state_untouched(self):
        worker, executor, prepare = fixture()
        a, b = executor.attach(prepare("a")), executor.attach(prepare("b"))
        original = executor.model.blocks[0].forward
        def operation(*args, **kwargs):
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                executor.advance(b)
            return original(*args, **kwargs)
        with patch.object(executor.model.blocks[0], "forward", side_effect=operation):
            executor.advance(a)
        self.assertIsNone(b.packet)
        self.assertEqual(b.base.step_index, 0)

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

    def test_policies_and_constant_storage_do_not_claim_k_times_checkpoints(self):
        worker, executor, prepare = fixture()
        continuation = executor.attach(prepare("a"))
        executor.advance(continuation, 1)
        first = executor.extra_state_bytes(continuation)
        self.assertEqual(executor.quantum(continuation, "cfg"), 29)
        executor.advance(continuation, 8)
        self.assertEqual(first, executor.extra_state_bytes(continuation))
        self.assertEqual(executor.quantum(continuation, "step"), 60)
        self.assertEqual(executor.quantum(continuation, "block-4"), 4)


if __name__ == "__main__":
    unittest.main()
