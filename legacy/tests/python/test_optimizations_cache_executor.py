"""Tiny real Wan/cache-dit CPU trajectories; no weights or GPU inference."""

import copy
import unittest
from unittest.mock import patch

from ditserve.optimizations.cache_executor import PairedCacheExecutor
from ditserve.research.dual_runtime import correctness_state_digest
from test_research_cfg_split_prefix import fixture, split_adapter
from test_research_cfg_adapter import policy


def prepared(gates=None, boundary=8):
    worker, base, _ = fixture(gates, boundary=boundary)
    adapter = split_adapter(worker)
    executor = PairedCacheExecutor(worker, split=adapter._split_prefix)
    return worker, base, executor, executor.attach(base)


def clone_request(worker, base, name):
    value = copy.deepcopy(base, {id(worker._owner): worker._owner})
    value.request_id = value.spec["request_id"] = value.immutable_spec["request_id"] = name
    worker._states[name] = value
    return value


class PairedContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def test_all_paired_gate_paths_match_original_split_prefix(self):
        for gates in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(gates=gates):
                rw, ref, _ = fixture(gates)
                reference = split_adapter(rw)
                worker, base, ex, state = prepared(gates)
                reference.step(ref, policy=dict(policy(), implementation="split_prefix"))
                calls = 0
                while base.step_index == 8:
                    row = ex.advance(state, 4)
                    self.assertLessEqual(row["blocks"], 4)
                    calls += row["blocks"]
                    self.assertEqual(worker.manager._cached_context_manager, {})
                self.assertEqual(calls, 16 if all(gates) else 60)
                self.assertEqual(correctness_state_digest(worker, base), correctness_state_digest(rw, ref))
                self.assertEqual(len(base.events[-1]["gates"]), 2)
                self.assertEqual(ex.extra_state_bytes(state), 0)

    def test_full_32_step_natural_path_matches_uninterrupted_backend(self):
        worker, base, ex, state = prepared(boundary=0)
        rw, ref, rx, rs = prepared(boundary=0)
        for step in range(32):
            rx.advance(rs, 60)
            while base.step_index == step:
                ex.advance(state, 4)
            self.assertEqual(correctness_state_digest(worker, base), correctness_state_digest(rw, ref))
        self.assertEqual(len([r for r in base.events if r.get("kind") == "step"]), 32)

    def test_middle_tail_crosses_cfg_without_committing_solver_early(self):
        worker, base, ex, state = prepared((False, False))
        for _ in range(9):  # 16 prefix blocks + first20 conditional middle blocks
            ex.advance(state, 4)
        self.assertEqual((state.phase, state.branch, state.next_block), ("suffix", "cond", 28))
        row = ex.advance(state, 4)
        self.assertEqual(row["blocks"], 4)
        self.assertEqual((state.phase, state.branch, state.next_block), ("suffix", "uncond", 10))
        self.assertEqual(base.step_index, 8)
        self.assertEqual(len(state.gates), 2)

    def test_budget_failure_after_conditional_commit_never_rewinds_global_rng(self):
        worker, base, ex, state = prepared((False, False))
        for _ in range(10):
            ex.advance(state, 4)
        # A separate request may have consumed randomness while A was paused.
        self.torch.rand(17)
        before = self.torch.get_rng_state().clone()
        def stop():
            raise TimeoutError("budget")
        with self.assertRaises(TimeoutError):
            ex.advance(state, 4, check_budget=stop)
        self.assertTrue(self.torch.equal(before, self.torch.get_rng_state()))
        self.assertEqual(base.step_index, 8)
        # Failure before admission leaves an untouched resumable quantum; it is
        # then explicitly cancelled/closed by the phase's cleanup.
        ex.cancel(state)
        self.assertEqual(ex.extra_state_bytes(state), 0)
        ex.close(state)

    def test_failure_inside_quantum_is_discarded_not_rolled_back(self):
        worker, base, ex, state = prepared((False, False))
        for _ in range(10):
            ex.advance(state, 4)
        self.torch.rand(11)
        expected_rng = self.torch.get_rng_state().clone()
        with patch.object(ex.backend, "run", side_effect=RuntimeError("region failed")):
            with self.assertRaisesRegex(RuntimeError, "region failed"):
                ex.advance(state, 4)
        self.assertTrue(state.failed)
        self.assertEqual(ex.extra_state_bytes(state), 0)
        self.assertTrue(self.torch.equal(expected_rng, self.torch.get_rng_state()))
        self.assertEqual(worker.manager._cached_context_manager, {})
        with self.assertRaisesRegex(ValueError, "failed"):
            ex.advance(state)
        ex.close(state)

    def test_foreign_owner_external_step_and_policy_change_rejected(self):
        worker, base, ex, state = prepared()
        with self.assertRaisesRegex(ValueError, "already"):
            ex.attach(base)
        other = PairedCacheExecutor(worker, split=ex.split)
        with self.assertRaisesRegex(ValueError, "foreign"):
            other.advance(state)
        base.spec["threshold"] = .04
        with self.assertRaisesRegex(ValueError, "paired_008"):
            ex.advance(state)
        base.spec["threshold"] = .08
        base.step_index += 1
        with self.assertRaisesRegex(ValueError, "outside"):
            ex.advance(state)

    def test_two_requests_interleave_private_cache_and_commit_each_gate_once(self):
        worker, first, ex, a = prepared(boundary=0)
        second = clone_request(worker, first, "second")
        b = ex.attach(second)
        rw, ref, rx, rs = prepared(boundary=0)
        for step in range(12):  # Warmup and real accepted-reuse decisions.
            rx.advance(rs, 60)
            while first.step_index <= step or second.step_index <= step:
                if first.step_index <= step:
                    ex.advance(a, 4)
                if second.step_index <= step:
                    ex.advance(b, 7)
                self.assertEqual(worker.manager._cached_context_manager, {})
            expected = correctness_state_digest(rw, ref)
            self.assertEqual(correctness_state_digest(worker, first), expected)
            self.assertEqual(correctness_state_digest(worker, second), expected)
            self.assertEqual(len(first.events[-1]["gates"]), 2)
            self.assertEqual(len(second.events[-1]["gates"]), 2)
        self.assertIsNot(first.contexts["blocks"], second.contexts["blocks"])
        for key in first.contexts["blocks"].buffers:
            self.assertNotEqual(first.contexts["blocks"].buffers[key].data_ptr(),
                                second.contexts["blocks"].buffers[key].data_ptr())

    def test_failed_request_does_not_restore_or_corrupt_interleaved_peer(self):
        worker, first, ex, a = prepared((False, False))
        second = clone_request(worker, first, "second")
        b = ex.attach(second)
        # A has committed the conditional gate before yielding; B can progress.
        for _ in range(10):
            ex.advance(a, 4)
        ex.advance(b, 60)
        peer_digest = correctness_state_digest(worker, second)
        self.torch.rand(9)
        rng = self.torch.get_rng_state().clone()
        with patch.object(ex.backend, "run", side_effect=RuntimeError("A fails")):
            with self.assertRaisesRegex(RuntimeError, "A fails"):
                ex.advance(a, 4)
        self.assertEqual(correctness_state_digest(worker, second), peer_digest)
        self.assertTrue(self.torch.equal(rng, self.torch.get_rng_state()))
        self.assertFalse(b.failed)
        ex.close(a)
        ex.advance(b, 60)
        self.assertEqual(second.step_index, 10)

    def _assert_interleaved_peer_survives(self, *, oom):
        worker, first, ex, a = prepared((False, False))
        second = clone_request(worker, first, "second")
        b = ex.attach(second)
        rw, ref, rx, rs = prepared((False, False))

        def frozen(value):
            # These are tiny CPU tensors. Include identity/storage as well as
            # values: replacing B's packet with an equal clone is not isolation.
            if self.torch.is_tensor(value):
                self.assertEqual(value.device.type, "cpu")
                return (id(value), value.data_ptr(), str(value.dtype), tuple(value.shape),
                        tuple(value.stride()), value.detach().reshape(-1).tolist())
            if isinstance(value, dict):
                return {key: frozen(child) for key, child in value.items()}
            if isinstance(value, (tuple, list)):
                return [frozen(child) for child in value]
            return value

        def continuation_snapshot():
            return frozen({key: getattr(b, key) for key in (
                "committed_step", "phase", "branch", "next_block", "packets", "predictions",
                "probe", "reuse", "gates", "step_quanta", "events", "cancelled", "failed", "closed")})

        def assert_manager_detached():
            self.assertIsNone(worker.manager._current_context)
            self.assertEqual(worker.manager._cached_context_manager, {})
            self.assertFalse(worker.manager._current_step_refreshed)
            self.assertFalse(ex._advancing)
            self.assertIs(worker.pipe.transformer.forward, worker._cached_forward)

        try:
            # Both requests really execute before A is terminated. B remains
            # mid-step with its conditional prediction and uncond packet live.
            for _ in range(10):
                ex.advance(a, 4)
                ex.advance(b, 4)
                assert_manager_detached()
            self.assertEqual((a.phase, a.branch, a.next_block), ("suffix", "uncond", 10))
            self.assertEqual((b.phase, b.branch, b.next_block), ("suffix", "uncond", 10))
            self.assertEqual((first.step_index, second.step_index), (8, 8))
            self.assertIn("uncond", b.packets)
            self.assertIn("cond", b.predictions)
            self.assertGreater(ex.extra_state_bytes(b), 0)
            peer_digest = correctness_state_digest(worker, second)
            peer_continuation = continuation_snapshot()
            peer_buffers = frozen(second.contexts["blocks"].buffers)
            peer_context = second.contexts["blocks"]
            peer_generator = second.generator
            peer_rng = second.generator.get_state().clone()
            self.assertIsNot(first.generator, peer_generator)
            self.assertIsNot(first.contexts["blocks"], peer_context)
            self.torch.rand(9)  # Work between A's and B's quanta must not be rewound.
            expected_global_rng = [self.torch.get_rng_state().clone()]
            if oom:
                original_run = ex.backend.run
                own_rng = first.generator.get_state().clone()
                calls = []
                def allocate_then_oom(*args, **kwargs):
                    self.assertIs(worker.manager._current_context, first.contexts["blocks"])
                    original_run(*args, **kwargs)  # Execute a real tiny block before the failure.
                    self.torch.rand(5, generator=first.generator)
                    self.torch.rand(7)
                    expected_global_rng[0] = self.torch.get_rng_state().clone()
                    calls.append(True)
                    # CPU fault injection exercises the actual OOM exception
                    # path; it does not claim physical CUDA allocation coverage.
                    raise self.torch.OutOfMemoryError("injected A allocation OOM")
                with patch.object(ex.backend, "run", side_effect=allocate_then_oom):
                    with self.assertRaisesRegex(self.torch.OutOfMemoryError, "A allocation OOM"):
                        ex.advance(a, 4)
                self.assertEqual(calls, [True])
                self.assertTrue(a.failed)
                self.assertFalse(self.torch.equal(own_rng, first.generator.get_state()))
            else:
                ex.cancel(a)
                self.assertTrue(a.cancelled)
                self.assertFalse(a.failed)
            self.assertEqual(first.step_index, 8)  # No partial solver commit on A.
            self.assertEqual(ex.extra_state_bytes(a), 0)
            with self.assertRaisesRegex(ValueError, "failed|cancelled"):
                ex.advance(a, 4)
            # Check both before and after A's complete cleanup. In particular,
            # release() must not clear the peer's request-owned cache buffers.
            for clean_a in (False, True):
                if clean_a:
                    ex.close(a)
                    worker.release(first)
                self.assertEqual(correctness_state_digest(worker, second), peer_digest)
                self.assertEqual(continuation_snapshot(), peer_continuation)
                self.assertEqual(frozen(second.contexts["blocks"].buffers), peer_buffers)
                self.assertIs(second.contexts["blocks"], peer_context)
                self.assertIs(second.generator, peer_generator)
                self.assertTrue(self.torch.equal(peer_rng, second.generator.get_state()))
                self.assertTrue(self.torch.equal(expected_global_rng[0], self.torch.get_rng_state()))
                self.assertIs(ex._states[id(second)], b)
                self.assertIs(worker._states[second.request_id], second)
                assert_manager_detached()
            # B is still executable, not merely an unchanged abandoned object.
            # Complete its current step and one following step against a separate
            # uninterrupted request, including sampler/cache/RNG equality.
            for step in (8, 9):
                rx.advance(rs, 60)
                while second.step_index == step:
                    ex.advance(b, 4)
                    assert_manager_detached()
                self.assertEqual(correctness_state_digest(worker, second), correctness_state_digest(rw, ref))
        finally:
            for owner, executor, base, continuation in ((worker, ex, first, a), (worker, ex, second, b),
                                                        (rw, rx, ref, rs)):
                if not continuation.closed:
                    executor.close(continuation)
                if base.residency != "released":
                    owner.release(base)

    def test_cancelled_interleaved_request_preserves_peer_continuation_cache_manager_and_rng(self):
        self._assert_interleaved_peer_survives(oom=False)

    def test_oom_in_interleaved_request_preserves_peer_continuation_cache_manager_and_rng(self):
        self._assert_interleaved_peer_survives(oom=True)


if __name__ == "__main__":
    unittest.main()
