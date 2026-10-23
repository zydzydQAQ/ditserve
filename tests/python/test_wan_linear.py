"""CPU contracts for the opt-in Wan SP2 repair, not GPU qualification."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class WanLinearRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU torch required")
        torch.set_num_threads(1)
        cls.torch = torch

    def fixture(self, rank=0, degree=2, policy="full_rows_except_ffn1"):
        from ditserve.models.wan.parallel import LayoutHandle
        from ditserve.models.wan.sp import PinnedWanUlyssesBackend
        from ditserve.models.wan.linear import FullRowsLinearRepair
        torch, nn = self.torch, self.torch.nn
        projection = nn.Module()
        projection.proj = nn.Linear(3, 3)
        block = nn.Module()
        block.ffn = nn.Module()
        block.ffn.net = nn.Sequential(projection, nn.Identity(), nn.Linear(3, 2))
        block.attn1, block.attn2 = nn.Module(), nn.Module()
        block.attn1.to_q, block.attn2.to_k = nn.Linear(3, 2), nn.Linear(3, 2)
        model = nn.Module()
        model.blocks, model.proj_out = nn.ModuleList([block]), nn.Linear(3, 2)
        model.config = SimpleNamespace(patch_size=(1, 2, 2))
        handle = LayoutHandle(rank, (0, 1) if degree == 2 else (rank,))
        backend = PinnedWanUlyssesBackend(handle, SimpleNamespace(rank=rank),
                                        linear_policy=policy)
        backend.worker = SimpleNamespace(torch=torch, pipe=SimpleNamespace(transformer=model), _sync=lambda: None)
        backend.gate.consensus = lambda payload: payload
        backend._linear_repair = FullRowsLinearRepair(backend).attach()
        state = SimpleNamespace(request_id="a", step_index=0, spec=dict(guidance_scale=5.),
                                latents=torch.zeros(1, 3, 1, 4, 4))
        self.addCleanup(backend.close)
        return backend, model, state

    def test_rank_placement_projection_crop_ffn1_and_text_bypass(self):
        torch = self.torch
        for rank in (0, 1):
            backend, model, state = self.fixture(rank)
            seen, cropped = [], []
            projection = model.proj_out
            original = projection.forward
            projection.forward = lambda input: (seen.append(input.clone()), original(input))[1]
            # Registered after repair here; the repair itself also uses prepend
            # so existing production gather hooks see the same cropped shape.
            projection.register_forward_hook(lambda m, a, out: cropped.append(tuple(out.shape)))
            backend.begin_step(state)
            value = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
            expected = torch.nn.functional.linear(value, projection.weight, projection.bias)
            output = projection(input=value)
            self.assertTrue(torch.equal(output, expected))
            self.assertEqual(cropped, [(1, 2, 2)])
            self.assertEqual(tuple(seen[0].shape), (1, 4, 3))
            self.assertTrue(torch.equal(seen[0][:, 2 * rank:2 * rank + 2], value))
            self.assertEqual(seen[0][:, 2 * (1-rank):2 * (1-rank) + 2].count_nonzero().item(), 0)
            self.assertEqual(output.storage_offset(), 0)
            self.assertTrue(output.is_contiguous())
            first = model.blocks[0].ffn.net[0].proj
            self.assertFalse(first._forward_hooks or first._forward_pre_hooks)
            text = model.blocks[0].attn2.to_k
            text_value = torch.ones(1, 3, 3)
            text_seen = []
            text.register_forward_pre_hook(lambda m, a: text_seen.append(a[0]))
            text(text_value)
            self.assertIs(text_seen[0], text_value)
            audit = backend.end_step()
            self.assertEqual((audit["applied_calls"], audit["skipped_calls"]), (1, 1))

    @patch("ditserve.models.wan.sp.RequestGroupGate.consensus", lambda self, payload: payload)
    def test_sp1_expand_resume_both_cfg_branches_and_shrink(self):
        torch = self.torch
        backend, model, state = self.fixture(degree=1)
        names = ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_fp16_accumulation")
        flags = {name: getattr(torch.backends.cuda.matmul, name) for name in names}
        original = copy.deepcopy(model.proj_out.state_dict())
        for degree in (1, 2, 1):
            if backend.handle.current.degree != degree:
                backend.handle.commit((0, 1) if degree == 2 else (0,),
                                      expected_epoch=backend.handle.current.epoch)
            # Mid-branch resume has no patch_embedding call. The request shape
            # must provide correct geometry for every quantum independently.
            for branch in ("cond", "uncond"):
                continuation = SimpleNamespace(base=state, branch=branch, next_block=7,
                                               packet={}, conditional=None)
                backend.begin_quantum(continuation, backend.handle, 3, 1)
                value = torch.arange(12 // degree, dtype=torch.float32).reshape(1, 4 // degree, 3)
                actual = model.proj_out(value)
                expected = torch.nn.functional.linear(value, original["weight"], original["bias"])
                self.assertTrue(torch.equal(actual, expected))
                audit = backend.end_quantum(continuation, 3)
                self.assertEqual(audit["degree"], degree)
                self.assertEqual(audit["applied_calls"], int(degree == 2))
                self.assertEqual(audit["unchanged_sp1_calls"], int(degree == 1))
                self.assertIsNone(backend._linear_repair.geometry)
        self.assertTrue(all(torch.equal(value, original[name])
                            for name, value in model.proj_out.state_dict().items()))
        self.assertEqual(flags, {name: getattr(torch.backends.cuda.matmul, name) for name in names})

    def test_missing_geometry_changed_layout_and_failure_cleanup(self):
        torch = self.torch
        backend, model, state = self.fixture()
        with self.assertRaisesRegex(Exception, "outside"):
            model.proj_out(torch.ones(1, 2, 3))
        backend.begin_step(state)
        from ditserve.models.wan.parallel import FixedLayout
        backend.handle.current = FixedLayout((0, 1), 1)
        backend._bound_epoch = 1
        with self.assertRaisesRegex(Exception, "geometry"):
            model.proj_out(torch.ones(1, 2, 3))
        backend.abort_linear_repair()
        backend.handle.end_step()
        backend.begin_step(state)
        model.proj_out.forward = lambda value: (_ for _ in ()).throw(RuntimeError("GEMM failed"))
        with self.assertRaisesRegex(RuntimeError, "GEMM failed"):
            model.proj_out(torch.ones(1, 2, 3))
        self.assertTrue(all(not values for values in backend._linear_repair.pending.values()))
        backend.abort_linear_repair()
        repair = backend._linear_repair
        backend.close()
        self.assertFalse(repair.hooks)
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in model.modules()))

    def test_scratch_reuses_across_modules_and_clears_at_quantum_boundary(self):
        torch = self.torch
        for rank in (0, 1):
            backend, model, state = self.fixture(rank, policy="full_rows_except_ffn1_scratch")
            backend.begin_step(state)
            repair = backend._linear_repair
            value = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
            first = model.proj_out(value)
            pointer = repair.scratch.data_ptr()
            second = model.blocks[0].attn1.to_q(value * 3)
            self.assertEqual(repair.scratch.data_ptr(), pointer)
            self.assertTrue(torch.equal(first, torch.nn.functional.linear(value, model.proj_out.weight, model.proj_out.bias)))
            self.assertTrue(torch.equal(second, torch.nn.functional.linear(value * 3, model.blocks[0].attn1.to_q.weight, model.blocks[0].attn1.to_q.bias)))
            self.assertNotEqual(first.data_ptr(), pointer)
            audit = backend.end_step()
            self.assertEqual(audit["scratch_allocations"], 1)
            self.assertEqual(audit["scratch_hits"], 1)
            self.assertEqual(audit["zero_bytes"], value.numel() * value.element_size() * 2)
            self.assertIsNone(repair.scratch)
            state.request_id, state.step_index = "another-request", 7
            backend.begin_step(state)
            model.proj_out(value)
            self.assertEqual(backend.end_step()["scratch_allocations"], 1)

    def test_scratch_oversized_fallback_and_abort_release(self):
        backend, model, state = self.fixture(policy="full_rows_except_ffn1_scratch")
        value = self.torch.ones(1, 2, 3)
        backend.begin_step(state)
        with patch("ditserve.models.wan.linear.SCRATCH_LIMIT_BYTES", 1):
            actual = model.proj_out(value)
        self.assertTrue(self.torch.equal(actual, self.torch.nn.functional.linear(value, model.proj_out.weight, model.proj_out.bias)))
        self.assertIsNone(backend._linear_repair.scratch)
        audit = backend.end_step()
        self.assertEqual(audit["scratch_fallbacks"], 1)
        self.assertEqual(audit["scratch_allocations"], 0)
        backend.begin_step(state)
        model.proj_out(value)
        repair = backend._linear_repair
        self.assertIsNotNone(repair.scratch)
        repair.scratch_busy = True
        with self.assertRaisesRegex(Exception, "overlap"):
            model.proj_out(value)
        backend.abort_linear_repair()
        self.assertIsNone(repair.scratch)
        self.assertIsNone(repair.scratch_stream)
        self.assertFalse(repair.scratch_busy)

    def test_default_native_policy_has_no_repair_or_process_flag_mutation(self):
        from ditserve.models.wan.parallel import LayoutHandle
        from ditserve.models.wan.sp import PinnedWanUlyssesBackend
        from ditserve.models.wan.linear import validate_linear_policy
        backend = PinnedWanUlyssesBackend(LayoutHandle(0), SimpleNamespace(rank=0))
        self.assertEqual(backend.linear_policy, "native")
        self.assertIsNone(backend._linear_repair)
        for invalid in (None, True, "full_rows_linear", "global_tile_1024"):
            with self.assertRaisesRegex(Exception, "policy"):
                validate_linear_policy(invalid)

    def test_actual_block_executor_sp1_pause_cfg_resume_is_exact(self):
        from ditserve.models.wan.block import WanBlockExecutor, WanParallelBlockExecutor
        from ditserve.models.wan.linear import FullRowsLinearRepair
        from ditserve.models.wan.parallel import LayoutHandle
        from ditserve.models.wan.sp import PinnedWanUlyssesBackend
        try:
            from test_model_blocks import fixture, native_step
            import diffusers
        except ImportError:
            raise unittest.SkipTest("CPU diffusers and test fixture required")
        for policy in ("native", "full_rows_except_ffn1", "full_rows_except_ffn1_scratch"):
            worker, _, prepare = fixture(steps=2)
            state = prepare("actual")
            reference_worker, _, reference_prepare = fixture(steps=2)
            reference = reference_prepare("reference")
            state.request_id = "actual"
            kwargs = {} if policy == "native" else dict(linear_policy=policy)
            backend = PinnedWanUlyssesBackend(LayoutHandle(0), SimpleNamespace(rank=0), **kwargs)
            backend.worker = worker
            if policy != "native":
                backend._linear_repair = FullRowsLinearRepair(backend).attach()
            self.addCleanup(backend.close)
            with patch.object(WanBlockExecutor, "_validate_model"):
                executor = WanParallelBlockExecutor(worker, backend)
            continuation = executor.attach(state)
            for step in range(2):
                native_step(reference_worker, reference)
                for sequence, blocks in enumerate((23, 14, 23)):
                    result = executor.advance(continuation, blocks, layout=backend.handle,
                                              quantum_seq=step * 3 + sequence)
                    self.assertEqual(result["linear_repair"]["applied_calls"], 0)
                    if policy != "native":
                        self.assertGreater(result["linear_repair"]["unchanged_sp1_calls"], 0)
                self.assertTrue(self.torch.equal(state.latents, reference.latents))
                self.assertIsNone(continuation.packet)
                self.assertIsNone(continuation.conditional)
            executor.close(continuation)


if __name__ == "__main__":
    unittest.main()
