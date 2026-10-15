"""CPU continuation and external-callable contracts; no GPU performance claim."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ditserve.directions.block_executor import WanBlockExecutor
from ditserve.optimizations.wan import BlockBackend, NativeWanExecutor, WanCompiledExecutor
from test_directions_block_executor import fixture, native_step


class ExternalWanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def executor(self, worker, **kwargs):
        backend = BlockBackend(worker, **kwargs)
        with patch.object(WanBlockExecutor, "_validate_model"):
            return WanCompiledExecutor(worker, backend)

    def test_native_metadata_adapter_keeps_original_advance_identity_and_dispatch(self):
        worker, _, prepare = fixture(steps=1)
        with patch.object(WanBlockExecutor,"_validate_model"), \
             patch("ditserve.research.cfg_empty_context.EmptyCacheContextGuard",
                   side_effect=AssertionError("native must not add a per-quantum guard")):
            executor=NativeWanExecutor(worker)
        self.assertIs(NativeWanExecutor.advance,WanBlockExecutor.advance)
        self.assertIs(executor.advance.__func__,WanBlockExecutor.advance)
        self.assertNotIn("advance",vars(NativeWanExecutor))
        self.assertFalse(hasattr(executor,"_model_guard"))
        state=executor.attach(prepare("native"))
        reference=prepare("reference")
        native_step(worker,reference)
        with patch.object(BlockBackend,"run",side_effect=AssertionError("native cannot call external backend")):
            while state.base.step_index==0:
                executor.advance(state,4)
        self.assertTrue(self.torch.equal(state.base.latents,reference.latents))
        self.assertTrue(executor.backend.report()["original_advance"])
        self.assertFalse(executor.backend.report()["external_callable"])

    def test_external_regions_do_not_replace_forward_or_modules(self):
        worker, _, prepare = fixture(steps=2)
        model = worker.pipe.transformer
        modules = tuple(model.modules())
        forwards = [getattr(m.forward, "__func__", m.forward) for m in modules]
        builds = []
        def compiler(function):
            builds.append(function)
            return function
        ex = self.executor(worker, compiled=True, group_size=4, compiler=compiler)
        base, ref = prepare("a"), prepare("ref")
        state = ex.attach(base)
        for _ in range(2):
            native_step(worker, ref)
            step = base.step_index
            while base.step_index == step:
                ex.advance(state, 4)
            self.assertTrue(self.torch.equal(base.latents, ref.latents))
        self.assertEqual(tuple(model.modules()), modules)
        self.assertEqual([getattr(m.forward, "__func__", m.forward) for m in modules], forwards)
        self.assertTrue(builds)
        self.assertFalse(any(getattr(m, "_orig_mod", None) for m in modules))

    def test_group_tail_keeps_original_cross_cfg_k4_boundary(self):
        worker, _, prepare = fixture()
        ex = self.executor(worker, group_size=4)
        state = ex.attach(prepare("a"))
        for _ in range(7):
            ex.advance(state, 4)
        self.assertEqual((state.branch, state.next_block), ("cond", 28))
        row = ex.advance(state, 4)
        self.assertEqual(row["blocks"], 4)
        self.assertEqual((state.branch, state.next_block), ("uncond", 2))
        self.assertEqual(state.base.step_index, 0)

    def test_group4_continuous_and_k4_have_identical_cfg_aligned_regions(self):
        worker, _, prepare = fixture(steps=1)
        ex = self.executor(worker, group_size=4)
        regions = []
        original = ex.backend.run
        def record(index, width, *args):
            regions.append((index,width))
            return original(index,width,*args)
        with patch.object(ex.backend,"run",side_effect=record):
            continuous=ex.attach(prepare("continuous"))
            ex.advance(continuous,60)
            expected=list(regions)
            regions.clear()
            paused=ex.attach(prepare("paused"))
            while paused.base.step_index==0:
                ex.advance(paused,4)
        self.assertEqual(regions,expected)
        self.assertIn((2,4),regions)  # uncond2 is logical block32.
        self.assertEqual(regions.count((0,1)),1)
        self.assertTrue(self.torch.equal(continuous.base.latents,paused.base.latents))

    def test_host_control_probe_excludes_compute_sync_and_budget_intervals(self):
        worker, _, prepare = fixture(steps=1)
        ex = self.executor(worker)
        state = ex.attach(prepare("host"))
        clock=[0.]
        def charge(amount):
            clock[0]+=amount
        check, guard, run = ex._check, ex._model_guard.validate, ex.backend.run
        def checked(value):
            check(value)
            charge(2.)
        def guarded():
            guard()
            charge(1.)
        def computed(*args):
            charge(10.)
            return run(*args)
        with patch("ditserve.optimizations.wan.time.perf_counter",side_effect=lambda: clock[0]), \
             patch.object(ex,"_check",side_effect=checked), \
             patch.object(ex._model_guard,"validate",side_effect=guarded), \
             patch.object(ex.backend,"run",side_effect=computed), \
             patch.object(worker,"_sync",side_effect=lambda: charge(20.)):
            row=ex.advance(state,4,observe_host=True,check_budget=lambda: charge(30.))
        self.assertEqual(row["pure_host_control_ms"],3000.)
        self.assertGreater(row["quantum_total_wall_ms"],row["pure_host_control_ms"]+200000.)

    def test_late_hook_or_forward_change_still_rejected(self):
        worker, _, prepare = fixture()
        ex = self.executor(worker)
        state = ex.attach(prepare("a"))
        with patch.object(ex.model.blocks[0], "forward", lambda *x: x[0]):
            with self.assertRaisesRegex(ValueError, "forward"):
                ex.advance(state, 4)
        self.assertEqual(state.next_block, 0)

    def test_compiler_failure_discards_packet_without_solver_commit(self):
        worker, _, prepare = fixture()
        def compiler(function):
            def failed(*args):
                raise RuntimeError("compile failure")
            return failed
        ex = self.executor(worker, compiled=True, compiler=compiler)
        state = ex.attach(prepare("a"))
        with self.assertRaisesRegex(RuntimeError, "compile failure"):
            ex.advance(state, 4)
        self.assertTrue(state.failed)
        self.assertIsNone(state.packet)
        self.assertEqual(state.base.step_index, 0)
        ex.close(state)

    def test_backend_fallback_is_not_silently_qualified(self):
        worker, _, _ = fixture()
        backend = BlockBackend(worker, compiled=True)
        backend._callables[(0, 1)] = lambda *args: args[0]
        with self.assertRaisesRegex(RuntimeError, "without a compiled graph"):
            backend.run(0, 1, self.torch.zeros(1), None, None, None)

    def test_inductor_disables_cudagraphs_and_counts_real_graph_invocations(self):
        worker, _, _ = fixture()
        backend = BlockBackend(worker, compiled=True)
        calls = []
        def build(graph, inputs, **kwargs):
            calls.append(kwargs)
            return lambda x: x + 1
        with patch("torch._dynamo.backends.registry.lookup_backend", return_value=build):
            compiled = backend._inductor(object(), [])
        self.assertEqual(calls, [{"config_patches": {"triton.cudagraphs": False}}])
        self.assertEqual(compiled(2), 3)
        self.assertEqual(backend.graph_invocations, 1)
        self.assertEqual(backend.compilations, 1)

    def test_real_cpu_inductor_counter_survives_dynamo_backend_unwrapping(self):
        # Real Inductor returns callable metadata that a mocked graph.forward
        # does not have. Dynamo must execute our counter on cold and warm calls.
        torch = self.torch
        class TinyBlock(torch.nn.Module):
            def forward(self, hidden, encoder, projected, rotary):
                return hidden.sin() + hidden * hidden + encoder
        model = SimpleNamespace(blocks=[TinyBlock()])
        worker = SimpleNamespace(torch=torch, pipe=SimpleNamespace(transformer=model))
        backend = BlockBackend(worker, compiled=True)
        torch._dynamo.reset()
        try:
            hidden = torch.linspace(-1., 1., 17)
            encoder = torch.ones_like(hidden)
            expected = model.blocks[0](hidden, encoder, None, None)
            first = backend.run(0, 1, hidden, encoder, None, None)
            self.assertTrue(torch.allclose(first, expected, rtol=1e-5, atol=1e-6))
            self.assertEqual(backend.graph_invocations, 1)
            backend.freeze_compilation()
            second = backend.run(0, 1, hidden, encoder, None, None)
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(backend.graph_invocations, 2)
            self.assertEqual(backend.compilations, 1)
            self.assertEqual(backend.calls, 2)
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            torch._dynamo.reset()

    def test_real_dynamo_captures_all_thirty_external_regions_without_module_wrapping(self):
        # FX execution on CPU exercises real Dynamo guards/capture without
        # compiling CPU kernels, using GPU kernels, or claiming Inductor speed.
        worker, _, prepare = fixture(steps=1)
        captured = []
        def graph_backend(graph, inputs):
            captured.append(graph)
            return graph.forward
        self.torch._dynamo.reset()
        try:
            ex = self.executor(worker, compiled=True,
                compiler=lambda region: self.torch.compile(region, backend=graph_backend,
                    fullgraph=True, dynamic=False))
            base, reference = prepare("compiled"), prepare("reference")
            state = ex.attach(base)
            native_step(worker, reference)
            while base.step_index == 0:
                ex.advance(state, 4)
            self.assertTrue(self.torch.allclose(base.latents, reference.latents, rtol=1e-5, atol=1e-6))
            self.assertTrue(captured)
            self.assertEqual(len(ex.backend._callables), 30)
            self.assertFalse(any(hasattr(module, "_orig_mod") for module in ex.model.modules()))
        finally:
            self.torch._dynamo.reset()

    def test_failed_compile_attempt_retains_its_cost_and_error(self):
        worker, _, _ = fixture()
        backend = BlockBackend(worker, compiled=True)
        with patch("torch._dynamo.backends.registry.lookup_backend",
                   return_value=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("backend fails"))):
            with self.assertRaisesRegex(RuntimeError, "backend fails"):
                backend._inductor(object(), [])
        self.assertEqual(backend.compilations, 0)
        self.assertEqual(backend.compile_attempts[0]["status"], "failed")
        self.assertGreaterEqual(backend.compile_backend_ms, 0)
        self.assertIn("backend fails", backend.report()["compile_attempts"][0]["error"])


if __name__ == "__main__":
    unittest.main()
