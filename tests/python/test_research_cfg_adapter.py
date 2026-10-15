"""CPU tensor execution against the real pinned cache gate and UniPC sampler."""

import contextlib
import importlib.util
import threading
import types
import unittest

from ditserve.research.cfg_adapter import CfgResearchAdapter, _clone_graph


def policy(identity="paired_008"):
    return {"id": identity, "threshold": 0.0 if identity == "off" else
            0.04 if identity == "independent_004" else 0.08,
            "max_continuous": 1, "cfg_mode": "paired" if identity == "paired_008" else "independent",
            "implementation": "oracle" if identity == "paired_008" else "native"}


def fixture(gates=(True, True), *, identity="paired_008", boundary=8, random=False):
    import torch
    import cache_dit
    from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
    from diffusers import UniPCMultistepScheduler
    from ditserve.gpu.worker import WanStepWorker, RequestState, normalize_spec

    worker = WanStepWorker.__new__(WanStepWorker)
    worker.torch, worker.cache_dit, worker.device = torch, cache_dit, "cpu"
    worker._owner, worker._states = object(), {}
    worker._lock, worker.manager = threading.RLock(), CachedContextManager()
    worker._context_names = ("blocks",)
    worker._sync = lambda: None
    worker._dit_attention_context = contextlib.nullcontext
    worker.p = types.SimpleNamespace(_sdpa_math_context=lambda _: contextlib.nullcontext())
    selected = policy(identity)
    spec = normalize_spec(dict(request_id="test", prompt="a kettle", seed=101,
        height=32, width=32, frames=5, steps=32, guidance_scale=5.0,
        cache=identity != "off", threshold=selected["threshold"], max_continuous=1))
    scheduler = UniPCMultistepScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(32, device="cpu")
    state = RequestState(spec["request_id"], spec, scheduler=scheduler,
        cache_enabled=spec["cache"], latents=torch.ones(1, 2, 2, 2),
        prompt_embeds=torch.ones(1), negative_prompt_embeds=torch.zeros(1),
        generator=torch.Generator().manual_seed(101), owner=worker._owner,
        contexts=worker._contexts(spec) if spec["cache"] else {})
    state.immutable_spec = {key: value for key, value in spec.items()
                           if key not in ("cache", "threshold", "max_continuous")}
    worker._states[state.request_id] = state
    for index in range(boundary):
        state.latents = scheduler.step(torch.full_like(state.latents, 0.1),
            scheduler.timesteps[index], state.latents, return_dict=False)[0]
        state.step_index += 1
    for context in state.contexts.values():
        context.executed_steps = context.accumulated_executed_steps = boundary
        context.transformer_executed_steps = context.accumulated_transformer_executed_steps = boundary * 2
        context.buffers = {
            "None_Fn_residual_buffer": torch.ones_like(state.latents),
            "None_Fn_residual_buffer_cfg": torch.ones_like(state.latents),
            "None_Bn_residual_buffer": torch.full_like(state.latents, .12),
            "None_Bn_residual_buffer_cfg": torch.full_like(state.latents, .08)}

    class Transformer:
        dtype = torch.float32

        def __init__(self):
            self.branch = None
            self.calls = []
            self.fail_on_call = None
            self.consume_random = random

        @contextlib.contextmanager
        def cache_context(self, name):
            previous, self.branch = self.branch, name
            try:
                yield
            finally:
                self.branch = previous

        def __call__(self, **kwargs):
            return self.forward(**kwargs)

        def original(self, *, hidden_states, **kwargs):
            self.calls.append((self.branch, False))
            result = hidden_states * 0.001 + (.11 if self.branch == "cond" else .07)
            if self.consume_random:
                result = result + torch.rand(()) * .001 + torch.rand((), generator=state.generator) * .001
            return (result,)

        def cached(self, *, hidden_states, **kwargs):
            manager = worker.manager
            manager.set_context("blocks")
            manager.mark_step_begin()
            index = 0 if self.branch == "cond" else 1
            residual = torch.full_like(hidden_states, 1.01 if gates[index] else 1.5)
            reuse = manager.can_cache(residual, prefix="None_Fn_residual")
            self.calls.append((self.branch, reuse))
            if self.fail_on_call == len(self.calls):
                # Both tensor and metadata mutations must be rolled back.
                manager._current_context.buffers["None_Fn_residual_buffer"].add_(20)
                torch.rand(())
                torch.rand((), generator=state.generator)
                raise RuntimeError("injected branch failure")
            if reuse:
                manager.add_cached_step()
                result = manager.get_Bn_buffer(prefix="None_Bn_residual") + hidden_states * .001
            else:
                manager.set_Fn_buffer(residual, prefix="None_Fn_residual")
                result = hidden_states * .001 + (.11 if self.branch == "cond" else .07)
                manager.set_Bn_buffer(result, prefix="None_Bn_residual")
            if self.consume_random:
                result = result + torch.rand(()) * .001 + torch.rand((), generator=state.generator) * .001
            return (result,)

    transformer = Transformer()
    worker.pipe = types.SimpleNamespace(transformer=transformer)
    worker._original_forward = transformer.original
    worker._cached_forward = transformer.cached
    transformer.forward = worker._cached_forward
    return worker, state


class CfgAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ("torch", "diffusers", "cache_dit")):
            raise unittest.SkipTest("pinned CPU dependencies required")
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def test_all_four_native_gate_combinations_commit_one_solver_step(self):
        for gates in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(gates=gates):
                worker, state = fixture(gates)
                event = CfgResearchAdapter(worker).step(state, policy=policy())
                self.assertEqual(state.step_index, 9)
                self.assertEqual(state.scheduler.step_index, 9)
                self.assertEqual(event["path"], "reuse" if all(gates) else "compute")
                research = event["cfg_research"]
                self.assertEqual([v["native_reuse"] for v in research["native_attempt"]["gates"]], list(gates))
                mixed = gates[0] != gates[1]
                self.assertEqual(research["mixed_native_attempt"], mixed)
                self.assertEqual(len(worker.pipe.transformer.calls), 4 if mixed else 2)
                self.assertEqual(state.contexts["blocks"].transformer_executed_steps, 18)
                self.assertEqual(len(state.events), 1)
                self.assertNotIn("can_cache", vars(worker.manager))
                self.assertFalse(research["performance_eligible"])
                self.assertGreater(research["snapshot_ms"], 0)

    def test_replay_keeps_native_residual_side_effects_exactly_once(self):
        worker, state = fixture((True, False))
        event = CfgResearchAdapter(worker).step(state, policy=policy())
        context = state.contexts["blocks"]
        self.assertEqual(context.cached_steps, [])
        self.assertEqual(context.cfg_cached_steps, [])
        self.assertEqual(set(context.residual_diffs), {"8"})
        self.assertEqual(set(context.cfg_residual_diffs), {"8"})
        self.assertAlmostEqual(context.accumulated_residual_diff, context.residual_diffs["8"])
        self.assertAlmostEqual(context.cfg_accumulated_residual_diff, context.cfg_residual_diffs["8"])
        self.assertTrue(all(v["forced_full"] for v in event["cfg_research"]["accepted_pair"]["gates"]))
        self.assertEqual(context.cache_config.max_cached_steps, -1)

    def test_continuous_limit_reset_is_preserved_during_mixed_replay(self):
        worker, state = fixture((True, True))
        state.contexts["blocks"].continuous_cached_steps = 1
        event = CfgResearchAdapter(worker).step(state, policy=policy())
        self.assertTrue(event["cfg_research"]["mixed_native_attempt"])
        self.assertEqual(state.contexts["blocks"].continuous_cached_steps, 0)
        self.assertEqual(state.contexts["blocks"].cached_steps, [])

    def test_warmup_and_missing_cache_are_compute(self):
        for boundary, clear in ((0, False), (8, True)):
            with self.subTest(boundary=boundary):
                worker, state = fixture(boundary=boundary)
                if clear:
                    state.contexts["blocks"].buffers.clear()
                event = CfgResearchAdapter(worker).step(state, policy=policy())
                self.assertEqual(event["path"], "compute")
                self.assertFalse(event["cfg_research"]["mixed_native_attempt"])

    def test_cold_anchor_remains_forced_and_clears_once(self):
        worker, state = fixture()
        state.contexts["blocks"].buffers.clear()
        state.needs_anchor = True
        event = CfgResearchAdapter(worker).step(state, policy=policy(), diagnostic=True)
        self.assertTrue(event["forced_cold_anchor"])
        self.assertEqual(event["path"], "compute")
        self.assertFalse(state.needs_anchor)
        self.assertEqual(state.contexts["blocks"].cache_config.max_cached_steps, -1)

    def test_diagnostic_isolation_preserves_trajectory_rng_and_histories(self):
        for identity, gates in (("independent_008", (True, False)), ("paired_008", (True, True))):
            with self.subTest(identity=identity):
                worker, state = fixture(gates, identity=identity, random=True)
                self.torch.manual_seed(900)
                native = CfgResearchAdapter(worker).step(state, policy=policy(identity))
                rng = self.torch.get_rng_state().clone()
                request_rng = state.generator.get_state().clone()
                worker2, state2 = fixture(gates, identity=identity, random=True)
                self.torch.manual_seed(900)
                diagnostic = CfgResearchAdapter(worker2).step(state2, policy=policy(identity), diagnostic=True)
                self.assertTrue(self.torch.equal(state.latents, state2.latents))
                self.assertTrue(self.torch.equal(rng, self.torch.get_rng_state()))
                self.assertTrue(self.torch.equal(request_rng, state2.generator.get_state()))
                self.assertEqual(native["cache_contexts"], diagnostic["cache_contexts"])
                self.assertEqual(state.scheduler.step_index, state2.scheduler.step_index)
                self.assertIsNotNone(diagnostic["cfg_research"]["same_input_prediction_error"])
                self.assertGreater(diagnostic["cfg_research"]["diagnostic_ms"], 0)

    def test_compute_diagnostic_does_not_launch_shadow(self):
        worker, state = fixture((False, False))
        event = CfgResearchAdapter(worker).step(state, policy=policy(), diagnostic=True)
        self.assertEqual(len(worker.pipe.transformer.calls), 2)
        self.assertIsNone(event["cfg_research"]["same_input_prediction_error"])
        self.assertEqual(event["cfg_research"]["diagnostic_ms"], 0)

    def test_mixed_oracle_rng_matches_one_forced_full_pair(self):
        worker, state = fixture((True, False), random=True)
        self.torch.manual_seed(1234)
        CfgResearchAdapter(worker).step(state, policy=policy())
        rng = self.torch.get_rng_state().clone()
        request_rng = state.generator.get_state().clone()
        reference, ref = fixture((False, False), random=True)
        self.torch.manual_seed(1234)
        CfgResearchAdapter(reference).step(ref, policy=policy())
        self.assertTrue(self.torch.equal(state.latents, ref.latents))
        self.assertTrue(self.torch.equal(rng, self.torch.get_rng_state()))
        self.assertTrue(self.torch.equal(request_rng, ref.generator.get_state()))

    def test_branch_failure_rolls_back_payload_aliases_rng_and_manager(self):
        worker, state = fixture((True, False), random=True)
        context = state.contexts["blocks"]
        context.buffers["alias"] = context.buffers["None_Fn_residual_buffer"]
        before = state.latents.clone()
        rng = self.torch.get_rng_state().clone()
        request_rng = state.generator.get_state().clone()
        pointer = object()
        worker.manager.extra_marker = pointer
        forward = worker.pipe.transformer.forward
        worker.pipe.transformer.fail_on_call = 2
        with self.assertRaisesRegex(RuntimeError, "injected branch failure"):
            CfgResearchAdapter(worker).step(state, policy=policy())
        self.assertTrue(self.torch.equal(before, state.latents))
        self.assertEqual(state.step_index, 8)
        self.assertEqual(state.scheduler.step_index, 8)
        self.assertEqual(state.events, [])
        self.assertTrue(self.torch.equal(rng, self.torch.get_rng_state()))
        self.assertTrue(self.torch.equal(request_rng, state.generator.get_state()))
        self.assertIs(worker.manager.extra_marker, pointer)
        self.assertIs(worker.pipe.transformer.forward, forward)
        context = state.contexts["blocks"]
        self.assertIs(context.buffers["alias"], context.buffers["None_Fn_residual_buffer"])
        self.assertTrue(self.torch.equal(context.buffers["alias"], self.torch.ones_like(before)))
        self.assertIs(context.cache_config, context._init_kwargs["cache_config"])
        self.assertIs(state.scheduler.disable_corrector, state.scheduler.config.disable_corrector)
        for value in state.scheduler.timestep_list:
            self.assertEqual(value.untyped_storage().data_ptr(), state.scheduler.timesteps.untyped_storage().data_ptr())

    def test_nonfinite_solver_failure_rolls_back_scheduler_and_no_extra_state_fields(self):
        from ditserve.gpu.worker import REQUEST_STATE_FIELDS
        # An infinite stored prediction is accepted in both branches then rejected at commit.
        worker2, state2 = fixture((True, True))
        for key in ("None_Bn_residual_buffer", "None_Bn_residual_buffer_cfg"):
            state2.contexts["blocks"].buffers[key].fill_(float("inf"))
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            CfgResearchAdapter(worker2).step(state2, policy=policy())
        self.assertEqual(state2.step_index, 8)
        self.assertEqual(state2.scheduler.step_index, 8)
        self.assertEqual(set(vars(state2)), REQUEST_STATE_FIELDS)
        self.assertTrue(self.torch.isfinite(state2.latents).all())

    def test_replay_and_shadow_failure_restore_the_whole_step(self):
        for gates, diagnostic in (((True, False), False), ((True, True), True)):
            with self.subTest(gates=gates):
                worker, state = fixture(gates, random=True)
                before = state.latents.clone()
                rng = self.torch.get_rng_state().clone()
                request_rng = state.generator.get_state().clone()
                worker.pipe.transformer.fail_on_call = 4
                with self.assertRaisesRegex(RuntimeError, "injected branch failure"):
                    CfgResearchAdapter(worker).step(state, policy=policy(), diagnostic=diagnostic)
                self.assertTrue(self.torch.equal(before, state.latents))
                self.assertTrue(self.torch.equal(rng, self.torch.get_rng_state()))
                self.assertTrue(self.torch.equal(request_rng, state.generator.get_state()))
                self.assertEqual(state.contexts["blocks"].cached_steps, [])
                self.assertEqual(state.contexts["blocks"].residual_diffs, {})
                self.assertEqual(state.step_index, 8)
                self.assertEqual(state.events, [])

    def test_storage_clone_preserves_views_strides_object_aliases_and_independence(self):
        torch = self.torch
        base = torch.arange(24, dtype=torch.float32)
        transposed = base.reshape(4, 6).transpose(0, 1)
        sliced = base[3:10]
        graph = _clone_graph(torch, {"base": base, "view": transposed, "slice": sliced, "alias": sliced})
        self.assertIs(graph["slice"], graph["alias"])
        self.assertEqual(graph["view"].stride(), transposed.stride())
        self.assertEqual(graph["slice"].storage_offset(), 3)
        self.assertEqual(graph["view"].untyped_storage().data_ptr(), graph["base"].untyped_storage().data_ptr())
        graph["base"][3] = -99
        self.assertEqual(graph["slice"][0].item(), -99)
        self.assertEqual(base[3].item(), 3)
        with self.assertRaisesRegex(TypeError, "ordinary strided"):
            _clone_graph(torch, torch.eye(2).to_sparse())

    def test_unknown_policy_and_spec_mismatch_rejected_before_execution(self):
        worker, state = fixture()
        for selected in ({**policy(), "id": "t008"}, {**policy(), "implementation": "split_prefix"},
                         {**policy(), "threshold": .04}):
            with self.assertRaises(ValueError):
                CfgResearchAdapter(worker).step(state, policy=selected)
        state.spec["threshold"] = .04
        with self.assertRaisesRegex(ValueError, "spec does not match"):
            CfgResearchAdapter(worker).step(state, policy=policy())
        self.assertEqual(worker.pipe.transformer.calls, [])

    def test_independent_and_off_without_diagnostics_delegate_to_worker(self):
        for identity in ("off", "independent_004", "independent_008"):
            with self.subTest(identity=identity):
                worker, state = fixture(identity=identity)
                expected = worker.step(state)
                worker2, state2 = fixture(identity=identity)
                actual = CfgResearchAdapter(worker2).step(state2, policy=policy(identity))
                self.assertTrue(self.torch.equal(state.latents, state2.latents))
                self.assertEqual(expected["cache_contexts"], actual["cache_contexts"])
                self.assertEqual(actual["cfg_research"]["implementation"], "native")

    def test_reference_requires_next_matching_off_step(self):
        worker, state = fixture()
        other, reference = fixture(identity="off")
        adapter = CfgResearchAdapter(worker)
        with self.assertRaisesRegex(ValueError, "exactly the next"):
            adapter.step(state, policy=policy(), diagnostic=True, reference_state=reference)
        other.step(reference)
        event = adapter.step(state, policy=policy(), diagnostic=True, reference_state=reference)
        self.assertGreaterEqual(event["cfg_research"]["off_trajectory_latent_error"]["relative_l2"], 0)

    def test_interleaved_workers_leave_no_manager_override(self):
        worker, state = fixture((True, False))
        other, request = fixture((True, True))
        adapter = CfgResearchAdapter(worker)
        adapter.step(state, policy=policy())
        CfgResearchAdapter(other).step(request, policy=policy())
        adapter.step(state, policy=policy())
        self.assertEqual(state.step_index, 10)
        self.assertEqual(request.step_index, 9)
        for manager in (worker.manager, other.manager):
            self.assertIsNone(manager._current_context)
            self.assertEqual(manager._cached_context_manager, {})
            self.assertNotIn("can_cache", vars(manager))

    def test_two_requests_share_worker_without_mixing_contexts(self):
        worker, first = fixture((True, False))
        _, second = fixture((True, False))
        second.request_id = "second"
        second.spec["request_id"] = second.immutable_spec["request_id"] = "second"
        second.owner = worker._owner
        worker._states[second.request_id] = second
        before = second.latents.clone()
        adapter = CfgResearchAdapter(worker)
        adapter.step(first, policy=policy())
        self.assertTrue(self.torch.equal(before, second.latents))
        self.assertEqual(second.contexts["blocks"].executed_steps, 8)
        adapter.step(second, policy=policy(), diagnostic=True)
        self.assertTrue(self.torch.equal(first.latents, second.latents))
        self.assertIsNot(first.contexts["blocks"], second.contexts["blocks"])
        adapter.step(first, policy=policy())
        self.assertEqual(first.step_index, 10)
        self.assertEqual(second.step_index, 9)
        self.assertEqual(second.contexts["blocks"].executed_steps, 9)


if __name__ == "__main__":
    unittest.main()
