"""CPU tensor execution against the real pinned cache gate and UniPC sampler."""

import contextlib
import importlib.util
import threading
import types
import unittest

from ditserve.models.wan.cfg import CfgResearchAdapter, _clone_graph


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
    from ditserve.models.wan.runtime import WanStepWorker, RequestState, normalize_spec

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






if __name__ == "__main__":
    unittest.main()
