"""Small real Wan blocks exercise the split executor without model weights/GPU."""

import contextlib
import copy
import functools
import importlib.util
import unittest
from unittest.mock import patch

from ditserve.research.cfg_adapter import CfgResearchAdapter, _clone_graph
from ditserve.research.cfg_split_prefix import CfgSplitPrefix
from ditserve.research.dual_runtime import correctness_state_digest
from test_research_cfg_adapter import fixture as simple_fixture, policy


def fixture(gates=None, boundary=8):
    import torch
    from diffusers import WanTransformer3DModel
    from cache_dit.caching.cache_blocks import CachedBlocks
    from cache_dit.caching.forward_pattern import ForwardPattern
    worker, state = simple_fixture(boundary=0)
    torch.manual_seed(701)
    model = WanTransformer3DModel(num_attention_heads=1, attention_head_dim=12, in_channels=2,
        out_channels=2, num_layers=30, ffn_dim=24, freq_dim=4, text_dim=4, patch_size=(1, 2, 2),
        rope_max_seq_len=16).eval()
    worker.pipe.transformer = model
    worker.attention = "flash"
    worker._scheduler_type = type(state.scheduler)
    state.latents = torch.linspace(-.1, .1, 32).reshape(1, 2, 1, 4, 4)
    state.prompt_embeds = torch.ones(1, 3, 4)
    state.negative_prompt_embeds = torch.zeros(1, 3, 4)
    for index in range(boundary):
        state.latents = state.scheduler.step(torch.full_like(state.latents, .01),
            state.scheduler.timesteps[index], state.latents, return_dict=False)[0]
        state.step_index += 1
    context = state.contexts["blocks"]
    context.executed_steps = context.accumulated_executed_steps = boundary
    context.transformer_executed_steps = context.accumulated_transformer_executed_steps = boundary * 2
    context.buffers = {
        "blocks_Fn_residual_buffer": torch.ones(1, 4, 12),
        "blocks_Fn_residual_buffer_cfg": torch.ones(1, 4, 12),
        "blocks_Bn_residual_buffer": torch.full((1, 4, 12), .01),
        "blocks_Bn_residual_buffer_cfg": torch.full((1, 4, 12), .02),
        "blocks_Bn_residual_encoder_buffer": torch.zeros(1, 3, 12),
        "blocks_Bn_residual_encoder_buffer_cfg": torch.zeros(1, 3, 12),
    }
    helper = CachedBlocks(model.blocks, transformer=model, forward_pattern=ForwardPattern.Pattern_2,
                           cache_prefix="blocks", cache_context="blocks", context_manager=worker.manager)
    original = model.forward
    model._original_forward = original
    worker._original_forward = original

    def cached(*args, **kwargs):
        with patch.object(model, "blocks", torch.nn.ModuleList([helper])):
            return original(*args, **kwargs)

    worker._cached_forward = cached
    model.forward = cached
    active = [None]
    worker._test_active_branch = active
    counters = {branch: [0] * 30 for branch in ("cond", "uncond")}
    cache_context = model.cache_context

    @contextlib.contextmanager
    def labeled(branch):
        old, active[0] = active[0], branch
        try:
            with cache_context(branch):
                yield
        finally:
            active[0] = old

    model.cache_context = labeled
    for i, block in enumerate(model.blocks):
        forward = block.forward

        @functools.wraps(forward)
        def counted(*args, _i=i, _forward=forward, **kwargs):
            # Native diagnostic shadows after split setup have no context label.
            # Prefix/suffix wrappers below label all accepted split computation.
            if active[0] is not None:
                counters[active[0]][_i] += 1
            return _forward(*args, **kwargs)

        block.forward = counted
    if gates is not None:
        native_gate = worker.manager.can_cache

        def controlled(*args, **kwargs):
            native_gate(*args, **kwargs)
            return gates[1 if worker.manager.is_separate_cfg_step() else 0]

        worker.manager.can_cache = controlled
    return worker, state, counters


def split_adapter(worker):
    adapter = CfgResearchAdapter(worker)
    # The real empty-context proof must see the native method. Branch counting
    # belongs to this fixture's split packets, never a production context hook.
    del worker.pipe.transformer.cache_context
    # Real pinned implementation runs on a tiny shape only in this CPU test;
    # the public constructor continues to reject every non-1.3B model.
    with patch.object(CfgSplitPrefix, "_validate_model", return_value=None):
        adapter._split_prefix = CfgSplitPrefix(worker)
    split, active = adapter._split_prefix, worker._test_active_branch
    prefix, suffix = split._prefix, split._suffix

    def labeled_prefix(state, embedding):
        branch = "cond" if embedding is state.prompt_embeds else "uncond"
        old, active[0] = active[0], branch
        try:
            packet = prefix(state, embedding)
            packet["_test_branch"] = branch
            return packet
        finally:
            active[0] = old

    def labeled_suffix(packet, reused):
        old, active[0] = active[0], packet["_test_branch"]
        try:
            return suffix(packet, reused)
        finally:
            active[0] = old

    split._prefix, split._suffix = labeled_prefix, labeled_suffix
    return adapter


class CfgSplitPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ("torch", "diffusers", "cache_dit")):
            raise unittest.SkipTest("pinned CPU dependencies required")
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def selected(self):
        return dict(policy(), implementation="split_prefix")

    def test_four_gates_match_oracle_state_and_call_each_fn_exactly_once(self):
        for gates in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(gates=gates):
                oracle_worker, oracle, oracle_counts = fixture(gates)
                split_worker, state, counts = fixture(gates)
                before = correctness_state_digest(split_worker, state)
                expected = CfgResearchAdapter(oracle_worker).step(oracle, policy=policy())
                actual = split_adapter(split_worker).step(state, policy=self.selected())
                self.assertEqual(correctness_state_digest(oracle_worker, oracle),
                                 correctness_state_digest(split_worker, state))
                self.assertNotEqual(before, correctness_state_digest(split_worker, state))
                self.assertEqual(actual["path"], expected["path"])
                for branch in ("cond", "uncond"):
                    self.assertEqual(counts[branch][:8], [1] * 8)
                    self.assertEqual(counts[branch][8:], [0 if all(gates) else 1] * 22)
                    if gates[0] != gates[1]:
                        self.assertEqual(oracle_counts[branch][:8], [2] * 8)
                event = actual["cfg_research"]
                self.assertTrue(event["performance_eligible"])
                self.assertEqual(event["replay_ms"], 0)
                self.assertEqual(event["discarded_probe_ms"], 0)
                self.assertGreater(event["split_prefix"]["gate_probe_ms"], 0)
                self.assertGreater(event["split_prefix"]["context_guard_ms"], 0)
                self.assertEqual(event["split_prefix"]["cache_context_mode"], "noop_guarded_v1")
                self.assertNotIn("optimistic_compute_ms", event)
                self.assertEqual(event["split_prefix"]["gate_calls"], dict(speculative=2, committed=2))

    def test_old_tensor_storage_is_unchanged_on_compute_reuse_and_mixed_paths(self):
        for gates in ((False, False), (True, True), (False, True), (True, False)):
            with self.subTest(gates=gates):
                worker, state, _ = fixture(gates)
                originals = []

                def remember(tensor):
                    originals.append((tensor, tensor.clone()))
                    return tensor

                from ditserve.gpu.worker import _tree_map
                _tree_map(worker._payload(state), self.torch.is_tensor, remember)
                split_adapter(worker).step(state, policy=self.selected())
                for original, backup in originals:
                    self.assertTrue(self.torch.equal(original, backup), tuple(original.shape))

    def test_actual_wan_warmup_and_following_steps_match_oracle(self):
        oracle_worker, oracle, _ = fixture(boundary=0)
        worker, state, counts = fixture(boundary=0)
        left, right = CfgResearchAdapter(oracle_worker), split_adapter(worker)
        for _ in range(12):
            left.step(oracle, policy=policy())
            right.step(state, policy=self.selected())
            self.assertEqual(correctness_state_digest(oracle_worker, oracle), correctness_state_digest(worker, state))
        self.assertEqual(counts["cond"][:8], [12] * 8)
        self.assertEqual(counts["uncond"][:8], [12] * 8)

    def test_diagnostics_use_same_input_and_leave_state_identical(self):
        clean_worker, clean, _ = fixture((True, True))
        worker, state, _ = fixture((True, True))
        split_adapter(clean_worker).step(clean, policy=self.selected())
        event = split_adapter(worker).step(state, policy=self.selected(), diagnostic=True)
        self.assertEqual(correctness_state_digest(clean_worker, clean), correctness_state_digest(worker, state))
        self.assertFalse(event["cfg_research"]["performance_eligible"])
        self.assertIsNotNone(event["cfg_research"]["same_input_prediction_error"])

    def test_complete_boundary_restore_preserves_next_split_steps(self):
        worker, state, _ = fixture(boundary=0)
        adapter = split_adapter(worker)
        for _ in range(9):
            adapter.step(state, policy=self.selected())
        with self.torch.inference_mode():
            checkpoint = adapter._snapshot(state)
        expected = []
        for _ in range(3):
            adapter.step(state, policy=self.selected())
            expected.append(correctness_state_digest(worker, state))
        with self.torch.inference_mode():
            adapter._restore(state, checkpoint)
        for expected_digest in expected:
            adapter.step(state, policy=self.selected())
            self.assertEqual(expected_digest, correctness_state_digest(worker, state))

    def test_cold_missing_cache_and_continuous_limit_match_oracle(self):
        for mode in ("cold", "missing", "limit"):
            with self.subTest(mode=mode):
                oracle_worker, oracle, _ = fixture()
                worker, state, _ = fixture()
                for request in (oracle, state):
                    if mode in ("cold", "missing"):
                        request.contexts["blocks"].buffers.clear()
                        request.needs_anchor = mode == "cold"
                    else:
                        request.contexts["blocks"].continuous_cached_steps = 1
                        request.contexts["blocks"].cfg_continuous_cached_steps = 1
                CfgResearchAdapter(oracle_worker).step(oracle, policy=policy())
                split_adapter(worker).step(state, policy=self.selected())
                self.assertEqual(correctness_state_digest(oracle_worker, oracle), correctness_state_digest(worker, state))

    def test_failure_after_conditional_commit_rolls_back_request_and_rng(self):
        worker, state, _ = fixture((False, False))
        adapter = split_adapter(worker)
        before = correctness_state_digest(worker, state)
        cpu_rng = self.torch.get_rng_state().clone()
        request_rng = state.generator.get_state().clone()
        suffix = adapter._split_prefix._suffix
        calls = [0]

        def fail(packet, reuse):
            calls[0] += 1
            if calls[0] == 2:
                self.torch.rand(())
                self.torch.rand((), generator=state.generator)
                raise RuntimeError("injected suffix failure")
            return suffix(packet, reuse)

        with patch.object(adapter._split_prefix, "_suffix", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "injected suffix failure"):
                adapter.step(state, policy=self.selected())
        self.assertEqual(before, correctness_state_digest(worker, state))
        self.assertTrue(self.torch.equal(cpu_rng, self.torch.get_rng_state()))
        self.assertTrue(self.torch.equal(request_rng, state.generator.get_state()))
        self.assertEqual(state.events, [])
        self.assertIsNone(worker.manager._current_context)
        self.assertEqual(worker.manager._cached_context_manager, {})

    def test_probe_commit_disagreement_is_rejected_with_full_rollback(self):
        worker, state, _ = fixture((True, True))
        adapter = split_adapter(worker)
        before = correctness_state_digest(worker, state)
        gate = adapter._split_prefix._gate
        calls = [0]

        def changed(packet, branch):
            event = gate(packet, branch)
            calls[0] += 1
            if calls[0] == 3:
                event["native_reuse"] = False
            return event

        with patch.object(adapter._split_prefix, "_gate", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "gate changed"):
                adapter.step(state, policy=self.selected())
        self.assertEqual(before, correctness_state_digest(worker, state))

    def test_late_hook_rejection_precedes_computation_and_restores_request(self):
        worker, state, counts = fixture((True, True))
        adapter = split_adapter(worker)
        before = correctness_state_digest(worker, state)
        cpu_rng = self.torch.get_rng_state().clone()
        handle = worker.pipe.transformer.blocks[0].register_forward_hook(lambda *args: None)
        try:
            with self.assertRaisesRegex(ValueError, "empty cache-context guard"):
                adapter.step(state, policy=self.selected())
        finally:
            handle.remove()
        self.assertEqual(before, correctness_state_digest(worker, state))
        self.assertTrue(self.torch.equal(cpu_rng, self.torch.get_rng_state()))
        self.assertTrue(all(not any(values) for values in counts.values()))
        self.assertEqual(state.events, [])
        # Removing the unsupported mutation makes the same request resumable.
        adapter.step(state, policy=self.selected())
        self.assertEqual(state.step_index, 9)

    def test_explicit_attention_context_covers_prefix_and_suffix(self):
        worker, state, _ = fixture((False, False))
        adapter, active = split_adapter(worker), [False]

        @contextlib.contextmanager
        def attention():
            active[0] = True
            try:
                yield
            finally:
                active[0] = False

        worker._dit_attention_context = attention
        prefix, suffix = adapter._split_prefix._prefix, adapter._split_prefix._suffix

        def guarded(call):
            def checked(*args):
                self.assertTrue(active[0])
                return call(*args)
            return checked

        with patch.object(adapter._split_prefix, "_prefix", side_effect=guarded(prefix)), \
                patch.object(adapter._split_prefix, "_suffix", side_effect=guarded(suffix)):
            adapter.step(state, policy=self.selected())
        self.assertFalse(active[0])

    def test_pinned_contract_rejects_tiny_model_and_unknown_cache_modes(self):
        worker, state, _ = fixture()
        with self.assertRaisesRegex(ValueError, "unsupported model"):
            CfgSplitPrefix(worker)
        for name, value in (("Fn_compute_blocks", 7), ("Bn_compute_blocks", 1),
                            ("cfg_diff_compute_separate", False), ("max_continuous_cached_steps", 2)):
            with self.subTest(name=name):
                worker, state, _ = fixture()
                adapter = split_adapter(worker)
                setattr(state.contexts["blocks"].cache_config, name, value)
                with self.assertRaises(ValueError):
                    adapter.step(state, policy=self.selected())

    def test_unknown_cache_buffers_rejected_before_execution(self):
        for key, value in (("blocks_unknown_buffer", self.torch.zeros(1)),
                           ("other_Fn_residual_buffer", self.torch.zeros(1)),
                           ("blocks_Fn_residual_buffer", object()),
                           (17, self.torch.zeros(1))):
            with self.subTest(key=key):
                worker, state, counts = fixture(boundary=0)
                adapter = split_adapter(worker)
                state.contexts["blocks"].buffers[key] = value
                with self.assertRaisesRegex(ValueError, "unknown feature buffers"):
                    adapter.step(state, policy=self.selected())
                self.assertEqual(state.step_index, 0)
                self.assertTrue(all(not any(values) for values in counts.values()))

    def test_full_shape_meta_model_contract_rejects_training_hooks_sp_compile_and_random_forward(self):
        from diffusers import WanTransformer3DModel
        worker, state = simple_fixture()
        worker._scheduler_type = type(state.scheduler)
        worker.attention = "flash"
        with self.torch.device("meta"):
            model = WanTransformer3DModel(num_attention_heads=12, attention_head_dim=128,
                in_channels=16, out_channels=16, num_layers=30, ffn_dim=8960,
                text_dim=4096, patch_size=(1, 2, 2)).to(dtype=self.torch.float16).eval()
        model._original_forward = model.forward
        worker.pipe.transformer = model
        CfgSplitPrefix(worker)
        model.train()
        with self.assertRaisesRegex(ValueError, "training"):
            CfgSplitPrefix(worker)
        model.eval()
        for attribute, value in (("_is_parallelized", True), ("_compiled_call_impl", object()), ("_hf_hook", object())):
            with self.subTest(attribute=attribute):
                setattr(model, attribute, value)
                with self.assertRaises(ValueError):
                    CfgSplitPrefix(worker)
                delattr(model, attribute)
        handle = model.blocks[0].register_forward_hook(lambda *args: None)
        with self.assertRaisesRegex(ValueError, "hooks"):
            CfgSplitPrefix(worker)
        handle.remove()
        model.blocks[0].forward = lambda *args: self.torch.rand(())
        with self.assertRaisesRegex(ValueError, "modified module forwards"):
            CfgSplitPrefix(worker)


if __name__ == "__main__":
    unittest.main()
