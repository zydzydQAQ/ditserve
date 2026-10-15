"""CPU-only state accounting and conservative refresh-proof contracts."""

import dataclasses
import copy
import gc
import importlib.metadata
import json
import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

from ditserve.gpu.worker import (CACHED_CONTEXT_FIELDS, DBCACHE_CONFIG_FIELDS, EXTRA_CACHE_CONFIG_FIELDS,
                                 UNIPC_REQUIRED_FIELDS, RequestState, normalize_spec)
from ditserve.research.state import (assert_transfer_supported, refresh_certificate, state_inventory,
                                     transition_plan, validate_refresh_certificate)


def make_state(torch):
    spec = normalize_spec(dict(request_id="a", prompt="a boat", seed=1, height=32, width=32,
                               frames=5, steps=32, guidance_scale=5., cache=True,
                               threshold=.08, max_continuous=1))
    cfg = dict.fromkeys(DBCACHE_CONFIG_FIELDS)
    cfg.update(cache_type="DBCache", Fn_compute_blocks=8, Bn_compute_blocks=0, max_warmup_steps=8,
               warmup_interval=1, max_cached_steps=-1, max_continuous_cached_steps=1,
               enable_separate_cfg=True, cfg_compute_first=False, cfg_diff_compute_separate=True,
               steps_computation_policy="dynamic", force_refresh_step_policy="once",
               num_inference_steps=32, residual_diff_threshold=.08)
    Config = dataclasses.make_dataclass("Config", [(k, object) for k in cfg])
    config = Config(**cfg)
    extra = dict(l1_hidden_states_diff_threshold=None, important_condition_threshold=0., downsample_factor=1)
    Extra = dataclasses.make_dataclass("Extra", [(k, object) for k in EXTRA_CACHE_CONFIG_FIELDS])
    fields = dict.fromkeys(CACHED_CONTEXT_FIELDS)
    fields.update(name="blocks", cache_config=config, extra_cache_config=Extra(**extra),
                  buffers={"None_Fn_residual_buffer": torch.ones(8),
                           "None_Bn_residual_buffer": torch.full((8,), 2.),
                           "None_Fn_residual_buffer_cfg": torch.full((8,), 3.),
                           "None_Bn_residual_buffer_cfg": torch.full((8,), 4.)},
                  cached_steps=[15], cfg_cached_steps=[15], residual_diffs={"15": .01},
                  cfg_residual_diffs={"15": .02}, continuous_cached_steps=1, cfg_continuous_cached_steps=1,
                  executed_steps=16, accumulated_executed_steps=16, transformer_executed_steps=32,
                  accumulated_transformer_executed_steps=32, accumulated_cached_steps=1,
                  cfg_accumulated_cached_steps=1, accumulated_residual_diff=.01,
                  cfg_accumulated_residual_diff=.02, _init_args=[], _init_kwargs={"cache_config": config})
    context = SimpleNamespace(**fields)
    scheduler = dict.fromkeys(UNIPC_REQUIRED_FIELDS)
    scheduler.update(_step_index=16, this_order=2, model_outputs=[torch.full((4,), 5.)],
                     timestep_list=[3, 2], last_sample=torch.full((4,), 6.),
                     timesteps=torch.arange(32), _internal_dict={"solver_order": 2})
    state = RequestState(spec["request_id"], spec, step_index=16, latents=torch.arange(4, dtype=torch.float32),
                         cache_enabled=True, scheduler=SimpleNamespace(**scheduler),
                         generator=torch.Generator().manual_seed(4), prompt_embeds=torch.zeros(2, 3),
                         negative_prompt_embeds=torch.ones(2, 3), contexts={"blocks": context}, owner=object())
    state.immutable_spec = {k: v for k, v in spec.items() if k not in ("cache", "threshold", "max_continuous")}
    return state


class ResearchStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU torch required")
        cls.torch = torch

    def test_counts_underlying_storage_and_tracks_every_alias_role(self):
        state = make_state(self.torch)
        tensor = self.torch.arange(20, dtype=self.torch.float32)
        state.latents = tensor
        state.scheduler.last_sample = tensor
        state.contexts["blocks"].buffers["None_Fn_residual_buffer"] = tensor[2:10]
        report = state_inventory(state)
        row = next(r for r in report["storages"] if "latents" in r["paths"])
        self.assertEqual(row["bytes"], 80)
        self.assertEqual(row["roles"], ["cache_feature", "sampler"])
        self.assertEqual(len(row["paths"]), 3)
        self.assertFalse(row["feature_only"])
        self.assertTrue(row["distinct_view_alias"])
        self.assertEqual(report["shared_role_bytes"], 80)
        self.assertEqual(report["feature_only_bytes"], 3 * 32)
        with self.assertRaisesRegex(ValueError, "distinct tensor views"):
            assert_transfer_supported(report)
        json.dumps(report)

    def test_same_tensor_alias_is_supported_and_not_counted_twice(self):
        state = make_state(self.torch)
        state.scheduler.last_sample = state.latents
        report = state_inventory(state)
        row = next(r for r in report["storages"] if "latents" in r["paths"])
        self.assertEqual(len(row["tensor_ids"]), 1)
        assert_transfer_supported(report)

    def test_reports_do_not_retain_tensors(self):
        state = make_state(self.torch)
        tensor = state.contexts["blocks"].buffers["None_Fn_residual_buffer"]
        ref = weakref.ref(tensor)
        report = state_inventory(state)
        del state, tensor
        gc.collect()
        self.assertIsNone(ref())
        self.assertGreater(report["storage_bytes"], 0)

    def test_certificate_is_pure_and_repeated_inspection_stable(self):
        state = make_state(self.torch)
        before_rng = state.generator.get_state().clone()
        context = state.contexts["blocks"]
        counts = (context.continuous_cached_steps, context.cfg_continuous_cached_steps)
        cert = refresh_certificate(state)
        self.assertTrue(cert["eligible"], cert)
        self.assertEqual(cert, refresh_certificate(state))
        self.assertEqual(counts, (context.continuous_cached_steps, context.cfg_continuous_cached_steps))
        self.assertTrue(self.torch.equal(before_rng, state.generator.get_state()))
        self.assertEqual(state.events, [])
        self.assertEqual(cert["qualification"], "validation_only_requires_GPU_trajectory_evidence")

    def test_requires_both_cfg_branches_at_natural_m1_refresh(self):
        for change in ("one_branch", "m3", "anchor", "warmup", "wrong_history", "empty", "mask"):
            state = make_state(self.torch)
            context = state.contexts["blocks"]
            if change == "one_branch": context.cfg_continuous_cached_steps = 0
            if change == "m3": state.spec["max_continuous"] = 3
            if change == "anchor": state.needs_anchor = True
            if change == "warmup": state.step_index = 7
            if change == "wrong_history": context.cached_steps = [12]
            if change == "empty": context.buffers.clear()
            if change == "mask": context.cache_config.steps_computation_mask = [1] * 32
            with self.subTest(change=change):
                self.assertFalse(refresh_certificate(state)["eligible"])

    def test_pinned_dependency_natural_guard_resets_both_branches_before_buffer_read(self):
        try:
            version = importlib.metadata.version("cache-dit")
        except importlib.metadata.PackageNotFoundError:
            raise unittest.SkipTest("cache-dit 1.5.0 is required for the dependency contract check")
        if version != "1.5.0":
            raise unittest.SkipTest("dependency control-flow proof is pinned to cache-dit 1.5.0")
        from cache_dit.caching.cache_contexts.cache_config import DBCacheConfig
        from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager

        endings = []
        for clear_features, cold_anchor in ((False, False), (True, False), (True, True)):
            state = make_state(self.torch)
            original = state.contexts["blocks"]
            manager = CachedContextManager()
            config = DBCacheConfig(**vars(original.cache_config))
            context = manager.new_context(name="blocks", cache_config=config)
            for key, value in vars(original).items():
                if key not in ("cache_config", "extra_cache_config", "_init_args", "_init_kwargs"):
                    setattr(context, key, value)
            state.contexts["blocks"] = context
            # In 1.5.0 a gap followed by consecutive reuse can produce 2 even
            # with m=1. The actual natural guard is >=1, not ==1.
            context.continuous_cached_steps = context.cfg_continuous_cached_steps = 2
            self.assertTrue(refresh_certificate(state)["eligible"])
            if clear_features:
                context.buffers.clear()
            if cold_anchor:
                config.max_cached_steps = 0
            manager.set_context(context)
            with patch.object(manager, "get_Fn_buffer", side_effect=AssertionError("old feature was read")):
                for _ in range(2):
                    manager.mark_step_begin()
                    self.assertFalse(manager.can_cache(self.torch.ones(8), prefix="None_Fn_residual"))
            self.assertEqual(context.cached_steps, [15])
            self.assertEqual(context.cfg_cached_steps, [15])
            endings.append((context.continuous_cached_steps, context.cfg_continuous_cached_steps,
                            context.executed_steps, context.transformer_executed_steps))
        self.assertEqual(endings[0], (0, 0, 17, 34))
        self.assertEqual(endings[0], endings[1])
        self.assertEqual(endings[2], (2, 2, 17, 34))

    def test_certificate_rejects_config_scheduler_rng_and_tensor_mutation(self):
        changes = [lambda s: s.spec.update(threshold=.04),
                   lambda s: setattr(s.scheduler, "_step_index", 17),
                   lambda s: s.generator.manual_seed(19),
                   lambda s: s.latents.add_(1),
                   lambda s: s.scheduler.model_outputs[0].add_(1),
                   lambda s: s.contexts["blocks"].residual_diffs.update({"15": .03}),
                   lambda s: setattr(s, "owner", object())]
        for change in changes:
            state = make_state(self.torch)
            cert = refresh_certificate(state)
            change(state)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_refresh_certificate(state, cert)

    def test_certificate_detects_inference_tensor_mutation_without_version_counter(self):
        state = make_state(self.torch)
        with self.torch.inference_mode():
            state.latents = self.torch.ones(4)
            cert = refresh_certificate(state)
            state.latents.add_(1)
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_refresh_certificate(state, cert)

    def test_certificate_is_bound_to_layout_request_and_its_own_contents(self):
        state = make_state(self.torch)
        cert = refresh_certificate(state)
        with self.assertRaises(ValueError):
            validate_refresh_certificate(state, cert, layout=dict(sp_size=1, ranks=[1], device="cpu", version=1))
        cert["reclaimable_bytes"] += 1
        with self.assertRaises(ValueError):
            validate_refresh_certificate(state, cert)
        self.assertFalse(refresh_certificate(state, dependency_version="1.6.0")["eligible"])

    def test_unknown_context_and_tensor_metadata_are_not_silently_ignored(self):
        state = make_state(self.torch)
        state.contexts["blocks"].unexpected = self.torch.ones(2)
        with self.assertRaises(TypeError):
            state_inventory(state)
        state = make_state(self.torch)
        state.scheduler._internal_dict["opaque"] = threading.Lock()
        with self.assertRaises(TypeError):
            refresh_certificate(state)

    def test_shared_containers_can_be_fingerprinted_but_cycles_are_rejected(self):
        state = make_state(self.torch)
        shared = [1, 2]
        state.scheduler._internal_dict.update(left=shared, right=shared)
        self.assertTrue(refresh_certificate(state)["eligible"])
        shared.append(shared)
        with self.assertRaises(TypeError):
            refresh_certificate(state)

    def test_transition_semantics_never_promote_cold_or_reshard_to_exact(self):
        state = make_state(self.torch)
        cert = refresh_certificate(state)
        exact = transition_plan(state, "exact_refresh", certificate=cert)
        self.assertEqual(exact["implementation_status"], "validation_only")
        self.assertEqual(transition_plan(state, "cold")["trajectory_semantics"], "approximate_cold_anchor")
        for action, target in (("reshard", None), ("hot", dict(sp_size=2, ranks=[0, 1], device="cpu", version=1))):
            with self.assertRaises(NotImplementedError):
                transition_plan(state, action, target_layout=target)


class ResearchEncoderResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for package, version in (("cache-dit", "1.5.0"), ("diffusers", "0.36.0")):
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                raise unittest.SkipTest("pinned " + package + " is required")
            if actual != version:
                raise unittest.SkipTest("dependency contract requires " + package + " " + version)
        import torch
        from cache_dit.caching.cache_blocks.pattern_0_1_2 import CachedBlocks_Pattern_0_1_2
        from cache_dit.caching.cache_contexts.cache_config import DBCacheConfig
        from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
        from cache_dit.caching.forward_pattern import ForwardPattern
        from diffusers.models.transformers.transformer_wan import WanTransformerBlock
        cls.torch = torch
        cls.CachedBlocks = CachedBlocks_Pattern_0_1_2
        cls.CacheConfig = DBCacheConfig
        cls.CacheManager = CachedContextManager
        cls.Pattern = ForwardPattern.Pattern_2
        cls.WanBlock = WanTransformerBlock

    def wrapper(self, blocks, manager):
        return self.CachedBlocks(blocks, forward_pattern=self.Pattern, cache_prefix="blocks",
                                 cache_context="blocks", context_manager=manager)

    def natural_boundary(self):
        torch = self.torch
        # Fork only the CPU RNG; no CUDA state or model weights are loaded.
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(7)
            blocks = torch.nn.ModuleList([
                self.WanBlock(dim=8, ffn_dim=16, num_heads=2) for _ in range(9)])
            self.assertTrue(all(p.device.type == "cpu" for p in blocks.parameters()))
            manager = self.CacheManager()
            context = manager.new_context(name="blocks", cache_config=self.CacheConfig(
                Fn_compute_blocks=8, Bn_compute_blocks=0, max_warmup_steps=8,
                max_continuous_cached_steps=1, enable_separate_cfg=True,
                cfg_compute_first=False, cfg_diff_compute_separate=True,
                num_inference_steps=32, residual_diff_threshold=.08))
            cached = self.wrapper(blocks, manager)
            # Distinct CFG inputs, repeated per branch across steps, let the
            # real similarity decision naturally reach a reuse boundary.
            inputs = [(torch.randn(1, 3, 8), torch.randn(1, 2, 8), torch.randn(1, 6, 8), None)
                      for _ in range(2)]
            for completed in range(1, 17):
                for branch in inputs:
                    cached(*branch)
                if (completed >= 8 and context.continuous_cached_steps >= 1
                        and context.cfg_continuous_cached_steps >= 1
                        and context.cached_steps[-1] == completed - 1
                        and context.cfg_cached_steps[-1] == completed - 1):
                    break
            else:
                self.fail("real pinned dependency never reached a natural m1 refresh boundary")
            expected = {"blocks_" + kind + "_residual" + middle + "_buffer" + cfg
                        for kind, middle in (("Fn", ""), ("Bn", ""), ("Bn", "_encoder"))
                        for cfg in ("", "_cfg")}
            self.assertEqual(set(context.buffers), expected)
            self.assertTrue(all(value.device.type == "cpu" for value in context.buffers.values()))
            encoder_keys = [key for key in expected if "encoder" in key]
            self.assertTrue(all(torch.count_nonzero(context.buffers[key]).item() == 0 for key in encoder_keys))
            state = make_state(torch)
            state.step_index = state.scheduler._step_index = completed
            state.contexts = {"blocks": context}
            return state, blocks, manager, cached, inputs

    def assert_context_equal(self, left, right):
        self.assertEqual(set(left.buffers), set(right.buffers))
        for key in left.buffers:
            self.assertEqual(left.buffers[key].dtype, right.buffers[key].dtype)
            self.assertTrue(self.torch.equal(left.buffers[key], right.buffers[key]), key)
        self.assertEqual({k: v for k, v in vars(left).items() if k != "buffers"},
                         {k: v for k, v in vars(right).items() if k != "buffers"})

    def test_real_pattern2_rebuilds_encoder_buffers_without_reading_old_features(self):
        state, blocks, manager, cached, inputs = self.natural_boundary()
        context = state.contexts["blocks"]
        cleared_manager = self.CacheManager()
        with self.torch.inference_mode():
            cleared_context = copy.deepcopy(context)
            cleared_manager._cached_context_manager = {"blocks": cleared_context}
            cleared_context.buffers.clear()
            cleared = self.wrapper(blocks, cleared_manager)
            # Catch *all* old-feature reads, including Fn, Bn and encoder
            # buffers. Zero-valued encoder residuals cannot hide a bad read.
            with patch.object(manager, "get_buffer", side_effect=AssertionError("old feature read on natural full")), \
                    patch.object(cleared_manager, "get_buffer", side_effect=AssertionError("reclaimed feature read")):
                for branch in inputs:
                    self.assertTrue(self.torch.equal(cached(*branch), cleared(*branch)))
            for current in (context, cleared_context):
                self.assertEqual(current.continuous_cached_steps, 0)
                self.assertEqual(current.cfg_continuous_cached_steps, 0)
            self.assert_context_equal(context, cleared_context)
            # Continue through future real cache decisions, not just the
            # immediate full step, to verify all rebuilt features and metadata.
            for _ in range(3):
                for branch in inputs:
                    self.assertTrue(self.torch.equal(cached(*branch), cleared(*branch)))
                self.assert_context_equal(context, cleared_context)

    def test_certificate_accepts_real_six_buffers_and_rejects_unknown_encoder_types(self):
        state, _, _, _, _ = self.natural_boundary()
        context = state.contexts["blocks"]
        certificate = refresh_certificate(state)
        self.assertTrue(certificate["eligible"], certificate["reasons"])
        self.assertEqual(certificate["qualification"], "validation_only_requires_GPU_trajectory_evidence")
        for unknown in ("blocks_Fn_residual_encoder_buffer", "blocks_Bn_hidden_states_encoder_buffer"):
            context.buffers[unknown] = self.torch.ones(1)
            try:
                rejected = refresh_certificate(state)
                self.assertFalse(rejected["eligible"])
                self.assertIn("blocks:unsupported_feature_buffer", rejected["reasons"])
            finally:
                del context.buffers[unknown]
        context.cfg_continuous_cached_steps = 0
        self.assertFalse(refresh_certificate(state)["eligible"])


if __name__ == "__main__":
    unittest.main()
