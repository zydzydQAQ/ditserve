"""CPU contracts for cached heterogeneous migration, independent of model weights."""

import copy
import importlib.util
import threading
import unittest
from collections import defaultdict
from unittest.mock import patch

from ditserve.research.dual_state import (MigrationEndpoint, ParallelContractError, migrate_request,
                                          pack_request, unpack_request, local_roundtrip)
from ditserve.research.state import refresh_certificate


def fixture(*, cached=True, boundary=16, context_name="blocks"):
    import torch
    import cache_dit
    from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
    from diffusers import UniPCMultistepScheduler
    from ditserve.gpu.worker import WanStepWorker, RequestState, normalize_spec
    worker = WanStepWorker.__new__(WanStepWorker)
    worker.torch, worker.cache_dit, worker.device = torch, cache_dit, "cpu"
    worker._owner, worker._states = object(), {}
    worker._lock, worker.manager = threading.RLock(), CachedContextManager()
    worker._context_names = (context_name,)
    worker._sync = lambda: None
    worker._scheduler_type = UniPCMultistepScheduler
    spec = normalize_spec(dict(request_id="request-a", prompt="a kettle", seed=101,
                               height=32, width=32, frames=5, steps=32, guidance_scale=5.,
                               cache=cached, threshold=.08 if cached else 0., max_continuous=1))
    scheduler = UniPCMultistepScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(32, device="cpu")
    state = RequestState(spec["request_id"], spec, scheduler=scheduler, cache_enabled=cached,
                         latents=torch.arange(512, dtype=torch.float32).reshape(1, 16, 2, 4, 4) / 512,
                         prompt_embeds=torch.ones(1, 512, 4096, dtype=torch.float16),
                         negative_prompt_embeds=torch.zeros(1, 512, 4096, dtype=torch.float16),
                         generator=torch.Generator().manual_seed(spec["seed"]), owner=worker._owner,
                         contexts=worker._contexts(spec) if cached else {})
    state.immutable_spec = {key: value for key, value in spec.items() if key not in ("cache", "threshold", "max_continuous")}
    worker._states[state.request_id] = state
    advance(state, boundary)
    if cached:
        context = state.contexts[context_name]
        context.executed_steps = context.accumulated_executed_steps = boundary
        context.transformer_executed_steps = context.accumulated_transformer_executed_steps = 2 * boundary
        context.continuous_cached_steps = context.cfg_continuous_cached_steps = 1
        context.cached_steps, context.cfg_cached_steps = [boundary - 1], [boundary - 1]
        context.accumulated_cached_steps = context.cfg_accumulated_cached_steps = 1
        context.residual_diffs = defaultdict(float, {str(boundary - 1): .01})
        context.cfg_residual_diffs = defaultdict(float, {str(boundary - 1): .02})
        context.accumulated_residual_diff, context.cfg_accumulated_residual_diff = .01, .02
        context.buffers = {"None_Fn_residual_buffer": torch.ones(8),
                           "None_Fn_residual_buffer_cfg": torch.full((8,), 2.),
                           "None_Bn_residual_buffer": torch.full((8,), 3.),
                           "None_Bn_residual_buffer_cfg": torch.full((8,), 4.)}
    return worker, state


def advance(state, count=1):
    import torch
    for _ in range(count):
        index = state.step_index
        prediction = torch.full_like(state.latents, .05 + .001 * index, dtype=torch.float16)
        state.latents = state.scheduler.step(prediction, state.scheduler.timesteps[index], state.latents,
                                            return_dict=False)[0]
        state.step_index += 1


def empty_target(context_name="blocks"):
    worker, state = fixture(cached=False, boundary=0, context_name=context_name)
    worker.release(state)
    return worker


class DualStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ("torch", "diffusers", "cache_dit")):
            raise unittest.SkipTest("pinned CPU dependencies required")
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def export(self, worker, state, action="full_migrate", **kwargs):
        return pack_request(worker, state, action=action, target_device="GPU-target",
                            migration_id="migration-1", **kwargs)

    def decode(self, target, manifest, tensors):
        return unpack_request(target, manifest, tensors, target_device="GPU-target", migration_id="migration-1")

    def test_full_roundtrip_preserves_sampler_rng_cfg_and_known_aliases(self):
        worker, state = fixture()
        context = state.contexts["blocks"]
        context.buffers["None_Bn_residual_encoder_buffer"] = context.buffers["None_Fn_residual_buffer"]
        original_rng = state.generator.get_state().clone()
        manifest, tensors = self.export(worker, state)
        restored = self.decode(empty_target(), manifest, tensors)
        self.assertIs(restored.contexts["blocks"]._init_kwargs["cache_config"], restored.contexts["blocks"].cache_config)
        self.assertIs(restored.contexts["blocks"].buffers["None_Bn_residual_encoder_buffer"],
                      restored.contexts["blocks"].buffers["None_Fn_residual_buffer"])
        self.assertIsInstance(restored.contexts["blocks"].residual_diffs, defaultdict)
        self.assertIs(restored.contexts["blocks"].residual_diffs.default_factory, float)
        self.assertEqual(restored.contexts["blocks"].cfg_residual_diffs["15"], .02)
        self.assertTrue(self.torch.equal(original_rng, restored.generator.get_state()))
        self.assertIs(restored.scheduler.disable_corrector, restored.scheduler.config.disable_corrector)
        for view in restored.scheduler.timestep_list:
            self.assertEqual(view.untyped_storage().data_ptr(), restored.scheduler.timesteps.untyped_storage().data_ptr())
        for _ in range(16):
            advance(state)
            advance(restored)
            self.assertTrue(self.torch.equal(state.latents, restored.latents))

    def test_omit_requires_fresh_source_proof_and_does_not_force_cold_anchor(self):
        worker, state = fixture()
        cert = refresh_certificate(state)
        self.assertTrue(cert["eligible"], cert["reasons"])
        manifest, tensors = self.export(worker, state, "certified_omit", certificate=cert)
        restored = self.decode(empty_target(), manifest, tensors)
        self.assertFalse(restored.needs_anchor)
        self.assertFalse(restored.contexts["blocks"].buffers)
        self.assertTrue(state.contexts["blocks"].buffers)
        self.assertEqual(manifest["certificate_binding"]["source_state_sha256"], manifest["source_state_sha256"])
        state.generator.manual_seed(9)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.export(worker, state, "certified_omit", certificate=cert)

    def test_cold_keeps_cfg_metadata_but_sets_explicit_anchor(self):
        worker, state = fixture()
        manifest, tensors = self.export(worker, state, "cold_migrate")
        restored = self.decode(empty_target(), manifest, tensors)
        self.assertTrue(restored.needs_anchor)
        self.assertFalse(state.needs_anchor)
        self.assertFalse(restored.contexts["blocks"].buffers)
        self.assertEqual(restored.contexts["blocks"].continuous_cached_steps, 1)
        self.assertEqual(restored.contexts["blocks"].cfg_residual_diffs, state.contexts["blocks"].cfg_residual_diffs)
        self.assertGreater(manifest["omitted_feature_bytes"], 0)

    def test_unknown_fields_views_shared_roles_and_broken_config_alias_rejected(self):
        for mutation in (lambda state: setattr(state.contexts["blocks"], "unknown", 1),
                         lambda state: state.contexts["blocks"].buffers.update(None_Fn_residual_buffer=state.latents),
                         lambda state: state.contexts["blocks"].buffers.update(
                             None_Bn_residual_buffer=state.contexts["blocks"].buffers["None_Fn_residual_buffer"][:4]),
                         lambda state: state.contexts["blocks"]._init_kwargs.update(cache_config=copy.deepcopy(state.contexts["blocks"].cache_config))):
            worker, state = fixture()
            mutation(state)
            with self.assertRaises((ParallelContractError, ValueError, TypeError)):
                self.export(worker, state)

    def test_payload_corruption_and_destination_binding_rejected(self):
        worker, state = fixture()
        manifest, tensors = self.export(worker, state)
        tensors["cache/0"][0] += 1
        with self.assertRaisesRegex(ParallelContractError, "checksum"):
            self.decode(empty_target(), manifest, tensors)
        with self.assertRaisesRegex(ParallelContractError, "destination"):
            unpack_request(empty_target(), manifest, tensors, target_device="wrong", migration_id="migration-1")

    def test_target_failure_restores_source_and_never_registers_target(self):
        worker, state = fixture()
        source, target = MigrationEndpoint(worker, "GPU-source"), MigrationEndpoint(empty_target(), "GPU-target")
        with patch.object(target, "prepare", side_effect=MemoryError("target allocation failed")):
            with self.assertRaises(MemoryError):
                migrate_request(source, target, state, action="full_migrate", migration_id="oom")
        worker._check(state, "gpu")
        self.assertFalse(target.worker._states)
        advance(state)
        self.assertEqual(state.step_index, 17)

    def test_ownership_ack_release_commit_prevents_double_execution_and_replay(self):
        worker, state = fixture()
        source, target = MigrationEndpoint(worker, "GPU-source"), MigrationEndpoint(empty_target(), "GPU-target")
        manifest, tensors, _ = source.pause_export(state, action="full_migrate", target_device="GPU-target", migration_id="tx")
        with self.assertRaisesRegex(ValueError, "reside"):
            worker._check(state, "gpu")
        ack = target.prepare(manifest, tensors)
        self.assertFalse(target.worker._states)
        with self.assertRaises(ParallelContractError):
            source.release_source("tx")
        with self.assertRaises(ParallelContractError):
            target.commit("tx", ack)
        source.acknowledge("tx", ack)
        receipt = source.release_source("tx")
        self.assertFalse(worker._states)
        restored = target.commit("tx", receipt)
        target.worker._check(restored, "gpu")
        with self.assertRaises(ParallelContractError):
            source.abort("tx")
        with self.assertRaises(ParallelContractError):
            target.prepare(manifest, tensors)
        self.assertEqual(target._epochs[restored.request_id], 1)

    def test_cache_off_base_path_and_full_cached_transaction(self):
        for cached in (False, True):
            worker, state = fixture(cached=cached)
            original = state.latents.clone()
            source, target = MigrationEndpoint(worker, "GPU-source"), MigrationEndpoint(empty_target(), "GPU-target")
            restored, manifest, costs = migrate_request(source, target, state, action="full_migrate", migration_id="done")
            self.assertTrue(self.torch.equal(original, restored.latents))
            self.assertEqual(restored.cache_enabled, cached)
            self.assertEqual(state.residency, "released")
            self.assertGreater(costs["transfer_bytes"], 0)
            self.assertGreaterEqual(costs["transaction_ms"], costs["export_ms"])

    def test_rpc_coordinator_uses_release_receipt_and_records_actual_checkpoint_hash(self):
        from ditserve.research.dual_runtime import transfer

        class LocalRPC:
            def __init__(self, endpoint):
                self.endpoint, self.device = endpoint, endpoint.device_id

            def call(self, command, **args):
                if command == "pause_export":
                    state = self.endpoint.worker._states[args.pop("request_id")]
                    return self.endpoint.pause_export(state, **args)
                method = "prepare" if command == "migration_prepare" else command
                return getattr(self.endpoint, method)(**args)

        worker, state = fixture()
        source = LocalRPC(MigrationEndpoint(worker, "GPU-source"))
        target = LocalRPC(MigrationEndpoint(empty_target(), "GPU-target"))
        evidence = transfer(source, target, state.request_id, "full_migrate", "rpc")
        self.assertEqual(len(evidence["state_sha256"]), 64)
        self.assertEqual(evidence["ownership_epoch"], 1)
        self.assertTrue(evidence["byte_exact_before_first_compute"])
        self.assertFalse(source.endpoint.worker._states)
        target.endpoint.worker._check(target.endpoint.worker._states[state.request_id], "gpu")

    def test_rpc_target_prepare_failure_preserves_original_exception_and_resumes_source(self):
        from ditserve.research.dual_runtime import transfer

        class LocalRPC:
            def __init__(self, endpoint):
                self.endpoint, self.device = endpoint, endpoint.device_id

            def call(self, command, **args):
                if command == "pause_export":
                    state = self.endpoint.worker._states[args.pop("request_id")]
                    return self.endpoint.pause_export(state, **args)
                if command == "migration_prepare":
                    raise MemoryError("real target OOM")
                return getattr(self.endpoint, command)(**args)

        worker, state = fixture()
        source = LocalRPC(MigrationEndpoint(worker, "GPU-source"))
        target = LocalRPC(MigrationEndpoint(empty_target(), "GPU-target"))
        with self.assertRaisesRegex(MemoryError, "real target OOM"):
            transfer(source, target, state.request_id, "full_migrate", "rpc-oom")
        worker._check(state, "gpu")

    def test_local_roundtrip_has_one_owner_and_exact_sampler_suffix(self):
        worker, original = fixture()
        baseline = copy.deepcopy(original)
        restored, evidence = local_roundtrip(worker, original, action="full_migrate", migration_id="local")
        self.assertEqual(original.residency, "released")
        self.assertEqual(list(worker._states), [restored.request_id])
        worker._check(restored, "gpu")
        self.assertTrue(evidence["byte_exact_before_first_compute"])
        for _ in range(16):
            advance(baseline)
            advance(restored)
            self.assertTrue(self.torch.equal(baseline.latents, restored.latents))

    def test_local_target_validation_failure_preserves_source_owner(self):
        worker, state = fixture()
        with patch("ditserve.research.dual_state.unpack_request", side_effect=MemoryError("allocation failed")):
            with self.assertRaises(MemoryError):
                local_roundtrip(worker, state, action="full_migrate", migration_id="local-fail")
        worker._check(state, "gpu")

    def wan_fn_residual(self, sequence=11):
        # Follow Wan's transpose and cache-dit's allocating subtraction. The
        # resulting residual owns dense storage; it is not a live tensor view.
        original = self.torch.arange(sequence * 1536, dtype=self.torch.float16).reshape(1, 1536, sequence).transpose(1, 2)
        residual = original - self.torch.ones_like(original)
        self.assertEqual(residual.stride(), (sequence * 1536, 1, sequence))
        self.assertIsNone(residual._base)
        return residual

    def test_observed_wan_fn_layout_preserves_exact_strides_storage_bytes_and_alias_identity(self):
        from ditserve.research.dual_state import _raw_storage, _storage_hash
        worker, state = fixture()
        residual = self.wan_fn_residual()
        state.contexts["blocks"].buffers.update(None_Fn_residual_buffer=residual, None_Fn_residual_buffer_cfg=residual)
        manifest, tensors = self.export(worker, state)
        references = manifest["contexts"]["blocks"]["buffer_references"]
        name = references["None_Fn_residual_buffer"]
        self.assertEqual(name, references["None_Fn_residual_buffer_cfg"])
        record = manifest["features"][name]
        self.assertEqual(record["stride"], list(residual.stride()))
        self.assertEqual(record["storage_sha256"], _storage_hash(residual))
        self.assertEqual(tensors[name].stride(), residual.stride())
        restored = self.decode(empty_target(), manifest, tensors)
        result = restored.contexts["blocks"].buffers["None_Fn_residual_buffer"]
        self.assertEqual(result.stride(), residual.stride())
        self.assertTrue(self.torch.equal(result, residual))
        self.assertTrue(self.torch.equal(_raw_storage(result), _raw_storage(residual)))
        self.assertIs(result, restored.contexts["blocks"].buffers["None_Fn_residual_buffer_cfg"])
        self.assertNotEqual(result.untyped_storage().data_ptr(), residual.untyped_storage().data_ptr())

    def test_observed_layout_does_not_admit_unknown_transposes_views_or_overlapping_storage(self):
        for kind in ("wrong_channels", "wrong_buffer", "live_view", "overlap", "shared_distinct_tensor"):
            with self.subTest(kind=kind):
                worker, state = fixture()
                key = "None_Fn_residual_buffer"
                value = self.wan_fn_residual()
                if kind == "wrong_channels":
                    value = self.torch.empty_strided((1, 11, 32), (352, 1, 11), dtype=self.torch.float16)
                elif kind == "wrong_buffer":
                    key = "None_Bn_residual_buffer"
                elif kind == "live_view":
                    value = value.view_as(value)
                elif kind == "overlap":
                    value = self.torch.empty(11 * 1536, dtype=self.torch.float16).as_strided((1, 11, 1536), (11 * 1536, 1, 1))
                elif kind == "shared_distinct_tensor":
                    state.contexts["blocks"].buffers["None_Fn_residual_buffer_cfg"] = value.detach()
                state.contexts["blocks"].buffers[key] = value
                with self.assertRaises((ParallelContractError, ValueError)):
                    self.export(worker, state)

    def test_wire_rejects_changed_feature_strides_even_if_logical_tensor_values_match(self):
        worker, state = fixture()
        state.contexts["blocks"].buffers["None_Fn_residual_buffer"] = self.wan_fn_residual()
        manifest, tensors = self.export(worker, state)
        name = manifest["contexts"]["blocks"]["buffer_references"]["None_Fn_residual_buffer"]
        tensors[name] = tensors[name].contiguous()
        with self.assertRaisesRegex(ParallelContractError, "checksum/schema"):
            self.decode(empty_target(), manifest, tensors)

    def test_native_context_object_ids_translate_explicitly_for_full_cold_and_omit(self):
        for action in ("full_migrate", "cold_migrate", "certified_omit"):
            with self.subTest(action=action):
                worker, state = fixture(context_name="blocks_111")
                source, target = MigrationEndpoint(worker, "GPU-source"), MigrationEndpoint(empty_target("blocks_222"), "GPU-target")
                certificate = refresh_certificate(state) if action == "certified_omit" else None
                manifest, tensors, _ = source.pause_export(state, action=action, target_device="GPU-target",
                    migration_id="named", certificate=certificate)
                ack = target.prepare(manifest, tensors)
                self.assertEqual(ack["context_name_mapping"], {"blocks_111": "blocks_222"})
                source.acknowledge("named", ack)
                receipt = source.release_source("named")
                self.assertEqual(receipt["context_name_mapping"], ack["context_name_mapping"])
                result = target.commit("named", receipt)
                self.assertEqual(set(result.contexts), {"blocks_222"})
                context = result.contexts["blocks_222"]
                self.assertEqual(context.name, "blocks_222")
                self.assertEqual(context._init_kwargs["name"], "blocks_222")
                self.assertIs(context._init_kwargs["cache_config"], context.cache_config)
                self.assertEqual(context.executed_steps, 16)
                self.assertEqual(context.transformer_executed_steps, 32)
                self.assertEqual(context.cfg_residual_diffs["15"], .02)
                self.assertEqual(bool(context.buffers), action == "full_migrate")
                self.assertEqual(result.needs_anchor, action == "cold_migrate")

    def test_context_translation_does_not_rename_arbitrary_or_multiple_contexts(self):
        worker, state = fixture(context_name="other_111")
        manifest, tensors = self.export(worker, state)
        with self.assertRaisesRegex(ParallelContractError, "object-ID"):
            self.decode(empty_target("blocks_222"), manifest, tensors)

    def test_target_release_receipt_binds_exact_acknowledged_context_mapping(self):
        worker, state = fixture(context_name="blocks_111")
        source, target = MigrationEndpoint(worker, "GPU-source"), MigrationEndpoint(empty_target("blocks_222"), "GPU-target")
        manifest, tensors, _ = source.pause_export(state, action="full_migrate", target_device="GPU-target", migration_id="named")
        ack = target.prepare(manifest, tensors)
        source.acknowledge("named", ack)
        receipt = source.release_source("named")
        bad = copy.deepcopy(receipt)
        bad["context_name_mapping"]["blocks_111"] = "blocks_333"
        with self.assertRaises(ParallelContractError):
            target.commit("named", bad)
        self.assertFalse(target.worker._states)
        target.commit("named", receipt)


if __name__ == "__main__":
    unittest.main()
