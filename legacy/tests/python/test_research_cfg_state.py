"""Policy identity is part of the full/cold transactional wire state."""

import copy
import importlib.util
import unittest
from unittest.mock import patch

from ditserve.research.common import read_json
from ditserve.research.cfg_policy import resolve_policy
from ditserve.research.cfg_protocol import DEFAULT_CONFIG
from ditserve.research.dual_state import (
    MigrationEndpoint, ParallelContractError, SCHEMA, local_roundtrip,
    migrate_request, pack_request, unpack_request)
from test_research_dual_state import fixture, empty_target, advance


def bind(worker, state=None, identity="independent_008", implementation=None):
    worker._cfg_research_config = read_json(DEFAULT_CONFIG)
    worker._cfg_policies = {}
    if state is not None:
        value = resolve_policy(worker._cfg_research_config, identity, implementation)
        state.spec.update({key: value[key] for key in ("cache", "threshold", "max_continuous")})
        state.cache_enabled = value["cache"]
        worker._cfg_policies[state.request_id] = value
    return worker


class CfgStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ("torch", "diffusers", "cache_dit")):
            raise unittest.SkipTest("pinned CPU dependencies required")
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def export(self, worker, state, action="full_migrate"):
        return pack_request(worker, state, action=action, target_device="GPU-target", migration_id="cfg-test")

    def decode(self, target, manifest, tensors):
        return unpack_request(target, manifest, tensors, target_device="GPU-target", migration_id="cfg-test")

    def test_full_and_cold_roundtrip_bind_explicit_strategy_and_implementation(self):
        for identity in ("independent_004", "independent_008", "paired_008"):
            for action in ("full_migrate", "cold_migrate"):
                with self.subTest(identity=identity, action=action):
                    worker, state = fixture()
                    bind(worker, state, identity)
                    context = state.contexts["blocks"]
                    context.buffers["None_Bn_residual_encoder_buffer"] = context.buffers["None_Fn_residual_buffer"]
                    expected = copy.deepcopy(worker._cfg_policies[state.request_id])
                    manifest, tensors = self.export(worker, state, action)
                    self.assertEqual(manifest["schema"], SCHEMA + ".cfg1")
                    self.assertEqual(manifest["research_policy"], expected)
                    target = bind(empty_target())
                    restored = self.decode(target, manifest, tensors)
                    self.assertEqual(target._cfg_policies[state.request_id], expected)
                    self.assertTrue(self.torch.equal(state.latents, restored.latents))
                    self.assertTrue(self.torch.equal(state.generator.get_state(), restored.generator.get_state()))
                    self.assertEqual(restored.spec["threshold"], expected["threshold"])
                    self.assertEqual(restored.needs_anchor, action == "cold_migrate")
                    restored_context = restored.contexts["blocks"]
                    self.assertEqual(restored_context.cached_steps, context.cached_steps)
                    self.assertIs(restored.scheduler.disable_corrector, restored.scheduler.config.disable_corrector)
                    if action == "cold_migrate":
                        self.assertFalse(restored_context.buffers)
                    else:
                        self.assertIs(restored_context.buffers["None_Bn_residual_encoder_buffer"],
                                      restored_context.buffers["None_Fn_residual_buffer"])
                    self.assertNotIn(state.request_id, target._states)
                    self.assertEqual(state.residency, "gpu")

    def test_same_device_roundtrip_preserves_binding_and_complete_sampler(self):
        worker, state = fixture()
        bind(worker, state, "paired_008")
        expected_policy = copy.deepcopy(worker._cfg_policies[state.request_id])
        old_latent = state.latents.clone()
        restored, evidence = local_roundtrip(worker, state, action="full_migrate", migration_id="cfg-local")
        self.assertEqual(state.residency, "released")
        self.assertIs(worker._states[restored.request_id], restored)
        self.assertEqual(worker._cfg_policies[restored.request_id], expected_policy)
        self.assertTrue(self.torch.equal(old_latent, restored.latents))
        self.assertTrue(evidence["byte_exact_before_first_compute"])
        advance(restored)
        self.assertEqual(restored.step_index, 17)

    def test_tampered_identity_and_unknown_implementation_rejected(self):
        worker, state = fixture()
        bind(worker, state)
        manifest, tensors = self.export(worker, state)
        modified = copy.deepcopy(manifest)
        modified["research_policy"]["id"] = "independent_004"
        target = bind(empty_target())
        with self.assertRaisesRegex(ParallelContractError, "checksum"):
            self.decode(target, modified, tensors)
        self.assertFalse(target._cfg_policies)
        worker._cfg_policies[state.request_id]["implementation"] = "unknown"
        with self.assertRaises((ValueError, ParallelContractError)):
            self.export(worker, state)
        self.assertEqual(state.residency, "gpu")
        self.assertIs(worker._states[state.request_id], state)

    def test_policy_hash_and_spec_mismatch_rejected_before_export(self):
        for change in ("sha256", "threshold"):
            with self.subTest(change=change):
                worker, state = fixture()
                bind(worker, state)
                if change == "sha256":
                    worker._cfg_policies[state.request_id][change] = "0" * 64
                else:
                    state.spec[change] = .04
                with self.assertRaisesRegex(ParallelContractError, "policy does not match"):
                    self.export(worker, state)

    def test_bound_and_legacy_targets_cannot_silently_mix_protocols(self):
        source, state = fixture()
        bind(source, state)
        manifest, tensors = self.export(source, state)
        with self.assertRaisesRegex(ParallelContractError, "does not support CFG"):
            self.decode(empty_target(), manifest, tensors)
        legacy, old = fixture()
        old_manifest, old_tensors = self.export(legacy, old)
        target = bind(empty_target())
        with self.assertRaises(ParallelContractError):
            self.decode(target, old_manifest, old_tensors)

    def test_target_failure_keeps_source_and_policy_live(self):
        worker, state = fixture()
        bind(worker, state, "paired_008")
        target_worker = bind(empty_target())
        source, target = MigrationEndpoint(worker, "GPU-source"), MigrationEndpoint(target_worker, "GPU-target")
        original = copy.deepcopy(worker._cfg_policies[state.request_id])
        latent = state.latents.clone()
        with patch.object(target_worker, "_contexts", side_effect=RuntimeError("allocation failed")):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                migrate_request(source, target, state, action="full_migrate", migration_id="cfg-failure")
        self.assertEqual(state.residency, "gpu")
        self.assertTrue(self.torch.equal(latent, state.latents))
        self.assertEqual(worker._cfg_policies[state.request_id], original)
        self.assertFalse(target_worker._cfg_policies)
        self.assertFalse(target_worker._states)

    def test_ownership_transaction_removes_only_released_source_binding(self):
        worker, state = fixture()
        bind(worker, state, "independent_004")
        target_worker = bind(empty_target())
        source, target = MigrationEndpoint(worker, "GPU-source"), MigrationEndpoint(target_worker, "GPU-target")
        manifest, tensors, _ = source.pause_export(state, action="full_migrate", target_device="GPU-target",
                                                   migration_id="cfg-owner")
        ack = target.prepare(manifest, tensors)
        with self.assertRaises(ValueError):
            worker._check(state, "gpu")
        self.assertNotIn(state.request_id, target_worker._states)
        with self.assertRaises(ParallelContractError):
            target.commit("cfg-owner", ack)
        source.acknowledge("cfg-owner", ack)
        receipt = source.release_source("cfg-owner")
        self.assertNotIn(state.request_id, worker._cfg_policies)
        self.assertNotIn(state.request_id, worker._states)
        restored = target.commit("cfg-owner", receipt)
        self.assertEqual(target_worker._cfg_policies[restored.request_id]["id"], "independent_004")
        self.assertIs(target_worker._states[restored.request_id], restored)

    def test_abort_prepared_target_removes_its_policy_and_resumes_source(self):
        worker, state = fixture()
        bind(worker, state)
        target_worker = bind(empty_target())
        source, target = MigrationEndpoint(worker, "GPU-source"), MigrationEndpoint(target_worker, "GPU-target")
        manifest, tensors, _ = source.pause_export(state, action="full_migrate", target_device="GPU-target",
                                                   migration_id="cfg-abort")
        target.prepare(manifest, tensors)
        self.assertIn(state.request_id, target_worker._cfg_policies)
        target.abort("cfg-abort")
        source.abort("cfg-abort")
        self.assertFalse(target_worker._cfg_policies)
        self.assertIn(state.request_id, worker._cfg_policies)
        self.assertEqual(state.residency, "gpu")


if __name__ == "__main__":
    unittest.main()
