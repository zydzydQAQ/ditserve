"""CPU contracts for cached heterogeneous migration, independent of model weights."""

import copy
import importlib.util
import json
import threading
import tempfile
from pathlib import Path
import unittest
from collections import defaultdict
from unittest.mock import patch
from types import SimpleNamespace

from ditserve.models.wan.migration import (MigrationEndpoint, ParallelContractError, migrate_request,
                                          pack_request, unpack_request, local_roundtrip)
from ditserve.models.wan.state import refresh_certificate


class CertificateIdentityTests(unittest.TestCase):
    def test_shared_digest_preserves_frozen_unicode_identity_and_rejects_nonfinite(self):
        from ditserve.models.base import digest
        from ditserve.models.wan.migration import _hash as migration_digest
        from ditserve.models.wan.state import _hash as certificate_digest
        value = dict(name="实验", threshold=.08, scopes=["cfg", None])
        for encode in (digest, migration_digest, certificate_digest):
            self.assertEqual(encode(value), "6b7b8cc41a0f89326c7a93066d80ec00cc4020094622f5f529720783bdd42812")
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.assertRaises(ValueError):
                    encode(dict(threshold=invalid))


def fixture(*, cached=True, boundary=16, context_name="blocks"):
    import torch
    import cache_dit
    from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
    from diffusers import UniPCMultistepScheduler
    from ditserve.models.wan.runtime import WanStepWorker, RequestState, normalize_spec
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

    def test_worker_file_transport_preserves_two_phase_ownership(self):
        from ditserve.models.base import Backend
        from ditserve.models.wan.backend import WanBackend
        def adapter(runtime, identity):
            result = WanBackend.__new__(WanBackend)
            Backend.__init__(result, {})
            result.runtime = runtime
            result.endpoint = MigrationEndpoint(runtime, identity)
            result.continuations, result.policies, result.pending = {}, {}, {}
            result.parallel = result.decode_state = None
            return result
        runtime, state = fixture()
        source, target = adapter(runtime, "GPU-source"), adapter(empty_target(), "GPU-target")
        source.states["source"] = state
        with tempfile.TemporaryDirectory() as directory:
            snapshot = source.snapshot(dict(state_id="source", mode="hot", target_device="GPU-target",
                migration_id="file1", path=str(Path(directory) / "state.pt")))
            prepared = target.restore(dict(state_id="target", path=snapshot["path"], sha256=snapshot["sha256"]))
            self.assertFalse(prepared["runnable"])
            self.assertNotIn("target", target.states)
            receipt = source.migration_release(dict(migration_id="file1", acknowledgement=prepared["acknowledgement"]))
            committed = target.restore(dict(migration_id="file1", receipt=receipt["receipt"]))
            self.assertTrue(committed["runnable"])
            self.assertFalse(source.states)
            self.assertEqual(committed["progress"]["step"], 16)
            target.release(dict(state_id="target"))

    def test_worker_outputs_preserve_identity_and_optional_latents_for_offline_evaluation(self):
        import numpy as np
        from ditserve.models.base import Backend
        from ditserve.models.wan.backend import WanBackend
        runtime, state = fixture(cached=False, boundary=32)
        runtime.attention, runtime.model_path = "math", Path("/fixed-local-snapshot")
        runtime.vae_dtype = "float32"
        pixels = np.zeros((1, 5, 32, 32, 3), dtype=np.float32)
        path = dict(step_index=31, cache_contexts=[], reused_forwards=0, path_scope="DBCache", path="compute")
        state.events.append(dict(path, kind="step", ms=3.))
        def finish(unused):
            state.finished = True
            state.events.append(dict(kind="decode", ms=1.))
            return pixels
        runtime.finish = finish
        runtime.reference = lambda spec: dict(output=pixels, latents=state.latents, path_history=[path])
        with tempfile.TemporaryDirectory() as directory:
            adapter = WanBackend.__new__(WanBackend)
            Backend.__init__(adapter, dict(work_dir=directory))
            adapter.runtime, adapter.options = runtime, {}
            adapter.states["result"] = state
            adapter.continuations = {}
            adapter.parallel = adapter.decode_state = adapter.decode_ticket = None
            plain = adapter.finish(dict(id=1, state_id="result"))
            self.assertNotIn("latent", plain)
            self.assertEqual(plain["path_history"], [path])
            self.assertEqual(plain["generation"]["prompt"], state.spec["prompt"])
            self.assertFalse(plain["generation"]["cache_enabled"])
            reference = adapter.reference(dict(id=2, spec=state.spec, save_latents=True))
            self.assertTrue(np.array_equal(np.load(reference["latent"]["path"]), state.latents.numpy()))
            adapter.decoder = SimpleNamespace(collect=lambda **unused: (pixels, dict(kind="decode", ms=2.)))
            adapter.decode_state, adapter.decode_ticket = "result", "pending"
            adapter.decode_request = dict(id=3, state_id="result", save_latents=True)
            asynchronous = adapter.decode_poll(dict(id=4, ticket="pending", state_id="result"))
            self.assertTrue(asynchronous["ready"])
            self.assertTrue(Path(asynchronous["latent"]["path"]).is_file())
            self.assertEqual(reference["generation"], asynchronous["generation"])
            json.dumps([plain, reference, asynchronous], allow_nan=False)

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


    def wan_fn_residual(self, sequence=11):
        # Follow Wan's transpose and cache-dit's allocating subtraction. The
        # resulting residual owns dense storage; it is not a live tensor view.
        original = self.torch.arange(sequence * 1536, dtype=self.torch.float16).reshape(1, 1536, sequence).transpose(1, 2)
        residual = original - self.torch.ones_like(original)
        self.assertEqual(residual.stride(), (sequence * 1536, 1, sequence))
        self.assertIsNone(residual._base)
        return residual

    def test_observed_wan_fn_layout_preserves_exact_strides_storage_bytes_and_alias_identity(self):
        from ditserve.models.wan.migration import _raw_storage, _storage_hash
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






if __name__ == "__main__":
    unittest.main()
