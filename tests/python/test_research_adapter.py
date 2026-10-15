"""Fake-worker research execution; no models or CUDA are accessed."""

import copy
import dataclasses
import threading
import unittest
from unittest.mock import Mock, patch

from ditserve.gpu.worker import WanStepWorker
from ditserve.research.adapter import ResearchWanAdapter
from test_research_state import make_state


class ResearchAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU torch required")
        cls.torch = torch

    def fixture(self, validation_only=True):
        state = make_state(self.torch)
        worker = WanStepWorker.__new__(WanStepWorker)
        worker.torch, worker.device, worker.attention = self.torch, "cpu", "flash"
        worker._owner = state.owner
        worker._states = {state.request_id: state}
        worker._lock = threading.RLock()
        worker._sync = Mock()
        worker._memory = Mock(side_effect=lambda: dict(allocated_bytes=sum(
            t.numel() * t.element_size() for c in state.contexts.values() for t in c.buffers.values()),
            reserved_bytes=4096, driver_free_bytes=4096, device_total_bytes=8192))
        worker.snapshot = Mock(return_value={"kind": "snapshot", "mode": "hot"})
        worker.restore = Mock(return_value={"kind": "restore"})
        worker.prepare = Mock(return_value=state)
        worker.encode_prompt = Mock(return_value="encoded")
        worker.prepare_encoded = Mock(return_value=state)
        worker.step = Mock(return_value={"kind": "step"})
        worker.finish = Mock(return_value="output")
        worker.release = Mock()
        with patch("ditserve.research.adapter.importlib.metadata.version", return_value="1.5.0"):
            adapter = ResearchWanAdapter(worker=worker, validation_only=validation_only)
        return state, worker, adapter

    def test_rejects_wrong_attention_and_dependency(self):
        state, worker, _ = self.fixture()
        worker.attention = "math"
        with self.assertRaisesRegex(ValueError, "Flash"):
            ResearchWanAdapter(worker=worker)
        worker.attention = "flash"
        with patch("ditserve.research.adapter.importlib.metadata.version", return_value="1.6.0"):
            with self.assertRaisesRegex(ValueError, "1.5.0"):
                ResearchWanAdapter(worker=worker)

    def test_exact_reclaim_requires_validation_flag(self):
        state, worker, adapter = self.fixture(validation_only=False)
        cert = adapter.certificate(state)
        with self.assertRaisesRegex(ValueError, "validation-only"):
            adapter.snapshot(state, "exact_refresh", certificate=cert)
        self.assertEqual(len(state.contexts["blocks"].buffers), 4)

    def test_exact_reclaim_never_uses_cold_or_changes_solver_rng_counters(self):
        state, worker, adapter = self.fixture()
        context = state.contexts["blocks"]
        scheduler = copy.deepcopy(vars(state.scheduler))
        rng = state.generator.get_state().clone()
        metadata = copy.deepcopy({k: v for k, v in vars(context).items() if k != "buffers"})
        latents = state.latents.clone()
        cert = adapter.certificate(state)
        event = adapter.snapshot(state, "exact_refresh", certificate=cert)
        self.assertEqual(context.buffers, {})
        self.assertFalse(state.needs_anchor)
        self.assertEqual(state.residency, "gpu")
        self.assertIsNone(state.pending_snapshot)
        self.assertEqual(metadata, {k: v for k, v in vars(context).items() if k != "buffers"})
        self.assertEqual(state.scheduler._step_index, scheduler["_step_index"])
        self.assertTrue(self.torch.equal(state.scheduler.model_outputs[0], scheduler["model_outputs"][0]))
        self.assertTrue(self.torch.equal(state.latents, latents))
        self.assertTrue(self.torch.equal(state.generator.get_state(), rng))
        self.assertEqual(event["transfer_bytes"], 0)
        self.assertEqual(event["allocated_delta_bytes"], 128)
        self.assertEqual(event["qualification"], "validation_only_not_GPU_qualified")
        worker.snapshot.assert_not_called()
        with self.assertRaises(ValueError):
            adapter.snapshot(state, "exact_refresh", certificate=cert)

    def test_mutation_between_certificate_and_action_fails_before_reclaim(self):
        state, worker, adapter = self.fixture()
        cert = adapter.certificate(state)
        state.generator.manual_seed(77)
        with self.assertRaisesRegex(ValueError, "stale"):
            adapter.snapshot(state, "exact_refresh", certificate=cert)
        self.assertEqual(len(state.contexts["blocks"].buffers), 4)
        worker.snapshot.assert_not_called()

    def test_transfer_rejects_distinct_view_alias_before_worker_call(self):
        for mode in ("hot", "cold"):
            state, worker, adapter = self.fixture()
            state.scheduler.last_sample = state.latents[:]
            with self.assertRaisesRegex(ValueError, "distinct tensor views"):
                adapter.snapshot(state, mode)
            worker.snapshot.assert_not_called()

    def test_real_worker_transfer_preserves_solver_rng_and_same_object_aliases_on_cpu(self):
        for mode in ("hot", "cold"):
            state, worker, adapter = self.fixture()
            original_context = state.contexts["blocks"]
            Context = dataclasses.make_dataclass("Context", [(k, object) for k in vars(original_context)])
            state.contexts["blocks"] = Context(**vars(original_context))
            state.scheduler.last_sample = state.latents
            latents, solver = state.latents.clone(), state.scheduler.model_outputs[0].clone()
            rng = state.generator.get_state().clone()
            worker.snapshot = WanStepWorker.snapshot.__get__(worker)
            worker.restore = WanStepWorker.restore.__get__(worker)
            with self.subTest(mode=mode):
                adapter.snapshot(state, mode)
                self.assertEqual(state.residency, "cpu")
                adapter.restore(state)
                self.assertEqual(state.residency, "gpu")
                self.assertIsNone(state.pending_snapshot)
                self.assertIs(state.scheduler.last_sample, state.latents)
                self.assertEqual(state.scheduler._step_index, 16)
                self.assertTrue(self.torch.equal(state.latents, latents))
                self.assertTrue(self.torch.equal(state.scheduler.model_outputs[0], solver))
                self.assertTrue(self.torch.equal(state.generator.get_state(), rng))
                self.assertEqual(state.needs_anchor, mode == "cold")
                context = state.contexts["blocks"]
                self.assertEqual(len(context.buffers), 4 if mode == "hot" else 0)
                self.assertEqual((context.continuous_cached_steps, context.cfg_continuous_cached_steps), (1, 1))

    def test_worker_compatible_delegation_and_m1_validation(self):
        state, worker, adapter = self.fixture()
        adapter.prepare(state.spec)
        adapter.encode_prompt(state.spec)
        adapter.prepare_encoded(state.spec, "encoded")
        adapter.step(state)
        self.assertEqual(adapter.finish(state), "output")
        adapter.snapshot(state, "hot")
        adapter.restore(state)
        worker.snapshot.assert_called_once_with(state, "hot")
        with self.assertRaisesRegex(ValueError, "max_continuous"):
            adapter.prepare(dict(state.spec, max_continuous=3))

    def test_approximation_is_single_step16_action_not_a_quality_guarantee(self):
        state, worker, adapter = self.fixture()
        event = adapter.set_approximation(state, threshold=.04)
        self.assertEqual(state.spec["threshold"], .04)
        self.assertFalse(event["qualified"])
        self.assertEqual(event["quality_scope"], "whole_prefix_and_suffix_requires_independent_calibration")
        with self.assertRaisesRegex(ValueError, "only one"):
            adapter.set_approximation(state, threshold=.08)
        state, worker, adapter = self.fixture()
        event = adapter.set_approximation(state, threshold=0., cache_enabled=False)
        self.assertFalse(state.cache_enabled)
        self.assertFalse(event["qualified"])
        self.assertEqual(state.contexts, {})

    def test_approximation_outside_fixed_action_set_is_rejected_without_mutation(self):
        for threshold, cache in ((.1, True), (.08, False), (0., True), (float("nan"), True)):
            state, worker, adapter = self.fixture()
            before = dict(state.spec)
            with self.assertRaises(ValueError):
                adapter.set_approximation(state, threshold=threshold, cache_enabled=cache)
            self.assertEqual(state.spec, before)
        state, worker, adapter = self.fixture()
        state.step_index = 15
        with self.assertRaisesRegex(ValueError, "step 16"):
            adapter.set_approximation(state, threshold=.04)

    def test_reshard_is_explicitly_unsupported(self):
        state, worker, adapter = self.fixture()
        with self.assertRaises(NotImplementedError):
            adapter.plan(state, "reshard", target_layout=dict(sp_size=2, ranks=[0, 1], device="cpu", version=1))


if __name__ == "__main__":
    unittest.main()
