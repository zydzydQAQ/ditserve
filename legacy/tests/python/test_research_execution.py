"""Real adapter/transfer contracts inside a CPU-only fake denoising loop."""

import dataclasses
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ditserve.gpu.worker import WanStepWorker, normalize_spec
from ditserve.research.adapter import ResearchWanAdapter
from ditserve.research.budget import BudgetStop
from ditserve.research.common import atomic_json, file_digest, read_json
from ditserve.research.experiment import Experiment, IneligibleAction, path_signature
from ditserve.research.protocol import DEFAULT_CONFIG
from test_research_state import make_state


class CpuWanWorker(WanStepWorker):
    """Deterministic fake math; real request schema and checkpoint transfers."""

    def __init__(self, torch):
        self.torch, self.device, self.attention = torch, "cpu", "flash"
        self._owner, self._states, self._lock = object(), {}, threading.RLock()
        self.released = []

    def _sync(self):
        pass

    def _detach_contexts(self):
        pass

    def _memory(self):
        return dict(allocated_bytes=sum(t.numel() * t.element_size()
                    for s in self._states.values() for c in s.contexts.values() for t in c.buffers.values()),
                    reserved_bytes=4096, driver_free_bytes=4096, device_total_bytes=8192)

    def prepare(self, spec):
        state = make_state(self.torch)
        state.spec = normalize_spec(spec)
        state.request_id, state.owner = spec["request_id"], self._owner
        state.immutable_spec = {k: v for k, v in state.spec.items() if k not in ("cache", "threshold", "max_continuous")}
        state.step_index, state.scheduler._step_index = 0, None
        state.generator.manual_seed(spec["seed"])
        state.latents.zero_()
        state.cache_enabled = spec["cache"]
        if state.cache_enabled:
            original = state.contexts["blocks"]
            Context = dataclasses.make_dataclass("Context", [(k, object) for k in vars(original)])
            context = Context(**vars(original))
            context.buffers = {}
            context.cached_steps, context.cfg_cached_steps = [], []
            context.residual_diffs, context.cfg_residual_diffs = {}, {}
            for key in ("executed_steps", "accumulated_executed_steps", "transformer_executed_steps",
                        "accumulated_transformer_executed_steps", "continuous_cached_steps",
                        "cfg_continuous_cached_steps", "accumulated_cached_steps", "cfg_accumulated_cached_steps"):
                setattr(context, key, 0)
            context.cache_config.residual_diff_threshold = spec["threshold"]
            state.contexts = {"blocks": context}
        else:
            state.contexts = {}
        state.prepare_ms = 2.
        self._states[state.request_id] = state
        return state

    def step(self, state):
        self._check(state, "gpu")
        self._update_policy(state)
        state.step_index += 1
        index = state.step_index - 1
        state.latents = state.latents + self.torch.rand((), generator=state.generator)
        state.scheduler._step_index = state.step_index
        state.scheduler.last_sample = state.latents
        state.scheduler.model_outputs = [state.latents.clone()]
        reuse = bool(state.cache_enabled and state.step_index >= 10 and state.step_index % 2 == 0
                     and not state.needs_anchor)
        for context in state.contexts.values():
            context.executed_steps = context.accumulated_executed_steps = state.step_index
            context.transformer_executed_steps = context.accumulated_transformer_executed_steps = state.step_index * 2
            context.cache_config.residual_diff_threshold = state.spec["threshold"]
            context.continuous_cached_steps = context.cfg_continuous_cached_steps = int(reuse)
            if reuse:
                context.cached_steps.append(index)
                context.cfg_cached_steps.append(index)
                context.accumulated_cached_steps += 1
                context.cfg_accumulated_cached_steps += 1
            else:
                context.buffers = {"None_" + name + "_residual_buffer" + suffix: state.latents.clone()
                                   for name in ("Fn", "Bn") for suffix in ("", "_cfg")}
        state.needs_anchor = False
        record = dict(kind="step", step_index=index, completed_steps=state.step_index, ms=1.,
                      path="cached" if reuse else "full", reused_forwards=2 if reuse else 0)
        state.events.append(record)
        return record

    def finish(self, state):
        self._check(state, "gpu")
        state.finished = True
        state.events.append(dict(kind="decode", ms=3.))
        return state.latents.detach().numpy().copy().reshape(1, 1, 2, 2, 1)

    def release(self, state):
        self.released.append(state.request_id)
        return super().release(state)


class ResearchExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU torch required")
        cls.torch = torch

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_json(DEFAULT_CONFIG)
        frozen = dict(config=self.config, sha256="protocol", source_sha256="source")
        atomic_json(self.root / "protocol.json", frozen)
        self.exp = Experiment(dict(output=str(self.root), stage="correctness", deadline_monotonic=time.monotonic() + 300,
                                   physical_devices=[dict(uuid="GPU-FAKE", name="CPU contract")]))
        self.worker = CpuWanWorker(self.torch)
        with patch("ditserve.research.adapter.importlib.metadata.version", return_value="1.5.0"):
            self.adapter = ResearchWanAdapter(worker=self.worker, validation_only=True)
        self.exp.worker, self.exp.adapter = self.worker, self.adapter

    def spec(self, seed=101, cache=True):
        recipe = self.config["recipes"][2 if cache else 0]
        return self.exp.spec(self.config["prompts"][0], seed,
                             dict(height=32, width=32, frames=5, steps=32), recipe)

    def test_keep_hot_exact_preserve_whole_trajectory_with_interleaved_request(self):
        baseline = self.exp.generate(self.spec(), instrument=True)
        for action in ("keep", "hot", "exact_refresh"):
            with self.subTest(action=action):
                candidate = self.exp.generate(self.spec(), action=action, instrument=True,
                                               urgent_spec=self.spec(seed=102, cache=False))
                self.assertEqual(candidate["traces"], baseline["traces"])
                self.assertEqual(path_signature(candidate["events"]), path_signature(baseline["events"]))
                self.assertTrue((candidate["output"] == baseline["output"]).all())
                self.assertEqual(candidate["action_step"], 16)
                self.assertEqual(candidate["urgent"]["status"], "completed")
                self.assertEqual(candidate["output_sha256"], file_digest(self.root / candidate["output_path"]))
                self.assertEqual(self.worker._states, {})

    def test_fixed_exact_boundary_is_inspected_once_not_after_action(self):
        original = self.adapter.certificate
        inspected = []
        def certificate(state):
            inspected.append(state.step_index)
            return original(state)
        self.adapter.certificate = certificate
        self.exp.generate(self.spec(), action="exact_refresh")
        self.assertEqual(inspected, [16])

    def test_critical_timing_includes_certificate_and_excludes_urgent_artifact_writes(self):
        import numpy as np
        clock = [0.]
        prepare, step, finish = self.adapter.prepare, self.adapter.step, self.adapter.finish
        certificate, save = self.adapter.certificate, np.save
        def charged(function, seconds):
            def call(*args, **kwargs):
                clock[0] += seconds
                return function(*args, **kwargs)
            return call
        self.adapter.prepare = charged(prepare, 2.)
        self.adapter.step = charged(step, 1.)
        self.adapter.finish = charged(finish, 3.)
        self.adapter.certificate = charged(certificate, 5.)
        with patch("ditserve.research.experiment.time.perf_counter", side_effect=lambda: clock[0]), patch(
                "numpy.save", side_effect=charged(save, 7.)):
            result = self.exp.generate(self.spec(), action="exact_refresh",
                                       urgent_spec=self.spec(seed=102, cache=False))
        # 2+32+3 for urgent execution, 16+3 resumed suffix, 5 certificate.
        self.assertEqual(result["critical_path_ms"], 61000.)
        self.assertEqual(result["elapsed_ms"], 79000.)
        self.assertEqual(result["artifact_io_ms"], 14000.)
        self.assertEqual(result["urgent"]["artifact_io_ms"], 14000.)
        cert = next(r for r in result["action_records"] if r["kind"] == "refresh_certificate")
        self.assertEqual(cert["ms"], 5000.)

    def test_step16_suffix_is_applied_once_to_full_trajectory(self):
        self.adapter.set_approximation = Mock(wraps=self.adapter.set_approximation)
        result = self.exp.generate(self.spec(), action="hot", suffix=self.config["recipes"][0])
        self.adapter.set_approximation.assert_called_once()
        changes = [e for e in result["events"] if e["kind"] == "research_suffix_change"]
        self.assertEqual(changes[0]["completed_steps"], 16)
        self.assertFalse(changes[0]["qualified"])
        self.assertEqual(result["phase_cost_ms"]["denoise"], 32.)
        self.assertEqual(self.worker._states, {})

    def test_deadline_and_ineligible_exact_release_state_without_success_artifacts(self):
        self.exp.deadline = time.monotonic() - 1
        with self.assertRaises(BudgetStop):
            self.exp.generate(self.spec())
        self.assertEqual(self.worker._states, {})
        self.exp.deadline = time.monotonic() + 300
        with self.assertRaises(IneligibleAction):
            self.exp.generate(self.spec(cache=False), action="exact_refresh")
        self.assertEqual(self.worker._states, {})
        self.assertFalse((self.root / "arrays").exists())

    def test_failed_urgent_request_also_releases_paused_request(self):
        original = self.worker.step
        def step(state):
            if state.spec["seed"] == 102:
                raise RuntimeError("urgent failure")
            return original(state)
        self.worker.step = step
        with self.assertRaisesRegex(RuntimeError, "urgent failure"):
            self.exp.generate(self.spec(), action="hot", urgent_spec=self.spec(seed=102, cache=False))
        self.assertEqual(self.worker._states, {})
        self.assertEqual(len(self.worker.released), 2)


if __name__ == "__main__":
    unittest.main()
