"""Natural-admission orchestration contracts; these fixtures use no CUDA."""

import copy
import json
import tempfile
import time
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ditserve.research.adapter import ResearchWanAdapter
from ditserve.research.budget import BudgetStop
from ditserve.research.common import atomic_json, digest, read_json
from ditserve.research.evidence import trajectory_evidence
from ditserve.research.experiment import AdmissionCapacityError, Experiment
from ditserve.research.memory_experiment import run_memory_admission
from ditserve.research.protocol import DEFAULT_CONFIG
from test_research_execution import CpuWanWorker


class FakeOOM(RuntimeError):
    pass


class FailedFramePayload:
    pass


class FakeMemoryExperiment(Experiment):
    def __init__(self, root, outcomes):
        self.config = read_json(DEFAULT_CONFIG)
        self.frozen = dict(config=self.config, sha256="a" * 64, source_sha256="b" * 64)
        self.root, self.counter, self.records, self.calls = root, 0, [], []
        self.job = dict(stage="mechanism", physical_devices=[dict(name="CPU contract fixture")])
        self.worker = SimpleNamespace(torch=SimpleNamespace(OutOfMemoryError=FakeOOM))
        self.outcomes, self.checked, self.blocked = list(outcomes), 0, False
        self.failed_frames, self.quality_calls, self.bad_quality = [], [], None
        self.release_bytes = 600
        self.prompt_bank = None

    def check_unit(self, estimate=None):
        self.checked += 1
        if self.blocked:
            raise BudgetStop("complete unit cannot fit")

    def record(self, value):
        json.dumps(value, allow_nan=False)
        self.records.append(copy.deepcopy(value))
        return value

    def generate(self, spec, **kwargs):
        if not self.checked:
            raise AssertionError("GPU unit started before budget admission")
        if len(self.calls) == 3 and any(ref() is not None for ref in self.failed_frames):
            raise AssertionError("failed keep traceback retained payload into candidate run")
        self.calls.append(dict(spec=copy.deepcopy(spec), action=kwargs.get("action", "none"),
                               urgent_spec=copy.deepcopy(kwargs.get("urgent_spec"))))
        outcome = self.outcomes.pop(0)
        if outcome == "own_oom":
            raise FakeOOM("single-request capacity")
        if outcome == "budget":
            raise BudgetStop("deadline")
        admission = kwargs.get("admission")
        if admission is not None:
            memory = dict(measurement_kind="measured_gpu", allocated_bytes=1000,
                          reserved_bytes=2000, driver_free_bytes=4000)
            admission.update(before=memory, after=dict(memory, allocated_bytes=1000 - (
                0 if kwargs["action"] == "keep" else self.release_bytes)), transfer_complete=True,
                action=kwargs["action"], at_step=16, artificially_filled_memory=False, single_request_oom=False)
        if outcome in ("urgent_oom", "wrong_phase_oom", "artificial_oom"):
            payload = FailedFramePayload()
            self.failed_frames.append(weakref.ref(payload))
            admission.update(urgent_capacity_failed=True, phase="urgent")
            if outcome == "wrong_phase_oom":
                admission["phase"] = "main"
            if outcome == "artificial_oom":
                admission["artificially_filled_memory"] = True
            raise AdmissionCapacityError(dict(admission))
        def completed(value, who):
            return dict(status="completed", who=who, output="CPU test fixture",
                        output_sha256=digest(value), elapsed_ms=100., critical_path_ms=50.)
        run = completed(spec, "main")
        if kwargs.get("urgent_spec") is not None:
            run["urgent"] = completed(kwargs["urgent_spec"], "urgent")
            admission.update(main_completed=True, urgent_allocation_succeeded=True,
                             urgent_compute_completed=True, urgent_decode_completed=True)
            if outcome == "unfinished_main":
                admission["main_completed"] = False
            if outcome == "wrong_action":
                admission["action"] = "keep"
        return run

    def quality(self, run, reference, identity, reference_identity):
        self.quality_calls.append((identity, reference_identity))
        value = .2 if run["who"] == self.bad_quality else .01
        metrics = dict(status="ok", lpips_status="ok", finite=True, range_valid=True,
                       lpips_per_frame=[value] * identity["spec"]["frames"], lpips_mean=value,
                       lpips_p95=value, temporal_delta_rmse=.01, frame_count=identity["spec"]["frames"])
        return trajectory_evidence(identity, metrics, correctness="passed", completed_steps=32,
                                   reference_output_sha256=reference["output_sha256"],
                                   candidate_output_sha256=run["output_sha256"], reference_identity=reference_identity)


class ResearchMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_json(DEFAULT_CONFIG)
        self.prefix = self.config["recipes"][2]
        self.shape = dict(height=32, width=32, frames=5, steps=32)
        self.candidate = dict(candidate_id="hot-t004", prefix=self.prefix,
                              action=dict(kind="hot", at_step=16), suffix=self.config["recipes"][1])

    def test_real_keep_success_stops_without_trying_release_or_claiming_admission(self):
        exp = FakeMemoryExperiment(self.root, ["ok", "ok", "ok"])
        row = run_memory_admission(exp, self.prefix, self.shape, self.candidate)
        self.assertEqual(row["status"], "no_admission_difference")
        self.assertFalse(row["supported"])
        self.assertEqual([r["action"] for r in exp.calls], ["none", "none", "keep"])
        self.assertEqual(exp.quality_calls, [])
        self.assertEqual(exp.checked, 1)

    def test_keep_urgent_capacity_then_fresh_release_requires_both_quality_proofs(self):
        for action in ("hot", "cold", "exact_refresh"):
            exp = FakeMemoryExperiment(self.root, ["ok", "ok", "urgent_oom", "ok"])
            candidate = dict(self.candidate, action=dict(kind=action, at_step=16))
            with self.subTest(action=action):
                row = run_memory_admission(exp, self.prefix, self.shape, candidate)
                self.assertTrue(row["supported"], row)
                self.assertEqual(row["gate"]["accepted_trial_ids"], [row["trial_id"]])
                self.assertEqual(row["keep_request_identity"], row["admitted"]["identity"])
                self.assertEqual(len(exp.quality_calls), 2)
                for identity, reference in exp.quality_calls:
                    self.assertEqual(identity["input"], reference["input"])
                    self.assertFalse(reference["prefix"]["cache"])
                self.assertEqual(exp.calls[2]["spec"]["seed"], exp.calls[3]["spec"]["seed"])
                self.assertNotEqual(exp.calls[2]["spec"]["request_id"], exp.calls[3]["spec"]["request_id"])
                self.assertTrue(all(ref() is None for ref in exp.failed_frames))

    def test_single_request_oom_never_becomes_a_state_management_win(self):
        for outcomes, expected_calls in ((["own_oom"], 1), (["ok", "own_oom"], 2),
                                         (["ok", "ok", "own_oom"], 3)):
            exp = FakeMemoryExperiment(self.root, outcomes)
            row = run_memory_admission(exp, self.prefix, self.shape, self.candidate)
            self.assertFalse(row["supported"])
            self.assertTrue(row["single_request_oom"])
            self.assertEqual(len(exp.calls), expected_calls)

    def test_wrong_scope_or_artificial_pressure_does_not_admit_candidate_trial(self):
        for outcome in ("wrong_phase_oom", "artificial_oom"):
            exp = FakeMemoryExperiment(self.root, ["ok", "ok", outcome])
            row = run_memory_admission(exp, self.prefix, self.shape, self.candidate)
            self.assertFalse(row["supported"])
            self.assertEqual(len(exp.calls), 3)

    def test_release_must_observe_allocation_drop_and_qualify_both_completed_videos(self):
        for fault in ("main", "urgent", "no_release", "unfinished_main", "urgent_oom", "wrong_action"):
            exp = FakeMemoryExperiment(self.root, ["ok", "ok", "urgent_oom",
                                                   fault if fault in ("unfinished_main", "urgent_oom", "wrong_action") else "ok"])
            exp.bad_quality = fault
            if fault == "no_release":
                exp.release_bytes = 0
            row = run_memory_admission(exp, self.prefix, self.shape, self.candidate)
            with self.subTest(fault=fault):
                self.assertFalse(row["supported"])
                self.assertTrue(row["keep_rejected"])

    def test_budget_shortfall_starts_no_work_and_invalid_action_is_rejected(self):
        exp = FakeMemoryExperiment(self.root, [])
        exp.blocked = True
        row = run_memory_admission(exp, self.prefix, self.shape, self.candidate)
        self.assertFalse(row["supported"])
        self.assertEqual(exp.calls, [])
        with self.assertRaises(ValueError):
            run_memory_admission(exp, self.prefix, self.shape,
                                 dict(self.candidate, action=dict(kind="keep", at_step=16)))

    def test_actual_generate_wraps_only_urgent_oom_and_cleans_both_cpu_requests(self):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU torch required")
        frozen = dict(config=self.config, sha256="a" * 64, source_sha256="b" * 64)
        atomic_json(self.root / "protocol.json", frozen)
        exp = Experiment(dict(output=str(self.root), stage="mechanism", deadline_monotonic=time.monotonic() + 300,
                              physical_devices=[dict(uuid="GPU-FAKE", name="CPU contract")]))
        worker = CpuWanWorker(torch)
        with patch("ditserve.research.adapter.importlib.metadata.version", return_value="1.5.0"):
            exp.adapter = ResearchWanAdapter(worker=worker, validation_only=True)
        exp.worker = worker
        original = worker.step
        def step(state):
            if state.spec["seed"] == 102:
                raise torch.OutOfMemoryError("fixture urgent capacity")
            return original(state)
        worker.step = step
        main = exp.spec(self.config["prompts"][0], 101, self.shape, self.prefix)
        urgent = exp.spec(self.config["prompts"][1], 102, self.shape, self.config["recipes"][0])
        observation = {}
        with self.assertRaises(AdmissionCapacityError) as caught:
            exp.generate(main, action="keep", urgent_spec=urgent, admission=observation)
        self.assertEqual(caught.exception.observation["phase"], "urgent")
        self.assertEqual(caught.exception.observation["at_step"], 16)
        self.assertNotIn("main_completed", caught.exception.observation)
        json.dumps(caught.exception.observation)
        self.assertEqual(worker._states, {})
        with self.assertRaises(torch.OutOfMemoryError):
            exp.generate(exp.spec(self.config["prompts"][1], 102, self.shape, self.prefix))
        self.assertEqual(worker._states, {})


if __name__ == "__main__":
    unittest.main()
