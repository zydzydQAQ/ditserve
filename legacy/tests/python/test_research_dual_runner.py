"""No-GPU authorization, reservation and failed-unit supervision contracts."""

import contextlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from ditserve.research.budget import WorkerCleanupError
from ditserve.research.common import atomic_json, read_json
from ditserve.research.dual_protocol import DEFAULT_CONFIG
from ditserve.research.dual_runner import supervise, worker_main, probe_retry_inputs, fallback_backend_cost_estimate
from ditserve.research.dual_matrix import seal


class FakeBudget:
    def __init__(self, frozen):
        self.frozen = frozen
        self.outcomes, self.registrations, self.reservations = [], [], []
        self._data = dict(used_gpu_seconds=500., active=dict(session_id="session"))

    def reallocate_remaining(self, **kwargs):
        return {}

    @contextlib.contextmanager
    def session(self, stage, devices, estimate, cleanup, **kwargs):
        self.reservations.append((stage, devices, estimate, cleanup))
        yield self

    def check(self):
        return 100.

    def allocation_remaining(self):
        return dict(correctness=100., calibration=100., migration=100., mechanism=100., capacity=100., reserve=100.)

    def expect_workers(self):
        self.registrations.append("expected")

    def register_worker_group(self, pid):
        self.registrations.append(pid)

    def confirm_no_workers_started(self):
        self.registrations.append("none")

    def set_outcome(self, outcome):
        self.outcomes.append(outcome)


class DualRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = self.root / "ledger.json"
        self.config = read_json(DEFAULT_CONFIG)
        self.frozen = dict(config=self.config, ledger=str(self.ledger), sha256="protocol", source_sha256="source")
        self.devices = [dict(name="A100", uuid="GPU-a100"), dict(name="V100", uuid="GPU-v100")]
        self.process = Mock(pid=654321)
        self.process.wait.return_value = 0
        self.budget = FakeBudget(self.frozen)
        self.patchers = [patch("ditserve.research.dual_runner.prerequisites"),
                         patch("ditserve.research.dual_runner.DualResearchBudget", return_value=self.budget),
                         patch("ditserve.research.dual_runner.subprocess.Popen", return_value=self.process),
                         patch("ditserve.research.dual_runner.terminate_process_group"),
                         patch("ditserve.research.dual_runner.gpu_occupancy", return_value=dict(status="exclusive_observed"))]
        self.mocks = [p.start() for p in self.patchers]
        for p in self.patchers:
            self.addCleanup(p.stop)

    def test_reserves_selected_card_before_spawn_and_seals_success(self):
        def spawned(*args, **kwargs):
            self.assertEqual(self.budget.reservations[0][0:2], ("calibration", ["GPU-v100"]))
            self.assertEqual(self.budget.registrations, ["expected"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-v100")
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertTrue(kwargs["start_new_session"])
            return self.process
        self.mocks[2].side_effect = spawned
        value = supervise("calibrate-v100", self.frozen, self.root, self.devices)
        self.assertEqual(value["status"], "completed")
        job = read_json(self.root / "calibrate-v100-job.json")
        self.assertEqual(job["physical_devices"], [self.devices[1]])
        self.assertEqual(job["all_physical_devices"], self.devices)
        self.assertEqual(self.budget.registrations, ["expected", self.process.pid])
        self.assertEqual(self.budget.outcomes, ["completed"])
        self.mocks[3].assert_called_once_with(self.process)

    def test_retry_uses_reserve_and_protects_later_setup_costs_without_reset(self):
        seed = dict(source_protocol_sha256="previous", measured_costs_seconds=dict(a100=22., v100=174.))
        self.budget.bind_protocol_revision = Mock()
        self.budget.allocation_remaining = Mock(return_value=dict(
            correctness=700., calibration=3700., migration=5600., mechanism=4100., capacity=900., reserve=2800.))
        with patch("ditserve.research.dual_runner.retry_inputs", return_value=seed):
            value = supervise("probe", self.frozen, self.root, self.devices)
        self.assertEqual(value["status"], "completed")
        self.assertEqual(self.budget.reservations[0][0], "reserve")
        self.budget.bind_protocol_revision.assert_called_once()
        job = read_json(self.root / "probe-job.json")
        self.assertEqual(job["planning_seed"], seed)
        self.assertEqual(job["budget_stage"], "reserve")
        self.assertEqual(job["phase_remaining_gpu_seconds"]["correctness"], 1480.)
        self.assertEqual(job["phase_remaining_gpu_seconds"]["reserve"], 2800.)

    def test_retry_costs_require_sealed_matching_prior_failed_probe(self):
        old = dict(self.frozen, sha256="previous", model_manifest=dict(sha256="model"))
        current = dict(self.frozen, model_manifest=dict(sha256="model"))
        atomic_json(self.root / "probe-attempt.json", seal(dict(
            protocol_sha256="previous", gpu_work_started=True, status="budget_stopped")))
        atomic_json(self.root / "units/probe-main-baselines.json", dict(sha256="unit"))
        rows = [dict(type="probe", case_id=role + "-main", unit_id="probe-main-baselines",
                     performance_valid=True, run=dict(recipe_id="off", action="stay", elapsed_ms=cost))
                for role, cost in (("a100", 22000.), ("v100", 174000.))]
        with patch("ditserve.research.dual_runner.verify_protocol", return_value=old), \
                patch("ditserve.research.dual_runner.read_records", return_value=rows):
            seed = probe_retry_inputs(self.root, current)
            self.assertEqual(seed["measured_costs_seconds"], dict(a100=22., v100=174.))
            self.assertTrue(seed["validated"])
            rows[1]["performance_valid"] = False
            with self.assertRaisesRegex(ValueError, "sealed complete"):
                probe_retry_inputs(self.root, current)
            rows[1]["performance_valid"] = True
            changed = dict(current, model_manifest=dict(sha256="changed"))
            with self.assertRaisesRegex(ValueError, "identical scientific recipe"):
                probe_retry_inputs(self.root, changed)

    def test_repeated_retry_binds_latest_failure_but_keeps_original_sealed_cost_source(self):
        original_root, previous_root = self.root / "original", self.root / "previous"
        original = dict(self.frozen, sha256="original", model_manifest=dict(sha256="model"))
        previous = dict(original, sha256="previous")
        current = dict(original, sha256="current")
        for root, frozen in ((original_root, original), (previous_root, previous)):
            atomic_json(root / "probe-attempt.json", seal(dict(protocol_sha256=frozen["sha256"],
                                                               gpu_work_started=True, status="failed")))
        atomic_json(original_root / "units/probe-main-baselines.json", dict(sha256="unit"))
        rows = [dict(type="probe", case_id=role + "-main", unit_id="probe-main-baselines",
                     performance_valid=True, run=dict(recipe_id="off", action="stay", elapsed_ms=cost))
                for role, cost in (("a100", 22000.), ("v100", 174000.))]

        def verify(path, **kwargs):
            return original if path.parent == original_root else previous

        with patch("ditserve.research.dual_runner.verify_protocol", side_effect=verify), \
                patch("ditserve.research.dual_runner.read_records", return_value=rows):
            inherited = probe_retry_inputs(original_root, previous)
            atomic_json(previous_root / "probe-retry-inputs.json", inherited)
            seed = probe_retry_inputs(previous_root, current)
            self.assertEqual(seed["source_protocol_sha256"], "original")
            self.assertEqual(seed["source_run"], str(original_root))
            self.assertEqual(seed["retry_predecessor"]["protocol_sha256"], "previous")
            self.assertEqual(seed["retry_predecessor"]["run"], str(previous_root))
            inherited["measured_costs_seconds"]["a100"] = 1.
            inherited.pop("sha256")
            atomic_json(previous_root / "probe-retry-inputs.json", seal(inherited))
            with self.assertRaisesRegex(ValueError, "independently sealed"):
                probe_retry_inputs(previous_root, current)

    def test_conditional_candidate_estimate_requires_two_sealed_consistent_order_costs(self):
        rows = []
        for i, order in enumerate((["math", "efficient"], ["efficient", "math"])):
            cost = 13000. + i
            rows.append(dict(type="backend_pair", case_id="fallback-" + str(i), order=order,
                unit_id="pair-" + str(i), performance_valid=True, context_sha256="same",
                shape=self.config["shapes"]["fallback"], costs_ms=dict(efficient=cost),
                runs=dict(efficient=dict(elapsed_ms=cost, recipe_id="off", device_role="v100",
                                         action="stay", identity=dict(attention_path=["efficient"])))))
            atomic_json(self.root / "units" / ("pair-" + str(i) + ".json"), dict(sha256=str(i)))
        atomic_json(self.root / "probe-job.json", dict(physical_devices=self.devices))
        with patch("ditserve.research.dual_runner.read_records", return_value=rows):
            estimate = fallback_backend_cost_estimate(self.root, self.frozen)
            self.assertEqual(estimate["v100_seconds"], 13.001)
            self.assertNotIn("selected", estimate)
            self.assertIn("fresh_backend_qualification_required", estimate["purpose"])
            rows[0]["performance_valid"] = False
            self.assertIsNone(fallback_backend_cost_estimate(self.root, self.frozen))
            rows[0]["performance_valid"] = True
            rows[0]["runs"]["efficient"]["elapsed_ms"] = 1.
            with self.assertRaisesRegex(ValueError, "planning cost"):
                fallback_backend_cost_estimate(self.root, self.frozen)

    def active_unit(self):
        active = dict(unit_id="probe-incomplete", status="running", records=[], observations=[],
                      protocol_sha256="protocol", source_sha256="source")
        atomic_json(self.root / "active-unit.json", active)
        atomic_json(self.root / "units" / "probe-completed.json", dict(status="completed", marker="immutable"))
        return (self.root / "units" / "probe-completed.json").read_bytes()

    def test_worker_failure_seals_pending_unit_without_retroactive_invalidations(self):
        previous = self.active_unit()
        self.process.wait.return_value = 1
        value = supervise("probe", self.frozen, self.root, self.devices)
        self.assertEqual(value["status"], "failed")
        self.assertEqual(read_json(self.root / "active-unit.json")["status"], "idle")
        self.assertEqual(read_json(self.root / "units/probe-incomplete.json")["status"], "failed")
        self.assertEqual((self.root / "units/probe-completed.json").read_bytes(), previous)
        self.assertEqual(read_json(self.root / "probe-attempt.json")["status"], "failed")

    def test_transient_contention_aborts_active_unit_and_records_failed_attempt(self):
        previous = self.active_unit()
        self.mocks[4].side_effect = [dict(status="exclusive_observed"), dict(status="occupied", unexpected=[123])]
        value = supervise("probe", self.frozen, self.root, self.devices)
        self.assertEqual(value["status"], "blocked_contention")
        self.process.wait.assert_not_called()
        self.assertFalse(read_json(self.root / "units/probe-incomplete.json")["performance_valid"])
        self.assertEqual((self.root / "units/probe-completed.json").read_bytes(), previous)

    def test_cleanup_failure_does_not_leave_started_attempt_or_running_unit(self):
        self.active_unit()
        self.mocks[3].side_effect = WorkerCleanupError("owned process exit not confirmed")
        with self.assertRaises(WorkerCleanupError):
            supervise("probe", self.frozen, self.root, self.devices)
        self.assertNotEqual(read_json(self.root / "probe-attempt.json")["status"], "started")
        self.assertEqual(read_json(self.root / "active-unit.json")["status"], "idle")

    def authorized_job(self):
        job = dict(phase="probe", output=str(self.root), physical_devices=self.devices,
                   deadline_monotonic=time.monotonic() + 100., session_id="session",
                   protocol_sha256="protocol", source_sha256="source")
        active = dict(pid=12345, worker_pgid=98765, session_id="session", protocol_sha256="protocol",
                      devices=[d["uuid"] for d in self.devices], stage="correctness")
        atomic_json(self.ledger, dict(active=active))
        return job, active

    def invoke_worker(self, job, active):
        path = self.root / "job.json"
        # Intentionally bypass our strict writer so malformed/nonfinite incoming
        # jobs exercise worker validation rather than failing in this fixture.
        path.write_text(json.dumps(job))
        with patch("ditserve.research.dual_runner.verify_protocol", return_value=self.frozen), \
                patch("ditserve.research.dual_runner.os.getppid", return_value=active["pid"]), \
                patch("ditserve.research.dual_runner.os.getpgrp", return_value=active["worker_pgid"]), \
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(active["devices"])), \
                patch("ditserve.research.dual_experiment.run", return_value=dict(status="completed")) as run:
            result = worker_main(path)
            return result, run.call_count

    def test_worker_requires_bound_parent_group_session_and_device_set(self):
        job, active = self.authorized_job()
        self.assertEqual(self.invoke_worker(job, active), (0, 1))
        for changed in (dict(session_id="other"), dict(phase="migration"),
                        dict(physical_devices=[self.devices[0]]), dict(planning_seed={"validated": True}),
                        dict(budget_stage="reserve")):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                self.invoke_worker(dict(job, **changed), active)

    def test_worker_rejects_altered_protocol_source_and_expired_or_nonfinite_deadline(self):
        job, active = self.authorized_job()
        for changed in (dict(protocol_sha256="other"), dict(source_sha256="other"),
                        dict(deadline_monotonic=0.), dict(deadline_monotonic=float("inf"))):
            with self.subTest(changed=changed), self.assertRaises((RuntimeError, ValueError)):
                self.invoke_worker(dict(job, **changed), active)

    def test_worker_rejects_missing_active_session_before_loading_model(self):
        job, active = self.authorized_job()
        atomic_json(self.ledger, dict(active=None))
        with patch("ditserve.research.dual_runner.time.monotonic", side_effect=[0., 3.]):
            with self.assertRaises(RuntimeError):
                self.invoke_worker(job, active)


if __name__ == "__main__":
    unittest.main()
