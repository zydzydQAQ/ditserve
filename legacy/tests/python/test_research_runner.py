"""CPU-only source identity, subprocess admission and lifecycle accounting."""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ditserve.research.budget import ResearchBudget, WorkerCleanupError
from ditserve.research.common import atomic_json, read_json
from ditserve.benchmark.common import seal
from ditserve.research.cost_planning import read_bound_cost_plan
from ditserve.research.preflight import cpu_validation, select_devices
from ditserve.research.protocol import DEFAULT_CONFIG, freeze_protocol, verify_protocol
from ditserve.research.runner import _valid_cpu, run_gpu_stage, terminate_process_group, validate_worker_job
from ditserve.research.isolation import GPUIsolationError


class ResearchRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = self.root / "budget.json"
        self.config = read_json(DEFAULT_CONFIG)
        self.frozen = dict(config=self.config, ledger=str(self.ledger), sha256="protocol", source_sha256="source")
        self.device = dict(uuid="GPU-FAKE", name="A100", index=0, memory_mib=40960, compute_capability="8.0")
        self.environment = dict(status="ready", reasons=[], selection=dict(devices=[self.device]),
                                model=dict(path="/unused-model"), offline_quality_weights_ready=True)
        self.now = 100.
        watcher = patch("ditserve.research.runner.wait_for_exclusive_worker",
                        side_effect=lambda process, **kwargs: process.wait(timeout=kwargs["timeout"]))
        self.waiter = watcher.start()
        self.addCleanup(watcher.stop)

    def budget(self):
        return ResearchBudget(self.ledger, clock=lambda: self.now, monotonic=lambda: self.now)

    def test_new_implementation_file_cannot_escape_frozen_source_manifest(self):
        source = self.root / "python/ditserve/first.py"
        source.parent.mkdir(parents=True)
        source.write_text("value = 1\n")
        with patch("ditserve.research.protocol.dependency_identity", return_value={}):
            frozen = freeze_protocol(DEFAULT_CONFIG, self.root / "run", self.ledger, root=self.root)
            (source.parent / "new_imported_impl.py").write_text("value = 2\n")
            with self.assertRaises(ValueError):
                verify_protocol(self.root / "run/protocol.json", root=self.root)
            self.assertEqual(verify_protocol(self.root / "run/protocol.json", root=self.root, verify_current=False), frozen)

    def test_physical_device_aliases_cannot_inflate_budget_device_count(self):
        rows = [dict(self.device, index=i, uuid="GPU-" + str(i)) for i in range(4)]
        with self.assertRaisesRegex(ValueError, "same physical"):
            select_devices(dict(status="available", devices=rows), ["0", "GPU-0", "1", "2"])

    def job(self, active):
        return dict(stage="mechanism", session_id=active["session_id"],
                    protocol_sha256=self.frozen["sha256"], source_sha256=self.frozen["source_sha256"],
                    physical_devices=[self.device], deadline_monotonic=time.monotonic() + 120)

    def test_worker_requires_matching_live_parent_device_protocol_and_stage(self):
        with self.budget().session("calibrate", ["GPU-FAKE"], 10, 1):
            active = read_json(self.ledger)["active"]
            job = self.job(active)
            with patch("ditserve.research.runner.os.getppid", return_value=active["pid"]), patch.dict(
                    os.environ, CUDA_VISIBLE_DEVICES="GPU-FAKE"):
                validate_worker_job(job, self.frozen)
                for change in (dict(session_id="other"), dict(stage="sp"), dict(source_sha256="other"),
                               dict(protocol_sha256="other"), dict(physical_devices=[dict(self.device, uuid="GPU-OTHER")]),
                               dict(deadline_monotonic=0)):
                    with self.subTest(change=change), self.assertRaises(RuntimeError):
                        validate_worker_job(dict(job, **change), self.frozen)
                with patch("ditserve.research.runner.os.getppid", return_value=active["pid"] + 1):
                    with self.assertRaises(RuntimeError):
                        validate_worker_job(job, self.frozen)
                with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-OTHER"):
                    with self.assertRaises(RuntimeError):
                        validate_worker_job(job, self.frozen)
        with patch("ditserve.research.runner.os.getppid", return_value=active["pid"]), patch.dict(
                os.environ, CUDA_VISIBLE_DEVICES="GPU-FAKE"):
            with self.assertRaises(RuntimeError):
                validate_worker_job(job, self.frozen)

    def test_explicit_recheck_reserve_never_borrows_sp_or_resets_usage(self):
        with self.budget().session("reserve", ["GPU-FAKE"], 10, 1):
            active = read_json(self.ledger)["active"]
            job = dict(self.job(active), use_reserve=True)
            with patch("ditserve.research.runner.os.getppid", return_value=active["pid"]), patch.dict(
                    os.environ, CUDA_VISIBLE_DEVICES="GPU-FAKE"):
                validate_worker_job(job, self.frozen)
                with self.assertRaises(RuntimeError):
                    validate_worker_job(dict(job, use_reserve=False), self.frozen)
            self.now += 30
        usage = read_json(self.ledger)["stage_used_gpu_seconds"]
        self.assertEqual(usage["reserve"], 30)
        self.assertEqual(usage["sp"], 0)

    def test_supervisor_reserves_before_spawn_and_charges_through_cleanup(self):
        process = Mock(pid=123456)
        def spawn(*args, **kwargs):
            active = read_json(self.ledger)["active"]
            self.assertEqual(active["devices"], ["GPU-FAKE"])
            job = read_json(self.root / "probe-job.json")
            self.assertEqual(job["session_id"], active["session_id"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-FAKE")
            self.assertTrue(kwargs["start_new_session"])
            return process
        def wait(timeout):
            self.now += 20
            return 0
        def cleanup(child):
            self.assertIs(child, process)
            self.assertIsNotNone(read_json(self.ledger)["active"])
            self.now += 7
        process.wait.side_effect = wait
        with patch("ditserve.research.runner.inspect_environment", return_value=self.environment), patch(
                "ditserve.research.runner.stage_prerequisites", return_value=[]), patch(
                "ditserve.research.runner.ResearchBudget", side_effect=lambda *a, **kw: self.budget()), patch(
                "ditserve.research.runner.subprocess.Popen", side_effect=spawn), patch(
                "ditserve.research.runner.terminate_process_group", side_effect=cleanup):
            result = run_gpu_stage("probe", self.frozen, self.root, ["0"])
        self.assertEqual(result["status"], "completed")
        ledger = read_json(self.ledger)
        self.assertEqual(ledger["used_gpu_seconds"], 27)
        self.assertIsNone(ledger["active"])
        self.assertEqual(ledger["sessions"][0]["status"], "completed")

    def test_failed_spawn_is_charged_and_blocked_environment_never_spawns(self):
        def fail(*args, **kwargs):
            self.assertIsNotNone(read_json(self.ledger)["active"])
            self.now += 4
            raise OSError("launch failed")
        with patch("ditserve.research.runner.inspect_environment", return_value=self.environment), patch(
                "ditserve.research.runner.stage_prerequisites", return_value=[]), patch(
                "ditserve.research.runner.ResearchBudget", side_effect=lambda *a, **kw: self.budget()), patch(
                "ditserve.research.runner.subprocess.Popen", side_effect=fail):
            with self.assertRaisesRegex(OSError, "launch failed"):
                run_gpu_stage("probe", self.frozen, self.root, ["0"])
        self.assertEqual(read_json(self.ledger)["used_gpu_seconds"], 4)
        blocked = dict(self.environment, status="blocked", reasons=["driver unavailable"])
        with patch("ditserve.research.runner.inspect_environment", return_value=blocked), patch(
                "ditserve.research.runner.subprocess.Popen") as spawn:
            result = run_gpu_stage("calibrate", self.frozen, self.root, ["0"])
            self.assertFalse(result["gpu_work_started"])
            spawn.assert_not_called()

    def test_contention_cleans_owned_group_and_charges_the_failed_measurement(self):
        process = Mock(pid=123456)
        def interrupted(*args, **kwargs):
            self.now += 20
            raise GPUIsolationError("foreign GPU process")
        def cleanup(child):
            self.assertIs(child, process)
            self.now += 7
        self.waiter.side_effect = interrupted
        with patch("ditserve.research.runner.inspect_environment", return_value=self.environment), patch(
                "ditserve.research.runner.stage_prerequisites", return_value=[]), patch(
                "ditserve.research.runner.ResearchBudget", side_effect=lambda *a, **kw: self.budget()), patch(
                "ditserve.research.runner.subprocess.Popen", return_value=process), patch(
                "ditserve.research.runner.terminate_process_group", side_effect=cleanup):
            result = run_gpu_stage("probe", self.frozen, self.root, ["0"])
        self.assertEqual(result["status"], "blocked_contention")
        ledger = read_json(self.ledger)
        self.assertEqual(ledger["used_gpu_seconds"], 27)
        self.assertIsNone(ledger["active"])
        self.assertEqual(ledger["sessions"][0]["status"], "blocked_contention")

    def test_unreaped_worker_retains_active_ledger_for_conservative_recovery(self):
        process = Mock(pid=123456)
        process.wait.return_value = 0
        with patch("ditserve.research.runner.inspect_environment", return_value=self.environment), patch(
                "ditserve.research.runner.stage_prerequisites", return_value=[]), patch(
                "ditserve.research.runner.ResearchBudget", side_effect=lambda *a, **kw: self.budget()), patch(
                "ditserve.research.runner.subprocess.Popen", return_value=process), patch(
                "ditserve.research.runner.terminate_process_group", side_effect=WorkerCleanupError("rank alive")):
            with self.assertRaises(WorkerCleanupError):
                run_gpu_stage("probe", self.frozen, self.root, ["0"])
        self.assertIsNotNone(read_json(self.ledger)["active"])
        self.now += 30
        with patch("ditserve.research.budget.process_group_alive", return_value=True):
            with self.assertRaisesRegex(WorkerCleanupError, "unconfirmed"):
                self.budget().initialize()
            self.assertIsNotNone(read_json(self.ledger)["active"])
        with patch("ditserve.research.budget.process_group_alive", return_value=False):
            self.assertEqual(self.budget().initialize()["used_gpu_seconds"], 30)

    def test_second_interrupt_during_cleanup_cannot_clear_live_reservation(self):
        process = Mock(pid=123456)
        process.wait.return_value = 0
        with patch("ditserve.research.runner.inspect_environment", return_value=self.environment), patch(
                "ditserve.research.runner.stage_prerequisites", return_value=[]), patch(
                "ditserve.research.runner.ResearchBudget", side_effect=lambda *a, **kw: self.budget()), patch(
                "ditserve.research.runner.subprocess.Popen", return_value=process), patch(
                "ditserve.research.runner.terminate_process_group", side_effect=KeyboardInterrupt):
            with self.assertRaisesRegex(WorkerCleanupError, "interrupted"):
                run_gpu_stage("probe", self.frozen, self.root, ["0"])
        ledger = read_json(self.ledger)
        self.assertIsNotNone(ledger["active"])
        self.assertEqual(ledger["active"]["cleanup_status"], "unreaped_workers")
        self.now += 30
        with patch("ditserve.research.budget.process_group_alive", return_value=True):
            with self.assertRaises(WorkerCleanupError):
                self.budget().initialize()

    def test_probe_cost_planning_requires_actual_offline_quality_weights(self):
        with patch("ditserve.research.runner.inspect_environment", return_value=dict(
                self.environment, offline_quality_weights_ready=False)), patch(
                "ditserve.research.runner.stage_prerequisites", return_value=[]), patch(
                "ditserve.research.runner.subprocess.Popen") as spawn:
            result = run_gpu_stage("probe", self.frozen, self.root, ["0"])
        self.assertEqual(result["status"], "blocked")
        spawn.assert_not_called()

    def test_foreign_sealed_plan_cannot_lower_complete_unit_estimate(self):
        plan = seal(dict(schema_version="ditserve.research.cost_plan.v1", status="planned",
            protocol_sha256="foreign", source_sha256="foreign", unit_estimates=dict(calibrate=1, mechanism=1),
            candidate_limit=1, selected_shape=dict(frames=17)))
        atomic_json(self.root / "cost-plan.json", plan)
        atomic_json(self.root / "probe.json", dict(status="completed", protocol_sha256="foreign",
                    selected_shape=plan["selected_shape"], cost_plan=plan))
        with self.assertRaisesRegex(ValueError, "not bound"):
            read_bound_cost_plan(self.root, self.frozen)

    def test_process_group_cleanup_reaps_child_after_launcher_has_exited(self):
        code = ("import subprocess, sys; "
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
                "print(child.pid, flush=True)")
        leader = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                                  start_new_session=True)
        try:
            child_pid = int(leader.stdout.readline().strip())
            self.assertEqual(leader.wait(timeout=5), 0)
            self.assertTrue(Path("/proc", str(child_pid), "stat").exists())
            terminate_process_group(leader)
            stat = Path("/proc", str(child_pid), "stat")
            if stat.exists():
                self.assertEqual(stat.read_text().rsplit(")", 1)[1].split()[0], "Z")
        finally:
            try:
                os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            leader.wait(timeout=5)
            leader.stdout.close()

    def test_cpu_attestation_rejects_skips_and_protocol_checksum_mutation(self):
        skipped = lambda *a, **kw: subprocess.CompletedProcess(a, 0, "", "Ran 10 tests in 1.0s\nOK (skipped=1)\n")
        result = cpu_validation(self.frozen, self.root, run=skipped)
        self.assertEqual(result["status"], "failed")
        good = lambda *a, **kw: subprocess.CompletedProcess(a, 0, "", "Ran 10 tests in 1.0s\nOK\n")
        result = cpu_validation(self.frozen, self.root, run=good)
        self.assertTrue(_valid_cpu(self.root / "cpu-validation.json", self.frozen))
        result["source_sha256"] = "drift"
        atomic_json(self.root / "cpu-validation.json", result)
        self.assertFalse(_valid_cpu(self.root / "cpu-validation.json", self.frozen))


if __name__ == "__main__":
    unittest.main()
