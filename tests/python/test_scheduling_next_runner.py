from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.scheduling import next_runner as runner, next_worker as worker
from ditserve.scheduling.next_budget import Allocation, AUTHORIZATION, WatchdogStop


class NextRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output, self.ledger = self.root / "run", self.root / "gpu-budget.json"
        self.history = self.root / "old.json"
        atomic_json(self.history, dict(used_gpu_seconds=43000., limit_gpu_seconds=43200.))
        self.config = dict(budget_gpu_seconds=None, authorization=AUTHORIZATION, device_preference=["V100"],
            gpu_uuid=runner.GPU_UUID, python=sys.executable, v100_python=sys.executable,
            torch_home=str(self.root / "torch"), models={"wan": {}, "sana": {}}, seed=301,
            phases=list(runner.PHASES), phase_watchdog_seconds=172800)
        self.runtime = dict(self.config, device="cuda:0", gpu_name="Tesla V100-PCIE-32GB",
                            wan_attention="efficient", sana_profile="v100_fp32",
                            main_config_sha256=digest(self.config))
        self.v100 = dict(uuid=runner.GPU_UUID, index=1, name="Tesla V100-PCIE-32GB", compute_capability="7.0")
        self.a100 = dict(uuid="GPU-A100", index=0, name="NVIDIA A100", compute_capability="8.0")
        self.mono = 1.0
        replacement = patch.object(runner, "Allocation", side_effect=lambda path: Allocation(
            path, historical_path=self.history, monotonic=lambda: self.mono))
        replacement.start()
        self.addCleanup(replacement.stop)

    def fake_archive(self, output, phase, runtime):
        root = Path(output) / phase / "sources"
        module = root / "python/ditserve/scheduling/next_worker.py"
        module.parent.mkdir(parents=True)
        module.write_text("# frozen test worker\n", encoding="utf-8")
        files = {"python/ditserve/scheduling/next_worker.py": file_digest(module)}
        record = dict(phase=phase, execution_root=str(root), source_files=files,
                      source_sha256=digest(files), runtime_config_sha256=digest(runtime))
        record["sha256"] = digest(record)
        atomic_json(root / "manifest.json", record)
        return record

    def prerequisite(self, phase, **changes):
        result = dict(status="passed", eligible=False, returncode=0, main_config_sha256=digest(self.config),
                      runtime_config_sha256=digest(self.runtime))
        result.update(changes)
        atomic_json(self.output / phase / "supervisor.json", result)
        if phase == "completion":
            atomic_json(self.output / phase / "runtime-config.json", self.runtime)

    def execute_mocked(self, *, cleanup=True, result=None, isolation=None, spawn_error=None):
        result = result or dict(status="passed", eligible=False, technical_complete=True)
        owner = self
        class Child:
            pid = 123456
            def wait(self, timeout=None):
                owner.mono += 100
                session = read_json(owner.ledger)["active"]["session_id"]
                atomic_json(owner.output / "completion/worker.json", dict(
                    result, phase="completion", session_id=session))
                return 0
        def spawn(*args, **kwargs):
            owner.mono += 50
            if spawn_error:
                raise spawn_error
            return Child()
        def stop(process):
            owner.mono += 20
            return cleanup
        with patch.object(runner, "inspect", return_value=dict(status="ready", runtime_config=self.runtime)), \
             patch.object(runner, "freeze_source", side_effect=self.fake_archive), \
             patch.object(runner.subprocess, "Popen", side_effect=spawn) as popen, \
             patch.object(runner, "stop_owned_group", side_effect=stop), \
             patch.object(runner, "gpu_occupancy", side_effect=isolation,
                          return_value=dict(status="exclusive_observed", owned=[])):
            value = runner.execute(self.config, self.output, self.ledger, "completion")
        return value, popen

    def test_configuration_requires_explicit_unlimited_authorization_and_device(self):
        for changes in ({"budget_gpu_seconds": 43200}, {"authorization": "old"},
                        {"device_preference": ["A100", "V100"]}, {"gpu_uuid": "GPU-other"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.validate_config(dict(self.config, **changes))
        config = dict(self.config)
        del config["budget_gpu_seconds"]
        with self.assertRaises(ValueError):
            runner.validate_config(config)

    def test_watchdog_is_finite_whole_work_protection_independent_of_accounting(self):
        self.assertEqual(runner.phase_watchdog(self.config, "completion"), 172800.)
        config = dict(self.config, phase_watchdog_seconds={}, phase_estimated_seconds={"completion": 200000})
        self.assertEqual(runner.phase_watchdog(config, "completion"), 300000.)
        with self.assertRaisesRegex(ValueError, "complete work"):
            runner.phase_watchdog(config, "completion", timeout=299999)
        for duration in (0, 10, float("inf"), float("nan")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                runner.phase_watchdog(self.config, "completion", timeout=duration)

    def test_only_pinned_v100_is_considered_even_if_other_cards_are_idle(self):
        other_v100 = dict(self.v100, uuid="GPU-other-V100")
        with patch.object(runner, "discover_gpus", return_value={"devices": [self.a100, other_v100, self.v100]}), \
             patch.object(runner, "gpu_occupancy", return_value={"status": "exclusive_observed"}) as occupancy, \
             patch.object(runner, "model_inventory", return_value={"available": True}):
            result = runner.inspect(self.config)
        occupancy.assert_called_once_with([runner.GPU_UUID])
        self.assertEqual(result["runtime_config"]["gpu_uuid"], runner.GPU_UUID)
        self.assertEqual(result["runtime_config"]["sana_profile"], "v100_fp32")
        self.assertEqual(result["runtime_config"]["wan_attention"], "efficient")
        self.assertFalse(self.ledger.exists())

    def test_busy_device_never_spawns_or_starts_accounting(self):
        with patch.object(runner, "inspect", return_value={"status": "blocked"}), \
             patch.object(runner.subprocess, "Popen") as popen:
            result = runner.execute(self.config, self.output, self.ledger, "completion")
        popen.assert_not_called()
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(self.ledger.exists())

    def test_complete_but_unqualified_prerequisites_do_not_block_declared_next_phase(self):
        for previous, phase in (("completion", "granularity"), ("granularity", "replicate")):
            self.prerequisite(previous, status="stopped", eligible=False, technical_complete=True)
            with self.subTest(phase=phase), patch.object(runner, "inspect", return_value={"status": "blocked"}) as inspect:
                runner.execute(self.config, self.output, self.ledger, phase)
                inspect.assert_called_once()
        self.prerequisite("granularity", status="incomplete", technical_complete=False)
        with patch.object(runner, "inspect") as inspect:
            result = runner.execute(self.config, self.output, self.ledger, "replicate")
        inspect.assert_not_called()
        self.assertEqual(result["status"], "blocked")

    def test_failed_or_tampered_supervision_cannot_satisfy_prerequisite(self):
        for changes in ({"supervisor_failure": {"reason": "isolation"}}, {"returncode": 1},
                        {"watchdog_stopped": True}, {"main_config_sha256": "wrong"}):
            self.prerequisite("completion", **changes)
            with self.subTest(changes=changes), patch.object(runner, "inspect") as inspect:
                runner.execute(self.config, self.output, self.ledger, "granularity")
                inspect.assert_not_called()

    def test_completion_locks_runtime_even_when_performance_eligibility_is_false(self):
        self.prerequisite("completion", eligible=False)
        self.assertEqual(runner._locked_runtime(self.output, self.config), self.runtime)
        atomic_json(self.output / "completion/runtime-config.json", dict(self.runtime, python="wrong"))
        with self.assertRaisesRegex(ValueError, "runtime changed"):
            runner._locked_runtime(self.output, self.config)

    def test_prepare_binds_new_ledger_without_resetting_cost_or_old_evidence(self):
        before = self.history.read_bytes()
        result = runner.prepare(self.config, self.output, self.ledger)
        self.assertIsNone(result["remaining_gpu_seconds"])
        with Allocation(self.ledger, historical_path=self.history, monotonic=lambda: self.mono) as account:
            account.begin(phase="completion", identity={})
            self.mono += 200000
            account.finish("completed")
        original = self.ledger.read_bytes()
        runner.prepare(self.config, self.root / "another-run", self.ledger)
        self.assertEqual(self.ledger.read_bytes(), original)
        self.assertEqual(self.history.read_bytes(), before)
        with self.assertRaises(ValueError):
            runner.prepare(dict(self.config, seed=999), self.output, self.ledger)

    def test_archive_cwd_process_group_and_loading_cleanup_are_charged(self):
        result, popen = self.execute_mocked()
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["eligible"])
        self.assertEqual(result["charge"]["gpu_seconds"], 170.)
        self.assertEqual(result["watchdog_seconds"], 172800.)
        self.assertIsNone(result["limit_gpu_seconds"])
        self.assertIsNone(read_json(self.ledger)["active"])
        call = popen.call_args
        archive = self.output / "completion/sources"
        self.assertEqual(call.kwargs["cwd"], str(archive))
        self.assertEqual(call.kwargs["env"]["PYTHONPATH"], str(archive / "python"))
        self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], runner.GPU_UUID)
        self.assertTrue(call.kwargs["start_new_session"])
        self.assertIn("ditserve.scheduling.next_worker", call.args[0])
        with self.assertRaisesRegex(ValueError, "already attempted"):
            runner.execute(self.config, self.output, self.ledger, "completion")

    def test_spawn_failure_still_counts_owned_cleanup_time(self):
        result, unused = self.execute_mocked(spawn_error=OSError("spawn failed"))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["charge"]["gpu_seconds"], 70.)
        self.assertEqual(result["charge"]["status"], "failed")

    def test_isolation_loss_revokes_result_and_charges_failed_cleaned_session(self):
        result, unused = self.execute_mocked(isolation=[dict(status="contended", owned=[]),
                                                       dict(status="exclusive_observed", owned=[])])
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["eligible"])
        self.assertEqual(result["charge"]["gpu_seconds"], 70.)
        self.assertIsNone(read_json(self.ledger)["active"])

    def test_cleanup_uncertainty_keeps_session_active(self):
        with self.assertRaisesRegex(RuntimeError, "unsettled"):
            self.execute_mocked(cleanup=False)
        self.assertIsNotNone(read_json(self.ledger)["active"])
        self.assertTrue((self.output / "completion/cleanup-unverified.json").exists())

    def test_worker_admission_checks_exact_source_device_deadline_and_parent(self):
        source = self.fake_archive(self.output, "completion", self.runtime)
        args = SimpleNamespace(ledger=self.ledger, output=self.output, phase="completion", session_id="unit",
                               deadline=100., source_manifest=Path(source["execution_root"]) / "manifest.json")
        with Allocation(self.ledger, historical_path=self.history) as account:
            account.begin(phase="completion", identity=dict(output=str(self.output), phase="completion",
                gpu_uuid=runner.GPU_UUID, authorization=AUTHORIZATION, runtime_config_sha256=digest(self.runtime),
                main_config_sha256=digest(self.config), deadline_monotonic=100.,
                execution_root=source["execution_root"], source_sha256=source["source_sha256"]))
            account.data["active"].update(session_id="unit", supervisor_pid=10, child_pgid=20)
            account.save()
        arguments = dict(getpid=lambda: 20, getpgrp=lambda: 20, getppid=lambda: 10, monotonic=lambda: 1.,
                         environ={"CUDA_VISIBLE_DEVICES": runner.GPU_UUID}, executable=sys.executable,
                         module_file=str(Path(source["execution_root"]) / "python/ditserve/scheduling/next_worker.py"))
        self.assertEqual(worker.verify_admission(args, self.runtime, **arguments)["session_id"], "unit")
        for changes in ({"getppid": lambda: 99}, {"getpgrp": lambda: 21}, {"module_file": __file__},
                        {"executable": "/wrong/python"}, {"environ": {"CUDA_VISIBLE_DEVICES": "GPU-other"}}):
            with self.subTest(changes=changes), self.assertRaises(WatchdogStop):
                worker.verify_admission(args, self.runtime, **dict(arguments, **changes))
        args.deadline = 200.
        with self.assertRaises(WatchdogStop):
            worker.verify_admission(args, self.runtime, **arguments)

    def test_worker_rejects_admission_before_importing_execution_or_models(self):
        config = self.root / "runtime.json"
        atomic_json(config, self.runtime)
        with patch.object(worker, "verify_admission", side_effect=WatchdogStop("denied")), \
             patch.object(worker.importlib, "import_module") as importer, patch("builtins.print"):
            code = worker.main(["--config", str(config), "--output", str(self.output), "--phase", "completion",
                "--deadline", "100", "--ledger", str(self.ledger), "--session-id", "unit",
                "--source-manifest", str(self.root / "manifest.json")])
        importer.assert_not_called()
        self.assertEqual(code, 2)
        self.assertTrue(read_json(self.output / "completion/worker.json")["watchdog_stopped"])


if __name__ == "__main__":
    unittest.main()
