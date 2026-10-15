import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.scheduling.budget import Allocation, BudgetStop, TOTAL
from ditserve.scheduling import runner, worker


class SchedulingRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output, self.ledger = self.root / "run", self.root / "budget.json"
        self.history = self.root / "historical.json"
        atomic_json(self.history, dict(used_gpu_seconds=123.0, limit_gpu_seconds=86400.0))
        self.config = dict(budget_gpu_seconds=TOTAL, python=sys.executable, v100_python=sys.executable,
                           torch_home=str(self.root / "torch"), models={"wan": {}, "sana": {}}, seed=301)
        self.a100 = dict(uuid="GPU-A", index=0, name="NVIDIA A100", compute_capability="8.0")
        self.v100 = dict(uuid="GPU-V", index=1, name="Tesla V100-PCIE-32GB", compute_capability="7.0")
        self.runtime = dict(self.config, gpu_uuid="GPU-A", device="cuda:0", gpu_name="NVIDIA A100",
                            wan_attention="flash", sana_profile="a100_bf16",
                            main_config_sha256=digest(self.config))
        allocation_patch = patch.object(runner, "Allocation", side_effect=lambda path: Allocation(
            path, historical_path=self.history))
        allocation_patch.start()
        self.addCleanup(allocation_patch.stop)

    def ready(self):
        return dict(status="ready", runtime_config=self.runtime)

    def fake_archive(self, output, phase, runtime):
        root = Path(output) / phase / "sources"
        module = root / "python/ditserve/scheduling/worker.py"
        module.parent.mkdir(parents=True)
        module.write_text("# immutable CPU test fixture\n")
        files = {"python/ditserve/scheduling/worker.py": file_digest(module)}
        record = dict(phase=phase, execution_root=str(root), source_files=files,
                      source_sha256=digest(files), runtime_config_sha256=digest(runtime))
        record["sha256"] = digest(record)
        atomic_json(root / "manifest.json", record)
        return record

    def prerequisite(self, phase, *, eligible=True):
        atomic_json(self.output / phase / "supervisor.json", dict(status="passed", eligible=eligible,
                    main_config_sha256=digest(self.config), runtime_config_sha256=digest(self.runtime)))
        if phase == "calibrate":
            atomic_json(self.output / phase / "runtime-config.json", self.runtime)

    def test_selects_idle_a100_then_v100_without_gpu_work(self):
        for busy, expected in ((None, "a100_bf16"), ("GPU-A", "v100_fp32")):
            with self.subTest(busy=busy), \
                 patch.object(runner, "discover_gpus", return_value={"devices": [self.v100, self.a100]}), \
                 patch.object(runner, "gpu_occupancy", side_effect=lambda ids: dict(
                     status="contended" if ids[0] == busy else "exclusive_observed")), \
                 patch.object(runner, "model_inventory", return_value={"available": True}):
                result = runner.inspect(self.config)
            self.assertEqual(result["runtime_config"]["sana_profile"], expected)
            self.assertFalse(result["gpu_work_started"])
        self.assertFalse(self.ledger.exists())

    def test_calibration_locks_device_instead_of_switching_to_an_idle_other_gpu(self):
        self.prerequisite("calibrate")
        with patch.object(runner, "discover_gpus", return_value={"devices": [self.a100, self.v100]}), \
             patch.object(runner, "gpu_occupancy", return_value={"status": "contended"}), \
             patch.object(runner, "model_inventory", return_value={"available": True}):
            result = runner.inspect(self.config, self.output, "pilot")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(result["observations"]), 1)
        self.assertEqual(result["observations"][0]["device"]["uuid"], "GPU-A")

    def test_v100_only_never_uses_idle_a100_or_legacy_a100(self):
        config = dict(self.config, device_preference=["V100"])
        with patch.object(runner, "discover_gpus", return_value={"devices": [self.a100, self.v100]}), \
             patch.object(runner, "gpu_occupancy", return_value={"status": "exclusive_observed"}) as occupancy, \
             patch.object(runner, "model_inventory", return_value={"available": True}):
            result = runner.inspect(config)
            self.assertEqual(result["runtime_config"]["gpu_uuid"], "GPU-V")
            occupancy.assert_called_once_with(["GPU-V"])
            legacy = runner.inspect(config, phase="legacy")
            self.assertEqual(legacy["status"], "blocked")
            self.assertIn("excluded by device_preference", legacy["reason"])
        with patch.object(runner, "discover_gpus", return_value={"devices": [self.a100, self.v100]}), \
             patch.object(runner, "gpu_occupancy", return_value={"status": "contended"}), \
             patch.object(runner, "model_inventory", return_value={"available": True}):
            self.assertEqual(runner.inspect(config)["status"], "blocked")

    def test_modified_calibration_runtime_fails_closed(self):
        self.prerequisite("calibrate")
        atomic_json(self.output / "calibrate/runtime-config.json", dict(self.runtime, gpu_uuid="GPU-V"))
        with self.assertRaisesRegex(ValueError, "changed after"):
            runner._locked_runtime(self.output, self.config)

    def test_legacy_never_substitutes_v100_for_old_a100_evidence(self):
        with patch.object(runner, "discover_gpus", return_value={"devices": [self.v100]}), \
             patch.object(runner, "model_inventory", return_value={"available": True}):
            self.assertEqual(runner.inspect(self.config, phase="legacy")["status"], "blocked")

    def test_busy_device_cannot_spend_budget_or_start_worker(self):
        with patch.object(runner, "inspect", return_value={"status": "blocked"}), \
             patch.object(runner.subprocess, "Popen") as popen:
            result = runner.execute(self.config, self.output, self.ledger, "calibrate")
        self.assertEqual(result["status"], "blocked")
        popen.assert_not_called()
        self.assertFalse(self.ledger.exists())

    def test_prerequisite_needs_supervisor_but_pilot_completeness_is_not_scientific_qualification(self):
        atomic_json(self.output / "calibrate/worker.json", {"status": "passed", "eligible": True})
        with patch.object(runner, "inspect") as inspect:
            result = runner.execute(self.config, self.output, self.ledger, "pilot")
        self.assertEqual(result["status"], "blocked")
        inspect.assert_not_called()
        self.prerequisite("pilot", eligible=False)
        with patch.object(runner, "inspect", return_value={"status": "blocked"}) as inspect:
            result = runner.execute(self.config, self.output, self.ledger, "screen")
        inspect.assert_called_once()
        self.assertEqual(result["status"], "blocked")
        self.prerequisite("screen", eligible=False)
        with patch.object(runner, "inspect") as inspect:
            runner.execute(self.config, self.output, self.ledger, "validate")
        inspect.assert_not_called()

    def test_prepare_binds_run_config_and_ledger_without_reset(self):
        runner.prepare(self.config, self.output, self.ledger)
        with Allocation(self.ledger, historical_path=self.history) as budget:
            budget.begin(phase="calibrate", identity={})
            budget.finish("completed")
        original = self.ledger.read_bytes()
        runner.prepare(self.config, self.root / "second-run", self.ledger)
        self.assertEqual(original, self.ledger.read_bytes())
        with self.assertRaises(ValueError):
            runner.prepare(dict(self.config, seed=401), self.output, self.ledger)
        with self.assertRaises(ValueError):
            runner.prepare(self.config, self.output, self.root / "new-ledger.json")
        self.assertFalse((self.root / "new-ledger.json").exists())

    def test_timeout_too_short_keeps_phase_retryable(self):
        with patch.object(runner, "inspect", return_value=self.ready()), \
             patch.object(runner, "freeze_source") as freeze:
            result = runner.execute(self.config, self.output, self.ledger, "calibrate", timeout=10)
        self.assertEqual(result["status"], "blocked")
        freeze.assert_not_called()
        self.assertIsNone(read_json(self.ledger)["active"])
        self.assertFalse((self.output / "calibrate/runtime-config.json").exists())

    def test_archive_runs_unchanged_when_workspace_changes_but_rejects_archive_tampering(self):
        source_root = self.root / "workspace"
        module = source_root / "python/ditserve/fixture.py"
        module.parent.mkdir(parents=True)
        module.write_text("x = 1\n")
        with patch.object(runner, "ROOT", source_root):
            source = runner.freeze_source(self.output, "calibrate", self.runtime)
        module.write_text("x = 2\n")
        runner.verify_archive(source)
        (Path(source["execution_root"]) / "python/ditserve/fixture.py").write_text("x = 3\n")
        with self.assertRaisesRegex(ValueError, "archived executable"):
            runner.verify_archive(source)

    def execute_mocked(self, *, cleanup=True, result=None, isolation=None):
        result = result or dict(status="passed", eligible=True)
        instance = self
        class Child:
            pid = 123456
            def wait(self, timeout=None):
                session = read_json(instance.ledger)["active"]["session_id"]
                atomic_json(instance.output / "calibrate/worker.json", dict(result, phase="calibrate", session_id=session))
                return 0
        with patch.object(runner, "inspect", return_value=self.ready()), \
             patch.object(runner, "freeze_source", side_effect=self.fake_archive), \
             patch.object(runner.subprocess, "Popen", return_value=Child()) as popen, \
             patch.object(runner, "stop_owned_group", return_value=cleanup), \
             patch.object(runner, "gpu_occupancy", side_effect=isolation,
                          return_value=dict(status="exclusive_observed", owned=[])):
            value = runner.execute(self.config, self.output, self.ledger, "calibrate", timeout=120)
        return value, popen

    def test_supervisor_uses_archived_cwd_and_preserves_qualification(self):
        result, popen = self.execute_mocked(result=dict(status="passed", eligible=False, technical_complete=True))
        self.assertFalse(result["eligible"])
        self.assertTrue(result["technical_complete"])
        self.assertEqual(result["status"], "passed")
        self.assertIsNone(read_json(self.ledger)["active"])
        call = popen.call_args
        archive = self.output / "calibrate/sources"
        self.assertEqual(call.kwargs["cwd"], str(archive))
        self.assertEqual(call.kwargs["env"]["PYTHONPATH"], str(archive / "python"))
        self.assertTrue(call.kwargs["start_new_session"])
        with self.assertRaisesRegex(ValueError, "already attempted"):
            runner.execute(self.config, self.output, self.ledger, "calibrate")

    def test_uncertain_cleanup_keeps_budget_active(self):
        with self.assertRaisesRegex(RuntimeError, "remains unsettled"):
            self.execute_mocked(cleanup=False)
        self.assertIsNotNone(read_json(self.ledger)["active"])
        self.assertTrue((self.output / "calibrate/cleanup-unverified.json").exists())

    def test_isolation_loss_during_execution_revokes_results_but_still_charges_cleaned_session(self):
        result, _ = self.execute_mocked(isolation=[
            dict(status="contended", foreign=[{"pid": 987654}], owned=[]),
            dict(status="exclusive_observed", owned=[])])
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["eligible"])
        self.assertIn("isolation lost", result["supervisor_failure"]["reason"])
        ledger = read_json(self.ledger)
        self.assertIsNone(ledger["active"])
        self.assertGreater(ledger["used_gpu_seconds"], 0.)
        self.assertEqual(ledger["sessions"][0]["status"], "failed")

    def test_final_isolation_loss_revokes_a_passed_worker_report(self):
        result, _ = self.execute_mocked(isolation=[dict(status="exclusive_observed", owned=[]),
            dict(status="contended", foreign=[{"pid": 987654}], owned=[])])
        self.assertEqual(read_json(self.output / "calibrate/worker.json")["status"], "passed")
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["eligible"])
        self.assertEqual(result["supervisor_failure"]["error_type"], "GPUIsolationError")
        self.assertIsNone(read_json(self.ledger)["active"])

    def test_worker_admission_binds_archive_device_deadline_and_process_group(self):
        source = self.fake_archive(self.output, "calibrate", self.runtime)
        args = SimpleNamespace(ledger=self.ledger, output=self.output, phase="calibrate", session_id="unit",
                               deadline=100.0, source_manifest=Path(source["execution_root"]) / "manifest.json")
        with Allocation(self.ledger, historical_path=self.history) as budget:
            budget.begin(phase="calibrate", identity=dict(output=str(self.output), phase="calibrate", gpu_uuid="GPU-A",
                runtime_config_sha256=digest(self.runtime), main_config_sha256=digest(self.config),
                deadline_monotonic=100.0, execution_root=source["execution_root"], source_sha256=source["source_sha256"]))
            budget.data["active"].update(session_id="unit", supervisor_pid=10, child_pgid=20)
            budget.save()
        arguments = dict(getpid=lambda: 20, getpgrp=lambda: 20, getppid=lambda: 10,
                         monotonic=lambda: 1.0, environ={"CUDA_VISIBLE_DEVICES": "GPU-A"},
                         module_file=str(Path(source["execution_root"]) / "python/ditserve/scheduling/worker.py"))
        self.assertEqual(worker.verify_admission(args, self.runtime, **arguments)["session_id"], "unit")
        for changed in ({"getppid": lambda: 99}, {"getpgrp": lambda: 21},
                        {"environ": {"CUDA_VISIBLE_DEVICES": "GPU-V"}}, {"module_file": __file__}):
            with self.subTest(changed=list(changed)), self.assertRaises(BudgetStop):
                worker.verify_admission(args, self.runtime, **dict(arguments, **changed))
        args.deadline = 200.0
        with self.assertRaises(BudgetStop):
            worker.verify_admission(args, self.runtime, **arguments)

    def test_failed_admission_does_not_import_execution(self):
        config_path = self.root / "runtime.json"
        atomic_json(config_path, self.runtime)
        with patch.object(worker, "verify_admission", side_effect=BudgetStop("denied")), \
             patch.object(worker.importlib, "import_module") as importer, \
             patch("builtins.print"):
            code = worker.main(["--config", str(config_path), "--output", str(self.output), "--phase", "calibrate",
                "--deadline", "100", "--ledger", str(self.ledger), "--session-id", "unit",
                "--source-manifest", str(self.root / "manifest.json")])
        self.assertEqual(code, 2)
        importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
