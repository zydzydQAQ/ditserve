from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.sessions import runner, worker
from ditserve.sessions.budget import Allocation, AUTHORIZATION, WatchdogStop

REAL_CPU_READINESS = runner.cpu_readiness

class SessionRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output, self.ledger = self.root / "run", self.root / "gpu-budget.json"
        self.history = self.root / "old.json"
        atomic_json(self.history, dict(used_gpu_seconds=43000., limit_gpu_seconds=43200.))
        self.config = dict(budget_gpu_seconds=None, authorization=AUTHORIZATION, device_preference=["V100"],
            gpu_uuid=runner.GPU_UUID, python=sys.executable, v100_python=sys.executable,
            torch_home=str(self.root / "torch"), models={"llm": {}, "sana": {}}, seed=301,
            phases=list(runner.PHASES), phase_watchdog_seconds=172800)
        self.runtime = dict(self.config, device="cuda:0", gpu_name="Tesla V100-PCIE-32GB",
                            llm_dtype="float16", sana_profile="v100_fp32",
                            main_config_sha256=digest(self.config))
        self.v100 = dict(uuid=runner.GPU_UUID, index=1, name="Tesla V100-PCIE-32GB", compute_capability="7.0")
        self.a100 = dict(uuid="GPU-A100", index=0, name="NVIDIA A100", compute_capability="8.0")
        self.mono = 1.0
        replacement = patch.object(runner, "Allocation", side_effect=lambda path: Allocation(
            path, historical_paths=(self.history,), monotonic=lambda: self.mono))
        replacement.start()
        self.addCleanup(replacement.stop)
        upstream = patch.object(runner, "upstream_completion", return_value={"status": "passed"})
        upstream.start()
        self.addCleanup(upstream.stop)
        ready = patch.object(runner, "cpu_readiness", return_value=dict(status="passed", source_files={}))
        ready.start()
        self.addCleanup(ready.stop)

    def fake_archive(self, output, phase, runtime):
        root = Path(output) / phase / "sources"
        module = root / "python/ditserve/sessions/worker.py"
        module.parent.mkdir(parents=True)
        module.write_text("# frozen test worker\n", encoding="utf-8")
        files = {"python/ditserve/sessions/worker.py": file_digest(module)}
        record = dict(phase=phase, execution_root=str(root), source_files=files,
                      source_sha256=digest(files), runtime_config_sha256=digest(runtime))
        record["sha256"] = digest(record)
        atomic_json(root / "manifest.json", record)
        return record

    def prerequisite(self, phase, **changes):
        result = dict(status="passed", eligible=True, returncode=0, main_config_sha256=digest(self.config),
                      runtime_config_sha256=digest(self.runtime))
        result.update(changes)
        atomic_json(self.output / phase / "supervisor.json", result)
        if phase == "qualify":
            atomic_json(self.output / phase / "runtime-config.json", self.runtime)

    def execute_mocked(self, *, cleanup=True, result=None, isolation=None, spawn_error=None):
        result = result or dict(status="passed", eligible=False, technical_complete=True)
        owner = self
        class Child:
            pid = 123456
            def wait(self, timeout=None):
                owner.mono += 100
                session = read_json(owner.ledger)["active"]["session_id"]
                atomic_json(owner.output / "qualify/worker.json", dict(
                    result, phase="qualify", session_id=session))
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
            value = runner.execute(self.config, self.output, self.ledger, "qualify")
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

    def llm_assets(self):
        path = self.root / "llm"
        path.mkdir()
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            atomic_json(path / name, {})
        return path

    def test_inventory_checks_only_selected_safetensors_index_and_records_its_hash(self):
        path = self.llm_assets()
        index = path / "model.safetensors.index.json"
        atomic_json(index, dict(weight_map={"a": "model-00001.safetensors", "b": "model-00002.safetensors"}))
        for name in ("model-00001.safetensors", "model-00002.safetensors"):
            (path / name).write_bytes(b"unit weights")
        # A stale alternate-format index must not require unselected .bin shards.
        atomic_json(path / "pytorch_model.bin.index.json", dict(weight_map={"a": "missing.bin"}))
        result = runner.model_inventory("llm", {"path": str(path)})
        self.assertTrue(result["available"])
        self.assertEqual(result["weight_format"], "safetensors")
        self.assertEqual(result["chosen_index"], dict(path=str(index), sha256=file_digest(index)))
        self.assertEqual(result["weight_files"], ["model-00001.safetensors", "model-00002.safetensors"])
        (path / "model-00002.safetensors").unlink()
        self.assertEqual(runner.model_inventory("llm", {"path": str(path)})["missing"], ["model-00002.safetensors"])

    def test_inventory_matches_single_safe_weight_precedence_and_never_falls_back_to_bin(self):
        path = self.llm_assets()
        atomic_json(path / "model.safetensors.index.json", dict(weight_map={"a": "../unsafe.safetensors"}))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            runner.model_inventory("llm", {"path": str(path)})
        (path / "model.safetensors").write_bytes(b"unit weights")
        result = runner.model_inventory("llm", {"path": str(path)})
        self.assertTrue(result["available"])
        self.assertIsNone(result["chosen_index"])
        self.assertEqual(result["weight_files"], ["model.safetensors"])
        (path / "model.safetensors").unlink()
        (path / "model.safetensors.index.json").unlink()
        (path / "pytorch_model.bin").write_bytes(b"unused weights")
        result = runner.model_inventory("llm", {"path": str(path)})
        self.assertFalse(result["available"])
        self.assertEqual(result["missing"], ["<missing-safetensors-model-weights>"])

    def test_inventory_rejects_empty_safe_index(self):
        path = self.llm_assets()
        atomic_json(path / "model.safetensors.index.json", {"weight_map": {}})
        result = runner.model_inventory("llm", {"path": str(path)})
        self.assertFalse(result["available"])
        self.assertEqual(result["missing"], ["<missing-indexed-model-weights>"])

    def test_watchdog_is_finite_whole_work_protection_independent_of_accounting(self):
        self.assertEqual(runner.phase_watchdog(self.config, "qualify"), 172800.)
        config = dict(self.config, phase_watchdog_seconds={}, phase_estimated_seconds={"qualify": 200000})
        self.assertEqual(runner.phase_watchdog(config, "qualify"), 300000.)
        with self.assertRaisesRegex(ValueError, "complete work"):
            runner.phase_watchdog(config, "qualify", timeout=299999)
        for duration in (0, 10, float("inf"), float("nan")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                runner.phase_watchdog(self.config, "qualify", timeout=duration)

    def test_only_pinned_v100_is_considered_even_if_other_cards_are_idle(self):
        other_v100 = dict(self.v100, uuid="GPU-other-V100")
        with patch.object(runner, "discover_gpus", return_value={"devices": [self.a100, other_v100, self.v100]}), \
             patch.object(runner, "gpu_occupancy", return_value={"status": "exclusive_observed"}) as occupancy, \
             patch.object(runner, "model_inventory", return_value={"available": True}):
            result = runner.inspect(self.config)
        occupancy.assert_called_once_with([runner.GPU_UUID])
        self.assertEqual(result["runtime_config"]["gpu_uuid"], runner.GPU_UUID)
        self.assertEqual(result["runtime_config"]["sana_profile"], "v100_fp32")
        self.assertEqual(result["runtime_config"]["llm_dtype"], "float16")
        self.assertFalse(self.ledger.exists())

    def test_busy_device_never_spawns_or_starts_accounting(self):
        with patch.object(runner, "inspect", return_value={"status": "blocked"}), \
             patch.object(runner.subprocess, "Popen") as popen:
            result = runner.execute(self.config, self.output, self.ledger, "qualify")
        popen.assert_not_called()
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(self.ledger.exists())

    def test_complete_but_unqualified_prerequisites_do_not_block_declared_next_phase(self):
        for previous, phase in (("qualify", "pilot"), ("pilot", "state")):
            self.prerequisite(previous, status="stopped", eligible=previous == "qualify", technical_complete=True)
            with self.subTest(phase=phase), patch.object(runner, "inspect", return_value={"status": "blocked"}) as inspect:
                runner.execute(self.config, self.output, self.ledger, phase)
                inspect.assert_called_once()
        self.prerequisite("pilot", status="incomplete", technical_complete=False)
        with patch.object(runner, "inspect") as inspect:
            result = runner.execute(self.config, self.output, self.ledger, "state")
        inspect.assert_not_called()
        self.assertEqual(result["status"], "blocked")

    def test_failed_or_tampered_supervision_cannot_satisfy_prerequisite(self):
        for changes in ({"supervisor_failure": {"reason": "isolation"}}, {"returncode": 1},
                        {"watchdog_stopped": True}, {"main_config_sha256": "wrong"}):
            self.prerequisite("qualify", **changes)
            with self.subTest(changes=changes), patch.object(runner, "inspect") as inspect:
                runner.execute(self.config, self.output, self.ledger, "pilot")
                inspect.assert_not_called()

    def test_qualify_locks_runtime_even_when_performance_eligibility_is_false(self):
        self.prerequisite("qualify", eligible=True)
        self.assertEqual(runner._locked_runtime(self.output, self.config), self.runtime)
        atomic_json(self.output / "qualify/runtime-config.json", dict(self.runtime, python="wrong"))
        with self.assertRaisesRegex(ValueError, "runtime changed"):
            runner._locked_runtime(self.output, self.config)

    def test_prepare_binds_new_ledger_without_resetting_cost_or_old_evidence(self):
        before = self.history.read_bytes()
        result = runner.prepare(self.config, self.output, self.ledger)
        self.assertIsNone(result["remaining_gpu_seconds"])
        with Allocation(self.ledger, historical_paths=(self.history,), monotonic=lambda: self.mono) as account:
            account.begin(phase="qualify", identity={})
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
        archive = self.output / "qualify/sources"
        self.assertEqual(call.kwargs["cwd"], str(archive))
        self.assertEqual(call.kwargs["env"]["PYTHONPATH"], str(archive / "python"))
        self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], runner.GPU_UUID)
        self.assertTrue(call.kwargs["start_new_session"])
        self.assertIn("ditserve.sessions.worker", call.args[0])
        with self.assertRaisesRegex(ValueError, "already attempted"):
            runner.execute(self.config, self.output, self.ledger, "qualify")

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
        self.assertTrue((self.output / "qualify/cleanup-unverified.json").exists())

    def test_worker_admission_checks_exact_source_device_deadline_and_parent(self):
        source = self.fake_archive(self.output, "qualify", self.runtime)
        validation = dict(status="passed", main_config_sha256=digest(self.config),
                          tests={"returncode": 0}, source_files=source["source_files"])
        validation["sha256"] = digest(validation)
        args = SimpleNamespace(ledger=self.ledger, output=self.output, phase="qualify", session_id="unit",
                               deadline=100., source_manifest=Path(source["execution_root"]) / "manifest.json")
        with Allocation(self.ledger, historical_paths=(self.history,)) as account:
            account.begin(phase="qualify", identity=dict(output=str(self.output), phase="qualify",
                gpu_uuid=runner.GPU_UUID, authorization=AUTHORIZATION, runtime_config_sha256=digest(self.runtime),
                main_config_sha256=digest(self.config), deadline_monotonic=100.,
                execution_root=source["execution_root"], source_sha256=source["source_sha256"], cpu_validation=validation))
            account.data["active"].update(session_id="unit", supervisor_pid=10, child_pgid=20)
            account.save()
        arguments = dict(getpid=lambda: 20, getpgrp=lambda: 20, getppid=lambda: 10, monotonic=lambda: 1.,
                         environ={"CUDA_VISIBLE_DEVICES": runner.GPU_UUID}, executable=sys.executable,
                         module_file=str(Path(source["execution_root"]) / "python/ditserve/sessions/worker.py"))
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
            code = worker.main(["--config", str(config), "--output", str(self.output), "--phase", "qualify",
                "--deadline", "100", "--ledger", str(self.ledger), "--session-id", "unit",
                "--source-manifest", str(self.root / "manifest.json")])
        importer.assert_not_called()
        self.assertEqual(code, 2)
        self.assertTrue(read_json(self.output / "qualify/worker.json")["watchdog_stopped"])

    def test_cpu_readiness_binds_passing_tests_and_critical_sources(self):
        workspace = self.root / "cpu-workspace"
        package = workspace / "python/ditserve"
        for name in ("runner", "budget", "worker", "study", "execution", "runtime", "policy", "workload", "reporting", "state"):
            path = package / "sessions" / (name + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# test source\n")
        for name in runner.CRITICAL_SOURCES:
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# shared source\n")
        with patch.object(runner, "ROOT", workspace), patch.object(runner, "cpu_readiness", REAL_CPU_READINESS):
            self.assertEqual(runner.cpu_readiness(self.config, self.output)["status"], "waiting")
            files = runner.cpu_source_inventory()
            runner.mark_cpu_ready(self.config, self.output, source_files=files,
                                  tests=dict(command=["python", "cpu-tests"], returncode=0))
            self.assertEqual(runner.cpu_readiness(self.config, self.output)["status"], "passed")
            unrelated = package / "optimizations/change.py"
            unrelated.parent.mkdir()
            unrelated.write_text("# unrelated work\n")
            self.assertEqual(runner.cpu_readiness(self.config, self.output)["status"], "passed")
            (package / "sessions/state.py").write_text("# changed after validation\n")
            self.assertEqual(runner.cpu_readiness(self.config, self.output)["status"], "waiting")
            with self.assertRaisesRegex(ValueError, "changed during"):
                runner.mark_cpu_ready(self.config, self.output, source_files=files,
                                      tests=dict(command=["python", "cpu-tests"], returncode=0))
            (package / "sessions/state.py").unlink()
            with self.assertRaisesRegex(ValueError, "not complete"):
                runner.cpu_source_inventory()

    def test_missing_cpu_readiness_never_observes_gpu_or_spends_accounting(self):
        with patch.object(runner, "cpu_readiness", return_value=dict(status="waiting", reason="tests pending")), \
             patch.object(runner, "inspect") as inspect, patch.object(runner.subprocess, "Popen") as popen:
            result = runner.execute(self.config, self.output, self.ledger, "qualify")
        self.assertEqual(result["reason_code"], "implementation_pending")
        inspect.assert_not_called()
        popen.assert_not_called()
        self.assertFalse(self.ledger.exists())

    def test_archive_must_match_the_version_that_passed_cpu_validation(self):
        with patch.object(runner, "cpu_readiness", return_value=dict(status="passed", source_files={
                "python/ditserve/sessions/worker.py": "wrong-hash"})), self.assertRaisesRegex(ValueError, "passed CPU validation"):
            self.execute_mocked()
        self.assertIsNone(read_json(self.ledger)["active"])


if __name__ == "__main__":
    unittest.main()
