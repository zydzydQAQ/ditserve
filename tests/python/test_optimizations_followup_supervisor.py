"""Real accounting with fake GPUs/processes, entirely in temporary directories."""

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ditserve.optimizations import followup_supervisor as supervisor
from ditserve.optimizations.next_budget import NextAllocation as RealNextAllocation
from ditserve.research.common import atomic_json, digest, file_digest, read_json


class FollowupSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_root = self.root / "new/run01"
        atomic_json(self.run_root / "study.json", dict(cache_local_binding="fixture-cache-math",
            baseline_bindings=["fixture-Sana"], trace_sha256="fixture-traces"))
        self.ledger = self.root / "new/gpu-budget.json"
        self.parents = [self.root / "old/gpu-budget.json"]
        atomic_json(self.parents[0], dict(active=None, budget_unlimited=True,
            limit_gpu_seconds=None, used_gpu_seconds=1234.0))
        self.parent_bytes = self.parents[0].read_bytes()
        self.authorization = self.root / "authorization.json"
        atomic_json(self.authorization, dict(approved_by_user=True, budget_unlimited=True,
            authorized_gpu_seconds=None, scope="serving_next_validation",
            user_instruction="就按照这个继续；取消8小时GPU小时硬上限，GPU无上限，开始在线对照；继续优化"))
        self.config = dict(gpu_uuid="GPU-fixture-a100", device="cuda:0", python="fixture-python",
                           torch_home=str(self.root / "torch"), followup=dict(wait_idle_seconds=0))
        self.clock = [100.0]
        self.source = None
        self.destination = None

    @staticmethod
    def occupancy(status="exclusive_observed", owned=None, foreign=None):
        return dict(status=status, owned=[] if owned is None else owned,
                    foreign=[] if foreign is None else foreign)

    def freeze(self, config, output, phase, attempt):
        self.assertEqual(attempt, "attempt000")
        execution = Path(output) / "sources" / (phase + "-" + attempt)
        path = execution / "python/fixture.py"
        path.parent.mkdir(parents=True)
        path.write_text("PHASE = %r\n" % phase)
        files = {"python/fixture.py": file_digest(path)}
        self.source = dict(execution_root=str(execution), source_files=files,
            source_sha256=digest(files), config_sha256=digest(config), phase=phase, attempt=attempt)
        atomic_json(execution / "manifest.json", self.source)
        return self.source

    def execute(self, phase="finish_copy", *, code=0, preflight=None, popen_error=None,
                wait_error=None, cleanup=True, initial=None, during=None, final=None,
                mutate=None, malformed_result=False):
        process = MagicMock(pid=987654)
        self.process = process
        self.destination = self.run_root / phase / "attempt000"
        selected = dict(uuid=self.config["gpu_uuid"], name="NVIDIA A100-SXM4-40GB")
        preflight = dict(status="ready", selected=[selected]) if preflight is None else preflight
        initial = self.occupancy() if initial is None else initial
        during = self.occupancy(owned=[dict(pid=process.pid)]) if during is None else during
        final = self.occupancy() if final is None else final
        observations = [initial]
        if initial.get("status") == "exclusive_observed" and popen_error is None:
            observations.append(during)
        observations.append(final)

        def allocation(*args, **kwargs):
            return RealNextAllocation(*args, **kwargs, monotonic=lambda: self.clock[0],
                                      clock=lambda: self.clock[0] + 1000.0)

        def launch(command, **kwargs):
            active = read_json(self.ledger)["active"]
            self.assertIsNotNone(active)
            self.assertIsNone(active["child_pgid"])
            self.assertEqual(active["gpu_count"], 1)
            self.assertEqual(active["identity"]["gpu_uuids"], [self.config["gpu_uuid"]])
            self.assertEqual(command[command.index("--phase") + 1], phase)
            self.assertEqual(command[command.index("--supervisor-pid") + 1], str(active["supervisor_pid"]))
            self.assertEqual(command[command.index("--run-root") + 1], str(self.run_root))
            self.assertEqual(command[command.index("--output") + 1], str(self.destination))
            self.assertEqual(command[command.index("--ledger") + 1], str(self.ledger))
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], self.config["gpu_uuid"])
            self.assertEqual(kwargs["env"]["PYTHONPATH"], str(Path(self.source["execution_root"]) / "python"))
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
                self.assertEqual(kwargs["env"][name], "1")
            self.assertEqual(kwargs["env"]["TORCHINDUCTOR_CACHE_DIR"], str(self.destination / "compiler-cache"))
            self.assertEqual(kwargs["env"]["TRITON_CACHE_DIR"], str(self.destination / "triton-cache"))
            admission = read_json(self.destination / "admission.json")
            self.assertEqual(admission["schema_version"], supervisor.SCHEMA)
            self.assertEqual(admission["parent_ledgers"][0]["sha256"], file_digest(self.parents[0]))
            self.assertEqual(admission["study_binding"], dict(path=str(self.run_root / "study.json"),
                sha256=file_digest(self.run_root / "study.json")))
            self.assertEqual(admission["study_binding"], active["identity"]["study_binding"])
            self.assertEqual(file_digest(self.destination / "admission.json"), active["identity"]["admission"]["sha256"])
            self.clock[0] += 3.0
            if popen_error:
                raise popen_error
            atomic_json(self.destination / "worker.json", dict(status="completed", local_qualified=True,
                online_qualified=True, decision="值得继续", fixture_partial_metric=3.25))
            if malformed_result:
                (self.destination / "worker.json").write_text("[invalid json")
            atomic_json(self.destination / "partial.json", dict(complete=4, total=8))
            return process

        def wait(timeout):
            self.assertEqual(read_json(self.ledger)["active"]["child_pgid"], process.pid)
            self.clock[0] += 13.0
            if mutate:
                mutate()
            if wait_error:
                raise wait_error
            return code

        def stop(child):
            self.clock[0] += 7.0
            return cleanup

        process.wait.side_effect = wait
        with ExitStack() as stack:
            self.popen = stack.enter_context(patch.object(supervisor.subprocess, "Popen", side_effect=launch))
            self.stop = stack.enter_context(patch.object(supervisor, "stop_owned_group", side_effect=stop))
            self.inspect = stack.enter_context(patch.object(supervisor, "inspect_gpu", return_value=preflight))
            stack.enter_context(patch.object(supervisor, "gpu_occupancy", side_effect=observations))
            stack.enter_context(patch.object(supervisor, "freeze", side_effect=self.freeze))
            stack.enter_context(patch.object(supervisor, "NextAllocation", side_effect=allocation))
            stack.enter_context(patch.object(supervisor.time, "monotonic", side_effect=lambda: self.clock[0]))
            return supervisor.execute(self.config, phase, self.run_root, self.ledger,
                                      self.parents, self.authorization)

    def settled(self, result, status, elapsed):
        ledger = read_json(self.ledger)
        self.assertIsNone(ledger["active"])
        self.assertEqual(result["charge"], ledger["sessions"][-1])
        self.assertEqual(result["charge"]["status"], status)
        self.assertEqual(result["charge"]["gpu_seconds"], elapsed)
        self.assertEqual(result["charge"]["elapsed_seconds"], elapsed)
        self.assertEqual(read_json(self.destination / "supervisor.json"), result)
        self.assertEqual(read_json(self.destination.parent / "terminal.json"), result)
        self.assertTrue(result["cleanup_confirmed"])

    def test_success_one_device_source_admission_and_complete_lifetime_charge(self):
        result = self.execute()
        self.settled(result, "completed", 23.0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["isolation"]["observations"], 3)
        self.assertEqual(self.parents[0].read_bytes(), self.parent_bytes)
        self.assertEqual(result["fixture_partial_metric"], 3.25)

    def test_spawn_failure_is_charged_and_settled(self):
        result = self.execute(popen_error=OSError("cannot launch"))
        self.settled(result, "failed", 10.0)
        self.assertEqual(result["supervisor_failure"]["error_type"], "OSError")
        self.assertFalse(result["online_qualified"])
        self.stop.assert_called_once_with(None)

    def test_nonzero_exit_keeps_partial_and_does_not_promote(self):
        result = self.execute(code=1)
        self.settled(result, "failed", 23.0)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["local_qualified"])
        self.assertFalse(result["online_qualified"])
        self.assertEqual(read_json(self.destination / "partial.json"), dict(complete=4, total=8))

    def test_cancel_still_cleans_and_charges(self):
        result = self.execute(wait_error=KeyboardInterrupt("cancel"))
        self.settled(result, "failed", 23.0)
        self.assertEqual(result["supervisor_failure"]["error_type"], "KeyboardInterrupt")
        self.stop.assert_called_once_with(self.process)

    def test_prelaunch_contention_never_spawns(self):
        result = self.execute(initial=self.occupancy("contended", foreign=[dict(pid=42)]))
        self.settled(result, "failed", 7.0)
        self.popen.assert_not_called()
        self.stop.assert_called_once_with(None)

    def test_runtime_contention_signals_only_owned_group(self):
        result = self.execute(during=self.occupancy("contended", foreign=[dict(pid=42)]))
        self.settled(result, "failed", 10.0)
        self.process.wait.assert_not_called()
        self.process.kill.assert_not_called()
        self.process.terminate.assert_not_called()
        self.stop.assert_called_once_with(self.process)

    def test_foreign_process_at_cleanup_invalidates_success_but_settles(self):
        result = self.execute(final=self.occupancy("contended", foreign=[dict(pid=42)]))
        self.settled(result, "failed", 23.0)
        self.assertEqual(result["supervisor_failure"]["error_type"], "GPUIsolationError")

    def test_unconfirmed_cleanup_preserves_active_lease(self):
        with self.assertRaisesRegex(RuntimeError, "cleanup unconfirmed"):
            self.execute(cleanup=False)
        ledger = read_json(self.ledger)
        self.assertIsNotNone(ledger["active"])
        self.assertEqual(ledger["sessions"], [])
        record = read_json(self.destination / "supervisor.json")
        self.assertFalse(record["cleanup_confirmed"])
        self.assertIsNone(record["charge"])
        self.assertFalse((self.destination.parent / "terminal.json").exists())

    def test_unknown_cleanup_is_not_empty_ownership_proof(self):
        with self.assertRaisesRegex(RuntimeError, "cleanup unconfirmed"):
            self.execute(final=dict(status="unknown", foreign=[]))
        self.assertIsNotNone(read_json(self.ledger)["active"])

    def test_source_changed_or_missing_is_charged_failure(self):
        result = self.execute(mutate=lambda: (Path(self.source["execution_root"]) / "python/fixture.py").unlink())
        self.settled(result, "failed", 23.0)
        self.assertEqual(result["supervisor_failure"]["error_type"], "SourceIdentityError")

    def test_parent_change_still_settles_failed_actual_usage(self):
        result = self.execute(mutate=lambda: self.parents[0].unlink())
        self.settled(result, "failed", 23.0)
        self.assertIn("parent_validation_error", result["charge"])

    def test_admission_change_cannot_promote(self):
        result = self.execute(mutate=lambda: atomic_json(self.destination / "admission.json", dict(changed=True)))
        self.settled(result, "failed", 23.0)
        self.assertEqual(result["supervisor_failure"]["error_type"], "SourceIdentityError")

    def test_study_change_cannot_promote_or_avoid_settlement(self):
        result = self.execute(mutate=lambda: atomic_json(self.run_root / "study.json", dict(changed=True)))
        self.settled(result, "failed", 23.0)
        self.assertEqual(result["supervisor_failure"]["error_type"], "SourceIdentityError")

    def test_malformed_worker_result_is_charged_failure(self):
        result = self.execute(malformed_result=True)
        self.settled(result, "failed", 23.0)
        self.assertEqual(result["supervisor_failure"]["error_type"], "WorkerEvidenceError")

    def test_busy_gpu_wait_check_never_creates_lease_or_child(self):
        result = self.execute(preflight=dict(status="blocked", selected=[]))
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["charge"])
        self.assertFalse(self.ledger.exists())
        self.assertFalse(self.destination.exists())
        self.popen.assert_not_called()
        self.stop.assert_not_called()

    def test_existing_attempt_is_not_overwritten(self):
        first = self.execute()
        before = (self.destination / "supervisor.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual((self.destination / "supervisor.json").read_bytes(), before)
        self.assertEqual(read_json(self.ledger)["used_gpu_seconds"], first["charge"]["gpu_seconds"])

    def test_ordered_stages_have_separate_same_config_leases(self):
        with self.assertRaisesRegex(ValueError, "earlier"):
            self.execute("sana_combo")
        first = self.execute()
        second = self.execute("sana_combo")
        self.settled(second, "completed", 23.0)
        self.assertEqual(second["config_sha256"], first["config_sha256"])
        self.assertNotEqual(second["source_sha256"], first["source_sha256"])
        ledger = read_json(self.ledger)
        self.assertEqual(ledger["used_gpu_seconds"], 46.0)
        self.assertEqual([row["stage"] for row in ledger["sessions"]], ["finish_copy", "sana_combo"])

    def test_active_parent_prevents_any_gpu_work(self):
        parent = read_json(self.parents[0])
        parent["active"] = dict(worker="other owned study")
        atomic_json(self.parents[0], parent)
        with self.assertRaisesRegex(RuntimeError, "historical allocation"):
            self.execute()
        self.popen.assert_not_called()

    def test_wrong_gpu_model_and_history_tree_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "selected A100"):
            self.execute(preflight=dict(status="ready", selected=[dict(uuid=self.config["gpu_uuid"], name="V100")]))
        with self.assertRaisesRegex(ValueError, "historical"):
            supervisor.execute(self.config, "finish_copy", self.root / "old/run02",
                self.root / "old/gpu-budget.json", self.parents, self.authorization)


if __name__ == "__main__":
    unittest.main()
