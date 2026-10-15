"""Real NextAllocation + next-study supervisor; all GPU/process calls are fake.

Every ledger, authorization, source snapshot, and worker record is confined to
a temporary directory. These fixtures never load torch, models, or GPU tools.
"""
from contextlib import ExitStack
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ditserve.optimizations import next_study as study
from ditserve.optimizations.next_budget import NextAllocation as RealNextAllocation
from ditserve.optimizations.budget import BudgetStop
from ditserve.research.common import atomic_json, digest, file_digest, read_json


class NextSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "new/run01"
        self.ledger = self.output.parent / "gpu-budget.json"
        self.parents = tuple(self.root / (name + "/gpu-budget.json")
                             for name in ("old24", "old8", "old_unlimited"))
        for path, used, limit in zip(self.parents, (40000., 22000., 8044.), (86400., 28800., None)):
            atomic_json(path, dict(active=None, used_gpu_seconds=used, limit_gpu_seconds=limit,
                                  budget_unlimited=limit is None))
        self.original_parents = {p: p.read_bytes() for p in self.parents}
        self.auth = self.root / "authorization.json"
        atomic_json(self.auth, dict(approved_by_user=True, budget_unlimited=True, authorized_gpu_seconds=None,
            scope="serving_next_validation", user_instruction="就按照这个继续；取消8小时GPU小时硬上限，GPU无上限，开始在线对照"))
        self.baseline = self.root / "qualified-baseline.json"
        atomic_json(self.baseline, dict(online_qualified=True, fixture=True))
        self.old_cache = self.root / "qualified-local.json"
        atomic_json(self.old_cache, dict(local_qualified=True, fixture=True))
        self.math_source = b"CACHE_MATH = 'unchanged fixture arithmetic'\n"
        hashes = {"python/cache_arithmetic.py": hashlib.sha256(self.math_source).hexdigest()}
        self.config = dict(gpu_uuid="GPU-fixture-a100", python="fixture-python", torch_home=str(self.root / "torch"))
        self.workloads = dict(frozen=True, fixture_requests=["same-arrivals-and-seeds"])
        self.prepared = dict(workloads=self.workloads, workloads_sha256=digest(self.workloads),
            cache_local_binding=dict(mathematical_source_files=hashes, mathematical_source_sha256=digest(hashes),
                bindings=dict(local=dict(path=str(self.old_cache), sha256=file_digest(self.old_cache)))),
            online_baseline_binding=dict(path=str(self.baseline), sha256=file_digest(self.baseline)),
            historical_ledgers=[dict(path=str(p), sha256=file_digest(p)) for p in self.parents])
        self.now = [100.]
        self.phases = ("diagnosis", "sana_vae")
        self.executions = []
        self.sources = []

    @staticmethod
    def occupancy(status="exclusive_observed", owned=None, foreign=None):
        return dict(status=status, owned=[] if owned is None else owned,
                    foreign=[] if foreign is None else foreign)

    def freeze(self, config, output, phase, attempt):
        root = Path(output) / "sources" / (phase + "-" + attempt)
        source_file = root / "python/fixture_runtime.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("PHASE = %r\nATTEMPT = %r\n" % (phase, attempt))
        (root / "python/cache_arithmetic.py").write_bytes(self.math_source)
        files = {"python/fixture_runtime.py": file_digest(source_file),
                 "python/cache_arithmetic.py": file_digest(root / "python/cache_arithmetic.py")}
        manifest = dict(execution_root=str(root), source_files=files, source_sha256=digest(files))
        atomic_json(root / "manifest.json", manifest)
        self.sources.append((manifest, source_file))
        return manifest

    def execute(self, *, phase="diagnosis", code=0, wait_error=None, popen_error=None,
                observations=None, final=None, cleanup=True, mutate=None,
                retry_from=None, preflight="ready", conditional_trigger=None, auxiliary_ready=True):
        process = MagicMock(pid=987654)
        self.last_process = process
        self.last_destination = None
        self.partial_bytes = None
        final = self.occupancy() if final is None else final
        # Popen failure proceeds directly to the final occupancy observation.
        observations = ([] if popen_error else [self.occupancy(owned=[dict(pid=process.pid)])]) if observations is None else observations
        occupancy_values = list(observations) + [final]
        if conditional_trigger is not None:
            occupancy_values.insert(0, self.occupancy() if auxiliary_ready else
                self.occupancy("contended", foreign=[dict(pid=123, uuid=self.config["v100_uuid"])]))

        def allocation(*args, **kwargs):
            return RealNextAllocation(*args, **kwargs, monotonic=lambda: self.now[0], clock=lambda: 1000. + self.now[0])

        def started(command, **kwargs):
            active = read_json(self.ledger)["active"]
            self.assertIsNotNone(active)  # Accounted before loading/Popen.
            self.assertIsNone(active["child_pgid"])
            self.assertTrue(active["identity"]["budget_unlimited"])
            self.assertNotIn("--deadline", command)
            self.assertEqual(command[command.index("--ledger") + 1], str(self.ledger))
            self.assertEqual(command[command.index("--session-id") + 1], active["session_id"])
            self.assertTrue(kwargs["start_new_session"])
            expected_devices = [self.config["gpu_uuid"]] + ([self.config["v100_uuid"]]
                if conditional_trigger is not None and auxiliary_ready else [])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], ",".join(expected_devices))
            self.assertEqual(active["gpu_count"], len(expected_devices))
            self.assertEqual("--auxiliary-encoder" in command, len(expected_devices) == 2)
            self.assertEqual(kwargs["env"]["PYTHONPATH"], str(Path(self.sources[-1][0]["execution_root"]) / "python"))
            self.executions.append(dict(command=command, env=kwargs["env"]))
            self.now[0] += 3.
            if popen_error:
                raise popen_error
            destination = Path(command[command.index("--output") + 1])
            self.last_destination = destination
            atomic_json(destination / "partial.json", dict(status="running", requests=[dict(request_id="r000", status="completed")]))
            self.partial_bytes = (destination / "partial.json").read_bytes()
            atomic_json(destination / "worker.json", dict(status="passed", local_qualified=True,
                online_qualified=True, online_evaluation_complete=True,
                decision="值得继续", candidate="fixture-candidate", evidence_path=str(destination / "partial.json")))
            return process

        def wait(timeout):
            active = read_json(self.ledger)["active"]
            self.assertEqual(active["child_pgid"], process.pid)
            self.now[0] += 13.
            if mutate:
                mutate()
            if wait_error:
                raise wait_error
            return code

        def stop(value):
            self.now[0] += 7.  # Cleanup time is billed too.
            self.assertIs(value, None if popen_error else process)
            return cleanup

        process.wait.side_effect = wait
        with ExitStack() as stack:
            stack.enter_context(patch.object(study, "PARENTS", self.parents))
            stack.enter_context(patch.object(study, "PHASES", self.phases))
            stack.enter_context(patch.object(study, "inspect_study", return_value=copy.deepcopy(self.prepared)))
            stack.enter_context(patch("ditserve.optimizations.runner.inspect", side_effect=lambda config: dict(status=preflight,
                devices=dict(devices=[dict(uuid=config["gpu_uuid"])] +
                    ([dict(uuid=config["v100_uuid"])] if config.get("v100_uuid") else [])))))
            if conditional_trigger is not None:
                stack.enter_context(patch("ditserve.optimizations.next_conditional.select_trigger_evidence", return_value=conditional_trigger))
            stack.enter_context(patch.object(study, "freeze", side_effect=self.freeze))
            stack.enter_context(patch.object(study, "gpu_occupancy", side_effect=occupancy_values))
            self.last_stop = stack.enter_context(patch.object(study, "stop_owned_group", side_effect=stop))
            self.last_popen = stack.enter_context(patch.object(study.subprocess, "Popen", side_effect=started))
            stack.enter_context(patch("ditserve.optimizations.next_budget.NextAllocation", side_effect=allocation))
            stack.enter_context(patch.object(study.time, "monotonic", side_effect=lambda: self.now[0]))
            return study.execute(self.config, self.output, self.ledger, phase, self.auth,
                                 wait_idle=0., retry_from=retry_from)

    def assert_settled(self, result, *, elapsed, status):
        ledger = read_json(self.ledger)
        self.assertIsNone(ledger["active"])
        self.assertEqual(result["charge"], ledger["sessions"][-1])
        self.assertEqual(result["charge"]["gpu_seconds"], elapsed)
        self.assertEqual(result["charge"]["status"], status)
        self.assertTrue(ledger["budget_unlimited"])
        self.assertIsNone(ledger["limit_gpu_seconds"])
        self.assertEqual(ledger["used_gpu_seconds"], sum(r["gpu_seconds"] for r in ledger["sessions"]))

    def test_conditional_reserves_and_bills_both_admitted_gpus(self):
        self.phases = ("conditional",)
        self.config["v100_uuid"] = "GPU-fixture-v100"
        trigger = dict(encoder_trigger=True, vae_trigger=False, admitted_sources=[{}],
            immutable_bindings=[dict(path=str(self.old_cache), sha256=file_digest(self.old_cache))])
        result = self.execute(phase="conditional", conditional_trigger=trigger)
        self.assert_settled(result, elapsed=46., status="completed")
        self.assertEqual(result["charge"]["elapsed_seconds"], 23.)
        self.assertEqual(result["charge"]["gpu_count"], 2)
        admission = read_json(self.last_destination / "conditional-admission.json")
        self.assertTrue(admission["auxiliary_encoder"])
        self.assertEqual(admission["devices"], [self.config["gpu_uuid"], self.config["v100_uuid"]])

    def test_busy_v100_is_never_admitted_or_charged(self):
        self.phases = ("conditional",)
        self.config["v100_uuid"] = "GPU-fixture-v100"
        trigger = dict(encoder_trigger=True, vae_trigger=True, admitted_sources=[{}],
            immutable_bindings=[dict(path=str(self.old_cache), sha256=file_digest(self.old_cache))])
        result = self.execute(phase="conditional", conditional_trigger=trigger, auxiliary_ready=False)
        self.assert_settled(result, elapsed=23., status="completed")
        admission = read_json(self.last_destination / "conditional-admission.json")
        self.assertFalse(admission["auxiliary_encoder"])
        self.assertEqual(admission["devices"], [self.config["gpu_uuid"]])

    def assert_no_promotion_and_partial_kept(self, result):
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["decision"], "未验证")
        self.assertFalse(result["online_qualified"])
        self.assertEqual((self.last_destination / "partial.json").read_bytes(), self.partial_bytes)
        self.assertEqual(result["evidence_path"], str(self.last_destination / "partial.json"))

    def test_success_freezes_execution_source_and_writes_matching_terminal_and_charge(self):
        result = self.execute()
        self.assert_settled(result, elapsed=23., status="completed")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["source_sha256"], self.sources[-1][0]["source_sha256"])
        self.assertEqual(result["config_sha256"], digest(self.config))
        self.assertEqual(read_json(self.output / "diagnosis/terminal.json"), result)
        self.assertEqual(read_json(self.last_destination / "supervisor.json"), result)
        self.assertEqual(result["final_occupancy"]["owned"], [])
        self.assertEqual(result["final_occupancy"]["foreign"], [])
        for path, original in self.original_parents.items():
            self.assertEqual(path.read_bytes(), original)

    def test_popen_failure_is_fully_charged_and_settled_failed(self):
        result = self.execute(popen_error=OSError("fixture spawn failure"))
        self.assert_settled(result, elapsed=10., status="failed")
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["online_qualified"])
        self.assertIsNone(result["charge"]["child_pgid"])
        self.assertEqual(result["supervisor_failure"]["error_type"], "OSError")
        self.last_stop.assert_called_once_with(None)

    def test_nonzero_child_cannot_promote_candidate_or_erase_partial(self):
        result = self.execute(code=1)
        self.assert_settled(result, elapsed=23., status="failed")
        self.assert_no_promotion_and_partial_kept(result)
        self.assertEqual(result["returncode"], 1)

    def test_budgetstop_still_cleans_owned_group_charges_and_keeps_partial(self):
        result = self.execute(wait_error=BudgetStop("fixture supervisor cancellation"))
        self.assert_settled(result, elapsed=23., status="failed")
        self.assert_no_promotion_and_partial_kept(result)
        self.assertEqual(result["supervisor_failure"]["error_type"], "BudgetStop")
        self.last_stop.assert_called_once_with(self.last_process)

    def test_external_gpu_process_causes_only_owned_group_cleanup_and_retains_failure(self):
        foreign = self.occupancy(status="contended", owned=[dict(pid=987654)], foreign=[dict(pid=111111)])
        result = self.execute(observations=[foreign])
        self.assert_settled(result, elapsed=10., status="failed")
        self.assert_no_promotion_and_partial_kept(result)
        self.last_stop.assert_called_once_with(self.last_process)
        self.last_process.wait.assert_not_called()
        self.last_process.kill.assert_not_called()
        self.last_process.terminate.assert_not_called()
        self.assertIn("isolation lost", result["reason"])

    def test_foreign_process_at_cleanup_invalidates_successful_child(self):
        result = self.execute(final=self.occupancy(status="contended", foreign=[dict(pid=111111)]))
        self.assert_settled(result, elapsed=23., status="failed")
        self.assert_no_promotion_and_partial_kept(result)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["supervisor_failure"]["error_type"], "GPUIsolationError")

    def test_unconfirmed_group_cleanup_leaves_active_and_writes_no_terminal(self):
        with self.assertRaisesRegex(RuntimeError, "cleanup unconfirmed"):
            self.execute(cleanup=False)
        ledger = read_json(self.ledger)
        self.assertIsNotNone(ledger["active"])
        self.assertEqual(ledger["active"]["child_pgid"], 987654)
        self.assertEqual(ledger["sessions"], [])
        self.assertEqual(ledger["used_gpu_seconds"], 0.)
        self.assertFalse((self.output / "diagnosis/terminal.json").exists())
        self.assertEqual((self.last_destination / "partial.json").read_bytes(), self.partial_bytes)

    def test_unknown_gpu_cleanup_leaves_active(self):
        with self.assertRaisesRegex(RuntimeError, "cleanup unconfirmed"):
            self.execute(final=self.occupancy(status="unknown"))
        self.assertIsNotNone(read_json(self.ledger)["active"])

    def test_missing_cleanup_ownership_field_is_not_proof_of_empty_owned_group(self):
        with self.assertRaisesRegex(RuntimeError, "cleanup unconfirmed"):
            self.execute(final=dict(status="exclusive_observed", foreign=[]))
        self.assertIsNotNone(read_json(self.ledger)["active"])

    def test_explicit_same_stage_retry_hash_creates_new_charge_and_preserves_parent_bytes(self):
        first = self.execute(code=1)
        first_path = self.last_destination / "supervisor.json"
        first_bytes, first_hash = first_path.read_bytes(), file_digest(first_path)
        # Without explicit retry, the saved conclusion is returned for free.
        reused = self.execute()
        self.assertEqual(reused, first)
        self.last_popen.assert_not_called()
        second = self.execute(retry_from=first_path)
        self.assert_settled(second, elapsed=23., status="completed")
        self.assertEqual(second["charge"]["identity"]["retry_from"], dict(path=str(first_path), sha256=first_hash))
        self.assertEqual(read_json(self.ledger)["used_gpu_seconds"], 46.)
        self.assertNotEqual(first["charge"]["session_id"], second["charge"]["session_id"])
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertTrue(second["output"].endswith("attempt01"))
        for path, original in self.original_parents.items():
            self.assertEqual(path.read_bytes(), original)

    def test_next_stage_is_separate_session_with_same_configuration(self):
        first = self.execute()
        second = self.execute(phase="sana_vae")
        self.assert_settled(second, elapsed=23., status="completed")
        ledger = read_json(self.ledger)
        self.assertEqual([r["stage"] for r in ledger["sessions"]], list(self.phases))
        self.assertEqual(ledger["stage_used_gpu_seconds"], dict(diagnosis=23., sana_vae=23.))
        self.assertEqual(first["config_sha256"], second["config_sha256"])
        self.assertNotEqual(first["source_sha256"], second["source_sha256"])

    def test_resource_wait_never_starts_allocation_or_child(self):
        result = self.execute(preflight="waiting_resource")
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["gpu_work_started"])
        self.assertFalse(self.ledger.exists())
        self.last_popen.assert_not_called()
        self.last_stop.assert_not_called()
        self.assertEqual(self.sources, [])

    def test_source_hash_change_retains_charge_and_cannot_promote_worker(self):
        result = self.execute(mutate=lambda: self.sources[-1][1].write_text("changed fixture source\n"))
        self.assert_settled(result, elapsed=23., status="failed")
        self.assert_no_promotion_and_partial_kept(result)
        self.assertEqual(result["supervisor_failure"]["error_type"], "SourceIdentityError")

    def test_missing_source_after_cleanup_is_charged_failure_not_unsettled_session(self):
        result = self.execute(mutate=lambda: self.sources[-1][1].unlink())
        self.assert_settled(result, elapsed=23., status="failed")
        self.assert_no_promotion_and_partial_kept(result)
        self.assertEqual(result["supervisor_failure"]["error_type"], "SourceIdentityError")

    def test_missing_historical_parent_after_cleanup_still_settles_full_failed_cost(self):
        result = self.execute(mutate=lambda: self.parents[-1].unlink())
        self.assert_settled(result, elapsed=23., status="failed")
        self.assert_no_promotion_and_partial_kept(result)
        self.assertIn("parent_validation_error", result["charge"])

    def test_inherited_cache_quality_change_is_a_charged_failure(self):
        result = self.execute(mutate=lambda: atomic_json(self.old_cache, dict(local_qualified=False)))
        self.assert_settled(result, elapsed=23., status="failed")
        self.assert_no_promotion_and_partial_kept(result)
        self.assertEqual(result["supervisor_failure"]["error_type"], "SourceIdentityError")

    def test_cache_math_mismatch_before_begin_never_charges_or_spawns(self):
        self.math_source = b"CACHE_MATH = 'different fixture arithmetic'\n"
        with self.assertRaisesRegex(ValueError, "cache arithmetic differs"):
            self.execute()
        ledger = read_json(self.ledger)
        self.assertIsNone(ledger["active"])
        self.assertEqual(ledger["sessions"], [])
        self.assertEqual(ledger["used_gpu_seconds"], 0.)
        self.last_popen.assert_not_called()
        self.last_stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
