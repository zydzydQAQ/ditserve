"""No-GPU tests for storage-only replay and independently charged supervision."""
from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
from types import FunctionType
import unittest
from unittest.mock import MagicMock, patch

from ditserve.optimizations import followup_combo_retry as retry
from ditserve.optimizations import followup_supervisor as supervisor
from ditserve.optimizations.next_budget import NextAllocation
from ditserve.research.common import atomic_json, digest, file_digest, read_json


def scope_probe(raise_error=False):
    values = (STAGES, freeze)  # Supplied only in this fixture's private namespace.
    if raise_error:
        raise RuntimeError("private probe interrupted")
    return values


class RetryFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.parent = self.base / "old/run01"
        self.parent.mkdir(parents=True)
        self.work = self.base / "new/run01"
        self.work.mkdir(parents=True)
        self.original = dict(artifact_root=str(self.parent / "raw-arrays"),
            gpu_uuid="GPU-fixture-a100", device="cuda:0", python="fixture-python",
            torch_home=str(self.base / "torch"), steps=32,
            followup=dict(wait_idle_seconds=0, seed=101, expected_replays=8))
        self.original_config = self.parent / "config.json"
        atomic_json(self.original_config, self.original)
        self.config = retry.storage_config(self.original, self.work)
        atomic_json(self.work / "protocol.json", dict(config=self.config, sha256=digest(self.config)))
        self.snapshot = self.parent / "sources/sana_combo-attempt000"
        self.snapshot.mkdir(parents=True)
        source_file = self.snapshot / "python/ditserve/optimizations/followup_study.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("# original worker fixture; never executed\n")
        self.source_files = {str(source_file.relative_to(self.snapshot)): file_digest(source_file)}
        self.manifest_value = dict(execution_root=str(self.snapshot), source_files=self.source_files,
            source_sha256=digest(self.source_files), config_sha256=digest(self.original),
            phase="sana_combo", attempt="attempt000")
        self.manifest_path = self.snapshot / "manifest.json"
        atomic_json(self.manifest_path, self.manifest_value)
        self.controller = self.base / "controller.py"
        self.controller.write_text("# retry controller\n")
        self.control = self.base / "frozen-control.json"
        atomic_json(self.control, dict(preserve="original failure"))
        self.callbacks = []
        self.factory = self.make_factory()

    def make_factory(self):
        return retry.RetryFreezer(retry.binding(self.manifest_path),
            retry.binding(self.original_config), [retry.binding(self.controller)],
            controls=[retry.binding(self.control)], on_freeze=lambda: self.callbacks.append("frozen"))

    def derive(self, **changes):
        values = dict(config=self.config, run_root=self.work, phase="sana_combo", attempt="attempt000")
        values.update(changes)
        return self.factory(**values)

    def rewrite_manifest(self, **changes):
        self.manifest_value.update(changes)
        atomic_json(self.manifest_path, self.manifest_value)
        self.factory = self.make_factory()


class RetryFreezerTests(RetryFixture):
    def test_only_artifact_root_changes_and_input_is_not_mutated(self):
        previous = copy.deepcopy(self.original)
        value = retry.storage_config(self.original, self.work)
        self.assertEqual(self.original, previous)
        self.assertEqual(value["artifact_root"], str(self.work / "raw-arrays"))
        self.assertIsNot(value["followup"], self.original["followup"])
        self.assertIs(retry.validate_storage_config(value, self.original, self.work), value)

    def test_scientific_or_wrong_storage_change_is_rejected(self):
        alternatives = [dict(self.config, steps=16), dict(self.config, artifact_root=self.original["artifact_root"]),
                        dict(self.config, followup=dict(self.config["followup"], seed=102))]
        for value in alternatives:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.derive(config=value)
        self.assertFalse((self.work / "sources").exists())

    def test_frozen_bytes_are_copied_with_explicit_new_config_identity(self):
        before = {str(p): p.read_bytes() for p in self.parent.rglob("*") if p.is_file()}
        mutable = self.base / "workspace/python/ditserve/optimizations"
        mutable.mkdir(parents=True)
        (mutable / "followup_study.py").write_text("# workspace edit must never enter the worker\n")
        value = self.derive()
        self.assertEqual(value["source_files"], self.source_files)
        self.assertEqual(value["source_sha256"], digest(self.source_files))
        self.assertEqual(value["config_sha256"], digest(self.config))
        self.assertNotEqual(value["config_sha256"], digest(self.original))
        self.assertTrue(value["worker_source_unchanged"])
        self.assertEqual(value["storage_only_change"], dict(field="artifact_root",
            before=self.original["artifact_root"], after=self.config["artifact_root"]))
        for name, sha in self.source_files.items():
            self.assertEqual(file_digest(Path(value["execution_root"]) / name), sha)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.parent.rglob("*") if p.is_file()})
        self.assertEqual(self.callbacks, ["frozen"])

    def test_manifest_original_config_implementation_and_control_are_bound(self):
        for path in [self.manifest_path, self.original_config, self.controller, self.control]:
            previous = path.read_bytes()
            path.write_bytes(previous + b"\n")
            try:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    self.derive()
            finally:
                path.write_bytes(previous)
        self.assertEqual(self.callbacks, [])

    def test_frozen_source_bytes_and_manifest_digest_are_checked(self):
        source = self.snapshot / next(iter(self.source_files))
        old = source.read_bytes()
        source.write_bytes(b"modified source")
        with self.assertRaises(ValueError):
            self.derive()
        source.write_bytes(old)
        self.rewrite_manifest(source_sha256="0" * 64)
        with self.assertRaises(ValueError):
            self.derive()

    def test_original_config_digest_must_match_snapshot(self):
        self.rewrite_manifest(config_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "source/config"):
            self.derive()

    def test_protocol_both_content_and_digest_must_match(self):
        for value in [dict(config=dict(self.config, steps=16), sha256=digest(self.config)),
                      dict(config=self.config, sha256="0" * 64)]:
            atomic_json(self.work / "protocol.json", value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.derive()

    def test_no_other_stage_or_reused_attempt(self):
        for values in [dict(phase="finish_copy"), dict(phase="granularity"),
                       dict(phase="../outside"), dict(attempt="attempt001")]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.derive(**values)

    def test_existing_target_keeps_prior_evidence(self):
        value = self.derive()
        target = Path(value["execution_root"])
        before = {str(p): p.read_bytes() for p in target.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError):
            self.derive()
        self.assertEqual(before, {str(p): p.read_bytes() for p in target.rglob("*") if p.is_file()})

    def test_source_absolute_and_traversing_paths_fail_closed(self):
        outside = self.base / "outside.py"
        outside.write_text("outside source")
        for name in [str(outside), "../../../outside.py"]:
            files = {name: file_digest(outside)}
            self.rewrite_manifest(source_files=files, source_sha256=digest(files))
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.derive()

    def test_source_file_symlink_is_not_allowed_even_with_identical_bytes(self):
        source = self.snapshot / next(iter(self.source_files))
        outside = self.base / "copy.py"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.derive()

    def test_source_root_symlink_fails_closed(self):
        alias = self.base / "snapshot-link"
        alias.symlink_to(self.snapshot, target_is_directory=True)
        self.rewrite_manifest(execution_root=str(alias))
        with self.assertRaises(ValueError):
            self.derive()

    def test_target_parent_and_work_symlinks_do_not_create_external_files(self):
        outside = self.base / "outside-destination"
        outside.mkdir()
        link = self.work / "sources"
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.derive()
        self.assertEqual(list(outside.iterdir()), [])
        link.unlink()
        alias = self.base / "work-link"
        alias.symlink_to(self.work, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.derive(run_root=alias)

    def test_protocol_and_control_symlinks_fail_closed(self):
        for path in [self.work / "protocol.json", self.control]:
            old = path.read_bytes()
            outside = self.base / (path.name + ".outside")
            outside.write_bytes(old)
            path.unlink()
            path.symlink_to(outside)
            try:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    self.derive()
            finally:
                path.unlink()
                path.write_bytes(old)

    def test_private_function_namespace_never_mutates_shared_state(self):
        original = FunctionType(scope_probe.__code__, dict(scope_probe.__globals__,
            STAGES=("original-stage",), freeze="original-freezer"), "original_probe", scope_probe.__defaults__)
        scope = dict(original.__globals__)
        stages, freezer = supervisor.STAGES, supervisor.freeze
        isolated = retry.isolated_executor(self.factory, implementation=original)
        self.assertEqual(isolated(), (("sana_combo",), self.factory))
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            isolated(True)
        self.assertEqual(original.__globals__, scope)
        self.assertEqual(original(), (("original-stage",), "original-freezer"))
        self.assertIs(supervisor.STAGES, stages)
        self.assertIs(supervisor.freeze, freezer)


class ComboRetrySupervisorTests(RetryFixture):
    def setUp(self):
        super().setUp()
        self.ledger = self.work.parent / "gpu-budget.json"
        self.parent_ledger = self.parent.parent / "gpu-budget.json"
        atomic_json(self.parent_ledger, dict(active=None, budget_unlimited=True,
            limit_gpu_seconds=None, used_gpu_seconds=11661.441948581021))
        self.parent_bytes = self.parent_ledger.read_bytes()
        self.auth = self.work.parent / "authorization.json"
        atomic_json(self.auth, dict(approved_by_user=True, budget_unlimited=True,
            authorized_gpu_seconds=None, scope="serving_next_validation",
            user_instruction="就按照这个继续；取消8小时GPU小时硬上限，GPU无上限，开始在线对照；补齐一下"))
        atomic_json(self.work / "study.json", dict(cache_local_binding="bound original cache",
            baseline_bindings=[], trace_sha256="fixed arrivals and deadlines",
            ledger=str(self.ledger), phases=["sana_combo"], original_samples_reused=0))
        self.clock = [100.]
        self.destination = self.work / "sana_combo/attempt000"

    @staticmethod
    def occupancy(status="exclusive_observed", owned=None, foreign=None):
        return dict(status=status, owned=[] if owned is None else owned,
            foreign=[] if foreign is None else foreign)

    def execute(self, *, busy=False, during=None, final=None, worker_status="completed", wait_error=None):
        process = self.process = MagicMock(pid=987654)
        stage_before, freeze_before = supervisor.STAGES, supervisor.freeze
        observations = [self.occupancy(), during or self.occupancy(owned=[dict(pid=process.pid)]),
                        final or self.occupancy()]

        def allocation(*args, **kwargs):
            return NextAllocation(*args, **kwargs, monotonic=lambda: self.clock[0],
                clock=lambda: self.clock[0] + 1000.)

        def launch(command, **kwargs):
            value = read_json(self.ledger)
            self.assertEqual(value["stages"], ["sana_combo"])
            self.assertEqual(value["sessions"], [])
            self.assertEqual(value["active"]["identity"]["phase"], "sana_combo")
            self.assertIsNone(value["active"]["identity"]["retry_from"])
            self.assertEqual(command[command.index("--phase") + 1], "sana_combo")
            self.assertFalse((self.work / "finish_copy").exists())
            self.assertFalse((self.work / "granularity").exists())
            expected = self.work / "sources/sana_combo-attempt000/python"
            self.assertEqual(kwargs["env"]["PYTHONPATH"], str(expected))
            self.assertTrue(kwargs["start_new_session"])
            self.clock[0] += 3.
            atomic_json(self.destination / "worker.json", dict(status=worker_status,
                reason="fixture OOM" if worker_status == "failed" else None,
                decision="值得继续", fixture_metric=1.25))
            atomic_json(self.destination / "partial.json", dict(complete=2, total=8))
            return process

        def wait(timeout):
            self.assertEqual(read_json(self.ledger)["active"]["child_pgid"], process.pid)
            self.clock[0] += 13.
            if wait_error:
                raise wait_error
            return 0 if worker_status == "completed" else 1

        def stop(child):
            self.clock[0] += 7.
            return True

        process.wait.side_effect = wait
        preflight = dict(status="blocked", selected=[]) if busy else dict(status="ready",
            selected=[dict(uuid=self.config["gpu_uuid"], name="NVIDIA A100 40GB")])
        with ExitStack() as stack:
            self.launch = stack.enter_context(patch.object(supervisor.subprocess, "Popen", side_effect=launch))
            self.stop = stack.enter_context(patch.object(supervisor, "stop_owned_group", side_effect=stop))
            stack.enter_context(patch.object(supervisor, "inspect_gpu", return_value=preflight))
            stack.enter_context(patch.object(supervisor, "gpu_occupancy", side_effect=observations))
            stack.enter_context(patch.object(supervisor, "NextAllocation", side_effect=allocation))
            stack.enter_context(patch.object(supervisor.time, "monotonic", side_effect=lambda: self.clock[0]))
            execute = retry.isolated_executor(self.factory)
            try:
                return execute(self.config, "sana_combo", self.work, self.ledger,
                    (self.parent_ledger,), self.auth)
            finally:
                self.assertIs(supervisor.STAGES, stage_before)
                self.assertIs(supervisor.freeze, freeze_before)

    def settled(self, result, *, status, seconds):
        ledger = read_json(self.ledger)
        self.assertIsNone(ledger["active"])
        self.assertEqual(ledger["stages"], ["sana_combo"])
        self.assertEqual(len(ledger["sessions"]), 1)
        self.assertEqual(ledger["used_gpu_seconds"], seconds)
        self.assertEqual(ledger["sessions"][0], result["charge"])
        self.assertEqual(result["charge"]["status"], status)
        self.assertEqual(self.parent_ledger.read_bytes(), self.parent_bytes)
        self.assertEqual(read_json(self.destination / "supervisor.json"), result)
        self.assertTrue(result["cleanup_confirmed"])

    def test_real_supervisor_and_accounting_run_only_sana_without_prior_fake_stages(self):
        result = self.execute()
        self.settled(result, status="completed", seconds=23.)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["fixture_metric"], 1.25)
        self.assertEqual(self.callbacks, ["frozen"])
        self.assertEqual(result["source_sha256"], digest(self.source_files))
        self.assertEqual(result["config_sha256"], digest(self.config))

    def test_busy_gpu_starts_no_process_or_accounting_lease(self):
        result = self.execute(busy=True)
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["gpu_work_started"])
        self.assertFalse(self.ledger.exists())
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.parent_ledger.read_bytes(), self.parent_bytes)
        self.launch.assert_not_called()
        self.stop.assert_not_called()

    def test_unknown_runtime_ownership_stops_only_owned_group_and_keeps_failure(self):
        unknown = self.occupancy("unknown")
        unknown["reason"] = "nvidia-smi fixture timeout"
        result = self.execute(during=unknown)
        self.settled(result, status="failed", seconds=10.)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["reason"], "A100 isolation lost: unknown")
        self.assertFalse(result["online_qualified"])
        self.assertFalse(result["local_qualified"])
        self.stop.assert_called_once_with(self.process)
        self.process.wait.assert_not_called()
        self.assertEqual(read_json(self.destination / "partial.json"), dict(complete=2, total=8))

    def test_unknown_cleanup_keeps_active_charge_and_does_not_claim_completion(self):
        with self.assertRaisesRegex(RuntimeError, "cleanup unconfirmed"):
            self.execute(final=self.occupancy("unknown"))
        ledger = read_json(self.ledger)
        self.assertIsNotNone(ledger["active"])
        self.assertEqual(ledger["sessions"], [])
        self.assertFalse(read_json(self.destination / "supervisor.json")["cleanup_confirmed"])
        self.assertEqual(self.parent_ledger.read_bytes(), self.parent_bytes)

    def test_worker_failure_is_charged_and_partial_evidence_remains(self):
        result = self.execute(worker_status="failed")
        self.settled(result, status="failed", seconds=23.)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["reason"], "fixture OOM")
        self.assertEqual(read_json(self.destination / "partial.json")["complete"], 2)

    def test_cancellation_cleans_and_charges_without_global_scope_changes(self):
        result = self.execute(wait_error=KeyboardInterrupt("cancel fixture"))
        self.settled(result, status="failed", seconds=23.)
        self.assertEqual(result["supervisor_failure"]["error_type"], "KeyboardInterrupt")
        self.stop.assert_called_once_with(self.process)

    def test_existing_retry_attempt_cannot_be_reused(self):
        first = self.execute()
        bytes_before = self.ledger.read_bytes()
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(self.ledger.read_bytes(), bytes_before)
        self.assertEqual(read_json(self.destination / "supervisor.json"), first)

    def test_active_parent_blocks_before_launch(self):
        value = read_json(self.parent_ledger)
        value["active"] = dict(stage="other")
        atomic_json(self.parent_ledger, value)
        with self.assertRaisesRegex(RuntimeError, "historical allocation"):
            self.execute()
        self.launch.assert_not_called()
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
