"""CPU-only checks for phase-conflict orchestration and scientific pairing.

All GPU queries, worker launches and telemetry are mocked. These tests do not
validate CUDA concurrency, model correctness or hardware performance.
"""

import builtins
import copy
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import Mock, patch

from ditserve.experiments import run_phase_conflict as runner


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "v100.json"
UUID = "GPU-a058066e-bde9-c534-efb1-d2a68c117681"
GPU = {"uuid": UUID, "physical_index": 1, "name": "Mock V100"}
HASH_A = "a" * 64
HASH_B = "b" * 64


def arguments(directory, *extra):
    return runner.parser().parse_args([
        "--output", str(Path(directory) / "suite"), "--model-id", "mock-wan",
        "--gpu-hours", "1", "--worker-timeout", "30",
        "--config", str(CONFIG),
    ] + list(extra))


def finite_profile(job):
    profile = {
        "schema_version": "ditserve.phase-conflict-worker.v1",
        "kind": "real_gpu_phase_conflict", "status": "completed", "measured": True,
        "config": copy.deepcopy(job),
        "environment": {
            "ready": True, "gpu": {"uuid": UUID}, "cuda_visible_devices": UUID,
            "cuda_device_order": "PCI_BUS_ID", "device": "cuda:0",
        },
        "output_b": {"finite": True, "sha256": HASH_B},
        "timings": {"remaining_ms": 100.0},
    }
    if job["mode"] == "case":
        profile["output_a"] = {"finite": True, "sha256": HASH_A}
    return profile


class FakeProcess:
    """Deterministic wait outcomes; never creates an operating-system process."""

    pid = 424242

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.returncode = None
        self.wait_timeouts = []

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        outcome = self.outcomes.pop(0) if self.outcomes else -9
        if isinstance(outcome, BaseException):
            raise outcome
        self.returncode = outcome
        return outcome

    def poll(self):
        return self.returncode


class BudgetTests(unittest.TestCase):
    def test_exact_budget_and_overrun_never_create_more_available_time(self):
        budget = runner.Budget(100, 25)
        self.assertEqual(budget.remaining_seconds, 75)
        self.assertTrue(budget.can_start(75))
        self.assertFalse(budget.can_start(75.001))
        budget.charge(75)
        self.assertEqual(budget.remaining_seconds, 0)
        self.assertFalse(budget.can_start(.001))
        budget.charge(2)  # Teardown may overrun a deadline; keep the real charge.
        self.assertEqual(budget.used_seconds, 102)
        self.assertEqual(budget.remaining_seconds, 0)

    def test_invalid_time_never_mutates_budget(self):
        for value in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                budget = runner.Budget(100, 4)
                with self.assertRaises(ValueError):
                    budget.charge(value)
                self.assertEqual(budget.used_seconds, 4)
                self.assertFalse(budget.can_start(value))
        self.assertFalse(runner.Budget(100).can_start(0))

    def test_invalid_initial_budget_is_rejected(self):
        for limit, used in ((0, 0), (-1, 0), (float("inf"), 0),
                            (100, -1), (100, float("nan"))):
            with self.subTest(limit=limit, used=used), self.assertRaises(ValueError):
                runner.Budget(limit, used)


class ResumeTests(unittest.TestCase):
    def test_resume_rejects_active_or_spent_diagnostics_before_any_manifest_write(self):
        for record in ({"status": "running", "elapsed_seconds": 0},
                       {"status": "completed", "elapsed_seconds": 20},
                       {"status": "oom", "elapsed_seconds": 7},
                       {"status": "failed", "elapsed_seconds": .001}):
            with self.subTest(record=record), tempfile.TemporaryDirectory() as tmp:
                args, suite = self.make_manifest(tmp, [{"id": "core", "status": "completed", "elapsed_seconds": 83.9}])
                path = suite.root / "trace-attempts.json"
                runner.atomic_json(path, {"schema_version": "ditserve.phase-conflict-trace.v1", "records": [record]})
                before, diagnostic_before = suite.path.read_bytes(), path.read_bytes()
                args.resume = True
                with patch.object(runner, "atomic_json", side_effect=AssertionError("write")), \
                        patch.object(runner, "gpu_available", side_effect=AssertionError("GPU query")), \
                        patch.object(runner.subprocess, "Popen", side_effect=AssertionError("launch")):
                    with self.assertRaisesRegex(ValueError, "diagnostic.*budget|budget.*diagnostic"):
                        runner.Suite(args)
                self.assertEqual(suite.path.read_bytes(), before)
                self.assertEqual(path.read_bytes(), diagnostic_before)

    def test_resume_rejects_malformed_diagnostic_accounting_instead_of_assuming_zero(self):
        for records in (None, {}, [None], [{"status": "failed"}],
                        [{"status": "unknown", "elapsed_seconds": 0}],
                        [{"status": "failed", "elapsed_seconds": -1}],
                        [{"status": "failed", "elapsed_seconds": True}]):
            with self.subTest(records=records), tempfile.TemporaryDirectory() as tmp:
                args, suite = self.make_manifest(tmp, [])
                runner.atomic_json(suite.root / "trace-attempts.json", {
                    "schema_version": "ditserve.phase-conflict-trace.v1", "records": records})
                args.resume = True
                with patch.object(runner, "atomic_json", side_effect=AssertionError("write")), \
                        self.assertRaisesRegex(ValueError, "diagnostic budget must not be omitted"):
                    runner.Suite(args)

    def test_old_diagnostic_charge_cannot_be_hidden_by_a_later_zero_cost_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, suite = self.make_manifest(tmp, [])
            runner.atomic_json(suite.root / "trace-attempts.json", {
                "schema_version": "ditserve.phase-conflict-trace.v1", "records": [
                    {"id": "same", "status": "failed", "elapsed_seconds": 20},
                    {"id": "same", "status": "failed", "elapsed_seconds": 0}]})
            args.resume = True
            with self.assertRaisesRegex(ValueError, "omit diagnostic charges"):
                runner.Suite(args)

    def test_broken_json_or_unknown_schema_diagnostic_ledger_blocks_resume(self):
        for contents in ("{partial JSON", '{"schema_version":"other","records":[]}'):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as tmp:
                args, suite = self.make_manifest(tmp, [])
                (suite.root / "trace-attempts.json").write_text(contents)
                before = suite.path.read_bytes()
                args.resume = True
                with self.assertRaisesRegex(ValueError, "diagnostic budget must not be omitted"):
                    runner.Suite(args)
                self.assertEqual(suite.path.read_bytes(), before)

    def test_empty_or_zero_cost_finished_diagnostics_allow_ordinary_core_accounting(self):
        for records in ([], [{"status": "failed", "elapsed_seconds": 0}]):
            with self.subTest(records=records), tempfile.TemporaryDirectory() as tmp:
                args, suite = self.make_manifest(tmp, [{"id": "core", "status": "completed", "elapsed_seconds": 83.9}])
                path = suite.root / "trace-attempts.json"
                runner.atomic_json(path, {"schema_version": "ditserve.phase-conflict-trace.v1", "records": records})
                before = path.read_bytes()
                args.resume = True
                resumed = runner.Suite(args)
                self.assertEqual(resumed.budget.used_seconds, 83.9)
                self.assertEqual(path.read_bytes(), before)

    def make_manifest(self, directory, records):
        args = arguments(directory)
        suite = runner.Suite(args)
        suite.records.extend(records)
        suite.save()
        return args, suite

    def test_resume_counts_success_failure_timeout_and_does_not_expand_budget(self):
        records = [
            {"id": "ok", "status": "completed", "elapsed_seconds": 120},
            {"id": "oom", "status": "oom", "elapsed_seconds": 47},
            {"id": "timeout", "status": "failed", "elapsed_seconds": 33},
            {"id": "skip", "status": "skipped_budget", "elapsed_seconds": 0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            args, original = self.make_manifest(tmp, records)
            args.resume = True
            args.gpu_hours = 6
            resumed = runner.Suite(args)
            self.assertEqual(resumed.budget.used_seconds, 200)
            self.assertEqual(resumed.budget.limit_seconds, 3600)
            resumed_again = runner.Suite(args)
            self.assertEqual(resumed_again.budget.used_seconds, 200)
            self.assertEqual(len(resumed_again.records), 4)
            self.assertEqual(original.manifest["created_at_utc"], resumed.manifest["created_at_utc"])

    def test_unfinished_worker_prevents_resume_and_does_not_rewrite_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, suite = self.make_manifest(tmp, [{"id": "unfinished", "status": "running", "elapsed_seconds": 0}])
            before = suite.path.read_bytes()
            args.resume = True
            with self.assertRaisesRegex(ValueError, "running"):
                runner.Suite(args)
            self.assertEqual(suite.path.read_bytes(), before)

    def test_missing_failed_attempt_elapsed_is_not_silently_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, suite = self.make_manifest(tmp, [{"id": "lost-time", "status": "failed"}])
            args.resume = True
            with self.assertRaises(ValueError):
                runner.Suite(args)

    def test_corrupt_attempt_elapsed_cannot_cancel_or_hide_a_real_charge(self):
        for invalid in (None, True, "20", -20, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runner.elapsed_from_records([
                    {"status": "completed", "elapsed_seconds": 20},
                    {"status": "failed", "elapsed_seconds": invalid},
                ])

    def test_resume_rejects_changed_protocol_or_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, suite = self.make_manifest(tmp, [])
            args.resume = True
            args.model_id = "different-model"
            with self.assertRaisesRegex(ValueError, "same model"):
                runner.Suite(args)
            args.model_id = "mock-wan"
            suite.manifest["protocol_config"]["revision"] = "different-revision"
            suite.save()
            with self.assertRaisesRegex(ValueError, "same model"):
                runner.Suite(args)


class IdentityAndCompletionTests(unittest.TestCase):
    def setUp(self):
        self.job = {
            "id": "same-f17-b17-s0-immediate-k25-r0-none", "mode": "case",
            "height": 480, "width": 832, "frames": 17, "steps": 50,
            "seed": 0, "cache": "none", "model_id": "mock-wan", "revision": "fixed",
            "strategy": "immediate", "insert_after": 25, "b_latent_path": "/mock/b.pt",
            "b_metadata_path": "/mock/b.json", "b_frames": 17,
            "repeat": 0, "preset": "v100-wan", "diagnostic": False,
        }

    def test_every_strategy_has_a_distinct_verified_identity(self):
        for strategy in runner.STRATEGIES:
            with self.subTest(strategy=strategy):
                job = dict(self.job, strategy=strategy, id="case-" + strategy)
                profile = finite_profile(job)
                self.assertEqual(runner.validate_identity(profile, job), [])
                for other in runner.STRATEGIES:
                    if other == strategy:
                        continue
                    profile["config"]["strategy"] = other
                    self.assertIn("strategy", runner.validate_identity(profile, job))

    def test_shape_seed_model_case_and_reference_mismatch_are_rejected(self):
        changes = {
            "id": "another-case", "mode": "prepare", "height": 256, "width": 256,
            "frames": 25, "steps": 12, "seed": 1, "cache": "cache-dit",
            "model_id": "other-model", "revision": "other-revision", "insert_after": 10,
            "b_latent_path": "/mock/other.pt", "b_metadata_path": "/mock/other.json",
            "b_frames": 33, "repeat": 1, "preset": "other-preset", "diagnostic": True,
        }
        for key, changed in changes.items():
            with self.subTest(key=key):
                profile = finite_profile(self.job)
                profile["config"][key] = changed
                self.assertIn(key, runner.validate_identity(profile, self.job))

    def test_missing_failed_partial_or_nonfinite_profiles_never_complete(self):
        profiles = [None, {}, {"status": "completed"},
                    dict(finite_profile(self.job), status="failed"),
                    dict(finite_profile(self.job), status="partial")]
        for output in ("output_a", "output_b"):
            for change in ({"finite": False}, {"finite": None}, {"sha256": None}):
                profile = finite_profile(self.job)
                profile[output].update(change)
                profiles.append(profile)
            profile = finite_profile(self.job)
            del profile[output]
            profiles.append(profile)
        for index, profile in enumerate(profiles):
            with self.subTest(index=index):
                self.assertNotEqual(runner.classify_profile(profile)[0], "completed")
        self.assertNotEqual(runner.classify_profile(finite_profile(self.job), returncode=1)[0], "completed")

    def test_oom_is_a_negative_measurement_and_not_a_completed_output(self):
        profile = {"status": "failed", "failure": {"category": "cuda_oom"}}
        self.assertEqual(runner.classify_profile(profile, 1)[0], "oom")

    def test_output_alone_is_not_evidence_of_a_finished_timed_measurement(self):
        for change in ({"measured": False}, {"measured": None}, {"kind": "simulation"}):
            with self.subTest(change=change):
                profile = finite_profile(self.job)
                profile.update(change)
                self.assertNotEqual(runner.classify_profile(profile)[0], "completed")
        for value in (None, True, 0, -1, float("nan"), float("inf")):
            with self.subTest(remaining_ms=value):
                profile = finite_profile(self.job)
                profile["timings"]["remaining_ms"] = value
                self.assertNotEqual(runner.classify_profile(profile)[0], "completed")

    def test_actual_device_identity_requires_both_uuid_evidence_and_logical_mapping(self):
        base = finite_profile(self.job)
        self.assertTrue(runner.validate_environment(base, UUID))
        changes = (
            ("ready", False), ("device", "cuda:1"), ("cuda_device_order", "FASTEST_FIRST"),
            ("cuda_visible_devices", "1"), ("cuda_visible_devices", UUID + ",GPU-other"),
            ("gpu", {"uuid": "GPU-b058066e-bde9-c534-efb1-d2a68c117681"}),
        )
        for key, value in changes:
            with self.subTest(key=key, value=value):
                profile = copy.deepcopy(base)
                profile["environment"][key] = value
                self.assertFalse(runner.validate_environment(profile, UUID))
        del base["environment"]
        self.assertFalse(runner.validate_environment(base, UUID))


class ScreeningTests(unittest.TestCase):
    def test_33_oom_branches_to_25_and_never_49(self):
        outcomes = {}
        visited = []
        while True:
            frames = runner.next_screen_shape(outcomes)
            if frames is None:
                break
            visited.append(frames)
            self.assertLessEqual(len(visited), 4)
            outcomes[frames] = "oom" if frames == 33 else "completed"
        self.assertEqual(visited, [17, 33, 25])
        self.assertNotIn(49, visited)

    def test_success_reaches_49_once_and_other_failures_stop(self):
        self.assertEqual(runner.next_screen_shape({17: "completed", 33: "completed"}), 49)
        self.assertIsNone(runner.next_screen_shape({17: "completed", 33: "completed", 49: "oom"}))
        for outcomes in ({17: "oom"}, {17: "failed"}, {17: "completed", 33: "failed"},
                         {17: "completed", 33: "oom", 25: "oom"}):
            with self.subTest(outcomes=outcomes):
                self.assertIsNone(runner.next_screen_shape(outcomes))


class DeferCacheTests(unittest.TestCase):
    def test_defer_finishes_all_core_groups_before_skipping_cache_prepare(self):
        for defer in (False, True):
            with self.subTest(defer=defer), tempfile.TemporaryDirectory() as tmp:
                suite = runner.Suite(arguments(tmp, *(["--defer-cache"] if defer else [])))
                groups, prepares = [], []

                def prepare(frames, **kwargs):
                    prepares.append((frames, kwargs))
                    return {"status": "completed", "job": {"frames": frames}}

                def case_job(b, frames, strategy, **kwargs):
                    return dict(frames=frames, strategy=strategy, **kwargs)

                def group(jobs, label):
                    groups.append(label)
                    return True

                with patch.object(suite, "prepare", side_effect=prepare), \
                        patch.object(suite, "case_job", side_effect=case_job), \
                        patch.object(suite, "run", return_value={"status": "completed"}), \
                        patch.object(suite, "run_group", side_effect=group), \
                        patch.object(suite, "check_group_outputs") as checks, \
                        patch.object(suite, "has_interference", return_value=True) as interference:
                    suite.execute()
                self.assertEqual(groups[:6], ["smoke correctness", "primary four strategies", "reversed key pair",
                                              "seed 1", "earlier insertion", "mixed frame counts"])
                saved = runner.read_json(suite.path)
                self.assertTrue(saved["core_protocol_finished"])
                self.assertTrue(saved["smoke_validated"])
                self.assertEqual(saved["selected_frames"], 49)
                self.assertEqual(checks.call_count, 6 if defer else 7)
                cached_prepares = [item for item in prepares if item[1].get("cache") == "cache-dit"]
                if defer:
                    self.assertEqual(cached_prepares, [])
                    interference.assert_not_called()
                    self.assertEqual(len(groups), 6)
                    self.assertIn("--defer-cache", saved["decisions"][-1])
                else:
                    self.assertEqual(len(cached_prepares), 1)
                    self.assertEqual(groups[-1], "conditional fixed-cache check")
                    interference.assert_called_once_with()

    def test_defer_does_not_mark_an_unfinished_core_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = runner.Suite(arguments(tmp, "--defer-cache"))
            with patch.object(suite, "prepare", return_value={"status": "oom"}), \
                    patch.object(suite, "has_interference", side_effect=AssertionError("optional stage")):
                suite.execute()
            self.assertNotIn("core_protocol_finished", runner.read_json(suite.path))
            self.assertEqual(suite.manifest["decisions"], [])


class WorkerAccountingTests(unittest.TestCase):
    def run_attempt(self, outcomes, profile_change=None, launch_error=None, used=0):
        """Keep all fixture files under a temporary root and all processes fake."""
        stack = ExitStack()
        self.addCleanup(stack.close)
        temporary = stack.enter_context(tempfile.TemporaryDirectory())
        root = Path(temporary)
        suite = runner.Suite(arguments(temporary))
        suite.budget.used_seconds = used
        process = FakeProcess(outcomes)
        telemetry = types.SimpleNamespace(start=Mock(), stop=Mock(return_value={"status": "completed"}))

        def launch(command, **kwargs):
            if launch_error is not None:
                raise launch_error
            job = runner.read_json(command[-1])
            profile = finite_profile(job)
            if profile_change is not None:
                profile = profile_change(profile)
            if profile is not None:
                runner.atomic_json(Path(job["output_dir"]) / "profile.json", profile)
            return process

        stack.enter_context(patch.object(runner, "ROOT", root))
        stack.enter_context(patch.object(runner, "gpu_available", return_value=GPU))
        stack.enter_context(patch.object(suite, "estimate", return_value=1))
        popen = stack.enter_context(patch.object(runner.subprocess, "Popen", side_effect=launch))
        killpg = stack.enter_context(patch.object(runner.os, "killpg"))
        stack.enter_context(patch.object(runner.common, "GpuTelemetry", return_value=telemetry))
        stack.enter_context(patch.object(runner.time, "monotonic", side_effect=[100.0, 117.0]))
        stack.enter_context(redirect_stdout(io.StringIO()))
        job = suite.job("test-attempt")
        return suite, job, process, telemetry, popen, killpg

    def assert_charge_saved(self, suite, status, used=17):
        saved = runner.read_json(suite.path)
        self.assertEqual(saved["budget"]["used_seconds"], used)
        self.assertEqual(saved["records"][-1]["elapsed_seconds"], 17)
        self.assertEqual(saved["records"][-1]["status"], status)

    def test_success_is_charged_and_child_is_bound_to_only_the_target_uuid(self):
        suite, job, process, telemetry, popen, killpg = self.run_attempt([0], used=5)
        self.assertEqual(suite.run(job)["status"], "completed")
        self.assert_charge_saved(suite, "completed", 22)
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], UUID)
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(process.wait_timeouts, [30])
        telemetry.stop.assert_called_once()
        killpg.assert_not_called()

    def test_failed_exit_or_missing_profile_is_charged_before_raising(self):
        for outcomes, profile_change in (([1], None), ([0], lambda profile: None)):
            with self.subTest(outcomes=outcomes):
                suite, job, _, telemetry, _, _ = self.run_attempt(outcomes, profile_change)
                with self.assertRaises(RuntimeError):
                    suite.run(job)
                self.assert_charge_saved(suite, "failed")
                telemetry.stop.assert_called_once()

    def test_wrong_reported_device_after_successful_exit_is_a_charged_failure(self):
        def wrong_gpu(profile):
            profile["environment"]["gpu"]["uuid"] = "GPU-b058066e-bde9-c534-efb1-d2a68c117681"
            return profile

        suite, job, _, _, _, _ = self.run_attempt([0], wrong_gpu)
        with self.assertRaises(RuntimeError):
            suite.run(job)
        self.assert_charge_saved(suite, "failed")
        self.assertEqual(suite.records[-1]["failure_category"], "actual_device_mapping_mismatch")

    def test_timeout_terminates_then_kills_and_reaps_before_accounting(self):
        timeout = subprocess.TimeoutExpired(["mock-worker"], 30)
        suite, job, process, telemetry, _, killpg = self.run_attempt([timeout, timeout, -9])
        with self.assertRaises(RuntimeError):
            suite.run(job)
        self.assertEqual([call.args for call in killpg.call_args_list],
                         [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)])
        self.assertEqual(process.wait_timeouts, [30, 5, None])
        self.assert_charge_saved(suite, "failed")
        self.assertEqual(suite.records[-1]["failure_category"], "budget_or_worker_timeout")
        telemetry.stop.assert_called_once()

    def test_interrupt_kills_and_reaps_worker_and_preserves_elapsed(self):
        suite, job, process, telemetry, _, killpg = self.run_attempt([KeyboardInterrupt(), -9])
        with self.assertRaises(KeyboardInterrupt):
            suite.run(job)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        self.assertEqual(process.wait_timeouts, [30, None])
        self.assert_charge_saved(suite, "failed")
        telemetry.stop.assert_called_once()

    def test_launch_failure_is_charged_and_does_not_leave_running_record(self):
        suite, job, _, _, _, killpg = self.run_attempt([], launch_error=OSError("mock launch failure"))
        with self.assertRaises(OSError):
            suite.run(job)
        self.assert_charge_saved(suite, "failed")
        killpg.assert_not_called()

    def test_remaining_total_budget_bounds_worker_timeout(self):
        suite, job, process, _, _, _ = self.run_attempt([0], used=3590)
        suite.run(job)
        self.assertEqual(process.wait_timeouts, [10])
        self.assert_charge_saved(suite, "completed", 3607)
        self.assertEqual(suite.budget.remaining_seconds, 0)

    def test_exhausted_budget_does_not_query_or_launch_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = runner.Suite(arguments(tmp))
            suite.budget.charge(suite.budget.limit_seconds)
            with patch.object(runner, "gpu_available", side_effect=AssertionError("GPU query")), \
                    patch.object(runner.subprocess, "Popen", side_effect=AssertionError("launch")):
                record = suite.run(suite.job("no-budget"))
            self.assertEqual(record["status"], "skipped_budget")
            self.assertEqual(record["elapsed_seconds"], 0)


class CompletedRecordAndOutputTests(unittest.TestCase):
    def stored(self, suite, job, profile=None):
        path = suite.root / (job["id"] + ".json")
        runner.atomic_json(path, profile or finite_profile(job))
        record = {"id": job["id"], "job": copy.deepcopy(job), "status": "completed",
                  "profile_path": str(path), "elapsed_seconds": 10}
        suite.records.append(record)
        suite.save()
        return record

    def test_same_id_cannot_resume_another_seed_or_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = runner.Suite(arguments(tmp))
            job = suite.job("fixed-id")
            self.stored(suite, job)
            with patch.object(runner, "gpu_available", side_effect=AssertionError("must not launch")):
                with self.assertRaises((ValueError, RuntimeError)):
                    suite.run(dict(job, seed=1))

    def test_valid_completed_profile_is_reused_without_a_second_charge_or_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = runner.Suite(arguments(tmp))
            job = suite.job("validated-record")
            record = self.stored(suite, job)
            suite.budget.charge(10)
            with patch.object(runner, "gpu_available", side_effect=AssertionError("GPU query")), \
                    patch.object(runner.subprocess, "Popen", side_effect=AssertionError("launch")):
                self.assertIs(suite.run(job), record)
            self.assertEqual(suite.budget.used_seconds, 10)
            self.assertEqual(len(suite.records), 1)

    def test_completed_ledger_does_not_hide_missing_or_nonfinite_raw_profile(self):
        for invalid in ("missing", "nonfinite", "failed"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                suite = runner.Suite(arguments(tmp))
                job = suite.job("stale-record")
                record = self.stored(suite, job)
                path = Path(record["profile_path"])
                if invalid == "missing":
                    path.unlink()
                else:
                    profile = finite_profile(job)
                    if invalid == "nonfinite":
                        profile["output_b"]["finite"] = False
                    else:
                        profile["status"] = "failed"
                    runner.atomic_json(path, profile)
                with patch.object(runner, "gpu_available", side_effect=AssertionError("must not launch")):
                    with self.assertRaises((ValueError, RuntimeError, OSError)):
                        suite.run(job)

    def make_group(self, suite):
        metadata = suite.root / "b-reference.json"
        runner.atomic_json(metadata, {"output_b": {"sha256": HASH_B, "finite": True}})
        jobs = [suite.job("case-" + strategy, mode="case", strategy=strategy,
                          insert_after=25, b_latent_path="/mock/b.pt",
                          b_metadata_path=str(metadata), b_frames=17, repeat=0)
                for strategy in runner.STRATEGIES]
        for job in jobs:
            self.stored(suite, job)
        return jobs

    def test_four_matching_outputs_pass_but_wrong_b_reference_is_preserved_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = runner.Suite(arguments(tmp))
            jobs = self.make_group(suite)
            suite.check_group_outputs(jobs)
            record = suite.latest(jobs[2]["id"])
            profile = runner.read_json(record["profile_path"])
            profile["output_b"]["sha256"] = "c" * 64
            runner.atomic_json(record["profile_path"], profile)
            with self.assertRaisesRegex(RuntimeError, "B output mismatch"):
                suite.check_group_outputs(jobs)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["failure_category"], "b_output_mismatch")

    def test_a_output_disagreement_stops_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = runner.Suite(arguments(tmp))
            jobs = self.make_group(suite)
            record = suite.latest(jobs[-1]["id"])
            profile = runner.read_json(record["profile_path"])
            profile["output_a"]["sha256"] = "d" * 64
            runner.atomic_json(record["profile_path"], profile)
            with self.assertRaisesRegex(RuntimeError, "A output mismatch"):
                suite.check_group_outputs(jobs)


class DryRunTests(unittest.TestCase):
    def test_dryrun_never_imports_torch_queries_gpu_launches_or_writes(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "diffusers", "cache_dit") or name == "ditserve.gpu.phase_conflict":
                raise AssertionError("heavy dependency imported by dry-run: " + name)
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "must-not-exist"
            captured = io.StringIO()
            with patch.object(builtins, "__import__", side_effect=guarded_import), \
                    patch.object(runner, "gpu_available", side_effect=AssertionError("GPU query")), \
                    patch.object(runner.common, "resolve_gpu", side_effect=AssertionError("GPU query")), \
                    patch.object(runner.subprocess, "Popen", side_effect=AssertionError("launch")), \
                    patch.object(runner.subprocess, "run", side_effect=AssertionError("subprocess")), \
                    patch.object(runner, "atomic_json", side_effect=AssertionError("write")), \
                    patch.object(Path, "open", side_effect=AssertionError("file access")), \
                    patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")), \
                    patch.object(Path, "write_text", side_effect=AssertionError("write")), \
                    patch.object(Path, "write_bytes", side_effect=AssertionError("write")), \
                    redirect_stdout(captured):
                self.assertEqual(runner.main(["--dry-run", "--output", str(output), "--defer-cache"]), 0)
            self.assertFalse(output.exists())
            plan = json.loads(captured.getvalue())
            self.assertEqual(plan["strategies"], list(runner.STRATEGIES))
            self.assertEqual(plan["max_frames"], 49)
            self.assertTrue(plan["defer_cache"])
            self.assertFalse(runner.parser().parse_args([]).defer_cache)


class GpuAvailabilityTests(unittest.TestCase):
    def test_stale_utilization_then_idle_retries_the_same_target(self):
        samples = [subprocess.CompletedProcess([], 0, "32490, 34\n", ""),
                   subprocess.CompletedProcess([], 0, "32490, 0\n", "")]
        with patch.object(runner.common, "resolve_gpu", return_value=copy.deepcopy(GPU)) as resolve, \
                patch.object(runner.subprocess, "run", side_effect=samples) as query, \
                patch.object(runner.time, "sleep") as sleep, \
                patch.object(runner.subprocess, "Popen", side_effect=AssertionError("worker launch")):
            result = runner.gpu_available(UUID)
        resolve.assert_called_once_with(UUID)
        self.assertEqual(query.call_count, 2)
        sleep.assert_called_once_with(2)
        self.assertEqual(result["uuid"], UUID)
        self.assertEqual(result["utilization_gpu_percent"], 0)
        self.assertEqual(result["memory_free_mib"], 32490)
        self.assertEqual([sample["utilization_gpu_percent"] for sample in result["availability_samples"]],
                         [34, 0])
        for call in query.call_args_list:
            self.assertIn("--id=" + UUID, call.args[0])
            self.assertEqual(call.kwargs["timeout"], 10)

    def test_persistently_busy_gpu_stops_before_launch_without_spending_worker_budget(self):
        # Either retained allocations or sustained utilization must block launch.
        for sample in ("32490, 34\n", "30000, 0\n"):
            with self.subTest(sample=sample), tempfile.TemporaryDirectory() as tmp:
                suite = runner.Suite(arguments(tmp))
                suite.budget.charge(83.9)
                suite.save()
                with patch.object(runner.common, "resolve_gpu", return_value=copy.deepcopy(GPU)), \
                        patch.object(runner.subprocess, "run", return_value=
                                     subprocess.CompletedProcess([], 0, sample, "")) as query, \
                        patch.object(runner.time, "sleep") as sleep, \
                        patch.object(runner.subprocess, "Popen", side_effect=AssertionError("worker launch")) as launch:
                    with self.assertRaises(RuntimeError):
                        suite.run(suite.job("must-not-start"))
                self.assertEqual(query.call_count, 4)
                self.assertEqual([call.args for call in sleep.call_args_list], [(2,), (2,), (2,)])
                launch.assert_not_called()
                self.assertEqual(suite.budget.used_seconds, 83.9)
                self.assertEqual(runner.read_json(suite.path)["budget"]["used_seconds"], 83.9)
                self.assertEqual(suite.records, [])
                self.assertFalse((suite.root / "cases").exists())

    def test_nonfinite_or_negative_availability_never_counts_as_idle(self):
        for sample in ("nan, 0\n", "inf, 0\n", "32490, nan\n", "32490, inf\n", "32490, -1\n"):
            with self.subTest(sample=sample), \
                    patch.object(runner.common, "resolve_gpu", return_value=copy.deepcopy(GPU)), \
                    patch.object(runner.subprocess, "run", return_value=
                                 subprocess.CompletedProcess([], 0, sample, "")) as query, \
                    patch.object(runner.time, "sleep"), \
                    patch.object(runner.subprocess, "Popen", side_effect=AssertionError("worker launch")):
                with self.assertRaises(RuntimeError):
                    runner.gpu_available(UUID)
                self.assertEqual(query.call_count, 4)


class InterferenceSelectionTests(unittest.TestCase):
    def test_retry_entries_do_not_fake_repeated_oom_in_distinct_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = runner.Suite(arguments(tmp))
            suite.manifest["selected_frames"] = 17
            job = suite.job("overlap-r0", mode="case", strategy="overlap",
                            insert_after=25, b_frames=17, repeat=0)
            record = {"id": job["id"], "job": job, "status": "oom", "elapsed_seconds": 10}
            suite.records.extend([copy.deepcopy(record), copy.deepcopy(record)])
            self.assertFalse(suite.has_interference())
            repeated = copy.deepcopy(record)
            repeated["id"] = repeated["job"]["id"] = "overlap-r1"
            repeated["job"]["repeat"] = 1
            suite.records.append(repeated)
            self.assertTrue(suite.has_interference())

    def test_mixed_b_workload_cannot_change_same_shape_interference_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = runner.Suite(arguments(tmp))
            suite.manifest["selected_frames"] = 33

            def append(strategy, milliseconds, repeat=0, b_frames=33):
                case_id = "%s-r%d-b%d" % (strategy, repeat, b_frames)
                job = suite.job(case_id, mode="case", frames=33, strategy=strategy,
                                insert_after=25, b_frames=b_frames, repeat=repeat)
                profile = finite_profile(job)
                profile["timings"]["remaining_ms"] = milliseconds
                path = suite.root / (case_id + ".json")
                runner.atomic_json(path, profile)
                suite.records.append({"id": case_id, "job": job, "status": "completed",
                                      "elapsed_seconds": 10, "profile_path": str(path)})

            # Both same-shape strategies have a 100 ms median before mixed cases.
            append("blocking", 100, 0)
            append("blocking", 100, 1)
            append("overlap", 80, 0)
            append("overlap", 120, 1)
            self.assertFalse(suite.has_interference())
            # Pooling the distinct B workload would incorrectly move overlap's
            # median to 120 ms while blocking's median remains 100 ms.
            append("blocking", 1000, b_frames=17)
            append("overlap", 1000, b_frames=17)
            self.assertFalse(suite.has_interference())

    def test_repeated_same_workload_penalty_can_trigger_attribution_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = runner.Suite(arguments(tmp))
            suite.manifest["selected_frames"] = 17
            for strategy, duration in (("blocking", 100), ("overlap", 130)):
                for repeat in (0, 1):
                    job = suite.job(strategy + str(repeat), mode="case", strategy=strategy,
                                    insert_after=25, b_frames=17, repeat=repeat)
                    profile = finite_profile(job)
                    profile["timings"]["remaining_ms"] = duration
                    path = suite.root / (job["id"] + ".json")
                    runner.atomic_json(path, profile)
                    suite.records.append({"id": job["id"], "job": job, "status": "completed",
                                          "elapsed_seconds": 10, "profile_path": str(path)})
            self.assertTrue(suite.has_interference())


if __name__ == "__main__":
    unittest.main()
