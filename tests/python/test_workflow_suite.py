"""GPU admission uses ownership identities; tests never contact Docker or CUDA."""

import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "workflow_suite", Path(__file__).resolve().parents[2] / "docker/run-workflow-suite.py")
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)
CONTAINER_ID = "a" * 64


def process(pid, parent=1, start=1, own=False, state="S"):
    return dict(pid=pid, ppid=parent, start_ticks=start, state=state, name="test-worker",
                cgroups=[("0", "", "/docker/" + CONTAINER_ID if own else "/")])


def sample(*pids, device=1, graphics=()):
    return [dict(device=device, uuid="GPU-test", compute_processes=[dict(pid=pid) for pid in pids],
                 graphics_processes=[dict(pid=pid) for pid in graphics])]


class FakeProcess:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


class WorkflowDeviceTests(unittest.TestCase):
    def test_container_startup_messages_are_case_insensitive(self):
        for message in ("error: no such object: pending", "Error: No such container: pending"):
            with patch.object(suite, "docker_command", return_value=SimpleNamespace(returncode=1, stderr=message)):
                self.assertIsNone(suite.inspect_container("pending"))
        with patch.object(suite, "docker_command", return_value=SimpleNamespace(returncode=1, stderr="permission denied")):
            with self.assertRaises(suite.MonitoringError):
                suite.inspect_container("pending")

    def test_default_single_card_and_both_overlap_variants(self):
        for experiment in ("E1", "E3", "E4", "E5"):
            self.assertEqual(suite.selected_devices(dict(command=["--experiment", experiment])), [1])
        for experiment in ("E2", "E6"):
            self.assertEqual(suite.selected_devices(dict(command=["--experiment", experiment])), [0, 1])
        self.assertEqual(suite.selected_devices(dict(command=["--experiment", "E2", "--devices=2,3", "--device", "cuda:3"])), [2, 3])

    def test_encoder_dedicated_and_explicit_coverage(self):
        base = ["-m", "ditserve.evaluation.workflow_encoder"]
        self.assertEqual(suite.selected_devices(dict(command=base + ["--phases", "T0"])), [1])
        self.assertEqual(suite.selected_devices(dict(command=base + ["--phases", "T5"])), [0, 1])
        self.assertEqual(suite.selected_devices(dict(command=base)), [0, 1])
        self.assertEqual(suite.selected_devices(dict(command=base + ["--child", "T0"])), [1])
        self.assertEqual(suite.selected_devices(dict(command=base + ["--child=T5"])), [0, 1])
        self.assertEqual(suite.selected_devices(dict(command=base + ["--phases", "T5", "--child", "T2"])), [1])
        with self.assertRaisesRegex(ValueError, "omits"):
            suite.selected_devices(dict(command=base + ["--phases", "T5"], devices=[1]))


class WorkflowOwnershipTests(unittest.TestCase):
    def owner(self, rows):
        owner = suite.ContainerOwnership(rows.get)
        owner.update(dict(id=CONTAINER_ID, pid=100, running=True))
        return owner

    def test_nested_children_are_owned_without_false_gpu_conflicts(self):
        rows = {100: process(100, own=True), 200: process(200, parent=100), 300: process(300, parent=200)}
        owner = self.owner(rows)
        self.assertEqual(suite.foreign_processes(sample(300, graphics=(300,)), owner, rows.get), [])
        self.assertIn((200, 1), owner.known)
        self.assertIn((300, 1), owner.known)

    def test_seen_and_unseen_orphans_are_owned_but_pid_reuse_is_not(self):
        rows = {100: process(100, own=True), 200: process(200, parent=100)}
        owner = self.owner(rows)
        self.assertTrue(owner.owns(rows[200]))
        rows[200] = process(200, parent=1)
        self.assertTrue(owner.owns(rows[200]))
        self.assertTrue(owner.owns(process(300, own=True)))
        rows[200] = process(200, parent=1, start=999)
        self.assertFalse(owner.owns(rows[200]))

    def test_shared_host_cgroup_does_not_authorize_foreign_process(self):
        rows = {100: process(100), 200: process(200)}
        owner = self.owner(rows)
        self.assertFalse(owner.owns(rows[200]))

    def test_missing_and_zombie_nvml_records_are_not_live_conflicts(self):
        rows = {100: process(100, state="Z"), 200: process(200)}
        conflicts = suite.foreign_processes(sample(100, 200, 300, graphics=(200,)), lookup=rows.get)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["pid"], 200)
        self.assertEqual(conflicts[0]["kinds"], ["compute", "graphics"])

    def test_missing_query_fails_closed_instead_of_claiming_empty_gpu(self):
        rows = sample()
        rows[0]["graphics_processes"] = None
        with self.assertRaises(suite.MonitoringError):
            suite.foreign_processes(rows, lookup=lambda _: None)

    def test_container_identity_changes_are_rejected(self):
        rows = {100: process(100, own=True)}
        owner = self.owner(rows)
        with self.assertRaisesRegex(suite.MonitoringError, "identity changed"):
            owner.update(dict(id="b" * 64, pid=100, running=True))

    def test_proc_parser_handles_parentheses_and_start_ticks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "100"
            directory.mkdir()
            fields = ["S", "1"] + ["0"] * 17 + ["123456"]
            (directory / "stat").write_text("100 (worker (name)) " + " ".join(fields))
            (directory / "cgroup").write_text("0::/docker/" + CONTAINER_ID + "\n")
            value = suite.process_identity(100, root)
            self.assertEqual((value["pid"], value["ppid"], value["start_ticks"]), (100, 1, 123456))
            self.assertEqual(value["name"], "worker (name)")
            self.assertIsNone(suite.process_identity(999, root))


class WorkflowMonitorTests(unittest.TestCase):
    def record(self):
        return dict(timeout_seconds=30, status="running",
                    gpu_monitor=dict(conflicts=[], errors=[]))

    def test_foreign_arrival_invalidates_only_own_job(self):
        rows = {100: process(100, own=True), 200: process(200, parent=100), 900: process(900)}
        class Probe:
            def sample(self):
                return sample(200, 900)
        record = self.record()
        owner = suite.monitor_process(FakeProcess(), Probe(), "unique-own-name", record,
            inspect=lambda _: dict(id=CONTAINER_ID, pid=100, running=True),
            lookup=rows.get, sleep=lambda _: None, monotonic=lambda: 0)
        self.assertEqual(record["status"], "interrupted_foreign_gpu_workload")
        self.assertEqual(record["gpu_monitor"]["conflicts"][0]["processes"][0]["pid"], 900)
        self.assertEqual(owner.container_id, CONTAINER_ID)

    def test_raw_capacity_samples_reproduce_monitor_aggregates(self):
        child = FakeProcess()
        raw = [dict(sample()[0], monotonic_ns=100, memory_total_bytes=1000, memory_used_bytes=100),
               dict(sample()[0], monotonic_ns=200, memory_total_bytes=1000, memory_used_bytes=350)]
        class Probe:
            def __init__(self): self.index = 0
            def sample(self):
                row = raw[self.index]
                self.index += 1
                return [dict(row)]
        probe = Probe()
        record = self.record()
        suite.monitor_process(child, probe, "unique-own-name", record,
            inspect=lambda _: dict(id=CONTAINER_ID, pid=100, running=True),
            lookup=lambda pid: process(pid, own=True), monotonic=lambda: 0,
            sleep=lambda _: setattr(child, "returncode", 0) if probe.index == 2 else None)
        observed = record["gpu_monitor"]["samples"]
        self.assertEqual(observed, raw)
        self.assertEqual(record["gpu_monitor"]["checks"], 2)
        self.assertEqual(record["gpu_monitor"]["memory"]["1"], dict(samples=len(observed), total_bytes=1000,
            peak_used_bytes=max(row["memory_used_bytes"] for row in observed),
            minimum_free_bytes=min(row["memory_total_bytes"] - row["memory_used_bytes"] for row in observed)))

    def test_container_startup_race_reinspect_recognizes_new_children(self):
        rows = {100: process(100, own=True), 200: process(200, parent=100)}
        child = FakeProcess()
        class Probe:
            def sample(self):
                return sample(200)
        inspections = iter([None, dict(id=CONTAINER_ID, pid=100, running=True)])
        record = self.record()
        suite.monitor_process(child, Probe(), "unique-own-name", record, inspect=lambda _: next(inspections),
            lookup=rows.get, sleep=lambda _: setattr(child, "returncode", 0), monotonic=lambda: 0)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["gpu_monitor"]["conflicts"], [])

    def test_inspection_failure_invalidates_measurement(self):
        def broken(_):
            raise suite.MonitoringError("daemon unavailable")
        record = self.record()
        suite.monitor_process(FakeProcess(), None, "unique-own-name", record,
                              inspect=broken, monotonic=lambda: 0)
        self.assertEqual(record["status"], "gpu_monitor_unavailable")
        self.assertIn("daemon unavailable", record["gpu_monitor"]["errors"])

    def test_stop_targets_immutable_own_container_id_never_gpu_pids(self):
        owner = suite.ContainerOwnership()
        owner.container_id = CONTAINER_ID
        with patch.object(suite, "docker_command") as command, \
                patch.object(suite, "inspect_container", return_value=None) as inspect:
            command.return_value.returncode = 0
            result = suite.stop_own_container("unique-own-name", owner, FakeProcess(), None, None)
        self.assertEqual(command.call_args.args[0], ["stop", "--time", "10", CONTAINER_ID])
        inspect.assert_called_once_with(CONTAINER_ID)
        self.assertEqual(result["target_kind"], "container_id")
        self.assertTrue(result["verified_stopped"])

    def test_monotonic_budget_deadline_survives_wall_clock_rollback(self):
        record = self.record()
        record.update(timeout_seconds=1000, budget_stop_at_monotonic=210.,
                      budget_stop_at_unix=1210.)
        with patch.object(suite.time, "time", return_value=800.):
            suite.monitor_process(FakeProcess(), None, "own-name", record,
                inspect=lambda _: self.fail("expired budget must stop before inspecting Docker"),
                monotonic=lambda: 220.)
        self.assertEqual(record["status"], "budget_exhausted")
        self.assertEqual(record["gpu_monitor"]["checks"], 0)

    def test_unlimited_task_deadline_is_a_timeout_and_still_precedes_gpu_inspection(self):
        record = self.record()
        record.update(timeout_seconds=1000, budget_stop_at_monotonic=210., deadline_reason="task_timeout")
        suite.monitor_process(FakeProcess(), None, "own-name", record,
            inspect=lambda _: self.fail("expired task must stop before inspecting Docker"), monotonic=lambda: 220.)
        self.assertEqual(record["status"], "timeout")

    def test_failed_or_timed_out_stop_requires_kill_and_verified_container_exit(self):
        for stop_result in (SimpleNamespace(returncode=1), subprocess.TimeoutExpired("docker stop", 25)):
            with self.subTest(stop_result=type(stop_result).__name__):
                owner = suite.ContainerOwnership()
                owner.container_id = CONTAINER_ID
                alive = dict(id=CONTAINER_ID, pid=100, running=True)
                with patch.object(suite, "docker_command", side_effect=[stop_result, SimpleNamespace(returncode=0)]) as command, \
                        patch.object(suite, "inspect_container", side_effect=[alive, None]) as inspect:
                    result = suite.stop_own_container("unique-own-name", owner, FakeProcess(), None, None)
                self.assertEqual([call.args[0] for call in command.call_args_list],
                    [["stop", "--time", "10", CONTAINER_ID], ["kill", CONTAINER_ID]])
                self.assertEqual([call.args[0] for call in inspect.call_args_list], [CONTAINER_ID, CONTAINER_ID])
                self.assertTrue(result["verified_stopped"])
                self.assertEqual(result["kill_returncode"], 0)

    def test_exited_docker_client_does_not_certify_a_still_running_container(self):
        owner = suite.ContainerOwnership()
        owner.container_id = CONTAINER_ID
        child = FakeProcess()
        alive = dict(id=CONTAINER_ID, pid=100, running=True)
        with patch.object(suite, "docker_command", return_value=SimpleNamespace(returncode=1)), \
                patch.object(suite, "inspect_container", return_value=alive):
            result = suite.stop_own_container("unique-own-name", owner, child, None, None)
        self.assertEqual(child.poll(), 0)
        self.assertFalse(result["verified_stopped"])

    def test_unavailable_inspection_never_certifies_cleanup(self):
        with patch.object(suite, "docker_command", return_value=SimpleNamespace(returncode=0)), \
                patch.object(suite, "inspect_container", side_effect=suite.MonitoringError("daemon unavailable")):
            result = suite.stop_own_container("unique-own-name", suite.ContainerOwnership(), FakeProcess(), None, None)
        self.assertFalse(result["verified_stopped"])
        self.assertIn("daemon unavailable", result["verification_error"])


class WorkflowStrictResumeTests(unittest.TestCase):
    """Exercise main's early resume path without starting Docker or NVML."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "sources/frozen"
        self.source.mkdir(parents=True)
        self.source_file = self.source / "worker.py"
        self.source_file.write_text("value = 1\n")
        self.config = self.root / "input.json"
        self.config.write_text('{"frames":17}\n')
        self.output = self.root / "output/run.json"
        self.output.parent.mkdir()
        self.output.write_text('{"status":"completed","configuration":{"frames":17}}\n')
        source_files = [dict(path="worker.py", bytes=self.source_file.stat().st_size,
                            sha256=self.sha256(self.source_file))]
        (self.root / "study.json").write_text(json.dumps(dict(source_files=source_files)))
        digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.job = dict(id="finished", command=["unused"], input=str(self.config),
                        configuration_sha256=digest(dict(frames=17)), output=str(self.output.parent))
        self.bundle = dict(root=str(self.root), source=str(self.source), source_digest=digest(source_files),
                           image="sha256:immutable", strict_resume=True, jobs=[self.job])
        fingerprint = hashlib.sha256(json.dumps(dict(job=self.job,
            source_digest=self.bundle["source_digest"], image=self.bundle["image"]),
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.ledger = self.root / "launches/finished/launch.json"
        self.ledger.parent.mkdir(parents=True)
        self.ledger.write_text(json.dumps(dict(status="completed", job_fingerprint=fingerprint,
            output_sha256=self.sha256(self.output))))
        self.jobs_path = self.root / "jobs.json"
        self.jobs_path.write_text(json.dumps(self.bundle))

    @staticmethod
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def resume(self):
        with patch.object(sys, "argv", ["suite", "--jobs", str(self.jobs_path)]), \
                patch.object(suite.subprocess, "Popen", side_effect=AssertionError("must not launch Docker")), \
                patch.object(suite, "NvmlProbe", side_effect=AssertionError("must not query GPUs")), \
                patch.object(sys, "stdout", new_callable=io.StringIO):
            return suite.main()

    def test_unchanged_completed_binding_reuses_result_without_runtime(self):
        self.assertFalse(self.resume())

    def test_changed_input_bytes_rejected_even_with_old_job_digest(self):
        self.config.write_text('{"frames":33}\n')
        with self.assertRaisesRegex(ValueError, "input contents changed"):
            self.resume()

    def test_changed_source_same_size_rejected_even_with_old_manifest(self):
        self.source_file.write_text("value = 2\n")
        with self.assertRaisesRegex(ValueError, "source contents changed"):
            self.resume()

    def test_missing_frozen_source_rejected(self):
        self.source_file.unlink()
        with self.assertRaisesRegex(ValueError, "source contents changed"):
            self.resume()

    def test_added_executable_file_rejected(self):
        (self.source / "injected.py").write_text("value = 3\n")
        with self.assertRaisesRegex(ValueError, "unexpected executable"):
            self.resume()

    def test_changed_output_rejected_even_if_status_still_completed(self):
        self.output.write_text('{"status":"completed","configuration":{"frames":33}}\n')
        with self.assertRaisesRegex(RuntimeError, "output missing or changed"):
            self.resume()

    def test_missing_output_rejected(self):
        self.output.unlink()
        with self.assertRaisesRegex(RuntimeError, "output missing or changed"):
            self.resume()


class WorkflowUnlimitedBudgetTests(unittest.TestCase):
    def test_unlimited_lease_retains_timeout_and_explicit_task_ceiling(self):
        from ditserve.evaluation.workflow_budget import StudyBudget
        with tempfile.TemporaryDirectory() as directory:
            with StudyBudget(directory, None, clock=lambda: 1000., monotonic=lambda: 50.) as budget:
                record = {}
                lease = suite.start_budget_attempt(budget, "bounded", dict(timeout_seconds=3600), record)
                self.assertIsNone(lease["remaining_at_start"])
                self.assertEqual(record["budget_stop_at_monotonic"], 3650.)
                self.assertEqual(record["deadline_reason"], "task_timeout")
                budget.finish("completed")
                suite.start_budget_attempt(budget, "family-bound", dict(timeout_seconds=3600, budget_ceiling_seconds=600), record)
                self.assertEqual(record["budget_stop_at_monotonic"], 560.)
                self.assertEqual(record["deadline_reason"], "budget_deadline")

    def test_invalid_timeouts_and_task_ceilings_fail_before_attempt(self):
        from ditserve.evaluation.workflow_budget import StudyBudget
        with tempfile.TemporaryDirectory() as directory:
            with StudyBudget(directory, None) as budget:
                for value in (None, 0, -1, float("inf"), float("nan"), True):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        suite.start_budget_attempt(budget, "bad", dict(timeout_seconds=value), {})
                    self.assertIsNone(budget.record["active"])
                for value in (0, 90, float("inf"), float("nan"), True):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        suite.start_budget_attempt(budget, "bad", dict(budget_ceiling_seconds=value), {})
                    self.assertIsNone(budget.record["active"])

    def test_explicit_null_budget_runs_accounting_and_independent_watchdog(self):
        """Exercise main with CPU fakes only; no subprocess or NVML access."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "jobs.json"
            path.write_text(json.dumps(dict(root=str(root), budget_seconds=None, jobs=[dict(
                id="cpu-test", cpu_only=True, devices=[], command=["--kind", "cpu_micro"], timeout_seconds=12)])))
            children = []
            def launch(command, **kwargs):
                children.append(command)
                child = FakeProcess()
                child.pid = 123
                if "--armed" in command:
                    Path(command[command.index("--armed") + 1]).touch()
                else:
                    child.returncode = 0
                return child
            with patch.object(sys, "argv", ["suite", "--jobs", str(path)]), \
                    patch.object(suite.subprocess, "Popen", side_effect=launch), \
                    patch.object(suite, "process_identity", return_value=dict(start_ticks=1)), \
                    patch.object(suite, "inspect_container", return_value=None), \
                    patch.object(suite, "NvmlProbe", side_effect=AssertionError("must not access GPUs")), \
                    patch.object(sys, "stdout", new_callable=io.StringIO):
                self.assertFalse(suite.main())
            budget = json.loads((root / "budget.json").read_text())
            self.assertIsNone(budget["limit_seconds"])
            self.assertIsNone(budget["active"])
            self.assertEqual([row["status"] for row in budget["attempts"]], ["completed"])
            self.assertEqual(len(children), 2)
            watchdog = children[0]
            self.assertEqual(watchdog[watchdog.index("--deadline-reason") + 1], "task_timeout")
            for name in ("--deadline", "--monotonic-deadline"):
                self.assertTrue(math.isfinite(float(watchdog[watchdog.index(name) + 1])))
            ledger = json.loads((root / "launches/cpu-test/launch.json").read_text())
            self.assertEqual(ledger["gpu_monitor"]["admission"], "passed")
            self.assertTrue((root / "launches/cpu-test/watchdog.done").exists())

    def test_explicit_null_does_not_silently_convert_finite_existing_budget(self):
        from ditserve.evaluation.workflow_budget import StudyBudget
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with StudyBudget(root, 300):
                pass
            path = root / "jobs.json"
            path.write_text(json.dumps(dict(root=str(root), budget_seconds=None, jobs=[])))
            with patch.object(sys, "argv", ["suite", "--jobs", str(path)]), self.assertRaisesRegex(ValueError, "cannot change"):
                suite.main()


class WorkflowIsolatedGpuTests(unittest.TestCase):
    def setUp(self):
        from ditserve.evaluation.workflow_serving_study import digest
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "workflow/study"
        self.root.mkdir(parents=True)
        self.source = self.root / "source"
        self.source.mkdir()
        source_file = self.source / "test.py"
        source_file.write_text("pass\n")
        manifest = [dict(path="test.py", bytes=5, sha256=hashlib.sha256(source_file.read_bytes()).hexdigest())]
        (self.root / "study.json").write_text(json.dumps(dict(source_files=manifest)))
        self.uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
        self.config = self.root / "qualification.json"
        config = dict(kind="wan", gpu_uuid=self.uuid, device="cuda:0")
        self.config.write_text(json.dumps(config))
        self.job = dict(id="qualification", command_type="isolated_gpu", devices=[1],
            gpu_mapping=[dict(ordinal=1, uuid=self.uuid)], input=str(self.config),
            output=str(self.root / "output"), configuration_sha256=digest(config),
            command=["/usr/bin/env", "CUDA_VISIBLE_DEVICES=" + self.uuid, "/opt/venv/bin/python", "-m",
                "ditserve.evaluation.workflow_encoding_qualification", "--kind", "wan", "--config",
                "/data/workflow/study/qualification.json", "--out", "/data/workflow/study/output"])
        self.spec = dict(root=str(self.root), source=str(self.source), source_digest=digest(manifest),
                         strict_resume=True, jobs=[self.job])

    def test_frozen_isolated_uuid_maps_to_physical_ordinal(self):
        suite.validate_frozen_job(self.spec, self.job, self.root)
        self.assertEqual(suite.selected_devices(self.job), [1])

    def test_mask_only_allowed_for_frozen_supported_module_and_exact_mapping(self):
        variants = []
        for key, value in (("devices", [0]), ("gpu_mapping", []), ("cpu_only", True)):
            variants.append(dict(self.job, **{key: value}))
        for index, value in ((1, "CUDA_VISIBLE_DEVICES=0"), (4, "other_module"), (6, "sana")):
            command = list(self.job["command"])
            command[index] = value
            variants.append(dict(self.job, command=command))
        variants.append(dict(self.job, command=self.job["command"] + ["--arm", "enabled"]))
        variants.append(dict(self.job, command_type="ordinary"))
        for job in variants:
            with self.subTest(job=job), self.assertRaises(ValueError):
                suite.selected_devices(job)
        with self.assertRaisesRegex(ValueError, "frozen DATA_ROOT"):
            suite.validate_frozen_job(dict(self.spec, strict_resume=False), self.job, self.root)
        command = list(self.job["command"])
        command[8] = "/data/workflow/study/other.json"
        with self.assertRaisesRegex(ValueError, "frozen input/output"):
            suite.validate_frozen_job(self.spec, dict(self.job, command=command), self.root)
        self.config.write_text(self.config.read_text().replace("cuda:0", "cuda:1"))
        with self.assertRaisesRegex(ValueError, "frozen mapping"):
            suite.selected_devices(self.job)

    def test_actual_physical_inventory_mismatch_blocks_before_launch(self):
        class Probe:
            def __init__(self, *_): pass
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def sample(self): return [dict(device=1, uuid="GPU-wrong")]
        path = self.root / "jobs.json"
        path.write_text(json.dumps(self.spec))
        with patch.object(sys, "argv", ["suite", "--jobs", str(path)]), \
                patch.object(suite, "NvmlProbe", Probe), \
                patch.object(suite.subprocess, "Popen", side_effect=AssertionError("must not launch")), \
                self.assertRaisesRegex(ValueError, "physical GPU inventory changed"):
            suite.main()


class WorkflowWatchdogTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.spec_from_file_location("workflow_watchdog", Path(suite.__file__).with_name("workflow-budget-watchdog.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def argv(self, root, deadline="10"):
        return ["watchdog", "--container", "ditserve-workflow-test-012345abcdef", "--parent-pid", "123", "--parent-start", "1",
                "--deadline", deadline, "--monotonic-deadline", deadline, "--deadline-reason", "task_timeout",
                "--event", str(root / "event.json"), "--done", str(root / "done"), "--armed", str(root / "armed"),
                "--launcher-record", str(root / "launcher.json")]

    def test_watchdog_rejects_infinite_deadline_before_arming(self):
        watchdog = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for deadline in ("inf", "nan"):
                with patch.object(sys, "argv", self.argv(root, deadline)), self.assertRaisesRegex(ValueError, "finite"):
                    watchdog.main()
                self.assertFalse((root / "armed").exists())

    def test_task_timeout_still_stops_and_verifies_only_named_container(self):
        watchdog = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(sys, "argv", self.argv(root)), \
                    patch.object(watchdog, "identity", return_value="1"), \
                    patch.object(watchdog.time, "monotonic", return_value=11.), \
                    patch.object(watchdog.subprocess, "run", side_effect=[
                        SimpleNamespace(returncode=0, stdout="", stderr=""),
                        SimpleNamespace(returncode=0, stdout="false\n", stderr="")]) as run:
                watchdog.main()
            event = json.loads((root / "event.json").read_text())
            self.assertEqual(event["reason"], "task_timeout")
            self.assertTrue(event["verified_stopped"])
            self.assertEqual(len(run.call_args_list), 2)
            self.assertIn("stop --time 10 ditserve-workflow-test-012345abcdef", run.call_args_list[0].args[0][-1])


if __name__ == "__main__":
    unittest.main()
