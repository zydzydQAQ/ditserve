"""CPU-only admission and command construction for the privileged-profiler wrapper."""

import fcntl
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch


from ditserve.evaluation.wan_profile import capture


class WaitClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class CaptureWanHardwareTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.p = capture.paths(self.root)
        self.p["sources"].mkdir(parents=True)
        self.p["ledger"].parent.mkdir(parents=True)
        (self.p["work"] / "run.lock").touch()
        Path(str(self.p["ledger"]) + ".lock").touch()
        source = {"files": {}, "native_files": {}}
        for name in ("ditserve/cli.py", "ditserve/models/wan_profile.py", "ditserve/evaluation/nsys_metrics.py"):
            path = self.p["sources"] / "python" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# frozen CPU test fixture\n")
            source["files"][name] = capture.digest(path)
        binary = self.p["sources"] / "ditserve-core"
        binary.write_text("fixture, never executed")
        binary.chmod(0o700)
        source["core_sha256"] = capture.digest(binary)
        self.save(self.p["sources"] / "manifest.json", source)
        config = {"workers": {"model": {"backend": "wan_profile", "gpu_uuid": capture.GPU_UUID,
                   "python": str(self.p["python"]), "config": {"required_uid": 1002}}},
                  "phases": [{"id": "software_trace"}, {"id": "hardware_metrics", "admission_wait_seconds": 0,
                              "actions": [{"mode": "hardware"}]}]}
        self.run = {"schema": "ditserve.run.v1", "experiment": capture.EXPERIMENT, "run_id": "run01",
                    "status": "incomplete", "budget_ledger": str(self.p["ledger"]), "source": source,
                    "config": config, "config_sha256": capture.canonical_digest(config),
                    "phases": [{"id": "software_trace", "status": "completed", "session_id": "one",
                                "qualification": {"qualified": True}}]}
        self.ledger = {"schema_version": "ditserve.budget.v1", "active": None, "allocation_id": "fixture",
                       "limits": {"total_gpu_seconds": 3600.0, "repair_reserve_gpu_seconds": 300.0,
                                  "carry": "none", "phase_order": ["software_trace", "hardware_metrics"], "phase_gpu_seconds": {}},
                       "used_gpu_seconds": 1000.0, "used_regular_gpu_seconds": 1000.0,
                       "sessions": [{"session_id": "one", "status": "completed", "cleanup_verified": True,
                                     "gpu_seconds": 1000.0, "identity": {"run_id": "run01",
                                     "config_sha256": self.run["config_sha256"], "source_sha256": capture.canonical_digest(source)}}]}
        self.save(self.p["run"], self.run)
        self.save(self.p["ledger"], self.ledger)

    def save(self, path, value):
        path.write_text(json.dumps(value))

    def inspect(self):
        with patch.object(capture.os, "geteuid", return_value=1002), \
                patch.object(capture.subprocess, "run", side_effect=AssertionError("CPU preflight must not run sudo/GPU commands")):
            return capture.inspect_run(self.root)

    def test_completed_bound_software_is_ready_without_running_any_command(self):
        result = self.inspect()
        self.assertTrue(result["ready"], result)
        self.assertEqual(result["remaining_regular_gpu_seconds"], 2300)
        self.assertEqual(result["gpu_status"], "not_queried")
        self.assertFalse(self.p["output"].exists())

    def test_module_layout_keeps_frozen_source_binding(self):
        old, new = "ditserve/models/wan_profile.py", "ditserve/models/wan/profile.py"
        target = self.p["sources"] / "python" / new
        target.parent.mkdir(parents=True)
        (self.p["sources"] / "python" / old).rename(target)
        source = self.run["source"]
        source["files"][new] = source["files"].pop(old)
        self.save(self.p["sources"] / "manifest.json", source)
        self.save(self.p["run"], self.run)
        # Moving sources without rebinding the fixture's ledger is rejected.
        self.assertFalse(self.inspect()["ready"])
        self.ledger["sessions"][0]["identity"]["source_sha256"] = capture.canonical_digest(source)
        self.save(self.p["ledger"], self.ledger)
        self.assertTrue(self.inspect()["ready"])

    def test_native_launcher_requires_bound_main_and_client(self):
        source = self.run["source"]
        del source["files"]["ditserve/cli.py"]
        (self.p["sources"] / "python/ditserve/cli.py").unlink()
        for name in ("ditserve/__main__.py", "ditserve/runtime/client.py"):
            path = self.p["sources"] / "python" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# native-launcher fixture\n")
            source["files"][name] = capture.digest(path)
            self.save(self.p["sources"] / "manifest.json", source)
            self.save(self.p["run"], self.run)
            self.assertFalse(self.inspect()["ready"])
        self.ledger["sessions"][0]["identity"]["source_sha256"] = capture.canonical_digest(source)
        self.save(self.p["ledger"], self.ledger)
        self.assertEqual(self.inspect()["frozen_frontend"], "native")
        command = capture.capture_command(self.p, 0)
        binary = str(self.p["sources"] / "ditserve-core")
        self.assertIn(binary, command)
        self.assertNotIn("-m", command)
        for option, value in (("--root", self.root), ("--package-root", self.p["sources"] / "python"),
                              ("--python", self.p["python"]), ("--ledger", self.p["ledger"])):
            self.assertEqual(command[command.index(option) + 1], str(value))

    def test_frozen_parser_supports_original_and_relative_imports(self):
        package = self.p["sources"] / "python/ditserve"
        for directory in (package, package / "evaluation"):
            (directory / "__init__.py").write_text("")
        (package / "evaluation/common.py").write_text("SOURCE = 'frozen-relative'\n")
        parser = package / "evaluation/nsys_metrics.py"
        fixture_paths = dict(self.p, python=Path(sys.executable))
        for header, expected in (("SOURCE = 'original-standalone'", "original-standalone"),
                                 ("from .common import SOURCE", "frozen-relative")):
            with self.subTest(layout=expected):
                parser.write_text(header + "\ndef summarize_nsys_metrics(path, *, expected_sample_hz):\n"
                                  "    return {'source': SOURCE, 'path': path, 'hz': expected_sample_hz}\n")
                result = capture.frozen_metrics(fixture_paths, self.root / "fixture.sqlite")
                self.assertEqual(result["source"], expected)
                self.assertEqual(result["hz"], 1000.0)
                self.assertFalse(list(package.rglob("*.pyc")))

    def test_waiting_software_and_live_controller_lock_block(self):
        self.run["status"] = "waiting_resource"
        self.save(self.p["run"], self.run)
        self.assertFalse(self.inspect()["ready"])
        self.run["status"] = "incomplete"
        self.save(self.p["run"], self.run)
        with (self.p["work"] / "run.lock").open() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.inspect()
        self.assertFalse(result["ready"])
        self.assertIn("BlockingIOError", result["reasons"][0])

    def test_root_must_never_run_the_wrapper(self):
        with patch.object(capture.os, "geteuid", return_value=0):
            result = capture.inspect_run(self.root)
        self.assertFalse(result["ready"])
        self.assertIn("do not sudo", result["reasons"][0])

    def test_source_tampering_existing_output_and_exhausted_ledger_block(self):
        binary = self.p["sources"] / "ditserve-core"
        original = binary.read_bytes()
        binary.write_text("modified")
        self.assertFalse(self.inspect()["ready"])
        binary.write_bytes(original)
        self.p["output"].mkdir()
        self.assertFalse(self.inspect()["ready"])
        self.p["output"].rmdir()
        self.ledger["used_gpu_seconds"] = self.ledger["used_regular_gpu_seconds"] = 3290
        self.ledger["sessions"][0]["gpu_seconds"] = 3290
        self.save(self.p["ledger"], self.ledger)
        self.assertFalse(self.inspect()["ready"])

    def test_only_fixed_profiler_is_sudoed_and_target_uses_same_run_ledger(self):
        command = capture.capture_command(self.p, 0)
        self.assertEqual(command[:3], ["sudo", "--", str(capture.NSYS)])
        for option in ("--run-as=zyd", "--inherit-environment=false", "--kill=none", "--resume",
                       "--sample=none", "--cpuctxsw=none", "--gpu-metrics-frequency=1000"):
            self.assertIn(option, command)
        self.assertEqual(command[command.index("--phase") + 1], "hardware_metrics")
        self.assertEqual(command[command.index("--ledger") + 1], str(self.p["ledger"]))
        self.assertEqual(command[command.index("--work-dir") + 1], str(self.p["work"]))
        self.assertTrue(any("DITSERVE_CORE=" + str(self.p["sources"] / "ditserve-core") in part for part in command))
        self.assertEqual(command[command.index("-m") - 1:command.index("-m") + 2],
                         [str(self.p["python"]), "-m", "ditserve"])

    def test_publisher_module_preserves_root_package_and_failure_status(self):
        for captured, published, expected in ((0, 0, 0), (0, 2, 1), (1, 0, 1)):
            with self.subTest(captured=captured, published=published), \
                    patch.object(capture, "capture", return_value=captured), \
                    patch.object(capture, "ROOT", self.root), \
                    patch.object(capture, "paths", return_value=self.p), \
                    patch.dict(os.environ, PYTHONPATH="unrelated", DITSERVE_ROOT="unrelated"), \
                    patch.object(capture.subprocess, "run", return_value=
                        capture.subprocess.CompletedProcess([], published)) as run:
                self.assertEqual(capture.main([]), expected)
                self.assertEqual(run.call_args.args[0],
                    [str(self.p["python"]), "-m", "ditserve.evaluation.wan_profile.publish"])
                self.assertEqual(run.call_args.kwargs["env"]["PYTHONPATH"], str(capture.PACKAGE_ROOT))
                self.assertEqual(run.call_args.kwargs["env"]["DITSERVE_ROOT"], str(self.root))

    def test_unsupported_device_help_cannot_launch_models(self):
        gpu = {"index": 0}
        self.assertFalse(capture._supported("Some GPUs are not supported:\n NVIDIA A100-PCIE-40GB", gpu))
        self.assertFalse(capture._supported("1: Ampere GA100 | NVIDIA A100-PCIE-40GB", gpu))
        self.assertFalse(capture._supported("0: NVIDIA A100-PCIE-40GB (Unsupported GPU)", gpu))
        self.assertFalse(capture._supported("Some GPUs are not supported:\n0: NVIDIA A100-PCIE-40GB", gpu))
        self.assertFalse(capture._supported("0: NVIDIA A100-PCIE-40GB (no permission)", gpu))
        self.assertTrue(capture._supported("0: Ampere GA100 | NVIDIA A100-PCIE-40GB PCI[0000:86:00.0]", gpu))
        gpu["pci_bus_id"] = "00000000:86:00.0"
        self.assertTrue(capture._supported("0: Ampere GA100 | NVIDIA A100-PCIE-40GB PCI[0000:86:00.0]", gpu))
        self.assertFalse(capture._supported("0: Ampere GA100 | NVIDIA A100-PCIE-40GB PCI[0000:87:00.0]", gpu))

    def test_wait_requires_two_consecutive_idle_probes_and_refreshes_only_existing_ticket(self):
        check, clock = self.inspect(), WaitClock()
        gpu = {"index": 0, "uuid": capture.GPU_UUID}
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with patch.object(capture, "inspect_run", return_value=check), \
                patch.object(capture.time, "monotonic", side_effect=clock.monotonic), \
                patch.object(capture.time, "sleep", side_effect=clock.sleep), \
                patch.object(capture, "_gpu_idle", side_effect=[capture.GpuBusy("busy"), gpu,
                    capture.GpuBusy("busy again"), gpu, gpu]) as probe, \
                patch.object(capture.subprocess, "run", return_value=
                    capture.subprocess.CompletedProcess([], 0)) as sudo, redirect_stdout(io.StringIO()):
            result = capture.wait_for_admission(3600)
        self.assertEqual(result["wait_seconds"], 60)
        self.assertEqual(probe.call_count, 5)
        self.assertEqual(clock.sleeps, [15, 15, 15, 15])
        self.assertEqual(sudo.call_count, 2)
        self.assertTrue(all(call.args[0] == ["sudo", "-n", "-v"] for call in sudo.call_args_list))
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})
        self.assertFalse(self.p["output"].exists())

    def test_wait_timeout_creates_no_capture_and_respects_deadline(self):
        check, clock = self.inspect(), WaitClock()
        with patch.object(capture, "inspect_run", return_value=check), \
                patch.object(capture.time, "monotonic", side_effect=clock.monotonic), \
                patch.object(capture.time, "sleep", side_effect=clock.sleep), \
                patch.object(capture, "_gpu_idle", side_effect=capture.GpuBusy("busy")) as probe, \
                patch.object(capture.subprocess, "run", return_value=capture.subprocess.CompletedProcess([], 0)), \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(TimeoutError, "timed out"):
            capture.wait_for_admission(40)
        self.assertEqual(clock.now, 40)
        self.assertEqual(clock.sleeps, [15, 15, 10])
        self.assertEqual(probe.call_count, 3)
        self.assertTrue(all(call.kwargs["deadline"] == 40 for call in probe.call_args_list))
        self.assertFalse(self.p["output"].exists())

    def test_expired_authentication_exits_without_interactive_fallback(self):
        check, clock = self.inspect(), WaitClock()
        with patch.object(capture, "inspect_run", return_value=check), \
                patch.object(capture.time, "monotonic", side_effect=clock.monotonic), \
                patch.object(capture.time, "sleep", side_effect=clock.sleep), \
                patch.object(capture, "_gpu_idle", side_effect=capture.GpuBusy("busy")) as probe, \
                patch.object(capture.subprocess, "run", side_effect=[
                    capture.subprocess.CompletedProcess([], 0), capture.subprocess.CompletedProcess([], 1)]) as sudo, \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "authentication expired"):
            capture.wait_for_admission(3600)
        self.assertEqual(clock.now, 60)
        self.assertEqual(probe.call_count, 4)
        self.assertTrue(all(call.args[0] == ["sudo", "-n", "-v"] for call in sudo.call_args_list))
        command = capture.capture_command(self.p, 0, non_interactive=True)
        self.assertEqual(command[:4], ["sudo", "-n", "--", str(capture.NSYS)])

    def test_queue_aborts_when_source_binding_or_gpu_identity_is_unknown(self):
        check = self.inspect()
        for changed in (dict(check, source_sha256="changed"), {"ready": False, "reasons": ["source mismatch"]}):
            clock = WaitClock()
            with patch.object(capture, "inspect_run", side_effect=[check, changed]), \
                    patch.object(capture.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(capture.time, "sleep", side_effect=clock.sleep), \
                    patch.object(capture, "_gpu_idle", side_effect=capture.GpuBusy("busy")) as probe, \
                    patch.object(capture.subprocess, "run", return_value=capture.subprocess.CompletedProcess([], 0)), \
                    redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                capture.wait_for_admission(3600)
            self.assertEqual(probe.call_count, 1)
        with patch.object(capture, "inspect_run", return_value=check), \
                patch.object(capture, "_gpu_idle", side_effect=ValueError("UUID missing")), \
                patch.object(capture.time, "sleep", side_effect=AssertionError("must not wait on unknown GPU")), \
                patch.object(capture.subprocess, "run", return_value=capture.subprocess.CompletedProcess([], 0)), \
                self.assertRaisesRegex(ValueError, "UUID missing"):
            capture.wait_for_admission(3600)

    def test_check_only_never_waits_or_refreshes_authentication(self):
        with patch.object(capture, "inspect_run", return_value={"ready": False, "reasons": ["busy"]}), \
                patch.object(capture, "wait_for_admission", side_effect=AssertionError("check-only must not wait")), \
                patch.object(capture, "capture", side_effect=AssertionError("check-only must not capture")), \
                patch.object(capture.subprocess, "run", side_effect=AssertionError("check-only must not sudo")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(capture.main(["--check-only", "--wait-seconds", "3600"]), 1)

    def test_successful_wait_rechecks_capture_in_noninteractive_mode(self):
        with patch.object(capture, "wait_for_admission", return_value={}) as wait, \
                patch.object(capture, "capture", return_value=0) as collect, \
                patch.object(capture.subprocess, "run", return_value=capture.subprocess.CompletedProcess([], 0)):
            self.assertEqual(capture.main(["--wait-seconds", "3600"]), 0)
        wait.assert_called_once_with(3600)
        collect.assert_called_once_with(non_interactive=True)

    def test_cancelled_wait_does_not_capture_publish_or_write_output(self):
        check = self.inspect()
        with patch.object(capture, "inspect_run", return_value=check), \
                patch.object(capture, "_gpu_idle", side_effect=capture.GpuBusy("busy")), \
                patch.object(capture.time, "sleep", side_effect=KeyboardInterrupt), \
                patch.object(capture, "capture", side_effect=AssertionError("cancelled queue must not capture")), \
                patch.object(capture.subprocess, "run", return_value=capture.subprocess.CompletedProcess([], 0)) as sudo, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(capture.main(["--wait-seconds", "3600"]), 130)
        self.assertEqual(sudo.call_count, 1)
        self.assertEqual(sudo.call_args.args[0], ["sudo", "-n", "-v"])
        self.assertFalse(self.p["output"].exists())


if __name__ == "__main__":
    unittest.main()
