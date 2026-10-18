"""Native user commands run without PYTHONPATH; all evidence stays in fixtures."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get("DITSERVE_CORE", ROOT / ".cache/build/ditserve"))


@unittest.skipUnless(BINARY.is_file(), "build the native ditserve executable first")
class NativeCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                                TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        self.environment.pop("PYTHONPATH", None)

    def invoke(self, arguments, *, code=0, payload=None, python=True, environment=None, package_root=None):
        command = [str(BINARY), "--root", str(self.root), "--package-root", str(package_root or ROOT / "workers")]
        if python:
            command += ["--python", sys.executable]
        completed = subprocess.run(command + list(arguments), cwd=self.root,
                                   env=environment or self.environment,
                                   input=json.dumps(payload) if payload is not None else None,
                                   text=True, capture_output=True, timeout=45)
        self.assertEqual(completed.returncode, code, completed.stdout + completed.stderr)
        return completed

    def structured(self, arguments, **kwargs):
        return json.loads(self.invoke(arguments, **kwargs).stdout)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def study(self, *, failure=False):
        return {"schema": "ditserve.study.v1", "experiment": "cli-fixture",
                "workers": {"model": {"backend": "mock", "config": {
                    "options": {"fail_on": "advance"} if failure else {}}}},
                "budget": {"limit_gpu_seconds": 10, "watchdog_seconds": 10},
                "phases": [{"id": "online", "kind": "online", "policy": "edf", "workload": [
                    {"request_id": "one", "arrival_ms": 0, "service_ms": 1, "slo_ms": 1000,
                     "spec": {"steps": 1, "prompt": "CPU CLI fixture", "seed": 7}}]}]}

    def fake_worker(self, code):
        package = self.root / "fake-package/ditserve"
        (package / "runtime").mkdir(parents=True)
        (package / "__init__.py").touch()
        (package / "runtime/__init__.py").touch()
        (package / "runtime/worker.py").write_text(code)
        return package.parent

    def test_help_errors_and_low_level_rpc_need_no_python(self):
        environment = dict(self.environment, DITSERVE_PYTHON=str(self.root / "absent-python"))
        for arguments in (["--help"], ["--version"], ["run", "--help"], ["results", "--help"]):
            with self.subTest(arguments=arguments):
                self.assertTrue(self.invoke(arguments, python=False, environment=environment).stdout.strip())
        for arguments in ([], ["unknown"], ["run"], ["results", "unknown"],
                          ["simulate", "--count", "bad"], ["rpc", "unknown"],
                          ["simulate", "--unexpected"], ["report"]):
            with self.subTest(arguments=arguments):
                self.invoke(arguments, code=2, python=False, environment=environment)
        cases = [
            ("generate", {"generator": {"count": 2, "seed": 7, "pattern": "poisson", "rate_rps": 2, "classes": [
                {"class": "video", "kind": "video", "service_ms": 3400}, {"class": "image", "kind": "image", "service_ms": 40}]}}),
            ("control", {"op": "matrix", "phase": {"id": "fixture", "matrix": {"/policy": ["edf", "fcfs"]}}}),
        ]
        generated = None
        for operation, payload in cases:
            with self.subTest(operation=operation):
                value = self.structured(["rpc", operation], payload=payload, python=False, environment=environment)
                self.assertEqual(len(value), 2)
                if operation == "generate":
                    generated = value
        value = self.structured(["rpc", "simulate"], payload={"policy": "block_elastic", "gpus": 2,
            "workload": generated}, python=False, environment=environment)
        self.assertEqual(value["metrics"]["requests"], 2)
        self.assertEqual(value["provenance"], "synthetic")
        self.assertEqual(len(value["requests"]), 2)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_doctor_is_metadata_only_without_pythonpath(self):
        empty_path = self.root / "empty-path"
        empty_path.mkdir()
        environment = dict(self.environment, PATH=str(empty_path))
        value = self.structured(["doctor", "--json"], environment=environment)
        self.assertEqual(Path(value["python_executable"]).resolve(), Path(sys.executable).resolve())
        self.assertTrue(value["reference_simulation_ready"])
        self.assertFalse(value["gpu_driver"]["available"])
        self.assertIn("torch", value["gpu_stack_versions"])
        self.assertTrue(value["notice"])
        interpreter = self.root / ".venv-previs/bin/python"
        interpreter.parent.mkdir(parents=True)
        marker = self.root / "selected-interpreter"
        interpreter.write_text("#!/bin/sh\nprintf selected > %s\nexec %s \"$@\"\n" %
                               (shlex.quote(str(marker)), shlex.quote(sys.executable)))
        interpreter.chmod(0o700)
        environment.pop("DITSERVE_PYTHON", None)
        self.structured(["doctor"], python=False, environment=environment)
        self.assertEqual(marker.read_text(), "selected")

    def test_simulate_publishes_verifiable_two_file_archive(self):
        value = self.structured(["--json", "simulate", "--count", "2", "--seed", "9", "--run-id", "sim"])
        self.assertEqual(value["status"], "completed")
        self.assertEqual(value["budget"]["gpu_seconds"], 0)
        destination = self.root / "results/synthetic/sim"
        self.assertEqual({path.name for path in destination.iterdir()}, {"results.xlsx", "REPORT.md"})
        verified = self.structured(["results", "verify", str(destination)])
        saved = self.structured(["results", "inspect", str(destination)])
        self.assertEqual(verified["status"], "verified")
        self.assertEqual(saved["run"]["run_id"], "sim")
        self.assertEqual(saved["run"]["phases"], value["phases"])

    def test_mock_resume_report_and_new_compact_preserve_completed_evidence(self):
        config = self.write("study.json", self.study())
        for options in (["--resume"], ["--repair"], ["--retry-of", "missing-session"]):
            with self.subTest(invalid=options):
                self.invoke(["run", "--config", str(config)] + options, code=2)
        self.assertFalse((self.root / ".state").exists())
        arguments = ["run", "--json", "--config", str(config), "--run-id", "mock",
                     "--results-root", str(self.root / "custom-results")]
        first = self.structured(arguments)
        self.assertEqual(first["status"], "completed")
        self.assertIsNone(first["budget"]["active"])
        self.assertEqual(first["budget"]["used_gpu_seconds"], 0)
        ledger = self.root / ".state/cli-fixture/budget.json"
        charged = ledger.read_bytes()
        resumed = self.structured(arguments + ["--resume"])
        self.assertEqual(resumed["phases"], first["phases"])
        self.assertEqual(ledger.read_bytes(), charged)
        destination = self.root / "custom-results/cli-fixture/mock"
        work = self.root / ".cache/results-work/cli-fixture/mock"
        before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in destination.iterdir()}
        for command in (["report", str(work), "--keep-raw"],
                        ["report", str(work), "--output", str(destination), "--keep-raw"],
                        ["results", "compact", str(work), "--destination", str(destination)]):
            with self.subTest(command=command[0]):
                self.assertEqual(self.structured(command)["status"], "completed")
        self.assertEqual({path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in destination.iterdir()}, before)

    def test_legacy_compact_alias_and_explicit_fixture_cleanup(self):
        for index, prefix in enumerate((["results", "compact"], ["compact-results"])):
            with self.subTest(prefix=prefix):
                raw = self.root / ("legacy-%d" % index)
                raw.mkdir()
                self.write(raw.name + "/run.json", {"schema_version": "ditserve.benchmark.run.v1"})
                (raw / "output.npy").write_bytes(b"CPU fixture only")
                arguments = prefix + [str(raw)] + (["--prune-raw"] if index else [])
                self.assertEqual(self.structured(arguments)["status"], "verified")
                self.assertEqual((raw / "output.npy").exists(), not bool(index))
                self.assertEqual(self.structured(["results", "verify", str(raw / "excel")])["status"], "verified")

    def test_failed_execution_and_protected_report_have_nonzero_status(self):
        config = self.write("failed.json", self.study(failure=True))
        value = self.structured(["run", "--config", str(config), "--run-id", "failure", "--json"], code=1)
        self.assertEqual(value["status"], "failed")
        self.assertIsNone(value["budget"]["active"])
        self.assertIn("injected mock advance failure", value["phases"][0]["error"])
        work = self.root / "active"
        self.write("active/run.json", {"schema": "ditserve.run.v1", "experiment": "active", "run_id": "fixture",
                                      "status": "running", "budget": {"active": {"session_id": "fixture"}}})
        output = self.root / "unpublished"
        result = self.structured(["report", str(work), "--output", str(output)], code=1)
        self.assertEqual(result["status"], "protected")
        self.assertFalse(output.exists())

    def test_result_service_preserves_json_larger_than_four_mib(self):
        package = self.fake_worker(
            "import json,sys\n"
            "for line in sys.stdin:\n"
            "    request = json.loads(line)\n"
            "    result = {'evidence': 'x' * (5 * 1024 * 1024) + 'END'}\n"
            "    print(json.dumps(dict(version=1,id=request['id'],ok=True,result=result)),flush=True)\n")
        value = self.structured(["results", "inspect", str(self.root)], package_root=package)
        self.assertEqual(value["evidence"], "x" * (5 * 1024 * 1024) + "END")

    def test_cancelled_service_reaps_its_worker_and_descendant(self):
        marker = self.root / "fixture-pids.json"
        child_code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(60)"
        package = self.fake_worker(
            "import json,os,subprocess,sys,time\nfrom pathlib import Path\n"
            "json.loads(sys.stdin.readline())\n"
            "child = subprocess.Popen([sys.executable, '-c', %r], stdout=subprocess.PIPE)\n"
            "child.stdout.readline()\n"
            "path = Path(%r)\n"
            "temporary = path.with_suffix('.tmp')\n"
            "temporary.write_text(json.dumps(dict(worker=os.getpid(), child=child.pid)))\n"
            "temporary.replace(path)\n"
            "time.sleep(60)\n" % (child_code, str(marker)))
        process = subprocess.Popen([str(BINARY), "--root", str(self.root), "--package-root", str(package),
                                    "--python", sys.executable, "doctor"],
                                   cwd=self.root, env=self.environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        pids = {}
        try:
            deadline = time.monotonic() + 10
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(marker.exists(), "fixture worker did not start")
            pids = json.loads(marker.read_text())
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 2, stdout + stderr)
            self.assertIn("cancelled", stderr)
            for role, pid in pids.items():
                with self.subTest(role=role), self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
            for pid in pids.values():
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
