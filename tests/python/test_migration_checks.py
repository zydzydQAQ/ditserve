"""CPU-only acceptance contracts, including real rank ownership and cleanup."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.migration import checks, configure, probes


UUIDS = ["GPU-%08x-0000-0000-0000-000000000000" % i for i in range(8)]


def discovery(memory=40960, count=8):
    return dict(status="available", devices=[dict(index=i, uuid=UUIDS[i], name="NVIDIA A100-SXM4-40GB",
        memory_mib=memory, compute_capability="8.0") for i in range(count)])


class MigrationCheckTests(unittest.TestCase):
    def test_eight_homogeneous_a100_40_and_80_accepted(self):
        for size in (40960, 81920):
            self.assertEqual(len(checks.select_a100(discovery(size))), 8)
        self.assertEqual(len(checks.select_a100(discovery(), UUIDS[4], expected=1)), 1)

    def test_missing_duplicate_alias_and_heterogeneous_devices_fail(self):
        cases = []
        cases.append(discovery(count=7))
        duplicate = discovery()
        duplicate["devices"][1]["uuid"] = UUIDS[0]
        cases.append(duplicate)
        for field, value in (("memory_mib", 81920), ("name", "NVIDIA A100-PCIE-40GB"),
                             ("compute_capability", "7.0")):
            altered = discovery()
            altered["devices"][0][field] = value
            cases.append(altered)
        for value in cases:
            with self.assertRaises(ValueError):
                checks.select_a100(value)
        with self.assertRaises(ValueError):
            checks.select_a100(discovery(), "0," + UUIDS[0], expected=2)

    def test_uuid_is_normalized_in_both_deployment_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_uuid = "  " + UUIDS[3][4:].upper() + "  "
            result = configure.configure(root, "/models", "/data", source_uuid, python="/opt/venv/bin/python")
            self.assertEqual(result["status"], "configured")
            for name in ("environment.json", "scheduling-a100.json"):
                value = json.loads((root / name).read_text())
                self.assertEqual(value["gpu_uuid"], UUIDS[3])
                self.assertNotIn("/home/zyd", json.dumps(value))
            # Equivalent identity is idempotent, a different identity is rejected.
            configure.configure(root, "/models", "/data", UUIDS[3], python="/opt/venv/bin/python")
            with self.assertRaises(ValueError):
                configure.configure(root, "/models", "/data", UUIDS[4], python="/opt/venv/bin/python")

    def test_run_owned_times_out_without_touching_foreign_group(self):
        foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            result = checks.run_owned([sys.executable, "-c", "import time; time.sleep(30)"], timeout=.2)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["reason"], "timeout")
            self.assertTrue(result["cleanup_confirmed"])
            self.assertIsNone(foreign.poll())
            self.assertLess(result["elapsed_seconds"], 5)
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)

    def test_probe_non_object_is_rejected(self):
        result = checks.run_owned([sys.executable, "-c", "print('MIGRATION_RESULT=[]')"], timeout=5)
        self.assertEqual(result["status"], "failed")

    def test_full_dependency_artifact_versions_checked_before_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/artifacts-runtime.json").write_text(json.dumps({"artifacts": [
                {"name": "indirect-dependency", "version": "1.2.3"}]}))
            info = dict(architecture="x86_64", packages={})
            with patch.object(checks, "ROOT", root), patch.object(checks, "inventory", return_value=info), \
                 patch.object(checks.importlib.metadata, "version", return_value="9.9.9"), \
                 patch.object(checks, "run_owned") as run:
                result = checks.cpu_check()
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("indirect-dependency" in error for error in result["errors"]))
            run.assert_not_called()

    def test_core_protocol_is_supervised_with_the_configured_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/artifacts-runtime.json").write_text('{"artifacts": []}')
            info = dict(architecture="x86_64", packages={})
            with patch.object(checks, "ROOT", Path(temporary)), \
                 patch.object(checks, "inventory", return_value=info), \
                 patch.object(checks, "run_owned", return_value=dict(status="passed", output_tail="")) as run:
                result = checks.cpu_check(timeout=7)
            core = run.call_args_list[-1]
            self.assertIn("CoreBridge", " ".join(core.args[0]))
            self.assertEqual(core.kwargs["timeout"], 7)
            self.assertEqual(core.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(result["status"], "passed")

    def test_cuda_probe_rejects_mig_memory_even_if_physical_inventory_is_full_a100(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1,
            get_device_capability=lambda index: (8, 0),
            get_device_properties=lambda index: SimpleNamespace(total_memory=5 * 1024 ** 3)))
        with patch.dict("sys.modules", {"torch": fake}):
            with self.assertRaisesRegex(ValueError, "MIG"):
                probes.cuda_probe()

    def test_cuda_requires_completion_and_nccl_requires_every_rank(self):
        def evidence(command, **kwargs):
            if "ditserve.migration.distributed" in command:
                return dict(status="passed", gpu_seconds=1,
                            probe=dict(status="passed", ranks=8, completion_ranks=list(range(7))))
            return dict(status="passed", gpu_seconds=1, probe=dict(status="passed"))
        with patch("ditserve.research.preflight.discover_gpus", return_value=discovery()), \
             patch.object(checks, "run_owned", side_effect=evidence):
            self.assertEqual(checks.gpu_check()["status"], "failed")
        with patch("ditserve.research.preflight.discover_gpus", return_value=discovery()), \
             patch.object(checks, "run_owned", return_value=dict(status="passed", gpu_seconds=0)) as run:
            self.assertEqual(checks.gpu_check()["status"], "failed")
            self.assertEqual(run.call_count, 1)

    def test_scheduling_first_calibration_honors_bound_uuid_and_never_falls_back(self):
        from ditserve.scheduling import runner
        config = dict(python=sys.executable, v100_python=sys.executable, models={"wan": {}, "sana": {}},
                      gpu_uuid=UUIDS[3], device_preference=["A100"])
        for busy, expected in ((False, "ready"), (True, "blocked")):
            with patch.object(runner, "discover_gpus", return_value=discovery()), \
                 patch.object(runner, "model_inventory", return_value=dict(available=True)), \
                 patch.object(runner, "gpu_occupancy", return_value=dict(
                     status="contended" if busy else "exclusive_observed")) as observe:
                result = runner.inspect(config, phase="calibrate")
            self.assertEqual(result["status"], expected)
            observe.assert_called_once_with([UUIDS[3]])
            if not busy:
                self.assertEqual(result["runtime_config"]["gpu_uuid"], UUIDS[3])

    def test_session_probe_uses_real_finish_string_contract(self):
        import numpy as np
        from ditserve.sessions import runtime
        calls = []
        class LLM:
            def __init__(self, path, profile):
                calls.append(("llm", profile))
            def prepare(self, messages, max_new_tokens):
                return SimpleNamespace(done=False, generated_ids=[])
            def advance(self, state):
                state.generated_ids.append(7)
                state.done = len(state.generated_ids) == 4
            def snapshot(self, state):
                return {"cache": ()}
            def offload_to_cpu(self, state): pass
            def restore(self, state): pass
            def finish(self, state): return "A generated prompt"
            def reference(self, *args, **kwargs): return {"token_ids": [7] * 4}
            def release(self, state): pass
            def close(self): calls.append(("llm_closed", True))
        class Sana:
            def __init__(self, path, profile):
                calls.append(("sana", profile))
            def prepare(self, prompt, seed):
                self.prompt = prompt
                return SimpleNamespace(done=False)
            def advance(self, state): state.done = True
            def offload_to_cpu(self, state): pass
            def restore(self, state): pass
            def finish(self, state): return np.zeros((1, 2, 2, 3))
            def reference(self, prompt, seed):
                self.assert_prompt = prompt
                return {"output": np.zeros((1, 2, 2, 3))}
            def release(self, state): pass
            def close(self): pass
        config = {"models": {"mistral": {"path": "/model"}, "sana": {"path": "/sana"}}}
        with patch.object(runtime, "MistralRuntime", LLM), patch.object(runtime, "SanaRuntime", Sana), \
             patch.object(runtime, "cache_comparison", return_value=dict(passed=True, exact=True)):
            result = probes.sessions_probe(config)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["token_ids"], [7] * 4)
        self.assertEqual(calls, [("llm", "a100_fp32"), ("llm_closed", True), ("sana", "a100_fp32")])


class DistributedOwnershipTests(unittest.TestCase):
    def launch_command(self, code, *, count=2, timeout=5):
        return [sys.executable, "-m", "ditserve.migration.distributed", "--nproc", str(count),
                "--timeout", str(timeout), "--master-port", "12345", "--", sys.executable, "-c", code]

    def test_real_ranks_share_supervised_pgid_and_all_complete(self):
        code = ("import json,os; print('MIGRATION_RESULT='+json.dumps(dict(status='passed',"
                "rank=int(os.environ['RANK']),ranks=int(os.environ['WORLD_SIZE']),pgid=os.getpgrp())))")
        result = checks.run_owned(self.launch_command(code), timeout=10)
        self.assertEqual(result["status"], "passed", result)
        probe = result["probe"]
        self.assertEqual(probe["completion_ranks"], [0, 1])
        self.assertTrue(probe["cleanup_confirmed"])
        self.assertTrue(all(row["probe"]["pgid"] == probe["launcher_pgid"] for row in probe["workers"]))
        self.assertNotEqual(probe["launcher_pgid"], os.getpgrp())

    def test_zero_exit_without_rank_evidence_is_failure(self):
        result = checks.run_owned(self.launch_command("pass"), timeout=10)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["probe"]["completion_ranks"], [])

    def test_wrong_rank_evidence_is_failure(self):
        code = "print('MIGRATION_RESULT={\"status\":\"passed\",\"rank\":0,\"ranks\":2}')"
        result = checks.run_owned(self.launch_command(code), timeout=10)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["probe"]["completion_ranks"], [0])

    def test_rank_failure_stops_other_rank_without_waiting_for_timeout(self):
        code = "import os,time; rank=int(os.environ['RANK']); exit(3) if rank==0 else time.sleep(30)"
        result = checks.run_owned(self.launch_command(code, timeout=20), timeout=25)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["probe"]["reason"], "rank_process_failed")
        self.assertLess(result["elapsed_seconds"], 5)
        self.assertTrue(result["probe"]["cleanup_confirmed"])

    def test_rank_timeout_reaps_children_and_keeps_foreign_process_alive(self):
        foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                code = ("import os,time,pathlib; "
                        "pathlib.Path(%r,os.environ['RANK']).write_text(str(os.getpid())); time.sleep(30)" % temporary)
                result = checks.run_owned(self.launch_command(code, timeout=2), timeout=8)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["probe"]["reason"], "timeout")
                self.assertTrue(result["probe"]["cleanup_confirmed"])
                self.assertEqual(len(list(Path(temporary).iterdir())), 2)
                for path in Path(temporary).iterdir():
                    with self.assertRaises(ProcessLookupError):
                        os.kill(int(path.read_text()), 0)
                self.assertIsNone(foreign.poll())
                self.assertLess(result["elapsed_seconds"], 8)
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)

    def test_outer_timeout_also_stops_rank_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            code = ("import os,time,pathlib; "
                    "pathlib.Path(%r,os.environ['RANK']).write_text(str(os.getpid())); time.sleep(30)" % temporary)
            # Synchronize with real rank startup. A fixed subsecond window can
            # expire before Python starts in a CPU-limited container, which would
            # test an empty launcher instead of descendant cleanup.
            real_popen = subprocess.Popen
            def started_ranks(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                deadline = time.monotonic() + 10
                def ready():
                    values = [path.read_text().strip() for path in Path(temporary).iterdir()]
                    return len(values) == 2 and all(value.isdigit() and int(value) > 1 for value in values)
                while not ready() and time.monotonic() < deadline:
                    time.sleep(.01)
                if not ready():
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                    self.fail("rank startup did not complete before the test deadline")
                return process
            with patch.object(checks.subprocess, "Popen", side_effect=started_ranks):
                result = checks.run_owned(self.launch_command(code, timeout=20), timeout=.4)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["reason"], "timeout")
            self.assertTrue(result["cleanup_confirmed"])
            self.assertEqual(len(list(Path(temporary).iterdir())), 2)
            for path in Path(temporary).iterdir():
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(path.read_text()), 0)


if __name__ == "__main__":
    unittest.main()
