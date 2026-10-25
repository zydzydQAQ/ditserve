"""Persistent JSONL framing and state semantics, without CUDA or model weights."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ditserve.runtime.worker import BACKENDS, Worker, occupancy, serve


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = dict(work_dir=self.directory.name)
        self.worker = Worker("mock", self.config)
        self.addCleanup(self.worker.close)
        self.sequence = 0

    def call(self, op, **fields):
        self.sequence += 1
        return self.worker.dispatch(dict(version=1, id=self.sequence, op=op, **fields))

    def test_lifecycle_and_deterministic_pause_restore(self):
        outputs = []
        for name, pause in (("a", False), ("b", True)):
            result = self.call("prepare", state_id=name, spec=dict(seed=71, prompt="boat", steps=3))
            self.assertEqual(result["progress"], dict(step=0, steps=3, done=False, next_action="advance"))
            self.call("advance", state_id=name)
            if pause:
                snapshot = self.call("snapshot", state_id=name, mode="cold")
                with self.assertRaisesRegex(ValueError, "restore"):
                    self.call("advance", state_id=name)
                self.call("restore", snapshot_id=snapshot["snapshot_id"])
            result = self.call("advance", state_id=name, steps=8)
            self.assertTrue(result["progress"]["done"])
            output = self.call("finish", state_id=name)["output"]
            outputs.append(output)
            self.assertTrue(Path(output["path"]).is_file())
            self.assertGreater(output["bytes"], 0)
            self.call("release", state_id=name)
            self.assertTrue(self.call("release", state_id=name)["already_released"])
        self.assertEqual(outputs[0]["sha256"], outputs[1]["sha256"])
        self.assertTrue(self.call("evaluate", reference=outputs[0]["path"], output=outputs[1]["path"])["exact"])

    def test_invalid_commands_do_not_advance_state(self):
        for request in ([], {}, dict(version=2, id=1, op="hello"),
                        dict(version=1, id=True, op="hello"), dict(version=1, id=1, op="missing")):
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.worker.dispatch(request)
        self.call("prepare", state_id="a", spec=dict(steps=1))
        for op, fields in (("prepare", dict(state_id="a")), ("finish", dict(state_id="a")),
                           ("advance", dict(state_id="a", steps=0)), ("advance", dict(state_id="other"))):
            with self.subTest(op=op), self.assertRaises(ValueError):
                self.call(op, **fields)
        self.assertFalse(self.call("inventory", state_id="a")["paused"])
        self.assertEqual(self.call("inventory", state_id="a")["step"], 0)

    def test_removed_model_backends_are_rejected(self):
        for backend in ("mistral", "longlive"):
            with self.subTest(backend=backend):
                self.assertNotIn(backend, BACKENDS)
                with self.assertRaisesRegex(ValueError, "unknown backend"):
                    Worker(backend, self.config)

    def test_xdit_capabilities_describe_the_actual_execution_contract(self):
        from ditserve.models.xdit import XDIT_REVISION
        cases = (("wan", "block", ["step", "block"], ["state", "block"]),
                 ("wan_pipeline", None, ["atomic"], []),
                 ("sana", "pipeline", ["atomic"], []),
                 ("flux", None, ["atomic"], []),
                 ("sd35", None, ["atomic"], []))
        for backend, execution, quanta, migration in cases:
            with self.subTest(backend=backend):
                worker = Worker(backend, dict(options={} if execution is None else dict(execution=execution)))
                self.addCleanup(worker.close)
                result = worker.dispatch(dict(version=1, id=1, op="hello"))
                capabilities = result["capabilities"]
                self.assertEqual(capabilities["engine"]["revision"], XDIT_REVISION)
                self.assertEqual(capabilities["engine_status"], "configured")
                self.assertEqual((capabilities["quanta"], capabilities["migration"]), (quanta, migration))
                self.assertTrue(capabilities["model_wrapper"].startswith("xFuser"))
                self.assertEqual(capabilities["continuous_batch"], backend == "sana" and execution != "pipeline")

    def test_framing_errors_stdout_and_recovery(self):
        commands = ["not json", '{"version":1,"id":1,"op":"hello","value":NaN}',
            json.dumps(dict(version=1, id=2, op="prepare", state_id="a",
                            spec=dict(steps=1), inject=dict(print="model banner"))),
            json.dumps(dict(version=1, id=3, op="advance", state_id="a", inject=dict(fail_on="advance"))),
            json.dumps(dict(version=1, id=4, op="advance", state_id="a")),
            json.dumps(dict(version=1, id=5, op="shutdown"))]
        output, diagnostics = io.StringIO(), io.StringIO()
        with patch("sys.stderr", diagnostics):
            serve(self.worker, io.StringIO("\n".join(commands)), output)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row["ok"] for row in rows], [False, False, True, False, True, True])
        self.assertEqual(rows[3]["error"]["type"], "RuntimeError")
        self.assertIn("model banner", diagnostics.getvalue())
        self.assertNotIn("model banner", output.getvalue())
        self.assertTrue(self.worker.closed)
        self.assertFalse(self.worker.backend.states)

    def test_subprocess_all_hello_paths_are_lazy(self):
        config = Path(self.directory.name) / "config.json"
        config.write_text(json.dumps(self.config))
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                payload = '\n'.join(json.dumps(dict(version=1, id=i, op=op))
                                    for i, op in enumerate(("hello", "shutdown"))) + '\n'
                result = subprocess.run([sys.executable, "-m", "ditserve.runtime.worker", "--backend", backend,
                    "--config", str(config)], input=payload, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = [json.loads(line) for line in result.stdout.splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertTrue(rows[0]["ok"], rows)
                self.assertFalse(rows[0]["result"]["loaded"])
                self.assertTrue(rows[1]["ok"])

    def test_occupancy_requires_full_uuid_and_accounts_owned_groups(self):
        self.assertEqual(occupancy({}, mock=True)["status"], "exclusive_observed")
        self.assertEqual(occupancy({})["status"], "unknown")
        self.assertEqual(occupancy(dict(gpu_uuids=["0"]))["status"], "unknown")
        hardware = subprocess.CompletedProcess([], 0, "GPU-abcdef, Mock GPU\n", "")
        apps = subprocess.CompletedProcess([], 0, "10, GPU-abcdef, 100\n11, GPU-abcdef, 200\n", "")
        with patch("ditserve.runtime.worker.subprocess.run", side_effect=[hardware, apps]), patch("os.getpgid", side_effect=[2, 3]):
            result = occupancy(dict(gpu_uuids=["GPU-abcdef"], allowed_pgids=[2]))
        self.assertEqual(result["status"], "busy")
        self.assertEqual(result["owned"][0]["pid"], 10)
        self.assertEqual(result["foreign"][0]["pid"], 11)
        with patch("ditserve.runtime.worker.subprocess.run", side_effect=[hardware, apps]), patch("os.getpgid", side_effect=ProcessLookupError):
            result = occupancy(dict(gpu_uuids=["GPU-abcdef"], allowed_pids=[10]))
        self.assertEqual(result["status"], "busy")
        self.assertEqual(result["owned"][0]["pid"], 10)
        self.assertEqual(result["foreign"][0]["pid"], 11)
        self.assertIsNone(result["owned"][0]["pgid"])

    def test_mps_control_requires_owned_server_and_verifies_exit(self):
        with self.assertRaises(ValueError):
            self.call("mps_control", action="start", server_pid=123, pipe_directory=self.directory.name)
        replies = [subprocess.CompletedProcess([], 0, "123\n", ""),
                   subprocess.CompletedProcess([], 0, "", "")]
        with patch("ditserve.runtime.worker.subprocess.run", side_effect=replies) as run, \
                patch("ditserve.runtime.worker.os.kill", side_effect=ProcessLookupError):
            result = self.call("mps_control", action="quit", server_pid=123,
                               pipe_directory=self.directory.name)
        self.assertEqual(result, dict(action="quit", server_pid=123, shutdown=True, verified=True))
        self.assertEqual([call.kwargs["input"] for call in run.call_args_list], ["get_server_list\n", "quit\n"])
        for call in run.call_args_list:
            self.assertEqual(call.args[0], ["nvidia-cuda-mps-control"])
            self.assertFalse(call.kwargs.get("shell", False))
            self.assertEqual(call.kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"], self.directory.name)
        with patch("ditserve.runtime.worker.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0, "999\n", "")) as run:
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                self.call("mps_control", action="quit", server_pid=123, pipe_directory=self.directory.name)
        self.assertEqual(run.call_count, 1)

    def test_parent_eof_closes_live_state(self):
        line = json.dumps(dict(version=1, id=1, op="prepare", state_id="a"))
        serve(self.worker, io.StringIO(line), io.StringIO())
        self.assertTrue(self.worker.closed)
        self.assertFalse(self.worker.backend.states)

    def test_mock_migration_requires_target_ack_and_source_release(self):
        target = Worker("mock", self.config)
        self.addCleanup(target.close)
        self.call("prepare", state_id="a", spec=dict(steps=2))
        self.call("advance", state_id="a")
        snapshot = self.call("snapshot", state_id="a", target_device="GPU-target",
                             migration_id="m1", path=str(Path(self.directory.name) / "migration.json"))
        ack = target.dispatch(dict(version=1, id=1, op="restore", state_id="b",
                                   path=snapshot["path"], sha256=snapshot["sha256"]))
        with self.assertRaises(ValueError):
            target.dispatch(dict(version=1, id=2, op="advance", state_id="b"))
        with self.assertRaisesRegex(ValueError, "commit or abort"):
            target.dispatch(dict(version=1, id=2, op="restore", state_id="b"))
        receipt = self.call("migration_release", migration_id="m1", acknowledgement=ack["acknowledgement"])
        restored = target.dispatch(dict(version=1, id=3, op="restore", migration_id="m1", receipt=receipt["receipt"]))
        self.assertEqual(restored["progress"]["step"], 1)
        self.assertNotIn("a", self.worker.backend.states)
        result = target.dispatch(dict(version=1, id=4, op="advance", state_id="b"))
        self.assertTrue(result["progress"]["done"])

        self.call("prepare", state_id="c", spec=dict(steps=2))
        snapshot = self.call("snapshot", state_id="c", target_device="GPU-target", migration_id="m2",
                             path=str(Path(self.directory.name) / "abort.json"))
        target.dispatch(dict(version=1, id=5, op="restore", state_id="d", path=snapshot["path"], sha256=snapshot["sha256"]))
        target.dispatch(dict(version=1, id=6, op="migration_abort", migration_id="m2"))
        self.call("migration_abort", migration_id="m2")
        self.assertNotIn("d", target.backend.states)
        self.assertEqual(self.call("advance", state_id="c")["progress"]["step"], 1)

    def test_text_evaluation_is_exact_and_does_not_claim_perceptual_quality(self):
        left, right = (Path(self.directory.name) / name for name in ("left.txt", "right.txt"))
        left.write_text("hello")
        right.write_text("hello")
        result = self.call("evaluate", reference=str(left), output=str(right))
        self.assertTrue(result["passed"])
        self.assertEqual(result["perceptual_quality"], "unverified")
        right.write_text("world")
        self.assertFalse(self.call("evaluate", reference=str(left), output=str(right))["passed"])

    def test_evaluation_gpu_requires_accounted_device_and_never_uses_mock(self):
        request = dict(version=1, id=1, op="evaluate", evaluator="vbench")
        for name, config in (("mock", {}), ("evaluation", {}),
                             ("evaluation", dict(device="cuda:0", gpu_uuid="GPU-test"))):
            worker = Worker(name, config)
            with self.subTest(name=name, config=config), patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    worker.dispatch(request)
                self.assertIsNone(worker.backend)
        with self.assertRaisesRegex(ValueError, "CPU evaluation only"):
            self.call("evaluate", device="cuda:0")
        worker = Worker("evaluation", dict(device="cuda:0", gpu_uuid="GPU-test"))
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-test"), patch("ditserve.evaluation.reference.evaluate", return_value={"status": "stub"}) as evaluate:
            self.assertEqual(worker.dispatch(request)["status"], "stub")
            evaluate.assert_called_once_with(request)
        self.assertIsNone(worker.backend)


if __name__ == "__main__":
    unittest.main()
