import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ditserve.research.common import atomic_json, read_json
from ditserve.research.isolation import GPUIsolationError, gpu_occupancy, wait_for_exclusive_worker
from ditserve.research.preflight import inspect_environment


class GPUIsolationTests(unittest.TestCase):
    def result(self, text):
        return lambda *a, **kw: subprocess.CompletedProcess(a, 0, text, "")

    def test_only_this_worker_group_may_use_selected_devices(self):
        run = self.result("GPU-A, 10\nGPU-A, 11\nGPU-B, 20\n")
        snapshot = gpu_occupancy(["GPU-A"], 10, run=run, getpgid=lambda pid: 10)
        self.assertEqual(snapshot["status"], "exclusive_observed")
        self.assertEqual([p["pid"] for p in snapshot["owned"]], [10, 11])
        snapshot = gpu_occupancy(["GPU-A"], 10, run=run, getpgid=lambda pid: pid)
        self.assertEqual(snapshot["status"], "contended")
        self.assertEqual([p["pid"] for p in snapshot["foreign"]], [11])
        self.assertEqual(gpu_occupancy(["GPU-A"], run=run)["status"], "contended")

    def test_unknown_inventory_or_ownership_fails_closed(self):
        for text in ("GPU-A, N/A\n", "GPU-A, -1\n", "invalid\n"):
            self.assertEqual(gpu_occupancy(["GPU-A"], run=self.result(text))["status"], "unknown")
        denied = Mock(side_effect=PermissionError("denied"))
        self.assertEqual(gpu_occupancy(["GPU-A"], 1, run=self.result("GPU-A, 1\n"),
                                      getpgid=denied)["status"], "unknown")
        self.assertEqual(gpu_occupancy(["GPU-A"], run=self.result(""))["status"], "exclusive_observed")

    def test_foreign_process_arriving_mid_unit_invalidates_costs_without_signalling_it(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen = dict(sha256="protocol", source_sha256="source")
            process = Mock(pid=100, args=["owned-worker"])
            process.wait.side_effect = subprocess.TimeoutExpired("owned-worker", 2)
            observe = Mock(side_effect=[dict(status="exclusive_observed"),
                                       dict(status="contended", foreign=[dict(pid=200)])])
            with patch("os.kill") as kill, patch("os.killpg") as killpg:
                with self.assertRaises(GPUIsolationError):
                    wait_for_exclusive_worker(process, timeout=10, output=directory, frozen=frozen,
                                              stage="probe", devices=["GPU-A"], observe=observe)
                kill.assert_not_called()
                killpg.assert_not_called()
            summary = read_json(Path(directory) / "measurement-environment.json")
            self.assertFalse(summary["cost_measurements_valid"])
            self.assertEqual(summary["stages"]["probe"]["observations_count"], 2)
            process.terminate.assert_not_called()

    def test_monitor_checks_completion_and_preserves_earlier_contamination(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen = dict(sha256="protocol", source_sha256="source")
            process = Mock(pid=100, args=["owned-worker"])
            process.wait.return_value = 0
            observe = Mock(side_effect=[dict(status="exclusive_observed"), dict(status="unknown", reason="driver read failed")])
            with self.assertRaises(GPUIsolationError):
                wait_for_exclusive_worker(process, timeout=10, output=directory, frozen=frozen,
                                          stage="probe", devices=["GPU-A"], observe=observe)
            self.assertFalse(read_json(Path(directory) / "measurement-environment.json")["cost_measurements_valid"])

    def test_busy_gpu_is_blocked_at_read_only_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "revision"
            model.mkdir()
            (model / "model_index.json").write_text("{}")
            def run(command, **kwargs):
                output = ("0, GPU-A, A100, 40960, 8.0\n" if "--query-gpu=index,uuid,name,memory.total,compute_cap" in command
                          else "GPU-A, 100\n")
                return subprocess.CompletedProcess(command, 0, output, "")
            frozen = dict(config=dict(dependencies={}, revision="revision"))
            result = inspect_environment(frozen, ["0"], model_path=model, run=run)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["gpu_occupancy"]["status"], "contended")
            self.assertFalse(result["gpu_work_started"])



if __name__ == "__main__":
    unittest.main()
