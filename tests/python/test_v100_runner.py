"""Mock-only checks for trustworthy V100 pairing; never access a real GPU."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


from ditserve.experiments import run_v100 as runner
UUID = "GPU-a058066e-bde9-c534-efb1-d2a68c117681"
OTHER_UUID = "GPU-b058066e-bde9-c534-efb1-d2a68c117681"
ENV = {"CUDA_VISIBLE_DEVICES": UUID, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
       "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
GPU = {"physical_index": 1, "uuid": UUID, "name": "Mock V100", "pci_bus_id": "mock"}


def fixture(root):
    args = runner.make_parser().parse_args(["--stage", "smoke", "--no-export-media"])
    args.revision = None
    config = runner.load_config(runner.DEFAULT_CONFIG)
    job = runner.build_plan(args, config, "float16", Path(root))[0]
    profile = {
        "status": "completed", "kind": "gpu_profile", "measured": True,
        "config": dict(job["shape"], dtype="float16", cache="none", seed=0,
                       warmup=1, repeats=1, batch_size=1, device="cuda:0"),
        "environment": {"gpu": {"uuid": UUID}, "cuda_visible_devices": UUID,
                        "cuda_device_order": "PCI_BUS_ID", "device": "cuda:0"},
        "runs": [{"output": {}}],
        "aggregate": {"e2e_ms_median": 100, "pipeline_plus_refresh_ms_median": 101,
                      "peak_allocated_bytes_max": 1024},
    }
    return args, job, profile


def telemetry_result(stdout=None):
    return subprocess.CompletedProcess([], 0, stdout or UUID + ", 40, 20, 55, 155.25, 1234, 32768\n", "")


def fake_record(job, status="completed"):
    # Different ratios make it observable if superseded FP16 data leaks into FP32.
    baseline = 100 if job["dtype"] == "float32" else 1000
    milliseconds = baseline if job["arm"] == "none" else 50
    output = {"finite": True, "sha256": "a" * 64}
    profile = {"status": status, "kind": "gpu_profile", "measured": True,
               "config": dict(job["shape"], dtype=job["dtype"], cache=job["arm"], seed=job["seed"]),
               "runs": [{"status": "completed", "measured": True,
                         "uninstrumented": {"e2e_ms": milliseconds, "peak_allocated_bytes": 1024},
                         "output": output, "diagnostic": {"output": output},
                         "instrumentation_output_exact_match": True}]}
    runner.write_json(Path(job["directory"]) / "profile.json", profile)
    return dict(job, status=status, failure_category=None if status == "completed" else status,
                raw_log=str(Path(job["directory"]) / "raw.log"), artifacts=[],
                aggregate={"e2e_ms_median": milliseconds,
                           "pipeline_plus_refresh_ms_median": milliseconds,
                           "peak_allocated_bytes_max": 1024},
                telemetry={"status": "completed", "samples": 1, "complete_samples": 1,
                           "path": str(Path(job["directory"]) / "gpu-telemetry.jsonl"), "metrics": {}})


class PairingTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(runner, "archive_run", return_value={"report": "/mock/REPORT.md"})
        guard.start()
        self.addCleanup(guard.stop)

    def test_fp16_nonfinite_restarts_whole_selected_suite_in_fp32(self):
        attempts = []

        def execute(job, args, environment):
            attempts.append(copy.deepcopy(job))
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], UUID)
            self.assertEqual(environment["OMP_NUM_THREADS"], "8")
            status = "nonfinite" if job["dtype"] == "float16" and job["stage"] == "formal" else "completed"
            return fake_record(job, status)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner, "resolve_gpu", return_value=GPU), \
                patch.object(runner, "run_one", side_effect=execute):
            exit_code = runner.main(["--stage", "all", "--output", tmp, "--no-export-media"])
            summary = json.loads((Path(tmp) / "v100-summary.json").read_text())
            records = json.loads((Path(tmp) / "records.json").read_text())
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(attempts), 17)  # Two FP16 smoke arms, failed formal, full FP32 suite.
        fp32 = [job for job in attempts if job["dtype"] == "float32"]
        self.assertEqual(len(fp32), 14)
        self.assertEqual([job["stage"] for job in fp32[:2]], ["smoke", "smoke"])
        self.assertEqual(len([record for record in records if record["status"] == "nonfinite"]), 1)
        self.assertEqual(summary["counts"]["formal_completed_pairs"], 6)
        self.assertTrue(summary["manifest"]["fp32_rerun"])
        self.assertTrue(all(pair["dtype"] == "float32" for pair in summary["pairs"]))
        self.assertTrue(all(pair["e2e_baseline_over_cache"] == 2 for pair in summary["pairs"]))

    def test_smoke_oom_preserves_shape_and_blocks_formal(self):
        attempts = []

        def execute(job, args, environment):
            attempts.append(copy.deepcopy(job))
            return fake_record(job, "oom" if job["arm"] == "none" else "completed")

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner, "resolve_gpu", return_value=GPU), \
                patch.object(runner, "run_one", side_effect=execute):
            exit_code = runner.main(["--stage", "all", "--output", tmp, "--no-export-media"])
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
        self.assertEqual(exit_code, 2)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(manifest["suite_result"]["skipped_stages"], ["formal"])
        self.assertFalse(manifest["fp32_rerun"])
        for job in attempts:
            self.assertEqual(job["shape"], {"height": 256, "width": 256, "frames": 17, "steps": 12})
            self.assertEqual(job["dtype"], "float16")

    def test_formal_oom_is_retained_without_shape_reduction_or_hidden_pair(self):
        attempts = []

        def execute(job, args, environment):
            attempts.append(copy.deepcopy(job))
            return fake_record(job, "oom" if len(attempts) == 1 else "completed")

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner, "resolve_gpu", return_value=GPU), \
                patch.object(runner, "run_one", side_effect=execute):
            exit_code = runner.main(["--stage", "formal", "--output", tmp, "--no-export-media"])
            summary = json.loads((Path(tmp) / "v100-summary.json").read_text())
        self.assertEqual(exit_code, 2)
        self.assertEqual(len(attempts), 12)
        self.assertEqual(summary["manifest"]["status"], "incomplete")
        self.assertEqual(summary["counts"]["formal_completed_pairs"], 5)
        self.assertEqual(len(summary["pairs"]), 6)
        self.assertEqual(summary["pairs"][0]["status"], "incomplete")
        self.assertNotIn("e2e_baseline_over_cache", summary["pairs"][0])
        for job in attempts:
            self.assertEqual(job["shape"], {"height": 480, "width": 832, "frames": 17, "steps": 50})
            self.assertEqual(job["dtype"], "float16")


class DeviceContractTests(unittest.TestCase):
    def test_matching_config_cannot_hide_different_actual_gpu(self):
        _, job, profile = fixture("/tmp/never-executed")
        self.assertIsNone(runner.validate_profile_contract(profile, job, ENV))
        profile["environment"]["gpu"]["uuid"] = OTHER_UUID
        self.assertIn("GPU UUID", runner.validate_profile_contract(profile, job, ENV))

    def test_missing_unknown_or_ambiguous_device_identity_fails_closed(self):
        _, job, base = fixture("/tmp/never-executed")
        for selector in ("0", OTHER_UUID, UUID + "," + OTHER_UUID, None):
            with self.subTest(selector=selector):
                profile = copy.deepcopy(base)
                profile["environment"]["cuda_visible_devices"] = selector
                self.assertIn("CUDA_VISIBLE_DEVICES", runner.validate_profile_contract(profile, job, ENV))
        for uuid in ("unknown", None, "0"):
            profile = copy.deepcopy(base)
            profile["environment"]["gpu"]["uuid"] = uuid
            self.assertIn("GPU UUID", runner.validate_profile_contract(profile, job, ENV))
        profile = copy.deepcopy(base)
        del profile["environment"]
        self.assertIsNotNone(runner.validate_profile_contract(profile, job, ENV))

    def test_uuid_prefix_and_case_are_normalized_but_device_is_checked(self):
        _, job, profile = fixture("/tmp/never-executed")
        profile["environment"]["gpu"]["uuid"] = UUID[4:].upper()
        self.assertIsNone(runner.validate_profile_contract(profile, job, ENV))
        profile["environment"]["device"] = "cuda:1"
        self.assertIn("logical device", runner.validate_profile_contract(profile, job, ENV))


class TelemetryTests(unittest.TestCase):
    def test_target_only_sample_preserves_timestamps_and_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler = runner.GpuTelemetry(tmp, UUID)
            with patch.object(runner.subprocess, "run", return_value=telemetry_result()) as mock:
                sample = sampler.sample()
            self.assertEqual(sample["status"], "completed")
            self.assertEqual(sample["utilization_gpu_percent"], 40)
            self.assertEqual(sample["power_draw_w"], 155.25)
            self.assertEqual(sample["memory_used_mib"], 1234)
            self.assertEqual(mock.call_args[0][0][1], "--id=" + UUID)
            self.assertEqual(mock.call_args[1]["timeout"], 2.0)
            for clock in ("unix_ns", "monotonic_ns"):
                self.assertLessEqual(sample["query_started_" + clock], sample["query_finished_" + clock])
            with patch.object(runner.subprocess, "run", return_value=telemetry_result(UUID + ", 40, 20, 55, N/A, 1234, 32768\n")):
                sample = sampler.sample()
            self.assertEqual(sample["status"], "partial_failure")
            self.assertIsNone(sample["power_draw_w"])
            with patch.object(runner.subprocess, "run", return_value=telemetry_result(OTHER_UUID + ", 40, 20, 55, 150, 1234, 32768\n")):
                self.assertEqual(sampler.sample()["status"], "failed")
            with patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired(["mock"], 2)):
                self.assertEqual(sampler.sample()["error"]["type"], "TimeoutExpired")

    def test_periodic_sampler_stops_and_flushes(self):
        calls = []
        second_sample = threading.Event()

        def fake(*args, **kwargs):
            calls.append(time.monotonic())
            if len(calls) == 2:
                second_sample.set()
            return telemetry_result()

        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.subprocess, "run", side_effect=fake):
            sampler = runner.GpuTelemetry(tmp, UUID, 2)
            sampler.start()
            try:
                self.assertTrue(second_sample.wait(4))
            finally:
                summary = sampler.stop()
            self.assertFalse(sampler._thread.is_alive())
            self.assertGreaterEqual(calls[1] - calls[0], 1.9)
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(len(Path(summary["path"]).read_text().splitlines()), 2)

    def test_profiler_success_failure_timeout_and_interrupt_all_stop_sampler(self):
        for failure in (None, OSError("mock launch error"), subprocess.TimeoutExpired(["mock"], 1), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as tmp:
                args, job, profile = fixture(tmp)
                sampled = threading.Event()

                def fake(command, **kwargs):
                    if command[0] == "nvidia-smi":
                        sampled.set()
                        return telemetry_result()
                    self.assertTrue(sampled.wait(1))
                    if failure is not None:
                        raise failure
                    runner.write_json(Path(job["directory"]) / "profile.json", profile)
                    return subprocess.CompletedProcess(command, 0)

                with patch.object(runner.subprocess, "run", side_effect=fake):
                    record = runner.run_one(job, args, ENV)
                self.assertEqual(record["status"], "completed" if failure is None else "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed")
                self.assertEqual(record["telemetry"]["status"], "completed")
                self.assertTrue(Path(record["telemetry"]["path"]).exists())
                self.assertFalse(any(thread.name == "v100-gpu-telemetry" for thread in threading.enumerate()))
                saved = json.loads((Path(job["directory"]) / "record.json").read_text())
                self.assertLessEqual(saved["profiler_started_unix_ns"], saved["profiler_finished_unix_ns"])

    def test_nonfinite_preflight_shares_nonfinite_failure_class(self):
        self.assertEqual(runner.classify_result(2, {"failure": {"category": "nonfinite_preflight"}}, ""),
                         ("nonfinite", "nonfinite_preflight"))


if __name__ == "__main__":
    unittest.main()
