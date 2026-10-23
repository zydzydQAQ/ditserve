"""CPU-only contracts for fresh-process overlap lifecycle controls."""

import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_overlap_isolated as workflow


UUID = "GPU-12345678-1234-1234-1234-123456789abc"
MiB = 1024**2
GiB = 1024**3


def config():
    return workflow.normalize_config(dict(gpu_uuid=UUID, device="cuda:1", cpu_threads=8,
        cpu_masks={"cuda:1": list(range(60, 68))}, frame_pairs=[[17, 17]],
        reuse_streams=True, trim_after_references=True, release_b_before_a_decode=True))


def boundary(allocated=3*GiB, reserved=12*GiB, used=13*GiB, free=11*GiB):
    return dict(allocated_bytes=allocated, reserved_bytes=reserved, state_count=0,
                nvml=dict(memory_used_bytes=used, memory_free_bytes=free, memory_total_bytes=24*GiB),
                nvml_errors=[])


def measurement(capacity=True, free=2*GiB, exact=True):
    hashes = {name: dict(sha256=name, shape=[1, 17, 4], dtype="float32", finite=True)
              for name in workflow.OUTPUTS}
    return dict(**hashes, correctness=dict(passed=exact),
        memory_margin_gate=dict(eligible=capacity, minimum_free_bytes=free, errors=[]),
        peak_allocated_bytes=11*GiB, peak_reserved_bytes=20*GiB, performance_evidence=False)


def completed_child(capacity=True, warmup_pairs=0):
    value_config = dict(config(), warmup_pairs=warmup_pairs)
    value = dict(status="running", stream_ids=[1, 2], rounds=[
        dict(index=i, phase="warmup" if i < warmup_pairs else "measured",
             measured_index=None if i < warmup_pairs else i-warmup_pairs,
             measurement=measurement(capacity=capacity), after_release=boundary(),
             repeat_exact=True, stream_ids=[1, 2]) for i in range(warmup_pairs+3)])
    workflow._finish_child_result(value, value_config)
    return value


class ConfigTests(unittest.TestCase):
    def test_required_causal_controls_and_rebound_cpu_gpu(self):
        value = config()
        self.assertEqual(value["arm_specs"], workflow.ARMS)
        self.assertEqual(value["lifecycle_pairs"], 3)
        self.assertEqual(value["warmup_pairs"], 0)
        self.assertTrue(value["memory_diagnostics"])
        child = workflow.child_config(value, workflow.ARMS[2], [17, 17])
        self.assertEqual(child["device"], "cuda:0")
        self.assertEqual(child["cpu_masks"], {"cuda:0": list(range(60, 68))})
        self.assertTrue(child["decode_a_on_b_stream"])
        self.assertEqual(value["device"], "cuda:1")
        for overrides in ({"lifecycle_pairs": 2}, {"lifecycle_pairs": 17},
                          {"warmup_pairs": -1}, {"warmup_pairs": 17}, {"warmup_pairs": True},
                          {"gpu_uuid": "1"}, {"arm_specs": workflow.ARMS[:2]},
                          {"reuse_streams": False}, {"trim_after_references": False},
                          {"release_b_before_a_decode": False}, {"steady_reserved_growth_bytes": -1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                workflow.normalize_config(dict(value, **overrides))

    def test_steady_growth_uses_post_first_pair_baseline_and_real_nvml(self):
        rows = [boundary(), boundary(reserved=12*GiB+32*MiB), boundary(reserved=12*GiB+64*MiB)]
        gate = workflow.growth_gate(rows, config())
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["metrics"]["reserved_bytes"]["maximum_growth_bytes"], 64*MiB)
        rows[-1]["nvml"]["memory_used_bytes"] += 65*MiB
        self.assertFalse(workflow.growth_gate(rows, config())["passed"])
        self.assertEqual(workflow.growth_gate(rows[:2], config())["status"], "unverified")
        rows[1]["nvml_errors"] = ["query"]
        self.assertEqual(workflow.growth_gate(rows, config())["status"], "unverified")

    def test_emergency_floor_missing_sampling_and_live_states_stop(self):
        value = config()
        self.assertIsNone(workflow.boundary_stop_reason(boundary(), value))
        self.assertEqual(workflow.boundary_stop_reason(boundary(free=511*MiB), value),
                         "emergency_memory_floor")
        self.assertEqual(workflow.boundary_stop_reason(dict(boundary(), state_count=1), value),
                         "request_states_not_released")
        self.assertEqual(workflow.boundary_stop_reason(dict(boundary(), nvml=None), value),
                         "missing_or_failed_nvml_boundary")


class ChildTests(unittest.TestCase):
    def run_child(self, *, measurements=None, boundaries=None, warmup_pairs=0):
        value = workflow.child_config(dict(config(), warmup_pairs=warmup_pairs),
                                      workflow.ARMS[2], [17, 17])
        calls = []
        cuda = types.SimpleNamespace(device_count=lambda: 1, synchronize=lambda *_: calls.append("sync"),
                                     empty_cache=lambda: calls.append("trim"))
        worker = types.SimpleNamespace(torch=types.SimpleNamespace(cuda=cuda), device="cuda:0",
                                       workflow_cpu_affinity={"bound_cpus": list(range(60, 68))})
        references = {(role, 17): dict(directory=role, hashes={"video": role, "latent": role})
                      for role in ("A", "B")}
        timeline = types.SimpleNamespace(records=[])
        values = measurements or [measurement() for _ in range(warmup_pairs+3)]
        snapshots = boundaries or [boundary() for _ in range(warmup_pairs+4)]
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=UUID), \
                patch.object(workflow.overlap, "make_worker", return_value=worker), \
                patch.object(workflow.overlap, "_references", return_value=references), \
                patch.object(workflow.overlap, "Timeline", return_value=timeline), \
                patch.object(workflow.overlap, "_memory_boundary", return_value={"reserved_bytes": 5*GiB}), \
                patch.object(workflow, "_boundary", side_effect=snapshots), \
                patch.object(workflow, "_stream_identity", return_value=[1, 2]), \
                patch.object(workflow.overlap, "_run_arm", side_effect=values) as arm:
            result = workflow.run_child(value, temp)
            stored = json.loads((Path(temp) / "run.json").read_text())
        self.assertEqual(stored, result)
        return result, calls, arm.call_args_list

    def test_three_consecutive_pairs_one_startup_trim_no_timed_arms(self):
        result, calls, arms = self.run_child()
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["qualification"]["passed"])
        self.assertFalse(result["performance_evidence"])
        self.assertEqual(result["unrun_pairs"], [])
        self.assertEqual(calls.count("trim"), 1)
        self.assertEqual([call.args[6] for call in arms], [0, 1, 2])
        self.assertEqual([call.args[7] for call in arms], ["lifecycle"]*3)
        self.assertTrue(all(row["repeat_exact"] for row in result["rounds"]))
        self.assertEqual(result["completed_warmup_pairs"], 0)
        self.assertEqual(result["completed_measured_pairs"], 3)
        self.assertEqual(result["steady_state"]["measured_pair_indices"], [0, 1, 2])

    def test_fixed_warmups_preserve_every_pair_and_only_exclude_growth(self):
        # The cache grows in warmup, then remains fixed through three measured
        # pairs. Exactness, margins, release and stream identity still use all five.
        snapshots = [boundary(), boundary(reserved=12*GiB, used=13*GiB)]
        snapshots += [boundary(reserved=13*GiB, used=14*GiB) for _ in range(4)]
        result, calls, arms = self.run_child(warmup_pairs=2, boundaries=snapshots)
        self.assertTrue(result["qualification"]["passed"])
        self.assertEqual(calls.count("trim"), 1)
        self.assertEqual([call.args[6] for call in arms], list(range(5)))
        self.assertEqual([call.args[7] for call in arms],
                         ["lifecycle_warmup"]*2 + ["lifecycle"]*3)
        self.assertEqual([row["phase"] for row in result["rounds"]],
                         ["warmup"]*2 + ["measured"]*3)
        self.assertEqual([row["measured_index"] for row in result["rounds"]],
                         [None, None, 0, 1, 2])
        self.assertEqual(result["completed_pairs"], 5)
        self.assertEqual(result["expected_total_pairs"], 5)
        self.assertEqual(result["completed_warmup_pairs"], 2)
        self.assertEqual(result["completed_measured_pairs"], 3)
        self.assertEqual(result["unrun_measured_pairs"], [])
        self.assertEqual(result["unrun_warmup_pairs"], [])
        self.assertEqual(result["steady_state"]["excluded_warmup_pairs"], 2)
        self.assertEqual(result["steady_state"]["measured_pair_indices"], [2, 3, 4])
        self.assertEqual(result["steady_state"]["metrics"]["reserved_bytes"]["values"],
                         [13*GiB]*3)
        compact = workflow._compact_child(result)
        self.assertEqual(len(compact["output_hashes"]), 5)
        self.assertEqual([row["phase"] for row in compact["memory_series"]],
                         ["warmup"]*2 + ["measured"]*3)

    def test_measured_growth_threshold_is_not_relaxed_after_warmup(self):
        snapshots = [boundary() for _ in range(5)]
        snapshots.append(boundary(reserved=12*GiB+65*MiB))
        result, _, _ = self.run_child(warmup_pairs=2, boundaries=snapshots)
        self.assertEqual(result["status"], "lifecycle_failed")
        gate = result["steady_state"]["metrics"]["reserved_bytes"]
        self.assertEqual(gate["limit_bytes"], 64*MiB)
        self.assertEqual(gate["maximum_growth_bytes"], 65*MiB)
        self.assertFalse(gate["passed"])

    def test_warmup_capacity_failure_is_retained_without_unbounded_retry(self):
        values = [measurement(capacity=False, free=900*MiB)] + [measurement() for _ in range(4)]
        result, calls, arms = self.run_child(warmup_pairs=2, measurements=values)
        self.assertEqual(len(arms), 5)
        self.assertEqual(calls.count("trim"), 1)
        self.assertEqual(result["status"], "capacity_failed")
        self.assertFalse(result["qualification"]["passed"])
        self.assertTrue(result["steady_state"]["passed"])
        self.assertEqual(result["qualification"]["scopes"]["capacity"]["status"], "failed")

    def test_warmup_numeric_or_emergency_failure_stops_before_measurement(self):
        for bad, reason in ((measurement(exact=False), "numeric_mismatch"),
                            (measurement(capacity=False, free=500*MiB), "emergency_hot_memory_floor")):
            with self.subTest(reason=reason):
                result, _, arms = self.run_child(warmup_pairs=2, measurements=[bad])
                self.assertEqual(len(arms), 1)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(result["completed_warmup_pairs"], 1)
                self.assertEqual(result["completed_measured_pairs"], 0)
                self.assertEqual(result["unrun_pairs"], [1, 2, 3, 4])
                self.assertEqual(result["unrun_warmup_pairs"], [1])
                self.assertEqual(result["unrun_measured_pairs"], [0, 1, 2])
                self.assertFalse(result["qualification"]["passed"])

    def test_margin_failure_above_emergency_floor_still_observes_three_pairs(self):
        result, calls, arms = self.run_child(measurements=[
            measurement(capacity=False, free=900*MiB) for _ in range(3)])
        self.assertEqual(len(arms), 3)
        self.assertEqual(result["status"], "capacity_failed")
        self.assertFalse(result["qualification"]["passed"])
        self.assertEqual(result["qualification"]["scopes"]["numeric"]["status"], "passed")
        self.assertEqual(calls.count("trim"), 1)

    def test_emergency_low_hot_margin_stops_and_marks_missing_pairs(self):
        result, calls, arms = self.run_child(measurements=[measurement(capacity=False, free=500*MiB)])
        self.assertEqual(len(arms), 1)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["unrun_pairs"], [1, 2])
        self.assertEqual(result["stop_reason"], "emergency_hot_memory_floor")
        self.assertFalse(result["qualification"]["passed"])

    def test_hash_drift_and_post_release_margin_are_not_qualified(self):
        values = [measurement() for _ in range(3)]
        values[1]["a_video"]["sha256"] = "changed"
        result, _, arms = self.run_child(measurements=values)
        self.assertEqual(len(arms), 2)
        self.assertEqual(result["stop_reason"], "numeric_mismatch")
        snapshots = [boundary()] + [boundary(free=900*MiB) for _ in range(3)]
        result, _, arms = self.run_child(boundaries=snapshots)
        self.assertEqual(result["status"], "capacity_failed")
        self.assertEqual(len(arms), 3)

    def test_worker_exception_is_durable_and_not_qualified(self):
        result, _, _ = self.run_child(measurements=[RuntimeError("CUDA out of memory")])
        self.assertEqual(result["status"], "failed")
        self.assertIn("out of memory", result["error"])
        self.assertEqual(result["unrun_pairs"], [0, 1, 2])
        self.assertFalse(result["qualification"]["passed"])


class ParentTests(unittest.TestCase):
    def test_warmed_aggregate_requires_all_five_outputs_and_cross_arm_exactness(self):
        value = dict(config(), warmup_pairs=2)
        rows = [dict(id=arm["id"], frame_pair=[17, 17],
                     result=workflow._compact_child(completed_child(warmup_pairs=2)),
                     launch=dict(reaped=True, returncode=0, timed_out=False))
                for arm in workflow.ARMS]
        gate = workflow.aggregate(rows, value)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["cross_arm"][0]["expected_pairs_per_arm"], 5)
        self.assertEqual(gate["control_cross_arm"][0]["expected_pairs"], 5)
        self.assertTrue(gate["control_cross_arm"][0]["complete"])
        rows[2]["result"]["output_hashes"][0]["a_video"]["sha256"] = "changed-warmup"
        self.assertFalse(workflow.aggregate(rows, value)["passed"])
        rows[2]["result"]["output_hashes"] = rows[2]["result"]["output_hashes"][2:]
        self.assertFalse(workflow.aggregate(rows, value)["passed"])

    def test_failing_control_does_not_hide_or_certify_independent_repair(self):
        rows = []
        for arm in workflow.ARMS:
            child = completed_child(capacity=arm["id"] != "streams-current-decode")
            rows.append(dict(id=arm["id"], frame_pair=[17, 17], result=workflow._compact_child(child),
                             launch=dict(reaped=True, returncode=0, timed_out=False)))
        gate = workflow.aggregate(rows, config())
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["control_cross_arm"][0]["exact"])
        rows[2]["result"]["output_hashes"][1]["a_video"]["sha256"] = "changed"
        self.assertFalse(workflow.aggregate(rows, config())["passed"])
        rows[2]["launch"]["timed_out"] = True
        self.assertFalse(workflow.aggregate(rows, config())["passed"])

    def test_wrapper_runs_each_arm_sequentially_with_immutable_inputs(self):
        calls = []
        def launch(value, directory):
            calls.append((copy.deepcopy(value), directory.name))
            child = completed_child()
            return child, dict(reaped=True, returncode=0, timed_out=False)
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_launch_child", launch):
            result = workflow.run_isolated(config(), temp)
            with self.assertRaises(FileExistsError):
                workflow.run_isolated(config(), temp)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([value["selected_arm"] for value, _ in calls], workflow.ARMS)
        self.assertTrue(all(value["device"] == "cuda:0" for value, _ in calls))
        self.assertFalse(result["performance_evidence"])

    def test_timeout_reaps_before_returning_and_binds_uuid(self):
        class Process:
            pid, returncode = 4321, None
            def __init__(self):
                self.waits, self.terminated, self.killed = 0, False, False
            def wait(self, timeout):
                self.waits += 1
                if self.waits < 3:
                    raise subprocess.TimeoutExpired("child", timeout)
                self.returncode = -9
                return self.returncode
            def terminate(self):
                self.terminated = True
            def kill(self):
                self.killed = True
            def poll(self):
                return self.returncode
        process = Process()
        with tempfile.TemporaryDirectory() as temp, patch.object(
                workflow.subprocess, "Popen", return_value=process) as popen:
            child, launch = workflow._launch_child(
                workflow.child_config(config(), workflow.ARMS[0], [17, 17]), Path(temp))
            self.assertTrue(launch["reaped"])
            self.assertTrue(launch["timed_out"])
            self.assertTrue(process.terminated and process.killed)
            self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], UUID)
            self.assertEqual(child, {})
            self.assertTrue((Path(temp) / "child-launch.json").exists())

    def test_real_cpu_children_are_fresh_isolated_and_reaped_sequentially(self):
        # Exercise the real OS subprocess path without importing torch or
        # invoking the model child. Only the command payload is replaced.
        original_popen = subprocess.Popen
        script = (
            "import json,os,sys;from pathlib import Path;"
            "Path(sys.argv[1],'run.json').write_text(json.dumps({"
            "'pid':os.getpid(),'uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),"
            "'threads':os.environ.get('OMP_NUM_THREADS'),'torch_loaded':'torch' in sys.modules}))"
        )
        def cpu_process(command, **kwargs):
            return original_popen([command[0], "-c", script, command[-1]], **kwargs)
        with tempfile.TemporaryDirectory() as temp, patch.object(
                workflow.subprocess, "Popen", side_effect=cpu_process):
            children = [workflow._launch_child(
                workflow.child_config(config(), arm, [17, 17]), Path(temp) / arm["id"])
                for arm in workflow.ARMS]
        self.assertEqual(len({child["pid"] for child, _ in children}), 3)
        for child, launch in children:
            self.assertEqual(child["uuid"], UUID)
            self.assertEqual(child["threads"], "8")
            self.assertFalse(child["torch_loaded"])
            self.assertTrue(launch["reaped"])
            self.assertEqual(launch["returncode"], 0)
        self.assertTrue(all(left[1]["finished_unix_ns"] <= right[1]["started_unix_ns"]
                            for left, right in zip(children, children[1:])))


if __name__ == "__main__":
    unittest.main()
