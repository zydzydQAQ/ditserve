"""CPU tests for isolated normal timing, evidence gates, and child cleanup."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_overlap_performance as workflow

UUID = "GPU-12345678-1234-1234-1234-123456789abc"
GiB = 1024**3
MiB = 1024**2


def config():
    return workflow.normalize_config(dict(device="cuda:1", gpu_uuid=UUID, cpu_threads=8,
        cpu_masks={"cuda:1": list(range(60, 68))}, frame_pairs=[[17, 17]], reuse_streams=True,
        trim_after_references=True, release_b_before_a_decode=True, memory_diagnostics=False))


def boundary():
    return dict(allocated_bytes=3*GiB, reserved_bytes=12*GiB, state_count=0,
                nvml=dict(memory_used_bytes=13*GiB, memory_free_bytes=11*GiB, memory_total_bytes=24*GiB, uuid=UUID),
                nvml_errors=[])


def hashes():
    return {name: dict(sha256=hashlib.sha256(name.encode()).hexdigest(), shape=[1, 17, 4],
                      dtype="float32", finite=True) for name in workflow.OUTPUTS}


def measurement(index=0, arm=None):
    arm = arm or workflow.ARMS[0]
    start = (index+1) * 100000000
    finish = start + 20000000
    samples = [dict(timestamp_ns=start+offset, memory_free_bytes=11*GiB, memory_total_bytes=24*GiB, uuid=UUID)
               for offset in (1, 10000000, 19999999)]
    return dict(**hashes(), mode=arm["mode"], repeat=index,
                purpose="performance_warmup" if index < 2 else "timed",
                correctness=dict(passed=True, exact={key: True for key in workflow.OUTPUTS}),
                memory_margin_gate=workflow.overlap.memory_margin_gate(samples, [], start, finish),
                peak_allocated_bytes=11*GiB, peak_reserved_bytes=20*GiB,
                performance_evidence=index >= 2, memory_diagnostics=False, memory_boundaries=[],
                a_decode_stream="b_stream" if arm["decode_a_on_b_stream"] else "current_stream",
                started_monotonic_ns=start, finished_monotonic_ns=finish,
                stage_started_monotonic_ns=start+1000, stage_finished_monotonic_ns=finish-1000,
                wall_ms=20., stage_pair_wall_ms=19.998, nvml_samples=samples, nvml=dict(errors=[]),
                sample_interval_seconds=.1)


def completed_child(arm=None, block=0, position=0):
    arm = arm or workflow.ARMS[0]
    c = workflow.child_config(config(), arm, block)
    streams = [0, 0] if arm["mode"] == "serial" else [0, 123]
    result = dict(schema=workflow.SCHEMA, role="isolated_performance_arm", configuration=c,
                  process=dict(pid=100+position, start_ticks=1000+position),
                  started_unix_ns=position*1000+100, finished_unix_ns=position*1000+800,
                  cpu_affinity=dict(bound_cpus=list(range(60, 68))), startup_cache_trims=1,
                  baseline=boundary(), stream_ids=streams,
                  references=[dict(role=role, frames=17, path=role,
                                   latent=hashes()[role.lower()+"_latent"],
                                   video=hashes()[role.lower()+"_video"]) for role in ("A", "B")],
                  rounds=[dict(index=i, phase="warmup" if i < 2 else "measured",
                               measured_index=None if i < 2 else i-2,
                               measurement=measurement(i, arm), after_release=boundary(),
                               repeat_exact=True, stream_ids=streams) for i in range(5)])
    workflow._finish_child(result, c)
    return result


def completed_result():
    c = config()
    result = dict(schema=workflow.SCHEMA, role="isolated_performance_comparison", configuration=c,
                  status="completed", performance_evidence=True, arms=[], timing_scope=workflow.overlap.TIMING_SCOPE,
                  overlap_requires_cuda_trace=True, qualification=workflow._qualification(True))
    for index, (block, arm) in enumerate(workflow.arm_schedule(c)):
        child = completed_child(arm, block, index)
        launch = dict(pid=child["process"]["pid"], returncode=0, reaped=True, timed_out=False,
                      result_sha256=hashlib.sha256((json.dumps(child, indent=2, ensure_ascii=False,
                          allow_nan=False) + "\n").encode()).hexdigest(),
                      started_unix_ns=index*1000+1, finished_unix_ns=index*1000+900)
        result["arms"].append(dict(block=block, id=arm["id"], result=child, launch=launch))
    result["summary"] = workflow._statistics(result)
    return result


class ContractTests(unittest.TestCase):
    def test_fixed_qualified_matrix_and_normal_timing(self):
        value = config()
        self.assertEqual([(block, arm["id"]) for block, arm in workflow.arm_schedule(value)],
                         [(0, "serial"), (0, "streams-shared-decode"),
                          (1, "streams-shared-decode"), (1, "serial")])
        for overrides in ({"process_blocks": 1}, {"repeats": 4}, {"warmup_pairs": 1},
                          {"memory_diagnostics": True}, {"gpu_uuid": "1"},
                          {"frame_pairs": [[17, 33]]}, {"reuse_streams": False},
                          {"steady_reserved_growth_bytes": 65*MiB}, {"arm_specs": workflow.ARMS[:1]}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                workflow.normalize_config(dict(value, **overrides))
        child = workflow.child_config(value, workflow.ARMS[1], 1)
        self.assertEqual(child["cpu_masks"], {"cuda:0": list(range(60, 68))})
        self.assertEqual(child["device"], "cuda:0")
        self.assertTrue(child["decode_a_on_b_stream"])
        self.assertFalse(child["memory_diagnostics"])

    def test_validate_complete_raw_evidence_and_descriptive_summary(self):
        result = completed_result()
        gate = workflow.validate_result(result)
        self.assertTrue(gate["passed"], gate)
        self.assertGreater(gate["checks"], 300)
        self.assertEqual(result["summary"]["median_block_speedup"], 1.)
        self.assertFalse(result["summary"]["cuda_overlap_proven"])
        self.assertEqual(result["summary"]["technical_repeats_per_arm_per_block"], 3)

    def test_gate_rejects_corrupted_numeric_capacity_lifecycle_and_timing(self):
        changes = [
            lambda x: x["arms"][0]["result"]["rounds"][0]["measurement"]["a_video"].update(sha256="f"*64),
            lambda x: x["arms"][0]["result"]["rounds"][0]["measurement"]["nvml_samples"][0].update(memory_free_bytes=900*MiB),
            lambda x: x["arms"][0]["result"]["rounds"][4]["after_release"].update(reserved_bytes=12*GiB+65*MiB),
            lambda x: x["arms"][0]["result"]["rounds"][0]["after_release"].update(state_count=1),
            lambda x: x["arms"][0]["result"]["rounds"][0].update(stream_ids=[0, 999]),
            lambda x: x["arms"][0]["result"]["rounds"][2]["measurement"].update(memory_diagnostics=True),
            lambda x: x["arms"][0]["result"]["rounds"][2]["measurement"].update(wall_ms=1.),
            lambda x: x["arms"][0]["result"]["rounds"].pop(0),
            lambda x: x["arms"][0]["launch"].update(reaped=False),
            lambda x: x["arms"][0]["launch"].update(returncode=1),
            lambda x: x["arms"][1]["launch"].update(started_unix_ns=10),
            lambda x: x["arms"][0]["result"]["configuration"].update(memory_diagnostics=True),
            lambda x: x["summary"].update(median_block_speedup=3.),
            lambda x: x["qualification"]["scopes"]["numeric"].update(status="unverified"),
            lambda x: x["arms"].reverse(),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                result = completed_result()
                change(result)
                self.assertFalse(workflow.validate_result(result)["passed"])
        for malformed in ({}, {"configuration": None}, {"configuration": config(), "arms": None}):
            self.assertFalse(workflow.validate_result(malformed)["passed"])


class ChildTests(unittest.TestCase):
    def run_child(self, *, values=None, snapshots=None):
        arm = workflow.ARMS[1]
        child = workflow.child_config(config(), arm, 0)
        calls = []
        cuda = types.SimpleNamespace(device_count=lambda: 1, synchronize=lambda *_: calls.append("sync"),
                                     empty_cache=lambda: calls.append("trim"))
        worker = types.SimpleNamespace(torch=types.SimpleNamespace(cuda=cuda), device="cuda:0",
                                       workflow_cpu_affinity=dict(bound_cpus=list(range(60, 68))))
        refs = {(role, 17): dict(directory=role, hashes={
                    output: hashes()[role.lower()+"_"+output] for output in ("latent", "video")})
                for role in ("A", "B")}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=UUID), \
                patch.object(workflow.overlap, "make_worker", return_value=worker), \
                patch.object(workflow.overlap, "_references", return_value=refs), \
                patch.object(workflow.overlap, "Timeline", return_value=types.SimpleNamespace(records=[])), \
                patch.object(workflow, "_boundary", side_effect=snapshots or [boundary() for _ in range(6)]), \
                patch.object(workflow.lifecycle, "_stream_identity", return_value=[0, 1]), \
                patch.object(workflow.overlap, "_run_arm", side_effect=values or [measurement(i, arm) for i in range(5)]) as run:
            result = workflow.run_child(child, temp)
            self.assertEqual(json.loads((Path(temp) / "run.json").read_text()), result)
        return result, calls, run.call_args_list

    def test_two_warmups_three_normal_times_one_startup_trim(self):
        result, calls, arms = self.run_child()
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["performance_evidence"])
        self.assertEqual(calls.count("trim"), 1)
        self.assertEqual([call.args[7] for call in arms], ["performance_warmup"]*2 + ["timed"]*3)
        self.assertTrue(all(call.args[1]["memory_diagnostics"] is False for call in arms))
        self.assertEqual(result["completed_measured_pairs"], 3)

    def test_warmup_capacity_failure_stops_and_never_qualifies_timing(self):
        bad = measurement(0, workflow.ARMS[1])
        bad["memory_margin_gate"].update(eligible=False)
        result, calls, arms = self.run_child(values=[bad])
        self.assertEqual(len(arms), 1)
        self.assertEqual(result["status"], "qualification_failed")
        self.assertEqual(result["stop_reason"], "hot_memory_margin_not_qualified")
        self.assertFalse(result["performance_evidence"])
        self.assertEqual(calls.count("trim"), 1)

    def test_boundary_diagnostics_are_opted_in_only_for_boundary_call(self):
        c = config()
        with patch.object(workflow.lifecycle, "_boundary", return_value=boundary()) as capture:
            workflow._boundary(object(), c, "after_release")
        self.assertTrue(capture.call_args.args[1]["memory_diagnostics"])
        self.assertFalse(c["memory_diagnostics"])

    def test_exception_has_durable_failed_manifest(self):
        result, _, arms = self.run_child(values=[RuntimeError("CUDA out of memory")])
        self.assertEqual(len(arms), 1)
        self.assertEqual(result["status"], "failed")
        self.assertIn("out of memory", result["error"])
        self.assertFalse(result["performance_evidence"])


class ParentTests(unittest.TestCase):
    def test_independent_process_ab_ba_and_immutable_output(self):
        prepared = completed_result()
        calls = []
        def launch(config, directory):
            index = len(calls)
            calls.append((copy.deepcopy(config), directory.name))
            arm = prepared["arms"][index]
            return arm["result"], arm["launch"]
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_launch_child", launch):
            result = workflow.run_performance(config(), temp)
            with self.assertRaises(FileExistsError):
                workflow.run_performance(config(), temp)
        self.assertEqual(result["status"], "completed", result.get("validation"))
        self.assertTrue(workflow.validate_result(result)["passed"])
        self.assertEqual([c["selected_arm"]["id"] for c, _ in calls],
                         ["serial", "streams-shared-decode", "streams-shared-decode", "serial"])

    def test_failed_exit_stops_future_children(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_launch_child", return_value=(
                completed_child(), dict(reaped=True, returncode=1, timed_out=False))) as launch:
            result = workflow.run_performance(config(), temp)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(launch.call_count, 1)
        self.assertFalse(result["performance_evidence"])
        self.assertEqual(result["qualification"]["scopes"]["performance"]["status"], "unverified")

    def test_timeout_terminate_kill_and_reap(self):
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
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow.subprocess, "Popen", return_value=process) as popen:
            child, launch = workflow._launch_child(workflow.child_config(config(), workflow.ARMS[0], 0), Path(temp))
            self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], UUID)
            self.assertTrue((Path(temp) / "child-launch.json").exists())
        self.assertEqual(child, {})
        self.assertTrue(launch["reaped"] and launch["timed_out"])
        self.assertTrue(process.terminated and process.killed)
        self.assertEqual(launch["returncode"], -9)

    def test_actual_cpu_processes_are_fresh_uuid_bound_and_reaped(self):
        original = subprocess.Popen
        code = ("import os,json,sys;from pathlib import Path;"
                "Path(sys.argv[1],'run.json').write_text(json.dumps({'pid':os.getpid(),"
                "'uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch_loaded':'torch' in sys.modules}))")
        def cpu_child(command, **kwargs):
            return original([command[0], "-c", code, command[-1]], **kwargs)
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow.subprocess, "Popen", side_effect=cpu_child):
            children = [workflow._launch_child(workflow.child_config(config(), arm, block),
                        Path(temp) / str(index)) for index, (block, arm) in enumerate(workflow.arm_schedule(config()))]
        self.assertEqual(len({child["pid"] for child, _ in children}), 4)
        self.assertTrue(all(launch["reaped"] and launch["returncode"] == 0 for _, launch in children))
        self.assertTrue(all(child["uuid"] == UUID and child["torch_loaded"] is False for child, _ in children))
        self.assertTrue(all(left[1]["finished_unix_ns"] <= right[1]["started_unix_ns"]
                            for left, right in zip(children, children[1:])))


if __name__ == "__main__":
    unittest.main()
