"""CPU contracts for opt-in long pairs and separately profiled captures."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_overlap_performance as performance
from ditserve.evaluation import workflow_overlap_trace as trace
import test_workflow_overlap_performance as fixtures


def config(pair=(17, 33)):
    value = dict(fixtures.config(), device="cuda:0", cpu_masks={"cuda:0": list(range(60, 68))},
                 frame_pairs=[list(pair)])
    if pair != (17, 17):
        value["experiment_scope"] = "long_frame_pairs_v1"
    return trace.normalize_config(value)


def references(pair):
    return [dict(role=role, frames=frames, path=role,
                 latent=fixtures.hashes()[role.lower()+"_latent"],
                 video=fixtures.hashes()[role.lower()+"_video"])
            for role, frames in zip(("A", "B"), pair)]


def trace_result(pair=(17, 33)):
    value = config(pair)
    result = dict(schema=trace.SCHEMA, configuration=value, status="completed",
                  performance_evidence=False, cuda_overlap_proven=False,
                  capture_range=trace.capture_range(value),
                  cpu_affinity=dict(bound_cpus=list(range(60, 68))), startup_cache_trims=1,
                  baseline=fixtures.boundary(), stream_ids=[0, 123], references=references(pair),
                  rounds=[], phases=[], qualification=dict(passed=True))
    for index in range(3):
        purpose, repeat = ("trace_warmup", index) if index < 2 else ("trace", 0)
        measured = fixtures.measurement(index, performance.ARMS[1])
        measured.update(purpose=purpose, repeat=repeat, performance_evidence=False)
        result["rounds"].append(dict(index=index, measurement=measured,
                                      after_release=fixtures.boundary(), stream_ids=[0, 123]))
        a, b = pair
        meta = dict(frames=a, mode="streams", peer_frames=b, purpose=purpose, repeat=repeat,
                    request_id=f"P4-{purpose}-a{a}-b{b}-r{repeat}-streams", variant="P4")
        result["phases"].append(dict(name="P4/stage_pair", metadata=meta,
            nvtx_label="ditserve.workflow/P4/stage_pair|" + json.dumps(meta, sort_keys=True, separators=(",", ":")),
            status="completed", errors=[], timing_scope="host_interval",
            start_ns=measured["stage_started_monotonic_ns"]+1,
            end_ns=measured["stage_finished_monotonic_ns"]-1))
    return result


def long_performance_result(pair):
    result = fixtures.completed_result()
    result["configuration"] = performance.normalize_config(dict(result["configuration"],
        experiment_scope="long_frame_pairs_v1", frame_pairs=[list(pair)]))
    for row, (block, arm) in zip(result["arms"], performance.arm_schedule(result["configuration"])):
        child = row["result"]
        child["configuration"] = performance.child_config(result["configuration"], arm, block)
        child["references"] = references(pair)
        row["launch"]["result_sha256"] = hashlib.sha256((json.dumps(child, indent=2, ensure_ascii=False,
            allow_nan=False) + "\n").encode()).hexdigest()
    return result


class LongPairTests(unittest.TestCase):
    def test_only_explicit_registered_long_scope_is_accepted(self):
        for pair in ((17, 33), (33, 17), (33, 33)):
            with self.subTest(pair=pair):
                value = config(pair)
                self.assertEqual(value["frame_pairs"], [list(pair)])
                self.assertTrue(performance.validate_result(long_performance_result(pair))["passed"])
                del value["experiment_scope"]
                with self.assertRaises(ValueError):
                    performance.normalize_config(value)
        for updates in (dict(frame_pairs=[[17, 17]], experiment_scope="long_frame_pairs_v1"),
                        dict(frame_pairs=[[17, 65]]), dict(experiment_scope="unregistered")):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                performance.normalize_config(dict(config(), **updates))

    def test_long_reference_frame_mismatch_fails_even_after_digest_rebinding(self):
        result = long_performance_result((17, 33))
        result["arms"][0]["result"]["references"][1]["frames"] = 17
        child = result["arms"][0]["result"]
        result["arms"][0]["launch"]["result_sha256"] = hashlib.sha256((json.dumps(child,
            indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode()).hexdigest()
        self.assertFalse(performance.validate_result(result)["passed"])

    def test_actual_child_passes_long_pair_to_runtime(self):
        value = performance.child_config(config((33, 17)), performance.ARMS[1], 0)
        worker = types.SimpleNamespace(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(
            device_count=lambda: 1, synchronize=lambda *_: None, empty_cache=lambda: None)),
            device="cuda:0", workflow_cpu_affinity=dict(bound_cpus=list(range(60, 68))))
        refs = {(row["role"], row["frames"]): dict(directory=row["path"],
                hashes={key: row[key] for key in ("latent", "video")}) for row in references((33, 17))}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=fixtures.UUID), \
                patch.object(performance.overlap, "make_worker", return_value=worker), \
                patch.object(performance.overlap, "_references", return_value=refs), \
                patch.object(performance.overlap, "Timeline", return_value=types.SimpleNamespace(records=[])), \
                patch.object(performance, "_boundary", side_effect=[fixtures.boundary() for _ in range(6)]), \
                patch.object(performance.lifecycle, "_stream_identity", return_value=[0, 123]), \
                patch.object(performance.overlap, "_run_arm", side_effect=[
                    fixtures.measurement(i, performance.ARMS[1]) for i in range(5)]) as run:
            result = performance.run_child(value, temp)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([call.args[3:5] for call in run.call_args_list], [(33, 17)]*5)


class TraceTests(unittest.TestCase):
    def test_four_registered_pairs_only_qualify_capture_evidence(self):
        for pair in ((17, 17), (17, 33), (33, 17), (33, 33)):
            with self.subTest(pair=pair):
                result = trace_result(pair)
                self.assertTrue(trace.validate_result(result)["passed"], trace.validate_result(result))
                self.assertFalse(result["performance_evidence"])
                self.assertFalse(result["cuda_overlap_proven"])

    def test_rejects_missing_duplicated_or_misdirected_nvtx_capture(self):
        changes = [lambda x: x.pop("phases"),
                   lambda x: x["phases"].append(copy.deepcopy(x["phases"][-1])),
                   lambda x: x["phases"][-1].update(nvtx_label="different"),
                   lambda x: x["phases"][-1]["metadata"].update(peer_frames=17),
                   lambda x: x["phases"][-1].update(errors=["NVTX: unavailable"]),
                   lambda x: x["phases"][-1].update(status="failed"),
                   lambda x: x["phases"][-1].update(start_ns=1),
                   lambda x: x.update(cuda_overlap_proven=True),
                   lambda x: x.update(performance_evidence=True)]
        self.assert_corruptions_fail(changes)

    def assert_corruptions_fail(self, changes):
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                result = trace_result()
                change(result)
                self.assertFalse(trace.validate_result(result)["passed"])

    def test_rejects_degenerate_time_reordered_rounds_and_cadence_tampering(self):
        self.assert_corruptions_fail([
            lambda x: x["rounds"][2]["measurement"].update(finished_monotonic_ns=1),
            lambda x: x["rounds"][2]["measurement"].update(started_monotonic_ns=1),
            lambda x: x["rounds"][2]["measurement"].update(stage_finished_monotonic_ns=300001000),
            lambda x: x["rounds"][2]["measurement"].update(wall_ms=float("nan")),
            lambda x: x["rounds"][2]["measurement"].update(stage_pair_wall_ms=1),
            lambda x: x["rounds"][2]["measurement"].update(sample_interval_seconds=1),
            lambda x: x["rounds"][2]["measurement"]["nvml_samples"].reverse(),
            lambda x: x["rounds"].reverse(),
        ])

    def test_rejects_output_capacity_release_and_scope_corruption(self):
        self.assert_corruptions_fail([
            lambda x: x["rounds"][2]["measurement"]["a_video"].update(sha256="f"*64),
            lambda x: x["rounds"][2]["measurement"]["nvml_samples"][0].update(memory_free_bytes=1),
            lambda x: x["rounds"][2]["after_release"].update(state_count=1),
            lambda x: x["rounds"][2].update(stream_ids=[0, 124]),
            lambda x: x["rounds"][2]["measurement"].update(memory_boundaries=[{}]),
            lambda x: x["rounds"][2]["measurement"].update(performance_evidence=True),
            lambda x: x["references"][1].update(frames=17),
        ])
        for malformed in ({}, {"configuration": None}, {"configuration": config(), "rounds": None}):
            self.assertFalse(trace.validate_result(malformed)["passed"])

    def test_capture_executes_two_warmups_and_one_unique_trace_with_durable_receipt(self):
        prepared = trace_result()
        worker = types.SimpleNamespace(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(
            device_count=lambda: 1, synchronize=lambda *_: None, empty_cache=lambda: None)),
            device="cuda:0", workflow_cpu_affinity=prepared["cpu_affinity"])
        refs = {(row["role"], row["frames"]): dict(directory=row["path"],
                hashes={key: row[key] for key in ("latent", "video")}) for row in prepared["references"]}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=fixtures.UUID), \
                patch.object(trace.overlap, "make_worker", return_value=worker), \
                patch.object(trace.overlap, "_references", return_value=refs), \
                patch.object(trace.overlap, "Timeline", return_value=types.SimpleNamespace(records=prepared["phases"])), \
                patch.object(trace.performance, "_boundary", side_effect=[fixtures.boundary() for _ in range(4)]), \
                patch.object(trace.lifecycle, "_stream_identity", return_value=[0, 123]), \
                patch.object(trace.overlap, "_run_arm", side_effect=[row["measurement"] for row in prepared["rounds"]]) as run:
            result = trace.run_capture(prepared["configuration"], temp)
            self.assertEqual(json.loads((Path(temp)/"run.json").read_text()), result)
            with self.assertRaises(FileExistsError):
                trace.run_capture(prepared["configuration"], temp)
        self.assertEqual(result["status"], "completed", result.get("audit"))
        self.assertEqual([call.args[3:5] for call in run.call_args_list], [(17, 33)]*3)
        self.assertEqual([call.args[7] for call in run.call_args_list], ["trace_warmup", "trace_warmup", "trace"])


if __name__ == "__main__":
    unittest.main()
