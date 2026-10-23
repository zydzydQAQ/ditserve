"""CPU doubles verify timing boundaries, stream ownership and gating."""

import contextlib
import dataclasses
import gc
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from ditserve.evaluation import workflow_serving_overlap as workflow


class Tensor:
    def __init__(self, value):
        self.value = value
        self.streams = []

    def record_stream(self, stream):
        self.streams.append(stream)

    def detach(self):
        return self

    float = cpu = detach

    def numpy(self):
        return self.value


class Stream:
    def __init__(self, **_):
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)


class Event:
    def __init__(self, **_):
        self.waited = False

    def record(self, stream=None):
        self.stream = stream

    def synchronize(self):
        self.waited = True

    def elapsed_time(self, other):
        if not other.waited:
            raise AssertionError("reading an incomplete event")
        return 1.0


class Timeline:
    def __init__(self):
        self.hot = False
        self.records = []

    @contextlib.contextmanager
    def phase(self, name, **metadata):
        self.records.append((name, metadata))
        previous = self.hot
        if name == "P4/stage_pair":
            self.hot = True
        try:
            yield
        finally:
            self.hot = previous


class Sampler:
    errors = []

    def __init__(self):
        self.stopped = False
        self.samples = [dict(timestamp_ns=index * 1000000, memory_total_bytes=24 * 1024**3,
                             memory_free_bytes=2 * 1024**3) for index in range(200)]

    def stop(self):
        self.stopped = True

    def summary(self, *_):
        return {"errors": []}


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.events, self.calls, self.tensors, self.states = [], [], [], []
        self.timeline = Timeline()
        self.stream = Stream()

        def event(**kwargs):
            value = Event(**kwargs)
            self.events.append(value)
            return value

        def sync(device):
            self.assertFalse(self.timeline.hot, "global synchronization inside the overlap window")
            self.calls.append("sync")

        cuda = types.SimpleNamespace(Stream=Stream, Event=event, synchronize=sync,
                                     reset_peak_memory_stats=lambda *_: None,
                                     current_stream=lambda *_: self.stream,
                                     stream=lambda *_: contextlib.nullcontext(),
                                     max_memory_allocated=lambda *_: 100,
                                     max_memory_reserved=lambda *_: 200,
                                     memory_allocated=lambda *_: 100,
                                     memory_reserved=lambda *_: 200,
                                     memory_stats=lambda *_: {"inactive_split_bytes.all.current": 50})
        self.worker = types.SimpleNamespace(torch=types.SimpleNamespace(cuda=cuda), device="cuda:1")

        def finish(state):
            self.assertFalse(self.timeline.hot)
            self.assertEqual(state.steps, 20)
            self.calls.append("finish-A")
            return "A-video"

        def release(state):
            self.assertFalse(self.timeline.hot)
            self.calls.append("release-" + state.role)
            state.latents = state.prompt_embeds = state.negative_prompt_embeds = None

        self.worker.finish, self.worker.release = finish, release
        self.sampler = Sampler()
        self.config = workflow.normalize_config({"frame_pairs": [[17, 33]]})
        self.references = {(role, frames): dict(
            spec=workflow.request_spec(self.config, frames, role, role), hosts=(role,),
            hashes={key: {"sha256": role + "-" + key} for key in ("latent", "video")})
            for role, frames in (("A", 17), ("B", 33))}

    def install(self, worker, spec, hosts):
        self.assertFalse(self.timeline.hot)
        role = hosts[0]
        self.calls.append("prepare-" + role)
        state = types.SimpleNamespace(latents=Tensor(role + "-latent"), prompt_embeds=Tensor("prompt"),
                                      negative_prompt_embeds=Tensor("negative"), role=role, steps=0)
        self.states.append(weakref.ref(state)) if hasattr(state, "__weakref__") else None
        self.tensors.append(weakref.ref(state.latents))
        return state, Event()

    def step(self, worker, state):
        self.assertEqual(self.timeline.hot, state.role == "A")
        state.steps += 1
        self.calls.append("step-" + state.role)

    def decode(self, worker, state):
        self.assertTrue(self.timeline.hot)
        self.assertEqual((state.role, state.steps), ("B", 20))
        tensor = Tensor("B-video")
        self.tensors.append(weakref.ref(tensor))
        self.calls.append("decode-B")
        return tensor

    def hash(self, value):
        self.assertFalse(self.timeline.hot)
        self.calls.append("hash-" + value)
        return dict(sha256=value)

    def run_arm(self, mode="streams", decode=None):
        def clock():
            self.calls.append("clock")
            return len(self.calls) * 1000000
        with patch.object(workflow, "_install_inputs", self.install), \
                patch.object(workflow, "_step_async", self.step), \
                patch.object(workflow, "_decode_gpu", decode or self.decode), \
                patch.object(workflow, "_hash_array", self.hash), \
                patch.object(workflow, "_start_sampler", return_value=self.sampler), \
                patch.object(workflow.time, "perf_counter_ns", clock):
            return workflow._run_arm(self.worker, self.config, self.references, 17, 33,
                                     mode, 0, "timed", self.timeline)

    def test_full_timer_covers_both_requests_and_checks_without_global_sync_in_stage(self):
        result = self.run_arm()
        self.assertTrue(result["correctness"]["passed"])
        self.assertTrue(result["performance_evidence"])
        self.assertTrue(self.sampler.stopped)
        self.assertEqual(self.calls.count("step-A"), 20)
        self.assertEqual(self.calls.count("step-B"), 20)
        self.assertLess(self.calls.index("clock"), self.calls.index("prepare-A"))
        self.assertLess(self.calls.index("prepare-B"), self.calls.index("decode-B"))
        self.assertLess(self.calls.index("hash-A-video"), len(self.calls) - self.calls[::-1].index("clock"))
        self.assertGreater(result["wall_ms"], result["stage_pair_wall_ms"])
        self.assertEqual(self.calls[-2:], ["release-A", "release-B"])
        gc.collect()
        self.assertTrue(all(reference() is None for reference in self.tensors))

    def test_exception_drains_and_releases_both_states_and_output(self):
        def failure(*args):
            raise RuntimeError("decode failed")
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            self.run_arm(decode=failure)
        self.assertEqual(self.calls[-3:], ["sync", "release-A", "release-B"])
        self.assertTrue(self.sampler.stopped)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in self.tensors))

    def test_b_latent_is_part_of_exact_gate(self):
        result = self.run_arm("serial")
        result["b_latent"]["sha256"] = "different"
        self.assertFalse(workflow._exact_result(result, self.references[("A", 17)]["hashes"],
                                               self.references[("B", 33)]["hashes"])["passed"])

    def test_reused_streams_keep_serial_on_warmed_stream(self):
        self.config["reuse_streams"] = True
        serial = workflow._arm_streams(self.worker, self.config, "serial")
        concurrent = workflow._arm_streams(self.worker, self.config, "streams")
        self.assertIs(serial[0], self.stream)
        self.assertIs(serial[0], serial[1])
        self.assertIs(concurrent[0], serial[0])
        self.assertIsNot(concurrent[1], concurrent[0])
        self.assertIs(concurrent[1], workflow._arm_streams(self.worker, self.config, "streams")[1])

    def test_early_b_release_occurs_after_hash_before_a_decode_and_only_once(self):
        self.config["release_b_before_a_decode"] = True
        result = self.run_arm()
        self.assertTrue(result["correctness"]["passed"])
        self.assertLess(self.calls.index("hash-B-video"), self.calls.index("release-B"))
        self.assertLess(self.calls.index("release-B"), self.calls.index("finish-A"))
        self.assertEqual(self.calls.count("release-B"), 1)

    def test_allocator_diagnostic_timings_do_not_become_performance_evidence(self):
        self.config["memory_diagnostics"] = True
        result = self.run_arm()
        self.assertFalse(result["performance_evidence"])
        self.assertEqual(result["memory_boundaries"][0]["label"], "before_prepare")
        self.assertEqual(result["memory_boundaries"][-1]["label"], "after_a_decode")
        self.assertEqual(result["memory_boundaries"][0]["inactive_split_bytes"], 50)

    def test_shared_decode_uses_existing_b_stream_after_stage_and_b_release(self):
        self.config.update(reuse_streams=True, release_b_before_a_decode=True, decode_a_on_b_stream=True)
        original_finish = self.worker.finish
        active = []
        @contextlib.contextmanager
        def stream_context(stream):
            active.append(stream)
            try:
                yield
            finally:
                active.pop()
        def finish(state):
            b_stream = self.worker._workflow_overlap_streams[1]
            self.assertIs(active[-1], b_stream)
            self.assertIn(b_stream, state.latents.streams)
            self.assertIn("release-B", self.calls)
            self.assertFalse(self.timeline.hot)
            return original_finish(state)
        self.worker.torch.cuda.stream = stream_context
        self.worker.finish = finish
        result = self.run_arm()
        self.assertTrue(result["correctness"]["passed"])
        self.assertEqual(result["a_decode_stream"], "b_stream")
        self.assertEqual(self.calls.count("finish-A"), 1)

    def test_default_decode_stream_policy_remains_unchanged(self):
        self.assertFalse(self.config["decode_a_on_b_stream"])
        self.assertEqual(self.run_arm()["a_decode_stream"], "current_stream")
        with self.assertRaises(ValueError):
            workflow.normalize_config({"decode_a_on_b_stream": 1})


class QualificationTests(unittest.TestCase):
    def test_default_reference_omits_video_array_but_preserves_exact_hash_and_latents(self):
        @dataclasses.dataclass
        class Payload:
            consumed: bool = False
        for retain in (False, True):
            config = workflow.normalize_config(dict(frame_pairs=[[17, 17]], save_videos=retain))
            states, released = [], []
            def prepare(spec, payload):
                state = types.SimpleNamespace(latents=Tensor(np.array([1.], dtype=np.float32)))
                states.append(state)
                return state
            video = np.array([2.], dtype=np.float32)
            worker = types.SimpleNamespace(encode_prompt=lambda spec: Payload(), prepare_encoded=prepare,
                step=lambda state: None, finish=lambda state: video, release=released.append)
            with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_host_inputs", return_value={}):
                result = workflow._references(worker, config, Path(temp), Timeline())
                self.assertEqual(len(result), 2)
                self.assertEqual(len(list(Path(temp).rglob("latent.npy"))), 2)
                self.assertEqual(len(list(Path(temp).rglob("video.npy"))), 2 if retain else 0)
                for reference in result.values():
                    self.assertEqual(reference["hashes"]["video"], workflow._hash_array(video))
                    self.assertEqual(json.loads((Path(reference["directory"]) / "hashes.json").read_text()), reference["hashes"])
                first, second = result[("A", 17)]["hashes"], result[("B", 17)]["hashes"]
                actual = dict(a_latent=first["latent"], a_video=first["video"],
                              b_latent=second["latent"], b_video=workflow._hash_array(video + .001))
                self.assertFalse(workflow._exact_result(actual, first, second)["passed"])
            self.assertEqual(released, states)
        self.assertFalse(workflow.normalize_config({})["save_videos"])
        with self.assertRaises(ValueError):
            workflow.normalize_config(dict(save_videos=1))

    def test_failed_serial_bridge_never_executes_stream_or_timed_arm(self):
        config = workflow.normalize_config({"frame_pairs": [[17, 17]]})
        result = {"pairs": []}
        calls = []
        def arm(*args):
            calls.append((args[5], args[7]))
            return {"correctness": {"passed": False}, "memory_margin_gate": {"eligible": True}}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(workflow, "Timeline", lambda *_: Timeline()), \
                patch.object(workflow, "_references", return_value={}), \
                patch.object(workflow, "_run_arm", arm):
            workflow._measure(None, config, Path(temp), result)
        self.assertEqual(calls, [("serial", "qualification")])
        self.assertEqual(result["status"], "qualification_failed")
        self.assertFalse(result["pairs"])

    def test_every_pair_qualifies_before_any_timed_repeat(self):
        config = workflow.normalize_config({"repeats": 3})
        result, calls = {"pairs": []}, []
        def arm(*args):
            calls.append((args[3], args[4], args[5], args[7]))
            return {"correctness": {"passed": True}, "memory_margin_gate": {"eligible": True},
                    "wall_ms": 10, "stage_pair_wall_ms": 5}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(workflow, "Timeline", lambda *_: Timeline()), \
                patch.object(workflow, "_references", return_value={}), \
                patch.object(workflow, "_run_arm", arm):
            workflow._measure(None, config, Path(temp), result)
        self.assertEqual([row[3] for row in calls[:8]], ["qualification"] * 8)
        self.assertEqual([row[3] for row in calls[8:]], ["timed"] * 24)
        self.assertEqual(len(result["pairs"]), 12)
        self.assertEqual(result["pairs"][1]["order"], ["streams", "serial"])

    def test_operational_error_is_durable_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "make_worker", side_effect=RuntimeError("load")):
            result = workflow.run_overlap({}, temp)
            stored = json.loads((Path(temp) / "run.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertGreater(stored["finished_unix_ns"], stored["started_unix_ns"])
            with self.assertRaises(FileExistsError):
                workflow.run_overlap({}, temp)

    def test_missing_and_low_memory_samples_never_qualify(self):
        good = [dict(timestamp_ns=index, memory_total_bytes=24 * 1024**3,
                     memory_free_bytes=2 * 1024**3) for index in (1, 2)]
        self.assertTrue(workflow.memory_margin_gate(good, [], 0, 3)["eligible"])
        self.assertEqual(workflow.memory_margin_gate([], [], 0, 3)["status"], "unverified")
        self.assertFalse(workflow.memory_margin_gate(good, ["NVML failed"], 0, 3)["eligible"])
        good[1]["memory_free_bytes"] = 1024**3
        self.assertEqual(workflow.memory_margin_gate(good, [], 0, 3)["status"], "failed")

    def test_unverified_qualification_capacity_prevents_timing(self):
        config, result, calls = workflow.normalize_config({"frame_pairs": [[17, 17]]}), {"pairs": []}, []
        def arm(*args):
            calls.append(args[7])
            return {"correctness": {"passed": True}, "memory_margin_gate": {"eligible": False}}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(workflow, "Timeline", lambda *_: Timeline()), \
                patch.object(workflow, "_references", return_value={}), \
                patch.object(workflow, "_run_arm", arm):
            workflow._measure(None, config, Path(temp), result)
        self.assertEqual(calls, ["qualification"])
        self.assertEqual(result["status"], "capacity_failed")
        self.assertFalse(result["performance_evidence"])


if __name__ == "__main__":
    unittest.main()
