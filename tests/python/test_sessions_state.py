"""CPU state-machine tests; simulated capacity is never experiment evidence."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from ditserve.research.common import atomic_json, file_digest
from ditserve.sessions import workload
from ditserve.sessions.state import run_state


class OOM(RuntimeError):
    pass


class Memory:
    def __init__(self, limit=100, extra=None, transient=False):
        self.limit, self.extra, self.transient = limit, extra or {}, transient
        self.states, self.moves, self.prepared = [], [], []
        self.corrupt_restore = False

    @property
    def used(self):
        return sum(not s.closed and s.gpu for s in self.states)

    def allocate(self, kind, index, **kwargs):
        if self.used + 1 > self.limit:
            raise OOM("real-state test admission limit")
        state = SimpleNamespace(kind=kind, index=index, closed=False, gpu=True, **kwargs)
        self.states.append(state)
        self.prepared.append((kind, index))
        return state

    def observation(self, _llm):
        return dict(allocated_bytes=self.used * 100, reserved_bytes=self.used * 100,
                    peak_allocated_bytes=self.used * 100, free_bytes=(self.limit - self.used) * 100,
                    total_bytes=self.limit * 100)

    def move(self, state, gpu):
        if gpu and not state.gpu and self.used + 1 > self.limit:
            raise OOM("real-state test restore limit")
        state.gpu = gpu
        self.moves.append((state.kind, state.index, gpu))
        if gpu and self.corrupt_restore:
            state.payload += 1
        return dict(ms=1., bytes=100, location="cuda:0" if gpu else "cpu")


class FakeLLM:
    device = "cuda:0"

    def __init__(self, memory):
        self.memory = memory
        self.wrong_token = None
        self.torch = SimpleNamespace(OutOfMemoryError=OOM,
                                    cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None))

    def prepare(self, messages, *, max_new_tokens, request_id, previous=None):
        if previous is not None:
            self.release(previous)
        index, turn = int(request_id[1:4]), int(request_id[-1])
        return self.memory.allocate("llm", index, turn=turn, done=False, action="prefill",
                                    generated_ids=[], payload=np.array([index, turn], dtype=np.float32))

    def advance(self, state):
        if self.memory.used + self.memory.extra.get(state.index, 0) > self.memory.limit:
            raise OOM("real-state test prefill workspace limit")
        if self.memory.transient and state.index == 1 and self.memory.used >= 3:
            self.memory.transient = False
            raise OOM("transient test allocation failure")
        token = state.index * 10 + state.turn + (100 if self.wrong_token == state.index else 0)
        state.generated_ids, state.done = [token], True
        return dict(ms=1., done=True, action="prefill")

    def finish(self, state):
        return "scene-%s-turn-%s" % (state.index, state.turn)

    def release(self, state):
        state.closed = True

    def snapshot(self, state):
        return dict(cache=state.payload.copy(), generated_ids=list(state.generated_ids), done=state.done)

    def inventory(self, state):
        return dict(bytes=100, tensor_bytes=100, device="cuda:0" if state.gpu else "cpu", tensors=[])

    def offload_to_cpu(self, state):
        return self.memory.move(state, False)

    def restore(self, state):
        return self.memory.move(state, True)


class FakeSana:
    def __init__(self, memory):
        self.memory = memory

    def prepare(self, prompt, *, seed, request_id):
        index = int(request_id[1:4])
        return self.memory.allocate("image", index, seed=seed, step_index=0, done=False,
                                    payload=np.array([seed % 997], dtype=np.float32))

    def advance(self, state):
        state.step_index += 1
        state.done = state.step_index == 2
        state.payload += 1
        return dict(ms=1., step=state.step_index, done=state.done)

    def finish(self, state):
        return state.payload.copy()

    def release(self, state):
        state.closed = True

    def snapshot(self, state):
        return dict(latents=state.payload.copy(), step_index=state.step_index, done=state.done)

    def inventory(self, state):
        return dict(bytes=100, tensor_bytes=100, device="cuda:0" if state.gpu else "cpu", tensors=[])

    def offload_to_cpu(self, state):
        return self.memory.move(state, False)

    def restore(self, state):
        return self.memory.move(state, True)


def fixture(root, count, counts):
    config = dict(session_count=count, session_inputs=[["user-%s-first" % i, "user-%s-second" % i] for i in range(count)],
                  seed=301, loads=[.5], max_new_tokens=64, think_ms=1000, deadline_multiplier=2.5,
                  state_capacity_counts=counts)
    costs = dict(stage_ms={stage: 1. for stage in workload.STAGES})
    trace = workload.make_trace(config, costs, .5, 301)
    rows = []
    arrays = root / "pilot/arrays/reference"
    arrays.mkdir(parents=True)
    for index, session in enumerate(trace["sessions"]):
        for turn in session["turns"]:
            path = arrays / (turn["request_id"] + ".npy")
            np.save(path, np.array([turn["image_seed"] % 997 + 2], dtype=np.float32), allow_pickle=False)
            rows.append(dict(turn, status="completed", llm_token_ids=[index * 10 + turn["round"]],
                llm_text="scene-%s-turn-%s" % (index, turn["round"]), outputs=dict(output=str(path)),
                quality=dict(passed=True, output_sha256=file_digest(path))))
    record = dict(arm="session_serial", load=.5, seed=301, trace_sha256=trace["trace_sha256"],
                  quality=dict(passed=True), device_identity={"gpu_uuid": "test"}, cost_model_sha256="test", rows=rows)
    atomic_json(root / "pilot/result.json", dict(status="passed", records=[record]))
    atomic_json(root / "pilot/supervisor.json", dict(status="passed", returncode=0))
    result = dict(device_identity=record["device_identity"], cost_model_sha256="test", status="running")
    return config, costs, result


class SessionStateTests(unittest.TestCase):
    def run_case(self, root, config, costs, result, memory, *, llm=None):
        llm = llm or FakeLLM(memory)
        snapshots = []

        def save():
            snapshots.append(json.loads(json.dumps(result, allow_nan=False)))

        with mock.patch("ditserve.sessions.execution.memory", side_effect=memory.observation):
            value = run_state(config, root, llm, FakeSana(memory), result, save, lambda: None, costs)
        self.assertTrue(all(state.closed for state in memory.states))
        return value, snapshots

    def test_no_pressure_completes_all_real_two_turn_cohorts_without_offload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, costs, result = fixture(root, 3, [1, 2, 3])
            memory = Memory()
            value, snapshots = self.run_case(root, config, costs, result, memory)
            self.assertEqual(value["status"], "passed")
            self.assertTrue(value["eligible"] and value["technical_complete"])
            self.assertEqual(value["reason"], "no_observed_capacity_pressure")
            self.assertEqual(value["qualified_retained_sessions"], 3)
            self.assertEqual([t["completed_turns"] for t in value["capacity_trials"]], [2, 4, 6])
            self.assertEqual(memory.moves, [])
            self.assertEqual(value["state_policies"], [])
            self.assertTrue(all(index < 3 for kind, index in memory.prepared))
            self.assertGreater(len(snapshots), 1)

    def test_real_boundary_requires_keep_failure_then_compares_independent_offload_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, costs, result = fixture(root, 4, [1, 2, 4])
            memory = Memory(limit=5)
            value, _ = self.run_case(root, config, costs, result, memory)
            self.assertTrue(value["capacity_pressure_observed"])
            self.assertEqual(value["capacity_trials"][-1]["status"], "capacity_boundary")
            arms = value["state_policies"]
            self.assertEqual([a["policy"] for a in arms], ["keep", "offload_kv", "offload_image"])
            self.assertEqual(arms[0]["status"], "capacity_failure")
            self.assertEqual(value["successful_state_policies"], ["offload_kv", "offload_image"])
            for arm in arms[1:]:
                self.assertEqual(arm["completed_turns"], 6)
                self.assertTrue(all(t["comparison"]["exact"] for t in arm["transfers"]))
            self.assertEqual(value["conclusion"], "observed_state_capacity_relief")

    def test_both_components_are_tried_only_when_single_component_policies_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, costs, result = fixture(root, 2, [1, 2])
            value, _ = self.run_case(root, config, costs, result, Memory(limit=6, extra={1: 5}))
            self.assertEqual([a["policy"] for a in value["state_policies"]],
                             ["keep", "offload_kv", "offload_image", "offload_both"])
            self.assertEqual(value["successful_state_policies"], ["offload_both"])
            self.assertEqual([a["status"] for a in value["state_policies"][:3]], ["capacity_failure"] * 3)

    def test_transient_cleanup_recovery_is_not_reported_as_offload_benefit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, costs, result = fixture(root, 3, [1, 2, 3])
            memory = Memory(transient=True)
            value, _ = self.run_case(root, config, costs, result, memory)
            self.assertEqual(value["conclusion"], "observed_transient_capacity_failure")
            self.assertFalse(value["eligible"])
            self.assertEqual(value["unmeasured_capacity_counts"], [3])
            self.assertEqual(memory.moves, [])
            self.assertEqual([a["policy"] for a in value["state_policies"]], ["keep"])

    def test_numerical_failure_is_not_reclassified_as_capacity_and_all_states_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, costs, result = fixture(root, 2, [1, 2])
            memory = Memory()
            llm = FakeLLM(memory)
            llm.wrong_token = 1
            with self.assertRaisesRegex(RuntimeError, "image prompt"):
                self.run_case(root, config, costs, result, memory, llm=llm)
            self.assertFalse(result["capacity_pressure_observed"])
            self.assertEqual(result["capacity_trials"][-1]["status"], "failed")
            self.assertTrue(all(state.closed for state in memory.states))

    def test_corrupted_restore_stops_and_cleans_every_cpu_and_gpu_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, costs, result = fixture(root, 4, [1, 2, 4])
            memory = Memory(limit=5)
            memory.corrupt_restore = True
            with self.assertRaisesRegex(RuntimeError, "restore changed"):
                self.run_case(root, config, costs, result, memory)
            self.assertEqual(result["state_policies"][-1]["status"], "failed")
            self.assertTrue(all(state.closed for state in memory.states))

    def test_changed_reference_fails_before_any_model_state_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, costs, result = fixture(root, 2, [1, 2])
            path = root / "pilot/arrays/reference/s000-r1.npy"
            np.save(path, np.array([0.], dtype=np.float32))
            memory = Memory()
            with self.assertRaisesRegex(ValueError, "baseline output"):
                self.run_case(root, config, costs, result, memory)
            self.assertEqual(memory.states, [])

    def test_boundary_with_no_retained_states_does_not_invent_an_offload_strategy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, costs, result = fixture(root, 2, [1, 2])
            memory = Memory(limit=1)
            value, _ = self.run_case(root, config, costs, result, memory)
            self.assertEqual(value["conclusion"], "observed_no_viable_state_policy")
            self.assertFalse(value["eligible"])
            self.assertTrue(value["technical_complete"])
            self.assertEqual(memory.moves, [])
            self.assertEqual(value["state_policies"], [])

    def test_checkpoint_failure_still_releases_every_held_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, costs, result = fixture(root, 2, [1, 2])
            memory, calls = Memory(), []

            def save():
                calls.append(1)
                if len(calls) == 2:
                    raise OSError("simulated checkpoint failure")

            with mock.patch("ditserve.sessions.execution.memory", side_effect=memory.observation):
                with self.assertRaisesRegex(OSError, "checkpoint failure"):
                    run_state(config, root, FakeLLM(memory), FakeSana(memory), result, save, lambda: None, costs)
            self.assertTrue(memory.states)
            self.assertTrue(all(state.closed for state in memory.states))


if __name__ == "__main__":
    unittest.main()
