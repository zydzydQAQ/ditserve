import copy
from collections import Counter
from concurrent.futures import Future
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ditserve.scheduling.execution import _arrays_equal, estimate, remaining_view
from ditserve.scheduling import execution


class SchedulingExecutionTests(unittest.TestCase):
    def setUp(self):
        self.costs = dict(steps=4, num_blocks=3, block_position_ms=[1., 2., 3., 10., 20., 30.],
                          prepare_ms=5., decode_ms=7., image_ms=4.)
        self.video = dict(request_id="v", kind="video", arrival_ms=0., deadline_ms=300., status="pending")

    @staticmethod
    def pair(step, branch="cond", next_block=0):
        return (SimpleNamespace(step_index=step), SimpleNamespace(branch=branch, next_block=next_block))

    def test_unprepared_video_includes_all_remaining_compute_and_decode(self):
        row = remaining_view(self.video, None, self.costs, 10.)
        self.assertEqual(row["action"], "prepare")
        self.assertEqual(row["remaining_ms"], 276.)
        self.assertEqual(set(row["action_ms_by_quantum"].values()), {5.})
        self.assertFalse(row["started"])

    def test_half_step_conditional_suffix_accounts_for_both_branches_once(self):
        row = remaining_view(self.video, self.pair(1, "cond", 2), self.costs, 10.)
        self.assertEqual(row["action"], "denoise")
        self.assertEqual(row["remaining_ms"], 202.)
        self.assertEqual(row["action_ms_by_quantum"], {"1": 3., "4": 63., "8": 63., "step": 63.})

    def test_unconditional_suffix_excludes_completed_conditional_branch(self):
        row = remaining_view(self.video, self.pair(1, "uncond", 1), self.costs, 10.)
        self.assertEqual(row["remaining_ms"], 189.)
        self.assertEqual(row["action_ms_by_quantum"], {"1": 20., "4": 50., "8": 50., "step": 50.})

    def test_quantum_is_capped_at_remaining_step_and_next_step_resets_position(self):
        tail = remaining_view(self.video, self.pair(3, "uncond", 2), self.costs, 10.)
        self.assertEqual(tail["remaining_ms"], 37.)
        self.assertEqual(set(tail["action_ms_by_quantum"].values()), {30.})
        boundary = remaining_view(self.video, self.pair(2), self.costs, 10.)
        self.assertEqual(boundary["remaining_ms"], 139.)
        self.assertEqual(boundary["action_ms_by_quantum"], {"1": 1., "4": 16., "8": 66., "step": 66.})

    def test_completed_denoising_requires_one_atomic_decode(self):
        row = remaining_view(self.video, self.pair(4), self.costs, 10.)
        self.assertEqual(row["action"], "decode")
        self.assertEqual(row["remaining_ms"], 7.)
        self.assertEqual(set(row["action_ms_by_quantum"].values()), {7.})

    def test_image_is_atomic_and_view_does_not_mutate_input_state(self):
        image = dict(self.video, request_id="i", kind="image")
        original = copy.deepcopy(image), copy.deepcopy(self.costs)
        row = remaining_view(image, None, self.costs, 10.)
        self.assertEqual(row["action"], "image")
        self.assertEqual(row["remaining_ms"], 4.)
        self.assertEqual(set(row["action_ms_by_quantum"].values()), {4.})
        self.assertEqual((image, self.costs), original)
        self.assertTrue(row["arrived"] and row["gpu_ready"])

    def test_cost_estimate_retains_mean_margin_and_observed_tail(self):
        self.assertAlmostEqual(estimate([10., 10., 10.]), 11.)
        self.assertAlmostEqual(estimate([1., 2., 3., 4., 100.]), 80.8)
        with self.assertRaises(ValueError):
            estimate([])

    def test_cost_estimate_rejects_nonfinite_or_nonpositive_observations(self):
        for samples in ([math.nan], [math.inf], [1., -1.], [0.], [True]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                estimate(samples)

    def test_output_comparison_separates_tolerated_roundoff_from_exact_equality(self):
        reference = np.array([0., 1., -2.], dtype=np.float64)
        exact = _arrays_equal(reference.copy(), reference)
        self.assertTrue(exact["passed"] and exact["exact"])
        actual = reference + np.array([5e-7, 5e-6, 5e-6])
        tolerant = _arrays_equal(actual, reference)
        self.assertTrue(tolerant["passed"])
        self.assertFalse(tolerant["exact"])
        self.assertAlmostEqual(tolerant["max_abs"], 5e-6)
        outside = _arrays_equal(reference + 1e-3, reference)
        self.assertFalse(outside["passed"])

    def test_nonfinite_or_shape_mismatch_never_passes_even_when_identical(self):
        for value in (math.nan, math.inf, -math.inf):
            array = np.array([value])
            result = _arrays_equal(array, array.copy())
            self.assertFalse(result["passed"] or result["exact"])
            self.assertEqual(result["reason"], "nonfinite")
            self.assertIsNone(result["max_abs"])
        result = _arrays_equal(np.zeros((2, 2)), np.zeros(4))
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "shape_mismatch")

    def test_chunked_comparison_checks_last_chunk_and_preserves_inputs(self):
        reference = np.zeros((1 << 20) + 1, dtype=np.float32)
        actual = reference.copy()
        actual[-1] = 0.25
        result = _arrays_equal(actual, reference)
        self.assertFalse(result["passed"])
        self.assertEqual(result["max_abs"], 0.25)
        self.assertEqual(actual[-1], 0.25)
        self.assertEqual(reference[-1], 0.)


class SchedulingReplayIntegrationTests(unittest.TestCase):
    """Exercise the real replay loop and policy with deterministic CPU models."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name) / "trace"
        self.clock_seconds = 0.
        self.actions, self.snapshots = [], []
        self.arrivals = None
        self.fail_advance = False
        self.check_count = 0
        self.config = dict(steps=1, shape=dict(height=16, width=16, frames=1))
        self.costs = dict(steps=1, num_blocks=2, block_position_ms=[1., 2., 3., 4.],
                          prepare_ms=1., decode_ms=2., image_ms=4., image_deadline_ms=100.)
        self.requests = [dict(request_id=rid, kind=kind, arrival_ms=0., deadline_ms=0.,
                              service_ms=1000., seed=301, prompt="prompt-" + rid,
                              status="pending", start_ms=None, finish_ms=None)
                         for rid, kind in (("v", "video"), ("z", "image"))]
        suite = self

        class Tensor:
            def __init__(self):
                self.value = np.zeros((1, 2), dtype=np.float32)
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.value

        class Worker:
            def encode_prompt(self, spec):
                suite.actions.append(("encode", spec["request_id"]))
                return SimpleNamespace(prompt=spec["prompt"])
            def prepare_encoded(self, spec, payload):
                suite.assertEqual(payload.prompt, spec["prompt"])
                suite.actions.append(("prepare", spec["request_id"]))
                suite.clock_seconds += .001
                return SimpleNamespace(request_id=spec["request_id"], step_index=0,
                                       residency="gpu", latents=Tensor())
            def finish(self, state):
                suite.assertEqual(state.step_index, 1)
                suite.actions.append(("decode", state.request_id))
                suite.clock_seconds += .002
                return np.full((1, 2, 2, 3), 0.5, dtype=np.float32)
            def release(self, state):
                suite.actions.append(("release", state.request_id))
                state.residency = "released"

        class Executor:
            def attach(self, state):
                suite.actions.append(("attach", state.request_id))
                return SimpleNamespace(base=state, branch="cond", next_block=0,
                                       closed=False, events=[])
            def advance(self, continuation, quantum, check_budget):
                check_budget()
                suite.actions.append(("advance", continuation.base.request_id,
                                      continuation.branch, continuation.next_block, quantum))
                if suite.fail_advance:
                    raise RuntimeError("injected denoising failure")
                for unused in range(quantum):
                    if continuation.base.step_index == 1:
                        break
                    suite.clock_seconds += .001
                    continuation.next_block += 1
                    continuation.events.append({"measured": True})
                    if continuation.next_block == 2:
                        continuation.next_block = 0
                        if continuation.branch == "cond":
                            continuation.branch = "uncond"
                        else:
                            continuation.base.step_index += 1
                            continuation.branch = "cond"
                continuation.base.latents.value += quantum
            def native_step(self, continuation, check_budget):
                self.advance(continuation, 4, check_budget)
            def close(self, continuation):
                suite.actions.append(("close", continuation.base.request_id))
                continuation.closed = True

        class Sana:
            def generate(self, prompt, *, seed, check_budget):
                check_budget()
                suite.actions.append(("image", prompt, seed))
                suite.clock_seconds += .004
                return np.ones((1, 2, 2, 3), dtype=np.float32), dict(
                    ms=4., encode_ms=1., encode_end_offset_ms=1., prompt_preencoded=False)

        class Arrivals:
            def __init__(self, jobs, encoder, origin):
                suite.arrivals = self
                self.closed, self.records = False, {}
                for rid, arrival, spec in jobs:
                    suite.assertEqual(arrival, 0.)
                    future = Future()
                    future.set_result(encoder(spec))
                    self.records[rid] = dict(submit_ms=0., start_ms=0., end_ms=0., future=future)
            def snapshot(self):
                return dict(self.records)
            def close(self):
                self.closed = True
                suite.actions.append(("arrivals_close",))

        self.worker, self.executor, self.sana = Worker(), Executor(), Sana()
        self.arrival_type = Arrivals

    def tick(self):
        result = self.clock_seconds
        self.clock_seconds += .0001
        return result

    def check(self):
        self.check_count += 1
        if self.check_count > 100:
            raise AssertionError("replay did not drain within its finite CPU fixture")

    def run_trace(self, arm):
        with patch.object(execution, "EncodedArrivals", self.arrival_type), \
             patch.object(execution.time, "perf_counter", side_effect=self.tick), \
             patch.object(execution.time, "sleep", side_effect=lambda seconds: self.tick()):
            return execution.run_trace(self.config, self.worker, self.executor, self.sana,
                self.requests, arm, self.costs, self.directory, self.check,
                lambda rows, elapsed: self.snapshots.append((copy.deepcopy(rows), elapsed)))

    def test_real_replay_prepares_both_half_steps_decodes_and_persists_outputs(self):
        original = copy.deepcopy(self.requests)
        result = self.run_trace("edf_b1")
        ordered = [action for action in self.actions if action[0] in ("prepare", "advance", "decode")]
        self.assertEqual(ordered, [("prepare", "v"), ("advance", "v", "cond", 0, 1),
            ("advance", "v", "cond", 1, 1), ("advance", "v", "uncond", 0, 1),
            ("advance", "v", "uncond", 1, 1), ("decode", "v")])
        rows = {row["request_id"]: row for row in result["rows"]}
        self.assertEqual(rows["v"]["action_counts"], {"prepare": 1, "denoise": 4, "decode": 1})
        self.assertEqual(rows["z"]["action_counts"], {"image": 1})
        self.assertEqual(result["policy_triggers"], {"edf": 7})
        self.assertEqual(result["switches"], 1)
        self.assertEqual(set(rows["v"]["outputs"]), {"output", "latent"})
        self.assertEqual(set(rows["z"]["outputs"]), {"output"})
        for row in rows.values():
            for path in row["outputs"].values():
                self.assertTrue(Path(path).is_absolute() and Path(path).is_file())
                self.assertTrue(np.isfinite(np.load(path, allow_pickle=False)).all())
        self.assertFalse(rows["z"]["service"]["prompt_preencoded"])
        self.assertEqual(rows["z"]["encoding_work_ms"], 1.)
        self.assertGreater(rows["z"]["blocked_by_ms"]["video:denoise"], 0.)
        self.assertEqual(self.requests, original)
        self.assertTrue(self.arrivals.closed)
        self.assertEqual(self.actions.count(("release", "v")), 1)
        self.assertEqual(self.actions.count(("close", "v")), 1)
        self.assertFalse(result["quality"]["passed"])
        self.assertEqual(result["quality"]["status"], "pending_same_trace_reference")

    def test_all_predicted_late_guarded_requests_still_drain_without_claiming_on_time_success(self):
        result = self.run_trace("guarded_b4")
        self.assertTrue(all(row["status"] == "completed" for row in result["rows"]))
        self.assertEqual(result["metrics"]["completed"], 2)
        self.assertEqual(result["metrics"]["on_time"], 0)
        self.assertEqual(result["metrics"]["goodput_rps"], 0.)
        self.assertEqual(result["policy_triggers"], {"edf_all_predicted_late": 4})
        self.assertTrue(all(row["finish_ms"] > row["deadline_ms"] for row in result["rows"]))
        self.assertTrue(self.arrivals.closed)

    def test_denoising_exception_marks_interrupted_and_releases_owned_state_and_arrivals(self):
        self.fail_advance = True
        with self.assertRaisesRegex(RuntimeError, "injected denoising failure"):
            self.run_trace("edf_b1")
        self.assertTrue(self.arrivals.closed)
        self.assertEqual(self.actions.count(("release", "v")), 1)
        self.assertEqual(self.actions.count(("close", "v")), 1)
        self.assertFalse(any(action[0] == "decode" for action in self.actions))
        self.assertEqual({row["status"] for row in self.snapshots[-1][0]}, {"interrupted"})
        self.assertEqual(list(self.directory.glob("*.npy")), [])

    def test_event_file_matches_action_inventory_and_partial_rows_have_no_history(self):
        result = self.run_trace("edf_b1")
        manifest = result["execution_events"]
        path = Path(manifest["path"])
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.parent, self.directory)
        payload = path.read_bytes()
        events = [json.loads(line) for line in payload.splitlines()]
        self.assertEqual(manifest["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(manifest["count"], len(events))
        self.assertEqual(len(events), 7)
        observed = Counter((event["request_id"], event["action"]) for event in events)
        expected = Counter({(row["request_id"], action): count
                            for row in result["rows"] for action, count in row["action_counts"].items()})
        self.assertEqual(observed, expected)
        self.assertEqual(sum(expected.values()), manifest["count"])
        for previous, current in zip(events, events[1:]):
            self.assertGreaterEqual(current["dispatch_ms"], previous["end_ms"])
        self.assertTrue(all(event["dispatch_ms"] <= event["end_ms"] <= result["elapsed_ms"] for event in events))
        self.assertGreaterEqual(result["checkpoint_ms"], 0.)
        self.assertTrue(self.snapshots)
        for rows, elapsed in [*self.snapshots, (result["rows"], result["elapsed_ms"])]:
            # The periodic phase checkpoint must never rewrite the growing
            # action-event history, including histories of completed requests.
            self.assertNotIn("action_spans", json.dumps(rows))
            self.assertNotIn("dispatch_ms", json.dumps(rows))
            self.assertEqual(len(rows), len(self.requests))

    def fail_event_writer(self, *, close_also_fails=False):
        """Patch only the event stream, leaving NumPy artifacts untouched."""
        original_open = Path.open
        handles = []
        class FailingEventFile:
            def __init__(self, file):
                self.file, self.close_attempted = file, False
            def write(self, value):
                raise OSError("injected event write failure")
            def flush(self):
                return self.file.flush()
            def close(self):
                self.close_attempted = True
                self.file.close()
                if close_also_fails:
                    raise OSError("injected event close failure")
        def open_file(path, *args, **kwargs):
            actual = original_open(path, *args, **kwargs)
            mode = args[0] if args else kwargs.get("mode", "r")
            if path.name == "execution-events.jsonl" and mode == "w":
                wrapped = FailingEventFile(actual)
                handles.append(wrapped)
                return wrapped
            return actual
        return patch.object(Path, "open", new=open_file), handles

    def test_event_write_failure_keeps_interrupted_evidence_and_cleans_every_owner(self):
        patched, handles = self.fail_event_writer()
        with patched, self.assertRaisesRegex(OSError, "event write failure"):
            self.run_trace("edf_b1")
        self.assertEqual(len(handles), 1)
        self.assertTrue(handles[0].close_attempted and handles[0].file.closed)
        self.assertTrue(self.arrivals.closed)
        self.assertEqual(self.actions.count(("close", "v")), 1)
        self.assertEqual(self.actions.count(("release", "v")), 1)
        self.assertEqual({row["status"] for row in self.snapshots[-1][0]}, {"interrupted"})
        self.assertFalse(any(action[0] in ("decode", "image") for action in self.actions))

    def test_event_close_failure_cannot_skip_arrivals_and_state_cleanup(self):
        patched, handles = self.fail_event_writer(close_also_fails=True)
        with patched, self.assertRaisesRegex(OSError, "event close failure"):
            self.run_trace("edf_b1")
        self.assertTrue(handles[0].close_attempted and handles[0].file.closed)
        self.assertTrue(self.arrivals.closed)
        self.assertEqual(self.actions.count(("close", "v")), 1)
        self.assertEqual(self.actions.count(("release", "v")), 1)


if __name__ == "__main__":
    unittest.main()
