"""CPU integration of the real executor with deterministic model adapters."""

import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from ditserve.sessions import execution, workload
from ditserve.sessions.policy import ARMS


class Clock:
    def __init__(self):
        self.value = 0.

    def perf_counter(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds

    def work(self):
        self.value += .001


class LLM:
    def __init__(self, clock):
        self.clock, self.states, self.preparations = clock, [], []
        self.device = "cpu"
        self.torch = SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda *args: None))

    def prepare(self, messages, *, max_new_tokens, previous=None, request_id=None):
        self.clock.work()
        if previous is not None:
            assert previous.done and not previous.released
            assert messages[1] == dict(role="assistant", content=self.finish(previous))
            self.release(previous)
        state = SimpleNamespace(request_id=request_id, generated_ids=[], done=False, released=False,
                                reused_tokens=3 if previous is not None else 0, maximum=max_new_tokens)
        self.states.append(state)
        self.preparations.append((request_id, copy.deepcopy(messages), previous))
        return state

    def advance(self, state):
        assert not state.done and not state.released
        self.clock.work()
        action = "prefill" if not state.generated_ids else "decode"
        state.generated_ids.append(10 + len(state.generated_ids))
        state.done = len(state.generated_ids) == state.maximum
        return dict(action=action, ms=1.)

    def finish(self, state):
        assert state.done
        return "image prompt for " + state.request_id

    def release(self, state):
        state.released = True


class Sana:
    def __init__(self, clock, fail=False):
        self.clock, self.states, self.fail = clock, [], fail

    def prepare(self, prompt, *, seed, request_id):
        self.clock.work()
        state = SimpleNamespace(seed=seed, request_id=request_id, num_steps=2, step_index=0, done=False, released=False)
        self.states.append(state)
        return state

    def advance(self, state):
        self.clock.work()
        if self.fail:
            raise RuntimeError("injected denoise failure")
        assert not state.done and not state.released
        state.step_index += 1
        state.done = state.step_index == state.num_steps
        return dict(action="denoise", ms=1.)

    def finish(self, state):
        self.clock.work()
        assert state.done and not state.released
        return np.array([state.seed % 1000, state.step_index], dtype=np.float32)

    def release(self, state):
        state.released = True


class SessionExecutionTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(session_count=3, think_ms=3, max_new_tokens=3)
        self.costs = dict(stage_ms=dict(zip(workload.STAGES, (1., 1., 2., 1., 2., 1.))),
                          expected_generated_tokens=3, decode_token_ms=1., image_step_ms=1.)
        self.trace = workload.make_trace(self.config, self.costs, .9)
        for session, start in zip(self.trace["sessions"], (0., 0., 5.)):
            session["start_ms"] = start
        self.partial = []

    def run_fake(self, directory, arm="session_serial", *, fail=False, trace=None, check=None, llm=None):
        clock = Clock() if llm is None else llm.clock
        llm, sana = llm or LLM(clock), Sana(clock, fail=fail)
        actual_select = execution.policy.select
        selected = []
        def selecting(rows, now, policy, active_id=None):
            self.assertTrue(all(row["release_ms"] <= now and row["ready"] is True for row in rows))
            rid = actual_select(rows, now, policy, active_id=active_id)
            selected.append((now, rid))
            return rid
        def partial(rows, elapsed):
            self.partial.append((copy.deepcopy(rows), elapsed))
        with mock.patch.object(execution, "time", SimpleNamespace(perf_counter=clock.perf_counter, sleep=clock.sleep)), \
                mock.patch.object(execution, "memory", return_value={"peak_allocated_bytes": 0}), \
                mock.patch.object(execution.policy, "select", side_effect=selecting):
            result = execution.run_trace(self.config, llm, sana, trace or self.trace, arm, self.costs,
                                         directory, check or (lambda: None), partial)
        return dict(result, arm=arm), llm, sana, selected

    def test_real_control_flow_keeps_causal_history_and_same_action_quanta(self):
        with tempfile.TemporaryDirectory() as folder:
            records = []
            for arm in ARMS:
                result, llm, sana, selected = self.run_fake(Path(folder) / arm, arm)
                records.append(result)
                self.assertEqual(6, len(result["rows"]))
                for sid in ("s000", "s001", "s002"):
                    first, second = sorted((row for row in result["rows"] if row["session_id"] == sid), key=lambda row: row["round"])
                    self.assertAlmostEqual(first["finish_ms"] + 3, second["release_ms"])
                    self.assertEqual(3, second["reused_tokens"])
                for row in result["rows"]:
                    self.assertEqual(dict(zip(workload.STAGES, (1, 1, 2, 1, 2, 1))), row["action_counts"])
                    self.assertGreaterEqual(row["first_token_ms"], row["llm_start_ms"] + 1)
                    self.assertEqual(2, len(row["token_intervals_ms"]))
                    self.assertTrue(all(interval >= .999999 for interval in row["token_intervals_ms"]))
                self.assertTrue(all(state.released for state in llm.states + sana.states))
                self.assertEqual(6, result["persistence"]["completed"])
            self.assertTrue(execution.verify_group(records))
            self.assertTrue(all(record["quality"]["passed"] for record in records))
            self.assertTrue(all(row["quality"]["tokens_exact"] for record in records for row in record["rows"]))
            self.assertFalse(any(thread.name == "scheduling-artifacts" for thread in threading.enumerate()))

    def test_all_overdue_rounds_complete_instead_of_being_dropped(self):
        trace = copy.deepcopy(self.trace)
        trace["turn_deadline_window_ms"] = .01
        with tempfile.TemporaryDirectory() as folder:
            for arm in ARMS:
                result, _, _, _ = self.run_fake(Path(folder) / arm, arm, trace=trace)
                self.assertEqual(6, len(result["rows"]))
                self.assertTrue(all(row["status"] == "completed" and row["finish_ms"] > row["deadline_ms"] for row in result["rows"]))

    def test_failure_cleans_every_live_state_and_writer_thread(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = Clock()
            llm = LLM(clock)
            sana = Sana(clock, fail=True)
            with mock.patch.object(execution, "time", SimpleNamespace(perf_counter=clock.perf_counter, sleep=clock.sleep)), \
                    mock.patch.object(execution, "memory", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "denoise failure"):
                    execution.run_trace(self.config, llm, sana, self.trace, "stage_edf", self.costs,
                                        Path(folder) / "failed", lambda: None,
                                        lambda rows, elapsed: self.partial.append(copy.deepcopy(rows)))
            self.assertTrue(llm.states)
            self.assertTrue(all(state.released for state in llm.states + sana.states))
            self.assertTrue(all(row["status"] in ("completed", "failed") for row in self.partial[-1]))
            self.assertFalse(any(thread.name == "scheduling-artifacts" for thread in threading.enumerate()))

    def test_initialization_failure_closes_writer_before_any_model_action(self):
        for fault in ("reset", "event_open"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                clock = Clock()
                llm, sana, writer = LLM(clock), Sana(clock), mock.Mock()
                if fault == "reset":
                    llm.torch.cuda.reset_peak_memory_stats = mock.Mock(side_effect=RuntimeError("initial reset failed"))
                    injected = mock.patch.object(execution, "memory", return_value={})
                else:
                    injected = mock.patch.object(Path, "open", side_effect=RuntimeError("event open failed"))
                with mock.patch.object(execution, "AsyncArtifacts", return_value=writer), injected:
                    with self.assertRaisesRegex(RuntimeError, "failed"):
                        execution.run_trace(self.config, llm, sana, self.trace, "session_serial", self.costs,
                                            Path(folder) / "init-failed", lambda: None, lambda rows, elapsed: None)
                writer.close.assert_called_once_with()
                self.assertFalse(llm.states or sana.states)

    def test_writer_failure_is_not_a_successful_replay_and_still_joins(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch("ditserve.scheduling.next_persistence.np.save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "artifact persistence failed"):
                self.run_fake(Path(folder) / "failed")
            self.assertFalse(any(thread.name == "scheduling-artifacts" for thread in threading.enumerate()))

    def test_quality_pair_rejects_token_image_and_prompt_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            records = [self.run_fake(Path(folder) / arm, arm)[0] for arm in ARMS]
            self.assertTrue(execution.verify_group(records))
            for field in ("llm_token_ids", "llm_text", "output"):
                trial = copy.deepcopy(records)
                row = trial[1]["rows"][0]
                if field == "llm_token_ids":
                    row[field][0] += 1
                elif field == "llm_text":
                    row[field] += " changed"
                else:
                    path = Path(folder) / "changed.npy"
                    np.save(path, np.array([99, 2], dtype=np.float32))
                    row["outputs"]["output"] = str(path)
                self.assertFalse(execution.verify_group(trial))

    def test_first_token_and_intervals_are_end_to_end_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _, _, _ = self.run_fake(Path(folder) / "serial")
            rows = {row["request_id"]: row for row in result["rows"]}
            first, queued = rows["s000-r1"], rows["s001-r1"]
            self.assertAlmostEqual(2., first["first_token_ms"] - first["release_ms"])
            self.assertGreater(queued["first_token_ms"] - queued["release_ms"], 2.)
            self.assertTrue(all(abs(value - 1.) < 1e-8 for value in first["token_intervals_ms"]))

    def test_missing_history_is_never_substituted_for_real_first_response(self):
        with self.assertRaises(ValueError):
            execution.messages_for(self.trace["sessions"][0], 1)
        messages = execution.messages_for(self.trace["sessions"][0], 1, "actual output")
        self.assertEqual(dict(role="assistant", content="actual output"), messages[1])

    def test_sampler_snapshot_detects_index_rng_and_latent_mutations(self):
        expected = dict(step_index=1, sampler_step=1, sampler_begin=0, done=False,
                        generator_state=np.array([1, 2, 3], dtype=np.uint8),
                        timesteps=np.array([1.5, 1.3, 0.], dtype=np.float32),
                        latents=np.array([1.], dtype=np.float32), denoised=np.array([2.], dtype=np.float32))
        self.assertTrue(execution._sana_check(copy.deepcopy(expected), expected)["exact"])
        for field in ("step_index", "sampler_step", "sampler_begin", "done", "generator_state", "latents"):
            with self.subTest(field=field):
                changed = copy.deepcopy(expected)
                if field == "done":
                    changed[field] = True
                elif field in ("generator_state", "latents"):
                    changed[field][0] += 1
                else:
                    changed[field] += 1
                comparison = execution._sana_check(changed, expected)
                self.assertFalse(comparison["passed"])
                self.assertFalse(comparison["exact"])
        close = copy.deepcopy(expected)
        close["latents"][0] += 1e-6
        tolerated = execution._sana_check(close, expected)
        self.assertTrue(tolerated["passed"])
        self.assertFalse(tolerated["exact"])

    def test_qualification_uses_completed_continuous_kv_reference_before_interference(self):
        clock, order = Clock(), []

        class QualifiedLLM(LLM):
            def prepare(self, messages, **kwargs):
                state = super().prepare(messages, **kwargs)
                state.maximum = min(state.maximum, 3)
                state.prepare_ms = 1.
                return state

            def advance(self, state):
                event = super().advance(state)
                order.append(("llm", state.request_id, len(state.generated_ids)))
                return event

            def finish(self, state):
                assert state.done
                return "same actual generated image prompt"

            def reference(self, messages, *, max_new_tokens):
                return dict(text="same actual generated image prompt", token_ids=[10, 11, 12])

            def snapshot(self, state):
                clock.sleep(.5)  # Deliberately expensive copies are not action cost.
                values = np.array([len(state.generated_ids)], dtype=np.float32)
                return dict(prompt_ids=[1, 2], generated_ids=list(state.generated_ids),
                            cached_ids=[1, 2] + state.generated_ids[:-1], done=state.done,
                            cache=((values, values.copy()),))

        class QualifiedSana(Sana):
            def prepare(self, prompt, seed=0, **kwargs):
                state = super().prepare(prompt, seed=seed, **kwargs)
                state.prepare_ms = 1.
                state.latents = np.array([0.], dtype=np.float32)
                return state

            def advance(self, state):
                event = super().advance(state)
                state.latents = np.array([state.step_index], dtype=np.float32)
                order.append(("sana", state.request_id, state.step_index))
                return event

            def finish(self, state):
                result = super().finish(state)
                state.decode_ms = 1.
                return result

            def snapshot(self, state):
                clock.sleep(.5)
                return dict(step_index=state.step_index, sampler_step=state.step_index, sampler_begin=0, done=state.done,
                            generator_state=np.array([state.step_index], dtype=np.uint8), latents=state.latents.copy())

            def reference(self, prompt, seed=0):
                order.append(("sana-reference", prompt, seed))
                return dict(output=np.array([seed % 1000, 2], dtype=np.float32),
                            step_latents=[np.array([1.], dtype=np.float32), np.array([2.], dtype=np.float32)])

        llm, sana = QualifiedLLM(clock), QualifiedSana(clock)
        config = dict(self.config, seed=301, qualification_session_indices=[0],
                      session_inputs=[["first input", "second input"]])
        result = {}
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(execution, "memory", return_value={}), \
                mock.patch("builtins.print"):
            execution._qualification(config, Path(folder), llm, sana, result, lambda: None, lambda: None)
        self.assertEqual("passed", result["status"])
        self.assertEqual(2, len(result["checks"]))
        for turn in (0, 1):
            native = [index for index, item in enumerate(order) if item[:2] == ("llm", "native-0-%d" % turn)]
            resumed = [index for index, item in enumerate(order) if item[:2] == ("llm", "resumed-0-%d" % turn)]
            self.assertEqual(3, len(native))
            self.assertEqual(3, len(resumed))
            self.assertLess(max(native), min(resumed))
            self.assertTrue(any(order[index][0] == "sana-reference" for index in range(min(resumed), max(resumed))))
            self.assertFalse(any(order[index][0] == "sana-reference" for index in range(min(native), max(native))))
        self.assertAlmostEqual(2.2, result["cost_model"]["stage_ms"]["llm_decode"])
        self.assertAlmostEqual(2.2, result["cost_model"]["stage_ms"]["image_denoise"])
        self.assertTrue(all(state.released for state in llm.states + sana.states))


if __name__ == "__main__":
    unittest.main()
