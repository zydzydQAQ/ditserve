import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.directions import direction3
from ditserve.directions.budget import BudgetStop
from ditserve.directions.streaming import WallClockPlayer


def configuration():
    return {"direction3": {}, "models": {"longlive": {"revision": direction3.SOURCE_REVISION}}}


class Direction3Tests(unittest.TestCase):
    def test_defaults_freeze_scientific_options(self):
        value = direction3.effective_options(configuration())
        self.assertEqual(value["correctness_blocks"], 6)
        self.assertEqual(len(value["prompt_pairs"]), 2)
        self.assertEqual({a[1] for a in value["update_arms"] if a[0] == "latest_only"}, {0, 1, 2})
        self.assertEqual(value["multisession_policies"], ["round_robin", "minimum_playback_slack"])

    def test_profile_requires_correctness_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(direction3, "NativeLongLive", side_effect=AssertionError("GPU touched")):
                result = direction3.run_phase(configuration(), Path(directory), lambda: None, "profile")
            self.assertEqual(result["status"], "blocked")
            self.assertFalse(result["eligible"])

    def test_multisession_throughput_gate_avoids_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "worker-profile.json").write_text(json.dumps({"eligible": True, "measured_fps": 20.}))
            (path / "worker-update.json").write_text(json.dumps({"eligible": True}))
            with patch.object(direction3, "NativeLongLive", side_effect=AssertionError("GPU touched")):
                result = direction3.run_phase(configuration(), path, lambda: None, "multisession")
            self.assertEqual(result["status"], "stopped")
            self.assertIn("two_times_16fps", result["reason"])

    def test_budget_stop_persists_partial_without_loading_model(self):
        class BudgetStop(RuntimeError):
            pass
        def stop():
            raise BudgetStop("spent")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(direction3, "NativeLongLive", side_effect=AssertionError("GPU touched")):
                result = direction3.run_phase(configuration(), Path(directory), stop, "correctness")
            self.assertEqual(result["status"], "stopped")
            saved = json.loads((Path(directory) / "correctness/partial.json").read_text())
            self.assertEqual(saved["reason"], "budget_stop")

    def test_fixed_fps_cannot_be_lowered_to_pass(self):
        config = configuration()
        config["direction3"]["fps"] = 8
        with self.assertRaises(ValueError):
            direction3.validate_config(config)

    def test_trace_is_exogenous_and_includes_supersedable_burst(self):
        sparse = direction3.update_events(("before", "after"), .5, "sparse")
        burst = direction3.update_events(("before", "after"), .5, "burst")
        self.assertEqual(sparse[1].arrival_s, burst[1].arrival_s)
        self.assertLess(burst[2].arrival_s - burst[1].arrival_s, .5)
        self.assertEqual(len(burst), 9)
        self.assertEqual([e.prompt for e in burst[:3]], ["before", "after", "before"])
        self.assertGreater(direction3.observation_seconds(burst, {}), burst[-1].arrival_s)

    def test_profile_fast_transition_chunks_cannot_qualify_slow_rolling_steady_state(self):
        class Runtime:
            def __init__(self):
                self.index, self.released = 0, False
            def new_session(self, *args):
                return object()
            def condition(self, *args):
                return dict(encode_s=0., recache_s=0.)
            def chunk(self, session):
                index = self.index
                self.index += 1
                return None, [], dict(block=index, pixel_frames=9 if index == 0 else 12,
                    total_s=.01 if index < 4 else .9)
            def release(self, session):
                self.released = True
        config = configuration()
        config["direction3"] = direction3.effective_options(config)
        runtime, report = Runtime(), dict(samples=[])
        with tempfile.TemporaryDirectory() as directory, patch.object(direction3, "_artifact", return_value={}):
            direction3._profile(runtime, config, Path(directory), report)
        self.assertEqual(report["configured_warmup_blocks"], 2)
        self.assertEqual(report["effective_warmup_blocks"], 4)
        self.assertEqual(report["first_kv_roll_chunk_index"], 4)
        self.assertEqual(report["measured_chunk_indices"], list(range(4, 10)))
        self.assertEqual(config["direction3"]["warmup_blocks"], 2)
        self.assertAlmostEqual(report["measured_fps"], 12 / .9)
        self.assertLess(report["measured_fps"], 16)
        old_measured = report["samples"][2:]
        self.assertGreater(sum(row["pixel_frames"] for row in old_measured) /
                           sum(row["total_s"] for row in old_measured), 16)
        self.assertEqual(report["status"], "stopped")
        self.assertFalse(report["eligible"])
        self.assertTrue(runtime.released)

    def test_profile_requires_three_chunks_after_effective_not_configured_warmup(self):
        config = configuration()
        config["direction3"] = direction3.effective_options(config)
        config["direction3"]["profile_blocks"] = 6
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimpleNamespace(new_session=lambda *args: self.fail("unqualified profile allocated model state"))
            with self.assertRaisesRegex(ValueError, "three measured steady-state chunks"):
                direction3._profile(runtime, config, Path(directory), dict(samples=[]))

    def test_artifact_writer_does_not_block_generation(self):
        entered, unblock = threading.Event(), threading.Event()
        def save(*unused):
            entered.set()
            if not unblock.wait(2.):
                raise AssertionError("writer did not run independently")
            return {"path": "actual", "frames": 1}
        with patch.object(direction3, "_artifact", save):
            writer, row = direction3.ArtifactWriter(), {}
            try:
                writer.submit(Path("."), "test", object(), row)
                self.assertTrue(entered.wait(1.))
                self.assertEqual(row["artifact"]["status"], "pending")
                unblock.set()
                stats = writer.finish(lambda: None)
                self.assertEqual(row["artifact"]["path"], "actual")
                self.assertGreaterEqual(stats["peak_pending_chunks"], 1)
            finally:
                unblock.set()
                writer.close()

    def test_multisession_slack_priority_and_fair_ties(self):
        players = {"s0": SimpleNamespace(buffered_seconds=lambda: 1.),
                   "s1": SimpleNamespace(buffered_seconds=lambda: 0.)}
        args = (["s0", "s1"], ["s0", "s1"], players, 0)
        self.assertEqual(direction3.choose_session(*args, "round_robin")[0], "s0")
        self.assertEqual(direction3.choose_session(*args, "minimum_playback_slack")[0], "s1")
        players["s0"] = players["s1"]
        first, cursor = direction3.choose_session(*args, "minimum_playback_slack")
        second, _ = direction3.choose_session(args[0], args[1], players, cursor, "minimum_playback_slack")
        self.assertEqual([first, second], ["s0", "s1"])

    def test_minimum_block_capacity_cannot_end_generation_early(self):
        class Runtime:
            check_budget = staticmethod(lambda: None)
            def new_session(self, sid, seed, blocks):
                return SimpleNamespace(block=0, conditional=None, prompt_version=None)
            def condition(self, state, prompt, version):
                state.conditional, state.prompt_version = prompt, version
                return {"encode_s": 0., "recache_s": 0.}
            def chunk(self, state):
                state.block += 1
                return None, [object() for _ in range(12)], {"block": state.block - 1, "pixel_frames": 12}
            def release(self, state):
                pass
        events = [direction3.PromptEvent(0, 0., "initial"),
                  direction3.PromptEvent(1, .03, "late but observed")]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(direction3, "_artifact", return_value={"path": "test", "frames": 1}):
                row = direction3._stream_run(Runtime(), events, "fifo", 1, 1, 1,
                    Path(directory), observation_end_s=.09)
        self.assertEqual(row["scheduled_events"], row["observed_events"])
        self.assertIn(row["updates"]["s0"][-1]["status"], ("pending_at_stop", "applied"))
        self.assertGreater(len(row["chunks"]), 1)
        self.assertTrue(row["generation_ends_only_at_window"])
        self.assertGreaterEqual(row["observation_end_s"], .09)


class Direction3StreamingAdmissionTests(unittest.TestCase):
    class Runtime:
        check_budget = staticmethod(lambda: None)

        def __init__(self, condition_hook=lambda *args: None, failure=None, error=BudgetStop):
            self.condition_hook, self.failure, self.error = condition_hook, failure, error
            self.calls, self.created, self.released = [], [], []

        def new_session(self, sid, seed, blocks):
            if self.failure == "new_session" and sid == "s1":
                raise self.error("cancel during session setup")
            state = SimpleNamespace(session_id=sid, block=0, conditional=None, prompt_version=None)
            self.created.append(state)
            return state

        def condition(self, state, prompt, version):
            self.calls.append(("condition", version))
            started = time.monotonic()
            if self.failure == "condition":
                raise self.error("cancel during condition")
            self.condition_hook(state, prompt, version)
            state.conditional, state.prompt_version = True, version
            elapsed = time.monotonic() - started
            return dict(encode_s=elapsed / 2, recache_s=elapsed / 2)

        def chunk(self, state):
            self.calls.append(("chunk", state.prompt_version))
            if self.failure == "chunk":
                raise self.error("cancel during reserved generation")
            block = state.block
            state.block += 1
            count = 9 if block == 0 else 12
            return None, [object() for _ in range(count)], dict(
                block=block, pixel_frames=count, denoise_context_s=.0001)

        def release(self, state):
            self.released.append(state)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.players = []
        def player(*args, **kwargs):
            value = WallClockPlayer(*args, **kwargs)
            self.players.append(value)
            return value
        self.player_patch = patch.object(direction3, "WallClockPlayer", side_effect=player)
        self.player_patch.start()
        self.addCleanup(self.player_patch.stop)
        self.artifact_patch = patch.object(direction3, "_artifact", side_effect=lambda directory, name, pixels:
                                    dict(path=str(directory / (name + ".npy")), frames=len(pixels)))
        self.artifact_patch.start()
        self.addCleanup(self.artifact_patch.stop)

    def test_full_buffer_does_not_delay_latest_condition_or_flush(self):
        for policy in ("latest_only", "latest_only_flush"):
            with self.subTest(policy=policy):
                observed = []
                def condition(state, prompt, version):
                    if version == 1:
                        observed.append(self.players[-1].report())
                runtime = self.Runtime(condition_hook=condition)
                events = [direction3.PromptEvent(0, 0., "initial"),
                          direction3.PromptEvent(1, .02, "new")]
                result = direction3._stream_run(runtime, events, policy, 0, 1, 1,
                    self.root / policy, observation_end_s=.15)
                self.assertEqual([call for call in runtime.calls if call[0] == "condition"],
                                 [("condition", 0), ("condition", 1)])
                self.assertGreater(observed[0]["unconsumed_frames"], 0)
                self.assertEqual(observed[0]["resident_chunks"], 1)
                self.assertEqual(result["updates"]["s0"][1]["status"], "applied")
                self.assertLess(result["updates"]["s0"][1]["condition_applied_s"], .15)
                self.assertEqual(result["scheduled_events"], result["observed_events"])
                playback = result["playback"]["s0"]
                self.assertEqual(playback["peak_ahead_chunks"], 0)
                if policy.endswith("flush"):
                    self.assertGreater(len(playback["dropped"]), 0)
                    self.assertTrue(any(chunk["prompt_version"] == 1 for chunk in result["chunks"]))
                self.assertTrue(all(not player._thread.is_alive() for player in self.players))

    def test_updates_arriving_during_condition_merge_before_any_chunk_and_keep_wasted_work(self):
        earliest = time.monotonic()
        class GuardedEvent:
            session_id = "s0"
            def __init__(self, version, arrival_s):
                self.version, self.arrival_s = version, arrival_s
            @property
            def prompt(self):
                if time.monotonic() < earliest + self.arrival_s:
                    raise AssertionError("future condition payload was read")
                return "prompt%d" % self.version
        def condition(state, prompt, version):
            if version == 0:
                time.sleep(.045)
        runtime = self.Runtime(condition_hook=condition)
        events = [GuardedEvent(0, 0.), GuardedEvent(1, .01), GuardedEvent(2, .02)]
        result = direction3._stream_run(runtime, events, "latest_only", 0, 1, 1,
            self.root / "merge", observation_end_s=.16)
        self.assertEqual(runtime.calls[:3], [("condition", 0), ("condition", 2), ("chunk", 2)])
        self.assertNotIn(("condition", 1), runtime.calls)
        self.assertTrue(all(chunk["prompt_version"] == 2 for chunk in result["chunks"]))
        updates = result["updates"]["s0"]
        self.assertEqual(len(updates), 3)
        self.assertEqual(updates[1]["status"], "superseded_before_encoding")
        wasted = result["discarded_work"]["s0"]["conditioning_without_decoded_output"]
        self.assertEqual([row["version"] for row in wasted], [0])
        self.assertEqual(wasted[0]["reason"], "overwritten_without_decoded_chunk")
        self.assertGreater(wasted[0]["encode_wall_s"], 0)
        self.assertGreater(wasted[0]["recache_wall_s"], 0)
        partial = json.loads((self.root / "merge/partial.json").read_text())
        self.assertEqual(partial["discarded_work"]["s0"]["conditioning_without_decoded_output"], wasted)

    def test_fifo_produces_each_updated_version_before_admitting_next(self):
        def condition(state, prompt, version):
            if version == 0:
                time.sleep(.045)
        runtime = self.Runtime(condition_hook=condition)
        events = [direction3.PromptEvent(version, arrival, "prompt%d" % version)
                  for version, arrival in enumerate((0., .01, .02))]
        result = direction3._stream_run(runtime, events, "fifo", 2, 1, 1,
            self.root / "fifo", observation_end_s=.16)
        self.assertEqual(runtime.calls[:6], [("condition", 0), ("chunk", 0),
                         ("condition", 1), ("chunk", 1), ("condition", 2), ("chunk", 2)])
        self.assertEqual([chunk["pixel_frames"] for chunk in result["chunks"][:3]], [9, 12, 12])
        self.assertEqual([chunk["prompt_version"] for chunk in result["chunks"][:3]], [0, 1, 2])
        self.assertEqual(result["scheduled_events"], result["observed_events"])
        self.assertLessEqual(result["playback"]["s0"]["peak_ahead_chunks"], 2)

    def test_condition_crossing_window_keeps_all_scheduled_updates_in_denominator(self):
        runtime = self.Runtime(condition_hook=lambda *args: time.sleep(.06))
        events = [direction3.PromptEvent(version, arrival, "prompt%d" % version)
                  for version, arrival in enumerate((0., .01, .02))]
        result = direction3._stream_run(runtime, events, "latest_only", 0, 1, 1,
            self.root / "cutoff", observation_end_s=.04)
        self.assertEqual(result["scheduled_events"], 3)
        self.assertEqual(result["observed_events"], 3)
        self.assertEqual(len(result["updates"]["s0"]), 3)
        self.assertEqual(result["chunks"], [])
        self.assertTrue(result["updates"]["s0"][0]["condition_after_observation_end"])
        self.assertEqual(result["updates"]["s0"][2]["status"], "pending_at_stop")
        wasted = result["discarded_work"]["s0"]["conditioning_without_decoded_output"]
        self.assertEqual(wasted[0]["reason"], "window_cutoff_without_decoded_chunk")

    def test_budget_or_cancellation_preserves_partial_and_releases_players_states_and_reservations(self):
        for failure in ("new_session", "condition", "chunk"):
            for error in (BudgetStop, KeyboardInterrupt):
                with self.subTest(failure=failure, error=error):
                    runtime = self.Runtime(failure=failure, error=error)
                    sessions = 2 if failure == "new_session" else 1
                    events = [direction3.PromptEvent(0, 0., "initial", "s%d" % i) for i in range(sessions)]
                    events.append(direction3.PromptEvent(1, .1, "future"))
                    output = self.root / (failure + error.__name__)
                    with self.assertRaises(error):
                        direction3._stream_run(runtime, events, "latest_only", 0, 1, 1,
                            output, session_count=sessions, observation_end_s=.2)
                    partial = json.loads((output / "partial.json").read_text())
                    self.assertEqual(partial["scheduled_events"], len(events))
                    self.assertLessEqual(partial["observed_events"], partial["scheduled_events"])
                    self.assertTrue(all(row["reserved_chunks"] == 0 for row in partial["playback"].values()))
                    self.assertEqual(runtime.created, runtime.released)
                    self.assertTrue(all(not player._thread.is_alive() for player in self.players))
                    if failure == "condition":
                        self.assertEqual(partial["updates"]["s0"][0]["status"], "conditioning_interrupted")


if __name__ == "__main__":
    unittest.main()
