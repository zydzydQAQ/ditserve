import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.scheduling import execution, next_execution
from ditserve.scheduling.next_policy import select_dispatch
import test_scheduling_execution as fixtures


class NextExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SchedulingReplayIntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def replay(self, arm, writer=None):
        f = self.fixture
        with patch.object(execution, "EncodedArrivals", f.arrival_type), \
             patch.object(execution.time, "perf_counter", side_effect=f.tick), \
             patch.object(execution.time, "sleep", side_effect=lambda seconds: f.tick()):
            return execution.run_trace(f.config, f.worker, f.executor, f.sana,
                f.requests, arm, f.costs, f.directory, f.check,
                lambda rows, elapsed: f.snapshots.append((copy.deepcopy(rows), elapsed)),
                dispatch=select_dispatch, view_factory=next_execution.remaining_view,
                artifact_writer=writer, instrument_control=True)

    def test_two_block_resumes_both_branches_and_finishes_all_overdue_requests(self):
        result = self.replay("edf_b2")
        advances = [a for a in self.fixture.actions if a[0] == "advance"]
        self.assertEqual([a[2:] for a in advances], [("cond", 0, 2), ("uncond", 0, 2)])
        self.assertTrue(all(r["status"] == "completed" for r in result["rows"]))
        self.assertEqual(result["metrics"]["on_time"], 0)
        self.assertTrue(self.fixture.arrivals.closed)
        for row in result["rows"]:
            self.assertLessEqual(row["finish_ms"], row["output_persistence_start_ms"])
            self.assertLessEqual(row["output_persistence_start_ms"], row["output_persistence_end_ms"])
            self.assertEqual(row["output_persistence_mode"], "synchronous")
        video = next(r for r in result["rows"] if r["kind"] == "video")
        self.assertLessEqual(video["output_persistence_end_ms"], video["state_release_start_ms"])
        self.assertLessEqual(video["state_release_end_ms"], result["elapsed_ms"])

    def test_two_block_estimate_caps_at_unconditional_tail(self):
        f = self.fixture
        state = SimpleNamespace(step_index=0)
        continuation = SimpleNamespace(branch="uncond", next_block=1)
        view = next_execution.remaining_view(f.requests[0], (state, continuation), f.costs, 0.)
        self.assertEqual(view["action_ms_by_quantum"]["2"], 4.)
        self.assertEqual(view["remaining_ms"], 6.)
        self.assertNotIn("2", execution.remaining_view(f.requests[0], (state, continuation), f.costs, 0.)["action_ms_by_quantum"])

    def test_async_results_are_drained_before_any_replay_can_qualify(self):
        from ditserve.scheduling.next_persistence import AsyncArtifacts
        writer = AsyncArtifacts(self.fixture.directory)
        try:
            result = self.replay("edf_b1_async", writer)
        finally:
            writer.close()
        self.assertEqual(result["output_persistence_mode"], "background")
        self.assertTrue(all(Path(p).is_file() for r in result["rows"] for p in r["outputs"].values()))
        self.assertFalse(result["quality"]["passed"])
        self.assertEqual(len([a for a in self.fixture.actions if a[0] == "advance"]), 4)
        self.assertTrue(self.fixture.arrivals.closed)

    def test_writer_failure_never_returns_success_or_leaks_model_states(self):
        class FailingWriter:
            def save(self, *unused):
                raise OSError("injected artifact persistence failure")
        with self.assertRaisesRegex(OSError, "artifact persistence failure"):
            self.replay("edf_b1_async", FailingWriter())
        self.assertTrue(self.fixture.arrivals.closed)
        self.assertEqual(self.fixture.actions.count(("release", "v")), 1)

    def test_phase_matrix_preserves_completion_and_fresh_baselines(self):
        config = dict(seed=301, validation_seeds=[401, 402])
        completion = next_execution.phase_matrix(config, "completion")
        self.assertEqual(len(completion), 1)
        self.assertEqual(completion[0][:2], ("video_contention", 301))
        self.assertEqual(len(completion[0][2]), 6)
        development = next_execution.phase_matrix(config, "granularity")
        self.assertEqual(sum(len(g[2]) for g in development), 16)
        self.assertEqual(development[0][2], list(reversed(development[1][2])))
        for selected, expected in [(None, 24), ("edf_b2", 32), ("edf_b1_async", 32)]:
            matrix = next_execution.phase_matrix(config, "replicate", selected)
            self.assertEqual(sum(len(g[2]) for g in matrix), expected)
            self.assertEqual({g[1] for g in matrix}, {401, 402})
            for _, _, arms in matrix:
                self.assertTrue(set(("edf_b1", "edf_b4", "edf_step")).issubset(arms))
        with self.assertRaises(ValueError):
            next_execution.phase_matrix(config, "replicate", "guarded_b4")


if __name__ == "__main__":
    unittest.main()
