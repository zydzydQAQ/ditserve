"""CPU checks for the explicit, non-promoting single-seed screening protocol."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.optimizations import next_workloads as module
from ditserve.optimizations.online import request_identity
from test_optimizations_next_workloads import Fixture, measured


class QuickCacheWorkloadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.fixture = Fixture(Path(directory.name))
        self.workloads = self.fixture.build()
        self.context = self.fixture.context()
        self.traces = self.workloads["highload"]["0.9"]

    def test_one_seed_both_patterns_abba_preserve_original_requests(self):
        original = copy.deepcopy(self.traces)
        with patch.object(module, "run_trace", side_effect=measured):
            result = module.run_cache_load(self.context, self.traces, "quick_09", quick_seed=101)
        self.assertEqual(self.traces, original)
        self.assertEqual(result["planned_replays"], 8)
        self.assertEqual(result["planned_requests"], 192)
        self.assertEqual(len(result["rows"]), 8)
        self.assertEqual(result["protocol_mode"], "single_seed_screening")
        self.assertEqual(result["selected_seed"], 101)
        self.assertEqual(result["status"], "stopped")
        self.assertTrue(result["screening_complete"])
        self.assertFalse(result["evaluation_complete"])
        self.assertEqual({row["seed"] for row in result["rows"]}, {101})
        expected = [t for t in original if t["seed"] == 101]
        self.assertEqual(result["expected_trace_ids"], [t["trace_sha256"] for t in expected])
        for index, trace in enumerate(expected):
            rows = result["rows"][4*index:4*(index+1)]
            self.assertEqual([(r["order_index"], r["arm"]) for r in rows],
                [(0, "baseline"), (0, "candidate"), (1, "candidate"), (1, "baseline")])
            for row in rows:
                self.assertEqual(module._inputs(row["requests"]), module._inputs(trace["requests"]))
                self.assertEqual(row["trace_sha256"], trace["trace_sha256"])

    def test_even_passing_screening_cannot_promote_or_claim_full_qualification(self):
        gate = dict(passed=True, failures=[], pooled={"fixture": True})
        with patch.object(module, "run_trace", side_effect=measured), \
                patch.object(module, "paired_gate", return_value=gate):
            result = module.run_cache_load(self.context, self.traces, "quick_pass", quick_seed=101)
        self.assertTrue(result["screening_passed"])
        self.assertEqual(result["decision"], "screening_gate_passed")
        self.assertFalse(result["eligible"])
        self.assertFalse(result["online_qualified"])
        self.assertFalse(result["full_qualification_completed"])
        self.assertTrue(result["screening_complete"])
        self.assertFalse(result["evaluation_complete"])
        self.assertNotIn("comparison", result)
        self.assertEqual(result["screening_comparison"], gate)
        self.assertEqual(json.loads((self.context.output / "quick_pass.json").read_text()), result)

    def test_subset_is_only_accepted_with_explicit_screening_mode(self):
        subset = [t for t in self.traces if t["seed"] == 101]
        with patch.object(module, "run_trace", side_effect=measured) as run:
            with self.assertRaisesRegex(ValueError, "three seeds"):
                module.run_cache_load(self.context, subset, "unapproved_subset")
            run.assert_not_called()
            result = module.run_cache_load(self.context, list(reversed(subset)), "explicit_subset", quick_seed=101)
        self.assertEqual([r["pattern"] for r in result["rows"]], ["ordinary"]*4 + ["burst"]*4)

    def test_rejects_unfrozen_seed_missing_pattern_duplicate_and_mixed_load(self):
        subset = [t for t in self.traces if t["seed"] == 101]
        mixed = copy.deepcopy(subset)
        mixed[1]["load"] = .7
        cases = [(self.traces, s) for s in (True, 101., "101", 102, 103)]
        cases += [(subset[:1], 101), ([subset[0], subset[0]], 101), (mixed, 101)]
        with patch.object(module, "run_trace") as run:
            for traces, seed in cases:
                with self.subTest(seed=seed, count=len(traces)):
                    with self.assertRaises(ValueError):
                        module.run_cache_load(self.context, traces, "invalid", quick_seed=seed)
            run.assert_not_called()

    def test_rehashed_deadline_mutation_is_rejected_before_gpu_work(self):
        mutated = copy.deepcopy(self.traces)
        mutated[0]["requests"][1]["deadline_ms"] += 1
        mutated[0]["trace_sha256"] = request_identity(mutated[0]["requests"])
        with patch.object(module, "run_trace") as run:
            with self.assertRaisesRegex(ValueError, "generator/calibration"):
                module.run_cache_load(self.context, mutated, "invalid_clock", quick_seed=101)
            run.assert_not_called()

    def test_interruption_keeps_completed_and_active_denominators(self):
        calls = []
        original_executor, original_partial = self.context.executor, self.context.save_partial

        def interrupted(context, requests, options):
            calls.append(1)
            if len(calls) == 1:
                return measured(context, requests, options)
            context.save_partial(dict(status="interrupted", complete=False, events=[],
                requests=[dict(r, arrived=True, status="interrupted") for r in requests]))
            raise KeyboardInterrupt("budget or cancellation")

        with patch.object(module, "run_trace", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                module.run_cache_load(self.context, self.traces, "quick_interrupted", quick_seed=101)
        record = json.loads((self.context.output / "quick_interrupted.json").read_text())
        self.assertEqual(record["status"], "incomplete")
        self.assertEqual(record["planned_requests"], 192)
        self.assertEqual(len(record["rows"]), 1)
        self.assertEqual(record["incomplete_trace"]["arrived"], 24)
        self.assertFalse(record["screening_passed"])
        self.assertFalse(record["online_qualified"])
        self.assertFalse(record["evaluation_complete"])
        self.assertIs(self.context.executor, original_executor)
        self.assertIs(self.context.save_partial, original_partial)


if __name__ == "__main__":
    unittest.main()
