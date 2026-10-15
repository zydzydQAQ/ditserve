"""Gates must not promote speed-only or incomplete Sana candidates."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.optimizations import next_phases as phases
from ditserve.research.common import atomic_json, read_json


class Image:
    def __init__(self, value):
        self.value = value
    def tobytes(self):
        return self.value.encode()
    def copy(self):
        return Image(self.value)


class Adapter:
    def __init__(self, clock, speed=.8, second_speed=None):
        self.clock, self.speed, self.candidate = clock, speed, "eager"
        self.second_speed = second_speed
        self.calls = 0
    def set_candidate(self, name):
        self.candidate = name
    def candidate_summary(self):
        return dict(vae_regions=[], transformer_graph=None)
    def steady_context(self):
        return nullcontext()
    def generate(self, prompt, *, seed, check_budget):
        check_budget()
        self.calls += 1
        candidate_cost = (self.second_speed if self.second_speed is not None and self.calls > 52 else self.speed)
        self.clock[0] += 1. if self.candidate == "eager" else candidate_cost
        return Image(prompt+str(seed)), dict(ms=999., finite=True, candidate=self.candidate)


class CandidateGates(unittest.TestCase):
    def run_case(self, *, quality=True, speed=.8, second_speed=None, resume=True, capacity_passed=True,
                 online_error=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root, clock = Path(self.tmp.name), [0.]
        adapter = Adapter(clock, speed, second_speed)
        context = SimpleNamespace(output=root, run_root=root, sana=adapter,
            opts=dict(prompts=["one", "two"]), config=dict(seeds=[101,102,201,202], quality={}),
            check_budget=lambda: None)
        def recover(ctx):
            self.assertEqual(ctx.sana.candidate, "vae_compile")
            result = dict(status="passed" if resume else "failed", resume_qualified=resume, eligible=resume)
            atomic_json(ctx.output/"sana-wan-resume.json", result)
            return result
        with patch.object(phases.time, "perf_counter", side_effect=lambda: clock[0]), \
             patch.object(phases, "_key_image", return_value={}), \
             patch("ditserve.optimizations.sana_phases._quality", return_value=dict(passed=quality)), \
             patch("ditserve.optimizations.worker.capacity", return_value=dict(passed=capacity_passed)) as capacity, \
             patch("ditserve.optimizations.next_resume.run_candidate_resume", side_effect=recover) as recovery, \
             patch.object(phases, "_candidate_online", side_effect=online_error) as online:
            result = phases.run_candidate(context, "vae_compile")
        return result, adapter, capacity, recovery, online

    def test_failed_quality_never_runs_capacity_or_online(self):
        result, adapter, capacity, recovery, online = self.run_case(quality=False)
        self.assertFalse(result["local_qualified"])
        capacity.assert_not_called()
        recovery.assert_not_called()
        online.assert_not_called()
        self.assertEqual(adapter.candidate, "eager")

    def test_pipeline_timer_cannot_replace_full_invocation_gate(self):
        result, adapter, capacity, recovery, online = self.run_case(speed=.99)
        self.assertEqual(result["quality_checked"], 32)
        self.assertAlmostEqual(result["round_speed_gains"][0], .01)
        self.assertFalse(result["local_qualified"])
        capacity.assert_not_called()
        online.assert_not_called()

    def test_restore_failure_blocks_online_even_after_speed_and_capacity(self):
        result, adapter, capacity, recovery, online = self.run_case(resume=False)
        self.assertTrue(result["performance_passed"])
        self.assertFalse(result["local_qualified"])
        capacity.assert_called_once()
        recovery.assert_called_once()
        online.assert_not_called()

    def test_only_one_execution_order_improves_is_rejected(self):
        result, adapter, capacity, recovery, online = self.run_case(second_speed=.99)
        self.assertGreater(result["round_speed_gains"][0], .05)
        self.assertLess(result["round_speed_gains"][1], .05)
        self.assertFalse(result["local_qualified"])
        capacity.assert_not_called()
        online.assert_not_called()

    def test_complete_local_gate_only_admits_separate_online_evaluation(self):
        result, adapter, capacity, recovery, online = self.run_case()
        self.assertTrue(result["local_qualified"])
        self.assertFalse(result["online_qualified"])
        self.assertEqual(len([r for r in result["rows"] if r["kind"]=="measurement"]), 96)
        self.assertEqual(result["quality_checked"], 32)
        self.assertEqual(result["summaries"]["eager"]["n"], 48)
        self.assertAlmostEqual(result["summaries"]["vae_compile"]["mean"], 800.)
        self.assertEqual(result["decision"], "未验证")
        self.assertEqual(result["reason"], "local gates passed; fixed online evaluation pending")
        online.assert_called_once()
        self.assertEqual(adapter.candidate, "eager")

    def test_capacity_negative_has_its_own_unverified_reason(self):
        result, adapter, capacity, recovery, online = self.run_case(capacity_passed=False)
        self.assertTrue(result["performance_passed"])
        self.assertFalse(result["local_qualified"])
        self.assertEqual(result["decision"], "未验证")
        self.assertEqual(result["reason"], "candidate co-residency capacity check did not pass")
        recovery.assert_not_called()
        online.assert_not_called()

    def test_online_failure_preserves_earlier_local_truth_and_propagates(self):
        with self.assertRaisesRegex(RuntimeError, "online execution failed"):
            self.run_case(online_error=RuntimeError("online execution failed"))
        record = read_json(Path(self.tmp.name) / "sana-candidate.json")
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["local_qualified"])
        self.assertFalse(record["online_qualified"])
        self.assertNotIn("expected_failure", record)


if __name__ == "__main__":
    unittest.main()
