"""Deterministic CPU fixtures for component gates, accounting and partial data."""

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ditserve.directions.budget import BudgetStop
from ditserve.optimizations.sana_phases import SanaPhaseContext, run_compile, run_batch, _burst


class Clock:
    def __init__(self):
        self.now = 1.
    def __call__(self):
        return self.now


class Adapter:
    def __init__(self, clock, compiled_ms=240., batch_ms=350.):
        self.clock, self.compiled_ms, self.batch_ms = clock, compiled_ms, batch_ms
        self.backend = "eager"
        self.count = 0
    def _sync(self):
        pass
    def set_backend(self, name):
        self.backend = name
    def steady_context(self):
        return nullcontext()
    def compilation_summary(self):
        return dict(graph_compile_count=0, records=[])
    def generate(self, prompt, *, seed, check_budget):
        return self.generate_batch([prompt], seeds=[seed], check_budget=check_budget)
    def generate_batch(self, prompts, *, seeds, check_budget):
        check_budget()
        self.count += 1
        duration = self.batch_ms if len(prompts) == 2 else (300. if self.backend == "eager" else self.compiled_ms)
        self.clock.now += duration / 1000
        images = np.stack([np.full((2, 2, 3), (seed % 17) / 17, dtype=np.float32) for seed in seeds])
        return images, dict(ms=duration, cost_per_image_ms=duration / len(prompts), batch_size=len(prompts),
                            compile_delta=0, backend=self.backend)


def quality(reference, candidate, **kwargs):
    value = float(np.max(np.abs(reference - candidate)))
    return dict(status="ok", finite=True, range_valid=True, lpips_mean=value, lpips_p95=value,
                psnr_db=None, temporal_delta_rmse=None)


class PhaseTests(unittest.TestCase):
    def context(self, path, clock, **kwargs):
        return SanaPhaseContext(adapter=Adapter(clock), output=Path(path), prompts=["a", "b"],
                                check_budget=lambda: None, capacity_check=lambda: {"passed": True},
                                quality_evaluator=quality, metadata={"image_deadline_ms": 791.8668436468579},
                                **kwargs)

    def test_compile_two_orders_full_input_denominator_and_local_only(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            context = self.context(directory, clock)
            with patch("ditserve.optimizations.sana_phases.time.perf_counter", clock):
                result = run_compile(context)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["local_qualified"])
            self.assertFalse(result["system_qualified"])
            self.assertEqual(result["selected_backend"], "compiled")
            self.assertEqual(result["quality_checked"], 32)
            self.assertEqual(result["summaries"]["eager"]["n"], 48)
            self.assertEqual(result["summaries"]["compiled"]["n"], 48)
            self.assertTrue(all(abs(gain - .2) < 1e-9 for gain in result["round_speed_gains"]))

    def test_compile_slow_or_missing_capacity_keeps_eager(self):
        for missing in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                clock = Clock()
                context = self.context(directory, clock)
                if missing:
                    context.capacity_check = None
                else:
                    context.adapter.compiled_ms = 301
                with patch("ditserve.optimizations.sana_phases.time.perf_counter", clock):
                    result = run_compile(context)
                self.assertFalse(result["local_qualified"])
                self.assertEqual(result["selected_backend"], "eager")

    def test_quality_failure_does_not_disappear_in_average(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            context = self.context(directory, clock)
            calls = []
            def one_bad(reference, candidate, **kwargs):
                calls.append(1)
                value = quality(reference, candidate)
                if len(calls) == 9:
                    value.update(lpips_mean=.011, lpips_p95=.011)
                return value
            context.quality_evaluator = one_bad
            with patch("ditserve.optimizations.sana_phases.time.perf_counter", clock):
                result = run_compile(context)
            self.assertFalse(result["quality_passed"])
            self.assertFalse(result["local_qualified"])

    def test_batch_local_gate_uses_original_deadline_and_full_batch_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            context = self.context(directory, clock,
                compile_report={"status": "passed", "local_qualified": True, "selected_backend": "compiled"})
            with patch("ditserve.optimizations.sana_phases.time.perf_counter", clock):
                result = run_batch(context)
            self.assertTrue(result["local_qualified"])
            self.assertEqual(result["selected_batch_size"], 2)
            self.assertEqual(result["quality_checked"], 68)
            self.assertFalse(result["system_qualified"])
            self.assertEqual(result["summaries"]["1"]["invocation_ms"]["n"], 48)
            self.assertEqual(result["summaries"]["2"]["invocation_ms"]["n"], 24)
            burst = next(row for row in result["rows"] if row["kind"] == "burst" and row["batch_size"] == 2)
            self.assertEqual(burst["requests"][0]["finish_ms"], burst["requests"][1]["finish_ms"])
            self.assertAlmostEqual(burst["requests"][0]["finish_ms"], 350.)

    def test_throughput_gain_without_more_timely_images_does_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            context = self.context(directory, clock)
            context.metadata["image_deadline_ms"] = 10000.
            with patch("ditserve.optimizations.sana_phases.time.perf_counter", clock):
                result = run_batch(context)
            self.assertFalse(result["local_qualified"])
            self.assertFalse(result["performance_passed"])
            self.assertEqual(result["selected_batch_size"], 1)

    def test_failed_compile_report_cannot_select_compiled(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            context = self.context(directory, clock,
                compile_report={"status": "failed", "local_qualified": True, "selected_backend": "compiled"})
            with patch("ditserve.optimizations.sana_phases.time.perf_counter", clock):
                result = run_batch(context)
            self.assertEqual(result["selected_backend"], "eager")

    def test_budget_stop_preserves_prior_completed_rows_and_no_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            context = self.context(directory, clock)
            def budget():
                if context.adapter.count >= 9:
                    raise BudgetStop("fixture cap")
            context.check_budget = budget
            with patch("ditserve.optimizations.sana_phases.time.perf_counter", clock):
                with self.assertRaises(BudgetStop):
                    run_compile(context)
            result = json.loads((Path(directory) / "sana-compile.json").read_text())
            self.assertTrue(result["budget_stopped"])
            self.assertFalse(result["local_qualified"])
            self.assertEqual(len(result["rows"]), 9)
            self.assertIsNotNone(result["active_unit"])

    def test_burst_cancel_keeps_all_arrived_members_and_completed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            context = self.context(directory, clock)
            def budget():
                if context.adapter.count >= 1:
                    raise BudgetStop("fixture cap")
            context.check_budget = budget
            items = [dict(input_id=str(i), prompt="a", seed=i) for i in range(5)]
            with patch("ditserve.optimizations.sana_phases.time.perf_counter", clock):
                with self.assertRaises(BudgetStop) as raised:
                    _burst(context, items, 2, 792.)
            rows = raised.exception.sana_partial_burst["requests"]
            self.assertEqual(len(rows), 5)
            self.assertEqual(sum(row["status"] == "completed" for row in rows), 2)
            self.assertEqual(sum(row["status"] == "interrupted" for row in rows), 3)

    def test_guard_admits_cold_compile_with_explicit_600_second_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            context = self.context(directory, clock)
            class Guard:
                def __init__(self):
                    self.units = []
                def __call__(self):
                    pass
                def ensure_unit(self, seconds):
                    self.units.append(seconds)
            context.check_budget = guard = Guard()
            with patch("ditserve.optimizations.sana_phases.time.perf_counter", clock):
                run_compile(context)
            self.assertEqual(guard.units.count(600.), 1)
            self.assertTrue(all(unit in (60., 600.) for unit in guard.units))


if __name__ == "__main__":
    unittest.main()
