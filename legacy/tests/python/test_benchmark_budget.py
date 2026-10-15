import tempfile
import unittest
from pathlib import Path

from ditserve.benchmark.budget import BudgetExceeded, GpuBudget
from ditserve.benchmark.common import atomic_json, read_json, seal, verify_seal


class BudgetTests(unittest.TestCase):
    def test_failed_work_is_charged_across_sessions_and_budget_cannot_be_reset(self):
        now = [100.]
        clock = lambda: now[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            budget = GpuBudget(path, clock=clock, monotonic=clock)
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with budget.session():
                    now[0] += 120
                    raise RuntimeError("failure")
            self.assertEqual(read_json(path)["used_seconds"], 120)
            with budget.session():
                self.assertEqual(budget.check(), 7080)
                with self.assertRaises(BudgetExceeded):
                    budget.check(7080)
                now[0] += 30
            self.assertEqual(read_json(path)["used_seconds"], 150)
            with self.assertRaisesRegex(ValueError, "cannot be changed"):
                GpuBudget(path, hours=1).start()

    def test_exclusive_ledger_and_interrupted_session_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            first = GpuBudget(path, clock=lambda: 100, monotonic=lambda: 100)
            with first.session():
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    GpuBudget(path).start()
            ledger = read_json(path)
            ledger["active"] = {"session_id": "crashed", "started_at": 100, "pid": 0}
            atomic_json(path, ledger)
            recovered = GpuBudget(path, clock=lambda: 150, monotonic=lambda: 150)
            with recovered.session():
                self.assertEqual(recovered.check(), 7150)
            self.assertEqual(read_json(path)["sessions"][-2]["status"], "interrupted_conservatively_charged")

    def test_work_outside_session_and_over_authorized_limit_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            with self.assertRaises(RuntimeError):
                GpuBudget(path).check()
            with self.assertRaises(ValueError):
                GpuBudget(path, hours=3)

    def test_frozen_evidence_detects_tampering_and_nonfinite_values(self):
        value = seal({"status": "completed", "requests": [{"arrival_ms": 0.}]})
        verify_seal(value)
        value["requests"][0]["arrival_ms"] = 1.
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_seal(value)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                atomic_json(Path(directory) / "bad.json", {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
