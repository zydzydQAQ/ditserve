import tempfile
from pathlib import Path
import unittest

from ditserve.research.budget import BudgetStop, ResearchBudget
from ditserve.research.common import atomic_json, read_json


class ResearchBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ledger.json"
        self.now = 100.

    def budget(self, limit=28800.):
        return ResearchBudget(self.path, limit, clock=lambda: self.now, monotonic=lambda: self.now)

    def test_four_gpus_charge_lifecycle_and_failure(self):
        with self.assertRaisesRegex(RuntimeError, "failed generation"):
            with self.budget().session("sp", ["a", "b", "c", "d"], 60, 30):
                self.now += 100
                raise RuntimeError("failed generation")
        ledger = read_json(self.path)
        self.assertEqual(ledger["used_gpu_seconds"], 400)
        self.assertEqual(ledger["stage_used_gpu_seconds"]["sp"], 400)
        self.assertEqual(ledger["sessions"][0]["status"], "failed")
        self.assertIsNone(ledger["active"])

    def test_incomplete_pair_reservation_rejected_without_charge(self):
        with self.assertRaises(BudgetStop):
            with self.budget(100).session("probe", ["a"], 80, 30):
                self.fail("must not allocate")
        self.assertEqual(read_json(self.path)["used_gpu_seconds"], 0)

    def test_stage_cap_reserves_sp_budget(self):
        with self.budget().session("correctness", ["a"], 100, 30):
            self.now += 1700
        with self.assertRaises(BudgetStop):
            with self.budget().session("correctness", ["a"], 100, 30):
                self.fail("stage exhausted")
        with self.budget().session("sp", ["a", "b", "c", "d"], 100, 30):
            self.now += 5
        self.assertEqual(read_json(self.path)["used_gpu_seconds"], 1720)

    def test_worker_stop_outcome_survives_normal_supervisor_return(self):
        with self.budget().session("correctness", ["a"], 100, 30) as session:
            self.now += 12
            session.set_outcome("budget_stopped")
        self.assertEqual(read_json(self.path)["sessions"][0]["status"], "budget_stopped")

    def test_active_reservation_checks_gpu_count(self):
        with self.budget(500).session("sp", ["a", "b", "c", "d"], 50, 10) as budget:
            self.now += 100
            with self.assertRaises(BudgetStop):
                budget.check(20, 10)

    def test_lock_excludes_competing_runs_and_limits_are_immutable(self):
        with self.budget().session("probe", ["a"], 50):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                self.budget().initialize()
        with self.assertRaisesRegex(ValueError, "cannot be changed"):
            self.budget(1000).initialize()
        with self.assertRaises(ValueError):
            self.budget(28801)

    def test_crash_is_conservatively_charged_and_cannot_reset(self):
        self.budget().initialize()
        ledger = read_json(self.path)
        ledger["active"] = dict(stage="sp", gpu_count=4, started_at=50, devices=["a", "b", "c", "d"])
        atomic_json(self.path, ledger)
        recovered = self.budget().initialize()
        self.assertEqual(recovered["used_gpu_seconds"], 200)
        self.assertEqual(recovered["stage_used_gpu_seconds"]["sp"], 200)
        self.assertIsNone(recovered["active"])
        self.assertEqual(self.budget().initialize()["used_gpu_seconds"], 200)


if __name__ == "__main__":
    unittest.main()
