import json
from pathlib import Path
import tempfile
import unittest

from ditserve.scheduling.budget import Allocation, BudgetStop, DeadlineCheck, TOTAL


class SchedulingBudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.history = self.root / "historical.json"
        self.history.write_text(json.dumps(dict(used_gpu_seconds=39669.6, limit_gpu_seconds=86400.0)))
        self.path = self.root / "budget.json"
        self.wall, self.mono = 100.0, 50.0

    def allocation(self):
        return Allocation(self.path, clock=lambda: self.wall, monotonic=lambda: self.mono,
                          historical_path=self.history)

    def test_independent_allowance_records_but_never_inherits_or_edits_history(self):
        original = self.history.read_bytes()
        with self.allocation() as budget:
            self.assertEqual(budget.remaining(), TOTAL)
            self.assertEqual(budget.data["historical_source"]["used_gpu_seconds"], 39669.6)
            self.assertFalse(budget.data["historical_source"]["inherited_budget"])
            budget.begin(phase="calibrate", identity={"output": "run1"})
            self.mono += 17.5
            record = budget.finish("failed")
            self.assertEqual(record["gpu_seconds"], 17.5)
        with self.allocation() as budget:
            self.assertEqual(budget.remaining(), TOTAL - 17.5)
            budget.begin(phase="pilot", identity={"output": "run2"})
            self.mono += 2.5
            budget.finish("completed")
            self.assertEqual(budget.remaining(), TOTAL - 20.0)
        self.assertEqual(self.history.read_bytes(), original)

    def test_unsettled_session_and_concurrent_supervisor_cannot_be_overwritten(self):
        with self.allocation() as budget:
            with self.assertRaises(BlockingIOError):
                with self.allocation():
                    pass
            budget.begin(phase="calibrate", identity={})
            with self.assertRaises(BudgetStop):
                budget.begin(phase="pilot", identity={})
        with self.assertRaises(BudgetStop):
            with self.allocation():
                pass
        self.assertIsNotNone(json.loads(self.path.read_text())["active"])

    def test_wall_clock_correction_does_not_change_charge(self):
        with self.allocation() as budget:
            budget.begin(phase="pilot", identity={})
            self.wall -= 5000.0
            self.mono += 14.0
            self.assertEqual(budget.finish("completed")["gpu_seconds"], 14.0)

    def test_clock_reversal_keeps_unsettled_record(self):
        with self.allocation() as budget:
            budget.begin(phase="pilot", identity={})
            self.mono -= 1
            with self.assertRaises(BudgetStop):
                budget.finish("completed")
            self.assertIsNotNone(budget.data["active"])

    def test_exhaustion_is_global_across_phases_and_runs(self):
        with self.allocation() as budget:
            budget.begin(phase="pilot", identity={})
            self.mono += TOTAL - 100
            budget.finish("completed")
        with self.allocation() as budget:
            with self.assertRaises(BudgetStop):
                budget.begin(phase="validate", identity={"output": "different-run"})

    def test_ledger_history_and_limit_cannot_be_reset(self):
        with self.allocation():
            pass
        data = json.loads(self.path.read_text())
        data["limit_gpu_seconds"] *= 2
        self.path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            with self.allocation():
                pass
        data["limit_gpu_seconds"] = TOTAL
        data["used_gpu_seconds"] = 5
        self.path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            with self.allocation():
                pass

    def test_deadline_exposes_remaining_and_rejects_incomplete_group_admission(self):
        check = DeadlineCheck(100.0, monotonic=lambda: self.mono)
        self.assertEqual(check.remaining_seconds(), 50.0)
        check.require_seconds(50.0)
        with self.assertRaises(BudgetStop):
            check.require_seconds(50.1, "five-arm group")
        self.mono = 100
        with self.assertRaises(BudgetStop):
            check()

    def test_orphaned_worker_stops_before_model_work(self):
        check = DeadlineCheck(100.0, monotonic=lambda: 1, parent_pid=99, getppid=lambda: 1)
        with self.assertRaises(BudgetStop):
            check()


if __name__ == "__main__":
    unittest.main()
