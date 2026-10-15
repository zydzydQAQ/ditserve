import json
import os
from pathlib import Path
import tempfile
import unittest

from ditserve.directions.budget import Allocation, BudgetStop, TOTAL


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.parent = self.root / "historical.json"
        self.parent.write_text(json.dumps({"used_gpu_seconds": 17342., "limit_gpu_seconds": 28800.}))
        self.ledger = self.root / "new.json"
        self.now = 100.
        self.elapsed = 50.

    def tearDown(self):
        self.directory.cleanup()

    def allocation(self):
        return Allocation(self.ledger, self.parent, clock=lambda: self.now,
                          monotonic=lambda: self.elapsed)

    def test_additional_budget_preserves_parent_and_counts_all_holding_time(self):
        old = self.parent.read_bytes()
        with self.allocation() as b:
            self.assertEqual(b.remaining("direction1"), 28800.)
            b.begin("direction1", gpu_count=2, identity={"phase": "correctness"})
            b.child_started(12345)
            self.now += 12.5
            self.elapsed += 12.5
            record = b.finish("failed")
            self.assertEqual(record["gpu_seconds"], 25.)
            self.assertEqual(b.data["used_gpu_seconds"], 25.)
            self.assertEqual(b.data["parent"]["used_gpu_seconds"], 17342.)
        self.assertEqual(self.parent.read_bytes(), old)

    def test_stage_cannot_borrow_another_stages_or_reserve_budget(self):
        with self.allocation() as b:
            b.data["stage_used_gpu_seconds"]["direction2"] = 21599.
            b.data["used_gpu_seconds"] = 21599.
            b.save()
            with self.assertRaises(BudgetStop):
                b.begin("direction2", gpu_count=1, identity={})
            self.assertEqual(b.remaining("reserve"), 7200.)

    def test_unsettled_session_is_not_erased_by_restart(self):
        with self.allocation() as b:
            b.begin("direction1", gpu_count=1, identity={})
        with self.assertRaises(BudgetStop):
            with self.allocation():
                pass
        self.assertIsNotNone(json.loads(self.ledger.read_text())["active"])

    def test_concurrent_supervisor_cannot_enter(self):
        with self.allocation():
            with self.assertRaises(BlockingIOError):
                with self.allocation():
                    pass

    def test_wall_clock_corrections_do_not_change_elapsed_charge(self):
        with self.allocation() as b:
            b.begin("direction1", gpu_count=1, identity={})
            self.now -= 1
            self.elapsed += 12.5
            first = b.finish("completed")
            self.assertEqual(first["gpu_seconds"], 12.5)
            b.begin("direction1", gpu_count=1, identity={})
            self.now += 10000
            self.elapsed += 3
            self.assertEqual(b.finish("completed")["gpu_seconds"], 3.)

    def test_monotonic_reversal_preserves_unsettled_session(self):
        with self.allocation() as b:
            b.begin("direction1", gpu_count=1, identity={})
            self.elapsed -= 1
            with self.assertRaises(BudgetStop):
                b.finish("completed")
            self.assertIsNotNone(b.data["active"])

    def test_admissions_have_unique_identity_and_owning_supervisor(self):
        with self.allocation() as b:
            b.begin("direction1", gpu_count=1, identity={})
            first = b.data["active"].copy()
            self.assertEqual(first["supervisor_pid"], os.getpid())
            b.finish("completed")
            b.begin("direction1", gpu_count=1, identity={})
            self.assertNotEqual(first["session_id"], b.data["active"]["session_id"])
            b.finish("completed")

    def test_inflight_session_cannot_be_replaced(self):
        with self.allocation() as b:
            b.begin("direction1", gpu_count=1, identity={})
            with self.assertRaises(BudgetStop):
                b.begin("direction2", gpu_count=1, identity={})
            b.finish("completed")


if __name__ == "__main__":
    unittest.main()
