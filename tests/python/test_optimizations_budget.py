import json
import tempfile
import unittest
from pathlib import Path

from ditserve.optimizations.budget import Allocation, BudgetStop
from ditserve.optimizations.protocol import TOTAL, LIMITS


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = self.root / "parent.json"
        self.parent.write_text(json.dumps(dict(active=None, used_gpu_seconds=39669., limit_gpu_seconds=86400.)))
        self.path = self.root / "gpu-budget.json"
        self.now = [100.]

    def tearDown(self):
        self.temp.cleanup()

    def allocation(self):
        return Allocation(self.path, self.parent, monotonic=lambda: self.now[0], clock=lambda: self.now[0])

    def test_parent_immutable_and_full_process_lifetime_charged(self):
        original = self.parent.read_bytes()
        with self.allocation() as budget:
            budget.begin("sana_compile", gpu_count=1, identity={})
            budget.child_started(1001)
            self.now[0] += 50
            record = budget.finish("failed")
            self.assertEqual(record["gpu_seconds"], 50)
            self.assertEqual(budget.remaining("sana_compile"), 3550)
        self.assertEqual(self.parent.read_bytes(), original)

    def test_carry_only_forward_no_future_or_reserve_borrow(self):
        with self.allocation() as budget:
            self.assertEqual(budget.remaining("sana_compile"), 3600)
            budget.data["stage_used_gpu_seconds"]["sana_compile"] = 1200
            budget.data["used_gpu_seconds"] = 1200
            self.assertEqual(budget.remaining("sana_batch"), 7800)
            self.assertEqual(budget.remaining("reserve"), 3600)
            budget.data["stage_used_gpu_seconds"]["sana_batch"] = 7800
            budget.data["used_gpu_seconds"] = 9000
            self.assertEqual(budget.remaining("sana_compile"), 0)
            self.assertEqual(budget.remaining("sana_batch"), 0)
            self.assertEqual(budget.remaining("wan_compile"), 7200)

    def test_two_gpu_seconds(self):
        with self.allocation() as budget:
            budget.begin("text_encoder", gpu_count=2, identity={})
            self.now[0] += 17
            self.assertEqual(budget.finish("completed")["gpu_seconds"], 34)

    def test_unsettled_session_blocks_reentry(self):
        with self.allocation() as budget:
            budget.begin("sana_compile", gpu_count=1, identity={})
        with self.assertRaises(BudgetStop), self.allocation():
            pass

    def test_parent_change_blocks_reentry(self):
        with self.allocation():
            pass
        self.parent.write_text(json.dumps(dict(active=None, used_gpu_seconds=39670., limit_gpu_seconds=86400.)))
        with self.assertRaises(ValueError), self.allocation():
            pass

    def test_cleanup_minimum_and_total_cap(self):
        with self.allocation() as budget:
            budget.data["used_gpu_seconds"] = TOTAL - 119
            self.assertEqual(budget.remaining("reserve"), 119)
            with self.assertRaises(BudgetStop):
                budget.begin("reserve", gpu_count=1, identity={})
        self.assertEqual(sum(LIMITS.values()), TOTAL)


if __name__ == "__main__":
    unittest.main()
