import json
from pathlib import Path
import tempfile
import unittest

from ditserve.research.common import atomic_json, read_json
from ditserve.sessions.budget import Allocation, AccountingStop, WatchdogStop, DeadlineCheck


class SessionAccountingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path, self.history = self.root / "new/gpu-budget.json", self.root / "old/gpu-budget.json"
        atomic_json(self.history, dict(used_gpu_seconds=43199.0, limit_gpu_seconds=43200.0))
        self.next_history = self.root / "next/gpu-budget.json"
        atomic_json(self.next_history, dict(used_gpu_seconds=200000., limit_gpu_seconds=None, active={"session_id": "old-active"}))
        self.mono, self.wall = 1.0, 100.0

    def account(self, path=None):
        return Allocation(path or self.path, historical_paths=(self.history, self.next_history),
                          clock=lambda: self.wall, monotonic=lambda: self.mono)

    def test_unlimited_charges_beyond_old_limit_without_editing_old_ledger(self):
        old = self.history.read_bytes()
        next_old = self.next_history.read_bytes()
        for phase, elapsed in (("completion", 200000.0), ("granularity", 300000.0)):
            with self.account() as account:
                self.assertIsNone(account.remaining())
                account.begin(phase=phase, identity={})
                self.mono += elapsed
                account.finish("failed")
        data = read_json(self.path)
        self.assertIsNone(data["limit_gpu_seconds"])
        self.assertEqual(data["used_gpu_seconds"], 500000.0)
        self.assertEqual(len(data["sessions"]), 2)
        self.assertFalse(data["historical_sources"][0]["inherited_budget"])
        self.assertEqual(self.history.read_bytes(), old)
        self.assertEqual(self.next_history.read_bytes(), next_old)
        self.assertEqual(data["historical_sources"][1]["active_session_id"], "old-active")
        self.assertNotIn("Infinity", self.path.read_text())

    def test_old_ledger_is_never_a_write_target(self):
        original = self.history.read_bytes()
        with self.assertRaisesRegex(ValueError, "read-only"):
            self.account(self.history)
        self.assertEqual(self.history.read_bytes(), original)

    def test_active_or_concurrent_session_cannot_be_replaced(self):
        with self.account() as account:
            with self.assertRaises(BlockingIOError):
                with self.account():
                    pass
            account.begin(phase="completion", identity={})
            with self.assertRaises(AccountingStop):
                account.begin(phase="granularity", identity={})
        with self.assertRaises(AccountingStop):
            with self.account():
                pass
        self.assertIsNotNone(read_json(self.path)["active"])

    def test_ledger_limit_and_charge_history_are_validated(self):
        with self.account():
            pass
        original = read_json(self.path)
        for changes in ({"limit_gpu_seconds": 43200.0}, {"authorization": "old"}, {"used_gpu_seconds": 1.0}):
            with self.subTest(changes=changes):
                atomic_json(self.path, dict(original, **changes))
                with self.assertRaises(ValueError):
                    with self.account():
                        pass

    def test_monotonic_time_counts_loading_and_cleanup_despite_wall_clock_changes(self):
        with self.account() as account:
            account.begin(phase="completion", identity={})
            self.mono += 40
            account.child_started(1234)
            self.mono += 100
            self.wall -= 500
            self.mono += 20
            self.assertEqual(account.finish("failed")["gpu_seconds"], 160)

    def test_backward_monotonic_clock_keeps_active_session(self):
        with self.account() as account:
            account.begin(phase="completion", identity={})
            self.mono -= 2
            with self.assertRaises(AccountingStop):
                account.finish("completed")
            self.assertIsNotNone(read_json(self.path)["active"])

    def test_deadline_is_a_phase_watchdog_not_cumulative_limit(self):
        check = DeadlineCheck(100.0, monotonic=lambda: self.mono, parent_pid=50, getppid=lambda: 50)
        self.assertEqual(check.remaining_seconds(), 99.0)
        check.require_seconds(99.0)
        with self.assertRaisesRegex(WatchdogStop, "watchdog"):
            check.require_seconds(100)
        self.mono = 101
        with self.assertRaisesRegex(WatchdogStop, "unlimited"):
            check()
        orphan = DeadlineCheck(200, monotonic=lambda: self.mono, parent_pid=50, getppid=lambda: 1)
        with self.assertRaisesRegex(WatchdogStop, "orphaned"):
            orphan()


if __name__ == "__main__":
    unittest.main()
