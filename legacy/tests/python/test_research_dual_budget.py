import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.budget import BudgetStop, ResearchBudget, WorkerCleanupError
from ditserve.research.common import atomic_json, digest, read_json
from ditserve.research.dual_budget import DualResearchBudget, STAGE_SHARES


class DualBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "budget.json"
        self.now = 100.
        original = ResearchBudget(self.path, clock=lambda: self.now, monotonic=lambda: self.now)
        with original.session("probe", ["GPU-A"], 10, 1):
            self.now += 120
        with original.session("correctness", ["GPU-A"], 10, 1):
            self.now += 60
        self.original = read_json(self.path)

    def budget(self):
        return DualResearchBudget(self.path, clock=lambda: self.now, monotonic=lambda: self.now)

    def allocate(self):
        return self.budget().reallocate_remaining(protocol_sha256="frozen-dual")

    def test_reallocation_preserves_all_old_spending_and_apportions_only_remaining(self):
        allocation = self.allocate()
        current = read_json(self.path)
        self.assertEqual(current["used_gpu_seconds"], 180)
        self.assertEqual(current["sessions"], self.original["sessions"])
        self.assertEqual(allocation["remaining_gpu_seconds"], 28620)
        self.assertEqual(allocation["original_stage_limits"], self.original["stage_limits_gpu_seconds"])
        for stage, share in STAGE_SHARES.items():
            self.assertEqual(allocation["stage_allowances_gpu_seconds"][stage], 28620 * share)
            self.assertEqual(current["stage_used_gpu_seconds"][stage], self.original["stage_used_gpu_seconds"].get(stage, 0))
        self.assertAlmostEqual(sum(allocation["stage_allowances_gpu_seconds"].values()), 28620)
        self.assertEqual(len(current["reallocations"]), 1)
        # The old fixed-stage supervisor must no longer spend this allocation.
        with self.assertRaisesRegex(ValueError, "identity/limits"):
            ResearchBudget(self.path).initialize()

    def test_allocation_is_idempotent_and_cannot_reset_after_spending(self):
        first = self.allocate()
        with self.budget().session("migration", ["GPU-A", "GPU-B"], 5, 1):
            self.now += 20
        before = self.path.read_bytes()
        self.assertEqual(self.allocate(), first)
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "different frozen"):
            self.budget().reallocate_remaining(protocol_sha256="new-protocol")

    def test_protocol_revision_preserves_caps_history_and_remaining_allowances(self):
        allocation = self.allocate()
        with self.budget().session("correctness", ["GPU-A", "GPU-B"], 5, 1) as session:
            self.now += 20
            session.set_outcome("budget_stopped")
        before = read_json(self.path)
        revision = self.budget().bind_protocol_revision("frozen-dual", "fixed-fallback", "Fix shape admission order")
        after = read_json(self.path)
        self.assertEqual({k: v for k, v in after.items() if k != "protocol_revisions"}, before)
        self.assertEqual(after["reallocations"], [allocation])
        self.assertEqual(revision["used_gpu_seconds"], 220)
        self.assertEqual(revision["sessions_sha256"], digest(before["sessions"]))
        self.assertEqual(revision["previous_binding_sha256"], allocation["sha256"])
        self.assertEqual(revision["sha256"], digest({k: v for k, v in revision.items() if k != "sha256"}))
        sealed_bytes = self.path.read_bytes()
        self.assertEqual(self.budget().reallocate_remaining(protocol_sha256="fixed-fallback"), allocation)
        self.assertEqual(self.path.read_bytes(), sealed_bytes)
        with self.assertRaisesRegex(ValueError, "different frozen"):
            self.allocate()
        budget = self.budget()
        budget._open()
        try:
            self.assertEqual(budget.allocation_remaining()["correctness"],
                             allocation["stage_allowances_gpu_seconds"]["correctness"] - 40)
        finally:
            budget._close()

    def test_revision_sessions_use_latest_protocol_and_preserve_original_allocation_id(self):
        allocation = self.allocate()
        first = self.budget().bind_protocol_revision("frozen-dual", "revision-1", "First fix")
        with self.budget().session("reserve", ["GPU-A", "GPU-B"], 5, 1,
                                   protocol_sha256="revision-1"):
            active = read_json(self.path)["active"]
            self.assertEqual(active["protocol_sha256"], "revision-1")
            self.assertEqual(active["allocation_id"], allocation["allocation_id"])
            self.now += 10
        second = self.budget().bind_protocol_revision("revision-1", "revision-2", "Second fix")
        self.assertEqual(second["previous_binding_sha256"], first["sha256"])
        with self.budget().session("migration", ["GPU-A"], 1, 1):
            self.assertEqual(read_json(self.path)["active"]["protocol_sha256"], "revision-2")
            self.now += 1
        result = self.budget().initialize()
        self.assertEqual(result["used_gpu_seconds"], 201)
        self.assertEqual(result["sessions"][-2]["protocol_sha256"], "revision-1")
        self.assertEqual(result["sessions"][-1]["protocol_sha256"], "revision-2")
        self.assertEqual(result["reallocations"], [allocation])

    def test_repeating_identical_latest_revision_is_idempotent_even_after_spending(self):
        self.allocate()
        first = self.budget().bind_protocol_revision("frozen-dual", "revision-1", "First fix")
        with self.budget().session("reserve", ["GPU-A"], 1, 1, protocol_sha256="revision-1"):
            self.now += 3
        before = self.path.read_bytes()
        self.assertEqual(self.budget().bind_protocol_revision("frozen-dual", "revision-1", "First fix"), first)
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "latest frozen"):
            self.budget().bind_protocol_revision("frozen-dual", "revision-1", "Changed reason")
        self.assertEqual(self.path.read_bytes(), before)
        self.budget().bind_protocol_revision("revision-1", "revision-2", "Second fix")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "latest frozen"):
            self.budget().bind_protocol_revision("frozen-dual", "revision-1", "First fix")
        self.assertEqual(self.path.read_bytes(), before)

    def test_revision_rejects_active_reservation_without_recovering_or_charging(self):
        self.allocate()
        current = read_json(self.path)
        current["active"] = dict(allocation_id=current["reallocations"][0]["allocation_id"],
            protocol_sha256="frozen-dual", stage="correctness", devices=["GPU-A"], gpu_count=1,
            started_at=100, workers_expected=True, worker_pgid=99999)
        atomic_json(self.path, current)
        before = self.path.read_bytes()
        with patch("ditserve.research.dual_budget.process_group_alive", return_value=False), self.assertRaises(WorkerCleanupError):
            self.budget().bind_protocol_revision("frozen-dual", "new", "Fix")
        self.assertEqual(self.path.read_bytes(), before)

    def test_revision_uses_lifetime_lock_and_rejects_stale_session_binding(self):
        self.allocate()
        with self.budget().session("correctness", ["GPU-A"], 1, 1):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                self.budget().bind_protocol_revision("frozen-dual", "new", "Fix")
        self.budget().bind_protocol_revision("frozen-dual", "new", "Fix")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "latest frozen"):
            with self.budget().session("reserve", ["GPU-A"], 1, 1, protocol_sha256="frozen-dual"):
                self.fail("a stale caller must not start work")
        self.assertEqual(self.path.read_bytes(), before)

    def test_revision_rejects_wrong_previous_blank_reason_same_hash_and_rollback(self):
        self.allocate()
        before = self.path.read_bytes()
        for previous, new, reason in (("wrong", "new", "Fix"), ("frozen-dual", "new", " "),
                                      ("frozen-dual", "frozen-dual", "Fix"), ("frozen-dual", "", "Fix")):
            with self.subTest(previous=previous, new=new, reason=reason), self.assertRaises(ValueError):
                self.budget().bind_protocol_revision(previous, new, reason)
            self.assertEqual(self.path.read_bytes(), before)
        self.budget().bind_protocol_revision("frozen-dual", "new", "Fix")
        before = self.path.read_bytes()
        for previous, new in (("frozen-dual", "another"), ("new", "frozen-dual")):
            with self.subTest(previous=previous, new=new), self.assertRaises(ValueError):
                self.budget().bind_protocol_revision(previous, new, "Fix")
            self.assertEqual(self.path.read_bytes(), before)

    def test_revision_history_and_original_caps_fail_closed_on_tampering(self):
        self.allocate()
        self.budget().bind_protocol_revision("frozen-dual", "new", "Fix")
        original = read_json(self.path)
        for kind in ("seal", "chain", "history", "caps", "allocation-replacement"):
            current = copy.deepcopy(original)
            if kind == "seal":
                current["protocol_revisions"][0]["reason"] = "Unsealed edit"
            elif kind in ("chain", "history"):
                row = current["protocol_revisions"][0]
                row["previous_binding_sha256" if kind == "chain" else "sessions_sha256"] = "wrong"
                row["sha256"] = digest({k: v for k, v in row.items() if k != "sha256"})
            elif kind == "caps":
                current["stage_limits_gpu_seconds"]["reserve"] += 1
            else:
                current["reallocations"].append(copy.deepcopy(current["reallocations"][0]))
            atomic_json(self.path, current)
            before = self.path.read_bytes()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.budget().initialize()
            self.assertEqual(self.path.read_bytes(), before)

    def test_reallocation_rejects_active_reservation_even_if_processes_appear_dead(self):
        current = read_json(self.path)
        current["active"] = dict(stage="probe", devices=["GPU-A"], gpu_count=1, started_at=100, worker_pgid=99999)
        atomic_json(self.path, current)
        before = self.path.read_bytes()
        with patch("ditserve.research.dual_budget.process_group_alive", return_value=False), self.assertRaises(WorkerCleanupError):
            self.allocate()
        self.assertEqual(self.path.read_bytes(), before)

    def test_same_lifetime_lock_excludes_old_and_new_supervisors(self):
        original = ResearchBudget(self.path, clock=lambda: self.now, monotonic=lambda: self.now)
        with original.session("probe", ["GPU-A"], 5, 1):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                self.allocate()
        self.allocate()
        with self.budget().session("migration", ["GPU-A", "GPU-B"], 5, 1):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                self.budget().initialize()

    def test_two_gpu_session_counts_full_lifetime_and_cleanup(self):
        self.allocate()
        with self.budget().session("migration", ["GPU-A", "GPU-B"], 5, 1) as session:
            self.now += 11
            self.assertIsNotNone(read_json(self.path)["active"])
            self.now += 7  # cleanup remains inside this allocation
            session.set_outcome("blocked_contention")
        current = read_json(self.path)
        self.assertEqual(current["used_gpu_seconds"], 180 + 36)
        self.assertEqual(current["stage_used_gpu_seconds"]["migration"], 36)
        self.assertEqual(current["sessions"][-1]["status"], "blocked_contention")
        self.assertIsNone(current["active"])

    def test_stage_caps_cannot_borrow_and_guards_count_both_gpus(self):
        allocation = self.allocate()
        cap = allocation["stage_allowances_gpu_seconds"]["capacity"]
        with self.assertRaises(BudgetStop):
            with self.budget().session("capacity", ["GPU-A", "GPU-B"], cap / 2, 1):
                self.fail("unit should never start")
        self.assertEqual(read_json(self.path)["used_gpu_seconds"], 180)
        with self.budget().session("capacity", ["GPU-A", "GPU-B"], 1, 1) as session:
            self.now += cap / 2 - 3
            with self.assertRaises(BudgetStop):
                session.check(2, 1)
            self.assertAlmostEqual(session.check(), 3)
        self.assertLess(read_json(self.path)["used_gpu_seconds"], 28800)

    def test_failed_work_is_charged_and_bad_devices_never_start(self):
        self.allocate()
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with self.budget().session("migration", ["GPU-A", "GPU-B"], 2, 1):
                self.now += 10
                raise RuntimeError("failure")
        self.assertEqual(read_json(self.path)["sessions"][-1]["status"], "failed")
        self.assertEqual(read_json(self.path)["used_gpu_seconds"], 200)
        for devices in ([], ["0"], ["GPU-A", "GPU-A"], ["GPU-A", "GPU-B", "GPU-C"]):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                with self.budget().session("migration", devices, 1, 1):
                    pass

    def test_keyboard_interrupt_during_cleanup_retains_active_until_proven_exit(self):
        self.allocate()
        with patch("ditserve.research.dual_budget.process_group_alive", return_value=True), self.assertRaises(WorkerCleanupError):
            with self.budget().session("migration", ["GPU-A", "GPU-B"], 2, 1) as session:
                session.expect_workers()
                session.register_worker_group(123456)
                self.now += 10
                raise KeyboardInterrupt()
        current = read_json(self.path)
        self.assertIsNotNone(current["active"])
        self.assertEqual(current["used_gpu_seconds"], 180)
        self.now += 7
        with patch("ditserve.research.dual_budget.process_group_alive", return_value=True), self.assertRaises(WorkerCleanupError):
            self.budget().initialize()
        with patch("ditserve.research.dual_budget.process_group_alive", return_value=False):
            recovered = self.budget().initialize()
        self.assertEqual(recovered["used_gpu_seconds"], 214)
        self.assertEqual(recovered["sessions"][-1]["status"], "interrupted_conservatively_charged")
        self.assertIsNone(recovered["active"])

    def test_normal_return_with_live_rank_also_preserves_active(self):
        self.allocate()
        with patch("ditserve.research.dual_budget.process_group_alive", return_value=True), self.assertRaises(WorkerCleanupError):
            with self.budget().session("correctness", ["GPU-A"], 1, 1) as session:
                session.register_worker_group(123456)
        self.assertIsNotNone(read_json(self.path)["active"])

    def test_no_worker_spawn_failure_can_be_explicitly_closed_but_registered_group_cannot(self):
        self.allocate()
        with self.budget().session("correctness", ["GPU-A"], 1, 1) as session:
            session.expect_workers()
            session.confirm_no_workers_started()
            self.now += 2
            session.set_outcome("failed")
        self.assertEqual(read_json(self.path)["used_gpu_seconds"], 182)
        with patch("ditserve.research.dual_budget.process_group_alive", return_value=False):
            with self.budget().session("correctness", ["GPU-A"], 1, 1) as session:
                session.register_worker_group(123456)
                with self.assertRaises(WorkerCleanupError):
                    session.confirm_no_workers_started()

    def test_missing_ledger_corrupt_totals_and_no_allocation_cannot_reset_budget(self):
        with self.assertRaisesRegex(ValueError, "not been reallocated"):
            self.budget().initialize()
        current = read_json(self.path)
        current["used_gpu_seconds"] = 0
        atomic_json(self.path, current)
        with self.assertRaisesRegex(ValueError, "do not reconcile"):
            self.allocate()
        with self.assertRaisesRegex(ValueError, "existing research"):
            DualResearchBudget(self.path.parent / "new-ledger.json").reallocate_remaining(protocol_sha256="protocol")


if __name__ == "__main__":
    unittest.main()
