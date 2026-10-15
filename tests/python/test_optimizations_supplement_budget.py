"""Standard-library-only budget fixtures; no real authorization ledger or GPU."""

import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from ditserve.optimizations.budget import BudgetGuard as ExistingGuard
from ditserve.optimizations.supplement_budget import (
    Allocation, BudgetGuard, BudgetStop, SupplementAllocation)
from ditserve.research.common import atomic_json, file_digest, read_json


class SupplementBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = self.root / "optimization.json"
        self.research = self.root / "research.json"
        self.path = self.root / "supplement/gpu-budget.json"
        for path, used, limit in ((self.parent, 22000., 28800.), (self.research, 40000., 86400.)):
            atomic_json(path, dict(active=None, used_gpu_seconds=used, limit_gpu_seconds=limit))
        self.now = [100.]
        self.wall = [1000.]
        self.gpu = "GPU-fixture-a100"
        self.identity = dict(phase="cache_k4", output=str(self.root / "attempt00"),
            config_sha256="a" * 64, source_sha256="b" * 64,
            gpu_uuids=[self.gpu], auxiliary_encoder=False, retry_from=None)

    def tearDown(self):
        self.temp.cleanup()

    def allocation(self, **overrides):
        args = dict(authorized_gpu_seconds=600., authorization="fixture only: explicit 600 GPU seconds",
            gpu_uuid=self.gpu, monotonic=lambda: self.now[0], clock=lambda: self.wall[0])
        args.update(overrides)
        return SupplementAllocation(self.path, self.parent, self.research, **args)

    def begin(self, budget, **overrides):
        args = dict(gpu_count=1, identity=copy.deepcopy(self.identity))
        args.update(overrides)
        budget.begin("cache_k4", **args)

    def failure(self, budget, elapsed=50.):
        self.begin(budget)
        budget.child_started(987654)
        self.now[0] += elapsed
        charge = budget.finish("failed")
        path = self.root / ("supervisor-" + charge["session_id"] + ".json")
        atomic_json(path, dict(phase="cache_k4", status="incomplete", charge=charge,
            config_sha256=self.identity["config_sha256"],
            final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[])))
        return dict(path=str(path), sha256=file_digest(path)), charge

    def test_authorization_is_required_finite_positive_and_never_created_by_import_or_init(self):
        self.assertIs(Allocation, SupplementAllocation)
        self.assertIs(BudgetGuard, ExistingGuard)
        with self.assertRaises(TypeError):
            SupplementAllocation(self.path, self.parent, self.research,
                authorization="no amount", gpu_uuid=self.gpu)
        for value in (None, True, 0, -1, float("nan"), float("inf"), "600"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.allocation(authorized_gpu_seconds=value)
        for value in (None, "", "   "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.allocation(authorization=value)
        self.allocation()
        self.assertFalse(self.path.exists())

    def test_two_immutable_parents_no_borrow_and_complete_lifetime_charged(self):
        original = {path: path.read_bytes() for path in (self.parent, self.research)}
        with self.allocation() as budget:
            self.assertEqual(budget.remaining("cache_k4"), 600.)
            self.assertEqual(budget.data["parents"]["optimization"]["sha256"], file_digest(self.parent))
            self.assertEqual(budget.data["parents"]["research"]["sha256"], file_digest(self.research))
            self.begin(budget)
            self.now[0] += 10.  # Loading before child registration is still charged.
            budget.child_started(987654)
            self.now[0] += 20.  # Warmup and failed work.
            self.assertEqual(budget.remaining("cache_k4"), 570.)
            self.now[0] += 7.  # Cleanup is inside the allocation.
            charge = budget.finish("failed")
            self.assertEqual(charge["gpu_seconds"], 37.)
            self.assertEqual(charge["child_pgid"], 987654)
            self.assertEqual(charge["supervisor_pid"], os.getpid())
            self.assertEqual(charge["supervisor_pgid"], os.getpgrp())
            self.assertEqual(budget.remaining("cache_k4"), 563.)
        for path, data in original.items():
            self.assertEqual(path.read_bytes(), data)

    def test_unlimited_requires_explicit_boolean_and_null_amount(self):
        for flag in (None, 1, "true"):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self.allocation(budget_unlimited=flag, authorized_gpu_seconds=None)
        for amount in (600., float("inf"), 0):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                self.allocation(budget_unlimited=True, authorized_gpu_seconds=amount)
        with self.assertRaises(ValueError):
            self.allocation(budget_unlimited=False, authorized_gpu_seconds=None)
        self.assertFalse(self.path.exists())

    def test_unlimited_serializes_null_but_keeps_finite_complete_cost_and_retry_charge(self):
        original = {p: p.read_bytes() for p in (self.parent, self.research)}
        args = dict(budget_unlimited=True, authorized_gpu_seconds=None)
        with self.allocation(**args) as budget:
            self.assertEqual(budget.remaining("cache_k4"), math.inf)
            retry, charge = self.failure(budget, elapsed=90000.)
            self.assertEqual(charge["gpu_seconds"], 90000.)
            self.assertEqual(charge["authorized_overrun_gpu_seconds"], 0.)
            self.assertEqual(budget.remaining("cache_k4"), math.inf)
        with self.allocation(**args) as budget:
            self.begin(budget, identity=dict(self.identity, retry_from=retry))
            self.now[0] += 80000.
            self.assertEqual(budget.remaining("cache_k4"), math.inf)
            budget.finish("completed")
        data = read_json(self.path)
        self.assertTrue(data["budget_unlimited"])
        self.assertIsNone(data["limit_gpu_seconds"])
        self.assertEqual(data["stage_limits_gpu_seconds"], {"cache_k4": None})
        self.assertEqual(data["used_gpu_seconds"], 170000.)
        self.assertNotIn("Infinity", self.path.read_text())
        self.assertNotIn("NaN", self.path.read_text())
        with self.assertRaises(ValueError), self.allocation():
            pass
        for p, expected in original.items():
            self.assertEqual(p.read_bytes(), expected)

    def test_legacy_finite_ledger_without_unlimited_flag_remains_compatible(self):
        with self.allocation():
            pass
        data = read_json(self.path)
        data.pop("budget_unlimited")
        atomic_json(self.path, data)
        with self.allocation() as budget:
            self.assertEqual(budget.remaining("cache_k4"), 600.)
        with self.assertRaises(ValueError), self.allocation(budget_unlimited=True, authorized_gpu_seconds=None):
            pass

    def test_only_one_bound_gpu_and_cache_stage_are_admitted(self):
        with self.allocation() as budget:
            for count in (True, 0, 2, 1.):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    self.begin(budget, gpu_count=count)
            for stage in ("reserve", "sana_compile", "direction2"):
                with self.subTest(stage=stage), self.assertRaises(ValueError):
                    budget.begin(stage, gpu_count=1, identity=self.identity)
            for change in (dict(gpu_uuids=["GPU-other"]), dict(gpu_uuids=[self.gpu, "GPU-v100"]),
                           dict(auxiliary_encoder=True), dict(phase="wan_compile"), dict(config_sha256="short")):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.begin(budget, identity=dict(self.identity, **change))
            self.assertIsNone(budget.data["active"])

    def test_atomic_lock_and_unsettled_session_refuse_reentry(self):
        with self.allocation() as budget:
            with self.assertRaises(BlockingIOError), self.allocation():
                pass
            self.begin(budget)
            with self.assertRaises(BudgetStop):
                self.begin(budget)
        with self.assertRaises(BudgetStop), self.allocation():
            pass
        self.assertIsNotNone(read_json(self.path)["active"])

    def test_explicit_same_config_failed_retry_is_separately_charged(self):
        with self.allocation() as budget:
            retry, first = self.failure(budget)
        with self.allocation() as budget:
            identity = dict(self.identity, retry_from=retry, source_sha256="c" * 64,
                            output=str(self.root / "attempt01"))
            self.begin(budget, identity=identity)
            self.now[0] += 25.
            second = budget.finish("completed")
            self.assertNotEqual(first["session_id"], second["session_id"])
            self.assertEqual(budget.data["used_gpu_seconds"], 75.)
            self.assertEqual(budget.remaining("cache_k4"), 525.)
        with self.allocation() as budget:
            with self.assertRaises(ValueError):
                self.begin(budget, identity=identity)

    def test_retry_requires_explicit_latest_failure_hash_config_charge_and_cleanup(self):
        with self.allocation() as budget:
            retry, charge = self.failure(budget)
            with self.assertRaises(ValueError):
                self.begin(budget)
            with self.assertRaises(ValueError):
                self.begin(budget, identity=dict(self.identity, retry_from=retry, config_sha256="d" * 64))
            with self.assertRaises(ValueError):
                self.begin(budget, identity=dict(self.identity, retry_from=dict(retry, sha256="0" * 64)))
            path = Path(retry["path"])
            original = read_json(path)
            variants = [dict(original, status="passed"), dict(original, phase="vae"),
                dict(original, config_sha256="d" * 64),
                dict(original, charge=dict(charge, session_id="older-session")),
                dict(original, final_occupancy=dict(status="exclusive_observed", owned=[1], foreign=[])),
                dict(original, final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[2])),
                dict(original, final_occupancy=dict(status="exclusive_observed", foreign=[])),
                dict(original, final_occupancy=dict(status="exclusive_observed", owned=[], foreign=None)),
                dict(original, final_occupancy=dict(status="unknown"))]
            for variant in variants:
                atomic_json(path, variant)
                binding = dict(path=str(path), sha256=file_digest(path))
                with self.subTest(variant=variant), self.assertRaises(ValueError):
                    self.begin(budget, identity=dict(self.identity, retry_from=binding))
            self.assertIsNone(budget.data["active"])

    def test_first_session_cannot_import_retry_from_old_budget(self):
        with self.allocation() as budget, self.assertRaises(ValueError):
            self.begin(budget, identity=dict(self.identity, retry_from=dict(path="old", sha256="a" * 64)))

    def test_minimum_validation_and_exhaustion_never_borrow_old_credit(self):
        with self.allocation(authorized_gpu_seconds=119.) as budget:
            with self.assertRaises(BudgetStop):
                self.begin(budget)
            for minimum in (0, -1, True, float("nan")):
                with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                    self.begin(budget, minimum_seconds=minimum)
            self.begin(budget, minimum_seconds=100.)
            self.now[0] += 125.
            self.assertEqual(budget.remaining("cache_k4"), 0.)
            charge = budget.finish("budget_stopped")
            self.assertEqual(charge["gpu_seconds"], 125.)
            self.assertEqual(charge["authorized_overrun_gpu_seconds"], 6.)
            self.assertEqual(budget.data["used_gpu_seconds"], 125.)

    def test_auth_limit_device_and_parent_hash_cannot_change_on_reopen(self):
        with self.allocation():
            pass
        for change in (dict(authorized_gpu_seconds=601.), dict(authorization="different approval"),
                       dict(gpu_uuid="GPU-another-a100")):
            with self.subTest(change=change), self.assertRaises(ValueError), self.allocation(**change):
                pass
        for parent in (self.parent, self.research):
            original = parent.read_bytes()
            value = read_json(parent)
            value["used_gpu_seconds"] += 1.
            atomic_json(parent, value)
            with self.subTest(parent=parent), self.assertRaises(ValueError), self.allocation():
                pass
            parent.write_bytes(original)

    def test_parent_active_blocks_creation_and_changed_parent_blocks_begin(self):
        for parent in (self.parent, self.research):
            original = parent.read_bytes()
            value = read_json(parent)
            value["active"] = dict(session_id="busy")
            atomic_json(parent, value)
            with self.subTest(parent=parent), self.assertRaises(BudgetStop), self.allocation():
                pass
            self.assertFalse(self.path.exists())
            parent.write_bytes(original)
        with self.allocation() as budget:
            value = read_json(self.parent)
            value["used_gpu_seconds"] += 1.
            atomic_json(self.parent, value)
            with self.assertRaises(ValueError):
                self.begin(budget)

    def test_parent_change_during_gpu_session_does_not_erase_charge(self):
        with self.allocation() as budget:
            self.begin(budget)
            value = read_json(self.research)
            value["used_gpu_seconds"] += 1.
            atomic_json(self.research, value)
            self.now[0] += 81.
            charge = budget.finish("completed")
            self.assertEqual(charge["status"], "failed")
            self.assertIn("parent_validation_error", charge)
            self.assertEqual(budget.data["used_gpu_seconds"], 81.)
            self.assertIsNone(budget.data["active"])

    def test_wall_clock_correction_does_not_change_charge_and_reversal_stays_active(self):
        with self.allocation() as budget:
            self.begin(budget)
            self.wall[0] -= 5000.
            self.now[0] += 20.
            self.assertEqual(budget.finish("failed")["gpu_seconds"], 20.)
        second = self.root / "another/gpu-budget.json"
        original = self.path
        self.path = second
        with self.allocation() as budget:
            self.begin(budget)
            self.now[0] -= 1.
            with self.assertRaises(BudgetStop):
                budget.finish("failed")
            self.assertIsNotNone(read_json(second)["active"])
        self.path = original

    def test_invalid_child_or_finish_cannot_clear_active_session(self):
        with self.allocation() as budget:
            with self.assertRaises(ValueError):
                budget.child_started(1)
            self.begin(budget)
            for pid in (True, 0, -1, 1.5):
                with self.subTest(pid=pid), self.assertRaises(ValueError):
                    budget.child_started(pid)
            budget.child_started(987654)
            with self.assertRaises(ValueError):
                budget.child_started(987655)
            with self.assertRaises(ValueError):
                budget.finish("passed")
            self.assertIsNotNone(budget.data["active"])

    def test_operations_require_lock_and_ledger_total_cannot_be_reset(self):
        budget = self.allocation()
        for operation in (lambda: budget.save(), lambda: budget.remaining("cache_k4"),
                          lambda: self.begin(budget), lambda: budget.child_started(1),
                          lambda: budget.finish("failed")):
            with self.assertRaises(RuntimeError):
                operation()
        with self.allocation() as budget:
            self.failure(budget)
        data = read_json(self.path)
        data["used_gpu_seconds"] = 0.
        data["stage_used_gpu_seconds"]["cache_k4"] = 0.
        atomic_json(self.path, data)
        with self.assertRaises(ValueError), self.allocation():
            pass

    def test_historical_path_alias_and_symlinks_are_rejected(self):
        with self.assertRaises(ValueError):
            SupplementAllocation(self.parent, self.parent, self.research,
                authorized_gpu_seconds=1., authorization="fixture", gpu_uuid=self.gpu)
        self.path.parent.mkdir()
        self.path.symlink_to(self.parent)
        with self.assertRaises(ValueError):
            self.allocation()
        self.path.unlink()
        os.link(self.parent, self.path)
        with self.assertRaises(ValueError):
            self.allocation()


if __name__ == "__main__":
    unittest.main()
