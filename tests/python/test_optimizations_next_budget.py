"""Standard-library-only accounting fixtures; temporary fake ledgers, no GPU."""
import copy
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.optimizations.next_budget import (
    Allocation, BudgetStop, NextAllocation, CONTINUATION_AUTHORIZATION, PRIOR_UNLIMITED_AUTHORIZATION)
from ditserve.research.common import atomic_json, file_digest, read_json


class NextBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "new/gpu-budget.json"
        self.parents = [self.root / (name + ".json") for name in ("research", "optimization", "cache_online")]
        for path, used, limit in zip(self.parents, (40000., 22000., 8000.), (86400., 28800., None)):
            atomic_json(path, dict(active=None, used_gpu_seconds=used, limit_gpu_seconds=limit,
                                  budget_unlimited=limit is None))
        self.auth = self.root / "authorization.json"
        atomic_json(self.auth, dict(approved_by_user=True, budget_unlimited=True, authorized_gpu_seconds=None,
            scope="serving_next_validation", user_instruction=CONTINUATION_AUTHORIZATION + "；" + PRIOR_UNLIMITED_AUTHORIZATION))
        self.stages = ("cache_load", "encoder", "vae")
        self.now, self.wall = [100.], [1000.]
        self.gpu = "GPU-fixture-a100"

    def tearDown(self):
        self.temp.cleanup()

    def allocation(self, **overrides):
        args = dict(parents=self.parents, authorization_binding=dict(path=str(self.auth), sha256=file_digest(self.auth)),
            gpu_uuid=self.gpu, stages=self.stages, monotonic=lambda: self.now[0], clock=lambda: self.wall[0])
        args.update(overrides)
        return NextAllocation(self.path, **args)

    def identity(self, stage="cache_load", attempt=0, **changes):
        value = dict(phase=stage, config_sha256="a" * 64, source_sha256="b" * 64,
            output=str(self.root / ("run/" + stage + "/attempt%02d" % attempt)), run_root=str(self.root / "run"),
            gpu_uuids=[self.gpu], auxiliary_encoder=False, budget_unlimited=True, retry_from=None)
        value.update(changes)
        return value

    def begin(self, budget, stage="cache_load", **changes):
        budget.begin(stage, gpu_count=1, identity=self.identity(stage, **changes))

    def settled(self, budget, stage="cache_load", status="completed", elapsed=10., **changes):
        self.begin(budget, stage, **changes)
        budget.child_started(987654)
        self.now[0] += elapsed
        return budget.finish(status)

    def failed_retry(self, budget):
        charge = self.settled(budget, status="failed")
        path = self.root / "failure-supervisor.json"
        atomic_json(path, dict(status="incomplete", phase="cache_load", config_sha256="a" * 64,
            charge=charge, final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[])))
        return dict(path=str(path), sha256=file_digest(path)), charge

    def test_construction_is_read_only_and_requires_explicit_schema(self):
        self.assertIs(Allocation, NextAllocation)
        self.allocation()
        self.assertFalse(self.path.exists())
        for value in ((), ["cache_load"], ("cache_load", "cache_load"), ("../bad",)):
            with self.subTest(stages=value), self.assertRaises(ValueError):
                self.allocation(stages=value)
        for value in ([], None):
            with self.subTest(parents=value), self.assertRaises(ValueError):
                self.allocation(parents=value)
        with self.assertRaises(ValueError):
            self.allocation(authorization_binding=dict(path=str(self.auth), sha256="short"))

    def test_unlimited_parent_and_all_holding_cost_are_bound_without_old_writes(self):
        originals = {p: p.read_bytes() for p in self.parents}
        with self.allocation() as budget:
            self.assertEqual(budget.remaining("cache_load"), math.inf)
            latest = budget.data["parents"][str(self.parents[-1])]
            self.assertIsNone(latest["limit_gpu_seconds"])
            self.assertTrue(latest["budget_unlimited"])
            self.begin(budget)
            self.now[0] += 11.  # Charged setup before child registration.
            budget.child_started(987654)
            self.now[0] += 90000.  # No hard deadline or inherited credit limit.
            self.now[0] += 9.  # Cleanup remains inside charged lifetime.
            charge = budget.finish("failed")
            self.assertEqual(charge["gpu_seconds"], 90020.)
            self.assertEqual(charge["supervisor_pid"], os.getpid())
            self.assertEqual(charge["supervisor_pgid"], os.getpgrp())
            self.assertEqual(budget.remaining("cache_load"), math.inf)
            self.assertIsNone(budget.data["limit_gpu_seconds"])
        for p, original in originals.items():
            self.assertEqual(p.read_bytes(), original)
        self.assertNotIn("Infinity", self.path.read_text())
        self.assertNotIn("NaN", self.path.read_text())

    def test_new_stages_advance_with_changed_source_but_same_config(self):
        with self.allocation() as budget:
            first = self.settled(budget, elapsed=15.)
        with self.allocation() as budget:
            second = self.settled(budget, "encoder", elapsed=17., source_sha256="c" * 64)
            third = self.settled(budget, "vae", elapsed=19., source_sha256="d" * 64)
            self.assertEqual(budget.data["stage_used_gpu_seconds"], dict(cache_load=15., encoder=17., vae=19.))
            self.assertEqual(budget.data["used_gpu_seconds"], 51.)
            self.assertEqual(len({r["session_id"] for r in (first, second, third)}), 3)
        with self.allocation() as budget:
            for stage in self.stages:
                with self.subTest(stage=stage), self.assertRaises(ValueError):
                    self.begin(budget, stage)

    def test_untriggered_stage_may_be_skipped_but_never_revisited(self):
        with self.allocation() as budget:
            self.settled(budget, "encoder")
            with self.assertRaises(ValueError):
                self.begin(budget, "cache_load")
            self.settled(budget, "vae")

    def test_failed_phase_can_continue_to_next_phase_without_retry(self):
        with self.allocation() as budget:
            self.settled(budget, status="failed")
            self.settled(budget, "encoder")
            self.assertEqual(budget.data["used_gpu_seconds"], 20.)

    def test_explicit_same_config_retry_is_new_charged_attempt(self):
        with self.allocation() as budget:
            retry, first = self.failed_retry(budget)
            with self.assertRaises(ValueError):
                self.begin(budget)
        with self.allocation() as budget:
            second = self.settled(budget, attempt=1, retry_from=retry, source_sha256="c" * 64, elapsed=21.)
            self.assertEqual(budget.data["used_gpu_seconds"], 31.)
            self.assertNotEqual(first["session_id"], second["session_id"])
            with self.assertRaises(ValueError):
                self.begin(budget, attempt=2, retry_from=retry)

    def test_retry_rejects_old_charge_wrong_hash_config_phase_and_uncertain_cleanup(self):
        with self.allocation() as budget:
            retry, charge = self.failed_retry(budget)
            with self.assertRaises(ValueError):
                self.begin(budget, attempt=1, retry_from=dict(retry, sha256="0" * 64))
            with self.assertRaises(ValueError):
                self.begin(budget, attempt=1, retry_from=retry, config_sha256="e" * 64)
            with self.assertRaises(ValueError):
                self.begin(budget, retry_from=retry)  # Must preserve failed attempt output.
            path = Path(retry["path"]); original = read_json(path)
            variants = [dict(original, status="passed"), dict(original, phase="encoder"),
                dict(original, charge=dict(charge, session_id="older")),
                dict(original, final_occupancy=dict(status="exclusive_observed", foreign=[])),
                dict(original, final_occupancy=dict(status="exclusive_observed", owned=[], foreign=None)),
                dict(original, final_occupancy=dict(status="exclusive_observed", owned=[1], foreign=[])),
                dict(original, final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[1]))]
            for variant in variants:
                atomic_json(path, variant)
                binding = dict(path=str(path), sha256=file_digest(path))
                with self.subTest(variant=variant), self.assertRaises(ValueError):
                    self.begin(budget, attempt=1, retry_from=binding)
            self.assertIsNone(budget.data["active"])

    def test_new_stage_cannot_import_other_stage_or_historical_retry(self):
        retry = dict(path=str(self.root / "nonexistent.json"), sha256="0" * 64)
        with self.allocation() as budget:
            with self.assertRaises(ValueError):
                self.begin(budget, retry_from=retry)
            binding, _ = self.failed_retry(budget)
            with self.assertRaises(ValueError):
                self.begin(budget, "encoder", retry_from=binding)

    def test_lock_and_active_session_are_not_cleared_by_exit_or_reentry(self):
        with self.allocation() as budget:
            with self.assertRaises(BlockingIOError), self.allocation():
                pass
            self.begin(budget)
            with self.assertRaises(BudgetStop):
                self.begin(budget)
        before = self.path.read_bytes()
        with self.assertRaises(BudgetStop), self.allocation():
            pass
        self.assertEqual(before, self.path.read_bytes())
        self.assertIsNotNone(read_json(self.path)["active"])

    def test_each_parent_active_or_ambiguous_null_limit_blocks_admission(self):
        for path in self.parents:
            original = path.read_bytes()
            value = read_json(path); value["active"] = dict(session_id="foreign")
            atomic_json(path, value)
            with self.subTest(path=path), self.assertRaises(BudgetStop), self.allocation():
                pass
            self.assertFalse(self.path.exists()); path.write_bytes(original)
        latest = self.parents[-1]; value = read_json(latest); value.pop("budget_unlimited")
        atomic_json(latest, value)
        with self.assertRaises(ValueError), self.allocation():
            pass

    def test_parent_hash_change_before_admission_or_reopen_is_rejected(self):
        with self.allocation() as budget:
            path = self.parents[-1]; value = read_json(path); value["used_gpu_seconds"] += 1
            atomic_json(path, value)
            with self.assertRaises(ValueError):
                self.begin(budget)
        with self.assertRaises(ValueError), self.allocation():
            pass

    def test_parent_mutation_during_work_is_failed_but_fully_charged(self):
        with self.allocation() as budget:
            self.begin(budget); budget.child_started(987654)
            value = read_json(self.parents[-1]); value["used_gpu_seconds"] += 10
            atomic_json(self.parents[-1], value)
            self.now[0] += 27.
            charge = budget.finish("completed")
            self.assertEqual(charge["status"], "failed")
            self.assertIn("parent_validation_error", charge)
            self.assertEqual(budget.data["used_gpu_seconds"], 27.)
            self.assertIsNone(budget.data["active"])

    def test_missing_parent_during_work_preserves_full_failed_charge(self):
        with self.allocation() as budget:
            self.begin(budget); self.parents[-1].unlink(); self.now[0] += 33.
            charge = budget.finish("failed")
            self.assertEqual(charge["gpu_seconds"], 33.)
            self.assertIn("parent_validation_error", charge)

    def test_authorization_scope_text_flags_and_sha_are_required(self):
        original = read_json(self.auth)
        changes = [dict(scope="cache_online_factorial_only"), dict(approved_by_user=False),
            dict(budget_unlimited=False), dict(authorized_gpu_seconds=28800.),
            dict(user_instruction=CONTINUATION_AUTHORIZATION), dict(user_instruction=PRIOR_UNLIMITED_AUTHORIZATION)]
        for change in changes:
            atomic_json(self.auth, dict(original, **change))
            with self.subTest(change=change), self.assertRaises(ValueError), self.allocation():
                pass
            self.assertFalse(self.path.exists())
        atomic_json(self.auth, original)
        with self.assertRaises(ValueError), self.allocation(authorization_binding=dict(path=str(self.auth), sha256="0" * 64)):
            pass

    def test_authorization_mutation_is_not_a_free_session(self):
        with self.allocation() as budget:
            self.begin(budget); value = read_json(self.auth); value["note"] = "changed"
            atomic_json(self.auth, value); self.now[0] += 12.
            charge = budget.finish("failed")
            self.assertEqual(charge["gpu_seconds"], 12.)
            self.assertIn("authorization_validation_error", charge)

    def test_invalid_identity_gpu_source_config_output_and_minimum_never_start(self):
        with self.allocation() as budget:
            changes = [dict(phase="encoder"), dict(config_sha256="short"), dict(source_sha256="short"),
                dict(gpu_uuids=["GPU-other"]), dict(gpu_uuids=[self.gpu,"GPU-v100"]), dict(auxiliary_encoder=True),
                dict(budget_unlimited=False), dict(output="relative"), dict(run_root="relative"),
                dict(output=str(self.root / "outside")), dict(authorization_source={})]
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    budget.begin("cache_load", gpu_count=1, identity=self.identity(**change))
            for count in (True, 0, 2, 1.):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    budget.begin("cache_load", gpu_count=count, identity=self.identity())
            for minimum in (0, -1, True, float("nan"), float("inf")):
                with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                    budget.begin("cache_load", gpu_count=1, identity=self.identity(), minimum_seconds=minimum)
            self.assertIsNone(budget.data["active"])
            self.settled(budget)
            with self.assertRaises(ValueError):
                self.begin(budget, "encoder", config_sha256="c" * 64)

    def test_nonfinite_start_does_not_mutate_ledger_or_pin_config(self):
        with self.allocation() as budget:
            original = copy.deepcopy(budget.data)
            for value in (float("nan"), float("inf"), -1., True):
                self.now[0] = value
                with self.subTest(value=value), self.assertRaises(BudgetStop):
                    self.begin(budget)
                self.assertEqual(budget.data, original)

    def test_monotonic_charge_ignores_wall_correction_and_invalid_end_stays_active(self):
        with self.allocation() as budget:
            self.begin(budget); self.wall[0] -= 5000.; self.now[0] += 20.
            self.assertEqual(budget.finish("failed")["gpu_seconds"], 20.)
            self.begin(budget, "encoder"); start = self.now[0]
            for value in (float("nan"), float("inf"), start - 1.):
                self.now[0] = value
                with self.subTest(value=value), self.assertRaises(BudgetStop):
                    budget.finish("failed")
                self.assertIsNotNone(read_json(self.path)["active"])
            self.now[0] = start + 1.; self.wall[0] = float("nan")
            with self.assertRaises(BudgetStop):
                budget.finish("failed")

    def test_registered_pgid_required_for_success_and_never_overwritten(self):
        with self.allocation() as budget:
            with self.assertRaises(ValueError):
                budget.child_started(987654)
            self.begin(budget)
            with self.assertRaises(ValueError):
                budget.finish("completed")
            for pid in (True, 0, -1, 1.5, os.getpid(), os.getpgrp()):
                with self.subTest(pid=pid), self.assertRaises(ValueError):
                    budget.child_started(pid)
            budget.child_started(987654)
            with self.assertRaises(ValueError):
                budget.child_started(987655)
            with self.assertRaises(ValueError):
                budget.finish("passed")
            self.now[0] += 1.; budget.finish("completed")

    def test_settled_totals_and_phase_times_cannot_be_reset(self):
        with self.allocation() as budget:
            self.settled(budget)
        original = read_json(self.path)
        variants = []
        zero = copy.deepcopy(original); zero["used_gpu_seconds"] = 0.; zero["stage_used_gpu_seconds"]["cache_load"] = 0.
        variants.append(zero)
        wrong = copy.deepcopy(original); wrong["sessions"][0]["gpu_seconds"] = 1.; variants.append(wrong)
        wrong = copy.deepcopy(original); wrong["sessions"][0]["ended_monotonic"] += 1.; variants.append(wrong)
        wrong = copy.deepcopy(original); wrong["sessions"][0]["identity"]["config_sha256"] = "c" * 64; variants.append(wrong)
        for variant in variants:
            atomic_json(self.path, variant)
            with self.subTest(variant=variant), self.assertRaises(ValueError), self.allocation():
                pass

    def test_only_owning_supervisor_can_operate_under_lock(self):
        budget = self.allocation()
        with self.assertRaises(RuntimeError):
            budget.remaining("cache_load")
        with budget:
            with patch('ditserve.optimizations.next_budget.os.getpid', return_value=os.getpid() + 1):
                with self.assertRaises(RuntimeError):
                    self.begin(budget)

    def test_bound_paths_cannot_alias_or_traverse_symlinks(self):
        with self.assertRaises(ValueError):
            self.allocation(parents=self.parents + [self.parents[0]])
        self.path.parent.mkdir(); self.path.symlink_to(self.parents[0])
        with self.assertRaises(ValueError):
            self.allocation()
        self.path.unlink(); os.link(self.parents[0], self.path)
        with self.assertRaises(ValueError):
            self.allocation()

    def test_optional_auxiliary_uuid_requires_distinct_verified_identity(self):
        for value in (self.gpu, "", "GPU-", "V100", True, 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.allocation(auxiliary_gpu_uuid=value)
        self.assertFalse(self.path.exists())

    def test_conditional_two_devices_charge_entire_lifetime_and_reopen(self):
        self.stages = ("cache_load", "conditional")
        auxiliary = "GPU-fixture-v100"
        originals = {p: p.read_bytes() for p in self.parents}
        with self.allocation() as budget:
            first = self.settled(budget, elapsed=11.)
            header = {k: copy.deepcopy(v) for k, v in budget.data.items()
                      if k not in ("sessions", "active", "used_gpu_seconds", "stage_used_gpu_seconds")}
        with self.allocation(auxiliary_gpu_uuid=auxiliary) as budget:
            budget.begin("conditional", gpu_count=2,
                identity=self.identity("conditional", auxiliary_encoder=True, gpu_uuids=[self.gpu, auxiliary]))
            self.now[0] += 3.  # Both devices held during pre-child setup.
            budget.child_started(987654)
            self.now[0] += 13.
            self.now[0] += 7.  # Both remain charged through cleanup.
            row = budget.finish("completed")
            self.assertEqual(row["elapsed_seconds"], 23.)
            self.assertEqual(row["gpu_seconds"], 46.)
            self.assertEqual(row["gpu_count"], 2)
            self.assertEqual(budget.data["sessions"][0], first)
            self.assertEqual(budget.data["used_gpu_seconds"], 57.)
            self.assertEqual(budget.data["stage_used_gpu_seconds"], dict(cache_load=11., conditional=46.))
            self.assertEqual({k: budget.data[k] for k in header}, header)
        with self.allocation(auxiliary_gpu_uuid=auxiliary) as budget:
            self.assertEqual(budget.data["used_gpu_seconds"], 57.)
            self.assertEqual(budget.data["gpu_count"], 1)  # Immutable primary default header.
            self.assertEqual(budget.data["sessions"][-1]["gpu_count"], 2)
        for path, before in originals.items():
            self.assertEqual(path.read_bytes(), before)
        for kwargs in ({}, dict(auxiliary_gpu_uuid="GPU-other-v100")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError), self.allocation(**kwargs):
                pass

    def test_two_devices_rejected_outside_conditional_or_without_all_explicit_fields(self):
        self.stages = ("cache_load", "conditional")
        auxiliary = "GPU-fixture-v100"
        with self.allocation(auxiliary_gpu_uuid=auxiliary) as budget:
            variants = [
                ("cache_load", 2, dict(auxiliary_encoder=True, gpu_uuids=[self.gpu, auxiliary])),
                ("conditional", 2, dict(auxiliary_encoder=False, gpu_uuids=[self.gpu, auxiliary])),
                ("conditional", 2, dict(auxiliary_encoder=True, gpu_uuids=[auxiliary, self.gpu])),
                ("conditional", 2, dict(auxiliary_encoder=True, gpu_uuids=[self.gpu, "GPU-other"])),
                ("conditional", 2, dict(auxiliary_encoder=True, gpu_uuids=[self.gpu])),
                ("conditional", 1, dict(auxiliary_encoder=True, gpu_uuids=[self.gpu, auxiliary])),
                ("conditional", 2, {}),
                ("conditional", 2., dict(auxiliary_encoder=True, gpu_uuids=[self.gpu, auxiliary])),
                ("conditional", 2, dict(auxiliary_encoder=1, gpu_uuids=[self.gpu, auxiliary])),
            ]
            for stage, count, changes in variants:
                with self.subTest(stage=stage, count=count, changes=changes), self.assertRaises(ValueError):
                    budget.begin(stage, gpu_count=count, identity=self.identity(stage, **changes))
            self.assertIsNone(budget.data["active"])
        with self.allocation() as budget:
            with self.assertRaises(ValueError):
                budget.begin("conditional", gpu_count=2,
                    identity=self.identity("conditional", auxiliary_encoder=True, gpu_uuids=[self.gpu, auxiliary]))

    def test_conditional_without_available_auxiliary_keeps_original_single_card_scope(self):
        self.stages = ("cache_load", "conditional")
        with self.allocation(auxiliary_gpu_uuid="GPU-fixture-v100") as budget:
            row = self.settled(budget, "conditional", elapsed=19.)
            self.assertEqual(row["gpu_count"], 1)
            self.assertEqual(row["gpu_seconds"], 19.)
            self.assertEqual(row["identity"]["gpu_uuids"], [self.gpu])
        with self.allocation() as budget:
            self.assertEqual(budget.data["used_gpu_seconds"], 19.)

    def test_conditional_two_card_failure_retries_explicitly_and_preserves_all_charges(self):
        self.stages = ("cache_load", "conditional")
        auxiliary = "GPU-fixture-v100"
        devices = dict(auxiliary_encoder=True, gpu_uuids=[self.gpu, auxiliary])
        with self.allocation(auxiliary_gpu_uuid=auxiliary) as budget:
            budget.begin("conditional", gpu_count=2, identity=self.identity("conditional", **devices))
            self.now[0] += 17.
            charge = budget.finish("failed")  # Launch failure still charges both held devices.
            path = self.root / "conditional-failure.json"
            atomic_json(path, dict(status="incomplete", phase="conditional", config_sha256="a" * 64,
                charge=charge, final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[])))
            retry = dict(path=str(path), sha256=file_digest(path))
            with self.assertRaises(ValueError):
                budget.begin("conditional", gpu_count=2, identity=self.identity("conditional", attempt=1, **devices))
            budget.begin("conditional", gpu_count=2,
                identity=self.identity("conditional", attempt=1, retry_from=retry, **devices))
            budget.child_started(987654)
            self.now[0] += 11.
            second = budget.finish("completed")
            self.assertEqual(charge["gpu_seconds"], 34.)
            self.assertEqual(second["gpu_seconds"], 22.)
            self.assertEqual(budget.data["used_gpu_seconds"], 56.)
        with self.allocation(auxiliary_gpu_uuid=auxiliary) as budget:
            self.assertEqual(len(budget.data["sessions"]), 2)

    def test_two_card_parent_mutation_fails_without_losing_second_device_charge(self):
        self.stages = ("conditional",)
        auxiliary = "GPU-fixture-v100"
        with self.allocation(auxiliary_gpu_uuid=auxiliary) as budget:
            budget.begin("conditional", gpu_count=2,
                identity=self.identity("conditional", auxiliary_encoder=True, gpu_uuids=[self.gpu, auxiliary]))
            budget.child_started(987654)
            self.parents[-1].unlink()
            self.now[0] += 29.
            charge = budget.finish("completed")
            self.assertEqual(charge["status"], "failed")
            self.assertEqual(charge["gpu_seconds"], 58.)
            self.assertEqual(budget.data["used_gpu_seconds"], 58.)
            self.assertIn("parent_validation_error", charge)

    def test_two_card_undercharge_or_changed_identity_is_rejected_on_reopen(self):
        self.stages = ("conditional",)
        auxiliary = "GPU-fixture-v100"
        with self.allocation(auxiliary_gpu_uuid=auxiliary) as budget:
            budget.begin("conditional", gpu_count=2,
                identity=self.identity("conditional", auxiliary_encoder=True, gpu_uuids=[self.gpu, auxiliary]))
            budget.child_started(987654)
            self.now[0] += 10.
            budget.finish("completed")
        original = read_json(self.path)
        undercharged = copy.deepcopy(original)
        undercharged["sessions"][0]["gpu_seconds"] = 10.
        undercharged["used_gpu_seconds"] = 10.
        undercharged["stage_used_gpu_seconds"]["conditional"] = 10.
        identity = copy.deepcopy(original)
        identity["sessions"][0]["identity"]["auxiliary_encoder"] = False
        for value in (undercharged, identity):
            atomic_json(self.path, value)
            with self.subTest(value=value), self.assertRaises(ValueError), self.allocation(auxiliary_gpu_uuid=auxiliary):
                pass


if __name__ == "__main__":
    unittest.main()
