import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ditserve.research.cfg_cost_plan import pending_cost_plan
from ditserve.research.cfg_protocol import DEFAULT_CONFIG
from ditserve.research.common import atomic_json, read_json
from ditserve.research.dual_evidence import UnitJournal
from ditserve.research.dual_matrix import seal
from test_research_cfg_matrix import cache_rows


class PendingBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frozen = dict(sha256="current", source_sha256="current-source", config=read_json(DEFAULT_CONFIG))
        self.original = seal(dict(status="fits", protocol_sha256="current", source_sha256="current-source",
            per_device={role: dict(physical_uuid="GPU-" + role, correctness_gpu_seconds=100.,
                                  measured_max_pair_wall_seconds=10., measured_setup_wall_seconds=5.)
                        for role in ("a100", "v100")}))
        self.ledger = dict(active=None, used_gpu_seconds=1000., limit_gpu_seconds=28800., sessions=[],
            stage_used_gpu_seconds={}, reallocations=[dict(baseline_stage_used={},
                stage_allowances_gpu_seconds=dict(correctness=1000., calibration=3000., capacity=900.))])
        self.patch = patch("ditserve.research.cfg_continuation.screening_decisions",
                           return_value={name: dict(eligible=True) for name in ("independent_004", "paired_008")})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.journal = UnitJournal(self.root, self.frozen, ["GPU-a100", "GPU-v100"],
                                   observe=lambda *a, **kw: dict(status="exclusive_observed"))

    def plan(self, ledger=None, original=None):
        return pending_cost_plan(self.frozen, self.root, ledger or self.ledger, original or self.original)

    def unit(self, name, records, wall=1.):
        with self.journal.unit(name, 1., time.monotonic() + 60):
            for record in records:
                self.journal.append(dict(status="completed", device_role="a100", **record)
                                    if "device_role" not in record else dict(status="completed", **record)
                                    if "status" not in record else record)
        path = self.root / "units" / (name + ".json")
        value = read_json(path)
        value["ended_at"] = value["started_at"] + wall
        value.pop("sha256")
        atomic_json(path, seal(value))

    def complete(self, phase, wall=1., trajectory_ms=1000.):
        kind, role = phase.rsplit("-", 1)
        if kind == "correctness":
            for recipe in ("off", "independent_008", "paired_008"):
                run = dict(elapsed_ms=trajectory_ms,
                           identity=dict(implementation="split_prefix" if recipe == "paired_008" else "native"))
                rows = [dict(type="cfg_correctness", case_id=recipe, device_role=role,
                    recipe_id=recipe, equal=True, continuous=run, restored=copy.deepcopy(run))]
                if recipe == "paired_008":
                    rows.append(dict(type="cfg_implementation_equivalence", case_id="equivalence", device_role=role,
                                     recipe_id=recipe, equal=True))
                self.unit(phase + "-correctness-" + role + "-" + recipe, rows, wall)
            self.unit(phase + "-diagnostic-neutrality-" + role,
                      [dict(type="cfg_diagnostic_neutrality", case_id="diagnostic", device_role=role, equal=True)], wall)
            summary = dict(checks_passed=True)
        else:
            split = "dev" if kind == "calibrate" else "holdout"
            recipes = ("independent_004", "paired_008") if split == "dev" else ("independent_004",)
            for recipe in recipes:
                for row in cache_rows(recipe, split):
                    if row["device_role"] == role:
                        self.unit(phase + "-" + row["case_id"], [row], wall)
            summary = dict(selected="independent_004") if split == "holdout" else dict(candidates=list(recipes))
        atomic_json(self.root / (phase + ".json"), seal(dict(status="completed", protocol_sha256="current",
            source_sha256="current-source", **summary)))
        atomic_json(self.root / (phase + "-attempt.json"),
                    seal(dict(status="completed", returncode=0, protocol_sha256="current")))

    def test_latest_balance_must_cover_whole_pending_matrix_even_when_one_stage_start_fits(self):
        baseline = self.plan()
        self.assertEqual(baseline["status"], "fits")
        self.ledger["stage_used_gpu_seconds"]["calibration"] = 2200.
        updated = self.plan()
        self.assertGreater(updated["stage_remaining_gpu_seconds"]["calibration"],
                           updated["phase_startup_wall_seconds"]["calibrate-a100"] + 30.)
        self.assertLess(updated["stage_remaining_gpu_seconds"]["calibration"],
                        updated["pending_stage_estimates_gpu_seconds"]["calibration"])
        self.assertEqual(updated["status"], "budget_stopped")
        self.assertEqual(updated["stage_remaining_gpu_seconds"]["capacity"], 900.)

    def test_small_complete_holdout_gets_measured_startup_instead_of_600_seconds(self):
        for kind in ("correctness", "calibrate"):
            for role in ("a100", "v100"):
                self.complete(kind + "-" + role)
        self.complete("holdout-a100")
        self.ledger["stage_used_gpu_seconds"]["calibration"] = 2850.
        plan = self.plan()
        self.assertEqual(plan["status"], "fits")
        self.assertEqual(plan["pending_stage_estimates_gpu_seconds"]["correctness"], 0.)
        self.assertAlmostEqual(plan["pending_stage_estimates_gpu_seconds"]["calibration"], 88.5)
        self.assertAlmostEqual(plan["phase_startup_wall_seconds"]["holdout-v100"], 58.5)
        self.assertLess(58.5 + 30., plan["stage_remaining_gpu_seconds"]["calibration"])
        self.assertLess(plan["stage_remaining_gpu_seconds"]["calibration"], 630.)

    def test_unsealed_or_partial_completion_cannot_remove_work(self):
        phase = "calibrate-a100"
        atomic_json(self.root / (phase + ".json"), seal(dict(status="completed", protocol_sha256="current",
                                                           source_sha256="current-source")))
        atomic_json(self.root / (phase + "-attempt.json"), seal(dict(status="completed", returncode=0,
                                                                   protocol_sha256="current")))
        self.assertNotIn(phase, self.plan()["completed_phases"])
        self.complete(phase)
        self.assertIn(phase, self.plan()["completed_phases"])
        path = next((self.root / "units").glob("calibrate-a100-*.json"))
        unit = read_json(path)
        unit["performance_valid"] = False
        unit.pop("sha256")
        atomic_json(path, seal(unit))
        self.assertNotIn(phase, self.plan()["completed_phases"])

    def test_slower_current_correctness_raises_pending_formal_cost_without_granting_qualification(self):
        before = self.plan()
        self.complete("correctness-a100", trajectory_ms=20000.)
        after = self.plan()
        self.assertEqual(after["measured_per_device"]["a100"]["pair_wall_seconds"], 40.)
        self.assertGreater(after["pending_stage_estimates_gpu_seconds"]["calibration"],
                           before["pending_stage_estimates_gpu_seconds"]["calibration"])
        self.assertNotIn("quality_qualified", after)
        self.assertNotIn("performance_qualified", after)

    def test_slower_current_formal_units_raise_holdout_forecast(self):
        self.complete("calibrate-a100", wall=25.)
        plan = self.plan()
        self.assertEqual(plan["measured_per_device"]["a100"]["pair_wall_seconds"], 25.)
        self.assertAlmostEqual(plan["phase_startup_wall_seconds"]["holdout-a100"], (4 * 25 + 5) * 1.3)

    def test_current_settled_loading_and_cleanup_overhead_is_reserved_again(self):
        self.complete("correctness-a100", wall=1.)
        atomic_json(self.root / "correctness-a100-job.json", dict(session_id="measured-session"))
        self.ledger["sessions"].append(dict(session_id="measured-session", protocol_sha256="current",
            stage="correctness", status="completed", devices=["GPU-a100"], wall_seconds=44.))
        plan = self.plan()
        self.assertEqual(plan["measured_per_device"]["a100"]["setup_wall_seconds"], 40.)
        self.assertAlmostEqual(plan["phase_startup_wall_seconds"]["holdout-a100"], (4 * 10 + 40) * 1.3)

    def test_foreign_plan_active_view_and_parent_summary_never_admit_current_work(self):
        with self.assertRaisesRegex(ValueError, "settled"):
            self.plan(ledger=dict(self.ledger, active=dict(stage="correctness")))
        value = dict(self.original, source_sha256="parent-source")
        value.pop("sha256")
        with self.assertRaisesRegex(ValueError, "current-source"):
            self.plan(original=seal(value))
        self.complete("correctness-a100")
        path = self.root / "correctness-a100.json"
        value = read_json(path)
        value["source_sha256"] = "parent-source"
        value.pop("sha256")
        atomic_json(path, seal(value))
        self.assertNotIn("correctness-a100", self.plan()["completed_phases"])


if __name__ == "__main__":
    unittest.main()
