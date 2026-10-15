"""Algebraic cost-planning fixtures; numbers are not measured GPU performance."""

import unittest

from ditserve.benchmark.common import verify_seal
from ditserve.research.cost_planning import choose_cost_plan, work_counts


def probe(frames, generation=30., quality=3., transfer=1.):
    return dict(shape=dict(height=480, width=832, frames=frames, steps=32),
        measurement_kind="measured_gpu", status="completed", split="dev", source_sha256="a" * 64,
        protocol_sha256="b" * 64, generation_wall_seconds=generation,
        quality_wall_seconds=quality, transfer_wall_seconds=transfer,
        timing_scope=dict(encoding_included=True, archive_included=True))


class ResearchCostPlanningTests(unittest.TestCase):
    def test_conservative_work_counts_include_full_matrix_not_just_steps(self):
        one, two = work_counts(1), work_counts(2)
        self.assertEqual((one["generations"], one["quality_calls"], one["state_actions"]), (122, 66, 34))
        self.assertEqual((two["generations"], two["quality_calls"], two["state_actions"]), (168, 108, 56))
        self.assertEqual(two["stages"]["initial_calibration"]["generations"], 24)

    def test_chooses_largest_affordable_shape_first_not_smaller_two_candidate_plan(self):
        rows = [probe(17, 25), probe(33, 40), probe(81, 80)]
        result = choose_cost_plan(rows, remaining_calibrate_seconds=9000)
        verify_seal(result)
        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["choice"]["shape"]["frames"], 33)
        self.assertEqual(result["choice"]["candidate_count"], 1)
        self.assertEqual([row["frames"] for row in result["choice"]["paired_conditions"]], [17, 33])
        self.assertNotIn(81, [row["frames"] for row in result["choice"]["paired_conditions"]])

    def test_second_candidate_requires_same_shape_budget_and_can_be_disabled(self):
        rows = [probe(17, 25), probe(33, 30)]
        result = choose_cost_plan(rows, remaining_calibrate_seconds=9000)
        self.assertEqual(result["choice"]["candidate_count"], 2)
        result = choose_cost_plan(rows, remaining_calibrate_seconds=9000, max_candidates=1)
        self.assertEqual(result["choice"]["candidate_count"], 1)

    def test_cost_decomposition_fixed_reserve_and_margin_are_auditable(self):
        result = choose_cost_plan([probe(17, 30, 5, 2)], remaining_calibrate_seconds=9000, max_candidates=1)
        choice = result["choice"]
        self.assertEqual(choice["unscaled_work_seconds"], 122*30 + 66*5 + 34*2)
        self.assertAlmostEqual(choice["estimate_seconds"], 600 + 1.3*(122*30 + 66*5 + 34*2))
        self.assertEqual(result["measurement_records"], [probe(17, 30, 5, 2)])
        self.assertFalse(result["executes_gpu_work"])
        self.assertFalse(result["holdout_used"])

    def test_single_unique_shape_is_not_fabricated_into_two_conditions(self):
        result = choose_cost_plan([probe(17)], remaining_calibrate_seconds=9000)
        self.assertEqual(result["choice"]["distinct_condition_count"], 1)
        self.assertFalse(result["choice"]["crossover_design_complete"])
        self.assertEqual(result["choice"]["work_counts"]["budgeted_condition_slots"], 2)

    def test_missing_generation_or_quality_is_not_zero_or_negative_evidence(self):
        for field in ("generation_wall_seconds", "quality_wall_seconds"):
            row = probe(17)
            del row[field]
            result = choose_cost_plan([row], remaining_calibrate_seconds=9000)
            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertIsNone(result["choice"])
            self.assertIn("missing_measured_" + field, result["rejected_measurements"][0]["reasons"])

    def test_missing_transfer_requires_explicit_configuration_reservation(self):
        row = probe(17, transfer=None)
        missing = choose_cost_plan([row], remaining_calibrate_seconds=9000)
        self.assertEqual(missing["status"], "insufficient_evidence")
        result = choose_cost_plan([row], remaining_calibrate_seconds=9000, transfer_reserve_seconds=4)
        self.assertEqual(result["status"], "planned")
        bound = result["choice"]["cost_bounds"]["transfers"][0]
        self.assertIsNone(bound["observed_seconds"])
        self.assertEqual(bound["provenance"], "configuration")
        self.assertEqual(bound["seconds"], 4)
        self.assertFalse(result["transfer_reservation"]["measured"])

    def test_explicit_reserve_never_undercuts_larger_measured_transfer(self):
        result = choose_cost_plan([probe(17, transfer=10)], remaining_calibrate_seconds=9000, transfer_reserve_seconds=1)
        self.assertEqual(result["choice"]["cost_bounds"]["extra_state_action_wall_seconds"], 10)
        self.assertEqual(result["choice"]["cost_bounds"]["transfers"][0]["provenance"], "measurement")

    def test_excluding_encoding_or_archive_makes_cost_basis_ineligible(self):
        for field in ("encoding_included", "archive_included"):
            row = probe(17)
            row["timing_scope"][field] = False
            self.assertEqual(choose_cost_plan([row], remaining_calibrate_seconds=9000)["status"], "insufficient_evidence")

    def test_frozen_encoding_setup_is_separately_reserved_and_root_fields_available(self):
        row = probe(17, generation=30, quality=5, transfer=None)
        row["timing_scope"].update(encoding_included=False, encoding_mode="frozen_text_stage_setup_reserved_separately")
        row["transfer_reservation_seconds"] = 1.
        result = choose_cost_plan([row], remaining_calibrate_seconds=9000)
        verify_seal(result)
        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["selected_shape"], row["shape"])
        self.assertEqual(result["candidate_limit"], 2)
        self.assertEqual(result["paired_conditions"], [row["shape"]])
        self.assertEqual(result["setup_reservation"]["seconds"], 600.)
        self.assertFalse(result["setup_reservation"]["measured"])
        observation = result["selected_cost_observation"]
        self.assertEqual(observation["transfer_reservation_seconds"], 1.)
        self.assertEqual(observation["transfer_provenance"][0]["provenance"], "configuration")
        self.assertEqual(result["unit_estimates"]["calibrate"], 1.3*(3*30+3*5)+90)
        self.assertEqual(result["unit_estimates"]["mechanism"], 1.3*(6*30+4*5+2)+90)

    def test_global_transfer_reservation_alias_and_invalid_per_probe_reservation(self):
        row = probe(17, transfer=None)
        self.assertEqual(choose_cost_plan([row], remaining_calibrate_seconds=9000,
            transfer_reservation_seconds=1)["status"], "planned")
        row["transfer_reservation_seconds"] = 0
        self.assertEqual(choose_cost_plan([row], remaining_calibrate_seconds=9000)["status"], "insufficient_evidence")

    def test_completed_cost_basis_that_cannot_fit_reports_insufficient_budget(self):
        result = choose_cost_plan([probe(17, generation=80)], remaining_calibrate_seconds=9000)
        self.assertEqual(result["status"], "insufficient_budget")
        self.assertIsNone(result["choice"])
        self.assertEqual(result["rejected_measurements"], [])

    def test_unaffordable_or_incomplete_larger_probe_is_never_extrapolated(self):
        larger = probe(81, generation=None)
        result = choose_cost_plan([probe(17), larger], remaining_calibrate_seconds=9000)
        self.assertEqual(result["choice"]["shape"]["frames"], 17)
        self.assertEqual(len(result["rejected_measurements"]), 1)

    def test_probe_context_mixing_duplicate_maxima_and_holdout_do_not_select(self):
        rows = [probe(17), probe(33)]
        rows[1]["source_sha256"] = "c" * 64
        self.assertEqual(choose_cost_plan(rows, remaining_calibrate_seconds=9000)["status"], "insufficient_evidence")
        self.assertEqual(choose_cost_plan([probe(17), probe(17)], remaining_calibrate_seconds=9000)["status"], "insufficient_evidence")
        holdout = probe(17)
        holdout["split"] = "holdout"
        self.assertEqual(choose_cost_plan([holdout], remaining_calibrate_seconds=9000)["status"], "insufficient_evidence")

    def test_budget_equality_rejected_and_cannot_expand_or_weaken_scientific_limits(self):
        rows = [probe(17)]
        result = choose_cost_plan(rows, remaining_calibrate_seconds=9000, max_candidates=1)
        exactly = choose_cost_plan(rows, remaining_calibrate_seconds=result["choice"]["estimate_seconds"], max_candidates=1)
        self.assertEqual(exactly["status"], "insufficient_budget")
        for args in (dict(screen_count=13), dict(max_candidates=3), dict(safety_multiplier=1),
                     dict(fixed_reserve_seconds=0), dict(transfer_reserve_seconds=0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                choose_cost_plan(rows, remaining_calibrate_seconds=9000, **args)


if __name__ == "__main__":
    unittest.main()
