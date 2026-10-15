import copy
import unittest

from ditserve.research.common import digest, read_json
from ditserve.research.cfg_policy import resolve_policy
from ditserve.research.cfg_protocol import DEFAULT_CONFIG
from ditserve.research.dual_matrix import inputs
from ditserve.research.cfg_matrix import (BOUND_METHOD, CANDIDATES, DEVICES, MIGRATION_ACTIONS,
    SCREEN_INPUTS, development_gate, holdout_gate, mechanism_gate, migration_quality_gate,
    optimistic_cost, select_candidate, stage_a_gate)


def config():
    return read_json(DEFAULT_CONFIG)


def trajectory(item, role, recipe, generation, *, phase="dev", action="stay", cost=90., context="ctx"):
    cfg = config()
    implementation = "oracle" if phase == "stage_a" and recipe == "paired_008" else (
        "split_prefix" if recipe == "paired_008" else "native")
    policy = resolve_policy(cfg, recipe, implementation)
    path = [role] if action == "stay" else ["v100", "a100"]
    identity = dict(context_sha256=context, input=copy.deepcopy(item), recipe=recipe, action=action,
                    path=path, attention_path=[cfg["attention"][d.upper()] for d in path], boundary=16,
                    policy_sha256=policy["sha256"], implementation=implementation)
    return dict(status="completed", identity=identity, output_sha256=digest(identity), generation_id=generation,
                elapsed_ms=cost, device_role=role, recipe_id=recipe, action=action, action_step=16,
                prompt_id=item["prompt_id"], seed=item["seed"], split=item["split"])


def quality(candidate, reference, loss=.01):
    return dict(status="ok", finite=True, range_valid=True, frame_count=17, lpips_per_frame=[loss] * 17,
                lpips_mean=loss, candidate_output_sha256=candidate["output_sha256"],
                reference_output_sha256=reference["output_sha256"],
                candidate_identity_sha256=digest(candidate["identity"]), reference_identity_sha256=digest(reference["identity"]))


def cache_rows(recipe="independent_004", phase="dev", cost=90.):
    split = "holdout" if phase == "holdout" else "dev"
    items = inputs(config(), split)
    if phase == "stage_a":
        items = [i for i in items if (i["prompt_id"], i["seed"]) in SCREEN_INPUTS]
    result = []
    orders = (["off", recipe], [recipe, "off"]) if phase == "dev" else (["off", recipe],)
    for role in DEVICES:
        for order_idx, order in enumerate(orders):
            for item in items:
                pair = "-".join((phase, recipe, role, str(order_idx), item["prompt_id"], str(item["seed"])))
                candidate = trajectory(item, role, recipe, pair + "-candidate", phase=phase, cost=cost)
                reference = trajectory(item, role, "off", pair + "-reference", phase=phase, cost=100.)
                if recipe == "paired_008" and phase == "stage_a":
                    candidate["optimistic_bound"] = dict(method=BOUND_METHOD, elapsed_ms=cost,
                        snapshot_ms=5., discarded_trial_ms=10., optimistic_ms=cost - 15.)
                result.append(dict(item, type="cfg_screen" if phase == "stage_a" else "cfg_calibration",
                    phase=phase, recipe_id=recipe, device_role=role, case_id=pair, pair_id=pair, fresh_pair=True,
                    order=list(order), status="completed", performance_valid=True,
                    timing_eligible=phase != "stage_a", measurement_kind="oracle" if phase == "stage_a" and recipe == "paired_008" else "full_trajectory_pair",
                    context_sha256="ctx", candidate=candidate, reference=reference, quality=quality(candidate, reference)))
    return result


def migration_rows(recipe="independent_004", action="full_migrate", split="dev"):
    result = []
    for item in inputs(config(), split):
        tag = action + item["prompt_id"] + str(item["seed"])
        candidate = trajectory(item, "v100", recipe, tag + "c", action=action)
        common = trajectory(item, "a100", "off", tag + "a")
        path = trajectory(item, "v100", "off", tag + "p", action="full_migrate")
        result.append(dict(item, type="cfg_migration_quality", recipe_id=recipe, action=action,
            status="completed", performance_valid=True, context_sha256="ctx", candidate=candidate,
            common_reference=common, path_reference=path, canonical_quality=quality(candidate, common),
            path_quality=quality(candidate, path), path_reference_quality=quality(path, common)))
    return result


def mechanism_pairs(condition="idle", actions=MIGRATION_ACTIONS):
    order = ["stay", *actions]
    return [dict(type="cfg_mechanism", condition=condition, context_sha256="ctx", status="completed",
        recipe_id="independent_004",
        timing_eligible=True, performance_valid=True, measurement_kind="full_trajectory_pair",
        order=run_order, costs_ms=dict(stay=100., **{a: 90. for a in actions}),
        quality_passed=True, all_requests_decoded=True, cost_scope="all_requests_decode_complete",
        real_queue=condition == "busy", queue_coverage=condition == "busy", queue_length=2,
        queue_wait_ms=130., break_even_wait_ms=100.) for run_order in (order, list(reversed(order)))]


class CFGMatrixTests(unittest.TestCase):
    def test_stage_a_requires_all_three_inputs_on_both_devices_and_every_quality(self):
        for recipe in CANDIDATES:
            rows = cache_rows(recipe, "stage_a")
            self.assertTrue(stage_a_gate(rows, config(), recipe, "ctx")["eligible"])
            self.assertFalse(stage_a_gate(rows[:-1], config(), recipe, "ctx")["eligible"])
            bad = copy.deepcopy(rows)
            bad[0]["quality"] = quality(bad[0]["candidate"], bad[0]["reference"], .051)
            self.assertEqual(stage_a_gate(bad, config(), recipe, "ctx")["status"], "failed")

    def test_oracle_bound_is_recomputed_and_both_devices_must_pass(self):
        rows = cache_rows("paired_008", "stage_a", cost=110.)
        gate = stage_a_gate(rows, config(), "paired_008", "ctx")
        self.assertTrue(gate["eligible"])
        self.assertFalse(gate["performance_qualified"])
        for field, value in (("optimistic_ms", 1), ("snapshot_ms", float("nan")),
                             ("elapsed_ms", 1), ("method", "free_cache")):
            bad = copy.deepcopy(rows)
            bad[0]["candidate"]["optimistic_bound"][field] = value
            self.assertEqual(stage_a_gate(bad, config(), "paired_008", "ctx")["status"], "unverified")
        bad = copy.deepcopy(rows)
        for row in bad:
            if row["device_role"] == "v100":
                row["candidate"]["elapsed_ms"] = 120.
                row["candidate"]["optimistic_bound"].update(elapsed_ms=120., optimistic_ms=105.)
        self.assertEqual(stage_a_gate(bad, config(), "paired_008", "ctx")["status"], "failed")
        self.assertIsNone(optimistic_cost({}))

    def test_formal_requires_two_orders_each_eight_inputs_per_card_and_every_quality(self):
        rows = cache_rows()
        gate = development_gate(rows, config(), "independent_004", "ctx")
        self.assertTrue(gate["eligible"])
        self.assertEqual(gate["required"], 32)
        self.assertEqual(len(gate["device_orders"]), 4)
        for changed in (rows[:-1], rows + [rows[0]], [dict(rows[0], order=["off", "off"])] + rows[1:]):
            self.assertEqual(development_gate(changed, config(), "independent_004", "ctx")["status"], "unverified")
        bad = copy.deepcopy(rows)
        bad[0]["quality"] = quality(bad[0]["candidate"], bad[0]["reference"], .06)
        self.assertEqual(development_gate(bad, config(), "independent_004", "ctx")["status"], "failed")

    def test_formal_uses_ratio_of_total_costs_not_average_gains(self):
        rows = cache_rows()
        for row in rows:
            if row["prompt_id"] == "kettle" and row["seed"] == 101:
                row["reference"]["elapsed_ms"] = 10000.
                row["candidate"]["elapsed_ms"] = 11000.
            else:
                row["candidate"]["elapsed_ms"] = 1.
        self.assertEqual(development_gate(rows, config(), "independent_004", "ctx")["status"], "failed")

    def test_formal_oracle_diagnostic_stale_identity_and_reused_generation_never_qualify(self):
        for mutate in (
            lambda r: r.update(timing_eligible=False), lambda r: r.update(measurement_kind="diagnostic"),
            lambda r: r.update(measurement_kind="oracle"), lambda r: r.update(fresh_pair=False),
            lambda r: r["candidate"].update(diagnostic=True),
            lambda r: r["candidate"].update(generation_id=r["reference"]["generation_id"]),
            lambda r: r["candidate"]["identity"].update(policy_sha256="0" * 64),
            lambda r: r["candidate"]["identity"].update(implementation="oracle"),
            lambda r: r["candidate"]["identity"]["input"].update(prompt="different"),
            lambda r: r.update(quality={}), lambda r: r.update(performance_valid=False)):
            rows = cache_rows()
            mutate(rows[0])
            self.assertEqual(development_gate(rows, config(), "independent_004", "ctx")["status"], "unverified")

    def test_policy_outputs_and_backend_are_bound_even_with_rehashed_quality(self):
        rows = cache_rows()
        rows[0]["quality"] = rows[1]["quality"]
        self.assertFalse(development_gate(rows, config(), "independent_004", "ctx")["eligible"])
        rows = cache_rows()
        row = next(r for r in rows if r["device_role"] == "v100")
        for name in ("candidate", "reference"):
            row[name]["identity"]["attention_path"] = ["math"]
        row["quality"] = quality(row["candidate"], row["reference"])
        self.assertFalse(development_gate(rows, config(), "independent_004", "ctx")["eligible"])

    def test_selection_uses_development_ratios_tie_breaks_and_never_holdout(self):
        rows = sum((cache_rows(p, "stage_a") + cache_rows(p) for p in CANDIDATES), [])
        selection = select_candidate(rows, config(), "ctx")
        self.assertEqual(selection["selected"], "independent_004")
        self.assertFalse(selection["holdout_used"])
        selection2 = select_candidate(rows + cache_rows("paired_008", "holdout", cost=1.), config(), "ctx")
        self.assertEqual(selection2, selection)
        for row in rows:
            if row["recipe_id"] == "paired_008" and row["phase"] == "dev":
                row["candidate"]["elapsed_ms"] = 80.
        self.assertEqual(select_candidate(rows, config(), "ctx")["selected"], "paired_008")

    def test_missing_candidate_screen_prevents_premature_selection_and_diagnostic_policy_is_rejected(self):
        rows = cache_rows("independent_004", "stage_a") + cache_rows()
        self.assertIsNone(select_candidate(rows, config(), "ctx")["selected"])
        with self.assertRaises(ValueError):
            stage_a_gate([], config(), "independent_008", "ctx")

    def test_screen_failure_eliminates_candidate_but_does_not_demand_its_formal_matrix(self):
        rows = cache_rows("independent_004", "stage_a") + cache_rows() + cache_rows("paired_008", "stage_a")
        row = rows[-1]
        row["quality"] = quality(row["candidate"], row["reference"], .06)
        self.assertEqual(select_candidate(rows, config(), "ctx")["selected"], "independent_004")

    def test_holdout_requires_four_inputs_per_card_and_forbids_reselection(self):
        rows = cache_rows(phase="holdout")
        gate = holdout_gate(rows, config(), "independent_004", "ctx")
        self.assertTrue(gate["eligible"])
        self.assertEqual(gate["required"], 8)
        self.assertFalse(gate["reselect_allowed"])
        self.assertFalse(holdout_gate(rows[:-1], config(), "independent_004", "ctx")["eligible"])
        rows[0]["quality"] = quality(rows[0]["candidate"], rows[0]["reference"], .06)
        self.assertEqual(holdout_gate(rows, config(), "independent_004", "ctx")["status"], "failed")

    def test_migration_every_action_and_split_requires_all_three_bound_quality_comparisons(self):
        for action in MIGRATION_ACTIONS:
            for split in ("dev", "holdout"):
                rows = migration_rows(action=action, split=split)
                self.assertTrue(migration_quality_gate(rows, config(), split, "independent_004", action, "ctx")["eligible"])
                self.assertFalse(migration_quality_gate(rows[:-1], config(), split, "independent_004", action, "ctx")["eligible"])
                for key, c, r in (("path_reference_quality", "path_reference", "common_reference"),
                                  ("canonical_quality", "candidate", "common_reference"),
                                  ("path_quality", "candidate", "path_reference")):
                    bad = copy.deepcopy(rows)
                    bad[0][key] = quality(bad[0][c], bad[0][r], .06)
                    self.assertEqual(migration_quality_gate(bad, config(), split, "independent_004", action, "ctx")["status"], "failed")
        rows = migration_rows()
        rows[0]["path_quality"] = rows[0]["canonical_quality"]
        self.assertEqual(migration_quality_gate(rows, config(), "dev", "independent_004", "full_migrate", "ctx")["status"], "unverified")

    def test_mechanism_requires_actual_queue_coverage_complete_decoding_and_opposite_orders(self):
        for condition in ("idle", "busy"):
            rows = mechanism_pairs(condition)
            self.assertTrue(mechanism_gate(rows, condition=condition, context_sha256="ctx", quality_passed=True)["supported"])
            for change in (dict(all_requests_decoded=False), dict(cost_scope="suffix_only"),
                           dict(timing_eligible=False), dict(measurement_kind="oracle"),
                           dict(context_sha256="other"), dict(order=rows[0]["order"])):
                bad = [rows[0], dict(rows[1], **change)]
                self.assertEqual(mechanism_gate(bad, condition=condition, context_sha256="ctx", quality_passed=True)["status"], "unverified")
        for change in (dict(queue_wait_ms=124), dict(real_queue=False), dict(queue_coverage=False), dict(queue_length=3)):
            rows = mechanism_pairs("busy")
            rows[1].update(change)
            self.assertFalse(mechanism_gate(rows, condition="busy", context_sha256="ctx", quality_passed=True)["supported"])

    def test_mechanism_can_compare_only_individually_qualified_migration_action(self):
        rows = mechanism_pairs(actions=("cold_migrate",))
        value = mechanism_gate(rows, condition="idle", context_sha256="ctx", quality_passed=dict(full_migrate=False, cold_migrate=True))
        self.assertTrue(value["supported"])
        self.assertEqual(value["supported_actions"], ["cold_migrate"])
        self.assertFalse(mechanism_gate(rows, condition="idle", context_sha256="ctx", quality_passed=False)["supported"])


if __name__ == "__main__":
    unittest.main()
