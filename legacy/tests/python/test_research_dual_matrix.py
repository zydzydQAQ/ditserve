import copy
import unittest

from ditserve.research.common import digest, read_json
from ditserve.research.dual_protocol import DEFAULT_CONFIG
from ditserve.research.dual_matrix import (ACTIONS, cache_gate, choose_queue, estimate_matrix, fits_plan,
    inputs, known_quality, mechanism_gate, migration_quality_gate, select_backend, select_migration)


def quality(value=.01, count=17, *, candidate=None, reference=None):
    result = dict(status="ok", finite=True, range_valid=True, lpips_per_frame=[value] * count,
                  frame_count=count, lpips_mean=value)
    if candidate is not None and reference is not None:
        result.update(candidate_output_sha256=candidate["output_sha256"], reference_output_sha256=reference["output_sha256"],
                      candidate_identity_sha256=digest(candidate["identity"]), reference_identity_sha256=digest(reference["identity"]))
    return result


def trajectory(item, role, recipe, action, context="context", v100_backend="math"):
    path = [role] if action == "stay" else ["v100", "a100"]
    identity = dict(context_sha256=context, input=copy.deepcopy(item), recipe=recipe, action=action,
                    path=path, attention_path=["flash" if device == "a100" else v100_backend for device in path], boundary=16)
    return dict(status="completed", output_sha256=digest(identity), identity=identity, device_role=role,
                recipe_id=recipe, action=action, action_step=16, prompt_id=item["prompt_id"],
                seed=item["seed"], split=item["split"], elapsed_ms=100.)


class DualMatrixTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(DEFAULT_CONFIG)
        self.shape = self.config["shapes"]["main"]

    def test_development_and_holdout_are_disjoint_and_invalid_split_never_vacuously_passes(self):
        dev, holdout = inputs(self.config), inputs(self.config, "holdout")
        self.assertEqual(len(dev), 8)
        self.assertEqual(len(holdout), 4)
        self.assertEqual({r["prompt_id"] for r in holdout}, {"tram", "aquarium"})
        self.assertEqual({r["seed"] for r in holdout}, {201, 202})
        with self.assertRaises(ValueError):
            inputs(self.config, "test")
        with self.assertRaises(ValueError):
            migration_quality_gate([], self.config, "test", "full_migrate", "context")

    def test_whole_trajectory_quality_rejects_missing_frames_nonfinite_and_false_mean(self):
        self.assertTrue(known_quality(quality()))
        for value in (quality(count=1), quality(count=16), quality(count=18), quality(value=.06),
                      dict(quality(), lpips_mean=0), dict(quality(0), lpips_mean=False),
                      dict(quality(), lpips_per_frame=[float("nan")] * 17),
                      dict(quality(), frame_count=True), dict(quality(), finite=False)):
            with self.subTest(value=value):
                self.assertFalse(known_quality(value))

    def test_backend_requires_both_orders_quality_context_and_each_gain(self):
        pairs = [dict(order=order, shape=self.shape, context_sha256="context", performance_valid=True,
                      quality=quality(), costs_ms=dict(math=100, efficient=90))
                 for order in (["math", "efficient"], ["efficient", "math"])]
        self.assertEqual(select_backend(pairs, shape=self.shape, context_sha256="context")["selected"], "efficient")
        for change in (dict(quality=quality(count=1)), dict(context_sha256="other"),
                       dict(performance_valid=False), dict(costs_ms=dict(math=100, efficient=98)),
                       dict(costs_ms=dict(math=float("inf"), efficient=90)), dict(order=pairs[0]["order"])):
            candidate = [pairs[0], dict(pairs[1], **change)]
            self.assertEqual(select_backend(candidate, shape=self.shape, context_sha256="context")["selected"], "math")
        self.assertEqual(select_backend(pairs[:1], shape=self.shape, context_sha256="context")["selected"], "math")

    def test_optional_capacity_does_not_block_complete_quality_mainline(self):
        plan = estimate_matrix(dict(a100=1., v100=2.))
        self.assertEqual(plan["optional_phases"], ["capacity"])
        self.assertGreater(plan["planned_with_optional_gpu_seconds"], plan["required_gpu_seconds"])
        remaining = dict(plan["phases_gpu_seconds"], capacity=0.)
        self.assertTrue(fits_plan(plan, plan["required_gpu_seconds"], remaining))
        remaining["migration"] -= 1
        self.assertFalse(fits_plan(plan, plan["required_gpu_seconds"], remaining))
        self.assertFalse(fits_plan(plan, plan["required_gpu_seconds"] - 1))
        self.assertFalse(plan["quality_samples_reduced"])

    def cache_rows(self):
        rows = []
        for item in inputs(self.config):
            candidate = trajectory(item, "a100", "t008", "stay")
            reference = trajectory(item, "a100", "off", "stay")
            rows.append(dict(item, device_role="a100", recipe_id="t008", status="completed", performance_valid=True,
                             context_sha256="context", candidate=candidate, reference=reference,
                             quality=quality(candidate=candidate, reference=reference)))
        return rows

    def test_cache_requires_all_eight_development_inputs_and_sealed_valid_units(self):
        rows = self.cache_rows()
        self.assertTrue(cache_gate(rows, self.config, "a100", "context")["eligible"])
        for changed in (rows[:-1], rows + [rows[0]], [dict(rows[0], split="holdout")] + rows[1:],
                        [dict(rows[0], performance_valid=False)] + rows[1:],
                        [dict(rows[0], quality=quality(count=1))] + rows[1:]):
            self.assertFalse(cache_gate(changed, self.config, "a100", "context")["eligible"])

    def screen(self):
        return [dict(action=action, prompt_id=prompt, seed=101, split="dev", elapsed_ms=100 + index,
                     context_sha256="context", performance_valid=True, status="completed")
                for index, action in enumerate(ACTIONS[1:]) for prompt in ("kettle", "pendulum")]

    def test_screen_never_selects_on_holdout_missing_or_contaminated_measurements(self):
        rows = self.screen()
        self.assertEqual(select_migration(rows, "context")["selected"], "full_migrate")
        for changed in (rows[:-1], [dict(rows[0], performance_valid=False)] + rows[1:],
                        [dict(rows[0], elapsed_ms=float("inf"))] + rows[1:]):
            self.assertIsNone(select_migration(changed, "context")["selected"])
        with self.assertRaisesRegex(ValueError, "held-out"):
            select_migration([dict(rows[0], split="holdout")] + rows[1:], "context")
        rows = [dict(r, status="ineligible") if r["action"] == "certified_omit" else r for r in rows]
        self.assertEqual(select_migration(rows, "context")["selected"], "full_migrate")

    def migration_rows(self, split):
        rows = []
        for item in inputs(self.config, split):
            candidate = trajectory(item, "v100", "t008", "full_migrate")
            common = trajectory(item, "a100", "off", "stay")
            path = trajectory(item, "v100", "off", "full_migrate")
            rows.append(dict(item, action="full_migrate", context_sha256="context", status="completed", performance_valid=True,
                candidate=candidate, common_reference=common, path_reference=path,
                path_reference_quality=quality(candidate=path, reference=common),
                canonical_quality=quality(candidate=candidate, reference=common),
                path_quality=quality(candidate=candidate, reference=path)))
        return rows

    def test_cache_metrics_bind_actual_videos_identity_and_configured_prompt_text(self):
        rows = self.cache_rows()
        for mutate in (lambda row: row.update(quality=rows[1]["quality"]),
                       lambda row: row["candidate"].update(output_sha256="0" * 64),
                       lambda row: row["candidate"]["identity"].update(context_sha256="other"),
                       lambda row: row["candidate"]["identity"]["input"].update(prompt="different text"),
                       lambda row: row["reference"]["identity"].update(attention_path=["math"]),
                       lambda row: row["candidate"].update(device_role="v100")):
            changed = copy.deepcopy(rows)
            mutate(changed[0])
            self.assertFalse(cache_gate(changed, self.config, "a100", "context")["eligible"])
        config = copy.deepcopy(self.config)
        config["prompts"][0]["text"] += " changed"
        self.assertFalse(cache_gate(rows, config, "a100", "context")["eligible"])

    def test_migration_three_metric_pairs_cannot_exchange_refs_actions_or_backends(self):
        rows = self.migration_rows("dev")
        for mutate in (lambda row: row.update(path_quality=row["canonical_quality"]),
                       lambda row: row.update(path_reference_quality=rows[1]["path_reference_quality"]),
                       lambda row: row["candidate"]["identity"].update(action="cold_migrate"),
                       lambda row: row["path_reference"]["identity"].update(attention_path=["efficient", "flash"]),
                       lambda row: row["common_reference"]["identity"]["input"].update(seed=201),
                       lambda row: row["candidate"].update(action_step=15)):
            changed = copy.deepcopy(rows)
            mutate(changed[0])
            self.assertFalse(migration_quality_gate(changed, self.config, "dev", "full_migrate", "context")["eligible"])
        # Rebinding both hashes cannot legitimize an undeclared device path or
        # mixing MATH/EFFICIENT samples within one configuration.
        changed = copy.deepcopy(rows)
        row = changed[0]
        for key in ("candidate", "path_reference"):
            row[key]["identity"]["attention_path"] = ["efficient", "flash"]
        row["path_reference_quality"] = quality(candidate=row["path_reference"], reference=row["common_reference"])
        row["canonical_quality"] = quality(candidate=row["candidate"], reference=row["common_reference"])
        row["path_quality"] = quality(candidate=row["candidate"], reference=row["path_reference"])
        self.assertFalse(migration_quality_gate(changed, self.config, "dev", "full_migrate", "context")["eligible"])

    def test_migration_quality_requires_whole_path_reference_canonical_and_path(self):
        for split in ("dev", "holdout"):
            rows = self.migration_rows(split)
            self.assertTrue(migration_quality_gate(rows, self.config, split, "full_migrate", "context")["eligible"])
            for key in ("path_reference_quality", "canonical_quality", "path_quality"):
                bad = [dict(rows[0], **{key: quality(.09)})] + rows[1:]
                self.assertFalse(migration_quality_gate(bad, self.config, split, "full_migrate", "context")["eligible"])
            self.assertFalse(migration_quality_gate(rows[:-1], self.config, split, "full_migrate", "context")["eligible"])
            self.assertFalse(migration_quality_gate([dict(rows[0], performance_valid=False)] + rows[1:], self.config,
                split, "full_migrate", "context")["eligible"])
        self.assertFalse(migration_quality_gate(self.migration_rows("dev"), self.config, "holdout", "full_migrate", "context")["eligible"])

    def test_queue_is_real_decoded_and_first_observed_crossing_only(self):
        observations = [dict(queue_length=n, wait_ms=100 * n, all_requests_decoded=True, performance_valid=True)
                        for n in (1, 2, 4, 8)]
        self.assertEqual(choose_queue(observations, 200), 4)
        self.assertIsNone(choose_queue(observations[1:], 200))
        self.assertIsNone(choose_queue([dict(observations[0], wait_ms=float("inf"))], 200))
        self.assertIsNone(choose_queue(observations, 0))
        self.assertIsNone(choose_queue([dict(observations[0], all_requests_decoded=False)] + observations[1:], 200))

    def test_mechanism_opposite_order_complete_makespan_costs_are_finite(self):
        pairs = [dict(order=order, performance_valid=True, quality_passed=True, all_requests_decoded=True,
                      costs_ms=dict(stay=100, candidate=90)) for order in (["stay", "candidate"], ["candidate", "stay"])]
        self.assertTrue(mechanism_gate(pairs)["supported"])
        for change in (dict(costs_ms=dict(stay=100, candidate=float("nan"))),
                       dict(costs_ms=dict(stay=float("inf"), candidate=90)),
                       dict(quality_passed=False), dict(performance_valid=False), dict(all_requests_decoded=False)):
            self.assertFalse(mechanism_gate([pairs[0], dict(pairs[1], **change)])["supported"])
        self.assertFalse(mechanism_gate([dict(p, order=[]) for p in pairs])["supported"])


if __name__ == "__main__":
    unittest.main()
