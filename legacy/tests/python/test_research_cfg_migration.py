import contextlib
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research import cfg_migration as migration
from ditserve.research.cfg_policy import resolve_policy
from ditserve.research.cfg_protocol import DEFAULT_CONFIG
from ditserve.research.common import atomic_json, digest, read_json
from ditserve.research.dual_matrix import inputs, seal


class FakeExperiment:
    def __init__(self, root):
        self.root = Path(root)
        self.config = read_json(DEFAULT_CONFIG)
        self.frozen = dict(sha256="p" * 64, source_sha256="s" * 64)
        self.rows, self.generated, self.estimates = [], [], []
        self.costs, self.implementations = dict(a100=1., v100=2.), {}
        self.serial = 0
        self.queue_wait_per_request = 2000.
        self.fail_cold = self.fail_path = self.corrupt_queue_identity = False
        self.selection = dict(status="selected", selected="independent_004", context_sha256=self.context())
        self.write("selection", self.selection)

    def context(self):
        return "context"

    def policy(self, recipe):
        return resolve_policy(self.config, recipe, self.implementations.get(recipe))

    @contextlib.contextmanager
    def unit(self, name, estimate):
        self.estimates.append((name, estimate))
        yield

    def write(self, name, value):
        value = dict(value, protocol_sha256=self.frozen["sha256"], source_sha256=self.frozen["source_sha256"])
        value.pop("sha256", None)
        row = seal(value)
        atomic_json(self.root / (name + ".json"), row)
        return row

    def record(self, type_, case_id, **row):
        result = dict(type=type_, case_id=case_id, context_sha256=self.context(),
                      performance_valid=True, measurement_kind="measured_gpu", **row)
        self.rows.append(result)
        return result

    def _run(self, item, recipe, role, action):
        self.serial += 1
        generation = "generation-%d" % self.serial
        policy = self.policy(recipe)
        path = [role] if action == "stay" else ["v100", "a100"]
        return dict(status="completed", recipe_id=recipe, device_role=role, action=action, action_step=16,
            prompt_id=item["prompt_id"], seed=item["seed"], split=item["split"], generation_id=generation,
            output_sha256=digest(generation), queued_outputs=[], all_requests_decoded=True, policy=policy,
            identity=dict(context_sha256=self.context(), input=copy.deepcopy(item), recipe=recipe, action=action,
                path=path, attention_path=["flash" if device == "a100" else "efficient" for device in path],
                boundary=16, policy_sha256=policy["sha256"], implementation=policy["implementation"]))

    def generate(self, item, recipe="off", *, role="a100", action="stay", queue=0):
        self.generated.append((item["prompt_id"], item["seed"], recipe, role, action, queue))
        run = self._run(item, recipe, role, action)
        base = {"stay": 1000., "full_migrate": 600., "cold_migrate": 500.}[action]
        wait = self.queue_wait_per_request * queue
        run.update(target_wait_ms=wait if action != "stay" else 0., elapsed_ms=2000 + wait,
                   critical_path_ms=max(base, wait) if action == "stay" else base + wait)
        for _ in range(queue):
            queued = self._run(inputs(self.config)[0], "off", "a100", "stay")
            if self.corrupt_queue_identity:
                queued["identity"]["input"]["seed"] = 999
            run["queued_outputs"].append(queued)
        return run

    def quality(self, candidate, reference):
        failed = ((self.fail_cold and candidate["recipe_id"] != "off" and candidate["action"] == "cold_migrate")
                  or (self.fail_path and candidate["recipe_id"] == "off" and candidate["action"] == "full_migrate"))
        mean = .06 if failed else .001
        return dict(status="ok", finite=True, range_valid=True, frame_count=17,
            lpips_per_frame=[mean] * 17, lpips_mean=mean, lpips_p95=mean, lpips_max=mean,
            candidate_output_sha256=candidate["output_sha256"], reference_output_sha256=reference["output_sha256"],
            candidate_identity_sha256=digest(candidate["identity"]), reference_identity_sha256=digest(reference["identity"]))


class CfgMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.experiment = FakeExperiment(self.temp.name)
        self.held = dict(status="passed", eligible=True)
        for name, callback in (("read_records", lambda *args: copy.deepcopy(self.experiment.rows)),
                               ("select_candidate", lambda *args, **kwargs: self.experiment.selection),
                               ("holdout_gate", lambda *args: self.held)):
            mocked = patch.object(migration, name, side_effect=callback)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_unqualified_and_tampered_selection_start_no_requests(self):
        self.held.update(status="failed", eligible=False)
        result = migration.migration(self.experiment)
        self.assertEqual(result["status"], "direction_stopped")
        self.assertFalse(self.experiment.generated)
        self.held.update(status="passed", eligible=True)
        selection = read_json(self.experiment.root / "selection.json")
        selection["selected"] = "paired_008"
        selection.pop("sha256")
        atomic_json(self.experiment.root / "selection.json", seal(selection))
        self.assertFalse(migration.eligibility(self.experiment)["eligible"])
        self.assertEqual(migration.mechanism(self.experiment)["status"], "unverified")
        self.assertFalse(self.experiment.generated)

    def test_both_actions_have_complete_independent_development_and_holdout_quality(self):
        result = migration.migration(self.experiment)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["qualified_actions"], ["full_migrate", "cold_migrate"])
        rows = [row for row in self.experiment.rows if row["type"] == "cfg_migration_quality"]
        self.assertEqual(len(rows), 24)
        self.assertEqual(len([row for row in rows if row["split"] == "dev"]), 16)
        for action in migration.MIGRATION_ACTIONS:
            self.assertEqual(result["actions"][action]["dev"]["observed"], 8)
            self.assertEqual(result["actions"][action]["holdout"]["observed"], 4)
        screen = [row for row in self.experiment.rows if row["type"] == "cfg_migration_screen"]
        self.assertEqual(len(screen), 6)
        self.assertEqual({row["prompt_id"] for row in screen}, {"kettle", "pendulum"})
        self.assertTrue(all(value > 10 for _, value in self.experiment.estimates))

    def test_invalid_path_reference_stops_before_candidate_and_marks_missing(self):
        self.experiment.fail_path = True
        result = migration.migration(self.experiment)
        self.assertEqual(result["status"], "quality_stopped")
        self.assertEqual(result["remaining_matrix"], "unverified")
        self.assertFalse([row for row in self.experiment.rows if row["type"] == "cfg_migration_quality"])
        self.assertEqual(len([row for row in self.experiment.rows if row["type"] == "cfg_migration_reference"]), 1)

    def test_failed_cold_is_excluded_and_full_can_run_real_queue_mechanism(self):
        self.experiment.fail_cold = True
        result = migration.migration(self.experiment)
        self.assertEqual(result["qualified_actions"], ["full_migrate"])
        start = len(self.experiment.generated)
        result = migration.mechanism(self.experiment)
        self.assertEqual(result["qualified_actions"], ["full_migrate"])
        self.assertEqual(result["queue_length"], 1)
        self.assertEqual(result["gates"]["idle"]["status"], "supported")
        self.assertFalse(any(row[4] == "cold_migrate" for row in self.experiment.generated[start:]))
        rows = [r for r in self.experiment.rows if r["type"] == "cfg_mechanism"]
        self.assertEqual(len(rows), 4)
        for condition in ("idle", "busy"):
            pairs = [r for r in rows if r["condition"] == condition]
            self.assertEqual(pairs[0]["order"], list(reversed(pairs[1]["order"])))
            self.assertTrue(all(r["quality_passed"] for r in pairs))
            self.assertTrue(all(set(r["costs_ms"]) == {"stay", "full_migrate"} for r in pairs))
        busy = [r for r in rows if r["condition"] == "busy"]
        self.assertTrue(all(r["queue_wait_ms"] >= r["break_even_wait_ms"] * 1.25 for r in busy))

    def test_mechanism_recomputes_migration_qualification_from_records(self):
        migration.migration(self.experiment)
        self.experiment.rows = [r for r in self.experiment.rows if r["type"] != "cfg_migration_quality"]
        start = len(self.experiment.generated)
        result = migration.mechanism(self.experiment)
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(len(self.experiment.generated), start)

    def test_queue_exhaustion_is_unverified_without_artificial_wait(self):
        migration.migration(self.experiment)
        self.experiment.queue_wait_per_request = 10
        result = migration.mechanism(self.experiment)
        self.assertIsNone(result["queue_length"])
        self.assertEqual(result["queue_probes"], 4)
        self.assertEqual(result["gates"]["busy"]["status"], "unverified")
        self.assertEqual([r["queue_length"] for r in self.experiment.rows if r["type"] == "cfg_queue_probe"], [1, 2, 4, 8])
        self.assertFalse(any(r["condition"] == "busy" for r in self.experiment.rows if r["type"] == "cfg_mechanism"))

    def test_queue_outputs_need_own_identity_and_quality_evidence(self):
        migration.migration(self.experiment)
        self.experiment.corrupt_queue_identity = True
        result = migration.mechanism(self.experiment)
        self.assertIsNone(result["queue_length"])
        self.assertTrue(all(not r["quality_passed"] for r in self.experiment.rows if r["type"] == "cfg_queue_probe"))

    def test_queue_repeated_output_cannot_count_as_multiple_completed_requests(self):
        item = inputs(self.experiment.config)[0]
        reference = self.experiment.generate(item)
        run = self.experiment.generate(item, "independent_004", role="v100", action="full_migrate", queue=2)
        self.assertTrue(migration._queue_quality(self.experiment, run, reference, 2))
        run["queued_outputs"][1] = copy.deepcopy(run["queued_outputs"][0])
        self.assertFalse(migration._queue_quality(self.experiment, run, reference, 2))

    def test_missing_or_invalid_development_costs_stop_before_mechanism(self):
        migration.migration(self.experiment)
        first = next(r for r in self.experiment.rows if r["type"] == "cfg_migration_screen")
        first["performance_valid"] = False
        start = len(self.experiment.generated)
        result = migration.mechanism(self.experiment)
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(len(self.experiment.generated), start)


if __name__ == "__main__":
    unittest.main()
