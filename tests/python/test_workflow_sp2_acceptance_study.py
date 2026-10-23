"""Actual-backend admission cannot inherit a diagnostic's success or reset budget."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_sp2_acceptance_study as acceptance


api = acceptance.api


class RepairedAcceptanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.study = dict(root=str(self.root), run_id="run03", source="/fixture/r009",
            source_digest="new-backend", jobs=[], decisions=[], selections={},
            budget_seconds=28800, active_job=None)
        self.evidence = {}
        self.run_ids = []
        self.accepted = api.add_job(self.study, "acceptance", "calibration", {"fixture": "acceptance"},
                                   module="fixture", devices=(0, 1))
        self.control = api.add_job(self.study, "sp-conservative-control", "sp",
            dict(variants=[acceptance.POLICY], seeds=[3501, 3502], frames=[17, 33],
                 steps=20, prefixes=[1, 4, 20], threshold=.05), module="fixture")
        self.control.update(source="/fixture/r008", source_digest="old-diagnostic")
        self.old = api.add_job(self.study, "sp-diagnostic", "sp", {"variants": ["baseline"]}, module="fixture")
        self.old["decision"] = "numerical_failed"
        self.old_pause = api.add_job(self.study, "elastic-pause", "sp", {"stage": "pause"},
                                     module="fixture", requires=["sp-diagnostic"])
        self.old_pause["decision"] = "prerequisite_not_qualified"
        self.complete(self.accepted)
        self.complete(self.control)
        self.original_jobs = copy.deepcopy(self.study["jobs"])
        api.atomic_json(self.root / "budget.json", dict(limit_seconds=28800,
            charged_seconds=22442.93002836965, active=None, attempts=[{"prior": "retained"}]))
        self.original_budget = (self.root / "budget.json").read_bytes()
        self.addCleanup(patch.stopall)
        patch.object(acceptance, "evidence", side_effect=lambda study, job: self.evidence.get(
            job["id"], dict(guarded=False, qualified=False, capacity=False, result={}, launch={}))).start()

    def complete(self, job, *, guarded=True, qualified=True, capacity=True):
        path = Path(job["output"]) / "run.json"
        result = dict(schema="ditserve.workflow.elastic_validation.v1", status="completed",
                      configuration=api.read(job["input"]))
        api.atomic_json(path, result)
        self.evidence[job["id"]] = dict(guarded=guarded, qualified=qualified, capacity=capacity,
            path=path, result=result, launch=dict(status="completed", output_sha256=api.binding(path)["sha256"]))

    def run_job(self, study, job):
        self.run_ids.append(job["id"])
        self.complete(job)
        return True

    def run_chain(self, callback=None):
        return acceptance.run_followup(self.study, callback or self.run_job)

    def test_exact_new_chain_bound_to_one_backend_preserves_prior_failures_and_budget(self):
        result = self.run_chain()
        expected_ids = [acceptance.PREFIX + stage for stage in acceptance.STAGES]
        self.assertEqual(self.run_ids, expected_ids)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["completed_jobs"], expected_ids)
        self.assertEqual(result["unqualified_jobs"], [])
        self.assertEqual(self.study["jobs"][:4], self.original_jobs)
        self.assertEqual((self.root / "budget.json").read_bytes(), self.original_budget)
        self.assertEqual(self.study["budget_seconds"], 28800)
        for index, (stage, job) in enumerate(zip(acceptance.STAGES, self.study["jobs"][4:])):
            config = api.read(job["input"])
            self.assertEqual(config, acceptance.configuration(stage))
            self.assertEqual(config["sp_linear_policy"], "full_rows_except_ffn1")
            self.assertEqual(config["pause_blocks"], [23, 14, 23])
            self.assertEqual(config["seeds"], [3501, 3502])
            self.assertEqual(config["frames"], [17, 33])
            self.assertEqual((job["estimated_seconds"], job["timeout_seconds"]), (600, 1500))
            self.assertEqual(job["family"], "sp")
            self.assertEqual(job["devices"], [0, 1])
            self.assertEqual(job["source_digest"], "new-backend")
            self.assertEqual(job["requires"], ["acceptance"] + (expected_ids[index-1:index] if index else []))
            provenance = job["metadata"]["diagnostic_provenance"]
            self.assertEqual(provenance["source_digest"], "old-diagnostic")
            self.assertEqual(provenance["role"], "provenance_only_not_backend_acceptance")
            self.assertFalse(job["metadata"]["performance_evidence"])
            self.assertEqual(len(job["input_bindings"]), 5 + 2 * index)
            self.assertEqual(job["input_bindings"], [api.binding(row["path"]) for row in job["input_bindings"]])
        self.assertEqual(result["online_scheduling"], "unverified")
        self.assertEqual(result["sana_interleave"], "unverified")

    def test_resume_does_not_register_new_jobs_or_change_frozen_bindings(self):
        self.run_chain()
        jobs_before = copy.deepcopy(self.study["jobs"])
        self.run_ids.clear()
        result = self.run_chain(lambda study, job: self.run_ids.append(job["id"]))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(self.study["jobs"], jobs_before)
        self.assertEqual(len(self.run_ids), 3)

    def test_existing_acceptance_needs_numeric_guard_and_capacity_before_any_registration(self):
        for key in ("guarded", "qualified", "capacity"):
            with self.subTest(key=key):
                self.evidence["acceptance"][key] = False
                result = self.run_chain()
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(len(self.study["jobs"]), 4)
                self.assertEqual(self.run_ids, [])
                self.evidence["acceptance"][key] = True

    def test_diagnostic_success_does_not_bypass_actual_stage_capacity_or_numeric_failure(self):
        for flag in ("guarded", "qualified", "capacity"):
            for failure_index in range(3):
                with self.subTest(flag=flag, stage=failure_index), tempfile.TemporaryDirectory() as directory:
                    # Each scenario owns its output files and immutable job IDs.
                    original_root = self.study["root"]
                    self.study["root"] = directory
                    self.study["jobs"] = copy.deepcopy(self.original_jobs)
                    self.run_ids.clear()
                    def run(study, job):
                        self.run_job(study, job)
                        if len(self.run_ids) == failure_index + 1:
                            self.evidence[job["id"]][flag] = False
                    result = self.run_chain(run)
                    self.assertEqual(result["status"], "blocked")
                    self.assertEqual(len(self.study["jobs"]), 5 + failure_index)
                    self.assertEqual(result["completed_jobs"], self.run_ids[:-1])
                    self.assertEqual(result["blocked_at"], self.run_ids[-1])
                    self.study["root"] = original_root

    def test_missing_launch_or_budget_deferral_stops_without_registering_downstream(self):
        result = self.run_chain(lambda study, job: False)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(self.study["jobs"]), 5)
        self.assertEqual(result["completed_jobs"], [])
        self.assertEqual(result["blocked_at"], "elastic-repaired-transfer")

    def test_raw_diagnostic_change_or_stale_launch_hash_cannot_register(self):
        result = self.evidence["sp-conservative-control"]
        result["launch"]["output_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "changed while binding"):
            self.run_chain()
        self.assertEqual(len(self.study["jobs"]), 4)

    def test_frozen_control_matrix_must_match_expected_policy_and_full_matrix(self):
        config = api.read(self.control["input"])
        config["frames"] = [17]
        api.atomic_json(self.control["input"], config)
        self.control["configuration_sha256"] = api.digest(config)
        self.control["input_bindings"] = [api.binding(self.control["input"])]
        with self.assertRaisesRegex(ValueError, "frozen acceptance recipe"):
            self.run_chain()
        self.assertEqual(self.run_ids, [])

    def test_existing_chain_cannot_switch_source_command_or_metadata(self):
        self.run_chain()
        job = self.study["jobs"][4]
        original = copy.deepcopy(job)
        for field, value in (("source_digest", "another-backend"), ("source", "/fixture/r010"),
                             ("command", ["different-runner"]), ("metadata", {}),
                             ("estimated_seconds", 1), ("timeout_seconds", 9999)):
            with self.subTest(field=field):
                job.clear()
                job.update(copy.deepcopy(original))
                job[field] = value
                self.run_ids.clear()
                with self.assertRaises(ValueError):
                    self.run_chain()
                self.assertEqual(self.run_ids, [])

    def test_missing_or_mutated_dependency_binding_cannot_resume(self):
        self.run_chain()
        job = self.study["jobs"][4]
        job["input_bindings"].pop()
        self.run_ids.clear()
        with self.assertRaisesRegex(ValueError, "dependency bindings"):
            self.run_chain()
        self.assertEqual(self.run_ids, [])

    def test_qualified_boolean_cannot_mask_wrong_actual_result_config_or_schema(self):
        def run(study, job):
            self.run_job(study, job)
            self.evidence[job["id"]]["result"]["configuration"]["sp_linear_policy"] = "native"
        with self.assertRaisesRegex(ValueError, "frozen configuration"):
            self.run_chain(run)
        self.assertEqual(self.run_ids, ["elastic-repaired-transfer"])
        self.assertEqual(len(self.study["jobs"]), 5)


if __name__ == "__main__":
    unittest.main()
