"""Runtime experiments retain immutable controls and independent failure scopes."""

import copy
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_runtime_optimization_study as runtime


class RuntimeOptimizationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.study = dict(root=str(self.root), run_id="run03", source="/frozen/new",
            source_digest="new-digest", jobs=[], selections={}, decisions=[], budget_seconds=28800)
        self.evidence = {}
        for stage in ("transfer", "pause", "resize"):
            identifier = "elastic-repaired-" + stage
            self.control(identifier, dict(sp_linear_policy="full_rows_except_ffn1",
                stage=stage, frames=[17, 33], seeds=[3501, 3502], steps=20, threshold=.05),
                dict(schema="ditserve.workflow.elastic_validation.v1"))
        q = dict(passed=True, required_arms=["serial", "streams-shared-decode"],
                 scopes={name: dict(status="passed") for name in ("numeric", "capacity", "steady_state")})
        arms = [dict(id=name, frame_pair=[17, 17], result=dict(completed_pairs=5,
                     unrun_pairs=[], qualification=dict(passed=True)),
                     launch=dict(returncode=0, reaped=True, timed_out=False))
                for name in q["required_arms"]]
        self.control("overlap-isolated-warmed-lifecycle",
            dict(frame_pairs=[[17, 17]], warmup_pairs=2, lifecycle_pairs=3, steps=20),
            dict(schema="ditserve.workflow.overlap-isolated.v1", qualification=q, arms=arms),
            capacity=False)
        self.old_jobs = copy.deepcopy(self.study["jobs"])
        runtime.api.atomic_json(self.root / "budget.json", dict(limit_seconds=28800,
            charged_seconds=23566.954382242635, active=None, attempts=[dict(history="retained")]))
        self.budget = (self.root / "budget.json").read_bytes()
        self.addCleanup(patch.stopall)
        patch.object(runtime, "evidence", side_effect=lambda study, job: self.evidence[job["id"]]).start()
        patch.object(runtime, "_sp_config", return_value=dict(fixture="sp normal timing")).start()
        patch.object(runtime, "_vae_config", return_value=dict(fixture="VAE normal timing")).start()

    def control(self, identifier, config, result, capacity=True):
        job = runtime.api.add_job(self.study, identifier, "sp", config, module="fixture")
        job.update(source="/frozen/old", source_digest="old-digest")
        self.complete(job, result, capacity=capacity)

    def complete(self, job, result=None, *, capacity=True, qualified=True):
        result = dict(result or {}, status="completed", configuration=runtime.api.read(job["input"]))
        path = Path(job["output"]) / "run.json"
        runtime.api.atomic_json(path, result)
        launch = dict(status="completed", output_sha256=runtime.api.binding(path)["sha256"])
        runtime.api.atomic_json(self.root / "launches" / job["id"] / "launch.json", launch)
        self.evidence[job["id"]] = dict(guarded=True, qualified=qualified, capacity=capacity,
                                       result=result, path=path, launch=launch)

    def test_binds_controls_without_changing_prior_jobs_or_budget(self):
        sp = runtime.sp_job(self.study)
        vae = runtime.vae_job(self.study)
        self.assertEqual(self.study["jobs"][:4], self.old_jobs)
        self.assertEqual((self.root / "budget.json").read_bytes(), self.budget)
        self.assertEqual((sp["family"], vae["family"]), ("sp", "extension"))
        self.assertEqual((sp["devices"], vae["devices"]), ([0, 1], [1]))
        self.assertEqual(len(sp["input_bindings"]), 10)
        self.assertEqual(len(vae["input_bindings"]), 4)
        for job in (sp, vae):
            self.assertEqual(job["source"], "/frozen/new")
            self.assertEqual(job["input_bindings"], [runtime.api.binding(b["path"])
                             for b in job["input_bindings"]])
        before = copy.deepcopy(self.study["jobs"])
        runtime.sp_job(self.study)
        runtime.vae_job(self.study)
        self.assertEqual(self.study["jobs"], before)

    def test_sp_prerequisites_require_guard_numeric_and_capacity(self):
        for identifier in ("elastic-repaired-transfer", "elastic-repaired-pause", "elastic-repaired-resize"):
            for key in ("guarded", "qualified", "capacity"):
                with self.subTest(identifier=identifier, key=key):
                    self.evidence[identifier][key] = False
                    with self.assertRaisesRegex(ValueError, "qualified control"):
                        runtime.sp_job(self.study)
                    self.evidence[identifier][key] = True
                    self.assertEqual(self.study["jobs"], self.old_jobs)

    def test_vae_failed_third_diagnostic_does_not_waive_required_arm_gates(self):
        ev = self.evidence["overlap-isolated-warmed-lifecycle"]
        self.assertFalse(ev["capacity"])
        original = copy.deepcopy(ev["result"])
        for scope in ("numeric", "capacity", "steady_state"):
            ev["result"]["qualification"]["scopes"][scope]["status"] = "failed"
            with self.assertRaisesRegex(ValueError, "qualified fixed-warmup"):
                runtime.vae_job(self.study)
            ev["result"] = copy.deepcopy(original)
        ev["result"]["arms"][1]["launch"]["reaped"] = False
        with self.assertRaisesRegex(ValueError, "not reaped"):
            runtime.vae_job(self.study)

    def test_mutated_control_hash_or_matrix_cannot_register(self):
        ev = self.evidence["elastic-repaired-resize"]
        ev["launch"]["output_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "output changed"):
            runtime.sp_job(self.study)
        self.assertEqual(self.study["jobs"], self.old_jobs)

    def test_registered_source_command_and_dependencies_are_immutable(self):
        job = runtime.sp_job(self.study)
        original = copy.deepcopy(job)
        for key, value in (("source", "/other"), ("source_digest", "other"),
                           ("metadata", {}), ("command", ["different"]), ("input_bindings", [])):
            with self.subTest(key=key):
                job.clear()
                job.update(copy.deepcopy(original))
                job[key] = value
                with self.assertRaises(ValueError):
                    runtime.sp_job(self.study)

    def test_result_requires_exact_frozen_configuration_and_schema(self):
        job = runtime.sp_job(self.study)
        self.complete(job, dict(schema="unrelated.completed.schema"))
        with self.assertRaisesRegex(ValueError, "configuration or schema"):
            runtime._run(self.study, lambda *args: True, job)
        self.evidence[job["id"]]["result"]["schema"] = "ditserve.workflow.sp2_performance.v1"
        self.evidence[job["id"]]["result"]["configuration"] = {}
        with self.assertRaisesRegex(ValueError, "configuration or schema"):
            runtime._run(self.study, lambda *args: True, job)

    def test_new_vae_launch_capacity_remains_mandatory(self):
        job = runtime.vae_job(self.study)
        self.complete(job, dict(schema="ditserve.workflow.overlap-performance.v1"), capacity=False)
        observed = runtime._run(self.study, lambda *args: True, job)
        self.assertEqual(observed["status"], "unqualified_or_deferred")
        self.assertFalse(observed["capacity"])

    def test_runtime_cli_cannot_accidentally_select_other_execution_paths(self):
        path = Path(__file__).resolve().parents[2] / "docker/run-next-study.py"
        spec = importlib.util.spec_from_file_location("runtime_driver_test", path)
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        for arguments in (["--optimize-runtime"], ["--run", "--optimize-runtime", "--accept-sp2"],
                          ["--run", "--optimize-runtime", "--only", "acceptance"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                driver.main(arguments)


if __name__ == "__main__":
    unittest.main()
