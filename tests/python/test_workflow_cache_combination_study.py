"""CPU checks for a single-variable combination trial and immutable history."""

import copy
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_cache_combination_study as combination
from ditserve.evaluation import workflow_encoding_optimization_study as encoding
from ditserve.evaluation import workflow_next_study as api
from ditserve.evaluation import workflow_occupancy_study as occupancy
import test_workflow_occupancy_study as previous_tests


class CacheCombinationStudyTests(unittest.TestCase):
    def setUp(self):
        fixture = previous_tests.OccupancyStudyTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.study = fixture.root, fixture.study
        self.profile, self.rows, self.evidence = fixture.profile, fixture.rows, fixture.evidence
        self._reseal = fixture._reseal
        source = self.root / "sources/r016"
        shutil.copytree(Path(self.study["source"]), source)
        self._reseal(source)
        for repeat, seed in enumerate(occupancy.SEEDS):
            self._job(occupancy.job_id(seed), occupancy.native_config(self.profile, self.rows, repeat=repeat),
                dict(role="occupancy_online", stage="confirmation", seed=seed, repeat=repeat), wall=330. + repeat)
        older = next(r for r in self.study["source_revisions"] if Path(r["source"]).name == "r013")
        for distribution in ("unique", "repeated"):
            self._job(f"encoding-sana-{distribution}-numeric", dict(fixture="numeric"),
                dict(role="encoding_numeric", optimization="sana_prompt_cache", seed=4100,
                     prompt_distribution=distribution), old=older)
            for repeat, seed in enumerate(encoding.SANA_SEEDS):
                self._job(f"sana-prompt-cache-{distribution}-s{seed}", encoding.native_online_config(
                    self.profile, self.rows, optimization="sana_prompt_cache", prompt_distribution=distribution, repeat=repeat),
                    dict(role="encoding_online", optimization="sana_prompt_cache", seed=seed, repeat=repeat,
                         prompt_distribution=distribution), old=older)
        for job in self.study["jobs"]:
            job.setdefault("output", str(self.root / "measurements" / job["id"]))
        while len(self.study["jobs"]) < combination.HISTORICAL_JOB_COUNT:
            self.study["jobs"].append(dict(id="old-unused-" + str(len(self.study["jobs"])), input_bindings=[]))
        self.study["selections"]["occupancy"] = dict(qualified=True, promotable=True,
                                                     confirmation_seeds=list(occupancy.SEEDS))
        self.budget = dict(schema="ditserve.workflow.budget.v1", limit_seconds=None, charged_seconds=700.,
            attempts=[dict(id=str(i), job_id=self.study["jobs"][i]["id"], charged_seconds=10.) for i in range(70)],
            active=None, overrun_seconds=0., authorization_history=[dict(old_limit_seconds=28800.,
                new_limit_seconds=None, authorization="取消预算，GPU预算无限，继续运行", authorized_unix=1.,
                snapshot="budget-history/old.json", snapshot_sha256="a" * 64)])
        api.save(self.study)
        api.atomic_json(self.root / "budget.json", self.budget)
        checkpoint = self.root / combination.CHECKPOINT
        checkpoint.mkdir(parents=True)
        manifest = []
        for name in ("study.json", "budget.json"):
            shutil.copyfile(self.root / name, checkpoint / name)
            bound = api.binding(checkpoint / name)
            manifest.append(dict(source=str(self.root / name), copy=bound["path"],
                                 bytes=bound["bytes"], sha256=bound["sha256"]))
        api.atomic_json(checkpoint / "manifest.json", manifest)
        api.atomic_json(self.root / combination.REFERENCE_AUDIT,
            dict(schema="ditserve.maintenance.occupancy-final-integrity.v1", status="passed", errors=[],
                 history=dict(current_jobs=82, current_attempts=70),
                 source_seals=[dict(source=self.study["source"], source_digest=self.study["source_digest"])]))
        api.atomic_json(self.root / combination.DIAGNOSIS,
            dict(schema="ditserve.cache-combination-diagnosis.v1", combination_has_gpu_evidence=False))
        candidate = self.root / "sources/r017"
        shutil.copytree(source, candidate)
        planning = candidate / "workers/ditserve/evaluation/workflow_cache_combination_study.py"
        planning.parent.mkdir(parents=True, exist_ok=True)
        planning.write_text("new planning only")
        self._reseal(candidate)
        self.gate = patch.object(combination.occupancy_report, "screening_gate", return_value=dict(passed=True))
        self.gate.start()
        self.addCleanup(self.gate.stop)

    def _job(self, identifier, config, metadata, *, wall=330., old=None):
        job = api.add_job(self.study, identifier, "extension", config, native="phases" in config,
                          module="workflow_encoding_qualification", metadata=metadata)
        if old is not None:
            job.update(source=old["source"], source_digest=old["source_digest"])
        path = Path(job["output"]) / "run.json"
        launch_path = self.root / "launches" / identifier / "launch.json"
        result, launch = dict(status="completed"), dict(status="completed", wall_seconds=wall)
        api.atomic_json(path, result)
        api.atomic_json(launch_path, launch)
        self.evidence[identifier] = dict(guarded=True, qualified=True, capacity=True,
                                        path=path, result=result, launch=launch)

    def test_single_cache_variable_and_balanced_order(self):
        before = copy.deepcopy((self.profile, self.rows))
        for repeat, order in enumerate((combination.ARMS, combination.ARMS[::-1], combination.ARMS)):
            config = combination.native_config(self.profile, self.rows, repeat=repeat)
            self.assertTrue(combination.validate_config(config, repeat=repeat))
            self.assertEqual([p["id"] for p in config["phases"]], list(order))
            normalized = []
            for phase in config["phases"]:
                phase = copy.deepcopy(phase)
                name = phase.pop("id")
                self.assertEqual(phase["occupancy_backend"], "nvml")
                self.assertIs(phase["dispatch_refresh"], False)
                self.assertEqual(phase["text_prefetch_depth"], 1)
                self.assertEqual((phase["options"]["quantum_ms"], phase["options"]["decode_frames"]), (300, 1))
                for rank in (0, 1):
                    sana = phase["worker_overrides"][f"sana{rank}"]
                    self.assertEqual(sana.pop("options"), {} if name == "reference" else
                                     dict(encoding.SANA_CACHE, prompt_cache_sync_mode="staged"))
                    wan = phase["worker_overrides"][f"wan{rank}"]
                    self.assertEqual({k:wan["options"][k] for k in encoding.WAN_CACHE}, encoding.WAN_CACHE)
                    self.assertEqual(wan["cpu_threads"], 8)
                normalized.append(phase)
            self.assertEqual(normalized[0], normalized[1])
        self.assertEqual((self.profile, self.rows), before)

    def test_control_or_cache_drift_and_invalid_seed_rejected(self):
        mutations = [lambda c:c["phases"][0].update(occupancy_backend="nvidia_smi"),
            lambda c:c["phases"][0].update(dispatch_refresh=True),
            lambda c:c["phases"][1]["worker_overrides"]["sana0"]["options"].update(prompt_cache_sync_mode="fused"),
            lambda c:c["phases"][1]["worker_overrides"]["sana1"]["options"].update(prompt_cache_entries=8),
            lambda c:c["phases"][0]["worker_overrides"]["wan0"]["options"].pop("negative_prompt_cache_entries"),
            lambda c:c["phases"][0]["workload"][0].update(arrival_ms=999),
            lambda c:c["phases"][0]["options"].update(decode_frames=True)]
        for mutation in mutations:
            config = combination.native_config(self.profile, self.rows)
            mutation(config)
            with self.assertRaises(ValueError):
                combination.validate_config(config, repeat=0)
        for seed in (True, 4400, 4504, 4500.0):
            with self.assertRaises(ValueError):
                combination.job_id(seed)

    def test_plan_registration_and_resume_preserve_history_and_ledger(self):
        history, budget = copy.deepcopy(self.study["jobs"]), (self.root / "budget.json").read_bytes()
        plan = combination.build_plan(self.study)
        self.assertEqual(self.study["jobs"], history)
        self.assertEqual([j["metadata"]["seed"] for j in plan["jobs"]], [4500, 4501, 4502, 4503])
        self.assertEqual(plan["history"]["attempts"], 70)
        bindings = {b["path"] for b in plan["input_bindings"]}
        self.assertIn(str(self.root / combination.REFERENCE_AUDIT), bindings)
        self.assertIn(str(self.root / combination.DIAGNOSIS), bindings)
        self.assertNotIn(str(self.root / "budget.json"), bindings)
        for template in plan["jobs"]:
            self.assertEqual(template["estimated_seconds"], 400)
            self.assertEqual(template["metadata"]["role"], "cache_combination_online")
            self.assertEqual(combination.job_id(4500) in template["requires"], template["metadata"]["stage"] == "confirmation")
            self.assertTrue(set(combination.REFERENCE_JOBS + combination.SANA_REFERENCES) <= set(template["requires"]))
        jobs = combination.register(self.study)
        self.assertEqual(self.study["jobs"][:82], history)
        self.assertEqual((self.root / "budget.json").read_bytes(), budget)
        before = copy.deepcopy(self.study)
        self.assertEqual(combination.register(self.study), jobs)
        self.assertEqual(self.study, before)
        # A real completed charge may append without invalidating frozen inputs.
        self.budget["attempts"].append(dict(id="new", job_id=jobs[0]["id"], charged_seconds=123.))
        self.budget["charged_seconds"] += 123.
        api.atomic_json(self.root / "budget.json", self.budget)
        self.assertEqual(combination.register(self.study), jobs)

    def test_history_job_attempt_authorization_and_charge_tamper_rejected(self):
        mutations = [lambda:self.study["jobs"][0].update(decision="rewritten"),
            lambda:self.budget["attempts"][0].update(charged_seconds=11.),
            lambda:self.budget["authorization_history"][0].update(authorization="changed"),
            lambda:self.budget.update(charged_seconds=699.),
            lambda:self.budget.update(limit_seconds=28800.)]
        baseline_study, baseline_budget = copy.deepcopy(self.study), copy.deepcopy(self.budget)
        for mutation in mutations:
            mutation()
            api.atomic_json(self.root / "budget.json", self.budget)
            with self.assertRaises(ValueError):
                combination.build_plan(self.study)
            self.study.clear()
            self.study.update(copy.deepcopy(baseline_study))
            self.budget = copy.deepcopy(baseline_budget)
        api.atomic_json(self.root / "budget.json", self.budget)

    def test_protected_source_drift_rejected_but_planning_change_allowed(self):
        source = Path(self.study["source"])
        for name in ("workers/ditserve/models/wan/backend.py", "workers/ditserve/runtime/worker.py",
                     "src/ditserve/control/study.cpp"):
            path = source / name
            original = path.read_bytes()
            path.write_text("unexpected execution change")
            self._reseal(source)
            with self.assertRaisesRegex(ValueError, "sources changed from r016"):
                combination.build_plan(self.study)
            path.write_bytes(original)
            self._reseal(source)
        self.assertTrue(combination.build_plan(self.study)["jobs"])

    def test_failed_numeric_anchor_or_performance_gate_prevents_registration(self):
        before = copy.deepcopy(self.study)
        for identifier in (combination.SANA_NUMERIC[0], combination.SANA_REFERENCES[-1], combination.REFERENCE_JOBS[1]):
            self.evidence[identifier]["qualified"] = False
            with self.assertRaisesRegex(ValueError, "qualified"):
                combination.register(self.study)
            self.assertEqual(self.study, before)
            self.evidence[identifier]["qualified"] = True
        with patch.object(combination.occupancy_report, "screening_gate", return_value=dict(passed=False)):
            with self.assertRaisesRegex(ValueError, "not promotable"):
                combination.register(self.study)
        self.assertEqual(self.study, before)

    def test_input_mutation_during_evidence_read_is_rejected(self):
        for name in ("profile.json", "budget.json"):
            target = self.root / name
            original = target.read_bytes()
            def evidence(study, job):
                target.write_text(target.read_text() + "\n")
                return self.evidence[job["id"]]
            with patch.object(combination.report, "evidence", side_effect=evidence):
                with self.assertRaisesRegex(ValueError, "changed during validation"):
                    combination.build_plan(self.study)
            target.write_bytes(original)

    def test_resume_command_change_never_appends(self):
        jobs = combination.register(self.study)
        jobs[-1]["command"] = ["other"]
        before = copy.deepcopy(self.study)
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            combination.register(self.study)
        self.assertEqual(self.study, before)

    def test_later_input_collision_is_rejected_before_first_append(self):
        path = self.root / "inputs" / (combination.job_id(4503) + ".json")
        path.write_text("unrelated existing input")
        before = copy.deepcopy(self.study)
        with self.assertRaises(FileExistsError):
            combination.register(self.study)
        self.assertEqual(self.study, before)
        self.assertFalse((self.root / "inputs" / (combination.job_id(4500) + ".json")).exists())

    def test_checkpoint_selection_or_manifest_tamper_is_rejected(self):
        path = self.root / combination.CHECKPOINT / "study.json"
        checkpoint = api.read(path)
        checkpoint["selections"]["occupancy"]["promotable"] = False
        api.atomic_json(path, checkpoint)
        with self.assertRaisesRegex(ValueError, "manifest"):
            combination.build_plan(self.study)
        manifest_path = path.parent / "manifest.json"
        manifest = api.read(manifest_path)
        bound = api.binding(path)
        for row in manifest:
            if row["copy"] == str(path):
                row.update(bytes=bound["bytes"], sha256=bound["sha256"])
        api.atomic_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "qualified and promotable"):
            combination.build_plan(self.study)


if __name__ == "__main__":
    unittest.main()
