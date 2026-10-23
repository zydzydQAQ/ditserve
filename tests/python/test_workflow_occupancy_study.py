"""New occupancy experiments retain historical jobs and model numerical evidence."""

import copy
from pathlib import Path
import shutil
import unittest

from ditserve.evaluation import workflow_dispatch_refresh_study as previous
from ditserve.evaluation import workflow_next_study as api
from ditserve.evaluation import workflow_occupancy_study as occupancy
import test_workflow_dispatch_refresh_study as previous_tests


class OccupancyStudyTests(unittest.TestCase):
    def setUp(self):
        fixture = previous_tests.DispatchRefreshStudyTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.study = fixture.root, fixture.study
        self.profile, self.rows, self.evidence = fixture.profile, fixture.rows, fixture.evidence
        reference = api.add_job(self.study, occupancy.REFERENCE_JOB, "extension",
            previous.native_config(self.profile, self.rows), native=True,
            metadata=dict(role="dispatch_refresh_online", seed=4300, repeat=0))
        result_path = self.root / "measurements" / reference["id"] / "run.json"
        launch_path = self.root / "launches" / reference["id"] / "launch.json"
        result, launch = dict(status="completed"), dict(status="completed", wall_seconds=480.)
        api.atomic_json(result_path, result)
        api.atomic_json(launch_path, launch)
        self.evidence[reference["id"]] = dict(guarded=True, qualified=True, capacity=True,
            path=result_path, result=result, launch=launch)
        for seed in (4301, 4302, 4303):
            self.study["jobs"].append(dict(id=f"wan-dispatch-refresh-s{seed}", metadata=dict(seed=seed),
                                           decision="dispatch_refresh_screening_not_promising"))
        source = self.root / "sources/r015"
        shutil.copytree(Path(self.study["source"]), source)
        (source / "workers/ditserve/runtime/worker.py").write_text("select occupancy backend")
        (source / "workers/ditserve/runtime/occupancy_nvml.py").write_text("fresh nvml process query")
        self._reseal(source)

    def _reseal(self, source):
        revision = api.seal(source)
        self.study["source_revisions"] = [r for r in self.study["source_revisions"] if r["source"] != str(source)] + [revision]
        self.study.update(source=str(source), source_digest=revision["source_digest"])
        api.atomic_json(source.parent / (source.name + ".binding.json"), revision)

    def test_only_probe_backend_differs_and_order_is_balanced(self):
        before = copy.deepcopy((self.profile, self.rows))
        for repeat, order in enumerate((occupancy.ARMS, occupancy.ARMS[::-1], occupancy.ARMS)):
            config = occupancy.native_config(self.profile, self.rows, repeat=repeat)
            self.assertTrue(occupancy.validate_config(config, repeat=repeat))
            self.assertEqual([p["id"] for p in config["phases"]], list(order))
            normalized = []
            for phase in config["phases"]:
                value = copy.deepcopy(phase)
                name = value.pop("id")
                self.assertEqual(value.pop("occupancy_backend"), occupancy.BACKENDS[name])
                self.assertIs(value["dispatch_refresh"], False)
                self.assertEqual(value["text_prefetch_depth"], 1)
                self.assertEqual(value["options"]["quantum_ms"], 300)
                self.assertEqual(value["options"]["decode_frames"], 1)
                self.assertEqual(len(value["workload"]), 24)
                for rank in (0, 1):
                    self.assertEqual(value["worker_overrides"][f"wan{rank}"]["options"]["negative_prompt_cache_entries"], 4)
                    self.assertEqual(value["worker_overrides"][f"wan{rank}"]["cpu_threads"], 8)
                normalized.append(value)
            self.assertEqual(normalized[0], normalized[1])
        self.assertEqual((self.profile, self.rows), before)

    def test_changed_backend_refresh_cpu_or_quantum_is_rejected(self):
        changes = [lambda p: p.update(occupancy_backend="unknown"), lambda p: p.update(dispatch_refresh=True),
                   lambda p: p.update(dispatch_refresh=0), lambda p: p["options"].update(quantum_ms=100),
                   lambda p: p["worker_overrides"]["wan0"].update(cpu_threads=4)]
        for change in changes:
            config = occupancy.native_config(self.profile, self.rows)
            change(config["phases"][0])
            with self.assertRaises(ValueError):
                occupancy.validate_config(config, repeat=0)

    def test_plan_and_append_only_resume_preserve_old_jobs_and_ledger(self):
        history, budget = copy.deepcopy(self.study["jobs"]), (self.root / "budget.json").read_bytes()
        plan = occupancy.build_plan(self.study)
        self.assertEqual(self.study["jobs"], history)
        self.assertEqual([j["metadata"]["seed"] for j in plan["jobs"]], [4400, 4401, 4402, 4403])
        self.assertEqual([j["metadata"]["stage"] for j in plan["jobs"]], ["screening"] + ["confirmation"] * 3)
        for template in plan["jobs"]:
            self.assertEqual(template["estimated_seconds"], 390)
            self.assertEqual(template["metadata"]["runtime_changed_files"], sorted(occupancy.RUNTIME_ALLOWLIST))
            self.assertEqual(occupancy.job_id(4400) in template["requires"], template["metadata"]["stage"] == "confirmation")
            self.assertIn(occupancy.REFERENCE_JOB, template["requires"])
        jobs = occupancy.register(self.study)
        self.assertEqual(len(jobs), 4)
        self.assertEqual(self.study["jobs"][:len(history)], history)
        self.assertEqual((self.root / "budget.json").read_bytes(), budget)
        before = copy.deepcopy(self.study)
        self.assertEqual(occupancy.register(self.study), jobs)
        self.assertEqual(self.study, before)
        jobs[-1]["command"] = ["other"]
        changed = copy.deepcopy(self.study)
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            occupancy.register(self.study)
        self.assertEqual(self.study, changed)

    def test_model_change_or_unlisted_runtime_change_requires_new_qualification(self):
        source = Path(self.study["source"])
        path = source / "workers/ditserve/models/wan/backend.py"
        original = path.read_text()
        path.write_text("changed math")
        self._reseal(source)
        with self.assertRaisesRegex(ValueError, "new model numerical"):
            occupancy.build_plan(self.study)
        path.write_text(original)
        (source / "workers/ditserve/runtime/client.py").write_text("unreviewed runtime change")
        self._reseal(source)
        with self.assertRaisesRegex(ValueError, "allowlist"):
            occupancy.build_plan(self.study)

    def test_failed_old_numeric_and_seed_reuse_are_rejected_before_append(self):
        before = copy.deepcopy(self.study)
        self.evidence[occupancy.NUMERIC_PREREQUISITES[0]]["qualified"] = False
        with self.assertRaises(ValueError):
            occupancy.register(self.study)
        self.assertEqual(self.study, before)
        self.evidence[occupancy.NUMERIC_PREREQUISITES[0]]["qualified"] = True
        self.study["jobs"][0]["metadata"]["seed"] = 4401
        with self.assertRaisesRegex(ValueError, "reuse"):
            occupancy.build_plan(self.study)


if __name__ == "__main__":
    unittest.main()
