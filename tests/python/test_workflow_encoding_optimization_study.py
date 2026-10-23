"""CPU-only encoding plans preserve paired controls and the cumulative budget."""

import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_encoding_optimization_study as encoding
from ditserve.evaluation import workflow_next_study as api
from ditserve.evaluation import workflow_encoding_qualification as qualification


class EncodingOptimizationStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = dict(run_id="run03", root="/data/workflow/run03", source="/frozen/r012",
            source_digest="old-source", jobs=[dict(id="historical", family="extension")],
            budget_seconds=28800, family_caps_seconds=copy.deepcopy(api.CAPS), active_job=None)
        self.budget = dict(limit_seconds=28800, charged_seconds=27403.253727,
                          attempts=[dict(job_id="historical", charged_seconds=27403.253727)], active=None)
        repo = Path(__file__).resolve().parents[2]
        self.profile = api.read(repo / "configs/profiles/rtx4090-sp1-local.json")
        for frames in (17, 33):
            self.profile["classes"][f"wan1.3b-480x832x{frames}"]["steps"] = 20
        self.rows = []
        for index in range(24):
            frames = (17, 33, 17, 33)[index] if index < 4 else None
            spec = dict(prompt="original prompt", seed=410100 + index)
            if frames:
                spec.update(height=480, width=832, frames=frames, steps=20,
                            guidance_scale=5., negative_prompt="")
            self.rows.append(dict(request_id=f"r{index}", kind="video" if frames else "image",
                **{"class": f"wan1.3b-480x832x{frames}" if frames else "sana-1024"},
                spec=spec, arrival_ms=index * 100., deadline_ms=index * 100. + 100000.))

    def test_current_budget_defers_complete_matrix_without_mutation(self):
        before = copy.deepcopy((self.study, self.budget))
        plan = encoding.build_plan(self.study, self.budget)
        self.assertEqual((self.study, self.budget), before)
        self.assertEqual(plan, encoding.build_plan(self.study, self.budget))
        self.assertEqual(plan["status"], "deferred_budget")
        self.assertFalse(plan["ready_for_gpu"])
        self.assertAlmostEqual(plan["budget"]["remaining_seconds"], 1396.746273)
        self.assertAlmostEqual(plan["budget"]["spendable_seconds"], 496.746273)
        self.assertAlmostEqual(plan["budget"]["family_available_seconds"]["extension"], 496.746273)
        self.assertEqual(plan["budget"]["reserve_seconds"], 900)
        total = plan["stage_costs"]["complete_matrix"]
        self.assertGreater(total["additional_available_budget_range_seconds"][0], 0)
        self.assertEqual(total["admission_range_seconds"][1],
                         sum(job["estimated_seconds"] * 1.2 + 90 for job in plan["jobs"]))
        for job in plan["jobs"]:
            self.assertNotIn("command", job)
            self.assertFalse(job["ready_for_gpu"])
            self.assertEqual(job["budget_status"], "deferred_budget")

    def test_qualification_is_separate_and_confirmation_uses_three_new_seeds(self):
        plan = encoding.build_plan(self.study, self.budget)
        numeric = [job for job in plan["jobs"] if job["stage"] == "qualification"]
        self.assertEqual(len(numeric), 4)
        for job in numeric:
            self.assertEqual(job["execution_status"], "configuration_template")
            self.assertEqual(job["runner"], "ditserve.evaluation.workflow_encoding_qualification")
            self.assertEqual(job["devices"], [1])
            self.assertEqual(job["qualification_status"], "unverified")
            self.assertFalse(job["config"]["performance_evidence"])
        for optimization, seeds in (("sana_prompt_cache", encoding.SANA_SEEDS),
                                    ("wan_negative_cache", encoding.WAN_SEEDS)):
            online = [job for job in plan["jobs"] if job["stage"] == "online_confirmation"
                      and job["optimization"] == optimization]
            self.assertEqual({job["config"]["seed"] for job in online}, set(seeds))
            required = {job["id"] for job in numeric if job["optimization"] == optimization}
            for job in online:
                self.assertTrue(required.issubset(job["requires"]))
        wan = [job["config"] for job in numeric if job["optimization"] == "wan_negative_cache"]
        self.assertEqual({case["guidance_scale"] for case in wan}, {1., 5.})
        for case in wan:
            self.assertEqual(case["frames"], [17, 33])
            self.assertEqual(case["negative_prompts"], ["", encoding.NEGATIVE_PROMPT])
            self.assertEqual(case["cache_options"], encoding.WAN_CACHE)
            self.assertEqual(case["negative_embedding_bytes"], 512 * 4096 * 4)

    def test_numeric_configuration_binds_uuid_cpu_and_fixed_runner_cases(self):
        uuid = "GPU-00000000-0000-0000-0000-000000000001"
        plan = encoding.build_plan(self.study, self.budget)
        numeric = [job for job in plan["jobs"] if job["stage"] == "qualification"]
        before = copy.deepcopy(numeric)
        for job in numeric:
            config = encoding.native_qualification_config(job, gpu_uuid=uuid)
            self.assertEqual(config["device"], "cuda:0")
            self.assertEqual(config["gpu_uuid"], uuid)
            self.assertEqual(config["cpu_mask"], list(range(60, 68)))
            self.assertEqual(config["cpu_threads"], 8)
            self.assertEqual(config["sample_interval"], .05)
            self.assertEqual(config["arm_timeout_seconds"], 1800)
            if config["kind"] == "sana":
                self.assertEqual(config["model_path"], "/models/sana-sprint-0.6b")
                self.assertEqual(config["prompt_distribution"], job["config"]["prompt_distribution"])
            else:
                self.assertEqual(config["model_path"], "/models/wan2.1-t2v-1.3b")
                self.assertEqual(config["guidance_scale"], job["config"]["guidance_scale"])
        self.assertEqual(numeric, before)

    def test_numeric_configuration_rejects_device_or_case_drift(self):
        uuid = "GPU-00000000-0000-0000-0000-000000000001"
        jobs = encoding.build_plan(self.study, self.budget)["jobs"]
        for job in (jobs[-1], dict(jobs[0], devices=[0, 1])):
            with self.assertRaises(ValueError):
                encoding.native_qualification_config(job, gpu_uuid=uuid)
        with self.assertRaisesRegex(ValueError, "UUID"):
            encoding.native_qualification_config(jobs[0], gpu_uuid="1")
        for index, field, value in ((0, "requests", 9), (2, "steps", 4),
                                    (2, "guidance_scale", True)):
            changed = copy.deepcopy(jobs[index])
            changed["config"][field] = value
            with self.assertRaises(ValueError):
                encoding.native_qualification_config(changed, gpu_uuid=uuid)

    def test_budget_uses_family_prefix_and_unattributed_charges_conservatively(self):
        self.study["family_caps_seconds"] = dict(early=100, extension=200, later=10000, reserve=900)
        self.budget.update(charged_seconds=50, attempts=[])
        plan = encoding.build_plan(self.study, self.budget)
        self.assertEqual(plan["budget"]["family_available_seconds"]["extension"], 250)
        self.assertEqual(plan["budget"]["unattributed_charged_seconds"], 50)

    def test_active_attempt_never_becomes_admitted(self):
        self.budget["active"] = dict(job_id="other")
        plan = encoding.build_plan(self.study, self.budget)
        self.assertEqual(plan["status"], "deferred_active_budget")
        self.assertTrue(all(job["budget_status"] == "deferred_active_budget" for job in plan["jobs"]))

    def test_invalid_budget_cannot_reduce_reserve_or_reset_ledger(self):
        for field, value in (("limit_seconds", 30000), ("charged_seconds", float("nan")),
                             ("charged_seconds", True), ("charged_seconds", -1)):
            with self.subTest(field=field, value=value):
                changed = dict(self.budget, **{field: value})
                with self.assertRaises(ValueError):
                    encoding.build_plan(self.study, changed)
        self.study["family_caps_seconds"]["reserve"] = 899
        with self.assertRaisesRegex(ValueError, "reserve"):
            encoding.build_plan(self.study, self.budget)

    def test_sana_three_arm_controls_orders_and_unique_distribution(self):
        before = copy.deepcopy((self.profile, self.rows))
        orders = []
        for repeat in range(3):
            config = encoding.native_online_config(self.profile, self.rows,
                optimization="sana_prompt_cache", prompt_distribution="unique", repeat=repeat)
            orders.append([phase["id"] for phase in config["phases"]])
            baseline = next(phase for phase in config["phases"] if phase["id"] == "disabled")
            prompts = [row["spec"]["prompt"] for row in baseline["workload"] if row["kind"] == "image"]
            self.assertEqual(len(set(prompts)), 20)
            warmup = [row["spec"]["prompt"] for row in baseline["warmup"]]
            self.assertFalse(set(prompts).intersection(warmup))
            for phase in config["phases"]:
                normalized = copy.deepcopy(phase)
                normalized["id"] = "disabled"
                for rank in (0, 1):
                    worker = normalized["worker_overrides"][f"sana{rank}"]
                    expected = {} if phase["id"] == "disabled" else dict(
                        encoding.SANA_CACHE, prompt_cache_sync_mode=phase["id"])
                    self.assertEqual(worker["options"], expected)
                    worker["options"] = {}
                self.assertEqual(normalized, baseline)
                self.assertEqual(phase["options"]["quantum_ms"], 300)
                self.assertEqual(phase["options"]["decode_frames"], 1)
                self.assertEqual(phase["text_prefetch_depth"], 1)
        self.assertEqual(orders, [["disabled", "staged", "fused"],
                                 ["staged", "fused", "disabled"], ["fused", "disabled", "staged"]])
        self.assertEqual((self.profile, self.rows), before)

    def test_repeated_sana_prompts_do_not_change_frozen_arrivals_or_video_specs(self):
        config = encoding.native_online_config(self.profile, self.rows,
            optimization="sana_prompt_cache", prompt_distribution="repeated", repeat=0)
        rows = config["phases"][0]["workload"]
        self.assertEqual(len({row["spec"]["prompt"] for row in rows if row["kind"] == "image"}), 4)
        for original, candidate in zip(self.rows, rows):
            for key in ("request_id", "arrival_ms", "deadline_ms"):
                self.assertEqual(candidate[key], original[key])
            self.assertEqual(candidate["spec"]["seed"], original["spec"]["seed"])
            if original["kind"] == "video":
                self.assertEqual(candidate, original)

    def test_wan_two_arm_controls_negative_coverage_and_fixed_cpu(self):
        before = copy.deepcopy(self.rows)
        for repeat in range(3):
            config = encoding.native_online_config(self.profile, self.rows,
                optimization="wan_negative_cache", prompt_distribution="mixed_negative", repeat=repeat)
            baseline = next(phase for phase in config["phases"] if phase["id"] == "disabled")
            candidate = next(phase for phase in config["phases"] if phase["id"] == "enabled")
            normalized = copy.deepcopy(candidate)
            normalized["id"] = "disabled"
            for rank in (0, 1):
                worker = normalized["worker_overrides"][f"wan{rank}"]
                self.assertEqual(worker["cpu_threads"], 8)
                self.assertEqual(worker["cpu_affinity"], list(range(60 * rank, 60 * rank + 8)))
                for key, value in encoding.WAN_CACHE.items():
                    self.assertEqual(worker["options"].pop(key), value)
                self.assertEqual(candidate["worker_overrides"][f"sana{rank}"]["options"], {})
            self.assertEqual(normalized, baseline)
            for frames in (17, 33):
                self.assertEqual({row["spec"]["negative_prompt"] for row in baseline["workload"]
                                  if row["kind"] == "video" and row["spec"]["frames"] == frames},
                                 {"", encoding.NEGATIVE_PROMPT})
        self.assertEqual(self.rows, before)

    def test_native_config_rejects_changed_recipe_or_incomplete_pair(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.assertRaises(ValueError):
                encoding.native_online_config(self.profile, rows,
                    optimization="sana_prompt_cache", prompt_distribution="unique", repeat=0)
        self.rows[0]["spec"]["steps"] = 4
        with self.assertRaisesRegex(ValueError, "twenty-step"):
            encoding.native_online_config(self.profile, self.rows,
                optimization="wan_negative_cache", prompt_distribution="mixed_negative", repeat=0)


class EncodingRegistrationTests(unittest.TestCase):
    def setUp(self):
        base = EncodingOptimizationStudyTests()
        base.setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name) / "run03"
        source = root / "sources/r013"
        (source / "configs/env").mkdir(parents=True)
        repo = Path(__file__).resolve().parents[2]
        shutil.copyfile(repo / "configs/env/rtx4090-2x24g.json", source / "configs/env/rtx4090-2x24g.json")
        self.study = dict(base.study, root=str(root), source=str(source), source_digest="frozen-new-source",
            gpu_mapping=[dict(ordinal=i, uuid=f"GPU-00000000-0000-0000-0000-00000000000{i}") for i in (0, 1)])
        self.rows, self.profile = base.rows, base.profile
        api.atomic_json(root / "profile.json", self.profile)
        api.atomic_json(root / "calibration.json", dict(profile=self.profile))
        api.save(self.study)
        api.atomic_json(root / "budget.json", base.budget)
        checkpoint = root / "checkpoints/before-encoding-unlimited-r012/work"
        checkpoint.mkdir(parents=True)
        for name in ("study.json", "budget.json"):
            shutil.copyfile(root / name, checkpoint / name)
        plan = encoding.build_plan(self.study, base.budget)
        self.proposal = Path(self.tmp.name) / "proposal"
        self.proposal.mkdir()
        plan["input_bindings"] = [api.binding(root / name) for name in
                                  ("study.json", "budget.json", "profile.json", "calibration.json")]
        for name, devices in (("environment.json", (0, 1)), ("qualification-environment.json", (1,))):
            api.atomic_json(self.proposal / name, api.environment(self.study, devices))
        plan["environment_bindings"] = [api.binding(self.proposal / name) for name in
                                       ("environment.json", "qualification-environment.json")]
        plan.update(qualification_configs=[], native_configs=[])
        for job in plan["jobs"]:
            numeric = job["stage"] == "qualification"
            config = job["config"]
            if numeric:
                identifier = job["id"]
                native = encoding.native_qualification_config(job, gpu_uuid=self.study["gpu_mapping"][1]["uuid"])
                native = qualification.normalize_config(native["kind"], native)
            else:
                identifier = "%s-%s-s%d" % (job["optimization"].replace("_", "-"),
                    config["prompt_distribution"], config["seed"])
                native = encoding.native_online_config(self.profile, self.rows, optimization=job["optimization"],
                    prompt_distribution=config["prompt_distribution"], repeat=config["repeat"])
                native["experiment"] = "encoding-optimization-" + identifier
                native["budget"] = dict(limit_gpu_seconds=3600, watchdog_seconds=1800,
                                        repair_reserve_gpu_seconds=0, carry="none")
                for phase in native["phases"]:
                    phase["artifact_dependencies"] = [dict(closed=False,
                        reason="encoding optimization numerical and performance review pending")]
            path = self.proposal / (identifier + ".json")
            api.atomic_json(path, native)
            plan["qualification_configs" if numeric else "native_configs"].append(
                dict(id=identifier, configuration_sha256=api.digest(native), binding=api.binding(path)))
        api.atomic_json(self.proposal / "plan.json", plan)
        self.study.update(budget_seconds=None, budget_mode="unlimited")
        self.plan = plan
        self.patches = [patch("ditserve.evaluation.workflow_contention_study._calibration", return_value=({}, [], [])),
                        patch("ditserve.evaluation.workflow_contention_study._anchors", return_value={}),
                        patch("ditserve.evaluation.workflow_contention_study.contention_trace", side_effect=lambda *a: copy.deepcopy(self.rows))]
        for mocked in self.patches:
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_register_unlimited_preserves_history_and_binds_source_inputs_commands(self):
        historical = copy.deepcopy(self.study["jobs"])
        jobs = encoding.register(self.study, self.proposal)
        self.assertEqual(len(jobs), 13)
        self.assertEqual(self.study["jobs"][:len(historical)], historical)
        self.assertEqual(self.study["budget_mode"], "unlimited")
        self.assertIsNone(self.study["budget_seconds"])
        self.assertEqual(jobs[:4], [j for j in jobs if j["metadata"]["role"] == "encoding_numeric"])
        for job in jobs:
            self.assertEqual(job["source_digest"], "frozen-new-source")
            self.assertIn(str(self.proposal / "plan.json"), [b["path"] for b in job["input_bindings"]])
            self.assertNotIn(str(Path(self.study["root"]) / "budget.json"), [b["path"] for b in job["input_bindings"]])
            if job["metadata"]["role"] == "encoding_numeric":
                self.assertEqual(job["command_type"], "isolated_gpu")
                self.assertEqual(job["devices"], [1])
                self.assertEqual(job["gpu_mapping"], [self.study["gpu_mapping"][1]])
                self.assertEqual(job["command"][:2], ["/usr/bin/env", "CUDA_VISIBLE_DEVICES=" + self.study["gpu_mapping"][1]["uuid"]])
                self.assertEqual(job["timeout_seconds"], 5580 if "sana" in job["id"] else 3780)
            else:
                prerequisites = [j for j in jobs[:4] if j["metadata"]["optimization"] == job["metadata"]["optimization"]]
                self.assertEqual(job["metadata"]["numeric_prerequisites"], [j["id"] for j in prerequisites])
                self.assertTrue(all(j["id"] in job["requires"] for j in prerequisites))
        before = copy.deepcopy(self.study)
        self.assertEqual(encoding.register(self.study, self.proposal), jobs)
        self.assertEqual(self.study, before)

    def test_config_tampering_is_rejected_before_any_registration(self):
        entry = self.plan["native_configs"][-1]
        config = api.read(entry["binding"]["path"])
        config["phases"][0]["text_prefetch_depth"] = 2
        api.atomic_json(entry["binding"]["path"], config)
        with self.assertRaisesRegex(ValueError, "changed proposal"):
            encoding.register(self.study, self.proposal)
        self.assertEqual(len(self.study["jobs"]), 1)

    def test_forged_rebound_config_still_must_match_calibrated_controls(self):
        entry = self.plan["native_configs"][0]
        config = api.read(entry["binding"]["path"])
        config["phases"][0]["text_prefetch_depth"] = 2
        api.atomic_json(entry["binding"]["path"], config)
        entry.update(binding=api.binding(entry["binding"]["path"]), configuration_sha256=api.digest(config))
        api.atomic_json(self.proposal / "plan.json", self.plan)
        with self.assertRaisesRegex(ValueError, "fixed calibrated"):
            encoding.register(self.study, self.proposal)
        self.assertEqual(len(self.study["jobs"]), 1)

    def test_historical_job_and_checkpoint_changes_are_rejected(self):
        self.study["jobs"][0]["family"] = "cpu"
        with self.assertRaisesRegex(ValueError, "historical jobs"):
            encoding.register(self.study, self.proposal)
        self.study["jobs"][0]["family"] = "extension"
        path = Path(self.study["root"]) / "checkpoints/before-encoding-unlimited-r012/work/budget.json"
        api.atomic_json(path, dict(limit_seconds=0))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            encoding.register(self.study, self.proposal)


if __name__ == "__main__":
    unittest.main()
