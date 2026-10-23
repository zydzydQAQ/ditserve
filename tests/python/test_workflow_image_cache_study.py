"""Bounded Sana cache screening keeps frozen controls and fails closed."""

import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from ditserve.evaluation import workflow_image_cache_study as cache
from ditserve.evaluation import workflow_next_report as report
from ditserve.evaluation import workflow_next_study as api
from ditserve.evaluation.workflow_budget import atomic_json
import test_workflow_adaptive_study as adaptive_fixture


class ImageCacheStudyTests(unittest.TestCase):
    def setUp(self):
        # Composition deliberately avoids collecting the adaptive tests twice.
        self.fixture = adaptive_fixture.AdaptiveStudyTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.study = self.fixture.root, self.fixture.study
        self.audit = Mock(return_value=dict(passed=True, scope="unit fixture"))
        module = types.ModuleType("ditserve.evaluation.workflow_image_cache_audit")
        module.audit = self.audit
        replacement = patch.dict(sys.modules, {module.__name__: module})
        replacement.start()
        self.addCleanup(replacement.stop)

    def observed(self, job, *, image_scale=1., video_scale=1., throughput_scale=1., slo_loss=False):
        value = self.fixture.observed(job)
        for phase in value["phases"]:
            candidate = phase["id"] == "candidate"
            cell = phase["cells"][0]
            cell["elapsed_ms"] = 150000. / (throughput_scale if candidate else 1.)
            images = []
            for row in cell["requests"]:
                latency = (500. * (image_scale if candidate else 1.) if row["kind"] == "image"
                           else 100000. * (video_scale if candidate else 1.))
                row["finish_ms"] = row["arrival_ms"] + latency
                if row["kind"] == "image":
                    images.append(row)
            if candidate and slo_loss:
                images[0]["finish_ms"] = images[0]["deadline_ms"] + 1.
        return value

    def completed(self, **options):
        jobs = cache.cache_jobs(self.study)
        for job in jobs:
            self.fixture.result(job, self.observed(job, **options), wall=350.)
        return jobs

    @staticmethod
    def arm(value, name):
        return next(phase for phase in value["phases"] if phase["id"] == name)["cells"][0]

    def test_registers_ab_ba_with_only_cache_treatment_and_480_second_extension_budget(self):
        old_jobs = copy.deepcopy(self.study["jobs"])
        old_selections = copy.deepcopy(self.study["selections"])
        atomic_json(self.root / "budget.json", dict(charged_seconds=26752.7, active=None))
        old_budget = (self.root / "budget.json").read_bytes()
        jobs = cache.cache_jobs(self.study)
        self.assertEqual(self.study["jobs"][:len(old_jobs)], old_jobs)
        self.assertEqual(self.study["selections"], old_selections)
        self.assertEqual((self.root / "budget.json").read_bytes(), old_budget)
        self.assertEqual(cache.SEEDS, (3901, 3902))
        orders, traces = [], []
        for job, seed in zip(jobs, cache.SEEDS):
            self.assertEqual(job["id"], f"image-prompt-cache-s{seed}")
            self.assertEqual((job["family"], job["devices"], job["timeout_seconds"]),
                             ("extension", [0, 1], 480))
            self.assertLessEqual(job["estimated_seconds"], 370)
            self.assertEqual(job["requires"], ["acceptance", "calibration-video", "calibration-mixed",
                "contention-s3701", "contention-s3702", "contention-s3703"])
            self.assertFalse(job["metadata"]["independent_confirmation"])
            self.assertEqual(job["metadata"]["cache"], dict(prompt_cache_entries=4,
                                                         prompt_cache_bytes=16 * 1024**2))
            config = api.read(job["input"])
            self.assertEqual(config["budget"], dict(limit_gpu_seconds=960, watchdog_seconds=480,
                                                  repair_reserve_gpu_seconds=0, carry="none"))
            phases = config["phases"]
            orders.append([phase["id"] for phase in phases])
            reference = next(phase for phase in phases if phase["id"] == "reference")
            candidate = next(phase for phase in phases if phase["id"] == "candidate")
            normalized = copy.deepcopy(candidate)
            normalized["id"] = "reference"
            for rank in range(2):
                name = f"sana{rank}"
                self.assertEqual(candidate["worker_overrides"][name]["options"], cache.CACHE_OPTIONS)
                self.assertEqual(reference["worker_overrides"][name]["options"], {})
                self.assertEqual(candidate["worker_overrides"][name]["cpu_affinity"],
                                 list(range(60 * rank + 16, 60 * rank + 20)))
                self.assertEqual(candidate["worker_overrides"][name]["cpu_threads"], 4)
                normalized["worker_overrides"][name]["options"] = {}
            self.assertEqual(normalized, reference)
            self.assertEqual(reference["options"]["quantum_ms"], 300)
            self.assertEqual(reference["options"]["decode_frames"], 1)
            self.assertEqual(reference["text_prefetch_depth"], 1)
            self.assertTrue(reference["record_dispatch_timeline"])
            self.assertEqual(reference["controller_cpu_affinity"], [59])
            rows = reference["workload"]
            self.assertEqual(len(rows), 24)
            images = [row for row in rows if row["kind"] == "image"]
            self.assertEqual(len(images), 20)
            self.assertEqual(len({row["spec"]["prompt"] for row in images}), 4)
            self.assertEqual(job["metadata"]["trace_sha256"], api.digest(rows))
            self.assertEqual(job["input_bindings"], [api.binding(bound["path"])
                                                   for bound in job["input_bindings"]])
            traces.append(rows)
        self.assertEqual(orders, [["reference", "candidate"], ["candidate", "reference"]])
        self.assertNotEqual(traces[0], traces[1])

    def test_idempotent_registration_preserves_frozen_files_and_jobs(self):
        jobs = cache.cache_jobs(self.study)
        before = copy.deepcopy(self.study["jobs"])
        contents = {bound["path"]: Path(bound["path"]).read_bytes()
                    for job in jobs for bound in job["input_bindings"]}
        self.assertEqual(cache.cache_jobs(self.study), jobs)
        self.assertEqual(self.study["jobs"], before)
        for path, value in contents.items():
            self.assertEqual(Path(path).read_bytes(), value)

    def test_registered_source_command_budget_and_dependencies_cannot_change(self):
        job = cache.cache_jobs(self.study)[0]
        original = copy.deepcopy(job)
        mutations = dict(source="/other", source_digest="other", metadata={}, estimated_seconds=1,
            timeout_seconds=481, command=["different"], command_type="python", input_bindings=[],
            requires=["acceptance"], family="confirmation", devices=[1])
        for key, value in mutations.items():
            with self.subTest(key=key):
                job.clear()
                job.update(copy.deepcopy(original))
                job[key] = value
                with self.assertRaises(ValueError):
                    cache.cache_jobs(self.study)
        job.clear()
        job.update(original)
        self.study["source_digest"] = "new-revision"
        with self.assertRaisesRegex(ValueError, "frozen image cache"):
            cache.cache_jobs(self.study)

    def test_frozen_input_and_bound_environment_cannot_change(self):
        job = cache.cache_jobs(self.study)[0]
        original = Path(job["input"]).read_bytes()
        config = api.read(job["input"])
        config["phases"][0]["options"]["quantum_ms"] = 100
        atomic_json(job["input"], config)
        with self.assertRaisesRegex(ValueError, "frozen input"):
            cache.cache_jobs(self.study)
        with self.assertRaisesRegex(ValueError, "frozen input"):
            cache.summarize(self.study)
        Path(job["input"]).write_bytes(original)
        environment = api.read(job["environment_input"])
        environment["name"] = "changed"
        atomic_json(job["environment_input"], environment)
        with self.assertRaisesRegex(ValueError, "dependencies"):
            cache.cache_jobs(self.study)
        with self.assertRaisesRegex(ValueError, "dependency changed"):
            cache.summarize(self.study)

    def test_calibration_and_historical_capacity_are_required(self):
        old_jobs = copy.deepcopy(self.study["jobs"])
        path = self.root / "calibration.json"
        original = path.read_bytes()
        calibration = api.read(path)
        calibration["profile"] = {}
        atomic_json(path, calibration)
        with self.assertRaisesRegex(ValueError, "unchanged calibration"):
            cache.cache_jobs(self.study)
        Path(path).write_bytes(original)
        path = self.root / "launches/contention-s3701/launch.json"
        launch = api.read(path)
        launch["gpu_monitor"]["memory"]["0"]["minimum_free_bytes"] = 0
        atomic_json(path, launch)
        with self.assertRaisesRegex(ValueError, "qualified control"):
            cache.cache_jobs(self.study)
        self.assertEqual(self.study["jobs"], old_jobs)

    def test_pair_estimate_cannot_exceed_predeclared_bound(self):
        old_jobs = copy.deepcopy(self.study["jobs"])
        with patch.object(cache.contention, "_estimate", return_value=(371., {})):
            with self.assertRaisesRegex(ValueError, "370-second"):
                cache.cache_jobs(self.study)
        self.assertEqual(self.study["jobs"], old_jobs)

    def test_exact_complete_equal_arms_qualify_without_inventing_benefit(self):
        self.completed()
        result = cache.summarize(self.study)
        self.assertTrue(result["qualified"], result)
        self.assertEqual(result["median_image_p95_reduction_fraction"], 0.)
        self.assertFalse(result["benefit_screened"])
        self.assertEqual(result["recommendation"], "retain_cache_disabled")
        self.assertIn("no unique-prompt or A100", result["scope"])
        self.assertEqual(len(result["jobs"]), 2)
        for row in result["jobs"]:
            self.assertTrue(row["comparison"]["complete_frozen_requests"])
            self.assertTrue(row["comparison"]["numeric"]["passed"])
            self.assertEqual(row["comparison"]["numeric"]["requests"], 24)
            self.assertEqual(row["result_binding"], api.binding(row["result_binding"]["path"]))

    def test_positive_screen_uses_two_seed_median_and_requires_confirmation(self):
        jobs = self.completed(image_scale=.80)
        self.fixture.result(jobs[1], self.observed(jobs[1], image_scale=.96), wall=350.)
        result = cache.summarize(self.study)
        self.assertTrue(result["qualified"], result)
        self.assertAlmostEqual(result["median_image_p95_reduction_fraction"], .12)
        self.assertTrue(result["benefit_screened"])
        self.assertEqual(result["recommendation"], "confirm_repeated_prompt_cache")
        self.assertNotIn("benefit_confirmed", result)
        self.fixture.result(jobs[0], self.observed(jobs[0], image_scale=.87), wall=350.)
        result = cache.summarize(self.study)
        self.assertAlmostEqual(result["median_image_p95_reduction_fraction"], .085)
        self.assertFalse(result["benefit_screened"])

    def test_per_trace_throughput_video_tail_and_image_slo_safety_bounds(self):
        jobs = self.completed(image_scale=.80)
        for bound, options in (("throughput", dict(throughput_scale=.949)),
                               ("video_p95", dict(video_scale=1.051)),
                               ("image_slo", dict(slo_loss=True))):
            with self.subTest(bound=bound):
                self.fixture.result(jobs[0], self.observed(jobs[0], image_scale=.80, **options), wall=350.)
                result = cache.summarize(self.study)
                self.assertTrue(result["qualified"], result)
                self.assertFalse(result["jobs"][0]["safety_bounds"][bound])
                self.assertFalse(result["benefit_screened"])
                self.assertEqual(result["recommendation"], "retain_cache_disabled")

    def test_real_pair_rejects_changed_hash_and_both_arms_missing_same_request(self):
        jobs = self.completed(image_scale=.80)
        for mutation in ("hash", "missing", "failed"):
            with self.subTest(mutation=mutation):
                value = self.observed(jobs[0], image_scale=.80)
                left, right = self.arm(value, "reference"), self.arm(value, "candidate")
                if mutation == "hash":
                    right["requests"][0]["outputs"]["video"]["sha256"] = "different"
                elif mutation == "missing":
                    left["requests"].pop()
                    right["requests"].pop()
                else:
                    right["requests"][0]["status"] = "failed"
                self.fixture.result(jobs[0], value, wall=350.)
                result = cache.summarize(self.study)
                self.assertFalse(result["qualified"])
                self.assertFalse(result["benefit_screened"])
                self.assertFalse(result["jobs"][0]["comparison"]["eligible"])
                if mutation == "missing":
                    self.assertTrue(result["jobs"][0]["comparison"]["numeric"]["passed"])
                    self.assertFalse(result["jobs"][0]["comparison"]["complete_frozen_requests"])

    def test_new_launch_guard_capacity_and_cache_audit_are_mandatory(self):
        jobs = self.completed(image_scale=.80)
        path = self.root / "launches" / jobs[0]["id"] / "launch.json"
        original = api.read(path)
        for mutation in ("capacity", "monitor_error", "conflict", "output_hash", "status"):
            with self.subTest(mutation=mutation):
                launch = copy.deepcopy(original)
                if mutation == "capacity":
                    launch["gpu_monitor"]["memory"]["1"]["minimum_free_bytes"] = 0
                elif mutation == "monitor_error":
                    launch["gpu_monitor"]["errors"] = ["unavailable"]
                elif mutation == "conflict":
                    launch["gpu_monitor"]["conflicts"] = ["other process"]
                elif mutation == "output_hash":
                    launch["output_sha256"] = "changed"
                else:
                    launch["status"] = "interrupted"
                atomic_json(path, launch)
                result = cache.summarize(self.study)
                self.assertFalse(result["qualified"])
                self.assertFalse(result["benefit_screened"])
        atomic_json(path, original)
        self.audit.return_value = dict(passed=False, reason="cache provenance missing")
        result = cache.summarize(self.study)
        self.assertFalse(result["qualified"])
        self.assertFalse(result["benefit_screened"])

    def test_report_binds_schema_order_and_all_phase_controls_to_frozen_input(self):
        jobs = self.completed(image_scale=.80)
        for mutation in ("schema", "options", "cpu", "cache_limit", "warmup", "order"):
            with self.subTest(mutation=mutation):
                value = self.observed(jobs[0], image_scale=.80)
                phase = value["config"]["phases"][1]
                if mutation == "schema":
                    value["schema"] = "unknown.completed.schema"
                elif mutation == "options":
                    phase["options"]["quantum_ms"] = 100
                elif mutation == "cpu":
                    phase["worker_overrides"]["sana0"]["cpu_threads"] = 8
                elif mutation == "cache_limit":
                    phase["worker_overrides"]["sana0"]["options"]["prompt_cache_entries"] = 8
                elif mutation == "warmup":
                    phase["warmup"] = []
                else:
                    value["config"]["phases"].reverse()
                self.fixture.result(jobs[0], value, wall=350.)
                self.assertFalse(report.evidence(self.study, jobs[0])["qualified"])
                self.assertFalse(cache.summarize(self.study)["benefit_screened"])

    def test_missing_unrun_and_failed_first_trace_do_not_promote_or_block_second_trace(self):
        result = cache.summarize(self.study)
        self.assertFalse(result["qualified"])
        jobs = cache.cache_jobs(self.study)
        result = cache.summarize(self.study)
        self.assertFalse(result["qualified"])
        self.assertIsNone(result["median_image_p95_reduction_fraction"])
        before = copy.deepcopy(self.study["selections"])
        seen = []
        def run(study, job):
            seen.append(job["id"])
            value = self.observed(job, image_scale=.80)
            if job["id"] == jobs[0]["id"]:
                value["status"] = "failed"
            self.fixture.result(job, value, wall=350.)
        result = cache.run_followup(self.study, run)
        self.assertEqual(seen, [job["id"] for job in jobs])
        self.assertFalse(result["qualified"])
        self.assertTrue(result["jobs"][1]["qualified"])
        self.assertEqual(self.study["selections"][cache.SELECTION], result)
        self.assertEqual({key: value for key, value in self.study["selections"].items()
                          if key != cache.SELECTION}, before)

    def test_cli_dispatches_only_requested_mode_and_rejects_mixed_modes(self):
        path = Path(__file__).resolve().parents[2] / "docker/run-next-study.py"
        spec = importlib.util.spec_from_file_location("image_cache_driver_test", path)
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        api.save(self.study)
        with patch.object(driver, "execute") as execute:
            driver.main(["--root", str(self.root), "--run", "--optimize-image-cache"])
        execute.assert_called_once()
        self.assertTrue(execute.call_args.kwargs["optimize_image_cache"])
        self.assertFalse(any(value for key, value in execute.call_args.kwargs.items()
                             if key != "optimize_image_cache"))
        combinations = [["--optimize-image-cache"]] + [
            ["--run", "--optimize-image-cache", *other] for other in (
                ["--diagnose-sp"], ["--optimize-followup"], ["--accept-sp2"],
                ["--optimize-runtime"], ["--only", "acceptance"], ["--retry-block", "acceptance"])]
        for arguments in combinations:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as raised:
                driver.main(arguments)
            self.assertEqual(raised.exception.code, 2)
        for options in (dict(retry="acceptance"), dict(only="acceptance"), dict(diagnose_sp=True),
                        dict(optimize_followup=True), dict(accept_sp2=True), dict(optimize_runtime=True)):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "another execution mode"):
                driver.execute({}, optimize_image_cache=True, **options)


if __name__ == "__main__":
    unittest.main()
