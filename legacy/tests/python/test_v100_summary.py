"""Reporter-only checks; synthetic fixtures never start GPU measurements."""

import copy
import json
from pathlib import Path
import tempfile
import unittest


from ditserve.experiments import summarize_v100 as reporter


def case(arm, prompt="object", valid=True):
    baseline = arm == "none"
    return {"dtype": "float16", "stage": "formal", "prompt_id": prompt, "seed": 0, "arm": arm,
            "status": "completed" if valid else "running", "valid_completion": valid,
            "profile_file": "/mock/%s/%s/profile.json" % (prompt, arm),
            "config": {"model": "wan", "prompt": prompt, "seed": 0, "dtype": "float16"},
            "scheduler": {"class": "FlowMatchEulerDiscreteScheduler", "config": {
                "shift": 3.0, "sigmas": [1.0, 0.5, 0.0],
                "_use_default_values": ["shift", "invert_sigmas"] if baseline else ["invert_sigmas", "shift"]}},
            "revision": {"requested": "a" * 40},
            "metrics": {"e2e_ms": 120000.0 if baseline else 60000.0,
                        "cpu_prompt_encode_ms": 28000.0, "prompt_embeddings_h2d_ms": 5.0,
                        "diagnostic_transformer_cuda_sum_ms": 100000.0 if baseline else 20000.0,
                        "peak_allocated_bytes": (4000 if baseline else 4256) * 1024 * 1024,
                        "peak_reserved_bytes": (5000 if baseline else 5256) * 1024 * 1024},
            "issues": [], "repeats": []}


def summary_fixture(cases):
    pairs = reporter.make_pairs(cases, "float16")
    complete = [pair for pair in pairs if pair["status"] == "completed"]
    return {"manifest": {"status": "running"}, "environment": {}, "run_directory": "/mock/paired",
            "selected_dtype": "float16", "issues": [], "cases": cases, "pairs": pairs,
            "counts": {"attempted": len(cases), "not_attempted": 0,
                       "completed_pairs_selected_dtype": len(complete), "formal_completed_pairs": len(complete)},
            "formal_e2e_ratio_median": 2.0 if complete else None}


class SchedulerComparisonTests(unittest.TestCase):
    def test_unordered_default_metadata_does_not_exclude_a_real_pair_or_mutate_evidence(self):
        cases = [case("none"), case("cache-dit")]
        original = copy.deepcopy(cases)
        pair = reporter.make_pairs(cases, "float16")[0]
        self.assertEqual(pair["status"], "completed")
        self.assertEqual(pair["e2e_baseline_over_cache"], 2.0)
        self.assertEqual(pair["diagnostic_transformer_cuda_baseline_over_cache"], 5.0)
        self.assertEqual(cases, original)

    def test_actual_scheduler_parameter_change_still_rejects_pairing(self):
        cases = [case("none"), case("cache-dit")]
        cases[1]["scheduler"]["config"]["shift"] = 5.0
        pair = reporter.make_pairs(cases, "float16")[0]
        self.assertEqual(pair["status"], "incomplete")
        self.assertIn("paired scheduler differs", pair["issues"])
        self.assertNotIn("e2e_baseline_over_cache", pair)

    def test_other_scheduler_lists_remain_order_sensitive(self):
        cases = [case("none"), case("cache-dit")]
        cases[1]["scheduler"]["config"]["sigmas"].reverse()
        self.assertEqual(reporter.make_pairs(cases, "float16")[0]["status"], "incomplete")
        cases = [case("none"), case("cache-dit")]
        cases[0]["scheduler"]["extra"] = {"_use_default_values": ["a", "b"]}
        cases[1]["scheduler"]["extra"] = {"_use_default_values": ["b", "a"]}
        self.assertEqual(reporter.make_pairs(cases, "float16")[0]["status"], "incomplete")

    def test_default_membership_or_non_string_metadata_changes_are_not_ignored(self):
        cases = [case("none"), case("cache-dit")]
        cases[1]["scheduler"]["config"]["_use_default_values"].append("new_default")
        self.assertEqual(reporter.make_pairs(cases, "float16")[0]["status"], "incomplete")
        cases[0]["scheduler"]["config"]["_use_default_values"] = [1, "a"]
        cases[1]["scheduler"]["config"]["_use_default_values"] = ["a", 1]
        self.assertEqual(reporter.make_pairs(cases, "float16")[0]["status"], "incomplete")


class ReadableSummaryTests(unittest.TestCase):
    def test_incomplete_pair_is_visible_but_excluded_from_descriptive_medians(self):
        cases = [case("none"), case("cache-dit"), case("none", "motion"), case("cache-dit", "motion", valid=False)]
        snapshot = reporter.formal_snapshot(reporter.make_pairs(cases, "float16"))
        self.assertEqual(snapshot["planned_pairs"], 2)
        self.assertEqual(snapshot["completed_pairs"], 1)
        self.assertEqual(snapshot["rows"][0]["baseline_e2e_seconds"], 120.0)
        self.assertEqual(snapshot["rows"][0]["cache_diagnostic_transformer_cuda_seconds"], 20.0)
        self.assertEqual(snapshot["rows"][0]["allocated_peak_cache_minus_baseline_mib"], 256.0)
        self.assertIsNone(snapshot["rows"][1]["baseline_e2e_seconds"])
        self.assertEqual(snapshot["descriptive_medians"]["e2e_baseline_over_cache"], {"value": 2.0, "n": 1})

    def test_missing_diagnostic_has_its_own_denominator(self):
        cases = [case("none"), case("cache-dit")]
        cases[1]["metrics"]["diagnostic_transformer_cuda_sum_ms"] = None
        snapshot = reporter.formal_snapshot(reporter.make_pairs(cases, "float16"))
        self.assertEqual(snapshot["descriptive_medians"]["e2e_baseline_over_cache"]["n"], 1)
        self.assertEqual(snapshot["descriptive_medians"]["diagnostic_transformer_cuda_baseline_over_cache"], {"value": None, "n": 0})
        self.assertEqual(snapshot["descriptive_medians"]["cache_diagnostic_transformer_cuda_seconds"]["n"], 0)

    def test_front_table_has_seconds_n_and_limits_without_scientific_notation(self):
        summary = summary_fixture([case("none"), case("cache-dit"), case("none", "motion"), case("cache-dit", "motion", valid=False)])
        with tempfile.TemporaryDirectory() as tmp:
            reporter.write_report(summary, Path(tmp))
            report = (Path(tmp) / "REPORT.md").read_text(encoding="utf-8")
            first_section = report.split("## 环境与计时口径")[0]
            self.assertIn("当前完整正式配对 n=1 / 2", first_section)
            self.assertIn("E2E 秒：无/缓存", first_section)
            self.assertIn("120 / 60", first_section)
            self.assertIn("100 / 20", first_section)
            self.assertIn("未完整配对", first_section)
            self.assertIn("不是完整去噪墙钟", first_section)
            self.assertIn("perceptual_quality_validated=false", first_section)
            self.assertNotRegex(report, r"\d\.\d+e[+-]\d+")
            saved = json.loads((Path(tmp) / "v100-summary.json").read_text())
            self.assertEqual(saved["formal_readable_summary"]["completed_pairs"], 1)
            self.assertFalse(saved["perceptual_quality_validated"])
            self.assertFalse(saved["research_gate_eligible"])

    def test_empty_formal_snapshot_has_null_medians_and_zero_n(self):
        snapshot = reporter.formal_snapshot([])
        self.assertEqual(snapshot["completed_pairs"], 0)
        self.assertTrue(all(value == {"value": None, "n": 0} for value in snapshot["descriptive_medians"].values()))


class PhaseSupplementTests(unittest.TestCase):
    @staticmethod
    def phase_record(arm):
        output = {"finite": True, "sha256": "b" * 64}
        reset = {"status": "not_applicable", "calls": []}
        if arm == "cache-dit":
            reset = {"status": "measured_populated_pipeline_reset", "instrumented": True,
                     "all_resets_precede_denoising_loop": True, "context_count": 1,
                     "calls": [{"status": "completed", "call_wall_ms": 0.1,
                                "synchronized_wall_ms": 0.2, "post_call_synchronize_ms": 0.1,
                                "before": {"executed_steps": 50, "transformer_executed_steps": 100,
                                           "buffer_inventory": {"tensors": [{"device": "cuda:0", "bytes": 16}]}}}]}
        return {"schema_version": "ditserve.v100-phases.v1", "kind": "gpu_phase_supplement",
                "arm": arm, "status": "completed", "measured": True, "instrumented": True,
                "telemetry_complete": True, "finished_at_utc": "2026-09-12T00:00:00+00:00",
                "config": {"seed": 0, "steps": 50, "height": 480, "width": 832, "frames": 17},
                "phase_run": {"status": "completed", "seed": 0, "output": output,
                              "denoising_loop": {"status": "completed", "instrumented": True,
                                                 "progress_updates": 50, "wall_ms": 9000},
                              "internal_cache_reset": reset,
                              "output_vs_existing_same_arm": {"reference_sha256": "b" * 64,
                                                              "phase_sha256": "b" * 64, "exact_match": True}}}

    @staticmethod
    def save_phase(root, arm, record):
        path = root / "phases" / arm / "profile.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_complete_supplement_is_separate_and_does_not_change_primary_statistics(self):
        summary = summary_fixture([case("none"), case("cache-dit")])
        primary = copy.deepcopy(summary["pairs"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self.save_phase(root, arm, self.phase_record(arm)) for arm in ("none", "cache-dit")]
            original = [path.read_bytes() for path in paths]
            reporter.write_report(summary, root)
            supplement = summary["phase_supplement"]
            self.assertEqual(supplement["completed_arms"], 2)
            self.assertEqual(supplement["rows"][0]["denoising_loop_seconds"], 9)
            self.assertEqual(supplement["rows"][1]["reset_call_wall_ms"], 0.1)
            self.assertEqual(supplement["rows"][1]["reset_synchronized_wall_ms"], 0.2)
            self.assertTrue(supplement["rows"][1]["reference_hash_equal"])
            self.assertFalse(supplement["included_in_primary_e2e_statistics"])
            self.assertEqual(summary["formal_readable_summary"]["descriptive_medians"]["baseline_e2e_seconds"], {"value": 120, "n": 1})
            self.assertEqual(summary["pairs"][0]["e2e_baseline_over_cache"], primary[0]["e2e_baseline_over_cache"])
            self.assertEqual(summary["formal_e2e_ratio_median"], 2)
            self.assertEqual(original, [path.read_bytes() for path in paths])
            self.assertIn("不替代原六对无观测 E2E 比值", (root / "REPORT.md").read_text())

    def test_failed_partial_loop_reset_or_telemetry_cannot_be_reported_completed(self):
        variants = []
        for location, key, value in (((), "status", "failed"), (("phase_run",), "status", "running"),
                                     (("phase_run", "denoising_loop"), "status", "failed"),
                                     ((), "telemetry_complete", False)):
            record = self.phase_record("cache-dit")
            target = record
            for part in location:
                target = target[part]
            target[key] = value
            variants.append(record)
        record = self.phase_record("cache-dit")
        record["phase_run"]["internal_cache_reset"]["calls"][0]["status"] = "failed"
        variants.append(record)
        for index, record in enumerate(variants):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.save_phase(root, "cache-dit", record)
                supplement = reporter.read_phase_supplement(root)
                row = supplement["rows"][1]
                self.assertEqual(supplement["completed_arms"], 0)
                self.assertEqual(row["status"], "incomplete")
                self.assertIsNone(row["denoising_loop_seconds"])
                self.assertIsNone(row["reset_call_wall_ms"])
                self.assertIsNone(row["reference_hash_equal"])
                self.assertEqual(row["record"], record)

    def test_missing_supplement_is_unknown_without_creating_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supplement = reporter.read_phase_supplement(root)
            self.assertEqual(supplement["status"], "unavailable")
            self.assertTrue(all(row["status"] == "missing" and row["denoising_loop_seconds"] is None for row in supplement["rows"]))
            self.assertFalse((root / "phases").exists())

    def test_explicit_retry_keeps_failed_attempt_visible_and_primary_statistics_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.save_phase(root, "none", self.phase_record("none"))
            failed = self.phase_record("cache-dit")
            failed.update(status="failed", failure={"type": "TypeError", "message": "multiple values for argument name"})
            previous = self.save_phase(root, "cache-dit", failed)
            retry = root / "phases-retry/cache-dit/profile.json"
            retry.parent.mkdir(parents=True)
            retry.write_text(json.dumps(self.phase_record("cache-dit")))
            selector = root / "phase-selection.json"
            selector.write_text(json.dumps({"selected_profiles": {"none": str(baseline), "cache-dit": str(retry)},
                                            "reason": "修正观测包装器签名后仅重跑缓存组", "previous_failed_profiles": [str(previous)]}))
            original = {path: path.read_bytes() for path in (baseline, previous, retry, selector)}
            summary = summary_fixture([case("none"), case("cache-dit")])
            reporter.write_report(summary, root)
            supplement = summary["phase_supplement"]
            self.assertEqual(supplement["completed_arms"], 2)
            self.assertEqual(supplement["rows"][1]["source"], str(retry))
            self.assertEqual(supplement["selection"]["previous_failed_attempts"][0]["status"], "failed")
            self.assertEqual(summary["formal_e2e_ratio_median"], 2)
            report = (root / "REPORT.md").read_text()
            self.assertIn("phase-selection.json", report)
            self.assertIn("修正观测包装器签名后仅重跑缓存组", report)
            self.assertIn("phases/cache-dit/profile.json", report)
            self.assertIn("multiple values for argument name", report)
            self.assertEqual(original, {path: path.read_bytes() for path in original})

    def test_selected_failed_retry_does_not_fall_back_to_valid_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.save_phase(root, "none", self.phase_record("none"))
            self.save_phase(root, "cache-dit", self.phase_record("cache-dit"))
            retry = root / "phases-retry/cache-dit/profile.json"
            retry.parent.mkdir(parents=True)
            failed = self.phase_record("cache-dit")
            failed["status"] = "failed"
            retry.write_text(json.dumps(failed))
            (root / "phase-selection.json").write_text(json.dumps({
                "selected_profiles": {"none": str(baseline), "cache-dit": str(retry)},
                "reason": "显式选择该补测尝试", "previous_failed_profiles": []}))
            supplement = reporter.read_phase_supplement(root)
            self.assertEqual(supplement["completed_arms"], 1)
            self.assertEqual(supplement["rows"][1]["source"], str(retry))
            self.assertEqual(supplement["rows"][1]["status"], "incomplete")
            self.assertIsNone(supplement["rows"][1]["denoising_loop_seconds"])

    def test_invalid_explicit_selection_does_not_silently_use_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.save_phase(root, "none", self.phase_record("none"))
            self.save_phase(root, "cache-dit", self.phase_record("cache-dit"))
            (root / "phase-selection.json").write_text(json.dumps({
                "selected_profiles": {"none": str(baseline), "cache-dit": "relative/profile.json"},
                "reason": "无效相对路径", "previous_failed_profiles": []}))
            supplement = reporter.read_phase_supplement(root)
            self.assertEqual(supplement["selection"]["status"], "invalid")
            self.assertEqual(supplement["completed_arms"], 0)
            self.assertTrue(all(row["denoising_loop_seconds"] is None for row in supplement["rows"]))


if __name__ == "__main__":
    unittest.main()
