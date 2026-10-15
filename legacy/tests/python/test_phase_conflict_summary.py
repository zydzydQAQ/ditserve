"""CPU evidence checks for phase conflict reports; no model or CUDA is loaded."""

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

from ditserve.experiments import summarize_phase_conflict as reporter


def fingerprint(letter, frames=17):
    return {"sha256": letter * 64, "finite": True,
            "shape": [1, frames, 16, 16, 3], "dtype": "float32"}


class SummaryFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = []
        self.b_path = self.root / "b" / "latent.pt"
        self.meta_path = self.root / "b" / "provenance.json"
        self.meta = {"schema_version": reporter.PROFILE_SCHEMA,
                     "config": self.config("b", mode="prepare", frames=17, seed=9),
                     "model_id": "/model/wan", "revision": "a" * 40,
                     "b_latent_path": str(self.b_path),
                     "latent": fingerprint("c", frames=5), "output_b": fingerprint("b")}
        self.write(self.meta_path, self.meta)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def config(self, case_id, **changes):
        result = {"id": case_id, "mode": "case", "strategy": "immediate", "insert_after": 2,
                  "height": 16, "width": 16, "frames": 33, "steps": 4, "seed": 0, "cache": "none",
                  "dtype": "float16", "vae_dtype": "float32", "text_encoder_dtype": "float32",
                  "model_id": "/model/wan", "revision": "a" * 40, "prompt": "an object",
                  "gpu_uuid": "GPU-a058066e-bde9-c534-efb1-d2a68c117681",
                  "cache_parameters": {"threshold": 0.15}, "diagnostic": False, "repeat": 0,
                  "output_dir": str(self.root / case_id)}
        result.update(changes)
        return result

    def add_case(self, strategy, case_id=None, status="completed", profile_status=None,
                 remaining_ms=1000.0, config_changes=None):
        case_id = case_id or strategy
        attempt = sum(record["id"] == case_id for record in self.records)
        folder = self.root / (case_id + ("-attempt%d" % attempt if attempt else ""))
        config = self.config(case_id, strategy=strategy, b_metadata_path=str(self.meta_path),
                             b_latent_path=str(self.b_path), b_frames=17, b_seed=9, output_dir=str(folder))
        config.update(config_changes or {})
        job = copy.deepcopy(config)
        events = {"steps": [{"step_index": index, "completed_step": index + 1,
                             "forward_count": (index + 1) * 2, "host_monotonic_ns": 1000 + index * 100}
                            for index in range(4)],
                  "dit_forwards": [{"forward_index": index, "cuda_start_ms": index * 10.0,
                                    "cuda_end_ms": index * 10.0 + 8} for index in range(8)],
                  "step_count": 4, "forward_count": 8, "a_stream": 1, "b_stream": 2,
                  "b_decode": {"cuda_start_ms": 35.0, "cuda_finished_ms": 65.0,
                               "cuda_interval_ms": 30.0, "status": "completed", "boundary_step": 2}}
        profile = {"schema_version": reporter.PROFILE_SCHEMA, "job": job, "config": config,
                   "kind": "real_gpu_phase_conflict", "measured": True,
                   "revision": {"requested": config["revision"], "resolved_commits": []},
                   "environment": {"ready": True, "device": "cuda:0", "cuda_device_order": "PCI_BUS_ID",
                                   "cuda_visible_devices": config["gpu_uuid"],
                                   "gpu": {"uuid": config["gpu_uuid"].replace("GPU-", "")}},
                   "status": profile_status or status, "output_a": fingerprint("a", frames=config["frames"]),
                   "output_b": fingerprint("b"), "events": events,
                   "timings": {"remaining_ms": remaining_ms, "a_completed_ms": remaining_ms - 10,
                               "b_completed_ms": remaining_ms - 20, "cpu_prompt_encode_ms": 120,
                               "prompt_embeddings_h2d_ms": 0.5},
                   "memory": {"initial_allocated_bytes": 1024, "peak_allocated_bytes": 4096,
                              "peak_reserved_bytes": 8192, "before_a_vae_peak_allocated_bytes": 3072},
                   "b_source": {"config": copy.deepcopy(self.meta["config"]), "latent": self.meta["latent"],
                                "latent_path": str(self.b_path), "metadata_path": str(self.meta_path),
                                "output_b": self.meta["output_b"]}}
        if status != "completed":
            profile["status"] = profile_status or "failed"
            profile["failure"] = {"category": "cuda_oom" if status == "oom" else "worker_error",
                                  "message": "fixture failure", "measured_at_failure": True}
        path = folder / "profile.json"
        self.write(path, profile)
        self.records.append({"id": case_id, "job": job, "profile_path": str(path),
                             "status": status, "elapsed_seconds": 37.0})
        return profile, path

    def summarize(self):
        self.write(self.root / "manifest.json", {"status": "running", "records": self.records,
                   "budget": {"limit_seconds": 10000, "used_seconds": 37 * len(self.records)}})
        return reporter.summarize(self.root)

    def four(self, **kwargs):
        for strategy in reporter.STRATEGIES:
            self.add_case(strategy, **kwargs)


class ReferenceAndSelectionTests(SummaryFixture):
    def test_a_and_prepare_video_shapes_must_match_their_own_requests(self):
        self.four()
        profile, path = self.add_case("overlap", case_id="wrong-a-shape")
        profile["output_a"]["shape"][1] = 17
        self.write(path, profile)
        self.add_case("immediate", case_id="wrong-prepare-shape", config_changes={"mode": "prepare"})
        summary = self.summarize()
        self.assertEqual(summary["counts"]["valid_pairs"], 4)
        for case in summary["cases"][-2:]:
            self.assertFalse(case["valid_execution"])
            self.assertTrue(any("video shape differs" in issue for issue in case["issues"]))

    def test_matching_b_fingerprints_with_wrong_video_shape_still_fail_reference_validation(self):
        self.meta["output_b"]["shape"][2] = 32
        self.write(self.meta_path, self.meta)
        self.four()
        for record in self.records:
            path = Path(record["profile_path"])
            profile = json.loads(path.read_text())
            profile["output_b"] = copy.deepcopy(self.meta["output_b"])
            self.write(path, profile)
        summary = self.summarize()
        self.assertEqual(summary["counts"]["valid_pairs"], 0)
        for case in summary["cases"]:
            self.assertTrue(reporter.same_output(case["output_b"], self.meta["output_b"]))
            self.assertEqual(case["b_validation"]["status"], "invalid")
            self.assertTrue(any("reference B video shape differs" in issue for issue in case["b_validation"]["issues"]))
        self.assertTrue(reporter.output_valid(self.meta["latent"]))

    def test_mixed_b_shape_and_seed_use_own_metadata_and_offset_reuses_serial_references(self):
        self.four()
        for strategy in ("blocking", "overlap"):
            self.add_case(strategy, case_id=strategy + "-early", config_changes={"insert_after": 1})
        summary = self.summarize()
        self.assertEqual(summary["counts"]["valid_pairs"], 6)
        self.assertEqual(summary["counts"]["complete_four_strategy_groups"], 2)
        for case in summary["cases"]:
            self.assertEqual(case["b_validation"]["generation_identity"]["frames"], 17)
            self.assertEqual(case["generation_identity"]["frames"], 33)
            self.assertNotEqual(case["generation_identity"]["seed"], case["b_validation"]["generation_identity"]["seed"])
        self.assertEqual(summary["conclusion"]["status"], "insufficient_evidence")

    def test_a_reference_cannot_cross_seed_cache_shape_or_cache_parameters(self):
        self.add_case("immediate")
        self.add_case("deferred")
        changes = ({"seed": 1}, {"cache": "cache-dit"}, {"frames": 49},
                   {"cache_parameters": {"threshold": 0.25}})
        for index, change in enumerate(changes):
            self.add_case("overlap", case_id="wrong-%d" % index, config_changes=change)
        summary = self.summarize()
        self.assertEqual(summary["counts"]["valid_pairs"], 2)
        self.assertTrue(all(case["a_validation"]["status"] == "unknown" for case in summary["cases"][2:]))

    def test_b_output_loaded_latent_and_model_recipe_all_need_reference_evidence(self):
        self.four()
        profile, path = self.add_case("overlap", case_id="bad-output")
        profile["output_b"]["sha256"] = "d" * 64
        self.write(path, profile)
        profile, path = self.add_case("overlap", case_id="bad-latent")
        profile["b_source"]["latent"] = fingerprint("e", frames=5)
        self.write(path, profile)
        profile, path = self.add_case("overlap", case_id="bad-revision")
        profile["config"]["revision"] = "f" * 40
        profile["revision"]["requested"] = "f" * 40
        profile["job"]["revision"] = "f" * 40
        self.records[-1]["job"] = copy.deepcopy(profile["job"])
        self.write(path, profile)
        summary = self.summarize()
        self.assertEqual(summary["counts"]["valid_pairs"], 4)
        self.assertTrue(all(case["b_validation"]["status"] == "invalid" for case in summary["cases"][4:]))

    def test_retries_preserve_failure_but_count_latest_valid_unchanged_job_once(self):
        self.add_case("overlap", status="oom")
        self.four()
        self.records.append({"id": "not-started", "job": {"id": "not-started"},
                             "status": "skipped_budget", "reason": "budget exhausted", "elapsed_seconds": 0})
        summary = self.summarize()
        self.assertEqual(len(summary["cases"]), 6)
        self.assertFalse(summary["cases"][0]["selected"])
        self.assertEqual(summary["counts"]["valid_pairs"], 4)
        self.assertEqual(summary["counts"]["manifest_statuses"]["oom"], 1)
        self.assertEqual(summary["groups"][0]["strategies"]["overlap"]["valid_pairs"], 1)
        self.assertEqual(summary["cases"][-1]["reason"], "budget exhausted")

    def test_reusing_profile_or_changing_retry_identity_cannot_inflate_samples(self):
        self.four()
        self.add_case("overlap", config_changes={"seed": 1})
        summary = self.summarize()
        self.assertFalse(any(case["selected"] for case in summary["cases"] if case["id"] == "overlap"))
        self.records.pop()
        self.records.append(copy.deepcopy(self.records[-1]))
        summary = self.summarize()
        self.assertFalse(any(case["selected"] for case in summary["cases"] if case["id"] == "overlap"))

    def test_failed_profile_with_complete_looking_fields_and_missing_record_are_not_success(self):
        self.add_case("immediate")
        self.add_case("deferred")
        self.add_case("blocking", profile_status="failed")
        self.add_case("overlap", status="failed", profile_status="completed")
        self.records.append({"id": "missing", "job": {}, "profile_path": "missing.json",
                             "status": "completed", "elapsed_seconds": 10})
        summary = self.summarize()
        self.assertEqual(summary["counts"]["valid_pairs"], 2)
        self.assertEqual(summary["counts"]["complete_four_strategy_groups"], 0)


class EvidenceAndRenderingTests(SummaryFixture):
    def test_b_gpu_start_wait_is_distinct_from_decode_duration_and_missing_stays_unknown(self):
        self.add_case("immediate")
        profile, path = self.add_case("overlap", case_id="failed-without-start", status="oom")
        profile["events"]["b_decode"] = {"status": "failed"}
        self.write(path, profile)
        summary = self.summarize()
        self.assertEqual(summary["cases"][0]["timings"]["b_wait_to_gpu_decode_start_ms"], 35)
        self.assertEqual(summary["cases"][0]["events"]["b_decode"]["cuda_interval_ms"], 30)
        self.assertIsNone(summary["cases"][1]["timings"]["b_wait_to_gpu_decode_start_ms"])
        reporter.write_report(self.root, summary)
        report = (self.root / "REPORT.md").read_text()
        self.assertIn("B 等待至 GPU 解码开始 s", report)
        self.assertIn("| 0.035 | 0.03 |", report)

    def test_timeline_covers_cases_beyond_twelfth_record(self):
        for index in range(14):
            self.add_case("overlap", case_id="timeline-case-%02d" % index)
        summary = self.summarize()
        reporter.write_report(self.root, summary)
        tree = ET.parse(self.root / "timeline.svg")
        labels = [node.text for node in tree.findall(".//{http://www.w3.org/2000/svg}text")
                  if (node.text or "").startswith("timeline-case-")]
        self.assertEqual(len(labels), 14)
        self.assertIn("timeline-case-13 / overlap / completed", labels)
        self.assertNotIn("前 12 条", (self.root / "REPORT.md").read_text())

    def test_oom_observed_peaks_stay_lower_bounds_and_warmup_is_separate(self):
        self.add_case("immediate")
        self.add_case("deferred")
        for index, measured in enumerate((True, True, False)):
            profile, path = self.add_case("overlap", case_id="oom-%d" % index, status="oom")
            profile["measured"] = measured
            profile["failure"]["measured_at_failure"] = measured
            profile["failure"]["last_progress"] = {"stage": "b_decode_started" if measured else "warmup_4_steps"}
            profile["memory"] = {"observed_at_failure": {
                "allocated_bytes": 1024 ** 2, "reserved_bytes": 8 * 1024 ** 2,
                "peak_allocated_bytes": 7 * 1024 ** 2, "peak_reserved_bytes": 9 * 1024 ** 2}}
            self.write(path, profile)
        summary = self.summarize()
        reporter.write_report(self.root, summary)
        self.assertEqual(summary["counts"]["failure_categories"]["cuda_oom"], 3)
        self.assertEqual(summary["conclusion"]["phenomenon"]["repeated_oom_signal_groups"], 1)
        self.assertEqual(summary["conclusion"]["status"], "insufficient_evidence")
        observation = summary["cases"][2]["failure_memory_observation"]
        self.assertTrue(observation["is_observed_lower_bound"])
        self.assertFalse(observation["is_completed_peak"])
        self.assertEqual(summary["cases"][-1]["failure_memory_observation"]["scope"], "outside_measured_window")
        self.assertIsNone(summary["groups"][0]["strategies"]["overlap"]["peak_allocated_bytes"]["median"])
        self.assertEqual(summary["groups"][0]["strategies"]["overlap"]["metrics"]["remaining_ms"]["n"], 0)
        report = (self.root / "REPORT.md").read_text()
        self.assertIn("已观测 allocated 峰值下界 MiB", report)
        self.assertIn("测量窗口外（如 warmup）", report)
        self.assertIn("编码波动也会改变策略排名", report)

    def test_repeated_different_spec_optima_can_support_further_validation(self):
        for frames, best in ((33, "immediate"), (49, "overlap")):
            for strategy in reporter.STRATEGIES:
                for repeat in range(2):
                    self.add_case(strategy, case_id="%s-f%d-r%d" % (strategy, frames, repeat),
                                  remaining_ms=100 if strategy == best else 200,
                                  config_changes={"frames": frames, "repeat": repeat})
        summary = self.summarize()
        self.assertEqual(summary["conclusion"]["status"], "continue_validation")
        self.assertFalse(summary["research_gate_eligible"])

    def test_different_phase_optima_and_always_serial_have_different_research_decisions(self):
        for second_best, expected in (("overlap", "continue_validation"), ("immediate", "stop_complex_scheduling")):
            with self.subTest(second_best=second_best):
                self.records = []
                for offset, best in ((2, "immediate"), (1, second_best)):
                    for strategy in reporter.STRATEGIES:
                        for repeat in range(2):
                            self.add_case(strategy, case_id="%s-k%d-r%d" % (strategy, offset, repeat),
                                          remaining_ms=100 if strategy == best else 200,
                                          config_changes={"insert_after": offset, "repeat": repeat})
                result = self.summarize()["conclusion"]
                # Baselines have no insertion. If their repeated samples disagree
                # across offset tags, that is variance rather than a new optimum.
                if second_best == "overlap":
                    self.assertEqual(result["status"], "insufficient_evidence")
                else:
                    self.assertEqual(result["status"], expected)

    def test_environment_uuid_and_measurement_kind_are_independently_checked(self):
        self.four()
        for field, value in (("measured", False), ("kind", "synthetic")):
            profile, path = self.add_case("overlap", case_id="bad-" + field)
            profile[field] = value
            self.write(path, profile)
        profile, path = self.add_case("overlap", case_id="wrong-gpu")
        profile["environment"]["gpu"]["uuid"] = "11111111-2222-3333-4444-555555555555"
        self.write(path, profile)
        summary = self.summarize()
        self.assertEqual(summary["counts"]["valid_pairs"], 4)
        self.assertTrue(all(not case["valid_execution"] for case in summary["cases"][4:]))

    def test_repeated_signal_is_descriptive_and_smoke_never_supplies_formal_repeats(self):
        self.four()
        self.add_case("blocking", case_id="blocking-r1", config_changes={"repeat": 1})
        self.add_case("overlap", case_id="overlap-r1", remaining_ms=1400, config_changes={"repeat": 1})
        summary = self.summarize()
        self.assertEqual(summary["conclusion"]["status"], "insufficient_evidence")
        self.assertEqual(summary["conclusion"]["phenomenon"]["status"], "observed")
        self.assertFalse(summary["online_goodput_measured"])
        self.assertFalse(summary["research_gate_eligible"])
        for record in self.records:
            path = Path(record["profile_path"])
            profile = json.loads(path.read_text())
            record["job"]["diagnostic"] = profile["job"]["diagnostic"] = profile["config"]["diagnostic"] = True
            self.write(path, profile)
        self.assertEqual(self.summarize()["conclusion"]["status"], "insufficient_evidence")

    def test_repeated_insertion_oom_is_visible_with_valid_serial_references(self):
        self.add_case("immediate")
        self.add_case("deferred")
        self.add_case("blocking", status="oom")
        self.add_case("overlap", status="oom")
        summary = self.summarize()
        self.assertEqual(summary["conclusion"]["status"], "insufficient_evidence")
        self.assertEqual(summary["conclusion"]["phenomenon"]["status"], "observed")
        self.assertEqual(summary["counts"]["valid_pairs"], 2)
        self.assertEqual(summary["counts"]["complete_four_strategy_groups"], 0)
        for record in self.records[-2:]:
            profile = json.loads(Path(record["profile_path"]).read_text())
            profile["measured"] = False
            profile["failure"]["measured_at_failure"] = False
            self.write(Path(record["profile_path"]), profile)
        self.assertEqual(self.summarize()["conclusion"]["phenomenon"]["status"], "insufficient_evidence")

    def test_prepare_phase_peaks_are_reported_separately_without_inventing_a_pair(self):
        profile, path = self.add_case("immediate", case_id="prepare", config_changes={"mode": "prepare"})
        profile["output_b"] = fingerprint("b", frames=profile["config"]["frames"])
        profile["phase_profiles"] = {
            "denoise": {"wall_ms": 2000, "initial_allocated_bytes": 1024 ** 2, "peak_allocated_bytes": 5 * 1024 ** 2},
            "decode": {"wall_ms": 3000, "initial_allocated_bytes": 2 * 1024 ** 2, "peak_allocated_bytes": 7 * 1024 ** 2}}
        self.write(path, profile)
        summary = self.summarize()
        reporter.write_report(self.root, summary)
        self.assertEqual(summary["counts"]["valid_pairs"], 0)
        self.assertEqual(summary["groups"], [])
        self.assertEqual(summary["counts"]["manifest_records"], 1)
        report = (self.root / "REPORT.md").read_text()
        self.assertIn("准备 B / 规格筛查", report)
        self.assertNotIn("### 组", report)
        self.assertIn("| denoise | 2 | 1 |", report)
        self.assertIn("| decode | 3 | 2 |", report)

    def test_running_report_leads_with_snapshot_and_uses_compact_group_identity(self):
        self.four()
        summary = self.summarize()
        reporter.write_report(self.root, summary)
        report = (self.root / "REPORT.md").read_text()
        self.assertTrue(report.splitlines()[2].startswith("**实验进行中。**"))
        self.assertIn("当前已完成记录的暂时判读", report)
        heading = next(line for line in report.splitlines() if line.startswith("### 组"))
        self.assertIn("16×16，A/B 33/17 帧，4 步，cache=none，seed=0，插入步=2", heading)
        self.assertNotIn("/model/wan", report)
        self.assertNotIn("an object", report)
        self.assertEqual(json.loads((self.root / "summary.json").read_text())["groups"][0]["identity"]["a"]["prompt"], "an object")
        summary["manifest"]["status"] = "completed"
        reporter.write_report(self.root, summary)
        self.assertNotIn("实验进行中", (self.root / "REPORT.md").read_text())

    def test_stage_csv_svg_and_budget_keep_measurement_scopes_separate(self):
        self.four()
        summary = self.summarize()
        event = summary["cases"][-1]["events"]
        self.assertTrue(event["intersections"])
        self.assertFalse(event["kernel_concurrency_verified"])
        self.assertEqual(event["step_rows"][0]["transformer_event_interval_sum_ms"], 16)
        self.assertEqual(summary["cases"][-1]["elapsed_seconds"], 37)
        self.assertEqual(summary["cases"][-1]["timings"]["remaining_ms"], 1000)
        summary["cases"][-1]["id"] = "<script>&unsafe"
        reporter.write_report(self.root, summary)
        ET.parse(self.root / "timeline.svg")
        self.assertNotIn("<script>", (self.root / "timeline.svg").read_text())
        with (self.root / "steps.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 16)
        report = (self.root / "REPORT.md").read_text()
        self.assertIn("不是从真实在线到达开始的完整 E2E", report)
        self.assertIn("不能把策略峰值差分解释为缓存状态大小", report)
        self.assertIn("kernel 时间总和或完整去噪墙钟", report)

    def test_incomplete_step_mapping_is_unknown_not_an_invented_time(self):
        profile, path = self.add_case("overlap")
        profile["events"]["dit_forwards"][0]["forward_index"] = 1
        self.write(path, profile)
        rows = self.summarize()["cases"][0]["events"]["step_rows"]
        self.assertTrue(all(row["mapping_status"] == "unknown" and row["transformer_event_interval_sum_ms"] is None for row in rows))

    def test_absent_or_corrupt_manifest_generates_auditable_empty_report(self):
        for value in (None, '{"records": [NaN]}'):
            if value is not None:
                (self.root / "manifest.json").write_text(value)
            summary = reporter.summarize(self.root)
            reporter.write_report(self.root, summary)
            self.assertTrue(summary["issues"])
            self.assertEqual(summary["conclusion"]["status"], "insufficient_evidence")
            self.assertEqual(summary["counts"]["valid_pairs"], 0)
            json.loads((self.root / "summary.json").read_text())


class MediaTests(SummaryFixture):
    def test_missing_optional_cpu_dependencies_are_recorded(self):
        with mock.patch.object(reporter.importlib, "import_module", side_effect=ImportError("fixture no NumPy")):
            result = reporter.export_media(self.root, {"cases": []})
        self.assertEqual(result["status"], "failed")
        self.assertTrue(json.loads((self.root / "media-export.json").read_text())["errors"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional CPU NumPy fixture")
    def test_pixel_validation_and_hash_reuse_never_silently_rescale(self):
        import numpy as np
        self.four()
        summary = self.summarize()
        cases = summary["cases"][:2]
        raw = np.full((1, 2, 16, 16, 3), 0.5, dtype=np.float32)
        digest = hashlib.sha256(raw.tobytes()).hexdigest()
        for case in cases:
            path = Path(case["profile_path"]).parent / "artifacts" / "output-a.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(path), raw)
            case["output_a"] = {"shape": list(raw.shape), "dtype": str(raw.dtype), "finite": True,
                                "sha256": digest, "artifact": str(path)}
            case["output_b"] = {}
        writer = mock.Mock()
        def write_video(path, frames, **options):
            self.assertEqual(options, {"fps": 16, "codec": "libx264", "quality": 8, "macro_block_size": 1})
            self.assertTrue(np.all(frames[0] == 128))
            Path(path).write_bytes(b"CPU fixture video")
        writer.mimwrite.side_effect = write_video
        real_import = importlib.import_module
        with mock.patch.object(reporter.importlib, "import_module", side_effect=lambda name: writer if name == "imageio.v2" else real_import(name)):
            result = reporter.export_media(self.root, {"cases": cases})
            self.assertEqual(result["status"], "completed")
            self.assertEqual(writer.mimwrite.call_count, 1)
            self.assertTrue(result["videos"][1]["reused_video"])
            raw[0, 0, 0, 0, 0] = 1.2
            np.save(cases[1]["output_a"]["artifact"], raw)
            result = reporter.export_media(self.root, {"cases": cases})
            self.assertEqual(result["status"], "partial_failure")
            self.assertIn("out-of-range", result["errors"][0]["message"])
            self.assertEqual(writer.mimwrite.call_count, 1)


if __name__ == "__main__":
    unittest.main()
