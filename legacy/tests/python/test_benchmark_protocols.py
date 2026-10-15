import copy
import json
import tempfile
import unittest
from pathlib import Path

from ditserve.benchmark.common import seal, verify_seal
from ditserve.benchmark.datasets import prepare_dataset
from ditserve.benchmark.evaluation import VBENCH_DIMENSIONS, export_vbench_manifest
from ditserve.benchmark.protocols import build_vbench_trace, generation_settings
from ditserve.benchmark.workload import build_trace, validate_trace


class BenchmarkProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        source = root / "VBench_full_info.json"
        self.first_dimensions = list(VBENCH_DIMENSIONS[:3])
        source.write_text(json.dumps([
            {"prompt_en": " shared prompt ", "dimension": self.first_dimensions},
            {"prompt_en": "second prompt", "dimension": list(VBENCH_DIMENSIONS[3:])},
        ]))
        self.dataset = prepare_dataset("vbench", source=source, dev_count=1)
        for index, row in enumerate(self.dataset["samples"]):
            row["service_split"] = "dev" if index == 0 else "test"
        self.dataset = seal(self.dataset)
        self.config = {"model": {"id": "fixture-model", "revision": "a" * 40},
                       "shapes": {"short": {"height": 32, "width": 32, "frames": 5, "steps": 2},
                                  "long": {"height": 32, "width": 32, "frames": 9, "steps": 2}},
                       "cache": {"enabled": False, "threshold": .08, "max_continuous": 3},
                       "count": 6, "trace_seeds": [20261012], "slo_scale": 2.5}
        self.profile = seal({"schema_version": "ditserve.benchmark.profile.v1", "backend": "cpu-test",
                             "measurement_kind": "synthetic", "status": "completed",
                             "dataset_sha256": self.dataset["sha256"],
                             "dev_sample_ids": [self.dataset["samples"][0]["sample_id"]],
                             "config": copy.deepcopy(self.config), "shapes": {
                                 key: dict(shape, service_ms=100 if key == "short" else 200,
                                           prepare_ms=10, step_ms=[20, 20], decode_ms=10,
                                           peak_increment_mb=8, state_mb=4)
                                 for key, shape in self.config["shapes"].items()}})
        self.manifest = export_vbench_manifest(source, root / "manifest.json",
                                              generation_settings=generation_settings(self.config, "long"))

    def build(self, **overrides):
        inputs = {"config": self.config, "dataset": self.dataset,
                  "profile": self.profile, "manifest": self.manifest}
        inputs.update(overrides)
        return build_vbench_trace(**inputs)

    def test_flat_generation_settings_freeze_defaults_and_selected_shape(self):
        settings = generation_settings(self.config, "long")
        self.assertEqual(settings["model"], "fixture-model")
        self.assertEqual(settings["frames"], 9)
        self.assertEqual(settings["fps"], 16)
        self.assertFalse(settings["cache_enabled"])
        self.assertEqual(settings["max_continuous"], 3)
        self.assertEqual(settings["negative_prompt"], "")
        self.assertEqual(settings["guidance_scale"], 5.0)
        with self.assertRaises(ValueError):
            generation_settings(self.config, "unknown")
        for change in ({"cache": {"enabled": "false"}}, {"cache": {"max_continuous": 0}},
                       {"guidance_scale": float("nan")}, {"negative_prompt": None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                generation_settings(dict(self.config, **change), "long")

    def test_complete_protocol_reuses_prompt_seed_across_dimensions_and_allows_service_dev(self):
        originals = copy.deepcopy((self.config, self.dataset, self.profile, self.manifest))
        trace = self.build()
        verify_seal(trace)
        self.assertIs(validate_trace(trace), trace)
        self.assertEqual((self.config, self.dataset, self.profile, self.manifest), originals)
        self.assertEqual(trace, self.build())
        self.assertEqual(len(self.manifest["requests"]), 100)
        self.assertEqual(len(trace["requests"]), 30)
        self.assertEqual(trace["arrival"]["count"], 30)
        self.assertEqual(trace["arrival"]["mean_standalone_service_ms"], 200)
        self.assertEqual({row["arrival_ms"] for row in trace["requests"]}, {0})
        self.assertEqual({row["deadline_ms"] for row in trace["requests"]}, {500})
        self.assertEqual({row["shape_id"] for row in trace["requests"]}, {"long"})
        shared = [row for row in trace["requests"] if row["prompt"] == " shared prompt "]
        self.assertEqual([row["seed"] for row in shared], list(range(25)))
        self.assertTrue(all(row["sample_id"] in self.profile["dev_sample_ids"] for row in shared))
        quality = trace["quality_protocol"]
        self.assertEqual(quality["expected_videos"], 100)
        self.assertEqual(quality["unique_generations"], 30)
        self.assertEqual(quality["scope"], "full_dimension_suite")
        self.assertEqual(quality["manifest_sha256"], self.manifest["sha256"])
        self.assertEqual(quality["full_info_sha256"], self.dataset["source"]["sha256"])
        self.assertFalse(trace["serving_performance_evidence"])
        self.assertEqual(trace["purpose"], "independent_quality_generation")
        self.assertEqual(trace["dataset_sha256"], self.dataset["sha256"])
        self.assertEqual(trace["profile_sha256"], self.profile["sha256"])

    def test_dimension_subset_keeps_flickering_seed_requirement_and_subset_scope(self):
        subset = dict(self.manifest, dimensions=self.first_dimensions, scope="dimension_subset")
        subset["requests"] = [row for row in self.manifest["requests"]
                              if row["dimension"] in self.first_dimensions]
        trace = self.build(manifest=seal(subset))
        self.assertEqual(len(trace["requests"]), 25)
        self.assertEqual(trace["quality_protocol"]["expected_videos"], 35)
        self.assertEqual(trace["quality_protocol"]["scope"], "dimension_subset")

    def test_missing_or_repeated_manifest_entries_are_rejected_even_when_resealed(self):
        for records in (self.manifest["requests"][:-1], self.manifest["requests"] + self.manifest["requests"][:1]):
            with self.subTest(count=len(records)), self.assertRaisesRegex(ValueError, "every source"):
                self.build(manifest=seal(dict(self.manifest, requests=records)))
        changed = copy.deepcopy(self.manifest)
        changed["requests"][0]["seed"] = False
        with self.assertRaisesRegex(ValueError, "every source"):
            self.build(manifest=seal(changed))

    def test_manifest_seal_version_source_hash_and_settings_are_mandatory(self):
        tampered = dict(self.manifest, version="other")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.build(manifest=tampered)
        mutations = ({"schema_version": "other"}, {"revision": "b" * 40}, {"version": "other"},
                     {"full_info_sha256": "b" * 64}, {"scope": "dimension_subset"},
                     {"seeds_per_prompt": 1}, {"generation_settings": {}},
                     {"generation_settings": dict(self.manifest["generation_settings"], guidance_scale=6)})
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.build(manifest=seal(dict(self.manifest, **mutation)))

    def test_wrong_profile_or_incomplete_prepared_dataset_is_not_hidden_by_bridge(self):
        wrong = copy.deepcopy(self.profile)
        wrong["config"]["model"]["revision"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "profile model"):
            self.build(profile=seal(wrong))
        incomplete = copy.deepcopy(self.dataset)
        incomplete["samples"] = incomplete["samples"][:1]
        incomplete["selection"]["rows_seen"] = 1
        incomplete["selection"]["samples_retained"] = 1
        with self.assertRaisesRegex(ValueError, "cover"):
            self.build(dataset=seal(incomplete))

    def test_quality_marker_alone_cannot_disable_service_dev_leakage_check(self):
        service = build_trace(self.config, self.dataset, self.profile, pattern="offline_all", count=1)
        service["requests"][0]["sample_id"] = self.profile["dev_sample_ids"][0]
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate_trace(seal(service))
        service["purpose"] = "independent_quality_generation"
        service["serving_performance_evidence"] = False
        with self.assertRaisesRegex(ValueError, "complete validated protocol"):
            validate_trace(seal(service))
        service["quality_protocol"] = {"kind": "vbench"}
        with self.assertRaisesRegex(ValueError, "invalid independent"):
            validate_trace(seal(service))

    def test_resealed_quality_trace_cannot_change_manifest_coverage_identity_or_purpose(self):
        trace = self.build()
        updates = (("prompt", "unlisted prompt"), ("seed", 77), ("sample_id", "other"),
                   ("shape_id", "short"), ("fps", 24), ("arrival_ms", 1))
        for field, value in updates:
            changed = copy.deepcopy(trace)
            changed["requests"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_trace(seal(changed))
        for field, value in (("manifest_sha256", "b" * 64), ("full_info_sha256", "b" * 64),
                             ("unique_generations", 29), ("expected_videos", 99),
                             ("serving_performance_evidence", True)):
            changed = copy.deepcopy(trace)
            changed["quality_protocol"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_trace(seal(changed))
        changed = dict(trace, serving_performance_evidence=True)
        with self.assertRaises(ValueError):
            validate_trace(seal(changed))
        changed = copy.deepcopy(trace)
        changed["quality_protocol"]["manifest"]["version"] = "other"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_trace(seal(changed))


if __name__ == "__main__":
    unittest.main()
