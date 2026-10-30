"""CPU-only contracts; the simulator subprocess is mocked, never built or run."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("hetero_study_contracts", ROOT / "benchmarks" / "run_hetero_8a100_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class HeteroStudyContracts(unittest.TestCase):
    def setUp(self):
        self.payloads = []
        process_patch = patch.object(study.subprocess, "run", side_effect=self.simulate)
        self.process = process_patch.start()
        self.addCleanup(process_patch.stop)
        self.trace = study.build_trace(study.workload_classes(), study.prompt_pools(),
                                       [dict(count=8, video_fraction=0.5, pattern="offline")], slots=8, seed=11)

    def simulate(self, command, **kwargs):
        self.assertEqual(command, [str(study.BINARY), "rpc", "simulate"])
        self.assertTrue(kwargs["check"])
        payload = json.loads(kwargs["input"])
        self.payloads.append(payload)
        return SimpleNamespace(stdout=json.dumps({
            "metrics": {"deadline_attainment": 0.5},
            "by_kind": {kind: {"attainment": 0.5, "latency_p99_ms": 100.0} for kind in ("image", "video")},
            "makespan_ms": 200.0,
        }))

    def run_spec(self, spec):
        return study.run_sim(self.trace["workload"], spec[1], spec[3])

    def test_main_policies_share_profile_execution_costs_and_candidates(self):
        outputs = [self.run_spec(spec) for spec in study.main_policy_specs()]
        baseline = self.payloads[0]
        execution_keys = (
            "cross_modal_swap_ms", "resume_scale", "quantum_ms", "chunked_decode", "decode_chunk_frames",
            "tiered_memory", "image_batch", "continuous_batch", "pairing", "forms", "widen_efficiency", "image_placement",
        )
        for payload, output in zip(self.payloads, outputs):
            with self.subTest(policy=payload["policy"]):
                self.assertEqual(payload["profile"], baseline["profile"])
                self.assertEqual(payload["workload"], baseline["workload"])
                for key in execution_keys:
                    self.assertEqual(payload["options"][key], baseline["options"][key], key)
                self.assertEqual(output["profile_sha256"], digest(payload["profile"]))
                self.assertEqual(output["trace_sha256"], self.trace["manifest"]["workload_sha256"])
                self.assertEqual(output["evidence_type"], "simulation")
                self.assertEqual(output["implementation_type"], "proxy")
                self.assertEqual(output["cost_source"], "mixed")
                self.assertEqual(payload["profile"]["provenance"], "mixed")
                self.assertIn("proxy", output["policy_label"])
                self.assertIn("uncalibrated_inputs", output["cost_source_details"])
                self.assertTrue(output["model_assumptions"])
                expected = "approximate_model" if payload["policy"] == "heterodit_approx" else "scheduling_exact"
                self.assertEqual(output["comparison_group"], expected)
        # Explicit image_batch=1 prevents the simulator's hidden DP batching path.
        self.assertEqual(baseline["options"]["image_batch"], 1)
        self.assertEqual(baseline["options"]["cross_modal_swap_ms"], 75.0)
        self.assertEqual(baseline["options"]["resume_scale"], 1.8)

    def test_v3_has_no_implicit_profile_discount(self):
        ordinary = study.run_sim(self.trace["workload"], "heterodit_approx", {})
        v3 = study.run_sim(self.trace["workload"], "heterodit_v3", {})
        self.assertEqual(self.payloads[0]["profile"], self.payloads[1]["profile"])
        self.assertEqual(self.payloads[0]["options"], self.payloads[1]["options"])
        self.assertEqual(ordinary["profile_sha256"], v3["profile_sha256"])
        self.assertEqual(v3["profile_scenario"], "shared")
        self.assertEqual(v3["comparison_group"], "bundle_sensitivity")
        self.assertNotIn("ring_quant_scale", self.payloads[1]["profile"])

    def test_discounts_require_named_sensitivity_and_keep_assumptions(self):
        specs = study.sensitivity_policy_specs()
        original_options = copy.deepcopy(specs[1][3])
        baseline, bundle = [self.run_spec(spec) for spec in specs]
        self.assertEqual(specs[1][3], original_options)
        self.assertNotEqual(baseline["profile_sha256"], bundle["profile_sha256"])
        self.assertEqual(baseline["trace_sha256"], bundle["trace_sha256"])
        self.assertEqual(bundle["profile_scenario"], "v3_bundle_sensitivity")
        self.assertEqual(bundle["comparison_group"], "bundle_sensitivity")
        self.assertEqual(bundle["profile_sha256"], digest(self.payloads[-1]["profile"]))
        self.assertIn("none are validated speedups", " ".join(bundle["model_assumptions"]))
        self.assertNotIn("profile_scenario", self.payloads[-1]["options"])
        self.assertEqual(bundle["simulation_options"], self.payloads[-1]["options"])
        calls = self.process.call_count
        for options in ({"use_v3_profile": True}, {"profile_scenario": "unknown"}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "profile_scenario"):
                study.run_sim(self.trace["workload"], "heterodit_v3", options)
        self.assertEqual(self.process.call_count, calls)

    def test_ablation_changes_exactly_one_effective_factor_per_transition(self):
        specs = study.ablation_policy_specs()
        outputs = [self.run_spec(spec) for spec in specs]
        self.assertEqual(len(specs), 6)
        expected_changes = ("tiered_memory", "forms", "policy", "cache_thresholds", "image_batch")
        for previous, current, expected in zip(self.payloads, self.payloads[1:], expected_changes):
            before = {"policy": previous["policy"], **previous["options"]}
            after = {"policy": current["policy"], **current["options"]}
            changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
            self.assertEqual(changed, {expected})
            self.assertEqual(previous["profile"], current["profile"])
            self.assertEqual(previous["workload"], current["workload"])
        self.assertEqual([output["comparison_group"] for output in outputs],
                         ["scheduling_exact"] * 4 + ["approximate_model"] * 2)
        self.assertEqual(self.payloads[3]["options"]["cache_thresholds"], [0.0])
        self.assertEqual(self.payloads[3]["policy"], self.payloads[4]["policy"])
        self.assertTrue(all("v3" not in spec[0] and "proxy" in spec[2] for spec in specs))

    def test_summary_preserves_evidence_hashes_and_separates_bundle(self):
        summary = study.evaluate_suite()
        self.assertEqual(summary["evidence_type"], "simulation")
        self.assertEqual(summary["cost_source"], "mixed")
        self.assertEqual(summary["implementation_type"], "proxy")
        self.assertEqual(summary["ablation"]["design"], "cumulative_one_factor")
        self.assertFalse(summary["sensitivity"]["orthogonal"])
        self.assertEqual(summary["sensitivity"]["design"], "bundle_sensitivity")
        for profile in summary["profiles"].values():
            self.assertEqual(profile["profile_sha256"], digest(profile["profile"]))
        for scenario in summary["scenarios"].values():
            self.assertNotIn("heterodit_v3", scenario["policies"])
            hashes = [row["trace_sha256"] for row in scenario["policies"].values()]
            self.assertTrue(all(h == hashes[0] for h in hashes))
            self.assertTrue(all(row["profile_sha256"] == summary["profiles"]["shared"]["profile_sha256"]
                                for row in scenario["policies"].values()))
        for section in [*summary["scenarios"].values(), summary["ablation"], summary["sensitivity"]]:
            self.assertEqual(section["evidence_type"], "simulation")
            for row in section["policies"].values():
                self.assertIn("proxy", row["label"])
                self.assertEqual(row["evidence_type"], "simulation")
                self.assertEqual(row["cost_source"], "mixed")
                self.assertEqual([run["seed"] for run in row["runs"]], summary["seeds"])
                self.assertEqual(row["total_slo_pct"], 50.0)
                for run in row["runs"]:
                    self.assertEqual(run["profile_sha256"], row["profile_sha256"])
                    self.assertEqual(run["trace_sha256"], run["manifest"]["workload_sha256"])
                    self.assertIn("slo_by_class", run["manifest"]["windows"][0])
        self.assertEqual(json.loads(json.dumps(summary, allow_nan=False)), summary)


if __name__ == "__main__":
    unittest.main()
