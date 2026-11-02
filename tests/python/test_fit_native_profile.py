import importlib.util
import math
import copy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fit_native_profile", ROOT / "benchmarks" / "fit_native_profile.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FitNativeProfileTest(unittest.TestCase):
    def test_constrained_fit_recovers_held_out_forms_without_row_overrides(self):
        policies = list(MODULE.FIT_POLICIES) + ["static_sp3", "static_cfg2sp3", "static_sp6", "static_cfg2sp4", "static_sp8"]
        spec = dict(frames=49, steps=20, height=512, width=896, guidance_scale=5.0)
        model = {
            "transformer_config": {
                "patch_size": [1, 2, 2], "num_attention_heads": 24,
                "attention_head_dim": 128, "num_layers": 30, "out_channels": 48,
            },
            "vae_config": {"scale_factor_temporal": 4, "scale_factor_spatial": 16},
        }
        facts = {"models": {"Wan2.2-TI2V-5B": model}}
        arch = MODULE.architecture(model, spec)
        requests = []
        for index, policy in enumerate(policies):
            step_ms = MODULE.predict(500.0, arch, *MODULE.form_of(policy), 12.0, 1.0, 0.05)
            requests.append({
                "cell": index, "kind": "video", "status": "completed", "model_ms": 20 * step_ms + 100.0,
                "outputs": [{"model_ms": 100.0}], "spec": spec, "encode_model_ms": 20.0,
                "advance_samples": [
                    {"steady": False, "model_ms": 999999.0},
                    {"steady": True, "model_ms": step_ms, "group": list(range(math.prod(MODULE.form_of(policy))))},
                    {"steady": True, "model_ms": step_ms, "group": list(range(math.prod(MODULE.form_of(policy))))},
                ],
                "transfer_export_ms": 2.0, "dispatch_gap_ms": 4.0, "timed_quanta": 2,
            })
        requests.append({"cell": 0, "kind": "image", "status": "completed", "model_ms": 250.0})
        record = {
            "experiment": "kairos-wan22-native-calibration", "run_id": "fixture", "status": "completed",
            "phases": [{"cells": [{"config": {"policy": policy}} for policy in policies]}],
            "requests": requests,
            "budget": {"sessions": [{"identity": {"gpu_uuids": [f"GPU-{i}" for i in range(8)]}}]},
        }
        profile = MODULE.build_profile(record, facts, "Wan2.2-TI2V-5B")
        self.assertEqual(profile["source"]["architecture"]["tokens"], 5824)
        self.assertEqual(profile["source"]["gpu_uuids"], [f"GPU-{i}" for i in range(8)])
        self.assertLess(profile["source"]["held_out_max_relative_error_pct"], 1e-8)
        self.assertEqual(profile["classes"]["sd35-preflight"]["service_ms"], 250.0)
        self.assertEqual(profile["classes"]["wan22-preflight"]["step_compute_ms"], 500.0)
        self.assertTrue(all(value == 2 for value in profile["source"]["steady_samples"].values()))
        self.assertEqual(profile["provenance"], "mixed")
        self.assertIn("cfg2sp3", profile["classes"]["wan22-preflight"]["forms"])
        self.assertEqual(profile["source"]["execution_cost_mode"], "placement_bound_form_table_with_analytical_fallback")
        for form, key in (("sp1", "0-0"), ("sp3", "0-2"), ("sp4", "0-3"), ("cfg2sp3", "0-5")):
            self.assertIn(key, profile["classes"]["wan22-preflight"]["forms"][form]["placements"])
        self.assertNotIn("groups", profile["links"])
        self.assertNotIn("sp_overlap", profile["links"]["2"])
        batch_record = copy.deepcopy(record)
        for index in range(3):
            batch_record["requests"].append({
                "cell": 0, "kind": "image", "status": "completed", "request_id": f"b{index}",
                "model_ms": 150.0, "batch_model_ms": 450.0, "batch_size": 3,
                "batch_request_ids": ["b0", "b1", "b2"], "model_time_scope": "amortized_batch_cost",
            })
        batched = MODULE.build_profile(batch_record, facts, "Wan2.2-TI2V-5B")
        self.assertEqual(batched["classes"]["sd35-preflight"]["batch_ms"]["3"], 450.0)
        self.assertEqual(batched["source"]["image_batch_samples"]["3"], 1)
        legacy = copy.deepcopy(record)
        for row in legacy["requests"]:
            for sample in row.get("advance_samples", []):
                sample.pop("group", None)
        unbound = MODULE.build_profile(legacy, facts, "Wan2.2-TI2V-5B")
        self.assertNotIn("forms", unbound["classes"]["wan22-preflight"])
        self.assertEqual(unbound["source"]["execution_cost_mode"], "analytical_fallback_only")
        self.assertEqual(unbound["source"]["unbound_measurements"], len(policies))


if __name__ == "__main__":
    unittest.main()
