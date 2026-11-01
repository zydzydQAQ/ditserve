import importlib.util
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
                    {"steady": True, "model_ms": step_ms},
                    {"steady": True, "model_ms": step_ms},
                ],
                "transfer_export_ms": 2.0, "dispatch_gap_ms": 4.0, "timed_quanta": 2,
            })
        requests.append({"cell": 0, "kind": "image", "status": "completed", "model_ms": 250.0})
        record = {
            "experiment": "kairos-wan22-native-calibration", "run_id": "fixture",
            "phases": [{"cells": [{"config": {"policy": policy}} for policy in policies]}],
            "requests": requests,
        }
        profile = MODULE.build_profile(record, facts, "Wan2.2-TI2V-5B")
        self.assertEqual(profile["source"]["architecture"]["tokens"], 5824)
        self.assertLess(profile["source"]["held_out_max_relative_error_pct"], 1e-8)
        self.assertEqual(profile["classes"]["sd35-preflight"]["service_ms"], 250.0)
        self.assertEqual(profile["classes"]["wan22-preflight"]["step_compute_ms"], 500.0)
        self.assertTrue(all(value == 2 for value in profile["source"]["steady_samples"].values()))
        self.assertEqual(profile["provenance"], "mixed")
        self.assertNotIn("groups", profile["links"])
        self.assertNotIn("sp_overlap", profile["links"]["2"])


if __name__ == "__main__":
    unittest.main()
