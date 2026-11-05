"""Regression tests for profile fitting and serving/worker config defaults (2026-11-05).

Each test failed on 7ee8beb: image finish_ms taken from the completion timestamp,
continuous batching auto-enabled from a profile, CB-off serving still sending
CB/packing/bubble-free to workers, S4/CB worker defaults, unimplemented memory
strategies reaching worker configs, and paper_results requiring workers/.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
for _path in (BENCHMARKS, ROOT / "tests" / "python"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cluster_config  # noqa: E402
import fit_native_profile  # noqa: E402
import native_serving  # noqa: E402
import native_study  # noqa: E402
import policy_display  # noqa: E402
from test_native_serving import pool, profile as serving_profile  # noqa: E402


UUIDS = [f"GPU-{index}" for index in range(8)]
LEGACY_IMAGE_OPTIONS = {"profile": False, "cache": False}
UNIMPLEMENTED = ("s0_naive_gpu", "s1_weight_swap", "s3_both_encoders_cpu", "s4_disaggregated_pool")
VIDEO_SPEC = dict(frames=49, steps=20, height=512, width=896, guidance_scale=5.0)
WAN22 = {
    "transformer_config": {
        "patch_size": [1, 2, 2], "num_attention_heads": 24,
        "attention_head_dim": 128, "num_layers": 30, "out_channels": 48,
    },
    "vae_config": {"scale_factor_temporal": 4, "scale_factor_spatial": 16},
}


def calibration_record(image_rows, cells_extra=None):
    """An atomic native calibration record shaped like study.cpp output."""
    arch = fit_native_profile.architecture(WAN22, VIDEO_SPEC)
    policies = sorted(fit_native_profile.FIT_POLICIES)
    requests = []
    for index, policy in enumerate(policies):
        step_ms = fit_native_profile.predict(500.0, arch, *fit_native_profile.form_of(policy), 12.0, 1.0, 0.05)
        requests.append({
            "cell": index, "kind": "video", "status": "completed", "model_ms": 20 * step_ms + 100.0,
            "outputs": [{"model_ms": 100.0}], "spec": VIDEO_SPEC, "encode_model_ms": 20.0,
            "advance_samples": [{"steady": True, "model_ms": step_ms}],
            "arrival_ms": 0.0, "finish_ms": 61234.0,
        })
    requests.extend(image_rows)
    cells = [{"config": {"policy": policy}} for policy in policies]
    for index, extra in (cells_extra or {}).items():
        cells[index].update(extra)
    return {
        "experiment": "kairos-wan22-native-calibration", "run_id": "regression", "status": "completed",
        "phases": [{"cells": cells}], "requests": requests,
    }


def serving(profile, **overrides):
    pools = {"image": pool("diffusiondb"), "video": pool("vidprom")}
    arguments = dict(seed=201, load=1.0, pattern="poisson", video_fraction=0.2, count=20, policies=["ditserve"])
    arguments.update(overrides)
    return native_serving.build_serving_config("wan22", UUIDS, profile, pools, **arguments)


def assert_no_cb(test, config):
    phase = config["phases"][0]
    for key in ("continuous_batch", "sequence_packing", "bubble_free"):
        test.assertNotIn(key, phase["options"])
    test.assertNotIn("continuous_batch", config["benchmark"])
    for name, worker in config["workers"].items():
        if worker["backend"] == "sd35":
            test.assertEqual(worker["config"]["options"], LEGACY_IMAGE_OPTIONS, name)


class ImageCostFitTest(unittest.TestCase):
    def test_atomic_calibration_finish_ms_is_never_the_completion_timestamp(self):
        # study.cpp sets row["finish_ms"]=now() at completion; atomic rows carry no stage durations.
        image = {"cell": 0, "kind": "image", "status": "completed", "model_ms": 4200.0,
                 "spec": {"steps": 28, "height": 1024, "width": 1024},
                 "arrival_ms": 1000.0, "finish_ms": 48211.5, "prepare_ms": 39000.0}
        fitted = fit_native_profile.build_profile(calibration_record([image]), {"models": {"W": WAN22}}, "W")
        image_cost = fitted["classes"]["sd35-preflight"]
        self.assertEqual(image_cost["service_ms"], 4200.0)
        self.assertNotIn("finish_ms", image_cost)
        self.assertNotIn("prepare_ms", image_cost)
        # No synthetic continuous-batch table is written into a measured profile.
        self.assertNotIn("batch_step_ms", image_cost)
        provenance = fitted["source"]["image_cost_provenance"]
        self.assertEqual(provenance, {"service_ms": "measured", "steps": "request_spec"})

    def test_continuous_batch_stage_durations_and_rounds_are_measured_only(self):
        rows = [
            {"cell": 0, "kind": "image", "status": "completed", "model_ms": 4200.0, "request_id": f"i{index}",
             "spec": {"steps": 28, "height": 1024, "width": 1024}, "finish_ms": 50000.0 + index,
             "prepare_cb_ms": prepare, "decode_cb_ms": decode}
            for index, (prepare, decode) in enumerate(((12.0, 60.0), (14.0, 70.0), (16.0, 80.0)))
        ]
        samples = {"samples": [{"batch_size": 1, "model_ms": 31.0}, {"batch_size": 2, "model_ms": 55.0},
                               {"batch_size": 1, "model_ms": 33.0}]}
        record = calibration_record(rows, {0: {"continuous_batch": samples}})
        fitted = fit_native_profile.build_profile(record, {"models": {"W": WAN22}}, "W")
        image_cost = fitted["classes"]["sd35-preflight"]
        self.assertEqual(image_cost["prepare_ms"], 14.0)
        self.assertEqual(image_cost["finish_ms"], 70.0)
        self.assertEqual(image_cost["batch_step_ms"], {"1": 32.0, "2": 55.0})
        provenance = fitted["source"]["image_cost_provenance"]
        self.assertEqual(provenance["finish_ms"], "measured_continuous_batch_stage")
        self.assertEqual(provenance["batch_step_ms"], "measured_continuous_batch_rounds")
        self.assertEqual(fitted["source"]["continuous_batch_round_samples"], {"1": 2, "2": 1})


class ServingContinuousBatchTest(unittest.TestCase):
    def test_batch_step_profile_without_continuous_batch_argument_stays_atomic(self):
        profile = serving_profile()
        profile["classes"]["sd35-preflight"].update(
            steps=28, prepare_ms=15.0, finish_ms=65.0,
            batch_step_ms=cluster_config.default_sd35_batch_step_ms(250.0, 28, resolution=1024))
        config = serving(profile)
        assert_no_cb(self, config)
        self.assertEqual(config["phases"][0]["options"]["image_batch"], 1)
        self.assertEqual(config["benchmark"]["image_batch"]["max_requests"], 1)

    def test_explicit_zero_and_atomic_batches_send_cb_off_to_workers(self):
        assert_no_cb(self, serving(serving_profile(), continuous_batch=0))
        measured = serving_profile()
        measured["classes"]["sd35-preflight"]["batch_ms"] = {"2": 400.0}
        assert_no_cb(self, serving(measured, image_batch=2))
        assert_no_cb(self, serving(measured, image_batch=2, continuous_batch=0))
        with self.assertRaisesRegex(ValueError, "require continuous_batch > 0"):
            serving(serving_profile(), continuous_batch=0, sequence_packing=True)

    def test_explicit_opt_in_matches_workers_and_labels_synthetic_costs(self):
        config = serving(serving_profile(), continuous_batch=4, sequence_packing=False)
        options = config["phases"][0]["options"]
        self.assertEqual((options["continuous_batch"], options["sequence_packing"], options["bubble_free"]),
                         (4, False, True))
        self.assertEqual(config["workers"]["image0"]["config"]["options"],
                         dict(LEGACY_IMAGE_OPTIONS, continuous_batch=True, sequence_packing=False, bubble_free=True))
        recorded = config["benchmark"]["continuous_batch"]
        self.assertEqual(recorded["max_requests"], 4)
        self.assertEqual(recorded["cost_provenance"]["batch_step_ms"], "synthetic_default_table")
        self.assertEqual(recorded["cost_provenance"]["finish_ms"], "synthetic_placeholder")
        self.assertEqual(config["phases"][0]["profile"]["source"]["image_cost_provenance"]["finish_ms"],
                         "synthetic_placeholder")

    def test_measured_table_is_never_extended_synthetically(self):
        profile = serving_profile()
        profile["classes"]["sd35-preflight"].update(batch_step_ms={"1": 30.0, "2": 50.0})
        with self.assertRaisesRegex(ValueError, "largest size"):
            serving(profile, continuous_batch=4)


class WorkerDefaultsTest(unittest.TestCase):
    def test_make_workers_defaults_are_the_legacy_worker_options(self):
        _, workers, _, _ = cluster_config.make_workers("wan21", UUIDS[:2], video_spec={}, image_spec={})
        self.assertEqual(workers["wan0"]["config"]["options"], {"execution": "block"})
        self.assertEqual(workers["image0"]["config"]["options"], LEGACY_IMAGE_OPTIONS)
        fold = {"folded_steps": [2]}
        _, folded, _, _ = cluster_config.make_workers("wan22", UUIDS[:2], video_spec={}, image_spec={}, cfg_fold=fold)
        self.assertEqual(folded["wan1"]["config"]["options"], {"execution": "block", "cfg_fold": fold})
        self.assertEqual(folded["image1"]["config"]["options"], LEGACY_IMAGE_OPTIONS)

    def test_preflight_and_calibration_defaults_are_legacy(self):
        preflight = cluster_config.build_config("wan21", UUIDS)
        self.assertEqual(preflight["workers"]["image0"]["config"]["options"], LEGACY_IMAGE_OPTIONS)
        self.assertEqual(set(preflight["workers"]["wan0"]["config"]["options"]), {"execution", "cfg_fold"})
        self.assertEqual(preflight["phases"][0]["options"], {
            "sp_transport": "export", "forms": ["cfg2sp3"], "quantum_blocks": 4,
            "async_rank0_decode": True, "image_batch": 1, "tiered_memory": False})
        self.assertEqual(preflight["phases"][0]["profile"]["classes"]["sd35-preflight"],
                         {"kind": "image", "service_ms": 1000.0})
        self.assertNotIn("memory_strategy", preflight["benchmark"])
        calibration = native_study.build_calibration_config("wan22", UUIDS)
        self.assertEqual(calibration["workers"]["wan0"]["config"]["options"], {"execution": "block"})
        self.assertEqual(calibration["workers"]["image0"]["config"]["options"], LEGACY_IMAGE_OPTIONS)

    def test_cli_defaults_are_legacy_and_opt_in_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            default_out, opt_in_out = Path(directory, "default.json"), Path(directory, "s2-cb2.json")
            base = ["cluster_config.py", "--model", "wan21", "--gpu-uuids", ",".join(UUIDS[:2])]
            with patch.object(sys, "argv", base + ["--output", str(default_out)]):
                cluster_config.main()
            with patch.object(sys, "argv", base + ["--memory-strategy", "s2_wan_cpu_sd35_gpu",
                                                   "--continuous-batch", "2", "--output", str(opt_in_out)]):
                cluster_config.main()
            default = json.loads(default_out.read_text(encoding="utf-8"))
            opt_in = json.loads(opt_in_out.read_text(encoding="utf-8"))
        self.assertEqual(default["workers"]["image0"]["config"]["options"], LEGACY_IMAGE_OPTIONS)
        self.assertEqual(opt_in["benchmark"]["memory_strategy"]["strategy"], "s2_wan_cpu_sd35_gpu")
        self.assertEqual(opt_in["benchmark"]["continuous_batch"]["max_requests"], 2)
        self.assertTrue(opt_in["workers"]["image0"]["config"]["options"]["continuous_batch"])
        self.assertTrue(opt_in["workers"]["wan0"]["config"]["options"]["encoder_pool"]["enabled"])


class MemoryStrategyTest(unittest.TestCase):
    def test_unimplemented_strategies_are_rejected_for_worker_configs(self):
        self.assertEqual(cluster_config.WORKER_MEMORY_STRATEGIES, ("s2_wan_cpu_sd35_gpu",))
        for strategy in UNIMPLEMENTED:
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(ValueError, "cross_modal_memory_study"):
                    cluster_config.make_workers("wan21", UUIDS[:2], video_spec={}, image_spec={},
                                                memory_strategy=strategy)
                with self.assertRaisesRegex(ValueError, "analytical only"):
                    cluster_config.build_config("wan21", UUIDS[:2], memory_strategy=strategy)
                with self.assertRaisesRegex(ValueError, "analytical only"):
                    serving(serving_profile(), memory_strategy=strategy)
                # Descriptors stay available for the analytical S0-S4 study.
                self.assertEqual(cluster_config.memory_strategy_descriptor(strategy)["strategy"], strategy)
                self.assertEqual(cluster_config.memory_strategy_Descriptor(strategy)["strategy"], strategy)
        with tempfile.TemporaryDirectory() as directory:
            argv = ["cluster_config.py", "--model", "wan21", "--gpu-uuids", ",".join(UUIDS[:2]),
                    "--memory-strategy", "s4_disaggregated_pool", "--output", str(Path(directory, "x.json"))]
            with patch.object(sys, "argv", argv), patch("sys.stderr"), self.assertRaises(SystemExit):
                cluster_config.main()

    def test_accepted_strategy_records_only_applied_worker_knobs(self):
        config = cluster_config.build_config("wan22", UUIDS[:4], memory_strategy="s2_wan_cpu_sd35_gpu")
        recorded = config["benchmark"]["memory_strategy"]
        self.assertEqual(set(recorded), {"strategy", "code", "label", "applied_worker_options"})
        flat = json.dumps(recorded)
        for claim in ("tiered_memory", "text_encoder_device", "weight_swap_on_switch"):
            self.assertNotIn(claim, flat)
        applied = recorded["applied_worker_options"]
        wan = config["workers"]["wan0"]["config"]["options"]
        image = config["workers"]["image0"]["config"]["options"]
        self.assertEqual({key: wan[key] for key in applied["wan"]}, applied["wan"])
        self.assertEqual({key: image[key] for key in applied["image"]}, applied["image"])
        self.assertEqual(set(wan) - set(applied["wan"]), {"execution", "cfg_fold"})
        self.assertEqual(set(image) - set(applied["image"]), set(LEGACY_IMAGE_OPTIONS))
        self.assertFalse(config["phases"][0]["options"]["tiered_memory"])
        self.assertNotIn("memory_strategy", config["phases"][0]["options"])
        served = serving(serving_profile(), memory_strategy="s2_wan_cpu_sd35_gpu")
        self.assertEqual(served["benchmark"]["memory_strategy"], recorded)


class PaperResultsImportTest(unittest.TestCase):
    def test_display_tables_are_shared_and_reexported(self):
        self.assertIs(native_serving.POLICY_DISPLAY, policy_display.POLICY_DISPLAY)
        self.assertIs(native_serving.IN_FRAMEWORK_MECHANISM_DISPLAY, policy_display.IN_FRAMEWORK_MECHANISM_DISPLAY)

    def test_paper_results_imports_with_only_benchmarks_on_path(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); sys.modules['ditserve'] = None; "
                "import paper_results; assert 'native_serving' not in sys.modules; "
                "assert 'cluster_config' not in sys.modules; print(paper_results.mechanism_label('tetriserve'))")
        result = subprocess.run([sys.executable, "-I", "-c", code, str(BENCHMARKS)],
                                capture_output=True, text=True, cwd=tempfile.gettempdir(), check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "TetriServe-policy")
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        environment["PYTHONPATH"] = str(BENCHMARKS)
        script = subprocess.run([sys.executable, str(BENCHMARKS / "paper_results.py"), "--help"],
                                capture_output=True, text=True, cwd=tempfile.gettempdir(), env=environment, check=False)
        self.assertEqual(script.returncode, 0, script.stderr)


if __name__ == "__main__":
    unittest.main()
