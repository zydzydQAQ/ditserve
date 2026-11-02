import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
SPEC = importlib.util.spec_from_file_location("native_quality", ROOT / "benchmarks" / "native_quality.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def artifact(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return {
        "path": str(path), "sha256": MODULE.sha256(path), "shape": list(array.shape),
        "dtype": str(array.dtype), "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def run(path, array, *, fold=0):
    spec = dict(seed=201, prompt="a kettle", negative_prompt="", steps=20,
                frames=array.shape[1], height=array.shape[2], width=array.shape[3], guidance_scale=5.0)
    generation = dict(spec, model_path="/models/wan", revision="revision", dtype="float16",
                      vae_dtype="bfloat16", text_encoder_dtype="float32", attention="flash",
                      engine="xdit", xdit_revision="xdit-revision", approximate=bool(fold), cfg_folded_steps=fold,
                      runtime_identity={"torch": "test", "cuda": "test"})
    return {
        "schema": "ditserve.run.v1", "experiment": "fixture", "run_id": "exact" if not fold else "approx",
        "status": "completed", "source": {"core_sha256": "a" * 64},
        "config": {"benchmark": {"native_serving": True}},
        "requests": [{
            "kind": "video", "status": "completed", "dataset_sample_id": "sample-0",
            "spec": spec, "forms": ["cfg2sp3"],
            "outputs": [{"generation": generation, "output": artifact(path, array),
                         "path_history": [{"step_index": index, "cfg_folded": 12 <= index < 12 + fold}
                                          for index in range(spec["steps"])]}],
            "cfg_fold": {"full_cfg_steps": spec["steps"] - fold, "folded_cfg_steps": fold,
                         "folded_step_indices": list(range(12, 12 + fold))},
        }],
    }


class NativeQualityTest(unittest.TestCase):
    def test_pairs_report_distribution_and_exact_matches(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(7)
            ref = rng.random((1, 2, 4, 4, 3), dtype=np.float32)
            exact = MODULE.compare_runs(run(root / "ref.npy", ref), run(root / "same.npy", ref), with_lpips=False,
                                        expected_fold_count=0)
            self.assertEqual(exact["status"], "completed")
            self.assertEqual(exact["metrics"]["exact_match_count"], 1)
            exact_without_stepper = run(root / "exact-without-stepper.npy", ref)
            exact_without_stepper["requests"][0].pop("cfg_fold")
            exact = MODULE.compare_runs(run(root / "ref-exact.npy", ref), exact_without_stepper,
                                        with_lpips=False, expected_fold_count=0)
            self.assertEqual(exact["status"], "completed")
            changed = ref.copy()
            changed[0, 0, 0, 0, 0] -= 0.1
            result = MODULE.compare_runs(run(root / "ref2.npy", ref), run(root / "changed.npy", changed, fold=2), with_lpips=False)
            self.assertEqual(result["pairs"][0]["candidate_cfg_folded_steps"], 2)
            self.assertLess(result["pairs"][0]["ssim"], 1.0)
            self.assertIsNotNone(result["pairs"][0]["psnr_db"])
            matched = MODULE.compare_runs(run(root / "ref3.npy", ref), run(root / "fixed.npy", changed, fold=2),
                                          with_lpips=False, expected_fold_count=2)
            self.assertEqual(matched["fold_count_requirement"], {"value": 2, "matched": 1})
            mismatched = MODULE.compare_runs(run(root / "ref4.npy", ref), run(root / "wrong.npy", changed, fold=2),
                                              with_lpips=False, expected_fold_count=1)
            self.assertEqual(mismatched["status"], "incomplete")
            missing_history = run(root / "missing-history.npy", changed, fold=2)
            missing_history["requests"][0]["outputs"][-1]["path_history"].pop()
            result = MODULE.compare_runs(run(root / "ref5.npy", ref), missing_history, with_lpips=False)
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("complete solver-step history", result["failures"][0]["error"])

    def test_lpips_gate_is_per_sample_and_fail_closed(self):
        from unittest.mock import patch
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ref = np.zeros((1, 2, 4, 4, 3), dtype=np.float32)
            metrics = {"psnr_db": 20.0, "psnr_is_infinite": False, "lpips_mean": 0.2,
                       "temporal_delta_rmse": 0.0, "exact_match": False, "status": "measured"}
            with patch.object(MODULE, "evaluate_quality", return_value=metrics):
                result = MODULE.compare_runs(run(root / "ref.npy", ref), run(root / "candidate.npy", ref, fold=1),
                                             max_lpips=0.1)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["numeric_qualification"], "failed")
            self.assertEqual(result["lpips_threshold"]["passed"], 0)
            with self.assertRaisesRegex(ValueError, "cannot be checked"):
                MODULE.compare_runs(run(root / "ref2.npy", ref), run(root / "candidate2.npy", ref, fold=1),
                                    with_lpips=False, max_lpips=0.1)
            with patch.object(MODULE, "evaluate_quality", return_value=dict(metrics, lpips_mean=None)):
                unavailable = MODULE.compare_runs(run(root / "ref3.npy", ref), run(root / "candidate3.npy", ref, fold=1))
            self.assertEqual(unavailable["status"], "incomplete")

    def test_paired_bootstrap_requires_equal_actual_compute(self):
        import copy
        candidate = {"status": "completed", "source": {"reference_run_sha256": "r"}, "pairs": [
            {"dataset_sample_id": str(i), "seed": 201, "prompt": "p", "candidate_cfg_folded_steps": 2,
             "psnr_db": 31.0 + i, "ssim": 0.95, "lpips_mean": 0.1 + i / 10} for i in range(2)]}
        control = copy.deepcopy(candidate)
        for row in control["pairs"]:
            row.update(psnr_db=30.0, ssim=0.94, lpips_mean=0.2)
        result = MODULE.paired_bootstrap(candidate, control, resamples=100)
        self.assertEqual(result["matched_pairs"], 2)
        self.assertEqual(len(result["metrics"]["lpips_mean"]["ci95"]), 2)
        control["pairs"][0]["candidate_cfg_folded_steps"] = 3
        with self.assertRaisesRegex(ValueError, "actual fold counts"):
            MODULE.paired_bootstrap(candidate, control, resamples=100)

    def test_matched_fold_bootstrap_stratifies_actual_compute(self):
        import copy
        candidate = {"status": "completed", "source": {"reference_run_sha256": "r"}, "pairs": [
            {"dataset_sample_id": str(i), "seed": 201, "prompt": "p",
             "candidate_cfg_folded_steps": 2 if i < 2 else 3,
             "psnr_db": 31.0 + i, "ssim": 0.95, "lpips_mean": 0.1 + i / 10} for i in range(4)]}
        controls = {}
        for count in (2, 3):
            control = copy.deepcopy(candidate)
            control["pairs"] = [row for row in control["pairs"] if row["candidate_cfg_folded_steps"] == count]
            controls[count] = control
        result = MODULE.matched_fold_bootstrap(candidate, controls, resamples=100)
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["matched_pairs"], 4)
        self.assertEqual(set(result["strata"]), {"2", "3"})
        missing = MODULE.matched_fold_bootstrap(candidate, {2: controls[2]}, resamples=100)
        self.assertEqual(missing["status"], "unverified")
        self.assertEqual(missing["strata"]["3"]["reason"], "no fixed-N control")

    def test_hash_mismatch_fails_instead_of_substituting_data(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ref = np.zeros((1, 2, 4, 4, 3), dtype=np.float32)
            reference = run(root / "reference.npy", ref)
            np.save(root / "reference.npy", np.ones_like(ref), allow_pickle=False)
            with self.assertRaises(ValueError):
                MODULE.checked_array(reference["requests"][0])


if __name__ == "__main__":
    unittest.main()
