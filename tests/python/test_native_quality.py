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
        "experiment": "fixture", "run_id": "exact" if not fold else "approx",
        "status": "completed", "source": {"core_sha256": "a" * 64},
        "requests": [{
            "kind": "video", "status": "completed", "dataset_sample_id": "sample-0",
            "spec": spec, "forms": ["cfg2sp3"],
            "outputs": [{"generation": generation, "output": artifact(path, array)}],
            "cfg_fold": {"folded_cfg_steps": fold},
        }],
    }


class NativeQualityTest(unittest.TestCase):
    def test_pairs_report_distribution_and_exact_matches(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(7)
            ref = rng.random((1, 2, 4, 4, 3), dtype=np.float32)
            exact = MODULE.compare_runs(run(root / "ref.npy", ref), run(root / "same.npy", ref), with_lpips=False)
            self.assertEqual(exact["status"], "completed")
            self.assertEqual(exact["metrics"]["exact_match_count"], 1)
            changed = ref.copy()
            changed[0, 0, 0, 0, 0] -= 0.1
            result = MODULE.compare_runs(run(root / "ref2.npy", ref), run(root / "changed.npy", changed, fold=2), with_lpips=False)
            self.assertEqual(result["pairs"][0]["candidate_cfg_folded_steps"], 2)
            self.assertLess(result["pairs"][0]["ssim"], 1.0)
            self.assertIsNotNone(result["pairs"][0]["psnr_db"])

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
