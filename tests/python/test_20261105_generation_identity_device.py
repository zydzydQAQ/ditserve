"""Unit tests for text_encoder_device in generation_identity and bind_generation_identity (2026-11-05)."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "workers", ROOT / "benchmarks", ROOT / "tests" / "python"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ditserve.evaluation.artifacts import evaluate_artifacts, generation_identity  # noqa: E402
from ditserve.runtime.worker import bind_generation_identity  # noqa: E402


def _make_sd35_generation(text_encoder_device: str, *, seed: int = 202) -> dict:
    return {
        "model": "sd35",
        "backend": "sd35",
        "model_path": "/shared/data/DITSERVE/models/sd3.5-medium",
        "revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
        "dtype": "bfloat16",
        "text_encoder_dtype": "bfloat16",
        "text_encoder_device": text_encoder_device,
        "attention": "flash",
        "prompt": "A lighthouse at dusk.",
        "negative_prompt": "blurry, low quality",
        "height": 1024,
        "width": 1024,
        "steps": 28,
        "frames": 1,
        "batch_size": 1,
        "guidance_scale": 4.5,
        "max_sequence_length": 256,
        "seed": seed,
    }


def _make_wan_generation(*, text_encoder_device: str = "cpu", seed: int = 201) -> dict:
    return {
        "model": "wan",
        "backend": "wan",
        "model_path": "/shared/data/DITSERVE/models/wan",
        "revision": "wan21-official",
        "dtype": "float16",
        "vae_dtype": "bfloat16",
        "text_encoder_dtype": "bfloat16",
        "text_encoder_device": text_encoder_device,
        "attention": "flash",
        "prompt": "A red ceramic kettle on a wooden table.",
        "negative_prompt": "",
        "height": 480,
        "width": 832,
        "frames": 81,
        "steps": 50,
        "guidance_scale": 5.0,
        "seed": seed,
    }


class GenerationIdentityDeviceTests(unittest.TestCase):
    def test_generation_identity_includes_text_encoder_device(self):
        cpu_gen = _make_sd35_generation("cpu")
        cuda_gen = _make_sd35_generation("cuda")

        cpu_id = generation_identity(cpu_gen)
        cuda_id = generation_identity(cuda_gen)

        self.assertIsNotNone(cpu_id)
        self.assertIsNotNone(cuda_id)
        self.assertEqual(cpu_id["text_encoder_device"], "cpu")
        self.assertEqual(cuda_id["text_encoder_device"], "cuda")
        self.assertNotEqual(cpu_id, cuda_id)

        # Two CPU-T5 generations with the same prompt/seed have identical generation_identity
        cpu_gen_2 = _make_sd35_generation("cpu")
        self.assertEqual(cpu_id, generation_identity(cpu_gen_2))

    def test_cpu_vs_cuda_text_encoder_device_yields_generation_identity_mismatch(self):
        candidate = {"generation": _make_sd35_generation("cpu")}
        reference = {"generation": _make_sd35_generation("cuda")}

        result = evaluate_artifacts(candidate, reference)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "generation_identity_mismatch")

    def test_wan_vs_wan_generation_identity_is_unchanged(self):
        wan_a = _make_wan_generation(text_encoder_device="cpu")
        wan_b = _make_wan_generation(text_encoder_device="cpu")
        self.assertEqual(generation_identity(wan_a), generation_identity(wan_b))
        self.assertEqual(generation_identity(wan_a)["text_encoder_device"], "cpu")

    def test_bind_generation_identity_copies_text_encoder_device_from_worker_config(self):
        worker_config = {
            "model_path": "/shared/data/DITSERVE/models/sd3.5-medium",
            "revision": "rev-1",
            "dtype": "bfloat16",
            "text_encoder_dtype": "bfloat16",
            "text_encoder_device": "cpu",
            "attention": "flash",
        }
        result = {"generation": {"prompt": "test"}}
        bind_generation_identity(result, worker_config)
        self.assertEqual(result["generation"]["text_encoder_device"], "cpu")

        # Does not overwrite explicit value already recorded by backend
        explicit = {"generation": {"prompt": "test", "text_encoder_device": "cuda"}}
        bind_generation_identity(explicit, worker_config)
        self.assertEqual(explicit["generation"]["text_encoder_device"], "cuda")


if __name__ == "__main__":
    unittest.main()
