import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ditserve.benchmark.backends import FixtureBackend, WanBackend, WAN_MODEL, WAN_REVISION, request_spec
from ditserve.benchmark.common import seal
from ditserve.benchmark.runner import _generation, _verify_execution_identity, build_profile, run_suite
from ditserve.benchmark.workload import build_trace


GPU_A = "3ab2fedc-ccc1-007d-f4a1-b6058b3479a3"
GPU_B = "a058066e-bde9-c534-efb1-d2a68c117681"


def identity(gpu_uuid=GPU_A):
    return dict(backend="wan", gpu=dict(uuid="GPU-" + gpu_uuid, name="fixture A100", total_memory_bytes=40000000000,
                                      compute_capability=[8, 0]),
                precision=dict(dit="float16", vae="float32", text_encoder="float32", latents="float32"),
                attention=dict(processor="native", sdpa_backend="MATH"), text_encoder_device="cpu",
                scheduler="UniPCMultistepScheduler", model_id=WAN_MODEL, revision=WAN_REVISION,
                torch_version="2.10.0+cu126", cuda_runtime_version="12.6", driver_version="560.00")


class BackendIdentityTests(unittest.TestCase):
    def test_wan_identity_reads_selected_physical_device_without_model_constructor(self):
        backend = WanBackend.__new__(WanBackend)
        properties = SimpleNamespace(uuid=GPU_A, name="fixture A100", total_memory=40000000000, major=8, minor=0)
        seen = []
        fake_torch = SimpleNamespace(__version__="2.10.0+cu126", version=SimpleNamespace(cuda="12.6"),
            cuda=SimpleNamespace(get_device_properties=lambda device: seen.append(device) or properties))
        backend.worker = SimpleNamespace(torch=fake_torch, device="cuda:1", attention="math")
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="560.00\n")) as query:
            actual = backend.execution_identity()
        self.assertEqual(seen, ["cuda:1"])
        self.assertEqual(actual["gpu"]["uuid"], "GPU-" + GPU_A)
        self.assertIn("--id=GPU-" + GPU_A, query.call_args[0][0])
        self.assertEqual(actual["logical_device"], "cuda:1")
        _verify_execution_identity(identity(), actual)

        backend.worker.attention = "flash"
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="560.00\n")):
            actual = backend.execution_identity()
        self.assertEqual(actual["attention"], dict(processor="native", sdpa_backend="FLASH_ATTENTION"))

    def test_wan_split_prepare_keeps_model_validation_and_request_binding(self):
        from unittest.mock import Mock
        backend = WanBackend.__new__(WanBackend)
        backend.worker = Mock()
        request = dict(request_id="x", prompt="example", seed=3, height=32, width=32, frames=5,
                       steps=2, model=WAN_MODEL, revision=WAN_REVISION)
        payload = backend.encode(request)
        backend.worker.encode_prompt.assert_called_once_with(request_spec(request))
        backend.prepare_encoded(request, payload)
        backend.worker.prepare_encoded.assert_called_once_with(request_spec(request), payload)
        for method, args in ((backend.encode, ()), (backend.prepare_encoded, (payload,))):
            with self.assertRaisesRegex(ValueError, "pinned"):
                method(dict(request, revision="unknown"), *args)

    def test_gpu_uuid_and_recipe_mismatches_fail_even_if_name_is_same(self):
        _verify_execution_identity(identity(), identity())
        for mutate, field in (
            (lambda x: x["gpu"].update(uuid="GPU-" + GPU_B), "gpu.uuid"),
            (lambda x: x["attention"].update(sdpa_backend="FLASH_ATTENTION"), "attention"),
            (lambda x: x["precision"].update(dit="bfloat16"), "precision"),
            (lambda x: x.update(driver_version="new-driver"), "driver_version"),
        ):
            changed = copy.deepcopy(identity())
            mutate(changed)
            with self.assertRaisesRegex(ValueError, field):
                _verify_execution_identity(identity(), changed)
        with self.assertRaisesRegex(ValueError, "lacks a physical GPU"):
            _verify_execution_identity({}, identity())

    def test_run_gpu_mismatch_is_saved_and_rejected_before_smoke_or_inference(self):
        config = dict(model=dict(id=WAN_MODEL, revision=WAN_REVISION),
                      shapes=dict(short=dict(height=32, width=32, frames=5, steps=2, weight=1.)),
                      cache=dict(enabled=False, threshold=.08, max_continuous=3), guidance_scale=5.)
        dataset = seal(dict(schema_version="ditserve.benchmark.dataset.v1", kind="test",
                            samples=[dict(sample_id="dev", prompt="dev", split="dev"),
                                     dict(sample_id="test", prompt="test", split="test")]))
        class FakeWan(FixtureBackend):
            smoke_called = False
            def execution_identity(self):
                return identity(GPU_B)
            def smoke(self, *args):
                self.smoke_called = True
                raise AssertionError("must not generate before identity check")
        adapter = FakeWan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = build_profile(config, dataset, "cpu-test", root / "profile")
            profile = seal(dict(profile, backend="wan", execution_identity=identity()))
            trace = build_trace(config, dataset, profile, pattern="offline_all", count=1)
            with patch("ditserve.benchmark.runner.make_backend", return_value=adapter):
                result = run_suite(trace, ["fcfs"], "wan", root / "run")
            self.assertFalse(adapter.smoke_called)
            self.assertEqual(result["status"], "unverified")
            self.assertIn("gpu.uuid", result["error"])
            self.assertFalse(result["measured"])
            self.assertEqual(result["policies"]["fcfs"]["status"], "not_run")
            self.assertTrue((root / "run" / "run.json").is_file())
            self.assertTrue((root / "run" / "profile.json").is_file())

    def test_default_cache_limit_matches_workload_and_worker_specs(self):
        config = dict(model=dict(id="fixture", revision="v1"),
                      shapes=dict(short=dict(height=32, width=32, frames=5, steps=2, weight=1.)))
        sample = dict(sample_id="dev", prompt="development")
        request = _generation(config, sample, "short")
        self.assertEqual(request["max_continuous"], 3)
        request.pop("max_continuous")
        self.assertEqual(request_spec(request)["max_continuous"], 3)


if __name__ == "__main__":
    unittest.main()
