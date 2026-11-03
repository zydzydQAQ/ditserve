"""Verify native checkpoint binding, exact requests and output qualification boundaries."""
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import external_xdit_native as native
import external_xdit as source


class NativeXDiTTests(unittest.TestCase):
    def fixture(self, directory, model="wan21"):
        base = Path(directory)
        root = base / "model"
        values = {"model_index.json": {"_class_name": "WanPipeline", "transformer_2": [None, None]},
                  "transformer/config.json": {"num_attention_heads": 12 if model == "wan21" else 24},
                  "scheduler/scheduler_config.json": {"flow_shift": 3.0 if model == "wan21" else 5.0},
                  "transformer/weights.safetensors": "a byte fixture, never loaded as weights"}
        for name, value in values.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        rows = [{"path": n, "bytes": (root/n).stat().st_size, "sha256": source.digest((root/n).read_bytes())} for n in values]
        manifest = base / "model.manifest.json"
        manifest.write_text(json.dumps({"schema": "ditserve.model-manifest.v1", "model": source.MODEL_IDS[model],
            "revision": native.REVISIONS[model], "component_paths": {"*": model}, "files": rows}))
        bindings = base / "models.json"
        bindings.write_text(json.dumps({model: {"path": str(manifest), "model_path": str(root),
                                              "sha256": source.digest(manifest.read_bytes())}}))
        return root, bindings, source.digest(bindings.read_bytes()), manifest

    def protocol(self):
        return json.loads(source.DEFAULT_PROTOCOL.read_bytes())

    def test_storage_boundary_is_fail_closed(self):
        def result(used, percent, returncode=0):
            return SimpleNamespace(returncode=returncode, stdout=f"Filesystem 1G-blocks Used Available Use% Mounted\n/dev/sdb 1024G {used}G 200G {percent}% /shared\n")
        with patch.object(native.subprocess, "run", return_value=result(790, 79)):
            self.assertEqual(native.check_storage()["used_gib"], 790)
        for value in [result(791, 79), result(789, 80), result(787, 77, 1), SimpleNamespace(returncode=0, stdout="bad")]:
            with patch.object(native.subprocess, "run", return_value=value):
                with self.assertRaisesRegex(ValueError, "storage"):
                    native.check_storage()

    def test_real_formal_and_quick_shapes_preserve_frames_resolution_guidance(self):
        protocol = self.protocol()
        for model, shape in [("wan21", "wan21-480p"), ("wan22", "wan22-720p")]:
            quick = native.request_plan(protocol, model, shape, "quick", "a dog", 42)
            formal = native.request_plan(protocol, model, shape, "formal", "a dog", 42)
            for key in ("height", "width", "num_frames", "guidance_scale", "prompt", "negative_prompt", "seed"):
                self.assertEqual(quick[key], formal[key])
            self.assertEqual(quick["num_inference_steps"], 1)
            self.assertEqual(formal["num_inference_steps"], 50)
            self.assertEqual(set(quick["changed_protocol_parameters"]), {"steps"})
            self.assertEqual(formal["changed_protocol_parameters"], {})
        self.assertEqual(formal["height"], 704)
        self.assertEqual(formal["width"], 1280)
        self.assertEqual(formal["num_frames"], 121)

    def test_wrong_model_development_formal_and_invalid_seed_fail_closed(self):
        protocol = self.protocol()
        for model, shape, mode, prompt, seed in [("wan21", "wan22-720p", "quick", "a", 42),
                ("wan21", "wan21-dev", "formal", "a", 42), ("sd35", "sd35-512", "quick", "a", 42),
                ("wan21", "wan21-480p", "quick", "", 42), ("wan21", "wan21-480p", "quick", "a", True),
                ("wan21", "wan21-480p", "quick", "a", -1)]:
            with self.assertRaises(ValueError):
                native.request_plan(protocol, model, shape, mode, prompt, seed)

    def test_both_official_models_bind_whole_snapshots(self):
        for model in ("wan21", "wan22"):
            with TemporaryDirectory() as directory:
                root, bindings, sha, manifest = self.fixture(directory, model)
                result = native.verify_model(model, root, bindings, sha)
                self.assertEqual(result["verified_files"], 4)
                self.assertEqual(result["revision"], native.REVISIONS[model])
                self.assertEqual(result["model_id"], source.MODEL_IDS[model])
                self.assertEqual(result["flow_shift"], 3 if model == "wan21" else 5)
                self.assertEqual(result["manifest_sha256"], source.digest(manifest.read_bytes()))

    def test_changed_weights_extra_files_and_wrong_frozen_binding_rejected(self):
        with TemporaryDirectory() as directory:
            root, bindings, sha, manifest = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                native.verify_model("wan21", root, bindings, "0"*64)
            file = root/"transformer/weights.safetensors"
            before = file.read_bytes()
            file.write_bytes(b"x" + before[1:])
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                native.verify_model("wan21", root, bindings, sha)
            file.write_bytes(before)
            (root/"unlisted.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "inventory incomplete"):
                native.verify_model("wan21", root, bindings, sha)

    def test_wan22_refuses_old_mixed_checkpoint_directory(self):
        with TemporaryDirectory() as directory:
            root, bindings, sha, manifest = self.fixture(directory, "wan22")
            other = Path(directory)/"old-mixed-model"
            other.mkdir()
            with self.assertRaisesRegex(ValueError, "root differs"):
                native.verify_model("wan22", other, bindings, sha)
            value = json.loads(manifest.read_text())
            value["component_paths"]["text_encoder/"] = "wan21"
            manifest.write_text(json.dumps(value))
            index = json.loads(bindings.read_text())
            index["wan22"]["sha256"] = source.digest(manifest.read_bytes())
            bindings.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "identity or component mapping"):
                native.verify_model("wan22", root, bindings, source.digest(bindings.read_bytes()))

    def test_upstream_request_changes_are_rejected(self):
        request = native.request_plan(self.protocol(), "wan22", "wan22-720p", "quick", "a dog", 42)
        native.check_effective_request(request, dict(request))
        for key in ("height", "width", "num_frames", "num_inference_steps", "guidance_scale", "negative_prompt", "seed", "prompt"):
            effective = dict(request)
            effective[key] = "different"
            with self.assertRaisesRegex(ValueError, "changed requested"):
                native.check_effective_request(request, effective)
        native.check_effective_request(request, dict(request, prompt=[request["prompt"]]))

    def test_native_config_disables_approximation_and_records_offload(self):
        request = native.request_plan(self.protocol(), "wan22", "wan22-small", "quick", "a", 42)
        config = native.runner_config("wan22", request, {"flow_shift": 5}, "/shared/output/DITSERVE/diagnostic", True)
        self.assertEqual(config["task"], "t2v")
        self.assertEqual(config["model"], source.MODEL_IDS["wan22"])
        self.assertEqual(config["num_frames"], 121)
        self.assertEqual(config["guidance_scale"], 5)
        self.assertEqual(config["flow_shift"], 5)
        self.assertTrue(config["enable_model_cpu_offload"])
        for key in ("use_cfg_parallel", "use_torch_compile", "use_teacache", "use_fbcache", "use_fp8_gemms", "use_fp8_comms", "enable_tiling"):
            self.assertFalse(config[key])
        self.assertIsNone(config["cache_method"])
        self.assertEqual(config["num_iterations"], 1)
        self.assertEqual(config["warmup_calls"], 0)

    def test_native_t2v_media_contract_matches_official_cli_empty_images(self):
        # Actual r1 GPU failures: Wan21 iterated None; Wan22 called len(None).
        # Explicit [] restores the pinned official CLI default before the real
        # dataclass/runner is constructed, without altering generation controls.
        configs = []
        for model, shape in (("wan21", "wan21-dev"), ("wan22", "wan22-dev")):
            request = native.request_plan(self.protocol(), model, shape, "quick", "a dog", 42)
            config = native.runner_config(model, request, {"flow_shift": 3 if model == "wan21" else 5},
                                          "/shared/output/DITSERVE/unit-contract-only")
            self.assertIsInstance(config["input_images"], list)
            self.assertEqual(list(config["input_images"]), [])
            self.assertEqual(len(config["input_images"]), 0)
            for key in ("height", "width", "num_frames", "num_inference_steps", "guidance_scale", "negative_prompt", "prompt", "seed"):
                self.assertEqual(config[key], request[key])
            configs.append(config)
        self.assertIsNot(configs[0]["input_images"], configs[1]["input_images"])

    def test_runtime_attention_selector_is_observed_not_inferred_from_request(self):
        from enum import Enum
        class Backend(Enum):
            CUDNN = "cuDNN"
            SDPA_EFFICIENT = "sdpa_efficient"
        def kernel():
            self.fail("metadata inspection must not invoke a kernel")
        class Processor:
            attention_function = staticmethod(kernel)
            is_cross_attention = False
        processor = Processor()
        pipeline = SimpleNamespace(transformer=SimpleNamespace(named_modules=lambda: iter([
            ("a", SimpleNamespace(processor=processor)), ("b", SimpleNamespace(processor=Processor()))])))
        state = SimpleNamespace(attention_backend=Backend.CUDNN, cross_attention_backend=None,
                                runtime_config=SimpleNamespace(use_hybrid_attn_schedule=False))
        entry = SimpleNamespace(impl=SimpleNamespace(target="kernel:cudnn"),
                                package="xfuser.core.attention.backends.sdpa", _resolved=kernel,
                                fallback=Backend.SDPA_EFFICIENT,
                                resolved=lambda: self.fail("must not resolve/import kernels"))
        modules = {"xfuser.core.distributed.runtime_state": SimpleNamespace(get_runtime_state=lambda: state),
                   "xfuser.core.attention.registry": SimpleNamespace(find=lambda backend: entry)}
        with patch.dict(native.sys.modules, modules):
            result = native.attention_observation(pipeline, "SDPA")
            enum_requested = native.attention_observation(pipeline, Backend.CUDNN)
        self.assertEqual(enum_requested["requested_attention_backend"], "CUDNN")
        json.dumps(enum_requested, allow_nan=False)
        self.assertEqual(result["requested_attention_backend"], "SDPA")
        self.assertEqual(result["runtime_selector_after_request"], "CUDNN")
        self.assertEqual(result["cross_attention_selector_after_request"], "CUDNN")
        self.assertEqual(result["selected_registry_entries"]["self"]["declared_fallback"], "SDPA_EFFICIENT")
        self.assertTrue(result["selected_registry_entries"]["self"]["cached_callable"]["qualname"].endswith("kernel"))
        self.assertEqual(result["transformer_processors"][0]["count"], 2)
        self.assertFalse(result["kernel_dispatch_verified"])
        self.assertFalse(result["observation_errors"])
        json.dumps(result, allow_nan=False)

    def test_missing_runtime_remains_unknown_even_with_loaded_backend_module(self):
        pipeline = SimpleNamespace(transformer=SimpleNamespace(named_modules=lambda: iter([])))
        with patch.dict(native.sys.modules, {
                "xfuser.core.distributed.runtime_state": None,
                "xfuser.core.attention.registry": None,
                "xfuser.core.attention.backends.sdpa.kernel": SimpleNamespace(cudnn=object())}):
            result = native.attention_observation(pipeline, None)
        self.assertIsNone(result["requested_attention_backend"])
        self.assertEqual(result["runtime_selector_after_request"], "unknown")
        self.assertEqual(result["selected_registry_entries"], {})
        self.assertFalse(result["kernel_dispatch_verified"])

    def test_empty_runtime_selector_is_unknown(self):
        pipeline = SimpleNamespace(transformer=SimpleNamespace(named_modules=lambda: iter([])))
        for value in (None, "", "   "):
            state = SimpleNamespace(attention_backend=value, cross_attention_backend=None)
            with patch.dict(native.sys.modules, {
                    "xfuser.core.distributed.runtime_state": SimpleNamespace(get_runtime_state=lambda: state),
                    "xfuser.core.attention.registry": None}):
                result = native.attention_observation(pipeline, "CUDNN")
            self.assertEqual(result["runtime_selector_after_request"], "unknown")
            self.assertEqual(result["cross_attention_selector_after_request"], "unknown")
            self.assertFalse(result["kernel_dispatch_verified"])

    def test_unavailable_runtime_inspection_preserves_unknown_and_reports_error(self):
        def unavailable():
            raise AssertionError("Runtime state has not been initialized")
        pipeline = SimpleNamespace(transformer=SimpleNamespace(named_modules=lambda: iter([])))
        with patch.dict(native.sys.modules, {"xfuser.core.distributed.runtime_state":
                                             SimpleNamespace(get_runtime_state=unavailable)}):
            result = native.attention_observation(pipeline, "CUDNN")
        self.assertEqual(result["runtime_selector_after_request"], "unknown")
        self.assertIn("AssertionError", result["observation_errors"][0])

    def test_uint8_finiteness_does_not_claim_prequantized_or_latent_finiteness(self):
        import numpy as np
        value = native.summarize_video(np.zeros((1, 2, 3, 4, 3), dtype=np.uint8),
                                       {"num_frames": 2, "height": 3, "width": 4})
        self.assertTrue(value["finite"])
        self.assertFalse(value["floating_point_frames"])
        self.assertFalse(value["pre_quantization_finite_checked"])
        self.assertFalse(value["pre_postprocess_finite_checked"])
        self.assertFalse(value["latent_finite_checked"])
        self.assertEqual(value["finite_scope"], "returned decoded frames after pipeline postprocessing")

    def test_output_digest_has_geometry_finite_dtype_and_complete_frame_checks(self):
        import numpy as np
        request = {"num_frames": 2, "height": 3, "width": 4}
        video = np.zeros((1, 2, 3, 4, 3), dtype=np.float32)
        good = native.summarize_video(video, request)
        self.assertEqual(good["shape"], [2, 3, 4, 3])
        self.assertTrue(good["finite"])
        self.assertTrue(good["floating_point_frames"])
        self.assertFalse(good["pre_postprocess_finite_checked"])
        self.assertFalse(good["quality_evaluated"])
        self.assertFalse(good["retained_arrays"])
        video[0, 0, 0, 0, 0] = 1
        self.assertNotEqual(good["sha256"], native.summarize_video(video, request)["sha256"])
        for value in [video[:, :1], video[:, :, :2], np.repeat(video, 2, axis=0)]:
            with self.assertRaises(ValueError):
                native.summarize_video(value, request)
        video[0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            native.summarize_video(video, request)


if __name__ == "__main__":
    unittest.main()
