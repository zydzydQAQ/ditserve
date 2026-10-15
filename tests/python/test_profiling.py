"""Stdlib tests exercise contracts; fake events are never GPU measurements."""

import contextlib
import dataclasses
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ditserve.gpu import profiling as p


def arguments(*argv):
    parser = p.make_parser()
    return p.normalize_args(parser.parse_args(list(argv)), parser)


class FakeTensor:
    shape = (2, 3)
    dtype = "fake_float32"
    device = "fake_device"

    def numel(self):
        return 6

    def element_size(self):
        return 4


class FakeFiniteResult:
    """Tiny tensor-like finite reduction for CPU-only contract tests."""
    def __init__(self, finite=True):
        self.finite = finite

    def all(self):
        return self

    def item(self):
        return self.finite

    def __bool__(self):
        return self.finite


class FakeEmbedding(FakeTensor):
    def __init__(self, device="cpu", dtype="float32"):
        self.device, self.dtype = device, dtype
        self.transfers = []

    def to(self, device, dtype=None, **kwargs):
        self.transfers.append((device, dtype))
        return FakeEmbedding(device, dtype or self.dtype)


def fake_gpu_environment(architectures=None, capability=(7, 0), visible_devices=1):
    """Model CUDA metadata only; no kernels or hardware measurements run here."""
    properties = types.SimpleNamespace(name="Mock V100", total_memory=32 * 1024 ** 3,
                                       major=capability[0], minor=capability[1],
                                       uuid="GPU-mock-visible-device")
    cuda = types.SimpleNamespace(
        is_available=mock.Mock(return_value=True),
        device_count=mock.Mock(return_value=visible_devices),
        get_device_properties=mock.Mock(return_value=properties),
        get_device_capability=mock.Mock(return_value=capability),
        get_arch_list=mock.Mock(return_value=["sm_70", "sm_80"] if architectures is None else architectures),
        device=mock.Mock(side_effect=lambda device: contextlib.nullcontext()),
        mem_get_info=mock.Mock(return_value=(30 * 1024 ** 3, 32 * 1024 ** 3)),
        is_bf16_supported=mock.Mock(side_effect=lambda including_emulation=True: including_emulation or capability[0] >= 8))
    torch = types.SimpleNamespace(cuda=cuda, version=types.SimpleNamespace(cuda="12.6"), __version__="2.10.0+cu126")
    diffusers = types.SimpleNamespace(FluxPipeline=mock.Mock(), WanPipeline=mock.Mock(), AutoencoderKLWan=mock.Mock())
    cache_dit = types.SimpleNamespace(**{name: mock.Mock() for name in (
        "DBCacheConfig", "enable_cache", "refresh_context", "summary", "set_attn_backend")})
    modules = {"torch": torch, "diffusers": diffusers, "cache_dit": cache_dit}
    versions = {"torch": "2.10.0+cu126", "diffusers": "0.36.0", "transformers": "4.57.6",
                "cache-dit": "1.5.0", "accelerate": "1.12.0", "numpy": "1.26.4", "huggingface-hub": "0.36.0"}
    return modules, versions


def gpu_preflight(args, modules, versions):
    with mock.patch.object(p, "package_versions", return_value=versions):
        with mock.patch.object(p.importlib, "import_module", side_effect=lambda name: modules[name]):
            # This isolates metadata validation from the separate real CUDA smoke gate.
            with mock.patch.object(p, "_kernel_smoke_tests", create=True, return_value=[
                    {"operation": "matmul", "dtype": "float16", "finite": True, "status": "passed"},
                    {"operation": "sdpa_math", "dtype": "float16", "finite": True, "status": "passed"}]):
                return p.preflight(args)


class ProfilingContractTests(unittest.TestCase):
    def test_defaults_are_model_specific(self):
        flux = arguments("--preflight")
        wan = arguments("--preflight", "--model", "wan")
        self.assertEqual((flux.height, flux.steps, flux.frames), (1024, 28, 1))
        self.assertEqual((wan.width, wan.steps, wan.frames), (832, 50, 81))

    def test_v100_preset_uses_wan_and_component_specific_precision(self):
        args = arguments("--preflight", "--preset", "v100-wan")
        self.assertEqual(args.model, "wan")
        self.assertEqual(args.dtype, "float16")
        self.assertEqual(args.text_encoder_device, "cpu")
        self.assertEqual(args.text_encoder_dtype, "float32")
        self.assertEqual(args.vae_dtype, "float32")
        self.assertEqual(args.attention, "cache-dit-native")

    def test_v100_preset_preserves_explicit_fp32_and_workload(self):
        args = arguments("--preflight", "--preset", "v100-wan", "--dtype", "float32",
                         "--height", "256", "--width", "256", "--frames", "17", "--steps", "50")
        self.assertEqual(args.dtype, "float32")
        self.assertEqual((args.height, args.width, args.frames, args.steps), (256, 256, 17, 50))

    def test_v100_wan_preset_rejects_conflicting_flux_model(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                arguments("--preflight", "--preset", "v100-wan", "--model", "flux")

    def test_invalid_requests_fail_before_dependencies(self):
        invalid = [[], ["--preflight", "--device", "cpu"],
                   ["--preflight", "--model", "wan", "--frames", "80"],
                   ["--preflight", "--height", "513"], ["--preflight", "--repeats", "0"],
                   ["--preflight", "--cache-threshold", "nan"],
                   ["--preflight", "--cache", "cache-dit"],
                   ["--preflight", "--negative-prompt", "unsupported"]]
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    arguments(*argv)

    def test_help_does_not_import_gpu_dependencies(self):
        with mock.patch.object(p.importlib, "import_module", side_effect=AssertionError("unexpected import")):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as result:
                    p.main(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--preflight", out.getvalue())

    def test_stdlib_preflight_reports_missing_dependencies(self):
        args = arguments("--preflight")
        versions = dict.fromkeys(["torch", "diffusers", "transformers", "accelerate", "numpy"])
        with mock.patch.object(p, "package_versions", return_value=versions):
            with mock.patch.object(p.importlib, "import_module", side_effect=ImportError("torch missing")):
                report = p.preflight(args)
        self.assertFalse(report["ready"])
        self.assertFalse(report["measured"])
        self.assertFalse(report["weights_loaded"])
        self.assertIn("Missing optional GPU dependency: torch", report["errors"])
        json.dumps(report, allow_nan=False)

    def test_preflight_rejects_emulated_bf16_on_volta(self):
        modules, versions = fake_gpu_environment()
        self.assertTrue(modules["torch"].cuda.is_bf16_supported())
        report = gpu_preflight(arguments("--preflight", "--dtype", "bfloat16"), modules, versions)
        self.assertFalse(report["ready"])
        self.assertTrue(any("bfloat16" in error.lower() or "bf16" in error.lower() for error in report["errors"]))
        self.assertFalse(report["measured"])

    def test_preflight_rejects_wheel_without_volta_architecture(self):
        modules, versions = fake_gpu_environment(architectures=["sm_75", "sm_80", "sm_90"])
        report = gpu_preflight(arguments("--preflight", "--dtype", "float16"), modules, versions)
        self.assertFalse(report["ready"])
        self.assertTrue(any("sm_70" in error for error in report["errors"]))
        modules["diffusers"].WanPipeline.assert_not_called()

    def test_device_numbers_are_logical_after_cuda_visibility_mapping(self):
        modules, versions = fake_gpu_environment(visible_devices=1)
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}):
            report = gpu_preflight(arguments("--preflight", "--dtype", "float16", "--device", "cuda:0"),
                                   modules, versions)
        self.assertTrue(report["ready"], report["errors"])
        modules["torch"].cuda.get_device_properties.assert_called_once_with("cuda:0")
        self.assertEqual(report["device"], "cuda:0")

    def test_preflight_rejects_nonvisible_logical_device_before_properties(self):
        modules, versions = fake_gpu_environment(visible_devices=1)
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}):
            report = gpu_preflight(arguments("--preflight", "--dtype", "float16", "--device", "cuda:1"),
                                   modules, versions)
        self.assertFalse(report["ready"])
        modules["torch"].cuda.get_device_properties.assert_not_called()

    def test_preflight_does_not_accept_metadata_when_kernel_check_failed(self):
        modules, versions = fake_gpu_environment()
        failed = [{"operation": "sdpa_math", "dtype": "float16", "finite": None,
                   "status": "failed", "error": "mock kernel image unavailable"}]
        with mock.patch.object(p, "package_versions", return_value=versions):
            with mock.patch.object(p.importlib, "import_module", side_effect=lambda name: modules[name]):
                with mock.patch.object(p, "_kernel_smoke_tests", return_value=failed):
                    report = p.preflight(arguments("--preflight", "--dtype", "float16"))
        self.assertFalse(report["ready"])
        self.assertEqual(report["kernel_checks"], failed)
        self.assertTrue(any("kernel image unavailable" in error for error in report["errors"]))

    def test_sdpa_context_pins_only_math_backend(self):
        math_backend = object()
        attention = types.SimpleNamespace(SDPBackend=types.SimpleNamespace(MATH=math_backend),
                                          sdpa_kernel=mock.Mock(return_value=contextlib.nullcontext()))
        torch = types.SimpleNamespace(nn=types.SimpleNamespace(attention=attention))
        with p._sdpa_math_context(torch):
            pass
        attention.sdpa_kernel.assert_called_once_with([math_backend])

    def test_failure_writes_explicit_failure_record_without_measured_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile.json"
            with mock.patch.object(p, "run_profile", side_effect=p.ProfilingError("no CUDA")):
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    code = p.main(["--output", str(output)])
            self.assertEqual(code, 2)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["measured"])
            self.assertIn("No synthetic measurements", err.getvalue())

    def test_oom_failure_preserves_requested_workload_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "oom.json"
            error = p.ProfilingError("CUDA out of memory", category="out_of_memory")
            with mock.patch.object(p, "run_profile", side_effect=error) as run:
                with contextlib.redirect_stderr(io.StringIO()):
                    code = p.main(["--output", str(output), "--preset", "v100-wan", "--frames", "81",
                                   "--height", "480", "--width", "832", "--dtype", "float16"])
            self.assertEqual(code, 2)
            run.assert_called_once()
            requested = run.call_args.args[0]
            self.assertEqual((requested.height, requested.width, requested.frames, requested.dtype),
                             (480, 832, 81, "float16"))
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failure"]["category"], "out_of_memory")
            self.assertFalse(report["measured"])

    def test_failure_record_keeps_completed_timing_evidence(self):
        # Deliberately fabricated fixture: validates JSON preservation only.
        completed = [{"index": 0, "seed": 0, "uninstrumented": {"e2e_ms": 7.0}}]
        partial = {"measured": True, "runs": completed}
        error = p.ProfilingError("nonfinite diagnostic", category="nonfinite_latents", partial_report=partial)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "partial.json"
            with mock.patch.object(p, "run_profile", side_effect=error):
                with contextlib.redirect_stderr(io.StringIO()):
                    code = p.main(["--output", str(output)])
            report = json.loads(output.read_text())
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["measured"])
        self.assertEqual(report["runs"], completed)
        self.assertEqual(report["failure"]["category"], "nonfinite_latents")

    def test_request_flow_prepares_each_run_and_preserves_diagnostic_failure(self):
        """Exercise orchestration with fabricated timings, without torch or weights."""
        args = arguments("--preflight", "--preset", "v100-wan", "--repeats", "1", "--warmup", "1")
        active = {"math": False}

        @contextlib.contextmanager
        def math_context(torch):
            active["math"] = True
            try:
                yield
            finally:
                active["math"] = False

        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(set_device=mock.Mock()),
                                      inference_mode=contextlib.nullcontext)
        pipe = mock.Mock(device=args.device, _execution_device=args.device,
                         scheduler=types.SimpleNamespace(config={}))
        backend = types.SimpleNamespace(config=None, reset=mock.Mock(),
                                        stats=mock.Mock(return_value={"status": "disabled"}))
        prepare = mock.Mock(side_effect=lambda torch, pipe, args, kwargs:
                            (dict(kwargs, prepared=True), {"cpu_prompt_encode_ms": 4.0,
                                                           "prompt_embeddings_h2d_ms": 1.0}))

        def timed(torch, pipeline, kwargs, device, prepare_kwargs=None):
            self.assertTrue(active["math"])
            self.assertIsNotNone(prepare_kwargs)
            prepared, stages = prepare_kwargs(kwargs)
            self.assertTrue(prepared["prepared"])
            return object(), dict(stages, e2e_ms=7.0, peak_allocated_bytes=123)

        def record_output(output, args, stage, label):
            stage["output"] = {"sha256": "fabricated-fixture", "finite": True}

        observer = mock.MagicMock()
        observer.__enter__.return_value = observer
        observer.report.side_effect = p.ProfilingError(
            "nonfinite diagnostic", category="nonfinite_latents",
            details={"diagnostic": {"steps": [{"step_index": 0, "latents_finite": False}]}})
        patches = {
            "preflight": mock.Mock(return_value={"ready": True}),
            "_pipeline": mock.Mock(return_value=pipe),
            "_component_metadata": mock.Mock(return_value={}),
            "_revision_metadata": mock.Mock(return_value={}),
            "CacheBackend": mock.Mock(return_value=backend),
            "_sdpa_math_context": math_context,
            "_prepare_prompt_kwargs": prepare,
            "_call_kwargs": lambda torch, args, seed: {"seed": seed},
            "timed_call": timed,
            "timed_cache_reset": lambda torch, backend, device: {"wall_ms": 0.0},
            "_record_output": record_output,
            "_save_decoded_output": mock.Mock(),
            "StepObserver": mock.Mock(return_value=observer),
        }
        with mock.patch.multiple(p, **patches):
            with mock.patch.object(p.importlib, "import_module", side_effect=lambda name: {"torch": torch, "diffusers": object()}[name]):
                with self.assertRaises(p.ProfilingError) as result:
                    p.run_profile(args)
        report = p._failure_report(result.exception, args)
        self.assertEqual(prepare.call_count, 3)  # Warmup, measured request, diagnostic request.
        self.assertFalse(active["math"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_type"], "nonfinite")
        self.assertEqual(report["completed_runs"], 0)
        self.assertTrue(report["measured"])
        self.assertEqual(report["warmups"][0]["status"], "completed")
        run = report["runs"][0]
        for phase in ("uninstrumented", "diagnostic"):
            self.assertEqual(run[phase]["e2e_ms"], 7)
            self.assertEqual(run[phase]["cpu_prompt_encode_ms"], 4)
        self.assertFalse(run["diagnostic"]["steps"][0]["latents_finite"])

    def test_inventory_deduplicates_references_handles_cycles_and_marks_limits(self):
        tensor = FakeTensor()
        state = {"first": tensor, "same": tensor, "nested": [FakeTensor()], "opaque": object()}
        state["cycle"] = state
        report = p.tensor_inventory(state, lambda value: isinstance(value, FakeTensor))
        self.assertEqual(report["bytes"], 48)
        self.assertEqual(len(report["tensors"]), 2)
        self.assertFalse(report["is_complete_checkpoint"])
        self.assertEqual(len(report["opaque_fields_not_traversed"]), 1)

    def test_full_cache_state_is_not_fabricated(self):
        report = p.cache_state_support("cache-dit")
        self.assertEqual(report["status"], "unsupported")
        self.assertIsNone(report["bytes"])
        self.assertFalse(report["restorable_snapshot"])

    def test_timing_synchronizes_before_reading_clock_and_peak(self):
        events = []
        cuda = types.SimpleNamespace(
            synchronize=lambda device: events.append("sync"),
            reset_peak_memory_stats=lambda device: events.append("reset"),
            memory_allocated=lambda device: 100,
            max_memory_allocated=lambda device: 300,
            max_memory_reserved=lambda device: 400)
        clock_values = iter([2.0, 2.025])

        def clock():
            events.append("clock")
            return next(clock_values)

        def pipe(**kwargs):
            events.append("pipeline")
            return "test_output"

        output, timing = p.timed_call(types.SimpleNamespace(cuda=cuda), pipe, {}, "cuda:0", clock)
        self.assertEqual(output, "test_output")
        self.assertAlmostEqual(timing["e2e_ms"], 25)
        self.assertEqual(events, ["sync", "reset", "clock", "pipeline", "sync", "clock"])
        self.assertEqual(timing["peak_allocated_bytes"], 300)

    def test_timing_includes_cpu_prompt_preparation_and_reports_its_stages(self):
        order = []
        cuda = types.SimpleNamespace(
            synchronize=lambda device: order.append("sync"),
            reset_peak_memory_stats=lambda device: order.append("reset"),
            memory_allocated=lambda device: 100,
            max_memory_allocated=lambda device: 300,
            max_memory_reserved=lambda device: 400)
        times = iter([1.0, 1.06])

        def clock():
            order.append("clock")
            return next(times)

        def prepare(kwargs):
            order.append("cpu_encode")
            self.assertEqual(kwargs, {"prompt": "test"})
            return {"prompt_embeds": "encoded"}, {"cpu_prompt_encode_ms": 40.0,
                                                     "prompt_embeddings_h2d_ms": 5.0}

        def pipe(**kwargs):
            order.append("pipeline")
            self.assertEqual(kwargs, {"prompt_embeds": "encoded"})
            return "mock_output"

        output, timing = p.timed_call(types.SimpleNamespace(cuda=cuda), pipe, {"prompt": "test"},
                                     "cuda:0", clock=clock, prepare_kwargs=prepare)
        self.assertEqual(output, "mock_output")
        self.assertAlmostEqual(timing["e2e_ms"], 60)
        self.assertEqual(timing["cpu_prompt_encode_ms"], 40)
        self.assertEqual(timing["prompt_embeddings_h2d_ms"], 5)
        self.assertEqual(order, ["sync", "reset", "clock", "cpu_encode", "pipeline", "sync", "clock"])

    def test_cpu_prompt_failure_prevents_pipeline_execution(self):
        cuda = types.SimpleNamespace(synchronize=mock.Mock(), reset_peak_memory_stats=mock.Mock(),
                                     memory_allocated=mock.Mock(return_value=0))
        pipe = mock.Mock()
        prepare = mock.Mock(side_effect=RuntimeError("CPU encoder failed"))
        with self.assertRaisesRegex(RuntimeError, "CPU encoder failed"):
            p.timed_call(types.SimpleNamespace(cuda=cuda), pipe, {}, "cuda:0", prepare_kwargs=prepare)
        pipe.assert_not_called()

    def test_cpu_prompt_encoding_transfers_both_embeddings_without_prompt_conflicts(self):
        args = arguments("--preflight", "--preset", "v100-wan", "--batch-size", "2")
        positive, negative = FakeEmbedding(), FakeEmbedding()
        torch = types.SimpleNamespace(float32="float32", device=lambda value: value,
                                      cuda=types.SimpleNamespace(synchronize=mock.Mock()),
                                      isfinite=lambda value: FakeFiniteResult())
        pipe = types.SimpleNamespace(transformer=types.SimpleNamespace(dtype="float16"),
                                     encode_prompt=mock.Mock(return_value=(positive, negative)))
        kwargs = {"prompt": ["one", "two"], "negative_prompt": ["bad one", "bad two"],
                  "guidance_scale": 5.0, "num_frames": 17, "generator": object()}
        original = dict(kwargs)
        times = iter([10.0, 10.025, 10.029])
        prepared, timing = p._prepare_prompt_kwargs(torch, pipe, args, kwargs, clock=lambda: next(times))
        self.assertEqual(kwargs, original)
        self.assertNotIn("prompt", prepared)
        self.assertNotIn("negative_prompt", prepared)
        self.assertEqual(prepared["num_frames"], 17)
        self.assertIs(prepared["generator"], kwargs["generator"])
        for name in ("prompt_embeds", "negative_prompt_embeds"):
            self.assertEqual(prepared[name].device, args.device)
            self.assertEqual(prepared[name].dtype, "float16")
        self.assertEqual(positive.transfers, [(args.device, "float16")])
        self.assertEqual(negative.transfers, [(args.device, "float16")])
        call = pipe.encode_prompt.call_args.kwargs
        self.assertEqual(str(call["device"]), "cpu")
        self.assertEqual(call["dtype"], "float32")
        self.assertEqual(call["prompt"], original["prompt"])
        self.assertEqual(call["negative_prompt"], original["negative_prompt"])
        self.assertTrue(call["do_classifier_free_guidance"])
        self.assertEqual(call["max_sequence_length"], 512)
        self.assertAlmostEqual(timing["cpu_prompt_encode_ms"], 25)
        self.assertAlmostEqual(timing["prompt_embeddings_h2d_ms"], 4)

    def test_cpu_prompt_encoding_without_cfg_does_not_require_negative_embedding(self):
        args = arguments("--preflight", "--preset", "v100-wan", "--guidance-scale", "1")
        positive = FakeEmbedding()
        torch = types.SimpleNamespace(float32="float32", device=lambda value: value,
                                      cuda=types.SimpleNamespace(synchronize=mock.Mock()),
                                      isfinite=lambda value: FakeFiniteResult())
        pipe = types.SimpleNamespace(transformer=types.SimpleNamespace(dtype="float16"),
                                     encode_prompt=mock.Mock(return_value=(positive, None)))
        prepared, _ = p._prepare_prompt_kwargs(torch, pipe, args,
                                               {"prompt": ["one"], "negative_prompt": [""], "guidance_scale": 1.0})
        self.assertIsNone(prepared.get("negative_prompt_embeds"))
        self.assertFalse(pipe.encode_prompt.call_args.kwargs["do_classifier_free_guidance"])

    def test_cpu_prompt_encoding_rejects_nonfinite_embeddings_before_transfer(self):
        args = arguments("--preflight", "--preset", "v100-wan")
        positive, negative = FakeEmbedding(), FakeEmbedding()
        torch = types.SimpleNamespace(float32="float32", device=lambda value: value,
                                      cuda=types.SimpleNamespace(synchronize=mock.Mock()),
                                      isfinite=lambda value: FakeFiniteResult(False))
        pipe = types.SimpleNamespace(transformer=types.SimpleNamespace(dtype="float16"),
                                     encode_prompt=mock.Mock(return_value=(positive, negative)))
        with self.assertRaises(p.ProfilingError):
            p._prepare_prompt_kwargs(torch, pipe, args,
                                     {"prompt": ["one"], "negative_prompt": [""], "guidance_scale": 5.0})
        self.assertEqual(positive.transfers, [])
        self.assertEqual(negative.transfers, [])

    def test_pipeline_loads_fp32_encoder_directly_and_never_moves_whole_pipe_to_gpu(self):
        args = arguments("--preflight", "--preset", "v100-wan", "--revision", "pinned-revision")
        torch = types.SimpleNamespace(float16="float16", float32="float32")

        class Component:
            def __init__(self, dtype):
                self.dtype, self.device = dtype, "cpu"
                self.to = mock.Mock(side_effect=self.move)

            def move(self, device, dtype):
                self.device, self.dtype = device, dtype
                return self

        class EncoderFirstWanPipeline:
            """Model the problematic encoder-first device inference in Diffusers."""
            load_calls = []

            def __init__(self, text_encoder, vae, transformer):
                self.text_encoder, self.vae, self.transformer = text_encoder, vae, transformer
                self.to, self.set_progress_bar_config = mock.Mock(), mock.Mock()

            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                cls.load_calls.append(kwargs)
                return cls(kwargs["text_encoder"], kwargs["vae"], Component(kwargs["torch_dtype"]))

            @property
            def device(self):
                return self.text_encoder.device

            @property
            def _execution_device(self):
                return self.device

            def __call__(self):
                # The real Wan call uses this property for scheduler/latent placement.
                return {"latent_device": self._execution_device, "timestep_device": self._execution_device}

        encoder, vae = Component("float32"), Component("float32")
        diffusers = types.SimpleNamespace(
            AutoencoderKLWan=types.SimpleNamespace(from_pretrained=mock.Mock(return_value=vae)),
            WanPipeline=EncoderFirstWanPipeline)
        transformers = types.SimpleNamespace(
            UMT5EncoderModel=types.SimpleNamespace(from_pretrained=mock.Mock(return_value=encoder)))
        with mock.patch.object(p.importlib, "import_module", return_value=transformers):
            pipe = p._pipeline(torch, diffusers, args)
        encoder_load = transformers.UMT5EncoderModel.from_pretrained.call_args.kwargs
        vae_load = diffusers.AutoencoderKLWan.from_pretrained.call_args.kwargs
        pipeline_load = EncoderFirstWanPipeline.load_calls[0]
        self.assertEqual(encoder_load["torch_dtype"], "float32")
        self.assertEqual(vae_load["torch_dtype"], "float32")
        self.assertEqual(pipeline_load["torch_dtype"], "float16")
        for load in (encoder_load, vae_load, pipeline_load):
            self.assertEqual(load["revision"], "pinned-revision")
        self.assertIs(pipeline_load["text_encoder"], encoder)
        self.assertIs(pipeline_load["vae"], vae)
        encoder.to.assert_called_once_with(device="cpu", dtype="float32")
        pipe.transformer.to.assert_called_once_with(device=args.device, dtype="float16")
        vae.to.assert_called_once_with(device=args.device, dtype="float32")
        pipe.to.assert_not_called()
        self.assertEqual(EncoderFirstWanPipeline.device.fget(pipe), "cpu")
        self.assertEqual(pipe.device, args.device)
        self.assertEqual(pipe._execution_device, args.device)
        self.assertEqual(pipe(), {"latent_device": args.device, "timestep_device": args.device})

    def test_full_pipeline_backend_uses_automatic_cache_lifecycle(self):
        @dataclasses.dataclass
        class Stats:
            cached_steps: list
            accumulated_executed_steps: int

        module = mock.Mock()
        module.DBCacheConfig.side_effect = lambda **kwargs: kwargs
        module.summary.return_value = [Stats([2], 4)]
        pipe = types.SimpleNamespace(transformer=object())
        args = arguments("--preflight", "--cache", "cache-dit", "--attention", "cache-dit-native")
        with mock.patch.object(p.importlib, "import_module", return_value=module):
            backend = p.CacheBackend(pipe, args)
        backend.reset()
        module.enable_cache.assert_called_once_with(pipe, cache_config=backend.config, attention_backend="native")
        # cache-dit 1.5 wraps pipe.__call__ with reset/stats. Public refresh is
        # for persistent transformer-only contexts and is a no-op for this pipe.
        module.refresh_context.assert_not_called()
        result = backend.stats()
        module.summary.assert_called_once_with(pipe, details=True, logging=False)
        self.assertEqual(result["entries"][0]["cached_steps"], [2])
        module.summary.return_value = [Stats([], 0)]
        with self.assertRaises(p.ProfilingError):
            backend.stats()

    def test_baseline_can_match_cache_backend_attention_without_enabling_cache(self):
        module = mock.Mock()
        args = arguments("--preflight", "--attention", "cache-dit-native")
        with mock.patch.object(p.importlib, "import_module", return_value=module):
            backend = p.CacheBackend(object(), args)
        module.set_attn_backend.assert_called_once()
        module.enable_cache.assert_not_called()
        self.assertEqual(backend.stats()["status"], "disabled")

    def test_cache_installation_does_not_mutate_template_config(self):
        module = mock.Mock()
        module.DBCacheConfig.side_effect = lambda **kwargs: kwargs
        module.enable_cache.side_effect = lambda pipe, cache_config, **kwargs: cache_config.update(max_warmup_steps=999)
        module.refresh_context.side_effect = AssertionError("pipeline cache must reset inside its own call")
        args = arguments("--preflight", "--cache", "cache-dit", "--attention", "cache-dit-native")
        with mock.patch.object(p.importlib, "import_module", return_value=module):
            backend = p.CacheBackend(types.SimpleNamespace(transformer=object()), args)
        backend.reset()
        self.assertEqual(backend.config["max_warmup_steps"], 8)
        module.refresh_context.assert_not_called()

    def test_cache_mapping_preserves_partial_scopes_and_rejects_bad_indices(self):
        diagnostic = {"step_count": 2, "forward_count": 4,
                      "steps": [{"step_index": 0}, {"step_index": 1}]}
        counters = {"accumulated_executed_steps": 2, "accumulated_transformer_executed_steps": 4}
        stats = {"status": "measured", "entries": [
            dict(counters, cached_steps=[1], cfg_cached_steps=[]),
            dict(counters, cached_steps=[], cfg_cached_steps=[1]),
            dict(counters, cached_steps=[2], cfg_cached_steps=[])]}
        p.annotate_cache_steps(diagnostic, stats)
        evidence = diagnostic["steps"][1]["cache_evidence"]
        self.assertEqual(len(evidence), 2)
        self.assertTrue(evidence[0]["reported_cached"])
        self.assertFalse(evidence[0]["reported_cfg_cached"])
        self.assertTrue(evidence[1]["reported_cfg_cached"])
        mapping = diagnostic["cache_step_mapping"]
        self.assertEqual(mapping["whole_transformer_compute_or_reuse"], "not_classified")
        self.assertEqual(mapping["entries"][2]["status"], "unsupported_counter_or_index_contract")

    def test_automatic_cache_reset_is_not_reported_as_an_external_measurement(self):
        backend = types.SimpleNamespace(config={"enabled": True}, reset=mock.Mock())
        synchronize = mock.Mock(side_effect=AssertionError("no standalone CUDA operation is required"))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=synchronize))
        clock = mock.Mock(side_effect=AssertionError("automatic reset cannot be timed outside the pipeline"))
        result = p.timed_cache_reset(torch, backend, "cuda:0", clock=clock)
        self.assertEqual(result["status"], "auto_inside_pipeline")
        self.assertEqual(result["wall_ms"], 0.0)
        self.assertFalse(result["separate_operation_executed"])
        clock.assert_not_called()
        synchronize.assert_not_called()

    def test_fingerprint_handles_numpy_protocol_and_rejects_wrong_output_shapes(self):
        class Array:
            shape = (1, 2, 2, 3)
            dtype = "float32"

            def tobytes(self, order):
                self.order = order
                return b"test array payload"

        array = Array()
        output = types.SimpleNamespace(images=array)
        numpy = types.SimpleNamespace(isfinite=lambda array: FakeFiniteResult())
        with mock.patch.object(p.importlib, "import_module", return_value=numpy):
            report = p._fingerprint(output, "flux")
        self.assertEqual(len(report["sha256"]), 64)
        self.assertTrue(report["finite"])
        self.assertEqual(array.order, "C")
        with self.assertRaises(p.ProfilingError):
            p._fingerprint(types.SimpleNamespace(frames=array), "wan")
        with self.assertRaises(p.ProfilingError):
            p._fingerprint(types.SimpleNamespace(images=[array]), "flux")

    def test_fingerprint_rejects_nonfinite_decoded_output(self):
        array = types.SimpleNamespace(shape=(1, 17, 2, 2, 3), dtype="float32", tobytes=lambda order: b"mock NaN")
        numpy = types.SimpleNamespace(isfinite=lambda array: FakeFiniteResult(False))
        with mock.patch.object(p.importlib, "import_module", return_value=numpy):
            with self.assertRaises(p.ProfilingError) as result:
                p._fingerprint(types.SimpleNamespace(frames=array), "wan")
        self.assertEqual(result.exception.category, "nonfinite_output")
        self.assertFalse(result.exception.details["output"]["finite"])

    def test_two_cfg_forwards_are_associated_with_one_denoising_step(self):
        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self):
                pass

            def elapsed_time(self, other):
                return 5.0

        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event),
                                      is_tensor=lambda value: isinstance(value, FakeTensor),
                                      isfinite=lambda value: FakeFiniteResult())
        pipe = types.SimpleNamespace(scheduler=types.SimpleNamespace(model_outputs=[FakeTensor()], _step_index=1))
        observer = p.StepObserver(torch, pipe)
        for _ in range(2):
            observer._before(None, None)
            observer._after(None, None, None)
        kwargs = {"latents": FakeTensor()}
        self.assertIs(observer.callback(pipe, 0, 900, kwargs), kwargs)
        report = observer.report(1)
        self.assertEqual(report["forward_count"], 2)
        self.assertEqual(report["steps"][0]["transformer_cuda_ms"], 10)
        self.assertEqual(report["steps"][0]["sampler"]["bytes"], 24)
        with self.assertRaises(p.ProfilingError):
            observer.report(2)

    def test_latent_finite_check_defers_host_read_and_preserves_failed_step_details(self):
        finite = mock.Mock()
        finite.all.return_value = finite
        finite.item.return_value = False
        torch = types.SimpleNamespace(is_tensor=lambda value: isinstance(value, FakeTensor),
                                      isfinite=mock.Mock(return_value=finite))
        pipe = types.SimpleNamespace(scheduler=types.SimpleNamespace())
        observer = p.StepObserver(torch, pipe)
        observer.calls.append((types.SimpleNamespace(elapsed_time=lambda end: 3.0), object()))
        observer.callback(pipe, 0, 900, {"latents": FakeTensor()})
        finite.item.assert_not_called()
        with self.assertRaises(p.ProfilingError) as result:
            observer.report(1)
        finite.item.assert_called_once()
        self.assertEqual(result.exception.category, "nonfinite_latents")
        diagnostic = result.exception.details["diagnostic"]
        self.assertFalse(diagnostic["steps"][0]["latents_finite"])
        self.assertEqual(diagnostic["steps"][0]["transformer_cuda_ms"], 3)

    def test_hooks_are_removed_after_failure(self):
        handles = [mock.Mock(), mock.Mock()]
        transformer = mock.Mock()
        transformer.register_forward_pre_hook.return_value = handles[0]
        transformer.register_forward_hook.return_value = handles[1]
        pipe = types.SimpleNamespace(transformer=transformer)
        with self.assertRaises(ValueError):
            with p.StepObserver(None, pipe):
                raise ValueError("pipeline failed")
        for handle in handles:
            handle.remove.assert_called_once()

    def test_partial_hook_registration_is_rolled_back(self):
        handle = mock.Mock()
        transformer = mock.Mock()
        transformer.register_forward_pre_hook.return_value = handle
        transformer.register_forward_hook.side_effect = RuntimeError("hook registration failed")
        with self.assertRaisesRegex(RuntimeError, "hook registration failed"):
            with p.StepObserver(None, types.SimpleNamespace(transformer=transformer)):
                self.fail("observer must not enter after partial registration failure")
        handle.remove.assert_called_once()


if __name__ == "__main__":
    unittest.main()
