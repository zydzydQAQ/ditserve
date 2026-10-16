"""CPU contract checks for the whole-pipeline profiler; no model or GPU load."""

from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.models.wan.profile import WanProfileBackend, _spec, _versions


class FakeTensor:
    def __init__(self, torch):
        self.torch = torch

    def to(self, **kwargs):
        self.torch.transfers.append((list(self.torch.scopes), kwargs))
        return self


class FakeModule:
    def __init__(self, torch, transformer=False):
        self.torch, self.transformer = torch, transformer
        self.dtype = "float16"
        if transformer:
            self.blocks = [SimpleNamespace(attn1=FakeModule(torch), attn2=FakeModule(torch),
                                            ffn=FakeModule(torch))]

    def forward(self):
        if self.transformer:
            if self.torch.attention[-1] != ["flash"]:
                raise AssertionError("transformer did not enter Flash-only context")
            for block in self.blocks:
                block.attn1.forward()
                block.attn2.forward()
                block.ffn.forward()
        return 1


class FakeScheduler:
    config = {"solver_order": 2}
    instances = []

    def __init__(self):
        self.steps = 0
        self.instances.append(self)

    @classmethod
    def from_config(cls, config):
        return cls()

    def set_timesteps(self, count):
        self.count = count

    def step(self):
        self.steps += 1


class FakePipeline:
    def __init__(self, torch):
        self.torch = torch
        self.transformer = FakeModule(torch, transformer=True)
        self.scheduler = FakeScheduler()
        self.vae = SimpleNamespace(decode=self.decode)
        self.video_processor = SimpleNamespace(postprocess_video=self.postprocess)
        self.fail = False
        self.variant = 0
        self.encoding_calls = []

    def encode_prompt(self, **kwargs):
        self.encoding_calls.append(kwargs)
        return FakeTensor(self.torch), (FakeTensor(self.torch) if kwargs["do_classifier_free_guidance"] else None)

    def prepare_latents(self):
        return None

    @contextmanager
    def progress_bar(self, **kwargs):
        yield SimpleNamespace(update=lambda *args: None)

    def decode(self):
        if self.torch.attention[-1] != ["math"]:
            raise AssertionError("VAE did not return to MATH context")
        return None

    def postprocess(self):
        import numpy as np
        return np.full(self.output_shape, self.variant, dtype=np.float32)

    def __call__(self, **kwargs):
        self.output_shape = (1, kwargs["num_frames"], kwargs["height"], kwargs["width"], 3)
        self.scheduler.set_timesteps(kwargs["num_inference_steps"])
        self.prepare_latents()
        with self.progress_bar(total=kwargs["num_inference_steps"]) as bar:
            for step in range(kwargs["num_inference_steps"]):
                self.transformer.forward()
                if self.fail:
                    raise RuntimeError("injected model failure")
                if kwargs["guidance_scale"] > 1:
                    self.transformer.forward()
                self.scheduler.step()
                bar.update()
        self.vae.decode()
        return (self.video_processor.postprocess_video(),)


class FakeProfiler:
    def __init__(self, torch):
        self.torch = torch

    def __enter__(self):
        self.torch.trace_active = True
        return self

    def __exit__(self, *args):
        self.torch.trace_active = False

    def export_chrome_trace(self, path):
        Path(path).write_text(json.dumps({"traceEvents": self.torch.events}), encoding="utf-8")


class FakeTorch:
    float32 = "float32"

    def __init__(self):
        self.events, self.scopes, self.attention, self.transfers, self.seeds, self.syncs = [], [], [], [], [], []
        self.trace_active = False
        self.nvtx_events = []
        self.profiler_creations = []
        self.profiler = SimpleNamespace(record_function=self.record,
            profile=self.make_profiler, ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"))
        self.nn = SimpleNamespace(attention=SimpleNamespace(sdpa_kernel=self.sdpa,
            SDPBackend=SimpleNamespace(MATH="math", FLASH_ATTENTION="flash")))
        self.cuda = SimpleNamespace(synchronize=lambda device: self.syncs.append(list(self.scopes)),
            reset_peak_memory_stats=lambda device: None, memory_allocated=lambda device: 10,
            memory_reserved=lambda device: 20, max_memory_allocated=lambda device: 30,
            max_memory_reserved=lambda device: 40, empty_cache=lambda: None,
            nvtx=SimpleNamespace(range=self.nvtx_range))
        self.inference_mode = nullcontext

    def Generator(self, device):
        self.seeds.append({"device": device})
        seed_record = self.seeds[-1]
        def manual_seed(seed):
            seed_record["seed"] = seed
            return seed_record
        return SimpleNamespace(manual_seed=manual_seed)

    def make_profiler(self, **kwargs):
        self.profiler_creations.append(kwargs)
        return FakeProfiler(self)

    @contextmanager
    def record(self, name):
        self.scopes.append(name)
        self.events.append({"enter": name})
        try:
            yield
        finally:
            self.events.append({"exit": name})
            self.scopes.pop()

    @contextmanager
    def sdpa(self, backends):
        self.attention.append(backends)
        try:
            yield
        finally:
            self.attention.pop()

    @contextmanager
    def nvtx_range(self, name):
        self.scopes.append(name)
        self.nvtx_events.append({"enter": name})
        try:
            yield
        finally:
            self.nvtx_events.append({"exit": name})
            self.scopes.pop()


def fixture(work):
    backend = WanProfileBackend.__new__(WanProfileBackend)
    backend.config = {"work_dir": work}
    backend.torch = FakeTorch()
    backend.pipe = FakePipeline(backend.torch)
    backend.baselines, backend.versions = {}, {"fixture": "cpu"}
    backend.model_load_ms = 1.
    backend.device = "cuda:0"
    backend.scheduler_class = FakeScheduler
    backend.scheduler_config = dict(FakeScheduler.config)
    return backend


class WanProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = fixture(self.temp.name)
        self.request = {"spec": {"prompt": "one bicycle", "steps": 2, "frames": 17, "seed": 4,
                                 "height": 16, "width": 16}}

    def summary_module(self, *, fails=False, status="ok", missing_leaf=False):
        def summarize(path):
            self.assertTrue(Path(path).is_file())
            if fails:
                raise ValueError("invalid CUPTI trace")
            return dict(status=status, kernel_active_pct=20., attribution_status="verified",
                stages=[dict(name=name, annotation_count=4) for name in
                        ("transformer", "self_attention", "cross_attention", "ffn")
                        if not (missing_leaf and name == "ffn")])
        return patch.dict("sys.modules", {"ditserve.evaluation.cuda_profile": SimpleNamespace(summarize_trace=summarize)})

    def test_clean_has_no_profiler_or_annotations_and_fresh_rng_and_scheduler(self):
        initial = self.backend.pipe.scheduler
        first = self.backend.profile(dict(self.request, mode="clean"))
        second = self.backend.profile(dict(self.request, mode="clean"))
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["numerical_validation"]["status"], "matched")
        self.assertFalse(self.backend.torch.events)
        self.assertFalse(self.backend.torch.profiler_creations)
        self.assertEqual(self.backend.torch.seeds, [{"device": "cpu", "seed": 4}] * 2)
        self.assertIsNot(initial, self.backend.pipe.scheduler)
        self.assertEqual(self.backend.pipe.scheduler.steps, 0)
        self.assertEqual(self.backend.pipe.encoding_calls[0]["device"], "cpu")
        self.assertEqual(self.backend.pipe.encoding_calls[0]["dtype"], "float32")
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertNotIn("forward", vars(self.backend.pipe.transformer))

    def test_warmup_does_not_satisfy_clean_reference_requirement(self):
        self.backend.profile(dict(self.request, mode="warmup"))
        for mode in ("trace", "hardware"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "successful clean"):
                self.backend.profile(dict(self.request, mode=mode))
        self.assertFalse(self.backend.baselines)

    def test_hardware_ranges_span_full_request_without_kineto_or_artifacts(self):
        self.backend.profile(dict(self.request, mode="clean"))
        result = self.backend.profile(dict(self.request, mode="hardware"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["numerical_validation"]["status"], "matched")
        self.assertEqual(result["artifacts"], [])
        self.assertEqual(self.backend.torch.profiler_creations, [])
        self.assertEqual(self.backend.torch.events, [])
        self.assertEqual(self.backend.torch.nvtx_events, [
            {"enter": "wan_profile/request_17"}, {"enter": "wan_profile/denoise_17"},
            {"exit": "wan_profile/denoise_17"}, {"exit": "wan_profile/request_17"}])
        self.assertEqual(self.backend.torch.syncs[-1], ["wan_profile/request_17"])
        self.assertNotIn("progress_bar", vars(self.backend.pipe))
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_hardware_output_mismatch_fails_validation(self):
        self.backend.profile(dict(self.request, mode="clean"))
        self.backend.pipe.variant = 2
        result = self.backend.profile(dict(self.request, mode="hardware"))
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["numerical_validation"]["passed"])

    def test_required_uid_rejected_before_import_or_model_load(self):
        with patch("ditserve.models.wan.profile.os.geteuid", return_value=0), \
                patch("ditserve.models.wan.profile._versions") as versions:
            with self.assertRaisesRegex(PermissionError, "required_uid=1002"):
                WanProfileBackend({"required_uid": 1002})
            versions.assert_not_called()

    def test_trace_has_leaf_scopes_and_preserves_raw_evidence_with_dependency(self):
        self.backend.profile(dict(self.request, mode="clean"))
        with self.summary_module():
            result = self.backend.profile(dict(self.request, mode="trace"))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["numerical_validation"]["passed"])
        enters = [event["enter"] for event in self.backend.torch.events if "enter" in event]
        for name in ("request", "text_encode", "embedding_transfer", "latent_init", "scheduler",
                     "denoise", "transformer", "self_attention", "cross_attention", "ffn", "vae", "postprocess"):
            self.assertIn("wan_profile/" + name, enters)
        self.assertEqual(enters.count("wan_profile/transformer"), 4)
        self.assertEqual(self.backend.torch.syncs[-1], ["wan_profile/request"])
        self.assertEqual(self.backend.torch.profiler_creations[0]["activities"], ["cpu", "cuda"])
        self.assertFalse(self.backend.torch.profiler_creations[0]["record_shapes"])
        artifact = result["artifacts"][0]
        self.assertEqual(artifact["dependencies"], ["profile_validation"])
        self.assertTrue(artifact["summarized"])
        data = Path(artifact["path"]).read_bytes()
        self.assertEqual(artifact["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(artifact["bytes"], len(data))
        self.assertEqual(Path(artifact["path"]).parent, Path(self.temp.name))

    def test_summary_failure_retains_trace_and_is_not_reported_as_success(self):
        self.backend.profile(dict(self.request, mode="clean"))
        with self.summary_module(fails=True):
            result = self.backend.profile(dict(self.request, mode="trace"))
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["artifacts"][0]["summarized"])
        self.assertTrue(Path(result["artifacts"][0]["path"]).is_file())
        self.assertEqual(result["profiler"]["error_type"], "ValueError")

    def test_missing_leaf_annotation_invalidates_precise_attribution(self):
        self.backend.profile(dict(self.request, mode="clean"))
        with self.summary_module(missing_leaf=True):
            result = self.backend.profile(dict(self.request, mode="trace"))
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["instrumentation_validation"]["passed"])
        self.assertTrue(result["artifacts"][0]["summarized"])
        self.assertTrue(Path(result["artifacts"][0]["path"]).is_file())

    def test_model_failure_restores_methods_and_retains_partial_trace(self):
        self.backend.profile(dict(self.request, mode="clean"))
        pipe = self.backend.pipe
        original_decode = pipe.vae.decode
        original_postprocess = pipe.video_processor.postprocess_video
        pipe.fail = True
        with self.summary_module():
            result = self.backend.profile(dict(self.request, mode="trace"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["message"], "injected model failure")
        self.assertIs(pipe.vae.decode, original_decode)
        self.assertIs(pipe.video_processor.postprocess_video, original_postprocess)
        self.assertNotIn("forward", vars(pipe.transformer))
        self.assertNotIn("prepare_latents", vars(pipe))
        self.assertNotIn("progress_bar", vars(pipe))
        self.assertNotIn("forward", vars(pipe.transformer.blocks[0].attn1))
        self.assertEqual(pipe.scheduler.steps, 0)
        self.assertEqual(self.backend.torch.syncs[-1], ["wan_profile/request"])
        self.assertTrue(Path(result["artifacts"][0]["path"]).is_file())

    def test_hash_mismatch_keeps_original_reference_and_marks_trace_failed(self):
        first = self.backend.profile(dict(self.request, mode="clean"))
        self.backend.pipe.variant = 1
        with self.summary_module():
            result = self.backend.profile(dict(self.request, mode="trace"))
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["numerical_validation"]["passed"])
        self.assertEqual(result["numerical_validation"]["reference_sha256"], first["output"]["sha256"])

    def test_different_prompt_or_guidance_cannot_reuse_clean_reference(self):
        self.backend.profile(dict(self.request, mode="clean"))
        for changed in ({"prompt": "another bicycle"}, {"guidance_scale": 1.}, {"frames": 33}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "identical inputs"):
                self.backend.profile({"mode": "trace", "spec": dict(self.request["spec"], **changed)})

    def test_invalid_specs_fail_before_work(self):
        for changed in ({"frames": 16}, {"height": 33}, {"steps": True}, {"seed": -1},
                        {"guidance_scale": float("nan")}, {"guidance_scale": True}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                _spec({}, {"spec": changed})

    def test_version_mismatch_rejected(self):
        with patch("ditserve.models.wan.profile.importlib.metadata.version", return_value="wrong"):
            with self.assertRaisesRegex(RuntimeError, "pinned packages"):
                _versions()

    def test_close_releases_pipeline_and_is_idempotent(self):
        self.backend.profile(dict(self.request, mode="clean"))
        self.backend.close()
        self.backend.close()
        self.assertIsNone(self.backend.pipe)
        self.assertEqual(self.backend.baselines, {})
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.backend.profile(self.request)


if __name__ == "__main__":
    unittest.main()
