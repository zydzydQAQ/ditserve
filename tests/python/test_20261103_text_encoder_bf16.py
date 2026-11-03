"""Wan BF16 load/identity contracts; CPU checks do not qualify GPU performance."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ditserve.models.encoder_pool import ProactiveEncoderPool
from ditserve.models.wan.backend import WanBackend
from ditserve.models.wan.pipeline import WanPipelineBackend
from ditserve.models.wan.profile import WanProfileBackend
from ditserve.models.wan.runtime import REQUIRED_VERSIONS, WanStepWorker, normalize_spec


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = os.environ.get("DITSERVE_PACKAGE_ROOT")
BENCHMARKS = Path(PACKAGE_ROOT).parent / "benchmarks" if PACKAGE_ROOT else ROOT / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))


class StopLoading(Exception):
    pass


class PrecisionContracts(unittest.TestCase):
    def test_serving_reference_and_profiler_load_the_same_precision(self):
        # Stop at the shared loader: no model download or CUDA initialization.
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(set_device=Mock(),
            is_available=lambda: True, get_device_capability=lambda device: (8, 0)), set_num_threads=Mock())
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "model_index.json").write_text("{}")
            for override, expected in (({}, "bfloat16"), ({"text_encoder_dtype": "bfloat16"}, "bfloat16"),
                                       ({"text_encoder_dtype": "float32"}, "float32")):
                config = dict(model_path=directory, **override)
                for construct in (lambda: WanStepWorker(**config),
                                  lambda: WanPipelineBackend(config), lambda: WanProfileBackend(config)):
                    loader = Mock(side_effect=StopLoading)
                    with self.subTest(override=override, construct=construct), \
                            patch.dict("sys.modules", torch=fake_torch, diffusers=SimpleNamespace(),
                                       cache_dit=SimpleNamespace()), \
                            patch("importlib.metadata.version", side_effect=REQUIRED_VERSIONS.__getitem__), \
                            patch("ditserve.models.wan.profile._versions", return_value={}), \
                            patch("ditserve.models.pipeline._pipeline", loader), \
                            patch("ditserve.models.wan.profile._pipeline", loader):
                        with self.assertRaises(StopLoading):
                            construct()
                    args = loader.call_args.args[2]
                    self.assertEqual(args.text_encoder_dtype, expected)
                    self.assertEqual(args.text_encoder_device, "cpu")
                    self.assertEqual(args.dtype, "float16")
                    self.assertEqual(args.vae_dtype, "float32")

    def test_unsupported_precision_is_rejected_before_model_loading(self):
        for value in (None, "float16", "int8", "auto"):
            with self.subTest(value=value), patch("importlib.metadata.version") as version:
                with self.assertRaisesRegex(ValueError, "text_encoder_dtype"):
                    WanStepWorker(text_encoder_dtype=value)
                with self.assertRaisesRegex(ValueError, "text_encoder_dtype"):
                    WanProfileBackend(dict(text_encoder_dtype=value))
                version.assert_not_called()

    def test_formal_workers_explicitly_record_bf16_for_both_wan_models(self):
        from cluster_config import make_workers
        for model in ("wan21", "wan22"):
            _, workers, _, _ = make_workers(model, ["GPU-a", "GPU-b"], video_spec={}, image_spec={})
            for worker in workers.values():
                self.assertEqual(worker["config"]["text_encoder_dtype"], "bfloat16")

    def test_state_migration_cannot_relabel_fp32_encoded_inputs_as_bf16(self):
        from ditserve.models.wan.parallel import ParallelContractError, request_engine, validate_request_engine
        source, target = WanStepWorker.__new__(WanStepWorker), WanStepWorker.__new__(WanStepWorker)
        binding = request_engine(source)
        self.assertEqual(binding["text_encoder_dtype"], "bfloat16")
        validate_request_engine(target, binding)
        target.text_encoder_dtype = "float32"
        with self.assertRaisesRegex(ParallelContractError, "text_encoder_dtype"):
            validate_request_engine(target, binding)

    def test_generation_records_actual_precision_and_rejects_cross_precision_comparison(self):
        from native_exact import validate_pair_identity
        from test_native_exact import record
        backend = WanBackend.__new__(WanBackend)
        backend.config = {}
        spec = normalize_spec(dict(request_id="one", prompt="a kettle", height=32, width=32,
                                   frames=5, steps=2, seed=7))
        for dtype in ("bfloat16", "float32"):
            backend.runtime = SimpleNamespace(model_path=Path("snapshot"), model_revision="pinned",
                vae_dtype="float32", text_encoder_dtype=dtype, attention="math")
            generation = backend._evidence({}, spec, None, [])["generation"]
            self.assertEqual(generation["text_encoder_dtype"], dtype)
            self.assertEqual(generation["text_encoder_device"], "cpu")
        reference, candidate = record()["requests"]
        candidate["outputs"][-1]["generation"]["text_encoder_dtype"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "execution identity differs"):
            validate_pair_identity(reference, candidate)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch required")
class EncoderPoolPrecision(unittest.TestCase):
    def test_cache_separates_precision_and_revision_and_owns_returned_tensors(self):
        import torch
        pool = ProactiveEncoderPool()
        self.addCleanup(pool.clear)
        runtime = SimpleNamespace(model_path=Path("snapshot"), model_revision="rev1",
            text_encoder_dtype="bfloat16", pipe=SimpleNamespace(text_encoder=SimpleNamespace(dtype=torch.bfloat16)),
            _owner=object())
        def encode(worker, text, length):
            return torch.ones(1, 4, 2, dtype=getattr(torch, worker.text_encoder_dtype)), 4
        pool._encode_single_wan_text_trimmed = Mock(side_effect=encode)
        spec = dict(request_id="a", prompt="a kettle", height=32, width=32, frames=5, steps=2, seed=7)
        first = pool.encode_wan_prompt_fast(runtime, spec, pin_memory=False)
        second = pool.encode_wan_prompt_fast(runtime, spec, pin_memory=False)
        self.assertTrue(second.negative_prompt_cache["pos_cache_hit"])
        self.assertEqual(second.prompt_embeds.dtype, torch.bfloat16)
        first.prompt_embeds.fill_(99)
        self.assertTrue(torch.equal(second.prompt_embeds, torch.ones_like(second.prompt_embeds)))
        third = pool.encode_wan_prompt_fast(runtime, spec, pin_memory=False)
        self.assertTrue(torch.equal(third.prompt_embeds, second.prompt_embeds))
        for dtype, revision in (("float32", "rev1"), ("bfloat16", "rev2")):
            runtime.text_encoder_dtype, runtime.model_revision = dtype, revision
            runtime.pipe.text_encoder.dtype = getattr(torch, dtype)
            payload = pool.encode_wan_prompt_fast(runtime, spec, pin_memory=False)
            self.assertFalse(payload.negative_prompt_cache["pos_cache_hit"])
            self.assertFalse(payload.negative_prompt_cache["neg_cache_hit"])
            self.assertEqual(payload.prompt_embeds.dtype, getattr(torch, dtype))

    def test_pool_cannot_recast_a_resident_worker_behind_its_cache_recipe(self):
        import torch
        encoder = SimpleNamespace(dtype=torch.bfloat16, to=Mock())
        runtime = SimpleNamespace(pipe=SimpleNamespace(text_encoder=encoder), text_encoder_dtype="bfloat16")
        pool = ProactiveEncoderPool()
        for mode in ("fp32_trimmed", "int8_dynamic_trimmed"):
            with self.assertRaisesRegex(ValueError, "loaded text_encoder_dtype"):
                pool.optimize_wan_cpu_encoder(runtime, mode=mode)
        encoder.to.assert_not_called()

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("transformers", "diffusers")),
                         "local UMT5/diffusers CPU stack required")
    def test_real_umt5_trimmed_encoding_and_padding_stay_bf16(self):
        import torch
        from transformers import UMT5Config, UMT5EncoderModel
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(703)
            encoder = UMT5EncoderModel(UMT5Config(vocab_size=16, d_model=8, d_ff=16,
                d_kv=4, num_heads=2, num_layers=1, dropout_rate=0.)).eval()
        fp32_bytes = sum(p.numel() * p.element_size() for p in encoder.parameters())
        encoder.to(torch.bfloat16)
        self.assertEqual(sum(p.numel() * p.element_size() for p in encoder.parameters()), fp32_bytes // 2)
        inputs = SimpleNamespace(input_ids=torch.tensor([[3, 4, 1, 0]]),
                                 attention_mask=torch.tensor([[1, 1, 1, 0]]))
        runtime = SimpleNamespace(text_encoder_dtype="bfloat16",
            pipe=SimpleNamespace(text_encoder=encoder, tokenizer=Mock(return_value=inputs)))
        pool = ProactiveEncoderPool()
        encoded, length = pool._encode_single_wan_text_trimmed(runtime, "kettle", 4)
        self.assertEqual(length, 3)
        self.assertEqual(encoded.dtype, torch.bfloat16)
        self.assertEqual(tuple(encoded.shape), (1, 4, 8))
        self.assertTrue(bool(torch.isfinite(encoded).all()))
        self.assertEqual(encoded[0, -1].count_nonzero().item(), 0)
        worker = WanStepWorker.__new__(WanStepWorker)
        worker.torch = torch
        worker._validate_cpu_embeddings(encoded, encoded, {"guidance_scale": 5.})
        with self.assertRaisesRegex(ValueError, "CPU bfloat16"):
            worker._validate_cpu_embeddings(encoded.float(), encoded, {"guidance_scale": 5.})


if __name__ == "__main__":
    unittest.main()
