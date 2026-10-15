"""CPU-only cache identity, integrity and no-fallback installation checks."""

import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from ditserve.research import cfg_embeddings as embeddings
from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.research.cfg_protocol import DEFAULT_CONFIG


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode_prompt(self, **kwargs):
        self.calls.append(kwargs)
        torch.rand(1)  # Verify the CPU encoder cannot perturb caller RNG.
        return torch.ones(embeddings.SHAPE), torch.zeros(embeddings.SHAPE)


class FakeWorker:
    def __init__(self):
        self._owner, self.torch = object(), torch
        self.payloads = []

    def encode_prompt(self, _):
        raise AssertionError("live encoder must never run after import")

    def prepare_encoded(self, spec, payload):
        if payload.owner is not self._owner or payload.spec != spec or payload.consumed:
            raise ValueError("bad installation")
        payload.consumed = True
        self.payloads.append(payload)
        return SimpleNamespace(events=[])


class CfgEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = read_json(DEFAULT_CONFIG)
        self.config["prompts"] = self.config["prompts"][:2]
        self.model = dict(sha256="a" * 64)
        self.encoder = FakeEncoder()
        self.real_loader = embeddings._load_encoder
        self.encoder_identity = dict(device="cpu", dtype="torch.float32", threads=8,
            max_sequence_length=512, num_videos_per_prompt=1, source_sha256="b" * 64)
        self.patches = [
            patch.object(embeddings.dual_protocol, "verify_model_manifest", return_value=self.model),
            patch.object(embeddings, "_encoder_identity", return_value=self.encoder_identity),
            patch.object(embeddings, "_load_encoder", return_value=self.encoder),
        ]
        self.mocks = [p.start() for p in self.patches]
        for p in self.patches:
            self.addCleanup(p.stop)
        self.spec = dict(request_id="first", prompt=self.config["prompts"][0]["text"],
            negative_prompt="", guidance_scale=5., seed=101, height=256, width=448, frames=17, steps=32)

    def prepare(self):
        return embeddings.prepare_cache(self.config, self.model, self.root)

    def bank(self, manifest):
        self.worker = FakeWorker()
        return embeddings.PreencodedPromptBank(self.worker, self.config, self.root,
            cache_manifest_sha256=manifest["sha256"], model_manifest_sha256=self.model["sha256"])

    def rewrite_manifest(self, manifest):
        manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "sha256"})
        atomic_json(self.root / "manifest.json", manifest)

    def test_preparation_is_cpu_fixed_shape_offline_rng_neutral_and_reusable(self):
        before = torch.get_rng_state().clone()
        threads = torch.get_num_threads()
        env = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "OMP_NUM_THREADS")}
        manifest = self.prepare()
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual({k: os.environ.get(k) for k in env}, env)
        self.assertEqual(manifest["preparation"]["gpu_count"], 0)
        self.assertEqual(len(self.encoder.calls), 2)
        self.assertTrue(all(call["device"] == "cpu" and call["dtype"] == torch.float32
                            and call["max_sequence_length"] == 512 for call in self.encoder.calls))
        self.assertEqual(len(list(self.root.glob("*.npy"))), 2)  # Content hashes deduplicate.
        self.assertEqual(self.prepare(), manifest)
        self.assertEqual(len(self.encoder.calls), 2)
        self.mocks[0].assert_called_with(self.model)

    def test_import_has_fresh_single_use_payloads_no_reencoding_and_provenance(self):
        manifest = self.prepare()
        bank = self.bank(manifest)
        bank.prepare(self.spec)
        state = bank.prepare(dict(self.spec, request_id="second", seed=102, cache=True, threshold=.04))
        self.assertTrue(all(payload.consumed and payload.cpu_prompt_encode_ms == 0
                            for payload in self.worker.payloads))
        self.assertNotEqual(self.worker.payloads[0].prompt_embeds.data_ptr(),
                            self.worker.payloads[1].prompt_embeds.data_ptr())
        event = state.events[-1]
        self.assertEqual(event["identity"]["cache_manifest_sha256"], manifest["sha256"])
        self.assertIn("before GPU allocation", event["timing_scope"])
        with self.assertRaisesRegex(ValueError, "live encoding is forbidden"):
            bank.warm(dict(self.spec, prompt="unknown"))
        with self.assertRaisesRegex(ValueError, "live encoding is forbidden"):
            bank.warm(dict(self.spec, negative_prompt="changed"))
        with self.assertRaisesRegex(ValueError, "live encoding is forbidden"):
            bank.warm(dict(self.spec, guidance_scale=4.))

    def test_import_detects_mutation_and_owner_change(self):
        bank = self.bank(self.prepare())
        next(iter(bank.entries.values()))["payload"].prompt_embeds[0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "content changed"):
            bank.prepare(self.spec)
        self.worker._owner = object()
        with self.assertRaisesRegex(ValueError, "another worker"):
            bank.warm(self.spec)

    def test_frozen_manifest_and_model_identity_must_match(self):
        manifest = self.prepare()
        with self.assertRaisesRegex(ValueError, "identity/hash"):
            embeddings.verify_cache(self.root, self.config, self.model, expected_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "identity/hash"):
            embeddings.verify_cache(self.root, self.config, dict(sha256="c" * 64))
        manifest["revision"] = "wrong"
        self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "identity/hash"):
            self.bank(manifest)

    def test_missing_changed_wrong_shape_nonfinite_and_unsafe_tensor_fail_closed(self):
        manifest = self.prepare()
        key = next(iter(manifest["entries"]))
        record = manifest["entries"][key]["prompt"]
        path = self.root / record["file"]
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.bank(manifest)
        path.write_bytes(original + b"tamper")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.bank(manifest)
        path.write_bytes(original)
        for array in (np.ones((1, 3, 4), dtype=np.float32),
                      np.ones(embeddings.SHAPE, dtype=np.float16),
                      np.full(embeddings.SHAPE, np.nan, dtype=np.float32)):
            np.save(path, array)
            record["file_sha256"] = file_digest(path)
            self.rewrite_manifest(manifest)
            with self.assertRaisesRegex(ValueError, "shape|finite"):
                self.bank(manifest)
        record["file"] = "../outside.npy"
        self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.bank(manifest)

    def test_partial_catalog_prompt_tamper_and_encoding_source_drift_rejected(self):
        manifest = self.prepare()
        original = copy.deepcopy(manifest)
        key = next(iter(manifest["entries"]))
        manifest["entries"].pop(key)
        self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.bank(manifest)
        manifest = copy.deepcopy(original)
        manifest["entries"][key]["identity"]["prompt"] = "modified"
        self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "prompt identity"):
            self.bank(manifest)
        self.rewrite_manifest(original)
        self.encoder_identity["source_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "identity/hash"):
            self.bank(original)

    def test_wrong_path_revision_threads_and_precision_rejected_before_encoder_load(self):
        for update in (dict(model_path="/tmp/model"), dict(revision="wrong"), dict(offline=False),
                       dict(runtime_threads=dict(omp=1, mkl=1, openblas=1)),
                       dict(precisions=dict(text_encoder="float16"))):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "fixed offline"):
                embeddings.prepare_cache(dict(self.config, **update), self.model, self.root)
        self.mocks[2].assert_not_called()

    def test_failed_encoder_does_not_publish_partial_manifest_or_change_rng(self):
        before = torch.get_rng_state().clone()
        def failing(**_):
            torch.rand(1)
            raise RuntimeError("encoder failed")
        self.encoder.encode_prompt = failing
        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            self.prepare()
        self.assertFalse((self.root / "manifest.json").exists())
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        (self.root / "stray.npy").write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "unfinished"):
            self.prepare()

    def test_cuda_initialized_process_cannot_prepare(self):
        with patch.object(torch.cuda, "is_initialized", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "must precede GPU"):
                self.prepare()
        self.mocks[2].assert_not_called()

    def test_loader_constructs_only_tokenizer_encoder_and_empty_wan_shell(self):
        import diffusers
        import transformers
        encoder = Mock()
        encoder.to.return_value = encoder
        encoder.parameters.return_value = [torch.zeros(1)]
        with patch.object(transformers.T5TokenizerFast, "from_pretrained") as tokenizer, \
                patch.object(transformers.UMT5EncoderModel, "from_pretrained", return_value=encoder) as load, \
                patch.object(diffusers, "WanPipeline") as pipeline:
            # Call the original function rather than the fixture's fake loader.
            self.real_loader(self.config, torch)
        self.assertTrue(tokenizer.call_args.kwargs["local_files_only"])
        self.assertTrue(load.call_args.kwargs["local_files_only"])
        self.assertEqual(load.call_args.kwargs["torch_dtype"], torch.float32)
        self.assertEqual(load.call_args.kwargs["subfolder"], "text_encoder")
        self.assertIsNone(pipeline.call_args.kwargs["vae"])
        self.assertIsNone(pipeline.call_args.kwargs["transformer"])
        self.assertIsNone(pipeline.call_args.kwargs["scheduler"])


if __name__ == "__main__":
    unittest.main()
