"""Unit tests for SD3.5 CPU T5-XXL split text encoding and pool/offload semantics (2026-11-05)."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "workers", ROOT / "benchmarks", ROOT / "tests" / "python"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cross_modal_memory_study as cmms  # noqa: E402
from ditserve.models.encoder_pool import ProactiveEncoderPool  # noqa: E402
from ditserve.models.pipeline import AtomicPipelineBackend, _sd35_pipeline_class  # noqa: E402
from ditserve.models.sd35 import SD35Backend  # noqa: E402
from ditserve.models.sd35_batch import SD35AsyncPipelineManager  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None


class _FakeClipOutput:
    def __init__(self, hidden, pooled):
        self.hidden_states = [hidden, hidden]
        self._pooled = pooled

    def __getitem__(self, idx):
        if idx == 0:
            return self._pooled
        raise IndexError(idx)


class SD35CpuT5ContractTests(unittest.TestCase):
    def _make_dummy_sd35(self, *, t5_device="cpu", pool=None, text_encoder_dtype="bfloat16"):
        backend = object.__new__(SD35Backend)
        backend.config = {
            "model_path": "/models/sd3.5-medium",
            "revision": "rev-sd35",
            "dtype": "bfloat16",
            "text_encoder_dtype": text_encoder_dtype,
            "text_encoder_device": t5_device,
            "options": {"offload_text_encoders_after_warmup": (t5_device == "cpu")},
        }
        backend.device = "cpu"
        backend.t5_device = t5_device
        backend.encoder_pool = pool
        backend.encoders_offloaded = (t5_device == "cpu")
        backend.encode_stream = None
        backend.vae_stream = None
        backend.async_pipeline = None
        backend.states = {}
        backend.completed = {}
        return backend

    def test_prompt_key_distinguishes_t5_device_dtype_and_revision(self):
        b_cpu = self._make_dummy_sd35(t5_device="cpu", text_encoder_dtype="bfloat16")
        b_cuda = self._make_dummy_sd35(t5_device="cuda", text_encoder_dtype="bfloat16")
        b_fp32 = self._make_dummy_sd35(t5_device="cpu", text_encoder_dtype="float32")
        b_rev2 = self._make_dummy_sd35(t5_device="cpu", text_encoder_dtype="bfloat16")
        b_rev2.config["revision"] = "rev-other"

        k_cpu = b_cpu._prompt_key("a lighthouse", "blurry", 256, True)
        k_cuda = b_cuda._prompt_key("a lighthouse", "blurry", 256, True)
        k_fp32 = b_fp32._prompt_key("a lighthouse", "blurry", 256, True)
        k_rev2 = b_rev2._prompt_key("a lighthouse", "blurry", 256, True)

        self.assertEqual(len({k_cpu, k_cuda, k_fp32, k_rev2}), 4)

    def test_restore_encoders_raises_when_t5_lives_on_cpu_and_offload_is_satisfied(self):
        backend = self._make_dummy_sd35(t5_device="cpu")
        self.assertTrue(backend.encoders_offloaded)
        off = backend.offload_encoders()
        self.assertTrue(off["offloaded"])
        self.assertTrue(off["encoders_offloaded"])
        self.assertEqual(off["t5_device"], "cpu")
        with self.assertRaisesRegex(RuntimeError, "cannot restore T5 encoder to GPU"):
            backend.restore_encoders()

    def test_ensure_staged_on_cpu_t5_never_calls_restore_encoders_and_supports_pool_disabled(self):
        had_torch = "torch" in sys.modules
        torch_mod, _ = cmms._ensure_torch_or_shim()
        try:
            with tempfile.TemporaryDirectory() as tmp_shm:
                # 1. Pool disabled (encoder_pool = None): direct encode without staging or slab files
                backend_no_pool = self._make_dummy_sd35(t5_device="cpu", pool=None)
                backend_no_pool.torch = torch_mod
                split_calls = []

                def fake_encode_split(*args, **kwargs):
                    split_calls.append((args, kwargs))
                    return (
                        torch_mod.empty((1, 16, 32), dtype=torch_mod.bfloat16, device="cpu"),
                        torch_mod.empty((1, 16, 32), dtype=torch_mod.bfloat16, device="cpu"),
                        torch_mod.empty((1, 16), dtype=torch_mod.bfloat16, device="cpu"),
                        torch_mod.empty((1, 16), dtype=torch_mod.bfloat16, device="cpu"),
                    )

                backend_no_pool._encode_split = fake_encode_split
                backend_no_pool.restore_encoders = mock.Mock(
                    side_effect=AssertionError("restore_encoders must never be called")
                )
                stage_info, fetched = backend_no_pool._fetch_staged("prompt one", "neg", 256, True)
                self.assertFalse(stage_info["enabled"])
                self.assertFalse(stage_info["staged"])
                self.assertFalse(fetched["hit"])
                self.assertEqual(len(split_calls), 1)
                self.assertEqual(list(Path(tmp_shm).glob("*.pt")), [])

                # 2. Pool enabled with CPU T5: stages to /dev/shm without calling restore_encoders
                pool = ProactiveEncoderPool({"shm_dir": tmp_shm, "write_shm": True, "pin_memory": False})
                backend_pooled = self._make_dummy_sd35(t5_device="cpu", pool=pool)
                backend_pooled.torch = torch_mod
                backend_pooled._encode_split = fake_encode_split
                backend_pooled.restore_encoders = mock.Mock(
                    side_effect=AssertionError("restore_encoders must never be called")
                )
                stage_info_2, fetched_2 = backend_pooled._fetch_staged("prompt two", "neg", 256, True)
                self.assertTrue(stage_info_2["staged"])
                self.assertFalse(stage_info_2["cached"])
                self.assertEqual(len(split_calls), 2)
                self.assertEqual(pool.stats()["slab_writes"], 1)
                backend_pooled.restore_encoders.assert_not_called()
        finally:
            if not had_torch and isinstance(sys.modules.get("torch"), cmms._CpuTorchModuleShim):
                sys.modules.pop("torch", None)

    def test_sd35_pipeline_class_keeps_execution_device_on_transformer_device(self):
        class BasePipe:
            @property
            def device(self):
                return "cpu"

            @property
            def _execution_device(self):
                return "cpu"

        wrapped_cls = _sd35_pipeline_class(BasePipe)
        instance = wrapped_cls()
        instance.transformer = SimpleNamespace(device="cuda:0")
        self.assertEqual(instance.device, "cuda:0")
        self.assertEqual(instance._execution_device, "cuda:0")


@unittest.skipUnless(HAS_TORCH, "PyTorch required for tensor equivalence tests")
class SD35SplitEncodeTorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def _build_synthetic_sd3_modules(self):
        torch = self.torch
        torch.manual_seed(20261105)
        clip_l_dim = 8
        clip_g_dim = 12
        t5_dim = 32  # >= clip_l_dim + clip_g_dim (20 -> padded to 32)
        clip_seq = 6
        t5_seq = 10

        class FakeTokenizer:
            def __init__(self, seq_len):
                self.model_max_length = seq_len

            def __call__(self, prompts, padding=True, max_length=None, truncation=True, return_tensors="pt", **_kwargs):
                batch = [prompts] if isinstance(prompts, str) else list(prompts)
                length = int(max_length or self.model_max_length)
                rows = []
                for text in batch:
                    seed_val = sum(ord(ch) for ch in text) % 97 + 1
                    row = [(seed_val + idx) % 127 for idx in range(length)]
                    rows.append(row)
                return SimpleNamespace(input_ids=torch.tensor(rows, dtype=torch.long))

        class FakeClipEncoder(torch.nn.Module):
            def __init__(self, vocab, dim):
                super().__init__()
                self.emb = torch.nn.Embedding(vocab, dim).to(dtype=torch.bfloat16)
                self.proj = torch.nn.Linear(dim, dim, bias=False).to(dtype=torch.bfloat16)

            def forward(self, input_ids, output_hidden_states=False):
                h0 = self.emb(input_ids)
                h1 = torch.tanh(self.proj(h0))
                pooled = h1[:, -1, :]
                out = SimpleNamespace(hidden_states=[h0, h1, h1])
                return (pooled, h1, out.hidden_states) if not output_hidden_states else _ClipTupleWithHidden(pooled, [h0, h1, h1])

        class _ClipTupleWithHidden(tuple):
            def __new__(cls, pooled, hidden_states):
                obj = super().__new__(cls, (pooled,))
                obj.hidden_states = hidden_states
                return obj

        class FakeT5Encoder(torch.nn.Module):
            def __init__(self, vocab, dim):
                super().__init__()
                self.emb = torch.nn.Embedding(vocab, dim).to(dtype=torch.bfloat16)
                self.linear = torch.nn.Linear(dim, dim, bias=False).to(dtype=torch.bfloat16)

            def forward(self, input_ids):
                x = self.emb(input_ids)
                return (self.linear(x),)

        tok1 = FakeTokenizer(clip_seq)
        tok2 = FakeTokenizer(clip_seq)
        tok3 = FakeTokenizer(t5_seq)
        enc1 = FakeClipEncoder(128, clip_l_dim).eval()
        enc2 = FakeClipEncoder(128, clip_g_dim).eval()
        enc3 = FakeT5Encoder(128, t5_dim).eval()

        def reference_encode_prompt(
            prompt,
            prompt_2=None,
            prompt_3=None,
            device="cpu",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
            negative_prompt_2=None,
            negative_prompt_3=None,
            max_sequence_length=t5_seq,
        ):
            prompt = [prompt] if isinstance(prompt, str) else list(prompt)
            # CLIP-L + CLIP-G
            ids1 = tok1(prompt, max_length=tok1.model_max_length).input_ids.to(device)
            out1 = enc1(ids1, output_hidden_states=True)
            emb1 = out1.hidden_states[-2]
            pool1 = out1[0]

            ids2 = tok2(prompt, max_length=tok2.model_max_length).input_ids.to(device)
            out2 = enc2(ids2, output_hidden_states=True)
            emb2 = out2.hidden_states[-2]
            pool2 = out2[0]

            clip_embeds = torch.cat([emb1, emb2], dim=-1)
            pooled = torch.cat([pool1, pool2], dim=-1)

            ids3 = tok3(prompt, max_length=max_sequence_length).input_ids.to(device)
            t5_embeds = enc3(ids3)[0]

            clip_padded = torch.nn.functional.pad(
                clip_embeds, (0, t5_embeds.shape[-1] - clip_embeds.shape[-1])
            )
            prompt_embeds = torch.cat([clip_padded, t5_embeds], dim=-2)

            if do_classifier_free_guidance:
                neg = [negative_prompt or ""] if isinstance(negative_prompt, str) else list(negative_prompt)
                n_ids1 = tok1(neg, max_length=tok1.model_max_length).input_ids.to(device)
                n_out1 = enc1(n_ids1, output_hidden_states=True)
                n_ids2 = tok2(neg, max_length=tok2.model_max_length).input_ids.to(device)
                n_out2 = enc2(n_ids2, output_hidden_states=True)
                n_clip = torch.cat([n_out1.hidden_states[-2], n_out2.hidden_states[-2]], dim=-1)
                n_pooled = torch.cat([n_out1[0], n_out2[0]], dim=-1)
                n_ids3 = tok3(neg, max_length=max_sequence_length).input_ids.to(device)
                n_t5 = enc3(n_ids3)[0]
                n_clip_padded = torch.nn.functional.pad(
                    n_clip, (0, n_t5.shape[-1] - n_clip.shape[-1])
                )
                neg_embeds = torch.cat([n_clip_padded, n_t5], dim=-2)
            else:
                neg_embeds = None
                n_pooled = None
            return prompt_embeds, neg_embeds, pooled, n_pooled

        module = SimpleNamespace(
            tokenizer=tok1,
            tokenizer_2=tok2,
            tokenizer_3=tok3,
            text_encoder=enc1,
            text_encoder_2=enc2,
            text_encoder_3=enc3,
            tokenizer_max_length=clip_seq,
            encode_prompt=reference_encode_prompt,
        )
        return module, t5_seq

    def test_encode_split_matches_reference_encode_prompt_and_records_device(self):
        torch = self.torch
        module, t5_seq = self._build_synthetic_sd3_modules()
        backend = object.__new__(SD35Backend)
        backend.config = {
            "model_path": "sd35",
            "revision": "rev1",
            "dtype": "bfloat16",
            "text_encoder_dtype": "bfloat16",
            "text_encoder_device": "cpu",
            "options": {},
        }
        backend.torch = torch
        backend.device = "cpu"
        backend.t5_device = "cpu"
        backend.encoder_pool = None
        backend.encoders_offloaded = True
        backend.pipe = SimpleNamespace(module=module)

        for do_cfg in (True, False):
            with self.subTest(do_cfg=do_cfg):
                split_out = backend._encode_split(
                    prompt="A sunlit alpine meadow",
                    negative_prompt="blurry, low quality",
                    max_sequence_length=t5_seq,
                    do_cfg=do_cfg,
                )
                ref_out = module.encode_prompt(
                    prompt="A sunlit alpine meadow",
                    negative_prompt="blurry, low quality",
                    max_sequence_length=t5_seq,
                    do_classifier_free_guidance=do_cfg,
                )
                self.assertTrue(torch.equal(split_out[0], ref_out[0]))
                self.assertTrue(torch.equal(split_out[2], ref_out[2]))
                if do_cfg:
                    self.assertTrue(torch.equal(split_out[1], ref_out[1]))
                    self.assertTrue(torch.equal(split_out[3], ref_out[3]))
                else:
                    self.assertIsNone(split_out[1])
                    self.assertIsNone(split_out[3])

    def test_async_pipeline_submit_encode_stages_on_thread_and_fetches_in_prepare(self):
        torch = self.torch
        module, t5_seq = self._build_synthetic_sd3_modules()
        with tempfile.TemporaryDirectory() as tmp_shm:
            pool = ProactiveEncoderPool({"shm_dir": tmp_shm, "write_shm": True, "pin_memory": False})
            backend = object.__new__(SD35Backend)
            backend.config = {
                "model_path": "sd35",
                "revision": "rev1",
                "dtype": "bfloat16",
                "text_encoder_dtype": "bfloat16",
                "text_encoder_device": "cpu",
                "options": {},
            }
            backend.torch = torch
            backend.device = "cpu"
            backend.t5_device = "cpu"
            backend.encoder_pool = pool
            backend.encoders_offloaded = True
            backend.encode_stream = None
            backend.vae_stream = None
            backend.pipe = SimpleNamespace(module=module)
            backend.states = {}
            backend.completed = {}

            mgr = SD35AsyncPipelineManager(backend)
            self.addCleanup(mgr.close)
            backend._async_pipeline_mgr = mgr

            spec = {
                "prompt": "A red lighthouse",
                "negative_prompt": "blurry",
                "max_sequence_length": t5_seq,
                "guidance_scale": 4.5,
                "steps": 4,
                "height": 512,
                "width": 512,
                "frames": 1,
                "batch_size": 1,
                "seed": 42,
            }
            started = backend.encode_start({
                "request_id": "req-1",
                "spec": spec,
            })
            ticket_id = started["ticket"]
            # Wait for background encode thread to finish staging
            _, future = mgr.encode_tickets[ticket_id]
            ticket_entry = future.result()
            self.assertIsNone(ticket_entry["tensors"])
            self.assertTrue(ticket_entry["stage_info"]["staged"])

            polled = backend.encode_poll({"ticket": ticket_id, "wait": True})
            self.assertTrue(polled["ready"])

            prep_out = backend.prepare({
                "state_id": "req-1",
                "encode_ticket": ticket_id,
                "spec": spec,
            })
            self.assertEqual(prep_out["state_id"], "req-1")
            self.assertIn("model_ms", prep_out)
            self.assertIn("encoder_pool_stage", prep_out)
            state = backend.states["req-1"]
            self.assertIn("prompt_embeds", state["embeds"])
            self.assertEqual(tuple(state["embeds"]["prompt_embeds"].shape), (1, 6 + t5_seq, 32))


if __name__ == "__main__":
    unittest.main()
