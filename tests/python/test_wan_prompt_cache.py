"""Negative embedding cache contracts and the locked Wan/UMT5 CPU bridge.

The real bridge uses tiny random UMT5 weights and a local deterministic tokenizer
stand-in with the official Wan encode methods. It never downloads model files,
constructs a GPU worker or initializes CUDA. GPU output acceptance is separate.
"""

import contextlib
from concurrent.futures import Future, ThreadPoolExecutor
import importlib.util
import json
import threading
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ditserve.models.wan.prompt_cache import NegativePromptCache, cache_options, tensor_storage
from ditserve.models.wan.runtime import EncodedPrompt, WanStepWorker, normalize_spec, REQUEST_STATE_FIELDS


def spec(identity="a", prompt="a kettle", negative="", guidance=5.):
    return normalize_spec(dict(request_id=identity, prompt=prompt, negative_prompt=negative,
                               seed=101, height=32, width=32, frames=5, steps=2,
                               guidance_scale=guidance))


class FakeTensor:
    def __init__(self, value=1., *, storage_bytes=32):
        self.value = value
        self.shape, self.dtype, self.device = (1, 2, 4), "torch.float32", "cpu"
        self.storage_bytes = storage_bytes

    def numel(self):
        return 8

    def element_size(self):
        return 4

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: self.storage_bytes)

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.value, storage_bytes=self.storage_bytes)


class CacheContracts(unittest.TestCase):
    def setUp(self):
        self.cache = NegativePromptCache(entries=2, byte_limit=256,
                                         recipe=dict(model="fixed", max_sequence_length=512, dtype="float32"))
        self.calls = []
        self.storage_bytes = 32

    def encoder(self, **arguments):
        self.calls.append(arguments)
        negative = arguments.get("negative_prompt_embeds")
        if negative is None and arguments["do_classifier_free_guidance"]:
            negative = FakeTensor(len(arguments["negative_prompt"]) + 1., storage_bytes=self.storage_bytes)
        return FakeTensor(), negative

    def encode(self, negative="", *, cfg=True, cache=None, encoder=None):
        return (cache or self.cache).encode(encoder or self.encoder,
            dict(prompt="positive", negative_prompt=negative, do_classifier_free_guidance=cfg,
                 num_videos_per_prompt=1, max_sequence_length=512, device="cpu", dtype="torch.float32"),
            request_id="request-%d" % len(self.calls), validate=lambda *unused: None)

    def test_defaults_and_pair_validation_before_model_construction(self):
        self.assertEqual(cache_options({}), {})
        from ditserve.models.wan.backend import WanBackend
        invalid = [None, [], {"negative_prompt_cache_entries": 1}, {"negative_prompt_cache_bytes": 64}]
        for entries, size in [(True, 64), (0, 64), (65, 64), (1, False), (1, 0), (1, 268435457), (1., 64)]:
            invalid.append(dict(negative_prompt_cache_entries=entries, negative_prompt_cache_bytes=size))
        for options in invalid:
            with self.subTest(options=options), patch("ditserve.models.wan.runtime.WanStepWorker") as runtime:
                with self.assertRaises(ValueError):
                    WanBackend(dict(options=options))
                runtime.assert_not_called()

    def test_miss_uses_original_call_hit_supplies_an_independent_tensor(self):
        _, first, miss = self.encode("noise")
        self.assertNotIn("negative_prompt_embeds", self.calls[0])
        self.assertEqual((miss["outcome"], miss["admitted"], miss["negative_encode_called"]), ("miss", True, True))
        first.value = -99
        _, second, hit = self.encode("noise")
        self.assertEqual(second.value, 6.)
        self.assertEqual((hit["outcome"], hit["negative_encode_called"]), ("hit", False))
        second.value = -5
        self.assertEqual(self.encode("noise")[1].value, 6.)
        self.assertEqual(hit["after"]["bytes"], 32)
        json.dumps(hit, allow_nan=False)

    def test_lru_counts_full_backing_storage_and_releases_evicted_entries(self):
        self.storage_bytes = 96  # Logical bytes are only 32.
        a = self.encode("a")[2]
        self.encode("b")
        self.assertEqual(self.cache.bytes, 192)
        self.encode("a")  # Touch a; b must be evicted next.
        c = self.encode("c")[2]
        self.assertEqual(c["entry_tensor"]["logical_bytes"], 32)
        self.assertEqual(c["entry_bytes"], 96)
        self.assertEqual(c["after"]["bytes"], 192)
        self.assertEqual(c["after"]["lru"][0]["key"], a["key"])
        self.assertEqual(len(c["evicted"]), 1)
        self.assertEqual(self.encode("b")[2]["outcome"], "miss")

    def test_byte_limit_evicts_even_with_available_entry_slots(self):
        cache = NegativePromptCache(entries=8, byte_limit=96, recipe={})
        self.storage_bytes = 64
        self.encode("a", cache=cache)
        receipt = self.encode("b", cache=cache)[2]
        self.assertEqual(receipt["after"]["entries"], 1)
        self.assertEqual(receipt["counters"]["evictions"], 1)

    def test_oversized_entry_bypasses_without_evicting_existing_entries(self):
        before = self.encode("small")[2]["after"]
        self.storage_bytes = 512
        receipt = self.encode("large")[2]
        self.assertEqual(receipt["after"], before)
        self.assertEqual(receipt["bypass_reason"], "entry_exceeds_byte_limit")
        self.assertFalse(receipt["admitted"])
        self.assertEqual(self.encode("large")[2]["outcome"], "miss")

    def test_cfg_disabled_never_queries_or_supplies_negative_embeddings(self):
        self.encode("same")
        _, negative, receipt = self.encode("same", cfg=False)
        self.assertIsNone(negative)
        self.assertNotIn("negative_prompt_embeds", self.calls[-1])
        self.assertEqual(receipt["outcome"], "bypass")
        self.assertEqual(receipt["bypass_reason"], "cfg_disabled")
        self.assertEqual(receipt["before"], receipt["after"])
        self.assertEqual(receipt["counters"]["lookups"], 1)

    def test_exact_utf8_and_recipe_keys_do_not_merge_distinct_inputs(self):
        first = self.encode("noise")[2]
        self.assertEqual(self.encode(" noise ")[2]["outcome"], "miss")
        other = NegativePromptCache(entries=2, byte_limit=256, recipe=dict(model="another"))
        changed = self.encode("noise", cache=other)[2]
        self.assertNotEqual(first["key"], changed["key"])
        self.assertNotEqual(first["instance_id"], changed["instance_id"])

    def test_receipts_are_snapshots_with_no_mutable_cache_alias(self):
        receipt = self.encode("a")[2]
        receipt["entry_tensor"]["shape"][0] = 99
        receipt["limits"]["bytes"] = 0
        receipt["recipe"]["model"] = "changed"
        fresh = self.encode("a")[2]
        self.assertEqual(fresh["entry_tensor"]["shape"][0], 1)
        self.assertEqual(fresh["limits"]["bytes"], 256)
        self.assertEqual(fresh["recipe"]["model"], "fixed")
        self.assertEqual(receipt["counters"]["hits"], 0)

    def test_encoding_and_validation_failures_clear_cached_tensors(self):
        def fail(**unused):
            raise RuntimeError("encoder failed")
        self.encode("a")
        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            self.encode("a", encoder=fail)
        self.assertEqual(self.cache.snapshot(), dict(entries=0, bytes=0, lru=[]))
        self.assertEqual(self.cache.counters["errors"], 1)
        self.encode("b")
        with self.assertRaisesRegex(ValueError, "invalid embedding"):
            self.cache.encode(self.encoder, self.calls[-1], request_id="bad",
                              validate=Mock(side_effect=ValueError("invalid embedding")))
        self.assertEqual(self.cache.bytes, 0)

    def test_backend_shutdown_clears_cache_even_if_release_fails(self):
        from ditserve.models.wan.backend import WanBackend
        worker = WanStepWorker.__new__(WanStepWorker)
        worker._encoder_lock, worker._negative_prompt_cache = threading.Lock(), self.cache
        backend = WanBackend.__new__(WanBackend)
        backend.runtime, backend.encoder = worker, SimpleNamespace(shutdown=Mock())
        backend.tickets, backend.states = {"ticket": object()}, {"a": object()}
        backend._release_state = Mock(side_effect=RuntimeError("release failed"))
        self.encode("a")
        with self.assertRaisesRegex(RuntimeError, "release failed"):
            backend.close()
        self.assertEqual(self.cache.bytes, 0)
        self.assertEqual(backend.tickets, {})
        backend.encoder.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    def test_poll_receipt_is_separate_from_single_use_payload(self):
        from ditserve.models.wan.backend import WanBackend
        backend = WanBackend.__new__(WanBackend)
        future = Future()
        payload = EncodedPrompt(spec(), FakeTensor(), FakeTensor(), 1., object(),
                                negative_prompt_cache=self.encode()[2])
        future.set_result(payload)
        backend.tickets, backend.ticket_limit, backend.ticket_peak = {"t": (payload.spec, future)}, 1, 1
        backend.cancelled_tickets = set()
        result = backend.encode_poll(dict(ticket="t"))
        result["negative_prompt_cache"]["outcome"] = "tampered"
        self.assertEqual(payload.negative_prompt_cache["outcome"], "miss")
        self.assertFalse(payload.consumed)

    def test_cancelled_running_encoder_is_joined_before_cache_clear_on_close(self):
        from ditserve.models.wan.backend import WanBackend
        entered, release, shutdown_started, closed = (threading.Event() for unused in range(4))
        worker = WanStepWorker.__new__(WanStepWorker)
        worker._encoder_lock, worker._negative_prompt_cache = threading.Lock(), self.cache
        def encode(value):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test release timed out")
            with worker._encoder_lock:
                prompt, negative, receipt = self.encode()
                return EncodedPrompt(value, prompt, negative, 1., object(), negative_prompt_cache=receipt)
        backend = WanBackend.__new__(WanBackend)
        backend.runtime = SimpleNamespace(encode_prompt=encode, _sync=lambda: None,
                                          clear_negative_prompt_cache=worker.clear_negative_prompt_cache)
        backend.encoder = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(backend.encoder.shutdown, wait=True, cancel_futures=True)
        self.addCleanup(release.set)
        original_shutdown = backend.encoder.shutdown
        def shutdown(**options):
            shutdown_started.set()
            original_shutdown(**options)
        backend.encoder.shutdown = shutdown
        backend._spec = lambda request: request["spec"]
        backend.tickets, backend.ticket_counter, backend.ticket_peak, backend.ticket_limit = {}, 0, 0, 1
        backend.states, backend.parallel, backend.owns_world = {}, None, False
        ticket = backend.encode_start(dict(spec=spec()))["ticket"]
        self.assertTrue(entered.wait(5))
        cancelled = backend.encode_cancel(dict(ticket=ticket))
        self.assertEqual(cancelled["outstanding_tickets"], 1)
        errors = []
        def close():
            try:
                backend.close()
            except BaseException as error:
                errors.append(error)
            finally:
                closed.set()
        thread = threading.Thread(target=close)
        thread.start()
        try:
            self.assertTrue(shutdown_started.wait(5))
            self.assertFalse(closed.is_set())
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.cache.counters["insertions"], 1)
        self.assertEqual(self.cache.snapshot(), dict(entries=0, bytes=0, lru=[]))
        self.assertEqual(backend.tickets, {})

    def test_cache_free_runtime_fixture_does_not_need_encoder_lock_on_close(self):
        WanStepWorker.__new__(WanStepWorker).clear_negative_prompt_cache()


HAS_REAL_STACK = all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers", "transformers"))


@unittest.skipUnless(HAS_REAL_STACK, "locked torch/diffusers/transformers CPU stack is unavailable")
class TinyUMT5Bridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import diffusers
        from diffusers import WanPipeline, UniPCMultistepScheduler
        from transformers import UMT5Config, UMT5EncoderModel
        if diffusers.__version__ != "0.36.0":
            raise RuntimeError("bridge test requires the locked diffusers 0.36.0")
        cls.torch, cls.WanPipeline, cls.Scheduler = torch, WanPipeline, UniPCMultistepScheduler
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7103)
            cls.encoder = UMT5EncoderModel(UMT5Config(vocab_size=64, d_model=16, d_ff=32,
                d_kv=8, num_heads=2, num_layers=1, dropout_rate=0.)).eval()
        cls.cuda_was_initialized = torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.previous_threads)
        if cls.torch.cuda.is_initialized() != cls.cuda_was_initialized:
            raise AssertionError("CPU bridge initialized CUDA")

    def worker(self, enabled=True):
        torch = self.torch
        def tokenize(prompts, *, max_length, **unused):
            ids = [[3 + byte % 61 for byte in text.encode("utf-8")][:max_length - 1] + [1] for text in prompts]
            return SimpleNamespace(
                input_ids=torch.tensor([row + [0] * (max_length - len(row)) for row in ids]),
                attention_mask=torch.tensor([[1] * len(row) + [0] * (max_length - len(row)) for row in ids]))
        pipe = SimpleNamespace(text_encoder=self.encoder, tokenizer=tokenize,
                               _execution_device=torch.device("cpu"))
        pipe.encode_prompt = MethodType(self.WanPipeline.encode_prompt, pipe)
        pipe._get_t5_prompt_embeds = MethodType(self.WanPipeline._get_t5_prompt_embeds, pipe)
        pipe.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=2))
        pipe.prepare_latents = lambda batch, channels, height, width, frames, dtype, device, generator, unused: torch.randn(
            (batch, channels, 2, 4, 4), dtype=dtype, device=device, generator=generator)
        worker = WanStepWorker.__new__(WanStepWorker)
        worker.torch, worker.pipe, worker.device = torch, pipe, "cpu"
        worker._lock, worker._encoder_lock = threading.RLock(), threading.Lock()
        worker._owner, worker._states = object(), {}
        worker._sync = lambda: None
        worker.p = SimpleNamespace(_sdpa_math_context=lambda unused: contextlib.nullcontext())
        worker._scheduler_type = self.Scheduler
        worker._scheduler_config = dict(self.Scheduler().config)
        worker._negative_prompt_cache = (NegativePromptCache(entries=2, byte_limit=1024 * 1024,
            recipe=dict(model="tiny-umt5", diffusers="0.36.0", max_sequence_length=512, dtype="torch.float32"))
            if enabled else None)
        return worker

    def test_official_negative_bridge_is_exact_and_skips_one_real_encoder_call(self):
        reference, candidate = self.worker(False), self.worker()
        calls = []
        handle = self.encoder.register_forward_hook(lambda *unused: calls.append(1))
        self.addCleanup(handle.remove)
        for index, (positive, negative) in enumerate((("cat", ""), ("dog", ""), ("kettle", "blur"), ("鸟", "blur"))):
            value = spec(str(index), positive, negative)
            expected = reference.encode_prompt(value)
            before = len(calls)
            actual = candidate.encode_prompt(value)
            self.assertEqual(len(calls) - before, 2 if index % 2 == 0 else 1)
            self.assertTrue(self.torch.equal(expected.prompt_embeds, actual.prompt_embeds))
            self.assertTrue(self.torch.equal(expected.negative_prompt_embeds, actual.negative_prompt_embeds))
            self.assertEqual(actual.negative_prompt_cache["outcome"], "miss" if index % 2 == 0 else "hit")
            self.assertFalse(actual.consumed)
            self.assertFalse(actual.negative_prompt_embeds.is_cuda)
            json.dumps(actual.negative_prompt_cache, allow_nan=False)

    def test_real_tensors_remain_request_owned_after_mutation_and_eviction(self):
        worker = self.worker()
        first = worker.encode_prompt(spec("a"))
        expected = first.negative_prompt_embeds.clone()
        with self.torch.inference_mode():
            first.negative_prompt_embeds.fill_(99)
        second = worker.encode_prompt(spec("b"))
        self.assertTrue(self.torch.equal(expected, second.negative_prompt_embeds))
        with self.torch.inference_mode():
            second.negative_prompt_embeds.fill_(88)
        third = worker.encode_prompt(spec("c"))
        self.assertTrue(self.torch.equal(expected, third.negative_prompt_embeds))
        worker.encode_prompt(spec("d", negative="one"))
        worker.encode_prompt(spec("e", negative="two"))
        self.assertTrue(self.torch.equal(expected, third.negative_prompt_embeds))
        self.assertEqual(worker.encode_prompt(spec("f")).negative_prompt_cache["outcome"], "miss")

    def test_real_storage_view_counts_backing_allocation(self):
        backing = self.torch.zeros(128)
        view = backing[:8]
        info = tensor_storage(view)
        self.assertEqual(info["logical_bytes"], 32)
        self.assertEqual(info["accounted_bytes"], 512)

    def test_prepare_preserves_request_schema_and_single_use_and_copies_receipt(self):
        worker = self.worker()
        value = spec()
        payload = worker.encode_prompt(value)
        state = worker.prepare_encoded(value, payload)
        self.assertEqual(set(vars(state)), REQUEST_STATE_FIELDS)
        self.assertTrue(payload.consumed)
        self.assertEqual(state.events[-1]["negative_prompt_cache"], payload.negative_prompt_cache)
        payload.negative_prompt_cache["outcome"] = "tampered"
        self.assertEqual(state.events[-1]["negative_prompt_cache"]["outcome"], "miss")
        with self.assertRaisesRegex(ValueError, "already consumed"):
            worker.prepare_encoded(value, payload)
        worker.release(state)
        other = self.worker()
        with self.assertRaisesRegex(ValueError, "another worker"):
            other.prepare_encoded(value, worker.encode_prompt(value))

    def test_disabled_path_and_cfg_bypass_match_official_cpu_outputs(self):
        disabled, enabled = self.worker(False), self.worker()
        for guidance in (0., 1., 5.):
            value = spec(str(guidance), guidance=guidance)
            reference, actual = disabled.encode_prompt(value), enabled.encode_prompt(value)
            self.assertIsNone(reference.negative_prompt_cache)
            self.assertTrue(self.torch.equal(reference.prompt_embeds, actual.prompt_embeds))
            if guidance <= 1:
                self.assertIsNone(actual.negative_prompt_embeds)
                self.assertEqual(actual.negative_prompt_cache["outcome"], "bypass")
            else:
                self.assertTrue(self.torch.equal(reference.negative_prompt_embeds, actual.negative_prompt_embeds))


if __name__ == "__main__":
    unittest.main()
