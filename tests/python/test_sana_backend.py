"""Sana request lifecycle against a stand-in adapter; no torch or weights."""

import gc
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import patch

from ditserve.models.sana.runtime import SanaPrompt
from ditserve.runtime.worker import Worker


class FakeTensor:
    def __init__(self, shape=(2, 4), element_size=4, storage_bytes=32):
        self.shape, self._element_size, self._storage_bytes = shape, element_size, storage_bytes

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def element_size(self):
        return self._element_size

    def untyped_storage(self):
        return self

    def nbytes(self):
        return self._storage_bytes


class FakeAdapter:
    def __init__(self, path, device):
        self.calls, self.closed = [], False
        self.encode_calls, self.preencoded, self.packet_refs = [], [], []
        self.fused_calls, self.generation_observer = [], None
        self.owner = object()
        self.tensor_factory = lambda prompt: (FakeTensor(storage_bytes=64), FakeTensor((1,), 8, 8))
        self.encode_error = self.generate_error = self.close_error = False

    def encode(self, prompt):
        self.encode_calls.append(prompt)
        if self.encode_error:
            raise RuntimeError("encoding failed")
        embeddings, mask = self.tensor_factory(prompt)
        packet = SanaPrompt(prompt, embeddings, mask, 2.5, self.owner)
        self.packet_refs.append(weakref.ref(packet))
        return packet

    def generate(self, prompt, *, seed=0, encoded=None):
        if encoded is not None and (encoded.owner is not self.owner or encoded.prompt != prompt):
            raise ValueError("encoded Sana prompt belongs to another request/adapter")
        self.calls.append((prompt, seed))
        self.preencoded.append(encoded is not None)
        if self.generation_observer is not None:
            self.generation_observer()
        if self.generate_error:
            raise RuntimeError("generation failed")
        return [[[float(seed)]]], dict(ms=7.5, finite=True, encode_ms=None if encoded is None else encoded.encode_ms)

    def encode_and_generate(self, prompt, *, seed=0, on_encoded=None):
        self.fused_calls.append((prompt, seed))
        packet = self.encode(prompt)
        packet.encode_ms = .5
        if on_encoded is not None:
            on_encoded(packet)
        output, _ = self.generate(prompt, seed=seed, encoded=packet)
        return packet, output, dict(ms=8., encode_ms=.5, generation_ms=7.5, finite=True,
            timing_scope="fused_miss_host_wall", encode_timing_scope="host encoding submission",
            generation_timing_scope="remaining wall including encoding tail")

    def close(self):
        self.closed = True
        if self.close_error:
            raise RuntimeError("close failed")


class SanaBackendTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        patcher = patch("ditserve.models.sana.runtime.SanaAdapter", FakeAdapter)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.worker = Worker("sana", dict(model_path="/models/sana", work_dir=self.directory.name))
        self.workers = [self.worker]
        self.addCleanup(self.close_workers)
        self.sequence = 0

    def close_workers(self):
        for worker in self.workers:
            if not worker.closed:
                worker.close()

    def enable_cache(self, entries=4, byte_limit=16777216, sync_mode=None):
        options = dict(prompt_cache_entries=entries, prompt_cache_bytes=byte_limit)
        if sync_mode is not None:
            options["prompt_cache_sync_mode"] = sync_mode
        self.worker = Worker("sana", dict(model_path="/models/sana", work_dir=self.directory.name,
                                         options=options))
        self.workers.append(self.worker)

    def complete(self, key, prompt, seed=3, release=True):
        self.call("prepare", state_id=key, spec=dict(prompt=prompt, seed=seed))
        advanced = self.call("advance", state_id=key)
        finished = self.call("finish", state_id=key, output_mode="check")
        self.assertEqual(advanced["prompt_cache"], finished["prompt_cache"])
        self.assertEqual(advanced["model_ms"], finished["prompt_cache"]["service_ms"])
        json.dumps(finished, allow_nan=False)
        if release:
            self.call("release", state_id=key)
        return finished["prompt_cache"]

    def call(self, op, **fields):
        self.sequence += 1
        return self.worker.dispatch(dict(version=1, id=self.sequence, op=op, **fields))

    def test_one_pipeline_call_per_request(self):
        self.call("prepare", state_id="a", spec=dict(prompt="cat", seed=3))
        with self.assertRaisesRegex(ValueError, "completed model work"):
            self.call("finish", state_id="a")
        result = self.call("advance", state_id="a")
        self.assertTrue(result["progress"]["done"])
        self.assertEqual(result["model_ms"], 7.5)
        receipt = result["prompt_cache"]
        self.assertEqual(receipt["outcome"], "disabled")
        self.assertFalse(receipt["encode_called"])
        self.assertEqual(receipt["before"], dict(entries=0, bytes=0, lru=[]))
        self.assertEqual(receipt["before"], receipt["after"])
        self.assertEqual(receipt["entry_tensors"], [])
        self.assertEqual(receipt["entry_bytes"], 0)
        self.assertEqual(receipt["counters"]["lookups"], 0)
        self.assertEqual(receipt["cold_service"], dict(count=0, total_ms=0.))
        self.assertEqual(receipt["warm_service"], dict(count=0, total_ms=0.))
        with self.assertRaisesRegex(ValueError, "no remaining"):
            self.call("advance", state_id="a")
        output = self.call("finish", state_id="a", output_mode="check")["output"]
        self.assertEqual((output["shape"], output["output_mode"]), ([1, 1, 1], "check"))
        self.assertEqual(len(output["sha256"]), 64)
        self.assertEqual(output["sha256"], output["array_sha256"])
        self.call("release", state_id="a")
        self.assertEqual(self.worker.backend.runtime.calls, [("cat", 3)])
        self.assertEqual(self.worker.backend.runtime.encode_calls, [])
        self.assertEqual(self.worker.backend.runtime.preencoded, [False])
        self.call("shutdown")
        self.assertTrue(self.worker.backend.runtime.closed)

    def test_rejects_options_and_malformed_specs(self):
        with self.assertRaisesRegex(ValueError, "takes no options"):
            Worker("sana", dict(model_path="/m", options=dict(candidate="eager"))).dispatch(
                dict(version=1, id=1, op="prepare", state_id="a", spec=dict(prompt="x")))
        with self.assertRaisesRegex(ValueError, "string prompt"):
            self.call("prepare", state_id="b", spec=dict(prompt=3))

    def test_warmup_is_first_cold_receipt_and_release_preserves_cache(self):
        self.enable_cache()
        cold = self.complete("warmup-2", "cat")
        warm = self.complete("image-1", "cat", seed=4)
        self.assertEqual((cold["sequence"], warm["sequence"]), (1, 2))
        self.assertEqual((cold["request_id"], cold["outcome"], warm["outcome"]), ("warmup-2", "miss", "hit"))
        self.assertEqual((cold["admitted"], warm["admitted"]), (True, False))
        self.assertEqual((cold["encode_called"], warm["encode_called"]), (True, False))
        self.assertEqual((cold["encode_ms"], warm["encode_ms"]), (2.5, 0.))
        self.assertEqual((cold["service_ms"], warm["service_ms"]), (10., 7.5))
        self.assertEqual((cold["entry_bytes"], warm["entry_bytes"]), (72, 72))
        self.assertEqual(warm["before"], cold["after"])
        self.assertEqual(warm["after"], cold["after"])
        self.assertEqual(warm["cold_service"], dict(count=1, total_ms=10.))
        self.assertEqual(warm["warm_service"], dict(count=1, total_ms=7.5))
        self.assertEqual(warm["counters"], dict(lookups=2, hits=1, misses=1, insertions=1,
                                              evictions=0, bypasses=0, errors=0, clears=0))
        self.assertTrue(warm["cumulative_includes_warmup"])
        self.assertEqual(cold["adapter_id"], warm["adapter_id"])
        self.assertEqual(self.worker.backend.runtime.encode_calls, ["cat"])
        self.assertEqual(self.worker.backend.runtime.calls, [("cat", 3), ("cat", 4)])
        self.assertEqual(self.worker.backend.runtime.preencoded, [True, True])
        self.assertEqual(self.worker.backend.states, {})

    def test_exact_prompt_key_and_adapter_instance_isolation(self):
        self.enable_cache()
        receipts = [self.complete(str(index), prompt) for index, prompt in enumerate(("Cat", "cat", "cat ", "cat"))]
        self.assertEqual([receipt["outcome"] for receipt in receipts], ["miss", "miss", "miss", "hit"])
        self.assertEqual(receipts[1]["prompt_sha256"], hashlib.sha256(b"cat").hexdigest())
        first_worker = self.worker
        self.enable_cache()
        separate = self.complete("other-worker", "cat")
        self.assertEqual(separate["outcome"], "miss")
        self.assertNotEqual(separate["adapter_id"], receipts[0]["adapter_id"])
        with self.assertRaisesRegex(ValueError, "another request/adapter"):
            self.worker.backend.runtime.generate("cat", encoded=first_worker.backend._cache["cat"]["encoded"])

    def test_lru_entry_limit_and_receipt_immutability(self):
        self.enable_cache(entries=2)
        first = self.complete("a0", "a", release=False)
        self.complete("b0", "b")
        self.complete("a1", "a")
        last = self.complete("c0", "c")
        self.assertEqual(last["evicted"], [dict(key=hashlib.sha256(b"b").hexdigest(), bytes=72)])
        self.assertEqual(list(self.worker.backend._cache), ["a", "c"])
        self.assertEqual(last["after"]["bytes"], 144)
        self.assertEqual(self.call("finish", state_id="a0", output_mode="check")["prompt_cache"], first)
        first["after"]["lru"].clear()
        self.assertEqual(len(self.call("finish", state_id="a0", output_mode="check")["prompt_cache"]["after"]["lru"]), 1)
        gc.collect()
        self.assertIsNone(self.worker.backend.runtime.packet_refs[1]())

    def test_byte_limit_evicts_even_with_spare_entry_slots(self):
        self.enable_cache(entries=4, byte_limit=100)
        self.complete("a", "a")
        last = self.complete("b", "b")
        self.assertEqual(len(last["evicted"]), 1)
        self.assertEqual(last["after"]["entries"], 1)
        self.assertEqual(last["after"]["bytes"], 72)
        self.assertEqual(last["counters"]["evictions"], 1)

    def test_storage_views_are_conservatively_charged_and_oversize_bypasses(self):
        self.enable_cache(entries=4, byte_limit=100)
        self.complete("a", "a")
        runtime = self.worker.backend.runtime
        runtime.tensor_factory = lambda prompt: (FakeTensor((1,), 4, 4096), FakeTensor((1,), 8, 8))
        last = self.complete("large0", "large")
        self.assertEqual(last["entry_bytes"], 4104)
        self.assertEqual(last["entry_tensors"][0]["logical_bytes"], 4)
        self.assertEqual(last["entry_tensors"][0]["accounted_bytes"], 4096)
        self.assertEqual(last["bypass_reason"], "entry_exceeds_byte_limit")
        self.assertFalse(last["admitted"])
        self.assertEqual(last["evicted"], [])
        self.assertEqual(last["before"], last["after"])
        again = self.complete("large1", "large")
        self.assertEqual(again["outcome"], "miss")
        self.assertEqual(again["counters"]["bypasses"], 2)
        gc.collect()
        self.assertIsNone(runtime.packet_refs[-1]())

    def test_shared_storage_is_not_undercounted(self):
        self.enable_cache()
        self.call("prepare", state_id="shared", spec=dict(prompt="x"))
        tensor = FakeTensor((1,), 4, 4096)
        self.worker.backend.runtime.tensor_factory = lambda prompt: (tensor, tensor)
        receipt = self.call("advance", state_id="shared")["prompt_cache"]
        self.assertEqual(receipt["entry_bytes"], 8192)
        tiny_storage = FakeTensor((2,), 4, 4)
        self.assertEqual(SanaPrompt("x", tiny_storage, tiny_storage, 1., object()).storage_details()[0]["accounted_bytes"], 8)

    def test_encoding_generation_and_storage_errors_clear_cached_packets(self):
        for error_kind in ("encode", "generate", "storage"):
            with self.subTest(error=error_kind):
                self.enable_cache()
                self.complete("valid", "valid")
                runtime = self.worker.backend.runtime
                self.call("prepare", state_id="failing", spec=dict(prompt="new"))
                if error_kind == "storage":
                    runtime.tensor_factory = lambda prompt: (FakeTensor((1,), 4, -1), FakeTensor((1,), 8, 8))
                    exception, message = ValueError, "storage metadata"
                else:
                    setattr(runtime, error_kind + "_error", True)
                    exception, message = RuntimeError, "failed"
                with self.assertRaisesRegex(exception, message):
                    self.call("advance", state_id="failing")
                self.assertEqual(self.worker.backend._snapshot(), dict(entries=0, bytes=0, lru=[]))
                self.assertIsNone(self.worker.backend.states["failing"]["output"])
                gc.collect()
                self.assertTrue(all(reference() is None for reference in runtime.packet_refs))
                runtime.encode_error = runtime.generate_error = False
                runtime.tensor_factory = lambda prompt: (FakeTensor(storage_bytes=64), FakeTensor((1,), 8, 8))
                receipt = self.call("advance", state_id="failing")["prompt_cache"]
                self.assertEqual(receipt["outcome"], "miss")
                self.assertEqual(receipt["counters"]["errors"], 1)
                self.assertEqual(receipt["counters"]["clears"], 1)
                self.assertEqual(receipt["cold_service"], dict(count=2, total_ms=20.))

    def test_close_releases_cache_even_if_adapter_close_raises(self):
        for fails in (False, True):
            with self.subTest(adapter_close_fails=fails):
                self.enable_cache()
                self.complete("a", "a", release=False)
                backend = self.worker.backend
                backend.runtime.close_error = fails
                if fails:
                    with self.assertRaisesRegex(RuntimeError, "close failed"):
                        self.call("shutdown")
                else:
                    self.call("shutdown")
                self.assertEqual(backend.states, {})
                self.assertEqual(backend._snapshot(), dict(entries=0, bytes=0, lru=[]))
                gc.collect()
                self.assertIsNone(backend.runtime.packet_refs[0]())

    def test_cache_options_are_strict_and_validated_before_loading(self):
        invalid = [None, [], "cache", {"candidate": "eager"}, {"prompt_cache_entries": 4},
                   {"prompt_cache_bytes": 100}, {"prompt_cache_sync_mode": "fused"},
                   {"prompt_cache_sync_mode": "fused", "prompt_cache_entries": 4}]
        for value in (None, True, 1, [], {}, "", "auto", "FUSED"):
            invalid.append(dict(prompt_cache_entries=4, prompt_cache_bytes=100, prompt_cache_sync_mode=value))
        for field, too_large in (("prompt_cache_entries", 65), ("prompt_cache_bytes", 268435457)):
            for value in (True, False, 0, -1, 4., "4", None, too_large):
                options = dict(prompt_cache_entries=4, prompt_cache_bytes=100)
                options[field] = value
                invalid.append(options)
        for options in invalid:
            with self.subTest(options=options), patch("ditserve.models.sana.runtime.SanaAdapter") as adapter:
                worker = Worker("sana", dict(model_path="/m", options=options))
                with self.assertRaises(ValueError):
                    worker.dispatch(dict(version=1, id=1, op="prepare", state_id="a", spec=dict(prompt="x")))
                adapter.assert_not_called()

    def test_explicit_staged_mode_preserves_disabled_and_enabled_paths(self):
        self.worker = Worker("sana", dict(model_path="/models/sana", work_dir=self.directory.name,
                                         options=dict(prompt_cache_sync_mode="staged")))
        self.workers.append(self.worker)
        disabled = self.complete("disabled", "cat")
        self.assertEqual(disabled["outcome"], "disabled")
        self.assertEqual(disabled["service_ms"], 7.5)
        self.assertNotIn("sync_mode", disabled)
        self.enable_cache(sync_mode="staged")
        staged = self.complete("miss", "cat")
        self.assertEqual(staged["service_ms"], 10.)
        self.assertNotIn("sync_mode", staged)
        self.assertEqual(self.worker.backend.runtime.fused_calls, [])

    def test_fused_miss_then_hit_charges_host_encoding_only_once(self):
        self.enable_cache(sync_mode="fused")
        cold = self.complete("miss", "cat", seed=5)
        warm = self.complete("hit", "cat", seed=7)
        self.assertEqual(self.worker.backend.runtime.fused_calls, [("cat", 5)])
        self.assertEqual(self.worker.backend.runtime.calls, [("cat", 5), ("cat", 7)])
        self.assertEqual((cold["encode_ms"], warm["encode_ms"]), (.5, 0.))
        self.assertEqual((cold["generation_ms"], warm["generation_ms"]), (7.5, 7.5))
        self.assertEqual((cold["service_ms"], warm["service_ms"]), (8., 7.5))
        self.assertEqual(cold["timing_scope"], "fused_miss_host_wall")
        self.assertEqual(cold["encode_timing_scope"], "host encoding submission")
        self.assertEqual(warm["timing_scope"], "synchronized_preencoded_generation")
        self.assertEqual(warm["cold_service"], dict(count=1, total_ms=8.))
        self.assertEqual(warm["warm_service"], dict(count=1, total_ms=7.5))

    def test_fused_finish_preserves_explicit_output_path_and_checked_digest(self):
        self.enable_cache(sync_mode="fused")
        self.call("prepare", state_id="image", spec=dict(prompt="cat", seed=11))
        self.call("advance", state_id="image")
        path = Path(self.directory.name) / "requested-image.npy"
        written = self.call("finish", state_id="image", output_mode="file", output_path=str(path))["output"]
        checked = self.call("finish", state_id="image", output_mode="check")["output"]
        self.assertTrue(path.is_file())
        self.assertEqual(written["path"], str(path.resolve()))
        self.assertEqual(written["array_sha256"], checked["array_sha256"])
        self.assertEqual(written["shape"], [1, 1, 1])

    def test_fused_admits_and_evicts_before_generation_with_bounded_cache(self):
        self.enable_cache(entries=1, byte_limit=100, sync_mode="fused")
        self.complete("a", "a")
        backend = self.worker.backend

        def observe():
            self.assertEqual(list(backend._cache), ["b"])
            self.assertEqual(backend._cache_bytes, 72)
            self.assertIsNone(backend.runtime.packet_refs[0]())

        backend.runtime.generation_observer = observe
        receipt = self.complete("b", "b")
        self.assertEqual(receipt["evicted"], [dict(key=hashlib.sha256(b"a").hexdigest(), bytes=72)])

    def test_fused_oversize_packet_lives_for_generation_then_is_released(self):
        self.enable_cache(byte_limit=100, sync_mode="fused")
        self.complete("small", "small")
        runtime = self.worker.backend.runtime
        runtime.tensor_factory = lambda prompt: (FakeTensor((1,), 4, 4096), FakeTensor((1,), 8, 8))
        runtime.generation_observer = lambda: self.assertIsNotNone(runtime.packet_refs[-1]())
        receipt = self.complete("large", "large")
        self.assertFalse(receipt["admitted"])
        self.assertEqual(receipt["bypass_reason"], "entry_exceeds_byte_limit")
        self.assertEqual(receipt["before"], receipt["after"])
        gc.collect()
        self.assertIsNone(runtime.packet_refs[-1]())

    def test_fused_failures_clear_cache_and_allow_request_retry(self):
        for error_kind in ("encode", "generate", "storage", "timing"):
            with self.subTest(error=error_kind):
                self.enable_cache(sync_mode="fused")
                self.complete("valid", "valid")
                backend, runtime = self.worker.backend, self.worker.backend.runtime
                self.call("prepare", state_id="failing", spec=dict(prompt="new"))
                original_fused = runtime.encode_and_generate
                if error_kind == "storage":
                    runtime.tensor_factory = lambda prompt: (FakeTensor((1,), 4, -1), FakeTensor((1,), 8, 8))
                elif error_kind == "timing":
                    def inconsistent(*args, **kwargs):
                        packet, output, service = original_fused(*args, **kwargs)
                        return packet, output, dict(service, ms=999.)
                    runtime.encode_and_generate = inconsistent
                else:
                    setattr(runtime, error_kind + "_error", True)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.call("advance", state_id="failing")
                self.assertEqual(backend._snapshot(), dict(entries=0, bytes=0, lru=[]))
                self.assertIsNone(backend.states["failing"]["output"])
                gc.collect()
                self.assertTrue(all(reference() is None for reference in runtime.packet_refs))
                runtime.encode_error = runtime.generate_error = False
                runtime.encode_and_generate = original_fused
                runtime.tensor_factory = lambda prompt: (FakeTensor(storage_bytes=64), FakeTensor((1,), 8, 8))
                receipt = self.call("advance", state_id="failing")["prompt_cache"]
                self.assertEqual(receipt["outcome"], "miss")
                self.assertEqual(receipt["counters"]["errors"], 1)
                self.assertEqual(receipt["counters"]["clears"], 1)


if __name__ == "__main__":
    unittest.main()
