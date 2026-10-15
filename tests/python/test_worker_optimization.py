"""CPU tests for the serving boundary; no model loading or CUDA execution."""

import contextlib
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ditserve.gpu.worker import ATTENTION_BACKENDS, WanStepWorker


def spec(**changes):
    return dict(request_id="request-a", prompt="a small boat", negative_prompt="blur",
                seed=12, height=32, width=32, frames=5, steps=2, guidance_scale=5., **changes)


class WorkerOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU torch is required for worker boundary checks")
        cls.torch = torch

    def worker(self, attention="math"):
        """Small numerical stand-in that executes the real serving methods."""
        from ditserve.gpu import profiling
        torch = self.torch
        worker = WanStepWorker.__new__(WanStepWorker)
        worker.torch, worker.p, worker.device = torch, profiling, "cpu"
        worker.attention = attention
        worker._lock, worker._encoder_lock = threading.RLock(), threading.Lock()
        worker._owner, worker._states = object(), {}
        worker._sync = Mock()
        # State-storage schema validation has its own contract; this scheduler
        # intentionally only implements the operations needed by these tests.
        worker._validate_request_schema = Mock()
        worker.manager = SimpleNamespace(_current_context=None, _cached_context_manager={},
                                         _current_step_refreshed=False)
        worker.observed_attention = []

        class Transformer(torch.nn.Module):
            dtype = torch.float16
            config = SimpleNamespace(in_channels=2)

            def cache_context(self, *args):
                return contextlib.nullcontext()

            def forward(self, hidden_states, **kwargs):
                worker.observed_attention.append(self_flags())
                return (hidden_states * .5,)

        def self_flags():
            cuda = torch.backends.cuda
            return dict(math=cuda.math_sdp_enabled(), flash=cuda.flash_sdp_enabled(),
                        efficient=cuda.mem_efficient_sdp_enabled(), cudnn=cuda.cudnn_sdp_enabled())

        class Scheduler:
            @classmethod
            def from_config(cls, config):
                return cls()

            def set_timesteps(self, steps, device):
                self.timesteps = torch.arange(steps, device=device)

            def step(self, prediction, timestep, latents, **kwargs):
                return (latents + prediction.float() * .01,)

        worker.flags = self_flags
        worker._scheduler_type, worker._scheduler_config = Scheduler, {}
        worker.pipe = SimpleNamespace(
            transformer=Transformer(), scheduler=Scheduler(),
            encode_prompt=Mock(side_effect=lambda **kwargs: (
                torch.ones((1, 4, 4), dtype=torch.float32),
                torch.zeros((1, 4, 4), dtype=torch.float32) if kwargs["do_classifier_free_guidance"] else None)),
            prepare_latents=lambda batch, channels, height, width, frames, dtype, device, generator, _: (
                torch.randn((batch, channels, 2, 2, 2), dtype=dtype, device=device, generator=generator)))
        worker._original_forward = worker._cached_forward = worker.pipe.transformer.forward
        return worker

    def test_invalid_attention_is_rejected_before_imports_or_loading(self):
        with patch("ditserve.gpu.worker.importlib.metadata.version") as version:
            with self.assertRaisesRegex(ValueError, "attention"):
                WanStepWorker(attention="automatic")
        version.assert_not_called()

    def test_cpu_encoding_overlaps_gpu_lock_without_cuda_or_sdpa_changes(self):
        worker = self.worker(attention="flash")
        worker._sync.side_effect = AssertionError("CPU encoder must not synchronize GPU")
        before = worker.flags()
        # The background thread must complete while the main thread still owns
        # the GPU lock. Acquiring that lock in encode_prompt would time out.
        with ThreadPoolExecutor(max_workers=1) as executor:
            with worker._lock, patch.object(self.torch.cuda, "synchronize", side_effect=AssertionError("CUDA")), \
                    patch.object(self.torch.nn.attention, "sdpa_kernel", side_effect=AssertionError("SDPA mutation")):
                payload = executor.submit(worker.encode_prompt, spec()).result(timeout=2)
        self.assertEqual(before, worker.flags())
        self.assertEqual(payload.prompt_embeds.device.type, "cpu")
        self.assertEqual(payload.prompt_embeds.dtype, self.torch.float32)
        self.assertFalse(payload.consumed)
        worker._sync.assert_not_called()
        self.assertEqual(worker.pipe.encode_prompt.call_args.kwargs["device"], "cpu")

    def test_encoder_serializes_only_encoding_and_never_reuses_embeddings(self):
        worker = self.worker()
        started, release = threading.Event(), threading.Event()
        original = worker.pipe.encode_prompt.side_effect
        calls = []

        def encode(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                started.set()
                if not release.wait(2):
                    raise AssertionError("test failed to release encoder")
            return original(**kwargs)

        worker.pipe.encode_prompt.side_effect = encode
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(worker.encode_prompt, spec())
            self.assertTrue(started.wait(1))
            second = executor.submit(worker.encode_prompt, dict(spec(), request_id="request-b"))
            self.assertFalse(second.done())
            release.set()
            left, right = first.result(timeout=2), second.result(timeout=2)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(left.prompt_embeds.data_ptr(), right.prompt_embeds.data_ptr())
        self.assertEqual(left.spec["request_id"], "request-a")
        self.assertEqual(right.spec["request_id"], "request-b")

    def test_split_prepare_accounts_service_work_and_consumes_payload_once(self):
        worker = self.worker()
        request = spec()
        payload = worker.encode_prompt(request)
        payload.cpu_prompt_encode_ms = 500.
        with patch("ditserve.gpu.worker.time.perf_counter", side_effect=[10., 10.003]):
            state = worker.prepare_encoded(request, payload)
        self.assertTrue(payload.consumed)
        self.assertAlmostEqual(state.prepare_ms, 503.)
        self.assertAlmostEqual(state.events[0]["gpu_prepare_ms"], 3.)
        self.assertEqual(state.events[0]["cpu_prompt_encode_ms"], 500.)
        self.assertEqual(state.prompt_embeds.dtype, self.torch.float16)
        self.assertEqual(worker.pipe.encode_prompt.call_count, 1)
        worker.release(state)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            worker.prepare_encoded(request, payload)

    def test_payload_cannot_move_between_workers_or_requests(self):
        worker = self.worker()
        original = spec()
        payload = worker.encode_prompt(original)
        original["prompt"] = "changed after encoding"
        with self.assertRaisesRegex(ValueError, "does not match"):
            worker.prepare_encoded(original, payload)
        for key, value in (("request_id", "different"), ("seed", 100), ("steps", 3)):
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "does not match"):
                worker.prepare_encoded(dict(spec(), **{key: value}), payload)
        with self.assertRaisesRegex(ValueError, "another worker"):
            self.worker().prepare_encoded(spec(), payload)
        self.assertFalse(payload.consumed)
        worker._sync.assert_not_called()

    def test_payload_validates_cpu_finite_precision_before_gpu_work(self):
        worker = self.worker()
        for tensor, exception, message in (
            (self.torch.ones(2, dtype=self.torch.float16), ValueError, "CPU float32"),
            (self.torch.ones(2, device="meta"), ValueError, "CPU float32"),
            (self.torch.tensor([float("nan")]), RuntimeError, "nonfinite"),
        ):
            payload = worker.encode_prompt(spec())
            payload.prompt_embeds = tensor
            with self.subTest(tensor=str(tensor)), self.assertRaisesRegex(exception, message):
                worker.prepare_encoded(spec(), payload)
        worker._sync.assert_not_called()

    def test_guidance_disabled_allows_no_negative_embeddings(self):
        worker = self.worker()
        state = worker.prepare(dict(spec(), guidance_scale=1.))
        self.assertIsNone(state.negative_prompt_embeds)
        self.assertEqual(state.prepare_ms, state.events[0]["ms"])

    def test_dit_backend_is_forced_and_decode_remains_math(self):
        for attention in ATTENTION_BACKENDS:
            with self.subTest(attention=attention):
                worker = self.worker(attention)
                before = worker.flags()
                state = worker.prepare(spec())
                while state.step_index < state.spec["steps"]:
                    worker.step(state)
                expected = dict(math=False, flash=False, efficient=False, cudnn=False)
                expected[attention] = True
                self.assertEqual(worker.observed_attention, [expected] * 4)
                decode_flags = []

                def decode(torch, pipe, latent):
                    decode_flags.append(worker.flags())
                    return latent.cpu().numpy()

                with patch("ditserve.gpu.phase_conflict.decode_final_latent", side_effect=decode):
                    worker.finish(state)
                self.assertEqual(decode_flags, [dict(math=True, flash=False, efficient=False, cudnn=False)])
                self.assertEqual(before, worker.flags())

    def test_reference_uses_same_dit_policy_and_restores_forward_after_failure(self):
        worker = self.worker("flash")
        before = worker.flags()
        decode_flags = []

        def official(pipe, **kwargs):
            latents = kwargs["latents"]
            for index in range(kwargs["num_inference_steps"]):
                pipe.transformer(hidden_states=latents)
                kwargs["callback_on_step_end"](pipe, index, None, dict(latents=latents))
            decode_flags.append(worker.flags())
            return SimpleNamespace(frames=latents.cpu().numpy())

        worker._official_call = official
        worker.reference(spec())
        self.assertEqual(worker.observed_attention,
                         [dict(math=False, flash=True, efficient=False, cudnn=False)] * 2)
        self.assertEqual(decode_flags, [dict(math=True, flash=False, efficient=False, cudnn=False)])
        self.assertEqual(before, worker.flags())
        self.assertFalse(worker._states)
        worker._official_call = Mock(side_effect=RuntimeError("No available kernel"))
        with self.assertRaisesRegex(RuntimeError, "No available kernel"):
            worker.reference(spec())
        self.assertFalse(worker._states)
        self.assertEqual(worker.pipe.transformer.forward, worker._cached_forward)
        self.assertEqual(before, worker.flags())

    def test_unsupported_dit_kernel_fails_without_retry_or_backend_fallback(self):
        worker = self.worker("efficient")
        state = worker.prepare(spec())
        forward = Mock(side_effect=RuntimeError("No available kernel"))
        worker._original_forward = forward
        before = worker.flags()
        with self.assertRaisesRegex(RuntimeError, "No available kernel"):
            worker.step(state)
        forward.assert_called_once()
        self.assertEqual(state.step_index, 0)
        self.assertEqual(before, worker.flags())
        self.assertEqual(worker.pipe.transformer.forward, worker._cached_forward)


if __name__ == "__main__":
    unittest.main()
