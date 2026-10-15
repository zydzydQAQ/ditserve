"""CPU tensor tests against installed diffusers; no model or CUDA execution."""

import gc
from contextlib import nullcontext
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch
import weakref

try:
    import numpy as np
    import torch
    from diffusers.video_processor import VideoProcessor
except ImportError:
    np = torch = VideoProcessor = None

from ditserve.optimizations.followup_finish import finish_copy, validate_processor_contract
from ditserve.gpu.worker import WanStepWorker


@unittest.skipUnless(VideoProcessor is not None, "requires CPU torch, numpy and pinned diffusers")
class FinishCopyTests(unittest.TestCase):
    def setUp(self):
        self.processor = VideoProcessor()
        self.worker = SimpleNamespace(pipe=SimpleNamespace(video_processor=self.processor), torch=torch)
        self.video = torch.linspace(-1.5, 1.5, 1 * 3 * 5 * 4 * 6).reshape(1, 3, 5, 4, 6)

    def test_exact_values_dtype_shape_and_expected_layout(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for video in (self.video.to(dtype), self.video.to(dtype).transpose(-1, -2)):
                reference = self.processor.postprocess_video(video, "np")
                with finish_copy(self.worker) as stats:
                    candidate = self.processor.postprocess_video(video, "np")
                np.testing.assert_array_equal(candidate, reference, strict=True)
                self.assertEqual(candidate.dtype, np.float32)
                self.assertFalse(candidate.flags.owndata)
                self.assertFalse(candidate.flags.c_contiguous)
                self.assertEqual(candidate.min(), 0)
                self.assertEqual(candidate.max(), 1)
                summary = stats.validate(expected_calls=1)
                self.assertEqual(summary["copy_avoided_bytes"], candidate.nbytes)
                self.assertEqual(summary["last_output"]["shape"], candidate.shape)
                self.assertFalse(summary["active"])
                self.assertNotIn("postprocess_video", vars(self.processor))

    def test_only_batch_assembly_is_removed(self):
        original = self.processor.postprocess
        captured = []
        def per_video(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(result)
            return result
        with patch.object(self.processor, "postprocess", side_effect=per_video) as call:
            with finish_copy(self.worker) as stats:
                with patch("numpy.stack", side_effect=AssertionError("must not stack")):
                    output = self.processor.postprocess_video(self.video)
        self.assertEqual(call.call_count, 1)
        self.assertTrue(np.shares_memory(output, captured[0]))
        np.testing.assert_array_equal(output, np.stack(captured), strict=True)
        stats.validate(1)

    def test_baseline_calls_original_stack_and_accounts_no_avoided_bytes(self):
        with finish_copy(self.worker, enabled=False) as stats:
            with patch("numpy.stack", wraps=np.stack) as stack:
                result = self.processor.postprocess_video(self.video)
        self.assertEqual(stack.call_count, 1)
        self.assertTrue(result.flags.owndata)
        self.assertEqual(stats.validate(1)["copy_avoided_bytes"], 0)

    def test_rejects_unsupported_modes_batches_shapes_and_empty_tensors(self):
        cases = [(self.video, "pt"), (self.video.repeat(2, 1, 1, 1, 1), "np"),
                 (self.video[0], "np"), (self.video[:, :, :0], "np"), (object(), "np")]
        for enabled in (False, True):
            for video, mode in cases:
                with self.assertRaises(ValueError):
                    with finish_copy(self.worker, enabled=enabled) as stats:
                        self.processor.postprocess_video(video, mode)
                self.assertEqual(stats.failures, 1)
                self.assertEqual(stats.successes, 0)
                self.assertNotIn("postprocess_video", vars(self.processor))
                with self.assertRaises(RuntimeError):
                    stats.validate()

    def test_bad_per_video_output_is_not_silently_fallback(self):
        for result in (np.zeros((5, 4, 6, 3), dtype=np.float64),
                       np.zeros((5, 4, 6, 2), dtype=np.float32), [], None):
            with patch.object(self.processor, "postprocess", return_value=result) as call:
                with self.assertRaises((TypeError, ValueError)):
                    with finish_copy(self.worker) as stats:
                        self.processor.postprocess_video(self.video)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(stats.summary()["fallback_calls"], 0)
            self.assertNotIn("postprocess_video", vars(self.processor))

    def test_exception_and_cancellation_restore_exact_prior_method(self):
        prior = self.processor.postprocess_video
        self.processor.postprocess_video = prior
        for error in (RuntimeError("decode failed"), KeyboardInterrupt("budget/cancel")):
            with self.assertRaises(type(error)):
                with finish_copy(self.worker) as stats:
                    raise error
            self.assertIs(self.processor.postprocess_video, prior)
            self.assertFalse(stats.active)
        with patch.object(self.processor, "postprocess", side_effect=MemoryError("allocation")):
            with self.assertRaises(MemoryError):
                with finish_copy(self.worker) as stats:
                    self.processor.postprocess_video(self.video)
        self.assertIs(self.processor.postprocess_video, prior)
        self.assertEqual(stats.failures, 1)

    def test_nested_patch_and_unverified_implementation_fail_closed(self):
        with finish_copy(self.worker):
            active = self.processor.postprocess_video
            with self.assertRaisesRegex(RuntimeError, "already active"):
                with finish_copy(self.worker):
                    self.fail("must reject nested replacement")
            self.assertIs(self.processor.postprocess_video, active)
        self.processor.postprocess_video = lambda video, output_type="np": video
        with self.assertRaisesRegex(RuntimeError, "verified diffusers"):
            validate_processor_contract(self.worker)

    def test_stats_do_not_retain_output_or_request_state(self):
        with finish_copy(self.worker) as stats:
            result = self.processor.postprocess_video(self.video)
            output_ref = weakref.ref(result)
        del result
        gc.collect()
        self.assertIsNone(output_ref())
        self.assertEqual(set(vars(self.worker)), {"pipe", "torch"})
        self.assertNotIn("postprocess_video", vars(self.processor))
        with self.assertRaisesRegex(RuntimeError, "count"):
            stats.validate(expected_calls=2)

    def test_real_worker_finish_still_checks_finite_output_and_completion(self):
        worker = WanStepWorker.__new__(WanStepWorker)
        worker._lock = threading.RLock()
        worker.torch = torch
        worker.pipe = self.worker.pipe
        worker.p = SimpleNamespace(_sdpa_math_context=lambda torch: nullcontext())
        worker._check = lambda *args: None
        worker._sync = lambda: None
        state = SimpleNamespace(step_index=32, spec={"steps": 32}, finished=False,
                                latents=self.video, events=[])
        def decode(torch, pipe, latent):
            return pipe.video_processor.postprocess_video(latent, "np")
        with patch("ditserve.gpu.phase_conflict.decode_final_latent", side_effect=decode):
            with finish_copy(worker) as stats:
                result = worker.finish(state)
        self.assertTrue(state.finished)
        self.assertTrue(state.events[0]["finite"])
        self.assertEqual(state.events[0]["shape"], list(result.shape))
        stats.validate(1)
        state.finished, state.events = False, []
        state.latents = self.video.clone()
        state.latents[0, 0, 0, 0, 0] = float("nan")
        with patch("ditserve.gpu.phase_conflict.decode_final_latent", side_effect=decode):
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                with finish_copy(worker):
                    worker.finish(state)
        self.assertFalse(state.finished)
        self.assertEqual(state.events, [])
        self.assertNotIn("postprocess_video", vars(self.processor))


if __name__ == "__main__":
    unittest.main()
