"""Optional, synchronous batch-one Wan output assembly without ``np.stack``.

This candidate changes only the final batch-axis assembly in the pinned
diffusers 0.36.0 VideoProcessor. Its original per-video postprocess still owns
denormalization, transfer, FP32 conversion and layout. Call ``worker.finish``
normally inside the context: its synchronization, finite check, completion
event and request-state ownership remain in effect.

The returned array is a view, and can be non-contiguous. No output arrays or
request states are retained by the statistics object. Importing this module
does not import torch, diffusers or numpy.
"""

from contextlib import contextmanager
import hashlib
import inspect
import time


POSTPROCESS_VIDEO_SHA256 = "689f2c79663f83a94d8463abf1ced5f3bf1aada773c41cf206049ca829e9a9d2"
_PATCH_TAG = "_ditserve_finish_copy_wrapper"


def validate_processor_contract(worker):
    """Reject unknown upstream assembly instead of silently changing semantics."""
    processor = worker.pipe.video_processor
    original = processor.postprocess_video
    if getattr(original, _PATCH_TAG, False):
        raise RuntimeError("finish_copy is already active on this video processor")
    if not callable(getattr(processor, "postprocess", None)):
        raise TypeError("video processor has no per-video postprocess callable")
    source = inspect.getsource(getattr(original, "__func__", original))
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != POSTPROCESS_VIDEO_SHA256:
        raise RuntimeError("finish_copy requires the verified diffusers 0.36.0 postprocess_video source")
    return {"original_source_sha256": digest,
            "scope": "verified original batch assembly; per-video postprocess reused unchanged"}


class FinishCopyStats:
    """Bounded scalar/shape metadata; timing excludes VAE decode and finite scan."""

    def __init__(self, enabled, contract):
        self.enabled = enabled
        self.contract = dict(contract)
        self.calls = self.successes = self.failures = 0
        self.total_ms = self.max_ms = 0.0
        self.min_ms = None
        self.copy_avoided_bytes = 0
        self.last_output = None
        self.last_failure = None
        self.active = False

    def summary(self):
        return dict(enabled=self.enabled, calls=self.calls, successes=self.successes,
            failures=self.failures, active=self.active, fallback_calls=0,
            postprocess_ms=dict(total=self.total_ms,
                mean=self.total_ms / self.calls if self.calls else None,
                min=self.min_ms, max=self.max_ms if self.calls else None),
            copy_avoided_bytes=self.copy_avoided_bytes,
            last_output=dict(self.last_output) if self.last_output is not None else None,
            last_failure=self.last_failure, contract=dict(self.contract),
            timing_scope="host postprocess_video including transfer/conversion and validation; excludes VAE decode and worker.finish finite scan",
            output_scope="full FP32 B,F,H,W,C output; view can be non-contiguous; no asynchronous completion")

    def validate(self, expected_calls=None):
        """Check actual executions; this is not an output-quality qualification."""
        if self.failures or not self.successes or self.successes != self.calls:
            raise RuntimeError("finish_copy did not complete every attempted postprocess call")
        if expected_calls is not None and self.calls != expected_calls:
            raise RuntimeError("finish_copy postprocess call count differs from expectation")
        return self.summary()


def _check_array(np, value, shape):
    if not isinstance(value, np.ndarray):
        raise TypeError("finish_copy requires NumPy output from original postprocess")
    if value.dtype != np.dtype("float32") or tuple(value.shape) != shape:
        raise ValueError("finish_copy requires unchanged FP32 B,F,H,W,C output shape")
    return dict(shape=tuple(int(size) for size in value.shape), dtype=str(value.dtype),
        strides=tuple(int(size) for size in value.strides), nbytes=int(value.nbytes),
        c_contiguous=bool(value.flags.c_contiguous), owns_data=bool(value.flags.owndata))


@contextmanager
def finish_copy(worker, enabled=True):
    """Instrument baseline or replace only singleton ``np.stack`` with a view.

    Usage: ``with finish_copy(worker) as stats: output = worker.finish(state)``.
    Unsupported output modes/batches, upstream changes and malformed output
    fail closed in both arms. The exact prior instance attribute (or absence)
    is restored even on failure or cancellation. The caller must serialize use
    of this worker, as required by the existing GPU execution protocol.
    """
    if type(enabled) is not bool:
        raise TypeError("finish_copy enabled must be bool")
    import numpy as np

    contract = validate_processor_contract(worker)
    processor = worker.pipe.video_processor
    original = processor.postprocess_video
    original_postprocess = processor.postprocess
    instance_attributes = vars(processor)
    had_instance_method = "postprocess_video" in instance_attributes
    prior_instance_method = instance_attributes.get("postprocess_video")
    stats = FinishCopyStats(enabled, contract)

    def wrapped(video, output_type="np"):
        stats.calls += 1
        started = time.perf_counter()
        try:
            if output_type != "np":
                raise ValueError("finish_copy only supports output_type='np'")
            if not worker.torch.is_tensor(video) or video.ndim != 5 or video.shape[0] != 1:
                raise ValueError("finish_copy only supports one B,C,F,H,W video tensor")
            if any(int(size) <= 0 for size in video.shape):
                raise ValueError("finish_copy does not support empty video dimensions")
            expected_shape = (1, int(video.shape[2]), int(video.shape[3]),
                              int(video.shape[4]), int(video.shape[1]))
            if enabled:
                batch_video = video[0].permute(1, 0, 2, 3)
                batch_output = original_postprocess(batch_video, output_type)
                _check_array(np, batch_output, expected_shape[1:])
                result = np.expand_dims(batch_output, axis=0)
                if not np.shares_memory(result, batch_output):
                    raise RuntimeError("finish_copy unexpectedly copied singleton batch output")
            else:
                result = original(video, output_type=output_type)
            metadata = _check_array(np, result, expected_shape)
            stats.last_output = metadata
            stats.successes += 1
            if enabled:
                stats.copy_avoided_bytes += metadata["nbytes"]
            return result
        except BaseException as error:
            stats.failures += 1
            stats.last_failure = (type(error).__name__ + ": " + str(error))[:512]
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            stats.total_ms += elapsed_ms
            stats.max_ms = max(stats.max_ms, elapsed_ms)
            stats.min_ms = elapsed_ms if stats.min_ms is None else min(stats.min_ms, elapsed_ms)

    setattr(wrapped, _PATCH_TAG, True)
    processor.postprocess_video = wrapped
    stats.active = True
    try:
        yield stats
    finally:
        if had_instance_method:
            processor.postprocess_video = prior_instance_method
        else:
            delattr(processor, "postprocess_video")
        stats.active = False
