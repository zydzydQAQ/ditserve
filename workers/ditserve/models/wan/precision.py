"""Shared Wan text-encoding precision for serving, reference and profiling."""

TEXT_ENCODER_DTYPE = "bfloat16"


def text_encoder_dtype(value=TEXT_ENCODER_DTYPE):
    """BF16 by default; FP32 is an explicit comparison mode, never a fallback."""
    if value not in ("bfloat16", "float32"):
        raise ValueError("Wan text_encoder_dtype must be bfloat16 or float32")
    return value
