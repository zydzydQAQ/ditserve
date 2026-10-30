"""Shared quality metrics, module memory footprint helpers, and fast state preparation for HeteroDiT."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def module_bytes(mod: Any) -> int:
    """Return exact unique parameter + buffer bytes of a PyTorch nn.Module."""
    if mod is None:
        return 0
    total = 0
    seen: set[int] = set()
    for p in mod.parameters(recurse=True):
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += p.numel() * p.element_size()
    for b in mod.buffers(recurse=True):
        ptr = b.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += b.numel() * b.element_size()
    return total


def module_params(mod: Any) -> int:
    """Return unique parameter count of a PyTorch nn.Module."""
    if mod is None:
        return 0
    seen: set[int] = set()
    total = 0
    for p in mod.parameters(recurse=True):
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += p.numel()
    return total


def compute_psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) for images/videos in [0, 1]."""
    ref_f = ref.astype(np.float64)
    test_f = test.astype(np.float64)
    mse = float(np.mean((ref_f - test_f) ** 2))
    if mse <= 1e-12:
        return 99.99
    return round(float(10.0 * math.log10(1.0 / mse)), 2)


def compute_ssim_batch(ref_imgs: np.ndarray, test_imgs: np.ndarray) -> float:
    """Compute mean SSIM across a batch of images or video frames [N, H, W, C] in [0, 1]."""
    import torch
    import torch.nn.functional as F

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    scores = []
    for i in range(ref_imgs.shape[0]):
        t_x = torch.from_numpy(ref_imgs[i].astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        t_y = torch.from_numpy(test_imgs[i].astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        mu_x = F.avg_pool2d(t_x, kernel_size=11, stride=1, padding=5)
        mu_y = F.avg_pool2d(t_y, kernel_size=11, stride=1, padding=5)
        sigma_x = F.avg_pool2d(t_x * t_x, 11, 1, 5) - mu_x.pow(2)
        sigma_y = F.avg_pool2d(t_y * t_y, 11, 1, 5) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(t_x * t_y, 11, 1, 5) - mu_x * mu_y
        ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
        )
        scores.append(float(ssim_map.mean().item()))
    return round(float(np.mean(scores)), 5)


compute_video_ssim = compute_ssim_batch


def fast_prepare_wan(wan: Any, spec: dict, req_id: str, cached_payload: Any) -> Any:
    """Prepare a Wan2.1 denoising state from pre-encoded prompt embeddings without re-encoding."""
    from ditserve.models.wan.block import WanBlockExecutor
    from ditserve.models.wan.runtime import EncodedPrompt, normalize_spec

    full_spec = normalize_spec(dict(wan.config.get("spec", {}), **spec, request_id=req_id))
    payload = EncodedPrompt(
        spec=full_spec,
        prompt_embeds=cached_payload.prompt_embeds,
        negative_prompt_embeds=cached_payload.negative_prompt_embeds,
        cpu_prompt_encode_ms=cached_payload.cpu_prompt_encode_ms,
        owner=wan.runtime._owner,
        consumed=False,
    )
    state = wan.runtime.prepare_encoded(full_spec, payload)
    if wan.execution.startswith("block"):
        if wan.executor is None:
            wan.executor = WanBlockExecutor(wan.runtime, interleave=wan.interleave)
        wan.continuations[req_id] = wan.executor.attach(state)
    wan.states[req_id] = state
    return state


__all__ = [
    "module_bytes",
    "module_params",
    "compute_psnr",
    "compute_ssim_batch",
    "compute_video_ssim",
    "fast_prepare_wan",
]
