"""Stream-isolated concurrent 3D-VAE chunked decode overlapping with T2I denoising.

When a Wan2.1 T2V request completes its final denoising step while bursty
SD3.5 T2I requests are pending, sequential execution either blocks SD3.5 behind
3D-VAE decode (~320-650 ms) or serializes chunked decode and SD3.5 kernels on
the same default CUDA stream.

``ConcurrentVAEDecodeStreamer`` binds ``ChunkedDecode`` to a dedicated secondary
CUDA stream (``vae_stream``) synchronized via CUDA events, allowing causal
3D-VAE frame-chunk decoding to overlap concurrently with compute-bound SD3.5
MM-DiT denoising on the primary stream without global SDPA context races and
while keeping peak HBM strictly bounded.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import importlib
import time
from typing import Any, Callable, Dict, Tuple

from ..pipeline import ChunkedDecode


class ConcurrentVAEDecodeStreamer:
    """Dedicated CUDA stream executor for overlapping Wan 3D-VAE decode with SD3.5 denoising."""

    def __init__(self, wan_runtime, *, chunk_frames: int = 1):
        self.runtime = wan_runtime
        self.torch = wan_runtime.torch
        self.device = wan_runtime.device
        self.chunk_frames = max(1, int(chunk_frames))
        self.vae_stream = (
            self.torch.cuda.Stream(device=self.device)
            if str(self.device).startswith("cuda") and self.torch.cuda.is_available()
            else None
        )
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wan-vae-stream")

    def _decode_worker(self, state, ready_event, chunk_frames: int) -> Tuple[Any, Dict[str, Any]]:
        torch = self.torch
        t0 = time.perf_counter()
        stream_ctx = torch.cuda.stream(self.vae_stream) if self.vae_stream is not None else torch.inference_mode()
        chunk_count = 0
        with stream_ctx, torch.inference_mode():
            if self.vae_stream is not None and ready_event is not None:
                self.vae_stream.wait_event(ready_event)
            decoder = ChunkedDecode(torch, self.runtime.pipe, state.latents)
            total_latent_frames = decoder.frames
            while not decoder.done:
                decoder.advance(chunk_frames)
                chunk_count += 1
            if self.vae_stream is not None:
                self.vae_stream.synchronize()
            video_np = decoder.finish()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        state.finished = True
        info = {
            "kind": "stream_overlapped_decode",
            "vae_stream_ms": round(elapsed_ms, 3),
            "latent_frames": total_latent_frames,
            "chunks_executed": chunk_count,
            "chunk_frames": chunk_frames,
            "shape": list(video_np.shape),
        }
        state.events.append(info)
        return video_np, info

    def submit_decode(self, state, *, chunk_frames: int | None = None) -> Future:
        """Launch chunked 3D-VAE decode on `` self.vae_stream`` after recording a ready event on default stream."""
        torch = self.torch
        if state.step_index != state.spec["steps"] or state.finished:
            raise ValueError("ConcurrentVAEDecodeStreamer requires a completed, undecoded Wan state")
        ready_event = None
        if self.vae_stream is not None:
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(self.device))
        cf = self.chunk_frames if chunk_frames is None else max(1, int(chunk_frames))
        return self._pool.submit(self._decode_worker, state, ready_event, cf)

    def run_overlapped(
        self,
        state,
        concurrent_fn: Callable[[], Dict[str, Any]],
        *,
        chunk_frames: int | None = None,
    ) -> Dict[str, Any]:
        """Run ``concurrent_fn()`` on the primary stream while decoding ``state`` on ``vae_stream``."""
        torch = self.torch
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        t_wall_start = time.perf_counter()
        fut = self.submit_decode(state, chunk_frames=chunk_frames)
        primary_res = concurrent_fn()
        video_np, vae_info = fut.result()
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        wall_ms = (time.perf_counter() - t_wall_start) * 1000.0
        peak_hbm_mb = (
            round(torch.cuda.max_memory_allocated(self.device) / (1 << 20), 2)
            if str(self.device).startswith("cuda")
            else 0.0
        )
        return {
            "wall_ms": round(wall_ms, 3),
            "vae_decode": vae_info,
            "primary_task": primary_res,
            "peak_hbm_mb": peak_hbm_mb,
            "video_np": video_np,
        }

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
