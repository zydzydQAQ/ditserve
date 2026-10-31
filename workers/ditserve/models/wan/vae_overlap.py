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

    def __init__(
        self,
        wan_runtime,
        *,
        chunk_frames: int = 1,
        default_chunk_frames: int | None = None,
        vae_dtype: str | None = None,
    ):
        self.runtime = wan_runtime
        self.torch = wan_runtime.torch
        self.device = wan_runtime.device
        cf = default_chunk_frames if default_chunk_frames is not None else chunk_frames
        self.chunk_frames = max(1, int(cf))
        self.vae_dtype = vae_dtype
        if vae_dtype is not None:
            target_dtype = getattr(self.torch, vae_dtype) if isinstance(vae_dtype, str) else vae_dtype
            if self.runtime.pipe.vae.dtype != target_dtype:
                self.runtime.pipe.vae.to(dtype=target_dtype)
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
            decoder = ChunkedDecode(torch, self.runtime.pipe, state.latents, vae_dtype=self.vae_dtype)
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
            "vae_decode_ms": round(elapsed_ms, 3),
            "latent_frames": total_latent_frames,
            "chunks_executed": chunk_count,
            "chunk_frames": chunk_frames,
            "shape": list(video_np.shape),
        }
        state.events.append(info)
        return video_np, info

    def submit_decode(self, state, *, chunk_frames: int | None = None, allow_partial: bool = False) -> Future:
        """Launch chunked 3D-VAE decode on ``self.vae_stream`` after recording a ready event on default stream."""
        torch = self.torch
        if state.finished or (not allow_partial and state.step_index != state.spec["steps"]):
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

    def submit_rank0_disaggregated(self, rank: int, state, *, chunk_frames: int | None = None):
        """Multi-GPU Rank-0 Disaggregated Async 3D-VAE Decode.

        On Rank 0, clones ``state.latents`` on ``vae_stream`` and launches asynchronous
        BF16 3D-VAE decoding without blocking Ranks 1..N-1 or Rank 0's primary compute stream.
        On Ranks 1..N-1, immediately marks the state finished and returns a zero-wait handle
        so worker ranks are freed immediately after the final diffusion step.
        """
        if int(rank) != 0:
            state.finished = True
            return None
        return self.submit_decode(state, chunk_frames=chunk_frames, allow_partial=True)

    def collect_rank0_disaggregated(self, rank: int, fut) -> Dict[str, Any]:
        """Collect the Rank-0 async 3D-VAE decode result (no-op on Ranks 1..N-1)."""
        if int(rank) != 0 or fut is None:
            return {"kind": "rank_non0_bypassed", "vae_stream_ms": 0.0, "vae_decode_ms": 0.0}
        _, vae_info = fut.result()
        return vae_info

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
