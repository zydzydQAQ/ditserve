"""One VAE stream overlapping the main Wan stream; no dispatch policy here."""

from concurrent.futures import ThreadPoolExecutor
import time

from ..pipeline import decode_final_latent


class AsyncDecoder:
    def __init__(self, worker):
        if worker.attention not in ("math", "flash"):
            raise ValueError("asynchronous VAE requires a pinned SDPA execution mode")
        if worker.attention == "flash" and worker.vae_dtype not in ("bfloat16", "float16"):
            raise ValueError("Flash asynchronous VAE requires a 16-bit VAE")
        self.worker, self.torch = worker, worker.torch
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wan-vae")
        self.stream = self.torch.cuda.Stream(device=worker.device)
        self.future = self.context = None
        self.original_sync = None
        self.failed = False

    def submit(self, state):
        if self.failed or self.future is not None:
            raise ValueError("decoder is failed or still owns an uncollected decode")
        worker, torch = self.worker, self.torch
        worker._check(state, "gpu")
        if state.finished or state.step_index != state.spec["steps"]:
            raise ValueError("async decode requires a complete undecoded state")
        if worker.attention == "flash":
            self.context = torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION])
        else:
            self.context = worker.p._sdpa_math_context(torch)
        self.context.__enter__()
        self.original_sync = worker._sync
        worker._sync = lambda: torch.cuda.current_stream(worker.device).synchronize()
        try:
            boundary = torch.cuda.Event()
            boundary.record(torch.cuda.current_stream(worker.device))
            state.latents.record_stream(self.stream)
            self.future = self.pool.submit(self._execute, state.latents, boundary)
        except BaseException:
            self._restore()
            raise

    def _execute(self, latent, boundary):
        torch, worker = self.torch, self.worker
        started = time.perf_counter()
        with torch.cuda.device(worker.device), torch.inference_mode(), torch.cuda.stream(self.stream):
            flags = torch.backends.cuda
            if worker.attention == "math":
                valid = (flags.math_sdp_enabled() and not flags.flash_sdp_enabled()
                         and not flags.mem_efficient_sdp_enabled() and not flags.cudnn_sdp_enabled())
            else:
                valid = (flags.flash_sdp_enabled() and not flags.math_sdp_enabled()
                         and not flags.mem_efficient_sdp_enabled() and not flags.cudnn_sdp_enabled())
            if not valid:
                raise RuntimeError("VAE thread escaped its pinned SDPA policy")
            self.stream.wait_event(boundary)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(self.stream)
            output = decode_final_latent(torch, worker.pipe, latent)
            end.record(self.stream)
            end.synchronize()
            import numpy as np
            if not np.isfinite(output).all():
                raise RuntimeError("nonfinite asynchronous VAE output")
            return output, dict(kind="decode", ms=(time.perf_counter() - started) * 1000,
                cuda_ms=float(begin.elapsed_time(end)), stream=int(self.stream.cuda_stream),
                asynchronous=True, timing_scope="decoder stream; main model stream is not synchronized")

    def collect(self, *, wait=False):
        if self.future is None:
            raise ValueError("no active decode")
        if not wait and not self.future.done():
            return None
        try:
            return self.future.result()
        except BaseException:
            self.failed = True
            raise
        finally:
            self.future = None
            self._restore()

    def _restore(self):
        if self.original_sync is not None:
            self.worker._sync, self.original_sync = self.original_sync, None
        if self.context is not None:
            self.context.__exit__(None, None, None)
            self.context = None

    def close(self):
        try:
            if self.future is not None:
                self.collect(wait=True)
        finally:
            self.pool.shutdown(wait=True)
            self._restore()
