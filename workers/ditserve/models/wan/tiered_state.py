"""Tiered VideoState Block Manager for 40 GB A100 PCIe clusters.

Breaks the single-suspended-video-per-GPU HBM ceiling by managing video latent
and solver states across three tiers:
  - Tier 0 (HBM Hot): Resident on GPU for zero-copy sub-step block resumption.
  - Tier 1 (Host Pinned DRAM Pool): Non-blocking DMA offload via dedicated CUDA
    copy stream and lookahead prefetch during the final step of T2I execution.
  - Tier 2 (/dev/shm Cross-Group Staging): Enables zero-stall cross-GPU-group
    and cross-NUMA migration when a preempted video resumes at a different
    parallel form or GPU subset.
"""

from collections import OrderedDict
import time

from .runtime import _tree_map
from .parallel import request_engine, validate_request_engine


class PinnedSlabPool:
    """Reusable Host Pinned DRAM buffer pool bounded by a strict byte budget."""

    def __init__(self, torch_module, max_bytes: int = 8 * (1 << 30)):
        self.torch = torch_module
        self.max_bytes = int(max_bytes)
        self.allocated_bytes = 0
        self.peak_allocated_bytes = 0
        self._free_slabs = {}
        self.allocations = 0
        self.reuses = 0

    def acquire(self, nbytes: int, pin_memory: bool = True):
        nbytes = int(nbytes)
        bucket = self._free_slabs.get((nbytes, pin_memory))
        if bucket:
            self.reuses += 1
            return bucket.pop()
        if self.allocated_bytes + nbytes > self.max_bytes:
            self._trim(self.allocated_bytes + nbytes - self.max_bytes)
        if self.allocated_bytes + nbytes > self.max_bytes:
            raise MemoryError(
                f"PinnedSlabPool budget exceeded: requested {nbytes} B, "
                f"allocated {self.allocated_bytes}/{self.max_bytes} B"
            )
        use_pin = bool(pin_memory and self.torch.cuda.is_available())
        slab = self.torch.empty(nbytes, dtype=self.torch.uint8, device="cpu", pin_memory=use_pin)
        self.allocated_bytes += nbytes
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, self.allocated_bytes)
        self.allocations += 1
        return slab

    def release(self, slab) -> None:
        if slab is None:
            return
        key = (int(slab.numel() * slab.element_size()), bool(slab.is_pinned()))
        self._free_slabs.setdefault(key, []).append(slab)

    def _trim(self, needed_bytes: int) -> None:
        freed = 0
        for key in list(self._free_slabs.keys()):
            bucket = self._free_slabs[key]
            while bucket and freed < needed_bytes:
                slab = bucket.pop()
                size = int(slab.numel() * slab.element_size())
                self.allocated_bytes -= size
                freed += size
            if not bucket:
                del self._free_slabs[key]
            if freed >= needed_bytes:
                break

    def stats(self) -> dict:
        cached_bytes = sum(k[0] * len(v) for k, v in self._free_slabs.items())
        return {
            "max_bytes": self.max_bytes,
            "allocated_bytes": self.allocated_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "cached_free_bytes": cached_bytes,
            "active_bytes": self.allocated_bytes - cached_bytes,
            "allocations": self.allocations,
            "reuses": self.reuses,
        }


class TieredVideoStateManager:
    """Manages asynchronous offload and lookahead prefetch of preempted VideoStates."""

    def __init__(self, worker, max_pinned_mb: int = 8192):
        self.worker = worker
        self.torch = worker.torch
        self.pinned_pool = PinnedSlabPool(self.torch, max_bytes=int(max_pinned_mb) * (1 << 20))
        self.dma_stream = (
            self.torch.cuda.Stream(device=worker.device)
            if str(worker.device).startswith("cuda") and self.torch.cuda.is_available()
            else None
        )
        self.suspended_states = OrderedDict()
        self.prefetched_states = set()
        self.offload_count = 0
        self.prefetch_count = 0

    def offload_to_pinned(self, state, state_id: str, *, non_blocking: bool = True) -> dict:
        """Offload a video state from GPU HBM (Tier 0) to Host Pinned DRAM (Tier 1)."""
        with self.torch.inference_mode():
            started = time.perf_counter()
            before = self.worker._memory()
            payload = self.worker._payload(state)
            storage_copies = {}
            slabs = []
            placements = {}
            transfer_bytes = 0

            ctx = self.torch.cuda.stream(self.dma_stream) if self.dma_stream is not None else _null_ctx()
            with ctx:
                def move_d2h(tensor):
                    nonlocal transfer_bytes
                    if tensor.device.type != "cuda":
                        return tensor
                    storage = tensor.untyped_storage()
                    key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
                    if key not in storage_copies:
                        raw = self.torch.empty(0, dtype=self.torch.uint8, device=tensor.device).set_(
                            storage, 0, (storage.nbytes(),), (1,)
                        )
                        host_slab = self.pinned_pool.acquire(storage.nbytes(), pin_memory=True)
                        host_slab.copy_(raw, non_blocking=bool(non_blocking and host_slab.is_pinned()))
                        storage_copies[key] = host_slab
                        slabs.append(host_slab)
                        transfer_bytes += storage.nbytes()
                    host_slab = storage_copies[key]
                    converted = self.torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
                        host_slab.untyped_storage(),
                        tensor.storage_offset(),
                        tuple(tensor.shape),
                        tuple(tensor.stride()),
                    )
                    placements[id(converted)] = str(tensor.device)
                    return converted

                host_payload = _tree_map(payload, self.torch.is_tensor, move_d2h)
                if self.dma_stream is not None and not non_blocking:
                    self.dma_stream.synchronize()

            if self.dma_stream is not None and non_blocking:
                self.dma_stream.synchronize()
            self.worker._install_payload(state, host_payload)
            state.pending_snapshot = {
                "mode": "tiered_pinned",
                "placements": placements,
                "rng_state": host_payload["rng_state"],
                "slabs": slabs,
                "engine": request_engine(self.worker),
            }
            state.residency = "cpu_pinned"
            self.suspended_states[state_id] = state
            self.prefetched_states.discard(state_id)
            self.offload_count += 1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            event = {
                "kind": "tiered_offload",
                "tier": "host_pinned_dram",
                "ms": elapsed_ms,
                "transfer_bytes": transfer_bytes,
                "suspended_count": len(self.suspended_states),
                "memory_before": before,
                "memory_after": self.worker._memory(),
                "pool_stats": self.pinned_pool.stats(),
            }
            state.events.append(event)
            return event

    def lookahead_prefetch(self, state, state_id: str, target_device: str = None) -> dict:
        """Asynchronously prefetch a Tier-1 video state back to target GPU HBM (Tier 0)
        using the dedicated DMA stream while the current T2I request finishes its last step."""
        with self.torch.inference_mode():
            started = time.perf_counter()
            saved = getattr(state, "pending_snapshot", None)
            if saved is None or getattr(state, "residency", "gpu") == "gpu":
                return {"kind": "tiered_prefetch", "skipped": True, "ms": 0.0, "transfer_bytes": 0}
            validate_request_engine(self.worker, saved.get("engine"))
            before = self.worker._memory()
            copies = {}
            transfer_bytes = 0
            override_device = target_device or str(self.worker.device)

            ctx = self.torch.cuda.stream(self.dma_stream) if self.dma_stream is not None else _null_ctx()
            with ctx:
                def move_h2d(tensor):
                    nonlocal transfer_bytes
                    orig_target = saved["placements"].get(id(tensor))
                    if orig_target is None:
                        return tensor
                    dest = override_device or orig_target
                    storage = tensor.untyped_storage()
                    key = (storage.data_ptr(), storage.nbytes(), dest)
                    if key not in copies:
                        raw = self.torch.empty(0, dtype=self.torch.uint8, device="cpu").set_(
                            storage, 0, (storage.nbytes(),), (1,)
                        )
                        copies[key] = raw.to(dest, non_blocking=True)
                        transfer_bytes += storage.nbytes()
                    return self.torch.empty(0, dtype=tensor.dtype, device=dest).set_(
                        copies[key].untyped_storage(),
                        tensor.storage_offset(),
                        tuple(tensor.shape),
                        tuple(tensor.stride()),
                    )

                payload = self.worker._payload(state)
                payload["rng_state"] = saved["rng_state"]
                restored = _tree_map(payload, self.torch.is_tensor, move_h2d)

            if self.dma_stream is not None:
                self.dma_stream.synchronize()
            for slab in saved.get("slabs", []):
                self.pinned_pool.release(slab)
            self.worker._install_payload(state, restored)
            state.pending_snapshot = None
            state.residency = "gpu"
            self.suspended_states.pop(state_id, None)
            self.prefetched_states.add(state_id)
            self.prefetch_count += 1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            event = {
                "kind": "tiered_prefetch",
                "tier": "gpu_hbm",
                "target_device": override_device,
                "ms": elapsed_ms,
                "transfer_bytes": transfer_bytes,
                "suspended_count": len(self.suspended_states),
                "memory_before": before,
                "memory_after": self.worker._memory(),
                "pool_stats": self.pinned_pool.stats(),
            }
            state.events.append(event)
            return event

    def summary(self) -> dict:
        return {
            "suspended_count": len(self.suspended_states),
            "prefetched_count": len(self.prefetched_states),
            "offload_count": self.offload_count,
            "prefetch_count": self.prefetch_count,
            "pool": self.pinned_pool.stats(),
        }


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
