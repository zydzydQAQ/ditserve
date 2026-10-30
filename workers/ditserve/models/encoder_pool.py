"""Cross-modal text encoder deduplication pool with proactive pinned/shm staging.

Decouples heavy T5-XXL / UMT5 text encoders (9.1-19.0 GB HBM) from per-GPU DiT
denoising workers on 40 GB A100 PCIe nodes. Prompt embeddings are computed once
(with deterministic LRU caching for repeated negative prompts) and staged into
Host Pinned DRAM or /dev/shm slabs for asynchronous H2D prefetch by target DiT
workers.
"""

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time


class ProactiveEncoderPool:
    """Centralized or cross-worker shared text encoder pool with proactive staging."""

    def __init__(self, config=None):
        config = config or {}
        self.shm_dir = Path(config.get("shm_dir", "/dev/shm/ditserve-hetero-embeds"))
        self.max_entries = int(config.get("max_entries", 64))
        self.max_bytes = int(config.get("max_bytes", 512 * (1 << 20)))
        self._entries = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.staged_pushes = 0

    @staticmethod
    def prompt_key(modality: str, model_id: str, prompt: str, negative_prompt: str = "",
                   max_sequence_length: int = 256, dtype: str = "bfloat16", do_cfg: bool = True) -> str:
        payload = json.dumps(
            {
                "modality": modality,
                "model_id": str(model_id),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "max_sequence_length": int(max_sequence_length),
                "dtype": dtype,
                "do_cfg": do_cfg,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _evict_to_fit(self, needed_bytes: int) -> None:
        while self._entries and (
            len(self._entries) >= self.max_entries or self._bytes + needed_bytes > self.max_bytes
        ):
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted["bytes"]
            slab_path = evicted.get("slab_path")
            if slab_path and Path(slab_path).is_file():
                try:
                    Path(slab_path).unlink()
                except OSError:
                    pass

    def stage_embeddings(self, key: str, tensors: dict, *, pin_memory: bool = True,
                         write_shm: bool = False) -> dict:
        """Stage prompt embedding tensors in pinned host memory (and optional /dev/shm slab)."""
        import torch

        started = time.perf_counter()
        staged_tensors = {}
        total_bytes = 0
        tensor_meta = {}

        for name, tensor in tensors.items():
            if tensor is None:
                staged_tensors[name] = None
                continue
            cpu_tensor = tensor.detach().to(device="cpu", non_blocking=False).contiguous()
            if pin_memory and torch.cuda.is_available():
                pinned = torch.empty_like(cpu_tensor, device="cpu", pin_memory=True)
                pinned.copy_(cpu_tensor, non_blocking=False)
                cpu_tensor = pinned
            nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
            total_bytes += nbytes
            staged_tensors[name] = cpu_tensor
            tensor_meta[name] = {
                "shape": list(cpu_tensor.shape),
                "dtype": str(cpu_tensor.dtype).replace("torch.", ""),
                "bytes": nbytes,
            }

        self._evict_to_fit(total_bytes)
        slab_path = None
        if write_shm:
            self.shm_dir.mkdir(parents=True, exist_ok=True)
            slab_path = str(self.shm_dir / f"{key[:24]}.pt")
            torch.save(staged_tensors, slab_path)

        record = {
            "key": key,
            "tensors": staged_tensors,
            "meta": tensor_meta,
            "bytes": total_bytes,
            "slab_path": slab_path,
            "stage_ms": (time.perf_counter() - started) * 1000.0,
        }
        self._entries[key] = record
        self._entries.move_to_end(key)
        self._bytes += total_bytes
        self.staged_pushes += 1
        return {
            "key": key,
            "bytes": total_bytes,
            "stage_ms": record["stage_ms"],
            "slab_path": slab_path,
            "tensors": tensor_meta,
        }

    def fetch_to_device(self, key: str, device: str, stream=None) -> dict:
        """Asynchronously push staged host embeddings to a target denoising GPU."""
        import torch

        started = time.perf_counter()
        if key in self._entries:
            record = self._entries[key]
            self._entries.move_to_end(key)
            self.hits += 1
            source = record["tensors"]
        else:
            slab_file = self.shm_dir / f"{key[:24]}.pt"
            if not slab_file.is_file():
                self.misses += 1
                raise KeyError(f"Prompt embedding key not found in pool: {key}")
            self.hits += 1
            source = torch.load(str(slab_file), map_location="cpu", weights_only=True)

        target = torch.device(device)
        out = {}
        ctx = torch.cuda.stream(stream) if (stream is not None and target.type == "cuda") else _null_ctx()
        with ctx:
            for name, tensor in source.items():
                if tensor is None:
                    out[name] = None
                else:
                    out[name] = tensor.to(device=target, non_blocking=(target.type == "cuda" and tensor.is_pinned()))
        return {
            "tensors": out,
            "push_ms": (time.perf_counter() - started) * 1000.0,
            "hit": True,
        }

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "bytes": self._bytes,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "staged_pushes": self.staged_pushes,
        }

    def clear(self) -> None:
        for record in self._entries.values():
            slab_path = record.get("slab_path")
            if slab_path and os.path.isfile(slab_path):
                try:
                    os.unlink(slab_path)
                except OSError:
                    pass
        self._entries.clear()
        self._bytes = 0


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
