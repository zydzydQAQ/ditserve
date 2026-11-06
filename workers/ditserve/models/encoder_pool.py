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
import pickle
import threading
import time


# A slab another worker unlinks, replaces or (never, given os.replace) half-writes reads as a miss.
_SLAB_READ_ERRORS = (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError)
DEFAULT_SHM_DIR = "/dev/shm/ditserve-hetero-embeds"


def resolve_shm_dir(config: dict | None = None, *, work_dir: str | Path | None = None) -> tuple[Path, bool]:
    """Resolve the `/dev/shm` slab directory, scoping per matrix cell when `work_dir` is set."""
    cfg = config if isinstance(config, dict) else {}
    raw = cfg.get("shm_dir")
    work = work_dir if work_dir is not None else cfg.get("work_dir")
    is_default = raw is None or str(raw) in ("", "auto", DEFAULT_SHM_DIR)
    base = Path(DEFAULT_SHM_DIR) if is_default else Path(str(raw))
    per_cell = bool(cfg.get("per_cell_isolation", is_default))
    if per_cell and work:
        wd = Path(str(work)).resolve()
        cell_scope = wd.parent.parent if wd.parent.name == "artifacts" else wd
        scope_id = hashlib.sha256(str(cell_scope).encode("utf-8")).hexdigest()[:16]
        return base / f"cell-{scope_id}", True
    return base, False


class ProactiveEncoderPool:
    """Centralized or cross-worker shared text encoder pool with proactive staging."""

    def __init__(
        self,
        config=None,
        *,
        max_cache_entries: int | None = None,
        max_entries: int | None = None,
        max_bytes: int | None = None,
        work_dir: str | Path | None = None,
    ):
        cfg = config if isinstance(config, dict) else {}
        eff_entries = max_cache_entries if max_cache_entries is not None else max_entries
        self.enabled = bool(cfg.get("enabled", True))
        self.shm_dir, self.cell_scoped = resolve_shm_dir(cfg, work_dir=work_dir)
        self.max_entries = int(eff_entries if eff_entries is not None else cfg.get("max_entries", 64))
        self.max_bytes = int(max_bytes if max_bytes is not None else cfg.get("max_bytes", 512 * (1 << 20)))
        self.write_shm = bool(cfg.get("write_shm", False))
        self.pin_memory = bool(cfg.get("pin_memory", True))
        self.trimmed = bool(cfg.get("trimmed", True))
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()  # encode thread and request thread share one pool
        self.hits = 0
        self.misses = 0
        self.staged_pushes = 0
        self.shm_adoptions = 0
        self.slab_writes = 0
        self.slab_unlinks = 0

    def _slab_file(self, key: str) -> Path:
        return self.shm_dir / f"{key[:24]}.pt"

    def has_key(self, key: str) -> bool:
        """Return True if prompt embedding key is in-memory or staged in /dev/shm.

        Advisory only: /dev/shm slabs are shared by every worker and another one may
        remove the slab right after this returns. Callers that must later fetch the
        key use ``ensure_resident`` instead.
        """
        if key in self._entries:
            return True
        return self._slab_file(key).is_file()

    def ensure_resident(self, key: str) -> bool:
        """Make ``key`` fetchable from this pool's own memory; False is a miss.

        An in-memory entry is a hit. Otherwise a /dev/shm slab staged by any worker is
        read now into this pool's LRU, so a later ``fetch_to_device`` no longer depends
        on the shared file; a slab that vanished or cannot be read is a miss and the
        caller encodes. The adopted entry does not own the slab and never unlinks it.
        """
        with self._lock:
            if key in self._entries:
                return True
        import torch

        try:
            source = torch.load(str(self._slab_file(key)), map_location="cpu", weights_only=True)
        except _SLAB_READ_ERRORS:
            return False
        if not isinstance(source, dict):
            return False
        started = time.perf_counter()
        staged_tensors, tensor_meta, total_bytes = self._host_copies(source, self.pin_memory)
        with self._lock:
            if key in self._entries:
                return True
            self._admit(key, {
                "key": key,
                "tensors": staged_tensors,
                "meta": tensor_meta,
                "bytes": total_bytes,
                "slab_path": None,
                "slab_id": None,
                "adopted_shm": True,
                "stage_ms": (time.perf_counter() - started) * 1000.0,
            })
            self.shm_adoptions += 1
        return True

    @staticmethod
    def prompt_key(
        modality: str,
        model_id: str,
        prompt: str,
        negative_prompt: str = "",
        max_sequence_length: int = 256,
        dtype: str = "bfloat16",
        do_cfg: bool = True,
        *,
        text_encoder_device: str = "",
        text_encoder_dtype: str = "",
        revision: str = "",
    ) -> str:
        body = {
            "modality": modality,
            "model_id": str(model_id),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "max_sequence_length": int(max_sequence_length),
            "dtype": dtype,
            "do_cfg": do_cfg,
        }
        if text_encoder_device:
            body["text_encoder_device"] = str(text_encoder_device)
        if text_encoder_dtype:
            body["text_encoder_dtype"] = str(text_encoder_dtype)
        if revision:
            body["revision"] = str(revision)
        payload = json.dumps(
            body,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _file_identity(stat_result) -> tuple:
        return (stat_result.st_dev, stat_result.st_ino, stat_result.st_size, stat_result.st_mtime_ns)

    @classmethod
    def _unlink_own_slab(cls, record: dict) -> bool:
        """Remove the slab this pool wrote, unless another worker has since replaced it.

        Slabs are shared by every worker on the node; one this pool only read (or one
        another worker re-staged over its own) belongs to someone else and is kept.
        """
        slab_path, slab_id = record.get("slab_path"), record.get("slab_id")
        if not slab_path or slab_id is None:
            return False
        try:
            if cls._file_identity(os.stat(slab_path)) == slab_id:
                os.unlink(slab_path)
                return True
        except OSError:
            pass
        return False

    def _evict_to_fit(self, needed_bytes: int) -> None:
        while self._entries and (
            len(self._entries) >= self.max_entries or self._bytes + needed_bytes > self.max_bytes
        ):
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted["bytes"]
            if self._unlink_own_slab(evicted):
                self.slab_unlinks += 1

    def _admit(self, key: str, record: dict) -> None:
        """Insert ``record`` as the most recent entry; the caller holds ``_lock``."""
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous["bytes"]
            if record.get("slab_id") is None and previous.get("slab_id") is not None:
                record["slab_path"], record["slab_id"] = previous["slab_path"], previous["slab_id"]
        self._evict_to_fit(record["bytes"])
        self._entries[key] = record
        self._bytes += record["bytes"]

    @staticmethod
    def _host_copies(tensors: dict, pin_memory: bool):
        import torch

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
        return staged_tensors, tensor_meta, total_bytes

    def stage_embeddings(self, key: str, tensors: dict, *, pin_memory: bool | None = None,
                         write_shm: bool | None = None) -> dict:
        """Stage prompt embedding tensors in pinned host memory (and optional /dev/shm slab)."""
        import torch

        if pin_memory is None:
            pin_memory = self.pin_memory
        if write_shm is None:
            write_shm = self.write_shm
        started = time.perf_counter()
        staged_tensors, tensor_meta, total_bytes = self._host_copies(tensors, pin_memory)

        slab_path = slab_id = None
        if write_shm:
            self.shm_dir.mkdir(parents=True, exist_ok=True)
            slab_path = str(self._slab_file(key))
            tmp_path = f"{slab_path}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
            torch.save(staged_tensors, tmp_path)
            slab_id = self._file_identity(os.stat(tmp_path))  # os.replace keeps inode and mtime
            os.replace(tmp_path, slab_path)

        record = {
            "key": key,
            "tensors": staged_tensors,
            "meta": tensor_meta,
            "bytes": total_bytes,
            "slab_path": slab_path,
            "slab_id": slab_id,
            "stage_ms": (time.perf_counter() - started) * 1000.0,
        }
        with self._lock:
            if slab_id is not None:
                self.slab_writes += 1
            self._admit(key, record)
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
        with self._lock:
            record = self._entries.get(key)
            if record is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                source = record["tensors"]
        if record is None:
            # Cross-worker read without adoption; a slab removed under us is a miss.
            try:
                source = torch.load(str(self._slab_file(key)), map_location="cpu", weights_only=True)
            except _SLAB_READ_ERRORS as error:
                self.misses += 1
                raise KeyError(f"Prompt embedding key not found in pool: {key}") from error
            self.hits += 1

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

    def optimize_wan_cpu_encoder(self, wan_runtime, *, mode: str = "bf16_trimmed") -> dict:
        """Check the loaded encoder before using the optional trimmed path.

        BF16 workers use ``bf16_trimmed``; explicit FP32 comparison workers use
        ``fp32_trimmed``. Changing precision or quantizing a resident encoder is
        unsupported: its evidence and negative-cache recipe are already bound.
        """
        import torch

        t0 = time.perf_counter()
        encoder = wan_runtime.pipe.text_encoder
        # A resident worker's precision and cache recipe are fixed at load time.
        # Changing only weights would mislabel evidence and reuse stale embeddings.
        expected_mode = {"bfloat16": "bf16_trimmed", "float32": "fp32_trimmed"}[wan_runtime.text_encoder_dtype]
        if mode != expected_mode:
            raise ValueError("encoder optimization must match loaded text_encoder_dtype; "
                             "construct a fresh worker for a different numerical mode")
        if encoder.dtype != getattr(torch, wan_runtime.text_encoder_dtype):
            raise ValueError("encoder weights differ from loaded text_encoder_dtype")
        size = sum(p.numel() * p.element_size() for p in encoder.parameters())
        self.trimmed = True
        return {
            "mode": mode,
            "trimmed": True,
            "orig_gb": round(size / (1 << 30), 3),
            "new_gb": round(size / (1 << 30), 3),
            "convert_ms": 0.0,
            "check_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def _encode_single_wan_text_trimmed(self, wan_runtime, text: str, max_sequence_length: int = 512):
        """Encode a single Wan2.1 text string using Active-Token Sequence Trimming on UMT5-XXL.

        Wan masks padding during attention and pads the active prefix back to the
        full length. Trimming reduces work, but different matrix shapes can change
        floating-point rounding; it needs its own numerical/performance validation.
        """
        override = getattr(wan_runtime, "_encode_single_wan_text_trimmed", None)
        if callable(override):
            return override(text, max_sequence_length)
        import torch
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        pipe = wan_runtime.pipe
        cleaned = prompt_clean(text)
        text_inputs = pipe.tokenizer(
            [cleaned],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids
        mask = text_inputs.attention_mask
        seq_len = max(1, int(mask.gt(0).sum(dim=1).max().item()))
        trimmed_ids = input_ids[:, :seq_len]
        trimmed_mask = mask[:, :seq_len]
        with torch.inference_mode():
            hidden = pipe.text_encoder(trimmed_ids, trimmed_mask).last_hidden_state
            hidden = hidden.to(dtype=getattr(torch, wan_runtime.text_encoder_dtype))
            valid = hidden[0, :seq_len]
            if seq_len < max_sequence_length:
                pad = torch.zeros(
                    (max_sequence_length - seq_len, valid.shape[-1]),
                    dtype=hidden.dtype,
                    device="cpu",
                )
                out = torch.cat([valid, pad], dim=0).unsqueeze(0).contiguous()
            else:
                out = valid.unsqueeze(0).contiguous()
        return out, seq_len

    def encode_wan_prompt_fast(self, wan_runtime, spec: dict, *, max_sequence_length: int = 512,
                               pin_memory: bool = True, use_cache: bool = True):
        """Fast Wan2.1 CPU prompt encoding with Active-Token Trimming + Pinned LRU Prompt/Neg-Prompt Cache."""
        from .wan.runtime import EncodedPrompt, normalize_spec

        dtype = wan_runtime.text_encoder_dtype
        text_encoder = getattr(getattr(wan_runtime, "pipe", None), "text_encoder", None)
        if text_encoder is not None and str(text_encoder.dtype) != "torch." + dtype:
            raise ValueError("encoder weights differ from loaded text_encoder_dtype")
        norm_spec = normalize_spec(spec)
        t0 = time.perf_counter()
        pos_text = norm_spec["prompt"]
        neg_text = norm_spec["negative_prompt"]
        do_cfg = norm_spec["guidance_scale"] > 1

        model_id = json.dumps([str(wan_runtime.model_path), wan_runtime.model_revision])
        pos_key = self.prompt_key("wan_pos", model_id, pos_text, "", max_sequence_length, dtype, False)
        neg_key = self.prompt_key("wan_neg", model_id, "", neg_text, max_sequence_length, dtype, False)

        pos_hit = use_cache and self.ensure_resident(pos_key)
        if pos_hit:
            pos_emb = self.fetch_to_device(pos_key, "cpu")["tensors"]["embed"].clone()
        else:
            pos_emb, _ = self._encode_single_wan_text_trimmed(wan_runtime, pos_text, max_sequence_length)
            if use_cache:
                self.stage_embeddings(
                    pos_key,
                    {"embed": pos_emb.clone()},
                    pin_memory=pin_memory,
                    write_shm=self.write_shm,
                )

        neg_emb = None
        neg_hit = False
        if do_cfg:
            neg_hit = use_cache and self.ensure_resident(neg_key)
            if neg_hit:
                neg_emb = self.fetch_to_device(neg_key, "cpu")["tensors"]["embed"].clone()
            else:
                neg_emb, _ = self._encode_single_wan_text_trimmed(wan_runtime, neg_text, max_sequence_length)
                if use_cache:
                    self.stage_embeddings(
                        neg_key,
                        {"embed": neg_emb.clone()},
                        pin_memory=pin_memory,
                        write_shm=self.write_shm,
                    )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        receipt = {
            "outcome": "bypass" if not do_cfg else ("hit" if neg_hit else "miss"),
            "pos_cache_hit": pos_hit,
            "neg_cache_hit": neg_hit,
            "text_encoder_dtype": dtype,
            "cpu_prompt_encode_ms": round(elapsed_ms, 3),
        }
        return EncodedPrompt(
            spec=norm_spec,
            prompt_embeds=pos_emb,
            negative_prompt_embeds=neg_emb,
            cpu_prompt_encode_ms=elapsed_ms,
            owner=wan_runtime._owner,
            negative_prompt_cache=receipt,
        )

    def summary(self) -> dict:
        return self.stats()

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "bytes": self._bytes,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "staged_pushes": self.staged_pushes,
            "shm_adoptions": self.shm_adoptions,
            "slab_writes": self.slab_writes,
            "slab_unlinks": self.slab_unlinks,
            "shm_dir": str(self.shm_dir),
            "cell_scoped": self.cell_scoped,
        }

    def clear(self) -> None:
        with self._lock:
            for record in self._entries.values():
                if self._unlink_own_slab(record):
                    self.slab_unlinks += 1
            self._entries.clear()
            self._bytes = 0

    def close(self) -> None:
        self.clear()
        if self.cell_scoped and self.shm_dir.is_dir():
            try:
                self.shm_dir.rmdir()
            except OSError:
                pass


from contextlib import nullcontext as _null_ctx
