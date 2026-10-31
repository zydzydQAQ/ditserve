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

    def __init__(self, config=None, *, max_cache_entries: int | None = None, max_entries: int | None = None, max_bytes: int | None = None):
        cfg = config if isinstance(config, dict) else {}
        eff_entries = max_cache_entries if max_cache_entries is not None else max_entries
        self.shm_dir = Path(cfg.get("shm_dir", "/dev/shm/ditserve-hetero-embeds"))
        self.max_entries = int(eff_entries if eff_entries is not None else cfg.get("max_entries", 64))
        self.max_bytes = int(max_bytes if max_bytes is not None else cfg.get("max_bytes", 512 * (1 << 20)))
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

    def optimize_wan_cpu_encoder(self, wan_runtime, *, mode: str = "bf16_trimmed") -> dict:
        """Convert CPU-resident UMT5-XXL text encoder to BF16 or Dynamic INT8 Weight-Only.

        Modes:
          - ``"fp32_trimmed"``: keep FP32 weights, apply exact active-token sequence trimming.
          - ``"bf16_trimmed"``: convert UMT5-XXL CPU weights from FP32 (21.67 GB) to BF16 (10.83 GB)
            to halve CPU DRAM bandwidth, plus active-token sequence trimming.
          - ``"int8_dynamic_trimmed"``: apply ``torch.ao.quantization.quantize_dynamic`` INT8
            weight-only quantization to Linear layers on CPU, plus active-token sequence trimming.
        """
        import torch

        t0 = time.perf_counter()
        encoder = wan_runtime.pipe.text_encoder
        orig_bytes = sum(p.numel() * p.element_size() for p in encoder.parameters())
        if mode == "bf16_trimmed":
            encoder.to(dtype=torch.bfloat16)
        elif mode == "int8_dynamic_trimmed":
            encoder.to(dtype=torch.float32)
            wan_runtime.pipe.text_encoder = torch.ao.quantization.quantize_dynamic(
                encoder, {torch.nn.Linear}, dtype=torch.qint8
            )
            encoder = wan_runtime.pipe.text_encoder
        elif mode == "fp32_trimmed":
            encoder.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported CPU encoder optimization mode: {mode}")
        new_bytes = sum(p.numel() * p.element_size() for p in encoder.parameters())
        for mod in encoder.modules():
            if hasattr(mod, "_packed_params") and hasattr(mod._packed_params, "_weight_bias"):
                try:
                    w_q, b_q = mod._packed_params._weight_bias()
                    new_bytes += w_q.numel() * w_q.element_size()
                    if b_q is not None:
                        new_bytes += b_q.numel() * b_q.element_size()
                except Exception:
                    pass
        return {
            "mode": mode,
            "orig_gb": round(orig_bytes / (1 << 30), 3),
            "new_gb": round(new_bytes / (1 << 30), 3),
            "convert_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def _encode_single_wan_text_trimmed(self, wan_runtime, text: str, max_sequence_length: int = 512):
        """Encode a single Wan2.1 text string using Active-Token Sequence Trimming on UMT5-XXL.

        Because ``WanPipeline._get_t5_prompt_embeds`` masks out padding tokens (``mask == 0``)
        during attention and immediately slices ``u[:seq_len]`` and zero-pads back to ``512``,
        evaluating UMT5-XXL only on the active ``[1, seq_len]`` prefix (typically 10..32 tokens
        instead of 512) is mathematically exact (``cos_sim = 1.000000``) while eliminating >90%
        of redundant CPU transformer FLOPs and memory traffic.
        """
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
            hidden_f32 = hidden.to(dtype=torch.float32)
            valid = hidden_f32[0, :seq_len]
            if seq_len < max_sequence_length:
                pad = torch.zeros(
                    (max_sequence_length - seq_len, valid.shape[-1]),
                    dtype=torch.float32,
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

        norm_spec = normalize_spec(spec)
        t0 = time.perf_counter()
        pos_text = norm_spec["prompt"]
        neg_text = norm_spec["negative_prompt"]
        do_cfg = norm_spec["guidance_scale"] > 1

        pos_key = self.prompt_key("wan_pos", str(wan_runtime.model_path), pos_text, "", max_sequence_length, "float32", False)
        neg_key = self.prompt_key("wan_neg", str(wan_runtime.model_path), "", neg_text, max_sequence_length, "float32", False)

        pos_hit = use_cache and (pos_key in self._entries)
        if pos_hit:
            pos_emb = self.fetch_to_device(pos_key, "cpu")["tensors"]["embed"].clone()
        else:
            pos_emb, _ = self._encode_single_wan_text_trimmed(wan_runtime, pos_text, max_sequence_length)
            if use_cache:
                self.stage_embeddings(pos_key, {"embed": pos_emb}, pin_memory=pin_memory)

        neg_emb = None
        neg_hit = False
        if do_cfg:
            neg_hit = use_cache and (neg_key in self._entries)
            if neg_hit:
                neg_emb = self.fetch_to_device(neg_key, "cpu")["tensors"]["embed"].clone()
            else:
                neg_emb, _ = self._encode_single_wan_text_trimmed(wan_runtime, neg_text, max_sequence_length)
                if use_cache:
                    self.stage_embeddings(neg_key, {"embed": neg_emb}, pin_memory=pin_memory)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        receipt = {
            "pos_cache_hit": pos_hit,
            "neg_cache_hit": neg_hit,
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


from contextlib import nullcontext as _null_ctx
