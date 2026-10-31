"""Asymmetric per-head/per-token INT8 Ring KV quantization and double-buffered P2P overlap.

In 2D Hybrid USP (Ulysses × Ring), intra-node NVLink/PCIe handles the Ulysses
All-to-All while cross-socket / cross-node links carry the Ring P2P KV
exchange. Because Ring Attention only transmits K and V tensors of shape
``[2, B, H_local, S_ring, D]`` (with head dimension ``D = 128``), symmetric
per-vector INT8 quantization packs each 128-element FP16/BF16 vector (256 B)
into 128 INT8 values (128 B) + 1 FP16 scale (2 B) = 130 B (a 1.969x payload
reduction) inside a single contiguous uint8 buffer for ``batch_isend_irecv``.
"""

from __future__ import annotations

import importlib
from typing import Tuple


class AsymmetricRingQuantizer:
    """Per-token-head INT8 KV packet codec and step-aware anchor policy for Ring Attention."""

    def __init__(
        self,
        torch_module=None,
        *,
        warmup_steps: int = 2,
        tail_anchor_steps: int = 2,
        enabled: bool = True,
    ):
        self.torch = torch_module or importlib.import_module("torch")
        self.warmup_steps = max(0, int(warmup_steps))
        self.tail_anchor_steps = max(0, int(tail_anchor_steps))
        self.enabled = bool(enabled)
        self.current_step_quantize: bool = bool(enabled)
        self.stats = {
            "quantized_packets": 0,
            "fp16_anchor_packets": 0,
            "orig_bytes": 0,
            "wire_bytes": 0,
        }

    def set_step(self, step_index: int, total_steps: int) -> bool:
        """Update current_step_quantize according to step-aware anchor schedule."""
        self.current_step_quantize = self.should_quantize(step_index, total_steps)
        return self.current_step_quantize

    def should_quantize(self, step_index: int, total_steps: int) -> bool:
        """Return True if step_index should use INT8 Ring KV compression."""
        if not self.enabled:
            return False
        if step_index < self.warmup_steps:
            return False
        if step_index >= max(0, total_steps - self.tail_anchor_steps):
            return False
        return True

    def quantize_kv(self, kv: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Quantize ``[..., D]`` FP16/BF16 KV tensor into ``(q_int8, scale_fp16)``."""
        torch = self.torch
        orig_dtype = kv.dtype
        # Compute per-vector absmax along head dim D
        abs_max = kv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (abs_max / 127.0).to(torch.float16)
        scaled = kv.to(torch.float32) / scale.to(torch.float32)
        q_int8 = torch.clamp(torch.round(scaled), -127.0, 127.0).to(torch.int8)
        return q_int8, scale

    def dequantize_kv(
        self,
        q_int8: "torch.Tensor",
        scale: "torch.Tensor",
        target_dtype: "torch.dtype | None" = None,
    ) -> "torch.Tensor":
        """Dequantize ``(q_int8, scale_fp16)`` back to ``target_dtype`` (defaults to FP16)."""
        torch = self.torch
        dtype = target_dtype or torch.float16
        return (q_int8.to(torch.float32) * scale.to(torch.float32)).to(dtype)

    def pack_packet(self, kv: "torch.Tensor") -> "torch.Tensor":
        """Pack ``[..., D]`` KV tensor into a single contiguous ``uint8`` buffer ``[..., D + 2]``.

        Layout per row of length D:
        - First ``D`` bytes: ``q_int8`` viewed as ``uint8``
        - Last ``2`` bytes: ``scale`` (``float16``) viewed as ``uint8``
        """
        torch = self.torch
        if kv.shape[-1] < 4:
            raise ValueError("head dimension D must be at least 4 for INT8 packet packing")
        q_int8, scale = self.quantize_kv(kv)
        q_u8 = q_int8.contiguous().view(torch.uint8)
        s_u8 = scale.contiguous().view(torch.uint8)
        packed = torch.cat([q_u8, s_u8], dim=-1).contiguous()
        self.stats["quantized_packets"] += 1
        self.stats["orig_bytes"] += int(kv.numel() * kv.element_size())
        self.stats["wire_bytes"] += int(packed.numel() * packed.element_size())
        return packed

    def unpack_packet(
        self,
        packed: "torch.Tensor",
        target_dtype: "torch.dtype | None" = None,
    ) -> "torch.Tensor":
        """Unpack a ``[..., D + 2]`` ``uint8`` packet back into ``[..., D]`` in ``target_dtype``."""
        torch = self.torch
        d_plus_2 = packed.shape[-1]
        if d_plus_2 < 6:
            raise ValueError("invalid packed Ring KV packet width")
        d = d_plus_2 - 2
        q_u8 = packed[..., :d].contiguous()
        s_u8 = packed[..., d:].contiguous()
        q_int8 = q_u8.view(torch.int8)
        scale = s_u8.view(torch.float16)
        return self.dequantize_kv(q_int8, scale, target_dtype=target_dtype)

    def ring_p2p_exchange(
        self,
        dist_module,
        kv_send: "torch.Tensor",
        send_peer: int,
        recv_peer: int,
        group=None,
        *,
        quantize: bool = True,
    ) -> "torch.Tensor":
        """Exchange KV shard with Ring neighbors using INT8 packet compression when ``quantize=True``."""
        torch = self.torch
        if not quantize:
            recv_buf = torch.empty_like(kv_send)
            ops = [
                dist_module.P2POp(dist_module.isend, kv_send.contiguous(), send_peer, group=group),
                dist_module.P2POp(dist_module.irecv, recv_buf, recv_peer, group=group),
            ]
            reqs = dist_module.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
            self.stats["fp16_anchor_packets"] += 1
            nbytes = int(kv_send.numel() * kv_send.element_size())
            self.stats["orig_bytes"] += nbytes
            self.stats["wire_bytes"] += nbytes
            return recv_buf

        packed_send = self.pack_packet(kv_send)
        packed_recv = torch.empty_like(packed_send)
        ops = [
            dist_module.P2POp(dist_module.isend, packed_send, send_peer, group=group),
            dist_module.P2POp(dist_module.irecv, packed_recv, recv_peer, group=group),
        ]
        reqs = dist_module.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        return self.unpack_packet(packed_recv, target_dtype=kv_send.dtype)

    def install_xfuser_hook(self, min_bytes: int = 2 * 1024 * 1024) -> None:
        """Embed this AsymmetricRingQuantizer into xfuser's USP Ring P2P kernel."""
        from xfuser.core.distributed.fp8_comms import register_int8_ring_quantizer

        register_int8_ring_quantizer(self, min_bytes=min_bytes)

    def uninstall_xfuser_hook(self) -> None:
        """Remove this AsymmetricRingQuantizer from xfuser's USP Ring P2P kernel."""
        from xfuser.core.distributed.fp8_comms import register_int8_ring_quantizer

        register_int8_ring_quantizer(None)

    def summary(self) -> dict:
        orig = max(1, self.stats["orig_bytes"])
        wire = max(1, self.stats["wire_bytes"])
        return {
            **self.stats,
            "compression_ratio": round(orig / wire, 4),
            "bandwidth_saving_pct": round(100.0 * (1.0 - wire / orig), 2),
        }
