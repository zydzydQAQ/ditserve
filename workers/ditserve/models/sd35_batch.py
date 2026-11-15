"""SD3.5-Medium advanced batching:
1. Bubble-Free Pipeline (FlashPS-style side-stream prompt prefetch + async 2D-VAE decode)
2. Per-request step-wise continuous batching (`prepare` -> `advance_batch` -> `finish` with vector timesteps)
3. Variable-resolution Token Sequence Packing (`cu_seqlens` packed MM-DiT forward without padding waste)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import copy
from dataclasses import dataclass
import inspect
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _build_varlen_flash_adapter() -> Tuple[Optional[Callable[..., Any]], str]:
    """Resolve a fused variable-length FlashAttention kernel across flash_attn, vLLM, and PyTorch >=2.8."""
    # 1. Upstream Dao-AILab flash_attn / FA3 flash_attn_interface
    for mod_name in ("flash_attn", "flash_attn_interface"):
        try:
            mod = __import__(mod_name, fromlist=["flash_attn_varlen_func"])
            fn = getattr(mod, "flash_attn_varlen_func", None)
            if callable(fn):
                sig_params = set(inspect.signature(fn).parameters)

                def _call_fa(
                    q: Any,
                    k: Any,
                    v: Any,
                    cu_q: Any,
                    cu_k: Any,
                    max_q: int,
                    max_k: int,
                    dropout_p: float = 0.0,
                    causal: bool = False,
                    _fn: Callable[..., Any] = fn,
                    _params: set[str] = sig_params,
                ) -> Any:
                    kw: Dict[str, Any] = {"causal": bool(causal)}
                    if "dropout_p" in _params:
                        kw["dropout_p"] = float(dropout_p)
                    out = _fn(q, k, v, cu_q, cu_k, int(max_q), int(max_k), **kw)
                    return out[0] if isinstance(out, tuple) else out

                return _call_fa, mod_name
        except Exception:  # pragma: no cover
            continue

    # 2. vLLM's bundled SM80+ FlashAttention2 varlen kernel (note swapped max_seqlen/cu_seqlens order)
    for mod_name in ("vllm.vllm_flash_attn.flash_attn_interface", "vllm.vllm_flash_attn"):
        try:
            mod = __import__(mod_name, fromlist=["flash_attn_varlen_func"])
            fn = getattr(mod, "flash_attn_varlen_func", None)
            if callable(fn):
                sig_params = set(inspect.signature(fn).parameters)

                def _call_vllm_fa(
                    q: Any,
                    k: Any,
                    v: Any,
                    cu_q: Any,
                    cu_k: Any,
                    max_q: int,
                    max_k: int,
                    dropout_p: float = 0.0,
                    causal: bool = False,
                    _fn: Callable[..., Any] = fn,
                    _params: set[str] = sig_params,
                ) -> Any:
                    kw: Dict[str, Any] = {
                        "q": q,
                        "k": k,
                        "v": v,
                        "cu_seqlens_q": cu_q,
                        "cu_seqlens_k": cu_k,
                        "max_seqlen_q": int(max_q),
                        "max_seqlen_k": int(max_k),
                        "causal": bool(causal),
                    }
                    if "dropout_p" in _params:
                        kw["dropout_p"] = float(dropout_p)
                    out = _fn(**kw)
                    return out[0] if isinstance(out, tuple) else out

                return _call_vllm_fa, "vllm_flash_attn"
        except Exception:  # pragma: no cover
            continue

    # 3. PyTorch native varlen FlashAttention (torch.nn.attention.varlen.varlen_attn, PyTorch >= 2.8)
    try:
        from torch.nn.attention.varlen import varlen_attn as _torch_varlen_attn  # type: ignore

        if callable(_torch_varlen_attn):
            def _call_torch_varlen(
                q: Any,
                k: Any,
                v: Any,
                cu_q: Any,
                cu_k: Any,
                max_q: int,
                max_k: int,
                dropout_p: float = 0.0,
                causal: bool = False,
                _fn: Callable[..., Any] = _torch_varlen_attn,
            ) -> Any:
                out = _fn(q, k, v, cu_q, cu_k, int(max_q), int(max_k), is_causal=bool(causal))
                return out[0] if isinstance(out, tuple) else out

            return _call_torch_varlen, "torch.nn.attention.varlen"
    except Exception:  # pragma: no cover
        pass

    return None, "none"


_FLASH_ATTN_VARLEN_FUNC, _FLASH_ATTN_VARLEN_PROVIDER = _build_varlen_flash_adapter()


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Compute SD3 FlowMatch dynamic resolution shift ``mu`` from patch token count."""
    m = (max_shift - base_shift) / max(1, (max_seq_len - base_seq_len))
    b = base_shift - m * base_seq_len
    return float(image_seq_len * m + b)


def build_cu_seqlens(lengths: Sequence[int]) -> List[int]:
    """Build cumulative sequence lengths [0, L_0, L_0 + L_1, ...] for packed token streams."""
    cu = [0]
    total = 0
    for length in lengths:
        val = int(length)
        if val <= 0:
            raise ValueError("sequence length must be positive")
        total += val
        cu.append(total)
    return cu


def packing_stats(heights: Sequence[int], widths: Sequence[int], *, vae_scale_factor: int = 8, patch_size: int = 2) -> Dict[str, Any]:
    """Compute token counts, cu_seqlens, and padding savings for a (possibly heterogeneous) resolution batch."""
    if len(heights) != len(widths) or not heights:
        raise ValueError("heights and widths must be non-empty sequences of equal length")
    stride = max(1, int(vae_scale_factor) * int(patch_size))
    token_counts = []
    for h, w in zip(heights, widths):
        ih, iw = int(h), int(w)
        if ih <= 0 or iw <= 0 or ih % stride or iw % stride:
            raise ValueError(f"resolution ({ih}, {iw}) must be positive and divisible by {stride}")
        token_counts.append((ih // stride) * (iw // stride))
    cu_seqlens = build_cu_seqlens(token_counts)
    packed_tokens = cu_seqlens[-1]
    max_tokens = max(token_counts)
    padded_tokens = len(token_counts) * max_tokens
    saved_tokens = padded_tokens - packed_tokens
    return {
        "batch_size": len(token_counts),
        "heterogeneous_resolutions": len(set(zip(heights, widths))) > 1,
        "resolutions": [[int(h), int(w)] for h, w in zip(heights, widths)],
        "image_token_counts": token_counts,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_tokens,
        "packed_image_tokens": packed_tokens,
        "padded_image_tokens": padded_tokens,
        "padding_saved_tokens": saved_tokens,
        "padding_waste_ratio": round(saved_tokens / padded_tokens, 4) if padded_tokens > 0 else 0.0,
        "packing_efficiency": round(packed_tokens / padded_tokens, 4) if padded_tokens > 0 else 1.0,
    }


def _unwrap_transformer(transformer: Any) -> Any:
    """Return the underlying diffusers SD3Transformer2DModel if wrapped by xFuser."""
    return getattr(transformer, "module", transformer)


def _unwrap_attn(attn: Any) -> Any:
    """Return the underlying diffusers Attention module if wrapped by xFuser."""
    return getattr(attn, "module", attn)


def _can_use_flash_varlen(torch_mod: Any, sample_tensor: Any) -> bool:
    """Return True when fused ``flash_attn_varlen_func`` can execute on ``sample_tensor``."""
    if _FLASH_ATTN_VARLEN_FUNC is None:
        return False
    dev = getattr(sample_tensor, "device", None)
    if getattr(dev, "type", "") != "cuda":
        return False
    allowed_dtypes = tuple(
        dt for dt in (getattr(torch_mod, "float16", None), getattr(torch_mod, "bfloat16", None)) if dt is not None
    )
    if sample_tensor.dtype not in allowed_dtypes:
        return False
    cuda_backends = getattr(getattr(torch_mod, "backends", None), "cuda", None)
    if cuda_backends is not None and callable(getattr(cuda_backends, "flash_sdp_enabled", None)):
        if not cuda_backends.flash_sdp_enabled():
            return False
    return True


@dataclass
class _PackedBatchPlan:
    """Precomputed per-step metadata for packed heterogeneous MM-DiT execution."""

    cu_img: List[int]
    cu_txt: List[int]
    img_counts: List[int]
    txt_counts: List[int]
    img_seg_ids: Any
    txt_seg_ids: Any
    joint_groups: List[Tuple[Tuple[int, int], List[int]]]
    img_only_groups: List[Tuple[int, List[int]]]
    use_fa_varlen: bool
    cu_img_i32: Optional[Any] = None
    max_seqlen_img: int = 0
    cu_joint_i32: Optional[Any] = None
    max_seqlen_joint: int = 0
    total_joint: int = 0
    img_in_joint_idx: Optional[Any] = None
    txt_in_joint_idx: Optional[Any] = None


def _build_packed_batch_plan(
    torch_mod: Any,
    img_counts: List[int],
    txt_counts: List[int],
    sample_tensor: Any,
) -> _PackedBatchPlan:
    device = sample_tensor.device
    cu_img = build_cu_seqlens(img_counts)
    cu_txt = build_cu_seqlens(txt_counts)
    num_segments = len(img_counts)

    img_seg_cpu = torch_mod.cat(
        [torch_mod.full((n_m,), m, dtype=torch_mod.long) for m, n_m in enumerate(img_counts)],
        dim=0,
    )
    txt_seg_cpu = torch_mod.cat(
        [torch_mod.full((t_m,), m, dtype=torch_mod.long) for m, t_m in enumerate(txt_counts)],
        dim=0,
    )
    non_blocking = getattr(device, "type", "") == "cuda"
    img_seg_ids = img_seg_cpu.to(device=device, non_blocking=non_blocking)
    txt_seg_ids = txt_seg_cpu.to(device=device, non_blocking=non_blocking)

    j_groups: Dict[Tuple[int, int], List[int]] = {}
    i_groups: Dict[int, List[int]] = {}
    for m in range(num_segments):
        n_m = img_counts[m]
        t_m = txt_counts[m]
        j_groups.setdefault((n_m, t_m), []).append(m)
        i_groups.setdefault(n_m, []).append(m)

    use_fa = _can_use_flash_varlen(torch_mod, sample_tensor)
    if not use_fa:
        return _PackedBatchPlan(
            cu_img=cu_img,
            cu_txt=cu_txt,
            img_counts=img_counts,
            txt_counts=txt_counts,
            img_seg_ids=img_seg_ids,
            txt_seg_ids=txt_seg_ids,
            joint_groups=list(j_groups.items()),
            img_only_groups=list(i_groups.items()),
            use_fa_varlen=False,
        )

    cu_joint = [cu_img[m] + cu_txt[m] for m in range(num_segments + 1)]
    max_seqlen_img = max(img_counts)
    max_seqlen_joint = max(img_counts[m] + txt_counts[m] for m in range(num_segments))
    total_joint = cu_joint[-1]

    img_idx_cpu = torch_mod.cat(
        [torch_mod.arange(cu_joint[m], cu_joint[m] + img_counts[m], dtype=torch_mod.long) for m in range(num_segments)],
        dim=0,
    )
    txt_idx_cpu = torch_mod.cat(
        [
            torch_mod.arange(
                cu_joint[m] + img_counts[m],
                cu_joint[m + 1],
                dtype=torch_mod.long,
            )
            for m in range(num_segments)
        ],
        dim=0,
    )
    cu_img_i32 = torch_mod.tensor(cu_img, dtype=torch_mod.int32, device=device)
    cu_joint_i32 = torch_mod.tensor(cu_joint, dtype=torch_mod.int32, device=device)
    img_in_joint_idx = img_idx_cpu.to(device=device, non_blocking=non_blocking)
    txt_in_joint_idx = txt_idx_cpu.to(device=device, non_blocking=non_blocking)

    return _PackedBatchPlan(
        cu_img=cu_img,
        cu_txt=cu_txt,
        img_counts=img_counts,
        txt_counts=txt_counts,
        img_seg_ids=img_seg_ids,
        txt_seg_ids=txt_seg_ids,
        joint_groups=list(j_groups.items()),
        img_only_groups=list(i_groups.items()),
        use_fa_varlen=True,
        cu_img_i32=cu_img_i32,
        max_seqlen_img=max_seqlen_img,
        cu_joint_i32=cu_joint_i32,
        max_seqlen_joint=max_seqlen_joint,
        total_joint=total_joint,
        img_in_joint_idx=img_in_joint_idx,
        txt_in_joint_idx=txt_in_joint_idx,
    )


def _expand_per_segment(
    torch_mod: Any,
    values: Any,
    counts_tensor: Any = None,
    total_tokens: Optional[int] = None,
    *,
    seg_ids: Any = None,
) -> Any:
    """Expand [M, D] per-item modulation vectors to [1, sum(counts), D] along the packed token dimension."""
    if seg_ids is not None:
        return values.index_select(0, seg_ids).unsqueeze(0)
    if total_tokens is not None:
        return torch_mod.repeat_interleave(values, counts_tensor, dim=0, output_size=int(total_tokens)).unsqueeze(0)
    return torch_mod.repeat_interleave(values, counts_tensor, dim=0).unsqueeze(0)


def _packed_adaln_zero(
    torch_mod: Any,
    norm_layer: Any,
    packed_x: Any,
    temb_vec: Any,
    counts_tensor: Any = None,
    *,
    seg_ids: Any = None,
):
    """Apply AdaLayerNormZero or SD35AdaLayerNormZeroX to a 1D packed token tensor [1, N_total, D]."""
    total_tokens = int(packed_x.shape[1])
    dim = int(packed_x.shape[-1])
    emb = norm_layer.linear(norm_layer.silu(temb_vec))
    emb_tok = _expand_per_segment(torch_mod, emb, counts_tensor, total_tokens, seg_ids=seg_ids)
    norm_x = norm_layer.norm(packed_x)
    if emb_tok.shape[-1] == 9 * dim:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_msa2,
            scale_msa2,
            gate_msa2,
        ) = emb_tok.chunk(9, dim=-1)
        normed = norm_x * (1.0 + scale_msa) + shift_msa
        normed2 = norm_x * (1.0 + scale_msa2) + shift_msa2
        return normed, gate_msa, shift_mlp, scale_mlp, gate_mlp, normed2, gate_msa2
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb_tok.chunk(6, dim=-1)
    normed = norm_x * (1.0 + scale_msa) + shift_msa
    return (
        normed,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        None,
        None,
    )


def _packed_adaln_continuous(
    torch_mod: Any,
    norm_layer: Any,
    packed_x: Any,
    temb_vec: Any,
    counts_tensor: Any = None,
    *,
    seg_ids: Any = None,
):
    """Apply AdaLayerNormContinuous to a 1D packed token tensor [1, N_total, D] with per-segment temb [M, D]."""
    total_tokens = int(packed_x.shape[1])
    emb = norm_layer.linear(norm_layer.silu(temb_vec))
    emb_tok = _expand_per_segment(torch_mod, emb, counts_tensor, total_tokens, seg_ids=seg_ids)
    scale, shift = emb_tok.chunk(2, dim=-1)
    return norm_layer.norm(packed_x) * (1.0 + scale) + shift


def _packed_joint_attention(
    torch_mod: Any,
    attn_layer: Any,
    norm_hidden_states: Any,
    norm_encoder_hidden_states: Optional[Any],
    cu_img: List[int],
    cu_txt: List[int],
    *,
    plan: Optional[_PackedBatchPlan] = None,
) -> Tuple[Any, Optional[Any]]:
    """Execute MM-DiT Joint Attention on packed [1, N_total, D] image tokens and [1, L_total, D] text tokens.

    All QKV and output linear projections execute as single packed GEMMs without padding tokens.
    On CUDA FP16/BF16 with FlashAttention enabled, variable-length segments execute in a single
    fused ``flash_attn_varlen_func`` kernel call across all resolutions without per-resolution Python loops.
    """
    attn = _unwrap_attn(attn_layer)
    F = torch_mod.nn.functional

    # 1. Packed single-GEMM QKV projections on image tokens [1, N_total, D]
    q_img = attn.to_q(norm_hidden_states)
    k_img = attn.to_k(norm_hidden_states)
    v_img = attn.to_v(norm_hidden_states)

    inner_dim = k_img.shape[-1]
    heads = int(attn.heads)
    head_dim = inner_dim // heads

    q_img = q_img.view(1, -1, heads, head_dim)
    k_img = k_img.view(1, -1, heads, head_dim)
    v_img = v_img.view(1, -1, heads, head_dim)

    if getattr(attn, "norm_q", None) is not None:
        q_img = attn.norm_q(q_img)
    if getattr(attn, "norm_k", None) is not None:
        k_img = attn.norm_k(k_img)

    # 2. Packed single-GEMM QKV projections on text context tokens [1, L_total, D]
    if norm_encoder_hidden_states is not None:
        q_txt = attn.add_q_proj(norm_encoder_hidden_states).view(1, -1, heads, head_dim)
        k_txt = attn.add_k_proj(norm_encoder_hidden_states).view(1, -1, heads, head_dim)
        v_txt = attn.add_v_proj(norm_encoder_hidden_states).view(1, -1, heads, head_dim)
        if getattr(attn, "norm_added_q", None) is not None:
            q_txt = attn.norm_added_q(q_txt)
        if getattr(attn, "norm_added_k", None) is not None:
            k_txt = attn.norm_added_k(k_txt)
    else:
        q_txt = k_txt = v_txt = None

    context_pre_only = bool(getattr(attn, "context_pre_only", False))

    # 3a. Fast path: single fused FlashAttention varlen kernel across all variable-resolution segments
    if plan is not None and plan.use_fa_varlen and _FLASH_ATTN_VARLEN_FUNC is not None:
        if q_txt is None:
            out_img = _FLASH_ATTN_VARLEN_FUNC(
                q_img.squeeze(0).contiguous(),
                k_img.squeeze(0).contiguous(),
                v_img.squeeze(0).contiguous(),
                plan.cu_img_i32,
                plan.cu_img_i32,
                plan.max_seqlen_img,
                plan.max_seqlen_img,
                dropout_p=0.0,
                causal=False,
            )
            packed_img_attn = out_img.reshape(1, -1, heads * head_dim).to(q_img.dtype)
            packed_img_out = attn.to_out[0](packed_img_attn)
            packed_img_out = attn.to_out[1](packed_img_out)
            return packed_img_out, None

        q_joint = torch_mod.empty((plan.total_joint, heads, head_dim), device=q_img.device, dtype=q_img.dtype)
        k_joint = torch_mod.empty((plan.total_joint, heads, head_dim), device=q_img.device, dtype=q_img.dtype)
        v_joint = torch_mod.empty((plan.total_joint, heads, head_dim), device=q_img.device, dtype=q_img.dtype)
        q_joint.index_copy_(0, plan.img_in_joint_idx, q_img.squeeze(0))
        q_joint.index_copy_(0, plan.txt_in_joint_idx, q_txt.squeeze(0))
        k_joint.index_copy_(0, plan.img_in_joint_idx, k_img.squeeze(0))
        k_joint.index_copy_(0, plan.txt_in_joint_idx, k_txt.squeeze(0))
        v_joint.index_copy_(0, plan.img_in_joint_idx, v_img.squeeze(0))
        v_joint.index_copy_(0, plan.txt_in_joint_idx, v_txt.squeeze(0))

        out_joint = _FLASH_ATTN_VARLEN_FUNC(
            q_joint,
            k_joint,
            v_joint,
            plan.cu_joint_i32,
            plan.cu_joint_i32,
            plan.max_seqlen_joint,
            plan.max_seqlen_joint,
            dropout_p=0.0,
            causal=False,
        )
        packed_img_attn = out_joint.index_select(0, plan.img_in_joint_idx).reshape(1, -1, heads * head_dim).to(q_img.dtype)
        packed_img_out = attn.to_out[0](packed_img_attn)
        packed_img_out = attn.to_out[1](packed_img_out)

        packed_txt_out = None
        if not context_pre_only:
            packed_txt_attn = out_joint.index_select(0, plan.txt_in_joint_idx).reshape(1, -1, heads * head_dim).to(q_img.dtype)
            packed_txt_out = attn.to_add_out(packed_txt_attn)
        return packed_img_out, packed_txt_out

    # 3b. Fallback path (CPU / FP32 / strict SDPA MATH mode): grouped SDPA by (N_m, L_m)
    num_segments = len(cu_img) - 1
    img_outputs: List[Any] = [None] * num_segments
    txt_outputs: List[Any] = [None] * num_segments

    if plan is not None:
        if q_txt is not None:
            group_iter = plan.joint_groups
        else:
            group_iter = [((n_m, 0), seg_indices) for n_m, seg_indices in plan.img_only_groups]
    else:
        groups: Dict[Tuple[int, int], List[int]] = {}
        for m in range(num_segments):
            n_m = cu_img[m + 1] - cu_img[m]
            t_m = (cu_txt[m + 1] - cu_txt[m]) if q_txt is not None else 0
            groups.setdefault((n_m, t_m), []).append(m)
        group_iter = list(groups.items())

    for (n_m, t_m), seg_indices in group_iter:
        if len(seg_indices) == 1:
            m = seg_indices[0]
            i0, i1 = cu_img[m], cu_img[m + 1]
            qi = q_img[:, i0:i1]
            ki = k_img[:, i0:i1]
            vi = v_img[:, i0:i1]
            if q_txt is not None:
                t0, t1 = cu_txt[m], cu_txt[m + 1]
                q_joint = torch_mod.cat([qi, q_txt[:, t0:t1]], dim=1).transpose(1, 2)
                k_joint = torch_mod.cat([ki, k_txt[:, t0:t1]], dim=1).transpose(1, 2)
                v_joint = torch_mod.cat([vi, v_txt[:, t0:t1]], dim=1).transpose(1, 2)
            else:
                q_joint = qi.transpose(1, 2)
                k_joint = ki.transpose(1, 2)
                v_joint = vi.transpose(1, 2)
            out_joint = F.scaled_dot_product_attention(
                q_joint, k_joint, v_joint, dropout_p=0.0, is_causal=False
            )
            out_joint = out_joint.transpose(1, 2).reshape(1, -1, heads * head_dim).to(q_img.dtype)
            if q_txt is not None:
                img_outputs[m] = out_joint[:, :n_m]
                txt_outputs[m] = out_joint[:, n_m:]
            else:
                img_outputs[m] = out_joint
        else:
            q_batch, k_batch, v_batch = [], [], []
            for m in seg_indices:
                i0, i1 = cu_img[m], cu_img[m + 1]
                qi = q_img[:, i0:i1]
                ki = k_img[:, i0:i1]
                vi = v_img[:, i0:i1]
                if q_txt is not None:
                    t0, t1 = cu_txt[m], cu_txt[m + 1]
                    q_batch.append(torch_mod.cat([qi, q_txt[:, t0:t1]], dim=1))
                    k_batch.append(torch_mod.cat([ki, k_txt[:, t0:t1]], dim=1))
                    v_batch.append(torch_mod.cat([vi, v_txt[:, t0:t1]], dim=1))
                else:
                    q_batch.append(qi)
                    k_batch.append(ki)
                    v_batch.append(vi)
            q_joint = torch_mod.cat(q_batch, dim=0).transpose(1, 2)
            k_joint = torch_mod.cat(k_batch, dim=0).transpose(1, 2)
            v_joint = torch_mod.cat(v_batch, dim=0).transpose(1, 2)
            out_joint = F.scaled_dot_product_attention(
                q_joint, k_joint, v_joint, dropout_p=0.0, is_causal=False
            )
            out_joint = out_joint.transpose(1, 2).reshape(len(seg_indices), -1, heads * head_dim).to(q_img.dtype)
            for b_idx, m in enumerate(seg_indices):
                item_out = out_joint[b_idx : b_idx + 1]
                if q_txt is not None:
                    img_outputs[m] = item_out[:, :n_m]
                    txt_outputs[m] = item_out[:, n_m:]
                else:
                    img_outputs[m] = item_out

    packed_img_attn = torch_mod.cat(img_outputs, dim=1)
    # 4. Packed single-GEMM output projection on image tokens [1, N_total, D]
    packed_img_out = attn.to_out[0](packed_img_attn)
    packed_img_out = attn.to_out[1](packed_img_out)

    packed_txt_out = None
    if q_txt is not None and not context_pre_only:
        packed_txt_attn = torch_mod.cat(txt_outputs, dim=1)
        packed_txt_out = attn.to_add_out(packed_txt_attn)

    return packed_img_out, packed_txt_out


def forward_packed_sd35(
    torch_mod: Any,
    transformer: Any,
    latents_list: Sequence[Any],
    timesteps_list: Sequence[Any],
    encoder_hidden_states_list: Sequence[Any],
    pooled_projections_list: Sequence[Any],
) -> Tuple[List[Any], Dict[str, Any]]:
    """Run SD3Transformer2DModel over a heterogeneous-resolution list of inputs using Token Sequence Packing.

    Each element ``m`` in ``latents_list`` has shape ``[1, C, h_m, w_m]`` where ``(h_m, w_m)`` may differ
    across elements. Patch tokens are packed into a single contiguous ``[1, sum(N_m), D]`` tensor so all
    Linear projections, LayerNorms, AdaLNs, and FFNs execute without padding tokens, while Joint Attention
    is isolated per segment via ``cu_seqlens``—preserving exact bitwise equivalence with solo execution.
    """
    raw_tf = _unwrap_transformer(transformer)
    num_items = len(latents_list)
    if num_items == 0:
        raise ValueError("forward_packed_sd35 requires at least one item")
    if not (len(timesteps_list) == len(encoder_hidden_states_list) == len(pooled_projections_list) == num_items):
        raise ValueError("all input lists to forward_packed_sd35 must have matching lengths")

    patch_size = int(raw_tf.config.patch_size)
    out_channels = int(raw_tf.out_channels)
    device = latents_list[0].device

    # 1. Per-item 2D patchification + positional embedding, then pack into [1, N_total, D]
    img_segments = []
    img_counts = []
    patch_Verify_hw: List[Tuple[int, int]] = []
    for lat in latents_list:
        if lat.ndim != 4 or lat.shape[0] != 1:
            raise ValueError("each latent in latents_list must have shape [1, C, H_lat, W_lat]")
        h_lat, w_lat = int(lat.shape[-2]), int(lat.shape[-1])
        patch_Verify_hw.append((h_lat // patch_size, w_lat // patch_size))
        # Call underlying PatchEmbed directly so runtime_state global height/width does not override h_lat/w_lat
        pos_embed_mod = getattr(raw_tf.pos_embed, "module", raw_tf.pos_embed)
        seg = pos_embed_mod(lat)
        img_segments.append(seg)
        img_counts.append(int(seg.shape[1]))

    packed_hidden_states = torch_mod.cat(img_segments, dim=1)

    # 2. Pack context embeddings into [1, L_total, D] and compute [M, D] timestep+pooled embeddings
    txt_segments = []
    txt_counts = []
    for enc in encoder_hidden_states_list:
        if enc.ndim != 3 or enc.shape[0] != 1:
            raise ValueError("each encoder_hidden_states item must have shape [1, L, D_ctx]")
        proj_enc = raw_tf.context_embedder(enc)
        txt_segments.append(proj_enc)
        txt_counts.append(int(proj_enc.shape[1]))

    packed_encoder_hidden_states = torch_mod.cat(txt_segments, dim=1)
    plan = _build_packed_batch_plan(torch_mod, img_counts, txt_counts, packed_hidden_states)
    cu_img = plan.cu_img
    cu_txt = plan.cu_txt

    t_stacked = torch_mod.cat([t.reshape(1).to(device=device) for t in timesteps_list], dim=0)
    pooled_stacked = torch_mod.cat([p.to(device=device) for p in pooled_projections_list], dim=0)
    temb_vec = raw_tf.time_text_embed(t_stacked, pooled_stacked)

    # 3. Execute packed JointTransformerBlocks
    for block in raw_tf.transformer_blocks:
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_hidden_states2,
            gate_msa2,
        ) = _packed_adaln_zero(
            torch_mod, block.norm1, packed_hidden_states, temb_vec, seg_ids=plan.img_seg_ids
        )
        context_pre_only = bool(getattr(block, "context_pre_only", False))
        if context_pre_only:
            norm_encoder_hidden_states = _packed_adaln_continuous(
                torch_mod, block.norm1_context, packed_encoder_hidden_states, temb_vec, seg_ids=plan.txt_seg_ids
            )
            c_gate_msa = c_shift_mlp = c_scale_mlp = c_gate_mlp = None
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp, _, _ = _packed_adaln_zero(
                torch_mod, block.norm1_context, packed_encoder_hidden_states, temb_vec, seg_ids=plan.txt_seg_ids
            )

        attn_output, context_attn_output = _packed_joint_attention(
            torch_mod,
            block.attn,
            norm_hidden_states,
            norm_encoder_hidden_states,
            cu_img,
            cu_txt,
            plan=plan,
        )
        packed_hidden_states = packed_hidden_states + gate_msa * attn_output

        # Optional dual spatial self-attention (SD3.5-Medium blocks with use_dual_attention)
        if getattr(block, "use_dual_attention", False) and getattr(block, "attn2", None) is not None:
            if norm_hidden_states2 is None and getattr(block, "norm1_dual", None) is not None:
                norm_hidden_states2 = block.norm1_dual.norm(packed_hidden_states)
                emb_dual = block.norm1_dual.linear(block.norm1_dual.silu(temb_vec))
                emb_dual_tok = _expand_per_segment(
                    torch_mod, emb_dual, seg_ids=plan.img_seg_ids
                )
                shift_msa2, scale_msa2, gate_msa2 = emb_dual_tok.chunk(3, dim=-1)
                norm_hidden_states2 = norm_hidden_states2 * (1.0 + scale_msa2) + shift_msa2
            attn_output2, _ = _packed_joint_attention(
                torch_mod, block.attn2, norm_hidden_states2, None, cu_img, cu_txt, plan=plan
            )
            packed_hidden_states = packed_hidden_states + gate_msa2 * attn_output2

        # Packed FFN on image tokens [1, N_total, D]
        norm_hidden_states = block.norm2(packed_hidden_states) * (1.0 + scale_mlp) + shift_mlp
        ff_output = block.ff(norm_hidden_states)
        packed_hidden_states = packed_hidden_states + gate_mlp * ff_output

        # Packed FFN on text tokens [1, L_total, D]
        if not context_pre_only:
            packed_encoder_hidden_states = packed_encoder_hidden_states + c_gate_msa * context_attn_output
            norm_encoder_hidden_states = (
                block.norm2_context(packed_encoder_hidden_states) * (1.0 + c_scale_mlp) + c_shift_mlp
            )
            context_ff_output = block.ff_context(norm_encoder_hidden_states)
            packed_encoder_hidden_states = packed_encoder_hidden_states + c_gate_mlp * context_ff_output

    # 4. Final packed AdaLN + output projection + per-resolution unpatchify
    packed_hidden_states = _packed_adaln_continuous(
        torch_mod, raw_tf.norm_out, packed_hidden_states, temb_vec, seg_ids=plan.img_seg_ids
    )
    packed_hidden_states = raw_tf.proj_out(packed_hidden_states)

    outputs = []
    for m, (p_h, p_w) in enumerate(patch_Verify_hw):
        i0, i1 = cu_img[m], cu_img[m + 1]
        seg = packed_hidden_states[:, i0:i1]
        seg = seg.reshape(1, p_h, p_w, patch_size, patch_size, out_channels)
        seg = torch_mod.einsum("nhwpqc->nchpwq", seg)
        out = seg.reshape(1, out_channels, p_h * patch_size, p_w * patch_size)
        outputs.append(out)

    meta = {
        "sequence_packed": True,
        "attention_backend": "flash_attn_varlen" if plan.use_fa_varlen else "grouped_sdpa",
        "attention_provider": _FLASH_ATTN_VARLEN_PROVIDER if plan.use_fa_varlen else "grouped_sdpa",
        "packed_segments": num_items,
        "cu_seqlens_img": cu_img,
        "cu_seqlens_txt": cu_txt,
        "packed_image_tokens": cu_img[-1],
        "padded_image_tokens": num_items * max(img_counts),
        "padding_saved_tokens": num_items * max(img_counts) - cu_img[-1],
    }
    return outputs, meta


class SD35AsyncPipelineManager:
    """Bubble-free pipeline manager for SD3.5-Medium (FlashPS-inspired side-stream encode & 2D-VAE decode)."""

    def __init__(self, backend: Any):
        self.backend = backend
        self.encoder_pool_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sd35-encode")
        self.decoder_pool_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sd35-vae")
        self.encode_tickets: Dict[str, Tuple[Dict[str, Any], Any]] = {}
        self.decode_tickets: Dict[str, Tuple[str, Dict[str, Any], Any]] = {}
        self.encode_counter = 0
        self.decode_counter = 0
        self._encode_stream = None
        self._vae_stream = None
        self._vae_lock = threading.Lock()

    def _get_cuda_stream(self, attr: str):
        torch = getattr(self.backend, "torch", None)
        device = str(getattr(self.backend, "device", "cpu"))
        if (
            torch is not None
            and device.startswith("cuda")
            and hasattr(torch, "cuda")
            and callable(getattr(torch.cuda, "is_available", None))
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "Stream")
        ):
            stream = getattr(self, attr, None)
            if stream is None:
                try:
                    stream = torch.cuda.Stream(device=device)
                    setattr(self, attr, stream)
                except Exception:
                    return None
            return stream
        return None

    @property
    def encode_stream(self):
        return self._get_cuda_stream("_encode_stream")

    @property
    def vae_stream(self):
        return self._get_cuda_stream("_vae_stream")

    def submit_encode(self, request: Dict[str, Any]) -> Dict[str, Any]:
        spec = self.backend._spec(request)
        self.encode_counter += 1
        ticket = f"sd35-encode-{self.encode_counter}"
        delay_s = float(self.backend.config.get("options", {}).get("encode_delay_ms", 0.0)) / 1000.0

        def _task():
            t0 = time.perf_counter()
            if delay_s > 0:
                time.sleep(delay_s)
            if getattr(self.backend, "_effective_t5_device", lambda: "cuda")() == "cpu":
                stream = self.encode_stream
                if getattr(self.backend, "_supports_stream_stage", False):
                    stage_info = self.backend._ensure_staged(
                        spec["prompt"],
                        spec["negative_prompt"],
                        spec["max_sequence_length"],
                        spec["guidance_scale"] > 1.0,
                        stream=stream,
                    )
                else:
                    stage_info = self.backend._ensure_staged(
                        spec["prompt"],
                        spec["negative_prompt"],
                        spec["max_sequence_length"],
                        spec["guidance_scale"] > 1.0,
                    )
                if stream is not None and hasattr(stream, "synchronize"):
                    stream.synchronize()
                direct_tensors = stage_info.pop("_direct_tensors", None)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "spec": spec,
                    "key": stage_info["key"],
                    "stage_info": stage_info,
                    "tensors": direct_tensors,
                    "push_ms": 0.0,
                    "encode_ms": round(elapsed_ms, 3),
                    "bubble_free_prefetch": True,
                }
            stream = self.encode_stream
            stage_info, fetched = self.backend._fetch_staged(
                spec["prompt"],
                spec["negative_prompt"],
                spec["max_sequence_length"],
                spec["guidance_scale"] > 1.0,
                stream=stream,
            )
            if stream is not None and hasattr(stream, "synchronize"):
                stream.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "spec": spec,
                "key": stage_info["key"],
                "stage_info": stage_info,
                "tensors": fetched["tensors"],
                "push_ms": fetched["push_ms"],
                "encode_ms": round(elapsed_ms, 3),
                "bubble_free_prefetch": True,
            }

        self.encode_tickets[ticket] = (spec, self.encoder_pool_exec.submit(_task))
        return {"ticket": ticket, "ready": False, "bubble_free": True}

    def poll_encode(self, request: Dict[str, Any]) -> Dict[str, Any]:
        ticket = request.get("ticket")
        if not isinstance(ticket, str) or ticket not in self.encode_tickets:
            raise ValueError("unknown encode ticket")
        _, future = self.encode_tickets[ticket]
        if request.get("wait", False):
             payload = future.result()
        elif not future.done():
            return {"ticket": ticket, "ready": False}
        else:
            payload = future.result()
        stage_info = payload["stage_info"]
        out = {
            "ticket": ticket,
            "ready": True,
            "model_ms": payload["encode_ms"],
            "push_ms": payload["push_ms"],
            "cached": stage_info.get("cached", False),
            "shm_staged": bool(stage_info.get("shm_staged", False)),
            "bubble_free": True,
        }
        if isinstance(stage_info.get("negative_prompt_cache"), dict):
            out["negative_prompt_cache"] = stage_info["negative_prompt_cache"]
        return out

    def discard_encode(self, request: Dict[str, Any]) -> Dict[str, Any]:
        ticket = request.get("ticket")
        if not isinstance(ticket, str) or not ticket:
            raise ValueError("encode_discard requires ticket")
        entry = self.encode_tickets.pop(ticket, None)
        if entry is not None:
            _, future = entry
            future.cancel()
        return {"discarded": True, "ticket": ticket}

    def consume_encode_ticket(self, ticket: str, expected_spec: Dict[str, Any]) -> Dict[str, Any]:
        if ticket not in self.encode_tickets:
            raise ValueError("unknown encode ticket")
        spec, future = self.encode_tickets[ticket]
        if not future.done():
            raise ValueError("encode ticket is not ready")
        if spec != expected_spec:
            raise ValueError("encode ticket belongs to a different spec")
        payload = future.result()
        del self.encode_tickets[ticket]
        return payload

    def submit_decode(self, request: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        state_id = request["state_id"]
        if any(sid == state_id for sid, _, _ in self.decode_tickets.values()):
            raise ValueError("state is already owned by an outstanding async decode")
        if state.get("output") is None and state.get("latents") is None:
            raise ValueError("decode_start requires completed latents or output")
        if state.get("steps") is not None and state.get("step", state["steps"]) < state["steps"]:
            raise ValueError("decode_start requires all denoising steps to be complete")

        self.decode_counter += 1
        ticket = f"sd35-decode-{state_id}-{request.get('id', self.decode_counter)}"
        delay_s = float(self.backend.config.get("options", {}).get("decode_delay_ms", 0.0)) / 1000.0
        latents = state.get("latents")
        existing_output = state.get("output")
        if existing_output is None and latents is not None:
            state["latents"] = None

        torch = getattr(self.backend, "torch", None)
        stream = self.vae_stream
        boundary = None
        if stream is not None and torch is not None and hasattr(torch.cuda, "Event"):
            try:
                boundary = torch.cuda.Event()
                boundary.record(torch.cuda.current_stream(self.backend.device))
                if latents is not None and hasattr(latents, "record_stream"):
                    latents.record_stream(stream)
            except Exception:
                boundary = None

        def _decode_task():
            t0 = time.perf_counter()
            if delay_s > 0:
                time.sleep(delay_s)
            if existing_output is not None:
                arr = existing_output
            else:
                with self._vae_lock:
                    ctx = torch.cuda.stream(stream) if (stream is not None and torch is not None) else nullcontext()
                    with ctx:
                        if boundary is not None and stream is not None:
                            stream.wait_event(boundary)
                        arr = self.backend._decode_latents_to_np(latents)
                        if stream is not None and hasattr(stream, "synchronize"):
                            stream.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return arr, {
                "kind": "decode",
                "ms": round(elapsed_ms, 3),
                "asynchronous": True,
                "bubble_free": True,
                "stream": int(getattr(stream, "cuda_stream", 0)) if stream is not None else 0,
            }

        self.decode_tickets[ticket] = (state_id, dict(request), self.decoder_pool_exec.submit(_decode_task))
        return {"ticket": ticket, "ready": False, "state_id": state_id, "bubble_free": True}

    def poll_decode(self, request: Dict[str, Any]) -> Dict[str, Any]:
        ticket = request.get("ticket")
        if not isinstance(ticket, str) or ticket not in self.decode_tickets:
            raise ValueError("unknown decode ticket")
        state_id, orig_req, future = self.decode_tickets[ticket]
        if not request.get("wait", False) and not future.done():
            return {"ticket": ticket, "ready": False, "state_id": state_id}
        output_arr, event = future.result()
        del self.decode_tickets[ticket]
        if state_id in self.backend.states:
            self.backend.states[state_id]["output"] = output_arr
            merged_req = dict(orig_req, **request, state_id=state_id)
            finished = self.backend.finish(merged_req)
            return dict(
                finished,
                ticket=ticket,
                ready=True,
                state_id=state_id,
                model_ms=event["ms"],
                event=event,
                bubble_free=True,
            )
        return {
            "ticket": ticket,
            "ready": True,
            "state_id": state_id,
            "model_ms": event["ms"],
            "event": event,
            "bubble_free": True,
        }

    def has_active_decode(self, state_id: str) -> bool:
        return any(sid == state_id for sid, _, _ in self.decode_tickets.values())

    def inflight_decode_count(self) -> int:
        return len(self.decode_tickets)

    def close(self) -> None:
        for _, (_, _, fut) in list(self.decode_tickets.items()):
            try:
                fut.result(timeout=5.0)
            except Exception:
                pass
        for _, (_, fut) in list(self.encode_tickets.items()):
            try:
                fut.result(timeout=5.0)
            except Exception:
                pass
        self.decode_tickets.clear()
        self.encode_tickets.clear()
        self.decoder_pool_exec.shutdown(wait=True, cancel_futures=True)
        self.encoder_pool_exec.shutdown(wait=True, cancel_futures=True)
