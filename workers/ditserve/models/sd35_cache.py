"""Hook-based MM-DiT Taylor block cache for SD3.5-Medium (single-GPU & xDiT multi-GPU)."""

from __future__ import annotations

from typing import Any, Dict, Optional


def install_sd35_fast_cache_hooks(transformer: Any, torch_mod: Any) -> None:
    """Attach block-level Taylor residual cache hooks to SD3Transformer2DModel.transformer_blocks."""
    blocks = getattr(transformer, "transformer_blocks", None)
    if transformer is None or blocks is None or getattr(transformer, "_fast_cache_hooked", False):
        return
    num_blocks = len(blocks)
    if num_blocks < 8:
        return
    F = torch_mod.nn.functional

    for idx, blk in enumerate(blocks):
        orig_forward = blk.forward

        def _make_block_forward(block_idx: int, orig_fn: Any):
            def _wrapped_forward(hidden_states, encoder_hidden_states=None, temb=None, *args, **kwargs):
                fc = getattr(transformer, "_fast_cache", None)
                if fc is None or not fc.get("enabled", False):
                    return orig_fn(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        *args,
                        **kwargs,
                    )
                fn_blocks = int(fc.get("fn_blocks", 6))
                if block_idx == 0:
                    fc["_h0"] = hidden_states
                    fc["_skip_this_step"] = False
                    return orig_fn(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        *args,
                        **kwargs,
                    )
                if block_idx == fn_blocks - 1:
                    out_enc, out_hid = orig_fn(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        *args,
                        **kwargs,
                    )
                    front_delta = out_hid - fc["_h0"]
                    fc["_front_delta"] = front_delta
                    fc["_h_fn"] = out_hid
                    step = int(fc.get("step", 0))
                    total_steps = int(fc.get("total_steps", 12))
                    warmup_steps = int(fc.get("warmup_steps", 3))
                    max_cont = int(fc.get("max_continuous", 1))
                    thresh = float(fc.get("threshold", 0.22))
                    can_check = (
                        step >= warmup_steps
                        and step < total_steps - 1
                        and int(fc.get("consecutive_skips", 0)) < max_cont
                        and fc.get("prev_front_delta") is not None
                        and fc.get("prev_tail_delta") is not None
                    )
                    skip_tail = False
                    if can_check:
                        prev_fd = fc["prev_front_delta"]
                        fd_f = front_delta.flatten(1).float()
                        pfd_f = prev_fd.flatten(1).float()
                        cos_sim = F.cosine_similarity(fd_f, pfd_f, dim=1).mean()
                        cos_dist = float((1.0 - cos_sim).item())
                        rel_l1 = float(((fd_f - pfd_f).abs().mean() / pfd_f.abs().mean().clamp(min=1e-6)).item())
                        metric = min(cos_dist, 0.55 * rel_l1)
                        fc["last_metric"] = round(metric, 4)
                        skip_tail = metric < thresh
                    fc["_skip_this_step"] = skip_tail
                    return out_enc, out_hid

                if fc.get("_skip_this_step", False):
                    if block_idx < num_blocks - 1:
                        return encoder_hidden_states, hidden_states
                    front_delta = fc["_front_delta"]
                    prev_fd = fc["prev_front_delta"]
                    cur_norm = torch_mod.linalg.vector_norm(front_delta.float(), dim=(1, 2), keepdim=True)
                    prev_norm = torch_mod.linalg.vector_norm(prev_fd.float(), dim=(1, 2), keepdim=True).clamp(min=1e-6)
                    norm_scale = torch_mod.clamp(cur_norm / prev_norm, min=0.85, max=1.15).to(hidden_states.dtype)
                    tail_est = fc["prev_tail_delta"] * norm_scale
                    vel = fc.get("tail_velocity")
                    taylor_alpha = float(fc.get("taylor_alpha", -0.10))
                    if vel is not None and taylor_alpha != 0.0:
                        tail_est = tail_est + (taylor_alpha * vel).to(hidden_states.dtype)
                    out_hid = hidden_states + tail_est
                    fc["consecutive_skips"] = int(fc.get("consecutive_skips", 0)) + 1
                    fc["reused_steps"] = int(fc.get("reused_steps", 0)) + 1
                    fc["step"] = int(fc.get("step", 0)) + 1
                    return encoder_hidden_states, out_hid

                out_enc, out_hid = orig_fn(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    *args,
                    **kwargs,
                )
                if block_idx == num_blocks - 1:
                    tail_delta = out_hid - fc["_h_fn"]
                    if fc.get("prev_tail_delta") is not None:
                        fc["tail_velocity"] = (tail_delta - fc["prev_tail_delta"]).detach()
                    fc["prev_front_delta"] = fc["_front_delta"].detach()
                    fc["prev_tail_delta"] = tail_delta.detach()
                    fc["consecutive_skips"] = 0
                    fc["step"] = int(fc.get("step", 0)) + 1
                return out_enc, out_hid

            return _wrapped_forward

        blk.forward = _make_block_forward(idx, orig_forward)
    transformer._fast_cache_hooked = True


def configure_sd35_fast_cache(transformer: Any, request: Dict[str, Any], steps: int) -> None:
    """Configure or disable per-run MM-DiT fast cache state on the transformer."""
    if transformer is None:
        return
    if not bool(request.get("cache", False)):
        transformer._fast_cache = None
        return
    transformer._fast_cache = {
        "enabled": True,
        "step": 0,
        "total_steps": steps,
        "fn_blocks": int(request.get("cache_fn", 6)),
        "warmup_steps": int(request.get("cache_warmup", 3)),
        "threshold": float(request.get("threshold", 0.10)),
        "max_continuous": int(request.get("cache_max_continuous", 2)),
        "taylor_alpha": float(request.get("taylor_alpha", -0.15)),
        "prev_front_delta": None,
        "prev_tail_delta": None,
        "tail_velocity": None,
        "consecutive_skips": 0,
        "reused_steps": 0,
    }


def pop_sd35_reused_steps(transformer: Any) -> int:
    """Read and clear reused_steps from the transformer fast-cache state."""
    fc: Optional[Dict[str, Any]] = getattr(transformer, "_fast_cache", None) if transformer is not None else None
    if fc is None:
        return 0
    reused = int(fc.get("reused_steps", 0))
    transformer._fast_cache = None
    return reused
