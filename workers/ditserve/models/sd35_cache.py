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
                bn_blocks = max(0, int(fc.get("bn_blocks", 0)))
                tail_end = max(fn_blocks + 1, num_blocks - bn_blocks)
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
                    fc["_enc_fn"] = out_enc
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
                    forced = fc.get("forced_skip_steps")
                    skip_tail = False
                    if forced is not None:
                        skip_tail = (
                            step in forced
                            and fc.get("prev_front_delta") is not None
                            and fc.get("prev_tail_delta") is not None
                        )
                    elif can_check:
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
                    if skip_tail:
                        fc.setdefault("skipped_step_indices", []).append(step)
                    return out_enc, out_hid

                def _compensate_residual(
                    anchor_key: str,
                    vel_key: str,
                    prev_vel_key: str,
                    alpha_key: str,
                    step_idx: int,
                    dtype: Any,
                ) -> Any:
                    est = fc[anchor_key]
                    mode = str(fc.get("cache_mode", "taylor"))
                    if mode == "static":
                        return est
                    if bool(fc.get("use_norm_scale", False)) and anchor_key == "prev_tail_delta":
                        front_delta = fc["_front_delta"]
                        prev_fd = fc["prev_front_delta"]
                        cur_norm = torch_mod.linalg.vector_norm(front_delta.float(), dim=(1, 2), keepdim=True)
                        prev_norm = torch_mod.linalg.vector_norm(prev_fd.float(), dim=(1, 2), keepdim=True).clamp(min=1e-6)
                        norm_scale = torch_mod.clamp(cur_norm / prev_norm, min=0.94, max=1.06).to(dtype)
                        est = est * norm_scale
                    vel = fc.get(vel_key)
                    base_alpha = float(fc.get("taylor_alpha", 0.25))
                    if vel is not None and base_alpha != 0.0:
                        prev_step = int(fc.get("prev_anchor_step", step_idx - 1))
                        dt = max(1, step_idx - prev_step)
                        damping = float(fc.get("taylor_damping", 0.80))
                        eff_dt = min(float(dt), 2.5) * (damping ** max(0, dt - 1))
                        eff_alpha = fc.get(alpha_key) if bool(fc.get("adaptive_lsq", True)) else None
                        if eff_alpha is None:
                            eff_alpha = base_alpha
                        vel_eff = vel
                        prev_vel = fc.get(prev_vel_key)
                        if bool(fc.get("monotonic_gate", True)) and prev_vel is not None:
                            if isinstance(eff_alpha, torch_mod.Tensor):
                                agree_mask = torch_mod.where(
                                    eff_alpha >= 0.0,
                                    (prev_vel * vel) > 0,
                                    (prev_vel * vel) < 0,
                                ).to(vel.dtype)
                            elif float(eff_alpha) >= 0.0:
                                agree_mask = ((prev_vel * vel) > 0).to(vel.dtype)
                            else:
                                agree_mask = ((prev_vel * vel) < 0).to(vel.dtype)
                            vel_eff = vel * agree_mask
                        delta_r = (eff_alpha * eff_dt) * vel_eff
                        max_ratio = float(fc.get("max_norm_ratio", 0.14))
                        r_norm = torch_mod.linalg.vector_norm(est.float(), dim=(1, 2), keepdim=True).clamp(min=1e-6)
                        d_norm = torch_mod.linalg.vector_norm(delta_r, dim=(1, 2), keepdim=True).clamp(min=1e-6)
                        clamp_scale = torch_mod.clamp((max_ratio * r_norm) / d_norm, max=1.0)
                        est = (est.float() + delta_r * clamp_scale).to(dtype)
                    return est

                if fc.get("_skip_this_step", False) and block_idx < tail_end:
                    if block_idx < tail_end - 1:
                        return encoder_hidden_states, hidden_states
                    step = int(fc.get("step", 0))
                    tail_est = _compensate_residual(
                        "prev_tail_delta",
                        "tail_velocity",
                        "prev_tail_velocity",
                        "adaptive_alpha",
                        step,
                        hidden_states.dtype,
                    )
                    out_hid = hidden_states + tail_est
                    out_enc = encoder_hidden_states
                    if bn_blocks > 0 and encoder_hidden_states is not None and fc.get("prev_enc_delta") is not None:
                        enc_est = _compensate_residual(
                            "prev_enc_delta",
                            "enc_velocity",
                            "prev_enc_velocity",
                            "adaptive_enc_alpha",
                            step,
                            encoder_hidden_states.dtype,
                        )
                        out_enc = encoder_hidden_states + enc_est
                    if bn_blocks == 0:
                        fc["consecutive_skips"] = int(fc.get("consecutive_skips", 0)) + 1
                        fc["reused_steps"] = int(fc.get("reused_steps", 0)) + 1
                        fc["step"] = step + 1
                    return out_enc, out_hid

                out_enc, out_hid = orig_fn(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    *args,
                    **kwargs,
                )
                if not fc.get("_skip_this_step", False) and block_idx == tail_end - 1:
                    step = int(fc.get("step", 0))
                    tail_delta = out_hid - fc["_h_fn"]
                    prev_td = fc.get("prev_tail_delta")
                    prev_step = fc.get("prev_anchor_step")
                    base_alpha = float(fc.get("taylor_alpha", 0.25))
                    max_a = abs(base_alpha)
                    if prev_td is not None and prev_step is not None and step > int(prev_step):
                        dt = float(step - int(prev_step))
                        new_vel = (tail_delta.detach().float() - prev_td.float()) / dt
                        old_vel = fc.get("tail_velocity")
                        if old_vel is not None:
                            fc["prev_tail_velocity"] = old_vel
                            if bool(fc.get("adaptive_lsq", True)):
                                dot = torch_mod.sum(new_vel * old_vel, dim=(1, 2), keepdim=True)
                                denom = torch_mod.sum(old_vel * old_vel, dim=(1, 2), keepdim=True).clamp(min=1e-6)
                                raw_ratio = dot / denom
                                fc["adaptive_alpha"] = torch_mod.clamp(base_alpha * raw_ratio, min=-max_a, max=max_a)
                        fc["tail_velocity"] = new_vel
                    fc["prev_front_delta"] = fc["_front_delta"].detach()
                    fc["prev_tail_delta"] = tail_delta.detach()
                    if out_enc is not None and fc.get("_enc_fn") is not None:
                        enc_delta = out_enc - fc["_enc_fn"]
                        prev_ed = fc.get("prev_enc_delta")
                        if prev_ed is not None and prev_step is not None and step > int(prev_step):
                            dt = float(step - int(prev_step))
                            new_evel = (enc_delta.detach().float() - prev_ed.float()) / dt
                            old_evel = fc.get("enc_velocity")
                            if old_evel is not None:
                                fc["prev_enc_velocity"] = old_evel
                                if bool(fc.get("adaptive_lsq", True)):
                                    edot = torch_mod.sum(new_evel * old_evel, dim=(1, 2), keepdim=True)
                                    edenom = torch_mod.sum(old_evel * old_evel, dim=(1, 2), keepdim=True).clamp(min=1e-6)
                                    eratio = edot / edenom
                                    fc["adaptive_enc_alpha"] = torch_mod.clamp(base_alpha * eratio, min=-max_a, max=max_a)
                            fc["enc_velocity"] = new_evel
                        fc["prev_enc_delta"] = enc_delta.detach()
                    fc["prev_anchor_step"] = step
                    if bn_blocks == 0:
                        fc["consecutive_skips"] = 0
                        fc["step"] = step + 1
                elif block_idx == num_blocks - 1 and bn_blocks > 0:
                    step = int(fc.get("step", 0))
                    if fc.get("_skip_this_step", False):
                        fc["consecutive_skips"] = int(fc.get("consecutive_skips", 0)) + 1
                        fc["reused_steps"] = int(fc.get("reused_steps", 0)) + 1
                    else:
                        fc["consecutive_skips"] = 0
                    fc["step"] = step + 1
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
    forced = request.get("forced_skip_steps")
    transformer._fast_cache = {
        "enabled": True,
        "step": 0,
        "total_steps": steps,
        "fn_blocks": int(request.get("cache_fn", 6)),
        "bn_blocks": int(request.get("cache_bn", 0)),
        "warmup_steps": int(request.get("cache_warmup", 3)),
        "threshold": float(request.get("threshold", 0.10)),
        "max_continuous": int(request.get("cache_max_continuous", 2)),
        "cache_mode": str(request.get("cache_mode", "taylor")),
        "use_norm_scale": bool(request.get("use_norm_scale", False)),
        "taylor_alpha": float(request.get("taylor_alpha", 0.25)),
        "taylor_damping": float(request.get("taylor_damping", 0.80)),
        "max_norm_ratio": float(request.get("max_norm_ratio", 0.14)),
        "adaptive_lsq": bool(request.get("adaptive_lsq", True)),
        "monotonic_gate": bool(request.get("monotonic_gate", True)),
        "forced_skip_steps": set(int(x) for x in forced) if forced is not None else None,
        "skipped_step_indices": [],
        "prev_front_delta": None,
        "prev_tail_delta": None,
        "prev_enc_delta": None,
        "prev_anchor_step": None,
        "tail_velocity": None,
        "prev_tail_velocity": None,
        "enc_velocity": None,
        "prev_enc_velocity": None,
        "adaptive_alpha": None,
        "adaptive_enc_alpha": None,
        "consecutive_skips": 0,
        "reused_steps": 0,
    }


def pop_sd35_reused_steps(transformer: Any) -> int:
    """Read and clear reused_steps from the transformer fast-cache state."""
    fc: Optional[Dict[str, Any]] = getattr(transformer, "_fast_cache", None) if transformer is not None else None
    if fc is None:
        return 0
    reused = int(fc.get("reused_steps", 0))
    transformer._last_fast_cache_skipped_steps = list(fc.get("skipped_step_indices", []))
    transformer._fast_cache = None
    return reused
