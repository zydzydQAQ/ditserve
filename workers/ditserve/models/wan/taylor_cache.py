"""First-Order Taylor Velocity-Compensated Residual Extrapolation for Wan2.1 DBCache.

Zero-order residual caching (standard DBCache) reuses the last computed anchor
step's residual buffer unchanged on skipped steps:
    R_hat(t) = R(t_anchor)

At higher skip thresholds (e.g. tau = 0.16 ~ 0.24, skipping 25% ~ 35% of steps),
zero-order hold introduces trajectory lag along curved flow-matching paths.
``TaylorResidualCompensator`` tracks the finite-difference velocity of the
middle/tail residual buffer (``blocks_Bn_residual_buffer`` and
``blocks_Bn_residual_buffer_cfg``) across anchor steps:
    V_Bn = (R(t_anchor) - R(t_prev)) / (t_anchor - t_prev)
and pre-compensates ``context.buffers`` before candidate skip steps:
    R_hat(t) = R(t_anchor) + clamp_norm(alpha * dt * V_Bn, max_ratio * ||R(t_anchor)||)

Because ``blocks_Fn_residual_buffer`` (used by cache-dit for the L1 threshold
decision) is never modified, the skip schedule remains identical while decoded
video PSNR and SSIM improve significantly.
"""

from __future__ import annotations

import importlib
from typing import Dict


class TaylorResidualCompensator:
    """Per-request first-order Taylor residual extrapolator for WanStepWorker."""

    BRANCH_KEYS = (
        ("cond_reused", "blocks_Bn_residual_buffer"),
        ("uncond_reused", "blocks_Bn_residual_buffer_cfg"),
    )

    def __init__(
        self,
        torch_module=None,
        *,
        alpha: float = 0.28,
        damping: float = 0.70,
        max_norm_ratio: float = 0.12,
        adaptive_lsq: bool = True,
        enabled: bool = True,
    ):
        self.torch = torch_module or importlib.import_module("torch")
        self.alpha = float(alpha)
        self.damping = float(damping)
        self.max_norm_ratio = float(max_norm_ratio)
        self.adaptive_lsq = bool(adaptive_lsq)
        self.enabled = bool(enabled)
        self._anchors: Dict[str, "torch.Tensor"] = {}
        self._anchor_steps: Dict[str, int] = {}
        self._velocities: Dict[str, "torch.Tensor"] = {}
        self._prev_velocities: Dict[str, "torch.Tensor"] = {}
        self._adaptive_alphas: Dict[str, float] = {}
        self.monotonic_gate = True
        self.stats = {
            "anchor_updates": 0,
            "taylor_compensated_forwards": 0,
            "zero_order_forwards": 0,
        }

    def reset(self) -> None:
        self._anchors.clear()
        self._anchor_steps.clear()
        self._velocities.clear()
        self._prev_velocities.clear()
        self._adaptive_alphas.clear()

    def _extrapolate(
        self,
        buf_key: str,
        anchor: "torch.Tensor",
        velocity: "torch.Tensor",
        dt: int,
    ) -> "torch.Tensor":
        torch = self.torch
        eff_dt = min(float(dt), 2.5) * (self.damping ** max(0, dt - 1))
        vel_eff = velocity
        prev_vel = self._prev_velocities.get(buf_key)
        eff_alpha = self._adaptive_alphas.get(buf_key, self.alpha) if self.adaptive_lsq else self.alpha
        if self.monotonic_gate and prev_vel is not None:
            # Only extrapolate coordinates that drift consistently with the fitted sign
            if eff_alpha >= 0.0:
                agree_mask = ((prev_vel * velocity) > 0).to(velocity.dtype)
            else:
                agree_mask = ((prev_vel * velocity) < 0).to(velocity.dtype)
            vel_eff = velocity * agree_mask
        delta = (eff_alpha * eff_dt) * vel_eff
        anchor_f32 = anchor.to(torch.float32)
        anchor_norm = torch.linalg.vector_norm(anchor_f32).clamp(min=1e-6)
        delta_norm = torch.linalg.vector_norm(delta).clamp(min=1e-6)
        max_allowed = self.max_norm_ratio * anchor_norm
        scale = torch.clamp(max_allowed / delta_norm, max=1.0)
        compensated = anchor_f32 + delta * scale
        return compensated.to(anchor.dtype).contiguous()

    def pre_step(self, state) -> None:
        """Apply first-order Taylor extrapolation to Bn residual buffers before ``runtime.step(state)``."""
        if not self.enabled or not getattr(state, "cache_enabled", False):
            return
        step_idx = int(state.step_index)
        for context in state.contexts.values():
            buffers = getattr(context, "buffers", None)
            if not isinstance(buffers, dict):
                continue
            for _reused_flag, buf_key in self.BRANCH_KEYS:
                if buf_key not in buffers or buffers[buf_key] is None:
                    continue
                anchor = self._anchors.get(buf_key)
                vel = self._velocities.get(buf_key)
                anchor_step = self._anchor_steps.get(buf_key)
                if anchor is not None and vel is not None and anchor_step is not None and step_idx > anchor_step:
                    dt = step_idx - anchor_step
                    buffers[buf_key] = self._extrapolate(buf_key, anchor, vel, dt)

    def post_step(self, state, record: dict) -> dict:
        """Update finite-difference velocity on anchor steps or restore clean anchor after a skipped step."""
        if not getattr(state, "cache_enabled", False):
            return record
        torch = self.torch
        step_idx = int(record.get("step_index", state.step_index - 1))
        ctx_list = record.get("cache_contexts") or []
        ctx_info = ctx_list[0] if ctx_list else {}

        compensated_this_step = 0
        for context in state.contexts.values():
            buffers = getattr(context, "buffers", None)
            if not isinstance(buffers, dict):
                continue
            for reused_flag, buf_key in self.BRANCH_KEYS:
                cur_buf = buffers.get(buf_key)
                if cur_buf is None:
                    continue
                was_reused = bool(ctx_info.get(reused_flag, False))
                if not was_reused:
                    # Exact Bn residual was freshly computed by the transformer
                    prev_anchor = self._anchors.get(buf_key)
                    prev_step = self._anchor_steps.get(buf_key)
                    if prev_anchor is not None and prev_step is not None and step_idx > prev_step:
                        dt = float(step_idx - prev_step)
                        new_vel = (
                            cur_buf.detach().to(torch.float32) - prev_anchor.to(torch.float32)
                        ) / dt
                        if buf_key in self._velocities:
                            old_vel = self._velocities[buf_key]
                            self._prev_velocities[buf_key] = old_vel
                            if self.adaptive_lsq:
                                dot = float(torch.sum(new_vel * old_vel).item())
                                denom = float(torch.sum(old_vel * old_vel).item()) + 1e-6
                                raw_ratio = dot / denom
                                max_a = abs(self.alpha)
                                fitted = max(-max_a, min(max_a, self.alpha * raw_ratio))
                                self._adaptive_alphas[buf_key] = fitted
                        self._velocities[buf_key] = new_vel
                    self._anchors[buf_key] = cur_buf.detach().clone()
                    self._anchor_steps[buf_key] = step_idx
                    self.stats["anchor_updates"] += 1
                else:
                    # Cache was reused; restore clean anchor buffer for subsequent steps
                    if self.enabled and buf_key in self._anchors and buf_key in self._velocities:
                        buffers[buf_key] = self._anchors[buf_key].detach().clone()
                        self.stats["taylor_compensated_forwards"] += 1
                        compensated_this_step += 1
                    else:
                        self.stats["zero_order_forwards"] += 1

        record["taylor_compensated_forwards"] = compensated_this_step
        return record

    def step(self, runtime, state) -> dict:
        """Convenience wrapper around ``runtime.step(state)`` with pre/post Taylor hooks."""
        with runtime.torch.inference_mode():
            self.pre_step(state)
            record = runtime.step(state)
            return self.post_step(state, record)
