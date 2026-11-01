"""Optional single-branch CFG execution with a per-request fold budget.

For guidance scale ``g``, a folded step reconstructs the guided prediction from
its conditional branch and the most recent stored CFG delta. Full-CFG steps
refresh that delta. Fixed windows and curvature gating are explicit numerical
modes; quality must be measured against the same runtime without folding.
"""

from __future__ import annotations

import importlib
import time
from typing import Optional, Set

from .runtime import model_timestep
from .taylor_cache import TaylorResidualCompensator


class AsymmetricCFGStepper:
    """Combined Taylor-DBCache + Late-Step Asymmetric Single-Branch CFG Reuse executor."""

    def __init__(
        self,
        torch_module=None,
        *,
        folded_steps: Optional[Set[int]] = None,
        cfg_damping: float = 0.96,
        cfg_vel_alpha: float = 0.15,
        final_step_exact_anchor: bool = True,
        force_exact_cond_on_fold: bool = True,
        curvature_threshold: Optional[float] = None,
        min_full_cfg_steps: int = 5,
        max_curvature_guard: float = 1.65,
        max_folded_steps: Optional[int] = None,
        enabled: bool = True,
    ):
        self.torch = torch_module or importlib.import_module("torch")
        if max_folded_steps is not None and (type(max_folded_steps) is not int or max_folded_steps < 0):
            raise ValueError("max_folded_steps must be a nonnegative integer or None")
        self.max_folded_steps = max_folded_steps
        self.curvature_threshold = float(curvature_threshold) if curvature_threshold is not None else None
        if folded_steps is not None:
            self.folded_steps: Set[int] = set(folded_steps)
        elif self.curvature_threshold is not None:
            self.folded_steps = set()
        else:
            self.folded_steps = {17, 18}
        self.pre_fold_anchor_step: Optional[int] = (
            min(self.folded_steps) - 1 if self.folded_steps else (int(min_full_cfg_steps) - 1 if self.curvature_threshold is not None else None)
        )
        self.force_exact_cond_on_fold: bool = bool(force_exact_cond_on_fold)
        self.cfg_damping = float(cfg_damping)
        self.cfg_vel_alpha = float(cfg_vel_alpha)
        self.final_step_exact_anchor = bool(final_step_exact_anchor)
        self.min_full_cfg_steps = int(min_full_cfg_steps)
        self.max_curvature_guard = float(max_curvature_guard)
        self.enabled = bool(enabled)

        self._last_delta_cfg: Optional["torch.Tensor"] = None
        self._last_delta_step: Optional[int] = None
        self._delta_velocity: Optional["torch.Tensor"] = None
        self._warmup_ref_l1: Optional[float] = None
        self._last_curvature: Optional[float] = None
        self._last_raw_curvature: Optional[float] = None
        self._had_folded_since_anchor: bool = False
        self.curvature_history: list[dict] = []
        self.dynamic_fold_start_step: Optional[int] = None

        self.stats = {
            "full_cfg_steps": 0,
            "folded_cfg_steps": 0,
            "saved_uncond_forwards": 0,
            "curvature_triggered_folds": 0,
        }

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "folded_steps": sorted(self.folded_steps),
            "cfg_damping": self.cfg_damping,
            "cfg_vel_alpha": self.cfg_vel_alpha,
            "final_step_exact_anchor": self.final_step_exact_anchor,
            "curvature_threshold": self.curvature_threshold,
            "min_full_cfg_steps": self.min_full_cfg_steps,
            "max_folded_steps": self.max_folded_steps,
            "dynamic_fold_start_step": self.dynamic_fold_start_step,
            "curvature_history": list(self.curvature_history),
            **self.stats,
        }

    def reset(self) -> None:
        self._last_delta_cfg = None
        self._last_delta_step = None
        self._delta_velocity = None
        self._warmup_ref_l1 = None
        self._last_curvature = None
        self._last_raw_curvature = None
        self._had_folded_since_anchor = False
        self.curvature_history.clear()
        self.dynamic_fold_start_step = None

    def fold_prediction(self, step_index: int, pred_cond: "torch.Tensor", guidance_scale: float) -> "torch.Tensor":
        prediction = pred_cond + (guidance_scale - 1.0) * self._predict_delta_cfg(step_index, pred_cond.dtype)
        self._had_folded_since_anchor = True
        if self.dynamic_fold_start_step is None:
            self.dynamic_fold_start_step = int(step_index)
        if self.curvature_threshold is not None:
            self.stats["curvature_triggered_folds"] += 1
        self.stats["folded_cfg_steps"] += 1
        self.stats["saved_uncond_forwards"] += 1
        return prediction

    def export_state(self) -> tuple[dict, dict]:
        config = {
            "folded_steps": sorted(self.folded_steps),
            "cfg_damping": self.cfg_damping,
            "cfg_vel_alpha": self.cfg_vel_alpha,
            "final_step_exact_anchor": self.final_step_exact_anchor,
            "force_exact_cond_on_fold": self.force_exact_cond_on_fold,
            "curvature_threshold": self.curvature_threshold,
            "min_full_cfg_steps": self.min_full_cfg_steps,
            "max_curvature_guard": self.max_curvature_guard,
            "max_folded_steps": self.max_folded_steps,
            "enabled": self.enabled,
        }
        runtime = {
            "last_delta_step": self._last_delta_step,
            "warmup_ref_l1": self._warmup_ref_l1,
            "last_curvature": self._last_curvature,
            "last_raw_curvature": self._last_raw_curvature,
            "had_folded_since_anchor": self._had_folded_since_anchor,
            "curvature_history": list(self.curvature_history),
            "dynamic_fold_start_step": self.dynamic_fold_start_step,
            "stats": dict(self.stats),
        }
        tensors = {}
        if self._last_delta_cfg is not None:
            tensors["delta"] = self._last_delta_cfg.detach().cpu().clone()
        if self._delta_velocity is not None:
            tensors["velocity"] = self._delta_velocity.detach().cpu().clone()
        return {"config": config, "runtime": runtime}, tensors

    def import_state(self, metadata: dict, tensors: dict) -> None:
        current, unused = self.export_state()
        if metadata.get("config") != current["config"]:
            raise ValueError("CFG-fold configuration differs across continuation import")
        runtime = metadata.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError("CFG-fold continuation is missing runtime state")
        self._last_delta_step = runtime["last_delta_step"]
        self._warmup_ref_l1 = runtime["warmup_ref_l1"]
        self._last_curvature = runtime["last_curvature"]
        self._last_raw_curvature = runtime["last_raw_curvature"]
        self._had_folded_since_anchor = bool(runtime["had_folded_since_anchor"])
        self.curvature_history = list(runtime["curvature_history"])
        self.dynamic_fold_start_step = runtime["dynamic_fold_start_step"]
        self.stats = dict(runtime["stats"])
        self._last_delta_cfg = None if self._last_delta_step is None else tensors["delta"]
        self._delta_velocity = tensors.get("velocity")

    def _update_curvature_metrics(self, index: int, delta_f32: "torch.Tensor") -> None:
        torch = self.torch
        cur_l1 = max(1e-6, float(torch.linalg.vector_norm(delta_f32, ord=1).item()))
        if self._warmup_ref_l1 is None or index <= min(4, max(1, self.min_full_cfg_steps // 2)):
            self._warmup_ref_l1 = cur_l1 if self._warmup_ref_l1 is None else max(self._warmup_ref_l1, cur_l1)
        if self._last_delta_cfg is not None and self._last_delta_step is not None and index > self._last_delta_step:
            dt = float(index - self._last_delta_step)
            self._delta_velocity = (delta_f32 - self._last_delta_cfg) / dt
            damped_prev = self._last_delta_cfg * (self.cfg_damping ** dt)
            diff_l1 = float(torch.linalg.vector_norm(delta_f32 - damped_prev, ord=1).item())
            raw_curv = (diff_l1 / cur_l1) / dt
            ref_denom = max(cur_l1, 0.65 * (self._warmup_ref_l1 or cur_l1))
            cos_sim = float(
                torch.nn.functional.cosine_similarity(
                    delta_f32.flatten().unsqueeze(0),
                    self._last_delta_cfg.flatten().unsqueeze(0),
                ).item()
            )
            ang_curv = max(0.0, 1.0 - cos_sim)
            rel_curv = 0.55 * ((diff_l1 / ref_denom) / dt) + 0.45 * ang_curv
            self._last_raw_curvature = raw_curv
            self._last_curvature = rel_curv
            self.curvature_history.append({
                "step": int(index),
                "curvature": round(float(rel_curv), 5),
                "raw_l1_curvature": round(float(raw_curv), 5),
                "angular_curvature": round(float(ang_curv), 5),
            })
        self._last_delta_cfg = delta_f32.clone()
        self._last_delta_step = int(index)

    def should_fold(self, step_index: int, total_steps: int) -> bool:
        if not self.enabled:
            return False
        if self.max_folded_steps is not None and self.stats["folded_cfg_steps"] >= self.max_folded_steps:
            return False
        if self._last_delta_cfg is None:
            return False
        if step_index >= total_steps - 1 and self.final_step_exact_anchor:
            return False
        if self.curvature_threshold is not None:
            if step_index < self.min_full_cfg_steps:
                return False
            if self.folded_steps and step_index not in self.folded_steps:
                return False
            return (self._last_curvature is not None
                    and self._last_curvature <= self.curvature_threshold
                    and self._last_curvature <= self.max_curvature_guard)
        return step_index in self.folded_steps

    def _predict_delta_cfg(self, step_index: int, dtype: "torch.dtype") -> "torch.Tensor":
        torch = self.torch
        assert self._last_delta_cfg is not None
        assert self._last_delta_step is not None
        dt = max(1, step_index - self._last_delta_step)
        damping = self.cfg_damping ** dt
        base = self._last_delta_cfg * damping
        if self._delta_velocity is not None and self.cfg_vel_alpha != 0.0:
            eff_dt = min(float(dt), 2.0) * (0.65 ** max(0, dt - 1))
            corr = (self.cfg_vel_alpha * eff_dt) * self._delta_velocity
            base_norm = torch.linalg.vector_norm(base).clamp(min=1e-6)
            corr_norm = torch.linalg.vector_norm(corr).clamp(min=1e-6)
            scale = torch.clamp((0.08 * base_norm) / corr_norm, max=1.0)
            base = base + corr * scale
        return base.to(dtype)

    def observe_cfg_pair(self, index: int, pred_cond: "torch.Tensor", pred_uncond: "torch.Tensor") -> None:
        """Record an exact (pred_cond, pred_uncond) pair from a CFG=2 parallel step."""
        torch = self.torch
        delta_f32 = pred_cond.detach().to(torch.float32) - pred_uncond.detach().to(torch.float32)
        self._update_curvature_metrics(int(index), delta_f32)
        self.stats["full_cfg_steps"] += 1

    def step(
        self,
        runtime,
        state,
        taylor_compensator: Optional[TaylorResidualCompensator] = None,
    ) -> dict:
        """Advance ``state`` by one solver step with optional Taylor cache + Late-Step CFG folding."""
        torch = self.torch
        with runtime._lock, torch.inference_mode(), runtime.p._sdpa_math_context(torch):
            runtime._check(state, "gpu")
            if state.finished or state.step_index >= state.spec["steps"]:
                raise ValueError("request has no remaining denoising steps")
            runtime._update_policy(state)

            index = int(state.step_index)
            total_steps = int(state.spec["steps"])
            g = float(state.spec["guidance_scale"])
            fold_this_step = (g > 1.0) and self.should_fold(index, total_steps)

            # Force an exact 2-branch anchor on the step immediately preceding the folded
            # window so Delta_CFG is anchored from an exact (non-DBCache) forward pass,
            # and keep the single `cond` branch exact during folded steps to prevent
            # compounding DBCache approximation with CFG branch folding.
            if (
                self.enabled
                and self.curvature_threshold is None
                and self.pre_fold_anchor_step is not None
                and index == self.pre_fold_anchor_step
            ):
                state.needs_anchor = True
            elif fold_this_step and self.force_exact_cond_on_fold:
                state.needs_anchor = True
            elif not fold_this_step and self._had_folded_since_anchor:
                state.needs_anchor = True
                self._had_folded_since_anchor = False

            if taylor_compensator is not None and not state.needs_anchor:
                taylor_compensator.pre_step(state)

            try:
                from xfuser.model_executor.layers.usp import get_active_ring_quantizer

                active_rq = get_active_ring_quantizer()
                if active_rq is not None and hasattr(active_rq, "set_step"):
                    active_rq.set_step(index, total_steps)
            except Exception:
                pass

            runtime._sync()
            started = time.perf_counter()
            anchor = state.needs_anchor

            with runtime._activate(state):
                timestep_scalar = state.scheduler.timesteps[index]
                if getattr(runtime, "xdit", None) is not None:
                    pred_cond = runtime.xdit.forward(state, "cond")
                    if g > 1.0:
                        if not fold_this_step:
                            pred_uncond = runtime.xdit.forward(state, "uncond")
                        else:
                            pred_uncond = None
                    else:
                        pred_uncond = None
                else:
                    timestep = model_timestep(torch, state.latents, timestep_scalar, runtime.expand_timesteps)
                    model_input = state.latents.to(runtime.pipe.transformer.dtype)
                    with runtime.pipe.transformer.cache_context("cond"):
                        pred_cond = runtime.pipe.transformer(
                            hidden_states=model_input,
                            timestep=timestep,
                            encoder_hidden_states=state.prompt_embeds,
                            attention_kwargs=None,
                            return_dict=False,
                        )[0]
                    if g > 1.0 and not fold_this_step:
                        with runtime.pipe.transformer.cache_context("uncond"):
                            pred_uncond = runtime.pipe.transformer(
                                hidden_states=model_input,
                                timestep=timestep,
                                encoder_hidden_states=state.negative_prompt_embeds,
                                attention_kwargs=None,
                                return_dict=False,
                            )[0]
                    else:
                        pred_uncond = None

                if g > 1.0:
                    if not fold_this_step and pred_uncond is not None:
                        delta_f32 = pred_cond.detach().to(torch.float32) - pred_uncond.detach().to(torch.float32)
                        self._update_curvature_metrics(index, delta_f32)
                        prediction = pred_uncond + g * (pred_cond - pred_uncond)
                        self.stats["full_cfg_steps"] += 1
                        if state.cache_enabled:
                            for context in state.contexts.values():
                                context.executed_steps = index + 1
                                context.accumulated_executed_steps = index + 1
                                context.transformer_executed_steps = (index + 1) * 2
                                context.accumulated_transformer_executed_steps = (index + 1) * 2
                    else:
                        prediction = self.fold_prediction(index, pred_cond, g)
                        for context in state.contexts.values():
                            context.executed_steps = index + 1
                            context.accumulated_executed_steps = index + 1
                            context.transformer_executed_steps = (index + 1) * 2
                            context.accumulated_transformer_executed_steps = (index + 1) * 2
                else:
                    prediction = pred_cond
                    self.stats["full_cfg_steps"] += 1

                state.latents = state.scheduler.step(prediction, timestep_scalar, state.latents, return_dict=False)[0]
                state.step_index += 1
                state.needs_anchor = False

            runtime._sync()
            finite = bool(torch.isfinite(state.latents).all().item())
            ms = (time.perf_counter() - started) * 1000.0
            if not finite:
                raise RuntimeError("nonfinite latent after solver step %d" % index)

            record = {
                "kind": "step",
                "step_index": index,
                "completed_steps": state.step_index,
                "ms": ms,
                "forced_cold_anchor": anchor,
                "cfg_folded": fold_this_step,
                "finite": finite,
                "threshold": state.spec["threshold"],
                "max_continuous": state.spec["max_continuous"],
                **runtime._path(state, index),
            }
            if taylor_compensator is not None:
                record = taylor_compensator.post_step(state, record)
            state.events.append(record)
            return record
