"""Pinned Wan CFG split-prefix executor; no model or dependency monkeypatches.

The preprocessing/output arithmetic follows diffusers 0.36.0 WanTransformer3DModel
(Apache-2.0). Cache arithmetic is delegated to cache-dit 1.5.0. This executor
reorders only deterministic, state-free prefix computation, never CFG commits.
"""

import importlib.metadata
import time


class CfgSplitPrefix:
    def __init__(self, worker):
        self.worker, self.model = worker, worker.pipe.transformer
        self._validate_model()
        from .cfg_empty_context import EmptyCacheContextGuard
        self._context_guard = EmptyCacheContextGuard(self.model)
        from cache_dit.caching.cache_blocks import CachedBlocks
        from cache_dit.caching.forward_pattern import ForwardPattern
        self.blocks = CachedBlocks(self.model.blocks, transformer=self.model,
            forward_pattern=ForwardPattern.Pattern_2, cache_prefix="blocks",
            cache_context=worker._context_names[0], context_manager=worker.manager)

    def _validate_model(self):
        from diffusers.models.transformers.transformer_wan import (
            WanTransformer3DModel, WanTransformerBlock, WanAttnProcessor)
        from diffusers import UniPCMultistepScheduler
        torch, model, worker = self.worker.torch, self.model, self.worker
        if any(importlib.metadata.version(name) != expected for name, expected in
               (("cache-dit", "1.5.0"), ("diffusers", "0.36.0"), ("torch", "2.10.0+cu126"))):
            raise ValueError("split-prefix requires the pinned dependency versions")
        if type(model) is not WanTransformer3DModel or len(model.blocks) != 30:
            raise ValueError("split-prefix requires exactly the Wan1.3B 30-block model")
        required = dict(num_attention_heads=12, attention_head_dim=128, in_channels=16,
                        out_channels=16, num_layers=30, ffn_dim=8960, text_dim=4096,
                        image_dim=None, added_kv_proj_dim=None)
        if (any(getattr(model.config, key, None) != value for key, value in required.items()) or
                tuple(model.config.patch_size) != (1, 2, 2) or model.dtype != torch.float16 or
                model.gradient_checkpointing or getattr(model, "_is_parallelized", False) or
                getattr(worker.pipe, "transformer_2", None) is not None or
                worker.manager.persistent_context or len(worker._context_names) != 1 or
                worker._scheduler_type is not UniPCMultistepScheduler):
            raise ValueError("unsupported model, precision, scheduler or parallel split-prefix contract")
        if (torch.distributed.is_initialized() or worker.attention not in ("flash", "efficient") or
                getattr(model, "peft_config", None) or getattr(model, "_hf_peft_config_loaded", False)):
            raise ValueError("split-prefix refuses distributed, unknown attention and adapted weights")
        original = getattr(model, "_original_forward", None)
        if getattr(original, "__func__", None) is not WanTransformer3DModel.forward:
            raise ValueError("split-prefix requires the unmodified native Wan forward")
        for block in model.blocks:
            if type(block) is not WanTransformerBlock:
                raise ValueError("split-prefix refuses unknown transformer block implementations")
            for attention in (block.attn1, block.attn2):
                if (type(attention.processor) is not WanAttnProcessor or attention.fused_projections or
                        getattr(attention.processor, "_parallel_config", None) is not None):
                    raise ValueError("split-prefix requires unfused native Wan attention")
        for module in model.modules():
            registry = getattr(module, "_diffusers_hook", None)
            if (module.training or getattr(module, "_hf_hook", None) is not None or
                    getattr(module, "_orig_mod", None) is not None or
                    getattr(module, "_compiled_call_impl", None) is not None or
                    getattr(module, "_forward_hooks", {}) or getattr(module, "_forward_pre_hooks", {}) or
                    registry is not None and registry.hooks or
                    type(module).__module__.split(".")[0] not in ("torch", "diffusers")):
                raise ValueError("split-prefix refuses training, hooks, offload, compiled or unknown modules")
            if module is not model and getattr(module.forward, "__func__", None) is not type(module).forward:
                raise ValueError("split-prefix refuses modified module forwards")
            if isinstance(module, torch.nn.Dropout) and module.p != 0:
                raise ValueError("split-prefix refuses stochastic/dropout modules")

    def validate_state(self, state):
        from diffusers import UniPCMultistepScheduler
        from .dual_state import BUFFER_SUFFIXES
        if type(state.scheduler) is not UniPCMultistepScheduler:
            raise ValueError("split-prefix requires native UniPC")
        if (state.latents.shape[0] != 1 or state.latents.ndim != 5 or
                len(state.contexts) != 1 or state.spec["guidance_scale"] != 5 or
                state.spec["steps"] != 32):
            raise ValueError("split-prefix requires the single-input frozen CFG trajectory")
        context = next(iter(state.contexts.values()))
        required = dict(Fn_compute_blocks=8, Bn_compute_blocks=0, enable_separate_cfg=True,
                        cfg_compute_first=False, cfg_diff_compute_separate=True,
                        max_warmup_steps=8, warmup_interval=1, max_cached_steps=-1,
                        max_continuous_cached_steps=1, residual_diff_threshold=.08,
                        num_inference_steps=32, steps_computation_mask=None,
                        force_refresh_step_hint=None, max_accumulated_residual_diff_threshold=None)
        if any(getattr(context.cache_config, name) != value for name, value in required.items()):
            raise ValueError("unsupported split-prefix cache decision configuration")
        extra = context.extra_cache_config
        if (extra.downsample_factor != 1 or extra.important_condition_threshold != 0 or
                extra.l1_hidden_states_diff_threshold not in (None, 0.) or context.calibrator_config is not None):
            raise ValueError("split-prefix refuses alternative similarity/calibrator paths")
        allowed_buffers = {"blocks" + suffix for suffix in BUFFER_SUFFIXES}
        if any(type(key) is not str or key not in allowed_buffers or
               not self.worker.torch.is_tensor(value) for key, value in context.buffers.items()):
            raise ValueError("split-prefix refuses unknown feature buffers")

    def _time(self, operation, *args):
        self.worker._sync()
        started = time.perf_counter()
        value = operation(*args)
        self.worker._sync()
        return value, (time.perf_counter() - started) * 1000

    def _prefix(self, state, embedding):
        model = self.model
        hidden = state.latents.to(model.dtype)
        batch, _, frames, height, width = hidden.shape
        pt, ph, pw = model.config.patch_size
        rotary = model.rope(hidden)
        hidden = model.patch_embedding(hidden).flatten(2).transpose(1, 2)
        timestep = state.scheduler.timesteps[state.step_index].expand(batch)
        temb, projected, encoder, image = model.condition_embedder(timestep, embedding, None, timestep_seq_len=None)
        if image is not None or temb.ndim != 2:
            raise ValueError("unsupported image or expanded timestep in split-prefix")
        projected = projected.unflatten(1, (6, -1))
        original = hidden
        hidden, encoder = self.blocks.call_Fn_blocks(hidden, encoder, projected, rotary)
        residual = self.blocks._get_Fn_residual(original, hidden)
        return dict(hidden=hidden, encoder=encoder, temb=temb, projected=projected,
                    rotary=rotary, residual=residual,
                    shape=(batch, frames // pt, height // ph, width // pw, pt, ph, pw))

    def _gate(self, packet, branch):
        manager = self.worker.manager
        manager.set_context(self.blocks.cache_context)
        manager.mark_step_begin()
        decision = bool(manager.can_cache(packet["residual"], parallelized=False, prefix="blocks_Fn_residual"))
        context = manager._current_context
        values = context.residual_diffs if branch == "cond" else context.cfg_residual_diffs
        return dict(branch=branch, native_reuse=decision, accepted_reuse=decision,
                    forced_full=False, residual_diff=values.get(str(context.executed_steps - 1)))

    def _probe(self, state, packets):
        from .cfg_adapter import _clone_graph
        manager = self.worker.manager
        previous = dict(vars(manager))
        try:
            manager._cached_context_manager = _clone_graph(self.worker.torch, state.contexts, copy_tensors=False)
            manager._current_context = None
            events = []
            for branch in ("cond", "uncond"):
                event = self._gate(packets[branch], branch)
                events.append(event)
                # Conditional bookkeeping is visible to the subsequent probe.
                # CFG buffers are separate; no conditional middle result can be
                # read by the unconditional gate under validate_state's contract.
                if event["native_reuse"]:
                    manager.add_cached_step()
                else:
                    manager.set_Fn_buffer(packets[branch]["residual"], prefix="blocks_Fn_residual")
            return events
        finally:
            vars(manager).clear()
            vars(manager).update(previous)

    def _suffix(self, packet, reused):
        manager, blocks = self.worker.manager, self.blocks
        hidden, encoder = packet["hidden"], packet["encoder"]
        args = (packet["projected"], packet["rotary"])
        if reused:
            manager.add_cached_step()
            hidden, encoder = manager.apply_cache(hidden, encoder, prefix="blocks_Bn_residual",
                                                   encoder_prefix="blocks_Bn_residual")
        else:
            manager.set_Fn_buffer(packet["residual"], prefix="blocks_Fn_residual")
            hidden, encoder, residual, encoder_residual = blocks.call_Mn_blocks(hidden, encoder, *args)
            manager.set_Bn_buffer(residual, prefix="blocks_Bn_residual")
            manager.set_Bn_encoder_buffer(encoder_residual, prefix="blocks_Bn_residual")
        hidden, encoder = blocks.call_Bn_blocks(hidden, encoder, *args)
        model, temb = self.model, packet["temb"]
        shift, scale = (model.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift, scale = shift.to(hidden.device), scale.to(hidden.device)
        hidden = (model.norm_out(hidden.float()) * (1 + scale) + shift).type_as(hidden)
        hidden = model.proj_out(hidden)
        batch, frames, height, width, pt, ph, pw = packet["shape"]
        hidden = hidden.reshape(batch, frames, height, width, pt, ph, pw, -1)
        hidden = hidden.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def pair(self, state):
        worker = self.worker
        started = time.perf_counter()
        self._context_guard.validate()
        context_guard_ms = (time.perf_counter() - started) * 1000
        packets, prefix_ms, commit_ms, outputs, accepted = {}, {}, {}, {}, []
        with worker._activate(state), worker._dit_attention_context():
            worker.manager.set_context(self.blocks.cache_context)
            for branch, embedding in (("cond", state.prompt_embeds), ("uncond", state.negative_prompt_embeds)):
                packets[branch], prefix_ms[branch] = self._time(self._prefix, state, embedding)
            probe, probe_ms = self._time(self._probe, state, packets)
            reuse = all(event["native_reuse"] for event in probe)
            for index, branch in enumerate(("cond", "uncond")):
                def commit():
                    event = self._gate(packets[branch], branch)
                    if event["native_reuse"] != probe[index]["native_reuse"]:
                        raise RuntimeError("split-prefix gate changed between isolated probe and commit")
                    event.update(accepted_reuse=reuse, forced_full=not reuse)
                    accepted.append(event)
                    return self._suffix(packets[branch], reuse)
                outputs[branch], commit_ms[branch] = self._time(commit)
            # No temporary tensor is installed on RequestState or the worker.
            del packets
        worker._sync()
        return outputs, dict(gates=accepted, probe_gates=probe, prefix_ms=prefix_ms,
            gate_probe_ms=probe_ms, commit_ms=commit_ms, context_guard_ms=context_guard_ms,
            cache_context_mode=self._context_guard.mode,
            branch_ms={key: prefix_ms[key] + commit_ms[key] for key in prefix_ms},
            pair_ms=(time.perf_counter() - started) * 1000,
            fn_block_calls=dict(cond=8, uncond=8), mn_block_calls=dict(cond=0 if reuse else 22, uncond=0 if reuse else 22),
            gate_calls=dict(speculative=2, committed=2),
            cost_semantics="all empty-context guards, prefix, isolated gate probe, committed native gates and suffix work included")
