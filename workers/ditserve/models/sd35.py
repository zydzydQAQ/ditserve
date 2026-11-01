"""SD3.5-medium image generation with cross-modal encoder pool & dynamic batching."""

import copy
import time
from contextlib import nullcontext

from .base import OUTPUT_MODES, positive_int, progress
from .encoder_pool import ProactiveEncoderPool
from .pipeline import AtomicPipelineBackend, _sdpa_context
from .sd35_batch import (
    SD35AsyncPipelineManager,
    _unwrap_transformer,
    calculate_shift,
    forward_packed_sd35,
    packing_stats,
)
from .sd35_cache import (
    configure_sd35_fast_cache,
    install_sd35_fast_cache_hooks,
    pop_sd35_reused_steps,
)


class SD35Backend(AtomicPipelineBackend):
    model = "sd35"
    operations = AtomicPipelineBackend.operations + (
        "encode_pool_stats",
        "offload_encoders",
        "restore_encoders",
        "run_micro_batch",
        "run_packed_batch",
        "generate_batch",
        "advance_batch",
        "encode_start",
        "encode_poll",
        "decode_start",
        "decode_poll",
    )
    defaults = dict(AtomicPipelineBackend.defaults, steps=28, guidance_scale=4.5, max_sequence_length=256)

    def __init__(self, config):
        super().__init__(config)
        options = config.get("options", {})
        self.attention = str(config.get("attention", options.get("attention", "math")))
        self.encoder_pool = ProactiveEncoderPool(options.get("encoder_pool", {}))
        self.encoders_offloaded = False
        self._async_pipeline_mgr = None
        self._install_fast_cache_hooks()
        if options.get("offload_text_encoders_after_warmup", False):
            self.offload_encoders({})

    def _pipeline_manager(self) -> SD35AsyncPipelineManager:
        mgr = getattr(self, "_async_pipeline_mgr", None)
        if mgr is None:
            mgr = SD35AsyncPipelineManager(self)
            self._async_pipeline_mgr = mgr
        return mgr

    def close(self):
        mgr = getattr(self, "_async_pipeline_mgr", None)
        if mgr is not None:
            mgr.close()
        return super().close()

    def release(self, request):
        state_id = request.get("state_id")
        mgr = getattr(self, "_async_pipeline_mgr", None)
        if mgr is not None and isinstance(state_id, str) and mgr.has_active_decode(state_id):
            raise ValueError("cannot release state while async decode is active")
        return super().release(request)

    def _install_fast_cache_hooks(self):
        """Install block-level Taylor MM-DiT fast-cache hooks on SD3Transformer2DModel.transformer_blocks."""
        install_sd35_fast_cache_hooks(getattr(self.pipe.module, "transformer", None), self.torch)

    def _memory(self):
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            return dict(
                allocated_mb=round(self.torch.cuda.memory_allocated(self.device) / (1 << 20), 2),
                reserved_mb=round(self.torch.cuda.memory_reserved(self.device) / (1 << 20), 2),
            )
        return dict(allocated_mb=0.0, reserved_mb=0.0)

    def offload_encoders(self, _request=None):
        """Move heavy T5-XXL (text_encoder_3, ~9.5 GB BF16) to CPU pinned memory when using pooled embeddings."""
        before = self._memory()
        t5 = getattr(self.pipe.module, "text_encoder_3", None)
        if t5 is not None and not self.encoders_offloaded:
            t5.to("cpu")
            self.release_cache()
            self.encoders_offloaded = True
        return dict(offloaded=self.encoders_offloaded, memory_before=before, memory_after=self._memory())

    def restore_encoders(self, _request=None):
        t5 = getattr(self.pipe.module, "text_encoder_3", None)
        if t5 is not None and self.encoders_offloaded:
            t5.to(self.device)
            self.encoders_offloaded = False
        return dict(offloaded=self.encoders_offloaded, memory_after=self._memory())

    def encode_pool_stats(self, _request=None):
        return dict(pool=self.encoder_pool.stats(), encoders_offloaded=self.encoders_offloaded)

    def _prompt_key(self, prompt: str, negative_prompt: str, max_sequence_length: int, do_cfg: bool) -> str:
        return self.encoder_pool.prompt_key(
            modality="t2i",
            model_id=str(self.config.get("model_path", "sd35")),
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
            dtype=str(self.config.get("dtype", "bfloat16")),
            do_cfg=do_cfg,
        )

    def _ensure_staged(self, prompt: str, negative_prompt: str, max_sequence_length: int, do_cfg: bool) -> dict:
        if not hasattr(self, "encoder_pool") or self.encoder_pool is None:
            self.encoder_pool = ProactiveEncoderPool(self.config.get("options", {}).get("encoder_pool", {}))
        if not hasattr(self, "encoders_offloaded"):
            self.encoders_offloaded = False
        key = self._prompt_key(prompt, negative_prompt, max_sequence_length, do_cfg)
        if key in self.encoder_pool._entries:
            return dict(key=key, staged=False, cached=True)
        was_offloaded = self.encoders_offloaded
        if was_offloaded:
            self.restore_encoders()
        try:
            with self.torch.inference_mode():
                pe, npe, ppe, nppe = self.pipe.module.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt,
                    prompt_3=prompt,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=negative_prompt,
                    negative_prompt_3=negative_prompt,
                    do_classifier_free_guidance=do_cfg,
                    device=self.device,
                    num_images_per_prompt=1,
                    max_sequence_length=max_sequence_length,
                )
            meta = self.encoder_pool.stage_embeddings(
                key,
                {
                    "prompt_embeds": pe,
                    "negative_prompt_embeds": npe,
                    "pooled_prompt_embeds": ppe,
                    "negative_pooled_prompt_embeds": nppe,
                },
                pin_memory=True,
            )
        finally:
            if was_offloaded:
                self.offload_encoders()
        return dict(key=key, staged=True, cached=False, stage_ms=meta["stage_ms"], bytes=meta["bytes"])

    def encode_start(self, request):
        return self._pipeline_manager().submit_encode(request)

    def encode_poll(self, request):
        return self._pipeline_manager().poll_encode(request)

    def decode_start(self, request):
        state = self.state(request)
        return self._pipeline_manager().submit_decode(request, state)

    def decode_poll(self, request):
        return self._pipeline_manager().poll_decode(request)

    def _decode_latents_to_np(self, latents):
        raw_pipe = getattr(self.pipe, "module", self.pipe)
        vae = getattr(raw_pipe, "vae", None)
        image_processor = getattr(raw_pipe, "image_processor", None)
        if vae is None or image_processor is None:
            if hasattr(latents, "detach"):
                arr = latents.detach().cpu().float().numpy()
                return arr
            return latents
        scaling_factor = float(getattr(vae.config, "scaling_factor", 1.5305))
        shift_factor = float(getattr(vae.config, "shift_factor", 0.0609))
        with self.torch.inference_mode():
            scaled = (latents / scaling_factor) + shift_factor
            vae_dtype = getattr(vae, "dtype", None)
            if vae_dtype is not None:
                scaled = scaled.to(dtype=vae_dtype)
            image = vae.decode(scaled, return_dict=False)[0]
            return image_processor.postprocess(image, output_type="np")

    def prepare(self, request):
        encode_ticket = request.get("encode_ticket")
        res = super().prepare(request)
        state = self.states[res["state_id"]]
        spec = state["spec"]
        if encode_ticket is not None:
            payload = self._pipeline_manager().consume_encode_ticket(str(encode_ticket), spec)
            stage_info = payload["stage_info"]
            state["embeds"] = payload["tensors"]
            state["encode_meta"] = dict(
                encoder_pool_hit=True,
                encoder_pool_push_ms=payload["push_ms"],
                encode_ticket=str(encode_ticket),
                bubble_free=True,
            )
        else:
            stage_info = self._ensure_staged(
                spec["prompt"],
                spec["negative_prompt"],
                spec["max_sequence_length"],
                spec["guidance_scale"] > 1.0,
            )
            state["embeds"] = None
            state["encode_meta"] = None
        state["step"] = 0
        state["steps"] = int(spec["steps"])
        state["paused"] = False
        state["latents"] = None
        state["scheduler"] = None
        state["timesteps"] = None
        state["_prefetch_decode_future"] = None
        return dict(res, encoder_pool_stage=stage_info)

    def _encode(self, arguments, *, stream=None):
        prompts = arguments.get("prompt")
        neg_prompts = arguments.get("negative_prompt")
        max_seq_len = int(arguments.get("max_sequence_length", 256))
        guidance_scale = float(arguments.get("guidance_scale", 4.5))
        prompt_list = [str(p) for p in prompts] if isinstance(prompts, list) else [str(prompts)]
        batch_size = len(prompt_list)
        if isinstance(neg_prompts, list):
            neg_list = [str(n) for n in neg_prompts]
            if len(neg_list) == 1 and batch_size > 1:
                neg_list = neg_list * batch_size
        else:
            neg_list = [str(neg_prompts or "")] * batch_size

        # Fast path when all prompts in the batch are identical
        if len(set(zip(prompt_list, neg_list))) == 1:
            stage_info = self._ensure_staged(prompt_list[0], neg_list[0], max_seq_len, guidance_scale > 1.0)
            fetched = self.encoder_pool.fetch_to_device(stage_info["key"], self.device, stream=stream)
            if stream is not None and hasattr(stream, "synchronize"):
                stream.synchronize()
            tensors = fetched["tensors"]
            for name in ("prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"):
                tensor = tensors.get(name)
                if tensor is not None:
                    if batch_size > 1 and tensor.shape[0] == 1:
                        tensor = tensor.repeat(batch_size, *([1] * (tensor.ndim - 1)))
                    arguments[name] = tensor
            arguments.pop("prompt", None)
            arguments.pop("negative_prompt", None)
            return dict(
                encoder_pool_hit=fetched["hit"],
                encoder_pool_push_ms=fetched["push_ms"],
                distinct_prompts=1,
                bubble_free_encode=bool(stream is not None),
            )

        # Heterogeneous multi-prompt micro-batch path: fetch each prompt and concatenate along dim=0
        per_field = {
            "prompt_embeds": [],
            "negative_prompt_embeds": [],
            "pooled_prompt_embeds": [],
            "negative_pooled_prompt_embeds": [],
        }
        hits = 0
        total_push_ms = 0.0
        for p_str, n_str in zip(prompt_list, neg_list):
            stage_info = self._ensure_staged(p_str, n_str, max_seq_len, guidance_scale > 1.0)
            fetched = self.encoder_pool.fetch_to_device(stage_info["key"], self.device, stream=stream)
            hits += int(bool(fetched["hit"]))
            total_push_ms += float(fetched["push_ms"])
            for name, bucket in per_field.items():
                t = fetched["tensors"].get(name)
                if t is not None:
                    bucket.append(t)
        if stream is not None and hasattr(stream, "synchronize"):
            stream.synchronize()
        for name, bucket in per_field.items():
            if len(bucket) == batch_size:
                arguments[name] = self.torch.cat(bucket, dim=0)
        arguments.pop("prompt", None)
        arguments.pop("negative_prompt", None)
        return dict(
            encoder_pool_hit=(hits == batch_size),
            encoder_pool_hits=hits,
            encoder_pool_push_ms=round(total_push_ms, 4),
            distinct_prompts=len(set(zip(prompt_list, neg_list))),
            bubble_free_encode=bool(stream is not None),
        )

    def _init_step_state(self, state: dict, *, bubble_free: bool = False):
        """Initialize per-request embeddings, independent FlowMatch scheduler, and latent tensor for step-wise execution."""
        if state.get("latents") is not None and state.get("scheduler") is not None:
            return
        spec = state["spec"]
        raw_pipe = getattr(self.pipe, "module", self.pipe)
        do_cfg = float(spec["guidance_scale"]) > 1.0

        # 1. Fetch/stage prompt embeddings (using side stream when bubble_free=True)
        stream = self._pipeline_manager().encode_stream if bubble_free else None
        if state.get("embeds") is None:
            stage_info = self._ensure_staged(
                spec["prompt"], spec["negative_prompt"], spec["max_sequence_length"], do_cfg
            )
            fetched = self.encoder_pool.fetch_to_device(stage_info["key"], self.device, stream=stream)
            state["embeds"] = fetched["tensors"]
            state["encode_meta"] = dict(
                encoder_pool_hit=fetched["hit"],
                encoder_pool_push_ms=fetched["push_ms"],
                bubble_free=bool(bubble_free),
            )

        # 2. Initialize per-request independent scheduler & resolution-shifted timesteps
        base_scheduler = getattr(raw_pipe, "scheduler", None)
        if base_scheduler is not None:
            scheduler = copy.deepcopy(base_scheduler)
            scheduler_kwargs = {}
            if getattr(scheduler, "config", None) is not None and scheduler.config.get("use_dynamic_shifting", False):
                raw_tf = _unwrap_transformer(raw_pipe.transformer)
                patch_size = int(getattr(raw_tf.config, "patch_size", 2))
                vae_scale = int(getattr(raw_pipe, "vae_scale_factor", 8))
                latent_h = int(spec["height"]) // vae_scale
                latent_w = int(spec["width"]) // vae_scale
                image_seq_len = (latent_h // patch_size) * (latent_w // patch_size)
                scheduler_kwargs["mu"] = calculate_shift(
                    image_seq_len,
                    scheduler.config.get("base_image_seq_len", 256),
                    scheduler.config.get("max_image_seq_len", 4096),
                    scheduler.config.get("base_shift", 0.5),
                    scheduler.config.get("max_shift", 1.16),
                )
            scheduler.set_timesteps(int(spec["steps"]), device=self.device, **scheduler_kwargs)
            state["scheduler"] = scheduler
            state["timesteps"] = scheduler.timesteps
        else:
            state["scheduler"] = None
            state["timesteps"] = self.torch.arange(int(spec["steps"]), 0, -1, dtype=self.torch.float32, device=self.device)

        # 3. Initialize per-request seeded latents while side-stream H2D embedding copy is in flight
        gen = self.torch.Generator(device="cpu").manual_seed(int(spec["seed"]))
        state["generator"] = gen
        pe = state["embeds"]["prompt_embeds"]
        if hasattr(raw_pipe, "prepare_latents") and getattr(raw_pipe, "transformer", None) is not None:
            raw_tf = _unwrap_transformer(raw_pipe.transformer)
            num_channels = int(raw_tf.config.in_channels)
            state["latents"] = raw_pipe.prepare_latents(
                int(spec["batch_size"]),
                num_channels,
                int(spec["height"]),
                int(spec["width"]),
                pe.dtype,
                self.device,
                gen,
                None,
            )
        else:
            vae_scale = int(getattr(raw_pipe, "vae_scale_factor", 8))
            state["latents"] = self.torch.randn(
                (int(spec["batch_size"]), 4, int(spec["height"]) // vae_scale, int(spec["width"]) // vae_scale),
                generator=gen,
                dtype=pe.dtype,
            ).to(self.device)

        if stream is not None and hasattr(stream, "synchronize"):
            stream.synchronize()

    def _step_members_once(self, states: list, *, force_packing: bool = False):
        """Advance a list of active request states by 1 denoising step (supports mixed timesteps and mixed resolutions)."""
        raw_pipe = getattr(self.pipe, "module", self.pipe)
        transformer = getattr(raw_pipe, "transformer", None)
        if transformer is None:
            for st in states:
                st["step"] = int(st.get("step", 0)) + 1
            return {"sequence_packed": False, "packed_segments": len(states)}

        resolutions = [(int(st["spec"]["height"]), int(st["spec"]["width"])) for st in states]
        cfg_flags = [float(st["spec"]["guidance_scale"]) > 1.0 for st in states]
        txt_lens = [int(st["embeds"]["prompt_embeds"].shape[1]) for st in states]
        hetero_res = len(set(resolutions)) > 1
        hetero_cfg = len(set(cfg_flags)) > 1
        hetero_txt = len(set(txt_lens)) > 1
        use_packing = bool(force_packing or hetero_res or hetero_cfg or hetero_txt)

        if use_packing:
            lat_list, t_list, enc_list, pooled_list = [], [], [], []
            segment_map = []
            for idx, st in enumerate(states):
                lat = st["latents"]
                t = st["timesteps"][int(st["step"])]
                embeds = st["embeds"]
                if cfg_flags[idx]:
                    s_uncond = len(lat_list)
                    lat_list.append(lat)
                    t_list.append(t)
                    enc_list.append(embeds["negative_prompt_embeds"])
                    pooled_list.append(embeds["negative_pooled_prompt_embeds"])
                    s_cond = len(lat_list)
                    lat_list.append(lat)
                    t_list.append(t)
                    enc_list.append(embeds["prompt_embeds"])
                    pooled_list.append(embeds["pooled_prompt_embeds"])
                    segment_map.append((idx, s_uncond, s_cond))
                else:
                    s_cond = len(lat_list)
                    lat_list.append(lat)
                    t_list.append(t)
                    enc_list.append(embeds["prompt_embeds"])
                    pooled_list.append(embeds["pooled_prompt_embeds"])
                    segment_map.append((idx, None, s_cond))

            outputs, pack_meta = forward_packed_sd35(
                self.torch, transformer, lat_list, t_list, enc_list, pooled_list
            )
            for idx, s_uncond, s_cond in segment_map:
                st = states[idx]
                t = st["timesteps"][int(st["step"])]
                if s_uncond is not None:
                    uncond_out = outputs[s_uncond]
                    cond_out = outputs[s_cond]
                    scale = float(st["spec"]["guidance_scale"])
                    noise_pred = uncond_out + scale * (cond_out - uncond_out)
                else:
                    noise_pred = outputs[s_cond]
                st["latents"] = st["scheduler"].step(noise_pred, t, st["latents"], return_dict=False)[0]
                st["step"] = int(st["step"]) + 1
            return pack_meta

        # Homogeneous resolution + homogeneous CFG fast path with 1D vector timestep = [t_1, ..., t_B]
        batch_size = len(states)
        do_cfg = cfg_flags[0]
        raw_tf = _unwrap_transformer(transformer)
        lat_batch = self.torch.cat([st["latents"] for st in states], dim=0)
        t_vec = self.torch.stack([st["timesteps"][int(st["step"])].reshape(()) for st in states], dim=0).to(self.device)
        pe_batch = self.torch.cat([st["embeds"]["prompt_embeds"] for st in states], dim=0)
        ppe_batch = self.torch.cat([st["embeds"]["pooled_prompt_embeds"] for st in states], dim=0)

        if do_cfg:
            npe_batch = self.torch.cat([st["embeds"]["negative_prompt_embeds"] for st in states], dim=0)
            nppe_batch = self.torch.cat([st["embeds"]["negative_pooled_prompt_embeds"] for st in states], dim=0)
            hidden_in = self.torch.cat([lat_batch, lat_batch], dim=0)
            timestep_in = self.torch.cat([t_vec, t_vec], dim=0)
            enc_in = self.torch.cat([npe_batch, pe_batch], dim=0)
            pooled_in = self.torch.cat([nppe_batch, ppe_batch], dim=0)
        else:
            hidden_in = lat_batch
            timestep_in = t_vec
            enc_in = pe_batch
            pooled_in = ppe_batch

        if hasattr(self.xdit, "activate_image"):
            h0, w0 = resolutions[0]
            self.xdit.activate_image(
                height=h0,
                width=w0,
                num_inference_steps=int(states[0]["spec"]["steps"]),
                batch_size=hidden_in.shape[0],
            )
            noise_all = transformer(
                hidden_states=hidden_in,
                timestep=timestep_in,
                encoder_hidden_states=enc_in,
                pooled_projections=pooled_in,
                return_dict=False,
            )[0]
        else:
            noise_all = raw_tf(
                hidden_states=hidden_in,
                timestep=timestep_in,
                encoder_hidden_states=enc_in,
                pooled_projections=pooled_in,
                return_dict=False,
            )[0]

        if do_cfg:
            noise_uncond, noise_cond = noise_all.chunk(2, dim=0)
        else:
            noise_uncond, noise_cond = None, noise_all

        for idx, st in enumerate(states):
            t = st["timesteps"][int(st["step"])]
            cond_i = noise_cond[idx : idx + 1]
            if do_cfg:
                uncond_i = noise_uncond[idx : idx + 1]
                scale = float(st["spec"]["guidance_scale"])
                noise_i = uncond_i + scale * (cond_i - uncond_i)
            else:
                noise_i = cond_i
            st["latents"] = st["scheduler"].step(noise_i, t, st["latents"], return_dict=False)[0]
            st["step"] = int(st["step"]) + 1

        return {
            "sequence_packed": False,
            "vector_timesteps": [float(t.item()) for t in t_vec],
            "batch_size": batch_size,
        }

    def advance_batch(self, request):
        """Advance 1..8 active image states by ``steps`` (default 1) in a shared MM-DiT forward pass.

        Supports:
        - Late joins and early exits (requests at different step indices ``step_m`` via 1D vector ``timestep``)
        - Heterogeneous prompts and seeds per request
        - Heterogeneous resolutions ``(height_m, width_m)`` via ``forward_packed_sd35`` (Token Sequence Packing)
        - Bubble-free side-stream H2D embedding prefetch for newly joined requests
        """
        state_ids = request.get("state_ids")
        if not isinstance(state_ids, list) or not 1 <= len(state_ids) <= 8:
            raise ValueError("advance_batch requires 1 to 8 state_ids")
        seen = set()
        states = []
        for sid in state_ids:
            if not isinstance(sid, str) or sid in seen or sid not in self.states:
                raise ValueError("advance_batch requires distinct active state_ids")
            seen.add(sid)
            st = self.states[sid]
            if st.get("output") is not None or int(st.get("step", 0)) >= int(st.get("steps", st["spec"]["steps"])):
                raise ValueError(f"state {sid} has already completed")
            states.append(st)

        steps_to_run = positive_int(request.get("steps", 1), "steps")
        options = self.config.get("options", {})
        bubble_free = bool(request.get("bubble_free", options.get("bubble_free", False)))
        force_packing = bool(request.get("sequence_packing", options.get("sequence_packing", False)))
        attn_mode = str(request.get("attention", self.attention))

        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()

        with self.torch.inference_mode(), _sdpa_context(self.torch, attn_mode):
            getattr(self.xdit, "bind_global", lambda: None)()
            for st in states:
                st["steps"] = int(st.get("steps", st["spec"]["steps"]))
                st["step"] = int(st.get("step", 0))
                self._init_step_state(st, bubble_free=bubble_free)

            step_meta = {}
            for _ in range(steps_to_run):
                active = [st for st in states if int(st["step"]) < int(st["steps"])]
                if not active:
                    break
                step_meta = self._step_members_once(active, force_packing=force_packing)

        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        heights = [int(st["spec"]["height"]) for st in states]
        widths = [int(st["spec"]["width"]) for st in states]
        raw_pipe = getattr(self.pipe, "module", self.pipe)
        raw_tf = _unwrap_transformer(getattr(raw_pipe, "transformer", None))
        vae_scale = int(getattr(raw_pipe, "vae_scale_factor", 8))
        patch_size = int(getattr(getattr(raw_tf, "config", None), "patch_size", 2))
        pack_summary = packing_stats(heights, widths, vae_scale_factor=vae_scale, patch_size=patch_size)

        results = []
        for sid, st in zip(state_ids, states):
            st["paused"] = int(st["step"]) < int(st["steps"])
            results.append(
                dict(
                    state_id=sid,
                    progress=progress(int(st["step"]), int(st["steps"])),
                    paused=st["paused"],
                )
            )

        return dict(
            results=results,
            batch_size=len(state_ids),
            quantum="continuous_batch",
            model_ms=round(elapsed_ms, 3),
            attention=attn_mode,
            bubble_free=bubble_free,
            sequence_packing=bool(step_meta.get("sequence_packed", False)),
            packing=pack_summary,
            step_meta=step_meta,
            memory_after=self._memory(),
        )

    def advance(self, request):
        state = self.state(request)
        if state["output"] is not None:
            raise ValueError("pipeline has already completed")
        options = self.config.get("options", {})
        if request.get("step_mode", False) or options.get("execution") == "step":
            batch_res = self.advance_batch(
                dict(
                    id=request.get("id", 0),
                    state_ids=[request["state_id"]],
                    steps=int(request.get("steps", 1)),
                    attention=request.get("attention", self.attention),
                    bubble_free=request.get("bubble_free", options.get("bubble_free", False)),
                    sequence_packing=request.get("sequence_packing", options.get("sequence_packing", False)),
                )
            )
            item = batch_res["results"][0]
            return dict(
                state_id=request["state_id"],
                progress=item["progress"],
                paused=item["paused"],
                model_ms=batch_res["model_ms"],
                attention=batch_res["attention"],
                atomic_pipeline=False,
                engine=self.xdit.info(),
            )
        spec = state["spec"]
        self._enable_cache(spec)
        attn_mode = str(request.get("attention", self.attention))
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        observer = None
        if options.get("profile", False):
            from .profiling import PipelineProfiler

            observer = PipelineProfiler(self.torch, self.pipe, self.device)
        bubble_free = bool(request.get("bubble_free", options.get("bubble_free", False)))
        stream = self._pipeline_manager().encode_stream if bubble_free else None
        with self.torch.inference_mode(), _sdpa_context(self.torch, attn_mode):
            getattr(self.xdit, "bind_global", lambda: None)()
            arguments = self._arguments(spec)
            encoding = self._encode(arguments, stream=stream)
            with observer or nullcontext():
                if observer:
                    arguments.update(
                        callback_on_step_end=observer.callback,
                        callback_on_step_end_tensor_inputs=["latents"],
                    )
                state["output"] = getattr(self.pipe(**arguments), self.output_field)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter() - started) * 1000
        state["step"] = int(spec["steps"])
        state["steps"] = int(spec["steps"])
        return dict(
            state_id=request["state_id"],
            progress=progress(1, 1),
            model_ms=elapsed,
            attention=attn_mode,
            atomic_pipeline=True,
            batch_size=spec["batch_size"],
            seed_protocol="single_cpu_generator_for_batch",
            engine=self.xdit.info(),
            **encoding,
            **(observer.report(spec["steps"]) if observer else {}),
        )

    def finish(self, request):
        state = self.state(request)
        decode_ms = 0.0
        if state["output"] is None and state.get("latents") is not None:
            if int(state.get("step", 0)) < int(state.get("steps", state["spec"]["steps"])):
                raise ValueError("pipeline has not completed all denoising steps")
            if self.device.startswith("cuda") and self.torch.cuda.is_available():
                self.torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            options = self.config.get("options", {})
            bubble_free = bool(request.get("bubble_free", options.get("bubble_free", False)))
            stream = self._pipeline_manager().vae_stream if bubble_free else None
            ctx = (
                self.torch.cuda.stream(stream)
                if (stream is not None and hasattr(self.torch, "cuda"))
                else nullcontext()
            )
            with ctx:
                state["output"] = self._decode_latents_to_np(state["latents"])
                if stream is not None and hasattr(stream, "synchronize"):
                    stream.synchronize()
            if self.device.startswith("cuda") and self.torch.cuda.is_available():
                self.torch.cuda.synchronize(self.device)
            decode_ms = (time.perf_counter() - t0) * 1000.0
            state["latents"] = None
        res = super().finish(request)
        if decode_ms > 0.0:
            res["model_ms"] = round(decode_ms, 3)
            res["decode_ms"] = round(decode_ms, 3)
        return res

    def run_packed_batch(self, request):
        """Execute a (possibly heterogeneous-resolution) micro-batch using Token Sequence Packing (`cu_seqlens`)."""
        prompts = list(request.get("prompts") or [])
        if not prompts:
            raise ValueError("run_packed_batch requires a non-empty prompts list")
        batch_size = len(prompts)
        neg_prompts = request.get("negative_prompts", request.get("negative_prompt", ""))
        if isinstance(neg_prompts, list):
            neg_list = [str(n) for n in neg_prompts]
            if len(neg_list) != batch_size:
                raise ValueError("negative_prompts length must match prompts length")
        else:
            neg_list = [str(neg_prompts or "")] * batch_size

        seeds = request.get("seeds")
        if seeds is None:
            base_seed = int(request.get("seed", 0))
            seeds = [base_seed + i for i in range(batch_size)]
        else:
            seeds = [int(s) for s in seeds]
            if len(seeds) != batch_size:
                raise ValueError("seeds length must match prompts length")

        default_h = int(request.get("height", self.defaults["height"]))
        default_w = int(request.get("width", self.defaults["width"]))
        heights = request.get("heights")
        widths = request.get("widths")
        h_list = [positive_int(h, "height") for h in heights] if isinstance(heights, list) else [positive_int(default_h, "height")] * batch_size
        w_list = [positive_int(w, "width") for w in widths] if isinstance(widths, list) else [positive_int(default_w, "width")] * batch_size
        if len(h_list) != batch_size or len(w_list) != batch_size:
            raise ValueError("heights and widths length must match prompts length")

        default_steps = int(request.get("steps", self.defaults["steps"]))
        steps_arg = request.get("steps_list")
        steps_list = [positive_int(s, "steps") for s in steps_arg] if isinstance(steps_arg, list) else [positive_int(default_steps, "steps")] * batch_size
        if len(steps_list) != batch_size:
            raise ValueError("steps_list length must match prompts length")

        guidance_scale = float(request.get("guidance_scale", self.defaults["guidance_scale"]))
        max_seq_len = positive_int(request.get("max_sequence_length", self.defaults["max_sequence_length"]), "max_sequence_length")
        attn_mode = str(request.get("attention", self.attention))
        bubble_free = bool(request.get("bubble_free", self.config.get("options", {}).get("bubble_free", False)))

        if request.get("prestage_pool", True):
            for p_str, n_str in zip(prompts, neg_list):
                self._ensure_staged(p_str, n_str, max_seq_len, guidance_scale > 1.0)

        raw_pipe = getattr(self.pipe, "module", self.pipe)
        raw_tf = _unwrap_transformer(getattr(raw_pipe, "transformer", None))
        vae_scale = int(getattr(raw_pipe, "vae_scale_factor", 8))
        patch_size = int(getattr(getattr(raw_tf, "config", None), "patch_size", 2))
        pack_summary = packing_stats(h_list, w_list, vae_scale_factor=vae_scale, patch_size=patch_size)

        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()

        ephemeral_states = []
        for i in range(batch_size):
            st = {
                "spec": {
                    "prompt": str(prompts[i]),
                    "negative_prompt": str(neg_list[i]),
                    "seed": int(seeds[i]),
                    "height": int(h_list[i]),
                    "width": int(w_list[i]),
                    "steps": int(steps_list[i]),
                    "guidance_scale": guidance_scale,
                    "max_sequence_length": max_seq_len,
                    "batch_size": 1,
                    "frames": 1,
                },
                "step": 0,
                "steps": int(steps_list[i]),
                "embeds": None,
                "latents": None,
                "scheduler": None,
                "timesteps": None,
            }
            ephemeral_states.append(st)

        with self.torch.inference_mode(), _sdpa_context(self.torch, attn_mode):
            getattr(self.xdit, "bind_global", lambda: None)()
            for st in ephemeral_states:
                self._init_step_state(st, bubble_free=bubble_free)

            max_steps = max(steps_list)
            for _ in range(max_steps):
                active = [st for st in ephemeral_states if int(st["step"]) < int(st["steps"])]
                if not active:
                    break
                self._step_members_once(active, force_packing=True)

            # Decode each item's latents (on side stream if bubble_free=True)
            vae_stream = self._pipeline_manager().vae_stream if bubble_free else None
            ctx = (
                self.torch.cuda.stream(vae_stream)
                if (vae_stream is not None and hasattr(self.torch, "cuda"))
                else nullcontext()
            )
            images = []
            with ctx:
                for st in ephemeral_states:
                    decoded = self._decode_latents_to_np(st["latents"])
                    images.append(decoded[0] if getattr(decoded, "ndim", 0) == 4 else decoded)
                if vae_stream is not None and hasattr(vae_stream, "synchronize"):
                    vae_stream.synchronize()

        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        hits = sum(1 for st in ephemeral_states if (st.get("encode_meta") or {}).get("encoder_pool_hit"))
        push_ms = sum(float((st.get("encode_meta") or {}).get("encoder_pool_push_ms", 0.0)) for st in ephemeral_states)
        result = dict(
            batch_size=batch_size,
            attention=attn_mode,
            steps=max(steps_list),
            reused_steps=0,
            height=h_list[0],
            width=w_list[0],
            heights=h_list,
            widths=w_list,
            sequence_packed=True,
            packing=pack_summary,
            bubble_free=bubble_free,
            model_ms=round(elapsed_ms, 3),
            per_image_ms=round(elapsed_ms / batch_size, 3),
            throughput_img_per_s=round(batch_size / max(elapsed_ms / 1000.0, 1e-9), 4),
            encoder_pool_hit=(hits == batch_size),
            encoder_pool_hits=hits,
            encoder_pool_push_ms=round(push_ms, 4),
            distinct_prompts=len(set(zip(prompts, neg_list))),
        )
        if request.get("return_images", False):
            result["images"] = images
        return result

    def run_micro_batch(self, request):
        """Execute a dynamic micro-batch of distinct prompts with per-request seeds and Flash/Math SDPA."""
        options = self.config.get("options", {})
        if (
            request.get("sequence_packing", options.get("sequence_packing", False))
            or isinstance(request.get("heights"), list)
            or isinstance(request.get("widths"), list)
        ):
            return self.run_packed_batch(request)

        prompts = list(request.get("prompts") or [])
        if not prompts:
            raise ValueError("run_micro_batch requires a non-empty prompts list")
        batch_size = len(prompts)
        neg_prompts = request.get("negative_prompts", request.get("negative_prompt", ""))
        if isinstance(neg_prompts, list):
            neg_list = [str(n) for n in neg_prompts]
            if len(neg_list) != batch_size:
                raise ValueError("negative_prompts length must match prompts length")
        else:
            neg_list = [str(neg_prompts or "")] * batch_size

        seeds = request.get("seeds")
        if seeds is None:
            base_seed = int(request.get("seed", 0))
            seeds = [base_seed + i for i in range(batch_size)]
        else:
            seeds = [int(s) for s in seeds]
            if len(seeds) != batch_size:
                raise ValueError("seeds length must match prompts length")

        height = positive_int(request.get("height", self.defaults["height"]), "height")
        width = positive_int(request.get("width", self.defaults["width"]), "width")
        steps = positive_int(request.get("steps", self.defaults["steps"]), "steps")
        guidance_scale = float(request.get("guidance_scale", self.defaults["guidance_scale"]))
        max_seq_len = positive_int(request.get("max_sequence_length", self.defaults["max_sequence_length"]), "max_sequence_length")
        attn_mode = str(request.get("attention", self.attention))
        bubble_free = bool(request.get("bubble_free", options.get("bubble_free", False)))
        if request.get("vae_channels_last", False):
            vae = getattr(self.pipe.module, "vae", None)
            if vae is not None and not getattr(self, "_vae_channels_last", False):
                vae.to(memory_format=self.torch.channels_last)
                self._vae_channels_last = True
        transformer_wrapper = getattr(self.pipe.module, "transformer", None)
        configure_sd35_fast_cache(transformer_wrapper, request, steps)

        # Optionally pre-stage all prompts before starting the GPU denoising timer
        if request.get("prestage_pool", True):
            for p_str, n_str in zip(prompts, neg_list):
                self._ensure_staged(p_str, n_str, max_seq_len, guidance_scale > 1.0)

        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        generators = [self.torch.Generator(device="cpu").manual_seed(s) for s in seeds]
        arguments = dict(
            prompt=prompts,
            negative_prompt=neg_list,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_seq_len,
            output_type="np",
            generator=generators,
        )
        stream = self._pipeline_manager().encode_stream if bubble_free else None
        with self.torch.inference_mode(), _sdpa_context(self.torch, attn_mode):
            getattr(self.xdit, "bind_global", lambda: None)()
            encoding = self._encode(arguments, stream=stream)
            images = getattr(self.pipe(**arguments), self.output_field)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        reused_steps = pop_sd35_reused_steps(transformer_wrapper)
        result = dict(
            batch_size=batch_size,
            attention=attn_mode,
            steps=steps,
            reused_steps=reused_steps,
            height=height,
            width=width,
            model_ms=round(elapsed_ms, 3),
            per_image_ms=round(elapsed_ms / batch_size, 3),
            throughput_img_per_s=round(batch_size / (elapsed_ms / 1000.0), 4),
            **encoding,
        )
        if request.get("return_images", False):
            result["images"] = images
        return result

    def generate_batch(self, request):
        members = request.get("requests")
        if not isinstance(members, list) or not 1 <= len(members) <= 8:
            raise ValueError("generate_batch requires 1 to 8 requests")
        specs, ids = [], set()
        for member in members:
            key = member.get("request_id")
            if not isinstance(key, str) or not key or key in ids or key in self.states:
                raise ValueError("batch request IDs must be distinct and inactive")
            if member.get("output_mode", "file") not in OUTPUT_MODES:
                raise ValueError("invalid batch output_mode")
            spec = self._spec(member)
            if spec["batch_size"] != 1 or spec.get("request_id", key) != key:
                raise ValueError("each batch member must identify exactly one image")
            spec["request_id"] = key
            specs.append(spec)
            ids.add(key)

        options = self.config.get("options", {})
        sequence_packing = bool(request.get("sequence_packing", options.get("sequence_packing", False)))
        bubble_free = bool(request.get("bubble_free", options.get("bubble_free", False)))
        ignored = {"request_id", "prompt", "negative_prompt", "seed"}
        if sequence_packing:
            ignored = ignored | {"height", "width"}
        common = {key: value for key, value in specs[0].items() if key not in ignored}
        if any({key: value for key, value in spec.items() if key not in ignored} != common for spec in specs[1:]):
            raise ValueError("batch members require compatible generation specifications")
        if options.get("cache"):
            raise ValueError("request batching cannot combine pipeline cache-dit with MM-DiT cache")

        hetero_res = len({(s["height"], s["width"]) for s in specs}) > 1
        batch_call = dict(
            common,
            prompts=[spec["prompt"] for spec in specs],
            negative_prompts=[spec["negative_prompt"] for spec in specs],
            seeds=[spec["seed"] for spec in specs],
            prestage_pool=False,
            return_images=True,
            bubble_free=bubble_free,
        )
        if sequence_packing or hetero_res:
            batch_call.update(
                sequence_packing=True,
                heights=[spec["height"] for spec in specs],
                widths=[spec["width"] for spec in specs],
            )
            measured = self.run_packed_batch(batch_call)
        else:
            measured = self.run_micro_batch(batch_call)

        images = measured.pop("images")
        if len(images) != len(members):
            raise RuntimeError("model output count differs from batch request count")
        results = []
        for index, (member, spec) in enumerate(zip(members, specs)):
            output = self.persist(
                images[index : index + 1],
                dict(
                    id=request["id"],
                    state_id=member["request_id"],
                    output_mode=member.get("output_mode", "file"),
                ),
            )
            results.append(
                dict(
                    request_id=member["request_id"],
                    output=output,
                    output_count=1,
                    progress=progress(1, 1),
                    generation=dict(
                        spec,
                        model=self.config.get("model_path"),
                        revision=self.config.get("revision"),
                        batch_protocol="packed_requests" if (sequence_packing or hetero_res) else "distinct_requests",
                        seed_protocol="per_request_cpu_generator",
                        engine=self.xdit.info(),
                        approximate=bool(measured.get("reused_steps", 0)),
                    ),
                )
            )
        self.release_cache()
        return dict(
            results=results,
            service=dict(
                total_ms=measured["model_ms"],
                batch_size=len(results),
                sequence_packed=bool(measured.get("sequence_packed", False)),
                bubble_free=bubble_free,
            ),
            model_ms=measured["model_ms"],
            execution=measured,
            qualified=False,
        )

    def _cache_target(self):
        from cache_dit.caching.block_adapters.block_registers import BlockAdapterRegister

        adapter = BlockAdapterRegister.get_adapter(self.pipe.module, blocks_name="transformer_blocks")
        if adapter is None:
            raise ValueError("cache-dit has no SD3.5 pipeline adapter")
        return adapter

    def _spec(self, request):
        spec = super()._spec(request)
        positive_int(spec["max_sequence_length"], "max_sequence_length")
        if spec["frames"] != 1:
            raise ValueError("SD3.5 requires frames=1")
        if spec["max_sequence_length"] > 512:
            raise ValueError("SD3.5 max_sequence_length cannot exceed the T5 tokenizer limit")
        return spec

    def _arguments(self, spec):
        return dict(
            super()._arguments(spec),
            negative_prompt=[spec["negative_prompt"]] * spec["batch_size"],
            max_sequence_length=spec["max_sequence_length"],
        )
