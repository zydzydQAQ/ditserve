"""SD3.5-medium image generation with cross-modal encoder pool & dynamic batching."""

from .base import OUTPUT_MODES, positive_int, progress
from .encoder_pool import ProactiveEncoderPool
from .flux import FluxBackend


class SD35Backend(FluxBackend):
    model = "sd35"
    operations = FluxBackend.operations + (
        "encode_pool_stats",
        "offload_encoders",
        "restore_encoders",
        "run_micro_batch",
        "generate_batch",
    )
    defaults = dict(FluxBackend.defaults, steps=28, guidance_scale=4.5, max_sequence_length=256)

    def __init__(self, config):
        super().__init__(config)
        options = config.get("options", {})
        self.attention = str(config.get("attention", options.get("attention", "math")))
        self.encoder_pool = ProactiveEncoderPool(options.get("encoder_pool", {}))
        self.encoders_offloaded = False
        self._install_fast_cache_hooks()
        if options.get("offload_text_encoders_after_warmup", False):
            self.offload_encoders({})

    def _install_fast_cache_hooks(self):
        """Install block-level Taylor MM-DiT fast-cache hooks on SD3Transformer2DModel.transformer_blocks."""
        transformer = getattr(self.pipe.module, "transformer", None)
        blocks = getattr(transformer, "transformer_blocks", None)
        if transformer is None or blocks is None or getattr(transformer, "_fast_cache_hooked", False):
            return
        num_blocks = len(blocks)
        if num_blocks < 8:
            return
        torch = self.torch
        F = torch.nn.functional

        for idx, blk in enumerate(blocks):
            orig_forward = blk.forward

            def _make_block_forward(block_idx, orig_fn):
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
                            rel_l1 = float(
                                ((fd_f - pfd_f).abs().mean() / pfd_f.abs().mean().clamp(min=1e-6)).item()
                            )
                            metric = min(cos_dist, 0.55 * rel_l1)
                            fc["last_metric"] = round(metric, 4)
                            skip_tail = metric < thresh
                        fc["_skip_this_step"] = skip_tail
                        return out_enc, out_hid

                    if fc.get("_skip_this_step", False):
                        if block_idx < num_blocks - 1:
                            return encoder_hidden_states, hidden_states
                        # Last block on a skipped step: apply AdaLN norm-scaled Taylor residual
                        front_delta = fc["_front_delta"]
                        prev_fd = fc["prev_front_delta"]
                        cur_norm = torch.linalg.vector_norm(front_delta.float(), dim=(1, 2), keepdim=True)
                        prev_norm = torch.linalg.vector_norm(prev_fd.float(), dim=(1, 2), keepdim=True).clamp(min=1e-6)
                        norm_scale = torch.clamp(cur_norm / prev_norm, min=0.85, max=1.15).to(hidden_states.dtype)
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

                    # Normal execution for blocks >= fn_blocks when not skipping
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

    def _prompt_key(self, prompt: str, negative_prompt: str, max_sequence_length: int) -> str:
        return self.encoder_pool.prompt_key(
            modality="t2i",
            model_id=str(self.config.get("model_path", "sd35")),
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
            dtype=str(self.config.get("dtype", "bfloat16")),
        )

    def _ensure_staged(self, prompt: str, negative_prompt: str, max_sequence_length: int, do_cfg: bool) -> dict:
        key = self._prompt_key(prompt, negative_prompt, max_sequence_length)
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

    def prepare(self, request):
        res = super().prepare(request)
        spec = self.states[res["state_id"]]["spec"]
        stage_info = self._ensure_staged(
            spec["prompt"],
            spec["negative_prompt"],
            spec["max_sequence_length"],
            spec["guidance_scale"] > 1.0,
        )
        return dict(res, encoder_pool_stage=stage_info)

    def _encode(self, arguments):
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
            fetched = self.encoder_pool.fetch_to_device(stage_info["key"], self.device)
            tensors = fetched["tensors"]
            for name in ("prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"):
                tensor = tensors.get(name)
                if tensor is not None:
                    if batch_size > 1 and tensor.shape[0] == 1:
                        tensor = tensor.repeat(batch_size, *([1] * (tensor.ndim - 1)))
                    arguments[name] = tensor
            arguments.pop("prompt", None)
            arguments.pop("negative_prompt", None)
            return dict(encoder_pool_hit=fetched["hit"], encoder_pool_push_ms=fetched["push_ms"], distinct_prompts=1)

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
            fetched = self.encoder_pool.fetch_to_device(stage_info["key"], self.device)
            hits += int(bool(fetched["hit"]))
            total_push_ms += float(fetched["push_ms"])
            for name, bucket in per_field.items():
                t = fetched["tensors"].get(name)
                if t is not None:
                    bucket.append(t)
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
        )

    def advance(self, request):
        import time
        from contextlib import nullcontext
        from .base import progress
        from .pipeline import _sdpa_context

        state = self.state(request)
        if state["output"] is not None:
            raise ValueError("pipeline has already completed")
        spec = state["spec"]
        self._enable_cache(spec)
        attn_mode = str(request.get("attention", self.attention))
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        observer = None
        if self.config.get("options", {}).get("profile", False):
            from .profiling import PipelineProfiler
            observer = PipelineProfiler(self.torch, self.pipe, self.device)
        with self.torch.inference_mode(), _sdpa_context(self.torch, attn_mode):
            self.xdit.bind_global()
            arguments = self._arguments(spec)
            encoding = self._encode(arguments)
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

    def run_micro_batch(self, request):
        """Execute a dynamic micro-batch of distinct prompts with per-request seeds and Flash/Math SDPA."""
        import time
        from .pipeline import _sdpa_context

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
        use_cache = bool(request.get("cache", False))
        cache_thresh = float(request.get("threshold", 0.10))

        if request.get("vae_channels_last", False):
            vae = getattr(self.pipe.module, "vae", None)
            if vae is not None and not getattr(self, "_vae_channels_last", False):
                vae.to(memory_format=self.torch.channels_last)
                self._vae_channels_last = True

        transformer_wrapper = getattr(self.pipe.module, "transformer", None)
        if use_cache and transformer_wrapper is not None:
            transformer_wrapper._fast_cache = {
                "enabled": True,
                "step": 0,
                "total_steps": steps,
                "fn_blocks": int(request.get("cache_fn", 6)),
                "warmup_steps": int(request.get("cache_warmup", 3)),
                "threshold": cache_thresh,
                "max_continuous": int(request.get("cache_max_continuous", 2)),
                "taylor_alpha": float(request.get("taylor_alpha", -0.15)),
                "prev_front_delta": None,
                "prev_tail_delta": None,
                "tail_velocity": None,
                "consecutive_skips": 0,
                "reused_steps": 0,
            }
        elif transformer_wrapper is not None:
            transformer_wrapper._fast_cache = None

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
        with self.torch.inference_mode(), _sdpa_context(self.torch, attn_mode):
            self.xdit.bind_global()
            encoding = self._encode(arguments)
            images = getattr(self.pipe(**arguments), self.output_field)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        reused_steps = 0
        if transformer_wrapper is not None and getattr(transformer_wrapper, "_fast_cache", None) is not None:
            reused_steps = int(transformer_wrapper._fast_cache.get("reused_steps", 0))
            transformer_wrapper._fast_cache = None
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
        ignored = {"request_id", "prompt", "negative_prompt", "seed"}
        common = {key: value for key, value in specs[0].items() if key not in ignored}
        if any({key: value for key, value in spec.items() if key not in ignored} != common for spec in specs[1:]):
            raise ValueError("batch members require compatible generation specifications")
        if self.config.get("options", {}).get("cache"):
            raise ValueError("request batching cannot combine pipeline cache-dit with MM-DiT cache")
        measured = self.run_micro_batch(dict(common,
            prompts=[spec["prompt"] for spec in specs],
            negative_prompts=[spec["negative_prompt"] for spec in specs],
            seeds=[spec["seed"] for spec in specs], prestage_pool=False, return_images=True))
        images = measured.pop("images")
        if len(images) != len(members):
            raise RuntimeError("model output count differs from batch request count")
        results = []
        for index, (member, spec) in enumerate(zip(members, specs)):
            output = self.persist(images[index:index + 1], dict(
                id=request["id"], state_id=member["request_id"],
                output_mode=member.get("output_mode", "file")))
            results.append(dict(request_id=member["request_id"], output=output, output_count=1,
                progress=progress(1, 1), generation=dict(spec,
                    model=self.config.get("model_path"), revision=self.config.get("revision"),
                    batch_protocol="distinct_requests", seed_protocol="per_request_cpu_generator",
                    engine=self.xdit.info(), approximate=bool(measured.get("reused_steps", 0)))))
        self.release_cache()
        return dict(results=results, service=dict(total_ms=measured["model_ms"], batch_size=len(results)),
                    model_ms=measured["model_ms"], execution=measured, qualified=False)

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
        return dict(super()._arguments(spec),
                    negative_prompt=[spec["negative_prompt"]] * spec["batch_size"],
                    max_sequence_length=spec["max_sequence_length"])
