"""SD3.5-medium image generation with cross-modal encoder pool & dynamic batching."""

from .base import positive_int
from .encoder_pool import ProactiveEncoderPool
from .flux import FluxBackend


class SD35Backend(FluxBackend):
    model = "sd35"
    operations = FluxBackend.operations + ("encode_pool_stats", "offload_encoders", "restore_encoders")
    defaults = dict(FluxBackend.defaults, steps=28, guidance_scale=4.5, max_sequence_length=256)

    def __init__(self, config):
        super().__init__(config)
        options = config.get("options", {})
        self.encoder_pool = ProactiveEncoderPool(options.get("encoder_pool", {}))
        self.encoders_offloaded = False
        if options.get("offload_text_encoders_after_warmup", False):
            self.offload_encoders({})

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
        prompt_str = prompts[0] if isinstance(prompts, list) else str(prompts)
        neg_str = neg_prompts[0] if isinstance(neg_prompts, list) else str(neg_prompts or "")
        batch_size = len(prompts) if isinstance(prompts, list) else 1

        stage_info = self._ensure_staged(prompt_str, neg_str, max_seq_len, guidance_scale > 1.0)
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
        return dict(encoder_pool_hit=fetched["hit"], encoder_pool_push_ms=fetched["push_ms"])

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
