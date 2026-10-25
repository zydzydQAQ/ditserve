"""Atomic SD3.5-medium image generation through the pinned xDiT pipeline."""

from .base import positive_int
from .flux import FluxBackend


class SD35Backend(FluxBackend):
    model = "sd35"
    defaults = dict(FluxBackend.defaults, steps=28, guidance_scale=4.5, max_sequence_length=256)

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
