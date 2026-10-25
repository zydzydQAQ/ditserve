"""Sana transformer and pipeline execution through the pinned xDiT wrappers."""

from .pipeline import ServingPipelineWrapper
from .runtime import XDitRuntime


def wrapper_class():
    from xfuser.model_executor.models.transformers.sana_transformer_2d import xFuserSanaTransformer2DWrapper
    return xFuserSanaTransformer2DWrapper


def pipeline_class():
    import inspect
    from diffusers.pipelines.sana.pipeline_sana import ASPECT_RATIO_1024_BIN
    from xfuser.core.distributed import get_runtime_state
    from xfuser.model_executor.pipelines.pipeline_sana_sprint import xFuserSanaSprintPipeline

    class SanaXDitSprintPipeline(ServingPipelineWrapper, xFuserSanaSprintPipeline):
        def convert_transformer(self, transformer):
            cls = wrapper_class()
            if type(transformer) is cls:
                return transformer
            from diffusers import SanaTransformer2DModel
            if type(transformer) is not SanaTransformer2DModel:
                raise ValueError("xDiT Sana requires an unmodified SanaTransformer2DModel")
            return cls(transformer)

        def activate_request(self, args, kwargs):
            values = inspect.signature(self.module.__call__).bind(*args, **kwargs)
            values.apply_defaults()
            values = values.arguments
            prompt = values.get("prompt")
            embeds = values.get("prompt_embeds")
            batch = len(prompt) if isinstance(prompt, list) else 1 if embeds is None else embeds.shape[0]
            height, width = values["height"], values["width"]
            if values["use_resolution_binning"]:
                height, width = self.image_processor.classify_height_width_bin(
                    height, width, ratios=ASPECT_RATIO_1024_BIN)
            get_runtime_state().set_input_parameters(
                height=height, width=width, batch_size=batch,
                num_inference_steps=values["num_inference_steps"],
                split_text_embed_in_sp=False, split_latents_by_rows=False)

        def encode_prompt(self, *args, **kwargs):
            return self.module.encode_prompt(*args, **kwargs)

    return SanaXDitSprintPipeline


class XDitSanaEngine:
    def __init__(self, pipeline, device, *, attention="math", groups=None):
        self.runtime = XDitRuntime(
            pipeline,
            model="sana",
            device=device,
            attention=attention,
            groups=groups,
            dtype=pipeline.transformer.dtype,
            wrapper=pipeline_class(),
        )
        self.pipe = self.runtime.pipe

    def forward(self, hidden_states, *, num_inference_steps, **kwargs):
        factor = self.pipe.vae_scale_factor
        self.runtime.activate_image(
            height=hidden_states.shape[-2] * factor,
            width=hidden_states.shape[-1] * factor,
            batch_size=hidden_states.shape[0],
            num_inference_steps=num_inference_steps,
        )
        return self.pipe.transformer(hidden_states=hidden_states, **kwargs)

    def info(self):
        return dict(self.runtime.info(), pipeline=type(self.pipe).__name__,
                    transformer=type(self.pipe.transformer).__name__)

    def close(self):
        self.runtime.close()
