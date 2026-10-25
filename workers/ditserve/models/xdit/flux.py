"""The complete xDiT FLUX pipeline with DITSERVE's fixed-batch protocol."""

from .pipeline import ServingPipelineWrapper


def transformer_class():
    from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxTransformer2DWrapper
    return xFuserFluxTransformer2DWrapper


def pipeline_class():
    import inspect
    from diffusers import FluxTransformer2DModel
    from xfuser.core.distributed import get_runtime_state
    from xfuser.model_executor.pipelines.pipeline_flux import xFuserFluxPipeline

    class FluxXDitPipeline(ServingPipelineWrapper, xFuserFluxPipeline):
        def convert_transformer(self, transformer):
            cls = transformer_class()
            if type(transformer) is cls:
                return transformer
            if type(transformer) is not FluxTransformer2DModel:
                raise ValueError("xDiT FLUX requires an unmodified FluxTransformer2DModel")
            return cls(transformer)

        def activate_request(self, args, kwargs):
            values = inspect.signature(self.module.__call__).bind(*args, **kwargs)
            values.apply_defaults()
            values = values.arguments
            prompt = values.get("prompt")
            embeds = values.get("prompt_embeds")
            batch = len(prompt) if isinstance(prompt, list) else 1 if embeds is None else embeds.shape[0]
            height = values["height"] or self.default_sample_size * self.vae_scale_factor
            width = values["width"] or self.default_sample_size * self.vae_scale_factor
            get_runtime_state().set_input_parameters(
                height=height, width=width, batch_size=batch,
                num_inference_steps=values["num_inference_steps"],
                max_condition_sequence_length=values["max_sequence_length"],
                split_text_embed_in_sp=False, split_latents_by_rows=False)

    return FluxXDitPipeline


def attach(pipeline, device, *, attention="math", groups=None):
    from .runtime import XDitRuntime

    return XDitRuntime(
        pipeline,
        model="flux",
        device=device,
        attention=attention,
        groups=groups,
        dtype=pipeline.transformer.dtype,
        wrapper=pipeline_class(),
    )
