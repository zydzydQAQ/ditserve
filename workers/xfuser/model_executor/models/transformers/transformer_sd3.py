from typing import Optional, Dict, Any, Union
import torch
import torch.distributed
import torch.nn as nn

from diffusers.models.embeddings import PatchEmbed
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import (
    is_torch_version,
    scale_lora_layers,
    USE_PEFT_BACKEND,
    unscale_lora_layers,
)

from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.logger import init_logger
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.core.distributed import is_pipeline_first_stage, is_pipeline_last_stage
from .register import xFuserTransformerWrappersRegister
from .base_transformer import xFuserTransformerBaseWrapper

logger = init_logger(__name__)


def sd3_attn_modules(transformer) -> list[nn.Module]:
    """Return every SD3 self-attention module that executes USP."""
    modules = []
    for block in transformer.transformer_blocks:
        modules.append(block.attn)
        attn2 = getattr(block, "attn2", None)
        if attn2 is not None:
            modules.append(attn2)
    return modules


@xFuserTransformerWrappersRegister.register(SD3Transformer2DModel)
class xFuserSD3Transformer2DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: SD3Transformer2DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[nn.Conv2d, PatchEmbed],
            submodule_name_to_wrap=["attn", "attn2"],
        )
        self.encoder_hidden_states_cache = [
            None for _ in range(len(self.transformer_blocks))
        ]
        register_fp8_comms_eligible_modules(self, sd3_attn_modules(self))

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        elif joint_attention_kwargs and "scale" in joint_attention_kwargs:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

        #! ---------------------------------------- MODIFIED BELOW ----------------------------------------
        # * get height & width from runtime state
        height, width = self._get_patch_height_width()

        # * only pp rank 0 needs pos_embed (patchify)
        if is_pipeline_first_stage():
            hidden_states = self.pos_embed(
                hidden_states
            )  # takes care of adding positional embeddings too.

        #! ORIGIN:
        # height, width = hidden_states.shape[-2:]
        # hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        #! ---------------------------------------- MODIFIED ABOVE ----------------------------------------
        temb = self.time_text_embed(timestep, pooled_projections)
        #! ---------------------------------------- ADD BELOW ----------------------------------------
        if is_pipeline_first_stage():
            #! ---------------------------------------- ADD ABOVE ----------------------------------------
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        fast_cache = getattr(self, "_fast_cache", None)
        if (
            fast_cache is not None
            and fast_cache.get("enabled", False)
            and not self.training
            and not get_runtime_state().patch_mode
        ):
            step = int(fast_cache.get("step", 0))
            total_steps = int(fast_cache.get("total_steps", 12))
            fn_blocks = min(int(fast_cache.get("fn_blocks", 6)), len(self.transformer_blocks))
            warmup_steps = int(fast_cache.get("warmup_steps", 4))
            threshold = float(fast_cache.get("threshold", 0.18))
            max_cont = int(fast_cache.get("max_continuous", 1))
            taylor_alpha = float(fast_cache.get("taylor_alpha", -0.10))

            h_0 = hidden_states
            for i in range(fn_blocks):
                encoder_hidden_states, hidden_states = self.transformer_blocks[i](
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                )
            front_delta = hidden_states - h_0
            can_check = (
                step >= warmup_steps
                and step < total_steps - 1
                and fast_cache.get("consecutive_skips", 0) < max_cont
                and fast_cache.get("prev_front_delta") is not None
                and fast_cache.get("prev_tail_delta") is not None
            )
            skip_tail = False
            if can_check:
                prev_fd = fast_cache["prev_front_delta"]
                # Scale-invariant cosine distance across AdaLN-Zero gated steps
                fd_flat = front_delta.flatten(1).float()
                pfd_flat = prev_fd.flatten(1).float()
                cos_sim = torch.nn.functional.cosine_similarity(fd_flat, pfd_flat, dim=1).mean()
                cos_dist = float((1.0 - cos_sim).item())
                skip_tail = cos_dist < threshold

            if skip_tail:
                prev_fd = fast_cache["prev_front_delta"]
                # Per-sample AdaLN norm-ratio scaling on cached 18-block tail residual
                cur_norm = torch.linalg.vector_norm(front_delta.float(), dim=(1, 2), keepdim=True)
                prev_norm = torch.linalg.vector_norm(prev_fd.float(), dim=(1, 2), keepdim=True).clamp(min=1e-6)
                norm_scale = torch.clamp(cur_norm / prev_norm, min=0.82, max=1.18).to(hidden_states.dtype)
                tail_est = fast_cache["prev_tail_delta"] * norm_scale
                vel = fast_cache.get("tail_velocity")
                if vel is not None and taylor_alpha != 0.0:
                    tail_est = tail_est + (taylor_alpha * vel).to(hidden_states.dtype)
                hidden_states = hidden_states + tail_est
                fast_cache["consecutive_skips"] = fast_cache.get("consecutive_skips", 0) + 1
                fast_cache["reused_steps"] = fast_cache.get("reused_steps", 0) + 1
            else:
                h_fn = hidden_states
                for i in range(fn_blocks, len(self.transformer_blocks)):
                    encoder_hidden_states, hidden_states = self.transformer_blocks[i](
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                    )
                tail_delta = hidden_states - h_fn
                if fast_cache.get("prev_tail_delta") is not None:
                    fast_cache["tail_velocity"] = (tail_delta - fast_cache["prev_tail_delta"]).detach()
                fast_cache["prev_front_delta"] = front_delta.detach()
                fast_cache["prev_tail_delta"] = tail_delta.detach()
                fast_cache["consecutive_skips"] = 0

            fast_cache["step"] = step + 1
        else:
            for i, block in enumerate(self.transformer_blocks):
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = (
                        {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    )
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )

                else:
                    if (
                        get_runtime_state().patch_mode
                        and get_runtime_state().pipeline_patch_idx == 0
                    ):
                        self.encoder_hidden_states_cache[i] = encoder_hidden_states
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                        )
                    elif get_runtime_state().patch_mode:
                        _, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=self.encoder_hidden_states_cache[i],
                            temb=temb,
                        )
                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                        )

        # * only the last pp rank needs unpatchify
        #! ---------------------------------------- ADD BELOW ----------------------------------------
        if is_pipeline_last_stage():
            #! ---------------------------------------- ADD ABOVE ----------------------------------------
            hidden_states = self.norm_out(hidden_states, temb)
            hidden_states = self.proj_out(hidden_states)

            # unpatchify
            patch_size = self.config.patch_size

            hidden_states = hidden_states.reshape(
                shape=(
                    hidden_states.shape[0],
                    height,
                    width,
                    patch_size,
                    patch_size,
                    self.out_channels,
                )
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(
                    hidden_states.shape[0],
                    self.out_channels,
                    height * patch_size,
                    width * patch_size,
                )
            )

            if USE_PEFT_BACKEND:
                # remove `lora_scale` from each PEFT layer
                unscale_lora_layers(self, lora_scale)
        #! ---------------------------------------- ADD BELOW ----------------------------------------
        else:
            output = hidden_states, encoder_hidden_states
        #! ---------------------------------------- ADD ABOVE ----------------------------------------

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
