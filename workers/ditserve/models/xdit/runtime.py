"""Pinned xDiT runtime construction and request-local layout activation."""

from . import XDIT_REVISION, engine_info
from .groups import ProcessGroupPool


ATTENTION_BACKENDS = {
    "auto": "sdpa",
    "math": "sdpa_math",
    "flash": "sdpa_flash",
    "efficient": "sdpa_efficient",
}


def engine_config(model, world_size=1, attention="math", dtype=None):
    import torch
    from xfuser.config.config import (
        DataParallelConfig,
        EngineConfig,
        FastAttnConfig,
        FullyShardConfig,
        ModelConfig,
        ParallelConfig,
        PipeFusionParallelConfig,
        RuntimeConfig,
        SequenceParallelConfig,
        TensorParallelConfig,
        VaeParallelConfig,
    )

    if attention not in ATTENTION_BACKENDS:
        raise ValueError("unsupported xDiT attention backend: " + str(attention))
    parallel = ParallelConfig(
        dp_config=DataParallelConfig(dp_degree=1, dit_parallel_size=1),
        sp_config=SequenceParallelConfig(ulysses_degree=1, ring_degree=1, dit_parallel_size=1),
        pp_config=PipeFusionParallelConfig(pp_degree=1, dit_parallel_size=1),
        tp_config=TensorParallelConfig(tp_degree=1, dit_parallel_size=1),
        fs_config=FullyShardConfig(fs_degree=1, dit_parallel_size=1),
        vae_config=VaeParallelConfig(use_parallel_vae=False),
        world_size=world_size,
        dit_parallel_size=1,
        vae_parallel_size=0,
    )
    parallel.pp_config.attn_layer_num_for_pp = None
    return EngineConfig(
        model_config=ModelConfig(model=model),
        runtime_config=RuntimeConfig(
            dtype=torch.float16 if dtype is None else dtype,
            attention_backend=ATTENTION_BACKENDS[attention],
        ),
        parallel_config=parallel,
        fast_attn_config=FastAttnConfig(),
    )


class XDitRuntime:
    def __init__(self, pipeline, *, model, device, attention="math", groups=None, dtype=None, wrapper=None):
        from xfuser.core.distributed.runtime_state import initialize_runtime_state, get_runtime_state

        self.pool = (ProcessGroupPool.local(device) if groups is None else
                     ProcessGroupPool.distributed(groups, device))
        self.pool.install()
        self.config = engine_config(model, self.pool.world_size, attention, dtype)
        if wrapper is None:
            initialize_runtime_state(pipeline=pipeline, engine_config=self.config)
            self.pipe = pipeline
        else:
            self.pipe = wrapper(pipeline, self.config)
        self.state = get_runtime_state()
        self.pool.activate(self.pool.current_ranks, self.state)
        self.revision = XDIT_REVISION

    def activate(self, layout, *, height, width, num_frames, batch_size=1, num_inference_steps=20):
        sp_ranks = layout.sp_ranks(self.pool.rank)
        self.pool.activate(sp_ranks, self.state)
        self.state.set_video_input_parameters(
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            split_text_embed_in_sp=False,
        )
        return sp_ranks

    def activate_image(self, *, height, width, batch_size=1, num_inference_steps=20):
        self.state.set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            split_text_embed_in_sp=False,
            split_latents_by_rows=False,
        )

    def info(self):
        return dict(engine_info(), attention=self.state.attention_backend.name,
                    sp_ranks=list(self.pool.current_ranks), pipeline=type(self.pipe).__name__,
                    transformer=type(self.pipe.transformer).__name__)

    def close(self):
        from xfuser.core.distributed import runtime_state as state

        self.pool.close()
        state._RUNTIME = None
