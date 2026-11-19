#!/usr/bin/env python3
"""Challenge 1 & Motivation 1 Pre-Experiment on 8x A100-PCIE-40GB (PHB):
1. Pure Sequence Parallelism (SP1, SP2, SP4, SP8) scaling curves, speedups, and communication fractions
   (both direct CUDA-event collective timing and effective scaling overhead) for Image (SD3.5-Medium)
   and Video (Wan2.2-TI2V-5B & Wan2.1-T2V-1.3B).
2. CFG Splitting (CFG2, CFG2SP2, CFG2SP4) vs. 1-GPU Fused/Sequential CFG & Continuous Batching:
   Why Video splits CFG across 2 GPUs (CFG2 / CFG2xSP), whereas Image stays on 1-GPU SP1
   (fused batch=2 CFG + continuous batching & cu_seqlens sequence packing).
"""

import argparse
import datetime
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.nn as nn

SD35_PATH = os.environ.get("DITSERVE_SD35_MODEL_PATH", "/shared/data/DITSERVE/models/sd3.5-medium")
WAN22_PATH = os.environ.get("DITSERVE_WAN22_MODEL_PATH", "/shared/data/DITSERVE/models/wan2.2-ti2v-5b")
WAN21_PATH = os.environ.get("DITSERVE_WAN21_MODEL_PATH", "/shared/data/DITSERVE/models/wan")

PROMPT = "A cinematic wide shot of a snow leopard walking across a high mountain ridge at sunrise, crisp detail."
NEG_PROMPT = "blurry, low quality, distorted, watermark, oversaturated, static"


class CollectiveTimer:
    """Hooks Ulysses all_to_all and CFG/SP all_gather to count calls, bytes, and optional CUDA-event ms."""

    def __init__(self):
        self.sync_events = False
        self.reset()
        self._installed = False

    def reset(self):
        self.a2a_calls = 0
        self.a2a_bytes = 0
        self.a2a_cuda_ms = 0.0
        self.ag_calls = 0
        self.ag_bytes = 0
        self.ag_cuda_ms = 0.0

    def install(self):
        if self._installed:
            return
        import xfuser.model_executor.layers.usp as usp_mod

        orig_sdpa_a2a = usp_mod._sdpa_all_to_all_single
        orig_ag_into = dist.all_gather_into_tensor
        orig_ag = dist.all_gather
        timer = self

        def wrapped_sdpa_a2a(x):
            timer.a2a_calls += 1
            timer.a2a_bytes += int(x.numel() * x.element_size())
            if timer.sync_events:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                out = orig_sdpa_a2a(x)
                ev1.record()
                torch.cuda.synchronize()
                timer.a2a_cuda_ms += float(ev0.elapsed_time(ev1))
                return out
            return orig_sdpa_a2a(x)

        def wrapped_ag_into(output_tensor, input_tensor, group=None, async_op=False):
            timer.ag_calls += 1
            timer.ag_bytes += int(output_tensor.numel() * output_tensor.element_size())
            if timer.sync_events and not async_op:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                out = orig_ag_into(output_tensor, input_tensor, group=group, async_op=async_op)
                ev1.record()
                torch.cuda.synchronize()
                timer.ag_cuda_ms += float(ev0.elapsed_time(ev1))
                return out
            return orig_ag_into(output_tensor, input_tensor, group=group, async_op=async_op)

        def wrapped_ag(tensor_list, tensor, group=None, async_op=False):
            timer.ag_calls += 1
            timer.ag_bytes += int(len(tensor_list) * tensor.numel() * tensor.element_size())
            if timer.sync_events and not async_op:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                out = orig_ag(tensor_list, tensor, group=group, async_op=async_op)
                ev1.record()
                torch.cuda.synchronize()
                timer.ag_cuda_ms += float(ev0.elapsed_time(ev1))
                return out
            return orig_ag(tensor_list, tensor, group=group, async_op=async_op)

        usp_mod._sdpa_all_to_all_single = wrapped_sdpa_a2a
        dist.all_gather_into_tensor = wrapped_ag_into
        dist.all_gather = wrapped_ag
        self._installed = True

    def snapshot_per_step(self, steps: int):
        s = max(1, int(steps))
        comm_cuda_ms = (self.a2a_cuda_ms + self.ag_cuda_ms) / s
        return {
            "a2a_calls_per_step": round(self.a2a_calls / s, 2),
            "ag_calls_per_step": round(self.ag_calls / s, 2),
            "a2a_mib_per_step": round((self.a2a_bytes / s) / (1024.0 * 1024.0), 3),
            "ag_mib_per_step": round((self.ag_bytes / s) / (1024.0 * 1024.0), 3),
            "comm_mib_per_step": round(((self.a2a_bytes + self.ag_bytes) / s) / (1024.0 * 1024.0), 3),
            "a2a_cuda_ms_per_step": round(self.a2a_cuda_ms / s, 3),
            "ag_cuda_ms_per_step": round(self.ag_cuda_ms / s, 3),
            "comm_cuda_ms_per_step": round(comm_cuda_ms, 3),
        }


def patch_xfuser_sd3_processor():
    """Ensure xFuserJointAttnProcessor2_0 initializes joint_strategy=None when encoder_hidden_states is None
    (required by SD3.5-Medium dual-attention blocks 0..12 attn2 under SP > 1) and skips PipeFusion KV cache
    when pp_degree == 1."""
    import xfuser.model_executor.layers.attention_processor as ap_mod
    from xfuser.core.distributed import (
        get_pipeline_parallel_world_size,
        get_sequence_parallel_world_size,
        get_runtime_state,
    )
    from xfuser.core.cache_manager.cache_manager import get_cache_manager
    from xfuser.model_executor.layers.usp import USP

    cls = ap_mod.xFuserJointAttnProcessor2_0
    HAS_LONG_CTX_ATTN = ap_mod.HAS_LONG_CTX_ATTN
    _joint_sp_padding_attention_kwargs = ap_mod._joint_sp_padding_attention_kwargs

    def patched_call(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask=None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        if encoder_hidden_states is not None:
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states).view(
                batch_size, -1, attn.heads, head_dim
            )
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states).view(
                batch_size, -1, attn.heads, head_dim
            )
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states).view(
                batch_size, -1, attn.heads, head_dim
            )
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
        else:
            encoder_hidden_states_query_proj = None
            encoder_hidden_states_key_proj = None
            encoder_hidden_states_value_proj = None

        if not self.use_long_ctx_attn_kvcache and get_pipeline_parallel_world_size() > 1:
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=1,
                layer_type="attn",
            )

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            attention_kwargs = None
            joint_strategy = None
            if encoder_hidden_states is not None:
                if get_runtime_state().split_text_embed_in_sp:
                    attention_kwargs = _joint_sp_padding_attention_kwargs(query, encoder_hidden_states_query_proj)
                    query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
                    key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
                    value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)
                    encoder_hidden_states_query_proj = None
                    encoder_hidden_states_key_proj = None
                    encoder_hidden_states_value_proj = None
                    joint_strategy = None
                else:
                    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.transpose(1, 2)
                    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.transpose(1, 2)
                    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.transpose(1, 2)
                    joint_strategy = "rear"

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            hidden_states = USP(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                joint_query=encoder_hidden_states_query_proj,
                joint_key=encoder_hidden_states_key_proj,
                joint_value=encoder_hidden_states_value_proj,
                joint_strategy=joint_strategy,
                attn_layer=attn,
                attention_kwargs=attention_kwargs,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        else:
            if encoder_hidden_states is not None:
                query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states = USP(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                head_balance_layer=attn,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if encoder_hidden_states is not None:
            if context_input_ndim == 4:
                encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )
            return hidden_states, encoder_hidden_states
        return hidden_states

    cls.__call__ = patched_call


class SubgroupCoordinatorManager:
    """Pre-creates all NCCL/Gloo groups across 8 ranks and switches (cfg_degree, sp_degree) dynamically."""

    def __init__(self, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.nccl_groups = {}
        self.gloo_groups = {}

        all_tuples = set()
        for r in range(world_size):
            all_tuples.add((r,))
        # Contiguous groups for SP1, SP2, SP4, SP8 and CFG2/CFG2SP2/CFG2SP4 worlds
        for deg in (2, 4, 8):
            if deg <= world_size:
                for start in range(0, world_size - deg + 1, deg):
                    all_tuples.add(tuple(range(start, start + deg)))
        # Strided 2-rank CFG groups for CFG2 (sp=1), CFG2SP2 (sp=2), CFG2SP4 (sp=4)
        for sp_deg in (1, 2, 4):
            if 2 * sp_deg <= world_size:
                for off in range(sp_deg):
                    all_tuples.add((off, off + sp_deg))

        for ranks in sorted(all_tuples, key=lambda x: (len(x), x)):
            if len(ranks) == 1:
                self.nccl_groups[ranks] = None
            else:
                self.nccl_groups[ranks] = dist.new_group(list(ranks), backend="nccl")
            self.gloo_groups[ranks] = None

    def _make_coord(self, ranks, *, sequence=False, pipeline=False, ulysses_degree=None, ring_degree=1):
        from xfuser.core.distributed.group_coordinator import (
            GroupCoordinator,
            PipelineGroupCoordinator,
            SequenceParallelGroupCoordinator,
        )

        ranks = tuple(ranks)
        cls = (
            SequenceParallelGroupCoordinator
            if sequence
            else PipelineGroupCoordinator
            if pipeline
            else GroupCoordinator
        )
        coord = cls.__new__(cls)
        coord.rank = self.rank
        coord.local_rank = self.rank
        coord.ranks = ranks
        coord.world_size = len(ranks)
        coord.rank_in_group = ranks.index(self.rank)
        coord.device_group = self.nccl_groups[ranks]
        coord.cpu_group = self.gloo_groups[ranks]
        coord.shm_broadcaster = None
        coord.device = self.device
        if sequence:
            deg = len(ranks)
            r_deg = int(ring_degree or 1)
            u_deg = int(ulysses_degree or (deg // r_deg))
            coord.ulysses_group = self.nccl_groups[ranks]
            coord.ring_group = self.nccl_groups[(self.rank,)]
            coord.ulysses_world_size = u_deg
            coord.ulysses_rank = coord.rank_in_group
            coord.ring_world_size = 1
            coord.ring_rank = 0
        if pipeline:
            coord.cpu_groups = []
            coord.device_groups = []
            coord.recv_buffer_set = False
            coord.dtype = None
            coord.num_pipefusion_patches = 1
            coord.skip_tensor_recv_buffer_set = False
            coord.skip_device_group = coord.device_group
            coord.reset_buffer()
        return coord

    def activate_layout(self, cfg_degree: int, sp_degree: int, runtime_state=None):
        from xfuser.core.distributed import parallel_state as ps
        from ditserve.models.xdit.process_group import PROCESS_GROUP

        total_gpus = cfg_degree * sp_degree
        if total_gpus == 1:
            world_ranks = (self.rank,)
            cfg_ranks = (self.rank,)
            sp_ranks = (self.rank,)
        else:
            assert self.rank < total_gpus, f"rank {self.rank} >= total_gpus {total_gpus}"
            world_ranks = tuple(range(total_gpus))
            if cfg_degree == 1:
                cfg_ranks = (self.rank,)
                sp_ranks = tuple(range(sp_degree))
            else:
                cfg_role = self.rank // sp_degree
                sp_off = self.rank % sp_degree
                cfg_ranks = (sp_off, sp_off + sp_degree)
                sp_ranks = tuple(range(cfg_role * sp_degree, (cfg_role + 1) * sp_degree))

        ps._WORLD = self._make_coord(world_ranks)
        ps._DP = self._make_coord((self.rank,))
        ps._CFG = self._make_coord(cfg_ranks)
        ps._SP = self._make_coord(sp_ranks, sequence=True, ulysses_degree=sp_degree, ring_degree=1)
        ps._PP = self._make_coord((self.rank,), pipeline=True)
        ps._TP = self._make_coord((self.rank,))
        ps._FS = self._make_coord((self.rank,))
        ps._MODEL_REPLICA = self._make_coord(sp_ranks)
        ps._DIT = self.nccl_groups[world_ranks]
        ps._VAE = None

        PROCESS_GROUP.ULYSSES_PG = ps._SP.ulysses_group
        PROCESS_GROUP.RING_PG = ps._SP.ring_group
        PROCESS_GROUP.ALL_TO_ALL = None

        if runtime_state is not None:
            cfg = runtime_state.parallel_config
            cfg.world_size = total_gpus
            cfg.dit_parallel_size = total_gpus
            cfg.dp_degree = 1
            cfg.dp_config.dp_degree = 1
            cfg.dp_config.dit_parallel_size = total_gpus
            cfg.cfg_degree = cfg_degree
            cfg.dp_config.cfg_degree = cfg_degree
            cfg.sp_degree = sp_degree
            cfg.ulysses_degree = sp_degree
            cfg.ring_degree = 1
            cfg.sp_config.sp_degree = sp_degree
            cfg.sp_config.ulysses_degree = sp_degree
            cfg.sp_config.ring_degree = 1
            cfg.sp_config.dit_parallel_size = total_gpus
            runtime_state.ready = False


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())


def run_sd35_experiments(rank: int, world_size: int, device: torch.device, mgr: SubgroupCoordinatorManager, timer: CollectiveTimer):
    """Run all Image (SD3.5-Medium) experiments on 8x A100-PCIE-40GB."""
    from diffusers import StableDiffusion3Pipeline
    from diffusers.models.embeddings import PatchEmbed
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift, retrieve_timesteps
    from xfuser.core.distributed.runtime_state import get_runtime_state
    from xfuser.core.distributed import (
        get_sequence_parallel_world_size,
        get_sp_group,
    )
    from ditserve.models.xdit.runtime import engine_config
    from ditserve.models.xdit.sd35 import pipeline_class, transformer_class
    from ditserve.models.sd35_batch import forward_packed_sd35, packing_stats

    patch_xfuser_sd3_processor()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Load SD3.5-Medium with T5-XXL on CPU so GPU holds Transformer (4.70 GiB) + VAE (0.16 GiB) + CLIP-L/G (0.93 GiB)
    from transformers import T5EncoderModel

    t0 = time.perf_counter()
    t5 = T5EncoderModel.from_pretrained(SD35_PATH, subfolder="text_encoder_3", torch_dtype=torch.bfloat16, local_files_only=True)
    raw_pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_PATH,
        text_encoder_3=t5,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    raw_pipe.transformer.to(device=device, dtype=torch.bfloat16)
    raw_pipe.vae.to(device=device, dtype=torch.bfloat16)
    raw_pipe.text_encoder.to(device=device, dtype=torch.bfloat16)
    raw_pipe.text_encoder_2.to(device=device, dtype=torch.bfloat16)
    raw_pipe.text_encoder_3.to(device="cpu", dtype=torch.bfloat16)
    raw_pipe.set_progress_bar_config(disable=True)

    # Activate SP8 temporarily while constructing xFuserSD3Transformer2DWrapper so _wrap_layers wraps pos_embed, attn, attn2 on all 8 ranks
    mgr.activate_layout(1, world_size)
    eng_cfg = engine_config("stable-diffusion-3.5-medium", world_size=world_size, attention="flash", dtype=torch.bfloat16)
    eng_cfg.parallel_config.sp_degree = world_size
    eng_cfg.parallel_config.ulysses_degree = world_size
    eng_cfg.parallel_config.sp_config.sp_degree = world_size
    eng_cfg.parallel_config.sp_config.ulysses_degree = world_size

    sd35_cls = pipeline_class()
    xpipe = sd35_cls(raw_pipe, eng_cfg)
    # Ensure all xFuserJointAttnProcessor2_0 instances have use_long_ctx_attn_kvcache = True (no PipeFusion KV cache)
    for mod in xpipe.transformer.modules():
        if hasattr(mod, "use_long_ctx_attn_kvcache"):
            mod.use_long_ctx_attn_kvcache = True

    rstate = get_runtime_state()
    mgr.activate_layout(1, 1, rstate)
    load_ms = round((time.perf_counter() - t0) * 1e3, 1)
    model_vram_gib = round(torch.cuda.memory_allocated(device) / (1024.0**3), 3)

    # Pre-encode prompt once (measuring CPU T5 + GPU CLIP-L/G latency)
    def encode_sd35_prompt(prompt_str, neg_str, max_seq_len=256):
        torch.cuda.synchronize(device)
        t_enc0 = time.perf_counter()
        with torch.inference_mode():
            t5_pos = raw_pipe._get_t5_prompt_embeds(
                prompt=prompt_str, num_images_per_prompt=1, max_sequence_length=max_seq_len, device="cpu"
            ).to(dtype=torch.bfloat16)
            t5_neg = raw_pipe._get_t5_prompt_embeds(
                prompt=neg_str, num_images_per_prompt=1, max_sequence_length=max_seq_len, device="cpu"
            ).to(dtype=torch.bfloat16)
            clip_l_pos, pooled_l_pos = raw_pipe._get_clip_prompt_embeds(
                prompt=prompt_str, device=device, num_images_per_prompt=1, clip_skip=None, clip_model_index=0
            )
            clip_g_pos, pooled_g_pos = raw_pipe._get_clip_prompt_embeds(
                prompt=prompt_str, device=device, num_images_per_prompt=1, clip_skip=None, clip_model_index=1
            )
            clip_l_neg, pooled_l_neg = raw_pipe._get_clip_prompt_embeds(
                prompt=neg_str, device=device, num_images_per_prompt=1, clip_skip=None, clip_model_index=0
            )
            clip_g_neg, pooled_g_neg = raw_pipe._get_clip_prompt_embeds(
                prompt=neg_str, device=device, num_images_per_prompt=1, clip_skip=None, clip_model_index=1
            )
            clip_pos = torch.cat([clip_l_pos, clip_g_pos], dim=-1)
            clip_pos = torch.nn.functional.pad(clip_pos, (0, int(t5_pos.shape[-1]) - int(clip_pos.shape[-1])))
            pos_pe = torch.cat([clip_pos.to(device=device, dtype=torch.bfloat16), t5_pos.to(device=device, dtype=torch.bfloat16)], dim=-2)
            pos_ppe = torch.cat([pooled_l_pos, pooled_g_pos], dim=-1).to(device=device, dtype=torch.bfloat16)

            clip_neg = torch.cat([clip_l_neg, clip_g_neg], dim=-1)
            clip_neg = torch.nn.functional.pad(clip_neg, (0, int(t5_neg.shape[-1]) - int(clip_neg.shape[-1])))
            neg_pe = torch.cat([clip_neg.to(device=device, dtype=torch.bfloat16), t5_neg.to(device=device, dtype=torch.bfloat16)], dim=-2)
            neg_ppe = torch.cat([pooled_l_neg, pooled_g_neg], dim=-1).to(device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize(device)
        enc_ms = (time.perf_counter() - t_enc0) * 1e3
        return pos_pe, neg_pe, pos_ppe, neg_ppe, enc_ms

    # Warmup encode on all ranks + measure 3 runs on rank 0
    pos_pe, neg_pe, pos_ppe, neg_ppe, ems0 = encode_sd35_prompt(PROMPT, NEG_PROMPT)
    enc_times = [ems0]
    if rank == 0:
        for _ in range(3):
            _, _, _, _, ems = encode_sd35_prompt(PROMPT, NEG_PROMPT)
            enc_times.append(ems)
    encode_ms_median = round(statistics.median(enc_times), 2)
    dist.barrier()
    if rank == 0:
        print(f"[Progress][SD3.5] Loaded & encoded ok (encode_ms={encode_ms_median:.2f}ms, vram={model_vram_gib:.2f}GiB)", flush=True)

    out_sd35 = {
        "model": "SD3.5-Medium",
        "heads": 24,
        "blocks": 24,
        "dual_attn_blocks": 13,
        "attn_layers_per_step": 37,
        "steps": 28,
        "dtype": "bfloat16",
        "load_ms": load_ms,
        "model_vram_gib": model_vram_gib,
        "encode_split_cpu_t5_gpu_clip_ms": encode_ms_median,
        "single_gpu_cfg_batching": {},
        "parallel_scaling": {},
        "continuous_batching_sp1": {},
    }

    # -------------------------------------------------------------------------
    # Experiment 1A: 1-GPU Batch=1 (single branch) vs 2xBatch=1 vs Batch=2 (fused CFG)
    # -------------------------------------------------------------------------
    if rank == 0:
        mgr.activate_layout(1, 1, rstate)
        for res in (512, 1024, 1536):
            h_lat, w_lat = res // 8, res // 8
            tokens = (h_lat // 2) * (w_lat // 2)
            rstate.ready = False
            rstate.set_input_parameters(
                height=res, width=res, batch_size=1, num_inference_steps=28,
                split_text_embed_in_sp=False, split_latents_by_rows=True,
            )
            gen = torch.Generator(device=device).manual_seed(42)
            lat1 = torch.randn((1, 16, h_lat, w_lat), generator=gen, device=device, dtype=torch.bfloat16)
            lat2 = torch.cat([lat1, lat1], dim=0)
            pe2 = torch.cat([neg_pe, pos_pe], dim=0)
            ppe2 = torch.cat([neg_ppe, pos_ppe], dim=0)
            t_val = torch.tensor([500.0], device=device, dtype=torch.bfloat16)
            t_val2 = t_val.expand(2)

            with torch.inference_mode():
                # Warmup
                for _ in range(2):
                    _ = xpipe.transformer(hidden_states=lat1, timestep=t_val, encoder_hidden_states=pos_pe, pooled_projections=pos_ppe, return_dict=False)[0]
                    _ = xpipe.transformer(hidden_states=lat2, timestep=t_val2, encoder_hidden_states=pe2, pooled_projections=ppe2, return_dict=False)[0]

                # 1. Single branch (batch=1)
                torch.cuda.reset_peak_memory_stats(device)
                b1_times = []
                for _ in range(6):
                    torch.cuda.synchronize(device)
                    ts0 = time.perf_counter()
                    _ = xpipe.transformer(hidden_states=lat1, timestep=t_val, encoder_hidden_states=pos_pe, pooled_projections=pos_ppe, return_dict=False)[0]
                    torch.cuda.synchronize(device)
                    b1_times.append((time.perf_counter() - ts0) * 1e3)
                b1_peak_gib = round(torch.cuda.max_memory_allocated(device) / (1024.0**3), 3)

                # 2. Sequential 2x batch=1 (uncond then cond)
                seq2_times = []
                for _ in range(6):
                    torch.cuda.synchronize(device)
                    ts0 = time.perf_counter()
                    out_u = xpipe.transformer(hidden_states=lat1, timestep=t_val, encoder_hidden_states=neg_pe, pooled_projections=neg_ppe, return_dict=False)[0]
                    out_c = xpipe.transformer(hidden_states=lat1, timestep=t_val, encoder_hidden_states=pos_pe, pooled_projections=pos_ppe, return_dict=False)[0]
                    _ = out_u + 4.5 * (out_c - out_u)
                    torch.cuda.synchronize(device)
                    seq2_times.append((time.perf_counter() - ts0) * 1e3)

                # 3. Fused batch=2 (standard 1-GPU SP1 CFG)
                torch.cuda.reset_peak_memory_stats(device)
                b2_times = []
                for _ in range(6):
                    torch.cuda.synchronize(device)
                    ts0 = time.perf_counter()
                    out_b2 = xpipe.transformer(hidden_states=lat2, timestep=t_val2, encoder_hidden_states=pe2, pooled_projections=ppe2, return_dict=False)[0]
                    ou, oc = out_b2.chunk(2)
                    _ = ou + 4.5 * (oc - ou)
                    torch.cuda.synchronize(device)
                    b2_times.append((time.perf_counter() - ts0) * 1e3)
                b2_peak_gib = round(torch.cuda.max_memory_allocated(device) / (1024.0**3), 3)

                b1_ms = statistics.median(b1_times)
                seq2_ms = statistics.median(seq2_times)
                b2_ms = statistics.median(b2_times)
                out_sd35["single_gpu_cfg_batching"][f"{res}x{res}"] = {
                    "resolution": f"{res}x{res}",
                    "tokens": tokens,
                    "batch1_single_branch_ms": round(b1_ms, 3),
                    "seq_2x_batch1_ms": round(seq2_ms, 3),
                    "fused_batch2_cfg_ms": round(b2_ms, 3),
                    "fused_b2_over_single_b1_ratio": round(b2_ms / b1_ms, 3),
                    "ideal_cfg2_compute_ceiling_speedup": round(b2_ms / b1_ms, 3),
                    "fused_vs_seq_speedup": round(seq2_ms / b2_ms, 3),
                    "batch1_peak_vram_gib": b1_peak_gib,
                    "batch2_peak_vram_gib": b2_peak_gib,
                    "activation_delta_mib": round((b2_peak_gib - b1_peak_gib) * 1024.0, 2),
                }
    dist.barrier()

    # -------------------------------------------------------------------------
    # Experiment 1B: Multi-GPU Parallel Scaling (SP1, SP2, SP4, SP8 & CFG2, CFG2SP2, CFG2SP4)
    # -------------------------------------------------------------------------
    configs = [
        ("SP1", 1, 1),
        ("SP2", 1, 2),
        ("SP4", 1, 4),
        ("SP8", 1, 8),
        ("CFG2", 2, 1),
        ("CFG2SP2", 2, 2),
        ("CFG2SP4", 2, 4),
    ]

    for res in (512, 1024):
        res_key = f"{res}x{res}"
        out_sd35["parallel_scaling"][res_key] = {}
        h_lat, w_lat = res // 8, res // 8
        tokens = (h_lat // 2) * (w_lat // 2)

        # Prepare identical initial latents and timesteps on all ranks
        gen = torch.Generator(device=device).manual_seed(1234)
        init_latents = torch.randn((1, 16, h_lat, w_lat), generator=gen, device=device, dtype=torch.bfloat16)
        mu = calculate_shift(tokens, 256, 4096, 0.5, 1.15)
        raw_sched = getattr(xpipe.scheduler, "module", xpipe.scheduler)

        ref_final_latents = None
        sp1_step_ms = None
        sp1_e2e_ms = None

        for cfg_name, cfg_deg, sp_deg in configs:
            total_gpus = cfg_deg * sp_deg
            dist.barrier()
            if rank < total_gpus:
                mgr.activate_layout(cfg_deg, sp_deg, rstate)
                rstate.set_input_parameters(
                    height=res,
                    width=res,
                    batch_size=1,
                    num_inference_steps=28,
                    split_text_embed_in_sp=False,
                    split_latents_by_rows=True,
                )
                xpipe._guidance_scale = 4.5
                xpipe._clip_skip = None
                xpipe._joint_attention_kwargs = None
                xpipe._interrupt = False

                def run_sd35_denoising_pass(num_steps: int, sync_comm: bool):
                    timesteps, _ = retrieve_timesteps(raw_sched, 28, device, None, None, mu=mu)
                    timesteps = timesteps[:num_steps]
                    pe, ppe = xpipe._process_cfg_split_batch(neg_pe, pos_pe, neg_ppe, pos_ppe)
                    lat = init_latents.clone()
                    lat, pe = xpipe._init_sync_pipeline(lat, pe)

                    timer.reset()
                    timer.sync_events = sync_comm
                    torch.cuda.synchronize(device)
                    t_loop0 = time.perf_counter()
                    step_times = []
                    for i, t in enumerate(timesteps):
                        if sync_comm:
                            torch.cuda.synchronize(device)
                        ts = time.perf_counter()
                        last_lat = lat
                        noise_pred, _ = xpipe._backbone_forward(
                            latents=lat,
                            encoder_hidden_states=pe,
                            pooled_prompt_embeds=ppe,
                            t=t,
                        )
                        lat = xpipe._scheduler_step(noise_pred, last_lat, t)
                        if sync_comm:
                            torch.cuda.synchronize(device)
                        step_times.append((time.perf_counter() - ts) * 1e3)
                    torch.cuda.synchronize(device)
                    denoise_ms = (time.perf_counter() - t_loop0) * 1e3

                    t_g0 = time.perf_counter()
                    if get_sequence_parallel_world_size() > 1:
                        sp_lat_list = get_sp_group().all_gather(lat, separate_tensors=True)
                        lat_list = []
                        for pp_patch_idx in range(rstate.num_pipeline_patch):
                            lat_list += [
                                sp_lat_list[sp_idx][
                                    :,
                                    :,
                                    rstate.pp_patches_start_idx_local[pp_patch_idx] : rstate.pp_patches_start_idx_local[
                                        pp_patch_idx + 1
                                    ],
                                    :,
                                ]
                                for sp_idx in range(sp_deg)
                            ]
                        lat = torch.cat(lat_list, dim=-2)
                    torch.cuda.synchronize(device)
                    sp_gather_ms = (time.perf_counter() - t_g0) * 1e3
                    return lat, denoise_ms, sp_gather_ms, step_times, timer.snapshot_per_step(num_steps)

                with torch.inference_mode():
                    # Warmup (2 steps)
                    _ = run_sd35_denoising_pass(2, False)
                    # Pass A: Unperturbed full 28-step run
                    final_lat, denoise_28_ms, sp_gather_ms, _, comm_counts = run_sd35_denoising_pass(28, False)
                    # VAE decode on rank 0
                    vae_ms = 0.0
                    if rank == 0:
                        torch.cuda.synchronize(device)
                        tv0 = time.perf_counter()
                        lat_vae = (final_lat / xpipe.vae.config.scaling_factor) + xpipe.vae.config.shift_factor
                        _ = xpipe.vae.decode(lat_vae, return_dict=False)[0]
                        torch.cuda.synchronize(device)
                        vae_ms = (time.perf_counter() - tv0) * 1e3
                    # Pass B: CUDA-event collective breakdown (5 steady-state steps)
                    _, denoise_sync_5_ms, _, _, comm_timed = run_sd35_denoising_pass(5, True)

                if rank == 0:
                    step_ms = denoise_28_ms / 28.0
                    e2e_ms = encode_ms_median + denoise_28_ms + sp_gather_ms + vae_ms
                    if cfg_name == "SP1":
                        ref_final_latents = final_lat.detach().clone()
                        sp1_step_ms = step_ms
                        sp1_e2e_ms = e2e_ms
                    cos_sim = cosine_sim(final_lat, ref_final_latents) if ref_final_latents is not None else 1.0

                    comm_cuda_ms = comm_timed["comm_cuda_ms_per_step"]
                    step_sync_ms = denoise_sync_5_ms / 5.0
                    comp_cuda_ms = max(0.0, step_sync_ms - comm_cuda_ms)
                    comm_cuda_pct = (comm_cuda_ms / step_sync_ms * 100.0) if (total_gpus > 1 and step_sync_ms > 0) else 0.0

                    # Effective communication/scaling overhead vs ideal 1/P scaling of SP1
                    ideal_step_ms = sp1_step_ms / total_gpus
                    comm_eff_ms = max(0.0, step_ms - ideal_step_ms) if total_gpus > 1 else 0.0
                    comm_eff_pct = (comm_eff_ms / step_ms * 100.0) if (total_gpus > 1 and step_ms > 0) else 0.0

                    gpu_s = total_gpus * (e2e_ms / 1000.0)
                    out_sd35["parallel_scaling"][res_key][cfg_name] = {
                        "config": cfg_name,
                        "gpus": total_gpus,
                        "cfg_degree": cfg_deg,
                        "sp_degree": sp_deg,
                        "tokens_global": tokens,
                        "tokens_per_gpu": tokens // sp_deg,
                        "step_ms": round(step_ms, 3),
                        "denoise_28steps_ms": round(denoise_28_ms, 2),
                        "sp_final_gather_ms": round(sp_gather_ms, 3),
                        "vae_decode_ms": round(vae_ms, 2),
                        "encode_ms": encode_ms_median,
                        "e2e_ms": round(e2e_ms, 2),
                        "step_speedup_vs_sp1": round(sp1_step_ms / step_ms, 3),
                        "e2e_speedup_vs_sp1": round(sp1_e2e_ms / e2e_ms, 3),
                        "parallel_efficiency_e2e_pct": round((sp1_e2e_ms / e2e_ms) / total_gpus * 100.0, 1),
                        "comm_cuda_ms_per_step": round(comm_cuda_ms, 3),
                        "comp_cuda_ms_per_step": round(comp_cuda_ms, 3),
                        "comm_cuda_pct": round(comm_cuda_pct, 2),
                        "comm_eff_ms_per_step": round(comm_eff_ms, 3),
                        "comm_eff_pct": round(comm_eff_pct, 2),
                        "a2a_calls_per_step": comm_counts["a2a_calls_per_step"],
                        "ag_calls_per_step": comm_counts["ag_calls_per_step"],
                        "comm_mib_per_step": comm_counts["comm_mib_per_step"],
                        "gpu_seconds_per_image": round(gpu_s, 3),
                        "throughput_img_s_per_gpu": round(1.0 / gpu_s, 4),
                        "two_gpu_pair_throughput_img_s": round((2.0 / total_gpus) / (e2e_ms / 1000.0), 4) if total_gpus <= 2 else None,
                        "cosine_sim_vs_sp1": round(cos_sim, 6),
                    }
                    if rank == 0:
                        print(
                            f"[Progress][SD3.5] {res_key} {cfg_name} (gpus={total_gpus}): "
                            f"step={step_ms:.2f}ms e2e={e2e_ms:.1f}ms speedup={sp1_e2e_ms/e2e_ms:.2f}x "
                            f"cuda_comm={comm_cuda_pct:.1f}% eff_comm={comm_eff_pct:.1f}% cos={cos_sim:.5f}",
                            flush=True,
                        )
            dist.barrier()

    # -------------------------------------------------------------------------
    # Experiment 1C: 1-GPU Continuous Batching + cu_seqlens Packing (B = 1, 2, 4)
    # -------------------------------------------------------------------------
    if rank == 0:
        mgr.activate_layout(1, 1, rstate)
        raw_tf = getattr(xpipe.transformer, "module", xpipe.transformer)
        # Unwrap pos_embed/attn temporarily or use forward_packed_sd35 which unwraps internally
        for res in (512, 1024):
            res_key = f"{res}x{res}"
            out_sd35["continuous_batching_sp1"][res_key] = {}
            h_lat, w_lat = res // 8, res // 8
            tokens = (h_lat // 2) * (w_lat // 2)
            rstate.ready = False
            rstate.set_input_parameters(
                height=res, width=res, batch_size=1, num_inference_steps=28,
                split_text_embed_in_sp=False, split_latents_by_rows=True,
            )
            b1_step_ms_ref = None
            b1_e2e_per_img_ref = None
            for batch_imgs in (1, 2, 4):
                # Each image has 2 CFG branches -> 2 * batch_imgs sequences
                lat_b = torch.randn((2 * batch_imgs, 16, h_lat, w_lat), device=device, dtype=torch.bfloat16)
                pe_b = torch.cat([neg_pe, pos_pe], dim=0).repeat(batch_imgs, 1, 1)
                ppe_b = torch.cat([neg_ppe, pos_ppe], dim=0).repeat(batch_imgs, 1)
                t_b = torch.tensor([500.0] * (2 * batch_imgs), device=device, dtype=torch.bfloat16)

                with torch.inference_mode():
                    for _ in range(2):
                        _ = xpipe.transformer(hidden_states=lat_b, timestep=t_b, encoder_hidden_states=pe_b, pooled_projections=ppe_b, return_dict=False)[0]
                    step_ts = []
                    for _ in range(5):
                        torch.cuda.synchronize(device)
                        ts0 = time.perf_counter()
                        _ = xpipe.transformer(hidden_states=lat_b, timestep=t_b, encoder_hidden_states=pe_b, pooled_projections=ppe_b, return_dict=False)[0]
                        torch.cuda.synchronize(device)
                        step_ts.append((time.perf_counter() - ts0) * 1e3)
                    batch_step_ms = statistics.median(step_ts)
                    per_img_step_ms = batch_step_ms / batch_imgs
                    vae_single_ms = out_sd35["parallel_scaling"][res_key]["SP1"]["vae_decode_ms"]
                    batch_e2e_ms = batch_imgs * encode_ms_median + 28.0 * batch_step_ms + batch_imgs * vae_single_ms
                    per_img_e2e_s = (batch_e2e_ms / 1000.0) / batch_imgs
                    if batch_imgs == 1:
                        b1_step_ms_ref = per_img_step_ms
                        b1_e2e_per_img_ref = per_img_e2e_s
                    out_sd35["continuous_batching_sp1"][res_key][f"B{batch_imgs}"] = {
                        "batch_images": batch_imgs,
                        "cfg_branches": 2 * batch_imgs,
                        "total_tokens_per_step": batch_imgs * tokens,
                        "batch_step_ms": round(batch_step_ms, 3),
                        "per_image_step_ms": round(per_img_step_ms, 3),
                        "per_image_gpu_seconds": round(per_img_e2e_s, 3),
                        "throughput_img_s_per_gpu": round(1.0 / per_img_e2e_s, 4),
                        "step_throughput_gain_vs_b1": round(b1_step_ms_ref / per_img_step_ms, 3),
                        "e2e_throughput_gain_vs_b1": round(b1_e2e_per_img_ref / per_img_e2e_s, 3),
                        "pcie_comm_bytes": 0,
                    }

        # Heterogeneous mixed resolution batch: [512x512, 512x512, 1024x1024] with cu_seqlens packing vs padded B=3 at 1024x1024
        with torch.inference_mode():
            lat_512 = torch.randn((1, 16, 64, 64), device=device, dtype=torch.bfloat16)
            lat_1024 = torch.randn((1, 16, 128, 128), device=device, dtype=torch.bfloat16)
            t_one = torch.tensor([500.0], device=device, dtype=torch.bfloat16)
            # 3 images (2x 512 + 1x 1024) -> 6 CFG segments
            lats_list = [lat_512, lat_512, lat_512, lat_512, lat_1024, lat_1024]
            ts_list = [t_one] * 6
            pe_list = [neg_pe, pos_pe, neg_pe, pos_pe, neg_pe, pos_pe]
            ppe_list = [neg_ppe, pos_ppe, neg_ppe, pos_ppe, neg_ppe, pos_ppe]
            for _ in range(2):
                _ = forward_packed_sd35(torch, raw_tf, lats_list, ts_list, pe_list, ppe_list)
            packed_ts = []
            for _ in range(5):
                torch.cuda.synchronize(device)
                ts0 = time.perf_counter()
                _, pmeta = forward_packed_sd35(torch, raw_tf, lats_list, ts_list, pe_list, ppe_list)
                torch.cuda.synchronize(device)
                packed_ts.append((time.perf_counter() - ts0) * 1e3)
            packed_ms = statistics.median(packed_ts)

            # Compare against naive padded batch of 3 images at 1024x1024 (6 branches x 4096 tokens = 24,576 tokens vs 12,288 packed tokens)
            rstate.ready = False
            rstate.set_input_parameters(
                height=1024, width=1024, batch_size=1, num_inference_steps=28,
                split_text_embed_in_sp=False, split_latents_by_rows=True,
            )
            lat_pad6 = torch.randn((6, 16, 128, 128), device=device, dtype=torch.bfloat16)
            pe_pad6 = torch.cat([neg_pe, pos_pe], dim=0).repeat(3, 1, 1)
            ppe_pad6 = torch.cat([neg_ppe, pos_ppe], dim=0).repeat(3, 1)
            t_pad6 = torch.tensor([500.0] * 6, device=device, dtype=torch.bfloat16)
            for _ in range(2):
                _ = xpipe.transformer(hidden_states=lat_pad6, timestep=t_pad6, encoder_hidden_states=pe_pad6, pooled_projections=ppe_pad6, return_dict=False)[0]
            pad_ts = []
            for _ in range(5):
                torch.cuda.synchronize(device)
                ts0 = time.perf_counter()
                _ = xpipe.transformer(hidden_states=lat_pad6, timestep=t_pad6, encoder_hidden_states=pe_pad6, pooled_projections=ppe_pad6, return_dict=False)[0]
                torch.cuda.synchronize(device)
                pad_ts.append((time.perf_counter() - ts0) * 1e3)
            padded_ms = statistics.median(pad_ts)
            solo_sum_ms = 2.0 * out_sd35["single_gpu_cfg_batching"]["512x512"]["fused_batch2_cfg_ms"] + out_sd35["single_gpu_cfg_batching"]["1024x1024"]["fused_batch2_cfg_ms"]

            out_sd35["continuous_batching_sp1"]["heterogeneous_2x512_1x1024"] = {
                "batch_composition": ["512x512", "512x512", "1024x1024"],
                "packed_image_tokens_per_branch": 1024 + 1024 + 4096,
                "padded_image_tokens_per_branch": 3 * 4096,
                "padding_saved_pct": 50.0,
                "packed_step_ms": round(packed_ms, 3),
                "padded_step_ms": round(padded_ms, 3),
                "solo_sequential_sum_step_ms": round(solo_sum_ms, 3),
                "speedup_vs_padded": round(padded_ms / packed_ms, 3),
                "speedup_vs_solo_sequential": round(solo_sum_ms / packed_ms, 3),
            }

    del xpipe, raw_pipe, t5
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return out_sd35


def run_wan_experiments(rank: int, world_size: int, device: torch.device, mgr: SubgroupCoordinatorManager, timer: CollectiveTimer):
    """Run Video (Wan2.2-TI2V-5B 24-head & Wan2.1-T2V-1.3B 12-head) experiments on 8x A100-PCIE-40GB."""
    from xfuser.core.distributed.runtime_state import initialize_runtime_state, get_runtime_state
    from ditserve.models.xdit.runtime import engine_config
    from ditserve.models.xdit.wan import load_wan_transformer

    out_video = {
        "wan22_ti2v_5b": {
            "model": "Wan2.2-TI2V-5B",
            "heads": 24,
            "blocks": 30,
            "steps": 20,
            "dtype": "float16",
            "single_gpu_cfg_batching": {},
            "parallel_scaling": {},
        },
        "wan21_t2v_1_3b": {
            "model": "Wan2.1-T2V-1.3B",
            "heads": 12,
            "blocks": 30,
            "steps": 50,
            "dtype": "float16",
            "single_gpu_cfg_batching": {},
        },
    }

    # -------------------------------------------------------------------------
    # 2A & 2B: Wan2.2-TI2V-5B (24 heads -> supports SP1, SP2, SP4, SP8 & CFG2, CFG2SP2, CFG2SP4!)
    # -------------------------------------------------------------------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    wan22_tf = load_wan_transformer(WAN22_PATH, torch.float16, local_files_only=True).to(device=device, dtype=torch.float16)
    wan22_tf.eval()

    class WanDummyPipeline:
        def __init__(self, transformer):
            self.transformer = transformer
            self.vae_scale_factor_spatial = 8
            self.vae_scale_factor_temporal = 4

    # Initialize xFuser runtime state for Wan
    mgr.activate_layout(1, 1)
    eng_cfg = engine_config("wan", world_size=1, attention="flash", dtype=torch.float16)
    initialize_runtime_state(pipeline=WanDummyPipeline(wan22_tf), engine_config=eng_cfg)
    rstate = get_runtime_state()
    load_ms = round((time.perf_counter() - t0) * 1e3, 1)
    model_vram_gib = round(torch.cuda.memory_allocated(device) / (1024.0**3), 3)
    out_video["wan22_ti2v_5b"]["load_ms"] = load_ms
    out_video["wan22_ti2v_5b"]["model_vram_gib"] = model_vram_gib

    # Wan2.2 shapes: latent channels = 48, temporal stride = 4, spatial stride = 16, patch_size = (1, 2, 2)
    # 121x512x896 -> F_lat = 31, H_lat = 32, W_lat = 56 -> post_patch = 31 x 16 x 28 = 13,888 tokens
    # 121x704x1280 -> F_lat = 31, H_lat = 44, W_lat = 80 -> post_patch = 31 x 22 x 40 = 27,280 tokens
    wan22_shapes = [
        ("121x512x896", 121, 512, 896, 31, 32, 56, 13888, 20),
        ("121x704x1280", 121, 704, 1280, 31, 44, 80, 27280, 20),
    ]
    in_ch_22 = int(wan22_tf.config.in_channels)

    # Experiment 2A.1: 1-GPU Batch=1 vs 2xBatch=1 vs Fused Batch=2 on Wan2.2-TI2V-5B
    if rank == 0:
        mgr.activate_layout(1, 1, rstate)
        for sname, frames, h, w, f_lat, h_lat, w_lat, tokens, steps in wan22_shapes:
            rstate.set_video_input_parameters(
                height=h, width=w, num_frames=frames, batch_size=1, num_inference_steps=steps, split_text_embed_in_sp=False
            )
            lat1 = torch.randn((1, in_ch_22, f_lat, h_lat, w_lat), device=device, dtype=torch.float16)
            enc1 = torch.randn((1, 512, 4096), device=device, dtype=torch.float16)
            ts1 = torch.tensor([500], device=device, dtype=torch.long)

            with torch.inference_mode():
                # Warmup
                _ = wan22_tf(hidden_states=lat1, timestep=ts1, encoder_hidden_states=enc1, return_dict=False)[0]
                torch.cuda.reset_peak_memory_stats(device)
                b1_ts = []
                for _ in range(3):
                    torch.cuda.synchronize(device)
                    ts0 = time.perf_counter()
                    _ = wan22_tf(hidden_states=lat1, timestep=ts1, encoder_hidden_states=enc1, return_dict=False)[0]
                    torch.cuda.synchronize(device)
                    b1_ts.append((time.perf_counter() - ts0) * 1e3)
                b1_ms = statistics.median(b1_ts)
                b1_peak = round(torch.cuda.max_memory_allocated(device) / (1024.0**3), 3)

                # Sequential 2x batch=1 (standard SP1 CFG in Wan)
                seq2_ts = []
                for _ in range(3):
                    torch.cuda.synchronize(device)
                    ts0 = time.perf_counter()
                    ou = wan22_tf(hidden_states=lat1, timestep=ts1, encoder_hidden_states=enc1, return_dict=False)[0]
                    oc = wan22_tf(hidden_states=lat1, timestep=ts1, encoder_hidden_states=enc1, return_dict=False)[0]
                    _ = ou + 5.0 * (oc - ou)
                    torch.cuda.synchronize(device)
                    seq2_ts.append((time.perf_counter() - ts0) * 1e3)
                seq2_ms = statistics.median(seq2_ts)

                # Fused batch=2 (test if it OOMs or gives 0 speedup)
                b2_ms = None
                b2_peak = None
                b2_oom = False
                try:
                    lat2 = torch.cat([lat1, lat1], dim=0)
                    enc2 = torch.cat([enc1, enc1], dim=0)
                    ts2 = torch.tensor([500, 500], device=device, dtype=torch.long)
                    torch.cuda.reset_peak_memory_stats(device)
                    _ = wan22_tf(hidden_states=lat2, timestep=ts2, encoder_hidden_states=enc2, return_dict=False)[0]
                    b2_ts = []
                    for _ in range(3):
                        torch.cuda.synchronize(device)
                        ts0 = time.perf_counter()
                        out2 = wan22_tf(hidden_states=lat2, timestep=ts2, encoder_hidden_states=enc2, return_dict=False)[0]
                        ou, oc = out2.chunk(2)
                        _ = ou + 5.0 * (oc - ou)
                        torch.cuda.synchronize(device)
                        b2_ts.append((time.perf_counter() - ts0) * 1e3)
                    b2_ms = round(statistics.median(b2_ts), 3)
                    b2_peak = round(torch.cuda.max_memory_allocated(device) / (1024.0**3), 3)
                    del lat2, enc2, ts2
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        b2_oom = True
                    else:
                        raise
                finally:
                    torch.cuda.empty_cache()

                out_video["wan22_ti2v_5b"]["single_gpu_cfg_batching"][sname] = {
                    "shape": sname,
                    "tokens": tokens,
                    "batch1_single_branch_ms": round(b1_ms, 3),
                    "seq_2x_batch1_sp1_ms": round(seq2_ms, 3),
                    "fused_batch2_cfg_ms": b2_ms,
                    "fused_batch2_oom": b2_oom,
                    "fused_b2_over_single_b1_ratio": round(b2_ms / b1_ms, 3) if b2_ms else None,
                    "fused_vs_seq_speedup": round(seq2_ms / b2_ms, 3) if b2_ms else None,
                    "batch1_peak_vram_gib": b1_peak,
                    "batch2_peak_vram_gib": b2_peak,
                }
    dist.barrier()

    # Experiment 2B: Multi-GPU Parallel Scaling on Wan2.2-TI2V-5B (SP1, SP2, SP4, SP8 & CFG2, CFG2SP2, CFG2SP4)
    configs = [
        ("SP1", 1, 1),
        ("SP2", 1, 2),
        ("SP4", 1, 4),
        ("SP8", 1, 8),
        ("CFG2", 2, 1),
        ("CFG2SP2", 2, 2),
        ("CFG2SP4", 2, 4),
    ]

    for sname, frames, h, w, f_lat, h_lat, w_lat, tokens, steps in wan22_shapes:
        out_video["wan22_ti2v_5b"]["parallel_scaling"][sname] = {}
        gen = torch.Generator(device=device).manual_seed(99)
        lat1 = torch.randn((1, in_ch_22, f_lat, h_lat, w_lat), generator=gen, device=device, dtype=torch.float16)
        enc_cond = torch.randn((1, 512, 4096), generator=gen, device=device, dtype=torch.float16)
        enc_uncond = torch.randn((1, 512, 4096), generator=gen, device=device, dtype=torch.float16)
        ts1 = torch.tensor([500], device=device, dtype=torch.long)

        sp1_step_ms = None
        for cfg_name, cfg_deg, sp_deg in configs:
            total_gpus = cfg_deg * sp_deg
            dist.barrier()
            if rank < total_gpus:
                mgr.activate_layout(cfg_deg, sp_deg, rstate)
                rstate.set_video_input_parameters(
                    height=h, width=w, num_frames=frames, batch_size=1, num_inference_steps=steps, split_text_embed_in_sp=False
                )
                cfg_group = mgr.nccl_groups[(rank % sp_deg, (rank % sp_deg) + sp_deg)] if cfg_deg == 2 else None

                def run_wan_steps(num_steps: int, sync_comm: bool):
                    timer.reset()
                    timer.sync_events = sync_comm
                    torch.cuda.synchronize(device)
                    t_s0 = time.perf_counter()
                    for _ in range(num_steps):
                        if cfg_deg == 1:
                            # Both branches run sequentially over the SP group
                            ou = wan22_tf(hidden_states=lat1, timestep=ts1, encoder_hidden_states=enc_uncond, return_dict=False)[0]
                            oc = wan22_tf(hidden_states=lat1, timestep=ts1, encoder_hidden_states=enc_cond, return_dict=False)[0]
                            _ = ou + 5.0 * (oc - ou)
                        else:
                            role = rank // sp_deg
                            enc = enc_cond if role == 0 else enc_uncond
                            pred = wan22_tf(hidden_states=lat1, timestep=ts1, encoder_hidden_states=enc, return_dict=False)[0].contiguous()
                            gathered = torch.empty((2, *pred.shape), dtype=pred.dtype, device=device)
                            dist.all_gather_into_tensor(gathered, pred, group=cfg_group)
                            oc, ou = gathered[0], gathered[1]
                            _ = ou + 5.0 * (oc - ou)
                    torch.cuda.synchronize(device)
                    elapsed_ms = (time.perf_counter() - t_s0) * 1e3
                    return elapsed_ms / num_steps, timer.snapshot_per_step(num_steps)

                with torch.inference_mode():
                    # Warmup 1 step
                    _ = run_wan_steps(1, False)
                    # Pass A: Unperturbed step timing (3 steps)
                    step_ms, comm_counts = run_wan_steps(3, False)
                    # Pass B: CUDA-event collective breakdown (2 steps)
                    step_sync_ms, comm_timed = run_wan_steps(2, True)

                if rank == 0:
                    if cfg_name == "SP1":
                        sp1_step_ms = step_ms
                    comm_cuda_ms = comm_timed["comm_cuda_ms_per_step"]
                    comp_cuda_ms = max(0.0, step_sync_ms - comm_cuda_ms)
                    comm_cuda_pct = (comm_cuda_ms / step_sync_ms * 100.0) if (total_gpus > 1 and step_sync_ms > 0) else 0.0
                    ideal_step_ms = sp1_step_ms / total_gpus
                    comm_eff_ms = max(0.0, step_ms - ideal_step_ms) if total_gpus > 1 else 0.0
                    comm_eff_pct = (comm_eff_ms / step_ms * 100.0) if (total_gpus > 1 and step_ms > 0) else 0.0

                    denoise_total_ms = step_ms * steps
                    out_video["wan22_ti2v_5b"]["parallel_scaling"][sname][cfg_name] = {
                        "config": cfg_name,
                        "gpus": total_gpus,
                        "cfg_degree": cfg_deg,
                        "sp_degree": sp_deg,
                        "tokens_global": tokens,
                        "tokens_per_gpu": tokens // sp_deg,
                        "step_ms": round(step_ms, 2),
                        "denoise_20steps_s": round(denoise_total_ms / 1000.0, 2),
                        "step_speedup_vs_sp1": round(sp1_step_ms / step_ms, 3),
                        "parallel_efficiency_pct": round((sp1_step_ms / step_ms) / total_gpus * 100.0, 1),
                        "comm_cuda_ms_per_step": round(comm_cuda_ms, 2),
                        "comp_cuda_ms_per_step": round(comp_cuda_ms, 2),
                        "comm_cuda_pct": round(comm_cuda_pct, 2),
                        "comm_eff_ms_per_step": round(comm_eff_ms, 2),
                        "comm_eff_pct": round(comm_eff_pct, 2),
                        "a2a_calls_per_step": comm_counts["a2a_calls_per_step"],
                        "ag_calls_per_step": comm_counts["ag_calls_per_step"],
                        "comm_mib_per_step": comm_counts["comm_mib_per_step"],
                        "denoise_gpu_seconds": round(total_gpus * (denoise_total_ms / 1000.0), 2),
                    }
                    if rank == 0:
                        print(
                            f"[Progress][Wan2.2] {sname} {cfg_name} (gpus={total_gpus}): "
                            f"step={step_ms:.1f}ms speedup={sp1_step_ms/step_ms:.2f}x "
                            f"cuda_comm={comm_cuda_pct:.1f}% eff_comm={comm_eff_pct:.1f}%",
                            flush=True,
                        )
            dist.barrier()

    del wan22_tf
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    # -------------------------------------------------------------------------
    # 2A.2: Wan2.1-T2V-1.3B 1-GPU Batch=1 vs 2xBatch=1 vs Fused Batch=2
    # -------------------------------------------------------------------------
    if rank == 0:
        wan21_tf = load_wan_transformer(WAN21_PATH, torch.float16, local_files_only=True).to(device=device, dtype=torch.float16)
        wan21_tf.eval()
        mgr.activate_layout(1, 1, rstate)
        in_ch_21 = int(wan21_tf.config.in_channels)
        wan21_shapes = [
            ("81x480x832", 81, 480, 832, 21, 60, 104, 32760, 50),
            ("81x720x1280", 81, 720, 1280, 21, 90, 160, 75600, 50),
        ]
        for sname, frames, h, w, f_lat, h_lat, w_lat, tokens, steps in wan21_shapes:
            rstate.set_video_input_parameters(
                height=h, width=w, num_frames=frames, batch_size=1, num_inference_steps=steps, split_text_embed_in_sp=False
            )
            lat1 = torch.randn((1, in_ch_21, f_lat, h_lat, w_lat), device=device, dtype=torch.float16)
            enc1 = torch.randn((1, 512, 4096), device=device, dtype=torch.float16)
            ts1 = torch.tensor([500], device=device, dtype=torch.long)

            with torch.inference_mode():
                _ = wan21_tf(hidden_states=lat1, timestep=ts1, encoder_hidden_states=enc1, return_dict=False)[0]
                torch.cuda.reset_peak_memory_stats(device)
                b1_ts = []
                for _ in range(3):
                    torch.cuda.synchronize(device)
                    ts0 = time.perf_counter()
                    _ = wan21_tf(hidden_states=lat1, timestep=ts1, encoder_hidden_states=enc1, return_dict=False)[0]
                    torch.cuda.synchronize(device)
                    b1_ts.append((time.perf_counter() - ts0) * 1e3)
                b1_ms = statistics.median(b1_ts)
                b1_peak = round(torch.cuda.max_memory_allocated(device) / (1024.0**3), 3)

                seq2_ms = 2.0 * b1_ms

                b2_ms = None
                b2_peak = None
                b2_oom = False
                try:
                    lat2 = torch.cat([lat1, lat1], dim=0)
                    enc2 = torch.cat([enc1, enc1], dim=0)
                    ts2 = torch.tensor([500, 500], device=device, dtype=torch.long)
                    torch.cuda.reset_peak_memory_stats(device)
                    _ = wan21_tf(hidden_states=lat2, timestep=ts2, encoder_hidden_states=enc2, return_dict=False)[0]
                    b2_ts = []
                    for _ in range(2):
                        torch.cuda.synchronize(device)
                        ts0 = time.perf_counter()
                        out2 = wan21_tf(hidden_states=lat2, timestep=ts2, encoder_hidden_states=enc2, return_dict=False)[0]
                        ou, oc = out2.chunk(2)
                        _ = ou + 5.0 * (oc - ou)
                        torch.cuda.synchronize(device)
                        b2_ts.append((time.perf_counter() - ts0) * 1e3)
                    b2_ms = round(statistics.median(b2_ts), 3)
                    b2_peak = round(torch.cuda.max_memory_allocated(device) / (1024.0**3), 3)
                    del lat2, enc2, ts2
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        b2_oom = True
                    else:
                        raise
                finally:
                    torch.cuda.empty_cache()

                out_video["wan21_t2v_1_3b"]["single_gpu_cfg_batching"][sname] = {
                    "shape": sname,
                    "tokens": tokens,
                    "batch1_single_branch_ms": round(b1_ms, 3),
                    "seq_2x_batch1_sp1_ms": round(seq2_ms, 3),
                    "fused_batch2_cfg_ms": b2_ms,
                    "fused_batch2_oom": b2_oom,
                    "fused_b2_over_single_b1_ratio": round(b2_ms / b1_ms, 3) if b2_ms else None,
                    "fused_vs_seq_speedup": round(seq2_ms / b2_ms, 3) if b2_ms else None,
                    "batch1_peak_vram_gib": b1_peak,
                    "batch2_peak_vram_gib": b2_peak,
                }
        del wan21_tf
        gc.collect()
        torch.cuda.empty_cache()

    dist.barrier()
    return out_video


def format_markdown_report(summary: dict) -> str:
    sd35 = summary["sd35"]
    wan22 = summary["video"]["wan22_ti2v_5b"]
    wan21 = summary["video"]["wan21_t2v_1_3b"]

    lines = [
        f"# Challenge 1 & Motivation 1 Pre-Experiment Report ({summary['run_id']})",
        "",
        f"- **Timestamp (UTC)**: `{summary['timestamp_utc']}`",
        f"- **Host / Hardware**: `{summary['hostname']}` — `8x {summary['gpu_name']}` (`PHB` PCIe Gen4 interconnect)",
        f"- **Build**: `{summary['build']}`",
        "",
        "## 1. Challenge 1: Pure Sequence Parallelism (`SP1, SP2, SP4, SP8`) Scaling & Communication Wall",
        "",
        "### 1.1 Image (`SD3.5-Medium`, 28 steps, 24 heads, 37 attn layers/step, BF16)",
        "",
        "| Resolution | Config | GPUs | Tokens/GPU | Step (ms) | 28-Step Denoise (ms) | VAE (ms) | E2E (ms) | Step Speedup | E2E Speedup | CUDA Comm (ms/step) | CUDA Comm (%) | Eff. Comm (%) | `all_to_all` Calls/Step | Comm (MiB/step) | GPU·s / Img |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for res_key in ("512x512", "1024x1024"):
        for cfg_name in ("SP1", "SP2", "SP4", "SP8"):
            r = sd35["parallel_scaling"][res_key][cfg_name]
            lines.append(
                f"| `{res_key}` | **{cfg_name}** | {r['gpus']} | {r['tokens_per_gpu']} | {r['step_ms']:.2f} | "
                f"{r['denoise_28steps_ms']:.1f} | {r['vae_decode_ms']:.1f} | {r['e2e_ms']:.1f} | "
                f"**{r['step_speedup_vs_sp1']:.2f}x** | **{r['e2e_speedup_vs_sp1']:.2f}x** | "
                f"{r['comm_cuda_ms_per_step']:.2f} | **{r['comm_cuda_pct']:.1f}%** | **{r['comm_eff_pct']:.1f}%** | "
                f"{r['a2a_calls_per_step']:.0f} | {r['comm_mib_per_step']:.1f} | {r['gpu_seconds_per_image']:.2f} |"
            )

    lines.extend([
        "",
        "### 1.2 Video (`Wan2.2-TI2V-5B`, 20 steps, 24 heads, 30 blocks, FP16)",
        "",
        "| Shape ($F\\times H\\times W$) | Config | GPUs | Tokens/GPU | Step (ms) | 20-Step Denoise (s) | Step Speedup | Parallel Eff. | CUDA Comm (ms/step) | CUDA Comm (%) | Eff. Comm (%) | `all_to_all` Calls/Step | Comm (MiB/step) | Denoise GPU·s |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for sname in ("121x512x896", "121x704x1280"):
        for cfg_name in ("SP1", "SP2", "SP4", "SP8"):
            r = wan22["parallel_scaling"][sname][cfg_name]
            lines.append(
                f"| `{sname}` | **{cfg_name}** | {r['gpus']} | {r['tokens_per_gpu']} | {r['step_ms']:.1f} | "
                f"{r['denoise_20steps_s']:.2f} | **{r['step_speedup_vs_sp1']:.2f}x** | {r['parallel_efficiency_pct']:.1f}% | "
                f"{r['comm_cuda_ms_per_step']:.1f} | **{r['comm_cuda_pct']:.1f}%** | **{r['comm_eff_pct']:.1f}%** | "
                f"{r['a2a_calls_per_step']:.0f} | {r['comm_mib_per_step']:.1f} | {r['denoise_gpu_seconds']:.1f} |"
            )

    lines.extend([
        "",
        "## 2. Motivation 1: Why Video Splits CFG Across 2 GPUs (`CFG2` / `CFG2xSP`) While Image Stays on 1-GPU `SP1`",
        "",
        "### 2.1 1-GPU Single-Branch (`batch=1`) vs. Sequential (`2x batch=1`) vs. Fused CFG (`batch=2`)",
        "",
        "| Modality / Model | Shape / Resolution | Tokens ($L$) | `batch=1` (1 Branch, ms) | `2x batch=1` (Seq CFG, ms) | `batch=2` (Fused CFG, ms) | `batch=2 / batch=1` Ratio | Fused vs. Seq Speedup | `batch=1` Peak VRAM (GiB) | `batch=2` Peak VRAM (GiB) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for res_key in ("512x512", "1024x1024", "1536x1536"):
        b = sd35["single_gpu_cfg_batching"][res_key]
        lines.append(
            f"| Image (`SD3.5-Medium`) | `{res_key}` | {b['tokens']} | {b['batch1_single_branch_ms']:.2f} | "
            f"{b['seq_2x_batch1_ms']:.2f} | {b['fused_batch2_cfg_ms']:.2f} | "
            f"**{b['fused_b2_over_single_b1_ratio']:.2f}x** | **{b['fused_vs_seq_speedup']:.2f}x** | "
            f"{b['batch1_peak_vram_gib']:.2f} | {b['batch2_peak_vram_gib']:.2f} |"
        )
    for sname in ("121x512x896", "121x704x1280"):
        b = wan22["single_gpu_cfg_batching"][sname]
        b2_str = "OOM" if b["fused_batch2_oom"] else f"{b['fused_batch2_cfg_ms']:.1f}"
        ratio_str = "OOM" if b["fused_batch2_oom"] else f"**{b['fused_b2_over_single_b1_ratio']:.2f}x**"
        spd_str = "OOM" if b["fused_batch2_oom"] else f"{b['fused_vs_seq_speedup']:.2f}x"
        pk2_str = "OOM" if b["fused_batch2_oom"] else f"{b['batch2_peak_vram_gib']:.2f}"
        lines.append(
            f"| Video (`Wan2.2-5B`) | `{sname}` | {b['tokens']} | {b['batch1_single_branch_ms']:.1f} | "
            f"{b['seq_2x_batch1_sp1_ms']:.1f} | {b2_str} | {ratio_str} | {spd_str} | "
            f"{b['batch1_peak_vram_gib']:.2f} | {pk2_str} |"
        )
    for sname in ("81x480x832", "81x720x1280"):
        b = wan21["single_gpu_cfg_batching"][sname]
        b2_str = "OOM" if b["fused_batch2_oom"] else f"{b['fused_batch2_cfg_ms']:.1f}"
        ratio_str = "OOM" if b["fused_batch2_oom"] else f"**{b['fused_b2_over_single_b1_ratio']:.2f}x**"
        spd_str = "OOM" if b["fused_batch2_oom"] else f"{b['fused_vs_seq_speedup']:.2f}x"
        pk2_str = "OOM" if b["fused_batch2_oom"] else f"{b['batch2_peak_vram_gib']:.2f}"
        lines.append(
            f"| Video (`Wan2.1-1.3B`) | `{sname}` | {b['tokens']} | {b['batch1_single_branch_ms']:.1f} | "
            f"{b['seq_2x_batch1_sp1_ms']:.1f} | {b2_str} | {ratio_str} | {spd_str} | "
            f"{b['batch1_peak_vram_gib']:.2f} | {pk2_str} |"
        )

    lines.extend([
        "",
        "### 2.2 Video `CFG2` & `CFG2xSP` vs. Pure `SP` (`Wan2.2-TI2V-5B`)",
        "",
        "| Shape ($F\\times H\\times W$) | Config | GPUs | Step (ms) | Step Speedup vs `SP1` | CUDA Comm (ms/step) | CUDA Comm (%) | Eff. Comm (%) | Collectives/Step | Comm (MiB/step) | Denoise GPU·s |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for sname in ("121x512x896", "121x704x1280"):
        for cfg_name in ("SP1", "SP2", "CFG2", "SP4", "CFG2SP2", "SP8", "CFG2SP4"):
            r = wan22["parallel_scaling"][sname][cfg_name]
            coll = r["a2a_calls_per_step"] + r["ag_calls_per_step"]
            lines.append(
                f"| `{sname}` | **{cfg_name}** | {r['gpus']} | {r['step_ms']:.1f} | **{r['step_speedup_vs_sp1']:.2f}x** | "
                f"{r['comm_cuda_ms_per_step']:.1f} | **{r['comm_cuda_pct']:.1f}%** | **{r['comm_eff_pct']:.1f}%** | "
                f"{coll:.0f} | {r['comm_mib_per_step']:.1f} | {r['denoise_gpu_seconds']:.1f} |"
            )

    lines.extend([
        "",
        "### 2.3 Image `CFG2` & `CFG2xSP` vs. `1-GPU SP1` & `1-GPU Continuous Batching` (`SD3.5-Medium`)",
        "",
        "| Resolution | Config / Mode | GPUs | Step (ms) | E2E (ms) | Step Speedup | E2E Speedup | CUDA Comm (%) | Eff. Comm (%) | GPU·s / Img | Throughput (img/s/GPU) | Cosine Sim vs `SP1` |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for res_key in ("512x512", "1024x1024"):
        for cfg_name in ("SP1", "SP2", "CFG2", "SP4", "CFG2SP2", "SP8", "CFG2SP4"):
            r = sd35["parallel_scaling"][res_key][cfg_name]
            lines.append(
                f"| `{res_key}` | **{cfg_name}** | {r['gpus']} | {r['step_ms']:.2f} | {r['e2e_ms']:.1f} | "
                f"{r['step_speedup_vs_sp1']:.2f}x | **{r['e2e_speedup_vs_sp1']:.2f}x** | "
                f"{r['comm_cuda_pct']:.1f}% | {r['comm_eff_pct']:.1f}% | "
                f"**{r['gpu_seconds_per_image']:.2f}** | **{r['throughput_img_s_per_gpu']:.3f}** | {r['cosine_sim_vs_sp1']:.5f} |"
            )
        for b_key in ("B2", "B4"):
            cb = sd35["continuous_batching_sp1"][res_key][b_key]
            lines.append(
                f"| `{res_key}` | **1-GPU SP1 ContBatch ({b_key})** | 1 | {cb['per_image_step_ms']:.2f}/img | "
                f"{cb['per_image_gpu_seconds']*1000.0:.1f}/img | {cb['step_throughput_gain_vs_b1']:.2f}x | "
                f"**{cb['e2e_throughput_gain_vs_b1']:.2f}x** | 0.0% | 0.0% | "
                f"**{cb['per_image_gpu_seconds']:.2f}** | **{cb['throughput_img_s_per_gpu']:.3f}** | 1.00000 |"
            )

    het = sd35["continuous_batching_sp1"]["heterogeneous_2x512_1x1024"]
    lines.extend([
        "",
        "### 2.4 Image Heterogeneous Resolution Batch (`[512x512, 512x512, 1024x1024]`) with `cu_seqlens` Packing on 1 GPU",
        "",
        f"- **Packed Step Latency (`cu_seqlens`, {het['packed_image_tokens_per_branch']} tokens/branch)**: `{het['packed_step_ms']:.2f} ms`",
        f"- **Padded Step Latency (`3x 1024x1024`, {het['padded_image_tokens_per_branch']} tokens/branch)**: `{het['padded_step_ms']:.2f} ms` (**{het['speedup_vs_padded']:.2f}x** speedup from `cu_seqlens` packing, saving `{het['padding_saved_pct']:.0f}%` padding tokens)",
        f"- **Sequential Solo Sum (`2x 512^2 + 1x 1024^2`)**: `{het['solo_sequential_sum_step_ms']:.2f} ms` (**{het['speedup_vs_solo_sequential']:.2f}x** speedup)",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-name", default="20261111-mot1-sp-cfg-study.md")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    if local_rank == 0:
        print(f"[Progress] dist.init_process_group(nccl) ok, world_size={world_size}", flush=True)

    timer = CollectiveTimer()
    timer.install()
    mgr = SubgroupCoordinatorManager(local_rank, world_size, device)
    if local_rank == 0:
        print("[Progress] SubgroupCoordinatorManager initialized ok", flush=True)

    sd35_res = run_sd35_experiments(local_rank, world_size, device, mgr, timer)
    if local_rank == 0:
        print("[Progress] SD3.5-Medium experiments complete, starting Wan experiments", flush=True)
    video_res = run_wan_experiments(local_rank, world_size, device, mgr, timer)

    if local_rank == 0:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema": "ditserve.mot1-sp-cfg-study.v1",
            "run_id": os.environ.get("DITSERVE_RUN_ID", "local"),
            "build": os.environ.get("DITSERVE_BUILD", "unknown"),
            "hostname": subprocess.check_output(["hostname"], text=True).strip(),
            "gpu_name": torch.cuda.get_device_name(device),
            "world_size": world_size,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "sd35": sd35_res,
            "video": video_res,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        report_md = format_markdown_report(summary)
        (out_dir / args.report_name).write_text(report_md)
        print(report_md, flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
