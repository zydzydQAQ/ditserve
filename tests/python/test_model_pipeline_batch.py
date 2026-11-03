"""CPU pipeline doubles validate offline batch contracts, not GPU performance."""

import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ditserve.models.base import Backend
from ditserve.models.pipeline import AtomicPipelineBackend
from ditserve.models.sd35 import SD35Backend
from ditserve.models.wan.pipeline import WanPipelineBackend


class PipelineBatchTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            import numpy
        except ImportError:
            self.skipTest("CPU torch/numpy required")
        self.torch, self.np = torch, numpy

    def backend(self, cls, config):
        backend = cls.__new__(cls)
        Backend.__init__(backend, config)
        backend.torch, backend.device = self.torch, "cpu"
        backend.text_encoder_dtype = config.get("text_encoder_dtype", "bfloat16")
        from xdit_wan_fixtures import EngineDouble
        backend.xdit = EngineDouble()
        if cls is SD35Backend:
            from ditserve.models.encoder_pool import ProactiveEncoderPool
            backend.attention = "math"
            backend.encoder_pool = ProactiveEncoderPool({})
            backend.encoders_offloaded = False
        return backend

    def test_offline_batches_repeat_prompts_share_seed_and_reencode_each_call(self):
        torch, np = self.torch, self.np
        for cls, encoder in ((AtomicPipelineBackend, "cuda"),
                             (WanPipelineBackend, "cuda"), (WanPipelineBackend, "cpu")):
            with self.subTest(backend=cls.__name__, encoder=encoder), tempfile.TemporaryDirectory() as directory:
                backend = self.backend(cls, dict(text_encoder_device=encoder, work_dir=directory))
                calls, encodings, outputs = [], [], {}

                class Pipeline:
                    transformer = SimpleNamespace(dtype=torch.float16)

                    def encode_prompt(self, **kwargs):
                        encodings.append(kwargs)
                        count = len(kwargs["prompt"])
                        return torch.ones(count, 3, dtype=kwargs["dtype"]), torch.zeros(count, 3, dtype=kwargs["dtype"])

                    def __call__(self, **kwargs):
                        calls.append(kwargs)
                        count = len(kwargs["prompt"] if "prompt" in kwargs else kwargs["prompt_embeds"])
                        value = torch.randn(count, 4, generator=kwargs["generator"]).numpy()
                        return SimpleNamespace(images=value, frames=value)

                backend.pipe = Pipeline()
                for index, (size, seed) in enumerate(((1, 7), (3, 7), (3, 7), (3, 8))):
                    key = str(index)
                    negative = "noise"
                    prepared = backend.prepare(dict(state_id=key, spec=dict(prompt="kettle", negative_prompt=negative,
                        seed=seed, batch_size=size, steps=2)))
                    self.assertEqual(prepared["batch_size"], size)
                    result = backend.advance(dict(state_id=key))
                    output = backend.finish(dict(id=index, state_id=key))
                    array = np.load(output["output"]["path"], allow_pickle=False)
                    outputs[index] = array
                    np.testing.assert_array_equal(array, torch.randn(size, 4,
                        generator=torch.Generator(device="cpu").manual_seed(seed)).numpy())
                    self.assertEqual(result["batch_size"], size)
                    self.assertEqual(output["generation"]["batch_size"], size)
                    kwargs = calls[-1]
                    self.assertIsInstance(kwargs["generator"], torch.Generator)
                    self.assertEqual(kwargs["generator"].initial_seed(), seed)
                    if cls is WanPipelineBackend:
                        self.assertEqual(kwargs["num_frames"], 81)
                    if encoder == "cpu":
                        self.assertNotIn("prompt", kwargs)
                        self.assertEqual(encodings[-1]["prompt"], ["kettle"] * size)
                        self.assertEqual(encodings[-1]["negative_prompt"], ["noise"] * size)
                        self.assertEqual(encodings[-1]["device"], "cpu")
                        self.assertEqual(encodings[-1]["dtype"], torch.bfloat16)
                        self.assertEqual(encodings[-1]["max_sequence_length"], 512)
                        self.assertEqual(kwargs["prompt_embeds"].dtype, torch.float16)
                        self.assertFalse(result["prompt_embeddings_reused"])
                    else:
                        self.assertEqual(kwargs["prompt"], ["kettle"] * size)
                        if cls is WanPipelineBackend:
                            self.assertEqual(kwargs["negative_prompt"], ["noise"] * size)
                    with self.assertRaisesRegex(ValueError, "already completed"):
                        backend.advance(dict(state_id=key))
                    backend.release(dict(state_id=key))
                np.testing.assert_array_equal(outputs[1], outputs[2])
                self.assertFalse(np.array_equal(outputs[1], outputs[3]))
                if encoder == "cpu":
                    self.assertEqual(len(encodings), 4)
        self.assertFalse(torch.cuda.is_initialized())

    def test_finish_releases_allocator_cache_without_changing_output(self):
        for cls in (AtomicPipelineBackend, SD35Backend):
            backend = self.backend(cls, {"options": {"release_cache": True}, "model_path": "fixture"})
            backend.device = "cuda:0"
            backend.torch = SimpleNamespace(cuda=Mock())
            backend.torch.cuda.is_available.return_value = True
            backend.torch.cuda.is_initialized.return_value = True
            image = self.np.ones((1, 2, 2, 3), dtype=self.np.float32)
            backend.states["image"] = {"spec": {"prompt": "kettle"}, "output": image}
            result = backend.finish(dict(id=1, state_id="image", output_mode="check"))
            self.assertEqual(result["output"]["output_mode"], "check")
            self.assertEqual(result["generation"]["model"], "fixture")
            backend.torch.cuda.empty_cache.assert_called_once_with()
            backend.config["options"]["release_cache"] = False
            backend.finish(dict(id=2, state_id="image", output_mode="check"))
            backend.torch.cuda.empty_cache.assert_called_once_with()
            self.assertIs(backend.states["image"]["output"], image)

    def test_invalid_batch_shapes_fail_before_state_creation(self):
        for cls in (AtomicPipelineBackend, SD35Backend, WanPipelineBackend):
            backend = self.backend(cls, {})
            bad = [dict(batch_size=0), dict(batch_size=True), dict(batch_size=1.5), dict(prompt=["kettle"]),
                   dict(seed=True), dict(seed=1 << 64), dict(guidance_scale=float("nan")), dict(height=17)]
            bad.append(dict(frames=2))
            if cls is SD35Backend:
                bad.extend((dict(max_sequence_length=0), dict(max_sequence_length=513)))
            for spec in bad:
                with self.subTest(backend=cls.__name__, spec=spec), self.assertRaises(ValueError):
                    backend.prepare(dict(state_id="invalid", spec=dict(dict(prompt="kettle"), **spec)))
                self.assertFalse(backend.states)

    def test_wan_loader_keeps_cpu_encoder_and_component_precisions(self):
        from ditserve.models.pipeline import _pipeline
        torch = self.torch
        calls = {}

        class Module:
            def __init__(self):
                self.to = Mock(return_value=self)

        class Wan:
            def __init__(self, vae, encoder):
                self.transformer, self.vae, self.text_encoder = Module(), vae, encoder
                self.transformer.device = "cuda:0"
                self.to = Mock(side_effect=AssertionError("must never move CPU encoder through CUDA"))
                self.set_progress_bar_config = Mock()

            @classmethod
            def from_pretrained(cls, model, **kwargs):
                calls["pipeline"] = kwargs
                return cls(kwargs["vae"], kwargs["text_encoder"])

        vae_loader, encoder_loader = Mock(return_value=Module()), Mock(return_value=Module())
        diffusers = SimpleNamespace(WanPipeline=Wan, AutoencoderKLWan=SimpleNamespace(from_pretrained=vae_loader))
        args = SimpleNamespace(model="wan", model_id="snapshot", revision="pinned", local_files_only=True,
            dtype="float16", vae_dtype="float32", text_encoder_dtype="float32", text_encoder_device="cpu", device="cuda:0")
        with patch("ditserve.models.pipeline.importlib.import_module", return_value=SimpleNamespace(
                UMT5EncoderModel=SimpleNamespace(from_pretrained=encoder_loader))):
            pipe = _pipeline(torch, diffusers, args)
        self.assertEqual(vae_loader.call_args.kwargs["torch_dtype"], torch.float32)
        self.assertEqual(encoder_loader.call_args.kwargs["torch_dtype"], torch.float32)
        self.assertEqual(calls["pipeline"]["torch_dtype"], torch.float16)
        self.assertTrue(calls["pipeline"]["local_files_only"])
        self.assertEqual(calls["pipeline"]["revision"], "pinned")
        pipe.transformer.to.assert_called_once_with(device="cuda:0", dtype=torch.float16)
        pipe.vae.to.assert_called_once_with(device="cuda:0", dtype=torch.float32)
        pipe.text_encoder.to.assert_called_once_with(device="cpu", dtype=torch.float32)
        self.assertEqual(pipe._execution_device, "cuda:0")
        self.assertFalse(torch.cuda.is_initialized())

    def test_sd35_loader_uses_the_native_sd3_pipeline(self):
        from ditserve.models.pipeline import _pipeline
        pipe = Mock()
        pipe.to.return_value = pipe
        loader = Mock(return_value=pipe)
        diffusers = SimpleNamespace(StableDiffusion3Pipeline=SimpleNamespace(from_pretrained=loader))
        args = SimpleNamespace(model="sd35", model_id="snapshot", revision="pinned", local_files_only=True,
            dtype="bfloat16", device="cpu")
        self.assertIs(_pipeline(self.torch, diffusers, args), pipe)
        self.assertEqual(loader.call_args.args, ("snapshot",))
        self.assertEqual(loader.call_args.kwargs,
                         dict(torch_dtype=self.torch.bfloat16, local_files_only=True, revision="pinned"))
        pipe.to.assert_called_once_with("cpu")
        pipe.set_progress_bar_config.assert_called_once_with(disable=True)
        self.assertFalse(self.torch.cuda.is_initialized())

    def _build_mmdit_pipeline(self):
        """Construct a pure-PyTorch SD3.5 MM-DiT pipeline compatible with SD3Transformer2DModel & FlowMatch."""
        torch = self.torch
        nn = torch.nn

        class AdaLNZero(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.silu = nn.SiLU()
                self.linear = nn.Linear(dim, 6 * dim)
                self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

            def forward(self, x, temb):
                emb = self.linear(self.silu(temb))
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
                normed = self.norm(x) * (1.0 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
                return normed, gate_msa, shift_mlp, scale_mlp, gate_mlp

        class AdaLNContinuous(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.silu = nn.SiLU()
                self.linear = nn.Linear(dim, 2 * dim)
                self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

            def forward(self, x, temb):
                emb = self.linear(self.silu(temb))
                scale, shift = emb.chunk(2, dim=1)
                return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        class JointAttn(nn.Module):
            def __init__(self, dim, heads=2, context_pre_only=False):
                super().__init__()
                self.heads = heads
                self.context_pre_only = context_pre_only
                self.to_q = nn.Linear(dim, dim)
                self.to_k = nn.Linear(dim, dim)
                self.to_v = nn.Linear(dim, dim)
                self.norm_q = nn.Identity()
                self.norm_k = nn.Identity()
                self.add_q_proj = nn.Linear(dim, dim)
                self.add_k_proj = nn.Linear(dim, dim)
                self.add_v_proj = nn.Linear(dim, dim)
                self.norm_added_q = nn.Identity()
                self.norm_added_k = nn.Identity()
                self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Identity()])
                if not context_pre_only:
                    self.to_add_out = nn.Linear(dim, dim)

            def forward(self, hidden_states, encoder_hidden_states=None):
                b, n_img, d = hidden_states.shape
                head_dim = d // self.heads
                qi = self.norm_q(self.to_q(hidden_states).view(b, n_img, self.heads, head_dim))
                ki = self.norm_k(self.to_k(hidden_states).view(b, n_img, self.heads, head_dim))
                vi = self.to_v(hidden_states).view(b, n_img, self.heads, head_dim)
                if encoder_hidden_states is not None:
                    n_txt = encoder_hidden_states.shape[1]
                    qt = self.norm_added_q(self.add_q_proj(encoder_hidden_states).view(b, n_txt, self.heads, head_dim))
                    kt = self.norm_added_k(self.add_k_proj(encoder_hidden_states).view(b, n_txt, self.heads, head_dim))
                    vt = self.add_v_proj(encoder_hidden_states).view(b, n_txt, self.heads, head_dim)
                    q = torch.cat([qi, qt], dim=1).transpose(1, 2)
                    k = torch.cat([ki, kt], dim=1).transpose(1, 2)
                    v = torch.cat([vi, vt], dim=1).transpose(1, 2)
                else:
                    q, k, v = qi.transpose(1, 2), ki.transpose(1, 2), vi.transpose(1, 2)
                out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
                out = out.transpose(1, 2).reshape(b, -1, d)
                if encoder_hidden_states is not None:
                    img_out = self.to_out[1](self.to_out[0](out[:, :n_img]))
                    txt_out = self.to_add_out(out[:, n_img:]) if not self.context_pre_only else None
                    return img_out, txt_out
                return self.to_out[1](self.to_out[0](out)), None

        class JointBlock(nn.Module):
            def __init__(self, dim, context_pre_only=False):
                super().__init__()
                self.context_pre_only = context_pre_only
                self.use_dual_attention = False
                self.norm1 = AdaLNZero(dim)
                self.norm1_context = AdaLNContinuous(dim) if context_pre_only else AdaLNZero(dim)
                self.attn = JointAttn(dim, heads=2, context_pre_only=context_pre_only)
                self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
                self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
                if not context_pre_only:
                    self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
                    self.ff_context = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

            def forward(self, hidden_states, encoder_hidden_states, temb):
                norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
                if self.context_pre_only:
                    norm_c = self.norm1_context(encoder_hidden_states, temb)
                    c_gate_msa = c_shift_mlp = c_scale_mlp = c_gate_mlp = None
                else:
                    norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, temb)
                attn_h, attn_c = self.attn(norm_h, norm_c)
                hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_h
                norm_h2 = self.norm2(hidden_states) * (1.0 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
                hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_h2)
                if not self.context_pre_only:
                    encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * attn_c
                    norm_c2 = self.norm2_context(encoder_hidden_states) * (1.0 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1)
                    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_c2)
                return encoder_hidden_states, hidden_states

        class PatchEmbedStub(nn.Module):
            def __init__(self, in_ch=4, dim=16, patch_size=2):
                super().__init__()
                self.patch_size = patch_size
                self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

            def forward(self, x):
                h, w = x.shape[-2], x.shape[-1]
                tokens = self.proj(x).flatten(2).transpose(1, 2)
                grid_y = torch.linspace(-1.0, 1.0, h // self.patch_size, device=x.device, dtype=x.dtype)
                grid_x = torch.linspace(-1.0, 1.0, w // self.patch_size, device=x.device, dtype=x.dtype)
                pos = (grid_y[:, None] + grid_x[None, :]).reshape(1, -1, 1)
                return tokens + pos

        class TimeTextEmbedStub(nn.Module):
            def __init__(self, pooled_dim=8, dim=16):
                super().__init__()
                self.t_proj = nn.Linear(1, dim)
                self.p_proj = nn.Linear(pooled_dim, dim)

            def forward(self, timestep, pooled):
                t_in = timestep.reshape(-1, 1).to(dtype=pooled.dtype) / 1000.0
                return self.t_proj(t_in) + self.p_proj(pooled)

        class MiniSD3Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(patch_size=2, in_channels=4, out_channels=4)
                self.out_channels = 4
                self.pos_embed = PatchEmbedStub(in_ch=4, dim=16, patch_size=2)
                self.context_embedder = nn.Linear(24, 16)
                self.time_text_embed = TimeTextEmbedStub(pooled_dim=8, dim=16)
                self.transformer_blocks = nn.ModuleList([
                    JointBlock(16, context_pre_only=False),
                    JointBlock(16, context_pre_only=True),
                ])
                self.norm_out = AdaLNContinuous(16)
                self.proj_out = nn.Linear(16, 2 * 2 * 4)

            def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections, return_dict=False, **kwargs):
                p = self.config.patch_size
                h, w = hidden_states.shape[-2] // p, hidden_states.shape[-1] // p
                x = self.pos_embed(hidden_states)
                c = self.context_embedder(encoder_hidden_states)
                temb = self.time_text_embed(timestep, pooled_projections)
                for blk in self.transformer_blocks:
                    c, x = blk(x, c, temb)
                x = self.norm_out(x, temb)
                x = self.proj_out(x)
                x = x.reshape(x.shape[0], h, w, p, p, self.out_channels)
                x = torch.einsum("nhwpqc->nchpwq", x)
                out = x.reshape(x.shape[0], self.out_channels, h * p, w * p)
                if not return_dict:
                    return (out,)
                return SimpleNamespace(sample=out)

        class FlowSchedulerStub:
            def __init__(self):
                self.config = {"use_dynamic_shifting": True, "base_image_seq_len": 4, "max_image_seq_len": 64,
                               "base_shift": 0.5, "max_shift": 1.15}
                self.timesteps = torch.tensor([1000.0, 500.0])
                self._step = 0

            def set_timesteps(self, steps, device="cpu", mu=0.5, **kwargs):
                base = torch.linspace(1000.0, 100.0, int(steps), device=device)
                self.timesteps = base + float(mu)
                self._step = 0

            def step(self, model_output, timestep, sample, return_dict=False):
                dt = 0.25 / (1.0 + float(timestep.reshape(()).item()) / 1000.0)
                next_sample = sample - dt * model_output
                self._step += 1
                return (next_sample,)

        class MiniSD3Pipeline:
            vae_scale_factor = 2

            def __init__(self):
                torch.manual_seed(99)
                self.transformer = MiniSD3Transformer().eval()
                self.scheduler = FlowSchedulerStub()
                self.vae = SimpleNamespace(
                    dtype=torch.float32,
                    config=SimpleNamespace(scaling_factor=0.5, shift_factor=0.1),
                    decode=lambda z, **kw: (z[:, :3].repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1),),
                )
                self.image_processor = SimpleNamespace(
                    postprocess=lambda img, output_type="np": img.detach().cpu().permute(0, 2, 3, 1).numpy().copy()
                )
                self.module = self

            def encode_prompt(self, prompt, do_classifier_free_guidance=True, **kwargs):
                prompts = prompt if isinstance(prompt, list) else [prompt]
                pe_list, ppe_list = [], []
                for p_str in prompts:
                    val = float(sum(ord(ch) for ch in str(p_str)) % 97) / 97.0 + 0.1
                    pe_list.append(torch.linspace(val, val + 0.5, 3 * 24, dtype=torch.float32).reshape(1, 3, 24))
                    ppe_list.append(torch.full((1, 8), val, dtype=torch.float32))
                pe = torch.cat(pe_list, dim=0)
                ppe = torch.cat(ppe_list, dim=0)
                if do_classifier_free_guidance:
                    return pe, -0.5 * pe, ppe, -0.5 * ppe
                return pe, None, ppe, None

            def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
                h_lat = int(height) // self.vae_scale_factor
                w_lat = int(width) // self.vae_scale_factor
                if isinstance(generator, list):
                    parts = [torch.randn((1, num_channels, h_lat, w_lat), generator=g, dtype=dtype) for g in generator]
                    return torch.cat(parts, dim=0).to(device)
                return torch.randn((batch_size, num_channels, h_lat, w_lat), generator=generator, dtype=dtype).to(device)

        return MiniSD3Pipeline()

    def test_sequence_packing_cu_seqlens_matches_solo_heterogeneous_resolutions(self):
        """Verify Optimization 3: forward_packed_sd35 matches solo forward across heterogeneous resolutions & timesteps."""
        from ditserve.models.sd35_batch import forward_packed_sd35, packing_stats

        torch = self.torch
        pipe = self._build_mmdit_pipeline()
        tf = pipe.transformer

        stats = packing_stats([512, 768], [512, 768], vae_scale_factor=8, patch_size=2)
        self.assertEqual(stats["image_token_counts"], [1024, 2304])
        self.assertEqual(stats["cu_seqlens"], [0, 1024, 3328])
        self.assertEqual(stats["padded_image_tokens"], 4608)
        self.assertEqual(stats["padding_saved_tokens"], 1280)
        self.assertTrue(stats["heterogeneous_resolutions"])

        # Heterogeneous latent shapes: (1, 4, 8, 8) [16 tokens], (1, 4, 12, 8) [24 tokens], (1, 4, 8, 16) [32 tokens]
        torch.manual_seed(314)
        latents = [
            torch.randn(1, 4, 8, 8),
            torch.randn(1, 4, 12, 8),
            torch.randn(1, 4, 8, 16),
        ]
        timesteps = [torch.tensor([900.0]), torch.tensor([600.0]), torch.tensor([300.0])]
        enc_states = [torch.randn(1, 3, 24), torch.randn(1, 5, 24), torch.randn(1, 4, 24)]
        pooled = [torch.randn(1, 8), torch.randn(1, 8), torch.randn(1, 8)]

        with torch.inference_mode():
            solo_outs = [
                tf(latents[i], timesteps[i], enc_states[i], pooled[i], return_dict=False)[0]
                for i in range(3)
            ]
            packed_outs, meta = forward_packed_sd35(torch, tf, latents, timesteps, enc_states, pooled)

        self.assertTrue(meta["sequence_packed"])
        self.assertEqual(meta["cu_seqlens_img"], [0, 16, 40, 72])
        self.assertEqual(meta["cu_seqlens_txt"], [0, 3, 8, 12])
        self.assertEqual(meta["padding_saved_tokens"], 3 * 32 - 72)
        for i in range(3):
            self.assertEqual(tuple(packed_outs[i].shape), tuple(solo_outs[i].shape))
            max_diff = float((packed_outs[i] - solo_outs[i]).abs().max().item())
            self.assertLess(max_diff, 1e-6)

    def test_sd35_continuous_batch_late_join_early_exit_and_bubble_free_pipeline(self):
        """Verify Optimizations 1, 2 & 3 on SD35Backend: encode/decode side streams, step-wise advance_batch, and resolution packing."""
        torch, np = self.torch, self.np
        with tempfile.TemporaryDirectory() as directory:
            backend = self.backend(
                SD35Backend,
                dict(
                    work_dir=directory,
                    model_path="sd35-fixture",
                    revision="pinned",
                    options=dict(continuous_batch=True, bubble_free=True, sequence_packing=True),
                ),
            )
            backend.pipe = self._build_mmdit_pipeline()
            self.addCleanup(backend.close)

            # 1. Solo reference execution for Request A (16x16, 2 steps) and Request B (32x16, 2 steps)
            spec_a = dict(prompt="mountain sunrise", negative_prompt="blur", seed=11, height=16, width=16, steps=2, guidance_scale=4.5, max_sequence_length=8)
            spec_b = dict(prompt="ocean sunset", negative_prompt="blur", seed=29, height=32, width=16, steps=2, guidance_scale=4.5, max_sequence_length=8)

            backend.prepare(dict(state_id="solo-a", spec=spec_a))
            backend.advance_batch(dict(state_ids=["solo-a"], steps=2))
            fin_solo_a = backend.finish(dict(id=1, state_id="solo-a", output_mode="file"))
            arr_solo_a = np.load(fin_solo_a["output"]["path"], allow_pickle=False)
            backend.release(dict(state_id="solo-a"))

            backend.prepare(dict(state_id="solo-b", spec=spec_b))
            backend.advance_batch(dict(state_ids=["solo-b"], steps=2))
            fin_solo_b = backend.finish(dict(id=2, state_id="solo-b", output_mode="file"))
            arr_solo_b = np.load(fin_solo_b["output"]["path"], allow_pickle=False)
            backend.release(dict(state_id="solo-b"))

            self.assertEqual(tuple(arr_solo_a.shape), (1, 16, 16, 3))
            self.assertEqual(tuple(arr_solo_b.shape), (1, 32, 16, 3))

            # 2. Bubble-free async encode + continuous batching with late join & early exit across heterogeneous resolutions
            enc_a = backend.encode_start(dict(id=10, spec=spec_a))
            polled_enc_a = backend.encode_poll(dict(ticket=enc_a["ticket"], wait=True))
            self.assertTrue(polled_enc_a["ready"])

            backend.prepare(dict(state_id="cb-a", spec=spec_a, encode_ticket=enc_a["ticket"]))
            # Round 1: cb-a advances from step 0 -> 1 alone
            r1 = backend.advance_batch(dict(state_ids=["cb-a"], steps=1, bubble_free=True))
            self.assertEqual(r1["results"][0]["progress"], dict(step=1, steps=2, done=False, next_action="advance"))

            # Request cb-b (different resolution 24x16!) arrives and joins cb-a (at step 1/2)
            backend.prepare(dict(state_id="cb-b", spec=spec_b))
            # Round 2: cb-a (step 1->2, exits) and cb-b (step 0->1, continues) execute in ONE packed MM-DiT forward!
            r2 = backend.advance_batch(dict(state_ids=["cb-a", "cb-b"], steps=1, bubble_free=True, sequence_packing=True))
            self.assertTrue(r2["sequence_packing"])
            self.assertTrue(r2["packing"]["heterogeneous_resolutions"])
            self.assertEqual(r2["results"][0]["progress"]["done"], True)
            self.assertEqual(r2["results"][1]["progress"], dict(step=1, steps=2, done=False, next_action="advance"))

            # Start async 2D-VAE decode for exiting cb-a while cb-b runs Round 3!
            dec_a = backend.decode_start(dict(id=20, state_id="cb-a", output_mode="file"))
            with self.assertRaisesRegex(ValueError, "async decode is active"):
                backend.release(dict(state_id="cb-a"))

            # Round 3: cb-b advances from step 1 -> 2
            r3 = backend.advance_batch(dict(state_ids=["cb-b"], steps=1, bubble_free=True))
            self.assertEqual(r3["results"][0]["progress"]["done"], True)

            done_a = backend.decode_poll(dict(id=21, ticket=dec_a["ticket"], wait=True))
            self.assertTrue(done_a["ready"])
            arr_cb_a = np.load(done_a["output"]["path"], allow_pickle=False)
            backend.release(dict(state_id="cb-a"))

            fin_cb_b = backend.finish(dict(id=22, state_id="cb-b", output_mode="file", bubble_free=True))
            arr_cb_b = np.load(fin_cb_b["output"]["path"], allow_pickle=False)
            backend.release(dict(state_id="cb-b"))

            np.testing.assert_allclose(arr_cb_a, arr_solo_a, atol=1e-5)
            np.testing.assert_allclose(arr_cb_b, arr_solo_b, atol=1e-5)

            # 3. Verify generate_batch with sequence_packing=True on heterogeneous resolutions
            packed_batch = backend.generate_batch(
                dict(
                    id=30,
                    sequence_packing=True,
                    bubble_free=True,
                    requests=[
                        dict(request_id="pack-a", spec=dict(spec_a, request_id="pack-a"), output_mode="file"),
                        dict(request_id="pack-b", spec=dict(spec_b, request_id="pack-b"), output_mode="file"),
                    ],
                )
            )
            self.assertTrue(packed_batch["service"]["sequence_packed"])
            self.assertTrue(packed_batch["service"]["bubble_free"])
            arr_pack_a = np.load(packed_batch["results"][0]["output"]["path"], allow_pickle=False)
            arr_pack_b = np.load(packed_batch["results"][1]["output"]["path"], allow_pickle=False)
            np.testing.assert_allclose(arr_pack_a, arr_solo_a, atol=1e-5)
            np.testing.assert_allclose(arr_pack_b, arr_solo_b, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
