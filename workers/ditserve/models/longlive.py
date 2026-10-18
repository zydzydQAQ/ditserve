"""Pinned LongLive numerical adapter; sequential isolated session state."""
import contextlib
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from .base import Backend, artifact, positive_int, progress

SOURCE_REVISION = "c6a27e771e803b1c12ffa28b82ba69b42472020e"

CHECKPOINT_REVISION = "cda9138d93872ef109d9a74c270184eb3a5acc6e"

def validate_config(config):
    options = config["direction3"]
    expected = {"height": 480, "width": 832, "fps": 16,
                "latent_frames_per_chunk": 3, "local_attn_size": 12, "sink_size": 3}
    for key, value in expected.items():
        if options.get(key, value) != value:
            raise ValueError("LongLive direction3 requires frozen %s=%s" % (key, value))
    if options.get("denoising_steps", [1000, 750, 500, 250]) != [1000, 750, 500, 250]:
        raise ValueError("denoising schedule changed")
    if config["models"]["longlive"]["revision"] != SOURCE_REVISION:
        raise ValueError("LongLive source revision is not the audited revision")
    if options.get("maximum_sessions", 2) not in (1, 2):
        raise ValueError("LongLive adapter supports at most two resident sessions")
    return options

class NativeLongLive:
    """One shared weight set, explicitly isolated state for sequential sessions."""

    VAE_STATE = ("_conv_idx", "_feat_map", "_enc_conv_idx", "_enc_feat_map")

    def __init__(self, config):
        self.options = validate_config(config)
        model = config["models"]["longlive"]
        self.source = Path(model["source_path"]).resolve()
        self.weights = Path(model["path"]).resolve()
        observed = subprocess.check_output(["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True).strip()
        if observed != SOURCE_REVISION:
            raise ValueError("LongLive source checkout changed")
        clean = subprocess.run(["git", "-C", str(self.source), "diff", "--quiet", "HEAD", "--", "LongLive1.0"])
        if clean.returncode:
            raise ValueError("LongLive tracked source differs from audited revision")
        required = ["longlive_models/models/longlive_base.pt", "longlive_models/models/lora.pt",
                    "wan_models/Wan2.1-T2V-1.3B/config.json",
                    "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                    "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json",
                    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model"]
        if any(not (self.weights / name).is_file() for name in required):
            raise FileNotFoundError("LongLive native assets are incomplete")
        import torch
        self.torch = torch
        self.device = torch.device(config.get("device", "cuda:0"))
        torch.cuda.set_device(self.device)
        self.device_index = torch.cuda.current_device()
        if "A100" not in torch.cuda.get_device_name(self.device):
            raise ValueError("LongLive adapter currently supports A100 only")
        torch.set_grad_enabled(False)
        self.old_cwd = Path.cwd()
        sys.path.insert(0, str(self.source / "LongLive1.0"))
        os.chdir(self.weights)
        try:
            from omegaconf import OmegaConf
            from unittest.mock import patch
            from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
            from wan.modules.causal_model import CausalWanModel
            from wan.modules import attention as native_attention
            from utils.lora_utils import configure_lora_for_model
            import peft
            args = OmegaConf.load(self.source / "LongLive1.0/configs/longlive_interactive_inference.yaml")
            if (args.global_sink != self.options.get("global_sink", True)
                    or args.model_kwargs.timestep_shift != self.options.get("timestep_shift", 5.0)
                    or args.context_noise != self.options.get("context_noise", 0)):
                raise ValueError("native LongLive conditioning settings disagree with frozen config")
            native_config = json.loads((self.weights / "wan_models/Wan2.1-T2V-1.3B/config.json").read_text())
            # The trained LongLive checkpoint contains the complete generator.
            # Avoid downloading/loading a second ordinary Wan transformer.
            def config_only(cls, unused_path, **kwargs):
                return cls.from_config(native_config, **kwargs)
            with patch.object(CausalWanModel, "from_pretrained", classmethod(config_only)):
                self.pipe = InteractiveCausalInferencePipeline(args, device=self.device)
            checkpoint = torch.load(self.weights / required[0], map_location="cpu", weights_only=True, mmap=True)
            self.pipe.generator.load_state_dict(checkpoint["generator"], strict=True)
            del checkpoint
            self.pipe.generator.model = configure_lora_for_model(
                self.pipe.generator.model, model_name="generator", lora_config=args.adapter,
                is_main_process=False)
            checkpoint = torch.load(self.weights / required[1], map_location="cpu", weights_only=True, mmap=True)
            loaded = peft.set_peft_model_state_dict(self.pipe.generator.model,
                checkpoint.get("generator_lora", checkpoint))
            missing_lora = [name for name in loaded.missing_keys if "lora_" in name]
            if loaded.unexpected_keys or missing_lora:
                raise ValueError("LongLive LoRA mismatch: unexpected=%r missing=%r" %
                    (loaded.unexpected_keys[:5], missing_lora[:5]))
            del checkpoint
            self.pipe.to(dtype=torch.bfloat16, device=self.device).eval().requires_grad_(False)
            self.pipe.generator.model.local_attn_size = self.pipe.local_attn_size
            self.pipe._set_all_modules_max_attention_size(self.pipe.local_attn_size)
            self.args = args
            self.sessions = {}
            torch.cuda.synchronize(self.device)
            self.identity = {"source_revision": observed, "checkpoint_revision": CHECKPOINT_REVISION,
                "device": torch.cuda.get_device_name(self.device), "torch": torch.__version__,
                "precision": "bfloat16", "shape": [480, 832], "fps": 16,
                "warped_denoising_timesteps": self.pipe.denoising_step_list.tolist(),
                "flash_attention_2_available": native_attention.FLASH_ATTN_2_AVAILABLE,
                "latent_frames_per_chunk": 3, "weights_shared": True,
                "source_path": str(self.source), "model_path": str(self.weights)}
        except BaseException:
            os.chdir(self.old_cwd)
            raise

    @contextlib.contextmanager
    def rng(self, session):
        torch = self.torch
        with torch.random.fork_rng(devices=[self.device_index]):
            torch.set_rng_state(session.cpu_rng)
            torch.cuda.set_rng_state(session.cuda_rng, self.device)
            try:
                yield
            finally:
                session.cpu_rng = torch.get_rng_state()
                session.cuda_rng = torch.cuda.get_rng_state(self.device)

    def new_session(self, session_id, seed, blocks):
        if session_id in self.sessions:
            raise ValueError("session already exists")
        if len(self.sessions) >= self.options.get("maximum_sessions", 2):
            raise ValueError("LongLive resident-session limit reached")
        torch, pipe = self.torch, self.pipe
        with torch.random.fork_rng(devices=[self.device_index]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            noise = torch.randn([1, blocks * 3, 16, 60, 104], device=self.device, dtype=torch.bfloat16)
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
        pipe._initialize_kv_cache(1, torch.bfloat16, self.device, 12 * 1560)
        pipe._initialize_crossattn_cache(1, torch.bfloat16, self.device)
        pipe.vae.model.clear_cache()
        state = SimpleNamespace(session_id=session_id, seed=seed, noise=noise, blocks=blocks,
            block=0, kv=pipe.kv_cache1, cross=pipe.crossattn_cache, history=[], conditional=None,
            prompt_version=None, block_mask=None, cpu_rng=cpu_rng, cuda_rng=cuda_rng,
            vae={name: getattr(pipe.vae.model, name) for name in self.VAE_STATE})
        self.sessions[session_id] = state
        return state

    def activate(self, session):
        self.pipe.kv_cache1, self.pipe.crossattn_cache = session.kv, session.cross
        self.pipe.generator.model.block_mask = session.block_mask
        for name, value in session.vae.items():
            setattr(self.pipe.vae.model, name, value)

    def capture(self, session):
        session.kv, session.cross = self.pipe.kv_cache1, self.pipe.crossattn_cache
        session.block_mask = self.pipe.generator.model.block_mask
        session.vae = {name: getattr(self.pipe.vae.model, name) for name in self.VAE_STATE}

    def condition(self, session, prompt, version):
        self.activate(session)
        start = time.monotonic()
        with self.torch.inference_mode(), self.rng(session):
            conditional = self.pipe.text_encoder(text_prompts=[prompt])
            self.torch.cuda.synchronize(self.device)
            encoded = time.monotonic()
            if session.block:
                history = self.torch.cat(session.history, dim=1)
                self.pipe._recache_after_switch(history, session.block * 3, conditional)
            self.torch.cuda.synchronize(self.device)
            session.conditional, session.prompt_version = conditional, version
        self.capture(session)
        return {"encode_s": encoded - start, "recache_s": time.monotonic() - encoded}

    def chunk(self, session):
        if session.conditional is None:
            raise RuntimeError("cannot denoise before an arrived prompt is encoded")
        if session.block >= session.blocks:
            raise ValueError("session is complete")
        torch, pipe = self.torch, self.pipe
        self.activate(session)
        start = time.monotonic()
        offset = session.block * 3
        with torch.inference_mode(), self.rng(session):
            noisy = session.noise[:, offset:offset + 3]
            for index, current in enumerate(pipe.denoising_step_list):
                timestep = torch.ones([1, 3], device=self.device, dtype=torch.int64) * current
                _, predicted = pipe.generator(noisy_image_or_video=noisy,
                    conditional_dict=session.conditional, timestep=timestep,
                    kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                    current_start=offset * pipe.frame_seq_length)
                if index < len(pipe.denoising_step_list) - 1:
                    next_step = pipe.denoising_step_list[index + 1]
                    noisy = pipe.scheduler.add_noise(predicted.flatten(0, 1),
                        torch.randn_like(predicted.flatten(0, 1)),
                        next_step * torch.ones([3], device=self.device, dtype=torch.long)
                    ).unflatten(0, predicted.shape[:2])
            pipe.generator(noisy_image_or_video=predicted, conditional_dict=session.conditional,
                timestep=torch.ones_like(timestep) * pipe.args.context_noise,
                kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                current_start=offset * pipe.frame_seq_length)
            torch.cuda.synchronize(self.device)
            denoised = time.monotonic()
            latent = predicted.detach().cpu()
            session.history.append(latent)
            video = (pipe.vae.decode_to_pixel(predicted, use_cache=True) * .5 + .5).clamp(0, 1)
            pixels = video[0].permute(0, 2, 3, 1).float().cpu().numpy()
            torch.cuda.synchronize(self.device)
        session.block += 1
        self.capture(session)
        return latent, pixels, {"block": session.block - 1, "prompt_version": session.prompt_version,
            "denoise_context_s": denoised - start, "decode_transfer_s": time.monotonic() - denoised,
            "total_s": time.monotonic() - start, "pixel_frames": len(pixels),
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)}

    def release(self, session):
        self.sessions.pop(session.session_id, None)
        if self.pipe.kv_cache1 is session.kv:
            self.pipe.kv_cache1 = self.pipe.crossattn_cache = None
            self.pipe.vae.model.clear_cache()
        session.kv = session.cross = session.conditional = session.noise = None
        session.history.clear()
        session.vae.clear()
        gc.collect()

    def close(self):
        for session in list(self.sessions.values()):
            self.release(session)
        del self.pipe
        gc.collect()
        self.torch.cuda.empty_cache()
        os.chdir(self.old_cwd)


class LongLiveBackend(Backend):
    operations = Backend.operations + ("update_prompt", "inventory")

    def __init__(self, config):
        super().__init__(config)
        native = dict(config)
        if "models" not in native:
            native["models"] = {"longlive": dict(source_path=config["source_path"], path=config["model_path"],
                                                 revision=config.get("revision", SOURCE_REVISION))}
        native.setdefault("direction3", config.get("options", {}))
        self.runtime = NativeLongLive(native)
        self.outputs = {}

    def prepare(self, request):
        key = self.new_id(request)
        spec = request.get("spec", {})
        state = self.runtime.new_session(key, spec.get("seed", 0), positive_int(spec.get("blocks", 16)))
        try:
            self.runtime.condition(state, spec["prompt"], spec.get("prompt_version", 0))
        except BaseException:
            self.runtime.release(state)
            raise
        self.states[key] = state
        return dict(state_id=key, progress=progress(0, state.blocks), identity=self.runtime.identity,
                    semantic_quality="unverified")

    def update_prompt(self, request):
        state = self.state(request)
        version = request.get("prompt_version", request.get("version"))
        if type(version) is not int or version <= state.prompt_version:
            raise ValueError("prompt_version must increase")
        return dict(self.runtime.condition(state, request["prompt"], version), prompt_version=version)

    def advance(self, request):
        state = self.state(request)
        latent, pixels, metrics = self.runtime.chunk(state)
        del latent
        output = self.persist(pixels[None], request)
        self.outputs[request["state_id"]] = output
        return dict(state_id=request["state_id"], progress=progress(state.block, state.blocks),
            output=output, chunk=metrics, chunk_duration_s=len(pixels) / 16.,
            frames=len(pixels), fps=16., duration_ms=len(pixels) / 16. * 1000,
            prompt_version=state.prompt_version, model_ms=metrics["total_s"] * 1000,
            semantic_quality="unverified")

    def finish(self, request):
        state = self.state(request)
        if state.block != state.blocks:
            raise ValueError("stream still has unfinished chunks")
        return dict(progress=progress(state.block, state.blocks), output=self.outputs[request["state_id"]],
                    output_scope="last_chunk; controller retains per-chunk artifacts", semantic_quality="unverified")

    def inventory(self, request):
        state = self.state(request)
        return dict(block=state.block, blocks=state.blocks, prompt_version=state.prompt_version,
                    allocated_bytes=self.runtime.torch.cuda.memory_allocated(self.runtime.device))

    def release(self, request):
        self.runtime.release(self.state(request))
        self.outputs.pop(request["state_id"], None)
        return super().release(request)

    def close(self):
        super().close()
        self.runtime.close()
