"""Frozen LongLive 1.3B streaming/prompt-update experiment.

Imports are CPU-only. CUDA and third-party model imports occur exclusively in
``run_phase`` after the supervisor's budget callback. Every result distinguishes
conditioning provenance, actual headless playout, and unjudged visual semantics.
"""

import contextlib
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from .streaming import Arrivals, PromptEvent, PromptQueue, WallClockPlayer, condition_metrics, work_accounting


SOURCE_REVISION = "c6a27e771e803b1c12ffa28b82ba69b42472020e"
CHECKPOINT_REVISION = "cda9138d93872ef109d9a74c270184eb3a5acc6e"
PROMPT_PAIRS = (
    ("A bright red ball rests on a plain white table. Static camera, natural light.",
     "A bright blue ball rests on a plain white table. Static camera, natural light."),
    ("A woman in a yellow coat stands still in a garden with both hands by her sides. Static camera.",
     "A woman in a yellow coat waves her right hand in the same garden. Static camera."),
)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError("only one or two sessions are authorized by this protocol")
    if sorted(options.get("buffer_chunks", [0, 1, 2])) != [0, 1, 2]:
        raise ValueError("baseline buffer gate requires 0, 1, and 2 chunks")
    return options


def effective_options(config):
    result = dict(config["direction3"])
    for key, value in {"correctness_blocks": 6, "profile_blocks": 10,
            "warmup_blocks": 2, "update_blocks": 16, "decode_max_abs": .005,
            "multisession_policies": ["round_robin", "minimum_playback_slack"],
            "global_sink": True, "timestep_shift": 5.0, "context_noise": 0,
            "burst_updates": 8, "burst_interval_chunks": .2,
            "observation_tail_seconds": 8., "minimum_observation_seconds": 20.,
            "stream_warmup_blocks": 5, "stream_warmup_seed": 303,
            "stream_warmup_prompts": ["A calm mountain landscape under a clear morning sky.",
                                       "A calm mountain landscape under an orange sunset sky."],
            "prompt_pairs": [list(pair) for pair in PROMPT_PAIRS],
            "update_patterns": ["sparse", "burst"],
            "update_arms": [["fifo", 1], ["latest_only", 0], ["latest_only", 1],
                            ["latest_only", 2], ["latest_only_flush", 1]]}.items():
        result.setdefault(key, value)
    return result


def phase_gate(phase, predecessors):
    required = {"profile": "correctness", "update": "profile", "multisession": "update"}.get(phase)
    if required and not predecessors.get(required, {}).get("eligible", False):
        return {"status": "blocked", "eligible": False,
                "reason": "required_%s_gate_not_passed" % required}
    return None


def update_events(pair, chunk_seconds, pattern, session_id="s0", options=None):
    """Freeze exogenous arrival times from a previous profile, never live progress."""
    if not chunk_seconds > 0:
        raise ValueError("invalid measured chunk duration")
    first = max(2.0, 3 * chunk_seconds)
    if pattern not in ("sparse", "burst"):
        raise ValueError("unknown update pattern")
    options = options or {}
    count = 2 if pattern == "sparse" else int(options.get("burst_updates", 8))
    interval = max(3., 4 * chunk_seconds) if pattern == "sparse" else (
        float(options.get("burst_interval_chunks", .2)) * chunk_seconds)
    if count < 2 or interval <= 0:
        raise ValueError("invalid exogenous update trace")
    return [PromptEvent(0, 0., pair[0], session_id)] + [
        PromptEvent(index + 1, first + index * interval, pair[(index + 1) % 2], session_id)
        for index in range(count)]


def observation_seconds(events, options):
    """A shared wall-clock follow-up window, independent of policy completion."""
    return max(float(options.get("minimum_observation_seconds", 20.)),
               max(event.arrival_s for event in events) + float(options.get("observation_tail_seconds", 8.)))


def choose_session(ordered, ready, players, cursor, policy):
    """Resolve equal slack by rotation so a stalled session cannot monopolize."""
    rotation = [ordered[(cursor + i) % len(ordered)] for i in range(len(ordered))]
    candidates = [sid for sid in rotation if sid in ready]
    if policy == "round_robin":
        selected = candidates[0]
    elif policy == "minimum_playback_slack":
        selected = min(candidates, key=lambda sid: players[sid].buffered_seconds())
    else:
        raise ValueError("unknown session scheduling policy")
    return selected, (ordered.index(selected) + 1) % len(ordered)


class NativeLongLive:
    """One shared weight set, explicitly isolated state for sequential sessions."""

    VAE_STATE = ("_conv_idx", "_feat_map", "_enc_conv_idx", "_enc_feat_map")

    def __init__(self, config, check_budget):
        self.check_budget = check_budget
        check_budget()
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
            raise ValueError("LongLive pilot is restricted to the authorized A100")
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
            check_budget()
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
        self.check_budget()
        if session_id in self.sessions:
            raise ValueError("session already exists")
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
        self.check_budget()
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
        self.check_budget()
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
                self.check_budget()
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


def _artifact(directory, name, pixels):
    import numpy as np
    path = directory / (name + ".npy")
    directory.mkdir(parents=True, exist_ok=True)
    np.save(path, np.rint(pixels * 255).astype(np.uint8), allow_pickle=False)
    return {"path": str(path), "sha256": _hash(path), "frames": len(pixels)}


class ArtifactWriter:
    """Save immutable CPU frame arrays off the measured generation path."""

    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dit-artifact")
        self.pending = []
        self.write_seconds = 0.
        self.peak_pending = 0

    def submit(self, directory, name, pixels, row):
        def save():
            started = time.monotonic()
            return _artifact(directory, name, pixels), time.monotonic() - started
        row["artifact"] = {"path": str(directory / (name + ".npy")), "status": "pending"}
        self.pending.append((self.pool.submit(save), row))
        self.peak_pending = max(self.peak_pending, len(self.pending))
        self.collect()

    def collect(self):
        remaining = []
        for future, row in self.pending:
            if future.done():
                row["artifact"], seconds = future.result()
                self.write_seconds += seconds
            else:
                remaining.append((future, row))
        self.pending = remaining

    def finish(self, check_budget):
        started = time.monotonic()
        while self.pending:
            check_budget()
            self.collect()
            if self.pending:
                time.sleep(.01)
        self.pool.shutdown(wait=True)
        return {"scope": "independent_CPU_writer_including_uint8_conversion_and_SHA256",
                "writer_seconds": self.write_seconds, "final_join_seconds": time.monotonic() - started,
                "peak_pending_chunks": self.peak_pending}

    def close(self):
        self.pool.shutdown(wait=True)


def _predecessors(output):
    result = {}
    for name in ("correctness", "profile", "update", "multisession"):
        for path in (output / ("worker-" + name + ".json"), output / name / "report.json"):
            if path.is_file():
                result[name] = json.loads(path.read_text())
                break
    return result


def _correctness(runtime, config, output, report):
    import numpy as np
    torch, options = runtime.torch, config["direction3"]
    blocks = int(options.get("correctness_blocks", 6))
    switch = 4 * 3
    if blocks < 6:
        raise ValueError("correctness must cross the 12-latent-frame history window")
    pairs = options.get("prompt_pairs", PROMPT_PAIRS)
    for pair_index, pair in enumerate(pairs):
        for seed in options.get("seeds", [101, 102]):
            runtime.check_budget()
            sid = "p%d-s%d" % (pair_index, seed)
            session = runtime.new_session(sid, seed, blocks)
            noise = session.noise.clone()
            # The official oracle allocates its own cache. Keep only this
            # session's RNG/noise, avoiding an unused extra KV allocation.
            runtime.pipe.kv_cache1 = runtime.pipe.crossattn_cache = None
            session.kv = session.cross = None
            # Independent official full-video reference. Future prompts are known
            # only in this correctness oracle; it is never a latency arm.
            try:
                with torch.inference_mode(), runtime.rng(session):
                    reference_video, reference_latent = runtime.pipe.inference(noise=noise,
                        text_prompts_list=[[pair[0]], [pair[1]]], switch_frame_indices=[switch],
                        return_latents=True)
                expected = reference_latent.detach().cpu()
                reference_pixels = reference_video[0].permute(0, 2, 3, 1).float().cpu().numpy()
            finally:
                # The oracle owns caches that are not attached to this session.
                # Drop them before the next allocation, including cancellation.
                runtime.pipe.kv_cache1 = runtime.pipe.crossattn_cache = None
                runtime.release(session)
            del reference_video, reference_latent, noise
            reference_artifact = _artifact(output / "artifacts", sid + "-reference", reference_pixels)
            report["current_sample"] = {"sample": sid, "status": "running",
                "reference_artifact": reference_artifact, "completed_chunks": []}
            _write(output / "partial.json", report)
            session = runtime.new_session(sid, seed, blocks)
            chunks, latents = [], []
            runtime.condition(session, pair[0], 0)
            for block in range(blocks):
                if block * 3 == switch:
                    runtime.condition(session, pair[1], 1)
                latent, pixels, measured = runtime.chunk(session)
                chunks.append(pixels)
                latents.append(latent)
                report["current_sample"]["completed_chunks"].append(
                    _artifact(output / "artifacts", sid + "-chunk-%02d" % block, pixels))
                _write(output / "partial.json", report)
            actual = torch.cat(latents, dim=1)
            latent_equal = torch.equal(expected, actual)
            latent_max_abs = (expected.float() - actual.float()).abs().max().item()
            pixels = np.concatenate(chunks)
            shape_equal = pixels.shape == reference_pixels.shape
            pixel_max_abs = float(np.max(np.abs(pixels - reference_pixels))) if shape_equal else None
            row = {"sample": sid, "latent_equal": latent_equal, "latent_max_abs": latent_max_abs,
                "pixel_shape_equal": shape_equal, "pixel_max_abs": pixel_max_abs,
                "reference_scope": "official_offline_prompt_switch_and_full_VAE",
                "artifact": _artifact(output / "artifacts", sid, pixels),
                "reference_artifact": reference_artifact}
            row["passed"] = latent_equal and shape_equal and pixel_max_abs <= float(options.get("decode_max_abs", .005))
            report["samples"].append(row)
            report.pop("current_sample", None)
            _write(output / "partial.json", report)
            runtime.release(session)
            if not row["passed"]:
                report.update(status="failed", eligible=False, reason="native_chunk_reference_mismatch")
                return
    report.update(status="passed", eligible=len(report["samples"]) >= 4,
                  reason="four_or_more_native_reference_pairs" if len(report["samples"]) >= 4 else "insufficient_samples")


def _profile(runtime, config, output, report):
    options = config["direction3"]
    blocks = int(options.get("profile_blocks", 10))
    configured_warmup = int(options.get("warmup_blocks", 2))
    first_roll = math.ceil(options.get("local_attn_size", 12) / options.get("latent_frames_per_chunk", 3))
    warmup = max(configured_warmup, first_roll)
    report.update(configured_warmup_blocks=configured_warmup, effective_warmup_blocks=warmup,
        first_kv_roll_chunk_index=first_roll, measured_chunk_indices=[],
        warmup_reason="exclude growing-KV and first-full-window chunks; begin at audited first roll_and_insert chunk",
        steady_state_basis="official causal attention source audit, not a runtime kernel trace")
    if blocks <= warmup + 2:
        raise ValueError("profile requires at least three measured steady-state chunks after effective warmup")
    session = runtime.new_session("profile", 101, blocks)
    try:
        report["initial_condition"] = runtime.condition(session, options["prompt_pairs"][0][0], 0)
        for index in range(blocks):
            _, pixels, row = runtime.chunk(session)
            row["warmup"] = index < warmup
            report["samples"].append(row)
            if not row["warmup"]:
                report["measured_chunk_indices"].append(index)
            if index in (0, blocks - 1):
                row["artifact"] = _artifact(output / "artifacts", "chunk-%02d" % index, pixels)
            _write(output / "partial.json", report)
        measured = report["samples"][warmup:]
        fps = sum(x["pixel_frames"] for x in measured) / sum(x["total_s"] for x in measured)
        report.update(measured_fps=fps, median_chunk_seconds=statistics.median(x["total_s"] for x in measured),
            admission_scope="single_session_steady_chunk_denoise_context_decode_transfer",
            status="passed" if fps >= 16 else "stopped", eligible=fps >= 16,
            reason="meets_fixed_16fps_service_gate" if fps >= 16 else "A100_480p_below_16fps_no_fps_relabeling")
    finally:
        runtime.release(session)


def _stream_warmup(runtime, config, output, report):
    """Warm kernel shapes/official recache using an unrelated private session."""
    options = config["direction3"]
    blocks = int(options["stream_warmup_blocks"])
    if blocks < 5:
        raise ValueError("stream warmup must exercise a full 12-latent recache")
    session = runtime.new_session("warmup", int(options["stream_warmup_seed"]), blocks)
    started = time.monotonic()
    warmup = {"scope": "unrelated_prompts_full_window_recache_included_in_GPU_budget",
              "blocks": [], "conditioning": []}
    report["warmup"] = warmup
    try:
        for index in range(blocks):
            if index in (0, 4):
                warmup["conditioning"].append(runtime.condition(session,
                    options["stream_warmup_prompts"][int(index == 4)], int(index == 4)))
            _, _, measured = runtime.chunk(session)
            warmup["blocks"].append(measured)
            _write(output / "partial.json", report)
    finally:
        warmup["wall_s"] = time.monotonic() - started
        runtime.release(session)


def _stream_run(runtime, events, policy, buffer_chunks, seed, blocks, directory, session_count=1,
                session_policy="round_robin", observation_end_s=20.):
    if any(event.arrival_s >= observation_end_s for event in events):
        raise ValueError("all scheduled updates require the same positive observation window")
    # Provision enough deterministic noise for a full playback window plus
    # worst-case buffer flushes. Capacity is identical across policy arms; the
    # old update_blocks option is a minimum allocation, never an early EOS.
    updates_per_session = max(sum(e.session_id == "s%d" % i for e in events) for i in range(session_count))
    noise_blocks = max(blocks, math.ceil(observation_end_s * 16 / 9) + 3 + 3 * updates_per_session)
    origin = time.monotonic()
    queues = {"s%d" % n: PromptQueue(policy) for n in range(session_count)}
    players = {sid: WallClockPlayer(origin, stop_at_s=observation_end_s) for sid in queues}
    states = {}
    arrivals = Arrivals(events)
    records = []
    writer = ArtifactWriter()
    next_session = 0
    def apply_pending(sid):
        state, pending, player = states[sid], queues[sid], players[sid]
        event = pending.take()
        if event is None:
            return
        encode_start = time.monotonic() - origin
        pending.records[event.version].update(status="conditioning", encode_start_s=encode_start)
        try:
            timing = runtime.condition(state, event.prompt, event.version)
        except BaseException:
            pending.records[event.version].update(status="conditioning_interrupted",
                partial_condition_wall_s=time.monotonic() - origin - encode_start)
            raise
        applied = time.monotonic() - origin
        pending.applied(event, encode_start, encode_start + timing["encode_s"], applied)
        pending.records[event.version]["recache_s"] = timing["recache_s"]
        pending.records[event.version]["condition_after_observation_end"] = applied >= observation_end_s
        if policy == "latest_only_flush" and applied < observation_end_s:
            player.discard_older(event.version, applied)
    try:
        for n, sid in enumerate(queues):
            states[sid] = runtime.new_session(sid, seed + n, noise_blocks)
        while time.monotonic() - origin < observation_end_s:
            runtime.check_budget()
            now = time.monotonic() - origin
            for event in arrivals.collect(now):
                queues[event.session_id].arrive(event, now)
            if any(state.block >= noise_blocks for state in states.values()):
                raise RuntimeError("noise_capacity_exhausted_before_observation_end")
            if policy != "fifo":
                # Prompt updates own the next legal GPU boundary even when
                # decoded playback slots are full. Returning to collect after
                # every condition call reveals updates that arrived during
                # encoding/recaching before another chunk can be produced.
                conditioning = [sid for sid in states if queues[sid].pending]
                if conditioning:
                    sid, next_session = choose_session(list(states), conditioning, players,
                                                       next_session, session_policy)
                    apply_pending(sid)
                    continue
            ready = [sid for sid in states if players[sid].may_generate(buffer_chunks, 12)]
            if not ready:
                time.sleep(.005)
                continue
            sid, next_session = choose_session(list(states), ready, players, next_session, session_policy)
            state, pending, player = states[sid], queues[sid], players[sid]
            if policy == "fifo":
                # Preserve FIFO's one admitted update followed by one chunk;
                # an update arriving during this call cannot starve its output.
                apply_pending(sid)
            if time.monotonic() - origin >= observation_end_s:
                break
            if state.conditional is None:
                time.sleep(.005)
                continue
            reserved = state.block
            player.reserve_chunk(buffer_chunks, reserved)
            try:
                _, pixels, measured = runtime.chunk(state)
                released = time.monotonic() - origin
                prefix = sid + "-chunk-%03d" % measured["block"]
                # Frames are CPU-materialized before release to the independent
                # player. Admission reserved one whole actual chunk, including
                # the first 9-frame block and later 12-frame blocks alike.
                if released < observation_end_s:
                    player.enqueue([{"frame_id": prefix + "-%02d" % i,
                        "session_id": sid, "prompt_version": state.prompt_version,
                        "chunk": measured["block"], "pixels": frame}
                        for i, frame in enumerate(pixels)], released)
            finally:
                # Includes budget/cancellation and work finishing after cutoff.
                player.cancel_reservation(reserved)
            row = dict(measured, session_id=sid, released_s=released, prompt_version=state.prompt_version,
                       output_after_observation_window=released >= observation_end_s)
            records.append(row)
            writer.submit(directory / "artifacts", prefix, pixels, row)
            _write(directory / "partial.json", {"chunks": records, "policy": policy,
                "buffer_chunks": buffer_chunks, "observation_end_s": observation_end_s})
        generation_stop = time.monotonic() - origin
        # Complete event accounting even if a nonpreemptive model call crosses
        # the cutoff. No event disappears from the shared trace denominator.
        while time.monotonic() - origin < observation_end_s:
            runtime.check_budget()
            now = time.monotonic() - origin
            for event in arrivals.collect(now):
                queues[event.session_id].arrive(event, now)
            writer.collect()
            time.sleep(min(.01, max(0., observation_end_s - now)))
        for event in arrivals.collect(observation_end_s):
            queues[event.session_id].arrive(event, time.monotonic() - origin)
        playback = {sid: player.close(runtime.check_budget, drain=False) for sid, player in players.items()}
        by_session = {sid: [chunk for chunk in records if chunk["session_id"] == sid] for sid in queues}
        requests = {sid: condition_metrics(queue.finish(), playback[sid], by_session[sid])
                    for sid, queue in queues.items()}
        discarded_work = {sid: work_accounting(by_session[sid], requests[sid], playback[sid]) for sid in queues}
        artifact_timing = writer.finish(runtime.check_budget)
        for sid, played in playback.items():
            timeline = sorted([e for e in events if e.session_id == sid], key=lambda e: e.arrival_s)
            stale = 0
            for frame in played["presentations"]:
                arrived = [e for e in timeline if e.arrival_s <= frame["presented_s"]]
                stale += bool(arrived and frame["prompt_version"] != arrived[-1].version)
            played["condition_provenance_stale_frames"] = stale
            played["condition_provenance_stale_fraction"] = stale / max(1, len(played["presentations"]))
        result = {"policy": policy, "buffer_chunks": buffer_chunks, "seed": seed,
            "buffer_semantics": next(iter(playback.values()))["buffer_semantics"],
            "current_chunk_excluded_from_buffer_limit": True,
            "conditioning_dispatch": ("one_FIFO_update_before_each_generated_chunk" if policy == "fifo"
                                      else "next_legal_GPU_boundary_independent_of_playback_buffer"),
            "session_policy": session_policy,
            "chunks": records, "playback": playback, "updates": requests,
            "discarded_work": discarded_work,
            "observation_end_s": observation_end_s, "generation_stop_s": generation_stop,
            "noise_capacity_blocks": noise_blocks, "generation_ends_only_at_window": True,
            "scheduled_events": len(events), "observed_events": sum(len(rows) for rows in requests.values()),
            "scheduled_update_events_excluding_initial": len(events) - session_count,
            "artifact_timing": artifact_timing,
            "semantic_status": "requires_blinded_video_adjudication"}
        _write(directory / "report.json", result)
        return result
    finally:
        # Cancellation still accounts for every payload that has actually
        # arrived, without reading future conditions or shrinking the trace.
        stopped_at = min(time.monotonic() - origin, observation_end_s)
        for event in arrivals.collect(stopped_at):
            queues[event.session_id].arrive(event, stopped_at)
        for player in players.values():
            player.close(drain=False)
        writer.close()
        writer.collect()
        _write(directory / "partial.json", {"chunks": records, "policy": policy,
            "buffer_chunks": buffer_chunks, "observation_end_s": observation_end_s,
            "scheduled_events": len(events),
            "observed_events": sum(len(queue.records) for queue in queues.values()),
            "scheduled_update_events_excluding_initial": len(events) - session_count,
            "updates": {sid: list(queue.records.values()) for sid, queue in queues.items()},
            "playback": {sid: player.report() for sid, player in players.items()},
            "discarded_work": {sid: work_accounting(
                [chunk for chunk in records if chunk["session_id"] == sid],
                list(queues[sid].records.values()), players[sid].report()) for sid in queues}})
        for state in states.values():
            runtime.release(state)


def _update(runtime, config, output, report, previous):
    options = config["direction3"]
    duration = previous["profile"]["median_chunk_seconds"]
    blocks = int(options.get("update_blocks", 16))
    # Small development runs compare simple buffer limits; there is no invented
    # optimized scheduler. All evidence is retained even if simple policies win.
    arms = options["update_arms"]
    for pattern in options["update_patterns"]:
        for seed in options.get("seeds", [101, 102]):
            for policy, buffer_chunks in arms:
                unit = "%s-%d-%s-b%d" % (pattern, seed, policy, buffer_chunks)
                events = update_events(options["prompt_pairs"][0], duration, pattern, options=options)
                row = _stream_run(runtime, events, policy, buffer_chunks, seed, blocks, output / unit,
                                  observation_end_s=observation_seconds(events, options))
                row.update(unit=unit, pattern=pattern)
                report["samples"].append(row)
                _write(output / "partial.json", report)
    report.update(status="passed", eligible=True,
        technical_eligible=True, semantic_eligible=False,
        semantic_status="pending_blinded_review_no_semantic_success_claim",
        reason="actual_prompt_consumption_and_headless_playout_recorded",
        research_conclusion="baseline_screening_only_no_proven_new_scheduler_gain")


def _multisession(runtime, config, output, report, previous):
    if config["direction3"].get("maximum_sessions", 2) < 2:
        report.update(status="stopped", eligible=False, reason="single_session_only_configuration")
        return
    # Necessity gate: one GPU must offer the aggregate service rate before two
    # 16 FPS sessions are tried. Memory admission is still verified by CUDA.
    if previous["profile"]["measured_fps"] < 32:
        report.update(status="stopped", eligible=False,
                      reason="single_GPU_service_curve_below_two_times_16fps")
        return
    # Verify session isolation before interpreting any multiplexed timing.
    import numpy as np
    references = {}
    for index in range(2):
        session = runtime.new_session("reference-%d" % index, 201 + index, 3)
        runtime.condition(session, PROMPT_PAIRS[index][0], 0)
        latent_chunks, pixels = [], []
        for block in range(3):
            if block == 1:
                runtime.condition(session, PROMPT_PAIRS[index][1], 1)
            latent, image, _ = runtime.chunk(session)
            latent_chunks.append(latent)
            pixels.append(image)
        references[index] = (runtime.torch.cat(latent_chunks, dim=1), np.concatenate(pixels))
        runtime.release(session)
    sessions = [runtime.new_session("isolation-%d" % n, 201 + n, 3) for n in range(2)]
    actual = {index: ([], []) for index in range(2)}
    try:
        for block in range(3):
            for index, session in enumerate(sessions):
                if block in (0, 1):
                    runtime.condition(session, PROMPT_PAIRS[index][block], block)
                latent, image, _ = runtime.chunk(session)
                actual[index][0].append(latent)
                actual[index][1].append(image)
        isolation = []
        for index in range(2):
            latents = runtime.torch.cat(actual[index][0], dim=1)
            images = np.concatenate(actual[index][1])
            expected_latent, expected_images = references[index]
            row = {"session": index, "latent_equal": runtime.torch.equal(latents, expected_latent),
                   "pixel_max_abs": float(np.max(np.abs(images - expected_images))),
                   "artifact": _artifact(output / "isolation", "session-%d" % index, images)}
            isolation.append(row)
        report["state_isolation"] = isolation
        if not all(r["latent_equal"] and r["pixel_max_abs"] <= .005 for r in isolation):
            report.update(status="failed", eligible=False, reason="multisession_state_contamination")
            return
    finally:
        for session in sessions:
            runtime.release(session)
    duration = previous["profile"]["median_chunk_seconds"]
    events = sum([update_events(config["direction3"]["prompt_pairs"][i], duration, "sparse", "s%d" % i,
                               config["direction3"])
                  for i in range(2)], [])
    for policy in config["direction3"]["multisession_policies"]:
        row = _stream_run(runtime, events, "latest_only", 1, 101,
            int(config["direction3"].get("update_blocks", 16)), output / policy, 2, policy,
            observation_seconds(events, config["direction3"]))
        report["samples"].append(row)
        _write(output / "partial.json", report)
    report.update(status="passed", eligible=False,
        reason="two_session_screen_complete_semantics_require_review",
        semantic_status="pending_blinded_review")


def run_phase(config, output: Path, check_budget, phase: str):
    base = Path(output)
    previous = _predecessors(base)
    output = base / phase
    output.mkdir(parents=True, exist_ok=True)
    report = {"direction": 3, "phase": phase, "status": "incomplete", "eligible": False,
        "samples": [], "semantic_status": "unjudged",
        "playback_scope": "actual_headless_consumption_no_browser_network_or_monitor"}
    config = dict(config, direction3=effective_options(config))
    report["effective_config"] = config["direction3"]
    gate = phase_gate(phase, previous)
    if gate:
        report.update(gate)
        _write(output / "report.json", report)
        return report
    if phase == "multisession" and (config["direction3"].get("maximum_sessions", 2) < 2
            or previous["profile"]["measured_fps"] < 32):
        report.update(status="stopped", eligible=False,
            reason="single_session_only_configuration" if config["direction3"].get("maximum_sessions", 2) < 2
                   else "single_GPU_service_curve_below_two_times_16fps")
        _write(output / "report.json", report)
        return report
    if phase == "report":
        report.update(status="passed", predecessors=previous,
                      reason="evidence_summary_only_semantic_adjudication_separate")
        _write(output / "report.json", report)
        return report
    runtime = None
    try:
        validate_config(config)
        check_budget()
        runtime = NativeLongLive(config, check_budget)
        report["identity"] = runtime.identity
        if phase == "correctness":
            _correctness(runtime, config, output, report)
        elif phase == "profile":
            _profile(runtime, config, output, report)
        elif phase == "update":
            _stream_warmup(runtime, config, output, report)
            _update(runtime, config, output, report, previous)
        elif phase == "multisession":
            # The 32 FPS necessity gate should avoid unnecessary GPU warmup.
            if previous["profile"]["measured_fps"] >= 32:
                _stream_warmup(runtime, config, output, report)
            _multisession(runtime, config, output, report, previous)
        else:
            raise ValueError("unknown direction3 phase: " + phase)
    except BaseException as error:
        stopped = error.__class__.__name__ == "BudgetStop"
        out_of_memory = "OutOfMemory" in type(error).__name__
        report.update(status="stopped" if stopped or out_of_memory else "failed", eligible=False,
                      reason="budget_stop" if stopped else "memory_admission_failed" if out_of_memory else type(error).__name__,
                      error=str(error), error_traceback=traceback.format_exc())
        _write(output / "partial.json", report)
        if not isinstance(error, Exception):
            raise
    finally:
        if runtime is not None:
            runtime.close()
        _write(output / "report.json", report)
    return report
