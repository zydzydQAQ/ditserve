#!/usr/bin/env python3
"""GPU quality microbenchmarks and manually ordered component replay.

The preencoded replay serves one video and three images. It does not exercise
an online scheduler or reproduce the original Vanilla-xDiT/GenServe systems.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "workers", ROOT / "python"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from ditserve.evaluation.hetero_metrics import compute_psnr_ssim, fast_prepare_wan  # noqa: E402
from ditserve.models.sd35 import SD35Backend  # noqa: E402
from ditserve.models.wan.backend import WanBackend  # noqa: E402


REPLAY_SCOPE = (
    "preencoded: excludes prompt encoding; includes GPU prepare, denoise, complete "
    "decode and arrival waits; decoded NumPy outputs, excludes file persistence"
)
IMAGE_SCOPE = (
    "preencoded: excludes prompt encoding; includes embedding retrieval/H2D, "
    "pipeline preparation, denoise and complete decode to NumPy; excludes persistence"
)
DENOISE_SCOPE = "denoise only; preencoded, excludes encoding, preparation and decode"


def _wait_until_arrival(t_start, arrival_ms, *, clock=time.perf_counter, sleep=time.sleep):
    if not math.isfinite(arrival_ms) or arrival_ms < 0:
        raise ValueError("arrival must be finite and nonnegative")
    while True:
        now_ms = (clock() - t_start) * 1000.0
        remaining_ms = arrival_ms - now_ms
        if remaining_ms <= 0:
            return now_ms
        sleep(remaining_ms / 1000.0)


def _image_requests(prompts, seeds, arrivals=None, **spec):
    arrivals = [0.0] * len(prompts) if arrivals is None else arrivals
    if not prompts or len(prompts) != len(seeds) or len(prompts) != len(arrivals):
        raise ValueError("each image requires its own prompt, seed and arrival")
    base = dict(height=512, width=512, frames=1, batch_size=1, steps=12,
                guidance_scale=4.5, max_sequence_length=256, negative_prompt="")
    base.update(spec)
    return [dict(request_id=f"img-{i + 1}", modality="image", arrival_ms=float(arrival),
                 spec=dict(base, prompt=prompt, seed=seed))
            for i, (prompt, seed, arrival) in enumerate(zip(prompts, seeds, arrivals))]


def _completion_records(requests, outputs, start_ms, finish_ms):
    """Only decoded outputs, never a declared batch size, can complete requests."""
    if not requests or len({r["request_id"] for r in requests}) != len(requests):
        raise ValueError("completion requires unique request identities")
    kind = requests[0]["modality"]
    if kind not in ("image", "video") or any(r["modality"] != kind for r in requests):
        raise ValueError("outputs must have a single known modality")
    shape = tuple(getattr(outputs, "shape", ()))
    rank = 4 if kind == "image" else 5
    if (len(shape) != rank or any(n <= 0 for n in shape) or shape[-1] not in (3, 4)
            or shape[0] != len(requests)):
        raise ValueError("decoded output count/shape does not match requests")
    if not math.isfinite(start_ms) or not math.isfinite(finish_ms) or finish_ms < start_ms:
        raise ValueError("invalid completion timestamps")
    for r in requests:
        arrival = r["arrival_ms"]
        if not math.isfinite(arrival) or arrival < 0 or start_ms < arrival:
            raise ValueError("cannot serve a request before arrival")
        if shape[-3:-1] != (r["spec"]["height"], r["spec"]["width"]):
            raise ValueError("decoded spatial dimensions do not match request")
        if kind == "video" and shape[1] != r["spec"]["frames"]:
            raise ValueError("video decode did not produce all requested frames")
    return [dict(r, spec=dict(r["spec"]), seed=r["spec"]["seed"],
                 start_ms=start_ms, finish_ms=finish_ms, output_count=1,
                 output_index=i, output_mode="decoded_numpy",
                 latency_ms=finish_ms - r["arrival_ms"],
                 batch_latency_ms=finish_ms - start_ms)
            for i, r in enumerate(requests)]


def _run_image_batch(sd35, requests, t_start, **options):
    shared = ("height", "width", "steps", "guidance_scale", "max_sequence_length")
    spec = requests[0]["spec"]
    if any(any(r["spec"][key] != spec[key] for key in shared) for r in requests):
        raise ValueError("incompatible image micro-batch")
    arguments = dict(options, **{key: spec[key] for key in shared})
    arguments.update(prompts=[r["spec"]["prompt"] for r in requests],
                     negative_prompts=[r["spec"]["negative_prompt"] for r in requests],
                     seeds=[r["spec"]["seed"] for r in requests], return_images=True)
    _wait_until_arrival(t_start, max(r["arrival_ms"] for r in requests))
    torch.cuda.synchronize()
    start_ms = (time.perf_counter() - t_start) * 1000.0
    result = sd35.run_micro_batch(arguments)
    torch.cuda.synchronize()
    finish_ms = (time.perf_counter() - t_start) * 1000.0
    records = _completion_records(requests, result.get("images"), start_ms, finish_ms)
    result["requests"] = [dict(r, execution_options=dict(options)) for r in records]
    result.pop("per_image_ms", None)
    result["batch_latency_ms"] = finish_ms - start_ms
    result["amortized_ms_per_image"] = result["batch_latency_ms"] / len(result["requests"])
    return result


def _decode_video(wan, state, **options):
    while True:
        video, info = wan.runtime.decode_chunks(state, 5, **options)
        if info["done"]:
            if video is None:
                raise ValueError("decode finished without video output")
            return video


def _replay_summary(requests, image_slo_ms, video_slo_ms):
    if len({r["request_id"] for r in requests}) != len(requests):
        raise ValueError("duplicate completion")
    images = [r for r in requests if r["modality"] == "image"]
    videos = [r for r in requests if r["modality"] == "video"]
    if not images or len(videos) != 1 or any(r["output_count"] != 1 for r in requests):
        raise ValueError("replay requires completed images and one completed video")
    latencies = [r["latency_ms"] for r in images]
    image_ontime = sum(x <= image_slo_ms for x in latencies)
    video_ontime = int(videos[0]["latency_ms"] <= video_slo_ms)
    return dict(
        evidence_type="component_replay", timing_scope=REPLAY_SCOPE,
        scheduler_scope="manual phase order; not evidence of online scheduling",
        requests=requests, completed_requests=len(requests),
        image_sample_count=len(images), image_latency_statistic="sample maximum, not p99",
        image_latencies_ms=latencies,
        image_mean_latency_ms=sum(latencies) / len(images),
        image_max_latency_ms=max(latencies), video_latency_ms=videos[0]["latency_ms"],
        makespan_ms=max(r["finish_ms"] for r in requests),
        image_slo_attainment_pct=100.0 * image_ontime / len(images),
        video_slo_attainment_pct=100.0 * video_ontime,
        total_slo_attainment_pct=100.0 * (image_ontime + video_ontime) / len(requests),
    )


def _fast_prepare_wan(wan: WanBackend, state_id: str, spec: dict, base_encoded) -> None:
    fast_prepare_wan(wan, spec, state_id, base_encoded)


def run_quality_pareto(wan: WanBackend, out_dir: Path) -> list[dict]:
    """Evaluate real 20-step Wan2.1-T2V-1.3B quality vs latency across cache thresholds."""
    thresholds = [0.00, 0.06, 0.08, 0.12, 0.16, 0.24]
    prompt = "Cinematic aerial drone shot over a turquoise mountain lake surrounded by pine forests at sunrise"
    base_spec = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, distorted",
        "height": 480,
        "width": 832,
        "frames": 17,
        "steps": 20,
        "guidance_scale": 5.0,
        "seed": 42,
    }

    # Encode prompt once and warm up 2 steps
    base_encoded = wan.runtime.encode_prompt(wan._spec({"state_id": "pre-q", "spec": base_spec}))
    _fast_prepare_wan(wan, "warmup-q", dict(base_spec, steps=4, cache=False), base_encoded)
    wan.advance({"state_id": "warmup-q", "steps": 2})
    wan.release({"state_id": "warmup-q"})
    torch.cuda.synchronize()

    ref_latents = None
    ref_frames = None
    base_step_ms = None
    rows = []

    for tau in thresholds:
        sid = f"pareto-tau-{str(tau).replace('.', '_')}"
        spec = dict(
            base_spec,
            request_id=sid,
            cache=(tau > 0.0),
            threshold=float(tau) if tau > 0.0 else 0.08,
            max_continuous=3,
        )
        _fast_prepare_wan(wan, sid, spec, base_encoded)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        wan.advance({"state_id": sid, "steps": 20})
        torch.cuda.synchronize()
        denoise_ms = (time.perf_counter() - t0) * 1000.0

        state = wan.states[sid]
        latents = state.latents.detach().float().cpu()
        reused = sum(int(ev.get("reused_forwards", 0)) for ev in state.events if ev.get("kind") == "step")
        total_passes = 20 * 2  # 20 steps x 2 CFG branches

        # Decode video frames via chunked 3D-VAE directly in memory (float32 [1, F, H, W, C])
        t_dec = time.perf_counter()
        decoded_arr = None
        while True:
            arr, info = wan.runtime.decode_chunks(state, 5)
            if info["done"]:
                decoded_arr = arr
                break
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t_dec) * 1000.0
        frames_f32 = np.asarray(decoded_arr, dtype=np.float32)
        wan.release({"state_id": sid})
        wan.release_cache()

        if tau == 0.0:
            ref_latents = latents.clone()
            ref_frames = frames_f32.copy()
            base_step_ms = denoise_ms
            cos_sim = 1.0
            rmse = 0.0
            psnr, ssim = 99.99, 1.0000
        else:
            cos_sim = float(
                F.cosine_similarity(
                    latents.flatten().unsqueeze(0), ref_latents.flatten().unsqueeze(0)
                ).item()
            )
            rmse = float(torch.sqrt(torch.mean((latents - ref_latents) ** 2)).item())
            psnr, ssim = compute_psnr_ssim(ref_frames, frames_f32)

        speedup = (base_step_ms / denoise_ms) if base_step_ms else 1.0
        rec = {
            "evidence_type": "measured_microbenchmark",
            "timing_scope": DENOISE_SCOPE,
            "decode_timing_scope": "complete chunked decode, excludes denoise",
            "quality_status": "reference" if tau == 0.0 else "measured_comparison",
            "spec": spec,
            "tau": tau,
            "mode": "Exact (Lossless)" if tau == 0.0 else f"DBCache (tau={tau:.2f})",
            "denoise_20steps_ms": round(denoise_ms, 2),
            "per_step_ms": round(denoise_ms / 20.0, 2),
            "decode_ms": round(decode_ms, 2),
            "speedup_vs_exact": round(speedup, 3),
            "reused_cfg_passes": reused,
            "total_cfg_passes": total_passes,
            "skip_fraction_pct": round(100.0 * reused / total_passes, 1),
            "latent_cos_sim": round(cos_sim, 6),
            "latent_rmse": round(rmse, 5),
            "pixel_psnr_db": psnr,
            "pixel_ssim": ssim,
        }
        print(
            f"[Quality-Pareto] tau={tau:0.2f} | denoise={denoise_ms:7.1f} ms ({speedup:4.2f}x) | "
            f"skip={rec['skip_fraction_pct']:4.1f}% | cos={cos_sim:.5f} | PSNR={psnr:5.2f} dB | SSIM={ssim:.4f}",
            flush=True,
        )
        rows.append(rec)

    return rows


def run_real_mixed_trace_replay(
    wan_step: WanBackend,
    sd35: SD35Backend,
    out_dir: Path,
) -> list[dict]:
    """Manually ordered component replay; not the original systems or an online scheduler."""
    vid_slo_ms = 14500.0
    img_slo_ms = 4000.0
    img_arrivals = [250.0, 280.0, 2400.0]

    v_spec = {
        "prompt": "A golden retriever running across a sunlit meadow in slow motion",
        "negative_prompt": "blurry, low quality",
        "height": 480,
        "width": 832,
        "frames": 17,
        "steps": 20,
        "guidance_scale": 5.0,
        "seed": 101,
    }
    i_spec = {
        "prompt": "Architectural photo of a glass pavilion in autumn forest",
        "negative_prompt": "blurry, low quality",
        "height": 512,
        "width": 512,
        "frames": 1,
        "batch_size": 1,
        "steps": 12,
        "guidance_scale": 4.5,
        "max_sequence_length": 256,
        "seed": 202,
    }

    images = _image_requests([i_spec["prompt"]] * 3, [202, 203, 204], img_arrivals,
                             **{k: v for k, v in i_spec.items() if k not in ("prompt", "seed")})
    v_encoded = wan_step.runtime.encode_prompt(wan_step._spec({"state_id": "pre-v", "spec": v_spec}))
    sd35._ensure_staged(i_spec["prompt"], i_spec["negative_prompt"], 256, True)
    sd35.offload_encoders({})
    sd35.run_micro_batch(dict(i_spec, prompts=[i_spec["prompt"]], seeds=[202], steps=2,
                             attention="flash", cache=False, return_images=True))
    torch.cuda.synchronize()

    replay_results = []
    original_execution = wan_step.execution
    policies = [
        ("Non-preemptive FCFS component", "fcfs"),
        ("Step-preempt + blocking tiered transfer component (not GenServe)", "step"),
        ("Block-preempt + micro-batch component (exact cache disabled)", "block"),
        ("Step-preempt + micro-batch + fixed cache tau=0.16 component", "approx"),
    ]
    for name, policy in policies:
        wan_step.execution = "block" if policy == "block" else "step"
        sid = f"replay-{policy}-v0"
        run_spec = dict(v_spec, cache=policy == "approx", threshold=0.16, max_continuous=3)
        video_request = dict(request_id="v0", modality="video", arrival_ms=0.0, spec=run_spec)
        completions = []
        transfers = []
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        v_start_ms = _wait_until_arrival(t_start, video_request["arrival_ms"])
        _fast_prepare_wan(wan_step, sid, run_spec, v_encoded)

        def serve(members, transfer=False):
            _wait_until_arrival(t_start, max(r["arrival_ms"] for r in members))
            if transfer:
                transfers.append(wan_step.tiered_offload({"state_id": sid, "non_blocking": policy != "step"}))
            batch = _run_image_batch(sd35, members, t_start, attention="flash", cache=False)
            completions.extend(batch["requests"])
            if transfer:
                transfers.append(wan_step.tiered_prefetch({"state_id": sid}))

        def finish_video():
            video = _decode_video(wan_step, wan_step.states[sid])
            torch.cuda.synchronize()
            finish_ms = (time.perf_counter() - t_start) * 1000.0
            completions.extend(_completion_records([video_request], video, v_start_ms, finish_ms))

        try:
            if policy == "fcfs":
                wan_step.advance({"state_id": sid, "steps": 20})
                finish_video()
                for request in images:
                    serve([request])
            elif policy == "step":
                wan_step.advance({"state_id": sid, "steps": 1})
                for request in images[:2]:
                    serve([request], transfer=True)
                wan_step.advance({"state_id": sid, "steps": 2})
                serve(images[2:], transfer=True)
                wan_step.advance({"state_id": sid, "steps": 17})
                finish_video()
            else:
                if policy == "block":
                    wan_step.advance({"state_id": sid, "blocks": 20})
                    serve(images[:2])
                    while wan_step.states[sid].step_index < 2:
                        wan_step.advance({"state_id": sid, "quantum": "step"})
                else:
                    wan_step.advance({"state_id": sid, "steps": 1})
                    serve(images[:2], transfer=True)
                    wan_step.advance({"state_id": sid, "steps": 1})
                serve(images[2:], transfer=True)
                while wan_step.states[sid].step_index < run_spec["steps"]:
                    wan_step.advance({"state_id": sid, "quantum": "step", "steps": 1})
                finish_video()
            reused_steps = sum(ev.get("reused_forwards", 0) > 0
                               for ev in wan_step.states[sid].events if ev.get("kind") == "step")
        finally:
            wan_step.release({"state_id": sid})
            wan_step.execution = original_execution

        rec = _replay_summary(completions, img_slo_ms, vid_slo_ms)
        rec.update(system=name, transfer_events=transfers, reused_steps=reused_steps,
                   transfer_scope="actual tiered offload/prefetch; no unpinned cold-swap emulation",
                   quality_status="not_evaluated_in_replay")
        print(
            f"[Component-Replay] {name} | Img max (n=3)={rec['image_max_latency_ms']:.1f} ms | "
            f"Vid={rec['video_latency_ms']:.1f} ms | Total SLO={rec['total_slo_attainment_pct']:.1f}%",
            flush=True,
        )
        replay_results.append(rec)

    return replay_results


def main() -> None:
    out_dir = Path(os.environ.get("RUN_DIR", str(ROOT / ".cache" / "quality_replay")))
    out_dir.mkdir(parents=True, exist_ok=True)
    wan_path = os.environ.get("WAN21_PATH", "/shared/data/DITSERVE/models/wan")
    sd35_path = os.environ.get("SD35_PATH", "/shared/data/DITSERVE/models/sd3.5-medium")
    sd35_rev = os.environ.get("SD35_REVISION", "b940f670f0eda2d07fbb75229e779da1ad11eb80")

    print("=== Initializing Real Wan2.1-T2V-1.3B and SD3.5-Medium on A100 ===", flush=True)
    wan_step = WanBackend({
        "device": "cuda:0",
        "model_path": wan_path,
        "attention": "flash",
        "vae_dtype": "bfloat16",
        "work_dir": str(out_dir),
        "options": {"engine": "xdit", "execution": "step", "max_pinned_mb": 4096},
    })
    ParetoRows = run_quality_pareto(wan_step, out_dir)

    sd35 = SD35Backend({
        "model_path": sd35_path,
        "revision": sd35_rev,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "text_encoder_dtype": "bfloat16",
        "work_dir": str(out_dir),
        "options": {"profile": False, "encoder_pool": {"max_entries": 64}},
    })

    ReplayRows = run_real_mixed_trace_replay(wan_step, sd35, out_dir)

    sd35.close()
    wan_step.close()

    payload = {
        "schema": "ditserve.hetero.quality_and_replay.v1",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "quality_pareto": ParetoRows,
        "evidence_types": ["measured_microbenchmark", "component_replay"],
        "component_replay": ReplayRows,
    }
    (out_dir / "QUALITY_AND_REPLAY.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# HeteroDiT GPU Quality Microbenchmarks & Component Replay",
        "",
        "Evidence: measured_microbenchmark / component_replay; not online scheduling evidence.",
        f"Replay timing: {REPLAY_SCOPE}.",
        "All policies use the same prompts/seeds, image flash attention and decoded output mode.",
        "Tiered transfers are real component calls, not the original GenServe cold-swap implementation.",
        "Quality reference/self rows are not independent quality passes.",
        "",
        f"- **GPU**: `{payload['gpu_name']}`",
        "",
        "## 1. `Wan2.1-T2V-1.3B` (20 Steps, `480x832x17`) Real Quality-Latency Pareto Curve",
        "",
        "| Mode / Threshold ($\\tau$) | 20-Step Denoise (ms) | Speedup | Skipped CFG Passes | Latent Cosine Sim | Latent RMSE | Decoded Video PSNR (dB) | Decoded Video SSIM |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in ParetoRows:
        md.append(
            f"| `{r['mode']}` | **{r['denoise_20steps_ms']:.1f}** | **{r['speedup_vs_exact']:.2f}×** | "
            f"`{r['reused_cfg_passes']}/{r['total_cfg_passes']}` (`{r['skip_fraction_pct']:.1f}%`) | "
            f"`{r['latent_cos_sim']:.6f}` | `{r['latent_rmse']:.5f}` | **{r['pixel_psnr_db']:.2f} dB** | **{r['pixel_ssim']:.4f}** |"
        )
    md.extend([
        "",
        "## 2. Manually Ordered Component Replay (`Wan2.1` + Bursty `SD3.5`)",
        "",
        "Image maximum is based on n=3, not a P99 estimate. Video completion includes full decode.",
        "",
        "| Component Policy | Image Mean Latency (ms) | Image Max Latency (ms, n=3) | Video Latency (ms) | Makespan (ms) | Image SLO% | Video SLO% | Total SLO% |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in ReplayRows:
        md.append(
            f"| `{r['system']}` | {r['image_mean_latency_ms']:.1f} | **{r['image_max_latency_ms']:.1f}** | "
            f"**{r['video_latency_ms']:.1f}** | {r['makespan_ms']:.1f} | {r['image_slo_attainment_pct']:.1f}% | "
            f"{r['video_slo_attainment_pct']:.1f}% | **{r['total_slo_attainment_pct']:.1f}%** |"
        )
    (out_dir / "QUALITY_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'QUALITY_AND_REPLAY.json'} and {out_dir / 'QUALITY_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
