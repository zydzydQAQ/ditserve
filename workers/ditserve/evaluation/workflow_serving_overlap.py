"""Opt-in Wan DiT(A)/VAE(B) overlap qualification and paired microbenchmark.

Only one host thread submits model work. The ordinary worker is untouched.
CUDA event envelopes are completion measurements, not proof of kernel overlap;
the emitted NVTX ranges are intended for a separately collected CUDA trace.
"""

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

from .workflow_overlap import (PROMPTS, Timeline, _decode_gpu, _hash_array,
                               _host_inputs, _install_inputs, _step_async,
                               atomic_json, bind_cpu_to_gpu)


SCHEMA = "ditserve.workflow.serving.overlap.v1"
FRAME_PAIRS = ((17, 17), (17, 33), (33, 17), (33, 33))
TIMING_SCOPE = (
    "Two preencoded requests: from A/B GPU input preparation through B's 20 DiT steps, "
    "B VAE plus A's 20 DiT steps (serial or streams), A VAE, both D2H and finite/hash checks. "
    "Excludes CPU encoding, model load, reference/warmup, and state release. "
    "stage_pair_wall_ms separately isolates A DiT plus B VAE. This is a two-request "
    "completion measurement, not online arrival/queue latency."
)


def normalize_config(config):
    """Keep caller metadata while making experimental arithmetic explicit."""
    value = json.loads(json.dumps(config, allow_nan=False))
    defaults = dict(device="cuda:1", model_path="/models/wan2.1-t2v-1.3b",
                    height=480, width=832, steps=20, repeats=3, seed=20261019,
                    guidance_scale=5.0, cpu_threads=8, cpu_masks={},
                    sample_interval_seconds=0.1,
                    reuse_streams=False, trim_after_references=False,
                    release_b_before_a_decode=False, memory_diagnostics=False,
                    decode_a_on_b_stream=False,
                    save_videos=False,
                    frame_pairs=[list(pair) for pair in FRAME_PAIRS])
    for key, default in defaults.items():
        value.setdefault(key, default)
    if value["steps"] != 20:
        raise ValueError("P4 fixes the scheduler at 20 steps")
    for key in ("height", "width", "repeats", "cpu_threads"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError("invalid positive integer " + key)
    if value["height"] % 16 or value["width"] % 16:
        raise ValueError("height and width must be multiples of 16")
    if not str(value["device"]).startswith("cuda:") or not str(value["device"])[5:].isdigit():
        raise ValueError("device must be an explicit cuda:N")
    if (not value["frame_pairs"] or any(len(pair) != 2 or any(
            type(frame) is not int or frame < 1 or (frame - 1) % 4 for frame in pair)
            for pair in value["frame_pairs"])):
        raise ValueError("frame_pairs require pairs of frames=4*k+1")
    if len({tuple(pair) for pair in value["frame_pairs"]}) != len(value["frame_pairs"]):
        raise ValueError("duplicate frame pairs")
    interval = value["sample_interval_seconds"]
    if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("sample_interval_seconds must be positive and finite")
    for key in ("reuse_streams", "trim_after_references", "release_b_before_a_decode",
                "memory_diagnostics", "decode_a_on_b_stream", "save_videos"):
        if type(value[key]) is not bool:
            raise ValueError(key + " must be boolean")
    return value


def _memory_boundary(worker, config, label):
    """Allocator attribution only; no added device synchronization."""
    if not config["memory_diagnostics"]:
        return None
    cuda, device = worker.torch.cuda, worker.device
    stats = cuda.memory_stats(device)
    return dict(label=label, timestamp_ns=time.perf_counter_ns(),
                allocated_bytes=cuda.memory_allocated(device),
                reserved_bytes=cuda.memory_reserved(device),
                inactive_split_bytes=stats.get("inactive_split_bytes.all.current"),
                active_bytes=stats.get("active_bytes.all.current"),
                num_alloc_retries=stats.get("num_alloc_retries"),
                num_ooms=stats.get("num_ooms"),
                scope="allocator observation, not NVML memory-margin qualification")


def _arm_streams(worker, config, mode):
    cuda, device = worker.torch.cuda, worker.device
    if config["reuse_streams"]:
        pool = getattr(worker, "_workflow_overlap_streams", None)
        if pool is None:
            # Reference warmup and serial execution share the current stream;
            # only the genuinely concurrent B arm needs another stream.
            pool = (cuda.current_stream(device), cuda.Stream(device=device))
            worker._workflow_overlap_streams = pool
        return pool[0], pool[0] if mode == "serial" else pool[1]
    first = cuda.Stream(device=device)
    return first, first if mode == "serial" else cuda.Stream(device=device)


def _physical_device(device):
    """NVML uses physical IDs, whereas CUDA uses the visible ordinal."""
    index = int(device.split(":")[1])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return index
    identifiers = [value.strip() for value in visible.split(",") if value.strip()]
    if index >= len(identifiers):
        raise ValueError("device outside CUDA_VISIBLE_DEVICES")
    selected = identifiers[index]
    if selected.startswith("GPU-"):
        return selected
    if selected.isdigit():
        return int(selected)
    raise ValueError("unsupported NVML/CUDA device mapping")


def _start_sampler(config):
    from .workflow_metrics import NvmlSampler
    return NvmlSampler(devices=(_physical_device(config["device"]),),
                       interval_s=config["sample_interval_seconds"]).start()


def memory_margin_gate(samples, errors, start_ns, end_ns):
    """A missing memory series never qualifies a performance measurement."""
    inside = [row for row in samples if start_ns <= row.get("timestamp_ns", -1) <= end_ns]
    valid = [row for row in inside if type(row.get("memory_free_bytes")) is int and
             type(row.get("memory_total_bytes")) is int and row["memory_total_bytes"] > 0]
    low = [row for row in valid if row["memory_free_bytes"] <
           max(1024**3, math.ceil(0.05 * row["memory_total_bytes"]))]
    complete = len(valid) >= 2 and len(valid) == len(inside) and not errors
    status = "failed" if low else "passed" if complete else "unverified"
    return dict(status=status, eligible=status == "passed", sample_count=len(inside),
                valid_memory_samples=len(valid), missing_memory_samples=len(inside)-len(valid),
                minimum_free_bytes=min((row["memory_free_bytes"] for row in valid), default=None),
                required_free_bytes=max((max(1024**3, math.ceil(row["memory_total_bytes"] * .05))
                                         for row in valid), default=None), errors=list(errors),
                rule="observed free >= max(1 GiB, 5% device capacity); >=2 valid samples; no sampling errors",
                caveat="sampled minimum; NVML can miss shorter-lived memory peaks")


def make_worker(config, device, out):
    """Bind CPU budget before torch import/model load; shared by SP diagnostics."""
    count = int(config.get("cpu_threads", 8))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(count)
    mask = config.get("cpu_masks", {}).get(device)
    if mask is None:
        affinity = bind_cpu_to_gpu(device, count)
    else:
        if (not isinstance(mask, list) or len(mask) != count or
                any(type(cpu) is not int or cpu < 0 for cpu in mask) or len(set(mask)) != len(mask)):
            raise ValueError("cpu_masks[device] must contain cpu_threads distinct CPU indices")
        allowed = sorted(os.sched_getaffinity(0))
        if not set(mask).issubset(allowed):
            raise ValueError("configured CPU mask is outside the inherited allowed mask")
        for task in Path("/proc/self/task").iterdir():
            try:
                os.sched_setaffinity(int(task.name), mask)
            except ProcessLookupError:
                pass
        affinity = dict(policy="explicit", device=device, bound_cpus=sorted(mask),
                        allowed_before=allowed, physical_core_count=count)
    atomic_json(Path(out) / ("cpu-affinity-" + device.replace(":", "-") + ".json"), affinity)
    import torch
    from ditserve.models.wan.runtime import WanStepWorker
    torch.set_num_threads(count)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    worker = WanStepWorker(device, config["model_path"], "flash")
    worker.workflow_cpu_affinity = affinity
    return worker


def request_spec(config, frames, role, request_id):
    return dict(request_id=request_id, prompt=PROMPTS[0 if role == "A" else 1],
                negative_prompt="", seed=config["seed"] + (0 if role == "A" else 1),
                height=config["height"], width=config["width"], frames=frames,
                steps=20, guidance_scale=config["guidance_scale"])


def _references(worker, config, out, timeline):
    """Ordinary runtime bridge; encode once per role/shape, retain CPU inputs."""
    import numpy as np
    result = {}
    keys = sorted({(role, pair[index]) for pair in config["frame_pairs"]
                   for index, role in enumerate(("A", "B"))})
    for role, frames in keys:
        name = f"{role}-f{frames}"
        spec = request_spec(config, frames, role, "P4-reference-" + name)
        with timeline.phase("P4/reference/cpu_encode", purpose="qualification", role=role, frames=frames):
            payload = worker.encode_prompt(spec)
        hosts = _host_inputs(worker, spec, payload, pinned=False)
        state = None
        try:
            state = worker.prepare_encoded(spec, dataclasses.replace(payload, consumed=False))
            for step in range(20):
                with timeline.phase("P4/reference/dit_step", purpose="qualification", role=role,
                                    frames=frames, step=step):
                    worker.step(state)
            latent = state.latents.detach().float().cpu().numpy().copy()
            with timeline.phase("P4/reference/vae_decode", purpose="qualification", role=role, frames=frames):
                video = worker.finish(state)
            directory = Path(out) / "references" / name
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "latent.npy", latent, allow_pickle=False)
            if config["save_videos"]:
                np.save(directory / "video.npy", video, allow_pickle=False)
            hashes = dict(latent=_hash_array(latent), video=_hash_array(video))
            atomic_json(directory / "hashes.json", hashes)
            result[(role, frames)] = dict(spec=spec, hosts=hosts, latent_cpu=latent,
                                         hashes=hashes, directory=str(directory))
        finally:
            if state is not None:
                worker.release(state)
    return result


def _exact_result(actual, reference_a, reference_b):
    checks = dict(a_latent=actual["a_latent"]["sha256"] == reference_a["latent"]["sha256"],
                  a_video=actual["a_video"]["sha256"] == reference_a["video"]["sha256"],
                  b_latent=actual["b_latent"]["sha256"] == reference_b["latent"]["sha256"],
                  b_video=actual["b_video"]["sha256"] == reference_b["video"]["sha256"])
    return dict(passed=all(checks.values()), exact=checks)


def _run_arm(worker, config, references, frames_a, frames_b, mode, repeat, purpose, timeline):
    """One submitter, independently owned state, no global sync inside the timer."""
    if mode not in ("serial", "streams"):
        raise ValueError("unknown arm")
    torch, device = worker.torch, worker.device
    ref_a, ref_b = references[("A", frames_a)], references[("B", frames_b)]
    name = f"P4-{purpose}-a{frames_a}-b{frames_b}-r{repeat}-{mode}"
    spec = dict(ref_a["spec"], request_id=name)
    state = b_state = b_video = None
    a_stream, b_stream = _arm_streams(worker, config, mode)
    memory_boundaries = []
    def observe(label):
        value = _memory_boundary(worker, config, label)
        if value is not None:
            memory_boundaries.append(value)
    anchor = torch.cuda.Event(enable_timing=True)
    a_done = torch.cuda.Event(enable_timing=True)
    b_done = torch.cuda.Event(enable_timing=True)
    meta = dict(variant="P4", purpose=purpose, mode=mode, repeat=repeat,
                frames=frames_a, peer_frames=frames_b, request_id=name)
    sampler = _start_sampler(config)
    sampler_stopped = False
    try:
        # CPU text encoding is precomputed equally for both arms. Both GPU
        # requests, including B's DiT, are recreated inside the full timer.
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        full_started = time.perf_counter_ns()
        observe("before_prepare")
        state, ready = _install_inputs(worker, spec, ref_a["hosts"])
        b_spec = dict(ref_b["spec"], request_id=name + "-B")
        b_state, b_ready = _install_inputs(worker, b_spec, ref_b["hosts"])
        observe("after_prepare")
        for step in range(20):
            with timeline.phase("P4/prefix_b_dit_step", role="B", step=step, **meta):
                _step_async(worker, b_state)
        b_ready.record(torch.cuda.current_stream(device))
        b_ready.synchronize()
        observe("after_b_dit")
        prefix_peak_allocated = torch.cuda.max_memory_allocated(device)
        prefix_peak_reserved = torch.cuda.max_memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        anchor.record(torch.cuda.current_stream(device))
        for stream in (a_stream, b_stream):
            stream.wait_event(ready)
            stream.wait_event(anchor)
        for tensor in (state.latents, state.prompt_embeds, state.negative_prompt_embeds):
            if tensor is not None:
                tensor.record_stream(a_stream)
        b_state.latents.record_stream(b_stream)
        started = time.perf_counter_ns()
        with timeline.phase("P4/stage_pair", **meta):
            with torch.cuda.stream(b_stream):
                with timeline.phase("P4/vae_decode", role="B", **meta):
                    b_video = _decode_gpu(worker, b_state)
                b_done.record(b_stream)
            with torch.cuda.stream(a_stream):
                for step in range(20):
                    with timeline.phase("P4/dit_step", role="A", step=step, **meta):
                        _step_async(worker, state)
                a_done.record(a_stream)
            # Endpoint events wait only for this arm's streams. These waits are
            # included in wall time; there is no worker._sync() in the window.
            a_done.synchronize()
            b_done.synchronize()
        finished = time.perf_counter_ns()
        observe("after_stage_pair")
        stage_peak_allocated = torch.cuda.max_memory_allocated(device)
        stage_peak_reserved = torch.cuda.max_memory_reserved(device)
        measured = dict(mode=mode, repeat=repeat, purpose=purpose,
                        stage_started_monotonic_ns=started, stage_finished_monotonic_ns=finished,
                        stage_pair_wall_ms=(finished-started)/1e6,
                        a_dit_complete_ms=anchor.elapsed_time(a_done),
                        b_vae_complete_ms=anchor.elapsed_time(b_done),
                        stage_pair_complete_ms=max(anchor.elapsed_time(a_done), anchor.elapsed_time(b_done)),
                        stage_peak_allocated_bytes=stage_peak_allocated,
                        stage_peak_reserved_bytes=stage_peak_reserved)
        # Lifetime: both CUDA endpoint events have completed before D2H,
        # ordinary VAE decode, state release, or deletion of either output.
        measured["a_latent"] = _hash_array(state.latents.detach().float().cpu().numpy())
        measured["b_latent"] = _hash_array(b_state.latents.detach().float().cpu().numpy())
        measured["b_video"] = _hash_array(b_video.float().cpu().numpy())
        measured["b_output_complete_ms"] = (time.perf_counter_ns() - full_started) / 1e6
        if config["release_b_before_a_decode"]:
            # Both endpoint events and B's D2H/hash completed above.
            b_video = None
            worker.release(b_state)
            b_state = None
            observe("after_b_release")
        if config["decode_a_on_b_stream"]:
            # Opt-in lifecycle experiment: both completed requests use the same
            # VAE stream, avoiding a second stream-local VAE allocator cache.
            # Both stage endpoints have already completed above. The ordinary
            # finish operation still performs its normal synchronization/checks.
            state.latents.record_stream(b_stream)
            with torch.cuda.stream(b_stream):
                measured["a_video"] = _hash_array(worker.finish(state))
        else:
            measured["a_video"] = _hash_array(worker.finish(state))
        measured["a_decode_stream"] = "b_stream" if config["decode_a_on_b_stream"] else "current_stream"
        observe("after_a_decode")
        measured["correctness"] = _exact_result(measured, ref_a["hashes"], ref_b["hashes"])
        full_finished = time.perf_counter_ns()
        sampler.stop()
        sampler_stopped = True
        measured.update(started_monotonic_ns=full_started, finished_monotonic_ns=full_finished,
                        wall_ms=(full_finished-full_started)/1e6,
                        a_output_complete_ms=(full_finished-full_started)/1e6,
                        peak_allocated_bytes=max(prefix_peak_allocated,
                                                 torch.cuda.max_memory_allocated(device)),
                        peak_reserved_bytes=max(prefix_peak_reserved,
                                                torch.cuda.max_memory_reserved(device)),
                        completion_policy="B D2H/hash after both stage endpoints; A VAE/D2H/hash last")
        measured["memory_boundaries"] = memory_boundaries
        measured["memory_diagnostics"] = config["memory_diagnostics"]
        measured.update(nvml=sampler.summary(full_started, full_finished), nvml_samples=sampler.samples,
                        sample_interval_seconds=config["sample_interval_seconds"],
                        memory_margin_gate=memory_margin_gate(sampler.samples, sampler.errors,
                                                              full_started, full_finished))
        measured["performance_evidence"] = (purpose == "timed" and measured["correctness"]["passed"] and
                                           measured["memory_margin_gate"]["eligible"] and
                                           not config["memory_diagnostics"])
        return measured
    finally:
        # Failure cleanup may synchronize. A failed arm never supplies a timing
        # or speedup. In success this is after all measured work and validation.
        try:
            torch.cuda.synchronize(device)
            if state is not None:
                worker.release(state)
            if b_state is not None:
                worker.release(b_state)
            b_state = b_video = state = None
        finally:
            if not sampler_stopped:
                sampler.stop()


def _measure(worker, config, out, result):
    timeline = Timeline(config["device"])
    references = _references(worker, config, out, timeline)
    if config["trim_after_references"]:
        # Startup-only intervention, applied equally before all arms. Never
        # hide a per-request cache flush outside the request timer.
        started = time.perf_counter_ns()
        worker.torch.cuda.synchronize(worker.device)
        before = _memory_boundary(worker, config, "after_reference_warmup")
        worker.torch.cuda.empty_cache()
        result["startup_allocator_preparation"] = dict(
            wall_ms=(time.perf_counter_ns()-started)/1e6, before=before,
            after=_memory_boundary(worker, config, "after_reference_trim"),
            scope="once after reference warmup, excluded from hot request timing")
    result["reference_artifacts"] = [dict(role=role, frames=frames, path=row["directory"],
                                          **row["hashes"]) for (role, frames), row in references.items()]
    result["qualification"] = dict(passed=False, exact_required=True, pairs=[])
    # Global gate: every configured pair must bridge to the ordinary runtime
    # and then pass async-vs-serial exactly before any performance repeats.
    for frames_a, frames_b in config["frame_pairs"]:
        arms = {}
        for mode in ("serial", "streams"):
            arm = _run_arm(worker, config, references, frames_a, frames_b, mode, -1, "qualification", timeline)
            arms[mode] = arm
            atomic_json(Path(out) / f"qualification-a{frames_a}-b{frames_b}-{mode}.json", arm)
            if not arm["correctness"]["passed"]:
                break
            if not arm["memory_margin_gate"]["eligible"]:
                break
        passed = len(arms) == 2 and all(arm["correctness"]["passed"] for arm in arms.values())
        row = dict(frames_a=frames_a, frames_b=frames_b, passed=passed, arms=arms)
        result["qualification"]["pairs"].append(row)
        atomic_json(Path(out) / "run.json", result)
        capacity = all(arm["memory_margin_gate"]["eligible"] for arm in arms.values())
        if not capacity:
            result.update(status="capacity_failed", performance_evidence=False, phases=timeline.records,
                          capacity_reason="insufficient or unverified sampled memory margin", failed_pair=row)
            return
        if not passed:
            result.update(status="qualification_failed", phases=timeline.records)
            return
    result["qualification"]["passed"] = True
    for frames_a, frames_b in config["frame_pairs"]:
        for repeat in range(config["repeats"]):
            order = ("serial", "streams") if repeat % 2 == 0 else ("streams", "serial")
            arms = {}
            for mode in order:
                arm = _run_arm(worker, config, references, frames_a, frames_b, mode, repeat, "timed", timeline)
                arms[mode] = arm
                atomic_json(Path(out) / f"a{frames_a}-b{frames_b}-r{repeat}-{mode}.json", arm)
                if not arm["correctness"]["passed"]:
                    result.update(status="correctness_failed", failed_arm=arm, phases=timeline.records)
                    return
                if not arm["memory_margin_gate"]["eligible"]:
                    result.update(status="capacity_failed", performance_evidence=False, failed_arm=arm,
                                  phases=timeline.records,
                                  capacity_reason="insufficient or unverified sampled memory margin")
                    return
            result["pairs"].append(dict(frames_a=frames_a, frames_b=frames_b, repeat=repeat,
                                        order=list(order), modes=arms,
                                        speedup=arms["serial"]["wall_ms"]/arms["streams"]["wall_ms"],
                                        stage_pair_speedup=arms["serial"]["stage_pair_wall_ms"]/
                                                           arms["streams"]["stage_pair_wall_ms"]))
            atomic_json(Path(out) / "run.json", result)
    result["summary"] = [dict(frames_a=a, frames_b=b, repeats=config["repeats"],
                              median_paired_speedup=statistics.median(
                                  row["speedup"] for row in result["pairs"]
                                  if (row["frames_a"], row["frames_b"]) == (a, b)))
                         for a, b in config["frame_pairs"]]
    result.update(status="completed", phases=timeline.records,
                  performance_evidence=not config["memory_diagnostics"])


def run_overlap(config, out):
    config, out = normalize_config(config), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError("refusing to overwrite " + str(out / "run.json"))
    result = dict(schema=SCHEMA, kind="P4", status="running", configuration=config,
                  started_unix_ns=time.time_ns(), qualification=None, pairs=[],
                  timing_scope=TIMING_SCOPE, overlap_requires_cuda_trace=True,
                  performance_evidence=False,
                  arithmetic=dict(dit="float16", vae="float32", encoder="cpu float32",
                                  matmul_tf32=False, cudnn_tf32=True),
                  submission="one host thread; per-call explicit Flash DiT and MATH VAE; two CUDA streams")
    atomic_json(out / "run.json", result)
    try:
        worker = make_worker(config, config["device"], out)
        result["cpu_affinity"] = worker.workflow_cpu_affinity
        _measure(worker, config, out, result)
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    finally:
        result["finished_unix_ns"] = time.time_ns()
        atomic_json(out / "run.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_overlap(json.loads(args.config.read_text()), args.out)
    print(json.dumps(dict(status=result["status"], manifest=str(args.out / "run.json"))))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
