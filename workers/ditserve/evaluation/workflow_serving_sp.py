"""Numerical SP diagnosis on prefixes of one fixed 20-step Wan schedule.

This is numerical evidence only. Copies, hooks, barriers, and diagnostic writes
are deliberately outside any performance claim. A failed 1/4-step gate prevents
the respective variant from advancing to 4/20 steps; tolerance is never relaxed.
"""

import argparse
import contextlib
import dataclasses
import datetime
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

from .workflow_overlap import Timeline, _decode_gpu, _hash_array, atomic_json
from .workflow_serving_overlap import make_worker, normalize_config as overlap_config, request_spec
from .workflow_attention import DIAGNOSTIC_VARIANTS


SCHEMA = "ditserve.workflow.serving.sp.v1"
DEFAULT_VARIANTS = ("cache_dit", "baseline", "qkv_pipeline", "heads_pipeline")
VARIANTS = DEFAULT_VARIANTS + DIAGNOSTIC_VARIANTS
PREFIXES = (1, 4, 20)
THRESHOLD = 0.05


def normalize_config(config):
    value = overlap_config(config)
    value.setdefault("devices", ["cuda:0", "cuda:1"])
    value.setdefault("frames", [17, 33])
    value.setdefault("variants", list(DEFAULT_VARIANTS))
    value.setdefault("threshold", THRESHOLD)
    value.setdefault("timeout_seconds", 3600)
    value.setdefault("group_timeout_seconds", 60)
    value.setdefault("extended_diagnostics", False)
    value.setdefault("save_videos", False)
    value.setdefault("localize_modes", list(value["variants"]))
    value.setdefault("seeds", [value["seed"]])
    if type(value["frames"]) is int:
        value["frames"] = [value["frames"]]
    if (not value["frames"] or any(type(frame) is not int or frame < 1 or (frame-1) % 4
                                   for frame in value["frames"])):
        raise ValueError("frames must contain frames=4*k+1")
    if (len(value["devices"]) != 2 or len(set(value["devices"])) != 2 or any(
            not str(device).startswith("cuda:") or not str(device)[5:].isdigit()
            for device in value["devices"])):
        raise ValueError("P5 requires two distinct explicit cuda:N devices")
    if value["threshold"] != THRESHOLD:
        raise ValueError("P5 fixes the numerical threshold at 0.05")
    if (not value["variants"] or len(set(value["variants"])) != len(value["variants"]) or
            any(mode not in VARIANTS for mode in value["variants"])):
        raise ValueError("unknown or duplicate SP variant")
    if type(value["timeout_seconds"]) is not int or value["timeout_seconds"] < 1:
        raise ValueError("timeout_seconds must be positive")
    if (type(value["group_timeout_seconds"]) not in (int, float) or
            not 0 < value["group_timeout_seconds"] <= 60):
        raise ValueError("group_timeout_seconds must be in (0, 60]")
    if type(value["extended_diagnostics"]) is not bool:
        raise ValueError("extended_diagnostics must be boolean")
    if type(value["save_videos"]) is not bool:
        raise ValueError("save_videos must be boolean")
    if (not isinstance(value["localize_modes"], list) or
            any(mode not in value["variants"] for mode in value["localize_modes"])):
        raise ValueError("localize_modes must be selected variants")
    if (not isinstance(value["seeds"], list) or not value["seeds"] or
            len(set(value["seeds"])) != len(value["seeds"]) or
            any(type(seed) is not int or seed < 0 for seed in value["seeds"])):
        raise ValueError("seeds must contain distinct nonnegative integers")
    if config.get("prefixes", list(PREFIXES)) != list(PREFIXES):
        raise ValueError("P5 requires sequential prefixes 1, 4, 20")
    value["prefixes"] = list(PREFIXES)
    return value


def comparison_passes(comparison, *, exact=False):
    """Reject absent/nonfinite/shape-mismatched evidence, including NaN errors."""
    if not (comparison.get("shape_match") and comparison.get("finite") and
            comparison.get("reference_finite")):
        return False
    error = comparison.get("max_abs")
    if type(error) not in (int, float) or not 0 <= error <= (0 if exact else THRESHOLD):
        return False
    return not exact or comparison.get("exact") is True


def active_variants(rows, prefix):
    """Only a globally passed predecessor permits another schedule prefix."""
    position = PREFIXES.index(prefix)
    return [row for row in rows if position == 0 or
            (len(row["prefixes"]) == position and row["prefixes"][-1]["passed"])]


def _cpu_array(tensor):
    return tensor.detach().float().cpu().numpy().copy()


def _statistics(array):
    import numpy as np
    array = np.asarray(array)
    finite = np.isfinite(array)
    result = dict(shape=list(array.shape), dtype=str(array.dtype), finite=bool(finite.all()),
                  nonfinite_count=int(array.size - np.count_nonzero(finite)))
    if result["finite"]:
        result.update(min=float(array.min()) if array.size else 0.0, max=float(array.max()) if array.size else 0.0,
                      mean=float(array.mean()), rms=float(np.sqrt(np.mean(array.astype(np.float64)**2))))
    return result


def _save_neighbors(directory, label, previous, current, reference_previous, reference_current):
    """Bounded artifact: inputs/outputs and CFG predictions around divergence."""
    import numpy as np
    paths = []
    for suffix, candidate, reference in (("previous", previous, reference_previous),
                                         ("current", current, reference_current)):
        if candidate is None:
            continue
        data = {"candidate_" + key: value for key, value in candidate.items()}
        data.update({"reference_" + key: value for key, value in reference.items()})
        path = directory / (label + "-" + suffix + ".npz")
        np.savez(path, **data)
        paths.append(str(path))
    return paths


@contextlib.contextmanager
def _reference_videos():
    """Keep decoded references only for one prefix's comparisons, never on disk."""
    videos = {}
    try:
        yield videos
    finally:
        videos.clear()


def _reference_video(directory, videos):
    key = str(Path(directory).resolve())
    if videos is not None and key in videos:
        return videos[key]
    # Explicit save_videos runs may also compare a retained reference. Default
    # runs always use the scoped in-memory reference, including failed arms.
    import numpy as np
    return np.load(Path(directory) / "video.npy", allow_pickle=False)


def _prefix_run(worker, spec, payload, prefix, directory, timeline, *, backend=None,
                mode="sp1", reference=None, exact=False, save_arrays=False, secondary_reference=None,
                video_cache=None, save_videos=False):
    """All runs use spec.steps==20; prefix only bounds the loop iteration count."""
    import numpy as np
    from .workflow_encoder import compare_arrays
    if spec["steps"] != 20 or prefix not in PREFIXES:
        raise ValueError("prefix diagnosis must retain the 20-step scheduler")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = hook = None
    rows, predictions = [], []
    previous = reference_previous = None
    first_nonzero = first_threshold = None
    artifacts = {}
    try:
        state = worker.prepare_encoded(spec, dataclasses.replace(payload, spec=spec, consumed=False))
        initial = _cpu_array(state.latents)
        schedule = [float(value) for value in state.scheduler.timesteps.detach().cpu().tolist()]
        if len(schedule) != 20:
            raise RuntimeError("scheduler did not retain the full 20-step grid")
        # Transformer.__call__ invokes this after the SP output gather and
        # unpatchification, so the two CFG predictions have full latent shape.
        hook = worker.pipe.transformer.register_forward_hook(
            lambda module, inputs, output: predictions.append(_cpu_array(output[0])))
        for step in range(prefix):
            predictions.clear()
            before = _cpu_array(state.latents)
            with timeline.phase("P5/dit_step", purpose="diagnostic", variant=mode,
                                frames=spec["frames"], prefix=prefix, step=step):
                (worker.step if backend is None else backend.step)(state)
            expected_predictions = 2 if spec["guidance_scale"] > 1 else 1
            if len(predictions) != expected_predictions:
                raise RuntimeError("unexpected transformer CFG hook count")
            arrays = dict(latent_before=before, latent_after=_cpu_array(state.latents), cond=predictions[0])
            if expected_predictions == 2:
                arrays["uncond"] = predictions[1]
            row = dict(step=step, completed_steps=step+1, timestep=schedule[step],
                       statistics={key: _statistics(array) for key, array in arrays.items()})
            if save_arrays:
                np.savez(directory / f"step-{step:02d}.npz", **arrays)
            if reference is None:
                np.savez(directory / f"step-{step:02d}.npz", **arrays)
                row["passed"] = all(value["finite"] for value in row["statistics"].values())
            else:
                with np.load(Path(reference) / f"step-{step:02d}.npz", allow_pickle=False) as stored:
                    expected = {key: stored[key] for key in stored.files}
                comparisons = {key: compare_arrays(array, expected[key]) for key, array in arrays.items()}
                # Latent-after is the numerical gate. Per-branch predictions
                # localize divergence; their scale is not an extra threshold.
                row.update(comparison=comparisons,
                           passed=(all(comparison_passes(item, exact=True) for item in comparisons.values())
                                   if exact else comparison_passes(comparisons["latent_after"])))
                nonzero = any(not item.get("exact", False) for item in comparisons.values())
                if nonzero and first_nonzero is None:
                    first_nonzero = step
                    artifacts["first_nonzero"] = _save_neighbors(directory, "first-nonzero", previous,
                                                                  arrays, reference_previous, expected)
                if not row["passed"] and first_threshold is None:
                    first_threshold = step
                    artifacts["first_threshold_failure"] = _save_neighbors(
                        directory, "first-threshold-failure", previous, arrays, reference_previous, expected)
                reference_previous = expected
            if secondary_reference is not None:
                with np.load(Path(secondary_reference) / f"step-{step:02d}.npz", allow_pickle=False) as stored:
                    row["candidate_sp1_comparison"] = {
                        key: compare_arrays(array, stored[key]) for key, array in arrays.items()}
            rows.append(row)
            previous = arrays
            atomic_json(directory / "progress.json", dict(
                status="running", mode=mode, prefix=prefix, scheduler_steps=20,
                timestep_grid=schedule, steps=rows, first_nonzero_step=first_nonzero,
                first_threshold_failure_step=first_threshold, artifacts=artifacts,
                performance_evidence=False))
        with timeline.phase("P5/vae_decode", purpose="diagnostic", variant=mode,
                            frames=spec["frames"], prefix=prefix):
            video = _decode_gpu(worker, state).float().cpu().numpy()
        video_hash = _hash_array(video)
        latent_hash = _hash_array(previous["latent_after"])
        if save_videos:
            np.save(directory / "video.npy", video, allow_pickle=False)
        output = dict(mode=mode, prefix=prefix, scheduler_steps=20, timestep_grid=schedule,
                      steps=rows, initial_latent=_hash_array(initial), latent=latent_hash, video=video_hash,
                      first_nonzero_step=first_nonzero, first_threshold_failure_step=first_threshold,
                      artifacts=artifacts, performance_evidence=False)
        if reference is None:
            output["passed"] = all(row["passed"] for row in rows)
        else:
            expected_video = _reference_video(reference, video_cache)
            reference_result = json.loads((Path(reference) / "result.json").read_text())
            output["initial_and_schedule_exact"] = (
                output["initial_latent"] == reference_result["initial_latent"] and
                output["timestep_grid"] == reference_result["timestep_grid"])
            output["video_comparison"] = compare_arrays(video, expected_video)
            output["passed"] = (output["initial_and_schedule_exact"] and
                                all(row["passed"] for row in rows) and comparison_passes(
                                    output["video_comparison"], exact=exact))
            if save_videos and not comparison_passes(output["video_comparison"], exact=exact):
                output["artifacts"]["failed_video"] = str(directory / "video.npy")
        if secondary_reference is not None:
            secondary = Path(secondary_reference)
            secondary_result = json.loads((secondary / "result.json").read_text())
            secondary_video = compare_arrays(video, _reference_video(secondary, video_cache))
            initial_exact = (output["initial_latent"] == secondary_result["initial_latent"] and
                             output["timestep_grid"] == secondary_result["timestep_grid"])
            output["comparison_to_candidate_sp1"] = dict(
                reference=str(secondary), initial_and_schedule_exact=initial_exact,
                video_comparison=secondary_video,
                passed=initial_exact and comparison_passes(secondary_video) and all(
                    comparison_passes(row["candidate_sp1_comparison"]["latent_after"]) for row in rows),
                scope="Additional same-recipe consistency; original SP1 remains the primary numerical gate.")
        atomic_json(directory / "result.json", output)
        atomic_json(directory / "progress.json", dict(status="completed", **output))
        if video_cache is not None and (reference is None or save_arrays):
            video_cache[str(directory.resolve())] = video
        return output
    finally:
        if hook is not None:
            hook.remove()
        if state is not None:
            worker.release(state)


def _all_pass(torch, device, value):
    decision = torch.tensor(int(bool(value)), device=device, dtype=torch.int32)
    torch.distributed.all_reduce(decision, op=torch.distributed.ReduceOp.MIN)
    return bool(decision.item())


def _layer_array(tensor, rank, *, reference):
    """SP1 block/QKV tensors are sliced exactly like SP2's sequence layout."""
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise RuntimeError("expected Wan block/projection tensor [batch, sequence, channels]")
    if reference:
        if tensor.shape[1] % 2:
            raise RuntimeError("SP1 reference sequence is not divisible by two")
        tensor = tensor.chunk(2, dim=1)[rank]
    # Preserve the original FP16/FP32 values, avoiding the large all-layer FP32
    # copy. compare_arrays promotes bounded chunks only while calculating errors.
    return tensor.detach().cpu().numpy().copy()


def _capture_layers(worker, spec, payload, target_step, directory, rank, *, backend=None,
                    reference=None, target_block=None, extended=False):
    """Replay only to one diagnostic step; collect blocks or one block's QKV.

    Reference files contain this rank's SP1 sequence slice, not all-rank data.
    Candidate files retain only the first differing layer in the block scan;
    targeted QKV diagnosis retains its small set of selected tensors.
    """
    import numpy as np
    from .workflow_encoder import compare_arrays
    if spec["steps"] != 20 or not 0 <= target_step < 20:
        raise ValueError("layer diagnosis retains the 20-step schedule")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reference = None if reference is None else Path(reference)
    rows, hooks, counts = [], [], {}
    expected_names = []
    holder = dict(step=-1)
    state, first = None, None
    blocks = worker.pipe.transformer.blocks
    output = dict(status="running", target_step=target_step, scheduler_steps=20,
                  rank=rank, scope="block_outputs" if target_block is None else "selected_block_qkv",
                  target_block=target_block, layers=rows, first_divergence=None,
                  comparison_layout="SP2 local sequence vs matching SP1 contiguous sequence shard",
                  performance_evidence=False)

    def capture(name, index, tensor):
        nonlocal first
        if holder["step"] != target_step:
            return
        ordinal = counts.get(name, 0)
        counts[name] = ordinal + 1
        branch = ("cond", "uncond")[ordinal] if ordinal < 2 else None
        if branch is None:
            raise RuntimeError("unexpected repeated Wan layer invocation within CFG step")
        key = f"{branch}.block{index:02d}.{name}"
        array = _layer_array(tensor, rank, reference=reference is None)
        row = dict(key=key, block=index, branch=branch, tensor=name, statistics=_statistics(array))
        path = directory / (key + ".npy")
        if reference is None:
            np.save(path, array, allow_pickle=False)
            row["path"] = str(path)
            row["hash"] = _hash_array(array)
        else:
            reference_path = reference / (key + ".npy")
            expected = np.load(reference_path, mmap_mode="r", allow_pickle=False)
            row.update(reference=str(reference_path), reference_statistics=_statistics(expected),
                       comparison=compare_arrays(array, expected))
            differs = not row["comparison"].get("exact", False)
            if differs and first is None:
                first = dict(key=key, block=index, branch=branch, tensor=name,
                             comparison=row["comparison"], reference=str(reference_path), candidate=str(path))
                output["first_divergence"] = first
            if target_block is not None or (first is not None and first["key"] == key):
                np.save(path, array, allow_pickle=False)
                row["path"] = str(path)
        rows.append(row)
        atomic_json(directory / "capture.json", output)

    def attach(module, name, index):
        expected_names.append(name)
        hooks.append(module.register_forward_hook(
            lambda module, inputs, result: capture(name, index, result)))

    try:
        if target_block is None:
            for index, block in enumerate(blocks):
                # A unique counter is required per block, not one shared
                # "output" count across the thirty-layer traversal.
                name = f"output{index:02d}"
                attach(block, name, index)
        else:
            if not 0 <= target_block < len(blocks):
                raise ValueError("diagnostic block outside the Wan model")
            block = blocks[target_block]
            def before(module, arguments, keywords):
                value = keywords.get("hidden_states", arguments[0] if arguments else None)
                if value is None:
                    raise RuntimeError("Wan block hidden_states argument missing")
                capture("input", target_block, value)
            hooks.append(block.register_forward_pre_hook(before, with_kwargs=True))
            expected_names.append("input")
            for name in ("to_q", "to_k", "to_v", "norm_q", "norm_k"):
                attach(getattr(block.attn1, name), name, target_block)
            attach(block.attn1.to_out[0], "attention_output_projection", target_block)
            if extended:
                def attach_input(module, name):
                    def capture_input(module, arguments, keywords):
                        value = arguments[0] if arguments else keywords.get("input", keywords.get("hidden_states"))
                        if value is None:
                            raise RuntimeError("diagnostic module input missing: " + name)
                        capture(name, target_block, value)
                    # The causal Linear control pads in its own prehook.
                    # Capture actual local model input before that intervention.
                    hooks.append(module.register_forward_pre_hook(
                        capture_input, with_kwargs=True, prepend=True))
                    expected_names.append(name)
                attach_input(block.attn1.to_q, "to_q_input")
                ffn = getattr(block, "ffn", None)
                if ffn is not None and hasattr(ffn, "register_forward_pre_hook"):
                    attach_input(ffn, "feed_forward_input")
                    net = getattr(ffn, "net", None)
                    if net is not None and len(net) > 2:
                        projection = getattr(net[0], "proj", None)
                        if projection is not None and hasattr(projection, "register_forward_hook"):
                            attach(projection, "feed_forward_first_projection", target_block)
                        if hasattr(net[0], "register_forward_hook"):
                            attach(net[0], "feed_forward_activation", target_block)
                        if hasattr(net[2], "register_forward_hook"):
                            attach(net[2], "feed_forward_second_projection", target_block)
                # The projection input is attention output after inverse
                # exchange, before the output GEMM. It is sequence-local in
                # SP2 and can be compared with the matching SP1 shard.
                def attention_result(module, arguments, keywords):
                    value = arguments[0] if arguments else keywords.get("input")
                    if value is None:
                        raise RuntimeError("attention output projection input missing")
                    capture("attention_result", target_block, value)
                expected_names.append("attention_result")
                hooks.append(block.attn1.to_out[0].register_forward_pre_hook(
                    attention_result, with_kwargs=True, prepend=True))
                for attribute, name in (("attn1", "self_attention"),
                                        ("attn2", "cross_attention"),
                                        ("ffn", "feed_forward"),
                                        ("norm1", "norm1"),
                                        ("norm2", "norm2"),
                                        ("norm3", "norm3")):
                    module = getattr(block, attribute, None)
                    if module is not None and hasattr(module, "register_forward_hook"):
                        attach(module, name, target_block)
            attach(block, "output", target_block)
        state = worker.prepare_encoded(spec, dataclasses.replace(payload, spec=spec, consumed=False))
        for step in range(target_step + 1):
            holder["step"] = step
            (worker.step if backend is None else backend.step)(state)
        expected_calls = (2 if spec["guidance_scale"] > 1 else 1)
        expected_layers = len(expected_names)
        if len(rows) != expected_layers * expected_calls or any(count != expected_calls for count in counts.values()):
            raise RuntimeError("incomplete block/QKV diagnostic capture")
        output.update(status="completed", extended=extended, expected_tensors=expected_names,
                      all_exact=(None if reference is None else
                      all(row["comparison"].get("exact", False) for row in rows)))
        atomic_json(directory / "capture.json", output)
        return output
    except Exception:
        output.update(status="failed", error=traceback.format_exc())
        atomic_json(directory / "capture.json", output)
        raise
    finally:
        for hook in reversed(hooks):
            hook.remove()
        if state is not None:
            worker.release(state)


def _localize_failure(worker, spec, payload, candidate, rank, groups, directory, *, extended=False):
    """Two-pass block/QKV replay, only after a numerical qualification failure."""
    from ditserve.models.wan.parallel import LayoutHandle
    from ditserve.models.wan.sp import PinnedWanUlyssesBackend
    from .workflow_attention import attention_variant
    torch = worker.torch
    steps = [None, None]
    first_step = candidate["first_nonzero_step"]
    if first_step is None:
        first_step = candidate["first_threshold_failure_step"]
    torch.distributed.all_gather_object(steps, first_step)
    step = min((value for value in steps if value is not None), default=candidate["prefix"] - 1)
    directory = Path(directory)
    shared = directory / "reference" / f"step{step:02d}"
    reference_blocks = shared / "blocks"
    mode = candidate["mode"]
    destination = directory / mode / f"prefix{candidate['prefix']}" / f"step{step:02d}"

    def reference_ready(path, block=None):
        manifest = path / "capture.json"
        if manifest.is_file() and json.loads(manifest.read_text()).get("status") == "completed":
            return
        _capture_layers(worker, spec, payload, step, path, rank, target_block=block, extended=extended)

    def candidate_capture(path, reference, block=None):
        backend = PinnedWanUlyssesBackend(LayoutHandle(rank, (0, 1)), groups).attach(worker)
        try:
            with attention_variant(backend, mode) as audit:
                captured = _capture_layers(worker, spec, payload, step, path, rank, backend=backend,
                                           reference=reference, target_block=block, extended=extended)
            captured["intervention_audit"] = audit
            atomic_json(path / "capture.json", captured)
            return captured
        finally:
            backend.close()

    reference_ready(reference_blocks)
    scanned = candidate_capture(destination / "blocks", reference_blocks)
    summary = dict(status="completed", target_step=step, mode=mode, rank=rank,
                   block_scan=str(destination / "blocks" / "capture.json"),
                   first_divergence=scanned["first_divergence"],
                   reference_blocks=str(reference_blocks / "capture.json"),
                   detailed_capture=None, performance_evidence=False,
                   gate_unchanged=True, note="layer differences diagnose arithmetic; they do not relax the 0.05 request gate")
    # Both ranks replay the same number of distributed forwards. The earliest
    # block across either rank is inspected by both; local first-difference
    # indices can legitimately differ without changing collective order.
    locations = [None, None]
    torch.distributed.all_gather_object(locations, None if scanned["first_divergence"] is None else
                                       scanned["first_divergence"]["block"])
    block = min((value for value in locations if value is not None), default=None)
    if block is not None:
        reference_detail = shared / f"block{block:02d}-qkv"
        reference_ready(reference_detail, block)
        detailed = candidate_capture(destination / f"block{block:02d}-qkv", reference_detail, block)
        summary.update(selected_block=block, detailed_capture=str(destination / f"block{block:02d}-qkv" / "capture.json"),
                       detailed_first_divergence=detailed["first_divergence"])
    atomic_json(destination / "localization.json", summary)
    return summary


def _sp_child(config, rank, rendezvous, out):
    out = Path(out)
    directory = out / f"rank{rank}"
    directory.mkdir(parents=True, exist_ok=True)
    device = config["devices"][rank]
    result = dict(rank=rank, device=device, status="running", qualification=[], variants=[])
    backend = None
    worker = None
    atomic_json(directory / "run.json", result)
    try:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        if os.environ["NCCL_P2P_DISABLE"] != "1":
            raise ValueError("P5 retains NCCL_P2P_DISABLE=1; do not change the measured transport")
        worker = make_worker(config, device, directory)
        torch = worker.torch
        from ditserve.models.wan.parallel import LayoutHandle, build_fixed_groups
        from ditserve.models.wan.sp import PinnedWanUlyssesBackend
        from .workflow_attention import arithmetic_flags, attention_variant, sp1_tile_variant, TILE_VARIANTS
        torch.distributed.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                                             timeout=datetime.timedelta(seconds=120))
        groups = build_fixed_groups(timeout_seconds=config["group_timeout_seconds"], backend="nccl")
        timeline = Timeline(device)
        result.update(cpu_affinity=worker.workflow_cpu_affinity,
                      reference_arithmetic_flags=arithmetic_flags(torch),
                      environment={key: os.environ.get(key) for key in
                                   ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "TORCH_NCCL_BLOCKING_WAIT")})
        for seed, frames in ((seed, frames) for seed in config["seeds"] for frames in config["frames"]):
            case_directory = directory if len(config["seeds"]) == 1 else directory / f"seed{seed}"
            spec = request_spec(dict(config, seed=seed), frames, "A", f"P5-s{seed}-f{frames}")
            payload = worker.encode_prompt(spec)
            inputs = dict(prompt=_hash_array(_cpu_array(payload.prompt_embeds)),
                          negative=None if payload.negative_prompt_embeds is None else
                          _hash_array(_cpu_array(payload.negative_prompt_embeds)))
            peers = [None, None]
            torch.distributed.all_gather_object(peers, inputs)
            if peers[0] != peers[1]:
                raise RuntimeError("SP ranks encoded different CPU payloads")
            modes = [dict(seed=seed, frames=frames, mode=mode, prefixes=[], status="pending",
                          diagnostic_only=mode in DIAGNOSTIC_VARIANTS,
                          deployment_recipe=mode == "cache_dit") for mode in config["variants"]]
            result["variants"].extend(modes)
            for prefix in PREFIXES:
                active = active_variants(modes, prefix)
                if not active:
                    break
                with _reference_videos() as video_cache:
                    base = case_directory / f"f{frames}" / f"prefix{prefix}"
                    reference_dir = base / "sp1-reference"
                    reference = _prefix_run(worker, spec, payload, prefix, reference_dir, timeline,
                        video_cache=video_cache, save_videos=config["save_videos"])
                    replay = _prefix_run(worker, spec, payload, prefix, base / "sp1-replay", timeline,
                                         reference=reference_dir, exact=True,
                                         video_cache=video_cache, save_videos=config["save_videos"])
                    # Both SP1 devices must have identical initial/schedule/final
                    # evidence before attributing their SP2 differences to sharding.
                    signatures = [None, None]
                    signature = {key: reference[key] for key in ("initial_latent", "timestep_grid", "latent", "video")}
                    torch.distributed.all_gather_object(signatures, signature)
                    qualified = _all_pass(torch, device, reference["passed"] and replay["passed"] and
                                          signatures[0] == signatures[1])
                    result["qualification"].append(dict(seed=seed, frames=frames, prefix=prefix, sp1_exact=qualified,
                                                         reference=str(reference_dir), replay=replay,
                                                         cross_rank_exact=signatures[0] == signatures[1]))
                    if not qualified:
                        for row in active:
                            row.update(status="sp1_qualification_failed", stopped_before_prefix=prefix)
                        break
                    for row in active:
                        mode = row["mode"]
                        bridge = None
                        tile_reference_dir = None
                        if mode in TILE_VARIANTS:
                            tile_reference_dir = base / (mode + "-sp1-reference")
                            with sp1_tile_variant(worker, mode, rank) as bridge_audit:
                                bridge = _prefix_run(worker, spec, payload, prefix, tile_reference_dir, timeline,
                                    mode=mode+"-sp1", reference=reference_dir, save_arrays=True,
                                    video_cache=video_cache, save_videos=config["save_videos"])
                            bridge["intervention_audit"] = bridge_audit
                            with sp1_tile_variant(worker, mode, rank) as replay_audit:
                                tile_replay = _prefix_run(worker, spec, payload, prefix,
                                    base / (mode+"-sp1-replay"), timeline, mode=mode+"-sp1-replay",
                                    reference=tile_reference_dir, exact=True,
                                    video_cache=video_cache, save_videos=config["save_videos"])
                            tile_replay["intervention_audit"] = replay_audit
                            tile_signatures = [None, None]
                            torch.distributed.all_gather_object(tile_signatures,
                                {key: bridge[key] for key in ("initial_latent", "timestep_grid", "latent", "video")})
                            bridge.update(replay=tile_replay, cross_rank_exact=tile_signatures[0] == tile_signatures[1],
                                          scope="Candidate SP1 compared to unchanged original SP1.")
                            bridge["passed"] = _all_pass(torch, device, bridge["passed"] and tile_replay["passed"] and
                                bridge_audit["applied_calls"] > 0 and replay_audit["applied_calls"] > 0 and
                                bridge["cross_rank_exact"])
                            atomic_json(tile_reference_dir / "result.json", bridge)
                            if not bridge["passed"]:
                                candidate = dict(mode=mode, prefix=prefix, passed=False, local_passed=False,
                                    threshold=THRESHOLD, bridge_to_original=bridge, diagnostic_only=True,
                                    status="candidate_sp1_bridge_failed", performance_evidence=False,
                                    sp2_not_run="candidate SP1 failed the original-reference or repeatability gate")
                                row["prefixes"].append(candidate)
                                row.update(status="numerical_failed",
                                           not_run_prefixes=[value for value in PREFIXES if value > prefix])
                                atomic_json(base / mode / "result.json", candidate)
                                atomic_json(directory / "run.json", result)
                                video_cache.pop(str(tile_reference_dir.resolve()), None)
                                continue
                        backend = PinnedWanUlyssesBackend(LayoutHandle(rank, (0, 1)), groups).attach(worker)
                        try:
                            with attention_variant(backend, mode) as audit:
                                candidate = _prefix_run(worker, spec, payload, prefix, base / mode, timeline,
                                                        backend=backend, mode=mode, reference=reference_dir,
                                                        secondary_reference=tile_reference_dir,
                                                        video_cache=video_cache, save_videos=config["save_videos"])
                            candidate["intervention_audit"] = audit
                            candidate["diagnostic_only"] = mode in DIAGNOSTIC_VARIANTS
                            if "applied_calls" in audit and not audit["applied_calls"]:
                                raise RuntimeError(mode + " did not exercise any token Linear")
                            if bridge is not None:
                                candidate["bridge_to_original"] = bridge
                                candidate["passed"] = candidate["passed"] and candidate["comparison_to_candidate_sp1"]["passed"]
                            atomic_json(base / mode / "result.json", candidate)
                        finally:
                            backend.close()
                            backend = None
                        if tile_reference_dir is not None:
                            video_cache.pop(str(tile_reference_dir.resolve()), None)
                        candidate["local_passed"] = candidate["passed"]
                        candidate["passed"] = _all_pass(torch, device, candidate["passed"])
                        candidate["threshold"] = THRESHOLD
                        row["prefixes"].append(candidate)
                        row["status"] = ("qualified" if prefix == 20 else "prefix_passed") if candidate["passed"] else "numerical_failed"
                        if not candidate["passed"]:
                            row["not_run_prefixes"] = [value for value in PREFIXES if value > prefix]
                        atomic_json(directory / "run.json", result)
                        if not candidate["passed"] and mode in config["localize_modes"]:
                            candidate["localization"] = dict(status="running", performance_evidence=False)
                            atomic_json(directory / "run.json", result)
                            candidate["localization"] = _localize_failure(
                                worker, spec, payload, candidate, rank, groups, case_directory / f"f{frames}" / "localization",
                                extended=config["extended_diagnostics"])
                            atomic_json(directory / "run.json", result)
        result["phases"] = timeline.records
        result["status"] = "completed" if all(row["status"] == "qualified" for row in result["variants"]) else "qualification_failed"
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    finally:
        if backend is not None:
            backend.close()
        result["finished_unix_ns"] = time.time_ns()
        atomic_json(directory / "run.json", result)
        if worker is not None and worker.torch.distributed.is_initialized():
            worker.torch.distributed.destroy_process_group()


def _run_ranks(config, out):
    context = multiprocessing.get_context("spawn")
    rendezvous = out / "nccl-rendezvous"
    if rendezvous.exists():
        raise FileExistsError(rendezvous)
    processes = [context.Process(target=_sp_child, args=(config, rank, "file://"+str(rendezvous.resolve()), out))
                 for rank in range(2)]
    deadline = time.monotonic() + config["timeout_seconds"]
    try:
        for process in processes:
            process.start()
        while any(process.is_alive() for process in processes):
            if time.monotonic() > deadline:
                raise TimeoutError("P5 numerical diagnosis exceeded its time budget")
            for rank, process in enumerate(processes):
                path = out / f"rank{rank}" / "run.json"
                if path.is_file() and json.loads(path.read_text()).get("status") == "failed":
                    raise RuntimeError(f"SP rank {rank} failed; see {path}")
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"SP rank {rank} exited {process.exitcode}")
            for process in processes:
                process.join(timeout=0.2)
        ranks = [json.loads((out / f"rank{rank}" / "run.json").read_text()) for rank in range(2)]
        if any(row["status"] == "failed" for row in ranks):
            raise RuntimeError("SP diagnostic rank failed; see rank manifests")
        return ranks
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)


def run_sp(config, out):
    config, out = normalize_config(config), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError("refusing to overwrite " + str(out / "run.json"))
    result = dict(schema=SCHEMA, kind="P5", status="running", configuration=config,
                  started_unix_ns=time.time_ns(), qualification=None, variants=[],
                  performance_evidence=False, scheduler_steps=20, prefixes=list(PREFIXES),
                  threshold=THRESHOLD, gate="every latent-after step and decoded prefix video; max_abs <= 0.05",
                  localization=("first nonzero and first threshold-exceeding step; previous/current latent and CFG predictions; "
                                "failed variants replay first differing step for all block outputs then selected block input/Q/K/V/"
                                "norm-Q/norm-K/output projection, comparing SP2 local sequence with matching SP1 shard"))
    atomic_json(out / "run.json", result)
    try:
        ranks = _run_ranks(config, out)
        numerical = len(ranks) == 2 and all(row["status"] == "completed" for row in ranks)
        expected = {(seed, frames) for seed in config["seeds"] for frames in config["frames"]}
        native = [row for rank in ranks for row in rank["variants"] if row["mode"] == "cache_dit"]
        deployment = len(ranks) == 2 and len(native) == 2 * len(expected) and all(
            {(row.get("seed"), row.get("frames")) for row in rank["variants"]
             if row["mode"] == "cache_dit" and row["status"] == "qualified"} == expected
            for rank in ranks)
        diagnostic = any(mode in DIAGNOSTIC_VARIANTS for mode in config["variants"])
        result.update(ranks=ranks, variants=ranks[0]["variants"],
                      qualification=dict(passed=numerical,
                                         numerical_variants_passed=numerical,
                                         diagnostic_only=diagnostic,
                                         scope="diagnostic_numerical_only" if diagnostic else "numerical_only",
                                         sp1=[row["qualification"] for row in ranks]),
                      role="diagnostic_only" if diagnostic else "numerical_control",
                      deployment_qualification=dict(passed=deployment, required_variant="cache_dit",
                          scope="Unmodified native processor only; diagnostic interventions cannot qualify deployment."),
                      status="completed" if numerical else "qualification_failed")
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        result["rank_manifests"] = [str(path) for path in sorted(out.glob("rank*/run.json"))]
    finally:
        result["finished_unix_ns"] = time.time_ns()
        atomic_json(out / "run.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_sp(json.loads(args.config.read_text()), args.out)
    print(json.dumps(dict(status=result["status"], manifest=str(args.out / "run.json"))))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
