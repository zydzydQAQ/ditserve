"""Four-process static and cache-off elastic SP measurements, not qualifications.

The charged research runner owns the GPU ledger and outer process-group
watchdog. This module adds finite collective timeouts and a local child watchdog.
Static cases use fresh requests; a later unit transports one request at
completed steps 16 and 24, alongside fixed SP4/SP2 control trajectories.
The t008 static unit requires a completed off unit and a complete-unit budget
estimate. The default order is off, t008, then cache-off elastic, as budget allows.
CUDA and model imports happen only inside explicitly launched rank processes.
"""

import datetime
import hashlib
import multiprocessing
import os
from pathlib import Path
import queue
import time
import traceback

from .common import atomic_json, digest, file_digest, finite, read_json
from .parallel import ParallelContractError


CASES = ("sp1", "two_sp2", "sp4")
STATIC_RECIPES = {"off": dict(cache=False, threshold=0.0, max_continuous=1),
                  "t008": dict(cache=True, threshold=0.08, max_continuous=1)}


def _static_recipe(job):
    recipe = job.get("static_recipe", "off")
    if type(recipe) is not str or recipe not in STATIC_RECIPES:
        raise ValueError("static_recipe must be off or t008")
    return recipe


def rank_case(case, rank):
    """Return the fixed group and input index; all ranks have real work."""
    if type(rank) is not int or rank not in range(4):
        raise ValueError("static SP requires ranks 0..3")
    if case == "sp1":
        return (rank,), 0 if rank < 2 else 1
    if case == "two_sp2":
        return ((0, 1), 0) if rank < 2 else ((2, 3), 1)
    if case == "sp4":
        return (0, 1, 2, 3), 0
    raise ValueError("unsupported static case")


def static_launch_plan(job):
    """Validate launch arguments without importing torch or touching a GPU."""
    from ..gpu.worker import normalize_spec
    recipe = _static_recipe(job)
    cache_spec = STATIC_RECIPES[recipe]
    root = Path(job["output"]).resolve()
    frozen = read_json(root / "protocol.json")
    config = frozen["config"]
    devices = job["physical_devices"]
    if (type(devices) is not list or len(devices) != 4
            or len({row["uuid"] for row in devices}) != 4
            or len({row["name"] for row in devices}) != 1):
        raise ValueError("static SP requires four distinct homogeneous devices")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if visible != [row["uuid"] for row in devices]:
        raise ValueError("parent must expose exactly the four assigned GPU UUIDs in rank order")
    deadline = finite(job["deadline_monotonic"], "stage deadline", positive=True)
    if deadline <= time.monotonic():
        raise TimeoutError("static SP stage deadline has expired")
    timeout = finite(job.get("timeout_seconds", 15), "collective timeout", positive=True)
    if timeout > 60:
        raise ValueError("collective timeout must not exceed 60 seconds")
    model_path = Path(job["model_path"]).resolve()
    if not (model_path / "model_index.json").is_file():
        raise ValueError("fixed local Wan model snapshot is missing")
    specs = []
    override = dict(job.get("spec", {}))
    for index, prompt in enumerate(config["prompts"][:2]):
        spec = dict(config["shapes"]["correctness"], request_id="static-input-%d" % index,
                    prompt=prompt["text"], negative_prompt="", seed=config["dev_seeds"][index],
                    guidance_scale=5.0, **cache_spec)
        spec.update(override)
        spec = normalize_spec(spec)
        if (any(spec[key] != value for key, value in cache_spec.items())
                or any(spec[key] != value for key, value in config["shapes"]["correctness"].items())):
            raise ValueError("static SP must use the frozen correctness shape and selected recipe")
        specs.append(spec)
    if len(specs) != 2 or specs[0]["prompt"] == specs[1]["prompt"]:
        raise ValueError("two independent requests require two different frozen prompts")
    return dict(output=str(root), model_path=str(model_path), specs=specs,
                physical_devices=devices, deadline_monotonic=deadline, timeout_seconds=timeout,
                world_size=4, backend="nccl", local_devices=list(range(4)), cases=list(CASES),
                static_recipe=recipe, artifact_subdir="static-sp" if recipe == "off" else "static-sp-t008",
                protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"],
                # CPU workers contend for host RAM/threads; their timings are
                # included in reserved-card wall time, not a serving claim.
                cpu_threads_per_rank=config.get("runtime_threads", {}).get("omp", 8))


def _cached_unit_budget(plan):
    """Admit all three fresh-request cases using an already measured off unit."""
    root = Path(plan["output"])
    path = root / "static-sp-off-unit.json"
    if not path.is_file():
        path = root / "static-sp-validation.json"
    if not path.is_file():
        raise ValueError("t008 requires a completed cache-off static unit")
    baseline = read_json(path)
    claimed_hash = baseline.get("sha256")
    if claimed_hash != digest({key: value for key, value in baseline.items() if key != "sha256"}):
        raise ValueError("cache-off static report hash differs from its contents")
    previous = baseline.get("plan", {})
    if (baseline.get("status") != "measured_unqualified" or baseline.get("cache_enabled") is not False
            or previous.get("static_recipe", "off") != "off"
            or any(previous.get(key) != plan[key] for key in (
                "protocol_sha256", "source_sha256", "model_path", "physical_devices", "cases"))):
        raise ValueError("t008 requires matching completed cache-off static evidence")
    old_specs = previous.get("specs", [])
    expected = [dict(spec, **STATIC_RECIPES["off"]) for spec in plan["specs"]]
    if old_specs != expected:
        raise ValueError("cached and cache-off unit inputs differ")
    seconds = finite(baseline.get("static_unit_wall_seconds", baseline.get("wall_seconds")),
                     "measured complete static unit seconds", positive=True)
    estimate = max(60., seconds * 1.25)
    remaining = max(0., plan["deadline_monotonic"] - time.monotonic())
    return dict(status="fits" if remaining > estimate else "budget_not_fit",
                next_complete_unit_estimate_seconds=estimate, remaining_seconds=remaining,
                estimate_scope="observed full off static unit including model load, multiplied by 1.25; minimum 60 seconds",
                baseline_report=path.name, baseline_report_sha256=claimed_hash,
                required_complete_cases=list(CASES))


def observed_path_costs(events, *, expected_steps, guidance_scale):
    """Summarize actual whole-step timings, leaving unobserved paths missing."""
    if type(events) is not list or len(events) != expected_steps:
        raise ParallelContractError("path observations require every completed step")
    paths = {name: [] for name in ("compute", "mixed", "reuse")}
    forward_count = 2 if guidance_scale > 1 else 1
    total_reused = 0
    for index, event in enumerate(events):
        reused = event.get("reused_forwards")
        if (type(reused) is not int or not 0 <= reused <= forward_count
                or event.get("step_index") != index or event.get("completed_steps") != index + 1):
            raise ParallelContractError("cache path counters differ from completed steps")
        expected_path = "compute" if reused == 0 else "reuse" if reused == forward_count else "mixed"
        if event.get("path") != expected_path:
            raise ParallelContractError("cache path label differs from observed reused forwards")
        costs = {name: finite(event[name], name) for name in ("ms", "gated_step_ms", "capture_inclusive_ms")}
        paths[expected_path].append(costs)
        total_reused += reused
    rows = {}
    for name, samples in paths.items():
        rows[name] = dict(observed_steps=len(samples),
                          **{key + "_total": sum(row[key] for row in samples) if samples else None
                             for key in ("ms", "gated_step_ms", "capture_inclusive_ms")})
    return dict(total_steps=expected_steps, observed_reused_forwards=total_reused,
                reuse_observed=total_reused > 0, full_reuse_step_observed=bool(paths["reuse"]), paths=rows,
                missing_observations=[name for name, samples in paths.items() if not samples],
                timing_scope="whole denoising steps grouped by observed path; mixed steps remain separate",
                serving_cost_eligible=False, calibrated_tolerances=None, cached_sp_qualified=False)


def _deadline(plan):
    remaining = plan["deadline_monotonic"] - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("static SP stage exhausted its parent allocation")
    return remaining


def _ready(plan, rank, phase, events, messages, payload=None):
    messages.put(dict(kind="ready", rank=rank, phase=phase, payload=payload))
    if not events[phase].wait(timeout=_deadline(plan)):
        raise TimeoutError("static SP preparation/decode synchronization timed out")


def _tensor_hash(tensor):
    if tensor is None:
        return None
    import torch
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(raw).hexdigest())


def _array_artifact(root, path, value):
    import numpy as np
    if not np.isfinite(value).all():
        raise ParallelContractError("nonfinite static SP artifact")
    np.save(path, value, allow_pickle=False)
    return dict(path=str(path.relative_to(root)), sha256=file_digest(path), shape=list(value.shape))


def _rank_main(rank, plan, rendezvous, events, messages):
    """One complete Wan worker/model per device; no request transport here."""
    backend = None
    try:
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        import numpy as np
        import torch
        from ..gpu.worker import WanStepWorker
        from .parallel import LayoutHandle, build_fixed_groups
        from .sp_backend import PinnedWanUlyssesBackend
        torch.set_num_threads(plan["cpu_threads_per_rank"])
        if torch.cuda.device_count() != 4:
            raise RuntimeError("rank process cannot see exactly four assigned GPUs")
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            "nccl", init_method="file://" + rendezvous, rank=rank, world_size=4,
            timeout=datetime.timedelta(seconds=plan["timeout_seconds"]),
            device_id=torch.device("cuda", rank))
        groups = build_fixed_groups(timeout_seconds=plan["timeout_seconds"], backend="nccl")
        handle = LayoutHandle(rank)
        load_started = time.perf_counter()
        worker = WanStepWorker(device="cuda:%d" % rank, model_path=plan["model_path"], attention="flash")
        model_load_ms = (time.perf_counter() - load_started) * 1000
        backend = PinnedWanUlyssesBackend(handle, groups).attach(worker)
        _ready(plan, rank, 0, events, messages)
        root = Path(plan["output"])
        records = []
        for case_index, case in enumerate(CASES):
            _deadline(plan)
            ranks, input_index = rank_case(case, rank)
            if ranks != handle.current.ranks:
                handle.commit(ranks, expected_epoch=handle.current.epoch)
            spec = dict(plan["specs"][input_index], request_id="static-%s-input%d" % (case, input_index))
            started = time.perf_counter()
            state = worker.prepare(spec)
            try:
                initial = dict(latent=_tensor_hash(state.latents), prompt=_tensor_hash(state.prompt_embeds),
                               negative=_tensor_hash(state.negative_prompt_embeds))
                _ready(plan, rank, 1 + 2 * case_index, events, messages, initial)
                torch.cuda.reset_peak_memory_stats(rank)
                latents = [state.latents.detach().cpu().numpy().copy()]
                costs = []
                for _ in range(spec["steps"]):
                    _deadline(plan)
                    step_started = time.perf_counter()
                    event = backend.step(state)
                    # Worker timing excludes gate/capture. Keep both measures;
                    # neither is an uninstrumented throughput benchmark.
                    gated_ms = (time.perf_counter() - step_started) * 1000
                    latents.append(state.latents.detach().cpu().numpy().copy())
                    costs.append(dict(event, gated_step_ms=gated_ms,
                                      capture_inclusive_ms=(time.perf_counter() - step_started) * 1000))
                prefix = root / plan["artifact_subdir"] / (case + "-rank%d" % rank)
                latent_artifact = _array_artifact(root, prefix.with_suffix(".latents.npy"), np.stack(latents))
                output_artifact = None
                if rank == ranks[0]:
                    output = worker.finish(state)
                    output_artifact = _array_artifact(root, prefix.with_suffix(".video.npy"), output)
                record = dict(case=case, rank=rank, ranks=list(ranks), input_index=input_index,
                              static_recipe=plan["static_recipe"],
                              model_load_ms=model_load_ms, model_load_is_once_per_rank_process=True,
                              layout_epoch=handle.current.epoch, fresh_request=True, spec=spec,
                              initial=initial, step_costs=costs, latents=latent_artifact, output=output_artifact,
                              path_observations=observed_path_costs(costs, expected_steps=spec["steps"],
                                                                    guidance_scale=spec["guidance_scale"]),
                              events=state.events, prepare_ms=state.prepare_ms, memory=worker._memory(),
                              wall_ms=(time.perf_counter() - started) * 1000)
                record_path = prefix.with_suffix(".json")
                atomic_json(record_path, record)
                records.append(dict(path=str(record_path.relative_to(root)), sha256=file_digest(record_path)))
            finally:
                worker.release(state)
            _ready(plan, rank, 2 + 2 * case_index, events, messages)
        backend.close()
        backend = None
        torch.distributed.destroy_process_group()
        messages.put(dict(kind="complete", rank=rank, records=records))
    except BaseException as error:
        # Do not attempt another distributed cleanup collective after failure.
        # The parent terminates all rank processes, including stuck peers.
        messages.put(dict(kind="error", rank=rank, error=repr(error), traceback=traceback.format_exc()))
        raise


def _validate_ready(phase, payloads):
    if phase % 2 == 0:
        return
    case = CASES[(phase - 1) // 2]
    for rank in range(4):
        ranks, _ = rank_case(case, rank)
        if any(payloads[member] != payloads[ranks[0]] for member in ranks):
            raise ParallelContractError("initial latent or conditioning differs within " + case)


def _launch(plan, *, context=None, target=None, ready_validator=None):
    """Bound child lifetime even when a peer never enters a collective."""
    context = context or multiprocessing.get_context("spawn")
    target = target or _rank_main
    messages = context.Queue()
    events = [context.Event() for _ in range(1 + 2 * len(CASES))]
    rendezvous = str(Path(plan["output"]) / plan.get("artifact_subdir", "static-sp") / "rendezvous")
    processes, complete, ready = [], {}, {}
    next_phase = [0] * 4
    try:
        for rank in range(4):
            process = context.Process(target=target, args=(rank, plan, rendezvous, events, messages))
            process.start()
            processes.append(process)
        while len(complete) != 4:
            remaining = _deadline(plan)
            try:
                message = messages.get(timeout=min(.1, remaining))
            except queue.Empty:
                for rank, process in enumerate(processes):
                    if process.exitcode is not None and rank not in complete:
                        raise RuntimeError("static SP rank %d exited without completion (code %s)" % (rank, process.exitcode))
                continue
            rank = message["rank"]
            if rank not in range(4):
                raise RuntimeError("invalid child rank")
            if message["kind"] == "error":
                raise RuntimeError("static SP rank %d failed: %s\n%s" % (rank, message["error"], message["traceback"]))
            if message["kind"] == "ready":
                phase = message["phase"]
                if phase != next_phase[rank] or phase >= len(events):
                    raise RuntimeError("duplicate or out-of-order rank readiness")
                next_phase[rank] += 1
                payloads = ready.setdefault(phase, {})
                payloads[rank] = message["payload"]
                if len(payloads) == 4:
                    (ready_validator or _validate_ready)(phase, payloads)
                    events[phase].set()
            elif message["kind"] == "complete":
                if next_phase[rank] != len(events) or rank in complete:
                    raise RuntimeError("rank completed without all static cases")
                complete[rank] = message["records"]
            else:
                raise RuntimeError("unknown child message")
        for process in processes:
            process.join(timeout=min(2, _deadline(plan)))
            if process.exitcode != 0:
                raise RuntimeError("static SP rank did not exit cleanly")
        return complete
    finally:
        # Fast kill follows a bounded grace period. The outer charged watchdog
        # also owns these descendants if this supervisor itself is interrupted.
        for process in processes:
            if process.is_alive():
                process.terminate()
        cleanup_deadline = time.monotonic() + 2
        for process in processes:
            process.join(timeout=max(0, cleanup_deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=.5)
        messages.close()
        messages.cancel_join_thread()


def numerical_error(reference, candidate):
    """Report error, never infer a pass threshold from the measured samples."""
    import numpy as np
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ParallelContractError("numerical comparison has an invalid shape/value")
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    delta = b - a
    denominator = float(np.linalg.norm(a.reshape(-1)))
    numerator = float(np.linalg.norm(delta.reshape(-1)))
    return dict(max_abs=float(np.max(np.abs(delta))), mean_abs=float(np.mean(np.abs(delta))),
                rmse=float(np.sqrt(np.mean(delta ** 2))),
                relative_l2=numerator / denominator if denominator else (0. if numerator == 0 else None),
                reference_l2=denominator, bitwise_equal=reference.dtype == candidate.dtype and
                reference.tobytes() == candidate.tobytes(), calibrated_tolerance=None, qualified=False)


def _static_records(root, manifests):
    records = {}
    for paths in manifests.values():
        for artifact in paths:
            path = root / artifact["path"]
            if file_digest(path) != artifact["sha256"]:
                raise ParallelContractError("rank record changed during collection")
            record = read_json(path)
            for value in (record["latents"], record["output"]):
                if value is not None and file_digest(root / value["path"]) != value["sha256"]:
                    raise ParallelContractError("rank array changed during collection")
            records[(record["case"], record["rank"])] = record
    if set(records) != {(case, rank) for case in CASES for rank in range(4)}:
        raise ParallelContractError("incomplete static SP rank/case evidence")
    return records


def _comparisons(root, manifests, *, off_manifests=None):
    import numpy as np
    records = _static_records(root, manifests)
    off_records = _static_records(root, off_manifests) if off_manifests is not None else None
    comparisons = []
    for (case, rank), record in sorted(records.items()):
        reference_rank = 0 if record["input_index"] == 0 else 2
        reference = records[("sp1", reference_rank)]
        a = np.load(root / reference["latents"]["path"], allow_pickle=False)
        b = np.load(root / record["latents"]["path"], allow_pickle=False)
        if a.shape != b.shape or len(b) != record["spec"]["steps"] + 1:
            raise ParallelContractError("incomplete or differently shaped static trajectory")
        if not np.array_equal(a[0], b[0]):
            raise ParallelContractError("static layouts did not start from identical latents")
        if reference["initial"] != record["initial"]:
            raise ParallelContractError("static layouts did not use identical conditioning")
        steps = [numerical_error(a[index], b[index]) for index in range(len(a))]
        output_error = None
        if record["output"] is not None:
            output_error = numerical_error(np.load(root / reference["output"]["path"], allow_pickle=False),
                                           np.load(root / record["output"]["path"], allow_pickle=False))
        off_reference = None
        if off_records is not None:
            off = off_records[("sp1", reference_rank)]
            off_latents = np.load(root / off["latents"]["path"], allow_pickle=False)
            if off["initial"] != record["initial"] or off_latents.shape != b.shape:
                raise ParallelContractError("cached and cache-off references differ in inputs or trajectory shape")
            if not np.array_equal(off_latents[0], b[0]):
                raise ParallelContractError("cached and cache-off references differ in initial latent")
            off_reference = dict(reference_recipe="off", reference_case="sp1", reference_rank=reference_rank,
                initial_and_per_step_latent_errors=[numerical_error(off_latents[index], b[index]) for index in range(len(b))],
                video_error=None if record["output"] is None else numerical_error(
                    np.load(root / off["output"]["path"], allow_pickle=False),
                    np.load(root / record["output"]["path"], allow_pickle=False)),
                quality_qualified=False, lpips_measured=False)
        comparisons.append(dict(case=case, rank=rank, input_index=record["input_index"],
                                reference_case="sp1", reference_rank=reference_rank,
                                path_observations=record["path_observations"],
                                cache_off_reference=off_reference,
                                initial_and_per_step_latent_errors=steps, video_error=output_error))
    return comparisons


def run_static_sp(job):
    """Entry called only in the research runner's already-charged child.

    A completed run returns measured_unqualified, never passed. It cannot
    qualify cached execution, model-state migration, or an online controller.
    The default order is off, t008, then cache-off elastic within this ledger.
    static_recipe=t008 runs only its own complete static unit after a measured
    off unit, allowing the same unit to be launched independently by a caller.
    """
    root = Path(job["output"]).resolve()
    recipe = _static_recipe(job)
    report_path = root / ("static-sp-validation.json" if recipe == "off" else "static-sp-t008-validation.json")
    if report_path.exists():
        raise ValueError("static SP evidence already exists; refusing to overwrite")
    started = time.monotonic()
    result = dict(schema_version="ditserve.research.static-sp.v1", status="failed",
                  static_recipe=recipe, cache_enabled=STATIC_RECIPES[recipe]["cache"],
                  calibrated_tolerances=None, elastic_qualified=False,
                  serving_cost_eligible=False, cached_sp_qualified=False,
                  measurement_scope="instrumented_fresh_requests_static_sp",
                  reference_scope="SP1 custom worker; official-loop correctness is a separate gate",
                  required_followup="GPU-validate request transport and numerical/quality tolerances; implement cached feature migration",
                  ledger_owner="outer research runner; no second allocation")
    try:
        plan = static_launch_plan(job)
        result.update(plan=plan, protocol_sha256=plan["protocol_sha256"], source_sha256=plan["source_sha256"])
        budget = _cached_unit_budget(plan) if recipe == "t008" else None
        if budget is not None:
            result["complete_unit_budget"] = budget
        if budget is not None and budget["status"] == "budget_not_fit":
            result.update(status="budget_not_fit", elastic=dict(status="not_requested_for_cached_recipe"))
        else:
            (root / plan["artifact_subdir"]).mkdir(exist_ok=False)
            manifests = _launch(plan)
            off_manifests = None if budget is None else read_json(root / budget["baseline_report"])["rank_records"]
            result.update(rank_records=manifests,
                          comparisons=_comparisons(root, manifests, off_manifests=off_manifests),
                          status="measured_unqualified", static_unit_complete=True)
            result["static_unit_wall_seconds"] = time.monotonic() - started
        if recipe == "off":
            # Keep a stable complete-unit reference: the outer report below will
            # later include cached/elastic children and their charged wall time.
            unit_path = root / "static-sp-off-unit.json"
            if unit_path.exists():
                raise ValueError("cache-off unit evidence already exists; refusing to overwrite")
            unit = dict(result, wall_seconds=result["static_unit_wall_seconds"])
            unit["sha256"] = digest(unit)
            atomic_json(unit_path, unit)
            result["static_unit_report"] = dict(path=unit_path.name, sha256=unit["sha256"])
            cached_job = dict(job, static_recipe="t008")
            if "spec" in cached_job:
                # An existing off caller may explicitly spell out its policy;
                # only that policy changes for the automatic cached unit.
                cached_job["spec"] = {key: value for key, value in cached_job["spec"].items()
                                      if key not in STATIC_RECIPES["off"]}
            result["cached_static"] = run_static_sp(cached_job)
            # Every child reload is part of this charged allocation. Prefer the
            # cached unit, then admit all three cache-off elastic controls only
            # when the remaining wall budget fits their measured estimate.
            estimate = max(60., result["static_unit_wall_seconds"] * 1.25)
            if result["cached_static"]["status"] == "failed":
                result["status"] = "failed"
                result["elastic"] = dict(status="blocked_cached_static_failed",
                                         elastic_qualified=False, cached_sp_qualified=False)
            elif time.monotonic() + estimate < plan["deadline_monotonic"]:
                result["elastic"] = run_elastic_sp(dict(job, parent_static_sha256=digest(manifests)))
            else:
                result["elastic"] = dict(status="budget_not_fit", next_complete_unit_estimate_seconds=estimate,
                                         elastic_qualified=False, cached_sp_qualified=False)
        else:
            result["elastic"] = dict(status="not_requested_for_cached_recipe",
                                     elastic_qualified=False, cached_sp_qualified=False)
    except Exception as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    result["wall_seconds"] = time.monotonic() - started
    result["sha256"] = digest(result)
    atomic_json(report_path, result)
    return result


ELASTIC_CASES = ("fixed_sp4", "fixed_sp2", "sp4_sp2_sp4")


def _elastic_ready(phase, payloads):
    if phase % 2 == 0:
        return
    case = ELASTIC_CASES[(phase - 1) // 2]
    active = (0, 1) if case == "fixed_sp2" else (0, 1, 2, 3)
    if any(payloads[rank] != payloads[0] for rank in active):
        raise ParallelContractError("elastic control started with different conditioning or latents")
    if any(payloads[rank] is not None for rank in range(4) if rank not in active):
        raise ParallelContractError("inactive elastic rank unexpectedly prepared a request")


def _elastic_rank_main(rank, plan, rendezvous, events, messages):
    try:
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        import numpy as np
        import torch
        from ..gpu.worker import WanStepWorker
        from .parallel import LayoutHandle, build_fixed_groups, CacheOffGroupMigration
        from .sp_backend import PinnedWanUlyssesBackend
        torch.set_num_threads(plan["cpu_threads_per_rank"])
        if torch.cuda.device_count() != 4:
            raise RuntimeError("elastic rank cannot see exactly the assigned four GPUs")
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group("nccl", init_method="file://" + rendezvous,
            rank=rank, world_size=4, timeout=datetime.timedelta(seconds=plan["timeout_seconds"]),
            device_id=torch.device("cuda", rank))
        groups = build_fixed_groups(timeout_seconds=plan["timeout_seconds"], backend="nccl")
        handle = LayoutHandle(rank)
        started = time.perf_counter()
        worker = WanStepWorker(device="cuda:%d" % rank, model_path=plan["model_path"], attention="flash")
        model_load_ms = (time.perf_counter() - started) * 1000
        backend = PinnedWanUlyssesBackend(handle, groups).attach(worker)
        control = CacheOffGroupMigration(worker, handle, groups)
        _ready(plan, rank, 0, events, messages)
        root, records = Path(plan["output"]), []
        for case_index, case in enumerate(ELASTIC_CASES):
            _deadline(plan)
            active = (0, 1) if case == "fixed_sp2" else (0, 1, 2, 3)
            placement = active if rank in active else (rank,)
            # Commit on every rank, including idle members, to keep the global
            # transition epoch aligned while retaining explicit local layouts.
            handle.commit(placement, expected_epoch=handle.current.epoch)
            spec = dict(plan["specs"][0], request_id="elastic-" + case)
            started = time.perf_counter()
            state = worker.prepare(spec) if rank in active else None
            initial = None if state is None else dict(latent=_tensor_hash(state.latents),
                prompt=_tensor_hash(state.prompt_embeds), negative=_tensor_hash(state.negative_prompt_embeds))
            _ready(plan, rank, 1 + 2 * case_index, events, messages, initial)
            latents = [] if state is None else [state.latents.detach().cpu().numpy().copy()]
            completed = [] if state is None else [0]
            costs, transitions = [], []
            prepare_ms = 0 if state is None else state.prepare_ms
            try:
                for index in range(spec["steps"]):
                    _deadline(plan)
                    if case == "sp4_sp2_sp4" and index in plan["elastic_boundaries"]:
                        target = (0, 1) if index == plan["elastic_boundaries"][0] else (0, 1, 2, 3)
                        state, transition = backend.migrate_cache_off(state, active, target,
                            request_id=spec["request_id"], step_index=index, expected_epoch=handle.current.epoch)
                        transitions.append(transition)
                        active = target
                    error = None
                    try:
                        if rank in active:
                            step_started = time.perf_counter()
                            event = backend.step(state)
                            costs.append(dict(event, ranks=list(active), layout_epoch=handle.current.epoch,
                                              gated_step_ms=(time.perf_counter() - step_started) * 1000))
                            latents.append(state.latents.detach().cpu().numpy().copy())
                            completed.append(state.step_index)
                    except Exception as exception:
                        error = repr(exception)
                    # Idle ranks keep only the controller loop counter. They
                    # receive current complete solver state when expanding.
                    control._reports(dict(error=error, phase="elastic_step_complete", completed_steps=index + 1))
                prefix = root / "elastic-sp" / (case + "-rank%d" % rank)
                latent_artifact = None if not latents else _array_artifact(root, prefix.with_suffix(".latents.npy"), np.stack(latents))
                video = _array_artifact(root, prefix.with_suffix(".video.npy"), worker.finish(state)) if rank == 0 else None
                record = dict(case=case, rank=rank, spec=spec, initial=initial, completed_steps=completed,
                    latents=latent_artifact, output=video, step_costs=costs, migrations=transitions,
                    prepare_ms=prepare_ms, model_load_ms=model_load_ms, model_load_is_once_per_rank_process=True,
                    decode_ms=sum(event["ms"] for event in state.events if event["kind"] == "decode") if state else 0,
                    wall_ms=(time.perf_counter() - started) * 1000, memory=worker._memory(),
                    inactive_members_hold_request_state=False)
                path = prefix.with_suffix(".json")
                atomic_json(path, record)
                records.append(dict(path=str(path.relative_to(root)), sha256=file_digest(path)))
            finally:
                if state is not None:
                    worker.release(state)
            _ready(plan, rank, 2 + 2 * case_index, events, messages)
        backend.close()
        torch.distributed.destroy_process_group()
        messages.put(dict(kind="complete", rank=rank, records=records))
    except BaseException as error:
        messages.put(dict(kind="error", rank=rank, error=repr(error), traceback=traceback.format_exc()))
        raise


def _elastic_comparisons(root, manifests):
    import numpy as np
    rows = {}
    for artifacts in manifests.values():
        for artifact in artifacts:
            if file_digest(root / artifact["path"]) != artifact["sha256"]:
                raise ParallelContractError("elastic rank record checksum mismatch")
            record = read_json(root / artifact["path"])
            rows[(record["case"], record["rank"])] = record
            for array in (record["latents"], record["output"]):
                if array is not None and file_digest(root / array["path"]) != array["sha256"]:
                    raise ParallelContractError("elastic array checksum mismatch")
    if set(rows) != {(case, rank) for case in ELASTIC_CASES for rank in range(4)}:
        raise ParallelContractError("incomplete elastic control trajectories")
    reference = rows[("fixed_sp4", 0)]
    a = np.load(root / reference["latents"]["path"], allow_pickle=False)
    comparisons = []
    for case in ELASTIC_CASES:
        record = rows[(case, 0)]
        b = np.load(root / record["latents"]["path"], allow_pickle=False)
        if (a.shape != b.shape or len(b) != record["spec"]["steps"] + 1
                or reference["initial"] != record["initial"]):
            raise ParallelContractError("elastic controls do not share a complete initial trajectory")
        comparisons.append(dict(case=case,
            per_step_latent_errors=[numerical_error(a[index], b[index]) for index in range(len(a))],
            video_error=numerical_error(np.load(root / reference["output"]["path"], allow_pickle=False),
                                        np.load(root / record["output"]["path"], allow_pickle=False)),
            sum_active_rank_step_ms=sum(e["gated_step_ms"] for rank in range(4) for e in rows[(case, rank)]["step_costs"]),
            migration_critical_path_ms=sum(max(rows[(case, rank)]["migrations"][index]["ms"] for rank in range(4))
                                           for index in range(len(record["migrations"]))),
            maximum_rank_case_wall_ms=max(rows[(case, rank)]["wall_ms"] for rank in range(4)),
            prepare_ms_by_rank=[rows[(case, rank)]["prepare_ms"] for rank in range(4)],
            decode_ms=record["decode_ms"], serving_cost_eligible=False))
    return dict(comparisons=comparisons,
                model_load_ms_by_rank=[rows[(ELASTIC_CASES[0], rank)]["model_load_ms"] for rank in range(4)])


def run_elastic_sp(job):
    """Called within run_static_sp's existing allocation; no standalone ledger."""
    if not job.get("parent_static_sha256"):
        raise ValueError("elastic execution requires the charged static-stage parent")
    if _static_recipe(job) != "off":
        raise ValueError("elastic execution only supports the cache-off recipe")
    root = Path(job["output"]).resolve()
    path = root / "elastic-sp-validation.json"
    if path.exists():
        raise ValueError("elastic evidence already exists; refusing to overwrite")
    started = time.monotonic()
    result = dict(schema_version="ditserve.research.elastic-sp.v1", status="failed", cache_enabled=False,
                  cached_sp_qualified=False, elastic_qualified=False, serving_cost_eligible=False,
                  calibrated_tolerances=None, parent_static_sha256=job["parent_static_sha256"],
                  boundaries_completed_steps=[16, 24], transport_scope="full cache-off Wan/UniPC request, SP4/group01",
                  reserved_devices=4, inactive_rank_models_resident=True,
                  limitation="CPU-staged instrumented transport; cached feature migration and calibrated GPU quality remain unqualified")
    try:
        plan = static_launch_plan(job)
        if plan["specs"][0]["steps"] != 32:
            raise ValueError("elastic pilot requires exactly 32 steps")
        plan.update(artifact_subdir="elastic-sp", elastic_boundaries=[16, 24])
        result.update(plan=plan, protocol_sha256=plan["protocol_sha256"], source_sha256=plan["source_sha256"])
        (root / "elastic-sp").mkdir(exist_ok=False)
        manifests = _launch(plan, target=_elastic_rank_main, ready_validator=_elastic_ready)
        result.update(rank_records=manifests, **_elastic_comparisons(root, manifests), status="measured_unqualified")
    except Exception as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
    result["wall_seconds"] = time.monotonic() - started
    result["reserved_gpu_seconds_observed"] = 4 * result["wall_seconds"]
    result["sha256"] = digest(result)
    atomic_json(path, result)
    return result
