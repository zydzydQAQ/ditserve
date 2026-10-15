"""Fresh paired V100 experiments, with immutable historical provenance."""

import copy
from pathlib import Path
import shutil
import time

from ..directions.block_executor import WanBlockExecutor
from ..research.common import atomic_json, digest, file_digest, read_json
from . import execution as original
from .next_policy import select_dispatch
from .runtime import SchedulingSana
from .workload import make_trace


SCENARIOS = original.SCENARIOS
NUMERICAL_FILES = ("gpu/worker.py", "gpu/phase_conflict.py", "directions/block_executor.py",
                   "directions/sana.py", "scheduling/runtime.py", "scheduling/workload.py",
                   "scheduling/policy.py")
FROZEN_FIELDS = ("models", "shape", "steps", "guidance_scale", "cache", "prompts", "seed",
                 "validation_seeds")


def source_inputs(config):
    """Authenticate the budget-stopped source without rewriting any evidence."""
    source = Path(config["source_run"]).resolve()
    evidence = config["source_evidence"]
    if evidence["source_run"] != str(source):
        raise ValueError("historical source location differs")
    for key in ("screen_result", "calibration", "source_supervisor", "calibration_supervisor"):
        path = Path(evidence[key + "_path"])
        path.resolve().relative_to(source)
        if file_digest(path) != evidence[key + "_sha256"]:
            raise ValueError("historical source changed: " + key)
    calibration = read_json(evidence["calibration_path"])
    supervised_calibration = read_json(evidence["calibration_supervisor_path"])
    screen = read_json(evidence["screen_result_path"])
    supervisor = read_json(evidence["source_supervisor_path"])
    if (calibration.get("status") != "passed" or not calibration.get("eligible") or
            supervised_calibration.get("status") != "passed" or
            supervised_calibration.get("returncode") != 0 or
            calibration.get("cost_model") != supervised_calibration.get("cost_model") or
            digest(calibration["cost_model"]) != calibration["cost_model_sha256"]):
        raise ValueError("historical same-device calibration did not qualify")
    if (not supervisor.get("budget_stopped") or supervisor.get("supervisor_failure") or
            supervisor.get("final_occupancy", {}).get("status") != "exclusive_observed" or
            screen.get("error_type") != "BudgetStop" or len(screen.get("records", [])) != 14):
        raise ValueError("source must be the isolated 14-replay budget stop")
    previous_config = read_json(source / "screen/runtime-config.json")
    if any(config.get(key) != previous_config.get(key) for key in FROZEN_FIELDS):
        raise ValueError("frozen workload/model configuration differs from historical source")
    current = Path(__file__).resolve().parents[1]
    archived = source / "screen/sources/python/ditserve"
    hashes = {}
    for name in NUMERICAL_FILES:
        if file_digest(current / name) != file_digest(archived / name):
            raise ValueError("qualified numerical implementation changed: " + name)
        hashes[name] = file_digest(current / name)
    from .reporting import complete_replay
    if not all(complete_replay(record) for record in screen["records"]):
        raise ValueError("historical source contains incomplete or unverified replay")
    return calibration, copy.deepcopy(screen["records"]), hashes


def remaining_view(row, pair, costs, now_ms):
    view = original.remaining_view(row, pair, costs, now_ms)
    if view["action"] == "denoise":
        _, continuation = pair
        position = (0 if continuation.branch == "cond" else costs["num_blocks"]) + continuation.next_block
        cost = sum(costs["block_position_ms"][position:position + 2])
    else:
        cost = view["action_ms_by_quantum"]["1"]
    view["action_ms_by_quantum"]["2"] = cost
    return view


def phase_matrix(config, phase, selection=None):
    if phase == "completion":
        return [("video_contention", config["seed"],
                 ["edf_b1", "edf_b4", "guarded_b4", "guarded_adaptive", "feasible_b4", "edf_step"])]
    if phase == "granularity":
        seeds, arms = [config["seed"]], ["edf_b1", "edf_b4", "edf_b2", "edf_b1_async"]
    elif phase == "replicate":
        seeds, arms = config["validation_seeds"], ["edf_b1", "edf_b4", "edf_step"]
        if selection not in (None, "edf_b2", "edf_b1_async"):
            raise ValueError("unknown frozen development candidate")
        if selection:
            arms = [*arms, selection]
    else:
        raise ValueError("unknown next scheduling phase")
    return [(scenario, seed, list(reversed(arms)) if (index + seed_index) % 2 else list(arms))
            for seed_index, seed in enumerate(seeds) for index, scenario in enumerate(SCENARIOS)]


def qualify_two_blocks(config, worker, executor, sana, result, save, check):
    """Two full prompt trajectories: native reference, solver state, Sana interference."""
    result["two_block_correctness"] = []
    for index, prompt in enumerate(config["prompts"]):
        check()
        print("two-block correctness: prompt %d continuous full reference" % index, flush=True)
        request = original.spec(config, "q2-reference-%d" % index, prompt, config["seed"])
        official = worker.reference(request)
        image_reference, _ = sana.generate(prompt, seed=config["seed"], check_budget=check)
        native = resumed = continuation = None
        checks, images, solver, solver_exact, injected = [], [], [], [], set()
        try:
            native = worker.prepare(dict(request, request_id="q2-native-%d" % index))
            resumed = worker.prepare(dict(request, request_id="q2-resume-%d" % index))
            continuation = executor.attach(resumed)
            while resumed.step_index < config["steps"]:
                check()
                worker.step(native)
                while resumed.step_index < native.step_index:
                    step = resumed.step_index
                    executor.advance(continuation, 2, check_budget=check)
                    marker = (step, continuation.branch)
                    if step in (0, 16, 30) and continuation.next_block == 14 and marker not in injected:
                        image, _ = sana.generate(prompt, seed=config["seed"], check_budget=check)
                        images.append(original._arrays_equal(image, image_reference))
                        injected.add(marker)
                        if not images[-1]["passed"]:
                            raise RuntimeError("two-block Sana interference output mismatch")
                    continuation.events.clear()
                checks.append(original._arrays_equal(resumed.latents.detach().cpu().numpy(), native.latents.detach().cpu().numpy()))
                solver.append(original.d1.solver_matches(worker.torch, resumed.scheduler, native.scheduler))
                solver_exact.append(original.d1.solver_matches(worker.torch, resumed.scheduler, native.scheduler, exact=True))
                if not checks[-1]["passed"] or not solver[-1]:
                    result["failed_comparison"] = dict(prompt_index=index, step=resumed.step_index,
                                                       latent=checks[-1], solver=solver[-1])
                    save()
                    raise RuntimeError("two-block latent or sampler state mismatch")
            latent = original._arrays_equal(resumed.latents.detach().cpu().numpy(), official["latents"].numpy())
            output = original._arrays_equal(worker.finish(resumed), official["output"])
            record = dict(prompt_index=index, seed=config["seed"], step_checks=checks, image_checks=images,
                          solver_checks=solver, solver_exact_checks=solver_exact,
                          latent=latent, output=output, injected=len(injected))
            record["passed"] = (latent["passed"] and output["passed"] and len(injected) == 6
                                and all(x["passed"] for x in checks + images) and all(solver))
            result["two_block_correctness"].append(record)
            save()
            if not record["passed"]:
                raise RuntimeError("two-block complete-trajectory correctness failed")
        finally:
            if resumed is not None:
                original.d1._release(worker, executor, resumed, continuation)
            if native is not None:
                worker.release(native)
            del official


def _online(config, root, phase, worker, executor, sana, result, save, check, calibration):
    from .next_reporting import evaluate_granularity
    costs = calibration["cost_model"]
    if result["device_identity"] != calibration["device_identity"]:
        raise ValueError("device or package identity changed since qualified calibration")
    selected = None
    if phase == "replicate":
        previous = read_json(root / "granularity/result.json")
        supervised = read_json(root / "granularity/supervisor.json")
        if previous.get("status") != "passed" or previous.get("selection") != supervised.get("selection"):
            raise ValueError("development selection was not frozen by successful supervision")
        selected = previous["selection"]["selected_candidate"]
        result.update(frozen_candidate=selected, selection_source=dict(
            path=str(root / "granularity/result.json"), sha256=file_digest(root / "granularity/result.json")))
    matrix = phase_matrix(config, phase, selected)
    result["planned_groups"] = [dict(scenario=s, seed=seed, arms=arms) for s, seed, arms in matrix]
    save()
    sana.generate(config["prompts"][0], seed=config["seed"], check_budget=check)
    warm = worker.prepare(original.spec(config, "warmup", config["prompts"][0], config["seed"]))
    try:
        worker.step(warm)
    finally:
        worker.release(warm)
    if phase == "granularity":
        qualify_two_blocks(config, worker, executor, sana, result, save, check)
    for scenario, seed, arms in matrix:
        check()
        requests = make_trace(config, costs, scenario, seed)
        trace_hash = digest(requests)
        trace_path = root / "traces" / ("%s-%s.json" % (scenario, seed))
        frozen = dict(sha256=trace_hash, requests=requests, cost_model_sha256=result["cost_model_sha256"])
        if trace_path.exists() and read_json(trace_path) != frozen:
            raise ValueError("frozen trace changed")
        if not trace_path.exists():
            atomic_json(trace_path, frozen)
        if hasattr(check, "require_seconds"):
            check.require_seconds(1.3 * (original._trace_duration(requests) + 30) * len(arms) + 45,
                                  "%s/%s complete paired group" % (scenario, seed))
        result["active_group"] = dict(scenario=scenario, seed=seed, arms=arms)
        save()
        group = []
        for arm in arms:
            def partial(rows, elapsed):
                result["active_trace"] = dict(scenario=scenario, seed=seed, arm=arm,
                    trace_sha256=trace_hash, rows=rows, elapsed_ms=elapsed)
                save()
            directory = root / phase / "arrays" / ("%s-%s-%s" % (scenario, seed, arm))
            writer = None
            try:
                if arm == "edf_b1_async":
                    from .next_persistence import AsyncArtifacts
                    writer = AsyncArtifacts(directory)
                trial = original.run_trace(config, worker, executor, sana, requests, arm, costs,
                    directory, check, partial, dispatch=select_dispatch, view_factory=remaining_view,
                    artifact_writer=writer, instrument_control=phase != "completion")
            finally:
                if writer is not None:
                    writer.close()
            record = dict(scenario=scenario, seed=seed, arm=arm, trace_sha256=trace_hash,
                device_identity=result["device_identity"], cost_model_sha256=result["cost_model_sha256"],
                status="completed", **trial)
            result["records"].append(record)
            group.append(record)
            result.pop("active_trace", None)
            if any(r["arm"] == "edf_b1" for r in group) and not original.verify_group(group):
                save()
                raise RuntimeError("same-input paired output or latent mismatch")
            save()
            print("completed %s seed=%s arm=%s elapsed=%.1fs" % (scenario, seed, arm, trial["elapsed_ms"] / 1000), flush=True)
        if not original.verify_group(group):
            save()
            raise RuntimeError("complete paired group output or latent mismatch")
        result.pop("active_group", None)
        save()
    if phase == "granularity":
        result["selection"] = evaluate_granularity(result["records"], expected_seeds=(config["seed"],))
    result.update(status="passed", eligible=True, technical_complete=True,
                  scientific_scope="technical_completion_does_not_imply_performance_gain")
    save()


def run_phase(config, output, check, phase):
    root = Path(output).resolve()
    destination = root / phase / "result.json"
    if destination.exists():
        raise FileExistsError("preserve existing attempt")
    result = dict(schema_version="ditserve.scheduling.next.execution.v1", phase=phase,
                  status="running", eligible=False, records=[], started_at=time.time())
    save = lambda: atomic_json(destination, result)
    save()
    worker = executor = sana = None
    try:
        check()
        calibration, inherited, hashes = source_inputs(config)
        result.update(source_evidence=config["source_evidence"], numerical_source_sha256=hashes,
                      cost_model_sha256=calibration["cost_model_sha256"])
        if phase == "completion":
            result["inherited_records"] = inherited
        if shutil.disk_usage(root).free < config.get("minimum_free_gib", 50) * (1 << 30):
            raise RuntimeError("insufficient free disk for a complete phase; retain prior evidence")
        save()
        from ..gpu.worker import WanStepWorker
        worker = WanStepWorker(device="cuda:0", model_path=config["models"]["wan"]["path"], attention=config["wan_attention"])
        executor = WanBlockExecutor(worker)
        check()
        sana = SchedulingSana(config["models"]["sana"]["path"], profile=config["sana_profile"])
        result["device_identity"] = original.identity(config, worker)
        save()
        _online(config, root, phase, worker, executor, sana, result, save, check, calibration)
    except BaseException as error:
        from .next_budget import WatchdogStop
        result.update(status="incomplete" if isinstance(error, (WatchdogStop, TimeoutError, KeyboardInterrupt)) else "failed",
                      eligible=False, reason=str(error), error_type=type(error).__name__)
        save()
        raise
    finally:
        try:
            if worker is not None:
                if executor is not None:
                    for continuation in list(executor._states.values()):
                        if not continuation.closed:
                            executor.close(continuation)
                for state in list(worker._states.values()):
                    worker.release(state)
        finally:
            if sana is not None:
                sana.close()
            result["finished_at"] = time.time()
            save()
    return result
