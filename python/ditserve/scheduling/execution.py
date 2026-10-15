"""Real, budgeted single-device scheduling experiments.

The supervisor owns charging and isolation. Importing this module is CPU-only.
Scientific inputs are frozen before replay and never adjusted from outcomes.
"""

import copy
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..directions import direction1 as d1
from ..directions.block_executor import WanBlockExecutor
from ..research.common import atomic_json, digest, file_digest, finite, read_json
from .arrivals import EncodedArrivals
from .policy import ARMS, select_dispatch
from .runtime import SchedulingSana
from .workload import make_trace


BASELINES = ("edf_b1", "edf_b4")
CANDIDATES = ("guarded_b4", "guarded_adaptive")
SCENARIOS = ("original_burst", "tight_deadline", "original_ordinary", "video_contention")


def estimate(samples):
    if not samples:
        raise ValueError("missing calibration observations")
    samples = [finite(value, "calibration_ms", positive=True) for value in samples]
    return max(d1.quantile(samples, .95), statistics.mean(samples) * 1.1)


def spec(config, rid, prompt, seed):
    return dict(request_id=rid, prompt=prompt, negative_prompt="", seed=seed,
                steps=config["steps"], guidance_scale=5., cache=False, threshold=0.,
                max_continuous=1, **config["shape"])


def identity(config, worker):
    return dict(gpu_uuid=config["gpu_uuid"], name=worker.torch.cuda.get_device_name(worker.device),
                capability=list(worker.torch.cuda.get_device_capability(worker.device)),
                wan_attention=config["wan_attention"], sana_profile=config["sana_profile"],
                versions=worker.p.package_versions())


def admit(check, seconds, label):
    check()
    required = 1.3 * seconds + 45.
    if hasattr(check, "remaining_seconds") and check.remaining_seconds() < required:
        from .budget import BudgetStop
        raise BudgetStop("cannot admit complete %s: need %.1fs including margin/cleanup" % (label, required))


def remaining_view(row, state_pair, costs, now_ms):
    """Estimate from already executed progress, never from future arrivals."""
    result = dict(row, arrived=True, gpu_ready=True, started=state_pair is not None)
    if row["kind"] == "image":
        action, remaining, action_cost = "image", costs["image_ms"], costs["image_ms"]
    elif state_pair is None:
        action, action_cost = "prepare", costs["prepare_ms"]
        remaining = action_cost + sum(costs["block_position_ms"]) * costs["steps"] + costs["decode_ms"]
    else:
        state, continuation = state_pair
        if state.step_index >= costs["steps"]:
            action, remaining, action_cost = "decode", costs["decode_ms"], costs["decode_ms"]
        else:
            position = (0 if continuation.branch == "cond" else costs["num_blocks"]) + continuation.next_block
            position_costs = costs["block_position_ms"]
            suffix = position_costs[position:]
            remaining = (sum(suffix) + sum(position_costs) * (costs["steps"] - state.step_index - 1)
                         + costs["decode_ms"])
            result.update(action="denoise", remaining_ms=remaining,
                action_ms_by_quantum={str(q): sum(suffix[:q]) for q in (1, 4, 8)})
            result["action_ms_by_quantum"]["step"] = sum(suffix)
            return result
    result.update(action=action, remaining_ms=remaining,
                  action_ms_by_quantum={str(q): action_cost for q in (1, 4, 8, "step")})
    return result


def _release(worker, executor, pair):
    state, continuation = pair
    d1._release(worker, executor, state, continuation)


def _arrays_equal(actual, reference):
    """Bounded-memory comparison, retaining exactness and tolerance evidence."""
    import numpy as np
    if actual.shape != reference.shape:
        return dict(passed=False, exact=False, reason="shape_mismatch")
    a, b = actual.reshape(-1), reference.reshape(-1)
    passed = exact = True
    maximum = 0.
    for start in range(0, a.size, 1 << 20):
        left, right = a[start:start + (1 << 20)], b[start:start + (1 << 20)]
        valid = bool(np.isfinite(left).all() and np.isfinite(right).all())
        passed = passed and valid and bool(np.allclose(left, right, rtol=1e-5, atol=1e-6))
        exact = exact and valid and bool(np.array_equal(left, right))
        if valid and left.size:
            maximum = max(maximum, float(np.max(np.abs(left.astype(np.float64) - right))))
        elif not valid:
            return dict(passed=False, exact=False, max_abs=None, reason="nonfinite")
    return dict(passed=passed, exact=exact, max_abs=maximum, rtol=1e-5, atol=1e-6,
                shape=list(actual.shape))


def _calibrate(config, worker, executor, sana, result, save, check):
    import numpy as np
    image_times, native_service, prepare_times, decode_times = [], [], [], []
    positions = [[] for _ in range(60)]
    result["checks"] = []
    print("calibration: real Sana text encoding and complete generation", flush=True)
    # Cold initialization is charged but excluded from the frozen service costs.
    sana.generate(config["prompts"][0], seed=config["seed"], check_budget=check)
    print("calibration: co-resident complete Wan warmup", flush=True)
    worker.reference(spec(config, "calibration-warmup", config["prompts"][0], config["seed"]))
    check()
    # Both modalities execute complete pipelines and retain request-local RNG.
    for index, prompt in enumerate(config["prompts"]):
        print("calibration: prompt %d independent complete references" % index, flush=True)
        check()
        image_reference, image_cost = sana.generate(prompt, seed=config["seed"], check_budget=check)
        image_times.append(image_cost["ms"])
        request = spec(config, "calibration-%d" % index, prompt, config["seed"])
        official = worker.reference(request)
        native_service.append(official["elapsed_ms"])
        reference = current = continuation = None
        step_checks, image_checks, histories, exact_histories = [], [], [], []
        try:
            reference = worker.prepare(dict(request, request_id="native-%d" % index))
            current = worker.prepare(dict(request, request_id="interleaved-%d" % index))
            prepare_times.extend([r["ms"] for r in current.events if r["kind"] == "prepare"])
            # Only GPU preparation belongs in the ready-request cost estimate.
            prepare_times[-1] -= next(r.get("cpu_prompt_encode_ms", 0.) for r in current.events if r["kind"] == "prepare")
            continuation = executor.attach(current)
            injected = set()
            quantum_counts = Counter()
            while current.step_index < config["steps"]:
                check()
                worker.step(reference)
                target = reference.step_index
                while current.step_index < target:
                    step = current.step_index
                    position = (0 if continuation.branch == "cond" else 30) + continuation.next_block
                    # Prompt 0 calibrates every block position. Prompt 1 also
                    # exercises 4/8-block resumes after a half-branch pause.
                    quantum = (1 if index == 0 or continuation.next_block < 14
                               else (4 if (step + continuation.next_block) % 2 else 8))
                    action_started = time.perf_counter()
                    event = executor.advance(continuation, quantum, check_budget=check)
                    action_ms = (time.perf_counter() - action_started) * 1000
                    quantum_counts[str(quantum)] += 1
                    if index == 0 and step > 0:
                        positions[position].append(action_ms)
                    marker = (step, continuation.branch)
                    if step in (0, 16, 30) and continuation.next_block == 14 and marker not in injected:
                        actual, _ = sana.generate(prompt, seed=config["seed"], check_budget=check)
                        image_checks.append(_arrays_equal(actual, image_reference))
                        if not image_checks[-1]["passed"]:
                            result["failed_comparison"] = dict(prompt_index=index, step=step,
                                                                 kind="interfering_image", comparison=image_checks[-1])
                            save()
                            raise RuntimeError("interfering Sana generation changed its output")
                        injected.add(marker)
                    continuation.events.clear()  # Scalar samples above are sufficient; no unbounded event history.
                step_checks.append(d1.numerical_error(current.latents.detach().cpu().numpy(),
                                                       reference.latents.detach().cpu().numpy()))
                histories.append(d1.solver_matches(worker.torch, current.scheduler, reference.scheduler))
                exact_histories.append(d1.solver_matches(worker.torch, current.scheduler, reference.scheduler, exact=True))
                if not step_checks[-1]["passed"] or not histories[-1]:
                    result["failed_comparison"] = dict(prompt_index=index, step=current.step_index,
                        kind="trajectory", latent=step_checks[-1], solver=histories[-1])
                    save()
                    raise RuntimeError("resumed latent or sampler state differs at a solver step")
                if current.step_index % 8 == 0:
                    print("calibration: prompt %d resumed step %d/%d" % (index, current.step_index, config["steps"]), flush=True)
            latent_check = d1.numerical_error(current.latents.detach().cpu().numpy(), official["latents"].numpy())
            started = time.perf_counter()
            output = worker.finish(current)
            decode_times.append((time.perf_counter() - started) * 1000)
            output_check = _arrays_equal(output, official["output"])
            passed = (all(x["passed"] for x in step_checks + image_checks)
                      and all(histories) and latent_check["passed"] and output_check["passed"]
                      and len(injected) == 6
                      and (index == 0 or all(quantum_counts[str(q)] > 0 for q in (1, 4, 8))))
            record = dict(prompt_index=index, seed=config["seed"], passed=passed, step_checks=step_checks,
                          solver_checks=histories, solver_exact_checks=exact_histories,
                          image_checks=image_checks, injected=len(injected),
                          quantum_counts=dict(quantum_counts),
                          latent=latent_check, output=output_check,
                          output_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(output))).hexdigest(),
                          reference_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(official["output"]))).hexdigest(),
                          memory=worker._memory())
            result["checks"].append(record)
            save()
            if not passed:
                raise RuntimeError("same-device complete-trajectory numerical gate failed")
        finally:
            if current is not None:
                d1._release(worker, executor, current, continuation)
            if reference is not None:
                worker.release(reference)
            del official
    costs = dict(schema_version="ditserve.scheduling.costs.v1", steps=config["steps"], num_blocks=30,
                 block_position_ms=[estimate(values) for values in positions],
                 prepare_ms=estimate(prepare_times), decode_ms=estimate(decode_times),
                 image_ms=estimate(image_times), video_service_ms=statistics.mean(native_service),
                 image_service_ms=statistics.mean(image_times), image_deadline_ms=statistics.mean(image_times) * 2.5,
                 estimation_rule="max(p95,1.1*mean); empirical prediction, not an upper bound",
                 observations=dict(image_ms=image_times, video_service_ms=native_service,
                                   prepare_ms=prepare_times, decode_ms=decode_times,
                                   block_position_counts=[len(v) for v in positions]))
    result.update(status="passed", eligible=True, cost_model=costs, cost_model_sha256=digest(costs),
                  correctness_scope="current device/profile, two prompts, seed301, full trajectories and Sana interference")
    save()


def _trace_duration(requests):
    # Arrival horizon plus measured full service is a conservative planning basis.
    return (max(r["arrival_ms"] for r in requests) + sum(r["service_ms"] for r in requests)) / 1000


def run_trace(config, worker, executor, sana, requests, arm, costs, directory, check, partial,
              *, dispatch=None, view_factory=None, artifact_writer=None, instrument_control=False):
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    dispatch = select_dispatch if dispatch is None else dispatch
    view_factory = remaining_view if view_factory is None else view_factory
    rows = copy.deepcopy(requests)
    states = {}
    for row in rows:
        row.update(arrived=False, gpu_ready=False, gpu_ready_ms=None, ready_since_ms=None,
                   pause_total_ms=0., pause_max_ms=0., gpu_work_ms=0.,
                   action_counts={}, stage_ms={}, dispatch_reasons={}, blocked_by_ms={}, outputs={})
    jobs = [(r["request_id"], r["arrival_ms"], spec(config, r["request_id"], r["prompt"], r["seed"]))
            for r in rows if r["kind"] == "video"]
    origin, last_saved = time.perf_counter(), 0.
    now = lambda: (time.perf_counter() - origin) * 1000
    previous_rid, switches, scheduling_ms, persistence_ms = None, 0, 0., 0.
    decisions, prediction_errors = Counter(), []
    event_path = directory / "execution-events.jsonl"
    event_file = event_path.open("w", buffering=1 << 20)
    try:
        arrivals = EncodedArrivals(jobs, worker.encode_prompt, origin)
    except BaseException:
        event_file.close()
        raise
    event_count, checkpoint_ms = 0, 0.

    def checkpoint(elapsed):
        nonlocal checkpoint_ms
        started = time.perf_counter()
        event_file.flush()
        partial(rows, elapsed)
        checkpoint_ms += (time.perf_counter() - started) * 1000

    def refresh(instant):
        encoding = arrivals.snapshot()
        for row in rows:
            rid = row["request_id"]
            row["arrived"] = row["arrival_ms"] <= instant
            ticket = encoding.get(rid)
            if ticket:
                row.update(encode_submit_ms=ticket["submit_ms"], encode_start_ms=ticket["start_ms"],
                           encode_end_ms=ticket["end_ms"])
                if ticket["start_ms"] is not None:
                    row["encoding_queue_ms"] = ticket["start_ms"] - row["arrival_ms"]
                if ticket["end_ms"] is not None:
                    row["encoding_work_ms"] = ticket["end_ms"] - ticket["start_ms"]
            ready_time = row["arrival_ms"] if row["kind"] == "image" else ticket.get("end_ms") if ticket else None
            is_ready = row["arrived"] and ready_time is not None and ready_time <= instant
            if is_ready and (row["kind"] == "image" or rid in states or ticket["future"] is not None and ticket["future"].done()):
                row["gpu_ready"] = True
                if row["gpu_ready_ms"] is None:
                    row["gpu_ready_ms"] = ready_time
                    row["ready_since_ms"] = ready_time
        return encoding

    try:
        while any(row["status"] == "pending" for row in rows):
            check()
            instant = now()
            encoding = refresh(instant)
            ready = [view_factory(row, states.get(row["request_id"]), costs, instant)
                     for row in rows if row["status"] == "pending" and row["gpu_ready"]]
            started = time.perf_counter()
            decision = dispatch(ready, instant, arm, costs)
            scheduling_ms += (time.perf_counter() - started) * 1000
            rid = decision["request_id"]
            if rid is None:
                time.sleep(.001)
                continue
            row = next(r for r in rows if r["request_id"] == rid)
            view = next(r for r in ready if r["request_id"] == rid)
            decisions[decision["reason"]] += 1
            row["dispatch_reasons"][decision["reason"]] = row["dispatch_reasons"].get(decision["reason"], 0) + 1
            row["predicted_deadline_conflicts"] = row.get("predicted_deadline_conflicts", 0) + int(decision["predicted_deadline_conflict"])
            start = now()
            if row["start_ms"] is None:
                row["start_ms"] = start
                row["initial_gpu_queue_ms"] = start - row["gpu_ready_ms"]
                row["initial_predicted_remaining_ms"] = view["remaining_ms"]
            else:
                pause = start - row["ready_since_ms"]
                row["pause_total_ms"] += pause
                row["pause_max_ms"] = max(row["pause_max_ms"], pause)
            if previous_rid is not None and previous_rid != rid:
                switches += 1
            previous_rid = rid
            action = view["action"]
            output = latent = None
            if action == "image":
                output, service = sana.generate(row["prompt"], seed=row["seed"], check_budget=check)
                row["service"] = service
                row.update(encode_submit_ms=start, encode_start_ms=start,
                           encode_end_ms=start + service["encode_end_offset_ms"],
                           encoding_queue_ms=0., encoding_work_ms=service["encode_ms"], encoding_device="gpu")
                row["status"] = "completed"
            elif action == "prepare":
                payload = encoding[rid]["future"].result()
                state = worker.prepare_encoded(spec(config, rid, row["prompt"], row["seed"]), payload)
                states[rid] = (state, executor.attach(state))
            elif action == "denoise":
                state, continuation = states[rid]
                quantum = decision["quantum"]
                if quantum == "step":
                    executor.native_step(continuation, check_budget=check)
                else:
                    executor.advance(continuation, quantum, check_budget=check)
                continuation.events.clear()
            else:
                state, continuation = states[rid]
                latent = state.latents.detach().cpu().numpy()
                output = worker.finish(state)
                row["status"] = "completed"
            finish = now()
            row["ready_since_ms"] = finish
            duration = finish - start
            row["gpu_work_ms"] += duration
            row["stage_ms"][action] = row["stage_ms"].get(action, 0.) + duration
            row["action_counts"][action] = row["action_counts"].get(action, 0) + 1
            event_file.write(json.dumps(dict(request_id=rid, dispatch_ms=start, end_ms=finish, action=action,
                                              quantum=decision["quantum"], reason=decision["reason"]),
                                        separators=(",", ":")) + "\n")
            event_count += 1
            predicted = view["action_ms_by_quantum"][str(decision["quantum"])]
            if len(prediction_errors) < 128 or action != "denoise":
                prediction_errors.append(dict(action=action, predicted_ms=predicted, actual_ms=duration,
                                              error_ms=duration - predicted))
            refresh(finish)
            for other in rows:
                if other["request_id"] == rid or other["status"] != "pending" or not other["gpu_ready"]:
                    continue
                overlap = max(0., finish - max(start, other["gpu_ready_ms"]))
                key = row["kind"] + ":" + action
                other["blocked_by_ms"][key] = other["blocked_by_ms"].get(key, 0.) + overlap
            if row["status"] == "completed":
                row["finish_ms"] = finish
                row["remaining_prediction_error_ms"] = finish - row["start_ms"] - row["initial_predicted_remaining_ms"] - row["pause_total_ms"]
                write_start = time.perf_counter()
                if instrument_control:
                    row["output_persistence_start_ms"] = (write_start - origin) * 1000
                if artifact_writer is not None:
                    row["outputs"].update(artifact_writer.save(rid, output, latent))
                else:
                    for kind, array in (("output", output), ("latent", latent)):
                        if array is not None:
                            path = directory / (rid + "-" + kind + ".npy")
                            np.save(str(path), array, allow_pickle=False)
                            row["outputs"][kind] = str(path.resolve())
                write_end = time.perf_counter()
                persistence_ms += (write_end - write_start) * 1000
                if instrument_control:
                    row["output_persistence_end_ms"] = (write_end - origin) * 1000
                    row["output_persistence_work_ms"] = (write_end - write_start) * 1000
                    row["output_persistence_mode"] = "background_submission" if artifact_writer is not None else "synchronous"
                if rid in states:
                    release_start = now() if instrument_control else None
                    _release(worker, executor, states.pop(rid))
                    if instrument_control:
                        row["state_release_start_ms"] = release_start
                        row["state_release_end_ms"] = now()
                        row["state_release_work_ms"] = row["state_release_end_ms"] - release_start
            if row["status"] == "completed" or now() - last_saved >= 2000:
                checkpoint(now())
                last_saved = now()
        serving_complete_ms = now()
        artifact_stats = artifact_writer.drain() if artifact_writer is not None else None
        elapsed = now()
        event_file.flush()
        if event_count != sum(sum(row["action_counts"].values()) for row in rows):
            raise RuntimeError("execution event inventory differs from request actions")
        return dict(rows=rows, elapsed_ms=elapsed, metrics=d1.request_metrics(rows, elapsed),
                    image_metrics=d1.request_metrics([r for r in rows if r["kind"] == "image"], elapsed),
                    video_metrics=d1.request_metrics([r for r in rows if r["kind"] == "video"], elapsed),
                    policy_triggers=dict(decisions), switches=switches, scheduling_ms=scheduling_ms,
                    predicted_deadline_conflicts=sum(r.get("predicted_deadline_conflicts", 0) for r in rows),
                    checkpoint_ms=checkpoint_ms,
                    execution_events=dict(path=str(event_path.resolve()), count=event_count, sha256=file_digest(event_path)),
                    output_persistence_ms=persistence_ms, action_prediction_samples=prediction_errors,
                    quality=dict(passed=False, status="pending_same_trace_reference"),
                    **(dict(artifact_writer=artifact_stats, serving_complete_ms=serving_complete_ms,
                            output_persistence_mode="background",
                            output_drain_ms=artifact_stats["drain_wait_ms"])
                       if artifact_writer is not None else {}))
    except BaseException:
        instant = now()
        refresh(instant)
        for row in rows:
            if row["arrived"] and row["status"] == "pending":
                row["status"] = "interrupted"
        checkpoint(instant)
        raise
    finally:
        try:
            event_file.close()
        finally:
            try:
                arrivals.close()
            finally:
                for pair in list(states.values()):
                    _release(worker, executor, pair)


def verify_group(records):
    import numpy as np
    reference = next(r for r in records if r["arm"] == "edf_b1")
    baseline = {r["request_id"]: r for r in reference["rows"]}
    for record in records:
        checks = []
        for row in record["rows"]:
            results = {}
            for kind, path in row["outputs"].items():
                reference_path = baseline[row["request_id"]]["outputs"][kind]
                results[kind] = _arrays_equal(np.load(path, mmap_mode="r", allow_pickle=False),
                                             np.load(reference_path, mmap_mode="r", allow_pickle=False))
                results[kind]["sha256"] = file_digest(path)
                results[kind]["reference_sha256"] = file_digest(reference_path)
            row["quality"] = dict(passed=bool(results) and all(v["passed"] for v in results.values()), comparisons=results)
            checks.append(row["quality"]["passed"])
        record["quality"] = dict(passed=len(checks) == 24 and all(checks),
                                 reference_arm="edf_b1", checked=len(checks), scope="same device/input trace")
    return all(r["quality"]["passed"] for r in records)


def _online(config, root, phase, worker, executor, sana, result, save, check):
    from .reporting import evaluate_screen, evaluate_validate
    calibration = read_json(root / "calibrate/result.json")
    if calibration.get("status") != "passed" or not calibration.get("eligible"):
        raise ValueError("current-device calibration/correctness must pass")
    if calibration["device_identity"] != result["device_identity"]:
        raise ValueError("device profile changed since calibration")
    costs = calibration["cost_model"]
    supervised = read_json(root / "calibrate/supervisor.json")
    if (digest(costs) != calibration["cost_model_sha256"]
            or supervised.get("cost_model_sha256") != digest(costs)
            or supervised.get("cost_model") != costs):
        raise ValueError("calibration costs changed after the supervised measurement")
    result["cost_model_sha256"] = calibration["cost_model_sha256"]
    if phase == "pilot":
        scenarios, seeds = SCENARIOS[:2], [config["seed"]]
        arms = (*BASELINES, "guarded_b4")
        inherited = []
    elif phase == "screen":
        pilot = read_json(root / "pilot/result.json")
        if pilot["status"] != "passed":
            raise ValueError("complete pilot required")
        if pilot["records"] != read_json(root / "pilot/supervisor.json").get("records"):
            raise ValueError("pilot records changed after supervised completion")
        scenarios, seeds, arms = SCENARIOS, [config["seed"]], (*BASELINES, *CANDIDATES)
        inherited = copy.deepcopy(pilot["records"])
        result["inherited_from"] = dict(path=str(root / "pilot/result.json"), sha256=file_digest(root / "pilot/result.json"))
    else:
        screen = read_json(root / "screen/result.json")
        selected = screen.get("selection", {}).get("selected_arm")
        if screen["status"] != "passed" or not selected:
            raise ValueError("development selection must pass before holdout")
        if screen["selection"] != read_json(root / "screen/supervisor.json").get("selection"):
            raise ValueError("development selection changed after supervised completion")
        scenarios, seeds = SCENARIOS, config["validation_seeds"]
        arms, inherited = (*BASELINES, selected), []
        result["selected_arm"] = selected
        result["selection_source"] = dict(path=str(root / "screen/result.json"), sha256=file_digest(root / "screen/result.json"))
    result["records"] = inherited
    result["arms"] = list(arms)
    save()
    # Each fresh process pays its own warmup; it never changes frozen service times.
    sana.generate(config["prompts"][0], seed=config["seed"], check_budget=check)
    warm = worker.prepare(spec(config, "warmup", config["prompts"][0], config["seed"]))
    try:
        worker.step(warm)
    finally:
        worker.release(warm)
    for seed in seeds:
        for scenario_index, scenario in enumerate(scenarios):
            requests = make_trace(config, costs, scenario, seed)
            trace_hash = digest(requests)
            trace_path = root / "traces" / ("%s-%s.json" % (scenario, seed))
            if trace_path.exists() and read_json(trace_path)["sha256"] != trace_hash:
                raise ValueError("frozen trace changed")
            if not trace_path.exists():
                atomic_json(trace_path, dict(sha256=trace_hash, requests=requests,
                                            cost_model_sha256=result["cost_model_sha256"]))
            group = [r for r in result["records"] if r["scenario"] == scenario and r["seed"] == seed]
            needed = [a for a in arms if not any(r["arm"] == a for r in group)]
            if phase == "screen" and scenario in ("tight_deadline", "video_contention"):
                needed += ["feasible_b4", "edf_step"]
            if (scenario_index + (0 if phase != "validate" else seeds.index(seed))) % 2:
                needed.reverse()
            if not needed:
                continue
            measured = [r["elapsed_ms"] / 1000 for r in result["records"] if r["scenario"] == scenario]
            replay_seconds = max([_trace_duration(requests)] + measured)
            # Include synchronous output comparison/hashing before admitting the entire group.
            admit(check, replay_seconds * len(needed) + 30. * (len(group) + len(needed)),
                  "%s/%s paired group" % (scenario, seed))
            result["active_group"] = dict(scenario=scenario, seed=seed, arms=needed)
            save()
            for arm in needed:
                def partial(rows, elapsed):
                    result["active_trace"] = dict(scenario=scenario, seed=seed, arm=arm,
                                                   trace_sha256=trace_hash, rows=rows, elapsed_ms=elapsed)
                    save()
                trial = run_trace(config, worker, executor, sana, requests, arm, costs,
                    root / phase / "arrays" / ("%s-%s-%s" % (scenario, seed, arm)), check, partial)
                record = dict(scenario=scenario, seed=seed, arm=arm, trace_sha256=trace_hash,
                              device_identity=result["device_identity"], cost_model_sha256=result["cost_model_sha256"],
                              status="completed", **trial)
                result["records"].append(record)
                group.append(record)
                result.pop("active_trace", None)
                if any(r["arm"] == "edf_b1" for r in group) and not verify_group(group):
                    save()
                    raise RuntimeError("same-trace online output differs from independent scheduling baseline")
                save()
                print("completed %s seed=%s arm=%s elapsed=%.1fs" % (scenario, seed, arm, trial["elapsed_ms"] / 1000), flush=True)
            if not verify_group(group):
                save()
                raise RuntimeError("same-trace online output differs from independent scheduling baseline")
            result.pop("active_group", None)
            save()
    if phase == "pilot":
        result.update(status="passed", eligible=False, complete_pairs=True,
                      conclusion="single_seed_descriptive_only_not_performance_qualification")
    elif phase == "screen":
        selection = evaluate_screen(result["records"], expected_seeds=(config["seed"],))
        result.update(selection=selection, eligible=bool(selection.get("selected_arm")),
                      status="passed" if selection.get("selected_arm") else "stopped")
    else:
        validation = evaluate_validate(result["records"], result["selected_arm"], expected_seeds=tuple(config["validation_seeds"]))
        result.update(validation=validation, eligible=validation.get("status") == "passed",
                      status="passed" if validation.get("status") == "passed" else "stopped")
    save()


def run_phase(config, output, check_budget, phase):
    root = Path(output).resolve()
    destination = root / phase / "result.json"
    if destination.exists():
        raise FileExistsError("preserve phase evidence; use a fresh attempt")
    result = dict(schema_version="ditserve.scheduling.execution.v1", phase=phase,
                  status="running", eligible=False, records=[], started_at=time.time())
    save = lambda: atomic_json(destination, result)
    save()
    worker = executor = sana = None
    try:
        check_budget()
        if phase == "legacy":
            from .legacy import run_legacy
            run_legacy(config, root, check_budget, result, save)
            return result
        from ..gpu.worker import WanStepWorker
        worker = WanStepWorker(device="cuda:0", model_path=config["models"]["wan"]["path"], attention=config["wan_attention"])
        executor = WanBlockExecutor(worker)
        check_budget()
        sana = SchedulingSana(config["models"]["sana"]["path"], profile=config["sana_profile"])
        result["device_identity"] = identity(config, worker)
        save()
        if phase == "calibrate":
            _calibrate(config, worker, executor, sana, result, save, check_budget)
        elif phase in ("pilot", "screen", "validate"):
            _online(config, root, phase, worker, executor, sana, result, save, check_budget)
        else:
            raise ValueError("unknown scheduling phase")
    except BaseException as error:
        from .budget import BudgetStop
        result.update(status="incomplete" if isinstance(error, (BudgetStop, TimeoutError, KeyboardInterrupt)) else "failed",
                      eligible=False, reason=str(error), error_type=type(error).__name__)
        save()
        raise
    finally:
        if worker is not None:
            if executor is not None:
                for continuation in list(executor._states.values()):
                    if not continuation.closed:
                        executor.close(continuation)
            for state in list(worker._states.values()):
                worker.release(state)
        if sana is not None:
            sana.close()
        result["finished_at"] = time.time()
        save()
    return result
