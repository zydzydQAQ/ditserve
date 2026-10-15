"""Real causal LLM -> image sessions; no synthetic model service is used."""

import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import statistics
import time

from ..research.common import atomic_json, digest, file_digest, read_json
from ..scheduling.execution import _arrays_equal, estimate
from ..scheduling.next_persistence import AsyncArtifacts
from . import workload, policy


def messages_for(session, round_index, previous_text=None):
    messages = [dict(role="user", content=session["turns"][0]["user_input"])]
    if round_index:
        if not isinstance(previous_text, str) or not previous_text.strip():
            raise ValueError("second turn requires the actually generated first assistant response")
        messages += [dict(role="assistant", content=previous_text),
                     dict(role="user", content=session["turns"][1]["user_input"])]
    return messages


def device_identity(config, llm):
    return dict(gpu_uuid=config["gpu_uuid"], name=llm.torch.cuda.get_device_name(llm.device),
        capability=list(llm.torch.cuda.get_device_capability(llm.device)), llm_dtype="float16",
        sana_dtype="float32", attention="sdpa_math", models=config["models"],
        versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "diffusers", "numpy")})


def memory(llm):
    torch = llm.torch
    free, total = torch.cuda.mem_get_info(llm.device)
    return dict(allocated_bytes=torch.cuda.memory_allocated(llm.device),
        reserved_bytes=torch.cuda.memory_reserved(llm.device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(llm.device), free_bytes=free, total_bytes=total)


def _cache_check(left, right):
    from .runtime import cache_comparison
    comparison = cache_comparison(left["cache"], right["cache"])
    # Keep exact scalar evidence; CPU snapshot tensors are temporary only.
    return dict(passed=comparison["passed"] and all(left[k] == right[k]
        for k in ("prompt_ids", "generated_ids", "cached_ids", "done")), exact=comparison["exact"],
        layers=comparison["layers"], max_abs=max((c["max_abs"] or 0 for c in comparison["tensors"]), default=0.))


def _sana_check(actual, expected):
    from .runtime import tensor_comparison
    scalar_keys = ("step_index", "sampler_step", "sampler_begin", "done")
    scalars = {key: actual[key] == expected[key] for key in scalar_keys}
    tensors = {key: tensor_comparison(actual[key], expected[key], rtol=1e-5, atol=1e-6)
               for key in expected if key not in scalar_keys}
    return dict(passed=all(scalars.values()) and all(c["passed"] for c in tensors.values()),
                exact=all(scalars.values()) and all(c["exact"] for c in tensors.values()),
                scalars=scalars, tensors=tensors)


def _qualification(config, root, llm, sana, result, save, check):
    from .runtime import tensor_comparison
    stages = {key: [] for key in workload.STAGES}
    token_ms, step_ms, lengths = [], [], []
    result.update(checks=[], calibration_observations=stages)
    # References and their snapshots qualify arithmetic; model action timing is
    # calibrated separately from these CPU copies and paired comparisons.
    warm_messages = [dict(role="user", content=config["session_inputs"][0][0])]
    warm = llm.prepare(warm_messages, max_new_tokens=4, request_id="warmup")
    try:
        while not warm.done:
            check()
            llm.advance(warm)
    finally:
        llm.release(warm)
    sana.reference("A red ceramic kettle on a wooden table.", seed=config["seed"])
    llm.torch.cuda.reset_peak_memory_stats(llm.device)
    for subject in config["qualification_session_indices"]:
        native_previous = resumed_previous = None
        previous_text = None
        try:
            for round_index in (0, 1):
                check()
                raw = config["session_inputs"][subject]
                session = dict(turns=[dict(user_input=text) for text in raw])
                messages = messages_for(session, round_index, previous_text)
                official = llm.reference(messages, max_new_tokens=config["max_new_tokens"])
                if not official["text"].strip():
                    raise RuntimeError("official LLM generated an empty image prompt")
                image_seed = config["seed"] * 100000 + subject * 10 + round_index * 2 + 1
                image_reference = sana.reference(official["text"], seed=image_seed)
                uninterrupted = sana.prepare(official["text"], seed=image_seed, request_id="sana-continuous")
                sampler_reference = []
                try:
                    while not uninterrupted.done:
                        check()
                        sana.advance(uninterrupted)
                        sampler_reference.append(sana.snapshot(uninterrupted))
                    continuous_check = _arrays_equal(sana.finish(uninterrupted), image_reference["output"])
                    if not continuous_check["passed"]:
                        result["failed_comparison"] = dict(subject=subject, round=round_index + 1,
                            sana_continuous=continuous_check)
                        save()
                        raise RuntimeError("continuous Sana adapter differs from official generation")
                finally:
                    sana.release(uninterrupted)
                native = llm.prepare(messages, previous=native_previous,
                    max_new_tokens=config["max_new_tokens"], request_id="native-%d-%d" % (subject, round_index))
                native_previous = native
                resumed = llm.prepare(messages, previous=resumed_previous,
                    max_new_tokens=config["max_new_tokens"], request_id="resumed-%d-%d" % (subject, round_index))
                resumed_previous = resumed
                stages["llm_prepare"].append(max(native.prepare_ms, .001))
                trajectory, native_snapshots, image_checks, decode_ms = [], [], [], []
                while not native.done:
                    check()
                    event = llm.advance(native)
                    if event["action"] == "prefill":
                        stages["llm_prefill"].append(event["ms"])
                    else:
                        token_ms.append(event["ms"])
                        decode_ms.append(event["ms"])
                    native_snapshots.append(llm.snapshot(native))
                # The reference trajectory above completes without model
                # interference. Only this second execution is paused for Sana.
                while not resumed.done:
                    check()
                    llm.advance(resumed)
                    token_index = len(resumed.generated_ids)
                    if token_index > len(native_snapshots):
                        raise RuntimeError("paused LLM generated beyond the continuous reference")
                    if token_index in (1, 8, 16):
                        interleaved = sana.reference(official["text"], seed=image_seed)
                        image_checks.append(_arrays_equal(interleaved["output"], image_reference["output"]))
                    comparison = _cache_check(llm.snapshot(resumed), native_snapshots[token_index - 1])
                    trajectory.append(comparison)
                    if not comparison["passed"] or any(not c["passed"] for c in image_checks):
                        result["failed_comparison"] = dict(subject=subject, round=round_index + 1,
                                                          token=token_index, cache=comparison)
                        save()
                        raise RuntimeError("LLM paused under real Sana interference changed tokens or KV")
                del native_snapshots
                stages["llm_decode"].append(max(sum(decode_ms), .001))
                lengths.append(len(native.generated_ids))
                tokens_exact = native.generated_ids == resumed.generated_ids == official["token_ids"]
                if not tokens_exact:
                    result["failed_comparison"] = dict(subject=subject, round=round_index + 1,
                        official_tokens=official["token_ids"], native_tokens=native.generated_ids,
                        resumed_tokens=resumed.generated_ids)
                    save()
                    raise RuntimeError("manual greedy tokens differ from official complete generation")
                previous_text = llm.finish(resumed)
                current = sana.prepare(previous_text, seed=image_seed, request_id="sana-%d-%d" % (subject, round_index))
                latent_checks, sampler_checks, denoise_ms = [], [], []
                try:
                    stages["image_encode"].append(current.prepare_ms)
                    while not current.done:
                        check()
                        event = sana.advance(current)
                        denoise_ms.append(event["ms"])
                        step_ms.append(event["ms"])
                        latent_checks.append(tensor_comparison(current.latents,
                            image_reference["step_latents"][current.step_index - 1], rtol=1e-5, atol=1e-6))
                        sampler_checks.append(_sana_check(sana.snapshot(current),
                            sampler_reference[current.step_index - 1]))
                        if not current.done:
                            other = llm.prepare(messages, max_new_tokens=16, request_id="sana-interference")
                            try:
                                while not other.done:
                                    check()
                                    llm.advance(other)
                                if other.generated_ids != official["token_ids"][:len(other.generated_ids)]:
                                    raise RuntimeError("interfering LLM generation changed its token prefix")
                            finally:
                                llm.release(other)
                    stages["image_denoise"].append(sum(denoise_ms))
                    output = sana.finish(current)
                    stages["image_decode"].append(current.decode_ms)
                    output_check = _arrays_equal(output, image_reference["output"])
                    passed = (tokens_exact and output_check["passed"] and all(x["passed"] for x in latent_checks)
                              and all(x["passed"] for x in sampler_checks))
                    result["checks"].append(dict(subject=subject, round=round_index + 1, passed=passed,
                        llm_tokens_exact=tokens_exact, generated_tokens=len(native.generated_ids),
                        reused_tokens=resumed.reused_tokens, cache_checks=trajectory, interference_images=image_checks,
                        sana_step_checks=latent_checks, sana_sampler_checks=sampler_checks,
                        sana_continuous=continuous_check, sana_output=output_check, memory=memory(llm)))
                    save()
                    if not passed:
                        raise RuntimeError("Sana step or output differs from official full pipeline under LLM interference")
                finally:
                    sana.release(current)
                print("qualified subject=%s round=%s tokens=%s" % (subject, round_index + 1, len(native.generated_ids)), flush=True)
        finally:
            for state in (native_previous, resumed_previous):
                if state is not None:
                    llm.release(state)
    if not token_ms:
        raise RuntimeError("qualification did not exercise any autoregressive decode token")
    costs = dict(schema_version="ditserve.sessions.costs.v1", stage_ms={key: estimate(values) for key, values in stages.items()},
        decode_token_ms=estimate(token_ms), image_step_ms=estimate(step_ms),
        expected_generated_tokens=max(lengths), observed_generated_tokens=lengths,
        estimation_rule="max(p95,1.1*mean); empirical frozen estimates, not deadline guarantees",
        reference_timing_excluded=True, numerical_scope="two real subjects and two dependent turns; current V100 profiles")
    result.update(status="passed", eligible=True, technical_complete=True,
                  cost_model=costs, cost_model_sha256=digest(costs), capacity=memory(llm))
    save()


def remaining_ms(row, llm_state, image_state, costs):
    stage, values = row["stage"], costs["stage_ms"]
    position = workload.STAGES.index(stage)
    remaining = sum(values[name] for name in workload.STAGES[position + 1:])
    if stage == "llm_decode":
        remaining += max(1, costs["expected_generated_tokens"] - len(llm_state.generated_ids)) * costs["decode_token_ms"]
    elif stage == "image_denoise":
        remaining += max(0, image_state.num_steps - image_state.step_index) * costs["image_step_ms"]
    else:
        remaining += values[stage]
    return remaining


def run_trace(config, llm, sana, trace, arm, costs, directory, check, partial):
    """Only actual predecessor completion releases a later user turn."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    writer = None
    rows, indexed, llm_states, image_states = [], {}, {}, {}
    sessions = {s["session_id"]: s for s in trace["sessions"]}
    next_round = {sid: 0 for sid in sessions}
    event_path = directory / "actions.jsonl"
    events = None
    origin = time.perf_counter()
    now = lambda: (time.perf_counter() - origin) * 1000
    active, last_partial, event_count = None, 0., 0
    selection_ms = checkpoint_ms = persistence_submit_ms = 0.

    def refresh(instant):
        for sid, session in sessions.items():
            turn = next_round[sid]
            if turn >= 2:
                continue
            previous = indexed.get(session["turns"][0]["request_id"])
            row = workload.release_turn(session, turn, trace, previous=previous)
            if row is None or row["release_ms"] > instant:
                continue
            row.update(start_ms=None, finish_ms=None, llm_start_ms=None, first_token_ms=None,
                token_intervals_ms=[], token_count=0, stage_ms={}, pause_max_ms=0., pause_total_ms=0.,
                last_action_end_ms=None, action_counts={}, outputs={}, quality={"passed": False})
            rows.append(row)
            indexed[row["request_id"]] = row
            next_round[sid] += 1

    def checkpoint():
        nonlocal checkpoint_ms, last_partial
        started = time.perf_counter()
        events.flush()
        partial(rows, now())
        checkpoint_ms += (time.perf_counter() - started) * 1000
        last_partial = now()

    try:
        writer = AsyncArtifacts(directory)
        events = event_path.open("w", buffering=1 << 20)
        llm.torch.cuda.reset_peak_memory_stats(llm.device)
        while sum(r["status"] == "completed" for r in rows) < trace["session_count"] * 2:
            check()
            instant = now()
            refresh(instant)
            ready = []
            for row in rows:
                if row["status"] == "completed":
                    continue
                sid = row["session_id"]
                row["remaining_total_ms"] = remaining_ms(row, llm_states.get(sid), image_states.get(sid), costs)
                row["stage_deadline_ms"] = row["stage_deadlines_ms"][row["stage"]]
                ready.append(row)
            started = time.perf_counter()
            rid = policy.select(ready, instant, arm, active_id=active)
            selection_ms += (time.perf_counter() - started) * 1000
            if rid is None:
                time.sleep(.001)
                continue
            row = indexed[rid]
            sid, stage = row["session_id"], row["stage"]
            active = rid
            start = now()
            if row["start_ms"] is None:
                row["start_ms"] = start
                row["initial_predicted_remaining_ms"] = row["remaining_total_ms"]
            elif row["last_action_end_ms"] is not None:
                pause = start - row["last_action_end_ms"]
                row["pause_total_ms"] += pause
                row["pause_max_ms"] = max(row["pause_max_ms"], pause)
            row["status"] = "running"
            output = None
            if stage == "llm_prepare":
                previous_row = indexed.get(sessions[sid]["turns"][0]["request_id"]) if row["round"] == 2 else None
                messages = messages_for(sessions[sid], row["round"] - 1,
                                        previous_row["llm_text"] if previous_row is not None else None)
                row["llm_start_ms"] = start
                llm_states[sid] = llm.prepare(messages, max_new_tokens=config["max_new_tokens"],
                    previous=llm_states.get(sid), request_id=rid)
                row["reused_tokens"] = llm_states[sid].reused_tokens
                row["stage"] = "llm_prefill"
            elif stage in ("llm_prefill", "llm_decode"):
                state = llm_states[sid]
                event = llm.advance(state)
                stamp = now()
                if row["first_token_ms"] is None:
                    row["first_token_ms"] = stamp
                else:
                    row["token_intervals_ms"].append(stamp - row["last_token_ms"])
                row["last_token_ms"] = stamp
                row["token_count"] = len(state.generated_ids)
                if state.done:
                    row["llm_token_ids"] = list(state.generated_ids)
                    row["llm_text"] = llm.finish(state)
                    row["llm_finish_ms"] = now()
                    row["image_ready_ms"] = row["llm_finish_ms"]
                    row["stage"] = "image_encode"
                else:
                    row["stage"] = "llm_decode"
            elif stage == "image_encode":
                row["image_start_ms"] = start
                image_states[sid] = sana.prepare(row["llm_text"], seed=row["image_seed"], request_id=rid)
                row["stage"] = "image_denoise"
            elif stage == "image_denoise":
                sana.advance(image_states[sid])
                if image_states[sid].done:
                    row["stage"] = "image_decode"
            elif stage == "image_decode":
                output = sana.finish(image_states[sid])
                completed_at = now()
                row.update(status="completed", stage="completed", finish_ms=completed_at, image_finish_ms=completed_at)
                active = None
            else:
                raise ValueError("unknown ready session stage")
            finish = now()
            row["last_action_end_ms"] = finish
            row["stage_ms"][stage] = row["stage_ms"].get(stage, 0.) + finish - start
            row["action_counts"][stage] = row["action_counts"].get(stage, 0) + 1
            events.write(json.dumps(dict(request_id=rid, action=stage, start_ms=start, end_ms=finish,
                remaining_prediction_ms=row["remaining_total_ms"], deadline_ms=row["deadline_ms"]), separators=(",", ":")) + "\n")
            event_count += 1
            if row["status"] == "completed":
                started = time.perf_counter()
                row["outputs"] = writer.save(rid, output, None)
                persistence_submit_ms += (time.perf_counter() - started) * 1000
                sana.release(image_states.pop(sid))
                row["remaining_prediction_error_ms"] = row["finish_ms"] - row["start_ms"] - row["pause_total_ms"] - row["initial_predicted_remaining_ms"]
                if row["round"] == 2:
                    llm.release(llm_states.pop(sid))
            if now() - last_partial >= 2000 or row["status"] == "completed":
                checkpoint()
        serving_complete_ms = now()
        persistence = writer.drain()
        elapsed = now()
        events.flush()
        if event_count != sum(sum(row["action_counts"].values()) for row in rows):
            raise RuntimeError("session action log inventory mismatch")
        return dict(status="completed", rows=rows, elapsed_ms=elapsed, serving_complete_ms=serving_complete_ms,
            planned_requests=trace["session_count"] * 2, planned_sessions=trace["session_count"],
            scheduling_ms=selection_ms, checkpoint_ms=checkpoint_ms, output_submission_ms=persistence_submit_ms,
            persistence=persistence, memory=memory(llm),
            events=dict(path=str(event_path.resolve()), count=event_count, sha256=file_digest(event_path)),
            quality=dict(passed=False, status="pending_same_external_trace_pair"))
    except BaseException:
        refresh(now())
        for row in rows:
            if row["status"] != "completed":
                row.update(status="failed", ready=False)
        if events is not None:
            checkpoint()
        else:
            partial(rows, now())
        raise
    finally:
        try:
            if events is not None:
                events.close()
        finally:
            try:
                if writer is not None:
                    writer.close()
            finally:
                for state in list(image_states.values()):
                    sana.release(state)
                for state in list(llm_states.values()):
                    llm.release(state)


def verify_group(records):
    import numpy as np
    reference = next(r for r in records if r["arm"] == "session_serial")
    expected = {row["request_id"]: row for row in reference["rows"]}
    valid = True
    for record in records:
        checks = []
        for row in record["rows"]:
            baseline = expected[row["request_id"]]
            paths = row["outputs"], baseline["outputs"]
            output = _arrays_equal(np.load(paths[0]["output"], mmap_mode="r", allow_pickle=False),
                                   np.load(paths[1]["output"], mmap_mode="r", allow_pickle=False))
            tokens = row["llm_token_ids"] == baseline["llm_token_ids"]
            same_prompt = row["llm_text"] == baseline["llm_text"]
            row["quality"] = dict(passed=tokens and same_prompt and output["passed"], tokens_exact=tokens,
                image_prompt_exact=same_prompt, image=output, output_sha256=file_digest(paths[0]["output"]),
                reference_sha256=file_digest(paths[1]["output"]))
            checks.append(row["quality"]["passed"])
        record["quality"] = dict(passed=len(checks) == record["planned_requests"] and all(checks),
            checked=len(checks), reference_arm="session_serial",
            scope="same external inputs and seeds; causally released second turns may arrive at different times")
        valid = valid and record["quality"]["passed"]
    return valid


def qualified_costs(config, root, result):
    qualified = read_json(root / "qualify/result.json")
    supervised = read_json(root / "qualify/supervisor.json")
    if (qualified.get("status") != "passed" or supervised.get("status") != "passed" or
            supervised.get("returncode") != 0 or supervised.get("supervisor_failure") or
            qualified["device_identity"] != result["device_identity"] or
            qualified.get("cost_model") != supervised.get("cost_model") or
            digest(qualified["cost_model"]) != qualified["cost_model_sha256"]):
        raise ValueError("same-device numerical qualification and frozen calibration required")
    result["cost_model_sha256"] = qualified["cost_model_sha256"]
    return qualified["cost_model"]


def _pilot(config, root, llm, sana, result, save, check):
    costs = qualified_costs(config, root, result)
    result["cost_model"] = costs
    for load_index, load in enumerate(config["loads"]):
        trace = workload.make_trace(config, costs, load, config["seed"])
        trace_path = root / "traces" / ("load-%s-seed-%s.json" % (load, config["seed"]))
        if trace_path.exists() and read_json(trace_path) != trace:
            raise ValueError("external session trace changed")
        if not trace_path.exists():
            atomic_json(trace_path, trace)
        arms = list(config["arms"])
        if load_index % 2:
            arms.reverse()
        result["active_group"] = dict(load=load, seed=config["seed"], arms=arms)
        save()
        group = []
        for arm in arms:
            partial_path = root / "pilot/partials" / ("%s-%s.json" % (load, arm))
            result["active_trace_path"] = str(partial_path)
            save()
            def partial(rows, elapsed):
                atomic_json(partial_path, dict(load=load, seed=config["seed"], arm=arm,
                    trace_sha256=trace["trace_sha256"], rows=rows, elapsed_ms=elapsed))
            directory = root / "pilot/arrays" / ("%s-%s" % (load, arm))
            trial = run_trace(config, llm, sana, trace, arm, costs, directory, check, partial)
            record = dict(load=load, seed=config["seed"], arm=arm, trace_sha256=trace["trace_sha256"],
                device_identity=result["device_identity"], cost_model_sha256=result["cost_model_sha256"], **trial)
            group.append(record)
            result["records"].append(record)
            result.pop("active_trace_path", None)
            if any(r["arm"] == "session_serial" for r in group) and not verify_group(group):
                save()
                raise RuntimeError("session scheduling changed LLM tokens or generated images")
            save()
            print("completed load=%s arm=%s rounds=%s elapsed=%.1fs" % (load, arm, len(trial["rows"]), trial["elapsed_ms"] / 1000), flush=True)
        if not verify_group(group):
            save()
            raise RuntimeError("complete session group failed output equivalence")
        result.pop("active_group", None)
        save()
    result.update(status="passed", eligible=True, technical_complete=True,
                  conclusion="single_seed_pilot_descriptive_only; no stable winner selected")
    save()


def run_phase(config, output, check, phase):
    root = Path(output).resolve()
    destination = root / phase / "result.json"
    if destination.exists():
        raise FileExistsError("preserve existing session attempt")
    result = dict(schema_version="ditserve.sessions.execution.v1", phase=phase,
                  status="running", eligible=False, records=[], started_at=time.time())
    save = lambda: atomic_json(destination, result)
    save()
    llm = sana = None
    try:
        check()
        if shutil.disk_usage(root).free < config["minimum_free_gib"] * (1 << 30):
            raise RuntimeError("insufficient space for complete paired session artifacts")
        from .runtime import MistralRuntime, SanaRuntime
        llm = MistralRuntime(config["models"]["llm"]["path"])
        check()
        sana = SanaRuntime(config["models"]["sana"]["path"])
        result.update(device_identity=device_identity(config, llm), loaded_memory=memory(llm))
        save()
        if phase == "qualify":
            _qualification(config, root, llm, sana, result, save, check)
        elif phase == "pilot":
            _pilot(config, root, llm, sana, result, save, check)
        elif phase == "state":
            from .state import run_state
            run_state(config, root, llm, sana, result, save, check, qualified_costs(config, root, result))
        else:
            raise ValueError("unknown session phase")
    except BaseException as error:
        from .budget import WatchdogStop
        result.update(status="incomplete" if isinstance(error, (WatchdogStop, TimeoutError, KeyboardInterrupt)) else "failed",
                      eligible=False, reason=str(error), error_type=type(error).__name__)
        save()
        raise
    finally:
        try:
            if sana is not None:
                sana.close()
        finally:
            if llm is not None:
                llm.close()
            result["finished_at"] = time.time()
            save()
    return result
