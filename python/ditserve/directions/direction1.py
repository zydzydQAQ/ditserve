"""Bounded, real single-A100 intra-step preemption pilot.

The caller owns GPU isolation, the cumulative budget and process supervision.
This module never starts work at import time. Numerical CPU tests do not count
as GPU evidence. Later phases consume completed, configuration-bound evidence.
"""

import hashlib
import itertools
import json
import math
import statistics
import threading
import time
from pathlib import Path

from ..util import atomic_json
from .block_executor import WanBlockExecutor
from .budget import BudgetStop
from .sana import SanaAdapter, MODEL_ID, RECIPE


PHASES = ("profile", "correctness", "preemption", "online")
POLICIES = ("step", "cfg", "block-1", "block-4", "block-8")
DEFAULT_PROMPTS = ("A red bicycle leaning against a blue wall in warm afternoon light.",
                   "A small sailboat glides across a calm lake in soft morning light.")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def settings(config):
    result = dict(config.get("direction1", {}))
    result.setdefault("shapes", [dict(height=480, width=832, frames=f) for f in (17, 49, 81)])
    result.setdefault("steps", 32)
    result.setdefault("prompts", list(DEFAULT_PROMPTS))
    result.setdefault("seeds", [101, 102])
    result.setdefault("policies", list(POLICIES))
    result.setdefault("profile_steps", min(4, result["steps"] - 1))
    result.setdefault("profile_repeats", 2)
    result.setdefault("arrival_fractions", [.1, .5, .9])
    result.setdefault("online_count", 24)
    result.setdefault("online_loads", [.5, .9])
    result.setdefault("online_seeds", [101, 102, 103])
    result.setdefault("patterns", result.get("online_patterns", ["ordinary", "burst"]))
    result.setdefault("max_split_overhead", .05)
    result.setdefault("minimum_wait_p95_reduction", .3)
    result.setdefault("maximum_video_attainment_drop", .05)
    result.setdefault("maximum_goodput_loss", .05)
    if any(p not in POLICIES for p in result["policies"]):
        raise ValueError("unsupported D1 policy")
    if "step" not in result["policies"] or "cfg" not in result["policies"]:
        raise ValueError("D1 requires step and CFG-boundary baselines")
    if type(result["steps"]) is not int or result["steps"] < 4:
        raise ValueError("D1 requires at least four actual Wan solver steps")
    if not 3 <= result["profile_steps"] < result["steps"]:
        raise ValueError("profile_steps must include warmup and measured steps")
    if not result["prompts"] or not result["seeds"]:
        raise ValueError("prompts and seeds must be nonempty")
    if result.get("cache", False) or result.get("guidance_scale", 5.) != 5.:
        raise ValueError("D1 freezes cache-off and CFG=5 for the Wan trajectory")
    return result


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def request_metrics(rows, elapsed_ms):
    """Failures/misses remain in the arrived denominator; no survivor-only SLO."""
    arrived = [r for r in rows if r.get("arrived", True)]
    complete = [r for r in arrived if r.get("status") == "completed"]
    timely = [r for r in complete if r["finish_ms"] <= r["deadline_ms"]]
    latencies = [r["finish_ms"] - r["arrival_ms"] for r in complete]
    waits = [r["start_ms"] - r["arrival_ms"] for r in complete]
    return dict(arrived=len(arrived), completed=len(complete), on_time=len(timely),
                failed_or_incomplete=len(arrived) - len(complete),
                on_time_fraction=len(timely) / len(arrived) if arrived else None,
                goodput_rps=len(timely) * 1000 / elapsed_ms if elapsed_ms > 0 else None,
                latency_mean_ms=statistics.mean(latencies) if latencies else None,
                latency_p95_ms=quantile(latencies, .95), wait_p95_ms=quantile(waits, .95),
                sample_count=len(complete), tail_claim="descriptive p95; no stable p99 claim")


def select_edf(rows, now_ms):
    """Both modalities share EDF; no hidden image priority or future arrivals."""
    ready = [r for r in rows if r["arrival_ms"] <= now_ms and r["status"] == "pending"]
    return min(ready, key=lambda r: (r["deadline_ms"], r["arrival_ms"], r["request_id"])) if ready else None


def frozen_requests(*, count, seed, load, pattern, video_ms, image_ms):
    """Two long videos and count-2 true images; service estimates fixed pre-policy."""
    import random
    if count < 4 or not 0 < load <= 2 or pattern not in ("ordinary", "burst"):
        raise ValueError("invalid D1 trace configuration")
    randomizer, rows, arrival = random.Random(seed), [], 0.
    mean_service = (2 * video_ms + (count - 2) * image_ms) / count
    for index in range(count):
        kind = "video" if index in (0, count // 2) else "image"
        if index:
            arrival += randomizer.expovariate(load / mean_service)
        if pattern == "burst":
            # Three image arrivals cluster; the generating random stream is unchanged.
            actual = math.floor(arrival / (3 * mean_service / load)) * (3 * mean_service / load)
        else:
            actual = arrival
        service = video_ms if kind == "video" else image_ms
        rows.append(dict(request_id="r%03d" % index, kind=kind, arrival_ms=actual,
                         deadline_ms=actual + service * 2.5, service_ms=service,
                         seed=seed + index, status="pending", start_ms=None, finish_ms=None))
    return rows


def numerical_error(actual, expected):
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return dict(passed=False, reason="shape_mismatch")
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return dict(passed=finite and bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-6)),
                exact=bool(np.array_equal(actual, expected)), finite=finite,
                max_abs=float(difference.max()) if finite and difference.size else None,
                rtol=1e-5, atol=1e-6, shape=list(actual.shape))


def solver_matches(torch, actual, expected, *, exact=False):
    """Check the evolving UniPC history, not just its public step counter."""
    def equal(left, right, strict):
        if torch.is_tensor(left) or torch.is_tensor(right):
            if (not torch.is_tensor(left) or not torch.is_tensor(right)
                    or left.shape != right.shape or left.dtype != right.dtype):
                return False
            if strict or not (left.is_floating_point() or left.is_complex()):
                return torch.equal(left, right)
            return bool(torch.isfinite(left).all() and torch.isfinite(right).all()
                        and torch.allclose(left, right, rtol=1e-5, atol=1e-6))
        if type(left) is not type(right):
            return False
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(equal(a, b, strict) for a, b in zip(left, right))
        return left == right
    names = ("model_outputs", "timestep_list", "last_sample", "_step_index", "_begin_index",
             "lower_order_nums", "this_order")
    return all(equal(getattr(actual, name, None), getattr(expected, name, None),
                     exact or name not in ("model_outputs", "last_sample")) for name in names)


def correctness_injections(steps):
    """Cover both CFG branches and early/middle/late trajectory safe points."""
    return {index * (steps - 1) // 3: policy
            for index, policy in enumerate(("block-1", "block-4", "block-8", "cfg"))}


class VideoArrivals:
    """Submit CPU encodes on a separate arrival clock, independent of GPU quanta."""

    def __init__(self, jobs, encode, origin):
        from concurrent.futures import ThreadPoolExecutor
        self._jobs, self._encode, self._origin = jobs, encode, origin
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._tickets, self._error = {}, None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d1-cpu-encode")
        self._thread = threading.Thread(target=self._arrive, name="d1-arrivals")
        self._thread.start()

    def _arrive(self):
        try:
            for rid, arrival_ms, spec in sorted(self._jobs, key=lambda job: (job[1], job[0])):
                remaining = self._origin + arrival_ms / 1000 - time.perf_counter()
                if self._stop.wait(max(0., remaining)):
                    return
                submitted = (time.perf_counter() - self._origin) * 1000
                future = self._pool.submit(self._encode, spec)
                with self._lock:
                    self._tickets[rid] = (spec, future, submitted)
        except BaseException as error:
            with self._lock:
                self._error = error

    def snapshot(self):
        with self._lock:
            if self._error is not None:
                raise self._error
            return dict(self._tickets)

    def close(self):
        self._stop.set()
        self._thread.join()
        self._pool.shutdown(wait=True, cancel_futures=True)


def _model_path(config, name):
    entry = config["models"][name]
    path = entry.get("path", entry.get("local_path")) if isinstance(entry, dict) else entry
    if not path:
        raise ValueError("missing local model path for " + name)
    return path


def _spec(options, shape, request_id, prompt=None, seed=None):
    return dict(request_id=request_id, prompt=prompt or options["prompts"][0], negative_prompt="",
                seed=options["seeds"][0] if seed is None else seed, steps=options["steps"],
                guidance_scale=5., cache=False, threshold=0., max_continuous=1, **shape)


class Evidence:
    def __init__(self, config, output, phase):
        self.path = Path(output) / ("direction1-" + phase + ".json")
        if self.path.exists():
            raise FileExistsError("D1 phase evidence is immutable; use a new attempt: " + str(self.path))
        self.data = dict(schema_version="ditserve.directions.d1.v1", phase=phase, status="running",
                         measured=False, eligible=False, config_sha256=_digest(config), rows=[], started_unix=time.time(),
                         scope="single-GPU cache-off exact execution; no SP qualification")
        self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def append(self, row):
        self.data["rows"].append(row)
        self.save()

    def finish(self, **values):
        status = "passed" if values.get("eligible") else "failed" if values.get("passed") is False else "stopped"
        values.setdefault("reason", values.get("decision", "phase evidence completed"))
        self.data.update(status=status, finished_unix=time.time(), **values)
        self.save()
        return self.data


def _previous(config, output, phase):
    path = Path(output) / ("direction1-" + phase + ".json")
    value = json.loads(path.read_text())
    if value["status"] != "passed" or value["config_sha256"] != _digest(config):
        raise ValueError("D1 prerequisite incomplete or configuration differs: " + phase)
    return value


def _release(worker, executor, state, continuation=None):
    if continuation is not None and not continuation.closed:
        executor.close(continuation)
    if state.residency != "released":
        worker.release(state)


def _advance_policy(executor, state, policy, check):
    if policy == "step":
        return executor.native_step(state, check_budget=check)
    return executor.advance(state, executor.quantum(state, policy), check_budget=check)


def _profile(worker, executor, sana, opts, evidence, check):
    import numpy as np
    prompt = opts["prompts"][0]
    encoded = sana.encode(prompt, check)
    sana.generate(prompt, seed=opts["seeds"][0], encoded=encoded, check_budget=check)
    image_times, image_full_times = [], []
    for index in range(opts["profile_repeats"]):
        _, row = sana.generate(prompt, seed=opts["seeds"][0] + index, encoded=encoded, check_budget=check)
        _, full = sana.generate(prompt, seed=opts["seeds"][0] + index, check_budget=check)
        image_times.append(row["ms"])
        image_full_times.append(full["ms"])
        evidence.append(dict(kind="image_profile", preencoded=row, end_to_end=full))
    successful = []
    for index, shape in enumerate(opts["shapes"]):
        step_times, quantum_times, memory, observations = [], [], [], []
        worker.torch.cuda.reset_peak_memory_stats(worker.device)
        try:
            for repeat in range(opts["profile_repeats"]):
                check()
                state = worker.prepare(_spec(opts, shape, "profile-%d-%d" % (index, repeat)))
                continuation = None
                try:
                    # Actual prefix of the 32-step trajectory; first step is untimed warmup.
                    for step in range(opts["profile_steps"]):
                        row = worker.step(state)
                        if step:
                            step_times.append(row["ms"])
                    continuation = executor.attach(state)
                    # Observe one later real step, after warmed native forwards.
                    while state.step_index == opts["profile_steps"]:
                        row = executor.advance(continuation, 1, observe=True, check_budget=check)
                        observations.extend(row["block_cuda_ms"])
                        quantum_times.append(row["ms"])
                        memory.append(row["extra_state_bytes"])
                    resident = worker._memory()
                finally:
                    _release(worker, executor, state, continuation)
            result = dict(kind="video_profile", shape=shape, status="completed", truncated=True,
                          completed_generation=False, sampler_steps=opts["steps"],
                          measured_step_range=[1, opts["profile_steps"] - 1],
                          observed_step_index=opts["profile_steps"], step_ms=step_times,
                          step_median_ms=float(np.median(step_times)), block_cuda=observations,
                          block_quantum_ms=quantum_times, extra_state_bytes=memory,
                          resident_memory=resident,
                          peak_allocated_bytes=int(worker.torch.cuda.max_memory_allocated(worker.device)))
            successful.append(result)
            evidence.append(result)
        except worker.torch.cuda.OutOfMemoryError as error:
            evidence.append(dict(kind="video_profile", shape=shape, status="oom", error=str(error)))
            # Increasing frames is a discovery scan, not a silent smaller-shape substitution.
            break
    if not successful:
        raise RuntimeError("no co-resident Sana/Wan shape passed profile")
    selected, full_service = None, None
    for candidate in reversed(successful):
        state = None
        check()
        try:
            state = worker.prepare(_spec(opts, candidate["shape"], "capacity-smoke"))
            origin = time.perf_counter()
            while state.step_index < state.spec["steps"]:
                check()
                worker.step(state)
            worker.finish(state)
            full_service = state.prepare_ms + (time.perf_counter() - origin) * 1000
            evidence.append(dict(kind="co_resident_full_generation", shape=candidate["shape"],
                                 status="completed", steps=state.spec["steps"], decode_completed=True,
                                 service_ms=full_service, memory=worker._memory()))
            selected = candidate
            break
        except worker.torch.cuda.OutOfMemoryError as error:
            evidence.append(dict(kind="co_resident_full_generation", shape=candidate["shape"],
                                 status="oom", error=str(error), fallback_explicit=True))
        finally:
            if state is not None:
                worker.release(state)
    if selected is None:
        return evidence.finish(eligible=False, co_resident=False, decision="stop_full_generation_capacity")
    return evidence.finish(selected_shape=selected["shape"], video_step_ms=selected["step_median_ms"],
                           video_end_to_end_ms=full_service,
                           image_preencoded_ms=statistics.median(image_times),
                           image_end_to_end_ms=statistics.median(image_full_times),
                           eligible=True, co_resident=True, profiling_only=True, single_request_quality_qualified=False,
                           memory_quantum_tradeoff="not assumed; packets retain only current boundary")


def _correctness(worker, executor, sana, opts, evidence, profile, output, check):
    import numpy as np
    shapes = [dict(height=256, width=256, frames=9), profile["selected_shape"]]
    passed, rows, all_overheads = True, [], []
    # Big-shape interleaving runs the whole trajectory and decode, not only a tiny gate.
    for shape_index, shape in enumerate(shapes):
        for sample, (prompt, seed) in enumerate(itertools.product(opts["prompts"], opts["seeds"])):
            check()
            spec = _spec(opts, shape, "reference", prompt, seed)
            official = worker.reference(spec)
            check()
            reference = worker.prepare(dict(spec, request_id="step-reference"))
            current = worker.prepare(dict(spec, request_id="interleaved"))
            uninterrupted = worker.prepare(dict(spec, request_id="split-no-preemption"))
            continuation = executor.attach(current)
            whole = executor.attach(uninterrupted)
            injections, covered = [], set()
            injection_plan = correctness_injections(opts["steps"])
            step_checks, solver_checks = [], []
            reference_ms, uninterrupted_ms = [], []
            try:
                while current.step_index < opts["steps"]:
                    check()
                    # Alternate order across samples; isolate ordinary splitting overhead.
                    operations = ("native", "split") if sample % 2 == 0 else ("split", "native")
                    for operation in operations:
                        started = time.perf_counter()
                        if operation == "native":
                            worker.step(reference)
                            reference_ms.append((time.perf_counter() - started) * 1000)
                        else:
                            executor.advance(whole, executor.quantum(whole, "step"), check_budget=check)
                            uninterrupted_ms.append((time.perf_counter() - started) * 1000)
                    target = reference.step_index
                    step_index = current.step_index
                    policy = injection_plan.get(step_index, "block-4")
                    while current.step_index < target:
                        executor.advance(continuation, executor.quantum(continuation, policy), check_budget=check)
                        boundary = "cfg" if policy == "cfg" and continuation.branch == "uncond" else continuation.branch
                        marker = (policy, boundary)
                        eligible_boundary = ((policy == "cfg" and boundary == "cfg")
                                             or continuation.packet is not None)
                        if step_index in injection_plan and marker not in covered and eligible_boundary:
                            # Actual other-model execution while Wan hidden state remains resident.
                            sana.generate(prompt, seed=seed, check_budget=check)
                            covered.add(marker)
                            injections.append(dict(step_index=step_index, policy=policy, boundary=boundary,
                                                   next_block=continuation.next_block))
                    step_checks.append(numerical_error(current.latents.detach().cpu().numpy(),
                                                       reference.latents.detach().cpu().numpy()))
                    histories = (current.scheduler, uninterrupted.scheduler)
                    solver_checks.append(dict(
                        passed=all(solver_matches(worker.torch, history, reference.scheduler) for history in histories),
                        exact=all(solver_matches(worker.torch, history, reference.scheduler, exact=True)
                                  for history in histories)))
                    step_checks.append(numerical_error(uninterrupted.latents.detach().cpu().numpy(),
                                                       reference.latents.detach().cpu().numpy()))
                latent = current.latents.detach().cpu().numpy()
                image = worker.finish(current)
                final = numerical_error(image, official["output"])
                final_latent = numerical_error(latent, official["latents"].numpy())
                expected_coverage = {(policy, branch) for policy in ("block-1", "block-4", "block-8")
                                     for branch in ("cond", "uncond")} | {("cfg", "cfg")}
                unit_pass = (covered == expected_coverage and all(r["passed"] for r in step_checks)
                             and all(r["passed"] for r in solver_checks) and final["passed"] and final_latent["passed"])
                # Exclude the first warmup step; retain both raw sequences.
                overhead = sum(uninterrupted_ms[1:]) / sum(reference_ms[1:]) - 1
                all_overheads.append(overhead)
                # Small numerical artifacts make the independently computed references auditable.
                artifact = Path(output) / ("d1-correctness-%d-%d.npz" % (shape_index, sample))
                with artifact.with_suffix(".tmp").open("wb") as handle:
                    np.savez_compressed(handle, latent=latent, reference_latent=official["latents"].numpy(),
                                        output=image, reference_output=official["output"])
                artifact.with_suffix(".tmp").replace(artifact)
                row = dict(shape=shape, prompt=prompt, seed=seed, steps=opts["steps"],
                           passed=unit_pass, step_checks=step_checks,
                           solver_history_checks=solver_checks,
                           solver_indices_match=all(r["passed"] for r in solver_checks),
                           output=final, latent=final_latent, interleaved_real_sana=bool(injections),
                           interleave_coverage=injections, mixed_quantum_trajectory=True,
                           native_step_ms=reference_ms, uncut_split_step_ms=uninterrupted_ms,
                           uncut_split_overhead=overhead,
                           artifact=str(artifact))
                evidence.append(row)
                rows.append(row)
                passed = passed and unit_pass
            finally:
                _release(worker, executor, current, continuation)
                _release(worker, executor, uninterrupted, whole)
                worker.release(reference)
            if not passed:
                return evidence.finish(passed=False, eligible=False, decision="stop_correctness")
    cost_pass = max(all_overheads) <= opts["max_split_overhead"]
    return evidence.finish(passed=passed, eligible=passed and cost_pass,
                           max_uncut_split_overhead=max(all_overheads),
                           decision="go_preemption" if passed and cost_pass else "stop_split_overhead")


def _preemption(worker, executor, sana, opts, evidence, profile, check):
    import numpy as np
    prompt, seed = opts["prompts"][0], opts["seeds"][0]
    encoded = sana.encode(prompt, check)
    image_reference, standalone = sana.generate(prompt, seed=seed, encoded=encoded, check_budget=check)
    # Reverse policy order pairs drift-sensitive measurements without selecting a favourable order.
    for round_index, policies in enumerate((opts["policies"], list(reversed(opts["policies"])))):
        for fraction in opts["arrival_fractions"]:
            for policy in policies:
                check()
                state = worker.prepare(_spec(opts, profile["selected_shape"], "preemption", prompt, seed))
                continuation = executor.attach(state)
                try:
                    executor.native_step(continuation, check_budget=check)
                    origin = time.perf_counter()
                    arrival_ms = fraction * profile["video_step_ms"]
                    image_start = image_end = None
                    max_extra, quanta = 0, []
                    while state.step_index < 4:
                        now = (time.perf_counter() - origin) * 1000
                        if image_start is None and now >= arrival_ms:
                            image_start = now
                            actual, service = sana.generate(prompt, seed=seed, encoded=encoded, check_budget=check)
                            image_end = (time.perf_counter() - origin) * 1000
                            parity = numerical_error(actual, image_reference)
                        row = _advance_policy(executor, continuation, policy, check)
                        max_extra = max(max_extra, row["extra_state_bytes"])
                        quanta.append(row["ms"])
                    if image_start is None:
                        raise RuntimeError("profiled arrival did not occur within the bounded video prefix")
                    elapsed = (time.perf_counter() - origin) * 1000
                    evidence.append(dict(policy=policy, round=round_index, fraction=fraction,
                        arrival_ms=arrival_ms, image_start_ms=image_start, image_finish_ms=image_end,
                        image_wait_ms=image_start - arrival_ms, image_latency_ms=image_end - arrival_ms,
                        image_service_ms=service["ms"], image_parity=parity,
                        prefix_video_ms=elapsed - (image_end - image_start), quantum_ms=quanta,
                        max_extra_state_bytes=max_extra, video_prefix_steps=3,
                        scope="preencoded image; bounded real video prefix, not video E2E goodput"))
                    if not parity["passed"]:
                        return evidence.finish(passed=False, eligible=False, decision="stop_image_parity")
                finally:
                    _release(worker, executor, state, continuation)
    rows = evidence.data["rows"]
    summaries = {p: dict(wait_p95_ms=quantile([r["image_wait_ms"] for r in rows if r["policy"] == p], .95),
                        prefix_video_ms=statistics.median([r["prefix_video_ms"] for r in rows if r["policy"] == p]))
                 for p in opts["policies"]}
    baseline = summaries["step"]
    candidates = []
    for policy, row in summaries.items():
        row["wait_reduction"] = 1 - row["wait_p95_ms"] / max(baseline["wait_p95_ms"], 1e-9)
        row["video_overhead"] = row["prefix_video_ms"] / baseline["prefix_video_ms"] - 1
        if (policy != "step" and row["wait_reduction"] >= opts["minimum_wait_p95_reduction"]
                and row["video_overhead"] <= opts["max_split_overhead"]):
            candidates.append(policy)
    return evidence.finish(passed=True, eligible=bool(candidates), summaries=summaries, standalone_image_ms=standalone["ms"],
                           decision="go_online" if candidates else "stop_no_benefit", candidates=candidates,
                           novelty_unproven=True, cfg_baseline_required=True)


def _run_trace(worker, executor, sana, opts, profile, requests, policy, check, save_partial):
    import copy
    rows, states = copy.deepcopy(requests), {}
    jobs = []
    for row in requests:
        if row["kind"] == "video":
            rid = row["request_id"]
            prompt = opts["prompts"][int(rid[1:]) % len(opts["prompts"])]
            jobs.append((rid, row["arrival_ms"], _spec(opts, profile["selected_shape"], rid, prompt, row["seed"])))
    origin = time.perf_counter()
    arrivals = VideoArrivals(jobs, worker.encode_prompt, origin)
    last_saved = origin
    def now():
        return (time.perf_counter() - origin) * 1000
    try:
        while any(row["status"] == "pending" for row in rows):
            check()
            instant = now()
            encoding = arrivals.snapshot()
            for row in rows:
                if row["request_id"] in encoding:
                    row["encode_submit_ms"] = encoding[row["request_id"]][2]
                row["arrived"] = row["arrival_ms"] <= instant
            ready_rows = [row for row in rows if row["kind"] == "image" or row["request_id"] in states
                          or (row["request_id"] in encoding and encoding[row["request_id"]][1].done())]
            request = select_edf(ready_rows, instant)
            if request is None:
                future = min(r["arrival_ms"] for r in rows if r["status"] == "pending")
                time.sleep(min(.01, max(.0005, (future - instant) / 1000)))
                continue
            rid = request["request_id"]
            request["arrived"] = True
            if request["start_ms"] is None:
                request["start_ms"] = now()
            prompt = opts["prompts"][int(rid[1:]) % len(opts["prompts"])]
            if request["kind"] == "image":
                _, service = sana.generate(prompt, seed=request["seed"], check_budget=check)
                request.update(status="completed", finish_ms=now(), service=service)
            elif rid not in states:
                spec, future, _ = encoding[rid]
                payload = future.result()
                request["cpu_prompt_encode_ms"] = payload.cpu_prompt_encode_ms
                state = worker.prepare_encoded(spec, payload)
                states[rid] = (state, executor.attach(state))
            else:
                state, continuation = states[rid]
                if state.step_index < opts["steps"]:
                    _advance_policy(executor, continuation, policy, check)
                else:
                    worker.finish(state)
                    request.update(status="completed", finish_ms=now())
                    _release(worker, executor, state, continuation)
                    del states[rid]
            if request["status"] == "completed" or time.perf_counter() - last_saved >= 1:
                save_partial(rows, now())
                last_saved = time.perf_counter()
        elapsed = now()
        return dict(rows=rows, elapsed_ms=elapsed, metrics=request_metrics(rows, elapsed),
                    image_metrics=request_metrics([r for r in rows if r["kind"] == "image"], elapsed),
                    video_metrics=request_metrics([r for r in rows if r["kind"] == "video"], elapsed))
    except BaseException:
        instant = now()
        for row in rows:
            row["arrived"] = row["arrival_ms"] <= instant
            if row["arrived"] and row["status"] == "pending":
                row["status"] = "interrupted"
        save_partial(rows, instant)
        raise
    finally:
        arrivals.close()
        for state, continuation in states.values():
            _release(worker, executor, state, continuation)


def _online(worker, executor, sana, opts, evidence, profile, preemption, check):
    if preemption.get("decision") != "go_online":
        return evidence.finish(decision="skipped_preemption_gate", eligible=False, measured_online=False)
    # Each phase is a fresh process. Warm both models before the first policy
    # so policy order cannot decide which arm pays first-use kernel setup.
    for index in range(opts.get("online_image_warmups", 2)):
        sana.generate(opts["prompts"][0], seed=opts["seeds"][0] + index, check_budget=check)
    # Actual full video reference includes CPU encoding, denoising and decode.
    check()
    reference = worker.reference(_spec(opts, profile["selected_shape"], "online-reference"))
    video_ms, image_ms = reference["elapsed_ms"], profile["image_end_to_end_ms"]
    del reference
    evidence.data["frozen_service_ms"] = dict(video=video_ms, image=image_ms)
    evidence.save()
    for seed in opts["online_seeds"]:
        for load in opts["online_loads"]:
            for pattern in opts["patterns"]:
                requests = frozen_requests(count=opts["online_count"], seed=seed, load=load,
                                           pattern=pattern, video_ms=video_ms, image_ms=image_ms)
                trace_id = _digest(requests)
                order = opts["policies"] if seed % 2 else list(reversed(opts["policies"]))
                for policy in order:
                    check()
                    def partial(rows, elapsed):
                        evidence.data["active_trace"] = dict(trace_sha256=trace_id, policy=policy,
                            rows=rows, elapsed_ms=elapsed, complete=False)
                        evidence.save()
                    result = _run_trace(worker, executor, sana, opts, profile, requests, policy, check, partial)
                    evidence.data.pop("active_trace", None)
                    evidence.append(dict(seed=seed, load=load, pattern=pattern, policy=policy,
                                         trace_sha256=trace_id, **result))
    summaries, best_simple = online_decision(evidence.data["rows"], opts)
    useful = any(r["meets_screen"] for r in summaries.values())
    return evidence.finish(decision="simple_policy_sufficient" if best_simple else
                           "potential_quantum_tradeoff" if useful else "stop_no_online_benefit",
                           eligible=useful and not best_simple, mechanism_benefit=useful,
                           summaries=summaries, best_simple_sufficient=best_simple,
                           scheduling="EDF shared by image and video; independent arrival thread and one CPU video encoder",
                           prepared_scope="end_to_end; CPU video encoding and GPU image encoding charged",
                           load_scope="arrival rate from isolated end-to-end service, not measured GPU utilization",
                           wait_scope="arrival to first GPU dispatch; video CPU preprocessing is included",
                           goodput_scope="all timely completions per complete trace wall time, including drain",
                           no_claims=["stable_p99", "SP", "adaptive_quantum_novelty"])


def online_decision(rows, opts):
    """Use only fully paired traces, with identical arrivals for every policy."""
    by_trace = {}
    for row in rows:
        by_trace.setdefault(row["trace_sha256"], {})[row["policy"]] = row
    complete = [group for group in by_trace.values() if set(group) == set(opts["policies"])]
    if not complete:
        raise ValueError("no complete paired D1 trace")
    summaries = {}
    for policy in opts["policies"]:
        members = [group[policy] for group in complete]
        requests = [request for member in members for request in member["rows"]]
        elapsed = sum(member["elapsed_ms"] for member in members)
        summaries[policy] = dict(overall=request_metrics(requests, elapsed),
            image=request_metrics([r for r in requests if r["kind"] == "image"], elapsed),
            video=request_metrics([r for r in requests if r["kind"] == "video"], elapsed),
            paired_traces=len(complete))
    baseline = summaries["step"]
    for policy, row in summaries.items():
        row["wait_reduction"] = 1 - row["image"]["wait_p95_ms"] / max(baseline["image"]["wait_p95_ms"], 1e-9)
        row["goodput_loss"] = 1 - row["overall"]["goodput_rps"] / max(baseline["overall"]["goodput_rps"], 1e-9)
        row["video_attainment_drop"] = baseline["video"]["on_time_fraction"] - row["video"]["on_time_fraction"]
        row["meets_screen"] = (policy != "step" and row["wait_reduction"] >= opts["minimum_wait_p95_reduction"]
                               and row["goodput_loss"] <= opts["maximum_goodput_loss"]
                               and row["video_attainment_drop"] <= opts["maximum_video_attainment_drop"])
    # A simple policy is sufficient if it passes overall and has no material
    # scenario-level reversal; otherwise a quantum tradeoff remains a hypothesis.
    sufficient = []
    for policy, summary in summaries.items():
        if not summary["meets_screen"]:
            continue
        stable = True
        for group in complete:
            actual, reference = group[policy], group["step"]
            image_wait = actual["image_metrics"]["wait_p95_ms"]
            reference_wait = reference["image_metrics"]["wait_p95_ms"]
            if (image_wait > max(1., reference_wait * 1.05)
                    or actual["metrics"]["goodput_rps"] < reference["metrics"]["goodput_rps"] * (1 - opts["maximum_goodput_loss"])
                    or actual["video_metrics"]["on_time_fraction"] < reference["video_metrics"]["on_time_fraction"] - opts["maximum_video_attainment_drop"]):
                stable = False
        if stable:
            sufficient.append(policy)
    return summaries, bool(sufficient)


def run_phase(config, output, check_budget, phase):
    """Run one real phase. Budget/isolation callbacks may interrupt at safe points."""
    if phase not in PHASES:
        raise ValueError("unknown direction1 phase")
    opts = settings(config)
    output = Path(output)
    profile = _previous(config, output, "profile") if phase != "profile" else None
    if phase in ("preemption", "online"):
        correctness = _previous(config, output, "correctness")
        if not correctness.get("eligible"):
            raise ValueError("D1 correctness gate has not passed")
    preemption = _previous(config, output, "preemption") if phase == "online" else None
    evidence = Evidence(config, output, phase)
    worker = executor = sana = None
    try:
        check_budget()
        from ..gpu.worker import WanStepWorker
        worker = WanStepWorker(device=config.get("device", "cuda:0"), model_path=_model_path(config, "wan"),
                               attention=opts.get("attention", config.get("attention", "flash")))
        executor = WanBlockExecutor(worker)
        check_budget()
        sana = SanaAdapter(_model_path(config, "sana"), device=worker.device)
        evidence.data["execution"] = dict(device=worker.device, wan_attention=worker.attention,
                                           sana_model=MODEL_ID, sana_recipe=RECIPE,
                                           dependencies=worker.p.package_versions(), co_resident=True,
                                           residency={"wan_transformer": worker.device, "wan_vae": worker.device,
                                                      "wan_text_encoder": "cpu_float32_existing_flash_baseline",
                                                      "sana_all_components": worker.device})
        evidence.data["measured"] = True
        evidence.save()
        if phase == "profile":
            return _profile(worker, executor, sana, opts, evidence, check_budget)
        if phase == "correctness":
            return _correctness(worker, executor, sana, opts, evidence, profile, output, check_budget)
        if phase == "preemption":
            return _preemption(worker, executor, sana, opts, evidence, profile, check_budget)
        return _online(worker, executor, sana, opts, evidence, profile, preemption, check_budget)
    except BaseException as error:
        evidence.data.update(status="incomplete" if isinstance(error, (KeyboardInterrupt, TimeoutError, BudgetStop)) else "failed",
                             eligible=False,
                             error=type(error).__name__ + ": " + str(error), finished_unix=time.time())
        evidence.save()
        raise
    finally:
        if worker is not None:
            if executor is not None:
                for continuation in list(executor._states.values()):
                    executor.close(continuation)
            for state in list(worker._states.values()):
                worker.release(state)
        if sana is not None:
            sana.close()
