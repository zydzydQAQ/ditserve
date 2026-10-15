"""Frozen-trace replay with independent arrivals and a bounded artifact writer.

Only ready requests are passed to the scheduling decision. GPU stages are
non-preemptible; decisions happen after prepare, each solver step, and decode.
The CPU fixture uses an analytic clock and never claims GPU measurements.
"""

import copy
import math
import queue
import statistics
import threading
import time
from pathlib import Path

from .backends import (WAN_MODEL, WAN_REVISION, RealClock, VirtualClock, canonical_backend, core_policy,
                       make_backend, run_core)
from .common import atomic_json, canonical_hash, environment_info, file_hash, seal, source_identity, verify_seal


POLICIES = ("fcfs", "edf", "srtf")


def _error_reason(error):
    """KeyboardInterrupt and several resource errors have empty string values."""
    message = str(error).strip()
    return type(error).__name__ + (": " + message if message else "")


def _check(budget, seconds=0):
    if budget is not None:
        budget.check(seconds)


def _fresh(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _generation(config, sample, shape_id, seed=0, request_id="profile"):
    shape = config["shapes"][shape_id]
    model = config.get("model", {})
    cache = config.get("cache", {})
    return dict(request_id=request_id, sample_id=sample["sample_id"],
                dataset=sample.get("dataset", "profile-development"), prompt=sample["prompt"],
                shape_id=shape_id, **{k: shape[k] for k in ("height", "width", "frames", "steps")},
                seed=seed, model=model.get("id"), revision=model.get("revision"),
                cache_enabled=cache.get("enabled", False), cache_threshold=cache.get("threshold", .08),
                max_continuous=cache.get("max_continuous", 3), guidance_scale=config.get("guidance_scale", 5.),
                negative_prompt=config.get("negative_prompt", ""))


def _wan_identity(request):
    if request.get("model") != WAN_MODEL or request.get("revision") != WAN_REVISION:
        raise ValueError("Wan execution requires the pinned Wan2.1 1.3B model and revision")


def _verify_execution_identity(expected, actual):
    """A service profile is specific to its physical GPU and execution recipe."""
    if not isinstance(expected, dict) or not isinstance(expected.get("gpu"), dict) or not expected["gpu"].get("uuid"):
        raise ValueError("Wan profile lacks a physical GPU identity; rebuild the service profile")
    differences = []
    for key in ("uuid", "name", "total_memory_bytes", "compute_capability"):
        if expected["gpu"].get(key) != actual.get("gpu", {}).get(key):
            differences.append("gpu." + key)
    for key in ("precision", "attention", "text_encoder_device", "scheduler", "model_id", "revision",
                "torch_version", "cuda_runtime_version", "driver_version"):
        if expected.get(key) != actual.get(key):
            differences.append(key)
    if differences:
        raise ValueError("profile/execution identity mismatch: " + ", ".join(differences))


class ArrivalSource:
    """Real arrivals have a dedicated producer; virtual arrivals occur at exact times.

    submit_ms is the actual real producer observation, not a copied schedule.
    The virtual case explicitly uses arrival_ms as its analytic submit instant.
    """

    def __init__(self, requests, clock):
        self.requests = sorted(requests, key=lambda r: (r["arrival_ms"], r["request_id"]))
        self.clock = clock
        self.submitted = {}
        self.cursor = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def start(self):
        if not self.clock.virtual:
            self.thread = threading.Thread(target=self._produce, name="benchmark-arrivals", daemon=True)
            self.thread.start()

    def _produce(self):
        for request in self.requests:
            while not self.stop.is_set():
                delay = (request["arrival_ms"] - self.clock.now()) / 1000
                if delay <= 0:
                    break
                self.stop.wait(min(delay, .02))
            if self.stop.is_set():
                return
            with self.lock:
                self.submitted[request["request_id"]] = self.clock.now()

    def collect(self):
        if self.clock.virtual:
            while self.cursor < len(self.requests) and self.requests[self.cursor]["arrival_ms"] <= self.clock.now():
                r = self.requests[self.cursor]
                self.submitted[r["request_id"]] = r["arrival_ms"]
                self.cursor += 1
        with self.lock:
            return dict(self.submitted)

    def next_time(self):
        submitted = self.collect()
        pending = [r["arrival_ms"] for r in self.requests if r["request_id"] not in submitted]
        return min(pending) if pending else self.clock.now()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1)


class ArtifactWriter:
    """At most max_pending queued outputs plus one active write occupy host RAM."""

    def __init__(self, directory, max_pending=2):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=max_pending)
        self.thread = threading.Thread(target=self._write, name="benchmark-artifacts", daemon=True)
        self.thread.start()

    def submit(self, record, output, latent, path_history):
        self.queue.put((record, output, latent, list(path_history)))

    def _write(self):
        while True:
            task = self.queue.get()
            try:
                if task is None:
                    return
                record, output, latent, path_history = task
                try:
                    import numpy as np
                    # Never interpolate externally supplied IDs into a filesystem path.
                    stem = canonical_hash(record["generation"])
                    paths = {"output": self.directory / (stem + "-output.npy"),
                             "latent": self.directory / (stem + "-latent.npy")}
                    for key, value in (("output", output), ("latent", latent)):
                        with paths[key].open("xb") as stream:
                            np.save(stream, value, allow_pickle=False)
                    artifact = {key + "_path": str(path) for key, path in paths.items()}
                    artifact.update({key + "_sha256": file_hash(path) for key, path in paths.items()})
                    artifact.update(generation=copy.deepcopy(record["generation"]), path_history=path_history)
                    record["artifact"] = artifact
                except Exception as error:
                    record["artifact_error"] = str(error)
                    record["quality"] = dict(status="unknown", reason="artifact_write_failed")
            finally:
                self.queue.task_done()

    def close(self):
        self.queue.put(None)
        self.queue.join()
        self.thread.join()


def _record(request):
    return dict(request_id=request["request_id"], sample_id=request.get("sample_id"),
                arrival_ms=request["arrival_ms"], deadline_ms=request["deadline_ms"],
                submit_ms=None, start_ms=None, finish_ms=None, status="pending",
                shape_id=request["shape_id"], generation=copy.deepcopy(request), events=[],
                artifact=None, quality=dict(status="unknown", reason="not_evaluated"))


def _standalone(adapter, request, budget=None):
    _check(budget)
    adapter.begin_measurement()
    started = adapter.clock.now()
    state = adapter.prepare(request)
    try:
        while state.step_index < request["steps"]:
            _check(budget)
            adapter.step(state)
        state_mb = adapter.state_mb(state)
        latent = adapter.latent(state)
        _check(budget)
        output = adapter.finish(state)
        events = copy.deepcopy(state.events)
        metrics = dict(service_ms=adapter.clock.now() - started,
                       prepare_ms=sum(e["ms"] for e in events if e["kind"] == "prepare"),
                       cpu_prompt_encode_ms=sum(e.get("cpu_prompt_encode_ms", 0.) for e in events if e["kind"] == "prepare"),
                       step_ms=[e["ms"] for e in events if e["kind"] == "step"],
                       decode_ms=sum(e["ms"] for e in events if e["kind"] == "decode"),
                       peak_increment_mb=adapter.peak_increment_mb(), state_mb=state_mb)
        return output, latent, [e["path"] for e in events if e["kind"] == "step"], metrics
    finally:
        adapter.release(state)


def build_profile(config, dataset, backend, output_dir, device="cuda:0", model_path=None, budget=None,
                  attention="math"):
    backend = canonical_backend(backend)
    for value in (config, dataset):
        if "sha256" in value:
            verify_seal(value)
    dev = [s for s in dataset["samples"] if s.get("service_split", s.get("split")) == "dev"]
    if not dev:
        raise ValueError("profiling requires a separate development sample split")
    # A bounded, deterministic dev sample count is distinct from the full quality suite.
    dev = dev[:int(config.get("profile_samples", 2))]
    if not dev:
        raise ValueError("profile_samples must be positive")
    root = _fresh(output_dir)
    result = dict(schema_version="ditserve.benchmark.profile.v1", backend=backend,
                  measurement_kind="unverified", status="unverified", shapes={},
                  completed_measurements=[],
                  dev_sample_ids=[s["sample_id"] for s in dev], config=copy.deepcopy(config),
                  config_sha256=config.get("sha256", canonical_hash(config)),
                  dataset_sha256=dataset.get("sha256", canonical_hash(dataset)),
                  environment=environment_info(), source_identity=source_identity())
    adapter = None
    try:
        _check(budget, 300. if backend == "wan" else 0.)
        if backend != "core":
            first = _generation(config, dev[0], next(iter(config["shapes"])))
            if backend == "wan":
                _wan_identity(first)
            adapter = make_backend(backend, device=device, model_path=model_path, attention=attention)
            result["execution_identity"] = adapter.execution_identity()
            result["smoke"] = adapter.smoke(first, budget)
            result["measurement_kind"] = adapter.measurement_kind
        else:
            result["measurement_kind"] = "synthetic_cpp_reference"
            result["execution_identity"] = dict(backend="core", gpu=None, clock="synthetic_event_simulation")
        for shape_id, shape in config["shapes"].items():
            measurements = []
            for index, sample in enumerate(dev):
                _check(budget)
                request = _generation(config, sample, shape_id, seed=index,
                                      request_id="profile-%s-%d" % (shape_id, index))
                if backend == "core":
                    raw = run_core([dict(request, arrival_ms=0, deadline_ms=1e12)],
                                   "cache_only", config.get("core", {}))
                    service = raw["records"][0]["completion_ms"]
                    factor = shape["height"] * shape["width"] * shape["frames"] / (512. * 512)
                    costs = raw["cost_model"]
                    state_mb = factor * (costs["latent_mb"] + (costs["cache_mb"] if request["cache_enabled"] else 0.))
                    measurements.append(dict(service_ms=service, prepare_ms=0., decode_ms=0.,
                        step_ms=[service / shape["steps"]] * shape["steps"],
                        peak_increment_mb=raw["summary"]["peak_device_state_and_workspace_mb"], state_mb=state_mb))
                else:
                    output, latent, path, measured = _standalone(adapter, request, budget)
                    measurements.append(measured)
                    del output, latent
                result["completed_measurements"].append(dict(shape_id=shape_id,
                    sample_id=sample["sample_id"], seed=index, generation=request,
                    metrics=copy.deepcopy(measurements[-1])))
            summary = {key: statistics.median(m.get(key, 0.) for m in measurements)
                       for key in ("service_ms", "prepare_ms", "cpu_prompt_encode_ms", "decode_ms")}
            summary.update(step_ms=[statistics.median(m["step_ms"][i] for m in measurements)
                                    for i in range(shape["steps"])],
                           peak_increment_mb=max(m["peak_increment_mb"] for m in measurements),
                           state_mb=max(m["state_mb"] for m in measurements), samples=measurements)
            summary.update({key: shape[key] for key in ("height", "width", "frames", "steps")})
            result["shapes"][shape_id] = summary
        result["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = _error_reason(error)
        result["status"] = "partial" if result["completed_measurements"] else "unverified"
        result["interrupted"] = isinstance(error, KeyboardInterrupt)
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except (Exception, KeyboardInterrupt) as error:
                result["cleanup_error"] = _error_reason(error)
                result["status"] = "partial"
    result = seal(result)
    atomic_json(root / "profile.json", result)
    return result


def _remaining(request, state, profile, encoded=False):
    shape = profile["shapes"][request["shape_id"]]
    if state is None:
        return max(0., shape["service_ms"] - (shape.get("cpu_prompt_encode_ms", 0.) if encoded else 0.))
    return sum(shape["step_ms"][state.step_index:]) + shape["decode_ms"]


def select_ready(policy, ready, current, states, profile, encoded_ids=()):
    """Only the ready list is accepted. Equal priority preserves current work."""
    if policy not in POLICIES:
        raise ValueError("unsupported scheduler: " + policy)
    if not ready:
        return None
    if policy == "fcfs" and current in {r["request_id"] for r in ready}:
        return next(r for r in ready if r["request_id"] == current)
    def priority(r):
        if policy == "fcfs":
            return r["arrival_ms"]
        if policy == "edf":
            return r["deadline_ms"]
        return _remaining(r, states.get(r["request_id"]), profile, r["request_id"] in encoded_ids)
    best = min(ready, key=lambda r: (priority(r), r["arrival_ms"], r["request_id"]))
    active = next((r for r in ready if r["request_id"] == current), None)
    return active if active is not None and priority(active) <= priority(best) else best


def _room(adapter, request, states, rows, profile, clock, budget):
    target = states.get(request["request_id"])
    required = profile["shapes"][request["shape_id"]]["peak_increment_mb"]
    if target is not None and target.residency == "gpu":
        required = max(0., required - adapter.state_mb(target))
    victims = [s for rid, s in states.items() if rid != request["request_id"] and s.residency == "gpu"]
    victims.sort(key=lambda s: (rows[s.spec["request_id"]].get("last_active_ms", 0), s.spec["request_id"]))
    while adapter.available_mb() < required and victims:
        _check(budget)
        victim = victims.pop(0)
        start = clock.now()
        event = adapter.snapshot(victim)
        rows[victim.spec["request_id"]]["events"].append(dict(event, start_ms=start, end_ms=clock.now()))
    return adapter.available_mb() >= required


def _collect_encoding(prefetch, rows, cutoff=None):
    if prefetch is None:
        return
    starts, events = prefetch.collect()
    for rid, instant in starts.items():
        if cutoff is not None and instant > cutoff:
            continue
        previous = rows[rid]["start_ms"]
        rows[rid]["start_ms"] = instant if previous is None else min(previous, instant)
    for rid, event in events:
        if cutoff is not None and event["end_ms"] > cutoff:
            event["outside_policy_horizon"] = True
        rows[rid]["events"].append(event)


def _run_policy(trace, policy, adapter, directory, budget, text_prefetch=0):
    from .optimization import execution_profile
    from .prefetch import PromptPrefetcher
    requests = trace["requests"]
    profile = execution_profile(trace)
    clock = VirtualClock() if adapter.virtual else RealClock()
    adapter.clock = clock
    arrivals = ArrivalSource(requests, clock)
    rows = {r["request_id"]: _record(r) for r in requests}
    states = {}
    current = None
    failure = None
    writer = ArtifactWriter(directory)
    prefetch = None
    observations = dict(step_preemptions=0, snapshot_count=0, restore_count=0, max_gpu_ready=0)
    last_executed = None
    timeout_scale = trace.get("config", {}).get("timeout_scale", 10.)
    if isinstance(timeout_scale, bool) or not isinstance(timeout_scale, (int, float)) or not math.isfinite(timeout_scale) or timeout_scale <= 0:
        writer.close()
        raise ValueError("timeout_scale must be positive and finite")
    if text_prefetch:
        try:
            prefetch = PromptPrefetcher(adapter, clock, text_prefetch)
        except BaseException:
            writer.close()
            raise
    quality_generation = trace.get("purpose") == "independent_quality_generation"

    def timeout_at(request):
        # A complete offline quality suite may wait for thousands of samples.
        # Its watchdog bounds execution after first dispatch, not that queue.
        origin = rows[request["request_id"]]["start_ms"] if quality_generation else request["arrival_ms"]
        if origin is None:
            return float("inf")
        # All ablation variants share the same watchdog as well as deadlines.
        return origin + timeout_scale * trace["profile"]["shapes"][request["shape_id"]]["service_ms"]

    arrivals.start()
    try:
        while any(row["status"] == "pending" for row in rows.values()):
            _check(budget)
            _collect_encoding(prefetch, rows)
            submits = arrivals.collect()
            for rid, instant in submits.items():
                rows[rid]["submit_ms"] = instant
            for request in requests:
                rid = request["request_id"]
                timeout = timeout_at(request)
                if rid in submits and rows[rid]["status"] == "pending" and clock.now() >= timeout:
                    rows[rid].update(status="timed_out", terminal_ms=clock.now(), timeout_ms=timeout,
                                     reason="timeout_at_safe_stage_boundary")
                    if rid in states:
                        adapter.release(states.pop(rid))
                    if current == rid:
                        current = None
                if prefetch is not None and rows[rid]["status"] != "pending":
                    prefetch.discard(rid)
            ready = [r for r in requests if r["request_id"] in submits and rows[r["request_id"]]["status"] == "pending"]
            if prefetch is not None:
                pending = [r for r in ready if r["request_id"] not in states and r["request_id"] not in prefetch.jobs]
                while pending and len(prefetch.jobs) < prefetch.capacity:
                    candidate = select_ready(policy, pending, None, states, profile)
                    prefetch.submit(candidate)
                    pending.remove(candidate)
                ready = [r for r in ready if r["request_id"] in states or prefetch.ready(r["request_id"])]
            observations["max_gpu_ready"] = max(observations["max_gpu_ready"], len(ready))
            if not ready:
                if not any(r["status"] == "pending" for r in rows.values()):
                    break
                clock.wait_until(arrivals.next_time())
                if not clock.virtual:
                    time.sleep(.0005)
                continue
            request = select_ready(policy, ready, current, states, profile,
                                   encoded_ids=set(prefetch.jobs) if prefetch else ())
            rid = request["request_id"]
            row = rows[rid]
            if not _room(adapter, request, states, rows, profile, clock, budget):
                row.update(status="resource_rejected", terminal_ms=clock.now(),
                           reason="profiled_peak_does_not_fit_after_full_hot_offload")
                if rid in states:
                    adapter.release(states.pop(rid))
                if current == rid:
                    current = None
                continue
            if last_executed is not None and rid != last_executed:
                previous = states.get(last_executed)
                if previous is not None and 0 < previous.step_index < previous.spec["steps"]:
                    observations["step_preemptions"] += 1
            last_executed = rid
            current = rid
            if row["start_ms"] is None:
                row["start_ms"] = clock.now()
            start = clock.now()
            memory_before = adapter.memory()
            if rid not in states:
                if prefetch is not None:
                    states[rid] = adapter.prepare_encoded(request, prefetch.take(rid))
                else:
                    states[rid] = adapter.prepare(request)
                row["events"].append(dict(kind="prepare", start_ms=start, end_ms=clock.now(),
                    ms=clock.now() - start,
                    cpu_prompt_encode_ms=0. if prefetch is not None else
                        next((e.get("cpu_prompt_encode_ms", 0.) for e in states[rid].events
                              if e["kind"] == "prepare"), 0.),
                    scope="gpu_install_only" if prefetch is not None else "cpu_encode_and_gpu_install"))
            else:
                state = states[rid]
                if state.residency == "cpu":
                    event = adapter.restore(state)
                    row["events"].append(dict(event, start_ms=start, end_ms=clock.now()))
                elif state.step_index < request["steps"]:
                    event = adapter.step(state)
                    row["events"].append(dict(event, start_ms=start, end_ms=clock.now(), ms=clock.now() - start))
                else:
                    latent = adapter.latent(state)
                    output = adapter.finish(state)
                    finish = clock.now()
                    row["events"].append(dict(kind="decode", start_ms=start, end_ms=finish, ms=finish - start))
                    row.update(status="completed", finish_ms=finish, terminal_ms=finish,
                               e2e_ms=finish - request["arrival_ms"],
                               deadline_met=finish <= request["deadline_ms"])
                    timeout = timeout_at(request)
                    if finish >= timeout:
                        row.update(status="timed_out", timeout_ms=timeout, reason="timeout_at_decode_boundary")
                    path = [e["path"] for e in state.events if e["kind"] == "step"]
                    adapter.release(state)
                    del states[rid]
                    # Bounded writer backpressure is visible to subsequent requests.
                    write_start = clock.now()
                    writer.submit(row, output, latent, path)
                    row["events"].append(dict(kind="artifact_enqueue", start_ms=write_start,
                        end_ms=clock.now(), ms=clock.now() - write_start, outside_this_request_e2e=True))
                    del output, latent
                    current = None
            stage = next((event for event in reversed(row["events"]) if event["kind"] != "artifact_enqueue"), None)
            if stage is not None:
                stage.update(memory_before=memory_before, memory_after=adapter.memory())
            row["last_active_ms"] = clock.now()
    except (Exception, KeyboardInterrupt) as error:
        failure = _error_reason(error)
    finally:
        cutoff = clock.now()
        submits = arrivals.collect()
        arrivals.close()
        for rid, row in rows.items():
            row["submit_ms"] = submits.get(rid, row["submit_ms"])
            if row["status"] == "pending":
                row.update(status="arrived_aborted" if row["arrival_ms"] <= cutoff else "not_arrived",
                           terminal_ms=cutoff, reason=failure or "run_stopped")
        cleanup_errors = []
        if prefetch is not None:
            try:
                prefetch.close()
                _collect_encoding(prefetch, rows, cutoff)
            except (Exception, KeyboardInterrupt) as error:
                cleanup_errors.append(_error_reason(error))
        for state in list(states.values()):
            try:
                adapter.release(state)
            except (Exception, KeyboardInterrupt) as error:
                cleanup_errors.append(_error_reason(error))
        writer.close()
    artifact_errors = [r["request_id"] for r in rows.values() if r.get("artifact_error")]
    stage_totals = {}
    for row in rows.values():
        for event in row["events"]:
            stage_totals[event["kind"]] = stage_totals.get(event["kind"], 0.) + event.get("ms", 0.)
            observations["snapshot_count"] += int(event["kind"] == "snapshot")
            observations["restore_count"] += int(event["kind"] == "restore")
    status = "completed" if failure is None and not artifact_errors and not cleanup_errors else "partial"
    return dict(status=status, horizon_ms=cutoff, requests=list(rows.values()),
        error=failure, artifact_errors=artifact_errors, cleanup_errors=cleanup_errors,
        timing_scope="scheduled arrival through CPU prepare, queue, solver, transfer, decode; bounded writer may delay subsequent work",
        arrival_mode="virtual_open_loop" if clock.virtual else "independent_producer_open_loop",
        measured=not clock.virtual, output_writer_max_pending=2, stage_wall_ms=stage_totals,
        text_prefetch=text_prefetch, text_prefetch_peak_pending=prefetch.peak_pending if prefetch else 0,
        stage_wall_semantics="CPU text_encode may overlap GPU phases; do not sum phases into E2E",
        observations=observations,
        device_reserved_seconds=None if clock.virtual else cutoff / 1000,
        device_reserved_scope="one allocated GPU during timed policy, including idle; not kernel busy time or full lifecycle budget")


def _core_run(trace, policy, root):
    raw = run_core(trace["requests"], policy, trace.get("config", {}).get("core", {}))
    atomic_json(root / ("core-" + core_policy(policy) + ".json"), raw)
    originals = {r["request_id"]: r for r in trace["requests"]}
    rows = []
    for entry in raw["records"]:
        row = _record(originals[entry["request_id"]])
        row.update(status=entry["status"], submit_ms=entry["arrival_ms"],
                   start_ms=entry["first_start_ms"], finish_ms=entry["completion_ms"],
                   terminal_ms=entry["terminal_ms"], core_reference_parity=entry["quality_ok"],
                   core_fingerprint=entry["fingerprint"], path_history=entry["path_history"],
                   events=[e for e in raw["events"] if e.get("request_id") == entry["request_id"]])
        row["quality"] = dict(status="unknown", reason="core_reference_parity_is_not_video_quality")
        rows.append(row)
    return dict(status="completed", horizon_ms=max((r["terminal_ms"] for r in rows), default=0),
                requests=rows, measured=False, core_policy=core_policy(policy),
                core_summary=raw["summary"], arrival_mode="cpp_synthetic_event_simulation")


def validate_execution_options(trace, backend, attention, text_prefetch):
    from .optimization import execution_profile
    profile = execution_profile(trace)
    if type(text_prefetch) is not int or not 0 <= text_prefetch <= 16:
        raise ValueError("text_prefetch must be an integer between 0 and 16")
    if text_prefetch and backend != "wan":
        raise ValueError("text prefetch requires the real Wan backend; CPU fixtures use a virtual clock")
    if "optimization" in trace and trace["optimization"].get("arguments") != dict(attention=attention, text_prefetch=text_prefetch):
        raise ValueError("execution options differ from frozen optimization variant")
    if backend == "wan":
        recipe = profile.get("execution_identity", {}).get("attention", {})
        expected = recipe.get("sdpa_backend") if isinstance(recipe, dict) else None
        mapping = {"math": "MATH", "flash": "FLASH_ATTENTION", "efficient": "EFFICIENT_ATTENTION"}
        if attention not in mapping or (expected is not None and expected != mapping[attention]):
            raise ValueError("profile attention differs from execution options")
    elif attention != "math":
        raise ValueError("attention selection only applies to Wan")
    return profile


def run_suite(trace, policies, backend, output_dir, device="cuda:0", model_path=None, budget=None,
              attention="math", text_prefetch=0, quality_reference=None, quality_epsilon=.05):
    from .workload import validate_trace
    validate_trace(trace)
    backend = canonical_backend(backend)
    profile = validate_execution_options(trace, backend, attention, text_prefetch)
    if "sha256" in profile:
        verify_seal(profile)
    if profile.get("backend") != backend or profile.get("status", "completed") != "completed":
        raise ValueError("trace requires a completed profile from the selected backend")
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("policies must be a nonempty unique list")
    for policy in policies:
        if backend == "core":
            core_policy(policy)
        elif policy not in POLICIES:
            raise ValueError("Unsupported policy %s; adaptive PreViS requires an explicit calibrated adapter" % policy)
    if not trace["requests"] or len({r["request_id"] for r in trace["requests"]}) != len(trace["requests"]):
        raise ValueError("trace must contain requests with unique IDs")
    root = _fresh(output_dir)
    atomic_json(root / "trace.json", trace)
    atomic_json(root / "profile.json", profile)
    result = dict(schema_version="ditserve.benchmark.run.v1", trace_sha256=trace["sha256"],
                  backend=backend, measurement_kind="unverified", measured=False, status="unverified",
                  purpose=trace.get("purpose", "online_serving"),
                  serving_performance_evidence=bool(backend == "wan" and trace.get("serving_performance_evidence", True)),
                  policies={}, references={}, config=trace.get("config", {}),
                  profile_sha256=trace["profile"].get("sha256"), execution_profile_sha256=profile.get("sha256"),
                  dataset_sha256=trace.get("dataset_sha256", trace.get("dataset", {}).get("sha256")),
                  execution_options=dict(attention=attention, text_prefetch=text_prefetch),
                  environment=environment_info(), source_identity=source_identity())
    adapter = None
    try:
        # Check the whole paired group plus independent references before doing any work.
        estimate = sum(profile["shapes"][r["shape_id"]]["service_ms"] for r in trace["requests"])
        arrival_span = max(r["arrival_ms"] for r in trace["requests"])
        # A conservative reservation includes idle arrival gaps plus full drain,
        # independent references, small smoke and loading allowance. Actual
        # lifecycle elapsed time is always charged by the enclosing budget session.
        reservation = (estimate * (len(policies) + 1) + arrival_span * len(policies)) / 1000
        if backend == "wan":
            reservation += 300.
        result["group_reservation_seconds"] = reservation
        _check(budget, reservation)
        if backend == "core":
            result["measurement_kind"] = "synthetic_cpp_reference"
            result["execution_identity"] = dict(backend="core", gpu=None, clock="synthetic_event_simulation")
            for policy in policies:
                result["policies"][policy] = _core_run(trace, policy, root)
        else:
            if backend == "wan":
                for request in trace["requests"]:
                    _wan_identity(request)
            adapter = make_backend(backend, device=device, model_path=model_path, attention=attention)
            result["execution_identity"] = adapter.execution_identity()
            if backend == "wan":
                _verify_execution_identity(profile.get("execution_identity"), result["execution_identity"])
            result["smoke"] = adapter.smoke(trace["requests"][0], budget)
            result["measurement_kind"] = adapter.measurement_kind
            result["measured"] = not adapter.virtual
            reference_rows = []
            writer = ArtifactWriter(root / "references")
            try:
                for request in trace["requests"]:
                    output, latent, path, metrics = _standalone(adapter, request, budget)
                    row = _record(request)
                    reference_rows.append(row)
                    writer.submit(row, output, latent, path)
                    del output, latent
            finally:
                writer.close()
                result["references"] = {r["request_id"]: r["artifact"] for r in reference_rows if r["artifact"]}
                result["reference_errors"] = {r["request_id"]: r["artifact_error"] for r in reference_rows if r.get("artifact_error")}
            if result["reference_errors"]:
                raise RuntimeError("independent reference artifact persistence failed")
            if quality_reference is not None:
                from .backend_quality import check_backend_quality
                result["backend_quality_gate"] = check_backend_quality(result["references"], quality_reference,
                    root, epsilon=quality_epsilon, budget=budget, candidate_identity=result["execution_identity"])
                atomic_json(root / "backend-quality.json", result["backend_quality_gate"])
                if result["backend_quality_gate"]["status"] != "passed":
                    raise RuntimeError("cross-attention quality gate failed; online execution not started")
            for policy in policies:
                _check(budget)
                options = dict(text_prefetch=text_prefetch) if text_prefetch else {}
                result["policies"][policy] = _run_policy(trace, policy, adapter, root / policy, budget, **options)
                # Preserve each completed attempt even if a subsequent policy fails.
                atomic_json(root / "run.json", seal(result))
                if result["policies"][policy]["status"] != "completed":
                    break
        result["status"] = ("completed" if len(result["policies"]) == len(policies)
                            and all(r["status"] == "completed" for r in result["policies"].values()) else "partial")
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = _error_reason(error)
        result["interrupted"] = isinstance(error, KeyboardInterrupt)
        result["status"] = "partial" if result["policies"] else "unverified"
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except (Exception, KeyboardInterrupt) as error:
                result["cleanup_error"] = _error_reason(error)
                result["status"] = "partial"
    for policy in policies:
        if policy not in result["policies"]:
            records = [_record(r) for r in trace["requests"]]
            for row in records:
                row.update(status="not_arrived", reason="policy_not_started")
            result["policies"][policy] = dict(status="not_run", horizon_ms=0., requests=records,
                                              reason=result.get("error", "previous_policy_incomplete"))
    if not result["measured"]:
        result["serving_performance_evidence"] = False
    result = seal(result)
    atomic_json(root / "run.json", result)
    return result
