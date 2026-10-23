"""Opt-in, arrival-driven Wan experiments; import and scheduling are CPU-only.

The manifest freezes inputs, arrival times, service estimates and SLOs. Model
loading/warmup precede each observation's clock. This is an experiment executor,
not a change to the production scheduler. No embeddings are cached or computed
before their request arrives. Every worker has at most one lookahead request.
"""

import argparse
from collections import deque
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import re
import signal
import threading
import time
import traceback


MANIFEST_SCHEMA = "ditserve.workflow.serving.manifest.v1"
SCHEMA = "ditserve.workflow.serving.v1"


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError("invalid " + name)
    return value


def validate_manifest(value):
    """Validate before importing torch or creating processes; return a copy."""
    from ..models.wan.runtime import normalize_spec
    result = copy.deepcopy(value)
    if result.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported serving manifest schema")
    if result.get("kind") not in ("calibration", "placement", "online"):
        raise ValueError("kind must be calibration, placement or online")
    result.setdefault("save_videos", False)
    if type(result["save_videos"]) is not bool:
        raise ValueError("save_videos must be a boolean")
    result["placement"] = {"A": "dual_cpu", "B": "dedicated_fp32"}.get(result.get("placement"), result.get("placement", "dual_cpu"))
    if result["placement"] not in ("dual_cpu", "dedicated_fp32"):
        raise ValueError("unsupported placement")
    result.setdefault("policy", "shared_fifo")
    if result["policy"] not in ("shared_fifo", "least_work_fifo"):
        raise ValueError("unsupported queue policy")
    result.setdefault("controller_cpu", 59)
    result.setdefault("worker_cpu_sets", [list(range(8)), list(range(60, 68))])
    result.setdefault("worker_devices", ["cuda:0", "cuda:1"])
    if result["worker_devices"] != ["cuda:0", "cuda:1"]:
        raise ValueError("this run02 executor requires physical cuda:0,cuda:1")
    cpus = result["worker_cpu_sets"]
    if len(cpus) != 2 or any(len(group) != 8 or len(set(group)) != 8 for group in cpus):
        raise ValueError("each process requires exactly eight distinct CPU IDs")
    if any(type(cpu) is not int or cpu < 0 for group in cpus for cpu in group) or set(cpus[0]) & set(cpus[1]):
        raise ValueError("worker CPU sets must be disjoint nonnegative integers")
    if type(result["controller_cpu"]) is not int or result["controller_cpu"] < 0 or any(result["controller_cpu"] in group for group in cpus):
        raise ValueError("controller CPU must be separate from worker CPUs")
    result.setdefault("queue_limit", 64)
    result.setdefault("prefetch_depth", 1)
    if result["queue_limit"] != 64 or result["prefetch_depth"] != 1:
        raise ValueError("run02 freezes queue_limit=64 and prefetch_depth=1")
    if not isinstance(result.get("model_path"), str) or not result["model_path"]:
        raise ValueError("model_path is required")
    result.setdefault("startup_timeout_seconds", 600)
    result.setdefault("observation_timeout_seconds", 1800)
    result.setdefault("sample_interval_seconds", .1)
    for key in ("startup_timeout_seconds", "observation_timeout_seconds", "sample_interval_seconds"):
        _positive(result[key], key)
    observations = result.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("nonempty observations are required")
    observation_ids = set()
    for observation in observations:
        identifier = observation.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier) or identifier in observation_ids:
            raise ValueError("unsafe or duplicate observation ID")
        observation_ids.add(identifier)
        observation.setdefault("role", "measurement")
        if observation["role"] not in ("measurement", "saturation", "isolated"):
            raise ValueError("unknown observation role")
        for key in ("arrival_window_ms", "observation_cutoff_ms"):
            if key in observation:
                _positive(observation[key], key, zero=key == "arrival_window_ms")
        requests = observation.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError("each observation needs requests")
        ids, previous = set(), -1
        estimates = observation.get("estimates_ms", result.get("estimates_ms", {}))
        for request in requests:
            identifier = request.get("request_id")
            if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier) or identifier in ids:
                raise ValueError("unsafe or duplicate request ID")
            ids.add(identifier)
            arrival = _positive(request.get("arrival_ms", 0), "arrival_ms", zero=True)
            if arrival < previous:
                raise ValueError("requests must be sorted by arrival_ms")
            request["arrival_ms"], previous = arrival, arrival
            _positive(request.get("slo_ms"), "slo_ms")
            spec = dict(request.get("spec", {}))
            if "request_id" in spec and spec["request_id"] != identifier:
                raise ValueError("request and spec IDs disagree")
            spec["request_id"] = identifier
            request["spec"] = normalize_spec(spec)
            if result["policy"] == "least_work_fifo":
                estimate = estimates.get(str(spec["frames"]), estimates.get(spec["frames"]))
                if not isinstance(estimate, dict):
                    raise ValueError("least_work_fifo needs frozen per-frame service estimates")
                for key in ("encode", "gpu"):
                    _positive(estimate.get(key), "estimated " + key)
        observation["estimates_ms"] = estimates
    return result


class ArrivalScheduler:
    """Pure CPU queue state. Only active service, never waiting, consumes work.

    Both policies retain a global unclaimed FIFO until an execution/lookahead
    slot is available. Least-work then selects the eligible worker with the
    smallest estimated remaining service sum. Each worker can have one current
    request plus one encoded/encoding lookahead. The 64-request waiting bound
    counts every arrived request whose encoding has not started.
    """

    def __init__(self, policy, workers, estimates=None, queue_limit=64, depth=1, rotate_idle=False):
        if policy not in ("shared_fifo", "least_work_fifo") or depth != 1 or not workers:
            raise ValueError("invalid scheduler configuration")
        self.policy, self.workers = policy, tuple(workers)
        self.estimates, self.queue_limit = estimates or {}, queue_limit
        self.rows, self.shared = {}, deque()
        self.active = {worker: [] for worker in workers}
        self.dispatch_counts = {worker: 0 for worker in workers}
        self.capacity = 1 + depth
        self.rotate_idle = rotate_idle

    def _prediction(self, row, worker, now_ns):
        estimate = self.estimates.get(str(row["spec"]["frames"]))
        row.update(route_decision_ns=now_ns, route_prediction_basis="sum of frozen remaining encode/GPU service; only active service subtracts elapsed")
        if estimate is None:
            row.update(predicted_complete_ns=None, estimated_work_ahead_ms=None, estimated_service_ms=None)
        else:
            ahead = self.remaining_ms(worker, now_ns)
            service = float(estimate["encode"]) + float(estimate["gpu"])
            row.update(estimated_work_ahead_ms=ahead, estimated_service_ms=service,
                       predicted_complete_ns=now_ns + round((ahead + service) * 1e6))

    def remaining_ms(self, worker, now_ns):
        total = 0.
        for identifier in self.active[worker]:
            row = self.rows[identifier]
            estimate = self.estimates.get(str(row["spec"]["frames"]), {})
            encode, gpu = float(estimate.get("encode", 0)), float(estimate.get("gpu", 0))
            if row.get("encode_end_ns") is not None:
                encode = 0.
            elif row.get("encode_start_ns") is not None:
                encode = max(0., encode - (now_ns - row["encode_start_ns"]) / 1e6)
            if row.get("gpu_start_ns") is not None:
                gpu = max(0., gpu - (now_ns - row["gpu_start_ns"]) / 1e6)
            total += encode + gpu
        return total

    def arrive(self, request, scheduled_ns, now_ns):
        identifier = request["request_id"]
        if identifier in self.rows or now_ns < scheduled_ns:
            raise ValueError("duplicate request or arrival released early")
        row = copy.deepcopy(request)
        row.update(status="queued", scheduled_arrival_ns=scheduled_ns, enqueue_ns=now_ns)
        self.rows[identifier] = row
        waiting = sum(item["status"] in ("queued", "submitted") for key, item in self.rows.items() if key != identifier)
        if waiting >= self.queue_limit:
            row.update(status="rejected", reason="waiting_queue_full", terminal_ns=now_ns)
            return False
        self.shared.append(identifier)
        return True

    def dispatch(self, now_ns):
        assignments = []
        while True:
            def gpu_busy(worker):
                return any(self.rows[key].get("gpu_start_ns") is not None for key in self.active[worker])
            # One lookahead exists only while an earlier request occupies the
            # GPU. Before that, the worker already has its one encoder slot.
            available = [worker for worker in self.workers if (not self.active[worker] or
                         (gpu_busy(worker) and len(self.active[worker]) < self.capacity))
                         and self.shared]
            if not available:
                break
            def priority(worker):
                if self.policy == "least_work_fifo":
                    return (self.remaining_ms(worker, now_ns), gpu_busy(worker), worker)
                return (gpu_busy(worker), self.dispatch_counts[worker] if self.rotate_idle else 0, worker)
            worker = min(available, key=priority)
            identifier = self.shared.popleft()
            row = self.rows[identifier]
            self._prediction(row, worker, now_ns)
            row.update(status="submitted", assigned_worker=worker, dispatch_ns=now_ns)
            self.active[worker].append(identifier)
            self.dispatch_counts[worker] += 1
            assignments.append((worker, row))
        return assignments

    def event(self, event):
        identifier = event["request_id"]
        if identifier not in self.rows:
            raise ValueError("event for unknown request")
        row = self.rows[identifier]
        # Multiprocessing preserves ordering per producer, not between the
        # dedicated encoder and consumer. Completion carries all timestamps.
        if row["status"] == "completed" and event["kind"] != "completed":
            return row
        if event["kind"] == "encode_started":
            if event["encode_start_ns"] < row["scheduled_arrival_ns"]:
                raise ValueError("encoding before request arrival")
            row["encode_start_ns"] = event["encode_start_ns"]
            if row["status"] == "submitted":
                row["status"] = "encoding"
        elif event["kind"] == "encoded":
            row.update(encode_start_ns=event["encode_start_ns"], encode_end_ns=event["encode_end_ns"])
            if row.get("gpu_start_ns") is None:
                row["status"] = "encoded"
        elif event["kind"] == "gpu_started":
            row.update(status="gpu", gpu_start_ns=event["gpu_start_ns"])
        elif event["kind"] == "completed":
            row.update({key: value for key, value in event.items() if key not in ("kind", "request_id", "process")})
            row.update(status="completed", terminal_ns=event["complete_ns"])
            self.active[row["assigned_worker"]].remove(identifier)
        else:
            raise ValueError("unknown request event")
        return row


def _quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def summarize_requests(rows, origin_ns, end_ns, arrival_window_ms=None):
    if end_ns < origin_ns:
        raise ValueError("observation ends before its origin")
    completed = [row for row in rows if row["status"] == "completed"]
    latencies = [(row["complete_ns"] - row["scheduled_arrival_ns"]) / 1e6 for row in completed]
    if any(value < 0 for value in latencies):
        raise ValueError("negative response latency")
    met = sum(latency <= row["slo_ms"] for row, latency in zip(completed, latencies))
    wall = (end_ns - origin_ns) / 1e6
    arrivals = [row["scheduled_arrival_ns"] for row in rows if row.get("scheduled_arrival_ns") is not None]
    rejected = sum(row["status"] == "rejected" for row in rows)
    arrival_window = arrival_window_ms if arrival_window_ms is not None else ((max(arrivals) - min(arrivals)) / 1e6 if arrivals else None)
    return dict(wall_ms=wall, arrival_window_ms=arrival_window,
                requests=len(rows), completed=len(completed), rejected=rejected,
                unfinished=len(rows) - len(completed) - rejected, slo_met=met,
                slo_attainment=met / len(rows) if rows else None,
                latency_p50_ms=_quantile(latencies, .5), latency_p95_ms=_quantile(latencies, .95),
                throughput_per_second=len(completed) * 1000 / wall if wall else None,
                ontime_throughput_per_second=met * 1000 / wall if wall else None,
                ontime_per_arrival_window_second=met * 1000 / arrival_window if arrival_window else None,
                arrival_window_basis="frozen observation value" if arrival_window_ms is not None else "last scheduled arrival minus first scheduled arrival",
                latency_samples=len(latencies), semantic="unverified", qualified_goodput_per_second=None,
                latency_basis="complete including D2H, finite and hashes minus scheduled arrival; completed requests only",
                denominator="all frozen requests; observation window includes drain or explicit interruption",
                tail_caveat="small finite trace: empirical p95 is exploratory, not a validated tail SLO")


def _decorate_rows(rows, origin):
    result = []
    for row in rows:
        value = copy.deepcopy(row)
        value["timestamps_ms"] = {key.removesuffix("_ns"): (item - origin) / 1e6
                                  for key, item in row.items() if key.endswith("_ns") and isinstance(item, int)}
        if row.get("encode_start_ns") is not None:
            value["queue_wait_ms"] = (row["encode_start_ns"] - row["scheduled_arrival_ns"]) / 1e6
        if row.get("complete_ns") is not None:
            value["response_ms"] = (row["complete_ns"] - row["scheduled_arrival_ns"]) / 1e6
            value["latency_ms"] = value["response_ms"]
            value["slo_met"] = value["response_ms"] <= row["slo_ms"]
            if row.get("encode_start_ns") is not None:
                value["service_ms"] = (row["complete_ns"] - row["encode_start_ns"]) / 1e6
            if row.get("gpu_start_ns") is not None:
                value["gpu_ms"] = (row["complete_ns"] - row["gpu_start_ns"]) / 1e6
            if row.get("predicted_complete_ns") is not None:
                value["prediction_error_ms"] = (row["complete_ns"] - row["predicted_complete_ns"]) / 1e6
        if row.get("encode_start_ns") is not None and row.get("encode_end_ns") is not None:
            value["encode_ms"] = (row["encode_end_ns"] - row["encode_start_ns"]) / 1e6
        value["frames"] = row["spec"]["frames"]
        result.append(value)
    return result


@contextmanager
def _range(torch, name, request):
    metadata = dict(request_id=request["request_id"], frames=request["spec"]["frames"],
                    purpose=request.get("purpose", "serving_measurement"), observation=request["observation_id"])
    torch.cuda.nvtx.range_push("ditserve.workflow/serving/" + name + "|" + json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _array_summary(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    finite = bool(np.isfinite(value).all())
    if not finite:
        raise RuntimeError("nonfinite serving output")
    return dict(shape=list(value.shape), dtype=str(value.dtype), finite=finite,
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _save_request_arrays(output, request, video, latent, *, save_videos=False):
    """Experiment artifacts only; video retention requires its own opt-in."""
    import numpy as np
    artifacts = {}
    for kind, array, retain in (("video", video, save_videos), ("latent", latent, request.get("save_output", False))):
        if not retain:
            continue
        directory = Path(output) / "outputs" / request["observation_id"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (request["request_id"] + "." + kind + ".npy")
        np.save(path, array, allow_pickle=False)
        artifacts[kind] = str(path)
    return artifacts


def _worker_main(configuration, index, role, incoming, bridge, events, output):
    """A CPU encoder thread feeds its local GPU loop; B transports CPU arrays."""
    name = "process%d" % index
    device = configuration["worker_devices"][index]
    worker = pipe = None
    cpu_begin = {}
    try:
        cpus = configuration["worker_cpu_sets"][index]
        os.sched_setaffinity(0, cpus)
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[variable] = str(len(cpus))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from ..models.wan.runtime import WanStepWorker, EncodedPrompt
        from .workflow_metrics import torch_snapshot
        torch.set_num_threads(len(cpus))
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(device)
        if role == "gpu_encoder":
            from .workflow_encoder import _standalone_encoder
            pipe = _standalone_encoder(configuration["model_path"], "float32", device)
        else:
            worker = WanStepWorker(device, configuration["model_path"], "flash")
        events.put(dict(kind="ready", process=name, role=role, pid=os.getpid(), device=device,
                        cpus=sorted(os.sched_getaffinity(0)), torch_version=torch.__version__,
                        matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                        cudnn_tf32=bool(torch.backends.cudnn.allow_tf32)))

        def control(command):
            observation = command["observation_id"]
            torch.cuda.synchronize(device)
            if command["op"] == "begin":
                torch.cuda.reset_peak_memory_stats(device)
                cpu_begin[observation] = time.process_time_ns()
            events.put(dict(kind="control", process=name, op=command["op"], observation_id=observation,
                            cpu_core_seconds=(time.process_time_ns() - cpu_begin.get(observation, time.process_time_ns())) / 1e9,
                            process_cpu_seconds=time.process_time_ns() / 1e9,
                            allocator=torch_snapshot(device), timestamp_ns=time.perf_counter_ns()))

        encoded = queue.Queue(maxsize=1) if role == "cpu_worker" else bridge

        def encoder_loop():
            try:
                while True:
                    command = incoming.get()
                    if command["op"] == "stop":
                        encoded.put(command)
                        return
                    if command["op"] in ("begin", "end"):
                        if role == "gpu_encoder":
                            control(command)
                        encoded.put(command)
                        continue
                    request = command["request"]
                    started = time.perf_counter_ns()
                    if started < request["scheduled_arrival_ns"]:
                        raise RuntimeError("worker received a request before arrival")
                    events.put(dict(kind="encode_started", process=name, request_id=request["request_id"], encode_start_ns=started))
                    with _range(torch, "text_encoder", request), torch.inference_mode():
                        if role == "cpu_worker":
                            payload = worker.encode_prompt(request["spec"])
                        else:
                            spec = request["spec"]
                            prompt, negative = pipe.encode_prompt(prompt=spec["prompt"], negative_prompt=spec["negative_prompt"],
                                do_classifier_free_guidance=spec["guidance_scale"] > 1, num_videos_per_prompt=1,
                                max_sequence_length=512, device=device, dtype=torch.float32)
                            payload = dict(prompt=prompt.detach().float().cpu().numpy().copy(),
                                           negative=negative.detach().float().cpu().numpy().copy() if negative is not None else None)
                    ended = time.perf_counter_ns()
                    events.put(dict(kind="encoded", process=name, request_id=request["request_id"],
                                    encode_start_ns=started, encode_end_ns=ended))
                    encoded.put(dict(op="request", request=request, payload=payload,
                                     encode_start_ns=started, encode_end_ns=ended))
            except BaseException:
                events.put(dict(kind="fatal", process=name, error=traceback.format_exc()))
                encoded.put(dict(op="stop"))

        if role != "consumer":
            thread = threading.Thread(target=encoder_loop, name="serving-single-encoder", daemon=True)
            thread.start()
            if role == "gpu_encoder":
                thread.join()
                return
        while True:
            command = encoded.get()
            if command["op"] == "stop":
                return
            if command["op"] in ("begin", "end"):
                control(command)
                continue
            request, payload = command["request"], command["payload"]
            if role == "consumer":
                # Ownership is process-local: never pickle/reuse another worker's token.
                payload = EncodedPrompt(spec=request["spec"], prompt_embeds=torch.from_numpy(payload["prompt"]),
                    negative_prompt_embeds=None if payload["negative"] is None else torch.from_numpy(payload["negative"]),
                    cpu_prompt_encode_ms=(command["encode_end_ns"] - command["encode_start_ns"]) / 1e6, owner=worker._owner)
            started = time.perf_counter_ns()
            events.put(dict(kind="gpu_started", process=name, request_id=request["request_id"], gpu_start_ns=started))
            state = None
            try:
                with _range(torch, "prepare", request):
                    state = worker.prepare_encoded(request["spec"], payload)
                prepared = time.perf_counter_ns()
                for step in range(request["spec"]["steps"]):
                    with _range(torch, "dit.step.%d" % step, request):
                        worker.step(state)
                denoised = time.perf_counter_ns()
                with _range(torch, "vae_decode_and_d2h", request):
                    video = worker.finish(state)
                decoded = time.perf_counter_ns()
                with _range(torch, "output_check", request):
                    latent = state.latents.detach().float().cpu().numpy()
                    video_summary, latent_summary = _array_summary(video), _array_summary(latent)
                    artifacts = _save_request_arrays(output, request, video, latent,
                                                    save_videos=configuration["save_videos"])
                complete = time.perf_counter_ns()
                worker.release(state)
                state = None
                events.put(dict(kind="completed", process=name, request_id=request["request_id"],
                    encode_start_ns=command["encode_start_ns"], encode_end_ns=command["encode_end_ns"],
                    gpu_start_ns=started, prepare_end_ns=prepared, dit_end_ns=denoised, decode_end_ns=decoded,
                    complete_ns=complete, video=video_summary, latent=latent_summary, artifacts=artifacts,
                    allocator=torch_snapshot(device)))
            finally:
                if state is not None:
                    worker.release(state)
    except BaseException:
        events.put(dict(kind="fatal", process=name, error=traceback.format_exc()))


class ProcessPipelines:
    def __init__(self, configuration, output):
        self.configuration, self.output = configuration, output
        context = multiprocessing.get_context("spawn")
        self.events = context.Queue()
        self.incoming = [context.Queue(maxsize=4), context.Queue(maxsize=4)]
        self.bridge = context.Queue(maxsize=1)
        self.processes, self.ready = [], []
        self.closed = False
        dedicated = configuration["placement"] == "dedicated_fp32"
        self.workers = [1] if dedicated else [0, 1]
        roles = ["gpu_encoder", "consumer"] if dedicated else ["cpu_worker", "cpu_worker"]
        for index, role in enumerate(roles):
            process = context.Process(target=_worker_main, args=(configuration, index, role, self.incoming[index],
                                      self.bridge, self.events, str(output)), name="serving-" + role + str(index))
            self.processes.append(process)

    def start(self):
        for process in self.processes:
            process.start()
        deadline = time.monotonic() + self.configuration["startup_timeout_seconds"]
        while len(self.ready) < len(self.processes):
            event = self.get(deadline)
            if event is None:
                continue
            if event["kind"] != "ready":
                raise RuntimeError("unexpected event during startup")
            self.ready.append(event)

    def get(self, deadline=None, timeout=.05):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("serving process deadline exceeded")
        try:
            event = self.events.get(timeout=timeout)
        except queue.Empty:
            if any(process.exitcode is not None for process in self.processes):
                raise RuntimeError("serving process exited before observation completion")
            return None
        if event["kind"] == "fatal":
            raise RuntimeError("%s: %s" % (event["process"], event["error"]))
        return event

    def send(self, worker, request):
        target = 0 if self.configuration["placement"] == "dedicated_fp32" else worker
        self.incoming[target].put(dict(op="request", request=request), timeout=5)

    def control(self, op, observation):
        command = dict(op=op, observation_id=observation)
        for worker in ([0] if self.configuration["placement"] == "dedicated_fp32" else self.workers):
            self.incoming[worker].put(command, timeout=5)
        acknowledgements = []
        deadline = time.monotonic() + 60
        while len(acknowledgements) != len(self.processes):
            event = self.get(deadline)
            if event is None:
                continue
            if event["kind"] in ("encode_started", "encoded", "gpu_started", "completed"):
                # Dedicated processes may publish delayed status events after
                # the consumer's completion. Begin/end is sent only at drain.
                continue
            if event["kind"] != "control" or event["op"] != op or event["observation_id"] != observation:
                raise RuntimeError("unexpected event at observation boundary: " + repr(event))
            acknowledgements.append(event)
        return acknowledgements

    def warm(self):
        shapes = {}
        for observation in self.configuration["observations"]:
            for request in observation["requests"]:
                spec = request["spec"]
                shapes.setdefault(tuple(spec[key] for key in ("height", "width", "frames", "steps")), spec)
        for number, spec in enumerate(shapes.values()):
            expected = set()
            for worker in self.workers:
                identifier = "warm-%d-%d" % (worker, number)
                expected.add(identifier)
                warm_spec = dict(spec, request_id=identifier)
                self.send(worker, dict(request_id=identifier, spec=warm_spec, purpose="warmup",
                    observation_id="warmup", scheduled_arrival_ns=time.perf_counter_ns()))
            deadline = time.monotonic() + self.configuration["startup_timeout_seconds"]
            while expected:
                event = self.get(deadline)
                if event is not None and event["kind"] == "completed":
                    expected.remove(event["request_id"])

    def close(self):
        if self.closed:
            return
        self.closed = True
        for target in ([0] if self.configuration["placement"] == "dedicated_fp32" else self.workers):
            try:
                self.incoming[target].put_nowait(dict(op="stop"))
            except queue.Full:
                pass
        for process in self.processes:
            if process.pid is None:
                continue
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
        for connection in self.incoming + [self.bridge, self.events]:
            connection.cancel_join_thread()
            connection.close()


def memory_margin_gate(rows, *, sampled=False, origin_ns=None, end_ns=None):
    """A missing counter is unavailable, never invented free capacity."""
    devices = {}
    for item in rows:
        row = item if sampled else item.get("allocator", {})
        if sampled and not origin_ns <= row.get("timestamp_ns", -1) <= end_ns:
            continue
        device = row.get("device")
        if isinstance(device, str) and device.startswith("cuda:"):
            device = int(device.split(":")[-1])
        free = row.get("memory_free_bytes" if sampled else "device_free_bytes")
        total = row.get("memory_total_bytes" if sampled else "device_total_bytes")
        if device not in (0, 1) or not isinstance(free, int) or not isinstance(total, int) or total <= 0:
            continue
        required = max(1024 ** 3, math.ceil(total * .05))
        previous = devices.get(device)
        if previous is None or free < previous["minimum_free_bytes"]:
            devices[device] = dict(device=device, minimum_free_bytes=free, total_bytes=total,
                                   required_free_bytes=required, passed=free >= required)
    status = "failed" if any(not row["passed"] for row in devices.values()) else "passed" if set(devices) == {0, 1} else "unverified"
    return dict(status=status, devices=list(devices.values()), eligible=status == "passed",
                basis="NVML sampled minimum free within observation; short peaks can be missed" if sampled else "post-warmup synchronized cuda.mem_get_info",
                rule="free >= max(1 GiB, 5% reported device capacity)")


def run_observation(configuration, observation, pipelines, checkpoint=None):
    from .workflow_metrics import NvmlSampler, process_snapshot
    begin = pipelines.control("begin", observation["id"])
    capacity = memory_margin_gate(begin)
    if not capacity["eligible"]:
        origin = time.perf_counter_ns()
        rows = [dict(row, status="not_admitted", scheduled_arrival_ns=None) for row in observation["requests"]]
        output = dict(id=observation["id"], role=observation["role"], status="capacity_rejected", origin_ns=origin,
                      requests=_decorate_rows(rows, origin), resources_begin=begin, capacity_gate=capacity,
                      performance_evidence=False, semantic="unverified",
                      summary=summarize_requests(rows, origin, origin, observation.get("arrival_window_ms")))
        if checkpoint is not None:
            checkpoint(output)
        return output
    sampler = NvmlSampler(devices=(0, 1), interval_s=configuration["sample_interval_seconds"])
    sampler.start()
    origin = time.perf_counter_ns()
    cpu_origin = time.process_time_ns()
    scheduler = ArrivalScheduler(configuration["policy"], pipelines.workers, observation["estimates_ms"],
                                 rotate_idle=observation["role"] == "isolated")
    output = dict(id=observation["id"], role=observation["role"], status="running", origin_ns=origin,
                  requests=[], resources_begin=begin, capacity_gate=capacity, semantic="unverified")
    pending = deque(copy.deepcopy(observation["requests"]))
    isolated = observation["role"] == "isolated"
    deadline = time.monotonic() + configuration["observation_timeout_seconds"]
    cutoff_ns = origin + round(observation["observation_cutoff_ms"] * 1e6) if observation.get("observation_cutoff_ms") is not None else None
    cutoff = False
    def dispatch():
        for worker, request in scheduler.dispatch(time.perf_counter_ns()):
            pipelines.send(worker, dict(request, observation_id=observation["id"], purpose="serving_measurement"))
    try:
        while pending or any(row["status"] not in ("completed", "rejected") for row in scheduler.rows.values()):
            now = time.perf_counter_ns()
            if cutoff_ns is not None and now >= cutoff_ns:
                cutoff = True
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("observation time limit exceeded")
            while pending:
                if isolated and any(row["status"] not in ("completed", "rejected") for row in scheduler.rows.values()):
                    break
                scheduled = now if isolated else origin + round(pending[0]["arrival_ms"] * 1e6)
                if scheduled > now:
                    break
                request = pending.popleft()
                scheduler.arrive(request, scheduled, now)
                dispatch()
                if isolated:
                    break
            wait_seconds = .002
            if pending and not isolated:
                next_arrival = origin + round(pending[0]["arrival_ms"] * 1e6)
                wait_seconds = min(wait_seconds, max(0., (next_arrival - time.perf_counter_ns()) / 1e9))
            if cutoff_ns is not None:
                wait_seconds = min(wait_seconds, max(0., (cutoff_ns - time.perf_counter_ns()) / 1e9))
            event = pipelines.get(deadline, timeout=wait_seconds)
            if event is not None:
                # Late IPC cannot turn a completion after the frozen cutoff
                # into in-window throughput. Other events retain diagnostic data.
                if event["kind"] == "completed" and cutoff_ns is not None and event["complete_ns"] > cutoff_ns:
                    scheduler.rows[event["request_id"]]["completion_after_cutoff_ns"] = event["complete_ns"]
                    cutoff = True
                    break
                scheduler.event(event)
                dispatch()
                if event["kind"] == "completed" and checkpoint is not None:
                    output["requests"] = _decorate_rows(list(scheduler.rows.values()), origin)
                    checkpoint(output)
        output.update(status="completed", stopped_at_cutoff=cutoff)
    except BaseException as error:
        output.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=traceback.format_exc())
        raise
    finally:
        end = cutoff_ns if cutoff else time.perf_counter_ns()
        controller_cpu_end = time.process_time_ns()
        if cutoff:
            # Drain a bounded IPC grace window without dispatching anything.
            # Only device-complete timestamps at/before the cutoff count.
            try:
                for unused in range(25):
                    event = pipelines.get(timeout=.002)
                    if event is None:
                        continue
                    if event.get("request_id") not in scheduler.rows:
                        continue
                    if event["kind"] == "completed":
                        if event["complete_ns"] <= end:
                            scheduler.event(event)
                        else:
                            scheduler.rows[event["request_id"]]["completion_after_cutoff_ns"] = event["complete_ns"]
                    elif all(value <= end for key, value in event.items() if key.endswith("_ns")):
                        scheduler.event(event)
            except Exception:
                # Cutoff is also a likely moment for a worker to fail. Keep
                # the bounded observation and always stop sampling/processes.
                output.update(status="failed", cutoff_drain_error=traceback.format_exc())
            resources = []
            for before in begin:
                process = next((row for row in pipelines.ready if row["process"] == before["process"]), {})
                snapshot = process_snapshot(process.get("pid"), include_children=False) if process.get("pid") else {}
                elapsed = snapshot.get("process_cpu_s")
                resources.append(dict(process=before["process"], process_snapshot=snapshot,
                    cpu_core_seconds=max(0., elapsed - before["process_cpu_seconds"]) if elapsed is not None else None,
                    basis="/proc counter at cutoff minus process CPU clock at begin; tick-resolution boundary"))
            output["resources_end"] = resources
            output["worker_cpu_core_seconds"] = sum(row["cpu_core_seconds"] for row in resources) if all(row["cpu_core_seconds"] is not None for row in resources) else None
            pipelines.close()
            output["worker_processes_stopped"] = True
            for row in scheduler.rows.values():
                if row["status"] not in ("completed", "rejected"):
                    row.update(status="unfinished_at_cutoff", terminal_ns=end)
        sampler.stop()
        rows = list(scheduler.rows.values())
        rows += [dict(row, status="not_arrived", scheduled_arrival_ns=None if isolated else origin + round(row["arrival_ms"] * 1e6)) for row in pending]
        output.update(end_ns=end, requests=_decorate_rows(rows, origin), summary=summarize_requests(rows, origin, end, observation.get("arrival_window_ms")),
                      controller_cpu_core_seconds=(controller_cpu_end - cpu_origin) / 1e9,
                      nvml=sampler.summary(origin, end), nvml_samples=sampler.samples,
                      sampling_interval_seconds=configuration["sample_interval_seconds"])
        output["memory_margin_gate"] = memory_margin_gate(sampler.samples, sampled=True, origin_ns=origin, end_ns=end)
        output["performance_evidence"] = output["status"] == "completed" and output["memory_margin_gate"]["eligible"]
        if output["status"] == "completed" and not output["memory_margin_gate"]["eligible"]:
            output["status"] = "invalid_memory_margin"
        if output["status"] in ("completed", "invalid_memory_margin") and not cutoff:
            output["resources_end"] = pipelines.control("end", observation["id"])
            output["worker_cpu_core_seconds"] = sum(row["cpu_core_seconds"] for row in output["resources_end"])
        if checkpoint is not None:
            checkpoint(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-videos", action="store_true", help="Retain raw video arrays for manual quality review")
    args = parser.parse_args(argv)
    raw_configuration = json.loads(args.manifest.read_text())
    if args.save_videos:
        raw_configuration["save_videos"] = True
    configuration = validate_manifest(raw_configuration)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("refusing to overwrite a serving attempt: " + str(args.out))
    available = os.sched_getaffinity(0)
    requested = {configuration["controller_cpu"]} | set(sum(configuration["worker_cpu_sets"], []))
    if not requested <= available:
        raise ValueError("requested experiment CPUs are outside the controller's available CPU set")
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(schema=SCHEMA, kind=configuration["kind"], status="running", configuration=configuration,
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  started_unix_ns=time.time_ns(), observations=[], semantic="unverified",
                  timing_scope="warm model; scheduled arrival through output D2H/finite/hash; bounded queues; no pre-arrival encoding")
    target = args.out / "run.json"
    _write(target, result)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("serving interrupted by signal %d" % signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    pipelines = ProcessPipelines(configuration, args.out)
    try:
        # Spawn first, then bind controller; workers bind their explicit sets
        # before torch import and therefore never inherit the controller's core.
        pipelines.start()
        os.sched_setaffinity(0, {configuration["controller_cpu"]})
        result["processes"] = pipelines.ready
        pipelines.warm()
        result["warmup_completed_unix_ns"] = time.time_ns()
        _write(target, result)
        for observation in configuration["observations"]:
            def checkpoint(value):
                existing = next((index for index, row in enumerate(result["observations"]) if row["id"] == value["id"]), None)
                if existing is None:
                    result["observations"].append(copy.deepcopy(value))
                else:
                    result["observations"][existing] = copy.deepcopy(value)
                _write(target, result)
            measured = run_observation(configuration, observation, pipelines, checkpoint)
            if measured["status"] in ("capacity_rejected", "invalid_memory_margin") or measured.get("stopped_at_cutoff"):
                break
        result["status"] = next((row["status"] for row in result["observations"] if row["status"] != "completed"), "completed")
        if len(result["observations"]) < len(configuration["observations"]):
            result["unattempted_observations"] = [row["id"] for row in configuration["observations"][len(result["observations"]):]]
            if result["status"] == "completed":
                result["status"] = "partial"
        result["performance_evidence"] = result["status"] == "completed" and bool(result["observations"]) and all(row.get("performance_evidence", False) for row in result["observations"])
    except BaseException as error:
        result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=traceback.format_exc())
        raise
    finally:
        pipelines.close()
        result["finished_unix_ns"] = time.time_ns()
        _write(target, result)
    return int(result["status"] in ("failed", "interrupted", "partial"))


if __name__ == "__main__":
    raise SystemExit(main())
