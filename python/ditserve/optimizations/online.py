"""Real arrival-clock mixed serving, with explicit host-wall-time attribution.

Importing this module does not import torch or start work. The caller owns GPU
admission, backend qualification, warmup and the complete charged interval.
Host intervals and actual pre-submission device stream markers remain separate;
neither is an inferred GPU kernel start timestamp.
"""

import contextlib
import hashlib
import json
import math
import threading
import time
from pathlib import Path
from unittest.mock import patch

from ..directions.direction1 import (VideoArrivals, _advance_policy, _digest,
                                    _release, _spec, quantile, request_metrics)


DEFAULT_SOURCE = (Path(__file__).resolve().parents[3] /
                  "results/research-directions/run01/direction1/direction1-online.json")
TRACE_SELECTION = ((102, .5, "burst"), (101, .5, "ordinary"))
FROZEN_TRACE_IDS = ("0071c0c25757609a582b4dc6c31e391a4bbad8482653237ad87f15ac7613aed3",
                    "bf91b23fd8377b39f0fdb14377e91751b7967ab57a43ebb1f79d6ae5efbf82c0")
INPUT_FIELDS = ("request_id", "kind", "arrival_ms", "deadline_ms", "service_ms", "seed")
DISPATCH_CLOCK_SCOPE = ("perf_counter host dispatch interval for backend work; may include CPU preparation, "
                        "asynchronous submission or synchronization; not a CUDA kernel timestamp")
STREAM_MARKER_SCOPE = ("CUDA stream marker recorded before the request's backend call; "
                       "not the first model kernel start; device elapsed time must not be subtracted from host arrival time")


class TraceJournal:
    """One trace per frequent snapshot; cumulative writes stay outside traces."""
    def __init__(self, path, write):
        self.path, self.write = Path(path), write
        self.partial_path = self.path.parent / "partial.json"
        self.flush_path = self.path.with_name(self.path.stem + "-flushes.json")
        self.flushes = []

    def _flush_log(self):
        self.write(self.flush_path, dict(schema_version="ditserve.host_persistence.v1",
            flushes=self.flushes, scope="host serialization/atomic-write callback outside trace arrival clock; "
            "sidecar writes are excluded to avoid recursive timing; no fsync durability claim"))

    def _timed_write(self, path, value, kind):
        started = time.perf_counter()
        try:
            self.write(path, value)
        except BaseException as error:
            ended = time.perf_counter()
            self.flushes.append(dict(target=str(path), kind=kind, started_perf_counter_s=started,
                finished_perf_counter_s=ended, host_ms=(ended-started)*1000, status="failed",
                error=type(error).__name__ + ": " + str(error)))
            try:
                self._flush_log()
            except BaseException:
                pass  # Preserve the original failed write as the experiment error.
            raise
        ended = time.perf_counter()
        self.flushes.append(dict(target=str(path), kind=kind, started_perf_counter_s=started,
            finished_perf_counter_s=ended, host_ms=(ended-started)*1000, status="completed"))
        self._flush_log()

    def save_record(self, record):
        record.update(active_trace_path=str(self.partial_path), flush_log_path=str(self.flush_path),
                      persistence_scope="current trace partial only during serving; cumulative rows written between traces")
        self._timed_write(self.path, record, "cumulative_record")

    def start_trace(self, identity):
        identity = dict(identity)
        self._timed_write(self.partial_path,
            dict(schema_version="ditserve.optimizations.online.v1", status="preparing", complete=False,
                 requests=[], events=[], **identity), "prepare_partial")

        def partial(value):
            # run_trace measures this call. No historical rows or cumulative
            # file serialization can enter the request-serving critical path.
            self.write(self.partial_path, dict(value, **identity))
        def terminal_flush(event, origin):
            # A terminal snapshot cannot contain its own completed write event.
            # Keep that one record in the small out-of-trace sidecar instead.
            self.flushes.append(dict(target=str(self.partial_path), kind="terminal_partial",
                started_perf_counter_s=origin+event["start_ms"]/1000,
                finished_perf_counter_s=origin+event["end_ms"]/1000,
                host_ms=event["end_ms"]-event["start_ms"], status=event["status"],
                report_status=event["report_status"], trace_identity=identity))
            self._flush_log()
        partial.record_terminal_flush = terminal_flush
        return partial


class A100StreamMarkers:
    """Bounded device events; only origin initialization may synchronize.

    The CUDA interface comes from the already-admitted worker. Importing this
    module and running fake CPU fixtures never imports torch or probes devices.
    """
    def __init__(self, cuda, capacity):
        self.cuda, self.capacity, self.records = cuda, capacity, []
        self.pool, self.anchor = [], None
        self.setup_ms = 0.
        if cuda is None:
            return
        started = time.perf_counter()
        stream = cuda.current_stream("cuda:0")
        self.pool = [cuda.Event(enable_timing=True) for _ in range(capacity)]
        # Event creation is lazy. Initialize every handle before the exogenous
        # arrival clock; priming records are never exposed as request evidence.
        for event in self.pool:
            event.record(stream)
        self.origin = cuda.Event(enable_timing=True)
        before = time.perf_counter()
        self.origin.record(stream)
        self.origin.synchronize()
        after = time.perf_counter()
        self.anchor = dict(record_before_perf_counter_s=before,
                           synchronize_after_perf_counter_s=after,
                           host_bracket_width_ms=(after-before)*1000,
                           stream_id=int(stream.cuda_stream))
        self.setup_ms = (after-started)*1000

    def record(self, stage, request_ids, host_origin):
        if self.cuda is None:
            return None
        if len(self.records) >= self.capacity:
            raise RuntimeError("preallocated stream marker pool exhausted")
        event = self.pool[len(self.records)]
        value = dict(marker_id=len(self.records), stage=stage, request_ids=list(request_ids),
                     device="cuda:0", status="pending", device_origin_elapsed_ms=None,
                     kernel_timestamp_status="unverified", scope=STREAM_MARKER_SCOPE)
        value["host_record_start_ms"] = (time.perf_counter()-host_origin)*1000
        try:
            stream = self.cuda.current_stream("cuda:0")
            value["stream_id"] = int(stream.cuda_stream)
            event.record(stream)
        except Exception as error:
            value.update(status="error", error=type(error).__name__ + ": " + str(error))
        value["host_record_end_ms"] = (time.perf_counter()-host_origin)*1000
        self.records.append((event, value))
        return dict(value)

    def resolve(self):
        """Query only: completed backend calls already supplied synchronization."""
        for event, value in self.records:
            if value["status"] != "pending":
                continue
            try:
                if event.query():
                    elapsed = float(self.origin.elapsed_time(event))
                    if not math.isfinite(elapsed) or elapsed < 0:
                        raise ValueError("invalid CUDA event elapsed time")
                    value.update(status="complete", device_origin_elapsed_ms=elapsed)
            except Exception as error:
                value.update(status="error", error=type(error).__name__ + ": " + str(error))

    def snapshot(self, marker):
        return None if marker is None else dict(self.records[marker["marker_id"]][1])

    def evidence(self, host_origin):
        anchor = None if self.anchor is None else dict(self.anchor,
            record_before_trace_ms=(self.anchor["record_before_perf_counter_s"]-host_origin)*1000,
            synchronize_after_trace_ms=(self.anchor["synchronize_after_perf_counter_s"]-host_origin)*1000)
        return dict(schema_version="ditserve.optimizations.cuda_stream_markers.v1",
                    status="enabled" if self.cuda is not None else "unavailable_no_worker_cuda_interface",
                    device="cuda:0", scope=STREAM_MARKER_SCOPE, host_anchor_bracket=anchor,
                    pre_trace_setup_ms=self.setup_ms, primed_event_count=len(self.pool),
                    origin_synchronizations=1 if self.cuda is not None else 0,
                    in_trace_added_synchronizations=0, recorded_count=len(self.records),
                    complete_count=sum(v["status"] == "complete" for _, v in self.records),
                    pending_count=sum(v["status"] == "pending" for _, v in self.records),
                    error_count=sum(v["status"] == "error" for _, v in self.records),
                    clock_alignment="host anchor bracket retained; no cross-clock arrival latency computed",
                    resolution="nonblocking query at final/exception report; unfinished events remain pending")


def request_identity(requests):
    """Historical D1 digest, including its original pending-state placeholders."""
    return _digest([dict({k: row[k] for k in INPUT_FIELDS}, status="pending",
                         start_ms=None, finish_ms=None) for row in requests])


def _fresh_requests(requests):
    result, ids = [], set()
    for item in requests:
        row = {key: item[key] for key in INPUT_FIELDS}
        if row["request_id"] in ids or row["kind"] not in ("image", "video"):
            raise ValueError("duplicate request id or invalid modality")
        ids.add(row["request_id"])
        for key in ("arrival_ms", "deadline_ms", "service_ms"):
            if not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("invalid frozen request timing")
        if row["deadline_ms"] < row["arrival_ms"]:
            raise ValueError("deadline precedes arrival")
        row.update(status="pending", arrived=False, start_ms=None, finish_ms=None)
        result.append(row)
    if not result:
        raise ValueError("empty request trace")
    return result


def load_frozen_traces(path=None, selection=TRACE_SELECTION):
    """Read complete saved K4 rows, preserving all six original input fields.

    The old phase may have exhausted its budget after these rows completed. Its
    phase status is therefore not used to discard complete selected traces.
    """
    path = Path(path or DEFAULT_SOURCE).resolve()
    raw = path.read_bytes()
    document = json.loads(raw)
    def request_rows(row):
        return row.get("requests", row.get("rows", []))
    result = []
    for seed, load, pattern in selection:
        matches = [row for row in document["rows"]
                   if (row.get("seed"), row.get("load"), row.get("pattern")) == (seed, load, pattern)]
        candidates = [row for row in matches if row.get("policy") == "block-4"
                      and request_rows(row)
                      and all(r.get("status") == "completed" for r in request_rows(row))]
        if len(candidates) != 1:
            raise ValueError("expected one complete historical K4 trace: " + str((seed, load, pattern)))
        chosen = candidates[0]
        requests = _fresh_requests(request_rows(chosen))
        for row in requests:
            row.pop("arrived")
        identity = request_identity(requests)
        if chosen.get("trace_sha256") != identity:
            raise ValueError("historical trace input digest does not match saved hash")
        for other in matches:
            if request_rows(other) and request_identity(request_rows(other)) != identity:
                raise ValueError("historical policies disagree about request identity")
        result.append(dict(seed=seed, load=load, pattern=pattern, trace_sha256=identity,
                           requests=requests, source_path=str(path),
                           source_sha256=hashlib.sha256(raw).hexdigest(), historical_policy="block-4"))
    return result


class EventLog:
    def __init__(self, origin):
        self.origin, self.events, self.lock = origin, [], threading.Lock()

    def now(self):
        return (time.perf_counter() - self.origin) * 1000

    def emit(self, kind, start_ms=None, end_ms=None, **fields):
        if "start_s" in fields:
            start_ms = (fields.pop("start_s") - self.origin) * 1000
        if "end_s" in fields:
            end_ms = (fields.pop("end_s") - self.origin) * 1000
        start_ms = self.now() if start_ms is None else start_ms
        end_ms = start_ms if end_ms is None else end_ms
        if end_ms < start_ms:
            raise ValueError("event end precedes start")
        event = dict(kind=kind, start_ms=start_ms, end_ms=end_ms, **fields)
        with self.lock:
            if (kind == "scheduler_idle" and self.events and self.events[-1]["kind"] == kind
                    and self.events[-1].get("reason") == fields.get("reason")):
                # No service event intervened: include scheduler polling in the
                # same actual idle interval instead of writing thousands of polls.
                self.events[-1]["end_ms"] = end_ms
                self.events[-1]["polls"] = self.events[-1].get("polls", 1) + 1
                return self.events[-1]
            event["event_index"] = len(self.events)
            self.events.append(event)
        return event

    def snapshot(self):
        with self.lock:
            return [dict(event) for event in self.events]

    @contextlib.contextmanager
    def span(self, kind, **fields):
        start = self.now()
        try:
            yield
        except BaseException as error:
            self.emit(kind, start, self.now(), status="failed", error=type(error).__name__, **fields)
            raise
        else:
            self.emit(kind, start, self.now(), status="completed", **fields)


class MeasuredVideoArrivals(VideoArrivals):
    """D1's separate exogenous clock; the job handle never reads future prompts."""
    def __init__(self, jobs, encode, origin, log, encode_device="cpu"):
        self.log, self.encode_device = log, encode_device
        super().__init__(jobs, encode, origin)

    def _arrive(self):
        try:
            for rid, arrival_ms, handle in sorted(self._jobs, key=lambda job: (job[1], job[0])):
                remaining = self._origin + arrival_ms / 1000 - time.perf_counter()
                if self._stop.wait(max(0., remaining)):
                    return
                submitted = self.log.now()
                self.log.emit("encode_queued", submitted, request_id=rid, scheduled_arrival_ms=arrival_ms,
                              resource=self.encode_device, compute_device=self.encode_device,
                              queue_actor="arrival_thread", executor_actor="encoder_thread")
                future = self._pool.submit(self._encode, handle)
                with self._lock:
                    self._tickets[rid] = (handle, future, submitted)
        except BaseException as error:
            with self._lock:
                self._error = error

    def close(self):
        self._stop.set()
        self._thread.join()
        with self._lock:
            for _, future, _ in self._tickets.values():
                future.cancel()
        self._pool.shutdown(wait=True)


def instrumented_finish(worker, state, emit):
    """Call the original finish exactly, observing its object-local API calls.

    decode and convert are host-call spans; conversion can include pending GPU
    work. Explicit synchronization spans are recorded separately. No extra
    synchronization is inserted and no span is claimed to be kernel execution.
    """
    def wrapper(kind, fn):
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                emit(kind, start_s=start, end_s=time.perf_counter(), request_id=state.request_id,
                     clock_scope="host API wall time; may include asynchronous work or synchronization")
        return call
    with contextlib.ExitStack() as stack:
        for owner, name, kind in ((worker, "_sync", "video_sync"),
                                  (worker.pipe.vae, "decode", "video_decode"),
                                  (worker.pipe.video_processor, "postprocess_video", "video_convert")):
            stack.enter_context(patch.object(owner, name, wrapper(kind, getattr(owner, name))))
        return worker.finish(state)


def ready_batch(rows, ready_ids, now_ms, size=1):
    if size not in (1, 2):
        raise ValueError("only single-image or batch-two serving is qualified")
    arrived = sorted((row for row in rows if row["status"] == "pending" and row["arrival_ms"] <= now_ms),
                     key=lambda row: (row["deadline_ms"], row["arrival_ms"], row["request_id"]))
    ordered = [row for row in arrived if row["request_id"] in ready_ids]
    if not ordered:
        return []
    # Single-request work can run while an earlier video awaits encoding. A
    # batch has the stricter contract: the complete arrived EDF head must be
    # precisely these two ready images, without jumping an unready video.
    count = 2 if (size == 2 and len(ordered) >= 2 and
                  [r["request_id"] for r in arrived[:2]] == [r["request_id"] for r in ordered[:2]] and
                  all(r["kind"] == "image" for r in ordered[:2])) else 1
    return ordered[:count]


def _overlap(intervals, start, end):
    clipped = sorted((max(start, a), min(end, b)) for a, b in intervals if b > start and a < end)
    total, previous = 0., start
    for a, b in clipped:
        total += max(0., b - max(a, previous))
        previous = max(previous, b)
    return total


def attribution(rows, events, elapsed_ms):
    idle = sum(e["end_ms"] - e["start_ms"] for e in events
               if e["kind"] == "scheduler_idle" and e.get("reason") == "waiting_encoding")
    videos = [r for r in rows if r["kind"] == "video" and r.get("status") == "completed"]
    encode = sum(e["end_ms"] - e["start_ms"] for e in events
                 if e["kind"] == "encode_service" and e.get("status") == "completed"
                 and e.get("request_id") in {r["request_id"] for r in videos})
    video_e2e = sum(r["finish_ms"] - r["arrival_ms"] for r in videos)
    idle_fraction = idle / elapsed_ms if elapsed_ms else None
    encode_fraction = encode / video_e2e if video_e2e else None
    images = sorted((r for r in rows if r["kind"] == "image" and r.get("start_ms") is not None),
                    key=lambda r: (-(r["start_ms"] - r["arrival_ms"]), r["request_id"]))
    tail = images[:max(1, math.ceil(.05 * len(images)))] if images else []
    finish = [(e["start_ms"], e["end_ms"]) for e in events if e["kind"] == "video_finish"]
    tail_rows = [dict(request_id=r["request_id"], wait_ms=r["start_ms"] - r["arrival_ms"],
                      finish_overlap_ms=_overlap(finish, r["arrival_ms"], r["start_ms"])) for r in tail]
    wait = sum(r["wait_ms"] for r in tail_rows)
    overlap = sum(r["finish_overlap_ms"] for r in tail_rows)
    fraction = overlap / wait if wait else 0.
    return dict(encoding_idle_ms=idle, encoding_idle_fraction=idle_fraction,
                encoding_service_ms=encode, video_e2e_sum_ms=video_e2e,
                encoding_video_e2e_fraction=encode_fraction, video_sample_count=len(videos),
                encoding_trigger=bool((idle_fraction is not None and idle_fraction >= .05) or
                                      (encode_fraction is not None and encode_fraction >= .20)),
                tail_image_count=len(tail_rows), started_image_count=len(images), tail_images=tail_rows,
                tail_wait_ms=wait, tail_finish_overlap_ms=overlap, tail_finish_overlap_fraction=fraction,
                vae_trigger=bool(tail_rows and fraction >= .20),
                scope="observed host intervals; encoding fraction uses encode service, excluding its CPU queue; tail is ceil(5% of started images)")


def run_trace(context, requests, backend_options=None):
    """Serve a frozen trace; hooks only execute after the request has arrived.

    Hooks: encode_video(spec), prepare_video(spec,payload),
    finish_video(state,emit), generate_images(request_batch,emit). Image hook
    rows include arrived prompts; it returns (images, service). emit accepts
    trace-relative start_ms/end_ms or absolute perf_counter start_s/end_s.
    encode_on_scheduler=True serializes a GPU encoder through the same EDF loop.
    encode_device describes computation separately from the thread dispatching it.
    Host dispatch and pre-submission CUDA stream markers are separate evidence;
    neither claims the exact first model kernel start.
    """
    options = dict(backend_options or {})
    rows = _fresh_requests(requests)
    original = {row["request_id"]: row for row in requests}
    worker, executor, sana = context.worker, context.executor, context.sana
    policy, batch_size = options.get("policy", "block-4"), options.get("image_batch_size", 1)
    if policy not in ("step", "cfg", "block-1", "block-4", "block-8") or batch_size not in (1, 2):
        raise ValueError("unsupported online policy or batch size")
    synchronous_encode = bool(options.get("encode_on_scheduler", False))
    encode_device = options.get("encode_device", "cpu")
    encode_actor = "scheduler" if synchronous_encode else "encoder_thread"
    for row in rows:
        row.update(first_a100_dispatch=None, first_dit_quantum_dispatch=None,
                   first_a100_stream_marker=None, first_dit_stream_marker=None,
                   kernel_timestamp_status="unverified", dispatch_clock_scope=DISPATCH_CLOCK_SCOPE)
        if row["kind"] == "video":
            row.update(encode_device=encode_device, encode_executor_actor=encode_actor)
    markers = A100StreamMarkers(getattr(getattr(worker, "torch", None), "cuda", None),
                               len(rows) + sum(r["kind"] == "video" for r in rows))
    origin = time.perf_counter()
    log = EventLog(origin)
    states, continuations, encoded = {}, {}, {}
    last_saved = origin
    arrivals = None

    @contextlib.contextmanager
    def work_span(kind, members, compute_device="cuda:0", executor_actor="scheduler", dit=False, **fields):
        """Observe dispatch and its stream marker without adding synchronization."""
        started = log.now()
        first = []
        for member in members:
            names = []
            if compute_device == "cuda:0" and member["first_a100_dispatch"] is None:
                names.append("first_a100_dispatch")
            if dit and member["first_dit_quantum_dispatch"] is None:
                names.append("first_dit_quantum_dispatch")
            for name in names:
                member[name] = dict(stage=kind, start_ms=started, end_ms=None, status="dispatched",
                                    compute_device=compute_device, executor_actor=executor_actor,
                                    clock_scope=DISPATCH_CLOCK_SCOPE, kernel_timestamp_status="unverified")
                first.append((member, name))
        if first and compute_device == "cuda:0":
            marker = markers.record(kind, [member["request_id"] for member in members], origin)
            for member, name in first:
                field = "first_a100_stream_marker" if name == "first_a100_dispatch" else "first_dit_stream_marker"
                member[field] = marker
        status, failure = "completed", {}
        try:
            yield
        except BaseException as error:
            status, failure = "failed", {"error": type(error).__name__}
            raise
        finally:
            ended = log.now()
            for member, name in first:
                # Replace the value so already-persisted partial snapshots do not mutate.
                member[name] = dict(member[name], end_ms=ended, status=status, **failure)
            log.emit(kind, started, ended, status=status, resource=compute_device,
                     compute_device=compute_device, executor_actor=executor_actor,
                     first_a100_dispatch=any(name == "first_a100_dispatch" for _, name in first),
                     first_dit_quantum_dispatch=any(name == "first_dit_quantum_dispatch" for _, name in first),
                     clock_scope=DISPATCH_CLOCK_SCOPE, kernel_timestamp_status="unverified", **failure, **fields)

    def prompt(row):
        # Access payload only at encode/dispatch, never when constructing jobs.
        source = original[row["request_id"]]
        if "prompt" in source:
            return source["prompt"]
        index = int(row["request_id"][1:])
        return context.opts["prompts"][index % len(context.opts["prompts"])]

    def encode(row):
        rid = row["request_id"]
        started = log.now()
        log.emit("encode_start", started, request_id=rid, resource=encode_device,
                 compute_device=encode_device, executor_actor=encode_actor)
        shape = context.profile.get("selected_shape", context.profile.get("shape"))
        if shape is None:
            raise ValueError("online context.profile requires selected_shape")
        spec = _spec(context.opts, shape, rid, prompt=prompt(row), seed=row["seed"])
        with work_span("encode_service", [row], compute_device=encode_device,
                       executor_actor=encode_actor, request_id=rid):
            payload = options.get("encode_video", worker.encode_prompt)(spec)
        log.emit("encode_ready", request_id=rid, resource=encode_device,
                 compute_device=encode_device, executor_actor=encode_actor)
        return spec, payload

    def update_arrived():
        now = log.now()
        for row in rows:
            row["arrived"] = row["arrival_ms"] <= now
        return now

    def report(status):
        elapsed = update_arrived()
        if status != "running":
            markers.resolve()
        request_rows = []
        for row in rows:
            item = dict(row)
            for field in ("first_a100_stream_marker", "first_dit_stream_marker"):
                item[field] = markers.snapshot(row[field])
            request_rows.append(item)
        events = log.snapshot()
        return dict(schema_version="ditserve.optimizations.online.v1", status=status,
                    policy=policy, image_batch_size=batch_size, encode_on_scheduler=synchronous_encode,
                    encode_device=encode_device,
                    trace_sha256=request_identity(rows), requests=request_rows,
                    elapsed_ms=elapsed, metrics=request_metrics(rows, elapsed),
                    image_metrics=request_metrics([r for r in rows if r["kind"] == "image"], elapsed),
                    video_metrics=request_metrics([r for r in rows if r["kind"] == "video"], elapsed),
                    events=events, attribution=attribution(rows, events, elapsed),
                    scheduled_count=len(rows), complete=all(r["status"] == "completed" for r in rows),
                    clock_scope="perf_counter host wall clock, independent arrival thread; GPU APIs synchronized by backends",
                    stream_marker_evidence=markers.evidence(origin),
                    dispatch_evidence=dict(clock_scope=DISPATCH_CLOCK_SCOPE, kernel_timestamp_status="unverified",
                        first_a100_dispatch_count=sum(r["first_a100_dispatch"] is not None for r in rows),
                        first_dit_quantum_dispatch_count=sum(r["first_dit_quantum_dispatch"] is not None for r in rows),
                        image_dit_quantum_scope="not instrumented; only image_service host dispatch is observed"))

    def save(status, force=False):
        nonlocal last_saved
        now = time.perf_counter()
        if force or now - last_saved >= 1.:
            value = build_report(status)
            persist_partial(value)
            last_saved = now

    def build_report(status):
        with log.span("partial_snapshot_build", resource="cpu", executor_actor="scheduler",
                      clock_scope="host snapshot construction; not GPU execution"):
            return report(status)

    def persist_partial(value):
        # The event is appended AFTER this snapshot has been written. A later
        # snapshot or the final cumulative row carries it; never write again
        # merely to include the write's own timing.
        with log.span("partial_flush", resource="cpu", executor_actor="scheduler",
                      report_status=value["status"],
                      clock_scope="host serialization/atomic-write callback; no fsync durability claim",
                      inside_trace_interval=value["status"] == "running"):
            context.save_partial(value)
        terminal_hook = getattr(context.save_partial, "record_terminal_flush", None)
        if value["status"] != "running" and terminal_hook is not None:
            latest = next(event for event in reversed(log.snapshot()) if event["kind"] == "partial_flush")
            terminal_hook(latest, origin)

    def attach_persistence(value):
        value["events"] = log.snapshot()
        flushes = [event for event in value["events"] if event["kind"] == "partial_flush"]
        value["persistence"] = dict(partial_flush_count=len(flushes),
            partial_flush_host_ms=sum(event["end_ms"]-event["start_ms"] for event in flushes),
            scope="host observer cost; final flush follows elapsed_ms cutoff but remains GPU-allocation charged; "
                  "current flush appears only in a later snapshot or final cumulative row")
        return value

    def release(rid):
        if rid in states:
            _release(worker, executor, states[rid], continuations.get(rid))
            states.pop(rid, None)
            continuations.pop(rid, None)

    try:
        if not synchronous_encode:
            arrivals = MeasuredVideoArrivals([(r["request_id"], r["arrival_ms"], r)
                                             for r in rows if r["kind"] == "video"], encode, origin, log,
                                             encode_device=encode_device)
        while any(row["status"] == "pending" for row in rows):
            context.check_budget()
            now = update_arrived()
            tickets = arrivals.snapshot() if arrivals is not None else {}
            ready = {r["request_id"] for r in rows if r["kind"] == "image" or synchronous_encode
                     or r["request_id"] in states or
                     (r["request_id"] in tickets and tickets[r["request_id"]][1].done())}
            batch = ready_batch(rows, ready, now, batch_size)
            if not batch:
                waiting_encode = any(r["arrived"] and r["status"] == "pending" and r["kind"] == "video" for r in rows)
                future = [r["arrival_ms"] for r in rows if not r["arrived"] and r["status"] == "pending"]
                delay = min(.002, max(0., (min(future) - now) / 1000)) if future else .002
                with log.span("scheduler_idle", reason="waiting_encoding" if waiting_encode else "future_arrival"):
                    time.sleep(delay)
                save("running")
                continue
            row, rid = batch[0], batch[0]["request_id"]
            try:
                if row["kind"] == "image":
                    started = log.now()
                    for member in batch:
                        member["start_ms"] = started
                    prompt_rows = [dict(member, prompt=prompt(member)) for member in batch]
                    with work_span("image_service", batch, request_ids=[r["request_id"] for r in batch], batch_size=len(batch)):
                        if "generate_images" in options:
                            array, service = options["generate_images"](prompt_rows, log.emit)
                        elif len(batch) == 2:
                            array, service = sana.generate_batch([r["prompt"] for r in prompt_rows],
                                seeds=[r["seed"] for r in batch], check_budget=context.check_budget)
                        else:
                            array, service = sana.generate(prompt_rows[0]["prompt"], seed=row["seed"],
                                                           check_budget=context.check_budget)
                    finish = log.now()
                    if len(array) != len(batch) or service.get("finite") is False:
                        raise RuntimeError("image batch size/finite validation failed")
                    for member in batch:
                        member.update(status="completed", finish_ms=finish, service=service,
                                      batch_request_ids=[r["request_id"] for r in batch])
                    service_origin = service.get("origin_perf_counter_s")
                    if service_origin is not None:
                        stages = service.get("stages", [])
                        if isinstance(stages, dict):
                            stages = [dict(value, stage=key) for key, value in stages.items() if isinstance(value, dict)]
                        for stage in stages:
                            if "start_ms" in stage and "end_ms" in stage:
                                fields = {k: v for k, v in stage.items() if k not in ("start_ms", "end_ms", "kind")}
                                log.emit("image_stage", start_s=service_origin + stage["start_ms"] / 1000,
                                         end_s=service_origin + stage["end_ms"] / 1000,
                                         request_ids=[r["request_id"] for r in batch], **fields)
                    del array
                    save("running", True)
                else:
                    if rid not in states:
                        if synchronous_encode and rid not in encoded:
                            row["start_ms"] = log.now()
                            log.emit("encode_queued", row["start_ms"], request_id=rid,
                                     scheduled_arrival_ms=row["arrival_ms"], resource=encode_device,
                                     compute_device=encode_device, queue_actor="scheduler", executor_actor=encode_actor)
                            encoded[rid] = encode(row)
                            save("running")
                            continue
                        elif synchronous_encode:
                            spec, payload = encoded.pop(rid)
                        else:
                            spec, future, submitted = tickets[rid]
                            spec, payload = future.result()
                            row["encode_submit_ms"] = submitted
                            row["start_ms"] = log.now()
                        row["cpu_prompt_encode_ms"] = getattr(payload, "cpu_prompt_encode_ms", None)
                        with work_span("video_prepare", [row], request_id=rid):
                            states[rid] = options.get("prepare_video", worker.prepare_encoded)(spec, payload)
                            continuations[rid] = executor.attach(states[rid])
                        # Reschedule after each indivisible phase, including encoding/preparation.
                        continue
                    state = states[rid]
                    if state.step_index < state.spec["steps"]:
                        with work_span("video_quantum", [row], dit=True, request_id=rid,
                                       policy=policy, step_index=state.step_index):
                            _advance_policy(executor, continuations[rid], policy, context.check_budget)
                    else:
                        context.check_budget()
                        with log.span("video_finish", request_id=rid):
                            array = options.get("finish_video", lambda state, emit: instrumented_finish(worker, state, emit))(state, log.emit)
                        row.update(status="completed", finish_ms=log.now())
                        del array
                        release(rid)
                        save("running", True)
                save("running")
            except BaseException as error:
                for member in batch:
                    member.update(status="failed", error=type(error).__name__ + ": " + str(error))
                raise
        if arrivals is not None:
            arrivals.close()
            arrivals = None
        result = build_report("completed")
        persist_partial(result)
        return attach_persistence(result)
    except BaseException as error:
        update_arrived()
        for row in rows:
            if row["arrived"] and row["status"] == "pending":
                row.update(status="interrupted", error=type(error).__name__)
        partial = build_report("interrupted")
        partial["error"] = type(error).__name__ + ": " + str(error)
        persist_partial(partial)
        raise
    finally:
        if arrivals is not None:
            arrivals.close()
        for rid in list(states):
            release(rid)


def paired_gate(rows, expected_trace_ids=FROZEN_TRACE_IDS):
    """AB/BA screening with complete identity-matched repeats; no survivor gate."""
    failures, traces = [], {}
    for row in rows:
        trace = row.get("trace_sha256")
        arm, order = row.get("arm"), row.get("order_index")
        if not trace or arm not in ("baseline", "candidate") or order not in (0, 1):
            failures.append("invalid trace/arm/order identity")
            continue
        key = (order, arm)
        group = traces.setdefault(trace, {})
        if key in group:
            failures.append("duplicate AB/BA member")
        group[key] = row
        requests = row.get("requests", [])
        if not requests or request_identity(requests) != trace:
            failures.append("request identity differs from trace hash")
        if row.get("status") != "completed" or not row.get("complete") or any(
                r.get("status") != "completed" or not r.get("arrived", True) for r in requests):
            failures.append("failed, incomplete or unarrived requests in paired trace")

    expected = {(order, arm) for order in (0, 1) for arm in ("baseline", "candidate")}

    def aggregate(members):
        requests = [r for member in members for r in member["requests"]]
        elapsed = sum(member["elapsed_ms"] for member in members)
        return dict(metrics=request_metrics(requests, elapsed),
                    images=request_metrics([r for r in requests if r["kind"] == "image"], elapsed),
                    videos=request_metrics([r for r in requests if r["kind"] == "video"], elapsed))

    def comparison(baseline, candidate):
        bg, cg = baseline["metrics"]["goodput_rps"], candidate["metrics"]["goodput_rps"]
        bl, cl = baseline["images"]["latency_p95_ms"], candidate["images"]["latency_p95_ms"]
        bv, cv = baseline["videos"]["on_time_fraction"], candidate["videos"]["on_time_fraction"]
        valid = all(value is not None and math.isfinite(value) for value in (bg, cg, bl, cl, bv, cv))
        goodput_gain = (cg - bg) / bg if valid and bg > 0 else None
        latency_gain = (bl - cl) / bl if valid and bl > 0 else None
        safe = valid and bg > 0 and bl > 0 and cg >= bg * .95 and cl <= bl * 1.05 and cv >= bv - .05
        return dict(baseline=baseline, candidate=candidate, goodput_gain=goodput_gain,
                    image_p95_reduction=latency_gain, video_ontime_drop=bv - cv if valid else None,
                    safe=bool(safe), system_gain=bool(valid and (
                        (goodput_gain is not None and goodput_gain >= .05) or
                        (latency_gain is not None and latency_gain >= .10))))

    trace_reports = []
    for trace, group in traces.items():
        if set(group) != expected:
            failures.append("missing complete AB/BA for trace " + trace)
            continue
        value = comparison(aggregate([group[(o, "baseline")] for o in (0, 1)]),
                           aggregate([group[(o, "candidate")] for o in (0, 1)]))
        trace_reports.append(dict(trace_sha256=trace, **value))
    if not traces:
        failures.append("no paired trace evidence")
    if set(traces) != set(expected_trace_ids):
        failures.append("paired traces do not match the complete frozen trace set")
    pooled = None
    if not failures and len(trace_reports) == len(traces):
        pooled = comparison(aggregate([r for r in rows if r["arm"] == "baseline"]),
                            aggregate([r for r in rows if r["arm"] == "candidate"]))
    passed = not failures and all(r["safe"] for r in trace_reports) and bool(pooled and pooled["system_gain"])
    return dict(passed=bool(passed), eligible=bool(passed), failures=failures,
                traces=trace_reports, pooled=pooled, sample_count=len(rows),
                thresholds=dict(minimum_goodput_gain=.05, minimum_image_p95_reduction=.10,
                                maximum_goodput_loss=.05, maximum_image_p95_increase=.05,
                                maximum_video_ontime_drop=.05),
                scope="finite AB/BA descriptive screening; quality, recovery and capacity must qualify separately")
