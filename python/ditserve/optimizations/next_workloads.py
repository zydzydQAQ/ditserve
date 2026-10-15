"""Frozen follow-up workloads and bounded diagnostic/cache replay entry points.

No model import or work occurs on import. The caller owns GPU admission, model
residency, warmup, quality qualification and the charged process lifetime.
"""
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics

from ..directions.block_executor import WanBlockExecutor
from ..directions.direction1 import frozen_requests, quantile, request_metrics
from ..research.common import atomic_json
from .cache_online import cache_spec
from .online import (FROZEN_TRACE_IDS, INPUT_FIELDS, TraceJournal,
                     load_frozen_traces, paired_gate, request_identity, run_trace)


SEEDS = (101, 102, 103)
PATTERNS = ("ordinary", "burst")
LOADS = (.5, .7, .9)
COUNT = 24


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _inputs(requests):
    return [{key: row[key] for key in INPUT_FIELDS} for row in requests]


def _fresh(requests):
    return [dict(row, status="pending", start_ms=None, finish_ms=None)
            for row in _inputs(requests)]


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _validate_trace(trace):
    requests = trace["requests"]
    if not requests or request_identity(requests) != trace["trace_sha256"]:
        raise ValueError("trace request identity mismatch")
    if len({r["request_id"] for r in requests}) != len(requests):
        raise ValueError("duplicate request identity")
    for row in requests:
        if row["kind"] not in ("image", "video") or not _positive(row["service_ms"]):
            raise ValueError("invalid request modality/service calibration")
        if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
            raise ValueError("invalid request seed")
        if not all(isinstance(row[k], (int, float)) and not isinstance(row[k], bool) and
                   math.isfinite(row[k]) and row[k] >= 0 for k in ("arrival_ms", "deadline_ms")):
            raise ValueError("invalid frozen request timing")
        if row["deadline_ms"] < row["arrival_ms"]:
            raise ValueError("deadline precedes arrival")


def build_workloads(config):
    """Return two exact historical diagnostic pairs and 18 calibrated traces.

    ``highload['0.5'|'0.7'|'0.9']`` each contains six 24-request traces.
    No current speed, candidate result or new measured service enters this API.
    """
    options = config.get("next_workloads", {})
    expected_matrix = dict(seeds=list(SEEDS), patterns=list(PATTERNS), loads=list(LOADS), count=COUNT)
    for key, expected in expected_matrix.items():
        if key in options and options[key] != expected:
            raise ValueError("follow-up workload matrix differs from frozen " + key)
    path = Path(config["old_online"]).resolve()
    raw = path.read_bytes()
    source_sha = _sha(raw)
    if options.get("expected_source_sha256", source_sha) != source_sha:
        raise ValueError("old online source SHA256 mismatch")
    document = json.loads(raw)
    costs = document.get("frozen_service_ms")
    if not isinstance(costs, dict) or set(costs) != {"video", "image"} or not all(_positive(v) for v in costs.values()):
        raise ValueError("missing/invalid original off-eager service calibration")
    if options.get("expected_calibration", costs) != costs:
        raise ValueError("original service calibration mismatch")
    old = load_frozen_traces(path)
    if tuple(trace["trace_sha256"] for trace in old) != FROZEN_TRACE_IDS:
        raise ValueError("diagnostic traces differ from the two frozen original identities")
    for trace in old:
        if trace["source_sha256"] != source_sha:
            raise ValueError("old online source changed during workload construction")
        _validate_trace(trace)
        rebuilt = frozen_requests(count=COUNT, seed=trace["seed"], load=trace["load"],
            pattern=trace["pattern"], video_ms=costs["video"], image_ms=costs["image"])
        if _inputs(rebuilt) != _inputs(trace["requests"]):
            raise ValueError("historical trace does not match original generator/calibration")
    source = dict(path=str(path), sha256=source_sha, calibration_pointer="/frozen_service_ms")
    generator_path = Path(inspect.getsourcefile(frozen_requests)).resolve()
    generator = dict(module="ditserve.directions.direction1", function="frozen_requests",
        path=str(generator_path), sha256=_sha(generator_path.read_bytes()),
        function_sha256=_sha(inspect.getsource(frozen_requests).encode()),
        deadline_multiplier=2.5, arrival_clock="original milliseconds; no compression or rebasing")
    calibration = dict(frozen_service_ms=dict(costs), source=source,
        reference="original D1 off/eager full-video reference and original standalone eager image profile",
        scope="fixed before all follow-up candidates; never recalibrated from candidate speed")
    diagnosis = []
    for original in old:
        mixed = dict(original, requests=_fresh(original["requests"]), source=dict(source))
        images = _fresh([r for r in original["requests"] if r["kind"] == "image"])
        image_only = dict(mixed, requests=images, trace_sha256=request_identity(images),
            parent_trace_sha256=mixed["trace_sha256"], scope="videos removed; image identities/times/deadlines unchanged")
        diagnosis.append(dict(parent_trace_sha256=mixed["trace_sha256"], seed=mixed["seed"],
            load=mixed["load"], pattern=mixed["pattern"], mixed=mixed, image_only=image_only))
    highload = {}
    for load in LOADS:
        traces = []
        for seed in SEEDS:
            for pattern in PATTERNS:
                requests = frozen_requests(count=COUNT, seed=seed, load=load, pattern=pattern,
                    video_ms=costs["video"], image_ms=costs["image"])
                trace = dict(seed=seed, load=load, pattern=pattern, requests=requests,
                    trace_sha256=request_identity(requests), source=dict(source),
                    calibration=dict(costs), generator_sha256=generator["function_sha256"])
                _validate_trace(trace)
                traces.append(trace)
        highload[str(load)] = traces
    if path.read_bytes() != raw:
        raise ValueError("old online source changed during workload construction")
    return dict(schema_version="ditserve.optimizations.next_workloads.v1", source=source,
                calibration=calibration, generator=generator, matrix=expected_matrix,
                diagnosis=diagnosis, highload=highload,
                scope="frozen exogenous arrivals; all prompt access/encoding remains inside arrival-gated run_trace")


def _stats(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                p95=quantile(values, .95), minimum=min(values) if values else None,
                maximum=max(values) if values else None)


def _image_summary(rows):
    requests = [r for row in rows for r in row["requests"] if r["kind"] == "image"]
    completed = [r for r in requests if r.get("arrived", True) and r.get("status") == "completed"]
    result = request_metrics(requests, sum(row["elapsed_ms"] for row in rows))
    # Image-only removes video work. Even image goodput uses a different window;
    # diagnosis compares image counts and latencies, never overall goodput gates.
    result.pop("goodput_rps", None)
    result["full_service_ms"] = _stats([r["finish_ms"] - r["start_ms"] for r in completed])
    result["wait_ms"] = _stats([r["start_ms"] - r["arrival_ms"] for r in completed])
    stages = {}
    for row in rows:
        for event in row.get("events", []):
            if event.get("kind") != "image_stage":
                continue
            stage = stages.setdefault(event.get("stage", "unknown"), dict(host=[], cuda=[]))
            if "start_ms" in event and "end_ms" in event:
                stage["host"].append(event["end_ms"] - event["start_ms"])
            if isinstance(event.get("cuda_elapsed_ms"), (float, int)):
                stage["cuda"].append(event["cuda_elapsed_ms"])
    result["stages"] = {key: dict(host_ms=_stats(v["host"]), cuda_ms=_stats(v["cuda"])) for key, v in stages.items()}
    result["stage_scope"] = "only recorded stages; host spans can nest/overlap, missing stages are unverified"
    return result


def burst_bounds(row):
    """Offline conditional bounds, not a scheduler or hardware lower bound."""
    grouped = {}
    for request in row["requests"]:
        if request["kind"] == "image":
            grouped.setdefault(request["arrival_ms"], []).append(request)
    result = []
    for arrival, requests in sorted(grouped.items()):
        deadlines = {r["deadline_ms"] for r in requests}
        common = next(iter(deadlines)) if len(deadlines) == 1 else None
        costs_complete = all(r.get("status") == "completed" and r.get("start_ms") is not None and
                             r.get("finish_ms") is not None for r in requests)
        costs = sorted(r["finish_ms"] - r["start_ms"] for r in requests) if costs_complete else []
        feasible = bound = None
        if common is not None and costs_complete:
            total = 0.; bound = 0
            for cost in costs:
                total += cost
                if total <= common - arrival:
                    bound += 1
            feasible = sum(costs) <= common - arrival
        result.append(dict(arrival_ms=arrival, request_ids=[r["request_id"] for r in requests],
            count=len(requests), relative_deadline_ms=common-arrival if common is not None else None,
            required_average_service_ms=(common-arrival)/len(requests) if common is not None else None,
            measured_on_time=sum(r.get("status") == "completed" and r["finish_ms"] <= r["deadline_ms"] for r in requests),
            complete_service_samples=len(costs), serial_service_sum_ms=sum(costs) if costs_complete else None,
            max_completions_given_observed_costs=bound, all_feasible_given_observed_costs=feasible,
            scope="optimistic shortest-service-first with observed costs, zero video/observer/wait; common deadline only; not a hardware impossibility proof"))
    return result


def _fixed_options(context, *, cached=False):
    options = dict(policy="block-4", sana_backend="eager", wan_backend="eager", vae_backend="eager",
                   image_batch_size=1, encode_device="cpu", encode_on_scheduler=False)
    if hasattr(context, "prepare_online_arm"):
        context.prepare_online_arm(options)
    # Do not inherit a previous candidate's compiled VAE/image or GPU encoder.
    options["encode_video"] = context.worker.encode_prompt
    options["prepare_video"] = context.worker.prepare_encoded
    if cached:
        encode, prepare = options["encode_video"], options["prepare_video"]
        options.update(encode_video=lambda spec: encode(cache_spec(spec)),
                       prepare_video=lambda spec, payload: prepare(cache_spec(spec), payload))
    return options


def _empty(context):
    if getattr(context.worker, "_states", {}):
        raise RuntimeError("live request state leaked across replay boundaries")
    if getattr(context.executor, "_states", {}):
        raise RuntimeError("live continuation leaked across replay boundaries")


def _replay(context, record, path, stage_name, trace, order, arm, executor, *, cached=False):
    _validate_trace(trace)
    context.check_budget()
    context.executor = executor
    _empty(context)
    context.sana.set_backend("eager")
    options = _fixed_options(context, cached=cached)
    index = len(record["rows"])
    directory = context.output / "partials" / stage_name / ("replay-%03d" % index)
    directory.mkdir(parents=True, exist_ok=False)
    journal = TraceJournal(directory / "trace.json", atomic_json)
    unit = dict(trace_sha256=trace["trace_sha256"], parent_trace_sha256=trace.get("parent_trace_sha256", trace["trace_sha256"]),
                order_index=order, arm=arm, seed=trace["seed"], load=trace["load"], pattern=trace["pattern"])
    journal.save_record(dict(status="preparing", **unit))
    context.save_partial = journal.start_trace(unit)
    record.update(active_unit=unit, active_trace_path=str(journal.partial_path), flush_log_path=str(journal.flush_path))
    atomic_json(path, record)
    # No cumulative result writes inside the exogenous arrival-clock window.
    with context.sana.steady_context():
        measured = run_trace(context, trace["requests"], options)
    if measured.get("trace_sha256") != trace["trace_sha256"] or request_identity(measured.get("requests", [])) != trace["trace_sha256"]:
        raise ValueError("measured replay request identity mismatch")
    row = dict(measured, **{key:value for key,value in unit.items() if key != "trace_sha256"},
               trace_artifacts=dict(partial_path=str(journal.partial_path), flush_log_path=str(journal.flush_path)))
    record["rows"].append(row)
    record["active_unit"] = None
    atomic_json(path, record)
    _empty(context)
    if row.get("status") != "completed" or not row.get("complete"):
        raise RuntimeError("replay did not complete every scheduled request")
    return row


def _failure(record, path, error):
    record.update(status="incomplete", decision="interrupted_or_failed", eligible=False,
                  evaluation_complete=False, reason=type(error).__name__ + ": " + str(error))
    partial_path = record.get("active_trace_path")
    if record.get("active_unit") is not None and partial_path and Path(partial_path).is_file():
        raw = Path(partial_path).read_bytes()
        partial = json.loads(raw)
        requests = partial.get("requests", [])
        record["incomplete_trace"] = dict(path=partial_path, sha256=_sha(raw), status=partial.get("status"),
            scheduled=len(requests), arrived=sum(bool(r.get("arrived")) for r in requests),
            completed=sum(r.get("status") == "completed" for r in requests),
            scope="all active arrived requests remain in original partial; not appended as a successful replay")
    atomic_json(path, record)


def run_diagnosis(context, workloads):
    """Eight unchanged-clock replays: image-only/mixed and reverse per trace."""
    if not context.metadata.get("wan_warmed"):
        raise ValueError("diagnosis requires fresh co-resident Wan capacity/warmup evidence")
    pairs = workloads["diagnosis"]
    if len(pairs) != 2 or tuple(pair["parent_trace_sha256"] for pair in pairs) != FROZEN_TRACE_IDS:
        raise ValueError("diagnosis requires both original frozen trace pairs")
    for pair in pairs:
        for arm in ("mixed", "image_only"):
            _validate_trace(pair[arm])
        if (pair["mixed"]["trace_sha256"] != pair["parent_trace_sha256"] or
            _inputs([r for r in pair["mixed"]["requests"] if r["kind"] == "image"]) != _inputs(pair["image_only"]["requests"])):
            raise ValueError("diagnosis image subset identity/timing mismatch")
    path = context.output / "diagnosis.json"
    if path.exists():
        raise FileExistsError("diagnosis evidence already exists")
    record = dict(schema_version="ditserve.optimizations.diagnosis.v1", status="running", rows=[],
        decision="pending", reason="", eligible=False, evaluation_complete=False, planned_replays=8,
        planned_requests=184, source=workloads["source"], calibration=workloads["calibration"],
        generator=workloads["generator"], warmup_metadata=copy.deepcopy(context.metadata),
        scope="diagnostic paired image requests only; image-only is not an overall goodput candidate; Wan stays resident")
    atomic_json(path, record)
    original_executor, original_partial = context.executor, getattr(context, "save_partial", None)
    off = getattr(context, "executors", {}).get("cache_factorial_off") or WanBlockExecutor(context.worker)
    try:
        for pair in pairs:
            for order, arms in enumerate((("image_only", "mixed"), ("mixed", "image_only"))):
                for arm in arms:
                    trace = dict(pair[arm], parent_trace_sha256=pair["parent_trace_sha256"])
                    row = _replay(context, record, path, "diagnosis", trace, order, arm, off)
                    row["burst_bounds"] = burst_bounds(row)
        summaries = []
        for pair in pairs:
            group = [r for r in record["rows"] if r["parent_trace_sha256"] == pair["parent_trace_sha256"]]
            summary = dict(parent_trace_sha256=pair["parent_trace_sha256"], pattern=pair["pattern"], arms={})
            for arm in ("image_only", "mixed"):
                members = [r for r in group if r["arm"] == arm]
                if sorted(r["order_index"] for r in members) != [0, 1]:
                    raise ValueError("missing diagnostic AB/BA member")
                summary["arms"][arm] = _image_summary(members)
            summaries.append(summary)
        image_only_miss = any(s["arms"]["image_only"]["on_time"] < s["arms"]["image_only"]["arrived"] for s in summaries)
        mixed_miss = any(s["arms"]["mixed"]["on_time"] < s["arms"]["mixed"]["arrived"] for s in summaries)
        decision = "image_only_misses_original_deadlines" if image_only_miss else (
            "mixed_work_interference_observed" if mixed_miss else "no_image_miss_observed")
        record.update(status="completed", decision=decision, evaluation_complete=True, summaries=summaries,
            reason="same-image finite diagnostic; conditional service bounds do not establish hardware impossibility")
        atomic_json(path, record)
        return record
    except BaseException as error:
        _failure(record, path, error)
        raise
    finally:
        context.executor, context.save_partial = original_executor, original_partial


def run_cache_load(context, traces, stage_name, *, quick_seed=None):
    """Replay frozen K4 off/paired008 traces in ABBA order.

    The default remains all three seeds and 24 replays.  Explicit
    ``quick_seed=101`` selects ordinary/burst for an eight-replay screening;
    even a passing screening cannot qualify or promote the candidate.
    """
    if not stage_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in stage_name):
        raise ValueError("stage_name must be a simple path component")
    from .next_quick_amendment import admit_context_amendment
    amendment = admit_context_amendment(context, stage_name)
    if amendment:
        if quick_seed is not None and quick_seed != amendment["seed"]:
            raise ValueError("explicit quick seed differs from the authorized amendment")
        quick_seed = amendment["seed"]
    screening = quick_seed is not None
    if screening and (type(quick_seed) is not int or quick_seed != 101):
        raise ValueError("quick screening requires the preselected seed 101")
    matrix = {(t["seed"], t["pattern"]) for t in traces}
    full_matrix = {(s,p) for s in SEEDS for p in PATTERNS}
    selected_matrix = {(quick_seed,p) for p in PATTERNS}
    if not (len(traces) == 6 and matrix == full_matrix):
        if not (screening and len(traces) == 2 and matrix == selected_matrix):
            raise ValueError("cache load requires exactly three seeds and two patterns; explicit quick screening requires both seed-101 patterns")
    if len({t["load"] for t in traces}) != 1 or traces[0]["load"] not in LOADS:
        raise ValueError("cache stage must contain one frozen load")
    if screening:
        # Filter the frozen workload; never regenerate its clock from a new
        # service measurement or select a seed based on candidate outcomes.
        traces = [next(t for t in traces if t["seed"] == quick_seed and t["pattern"] == pattern)
                  for pattern in PATTERNS]
    source_path = Path(context.config["old_online"]).resolve()
    source_raw = source_path.read_bytes()
    source_sha = _sha(source_raw)
    original_costs = json.loads(source_raw).get("frozen_service_ms")
    if not isinstance(original_costs, dict) or set(original_costs) != {"image", "video"} or not all(_positive(v) for v in original_costs.values()):
        raise ValueError("invalid original cache-load calibration")
    expected_sha = context.config.get("next_workloads", {}).get("expected_source_sha256", source_sha)
    if source_sha != expected_sha:
        raise ValueError("cache-load original source SHA256 mismatch")
    generator_sha = _sha(inspect.getsource(frozen_requests).encode())
    for trace in traces:
        _validate_trace(trace)
        if len(trace["requests"]) != COUNT or sum(r["kind"] == "video" for r in trace["requests"]) != 2:
            raise ValueError("cache stage requires 24 requests including two videos")
        if (trace.get("source", {}).get("sha256") != source_sha or
            trace.get("source", {}).get("path") != str(source_path) or
            trace.get("calibration") != original_costs or trace.get("generator_sha256") != generator_sha):
            raise ValueError("cache trace source/generator/calibration mismatch")
        regenerated = frozen_requests(count=COUNT, seed=trace["seed"], load=trace["load"], pattern=trace["pattern"],
            video_ms=original_costs["video"], image_ms=original_costs["image"])
        if _inputs(regenerated) != _inputs(trace["requests"]):
            raise ValueError("cache trace differs from original generator/calibration")
    expected_ids = [t["trace_sha256"] for t in traces]
    if len(set(expected_ids)) != len(traces):
        raise ValueError("duplicate cache trace identity")
    if not context.metadata.get("wan_warmed") or not context.metadata.get("cache_factorial_warmed"):
        raise ValueError("cache load requires fresh off/paired model warmup evidence")
    executors = context.executors
    off, paired = executors["cache_factorial_off"], executors["cache_factorial_paired"]
    path = context.output / (stage_name + ".json")
    if path.exists():
        raise FileExistsError("cache load evidence already exists")
    record = dict(schema_version="ditserve.optimizations.cache_load.v1", status="running", rows=[],
        decision="pending", reason="", eligible=False, evaluation_complete=False, planned_replays=4*len(traces),
        planned_requests=4*len(traces)*COUNT, expected_trace_ids=expected_ids, load=traces[0]["load"],
        traces=[{k:v for k,v in t.items() if k != "requests"} for t in traces],
        warmup_metadata=copy.deepcopy(context.metadata),
        scope="fixed K4/eager Sana+Wan+VAE/CPU encode; cache off versus paired008; reference SLO never recalibrated")
    if screening:
        record.update(protocol_mode="single_seed_screening", selected_seed=quick_seed,
            screening_passed=False, screening_complete=False, online_qualified=False,
            full_qualification_completed=False)
        record["scope"] += "; single preselected seed screening only; not the complete three-seed qualification"
    if amendment:
        record["amendment_binding"] = amendment["amendment_binding"]
    atomic_json(path, record)
    original_executor, original_partial = context.executor, getattr(context, "save_partial", None)
    try:
        for trace in traces:
            for order, arms in enumerate((("baseline", "candidate"), ("candidate", "baseline"))):
                for arm in arms:
                    cached = arm == "candidate"
                    _replay(context, record, path, stage_name, trace, order, arm, paired if cached else off, cached=cached)
        gate = paired_gate(record["rows"], expected_trace_ids=expected_ids)
        record.update(status="completed", eligible=gate["passed"], online_qualified=gate["passed"],
            evaluation_complete=not gate["failures"] and gate["pooled"] is not None, comparison=gate,
            decision="online_gate_passed" if gate["passed"] else "online_gate_not_met",
            reason="finite current-load AB/BA gate; quality/recovery/capacity qualification remains separately required")
        if screening:
            record.update(status="stopped", eligible=False, online_qualified=False,
                screening_passed=gate["passed"], screening_complete=record["evaluation_complete"],
                evaluation_complete=False, screening_comparison=record.pop("comparison"),
                decision="screening_gate_passed" if gate["passed"] else "screening_gate_not_met",
                reason="original three-seed expansion stopped by explicit single-seed screening protocol; eight-replay seed-101 result does not establish three-seed robustness or qualify a new default")
        atomic_json(path, record)
        if amendment:
            from .next_quick_report import write_quick_report
            write_quick_report(context.output, amendment_binding=amendment["amendment_binding"])
        return record
    except BaseException as error:
        _failure(record, path, error)
        raise
    finally:
        context.executor, context.save_partial = original_executor, original_partial
