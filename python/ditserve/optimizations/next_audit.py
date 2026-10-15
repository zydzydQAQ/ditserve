"""Independent, metadata-only final integrity gate before follow-on retention.

No model libraries or raw arrays are read. A passed audit verifies accounting,
frozen identities, recorded numerical evidence and literal Excel cells; it does
not turn a failed/incomplete scientific experiment into a qualified backend.
Dependency closure and explicit deletion authorization remain separate gates.
"""
import hashlib
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import zipfile

from ..research.common import atomic_json, digest
from .next_protocol import PHASES

FIELDS = ("request_id", "kind", "arrival_ms", "deadline_ms", "service_ms", "seed")
FILES = dict(diagnosis="diagnosis.json", sana_vae="sana-candidate.json",
    sana_graph="sana-candidate.json", sana_numerics="sana-numerics.json", conditional="conditional.json")
TERMINAL = {"passed", "completed", "stopped", "failed", "incomplete"}
LIMIT = 128 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same(left, right):
    if _number(left) and _number(right):
        return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


class Reader:
    def __init__(self):
        self.sources = {}

    def raw(self, path, *, limit=LIMIT):
        path = Path(path).absolute()
        _require(path.resolve() == path and not path.is_symlink(), "symlink source: " + str(path))
        before = path.stat()
        _require(path.is_file() and before.st_size <= limit, "oversize/nonfile metadata: " + str(path))
        raw = path.read_bytes()
        after = path.stat()
        _require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                 (after.st_ino, after.st_size, after.st_mtime_ns), "source changed while reading")
        source = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.sources[str(path)] = source
        return raw, source

    def read(self, path):
        raw, source = self.raw(path)
        def invalid(value):
            raise ValueError("nonfinite JSON: " + value)
        return json.loads(raw, parse_constant=invalid), source

    def bound(self, binding):
        value, source = self.read(binding["path"])
        _require(source["sha256"] == binding["sha256"], "source binding changed: " + binding["path"])
        return value


def _identity(requests):
    # D1's frozen identity predates the compact canonical metadata encoding.
    payload = [dict({key: row[key] for key in FIELDS}, status="pending", start_ms=None, finish_ms=None)
               for row in requests]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

def _p95(values):
    if not values:
        return None
    values = sorted(float(v) for v in values)
    point = (len(values)-1)*.95
    low, high = math.floor(point), math.ceil(point)
    return values[low] + (values[high]-values[low])*(point-low)


def _metrics(requests, elapsed):
    arrived = [r for r in requests if r.get("arrived", True)]
    complete = [r for r in arrived if r.get("status") == "completed"]
    on_time = sum(r["finish_ms"] <= r["deadline_ms"] for r in complete)
    latencies = [r["finish_ms"]-r["arrival_ms"] for r in complete]
    waits = [r["start_ms"]-r["arrival_ms"] for r in complete]
    return dict(arrived=len(arrived), completed=len(complete), on_time=on_time,
        failed_or_incomplete=len(arrived)-len(complete), on_time_fraction=on_time/len(arrived) if arrived else None,
        goodput_rps=on_time*1000/elapsed if elapsed > 0 else None,
        latency_mean_ms=statistics.mean(latencies) if latencies else None,
        latency_p95_ms=_p95(latencies), wait_p95_ms=_p95(waits), sample_count=len(complete),
        tail_claim="descriptive p95; no stable p99 claim")


def _aggregate(rows):
    requests = [request for row in rows for request in row["requests"]]
    elapsed = sum(row["elapsed_ms"] for row in rows)
    return dict(metrics=_metrics(requests, elapsed),
        images=_metrics([r for r in requests if r["kind"] == "image"], elapsed),
        videos=_metrics([r for r in requests if r["kind"] == "video"], elapsed))


def _compare(baseline, candidate):
    bg, cg = baseline["metrics"]["goodput_rps"], candidate["metrics"]["goodput_rps"]
    bl, cl = baseline["images"]["latency_p95_ms"], candidate["images"]["latency_p95_ms"]
    bv, cv = baseline["videos"]["on_time_fraction"], candidate["videos"]["on_time_fraction"]
    valid = all(_number(x) for x in (bg, cg, bl, cl, bv, cv)) and bg > 0 and bl > 0
    gain, reduction = ((cg-bg)/bg, (bl-cl)/bl) if valid else (None, None)
    return dict(baseline=baseline, candidate=candidate, goodput_gain=gain, image_p95_reduction=reduction,
        video_ontime_drop=bv-cv if valid else None,
        safe=bool(valid and cg >= bg*.95 and cl <= bl*1.05 and cv >= bv-.05),
        system_gain=bool(valid and (gain >= .05 or reduction >= .10)))


def _matrix(data, expected, *, diagnosis=False):
    """Independently recompute complete and failed-arrival denominators."""
    rows, seen, row_hashes = data.get("rows", []), set(), []
    complete = data.get("status") == "completed" and len(rows) == len(expected)
    for index, row in enumerate(rows):
        key = row.get("trace_sha256"), row.get("order_index"), row.get("arm")
        _require(key in expected and key not in seen, "unexpected or duplicate replay identity")
        seen.add(key)
        trace = expected[key]
        requests = row["requests"]
        _require(_identity(requests) == key[0] and
            [{k:r[k] for k in FIELDS} for r in requests] == [{k:r[k] for k in FIELDS} for r in trace["requests"]],
            "replay arrivals/deadlines/seeds differ from frozen trace")
        _require(all(row.get(k) == trace[k] for k in ("seed", "load", "pattern")), "trace metadata differs")
        elapsed = row["elapsed_ms"]
        _require(_number(elapsed) and elapsed > 0, "invalid measured trace duration")
        for request in requests:
            _require(type(request.get("arrived")) is bool, "explicit request arrival state missing")
            if request["status"] == "completed":
                _require(request["arrived"] and all(_number(request.get(k)) for k in ("arrival_ms", "start_ms", "finish_ms"))
                    and request["arrival_ms"] <= request["start_ms"] <= request["finish_ms"] <= elapsed + 1e-6,
                    "completed request has invalid host interval")
        for name, subset in (("metrics", requests), ("image_metrics", [r for r in requests if r["kind"] == "image"]),
                             ("video_metrics", [r for r in requests if r["kind"] == "video"])):
            _require(name in row and _same(row[name], _metrics(subset, elapsed)), "request metrics differ: " + name)
        is_complete = row.get("status") == "completed" and row.get("complete") is True and all(
            r["arrived"] and r["status"] == "completed" for r in requests)
        complete = complete and is_complete
        provenance = data.get("_audit_row_sources", {}).get(str(index), {})
        row_hashes.append(dict(pointer=provenance.get("pointer", "/rows/%d" % index),
            **({"source":provenance["source"]} if "source" in provenance else {}),
            sha256=digest(row), trace_sha256=key[0], order_index=key[1], arm=key[2]))
    complete = bool(complete and seen == set(expected))
    if data.get("status") == "completed":
        _require(complete, "completed phase lacks its full frozen request matrix")
    groups = []
    for key in ("pattern", "trace_sha256"):
        for value in sorted({r[key] for r in rows}):
            for arm in sorted({r["arm"] for r in rows}):
                members = [r for r in rows if r[key] == value and r["arm"] == arm]
                if members:
                    metrics = _aggregate(members)
                    if diagnosis:
                        metrics = dict(images={k:v for k,v in metrics["images"].items() if k != "goodput_rps"})
                    groups.append(dict(group=key, value=value, arm=arm, replays=len(members), **metrics))
    result = dict(matrix_complete=complete, expected_replays=len(expected), recorded_replays=len(rows),
        expected_requests=sum(len(t["requests"]) for t in expected.values()),
        recorded_requests=sum(len(r["requests"]) for r in rows), canonical_rows=row_hashes, groups=groups,
        online_qualified=False, scope="independent host request metrics; failures remain in arrived denominators; descriptive p95")
    if complete and not diagnosis:
        per_trace = []
        for trace_id in sorted({key[0] for key in expected}):
            per_trace.append(dict(trace_sha256=trace_id, **_compare(
                _aggregate([r for r in rows if r["trace_sha256"] == trace_id and r["arm"] == "baseline"]),
                _aggregate([r for r in rows if r["trace_sha256"] == trace_id and r["arm"] == "candidate"]))))
        pooled = _compare(_aggregate([r for r in rows if r["arm"] == "baseline"]),
                          _aggregate([r for r in rows if r["arm"] == "candidate"]))
        qualified = all(r["safe"] for r in per_trace) and pooled["system_gain"]
        from .online import paired_gate
        official = paired_gate(rows, expected_trace_ids=sorted({key[0] for key in expected}))
        _require(official["passed"] == qualified and _same(official["pooled"], pooled), "independent paired gate disagreement")
        _require(_same(data.get("comparison"), official), "recorded online comparison differs")
        result.update(online_qualified=qualified, pooled=pooled, traces=per_trace)
    return result


def _with_partial(data, reader, attempt):
    """Retain an interrupted arrived denominator without counting it as complete."""
    rows = list(data.get("rows", []))
    provenance = {}
    active = data.get("active_unit") or {}
    path = data.get("active_trace_path") or active.get("partial_path")
    if path and active:
        path = Path(path).absolute()
        _require(attempt in path.parents, "partial leaves its owned attempt")
        partial, source = reader.read(path)
        identity = lambda row:(row.get("trace_sha256"),row.get("order_index"),row.get("arm"))
        if partial.get("requests") and identity(partial) not in {identity(r) for r in rows}:
            provenance[str(len(rows))] = dict(source=source, pointer="")
            rows.append(partial)
        incomplete = data.get("incomplete_trace")
        if incomplete:
            _require(incomplete.get("path") == str(path) and incomplete.get("sha256") == source["sha256"],
                     "interrupted denominator source changed")
    return dict(data, rows=rows, _audit_row_sources=provenance)


def _expected(workloads, phase):
    if phase == "diagnosis":
        _require(len(workloads["diagnosis"]) == 2, "diagnosis must contain two frozen source traces")
        result = {}
        for pair in workloads["diagnosis"]:
            mixed, image = pair["mixed"], pair["image_only"]
            _require(len(mixed["requests"]) == 24 and len(image["requests"]) == 22 and
                [r for r in mixed["requests"] if r["kind"] == "image"] == image["requests"],
                "image-only diagnostic changed original image requests")
            for arm in ("image_only", "mixed"):
                trace = pair[arm]
                _require(_identity(trace["requests"]) == trace["trace_sha256"], "frozen diagnostic identity differs")
                for order in (0, 1):
                    result[trace["trace_sha256"], order, arm] = trace
        _require(len(result) == 8 and sum(len(t["requests"]) for t in result.values()) == 184, "diagnosis denominator differs")
        return result
    if phase.startswith("cache_load_"):
        load = {"cache_load_05":"0.5", "cache_load_07":"0.7", "cache_load_09":"0.9"}[phase]
        traces = workloads["highload"][load]
        _require(len(traces) == 6 and {(t["seed"],t["pattern"]) for t in traces} ==
            {(s,p) for s in (101,102,103) for p in ("ordinary","burst")}, "highload seed/pattern matrix differs")
        _require(all(t["load"] == float(load) for t in traces), "highload calibration load differs")
    else:
        traces = [pair["mixed"] for pair in workloads["diagnosis"]]
    for trace in traces:
        _require(len(trace["requests"]) == 24 and sum(r["kind"] == "video" for r in trace["requests"]) == 2
            and _identity(trace["requests"]) == trace["trace_sha256"], "fixed mixed request identity differs")
    _require(len({t["trace_sha256"] for t in traces}) == len(traces), "duplicate frozen trace identities")
    return {(t["trace_sha256"], order, arm):t for t in traces for order in (0,1) for arm in ("baseline","candidate")}


def _quality(value, epsilon=.01, frames=1):
    m = value.get("metrics", {})
    values = m.get("lpips_per_frame", [])
    return bool(m.get("status") == "ok" and m.get("finite") is True and m.get("range_valid") is True
        and m.get("frame_count") == frames and len(values) == frames and all(_number(x) and x >= 0 for x in values)
        and _number(m.get("lpips_mean")) and _number(m.get("lpips_p95"))
        and _same(m["lpips_mean"], statistics.mean(values)) and _same(m["lpips_p95"], _p95(values))
        and all(m.get(k) is None or _number(m[k]) for k in ("psnr_db","temporal_delta_rmse"))
        and 0 <= m["lpips_mean"] <= epsilon and m.get("lpips_status") == "ok")


def _recovery(data, candidate, reader, attempt):
    binding = data["recovery"]
    _require(Path(binding["path"]) == attempt/"sana-wan-resume.json", "recovery binding leaves attempt")
    value = reader.bound(binding)
    _require(value.get("candidate") == candidate and value.get("status") == "passed"
        and value.get("resume_qualified") is True and value.get("eligible") is True
        and binding.get("passed") is True and binding.get("status") == "passed"
        and value.get("gate") == binding.get("gate"), "recovery parent/child qualification differs")
    gate, protocol = value.get("gate", {}), value.get("fixed_protocol", {})
    _require(gate.get("passed") is True and gate.get("step_count") == 32 and gate.get("real_sana_insertions") == 3
        and protocol.get("seed") == 101 and protocol.get("input_id") == "kettle-101"
        and protocol.get("steps") == 32 and protocol.get("rtol") == 1e-5 and protocol.get("atol") == 1e-6,
        "recovery fixed protocol/gate differs")
    rows = value.get("rows", [])
    _require(len(rows) == 1 and rows[0].get("input_id") == "kettle-101", "recovery fixed input differs")
    result = rows[0]["result"]
    steps, insertions, final = result.get("steps", []), result.get("interruptions", []), result.get("final", {})
    _require(result.get("passed") is True and len(steps) == 32 and [r.get("step") for r in steps] == list(range(1,33))
        and all(all(r.get(k) is True for k in ("latent","solver","cache_state","rng","passed","cache_paths_equal")) for r in steps),
        "recovery lacks 32 strict state checks")
    _require([r.get("step") for r in insertions] == [0,8,16] and all(
        r.get("image_service", {}).get("candidate") == candidate and r["image_service"].get("finite") is True for r in insertions),
        "recovery lacks three real candidate insertions")
    _require(final.get("passed") is True and final.get("finite") is True and final.get("rtol") == 1e-5
        and final.get("atol") == 1e-6 and final.get("shape") == [1,81,480,832,3]
        and _number(final.get("max_abs")) and final["max_abs"] >= 0, "recovery final numerical evidence differs")
    paths = result.get("cache_paths", {})
    _require(paths.get("valid") is True and paths.get("paths_equal") is True, "recovery cache path mismatch")
    for name in ("continuous","k4"):
        path = paths.get(name, {})
        sequence = path.get("path_sequence", [])
        _require(path.get("valid") is True and len(sequence) == 32 and
            all(r.get("step_index") == i and r.get("completed_steps") == i+1
                and r.get("outcome") in ("full","reuse") for i,r in enumerate(sequence)), "recovery path coverage incomplete")
    _require(paths["continuous"]["path_sequence"] == paths["k4"]["path_sequence"], "recovery path sequences differ")
    cleanup = value.get("cleanup", {})
    _require(cleanup.get("passed") is True and cleanup.get("worker_live_states_after") == 0
        and cleanup.get("executor_live_states_after") == 0 and cleanup.get("prompt_bank_entries_after") == 0,
        "recovery state cleanup incomplete")
    return dict(passed=True, source=binding, strict_steps=32, real_sana_insertions=3,
                raw_arrays_rehashed=False, scope="recorded numerical metrics only; no raw array reevaluation")


def _sana_rejection(data, config, reader, attempt):
    """Verify a narrow local screening rejection, independently of its producer.

    It closes a failed candidate attempt without granting numerical, speed or
    serving qualification. Process release/accounting still belong to _audit.
    """
    candidate, failure = data["candidate"], data["expected_failure"]
    _require(data.get("status") == "stopped" and data.get("decision") == "未验证"
        and data.get("active_unit", "missing") is None
        and all(data.get(key) is False for key in ("local_qualified", "online_qualified", "online_evaluation_complete"))
        and not any(data.get(key) for key in ("quality_passed", "performance_passed", "round_speed_gains", "summaries",
                                             "system_qualified", "changes_serving_default", "capacity", "recovery")),
        "candidate rejection cannot retain qualification or complete quality/speed claims")
    cleanup = failure.get("cleanup", {})
    _require(all(cleanup.get(key) is True for key in ("passed", "hooks_restored", "graph_state_released")),
        "candidate rejection cleanup is unverified")
    unit = failure.get("failed_unit", {})
    _require(unit.get("kind") in ("warmup", "measurement") and unit.get("arm") == candidate
        and _number(failure.get("failed_invocation_wall_ms")) and failure["failed_invocation_wall_ms"] >= 0,
        "candidate rejection lacks selected local invocation and charged cold cost")
    sequence = [dict(kind="warmup", arm=arm, index=i) for arm in ("eager", candidate) for i in range(2)]
    sequence += [dict(kind="measurement", order_index=order, arm=arm, input_id="p%d-s%d" % (p,seed),
        split="holdout" if seed in (201,202) else "development", repeat=repeat)
        for order, arms in enumerate((("eager",candidate),(candidate,"eager"))) for arm in arms
        for p in range(len(config["direction1"]["prompts"])) for seed in config["seeds"] for repeat in range(3)]
    rows = data.get("rows", [])
    _require(len(rows) < len(sequence) and unit == sequence[len(rows)],
        "candidate rejection failed invocation differs from next frozen local unit")
    for row, identity in zip(rows, sequence):
        _require({key:row.get(key) for key in identity} == identity
            and row.get("service", {}).get("candidate") == identity["arm"],
            "candidate rejection partial rows are not the frozen execution prefix")
    operation = failure.get("operation", {})
    owner = "_CompiledRegion" if candidate == "vae_compile" else "_TransformerGraph"
    operations = ("__call__", "_backend") if candidate == "vae_compile" else ("_capture",)
    phase = "sana_vae" if candidate == "vae_compile" else "sana_graph"
    _require(attempt.parent.name == phase and operation.get("module") == "ditserve.optimizations.next_sana"
        and operation.get("owner") == owner and operation.get("operation") in operations,
        "candidate rejection operation is outside its selected implementation")
    root = attempt.parent.parent / "sources" / (phase + "-" + attempt.name)
    manifest, unused = reader.read(root / "manifest.json")
    relative = "python/ditserve/optimizations/next_sana.py"
    _require(operation.get("source_path") == str(root / relative)
        and manifest.get("execution_root") == str(root)
        and manifest.get("source_files", {}).get(relative) == operation.get("source_sha256")
        and "python/ditserve/optimizations/next_candidate_failures.py" in manifest.get("source_files", {}),
        "candidate rejection frame is not bound to its executed snapshot")
    unused, source = reader.raw(root / relative)
    _require(source["sha256"] == operation["source_sha256"], "candidate rejection frame source changed")
    chain = failure.get("exception_chain", [])
    _require(type(chain) is list and 0 < len(chain) <= 32, "candidate rejection exception chain missing")
    for item in chain:
        mro = item.get("type_mro", [])
        _require(isinstance(item.get("type"), str) and isinstance(item.get("message"), str)
            and type(mro) is list and bool(mro) and all(isinstance(value,str) for value in mro)
            and mro[0] == item["type"], "candidate rejection exception lineage missing")
        fatal_types = {"ditserve.directions.budget.BudgetStop", "ditserve.optimizations.budget.BudgetStop", "builtins.KeyboardInterrupt", "builtins.SystemExit",
            "builtins.ValueError", "builtins.MemoryError", "builtins.OSError"}
        _require(not fatal_types.intersection(mro) and not item["type"].endswith(".OutOfMemoryError")
            and not any(word in item["message"].lower() for word in ("out of memory", "illegal memory access",
                "device-side assert", "unspecified launch failure", "misaligned address", "driver shutting down",
                "invalid device function", "launch timeout")), "fatal error cannot become candidate screening rejection")
    _require(chain[0]["type"] == failure.get("module", "") + "." + failure.get("error_type", "")
        and chain[0]["message"] == failure.get("message"), "candidate rejection original error differs")
    category = None
    if candidate == "vae_compile":
        accepted = {"torch._dynamo.exc.Unsupported":"compiler_unsupported",
                    "torch._dynamo.exc.RecompileError":"measurement_recompile",
                    "torch._dynamo.exc.FailOnRecompileLimitHit":"measurement_recompile"}
        for item in chain:
            for kind in item["type_mro"]:
                match = accepted.get(kind)
                if match and (match != "measurement_recompile" or unit["kind"] == "measurement"):
                    category = match
            if item["type"] == "builtins.RuntimeError":
                message = item["message"]
                if message in ("Sana decoder executed without a compiled graph", "Sana post_quant_conv executed without a compiled graph"):
                    category = "compiled_graph_not_executed"
                elif unit["kind"] == "measurement" and (message in ("Sana decoder compiled during steady measurement",
                        "Sana post_quant_conv compiled during steady measurement")
                        or message.startswith("Detected recompile when torch.compile stance is 'fail_on_recompile'. ")):
                    category = "measurement_recompile"
    elif chain[0]["type"] == "builtins.RuntimeError" and chain[0]["message"].splitlines()[0:1] in (
            ["CUDA error: operation not permitted when stream is capturing"],
            ["CUDA error: operation not permitted when stream is capturing."]):
        category = "cuda_capture_unsupported"
    _require(category is not None and failure.get("classification") == category,
        "candidate rejection error is not a recognized compiler/capture limitation")
    error_text = failure["error_type"] + ": " + failure["message"]
    _require(data.get("failure_reason") == error_text and data.get("reason") == "candidate screening rejected: " + category
        and isinstance(data.get("traceback"), str) and operation["source_path"] in data["traceback"]
        and "in " + operation["operation"] in data["traceback"] and error_text in data["traceback"],
        "candidate rejection original traceback missing")
    mechanics = data.get("mechanics", {})
    failures = mechanics.get("failures", [])
    _require(mechanics.get("candidate") == candidate and mechanics.get("status") == "failed"
        and mechanics.get("qualified") is False and mechanics.get("cleanup_failed") is False
        and bool(failures) and failures[-1].get("candidate") == candidate and failures[-1].get("error") == error_text,
        "candidate rejection lacks matching failed execution mechanics")
    if candidate == "vae_compile":
        regions = failures[-1].get("vae_regions", [])
        _require(bool(regions) and all(r.get("name") in ("decoder", "post_quant_conv")
            and r.get("fullgraph") is True and r.get("dynamic") is False and r.get("cudagraphs") is False
            and r.get("original_module_preserved") is True
            and _number(r.get("wrapper_setup_ms")) and r["wrapper_setup_ms"] >= 0
            and r.get("graph_compile_count") == len(r.get("records", [])) for r in regions),
            "candidate rejection lacks fixed compiler mechanics/cold setup cost")
    else:
        graph = mechanics.get("transformer_graph", {})
        records = graph.get("records", [])
        _require(bool(records) and graph.get("compiled") is False
            and all(r.get("status") in ("completed", "failed") and _number(r.get("cold_host_ms"))
                and r["cold_host_ms"] >= 0 for r in records)
            and graph.get("capture_count") == sum(r["status"] == "completed" for r in records),
            "candidate rejection lacks actual capture mechanics/cold costs")
    return dict(classification=category, failed_unit=unit, failed_invocation_wall_ms=failure["failed_invocation_wall_ms"],
        source=source, adapter_cleanup_verified=True, process_cleanup_required=True,
        scope="recorded negative screening result; partial samples retained without qualification")


def _sana(data, config, reader, attempt, workloads):
    candidate = data["candidate"]
    _require(candidate in ("vae_compile","transformer_graph"), "unexpected Sana candidate")
    inputs = {"p%d-s%d"%(i,s) for i in range(len(config["direction1"]["prompts"])) for s in config["seeds"]}
    _require(len(inputs) == 8, "Sana requires eight frozen inputs")
    expected = {(o,a,i,r) for o in (0,1) for a in ("eager",candidate) for i in inputs for r in range(3)}
    rows = [r for r in data.get("rows", []) if r.get("kind") == "measurement"]
    seen, hashes, qualities = set(), {}, []
    all_repeats = all_finite = True
    for row in rows:
        key = row["order_index"], row["arm"], row["input_id"], row["repeat"]
        _require(key in expected and key not in seen, "duplicate/unexpected Sana local sample")
        seen.add(key)
        service = row["service"]
        _require(service.get("candidate") == row["arm"], "Sana measured backend differs from arm")
        _require(_number(service.get("invocation_wall_ms")) and service["invocation_wall_ms"] > 0
            and _number(service.get("ms")) and 0 <= service["ms"] <= service["invocation_wall_ms"] + 1e-5,
            "Sana service timer excludes pipeline or is invalid")
        all_finite = all_finite and service.get("finite") is True
        output_hash = row.get("output_sha256")
        _require(isinstance(output_hash,str) and len(output_hash) == 64 and
                 all(c in "0123456789abcdef" for c in output_hash), "Sana output hash missing")
        same = hashes.setdefault(key[1:3], output_hash) == output_hash
        _require(row.get("repeat_exact") is same, "recorded repeat consistency differs from output hash")
        all_repeats = all_repeats and same
        _require(("quality" in row) == (row["repeat"] == 0), "quality denominator differs from repeat zero")
        if "quality" in row:
            passed = _quality(row["quality"])
            _require(row["quality"].get("passed") is passed, "recorded quality contradicts finite/range/LPIPS evidence")
            qualities.append(passed)
    full = seen == expected
    gains = []
    if full:
        for order in (0,1):
            means = {a:statistics.mean(r["service"]["invocation_wall_ms"] for r in rows
                if r["order_index"] == order and r["arm"] == a) for a in ("eager",candidate)}
            gains.append(1-means[candidate]/means["eager"])
        if "round_speed_gains" in data or data.get("status") == "completed":
            _require(_same(data.get("round_speed_gains"), gains), "Sana gain differs from complete invocation wall")
    if data.get("status") == "completed":
        _require(full and len(qualities) == 32, "completed Sana local phase lacks 96/32 denominator")
    rejection = None
    if "expected_failure" in data and data.get("status") == "stopped":
        rejection = _sana_rejection(data, config, reader, attempt)
    if data.get("status") == "stopped" and not rows and rejection is None:
        _require(candidate == "transformer_graph" and data.get("trigger_evidence", {}).get("triggered") is False,
                 "empty stopped Sana phase lacks its explicit graph trigger conclusion")
    local = full and len(qualities) == 32 and all(qualities) and all_repeats and all_finite and all(g >= .05 for g in gains)
    result = dict(local_rows=len(rows), required_rows=96, quality_rows=len(qualities), required_quality_rows=32,
        quality_failures=sum(not q for q in qualities), full_matrix=full, recomputed_speed_gains=gains,
        component_gate=bool(local), local_qualified=False, online_qualified=False)
    if rejection is not None:
        result["expected_failure"] = rejection
    if data.get("local_qualified") is True:
        _require(local and data.get("quality_passed") is True and data.get("performance_passed") is True
            and data.get("capacity", {}).get("passed") is True, "Sana qualification skips full local gate or capacity")
        result["recovery"] = _recovery(data, candidate, reader, attempt)
        result["local_qualified"] = True
    online_path = attempt/"sana-online.json"
    if online_path.exists():
        online, source = reader.read(online_path)
        _require(result["local_qualified"], "online experiment lacks qualified local recovery")
        result["online"] = _matrix(_with_partial(online, reader, attempt), _expected(workloads, "sana"))
        result["online"]["source"] = source
        result["online_qualified"] = result["online"]["online_qualified"]
    _require(bool(data.get("online_qualified")) == result["online_qualified"], "Sana online qualification differs")
    return result


def _stage_local(data, kind):
    inputs = {"p%d-s%d"%(p,s) for p in range(2) for s in (101,102)}
    rows = data.get("rows", [])
    if kind == "video_vae":
        rows = [r for r in rows if r.get("kind") == "measurement"]
        expected = {(i,o,a,0) for i in inputs for o in (0,1) for a in ("eager","compiled")}
        identities, quality, costs = [], [], {}
        for row in rows:
            key = row["input_id"],row["round"],row["arm"],row["repeat"]
            identities.append(key)
            _require(key in expected and _number(row["service"].get("ms")) and row["service"]["ms"] > 0
                and row.get("generation_completed") is True, "invalid video-VAE decode sample")
            costs[key] = row["service"]["ms"]
            if "quality" in row:
                passed = _quality(row["quality"], .05, 81)
                _require(row["quality"].get("passed") is passed, "video-VAE LPIPS/finite/range gate differs")
                quality.append(passed)
        _require(len(set(identities)) == len(identities), "duplicate video-VAE decode")
        full = set(identities) == expected
        gains = [1-statistics.mean(costs[k] for k in expected if k[1]==o and k[2]=="compiled")/
                   statistics.mean(costs[k] for k in expected if k[1]==o and k[2]=="eager") for o in (0,1)] if full else []
        if full and "order_gains" in data:
            _require(_same(data["order_gains"], gains), "video-VAE gain differs from complete decode wall")
        passed = full and len(quality)==16 and all(quality) and all(g>=.05 for g in gains)
        return dict(passed=bool(passed), samples=len(rows), required_samples=16, quality_samples=len(quality),
                    order_gains=gains, full_matrix=full)
    expected = {(i,a) for i in inputs for a in ("cpu","gpu")}
    identities, by_input = [], {}
    for row in rows:
        key = row["input_id"],row["arm"]
        identities.append(key)
        _require(key in expected and row.get("completed") is True and row.get("frames")==81
            and row.get("completed_steps")==32 and row.get("cache")=="paired_008" and row.get("policy")=="block-4"
            and row.get("wan_backend")=="eager" and all(_number(row.get(k)) and row[k]>0 for k in ("encode_ms","generation_ms"))
            and row["generation_ms"]>=row["encode_ms"]
            and row.get("device")==("cpu" if row["arm"]=="cpu" else "cuda:1"), "encoder local control or interval differs")
        by_input[key] = row
    _require(len(set(identities)) == len(identities), "duplicate encoder local sample")
    pairs = data.get("pairs", [])
    seen, qualities = set(), []
    for pair in pairs:
        name=pair["input_id"]
        _require(name in inputs and name not in seen and all((name,a) in by_input for a in ("cpu","gpu")), "encoder pair denominator differs")
        seen.add(name)
        expected_order = ["cpu","gpu"] if name.endswith("s101") else ["gpu","cpu"]
        _require(pair.get("order") == expected_order, "encoder fixed alternating arm order differs")
        passed=_quality(pair["quality"],.05,81)
        _require(pair["quality"].get("passed") is passed, "encoder quality differs from finite/range LPIPS")
        qualities.append(passed)
        for metric,gain in (("encode_ms","encode_gain"),("generation_ms","generation_gain")):
            _require(_same(pair[gain],1-by_input[name,"gpu"][metric]/by_input[name,"cpu"][metric]), "encoder paired gain differs")
    full=set(identities)==expected and seen==inputs
    gains={}
    if full:
        for metric,gain in (("encode_ms","encode_gain"),("generation_ms","generation_gain")):
            gains[gain]=1-statistics.mean(r[metric] for r in rows if r["arm"]=="gpu")/statistics.mean(r[metric] for r in rows if r["arm"]=="cpu")
            if gain in data:
                _require(_same(data[gain],gains[gain]), "encoder aggregate gain differs")
    passed=full and all(qualities) and all(v>=.05 for v in gains.values())
    if data.get("local_qualified"):
        _require(data.get("encoder_device")=="cuda:1" and data.get("encoder_dtype")=="float32"
            and data.get("payload_device")=="cpu" and data.get("payload_dtype")=="float32", "encoder precision/payload control differs")
    return dict(passed=bool(passed),samples=len(rows),required_samples=8,quality_samples=len(qualities),
                full_matrix=full,**gains)


def _latent_paths(value):
    """A fixed input may legitimately take only full steps; do not require reuse."""
    sequence=value.get("path_sequence",[])
    counts=dict(full=0,reuse=0,invalid=0)
    valid=len(sequence)==32
    for index,row in enumerate(sequence):
        outcome=("full" if row.get("path")=="compute" and row.get("reused_forwards")==0 else
                 "reuse" if row.get("path")=="reuse" and row.get("reused_forwards")==2 else "invalid")
        counts[outcome]+=1
        valid=valid and row.get("step_index")==index and row.get("completed_steps")==index+1 and row.get("outcome")==outcome
    valid=bool(valid and not counts["invalid"])
    _require(value.get("valid") is valid and value.get("counts")==counts
        and value.get("full_reuse_observed") is bool(valid and counts["full"] and counts["reuse"]),
        "paired latent cache-path evidence differs")
    return valid


def _triggers(value, reader, output, ledger, sessions, workloads):
    for binding in value.get("immutable_bindings", []):
        _require(Path(binding["path"]) != ledger, "mutable whole ledger cannot be an immutable trigger source")
        unused, source=reader.raw(binding["path"])
        _require(source["sha256"]==binding["sha256"], "immutable trigger binding changed")
    flags=dict(vae_trigger=False,encoder_trigger=False)
    observed_sources=[]
    from .online import attribution
    for source in value.get("admitted_sources", []):
        phase=source["phase"]
        _require(phase in ("diagnosis","cache_load_05","cache_load_07","cache_load_09"), "invalid conditional trigger phase")
        _require(Path(source["terminal"]["path"])==output/phase/"terminal.json", "trigger terminal leaves phase")
        terminal=reader.bound(source["terminal"])
        _require(source["source"]["path"]==terminal.get("evidence_path"), "conditional trigger source differs from supervised native evidence")
        native=reader.bound(source["source"])
        found=[s for s in sessions if s["session_id"]==source["session_id"]]
        _require(found==[source["settled_session"]] and terminal.get("charge")==found[0]
            and found[0]["status"]=="completed" and terminal.get("status")=="completed"
            and terminal.get("returncode")==0 and not terminal.get("supervisor_failure"), "conditional trigger source not supervised complete")
        matrix=_matrix(native,_expected(workloads,phase),diagnosis=phase=="diagnosis")
        _require(matrix["matrix_complete"] and len(source["rows"])==len(native["rows"]), "conditional trigger matrix is incomplete")
        entry_flags=dict(vae_trigger=False,encoder_trigger=False)
        for index,(record,row) in enumerate(zip(source["rows"],native["rows"])):
            computed=attribution(row["requests"],row["events"],row["elapsed_ms"])
            trigger_arm=phase!="diagnosis" or row["arm"]=="mixed"
            _require(record["pointer"]=="/rows/%d"%index and record["attribution"]==computed==row["attribution"]
                and record["trigger_arm"] is trigger_arm and all(record[k]==row[k] for k in ("trace_sha256","arm","order_index","seed","pattern")),
                "conditional trigger attribution/identity differs")
            entry_flags["vae_trigger"] |= trigger_arm and computed["vae_trigger"]
            entry_flags["encoder_trigger"] |= trigger_arm and computed["encoding_trigger"]
        _require(all(source[k] is v for k,v in entry_flags.items()), "conditional source trigger flag differs")
        for key in flags:
            flags[key] |= entry_flags[key]
        observed_sources.append(dict(phase=phase,source=source["source"],**entry_flags))
    _require(all(value.get(k) is v for k,v in flags.items()) and value.get("triggered") is any(flags.values()),
             "conditional aggregate trigger differs")
    if any(flags.values()):
        _require(bool(value.get("immutable_bindings")), "trigger lacks immutable evidence bindings")
    return dict(**flags,sources=observed_sources)


def _conditional(data, session, reader, output, ledger, workloads, sessions):
    attempt=Path(session["identity"]["output"])
    binding=session["identity"].get("conditional_admission")
    _require(binding and Path(binding["path"])==attempt/"conditional-admission.json", "conditional admission binding missing")
    admission=reader.bound(binding)
    _require(admission["devices"]==session["identity"]["gpu_uuids"]==data["admitted_devices"]
        and admission["auxiliary_encoder"] is (session["gpu_count"]==2)
        and admission["trigger_evidence"]==data["trigger_evidence"], "conditional admission/aggregate differs")
    triggers=_triggers(data["trigger_evidence"],reader,output,ledger,sessions,workloads)
    result=dict(triggers=triggers,components={},independently_qualified=[],combined_qualified=False)
    for kind,trigger_key,summary_name,local_name,online_name in (
            ("video_vae","vae_trigger","conditional-video-vae.json","vae.json","video-vae-online.json"),
            ("text_encoder","encoder_trigger","conditional-text-encoder.json","text_encoder.json","encoder-online.json")):
        component=data.get(kind)
        if component is None:
            _require(data.get("status") in ("failed","incomplete"), "completed conditional lacks component conclusion")
            result["components"][kind]=dict(status="not_completed",online_qualified=False)
            continue
        source=component.get("component_evidence")
        if not source:
            _require(not any(component.get(k) for k in ("local_qualified","online_qualified","online_evaluation_complete","gpu_work_started"))
                and component.get("status")=="stopped" and bool(component.get("reason"))
                and (not triggers[trigger_key] or kind=="text_encoder" and session["gpu_count"]==1),
                "conditional skip lacks absent-trigger/resource justification")
            result["components"][kind]=dict(status="stopped",reason=component["reason"],online_qualified=False)
            continue
        _require(triggers[trigger_key] and component.get("triggered") is True
            and Path(source["path"])==attempt/summary_name, "conditional component lacks its actual trigger")
        producer=reader.bound(source)
        _require(producer=={k:v for k,v in component.items() if k not in ("triggered","component_evidence")}, "conditional summary changed")
        item=dict(status=producer["status"],source=source,local_qualified=False,online_qualified=False)
        local_binding=producer.get("local_evidence")
        if not local_binding:
            _require(producer.get("status") in ("failed","incomplete") and not any(producer.get(k) for k in
                ("local_qualified","online_qualified","system_qualified")), "conditional local evidence missing")
        else:
            _require(Path(local_binding["path"])==attempt/local_name, "conditional local source leaves attempt")
            native=reader.bound(local_binding)
            local=_stage_local(native,kind)
            item["local"]=local
            item["local_source"]=local_binding
            measured_local=native.get("local_qualified") is True
            withdrawn=measured_local and producer.get("local_qualified") is not True
            cleanup_ok=kind!="video_vae" or producer.get("decoder_cleanup_passed") is True
            if producer.get("local_qualified") is True or native.get("local_qualified") is True:
                _require(measured_local and local["passed"], "conditional local qualification differs")
                _require(not withdrawn or producer.get("status") in ("failed","incomplete")
                    and bool(producer.get("reason") or producer.get("cleanup_error")),
                    "conditional qualification withdrawal lacks its recorded failure")
                if kind=="video_vae":
                    compiler=producer.get("compiler",{})
                    records=compiler.get("compilation",[])
                    _require(compiler.get("callable_counting") is True
                        and bool(records) and all(r.get("status")=="completed" and type(r.get("graph_invocations")) is int and r["graph_invocations"]>0 for r in records)
                        and compiler.get("graph_compile_count")==len(records)
                        and compiler.get("graph_invocations")==sum(r["graph_invocations"] for r in records)
                        and type(compiler.get("compiled_forward_calls")) is int and compiler["compiled_forward_calls"]>=8,
                        "qualified VAE lacks actual compiled invocation proof")
                    latent=native.get("paired_latent_sources",[])
                    expected_latents={"stage-p%d-s%d-vae-latent"%(p,s):s for p in range(2) for s in (101,102)}
                    _require(len(latent)==4 and {r.get("request_id"):r.get("seed") for r in latent}==expected_latents
                        and all(r.get("completed_steps")==32 and r.get("quantum")==4
                        and r.get("policy")=="paired_008" and _latent_paths(r.get("cache_paths",{})) for r in latent),
                        "VAE local baseline lacks four complete paired/K4 latent sources")
                else:
                    _require(session["gpu_count"]==2, "V100 encoder experiment was not charged for two devices")
                _require(producer.get("local_qualified") is not True or cleanup_ok,
                         "conditional cleanup failure cannot retain qualification")
                item["local_qualified"]=not withdrawn and cleanup_ok
            item["measured_local_qualified"]=measured_local
            item["qualification_withdrawn"]=withdrawn
            measured_online=False
            online_path=producer.get("online_path") or native.get("online_path")
            if online_path:
                _require(measured_local and Path(online_path)==attempt/online_name, "conditional online source/prerequisite differs")
                online,online_source=reader.read(attempt/online_name)
                rows=_with_partial(online,reader,attempt)
                item["online"]=_matrix(rows,_expected(workloads,"sana"))
                item["online"]["source"]=online_source
                for row in rows["rows"]:
                    device="cuda:1" if kind=="text_encoder" and row["arm"]=="candidate" else "cpu"
                    _require(row.get("policy")=="block-4" and row.get("image_batch_size")==1 and row.get("encode_device")==device
                        and row.get("encode_on_scheduler") is False, "conditional online changed K/batch/encoding control")
                measured_online=item["online"]["online_qualified"]
                _require(not measured_online or online.get("evaluation_complete") is True
                    and native.get("online_evaluation_complete") is True
                    and producer.get("online_evaluation_complete") is True,
                    "conditional measured gate lacks recorded complete evaluation")
                item["online_qualified"]=bool(measured_online and item["local_qualified"])
            _require(bool(native.get("online_qualified"))==measured_online, "conditional native online gate differs")
        _require(bool(producer.get("online_qualified"))==item["online_qualified"] and
            bool(producer.get("system_qualified"))==item["online_qualified"], "conditional component online/system qualification differs")
        result["components"][kind]=item
        if item["online_qualified"]:
            result["independently_qualified"].append(kind)
    if data.get("status")=="completed":
        _require(data.get("independently_qualified")==result["independently_qualified"], "conditional independent selection differs")
    _require(not data.get("online_qualified") and not data.get("local_qualified"), "conditional components cannot implicitly qualify a combination")
    return result


def _snapshot(reader, output, session, config_sha, workloads_sha, mathematics):
    identity, phase = session["identity"], session["stage"]
    attempt = Path(identity["output"])
    _require(attempt.parent == output/phase and identity["run_root"] == str(output), "session output leaves phase")
    _require(identity["config_sha256"] == config_sha and identity["workloads_sha256"] == workloads_sha, "session frozen identity differs")
    config, unused = reader.read(attempt/"config.json")
    _require(digest(config) == config_sha, "attempt config changed")
    manifest_path = output/"sources"/(phase+"-"+attempt.name)/"manifest.json"
    manifest, source = reader.read(manifest_path)
    hashes, root = manifest["source_files"], manifest_path.parent
    _require(bool(hashes) and digest(hashes) == identity["source_sha256"] == manifest["source_sha256"]
        and manifest["execution_root"] == str(root) and manifest["config_sha256"] == config_sha, "snapshot manifest changed")
    _require(all(hashes.get(name) == sha for name,sha in mathematics.items()),
             "executed cache arithmetic differs from inherited qualification")
    for name, sha in hashes.items():
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts and relative.suffix in (".py",".json"), "invalid snapshot entry")
        unused, binding = reader.raw(root/relative, limit=32*1024*1024)
        _require(binding["sha256"] == sha, "executed snapshot source changed: " + name)
    return attempt, config, source


def _excel(path, tables, reader):
    """Read every literal worksheet cell with stdlib XML; never evaluate formulas."""
    unused, source = reader.raw(path)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        _require(sum(i.file_size for i in archive.infolist()) <= 512*1024*1024, "Excel expansion exceeds audit bound")
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        names = [s.attrib["name"] for s in workbook.findall(ns+"sheets/"+ns+"sheet")]
        _require(names == list(tables), "Excel sheet names/order differ")
        cells = 0
        for index, (name, expected) in enumerate(tables.items(), 1):
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet%d.xml" % index))
            rows = sheet.findall(ns+"sheetData/"+ns+"row")
            _require(len(rows) == len(expected), "Excel row count differs: " + name)
            actual_rows, expected_rows = [], []
            for row, wanted in zip(rows, expected):
                actual = [None]*len(wanted)
                seen = set()
                for cell in row:
                    ref = cell.attrib.get("r", "")
                    col = 0
                    for char in ref:
                        if char.isalpha():
                            col = col*26 + ord(char)-64
                        else:
                            break
                    _require(0 < col <= len(wanted) and col not in seen and cell.find(ns+"f") is None
                        and cell.attrib.get("t") in ("inlineStr","n"), "Excel nonliteral/duplicate cell")
                    seen.add(col)
                    if cell.attrib.get("t") == "inlineStr":
                        texts = cell.findall(".//"+ns+"t")
                        actual[col-1] = "".join(t.text or "" for t in texts) if texts else None
                    else:
                        _require(cell.find(ns+"v") is None, "Excel nonliteral numeric cell")
                wanted = [None if v is None else str(v) for v in wanted]
                actual_rows.append(tuple(actual))
                expected_rows.append(tuple(wanted))
                cells += len(wanted)
            # The report JSON is serialized with sorted object keys, whereas
            # its original in-memory Metrics traversal preserves insertion
            # order. Preserve every literal row and multiplicity, permitting
            # only that representational row-order difference in Metrics.
            _require((Counter(actual_rows) == Counter(expected_rows) if name == "Metrics" else
                      actual_rows == expected_rows), "Excel cell differs from source-bound report: " + name)
    return dict(status="verified", cells=cells, sheets=len(tables), sha256=source["sha256"])


def _audit(output, ledger, reader):
    budget, ledger_source = reader.read(ledger)
    _require("active" in budget, "ledger lacks explicit active state")
    if budget["active"] is not None:
        return dict(status="pending", all_phases_terminal=False, reason="GPU allocation remains unsettled")
    missing = [p for p in PHASES if not (output/p/"terminal.json").is_file()]
    if missing:
        return dict(status="pending", all_phases_terminal=False, missing_phases=missing, reason="phase conclusions are not all terminal")
    study, unused = reader.read(output/"study.json")
    workloads, workload_source = reader.read(output/"workloads.json")
    _require(digest(workloads) == study["workloads_sha256"], "frozen workloads changed")
    _require(budget["config_sha256"] == study["config_sha256"] and budget.get("budget_unlimited") is True
        and budget.get("limit_gpu_seconds", 0) is None and budget["authorization_binding"] == study["authorization"], "ledger protocol identity differs")
    auth = reader.bound(study["authorization"])
    _require(auth.get("approved_by_user") is True and auth.get("scope") == "serving_next_validation"
        and auth.get("budget_unlimited") is True and auth.get("authorized_gpu_seconds", 0) is None,
        "authorization does not permit this study")
    protocol, unused = reader.read(output/"protocol.json")
    _require(protocol.get("sha256") == study["config_sha256"] == digest(protocol["config"]),
             "root protocol configuration binding differs")
    parents = study["historical_ledgers"]
    _require(len(parents) == 3 and len({p["path"] for p in parents}) == 3 and
        {p["path"]:p["sha256"] for p in parents} ==
        {p["path"]:p["sha256"] for p in budget["parents"].values()}, "three historical ledger bindings differ")
    for parent in parents:
        _require(reader.bound(parent).get("active", "missing") is None, "historical ledger remains active")
    baseline = reader.bound(study["online_baseline_binding"])
    _require(all(baseline.get(k) is True for k in ("archive_verified","matrix_complete","online_evaluation_complete","online_qualified")),
             "inherited online baseline lacks qualification")
    cache_binding = study["cache_local_binding"]
    mathematics = cache_binding["mathematical_source_files"]
    _require(bool(mathematics) and digest(mathematics) == cache_binding["mathematical_source_sha256"]
        and bool(cache_binding["bindings"]), "inherited cache mathematical/evidence binding missing")
    for binding in cache_binding["bindings"].values():
        reader.bound(binding)
    sessions = budget["sessions"]
    _require(bool(sessions) and len({s["session_id"] for s in sessions}) == len(sessions), "missing/duplicate charged session")
    total, stage_totals, attempts = 0., {p:0. for p in PHASES}, {}
    previous_end, previous_stage = None, -1
    for session in sessions:
        phase = session["stage"]
        gpu_count = session.get("gpu_count")
        _require(phase in PHASES and session.get("status") in ("completed","failed","incomplete","budget_stopped","blocked_contention")
            and type(gpu_count) is int and (gpu_count == 1 or phase == "conditional" and gpu_count == 2),
            "invalid settled session/device count")
        _require(PHASES.index(phase) >= previous_stage, "settled phases moved backwards")
        previous_stage = PHASES.index(phase)
        start, end = session["started_monotonic"], session["ended_monotonic"]
        elapsed = session["elapsed_seconds"]
        _require(all(_number(v) for v in (start,end,elapsed,session["gpu_seconds"])) and
            elapsed == end-start and session["gpu_seconds"] == elapsed*gpu_count and elapsed >= 0 and
            (previous_end is None or start >= previous_end), "session does not charge full serial lifetime")
        previous_end = end
        total += session["gpu_seconds"]
        stage_totals[phase] += session["gpu_seconds"]
        attempt, config, snapshot = _snapshot(reader, output, session, study["config_sha256"], study["workloads_sha256"], mathematics)
        devices = [config["gpu_uuid"]]
        if gpu_count == 2:
            devices.append(config["v100_uuid"])
            _require(session["identity"].get("auxiliary_encoder") is True and devices[0] != devices[1],
                     "two-device conditional requires explicit auxiliary encoder admission")
        else:
            _require(not session["identity"].get("auxiliary_encoder"), "auxiliary encoder is not charged as two devices")
        _require(session["identity"].get("gpu_uuids") == devices and
            session["identity"].get("authorization_source") == study["authorization"], "session GPU/authorization differs")
        attempt_study, unused = reader.read(attempt/"study.json")
        _require(attempt_study == study, "attempt study changed")
        supervisor, supervisor_source = reader.read(attempt/"supervisor.json")
        _require(supervisor.get("charge") == session and supervisor.get("phase") == phase
            and supervisor.get("output") == str(attempt) and supervisor.get("status") in TERMINAL,
            "supervisor does not match exact ledger charge")
        _require(supervisor.get("source_sha256") == session["identity"]["source_sha256"] and
                 supervisor.get("config_sha256") == study["config_sha256"], "supervisor frozen source/config differs")
        reader.read(attempt/FILES.get(phase, phase+".json"))
        final = supervisor.get("final_occupancy", {})
        _require(final.get("status") == "exclusive_observed" and final.get("owned") == [] and final.get("foreign") == [],
            "owned/foreign process cleanup unconfirmed")
        _require(not supervisor.get("online_qualified") or
            (session["status"] == "completed" and supervisor.get("returncode") == 0 and not supervisor.get("supervisor_failure")),
            "failed supervisor claims online qualification")
        attempts[str(attempt)] = (session, config, supervisor, supervisor_source)
    _require(total == budget["used_gpu_seconds"] and stage_totals == budget["stage_used_gpu_seconds"], "ledger total/stage charges differ")
    summaries = []
    for phase in PHASES:
        terminal, terminal_source = reader.read(output/phase/"terminal.json")
        _require(terminal.get("status") in TERMINAL and terminal.get("output") in attempts, "unsettled or uncharged terminal")
        session, config, supervisor, supervisor_source = attempts[terminal["output"]]
        _require(terminal == supervisor and session["stage"] == phase, "terminal differs from settled supervisor")
        _require(session == [s for s in sessions if s["stage"] == phase][-1], "terminal does not select latest settled attempt")
        attempt = Path(terminal["output"])
        evidence_path = Path(terminal.get("evidence_path", attempt/FILES.get(phase, phase+".json")))
        _require(evidence_path == attempt/FILES.get(phase, phase+".json"), "unexpected native phase evidence path")
        data, source = reader.read(evidence_path)
        summary = dict(phase=phase, status=terminal["status"], source=source, terminal_source=terminal_source,
                       online_qualified=False, scope="settled integrity, not automatic scientific qualification")
        if phase == "diagnosis" or phase.startswith("cache_load_"):
            summary["matrix"] = _matrix(_with_partial(data, reader, attempt), _expected(workloads, phase), diagnosis=phase == "diagnosis")
            if phase != "diagnosis":
                _require(bool(data.get("online_qualified")) == summary["matrix"]["online_qualified"], "cache qualification differs")
                summary["online_qualified"] = summary["matrix"]["online_qualified"]
        elif phase in ("sana_vae","sana_graph"):
            summary["sana"] = _sana(data, config, reader, attempt, workloads)
            summary["online_qualified"] = summary["sana"]["online_qualified"]
        elif phase == "sana_numerics":
            _require(not any(data.get(k) for k in ("performance_claims","changes_serving_default","local_qualified","online_qualified")),
                "numerical diagnosis improperly grants serving qualification")
            rows = data.get("rows", [])
            _require([r["unit_index"] for r in rows] == list(range(len(rows))) and len(rows) <= 12, "numerical unit denominator differs")
            if data.get("diagnostic_complete"):
                _require(len(rows) == 12 and sum(len(r["input_ids"]) for r in rows) == 16,
                         "complete numerics lacks all 12 units/16 images")
            summary["numerics"] = dict(units=len(rows), required_units=12, performance_qualified=False)
        else:
            if any(k in data for k in ("video_vae","text_encoder")):
                summary["conditional"]=_conditional(data,session,reader,output,ledger,workloads,sessions)
            _require(not data.get("online_qualified"), "conditional conclusion cannot silently qualify a backend")
        supervised = session["status"] == "completed" and terminal.get("returncode") == 0 and not terminal.get("supervisor_failure")
        _require(bool(terminal.get("online_qualified")) == bool(supervised and summary["online_qualified"]), "terminal qualification differs from independently recomputed gate")
        summary["online_qualified"] = bool(supervised and summary["online_qualified"])
        summaries.append(summary)
    report, report_source = reader.read(output/"report.json")
    from .next_report import collect, _tables
    fresh = collect(output, ledger)
    _require({k:v for k,v in report.items() if k != "excel"} == fresh, "report is stale or differs from current source-bound collection")
    _require(not report.get("errors") and not any(p.get("errors") for p in report["phases"]), "report contains source integrity errors")
    for source in report["source_bindings"]:
        unused, actual = reader.raw(source["path"])
        _require(actual == source, "report source binding differs")
    excel = _excel(output/"results.xlsx", _tables(report), reader)
    _require(report.get("excel", {}).get("path") == str(output/"results.xlsx") and
        report["excel"].get("sha256") == excel["sha256"] and report["excel"].get("status") == "verified", "Excel report binding differs")
    verification, unused = reader.read(output/"excel-verification.json")
    _require(verification == report["excel"], "Excel verification receipt differs")
    # No evidence or ledger may change while the audit is reading other files.
    for path, expected in list(reader.sources.items()):
        unused, actual = reader.raw(path)
        _require(actual == expected, "source changed during final audit: " + path)
    return dict(status="passed", all_phases_terminal=True, report_sha256=report_source["sha256"], excel_sha256=excel["sha256"],
        ledger_source=ledger_source, workloads_source=workload_source, sessions=len(sessions),
        settled_gpu_seconds=total, stage_gpu_seconds=stage_totals, historical_parents=parents,
        phases=summaries, excel_verification=excel, deletion_authorized=False,
        dependency_closure_required=True,
        dependency_closure="separate explicit root decision required; this audit does not close continuation dependencies",
        limitations=["recorded finite numerical metrics verified; raw arrays not read, hashed or reevaluated",
                     "host request intervals and descriptive p95; no kernel-causality or stable-p99 claim"])


def final_audit(output, ledger):
    """Write audits/final-integrity.json; callers must accept only status passed."""
    output, ledger = Path(output).absolute(), Path(ledger).absolute()
    _require(output.resolve() == output and ledger.resolve() == ledger, "audit paths must not traverse symlinks")
    reader = Reader()
    try:
        result = _audit(output, ledger, reader)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, ET.ParseError) as error:
        result = dict(status="failed", all_phases_terminal=False, error=type(error).__name__+": "+str(error))
    result.update(schema_version="ditserve.optimizations.next_integrity.v1", output=str(output),
        ledger=str(ledger), source_bindings=list(reader.sources.values()), gpu_used=False,
        raw_arrays_read=False, originals_modified=False)
    atomic_json(output/"audits/final-integrity.json", result)
    return result
