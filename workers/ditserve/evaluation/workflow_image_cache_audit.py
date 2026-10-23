"""CPU-only replay of bounded Sana prompt-cache provenance.

This audit does not establish output equality, GPU admission, or a speedup.
Those gates belong to the paired experiment and its guarded launch evidence.
"""

from collections import OrderedDict
import copy
import hashlib
import math

from . import workflow_next_report as report


SCHEMA = "ditserve.workflow.image-cache-audit.v1"
RECEIPT_SCHEMA = "ditserve.sana.prompt-cache.v1"
LIMITS = dict(entries=4, bytes=16 * 1024**2)
COUNTERS = ("lookups", "hits", "misses", "insertions", "evictions", "bypasses", "errors", "clears")
OPTIONS = dict(quantum_ms=300, late_short_ms=2000, chunked_decode=True,
               decode_frames=1, demote_late=True)
SCOPE = ("Two-worker exact-prompt LRU replay including the one existing warmup prompt per worker; "
         "four recurring prompts only. No inference for unique prompts, another model, A100, "
         "or general production hit rates; output equality and guarded capacity are separate gates.")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _integer(value, *, positive=False):
    return type(value) is int and (value > 0 if positive else value >= 0)


def _key(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _state(cache):
    return dict(entries=len(cache), bytes=sum(row["bytes"] for row in cache.values()),
                lru=[dict(key=key, bytes=row["bytes"]) for key, row in cache.items()])


def _tensor_bytes(tensors):
    _require(isinstance(tensors, list) and len(tensors) == 2, "missing two retained tensors")
    _require([row["name"] for row in tensors] == ["embeddings", "attention_mask"], "wrong retained tensors")
    total = 0
    for tensor in tensors:
        shape = tensor["shape"]
        _require(isinstance(shape, list) and shape and all(_integer(x, positive=True) for x in shape),
                 "invalid tensor shape")
        _require(all(_integer(tensor[name], positive=True) for name in
                     ("numel", "element_size", "storage_bytes", "logical_bytes", "accounted_bytes")),
                 "invalid tensor byte evidence")
        _require(tensor["numel"] == math.prod(shape), "tensor numel differs from shape")
        _require(tensor["logical_bytes"] == tensor["numel"] * tensor["element_size"],
                 "tensor logical bytes differ from shape and element size")
        _require(tensor["accounted_bytes"] == max(tensor["storage_bytes"], tensor["logical_bytes"]),
                 "tensor storage is not conservatively accounted")
        total += tensor["accounted_bytes"]
    return total


def _configuration(result):
    _require(result.get("schema") == "ditserve.run.v1" and result.get("status") == "completed",
             "missing completed native run")
    config = result["config"]
    phases = config["phases"]
    observed = result["phases"]
    _require(isinstance(phases, list) and len(phases) == 2 and
             {p["id"] for p in phases} == {"reference", "candidate"}, "expected exactly two configured arms")
    _require(isinstance(observed, list) and len(observed) == 2 and
             [p["id"] for p in observed] == [p["id"] for p in phases], "phase order or multiplicity differs")
    normalized = []
    for phase in phases:
        _require(phase.get("kind") == "cluster" and phase.get("policy") == "block_elastic" and
                 phase.get("options") == OPTIONS, "fixed scheduling controls differ")
        _require(phase.get("text_prefetch_depth") == 1 and phase.get("output_mode") == "check" and
                 phase.get("record_dispatch_timeline") is True and not phase.get("real_sp"),
                 "fixed execution controls differ")
        _require(phase["gpus"] == [dict(video=f"wan{i}", image=f"sana{i}") for i in range(2)],
                 "expected two fixed worker slots")
        rows = phase["workload"]
        mapping = report.frozen_request_map(rows)
        _require(len(rows) == 24 and mapping is not None and len(mapping) == 24, "incomplete workload")
        images = [row for row in rows if row["kind"] == "image"]
        videos = [row for row in rows if row["kind"] == "video"]
        _require(len(images) == 20 and len(videos) == 4 and
                 sorted(row["spec"]["frames"] for row in videos) == [17, 17, 33, 33], "workload recipe differs")
        _require(all(row["spec"].get("steps") == 20 and row["spec"].get("height") == 480 and
                     row["spec"].get("width") == 832 and row["spec"].get("guidance_scale") == 5
                     for row in videos), "video recipe differs")
        _require(all(isinstance(row["spec"].get("prompt"), str) and
                     type(row["spec"].get("seed")) is int and _number(row["arrival_ms"]) and
                     _number(row["deadline_ms"]) and row["deadline_ms"] >= row["arrival_ms"] for row in rows),
                 "invalid frozen request")
        prompts = {row["spec"]["prompt"] for row in images}
        _require(len(prompts) == 4 and all(row["class"] == "sana-1024" for row in images),
                 "four-prompt image workload required")
        warmup = phase["warmup"]
        _require(len(warmup) == 4 and [row["worker"] for row in warmup] == ["wan0", "wan1", "sana0", "sana1"],
                 "unexpected warmup list or order")
        _require(all(row["spec"]["prompt"] in prompts and row["spec"].get("seed") == 7 for row in warmup[2:]) and
                 warmup[2]["spec"] == warmup[3]["spec"], "warmup prompt provenance differs")
        clean = copy.deepcopy(phase)
        clean.pop("id")
        overrides = clean["worker_overrides"]
        _require(set(overrides) == {"wan0", "wan1", "sana0", "sana1"}, "worker overrides incomplete")
        for slot in range(2):
            for kind, offset, threads in (("wan", 0, 8), ("sana", 16, 4)):
                settings = overrides[f"{kind}{slot}"]
                _require(settings.get("cpu_threads") == threads and settings.get("cpu_affinity") ==
                         list(range(slot * 60 + offset, slot * 60 + offset + threads)),
                         "fixed worker CPU controls differ")
        for name in ("sana0", "sana1"):
            _require(config["workers"][name]["backend"] == "sana", "image worker is not Sana")
            expected = {} if phase["id"] == "reference" else dict(
                prompt_cache_entries=LIMITS["entries"], prompt_cache_bytes=LIMITS["bytes"])
            _require(overrides[name].get("options") == expected, "cache option or limit differs")
            overrides[name].pop("options")
        normalized.append(clean)
    _require(normalized[0] == normalized[1], "arms differ outside the cache option")
    return phases, observed


def _replay(records, enabled):
    """Records are ordered independently from receipt sequence numbers."""
    cache = OrderedDict()
    counters = {key: 0 for key in COUNTERS}
    totals = {name: dict(count=0, total_ms=0.) for name in ("cold_service", "warm_service")}
    measured = dict(requests=0, hits=0, misses=0, bypasses=0, encode_ms=0., generation_ms=0.,
                    service_ms=0., lookup_ms=0., service_samples_ms=[], hit_service_samples_ms=[],
                    miss_service_samples_ms=[])
    owner = None
    peak_bytes = 0
    for sequence, record in enumerate(records, 1):
        receipt = record["receipt"]
        _require(isinstance(receipt, dict) and receipt.get("schema") == RECEIPT_SCHEMA, "missing cache receipt")
        _require(receipt.get("enabled") is enabled and receipt.get("sequence") == sequence and
                 type(receipt.get("sequence")) is int, "cache enablement or sequence differs")
        _require(receipt.get("request_id") == record["request_id"] and
                 receipt.get("prompt_sha256") == _key(record["prompt"]), "receipt request or exact prompt differs")
        adapter = receipt["adapter_id"]
        _require(isinstance(adapter, str) and len(adapter) >= 16 and
                 all(c in "0123456789abcdef" for c in adapter), "invalid adapter identity")
        owner = adapter if owner is None else owner
        _require(adapter == owner and receipt.get("owner_scope") == "adapter_instance" and
                 receipt.get("key_scope") == "exact_prompt_utf8" and
                 receipt.get("cumulative_includes_warmup") is True, "cache owner or provenance differs")
        _require(receipt.get("limits") == (LIMITS if enabled else dict(entries=0, bytes=0)), "cache limits differ")
        _require(receipt.get("before") == _state(cache), "cache before-state differs from replay")
        for name in ("encode_ms", "generation_ms", "service_ms", "lookup_ms"):
            _require(_number(receipt[name], positive=name in ("generation_ms", "service_ms")), "invalid cache timing")
        _require(receipt["service_ms"] == receipt["encode_ms"] + receipt["generation_ms"],
                 "encoding cost excluded from service")
        _require(record.get("model_ms", receipt["service_ms"]) == receipt["service_ms"], "model service differs")
        key = _key(record["prompt"])
        evicted = []
        admitted = False
        if not enabled:
            outcome = "disabled"
            _require(receipt.get("entry_bytes") == 0 and receipt.get("entry_tensors") == [] and
                     receipt.get("encode_called") is False and receipt["encode_ms"] == 0 and
                     receipt.get("bypass_reason") is None, "baseline is not the uncached pipeline")
        else:
            counters["lookups"] += 1
            size = _tensor_bytes(receipt["entry_tensors"])
            _require(type(receipt.get("entry_bytes")) is int and receipt["entry_bytes"] == size,
                     "retained tensor bytes differ")
            if key in cache:
                outcome = "hit"
                counters["hits"] += 1
                _require(cache[key] == dict(bytes=size, tensors=receipt["entry_tensors"]), "hit tensor metadata changed")
                cache.move_to_end(key)
                _require(receipt.get("encode_called") is False and receipt["encode_ms"] == 0 and
                         receipt.get("bypass_reason") is None, "cache hit encoded the prompt")
            else:
                outcome = "miss"
                counters["misses"] += 1
                _require(receipt.get("encode_called") is True and receipt["encode_ms"] > 0, "cache miss lacks encoding cost")
                if size <= LIMITS["bytes"]:
                    while len(cache) >= LIMITS["entries"] or _state(cache)["bytes"] + size > LIMITS["bytes"]:
                        old, row = cache.popitem(last=False)
                        evicted.append(dict(key=old, bytes=row["bytes"]))
                    cache[key] = dict(bytes=size, tensors=copy.deepcopy(receipt["entry_tensors"]))
                    counters["insertions"] += 1
                    counters["evictions"] += len(evicted)
                    admitted = True
                    _require(receipt.get("bypass_reason") is None, "fitting entry was bypassed")
                else:
                    counters["bypasses"] += 1
                    _require(receipt.get("bypass_reason") == "entry_exceeds_byte_limit", "oversized entry bypass differs")
            bucket = totals["warm_service" if outcome == "hit" else "cold_service"]
            bucket["count"] += 1
            bucket["total_ms"] += receipt["service_ms"]
        _require(receipt.get("outcome") == outcome and receipt.get("admitted") is admitted and
                 receipt.get("evicted") == evicted, "reported cache operation differs from LRU replay")
        _require(receipt.get("after") == _state(cache), "cache after-state differs from replay")
        _require(_state(cache)["entries"] <= LIMITS["entries"] and _state(cache)["bytes"] <= LIMITS["bytes"],
                 "cache exceeds entry or byte limit")
        actual_counters = receipt.get("counters", {})
        _require(set(actual_counters) == set(COUNTERS) and all(_integer(v) for v in actual_counters.values()) and
                 actual_counters == counters, "cumulative cache counters differ")
        for name, total in totals.items():
            actual = receipt.get(name, {})
            _require(_integer(actual.get("count")) and _number(actual.get("total_ms")) and actual == total,
                     "cumulative service timing differs")
        peak_bytes = max(peak_bytes, _state(cache)["bytes"])
        if not record["warmup"]:
            measured["requests"] += 1
            measured["hits"] += outcome == "hit"
            measured["misses"] += outcome == "miss"
            measured["bypasses"] += enabled and receipt.get("bypass_reason") is not None
            for name in ("encode_ms", "generation_ms", "service_ms", "lookup_ms"):
                measured[name] += receipt[name]
            measured["service_samples_ms"].append(receipt["service_ms"])
            if outcome in ("hit", "miss"):
                measured[outcome + "_service_samples_ms"].append(receipt["service_ms"])
    return dict(adapter_id=owner, total_requests=len(records), warmup_requests=sum(r["warmup"] for r in records),
                counters=counters, peak_retained_bytes=peak_bytes, final_state=_state(cache),
                measured=measured, **totals)


def _arm(phase, observed):
    _require(observed.get("status") == "completed" and len(observed["cells"]) == 1, "incomplete arm")
    cell = observed["cells"][0]
    # Native cells may also carry a separate cost-model replay under `simulated`.
    # Qualification always reads the measured request rows and real-backend flag.
    _require(cell.get("status") == "completed" and cell.get("real_backends_pinned_sp1") is True and
             cell.get("real_sp") is None, "missing real completed SP1 cell")
    _require(cell.get("policy") == phase["policy"] and cell.get("gpus") == 2 and
             cell.get("text_prefetch_depth") == 1, "cell execution controls differ")
    options = dict(phase["options"], degrees=[1], pin_started=True)
    _require(cell.get("scheduler_options") == options, "actual scheduler options differ")
    rows = cell["requests"]
    _require(len(rows) == 24 and report.frozen_request_map(rows) == report.frozen_request_map(phase["workload"]),
             "measured requests differ from frozen workload")
    expected = {row["request_id"]: row for row in phase["workload"]}
    _require(all(row.get("status") == "completed" and
                 all(row.get(key) == expected[row["request_id"]].get(key) for key in ("kind", "class")) for row in rows),
             "incomplete or reclassified request")
    for row in rows:
        if row["kind"] != "video":
            continue
        cost = phase["profile"]["classes"][row["class"]]
        quantum = max(1, min(math.floor(300 / (cost["step_compute_ms"] / cost["blocks_per_step"])),
                             cost["blocks_per_step"]))
        blocks, chunks = 0, 0
        timeline = row["dispatch_timeline"]
        _require(isinstance(timeline, list) and timeline, "missing measured video dispatches")
        for event in timeline:
            _require(event.get("group") in ([0], [1]) and _number(event["start_ms"]) and
                     _number(event["finish_ms"]) and event["finish_ms"] >= event["start_ms"],
                     "invalid SP1 video dispatch")
            if event["quantum"] == "finish":
                _require(event.get("decode_frames") == 1, "decode quantum differs")
                chunks += 1
            else:
                _require(type(event["quantum"]) is int and event["quantum"] == quantum and
                         event.get("block_count") == quantum and event.get("chosen_quantum_ms") == 300 and
                         _integer(event.get("executed_blocks"), positive=True) and
                         event["executed_blocks"] <= quantum, "fixed video quantum or actual work differs")
                blocks += event["executed_blocks"]
        _require(blocks == row["spec"]["steps"] * cost["blocks_per_step"] and chunks == cost["decode_chunks"],
                 "incomplete actual video work")
    warmed = cell["warmup"]
    _require(len(warmed) == 4 and [row["worker"] for row in warmed] == [row["worker"] for row in phase["warmup"]],
             "warmup evidence differs")
    workers = {}
    for slot, name in enumerate(("sana0", "sana1")):
        warm_index = slot + 2
        warm = warmed[warm_index]
        _require(_number(warm["load_ms"]) and _number(warm["elapsed_ms"], positive=True), "invalid warmup timing")
        records = [dict(receipt=warm["prompt_cache"], request_id=f"warmup-{warm_index}",
                        prompt=phase["warmup"][warm_index]["spec"]["prompt"], warmup=True)]
        selected = []
        pid = cell["worker_runtime"][name]["pid"]
        _require(_integer(pid, positive=True), "missing image worker PID")
        runtime = cell["worker_runtime"][name]
        binding = runtime["cpu_binding"]
        settings = phase["worker_overrides"][name]
        _require(runtime.get("backend") == "sana" and runtime.get("measured") is True and
                 binding.get("configured") is True and binding.get("requested_affinity") == settings["cpu_affinity"] and
                 binding.get("effective_affinity") == settings["cpu_affinity"] and
                 binding.get("requested_threads") == 4 and binding.get("torch_threads") == 4,
                 "actual image worker CPU controls differ")
        for row in rows:
            if row["kind"] != "image":
                continue
            timeline = row["dispatch_timeline"]
            _require(len(timeline) == 1 and timeline[0].get("quantum") == "atomic" and
                     timeline[0].get("group") in ([0], [1]), "image dispatch is not one atomic single-GPU operation")
            if timeline[0]["group"] != [slot]:
                continue
            event = timeline[0]
            _require(all(_number(value) for value in
                         (row["start_ms"], row["finish_ms"], event["start_ms"], event["finish_ms"])) and
                     row["arrival_ms"] <= row["start_ms"] <= event["start_ms"] <= event["finish_ms"] <= row["finish_ms"],
                     "invalid request or dispatch timing")
            _require(row.get("worker_pids") == {name: pid}, "image worker identity differs from dispatched GPU")
            _require(row.get("operation_counts") == dict(prepare=1, advance=1, finish=1, release=1) and
                     row.get("quanta") == 1 and row.get("degrees") == [1], "image execution counts differ")
            _require(len(row["outputs"]) == 1, "missing unique final image response")
            _require(_number(row["model_ms"], positive=True) and
                     row["model_ms"] <= event["finish_ms"] - event["start_ms"], "model service exceeds host dispatch")
            selected.append(row)
        selected.sort(key=lambda row: (row["start_ms"], row["request_id"]))
        _require(all(a["finish_ms"] <= b["start_ms"] for a, b in zip(selected, selected[1:])),
                 "same-worker image service overlaps")
        for row in selected:
            records.append(dict(receipt=row["outputs"][0]["prompt_cache"], request_id=row["request_id"],
                                prompt=row["spec"]["prompt"], warmup=False, model_ms=row["model_ms"]))
        workers[name] = _replay(records, phase["id"] == "candidate")
    _require(sum(worker["measured"]["requests"] for worker in workers.values()) == 20,
             "cache audit omitted image requests")
    hits = sum(worker["measured"]["hits"] for worker in workers.values())
    _require(phase["id"] != "candidate" or hits > 0, "candidate has no observed cache hits")
    return dict(workers=workers, measured_hits=hits, measured_images=20, unique_prompts=4,
                timing_scope="Model service equals synchronized encoding on misses plus generation; "
                             "lookup is recorded separately; arrival-to-finish includes cache/dispatch/output overhead")


def audit(result):
    """Fail closed on absent, malformed, contradictory or incomplete evidence."""
    errors, arms = [], {}
    try:
        phases, observed = _configuration(result)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError) as error:
        return dict(schema=SCHEMA, passed=False, errors=["configuration: " + str(error)], arms={}, scope=SCOPE)
    for phase, actual in zip(phases, observed):
        try:
            arms[phase["id"]] = _arm(phase, actual)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError) as error:
            errors.append(phase["id"] + ": " + str(error))
    owners = [worker["adapter_id"] for arm in arms.values() for worker in arm["workers"].values()]
    if len(arms) == 2 and len(set(owners)) != 4:
        errors.append("cache adapter identities were shared across workers or arms")
    return dict(schema=SCHEMA, passed=not errors and len(arms) == 2, errors=errors, arms=arms, scope=SCOPE)
