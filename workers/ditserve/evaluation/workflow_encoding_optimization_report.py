"""Read-only, fail-closed audits of the frozen encoding-cache experiment.

Numerical diagnostics and normal online timing are separate evidence. Reported
changes are descriptive paired observations, never automatic deployment gates.
"""

import copy
import math
from pathlib import Path
import statistics

from . import workflow_encoding_qualification as qualification
from . import workflow_encoding_optimization_study as study_api
from . import workflow_image_cache_audit as image_audit
from . import workflow_next_report as report
from .workflow_next_study import read
from .workflow_serving_study import digest


SCHEMA = "ditserve.workflow.encoding-optimization-report.v1"
_require = image_audit._require
_number = image_audit._number


def _configuration(result, job, frozen):
    metadata = job["metadata"]
    _require(result.get("schema") == "ditserve.run.v1" and result.get("status") == "completed",
             "missing completed native result")
    phases = frozen["phases"]
    _require(result.get("config", {}).get("phases") == phases, "result phases differ from frozen input")
    expected = study_api.native_online_config(phases[0]["profile"], phases[0]["workload"],
        optimization=metadata["optimization"], prompt_distribution=metadata["prompt_distribution"],
        repeat=metadata["repeat"])
    _require(frozen["workers"] == expected["workers"], "frozen worker definitions differ")
    clean = copy.deepcopy(phases)
    for phase in clean:
        phase.pop("artifact_dependencies", None)
    _require(clean == expected["phases"], "frozen configuration differs from controlled matrix")
    observed = result["phases"]
    _require([p["id"] for p in observed] == [p["id"] for p in phases], "missing or reordered arms")
    images = [r for r in phases[0]["workload"] if r["kind"] == "image"]
    distribution = metadata["prompt_distribution"]
    if metadata["optimization"] == "sana_prompt_cache":
        _require(len({r["spec"]["prompt"] for r in images}) == (20 if distribution == "unique" else 4),
                 "image prompt distribution differs")
    return phases, observed


def _cell(phase, observed):
    _require(observed.get("status") == "completed" and len(observed["cells"]) == 1, "incomplete arm")
    cell = observed["cells"][0]
    _require(cell.get("status") == "completed" and cell.get("real_backends_pinned_sp1") is True
             and cell.get("real_sp") is None and cell.get("gpus") == 2
             and cell.get("policy") == "block_elastic" and cell.get("text_prefetch_depth") == 1,
             "missing measured pinned-SP1 execution")
    _require(cell.get("scheduler_options") == dict(phase["options"], degrees=[1], pin_started=True),
             "actual scheduler controls differ")
    _require(_number(cell["elapsed_ms"], positive=True), "invalid normal timing interval")
    rows = cell["requests"]
    _require(len(rows) == 24 and report.frozen_request_map(rows) == report.frozen_request_map(phase["workload"]),
             "actual requests differ from frozen workload")
    planned = {row["request_id"]: row for row in phase["workload"]}
    for row in rows:
        _require(row.get("status") == "completed" and all(row.get(k) == planned[row["request_id"]][k]
                 for k in ("kind", "class")), "unfinished or reclassified request")
        _require(all(_number(row[k]) for k in ("arrival_ms", "start_ms", "finish_ms", "deadline_ms"))
                 and row["arrival_ms"] <= row["start_ms"] <= row["finish_ms"], "invalid request timing")
        events = row["dispatch_timeline"]
        _require(events and all(e.get("group") in ([0], [1]) and _number(e["start_ms"])
                 and _number(e["finish_ms"]) and row["start_ms"] <= e["start_ms"] <= e["finish_ms"] <= row["finish_ms"]
                 for e in events), "missing or invalid actual dispatch timeline")
        _require(all(e["group"] == events[0]["group"] for e in events), "pinned request migrated GPUs")
        slot = events[0]["group"][0]
        worker = ("sana" if row["kind"] == "image" else "wan") + str(slot)
        _require(row.get("worker_pids") == {worker: cell["worker_runtime"][worker]["pid"]},
                 "request worker differs from dispatched GPU")
        if row["kind"] == "image":
            _require(len(events) == 1 and events[0].get("quantum") == "atomic"
                     and row.get("operation_counts") == dict(prepare=1, advance=1, finish=1, release=1)
                     and row.get("quanta") == 1 and row.get("degrees") == [1]
                     and len(row["outputs"]) == 1, "image lifecycle differs")
            _require(_number(row["model_ms"], positive=True)
                     and row["model_ms"] <= events[0]["finish_ms"] - events[0]["start_ms"],
                     "model service exceeds dispatch")
        else:
            cost = phase["profile"]["classes"][row["class"]]
            quantum = max(1, min(math.floor(300 / (cost["step_compute_ms"] / cost["blocks_per_step"])),
                                 cost["blocks_per_step"]))
            blocks = chunks = 0
            for event in events:
                if event["quantum"] == "finish":
                    _require(event.get("decode_frames") == 1, "decode quantum differs")
                    chunks += 1
                else:
                    _require(type(event["quantum"]) is int and event["quantum"] == quantum
                             and event.get("block_count") == quantum and event.get("chosen_quantum_ms") == 300
                             and type(event.get("executed_blocks")) is int and 0 < event["executed_blocks"] <= quantum,
                             "video block quantum differs")
                    blocks += event["executed_blocks"]
            _require(blocks == row["spec"]["steps"] * cost["blocks_per_step"] and chunks == cost["decode_chunks"],
                     "actual video work is incomplete")
    _require(len(cell["warmup"]) == 4 and [r["worker"] for r in cell["warmup"]] ==
             [r["worker"] for r in phase["warmup"]], "warmup worker provenance differs")
    for name, runtime in cell["worker_runtime"].items():
        settings = phase["worker_overrides"][name]
        threads = 4 if name.startswith("sana") else 8
        binding = runtime["cpu_binding"]
        _require(type(runtime["pid"]) is int and runtime["pid"] > 0 and runtime.get("measured") is True
                 and runtime.get("backend") == ("sana" if threads == 4 else "wan")
                 and binding.get("configured") is True
                 and binding.get("requested_affinity") == settings["cpu_affinity"]
                 and binding.get("effective_affinity") == settings["cpu_affinity"]
                 and binding.get("requested_threads") == threads and binding.get("torch_threads") == threads,
                 "actual worker CPU or process identity differs")
    _require(set(cell["worker_runtime"]) == {"sana0", "sana1", "wan0", "wan1"}, "missing worker runtime")
    return cell


def _sana_cache(phase, cell, optimization):
    enabled = optimization == "sana_prompt_cache" and phase["id"] != "disabled"
    workers = {}
    for slot in (0, 1):
        name, index = f"sana{slot}", slot + 2
        warm = cell["warmup"][index]
        _require(_number(warm["load_ms"]) and _number(warm["elapsed_ms"], positive=True), "invalid warmup timing")
        records = [dict(receipt=warm["prompt_cache"], request_id=f"warmup-{index}",
                        prompt=phase["warmup"][index]["spec"]["prompt"], warmup=True)]
        selected = sorted((r for r in cell["requests"] if r["kind"] == "image"
                           and r["dispatch_timeline"][0]["group"] == [slot]), key=lambda r: (r["start_ms"], r["request_id"]))
        _require(all(a["finish_ms"] <= b["start_ms"] for a, b in zip(selected, selected[1:])),
                 "same-worker image operations overlap")
        records.extend(dict(receipt=r["outputs"][0]["prompt_cache"], request_id=r["request_id"],
            prompt=r["spec"]["prompt"], warmup=False, model_ms=r["model_ms"]) for r in selected)
        workers[name] = image_audit._replay(records, enabled)
        if enabled:
            for record in records:
                receipt = record["receipt"]
                _require(receipt.get("sync_mode", "staged") == phase["id"], "Sana synchronization mode differs")
                if phase["id"] == "fused" and receipt["outcome"] == "miss":
                    _require(receipt.get("timing_scope") == "fused_miss_host_wall", "fused miss timing scope missing")
    return workers


def _wan_cache(phase, cell):
    arm = phase["id"]
    workers = {}
    for slot in (0, 1):
        name = f"wan{slot}"
        runtime = cell["worker_runtime"][name]
        warm = cell["warmup"][slot]
        _require(warm.get("worker_pid") == runtime["pid"] and warm.get("state_id") == f"warmup-{slot}"
                 and warm.get("spec") == dict(phase["warmup"][slot]["spec"], request_id=f"warmup-{slot}"),
                 "Wan warmup process or request identity differs")
        selected = [r for r in cell["requests"] if r["kind"] == "video"
                    and r["dispatch_timeline"][0]["group"] == [slot]]
        if arm == "enabled":
            selected.sort(key=lambda r: r["negative_prompt_cache"]["sequence"])
        else:
            selected.sort(key=lambda r: r["start_ms"])
        records = [dict(warm, request_id=f"warmup-{slot}")] + selected
        previous, peak = None, 0
        totals = dict(requests=0, hits=0, misses=0, bypasses=0, cpu_prompt_encode_ms=0.,
                      encode_call_ms=0., lookup_ms=0., clone_ms=0., store_ms=0.)
        for index, row in enumerate(records):
            events = row["encoding_events"]
            _require(isinstance(events, list) and [e["op"] for e in events] in
                     (["prepare"], ["encode_poll", "prepare"]), "unexpected encoding lifecycle")
            for event in events:
                _require(event.get("worker") == name and event.get("pid") == runtime["pid"]
                         and event.get("negative_prompt_cache") == row.get("negative_prompt_cache")
                         and event.get("cpu_prompt_encode_ms") == row["cpu_prompt_encode_ms"],
                         "encoding receipt differs between worker, prefetch and prepare")
            if len(events) == 2:
                _require(events[0].get("ticket") and events[0]["ticket"] == events[1].get("ticket"),
                         "prefetch ticket was not consumed by prepare")
            _require(_number(row["cpu_prompt_encode_ms"], positive=True), "CPU encode time missing")
            receipt = qualification._check_receipt("wan", arm,
                dict(cache=row.get("negative_prompt_cache"), request_id=row["request_id"], spec=row["spec"]),
                dict(model_path="/models/wan2.1-t2v-1.3b"), previous)
            if receipt is not None:
                _require(receipt.get("cumulative_includes_warmup") is True
                         and receipt.get("cpu_prompt_encode_ms") == row["cpu_prompt_encode_ms"],
                         "Wan cache omits warmup or CPU encoding cost")
                for metric in ("encode_call_ms", "lookup_ms", "clone_ms", "store_ms"):
                    _require(_number(receipt[metric]), "invalid Wan cache timing")
                peak = max(peak, receipt["after"]["bytes"])
            previous = receipt
            if index:
                totals["requests"] += 1
                totals["cpu_prompt_encode_ms"] += row["cpu_prompt_encode_ms"]
                if receipt:
                    for outcome in ("hit", "miss", "bypass"):
                        totals[{"hit": "hits", "miss": "misses", "bypass": "bypasses"}[outcome]] += receipt["outcome"] == outcome
                    for metric in ("encode_call_ms", "lookup_ms", "clone_ms", "store_ms"):
                        totals[metric] += receipt[metric]
        workers[name] = dict(instance_id=previous["instance_id"] if previous else None,
            warmup_requests=1, measured=totals, peak_retained_bytes=peak,
            counters=previous["counters"] if previous else None,
            final_state=previous["after"] if previous else dict(entries=0, bytes=0, lru=[]))
    return workers


def audit_result(result, job, frozen_config):
    """Recompute numerical/cache evidence; absent and malformed data fail closed."""
    checked = dict(passed=False, errors=[], arms={}, exact={}, performance_evidence=False)
    try:
        _require(isinstance(frozen_config, dict) and digest(frozen_config) == job["configuration_sha256"],
                 "input configuration digest differs")
        metadata = job["metadata"]
        if metadata["role"] == "encoding_numeric":
            config = qualification.normalize_config(frozen_config["kind"], frozen_config)
            _require(result.get("schema") == qualification.SCHEMA and result.get("status") == "completed"
                     and result.get("kind") == config["kind"] and result.get("config") == config
                     and result.get("performance_evidence") is False, "numerical result configuration differs")
            numeric = qualification.qualify(config["kind"], config, result["arms"])
            checked.update(passed=numeric["passed"], numerical=numeric, raw_capacity=numeric["passed"],
                           errors=numeric["failures"])
            return checked
        _require(metadata["role"] == "encoding_online", "unsupported encoding job role")
        phases, observed = _configuration(result, job, frozen_config)
        cells = {}
        owners = []
        for phase, actual in zip(phases, observed):
            cell = _cell(phase, actual)
            cells[phase["id"]] = cell
            sana = _sana_cache(phase, cell, metadata["optimization"])
            wan = _wan_cache(phase, cell) if metadata["optimization"] == "wan_negative_cache" else {}
            checked["arms"][phase["id"]] = dict(sana=sana, wan=wan)
            owners.extend(w["adapter_id"] for w in sana.values())
            owners.extend(w["instance_id"] for w in wan.values() if w["instance_id"] is not None)
        _require(len(set(owners)) == len(owners), "cache owner reused across workers or arms")
        for candidate in cells:
            if candidate != "disabled":
                paired = report.exact_pair(cells["disabled"]["requests"], cells[candidate]["requests"])
                checked["exact"]["disabled/" + candidate] = paired
                _require(paired["passed"], "paired final image/video or video latent hashes differ")
        checked.update(passed=True, performance_evidence=True)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError) as error:
        checked["errors"].append(str(error))
    return checked


def validate_job(study, job, result):
    """Adapter for the central report dispatcher; does not call evidence()."""
    try:
        return audit_result(result, job, read(job["input"], {}))
    except (OSError, ValueError, TypeError) as error:
        return dict(passed=False, errors=[str(error)])


def raw_capacity(launch, job):
    """Replay the launcher's raw sampled NVML capacity and check its summary."""
    checked = dict(passed=False, devices={}, errors=[], scope="observed sampled capacity; peaks between polls are unobserved")
    try:
        monitor = launch["gpu_monitor"]
        mapping = {r["ordinal"]: r["uuid"] for r in job["gpu_mapping"]}
        _require(set(mapping) == set(job["devices"]), "GPU mapping differs")
        samples = monitor["samples"]
        _require(isinstance(samples, list) and samples and all(r["device"] in mapping for r in samples),
                 "raw NVML samples missing or include another GPU")
        for device in job["devices"]:
            rows = [r for r in samples if r["device"] == device]
            _require(len(rows) >= 2 and all(r["uuid"] == mapping[device] for r in rows), "raw GPU identity differs")
            _require(all(type(r["timestamp_ns"]) is int and r["timestamp_ns"] > 0 for r in rows)
                     and all(a["timestamp_ns"] < b["timestamp_ns"] for a, b in zip(rows, rows[1:])),
                     "raw sample timestamps are missing or reordered")
            for row in rows:
                _require(all(type(row[k]) is int and row[k] >= 0 for k in
                             ("memory_total_bytes", "memory_used_bytes", "memory_free_bytes"))
                         and row["memory_total_bytes"] > 0
                         and row["memory_total_bytes"] == row["memory_used_bytes"] + row["memory_free_bytes"],
                         "invalid raw NVML memory sample")
            total = rows[0]["memory_total_bytes"]
            _require(all(r["memory_total_bytes"] == total for r in rows), "GPU capacity changed")
            observed = dict(samples=len(rows), total_bytes=total,
                peak_used_bytes=max(r["memory_used_bytes"] for r in rows),
                minimum_free_bytes=min(r["memory_free_bytes"] for r in rows))
            _require(monitor["memory"][str(device)] == observed, "NVML aggregate differs from raw samples")
            _require(observed["minimum_free_bytes"] >= max(1024**3, total * .05), "observed memory margin below existing gate")
            checked["devices"][str(device)] = dict(observed, last_used_bytes=rows[-1]["memory_used_bytes"])
        checked.update(passed=True, interval_seconds=monitor.get("interval_seconds"))
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        checked["errors"].append(str(error))
    return checked


def _metrics(cell, audit):
    measured = report.summary(cell["requests"], cell["elapsed_ms"])
    for kind in ("total", "image", "video"):
        rows = [r for r in cell["requests"] if kind == "total" or r["kind"] == kind]
        waits = [r["start_ms"] - r["arrival_ms"] for r in rows]
        measured[kind].update(wait_p50_ms=report.percentile(waits, .5), wait_p95_ms=report.percentile(waits, .95))
    measured["cache"] = {}
    for kind in ("sana", "wan"):
        workers = audit[kind]
        if workers:
            measured["cache"][kind] = dict(
                hits=sum(w["measured"]["hits"] for w in workers.values()),
                misses=sum(w["measured"]["misses"] for w in workers.values()),
                bypasses=sum(w["measured"]["bypasses"] for w in workers.values()),
                peak_retained_bytes_per_worker={name: w["peak_retained_bytes"] for name, w in workers.items()},
                maximum_worker_retained_bytes=max(w["peak_retained_bytes"] for w in workers.values()),
                encode_ms=sum(w["measured"]["encode_ms" if kind == "sana" else "cpu_prompt_encode_ms"] for w in workers.values()))
    return measured


def _median_tree(values):
    if not values:
        return None
    if all(type(value) in (int, float) for value in values):
        return statistics.median(values)
    if all(isinstance(value, dict) for value in values):
        common = set.intersection(*(set(value) for value in values))
        return {key: _median_tree([value[key] for value in values]) for key in sorted(common)}
    return None


def _paired_changes(arms):
    pairs = [("disabled", candidate) for candidate in arms if candidate != "disabled"]
    if "staged" in arms and "fused" in arms:
        pairs.append(("staged", "fused"))
    changes = {}
    for reference, candidate in pairs:
        grouped = {}
        for kind in ("total", "image", "video"):
            metrics = {}
            for name in ("latency_p50_ms", "latency_p95_ms", "wait_p50_ms", "wait_p95_ms", "throughput_per_second"):
                baseline, value = arms[reference][kind][name], arms[candidate][kind][name]
                delta = value - baseline
                metrics[name] = dict(reference=baseline, candidate=value, candidate_minus_reference=delta,
                    improvement_percent=(delta if name == "throughput_per_second" else -delta) * 100 / baseline
                    if baseline else None)
            baseline, value = arms[reference][kind]["slo_attainment"], arms[candidate][kind]["slo_attainment"]
            metrics["slo_attainment"] = dict(reference=baseline, candidate=value,
                change_percentage_points=(value - baseline) * 100)
            grouped[kind] = metrics
        changes[reference + "/" + candidate] = grouped
    return changes


def summarize(study):
    """Each seed and medians of the complete independent three-seed groups."""
    result = dict(schema=SCHEMA, numerical=[], per_seed=[], groups={}, default_change=False,
        timing="arrival to final output; queue wait is arrival to first dispatch start",
        encoding_scope="Sana staged and fused encode/service scopes differ; compare end-to-end metrics",
        scope="three independent trajectories per distribution; descriptive medians, no automatic performance threshold")
    jobs = [j for j in study["jobs"] if not j.get("superseded")
            and j.get("metadata", {}).get("role") in ("encoding_numeric", "encoding_online")]
    for job in jobs:
        metadata = job["metadata"]
        ev = report.evidence(study, job)
        audit = validate_job(study, job, ev["result"])
        capacity = raw_capacity(ev["launch"], job)
        common = dict(job_id=job["id"], seed=metadata["seed"], optimization=metadata["optimization"],
            prompt_distribution=metadata["prompt_distribution"], guarded=ev["guarded"],
            sampled_capacity=ev["capacity"], raw_capacity=capacity, audit=audit,
            qualified=bool(ev["qualified"] and ev["capacity"] and capacity["passed"] and audit["passed"]))
        if metadata["role"] == "encoding_numeric":
            result["numerical"].append(dict(common, performance_evidence=False))
            continue
        prerequisites = []
        for logical in job["requires"]:
            try:
                dependency = report.current_job(study, logical)
                gate = report.evidence(study, dependency)
                passed = gate["qualified"] and gate["capacity"]
                if dependency.get("metadata", {}).get("role") == "encoding_numeric":
                    passed = passed and raw_capacity(gate["launch"], dependency)["passed"]
                prerequisites.append(dict(job_id=dependency["id"], passed=bool(passed)))
            except (ValueError, KeyError, OSError, TypeError):
                prerequisites.append(dict(job_id=logical, passed=False))
        common.update(prerequisites=prerequisites, qualified=common["qualified"] and bool(prerequisites)
                      and all(row["passed"] for row in prerequisites), arms={}, paired_changes={})
        if audit["passed"]:
            common["arms"] = {name: _metrics(cell, audit["arms"][name])
                              for name, cell in report.native_arms(ev["result"]).items()}
            common["paired_changes"] = _paired_changes(common["arms"])
        result["per_seed"].append(common)
    for optimization, distribution, seeds in (("sana_prompt_cache", "unique", study_api.SANA_SEEDS),
            ("sana_prompt_cache", "repeated", study_api.SANA_SEEDS),
            ("wan_negative_cache", "mixed_negative", study_api.WAN_SEEDS)):
        rows = [r for r in result["per_seed"] if r["optimization"] == optimization
                and r["prompt_distribution"] == distribution]
        complete = len(rows) == 3 and {r["seed"] for r in rows} == set(seeds) and all(r["qualified"] for r in rows)
        result["groups"][optimization + "/" + distribution] = dict(qualified=complete,
            independent_seeds=list(seeds), observed_seeds=[r["seed"] for r in rows],
            median=_median_tree([r["arms"] for r in rows]) if complete else None,
            paired_changes_median=_median_tree([r["paired_changes"] for r in rows]) if complete else None)
    result["qualified"] = bool(jobs) and all(group["qualified"] for group in result["groups"].values())
    return result
