"""Strict cache-combination qualification and prespecified, paired performance gates."""

import statistics

from . import workflow_cache_combination_study as study_api
from . import workflow_dispatch_refresh_report as dispatch
from . import workflow_encoding_optimization_report as encoding
from . import workflow_next_report as report
from . import workflow_occupancy_report as occupancy
from .workflow_next_study import read
from .workflow_serving_study import digest


SCHEMA = "ditserve.workflow.cache-combination-report.v1"
ROLE = "cache_combination_online"
RULE = ("Screen: mean Sana worker-reported model/service time reduction >=5%; "
        "image/video p95 <=105% and total throughput >=97%. Confirmation: median paired seed "
        "mean-service reduction >=5%, at least 2/3 improving, every seed technical and safety passed. "
        "Separate image-tail benefit: median paired image p95 reduction >=5%, at least 2/3 improving. "
        "Screen excluded from confirmation; no automatic default change.")
TIMING_SCOPE = ("worker_model_ms is worker-reported Sana model/service time, including prompt encoding "
                "on misses; host_dispatch_ms spans dispatch start to final RPC return, including IPC "
                "and guard work, excluding later harvest; neither is end-to-end latency")
_require = encoding._require
_number = encoding._number


def _change(reference, candidate, higher_better=False):
    delta = candidate - reference
    return dict(reference=reference, candidate=candidate, candidate_minus_reference=delta,
                improvement_percent=(delta if higher_better else -delta) * 100 / reference if reference else None)


def _image_service(cell):
    rows = [r for r in cell["requests"] if r["kind"] == "image"]
    series = dict(worker_model_ms=[r["model_ms"] for r in rows],
        host_dispatch_ms=[r["dispatch_timeline"][0]["rpc_completed_ms"] - r["dispatch_timeline"][0]["start_ms"] for r in rows],
        queue_after_admission_ms=[r["dispatch_timeline"][0]["start_ms"] - r["admission_ms"] for r in rows],
        harvest_delay_ms=[r["dispatch_timeline"][0]["harvest_delay_ms"] for r in rows])
    _require(len(rows) == 20 and all(_number(v) for values in series.values() for v in values), "invalid image service samples")
    return dict(requests=len(rows), timing_scope=TIMING_SCOPE, **{
        name: dict(mean=sum(values) / len(values), p50=report.percentile(values, .5),
                   p95=report.percentile(values, .95), maximum=max(values)) for name, values in series.items()})


def _service_pairing(cells):
    left = {r["request_id"]: r for r in cells["reference"]["requests"] if r["kind"] == "image"}
    pairs = [(left[r["request_id"]], r) for r in cells["candidate"]["requests"] if r["kind"] == "image"]
    subsets = dict(all=pairs, same_gpu=[p for p in pairs if p[0]["dispatch_timeline"][0]["group"] == p[1]["dispatch_timeline"][0]["group"]])
    for name in ("all", "same_gpu"):
        for outcome in ("hit", "miss"):
            subsets[name + "_candidate_" + outcome] = [p for p in subsets[name] if p[1]["outputs"][0]["prompt_cache"]["outcome"] == outcome]
    result = {}
    for name, selected in subsets.items():
        result[name] = dict(requests=len(selected), worker_model_mean_ms=None, host_dispatch_mean_ms=None)
        if selected:
            model = [sum(pair[i]["model_ms"] for pair in selected) / len(selected) for i in (0, 1)]
            host = [sum(pair[i]["dispatch_timeline"][0]["rpc_completed_ms"] - pair[i]["dispatch_timeline"][0]["start_ms"]
                        for pair in selected) / len(selected) for i in (0, 1)]
            result[name].update(worker_model_mean_ms=_change(*model), host_dispatch_mean_ms=_change(*host))
    return dict(timing_scope=TIMING_SCOPE, subsets=result)


def audit_result(result, job, frozen):
    """Replay actual cache routing and both unchanged NVML guards; no file writes."""
    checked = dict(passed=False, errors=[], arms={}, exact={}, performance_evidence=False)
    try:
        _require(job["metadata"]["role"] == ROLE, "wrong cache-combination role")
        _require(digest(frozen) == job["configuration_sha256"], "input configuration digest differs")
        study_api.validate_config(frozen, repeat=job["metadata"]["repeat"])
        _require(result.get("schema") == "ditserve.run.v1" and result.get("status") == "completed"
                 and result.get("config", {}).get("phases") == frozen["phases"], "completed result or frozen phases differ")
        phases, observed = frozen["phases"], result["phases"]
        _require([p["id"] for p in observed] == [p["id"] for p in phases], "missing or reordered cache-combination arms")
        mapping = {entry["ordinal"]: entry["uuid"] for entry in job["gpu_mapping"]}
        _require(set(mapping) == {0, 1} and len(job["gpu_mapping"]) == 2, "bound GPU mapping differs")
        worker_uuids = {name: mapping[settings["gpu"]] for name, settings in frozen["workers"].items()}
        _require(all(result["config"]["workers"][name]["gpu_uuid"] == uuid for name, uuid in worker_uuids.items()),
                 "resolved GPU UUID differs from frozen mapping")
        cells, owners = {}, []
        for phase, actual in zip(phases, observed):
            name = phase["id"]
            cell = encoding._cell(phase, actual)
            cells[name] = cell
            for index in (2, 3):
                warm = cell["warmup"][index]
                worker = phase["warmup"][index]["worker"]
                _require(warm.get("worker_pid") == cell["worker_runtime"][worker]["pid"]
                         and warm.get("state_id") == f"warmup-{index}"
                         and warm.get("spec") == dict(phase["warmup"][index]["spec"], request_id=f"warmup-{index}"),
                         "Sana warmup worker or prompt provenance differs")
            sana = encoding._sana_cache(dict(phase, id="staged" if name == "candidate" else "disabled"), cell, "sana_prompt_cache")
            wan = encoding._wan_cache(dict(phase, id="enabled"), cell)
            _require(all(w["counters"][k] == 0 for w in sana.values() for k in ("errors", "bypasses", "evictions")),
                     "Sana cache bypassed, evicted, or failed on the four-prompt workload")
            if name == "candidate":
                hits = sum(w["measured"]["hits"] for w in sana.values())
                _require(14 <= hits <= 17 and sum(w["measured"]["requests"] for w in sana.values()) == 20,
                         "Sana hit count violates actual warmed four-prompt routing bounds")
            _require(all(w["measured"]["bypasses"] == 0 for w in wan.values()), "Wan cache bypassed")
            for row in cell["requests"]:
                hashes = report.request_hashes(row)
                _require(len(hashes) == (2 if row["kind"] == "video" else 1) and all(
                    isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in hashes.values()),
                    "missing finite full output or video latent hash")
            checked["arms"][name] = dict(sana=sana, wan=wan, timing=dispatch._telemetry(phase, cell),
                guard=occupancy.guard_telemetry(phase, cell, worker_uuids), image_service=_image_service(cell))
            owners.extend(w["adapter_id"] for w in sana.values())
            owners.extend(w["instance_id"] for w in wan.values())
        _require(len(owners) == len(set(owners)), "cache owner reused across workers or arms")
        exact = report.exact_pair(cells["reference"]["requests"], cells["candidate"]["requests"])
        checked["exact"]["reference/candidate"] = exact
        _require(exact["passed"] and exact["requests"] == 24, "paired final outputs or video latent hashes differ")
        checked.update(passed=True, performance_evidence=True, service_pairing=_service_pairing(cells))
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError) as error:
        checked["errors"].append(str(error))
    return checked


def _metrics(cell, audit):
    return dict(encoding._metrics(cell, audit), guard=audit["guard"], image_service=audit["image_service"])


def paired_changes(arms):
    changes = occupancy.paired_changes(arms)["reference/candidate"]
    for kind in ("total", "image", "video"):
        changes[kind]["maximum_latency_ms"] = _change(*(arms[arm][kind]["maximum_latency_ms"] for arm in study_api.ARMS))
    changes["guard"]["cold_probe_ms"] = _change(*(arms[arm]["guard"]["cold_probe_ms"] for arm in study_api.ARMS))
    changes["image_service"] = {name: {metric: _change(*(arms[arm]["image_service"][name][metric] for arm in study_api.ARMS))
        for metric in ("mean", "p50", "p95", "maximum")} for name in
        ("worker_model_ms", "host_dispatch_ms", "queue_after_admission_ms", "harvest_delay_ms")}
    return {"reference/candidate": changes}


def performance_assessment(arms):
    left, right = (arms[name] for name in study_api.ARMS)
    image_ratio = right["image"]["latency_p95_ms"] / left["image"]["latency_p95_ms"]
    video_ratio = right["video"]["latency_p95_ms"] / left["video"]["latency_p95_ms"]
    throughput_ratio = right["total"]["throughput_per_second"] / left["total"]["throughput_per_second"]
    service = _change(left["image_service"]["worker_model_ms"]["mean"], right["image_service"]["worker_model_ms"]["mean"])
    safety = dict(image_p95=image_ratio <= 1.05, video_p95=video_ratio <= 1.05, throughput=throughput_ratio >= .97)
    mechanism = service["improvement_percent"] is not None and service["improvement_percent"] >= 5
    return dict(passed=mechanism and all(safety.values()), mechanism_passed=mechanism,
        safety_passed=all(safety.values()), safety=safety, rule=RULE,
        mean_service_reduction_percent=service["improvement_percent"], image_p95_reduction_percent=(1 - image_ratio) * 100,
        image_p95_ratio=image_ratio, video_p95_ratio=video_ratio, throughput_ratio=throughput_ratio,
        timing_scope=TIMING_SCOPE)


def screening_gate(ev):
    outcome = dict(passed=False, technical_passed=bool(ev["qualified"] and ev["capacity"]),
                   performance_confirmation=False, rule=RULE)
    if not outcome["technical_passed"]:
        return dict(outcome, reason="screening technical qualification failed")
    try:
        cells = report.native_arms(ev["result"])
        _require(set(cells) == set(study_api.ARMS), "screening arms incomplete")
        arms = {name: dict(report.summary(cell["requests"], cell["elapsed_ms"]), image_service=_image_service(cell)) for name, cell in cells.items()}
        return dict(outcome, **performance_assessment(arms))
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError) as error:
        return dict(outcome, technical_passed=False, reason=str(error))


def confirmation_assessment(rows):
    rows = [r for r in rows if r["stage"] == "confirmation"]
    complete = (len(rows) == 3 and {r["seed"] for r in rows} == set(study_api.SEEDS)
                and all(r["qualified"] and r["performance_assessment"] is not None for r in rows))
    out = dict(qualified=complete, safety_passed=False, mechanism_confirmed=False, image_tail_benefit=False,
        promotable=False, mean_service_reduction_median_percent=None, service_improving_seeds=0,
        image_p95_reduction_median_percent=None, image_p95_improving_seeds=0, rule=RULE)
    if complete:
        service = [r["performance_assessment"]["mean_service_reduction_percent"] for r in rows]
        tails = [r["performance_assessment"]["image_p95_reduction_percent"] for r in rows]
        safety = all(r["performance_assessment"]["safety_passed"] for r in rows)
        mechanism = statistics.median(service) >= 5 and sum(v > 0 for v in service) >= 2
        tail = statistics.median(tails) >= 5 and sum(v > 0 for v in tails) >= 2
        out.update(safety_passed=safety, mechanism_confirmed=mechanism, promotable=safety and mechanism,
            image_tail_benefit=safety and tail, mean_service_reduction_median_percent=statistics.median(service),
            service_improving_seeds=sum(v > 0 for v in service), image_p95_reduction_median_percent=statistics.median(tails),
            image_p95_improving_seeds=sum(v > 0 for v in tails))
    return out


def summarize(study):
    out = dict(schema=SCHEMA, per_seed=[], default_change=False, confirmation_seeds=list(study_api.SEEDS), rule=RULE,
        timing_scope=TIMING_SCOPE, scope="Only new confirmation seeds; screening excluded; paired medians and absolute arm medians differ")
    for job in study["jobs"]:
        if job.get("superseded") or job.get("metadata", {}).get("role") != ROLE:
            continue
        metadata, ev = job["metadata"], report.evidence(study, job)
        row = dict(job_id=job["id"], seed=metadata["seed"], stage=metadata["stage"],
            status=ev["launch"].get("status", "not_started"), decision=job.get("decision"), qualified=False,
            audit=None, sampled_capacity=None, arms={}, paired_changes={}, prerequisites={}, performance_assessment=None)
        if row["status"] == "completed":
            audit = audit_result(ev["result"], job, read(job["input"], {}))
            capacity = encoding.raw_capacity(ev["launch"], job)
            for logical in job["requires"]:
                try:
                    gate = report.evidence(study, report.current_job(study, logical))
                    row["prerequisites"][logical] = gate["qualified"] and gate["capacity"]
                except (KeyError, ValueError, OSError, TypeError):
                    row["prerequisites"][logical] = False
            row.update(audit=audit, sampled_capacity=capacity, qualified=bool(ev["qualified"] and ev["capacity"]
                and audit["passed"] and capacity["passed"] and row["prerequisites"] and all(row["prerequisites"].values())))
            if audit["passed"]:
                row["arms"] = {name: _metrics(cell, audit["arms"][name]) for name, cell in report.native_arms(ev["result"]).items()}
                row["paired_changes"] = paired_changes(row["arms"])
                row["performance_assessment"] = performance_assessment(row["arms"])
        out["per_seed"].append(row)
    conclusion = confirmation_assessment(out["per_seed"])
    out.update(conclusion)
    rows = [r for r in out["per_seed"] if r["stage"] == "confirmation"]
    out["paired_changes_median"] = encoding._median_tree([r["paired_changes"] for r in rows]) if out["qualified"] else None
    out["absolute_arm_medians"] = encoding._median_tree([r["arms"] for r in rows]) if out["qualified"] else None
    return out
