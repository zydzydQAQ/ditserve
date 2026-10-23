"""Strict paired audit for dispatch freshness, separate from frozen r013 evidence."""

import math

from . import workflow_dispatch_refresh_study as study_api
from . import workflow_encoding_optimization_report as encoding
from . import workflow_next_report as report
from .workflow_next_study import read
from .workflow_serving_study import digest

SCHEMA = "ditserve.workflow.dispatch-refresh-report.v1"
PAIRS = (("cache_off", "cache_on"), ("cache_on", "cache_on_refresh"),
         ("cache_off", "cache_on_refresh"))
_require = encoding._require
_number = encoding._number


def _telemetry(phase, cell):
    _require(cell.get("dispatch_refresh") is phase["dispatch_refresh"], "actual dispatch refresh differs")
    _require(cell["elapsed_ms"] >= max(r["finish_ms"] for r in cell["requests"]),
             "cell elapsed time precedes request completion")
    admissions, harvest, rpc = [], [], []
    for row in cell["requests"]:
        admitted = row["admission_ms"]
        _require(_number(admitted) and row["arrival_ms"] <= admitted <= row["start_ms"],
                 "invalid arrival admission time")
        admissions.append(admitted - row["arrival_ms"])
        for event in row["dispatch_timeline"]:
            complete, harvested, delay = event["rpc_completed_ms"], event["harvested_ms"], event["harvest_delay_ms"]
            _require(event.get("timing_scope") == "host_dispatch_task" and _number(complete)
                     and _number(harvested) and _number(delay)
                     and event["start_ms"] <= complete <= harvested <= event["finish_ms"]
                     and math.isclose(harvested - complete, delay, abs_tol=1e-5),
                     "invalid RPC completion or harvest delay")
            harvest.append(delay)
            if row["kind"] == "image":
                _require(complete - event["start_ms"] + 1e-6 >= row["model_ms"],
                         "image model time exceeds completed RPC window")
        if row["kind"] == "video":
            encoded = row["encoder_rpc"]
            _require(type(encoded["calls"]) is int and encoded["calls"] >= 2
                     and encoded.get("timing_scope") == "host_encoder_round_trip" and encoded["failed_calls"] == 0
                     and encoded["calls"] == row["operation_counts"]["encode_start"] + row["operation_counts"]["encode_poll"]
                     and _number(encoded["total_ms"]) and _number(encoded["max_ms"])
                     and encoded["total_ms"] >= encoded["max_ms"], "invalid encoding RPC aggregate")
            for name, endpoint in (("start", "encode_submit_ms"), ("ready_poll", "encode_ready_ms")):
                window = encoded[name]
                _require(_number(window["start_ms"]) and _number(window["finish_ms"])
                         and window["failed"] is False and _number(row[endpoint])
                         and admitted <= window["start_ms"] <= window["finish_ms"] <= row[endpoint] <= row["start_ms"],
                         "invalid encoding RPC boundary")
            durations = [encoded[name]["finish_ms"] - encoded[name]["start_ms"]
                         for name in ("start", "ready_poll")]
            _require(encoded["start"]["finish_ms"] <= encoded["ready_poll"]["start_ms"]
                     and encoded["max_ms"] + 1e-6 >= max(durations)
                     and encoded["total_ms"] + 1e-6 >= sum(durations)
                     and encoded["total_ms"] <= encoded["calls"] * encoded["max_ms"] + 1e-6,
                     "encoding aggregate disagrees with recorded RPC windows")
            rpc.append(encoded["max_ms"])
    return dict(admission_delay_p95_ms=report.percentile(admissions, .95),
                admission_delay_max_ms=max(admissions), harvest_delay_p95_ms=report.percentile(harvest, .95),
                harvest_delay_max_ms=max(harvest), encoding_rpc_max_ms=max(rpc),
                scope="host RPC and controller times; not CUDA kernel intervals")


def audit_result(result, job, frozen):
    checked = dict(passed=False, errors=[], arms={}, exact={}, performance_evidence=False)
    try:
        _require(job["metadata"]["role"] == "dispatch_refresh_online", "wrong study role")
        _require(digest(frozen) == job["configuration_sha256"], "input configuration digest differs")
        study_api.validate_config(frozen, repeat=job["metadata"]["repeat"])
        _require(result.get("schema") == "ditserve.run.v1" and result.get("status") == "completed"
                 and result.get("config", {}).get("phases") == frozen["phases"],
                 "result differs from frozen configuration")
        phases, observed = frozen["phases"], result["phases"]
        _require([p["id"] for p in observed] == [p["id"] for p in phases], "missing or reordered arms")
        cells, owners = {}, []
        for phase, actual in zip(phases, observed):
            name = phase["id"]
            cell = encoding._cell(phase, actual)
            cells[name] = cell
            sana = encoding._sana_cache(phase, cell, "wan_negative_cache")
            # Only adapt the receipt replay's historical arm naming. The full
            # new configuration and measured controls were checked above.
            receipt_phase = dict(phase, id="disabled" if name == "cache_off" else "enabled")
            wan = encoding._wan_cache(receipt_phase, cell)
            checked["arms"][name] = dict(sana=sana, wan=wan, timing=_telemetry(phase, cell))
            owners.extend(w["adapter_id"] for w in sana.values())
            owners.extend(w["instance_id"] for w in wan.values() if w["instance_id"] is not None)
        _require(len(owners) == len(set(owners)), "cache ownership reused between workers or arms")
        for reference, candidate in PAIRS:
            pair = report.exact_pair(cells[reference]["requests"], cells[candidate]["requests"])
            checked["exact"][reference + "/" + candidate] = pair
            _require(pair["passed"], "paired final outputs or latent hashes differ")
        checked.update(passed=True, performance_evidence=True)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError) as error:
        checked["errors"].append(str(error))
    return checked


def paired_changes(arms):
    values = {}
    for left, right in PAIRS:
        # Reuse the established paired-delta arithmetic, retaining new labels.
        delta = encoding._paired_changes({"disabled": arms[left], "enabled": arms[right]})
        values[left + "/" + right] = delta["disabled/enabled"]
    return values


def screening_gate(ev):
    """A loose, predeclared screen; independent confirmation is still required."""
    outcome = dict(passed=False, rule="exact/capacity qualified; image p95 not worse by >5%, throughput not lower by >3%",
                   performance_confirmation=False)
    if not (ev["qualified"] and ev["capacity"]):
        return dict(outcome, reason="technical qualification failed")
    cells = report.native_arms(ev["result"])
    reference = report.summary(cells["cache_on"]["requests"], cells["cache_on"]["elapsed_ms"])
    candidate = report.summary(cells["cache_on_refresh"]["requests"], cells["cache_on_refresh"]["elapsed_ms"])
    image_ratio = candidate["image"]["latency_p95_ms"] / reference["image"]["latency_p95_ms"]
    throughput_ratio = candidate["total"]["throughput_per_second"] / reference["total"]["throughput_per_second"]
    return dict(outcome, passed=image_ratio <= 1.05 and throughput_ratio >= .97,
                image_p95_ratio=image_ratio, throughput_ratio=throughput_ratio)


def summarize(study):
    out = dict(schema=SCHEMA, per_seed=[], default_cache_change=False,
               scope="screening separate from three independent confirmation trajectories; descriptive paired medians")
    for job in study["jobs"]:
        if job.get("superseded") or job.get("metadata", {}).get("role") != "dispatch_refresh_online":
            continue
        ev = report.evidence(study, job)
        audit = audit_result(ev["result"], job, read(job["input"], {}))
        row = dict(job_id=job["id"], seed=job["metadata"]["seed"], stage=job["metadata"]["stage"],
                   qualified=ev["qualified"] and ev["capacity"] and audit["passed"], audit=audit,
                   sampled_capacity=encoding.raw_capacity(ev["launch"], job), arms={}, paired_changes={})
        if audit["passed"]:
            row["arms"] = {name: encoding._metrics(cell, audit["arms"][name])
                           for name, cell in report.native_arms(ev["result"]).items()}
            row["paired_changes"] = paired_changes(row["arms"])
        out["per_seed"].append(row)
    rows = [r for r in out["per_seed"] if r["stage"] == "confirmation"]
    complete = (len(rows) == 3 and {r["seed"] for r in rows} == {4301, 4302, 4303}
                and all(r["qualified"] for r in rows))
    out.update(qualified=complete, confirmation_seeds=[4301, 4302, 4303],
               paired_changes_median=encoding._median_tree([r["paired_changes"] for r in rows]) if complete else None)
    return out
