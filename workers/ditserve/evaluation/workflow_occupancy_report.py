"""Strict two-arm occupancy guard audit and independent paired summaries."""

import math

from . import workflow_dispatch_refresh_report as previous
from . import workflow_encoding_optimization_report as encoding
from . import workflow_next_report as report
from . import workflow_occupancy_study as study_api
from .workflow_next_study import read
from .workflow_serving_study import digest


SCHEMA = "ditserve.workflow.occupancy-report.v1"
RULE = ("Exclude each cell's first cold probe; steady probe mean reduction >=80%, "
        "image and video p95 <=105% of reference, total throughput >=97%; "
        "technical gates and all three confirmation seeds must pass; no automatic default change")
_require = encoding._require
_number = encoding._number


def guard_telemetry(phase, cell, worker_uuids):
    """Replay fresh guard probes; bounded trace loss never qualifies this study."""
    value = cell["guard_telemetry"]
    origin = cell["host_origin_monotonic_seconds"]
    _require(value.get("schema") == "ditserve.guard-telemetry.v1"
             and value.get("occupancy_backend") == phase["occupancy_backend"]
             and value.get("cadence_seconds") == 2. and value.get("probe_timeout_seconds") == 3.
             and value.get("fail_closed") is True
             and value.get("scope") == "cell startup, warmup and execution; excludes shutdown",
             "guard controls or observed backend differ")
    _require(_number(origin, positive=True), "measured host origin missing")
    _require(_number(cell["elapsed_ms"], positive=True), "measured host interval missing")
    end = origin + cell["elapsed_ms"] / 1000
    names = {slot[kind] for slot in phase["gpus"] for kind in ("video", "image")}
    _require(set(worker_uuids) == names and len(names) == 4, "four bound GPU workers required")
    expected_owners = {(cell["worker_runtime"][name]["pid"], worker_uuids[name]) for name in names}
    _require(len(expected_owners) == 4 and all(type(pid) is int and pid > 0
             and isinstance(uuid, str) and uuid.startswith("GPU-") for pid, uuid in expected_owners),
             "invalid expected GPU owner identity")
    for name in ("heartbeat_calls", "probe_calls", "probe_errors", "dropped_events"):
        _require(type(value[name]) is int and value[name] >= 0, "invalid guard count: " + name)
    calls = value["probe_calls"]
    _require(2 <= calls <= value["heartbeat_calls"] and value["probe_errors"] == 0
             and value["dropped_events"] == 0, "guard probes failed, were dropped, or were not exercised")
    for prefix, count in (("lock_wait", value["heartbeat_calls"]),
                          ("heartbeat", value["heartbeat_calls"]), ("probe", calls)):
        total, maximum = value[prefix + "_total_ms"], value[prefix + "_max_ms"]
        _require(_number(total) and _number(maximum) and total + 1e-6 >= maximum
                 and total <= count * maximum + 1e-4, "invalid guard aggregate: " + prefix)
    _require(value["heartbeat_total_ms"] + 1e-4 >= value["probe_total_ms"] + value["lock_wait_total_ms"]
             and value["heartbeat_max_ms"] + 1e-4 >= max(value["probe_max_ms"], value["lock_wait_max_ms"]),
             "heartbeat duration excludes observed lock wait or probe time")
    events = value["events"]
    _require(isinstance(events, list) and len(events) == calls, "guard probe events are incomplete")
    durations, warmup, online, boundary = [], [], [], []
    previous_finish = None
    for event in events:
        start, finish, duration = (event[key] for key in
                                  ("start_monotonic_seconds", "finish_monotonic_seconds", "duration_ms"))
        _require(all(_number(v) for v in (start, finish, duration)) and start <= finish
                 and math.isclose((finish - start) * 1000, duration, rel_tol=1e-6, abs_tol=1e-4),
                 "guard event duration differs from monotonic endpoints")
        _require(previous_finish is None or start >= previous_finish, "serialized guard probes overlap or reorder")
        _require(finish <= end + 1e-6, "guard probe occurs after measured execution")
        _require(event.get("occupancy_backend") == phase["occupancy_backend"]
                 and event.get("status") == "exclusive_observed"
                 and type(event["owned_count"]) is int and event["owned_count"] >= 0
                 and type(event["foreign_count"]) is int and event["foreign_count"] == 0,
                 "guard did not observe a fresh exclusive response from the selected backend")
        processes = event["owned_processes"]
        _require(isinstance(processes, list) and len(processes) == event["owned_count"], "guard owner count differs from process rows")
        identities = []
        for process in processes:
            pid, pgid, uuid = process["pid"], process["pgid"], process["gpu_uuid"]
            _require(type(pid) is int and pid > 0 and (pgid is None or type(pgid) is int and pgid > 0)
                     and isinstance(uuid, str) and uuid.startswith("GPU-"), "invalid guard process identity")
            memory = process["memory_mib"]
            _require(memory == "N/A" or type(memory) in (int, float, str) and _number(float(memory)),
                     "invalid observed process memory")
            identities.append((pid, uuid))
        _require(len(set(identities)) == len(identities) and set(identities).issubset(expected_owners),
                 "guard owner is duplicated or differs from bound workers")
        durations.append(duration)
        if finish <= origin:
            warmup.append(duration)
        elif start >= origin:
            _require(set(identities) == expected_owners, "measured probe omits or misidentifies resident CUDA workers")
            online.append(duration)
        else:
            boundary.append(duration)
        previous_finish = finish
    _require(warmup and online, "guard must cover startup/warmup and measured execution")
    _require(math.isclose(sum(durations), value["probe_total_ms"], rel_tol=1e-8, abs_tol=1e-4)
             and math.isclose(max(durations), value["probe_max_ms"], rel_tol=1e-8, abs_tol=1e-4),
             "guard aggregate differs from raw probe events")
    return dict(occupancy_backend=phase["occupancy_backend"], probe_calls=calls,
        heartbeat_calls=value["heartbeat_calls"], warmup_probe_calls=len(warmup), online_probe_calls=len(online),
        boundary_probe_calls=len(boundary), expected_owners=[dict(pid=pid, gpu_uuid=uuid) for pid, uuid in sorted(expected_owners)],
        probe_mean_ms=sum(durations) / calls, probe_p50_ms=report.percentile(durations, .5),
        probe_p95_ms=report.percentile(durations, .95), probe_total_ms=value["probe_total_ms"],
        cold_probe_ms=durations[0], steady_probe_calls=calls - 1,
        probe_steady_mean_ms=sum(durations[1:]) / (calls - 1),
        probe_steady_p95_ms=report.percentile(durations[1:], .95),
        probe_max_ms=value["probe_max_ms"], online_probe_mean_ms=sum(online) / len(online),
        online_probe_p95_ms=report.percentile(online, .95),
        heartbeat_total_ms=value["heartbeat_total_ms"], heartbeat_max_ms=value["heartbeat_max_ms"],
        lock_wait_total_ms=value["lock_wait_total_ms"], lock_wait_max_ms=value["lock_wait_max_ms"],
        scope="host guard calls including warmup; aggregate call durations are not additive end-to-end savings")


def audit_result(result, job, frozen):
    checked = dict(passed=False, errors=[], arms={}, exact={}, performance_evidence=False)
    try:
        _require(job["metadata"]["role"] == "occupancy_online", "wrong occupancy role")
        _require(digest(frozen) == job["configuration_sha256"], "input configuration digest differs")
        study_api.validate_config(frozen, repeat=job["metadata"]["repeat"])
        _require(result.get("schema") == "ditserve.run.v1" and result.get("status") == "completed"
                 and result.get("config", {}).get("phases") == frozen["phases"],
                 "missing completed result or frozen phases differ")
        phases, observed = frozen["phases"], result["phases"]
        _require([p["id"] for p in observed] == [p["id"] for p in phases], "missing or reordered occupancy arms")
        cells, owners = {}, []
        mapping = {entry["ordinal"]: entry["uuid"] for entry in job["gpu_mapping"]}
        _require(set(mapping) == {0, 1} and len(job["gpu_mapping"]) == 2, "bound GPU mapping differs")
        worker_uuids = {name: mapping[settings["gpu"]] for name, settings in frozen["workers"].items()}
        _require(all(result["config"]["workers"][name]["gpu_uuid"] == uuid for name, uuid in worker_uuids.items()),
                 "resolved worker GPU UUID differs from frozen GPU mapping")
        for phase, actual in zip(phases, observed):
            name = phase["id"]
            cell = encoding._cell(phase, actual)
            cells[name] = cell
            sana = encoding._sana_cache(phase, cell, "wan_negative_cache")
            wan = encoding._wan_cache(dict(phase, id="enabled"), cell)
            checked["arms"][name] = dict(sana=sana, wan=wan, timing=previous._telemetry(phase, cell),
                                        guard=guard_telemetry(phase, cell, worker_uuids))
            owners.extend(w["adapter_id"] for w in sana.values())
            owners.extend(w["instance_id"] for w in wan.values())
        _require(len(owners) == len(set(owners)), "cache owner reused across workers or arms")
        exact = report.exact_pair(cells["reference"]["requests"], cells["candidate"]["requests"])
        checked["exact"]["reference/candidate"] = exact
        _require(exact["passed"] and exact["requests"] == 24, "paired final outputs or latent hashes differ")
        checked.update(passed=True, performance_evidence=True)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError) as error:
        checked["errors"].append(str(error))
    return checked


def paired_changes(arms):
    changes = encoding._paired_changes({"disabled": arms["reference"], "enabled": arms["candidate"]})["disabled/enabled"]
    guard = {}
    for name in ("probe_mean_ms", "probe_p95_ms", "probe_steady_mean_ms", "probe_steady_p95_ms", "online_probe_mean_ms", "online_probe_p95_ms",
                 "probe_total_ms", "probe_max_ms", "heartbeat_total_ms", "heartbeat_max_ms", "lock_wait_total_ms", "lock_wait_max_ms"):
        reference, candidate = (arms[arm]["guard"][name] for arm in study_api.ARMS)
        guard[name] = dict(reference=reference, candidate=candidate, candidate_minus_reference=candidate - reference,
                          improvement_percent=(reference - candidate) * 100 / reference if reference else None)
    return {"reference/candidate": dict(changes, guard=guard)}


def performance_assessment(arms):
    left, right = (arms[name] for name in study_api.ARMS)
    image_ratio = right["image"]["latency_p95_ms"] / left["image"]["latency_p95_ms"]
    throughput_ratio = right["total"]["throughput_per_second"] / left["total"]["throughput_per_second"]
    video_ratio = right["video"]["latency_p95_ms"] / left["video"]["latency_p95_ms"]
    reference = left["guard"]["probe_steady_mean_ms"]
    reduction = (1 - right["guard"]["probe_steady_mean_ms"] / reference) * 100 if reference > 0 else None
    safety = dict(image_p95=image_ratio <= 1.05, video_p95=video_ratio <= 1.05, throughput=throughput_ratio >= .97)
    mechanism = reduction is not None and reduction >= 80
    return dict(passed=mechanism and all(safety.values()), mechanism_passed=mechanism,
                safety_passed=all(safety.values()), safety=safety, rule=RULE,
                steady_probe_mean_reduction_percent=reduction, image_p95_ratio=image_ratio,
                video_p95_ratio=video_ratio, throughput_ratio=throughput_ratio)


def screening_gate(ev):
    outcome = dict(passed=False, technical_passed=bool(ev["qualified"] and ev["capacity"]),
                   performance_confirmation=False, rule=RULE)
    if not outcome["technical_passed"]:
        return dict(outcome, reason="screening technical qualification failed")
    try:
        cells = report.native_arms(ev["result"])
        phases = {p["id"]: p for p in ev["result"]["config"]["phases"]}
        worker_uuids = {name: worker["gpu_uuid"] for name, worker in ev["result"]["config"]["workers"].items()}
        arms = {name: dict(report.summary(cell["requests"], cell["elapsed_ms"]),
                          guard=guard_telemetry(phases[name], cell, worker_uuids)) for name, cell in cells.items()}
        return dict(outcome, **performance_assessment(arms))
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError) as error:
        return dict(outcome, technical_passed=False, reason=str(error))


def summarize(study):
    """Unstarted/blocked jobs remain pending, never numerical failures."""
    out = dict(schema=SCHEMA, per_seed=[], default_change=False, confirmation_seeds=list(study_api.SEEDS), rule=RULE,
        scope="screening separate from three new confirmation trajectories; medians of paired changes; no automatic default change")
    for job in study["jobs"]:
        if job.get("superseded") or job.get("metadata", {}).get("role") != "occupancy_online":
            continue
        metadata = job["metadata"]
        ev = report.evidence(study, job)
        row = dict(job_id=job["id"], seed=metadata["seed"], stage=metadata["stage"],
            status=ev["launch"].get("status", "not_started"), decision=job.get("decision"),
            qualified=False, audit=None, sampled_capacity=None, arms={}, paired_changes={}, prerequisites={}, performance_assessment=None)
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
                row["arms"] = {name: dict(encoding._metrics(cell, audit["arms"][name]), guard=audit["arms"][name]["guard"])
                               for name, cell in report.native_arms(ev["result"]).items()}
                row["paired_changes"] = paired_changes(row["arms"])
                row["performance_assessment"] = performance_assessment(row["arms"])
        out["per_seed"].append(row)
    rows = [r for r in out["per_seed"] if r["stage"] == "confirmation"]
    complete = len(rows) == 3 and {r["seed"] for r in rows} == set(study_api.SEEDS) and all(r["qualified"] for r in rows)
    out.update(qualified=complete, promotable=complete and all(r["performance_assessment"]["passed"] for r in rows),
               paired_changes_median=encoding._median_tree([r["paired_changes"] for r in rows]) if complete else None)
    return out
