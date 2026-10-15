"""Two-file, readback-verified reporting for the independent serving follow-up.

Read-only with respect to work evidence and ledgers. This module never deletes
raw data, starts a GPU, or promotes a candidate to a serving baseline.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .next_report import _Reader, _write_excel
from .online import INPUT_FIELDS, paired_gate
from .reporting import _stats, _walk


SCHEMA = "ditserve.followup.report.v1"
MARKER = "<!-- " + SCHEMA + " -->"
TERMINAL = frozenset(("completed", "passed", "stopped", "failed", "incomplete", "budget_stopped", "skipped"))
RAW_KEYS = frozenset(("events", "quanta", "stages", "stream_marker_evidence", "latents", "pixels",
                     "frames", "tensors", "arrays", "raw_array", "raw_values", "base64"))
MAX_FILES, MAX_ROWS = 20000, 1000000


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _safe(path):
    path = Path(path).absolute()
    if path.resolve() != path or path.is_symlink():
        raise ValueError("follow-up report paths must not traverse symlinks: " + str(path))
    return path


def _obj(value, errors, label):
    if value is None:
        return {}
    if not isinstance(value, dict):
        errors.append(label + " must be an object")
        return {}
    return value


def _rows(value, errors, label):
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(label + " must be a list")
        return []
    result = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            result.append(item)
        else:
            errors.append(label + "/%d must be an object; invalid row retained in source" % index)
    return result


def _compact(value, omissions, pointer="", depth=0):
    """Preserve bounded metadata; declare every excluded body and long array."""
    if depth > 24:
        omissions.append(dict(pointer=pointer, reason="nesting exceeds 24", retained_in_raw=True))
        return {"omitted": "nesting exceeds 24; raw retained"}
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            at = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            if key in RAW_KEYS:
                count = len(child) if isinstance(child, (list, dict, str)) else None
                omissions.append(dict(pointer=at, reason="raw body excluded", count=count, retained_in_raw=True))
                result[key] = {"omitted": "raw body excluded", "count": count}
            else:
                result[key] = _compact(child, omissions, at, depth + 1)
        return result
    if isinstance(value, list):
        if len(value) > 256 and all(_number(item) for item in value):
            omissions.append(dict(pointer=pointer, reason="long numeric list aggregated", count=len(value), retained_in_raw=True))
            return dict(aggregation="numeric list; individual elements retained in raw", **_stats(value))
        return [_compact(child, omissions, pointer + "/%d" % index, depth + 1) for index, child in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _metadata(table, phase, attempt, scope, value, source, omissions):
    compact = _compact(value, omissions)
    for pointer, leaf in _walk(compact):
        if isinstance(leaf, (dict, list)):
            if not leaf:
                table.append([phase, attempt, scope, pointer, json.dumps(leaf), 0, 1, source])
            continue
        # Error text and long source literals remain reversible, without an
        # oversized Excel cell or hidden truncation. All cells are literals.
        text = leaf if isinstance(leaf, str) else None
        chunks = [text[index:index + 12000] for index in range(0, len(text), 12000)] if text else ['""' if leaf == "" else leaf]
        for index, chunk in enumerate(chunks):
            table.append([phase, attempt, scope, pointer, chunk, index, len(chunks), source])


def _request(row):
    arrival, start, finish, deadline = (row.get(key) for key in ("arrival_ms", "start_ms", "finish_ms", "deadline_ms"))
    arrived = row.get("arrived", True)
    completed = bool(arrived and row.get("status") == "completed" and all(_number(v) for v in (arrival, finish, deadline)))
    return dict(row, arrived=arrived, measured_complete=completed,
        latency_ms=finish - arrival if completed else None,
        wait_ms=start - arrival if arrived and _number(start) and _number(arrival) else None,
        measured_service_ms=finish - start if completed and _number(start) else None,
        on_time=bool(completed and finish <= deadline))


def _aggregate(replays, kind=None):
    requests = [_request(r) for replay in replays for r in replay.get("requests", [])
                if isinstance(r, dict) and (kind is None or r.get("kind") == kind)]
    arrived = [r for r in requests if r["arrived"]]
    complete = [r for r in arrived if r["measured_complete"]]
    timely = sum(r["on_time"] for r in complete)
    windows = [r.get("elapsed_ms") for r in replays]
    elapsed = sum(windows) if windows and all(_number(v) and v >= 0 for v in windows) else None
    return dict(scheduled=len(requests), arrived=len(arrived), unarrived=len(requests) - len(arrived),
        completed=len(complete), failed_or_incomplete=len(arrived) - len(complete), on_time=timely,
        on_time_fraction=timely / len(arrived) if arrived else None,
        goodput_rps=timely * 1000 / elapsed if elapsed and elapsed > 0 else None,
        elapsed_ms=elapsed, latency_ms=_stats([r["latency_ms"] for r in complete]),
        wait_ms=_stats([r["wait_ms"] for r in complete]),
        denominator_scope="all arrived requests, including failed/incomplete; latency samples require completed timestamps")


def _event_summary(events):
    counts, durations = Counter(), defaultdict(list)
    invalid = 0
    if not isinstance(events, list):
        return dict(recorded=None, invalid=None, error="events not a list; raw retained")
    for event in events:
        if not isinstance(event, dict):
            invalid += 1
            continue
        name = str(event.get("kind", "unknown"))
        if name not in counts and len(counts) >= 128:
            name = "other_kinds_over_bound"
        counts[name] += 1
        start, end = event.get("start_ms"), event.get("end_ms")
        if _number(start) and _number(end) and end >= start:
            durations[name].append(end - start)
    return dict(recorded=len(events), invalid=invalid, counts=dict(counts),
        host_interval_ms={name: _stats(values) for name, values in durations.items()},
        scope="kind counts and host interval statistics only; overlapping spans are not additive GPU time; raw event bodies excluded")


def _paired_requests(rows, phase, attempt, table, errors):
    groups = defaultdict(lambda: defaultdict(list))
    for replay in rows:
        for request in replay.get("requests", []):
            if isinstance(request, dict):
                key = (replay.get("trace_sha256"), replay.get("order_index"), request.get("request_id"))
                groups[key][replay.get("arm")].append(_request(request))
    for (trace, order, rid), arms in sorted(groups.items(), key=lambda item: str(item[0])):
        baseline, candidate = arms.get("baseline", []), arms.get("candidate", [])
        valid_counts = len(baseline) == len(candidate) == 1
        b, c = baseline[0] if baseline else {}, candidate[0] if candidate else {}
        aligned = bool(valid_counts and all(key in b and key in c and b[key] == c[key] for key in INPUT_FIELDS))
        status = "paired" if aligned else "input_mismatch" if valid_counts else "missing_or_duplicate_arm"
        if status == "input_mismatch" or len(baseline) > 1 or len(candidate) > 1:
            errors.append("request pair " + str((trace, order, rid)) + ": " + status)
        values = [phase, attempt, trace, order, rid, b.get("kind", c.get("kind")), status,
            len(baseline), len(candidate), b.get("status"), c.get("status"), b.get("latency_ms"), c.get("latency_ms"),
            c["latency_ms"] - b["latency_ms"] if aligned and _number(b.get("latency_ms")) and _number(c.get("latency_ms")) else None,
            b.get("wait_ms"), c.get("wait_ms"), b.get("on_time"), c.get("on_time")]
        table.append(values)


def _settled(supervisor, budget):
    charge = supervisor.get("charge")
    if not isinstance(charge, dict):
        return False
    sessions = budget.get("sessions", [])
    matches = [s for s in sessions if isinstance(s, dict) and s.get("session_id") == charge.get("session_id")] if isinstance(sessions, list) else []
    seconds, elapsed, count = (charge.get(key) for key in ("gpu_seconds", "elapsed_seconds", "gpu_count"))
    cost_valid = (_number(seconds) and seconds >= 0 and _number(elapsed) and elapsed >= 0
                  and type(count) is int and count > 0 and math.isclose(seconds, elapsed * count, rel_tol=1e-10, abs_tol=1e-8))
    final = supervisor.get("final_occupancy", {})
    cleanup = isinstance(final, dict) and final.get("status") == "exclusive_observed" and final.get("owned") == [] and final.get("foreign") == []
    return bool(len(matches) == 1 and matches[0] == charge and charge.get("session_id")
                and charge.get("status") in TERMINAL and supervisor.get("status") in TERMINAL and cost_valid and cleanup)


def _inventory(root, reader_sources, closed, errors, closed_attempts=()):
    rows, total = [], 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        # Symlinks are recorded, never traversed or hashed through.
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                dirs.remove(name)
                files.append(name)
        for name in sorted(files):
            path = Path(directory) / name
            if len(rows) >= MAX_FILES:
                raise ValueError("retention inventory exceeds explicit file bound")
            file_closed = closed or any(attempt in path.parents for attempt in closed_attempts)
            item = dict(path=str(path), relative_path=str(path.relative_to(root)), status="retained", dependencies_closed=file_closed,
                deleted=False, cleanup_authorized=False, reason="report never deletes; explicit dependency closure and cleanup required")
            if path.is_symlink():
                item.update(bytes=None, sha256=None, kind="symlink_not_followed", dependencies_closed=False)
            else:
                source = reader_sources.get(str(path))
                try:
                    before = path.stat()
                    item["bytes"] = before.st_size
                    total += before.st_size
                    if source and not file_closed:
                        item["sha256"] = source["sha256"]
                        item["kind"] = "reported_metadata"
                        item["bytes"] = source["bytes"]
                    elif file_closed:
                        digest = hashlib.sha256()
                        with path.open("rb") as stream:
                            for block in iter(lambda: stream.read(1024 * 1024), b""):
                                digest.update(block)
                        after = path.stat()
                        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                            raise ValueError("file changed during inventory")
                        if source and (source["sha256"] != digest.hexdigest() or source["bytes"] != before.st_size):
                            raise ValueError("reported source changed before final inventory")
                        item.update(sha256=digest.hexdigest(), kind="reported_metadata" if source else "retained_unparsed_file")
                    else:
                        item.update(sha256=None, kind="active_or_dependent_file", hash_status="deferred until dependencies close")
                except (OSError, ValueError) as error:
                    item.update(sha256=None, dependencies_closed=False, inventory_error=str(error))
                    errors.append(str(path) + ": " + str(error))
            rows.append(item)
    return rows, total


def collect(work_root, *, ledger=None):
    root = _safe(work_root)
    if not root.is_dir():
        raise ValueError("work root must be an existing directory")
    errors, omissions, reader = [], [], _Reader()
    queue, _ = reader.read(root / "queue.json", errors)
    queue = _obj(queue, errors, "queue")
    config, _ = reader.read(root / "config.json", errors, required=True)
    study, _ = reader.read(root / "study.json", errors, required=queue.get("status") not in ("prepared", "waiting"))
    config, study = _obj(config, errors, "config"), _obj(study, errors, "study")
    cleanup, _ = reader.read(root / "cleanup.json", errors)
    cleanup = _obj(cleanup, errors, "cleanup")
    ledger_value = ledger if ledger is not None else study.get("ledger", queue.get("ledger"))
    if isinstance(ledger_value, dict):
        ledger_value = ledger_value.get("path")
    budget, budget_path = {}, None
    if ledger_value:
        budget_path = _safe(Path(ledger_value) if Path(ledger_value).is_absolute() else root / ledger_value)
        budget, _ = reader.read(budget_path, errors, required=queue.get("status") not in ("prepared", "waiting"))
        budget = _obj(budget, errors, "ledger")
    elif queue.get("status") not in ("prepared", "waiting"):
        errors.append("ledger path is missing; GPU usage is unrecorded")
    tables = {
        "Summary": [["phase", "attempt", "status", "recorded_verdict", "supervisor_status", "accounting_settled", "dependencies_closed", "recorded_replays", "recorded_requests", "reported_gate_passed", "recomputed_gate_passed", "errors"]],
        "Metrics": [["phase", "attempt", "scope", "JSON pointer", "value", "text_part", "text_parts", "source"]],
        "Requests": [["phase", "attempt", "trace", "order", "arm", "trace_seed", "load", "pattern", "request_id", "kind", "status", "arrived", "arrival_ms", "deadline_ms", "start_ms", "finish_ms", "latency_ms", "wait_ms", "measured_service_ms", "on_time", "request_seed", "service_estimate_ms", "source", "pointer"]],
        "PairedRequests": [["phase", "attempt", "trace", "order", "request_id", "kind", "pair_status", "baseline_count", "candidate_count", "baseline_status", "candidate_status", "baseline_latency_ms", "candidate_latency_ms", "latency_delta_ms", "baseline_wait_ms", "candidate_wait_ms", "baseline_on_time", "candidate_on_time"]],
        "Quality": [["phase", "attempt", "scope", "JSON pointer", "value", "text_part", "text_parts", "source"]],
        "Config": [["phase", "attempt", "scope", "JSON pointer", "value", "text_part", "text_parts", "source"]],
        "Budget": [["phase", "attempt", "scope", "JSON pointer", "value", "text_part", "text_parts", "source"]],
        "Cleanup": [["phase", "attempt", "scope", "JSON pointer", "value", "text_part", "text_parts", "source"]],
        "Failures": [["phase", "attempt", "scope", "JSON pointer", "value", "text_part", "text_parts", "source"]],
        "Sources": [["path", "sha256", "bytes"]],
        "Retention": [["path", "relative_path", "bytes", "sha256", "kind", "status", "dependencies_closed", "deleted", "cleanup_authorized", "reason", "inventory_error"]],
    }
    for label, value in (("config", config), ("study", study), ("queue", queue)):
        _metadata(tables["Config"], "study", "", label, value, str(root / (label + ".json")), omissions)
    _metadata(tables["Budget"], "study", "", "ledger; missing usage stays unrecorded", budget, str(budget_path) if budget_path else None, omissions)
    _metadata(tables["Cleanup"], "study", "", "prior runner cleanup receipts; reporting performs no deletion", cleanup, str(root / "cleanup.json"), omissions)
    phases = []
    attempts = sorted({p.parent for pattern in ("*/attempt*/phase-result.json", "*/attempt*/supervisor.json") for p in root.glob(pattern)})
    for attempt in attempts:
        issues = []
        phase_name = attempt.parent.name
        data, source = reader.read(attempt / "phase-result.json", issues, required=True)
        supervisor, _ = reader.read(attempt / "supervisor.json", issues)
        data, supervisor = _obj(data, issues, "phase result"), _obj(supervisor, issues, "supervisor")
        source_path = source["path"] if source else str(attempt / "phase-result.json")
        if data.get("schema_version") != 1:
            issues.append("unsupported or missing phase schema_version; expected 1")
        if data.get("phase") not in (None, phase_name):
            issues.append("phase identity differs from directory")
        phase = dict(phase=phase_name, attempt=attempt.name, status=data.get("status", "unrecorded"),
            recorded_verdict=data.get("verdict"), supervisor_status=supervisor.get("status"),
            accounting_settled=_settled(supervisor, budget), errors=issues)
        quality = _obj(data.get("quality"), issues, "quality")
        _metadata(tables["Metrics"], phase_name, attempt.name, "phase metadata",
            {key: value for key, value in data.items() if key not in ("quality", "local", "online", "costs", "failures")}, source_path, omissions)
        _metadata(tables["Quality"], phase_name, attempt.name, "recorded quality; no new qualification", quality, source_path, omissions)
        local = _rows(data.get("local"), issues, "local")
        _metadata(tables["Metrics"], phase_name, attempt.name, "local measurements", local, source_path, omissions)
        _metadata(tables["Metrics"], phase_name, attempt.name, "costs", _obj(data.get("costs"), issues, "costs"), source_path, omissions)
        _metadata(tables["Failures"], phase_name, attempt.name, "recorded failures", data.get("failures", []), source_path, omissions)
        _metadata(tables["Budget"], phase_name, attempt.name, "supervisor evidence", supervisor, str(attempt / "supervisor.json"), omissions)
        online = _obj(data.get("online"), issues, "online")
        replays = _rows(online.get("replays"), issues, "online/replays")
        replay_sources = {}
        active_unit = online.get("active_unit")
        partial_path = active_unit.get("partial_path") if isinstance(active_unit, dict) else None
        if partial_path:
            partial_path = Path(partial_path)
            partial_path = partial_path if partial_path.is_absolute() else attempt / partial_path
            if partial_path.resolve() != partial_path.absolute() or attempt not in partial_path.parents:
                issues.append("active partial evidence leaves its ordinary attempt directory")
            else:
                partial, partial_source = reader.read(partial_path, issues, required=True)
                partial = _obj(partial, issues, "active partial")
                identity = lambda row: (row.get("trace_sha256"), row.get("order_index"), row.get("arm"))
                if partial.get("requests") and identity(partial) not in {identity(row) for row in replays}:
                    replays.append(partial)
                    replay_sources[id(partial)] = partial_source
        clean_rows = []
        for index, replay in enumerate(replays):
            replay_source = replay_sources.get(id(replay))
            replay_source_path = replay_source["path"] if replay_source else source_path
            replay_pointer = "" if replay_source else "/online/replays/%d" % index
            requests = _rows(replay.get("requests"), issues, "online/replays/%d/requests" % index)
            clean_rows.append(dict(replay, requests=requests))
            ident = [phase_name, attempt.name, replay.get("trace_sha256"), replay.get("order_index"), replay.get("arm"), replay.get("seed"), replay.get("load"), replay.get("pattern")]
            for rindex, request in enumerate(requests):
                measured = _request(request)
                tables["Requests"].append(ident + [measured.get(k) for k in tables["Requests"][0][8:20]] +
                    [request.get("seed"), request.get("service_ms"), replay_source_path, replay_pointer + "/requests/%d" % rindex])
                _metadata(tables["Metrics"], phase_name, attempt.name, "request %d/%d service metadata" % (index, rindex),
                    {key: value for key, value in request.items() if key not in set(tables["Requests"][0][8:20]) | {"seed", "service_ms"}}, replay_source_path, omissions)
            compact_replay = {key: value for key, value in replay.items() if key not in ("requests", "events")}
            if "events" in replay:
                compact_replay["observed_event_summary"] = _event_summary(replay["events"])
            else:
                compact_replay["raw_event_body_status"] = "absent; recorded event_summary preserved, not recomputed as zero"
            _metadata(tables["Metrics"], phase_name, attempt.name, "replay %d metadata and event aggregation" % index, compact_replay, replay_source_path, omissions)
        phase.update(recorded_replays=len(clean_rows), recorded_requests=sum(len(r["requests"]) for r in clean_rows),
            online_status=online.get("status"), expected_replays=online.get("expected_replays"), expected_requests=online.get("expected_requests"))
        phase["aggregates"] = {str(arm): {kind or "all": _aggregate([r for r in clean_rows if r.get("arm") == arm], kind)
            for kind in (None, "image", "video")} for arm in sorted({r.get("arm") for r in clean_rows}, key=str)}
        _paired_requests(clean_rows, phase_name, attempt.name, tables["PairedRequests"], issues)
        recorded_gate = _obj(online.get("comparison_gate"), issues, "comparison_gate")
        expected_ids = online.get("expected_trace_ids")
        if expected_ids is None:
            expected_ids = sorted({r.get("trace_sha256") for r in clean_rows if isinstance(r.get("trace_sha256"), str)})
            phase["matrix_scope"] = "observed trace set only; complete planned matrix not established"
        else:
            phase["matrix_scope"] = "explicit expected_trace_ids; request-level gate recomputed"
        recomputed = {}
        if clean_rows:
            try:
                recomputed = paired_gate(clean_rows, expected_trace_ids=expected_ids)
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                issues.append("cannot recompute online gate: " + str(error))
        phase.update(reported_gate_passed=recorded_gate.get("passed"), recomputed_gate_passed=recomputed.get("passed"))
        if recorded_gate.get("passed") is not None and recorded_gate.get("passed") != recomputed.get("passed"):
            issues.append("reported and recomputed online gate disagree")
        if online.get("status") == "completed" and type(phase["expected_replays"]) is int and phase["expected_replays"] != len(clean_rows):
            issues.append("completed online result lacks expected replay count")
        if online.get("status") == "completed" and type(phase["expected_requests"]) is int and phase["expected_requests"] != phase["recorded_requests"]:
            issues.append("completed online result lacks expected request count")
        phase["dependencies_closed"] = bool(phase["accounting_settled"] and not issues
            and phase["status"] == "completed" and online.get("status") == "completed"
            and supervisor.get("returncode") == 0 and not supervisor.get("supervisor_failure")
            and supervisor.get("charge", {}).get("status") == "completed"
            and type(phase["expected_replays"]) is int and phase["expected_replays"] == len(clean_rows)
            and type(phase["expected_requests"]) is int and phase["expected_requests"] == phase["recorded_requests"])
        _metadata(tables["Metrics"], phase_name, attempt.name, "online plan and recorded gate", {k: v for k, v in online.items() if k != "replays"}, source_path, omissions)
        _metadata(tables["Metrics"], phase_name, attempt.name, "recomputed gate", recomputed, source_path, omissions)
        _metadata(tables["Metrics"], phase_name, attempt.name, "descriptive aggregates", phase["aggregates"], source_path, omissions)
        _metadata(tables["Failures"], phase_name, attempt.name, "report validation errors", issues, source_path, omissions)
        tables["Summary"].append([phase.get(key) if key != "errors" else " | ".join(issues) for key in tables["Summary"][0]])
        phases.append(phase)
    active = budget.get("active")
    declared_phases = queue.get("phases", study.get("phases", []))
    phase_set_complete = (isinstance(declared_phases, list) and bool(declared_phases)
                          and set(declared_phases) == {p["phase"] for p in phases})
    queue_closed = queue.get("status") in ("completed", "finalizing") and phase_set_complete
    explicit_closed = study.get("dependencies_closed") is True and study.get("status") in TERMINAL
    closed = bool((queue_closed or explicit_closed) and "active" in budget and active is None
                  and phases and not errors and all(p["dependencies_closed"] for p in phases))
    closed_attempts = [root / p["phase"] / p["attempt"] for p in phases if p["dependencies_closed"] and active is None]
    retention, total_bytes = _inventory(root, reader.sources, closed, errors, closed_attempts)
    if errors:
        closed = False
    for row in retention:
        tables["Retention"].append([row.get(key) for key in tables["Retention"][0]])
    for row in reader.sources.values():
        tables["Sources"].append([row.get(key) for key in tables["Sources"][0]])
    _metadata(tables["Failures"], "study", "", "report validation errors", errors, str(root), omissions)
    _metadata(tables["Metrics"], "study", "", "explicit metadata omissions; original retained", omissions, str(root), [])
    for name, rows in tables.items():
        if len(rows) > MAX_ROWS:
            raise ValueError("Excel sheet exceeds explicit row bound: " + name)
        # openpyxl reads an empty inline string as None. Empty metadata strings
        # were explicitly encoded above; structural blank cells are None.
        tables[name] = [[None if value == "" else value for value in row] for row in rows]
    cleanup_files = cleanup.get("files", [])
    prior_deleted = sum(isinstance(row, dict) and (row.get("deleted") is True or row.get("status") == "deleted")
                        for row in cleanup_files) if isinstance(cleanup_files, list) else 0
    prior_rewrites = sum(isinstance(row, dict) and row.get("status") in ("compacted", "completed", "aggregated")
        for row in cleanup.get("rewrites", [])) if isinstance(cleanup.get("rewrites"), list) else 0
    return dict(schema_version=SCHEMA, work_root=str(root), budget_path=str(budget_path) if budget_path else None,
        status=queue.get("status", study.get("status", "unrecorded")), active_unsettled=active is not None, phases=phases,
        errors=errors, omissions=omissions, tables=tables, sources=list(reader.sources.values()), retention=retention,
        work_bytes=total_bytes, dependencies_closed=closed, used_gpu_seconds=budget.get("used_gpu_seconds"),
        cleanup=cleanup, prior_cleanup_deleted=prior_deleted, prior_cleanup_rewrites=prior_rewrites,
        raw_retained=not (prior_deleted or prior_rewrites), originals_deleted=0, baseline_promotion=False,
        retention_scope="report preserves all current files; prior deletion/aggregation receipts retained; further closure and cleanup belong to separate runner")


def render(report):
    lines = [MARKER, "# DiT 混合负载后续优化", "",
        "本报告保存新实验的描述性结果，配套数据见 [results.xlsx](results.xlsx)。报告导出不授予候选资格，也不修改历史实验。", "",
        "状态：`%s`；账本存在未结算会话：`%s`。" % (report["status"], report["active_unsettled"]),
        "已结算 GPU 秒：`%s`（缺失表示未记录，不按零计）。" % report["used_gpu_seconds"], "",
        "| 阶段 | 尝试 | 状态 | 记录结论 | 重放／请求数 | GPU 账目已核对 | 在线门槛复算 |",
        "|---|---|---|---|---:|---|---|"]
    for phase in report["phases"]:
        cells = [phase["phase"], phase["attempt"], phase["status"], phase.get("recorded_verdict"),
            "%d／%d" % (phase["recorded_replays"], phase["recorded_requests"]), phase["accounting_settled"], phase["recomputed_gate_passed"]]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in cells) + " |")
    lines += ["", "每个尝试单独保存，失败或未完成请求保留在分母。均值与 p95 为已完成且有合法时间戳的样本统计，不宣称稳定 p99；goodput 使用完整重放窗口，不能单凭其增加推断容量提升。", ""]
    for phase in report["phases"]:
        lines.append("**%s／%s**" % (phase["phase"], phase["attempt"]))
        for arm, groups in phase["aggregates"].items():
            image, video, all_requests = groups["image"], groups["video"], groups["all"]
            lines.append("- `%s`：图像 n=%s，均值=%s ms，p95=%s ms；视频均值=%s ms；按时完成=%s/%s，goodput=%s 请求/秒。" %
                (arm, image["latency_ms"]["n"], image["latency_ms"]["mean"], image["latency_ms"]["p95"], video["latency_ms"]["mean"],
                 all_requests["on_time"], all_requests["arrived"], all_requests["goodput_rps"]))
        lines += ["", "完整性范围：" + phase["matrix_scope"] + "。", ""]
        if phase["errors"]:
            lines += ["报告校验问题：" + "；".join(phase["errors"]), ""]
    retention_text = ("原始文件全部保留" if report["raw_retained"] else
        "此前执行器清理记录：删除 %d 个文件、改写聚合 %d 个文件；已删事件正文或数组无法据本报告重新评测" %
        (report["prior_cleanup_deleted"], report["prior_cleanup_rewrites"]))
    lines += ["工作目录合计 **%d 字节、%d 个文件**；依赖显式关闭：`%s`。%s；本次报告删除 **0** 个文件。运行中、评测、续跑或故障调查依赖未关闭时不得清理。" %
              (report["work_bytes"], len(report["retention"]), report["dependencies_closed"], retention_text), "",
        "配置、账本、来源快照和必要控制元数据继续保留在工作目录用于审计；只有交付目录固定为工作簿与报告两个文件，并非整个工作目录只剩两个文件。历史删除与改写的前后大小、哈希和状态见 Cleanup 表。", "",
        "工作簿包含质量、局部指标、逐请求配对、账本、配置、来源哈希、失败与保留清单。原始事件只保存种类计数和时长摘要；数组、模型权重、视频与完整事件正文不进入工作簿。明确省略项和未解析文件仍需单独审核后才能清理。", "",
        "工作簿所有值以字面文本保存，避免公式执行并保持哈希、长整数和浮点文本精度；空字符串显式记为双引号，长文本按带序号的片段可逆保存。每个单元格发布前已回读核验。"]
    if report["errors"]:
        lines += ["", "全局校验问题：" + "；".join(report["errors"])]
    return "\n".join(lines) + "\n"


def write(work_root, output, *, ledger=None):
    """Publish only results.xlsx and REPORT.md; return evidence/retention facts."""
    root, destination = _safe(work_root), _safe(output)
    if root == destination or root in destination.parents or destination in root.parents:
        raise ValueError("work and final output directories must be disjoint")
    destination.mkdir(parents=True, exist_ok=True)
    markdown = destination / "REPORT.md"
    excel = destination / "results.xlsx"
    if markdown.is_symlink() or excel.is_symlink():
        raise ValueError("final artifacts must not be symlinks")
    if markdown.exists() and not markdown.read_text(encoding="utf-8").startswith(MARKER):
        raise ValueError("existing report belongs to another experiment; refusing overwrite")
    if excel.exists() and not markdown.exists():
        raise ValueError("existing workbook has no matching follow-up report")
    report = collect(root, ledger=ledger)
    with tempfile.TemporaryDirectory(prefix=".followup-report-", dir=str(destination)) as temporary:
        temporary = Path(temporary)
        verification = _write_excel(temporary / "results.xlsx", report["tables"])
        text = render(report)
        (temporary / "REPORT.md").write_text(text, encoding="utf-8")
        if (temporary / "REPORT.md").read_text(encoding="utf-8") != text:
            raise ValueError("report text readback differs")
        (temporary / "results.xlsx").replace(excel)
        (temporary / "REPORT.md").replace(markdown)
    return dict(status=report["status"], excel=dict(path=str(excel), sha256=hashlib.sha256(excel.read_bytes()).hexdigest(), **verification),
        markdown=str(markdown), sources=report["sources"], retention=report["retention"],
        phases=len(report["phases"]), requests=len(report["tables"]["Requests"]) - 1,
        paired_requests=len(report["tables"]["PairedRequests"]) - 1,
        work_bytes=report["work_bytes"], output_bytes=excel.stat().st_size + markdown.stat().st_size,
        dependencies_closed=report["dependencies_closed"], errors=report["errors"] + [error for p in report["phases"] for error in p["errors"]],
        omissions=report["omissions"], raw_retained=report["raw_retained"],
        prior_cleanup_deleted=report["prior_cleanup_deleted"], prior_cleanup_rewrites=report["prior_cleanup_rewrites"],
        originals_deleted=0, baseline_promotion=False)
