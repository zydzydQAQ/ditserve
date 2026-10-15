"""Metadata-only reports for the separately authorized cache online factorial.

No original local rows, arrays, event bodies or compiler caches enter the Excel
staging tree. This module never prunes files or writes to the historical run.
"""

import collections
import fcntl
import math
from pathlib import Path
import time

from ..research.common import atomic_json, digest, file_digest
from .online import FROZEN_TRACE_IDS, paired_gate, request_identity, request_metrics, _overlap
from .protocol import OUTPUT as HISTORICAL_OUTPUT, LEDGER as HISTORICAL_LEDGER, PARENT
from .reporting import _read, _small, _stats, _walk, _fmt


ARMS = ("off_step", "off_k4", "paired_step", "paired_k4")
COMPARISONS = (("preemption", "off_step", "off_k4"), ("cache", "off_step", "paired_step"),
               ("cache_on_k4", "off_k4", "paired_k4"), ("preemption_on_cache", "paired_step", "paired_k4"))
REQUEST_FIELDS = ("request_id", "kind", "arrival_ms", "deadline_ms", "service_ms", "seed", "arrived",
                  "status", "start_ms", "finish_ms", "encode_start_ms", "encode_finish_ms", "error")
SCOPE = "finite four-arm AB/BA screening; descriptive p95, no stable p99 or semantic-quality claim"


def _inside(path, root):
    return path == root or root in path.parents


def _safe_output(output, ledger, study=None):
    output, ledger = Path(output).absolute(), Path(ledger).absolute()
    if output.resolve() != output or ledger.resolve() != ledger:
        raise ValueError("new report paths must not traverse symlinks")
    old = [HISTORICAL_OUTPUT.resolve(), HISTORICAL_LEDGER.resolve().parent, PARENT.resolve().parent]
    if study and study.get("source_root"):
        old.append(Path(study["source_root"]).resolve())
    if any(_inside(output, root) or _inside(ledger, root) for root in old):
        raise ValueError("supplement report cannot write into historical result trees")
    if ledger.name != "gpu-budget.json" or not _inside(output, ledger.parent) or output == ledger.parent:
        raise ValueError("new report must be under its independent supplement ledger")
    return output, ledger


def _identity(row):
    return row.get("trace_sha256"), row.get("order_index"), row.get("factorial_arm", row.get("arm"))


def _cleaned(terminal):
    final = terminal.get("final_occupancy", {})
    return final.get("status") == "exclusive_observed" and final.get("owned") == [] and final.get("foreign") == []


def _valid_allowance(value, field):
    unlimited = value.get("budget_unlimited", False)
    seconds = value.get(field)
    return (type(unlimited) is bool and ((unlimited and seconds is None) or
            (not unlimited and type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0)))


def _events(events, elapsed_ms=None):
    """Bounded kinds; interval sums are deliberately absent (intervals overlap)."""
    groups, counts = collections.defaultdict(list), collections.Counter()
    for event in events:
        kind = str(event.get("kind", "unknown"))
        if kind not in counts and len(counts) >= 128:
            kind = "other_kinds_over_bound"
        counts[kind] += 1
        start, end = event.get("start_ms"), event.get("end_ms")
        if all(type(value) in (int, float) and math.isfinite(value) for value in (start, end)) and end >= start:
            groups[kind].append(end - start)
    window = elapsed_ms if type(elapsed_ms) in (float, int) and math.isfinite(elapsed_ms) and elapsed_ms >= 0 else None
    intervals = [(event["start_ms"], event["end_ms"]) for event in events
        if event.get("kind") in ("partial_snapshot_build", "partial_flush")
        and all(type(event.get(key)) in (int, float) and math.isfinite(event[key]) for key in ("start_ms", "end_ms"))
        and event["end_ms"] >= event["start_ms"]]
    union = _overlap(intervals, 0., window) if window is not None else None
    return dict(event_count=len(events), counts=dict(counts),
                host_interval_ms={key: _stats(values) for key, values in groups.items()},
                observer=dict(intervals_recorded=len(intervals), within_trace_union_ms=union,
                    trace_window_ms=window, fraction=union / window if window else None,
                    scope="union of snapshot-build/partial-flush host intervals clipped to measured trace wall window; final flush after cutoff excluded from ratio but charged; logging overhead is not subtracted from serving metrics"),
                scope="overlapping/nested host intervals; not GPU kernel time or utilization; raw events retained separately")


def _metrics(rows, expected_requests=96):
    requests = [request for row in rows for request in row.get("requests", [])]
    elapsed = [row.get("elapsed_ms") for row in rows]
    valid_elapsed = all(type(value) in (float, int) and math.isfinite(value) and value >= 0 for value in elapsed)
    elapsed_ms = sum(elapsed) if valid_elapsed else None
    def one(items):
        if elapsed_ms is None:
            return None
        value = request_metrics(items, elapsed_ms)
        return dict(value, scheduled_recorded=len(items),
                    latency_ms=dict(n=value["sample_count"], mean=value["latency_mean_ms"], p95=value["latency_p95_ms"]))
    return dict(traces=len(rows), elapsed_ms=elapsed_ms,
                all=one(requests), images=one([r for r in requests if r["kind"] == "image"]),
                videos=one([r for r in requests if r["kind"] == "video"]),
                expected_requests=expected_requests, recorded_requests=len(requests),
                unrecorded_expected_requests=max(0, expected_requests - len(requests)),
                scope="descriptive; failed/incomplete arrived requests stay in SLO denominator; goodput divides timely completions by all recorded trace wall time including idle and drain")


def _source_checks(study, terminal, output, errors, sources):
    """Verify explicit small source bindings; never inspect an artifact array."""
    def binding(value, label):
        if not isinstance(value, dict) or not value.get("path") or not value.get("sha256"):
            errors.append(label + ": missing source binding")
            return None
        try:
            data, observed = _read(Path(value["path"]))
            if observed["sha256"] != value["sha256"]:
                raise ValueError("bound source hash differs")
            sources.append(observed)
            return data
        except (OSError, ValueError, TypeError) as error:
            errors.append(label + ": " + str(error))
            return None

    replay = study.get("replay", {})
    local = None
    for name in ("supervisor", "local", "config", "protocol", "manifest", "ledger"):
        data = binding(replay.get("bindings", {}).get(name), "original " + name)
        if name == "local":
            local = data
    old_ledgers = study.get("old_ledgers", [])
    if len(old_ledgers) != 2 or len({item.get("path") for item in old_ledgers}) != 2:
        errors.append("both distinct historical ledgers must be source-bound")
    for index, value in enumerate(old_ledgers):
        binding(value, "old ledger %d" % index)
    authorization = binding(study.get("authorization"), "authorization")
    if authorization is not None and (authorization.get("approved_by_user") is not True
            or authorization.get("scope") != "cache_online_factorial_only"
            or not _valid_allowance(authorization, "authorized_gpu_seconds")
            or not _valid_allowance(study, "authorized_gpu_seconds")
            or authorization.get("budget_unlimited", False) != study.get("budget_unlimited", False)
            or authorization.get("authorized_gpu_seconds") != study.get("authorized_gpu_seconds")
            or not authorization.get("user_instruction")):
        errors.append("explicit authorization differs from the bound study")
    validation = replay.get("local_validation", {})
    if (validation.get("original_rows") != 84 or validation.get("new_local_executions") != 0
            or validation.get("qualification_pairs") != 20 or local is None or len(local.get("rows", [])) != 84):
        errors.append("84 original local rows / 20 qualification pairs reference is incomplete")
    if replay.get("config_sha256") != study.get("inherited_config_sha256"):
        errors.append("study/replay configuration hash differs")
    if terminal:
        if terminal.get("config_sha256") != study.get("inherited_config_sha256"):
            errors.append("terminal/study configuration hash differs")
        attempt = Path(terminal.get("output", ""))
        manifest_path = output / "sources" / ("cache_k4-" + attempt.name) / "manifest.json"
        try:
            manifest, source = _read(manifest_path)
            sources.append(source)
            hashes = manifest.get("source_files", {})
            if (not hashes or digest(hashes) != manifest.get("source_sha256")
                    or manifest.get("source_sha256") != terminal.get("source_sha256")
                    or manifest.get("config_sha256") != study.get("inherited_config_sha256")
                    or manifest.get("execution_root") != str(manifest_path.parent)):
                raise ValueError("new frozen source identity differs")
            for name, expected in hashes.items():
                relative = Path(name)
                path = manifest_path.parent / relative
                if (relative.is_absolute() or ".." in relative.parts or path.resolve() != path.absolute()
                        or path.suffix not in (".py", ".json") or not path.is_file()
                        or path.stat().st_size > 32 * 1024 * 1024 or file_digest(path) != expected):
                    raise ValueError("new frozen source changed: " + name)
        except (OSError, ValueError, TypeError) as error:
            errors.append("new source: " + str(error))
    return dict(original_rows=84, new_local_executions=0, remeasured=False,
        source=replay.get("bindings", {}).get("local"), validation=_small(validation),
        scope="archived local qualification reference only; no local rows, arrays, warm state or old charge copied")


def _attempt(path, expected_trace_ids, sources, errors):
    stage_metadata = {}
    for name in ("loading", "capacity", "cache-factorial-warmup", "memory"):
        metadata_path = path / (name + ".json")
        if metadata_path.exists():
            value, binding = _read(metadata_path)
            sources.append(binding)
            evidence = _small(value)
            if name == "cache-factorial-warmup":
                evidence["rows"] = [_small(row) for row in value.get("rows", [])]
                evidence["active_unit"] = _small(value.get("active_unit"))
            stage_metadata[name] = dict(source=binding, evidence=evidence,
                scope="current charged process phase metadata; allocator peaks belong to the producer's documented reset interval, not necessarily whole-process peak")
    source_path = path / "cache-online.json"
    if not source_path.exists():
        return dict(path=str(path), status="not_started", rows=[], request_samples=[], event_summaries=[],
                    source=None, raw_request_count=0, matrix_complete=False, comparisons={}, stage_metadata=stage_metadata)
    online, source = _read(source_path)
    sources.append(source)
    rows = list(online.get("rows", []))
    row_sources = [(source, "/rows/%d" % index) for index in range(len(rows))]
    active_path = online.get("active_trace_path")
    if active_path:
        partial_path = Path(active_path)
        if not _inside(partial_path.absolute(), path) or partial_path.name != "partial.json":
            errors.append("partial trace path is outside its attempt")
        elif partial_path.exists():
            partial, partial_source = _read(partial_path)
            sources.append(partial_source)
            if partial.get("requests") and _identity(partial) not in {_identity(row) for row in rows}:
                rows.append(partial)
                row_sources.append((partial_source, ""))
    expected = {(trace, order, arm) for trace in expected_trace_ids for order in (0, 1) for arm in ARMS}
    identities = [_identity(row) for row in rows]
    valid_rows = all(len(row.get("requests", [])) == 24 and request_identity(row["requests"]) == row.get("trace_sha256")
        and row.get("complete") is True and row.get("status") == "completed"
        and all(r.get("status") == "completed" and r.get("arrived", True) for r in row["requests"]) for row in rows)
    matrix_complete = (len(rows) == 16 and len(set(identities)) == 16 and set(identities) == expected
                       and valid_rows and online.get("status") == "completed")
    comparisons = {}
    for name, baseline, candidate in COMPARISONS:
        selected = [dict(row, arm="baseline" if _identity(row)[2] == baseline else "candidate")
                    for row in rows if _identity(row)[2] in (baseline, candidate)]
        gate = paired_gate(selected, expected_trace_ids=expected_trace_ids)
        comparisons[name] = dict(gate, baseline_arm=baseline, candidate_arm=candidate,
                                 qualified=False, descriptive_only=not matrix_complete)
    if matrix_complete and any(online.get("comparisons", {}).get(name) != {
            key: value for key, value in gate.items() if key not in ("baseline_arm", "candidate_arm", "qualified", "descriptive_only")}
            for name, gate in comparisons.items()):
        errors.append("recorded comparisons differ from raw-request recomputation")
    requests, events, traces = [], [], []
    for row, (row_source, pointer) in zip(rows, row_sources):
        identity = dict(trace_sha256=row.get("trace_sha256"), order_index=row.get("order_index"),
                        factorial_arm=_identity(row)[2])
        traces.append(dict(identity, status=row.get("status"), complete=row.get("complete"),
            elapsed_ms=row.get("elapsed_ms"), request_count=len(row.get("requests", [])),
            source=row_source, pointer=pointer))
        events.append(dict(identity, source=row_source, pointer=pointer + "/events", **_events(row.get("events", []), row.get("elapsed_ms"))))
        for index, request in enumerate(row.get("requests", [])):
            requests.append(dict(identity, source=row_source, pointer=pointer + "/requests/%d" % index,
                                 request={key: _small(request[key]) for key in REQUEST_FIELDS if key in request}))
    observer_by_arm = {}
    for arm in ARMS:
        observers = [row["observer"] for row in events if row["factorial_arm"] == arm]
        valid = all(row["within_trace_union_ms"] is not None and row["trace_window_ms"] is not None for row in observers)
        union = sum(row["within_trace_union_ms"] for row in observers) if valid else None
        wall = sum(row["trace_window_ms"] for row in observers) if valid else None
        observer_by_arm[arm] = dict(trace_count=len(observers), within_trace_union_ms=union, trace_window_ms=wall,
            fraction=union / wall if wall else None,
            scope="sum of per-trace interval unions divided by sum of trace windows; host observer share, not GPU time; no overhead deduction")
    return dict(path=str(path), status=online.get("status"), source=source, rows=traces, stage_metadata=stage_metadata,
        request_samples=requests, event_summaries=events, raw_request_count=len(requests),
        matrix_complete=matrix_complete, complete_traces=sum(row.get("complete") is True for row in rows),
        arms={arm: _metrics([row for row in rows if _identity(row)[2] == arm]) for arm in ARMS},
        per_trace_arms={trace: {arm: _metrics([row for row in rows if row.get("trace_sha256") == trace and
                         _identity(row)[2] == arm], expected_requests=48) for arm in ARMS} for trace in expected_trace_ids},
        observer_by_arm=observer_by_arm,
        comparisons=comparisons, active_unit=_small(online.get("active_unit")),
        scope="each arm expects four traces and 96 requests; retries are never pooled")


def collect(output, ledger):
    output, ledger = _safe_output(output, ledger)
    sources, errors = [], []
    def optional(path):
        if not path.exists():
            return {}
        try:
            value, source = _read(path)
            sources.append(source)
            return value
        except (OSError, ValueError, TypeError) as error:
            errors.append(str(path) + ": " + str(error))
            return {}
    budget = optional(ledger)
    study = optional(output / "study.json")
    _safe_output(output, ledger, study)
    terminal = optional(output / "cache_k4/terminal.json")
    if ((output / "log-finalization.json").exists() or
            (terminal.get("online_evaluation_complete") is True and terminal.get("output")
             and not (Path(terminal["output"]) / "cache-online.json").exists())):
        from .cache_archive_view import load_archive_report
        return load_archive_report(output, ledger)
    active = budget.get("active")
    elapsed = max(0., time.monotonic() - active["started_monotonic"]) if active else 0.
    local = _source_checks(study, terminal, output, errors, sources) if study else dict(
        original_rows=0, new_local_executions=0, source=None, remeasured=False,
        scope="not started; archived local source has not been bound")
    if budget and study:
        bound_parents = {value.get("path"): value.get("sha256") for value in budget.get("parents", {}).values()}
        expected_parents = {value.get("path"): value.get("sha256") for value in study.get("old_ledgers", [])}
        if (bound_parents != expected_parents or len(expected_parents) != 2
                or not _valid_allowance(budget, "limit_gpu_seconds")
                or budget.get("budget_unlimited", False) != study.get("budget_unlimited", False)
                or budget.get("limit_gpu_seconds") != study.get("authorized_gpu_seconds")):
            errors.append("independent ledger differs from study authorization or historical bindings")
    trace_ids = tuple(row.get("trace_sha256") for row in study.get("estimate", {}).get("trace_bindings", []))
    if study and (len(trace_ids) != 2 or set(trace_ids) != set(FROZEN_TRACE_IDS)):
        errors.append("study does not bind the two frozen traces")
    expected_ids = FROZEN_TRACE_IDS
    attempts = []
    for path in sorted((output / "cache_k4").glob("attempt[0-9][0-9]")):
        try:
            attempts.append(_attempt(path, expected_ids, sources, errors))
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(str(path) + ": " + str(error))
            attempts.append(dict(path=str(path), status="unreadable", matrix_complete=False,
                                 rows=[], request_samples=[], event_summaries=[], comparisons={}))
    selected_path = ((active or {}).get("identity", {}).get("output") or terminal.get("output")
                     or (attempts[-1]["path"] if attempts else None))
    selected = next((item for item in attempts if item["path"] == selected_path), None)
    if terminal and selected_path == terminal.get("output"):
        attempt_path = Path(selected_path)
        if not _inside(attempt_path, output / "cache_k4"):
            errors.append("terminal attempt is outside the new run")
        elif optional(attempt_path / "supervisor.json") != terminal:
            errors.append("terminal differs from its attempt supervisor")
    charge = terminal.get("charge", {})
    settled = [item for item in budget.get("sessions", []) if item.get("session_id") == charge.get("session_id")]
    settled_ok = bool(charge and len(settled) == 1 and settled[0] == charge
                      and charge.get("stage") == "cache_k4" and charge.get("gpu_count") == 1
                      and charge.get("identity", {}).get("output") == terminal.get("output")
                      and charge.get("identity", {}).get("run_root") == str(output)
                      and charge.get("identity", {}).get("config_sha256") == study.get("inherited_config_sha256")
                      and charge.get("identity", {}).get("source_sha256") == terminal.get("source_sha256")
                      and charge.get("identity", {}).get("authorization_source") == study.get("authorization")
                      and charge.get("identity", {}).get("cache_replay_from") == study.get("replay"))
    terminal_present = bool(terminal and terminal.get("status") in ("passed", "completed", "stopped", "failed", "incomplete"))
    authority = bool(terminal_present and not active and settled_ok and _cleaned(terminal)
        and terminal.get("returncode") == 0 and not terminal.get("supervisor_failure")
        and terminal.get("status") in ("passed", "completed", "stopped") and charge.get("status") == "completed")
    if terminal and not study:
        errors.append("terminal exists without a frozen study")
    if terminal and not settled_ok:
        errors.append("terminal charge is not identical to the independent settled ledger session")
    if terminal and selected_path != terminal.get("output"):
        authority = False  # A later active retry supersedes a prior terminal for reporting.
    if terminal and terminal.get("new_local_executions", 0) != 0:
        errors.append("supplement reports unexpected new local executions")
    matrix_complete = bool(selected and selected.get("matrix_complete"))
    if matrix_complete and authority and terminal.get("online_qualified") != selected["comparisons"]["cache_on_k4"]["passed"]:
        errors.append("supervisor online qualification differs from recomputed cache_on_k4 gate")
    evaluated = bool(authority and not errors and matrix_complete and terminal.get("online_evaluation_complete") is True)
    online_qualified = bool(evaluated and terminal.get("online_qualified") is True
                           and selected["comparisons"]["cache_on_k4"]["passed"])
    if selected:
        for gate in selected["comparisons"].values():
            gate["qualified"] = bool(evaluated and gate["passed"])
            gate["descriptive_only"] = not evaluated
    status = "running" if active else terminal.get("status", "partial" if attempts else "not_started")
    return dict(schema_version="ditserve.optimizations.cache_followup_report.v1", output=str(output),
        budget_path=str(ledger), status=status,
        decision="值得继续" if online_qualified else "当前配置无收益" if evaluated else "未验证",
        reason=terminal.get("reason", "完整四策略矩阵尚未执行" if not attempts else "完整矩阵与监督结算尚未核验"),
        online_qualified=online_qualified, online_evaluation_complete=evaluated, matrix_complete=matrix_complete,
        expected=dict(traces=16, requests=384, arms=4, traces_per_arm=4, requests_per_arm=96),
        terminal_source=next((s for s in sources if s["path"] == str(output / "cache_k4/terminal.json")), None),
        terminal=_small(terminal), study=_small(study), local_reference=local, new_local_executions=0,
        selected_attempt=selected_path, current=selected, previous_attempts=[a for a in attempts if a is not selected],
        budget=dict(limit_gpu_seconds=budget.get("limit_gpu_seconds", study.get("authorized_gpu_seconds")),
            budget_unlimited=(budget if budget else study).get("budget_unlimited", False) is True,
            authorization_verified=bool(study and not errors),
            settled_gpu_seconds=budget.get("used_gpu_seconds", 0.),
            active_gpu_seconds_estimate=elapsed, active_unsettled=bool(active),
            total_gpu_seconds_including_active=budget.get("used_gpu_seconds", 0.) + elapsed,
            sessions=_small(budget.get("sessions", [])), active=_small(active),
            scope="independent supplementary allowance; loading/warmup/failed work/cleanup included; no old balance borrowed"),
        source_bindings=list({s["path"]: s for s in sources}.values()), errors=errors,
        archive_ready=bool(terminal_present and not active and settled_ok and _cleaned(terminal) and not errors),
        retention=dict(raw_events_deleted=0, raw_arrays_deleted=0, prune_authorized=False), scope=SCOPE)


def render_markdown(report):
    current = report.get("current") or {}
    archive_only = report.get("evidence_mode") == "archive_only"
    lines = ["# 缓存与抢占四策略在线补充实验", "", "结论：**%s**；状态：%s。%s" %
             (report["decision"], report["status"], report["reason"]), "",
             "计划为 16 次轨迹重放、384 个请求；每策略 4 次重放、96 个请求。当前记录 %d 次、%d 个请求，完整轨迹 %d 次。" %
             (len(current.get("rows", [])), current.get("raw_request_count", 0), current.get("complete_traces", 0)),
             "失败与未完成的已到达请求保留在按时率分母；未到达/未启动数量另列。完整矩阵、来源、监督结算任一未通过时，指标仅供描述，不形成最终正负结论。", "",
             "| 策略 | 重放/4 | 记录请求/96 | 已到达/已完成 | 未记录/未到达/失败未完成 | goodput (请求/s) | 图像延迟 mean/p95 (ms), n | 视频按时率（按时/到达） |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    if archive_only:
        lines[2:2] = ["**仅查看已核验归档（archive-only）**：以下为冻结的已审计结论，不是重新评测或新增资格。原始事件已按明确回执清理，不能重建或重新开展事件级分析；账本、授权、配置及来源快照已重新核验。", ""]
    for arm in ARMS:
        value = current.get("arms", {}).get(arm, {})
        overall, images, videos = (value.get(name) or {} for name in ("all", "images", "videos"))
        lines.append("| %s | %s | %s | %s/%s | %s/%s/%s | %s | %s/%s, %s | %s (%s/%s) |" %
            (arm, value.get("traces", 0), value.get("recorded_requests", 0), _fmt(overall.get("arrived")),
             _fmt(overall.get("completed")), value.get("unrecorded_expected_requests", 96),
             _fmt(overall.get("scheduled_recorded", 0) - overall.get("arrived", 0)), _fmt(overall.get("failed_or_incomplete")),
             _fmt(overall.get("goodput_rps")), _fmt(images.get("latency_mean_ms")),
             _fmt(images.get("latency_p95_ms")), _fmt(images.get("sample_count")), _fmt(videos.get("on_time_fraction")),
             _fmt(videos.get("on_time")), _fmt(videos.get("arrived"))))
    lines += ["", "goodput 使用全部已记录轨迹的实测墙钟时间，包含到达间隔和最后请求排空；p95 是有限样本描述值。日志开销未从服务指标中扣除。", "",
              "| 配对比较 | %s | 有资格的结论 | goodput 变化 | 图像 p95 降幅 | 视频按时率下降 |" %
              ("归档 gate" if archive_only else "本地重算 gate"),
              "| --- | --- | --- | --- | --- | --- |"]
    for name, baseline, candidate in COMPARISONS:
        gate = current.get("comparisons", {}).get(name, {})
        pooled = gate.get("pooled") or {}
        lines.append("| %s (%s→%s) | %s | %s | %s | %s | %s |" %
            (name, baseline, candidate, gate.get("passed", False), gate.get("qualified", False),
             _fmt(pooled.get("goodput_gain")), _fmt(pooled.get("image_p95_reduction")), _fmt(pooled.get("video_ontime_drop"))))
    lines += ["", "门槛：汇总 goodput 增益至少 5% 或图像 p95 降幅至少 10%；每个 trace 的 goodput 降低不超过 5%、图像 p95 增加不超过 5%、视频按时率下降不超过 5 个百分点。后者是相对约束，不保证绝对 SLO 水平。", "",
              "| 冻结 trace | 策略 | 请求/48 | goodput | 图像 mean/p95 (ms), n | 视频按时率 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for trace, arms in current.get("per_trace_arms", {}).items():
        for arm, value in arms.items():
            overall, images, videos = (value.get(name) or {} for name in ("all", "images", "videos"))
            lines.append("| `%s` | %s | %s | %s | %s/%s, %s | %s |" % (trace[:12], arm,
                value["recorded_requests"], _fmt(overall.get("goodput_rps")), _fmt(images.get("latency_mean_ms")),
                _fmt(images.get("latency_p95_ms")), _fmt(images.get("sample_count")), _fmt(videos.get("on_time_fraction"))))
    lines += ["", "| 策略 | 日志观察区间并集 (ms) | 实测窗口 (ms) | 并集/窗口 |",
              "| --- | --- | --- | --- |"]
    for arm, value in current.get("observer_by_arm", {}).items():
        lines.append("| %s | %s | %s | %s |" % (arm, _fmt(value["within_trace_union_ms"]),
            _fmt(value["trace_window_ms"]), _fmt(value["fraction"])))
    lines.append("并集在每次 trace 内合并 snapshot/flush 重叠区间，再跨 trace 汇总。截止后的最终落盘不进入比例，但仍计入持卡费用；这不是 GPU 时间。")
    if current.get("source"):
        if archive_only:
            view = report["archive_view"]
            lines += ["", "原始在线日志已清理；历史来源 `%s`，SHA256 `%s`。" %
                      (current["source"]["path"], current["source"]["sha256"]),
                      "[日志摘要 Excel](%s) · [明确清理回执](%s)" %
                      (str(Path(view["log_manifest"]["path"]).parent / "results.xlsx"), view["prune_receipt"]["path"])]
        else:
            lines += ["", "[本轮原始在线证据](%s)，SHA256 `%s`。" % (current["source"]["path"], current["source"]["sha256"])]
    for name, record in current.get("stage_metadata", {}).items():
        evidence = record["evidence"]
        lines += ["", "[%s](%s)：状态 %s；用时 %s ms。" % (name, record["source"]["path"],
            evidence.get("status", evidence.get("passed", "未记录")),
            _fmt(evidence.get("elapsed_ms", evidence.get("model_load_ms"))))]
        if name == "capacity":
            lines.append("完整容量探测区间 peak allocated：%s bytes。" % _fmt(evidence.get("peak_allocated_bytes")))
        if name == "loading":
            lines.append("加载区间 peak allocated/reserved：%s/%s bytes。" %
                (_fmt(evidence.get("peak_allocated_bytes")), _fmt(evidence.get("peak_reserved_bytes"))))
        if name == "cache-factorial-warmup":
            for row in evidence.get("rows", []):
                lines.append("预热 %s：%s 步，%s ms；释放前该重置区间 peak allocated：%s bytes。" %
                    (row.get("arm"), row.get("completed_steps"), _fmt(row.get("elapsed_ms")),
                     _fmt(row.get("memory_before_release", {}).get("peak_allocated_bytes"))))
        if name == "memory":
            lines.append("最终 allocator peak allocated/reserved：%s/%s bytes；范围：%s。" %
                (_fmt(evidence.get("peak_allocated_bytes")), _fmt(evidence.get("peak_reserved_bytes")),
                 evidence.get("scope", "未记录；不能解释为全进程峰值")))
    local = report["local_reference"]
    lines += ["", "旧局部实验仅引用 %d 行；本轮新局部执行次数为 **0**。旧质量与完整视频不代表本轮生成或在线质量重测。" % local["original_rows"]]
    if local.get("source"):
        lines.append("原始局部证据：[cache-k4.json](%s)，SHA256 `%s`。" % (local["source"]["path"], local["source"]["sha256"]))
    budget = report["budget"]
    limit = (("无限（明确授权，仍完整计费）" if budget["authorization_verified"] else "无限标记（授权或来源核验未通过）")
             if budget["budget_unlimited"] else _fmt(budget["limit_gpu_seconds"]) + " GPU 秒")
    lines += ["", "独立预算：已结算 %s GPU 秒；当前未结算估计 %s GPU 秒；授权上限 %s。" %
              (_fmt(budget["settled_gpu_seconds"]), _fmt(budget["active_gpu_seconds_estimate"]), limit),
              ("仅保留已核验的历史指标、事件摘要与来源哈希；原始事件已按回执清理，不能重新评测。本次没有删除数组。" if archive_only else
               "事件只保留来源哈希、数量与有界 host 区间统计进入 Excel；重叠区间不累加为 GPU 时间。原始事件与数组不会自动删除。"), "",
              "[独立账本](%s) · [结构化报告](%s)" % (report["budget_path"], str(Path(report["output"]) / "report.json"))]
    if report["errors"]:
        lines += ["", "证据核验问题："] + ["- " + error for error in report["errors"]]
    workbook = Path(report["output"]) / "excel/results.xlsx"
    if workbook.is_file():
        lines += ["", "[轻量 Excel 汇总](%s) · [归档核验](%s)" %
                  (str(workbook), str(Path(report["output"]) / "archive.json"))]
    return "\n".join(lines) + "\n"


def write_report(output, ledger):
    report = collect(output, ledger)
    output = Path(report["output"])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    (output / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def archive(output, ledger):
    """Export/read back a new, bounded metadata tree; never prune originals."""
    from .. import compact_results
    output, ledger = _safe_output(output, ledger)
    report = collect(output, ledger)
    if not report["archive_ready"]:
        return dict(status="protected", reason="terminal, inactive ledger, cleanup and source hashes must verify", originals_deleted=0)
    stage, destination = output / "archive-metadata", output / "excel"
    with ledger.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = collect(output, ledger)
        if not report["archive_ready"]:
            return dict(status="protected", reason="admission changed before archive lock", originals_deleted=0)
        if stage.exists() or destination.exists():
            raise FileExistsError("supplement archive is immutable; do not overwrite an earlier export")
        stage.mkdir()
        summary = dict(report)
        summary["current"] = {key: value for key, value in (report.get("current") or {}).items()
                              if key not in ("request_samples", "event_summaries")}
        summary["previous_attempts"] = [{key: value for key, value in attempt.items()
            if key not in ("request_samples", "event_summaries")} for attempt in report["previous_attempts"]]
        atomic_json(stage / "report-summary.json", summary)
        for index, attempt in enumerate(report["previous_attempts"] + ([report["current"]] if report.get("current") else [])):
            atomic_json(stage / ("attempt%02d-requests.json" % index), attempt.get("request_samples", []))
            atomic_json(stage / ("attempt%02d-event-summary.json" % index), attempt.get("event_summaries", []))
        atomic_json(stage / "retention.json", dict(originals_deleted=0, prune_authorized=False,
            local_rows="source reference only; not copied", events="originals retained; only bounded summaries exported",
            arrays="not read, hashed, converted or pruned", compiler_cache="not traversed"))
        (stage / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
        metric_limit = 3000
        for path in stage.glob("*.json"):
            value, _ = _read(path)
            metric_limit = max(metric_limit, 1 + sum(not isinstance(v, (dict, list)) for _, v in _walk(value)))
        # compact_results acquires this same ledger lock. It protects export;
        # reacquire and revalidate every bound input before publishing archive.json.
        fcntl.flock(lock, fcntl.LOCK_UN)
        exported = compact_results.export(stage, destination, metrics_limit=metric_limit)
        if exported.get("status") != "verified":
            return dict(status="protected", export=exported, originals_deleted=0)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verified = compact_results.verify(destination, check_sources=True)
        final = collect(output, ledger)
        if not final["archive_ready"] or final["source_bindings"] != report["source_bindings"]:
            raise ValueError("source or accounting changed during archive; export cannot be published")
        result = dict(status="verified", workbook=exported["workbook"], manifest=exported["manifest"],
            verification=verified, staging_root=str(stage), source_bindings=report["source_bindings"],
            original_local_rows_copied=0, new_local_executions=0, original_events_copied=0,
            originals_deleted=0, original_prune_authorized=False, scope=SCOPE)
        atomic_json(output / "archive.json", result)
        return result
