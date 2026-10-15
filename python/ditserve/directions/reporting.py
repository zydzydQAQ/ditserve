"""CPU-only summaries of bounded experiments; supervisors own conclusions.

Local/partial reports supply descriptive measurements, never qualifications.
This module does not import any model runtime or alter a budget ledger.
"""

from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics
import time

from ..research.common import atomic_json, read_json


UNVERIFIED = "资源不足或未验证"
CONTINUE = "值得继续"
SIMPLE = "简单策略已足够"
NO_GAIN = "当前条件无收益"


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _stats(values):
    values = sorted(value for value in values if _number(value))
    if not values:
        return dict(n=0, mean=None, p95=None, maximum=None)
    position = (len(values) - 1) * .95
    lo, hi = math.floor(position), math.ceil(position)
    return dict(n=len(values), mean=statistics.mean(values),
                p95=values[lo] + (values[hi] - values[lo]) * (position - lo), maximum=values[-1])


def _complete_replay(replay, expected_requests=None):
    """Only finished, well-formed replays may enter descriptive paired totals."""
    requests = replay.get("rows", [])
    return (bool(requests) and replay.get("complete") is not False
            and _number(replay.get("elapsed_ms")) and replay["elapsed_ms"] > 0
            and (expected_requests is None or len(requests) == expected_requests)
            and all(r.get("status") == "completed" and r.get("arrived", True)
                    and r.get("kind") in ("image", "video")
                    and all(_number(r.get(k)) for k in ("arrival_ms", "start_ms", "finish_ms", "deadline_ms"))
                    and 0 <= r["arrival_ms"] <= r["start_ms"] <= r["finish_ms"] <= replay["elapsed_ms"]
                    for r in requests))


def _descriptive_requests(requests, elapsed_ms):
    """No gate evaluation: denominator includes each paired replay's full drain."""
    timely = sum(r["finish_ms"] <= r["deadline_ms"] for r in requests)
    latency = _stats(r["finish_ms"] - r["arrival_ms"] for r in requests)
    wait = _stats(r["start_ms"] - r["arrival_ms"] for r in requests)
    return dict(arrived=len(requests), completed=len(requests), on_time=timely,
                on_time_fraction=timely / len(requests) if requests else None,
                goodput_rps=timely * 1000 / elapsed_ms if elapsed_ms > 0 else None,
                latency_ms=latency, wait_ms=wait, sample_count=len(requests),
                wait_p95_ms=wait["p95"], latency_p95_ms=latency["p95"],
                scope="descriptive only; no gate or qualification inferred")


def _read(path, errors):
    if not path.is_file():
        return None
    try:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("report root is not an object")
        return value
    except (OSError, ValueError) as error:
        errors.append(dict(path=str(path), error=str(error)))
        return dict(status="incomplete", eligible=False, reason="unreadable evidence: " + str(error))


def _budget_stop(record):
    reason = str(record.get("reason", "")).lower()
    return bool(record.get("budget_stopped") or record.get("reason_code") == "budget_insufficient"
                or "budget" in reason or record.get("scientific_outcome") == "not_measured_budget_limited")


def _known_quality_failure(unit):
    quality = unit.get("quality", {})
    return (unit.get("quality_passed") is False and not unit.get("capacity_failed")
            and quality.get("status") == "ok" and quality.get("lpips_status") == "ok"
            and _number(quality.get("lpips_mean")))


def _phase(root, number, phase, errors):
    authority = next((p for p in (root / ("supervisor-%s.json" % phase),
                                 root / ("admission-%s.json" % phase)) if p.is_file()), None)
    result = _read(authority, errors) if authority else dict(status="not_run")
    native = {1: root / ("direction1-%s.json" % phase), 2: root / (phase + ".json"),
              3: root / phase / "report.json"}[number]
    candidates = [root / ("worker-%s.json" % phase), native, root / phase / "partial.json"]
    # A supervisor snapshot is immutable evidence even if a local file is newer.
    if any(key in result for key in ("rows", "units", "samples")):
        measurements, measured_path = result, authority
    else:
        available = [(p, _read(p, errors)) for p in candidates if p.is_file()]
        # The outer worker can write only a BudgetStop/error envelope after the
        # native phase saved its measurements. That envelope must not hide them.
        # Explicit empty collections still describe an observed empty result.
        measured_path, measurements = next(
            ((p, value) for p, value in available
             if any(key in value for key in ("rows", "units", "samples", "active_trace"))),
            available[0] if available else (None, {}))
    finalized = bool(authority and authority.name.startswith("supervisor-")
                     and result.get("status") in ("passed", "stopped") and not _budget_stop(result))
    links = [p for p in [authority, *candidates, root / (phase + ".log"),
                        root / "partial_units", root / phase / "partial.json"] if p and p.exists()]
    return dict(phase=phase, result=result, authority=str(authority) if authority else None,
                measurement_source=str(measured_path) if measured_path else None,
                measurement_status=measurements.get("status"),
                partial=not finalized or measured_path != authority,
                evidence=list(dict.fromkeys(str(p) for p in links))), measurements


def _outcome(number, entries):
    """Recognize only decisions written by the experiment and accepted supervisor."""
    # Independent cost completion is descriptive evidence, not a serving gate.
    # Its execution time/order must not replace a pressure terminal decision.
    if number == 2:
        entries = [entry for entry in entries if entry["phase"] != "state_cost"]
    attempted = [entry for entry in entries if entry["authority"]]
    fallback = dict(label=UNVERIFIED, final=False,
                    reason="尚未取得完整、受监督的方向判定；已有部分指标不形成最终正负结论。")
    if not attempted:
        return fallback
    last = attempted[-1]
    record, phase = last["result"], last["phase"]
    # Any broken prerequisite, engineering failure, or budget stop defeats a
    # later local success. In particular worker pass cannot override isolation.
    for entry in attempted:
        value = entry["result"]
        if (_budget_stop(value) or value.get("status") not in ("passed", "stopped")
                or value.get("supervisor_failure") or value.get("returncode", 0) not in (None, 0)):
            return dict(fallback, reason="%s: %s" % (entry["phase"], value.get("reason") or value.get("supervisor_failure") or value.get("status")))
    index = entries.index(last)
    if any(e["result"].get("status") != "passed" or e["result"].get("eligible") is not True
           or not e["authority"] or not Path(e["authority"]).name.startswith("supervisor-")
           for e in entries[:index]):
        return dict(fallback, reason="前置阶段缺少有效 supervisor 通过记录。")
    if not Path(last["authority"]).name.startswith("supervisor-"):
        return fallback
    decision, reason = record.get("decision"), record.get("reason") or ""
    label = None
    if number == 1:
        if (phase == "online" and record.get("status") == "passed" and record.get("eligible") is True
                and decision == "potential_quantum_tradeoff" and record.get("best_simple_sufficient") is False):
            label = CONTINUE
        elif (phase == "online" and record.get("eligible") is False
              and decision == "simple_policy_sufficient" and record.get("best_simple_sufficient") is True):
            label = SIMPLE
        elif record.get("eligible") is False and decision in (
                "stop_split_overhead", "stop_no_benefit", "stop_no_online_benefit"):
            label = NO_GAIN
    elif number == 2 and record.get("eligible") is False:
        failed_quality = [u for u in record.get("units", []) if u.get("quality_passed") is False]
        known_failures = bool(failed_quality) and all(_known_quality_failure(u) for u in failed_quality)
        if phase == "pressure" and reason == "keep admits urgent through K=16; no observed capacity opportunity":
            label = SIMPLE
        elif phase == "qualify" and reason == "both development orders must improve at least 5%":
            label = NO_GAIN
        elif (phase == "qualify" and known_failures
              and reason in ("development quality failed; no retuning", "held-out quality failed; selection remains frozen")):
            label = NO_GAIN
        elif (phase == "screen" and record.get("shapes") and (not failed_quality or known_failures)
              and all(s.get("quality_screen_passed") is False or
                      s.get("quality_screen_passed") is True and _number(s.get("descriptive_gain")) and s["descriptive_gain"] < .05
                      for s in record["shapes"])):
            label = NO_GAIN
    # D3 throughput admission only establishes whether the fixed hardware can
    # sustain playback; below 16 FPS leaves the serving-policy benefit untested.
    if label:
        return dict(label=label, final=True, phase=phase, decision=decision, reason=reason,
                    scope="仅本次固定模型、硬件、规格与有限样本；不证明系统新颖性或普遍结论。")
    return dict(fallback, phase=phase, reason="方向结论仍未验证。最近监督记录 %s：%s" %
                (phase, reason or "已通过阶段门槛，等待后续验证"))


def _d1(measurements, config):
    profile, correctness, preemption, online = (measurements[p] for p in ("profile", "correctness", "preemption", "online"))
    shapes = []
    for row in profile.get("rows", []):
        if row.get("kind") == "video_profile":
            shapes.append(dict(shape=row.get("shape"), status=row.get("status"), step_ms=_stats(row.get("step_ms", [])),
                truncated=row.get("truncated"), measured_step_range=row.get("measured_step_range"),
                block_samples=len(row.get("block_cuda", [])),
                max_extra_state_bytes=max(row.get("extra_state_bytes", []), default=None)))
    full = [dict(shape=r.get("shape"), status=r.get("status"), service_ms=r.get("service_ms"),
                 decode_completed=r.get("decode_completed")) for r in profile.get("rows", [])
            if r.get("kind") == "co_resident_full_generation"]
    cr = correctness.get("rows", [])
    by_shape = defaultdict(list)
    for row in cr:
        shape = row.get("shape", {})
        by_shape[(shape.get("height"), shape.get("width"), shape.get("frames"))].append(row)
    cases = [dict(shape=dict(zip(("height", "width", "frames"), key)), observed=len(rows),
                  passed=sum(r.get("passed") is True for r in rows),
                  overhead=_stats(r.get("uncut_split_overhead") for r in rows)) for key, rows in by_shape.items()]
    expected = (2 * len(config["prompts"]) * len(config["seeds"])
                if "prompts" in config and "seeds" in config else None)
    policies = config.get("policies") or list(preemption.get("summaries", {})) or list(online.get("summaries", {}))
    groups = defaultdict(dict)
    for row in preemption.get("rows", []):
        groups[(row.get("round"), row.get("fraction"))][row.get("policy")] = row
    paired = [g for g in groups.values() if policies and set(g) == set(policies)]
    pre = []
    for policy in policies:
        rows = [g[policy] for g in paired]
        observed = [r for r in preemption.get("rows", []) if r.get("policy") == policy]
        waits = [r["image_wait_ms"] - g["step"]["image_wait_ms"] for r, g in zip(rows, paired)
                 if "step" in g and _number(r.get("image_wait_ms")) and _number(g["step"].get("image_wait_ms"))]
        pre.append(dict(policy=policy, observed=len(observed), paired_n=len(rows),
            image_wait_ms=_stats(r.get("image_wait_ms") for r in rows), paired_wait_delta_ms=_stats(waits),
            max_extra_state_bytes=max((r["max_extra_state_bytes"] for r in observed if _number(r.get("max_extra_state_bytes"))), default=None),
            gate_metrics=preemption.get("summaries", {}).get(policy, {})))
    trace_groups = defaultdict(dict)
    for row in online.get("rows", []):
        if row.get("trace_sha256"):
            trace_groups[row["trace_sha256"]][row.get("policy")] = row
    paired_groups = {key: g for key, g in trace_groups.items() if policies and set(g) == set(policies)
                     and all(_complete_replay(r, config.get("online_count")) for r in g.values())}
    paired_traces = list(paired_groups.values())
    excluded_complete_replays = sum(_complete_replay(r, config.get("online_count"))
        for key, group in trace_groups.items() if key not in paired_groups for r in group.values())
    dimensions = (config.get("online_seeds"), config.get("online_loads"),
                  config.get("patterns", config.get("online_patterns")))
    expected_traces = math.prod(len(values) for values in dimensions) if all(isinstance(v, list) for v in dimensions) else None
    trace_metrics = []
    for group in paired_traces:
        values = {}
        for policy, replay in group.items():
            requests, elapsed = replay["rows"], replay["elapsed_ms"]
            values[policy] = dict(overall=_descriptive_requests(requests, elapsed),
                image=_descriptive_requests([r for r in requests if r["kind"] == "image"], elapsed),
                video=_descriptive_requests([r for r in requests if r["kind"] == "video"], elapsed))
        trace_metrics.append(values)
    on = []
    for policy in policies:
        rows = [r for r in online.get("rows", []) if r.get("policy") == policy]
        deltas = {key: [] for key in ("image_wait_p95_ms", "goodput_rps", "video_attainment")}
        for group in trace_metrics:
            if "step" not in group:
                continue
            for label, section, key in (("image_wait_p95_ms", "image", "wait_p95_ms"),
                                         ("goodput_rps", "overall", "goodput_rps"),
                                         ("video_attainment", "video", "on_time_fraction")):
                value, baseline = group[policy].get(section, {}).get(key), group["step"].get(section, {}).get(key)
                if _number(value) and _number(baseline):
                    deltas[label].append(value - baseline)
        requests = [r for group in paired_traces for r in group[policy]["rows"]]
        elapsed = sum(group[policy]["elapsed_ms"] for group in paired_traces)
        descriptive = dict(descriptive_only=True, qualified=False, elapsed_ms_including_drain=elapsed,
            overall=_descriptive_requests(requests, elapsed),
            image=_descriptive_requests([r for r in requests if r["kind"] == "image"], elapsed),
            video=_descriptive_requests([r for r in requests if r["kind"] == "video"], elapsed))
        on.append(dict(policy=policy, observed_replays=len(rows), observed_requests=sum(len(r.get("rows", [])) for r in rows),
                       paired_traces=len(paired_traces), paired_deltas_vs_step={k: _stats(v) for k, v in deltas.items()},
                       descriptive_aggregate=descriptive,
                       gate_metrics=online.get("summaries", {}).get(policy)))
    return dict(profile=dict(shapes=shapes, full_generation=full, co_resident=profile.get("co_resident"),
                    residency=profile.get("execution", {}).get("residency"),
                    selected_shape=profile.get("selected_shape"), image_preencoded_ms=profile.get("image_preencoded_ms"),
                    image_end_to_end_ms=profile.get("image_end_to_end_ms")),
                correctness=dict(observed=len(cr), expected=expected, passed=sum(r.get("passed") is True for r in cr),
                    failed=sum(r.get("passed") is False for r in cr), shapes=cases,
                    max_uncut_split_overhead=correctness.get("max_uncut_split_overhead"),
                    step_checks=sum(len(r.get("step_checks", [])) for r in cr),
                    solver_checks=sum(len(r.get("solver_history_checks", [])) for r in cr)),
                preemption=pre, online=on, active_trace=online.get("active_trace"),
                online_pair_coverage=dict(expected=expected_traces, observed=len(paired_traces),
                    observed_trace_groups=len(trace_groups), excluded_incomplete_groups=len(trace_groups) - len(paired_traces),
                    excluded_complete_replays=excluded_complete_replays),
                best_simple_sufficient=online.get("best_simple_sufficient"),
                scope="p95 为有限样本描述；block/step 测量不是独立请求。preemption 使用预编码图像，online 包含编码和 drain。配对差值为候选减 step。")


def _d2_state_cost(report):
    """Keep failed/partial operation measurements without promoting qualification."""
    units = report.get("units", [])
    timing_names = ("encoding_setup_ms", "prefix_ms", "d2h_ms", "h2d_ms", "flush_ms",
                    "action_ms", "next_step_ms", "suffix_complete_ms", "post_pause_work_ms")
    comparison_names = ("next_step_over_keep_ms", "suffix_over_keep_ms", "post_pause_work_over_keep_ms")
    def allocated_delta(row, before, after):
        memory = row.get("memory", {})
        first = (memory.get(before) or {}).get("allocated_bytes")
        second = (memory.get(after) or {}).get("allocated_bytes")
        return first - second if _number(first) and _number(second) else None
    arms = []
    for action in ("keep", "full_cpu", "flush"):
        rows = [r for r in units if r.get("action") == action]
        arms.append(dict(action=action, observed=len(rows), expected=2,
            completed=sum(r.get("complete") is True for r in rows),
            failed=sum(r.get("status") == "failed" or r.get("quality_passed") is False for r in rows),
            quality_passed=sum(r.get("quality_passed") is True for r in rows),
            quality_observed=sum(isinstance(r.get("quality_passed"), bool) for r in rows),
            exact_to_keep_passed=sum(r.get("exact_to_keep") is True for r in rows),
            exact_to_keep_observed=sum(isinstance(r.get("exact_to_keep"), bool) for r in rows),
            timings={key: _stats((r.get("timings") or {}).get(key) for r in rows) for key in timing_names},
            transfer={key: _stats((r.get("transfer") or {}).get(key) for r in rows)
                      for key in ("d2h_bytes", "h2d_bytes")},
            allocated_release_bytes=_stats(allocated_delta(r, "before", "after_action") for r in rows),
            allocated_restore_increase_bytes=_stats(allocated_delta(r, "after_restore", "after_action") for r in rows),
            lpips_mean=_stats((r.get("quality") or {}).get("lpips_mean") for r in rows),
            comparisons_ms={key: _stats((r.get("comparison") or {}).get(key) for r in rows)
                            for key in comparison_names}))
    return dict(descriptive_only=True, qualified=False,
        completed_cost_matrix=report.get("completed_cost_matrix"), expected_trials=6,
        observed_trials=len(units), completed_trials=sum(r.get("complete") is True for r in units),
        active_unit=report.get("active_unit"), arms=arms,
        trials=[{key: row.get(key) for key in ("unit_id", "round", "position", "action", "status", "complete", "generation_completed",
            "stage", "last_completed_step", "frames", "pause_step", "item", "timings", "transfer",
            "state_inventory", "memory", "quality", "quality_passed", "exact_to_keep", "comparison",
            "output_sha256", "retention", "error")} for row in units],
        cost_identity=report.get("cost_identity"), cost_protocol=report.get("cost_protocol"),
        reference_off=report.get("reference_off"),
        scope="仅单输入、81 帧、第 16 步的独立成本描述；每动作要求两序各一次。各统计 n 为实际有值的子操作数，包含失败试次中已完成的子操作，缺测不填零。allocated 差值不截断，reserved 不作为实际释放量。suffix 含第 17–32 步与 decode；post-pause 排除证据落盘与状态扫描，不是在线响应时间。成本阶段通过不授状态、容量或 SP 科研资格，也不覆盖压力阶段终止判定。")


def _d2(measurements):
    summaries = {}
    for phase, report in measurements.items():
        units = report.get("units", [])
        pairs = [r for r in units if "quality_passed" in r and "runs" in r]
        trials = report.get("action_trials") or [r for r in units if "action" in r and "count" in r]
        pressure = [r for r in units if "count" in r and "pressure_found" in r]
        summaries[phase] = dict(observed_units=len(units), active_unit=report.get("active_unit"),
            failed_units=sum(r.get("status") == "failed" or r.get("quality_passed") is False for r in units),
            capacity_failures=sum(bool(r.get("capacity_failed") or r.get("oom") or r.get("oom_stage")) for r in units),
            quality_pairs=len(pairs), quality_passed=sum(r.get("quality_passed") is True for r in pairs),
            measured_quality_failures=sum(_known_quality_failure(r) for r in pairs),
            unverified_quality_failures=sum(r.get("quality_passed") is False and not _known_quality_failure(r) for r in pairs),
            cache_quality_qualified=report.get("cache_quality_qualified"),
            cache_performance_qualified=report.get("cache_performance_qualified"),
            capacity_qualified=report.get("capacity_qualified"),
            state_mechanism_qualified=report.get("state_mechanism_qualified"), sp_resize_qualified=report.get("sp_resize_qualified"),
            shapes=report.get("shapes", []), development_gains=report.get("development_gains"),
            selected_frames=report.get("selected_frames"),
            pressure_trials=[{k: r.get(k) for k in ("count", "status", "pressure_found", "oom_stage",
                "paused_state_bytes", "memory_before_urgent", "elapsed_ms", "scope")} for r in pressure],
            pressure_count=report.get("pressure_count"), action_trials=len(trials),
            action_counts=sorted({r["count"] for r in trials}), actions=sorted({r["action"] for r in trials}),
            baseline_outcomes=report.get("baseline_outcomes", []),
            observed_admission_actions=report.get("observed_admission_actions", []),
            qualification_scope=report.get("qualification_scope") or report.get("performance_scope") or report.get("scope"))
        if phase == "state_cost":
            summaries[phase]["cost_measurements"] = _d2_state_cost(report)
    return summaries


def _d3(measurements):
    cr, profile = measurements["correctness"], measurements["profile"]
    result = dict(correctness=dict(observed=len(cr.get("samples", [])),
                    passed=sum(r.get("passed") is True for r in cr.get("samples", []))),
                  profile=dict(measured_fps=profile.get("measured_fps"), required_fps=16,
                    measured_chunks=sum(not r.get("warmup", False) for r in profile.get("samples", [])),
                    median_chunk_seconds=profile.get("median_chunk_seconds"), admission_scope=profile.get("admission_scope"),
                    configured_warmup_blocks=profile.get("configured_warmup_blocks"),
                    effective_warmup_blocks=profile.get("effective_warmup_blocks"),
                    measured_chunk_indices=profile.get("measured_chunk_indices")))
    for phase in ("update", "multisession"):
        report = measurements[phase]
        arms = []
        for sample in report.get("samples", []):
            events = [event for records in sample.get("updates", {}).values() for event in records]
            initial = [event for event in events if event.get("version") == 0]
            updates = [event for event in events if event.get("version") != 0]
            def delays(records):
                return {key: dict(_stats(e.get(key) for e in records),
                    unobserved=len(records) - sum(_number(e.get(key)) for e in records))
                    for key in ("condition_latency_s", "conditioned_decode_latency_s", "conditioned_playout_latency_s")}
            playback = list(sample.get("playback", {}).values())
            peaks = {key: max((p[key] for p in playback if _number(p.get(key))), default=None)
                     for key in ("peak_queued_frames", "peak_resident_chunks", "peak_ahead_chunks")}
            stale = [p.get("condition_provenance_stale_frames") for p in playback]
            presented = sum(len(p.get("presentations", [])) for p in playback)
            stale_fraction = sum(stale) / presented if presented and all(_number(v) for v in stale) else None
            work = list(sample.get("discarded_work", {}).values())
            flushed = [w.get("fully_flushed_denoise_context_s") for w in work]
            arms.append(dict(policy=sample.get("policy"), seed=sample.get("seed"), pattern=sample.get("pattern"),
                session_policy=sample.get("session_policy"),
                buffer_chunks=sample.get("buffer_chunks"), sessions=len(sample.get("updates", {})), events=len(events),
                initial_events=len(initial), update_events=len(updates),
                scheduled_events=sample.get("scheduled_events"), observed_events=sample.get("observed_events"),
                event_statuses=dict(Counter(str(e.get("status", "unknown")) for e in events)),
                delays_seconds=delays(updates), initial_delays_seconds=delays(initial),
                presented_frames=presented, buffer_peaks=peaks,
                condition_provenance_stale_fraction=stale_fraction,
                fully_flushed_denoise_context_wall_s=(sum(flushed) if flushed and all(_number(v) for v in flushed) else None),
                stall_seconds=sum(p.get("stall_seconds", 0.) for p in playback),
                observation_end_s=sample.get("observation_end_s")))
        result[phase] = dict(observed_arms=len(arms), arms=arms,
            technical_eligible=report.get("technical_eligible"), semantic_eligible=report.get("semantic_eligible"),
            semantic_status=report.get("semantic_status", "unverified"),
            state_isolation=report.get("state_isolation", []),
            scope="条件应用/解码/实际播放是技术延迟，不能替代语义成功。实际播放统一指独立实时无界面消费端的出队时间，不含浏览器、网络或物理显示延迟。缓冲 B 指当前播放块之外的最多 B 个预取块，按真实块身份计数；多会话峰值取各会话最大值。更新延迟排除单列的 version=0 初始条件；superseded/pending 仍在更新分母，缺失延迟不是零。旧条件比例按版本来源计算，不等于语义滞后；废弃工作表仅列完整丢弃块的同步去噪与上下文耗时，部分块估算和被覆盖 recache 成本见原始记录。")
    return result


def budget_summary(ledger, now_monotonic=None):
    if "used_gpu_seconds" not in ledger:
        return dict(status="not_started" if ledger.get("status") == "not_started" else "unavailable", phases=[], stages=[])
    now = time.monotonic() if now_monotonic is None else now_monotonic
    active = ledger.get("active")
    temporary = 0.
    if active:
        start, count = active.get("started_monotonic"), active.get("gpu_count")
        temporary = ((now - start) * count if _number(start) and _number(count) and now >= start and count > 0 else None)
    stages = []
    for name in sorted(set(ledger.get("stage_limits_gpu_seconds", {})) | set(ledger.get("stage_used_gpu_seconds", {}))):
        provisional = temporary if active and active.get("stage") == name else 0.
        settled = ledger.get("stage_used_gpu_seconds", {}).get(name, 0.)
        stages.append(dict(stage=name, settled_gpu_seconds=settled, provisional_gpu_seconds=provisional,
                           including_active_gpu_seconds=settled + provisional if provisional is not None else None,
                           limit_gpu_seconds=ledger.get("stage_limits_gpu_seconds", {}).get(name)))
    phases = {}
    for session in ledger.get("sessions", []) + ([active] if active else []):
        identity = session.get("identity", {})
        key = (session.get("stage", "unknown"), identity.get("direction"), identity.get("phase", "unknown"))
        value = phases.setdefault(key, dict(stage=key[0], direction=key[1], phase=key[2], settled_gpu_seconds=0.,
                                             provisional_gpu_seconds=0., settled_sessions=0, unsettled=False))
        if session is active:
            value.update(provisional_gpu_seconds=temporary, unsettled=True)
        else:
            value["settled_gpu_seconds"] += session.get("gpu_seconds", 0.)
            value["settled_sessions"] += 1
    return dict(status="unsettled" if active else "settled", settled_gpu_seconds=ledger["used_gpu_seconds"],
        provisional_gpu_seconds=temporary,
        including_active_gpu_seconds=ledger["used_gpu_seconds"] + temporary if temporary is not None else None,
        limit_gpu_seconds=ledger.get("limit_gpu_seconds"), stages=stages, phases=list(phases.values()),
        active=active, scope="active 按当前单调时钟临时计费，未结算；不写回账本。历史预算单独保留。")


def build_report(output, ledger, phases, *, now_monotonic=None):
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    errors, directions = [], []
    frozen = _read(output / "protocol.json", errors) or {}
    config = frozen.get("config", {})
    for number, names in phases.items():
        root = output / ("direction%d" % number)
        entries, measurements = [], {}
        for phase in names:
            entry, values = _phase(root, number, phase, errors)
            entries.append(entry)
            measurements[phase] = values
        local_config = (_read(root / "worker-config.json", errors) or config).get("direction1", {})
        metrics = _d1(measurements, local_config) if number == 1 else _d2(measurements) if number == 2 else _d3(measurements)
        # Qualification wording must not promote a worker flag when supervision
        # failed or has not finished. Descriptive samples remain available.
        for entry in entries:
            if not entry["partial"]:
                continue
            phase = entry["phase"]
            keys = {1: {"profile": ("co_resident",), "online": ()},
                    2: {phase: ("cache_quality_qualified", "cache_performance_qualified", "capacity_qualified", "state_mechanism_qualified", "sp_resize_qualified")},
                    3: {"update": ("technical_eligible", "semantic_eligible"), "multisession": ("technical_eligible", "semantic_eligible")}}
            for key in keys[number].get(phase, ()):
                metrics[phase][key] = None
            if number == 1 and phase == "online":
                metrics["best_simple_sufficient"] = None
                for row in metrics["online"]:
                    row["gate_metrics"] = None
        directions.append(dict(direction=number, phases=entries, outcome=_outcome(number, entries), metrics=metrics))
    budget = _read(ledger, errors) or dict(status="not_started")
    media = output / "media/manifest.json"
    blind = output / "direction3/update/blind-windows/public/manifest.json"
    return dict(schema_version="ditserve.directions.report.v1", summary_schema_version="ditserve.directions.summary.v1",
        directions=directions, budget=budget,
        budget_summary=budget_summary(budget, now_monotonic), budget_path=str(ledger), evidence_errors=errors,
        media_manifest=str(media) if media.is_file() else None,
        blind_manifest=str(blind) if blind.is_file() else None,
        blind_ratings=str(blind.parent / "ratings.csv") if (blind.parent / "ratings.csv").is_file() else None,
        scope="bounded research screening; supervisor status controls conclusions; partial and missing measurements are not final findings")


def _fmt(value, digits=2):
    if value is None:
        return "未测"
    if isinstance(value, bool):
        return "是" if value else "否"
    if _number(value):
        return ("%.*f" % (digits, value)) if type(value) is float else str(value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _pct(value):
    return _fmt(value * 100) + "%" if _number(value) else "未测"


def _link(label, path):
    return "[%s](<%s>)" % (label, path)


def _shape(value):
    value = value or {}
    return "%s×%s / %s 帧" % tuple(_fmt(value.get(k)) for k in ("height", "width", "frames"))


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows), ""]


def _cost_markdown(cost):
    triple = lambda value: " / ".join(_fmt(value[key]) for key in ("mean", "p95", "n"))
    lines = ["独立状态成本（描述性）：完成 %d / 要求 %d 个试次；记录 %d 个。" %
             (cost["completed_trials"], cost["expected_trials"], cost["observed_trials"]), ""]
    lines += _table(["动作", "完成 / 记录 / 要求", "失败试次", "质量通过 / 已检查", "exact-to-keep / 已检查", "LPIPS mean / p95 / n"], [
        (a["action"], "%d / %d / %d" % (a["completed"], a["observed"], a["expected"]), a["failed"],
         "%d / %d" % (a["quality_passed"], a["quality_observed"]),
         "%d / %d" % (a["exact_to_keep_passed"], a["exact_to_keep_observed"]), triple(a["lpips_mean"]))
        for a in cost["arms"]])
    timing_labels = {"encoding_setup_ms": "编码设置", "prefix_ms": "第 1–16 步前缀", "d2h_ms": "D2H 搬出",
        "h2d_ms": "H2D 恢复", "flush_ms": "flush 清缓存", "action_ms": "动作", "next_step_ms": "第 17 步",
        "suffix_complete_ms": "完整后缀含 decode", "post_pause_work_ms": "动作与后缀工作"}
    lines += _table(["动作 / 子操作", "耗时 mean / p95 ms / n"], [
        (a["action"] + " / " + label, triple(a["timings"][key]))
        for a in cost["arms"] for key, label in timing_labels.items()])
    lines += _table(["动作", "D2H bytes mean / p95 / n", "H2D bytes mean / p95 / n", "实际 allocated 释放 bytes mean / p95 / n", "恢复 allocated 增量 bytes mean / p95 / n"], [
        (a["action"], triple(a["transfer"]["d2h_bytes"]), triple(a["transfer"]["h2d_bytes"]),
         triple(a["allocated_release_bytes"]), triple(a["allocated_restore_increase_bytes"])) for a in cost["arms"]])
    lines += _table(["动作", "同轮 next-step 差值 mean / p95 ms / n", "同轮 suffix 差值 mean / p95 ms / n", "同轮 post-pause 差值 mean / p95 ms / n"], [
        (a["action"], *(triple(a["comparisons_ms"][key]) for key in
                       ("next_step_over_keep_ms", "suffix_over_keep_ms", "post_pause_work_over_keep_ms")))
        for a in cost["arms"]])
    failures = [r for r in cost["trials"] if r.get("status") == "failed" or r.get("complete") is not True
                or r.get("quality_passed") is False]
    if failures:
        lines += _table(["未完成 / 失败试次", "阶段 / 最后完成步", "状态", "错误 / 保留产物"], [
            (r["unit_id"], "%s / %s" % (_fmt(r["stage"]), _fmt(r["last_completed_step"])), r["status"],
             (_fmt(r["error"]) if r["error"] else "") + " " + " · ".join(
                 _link(Path(f["path"]).name, f["path"]) for f in (r.get("retention") or {}).get("files", []) if f.get("path")))
            for r in failures])
    lines += [cost["scope"], ""]
    return lines


def render_markdown(data):
    lines = ["# DiT serving 三方向验证", "", "仅 supervisor 的状态、eligible 和已记录门槛决定结论。部分结果仅作描述，不形成最终正负结论；缺测不记为零。", ""]
    for direction in data["directions"]:
        number, outcome, metrics = direction["direction"], direction["outcome"], direction["metrics"]
        lines += ["## 方向 %d：%s" % (number, outcome["label"]), "", _fmt(outcome["reason"]), ""]
        if outcome.get("scope"):
            lines += [outcome["scope"], ""]
        lines += _table(["阶段", "supervisor / admission", "eligible", "指标范围", "原始证据 / 失败"], [
            (e["phase"], e["result"]["status"] + ("（worker %s，未终审）" % e["measurement_status"]
                if not e["authority"] and e["measurement_status"] else ""), e["result"].get("eligible"),
             "部分 / 未验证" if e["partial"] else "已结束的受监督阶段",
             " · ".join(_link(Path(p).name, p) for p in e["evidence"]) or "未生成") for e in direction["phases"]])
        if number == 1:
            p, c = metrics["profile"], metrics["correctness"]
            lines += _table(["规格", "step mean / p95 ms", "step n", "嵌套 block n", "暂停额外状态最大字节", "截短 profile"], [
                (_shape(s["shape"]), "%s / %s" % (_fmt(s["step_ms"]["mean"]), _fmt(s["step_ms"]["p95"])),
                 s["step_ms"]["n"], s["block_samples"], s["max_extra_state_bytes"], s["truncated"]) for s in p["shapes"]])
            lines += ["记录的 co_resident / 完整生成容量检查：%s；选择规格：%s。Sana 预编码 / 端到端：%s / %s ms。" %
                      (_fmt(p["co_resident"]), _shape(p["selected_shape"]), _fmt(p["image_preencoded_ms"]), _fmt(p["image_end_to_end_ms"])), ""]
            residency = p["residency"]
            if isinstance(residency, dict) and residency:
                def location(key):
                    value = residency.get(key)
                    if value == "cpu_float32_existing_flash_baseline":
                        return "CPU FP32（沿用既有 Flash 基线）"
                    if isinstance(value, str) and value.startswith("cuda"):
                        return "GPU（%s）" % value
                    return _fmt(value) if value is not None else "未记录"
                lines += ["组件位置（execution.residency）：Wan DiT：%s；Wan VAE：%s；Sana 全部组件：%s；Wan 文本编码器：%s。" %
                          tuple(location(key) for key in ("wan_transformer", "wan_vae", "sana_all_components", "wan_text_encoder")), ""]
            else:
                lines += ["组件位置（execution.residency）：未记录，不能据此推断完整 pipeline 位于 GPU。", ""]
            for generation in p["full_generation"]:
                lines += ["完整生成容量试验：%s，%s，耗时 %s ms，decode 完成：%s。" %
                          (_shape(generation["shape"]), generation["status"], _fmt(generation["service_ms"]), _fmt(generation["decode_completed"])), ""]
            lines += ["正确性已完成 %d / %s 个样本，其中通过 %d、失败 %d；逐步检查 %d，solver 检查 %d。最终最大无中断拆分开销：%s。" %
                      (c["observed"], _fmt(c["expected"]), c["passed"], c["failed"], c["step_checks"], c["solver_checks"], _pct(c["max_uncut_split_overhead"])), ""]
            lines += _table(["正确性规格", "通过 / 已完成", "已完成样本最大 overhead"], [
                (_shape(s["shape"]), "%d / %d" % (s["passed"], s["observed"]), _pct(s["overhead"]["maximum"])) for s in c["shapes"]])
            lines += ["预编码抢占初筛（每个 round × arrival fraction 跨策略配对）：", ""]
            lines += _table(["策略", "已测 / 配对 n", "wait p95 ms", "配对 wait 差值 mean ms", "wait 降幅", "video overhead", "暂停额外字节"], [
                (r["policy"], "%d / %d" % (r["observed"], r["paired_n"]), r["image_wait_ms"]["p95"], r["paired_wait_delta_ms"]["mean"],
                 _pct(r["gate_metrics"].get("wait_reduction")), _pct(r["gate_metrics"].get("video_overhead")), r["max_extra_state_bytes"]) for r in metrics["preemption"]])
            coverage = metrics["online_pair_coverage"]
            lines += ["端到端在线重放描述性汇总：完整配对 %d / 预期 %s 个 trace；另有 %d 个不完整组未纳入，其中 %d 个完整但未配对的 replay 单独保留。仅从完整配对的原始请求记录计算，goodput 分母包含各 replay 的 idle 与 drain。" %
                      (coverage["observed"], _fmt(coverage["expected"]), coverage["excluded_incomplete_groups"],
                       coverage.get("excluded_complete_replays", 0)), ""]
            online_entry = next(e for e in direction["phases"] if e["phase"] == "online")
            if online_entry["partial"] or coverage["observed"] != coverage["expected"]:
                lines += ["在线矩阵尚未完成终审；以下为描述性结果，不形成最终正负结论，不重新计算或授予门槛资格。", ""]
            lines += _table(["策略", "已测 replay / 请求数", "配对 trace n", "SLO on-time / arrived", "goodput req/s", "图像 wait mean / p95 ms / n", "图像 latency mean / p95 ms / n", "video SLO", "配对 wait 差值 mean ms", "配对 goodput 差值", "配对 video SLO 差值"], [
                (r["policy"], "%d / %d" % (r["observed_replays"], r["observed_requests"]), r["paired_traces"],
                 "%s / %s" % (_fmt(r["descriptive_aggregate"]["overall"]["on_time"]), _fmt(r["descriptive_aggregate"]["overall"]["arrived"])),
                 _fmt(r["descriptive_aggregate"]["overall"]["goodput_rps"], 4),
                 " / ".join(_fmt(r["descriptive_aggregate"]["image"]["wait_ms"][k]) for k in ("mean", "p95", "n")),
                 " / ".join(_fmt(r["descriptive_aggregate"]["image"]["latency_ms"][k]) for k in ("mean", "p95", "n")),
                 _pct(r["descriptive_aggregate"]["video"]["on_time_fraction"]), r["paired_deltas_vs_step"]["image_wait_p95_ms"]["mean"],
                 _fmt(r["paired_deltas_vs_step"]["goodput_rps"]["mean"], 4), _pct(r["paired_deltas_vs_step"]["video_attainment"]["mean"])) for r in metrics["online"]])
            lines += ["best_simple_sufficient：%s。%s" % (_fmt(metrics["best_simple_sufficient"]), metrics["scope"]), ""]
            if metrics["active_trace"]:
                lines += ["另有未完成的 active_trace，原始到达/完成记录已保留；不混入完整配对指标。", ""]
        elif number == 2:
            lines += _table(["阶段", "记录单元 / 失败", "质量通过 / 配对", "质量未过：实测 / 未验证", "容量失败", "质量资格", "速度资格", "容量资格", "状态机制资格", "SP resize 资格"], [
                (phase, "%d / %d" % (r["observed_units"], r["failed_units"]), "%d / %d" % (r["quality_passed"], r["quality_pairs"]),
                 "%d / %d" % (r["measured_quality_failures"], r["unverified_quality_failures"]),
                 r["capacity_failures"], r["cache_quality_qualified"], r["cache_performance_qualified"], r["capacity_qualified"], r["state_mechanism_qualified"], r["sp_resize_qualified"])
                for phase, r in metrics.items()])
            for phase, r in metrics.items():
                if r["shapes"]:
                    lines += _table(["screen 帧数", "质量门槛", "已测 / 要求", "描述性收益"], [
                        (s.get("frames"), s.get("quality_screen_passed"), "%s / %s" % (s.get("observed"), s.get("required")), _pct(s.get("descriptive_gain"))) for s in r["shapes"]])
                if r["development_gains"] is not None:
                    lines += ["qualify 两种执行顺序的收益：%s（固定门槛均 ≥ 5%%）。" % " / ".join(_pct(v) for v in r["development_gains"]), ""]
                if r["pressure_trials"]:
                    lines += _table(["实际暂停请求数 K", "状态", "urgent 压力", "OOM 阶段", "必要状态 / 缓存 storage bytes", "必要状态 / 缓存 CUDA bytes", "urgent 前 allocated bytes"], [
                        (x["count"], x["status"], x["pressure_found"], x["oom_stage"],
                         " / ".join(_fmt((x.get("paused_state_bytes") or {}).get(k)) for k in
                                    ("necessary_storage_bytes", "cache_feature_storage_bytes")),
                         " / ".join(_fmt((x.get("paused_state_bytes") or {}).get(k)) for k in
                                    ("necessary_cuda_bytes", "cache_feature_cuda_bytes")),
                         (x.get("memory_before_urgent") or {}).get("allocated_bytes")) for x in r["pressure_trials"]])
                    lines += ["暂停状态按真实请求 storage 盘点，不含权重与临时工作区；allocated 是 urgent 前实测设备分配，reserved 不等于状态量或可接纳容量。压力探测未完成背景后缀，不能替代全部请求完成时间。", ""]
                if r["action_trials"]:
                    lines += ["actions 实测 %d 次；请求数 %s；动作 %s。观察到的 admission 动作：%s。" %
                              (r["action_trials"], r["action_counts"], ", ".join(r["actions"]), ", ".join(r["observed_admission_actions"]) or "无"), ""]
                    lines += _table(["K / 动作", "已测 / 要求", "全部完成且质量通过", "两序收益 vs keep", "重复 ≥ 5% 收益"], [
                        ("%s / %s" % (x.get("count"), x.get("action")), "%s / %s" % (x.get("observed"), x.get("required")),
                         x.get("all_completions_and_quality_passed"), ", ".join(_pct(v) for v in x.get("gains_vs_keep_when_keep_completes", [])) or "未测",
                         x.get("repeated_gain_observed")) for x in r["baseline_outcomes"]])
                if r.get("cost_measurements"):
                    cost = r["cost_measurements"]
                    if cost["observed_trials"] or cost["active_unit"] or cost["completed_cost_matrix"] is not None:
                        lines += _cost_markdown(cost)
                    else:
                        lines += ["state_cost：尚无独立成本测量，不从压力结果推断搬运或重算成本。", ""]
                if r["active_unit"]:
                    active = r["active_unit"]
                    lines += ["%s 保留未完成 active_unit：%s；未完成操作不作为成功或零成本。" %
                              (phase, _fmt(active.get("unit_id", active.get("key", active.get("action", "未记录标识"))))), ""]
                if r["qualification_scope"]:
                    lines += ["%s 范围：%s" % (phase, r["qualification_scope"]), ""]
            lines += ["速度口径为固定预编码输入下的状态准备、去噪及解码；文本编码作为设置成本另计 GPU 持卡预算，不据此宣称完整在线端到端加速。", "",
                      "失败保留在分母。screen 描述性速度不等于 qualify；动作测量不自动获得完整质量资格。仅同 A100、SP=1，不外推至跨卡弹性。", ""]
        else:
            p, c = metrics["profile"], metrics["correctness"]
            lines += ["正确性通过 %d / %d 个已完成样本。实测 %.0f FPS 门槛下的吞吐：%s FPS，测量 chunk n=%d；chunk 中位耗时 %s s。" %
                      (c["passed"], c["observed"], p["required_fps"], _fmt(p["measured_fps"]), p["measured_chunks"], _fmt(p["median_chunk_seconds"])), "",
                      "FPS 范围：%s；两会话另需单卡服务曲线 ≥ 32 FPS。" % _fmt(p["admission_scope"]), ""]
            if p["effective_warmup_blocks"] is not None:
                lines += ["配置预热 %s 块，排除上下文过渡后实际预热 %s 块；稳态测量索引（从 0 开始）：%s。" %
                          (p["configured_warmup_blocks"], p["effective_warmup_blocks"], p["measured_chunk_indices"]), ""]
            for phase in ("update", "multisession"):
                r = metrics[phase]
                lines += ["%s：已测 %d 个 arm；技术资格 %s；语义资格 %s；语义状态 %s。" %
                          (phase, r["observed_arms"], _fmt(r["technical_eligible"]), _fmt(r["semantic_eligible"]), r["semantic_status"]), ""]
                lines += _table(["会话调度 / 策略 / pattern / seed / buffer", "事件总数 / 初始 / 更新", "scheduled / observed", "更新应用 / 解码 / 播放 latency mean (p95) s", "相应已测 n : 更新总数", "实际播放帧", "stall s"], [
                    ("%s / %s / %s / %s / %s" % (a["session_policy"], a["policy"], a["pattern"], a["seed"], a["buffer_chunks"]),
                     "%d / %d / %d" % (a["events"], a["initial_events"], a["update_events"]),
                     "%s / %s" % (_fmt(a["scheduled_events"]), _fmt(a["observed_events"])),
                     " / ".join("%s (%s)" % (_fmt(d["mean"]), _fmt(d["p95"])) for d in a["delays_seconds"].values()),
                     " / ".join("%d:%d" % (d["n"], a["update_events"]) for d in a["delays_seconds"].values()),
                     a["presented_frames"], a["stall_seconds"]) for a in r["arms"]])
                if r["arms"]:
                    lines += _table(["会话调度 / 策略 / pattern / seed / B", "排队像素帧峰值", "驻留块 / 预取块峰值", "旧条件帧比例", "完整丢弃块去噪与上下文 wall s"], [
                        ("%s / %s / %s / %s / %s" % (a["session_policy"], a["policy"], a["pattern"], a["seed"], a["buffer_chunks"]),
                         a["buffer_peaks"]["peak_queued_frames"],
                         "%s / %s" % (_fmt(a["buffer_peaks"]["peak_resident_chunks"]), _fmt(a["buffer_peaks"]["peak_ahead_chunks"])),
                         _pct(a["condition_provenance_stale_fraction"]), a["fully_flushed_denoise_context_wall_s"]) for a in r["arms"]])
                if r["state_isolation"]:
                    lines += ["多会话状态隔离实测 n=%d，latent exact 通过 %d；逐样本像素误差见原始证据。" %
                              (len(r["state_isolation"]), sum(x.get("latent_equal") is True for x in r["state_isolation"])), ""]
                lines += [r["scope"], ""]
            lines += ["语义盲评尚未形成受监督的成功判定。导出窗口或空评分表不等于完成盲评。", ""]
    budget = data["budget_summary"]
    lines += ["## 新增 GPU 预算", "", _link("原始账本", data["budget_path"]), ""]
    if budget["status"] == "unavailable":
        lines += ["账本无法读取，当前费用未知；不按零费用处理。", ""]
    elif budget["status"] == "not_started":
        lines += ["尚未启动计费。", ""]
    else:
        hours = lambda value: value / 3600 if _number(value) else None
        lines += ["已结算 %s GPUh；active 临时未结算 %s GPUh；包含 active 合计 %s / %s GPUh。" % tuple(
            _fmt(hours(budget.get(k)), 4) for k in ("settled_gpu_seconds", "provisional_gpu_seconds", "including_active_gpu_seconds", "limit_gpu_seconds")), ""]
        lines += _table(["预算阶段", "已结算 GPUh", "临时未结算 GPUh", "额度 GPUh"], [
            (r["stage"], hours(r["settled_gpu_seconds"]), hours(r["provisional_gpu_seconds"]), hours(r["limit_gpu_seconds"])) for r in budget["stages"]])
        lines += _table(["计费阶段 / 实验 phase", "结算 session n", "已结算 GPUh", "临时 GPUh", "未结算"], [
            ("%s / D%s %s" % (r["stage"], r["direction"], r["phase"]), r["settled_sessions"], hours(r["settled_gpu_seconds"]),
             hours(r["provisional_gpu_seconds"]), r["unsettled"]) for r in budget["phases"]])
        lines += [budget["scope"], ""]
    lines += ["## 媒体与证据", ""]
    for key, title in (("media_manifest", "媒体 manifest（含失败与缺失项）"), ("blind_manifest", "匿名语义盲评窗口 manifest"), ("blind_ratings", "盲评评分表")):
        lines += ["- " + (_link(title, data[key]) if data[key] else title + "：尚未生成")]
    if data["evidence_errors"]:
        lines += ["", "以下证据读取失败，不能当作通过：", ""]
        lines += ["- %s：%s" % (_link(Path(e["path"]).name, e["path"]), e["error"]) for e in data["evidence_errors"]]
    return "\n".join(lines) + "\n"


def write_report(output, ledger, phases):
    data = build_report(output, ledger, phases)
    output = Path(output)
    atomic_json(output / "report.json", data)
    temporary = output / ".REPORT.md.tmp"
    temporary.write_text(render_markdown(data))
    temporary.replace(output / "REPORT.md")
    from ..result_storage import export_report
    export_report(output, ledger=ledger)
    return data
