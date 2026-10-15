"""Paired reports for the separately authorized V100 continuation study.

Historical budget exhaustion is not rewritten as historical qualification.
Complete historical observations can close a new, explicitly sourced matrix.
The two-block candidate is selected only by its predeclared development gate.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from ..directions.reporting import budget_summary
from ..research.common import atomic_json, canonical, digest, file_digest, read_json
from ..result_storage import export_report
from .reporting import (
    BASELINES, SCENARIOS, _candidate, _compact_request, _fmt, _gain, _number,
    _paired, _phase_records, _read, _stats, aggregate, complete_replay, evaluate_screen,
)


PHASES = ("completion", "granularity", "replicate")
COMPLETION_ARMS = ("edf_b1", "edf_b4", "guarded_b4", "guarded_adaptive", "feasible_b4", "edf_step")
GRANULARITY_CANDIDATES = ("edf_b2", "edf_b1_async")
GRANULARITY_ARMS = ("edf_b1", "edf_b4", *GRANULARITY_CANDIDATES)
REPLICATION_BASELINES = ("edf_b1", "edf_b4", "edf_step")


def evaluate_granularity(records, expected_seeds=(301,)):
    seeds = tuple(expected_seeds)
    candidates = {arm: _candidate(records, arm, seeds) for arm in GRANULARITY_CANDIDATES}
    expected = {(scenario, seed, arm) for scenario in SCENARIOS for seed in seeds for arm in GRANULARITY_ARMS}
    if (len(records) != len(expected) or {_key(record) for record in records} != expected
            or any(candidate["status"] == "incomplete" for candidate in candidates.values())):
        return dict(status="incomplete", passed=False, selected_candidate=None, candidates=candidates,
                    reason="the entire predeclared development matrix must be complete before selection")
    passed = [arm for arm, candidate in candidates.items() if candidate["passed"]]
    scores = {arm: candidates[arm]["metrics"]["overall"][arm]["overall"]["goodput_rps"] for arm in passed}
    selected = max(passed, key=lambda arm: scores[arm]) if passed else None
    if len(passed) == 2 and abs(scores["edf_b2"] - scores["edf_b1_async"]) / max(scores.values()) < .01:
        selected = "edf_b2"
    return dict(status="passed" if selected else "failed", passed=selected is not None,
                selected_candidate=selected, candidates=candidates,
                selection_rule="fixed development gates; higher overall goodput; under 1% difference prefers synchronous EDF2; no replication reselection")


def overhead_metrics(records):
    """Wall-time fractions; missing measurements stay unknown rather than zero."""
    elapsed = sum(r["elapsed_ms"] for r in records) if records and all(
        _number(r.get("elapsed_ms")) and r["elapsed_ms"] > 0 for r in records) else None
    measurements = {}
    for key in ("scheduling_ms", "checkpoint_ms", "output_persistence_ms"):
        values = [r.get(key) for r in records]
        observed = [value for value in values if _number(value) and value >= 0]
        total = sum(observed) if values and len(observed) == len(values) else None
        measurements[key] = dict(observed_replays=len(observed), expected_replays=len(values), total_ms=total,
                                 share_of_elapsed=total / elapsed if total is not None and elapsed else None,
                                 per_replay=_stats(observed))
    totals = [value["total_ms"] for value in measurements.values()]
    background = any(record.get("arm") == "edf_b1_async" or record.get("output_persistence_mode") == "background"
                     for record in records)
    async_records = [record for record in records if record.get("arm") == "edf_b1_async"
                     or record.get("output_persistence_mode") == "background"]
    # The new executor measures submit/backpressure in output_persistence_ms.
    # Background CPU write time is separately recorded in artifact_writer and
    # must not be added to foreground wall-time overhead.
    async_timing_known = all(isinstance(record.get("artifact_writer"), dict)
                            and _number(record.get("output_drain_ms")) for record in async_records)
    drain = sum(record["output_drain_ms"] for record in async_records) if async_timing_known else None
    combined = sum(totals) + drain if all(value is not None for value in totals) and drain is not None else None
    stages = defaultdict(list)
    for record in records:
        for row in record.get("rows", []):
            for stage, value in (row.get("stage_ms") or {}).items():
                if _number(value) and value >= 0:
                    stages[stage].append(value)
    return dict(elapsed_ms=elapsed, measurements=measurements, tracked_overhead_ms=combined,
                tracked_overhead_share=combined / elapsed if combined is not None and elapsed else None,
                background_output_overlap=background,
                foreground_persistence_scope="output_persistence_ms measures synchronous write or background submit/backpressure; final output drain is added once",
                background_writer={key: _stats((record.get("artifact_writer") or {}).get(key) for record in async_records)
                                   for key in ("write_ms", "queue_wait_ms", "drain_wait_ms")},
                output_drain_ms=_stats(record.get("output_drain_ms") for record in records),
                stage_work_ms={key: dict(total_ms=sum(values), request_samples=len(values),
                                         share_of_elapsed=sum(values) / elapsed if elapsed else None)
                               for key, values in sorted(stages.items())},
                scope="fractions of complete trace wall time including drain and idle; artifact_writer.write_ms overlaps model work and is never added to foreground overhead; request waits overlap and are not added; phase loading/warmup remain in GPU accounting")


def _summary(records):
    return dict(aggregate(records), overhead=overhead_metrics(records))


def _key(record):
    return record.get("scenario"), record.get("seed"), record.get("arm")


def _historical_keys():
    main = {(scenario, 301, arm) for scenario in SCENARIOS[:3]
            for arm in ("edf_b1", "edf_b4", "guarded_b4", "guarded_adaptive")}
    return main | {("tight_deadline", 301, arm) for arm in ("feasible_b4", "edf_step")}


def verify_completion_source(value):
    """Read-only audit of the exact fourteen records inherited from run01."""
    evidence, inherited = value.get("source_evidence") or {}, value.get("inherited_records", [])
    errors, sources = [], {}
    source_run = Path(evidence.get("source_run", "."))
    if not source_run.is_absolute():
        errors.append("source_run must be absolute")
    specifications = (
        ("screen", "screen_result_path", "screen_result_sha256", "screen/result.json"),
        ("calibration", "calibration_path", "calibration_sha256", "calibrate/result.json"),
        ("supervisor", "source_supervisor_path", "source_supervisor_sha256", "screen/supervisor.json"),
    )
    for label, path_field, sha_field, relative in specifications:
        path = Path(evidence.get(path_field, "."))
        if not path.is_absolute() or path != source_run / relative:
            errors.append(path_field + " differs from the explicit source run")
            continue
        try:
            if not evidence.get(sha_field) or file_digest(path) != evidence[sha_field]:
                errors.append(path_field + " digest changed")
                continue
            sources[label] = read_json(path)
        except (OSError, ValueError) as error:
            errors.append(path_field + ": " + str(error))
    screen, calibration, supervisor = (sources.get(key, {}) for key in ("screen", "calibration", "supervisor"))
    if (screen.get("status") != "incomplete" or screen.get("error_type") != "BudgetStop"
            or screen.get("active_group") or screen.get("active_trace")):
        errors.append("historical screen was not stopped at a complete-group budget boundary")
    occupancy = supervisor.get("final_occupancy") or {}
    if (supervisor.get("status") != "incomplete" or supervisor.get("returncode") != 2
            or supervisor.get("budget_stopped") is not True or supervisor.get("supervisor_failure")
            or occupancy.get("status") != "exclusive_observed" or occupancy.get("foreign") or occupancy.get("owned")):
        errors.append("historical supervisor does not establish a clean isolated budget stop")
    if canonical(screen.get("records", [])) != canonical(inherited):
        errors.append("inherited records differ from the original screen snapshot")
    if len(inherited) != 14 or {_key(record) for record in inherited} != _historical_keys():
        errors.append("expected exactly the original fourteen predeclared completed replays")
    if not all(complete_replay(record) for record in inherited):
        errors.append("historical inherited replay is incomplete or quality-unverified")
    if (calibration.get("status") != "passed" or calibration.get("eligible") is not True
            or not isinstance(calibration.get("cost_model"), dict)
            or digest(calibration.get("cost_model", {})) != calibration.get("cost_model_sha256")):
        errors.append("historical calibration/cost model is not qualified and frozen")
    for record in inherited:
        if (record.get("device_identity") != calibration.get("device_identity")
                or record.get("cost_model_sha256") != calibration.get("cost_model_sha256")):
            errors.append("inherited replay does not match historical device/cost identity")
            break
    for scenario in SCENARIOS[:3]:
        arms = COMPLETION_ARMS if scenario == "tight_deadline" else COMPLETION_ARMS[:4]
        unused, issues = _paired(inherited, arms, (scenario,), (301,))
        if issues:
            errors.append("inherited scenario is not fully paired: " + scenario)
    return dict(status="verified" if not errors else "unverified", passed=not errors, errors=errors,
                evidence=evidence, inherited_replays=len(inherited),
                device_identity=calibration.get("device_identity"),
                cost_model_sha256=calibration.get("cost_model_sha256"),
                historical_status="incomplete_budget_stop", historical_qualification_rewritten=False)


def evaluate_completion(value):
    source = verify_completion_source(value)
    records = value.get("records", [])
    inherited = value.get("inherited_records", [])
    groups, issues = _paired(records, COMPLETION_ARMS, ("video_contention",), (301,))
    expected = {("video_contention", 301, arm) for arm in COMPLETION_ARMS}
    if len(records) != 6 or {_key(record) for record in records} != expected:
        issues.append(dict(reason="completion requires exactly six newly measured predeclared arms"))
    if any(record.get("device_identity") != source["device_identity"] or
           record.get("cost_model_sha256") != source["cost_model_sha256"] for record in records):
        issues.append(dict(reason="new completion differs from the frozen historical device/cost identity"))
    valid = source["passed"] and len(groups) == 1 and not issues
    gate = evaluate_screen([*inherited, *records]) if valid else dict(status="incomplete", selected_arm=None)
    return dict(status="complete" if valid else "incomplete", technical_complete=valid, source=source, issues=issues,
                inherited_replays=len(inherited), new_replays=len(records), recomputed_original_gate=gate,
                scope="new completion matrix with explicit historical inputs; original report and original qualification remain unchanged")


def evaluate_replication(records, frozen_candidate, expected_seeds=(401, 402)):
    if frozen_candidate not in (None, *GRANULARITY_CANDIDATES):
        return dict(status="incomplete", technical_complete=False, reason="unrecognized frozen candidate", frozen_candidate=frozen_candidate,
                    candidate_validation=dict(status="unverified", passed=False), mechanism_comparisons=[], paired_groups=0)
    arms = (*REPLICATION_BASELINES, *((frozen_candidate,) if frozen_candidate else ()))
    seeds = tuple(expected_seeds)
    groups, issues = _paired(records, arms, SCENARIOS, seeds)
    expected = {(scenario, seed, arm) for scenario in SCENARIOS for seed in seeds for arm in arms}
    if len(records) != len(expected) or {_key(record) for record in records} != expected:
        issues.append(dict(reason="replication record set differs from the frozen arm/scenario/seed matrix"))
    complete = len(groups) == len(SCENARIOS) * len(seeds) and not issues
    comparisons = []
    for group in groups:
        baseline = _summary([group["members"]["edf_step"]])
        for arm in arms:
            if arm == "edf_step":
                continue
            actual = _summary([group["members"][arm]])
            comparisons.append(dict(scenario=group["scenario"], seed=group["seed"], arm=arm,
                goodput_change_vs_step=_gain(actual["overall"]["goodput_rps"], baseline["overall"]["goodput_rps"]),
                image_wait_p95_change_ms=actual["image"]["wait_p95_ms"] - baseline["image"]["wait_p95_ms"],
                image_latency_p95_change_ms=actual["image"]["latency_p95_ms"] - baseline["image"]["latency_p95_ms"],
                video_on_time_change=actual["video"]["on_time"] - baseline["video"]["on_time"],
                scope="prespecified mechanism replication; not candidate reselection"))
    validation = (_candidate(records, frozen_candidate, seeds, validate=True) if frozen_candidate else
                  dict(status="not_selected", passed=False, reason="no development-qualified new candidate"))
    return dict(status="complete" if complete else "incomplete", technical_complete=complete,
                frozen_candidate=frozen_candidate, expected_groups=len(SCENARIOS) * len(seeds), paired_groups=len(groups),
                issues=issues, mechanism_comparisons=comparisons, candidate_validation=validation,
                reselection_allowed=False,
                scope="fresh seeds401/402; fixed EDF 1/4/step mechanism replication always runs; only the frozen development-qualified candidate is added")


def _supervised(value):
    return bool(value and value.get("status") in ("passed", "stopped") and value.get("returncode") == 0
                and value.get("technical_complete", value.get("eligible")) is True and not value.get("supervisor_failure")
                and not value.get("watchdog_stopped") and not value.get("budget_stopped"))


def _phase(root, phase, errors):
    path = root / phase / "result.json"
    value = _read(path, errors) or dict(status="not_run", records=[])
    supervisor_path = root / phase / "supervisor.json"
    supervisor = _read(supervisor_path, errors)
    admission = _read(root / phase / "admission.json", errors)
    authority = supervisor or admission or value
    records = _phase_records(value)
    summaries = [dict(scenario=record.get("scenario"), seed=record.get("seed"), arm=record.get("arm"),
        complete=complete_replay(record), status=record.get("status"), quality=record.get("quality"),
        trace_sha256=record.get("trace_sha256"), device_identity=record.get("device_identity"),
        cost_model_sha256=record.get("cost_model_sha256"), metrics=_summary([record]),
        requests=[_compact_request(row) for row in record.get("rows", [])],
        execution_events=record.get("execution_events"), reason=record.get("reason")) for record in records]
    bound = bool(supervisor and all(canonical(supervisor.get(key)) == canonical(value.get(key)) for key in (
        "records", "device_identity", "cost_model_sha256", "inherited_records", "source_evidence", "selection", "frozen_candidate")))
    if supervisor and not bound:
        errors.append(dict(path=str(path), error="local measurements differ from the supervised snapshot"))
    return dict(status=authority.get("status"), reason=authority.get("reason"),
                supervised=_supervised(supervisor) and bound, source_path=str(path) if path.exists() else None,
                source_sha256=file_digest(path) if path.exists() else None,
                supervisor_path=str(supervisor_path) if supervisor else None,
                supervisor_sha256=file_digest(supervisor_path) if supervisor_path.exists() else None,
                summaries=summaries, active_group=value.get("active_group"), active_trace=value.get("active_trace")), value


def build_report(root, ledger=None):
    root = Path(root).resolve()
    errors, phases, measurements = [], {}, {}
    for phase in PHASES:
        phases[phase], measurements[phase] = _phase(root, phase, errors)
    completion = evaluate_completion(measurements["completion"])
    completion["supervised"] = phases["completion"]["supervised"]
    if not completion["supervised"]:
        completion["recomputed_original_gate"] = dict(status="incomplete", selected_arm=None,
                                                      reason="completion lacks successful matching supervision")
    granularity = evaluate_granularity(measurements["granularity"].get("records", []))
    recorded_selection = measurements["granularity"].get("selection")
    selection_matches = canonical(recorded_selection) == canonical(granularity)
    granularity_qualified = phases["granularity"]["supervised"] and completion["technical_complete"] and completion["supervised"]
    if not granularity_qualified or not selection_matches:
        if recorded_selection and not selection_matches:
            errors.append(dict(path=str(root / "granularity/result.json"), error="saved selection differs from recomputed fixed development gate"))
        granularity = dict(granularity, status="incomplete", passed=False, selected_candidate=None,
                           reason="development lacks matching complete supervision, source completion, or selection")
    frozen = measurements["replicate"].get("frozen_candidate")
    replication = evaluate_replication(measurements["replicate"].get("records", []), frozen)
    selection_bound = granularity_qualified and selection_matches and frozen == granularity["selected_candidate"]
    replication["supervised"] = phases["replicate"]["supervised"]
    replication["frozen_selection_verified"] = selection_bound
    if not replication["supervised"] or not selection_bound:
        replication["candidate_validation"] = dict(status="unverified", passed=False,
            reason="replication supervision or frozen development selection is unverified")
    budget_path = Path(ledger).resolve() if ledger is not None else root.parent / "gpu-budget.json"
    budget_value = _read(budget_path, errors)
    budget = budget_summary(budget_value) if budget_value else dict(status="unavailable")
    budget["budget_scope"] = "new unlimited V100 authorization; original 12 GPU-hour ledger is not modified or relabeled"
    budget["unlimited"] = bool(budget_value and budget_value.get("limit_gpu_seconds") is None)
    return dict(schema_version="ditserve.scheduling.next.report.v1", output=str(root), phases=phases,
                completion=completion, granularity=granularity, replication=replication,
                budget_path=str(budget_path), budget_summary=budget, evidence_errors=errors,
                status="complete" if replication["technical_complete"] and replication["supervised"] and selection_bound
                else "in_progress_or_incomplete",
                scope="complete pairs only for fixed gates; descriptive partial measurements remain visible; no stable p99, significance or cross-device claims")


def render_markdown(value):
    lines = ["# V100 调度续轮实验", "", "状态：%s。" % value["status"], "",
             "本轮独立累计 GPU 时间，无预算上限；旧 12 GPU·小时账本与旧报告保持原样。", ""]
    for phase in PHASES:
        item = value["phases"][phase]
        lines += ["## %s：%s" % (phase, item["status"]), ""]
        if item.get("reason"):
            lines += [str(item["reason"]), ""]
        if not item["summaries"]:
            lines += ["尚无新完成重放。", ""]
            continue
        lines += ["| 场景 | seed | 策略 | 完整且质量通过 | 按时/到达 | goodput 请求/秒 | 图像等待 p95 ms | 图像延迟 p95 ms | 视频按时/到达 | 已测开销占比 |",
                  "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in item["summaries"]:
            metrics = row["metrics"]
            overall, image, video = (metrics[key] for key in ("overall", "image", "video"))
            share = metrics["overhead"]["tracked_overhead_share"]
            lines.append("| %s | %s | %s | %s | %s/%s | %s | %s | %s | %s/%s | %s |" % (
                row["scenario"], row["seed"], row["arm"], row["complete"], overall["on_time"], overall["arrived"],
                _fmt(overall["goodput_rps"]), _fmt(image["wait_p95_ms"]), _fmt(image["latency_p95_ms"]),
                video["on_time"], video["arrived"], "%.3f%%" % (100 * share) if share is not None else "未测"))
        lines += ["", "来源：%s；SHA256：`%s`；成功监督：%s。" % (
            item["source_path"], item["source_sha256"], item["supervised"]), ""]
    lines += ["## 固定门槛与复现", "",
        "原矩阵补齐：%s；继承 %s 条、新测 %s 条；原候选重算门槛：%s。旧资格没有回写。" % (
            value["completion"]["status"], value["completion"]["inherited_replays"], value["completion"]["new_replays"],
            value["completion"]["recomputed_original_gate"]["status"]), "",
        "EDF 两 block／EDF 单 block 后台写入的开发门槛：%s；冻结候选：%s。" % (
            value["granularity"]["status"], value["granularity"]["selected_candidate"] or "无"), "",
        "新 seed 机制复现：%s；冻结候选验证：%s。固定 1/4/整步的复现不用于重选失败候选。" % (
            value["replication"]["status"], value["replication"]["candidate_validation"]["status"]), ""]
    seconds = value["budget_summary"].get("including_active_gpu_seconds")
    lines += ["新账本累计 GPU 时间：%s 小时（%s）。" % (
        _fmt(seconds / 3600 if _number(seconds) else None), value["budget_summary"]["status"]), "",
        "goodput 分母含完整 drain 与空闲。前台开销计调度、检查点、同步写入或后台提交／背压，以及最终 drain。后台实际写入与模型工作重叠，单列且不计入前台合计。加载、预热仍由 GPU 账本计费。阶段等待存在重叠，不相加冒充端到端延迟。p95 仅为有限样本描述。"]
    if value["evidence_errors"]:
        lines += ["", "来源／读取问题：", "", *["- %s：%s" % (item["path"], item["error"]) for item in value["evidence_errors"]]]
    if value["completion"]["source"]["errors"]:
        lines += ["", "历史来源尚未通过：", "", *["- " + error for error in value["completion"]["source"]["errors"]]]
    return "\n".join(lines) + "\n"


def _event_summary(root, record):
    descriptor = record.get("execution_events") or {}
    path = Path(descriptor.get("path", "."))
    if not path.is_absolute() or path.resolve() != path or root not in path.parents:
        raise ValueError("event source is not a canonical file inside this new study")
    if path.name != "execution-events.jsonl" or path.suffix != ".jsonl":
        raise ValueError("only explicit execution-events.jsonl sources may be summarized")
    relative = path.relative_to(root).as_posix()
    if Path(relative).parts[0] not in PHASES:
        raise ValueError("event source is outside a new study phase")
    before = path.stat()
    checksum = hashlib.sha256()
    count, malformed, missing, omitted = 0, 0, Counter(), Counter()
    grouped, observed_actions = {}, Counter()
    known = ("request_id", "dispatch_ms", "end_ms", "action", "quantum", "reason")
    with path.open("rb") as stream:
        for line in stream:
            checksum.update(line)
            count += 1
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("event is not an object")
            except (ValueError, UnicodeDecodeError):
                malformed += 1
                continue
            missing.update(key for key in known if key not in event)
            omitted.update(key for key in event if key not in known)
            if (not all(key in event for key in known) or not isinstance(event.get("request_id"), str)
                    or not isinstance(event.get("action"), str) or not isinstance(event.get("reason"), str)
                    or not _number(event.get("dispatch_ms")) or not _number(event.get("end_ms"))
                    or event["dispatch_ms"] < 0 or event["end_ms"] < event["dispatch_ms"]):
                malformed += 1
                continue
            key = (event["request_id"], event["action"], str(event["quantum"]), event["reason"])
            duration = event["end_ms"] - event["dispatch_ms"]
            item = grouped.setdefault(key, dict(request_id=key[0], action=key[1], quantum=key[2], reason=key[3],
                                                count=0, duration_sum_ms=0., duration_min_ms=duration, duration_max_ms=duration))
            item["count"] += 1
            item["duration_sum_ms"] += duration
            item["duration_min_ms"] = min(item["duration_min_ms"], duration)
            item["duration_max_ms"] = max(item["duration_max_ms"], duration)
            observed_actions[(event["request_id"], event["action"])] += 1
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("event source changed during bounded-memory aggregation")
    errors = []
    if checksum.hexdigest() != descriptor.get("sha256"):
        errors.append("event source digest differs from replay evidence")
    if count != descriptor.get("count"):
        errors.append("event source count differs from replay evidence")
    expected_actions = Counter({(row["request_id"], action): n for row in record.get("rows", [])
                                for action, n in (row.get("action_counts") or {}).items()})
    if observed_actions != expected_actions:
        errors.append("per-request action inventory differs from replay evidence")
    if malformed or missing:
        errors.append("event source contains malformed or missing required fields")
    for item in grouped.values():
        item["duration_mean_ms"] = item["duration_sum_ms"] / item["count"]
    return relative, dict(valid=not errors, schema_version="ditserve.scheduling.next.event-summary.v1",
        source_metadata=dict(source=str(path), size=after.st_size, sha256=checksum.hexdigest()),
        scenario=record.get("scenario"), seed=record.get("seed"), arm=record.get("arm"),
        trace_sha256=record.get("trace_sha256"), device_identity=record.get("device_identity"),
        events=count, malformed_records=malformed, missing_fields=dict(missing), omitted_fields=dict(omitted),
        groups=[grouped[key] for key in sorted(grouped)], errors=errors,
        individual_events_preserved=False, reconstructed_event_order_available=False,
        limitation="after explicit deletion, only aggregate counts/durations and original hash remain; individual events cannot be reconstructed")


def collect_event_summaries(root):
    """Return only verified new-run event summaries; never traverse source_run."""
    root = Path(root).resolve()
    summaries, errors = {}, []
    for phase in PHASES:
        result_path, supervisor_path = root / phase / "result.json", root / phase / "supervisor.json"
        if not result_path.exists() or not supervisor_path.exists():
            continue
        value, supervisor = read_json(result_path), read_json(supervisor_path)
        if not _supervised(supervisor) or canonical(value.get("records")) != canonical(supervisor.get("records")):
            continue
        for record in value.get("records", []):
            if not complete_replay(record) or not record.get("execution_events"):
                continue
            try:
                relative, summary = _event_summary(root, record)
                if not summary["valid"]:
                    raise ValueError("; ".join(summary["errors"]))
                if relative in summaries:
                    raise ValueError("one event source is referenced by multiple replays")
                summaries[relative] = summary
            except (OSError, ValueError) as error:
                errors.append(dict(path=(record.get("execution_events") or {}).get("path"), error=str(error)))
    return dict(summaries=summaries, errors=errors, cleanup_summarized_paths=sorted(summaries))


def report(root, ledger=None):
    value = build_report(root, ledger)
    root = Path(root)
    atomic_json(root / "report.json", value)
    (root / "REPORT.md").write_text(render_markdown(value), encoding="utf-8")
    exported = export_report(root, ledger=value["budget_path"])
    cleanup_paths, event_errors = [], []
    if value["status"] == "complete" and exported.get("status") == "verified":
        from ..compact_results import export
        events = collect_event_summaries(root)
        event_errors = events["errors"]
        if events["summaries"]:
            try:
                exported = export(root, root / "excel", replace_existing=True, summarized_text=events["summaries"])
                if exported.get("status") == "verified":
                    cleanup_paths = events["cleanup_summarized_paths"]
            except (OSError, ValueError, RuntimeError) as error:
                event_errors.append(dict(path=str(root), error=str(error)))
    return dict(value, excel=exported, cleanup_summarized_paths=cleanup_paths, event_summary_errors=event_errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export the V100 continuation study report")
    parser.add_argument("root", type=Path)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args(argv)
    result = report(args.root, args.ledger)
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
