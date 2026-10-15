"""CPU-only, provenance-aware paired evaluation of scheduling experiments.

Failed and unfinished attempts remain visible. They never become a smaller,
easier evaluation matrix; only complete pairs with identical inputs qualify.
"""

from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics

from ..directions.reporting import budget_summary
from ..research.common import atomic_json, canonical, file_digest, read_json
from ..result_storage import export_report


BASELINES = ("edf_b1", "edf_b4")
CANDIDATES = ("guarded_b4", "guarded_adaptive")
SCENARIOS = ("original_burst", "tight_deadline", "original_ordinary", "video_contention")
ORIGINAL = ("original_burst", "original_ordinary")
PRESSURE = ("tight_deadline", "video_contention")
PHASES = ("calibrate", "pilot", "screen", "validate", "legacy")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _stats(values):
    values = sorted(value for value in values if _number(value))
    if not values:
        return dict(n=0, mean=None, p95=None, maximum=None)
    location = (len(values) - 1) * .95
    low, high = math.floor(location), math.ceil(location)
    return dict(n=len(values), mean=statistics.mean(values),
                p95=values[low] + (values[high] - values[low]) * (location - low), maximum=values[-1])


def _quality(record):
    return isinstance(record.get("quality"), dict) and record["quality"].get("passed") is True


def complete_replay(record, expected_requests=24, require_quality=True):
    rows = record.get("rows", [])
    if (record.get("complete") is False or record.get("status") in
            ("failed", "running", "incomplete", "error", "stopped")
            or not _number(record.get("elapsed_ms")) or record["elapsed_ms"] <= 0
            or not isinstance(rows, list) or len(rows) != expected_requests
            or (require_quality and not _quality(record))):
        return False
    ids = [row.get("request_id") for row in rows]
    if len(set(ids)) != len(rows) or any(not isinstance(rid, str) or not rid for rid in ids):
        return False
    return all(row.get("arrived", True) and row.get("status") == "completed"
               and row.get("kind") in ("image", "video")
               and all(_number(row.get(key)) for key in ("arrival_ms", "start_ms", "finish_ms", "deadline_ms"))
               and 0 <= row["arrival_ms"] <= row["start_ms"] <= row["finish_ms"] <= record["elapsed_ms"]
               and row["deadline_ms"] >= row["arrival_ms"] for row in rows)


def request_metrics(rows, elapsed_ms):
    """Failures remain in attainment denominators, even outside paired gates."""
    arrived = [row for row in rows if row.get("arrived", True)]
    complete = [row for row in arrived if row.get("status") == "completed"
                and all(_number(row.get(key)) for key in ("arrival_ms", "finish_ms", "deadline_ms"))]
    timely = sum(row["finish_ms"] <= row["deadline_ms"] for row in complete)
    latency = _stats(row["finish_ms"] - row["arrival_ms"] for row in complete)
    wait = _stats(row["start_ms"] - row["arrival_ms"] for row in arrived
                  if _number(row.get("start_ms")) and _number(row.get("arrival_ms")))
    timing = defaultdict(list)
    for row in arrived:
        for key, value in {**row, **(row.get("timing") or {}), **(row.get("timings") or {})}.items():
            if key.endswith("_ms") and _number(value) and key not in (
                    "arrival_ms", "deadline_ms", "start_ms", "finish_ms"):
                timing[key].append(value)
    return dict(arrived=len(arrived), completed=len(complete), on_time=timely,
                failed_or_incomplete=len(arrived) - len(complete),
                on_time_fraction=timely / len(arrived) if arrived else None,
                goodput_rps=timely * 1000 / elapsed_ms if _number(elapsed_ms) and elapsed_ms > 0 else None,
                latency_ms=latency, wait_ms=wait, latency_p95_ms=latency["p95"], wait_p95_ms=wait["p95"],
                timings={key: _stats(values) for key, values in sorted(timing.items())},
                pause_max_ms=max((row.get("pause_max_ms", 0.) for row in arrived
                                  if _number(row.get("pause_max_ms", 0.))), default=None),
                sample_count=len(complete), tail_claim="descriptive p95; no stable p99 or significance claim")


def aggregate(records):
    rows = [row for record in records for row in record.get("rows", [])]
    times = [record.get("elapsed_ms") for record in records]
    elapsed = sum(times) if times and all(_number(value) and value > 0 for value in times) else None
    triggers = Counter()
    for record in records:
        triggers.update({key: value for key, value in
                         (record.get("policy_triggers") or record.get("trigger_counts") or {}).items()
                         if type(value) is int and value >= 0})
    return dict(replays=len(records), elapsed_ms=elapsed,
                overall=request_metrics(rows, elapsed),
                image=request_metrics([row for row in rows if row.get("kind") == "image"], elapsed),
                video=request_metrics([row for row in rows if row.get("kind") == "video"], elapsed),
                policy_triggers=dict(triggers))


def _inputs(record):
    keys = ("request_id", "kind", "arrival_ms", "deadline_ms", "seed", "prompt", "prompt_id", "service_ms")
    return sorted(({key: row[key] for key in keys if key in row} for row in record.get("rows", [])),
                  key=lambda row: row["request_id"])


def _paired(records, arms, scenarios, seeds):
    indexed = defaultdict(list)
    for record in records:
        indexed[(record.get("scenario"), record.get("seed"), record.get("arm"))].append(record)
    groups, issues = [], []
    identity = None
    for scenario in scenarios:
        for seed in seeds:
            members = []
            for arm in arms:
                matches = indexed[(scenario, seed, arm)]
                if len(matches) != 1:
                    issues.append(dict(scenario=scenario, seed=seed, arm=arm,
                                       reason="missing" if not matches else "duplicate", observed=len(matches)))
                elif not complete_replay(matches[0]):
                    issues.append(dict(scenario=scenario, seed=seed, arm=arm,
                                       reason="unfinished_failed_or_quality_unverified"))
                else:
                    members.append(matches[0])
            if len(members) != len(arms):
                continue
            reference = members[0]
            provenance = tuple(canonical(reference.get(key)) for key in ("device_identity", "cost_model_sha256"))
            valid = all(reference.get(key) for key in ("device_identity", "cost_model_sha256", "trace_sha256"))
            videos = 3 if scenario == "video_contention" else 2
            valid = valid and Counter(row["kind"] for row in reference["rows"]) == dict(video=videos, image=24-videos)
            valid = valid and all(all(member.get(key) == reference.get(key) for key in
                                     ("device_identity", "cost_model_sha256", "trace_sha256"))
                                  and _inputs(member) == _inputs(reference) for member in members)
            # One evaluation cohort cannot pool differently calibrated devices.
            if identity is None and valid:
                identity = provenance
            if not valid or provenance != identity:
                issues.append(dict(scenario=scenario, seed=seed, reason="input_or_device_provenance_mismatch"))
                continue
            groups.append(dict(scenario=scenario, seed=seed, members={r["arm"]: r for r in members}))
    return groups, issues


def _gain(actual, reference):
    if not _number(actual) or not _number(reference) or reference <= 0:
        return None
    return actual / reference - 1


def _candidate(records, arm, seeds, *, validate=False):
    groups, issues = _paired(records, (*BASELINES, arm), SCENARIOS, seeds)
    result = dict(arm=arm, passed=False, status="incomplete", expected_groups=len(SCENARIOS) * len(seeds),
                  paired_groups=len(groups), issues=issues, checks=[], metrics={})
    if issues or len(groups) != result["expected_groups"]:
        return result
    for name in (*SCENARIOS, "pressure", "overall"):
        chosen = [g for g in groups if name == "overall" or name == "pressure" and g["scenario"] in PRESSURE
                  or g["scenario"] == name]
        result["metrics"][name] = {a: aggregate([g["members"][a] for g in chosen]) for a in (*BASELINES, arm)}
    for baseline in BASELINES:
        pressure = result["metrics"]["pressure"]
        gain = _gain(pressure[arm]["overall"]["goodput_rps"], pressure[baseline]["overall"]["goodput_rps"])
        result["checks"].append(dict(name="pressure_goodput", baseline=baseline, gain=gain,
                                     minimum=.05, passed=gain is not None and gain >= .05 - 1e-12))
        for scenario in ORIGINAL:
            metrics = result["metrics"][scenario]
            gain = _gain(metrics[arm]["overall"]["goodput_rps"], metrics[baseline]["overall"]["goodput_rps"])
            result["checks"].append(dict(name="original_goodput", scenario=scenario, baseline=baseline,
                                         gain=gain, minimum=-.02, passed=gain is not None and gain >= -.02 - 1e-12))
        for scenario in SCENARIOS:
            metrics = result["metrics"][scenario]
            actual, expected = metrics[arm]["video"]["on_time"], metrics[baseline]["video"]["on_time"]
            result["checks"].append(dict(name="video_attainment", scenario=scenario, baseline=baseline,
                                         actual=actual, minimum=expected, passed=actual >= expected))
        if validate:
            for seed in seeds:
                selected = [g for g in groups if g["scenario"] in PRESSURE and g["seed"] == seed]
                actual = aggregate([g["members"][arm] for g in selected])["overall"]["goodput_rps"]
                expected = aggregate([g["members"][baseline] for g in selected])["overall"]["goodput_rps"]
                gain = _gain(actual, expected)
                result["checks"].append(dict(name="positive_seed_pressure_gain", seed=seed,
                                             baseline=baseline, gain=gain, passed=gain is not None and gain > 0))
    result["passed"] = all(check["passed"] for check in result["checks"])
    result["status"] = "passed" if result["passed"] else "failed"
    return result


def evaluate_screen(records, expected_seeds=(301,)):
    candidates = {arm: _candidate(records, arm, tuple(expected_seeds)) for arm in CANDIDATES}
    # Selection is frozen only after both predeclared candidates finish screening.
    if any(item["status"] == "incomplete" for item in candidates.values()):
        return dict(status="incomplete", selected_arm=None, candidates=candidates)
    passed = [arm for arm, result in candidates.items() if result["passed"]]
    selected = None
    if passed:
        scores = {arm: candidates[arm]["metrics"]["overall"][arm]["overall"]["goodput_rps"] for arm in passed}
        selected = max(passed, key=lambda arm: scores[arm])
        if len(passed) == 2 and abs(scores[CANDIDATES[0]] - scores[CANDIDATES[1]]) / max(scores.values()) < .01:
            selected = "guarded_b4"
    return dict(status="passed" if selected else "failed", selected_arm=selected, candidates=candidates,
                selection_rule="overall goodput; under 1% difference chooses guarded_b4; no holdout reselection")


def evaluate_validate(records, selected_arm, expected_seeds=(401, 402)):
    if selected_arm not in CANDIDATES:
        return dict(status="not_eligible", passed=False, selected_arm=selected_arm,
                    reason="development has no frozen passing selection")
    result = _candidate(records, selected_arm, tuple(expected_seeds), validate=True)
    return dict(result, selected_arm=selected_arm, reselection_allowed=False)


def _read(path, errors):
    if not path.is_file():
        return None
    try:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("evidence must be an object")
        return value
    except (OSError, ValueError) as error:
        errors.append(dict(path=str(path), error=str(error)))
        return dict(status="unreadable", reason=str(error))


def _phase_records(value):
    return [dict(record, **{key: record.get(key, value.get(key)) for key in
                           ("device_identity", "cost_model_sha256")}) for record in value.get("records", [])]


def _compact_request(row):
    """Retain per-request evidence without repeating thousands of action spans."""
    selected = {key: value for key, value in row.items() if value is None or type(value) in (str, int, float, bool)
                or key in ("timing", "timings", "stage_ms", "blocked_by_ms", "action_counts", "outputs", "output_sha256", "quality")}
    selected["action_spans_count"] = len(row.get("action_spans", []))
    selected["individual_action_spans_included"] = False
    return selected


def _combine(pilot, screen):
    """Identical inherited snapshots are one observation; conflicting repeats fail."""
    result, fingerprints = [], set()
    for record in [*pilot, *screen]:
        fingerprint = canonical(record)
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            result.append(record)
    return result


def build_report(output, ledger=None):
    root = Path(output).resolve()
    errors, phases = [], {}
    for phase in PHASES:
        path = root / phase / "result.json"
        value = _read(path, errors) or dict(status="not_run")
        supervisor_path = root / phase / "supervisor.json"
        supervisor = _read(supervisor_path, errors)
        admission = _read(root / phase / "admission.json", errors)
        authority = supervisor or admission or value
        supervised = bool(supervisor and supervisor.get("status") in ("passed", "stopped")
                          and supervisor.get("returncode") == 0 and not supervisor.get("supervisor_failure")
                          and not supervisor.get("budget_stopped"))
        records = _phase_records(value)
        summaries = []
        for record in records:
            summaries.append(dict(scenario=record.get("scenario"), seed=record.get("seed"), arm=record.get("arm"),
                device_identity=record.get("device_identity"), trace_sha256=record.get("trace_sha256"),
                cost_model_sha256=record.get("cost_model_sha256"), status=record.get("status"),
                complete=complete_replay(record), quality=record.get("quality"),
                metrics=aggregate([record]), requests=[_compact_request(row) for row in record.get("rows", [])],
                error=record.get("error"),
                action_prediction_error_ms={action: _stats(sample.get("error_ms") for sample in record.get("action_prediction_samples", [])
                                                           if sample.get("action") == action)
                    for action in sorted({sample.get("action", "unknown") for sample in record.get("action_prediction_samples", [])})}))
        phases[phase] = dict(status=authority.get("status"), reason=authority.get("reason"),
                             reason_code=authority.get("reason_code"), deferred=authority.get("deferred", False),
                             path=str(path) if path.exists() else None,
                             sha256=file_digest(path) if path.exists() else None,
                             records=records, summaries=summaries, active=value.get("active_trace"),
                             supervised=supervised,
                             supervisor={key: supervisor[key] for key in (
                                 "status", "eligible", "reason", "returncode", "supervisor_failure", "charge",
                                 "main_config_sha256", "runtime_config_sha256", "source_sha256", "gpu_uuid",
                                 "sana_profile", "budget_stopped") if key in supervisor} if supervisor else None,
                             authority_path=str(supervisor_path) if supervisor else None,
                             details={key: value[key] for key in (
                                 "eligible", "passed", "decision", "paired_groups", "inherited_replays", "new_replays",
                                 "source", "plan_sha256", "cost_model_sha256", "device_identity") if key in value})
    calibration = phases["calibrate"]
    qualified = calibration["supervised"] and calibration["supervisor"].get("eligible") is True
    def admitted(phase):
        records = phases[phase]["records"]
        return records if qualified and phases[phase]["supervised"] else [dict(record, complete=False) for record in records]
    development = _combine(admitted("pilot"), admitted("screen"))
    screen = evaluate_screen(development)
    validate_value = _read(root / "validate" / "result.json", errors) or {}
    selected = validate_value.get("selected_arm", screen["selected_arm"])
    validation = evaluate_validate(admitted("validate"), selected)
    if selected != screen["selected_arm"] or screen["status"] != "passed":
        validation.update(status="not_eligible", passed=False, reason="holdout selection differs from completed development")
    pilot_groups, pilot_issues = _paired(admitted("pilot"), (*BASELINES, "guarded_b4"),
                                        ("original_burst", "tight_deadline"), (301,))
    pilot = dict(status="complete" if len(pilot_groups) == 2 and not pilot_issues else "incomplete",
                 paired_groups=len(pilot_groups), issues=pilot_issues, descriptive_only=True, qualified=False,
                 metrics={arm: aggregate([g["members"][arm] for g in pilot_groups]) for arm in (*BASELINES, "guarded_b4")})
    budget_path = Path(ledger).resolve() if ledger is not None else root.parent / "gpu-budget.json"
    budget = _read(budget_path, errors)
    try:
        budget_value = budget_summary(budget) if budget and "used_gpu_seconds" in budget else dict(status="unavailable")
    except (TypeError, ValueError, KeyError) as error:
        budget_value = dict(status="unavailable", reason=str(error))
    # Original phase files remain the per-request source. Avoid embedding their
    # full records a second time in the lightweight report and Excel archive.
    for value in phases.values():
        value.pop("records")
    return dict(schema_version="ditserve.scheduling.report.v1", output=str(root), phases=phases,
                pilot=pilot, screen=screen, validate=validation, budget_path=str(budget_path),
                budget_summary=budget_value, evidence_errors=errors,
                scope="single seed is exploratory; no stable p99, significance or cross-device performance claim",
                status="validated" if validation.get("passed") else "screened" if screen["status"] == "passed"
                else "pilot_complete" if pilot["status"] == "complete" else "incomplete")


def _fmt(value):
    return "未测" if value is None else "%.4f" % value if type(value) is float else str(value)


def render_markdown(value):
    lines = ["# 单 seed 优先的抢占与调度实验", "",
             "状态：%s。首批单 seed 结果仅用于判断方向，不形成稳定收益结论。" % value["status"], "",
             "goodput 使用完整 trace 墙钟时间（包括 drain）；失败和未完成请求保留在到达分母。只有同设备、同成本模型、同输入且质量通过的完整配对参与门槛比较。", ""]
    for phase in PHASES:
        item = value["phases"][phase]
        lines += ["## %s：%s" % (phase, item["status"]), ""]
        if item.get("reason"):
            lines += [str(item["reason"]), ""]
        if phase == "legacy" and (item.get("deferred") or item.get("reason_code") == "device_constraint"):
            lines += ["旧 A100 矩阵因本轮设备约束暂缓；不以 V100 补入旧 A100 配对。", ""]
        if not item["summaries"]:
            lines += ["暂无在线 replay；阶段源文件：%s。" % (item["path"] or "未运行"), ""]
            if item["details"].get("decision"):
                lines += ["判定：%s；完整配对 %s，继承 replay %s，新测 replay %s。" % (
                    item["details"]["decision"], _fmt(item["details"].get("paired_groups")),
                    _fmt(item["details"].get("inherited_replays")), _fmt(item["details"].get("new_replays"))), ""]
            continue
        lines += ["| 场景 | seed | 策略 | 完整且质量通过 | goodput 请求/秒 | 按时/到达 | 图像等待 p95 ms | 图像延迟 p95 ms | 视频按时/到达 | 最长暂停 ms | 策略触发次数 |",
                  "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for row in item["summaries"]:
            metrics = row["metrics"]
            allm, image, video = (metrics[key] for key in ("overall", "image", "video"))
            lines.append("| %s | %s | %s | %s | %s | %s/%s | %s | %s | %s/%s | %s | %s |" % (
                row["scenario"], row["seed"], row["arm"], row["complete"], _fmt(allm["goodput_rps"]),
                allm["on_time"], allm["arrived"], _fmt(image["wait_p95_ms"]), _fmt(image["latency_p95_ms"]),
                video["on_time"], video["arrived"], _fmt(allm["pause_max_ms"]),
                ", ".join("%s:%s" % pair for pair in sorted(metrics["policy_triggers"].items())) or "未测"))
        lines += ["", "来源：%s；SHA256：`%s`。" % (item["path"], item["sha256"]), ""]
    lines += ["## 门槛与预算", "", "开发状态：%s；冻结候选：%s；留出状态：%s。" % (
        value["screen"]["status"], value["screen"]["selected_arm"] or "无", value["validate"]["status"]), ""]
    for arm, candidate in value["screen"]["candidates"].items():
        lines += ["- %s：%s，完整配对 %s/%s。" % (arm, candidate["status"], candidate["paired_groups"], candidate["expected_groups"])]
    budget = value["budget_summary"]
    seconds = budget.get("including_active_gpu_seconds")
    lines += ["", "累计 GPU 时间：%s 小时；账本状态：%s。" % (_fmt(seconds / 3600 if _number(seconds) else None), budget["status"]),
              "", "p95 保留有限样本口径；编码、GPU 与暂停等待可能重叠，不将阶段耗时相加当作端到端延迟。"]
    if value["evidence_errors"]:
        lines += ["", "读取错误（不回退为成功）：", "", *["- %s：%s" % (e["path"], e["error"]) for e in value["evidence_errors"]]]
    return "\n".join(lines) + "\n"


def report(output, ledger=None):
    value = build_report(output, ledger)
    root = Path(output)
    atomic_json(root / "report.json", value)
    (root / "REPORT.md").write_text(render_markdown(value), encoding="utf-8")
    exported = export_report(root, ledger=value["budget_path"])
    return dict(value, excel=exported)
