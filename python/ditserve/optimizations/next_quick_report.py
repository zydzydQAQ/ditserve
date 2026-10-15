"""Small source-bound report for the explicitly amended seed-101 screening.

This report never grants serving qualification, settles a GPU charge, changes
the original 24-replay denominator, or writes the parent study report.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

from ..research.common import atomic_json, file_digest
from .next_report import _write_excel
from .online import paired_gate, request_identity, request_metrics


def _read(path):
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_file() or path.stat().st_size > 128*1024*1024:
        raise ValueError("quick report source must be a bounded ordinary file")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if file_digest(path) != sha:
        raise ValueError("quick report source changed while reading")
    return json.loads(raw), dict(path=str(path), sha256=sha, bytes=len(raw))


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def collect_quick(attempt, *, amendment_binding=None):
    attempt = Path(attempt).absolute()
    if attempt.resolve() != attempt or attempt.parent.name != "cache_load_09":
        raise ValueError("quick report requires the load-0.9 attempt directory")
    data, native_source = _read(attempt/"cache_load_09.json")
    binding = amendment_binding or data.get("amendment_binding")
    if not isinstance(binding, dict) or not binding.get("path") or not binding.get("sha256"):
        raise ValueError("quick report requires the explicit frozen amendment binding")
    amendment, amendment_source = _read(binding["path"])
    if amendment_source["sha256"] != binding["sha256"]:
        raise ValueError("quick screening amendment SHA256 differs")
    if (data.get("protocol_mode") != "single_seed_screening" or data.get("selected_seed") != 101
            or data.get("load") != .9 or data.get("planned_replays") != 8
            or data.get("planned_requests") != 192
            or any(data.get(key) is True for key in ("eligible", "online_qualified", "full_qualification_completed"))):
        raise ValueError("quick screening scope or qualification differs")
    identities = data.get("expected_trace_ids", [])
    if len(identities) != 2 or len(set(identities)) != 2:
        raise ValueError("quick screening requires two frozen trace identities")
    expected = {(trace, order, arm) for trace in identities for order in (0, 1)
                for arm in ("baseline", "candidate")}
    rows, seen, patterns, replays = data.get("rows", []), set(), {}, []
    for row in rows:
        key = row.get("trace_sha256"), row.get("order_index"), row.get("arm")
        requests, elapsed = row.get("requests", []), row.get("elapsed_ms")
        if (key not in expected or key in seen or row.get("seed") != 101 or row.get("load") != .9
                or row.get("pattern") not in ("ordinary", "burst")
                or len(requests) != 24 or sum(r.get("kind") == "video" for r in requests) != 2
                or request_identity(requests) != key[0] or not _number(elapsed) or elapsed <= 0):
            raise ValueError("quick replay identity or frozen request denominator differs")
        seen.add(key)
        if key[0] in patterns and patterns[key[0]] != row["pattern"]:
            raise ValueError("quick trace pattern changed")
        patterns[key[0]] = row["pattern"]
        for request in requests:
            if request.get("status") == "completed" and (not request.get("arrived", True)
                    or not all(_number(request.get(k)) for k in ("arrival_ms", "start_ms", "finish_ms"))
                    or not 0 <= request["arrival_ms"] <= request["start_ms"] <= request["finish_ms"] <= elapsed + 1e-6):
                raise ValueError("quick completed request has invalid host interval")
        metrics = dict(metrics=request_metrics(requests, elapsed),
            images=request_metrics([r for r in requests if r["kind"] == "image"], elapsed),
            videos=request_metrics([r for r in requests if r["kind"] == "video"], elapsed))
        replays.append(dict(trace_sha256=key[0], order_index=key[1], arm=key[2], pattern=row["pattern"],
            status=row.get("status"), complete=row.get("complete"), elapsed_ms=elapsed,
            recorded_requests=len(requests), **metrics))
    if len(patterns) == 2 and set(patterns.values()) != {"ordinary", "burst"}:
        raise ValueError("quick screening lacks the two arrival patterns")
    comparison = paired_gate(rows, expected_trace_ids=identities)
    complete = bool(seen == expected and not comparison["failures"]
        and data.get("screening_complete") is True)
    recorded = data.get("screening_comparison")
    if recorded is not None and recorded != comparison:
        raise ValueError("recorded quick comparison differs from request-level recomputation")
    sources = [native_source, amendment_source]
    report = dict(schema_version="ditserve.optimizations.quick_report.v1", attempt=str(attempt),
        status="screening_completed" if complete else "screening_incomplete", selected_seed=101, load=.9,
        planned_replays=8, recorded_replays=len(rows), planned_requests=192,
        recorded_requests=sum(len(r["requests"]) for r in rows), original_planned_replays=24,
        original_planned_requests=576, original_matrix_complete=False,
        unexecuted_original_replays=24-len(rows), screening_complete=complete,
        screening_passed=bool(complete and comparison["passed"]), online_qualified=False,
        serving_default_changed=False, supervision_status="pending_supervision",
        measurement_only=True, amendment=amendment, comparison=comparison, replays=replays,
        source_bindings=sources, raw_events_in_report=False,
        scope="single preselected seed; descriptive means/p95 and counts; no robustness, p99 or default qualification claim")
    supervisor_path = attempt/"supervisor.json"
    if supervisor_path.exists():
        supervisor, source = _read(supervisor_path)
        sources.append(source)
        final = supervisor.get("final_occupancy", {})
        recorded_supervision = bool(supervisor.get("output") == str(attempt)
            and supervisor.get("returncode") == 0 and not supervisor.get("supervisor_failure")
            and supervisor.get("charge", {}).get("status") == "completed"
            and final.get("status") == "exclusive_observed" and final.get("owned") == [] and final.get("foreign") == [])
        report["supervision_status"] = "recorded_supervision_parent_audit_separate" if recorded_supervision else "supervision_not_confirmed"
    return report


def _tables(report):
    summary = [["field", "value"]]
    for name in ("status", "selected_seed", "load", "planned_replays", "recorded_replays", "planned_requests",
                 "recorded_requests", "original_planned_replays", "original_planned_requests",
                 "original_matrix_complete", "screening_complete", "screening_passed", "online_qualified",
                 "supervision_status", "scope"):
        summary.append([name, report[name]])
    pairs = [["scope", "trace", "goodput_gain", "image_p95_reduction", "video_ontime_drop", "safe", "system_gain"]]
    for scope, values in (("pooled", [report["comparison"].get("pooled")]), ("trace", report["comparison"]["traces"])):
        for value in values:
            if value:
                pairs.append([scope, value.get("trace_sha256")] + [value.get(key) for key in pairs[0][2:]])
    requests = [["trace", "pattern", "order", "arm", "kind", "arrived", "completed", "failed_or_incomplete",
                 "on_time", "goodput_rps", "latency_mean_ms", "latency_p95_ms", "sample_count"]]
    for row in report["replays"]:
        for kind, metrics in (("all", row["metrics"]), ("image", row["images"]), ("video", row["videos"])):
            requests.append([row["trace_sha256"], row["pattern"], row["order_index"], row["arm"], kind]
                + [metrics.get(key) for key in requests[0][5:]])
    return dict(Summary=summary, Paired=pairs, ReplayMetrics=requests,
        Sources=[["path", "sha256", "bytes"]]+[[s[k] for k in ("path", "sha256", "bytes")] for s in report["source_bindings"]])


def _percent(value):
    return "未验证" if value is None else "%.2f%%" % (100*value)


def render_quick(report):
    comparison = report["comparison"].get("pooled") or {}
    return ("# 负载 0.9：单种子快速初筛\n\n"
        "固定 seed=101，普通与突发到达，基线/候选各做正反执行顺序。\n\n"
        "已记录 **%d/8 次重放、%d/192 个请求**。原三种子扩展矩阵仍为 **%d/24**，未执行部分没有补作成功。\n\n"
        "初筛完整：%s；初筛门槛通过：%s。**不授予在线资格，不改变默认策略。**\n\n"
        "整体 goodput 变化：%s；图像端到端 p95 降幅：%s；视频按时完成率下降：%s。\n\n"
        "监督状态：`%s`。本报告只重算已记录请求指标；完整监督、账单与独立审计见父实验。\n\n"
        "只报告描述性均值、p95 和样本数；失败请求保留，尚未执行请求另列，不宣称跨种子稳定性。"
        "原始事件未写入 Excel，没有删除证据或修改主报告与账本。\n\n"
        "[轻量 Excel](%s) · [结构化报告](%s)\n") % (
            report["recorded_replays"], report["recorded_requests"], report["recorded_replays"],
            report["screening_complete"], report["screening_passed"], _percent(comparison.get("goodput_gain")),
            _percent(comparison.get("image_p95_reduction")), _percent(comparison.get("video_ontime_drop")),
            report["supervision_status"], Path(report["attempt"])/"quick-results.xlsx", Path(report["attempt"])/"quick-report.json")


def write_quick_report(attempt, *, amendment_binding=None):
    report = collect_quick(attempt, amendment_binding=amendment_binding)
    output = Path(report["attempt"])
    with tempfile.TemporaryDirectory(prefix=".quick-report-", dir=output) as directory:
        temporary = Path(directory)/"quick-results.xlsx"
        verification = _write_excel(temporary, _tables(report))
        for source in report["source_bindings"]:
            if file_digest(source["path"]) != source["sha256"]:
                raise ValueError("quick report source changed during export")
        temporary.replace(output/"quick-results.xlsx")
    report["excel"] = dict(path=str(output/"quick-results.xlsx"), sha256=file_digest(output/"quick-results.xlsx"), **verification)
    atomic_json(output/"quick-report.json", report)
    (output/"QUICK_REPORT.md").write_text(render_quick(report), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Report the explicitly amended single-seed cache screening")
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args(argv)
    report = write_quick_report(args.attempt)
    print(json.dumps({key: report[key] for key in ("status", "recorded_replays", "screening_passed", "online_qualified")}))


if __name__ == "__main__":
    main()
