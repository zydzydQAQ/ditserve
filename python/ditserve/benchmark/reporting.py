"""Lossless serving denominators, paired comparisons, and portable reports."""

import csv
from collections import Counter, defaultdict
import html
import json
import math
from pathlib import Path
import shutil
import zipfile

from .common import atomic_json, canonical_hash, file_hash, read_json, seal, verify_seal
from .evaluation import generation_identity, resolve_artifact_path, _run_path


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _synthetic_run(run):
    kind = run.get("measurement_kind")
    return (run.get("backend") in ("cpu-test", "core") or run.get("measured") is False
            or (isinstance(kind, str) and kind.startswith("synthetic_")))


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    left = int(index)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (index - left)


def _stats(values):
    return {"samples": len(values), "mean": sum(values) / len(values) if values else None,
            "p50": _percentile(values, .5), "p95": _percentile(values, .95), "p99": _percentile(values, .99)}


def _difference(row, end, start):
    left, right = row.get(end), row.get(start)
    if _finite(left) and _finite(right) and left >= right:
        return float(left - right)
    return None


def summarize_requests(requests, horizon_ms, qualities=None):
    """All arrivals include failed, pending, and timed-out requests.

    Completed latency excludes failures explicitly and exposes sample counts;
    it is never passed off as the latency distribution of all arrivals.
    """
    qualities = qualities or {}
    arrived = [row for row in requests if row.get("status") != "not_arrived"]
    completed = [row for row in arrived if row.get("status") == "completed"]
    deadlines = [row for row in completed if _difference(row, "finish_ms", "arrival_ms") is not None and _finite(row.get("deadline_ms"))
                 and row["finish_ms"] <= row["deadline_ms"]]
    qualified = [row for row in completed if qualities.get(str(row.get("request_id", row.get("id"))), {}).get("qualified") is True
                 and qualities.get(str(row.get("request_id", row.get("id"))), {}).get("status") == "pass"]
    qualified_ids = {str(row.get("request_id", row.get("id"))) for row in qualified}
    qualified_deadlines = [row for row in deadlines if str(row.get("request_id", row.get("id"))) in qualified_ids]
    horizon = horizon_ms if _finite(horizon_ms) and horizon_ms > 0 else None
    latency = [_difference(row, "finish_ms", "arrival_ms") for row in completed]
    service = [_difference(row, "finish_ms", "start_ms") for row in completed]
    queue = [_difference(row, "start_ms", "submit_ms") for row in arrived]
    submit = [_difference(row, "submit_ms", "arrival_ms") for row in arrived]
    statuses = Counter(row.get("status", "unknown") for row in arrived)
    quality_status = Counter(qualities.get(str(row.get("request_id", row.get("id"))), {}).get("status", "unknown") for row in arrived)
    total = len(arrived)
    rejected = sum(statuses.get(status, 0) for status in ("resource_rejected", "rejected"))
    timed_out = sum(statuses.get(status, 0) for status in ("timed_out", "timeout"))
    failed = sum(statuses.get(status, 0) for status in ("failed", "arrived_aborted"))
    return {"arrivals": total, "not_arrived": len(requests) - total, "completed": len(completed),
            "unfinished_or_failed": total - len(completed), "status_counts": dict(statuses),
            "rejected": rejected, "timed_out": timed_out, "failed": failed,
            "rejection_rate": rejected / total if total else None,
            "timeout_rate": timed_out / total if total else None,
            "failure_rate": failed / total if total else None,
            "failure_rate_statuses": ["failed", "arrived_aborted"],
            "rejection_rate_statuses": ["resource_rejected", "rejected"],
            "timeout_rate_statuses": ["timed_out", "timeout"],
            "quality_status_counts": dict(quality_status), "quality_qualified": len(qualified),
            "deadline_completed": len(deadlines), "quality_deadline_completed": len(qualified_deadlines),
            "completion_rate": len(completed) / total if total else None,
            "deadline_completion_rate": len(deadlines) / total if total else None,
            "quality_deadline_completion_rate": len(qualified_deadlines) / total if total else None,
            "throughput_per_s": len(completed) * 1000 / horizon if horizon else None,
            "deadline_goodput_per_s": len(deadlines) * 1000 / horizon if horizon else None,
            "quality_qualified_goodput_per_s": len(qualified_deadlines) * 1000 / horizon if horizon else None,
            "horizon_ms": horizon, "rate_denominator": "all_arrivals_excluding_not_arrived",
            "completed_latency_ms": _stats([value for value in latency if value is not None]),
            "completed_service_ms": _stats([value for value in service if value is not None]),
            "queue_ms": _stats([value for value in queue if value is not None]),
            "submit_delay_ms": _stats([value for value in submit if value is not None]),
            "completed_missing_latency": sum(value is None for value in latency),
            "completed_unknown_deadline": sum(not _finite(row.get("deadline_ms")) or not _finite(row.get("finish_ms")) for row in completed)}


def paired_improvements(run, summaries, trace_requests=None):
    """Only fully completed comparable policy attempts can yield improvements."""
    policies = run.get("policies", {})
    baseline_name = "fcfs" if "fcfs" in policies else next(iter(policies), None)
    if baseline_name is None:
        return []
    baseline = policies[baseline_name]
    result = []
    for name, policy in policies.items():
        if name == baseline_name:
            continue
        record = {"baseline": baseline_name, "candidate": name, "status": "excluded", "reason": None,
                  "trace_sha256": run.get("trace_sha256"), "paired_requests": 0,
                  "completed_mean_latency_improvement_fraction": None, "deadline_goodput_improvement_fraction": None}
        left = baseline.get("requests", [])
        right = policy.get("requests", [])
        if run.get("purpose") == "independent_quality_generation":
            record["reason"] = "independent_quality_generation"
        elif run.get("purpose") == "offline_diagnostic" and run.get("serving_performance_evidence") is False:
            record["reason"] = "offline_diagnostic"
        elif _synthetic_run(run):
            record["reason"] = "synthetic_backend"
        elif run.get("serving_performance_evidence") is False:
            record["reason"] = "non_serving_performance_evidence"
        elif run.get("status") != "completed" or baseline.get("status") != "completed" or policy.get("status") != "completed":
            record["reason"] = "incomplete_attempt"
        elif not left or any(row.get("status") != "completed" for row in left + right):
            record["reason"] = "not_all_trace_requests_completed"
        else:
            left_map = {str(row.get("request_id", row.get("id"))): row for row in left}
            right_map = {str(row.get("request_id", row.get("id"))): row for row in right}
            if len(left_map) != len(left) or len(right_map) != len(right) or set(left_map) != set(right_map):
                record["reason"] = "request_identity_mismatch"
            elif trace_requests is not None and set(left_map) != {str(row.get("request_id", row.get("id"))) for row in trace_requests}:
                record["reason"] = "incomplete_trace_coverage"
            elif any(not left_map[key].get("generation") or generation_identity(left_map[key].get("generation")) != generation_identity(right_map[key].get("generation")) for key in left_map):
                record["reason"] = "generation_identity_missing_or_mismatch"
            elif any(_difference(row, "finish_ms", "arrival_ms") is None for row in left + right):
                record["reason"] = "missing_completed_latency"
            else:
                lvalue = summaries[baseline_name]["completed_latency_ms"]["mean"]
                rvalue = summaries[name]["completed_latency_ms"]["mean"]
                lgood = summaries[baseline_name]["deadline_goodput_per_s"]
                rgood = summaries[name]["deadline_goodput_per_s"]
                record.update(status="included", paired_requests=len(left_map),
                              completed_mean_latency_improvement_fraction=(lvalue - rvalue) / lvalue if lvalue and lvalue > 0 else None,
                              deadline_goodput_improvement_fraction=(rgood - lgood) / lgood if lgood and lgood > 0 and rgood is not None else None)
        result.append(record)
    return result


def _load_overlay(run, run_path):
    path = run_path.parent / "evaluation.json"
    if not path.is_file():
        return {}, {"status": "not_run", "reason": "evaluation.json missing"}
    value = verify_seal(read_json(path))
    if value.get("run_sha256") != run["sha256"] or value.get("run_file_sha256") != file_hash(run_path):
        raise ValueError("evaluation overlay belongs to a different run")
    if value.get("trace_sha256") != run.get("trace_sha256"):
        raise ValueError("evaluation overlay trace differs")
    quality = {}
    for name, policy in run.get("policies", {}).items():
        rows = value.get("policies", {}).get(name, {}).get("requests", {})
        quality[name] = {}
        for request in policy.get("requests", []):
            identity = str(request.get("request_id", request.get("id")))
            item = dict(rows.get(identity, {"status": "unknown", "qualified": False}))
            artifact = request.get("artifact") or {}
            if item.get("qualified"):
                evidence = item.get("artifact_evidence", [])
                if not evidence:
                    item = {"status": "unknown", "qualified": False, "reason": "missing_evaluation_artifact_evidence"}
                for entry in evidence:
                    filename, digest = entry.get("path"), entry.get("sha256")
                    try:
                        resolve_artifact_path(filename, digest, run_path.parent)
                    except (LookupError, ValueError, OSError):
                        item = {"status": "unknown", "qualified": False, "reason": "evaluated_artifact_missing_or_changed"}
                        break
            quality[name][identity] = item
    return quality, {"status": "loaded", "path": str(path), "sha256": file_hash(path), "vbench": value.get("vbench")}


def _flatten(prefix, value, result):
    for key, child in value.items():
        name = prefix + key
        if isinstance(child, dict):
            _flatten(name + ".", child, result)
        else:
            result[name] = child
    return result


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _xlsx(path, sheets):
    """Small dependency-free OOXML workbook; strings never become formulas."""
    def column(number):
        result = ""
        while number:
            number, remainder = divmod(number - 1, 26)
            result = chr(65 + remainder) + result
        return result

    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relationships = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(str(path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        types = ['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                 '<Default Extension="xml" ContentType="application/xml"/>',
                 '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
        workbook, rels = [], []
        for index, (name, rows) in enumerate(sheets, 1):
            fields = list(dict.fromkeys(key for row in rows for key in row))
            matrix = [fields] + [[row.get(key) for key in fields] for row in rows]
            xmlrows = []
            for row_index, values in enumerate(matrix, 1):
                cells = []
                for col_index, value in enumerate(values, 1):
                    ref = column(col_index) + str(row_index)
                    if value is None:
                        continue
                    if _finite(value):
                        cells.append('<c r="%s"><v>%s</v></c>' % (ref, value))
                    else:
                        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                        text = "".join(char for char in text if ord(char) >= 32 or char in "\t\r\n")
                        cells.append('<c r="%s" t="inlineStr"><is><t xml:space="preserve">%s</t></is></c>' % (ref, html.escape(text)))
                xmlrows.append('<row r="%d">%s</row>' % (row_index, "".join(cells)))
            archive.writestr("xl/worksheets/sheet%d.xml" % index, '<worksheet xmlns="%s"><sheetData>%s</sheetData></worksheet>' % (ns, "".join(xmlrows)))
            types.append('<Override PartName="/xl/worksheets/sheet%d.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' % index)
            workbook.append('<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (html.escape(name), index, index))
            rels.append('<Relationship Id="rId%d" Type="%s/worksheet" Target="worksheets/sheet%d.xml"/>' % (index, relationships, index))
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">%s</Types>' % "".join(types))
        archive.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="%s/officeDocument" Target="xl/workbook.xml"/></Relationships>' % relationships)
        archive.writestr("xl/workbook.xml", '<workbook xmlns="%s" xmlns:r="%s"><sheets>%s</sheets></workbook>' % (ns, relationships, "".join(workbook)))
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">%s</Relationships>' % "".join(rels))


def _figure(path, summaries, metric, title):
    maximum = max([item.get(metric) or 0 for item in summaries.values()] + [1e-12])
    height = 90 + 62 * len(summaries)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="%d" viewBox="0 0 900 %d">' % (height, height),
             '<rect width="900" height="100%%" fill="white"/>',
             '<text x="24" y="30" font-family="sans-serif" font-size="20">%s</text>' % html.escape(title)]
    for index, (name, item) in enumerate(summaries.items()):
        value = item.get(metric)
        y = 65 + index * 62
        label = "%s (n=%d/%d)" % (name, item["completed"], item["arrivals"])
        parts.append('<text x="24" y="%d" font-family="sans-serif" font-size="13">%s</text>' % (y, html.escape(label)))
        width = (value or 0) / maximum * 520
        parts.append('<rect x="270" y="%d" width="%.2f" height="22" fill="#2563eb"/>' % (y - 16, width))
        parts.append('<text x="%.2f" y="%d" font-family="sans-serif" font-size="13">%s</text>' % (280 + width, y, "unknown" if value is None else "%.4g" % value))
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def _copy_evidence(run, run_path, output):
    evidence = output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    files, missing = [], []
    sources = [run_path, run_path.parent / "trace.json", run_path.parent / "evaluation.json"]
    for name in ("vbench_generation.json", "vbench_evaluation.json", "config.json", "profile.json"):
        if (run_path.parent / name).is_file():
            sources.append(run_path.parent / name)
    for source in sources:
        if not source.is_file():
            if source.name != "evaluation.json":
                missing.append(str(source))
            continue
        target = evidence / source.name
        if target.is_file() and file_hash(target) != file_hash(source):
            target = evidence / (source.stem + "-" + file_hash(source) + source.suffix)
        if source.resolve() != target.resolve():
            shutil.copy2(str(source), str(target))
        files.append({"source": str(source), "copy": str(target), "sha256": file_hash(target)})
    artifacts = list(run.get("references", {}).values())
    artifacts += [row.get("artifact") or {} for policy in run.get("policies", {}).values() for row in policy.get("requests", [])]
    seen = set()
    for artifact in artifacts:
        for kind in ("output", "latent", "cache"):
            filename = artifact.get(kind + "_path")
            if not filename:
                continue
            source = Path(filename)
            if not source.is_absolute():
                source = run_path.parent / source
            if str(source) in seen:
                continue
            seen.add(str(source))
            if not source.is_file():
                try:
                    source = resolve_artifact_path(filename, artifact.get(kind + "_sha256"), run_path.parent)
                except (LookupError, ValueError, OSError):
                    missing.append(str(source))
                    continue
            digest = file_hash(source)
            target = evidence / "artifacts" / (digest + source.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(str(source), str(target))
            if file_hash(target) != digest:
                raise ValueError("existing evidence artifact hash mismatch")
            files.append({"source": str(source), "copy": str(target), "sha256": digest,
                          "expected_sha256": artifact.get(kind + "_sha256"),
                          "matches_expected": digest == artifact.get(kind + "_sha256")})
    return {"files": files, "missing": missing}


def write_report(run_path, output_dir=None):
    """Rebuild JSON/CSV/XLSX/Chinese Markdown/SVG from immutable evidence."""
    path = _run_path(run_path)
    run = verify_seal(read_json(path))
    if run.get("schema_version") != "ditserve.benchmark.run.v1":
        raise ValueError("unsupported run schema")
    trace_path = path.parent / "trace.json"
    if trace_path.is_file():
        trace = verify_seal(read_json(trace_path))
        if trace["sha256"] != run.get("trace_sha256"):
            raise ValueError("trace and run hash differ")
    else:
        raise ValueError("trace.json is required to rebuild report")
    output = Path(output_dir).resolve() if output_dir else path.parent / "report"
    output.mkdir(parents=True, exist_ok=True)
    prior_summary = output / "summary.json"
    if prior_summary.is_file() and read_json(prior_summary).get("run_sha256") != run["sha256"]:
        raise ValueError("report output directory belongs to a different immutable run")
    if prior_summary.is_file():
        prior = verify_seal(read_json(prior_summary))
        history = output / "report_history" / prior["sha256"]
        history.mkdir(parents=True, exist_ok=True)
        for previous_file in output.iterdir():
            if previous_file.is_file() and not (history / previous_file.name).exists():
                shutil.copy2(str(previous_file), str(history / previous_file.name))
    qualities, evaluation = _load_overlay(run, path)
    summaries, by_shape, requests = {}, {}, []
    for name, policy in run.get("policies", {}).items():
        quality = qualities.get(name, {})
        rows = policy.get("requests", [])
        summaries[name] = summarize_requests(rows, policy.get("horizon_ms"), quality)
        reserved = policy.get("gpu_reserved_seconds", policy.get("device_reserved_seconds"))
        summaries[name]["gpu_reserved_seconds"] = reserved if _finite(reserved) and reserved >= 0 else None
        summaries[name]["gpu_reserved_scope"] = policy.get("gpu_reserved_scope", policy.get("device_reserved_scope", "unverified"))
        groups = defaultdict(list)
        for row in rows:
            groups[str(row.get("shape_id", "unknown"))].append(row)
            values = {key: row.get(key) for key in ("request_id", "id", "shape_id", "arrival_ms", "submit_ms", "start_ms", "finish_ms", "deadline_ms", "status")}
            values.update(policy=name, quality_status=quality.get(str(row.get("request_id", row.get("id"))), {}).get("status", "unknown"),
                          completed_latency_ms=_difference(row, "finish_ms", "arrival_ms") if row.get("status") == "completed" else None,
                          queue_ms=_difference(row, "start_ms", "submit_ms"), submit_delay_ms=_difference(row, "submit_ms", "arrival_ms"),
                          artifact=row.get("artifact"), events=row.get("events", []))
            requests.append(values)
        by_shape[name] = {shape: summarize_requests(group, policy.get("horizon_ms"), quality) for shape, group in groups.items()}
    evidence = _copy_evidence(run, path, output)
    purpose = run.get("purpose", trace.get("purpose"))
    source_serving_evidence = run.get("serving_performance_evidence", trace.get("serving_performance_evidence"))
    synthetic = _synthetic_run(run)
    serving_evidence = False if synthetic else source_serving_evidence
    comparison_run = dict(run, purpose=purpose, serving_performance_evidence=serving_evidence)
    report = seal({"schema_version": "ditserve.benchmark.report.v1", "run_sha256": run["sha256"],
                   "trace_sha256": run.get("trace_sha256"), "backend": run.get("backend"),
                   "execution_identity": run.get("execution_identity"),
                   "purpose": purpose, "serving_performance_evidence": serving_evidence,
                   "source_serving_performance_evidence": source_serving_evidence, "measured": run.get("measured"),
                   "measurement_kind": run.get("measurement_kind"), "status": run.get("status"),
                   "policies": summaries, "by_shape": by_shape, "paired_improvements": paired_improvements(comparison_run, summaries, trace.get("requests", [])),
                   "evaluation": evaluation, "evidence": evidence})
    atomic_json(output / "summary.json", report)
    summary_rows = [dict(policy=name, **_flatten("", value, {})) for name, value in summaries.items()]
    shape_rows = [dict(policy=name, shape_id=shape, **_flatten("", value, {})) for name, shapes in by_shape.items() for shape, value in shapes.items()]
    csv_requests = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in requests]
    _csv(output / "summary.csv", summary_rows)
    _csv(output / "requests.csv", csv_requests)
    _csv(output / "by_shape.csv", shape_rows)
    _xlsx(output / "results.xlsx", [("汇总", summary_rows), ("请求", requests), ("规格", shape_rows),
                                    ("配对比较", report["paired_improvements"]), ("证据", evidence["files"])])
    _figure(output / "completion.svg", summaries, "completion_rate", "完成率：分母为所有已到达请求")
    _figure(output / "deadline_goodput.svg", summaries, "deadline_goodput_per_s", "按时完成吞吐量（请求/秒）")
    lines = ["# 视频 DiT 服务基准报告", "", "后端：`%s`；测量类型：`%s`；运行状态：`%s`。" % (run.get("backend"), run.get("measurement_kind"), run.get("status")),
             "", "Trace SHA-256：`%s`。Run SHA-256：`%s`。" % (run.get("trace_sha256"), run["sha256"]),
             "", "所有完成率以已到达请求为分母；失败、超时和未完成请求保留，仅排除 not_arrived。平均值与 P50/P95/P99 仅统计已完成且时间戳有效的请求，样本数单独列出。排队时间为开始减提交，提交延迟为提交减计划到达。",
             "", "拒绝率包括 resource_rejected/rejected，超时率包括 timed_out/timeout，失败率包括 failed/arrived_aborted，分母均为所有已到达请求。GPU 保留秒数按运行记录汇总，包含 GPU 分配期间空闲时间，不等同于内核忙碌时间。",
             "", "| 策略 | 已到达 | 已完成 | 失败/未完成 | 延迟样本 | 平均延迟 ms | P95 ms | 按时 goodput/s | 质量合格 goodput/s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if purpose == "independent_quality_generation":
        lines[2:2] = ["**本次为独立质量评价的离线生成 trace，不是在线服务性能证据。下列时间仅用于诊断，不用于声明服务吞吐、延迟或策略性能提升；所有配对提升比较均已排除。**", ""]
    elif purpose == "offline_diagnostic" and serving_evidence is False:
        lines[2:2] = ["**本次 trace 仅用于离线诊断，不是在线服务性能证据。下列时间和吞吐量为诊断观测，不用于声明在线服务性能提升；所有配对提升比较均已排除。**", ""]
    elif synthetic:
        lines[2:2] = ["**本次结果来自测试或模拟后端，不是真实模型在线服务性能测量。质量通过仅表示测试输出的参考一致性；时间、吞吐量和策略差异仅用于逻辑诊断，全部性能提升配对已排除。**", ""]
    elif serving_evidence is False:
        lines[2:2] = ["**本次 trace 仅用于离线诊断，不是在线服务性能证据。下列时间和吞吐量为诊断观测，不用于声明在线服务性能提升；所有配对提升比较均已排除。**", ""]
    if synthetic and purpose in ("offline_diagnostic", "independent_quality_generation"):
        lines[2:2] = ["**后端为测试或模拟，输出及时间不能作为真实模型性能或生成质量测量。**", ""]
    def number(value):
        return "未知" if value is None else "%.4g" % value
    for name, value in summaries.items():
        latency = value["completed_latency_ms"]
        lines.append("| %s | %d | %d | %d | %d | %s | %s | %s | %s |" % (name.replace("|", "\\|"), value["arrivals"], value["completed"], value["unfinished_or_failed"], latency["samples"], number(latency["mean"]), number(latency["p95"]), number(value["deadline_goodput_per_s"]), number(value["quality_qualified_goodput_per_s"])))
    lines += ["", "质量合格 goodput 同时要求按时完成和质量验证通过；缺少参考、损坏或缺失输出、未安装评价模型均不能算通过。参考一致性只检查数值输出与缓存分支路径，不等于 VBench 生成质量分数。", "",
              "本报告不从少量维度平均出完整 VBench 总分。官方 VBench 需要独立生成协议及固定版本评价器；未运行状态保留为未验证。", "",
              "配对提升只计算同一 trace、相同生成配置、两侧全部请求完成的完整尝试。中断和不完整尝试仅展示观测值，不作为提升证据。", ""]
    for pair in report["paired_improvements"]:
        lines.append("- `%s → %s`：%s%s。" % (pair["baseline"], pair["candidate"], "纳入配对比较" if pair["status"] == "included" else "排除", "（%s）" % pair["reason"] if pair["reason"] else ""))
    lines += ["", "![完成率](completion.svg)", "", "![按时 goodput](deadline_goodput.svg)", "",
              "详细请求、各规格统计和证据路径见 `requests.csv`、`by_shape.csv`、`results.xlsx` 与 `summary.json`。原始 run、trace、评价记录和输出副本保存在 evidence，未删除历史证据。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output_dir": str(output), "report_path": str(output / "REPORT.md"), "summary": report,
            "files": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
