"""Rebuildable experiment archives: data.json.gz + results.xlsx + REPORT.md.

Each archived step stores every small text source file inside data.json.gz
with per-file SHA256. Binary GPU artifacts (raw arrays, media) are recorded
by hash in `external_artifacts` but never embedded. The workbook and report
are pure functions of that payload, so `verify` can rebuild them cell by cell
and `rebuild` restores them after loss. Synthetic results are not measured
GPU performance.
"""

import gzip
import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA = "ditserve.archive.v1"
NOTICE = ("本归档保留表格、报告和小型历史记录，不包含原始数组、latent 或完整 trace。"
          "合成结果不是真机性能，参考一致性不等于感知质量，未知质量不计入 SLO 成功。")
TABLE_NAMES = ("Config", "Requests", "Results", "Files")
PURPOSE = "比较固定请求流下各调度策略的参考行为。"
METHOD = "使用保存的 workload、成本模型和全部到达请求，比较完成、超时、SLO 与参考状态一致性。"
# GPU runs produce raw arrays/media; the archive records their hashes but does
# not embed binary payloads (originals are deleted after verification anyway).
EXCLUDED_SUFFIXES = {".npy", ".pt", ".mp4", ".png", ".jpg", ".jpeg"}


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_segment(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("series and step must be single safe directory names")
    return value


def _flatten(value, prefix="", lists=False):
    result = {}
    for key, child in value.items():
        name = prefix + key
        if isinstance(child, dict):
            result.update(_flatten(child, name + ".", lists))
        elif lists or not isinstance(child, list):
            result[name] = child
    return result


def _unit(name):
    if name == "goodput_per_sec":
        return "requests/s"
    if name.endswith(("_ms", ".wall_ms", "_wall_ms_sum")):
        return "ms"
    if name.endswith(("_seconds", "_sec")):
        return "s"
    if name.endswith(("_bytes", ".bytes")):
        return "bytes"
    if name.endswith("_mib"):
        return "MiB"
    if "_over_" in name:
        return "ratio"
    return ""


def _cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2 ** 53:
        return str(value)
    return value


def _columns(rows):
    return list(dict.fromkeys(key for row in rows for key in row))


def _tables(data):
    tables = {name: [] for name in TABLE_NAMES}
    summaries = [(name, file) for name, file in data["files"].items()
                 if Path(name).name == "summary.json" and isinstance(json.loads(file["text"]).get("runs"), dict)]
    selected_summary = min(summaries, key=lambda item: len(Path(item[0]).parts))[0] if summaries else None
    for name, file in data["files"].items():
        tables["Files"].append({"source": name, "bytes": file["bytes"], "sha256": file["sha256"]})
        if not name.endswith(".json"):
            continue
        value = json.loads(file["text"])
        if Path(name).name == "workload.json" and isinstance(value, list):
            tables["Requests"].extend(dict(source=name, **row) for row in value)
        if not isinstance(value, dict):
            continue
        config = value.get("config")
        if isinstance(config, dict):
            tables["Config"].extend({"source": name, "parameter": key, "value": item, "unit": _unit(key)}
                                    for key, item in _flatten(config, lists=True).items())
        for key in ("cost_model", "core"):
            if isinstance(value.get(key), dict):
                parameters = _flatten({key: value[key]}, lists=True)
                tables["Config"].extend({"source": name, "parameter": field, "value": item, "unit": _unit(field)}
                                        for field, item in parameters.items())
        if Path(name).name == "cost_model.json":
            tables["Config"].extend({"source": name, "parameter": "cost_model." + key, "value": item, "unit": _unit(key)}
                                    for key, item in _flatten(value, lists=True).items())
        if name == selected_summary:
            tables["Results"].extend(dict(source=name, case=case, **row) for case, row in value["runs"].items())
    return tables


def _markdown_table(rows, fields):
    if not rows or not fields:
        return []
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "") if row.get(field) is not None else "").replace("|", "\\|").replace("\n", " ") for field in fields) + " |")
    return lines


def _conclusion(data, tables):
    rows = tables["Results"]
    observed = [row["parity_failures"] for row in rows if row.get("parity_failures") is not None]
    failures = sum(observed) if observed else "未知"
    return ("保存 %d 条策略/配置结果，参考状态一致性失败计数为 %s。"
            "时间、资源和 goodput 均来自合成成本；它们不能作为真实 GPU 性能或感知质量证据。" % (len(rows), failures))


def _report(data, tables):
    rows = tables["Results"]
    statuses = {}
    for row in rows:
        status = row.get("status") or "synthetic"
        statuses[status] = statuses.get(status, 0) + 1
    lines = ["# " + data["title"], "", "## 目的与方法", "", PURPOSE, "", METHOD, "",
             "工作簿：[results.xlsx](results.xlsx)。可重建数据：[data.json.gz](data.json.gz)。", "",
             "Config 表保存逐实验配置；Results 保存逐策略汇总。"
             "字段后缀 `_ms`、`_seconds`、`_bytes`、`_mib` 分别表示毫秒、秒、字节、MiB；`goodput_per_sec` 为请求/秒。", "",
             "## 实际结果", "", _conclusion(data, tables), "",
             "保存 %d 份原始小文件、%d 条结果记录；另有 %d 份二进制产物仅登记哈希、不嵌入归档。" %
             (len(data["files"]), len(rows), len(data.get("external_artifacts", []))),
             "状态计数：`%s`。空白测项表示未知或不适用，不填零。" % json.dumps(statuses, ensure_ascii=False), ""]
    fields = [field for field in ("case", "completed", "rejected", "timed_out", "slo_attainment", "goodput_per_sec",
                                  "p99_latency_ms", "mean_batch_size", "preemptions", "offloads", "restores", "parity_failures")
              if any(field in row for row in rows)]
    lines.extend(_markdown_table(rows, fields))
    lines.extend(["", "## 结论边界与保留策略", "", NOTICE, "",
                  "原始 JSON 与报告文本原样存于压缩数据中；重建报告无需 GPU、模型或原始数组。", ""])
    return "\n".join(lines)


def rebuild(step):
    """Regenerate the workbook and report using only this step's data.json.gz."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    step = Path(step)
    with gzip.open(step / "data.json.gz", "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("schema") != SCHEMA:
        raise ValueError("unsupported archive schema")
    tables = _tables(data)
    workbook = Workbook()
    overview = workbook.active
    overview.title = "Overview"
    overview.append(["title", data["title"]])
    overview.append(["scope", NOTICE])
    overview.append(["raw_artifacts_available", False])
    overview.append(["historical_source_files", len(data["files"])])
    for key, value in data.get("counts", {}).items():
        overview.append([key, _cell(value)])
    for name, rows in tables.items():
        sheet = workbook.create_sheet(name)
        columns = _columns(rows) or ["no_records"]
        sheet.append(columns)
        for row in rows:
            sheet.append([_cell(row.get(key)) for key in columns])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for index, key in enumerate(columns, 1):
            sheet.column_dimensions[get_column_letter(index)].width = min(52, max(16, len(key) + 2))
    for sheet in workbook:
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="23485B")
        for row in sheet:
            for cell in row:
                if isinstance(cell.value, str):
                    # Historical logs are data, including strings beginning with '='.
                    cell.data_type = "s"
        sheet.sheet_view.showGridLines = False
    temporary = step / "results.tmp.xlsx"
    workbook.save(temporary)
    temporary.replace(step / "results.xlsx")
    (step / "REPORT.md").write_text(_report(data, tables), encoding="utf-8")
    return {"step": str(step), "workbook": str(step / "results.xlsx"), "report": str(step / "REPORT.md"),
            "data": str(step / "data.json.gz"), "raw_retained": True}


def _save_data(step, data):
    step.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = step / "data.json.gz.tmp"
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(encoded)
    temporary.replace(step / "data.json.gz")
    return rebuild(step)


def _collect_text(paths, source_root, key, title):
    data = {"schema": SCHEMA, "title": title, "step": key, "raw_source_root": str(source_root),
            "files": {}, "artifacts": [], "external_artifacts": [],
            "raw_artifacts_available": False, "notice": NOTICE}
    for path in paths:
        relative = str(path.relative_to(source_root))
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            raw = path.read_bytes()
            data["external_artifacts"].append({"source": relative, "bytes": len(raw),
                                               "sha256": hashlib.sha256(raw).hexdigest()})
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("unclassified binary evidence: " + str(path)) from error
        data["files"][relative] = {"text": text, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return data


def _check_links(path):
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("#"):
            continue
        if not (path.parent / target).is_file():
            raise ValueError("broken report link: " + target)


def _check_cells(sheet, expected):
    if sheet.max_row != len(expected) or sheet.max_column != max(map(len, expected)):
        raise ValueError("workbook dimensions differ: " + sheet.title)
    for row, values in zip(sheet, expected):
        for index, cell in enumerate(row):
            wanted = _cell(values[index]) if index < len(values) else None
            if isinstance(wanted, str):
                wanted = wanted[:32767] or None
            actual = cell.value
            if wanted is None:
                valid = actual is None
            elif isinstance(wanted, bool):
                valid = isinstance(actual, bool) and actual == wanted and cell.data_type == "b"
            elif isinstance(wanted, (int, float)):
                valid = (isinstance(actual, (int, float)) and not isinstance(actual, bool) and
                         cell.data_type == "n" and math.isclose(actual, wanted, rel_tol=1e-14, abs_tol=1e-12))
            else:
                valid = isinstance(actual, str) and actual == wanted and cell.data_type == "s"
            if not valid:
                raise ValueError("workbook cell differs: " + sheet.title + "!" + cell.coordinate)


def _verify_step(root, key):
    from openpyxl import load_workbook

    step = Path(root) / key
    with gzip.open(step / "data.json.gz", "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    for name, record in data["files"].items():
        encoded = record["text"].encode("utf-8")
        if len(encoded) != record["bytes"] or hashlib.sha256(encoded).hexdigest() != record["sha256"]:
            raise ValueError("archived evidence changed: " + name)
    tables = _tables(data)
    book = load_workbook(step / "results.xlsx", read_only=True, data_only=False)
    try:
        overview = [["title", data["title"]], ["scope", NOTICE], ["raw_artifacts_available", False],
                    ["historical_source_files", len(data["files"])]]
        overview.extend([name, _cell(value)] for name, value in data.get("counts", {}).items())
        _check_cells(book["Overview"], overview)
        for name, rows in tables.items():
            columns = _columns(rows) or ["no_records"]
            _check_cells(book[name], [columns] + [[row.get(column) for column in columns] for row in rows])
    finally:
        book.close()
    if (step / "REPORT.md").read_text(encoding="utf-8") != _report(data, tables):
        raise ValueError("report differs from archived data: " + key)
    _check_links(step / "REPORT.md")
    return data


def _write_index(root, manifest):
    lines = ["# 实验结果索引", "", NOTICE, ""]
    lines.extend("- [%s](%s/REPORT.md) · [Excel](%s/results.xlsx)" % (manifest.get("step_titles", {}).get(key, key), key, key)
                 for key in manifest["steps"])
    (root / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _check_links(root / "INDEX.md")


def verify(root):
    """Verify evidence bytes, every workbook cell and report links."""
    root = Path(root).resolve()
    manifest = json.loads((root / "archive.json").read_text(encoding="utf-8"))
    data = {key: _verify_step(root, key) for key in manifest["steps"]}
    _check_links(root / "INDEX.md")
    return {"status": "verified", "steps": len(data),
            "source_files": sum(len(step["files"]) for step in data.values()),
            "raw_artifacts_available": False}


def finalize_run(raw_dir, series, step, results_root="results", title=None):
    """Publish one run directory as an archive step, then verify it in place."""
    raw_dir, results_root = Path(raw_dir).resolve(), Path(results_root).resolve()
    if not raw_dir.is_dir():
        raise ValueError("raw run directory does not exist")
    if results_root == raw_dir or raw_dir in results_root.parents:
        raise ValueError("results cannot be nested in the raw run")
    key = _safe_segment(series) + "/" + _safe_segment(step)
    prior = results_root / key / "data.json.gz"
    if prior.exists():
        with gzip.open(prior, "rt", encoding="utf-8") as stream:
            old = json.load(stream)
        if old.get("raw_source_root") != str(raw_dir):
            raise ValueError("step already belongs to another or unknown raw source: " + key)
    paths = sorted(path for path in raw_dir.rglob("*") if path.is_file())
    data = _collect_text(paths, raw_dir, key, title or step)
    result = _save_data(results_root / key, data)
    manifest_path = results_root / "archive.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "schema": SCHEMA, "steps": [], "step_titles": {}, "notice": NOTICE}
    if key not in manifest["steps"]:
        manifest["steps"].append(key)
    manifest.setdefault("step_titles", {})[key] = title or step
    _json_write(manifest_path, manifest)
    _write_index(results_root, manifest)
    _verify_step(results_root, key)
    result["verified"] = True
    return result
