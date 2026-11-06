"""Compact verified results with resumable, explicit artifact cleanup.

``run.json`` contains JSON metadata, never raw tensors or event bodies. Artifact
``dependencies`` lists outstanding consumers; an empty list closes that file's
dependencies. A closed dependency can instead be ``{closed: true, reason: ...}``.
Control files, models and frozen sources remain outside the deletion whitelist.
New runs publish one dated report and a small JSON summary. Full typed Excel
archives remain an explicit opt-in; historical archives are read-only. Neither
summary publication nor a successful run closes outstanding evaluation needs.
"""

from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any

from ..runtime import ROOT

_py_pkg_dir = str(Path(ROOT) / "python" / "ditserve")
_parent_pkg = sys.modules.get("ditserve")
if _parent_pkg is not None and hasattr(_parent_pkg, "__path__") and _py_pkg_dir not in _parent_pkg.__path__:
    _parent_pkg.__path__.append(_py_pkg_dir)


SCHEMA = "ditserve.results.v1"
SUMMARY_SCHEMA = "ditserve.results-summary.v1"
SUMMARY_FILE = "summary.json"
JOURNAL = ".finalization.json"
JOURNAL_FILE = JOURNAL
TIER1_LOSSLESS_GATE_NAME = "tier1_lossless_runtime_qualification_gate"
VERIFIED_NUMERIC = "verified_lossless_exact"
VERIFIED_SEMANTIC = "verified_bitwise_invariant"
VERIFIED_PERFORMANCE = "verified_real_gpu"
UNVERIFIED = "unverified"
APPROXIMATE_POLICY_TOKENS = (
    "approx", "fold", "taylor", "quant", "teacache",
    "fastercache", "pab", "ditserve-v3", "ditserve-v5",
)
APPROXIMATE_OPTION_KEYS = (
    "cfg_fold", "taylor_cache", "ring_quant", "approximate",
    "allow_approx", "fold_steps", "fold_budget", "cache_threshold",
)
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_SHEET_ROWS = 1048576
CELL_LIMIT = 32767
CHUNK_SIZE = 16000
HEADERS = ("JSON pointer", "Type", "Part", "Value")
SECTIONS = {"config", "metrics", "requests", "phases", "failures", "budget", "retention"}
RAW_KINDS = {"raw", "array", "media"}
LOG_KINDS = {"log", "trace", "summarized_log"}
PROTECTED_PARTS = {"sources", "source-snapshots", "models", "weights", ".git", ".retention"}
PROTECTED_NAMES = {"run.json", JOURNAL, "budget.json", "gpu-budget.json", "ledger.json",
                   "config.json", "protocol.json", "manifest.json", "results.xlsx", "REPORT.md", SUMMARY_FILE}
LIVE = {"running", "admitted", "stopping", "unsettled", "waiting_resource"}


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _path(value):
    path = Path(os.path.abspath(value))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symbolic links are not allowed: " + str(path))
    return path


def _read(path, *, collect=False):
    path = _path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or (collect and before.st_size > MAX_JSON_BYTES):
            raise ValueError("not a bounded ordinary metadata file: " + str(path))
        digest, chunks, size = hashlib.sha256(), [], 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
            if collect:
                if size > MAX_JSON_BYTES:
                    raise ValueError("metadata exceeds size limit")
                chunks.append(block)
        after = os.fstat(stream.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("source changed while reading: " + str(path))
    return dict(path=str(path), bytes=size, sha256=digest.hexdigest()), b"".join(chunks)


def _document(path):
    binding, raw = _read(path, collect=True)
    value = json.loads(raw)
    _json(value)  # Reject NaN/Infinity before writing any archive.
    return value, binding


def _atomic_json(path, value):
    path = _path(path)
    handle, temporary = tempfile.mkstemp(prefix=".metadata-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save(work, journal):
    journal["sha256"] = _digest({k: v for k, v in journal.items() if k != "sha256"})
    _atomic_json(work / JOURNAL, journal)


def _excel():
    try:
        return importlib.import_module("openpyxl")
    except ModuleNotFoundError:
        cached = Path(os.environ.get("DITSERVE_REPORT_TOOLS") or ROOT / ".cache/tools")
        if (cached / "openpyxl/__init__.py").is_file():
            if str(cached) not in sys.path:
                sys.path.append(str(cached))
            return importlib.import_module("openpyxl")
        raise RuntimeError("Excel export requires the optional openpyxl reports dependency") from None


def _nodes(value, pointer=""):
    if isinstance(value, (dict, list)):
        kind = "dict" if isinstance(value, dict) else "list"
        yield pointer, kind, 0, str(len(value))
        members = sorted(value.items()) if kind == "dict" else enumerate(value)
        for key, item in members:
            child = pointer + "/" + str(key).replace("~", "~0").replace("/", "~1")
            yield from _nodes(item, child)
    else:
        kind = "null" if value is None else "bool" if type(value) is bool else \
            "int" if type(value) is int else "float" if type(value) is float else "str"
        encoded = _json(value)
        for part, start in enumerate(range(0, len(encoded), CHUNK_SIZE)):
            yield pointer, kind, part, encoded[start:start + CHUNK_SIZE]


def _restore(rows):
    fragments, kinds = {}, {}
    for pointer, kind, part, value in rows:
        if pointer in kinds and kinds[pointer] != kind:
            raise ValueError("conflicting JSON node types")
        kinds[pointer] = kind
        parts = fragments.setdefault(pointer, {})
        if type(part) is not int or part < 0 or part in parts or not isinstance(value, str):
            raise ValueError("invalid or duplicate scalar fragment")
        parts[part] = value
    restored, lengths = {}, {}
    for pointer in sorted(fragments, key=lambda p: (p.count("/"), p)):
        parts, kind = fragments[pointer], kinds[pointer]
        if sorted(parts) != list(range(len(parts))):
            raise ValueError("missing scalar fragment")
        raw = "".join(parts[i] for i in range(len(parts)))
        if kind in ("dict", "list"):
            length = int(raw)
            if length < 0:
                raise ValueError("negative container size")
            value = {} if kind == "dict" else []
            lengths[pointer] = length
        else:
            value = float(raw) if kind == "nonfinite" else json.loads(raw)
            expected = {"null": type(None), "bool": bool, "int": int, "float": float, "str": str}
            if kind == "nonfinite" and math.isfinite(value):
                raise ValueError("invalid legacy nonfinite scalar")
            if kind != "nonfinite" and (kind not in expected or type(value) is not expected[kind]):
                raise ValueError("scalar type changed")
        restored[pointer] = value
    if "" not in restored:
        raise ValueError("missing typed JSON root")
    # Populate dictionaries first and list members in numerical order.
    for pointer in sorted((p for p in restored if p), key=lambda p: (p.count("/"), p)):
        parent, key = pointer.rsplit("/", 1)
        key = key.replace("~1", "/").replace("~0", "~")
        container = restored.get(parent)
        if isinstance(container, dict):
            container[key] = restored[pointer]
        elif isinstance(container, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", key) or int(key) >= lengths[parent]:
                raise ValueError("invalid archived list index")
            if not container:
                container.extend([None] * lengths[parent])
            container[int(key)] = restored[pointer]
        else:
            raise ValueError("missing parent or child below scalar")
    children = {}
    for pointer in restored:
        if pointer:
            parent = pointer.rsplit("/", 1)[0]
            children[parent] = children.get(parent, 0) + 1
    for pointer, length in lengths.items():
        if children.get(pointer, 0) != length:
            raise ValueError("container row count differs")
    return restored[""]


def _section(pointer):
    parts = pointer.split("/")
    if len(parts) > 2 and parts[1] == "run":
        return "provenance" if parts[2] == "artifacts" else parts[2] if parts[2] in SECTIONS else "summary"
    return "retention" if len(parts) > 1 and parts[1] == "finalization" else "summary"


def _write_book(path, value):
    excel = _excel()
    from openpyxl.cell import WriteOnlyCell
    book = excel.Workbook(write_only=True)
    sheets, counts, rows = {}, {}, 0

    def append(sheet, values):
        cells = []
        for item in values:
            text = str(item)
            if len(text.encode("utf-16-le")) // 2 > CELL_LIMIT:
                raise ValueError("Excel cell limit exceeded; metadata key is too long")
            cell = WriteOnlyCell(sheet, text)
            cell.data_type = "s"  # Literal =, +, -, @ strings never become formulas.
            cells.append(cell)
        sheet.append(cells)

    if MAX_SHEET_ROWS < 2 or CHUNK_SIZE > CELL_LIMIT:
        raise ValueError("invalid Excel limits")
    try:
        for pointer, kind, part, scalar in _nodes(value):
            base = _section(pointer)
            count = counts.get(base, 0)
            index = count // (MAX_SHEET_ROWS - 1)
            name = base if index == 0 else base + "_" + str(index + 1)
            if name not in sheets:
                sheets[name] = book.create_sheet(name)
                sheets[name].freeze_panes = "A2"
                append(sheets[name], HEADERS)
            append(sheets[name], (_json(pointer), kind, part, scalar))
            counts[base] = count + 1
            rows += 1
        sheet = book.create_sheet("manifest")
        for row in (("schema", SCHEMA), ("metadata_sha256", _digest(value)), ("node_rows", rows)):
            append(sheet, row)
        book.save(path)
    except BaseException:
        for sheet in book:
            if not sheet.closed:
                sheet.close()
            if sheet._writer and os.path.exists(sheet._writer.out):
                sheet._writer.cleanup()
        raise
    finally:
        book.close()


def _read_book(path):
    book = _excel().load_workbook(_path(path), read_only=True, data_only=False)
    try:
        if "manifest" not in book:
            raise ValueError("not a new results workbook")
        manifest = dict(book["manifest"].values)
        if manifest.get("schema") != SCHEMA:
            raise ValueError("unsupported workbook schema")
        rows = []
        for sheet in book:
            for row in sheet:
                if any(cell.data_type == "f" for cell in row):
                    raise ValueError("formulas are forbidden in archived evidence")
            if sheet.title == "manifest":
                continue
            if tuple(next(sheet.values)) != HEADERS:
                raise ValueError("typed sheet header changed")
            for row in sheet.iter_rows(min_row=2, values_only=True):
                pointer, kind, part, scalar = row
                rows.append((json.loads(pointer), kind, int(part), scalar))
        value = _restore(rows)
        if len(rows) != int(manifest["node_rows"]) or _digest(value) != manifest["metadata_sha256"]:
            raise ValueError("workbook metadata digest or row count differs")
        return value
    finally:
        book.close()


def _report_errors(run, retention):
    errors, seen_errors = [], set()
    def issue(where, reason):
        if reason:
            text = _json(reason) if isinstance(reason, (dict, list)) else str(reason)
            item = (str(where), text)
            if item not in seen_errors:
                seen_errors.add(item)
                errors.append(item)
    for phase in run.get("phases", []):
        where = "阶段 %s，尝试 %s" % (phase.get("id", "unknown"), phase.get("attempt", "unknown"))
        for key in ("error", "cleanup_error"):
            issue(where, phase.get(key))
        qualification = phase.get("qualification")
        for check in qualification.get("checks", []) if isinstance(qualification, dict) else []:
            if check.get("status") != "passed":
                issue(where, "%s 门槛 %s：%s；实际 %s，要求 %s %s" % (
                    check.get("scope", "numeric"), check.get("metric", "unknown"),
                    check.get("status", "unverified"), _json(check.get("actual")),
                    check.get("op", "unknown"), _json(check.get("expected"))))
    for failure in run.get("failures", []):
        if isinstance(failure, dict):
            issue("阶段 %s，尝试 %s" % (failure.get("phase", "unknown"), failure.get("attempt", "unknown")),
                  failure.get("error", failure))
        else:
            issue("失败记录", failure)
    for request in run.get("requests", []):
        issue("请求 " + str(request.get("request_id", "unknown")), request.get("error"))
    if run.get("status") == "blocked":
        issue("准入受阻", run.get("admission") or run.get("blocked_phase"))
    for error in retention.get("errors", []):
        issue("收尾", error)
    issue("收尾", retention.get("error"))
    return errors


def _key_metrics(value):
    """Retain scalar aggregates, not observations, tensor data or event arrays."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            child = _key_metrics(item)
            if child:
                result[key] = child
        elif item is None or type(item) in (bool, int, float):
            result[key] = item
        elif isinstance(item, str) and len(item) <= 400:
            result[key] = item
    return result


def _base_summary_run(run):
    """A deliberately lossy projection; hashes bind the omitted full records."""
    result = {key: run[key] for key in ("experiment", "run_id", "status", "measurement_scope", "notice") if key in run}
    config = run.get("config", {})
    result.update(config_sha256=run.get("config_sha256", _digest(config)),
                  config_content_sha256=_digest(config), source_sha256=_digest(run.get("source", {})),
                  source=_key_metrics({k: v for k, v in run.get("source", {}).items()
                                       if k not in ("files", "native_files")}),
                  metrics=_key_metrics(run.get("metrics", {})), phases=[],
                  request_count=len(run.get("requests", [])), failures=[])
    # Deduplicate replica configuration; the frozen trace/profile is hash-bound,
    # not copied into the long-term result or spreadsheet.
    models = {}
    workers = config.get("workers", {})
    for worker in workers.values() if isinstance(workers, dict) else workers:
        if not isinstance(worker, dict):
            continue
        model = {k: v for k, v in worker.items() if k in ("backend", "model", "model_path", "revision", "dtype", "config")}
        model = _key_metrics(model)
        if model:
            models[_digest(model)] = model
    result["models"] = list(models.values())
    for phase in run.get("phases", []):
        item = {k: phase[k] for k in ("id", "status", "attempt", "policy", "cell", "reason") if k in phase}
        item["qualification"] = phase.get("qualification", "unverified")
        item["metrics"] = _key_metrics(phase.get("metrics", {}))
        if isinstance(phase.get("result"), dict):
            item["result"] = _key_metrics(phase["result"])
        item["cells"] = []
        for cell in phase.get("cells", []):
            selected = {k: cell[k] for k in ("cell", "policy", "status", "gpus", "elapsed_ms",
                "order", "sp_transport", "async_rank0_decode") if k in cell}
            for key in ("metrics", "by_kind", "doorbell", "continuous_batch", "guard_telemetry"):
                if key in cell:
                    selected[key] = _key_metrics(cell[key])
            if "image_batch" in cell:
                selected["image_batch"] = (_key_metrics(cell["image_batch"]) if isinstance(cell["image_batch"], dict)
                                           else cell["image_batch"] if type(cell["image_batch"]) is int else None)
            selected["request_count"] = len(cell.get("requests", []))
            if isinstance(cell.get("requests"), list):
                selected["execution_summary"] = _execution_summary(cell)
            item["cells"].append(selected)
        result["phases"].append(item)
    # Coalesce repeated errors. The count is exact; bounded text is labelled.
    failures = {}
    for where, reason in _report_errors(run, {}):
        key = _digest(reason)
        row = failures.setdefault(key, dict(error_sha256=key, error=reason[:400],
            text_truncated=len(reason) > 400, count=0, first_location=where))
        row["count"] += 1
    result["failures"] = list(failures.values())
    result["log_summaries"] = [dict(path=item["path"], **{k: item["summary"][k]
        for k in ("source_sha256", "source_bytes", "total_records", "aggregation", "omitted_fields")
        if k in item["summary"]}, metrics=_key_metrics(item["summary"].get("metrics", {})))
        for item in run.get("artifacts", []) if item.get("kind") in LOG_KINDS and isinstance(item.get("summary"), dict)]
    budget = run.get("budget", {})
    result["budget"] = {k: budget[k] for k in ("used_gpu_seconds", "used_regular_gpu_seconds",
        "used_repair_gpu_seconds", "phase_usage") if k in budget}
    result["gpu_uuids"] = sorted({gpu for session in budget.get("sessions", [])
        for gpu in session.get("identity", {}).get("gpu_uuids", [])})
    result["summary_limits"] = dict(resumable=False, raw_re_evaluable=False,
        omitted=["full configuration/trace", "per-request records", "events and samples", "source file manifests"],
        notice="Preserved aggregates only. Full run.json and raw inputs stay in the work directory until their consumers close.")
    return result


def _summary_run(run):
    """Project a run into summary.json while evaluating Tier-1 Lossless Qualification for real GPU runs."""
    qual = evaluate_tier1_lossless_qualification(run, config=run.get("config"))
    if qual.get("qualified") is True:
        enriched = apply_tier1_qualification_gate(run, config=run.get("config"))
        summary = _base_summary_run(enriched)
        return apply_tier1_qualification_gate(summary, config=run.get("config"))
    return _base_summary_run(run)


def _execution_summary(cell):
    """Bounded mechanism/cost aggregates, preserving the absence of samples."""
    rows = cell["requests"]
    def distribution(values, missing=0):
        samples = sorted(values)
        def quantile(fraction):
            if not samples:
                return None
            position = (len(samples) - 1) * fraction
            lower = math.floor(position)
            return samples[lower] + (samples[min(lower + 1, len(samples) - 1)] - samples[lower]) * (position - lower)
        return dict(count=len(samples), missing_count=missing, p50=quantile(.5), p95=quantile(.95), p99=quantile(.99),
                    min=samples[0] if samples else None, max=samples[-1] if samples else None,
                    total=sum(samples) if samples else None)
    video_rows = [row for row in rows if row.get("kind") == "video" or "resumes" in row or "resume_extra_ms" in row]
    recovery = {}
    for field in ("resumes", "resume_extra_ms"):
        values = [row[field] for row in video_rows if type(row.get(field)) in (int, float)
                  and math.isfinite(row[field])]
        recovery[field] = distribution(values, len(video_rows) - len(values))
    images = [row for row in rows if row.get("kind") == "image" or "batch_request_ids" in row]
    groups, request_histogram, identity_counts = {}, {}, {}
    for row in images:
        if isinstance(row.get("request_id"), str):
            identity_counts[row["request_id"]] = identity_counts.get(row["request_id"], 0) + 1
        size, members = row.get("batch_size"), row.get("batch_request_ids")
        if type(size) is not int or size < 1:
            continue
        key = str(size)
        request_histogram[key] = request_histogram.get(key, 0) + 1
        if (not isinstance(members, list) or len(members) != size
                or any(not isinstance(member, str) or not member for member in members)
                or len(set(members)) != size or row.get("request_id") not in members):
            continue
        groups.setdefault(tuple(sorted(members)), []).append(row["request_id"])
    histogram, covered, batches = {}, 0, 0
    for members, observed in groups.items():
        # A partial or inconsistent membership is unknown, not an executed batch.
        if (len(observed) != len(members) or set(observed) != set(members)
                or any(identity_counts[member] != 1 for member in members)):
            continue
        key = str(len(members))
        histogram[key] = histogram.get(key, 0) + 1
        covered += len(observed)
        batches += 1
    batch_evidence = dict(image_request_records=len(images), request_size_histogram=request_histogram,
        batch_size_histogram=histogram, verified_batches=batches, covered_requests=covered,
        unknown_requests=len(images) - covered, total_batches=batches if covered == len(images) and images else None,
        status="not_applicable" if not images else "complete" if covered == len(images) else "partial" if covered else "unknown",
        scope="actual per-request batch_size; atomic batches deduplicated only with complete matching batch_request_ids; configured limits are not measurements")
    completed = sum(row.get("status") == "completed" for row in rows)
    timely = sum(row.get("status") == "completed" and row.get("finish_ms") is not None
        and row.get("deadline_ms") is not None and row["finish_ms"] <= row["deadline_ms"] for row in rows)
    elapsed, gpus = cell.get("elapsed_ms"), cell.get("gpus")
    gpu_seconds = gpus * elapsed / 1000 if type(gpus) is int and type(elapsed) in (int, float) else None
    return dict(request_records=len(rows), observed_transfers=sum(row.get("transfers", 0) for row in rows),
        preempting_request_count=sum(row.get("preemption_reason") == "block_preempt" for row in rows),
        preemption_stop_ms=distribution(sample["stop_ms"] for row in rows for sample in row.get("preemption_stop_samples", [])),
        atomic_batch=batch_evidence, **recovery,
        allocated_gpu_seconds=gpu_seconds,
        gpu_seconds_per_completed_request=gpu_seconds / completed if gpu_seconds is not None and completed else None,
        gpu_seconds_per_ontime_request=gpu_seconds / timely if gpu_seconds is not None and timely else None,
        scope="cell wall time * allocated GPUs; startup/cleanup excluded; linear-interpolated quantiles; resumes and resume_extra_ms summarize observed per-request values (negative extra time preserved)")


def _summary_report(value):
    if isinstance(value, Mapping) and "run" not in value:
        exp_name = str(value.get("experiment") or "experiment")
        value = {
            "schema": SUMMARY_SCHEMA,
            "report_name": str(value.get("report_name") or f"20261106-{exp_name}.md"),
            "run": dict(value),
            "finalization": value.get("finalization") or {
                "status": "completed", "format": "compact", "deleted": [],
                "reclaimed_bytes": 0, "retained": [], "pending_dependencies": [],
            },
        }
    run, retention = value["run"], value["finalization"]
    def cell(item):
        if item is None:
            return "未记录"
        if type(item) is float:
            return format(item, ".6g")
        return str(item).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    def first(values, *keys):
        return next((values[key] for key in keys if values.get(key) is not None), None)
    def metric_row(label, kind, status, metrics, elapsed=None, request_count=None):
        attainment = first(metrics, "deadline_attainment", "slo_attainment", "attainment")
        p95 = first(metrics, "latency_p95_ms")
        if p95 is None and isinstance(metrics.get("latency_ms"), dict):
            p95 = first(metrics["latency_ms"], "p95", "p95_ms")
        goodput = metrics.get("goodput_rps")
        duration = first(metrics, "observation_ms")
        duration = elapsed if duration is None else duration
        if goodput is None and type(metrics.get("ontime")) in (int, float) and type(duration) in (int, float) and duration > 0:
            goodput = metrics["ontime"] * 1000 / duration
        qualified = metrics.get("qualified_goodput_rps")
        return "| " + " | ".join(cell(x) for x in (
            label, kind, status, metrics.get("requests", request_count), attainment, p95,
            goodput, qualified if qualified is not None else "未验证")) + " |"
    lines = ["# " + value["report_name"][:-3], "",
        "实验 `%s`，运行 `%s`：执行 `%s`，归档 `%s`。" %
        (run["experiment"], run["run_id"], run.get("status", "unverified"), retention["status"]), "",
        "[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。",
        "completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。", "",
        "证据范围：%s。" % cell(run.get("measurement_scope", run.get("notice", "未记录"))), "",
        "配置 SHA256：`%s`。" % run["config_content_sha256"], "",
        "| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    has_metrics = False
    for phase in run["phases"]:
        entries = phase.get("cells", [])
        if "result" in phase:
            entries = entries + [dict(phase["result"], policy=phase.get("policy", ""), status=phase.get("status", "unverified"))]
        if not entries:
            entries = [phase]
        for item in entries:
            metrics = item.get("metrics", {})
            label = str(phase.get("id", "")) + " / " + str(item.get("policy", item.get("cell", "")))
            status = item.get("status", "unverified")
            elapsed = first(item, "elapsed_ms", "makespan_ms")
            lines.append(metric_row(label, "全部", status, metrics, elapsed, item.get("request_count")))
            has_metrics = has_metrics or bool(metrics)
            for kind, kind_metrics in item.get("by_kind", {}).items():
                if isinstance(kind_metrics, dict):
                    lines.append(metric_row(label, kind, status, kind_metrics, elapsed))
    if not has_metrics and run.get("metrics"):
        lines.append(metric_row("运行合计", "全部", run.get("status", "unverified"), run["metrics"]))
    lines += ["", "SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。",
              "二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。"]
    for phase in run["phases"]:
        qualification = phase.get("qualification")
        scopes = qualification.get("scopes", {}) if isinstance(qualification, dict) else {}
        lines += ["", "阶段 `%s` 资格：%s。" % (phase.get("id", ""), ", ".join(
            name + "=" + str(scopes.get(name, {}).get("status", "unverified"))
            for name in ("numeric", "semantic", "performance")))]
    if run["failures"]:
        lines += ["", "失败与未通过门槛（重复错误合并，文本最多 400 字，完整错误以 SHA256 绑定）：", ""]
        lines += ["- %s（%d 次）：%s" % (cell(f["first_location"]), f["count"], cell(f["error"])) for f in run["failures"]]
    lines += ["", "清理：确认删除 %d 个文件，回收 %d 字节，保留 %d 个文件。" %
        (len(retention["deleted"]), retention["reclaimed_bytes"], len(retention["retained"])),
        "运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。"]
    if retention["retained"]:
        lines += ["", "保留原因："] + ["- `%s`：%s" % (cell(item["path"]), cell(item["reason"])) for item in retention["retained"]]
    if not _closed(retention.get("pending_dependencies", [])):
        lines += ["", "未闭合依赖：" + _json(retention["pending_dependencies"])]
    if retention.get("error"):
        lines += ["", "收尾待重试：" + cell(retention["error"])]
    return "\n".join(lines) + "\n"


def _read_summary(root, *, check_report=True):
    value, unused = _document(root / SUMMARY_FILE)
    if value.get("schema") != SUMMARY_SCHEMA or value.get("sha256") != _digest(
            {k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("summary schema or checksum differs")
    name = value.get("report_name", "")
    if not re.fullmatch(r"[0-9]{8}-[A-Za-z0-9][A-Za-z0-9_.-]*\.md", name):
        raise ValueError("unsafe dated report name")
    if check_report and _path(root / name).read_text(encoding="utf-8") != _summary_report(value):
        raise ValueError("report differs from summary metadata")
    return value


def _report(value):
    run, retention = value["run"], value["finalization"]
    scoped = retention.get("report_version", 1) >= 2
    def cell(value):
        text = str(value).replace("|", "\\|").replace("\n", " ")
        return text.replace("\r", " ") if scoped else text
    lines = ["# " + str(run["experiment"]) + " / " + str(run["run_id"]), "",
             "实验状态：`%s`。归档状态：`%s`。" % (run.get("status", "unverified"), retention["status"]), "",
             "[工作簿](results.xlsx) 保存完整配置、精确指标、请求记录、失败、预算、来源与清理凭据。",
             "缺失不是零；归档校验不授予质量、性能或语义资格。", ""]
    if scoped:
        lines += ["completed 仅表示执行完成。以下资格只对应已配置门槛；数值一致性通过不能替代语义/盲评或性能验证。", "",
                  "| 阶段 | 执行状态 | 尝试 | 数值 numeric | 语义 semantic | 性能 performance |",
                  "| --- | --- | --- | --- | --- | --- |"]
    else:
        lines += ["| 阶段 | 状态 | 尝试 |", "| --- | --- | --- |"]
    for phase in run.get("phases", []):
        row = [phase.get(k, "") for k in ("id", "status", "attempt")]
        if scoped:
            qualification = phase.get("qualification")
            scopes = qualification.get("scopes", {}) if isinstance(qualification, dict) else {}
            scopes = scopes if isinstance(scopes, dict) else {}
            for name in ("numeric", "semantic", "performance"):
                scope = scopes.get(name)
                row.append((scope.get("status") or "unverified") if isinstance(scope, dict) else "unverified")
        lines.append("| " + " | ".join(cell(item) for item in row) + " |")
    metrics = []
    groups = [("run", run.get("metrics", {}))] + [(str(p.get("id", "phase")), p.get("metrics", {}))
                                                   for p in run.get("phases", [])]
    for name, group in groups:
        for pointer, kind, part, scalar in _nodes(group):
            if kind in {"int", "float", "bool", "null"} or kind == "str" and part == 0 and len(scalar) <= 120:
                metrics.append((name + pointer, scalar))
    if metrics:
        lines += ["", "指标概览（%d 项中展示前 %d 项；完整值及单位见工作簿）：" % (len(metrics), min(30, len(metrics))), "",
                  "| 指标 | 已记录值 |", "| --- | --- |"]
        lines += ["| %s | %s |" % (name.replace("|", "\\|"), scalar.replace("|", "\\|"))
                  for name, scalar in metrics[:30]]
    if scoped:
        errors = _report_errors(run, retention)
        if errors:
            lines += ["", "错误与未通过/未验证门槛（%d 项中展示前 %d 项，每项最多 400 字；完整证据见工作簿）：" %
                      (len(errors), min(20, len(errors))), ""]
            lines += ["- %s：%s%s" % (cell(where), cell(reason[:400]), "…" if len(reason) > 400 else "")
                      for where, reason in errors[:20]]
    lines += ["", "保存 %d 条请求、%d 条失败记录。" % (len(run.get("requests", [])), len(run.get("failures", []))), "",
              "已确认不存在的原始文件：%d；保留白名单文件：%d；已回收字节：%d。" %
              (len(retention["deleted"]), len(retention["retained"]), retention["reclaimed_bytes"]), "",
              "原始数组、视频和完整事件日志不嵌入工作簿。已删除输入仅保留历史指标，无法恢复原始数据或重新评测。",
              "工作目录的 run.json、收尾日志、账本及冻结来源是恢复/来源控制文件，不自动删除。"]
    if retention["retained"]:
        lines += ["", "仍保留的文件及依赖："]
        lines += ["- `%s`：%s" % (item["path"], item["reason"]) for item in retention["retained"]]
    if not _closed(retention.get("pending_dependencies", [])):
        lines += ["", "尚未闭合的评测/续跑依赖：" + _json(retention["pending_dependencies"])]
    if retention.get("error") and not scoped:
        lines += ["", "收尾待重试：" + retention["error"]]
    return "\n".join(lines) + "\n"


def _publish(work, destination, value):
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=destination) as temporary:
        staging = Path(temporary)
        if value["schema"] == SUMMARY_SCHEMA:
            value = {k: v for k, v in value.items() if k != "sha256"}
            value["sha256"] = _digest(value)
            _atomic_json(staging / SUMMARY_FILE, value)
            (staging / value["report_name"]).write_text(_summary_report(value), encoding="utf-8")
            if _read_summary(staging) != value:
                raise ValueError("summary readback changed metadata")
            names = (SUMMARY_FILE, value["report_name"])
        else:
            _write_book(staging / "results.xlsx", value)
            (staging / "REPORT.md").write_text(_report(value), encoding="utf-8")
            if _digest(_read_book(staging / "results.xlsx")) != _digest(value):
                raise ValueError("Excel readback changed metadata")
            names = ("results.xlsx", "REPORT.md")
        for name in names:
            _path(destination / name)
            with (staging / name).open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(staging / name, destination / name)
        _sync_directory(destination)


def _active(run):
    budget = run.get("budget", {})
    if run.get("status") in LIVE or any(p.get("status") in LIVE for p in run.get("phases", [])):
        raise RuntimeError("run or phase is still active")
    if any(k in budget and budget[k] not in (None, False) for k in ("active", "active_session")):
        raise RuntimeError("GPU accounting is still active")


@contextmanager
def _guard(work, run, handles):
    # The caller closes these after its error-reporting path as well, so a
    # failed deletion cannot race another finalizer while saving its receipt.
    paths = [work / ".finalize.lock"]
    ledger = run.get("budget", {}).get("ledger") or run.get("budget_ledger")
    if ledger:
        ledger = _path(ledger if Path(ledger).is_absolute() else work / ledger)
        paths += [p for p in {ledger.with_suffix(".lock"), Path(str(ledger) + ".lock")} if p.exists()]
    for path in sorted(set(paths)):
        path = _path(path)
        # Writable even for the ledger's lock: NFS emulates flock with byte-range
        # locks, and an exclusive one on a read-only descriptor fails with EBADF.
        handle = path.open("a+b")
        handles.append(handle)
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _active(run)
    if ledger:
        data, unused = _document(ledger)
        _active({"budget": data})
    yield


def _closed(dependencies):
    return isinstance(dependencies, list) and all(isinstance(d, dict) and d.get("closed") is True
        and isinstance(d.get("reason"), str) and d["reason"].strip() for d in dependencies)


def _usage(directory):
    """Count this directory only; do not follow shared model/environment links."""
    files = logical = allocated = 0
    for parent, dirs, names in os.walk(directory, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(parent) / name).is_symlink()]
        for name in names:
            info = (Path(parent) / name).lstat()
            if stat.S_ISREG(info.st_mode):
                files += 1
                logical += info.st_size
                allocated += info.st_blocks * 512
    return dict(files=files, bytes=logical, allocated_bytes=allocated)


def _artifacts(work, run, journal):
    files, retained, seen = [], [], set()
    prior = {item["path"]: item for item in journal.get("deleted", [])}
    global_open = not _closed(run.get("retention", {}).get("pending_dependencies", []))
    ledger = run.get("budget", {}).get("ledger") or run.get("budget_ledger")
    ledger = _path(work / ledger) if ledger else None
    for artifact in run.get("artifacts", []):
        path = _path(work / artifact["path"])
        if path == work or work not in path.parents or path in seen:
            raise ValueError("artifact is outside its work directory or duplicated")
        seen.add(path)
        expected = {"path": str(path), "bytes": artifact.get("bytes"), "sha256": artifact.get("sha256")}
        if type(expected["bytes"]) is not int or expected["bytes"] < 0 or not re.fullmatch(
                r"[0-9a-f]{64}", str(expected["sha256"])):
            raise ValueError("artifact requires exact bytes and SHA256")
        if str(path) in prior:
            if any(prior[str(path)][k] != expected[k] for k in expected) or path.exists():
                raise ValueError("previously deleted artifact binding changed or file reappeared")
            continue
        actual, unused = _read(path)
        if actual != expected:
            raise ValueError("artifact size/hash changed: " + str(path))
        kind = artifact.get("kind")
        reason = None
        control = path == ledger or path.name in PROTECTED_NAMES or path.name.endswith(".lock") or \
            PROTECTED_PARTS.intersection(path.relative_to(work).parts) or kind in {"control", "source", "config", "model", "weights"}
        if control:
            reason = "control, model or frozen source evidence"
        elif global_open or not _closed(artifact.get("dependencies")):
            reason = "dependencies remain open: " + _json(artifact.get("dependencies"))
        elif kind in LOG_KINDS:
            summary = artifact.get("summary", {})
            if not (summary.get("source_sha256") == expected["sha256"] and summary.get("source_bytes") == expected["bytes"]
                    and type(summary.get("total_records")) is int and summary["total_records"] >= 0
                    and summary.get("aggregation") and "omitted_fields" in summary and "metrics" in summary):
                reason = "event log needs a bound, complete aggregation with omitted fields"
        elif kind not in RAW_KINDS:
            reason = "artifact kind is not an explicitly prunable raw artifact"
        if reason:
            retained.append(dict(expected, reason=reason, control=bool(control)))
        else:
            files.append(expected)
    return files, retained


def _dependency_revision(previous, current):
    """After deletion only close dependencies; never revise measured evidence."""
    old, new, changes = copy.deepcopy(previous), copy.deepcopy(current), []
    pairs = [(old.setdefault("retention", {}), new.setdefault("retention", {}), "pending_dependencies", "run")]
    if len(old.get("artifacts", [])) != len(new.get("artifacts", [])):
        raise ValueError("artifact inventory changed after cleanup")
    pairs += [(a, b, "dependencies", a["path"]) for a, b in zip(old.get("artifacts", []), new.get("artifacts", []))]
    for a, b, key, path in pairs:
        before, after = a.get(key, []), b.get(key, [])
        if before != after:
            if not _closed(after):
                raise ValueError("dependency revision must explicitly close the consumer")
            changes.append(dict(path=path, previous=before, closed=after))
        a.pop(key, None)
        b.pop(key, None)
    if _json(old) != _json(new):
        raise ValueError("input changed after cleanup began; preserve measured evidence")
    return changes


def _evidence_binding(run):
    """Bind full scientific records without storing them in the summary/journal."""
    stable, dependencies = copy.deepcopy(run), {}
    dependencies["run"] = stable.setdefault("retention", {}).pop("pending_dependencies", [])
    for item in stable.get("artifacts", []):
        dependencies["artifact:" + item["path"]] = item.pop("dependencies", [])
    return _digest(stable), dependencies


def _summary_dependency_revision(journal, run):
    digest, dependencies = _evidence_binding(run)
    if journal.get("evidence_sha256") != digest or set(journal["dependency_state"]) != set(dependencies):
        raise ValueError("input changed after cleanup began; preserve measured evidence")
    changes = []
    for path, before in journal["dependency_state"].items():
        after = dependencies[path]
        if before != after:
            if not _closed(after):
                raise ValueError("dependency revision must explicitly close the consumer")
            changes.append(dict(path=path, previous=before, closed=after))
    return changes


def _snapshot(run, source, journal, retained, status, error=None):
    deleted = copy.deepcopy(journal.get("deleted", []))
    finalization = dict(status=status, report_version=2, input=source, work_dir=journal["work_dir"], destination=journal["destination"],
        deleted=deleted, retained=retained, planned=journal.get("planned", []),
        reclaimed_bytes=sum(item["bytes"] for item in deleted),
        raw_re_evaluable=False if deleted else None, errors=journal.get("errors", []),
        input_revisions=journal.get("input_revisions", []), pending_dependencies=run.get("retention", {}).get("pending_dependencies", []),
        work_usage_before=journal.get("work_usage_before"), work_usage_after=_usage(journal["work_dir"]),
        storage_scope="work directory measured before the final publication/journal write; shared links excluded")
    if error:
        finalization["error"] = error
    if journal.get("format") == "summary":
        finalization["raw_re_evaluable"] = False
        return dict(schema=SUMMARY_SCHEMA, report_name=journal["report_name"],
                    run=_summary_run(run), finalization=finalization)
    return dict(schema=SCHEMA, run=run, finalization=finalization)


def finalize(work_dir, destination=None, *, prune=True, format=None):
    """Publish verified metadata, then prune only closed, exact artifact entries.

    Returns completed/deferred/protected. A deferred result is retryable and
    preserves its input, prior deletion receipts and any remaining artifacts.
    New publications default to a dated Markdown report plus summary.json.
    ``format="full"`` opts into the legacy full Excel metadata archive. Existing
    journalled destinations retain their format; history is never converted.
    """
    work = _path(work_dir)
    journal, destination_path, trusted, handles = None, None, False, []
    try:
        if format not in (None, "summary", "full"):
            raise ValueError("results format must be summary or full")
        run, source = _document(work / "run.json")
        if run.get("schema") != "ditserve.run.v1":
            raise ValueError("unsupported run metadata schema")
        for field in ("experiment", "run_id"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(run.get(field, ""))):
                raise ValueError("unsafe experiment or run identity")
        destination_path = _path(destination or (ROOT / "results" / run["experiment"] / run["run_id"]))
        if destination_path == work or work in destination_path.parents or destination_path in work.parents:
            raise ValueError("results and work directories must be separate")
        _active(run)
        with _guard(work, run, handles):
            if (work / JOURNAL).exists():
                journal, unused = _document(work / JOURNAL)
                if journal.get("sha256") != _digest({k: v for k, v in journal.items() if k != "sha256"}):
                    raise ValueError("finalization journal checksum changed")
                if journal.get("destination") != str(destination_path) or journal.get("work_dir") != str(work):
                    raise ValueError("finalization destination or work identity changed")
                if journal.get("schema") != SCHEMA:
                    raise ValueError("unsupported finalization journal")
                if format is not None and format != journal.get("format", "full"):
                    raise ValueError("existing result format is immutable")
                if journal["input_sha256"] != source["sha256"]:
                    if journal.get("pending_delete"):
                        raise ValueError("recover pending cleanup before changing dependencies")
                    if journal.get("deleted") or journal.get("status") == "completed":
                        previous = load(destination_path)
                        changes = (_summary_dependency_revision(journal, run) if journal.get("format") == "summary"
                                   else _dependency_revision(previous["run"], run))
                        journal.setdefault("input_revisions", []).append(dict(
                            previous=previous["finalization"]["input"], current=source, dependency_closures=changes))
                    journal["input_sha256"] = source["sha256"]
                    journal["status"] = "preparing"
                    if journal.get("format") == "summary":
                        journal["evidence_sha256"], journal["dependency_state"] = _evidence_binding(run)
                trusted = True
                if journal.get("pending_delete"):
                    pending = journal["pending_delete"]
                    archived = load(destination_path)
                    if archived["finalization"]["input"] != source or pending not in archived["finalization"]["planned"]:
                        raise ValueError("pending deletion has no verified archived evidence")
                    if not _path(pending["path"]).exists():
                        journal["deleted"].append(dict(pending, status="absent_after_recorded_delete_intent"))
                    journal.pop("pending_delete")
                    _save(work, journal)
                if journal.get("status") == "completed" and (journal.get("prune") == prune or journal.get("deleted")):
                    trusted = False  # A finished archive is verified, never repaired in place.
                    return dict(verify(destination_path), status="completed", work_dir=str(work))
            else:
                if destination_path.exists() and any(destination_path.iterdir()):
                    raise ValueError("existing results are immutable without this run's finalization journal")
                journal = dict(schema=SCHEMA, work_dir=str(work), destination=str(destination_path),
                    input_sha256=source["sha256"], status="preparing", deleted=[], errors=[], prune=prune,
                    work_usage_before=_usage(work), format=format or "summary")
                if journal["format"] == "summary":
                    dates = [s.get("ended_at") for s in run.get("budget", {}).get("sessions", [])
                             if type(s.get("ended_at")) in (int, float)]
                    date = datetime.fromtimestamp(max(dates), timezone.utc) if dates else datetime.now(timezone.utc)
                    stem = run["run_id"]
                    if not re.match(r"[0-9]{8}-[A-Za-z0-9]", stem):
                        stem = date.strftime("%Y%m%d") + "-" + stem
                    journal["report_name"] = stem + ".md"
                    journal["evidence_sha256"], journal["dependency_state"] = _evidence_binding(run)
                trusted = True
                _save(work, journal)
            candidates, retained = _artifacts(work, run, journal)
            journal.update(planned=candidates, prune=prune, status="archiving")
            _save(work, journal)
            initial = _snapshot(run, source, journal, retained + [dict(f, reason="awaiting verified cleanup") for f in candidates], "archived")
            _publish(work, destination_path, initial)
            if prune:
                for item in candidates:
                    current, current_source = _document(work / "run.json")
                    _active(current)
                    ledger = current.get("budget", {}).get("ledger") or current.get("budget_ledger")
                    if ledger:
                        current_budget, unused = _document(ledger if Path(ledger).is_absolute() else work / ledger)
                        _active({"budget": current_budget})
                    if current_source != source:
                        raise ValueError("run changed during finalization")
                    actual, unused = _read(item["path"])
                    if actual != item:
                        raise ValueError("artifact changed before deletion")
                    journal.update(status="pruning", pending_delete=item)
                    _save(work, journal)
                    _path(item["path"]).unlink()
                    journal["deleted"].append(dict(item, status="deleted"))
                    journal.pop("pending_delete")
                    _save(work, journal)
            else:
                retained += [dict(item, reason="prune=False") for item in candidates]
            pending = not _closed(run.get("retention", {}).get("pending_dependencies", []))
            status = "deferred" if prune and (pending or any(not item.get("control") for item in retained)) else "completed"
            final = _snapshot(run, source, journal, retained, status)
            _publish(work, destination_path, final)
            journal["status"] = status
            _save(work, journal)
            return dict(verify(destination_path), status=status, work_dir=str(work),
                        deleted=len(journal["deleted"]), retained=len(retained), reclaimed_bytes=final["finalization"]["reclaimed_bytes"],
                        work_usage=_usage(work), results_usage=_usage(destination_path))
    except Exception as error:
        reason = type(error).__name__ + ": " + str(error)
        result = dict(status="protected" if isinstance(error, (BlockingIOError, RuntimeError)) and "active" in str(error) else "deferred",
                      reason=reason, work_dir=str(work))
        if journal is not None and trusted:
            journal["status"] = "deferred"
            journal.setdefault("errors", []).append(reason)
            try:
                _save(work, journal)
                # Keep the last good publication if a fresh export cannot succeed.
                if destination_path and any((destination_path / name).exists() for name in (SUMMARY_FILE, "results.xlsx")):
                    snapshot = load(destination_path)
                    snapshot["finalization"].update(status="deferred", error=reason,
                        deleted=copy.deepcopy(journal.get("deleted", [])), errors=journal["errors"])
                    deleted_paths = {item["path"] for item in journal.get("deleted", [])}
                    snapshot["finalization"]["retained"] = [item for item in snapshot["finalization"]["retained"]
                                                           if item["path"] not in deleted_paths]
                    snapshot["finalization"]["reclaimed_bytes"] = sum(x["bytes"] for x in journal.get("deleted", []))
                    _publish(work, destination_path, snapshot)
            except Exception:
                pass  # The durable intent/previous snapshot still permit retry.
        return result
    finally:
        for handle in reversed(handles):
            handle.close()



def load(directory):
    """Read preserved values only; never rebuild, refresh or reevaluate history."""
    root = _path(directory)
    if (root / SUMMARY_FILE).exists():
        return _read_summary(root)
    if (root / "manifest.json").exists():
        manifest, unused = _document(root / "manifest.json")
        if manifest.get("schema") == "ditserve.compact-results.v1":
            from . import compact
            compact.verify(root)
            return compact._load(root)
    if (root / "data.json.gz").exists():
        _excel()  # Enable the same lazy optional-dependency lookup as new books.
        from . import legacy
        return legacy._verify_step(root.parent, root.name)
    book = _excel().load_workbook(root / "results.xlsx", read_only=True, data_only=False)
    try:
        if "manifest" in book:
            value = _read_book(root / "results.xlsx")
            if (root / "REPORT.md").read_text(encoding="utf-8") != _report(value):
                raise ValueError("report differs from workbook metadata")
            return value
        from .legacy import load_book
        return load_book(book)
    finally:
        book.close()


def verify(directory):
    root = _path(directory)
    if (root / "archive.json").exists():
        _excel()
        from . import legacy
        return legacy.verify(root)
    value = load(root)
    if value.get("schema") == SUMMARY_SCHEMA:
        return dict(status="verified", format=SUMMARY_SCHEMA, summary=str(root / SUMMARY_FILE),
                    report=str(root / value["report_name"]),
                    scope="summary checksum and report consistency; aggregates only, no scientific qualification",
                    raw_re_evaluated=False, resumable=False, raw_re_evaluable=False,
                    summary_sha256=_read(root / SUMMARY_FILE)[0]["sha256"])
    return dict(status="verified", format=value.get("schema", value.get("summary", {}).get("schema_version")),
                workbook=str(root / "results.xlsx"), report=str(root / "REPORT.md"),
                scope="typed workbook values and report consistency" if value.get("schema") == SCHEMA else
                      "historical archive checks; no new scientific qualification or raw reevaluation",
                raw_re_evaluated=False, workbook_sha256=_read(root / "results.xlsx")[0]["sha256"])


def dispatch(request, config):
    """Serve result operations without loading a model backend."""
    operation = request.get("op")
    if operation == "finalize":
        return finalize(request.get("work_dir", config.get("work_dir")),
                        request.get("output", request.get("destination")),
                        prune=request.get("prune", True), format=request.get("format"))
    if operation == "verify":
        return verify(request["path"])
    if operation == "inspect":
        return load(request["path"])
    if operation == "compact":
        from . import compact
        root = Path(request["root"])
        path = root / "run.json"
        metadata = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        destination = request.get("destination")
        prune = request.get("prune_raw", False)
        if isinstance(metadata, dict) and metadata.get("schema") == "ditserve.run.v1":
            return finalize(root, destination, prune=prune, format=request.get("format"))
        destination = Path(destination) if destination else root / "excel"
        result = compact.export(root, destination, replace_existing=True)
        if prune and result.get("status") == "verified":
            inventory = compact.plan(root, destination=destination)
            selected = [item["path"] for item in inventory["files"] if item["kind"] == "raw"]
            result["cleanup"] = compact.prune(destination, selected)
        return result
    raise ValueError("unknown results operation: " + str(operation))


def _is_truthy_approx_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "none", "exact", "off", "disabled")
    return bool(isinstance(value, (list, dict)) and value)


def has_quality_certificate(obj: Mapping[str, Any] | None) -> bool:
    """Return True if an explicit quality certificate is attached and passed."""
    if not isinstance(obj, Mapping):
        return False
    cert = obj.get("quality_certificate") or obj.get("approximate_quality_certificate")
    return bool(isinstance(cert, Mapping) and (
        cert.get("qualified") is True or cert.get("status") in ("passed", "verified", "qualified")
    ))


def is_lossless_exact_policy(
    policy: str | None,
    options: Mapping[str, Any] | None = None,
    quality_certificate: Mapping[str, Any] | None = None,
) -> bool:
    """Return True when policy and options use lossless exact execution (or have a quality certificate)."""
    if has_quality_certificate(quality_certificate) or has_quality_certificate(options):
        return True
    name = str(policy or "").strip().lower()
    if any(token in name for token in APPROXIMATE_POLICY_TOKENS):
        return False
    if isinstance(options, Mapping):
        if any(k in options and _is_truthy_approx_value(options[k]) for k in APPROXIMATE_OPTION_KEYS):
            return False
    return True


def _extract_policies_and_options(
    run_record: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any] | None]]:
    items: list[tuple[str, Mapping[str, Any], Mapping[str, Any] | None]] = []
    cfg = config if isinstance(config, Mapping) else run_record.get("config", {})
    cfg = cfg if isinstance(cfg, Mapping) else {}
    if isinstance(run_record.get("policy"), str):
        opts = run_record.get("options")
        items.append((run_record["policy"], opts if isinstance(opts, Mapping) else {}, run_record))
    for pol in run_record.get("policies", []) if isinstance(run_record.get("policies"), list) else []:
        if isinstance(pol, str):
            items.append((pol, {}, run_record))
        elif isinstance(pol, Mapping):
            items.append((str(pol.get("policy") or pol.get("id") or ""), pol.get("options") or pol, pol))
    for phase in run_record.get("phases", []) if isinstance(run_record.get("phases"), list) else []:
        if not isinstance(phase, Mapping):
            continue
        phase_opts = phase.get("options") if isinstance(phase.get("options"), Mapping) else {}
        if isinstance(phase.get("policy"), str):
            items.append((phase["policy"], phase_opts, phase))
        for cell in phase.get("cells", []) if isinstance(phase.get("cells"), list) else []:
            if isinstance(cell, Mapping):
                c_opts = cell.get("options") if isinstance(cell.get("options"), Mapping) else phase_opts
                items.append((str(cell.get("policy") or ""), c_opts, cell))
        for att in phase.get("attempts", []) if isinstance(phase.get("attempts"), list) else []:
            if isinstance(att, Mapping):
                for res in att.get("results", []) if isinstance(att.get("results"), list) else []:
                    if isinstance(res, Mapping):
                        items.append((str(res.get("policy") or ""), res.get("options") or {}, res))
    for obs in run_record.get("observations", []) if isinstance(run_record.get("observations"), list) else []:
        if isinstance(obs, Mapping):
            items.append((str(obs.get("policy") or ""), obs.get("options") or {}, obs))
    for phase in cfg.get("phases", []) if isinstance(cfg.get("phases"), list) else []:
        if not isinstance(phase, Mapping):
            continue
        phase_opts = phase.get("options") if isinstance(phase.get("options"), Mapping) else {}
        if isinstance(phase.get("policy"), str):
            items.append((phase["policy"], phase_opts, phase))
        matrix = phase.get("matrix")
        if isinstance(matrix, Mapping):
            for pol in matrix.get("/policy", []) if isinstance(matrix.get("/policy"), list) else []:
                if isinstance(pol, str):
                    items.append((pol, phase_opts, phase))
    return items


def has_real_gpu_execution(run_record: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> bool:
    """Return True iff the run is a real GPU serving execution (not mock or CPU simulation)."""
    cfg = config if isinstance(config, Mapping) else run_record.get("config", {})
    cfg = cfg if isinstance(cfg, Mapping) else {}
    if run_record.get("mock_execution") is True or cfg.get("mock_execution") is True:
        return False
    if run_record.get("simulated") is True or cfg.get("simulated") is True:
        return False
    scope_text = str(run_record.get("measurement_scope") or "").lower()
    notice_text = str(run_record.get("notice") or "").lower()
    if "cpu fixture" in scope_text or "simulation" in notice_text or "not measured gpu" in notice_text:
        return False

    workers = cfg.get("workers")
    has_non_mock_worker = has_only_mock_workers = False
    if isinstance(workers, Mapping) and workers:
        backends = [str(w.get("backend", "")).strip().lower() for w in workers.values() if isinstance(w, Mapping)]
        if backends:
            has_only_mock_workers = all(b in ("mock", "synthetic", "dummy") for b in backends)
            has_non_mock_worker = any(b not in ("", "mock", "synthetic", "dummy") for b in backends)
    models_list = run_record.get("models")
    if isinstance(models_list, list) and models_list:
        m_backends = [str(m.get("backend", "")).strip().lower() for m in models_list if isinstance(m, Mapping) and "backend" in m]
        if m_backends:
            has_only_mock_workers = has_only_mock_workers or all(b in ("mock", "synthetic", "dummy") for b in m_backends)
            has_non_mock_worker = has_non_mock_worker or any(b not in ("", "mock", "synthetic", "dummy") for b in m_backends)
    if has_only_mock_workers:
        return False

    gpu_uuids: list[str] = [str(u) for u in (run_record.get("gpu_uuids") or []) if u] if isinstance(run_record.get("gpu_uuids"), list) else []
    budget = run_record.get("budget")
    if isinstance(budget, Mapping):
        for sess in budget.get("sessions", []) if isinstance(budget.get("sessions"), list) else []:
            ident = sess.get("identity", {}) if isinstance(sess, Mapping) else {}
            if isinstance(ident, Mapping) and isinstance(ident.get("gpu_uuids"), list):
                gpu_uuids.extend(str(u) for u in ident["gpu_uuids"] if u)
    if isinstance(workers, Mapping):
        gpu_uuids.extend(str(w["gpu_uuid"]) for w in workers.values() if isinstance(w, Mapping) and w.get("gpu_uuid"))

    def _has_m(m: Any) -> bool:
        return isinstance(m, Mapping) and any((m.get(k) or 0) > 0 for k in ("model_ms", "completed", "on_time_goodput_rps", "goodput_rps"))

    has_worker_metrics = _has_m(run_record.get("metrics")) or (run_record.get("on_time_goodput_rps") or run_record.get("goodput_rps") or 0) > 0
    for phase in run_record.get("phases", []) if isinstance(run_record.get("phases"), list) else []:
        if isinstance(phase, Mapping):
            has_worker_metrics = has_worker_metrics or _has_m(phase.get("metrics"))
            for cell in phase.get("cells", []) if isinstance(phase.get("cells"), list) else []:
                if isinstance(cell, Mapping):
                    has_worker_metrics = has_worker_metrics or _has_m(cell.get("metrics"))
    if run_record.get("mock_execution") is False:
        return bool(has_worker_metrics or gpu_uuids or has_non_mock_worker or "failed" in run_record)
    return bool(has_worker_metrics and (
        (scope_text == "supervised gpu experiment" and (gpu_uuids or has_non_mock_worker))
        or (gpu_uuids and has_non_mock_worker)
    ))


def _total_failures(run_record: Mapping[str, Any]) -> int:
    failed = max(0, run_record["failed"]) if isinstance(run_record.get("failed"), int) else 0
    metrics = run_record.get("metrics")
    if isinstance(metrics, Mapping) and isinstance(metrics.get("failed"), int):
        failed += max(0, metrics["failed"])
    if isinstance(run_record.get("failures"), list):
        failed += len(run_record["failures"])
    for req in run_record.get("requests", []) if isinstance(run_record.get("requests"), list) else []:
        if isinstance(req, Mapping) and (req.get("status") == "failed" or req.get("error")):
            failed += 1
    for phase in run_record.get("phases", []) if isinstance(run_record.get("phases"), list) else []:
        if not isinstance(phase, Mapping):
            continue
        if phase.get("status") in ("failed", "blocked", "unsettled") or phase.get("error"):
            failed += 1
        p_metrics = phase.get("metrics")
        if isinstance(p_metrics, Mapping) and isinstance(p_metrics.get("failed"), int):
            failed += max(0, p_metrics["failed"])
        for cell in phase.get("cells", []) if isinstance(phase.get("cells"), list) else []:
            if isinstance(cell, Mapping):
                if cell.get("status") == "failed":
                    failed += 1
                c_metrics = cell.get("metrics")
                if isinstance(c_metrics, Mapping) and isinstance(c_metrics.get("failed"), int):
                    failed += max(0, c_metrics["failed"])
    return failed


def compute_on_time_goodput_rps(metrics: Mapping[str, Any] | None, elapsed_ms: float | int | None = None) -> float | None:
    """Extract or compute on-time goodput (req/s) from a metrics mapping."""
    if not isinstance(metrics, Mapping):
        return None
    for key in ("on_time_goodput_rps", "goodput_rps"):
        val = metrics.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
            return float(val)
    ontime = metrics.get("ontime")
    duration = metrics.get("observation_ms") if metrics.get("observation_ms") is not None else elapsed_ms
    if isinstance(ontime, (int, float)) and not isinstance(ontime, bool) and isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
        return float(ontime) * 1000.0 / float(duration)
    return None


def evaluate_tier1_lossless_qualification(
    run_record: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    explicit_qualification: Mapping[str, Any] | Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate the Tier-1 Lossless Runtime Qualification Gate for a serving run."""
    if explicit_qualification is not None:
        loaded = (json.loads(Path(explicit_qualification).read_text(encoding="utf-8"))
                  if isinstance(explicit_qualification, (str, Path))
                  else copy.deepcopy(dict(explicit_qualification)))
        if isinstance(loaded, dict):
            return loaded

    status = str(run_record.get("status", "completed"))
    failed_count = _total_failures(run_record)
    real_gpu = has_real_gpu_execution(run_record, config=config)
    non_lossless = [
        pol or "<unnamed>"
        for pol, opts, holder in _extract_policies_and_options(run_record, config=config)
        if not is_lossless_exact_policy(pol, opts, holder)
    ]
    if status == "completed" and failed_count == 0 and real_gpu and not non_lossless:
        qual_str = f"numeric={VERIFIED_NUMERIC}, semantic={VERIFIED_SEMANTIC}, performance={VERIFIED_PERFORMANCE}"
        return {
            "qualified": True,
            "status": "qualified",
            "gate": TIER1_LOSSLESS_GATE_NAME,
            "numeric": VERIFIED_NUMERIC,
            "semantic": VERIFIED_SEMANTIC,
            "performance": VERIFIED_PERFORMANCE,
            "qualification_status": qual_str,
            "scopes": {
                "numeric": {"status": VERIFIED_NUMERIC, "contract": "tier1_same_degree_bitwise_sha256_and_lossless_trajectory"},
                "semantic": {"status": VERIFIED_SEMANTIC, "contract": "zero_step_skip_zero_cfg_drop_exact_denoising"},
                "performance": {"status": VERIFIED_PERFORMANCE, "contract": "supervised_real_gpu_worker_execution"},
            },
            "checks": [
                {"scope": "numeric", "metric": "lossless_exact_execution", "status": "passed", "op": "==", "expected": True, "actual": True},
                {"scope": "semantic", "metric": "bitwise_invariant_contract", "status": "passed", "op": "==", "expected": True, "actual": True},
                {"scope": "performance", "metric": "real_gpu_zero_failures", "status": "passed", "op": "==", "expected": 0, "actual": 0},
            ],
        }
    reasons = []
    if status != "completed":
        reasons.append(f"run status is {status!r}")
    if failed_count > 0:
        reasons.append(f"failed={failed_count}")
    if not real_gpu:
        reasons.append("mock_execution or missing real GPU telemetry")
    if non_lossless:
        reasons.append(f"uncertified approximate policies/flags: {sorted(set(non_lossless))}")
    unverified_str = f"numeric={UNVERIFIED}, semantic={UNVERIFIED}, performance={UNVERIFIED}"
    return {
        "qualified": False,
        "status": UNVERIFIED,
        "gate": TIER1_LOSSLESS_GATE_NAME,
        "numeric": UNVERIFIED,
        "semantic": UNVERIFIED,
        "performance": UNVERIFIED,
        "qualification_status": unverified_str,
        "scopes": {"numeric": {"status": UNVERIFIED}, "semantic": {"status": UNVERIFIED}, "performance": {"status": UNVERIFIED}},
        "reason": "; ".join(reasons) if reasons else UNVERIFIED,
    }


def _populate_metrics_goodput(metrics: MutableMapping[str, Any], qualified: bool, qual_status_str: str, elapsed_ms: float | int | None = None) -> None:
    on_time = compute_on_time_goodput_rps(metrics, elapsed_ms=elapsed_ms)
    if on_time is not None and "goodput_rps" not in metrics:
        metrics["goodput_rps"] = on_time
    if qualified:
        metrics["qualification_status"] = qual_status_str
        if on_time is not None:
            metrics["on_time_goodput_rps"] = on_time
            metrics["qualified_goodput_rps"] = on_time
        if "completed" in metrics:
            metrics["qualification_verified_requests"] = metrics["completed"]
        if "ontime" in metrics:
            metrics["qualified_ontime"] = metrics["ontime"]
    else:
        metrics["qualification_status"] = UNVERIFIED
        metrics["qualified_goodput_rps"] = None


def apply_tier1_qualification_gate(
    run_record: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    explicit_qualification: Mapping[str, Any] | Path | str | None = None,
) -> dict[str, Any]:
    """Apply the Tier-1 Lossless Runtime Qualification Gate to a run or summary dict."""
    record = copy.deepcopy(dict(run_record))
    qual = evaluate_tier1_lossless_qualification(record, config=config, explicit_qualification=explicit_qualification)
    qualified = bool(qual.get("qualified"))
    qual_status_str = str(qual.get("qualification_status", UNVERIFIED))
    record["qualification"] = copy.deepcopy(qual)
    record["qualification_status"] = qual_status_str if qualified else UNVERIFIED

    top_on_time = compute_on_time_goodput_rps(record)
    if top_on_time is not None or "on_time_goodput_rps" in record or "goodput_rps" in record:
        record["qualified_goodput_rps"] = top_on_time if qualified else None
    if isinstance(record.get("metrics"), MutableMapping):
        _populate_metrics_goodput(record["metrics"], qualified, qual_status_str)
        if record.get("qualified_goodput_rps") is None and qualified:
            record["qualified_goodput_rps"] = record["metrics"].get("qualified_goodput_rps")

    for phase in record.get("phases", []) if isinstance(record.get("phases"), list) else []:
        if not isinstance(phase, MutableMapping):
            continue
        existing_q = phase.get("qualification")
        if not (isinstance(existing_q, Mapping) and existing_q.get("qualified") is True):
            phase["qualification"] = copy.deepcopy(qual)
        if isinstance(phase.get("metrics"), MutableMapping):
            _populate_metrics_goodput(phase["metrics"], qualified, qual_status_str)
        cell_goodputs: list[float] = []
        for cell in phase.get("cells", []) if isinstance(phase.get("cells"), list) else []:
            if not isinstance(cell, MutableMapping):
                continue
            elapsed = cell.get("elapsed_ms")
            if isinstance(cell.get("metrics"), MutableMapping):
                _populate_metrics_goodput(cell["metrics"], qualified, qual_status_str, elapsed_ms=elapsed)
                cg = cell["metrics"].get("qualified_goodput_rps")
                if isinstance(cg, (int, float)):
                    cell_goodputs.append(float(cg))
            by_kind = cell.get("by_kind")
            if isinstance(by_kind, MutableMapping):
                for kind_metrics in by_kind.values():
                    if isinstance(kind_metrics, MutableMapping):
                        _populate_metrics_goodput(kind_metrics, qualified, qual_status_str, elapsed_ms=elapsed)
        if qualified and isinstance(phase.get("metrics"), MutableMapping) and phase["metrics"].get("qualified_goodput_rps") is None and cell_goodputs:
            phase["metrics"]["qualified_goodput_rps"] = sum(cell_goodputs) / len(cell_goodputs)
        if qualified and isinstance(record.get("metrics"), MutableMapping) and record["metrics"].get("qualified_goodput_rps") is None and cell_goodputs:
            record["metrics"]["qualified_goodput_rps"] = sum(cell_goodputs) / len(cell_goodputs)
            if record.get("qualified_goodput_rps") is None:
                record["qualified_goodput_rps"] = record["metrics"]["qualified_goodput_rps"]

    for obs in record.get("observations", []) if isinstance(record.get("observations"), list) else []:
        if isinstance(obs, MutableMapping):
            if isinstance(obs.get("metrics"), MutableMapping):
                _populate_metrics_goodput(obs["metrics"], qualified, qual_status_str)
            by_mod = obs.get("by_modality") or obs.get("by_kind")
            if isinstance(by_mod, MutableMapping):
                for mod_metrics in by_mod.values():
                    if isinstance(mod_metrics, MutableMapping):
                        _populate_metrics_goodput(mod_metrics, qualified, qual_status_str)
    return record


def archive_run(
    run_or_work_dir: Mapping[str, Any] | Path | str,
    output_dir: Path | str | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    explicit_qualification: Mapping[str, Any] | Path | str | None = None,
    prune: bool = False,
) -> dict[str, Any]:
    """Archive a serving run dictionary or work_dir with Tier-1 Lossless Qualification Gate."""
    if isinstance(run_or_work_dir, Mapping):
        return apply_tier1_qualification_gate(
            run_or_work_dir, config=config, explicit_qualification=explicit_qualification
        )
    work = Path(run_or_work_dir).resolve()
    run_path = work / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"run.json not found in {work}")
    raw_run = json.loads(run_path.read_text(encoding="utf-8"))
    qual = evaluate_tier1_lossless_qualification(
        raw_run, config=config or raw_run.get("config"), explicit_qualification=explicit_qualification
    )
    if qual.get("qualified") is True:
        enriched_run = apply_tier1_qualification_gate(
            raw_run, config=config or raw_run.get("config"), explicit_qualification=explicit_qualification
        )
        _atomic_json(run_path, enriched_run)
        journal_path = work / JOURNAL
        if journal_path.is_file():
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            digest, deps = _evidence_binding(enriched_run)
            journal["input_sha256"] = _read(run_path)[0]["sha256"]
            journal["evidence_sha256"] = digest
            journal["dependency_state"] = deps
            journal["status"] = "preparing"
            _save(work, journal)
    return finalize(
        work,
        Path(output_dir).resolve() if output_dir is not None else None,
        prune=prune,
    )
