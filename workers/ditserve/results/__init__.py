"""Verified Excel + Markdown results, with resumable, explicit artifact cleanup.

``run.json`` contains JSON metadata, never raw tensors or event bodies. Artifact
``dependencies`` lists outstanding consumers; an empty list closes that file's
dependencies. A closed dependency can instead be ``{closed: true, reason: ...}``.
Control files, models and frozen sources remain outside the deletion whitelist.
The workbook stores typed JSON nodes, not JSON blobs, and is independently
readable after the work directory is retired. Old archives are read-only.
"""

from contextlib import contextmanager
import copy
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

from ..runtime import ROOT


SCHEMA = "ditserve.results.v1"
JOURNAL = ".finalization.json"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_SHEET_ROWS = 1048576
CELL_LIMIT = 32767
CHUNK_SIZE = 16000
HEADERS = ("JSON pointer", "Type", "Part", "Value")
SECTIONS = {"config", "metrics", "requests", "phases", "failures", "budget", "retention"}
RAW_KINDS = {"raw", "array", "media"}
LOG_KINDS = {"log", "trace", "summarized_log"}
PROTECTED_PARTS = {"sources", "source-snapshots", "models", "weights", ".git", ".retention"}
PROTECTED_NAMES = {"run.json", JOURNAL, "budget.json", "gpu-budget.json", "ledger.json",
                   "config.json", "protocol.json", "manifest.json", "results.xlsx", "REPORT.md"}
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
        _write_book(staging / "results.xlsx", value)
        (staging / "REPORT.md").write_text(_report(value), encoding="utf-8")
        if _digest(_read_book(staging / "results.xlsx")) != _digest(value):
            raise ValueError("Excel readback changed metadata")
        for name in ("results.xlsx", "REPORT.md"):
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
        handle = path.open("a+b" if path == work / ".finalize.lock" else "rb")
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
    return dict(schema=SCHEMA, run=run, finalization=finalization)


def finalize(work_dir, destination=None, *, prune=True):
    """Publish verified metadata, then prune only closed, exact artifact entries.

    Returns completed/deferred/protected. A deferred result is retryable and
    preserves its input, prior deletion receipts and any remaining artifacts.
    Existing historical destinations are never replaced.
    """
    work = _path(work_dir)
    journal, destination_path, trusted, handles = None, None, False, []
    try:
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
                if journal["input_sha256"] != source["sha256"]:
                    if journal.get("pending_delete"):
                        raise ValueError("recover pending cleanup before changing dependencies")
                    if journal.get("deleted"):
                        previous = load(destination_path)
                        changes = _dependency_revision(previous["run"], run)
                        journal.setdefault("input_revisions", []).append(dict(
                            previous=previous["finalization"]["input"], current=source, dependency_closures=changes))
                    journal["input_sha256"] = source["sha256"]
                    journal["status"] = "preparing"
                trusted = True
                if journal.get("pending_delete"):
                    pending = journal["pending_delete"]
                    archived = _read_book(destination_path / "results.xlsx")
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
                    work_usage_before=_usage(work))
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
                # Keep the last good workbook if a fresh export cannot succeed.
                if destination_path and (destination_path / "results.xlsx").exists():
                    snapshot = _read_book(destination_path / "results.xlsx")
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
                        prune=request.get("prune", True))
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
            return finalize(root, destination, prune=prune)
        destination = Path(destination) if destination else root / "excel"
        result = compact.export(root, destination, replace_existing=True)
        if prune and result.get("status") == "verified":
            inventory = compact.plan(root, destination=destination)
            selected = [item["path"] for item in inventory["files"] if item["kind"] == "raw"]
            result["cleanup"] = compact.prune(destination, selected)
        return result
    raise ValueError("unknown results operation: " + str(operation))
