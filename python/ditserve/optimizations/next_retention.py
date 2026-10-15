"""Explicit, CPU-only retention for a closed next-study run.

``plan`` is read-only. ``archive_and_prune`` defaults to export/readback only;
deletion additionally requires exact per-file approvals and dependency closure.
No function is called at import time, and no tensor or model library is used.
The archive contains recorded metrics, not enough data to rerun quality tests.
"""

import collections
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import zipfile
import xml.etree.ElementTree as ET

from ..research.common import atomic_json, digest
from .next_protocol import PHASES, ROOT, safe_paths


SCHEMA = "ditserve.optimizations.next_retention.v1"
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_FILES = 100000
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
RAW_NAMES = frozenset(("diagnosis.json", "cache_load_05.json", "cache_load_07.json",
                       "cache_load_09.json", "sana-online.json"))
TRACE_NAMES = frozenset(("trace.json", "partial.json", "trace-flushes.json"))


def _path(value):
    path = Path(value).absolute()
    if path != path.resolve() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("symbolic links and noncanonical paths are not retention inputs: " + str(path))
    return path


def _inside(path, root):
    return path == root or root in path.parents


def _read(path):
    path = _path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("metadata must be an ordinary file: " + str(path))
    if before.st_size > MAX_JSON_BYTES:
        raise ValueError("bounded metadata JSON exceeded: " + str(path))
    value = json.loads(path.read_bytes())
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("metadata changed during read: " + str(path))
    return value


def _binding(path, private=False):
    path = _path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or (private and before.st_nlink != 1):
        raise ValueError("retention requires ordinary unshared private files: " + str(path))
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink):
        raise ValueError("file changed while hashing: " + str(path))
    return dict(path=str(path), bytes=before.st_size, sha256=sha.hexdigest())


def _check(binding, deleted=None):
    if not Path(binding["path"]).is_absolute():
        raise ValueError("source bindings must use absolute paths")
    path = _path(binding["path"])
    if deleted and str(path) in deleted:
        approved = deleted[str(path)]
        if binding.get("sha256") != approved["sha256"] or (
                "bytes" in binding and binding["bytes"] != approved["bytes"]):
            raise ValueError("deleted source differs from approved raw identity")
        if path.exists():
            raise ValueError("deleted source has reappeared: " + str(path))
        return approved
    actual = _binding(path)
    if actual["sha256"] != binding["sha256"] or ("bytes" in binding and actual["bytes"] != binding["bytes"]):
        raise ValueError("bound source changed: " + str(path))
    return actual


def _excel_readback(path, report):
    """Read literal XLSX cells with stdlib; compare every cell to report tables."""
    from .next_report import _tables
    tables = _tables(report)
    with zipfile.ZipFile(_path(path)) as book:
        infos = book.infolist()
        if len(infos) > 1000 or sum(i.file_size for i in infos) > MAX_JSON_BYTES * 4:
            raise ValueError("workbook exceeds bounded readback size")
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        links = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        targets = {node.attrib["Id"]: node.attrib["Target"] for node in links}
        sheets = workbook.find(NS + "sheets")
        if [sheet.attrib["name"] for sheet in sheets] != list(tables):
            raise ValueError("workbook sheet identities differ")
        shared = []
        if "xl/sharedStrings.xml" in book.namelist():
            shared = ["".join(node.itertext()) for node in ET.fromstring(book.read("xl/sharedStrings.xml"))]
        cells = 0
        for sheet in sheets:
            name = sheet.attrib["name"]
            target = targets[sheet.attrib[REL + "id"]]
            member = target.lstrip("/") if target.startswith("/") else "xl/" + target
            if ".." in Path(member).parts:
                raise ValueError("workbook relationship leaves its archive")
            expected = [[None if value is None else str(value) for value in row] for row in tables[name]]
            actual = [[None] * len(row) for row in expected]
            root = ET.fromstring(book.read(member))
            rows = root.find(NS + "sheetData")
            if len(rows) != len(expected):
                raise ValueError("workbook row count differs: " + name)
            seen = set()
            for row in rows:
                for cell in row:
                    address = cell.attrib.get("r", "")
                    match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", address)
                    if not match or address in seen or cell.find(NS + "f") is not None:
                        raise ValueError("workbook has duplicate, invalid, or formula cells")
                    seen.add(address)
                    col = 0
                    for letter in match[1]:
                        col = col * 26 + ord(letter) - 64
                    index = int(match[2]) - 1
                    if index >= len(actual) or col > len(actual[index]):
                        raise ValueError("workbook has unexpected extra cells")
                    kind = cell.attrib.get("t")
                    if kind == "inlineStr":
                        value = "".join(t.text or "" for t in cell.iter(NS + "t"))
                    elif kind == "s":
                        value = shared[int(cell.find(NS + "v").text)]
                    else:
                        raise ValueError("workbook cells must remain literal strings")
                    actual[index][col - 1] = value
                    cells += 1
            # The writer validates exact row order before atomic_json sorts
            # object keys. Reconstructing Metrics from that saved JSON changes
            # dictionary-walk order, but not literal row identity or count.
            equal = (actual == expected) if name != "Metrics" else (
                actual[:1] == expected[:1]
                and collections.Counter(map(tuple, actual[1:])) == collections.Counter(map(tuple, expected[1:])))
            if not equal:
                raise ValueError("workbook cell readback differs: " + name)
    return dict(status="verified", cells=cells, rows_by_sheet={k: len(v)-1 for k, v in tables.items()},
                scope="every emitted literal cell; no formulas or raw event bodies")


def _validate(output, ledger, deleted=None):
    # Admission deliberately precedes reading or hashing any large raw files.
    output, ledger = _path(output), _path(ledger)
    safe_paths(output, ledger)
    budget = _read(ledger)
    if "active" not in budget or budget["active"] is not None:
        raise ValueError("active or unconfirmed GPU session; dependencies remain open")
    if (budget.get("schema_version") != "ditserve.optimizations.next_budget.v1"
            or budget.get("budget_unlimited") is not True or budget.get("limit_gpu_seconds", 0) is not None):
        raise ValueError("not the independent unlimited next-study ledger")
    terminals = []
    for phase in PHASES:
        terminal = _read(output / phase / "terminal.json")
        attempt = _path(terminal["output"])
        if attempt.parent != output / phase or re.fullmatch(r"attempt[0-9]+", attempt.name) is None:
            raise ValueError("terminal refers outside its phase")
        if terminal != _read(attempt / "supervisor.json"):
            raise ValueError("terminal and supervisor differ")
        terminals.append(_binding(output / phase / "terminal.json"))
    report = _read(output / "report.json")
    audit = _read(output / "audits/final-integrity.json")
    if (audit.get("status") != "passed" or audit.get("output") != str(output)
            or audit.get("all_phases_terminal") is not True):
        raise ValueError("independent final integrity audit has not passed")
    if audit.get("pending_dependencies") or report.get("pending_dependencies"):
        raise ValueError("explicit pending evaluation or continuation dependencies")
    if (report.get("schema_version") != "ditserve.optimizations.next_report.v1"
            or report.get("output") != str(output) or report.get("budget_path") != str(ledger)
            or [p.get("phase") for p in report.get("phases", [])] != list(PHASES)):
        raise ValueError("final report identity or phase list differs")
    report_binding = _binding(output / "report.json")
    excel_binding = _binding(output / "results.xlsx")
    if audit.get("report_sha256") != report_binding["sha256"] or audit.get("excel_sha256") != excel_binding["sha256"]:
        raise ValueError("audit does not bind the current final report and workbook")
    excel = report.get("excel", {})
    if (excel.get("status") != "verified" or excel.get("sha256") != excel_binding["sha256"]
            or excel.get("path") != excel_binding["path"] or _read(output / "excel-verification.json") != excel):
        raise ValueError("Excel verification binding differs")
    workbook_check = _excel_readback(output / "results.xlsx", report)
    bindings = {}
    for source in report.get("source_bindings", []):
        checked = _check(source, deleted)
        bindings[checked["path"]] = {k: checked[k] for k in ("path", "sha256", "bytes")}
    for source in budget.get("parents", {}).values():
        checked = _check(source)
        bindings[checked["path"]] = checked
    attempts = []
    sessions = budget.get("sessions", [])
    for phase in PHASES:
        for attempt in sorted((output / phase).glob("attempt*")):
            attempt = _path(attempt)
            if not attempt.is_dir() or not re.fullmatch(r"attempt[0-9]+", attempt.name):
                raise ValueError("unknown attempt directory")
            supervisor = _read(attempt / "supervisor.json")
            charge = supervisor.get("charge")
            if not isinstance(charge, dict) or sessions.count(charge) != 1:
                raise ValueError("attempt is not bound to exactly one settled charge")
            cleanup = supervisor.get("final_occupancy", {})
            if cleanup.get("owned") != [] or not isinstance(cleanup.get("foreign"), list):
                raise ValueError("attempt cleanup has not confirmed ownership release")
            if charge.get("identity", {}).get("output") != str(attempt):
                raise ValueError("settled charge output differs")
            snapshot = output / "sources" / (phase + "-" + attempt.name)
            manifest = _read(snapshot / "manifest.json")
            files = manifest.get("source_files", {})
            config = _read(attempt / "config.json")
            config_sha = digest(config)
            from .next_report import _device_count
            count = _device_count(charge, phase, config)
            elapsed, seconds = charge.get("elapsed_seconds"), charge.get("gpu_seconds")
            if (count is None or type(elapsed) not in (int, float) or type(seconds) not in (int, float)
                    or not math.isfinite(elapsed) or not math.isfinite(seconds) or elapsed < 0
                    or not math.isclose(seconds, elapsed * count, rel_tol=1e-12, abs_tol=1e-8)):
                raise ValueError("settled count/devices or full lifetime charge differs from the frozen configuration")
            if (not files or manifest.get("execution_root") != str(snapshot)
                    or manifest.get("source_sha256") != digest(files)
                    or supervisor.get("source_sha256") != digest(files)
                    or charge.get("identity", {}).get("source_sha256") != digest(files)
                    or any(value != config_sha for value in (manifest.get("config_sha256"),
                        supervisor.get("config_sha256"), charge.get("identity", {}).get("config_sha256"),
                        budget.get("config_sha256")))):
                raise ValueError("frozen source manifest differs")
            for relative, sha in files.items():
                path = _path(snapshot / relative)
                if not _inside(path, snapshot):
                    raise ValueError("snapshot file escaped its root")
                checked = _check(dict(path=str(path), sha256=sha))
                bindings[checked["path"]] = checked
            for path in (snapshot / "manifest.json", attempt / "supervisor.json", attempt / "config.json"):
                checked = _binding(path)
                bindings[checked["path"]] = checked
            attempts.append((phase, attempt))
    for path in (ledger, output / "report.json", output / "results.xlsx", output / "excel-verification.json",
                 output / "audits/final-integrity.json"):
        checked = _binding(path)
        bindings[checked["path"]] = checked
    for source in terminals:
        bindings[source["path"]] = source
    return dict(output=str(output), ledger=str(ledger), report=report, bindings=list(bindings.values()),
                excel_readback=workbook_check, attempts=attempts)


def _isolation_summary(path):
    counts, lines, malformed, unknown, foreign = {}, 0, 0, 0, 0
    with path.open("rb") as stream:
        while True:
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError("isolation log line exceeds bounded streaming limit")
            lines += 1
            try:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("non-object")
                status = value.get("status")
                counts[str(status)] = counts.get(str(status), 0) + 1
                unknown += int(status in (None, "unknown"))
                foreign += int(bool(value.get("foreign")))
            except (ValueError, UnicodeError):
                malformed += 1
    return dict(lines=lines, status_counts=counts, malformed_lines=malformed,
                unknown_status_lines=unknown, foreign_observation_lines=foreign,
                scope="streamed counts; failures retained; no original line text")


def _walk_refs(value, pointer=""):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and value["path"].endswith(".npy"):
            yield pointer, value
        for key, child in value.items():
            yield from _walk_refs(child, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_refs(child, pointer + "/" + str(index))


def _candidates(validated):
    output = Path(validated["output"])
    candidates = {}

    def add(path, kind, phase, attempt, **details):
        checked = _binding(path, private=True)
        old = candidates.get(checked["path"])
        if old:
            if checked["sha256"] != old["sha256"]:
                raise ValueError("duplicate artifact identity differs")
            if "reference" in details:
                old.setdefault("references", []).append(details["reference"])
            return
        row = dict(checked, kind=kind, phase=phase, attempt=str(attempt), **details)
        if "reference" in row:
            row["references"] = [row.pop("reference")]
        candidates[checked["path"]] = row
        if len(candidates) > MAX_FILES:
            raise ValueError("explicit retention file limit exceeded")

    for phase, attempt in validated["attempts"]:
        for name in RAW_NAMES:
            path = attempt / name
            if path.exists() or path.is_symlink():
                add(path, "phase_replay", phase, attempt,
                    summary=dict(source="frozen-report.json", phase=phase,
                                 scope="existing metrics and missing/failed states; raw events not copied"))
        for directory in (attempt / "partials", attempt / "online-partials"):
            if directory.exists() or directory.is_symlink():
                _path(directory)
                for parent, directories, files in os.walk(str(directory), followlinks=False):
                    for name in directories:
                        _path(Path(parent) / name)
                    for name in files:
                        path = _path(Path(parent) / name)
                        if name in TRACE_NAMES:
                            add(path, "trace_detail", phase, attempt,
                                summary=dict(source="frozen-report.json", phase=phase,
                                             scope="bounded request/event summaries; no raw event text"))
        isolation = attempt / "isolation.jsonl"
        if isolation.exists() or isolation.is_symlink():
            _path(isolation)
            add(isolation, "isolation_log", phase, attempt, summary=_isolation_summary(isolation))
        environment_path = attempt / "environment.json"
        environment = _read(environment_path) if environment_path.exists() else {}
        for name, key in (("compiler-cache", "compiler_cache"), ("triton-cache", "triton_cache")):
            directory = attempt / name
            if directory.exists() or directory.is_symlink():
                _path(directory)
                if environment.get(key) != str(directory):
                    raise ValueError("private compiler cache lacks matching environment ownership")
                for parent, directories, files in os.walk(str(directory), followlinks=False):
                    for child in directories:
                        _path(Path(parent) / child)
                    for child in files:
                        add(Path(parent) / child, "private_compiler_cache", phase, attempt,
                            ownership_source=_binding(environment_path))
        resume = attempt / "sana-wan-resume.json"
        if resume.exists():
            config = _read(attempt / "config.json")
            artifact_root = _path(config["artifact_root"])
            if artifact_root != _path(ROOT / ".cache/optimization-next"):
                raise ValueError("resume arrays must belong to the configured next-study private root")
            owned = artifact_root / output.name / phase / attempt.name
            source = _binding(resume)
            for pointer, reference in _walk_refs(_read(resume)):
                path = _path(reference["path"])
                if path.parent != owned or not reference.get("sha256"):
                    raise ValueError("resume array reference is outside this run's private attempt")
                _check(reference)
                add(path, "resume_npy", phase, attempt,
                    reference=dict(source=source, pointer=pointer, recorded_sha256=reference["sha256"],
                                   shape=reference.get("shape")),
                    summary=dict(shape=reference.get("shape"), raw_re_evaluable=False))
    return sorted(candidates.values(), key=lambda row: row["path"])


def plan(output, ledger):
    """Read-only closed-run candidate list. No raw scan if admission fails.

    A terminal quality/compile failure may be archived. ``report.partial`` is
    descriptive scientific incompleteness, not an automatic dependency veto.
    Explicit pending dependencies or an unpassed integrity audit do block.
    """
    try:
        validated = _validate(output, ledger)
        candidates = _candidates(validated)
        return dict(schema_version=SCHEMA, status="planned", ready=True,
            output=validated["output"], ledger=validated["ledger"], candidates=candidates,
            candidate_count=len(candidates), candidate_bytes=sum(r["bytes"] for r in candidates),
            source_bindings=validated["bindings"], excel_readback=validated["excel_readback"],
            explicit_dependencies_closed=False, originals_deleted=0,
            retained="report/Excel, ledger, terminals/supervisors, configurations, frozen sources, "
                     "quality/numerics/recovery summaries, images, and all historical/shared files",
            scope="read-only list; no inferred evaluation/continuation closure or deletion authorization")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, ET.ParseError) as error:
        return dict(schema_version=SCHEMA, status="blocked", ready=False, output=str(Path(output).absolute()),
                    ledger=str(Path(ledger).absolute()), errors=[type(error).__name__ + ": " + str(error)],
                    candidates=[], originals_deleted=0)


def _approved(whitelist, candidates):
    if not isinstance(whitelist, list) or not whitelist:
        raise ValueError("pruning requires an explicit nonempty per-file whitelist")
    available = {row["path"]: row for row in candidates}
    approved = []
    seen = set()
    for row in whitelist:
        path = str(_path(row["path"]))
        if path in seen or path not in available:
            raise ValueError("duplicate or noncandidate deletion path")
        source = available[path]
        if row.get("sha256") != source["sha256"] or row.get("bytes") != source["bytes"]:
            raise ValueError("whitelist size/hash no longer matches")
        if row.get("dependencies_closed") is not True or not str(row.get("closure_reason", "")).strip():
            raise ValueError("each deletion needs explicit dependency closure and its reason")
        seen.add(path)
        approved.append(dict(source, dependencies_closed=True, closure_reason=row["closure_reason"]))
    return approved


def _copy_exact(source, destination):
    _path(destination)
    expected = _binding(source)
    if destination.exists():
        actual = _binding(destination)
        if (actual["sha256"], actual["bytes"]) != (expected["sha256"], expected["bytes"]):
            raise ValueError("existing archive differs; it cannot be overwritten")
    else:
        with _path(source).open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
    return _binding(destination)


def archive_and_prune(output, ledger, *, explicit_dependencies_closed=False, whitelist=None, prune=False):
    """Export verified summaries; delete only an explicitly approved subset.

    Callers must close quality/blind-review/continuation dependencies themselves.
    ``prune=False`` never removes originals or activates archive-only reporting.
    A failed deletion leaves a journal and blocks raw fallback; it is not retried
    automatically and never pretends that all original data remain available.
    """
    if explicit_dependencies_closed is not True:
        raise ValueError("explicit evaluation, blind-review, and continuation dependency closure is required")
    output, ledger = _path(output), _path(ledger)
    lock = _path(ledger.with_suffix(".lock"))
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finalization_path = output / "next-archive/finalization.json"
        if finalization_path.exists() or finalization_path.is_symlink():
            raise ValueError("existing finalization must be reviewed; no automatic repeat cleanup")
        planned = plan(output, ledger)
        if not planned["ready"]:
            raise ValueError("retention admission blocked: " + "; ".join(planned["errors"]))
        selected = _approved(whitelist, planned["candidates"]) if prune else []
        archive = _path(output / "next-archive")
        archive.mkdir(exist_ok=True)
        frozen = _copy_exact(output / "report.json", archive / "frozen-report.json")
        workbook = _copy_exact(output / "results.xlsx", archive / "results.xlsx")
        readback = _excel_readback(archive / "results.xlsx", _read(archive / "frozen-report.json"))
        manifest = dict(schema_version=SCHEMA, output=str(output), ledger=str(ledger),
            frozen_report=frozen, workbook=workbook, source_bindings=planned["source_bindings"],
            candidates=planned["candidates"], excel_readback=readback,
            raw_re_evaluable=False, qualification_recomputed=False,
            scope="recorded metrics only; original sizes/hashes and missing/failed states retained")
        manifest_path = archive / "manifest.json"
        if manifest_path.exists() and _read(manifest_path) != manifest:
            raise ValueError("existing archive manifest differs")
        atomic_json(manifest_path, manifest)
        if _read(manifest_path) != manifest:
            raise ValueError("archive manifest readback differs")
        for binding in planned["source_bindings"]:
            _check(binding)
        result = dict(schema_version=SCHEMA, status="archived", output=str(output), ledger=str(ledger),
            explicit_dependencies_closed=True, manifest=_binding(manifest_path),
            selected=selected, deleted=[], originals_deleted=0, archive_verified=True,
            raw_re_evaluable=not prune, qualification_recomputed=False)
        if not prune:
            return result
        result["status"] = "in_progress"
        atomic_json(finalization_path, result)
        try:
            for row in selected:
                # The shared ledger lock prevents a new supervised phase. Recheck
                # the ledger binding and exact target immediately before unlink.
                ledger_binding = next(b for b in planned["source_bindings"] if b["path"] == str(ledger))
                _check(ledger_binding)
                actual = _binding(row["path"], private=True)
                if any(actual[key] != row[key] for key in ("sha256", "bytes")):
                    raise ValueError("raw file changed after approval")
                result["pending_delete"] = {k: row[k] for k in ("path", "sha256", "bytes")}
                atomic_json(finalization_path, result)
                Path(row["path"]).unlink()
                result["deleted"].append(dict(result.pop("pending_delete"), status="deleted"))
                result["originals_deleted"] = len(result["deleted"])
                atomic_json(finalization_path, result)
            deleted = {row["path"]: row for row in selected}
            for binding in planned["source_bindings"]:
                _check(binding, deleted)
            if any(Path(row["path"]).exists() for row in selected):
                raise ValueError("approved originals remain after pruning")
            result.update(status="completed", raw_re_evaluable=False,
                          reclaimed_bytes=sum(row["bytes"] for row in selected))
            atomic_json(finalization_path, result)
            return result
        except BaseException as error:
            result.update(status="failed", raw_re_evaluable=False,
                          error=type(error).__name__ + ": " + str(error))
            atomic_json(finalization_path, result)
            raise


def load_archive_report(output, ledger):
    """Verify an immutable historical view; never grant/recompute qualification."""
    output, ledger = _path(output), _path(ledger)
    final = _read(output / "next-archive/finalization.json")
    if (final.get("schema_version") != SCHEMA or final.get("status") != "completed"
            or final.get("explicit_dependencies_closed") is not True
            or final.get("output") != str(output) or final.get("ledger") != str(ledger)):
        raise ValueError("archive finalization is incomplete, failed, or belongs to another run")
    manifest_path = output / "next-archive/manifest.json"
    if final.get("manifest", {}).get("path") != str(manifest_path):
        raise ValueError("archive manifest identity differs")
    _check(final["manifest"])
    manifest = _read(manifest_path)
    selected = _approved(final.get("selected"), manifest["candidates"])
    expected = [dict({key: row[key] for key in ("path", "sha256", "bytes")}, status="deleted") for row in selected]
    if final.get("deleted") != expected or final.get("originals_deleted") != len(expected):
        raise ValueError("deletion receipt differs from the approved exact whitelist")
    deleted = {row["path"]: row for row in selected}
    for row in selected:
        if _path(row["path"]).exists():
            raise ValueError("deleted private artifact has reappeared")
    for binding in manifest["source_bindings"]:
        _check(binding, deleted)
    validated = _validate(output, ledger, deleted)
    for name, filename in (("frozen_report", "frozen-report.json"), ("workbook", "results.xlsx")):
        if manifest[name]["path"] != str(output / "next-archive" / filename):
            raise ValueError("archive copy escaped its fixed destination")
        _check(manifest[name])
    report = _read(manifest["frozen_report"]["path"])
    if report != validated["report"]:
        raise ValueError("frozen historical report differs from its retained original")
    _excel_readback(manifest["workbook"]["path"], report)
    result = copy.deepcopy(report)
    result.update(archive_only=True, raw_re_evaluable=False, raw_events_reevaluable=False,
        qualification_recomputed=False, new_qualification_granted=False,
        originals_deleted=len(expected), archive_view=dict(status="verified", manifest=final["manifest"],
            finalization=_binding(output / "next-archive/finalization.json"),
            deleted_files=len(expected), reclaimed_bytes=sum(row["bytes"] for row in selected),
            scope="historical recorded metrics only; deleted arrays/events cannot be reevaluated"))
    return result
