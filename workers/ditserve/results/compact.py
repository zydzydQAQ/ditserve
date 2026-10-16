"""Small, independently verifiable Excel archives of experiment metrics.

The gzip payload preserves UTF-8 metadata byte for byte. Arrays are indexed,
never embedded. Export never deletes evidence; ``prune`` requires an explicit
list of raw artifacts or explicitly aggregated logs, and verifies both the
archive and current sources first. Aggregation is labeled as irreversible.
Only the standard library is required, including for XLSX round-trip checking.
"""

from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET
import zipfile
import warnings

from ..evaluation.common import canonical_json as _json
from ..runtime import CONFIGS


SCHEMA = "ditserve.compact-results.v1"
MAX_TEXT_BYTES = 4 * 1024 * 1024
RAW_SUFFIXES = {".npy", ".npz", ".pkl", ".pt", ".pth", ".safetensors"}
PRUNABLE_KINDS = {"raw", "summarized_text"}
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".log", ".csv", ".tsv", ".yaml", ".yml"}
PROTECTED_NAMES = {"source-snapshots", "sources", "sources-final", "sources-after-validation",
                   ".git", ".retention", "__pycache__"}
MARKERS = {"ACTIVE", "RESUME_PENDING", "retention-blocked.json"}
FILES = {"data.json.gz", "results.xlsx", "manifest.json"}
INDEX_NAME = ".compact-results.json"
RECEIPT_NAME = "pruned.json"
NOTICE = ("metrics-only archive: original arrays are not embedded; saved metrics cannot "
          "be independently reevaluated without the original artifacts. Source files remain "
          "unchanged unless explicitly pruned after verification.")
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
AGGREGATION_NOTICE = ("aggregated_only: original individual records are not embedded and cannot "
                      "be reconstructed after deletion; the complete supplied summary, including "
                      "its limits and omitted-field counts, is preserved in the compressed payload.")


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _inside(path, root):
    return path == root or root in path.parents


def _real(path):
    path = Path(os.path.abspath(path))
    if path.resolve() != path:
        raise ValueError("symlink paths are not allowed: " + str(path))
    return path


def _read(path, collect=False, limit=None):
    """Read regular files without following symlinks, with bounded collection."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(str(path), flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("not a regular file: " + str(path))
        if collect and limit is not None and before.st_size > limit:
            raise ValueError("source exceeds text collection limit")
        digest, chunks = hashlib.sha256(), []
        count = 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            if collect:
                count += len(block)
                if limit is not None and count > limit:
                    raise ValueError("source grew beyond text collection limit")
                chunks.append(block)
        after = os.fstat(stream.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("source changed during read: " + str(path))
    return after.st_size, digest.hexdigest(), b"".join(chunks) if collect else None


def _walk(root, excluded=()):
    """Yield real directories, files and skipped paths without traversing links."""
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        skipped = [parent / name for name in dirs
                   if name in PROTECTED_NAMES or (parent / name).is_symlink()
                   or any(_inside(parent / name, p) for p in excluded)]
        dirs[:] = sorted(name for name in dirs if parent / name not in skipped)
        yield parent, sorted(files), skipped


def _ledgers(root, excluded=()):
    found = set()
    for parent, names, unused in _walk(root, excluded):
        found.update(parent / name for name in names if name in ("budget.json", "gpu-budget.json"))
    # A run often shares the ledger stored in its experiment's parent.
    for parent in root.parents:
        found.update(parent / name for name in ("budget.json", "gpu-budget.json")
                     if (parent / name).exists())
        if parent.name == "results" or (parent / ".git").exists():
            break
    return sorted(found)


@contextmanager
def _guard(root, excluded=(), lock=False):
    protected, handles = {}, []
    try:
        for ledger in _ledgers(root, excluded):
            try:
                _real(ledger)
                if ledger.stat().st_size > MAX_TEXT_BYTES:
                    raise ValueError("oversize ledger")
                data = json.loads(ledger.read_text(encoding="utf-8"))
                if data.get("active") or data.get("active_session"):
                    protected[ledger.parent] = "active experiment ledger"
                    continue
                if lock:
                    # ResearchBudget and directions.Allocation use different names.
                    for lock_path in sorted({ledger.with_suffix(".lock"), Path(str(ledger) + ".lock")}):
                        _real(lock_path)
                        handle = lock_path.open("a+")
                        handles.append(handle)
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    data = json.loads(ledger.read_text(encoding="utf-8"))
                    if data.get("active") or data.get("active_session"):
                        protected[ledger.parent] = "active experiment ledger"
            except (OSError, ValueError, TypeError, AttributeError):
                protected[ledger.parent] = "unreadable or locked experiment ledger"
        for parent, names, unused in _walk(root, excluded):
            if MARKERS.intersection(names):
                protected[parent] = "active or pending continuation marker"
        if lock and not any(_inside(root, path) for path in protected):
            # Serialize overlapping compact exports/deletions, including roots
            # without a GPU ledger. The results-level lock covers sibling runs.
            boundary = next((p for p in (root,) + tuple(root.parents)
                             if p.name == "results" or (p / ".git").exists()), root)
            scopes = [root] + [p for p in root.parents if _inside(p, boundary)]
            for directory in reversed(scopes):
                lock_path = _real(directory / ".compact-results.lock")
                handle = lock_path.open("a+")
                handles.append(handle)
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError("another compact export or deletion is active") from error
        yield protected
    finally:
        for handle in reversed(handles):
            handle.close()


def _plan(root, protected, destination=None, max_text_bytes=MAX_TEXT_BYTES):
    excluded = [p for p in protected if _inside(p, root)]
    if destination is not None:
        excluded.append(destination)
    entries, skipped = [], {str(p): reason for p, reason in protected.items()}
    if any(_inside(root, p) for p in protected) or any(part in PROTECTED_NAMES for part in root.parts):
        return dict(schema=SCHEMA, root=str(root), files=[], protected=skipped, status="protected")
    for parent, names, ignored in _walk(root, excluded):
        for path in ignored:
            skipped.setdefault(str(path), "source snapshot, archive destination, or symlink")
        for name in names:
            path = parent / name
            if path.is_symlink():
                skipped[str(path)] = "symlink (not followed)"
                continue
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or name.endswith(".lock") or name == INDEX_NAME:
                continue
            suffix = path.suffix.lower()
            kind = ("raw" if suffix in RAW_SUFFIXES else "text"
                    if suffix in TEXT_SUFFIXES and info.st_size <= max_text_bytes else "retained")
            entries.append(dict(path=str(path.relative_to(root)), bytes=info.st_size, kind=kind))
    return dict(schema=SCHEMA, root=str(root), files=entries, protected=skipped, status="planned")


def plan(root, *, destination=None, max_text_bytes=MAX_TEXT_BYTES):
    """Read-only inventory; large text and all unknown binaries remain protected."""
    root = _real(root)
    destination = _real(destination) if destination is not None else None
    with _guard(root, [destination] if destination else []) as protected:
        return _plan(root, protected, destination, max_text_bytes)


def _leaves(value, pointer=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaves(child, pointer + "/" + str(index))
    else:
        yield pointer or "/", type(value).__name__, str(value) if isinstance(value, str) else _json(value)


def _metrics(path, text, limit):
    try:
        if path.endswith(".json"):
            documents = [("", json.loads(text))]
        elif path.endswith(".jsonl"):
            documents = ((str(i), json.loads(line)) for i, line in enumerate(text.splitlines(), 1) if line.strip())
        else:
            return [], "text preserved in compressed payload"
        rows = []
        leaves = ((record, pointer, kind, value) for record, data in documents
                  for pointer, kind, value in _leaves(data))
        for index, (record, pointer, kind, value) in enumerate(leaves):
            if index >= limit:
                return rows, "additional fields preserved in compressed payload (Excel view limit)"
            # XML cannot contain arbitrary control characters. Encode only the
            # display; the complete original bytes live in data.json.gz.
            value = "".join(ch if ord(ch) >= 32 or ch in "\n\r\t" else "\\u%04x" % ord(ch) for ch in value)
            for part, start in enumerate(range(0, max(1, len(value)), 8000), 1):
                rows.append([path, record, pointer, kind, str(part), value[start:start + 8000]])
        return rows, "all scalar metrics shown; exact original text in compressed payload"
    except (ValueError, TypeError, RecursionError):
        return [], "unparsed text preserved exactly in compressed payload"


def _validate_summary(item, summary, root):
    """Bind an explicitly supplied valid aggregation to the exact source bytes."""
    if (Path(item["path"]).suffix.lower() not in {".json", ".jsonl"}
            or not isinstance(summary, dict) or summary.get("valid") is not True):
        raise ValueError("summarized text requires valid JSON/JSONL aggregation")
    metadata = summary.get("source_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("summary source metadata is required")
    size = metadata.get("size", metadata.get("bytes"))
    if (type(size) is not int or size != item["bytes"] or metadata.get("sha256") != item["sha256"]
            or ("bytes" in metadata and metadata["bytes"] != item["bytes"])):
        raise ValueError("summary source hash or size mismatch")
    if "source" in metadata and metadata["source"] != str(Path(root) / item["path"]):
        raise ValueError("summary source path mismatch")
    # Reject non-JSON values/NaN rather than silently changing the supplied data.
    _json(summary)


def _tables(data):
    summary = [["Field", "Value"], ["Schema", SCHEMA], ["Evidence", NOTICE], ["Source", data["root"]],
               ["Files", str(len(data["files"]))], ["Protected paths", str(len(data["protected"]))]]
    files = [["Path", "Bytes", "SHA256", "Kind", "Excel coverage"]]
    metrics = [["Source", "JSONL record", "JSON pointer", "Type", "Part", "Value"]]
    aggregated = sum(item["kind"] == "summarized_text" for item in data["files"])
    if aggregated:
        summary.extend([["Aggregated sources", str(aggregated)], ["Aggregation evidence", AGGREGATION_NOTICE]])
    for item in data["files"]:
        if item["kind"] == "summarized_text":
            rows, detail = _metrics("summary.json", _json(item["summary"]), data["metrics_limit"])
            for row in rows:
                row[0] = item["path"]
            coverage = AGGREGATION_NOTICE
            if "Excel view limit" in detail:
                coverage += " Excel view limit; remaining summary fields are in the compressed payload."
        else:
            rows, coverage = _metrics(item["path"], item["text"], data["metrics_limit"]) if "text" in item else (
                [], "original content not embedded; preserve source" if item["kind"] != "raw" else "raw artifact hash only")
        metrics.extend(rows)
        files.append([item["path"], str(item["bytes"]), item["sha256"], item["kind"], coverage])
    protected = [["Path", "Reason"]] + [[path, reason] for path, reason in sorted(data["protected"].items())]
    return [("Summary", summary), ("Files", files), ("Metrics", metrics), ("Protected", protected)]


def _sheets(data):
    for name, rows in _tables(data):
        rows = [[_xml_text(str(value)) for value in row] for row in rows]
        if len(rows) <= 100000:
            yield name, rows
        else:
            for index, start in enumerate(range(1, len(rows), 99999), 1):
                yield name + str(index), [rows[0]] + rows[start:start + 99999]


def _xml_text(value):
    return "".join(ch if ch in "\n\t" or 32 <= ord(ch) <= 0xD7FF
                   or 0xE000 <= ord(ch) <= 0xFFFD or 0x10000 <= ord(ch) <= 0x10FFFF
                   else "\\u%04x" % ord(ch) for ch in value)


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _xlsx(path, data):
    """Write literal inline strings: no formulas, hyperlinks, or truncation."""
    content = 'application/vnd.openxmlformats-officedocument.spreadsheetml.'
    sheets = list(_sheets(data))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        types = ET.Element("Types", xmlns="http://schemas.openxmlformats.org/package/2006/content-types")
        ET.SubElement(types, "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
        ET.SubElement(types, "Default", Extension="xml", ContentType="application/xml")
        ET.SubElement(types, "Override", PartName="/xl/workbook.xml", ContentType=content + "sheet.main+xml")
        relationships = "http://schemas.openxmlformats.org/package/2006/relationships"
        relroot = ET.Element("Relationships", xmlns=relationships)
        ET.SubElement(relroot, "Relationship", Id="rId1", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", Target="xl/workbook.xml")
        archive.writestr("_rels/.rels", ET.tostring(relroot))
        workbook = ET.Element("workbook", xmlns=NS, attrib={"xmlns:r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"})
        children = ET.SubElement(workbook, "sheets")
        rels = ET.Element("Relationships", xmlns=relationships)
        for index, (name, rows) in enumerate(sheets, 1):
            ET.SubElement(types, "Override", PartName="/xl/worksheets/sheet%d.xml" % index, ContentType=content + "worksheet+xml")
            ET.SubElement(children, "sheet", name=name, sheetId=str(index), attrib={"r:id": "rId%d" % index})
            ET.SubElement(rels, "Relationship", Id="rId%d" % index, Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet", Target="worksheets/sheet%d.xml" % index)
            sheet = ET.Element("worksheet", xmlns=NS)
            view = ET.SubElement(ET.SubElement(sheet, "sheetViews"), "sheetView", workbookViewId="0")
            ET.SubElement(view, "pane", ySplit="1", topLeftCell="A2", activePane="bottomLeft", state="frozen")
            content_node = ET.SubElement(sheet, "sheetData")
            for number, row in enumerate(rows, 1):
                row_node = ET.SubElement(content_node, "row", r=str(number))
                for column, value in enumerate(row):
                    value = str(value)
                    if len(value.encode("utf-16-le")) // 2 > 32767:
                        raise ValueError("Excel cell would exceed its limit")
                    cell = ET.SubElement(row_node, "c", r=chr(65 + column) + str(number), t="inlineStr")
                    node = ET.SubElement(ET.SubElement(cell, "is"), "t", attrib={"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
                    node.text = value
            archive.writestr("xl/worksheets/sheet%d.xml" % index, ET.tostring(sheet, encoding="utf-8", xml_declaration=True))
        archive.writestr("[Content_Types].xml", ET.tostring(types))
        archive.writestr("xl/workbook.xml", ET.tostring(workbook))
        archive.writestr("xl/_rels/workbook.xml.rels", ET.tostring(rels))


def export(root, destination, *, max_text_bytes=MAX_TEXT_BYTES, metrics_limit=3000,
           replace_existing=False, summarized_text=None):
    """Export/read back metadata and Excel without removing sources.

    ``summarized_text`` maps explicit relative JSON/JSONL paths to complete
    externally generated summaries. Each must have ``valid: true`` and matching
    ``source_metadata.sha256`` and ``source_metadata.size`` (or ``bytes``).
    Aggregation loses original record detail; that limitation is shown in Excel.
    """
    root, destination = _real(root), _real(destination)
    if not root.is_dir() or destination == root or _inside(root, destination):
        raise ValueError("destination must be separate from source root")
    if max_text_bytes < 0 or metrics_limit < 1:
        raise ValueError("invalid export limits")
    with _guard(root, [destination], lock=True) as protected:
        inventory = _plan(root, protected, destination, max_text_bytes)
        if inventory["status"] == "protected":
            return inventory
        if any(_inside(destination, p) for p in protected):
            raise ValueError("destination belongs to a protected experiment")
        if destination.exists():
            if not replace_existing:
                raise FileExistsError(str(destination))
            verify(destination)
            if (destination / RECEIPT_NAME).exists():
                raise ValueError("an archive with pruned evidence cannot be replaced")
            if {p.name for p in destination.iterdir()} != FILES:
                raise ValueError("existing archive has additional files")
        data = dict(inventory, notice=NOTICE, metrics_limit=metrics_limit)
        summaries = {} if summarized_text is None else summarized_text
        if not isinstance(summaries, dict):
            raise ValueError("summarized_text must map relative paths to summary dictionaries")
        if not set(summaries).issubset({entry["path"] for entry in data["files"]}):
            raise ValueError("summarized source is missing, protected, or not a canonical relative path")
        for entry in data["files"]:
            source = _real(root / entry["path"])
            has_summary = entry["path"] in summaries
            size, digest, raw = _read(source, collect=entry["kind"] == "text" and not has_summary, limit=max_text_bytes)
            if size != entry["bytes"]:
                raise ValueError("source size changed since planning")
            entry["sha256"] = digest
            if has_summary:
                _validate_summary(entry, summaries[entry["path"]], root)
                entry.update(kind="summarized_text", summary=json.loads(_json(summaries[entry["path"]])),
                             evidence_status="aggregated_only", original_recoverable=False)
            elif raw is not None:
                try:
                    entry["text"] = raw.decode("utf-8")
                except UnicodeDecodeError:
                    entry["kind"] = "retained"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".compact-export-", dir=destination.parent))
        try:
            with gzip.GzipFile(str(temporary / "data.json.gz"), "wb", mtime=0) as stream:
                stream.write(_json(data).encode("utf-8"))
            _xlsx(temporary / "results.xlsx", data)
            manifest = dict(schema=SCHEMA, root=str(root), notice=NOTICE, files=len(data["files"]),
                            payload_sha256=_read(temporary / "data.json.gz")[1],
                            workbook_sha256=_read(temporary / "results.xlsx")[1])
            (temporary / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
            result = verify(temporary)
            if destination.exists():
                # Verify first, then replace only known files. A partial interruption
                # fails verification and can never authorize pruning.
                for name in ("data.json.gz", "results.xlsx", "manifest.json"):
                    (temporary / name).replace(destination / name)
                temporary.rmdir()
            else:
                temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        result.update(workbook=str(destination / "results.xlsx"), data=str(destination / "data.json.gz"),
                      manifest=str(destination / "manifest.json"), protected=data["protected"])
        return result


def _load(destination):
    destination = _real(destination)
    manifest = json.loads(_real(destination / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported compact archive schema")
    for name, field in (("data.json.gz", "payload_sha256"), ("results.xlsx", "workbook_sha256")):
        if _read(_real(destination / name))[1] != manifest[field]:
            raise ValueError("archive hash mismatch: " + name)
    with gzip.open(destination / "data.json.gz", "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("schema") != SCHEMA or data["root"] != manifest["root"]:
        raise ValueError("archive identity mismatch")
    return data


def verify(destination, *, check_sources=False):
    """Validate embedded text and compare every Excel cell to rebuilt tables."""
    destination = _real(destination)
    data = _load(destination)
    seen = set()
    for item in data["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or item["path"] in seen:
            raise ValueError("unsafe or duplicate archive source path")
        seen.add(item["path"])
        if "text" in item:
            raw = item["text"].encode("utf-8")
            if len(raw) != item["bytes"] or _hash(raw) != item["sha256"]:
                raise ValueError("embedded source hash mismatch")
        if item["kind"] == "summarized_text":
            _validate_summary(item, item.get("summary"), data["root"])
            if item.get("evidence_status") != "aggregated_only" or item.get("original_recoverable") is not False:
                raise ValueError("summarized text must declare irrecoverable original records")
        if check_sources:
            source = _real(Path(data["root"]) / relative)
            size, digest, unused = _read(source)
            if size != item["bytes"] or digest != item["sha256"]:
                raise ValueError("source hash mismatch: " + str(source))
    with zipfile.ZipFile(destination / "results.xlsx") as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        names = [node.attrib["name"] for node in workbook.findall("{%s}sheets/{%s}sheet" % (NS, NS))]
        expected = list(_sheets(data))
        if names != [name for name, unused in expected]:
            raise ValueError("Excel sheets differ")
        for index, (unused, rows) in enumerate(expected, 1):
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet%d.xml" % index))
            actual = []
            for row in sheet.findall("{%s}sheetData/{%s}row" % (NS, NS)):
                values = []
                for cell in row:
                    if cell.attrib.get("t") != "inlineStr" or cell.find("{%s}f" % NS) is not None:
                        raise ValueError("Excel formulas or nonliteral cells are forbidden")
                    values.append(cell.find("{%s}is/{%s}t" % (NS, NS)).text or "")
                actual.append(values)
            if actual != rows:
                raise ValueError("Excel cells differ from embedded evidence")
    return dict(status="verified", files=len(data["files"]),
                raw_bytes=sum(item["bytes"] for item in data["files"] if item["kind"] == "raw"),
                summarized_text_bytes=sum(item["bytes"] for item in data["files"] if item["kind"] == "summarized_text"))


def prune(destination, paths):
    """Delete explicitly selected raw arrays or validated aggregated JSON/JSONL.

    Ordinary JSON/JSONL, reports, snapshots, links and unknown types are ineligible.
    Callers choose policy/key samples; an export alone does not select deletions.
    """
    destination = _real(destination)
    verify(destination)
    data = _load(destination)
    root = _real(data["root"])
    chosen = set(map(str, paths))
    entries = {item["path"]: item for item in data["files"]}
    if not chosen.issubset(entries):
        raise ValueError("selected file is absent from the verified archive")
    with _guard(root, [destination], lock=True) as protected:
        # Revalidate after taking the module lock; an earlier concurrent export
        # must not change which evidence this deletion refers to.
        verify(destination)
        if _load(destination) != data:
            raise ValueError("compact archive changed before deletion")
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        receipt_path = destination / RECEIPT_NAME
        if receipt_path.exists():
            receipt = _receipt(destination, manifest, allow_incomplete=True)
        else:
            receipt = dict(schema=SCHEMA, root=str(root), status="in_progress", deleted={},
                           payload_sha256=manifest["payload_sha256"], workbook_sha256=manifest["workbook_sha256"])
        for relative, item in receipt["deleted"].items():
            if (relative not in entries or item.get("kind") not in PRUNABLE_KINDS
                    or any(item.get(key) != entries[relative][key] for key in ("bytes", "sha256", "kind"))
                    or item.get("state", "deleted") not in ("delete_intent", "deleted")):
                raise ValueError("deletion receipt differs from archived evidence")
        pending = {path for path, item in receipt["deleted"].items() if item.get("state") == "delete_intent"}
        chosen.update(pending)
        current = _plan(root, protected, destination)["files"]
        eligible = {item["path"] for item in current
                    if item["kind"] == "raw" or (item["path"] in entries
                                                 and entries[item["path"]]["kind"] == "summarized_text")}
        previously_missing = {path for path in chosen if path in receipt["deleted"] and not (root / path).exists()}
        if (not (chosen - previously_missing).issubset(eligible)
                or any(_inside(root / path, p) for path in chosen for p in protected)):
            raise ValueError("selected file is protected or is not a raw array or validated summarized text")
        for item in data["files"]:
            if "text" in item:
                size, digest, unused = _read(_real(root / item["path"]))
                if size != item["bytes"] or digest != item["sha256"]:
                    raise ValueError("source metadata changed since export")
        # Verify all chosen files before the first unlink. Recheck each immediately
        # before deleting, with the experiment locks held across both passes.
        for relative in sorted(chosen):
            item = entries[relative]
            source = _real(root / relative)
            if relative in previously_missing:
                continue
            size, digest, unused = _read(source)
            if item["kind"] not in PRUNABLE_KINDS or size != item["bytes"] or digest != item["sha256"]:
                raise ValueError("selected source changed since export")
        # The locator is only a sidecar. Historical JSON/JSONL remain unchanged.
        locator_path = _real(root / INDEX_NAME)
        locator = json.loads(locator_path.read_text(encoding="utf-8")) if locator_path.exists() else {
            "schema": SCHEMA, "archives": []}
        if locator.get("schema") != SCHEMA:
            raise ValueError("unsupported compact archive locator")
        locator["archives"] = sorted(set(locator["archives"] + [str(destination)]))
        _write_json(locator_path, locator)
        receipt["status"] = "in_progress"
        _save_receipt(receipt_path, receipt)
        deleted = []
        for relative in sorted(chosen):
            source = _real(root / relative)
            if relative not in previously_missing:
                if _read(source)[1] != entries[relative]["sha256"]:
                    raise ValueError("selected source changed before deletion")
                receipt["deleted"][relative] = dict(
                    {key: entries[relative][key] for key in ("bytes", "sha256", "kind")}, state="delete_intent")
                if entries[relative]["kind"] == "summarized_text":
                    receipt["deleted"][relative].update(evidence_status="aggregated_only", original_recoverable=False)
                _save_receipt(receipt_path, receipt)
                source.unlink()
                deleted.append(relative)
            receipt["deleted"][relative]["state"] = "deleted"
            _save_receipt(receipt_path, receipt)
        receipt["status"] = "completed"
        _save_receipt(receipt_path, receipt)
    return dict(status="pruned", deleted=deleted, bytes=sum(entries[p]["bytes"] for p in deleted), notice=NOTICE)


def _save_receipt(path, receipt):
    receipt["sha256"] = _hash(_json({key: value for key, value in receipt.items() if key != "sha256"}).encode("utf-8"))
    _write_json(path, receipt)


def _receipt(destination, manifest, allow_incomplete=False):
    receipt = json.loads(_real(destination / RECEIPT_NAME).read_text(encoding="utf-8"))
    expected = _hash(_json({key: value for key, value in receipt.items() if key != "sha256"}).encode("utf-8"))
    if (receipt.get("sha256") != expected or receipt.get("schema") != SCHEMA
            or receipt.get("status") not in (("completed", "in_progress") if allow_incomplete else ("completed",))
            or receipt.get("root") != manifest["root"]
            or receipt.get("payload_sha256") != manifest["payload_sha256"]
            or receipt.get("workbook_sha256") != manifest["workbook_sha256"]):
        raise ValueError("invalid or incomplete compact deletion receipt")
    return receipt


def verify_compacted_reference(root, path, expected_hash):
    """Validate a declared missing artifact, never silently accept other loss.

    Returns ``retained_metrics_only`` with no raw availability. The complete
    workbook and compressed metadata are reverified before honoring a receipt.
    """
    root = _real(root)
    source = _real(Path(path) if Path(path).is_absolute() else root / path)
    if source.exists() or not expected_hash:
        raise ValueError("compact reference requires a missing path and expected hash")
    for parent in [root] + list(root.parents):
        locator = parent / INDEX_NAME
        if locator.exists():
            index = json.loads(_real(locator).read_text(encoding="utf-8"))
            if index.get("schema") != SCHEMA:
                raise ValueError("unsupported compact archive locator")
            for archive_path in index["archives"]:
                destination = _real(archive_path)
                manifest = json.loads(_real(destination / "manifest.json").read_text(encoding="utf-8"))
                archive_root = _real(manifest["root"])
                if not _inside(source, archive_root):
                    continue
                relative = str(source.relative_to(archive_root))
                receipt = _receipt(destination, manifest)
                item = receipt["deleted"].get(relative)
                if item is None:
                    continue
                verify(destination)
                entry = next((item for item in _load(destination)["files"] if item["path"] == relative), None)
                if (entry is None or entry["kind"] not in PRUNABLE_KINDS or item["kind"] != entry["kind"]
                        or item.get("state", "deleted") != "deleted"
                        or entry["sha256"] != expected_hash or item["sha256"] != expected_hash
                        or entry["bytes"] != item["bytes"]):
                    raise ValueError("compact reference hash mismatch")
                result = dict(status="retained_metrics_only", availability="retained_metrics_only", raw_available=False,
                            re_evaluable=False,
                            path=None, original_path=str(source), sha256=expected_hash, archive=str(destination), notice=NOTICE)
                if entry["kind"] == "summarized_text":
                    if item.get("evidence_status") != "aggregated_only" or item.get("original_recoverable") is not False:
                        raise ValueError("summarized deletion receipt lacks evidence limitation")
                    result.update(evidence_status="aggregated_only", original_recoverable=False, notice=AGGREGATION_NOTICE)
                return result
        if parent.name == "results" or (parent / ".git").exists():
            break
    raise FileNotFoundError("missing artifact has no verified compact deletion receipt: " + str(source))


def export_report(root, *, ledger=None):
    """Refresh the lightweight workbook; export failures never fail a report.

    Raw files are working inputs until a separate verified cleanup is possible.
    Import the optional workbook exporter only after accounting is settled.
    """
    root = Path(root).resolve()
    try:
        run_path = root / "run.json"
        metadata = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
        if isinstance(metadata, dict) and metadata.get("schema") == "ditserve.run.v1":
            from . import finalize
            return finalize(root, prune=False)
        policy = json.loads((CONFIGS / "result_storage.json").read_text(encoding="utf-8"))
        if not policy.get("auto_export_reports", True):
            return dict(status="disabled")
        directory = policy.get("export_directory", "excel")
        if not isinstance(directory, str) or directory in ("", ".", "..") or Path(directory).name != directory:
            raise ValueError("Excel export directory must be one directory name")
        ledgers = {Path(ledger).resolve()} if ledger is not None else set()
        for name, field in (("protocol.json", "ledger"), ("report.json", "budget_path")):
            path = root / name
            if path.is_file():
                metadata = json.loads(path.read_text(encoding="utf-8"))
                reference = metadata.get(field)
                if reference:
                    reference = Path(reference)
                    ledgers.add(reference if reference.is_absolute() else root / reference)
        for path in ledgers:
            if not path.exists():
                # A missing ledger may simply mean that GPU work never started.
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("active") or value.get("active_session"):
                return dict(status="protected", reason="GPU accounting is still active", ledger=str(path))
        return export(root, root / directory, replace_existing=True)
    except Exception as error:
        # Reporting must remain available with missing optional dependencies,
        # changing active evidence, or a previously finalized archive.
        warnings.warn("Excel export deferred for %s: %s" % (root, error), RuntimeWarning, stacklevel=2)
        return dict(status="deferred", reason=str(error))
