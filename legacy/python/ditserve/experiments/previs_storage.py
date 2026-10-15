"""Optional, read-only registration of externally stored raw NumPy artifacts.

An absent artifact-storage.json preserves the original suite containment rule.
This module never moves data, creates links, or writes an index. An enabled
index authorizes only individually listed stage/{arrays,raw}/.../*.npy links;
records, calibration, directories, and direct external paths remain forbidden.
The index's file SHA256 includes the NPY header, unlike measurement array hashes.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat


INDEX_NAME = "artifact-storage.json"
SCHEMA = "previs-artifact-storage.v1"
CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Resolution:
    path: Path
    logical_path: Path
    storage: object = None


def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _signature(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("artifact_storage_duplicate_json_key")
        result[key] = value
    return result


def _path(value, absolute=False):
    if (not isinstance(value, str) or not value or "\x00" in value or "\\" in value
            or any(part in (".", "..") for part in value.split("/"))):
        raise ValueError("artifact_storage_noncanonical_path")
    path = Path(value)
    if path.is_absolute() is not absolute or str(path) != value:
        raise ValueError("artifact_storage_noncanonical_path")
    return path


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_index(suite):
    """Validate index syntax/boundaries without scanning any registered arrays."""
    suite = Path(suite).resolve()
    path = suite / INDEX_NAME
    if not os.path.lexists(path):
        return None
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024 * 1024:
        raise ValueError("artifact_storage_index_must_be_regular_bounded_file")
    raw = path.read_bytes()
    if _signature(before) != _signature(path.lstat()):
        raise ValueError("artifact_storage_index_changed_during_read")
    value = json.loads(raw, object_pairs_hook=_pairs)
    if (not isinstance(value, dict)
            or set(value) != {"schema_version", "suite_root", "storage_root", "entries"}
            or value.get("schema_version") != SCHEMA or value.get("suite_root") != str(suite)):
        raise ValueError("artifact_storage_index_schema_or_suite_mismatch")
    storage = _path(value["storage_root"], absolute=True)
    if (not storage.is_dir() or storage.resolve() != storage
            or _within(storage, suite) or _within(suite, storage)):
        raise ValueError("artifact_storage_root_must_be_distinct_real_external_directory")
    entries = value["entries"]
    if not isinstance(entries, list) or not entries or len(entries) > 100000:
        raise ValueError("artifact_storage_entries_must_be_nonempty_bounded_list")
    by_logical, targets = {}, set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"logical_path", "target_path", "file_sha256", "size_bytes"}:
            raise ValueError("artifact_storage_entry_schema_mismatch")
        logical = _path(entry["logical_path"])
        target = _path(entry["target_path"])
        if (len(logical.parts) < 3 or logical.parts[1] not in ("arrays", "raw")
                or logical.suffix != ".npy" or target.suffix != ".npy"):
            raise ValueError("artifact_storage_only_stage_raw_npy_allowed")
        if (not _sha(entry["file_sha256"]) or type(entry["size_bytes"]) is not int
                or entry["size_bytes"] <= 0):
            raise ValueError("artifact_storage_invalid_file_digest_or_size")
        if str(logical) in by_logical or str(target) in targets:
            raise ValueError("artifact_storage_duplicate_logical_or_target_path")
        by_logical[str(logical)] = dict(entry)
        targets.add(str(target))
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), storage_root=storage,
                entries=by_logical)


def _file_digest(path, expected_size, expected_sha):
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
        raise ValueError("artifact_storage_target_size_or_type_mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if _signature(os.fstat(handle.fileno())) != _signature(before):
            raise ValueError("artifact_storage_target_changed_before_read")
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
        if _signature(os.fstat(handle.fileno())) != _signature(before):
            raise ValueError("artifact_storage_target_changed_during_read")
    if _signature(path.stat()) != _signature(before):
        raise ValueError("artifact_storage_target_changed_during_read")
    if digest.hexdigest() != expected_sha:
        raise ValueError("artifact_storage_target_file_sha256_mismatch")


def _registered(logical, suite, index):
    # Lexical identity is checked before resolving the final, explicitly allowed
    # link. Neither directory links nor an unlisted alias inherit authorization.
    relative = logical.relative_to(suite)
    _path(relative.as_posix())
    entry = index["entries"].get(relative.as_posix())
    if entry is None:
        raise ValueError("artifact_storage_unregistered_external_path")
    if logical.parent.resolve() != logical.parent or not logical.is_symlink():
        raise ValueError("artifact_storage_requires_individual_npy_symlink")
    record = suite / relative.parts[0] / "records.jsonl"
    if not record.is_file() or record.is_symlink() or record.resolve() != record:
        raise ValueError("artifact_storage_stage_records_must_remain_inside_suite")
    target = index["storage_root"] / entry["target_path"]
    if target.resolve() != target or not _within(target, index["storage_root"]):
        raise ValueError("artifact_storage_target_symlink_or_escape")
    if _path(os.readlink(logical), absolute=True) != target or logical.resolve() != target:
        raise ValueError("artifact_storage_symlink_target_mismatch")
    link_before = logical.lstat()
    _file_digest(target, entry["size_bytes"], entry["file_sha256"])
    if (_signature(logical.lstat()) != _signature(link_before) or logical.resolve() != target
            or target.resolve() != target):
        raise ValueError("artifact_storage_link_changed_during_read")
    metadata = dict(index_path=index["path"], index_sha256=index["sha256"],
                    storage_root=str(index["storage_root"]), **entry)
    return Resolution(target, logical, metadata)


def resolve_candidates(choices, suite, *, outside_error="artifact_path_outside_suite"):
    """Resolve legacy candidates, optionally admitting explicitly indexed links.

    Direct external paths are never admitted, even if their file is registered.
    The full file is hashed each time: filesystem timestamps can be too coarse
    to detect same-size edits. Original array payload validation remains needed.
    """
    suite = Path(suite).resolve()
    index = load_index(suite)
    allowed = []
    for choice in dict.fromkeys(Path(item) for item in choices):
        choice = choice.absolute()
        physical = choice.resolve()
        lexical_inside = _within(choice, suite)
        entry = index["entries"].get(choice.relative_to(suite).as_posix()) if index and lexical_inside else None
        if index and lexical_inside and (entry is not None or not _within(physical, suite)):
            allowed.append(_registered(choice, suite, index))
        elif _within(physical, suite):
            allowed.append(Resolution(physical, physical))
    if not allowed:
        raise ValueError(outside_error)
    existing = [item for item in allowed if item.path.exists()]
    if len({(item.path.stat().st_dev, item.path.stat().st_ino) for item in existing}) > 1:
        raise ValueError("ambiguous_relative_artifact_path")
    return existing[0] if existing else allowed[0]


def index_metadata(suite):
    """Small provenance summary for derived reports; never hashes raw arrays."""
    index = load_index(suite)
    if index is None:
        return {"enabled": False}
    return dict(enabled=True, path=index["path"], sha256=index["sha256"],
                storage_root=str(index["storage_root"]), registered_files=len(index["entries"]),
                hash_semantics="SHA256 of complete NPY file including header; array payload hashes remain independently verified")
