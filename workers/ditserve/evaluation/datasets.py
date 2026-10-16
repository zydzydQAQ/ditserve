"""Prompt metadata only: local imports and opt-in, revision-pinned downloads.

VBench retains its complete official evaluation list. VidProM is read row by
row and reduced to a bounded, deterministic hash sample; no video is fetched.
"""

import csv
import hashlib
import heapq
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from .common import atomic_json, canonical_hash, file_hash, read_json, seal


SCHEMA = "ditserve.benchmark.dataset.v1"
OFFICIAL = {
    "vbench": {
        "repository": "https://github.com/Vchitect/VBench",
        "metadata_path": "vbench/VBench_full_info.json",
        "default_ref": "1039fe66d7aa2f4b214003ca2d35f5011c48bfee",
        "license_notice": "See the license and benchmark protocol in Vchitect/VBench.",
    },
    "vidprom": {
        "repository": "https://huggingface.co/datasets/WenhaoWang/VidProM",
        "metadata_path": "VidProM_unique.csv",
        "default_ref": "main",
        "license_notice": "See the WenhaoWang/VidProM dataset card and original data terms.",
    },
}


def _json_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ditserve-benchmark/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _official_file_metadata(kind, resolved):
    """Read a file's expected identity from its immutable repository revision."""
    metadata_path = OFFICIAL[kind]["metadata_path"]
    if kind == "vbench":
        url = ("https://api.github.com/repos/Vchitect/VBench/contents/" + metadata_path +
               "?ref=" + resolved)
        entry = _json_url(url)
        if not isinstance(entry, dict) or entry.get("type") != "file":
            raise ValueError("official VBench file metadata is missing")
        algorithm, digest, size = "git_blob_sha1", entry.get("sha"), entry.get("size")
    else:
        url = "https://huggingface.co/api/datasets/WenhaoWang/VidProM/tree/" + resolved + "?recursive=false"
        entries = _json_url(url)
        if not isinstance(entries, list):
            raise ValueError("official VidProM repository file metadata is missing")
        matching = [entry for entry in entries if isinstance(entry, dict) and entry.get("path") == metadata_path]
        if len(matching) != 1 or matching[0].get("type") != "file":
            raise ValueError("official VidProM CSV metadata is missing or ambiguous")
        entry = matching[0]
        lfs = entry.get("lfs")
        if not isinstance(lfs, dict) or lfs.get("size") != entry.get("size"):
            raise ValueError("official VidProM CSV requires matching LFS size and SHA256 metadata")
        algorithm, digest, size = "sha256", lfs.get("oid"), entry.get("size")
    width = 64 if algorithm == "sha256" else 40
    if type(size) is not int or size <= 0 or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{%d}" % width, digest):
        raise ValueError("official file metadata requires a positive size and complete content hash")
    return {"metadata_url": url, "expected_bytes": size,
            "hash_algorithm": algorithm, "expected_hash": digest.lower()}


def _file_integrity(path, expected):
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(("blob %d\0" % size).encode("ascii")) if expected["hash_algorithm"] == "git_blob_sha1" else None
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
            if blob is not None:
                blob.update(chunk)
    result = {"bytes": size, "sha256": sha256.hexdigest()}
    if blob is not None:
        result["git_blob_sha1"] = blob.hexdigest()
    return result


def _check_integrity(actual, expected):
    if actual["bytes"] != expected["expected_bytes"]:
        raise ValueError("metadata download size mismatch: expected %d bytes, received %d" %
                         (expected["expected_bytes"], actual["bytes"]))
    if actual[expected["hash_algorithm"]] != expected["expected_hash"]:
        raise ValueError("metadata download %s mismatch" % expected["hash_algorithm"])


def _temporary_file(target, suffix):
    descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=suffix, dir=str(target.parent))
    os.close(descriptor)
    return Path(name)


def _record_failed_file(path, expected, reason, actual=None, content_length=None):
    atomic_json(str(path) + ".error.json", {"schema_version": "ditserve.benchmark.download_failure.v1",
                                           "path": str(path.resolve()), "status": "untrusted",
                                           "reason": str(reason), "expected": expected,
                                           "actual": actual or _file_integrity(path, expected),
                                           "content_length": content_length})


def _download(kind, revision, cache_dir):
    """Verify both new downloads and cache hits against official file metadata.

    Failed downloads and stale cache files are preserved under unique
    .partial/.corrupt names with an error receipt. They are never parsed as data.
    """
    info = OFFICIAL[kind]
    requested = revision or info["default_ref"]
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("revision must be a nonempty string")
    resolved = requested
    if not re.fullmatch(r"[0-9a-fA-F]{40}", requested):
        ref = urllib.parse.quote(requested, safe="")
        api = ("https://api.github.com/repos/Vchitect/VBench/commits/" + ref
               if kind == "vbench" else
               "https://huggingface.co/api/datasets/WenhaoWang/VidProM/revision/" + ref)
        resolved = _json_url(api).get("sha")
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", resolved):
        raise ValueError("official repository did not resolve to an immutable commit SHA")
    resolved = resolved.lower()
    url = ("https://raw.githubusercontent.com/Vchitect/VBench/" + resolved + "/" + info["metadata_path"]
           if kind == "vbench" else
           info["repository"] + "/resolve/" + resolved + "/" + info["metadata_path"])
    root = Path(cache_dir) if cache_dir is not None else Path(".cache/benchmark/datasets")
    target = root / kind / resolved / Path(info["metadata_path"]).name
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = _official_file_metadata(kind, resolved)
    if target.is_file():
        actual = _file_integrity(target, expected)
        try:
            _check_integrity(actual, expected)
        except ValueError as error:
            corrupt = _temporary_file(target, ".corrupt")
            target.replace(corrupt)
            _record_failed_file(corrupt, expected, error, actual=actual)
    if not target.is_file():
        temporary = _temporary_file(target, ".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "ditserve-benchmark/1", "Accept-Encoding": "identity"})
        content_length = None
        actual = None
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                length_header = getattr(response, "headers", {}).get("Content-Length")
                if length_header is not None:
                    if not re.fullmatch(r"[0-9]+", str(length_header).strip()):
                        raise ValueError("invalid HTTP Content-Length for metadata download")
                    content_length = int(length_header)
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
                    if handle.tell() > expected["expected_bytes"]:
                        raise ValueError("metadata download exceeded the official file size")
            actual = _file_integrity(temporary, expected)
            if content_length is not None and actual["bytes"] != content_length:
                raise ValueError("metadata download Content-Length mismatch: expected %d bytes, received %d" %
                                 (content_length, actual["bytes"]))
            _check_integrity(actual, expected)
            temporary.replace(target)
        except Exception as error:
            _record_failed_file(temporary, expected, error, actual=actual, content_length=content_length)
            raise
    return target, {"requested_revision": requested, "revision": resolved, "url": url,
                    "downloaded_metadata_only": True, "integrity": dict(expected, status="verified")}


def _prompt(value, row):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty or non-text prompt at row %d" % row)
    return value


def _vbench(path):
    with path.open(encoding="utf-8-sig") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError("VBench metadata must be a nonempty JSON array")
    result = []
    for index, row in enumerate(rows):
        if isinstance(row, str):
            row = {"prompt_en": row}
        if not isinstance(row, dict):
            raise ValueError("invalid VBench metadata row")
        prompt = _prompt(row.get("prompt_en", row.get("prompt")), index + 1)
        result.append({"sample_id": "vbench-%06d" % index, "prompt": prompt,
                       "split": "evaluation", "dimensions": row.get("dimension", []), "metadata": row})
    return result, len(rows)


def _vidprom(path, seed, sample_count):
    """Keep the lowest stable hash priorities without loading the full CSV."""
    heap = []
    rows_seen = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "prompt" not in reader.fieldnames:
            raise ValueError("VidProM CSV requires a prompt column")
        for index, row in enumerate(reader):
            prompt = _prompt(row.get("prompt"), index + 2)
            identity = row.get("uuid") or row.get("sample_id") or row.get("id") or "row-%09d" % index
            sample = {"sample_id": "vidprom-" + identity, "prompt": prompt}
            timestamp = row.get("time") or row.get("timestamp")
            if timestamp:
                sample["timestamp"] = timestamp
            priority = int(canonical_hash({"seed": seed, "id": identity, "prompt": prompt}), 16)
            entry = (-priority, -index, sample)
            if len(heap) < sample_count:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
            rows_seen += 1
    if not heap:
        raise ValueError("VidProM CSV contains no prompts")
    # Preserve source ordering so collection timestamps and row identity survive.
    return [item[2] for item in sorted(heap, key=lambda item: -item[1])], rows_seen


def _assign_roles(samples, kind, dev_count, seed):
    # Group exact prompt duplicates to prevent a prompt crossing dev/test roles.
    keys = {canonical_hash(sample["prompt"]) for sample in samples}
    if len(keys) < 2 and dev_count:
        raise ValueError("at least two distinct prompts are needed for disjoint dev/test roles")
    if dev_count >= len(keys):
        raise ValueError("dev_count must leave at least one distinct test prompt")
    ordered = sorted(keys, key=lambda key: canonical_hash({"seed": seed, "prompt_hash": key}))
    dev = set(ordered[:dev_count])
    for sample in samples:
        role = "dev" if canonical_hash(sample["prompt"]) in dev else "test"
        if kind == "vbench":
            sample["service_split"] = role
        else:
            sample["split"] = role


def prepare_dataset(kind, source=None, revision=None, dev_count=32, seed=20261012,
                    download=False, cache_dir=None, sample_count=4096, output_path=None):
    """Return a sealed prompt pool; downloading requires explicit opt-in.

    ``sample_count`` bounds VidProM memory, and is ignored for VBench because
    official quality evaluation must retain every entry. ``dev_count`` counts
    distinct prompts; duplicate rows always receive the same service role.
    """
    if kind not in OFFICIAL:
        raise ValueError("dataset kind must be vbench or vidprom")
    if type(download) is not bool:
        raise ValueError("download must be an explicit boolean")
    if type(seed) is not int or type(dev_count) is not int or dev_count < 0:
        raise ValueError("seed must be an integer and dev_count a nonnegative integer")
    if type(sample_count) is not int or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if source is not None and download:
        raise ValueError("choose a local source or an official download, not both")
    if source is None:
        if not download:
            raise ValueError("provide local prompt metadata or explicitly enable download")
        path, origin = _download(kind, revision, cache_dir)
    else:
        path = Path(source)
        if not path.is_file():
            raise ValueError("prompt metadata file does not exist: %s" % path)
        origin = {"revision": revision, "local_path": str(path.resolve()),
                  "downloaded_metadata_only": False}
    origin.update(OFFICIAL[kind])
    origin["sha256"] = file_hash(path)
    origin["bytes"] = path.stat().st_size
    samples, rows_seen = _vbench(path) if kind == "vbench" else _vidprom(path, seed, sample_count)
    if file_hash(path) != origin["sha256"] or path.stat().st_size != origin["bytes"]:
        raise ValueError("dataset source changed while preparing frozen metadata")
    ids = [sample["sample_id"] for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs in selected dataset")
    _assign_roles(samples, kind, dev_count, seed)
    result = seal({"schema_version": SCHEMA, "kind": kind, "source": origin, "samples": samples,
                 "selection": {"seed": seed, "rows_seen": rows_seen, "samples_retained": len(samples),
                               "method": "full_official_list" if kind == "vbench" else "lowest_sha256_priority",
                               "sample_count_limit": None if kind == "vbench" else sample_count,
                               "dev_distinct_prompts": dev_count,
                               "prompt_transform": "none; original whitespace preserved; whitespace-only prompts rejected"},
                 "split_protocol": {
                     "service_roles": "Deterministic disjoint prompt groups; no prompt crosses dev/test.",
                     "quality_evaluation": ("All official entries remain split=evaluation. service_split is a separate serving partition, not an official VBench split. Full-set scores are not held-out if its service dev prompts were used for quality tuning."
                                            if kind == "vbench" else "Service dev/test partition only; this prompt pool is not a quality benchmark."),
                     "timestamp_semantics": "Collection timestamps, when present, are not production service request arrival logs."}})
    if output_path is not None:
        target = Path(output_path)
        if target.resolve() == path.resolve() or (target.exists() and read_json(target) != result):
            raise ValueError("cannot overwrite source metadata or frozen dataset")
        atomic_json(target, result)
    return result
