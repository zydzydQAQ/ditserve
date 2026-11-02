"""Verify a frozen complete model snapshot before loading any tensors.

The expected manifest digest must come from the experiment's frozen input
inventory. Generating a manifest from whatever happens to be installed is not
verification against an intended upstream revision.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def _fingerprint(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def verify_model_manifest(model_path, manifest_path, expected_sha256):
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("model manifest requires a frozen SHA256")
    root, manifest_file = Path(model_path).resolve(strict=True), Path(manifest_path).resolve(strict=True)
    if not root.is_dir() or manifest_file.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("invalid model root or oversized manifest")
    raw = manifest_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("model manifest SHA256 differs")
    manifest = json.loads(raw)
    if (manifest.get("schema") != "ditserve.model-manifest.v1"
            or not isinstance(manifest.get("revision"), str) or not manifest["revision"].strip()):
        raise ValueError("model manifest requires schema and revision")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("model manifest requires a complete file inventory")
    expected, total = {}, 0
    for row in rows:
        name = row.get("path", "")
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or ".." in relative.parts
                or str(relative) != name or "\\" in name or name in expected):
            raise ValueError("invalid or duplicate model file path")
        digest = row.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or type(row.get("bytes")) is not int or row["bytes"] < 0:
            raise ValueError("invalid model file size or SHA256")
        path = root / name
        if not path.resolve(strict=True).is_relative_to(root) or not path.is_file() or path.resolve() == manifest_file:
            raise ValueError("model file escapes snapshot or references the manifest")
        before = _fingerprint(path)
        if before[2] != row["bytes"]:
            raise ValueError("model file size differs: " + name)
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != digest or before != _fingerprint(path):
            raise ValueError("model file changed or SHA256 differs: " + name)
        expected[name] = before
        total += row["bytes"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.resolve() != manifest_file}
    if actual != set(expected):
        raise ValueError("model manifest does not cover the complete snapshot")
    if any(_fingerprint(root / name) != stat for name, stat in expected.items()):
        raise ValueError("model snapshot changed during verification")
    return {"model_integrity": "manifest_verified", "model_manifest_sha256": expected_sha256,
            "model_verified_revision": manifest["revision"], "model_verified_files": len(expected),
            "model_verified_bytes": total}
