#!/usr/bin/env python3
"""Relocate downloaded stage evidence using explicit remote-to-local prefixes.

Only reference paths change. Original source hashes and recomputed gate results
must agree before a new manifest and receipt are written. No files are fetched.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath

from stage_gates import evaluate, read_json


class PrefixMap:
    def __init__(self, values):
        self.roots = {}
        for value in values:
            remote, separator, local = value.partition("=")
            prefix = PurePosixPath(remote)
            if (not separator or not local or not prefix.is_absolute() or ".." in prefix.parts
                    or "\\" in remote or str(prefix) != remote or prefix in self.roots):
                raise ValueError("expected unique absolute REMOTE_PREFIX=LOCAL_DIRECTORY")
            root = Path(local).resolve(strict=True)
            if not root.is_dir():
                raise ValueError("local prefix must be an existing directory")
            self.roots[prefix] = root
        if not self.roots:
            raise ValueError("at least one explicit prefix mapping is required")

    def local(self, remote):
        for prefix in sorted(self.roots, key=lambda path: len(path.parts), reverse=True):
            if remote.is_relative_to(prefix):
                root = self.roots[prefix]
                path = (root / str(remote.relative_to(prefix))).resolve(strict=True)
                if not path.is_relative_to(root):
                    raise ValueError("mapped source escapes its local mirror directory")
                return path
        raise ValueError("no explicit prefix mapping for source: " + str(remote))


def remote_path(base, value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("source path must be a nonempty POSIX path")
    return PurePosixPath(posixpath.normpath(str(base / value)))


def verified_source(mapping, base, reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("source reference requires path and sha256 only")
    remote = remote_path(base, reference["path"])
    path = mapping.local(remote)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("original source SHA256 differs: " + str(remote))
    return remote, path, raw


def relocate(receipt_path, original_location, mappings, output_dir):
    receipt_path = Path(receipt_path).resolve(strict=True)
    output_dir = Path(output_dir).absolute()
    if output_dir.exists():
        raise FileExistsError("relocation requires a new output directory: " + str(output_dir))
    origin = PurePosixPath(original_location)
    if (not origin.is_absolute() or str(origin) != original_location or ".." in origin.parts
            or "\\" in original_location or not origin.name):
        raise ValueError("original receipt location must be an absolute POSIX file path")
    mapping = PrefixMap(mappings)
    original_raw = receipt_path.read_bytes()
    original = read_json(original_raw)
    if original.get("schema") != "ditserve.stage-gates.v1":
        raise ValueError("invalid original stage receipt")
    remote_manifest, _, manifest_raw = verified_source(mapping, origin.parent, original["source_manifest"])
    manifest = read_json(manifest_raw)
    # Resolve every reference against the original manifest, preserving each
    # original file digest. Paths inside measured run records are not edited.
    references = [manifest["protocol"], *manifest["contract"].values(), *manifest.get("runs", [])]
    for reference in references:
        _, path, _ = verified_source(mapping, remote_manifest.parent,
                                     {key: reference[key] for key in ("path", "sha256")})
        reference["path"] = str(path)
    result = evaluate(manifest, output_dir)
    if {key: original.get(key) for key in result} != result:
        raise ValueError("relocated gate results differ from the original receipt")
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    manifest_path = output_dir / "stage-evidence.json"
    result["source_manifest"] = {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    result["relocated_from"] = {"path": str(receipt_path), "original_location": str(origin),
                                "sha256": hashlib.sha256(original_raw).hexdigest(),
                                "source_manifest_sha256": original["source_manifest"]["sha256"]}
    receipt_bytes = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    output_dir.mkdir(parents=True, exist_ok=False)
    with manifest_path.open("xb") as stream:
        stream.write(manifest_bytes)
    receipt_output = output_dir / "stage-receipt.json"
    with receipt_output.open("xb") as stream:
        stream.write(receipt_bytes)
    return receipt_output, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path, help="downloaded original receipt")
    parser.add_argument("--original-location", required=True, help="original absolute container receipt path")
    parser.add_argument("--map", required=True, action="append", dest="mappings", metavar="REMOTE_PREFIX=LOCAL_DIRECTORY")
    parser.add_argument("--output-dir", required=True, type=Path, help="new local directory; existing output is never replaced")
    args = parser.parse_args()
    receipt, result = relocate(args.receipt, args.original_location, args.mappings, args.output_dir)
    print(json.dumps({"receipt": str(receipt), "stages": {
        name: row["status"] for name, row in result["stages"].items()}}))


if __name__ == "__main__":
    main()
