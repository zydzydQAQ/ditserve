#!/usr/bin/env python3
"""Verify a downloaded model directory or single file against a DitServe manifest."""
import argparse
import hashlib
import json
import pathlib
import sys


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_entries(manifest: dict):
    files = manifest.get("files")
    if isinstance(files, list):
        for item in files:
            yield item["path"], int(item["bytes"]), item["sha256"]
    elif isinstance(files, dict):
        for rel, meta in files.items():
            yield rel, int(meta["bytes"]), meta["sha256"]
    elif "filename" in manifest and "sha256" in manifest:
        yield manifest["filename"], int(manifest.get("bytes", -1)), manifest["sha256"]
    else:
        raise ValueError("Unsupported manifest format")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="Path to manifest JSON")
    ap.add_argument("--root", required=True, help="Local directory containing downloaded model files")
    ap.add_argument("--size-only", action="store_true", help="Skip SHA256 hashing and verify file existence + byte size only")
    args = ap.parse_args()

    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    root = pathlib.Path(args.root)
    ok = 0
    errors = []

    for rel, expected_bytes, expected_sha in load_entries(manifest):
        target = root / rel
        if not target.is_file():
            errors.append(f"MISSING: {rel}")
            continue
        actual_bytes = target.stat().st_size
        if expected_bytes >= 0 and actual_bytes != expected_bytes:
            errors.append(f"SIZE MISMATCH: {rel} (got {actual_bytes}, expected {expected_bytes})")
            continue
        if not args.size_only:
            actual_sha = sha256_file(target)
            if actual_sha.lower() != expected_sha.lower():
                errors.append(f"SHA256 MISMATCH: {rel} (got {actual_sha}, expected {expected_sha})")
                continue
        ok += 1

    print(f"Verified {ok} files in {root} against {args.manifest} (errors={len(errors)})")
    for err in errors:
        print("  ERROR:", err, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
