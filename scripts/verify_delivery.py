#!/usr/bin/env python3
"""Create or verify the source delivery manifest without external packages."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "DELIVERY_MANIFEST.json"
EXCLUDED = {".git", ".cache", ".migration", "__pycache__", "history", "results", "build", "dist"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Release preparation only; regenerate after reviewed source changes")
    args = parser.parse_args()
    if args.write:
        records = []
        for path in sorted(ROOT.rglob("*")):
            rel = path.relative_to(ROOT)
            if (path == MANIFEST or any(p in EXCLUDED or p.startswith(".venv") or p.endswith(".egg-info") for p in rel.parts)
                    or path.suffix in (".pyc", ".pyo") or not path.is_file()):
                continue
            if path.is_symlink():
                raise ValueError("source delivery contains a symlink: " + str(rel))
            records.append(dict(path=str(rel), bytes=path.stat().st_size, sha256=digest(path)))
        MANIFEST.write_text(json.dumps(dict(schema_version=1, files=records), indent=2) + "\n")
    value = json.loads(MANIFEST.read_text())
    for record in value["files"]:
        path = ROOT / record["path"]
        if (path.is_symlink() or ROOT not in path.resolve().parents or not path.is_file()
                or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]):
            raise ValueError("delivery mismatch: " + record["path"])
    print("Verified %d files (%d bytes)." % (len(value["files"]), sum(r["bytes"] for r in value["files"])))


if __name__ == "__main__":
    main()
