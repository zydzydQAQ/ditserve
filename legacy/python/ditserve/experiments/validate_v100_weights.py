"""Verify a pinned Wan snapshot against the saved official Hub file manifest."""

from ..util import atomic_json
import argparse
import hashlib
import json
from pathlib import Path


def verify(snapshot, manifest):
    records = []
    for entry in manifest["files"]:
        path = snapshot / entry["file"]
        record = dict(entry, exists=path.is_file(), verified=False)
        if path.is_file():
            record["actual_bytes"] = path.stat().st_size
            if entry.get("sha256"):
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for data in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(data)
                record["actual_sha256"] = digest.hexdigest()
            record["verified"] = (record["actual_bytes"] == entry["bytes"] and
                                  (not entry.get("sha256") or record.get("actual_sha256") == entry["sha256"]))
        records.append(record)
        print(entry["file"], "verified" if record["verified"] else "FAILED", flush=True)
    return {"revision": manifest["revision"], "snapshot": str(snapshot.resolve()),
            "status": "verified" if all(r["verified"] for r in records) else "failed",
            "scope": "official LFS SHA256 and file lengths; small Git files checked by length only",
            "files": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(".cache/runs/v100/environment/model-files.json"))
    parser.add_argument("--output", type=Path, default=Path(".cache/runs/v100/environment/weights-integrity.json"))
    args = parser.parse_args()
    report = verify(args.snapshot, json.loads(args.manifest.read_text()))
    atomic_json(args.output, report)
    return 0 if report["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
