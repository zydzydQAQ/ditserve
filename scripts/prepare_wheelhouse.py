#!/usr/bin/env python3
"""Prepare a hash-verified wheelhouse using concurrent HTTP byte ranges.

Only the Python standard library is required. Existing artifacts are reused by
exact filename and SHA256, without changing their source. Interrupted downloads
are never considered complete. Proxy settings are honored unless --direct is set.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid(path: Path, artifact: dict) -> bool:
    return (path.is_file()
            and (not artifact.get("bytes") or path.stat().st_size == artifact["bytes"])
            and sha256(path) == artifact["sha256"])


def load_manifests(paths: list[Path]) -> list[dict]:
    artifacts = {}
    for path in paths:
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1:
            raise ValueError(f"unsupported manifest schema: {path}")
        for source in data["artifacts"]:
            item = dict(source)
            filename = item["filename"]
            if not filename or Path(filename).name != filename or "\\" in filename:
                raise ValueError(f"unsafe artifact filename: {filename!r}")
            if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise ValueError(f"invalid SHA256 for {filename}")
            if not item["url"].startswith("https://"):
                raise ValueError(f"artifact URL must use HTTPS: {filename}")
            if item.get("bytes") is not None and (
                type(item["bytes"]) is not int or item["bytes"] <= 0
            ):
                raise ValueError(f"invalid byte count: {filename}")
            previous = artifacts.get(filename)
            if previous and (previous["sha256"] != item["sha256"] or
                             previous.get("bytes", item.get("bytes")) !=
                             item.get("bytes", previous.get("bytes"))):
                raise ValueError(f"conflicting manifests for {filename}")
            artifacts[filename] = item
    return list(artifacts.values())


class Fetcher:
    def __init__(self, direct: bool, retries: int, timeout: float):
        self.direct, self.retries, self.timeout = direct, retries, timeout
        self.local = threading.local()
        self.lock = threading.Lock()
        self.transferred = 0

    def open(self, url: str, *, method: str = "GET", headers: dict | None = None):
        if not hasattr(self.local, "opener"):
            handlers = [urllib.request.ProxyHandler({})] if self.direct else []
            self.local.opener = urllib.request.build_opener(*handlers)
        request_headers = {"User-Agent": "ditserve-wheelhouse/1", "Accept-Encoding": "identity"}
        request_headers.update(headers or {})
        request = urllib.request.Request(url, method=method, headers=request_headers)
        return self.local.opener.open(request, timeout=self.timeout)

    def size(self, item: dict) -> int:
        if item.get("bytes"):
            return item["bytes"]
        error = None
        for attempt in range(self.retries):
            try:
                with self.open(item["url"], method="HEAD") as response:
                    size = int(response.headers.get("Content-Length", "0"))
                    if response.status != 200 or size <= 0:
                        raise ValueError("HEAD did not provide a positive Content-Length")
                    return size
            except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"cannot determine size of {item['filename']}: {error}")

    def chunk(self, item: dict, path: Path, start: int, end: int) -> int:
        error = None
        length = end - start + 1
        for attempt in range(self.retries):
            try:
                with self.open(item["url"], headers={"Range": f"bytes={start}-{end}"}) as response:
                    expected_range = f"bytes {start}-{end}/{item['bytes']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError(f"invalid range response: status={response.status}, "
                                         f"Content-Range={response.headers.get('Content-Range')!r}, "
                                         f"expected={expected_range!r}")
                    advertised = response.headers.get("Content-Length")
                    if advertised is not None and int(advertised) != length:
                        raise ValueError("range Content-Length mismatch")
                    remaining = length
                    with path.open("r+b", buffering=0) as output:
                        output.seek(start)
                        while remaining:
                            block = response.read(min(1024 * 1024, remaining))
                            if not block:
                                raise OSError(f"truncated byte range ({remaining} bytes missing)")
                            output.write(block)
                            remaining -= len(block)
                            with self.lock:
                                self.transferred += len(block)
                        if response.read(1):
                            raise ValueError("byte range contains excess data")
                return length
            except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"{item['filename']} bytes {start}-{end}: {error}")


def reuse(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    fd, staging = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".reuse", dir=target.parent)
    os.close(fd)
    stage = Path(staging)
    try:
        stage.unlink()
        try:
            os.link(source, stage)
        except OSError:
            shutil.copyfile(source, stage)
        os.replace(stage, target)
    finally:
        stage.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-dir", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--direct", action="store_true", help="bypass HTTP(S) proxy environment variables")
    parser.add_argument("--dry-run", action="store_true", help="verify reusable files and print missing bytes without writing")
    args = parser.parse_args(argv)
    if min(args.workers, args.chunk_mib, args.retries, args.timeout) <= 0:
        parser.error("worker, chunk, retry and timeout values must be positive")
    artifacts = load_manifests(args.manifests)
    fetcher = Fetcher(args.direct, args.retries, args.timeout)
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, list[Path]] = {}
    names = {item["filename"] for item in artifacts}
    for directory in [args.output, *args.reuse_dir]:
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.name in names and path.is_file():
                    candidates.setdefault(path.name, []).append(path)
    missing = []
    reused_bytes = 0
    for item in artifacts:
        source = next((path for path in candidates.get(item["filename"], []) if valid(path, item)), None)
        if source:
            item["bytes"] = source.stat().st_size
            reused_bytes += item["bytes"]
            print(f"REUSE {item['filename']} ({item['bytes']} bytes)", flush=True)
            if not args.dry_run:
                reuse(source, args.output / item["filename"])
        else:
            missing.append(item)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        sizes = {pool.submit(fetcher.size, item): item for item in missing}
        for future in concurrent.futures.as_completed(sizes):
            sizes[future]["bytes"] = future.result()
    total = sum(item["bytes"] for item in missing)
    print(f"PLAN artifacts={len(artifacts)} missing={len(missing)} missing_bytes={total} "
          f"reused_bytes={reused_bytes} workers={args.workers}", flush=True)
    if args.dry_run or not missing:
        return 0
    available = shutil.disk_usage(args.output).free
    if available < total:
        raise RuntimeError(f"not enough free disk space: need {total}, available {available} bytes")
    chunk_bytes = args.chunk_mib * 1024 * 1024
    started = time.monotonic()
    completed_bytes = 0
    pending = {}
    states = {}
    # Each run owns its temporary directory; no old partial is trusted or overwritten.
    with tempfile.TemporaryDirectory(prefix=".download-", dir=args.output) as temporary:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for item in sorted(missing, key=lambda entry: entry["bytes"]):
                path = Path(temporary) / item["filename"]
                with path.open("wb") as output:
                    output.truncate(item["bytes"])
                count = (item["bytes"] + chunk_bytes - 1) // chunk_bytes
                states[item["filename"]] = [item, path, count, []]
                for start in range(0, item["bytes"], chunk_bytes):
                    future = pool.submit(fetcher.chunk, item, path, start, min(start + chunk_bytes, item["bytes"]) - 1)
                    pending[future] = item["filename"]
            last_report = started
            failed = []
            while pending:
                done, _ = concurrent.futures.wait(pending, timeout=10,
                                                  return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    name = pending.pop(future)
                    state = states[name]
                    try:
                        completed_bytes += future.result()
                    except Exception as exc:
                        state[3].append(str(exc))
                    state[2] -= 1
                    if state[2] == 0:
                        item, path, _, errors = state
                        if not errors and not valid(path, item):
                            errors.append("final size or SHA256 mismatch")
                        if errors:
                            failed.append(name)
                            print(f"FAIL {name}: {'; '.join(errors)}", file=sys.stderr, flush=True)
                        else:
                            os.replace(path, args.output / name)
                            print(f"READY {name}", flush=True)
                now = time.monotonic()
                if now - last_report >= 15 or not pending:
                    elapsed = now - started
                    speed = fetcher.transferred / max(elapsed, .001)
                    eta = (total - completed_bytes) / max(speed, 1)
                    print(f"PROGRESS verified_ranges_bytes={completed_bytes}/{total} "
                          f"transferred_bytes={fetcher.transferred} "
                          f"rate_mib_s={speed / 1048576:.2f} eta_seconds={eta:.0f} "
                          f"elapsed_seconds={elapsed:.0f}", flush=True)
                    last_report = now
            if failed:
                print(f"FAILED artifacts={len(failed)}; rerun to retry; completed artifacts remain verified", file=sys.stderr)
                return 1
    print(f"COMPLETE artifacts={len(artifacts)} bytes={total + reused_bytes} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
