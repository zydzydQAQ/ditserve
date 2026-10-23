"""Fetch the pinned CUDA-12.6-compatible profiler, without installing packages."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

BASE = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
PACKAGE = "nsight-systems-2024.6.2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(BASE + "Packages.gz", timeout=60).read()
    matches = []
    for paragraph in gzip.decompress(raw).decode().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in paragraph.splitlines()
                      if ": " in line and not line.startswith(" "))
        if fields.get("Package") == PACKAGE:
            matches.append(fields)
    if len(matches) != 1:
        raise RuntimeError("ambiguous pinned package")
    fields = matches[0]
    name, size = Path(fields["Filename"]).name, int(fields["Size"])
    target = args.directory / name
    if target.exists():
        if hashlib.file_digest(target.open("rb"), "sha256").hexdigest() == fields["SHA256"]:
            print("already verified", target)
            return
        target.rename(target.with_suffix(".serial.partial"))
    parts = args.directory / "parts"
    parts.mkdir(exist_ok=True)
    url = BASE + fields["Filename"]

    def fetch(index):
        begin, end = size * index // 8, size * (index + 1) // 8 - 1
        path = parts / ("%02d.part" % index)
        expected = end - begin + 1
        serial = target.with_suffix(".serial.partial")
        if not path.exists() and serial.exists() and serial.stat().st_size >= end + 1:
            with serial.open("rb") as source, path.open("wb") as output:
                source.seek(begin)
                output.write(source.read(expected))
        for attempt in range(8):
            completed = path.stat().st_size if path.exists() else 0
            if completed == expected:
                return path
            if completed > expected:
                raise RuntimeError("oversized package segment")
            offset = begin + completed
            request = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (offset, end)})
            try:
                with urllib.request.urlopen(request, timeout=60) as source:
                    if source.status != 206 or source.headers.get("Content-Range") != "bytes %d-%d/%d" % (offset, end, size):
                        raise RuntimeError("range response mismatch")
                    with path.open("ab") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                if path.stat().st_size != expected:
                    raise RuntimeError("truncated package segment")
                print("segment", index, "complete", flush=True)
                return path
            except (OSError, RuntimeError) as error:
                print("segment", index, "retry", attempt + 1, str(error), flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(attempt + 1, 5))

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(executor.map(fetch, range(8)))
    with target.open("xb") as output:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    actual = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
    if actual != fields["SHA256"]:
        raise RuntimeError("official package SHA256 mismatch")
    metadata = dict(source=url, package=PACKAGE, version=fields["Version"], bytes=size,
                    sha256=actual, index_sha256=hashlib.sha256(raw).hexdigest())
    (args.directory / "package.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
