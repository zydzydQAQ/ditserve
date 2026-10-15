"""Small, dependency-free provenance and persistence helpers."""

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from pathlib import Path


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal(payload):
    result = copy.deepcopy(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_hash(result)
    return result


def verify_seal(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("sha256"), str):
        raise ValueError("missing sealed object sha256")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if canonical_hash(body) != payload["sha256"]:
        raise ValueError("sealed object hash mismatch")
    return payload


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value):
    raise ValueError("nonfinite JSON value: " + value)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=_reject_constant)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(name + " must be finite numeric")
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(name + " must be " + ("nonnegative" if allow_zero else "positive"))
    return float(value)


def environment_info():
    versions = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "cache-dit", "numpy", "openpyxl", "vbench", "lpips", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                 "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "runtime_environment": {key: os.environ.get(key) for key in variables}, "packages": versions}


def source_identity():
    from ..util import ROOT
    package = Path(__file__).resolve().parent
    files = {str(path.relative_to(package.parent)): file_hash(path)
             for path in sorted(package.rglob("*.py"))}
    for name in ("gpu/worker.py", "gpu/quality.py", "gpu/profiling.py", "gpu/phase_conflict.py", "cli.py", "util.py"):
        path = package.parent / name
        files[name] = file_hash(path)
    binary = Path(os.environ.get("DITSERVE_CORE", str(ROOT / ".cache/build/ditserve-core")))
    if binary.is_file():
        files["external/core_binary"] = file_hash(binary)
    return {"files": files, "sha256": canonical_hash(files)}
