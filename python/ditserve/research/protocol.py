"""Freeze a separate protocol and executable sources; never edit PreViS runs."""

import importlib.metadata
import json
from pathlib import Path
import shutil

from .budget import MAX_GPU_SECONDS, STAGE_LIMITS
from .common import atomic_json, digest, file_digest, read_json


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research_pilot.json"
DEFAULT_OUTPUT = ROOT / "results/research-pilot/run01"
DEFAULT_LEDGER = ROOT / "results/research-pilot/gpu-budget.json"
DEPENDENCY_SOURCES = {
    "cache-dit": ["cache_dit/caching/cache_contexts/cache_manager.py",
                  "cache_dit/caching/cache_contexts/cached_context.py",
                  "cache_dit/distributed/transformers/wan.py",
                  "cache_dit/attention/backends/native.py"],
    "diffusers": ["diffusers/pipelines/wan/pipeline_wan.py",
                  "diffusers/models/transformers/transformer_wan.py",
                  "diffusers/schedulers/scheduling_unipc_multistep.py"],
}


def validate_protocol(config):
    """The pilot is deliberately narrow; expanding it needs a new protocol."""
    expected = read_json(DEFAULT_CONFIG)
    # Estimates/timeouts may be made more conservative before freezing. All
    # scientific and budget choices are fixed for this first pilot.
    configurable = {"estimate_wall_seconds", "timeout_wall_seconds", "cleanup_wall_seconds"}
    if set(config) != set(expected) or any(config[k] != expected[k] for k in expected if k not in configurable):
        raise ValueError("protocol differs from the fixed first-pilot scientific recipe")
    from .common import finite
    for field in ("estimate_wall_seconds", "timeout_wall_seconds"):
        if set(config[field]) != set(expected[field]):
            raise ValueError("missing stage estimates/timeouts")
        for value in config[field].values():
            finite(value, field, positive=True)
    finite(config["cleanup_wall_seconds"], "cleanup reserve", positive=True)
    if config["budget_gpu_seconds"] != MAX_GPU_SECONDS or config["stage_budgets_gpu_seconds"] != STAGE_LIMITS:
        raise ValueError("unsupported research budget")
    return config


def source_files(root=ROOT):
    """Archive executable local code, plus the tests defining its contracts."""
    root = Path(root)
    paths = sorted((root / "python/ditserve").rglob("*.py"))
    paths += sorted((root / "tests/python").glob("test_research*.py"))
    paths += [root / "configs/research_pilot.json"]
    return {str(p.relative_to(root)): p for p in paths if p.is_file()}


def dependency_identity():
    result = {}
    config = read_json(DEFAULT_CONFIG)
    for package, expected in config["dependencies"].items():
        try:
            dist = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = {"version": None, "expected": expected, "files": {}}
            continue
        files = {}
        # Hash the whole Python implementation of the two behavior-owning
        # packages, including indirect attention/cache helpers.
        if package in DEPENDENCY_SOURCES:
            prefix = "cache_dit/" if package == "cache-dit" else "diffusers/"
            for relative in dist.files or ():
                name = str(relative)
                if name.startswith(prefix) and name.endswith(".py"):
                    path = Path(dist.locate_file(relative))
                    if path.is_file():
                        files[name] = file_digest(path)
        result[package] = {"version": dist.version, "expected": expected, "files": files}
    return result


def quality_assets():
    """Bind the offline LPIPS network weights as well as library versions."""
    paths = {"alexnet": ROOT / ".cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"}
    try:
        distribution = importlib.metadata.distribution("lpips")
        paths["lpips_alex_v01"] = Path(distribution.locate_file("lpips/weights/v0.1/alex.pth"))
    except importlib.metadata.PackageNotFoundError:
        pass
    return {name: {"path": str(path.resolve()), "sha256": file_digest(path) if path.is_file() else None}
            for name, path in paths.items()}


def freeze_protocol(config_path, output, ledger, *, root=ROOT):
    config = validate_protocol(read_json(config_path))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "protocol.json"
    files = source_files(root)
    hashes = {name: file_digest(path) for name, path in files.items()}
    dependencies = dependency_identity()
    payload = {"schema_version": "ditserve.research.frozen.v1", "config": config,
               "source_files": hashes, "source_sha256": digest(hashes),
               "dependencies": dependencies, "quality_assets": quality_assets(),
               "ledger": str(Path(ledger).resolve())}
    payload["sha256"] = digest(payload)
    if frozen_path.exists():
        existing = read_json(frozen_path)
        if existing != payload:
            raise ValueError("frozen protocol/source/dependencies/ledger changed; use a new run directory with the same ledger")
        verify_protocol(frozen_path, root=root)
        return existing
    archive = output / "sources"
    for name, path in files.items():
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_digest(target) != hashes[name]:
            raise ValueError("existing source archive does not match: " + name)
        shutil.copyfile(path, target)
    for package, identity in dependencies.items():
        if not identity["files"]:
            continue
        dist = importlib.metadata.distribution(package)
        for name in identity["files"]:
            target = archive / "dependencies" / package / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dist.locate_file(name), target)
    atomic_json(frozen_path, payload)
    return payload


def verify_protocol(path, *, root=ROOT, verify_current=True):
    path = Path(path)
    frozen = read_json(path)
    unsigned = {k: v for k, v in frozen.items() if k != "sha256"}
    if frozen.get("schema_version") != "ditserve.research.frozen.v1" or frozen.get("sha256") != digest(unsigned):
        raise ValueError("invalid frozen protocol checksum")
    if frozen["source_sha256"] != digest(frozen["source_files"]):
        raise ValueError("invalid source manifest checksum")
    if verify_current and set(source_files(root)) != set(frozen["source_files"]):
        raise ValueError("current implementation source file set differs from frozen source")
    for name, expected in frozen["source_files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe source archive path")
        if file_digest(path.parent / "sources" / relative) != expected:
            raise ValueError("source archive mismatch: " + name)
        if verify_current and file_digest(Path(root) / relative) != expected:
            raise ValueError("current implementation differs from frozen source: " + name)
    for package, identity in frozen["dependencies"].items():
        for name, expected in identity["files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("unsafe dependency path")
            if file_digest(path.parent / "sources/dependencies" / package / name) != expected:
                raise ValueError("dependency archive mismatch: " + name)
    if verify_current and dependency_identity() != frozen["dependencies"]:
        raise ValueError("current dependency source differs from frozen protocol")
    if verify_current and quality_assets() != frozen["quality_assets"]:
        raise ValueError("offline quality weights differ from frozen protocol")
    return frozen
