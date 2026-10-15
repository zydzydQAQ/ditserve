"""Independent, offline SP1 protocol for the heterogeneous two-device pilot.

Model hashing is always explicit. Inventory discovery never streams weights;
freezing requires an already completed SHA256 manifest. No model is loaded.
"""

import importlib.metadata
import os
from pathlib import Path
import re
import shutil

from .common import atomic_json, digest, file_digest, finite, read_json
from .protocol import dependency_identity, quality_assets


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research_two_gpu.json"
DEFAULT_LEDGER = ROOT / "results/research-pilot/gpu-budget.json"
REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
MODEL_PATH = Path("/mnt/data/zyd-preempt-phase16-01a1296d/dit-runtime/huggingface/hub/"
                  "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots") / REVISION
ARTIFACT_ROOT = Path("/mnt/data/zyd-preempt-phase16-01a1296d/dit-research-artifacts")
DEFAULT_OUTPUT = ROOT / "results/research-two-gpu/run01"
MODEL_SCHEMA = "ditserve.research.dual.model.v1"
FROZEN_SCHEMA = "ditserve.research.dual.frozen.v1"
WEIGHT_GROUPS = {
    "text_encoder": ("model.safetensors.index.json", "model", 5),
    "transformer": ("diffusion_pytorch_model.safetensors.index.json", "diffusion_pytorch_model", 2),
}
REQUIRED_METADATA = (
    "model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
    "tokenizer/special_tokens_map.json", "tokenizer/spiece.model", "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json", "transformer/config.json", "vae/config.json",
)
MODEL_COMPONENTS = {
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}


def validate_protocol(config):
    if config.get("schema_version") == "ditserve.research.cfg.protocol.v1":
        from .cfg_protocol import validate_protocol as validate_cfg_protocol
        return validate_cfg_protocol(config)
    expected = read_json(DEFAULT_CONFIG)
    adjustable = {"estimate_wall_seconds", "timeout_wall_seconds", "cleanup_wall_seconds"}
    if (set(config) != set(expected)
            or any(config[key] != expected[key] for key in expected if key not in adjustable)):
        raise ValueError("protocol differs from the fixed two-GPU scientific recipe")
    for field in ("estimate_wall_seconds", "timeout_wall_seconds"):
        if set(config[field]) != set(expected[field]):
            raise ValueError("missing two-GPU stage estimates/timeouts")
        for value in config[field].values():
            finite(value, field, positive=True)
    finite(config["cleanup_wall_seconds"], "cleanup reserve", positive=True)
    if (config["model_path"] != str(MODEL_PATH) or config["revision"] != REVISION
            or config["artifact_root"] != str(ARTIFACT_ROOT)
            or config["budget_gpu_seconds"] != 28800):
        raise ValueError("fixed offline model, artifact root, revision or lifetime budget changed")
    return config


def attention_for_device(device, *, efficient_qualified=False):
    """Select only the frozen device family; an efficient V100 gate is explicit."""
    name = device.get("name", "")
    capability = str(device.get("compute_capability", ""))
    if "A100" in name and capability == "8.0":
        return "flash"
    if "V100" in name and capability == "7.0":
        return "efficient" if efficient_qualified is True else "math"
    raise ValueError("two-GPU protocol supports only A100/8.0 and V100/7.0, each at SP1")


def _fixed_model_path(model_path):
    path = Path(MODEL_PATH if model_path is None else model_path)
    # Retain the snapshot name instead of resolving its symlinked HF blob files.
    if path.absolute() != MODEL_PATH.absolute() or path.name != REVISION or not path.is_dir():
        raise ValueError("fixed offline model snapshot unavailable or path/revision differs; no fallback")
    return path


def _file_identity(path):
    stat = path.stat()
    if not path.is_file() or stat.st_size <= 0:
        raise ValueError("model file is absent or empty: " + str(path))
    return dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                inode=stat.st_ino, device=stat.st_dev, resolved_path=str(path.resolve()))


def inspect_model_snapshot(model_path=None, *, hash_files=False):
    """Validate indexes and files; hash all loadable files only when requested.

    Call hash_files=True only after the running allocation has stopped. The
    caller owns that scheduling decision; this function never changes a ledger.
    """
    path = _fixed_model_path(model_path)
    required = set(REQUIRED_METADATA) | {"vae/diffusion_pytorch_model.safetensors"}
    indexes = {}
    for component, (index_name, prefix, count) in WEIGHT_GROUPS.items():
        relative = component + "/" + index_name
        index = read_json(path / relative)
        weights = index.get("weight_map")
        if (not isinstance(weights, dict) or not weights
                or any(not isinstance(key, str) or not key or not isinstance(value, str)
                       for key, value in weights.items())):
            raise ValueError("invalid model shard index: " + relative)
        expected = {"%s-%05d-of-%05d.safetensors" % (prefix, i, count) for i in range(1, count + 1)}
        if set(weights.values()) != expected:
            raise ValueError("missing, extra, or unsafe model shard reference: " + relative)
        metadata_size = index.get("metadata", {}).get("total_size")
        if type(metadata_size) is not int or metadata_size <= 0:
            raise ValueError("invalid shard index total_size: " + relative)
        required.add(relative)
        required.update(component + "/" + name for name in expected)
        indexes[relative] = dict(shards=sorted(expected), tensors=len(weights), tensor_bytes=metadata_size)
    # An alternate .bin or extra safetensors file can change loader selection.
    actual_weights = {str(p.relative_to(path)) for p in path.rglob("*")
                      if p.suffix in (".safetensors", ".bin", ".pt", ".pth")}
    expected_weights = {name for name in required if name.endswith(".safetensors")}
    if actual_weights != expected_weights:
        raise ValueError("model weight file set differs from the fixed snapshot schema")
    model = read_json(path / "model_index.json")
    if model.get("_class_name") != "WanPipeline" or any(model.get(k) != v for k, v in MODEL_COMPONENTS.items()):
        raise ValueError("model pipeline/component identity is not pinned Wan1.3B")
    transformer = read_json(path / "transformer/config.json")
    identity = dict(_class_name="WanTransformer3DModel", num_layers=30, num_attention_heads=12,
                    attention_head_dim=128, in_channels=16, out_channels=16, patch_size=[1, 2, 2])
    if any(transformer.get(k) != v for k, v in identity.items()):
        raise ValueError("transformer is not the fixed Wan1.3B architecture")
    files = {}
    for name in sorted(required):
        before = _file_identity(path / name)
        sha256 = file_digest(path / name) if hash_files else None
        if _file_identity(path / name) != before:
            raise ValueError("model file changed while being inventoried/hashed: " + name)
        files[name] = dict(before, sha256=sha256)
    # Tensor bytes omit the safetensors headers, but must fit in the shards.
    for relative, index in indexes.items():
        component = str(Path(relative).parent)
        if index["tensor_bytes"] >= sum(files[component + "/" + name]["size_bytes"] for name in index["shards"]):
            raise ValueError("shard files cannot contain the indexed tensor bytes: " + relative)
    result = dict(schema_version=MODEL_SCHEMA, model_path=str(path.absolute()), revision=REVISION,
                  status="sha256_verified" if hash_files else "inventory_only",
                  hash_files=bool(hash_files), files=files, indexes=indexes,
                  total_file_bytes=sum(row["size_bytes"] for row in files.values()))
    result["sha256"] = digest(result)
    return result


def verify_model_manifest(manifest, *, rehash=False):
    unsigned = {key: value for key, value in manifest.items() if key != "sha256"}
    if (manifest.get("schema_version") != MODEL_SCHEMA or manifest.get("sha256") != digest(unsigned)
            or manifest.get("status") != "sha256_verified" or manifest.get("hash_files") is not True
            or manifest.get("model_path") != str(MODEL_PATH.absolute()) or manifest.get("revision") != REVISION):
        raise ValueError("complete bound SHA256 model manifest required")
    current = inspect_model_snapshot(hash_files=rehash)
    if (set(current["files"]) != set(manifest.get("files", {}))
            or current["indexes"] != manifest.get("indexes")
            or current["total_file_bytes"] != manifest.get("total_file_bytes")):
        raise ValueError("model inventory differs from hashed manifest")
    for name, row in manifest["files"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))):
            raise ValueError("model file lacks SHA256: " + name)
        observed = current["files"][name]
        if any(observed.get(k) != v for k, v in row.items() if k != "sha256" or rehash):
            raise ValueError("model file differs from hashed manifest: " + name)
    return manifest


def source_files(root=ROOT):
    root = Path(root)
    paths = sorted((root / "python/ditserve").rglob("*.py"))
    paths += sorted((root / "tests/python").glob("test_research*.py"))
    paths += [root / "configs/research_two_gpu.json", root / "configs/research_pilot.json",
              root / "configs/research_cfg.json"]
    return {str(p.relative_to(root)): p for p in paths if p.is_file()}


def freeze_protocol(config_path, output, ledger=DEFAULT_LEDGER, *, model_manifest, root=ROOT,
                    cfg_continuation=None, cfg_refinement=None, cpu_embeddings=None):
    """Freeze independent sources/model identity; never reallocate the ledger."""
    config = validate_protocol(read_json(config_path))
    if cfg_continuation is not None:
        if config.get("schema_version") != "ditserve.research.cfg.protocol.v1":
            raise ValueError("prototype continuation is exclusive to the CFG protocol")
        from .cfg_continuation import verify_continuation
        cfg_continuation = verify_continuation(cfg_continuation, config=config, ledger=ledger)
    if cfg_refinement is not None:
        from .cfg_refinement import verify_refinement, verify_cpu_binding
        cfg_refinement = verify_refinement(cfg_refinement, config=config, ledger=ledger)
        if not cfg_continuation or cfg_refinement["prototype_continuation_sha256"] != cfg_continuation["sha256"]:
            raise ValueError("refinement requires its original prototype continuation")
        verify_cpu_binding(cpu_embeddings, config=config, model_manifest=model_manifest)
    elif cpu_embeddings is not None:
        raise ValueError("CPU embeddings require an implementation refinement")
    verify_model_manifest(model_manifest)
    output = Path(output).resolve()
    files = source_files(root)
    hashes = {name: file_digest(path) for name, path in files.items()}
    dependencies = dependency_identity()
    archive_hash = digest(dict(source_files=hashes, dependencies=dependencies))
    archive = output.parent / "source-snapshots" / archive_hash
    archive_relative = os.path.relpath(archive, output)
    frozen_schema = ("ditserve.research.cfg.frozen.v1"
                     if config.get("schema_version") == "ditserve.research.cfg.protocol.v1" else FROZEN_SCHEMA)
    payload = dict(schema_version=frozen_schema, config=config, source_files=hashes,
                   source_sha256=digest(hashes), dependencies=dependencies, quality_assets=quality_assets(),
                   model_manifest=model_manifest, model_verification="SHA256 at freeze; file identity checked on each launch",
                   ledger=str(Path(ledger).resolve()), source_archive=dict(sha256=archive_hash, relative_path=archive_relative))
    if cfg_continuation is not None:
        payload["prototype_continuation"] = cfg_continuation
    if cfg_refinement is not None:
        payload["implementation_refinement"] = cfg_refinement
        payload["cpu_embeddings"] = cpu_embeddings
    payload["sha256"] = digest(payload)
    frozen_path = output / "protocol.json"
    if frozen_path.exists():
        if read_json(frozen_path) != payload:
            raise ValueError("frozen two-GPU protocol/source/model changed; use a new run directory")
        return verify_protocol(frozen_path, root=root)
    # Sibling runs share one immutable content-addressed archive. Structured
    # results stay in results/; only retained raw artifacts use ARTIFACT_ROOT.
    for name, source in files.items():
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_digest(target) != hashes[name]:
                raise ValueError("existing source archive differs: " + name)
        else:
            shutil.copyfile(source, target)
    for package, identity in dependencies.items():
        if identity["files"]:
            distribution = importlib.metadata.distribution(package)
            for name in identity["files"]:
                target = archive / "dependencies" / package / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if file_digest(target) != identity["files"][name]:
                        raise ValueError("existing dependency archive differs: " + name)
                else:
                    shutil.copyfile(distribution.locate_file(name), target)
    output.mkdir(parents=True, exist_ok=True)
    link = output / "sources"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != archive.resolve():
            raise ValueError("existing run source reference differs from content-addressed archive")
    else:
        link.symlink_to(archive_relative, target_is_directory=True)
    atomic_json(frozen_path, payload)
    return payload


def verify_protocol(path, *, root=ROOT, verify_current=True, rehash_model=False):
    path = Path(path)
    frozen = read_json(path)
    expected_schema = ("ditserve.research.cfg.frozen.v1"
                       if frozen.get("config", {}).get("schema_version") == "ditserve.research.cfg.protocol.v1"
                       else FROZEN_SCHEMA)
    if (frozen.get("schema_version") != expected_schema
            or frozen.get("sha256") != digest({k: v for k, v in frozen.items() if k != "sha256"})
            or frozen.get("source_sha256") != digest(frozen.get("source_files"))):
        raise ValueError("invalid frozen two-GPU protocol checksum")
    validate_protocol(frozen["config"])
    if "prototype_continuation" in frozen:
        if frozen["schema_version"] != "ditserve.research.cfg.frozen.v1":
            raise ValueError("prototype continuation is exclusive to the CFG protocol")
        from .cfg_continuation import verify_continuation
        verify_continuation(frozen["prototype_continuation"], config=frozen["config"], ledger=frozen["ledger"])
    if "implementation_refinement" in frozen:
        from .cfg_refinement import verify_refinement, verify_cpu_binding
        refinement = verify_refinement(frozen["implementation_refinement"], config=frozen["config"], ledger=frozen["ledger"])
        if refinement["prototype_continuation_sha256"] != frozen.get("prototype_continuation", {}).get("sha256"):
            raise ValueError("refinement lost its original prototype continuation")
        verify_cpu_binding(frozen.get("cpu_embeddings"), config=frozen["config"],
                           model_manifest=frozen["model_manifest"], verify_arrays=verify_current)
    elif "cpu_embeddings" in frozen:
        raise ValueError("CPU embedding binding without implementation refinement")
    archive_hash = digest(dict(source_files=frozen["source_files"], dependencies=frozen["dependencies"]))
    expected_archive = path.parent.parent / "source-snapshots" / archive_hash
    archive = frozen.get("source_archive", {})
    if (archive.get("sha256") != archive_hash
            or archive.get("relative_path") != os.path.relpath(expected_archive, path.parent)
            or not (path.parent / "sources").is_symlink()
            or (path.parent / "sources").resolve() != expected_archive.resolve()):
        raise ValueError("invalid content-addressed source archive reference")
    if verify_current:
        verify_model_manifest(frozen["model_manifest"], rehash=rehash_model)
        if set(source_files(root)) != set(frozen["source_files"]):
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
                raise ValueError("unsafe dependency archive path")
            if file_digest(path.parent / "sources/dependencies" / package / name) != expected:
                raise ValueError("dependency archive mismatch: " + name)
    if verify_current and (dependency_identity() != frozen["dependencies"] or quality_assets() != frozen["quality_assets"]):
        raise ValueError("current dependency/quality assets differ from frozen two-GPU protocol")
    return frozen
