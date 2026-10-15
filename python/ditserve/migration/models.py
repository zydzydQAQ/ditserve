"""Pinned, resumable model provisioning without importing or initializing CUDA.

The checked-in lock contains every selected runtime file, including configurations,
indices, tokenizers and weights. A local snapshot is accepted only by content, never
by a directory name or a previous machine's provisioning record.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import urllib.request


DEFAULT_LOCK = Path(__file__).resolve().parents[3] / "configs/models.lock.json"
GROUPS = ("core", "sessions", "longlive")
LONG_LIVE_PYTHON = "/opt/longlive-venv/bin/python"
LONG_LIVE_SOURCE = "/opt/longlive-source"

# Transformers and Diffusers can prefer an unsharded checkpoint or an optional
# tokenizer/config file over the locked assets. Inspect only actual loader
# directories: caches, documentation, and the shared LPIPS checkpoint directory
# are not part of this check.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".h5", ".onnx")
_LOADER_FILENAMES = frozenset({
    "config.json", "model_index.json", "generation_config.json", "scheduler_config.json",
    "adapter_config.json", "quantization_config.json", "preprocessor_config.json",
    "processor_config.json", "feature_extractor_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "tokenizer.json", "tokenizer.model",
    "spiece.model", "sentencepiece.bpe.model", "vocab.json", "vocab.txt", "merges.txt",
    "chat_template.jinja", "chat_templates",
})


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid lock path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in (".", "..") for p in value.split("/")):
        raise ValueError("unsafe lock path: " + value)
    return path


def load_lock(lock_path=None):
    """Read and structurally validate the complete immutable model lock."""
    lock = json.loads(Path(lock_path or DEFAULT_LOCK).read_text())
    if lock.get("schema_version") != "ditserve.migration.models.v1":
        raise ValueError("unsupported model lock schema")
    names, targets = set(), set()
    for model in lock["models"]:
        name = model["name"]
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or name in names:
            raise ValueError("invalid or duplicate model name")
        names.add(name)
        if model["group"] not in GROUPS or not model["files"]:
            raise ValueError("invalid model group or empty manifest")
        directory = _relative(model["directory"])
        for item in model["files"]:
            target = str(directory / _relative(item["path"]))
            if target in targets:
                raise ValueError("duplicate model file: " + target)
            targets.add(target)
            if (not isinstance(item["bytes"], int) or item["bytes"] < 0
                    or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                raise ValueError("invalid size or SHA256: " + target)
            if "url" in item:
                if not item["url"].startswith("https://"):
                    raise ValueError("model URL must use HTTPS")
            else:
                revision = item.get("revision", model.get("revision", ""))
                if not re.fullmatch(r"[0-9a-f]{40}", revision):
                    raise ValueError("Hugging Face revision must be a commit")
                repo = item.get("repo_id", model.get("repo_id", ""))
                if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
                    raise ValueError("invalid Hugging Face repository")
                prefix = item.get("download_directory")
                remote = _relative(item.get("remote_path", item["path"]))
                mapped = _relative(prefix) / remote if prefix else remote
                if str(mapped) != item["path"]:
                    raise ValueError("remote and local model paths disagree")
    return lock


def _selected(lock, groups):
    groups = (groups,) if isinstance(groups, str) else tuple(groups)
    if not groups or set(groups) - set(GROUPS):
        raise ValueError("groups must contain core, sessions and/or longlive")
    return [model for model in lock["models"] if model["group"] in groups]


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _file_error(path, item):
    if not path.is_file():
        return "missing"
    if path.stat().st_size != item["bytes"]:
        return "size_mismatch"
    if _sha256(path) != item["sha256"]:
        return "sha256_mismatch"
    return None


def _unexpected_loader_files(directory, model, listed):
    if model["name"] not in ("wan", "sana", "mistral"):
        return []
    errors = []
    parents = {PurePosixPath(path).parent for path in listed}
    for parent in sorted(parents):
        folder = directory / parent
        if not folder.is_dir():
            continue  # Missing locked assets are reported individually below.
        try:
            for entry in sorted(folder.iterdir()):
                relative = str(parent / entry.name)
                if relative in listed:
                    continue
                if (entry.name in _LOADER_FILENAMES
                        or entry.name.endswith(_WEIGHT_SUFFIXES)
                        or any(entry.name.endswith(suffix + ".index.json") for suffix in _WEIGHT_SUFFIXES)):
                    errors.append({"path": relative, "reason": "unexpected_loader_file"})
        except OSError as error:
            errors.append({"path": str(parent), "reason": type(error).__name__})
    return errors


def _inspect_model(root, model):
    directory = root / model["directory"]
    listed = {item["path"] for item in model["files"]}
    errors = _unexpected_loader_files(directory, model, listed)
    checked = 0
    for item in model["files"]:
        try:
            error = _file_error(directory / item["path"], item)
            if error:
                errors.append({"path": item["path"], "reason": error})
            else:
                checked += 1
                if item["path"].endswith(".index.json"):
                    index = json.loads((directory / item["path"]).read_text())
                    shards = index.get("weight_map")
                    if not isinstance(shards, dict) or not shards:
                        errors.append({"path": item["path"], "reason": "invalid_shard_index"})
                    else:
                        for shard in sorted(set(shards.values())):
                            relative = str(PurePosixPath(item["path"]).parent / _relative(shard))
                            if relative not in listed:
                                errors.append({"path": relative, "reason": "shard_not_in_lock"})
        except (OSError, ValueError, TypeError) as error:
            errors.append({"path": item["path"], "reason": type(error).__name__})
    return {"name": model["name"], "group": model["group"], "path": str(directory),
            "revision": model["revision"], "status": "failed" if errors else "verified",
            "checked_files": checked, "expected_files": len(model["files"]),
            "total_bytes": sum(item["bytes"] for item in model["files"]), "errors": errors}


def verify_models(root, groups=("core",), *, lock_path=None):
    """Read-only full SHA256 and shard-index verification; does not load models."""
    root = Path(root).absolute()
    lock = load_lock(lock_path)
    records = [_inspect_model(root, model) for model in _selected(lock, groups)]
    errors = [dict(error, model=record["name"]) for record in records for error in record["errors"]]
    return {"schema_version": "ditserve.migration.model_verification.v1",
            "status": "failed" if errors else "verified", "root": str(root),
            "models": records, "errors": errors, "gpu_work_started": False,
            "scope": "complete selected file SHA256, shard indices and loader override rejection; GPU execution and environment qualification are separate"}


def _download_url(url, target, item):
    """Resume the public AlexNet asset, validate first, then atomically publish."""
    temporary = target.with_name(target.name + "." + item["sha256"] + ".partial")
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > item["bytes"]:
        temporary.unlink()
        offset = 0
    if offset < item["bytes"]:
        request = urllib.request.Request(url, headers={"Range": "bytes=%d-" % offset} if offset else {})
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset and response.status == 206:
                expected_range = "bytes %d-%d/%d" % (offset, item["bytes"] - 1, item["bytes"])
                if response.headers.get("Content-Range") != expected_range:
                    raise ValueError("download server returned an unexpected byte range")
            elif response.status == 200:
                offset = 0
            else:
                raise ValueError("download server did not provide the requested file")
            with temporary.open("ab" if offset else "wb") as handle:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    offset += len(block)
                    if offset > item["bytes"]:
                        raise ValueError("download exceeds locked size")
                    handle.write(block)
    error = _file_error(temporary, item)
    if error:
        if error == "sha256_mismatch":
            temporary.unlink()
        raise ValueError("download " + error)
    temporary.replace(target)


def _download_file(root, model, item):
    directory = root / model["directory"]
    target = directory / item["path"]
    if _file_error(target, item) is None:
        return "reused"
    # Never repair through a symlink into an existing installation.
    if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != root.parent):
        raise ValueError("refusing to overwrite a corrupt or missing asset through a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    if "url" in item:
        _download_url(item["url"], target, item)
    else:
        from huggingface_hub import hf_hub_download
        # local_dir mode keeps a single asset copy and maintains resumable .incomplete
        # files. Credentials come only from HF_TOKEN/the standard HF credential store.
        prefix = item.get("download_directory")
        destination = directory / prefix if prefix else directory
        downloaded = hf_hub_download(
            repo_id=item.get("repo_id", model.get("repo_id")),
            revision=item.get("revision", model["revision"]),
            filename=item.get("remote_path", item["path"]),
            local_dir=str(destination), force_download=target.exists())
        if Path(downloaded).absolute() != target.absolute():
            raise ValueError("Hugging Face returned an unexpected target path")
        error = _file_error(target, item)
        if error:
            raise ValueError("download " + error)
    return "downloaded"


_LONG_LIVE_AUDIT = r'''
import collections, importlib.metadata, json, os, pathlib, subprocess, sys
from unittest.mock import patch
source, weights = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == sys.argv[3], 'LongLive source revision mismatch'
subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD', '--', 'LongLive1.0'], check=True)
assert not os.environ.get('CUDA_VISIBLE_DEVICES'), 'CPU validation must hide GPUs'
import torch
assert torch.__version__ == '2.8.0+cu126', 'LongLive requires torch 2.8.0+cu126'
assert torch._C._GLIBCXX_USE_CXX11_ABI, 'LongLive flash-attn ABI mismatch'
def forbidden(*args, **kwargs):
    raise RuntimeError('CPU audit attempted CUDA initialization')
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda._lazy_init = forbidden
import torchvision, flash_attn, diffusers, transformers, peft, omegaconf, einops
assert flash_attn.__version__ == '2.8.3', 'flash-attn version mismatch'
assert diffusers.__version__ == '0.31.0', 'diffusers version mismatch'
sys.path.insert(0, str(source / 'LongLive1.0'))
modules = ['wan.modules.attention', 'wan.modules.t5', 'wan.modules.vae', 'utils.lora_utils',
           'wan.modules.causal_model', 'utils.wan_wrapper', 'pipeline.interactive_causal_inference']
with patch.object(torch.cuda, 'current_device', return_value=0):
    for name in modules:
        __import__(name)
attention = sys.modules['wan.modules.attention']
assert attention.FLASH_ATTN_2_AVAILABLE is True, 'native FlashAttention 2 unavailable'
records = []
for relative, key in [('longlive_models/models/longlive_base.pt', 'generator'),
                      ('longlive_models/models/lora.pt', 'generator_lora'),
                      ('wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth', ''),
                      ('wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', '')]:
    value = torch.load(weights / relative, map_location='cpu', weights_only=True, mmap=True)
    state = value.get(key, value)
    assert isinstance(state, dict) and state and all(isinstance(v, torch.Tensor) for v in state.values())
    records.append({'path': relative, 'state_tensors': len(state),
                    'parameters': sum(v.numel() for v in state.values()),
                    'dtypes': dict(collections.Counter(str(v.dtype) for v in state.values()))})
    del state, value
from wan.modules.tokenizers import HuggingfaceTokenizer
tokenizer = HuggingfaceTokenizer(str(weights / 'wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl'),
                                seq_len=512, clean='whitespace', local_files_only=True)
ids, mask = tokenizer(['Migration CPU import audit.'], return_mask=True, add_special_tokens=True)
assert tuple(ids.shape) == (1, 512) and ids.device.type == 'cpu'
assert not torch.cuda.is_initialized(), 'CUDA initialized during CPU audit'
packages = ['torch', 'torchvision', 'flash-attn', 'diffusers', 'transformers', 'peft',
            'accelerate', 'omegaconf', 'einops', 'av', 'safetensors', 'huggingface-hub']
print(json.dumps({'status': 'verified_cpu_only', 'python': str(pathlib.Path(sys.executable).absolute()),
  'gpu_used': False, 'cuda_initialized': False, 'source_path': str(source), 'source_revision': revision,
  'versions': {name: importlib.metadata.version(name) for name in packages},
  'native_attention_backends': {'FLASH_ATTN_2_AVAILABLE': attention.FLASH_ATTN_2_AVAILABLE},
  'checkpoint_metadata': records, 'tokenizer': {'shape': list(ids.shape), 'device': 'cpu'},
  'scope': 'source/imports/checkpoint metadata only; native GPU execution remains unverified'}))
'''


def validate_longlive_environment(root, *, python=LONG_LIVE_PYTHON, source_path=LONG_LIVE_SOURCE,
                                  lock_path=None):
    """Write a fresh CPU-only record in the target environment, never copy old proof."""
    root = Path(root).absolute()
    source = Path(source_path).absolute()
    python = str(Path(python).absolute())
    lock = load_lock(lock_path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    try:
        process = subprocess.run([python, "-c", _LONG_LIVE_AUDIT, str(source),
                                  str(root / "longlive/weights"), lock["longlive_source"]["revision"]],
                                 env=env, capture_output=True, text=True, timeout=300)
        if process.returncode:
            # Do not persist arbitrary third-party stderr, which can include local
            # credentials/URLs. Only extract a module identifier or our fixed checks.
            result = {"status": "failed", "reason": "longlive_cpu_audit_failed",
                      "returncode": process.returncode}
            missing = re.findall(r"No module named '([\w.]+)'", process.stderr)
            if missing:
                result["missing_module"] = missing[-1]
            for check in ("LongLive source revision mismatch", "LongLive requires torch 2.8.0+cu126",
                          "LongLive flash-attn ABI mismatch", "flash-attn version mismatch",
                          "diffusers version mismatch", "native FlashAttention 2 unavailable",
                          "CPU audit attempted CUDA initialization"):
                if check in process.stderr:
                    result["failed_check"] = check
            return result
        result = json.loads(process.stdout.strip().splitlines()[-1])
        if result.get("status") != "verified_cpu_only" or result.get("python") != python:
            return {"status": "failed", "reason": "longlive_cpu_audit_identity_mismatch"}
        record = root / ".provisioning/longlive-environment.json"
        _atomic_json(record, result)
        return {"status": "passed", "python": python, "record": str(record), "sha256": _sha256(record)}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
        return {"status": "failed", "reason": type(error).__name__}


def download_models(root, groups=("core",), *, lock_path=None,
                    longlive_python=LONG_LIVE_PYTHON, source_path=LONG_LIVE_SOURCE):
    """Provision selected model groups and publish verified manifests atomically.

    Missing files resume using the HF downloader; present files are always hashed.
    No credentials are accepted in arguments or copied into reports. This operation
    writes only below root, except HF's normal credentials/cache metadata handling.
    """
    root = Path(root).absolute()
    lock = load_lock(lock_path)
    models = _selected(lock, groups)
    records, errors = [], []
    for model in models:
        manifest_path = root / ".provisioning" / (model["name"] + ".json")
        # Invalidate old readiness before repairing; an interrupted operation must
        # not leave a previous environment's ready marker usable by research code.
        _atomic_json(manifest_path, {"status": "pending", "model": model["name"]})
        failures, reused, downloaded = [], 0, 0
        for item in model["files"]:
            try:
                action = _download_file(root, model, item)
                reused += action == "reused"
                downloaded += action == "downloaded"
            except Exception as error:
                failures.append({"model": model["name"], "path": item["path"],
                                 "reason": "download_failed", "error_type": type(error).__name__})
        record = _inspect_model(root, model)
        record.update(reused_files=reused, downloaded_files=downloaded)
        failures.extend(dict(error, model=model["name"]) for error in record["errors"])
        if not failures:
            manifest = {"status": "verified", "model": {
                "id": model.get("repo_id", model["name"]), "revision": model["revision"],
                "path": str(root / model["directory"])},
                "files": [{"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]}
                          for item in model["files"]], "verification": "full SHA256 and shard indices",
                "lock_sha256": _sha256(lock_path or DEFAULT_LOCK)}
            if model["name"] == "longlive":
                manifest["source_revision"] = lock["longlive_source"]["revision"]
                validation = validate_longlive_environment(root, python=longlive_python,
                    source_path=source_path, lock_path=lock_path)
                manifest["environment_validation"] = validation
                if validation["status"] != "passed":
                    failures.append({"model": "longlive", "reason": "longlive_environment_not_validated",
                                     "detail": validation})
            _atomic_json(manifest_path, manifest)
        record["status"] = "failed" if failures else "verified"
        record["errors"] = failures
        record["manifest"] = str(manifest_path)
        records.append(record)
        errors.extend(failures)
    return {"schema_version": "ditserve.migration.provisioning.v1",
            "status": "failed" if errors else "verified", "root": str(root),
            "models": records, "errors": errors, "gpu_work_started": False}
