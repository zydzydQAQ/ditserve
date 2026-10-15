"""Strict, CPU-only text preparation before a new CFG GPU allocation.

This cache is an execution input, never quality or performance evidence. Only
the explicit research RPC installs it; the ordinary worker encoder is unchanged.
"""

import contextlib
import importlib.metadata
import os
from pathlib import Path
import re
import tempfile
import time

from .common import atomic_json, digest, file_digest, read_json
from . import dual_protocol
from .prompt_bank import FrozenPromptBank
from ..gpu.worker import EncodedPrompt, normalize_spec


SCHEMA = "ditserve.research.cfg.cpu_embeddings.v1"
SHAPE = (1, 512, 4096)
THREADS = 8
ENCODER_PACKAGES = ("torch", "diffusers", "transformers")


def _validate_config(config):
    if (config.get("schema_version") != "ditserve.research.cfg.protocol.v1"
            or config.get("model_path") != str(dual_protocol.MODEL_PATH)
            or config.get("revision") != dual_protocol.REVISION
            or config.get("offline") is not True
            or config.get("guidance_scale") != 5.
            or config.get("precisions", {}).get("text_encoder") != "float32"
            or config.get("runtime_threads") != dict(omp=8, mkl=8, openblas=8)):
        raise ValueError("CPU embedding cache requires the fixed offline CFG model and CPU FP32/8-thread recipe")
    prompts = config.get("prompts")
    if (not isinstance(prompts, list) or not prompts
            or any(not isinstance(p.get("text"), str) or not p["text"] for p in prompts)
            or len({p.get("id") for p in prompts}) != len(prompts)
            or not isinstance(config.get("negative_prompt"), str)):
        raise ValueError("invalid CPU embedding prompt catalog")


def _encoder_identity(config):
    from diffusers.pipelines.wan import pipeline_wan
    versions = {name: importlib.metadata.version(name) for name in ENCODER_PACKAGES}
    if any(versions[name] != config["dependencies"][name] for name in ENCODER_PACKAGES):
        raise ValueError("CPU embedding dependency version differs from the frozen recipe")
    return dict(device="cpu", dtype="torch.float32", threads=THREADS,
                max_sequence_length=512, num_videos_per_prompt=1,
                encoder_class="UMT5EncoderModel", tokenizer_class="T5TokenizerFast",
                dependencies=versions, pipeline_source_sha256=file_digest(pipeline_wan.__file__),
                preparation_source_sha256=file_digest(__file__))


def _catalog(config, model_manifest_sha256, encoder):
    entries = {}
    for prompt in config["prompts"]:
        identity = dict(prompt=prompt["text"], negative_prompt=config["negative_prompt"],
            guidance_scale=config["guidance_scale"], do_classifier_free_guidance=True,
            model_path=config["model_path"], revision=config["revision"],
            model_manifest_sha256=model_manifest_sha256, encoder=encoder)
        key = digest(identity)
        if key not in entries:
            entries[key] = dict(identity=identity, prompt_ids=[])
        entries[key]["prompt_ids"].append(prompt["id"])
    return entries


@contextlib.contextmanager
def _cpu_runtime():
    # No is_available/set_device call: even CUDA initialization is forbidden.
    names = dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                 OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    old = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    import torch
    previous_threads = torch.get_num_threads()
    try:
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU preparation must precede GPU initialization in a separate CPU process")
        torch.set_num_threads(THREADS)
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            yield torch
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU text preparation unexpectedly initialized CUDA")
    finally:
        torch.set_num_threads(previous_threads)
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_encoder(config, torch):
    from diffusers import WanPipeline
    from transformers import T5TokenizerFast, UMT5EncoderModel
    path = config["model_path"]
    tokenizer = T5TokenizerFast.from_pretrained(path, subfolder="tokenizer", local_files_only=True,
                                               revision=config["revision"])
    encoder = UMT5EncoderModel.from_pretrained(path, subfolder="text_encoder", local_files_only=True,
        revision=config["revision"], torch_dtype=torch.float32, use_safetensors=True)
    encoder.to(device="cpu", dtype=torch.float32).eval()
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in encoder.parameters()):
        raise ValueError("text encoder is not entirely CPU FP32")
    # Direct construction is deliberate: from_pretrained on a WanPipeline
    # would load DiT and VAE weights even when only encode_prompt is needed.
    return WanPipeline(tokenizer=tokenizer, text_encoder=encoder, vae=None, scheduler=None,
                       transformer=None, transformer_2=None)


def _tensor_record(tensor):
    import torch
    if (not torch.is_tensor(tensor) or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32 or tuple(tensor.shape) != SHAPE
            or not bool(torch.isfinite(tensor).all())):
        raise ValueError("cached text embedding must be finite CPU FP32 [1,512,4096]")
    import hashlib
    sha = hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()
    return dict(shape=list(SHAPE), dtype="torch.float32", tensor_sha256=sha, file=sha + ".npy")


def _write_tensor(root, tensor):
    import numpy as np
    record = _tensor_record(tensor)
    path = root / record["file"]
    if path.exists():
        existing = np.load(path, allow_pickle=False)
        if existing.dtype != np.float32 or existing.shape != SHAPE or not np.array_equal(existing, tensor.numpy()):
            raise ValueError("existing content-addressed embedding was altered")
    else:
        fd, temporary = tempfile.mkstemp(prefix=".embedding-", dir=str(root))
        try:
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, tensor.detach().contiguous().numpy(), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    record["file_sha256"] = file_digest(path)
    return record


def prepare_cache(config, model_manifest, output):
    """Encode all catalog prompts on CPU; callers must hold the idle guard.

    A complete existing manifest is verified and reused. Partial preparation is
    never silently accepted. No model weights are copied into this directory.
    """
    _validate_config(config)
    dual_protocol.verify_model_manifest(model_manifest)
    output = Path(output)
    if (output / "manifest.json").exists():
        return verify_cache(output, config, model_manifest)
    if output.exists() and any(output.iterdir()):
        raise ValueError("unfinished CPU embedding cache requires explicit recovery or a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with _cpu_runtime() as torch:
        encoder_identity = _encoder_identity(config)
        entries = _catalog(config, model_manifest["sha256"], encoder_identity)
        load_started = time.perf_counter()
        pipeline = _load_encoder(config, torch)
        load_seconds = time.perf_counter() - load_started
        for entry in entries.values():
            identity = entry["identity"]
            tick = time.perf_counter()
            positive, negative = pipeline.encode_prompt(prompt=identity["prompt"],
                negative_prompt=identity["negative_prompt"], do_classifier_free_guidance=True,
                num_videos_per_prompt=1, max_sequence_length=512, device="cpu", dtype=torch.float32)
            entry.update(encode_ms=(time.perf_counter() - tick) * 1000,
                         prompt=_write_tensor(output, positive), negative=_write_tensor(output, negative))
        del pipeline
    manifest = dict(schema_version=SCHEMA, model_manifest_sha256=model_manifest["sha256"],
        encoder=encoder_identity, entries=entries, model_path=config["model_path"], revision=config["revision"],
        preparation=dict(device="cpu", gpu_count=0, load_seconds=load_seconds,
                         wall_seconds=time.perf_counter() - started,
                         scope="CPU preparation before GPU allocation; mechanism input only; not online E2E"))
    manifest["sha256"] = digest(manifest)
    atomic_json(output / "manifest.json", manifest)
    return verify_cache(output, config, model_manifest)


def _read_cache(root, config, model_manifest_sha256, expected_sha256=None):
    import numpy as np
    import torch
    _validate_config(config)
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    unsigned = {k: v for k, v in manifest.items() if k != "sha256"}
    if (manifest.get("schema_version") != SCHEMA or manifest.get("sha256") != digest(unsigned)
            or expected_sha256 is not None and manifest["sha256"] != expected_sha256
            or manifest.get("model_manifest_sha256") != model_manifest_sha256
            or manifest.get("model_path") != config["model_path"]
            or manifest.get("revision") != config["revision"]
            or manifest.get("encoder") != _encoder_identity(config)):
        raise ValueError("CPU embedding manifest identity/hash mismatch")
    expected = _catalog(config, model_manifest_sha256, manifest["encoder"])
    if set(manifest.get("entries", {})) != set(expected):
        raise ValueError("CPU embedding catalog is incomplete or contains unknown inputs")
    loaded = {}
    for key, entry in manifest["entries"].items():
        if any(entry.get(field) != expected[key][field] for field in ("identity", "prompt_ids")):
            raise ValueError("CPU embedding prompt identity mismatch")
        tensors = []
        for label in ("prompt", "negative"):
            record = entry.get(label, {})
            name = record.get("file", "")
            if not re.fullmatch(r"[0-9a-f]{64}\.npy", name) or name != record.get("tensor_sha256", "") + ".npy":
                raise ValueError("unsafe CPU embedding file reference")
            path = root / name
            if path.is_symlink() or not path.is_file() or file_digest(path) != record.get("file_sha256"):
                raise ValueError("missing or changed CPU embedding file")
            array = np.load(path, allow_pickle=False)
            if array.dtype != np.float32 or tuple(array.shape) != SHAPE:
                raise ValueError("CPU embedding array dtype/shape mismatch")
            tensor = torch.from_numpy(array)
            if _tensor_record(tensor) != {k: v for k, v in record.items() if k != "file_sha256"}:
                raise ValueError("CPU embedding tensor hash mismatch")
            tensors.append(tensor)
        loaded[key] = tuple(tensors)
    return manifest, loaded


def verify_cache(root, config, model_manifest, *, expected_sha256=None):
    dual_protocol.verify_model_manifest(model_manifest)
    return _read_cache(root, config, model_manifest["sha256"], expected_sha256)[0]


class PreencodedPromptBank(FrozenPromptBank):
    """Explicitly installed immutable inputs; unknown prompts cannot encode."""

    def __init__(self, worker, config, root, *, cache_manifest_sha256, model_manifest_sha256):
        super().__init__(worker)
        manifest, tensors = _read_cache(root, config, model_manifest_sha256, cache_manifest_sha256)
        self.cache_manifest_sha256 = manifest["sha256"]
        self.input_identities = {}
        for key, entry in manifest["entries"].items():
            identity = entry["identity"]
            text = {k: identity[k] for k in ("prompt", "negative_prompt", "guidance_scale")}
            bank_key = digest(text)
            positive, negative = tensors[key]
            payload = EncodedPrompt(text, positive, negative, entry["encode_ms"], self.owner)
            self.entries[bank_key] = dict(payload=payload, prompt=self._hash(positive), negative=self._hash(negative))
            self.input_identities[bank_key] = key

    def warm(self, spec):
        spec = normalize_spec(spec)
        key = digest({k: spec[k] for k in ("prompt", "negative_prompt", "guidance_scale")})
        if key not in self.entries:
            raise ValueError("input absent from frozen CPU embedding cache; live encoding is forbidden")
        identity = super().warm(spec)
        return dict(identity, mode="preencoded_cpu_embeddings_mechanism_only",
                    cache_manifest_sha256=self.cache_manifest_sha256,
                    encoding_identity_sha256=self.input_identities[key])

    def prepare(self, spec):
        state = super().prepare(spec)
        state.events[-1]["timing_scope"] = "CPU encoding before GPU allocation; GPU installation measured"
        return state
