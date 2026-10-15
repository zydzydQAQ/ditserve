"""Frozen recipe for CFG-coordinated caching on the existing A100/V100 pair.

Model inventory, content-addressed source archives, dependency identities, and
the original lifetime ledger are shared with the two-GPU protocol. Scientific
qualification is deliberately not inherited from that protocol.
"""

from . import dual_protocol
from .cfg_policy import resolve_policy, validate_catalog
from .common import finite, read_json


ROOT = dual_protocol.ROOT
DEFAULT_CONFIG = ROOT / "configs/research_cfg.json"
DEFAULT_OUTPUT = ROOT / "results/research-cfg/run01"
DEFAULT_LEDGER = dual_protocol.DEFAULT_LEDGER
SCHEMA = "ditserve.research.cfg.protocol.v1"
FROZEN_SCHEMA = "ditserve.research.cfg.frozen.v1"
MODEL_PATH = dual_protocol.MODEL_PATH
REVISION = dual_protocol.REVISION
ARTIFACT_ROOT = dual_protocol.ARTIFACT_ROOT
PHASES = ("preflight", "probe", "calibrate", "mechanism", "report")


def validate_protocol(config):
    expected = read_json(DEFAULT_CONFIG)
    adjustable = {"estimate_wall_seconds", "timeout_wall_seconds", "cleanup_wall_seconds"}
    if (not isinstance(config, dict) or set(config) != set(expected)
            or any(config[key] != expected[key] for key in expected if key not in adjustable)):
        raise ValueError("protocol differs from the fixed CFG scientific recipe")
    if (config["schema_version"] != SCHEMA
            or config["model_path"] != str(dual_protocol.MODEL_PATH)
            or config["revision"] != dual_protocol.REVISION
            or config["artifact_root"] != str(dual_protocol.ARTIFACT_ROOT)
            or config["budget_gpu_seconds"] != 28800
            or config["phases"] != list(PHASES)):
        raise ValueError("fixed CFG model, revision, artifacts, phases or lifetime budget changed")
    for field in ("estimate_wall_seconds", "timeout_wall_seconds"):
        if set(config[field]) != set(expected[field]):
            raise ValueError("missing CFG stage estimates/timeouts")
        for value in config[field].values():
            finite(value, field, positive=True)
    finite(config["cleanup_wall_seconds"], "cleanup reserve", positive=True)
    validate_catalog(config)
    for policy in config["recipes"]:
        resolve_policy(config, policy["id"])
    return config


def attention_for_device(device):
    """Pinned backend only; no retry or silent fallback to another backend."""
    return dual_protocol.attention_for_device(device, efficient_qualified=True)


def freeze_protocol(config_path, output, ledger=DEFAULT_LEDGER, *, model_manifest, root=ROOT,
                    prototype_run=None, refinement_run=None, cpu_embeddings=None):
    config = validate_protocol(read_json(config_path))
    continuation = None
    refinement = None
    if prototype_run is not None and refinement_run is not None:
        raise ValueError("prototype and refinement continuations are mutually exclusive")
    if refinement_run is not None:
        from pathlib import Path
        from .cfg_refinement import build_refinement, verify_cpu_binding
        refinement = build_refinement(refinement_run, config=config, ledger=ledger)
        continuation = read_json(Path(refinement_run) / "protocol.json")["prototype_continuation"]
        verify_cpu_binding(cpu_embeddings, config=config, model_manifest=model_manifest)
    elif cpu_embeddings is not None:
        raise ValueError("CPU embedding binding is restricted to implementation refinement")
    if prototype_run is not None:
        from .cfg_continuation import build_continuation
        continuation = build_continuation(prototype_run, config=config, ledger=ledger)
    return dual_protocol.freeze_protocol(config_path, output, ledger, model_manifest=model_manifest,
        root=root, cfg_continuation=continuation, cfg_refinement=refinement, cpu_embeddings=cpu_embeddings)


def verify_protocol(path, *, root=ROOT, verify_current=True, rehash_model=False):
    if read_json(path).get("schema_version") != FROZEN_SCHEMA:
        raise ValueError("CFG qualification requires a new CFG frozen protocol")
    return dual_protocol.verify_protocol(path, root=root, verify_current=verify_current,
                                         rehash_model=rehash_model)
