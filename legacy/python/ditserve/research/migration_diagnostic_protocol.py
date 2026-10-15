"""Independent, cache-off diagnosis of the sealed heterogeneous-path failure.

The original CFG configuration is an unchanged worker/encoder input. The separate
diagnostic recipe permits only two kettle seeds, one step-16 transfer, and the
original/MATH backend comparison. No CFG qualification gate is imported.
"""

import copy
import importlib.metadata
import os
from pathlib import Path
import shutil

from . import dual_protocol
from .common import atomic_json, digest, file_digest, finite, read_json
from .dual_evidence import read_records
from .protocol import dependency_identity, quality_assets


ROOT = dual_protocol.ROOT
DEFAULT_CONFIG = ROOT / "configs/research_migration_diagnostic.json"
DEFAULT_OUTPUT = ROOT / "results/research-migration-diagnostic/run01"
DEFAULT_PARENT = ROOT / "results/research-cfg/run03"
DEFAULT_LEDGER = dual_protocol.DEFAULT_LEDGER
MODEL_PATH = dual_protocol.MODEL_PATH
REVISION = dual_protocol.REVISION
ARTIFACT_ROOT = dual_protocol.ARTIFACT_ROOT
SCHEMA = "ditserve.research.migration_diagnostic.protocol.v1"
FROZEN_SCHEMA = "ditserve.research.migration_diagnostic.frozen.v1"
PARENT_SCHEMA = "ditserve.research.migration_diagnostic.failure_anchor.v1"
PHASES = ("preflight", "probe", "calibrate", "mechanism", "report")


def _sealed(value):
    return isinstance(value, dict) and value.get("sha256") == digest(
        {key: child for key, child in value.items() if key != "sha256"})


def _seal(value):
    return dict(value, sha256=digest(value))


def validate_protocol(config):
    expected = read_json(DEFAULT_CONFIG)
    adjustable = {"estimate_wall_seconds", "timeout_wall_seconds", "cleanup_wall_seconds"}
    if (not isinstance(config, dict) or set(config) != set(expected) or
            any(config[key] != expected[key] for key in expected if key not in adjustable)):
        raise ValueError("protocol differs from the fixed cache-off migration diagnostic recipe")
    for field in ("estimate_wall_seconds", "timeout_wall_seconds"):
        if not isinstance(config[field], dict) or set(config[field]) != {"original", "math"}:
            raise ValueError("both original and MATH diagnostic estimates are required")
        for value in config[field].values():
            finite(value, field, positive=True)
    finite(config["cleanup_wall_seconds"], "diagnostic cleanup reserve", positive=True)
    return config


def attention_for_device(device, comparison="original"):
    if comparison not in ("original", "math"):
        raise ValueError("unknown diagnostic backend comparison")
    native = dual_protocol.attention_for_device(device, efficient_qualified=True)
    return native if comparison == "original" else "math"


def source_files(root=ROOT):
    root = Path(root)
    files = dual_protocol.source_files(root)
    path = root / "configs/research_migration_diagnostic.json"
    if not path.is_file():
        raise ValueError("diagnostic source archive requires its independent recipe")
    files[str(path.relative_to(root))] = path
    return files


def _verify_archive(path, frozen, *, root=ROOT, verify_current=False):
    path = Path(path)
    if not _sealed(frozen) or frozen.get("source_sha256") != digest(frozen.get("source_files")):
        raise ValueError("invalid frozen diagnostic or parent source checksum")
    archive_hash = digest(dict(source_files=frozen["source_files"], dependencies=frozen["dependencies"]))
    archive = path.parent.parent / "source-snapshots" / archive_hash
    if (frozen.get("source_archive") != dict(sha256=archive_hash, relative_path=os.path.relpath(archive, path.parent))
            or not (path.parent / "sources").is_symlink()
            or (path.parent / "sources").resolve() != archive.resolve()):
        raise ValueError("invalid content-addressed diagnostic or parent archive reference")
    if verify_current and set(source_files(root)) != set(frozen["source_files"]):
        raise ValueError("current diagnostic source file set changed")
    for name, expected in frozen["source_files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe archived source path")
        if file_digest(archive / relative) != expected:
            raise ValueError("archived source changed: " + name)
        if verify_current and file_digest(Path(root) / relative) != expected:
            raise ValueError("current diagnostic source changed: " + name)
    for package, identity in frozen["dependencies"].items():
        if Path(package).name != package:
            raise ValueError("unsafe dependency package path")
        for name, expected in identity["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe archived dependency path")
            if file_digest(archive / "dependencies" / package / relative) != expected:
                raise ValueError("archived diagnostic dependency changed: " + name)


def _parent_failure(parent_run, *, ledger):
    """Verify only sealed provenance and the two off-path observations."""
    root = Path(parent_run).resolve()
    parent = read_json(root / "protocol.json")
    if parent.get("schema_version") != "ditserve.research.cfg.frozen.v1":
        raise ValueError("diagnostic anchor requires a closed CFG run")
    _verify_archive(root / "protocol.json", parent)
    from .cfg_protocol import validate_protocol as validate_cfg_recipe
    validate_cfg_recipe(parent["config"])
    if Path(parent["ledger"]).resolve() != Path(ledger).resolve():
        raise ValueError("diagnostic must retain the parent lifetime ledger")
    budget = read_json(ledger)
    if (budget.get("active") or {}).get("protocol_sha256") == parent["sha256"]:
        raise ValueError("diagnostic parent has an unsettled allocation")
    if ((root / "active-unit.json").exists() and read_json(root / "active-unit.json").get("status") == "running"
            or not (root / "pending-references.json").is_file()
            or read_json(root / "pending-references.json").get("pending") is not False):
        raise ValueError("diagnostic parent has active or pending work")

    def bound(name, status):
        row = read_json(root / (name + ".json"))
        if (not _sealed(row) or row.get("status") != status or
                row.get("protocol_sha256") != parent["sha256"] or
                row.get("source_sha256") != parent["source_sha256"]):
            raise ValueError("diagnostic parent requires a closed bound " + name)
        return row

    terminal = bound("terminal", "completed")
    failure = bound("migration", "quality_stopped")
    if (failure.get("failed_reference_case") != "dev-kettle-102" or failure.get("qualified_actions") != []
            or failure.get("reason") != "cache-off device-path reference failed common reference"):
        raise ValueError("diagnostic anchor is not the fixed cache-off path-reference failure")
    attempt = read_json(root / "migration-attempt.json")
    job = read_json(root / "migration-job.json")
    if (not _sealed(attempt) or attempt.get("status") != "completed" or attempt.get("returncode") != 0
            or attempt.get("protocol_sha256") != parent["sha256"]
            or job.get("protocol_sha256") != parent["sha256"] or job.get("source_sha256") != parent["source_sha256"]
            or job.get("budget_stage") != "migration" or job.get("phase") != "migration"):
        raise ValueError("diagnostic parent migration supervisor did not settle")
    sessions = [session for session in budget.get("sessions", []) if session.get("session_id") == job.get("session_id")]
    devices = [device["uuid"] for device in job.get("physical_devices", [])]
    if (len(sessions) != 1 or sessions[0].get("status") != "completed" or sessions[0].get("stage") != "migration"
            or sessions[0].get("protocol_sha256") != parent["sha256"] or len(devices) != 2
            or sessions[0].get("devices") != devices or sessions[0].get("gpu_count") != 2):
        raise ValueError("diagnostic parent migration ledger session is not settled")
    finite(sessions[0].get("gpu_seconds"), "settled diagnostic parent cost", positive=True)
    rows, observations = read_records(root, parent), []
    for seed in (101, 102):
        case_id = "dev-kettle-" + str(seed)
        matches = [row for row in rows if row.get("case_id") == case_id + ("-full_migrate" if seed == 101 else "")
                   and row.get("type") == ("cfg_migration_quality" if seed == 101 else "cfg_migration_reference")]
        if len(matches) != 1:
            raise ValueError("diagnostic parent lacks unique paired kettle reference evidence")
        row = matches[0]
        if (row.get("performance_valid") is not True or row.get("prompt_id") != "kettle" or row.get("seed") != seed
                or row.get("status") != ("completed" if seed == 101 else "failed")):
            raise ValueError("diagnostic parent kettle evidence is incomplete or invalidated")
        quality = row.get("path_reference_quality", {})
        mean = finite(quality.get("lpips_mean"), "parent path LPIPS")
        if (quality.get("status") != "ok" or quality.get("lpips_status") != "ok"
                or quality.get("finite") is not True or quality.get("range_valid") is not True
                or (mean <= .05) != (seed == 101)):
            raise ValueError("diagnostic parent passing/failing reference roles differ")
        for name, route, backend, action in (
                ("common_reference", ["a100"], ["flash"], "stay"),
                ("path_reference", ["v100", "a100"], ["efficient", "flash"], "full_migrate")):
            run = row.get(name, {})
            identity = run.get("identity", {})
            if (identity.get("recipe") != "off" or identity.get("implementation") != "native"
                    or identity.get("path") != route or identity.get("attention_path") != backend
                    or identity.get("boundary") != 16 or identity.get("action") != action
                    or identity.get("input", {}).get("prompt_id") != "kettle" or identity["input"].get("seed") != seed):
                raise ValueError("diagnostic anchor references must be native cache-off fixed-device paths")
            prefix = "reference" if name == "common_reference" else "candidate"
            if (quality.get(prefix + "_identity_sha256") != digest(identity)
                    or quality.get(prefix + "_output_sha256") != run.get("output_sha256")):
                raise ValueError("diagnostic parent quality is not bound to its path outputs")
        unit = read_json(root / "units" / (row["unit_id"] + ".json"))
        observations.append(dict(case_id=case_id, record_sha256=row["sha256"], unit_id=row["unit_id"],
                                 unit_sha256=unit["sha256"], lpips_mean=mean))
    anchor = _seal(dict(schema_version=PARENT_SCHEMA, parent_run=str(root),
        parent_protocol_sha256=parent["sha256"], parent_source_sha256=parent["source_sha256"],
        parent_migration_sha256=failure["sha256"], parent_terminal_sha256=terminal["sha256"],
        parent_attempt_sha256=attempt["sha256"], parent_job_sha256=digest(job),
        parent_session_id=sessions[0]["session_id"], parent_session_sha256=digest(sessions[0]),
        observations=observations, purpose="diagnostic_input_selection_only", quality_qualified=False,
        performance_qualified=False, inherit_formal_evidence=False, raw_reanalysis=False))
    return parent, anchor


def _budget_binding(ledger):
    budget = read_json(ledger)
    if budget.get("limit_gpu_seconds") != 28800.:
        raise ValueError("diagnostic must retain the 28,800 GPU-second lifetime cap")
    limit = finite(budget.get("stage_limits_gpu_seconds", {}).get("migration"), "migration stage cap", positive=True)
    return dict(ledger=str(Path(ledger).resolve()), stage="migration", limit_gpu_seconds=28800.,
                stage_limit_gpu_seconds=limit, reallocation_allowed=False, borrowing_allowed=False)


def freeze_protocol(config_path, output, ledger=DEFAULT_LEDGER, *, parent_run=DEFAULT_PARENT,
                    model_manifest, root=ROOT):
    diagnostic = validate_protocol(read_json(config_path))
    parent, anchor = _parent_failure(parent_run, ledger=ledger)
    if model_manifest != parent["model_manifest"]:
        raise ValueError("diagnostic model manifest must match its recorded failure and CPU inputs")
    dual_protocol.verify_model_manifest(model_manifest)
    from .cfg_refinement import verify_cpu_binding
    cpu = copy.deepcopy(parent.get("cpu_embeddings"))
    verify_cpu_binding(cpu, config=parent["config"], model_manifest=model_manifest)
    output = Path(output).resolve()
    files = source_files(root)
    hashes = {name: file_digest(path) for name, path in files.items()}
    dependencies = dependency_identity()
    archive_hash = digest(dict(source_files=hashes, dependencies=dependencies))
    archive = output.parent / "source-snapshots" / archive_hash
    relative_archive = os.path.relpath(archive, output)
    payload = _seal(dict(schema_version=FROZEN_SCHEMA, config=copy.deepcopy(parent["config"]),
        diagnostic=diagnostic, parent_failure=anchor, cpu_embeddings=cpu, model_manifest=model_manifest,
        model_verification="parent SHA256 manifest; file identity checked at freeze and each launch",
        source_files=hashes, source_sha256=digest(hashes), dependencies=dependencies, quality_assets=quality_assets(),
        ledger=str(Path(ledger).resolve()), budget_binding=_budget_binding(ledger),
        source_archive=dict(sha256=archive_hash, relative_path=relative_archive)))
    frozen_path = output / "protocol.json"
    if frozen_path.exists():
        if read_json(frozen_path) != payload:
            raise ValueError("diagnostic protocol/source/input changed; use a new output directory")
        return verify_protocol(frozen_path, root=root)
    for name, source in files.items():
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_digest(target) != hashes[name]:
            raise ValueError("existing diagnostic source archive differs")
        if not target.exists():
            shutil.copyfile(source, target)
    for package, identity in dependencies.items():
        distribution = importlib.metadata.distribution(package) if identity["files"] else None
        for name, expected in identity["files"].items():
            target = archive / "dependencies" / package / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and file_digest(target) != expected:
                raise ValueError("existing diagnostic dependency archive differs")
            if not target.exists():
                shutil.copyfile(distribution.locate_file(name), target)
    output.mkdir(parents=True, exist_ok=True)
    link = output / "sources"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != archive.resolve():
            raise ValueError("diagnostic source archive link differs")
    else:
        link.symlink_to(relative_archive, target_is_directory=True)
    atomic_json(frozen_path, payload)
    return payload


def verify_protocol(path, *, root=ROOT, verify_current=True, rehash_model=False):
    path = Path(path)
    frozen = read_json(path)
    if frozen.get("schema_version") != FROZEN_SCHEMA:
        raise ValueError("a dedicated migration diagnostic frozen protocol is required")
    validate_protocol(frozen.get("diagnostic"))
    _verify_archive(path, frozen, root=root, verify_current=verify_current)
    anchor = frozen.get("parent_failure", {})
    if not _sealed(anchor) or anchor.get("schema_version") != PARENT_SCHEMA:
        raise ValueError("missing or invalid diagnostic failure anchor")
    parent, expected = _parent_failure(anchor.get("parent_run", ""), ledger=frozen["ledger"])
    if (anchor != expected or frozen.get("config") != parent["config"]
            or frozen.get("model_manifest") != parent["model_manifest"]
            or frozen.get("cpu_embeddings") != parent.get("cpu_embeddings")):
        raise ValueError("diagnostic inputs no longer match their sealed failure anchor")
    if frozen.get("budget_binding") != _budget_binding(frozen["ledger"]):
        raise ValueError("diagnostic lifetime ledger or migration stage cap changed")
    from .cfg_refinement import verify_cpu_binding
    verify_cpu_binding(frozen["cpu_embeddings"], config=frozen["config"],
                       model_manifest=frozen["model_manifest"], verify_arrays=verify_current)
    if verify_current:
        dual_protocol.verify_model_manifest(frozen["model_manifest"], rehash=rehash_model)
        if dependency_identity() != frozen["dependencies"] or quality_assets() != frozen["quality_assets"]:
            raise ValueError("current diagnostic dependencies or quality assets changed")
    return frozen
