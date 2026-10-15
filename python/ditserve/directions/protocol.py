"""Freeze scientific choices separately from per-phase executable snapshots."""

import shutil
from pathlib import Path

from ..research.common import atomic_json, digest, file_digest, read_json
from .budget import LIMITS, TOTAL


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research_directions.json"
DEFAULT_OUTPUT = ROOT / "results/research-directions/run01"
DEFAULT_LEDGER = ROOT / "results/research-directions/gpu-budget.json"
PARENT_LEDGER = ROOT / "results/research-pilot/gpu-budget.json"


def load_config(path=DEFAULT_CONFIG):
    value = read_json(path)
    if (value.get("budget_gpu_seconds") != TOTAL or
            value.get("stage_budgets_gpu_seconds") != LIMITS or
            value.get("order") != [1, 2, 3]):
        raise ValueError("the approved sequence and 24 GPU-hour allocation are fixed")
    if value.get("device") != "cuda:0":
        raise ValueError("workers see the selected physical A100 as cuda:0")
    return value


def freeze_config(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = dict(schema_version="ditserve.directions.protocol.v1", config=config,
                  configuration_sha256=digest(config))
    record["sha256"] = digest(record)
    path = output / "protocol.json"
    if path.exists() and read_json(path) != record:
        raise ValueError("scientific protocol changed: use a new run, retaining the same allocation")
    if not path.exists():
        atomic_json(path, record)
    return record


def freeze_phase(output, direction, phase, config):
    """Each phase is immutable; fixes require a fresh run rather than overwriting evidence."""
    output = Path(output)
    destination = output / "sources" / ("direction%d-%s" % (direction, phase))
    if destination.exists():
        raise ValueError("this phase already has a frozen attempt; use a new run")
    files = sorted((ROOT / "python/ditserve").rglob("*.py"))
    files += sorted((ROOT / "tests/python").glob("test_directions*.py"))
    files += sorted((ROOT / "configs").rglob("*.json"))
    files += [ROOT / name for name in (
        "results/research-pilot/run05/report.json", "results/research-cfg/run03/report.json",
        "results/research-migration-diagnostic/run01/summary.json") if (ROOT / name).exists()]
    hashes = {str(p.relative_to(ROOT)): file_digest(p) for p in files}
    for name in hashes:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if file_digest(target) != hashes[name]:
            raise ValueError("source changed while freezing: " + name)
    record = dict(direction=direction, phase=phase, config_sha256=digest(config),
                  source_files=hashes, source_sha256=digest(hashes),
                  execution_root=str(destination.resolve()))
    record["asset_manifests"] = {}
    for name, model in config["models"].items():
        manifest = model.get("verification_manifest")
        if manifest and Path(manifest).is_file():
            value = read_json(manifest)
            target = destination / "assets" / (name + ".json")
            atomic_json(target, value)
            record["asset_manifests"][name] = dict(source_path=manifest,
                source_sha256=file_digest(manifest), archived_sha256=file_digest(target))
    record["sha256"] = digest(record)
    atomic_json(destination / "manifest.json", record)
    return record
