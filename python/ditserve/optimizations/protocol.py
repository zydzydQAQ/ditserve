"""Immutable protocol and executable snapshots for the additional eight GPU hours."""
import copy
import shutil
from pathlib import Path

from ..research.common import atomic_json, digest, file_digest, read_json

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/serving_optimizations.json"
OUTPUT = ROOT / "results/optimization-screen/run01"
LEDGER = ROOT / "results/optimization-screen/gpu-budget.json"
PARENT = ROOT / "results/research-directions/gpu-budget.json"
PHASES = ("sana_compile", "sana_batch", "wan_compile", "text_encoder", "vae", "cache_k4")
LIMITS = dict(zip(PHASES, (3600., 5400., 7200., 1800., 1800., 5400.)))
LIMITS["reserve"] = 3600.
TOTAL = 28800.


def qualified(row):
    """Only a completed, isolated and charged online result selects a default."""
    if row and row.get("output"):
        audit = Path(row["output"]) / "online-measurement-audit.json"
        if audit.exists():
            try:
                note = read_json(audit)
                if not isinstance(note, dict) or note.get("invalidates_online_qualification") is True:
                    return False
            except (OSError, ValueError, TypeError):
                return False
    return bool(row and row.get("online_qualified") is True and row.get("local_qualified") is True
                and row.get("status") in ("passed", "completed", "stopped") and row.get("returncode") == 0
                and row.get("charge") and not row.get("supervisor_failure")
                and row.get("final_occupancy", {}).get("status") == "exclusive_observed")


def default_config():
    old = read_json(ROOT / "configs/research_directions.json")
    value = {k: copy.deepcopy(old[k]) for k in ("device", "gpu_uuid", "python", "torch_home", "direction1", "direction2")}
    value.update(schema_version="ditserve.optimizations.config.v1", order=list(PHASES),
                 budget_gpu_seconds=TOTAL, stage_budgets_gpu_seconds=LIMITS.copy(),
                 models={k: copy.deepcopy(old["models"][k]) for k in ("wan", "sana")},
                 v100_uuid="GPU-a058066e-bde9-c534-efb1-d2a68c117681",
                 artifact_root=str(ROOT / ".cache/optimization-screen"),
                 old_online=str(ROOT / "results/research-directions/run01/direction1/direction1-online.json"),
                 online_traces=[dict(seed=102, load=.5, pattern="burst"), dict(seed=101, load=.5, pattern="ordinary")],
                 quality=dict(image_lpips=.01, video_lpips=.05, rtol=1e-5, atol=1e-6),
                 compile=dict(backend="inductor", fullgraph=True, dynamic=False, cudagraphs=False),
                 metrics=dict(minimum_local_gain=.05, minimum_goodput_gain=.05, minimum_latency_gain=.10,
                              maximum_goodput_loss=.05, maximum_latency_increase=.05, maximum_video_drop=.05),
                 seeds=[101, 102, 201, 202], repeats=3, policy="block-4",
                 scope="finite paired screening, no p99, no inherited qualification")
    return value


def load_config(path=CONFIG):
    value = read_json(path)
    default = default_config()
    if value != default:
        raise ValueError("optimization protocol is fixed; changed scientific choices require a new authorized protocol")
    return value


def freeze(config, output, phase, attempt):
    output = Path(output)
    protocol = dict(config=config, sha256=digest(config), schema_version="ditserve.optimizations.protocol.v1")
    path = output / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("existing protocol differs")
    atomic_json(path, protocol)
    target = output / "sources" / (phase + "-" + attempt)
    if target.exists():
        raise ValueError("executable snapshot already exists")
    files = list((ROOT / "python/ditserve").rglob("*.py"))
    files += list((ROOT / "tests/python").glob("test_optimizations*.py"))
    files += list((ROOT / "configs").rglob("*.json"))
    hashes = {str(p.relative_to(ROOT)): file_digest(p) for p in files}
    for name, expected in hashes.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        if file_digest(destination) != expected:
            raise ValueError("source changed during snapshot")
    manifest = dict(execution_root=str(target), source_files=hashes, source_sha256=digest(hashes),
                    config_sha256=digest(config), phase=phase, attempt=attempt,
                    parent_ledger_sha256=file_digest(PARENT), old_online_sha256=file_digest(config["old_online"]))
    atomic_json(target / "manifest.json", manifest)
    return manifest
