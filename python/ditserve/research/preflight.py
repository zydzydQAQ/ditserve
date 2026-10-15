"""Read-only hardware/dependency discovery and CPU contract attestation."""

import csv
import importlib.metadata
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .common import atomic_json, digest
from .protocol import ROOT


def discover_gpus(run=subprocess.run):
    try:
        result = run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,compute_cap",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "devices": [], "reason": str(error)}
    if result.returncode:
        return {"status": "unavailable", "devices": [],
                "reason": (result.stderr + result.stdout).strip()}
    devices = []
    try:
        for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            index, uuid, name, memory, capability = [cell.strip() for cell in row]
            devices.append(dict(index=int(index), uuid=uuid, name=name,
                                memory_mib=int(memory), compute_capability=capability))
        if not devices or len({row["uuid"] for row in devices}) != len(devices):
            raise ValueError("empty or duplicate GPU list")
    except (TypeError, ValueError) as error:
        return {"status": "unavailable", "devices": [], "reason": "invalid GPU inventory: " + str(error)}
    return {"status": "available", "devices": devices}


def select_devices(discovery, requested):
    requested = list(requested)
    if len(requested) not in (1, 4) or len(set(requested)) != len(requested):
        raise ValueError("select one GPU for the single-GPU pilot or four unique GPUs for SP")
    if discovery["status"] != "available":
        return {"status": "blocked", "devices": [], "reason": discovery.get("reason", "driver unavailable")}
    selected = []
    for token in requested:
        matches = [d for d in discovery["devices"] if token in (str(d["index"]), d["uuid"])]
        if len(matches) != 1:
            return {"status": "blocked", "devices": [], "reason": "device absent or ambiguous: " + token}
        selected.append(matches[0])
    if len({d["uuid"] for d in selected}) != len(selected):
        raise ValueError("device aliases select the same physical GPU twice")
    if len(selected) == 4 and len({(d["name"], d["memory_mib"], d["compute_capability"]) for d in selected}) != 1:
        return {"status": "blocked", "devices": selected, "reason": "four homogeneous GPUs are required"}
    if any(float(d["compute_capability"]) < 8 for d in selected):
        return {"status": "blocked", "devices": selected, "reason": "pinned FP16 Flash path requires compute capability >= 8"}
    return {"status": "ready", "devices": selected}


def cpu_validation(frozen, output, run=subprocess.run):
    """Run real contracts, including four-process Gloo, before any GPU work."""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "python"), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests/python"),
               "-p", "test_research*.py", "-v"]
    start = time.monotonic()
    try:
        result = run(command, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=240)
        log = result.stdout + result.stderr
        count = re.search(r"Ran (\d+) tests?", log)
        skipped = re.search(r"skipped=(\d+)", log)
        passed = result.returncode == 0 and count is not None and int(count[1]) > 0 and not skipped
        record = {"status": "passed" if passed else "failed", "returncode": result.returncode,
                  "test_count": int(count[1]) if count else 0,
                  "skipped": int(skipped[1]) if skipped else 0, "log": log}
    except subprocess.TimeoutExpired as error:
        record = {"status": "failed", "reason": "CPU/Gloo validation timed out", "log": str(error)}
    record.update(source_sha256=frozen["source_sha256"], protocol_sha256=frozen["sha256"],
                  elapsed_seconds=time.monotonic() - start, command=command)
    record["sha256"] = digest(record)
    atomic_json(Path(output) / "cpu-validation.json", record)
    (Path(output) / "cpu-validation.log").write_text(record["log"])
    return record


def inspect_environment(frozen, requested, model_path=None, run=subprocess.run):
    from ..gpu.worker import DEFAULT_MODEL_PATH
    from .isolation import gpu_occupancy
    selected = select_devices(discover_gpus(run), requested)
    occupancy = (gpu_occupancy([d["uuid"] for d in selected["devices"]], run=run)
                 if selected["status"] == "ready" else {"status": "not_checked"})
    versions = {}
    for name, expected in frozen["config"]["dependencies"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = {"actual": actual, "expected": expected, "passed": actual == expected}
    path = Path(model_path or DEFAULT_MODEL_PATH).resolve()
    model = {"path": str(path), "available": (path / "model_index.json").is_file(),
             "revision_matches": path.name == frozen["config"]["revision"]}
    quality = frozen.get("quality_assets", {})
    quality_ready = all(quality.get(key, {}).get("sha256") for key in ("alexnet", "lpips_alex_v01"))
    host_memory = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                host_memory[key + "_bytes"] = int(value.split()[0]) * 1024
    reasons = []
    if selected["status"] != "ready":
        reasons.append(selected["reason"])
    elif occupancy["status"] != "exclusive_observed":
        reasons.append("selected GPUs are occupied or process ownership cannot be verified")
    if not all(v["passed"] for v in versions.values()):
        reasons.append("pinned dependency versions unavailable")
    if not model["available"] or not model["revision_matches"]:
        reasons.append("fixed local model revision unavailable")
    return {"status": "blocked" if reasons else "ready", "reasons": reasons,
            "selection": selected, "dependencies": versions, "model": model,
            "gpu_occupancy": occupancy,
            "offline_quality_weights_ready": bool(quality_ready), "host_memory": host_memory,
            "gpu_work_started": False, "scope": "discovery only; CUDA/Flash correctness requires charged execution"}
