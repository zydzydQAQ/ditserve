"""Read-only inventory and bounded, supervised migration checks."""

import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from ..util import ROOT, child_environment


VERSIONS = {"torch": "2.10.0+cu126", "torchvision": "0.25.0+cu126",
            "diffusers": "0.36.0", "transformers": "4.57.6", "cache-dit": "1.5.0",
            "accelerate": "1.12.0", "lpips": "0.1.4", "openpyxl": "3.1.5"}


def inventory():
    from ..research.preflight import discover_gpus
    versions = {}
    for name, expected in VERSIONS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = dict(expected=expected, actual=actual, passed=actual == expected)
    memory = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                memory[key] = int(value.split()[0]) * 1024
    return dict(status="observed", gpu_work_started=False, python=sys.version,
                executable=sys.executable, architecture=platform.machine(),
                packages=versions, gpus=discover_gpus(), memory_bytes=memory,
                tools={name: shutil.which(name) for name in ("docker", "nvidia-smi", "g++", "gcc")},
                disk_free_bytes=shutil.disk_usage(ROOT).free,
                scope="Inventory only; no CUDA initialization or model loading")


def select_a100(discovery, requested=None, expected=8):
    if expected < 1:
        raise ValueError("expected GPU count must be positive")
    if discovery.get("status") != "available":
        raise ValueError("GPU discovery unavailable: " + discovery.get("reason", "unknown"))
    devices = discovery["devices"]
    if len({row["uuid"] for row in devices}) != len(devices):
        raise ValueError("duplicate physical GPU UUIDs")
    if requested:
        tokens = [token.strip() for token in requested.split(",")]
        selected = []
        for token in tokens:
            matches = [row for row in devices if token in (str(row["index"]), row["uuid"])]
            if len(matches) != 1:
                raise ValueError("GPU selector absent or ambiguous: " + token)
            selected.append(matches[0])
    else:
        selected = devices
    if len(selected) != expected:
        raise ValueError("expected %d GPUs, found %d" % (expected, len(selected)))
    if len({row["uuid"] for row in selected}) != expected:
        raise ValueError("selectors alias the same physical GPU")
    for row in selected:
        if "A100" not in row["name"].upper() or float(row["compute_capability"]) != 8.0:
            raise ValueError("all selected GPUs must be A100 / SM80: " + row["name"])
        # Both PCIe and SXM 40/80GB devices are accepted; MIG slices are not.
        if row["memory_mib"] < 39000:
            raise ValueError("full A100 devices required; MIG slices are outside this deployment")
    if len({(row["name"].strip().upper(), row["memory_mib"], float(row["compute_capability"]))
            for row in selected}) != 1:
        raise ValueError("selected A100 GPUs must be homogeneous in model, memory and capability")
    return selected


def run_owned(command, *, env=None, timeout=180, gpu_uuids=()):
    """Capture bounded evidence and clean only this invocation's process group."""
    from ..research.isolation import gpu_occupancy
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive")
    if gpu_uuids:
        observed = gpu_occupancy(gpu_uuids)
        if observed["status"] != "exclusive_observed":
            return dict(status="failed", reason="GPU busy or ownership unknown", occupancy=observed,
                        gpu_work_started=False, elapsed_seconds=0, gpu_seconds=0)
    start = time.monotonic()
    reason = None
    cleanup = False
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env or child_environment(),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    reason = "timeout"
                    break
                if gpu_uuids:
                    observed = gpu_occupancy(gpu_uuids, allowed_pgid=process.pid)
                    if observed["status"] != "exclusive_observed":
                        reason = "GPU ownership changed: " + observed["status"]
                        break
                try:
                    process.wait(timeout=min(2.0, remaining))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Also clean orphaned descendants after an unsuccessful launcher exit.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                if process.poll() is None:
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            process.wait(timeout=10)
            cleanup = True
        size = log.tell()
        log.seek(max(0, size - 24000))
        tail = log.read().decode("utf-8", errors="replace")
    elapsed = time.monotonic() - start
    if gpu_uuids:
        after = gpu_occupancy(gpu_uuids)
        if after["status"] != "exclusive_observed":
            reason = reason or "GPU processes remain or final ownership is unknown"
            cleanup = False
    result = dict(status="passed" if process.returncode == 0 and reason is None else "failed",
                  command=command, returncode=process.returncode, reason=reason,
                  elapsed_seconds=elapsed, gpu_seconds=elapsed * len(gpu_uuids),
                  gpu_work_started=bool(gpu_uuids), cleanup_confirmed=cleanup,
                  output_tail=tail, output_bytes=size, output_truncated=size > 24000)
    # Probe output uses a stable prefix because libraries may print other diagnostics.
    result["probe_records"] = []
    for line in tail.splitlines():
        if line.startswith("MIGRATION_RESULT="):
            try:
                probe = json.loads(line.split("=", 1)[1])
                if not isinstance(probe, dict):
                    raise ValueError("probe must be a JSON object")
                result["probe"] = probe
                result["probe_records"].append(probe)
            except ValueError:
                result.update(status="failed", reason="invalid probe result")
    return result


def cpu_check(timeout=600):
    info = inventory()
    errors = []
    if sys.version_info[:2] != (3, 10):
        errors.append("runtime requires Python 3.10")
    if info["architecture"] not in ("x86_64", "amd64"):
        errors.append("this lock targets linux/amd64")
    for name, row in info["packages"].items():
        if not row["passed"]:
            errors.append("version mismatch: " + name)
    # Check all locked direct/transitive versions without importing GPU libraries.
    lock = ROOT / "configs/artifacts-runtime.json"
    if not lock.is_file():
        errors.append("runtime dependency artifact lock missing")
    else:
        for artifact in json.loads(lock.read_text())["artifacts"]:
            name, version = artifact["name"], artifact["version"]
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual = None
            if actual != version:
                errors.append("locked package mismatch: %s (%s != %s)" % (name, actual, version))
    if errors:
        return dict(status="failed", errors=errors, inventory=info, gpu_work_started=False)
    env = dict(child_environment(), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] += os.pathsep + str(ROOT / "tests/python")
    commands = [
        [sys.executable, "-m", "pip", "check"],
        [sys.executable, "-m", "ditserve", "doctor"],
        [sys.executable, "-m", "unittest",
         "test_optimizations_sana.SanaAdapterTests.test_real_cpu_inductor_compiles_and_reuses_graph",
         "test_optimizations_wan.ExternalWanTests.test_real_cpu_inductor_counter_survives_dynamo_backend_unwrapping"],
    ]
    checks = [run_owned(cmd, env=env, timeout=timeout) for cmd in commands]
    for row in checks:
        if re.search(r"\bskipped=[1-9]", row["output_tail"]):
            row.update(status="failed", reason="required acceptance test was skipped")
    # The bridge's stdio subprocess can block too; include it in our owned group
    # and deadline instead of invoking it in this unsupervised parent process.
    core_check = run_owned([sys.executable, "-c", "from ditserve.core import CoreBridge; "
        "core=CoreBridge(); assert core.available(), 'compiled ditserve-core missing'; "
        "result=core.generate(dict(count=2, seed=7)); "
        "assert isinstance(result, list) and len(result)==2 and all(isinstance(x, dict) for x in result), "
        "'invalid core response'"], env=env, timeout=timeout)
    core_check["name"] = "core-protocol"
    checks.append(core_check)
    return dict(status="passed" if all(r["status"] == "passed" for r in checks) else "failed",
                gpu_work_started=False, inventory=info, checks=checks)


def gpu_check(*, devices=None, expected=8, timeout=300):
    from ..research.preflight import discover_gpus
    selected = select_a100(discover_gpus(), devices, expected)
    uuids = [row["uuid"] for row in selected]
    env = dict(child_environment(), CUDA_VISIBLE_DEVICES=",".join(uuids),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    probes = []
    for uuid in uuids:
        one_env = dict(env, CUDA_VISIBLE_DEVICES=uuid)
        row = run_owned([sys.executable, "-m", "ditserve.migration.probes", "cuda"],
                        env=one_env, timeout=timeout, gpu_uuids=[uuid])
        if row["status"] == "passed" and row.get("probe", {}).get("status") != "passed":
            row.update(status="failed", reason="CUDA probe completion evidence missing or failed")
        row["gpu_uuid"] = uuid
        probes.append(row)
        if row["status"] != "passed":
            break
    if all(row["status"] == "passed" for row in probes):
        row = run_owned([sys.executable, "-m", "ditserve.migration.distributed",
            "--nproc", str(expected), "--timeout", str(max(1., timeout - 5.)), "--",
            sys.executable, "-m", "ditserve.migration.probes", "nccl"],
            env=env, timeout=timeout, gpu_uuids=uuids)
        completion = row.get("probe", {})
        if row["status"] == "passed" and (completion.get("status") != "passed"
                or completion.get("ranks") != expected
                or completion.get("completion_ranks") != list(range(expected))):
            row.update(status="failed", reason="NCCL completion evidence is incomplete")
        probes.append(row)
    return dict(status="passed" if all(r["status"] == "passed" for r in probes) else "failed",
                devices=selected, checks=probes, gpu_seconds=sum(r["gpu_seconds"] for r in probes),
                scope="CUDA and NCCL environment acceptance; not SP8 inference or research qualification")
