"""High-level experiment flows on top of the control-plane bridge.

Layering: cli.py parses arguments -> api.py orchestrates -> CoreBridge is the
only place that touches the C++ process. Tests inject a fake bridge here.
"""

import importlib.metadata
import importlib.util
import itertools
import platform
import shutil
import subprocess
import uuid
from pathlib import Path

from . import archive
from .core import CoreBridge, POLICIES
from .report import write_report
from .util import RUNS, ROOT, atomic_json
from .workload import generator_payload, load_trace


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def doctor(core=None):
    """Inspect local prerequisites without loading models or starting the GPU."""
    bridge = core or CoreBridge()
    packages = {name: importlib.util.find_spec(name) is not None for name in ("openpyxl", "torch", "diffusers")}
    gpu_stack = {name: _package_version(name) for name in ("torch", "diffusers", "cache-dit", "transformers", "lpips")}
    gpu = {"available": False, "detail": "nvidia-smi not found"}
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            gpu = {"available": result.returncode == 0, "detail": (result.stdout + result.stderr).strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            gpu["detail"] = str(error)
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "gpu_stack_versions": gpu_stack,
            "gpu_venv_present": (ROOT / ".venv-previs").exists(),
            "weights_present": (ROOT / ".cache" / "huggingface" / "hub").is_dir(),
            "gpu_driver": gpu, "reference_simulation_ready": bridge.available(), "core_binary": str(bridge.binary),
            "notice": "No model weights loaded, CUDA validation or performance measurements."}


def _fresh_destination(output, run_id):
    destination = (Path(output) if output else RUNS / run_id).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("raw output directory must be new or empty: %s" % destination)
    return destination


def _execute(bridge, specs, policies, config, cost_model, destination):
    """Run every policy over one workload and write raw outputs + report."""
    if not policies:
        raise ValueError("at least one policy is required")
    runs = {}
    for policy in policies:
        runs[policy] = bridge.simulate(dict(config, policy=policy), specs, cost_model)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "workload.json", specs)
    atomic_json(destination / "cost_model.json", next(iter(runs.values()))["cost_model"])
    for policy, run in runs.items():
        atomic_json(destination / (policy + ".json"), run)
    return runs, write_report(destination, runs)


def _publish(destination, run_id, results_root, keep_raw):
    results_root = Path(results_root) if results_root else ROOT / "results"
    archived = archive.finalize_run(destination, "synthetic", run_id, results_root=results_root)
    if not keep_raw:
        archive.verify(results_root)
        shutil.rmtree(destination)
    return archived


def simulate(count=64, seed=7, arrival_rate=8.0, pattern="steady", slo_scale=2.0,
             shapes=((512, 512, 1), (768, 768, 1)), step_choices=(20, 30), cache_threshold=0.15,
             trace=None, policies=POLICIES, num_gpus=1, memory_mb=256.0, max_batch=4,
             host_memory_mb=65536.0, cost_model=None, verify_outputs=True,
             output=None, results_root=None, keep_raw=False, core=None):
    """One workload x every policy -> report -> verified archive."""
    bridge = core or CoreBridge()
    specs = load_trace(trace) if trace else bridge.generate(
        generator_payload(count, seed, arrival_rate, pattern, slo_scale, shapes, step_choices, cache_threshold), cost_model)
    run_id = "simulate-" + uuid.uuid4().hex[:12]
    destination = _fresh_destination(output, run_id)
    config = {"num_gpus": num_gpus, "max_batch": max_batch, "memory_mb": memory_mb,
              "host_memory_mb": host_memory_mb, "verify_outputs": verify_outputs}
    runs, report = _execute(bridge, specs, list(policies), config, cost_model, destination)
    archived = _publish(destination, run_id, results_root, keep_raw)
    return {"run_id": run_id, "runs": runs, "report": report, "archive": archived}


def sweep(rates=(2.0, 8.0, 16.0), memories=(128.0, 256.0), gpu_counts=(1, 4), patterns=("steady", "bursty"),
          count=64, seed=7, slo_scale=2.0, shapes=((512, 512, 1), (768, 768, 1)), step_choices=(20, 30),
          cache_threshold=0.15, policies=POLICIES, max_batch=4, host_memory_mb=65536.0,
          cost_model=None, verify_outputs=True, output=None, results_root=None, keep_raw=False, core=None):
    """Replay generated workloads across a grid of synthetic resource budgets."""
    bridge = core or CoreBridge()
    run_id = "sweep-" + uuid.uuid4().hex[:12]
    destination = _fresh_destination(output, run_id)
    aggregate = {}
    for pattern, rate, memory, gpus in itertools.product(patterns, rates, memories, gpu_counts):
        name = "%s-rate%g-memory%g-gpus%d" % (pattern, rate, memory, gpus)
        specs = bridge.generate(generator_payload(count, seed, rate, pattern, slo_scale,
                                                  shapes, step_choices, cache_threshold), cost_model)
        config = {"num_gpus": gpus, "max_batch": max_batch, "memory_mb": memory,
                  "host_memory_mb": host_memory_mb, "verify_outputs": verify_outputs}
        runs, _ = _execute(bridge, specs, list(policies), config, cost_model, destination / name)
        for policy, result in runs.items():
            aggregate[name + "/" + policy] = result
    report = write_report(destination, aggregate)
    archived = _publish(destination, run_id, results_root, keep_raw)
    return {"run_id": run_id, "report": report, "archive": archived}
