"""Node inventory and dependency checks for the native CLI, without model imports or CUDA."""

import importlib.metadata
import importlib.util
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from . import ROOT


def gpu_inventory():
    """Every GPU nvidia-smi reports; the environment profile picks from these."""
    fields = ["index", "uuid", "name", "memory.total", "driver_version", "compute_cap"]
    for query in (fields, fields[:-1]):  # compute_cap needs a recent driver
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(query), "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            break
    else:
        raise RuntimeError("nvidia-smi failed: " + (result.stderr or result.stdout).strip())
    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        row = dict(index=int(values[0]), uuid=values[1], name=values[2], memory_mib=float(values[3]),
                   driver_version=values[4])
        if len(values) > 5:
            row["compute_capability"] = values[5]
        gpus.append(row)
    return dict(gpus=gpus)


def inspect_environment(request):
    root = Path(request.get("root", ROOT))
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ("openpyxl", "torch", "diffusers")}
    versions = {}
    for name in ("torch", "diffusers", "cache-dit", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    gpu = {"available": False, "detail": "nvidia-smi not found"}
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            gpu = {"available": result.returncode == 0, "detail": (result.stdout + result.stderr).strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            gpu["detail"] = str(error)
    return dict(python=platform.python_version(), python_executable=sys.executable,
                platform=platform.platform(), packages=packages, gpu_stack_versions=versions,
                gpu_venv_present=(root / ".venv-previs").exists(),
                weights_present=(root / ".cache/huggingface/hub").is_dir(), gpu_driver=gpu,
                notice="No model weights loaded, CUDA validation or performance measurements.")
