"""Node inventory and readiness checks for the native CLI, without model imports.

``ditserve doctor --env NAME`` passes the environment profile and the GPUs it
selected; CUDA is touched only with --allow-gpu, and then only the selected
GPUs are visible to this process.
"""

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from . import ROOT

# The pinned stack the Wan worker refuses to run without (models/wan/runtime.py).
PINNED = {"torch": "2.10.0+cu126", "diffusers": "0.36.0", "cache-dit": "1.5.0"}
MIN_DRIVER = 560  # CUDA 12.6 user-space libraries in the torch wheels


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


def _check(checks, name, ok, detail, *, warn=False):
    checks.append(dict(name=name, status="pass" if ok else "warn" if warn else "fail", detail=detail))


def _meminfo_gib(key):
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return None


def _host_checks(checks, env, root):
    host = env.get("host", {})
    gpus = env.get("gpus", {}).get("count", 1)
    ram = _meminfo_gib("MemTotal")
    # Each Wan worker keeps the FP32 UMT5 text encoder in host memory.
    needed = max(host.get("min_ram_gib", 0), gpus * host.get("text_encoder_gib_per_wan_worker", 0))
    _check(checks, "host_memory", ram is not None and ram >= needed,
           "%.0f GiB total, %d GiB needed for %d Wan workers" % (ram or 0, needed, gpus))
    try:
        stat = os.statvfs("/dev/shm")
        shm = stat.f_frsize * stat.f_blocks / 2 ** 30
    except OSError:
        shm = 0.
    _check(checks, "shared_memory", shm >= host.get("min_shm_gib", 0),
           "/dev/shm %.1f GiB, %s GiB needed (docker --shm-size or --ipc=host)" % (shm, host.get("min_shm_gib", 0)))
    threads = int(env.get("worker_environment", {}).get("OMP_NUM_THREADS", "1"))
    cores = os.cpu_count() or 1
    _check(checks, "cpu_cores", cores >= threads * gpus,
           "%d cores for %d Wan workers x %d prompt-encoding threads" % (cores, gpus, threads), warn=True)
    try:
        namespaces = [line for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("NSpid:")]
        nested = bool(namespaces) and len(namespaces[0].split()) > 2
    except OSError:
        nested = False
    _check(checks, "host_pid_namespace", not nested,
           "GPU isolation matches nvidia-smi PIDs to workers; run containers with --pid=host" if nested else "host PIDs visible")
    for name, model in env.get("models", {}).items():
        path = Path(model["path"])
        if not path.is_absolute():
            path = Path(env.get("models_root", "/models")) / path
        _check(checks, "model:" + name, (path / "model_index.json").is_file(),
               "%s (verify files with sha256sum -c configs/models/%s.sha256)" % (path, name))
    results = Path(root) / "results"
    writable = os.access(results if results.exists() else root, os.W_OK)
    _check(checks, "results_writable", writable, str(results))


def _gpu_checks(checks, env, selected):
    try:
        import torch
    except ImportError as error:
        _check(checks, "torch_cuda", False, str(error))
        return
    available = torch.cuda.is_available()
    _check(checks, "torch_cuda", available and torch.cuda.device_count() == len(selected),
           "%d visible CUDA devices for %d selected" % (torch.cuda.device_count() if available else 0, len(selected)))
    if not available:
        return
    arches = torch.cuda.get_arch_list()
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        # A cubin for sm_XY runs on sm_XZ with Z >= Y; PTX would JIT for newer majors.
        native = [arch for arch in arches if arch.startswith("sm_%d" % major) and int(arch[3 + len(str(major)):]) <= minor]
        _check(checks, "cuda_arch:%d" % index, bool(native),
               "%s sm_%d%d; wheel kernels %s" % (torch.cuda.get_device_name(index), major, minor, ",".join(arches)))
    if torch.cuda.device_count() > 1:
        peers = {"%d->%d" % (a, b): torch.cuda.can_device_access_peer(a, b)
                 for a in range(torch.cuda.device_count()) for b in range(torch.cuda.device_count()) if a != b}
        disabled = env.get("worker_environment", {}).get("NCCL_P2P_DISABLE") == "1"
        _check(checks, "peer_access", all(peers.values()) or disabled,
               "%s%s" % (peers, "; NCCL_P2P_DISABLE=1 routes collectives through host memory" if disabled else ""), warn=True)
    try:
        _check(checks, "nccl", True, ".".join(map(str, torch.cuda.nccl.version())))
    except Exception as error:  # NCCL absence only matters for SP phases
        _check(checks, "nccl", False, str(error), warn=True)


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
    result = dict(python=platform.python_version(), python_executable=sys.executable,
                  platform=platform.platform(), packages=packages, gpu_stack_versions=versions, gpu_driver=gpu,
                  notice="No model weights loaded; GPU checks run only with --allow-gpu.")
    env = request.get("env")
    if env is None:
        return result
    checks = []
    if request.get("selection_error"):
        _check(checks, "gpu_selection", False, request["selection_error"])
    else:
        _check(checks, "gpu_selection", True, ", ".join(request.get("selected", [])))
        drivers = {row.get("driver_version") for row in request.get("inventory", [])}
        major = min((int(str(value).split(".")[0]) for value in drivers if value), default=0)
        _check(checks, "driver", major >= MIN_DRIVER, "driver %s; CUDA 12.6 wheels need >= %d" % (",".join(sorted(drivers)), MIN_DRIVER))
    for name, expected in PINNED.items():
        _check(checks, "package:" + name, versions.get(name) == expected, "%s (pinned %s)" % (versions.get(name), expected))
    _check(checks, "package:openpyxl", packages["openpyxl"] or (root / ".cache/tools").is_dir(), "report writer")
    _host_checks(checks, env, root)
    if request.get("gpu_checks"):
        _gpu_checks(checks, env, request.get("selected", []))
    else:
        checks.append(dict(name="gpu_runtime", status="skip", detail="add --allow-gpu to initialize CUDA on the selected GPUs"))
    result.update(environment=env.get("name"), checks=checks,
                  status="failed" if any(check["status"] == "fail" for check in checks) else "ok")
    return result
