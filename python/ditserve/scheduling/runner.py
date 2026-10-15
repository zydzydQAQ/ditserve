"""Supervise isolated, archived scheduling experiments on one selected GPU."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from ..directions.runner import model_inventory, stop_owned_group
from ..research.common import atomic_json, digest, file_digest, finite, read_json
from ..research.isolation import gpu_occupancy
from ..research.preflight import discover_gpus
from .budget import Allocation, BudgetStop, TOTAL


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research_scheduling.json"
DEFAULT_OUTPUT = ROOT / "results/research-scheduling/run01"
DEFAULT_LEDGER = ROOT / "results/research-scheduling/gpu-budget.json"
PHASES = ("calibrate", "pilot", "screen", "validate", "legacy")
PREREQUISITES = {"pilot": "calibrate", "screen": "pilot", "validate": "screen"}
DEVICE_FIELDS = ("gpu_uuid", "device", "wan_attention", "sana_profile", "python")


def load_config(path=DEFAULT_CONFIG):
    config = read_json(path)
    if config.get("budget_gpu_seconds") != TOTAL:
        raise ValueError("the scheduling allocation is fixed at 12 GPU hours")
    if set(config.get("models", {})) != {"wan", "sana"}:
        raise ValueError("scheduling requires the pinned Wan and Sana models")
    return config


def bind_protocol(config, output, ledger):
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    record = dict(schema_version="ditserve.scheduling.protocol.v1", config=config,
                  config_sha256=digest(config), ledger_path=str(ledger), budget_gpu_seconds=TOTAL)
    record["sha256"] = digest(record)
    path = output / "protocol.json"
    if path.exists():
        if read_json(path) != record:
            raise ValueError("run configuration or ledger changed; preserve this run and use a new output")
    else:
        atomic_json(path, record)
    return record


def prepare(config, output, ledger):
    if config.get("budget_gpu_seconds") != TOTAL:
        raise ValueError("the scheduling allocation is fixed at 12 GPU hours")
    # Existing protocol identity is checked before creating a replacement ledger.
    protocol = bind_protocol(config, output, ledger)
    with Allocation(ledger) as budget:
        return dict(status="prepared", gpu_work_started=False, protocol=protocol,
                    allocation_id=budget.data["allocation_id"], remaining_gpu_seconds=budget.remaining())


def phase_result(output, phase):
    path = Path(output) / phase / "supervisor.json"
    return read_json(path) if path.exists() else dict(status="not_run", eligible=False)


def _locked_runtime(output, config):
    if output is None:
        return None
    root = Path(output) / "calibrate"
    path = root / "runtime-config.json"
    if not path.exists():
        return None
    runtime = read_json(path)
    if runtime.get("main_config_sha256") != digest(config):
        raise ValueError("calibration runtime belongs to another scientific configuration")
    supervisor = phase_result(output, "calibrate")
    if supervisor.get("status") != "passed" or supervisor.get("eligible") is False:
        raise ValueError("calibration attempt did not qualify; preserve it and use a new run")
    if supervisor.get("runtime_config_sha256") != digest(runtime):
        raise ValueError("calibration runtime changed after its supervised measurement")
    return runtime


def _device_profile(device):
    name = device.get("name", "").upper()
    capability = float(device.get("compute_capability", 0))
    if "A100" in name and capability >= 8:
        return "a100_bf16", "flash"
    if "V100" in name and capability == 7:
        return "v100_fp32", "efficient"
    return None


def inspect(config, output=None, phase=None):
    """Read-only selection restricted to the explicitly configured GPU families."""
    discovery = discover_gpus()
    preferences = config.get("device_preference", ["A100", "V100"])
    if (not isinstance(preferences, list) or not preferences or
            len(set(preferences)) != len(preferences) or any(p not in ("A100", "V100") for p in preferences)):
        raise ValueError("device_preference must contain unique supported GPU families")
    # Legacy evidence has its own original A100 identity. It may never relax the
    # current allowlist or substitute the new experiment's calibrated device.
    locked = None if phase == "legacy" else _locked_runtime(output, config)
    candidates = []
    for device in discovery.get("devices", []):
        profile = _device_profile(device)
        if profile is None or locked and device["uuid"] != locked["gpu_uuid"]:
            continue
        # A deployment binding is an allowlist, including the first calibration.
        # Historical legacy replay keeps its separate frozen provenance gate.
        if phase != "legacy" and config.get("gpu_uuid") and device["uuid"] != config["gpu_uuid"]:
            continue
        family = "A100" if profile[0] == "a100_bf16" else "V100"
        if family not in preferences:
            continue
        if phase == "legacy" and profile[0] != "a100_bf16":
            continue
        candidates.append((device, profile))
    candidates.sort(key=lambda item: (preferences.index("A100" if item[1][0] == "a100_bf16" else "V100"),
                                      item[0].get("index", 0)))
    observed, selected = [], None
    for device, (sana_profile, attention) in candidates:
        occupancy = gpu_occupancy([device["uuid"]])
        row = dict(device=device, occupancy=occupancy, sana_profile=sana_profile,
                   wan_attention=attention)
        observed.append(row)
        if selected is None and occupancy.get("status") == "exclusive_observed":
            selected = row
    models = {name: model_inventory(name, entry) for name, entry in config["models"].items()}
    missing = [name for name, value in models.items() if not value["available"]]
    runtime = None
    if selected is not None:
        profile = selected["sana_profile"]
        python = config["python"] if profile == "a100_bf16" else config["v100_python"]
        runtime = dict(config, gpu_uuid=selected["device"]["uuid"], device="cuda:0",
                       gpu_name=selected["device"]["name"], wan_attention=selected["wan_attention"],
                       sana_profile=profile, python=python, main_config_sha256=digest(config))
        if locked and any(runtime[key] != locked[key] for key in DEVICE_FIELDS):
            raise ValueError("calibrated device, interpreter or execution recipe changed")
        if not Path(python).is_file():
            missing.append("python:" + python)
    ready = selected is not None and not missing
    return dict(schema_version="ditserve.scheduling.preflight.v1", status="ready" if ready else "blocked",
                gpu_work_started=False, selected=selected, observations=observed, discovery=discovery,
                models=models, missing_models=missing, runtime_config=runtime,
                calibration_locked=locked is not None,
                reason=("ready" if ready else "legacy requires the original A100, excluded by device_preference"
                        if phase == "legacy" and "A100" not in preferences else
                        "selected device is unavailable/busy or required local assets are missing"))


def freeze_source(output, phase, runtime):
    """Read each file once and execute its archive, never the changing workspace."""
    destination = Path(output) / phase / "sources"
    if destination.exists():
        raise ValueError("this phase already owns an immutable source archive")
    destination.mkdir(parents=True)
    files = sorted((ROOT / "python/ditserve").rglob("*.py"))
    records = {}
    for source in files:
        name = str(source.relative_to(ROOT))
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        records[name] = file_digest(target)
    atomic_json(destination / "runtime-config.json", runtime)
    records["runtime-config.json"] = file_digest(destination / "runtime-config.json")
    record = dict(schema_version="ditserve.scheduling.source.v1", phase=phase,
                  execution_root=str(destination.resolve()), source_files=records,
                  source_sha256=digest(records), runtime_config_sha256=digest(runtime))
    record["sha256"] = digest(record)
    atomic_json(destination / "manifest.json", record)
    return record


def verify_archive(source):
    if source.get("sha256") != digest({k: v for k, v in source.items() if k != "sha256"}):
        raise ValueError("source manifest identity changed")
    if source.get("source_sha256") != digest(source["source_files"]):
        raise ValueError("source archive inventory identity changed")
    root = Path(source["execution_root"]).resolve()
    for name, expected in source["source_files"].items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file() or file_digest(path) != expected:
            raise ValueError("archived executable source changed: " + name)


def execute(config, output, ledger, phase, *, wait_idle=0.0, timeout=None):
    if phase not in PHASES:
        raise ValueError("unknown scheduling phase")
    if config.get("budget_gpu_seconds") != TOTAL:
        raise ValueError("the scheduling allocation is fixed at 12 GPU hours")
    wait_idle = finite(wait_idle, "wait_idle")
    if timeout is not None:
        timeout = finite(timeout, "timeout", positive=True)
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    root = output / phase
    if any((root / name).exists() for name in ("supervisor.json", "worker.json", "runtime-config.json", "sources", "worker.log")):
        raise ValueError("phase already attempted; preserve evidence and use a fresh run")
    bind_protocol(config, output, ledger)
    prerequisite = PREREQUISITES.get(phase)
    if prerequisite:
        previous = phase_result(output, prerequisite)
        requires_qualification = prerequisite != "pilot"
        if (previous.get("status") != "passed" or (requires_qualification and previous.get("eligible") is not True)
                or previous.get("main_config_sha256") != digest(config)):
            result = dict(status="blocked", eligible=False, phase=phase, gpu_work_started=False,
                          reason="required supervised phase did not qualify: " + prerequisite)
            atomic_json(root / "admission.json", result)
            return result
    wait_deadline = time.monotonic() + wait_idle
    while True:
        observation = inspect(config, output, phase)
        atomic_json(root / "preflight.json", observation)
        if observation["status"] == "ready":
            break
        if time.monotonic() >= wait_deadline:
            result = dict(status="blocked", eligible=False, phase=phase, gpu_work_started=False,
                          reason=observation.get("reason", "device or assets unavailable"))
            atomic_json(root / "admission.json", result)
            return result
        time.sleep(min(5.0, max(0.0, wait_deadline - time.monotonic())))
    runtime = observation["runtime_config"]
    process = None
    failure = None
    returncode = None
    with Allocation(ledger) as budget:
        available = budget.remaining()
        cleanup_seconds, minimum_run_seconds = 45.0, 75.0
        if available < cleanup_seconds + minimum_run_seconds:
            result = dict(status="stopped", eligible=False, phase=phase, gpu_work_started=False,
                          reason="remaining allocation cannot admit a minimum unit and cleanup",
                          reason_code="budget_insufficient", remaining_gpu_seconds=available)
            atomic_json(root / "admission.json", result)
            return result
        run_seconds = min(available - cleanup_seconds, timeout) if timeout is not None else available - cleanup_seconds
        if run_seconds < minimum_run_seconds:
            result = dict(status="blocked", eligible=False, phase=phase, gpu_work_started=False,
                          reason="explicit timeout is shorter than the minimum runnable unit", retryable=True)
            atomic_json(root / "admission.json", result)
            return result
        source = freeze_source(output, phase, runtime)
        verify_archive(source)
        atomic_json(root / "runtime-config.json", runtime)
        child_deadline = time.monotonic() + run_seconds
        identity = dict(output=str(output), phase=phase, gpu_uuid=runtime["gpu_uuid"],
                        source_sha256=source["source_sha256"], main_config_sha256=digest(config),
                        runtime_config_sha256=digest(runtime), execution_root=source["execution_root"],
                        deadline_monotonic=child_deadline)
        session = budget.begin(phase=phase, identity=identity)
        command = [runtime["python"], "-u", "-m", "ditserve.scheduling.worker",
                   "--config", str(root / "runtime-config.json"), "--output", str(output),
                   "--phase", phase, "--deadline", str(child_deadline), "--ledger", str(ledger),
                   "--session-id", session["session_id"],
                   "--source-manifest", str(root / "sources/manifest.json")]
        env = dict(os.environ, PYTHONPATH=str(Path(source["execution_root"]) / "python"),
                   CUDA_VISIBLE_DEVICES=runtime["gpu_uuid"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                   TORCH_HOME=config["torch_home"], PYTHONDONTWRITEBYTECODE="1")
        try:
            with (root / "worker.log").open("x") as log:
                process = subprocess.Popen(command, cwd=source["execution_root"], env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                budget.child_started(process.pid)
                with (root / "isolation.jsonl").open("x") as observations:
                    while True:
                        snapshot = gpu_occupancy([runtime["gpu_uuid"]], allowed_pgid=process.pid)
                        observations.write(json.dumps(snapshot) + "\n")
                        observations.flush()
                        if snapshot.get("status") != "exclusive_observed":
                            raise RuntimeError("GPU isolation lost: " + str(snapshot.get("status")))
                        if time.monotonic() >= child_deadline + 10.0:
                            raise BudgetStop("worker exceeded cooperative scheduling deadline")
                        try:
                            returncode = process.wait(timeout=2.0)
                            break
                        except subprocess.TimeoutExpired:
                            pass
        except BaseException as error:
            failure = dict(error_type=type(error).__name__, reason=str(error))
        finally:
            group_gone = stop_owned_group(process)
            final = gpu_occupancy([runtime["gpu_uuid"]], allowed_pgid=process.pid if process else None)
            if not group_gone or final.get("owned") or final.get("status") == "unknown":
                atomic_json(root / "cleanup-unverified.json", dict(status="blocked", eligible=False,
                            reason="owned process cleanup unverified; ledger remains active", occupancy=final))
                raise RuntimeError("cannot confirm owned GPU cleanup; scheduling session remains unsettled")
            if final.get("status") != "exclusive_observed":
                failure = dict(error_type="GPUIsolationError", reason="GPU isolation lost at final observation")
            try:
                verify_archive(source)
            except (OSError, ValueError) as error:
                failure = dict(error_type="SourceIdentityError", reason=str(error))
            charged = budget.finish("failed" if failure or returncode != 0 else "completed")
    worker_path = root / "worker.json"
    try:
        result = read_json(worker_path) if worker_path.exists() else dict(status="incomplete", eligible=False,
                                                                        reason="worker report missing")
    except (OSError, ValueError) as error:
        result = dict(status="incomplete", eligible=False, reason="worker report unreadable: " + str(error))
    if not isinstance(result, dict):
        result = dict(status="incomplete", eligible=False, reason="worker report is not an object")
    if result.get("status") == "passed" and (result.get("session_id") != session["session_id"]
                                             or result.get("phase") != phase):
        failure = dict(error_type="WorkerIdentityError", reason="passed worker report has a different session or phase")
    if failure:
        result = dict(result, status="incomplete", eligible=False, supervisor_failure=failure)
    elif returncode != 0:
        result = dict(result, status="failed" if result.get("status") == "passed" else result.get("status", "failed"),
                      eligible=False)
    record = dict(result, phase=phase, returncode=returncode, charge=charged, final_occupancy=final,
                  main_config_sha256=digest(config), runtime_config_sha256=digest(runtime),
                  source_sha256=source["source_sha256"], gpu_uuid=runtime["gpu_uuid"],
                  sana_profile=runtime["sana_profile"], gpu_work_started=True)
    atomic_json(root / "supervisor.json", record)
    return record


def report(output, ledger):
    from .reporting import report as build_report
    return build_report(output, ledger=ledger)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ditserve research-scheduling")
    parser.add_argument("command", choices=("preflight", "prepare", "run", "report"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--wait-idle-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float)
    args = parser.parse_args(argv)
    if args.command == "run" and args.phase is None:
        parser.error("run requires --phase")
    if args.command == "report":
        result = report(args.output, args.ledger)
    else:
        config = load_config(args.config)
        if args.command == "prepare":
            result = prepare(config, args.output, args.ledger)
        elif args.command == "preflight":
            result = inspect(config, args.output, args.phase)
        else:
            result = execute(config, args.output, args.ledger, args.phase,
                             wait_idle=args.wait_idle_seconds, timeout=args.timeout_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("status") in ("failed", "incomplete", "blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
