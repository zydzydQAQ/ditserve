"""Supervise the separately authorized, unlimited-accounting V100 study."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from ..directions.runner import model_inventory, stop_owned_group
from ..research.common import atomic_json, digest, finite, read_json
from ..research.isolation import gpu_occupancy
from ..research.preflight import discover_gpus
from .next_budget import Allocation, AUTHORIZATION, WatchdogStop
from .runner import freeze_source, verify_archive


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research_scheduling_next.json"
DEFAULT_OUTPUT = ROOT / "results/research-scheduling-next/run01"
DEFAULT_LEDGER = ROOT / "results/research-scheduling-next/gpu-budget.json"
GPU_UUID = "GPU-a058066e-bde9-c534-efb1-d2a68c117681"
PHASES = ("completion", "granularity", "replicate")
PREREQUISITES = {"granularity": "completion", "replicate": "granularity"}
DEVICE_FIELDS = ("gpu_uuid", "device", "wan_attention", "sana_profile", "python")
DEFAULT_WATCHDOG_SECONDS = 48 * 3600.0


def validate_config(config):
    if "budget_gpu_seconds" not in config or config["budget_gpu_seconds"] is not None:
        raise ValueError("new cumulative GPU accounting must explicitly have a null limit")
    if config.get("authorization") != AUTHORIZATION:
        raise ValueError("unlimited V100 study requires its explicit authorization")
    if config.get("device_preference") != ["V100"] or config.get("gpu_uuid", GPU_UUID) != GPU_UUID:
        raise ValueError("next scheduling is restricted to the authorized V100 UUID")
    if set(config.get("models", {})) != {"wan", "sana"}:
        raise ValueError("next scheduling requires the pinned Wan and Sana models")
    if tuple(config.get("phases", PHASES)) != PHASES:
        raise ValueError("next scheduling phase sequence differs")
    return config


def load_config(path=DEFAULT_CONFIG):
    return validate_config(read_json(path))


def bind_protocol(config, output, ledger):
    validate_config(config)
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    record = dict(schema_version="ditserve.scheduling.next.protocol.v1", config=config,
                  config_sha256=digest(config), ledger_path=str(ledger), budget_gpu_seconds=None,
                  authorization=AUTHORIZATION, gpu_uuid=GPU_UUID)
    record["sha256"] = digest(record)
    path = output / "protocol.json"
    if path.exists():
        if read_json(path) != record:
            raise ValueError("run configuration or ledger changed; preserve this run and use a new output")
    else:
        atomic_json(path, record)
    return record


def prepare(config, output, ledger):
    protocol = bind_protocol(config, output, ledger)
    with Allocation(ledger) as account:
        return dict(status="prepared", gpu_work_started=False, protocol=protocol,
                    allocation_id=account.data["allocation_id"], limit_gpu_seconds=None,
                    remaining_gpu_seconds=None, used_gpu_seconds=account.data["used_gpu_seconds"])


def phase_result(output, phase):
    path = Path(output) / phase / "supervisor.json"
    return read_json(path) if path.exists() else dict(status="not_run", eligible=False)


def prerequisite_complete(previous, phase, config):
    complete = previous.get("status") == "passed" or (
        previous.get("status") == "stopped" and previous.get("technical_complete") is True)
    return (complete and previous.get("returncode", 0) == 0 and
            not previous.get("supervisor_failure") and not previous.get("watchdog_stopped") and
            previous.get("main_config_sha256") == digest(config))


def _locked_runtime(output, config):
    if output is None:
        return None
    path = Path(output) / "completion/runtime-config.json"
    if not path.exists():
        return None
    runtime = read_json(path)
    previous = phase_result(output, "completion")
    if not prerequisite_complete(previous, "completion", config):
        raise ValueError("completion did not finish under the bound configuration")
    if (runtime.get("main_config_sha256") != digest(config) or
            previous.get("runtime_config_sha256") != digest(runtime)):
        raise ValueError("completion runtime changed after supervised measurement")
    return runtime


def inspect(config, output=None, phase=None):
    validate_config(config)
    discovery = discover_gpus()
    locked = _locked_runtime(output, config)
    observations, selected = [], None
    for device in discovery.get("devices", []):
        if (device.get("uuid") != GPU_UUID or "V100" not in device.get("name", "").upper()
                or float(device.get("compute_capability", 0)) != 7.0):
            continue
        occupancy = gpu_occupancy([GPU_UUID])
        row = dict(device=device, occupancy=occupancy, sana_profile="v100_fp32", wan_attention="efficient")
        observations.append(row)
        if occupancy.get("status") == "exclusive_observed":
            selected = row
    models = {name: model_inventory(name, entry) for name, entry in config["models"].items()}
    missing = [name for name, value in models.items() if not value["available"]]
    runtime = None
    if selected is not None:
        python = config.get("v100_python", config["python"])
        runtime = dict(config, gpu_uuid=GPU_UUID, device="cuda:0", gpu_name=selected["device"]["name"],
                       wan_attention="efficient", sana_profile="v100_fp32", python=python,
                       main_config_sha256=digest(config))
        if locked and any(runtime[key] != locked[key] for key in DEVICE_FIELDS):
            raise ValueError("completion device, interpreter or execution recipe changed")
        if not Path(python).is_file():
            missing.append("python:" + python)
    ready = selected is not None and not missing
    return dict(schema_version="ditserve.scheduling.next.preflight.v1", status="ready" if ready else "blocked",
                gpu_work_started=False, selected=selected, observations=observations, discovery=discovery,
                models=models, missing_models=missing, runtime_config=runtime, completion_locked=locked is not None,
                reason_code=None if ready else "assets_missing" if missing else "device_unavailable",
                reason="ready" if ready else "authorized V100 unavailable/busy or required local assets missing")


def phase_watchdog(config, phase, *, timeout=None):
    """Finite hang protection, independent of unlimited cumulative accounting."""
    estimates = config.get("phase_estimated_seconds", {})
    estimated = finite(estimates[phase], "estimated complete phase seconds", positive=True) if phase in estimates else 0.0
    settings = config.get("phase_watchdog_seconds", {})
    configured = settings.get(phase) if isinstance(settings, dict) else settings
    requested = timeout if timeout is not None else configured
    duration = (max(DEFAULT_WATCHDOG_SECONDS, 1.5 * estimated) if requested is None else
                finite(requested, "phase watchdog seconds", positive=True))
    if duration < max(120.0, 1.5 * estimated):
        raise ValueError("phase watchdog is too short for the estimated complete work and cleanup")
    return duration


def execute(config, output, ledger, phase, *, wait_idle=0.0, timeout=None):
    if phase not in PHASES:
        raise ValueError("unknown scheduling phase")
    validate_config(config)
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
        if not prerequisite_complete(previous, prerequisite, config):
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
                          reason_code=observation.get("reason_code", "device_unavailable"),
                          reason=observation.get("reason", "device or assets unavailable"))
            atomic_json(root / "admission.json", result)
            return result
        time.sleep(min(5.0, max(0.0, wait_deadline - time.monotonic())))
    runtime = observation["runtime_config"]
    process = None
    failure = None
    returncode = None
    with Allocation(ledger) as budget:
        run_seconds = phase_watchdog(config, phase, timeout=timeout)
        source = freeze_source(output, phase, runtime)
        verify_archive(source)
        atomic_json(root / "runtime-config.json", runtime)
        child_deadline = time.monotonic() + run_seconds
        identity = dict(output=str(output), phase=phase, gpu_uuid=runtime["gpu_uuid"],
                        source_sha256=source["source_sha256"], main_config_sha256=digest(config),
                        runtime_config_sha256=digest(runtime), execution_root=source["execution_root"],
                        deadline_monotonic=child_deadline, watchdog_seconds=run_seconds,
                        authorization=AUTHORIZATION)
        session = budget.begin(phase=phase, identity=identity)
        command = [runtime["python"], "-u", "-m", "ditserve.scheduling.next_worker",
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
                            raise WatchdogStop("worker exceeded finite phase watchdog")
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
    if worker_path.exists() and (result.get("session_id") != session["session_id"]
                                 or result.get("phase") != phase):
        failure = dict(error_type="WorkerIdentityError", reason="worker report has a different session or phase")
    if failure:
        result = dict(result, status="incomplete", eligible=False, supervisor_failure=failure)
    elif returncode != 0:
        result = dict(result, status="failed" if result.get("status") == "passed" else result.get("status", "failed"),
                      eligible=False)
    record = dict(result, phase=phase, returncode=returncode, charge=charged, final_occupancy=final,
                  main_config_sha256=digest(config), runtime_config_sha256=digest(runtime),
                  source_sha256=source["source_sha256"], gpu_uuid=runtime["gpu_uuid"],
                  sana_profile=runtime["sana_profile"], gpu_work_started=True,
                  watchdog_seconds=run_seconds, limit_gpu_seconds=None, authorization=AUTHORIZATION)
    atomic_json(root / "supervisor.json", record)
    return record


def report(output, ledger):
    from .next_reporting import report as build_report
    return build_report(output, ledger=ledger)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ditserve.scheduling.next_runner")
    parser.add_argument("command", choices=("preflight", "prepare", "run", "report", "continue"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--wait-idle-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, help="finite phase watchdog, not a cumulative GPU budget")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "run" and args.phase is None:
        parser.error("run requires --phase")
    if args.command == "report":
        result = report(args.output, args.ledger)
    elif args.command == "continue":
        from .next_study import continue_study
        result = continue_study(config, args.output, args.ledger, poll_seconds=args.poll_seconds)
    elif args.command == "prepare":
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
