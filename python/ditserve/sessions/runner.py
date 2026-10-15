"""Supervise the separately authorized, unlimited-accounting V100 study."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from ..directions.runner import model_inventory as diffusion_inventory, stop_owned_group
from ..research.common import atomic_json, digest, file_digest, finite, read_json
from ..research.isolation import gpu_occupancy
from ..research.preflight import discover_gpus
from .budget import Allocation, AUTHORIZATION, WatchdogStop
from ..scheduling.runner import freeze_source, verify_archive


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research_sessions.json"
DEFAULT_OUTPUT = ROOT / "results/research-sessions/run01"
DEFAULT_LEDGER = ROOT / "results/research-sessions/gpu-budget.json"
GPU_UUID = "GPU-a058066e-bde9-c534-efb1-d2a68c117681"
PHASES = ("qualify", "pilot", "state")
PREREQUISITES = {"pilot": "qualify", "state": "pilot"}
DEVICE_FIELDS = ("gpu_uuid", "device", "sana_profile", "python", "llm_dtype")
DEFAULT_WATCHDOG_SECONDS = 48 * 3600.0
CRITICAL_SOURCES = ("scheduling/runtime.py", "scheduling/execution.py", "scheduling/next_persistence.py",
                    "directions/sana.py", "research/common.py", "research/isolation.py")



DEFAULT_PREREQUISITE_RUN = ROOT / "results/research-scheduling-next/run01"


def cpu_source_inventory():
    package = ROOT / "python/ditserve"
    files = set((package / "sessions").glob("*.py"))
    files.update(package / name for name in CRITICAL_SOURCES)
    required = {"runner.py", "budget.py", "worker.py", "study.py", "execution.py", "runtime.py",
                "policy.py", "workload.py", "reporting.py", "state.py"}
    if not required.issubset({path.name for path in files if path.parent == package / "sessions"}):
        raise ValueError("session implementation is not complete")
    return {str(path.relative_to(ROOT)): file_digest(path) for path in sorted(files)}


def mark_cpu_ready(config, output, *, tests, source_files):
    """Bind an actual passing CPU test invocation to its pre-test source hashes."""
    validate_config(config)
    if not isinstance(tests, dict) or tests.get("returncode") != 0 or not tests.get("command"):
        raise ValueError("CPU readiness requires a recorded successful test invocation")
    if source_files != cpu_source_inventory():
        raise ValueError("critical source changed during CPU validation; test the final version")
    record = dict(schema_version="ditserve.sessions.cpu-validation.v1", status="passed",
                  main_config_sha256=digest(config), source_files=source_files, tests=tests, recorded_at=time.time())
    record["sha256"] = digest(record)
    atomic_json(Path(output) / "cpu-validation.json", record)
    return record


def cpu_readiness(config, output):
    path = Path(output) / "cpu-validation.json"
    if not path.exists():
        return dict(status="waiting", reason="final session implementation awaits passing CPU validation")
    record = read_json(path)
    if (record.get("status") != "passed" or record.get("tests", {}).get("returncode") != 0 or
            record.get("main_config_sha256") != digest(config) or
            record.get("sha256") != digest({key: value for key, value in record.items() if key != "sha256"}) or
            record.get("source_files") != cpu_source_inventory()):
        return dict(status="waiting", reason="CPU validation does not bind the current session implementation")
    return record


def model_inventory(name, entry):
    if name == "sana":
        return diffusion_inventory(name, entry)
    path = Path(entry["path"])
    required = ["config.json", "tokenizer_config.json"]
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        required.append("<missing-tokenizer-assets>")
    # Runtime explicitly loads use_safetensors=True. Match its canonical file
    # precedence rather than requiring unrelated, undownloaded PyTorch assets.
    chosen_index = None
    weights = []
    if (path / "model.safetensors").is_file():
        weights = ["model.safetensors"]
    elif (path / "model.safetensors.index.json").is_file():
        index = path / "model.safetensors.index.json"
        chosen_index = dict(path=str(index), sha256=file_digest(index))
        weight_map = read_json(index).get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError("invalid LLM safetensors weight map")
        for name in sorted(set(weight_map.values())):
            if not isinstance(name, str):
                raise ValueError("invalid LLM safetensors weight path")
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
                raise ValueError("unsafe LLM safetensors weight index path")
            weights.append(name)
        if not weights:
            required.append("<missing-indexed-model-weights>")
    else:
        required.append("<missing-safetensors-model-weights>")
    required.extend(weights)
    missing = [name for name in required if not (path / name).is_file()]
    return dict(available=not missing, missing=missing, path=str(path), revision=entry.get("revision"),
                weight_format="safetensors", chosen_index=chosen_index, weight_files=weights)


def upstream_completion(config):
    root = Path(config.get("prerequisite_run", DEFAULT_PREREQUISITE_RUN)).resolve()
    path = root / "completion/supervisor.json"
    if (root / "completion/cleanup-unverified.json").exists():
        return dict(status="failed", reason="prior completion GPU cleanup remains unverified", path=str(path))
    if not path.exists():
        return dict(status="waiting", reason="current video completion has not finished", path=str(path))
    value = read_json(path)
    protocol = read_json(root / "protocol.json")
    runtime = read_json(root / "completion/runtime-config.json")
    occupancy = value.get("final_occupancy", {})
    good = (value.get("status") == "passed" and value.get("returncode") == 0 and
            value.get("technical_complete") is True and not value.get("supervisor_failure") and
            not value.get("watchdog_stopped") and value.get("gpu_uuid") == GPU_UUID and
            occupancy.get("status") == "exclusive_observed" and not occupancy.get("owned") and
            not occupancy.get("foreign") and value.get("main_config_sha256") == protocol.get("config_sha256") and
            digest(protocol["config"]) == protocol.get("config_sha256") and
            value.get("runtime_config_sha256") == digest(runtime) and runtime.get("gpu_uuid") == GPU_UUID)
    return dict(status="passed" if good else "failed", reason="prior completion finished" if good else
                "prior completion did not finish under isolated supervision", path=str(path),
                sha256=file_digest(path), supervisor=value)


def validate_config(config):
    if "budget_gpu_seconds" not in config or config["budget_gpu_seconds"] is not None:
        raise ValueError("session cumulative GPU accounting must explicitly have a null limit")
    if config.get("authorization") != AUTHORIZATION:
        raise ValueError("unlimited V100 study requires its explicit authorization")
    if config.get("device_preference") != ["V100"] or config.get("gpu_uuid", GPU_UUID) != GPU_UUID:
        raise ValueError("session research is restricted to the authorized V100 UUID")
    if set(config.get("models", {})) != {"llm", "sana"}:
        raise ValueError("session research requires the pinned LLM and Sana models")
    if tuple(config.get("phases", PHASES)) != PHASES:
        raise ValueError("session research phase sequence differs")
    if not Path(config.get("prerequisite_run", DEFAULT_PREREQUISITE_RUN)).is_absolute():
        raise ValueError("prerequisite_run must be absolute")
    return config


def load_config(path=DEFAULT_CONFIG):
    return validate_config(read_json(path))


def bind_protocol(config, output, ledger):
    validate_config(config)
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    if (output / "finalizing.json").exists():
        raise ValueError("session run is finalizing; do not start or rewrite its controls")
    if (output / "results.xlsx").exists() and not (output / "protocol.json").exists():
        from .reporting import archived
        if archived(output):
            raise ValueError("session run is archived; read its report or use a new run")
    upstream = Path(config.get("prerequisite_run", DEFAULT_PREREQUISITE_RUN)).resolve()
    if output == upstream or output in upstream.parents or upstream in output.parents:
        raise ValueError("session output and historical prerequisite must be separate trees")
    record = dict(schema_version="ditserve.sessions.protocol.v1", config=config,
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
    complete = (previous.get("status") == "passed" and previous.get("technical_complete") is not False) or (
        previous.get("status") == "stopped" and previous.get("technical_complete") is True)
    return (complete and previous.get("returncode") == 0 and
            not previous.get("supervisor_failure") and not previous.get("watchdog_stopped") and
            previous.get("main_config_sha256") == digest(config) and
            (phase != "qualify" or previous.get("eligible") is True))


def _locked_runtime(output, config):
    if output is None:
        return None
    path = Path(output) / "qualify/runtime-config.json"
    if not path.exists():
        return None
    runtime = read_json(path)
    previous = phase_result(output, "qualify")
    if not prerequisite_complete(previous, "qualify", config):
        raise ValueError("qualification did not finish under the bound configuration")
    if (runtime.get("main_config_sha256") != digest(config) or
            previous.get("runtime_config_sha256") != digest(runtime)):
        raise ValueError("qualification runtime changed after supervised measurement")
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
        row = dict(device=device, occupancy=occupancy, sana_profile="v100_fp32")
        observations.append(row)
        if occupancy.get("status") == "exclusive_observed":
            selected = row
    models = {name: model_inventory(name, entry) for name, entry in config["models"].items()}
    missing = [name for name, value in models.items() if not value["available"]]
    runtime = None
    if selected is not None:
        python = config.get("v100_python", config["python"])
        runtime = dict(config, gpu_uuid=GPU_UUID, device="cuda:0", gpu_name=selected["device"]["name"],
                       sana_profile="v100_fp32", python=python, llm_dtype=config.get("llm_dtype", "float16"),
                       main_config_sha256=digest(config))
        if locked and any(runtime[key] != locked[key] for key in DEVICE_FIELDS):
            raise ValueError("qualified device, interpreter or execution recipe changed")
        if not Path(python).is_file():
            missing.append("python:" + python)
    ready = selected is not None and not missing
    return dict(schema_version="ditserve.sessions.preflight.v1", status="ready" if ready else "blocked",
                gpu_work_started=False, selected=selected, observations=observations, discovery=discovery,
                models=models, missing_models=missing, runtime_config=runtime, qualification_locked=locked is not None,
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
    upstream = upstream_completion(config)
    if upstream["status"] != "passed":
        result = dict(status="blocked", eligible=False, phase=phase, gpu_work_started=False,
                      reason_code="upstream_incomplete", reason=upstream["reason"], upstream=upstream)
        atomic_json(root / "admission.json", result)
        return result
    readiness = cpu_readiness(config, output)
    if readiness["status"] != "passed":
        result = dict(status="blocked", eligible=False, phase=phase, gpu_work_started=False,
                      reason_code="implementation_pending", reason=readiness["reason"])
        atomic_json(root / "admission.json", result)
        return result
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
        if any(source["source_files"].get(name) != expected for name, expected in readiness["source_files"].items()):
            raise ValueError("frozen executable source differs from the version that passed CPU validation")
        atomic_json(root / "runtime-config.json", runtime)
        child_deadline = time.monotonic() + run_seconds
        identity = dict(output=str(output), phase=phase, gpu_uuid=runtime["gpu_uuid"],
                        source_sha256=source["source_sha256"], main_config_sha256=digest(config),
                        runtime_config_sha256=digest(runtime), execution_root=source["execution_root"],
                        deadline_monotonic=child_deadline, watchdog_seconds=run_seconds,
                        authorization=AUTHORIZATION, cpu_validation=readiness)
        session = budget.begin(phase=phase, identity=identity)
        command = [runtime["python"], "-u", "-m", "ditserve.sessions.worker",
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


def report(output, ledger, **kwargs):
    from .reporting import report as build_report
    return build_report(output, ledger=ledger, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ditserve.sessions.runner")
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
        from .study import continue_study
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
