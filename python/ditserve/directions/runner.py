"""Supervise serial GPU phases, with independent budget and isolation checks."""

import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ..research.common import atomic_json, digest, file_digest, read_json
from ..research.isolation import gpu_occupancy
from ..research.preflight import discover_gpus
from .budget import Allocation, BudgetStop
from .protocol import (ROOT, DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_LEDGER,
                       PARENT_LEDGER, load_config, freeze_config, freeze_phase)


PHASES = {1: ("profile", "correctness", "preemption", "online"),
          2: ("screen", "qualify", "state_cost", "pressure", "actions"),
          3: ("correctness", "profile", "update", "multisession")}


def model_inventory(name, entry):
    path = Path(entry["path"])
    if name == "longlive":
        required = ["longlive_models/models/longlive_base.pt", "longlive_models/models/lora.pt",
                    "wan_models/Wan2.1-T2V-1.3B/config.json", "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                    "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model",
                    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json"]
    else:
        required = ["model_index.json", "transformer/config.json", "vae/config.json", "text_encoder/config.json"]
        for component in ("transformer", "vae", "text_encoder"):
            directory = path / component
            indices = list(directory.glob("*.index.json"))
            if indices:
                for index in indices:
                    required.extend(str((directory / shard).relative_to(path))
                                    for shard in set(read_json(index).get("weight_map", {}).values()))
            else:
                # These fixed snapshots use safetensors; missing files cannot be treated as an empty model.
                weights = list(directory.glob("*.safetensors"))
                required.extend(str(p.relative_to(path)) for p in weights)
                if not weights:
                    required.append(component + "/<missing-safetensors>")
    missing = [name for name in required if not (path / name).is_file()]
    manifest_path = entry.get("verification_manifest")
    manifest, manifest_sha256, environment_record_sha256 = {}, None, None
    if manifest_path:
        if not Path(manifest_path).is_file():
            missing.append("<download-SHA256-manifest>")
        else:
            manifest = read_json(manifest_path)
            if manifest.get("status") != "verified" or not manifest.get("files"):
                missing.append("<unverified-download-manifest>")
            else:
                for item in manifest["files"]:
                    target = path / item["path"]
                    if not target.is_file() or target.stat().st_size != item["bytes"]:
                        missing.append(item["path"] + " <manifest-size-mismatch>")
                manifest_sha256 = file_digest(manifest_path)
    if name == "longlive":
        validation = manifest.get("environment_validation")
        if not isinstance(validation, dict) or validation.get("status") != "passed":
            missing.append("<environment-not-validated>")
        else:
            python = entry.get("python")
            if (not isinstance(python, str) or not Path(python).is_absolute()
                    or validation.get("python") != python):
                missing.append("<environment-python-mismatch>")
            elif not Path(python).is_file():
                missing.append("<environment-python-unavailable>")
            record_path = validation.get("record")
            if not isinstance(record_path, str) or not Path(record_path).is_absolute():
                missing.append("<environment-validation-record-not-absolute>")
            elif not Path(record_path).is_file():
                missing.append("<environment-validation-record-missing>")
            else:
                try:
                    record_sha256 = file_digest(record_path)
                    record = read_json(record_path)
                except (OSError, ValueError):
                    missing.append("<environment-validation-record-invalid>")
                else:
                    if record_sha256 != validation.get("sha256"):
                        missing.append("<environment-validation-record-hash-mismatch>")
                    elif not isinstance(record, dict) or record.get("status") != "verified_cpu_only":
                        missing.append("<environment-validation-record-not-verified>")
                    elif (record.get("python") != python or record.get("gpu_used") is not False
                          or record.get("cuda_initialized") is not False):
                        missing.append("<environment-validation-record-identity-mismatch>")
                    elif (not isinstance(record.get("native_attention_backends"), dict)
                          or record["native_attention_backends"].get("FLASH_ATTN_2_AVAILABLE") is not True):
                        missing.append("<environment-attention-backend-not-validated>")
                    else:
                        environment_record_sha256 = record_sha256
    return dict(path=str(path), available=not missing, revision=entry.get("revision"),
                missing=missing, required_files=len(required),
                verification_manifest_sha256=manifest_sha256,
                environment_record_sha256=environment_record_sha256,
                scope=("file completeness; immutable download manifests hold weight SHA256 verification"
                       + ("; CPU environment validation record and native attention backend required" if name == "longlive" else "")))


def inspect(config, direction=None):
    discovery = discover_gpus()
    selected = [d for d in discovery["devices"] if d["uuid"] == config["gpu_uuid"]]
    occupied = gpu_occupancy([config["gpu_uuid"]]) if selected else {"status": "unavailable"}
    models = {}
    for key, value in config["models"].items():
        models[key] = model_inventory(key, value)
    needed = {1: ("wan", "sana"), 2: ("wan",), 3: ("longlive",)}.get(direction, tuple(models))
    missing = [key for key in needed if not models[key]["available"]]
    idle = bool(selected and occupied["status"] == "exclusive_observed")
    return dict(schema_version="ditserve.directions.preflight.v1", devices=discovery,
                selected=selected, occupancy=occupied, models=models, gpu_work_started=False,
                missing_models=missing, gpu_idle=idle,
                status="ready" if idle and not missing else "blocked")


def stop_owned_group(process):
    """Only signal the new session created by this supervisor, never foreign jobs."""
    if process is None:
        return True
    def alive():
        process.poll()  # Reap the group leader before inspecting group existence.
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        return True
    deadline = time.monotonic() + 15
    while alive() and time.monotonic() < deadline:
        time.sleep(.1)
    if alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 15
        while alive() and time.monotonic() < deadline:
            time.sleep(.1)
    return not alive()


def phase_result(root, phase, *, allow_worker=False):
    paths = [root / ("supervisor-%s.json" % phase)]
    if allow_worker:
        paths.append(root / ("worker-%s.json" % phase))
    path = next((p for p in paths if p.exists()), None)
    return read_json(path) if path else dict(status="incomplete", reason="worker report missing")


def execute(config, output, ledger, direction, phase, *, wait_idle=0., timeout=None, reserve_for=None):
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    root = output / ("direction%d" % direction)
    root.mkdir(parents=True, exist_ok=True)
    retry = None
    if reserve_for is not None:
        failed_path = Path(reserve_for).resolve()
        failed = read_json(failed_path)
        if (failed.get("status") not in ("failed", "incomplete") or
                failed.get("direction") != direction or failed.get("phase") != phase or
                not failed.get("charge") or failed.get("config_sha256") != digest(config)):
            raise ValueError("reserve requires a charged failed attempt of this same phase and protocol")
        retry = dict(path=str(failed_path), sha256=file_digest(failed_path), direction=direction, phase=phase)
    if direction > 1 and not (output / ("direction%d" % (direction - 1)) / "terminal.json").exists():
        result = dict(status="blocked", eligible=False, reason="previous direction has no terminal conclusion",
                      gpu_work_started=False, direction=direction, phase=phase)
        atomic_json(root / ("admission-%s.json" % phase), result)
        return result
    phase_index = PHASES[direction].index(phase)
    if phase_index:
        previous = phase_result(root, PHASES[direction][phase_index - 1])
        if previous.get("status") != "passed" or previous.get("eligible") is False:
            result = dict(status="blocked", eligible=False, reason="previous phase did not qualify",
                          direction=direction, phase=phase)
            atomic_json(root / ("admission-%s.json" % phase), result)
            return result
    existing = root / ("worker-%s.json" % phase)
    if existing.exists():
        raise ValueError("phase has already run; preserve it and use a fresh output directory")
    frozen = freeze_config(config, output)
    deadline = time.monotonic() + wait_idle
    while True:
        observation = inspect(config, direction)
        atomic_json(root / ("preflight-%s.json" % phase), observation)
        if observation.get("missing_models") and time.monotonic() >= deadline:
            result = dict(status="blocked", eligible=False, reason="required model files are incomplete",
                          models=observation["missing_models"], gpu_work_started=False, direction=direction, phase=phase)
            atomic_json(root / ("admission-%s.json" % phase), result)
            return result
        if observation["status"] == "ready":
            break
        if time.monotonic() >= deadline:
            result = dict(status="blocked", eligible=False, reason="selected A100 is busy or unavailable",
                          gpu_work_started=False, direction=direction, phase=phase)
            atomic_json(root / ("admission-%s.json" % phase), result)
            return result
        time.sleep(min(5., max(0., deadline - time.monotonic())))
    python = config["python"] if direction != 3 else config["models"]["longlive"]["python"]
    if not Path(python).exists():
        result = dict(status="blocked", eligible=False, reason="phase Python environment is unavailable",
                      python=python, gpu_work_started=False)
        atomic_json(root / ("admission-%s.json" % phase), result)
        return result
    process, failure, returncode = None, None, None
    stage = "reserve" if retry else "direction%d" % direction
    with Allocation(ledger, PARENT_LEDGER) as budget:
        available = budget.remaining(stage)
        cleanup_seconds, minimum_run_seconds = 45., 75.
        minimum_total_seconds = cleanup_seconds + minimum_run_seconds
        if available < minimum_total_seconds:
            # Exhausting a finite allocation is an unmeasured outcome, not a
            # failed scientific hypothesis. No child/session/source attempt has
            # started, and the next direction may use its own allocation.
            result = dict(status="stopped", eligible=False, direction=direction, phase=phase,
                          reason="remaining allocation cannot admit a minimum unit and cleanup",
                          reason_code="budget_insufficient", budget_stopped=True,
                          scientific_outcome="not_measured_budget_limited", gpu_work_started=False,
                          allocation_stage=stage, remaining_gpu_seconds=available,
                          minimum_run_seconds=minimum_run_seconds, cleanup_allowance_seconds=cleanup_seconds,
                          required_gpu_seconds=minimum_total_seconds, config_sha256=digest(config))
            atomic_json(root / ("admission-%s.json" % phase), result)
            atomic_json(root / "terminal.json", result)
            return result
        run_seconds = available - cleanup_seconds
        if timeout is not None:
            run_seconds = min(run_seconds, timeout)
        if run_seconds < minimum_run_seconds:
            # A caller's short timeout can be corrected without ending the
            # direction or consuming its immutable phase attempt.
            result = dict(status="blocked", eligible=False, direction=direction, phase=phase,
                          reason="explicit timeout is shorter than the minimum runnable unit",
                          reason_code="timeout_too_short", gpu_work_started=False, retryable=True,
                          timeout_seconds=timeout, minimum_run_seconds=minimum_run_seconds,
                          remaining_gpu_seconds=available, config_sha256=digest(config))
            atomic_json(root / ("admission-%s.json" % phase), result)
            return result
        source = freeze_phase(output, direction, phase, config)
        atomic_json(root / "worker-config.json", config)
        budget.begin(stage, gpu_count=1, identity=dict(direction=direction, phase=phase,
                     output=str(root.resolve()), source_sha256=source["source_sha256"],
                     config_sha256=frozen["configuration_sha256"], reserve_for=retry))
        child_deadline = time.monotonic() + run_seconds
        command = [python, "-u", "-m", "ditserve.directions.worker", "--config", str(root / "worker-config.json"),
                   "--output", str(root), "--direction", str(direction), "--phase", phase,
                   "--deadline", str(child_deadline), "--ledger", str(ledger),
                   "--session-id", budget.data["active"]["session_id"]]
        env = dict(os.environ, PYTHONPATH=str(Path(source["execution_root"]) / "python"),
                   CUDA_VISIBLE_DEVICES=config["gpu_uuid"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                   TORCH_HOME=config["torch_home"])
        try:
            with (root / ("%s.log" % phase)).open("x") as log:
                process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                budget.child_started(process.pid)
                with (root / ("%s-isolation.jsonl" % phase)).open("x") as observations:
                    while True:
                        snapshot = gpu_occupancy([config["gpu_uuid"]], allowed_pgid=process.pid)
                        observations.write(json.dumps(snapshot) + "\n")
                        observations.flush()
                        if snapshot["status"] != "exclusive_observed":
                            raise RuntimeError("GPU isolation lost: " + snapshot["status"])
                        if time.monotonic() >= child_deadline + 10.:
                            raise BudgetStop("worker exceeded cooperative phase deadline")
                        try:
                            returncode = process.wait(timeout=2.)
                            break
                        except subprocess.TimeoutExpired:
                            pass
        except BaseException as error:
            failure = dict(error_type=type(error).__name__, reason=str(error))
        finally:
            group_gone = stop_owned_group(process)
            # Do not release accounting if a child process group still owns CUDA.
            final = gpu_occupancy([config["gpu_uuid"]], allowed_pgid=process.pid if process else None)
            if not group_gone or final.get("owned") or final.get("status") == "unknown":
                raise RuntimeError("cannot confirm GPU cleanup; leave budget session unsettled")
            if final.get("status") != "exclusive_observed":
                failure = dict(error_type="GPUIsolationError", reason="GPU isolation lost at final observation")
            changed = [name for name, expected in source["source_files"].items()
                       if not (Path(source["execution_root"]) / name).is_file()
                       or file_digest(Path(source["execution_root"]) / name) != expected]
            if changed:
                failure = dict(error_type="SourceIdentityError", reason="frozen executable source changed", files=changed)
            charged = budget.finish("failed" if failure else "completed")
    report = phase_result(root, phase, allow_worker=True)
    if failure:
        report = dict(report, status="incomplete", eligible=False, supervisor_failure=failure)
    elif returncode != 0:
        report = dict(report, status="failed" if report["status"] == "passed" else report["status"], eligible=False)
    record = dict(report, direction=direction, phase=phase, returncode=returncode, charge=charged,
                  source_sha256=source["source_sha256"], config_sha256=digest(config),
                  final_occupancy=final)
    atomic_json(root / ("supervisor-%s.json" % phase), record)
    if record["status"] != "passed" or record.get("eligible") is False or phase == PHASES[direction][-1]:
        atomic_json(root / "terminal.json", dict(direction=direction, phase=phase,
                    status=record["status"], eligible=record.get("eligible", False),
                    reason=record.get("reason", "finite validation completed")))
    return record


def report(output, ledger):
    from .reporting import write_report
    return write_report(output, ledger, PHASES)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ditserve research-directions")
    parser.add_argument("command", choices=("prepare", "preflight", "run", "run-all", "report", "export-media", "export-blind"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--direction", type=int, choices=(1, 2, 3))
    parser.add_argument("--phase")
    parser.add_argument("--wait-idle-seconds", type=float, default=0.,
                        help="wait for idle A100 and complete model assets without charging GPU budget")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--limit", type=int, default=8, help="offline media previews per direction, at most 8")
    parser.add_argument("--reserve-for", type=Path,
                        help="charged failed supervisor report authorizing this same phase's repair retry")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.reserve_for and args.command != "run":
        parser.error("reserve is only available for an explicit single-phase repair retry")
    if args.command == "prepare":
        result = freeze_config(config, args.output)
        with Allocation(args.ledger, PARENT_LEDGER):
            pass
    elif args.command == "preflight":
        result = inspect(config, args.direction)
        atomic_json(args.output / "preflight.json", result)
    elif args.command == "report":
        result = report(args.output, args.ledger)
    elif args.command == "export-media":
        from .media import export_media
        result = export_media(args.output, limit=args.limit)
    elif args.command == "export-blind":
        from .blind import export_blind_windows
        result = export_blind_windows(args.output)
    elif args.command == "run":
        if args.direction is None or args.phase not in PHASES[args.direction]:
            parser.error("run requires a direction and one of its known phases")
        result = execute(config, args.output, args.ledger, args.direction, args.phase,
                         wait_idle=args.wait_idle_seconds, timeout=args.timeout_seconds,
                         reserve_for=args.reserve_for)
        report(args.output, args.ledger)
    else:
        for number, phases in PHASES.items():
            for phase in phases:
                result = execute(config, args.output, args.ledger, number, phase,
                                 wait_idle=args.wait_idle_seconds, timeout=args.timeout_seconds)
                print(json.dumps({"direction": number, "phase": phase,
                                  "status": result["status"], "reason": result.get("reason")}), flush=True)
                report(args.output, args.ledger)
                if result["status"] != "passed" or result.get("eligible") is False:
                    break
        result = report(args.output, args.ledger)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
