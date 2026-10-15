"""Research stage supervisor. This parent accounts for every allocated GPU."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .budget import BudgetStop, ResearchBudget, WorkerCleanupError
from .common import atomic_json, digest, file_digest, read_json, process_group_alive
from .preflight import cpu_validation, inspect_environment
from .isolation import GPUIsolationError, wait_for_exclusive_worker
from .protocol import DEFAULT_CONFIG, DEFAULT_LEDGER, DEFAULT_OUTPUT, ROOT, freeze_protocol, verify_protocol


def validate_worker_job(job, frozen):
    """A private worker must be a child of the parent holding the live ledger."""
    active = read_json(frozen["ledger"]).get("active") or {}
    # The parent persists the child group immediately after Popen. Waiting here
    # is CPU-only and avoids a launch race without permitting an unbound worker.
    end = time.monotonic() + 2
    while active.get("workers_expected") and active.get("worker_pgid") is None and time.monotonic() < end:
        time.sleep(.01)
        active = read_json(frozen["ledger"]).get("active") or {}
    expected_stage = "calibrate" if job["stage"] == "mechanism" else job["stage"]
    if job.get("use_reserve") is True:
        expected_stage = "reserve"
    uuids = [d["uuid"] for d in job["physical_devices"]]
    if (not active.get("session_id") or job.get("session_id") != active["session_id"]
            or active.get("pid") != os.getppid() or active.get("devices") != uuids
            or active.get("gpu_count") != len(uuids) or active.get("stage") != expected_stage
            or job.get("protocol_sha256") != frozen["sha256"]
            or job.get("source_sha256") != frozen["source_sha256"]
            or (active.get("workers_expected") and active.get("worker_pgid") != os.getpgrp())
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(uuids)
            or not isinstance(job.get("deadline_monotonic"), (int, float))
            or job["deadline_monotonic"] <= time.monotonic()):
        raise RuntimeError("worker has no matching live parent budget/device/protocol reservation")


def terminate_process_group(process):
    """Reap all ranks even if their launcher exited before its children."""
    def alive():
        return process_group_alive(process.pid)
    if alive():
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        end = time.monotonic() + 5
        while alive() and time.monotonic() < end:
            time.sleep(.05)
        if alive():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired as error:
        raise WorkerCleanupError("worker launcher not reaped; reservation remains active") from error
    end = time.monotonic() + 5
    while alive() and time.monotonic() < end:
        time.sleep(.05)
    if alive():
        raise WorkerCleanupError("GPU worker process group has not exited; ledger remains active")


def _valid_cpu(path, frozen):
    if not path.exists():
        return False
    record = read_json(path)
    unsigned = {k: v for k, v in record.items() if k != "sha256"}
    return (record.get("sha256") == digest(unsigned) and record.get("status") == "passed"
            and record.get("source_sha256") == frozen["source_sha256"]
            and record.get("protocol_sha256") == frozen["sha256"])


def stage_prerequisites(stage, output, frozen):
    output = Path(output)
    missing = []
    if not _valid_cpu(output / "cpu-validation.json", frozen):
        missing.append("passing CPU/Gloo validation for this frozen implementation")
    dependency_stages = [] if stage == "correctness" else ["correctness"]
    if stage in ("calibrate", "mechanism", "sp"):
        dependency_stages.append("probe")
    if stage in ("mechanism", "sp"):
        dependency_stages.append("calibrate")
    environment_path = output / "measurement-environment.json"
    if environment_path.exists():
        environment = read_json(environment_path)
        unsigned = {k: v for k, v in environment.items() if k != "sha256"}
        if (environment.get("sha256") != digest(unsigned)
                or environment.get("protocol_sha256") != frozen["sha256"]
                or environment.get("source_sha256") != frozen["source_sha256"]
                or environment.get("cost_measurements_valid") is not True):
            missing.append("valid GPU isolation observations for this run; preserve contaminated evidence and use a new run")
        for previous in dependency_stages:
            observation = environment.get("stages", {}).get(previous, {})
            log_path = output / (previous + "-gpu-isolation.jsonl")
            if (observation.get("status") != "exclusive_observed"
                    or observation.get("completion_observed") is not True
                    or observation.get("worker_returncode") != 0
                    or observation.get("observations_count", 0) < 2
                    or observation.get("log") != log_path.name
                    or not log_path.is_file()
                    or observation.get("log_sha256") != file_digest(log_path)):
                missing.append("complete GPU isolation observations for " + previous)
    elif dependency_stages:
        missing.append("GPU isolation observations missing for prerequisite stages")
    for previous in dependency_stages:
        attempt_path = output / (previous + "-attempt.json")
        attempt = read_json(attempt_path) if attempt_path.exists() else {}
        if (attempt.get("status") != "completed" or attempt.get("returncode") != 0
                or attempt.get("protocol_sha256") != frozen["sha256"]):
            missing.append("completed supervisor lifecycle for " + previous)
    if stage != "correctness":
        required = [("correctness.json", "status", "passed")]
        if stage in ("calibrate", "mechanism", "sp"):
            required += [("probe.json", "status", "completed")]
        if stage in ("mechanism", "sp"):
            required += [("calibration.json", "status", "selected")]
        for filename, field, expected in required:
            path = output / filename
            record = read_json(path) if path.exists() else {}
            if record.get(field) != expected or record.get("protocol_sha256") != frozen["sha256"]:
                missing.append(filename + " with " + field + "=" + expected)
    return missing


def run_gpu_stage(stage, frozen, output, requested, model_path=None, *, use_reserve=False):
    output = Path(output)
    config = frozen["config"]
    environment = inspect_environment(frozen, requested, model_path)
    missing = stage_prerequisites(stage, output, frozen)
    if stage == "sp" and len(requested) != 4:
        missing.append("four explicitly selected homogeneous GPUs")
    if stage != "sp" and len(requested) != 1:
        missing.append("single-GPU stages require exactly one selected GPU")
    if environment["status"] != "ready":
        missing.extend(environment["reasons"])
    if stage in ("probe", "calibrate", "mechanism") and not environment.get("offline_quality_weights_ready"):
        missing.append("offline AlexNet/LPIPS weights bound to the frozen protocol")
    if missing:
        record = dict(stage=stage, status="blocked", reasons=missing, environment=environment,
                      protocol_sha256=frozen["sha256"], gpu_work_started=False)
        atomic_json(output / (stage + "-attempt.json"), record)
        return record
    if (output / (stage + "-attempt.json")).exists():
        previous = read_json(output / (stage + "-attempt.json"))
        if previous.get("gpu_work_started"):
            raise ValueError("GPU stage was already attempted; preserve evidence and use a new run directory with the same ledger")
    budget_stage = "reserve" if use_reserve else "calibrate" if stage == "mechanism" else stage
    budget = ResearchBudget(frozen["ledger"], config["budget_gpu_seconds"])
    selected = environment["selection"]["devices"]
    devices = [d["uuid"] for d in selected]
    cleanup = config["cleanup_wall_seconds"]
    attempt = dict(stage=stage, budget_stage=budget_stage, status="started", protocol_sha256=frozen["sha256"], gpu_work_started=True)
    try:
        estimate = config["estimate_wall_seconds"][stage]
        plan_path = output / "cost-plan.json"
        if plan_path.exists():
            from .cost_planning import read_bound_cost_plan
            estimate = read_bound_cost_plan(output, frozen).get("unit_estimates", {}).get(stage, estimate)
        with budget.session(budget_stage, devices, estimate, cleanup) as session:
            remaining_wall = session.check()
            timeout = min(config["timeout_wall_seconds"][stage], remaining_wall - cleanup)
            job = dict(stage=stage, output=str(output), model_path=environment["model"]["path"],
                       use_reserve=bool(use_reserve),
                       physical_devices=selected, deadline_monotonic=time.monotonic() + timeout - cleanup,
                       session_id=session._data["active"]["session_id"],
                       protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"])
            job_path = output / (stage + "-job.json")
            atomic_json(job_path, job)
            atomic_json(output / (stage + "-attempt.json"), attempt)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices), HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       TORCH_HOME=str(ROOT / ".cache/torch"),
                       OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                       PYTHONPATH=str(ROOT / "python"), PYTHONUNBUFFERED="1")
            process = None
            with (output / (stage + ".log")).open("w") as log:
                try:
                    session.expect_workers()
                    process = subprocess.Popen([sys.executable, "-m", "ditserve.research.runner", "_worker", str(job_path)],
                                               cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    session.register_worker_group(process.pid)
                    returncode = wait_for_exclusive_worker(process, timeout=timeout, output=output,
                        frozen=frozen, stage=stage, devices=devices)
                    attempt.update(status="completed" if returncode == 0 else "budget_stopped" if returncode == 3 else "failed", returncode=returncode)
                except GPUIsolationError as error:
                    attempt.update(status="blocked_contention", reason=str(error))
                except subprocess.TimeoutExpired:
                    attempt.update(status="budget_stopped", reason="worker exceeded complete-unit estimate or stage deadline")
                finally:
                    if process is not None:
                        try:
                            terminate_process_group(process)
                        except WorkerCleanupError:
                            raise
                        except BaseException as error:
                            # A second interrupt or failed signal must not make
                            # still-resident ranks disappear from the ledger.
                            raise WorkerCleanupError("worker cleanup was interrupted or unconfirmed; reservation remains active") from error
                    # The budget remains open through worker exit, including
                    # destruction of the CUDA context and failed allocations.
            atomic_json(output / (stage + "-attempt.json"), attempt)
            session.set_outcome(attempt["status"])
    except BudgetStop as error:
        attempt.update(status="budget_stopped", reason=str(error), gpu_work_started=False)
        atomic_json(output / (stage + "-attempt.json"), attempt)
    return attempt


def worker_main(job_path):
    import gc
    from .experiment import Experiment, IneligibleAction
    job = read_json(job_path)
    experiment = Experiment(job)
    # Detect code/dependency drift before importing model code or loading CUDA.
    frozen = verify_protocol(Path(job["output"]) / "protocol.json")
    validate_worker_job(job, frozen)
    if job["stage"] == "sp":
        from .sp_experiment import run_static_sp
        result = run_static_sp(job)
        if result.get("status") != "measured_unqualified" or result.get("elastic", {}).get("status") == "failed":
            return 1
        return 3 if result.get("elastic", {}).get("status") == "budget_not_fit" else 0
    try:
        experiment.load()
        getattr(experiment, job["stage"])()
        return 0
    except (BudgetStop, IneligibleAction) as error:
        experiment.record(dict(type="stage_stop", case_id=job["stage"], status="insufficient", reason=str(error)))
        return 3
    except BaseException as error:
        experiment.record(dict(type="stage_failure", case_id=job["stage"], status="failed",
                               error=type(error).__name__ + ": " + str(error)))
        raise
    finally:
        if experiment.adapter is not None:
            experiment.adapter.close()
        experiment.adapter = experiment.worker = None
        gc.collect()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        return worker_main(argv[1])
    parser = argparse.ArgumentParser(description="Quality-constrained DiT cache/preemption/SP research pilot")
    parser.add_argument("stage", choices=("preflight", "probe", "calibrate", "mechanism", "report"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_CONFIG, help="unfrozen pilot configuration")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", default="0", help="one physical index/UUID, or four comma-separated homogeneous GPUs")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--budget-gpu-hours", type=float, default=8.)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--validate-cpu", action="store_true", help="run the CPU/Gloo contract suite during preflight")
    parser.add_argument("--use-reserve", action="store_true", help="explicitly charge a necessary recheck to the separate 3,600 GPU-second reserve")
    parser.add_argument("--kind", choices=("state", "sp"), default="state", help="mechanism stage kind")
    args = parser.parse_args(argv)
    if args.budget_gpu_hours != 8:
        parser.error("this frozen first-pilot protocol uses exactly 8 GPU-hours; the ledger enforces the cap")
    output = args.output.resolve()
    if args.stage == "report":
        from .report import build_report
        frozen = verify_protocol(output / "protocol.json", verify_current=False)
        result = build_report(output, frozen)
        from ..result_storage import export_report
        export_report(output, ledger=frozen["ledger"])
    else:
        frozen = freeze_protocol(args.protocol, output, args.ledger)
        ResearchBudget(frozen["ledger"]).initialize()
        requested = [token.strip() for token in args.devices.split(",") if token.strip()]
        if args.stage == "preflight":
            environment = inspect_environment(frozen, requested, args.model_path)
            if args.validate_cpu:
                cpu_validation(frozen, output)
            result = dict(environment, protocol_sha256=frozen["sha256"],
                          cpu_validation="passed" if _valid_cpu(output / "cpu-validation.json", frozen) else "unverified")
            atomic_json(output / "preflight.json", result)
            from .matrix import build_matrix
            atomic_json(output / "matrix.json", build_matrix(frozen["config"], protocol_sha256=frozen["sha256"]))
        else:
            stage = "sp" if args.stage == "mechanism" and args.kind == "sp" else args.stage
            if stage == "probe" and not (output / "correctness.json").exists():
                initial = run_gpu_stage("correctness", frozen, output, requested, args.model_path, use_reserve=args.use_reserve)
                if initial["status"] != "completed":
                    print(json.dumps(initial, ensure_ascii=False, indent=2))
                    return 2
            result = run_gpu_stage(stage, frozen, output, requested, args.model_path, use_reserve=args.use_reserve)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in ("ready", "completed", "passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
