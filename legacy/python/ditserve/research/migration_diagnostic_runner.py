"""Five-stage entry for independently frozen cache-off migration diagnosis."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .budget import BudgetStop, WorkerCleanupError
from .common import atomic_json, digest, read_json
from .dual_budget import DualResearchBudget
from .dual_evidence import invalidate_pending, invalidate_observation, read_records
from .dual_matrix import seal
from .dual_runner import environment, cpu_validate
from .isolation import gpu_occupancy, GPUIsolationError
from .runner import terminate_process_group
from .cfg_runner import bound_file, write_summary
from .migration_diagnostic_protocol import (ROOT, DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_LEDGER,
    DEFAULT_PARENT, freeze_protocol, verify_protocol, validate_protocol)


def completed(root, frozen, phase):
    from .migration_diagnostic_experiment import recompute_checks
    path = Path(root) / (phase + ".json")
    attempt_path = Path(root) / (phase + "-attempt.json")
    if not path.exists() or not attempt_path.exists():
        return False
    summary, attempt = read_json(path), read_json(attempt_path)
    if (summary.get("sha256") != digest({k: v for k, v in summary.items() if k != "sha256"})
            or summary.get("protocol_sha256") != frozen["sha256"]
            or summary.get("source_sha256") != frozen["source_sha256"]
            or summary.get("status") != "completed" or summary.get("checks_passed") is not True
            or summary.get("completed_units") != 4 or summary.get("expected_units") != 4
            or attempt.get("sha256") != digest({k: v for k, v in attempt.items() if k != "sha256"})
            or attempt.get("protocol_sha256") != frozen["sha256"]
            or attempt.get("status") != "completed" or attempt.get("returncode") != 0):
        return False
    rows = [r for r in read_records(root, frozen) if r.get("comparison") == phase]
    expected = {(order, c["prompt_id"], c["seed"]) for order in frozen["diagnostic"]["repetitions"]
                for c in frozen["diagnostic"]["cases"]}
    return (len(rows) == 4 and {(r.get("order"), r.get("prompt_id"), r.get("seed")) for r in rows} == expected
            and set(summary.get("row_sha256s", [])) == {r["sha256"] for r in rows}
            and all(r.get("type") == "migration_diagnostic" and r.get("performance_valid") is True
                    and r.get("status") == "completed" and r.get("checks_passed") is True
                    and set(r.get("checks", {})) == {"a100_local_restore", "v100_local_restore",
                        "migrated_prefix_equal", "same_input", "source_shadow_matches_native", "target_shadow_matches_migration",
                        "source_unchanged_and_paused"}
                    and all(v is True for v in r["checks"].values())
                    and r["checks"] == recompute_checks(r["runs"], r["shadow"]) for r in rows))


def pending_plan(root, frozen, ledger, original):
    if ledger.get("active"):
        raise ValueError("pending admission needs settled ledger totals")
    if (original.get("sha256") != digest({k: v for k, v in original.items() if k != "sha256"})
            or original.get("protocol_sha256") != frozen["sha256"] or original.get("status") != "fits"):
        raise ValueError("missing whole diagnostic matrix admission")
    phases = [p for p in ("original", "math") if not completed(root, frozen, p)]
    required = sum(original["comparisons"][p]["total_gpu_seconds"] for p in phases)
    allocation = ledger["reallocations"][-1]
    remaining = allocation["stage_allowances_gpu_seconds"]["migration"] - (
        ledger["stage_used_gpu_seconds"].get("migration", 0.) - allocation["baseline_stage_used"].get("migration", 0.))
    lifetime = ledger["limit_gpu_seconds"] - ledger["used_gpu_seconds"]
    return seal(dict(status="fits" if required < min(remaining, lifetime) else "budget_stopped",
        pending_phases=phases, required_gpu_seconds=required, remaining_migration_gpu_seconds=remaining,
        remaining_lifetime_gpu_seconds=lifetime, protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"]))


def supervise(phase, frozen, root, devices):
    bound_file(root, "cpu-validation", frozen)
    if phase not in ("original", "math") or phase == "math" and not completed(root, frozen, "original"):
        raise ValueError("MATH requires every original local restore and shadow neutrality check")
    path = root / (phase + "-attempt.json")
    if path.exists() and read_json(path).get("gpu_work_started"):
        raise ValueError("attempted diagnostic phase must remain immutable; freeze a new run for retry")
    observed = gpu_occupancy([d["uuid"] for d in devices])
    if observed["status"] != "exclusive_observed":
        value = seal(dict(status="blocked_contention", gpu_work_started=False, observation=observed,
                          protocol_sha256=frozen["sha256"]))
        atomic_json(path, value)
        return value
    planning = bound_file(root, "execution-plan", frozen, ("fits",))
    budget = DualResearchBudget(frozen["ledger"])
    current = read_json(frozen["ledger"])
    latest = (current.get("protocol_revisions") or [{}])[-1].get("new_protocol_sha256",
        current["reallocations"][-1]["protocol_sha256"])
    if latest != frozen["sha256"]:
        budget.bind_protocol_revision(latest, frozen["sha256"],
            "cache-off migration numerical diagnosis; preserve all lifetime spending and original stage caps")
    configuration = frozen["diagnostic"]
    cleanup = configuration["cleanup_wall_seconds"]
    attempt = dict(status="started", phase=phase, gpu_work_started=False, protocol_sha256=frozen["sha256"])
    try:
        with budget.session("migration", [d["uuid"] for d in devices],
                planning["comparisons"][phase]["phase_startup_wall_seconds"], cleanup,
                protocol_sha256=frozen["sha256"]) as session:
            pending = pending_plan(root, frozen, dict(session._data, active=None), planning)
            if pending["status"] != "fits":
                raise BudgetStop("remaining complete numerical control matrix exceeds unchanged migration allowance")
            session.check(pending["required_gpu_seconds"] / len(devices), 0.)
            atomic_json(root / (phase + "-budget-admission.json"), pending)
            timeout = min(session.check() - cleanup, configuration["timeout_wall_seconds"][phase])
            job = dict(phase=phase, output=str(root), physical_devices=devices, all_physical_devices=devices,
                deadline_monotonic=time.monotonic() + timeout,
                unit_estimate_seconds=planning["comparisons"][phase]["unit_wall_seconds"],
                session_id=session._data["active"]["session_id"], budget_stage="migration",
                protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"])
            job_path = root / (phase + "-job.json")
            atomic_json(job_path, job)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(d["uuid"] for d in devices),
                PYTHONPATH=str(ROOT / "python"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                TORCH_HOME=str(ROOT / ".cache/torch"), OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
            process = None
            with (root / (phase + ".log")).open("w") as stream:
                try:
                    session.expect_workers()
                    process = subprocess.Popen([sys.executable, "-m", "ditserve.research.migration_diagnostic_runner",
                        "_worker", str(job_path)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    session.register_worker_group(process.pid)
                    attempt["gpu_work_started"] = True
                    atomic_json(path, attempt)
                    end = time.monotonic() + timeout
                    while True:
                        start = time.time()
                        observation = gpu_occupancy([d["uuid"] for d in devices], allowed_pgid=process.pid)
                        finish = time.time()
                        observation.update(query_started_at=start, query_ended_at=finish)
                        with (root / (phase + "-isolation.jsonl")).open("a") as out:
                            out.write(json.dumps(observation, sort_keys=True) + "\n")
                        if observation["status"] != "exclusive_observed":
                            invalidate_observation(root, frozen, start, finish, "diagnostic GPU ownership lost")
                            raise GPUIsolationError("foreign or unknown GPU owner during numerical control")
                        if time.monotonic() >= end:
                            raise BudgetStop("diagnostic stage deadline exhausted")
                        try:
                            code = process.wait(timeout=min(2., end - time.monotonic()))
                            attempt.update(status="completed" if code == 0 else "budget_stopped" if code == 3 else "failed",
                                           returncode=code)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                except GPUIsolationError as error:
                    attempt.update(status="blocked_contention", reason=str(error))
                except BudgetStop as error:
                    attempt.update(status="budget_stopped", reason=str(error))
                except BaseException as error:
                    attempt.update(status="failed", reason=type(error).__name__ + ": " + str(error))
                    raise
                finally:
                    try:
                        if process is None:
                            session.confirm_no_workers_started()
                        else:
                            terminate_process_group(process)
                    except BaseException as error:
                        attempt.update(status="failed", cleanup_status="unconfirmed", reason=str(error))
                        raise WorkerCleanupError("diagnostic workers not confirmed stopped; preserve active ledger") from error
                    finally:
                        invalidate_pending(root, attempt.get("reason", attempt["status"]))
                        attempt = seal(attempt)
                        atomic_json(path, attempt)
                        session.set_outcome(attempt["status"])
    except BudgetStop as error:
        attempt.update(status="budget_stopped", reason=str(error))
        attempt.pop("sha256", None)
        attempt = seal(attempt)
        atomic_json(path, attempt)
    return attempt


def worker_main(path):
    job = read_json(path)
    frozen = verify_protocol(Path(job["output"]) / "protocol.json")
    deadline = time.monotonic() + 2
    while True:
        active = read_json(frozen["ledger"]).get("active") or {}
        if active.get("worker_pgid") is not None or time.monotonic() >= deadline:
            break
        time.sleep(.01)
    if (active.get("pid") != os.getppid() or active.get("worker_pgid") != os.getpgrp()
            or active.get("session_id") != job["session_id"] or active.get("protocol_sha256") != frozen["sha256"]
            or active.get("stage") != "migration" or job.get("budget_stage") != "migration"
            or job.get("phase") not in ("original", "math")
            or job.get("protocol_sha256") != frozen["sha256"] or job.get("source_sha256") != frozen["source_sha256"]
            or active.get("devices") != [d["uuid"] for d in job["physical_devices"]]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(active.get("devices", []))
            or type(job.get("deadline_monotonic")) not in (int, float)
            or not time.monotonic() < job["deadline_monotonic"] < float("inf")):
        raise RuntimeError("diagnostic worker requires matching parent session and physical devices")
    from .migration_diagnostic_experiment import run
    try:
        result = run(job)
        return 0 if result["status"] == "completed" else 4
    except BudgetStop:
        return 3


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        return worker_main(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "probe", "calibrate", "mechanism", "report"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--kind", choices=("original", "math", "all"), default="all")
    parser.add_argument("--validate-cpu", action="store_true")
    args = parser.parse_args(argv)
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        from .migration_diagnostic_report import report
        value = report(root)
        from ..result_storage import export_report
        export_report(root)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if read_json(args.ledger).get("active"):
        print(json.dumps(dict(status="blocked", reason="existing GPU session must settle")))
        return 2
    if args.stage in ("calibrate", "mechanism"):
        frozen = verify_protocol(root / "protocol.json")
        value = write_summary(root, args.stage, frozen, dict(status="deferred", gpu_work_started=False,
            reason="numerical diagnostic results grant no cache or migration performance qualification"))
        print(json.dumps(value)); return 0
    parent = read_json(args.parent_run / "protocol.json") if args.stage == "preflight" else read_json(root / "protocol.json")
    observed = environment(parent["config"], args.devices.split(","))
    if observed["status"] != "ready":
        atomic_json(root / "preflight-observation.json", observed)
        print(json.dumps(observed)); return 2
    if args.stage == "preflight":
        from .dual_protocol import inspect_model_snapshot
        from .retention import idle_guard
        from .migration_diagnostic_cost import cost_plan
        with idle_guard(root, [args.ledger]):
            validate_protocol(read_json(args.protocol))
            manifest = inspect_model_snapshot(hash_files=True)
            frozen = freeze_protocol(args.protocol, root, args.ledger, parent_run=args.parent_run, model_manifest=manifest)
            planning = cost_plan(frozen, read_json(args.ledger))
            write_summary(root, "execution-plan", frozen, planning)
            value = dict(observed, protocol_sha256=frozen["sha256"], execution_plan=planning)
            if planning["status"] != "fits": value["status"] = "blocked"
            if args.validate_cpu:
                value["cpu_validation"] = cpu_validate(frozen, root)
                if value["cpu_validation"]["status"] != "passed": value["status"] = "blocked"
        atomic_json(root / "preflight.json", value)
        atomic_json(root / "pending-references.json", dict(pending=True, reason="numerical controls pending"))
    else:
        frozen = verify_protocol(root / "protocol.json")
        attempts = []
        for phase in ("original", "math") if args.kind == "all" else (args.kind,):
            if completed(root, frozen, phase):
                continue
            attempts.append(supervise(phase, frozen, root, observed["devices"]))
            if attempts[-1]["status"] != "completed" or not completed(root, frozen, phase):
                break
        value = dict(status="completed" if all(completed(root, frozen, p) for p in ("original", "math"))
                     else "partial_or_stopped", attempts=attempts)
        atomic_json(root / "probe-observation.json", value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] in ("ready", "completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
