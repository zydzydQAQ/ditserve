"""Five-stage entry for the bounded two-GPU pilot, using the original ledger."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .budget import BudgetStop, WorkerCleanupError
from .common import atomic_json, digest, read_json
from .dual_budget import DualResearchBudget
from .dual_evidence import invalidate_pending, invalidate_observation, read_records
from .dual_matrix import seal, estimate_matrix
from .dual_protocol import (DEFAULT_CONFIG, DEFAULT_LEDGER, DEFAULT_OUTPUT, ROOT, freeze_protocol,
                            inspect_model_snapshot, verify_protocol, validate_protocol)
from .isolation import GPUIsolationError, gpu_occupancy
from .preflight import discover_gpus
from .runner import terminate_process_group

PHASE_BUCKET = dict(probe="correctness", **{"calibrate-a100": "calibration", "calibrate-v100": "calibration"},
                    migration="migration", mechanism="mechanism", capacity="capacity")


def fallback_backend_cost_estimate(root, frozen):
    """An upper measured candidate cost for conditional admission, not selection."""
    rows = read_records(root, frozen)
    pairs = [r for r in rows if r.get("type") == "backend_pair"
             and r.get("shape") == frozen["config"]["shapes"]["fallback"]
             and r.get("performance_valid") is True]
    if len(pairs) != 2:
        return None
    pairs.sort(key=lambda r: r["case_id"])
    if ([r.get("order") for r in pairs] != [["math", "efficient"], ["efficient", "math"]]
            or len({r.get("context_sha256") for r in pairs}) != 1):
        return None
    costs, units = [], []
    for row in pairs:
        cost = row.get("costs_ms", {}).get("efficient")
        run = row.get("runs", {}).get("efficient", {})
        if (type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0
                or cost != run.get("elapsed_ms") or run.get("recipe_id") != "off"
                or run.get("device_role") != "v100" or run.get("action") != "stay"
                or run.get("identity", {}).get("attention_path") != ["efficient"]):
            raise ValueError("invalid sealed EFFICIENT planning cost")
        costs.append(cost / 1000.)
        unit = read_json(root / "units" / (row["unit_id"] + ".json"))
        units.append(dict(unit_id=row["unit_id"], sha256=unit["sha256"]))
    job = read_json(root / "probe-job.json")
    return dict(validated=True, shape=frozen["config"]["shapes"]["fallback"], backend="efficient",
                v100_seconds=max(costs), source_run=str(root), source_protocol_sha256=frozen["sha256"],
                source_sha256=frozen["source_sha256"], source_units=units,
                physical_devices=job.get("all_physical_devices", job["physical_devices"]),
                purpose="conditional_cost_estimate_only; fresh_backend_qualification_required")


def probe_retry_inputs(previous_root, frozen):
    """Reuse only sealed costs for admission, never prior quality qualification."""
    previous_root = Path(previous_root).resolve()
    previous = verify_protocol(previous_root / "protocol.json", verify_current=False)
    attempt = read_json(previous_root / "probe-attempt.json")
    if (attempt.get("sha256") != digest({k: v for k, v in attempt.items() if k != "sha256"})
            or attempt.get("protocol_sha256") != previous["sha256"]
            or not attempt.get("gpu_work_started")
            or attempt.get("status") not in ("budget_stopped", "failed")
            or previous["ledger"] != frozen["ledger"]
            or previous["config"] != frozen["config"]
            or previous["model_manifest"]["sha256"] != frozen["model_manifest"]["sha256"]):
        raise ValueError("retry requires a settled unsuccessful probe of the identical scientific recipe")
    source_root, source = previous_root, previous
    inherited = None
    if not (previous_root / "units/probe-main-baselines.json").exists():
        inherited = read_json(previous_root / "probe-retry-inputs.json")
        if (inherited.get("sha256") != digest({k: v for k, v in inherited.items() if k != "sha256"})
                or inherited.get("validated") is not True):
            raise ValueError("invalid inherited planning evidence")
        source_root = Path(inherited["source_run"]).resolve()
        if source_root == previous_root:
            raise ValueError("planning evidence cannot reference itself")
        source = verify_protocol(source_root / "protocol.json", verify_current=False)
        if (source["config"] != frozen["config"] or source["ledger"] != frozen["ledger"]
                or source["model_manifest"]["sha256"] != frozen["model_manifest"]["sha256"]):
            raise ValueError("inherited planning source uses a different scientific recipe")
    rows = read_records(source_root, source)
    costs = {}
    for role in ("a100", "v100"):
        selected = [r for r in rows if r.get("type") == "probe" and r.get("case_id") == role + "-main"
                    and r.get("unit_id") == "probe-main-baselines" and r.get("performance_valid") is True]
        if len(selected) != 1:
            raise ValueError("retry planning requires independently sealed complete main-shape costs")
        run = selected[0]["run"]
        cost = run["elapsed_ms"] / 1000.
        if (run.get("recipe_id") != "off" or run.get("action") != "stay"
                or not math.isfinite(cost) or cost <= 0):
            raise ValueError("invalid no-cache planning cost")
        costs[role] = cost
    unit = read_json(source_root / "units/probe-main-baselines.json")
    result = dict(shape=frozen["config"]["shapes"]["main"], measured_costs_seconds=costs,
                  source_run=str(source_root), source_protocol_sha256=source["sha256"],
                  source_sha256=source["source_sha256"], source_unit_sha256=unit["sha256"], validated=True,
                  purpose="admission_estimates_only; no_quality_or_backend_qualification_reuse")
    if inherited is not None:
        if any(inherited.get(key) != value for key, value in result.items()):
            raise ValueError("inherited costs differ from independently sealed source evidence")
        result["retry_predecessor"] = dict(run=str(previous_root), protocol_sha256=previous["sha256"],
                                         attempt_sha256=attempt["sha256"])
    candidate_estimate = fallback_backend_cost_estimate(previous_root, previous)
    if candidate_estimate is not None:
        result["fallback_backend_cost_estimate"] = candidate_estimate
    return seal(result)


def retry_inputs(root, frozen):
    path = root / "probe-retry-inputs.json"
    if not path.exists():
        return None
    value = read_json(path)
    expected = probe_retry_inputs(value.get("retry_predecessor", {}).get("run", value["source_run"]), frozen)
    if value != expected:
        raise ValueError("retry planning evidence changed")
    return value


def environment(config, requested):
    discovered = discover_gpus()
    devices, reasons = [], []
    for token in requested:
        matches = [d for d in discovered["devices"] if token in (str(d["index"]), d["uuid"])]
        if len(matches) != 1:
            reasons.append("requested physical device unavailable: " + token)
        else:
            devices.append(matches[0])
    if (len(devices) != 2 or len({d["uuid"] for d in devices}) != 2
            or sum("A100" in d["name"] for d in devices) != 1
            or sum("V100" in d["name"] for d in devices) != 1):
        reasons.append("require exactly one A100 and one V100, each running SP1")
    devices.sort(key=lambda d: ("V100" in d["name"], d["uuid"]))
    versions = {}
    for package, expected in config["dependencies"].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[package] = actual
        if actual != expected:
            reasons.append("pinned dependency mismatch: " + package)
    occupancy = gpu_occupancy([d["uuid"] for d in devices])
    if occupancy["status"] != "exclusive_observed":
        reasons.append("exclusive GPU window unavailable")
    return dict(status="blocked" if reasons else "ready", reasons=reasons, devices=devices,
                dependencies=versions, occupancy=occupancy, gpu_work_started=False)


def cpu_validate(frozen, root):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "python"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests/python"),
               "-p", "test_research*.py", "-v"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    log = result.stdout + result.stderr
    count = re.search(r"Ran (\d+) tests?", log)
    value = seal(dict(status="passed" if result.returncode == 0 and count and "skipped=" not in log else "failed",
                      test_count=int(count[1]) if count else 0, returncode=result.returncode,
                      source_sha256=frozen["source_sha256"], protocol_sha256=frozen["sha256"]))
    (root / "cpu-validation.log").write_text(log)
    atomic_json(root / "cpu-validation.json", value)
    return value


def prerequisites(root, frozen, phase):
    required = ["cpu-validation"] + ([] if phase == "probe" else ["probe"])
    if phase in ("migration", "mechanism", "capacity"):
        required.append("calibrate")
    if phase == "mechanism":
        required.append("migration")
    for name in required:
        path = root / (name + ".json")
        row = read_json(path) if path.exists() else {}
        if (row.get("sha256") != digest({k: v for k, v in row.items() if k != "sha256"})
                or row.get("status") not in ("passed", "completed")
                or row.get("protocol_sha256") != frozen["sha256"]
                or row.get("source_sha256") != frozen["source_sha256"]):
            raise ValueError("missing successful bound prerequisite: " + name)
        if name not in ("cpu-validation", "calibrate"):
            attempt = read_json(root / (name + "-attempt.json"))
            if attempt.get("status") != "completed" or attempt.get("returncode") != 0:
                raise ValueError("prerequisite supervisor did not complete: " + name)


def supervise(phase, frozen, root, devices):
    prerequisites(root, frozen, phase)
    path = root / (phase + "-attempt.json")
    if path.exists() and read_json(path).get("gpu_work_started"):
        raise ValueError("preserve attempted stage; use a new independently frozen run")
    selected = [d for d in devices if
                (phase == "capacity" and "A100" in d["name"])
                or (phase in ("calibrate-a100", "calibrate-v100") and phase.split("-")[1].upper() in d["name"])
                or phase not in ("capacity", "calibrate-a100", "calibrate-v100")]
    uuids = [d["uuid"] for d in selected]
    occupancy = gpu_occupancy(uuids)
    if occupancy["status"] != "exclusive_observed":
        value = seal(dict(status="blocked_contention", gpu_work_started=False, occupancy=occupancy,
                          protocol_sha256=frozen["sha256"]))
        atomic_json(path, value)
        return value
    budget, config = DualResearchBudget(frozen["ledger"]), frozen["config"]
    retry = retry_inputs(root, frozen) if phase == "probe" else None
    if retry:
        predecessor = retry.get("retry_predecessor", {}).get("protocol_sha256", retry["source_protocol_sha256"])
        budget.bind_protocol_revision(predecessor, frozen["sha256"],
            "retry failed probe from reserve; sealed prior costs used only for shape admission")
    budget.reallocate_remaining(protocol_sha256=frozen["sha256"])
    bucket = "reserve" if retry else PHASE_BUCKET[phase]
    cleanup = config["cleanup_wall_seconds"]
    estimate = config["estimate_wall_seconds"][bucket]
    probe_path = root / "probe.json"
    if phase != "probe" and probe_path.exists():
        costs = read_json(probe_path)["cost_plan"]["measured_costs_seconds"]
        a, v = costs["a100"], costs["v100"]
        estimate = (2 * costs[phase.split("-")[1]] * 1.3 + 30 if phase.startswith("calibrate-")
                    else max(2.5 * v + 1.5 * a, v + 2 * a) * 1.3 + 60 if phase == "migration"
                    else 3 * ((a + v) / 2 + 8 * a) * 1.3 + 60 if phase == "mechanism"
                    else 2 * a * 1.3 + 30)
    attempt = dict(status="started", phase=phase, gpu_work_started=False, protocol_sha256=frozen["sha256"])
    try:
        with budget.session(bucket, uuids, estimate, cleanup, protocol_sha256=frozen["sha256"]) as session:
            timeout = min(session.check() - cleanup, config["timeout_wall_seconds"][bucket])
            remaining = session.allocation_remaining()
            if retry:
                # Both obligations draw from one reserve pool. Protect the
                # planned later setup/evaluation/cleanup amount instead of
                # independently admitting it and correctness against all R.
                protected = estimate_matrix(retry["measured_costs_seconds"])["phases_gpu_seconds"]["reserve"]
                remaining["correctness"] = max(0., remaining["reserve"] - protected)
            job = dict(phase=phase, output=str(root), physical_devices=selected, all_physical_devices=devices,
                deadline_monotonic=time.monotonic() + timeout, unit_estimate_seconds=estimate,
                remaining_gpu_seconds=config["budget_gpu_seconds"] - session._data["used_gpu_seconds"],
                phase_remaining_gpu_seconds=remaining, session_id=session._data["active"]["session_id"],
                protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"], budget_stage=bucket)
            if retry:
                job["planning_seed"] = retry
            jobpath = root / (phase + "-job.json")
            atomic_json(jobpath, job)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(uuids), PYTHONPATH=str(ROOT / "python"),
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                TORCH_HOME=str(ROOT / ".cache/torch"), OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
            process = None
            with (root / (phase + ".log")).open("w") as stream:
                try:
                    session.expect_workers()
                    process = subprocess.Popen([sys.executable, "-m", "ditserve.research.dual_runner", "_worker", str(jobpath)],
                        env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    session.register_worker_group(process.pid)
                    attempt["gpu_work_started"] = True
                    atomic_json(path, attempt)
                    end = time.monotonic() + timeout
                    while True:
                        observation_started = time.time()
                        observed = gpu_occupancy(uuids, allowed_pgid=process.pid)
                        observation_ended = time.time()
                        observed.update(query_started_at=observation_started, query_ended_at=observation_ended)
                        with (root / (phase + "-isolation.jsonl")).open("a") as log:
                            log.write(json.dumps(observed, sort_keys=True) + "\n")
                        if observed["status"] != "exclusive_observed":
                            invalidate_observation(root, frozen, observation_started, observation_ended,
                                                   "parent GPU ownership observation: " + observed["status"])
                            raise GPUIsolationError("foreign process or unknown ownership during active unit")
                        if time.monotonic() >= end:
                            raise BudgetStop("GPU stage deadline exhausted")
                        try:
                            code = process.wait(timeout=min(2., end - time.monotonic()))
                            attempt.update(status="completed" if code == 0 else "budget_stopped" if code == 3 else "failed",
                                           returncode=code)
                            break
                        except subprocess.TimeoutExpired:
                            continue
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
                        raise WorkerCleanupError("worker cleanup unconfirmed; ledger remains active") from error
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


def worker_main(job_path):
    job = read_json(job_path)
    frozen = verify_protocol(Path(job["output"]) / "protocol.json")
    retry = retry_inputs(Path(job["output"]), frozen) if job["phase"] == "probe" else None
    expected_bucket = "reserve" if retry else PHASE_BUCKET[job["phase"]]
    deadline = time.monotonic() + 2
    while True:
        active = read_json(frozen["ledger"]).get("active") or {}
        if active.get("worker_pgid") is not None or time.monotonic() >= deadline:
            break
        time.sleep(.01)
    if (active.get("pid") != os.getppid() or active.get("worker_pgid") != os.getpgrp()
            or active.get("session_id") != job["session_id"] or active.get("protocol_sha256") != frozen["sha256"]
            or active.get("devices") != [d["uuid"] for d in job["physical_devices"]]
            or active.get("stage") != expected_bucket
            or job.get("budget_stage", PHASE_BUCKET[job["phase"]]) != expected_bucket
            or job.get("planning_seed") != retry
            or job.get("protocol_sha256") != frozen["sha256"]
            or job.get("source_sha256") != frozen["source_sha256"]
            or type(job.get("deadline_monotonic")) not in (int, float)
            or not math.isfinite(job["deadline_monotonic"]) or job["deadline_monotonic"] <= time.monotonic()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(active.get("devices", []))):
        raise RuntimeError("private worker requires matching parent ledger/process/device authorization")
    from .dual_experiment import run
    try:
        result = run(job)
        return 0 if result["status"] in ("completed", "quality_stopped", "direction_stopped", "unverified") else 3
    except BudgetStop:
        return 3


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        return worker_main(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "probe", "calibrate", "mechanism", "report"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--kind", choices=("migration", "mechanism", "capacity", "all"), default="all")
    parser.add_argument("--validate-cpu", action="store_true")
    parser.add_argument("--retry-from", type=Path, help="preflight only: failed probe whose sealed costs guide a reserve-funded retry")
    args = parser.parse_args(argv)
    if args.retry_from and args.stage != "preflight":
        parser.error("--retry-from is frozen during preflight only")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = validate_protocol(read_json(args.protocol))
    if args.stage == "report":
        from .dual_report import report
        value = report(root)
        attempts = [read_json(p) for p in root.glob("*-attempt.json")]
        terminal_failure = any(a.get("gpu_work_started") and a.get("status") in
                               ("failed", "budget_stopped", "blocked_contention") for a in attempts)
        terminal_stages = all((root / (name + ".json")).exists() for name in ("calibrate", "migration", "mechanism", "capacity"))
        if terminal_failure or terminal_stages:
            atomic_json(root / "pending-references.json", dict(pending=False,
                        reason="pipeline terminal; failed GPU stages require a new independent run"))
        from ..result_storage import export_report
        export_report(root)
    else:
        observed = environment(config, args.devices.split(","))
        ledger = read_json(args.ledger)
        if ledger.get("active"):
            value = dict(status="blocked", reasons=["existing GPU session must settle before hashing, reallocation or new work"],
                         active=ledger["active"], environment=observed, gpu_work_started=False)
            atomic_json(root / "preflight.json", value)
            print(json.dumps(value, indent=2))
            return 2
        if args.stage == "preflight":
            # No content scan while occupied even by another experiment.
            if observed["status"] != "ready":
                value = dict(observed, model=inspect_model_snapshot(hash_files=False))
                atomic_json(root / "preflight.json", value)
            else:
                manifest = inspect_model_snapshot(hash_files=True)
                frozen = freeze_protocol(args.protocol, root, args.ledger, model_manifest=manifest)
                if args.retry_from:
                    retry = probe_retry_inputs(args.retry_from, frozen)
                    retry_path = root / "probe-retry-inputs.json"
                    if retry_path.exists() and read_json(retry_path) != retry:
                        raise ValueError("preserve the original retry evidence; use a new run")
                    atomic_json(retry_path, retry)
                atomic_json(root / "pending-references.json", dict(pending=True,
                    reason="formal stages may still require baseline arrays; clear only at terminal pipeline decision"))
                value = dict(observed, model_manifest_sha256=manifest["sha256"], protocol_sha256=frozen["sha256"])
                if args.validate_cpu:
                    value["cpu_validation"] = cpu_validate(frozen, root)
                    if value["cpu_validation"]["status"] != "passed":
                        value["status"] = "blocked"
                atomic_json(root / "preflight.json", value)
        else:
            frozen = verify_protocol(root / "protocol.json")
            if observed["status"] != "ready":
                value = observed
            elif args.stage == "calibrate":
                attempts = []
                for role in ("a100", "v100"):
                    attempt = supervise("calibrate-" + role, frozen, root, observed["devices"])
                    attempts.append(attempt)
                    if attempt["status"] != "completed":
                        break
                gates = {}
                if len(attempts) == 2 and all(a["status"] == "completed" for a in attempts):
                    for role in ("a100", "v100"):
                        gates.update(read_json(root / ("calibrate-" + role + ".json"))["cache_gates"])
                value = seal(dict(status="completed" if len(gates) == 2 else "unverified", cache_gates=gates,
                                   protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"]))
                atomic_json(root / "calibrate.json", value)
            else:
                phases = ["probe"] if args.stage == "probe" else ["migration", "mechanism", "capacity"] if args.kind == "all" else [args.kind]
                value = None
                for phase in phases:
                    if phase == "mechanism" and args.kind == "all":
                        migration = read_json(root / "migration.json")
                        if migration.get("status") != "completed":
                            atomic_json(root / "mechanism.json", seal(dict(status="unverified",
                                reason="migration quality gate not passed", protocol_sha256=frozen["sha256"],
                                source_sha256=frozen["source_sha256"])))
                            continue
                    value = supervise(phase, frozen, root, observed["devices"])
                    if value["status"] != "completed":
                        break
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value.get("status") in ("ready", "completed", "passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
