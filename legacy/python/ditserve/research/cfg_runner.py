"""Five stages of the CFG study, with unchanged lifetime and stage budget caps."""
import argparse
import json
import math
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
from .dual_protocol import inspect_model_snapshot
from .dual_runner import cpu_validate, environment
from .cfg_protocol import DEFAULT_CONFIG, DEFAULT_LEDGER, DEFAULT_OUTPUT, ROOT, freeze_protocol, verify_protocol, validate_protocol
from .isolation import GPUIsolationError, gpu_occupancy
from .runner import terminate_process_group

PHASE_BUCKET = {kind + "-" + role: bucket for kind, bucket in
                (("correctness", "correctness"), ("screen", "calibration"),
                 ("calibrate", "calibration"), ("holdout", "calibration"))
                for role in ("a100", "v100")}
PHASE_BUCKET.update(migration="migration", mechanism="mechanism")


def bound_file(root, name, frozen, statuses=("passed", "completed")):
    path = root / (name + ".json")
    row = read_json(path) if path.exists() else {}
    if (row.get("sha256") != digest({k: v for k, v in row.items() if k != "sha256"})
            or row.get("status") not in statuses or row.get("protocol_sha256") != frozen["sha256"]
            or row.get("source_sha256") != frozen["source_sha256"]):
        raise ValueError("missing successful current-source prerequisite: " + name)
    return row


def prerequisites(root, frozen, phase):
    bound_file(root, "cpu-validation", frozen)
    if not phase.startswith("correctness-"):
        for role in ("a100", "v100"):
            bound_file(root, "correctness-" + role, frozen)
        if frozen.get("implementation_refinement"):
            from .cfg_cost_plan import _completed
            rows = read_records(root, frozen)
            if not all(_completed(root, frozen, "correctness-" + role, rows, ["paired_008"])
                       for role in ("a100", "v100")):
                raise ValueError("refinement requires all three current-source correctness trajectories on both devices")
    if phase.startswith(("calibrate-", "holdout-")) or phase in ("migration", "mechanism"):
        bound_file(root, "probe", frozen)
    if phase.startswith("holdout-"):
        selected = bound_file(root, "selection", frozen, ("selected",))
        if not selected.get("selected"):
            raise ValueError("holdout requires frozen development selection")
    if phase in ("migration", "mechanism"):
        gate = bound_file(root, "calibrate", frozen)
        if not gate.get("eligible"):
            raise ValueError("selected strategy did not pass development and held-out checks")
        from .cfg_matrix import select_candidate, holdout_gate
        context = bound_file(root, "probe", frozen)["context_sha256"]
        rows = read_records(root, frozen)
        selection = select_candidate(rows, frozen["config"], context, frozen=frozen)
        locked = bound_file(root, "selection", frozen, ("selected",))
        if (not selection.get("selected") or selection["selected"] != locked.get("selected")
                or not holdout_gate(rows, frozen["config"], selection["selected"], context)["eligible"]):
            raise ValueError("sealed CFG quality/cost evidence did not pass")
    if phase == "mechanism":
        migration = bound_file(root, "migration", frozen)
        from .cfg_matrix import migration_quality_gate
        selected = locked["selected"]
        qualified = [action for action in ("full_migrate", "cold_migrate") if all(
            migration_quality_gate(rows, frozen["config"], split, selected, action, context)["eligible"]
            for split in ("dev", "holdout"))]
        if not qualified or qualified != migration.get("qualified_actions"):
            raise ValueError("migration action evidence is incomplete or changed")


def supervise(phase, frozen, root, devices):
    prerequisites(root, frozen, phase)
    path = root / (phase + "-attempt.json")
    if path.exists() and read_json(path).get("gpu_work_started"):
        raise ValueError("preserve attempted stage; use a new independently frozen run")
    selected = [d for d in devices if phase in ("migration", "mechanism")
                or phase.rsplit("-", 1)[-1].upper() in d["name"]]
    uuids = [d["uuid"] for d in selected]
    occupancy = gpu_occupancy(uuids)
    if occupancy["status"] != "exclusive_observed":
        value = seal(dict(status="blocked_contention", gpu_work_started=False, occupancy=occupancy,
                          protocol_sha256=frozen["sha256"]))
        atomic_json(path, value)
        return value
    budget, config = DualResearchBudget(frozen["ledger"]), frozen["config"]
    data = read_json(frozen["ledger"])
    if not data.get("reallocations"):
        raise ValueError("CFG research requires the existing settled two-GPU allocation")
    previous = (data.get("protocol_revisions") or [{}])[-1].get("new_protocol_sha256",
                  data["reallocations"][-1]["protocol_sha256"])
    if previous != frozen["sha256"]:
        budget.bind_protocol_revision(previous, frozen["sha256"],
            "new CFG study; preserve cumulative spending and original stage caps")
    bucket = PHASE_BUCKET[phase]
    cleanup = config["cleanup_wall_seconds"]
    estimate = config["estimate_wall_seconds"][bucket]
    planning = None
    if frozen.get("prototype_continuation"):
        from .cfg_cost_plan import pending_cost_plan
        planning = bound_file(root, "execution-plan", frozen, ("fits",))
        pending = pending_cost_plan(frozen, root, data, planning)
        if phase in pending["phase_startup_wall_seconds"]:
            estimate = pending["phase_startup_wall_seconds"][phase]
    attempt = dict(status="started", phase=phase, gpu_work_started=False, protocol_sha256=frozen["sha256"])
    try:
        with budget.session(bucket, uuids, estimate, cleanup, protocol_sha256=frozen["sha256"]) as session:
            if planning is not None:
                pending = pending_cost_plan(frozen, root, dict(session._data, active=None), planning)
                if pending["status"] != "fits":
                    raise BudgetStop("remaining complete CFG matrix no longer fits its unchanged stage allowances")
                session.check(pending["pending_stage_estimates_gpu_seconds"].get(bucket, 0.) / len(uuids), 0.)
                atomic_json(root / (phase + "-budget-admission.json"), seal(pending))
            timeout = min(session.check() - cleanup, config["timeout_wall_seconds"][bucket])
            remaining = session.allocation_remaining()
            job = dict(phase=phase, output=str(root), physical_devices=selected, all_physical_devices=devices,
                deadline_monotonic=time.monotonic() + timeout, unit_estimate_seconds=estimate,
                remaining_gpu_seconds=config["budget_gpu_seconds"] - session._data["used_gpu_seconds"],
                phase_remaining_gpu_seconds=remaining, session_id=session._data["active"]["session_id"],
                protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"], budget_stage=bucket)
            jobpath = root / (phase + "-job.json")
            atomic_json(jobpath, job)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(uuids), PYTHONPATH=str(ROOT / "python"),
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                TORCH_HOME=str(ROOT / ".cache/torch"), OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
            process = None
            with (root / (phase + ".log")).open("w") as stream:
                try:
                    session.expect_workers()
                    process = subprocess.Popen([sys.executable, "-m", "ditserve.research.cfg_runner", "_worker", str(jobpath)],
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
    deadline = time.monotonic() + 2
    while True:
        active = read_json(frozen["ledger"]).get("active") or {}
        if active.get("worker_pgid") is not None or time.monotonic() >= deadline:
            break
        time.sleep(.01)
    if (active.get("pid") != os.getppid() or active.get("worker_pgid") != os.getpgrp()
            or active.get("session_id") != job["session_id"] or active.get("protocol_sha256") != frozen["sha256"]
            or active.get("devices") != [d["uuid"] for d in job["physical_devices"]]
            or active.get("stage") != PHASE_BUCKET[job["phase"]]
            or job.get("budget_stage") != PHASE_BUCKET[job["phase"]]
            or job.get("protocol_sha256") != frozen["sha256"] or job.get("source_sha256") != frozen["source_sha256"]
            or type(job.get("deadline_monotonic")) not in (int, float)
            or not math.isfinite(job["deadline_monotonic"]) or job["deadline_monotonic"] <= time.monotonic()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(active.get("devices", []))):
        raise RuntimeError("private CFG worker requires matching parent ledger and physical devices")
    from .cfg_experiment import run
    try:
        result = run(job)
        return 0 if result["status"] in ("completed", "direction_stopped", "quality_stopped", "unverified") else 3
    except BudgetStop:
        return 3


def write_summary(root, name, frozen, value):
    value = {k: v for k, v in value.items() if k != "sha256"}
    value = seal(dict(value, protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"]))
    path = root / (name + ".json")
    if path.exists() and read_json(path) != value:
        raise ValueError("preserve frozen stage decision: " + name)
    atomic_json(path, value)
    return value


def context_identity(frozen, devices):
    return digest(dict(protocol=frozen["sha256"], source=frozen["source_sha256"],
                       shape=frozen["config"]["shapes"]["main"], backends=dict(a100="flash", v100="efficient"),
                       devices=devices, transition_step=16))


def execute_phases(phases, frozen, root, devices):
    results = []
    for phase in phases:
        if phase == "mechanism" and (root / "migration.json").exists():
            migration = read_json(root / "migration.json")
            if migration.get("status") in ("quality_stopped", "direction_stopped", "unverified"):
                value = write_summary(root, "mechanism", frozen, dict(status="unverified",
                    reason="no migration action qualified", gpu_work_started=False, execution_kind="cpu_gate"))
                results.append(dict(status="completed", returncode=0, gpu_work_started=False, decision=value))
                break
        completed = root / (phase + "-attempt.json")
        if completed.exists():
            old = read_json(completed)
            if old.get("status") == "completed" and old.get("returncode") == 0:
                bound_file(root, phase, frozen, ("completed", "direction_stopped", "quality_stopped", "unverified"))
                results.append(old)
                continue
        attempt = supervise(phase, frozen, root, devices)
        results.append(attempt)
        if attempt["status"] != "completed":
            return results
    return results


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
    parser.add_argument("--validate-cpu", action="store_true")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--prototype-run", type=Path,
                        help="bind sealed oracle screening for implementation continuation only")
    continuation.add_argument("--refinement-run", type=Path,
                        help="bind a closed failed implementation for a paired-only fresh retry")
    parser.add_argument("--cpu-embeddings", type=Path,
                        help="CPU input cache directory; required with --refinement-run")
    parser.add_argument("--kind", choices=("correctness", "screen", "migration", "mechanism", "all"), default="all")
    args = parser.parse_args(argv)
    if bool(args.refinement_run) != bool(args.cpu_embeddings):
        parser.error("--refinement-run requires --cpu-embeddings; both are restricted to refinement preflight")
    if args.stage != "preflight" and (args.refinement_run or args.prototype_run or args.cpu_embeddings):
        parser.error("continuation and CPU cache options are only valid at preflight")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = validate_protocol(read_json(args.protocol))
    if args.stage == "report":
        from .cfg_report import report
        value = report(root)
        from ..result_storage import export_report
        export_report(root)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    ledger = read_json(args.ledger)
    if ledger.get("active"):
        print(json.dumps(dict(status="blocked", reason="existing session must settle", gpu_work_started=False)))
        return 2
    observed = environment(config, args.devices.split(","))
    if observed["status"] != "ready":
        if args.stage == "preflight" and args.validate_cpu and not (root / "protocol.json").exists():
            from .dual_protocol import source_files
            from .common import file_digest
            sources = {name: file_digest(path) for name, path in source_files().items()}
            observed["cpu_validation"] = cpu_validate(dict(sha256=None, source_sha256=digest(sources)), root)
            observed["cpu_validation_scope"] = "unfrozen CPU verification only; GPU qualification unavailable"
        atomic_json(root / "preflight-observation.json", observed)
        print(json.dumps(observed, indent=2))
        return 2
    if args.stage == "preflight":
        from .retention import idle_guard
        with idle_guard(root, [args.ledger]):
            manifest = inspect_model_snapshot(hash_files=True)
            embeddings = None
            if args.refinement_run:
                # Validate the closed predecessor before expensive CPU encoding.
                from .cfg_refinement import build_refinement
                build_refinement(args.refinement_run, config=config, ledger=args.ledger)
                from .cfg_embeddings import prepare_cache
                encoded = prepare_cache(config, manifest, args.cpu_embeddings)
                embeddings = dict(root=str(args.cpu_embeddings.resolve()), manifest_sha256=encoded["sha256"],
                                  model_manifest_sha256=manifest["sha256"])
            frozen = freeze_protocol(args.protocol, root, args.ledger, model_manifest=manifest,
                prototype_run=args.prototype_run, refinement_run=args.refinement_run, cpu_embeddings=embeddings)
            value = dict(observed, model_manifest_sha256=manifest["sha256"], protocol_sha256=frozen["sha256"])
            if frozen.get("prototype_continuation"):
                from .cfg_cost_plan import continuation_cost_plan
                planning = continuation_cost_plan(frozen, read_json(args.ledger))
                write_summary(root, "execution-plan", frozen, planning)
                value["execution_plan"] = planning
                if planning["status"] != "fits":
                    value["status"] = "blocked"
            if args.validate_cpu:
                value["cpu_validation"] = cpu_validate(frozen, root)
                if value["cpu_validation"]["status"] != "passed":
                    value["status"] = "blocked"
        atomic_json(root / "preflight.json", value)
        atomic_json(root / "pending-references.json", dict(pending=True, reason="CFG paired stages may reference raw outputs"))
    else:
        frozen = verify_protocol(root / "protocol.json")
        context = context_identity(frozen, observed["devices"])
        from .cfg_matrix import stage_a_gate, select_candidate, holdout_gate
        from .cfg_continuation import screening_decisions
        if args.stage == "probe":
            phases = (["correctness-" + role for role in ("a100", "v100")]
                      if args.kind in ("all", "correctness") else [])
            if frozen.get("prototype_continuation") and args.kind == "screen":
                parser.error("continuation binds the old prototype screen; run correctness, then formal calibration")
            phases += (["screen-" + role for role in ("a100", "v100")]
                       if args.kind in ("all", "screen") and not frozen.get("prototype_continuation") else [])
            if not phases:
                parser.error("probe --kind must be correctness, screen or all")
            attempts = execute_phases(phases, frozen, root, observed["devices"])
            value = dict(status="completed" if all(a["status"] == "completed" for a in attempts) else "unverified", attempts=attempts)
            if value["status"] == "completed" and (args.kind != "correctness" or frozen.get("prototype_continuation")):
                rows = read_records(root, frozen)
                gates = screening_decisions(frozen, rows, context)
                value = write_summary(root, "probe", frozen, dict(status="completed", stage_a=gates,
                    context_sha256=context, shape=config["shapes"]["main"], backends=dict(a100="flash", v100="efficient"),
                    screening_scope="bound prototype implementation decision only" if frozen.get("prototype_continuation")
                                    else "current-source oracle screening"))
        elif args.stage == "calibrate":
            bound_file(root, "probe", frozen)
            rows = read_records(root, frozen)
            screens = screening_decisions(frozen, rows, context)
            if screens["paired_008"]["eligible"] and not frozen.get("prototype_continuation"):
                # Explicit implementation boundary, not permission to spend on an unqualified prototype.
                value = dict(status="implementation_required", reason="paired screen passed; implement and validate split-prefix before timing", gpu_work_started=False)
            elif not any(g["eligible"] for g in screens.values()):
                value = write_summary(root, "calibrate", frozen, dict(status="direction_stopped", eligible=False,
                    reason="both candidates failed screening", stage_a=screens))
            else:
                attempts = execute_phases(["calibrate-" + role for role in ("a100", "v100")], frozen, root, observed["devices"])
                if any(a["status"] != "completed" for a in attempts):
                    value = dict(status="unverified", attempts=attempts)
                else:
                    rows = read_records(root, frozen)
                    selection = select_candidate(rows, config, context, frozen=frozen)
                    value = write_summary(root, "selection", frozen, selection)
                    if selection["selected"]:
                        attempts = execute_phases(["holdout-" + role for role in ("a100", "v100")], frozen, root, observed["devices"])
                        rows = read_records(root, frozen)
                        held = holdout_gate(rows, config, selection["selected"], context)
                        value = write_summary(root, "calibrate", frozen, dict(status="completed" if held["eligible"] else "direction_stopped",
                            eligible=held["eligible"], selection=selection, holdout=held))
                    else:
                        value = write_summary(root, "calibrate", frozen, dict(status=selection["status"], eligible=False, selection=selection))
        else:
            gate_path = root / "calibrate.json"
            gate = read_json(gate_path) if gate_path.exists() else {}
            if not gate.get("eligible"):
                value = dict(status="direction_stopped", reason="no selected quality-and-cost-qualified policy", gpu_work_started=False)
                for phase in ("migration", "mechanism"):
                    write_summary(root, phase, frozen, value)
            else:
                phases = ["migration", "mechanism"] if args.kind == "all" else [args.kind]
                if any(p not in ("migration", "mechanism") for p in phases):
                    parser.error("mechanism --kind must be migration, mechanism or all")
                attempts = execute_phases(phases, frozen, root, observed["devices"])
                value = dict(status="completed" if all(a["status"] == "completed" for a in attempts) else "unverified", attempts=attempts)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value.get("status") in ("ready", "completed", "passed", "direction_stopped") else 2


if __name__ == "__main__":
    raise SystemExit(main())
