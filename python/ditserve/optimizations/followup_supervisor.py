"""A100-only ownership and accounting for the independent quick follow-up.

Resource waits are CPU-only. A charged lease begins before the child can load
models and ends only after its process group and GPU allocations are confirmed
gone. Failed and cancelled attempts retain their partial output and full cost.
The caller owns the CPU waiter, final Excel/report archive and exact cleanup.
"""

import copy
import math
import os
from pathlib import Path
import subprocess
import time

from ..directions.runner import stop_owned_group
from ..research.common import atomic_json, digest, file_digest, read_json
from ..research.isolation import gpu_occupancy
from .next_budget import NextAllocation
from .next_protocol import authorization
from .protocol import ROOT, freeze
from .runner import inspect as inspect_gpu


STAGES = ("finish_copy", "sana_combo", "granularity")
SCHEMA = "ditserve.optimizations.followup_admission.v1"


def _ordinary_path(value):
    path = Path(value).absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("follow-up control paths must not traverse symbolic links")
    return path.resolve()


def _paths(run_root, ledger, parents, authorization_path):
    run_root, ledger = _ordinary_path(run_root), _ordinary_path(ledger)
    parents = tuple(_ordinary_path(parent) for parent in parents)
    auth = _ordinary_path(authorization_path)
    if not parents or len(set(parents)) != len(parents):
        raise ValueError("distinct immutable parent ledgers are required")
    if ledger.name != "gpu-budget.json" or ledger.parent not in run_root.parents:
        raise ValueError("independent gpu-budget.json must be above the run directory")
    for parent in parents:
        for path in (run_root, ledger):
            if path == parent.parent or parent.parent in path.parents:
                raise ValueError("new follow-up must be outside historical result trees")
    if auth in parents or auth == ledger or run_root == auth:
        raise ValueError("authorization must not alias result or budget evidence")
    return run_root, ledger, parents, auth


def _binding(path):
    return dict(path=str(path), sha256=file_digest(path))


def _validate_sources(source, run_root):
    root = _ordinary_path(source["execution_root"])
    if run_root not in root.parents:
        raise ValueError("executable snapshot must belong to this run")
    files = source["source_files"]
    if not files or digest(files) != source["source_sha256"]:
        raise ValueError("executable source manifest identity differs")
    for name, expected in files.items():
        path = _ordinary_path(root / name)
        if root not in path.parents or file_digest(path) != expected:
            raise ValueError("executable source changed: " + name)


def _check_bindings(bindings):
    for binding in bindings:
        if file_digest(binding["path"]) != binding["sha256"]:
            raise ValueError("bound evidence changed: " + binding["path"])


def _observation(summary, value):
    summary["observations"] += 1
    if summary["first"] is None:
        summary["first"] = value
    summary["last"] = value
    if value.get("status") != "exclusive_observed" and summary["first_nonexclusive"] is None:
        summary["first_nonexclusive"] = value


def execute(config, phase, run_root, ledger, parents, authorization_path):
    """Execute one frozen phase once, returning its settled supervisor record.

    ``followup.wait_idle_seconds`` defaults to an unlimited CPU-only wait; a
    finite nonnegative value permits a bounded admission check. No retry is
    implicit: an existing phase/attempt000 is always rejected. Later stages
    require earlier terminal conclusions, which may include measured failures.
    """
    if phase not in STAGES:
        raise ValueError("unknown follow-up phase")
    if not isinstance(config, dict):
        raise TypeError("follow-up config must be a frozen ordinary dictionary")
    config = copy.deepcopy(config)
    if config.get("device", "cuda:0") != "cuda:0":
        raise ValueError("follow-up executes only the admitted A100 as cuda:0")
    run_root, ledger, parents, authorization_path = _paths(
        run_root, ledger, parents, authorization_path)
    _, auth_binding = authorization(authorization_path)
    study_path = _ordinary_path(run_root / "study.json")
    study_binding = _binding(study_path)
    if not isinstance(read_json(study_path), dict) or file_digest(study_path) != study_binding["sha256"]:
        raise ValueError("frozen follow-up study changed while reading")
    for previous in STAGES[:STAGES.index(phase)]:
        terminal = run_root / previous / "terminal.json"
        if not terminal.is_file():
            raise ValueError("earlier follow-up stages require terminal conclusions")
        earlier = read_json(terminal)
        if earlier.get("status") in (None, "running", "waiting_resource") or earlier.get("cleanup_confirmed") is not True:
            raise ValueError("earlier follow-up stage has not confirmed cleanup")
    phase_root = run_root / phase
    destination = phase_root / "attempt000"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("follow-up attempt000 already exists; refusing to overwrite")
    phase_root.mkdir(parents=True, exist_ok=True)
    wait_seconds = config.get("followup", {}).get("wait_idle_seconds")
    if wait_seconds is not None and (type(wait_seconds) not in (int, float)
            or not math.isfinite(wait_seconds) or wait_seconds < 0):
        raise ValueError("wait_idle_seconds must be nonnegative finite or null")
    deadline = None if wait_seconds is None else time.monotonic() + wait_seconds
    while True:
        preflight = inspect_gpu(config)
        preflight.pop("additional_budget_gpu_hours", None)
        atomic_json(phase_root / "preflight.json", preflight)
        if preflight.get("status") == "ready":
            selected = [row for row in preflight.get("selected", [])
                        if row.get("uuid") == config["gpu_uuid"]]
            if len(selected) != 1 or "A100" not in selected[0].get("name", ""):
                raise ValueError("ready preflight did not independently identify the selected A100")
            break
        if deadline is not None and time.monotonic() >= deadline:
            return dict(status="blocked", phase=phase, output=str(destination), returncode=None,
                charge=None, decision="未验证", gpu_work_started=False,
                reason="A100 or required models not ready; no GPU lease started",
                preflight=str(phase_root / "preflight.json"))
        time.sleep(5.0 if deadline is None else min(5.0, max(0.0, deadline - time.monotonic())))

    devices = [config["gpu_uuid"]]
    failure = returncode = process = charge = None
    result = {}
    isolation = dict(observations=0, first=None, last=None, first_nonexclusive=None,
                     scope="sampled A100 process-group ownership; not a reservation; intermediate identical samples aggregated")
    with NextAllocation(ledger, parents=parents, authorization_binding=auth_binding,
                        gpu_uuid=config["gpu_uuid"], stages=STAGES) as budget:
        destination.mkdir()
        source = freeze(config, run_root, phase, "attempt000")
        _validate_sources(source, run_root)
        atomic_json(destination / "config.json", config)
        history = [_binding(path) for path in parents]
        admission = dict(schema_version=SCHEMA, phase=phase, output=str(destination),
            run_root=str(run_root), ledger=str(ledger), supervisor_pid=os.getpid(),
            config_sha256=digest(config), source_sha256=source["source_sha256"],
            execution_root=source["execution_root"], gpu_uuids=devices,
            source_manifest=_binding(Path(source["execution_root"]) / "manifest.json"),
            study_binding=study_binding,
            authorization_source=auth_binding, parent_ledgers=history,
            budget_unlimited=True, gpu_count=1,
            accounting_scope="complete admitted A100 lifetime: loading, compile, warmup, measurements, failures and cleanup")
        atomic_json(destination / "admission.json", admission)
        admission_binding = _binding(destination / "admission.json")
        bindings = history + [auth_binding, study_binding, admission_binding, admission["source_manifest"],
                              _binding(destination / "config.json")]
        identity = dict(phase=phase, output=str(destination), run_root=str(run_root),
            config_sha256=digest(config), source_sha256=source["source_sha256"],
            gpu_uuids=devices, budget_unlimited=True, auxiliary_encoder=False,
            authorization_source=auth_binding, admission=admission_binding,
            parent_ledgers=history, study_binding=study_binding, retry_from=None)
        command = [config["python"], "-u", "-m", "ditserve.optimizations.followup_study", "worker",
            "--config", str(destination / "config.json"), "--phase", phase,
            "--run-root", str(run_root), "--output", str(destination), "--ledger", str(ledger),
            "--supervisor-pid", str(os.getpid())]
        env = dict(os.environ, PYTHONPATH=str(Path(source["execution_root"]) / "python"),
            CUDA_VISIBLE_DEVICES=config["gpu_uuid"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", TORCH_HOME=config["torch_home"],
            TOKENIZERS_PARALLELISM="false", TORCHINDUCTOR_CACHE_DIR=str(destination / "compiler-cache"),
            TRITON_CACHE_DIR=str(destination / "triton-cache"))
        _check_bindings(bindings)
        budget.begin(phase, gpu_count=1, identity=identity, minimum_seconds=120.0)
        try:
            initial = gpu_occupancy(devices)
            _observation(isolation, initial)
            if initial.get("status") != "exclusive_observed":
                raise RuntimeError("A100 isolation lost before child launch: " + str(initial.get("status")))
            with (destination / "worker.log").open("x") as log:
                process = subprocess.Popen(command, cwd=str(ROOT), env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                budget.child_started(process.pid)
                while True:
                    observed = gpu_occupancy(devices, allowed_pgid=process.pid)
                    _observation(isolation, observed)
                    if observed.get("status") != "exclusive_observed":
                        raise RuntimeError("A100 isolation lost: " + str(observed.get("status")))
                    try:
                        returncode = process.wait(timeout=2.0)
                        break
                    except subprocess.TimeoutExpired:
                        pass
        except BaseException as error:
            failure = dict(error_type=type(error).__name__, reason=str(error))
        finally:
            gone = stop_owned_group(process)
            final = gpu_occupancy(devices, allowed_pgid=process.pid if process is not None else None)
            _observation(isolation, final)
            if (not gone or final.get("owned") != [] or not isinstance(final.get("foreign"), list)
                    or final.get("status") in (None, "unknown")):
                atomic_json(destination / "supervisor.json", dict(status="incomplete", phase=phase,
                    output=str(destination), decision="未验证", online_qualified=False,
                    reason="cleanup unconfirmed; lease remains active and charged", charge=None,
                    returncode=returncode, cleanup_confirmed=False, final_occupancy=final,
                    isolation=isolation, admission=admission_binding, supervisor_failure=failure))
                raise RuntimeError("cleanup unconfirmed; lease remains active and charged")
            if final.get("status") != "exclusive_observed":
                failure = dict(error_type="GPUIsolationError", reason="foreign process observed at cleanup")
            try:
                _validate_sources(source, run_root)
                _check_bindings(bindings)
            except (OSError, ValueError, KeyError, TypeError) as error:
                failure = dict(error_type="SourceIdentityError", reason=type(error).__name__ + ": " + str(error))
            try:
                result = read_json(destination / "worker.json")
                if not isinstance(result, dict) or not isinstance(result.get("status"), str):
                    raise ValueError("worker result lacks terminal status")
                if result["status"] in ("running", "failed", "incomplete") and failure is None:
                    failure = dict(error_type="WorkerIncomplete", reason=result.get("reason", "worker did not complete"))
            except (OSError, ValueError, KeyError, TypeError) as error:
                result = {}
                if failure is None:
                    failure = dict(error_type="WorkerEvidenceError", reason=type(error).__name__ + ": " + str(error))
            charge = budget.finish("failed" if failure is not None or returncode != 0 else "completed")
            if charge.get("status") != "completed" and failure is None and returncode == 0:
                failure = dict(error_type="BudgetSettlementError",
                    reason=charge.get("parent_validation_error", charge.get("authorization_validation_error", "settlement failed")))
    if failure is not None or returncode != 0:
        result.update(status="incomplete", decision="未验证", online_qualified=False,
            local_qualified=False, reason=(failure or {}).get("reason", result.get("reason", "worker failed")),
            supervisor_failure=failure)
    result.update(phase=phase, output=str(destination), returncode=returncode, charge=charge,
        gpu_work_started=process is not None, cleanup_confirmed=True, final_occupancy=final,
        isolation=isolation, config_sha256=digest(config), source_sha256=source["source_sha256"],
        authorization_source=auth_binding, admission=admission_binding)
    atomic_json(destination / "supervisor.json", result)
    atomic_json(phase_root / "terminal.json", result)
    return result
