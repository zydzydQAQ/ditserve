"""One isolated, fully charged replay of the interrupted Sana combination.

Only the private artifact directory changes in the scientific configuration.
Workers use the complete old executable snapshot; old results, partial samples,
budgets and globals are never edited or imported as new measurements.
"""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import sys
import traceback
from types import FunctionType

from ..research.common import atomic_json, digest, file_digest, read_json
from .protocol import ROOT
from . import followup_supervisor as supervisor
from .followup_queue import cleanup, process_identity
from .followup_study import quick_traces
from .next_budget import NextAllocation
from .next_protocol import authorization

PARENT = ROOT / ".cache/results-work/optimization-followup/run01"
WORK = ROOT / ".cache/results-work/optimization-followup-combo-retry/run01"
OUTPUT = ROOT / "results/optimization-followup-combo-retry/run01"
LEDGER = WORK.parent / "gpu-budget.json"
AUTH = WORK.parent / "authorization.json"
PHASE = "sana_combo"
SCHEMA = "ditserve.followup_combo_retry.v1"


def ordinary(value):
    path = Path(value).absolute()
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("retry evidence cannot traverse symbolic links")
    return path


def binding(value):
    path = ordinary(value)
    return dict(path=str(path), sha256=file_digest(path), bytes=path.stat().st_size)


def check_binding(item):
    path = ordinary(item["path"])
    if file_digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
        raise ValueError("bound retry evidence changed: " + str(path))


def check_snapshot(source):
    root = ordinary(source["execution_root"])
    if not source.get("source_files") or digest(source["source_files"]) != source["source_sha256"]:
        raise ValueError("source manifest digest mismatch")
    for name, sha in source["source_files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("source names must be contained relative paths")
        path = ordinary(root / name)
        if root not in path.parents or file_digest(path) != sha:
            raise ValueError("original executable changed: " + name)


def storage_config(original, work_root):
    result = copy.deepcopy(original)
    result["artifact_root"] = str(ordinary(work_root) / "raw-arrays")
    return result


def validate_storage_config(config, original, work_root):
    if config != storage_config(original, work_root):
        raise ValueError("only the private artifact directory may change")
    return config


def check_loaded_modules(source, implementations):
    allowed = {ordinary(item["path"]): item for item in implementations}
    for item in allowed.values():
        check_binding(item)
    for name, module in list(sys.modules.items()):
        if not name.startswith("ditserve."):
            continue
        value = getattr(module, "__file__", None)
        if not value or not value.endswith(".py"):
            continue
        path = ordinary(value)
        if path in allowed:
            continue
        try:
            relative = str(path.relative_to(ROOT))
        except ValueError as error:
            raise ValueError("unexpected external supervisor import: " + name) from error
        if source["source_files"].get(relative) != file_digest(path):
            raise ValueError("loaded supervisor dependency differs: " + relative)


class RetryFreezer:
    """Copy old executable bytes with an explicit storage-only config change."""

    def __init__(self, manifest, original_config, implementations, controls=(), on_freeze=None):
        self.manifest = manifest
        self.original_config = original_config
        self.implementations = list(implementations)
        self.controls = list(controls)
        self.on_freeze = on_freeze

    def __call__(self, config, run_root, phase, attempt):
        for item in [self.manifest, self.original_config] + self.implementations + self.controls:
            check_binding(item)
        source = read_json(self.manifest["path"])
        original = read_json(self.original_config["path"])
        check_snapshot(source)
        if source["config_sha256"] != digest(original):
            raise ValueError("original source/config identity differs")
        root = ordinary(run_root)
        validate_storage_config(config, original, root)
        if phase != PHASE or attempt != "attempt000":
            raise ValueError("retry only admits Sana combo attempt000 in its new run")
        protocol = read_json(ordinary(root / "protocol.json"))
        if protocol.get("config") != config or protocol.get("sha256") != digest(config):
            raise ValueError("new frozen protocol differs")
        target = ordinary(root / "sources" / (phase + "-" + attempt))
        target.mkdir(parents=True, exist_ok=False)
        for name, sha in source["source_files"].items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(source["execution_root"]) / name, path)
            if file_digest(path) != sha:
                raise ValueError("derived executable bytes differ")
        manifest = dict(source, execution_root=str(target), phase=phase, attempt=attempt,
            config_sha256=digest(config), derived_from_manifest=self.manifest,
            original_config=self.original_config, retry_implementations=self.implementations,
            storage_only_change=dict(field="artifact_root", before=original["artifact_root"],
                                     after=config["artifact_root"]), worker_source_unchanged=True)
        atomic_json(target / "manifest.json", manifest)
        if self.on_freeze is not None:
            self.on_freeze()
        return manifest


def isolated_executor(factory, implementation=None):
    """Reuse supervisor code with a private namespace, not shared mutation."""
    original = implementation or supervisor.execute
    namespace = dict(original.__globals__, STAGES=(PHASE,), freeze=factory)
    isolated = FunctionType(original.__code__, namespace, original.__name__,
                            original.__defaults__, original.__closure__)
    isolated.__kwdefaults__ = copy.copy(original.__kwdefaults__)
    return isolated


def _paths(work_root, output, ledger, auth, parent):
    paths = tuple(ordinary(p) for p in (work_root, output, ledger, auth, parent))
    work_root, output, ledger, auth, parent = paths
    if ledger != work_root.parent / "gpu-budget.json" or auth != work_root.parent / "authorization.json":
        raise ValueError("retry ledger and authorization must be beside its run directory")
    if work_root == output or work_root in output.parents or output in work_root.parents:
        raise ValueError("work and delivery directories must be disjoint")
    for path in (work_root, output, ledger, auth):
        if path == parent.parent or parent.parent in path.parents or path in parent.parents:
            raise ValueError("retry must be outside original experiment tree")
    return paths


def _original(parent):
    budget_path = parent.parent / "gpu-budget.json"
    budget = read_json(ordinary(budget_path))
    record = read_json(ordinary(parent / "resume01.json"))
    if budget.get("active", "missing") is not None or record.get("status") != "completed":
        raise ValueError("original queue and budget must be settled")
    current = process_identity(record["pid"])
    if current is not None and current == record.get("process_identity"):
        raise ValueError("original continuation process is still alive")
    terminal_path = parent / PHASE / "terminal.json"
    terminal = read_json(ordinary(terminal_path))
    supervisor_path = parent / PHASE / "attempt000/supervisor.json"
    if read_json(ordinary(supervisor_path)) != terminal:
        raise ValueError("original terminal and supervisor disagree")
    charge = terminal.get("charge", {})
    matches = [s for s in budget.get("sessions", []) if s.get("session_id") == charge.get("session_id")]
    final = terminal.get("final_occupancy", {})
    if (terminal.get("phase") != PHASE or terminal.get("status") != "incomplete"
            or terminal.get("reason") != "A100 isolation lost: unknown"
            or terminal.get("cleanup_confirmed") is not True or matches != [charge]
            or charge.get("status") != "failed" or final.get("status") != "exclusive_observed"
            or final.get("owned") != [] or final.get("foreign") != []):
        raise ValueError("original failure lacks exact isolated cleanup and billing")
    return terminal, budget_path


def prepare(work_root=WORK, output=OUTPUT, ledger=LEDGER, auth=AUTH, parent=PARENT):
    from . import followup_combo_retry_report as reporting
    work_root, output, ledger, auth, parent = _paths(work_root, output, ledger, auth, parent)
    if any(path.exists() for path in (work_root, output, ledger, auth)):
        raise FileExistsError("retry destinations already exist; no automatic overwrite")
    terminal, parent_ledger = _original(parent)
    source_binding = binding(parent / "sources/sana_combo-attempt000/manifest.json")
    source = read_json(source_binding["path"])
    check_snapshot(source)
    original_config = binding(parent / "config.json")
    original = read_json(original_config["path"])
    if (digest(original) != source["config_sha256"] or terminal["config_sha256"] != digest(original)
            or terminal.get("source_sha256") != source["source_sha256"]
            or terminal["charge"].get("identity", {}).get("source_sha256") != source["source_sha256"]):
        raise ValueError("original scientific configuration differs")
    admission_binding = binding(parent / PHASE / "attempt000/admission.json")
    admission = read_json(admission_binding["path"])
    compact = lambda item: {key: item[key] for key in ("path", "sha256")}
    if (terminal.get("admission") != compact(admission_binding)
            or terminal["charge"]["identity"].get("admission") != compact(admission_binding)
            or admission.get("source_manifest") != compact(source_binding)
            or admission.get("config_sha256") != digest(original)):
        raise ValueError("original admission does not bind the reused executable")
    config = storage_config(original, work_root)
    implementations = [binding(__file__), binding(reporting.__file__)]
    check_loaded_modules(source, implementations)
    previous_study = read_json(parent / "study.json")
    if digest(quick_traces(config)) != previous_study["trace_sha256"]:
        raise ValueError("retry arrival/deadline trace differs")
    parents = [binding(item["path"]) for item in previous_study["parent_ledgers"]]
    for original_parent, current_parent in zip(previous_study["parent_ledgers"], parents):
        if original_parent["sha256"] != current_parent["sha256"]:
            raise ValueError("previous parent ledger changed")
    parents.append(binding(parent_ledger))
    for item in parents:
        if read_json(item["path"]).get("active", "missing") is not None:
            raise ValueError("parent budget remains active")
    for item in previous_study["baseline_bindings"]:
        check_binding(item)
    protected = [binding(parent / name) for name in ("resume01.json", "queue.json", "config.json",
        "protocol.json", "study.json", "sana_combo/terminal.json", "sana_combo/attempt000/supervisor.json",
        "sana_combo/attempt000/phase-result.json")]
    protected += parents + [source_binding, admission_binding, binding(parent.parent / "authorization.json")]
    old_auth = read_json(parent.parent / "authorization.json")
    new_auth = dict(old_auth, latest_user_instruction="补齐一下",
        scope_detail="one fresh Sana VAE plus CUDA Graph local/recovery/eight-replay screen; no sample pooling",
        parent_authorization=binding(parent.parent / "authorization.json"))
    work_root.mkdir(parents=True, exist_ok=False)
    atomic_json(auth, new_auth)
    _, auth_binding = authorization(auth)
    atomic_json(work_root / "config.json", config)
    atomic_json(work_root / "protocol.json", dict(config=config, sha256=digest(config),
        schema_version="ditserve.optimizations.protocol.v1"))
    storage_change = dict(field="artifact_root", before=original["artifact_root"], after=config["artifact_root"],
        all_other_fields_equal=True, purpose="keep failure arrays outside the immutable original run")
    study = dict(previous_study, ledger=str(ledger), parent_ledgers=parents, phases=[PHASE],
        config_sha256=digest(config), source_study=binding(parent / "study.json"),
        retry_of=binding(parent / PHASE / "terminal.json"), original_config=original_config,
        storage_only_change=storage_change, original_samples_reused=0,
        scope="fresh local quality, full recovery, and eight online replays; independent samples and budget")
    atomic_json(work_root / "study.json", study)
    controls = protected + [binding(p) for p in (auth, work_root / "config.json",
                                                work_root / "protocol.json", work_root / "study.json")]
    # Creating a zero-use ledger holds no GPU. begin() remains in the supervisor.
    with NextAllocation(ledger, parents=tuple(Path(x["path"]) for x in parents),
            authorization_binding=auth_binding, gpu_uuid=config["gpu_uuid"], stages=(PHASE,)):
        pass
    record = dict(schema_version=SCHEMA, status="prepared", current_phase=None, phases=[PHASE],
        work_root=str(work_root), output=str(output), ledger=str(ledger), authorization=binding(auth),
        config=binding(work_root / "config.json"), original_config=original_config,
        original_failure=binding(parent / PHASE / "terminal.json"),
        original_failure_reason=terminal["reason"], source_manifest=source_binding,
        controller=implementations[0], implementations=implementations, immutable_controls=controls,
        parent_ledgers=parents, completed_phases=[], storage_only_change=storage_change,
        gpu_work_started=False, automatic_retry=False, original_samples_reused=0, defaults_unchanged=True,
        isolation_policy="unchanged: any unknown or foreign GPU ownership terminates the admitted worker",
        scope="one complete fresh Sana combo stage; original failed attempt retained independently")
    atomic_json(work_root / "queue.json", record)
    return record


def run(work_root=WORK, output=OUTPUT, ledger=LEDGER, auth=AUTH, parent=PARENT):
    from . import followup_combo_retry_report as reporting
    work_root, output, ledger, auth, parent = _paths(work_root, output, ledger, auth, parent)
    queue_path = work_root / "queue.json"
    with ordinary(work_root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = read_json(queue_path)
        if record.get("schema_version") != SCHEMA or record.get("status") != "prepared":
            raise ValueError("retry already started or unknown receipt; preserve evidence")
        record.update(status="waiting", current_phase=PHASE, pid=os.getpid(),
                      process_identity=process_identity(os.getpid()))
        atomic_json(queue_path, record)
        export = lambda: reporting.write(work_root, output, ledger=ledger)
        try:
            for item in record["immutable_controls"] + record["implementations"]:
                check_binding(item)
            source = read_json(record["source_manifest"]["path"])
            check_snapshot(source)
            check_loaded_modules(source, record["implementations"])
            _original(parent)
            budget = read_json(ledger)
            if budget.get("active") is not None or budget.get("sessions") != [] or budget.get("used_gpu_seconds") != 0:
                raise ValueError("fresh retry ledger already contains work")
            config = read_json(work_root / "config.json")
            def on_freeze():
                check_loaded_modules(source, record["implementations"])
                record.update(status="running", gpu_activity_source="ledger.active; lease begins immediately before worker launch")
                atomic_json(queue_path, record)
            factory = RetryFreezer(record["source_manifest"], record["original_config"],
                record["implementations"], record["immutable_controls"], on_freeze)
            initial_report = export()
            if initial_report.get("errors"):
                raise ValueError("initial retry archive reports verification errors")
            result = isolated_executor(factory)(config, PHASE, work_root, ledger,
                tuple(Path(x["path"]) for x in record["parent_ledgers"]), auth)
            for item in record["immutable_controls"] + record["implementations"]:
                check_binding(item)
            record["gpu_work_started"] = bool(result.get("gpu_work_started"))
            record["completed_phases"] = [{key: result.get(key) for key in
                ("phase", "status", "decision", "output", "cleanup_confirmed")}]
            if result.get("cleanup_confirmed") is not True:
                raise RuntimeError("GPU cleanup unconfirmed; preserve unsettled state")
            record.update(status="finalizing", current_phase=None)
            atomic_json(queue_path, record)
            cleanup(work_root, ledger, export=export)
            record.update(status="completed" if result.get("status") in ("completed", "stopped") else "incomplete",
                          decision=result.get("decision"), reason=result.get("reason"))
            atomic_json(queue_path, record)
            final_report = export()
            if final_report.get("errors"):
                raise ValueError("final retry report contains verification errors")
            return record
        except BaseException as error:
            record.update(status="failed", reason=type(error).__name__ + ": " + str(error),
                          traceback=traceback.format_exc())
            atomic_json(queue_path, record)
            try:
                export()
            except BaseException as report_error:
                record["report_error"] = str(report_error)
                atomic_json(queue_path, record)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "report"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare()
    elif args.command == "run":
        result = run()
    else:
        from .followup_combo_retry_report import write
        result = write(WORK, OUTPUT, ledger=LEDGER)
    print({k: result.get(k) for k in ("status", "current_phase", "decision", "errors")})


if __name__ == "__main__":
    main()
