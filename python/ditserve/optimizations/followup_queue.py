"""Durable CPU waiter, serial follow-up execution, and verified light archive."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import time
import traceback

from ..research.common import atomic_json, file_digest, digest, read_json
from .protocol import ROOT
from .followup_study import WORK, OUTPUT, PHASES, new_config, validate_config, quick_traces

OLD = ROOT / "results/optimization-next/run01"
RETRY = ROOT / "results/optimization-next-numerics-retry/run02"
LEDGER = WORK.parent / "gpu-budget.json"
AUTH = WORK.parent / "authorization.json"
PARENT_LEDGERS = (OLD.parent / "gpu-budget.json", RETRY.parent / "gpu-budget.json")


def process_identity(pid):
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return None
    values = raw[raw.rfind(")")+2:].split()
    if values[0] == "Z":
        return None
    return dict(pid=pid, start_ticks=int(values[19]),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def parent_active(expected):
    actual = process_identity(expected["pid"])
    if actual is not None and actual != expected:
        raise RuntimeError("parent PID/boot identity changed; refuse inferred completion")
    return actual is not None


def _binding(path):
    return dict(path=str(Path(path).resolve()), sha256=file_digest(path), bytes=Path(path).stat().st_size)


def prepare(parents):
    if (WORK / "queue.json").exists():
        raise FileExistsError("follow-up queue already exists; do not overwrite or duplicate")
    WORK.mkdir(parents=True, exist_ok=True)
    config = new_config()
    atomic_json(WORK / "config.json", config)
    if AUTH.exists():
        raise FileExistsError("new follow-up authorization already exists")
    atomic_json(AUTH, dict(scope="serving_next_validation", approved_by_user=True,
        budget_unlimited=True, authorized_gpu_seconds=None,
        user_instruction="取消8小时GPU小时硬上限，GPU无上限，开始在线对照；就按照这个继续；做完0.7的用1个seed快速验证；继续优化",
        latest_user_instruction="继续优化", scope_detail="independent finish-copy, Sana combination, fixed K1/K4 single-seed screen"))
    paths = sorted((ROOT / "python/ditserve").rglob("*.py"))
    paths += sorted((ROOT / "tests/python").glob("test_optimizations_followup*.py"))
    source = {str(p.relative_to(ROOT)):file_digest(p) for p in paths}
    record = dict(schema_version=1, status="prepared", gpu_work_started=False, phases=list(PHASES),
        parents=parents, work_root=str(WORK), output=str(OUTPUT), ledger=str(LEDGER),
        authorization=_binding(AUTH), config=_binding(WORK / "config.json"), sources=source,
        source_sha256=digest(source), completed_phases=[],
        scope="CPU waiting until existing chain, finalizer and diagnostic retry exit; no foreign process signals")
    atomic_json(WORK / "queue.json", record)
    return record


def _sources(record):
    if digest(record["sources"]) != record["source_sha256"]:
        raise ValueError("queue source manifest changed")
    for name, sha in record["sources"].items():
        if file_digest(ROOT / name) != sha:
            raise ValueError("queued executable changed: " + name)
    for binding in (record["authorization"], record["config"]):
        if file_digest(binding["path"]) != binding["sha256"]:
            raise ValueError("queued config or authorization changed")


def _study(config):
    from .next_study import inspect_study
    from .protocol import qualified
    prepared = inspect_study(config, observe=False)
    baseline = OLD / "sana_vae/terminal.json"
    value = read_json(baseline)
    if not qualified(value) or value.get("candidate") != "vae_compile":
        raise ValueError("Sana VAE baseline lacks complete earlier local/online qualification")
    parents = [p for p in PARENT_LEDGERS if p.exists()]
    if not parents or parents[0] != PARENT_LEDGERS[0]:
        raise ValueError("main parent ledger missing")
    for path in parents:
        if read_json(path).get("active", "missing") is not None:
            raise ValueError("parent GPU lease has not settled")
    study = dict(schema_version=1, ledger=str(LEDGER), phases=list(PHASES),
        config_sha256=digest(config), cache_local_binding=prepared["cache_local_binding"],
        baseline_bindings=[_binding(baseline)], trace_sha256=digest(quick_traces(config)),
        parent_ledgers=[_binding(p) for p in parents],
        missing_parent_ledgers=[str(p) for p in PARENT_LEDGERS if not p.exists()],
        default_promotion=False, image_baseline="vae_compile",
        fixed_wan="paired_008 eager; K4 baseline", scope="eight replay one-seed screens, independent phases")
    atomic_json(WORK / "study.json", study)
    return tuple(parents)


def _ordinary(path, root):
    path, root = Path(path).absolute(), Path(root).resolve()
    if path.resolve() != path or path.is_symlink() or root not in path.parents:
        raise ValueError("cleanup path outside exact ordinary work tree")
    return path


def cleanup(work_root, ledger, *, export):
    """Only close successful new-phase raw dependencies; keep failures intact."""
    root = Path(work_root).resolve()
    if read_json(ledger).get("active", "missing") is not None:
        raise ValueError("cannot archive active GPU lease")
    receipt = root / "cleanup.json"
    if receipt.exists():
        raise FileExistsError("cleanup already attempted; preserve its exact evidence")
    plan = dict(schema_version=1, status="planned", files=[], rewrites=[],
        scope="successful phases only; retain failed/partial evidence for diagnosis",
        retained_dependencies=["configuration, parent and GPU ledgers, exact executable sources and admission bindings remain for provenance verification",
            "phase request metrics and quality scalars stay available; raw event sequences cannot be reanalysed after cleanup"],
        raw_reassessment_available=False)
    for phase in PHASES:
        destination = root / phase / "attempt000"
        result_path, supervisor_path = destination / "phase-result.json", destination / "supervisor.json"
        if not result_path.exists() or not supervisor_path.exists():
            continue
        result, supervisor = read_json(result_path), read_json(supervisor_path)
        if (result.get("status") != "completed" or result.get("online", {}).get("status") != "completed"
            or supervisor.get("returncode") != 0 or supervisor.get("cleanup_confirmed") is not True
            or supervisor.get("supervisor_failure") or supervisor.get("charge", {}).get("status") != "completed"):
            continue
        if len(result["online"].get("replays", [])) != 8:
            raise ValueError("complete phase has wrong replay denominator")
        events = 0
        for row in result["online"]["replays"]:
            if row.get("event_summary", {}).get("total_events") != len(row.get("events", [])):
                raise ValueError("event aggregate missing before cleanup")
            from .followup_study import event_summary
            rebuilt = event_summary(copy.deepcopy(row))
            if rebuilt["event_summary"] != row["event_summary"] or any(
                before.get("wait_attribution") != after.get("wait_attribution")
                for before, after in zip(row.get("requests", []),rebuilt.get("requests", []))):
                raise ValueError("event attribution differs from original raw intervals")
            events += len(row.get("events", []))
        plan["rewrites"].append(dict(path=str(result_path), before_sha256=file_digest(result_path),
            before_bytes=result_path.stat().st_size, events_removed=events, status="planned",
            retained="all request records, local checks, per-kind count/sum/mean/p95/max and per-request interval overlap"))
        candidates = [destination / "worker.log"]
        for name in ("online-partials", "compiler-cache", "triton-cache"):
            directory = destination / name
            if directory.exists():
                candidates.extend(path for path in directory.rglob("*") if path.is_file() or path.is_symlink())
        for path in sorted(set(candidates)):
            if path.exists():
                path = _ordinary(path, root)
                plan["files"].append(dict(_binding(path), status="planned", dependency="completed metrics/attribution and settled worker; no queued consumer"))
    atomic_json(receipt, plan)
    # A verified export of all scalars and the exact proposed deletion manifest
    # must exist before the first mutation. The callback raises on any mismatch.
    exported = export()
    if isinstance(exported, dict) and exported.get("errors"):
        raise ValueError("archive reports unresolved verification errors; raw inputs retained")
    for item in plan["rewrites"]:
        path = _ordinary(item["path"], root)
        if file_digest(path) != item["before_sha256"]:
            raise ValueError("result changed during archive")
        result = read_json(path)
        for row in result["online"]["replays"]:
            row.pop("events", None)
            row["raw_events_available"] = False
        result["retention"] = dict(raw_events="removed after verified scalar/attribution export",
            raw_reassessment_available=False, same_original_metrics=True)
        atomic_json(path, result)
        item.update(status="compacted", after_sha256=file_digest(path), after_bytes=path.stat().st_size)
        atomic_json(receipt, plan)
    for item in plan["files"]:
        path = _ordinary(item["path"], root)
        if path.stat().st_size != item["bytes"] or file_digest(path) != item["sha256"]:
            raise ValueError("planned raw input changed before deletion")
        if read_json(ledger).get("active", "missing") is not None:
            raise ValueError("GPU lease became active during cleanup")
        path.unlink()
        item.update(status="deleted", deleted=True)
        atomic_json(receipt, plan)
    plan.update(status="completed", freed_bytes=sum(i["bytes"] for i in plan["files"])+
        sum(i["before_bytes"]-i["after_bytes"] for i in plan["rewrites"]))
    atomic_json(receipt, plan)
    export()
    return plan


def run():
    lock = (WORK / "queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = read_json(WORK / "queue.json")
    if record["status"] != "prepared":
        raise ValueError("queue has already started; explicit audited retry required")
    record.update(status="waiting", pid=os.getpid(), process_identity=process_identity(os.getpid()))
    atomic_json(WORK / "queue.json", record)
    try:
        while True:
            _sources(record)
            waiting = [name for name, identity in record["parents"].items() if parent_active(identity)]
            record["waiting_for"] = waiting
            atomic_json(WORK / "queue.json", record)
            if not waiting:
                break
            time.sleep(10)
        config = validate_config(read_json(WORK / "config.json"))
        parents = _study(config)
        from .followup_supervisor import execute
        from .followup_report import write
        for phase in PHASES:
            _sources(record)
            record.update(status="running", current_phase=phase, waiting_for=[],
                gpu_work_started=True if record["gpu_work_started"] else None,
                gpu_activity_source="authoritative GPU admission and live charge are in ledger.active; null means pending current supervisor observation")
            atomic_json(WORK / "queue.json", record)
            result = execute(config, phase, WORK, LEDGER, parents, AUTH)
            record["gpu_work_started"] = record["gpu_work_started"] or bool(result.get("gpu_work_started"))
            record["completed_phases"].append(dict(phase=phase, status=result.get("status"),
                verdict=result.get("verdict", result.get("decision")), output=result.get("output")))
            atomic_json(WORK / "queue.json", record)
            write(WORK, OUTPUT, ledger=LEDGER)
            if result.get("cleanup_confirmed") is not True:
                raise RuntimeError("phase cleanup is not confirmed; following phase cannot start")
        record.update(status="finalizing", current_phase=None)
        atomic_json(WORK / "queue.json", record)
        cleanup(WORK, LEDGER, export=lambda:write(WORK,OUTPUT,ledger=LEDGER))
        record.update(status="completed", current_phase=None)
        atomic_json(WORK / "queue.json", record)
        write(WORK, OUTPUT, ledger=LEDGER)
    except BaseException as error:
        record.update(status="failed", reason=type(error).__name__+": "+str(error), traceback=traceback.format_exc())
        atomic_json(WORK / "queue.json", record)
        try:
            from .followup_report import write
            write(WORK,OUTPUT,ledger=LEDGER if LEDGER.exists() else None)
        except BaseException as report_error:
            record["report_error"] = type(report_error).__name__+": "+str(report_error)
            atomic_json(WORK / "queue.json", record)
        raise
    finally:
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "inspect", "report"))
    parser.add_argument("--parents", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.parents is None:
            parser.error("prepare requires observed pinned parent process identities")
        value = prepare(read_json(args.parents))
    elif args.command == "run":
        run()
        value = read_json(WORK / "queue.json")
    elif args.command == "report":
        from .followup_report import write
        value = write(WORK,OUTPUT,ledger=LEDGER if LEDGER.exists() else None)
    else:
        value = read_json(WORK / "queue.json")
    print({k:value.get(k) for k in ("status", "gpu_work_started", "waiting_for", "current_phase", "completed_phases")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
