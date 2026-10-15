"""Resume unstarted stages from the already verified executable snapshot.

The failed queue receipt and completed phase are immutable. Workspace changes
are recorded but are never copied into resumed workers. All new GPU sessions
append to the existing unlimited ledger using the unchanged supervisor.
"""
import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

from ..research.common import atomic_json, digest, file_digest, read_json
from .followup_queue import WORK, OUTPUT, LEDGER, AUTH, cleanup, process_identity
from . import followup_supervisor as supervisor
from . import followup_report as reporting
from .protocol import ROOT

REMAINING = ("sana_combo", "granularity")
RECEIPT = WORK / "resume01.json"


def ordinary(path):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve()!=path:
        raise ValueError("continuation paths cannot traverse symbolic links")
    return path


def binding(path):
    path = ordinary(path)
    return dict(path=str(path), sha256=file_digest(path), bytes=path.stat().st_size)


def check_binding(item):
    path = Path(item["path"])
    if path.is_symlink() or path.resolve()!=path or file_digest(path)!=item["sha256"]:
        raise ValueError("bound continuation evidence changed: " + str(path))


def check_snapshot(source):
    root = ordinary(source["execution_root"])
    if digest(source["source_files"]) != source["source_sha256"]:
        raise ValueError("snapshot manifest digest mismatch")
    for name, sha in source["source_files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("source manifest must use contained relative paths")
        path = root / name
        if path.resolve()!=path or root not in path.parents or path.is_symlink() or file_digest(path)!=sha:
            raise ValueError("frozen executable changed: " + name)


def check_loaded_modules(source):
    """Only actual supervisor dependencies need to match the old workspace.

    Workers never use these workspace imports: they run the complete copied
    snapshot. This also fails closed if an unexpected changed module is loaded.
    """
    for name, module in list(sys.modules.items()):
        if not name.startswith("ditserve.") or name==__name__:
            continue
        path = getattr(module, "__file__", None)
        if not path or not path.endswith(".py"):
            continue
        path = Path(path).resolve()
        try:
            relative = str(path.relative_to(ROOT))
        except ValueError:
            raise ValueError("supervisor import outside workspace: " + name)
        if relative not in source["source_files"] or file_digest(path)!=source["source_files"][relative]:
            raise ValueError("loaded supervisor dependency changed: " + relative)


class SnapshotFreezer:
    def __init__(self, manifest, controller):
        self.manifest, self.controller = manifest, controller

    def __call__(self, config, run_root, phase, attempt):
        check_binding(self.manifest)
        check_binding(self.controller)
        source = read_json(self.manifest["path"])
        check_snapshot(source)
        if phase not in REMAINING or attempt!="attempt000" or digest(config)!=source["config_sha256"]:
            raise ValueError("resume cannot change phase, attempt, or scientific configuration")
        run_root = ordinary(run_root)
        protocol = read_json(ordinary(run_root / "protocol.json"))
        if protocol.get("config")!=config or protocol.get("sha256")!=digest(config):
            raise ValueError("frozen protocol changed")
        target = ordinary(run_root / "sources" / (phase + "-" + attempt))
        target.mkdir(parents=True, exist_ok=False)
        for name, sha in source["source_files"].items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(source["execution_root"]) / name, path)
            if file_digest(path)!=sha:
                raise ValueError("snapshot copy did not match source")
        result = dict(source, execution_root=str(target), phase=phase, attempt=attempt,
            derived_from_manifest=self.manifest, continuation_controller=self.controller)
        atomic_json(target / "manifest.json", result)
        return result


@contextmanager
def frozen_source_factory(factory):
    original = supervisor.freeze
    supervisor.freeze = factory
    try:
        yield
    finally:
        supervisor.freeze = original


def write_report(work_root=WORK, output=OUTPUT, ledger=LEDGER, receipt=RECEIPT):
    """Retain the original failure alongside the separately bound live resume."""
    report = reporting.collect(work_root, ledger=ledger)
    continuation = read_json(receipt)
    if continuation.get("schema_version")!="ditserve.followup_resume.v1":
        raise ValueError("unknown continuation receipt")
    check_binding(continuation["original_queue"])
    item = binding(receipt)
    report["sources"].append(item)
    report["tables"]["Sources"].append([item[k] for k in ("path", "sha256", "bytes")])
    reporting._metadata(report["tables"]["Config"], "study", "", "separate continuation; original failed queue retained",
        continuation, str(receipt), report["omissions"])
    for name, rows in report["tables"].items():
        report["tables"][name] = [[None if value=="" else value for value in row] for row in rows]
    report["status"] = "resume01:" + continuation["status"]
    destination = ordinary(output)
    destination.mkdir(parents=True, exist_ok=True)
    markdown, excel = destination / "REPORT.md", destination / "results.xlsx"
    if markdown.is_symlink() or excel.is_symlink() or (markdown.exists() and
        not markdown.read_text().startswith(reporting.MARKER)):
        raise ValueError("refusing to replace unrelated report")
    with tempfile.TemporaryDirectory(prefix=".followup-resume-", dir=destination) as tmp:
        tmp = Path(tmp)
        verification = reporting._write_excel(tmp / "results.xlsx", report["tables"])
        rendered = reporting.render(report)
        rendered += "\n原队列因工作区文件变化停止；其失败记录保持原样。后两阶段从首阶段已校验的源码快照继续，工作区新修改未进入GPU实验。续跑状态与来源见工作簿 Config 表。\n"
        (tmp / "REPORT.md").write_text(rendered)
        if (tmp / "REPORT.md").read_text()!=rendered:
            raise ValueError("report readback mismatch")
        (tmp / "results.xlsx").replace(excel)
        (tmp / "REPORT.md").replace(markdown)
    return dict(status=report["status"], excel=verification, markdown=str(markdown),
        errors=report["errors"]+[e for p in report["phases"] for e in p["errors"]],
        original_queue_modified=False, originals_deleted=0)


def prepare():
    if RECEIPT.exists():
        raise FileExistsError("resume01 already exists; no automatic overwrite/retry")
    queue = read_json(WORK / "queue.json")
    if queue.get("status")!="failed" or not queue.get("reason", "").startswith("ValueError: queued executable changed:"):
        raise ValueError("expected independently failed source guard")
    if process_identity(queue["pid"]) is not None:
        raise ValueError("original queue process is still alive")
    budget = read_json(LEDGER)
    if budget.get("active", "missing") is not None or len(budget.get("sessions", []))!=1:
        raise ValueError("expected exactly one settled phase")
    terminal = read_json(WORK / "finish_copy/terminal.json")
    charge = budget["sessions"][0]
    if (terminal.get("status")!="completed" or terminal.get("returncode")!=0 or
        terminal.get("cleanup_confirmed") is not True or terminal.get("charge")!=charge or
        charge.get("status")!="completed" or charge.get("stage")!="finish_copy"):
        raise ValueError("finish-copy result lacks exact successful settlement")
    for phase in REMAINING:
        if (WORK / phase).exists():
            raise ValueError("remaining phase has already started: " + phase)
    manifest = binding(WORK / "sources/finish_copy-attempt000/manifest.json")
    source = read_json(manifest["path"])
    check_snapshot(source)
    check_loaded_modules(source)
    for key, sha in queue["sources"].items():
        if source["source_files"].get(key)!=sha:
            raise ValueError("original queue and executed snapshot differ")
    drift = [dict(path=name, frozen_sha256=sha,
             current_sha256=file_digest(ROOT/name) if (ROOT/name).is_file() else None)
             for name,sha in queue["sources"].items() if not (ROOT/name).is_file() or file_digest(ROOT/name)!=sha]
    config, study = read_json(WORK / "config.json"), read_json(WORK / "study.json")
    if digest(config)!=source["config_sha256"] or digest(config)!=budget["config_sha256"]:
        raise ValueError("configuration differs from completed phase")
    controls = [binding(WORK/name) for name in ("config.json", "protocol.json", "study.json", "finish_copy/terminal.json")]
    controls += [binding(AUTH)] + study["parent_ledgers"]
    receipt = dict(schema_version="ditserve.followup_resume.v1", status="prepared", remaining=list(REMAINING),
        original_queue=binding(WORK / "queue.json"), original_ledger=binding(LEDGER),
        original_sessions=budget["sessions"], original_used_gpu_seconds=budget["used_gpu_seconds"],
        source_manifest=manifest, controller=binding(__file__), immutable_controls=controls,
        observed_workspace_drift=drift, completed_phases=[], current_phase=None,
        ledger=str(LEDGER), source_policy="copy the completed phase snapshot; never refreeze mutable workspace",
        authorization=binding(AUTH), defaults_unchanged=True)
    atomic_json(RECEIPT, receipt)
    return receipt


def run():
    with (WORK / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = read_json(RECEIPT)
        if record["status"]!="prepared":
            raise ValueError("continuation already started; preserve evidence")
        record.update(status="starting", pid=os.getpid(), process_identity=process_identity(os.getpid()))
        atomic_json(RECEIPT, record)
        try:
            config = read_json(WORK / "config.json")
            study = read_json(WORK / "study.json")
            parents = tuple(Path(x["path"]) for x in study["parent_ledgers"])
            factory = SnapshotFreezer(record["source_manifest"],record["controller"])
            source = read_json(record["source_manifest"]["path"])
            for phase in REMAINING:
                for item in [record["original_queue"],record["controller"]]+record["immutable_controls"]:
                    check_binding(item)
                budget = read_json(LEDGER)
                if budget.get("active") is not None or budget["sessions"][:1]!=record["original_sessions"]:
                    raise ValueError("original billing history changed or lease remains active")
                check_loaded_modules(source)
                record.update(status="running", current_phase=phase)
                atomic_json(RECEIPT,record)
                with frozen_source_factory(factory):
                    result = supervisor.execute(config,phase,WORK,LEDGER,parents,AUTH)
                record["completed_phases"].append({key:result.get(key) for key in
                    ("phase","status","decision","output","cleanup_confirmed")})
                atomic_json(RECEIPT,record)
                write_report()
                if result.get("cleanup_confirmed") is not True:
                    raise RuntimeError("cleanup unconfirmed; cannot start next phase")
            record.update(status="finalizing",current_phase=None)
            atomic_json(RECEIPT,record)
            cleanup(WORK,LEDGER,export=write_report)
            record.update(status="completed")
            atomic_json(RECEIPT,record)
            write_report()
        except BaseException as error:
            record.update(status="failed",reason=type(error).__name__+": "+str(error),traceback=traceback.format_exc())
            atomic_json(RECEIPT,record)
            try:
                write_report()
            except BaseException as report_error:
                record["report_error"] = str(report_error)
                atomic_json(RECEIPT,record)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command",choices=("prepare","run","report"))
    args=parser.parse_args(argv)
    if args.command=="prepare":
        value=prepare()
    elif args.command=="run":
        run(); value=read_json(RECEIPT)
    else:
        value=write_report()
    print({key:value.get(key) for key in ("status","current_phase","completed_phases","errors")})


if __name__=="__main__":
    main()
