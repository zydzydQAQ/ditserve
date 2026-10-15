"""One CPU-only report/audit refresh after one exact serial chain exits.

No phase execution, retry, lease repair or retention is available here. This
only replaces a stale in-memory final audit with freshly imported code, while
preserving the chain record and a byte-for-byte copy of the preceding audit.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

from ..research.common import atomic_json, digest
from .next_chain import _ordinary, _success, _terminal
from .next_protocol import AUTHORIZATION, CONFIG, LEDGER, OUTPUT, PHASES, safe_paths

AUDIT_FAILURE = "RuntimeError: final independent integrity audit did not pass"
NATIVE = dict(diagnosis="diagnosis.json", sana_vae="sana-candidate.json",
    sana_graph="sana-candidate.json", sana_numerics="sana-numerics.json", conditional="conditional.json")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    path = _ordinary(path)
    before = path.stat()
    _require(path.is_file() and before.st_size <= 128*1024*1024, "metadata source exceeds bound")
    raw = path.read_bytes()
    after = path.stat()
    _require((before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_ino, after.st_size, after.st_mtime_ns), "metadata changed while reading")
    def invalid(value):
        raise ValueError("nonfinite metadata: " + value)
    return json.loads(raw, parse_constant=invalid), dict(path=str(path), bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest()), raw


def _process(pid):
    """Bind Linux process identity; zombies have exited but retain their ID."""
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return None
    fields = stat[stat.rfind(")")+1:].split()
    ticks = int(fields[19])
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    boot_seconds = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                        if line.startswith("btime "))
    return dict(pid=pid, start_ticks=ticks, boot_id=boot, alive=fields[0] not in ("Z", "X"),
                started_unix=boot_seconds+ticks/os.sysconf("SC_CLK_TCK"))


def _process_identity(value):
    return None if value is None else {key:value[key] for key in ("pid", "start_ticks", "boot_id")}


def _identity(chain):
    return {key:chain.get(key) for key in ("pid", "started_unix", "output", "ledger", "config",
                                          "config_sha256", "authorization", "wait_session_id")}


def _allowed_chain(chain):
    return chain.get("status") == "completed" or (chain.get("status") == "needs_attention"
                                                   and chain.get("reason") == AUDIT_FAILURE)


def _stable(bindings):
    for expected in bindings:
        unused, actual, raw = _read(expected["path"])
        _require(actual == expected, "protected metadata changed: " + expected["path"])


def _settled(output, ledger, chain, config_sha, auth):
    budget, ledger_binding, unused = _read(ledger)
    _require("active" in budget and budget["active"] is None, "GPU allocation is still active")
    _require(_allowed_chain(chain), "chain did not complete or stop specifically at its final audit")
    sessions = budget.get("sessions", [])
    _require(len({row.get("session_id") for row in sessions}) == len(sessions), "duplicate settled session IDs")
    entries = chain.get("phases", [])
    _require([entry.get("phase") for entry in entries] == list(PHASES), "chain lacks all eight ordered phase conclusions")
    bindings = [ledger_binding]
    for phase, entry in zip(PHASES, entries):
        terminal, source = _terminal(output, phase, budget, config_sha, auth)
        _require(_success(terminal), "phase failed or incomplete: " + phase)
        matching = [row for row in sessions if row.get("stage") == phase]
        _require(bool(matching) and terminal["charge"] == matching[-1], "terminal is not the latest settled attempt")
        _require(entry.get("status") == terminal["status"] and entry.get("session_id") == terminal["charge"]["session_id"]
            and entry.get("terminal") == source, "chain phase differs from its exact settled terminal")
        attempt = Path(terminal["output"])
        native = attempt / NATIVE.get(phase, phase+".json")
        _require(terminal.get("evidence_path") == str(native), "native evidence leaves its fixed phase path")
        for path in (Path(source["path"]), attempt/"supervisor.json", native, attempt/"config.json", attempt/"study.json"):
            unused, binding, raw = _read(path)
            bindings.append(binding)
    for name in ("study.json", "protocol.json", "workloads.json"):
        unused, binding, raw = _read(output/name)
        bindings.append(binding)
    return bindings


def finalize_after_chain(*, chain_pid, chain_started_unix, config_path=CONFIG, output=OUTPUT,
                         ledger=LEDGER, authorization_path=AUTHORIZATION, poll_seconds=5.,
                         wait_seconds=None, process_start_ticks=None, boot_id=None,
                         process_fn=_process, sleep=time.sleep, clock=time.monotonic,
                         report_fn=None, audit_fn=None):
    """Wait for this chain, then refresh once under its now-unoccupied lock.

    Injected clocks/process/report/audit callables are only for CPU fixtures.
    A rejected precondition never starts a GPU job or repairs any evidence.
    """
    _require(type(chain_pid) is int and chain_pid > 0 and chain_pid != os.getpid(), "invalid separate chain PID")
    _require(type(chain_started_unix) in (int,float) and math.isfinite(chain_started_unix), "invalid chain start timestamp")
    _require(type(poll_seconds) in (int,float) and 0 < poll_seconds <= 10, "poll interval must be at most 10 seconds")
    _require(wait_seconds is None or type(wait_seconds) in (int,float) and math.isfinite(wait_seconds) and wait_seconds >= 0,
             "invalid wait timeout")
    _require((process_start_ticks is None) == (boot_id is None), "process ticks and boot ID must be supplied together")
    output, ledger = safe_paths(_ordinary(output), _ordinary(ledger))
    config_path, authorization_path = _ordinary(config_path), _ordinary(authorization_path)
    chain_path = output/"chain.json"
    chain, chain_source, unused = _read(chain_path)
    config, config_source, unused = _read(config_path)
    authorization, auth_source, unused = _read(authorization_path)
    auth = dict(path=str(authorization_path), sha256=auth_source["sha256"])
    config_sha = digest(config)
    _require(chain.get("pid") == chain_pid and chain.get("started_unix") == chain_started_unix
        and chain.get("output") == str(output) and chain.get("ledger") == str(ledger)
        and chain.get("config_sha256") == config_sha
        and chain.get("config") == dict(path=str(config_path), sha256=config_source["sha256"])
        and chain.get("authorization") == auth, "chain/config/authorization identity differs")
    _require(authorization.get("approved_by_user") is True and authorization.get("scope") == "serving_next_validation"
        and authorization.get("budget_unlimited") is True and "authorized_gpu_seconds" in authorization
        and authorization["authorized_gpu_seconds"] is None, "study authorization differs")
    identity = _identity(chain)
    process = process_fn(chain_pid)
    if process is not None:
        _require(process.get("pid") == chain_pid and type(process.get("start_ticks")) is int and process["start_ticks"] > 0
            and isinstance(process.get("boot_id"),str) and bool(process["boot_id"])
            and type(process.get("alive")) is bool and process.get("started_unix", math.inf) <= chain_started_unix + 2.,
            "PID appears reused or process identity cannot be verified")
    process_identity = _process_identity(process)
    if process_start_ticks is not None:
        _require(process_identity == dict(pid=chain_pid,start_ticks=process_start_ticks,boot_id=boot_id),
                 "chain process start identity differs")
    _require(process is not None and process["alive"] or _allowed_chain(chain), "chain exited without an allowed terminal state")
    parent = output/"audits"
    _ordinary(parent)
    parent.mkdir(exist_ok=True)
    for index in range(1000):
        receipt_dir = parent/("finalize-attempt%03d" % index)
        try:
            receipt_dir.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("finalizer receipt attempt limit reached")
    receipt = receipt_dir/"receipt.json"
    record = dict(schema_version="ditserve.optimizations.next_finalize.v1", status="waiting",
        output=str(output), ledger=str(ledger), chain_initial=chain_source, chain_identity=identity,
        process_identity=process_identity, config=config_source, authorization=auth_source,
        pid=os.getpid(), receipt=str(receipt), gpu_used=False, phases_executed=0,
        chain_modified=False, ledger_modified=False, raw_exported=False, originals_deleted=0)
    atomic_json(receipt, record)
    started, handle = clock(), None
    def check():
        _stable((config_source,auth_source))
        current, binding, unused = _read(chain_path)
        _require(_identity(current) == identity, "chain identity changed while waiting")
        state = process_fn(chain_pid)
        _require(state is None or _process_identity(state) == process_identity, "chain PID was reused")
        return current, binding, state
    try:
        while True:
            chain, chain_source, process = check()
            if process is None or not process["alive"]:
                break
            elapsed = clock()-started
            if wait_seconds is not None and elapsed >= wait_seconds:
                raise RuntimeError("chain exit wait timed out")
            sleep(poll_seconds if wait_seconds is None else min(poll_seconds,wait_seconds-elapsed))
        lock_path = _ordinary(output/"chain.lock")
        handle = lock_path.open("a+")
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("chain lock is occupied after process exit")
        chain, chain_source, process = check()
        _require(process is None or not process["alive"], "chain process is still alive")
        protected = [chain_source,config_source,auth_source] + _settled(output,ledger,chain,config_sha,auth)
        previous_path = output/"audits/final-integrity.json"
        if previous_path.exists() or previous_path.is_symlink():
            unused, previous, raw = _read(previous_path)
            saved = receipt_dir/"previous-final-integrity.json"
            with saved.open("xb") as stream:
                stream.write(raw)
            unused, copy_binding, unused = _read(saved)
            _require(previous["sha256"] == copy_binding["sha256"], "previous audit copy changed")
            record["previous_audit"] = dict(source=previous,preserved_copy=copy_binding)
        record.update(status="refreshing",chain_final=chain_source,protected_sources=protected)
        atomic_json(receipt,record)
        # Deliberately import only now: the old chain's loaded function is never
        # reused, and no model/executor/retention module is imported or invoked.
        if report_fn is None:
            from .next_report import write_report
            report_fn = write_report
        if audit_fn is None:
            from .next_audit import final_audit
            audit_fn = final_audit
        _stable(protected)
        record["report_result"] = report_fn(output,ledger)
        _stable(protected)
        result = audit_fn(output,ledger)
        record["audit_result"] = dict(status=result.get("status"),reason=result.get("error") or result.get("reason"))
        _stable(protected)
        for name in ("report.json","results.xlsx","REPORT.md","excel-verification.json"):
            path = _ordinary(output/name)
            _require(path.is_file(), "fresh report artifact missing: " + name)
        audit, binding, unused = _read(previous_path)
        _require(audit == result and result.get("status") == "passed" and result.get("all_phases_terminal") is True,
                 "fresh independent final audit did not pass")
        record.update(status="completed",audit=binding,waited_seconds=clock()-started,
                      scope="fresh CPU report/audit only; previous chain record remains unchanged")
    except BaseException as error:
        record.update(status="failed",reason=type(error).__name__+": "+str(error))
        if isinstance(error,(KeyboardInterrupt,SystemExit)):
            atomic_json(receipt,record)
            raise
    finally:
        if handle is not None:
            handle.close()
        atomic_json(receipt,record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU-only audit refresh after one exact chain exits; no GPU work")
    parser.add_argument("--chain-pid",type=int,required=True)
    parser.add_argument("--chain-started-unix",type=float,required=True)
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--ledger",type=Path,default=LEDGER)
    parser.add_argument("--authorization",type=Path,default=AUTHORIZATION)
    parser.add_argument("--process-start-ticks",type=int)
    parser.add_argument("--boot-id")
    parser.add_argument("--wait-seconds",type=float)
    args = parser.parse_args(argv)
    value = finalize_after_chain(chain_pid=args.chain_pid,chain_started_unix=args.chain_started_unix,
        config_path=args.config,output=args.output,ledger=args.ledger,authorization_path=args.authorization,
        process_start_ticks=args.process_start_ticks,boot_id=args.boot_id,wait_seconds=args.wait_seconds)
    print(json.dumps({key:value.get(key) for key in ("status","reason","receipt")},ensure_ascii=False),flush=True)
    return 0 if value["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
