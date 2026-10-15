"""Durable serial continuation of the authorized follow-on study.

The chain never creates a GPU lease itself, repairs an active ledger, retries a
failed phase, deletes evidence, or kills a process. Each actual phase is owned
and charged by next_study.execute. Importing this module performs no work.
"""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

from ..research.common import atomic_json, digest, file_digest, read_json
from .next_protocol import CONFIG, OUTPUT, LEDGER, AUTHORIZATION, PHASES, safe_paths


GOOD = frozenset(("passed", "completed", "stopped"))


def _pid_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        # A finished child can remain a zombie until the launcher's next poll;
        # kill(pid, 0) alone would incorrectly keep the serial chain waiting.
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        if stat[stat.rfind(")") + 1:].split()[0] == "Z":
            return False
    except FileNotFoundError:
        return False
    except (OSError, IndexError):
        pass
    return True


def _ordinary(path):
    path = Path(path).absolute()
    if path.resolve() != path or path.is_symlink():
        raise ValueError("chain paths must not traverse symlinks")
    return path


def _ledger(path):
    if not path.exists():
        return dict(active=None, sessions=[], used_gpu_seconds=0.)
    value = read_json(path)
    if ("active" not in value or not isinstance(value.get("sessions"), list)
            or value["active"] is not None and not isinstance(value["active"], dict)):
        raise ValueError("ledger lacks explicit ownership/session evidence")
    return value


def _costs(value, clock):
    active = value.get("active")
    temporary = 0.
    if active:
        started, count = active.get("started_monotonic"), active.get("gpu_count")
        if type(started) not in (int, float) or not math.isfinite(started) or count not in (1, 2):
            raise ValueError("active cost identity is invalid")
        temporary = max(0., clock() - started) * count
    return dict(settled_gpu_seconds=value.get("used_gpu_seconds", 0.),
        stage_used_gpu_seconds=value.get("stage_used_gpu_seconds", {}),
        active_unsettled=bool(active), active_gpu_seconds_estimate=temporary,
        chain_owns_extra_gpu_lease=False)


def _identity(session, output, config_sha, auth):
    identity = session.get("identity", {})
    phase = session.get("stage")
    attempt = Path(identity.get("output", "/missing")).absolute()
    if (phase not in PHASES or identity.get("phase") != phase or identity.get("run_root") != str(output)
            or identity.get("config_sha256") != config_sha or identity.get("authorization_source") != auth
            or attempt.parent != output / phase or attempt.resolve() != attempt):
        raise ValueError("session does not match this run/configuration/authorization")
    return phase


def _terminal(output, phase, ledger, config_sha, auth, expected_session=None):
    path = _ordinary(output / phase / "terminal.json")
    if not path.exists():
        raise ValueError("settled phase terminal is missing: " + phase)
    value = read_json(path)
    charge = value.get("charge", {})
    matches = [row for row in ledger.get("sessions", []) if row.get("session_id") == charge.get("session_id")]
    if len(matches) != 1 or matches[0] != charge or (expected_session and charge.get("session_id") != expected_session):
        raise ValueError("terminal does not match the exact settled session")
    if _identity(charge, output, config_sha, auth) != phase:
        raise ValueError("terminal phase/session differs")
    attempt = Path(charge["identity"]["output"])
    if value.get("output") != str(attempt) or read_json(_ordinary(attempt / "supervisor.json")) != value:
        raise ValueError("terminal differs from the attempt supervisor")
    if value.get("config_sha256") != config_sha or value.get("authorization_source") != auth:
        raise ValueError("terminal configuration/authorization differs")
    final = value.get("final_occupancy", {})
    if final.get("status") != "exclusive_observed" or final.get("owned") != [] or final.get("foreign") != []:
        raise ValueError("settled phase lacks confirmed exclusive cleanup")
    return value, dict(path=str(path), sha256=file_digest(path))


def _success(value):
    return (value.get("status") in GOOD and value.get("returncode") == 0
            and not value.get("supervisor_failure") and value.get("charge", {}).get("status") == "completed")


def run_chain(config_path=CONFIG, output=OUTPUT, ledger=LEDGER, authorization_path=AUTHORIZATION,
              *, wait_session_id=None, execute_fn=None, report_fn=None, audit_fn=None,
              config_loader=None, authorization_loader=None, sleep=time.sleep,
              clock=time.monotonic, pid_alive=_pid_alive):
    """Continue one run, optionally waiting for one exact already-owned session.

    Injected callables exist solely for CPU fixtures. Production uses the
    existing supervisor, reporter and independent final audit without retries.
    """
    from .next_protocol import load_config, authorization
    from .next_study import execute
    from .next_report import write_report
    from .next_audit import final_audit
    execute_fn, report_fn, audit_fn = execute_fn or execute, report_fn or write_report, audit_fn or final_audit
    config_loader, authorization_loader = config_loader or load_config, authorization_loader or authorization
    output, ledger = safe_paths(_ordinary(output), _ordinary(ledger))
    config_path, authorization_path = _ordinary(config_path), _ordinary(authorization_path)
    config, (_, auth) = config_loader(config_path), authorization_loader(authorization_path)
    config_sha, config_file_sha = digest(config), file_digest(config_path)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = _ordinary(output / "chain.lock")
    record_path = _ordinary(output / "chain.json")
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise RuntimeError("another chain owns this run")
    record = dict(schema_version="ditserve.optimizations.next_chain.v1", status="running", pid=os.getpid(),
        output=str(output), ledger=str(ledger), config=dict(path=str(config_path), sha256=config_file_sha),
        config_sha256=config_sha, authorization=auth, wait_session_id=wait_session_id,
        phases=[], current_phase=None, reports=[], started_unix=time.time(),
        raw_retained=True, automatic_retry=False, cleanup_performed=False,
        scope="serial existing supervisors; no extra GPU lease, no raw cleanup; root analysis dependencies remain open")
    def checkpoint(**values):
        try:
            costs = _costs(_ledger(ledger), clock)
        except (OSError, KeyError, TypeError, ValueError) as error:
            costs = dict(unverified=True, reason=type(error).__name__ + ": " + str(error),
                         chain_owns_extra_gpu_lease=False)
        record.update(values, updated_unix=time.time(), costs=costs)
        atomic_json(record_path, record)
    def unchanged():
        if file_digest(config_path) != config_file_sha or file_digest(auth["path"]) != auth["sha256"]:
            raise ValueError("chain configuration/authorization changed")
    def report_checkpoint():
        value = report_fn(output, ledger)
        record["reports"].append({key:value.get(key) for key in ("status", "partial", "report", "markdown", "requests")})
        checkpoint()
    try:
        checkpoint()
        budget = _ledger(ledger)
        active = budget.get("active")
        if active and wait_session_id is None:
            raise ValueError("an active allocation requires its explicit --wait-session-id")
        if wait_session_id:
            known = active if active and active.get("session_id") == wait_session_id else None
            if active and known is None:
                raise ValueError("active allocation differs from the requested wait session")
            if known is None:
                matches = [s for s in budget["sessions"] if s.get("session_id") == wait_session_id]
                if len(matches) != 1:
                    raise ValueError("requested wait session is not active or uniquely settled")
                known = matches[0]
            phase = _identity(known, output, config_sha, auth)
            supervisor_pid = known.get("supervisor_pid")
            if type(supervisor_pid) is not int or supervisor_pid <= 0 or supervisor_pid == os.getpid():
                raise ValueError("wait session lacks a separate supervisor PID")
            checkpoint(status="waiting", current_phase=phase, waiting_for=dict(session_id=wait_session_id,
                supervisor_pid=supervisor_pid, phase=phase))
            while True:
                unchanged()
                budget = _ledger(ledger)
                current = budget.get("active")
                if not current:
                    break
                if current.get("session_id") != wait_session_id or current.get("supervisor_pid") != supervisor_pid:
                    raise ValueError("another session appeared while waiting")
                if current.get("identity") != known.get("identity") or current.get("gpu_count") != known.get("gpu_count"):
                    raise ValueError("wait session identity changed")
                _identity(current, output, config_sha, auth)
                if not pid_alive(supervisor_pid):
                    raise RuntimeError("supervisor exited with its session still active; ownership needs independent audit")
                sleep(5.)
                checkpoint(status="waiting")
            # finish() settles the ledger before execute writes its terminal.
            # Never launch a replacement inside this publication/exit gap.
            deadline, last_error = clock() + 10., None
            while True:
                unchanged()
                budget = _ledger(ledger)
                if budget.get("active"):
                    raise ValueError("new allocation appeared during terminal publication")
                try:
                    waited, binding = _terminal(output, phase, budget, config_sha, auth, wait_session_id)
                    ready = not pid_alive(supervisor_pid)
                except (OSError, KeyError, ValueError) as error:
                    last_error, ready = str(error), False
                if ready:
                    break
                if clock() >= deadline:
                    raise RuntimeError("settled session terminal/supervisor exit not confirmed within 10 seconds: " + (last_error or "supervisor still alive"))
                sleep(min(1., max(0., deadline-clock())))
            checkpoint(status="running", waiting_for=None, waited_terminal=binding)
            report_checkpoint()
            if not _success(waited):
                raise RuntimeError("waited phase failed or incomplete; no automatic retry")
        for phase in PHASES:
            unchanged()
            budget = _ledger(ledger)
            if budget.get("active"):
                raise RuntimeError("another allocation is active; chain will not overlap it")
            checkpoint(status="running", current_phase=phase)
            existing = output / phase / "terminal.json"
            action = "skipped_existing" if existing.exists() else "executed"
            if action == "executed":
                value = execute_fn(config, output, ledger, phase, authorization_path, wait_idle=3600.)
                # A preflight resource block has no allocation or terminal.
                if value.get("status") == "blocked":
                    record["phases"].append(dict(phase=phase, action=action, status="blocked", reason=value.get("reason")))
                    report_checkpoint()
                    raise RuntimeError("phase resource admission blocked; no automatic retry")
            budget = _ledger(ledger)
            if budget.get("active"):
                raise RuntimeError("phase returned while its allocation remains active")
            value, binding = _terminal(output, phase, budget, config_sha, auth)
            record["phases"].append(dict(phase=phase, action=action, status=value.get("status"),
                decision=value.get("decision"), reason=value.get("reason"), terminal=binding,
                session_id=value["charge"]["session_id"], gpu_seconds=value["charge"].get("gpu_seconds")))
            report_checkpoint()
            if not _success(value):
                raise RuntimeError("phase failed or incomplete; no automatic retry")
        unchanged()
        if _ledger(ledger).get("active"):
            raise RuntimeError("allocation appeared before final audit")
        report_checkpoint()
        audit = audit_fn(output, ledger)
        record["final_audit"] = dict(path=str(output / "audits/final-integrity.json"), status=audit.get("status"))
        if audit.get("status") != "passed":
            raise RuntimeError("final independent integrity audit did not pass")
        checkpoint(status="completed", current_phase=None, finished_unix=time.time())
    except BaseException as error:
        checkpoint(status="needs_attention", reason=type(error).__name__ + ": " + str(error))
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        handle.close()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serial charged serving stages; no retries or raw cleanup")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION)
    parser.add_argument("--wait-session-id")
    args = parser.parse_args(argv)
    result = run_chain(args.config, args.output, args.ledger, args.authorization, wait_session_id=args.wait_session_id)
    print(json.dumps({key:result.get(key) for key in ("status", "current_phase", "reason", "output", "costs")}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
