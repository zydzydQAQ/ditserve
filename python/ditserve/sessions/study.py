"""CPU-only continuation of finite V100 phases with unlimited accounting."""

import argparse
import fcntl
import json
import os
import time
import uuid
from pathlib import Path

from ..research.common import atomic_json, digest, file_digest, finite, read_json
from . import runner


ATTEMPT_FILES = ("runtime-config.json", "sources", "worker.log", "worker.json", "result.json")


def _active(ledger):
    return read_json(ledger).get("active") if ledger.exists() else None


def _relative(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError("cleanup requires a canonical relative path")
    (root / relative).resolve().relative_to(root)
    return relative


def process_alive(pid):
    if type(pid) is not int or pid <= 1:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        fields = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z"
    except (OSError, IndexError):
        return True


def verify_queue_amendment(config, output, *, alive=process_alive):
    """Read-only handoff audit; this function never sends signals or edits history."""
    upstream = Path(config.get("prerequisite_run", runner.DEFAULT_PREREQUISITE_RUN)).resolve()
    path = upstream / "queue-amendment.json"
    value = read_json(path)
    if (value.get("schema_version") != "ditserve.scheduling.queue_amendment.v1" or
            value.get("authorization") != "user_2026_10_15_start_ordered_session_experiments" or
            Path(value.get("replacement_run", ".")).resolve() != Path(output).resolve() or
            value.get("preserved_active_phase") != "completion" or
            set(value.get("cancelled_unstarted_phases", [])) != {"granularity", "replicate"}):
        raise ValueError("queue amendment does not authorize this replacement and cancelled phases")
    stopped = value.get("stopped_controller_pid")
    if (type(stopped) is not int or stopped <= 1 or
            stopped in (value.get("preserved_supervisor_pid"), value.get("preserved_worker_pid")) or alive(stopped)):
        raise ValueError("the old waiting controller is still active or its identity is unsafe")
    args = value.get("stopped_controller_args", [])
    if args[-2:] != ["research-scheduling-next", "continue"]:
        raise ValueError("amendment did not stop the expected waiting-only controller")
    snapshot = Path(value.get("prior_controller_snapshot", ".")).resolve()
    snapshot.relative_to(upstream)
    original = upstream / "controller.json"
    prior = read_json(snapshot)
    if (prior.get("phase") != "completion" or prior.get("status") != "waiting" or
            prior != read_json(original)):
        raise ValueError("prior waiting controller snapshot does not match preserved history")
    for phase in ("granularity", "replicate"):
        if any((upstream / phase / name).exists() for name in ATTEMPT_FILES + ("supervisor.json",)):
            raise ValueError("a cancelled phase was already attempted")
    with (upstream / "controller.lock").open("r") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return dict(path=str(path), sha256=file_digest(path), snapshot=str(snapshot),
                snapshot_sha256=file_digest(snapshot), original_controller_sha256=file_digest(original),
                stopped_controller_pid=stopped,
                cancelled_unstarted_phases=["granularity", "replicate"])


def _cleanup(output, ledger, report, *, receipt_path, summary_path, controller_id, clock):
    """Explicit new-run deletion after workbook coverage and dependency closure."""
    working = output / "retention-working.json"
    receipt = dict(schema_version="ditserve.sessions.retention.v1", controller_id=controller_id,
        status="retained", dependencies_closed=True, files=[], started_at=clock(),
        reason="all declared phases are technically complete; original measurements remain in verified workbook",
        retained_control_files="protocol, ledger, supervision and frozen source remain for identity and restart readers")
    try:
        if report.get("archive", {}).get("status") != "verified":
            raise ValueError("final workbook archive is not verified")
        candidates = []
        for item in report["archive"].get("files", []):
            relative = _relative(output, item["path"])
            if item.get("cleanup_eligible") is not True or relative.suffix.lower() not in (".npy", ".jsonl"):
                continue
            target = output / relative
            if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != output and output in parent.parents):
                raise ValueError("cleanup cannot follow symlinks")
            candidates.append(dict(path=item["path"], sha256=item["sha256"], bytes=item["bytes"],
                                   coverage=item.get("coverage"), state="retained"))
        if len({item["path"] for item in candidates}) != len(candidates):
            raise ValueError("duplicate cleanup inventory entries")
        receipt["files"] = candidates
        with ledger.with_suffix(ledger.suffix + ".lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not ledger.exists() or _active(ledger) or read_json(ledger).get("active_session"):
                raise ValueError("GPU ledger is missing or active")
            for item in candidates:
                target = output / item["path"]
                if target.stat().st_size != item["bytes"] or file_digest(target) != item["sha256"]:
                    raise ValueError("selected file changed since workbook verification")
            receipt["status"] = "in_progress"
            atomic_json(working, receipt)
            for item in candidates:
                target = output / item["path"]
                if target.stat().st_size != item["bytes"] or file_digest(target) != item["sha256"]:
                    raise ValueError("selected file changed before deletion")
                item["state"] = "delete_intent"
                atomic_json(working, receipt)
                target.unlink()
                item["state"] = "deleted"
                atomic_json(working, receipt)
            receipt.update(status="completed", finished_at=clock(),
                           deleted_bytes=sum(item["bytes"] for item in candidates))
            atomic_json(working, receipt)
    except (Exception, KeyboardInterrupt) as error:
        receipt.update(status="retained", reason=str(error), error_type=type(error).__name__, finished_at=clock())
        if any(item["state"] in ("delete_intent", "deleted") for item in receipt["files"]):
            receipt["reason"] += "; partial progress is recorded per file"
    # Keep one small receipt for restart/report readers, and embed the same
    # complete facts in the workbook before removing the temporary journal.
    atomic_json(summary_path, receipt)
    try:
        final = runner.report(output, ledger, retention=receipt)
        if final.get("archive", {}).get("status") == "verified":
            if working.exists():
                working.unlink()
        else:
            receipt["final_report_error"] = "workbook containing cleanup facts is not verified"
    except (Exception, KeyboardInterrupt) as error:
        receipt["final_report_error"] = str(error)
    if "final_report_error" in receipt:
        atomic_json(summary_path, receipt)
        atomic_json(working, receipt)
    return receipt


def _continue_study(config, output, ledger, *, poll_seconds=5.0,
                   monotonic=time.monotonic, sleep=time.sleep, clock=time.time):
    """Consume existing evidence and run each declared phase at most once."""
    poll_seconds = finite(poll_seconds, "poll_seconds", positive=True)
    if poll_seconds > 5:
        raise ValueError("controller polling must be at most five seconds")
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    if (output / "results.xlsx").exists() and not (output / "protocol.json").exists():
        from .reporting import archived, load_archive
        if archived(output):
            value = load_archive(output)
            closed = (value.get("retention") or {}).get("final_archive") is True
            return dict(status="completed" if closed else "stopped", technical_complete=True, archive_only=True,
                        archive=value.get("archive"), retention=value.get("retention"))
    source = Path(config.get("prerequisite_run", runner.DEFAULT_PREREQUISITE_RUN)).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("new output and read-only source run must be separate directory trees")
    amendment = verify_queue_amendment(config, output)
    runner.bind_protocol(config, output, ledger)
    path = output / "controller.json"
    with (output / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read_json(path) if path.exists() else None
        if previous and (previous.get("config_sha256") != digest(config) or previous.get("ledger") != str(ledger)):
            raise ValueError("controller configuration or ledger differs from its existing record")
        if previous and previous.get("technical_complete"):
            summary = Path(previous["cleanup"]["summary_path"])
            actual = read_json(summary) if summary.exists() else dict(
                status="retained", reason="terminal record already saved; inspect cleanup receipt",
                receipt_path=previous["cleanup"]["receipt_path"])
            return dict(previous, actual_cleanup=actual)
        controller_id = str(uuid.uuid4())
        summary_path = output / "cleanup-summary.json"
        if summary_path.exists():
            summary_path = output / ("cleanup-summary-" + controller_id + ".json")
        receipt_path = output / "cleanup-summary.json"
        state = dict(schema_version="ditserve.sessions.controller.v1", controller_id=controller_id,
            config_sha256=digest(config), output=str(output), prerequisite_run=str(source), ledger=str(ledger),
            started_at=clock(), status="waiting", phase="upstream_completion", phases={},
            queue_amendment=amendment,
            previous_status=previous.get("status") if previous else None,
            cleanup=dict(status="planned_verified_prune", receipt_path=str(receipt_path),
                summary_path=str(summary_path), prerequisite_run_read_only=True,
                reason="only verified new-run raw arrays and covered event logs after dependencies close; necessary control evidence retained"))

        def save(status=None, reason=None, **values):
            if status is not None:
                state["status"] = status
            if reason is not None:
                state["reason"] = reason
            state.update(values, updated_at=clock())
            atomic_json(path, state)

        def report_phase(phase, result):
            supervisor = output / phase / "supervisor.json"
            state["phases"][phase] = dict(status=result.get("status"), eligible=result.get("eligible"),
                technical_complete=runner.prerequisite_complete(result, phase, config),
                supervisor=str(supervisor) if supervisor.exists() else None,
                sha256=file_digest(supervisor) if supervisor.exists() else None)
            save(reason="writing phase report before continuing", phase=phase)
            report = runner.report(output, ledger)
            archive = report.get("archive", {})
            state["last_report"] = dict(phase=phase, status=report.get("status"),
                                        archive={key: value for key, value in archive.items() if key != "files"},
                                        generated_at=clock())
            save()

        def finish(reason, *, complete=False):
            if not complete:
                state["cleanup"].update(status="retained", reason="technical failure or incomplete evidence retained")
            save("completed" if complete else "stopped", reason, finished_at=clock(),
                 technical_complete=complete, scientific_complete=complete)
            # Terminal control state precedes final workbook generation; cleanup
            # facts are subsequently merged through the explicit retention API.
            try:
                report = runner.report(output, ledger)
            except (Exception, KeyboardInterrupt) as error:
                report = dict(status="failed", archive=dict(status="deferred", reason=str(error)))
            if complete:
                actual = _cleanup(output, ledger, report, receipt_path=receipt_path,
                    summary_path=summary_path, controller_id=controller_id, clock=clock)
                return dict(state, actual_cleanup=actual)
            return dict(state, final_report=report)

        def wait_existing(phase):
            save("waiting", "waiting for the existing supervised phase", phase=phase)
            no_session_since = None
            while True:
                supervisor = output / phase / "supervisor.json"
                if supervisor.exists():
                    return read_json(supervisor)
                if (output / phase / "cleanup-unverified.json").exists():
                    return dict(status="incomplete", eligible=False, reason="owned GPU cleanup remains unverified")
                active = _active(ledger)
                if active is not None:
                    if (active.get("phase") != phase or active.get("identity", {}).get("output") != str(output)):
                        return dict(status="incomplete", eligible=False, reason="another unresolved accounting session exists")
                    no_session_since = None
                    deadline = active.get("identity", {}).get("deadline_monotonic")
                    if deadline is not None and monotonic() > deadline + 120.0:
                        return dict(status="incomplete", eligible=False,
                                    reason="existing phase watchdog and cleanup grace expired; session was not restarted")
                else:
                    no_session_since = monotonic() if no_session_since is None else no_session_since
                    if monotonic() - no_session_since >= 10.0:
                        return dict(status="incomplete", eligible=False,
                                    reason="existing attempt has no active session or supervised completion; refusing to rerun")
                sleep(poll_seconds)

        try:
            save("waiting", "waiting for the preserved video completion before session work")
            while True:
                verify_queue_amendment(config, output)
                upstream = runner.upstream_completion(config)
                if upstream["status"] != "waiting":
                    break
                sleep(poll_seconds)
            from ..scheduling.next_reporting import report as report_upstream
            prior_report = report_upstream(source, ledger=source.parent / "gpu-budget.json")
            state["upstream_completion"] = dict(status=upstream["status"], path=upstream["path"],
                sha256=upstream.get("sha256"), report_path=str(source / "REPORT.md"),
                completion=prior_report.get("completion"), excel=prior_report.get("excel", {}))
            print(json.dumps(dict(event="prior_completion_reported", status=upstream["status"],
                report=str(source / "REPORT.md"), completion=prior_report.get("completion")), ensure_ascii=False), flush=True)
            save()
            if upstream["status"] != "passed":
                return finish("prior video completion did not pass supervision; no session GPU work started")
            for phase in runner.PHASES:
                supervisor = output / phase / "supervisor.json"
                if supervisor.exists():
                    result = read_json(supervisor)
                elif _active(ledger) or any((output / phase / name).exists() for name in ATTEMPT_FILES):
                    result = wait_existing(phase)
                else:
                    save("waiting", "waiting for the authorized V100 to become idle", phase=phase)
                    while True:
                        save("running", "starting supervised " + phase, phase=phase)
                        result = runner.execute(config, output, ledger, phase, wait_idle=0.0)
                        if (result.get("status") == "blocked" and result.get("gpu_work_started") is False
                                and result.get("reason_code") in ("device_unavailable", "implementation_pending")):
                            save("waiting", result.get("reason", "waiting for the authorized V100 to become idle"), phase=phase)
                            sleep(poll_seconds)
                            continue
                        break
                report_phase(phase, result)
                if not runner.prerequisite_complete(result, phase, config):
                    return finish(phase + " did not complete technically: " + str(result.get("reason", result.get("status"))))
            return finish("all declared phases completed; scientific eligibility remains in each phase report", complete=True)
        except (Exception, KeyboardInterrupt) as error:
            state["error_type"] = type(error).__name__
            return finish("controller stopped: " + str(error))


def finalize_closed_run(config, output, ledger):
    """After controller exit, close its file readers and retain two deliverables."""
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    if ledger == output or output in ledger.parents:
        return dict(status="retained", reason="the cumulative ledger still requires a live path inside this run")
    controller_path, lock_path = output / "controller.json", output / "controller.lock"
    controller = read_json(controller_path)
    if not controller.get("technical_complete") or controller.get("config_sha256") != digest(config):
        return dict(status="retained", reason="controller dependencies are not closed")
    # The active controller must have exited its with-block before this check.
    try:
        with lock_path.open("r") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return dict(status="retained", reason="controller still owns its live file readers")
    journal = output / "finalizing.json"
    if journal.exists():
        return dict(status="retained", reason="an existing finalization journal needs recovery", journal=str(journal))
    previous = read_json(output / "cleanup-summary.json") if (output / "cleanup-summary.json").exists() else {}
    receipt = dict(schema_version="ditserve.sessions.retention.v1", status="in_progress",
        dependencies_closed=True, final_archive=False, files=list(previous.get("files", [])),
        main_config_sha256=digest(config), started_at=time.time(),
        closure_reason="all declared phases, output comparisons and reporting finished; archived readers replace live control files",
        excluded_external_ledger=str(ledger))
    # Creating this marker blocks all new live-run admission before controls are
    # removed. It is the only temporary file exempt from the source whitelist.
    with journal.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
    try:
        with ledger.with_suffix(ledger.suffix + ".lock").open("a+") as account_lock:
            fcntl.flock(account_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if _active(ledger):
                raise ValueError("GPU accounting is active")
            report = runner.report(output, ledger, retention=receipt)
            archive = report.get("archive", {})
            if archive.get("status") != "verified":
                raise ValueError("workbook does not verify complete control evidence")
            if report.get("pilot", {}).get("technical_complete") is not True:
                raise ValueError("paired pilot evidence is not complete; keep its live sources")
            covered = {item["path"]: item for item in archive.get("files", [])}
            pending = []
            for target in sorted(output.rglob("*")):
                if target.is_symlink():
                    raise ValueError("finalization cannot remove or follow symbolic links")
                if not target.is_file() or target.name in ("results.xlsx", "REPORT.md") or target == journal:
                    continue
                name = target.relative_to(output).as_posix()
                item = covered.get(name)
                if (not item or item.get("error") or target.stat().st_size != item.get("bytes") or
                        file_digest(target) != item.get("sha256")):
                    raise ValueError("final source is not covered by the verified workbook: " + name)
                if target.suffix not in (".json", ".jsonl", ".npy", ".py", ".log", ".lock", ".txt"):
                    raise ValueError("unknown artifact type must be reviewed before finalization: " + name)
                if target.suffix in (".npy", ".jsonl") and item.get("cleanup_eligible") is not True:
                    raise ValueError("artifact dependencies or archive coverage remain open: " + name)
                if target.suffix == ".json" and item.get("coverage") != "complete_typed_scalars":
                    raise ValueError("JSON control evidence is not completely archived: " + name)
                if target.suffix == ".jsonl" and item.get("coverage") != "aggregated_events":
                    raise ValueError("event evidence lacks an explicit archive aggregation: " + name)
                pending.append(dict(path=name, bytes=item["bytes"], sha256=item["sha256"],
                                    coverage=item["coverage"], state="retained"))
            if len({item["path"] for item in pending}) != len(pending):
                raise ValueError("duplicate finalization source")
            receipt["files"].extend(pending)
            atomic_json(journal, receipt)
            for item in pending:
                target = output / item["path"]
                if target.stat().st_size != item["bytes"] or file_digest(target) != item["sha256"]:
                    raise ValueError("final source changed before deletion: " + item["path"])
                item["state"] = "delete_intent"
                atomic_json(journal, receipt)
                target.unlink()
                item["state"] = "deleted"
                atomic_json(journal, receipt)
            receipt.update(status="completed", final_archive=True, finished_at=time.time(),
                           deleted_bytes=sum(item.get("bytes", 0) for item in receipt["files"] if item.get("state") == "deleted"))
            atomic_json(journal, receipt)
            final = runner.report(output, ledger, retention=receipt, archive_only=True)
            if final.get("archive", {}).get("status") != "verified":
                raise ValueError("final archive with deletion facts did not verify")
            journal.unlink()
            for directory in sorted((item for item in output.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
                directory.rmdir()
            if {item.name for item in output.iterdir()} != {"results.xlsx", "REPORT.md"}:
                raise ValueError("unexpected files remain after finalization")
            return dict(status="completed", final_archive=True, files=["results.xlsx", "REPORT.md"],
                        deleted_bytes=receipt["deleted_bytes"], external_ledger=str(ledger))
    except (Exception, KeyboardInterrupt) as error:
        receipt.update(status="retained", final_archive=False, reason=str(error), finished_at=time.time())
        atomic_json(journal, receipt)
        try:
            runner.report(output, ledger, retention=receipt, archive_only=True)
        except Exception:
            pass
        return dict(status="retained", reason=str(error), journal=str(journal))


def continue_study(config, output, ledger, *, finalize=True, **kwargs):
    result = _continue_study(config, output, ledger, **kwargs)
    if finalize and result.get("technical_complete") and not result.get("archive_only"):
        cleanup = result.get("actual_cleanup", {})
        if cleanup.get("status") == "completed" and not cleanup.get("final_report_error"):
            result = dict(result, finalization=finalize_closed_run(config, output, ledger))
        else:
            result = dict(result, finalization=dict(status="retained", reason="verified partial cleanup did not finish"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ditserve.sessions.study")
    parser.add_argument("--config", type=Path, default=runner.DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=runner.DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=runner.DEFAULT_LEDGER)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    result = continue_study(runner.load_config(args.config), args.output, args.ledger, poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("technical_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
