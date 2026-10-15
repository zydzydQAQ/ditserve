"""CPU-only continuation of finite V100 phases with unlimited accounting."""

import argparse
import fcntl
import json
import time
import uuid
from pathlib import Path

from .. import compact_results
from ..research.common import atomic_json, digest, file_digest, finite, read_json
from . import next_runner as runner


ATTEMPT_FILES = ("runtime-config.json", "sources", "worker.log", "worker.json", "result.json")


def _active(ledger):
    return read_json(ledger).get("active") if ledger.exists() else None


def _relative(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError("cleanup requires a canonical relative path")
    (root / relative).resolve().relative_to(root)
    return relative


def _cleanup(output, ledger, report, *, receipt_path, summary_path, controller_id, clock):
    """Delete only verified new-run evidence; never modify exported metadata."""
    actual = dict(schema_version="ditserve.scheduling.next.cleanup.v1", controller_id=controller_id,
                  status="retained", receipt_path=str(receipt_path), summary_path=str(summary_path),
                  selected_paths=[], prune_attempted=False, excel=report.get("excel", {}))
    try:
        if actual["excel"].get("status") != "verified":
            raise ValueError("final Excel export was not verified")
        if not ledger.exists() or _active(ledger) or read_json(ledger).get("active_session"):
            raise ValueError("GPU ledger is missing or still active")
        if receipt_path.exists():
            raise ValueError("existing deletion receipt requires reconciliation")
        inventory = compact_results.plan(output, destination=output / "excel")
        if inventory.get("status") != "planned":
            raise ValueError("cleanup inventory is protected or unavailable")
        entries = {item["path"]: item for item in inventory["files"]}
        selected = []
        for name, item in entries.items():
            relative = _relative(output, name)
            if item.get("kind") == "raw" and relative.suffix.lower() == ".npy":
                selected.append(name)
        for name in report.get("cleanup_summarized_paths", []):
            relative = _relative(output, name)
            if (name not in entries or relative.name != "execution-events.jsonl" or
                    entries[name].get("kind") not in ("text", "retained", "summarized_text")):
                raise ValueError("summarized cleanup path is not an inventoried execution event log")
            selected.append(name)
        actual["selected_paths"] = sorted(set(selected))
        if _active(ledger) or read_json(ledger).get("active_session"):
            raise ValueError("GPU accounting became active before cleanup")
        if selected:
            actual["prune_attempted"] = True
            actual["result"] = compact_results.prune(output / "excel", actual["selected_paths"])
            actual["status"] = actual["result"]["status"]
        else:
            actual.update(status="no_eligible_files", reason="verified archive contains no selected raw arrays or summarized events")
    except (Exception, KeyboardInterrupt) as error:
        actual.update(status="retained", reason=str(error), error_type=type(error).__name__)
        if actual["prune_attempted"] and receipt_path.exists():
            actual["reason"] += "; inspect deletion receipt for any completed deletions"
    actual["finished_at"] = clock()
    try:
        with summary_path.open("x", encoding="utf-8") as stream:
            json.dump(actual, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (Exception, KeyboardInterrupt) as error:
        actual["summary_write_error"] = str(error)
    return actual


def continue_study(config, output, ledger, *, poll_seconds=5.0,
                   monotonic=time.monotonic, sleep=time.sleep, clock=time.time):
    """Consume existing evidence and run each declared phase at most once."""
    poll_seconds = finite(poll_seconds, "poll_seconds", positive=True)
    if poll_seconds > 5:
        raise ValueError("controller polling must be at most five seconds")
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    source = Path(config["source_run"]).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("new output and read-only source run must be separate directory trees")
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
        receipt_path = output / "excel" / compact_results.RECEIPT_NAME
        state = dict(schema_version="ditserve.scheduling.next.controller.v1", controller_id=controller_id,
            config_sha256=digest(config), output=str(output), source_run=str(source), ledger=str(ledger),
            started_at=clock(), status="waiting", phase=runner.PHASES[0], phases={},
            previous_status=previous.get("status") if previous else None,
            cleanup=dict(status="planned_verified_prune", receipt_path=str(receipt_path),
                summary_path=str(summary_path), source_run_read_only=True,
                reason="only completed new-run raw .npy and explicitly summarized execution events; actual deletion is recorded by receipt"))

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
            state["last_report"] = dict(phase=phase, status=report.get("status"),
                                        excel=report.get("excel", {}), generated_at=clock())
            save()

        def finish(reason, *, complete=False):
            if not complete:
                state["cleanup"].update(status="retained", reason="technical failure or incomplete evidence retained")
            save("completed" if complete else "stopped", reason, finished_at=clock(),
                 technical_complete=complete, scientific_complete=complete)
            # No changes to controller or report after this final export.
            try:
                report = runner.report(output, ledger)
            except (Exception, KeyboardInterrupt) as error:
                report = dict(status="failed", excel=dict(status="deferred", reason=str(error)))
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
            save()
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
                                and result.get("reason_code") == "device_unavailable"):
                            save("waiting", "waiting for the authorized V100 to become idle", phase=phase)
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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ditserve.scheduling.next_study")
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
