"""CPU controller that continues a supervised scheduling study between phases.

The controller never loads models or owns CUDA. Every new phase passes through
the existing supervisor and shared cumulative ledger. Existing attempts are
consumed or waited on, never retried in place.
"""

import argparse
import fcntl
import json
import time
import uuid
from pathlib import Path

from .. import compact_results
from ..research.common import atomic_json, digest, file_digest, finite, read_json
from . import runner
from .budget import TOTAL


ATTEMPT_FILES = ("runtime-config.json", "sources", "worker.log", "worker.json", "result.json")


def _supervised(output, phase):
    path = output / phase / "supervisor.json"
    return read_json(path) if path.exists() else None


def _active_phase(ledger, output, phase):
    if not ledger.exists():
        return False
    active = read_json(ledger).get("active")
    return bool(active and active.get("phase") == phase and
                active.get("identity", {}).get("output") == str(output))


def _successful(result, *, qualification):
    return (result.get("status") == "passed" and
            (not qualification or result.get("eligible") is True) and
            result.get("returncode", 0) == 0 and not result.get("supervisor_failure") and
            not result.get("budget_stopped"))


def continue_study(config, output, ledger, *, wait_seconds=TOTAL, poll_seconds=2.0,
                   monotonic=time.monotonic, sleep=time.sleep, clock=time.time):
    """Wait for calibration, report each result, and advance qualified phases."""
    wait_seconds = finite(wait_seconds, "wait_seconds", positive=True)
    poll_seconds = finite(poll_seconds, "poll_seconds", positive=True)
    if wait_seconds > TOTAL or poll_seconds > 5.0:
        raise ValueError("controller horizon must be <=12 hours and polling <=5 seconds")
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    runner.bind_protocol(config, output, ledger)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "controller.json"
    deadline = monotonic() + wait_seconds
    with (output / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read_json(path) if path.exists() else None
        if previous and (previous.get("config_sha256") != digest(config) or previous.get("ledger") != str(ledger)):
            raise ValueError("controller configuration or ledger differs from its existing record")
        if (previous and previous.get("scientific_complete") and
                previous.get("cleanup", {}).get("status") == "planned_verified_prune"):
            # A completed export may now authenticate this controller. Restarting
            # the controller must not rewrite metadata referenced by its receipt.
            summary = Path(previous["cleanup"]["summary_path"])
            actual = (read_json(summary) if summary.exists() else
                      dict(status="retained", reason="terminal controller already saved; inspect cleanup receipt",
                           receipt_path=previous["cleanup"]["receipt_path"]))
            return dict(previous, actual_cleanup=actual)
        controller_id = str(uuid.uuid4())
        summary_path = output / "cleanup-summary.json"
        if summary_path.exists():
            summary_path = output / ("cleanup-summary-" + controller_id + ".json")
        receipt_path = output / "excel" / compact_results.RECEIPT_NAME
        state = dict(schema_version="ditserve.scheduling.controller.v1", controller_id=controller_id,
                     config_sha256=digest(config), output=str(output), ledger=str(ledger),
                     started_at=clock(), status="waiting", phase="calibrate", phases={},
                     previous_status=previous.get("status") if previous else None,
                     cleanup=dict(status="planned_verified_prune", receipt_path=str(receipt_path),
                         summary_path=str(summary_path), scientific_complete=False,
                         reason="conditional on complete science, verified Excel and inactive ledger; actual deletion is recorded by receipt"))

        def save(status=None, reason=None, **values):
            if status is not None:
                state["status"] = status
            if reason is not None:
                state["reason"] = reason
            state.update(values, updated_at=clock())
            atomic_json(path, state)

        def report_phase(phase, result):
            result_path = output / phase / "supervisor.json"
            state["phases"][phase] = dict(status=result.get("status"), eligible=result.get("eligible"),
                reason=result.get("reason"), supervisor=str(result_path) if result_path.exists() else None,
                sha256=file_digest(result_path) if result_path.exists() else None)
            save(phase=phase, reason="writing report before any subsequent phase")
            report = runner.report(output, ledger)
            state["phases"][phase]["reported_at"] = clock()
            state["last_report"] = dict(status=report.get("status"), phase=phase,
                                        excel=report.get("excel", {}), generated_at=clock())
            save()

        def finish(reason, *, complete=False, status="stopped"):
            state["cleanup"].update(scientific_complete=complete)
            if not complete:
                state["cleanup"].update(status="retained",
                    reason="incomplete or unqualified evidence retained for investigation")
            save(status, reason, finished_at=clock(), scientific_complete=complete)
            if not complete:
                return state

            # Save terminal state BEFORE the final export. From this point on,
            # controller/report metadata must stay byte-identical for prune's
            # source verification. Only a new, unarchived sidecar may be written.
            actual = dict(schema_version="ditserve.scheduling.cleanup.v1", controller_id=controller_id,
                          status="retained", receipt_path=str(receipt_path), summary_path=str(summary_path),
                          selected_paths=[], prune_attempted=False, finished_at=None)
            try:
                report = runner.report(output, ledger)
                actual["excel"] = report.get("excel", {})
                if actual["excel"].get("status") != "verified":
                    raise ValueError("final Excel export was not verified")
                if not ledger.exists() or read_json(ledger).get("active") or read_json(ledger).get("active_session"):
                    raise ValueError("GPU ledger is missing or still active")
                if receipt_path.exists():
                    raise ValueError("existing deletion receipt requires reconciliation")
                inventory = compact_results.plan(output, destination=output / "excel")
                if inventory.get("status") != "planned":
                    raise ValueError("raw inventory is protected or unavailable")
                selected = []
                for item in inventory["files"]:
                    relative = Path(item["path"])
                    if item.get("kind") != "raw" or relative.suffix.lower() != ".npy":
                        continue
                    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != item["path"]:
                        raise ValueError("raw inventory contains a noncanonical path")
                    (output / relative).resolve().relative_to(output)
                    selected.append(item["path"])
                actual["selected_paths"] = sorted(set(selected))
                # Recheck after planning; compact_results also locks discovered
                # ledgers and verifies every selected source before unlinking.
                account = read_json(ledger)
                if account.get("active") or account.get("active_session"):
                    raise ValueError("GPU accounting became active before cleanup")
                if selected:
                    actual["prune_attempted"] = True
                    actual["result"] = compact_results.prune(output / "excel", actual["selected_paths"])
                    actual["status"] = actual["result"]["status"]
                else:
                    actual.update(status="no_raw_arrays", reason="verified inventory contains no eligible raw .npy files")
            except (Exception, KeyboardInterrupt) as error:
                actual.update(status="retained", reason=str(error), error_type=type(error).__name__)
                # If an unlink was interrupted, the compact receipt, not a claim
                # that all files remain present, is authoritative about progress.
                if actual["prune_attempted"] and receipt_path.exists():
                    actual["reason"] += "; inspect deletion receipt for any completed deletions"
            actual["finished_at"] = clock()
            try:
                with summary_path.open("x", encoding="utf-8") as summary:
                    json.dump(actual, summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                    summary.write("\n")
            except (Exception, KeyboardInterrupt) as error:
                actual["summary_write_error"] = str(error)
            return dict(state, actual_cleanup=actual)

        def wait_supervisor(phase):
            save("waiting", "waiting for the existing supervised " + phase + " attempt", phase=phase)
            while True:
                result = _supervised(output, phase)
                if result is not None:
                    return result
                if (output / phase / "cleanup-unverified.json").exists():
                    return dict(status="blocked", eligible=False, reason="owned GPU cleanup remains unverified")
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return dict(status="incomplete", eligible=False,
                                reason="controller wait horizon exhausted; existing attempt was not restarted")
                sleep(min(poll_seconds, remaining))

        try:
            if config.get("device_preference") == ["V100"]:
                legacy_path = output / "legacy/admission.json"
                if not (output / "legacy/supervisor.json").exists() and not legacy_path.exists():
                    atomic_json(legacy_path, dict(status="blocked", phase="legacy", eligible=False,
                        deferred=True, reason_code="device_constraint", gpu_work_started=False,
                        reason="V100-only study; original A100 legacy matrix is deferred"))
                state["legacy"] = dict(status="deferred", reason="V100-only; no A100 substitution or cross-device pairing")
            save()
            calibration = wait_supervisor("calibrate")
            report_phase("calibrate", calibration)
            if not _successful(calibration, qualification=True):
                return finish("calibration did not qualify: " + str(calibration.get("reason", calibration.get("status"))))
            for phase in ("pilot", "screen", "validate"):
                result = _supervised(output, phase)
                if result is None:
                    attempted = any((output / phase / name).exists() for name in ATTEMPT_FILES)
                    if attempted:
                        if _active_phase(ledger, output, phase):
                            result = wait_supervisor(phase)
                        else:
                            result = dict(status="incomplete", eligible=False,
                                reason="existing attempt has no active session or supervised completion; refusing to rerun")
                    else:
                        remaining = deadline - monotonic()
                        if remaining < 120.0:
                            result = dict(status="incomplete", eligible=False,
                                          reason="controller horizon cannot admit another phase and cleanup")
                        else:
                            save("running", "running supervised " + phase, phase=phase)
                            result = runner.execute(config, output, ledger, phase, timeout=remaining - 45.0)
                report_phase(phase, result)
                if phase == "pilot":
                    if not _successful(result, qualification=False):
                        return finish("pilot did not complete: " + str(result.get("reason", result.get("status"))))
                elif phase == "screen":
                    if not _successful(result, qualification=True):
                        complete = (result.get("status") == "stopped" and result.get("returncode", 0) == 0
                                    and not result.get("budget_stopped") and not result.get("supervisor_failure")
                                    and result.get("gpu_work_started") is not False
                                    and result.get("selection", {}).get("status") == "failed")
                        return finish("screen selected no qualified candidate: " + str(result.get("reason", result.get("status"))),
                                      complete=complete)
                else:
                    complete = (result.get("status") in ("passed", "stopped") and result.get("returncode", 0) == 0
                                and not result.get("budget_stopped") and not result.get("supervisor_failure")
                                and result.get("gpu_work_started") is not False
                                and result.get("validation", {}).get("status") in ("passed", "failed"))
                    return finish("validation finished: " + str(result.get("status")), complete=complete,
                                  status="completed" if complete else "stopped")
        except (Exception, KeyboardInterrupt) as error:
            save("stopped", str(error), error_type=type(error).__name__)
            try:
                report = runner.report(output, ledger)
                state["last_report"] = dict(status=report.get("status"), phase=state.get("phase"),
                                            excel=report.get("excel", {}), generated_at=clock())
            except Exception as report_error:
                state["report_error"] = str(report_error)
            return finish("controller stopped: " + str(error))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ditserve.scheduling.study")
    parser.add_argument("--after-calibration", action="store_true", required=True,
                        help="wait for the existing externally supervised calibration")
    parser.add_argument("--config", type=Path, default=runner.DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=runner.DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=runner.DEFAULT_LEDGER)
    parser.add_argument("--wait-seconds", type=float, default=TOTAL)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    result = continue_study(runner.load_config(args.config), args.output, args.ledger,
                            wait_seconds=args.wait_seconds, poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("scientific_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
