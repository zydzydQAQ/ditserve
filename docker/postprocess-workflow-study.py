"""Export and parse completed workflow captures on the host; never launch GPU work."""

import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback


DEFAULT_NSYS = Path("/mnt/ssd0/zyd/ditserve-data/tools/nsight-workflow/extracted/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys")
DEFAULT_NCU = Path("/opt/nvidia/nsight-compute/2024.1.0/ncu")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def binding(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError("capture changed while hashing; only completed captures may be processed: " + str(path))
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=after.st_size, mtime_ns=after.st_mtime_ns)


def archive(path):
    if not path.exists():
        return None
    backup = path.with_name(path.name + ".previous." + str(time.time_ns()) + ".bak")
    # .bak intentionally prevents the report's JSON discovery from counting
    # superseded parses as independent evidence.
    path.replace(backup)
    return str(backup)


def discover(root):
    captures, excluded = [], []
    for kind, directory, extension in (("nsys", "traces", ".nsys-rep"), ("ncu", "ncu", ".ncu-rep")):
        for path in sorted((root / directory).glob("*" + extension)):
            if "probe" in path.name.lower() or path.is_symlink():
                excluded.append(dict(path=str(path), reason="probe_or_symlink"))
            elif path.is_file():
                captures.append((kind, path))
    return captures, excluded


def integrity_check(path):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
    finally:
        connection.close()
    if rows != [("ok",)]:
        raise RuntimeError("SQLite integrity_check failed: " + repr(rows))


def run_logged(command, stdout_log, stderr_log, *, timeout, csv_target=None):
    """Only callers construct export/import commands; all tool chatter stays in logs."""
    with stdout_log.open("a", encoding="utf-8") as stdout, stderr_log.open("a", encoding="utf-8") as stderr:
        marker = "\n[postprocess %s] %s\n" % (datetime.datetime.now(datetime.timezone.utc).isoformat(), json.dumps(list(map(str, command))))
        stdout.write(marker)
        stderr.write(marker)
        stdout.flush()
        stderr.flush()
        if csv_target is None:
            result = subprocess.run(list(map(str, command)), stdout=stdout, stderr=stderr, timeout=timeout, check=False)
        else:
            with csv_target.open("x", encoding="utf-8") as csv:
                result = subprocess.run(list(map(str, command)), stdout=csv, stderr=stderr, timeout=timeout, check=False)
            # CSV is itself stdout; retain its path in the per-capture stdout log
            # without duplicating potentially large raw exports.
            stdout.write("stdout CSV: " + str(csv_target) + "\n")
        if result.returncode:
            raise RuntimeError("postprocess command returned %d; see %s" % (result.returncode, stderr_log))
    return list(map(str, command))


def export_sqlite(capture, sqlite_path, args, entry, stdout_log, stderr_log):
    if sqlite_path.exists():
        integrity_check(sqlite_path)
        entry["sqlite_export"] = "existing_integrity_checked"
        return
    temporary = sqlite_path.with_name(sqlite_path.stem + ".part.sqlite")
    if temporary.exists():
        entry.setdefault("archived", []).append(archive(temporary))
    command = [args.nsys, "export", "--type", "sqlite", "--output", temporary, capture]
    entry.setdefault("commands", []).append(run_logged(command, stdout_log, stderr_log, timeout=args.timeout))
    if not temporary.is_file():
        raise RuntimeError("Nsight Systems export did not create " + str(temporary))
    integrity_check(temporary)
    temporary.replace(sqlite_path)
    entry["sqlite_export"] = "exported_integrity_checked"


def parse_nsys(capture, parsed, args, entry, stdout_log, stderr_log):
    from ditserve.evaluation.workflow_trace import analyze_trace
    sqlite_path = capture.with_suffix(".sqlite")
    export_sqlite(capture, sqlite_path, args, entry, stdout_log, stderr_log)
    result = analyze_trace(sqlite_path)
    entry["sqlite"] = binding(sqlite_path)
    entry["availability"] = result.get("availability")
    entry["warnings"] = result.get("warnings", [])
    return result


def parse_ncu(capture, parsed, args, entry, stdout_log, stderr_log):
    from ditserve.evaluation.workflow_ncu import parse_csv
    paths = {}
    for label, rename in (("raw", False), ("nvtx", True)):
        target = capture.with_suffix("." + label + ".csv")
        temporary = target.with_suffix(target.suffix + ".part")
        for previous in (temporary, target):
            if previous.exists():
                entry.setdefault("archived", []).append(archive(previous))
        command = [args.ncu, "--import", capture, "--csv", "--page", "raw", "--print-units", "auto", "--print-fp"]
        if rename:
            command.extend(["--print-nvtx-rename", "kernel"])
        entry.setdefault("commands", []).append(run_logged(command, stdout_log, stderr_log,
                                                           timeout=args.timeout, csv_target=temporary))
        temporary.replace(target)
        paths[label] = target
    raw, renamed = paths["raw"].read_text(), paths["nvtx"].read_text()
    result = parse_csv(raw, renamed, scope="sampled_kernels")
    result["source"] = dict(report=str(capture), csv=str(paths["raw"]), nvtx_csv=str(paths["nvtx"]),
                            csv_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                            export_commands=entry["commands"], stderr_logs=[str(stderr_log)])
    entry["csv"] = {label: binding(path) for label, path in paths.items()}
    entry["coverage"] = result.get("coverage")
    entry["warnings"] = result.get("warnings", [])
    return result


def process_capture(kind, capture, args, previous=None, *, validate=None):
    parsed = capture.with_suffix(".parsed.json")
    stdout_log = capture.with_suffix(".postprocess.stdout.log")
    stderr_log = capture.with_suffix(".postprocess.stderr.log")
    entry = dict(kind=kind, capture=str(capture), parsed_path=str(parsed), status="processing",
                 started_unix_ns=time.time_ns(), stdout_log=str(stdout_log), stderr_log=str(stderr_log))
    try:
        if validate is not None:
            validate()
        source = binding(capture)
        entry["source"] = source
        old_source = (previous or {}).get("source")
        if old_source and old_source["sha256"] != source["sha256"]:
            entry["observed_changed_source"] = source
            entry["source"] = old_source
            raise RuntimeError("immutable capture SHA256 changed since previous postprocessing; use a new capture path")
        if parsed.exists() and not args.reparse:
            value = json.loads(parsed.read_text())
            expected = ("ditserve.workflow_trace.v1" if kind == "nsys" else "ditserve.workflow.ncu.v1")
            if value.get("schema_version", value.get("schema")) != expected:
                raise ValueError("existing parsed JSON has an unexpected schema")
            entry.update(status="already_parsed", parser_status=value.get("status"), parsed=binding(parsed),
                         source_binding="previous_ledger" if old_source else "preexisting_filename_association_not_regenerated",
                         availability=value.get("availability"), coverage=value.get("coverage"),
                         warnings=value.get("warnings", []))
            if validate is not None:
                validate()
            return entry
        if parsed.exists():
            entry.setdefault("archived", []).append(archive(parsed))
        value = (parse_nsys if kind == "nsys" else parse_ncu)(capture, parsed, args, entry, stdout_log, stderr_log)
        after = binding(capture)
        if after["sha256"] != source["sha256"] or after["bytes"] != source["bytes"]:
            raise RuntimeError("capture changed during export/parsing; parsed evidence was not published")
        if validate is not None:
            validate()
        value["postprocess"] = dict(capture=source, helper=str(Path(__file__).resolve()),
                                    parsed_unix_ns=time.time_ns(), cpu_only=True)
        atomic_json(parsed, value)
        entry.update(status="parsed", parser_status=value.get("status"), parsed=binding(parsed), source_binding="verified_immutable_capture")
    except Exception as error:
        entry.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        with stderr_log.open("a", encoding="utf-8") as stream:
            stream.write("\n" + traceback.format_exc())
    finally:
        entry["finished_unix_ns"] = time.time_ns()
    return entry


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--nsys", type=Path, default=DEFAULT_NSYS)
    parser.add_argument("--ncu", type=Path, default=DEFAULT_NCU)
    parser.add_argument("--timeout", type=int, default=1800, help="CPU export timeout per command, seconds")
    parser.add_argument("--reparse", action="store_true", help="Archive existing parsed JSON to .bak before regeneration")
    parser.add_argument("--kind", choices=("ncu", "nsys"), help="Process only this capture kind; retain other ledger entries unchanged")
    parser.add_argument("--preview", action="store_true", help="Refresh ANALYSIS-LIVE.html only; never finalize")
    parser.add_argument("--lock-wait", type=float, default=600, help="Bounded wait for another postprocessor, seconds; 0 fails immediately")
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    if not args.root.is_dir() or args.timeout < 1 or not math.isfinite(args.lock_wait) or args.lock_wait < 0:
        parser.error("root must exist; timeout must be positive and lock-wait nonnegative")
    return args


@contextmanager
def study_lock(root, wait_seconds):
    """A waiting primary import asks the incremental watcher to yield promptly."""
    marker = root / '.postprocess-primary-waiting.json'
    identity = dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
                    requested_unix_ns=time.time_ns(), deadline_unix_ns=time.time_ns() + int(wait_seconds * 1e9))
    announced = False
    with (root / '.postprocess.lock').open('a') as lock:
        deadline = time.monotonic() + wait_seconds
        try:
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("another postprocessor holds this study's lock")
                    if not announced:
                        atomic_json(marker, identity)
                        announced = True
                    time.sleep(min(.2, max(0., deadline - time.monotonic())))
            yield
        finally:
            if announced and marker.exists():
                try:
                    if json.loads(marker.read_text()) == identity:
                        marker.unlink()
                except (OSError, ValueError):
                    pass


def main(argv=None):
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "workers"))
    ledger_path = args.root / "postprocess.json"
    with study_lock(args.root, args.lock_wait):
        previous = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        previous_entries = {row["capture"]: row for row in previous.get("captures", [])}
        discovered, excluded = discover(args.root)
        captures = [(kind, path) for kind, path in discovered if args.kind is None or kind == args.kind]
        observed_paths = {str(path) for _, path in discovered}
        untouched = [row for path, row in previous_entries.items()
                     if path in observed_paths and args.kind is not None and row.get("kind") != args.kind]
        missing_previous = [dict(row, status="missing", previous_status=row.get("status"),
                                 observed_missing_unix_ns=time.time_ns(),
                                 missing_reason="Previously recorded capture is absent from the current discovery; immutable binding retained")
                            for path, row in previous_entries.items() if path not in observed_paths]
        ledger = dict(schema="ditserve.workflow.postprocess.v1", root=str(args.root), status="running", cpu_only=True,
            started_unix_ns=time.time_ns(), tools=dict(nsys=str(args.nsys), ncu=str(args.ncu)),
            helper_sha256=binding(Path(__file__).resolve())["sha256"], reparse=args.reparse, selected_kind=args.kind,
            discovery=dict(patterns=["traces/*.nsys-rep", "ncu/*.ncu-rep"], excluded=excluded,
                           selected_count=len(captures), preserved_untouched_count=len(untouched),
                           absent_capture_types=[kind for kind in ("nsys", "ncu") if not any(found == kind for found, _ in discovered)],
                           missing_policy="Absent captures are not measurements and are not mandatory export failures"), captures=missing_previous + untouched,
            interpretation="CPU-only export/import. Missing counters remain unavailable/null in parser output; no zero substitution. No archive finalization.")
        atomic_json(ledger_path, ledger)
        for kind, capture in captures:
            ledger["captures"].append(process_capture(kind, capture, args, previous_entries.get(str(capture))))
            atomic_json(ledger_path, ledger)
        if args.preview:
            stdout_log = args.root / "postprocess-preview.stdout.log"
            stderr_log = args.root / "postprocess-preview.stderr.log"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(repo / "workers") + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
            command = [sys.executable, "-m", "ditserve.evaluation.workflow_report", "--root", str(args.root), "--preview"]
            try:
                with stdout_log.open("a") as stdout, stderr_log.open("a") as stderr:
                    result = subprocess.run(command, env=environment, cwd=repo, stdout=stdout, stderr=stderr,
                                            check=False, timeout=args.timeout)
                ledger["preview"] = dict(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode,
                                         stdout_log=str(stdout_log), stderr_log=str(stderr_log), command=command)
            except Exception as error:
                ledger["preview"] = dict(status="failed", error=str(error), stdout_log=str(stdout_log), stderr_log=str(stderr_log))
        failures = sum(row["status"] == "failed" for row in ledger["captures"]) + int(ledger.get("preview", {}).get("status") == "failed")
        warnings = sum(bool(row.get("warnings")) or row.get("parser_status") in ("partial", "empty") for row in ledger["captures"])
        ledger.update(status="failed" if failures else "no_captures" if not captures else "completed_with_warnings" if warnings else "completed",
                      finished_unix_ns=time.time_ns(), failures=failures, warnings=warnings)
        atomic_json(ledger_path, ledger)
    print(json.dumps(dict(status=ledger["status"], captures=len(captures), failures=failures, ledger=str(ledger_path))), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
