"""Observe GPU process ownership before and during charged measurements.

This is an observation guard, not a GPU reservation service. It never signals
foreign processes, and cannot rule out interference between its samples.
"""

import csv
import io
import os
from pathlib import Path
import subprocess
import time

from .common import atomic_json, digest, file_digest, read_json


class GPUIsolationError(RuntimeError):
    pass


def gpu_occupancy(devices, allowed_pgid=None, *, run=subprocess.run, getpgid=os.getpgid):
    selected = set(devices)
    snapshot = dict(status="unknown", observed_at=time.time(), devices=sorted(selected),
                    allowed_pgid=allowed_pgid, owned=[], foreign=[])
    try:
        result = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise ValueError((result.stderr + result.stdout).strip())
        for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError("malformed GPU process inventory")
            uuid, raw_pid = (value.strip() for value in row)
            if uuid not in selected:
                continue
            pid = int(raw_pid)
            if pid <= 0:
                raise ValueError("GPU process PID unavailable")
            process = dict(uuid=uuid, pid=pid)
            owned = False
            if allowed_pgid is not None:
                try:
                    process["pgid"] = getpgid(pid)
                    owned = process["pgid"] == allowed_pgid
                except ProcessLookupError:
                    # A process that exited between the two reads cannot keep
                    # using the GPU. Other ownership errors remain unknown.
                    continue
            snapshot["owned" if owned else "foreign"].append(process)
        snapshot["status"] = "contended" if snapshot["foreign"] else "exclusive_observed"
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        snapshot["reason"] = str(error)
    return snapshot


def record_observation(output, frozen, stage, snapshot):
    import json
    root = Path(output)
    log_name = stage + "-gpu-isolation.jsonl"
    record = dict(snapshot, protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"])
    record["sha256"] = digest(record)
    with (root / log_name).open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    path = root / "measurement-environment.json"
    if path.exists():
        summary = read_json(path)
        unsigned = {k: v for k, v in summary.items() if k != "sha256"}
        if (summary.get("sha256") != digest(unsigned)
                or summary.get("protocol_sha256") != frozen["sha256"]
                or summary.get("source_sha256") != frozen["source_sha256"]):
            raise GPUIsolationError("measurement environment evidence is not bound to this run")
    else:
        summary = dict(schema_version="ditserve.research.measurement_environment.v1",
                       protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"],
                       status="exclusive_observed", cost_measurements_valid=True, stages={},
                       scope="sampled_process_ownership; not a reservation or proof against sub-sample interference")
    entry = summary["stages"].setdefault(stage, dict(status="exclusive_observed", observations_count=0,
                                                    completion_observed=False, log=log_name))
    entry["observations_count"] += 1
    if snapshot.get("observation_phase") == "completed" and snapshot["status"] == "exclusive_observed":
        entry.update(completion_observed=True, worker_returncode=snapshot.get("worker_returncode"),
                     log_sha256=file_digest(root / log_name))
    if snapshot["status"] != "exclusive_observed":
        entry.update(status=snapshot["status"], reason=snapshot.get("reason", "foreign GPU process observed"))
        summary.update(status=snapshot["status"], cost_measurements_valid=False)
    summary.pop("sha256", None)
    summary["sha256"] = digest(summary)
    atomic_json(path, summary)


def wait_for_exclusive_worker(process, *, timeout, output, frozen, stage, devices,
                              observe=gpu_occupancy, interval=2., monotonic=time.monotonic):
    """Stop the caller on contention; its existing finally cleans only its group."""
    deadline = monotonic() + timeout
    phase = "start"
    while True:
        snapshot = dict(observe(devices, allowed_pgid=process.pid), observation_phase=phase)
        record_observation(output, frozen, stage, snapshot)
        if snapshot["status"] != "exclusive_observed":
            raise GPUIsolationError("GPU measurement isolation lost: " + snapshot["status"])
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            returncode = process.wait(timeout=min(interval, remaining))
        except subprocess.TimeoutExpired:
            phase = "poll"
            continue
        snapshot = dict(observe(devices, allowed_pgid=process.pid), observation_phase="completed",
                        worker_returncode=returncode)
        record_observation(output, frozen, stage, snapshot)
        if snapshot["status"] != "exclusive_observed":
            raise GPUIsolationError("GPU measurement isolation lost: " + snapshot["status"])
        return returncode
