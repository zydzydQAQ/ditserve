"""A single-machine rank launcher whose children retain the supervisor's PGID.

Unlike torchrun's elastic subprocess path, ranks never create a new session. The
outer migration supervisor can therefore observe and terminate its complete CUDA
process group without treating owned ranks as foreign jobs. This launcher itself
signals only the child PIDs it started and never any unrelated process group.
"""

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time


class Interrupted(BaseException):
    pass


def _interrupt(signum, frame):
    raise Interrupted("launcher interrupted by signal %d" % signum)


def _stop_children(children):
    for process in children:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 1.0
    for process in children:
        try:
            process.wait(timeout=max(0., deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in children:
        if process.poll() is None:
            process.kill()
    deadline = time.monotonic() + 1.0
    clean = True
    for process in children:
        try:
            process.wait(timeout=max(0., deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            clean = False
    return clean


def _evidence(handle, rank, count, returncode):
    size = handle.tell()
    handle.seek(max(0, size - 16000))
    tail = handle.read().decode("utf-8", errors="replace")
    records = []
    for line in tail.splitlines():
        if line.startswith("MIGRATION_RESULT="):
            try:
                records.append(json.loads(line.split("=", 1)[1]))
            except ValueError:
                records.append(None)
    valid = (returncode == 0 and len(records) == 1 and isinstance(records[0], dict)
             and records[0].get("status") == "passed" and records[0].get("rank") == rank
             and records[0].get("ranks") == count)
    return dict(rank=rank, status="passed" if valid else "failed", returncode=returncode,
                probe=records[0] if len(records) == 1 else None,
                output_tail=tail if not valid else "", output_truncated=size > 16000)


def launch(count, command, *, timeout=180, master_port=None):
    """Run rank commands under a bounded deadline and require every rank's proof."""
    if type(count) is not int or count < 1 or not command:
        raise ValueError("positive rank count and nonempty command required")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    # Only the loopback rendezvous is exposed. A collision during the short bind
    # handoff fails the rank launch under the same bounded deadline.
    if master_port is None:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
    elif type(master_port) is int and 1 <= master_port <= 65535:
        port = master_port
    else:
        raise ValueError("master port must be an integer in 1..65535")
    children, logs = [], []
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    reason = None
    started = time.monotonic()
    try:
        for sig in handlers:
            signal.signal(sig, _interrupt)
        for rank in range(count):
            log = tempfile.TemporaryFile()
            logs.append(log)
            environment = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                               RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(count),
                               LOCAL_WORLD_SIZE=str(count), GROUP_RANK="0", PYTHONUNBUFFERED="1")
            children.append(subprocess.Popen(command, env=environment, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=False))
        while True:
            codes = [process.poll() for process in children]
            if any(code is not None and code != 0 for code in codes):
                reason = "rank_process_failed"
                break
            if all(code is not None for code in codes):
                break
            if time.monotonic() - started >= timeout:
                reason = "timeout"
                break
            time.sleep(min(.05, max(0., timeout - (time.monotonic() - started))))
    except Interrupted:
        reason = "interrupted"
    except OSError as error:
        reason = "rank_spawn_failed:" + type(error).__name__
    finally:
        # Ignore another TERM while reaping; an outer SIGKILL still enforces its
        # deadline. Kill only our child objects, never the inherited process group.
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        clean = _stop_children(children)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    try:
        workers = [_evidence(log, rank, count, process.returncode)
                   for rank, (log, process) in enumerate(zip(logs, children))]
        completed = [row["rank"] for row in workers if row["status"] == "passed"]
        passed = reason is None and clean and completed == list(range(count))
        return dict(status="passed" if passed else "failed", ranks=count,
                    completion_ranks=completed, workers=workers,
                    reason=reason or (None if passed else "incomplete_rank_evidence"),
                    cleanup_confirmed=clean, launcher_pgid=os.getpgrp(),
                    elapsed_seconds=time.monotonic() - started)
    finally:
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--master-port", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = launch(args.nproc, command, timeout=args.timeout, master_port=args.master_port)
    print("MIGRATION_RESULT=" + json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
