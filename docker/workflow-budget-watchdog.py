"""Independent cleanup watchdog; targets only one generated experiment container."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import time


def identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] in ('Z', 'X') else fields[19]
    except (OSError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--container', required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--parent-start', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--monotonic-deadline', type=float, required=True)
    parser.add_argument('--deadline-reason', choices=('budget_deadline', 'task_timeout'), default='budget_deadline')
    parser.add_argument('--event', type=Path, required=True)
    parser.add_argument('--done', type=Path, required=True)
    parser.add_argument('--armed', type=Path, required=True)
    parser.add_argument('--launcher-record', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.deadline) or not math.isfinite(args.monotonic_deadline):
        raise ValueError('watchdog deadlines must remain finite even with unlimited cumulative budget')
    if not re.fullmatch(r'ditserve-workflow-[a-z0-9_-]+-[0-9a-f]{12}', args.container):
        raise ValueError('watchdog requires an exact generated experiment container name')
    args.armed.touch()
    while not args.done.exists():
        reason = ('launcher_disappeared' if identity(args.parent_pid) != args.parent_start else
                  args.deadline_reason if time.monotonic() >= args.monotonic_deadline else None)
        if reason:
            launcher_known = False
            if args.launcher_record.exists():
                launch = json.loads(args.launcher_record.read_text())
                launcher_known = True
                if identity(launch['pid']) == str(launch['start_ticks']):
                    try:
                        os.killpg(launch['pid'], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            command = ['docker', '--host', 'unix:///run/ditserve-docker/docker.sock',
                       'stop', '--time', '10', args.container]
            record = dict(reason=reason, container=args.container, started_unix=time.time(), verified_stopped=False)
            try:
                try:
                    result = subprocess.run(['sg', 'ditserve-docker', '-c', shlex.join(command)],
                        capture_output=True, text=True, timeout=25)
                    record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                except subprocess.TimeoutExpired:
                    record['stop_timeout'] = True
                inspect = ['docker', '--host', 'unix:///run/ditserve-docker/docker.sock',
                           'inspect', '--format', '{{.State.Running}}', args.container]
                state = subprocess.run(['sg', 'ditserve-docker', '-c', shlex.join(inspect)],
                                       capture_output=True, text=True, timeout=10)
                absent = state.returncode != 0 and ('no such' in state.stderr.lower())
                if state.returncode == 0 and state.stdout.strip() == 'true':
                    kill = ['docker', '--host', 'unix:///run/ditserve-docker/docker.sock', 'kill', args.container]
                    subprocess.run(['sg', 'ditserve-docker', '-c', shlex.join(kill)],
                                   capture_output=True, text=True, timeout=15)
                    state = subprocess.run(['sg', 'ditserve-docker', '-c', shlex.join(inspect)],
                                           capture_output=True, text=True, timeout=10)
                    absent = state.returncode != 0 and ('no such' in state.stderr.lower())
                stopped = state.returncode == 0 and state.stdout.strip() == 'false'
                # A parent crash between Popen and PID publication may leave a
                # Docker create in flight. Keep watching the unique name through
                # the deadline instead of treating the first absence as cleanup.
                record['verified_stopped'] = stopped or (absent and
                    (launcher_known or time.monotonic() >= args.monotonic_deadline))
            except Exception as error:
                record['error'] = str(error)
            record['finished_unix'] = time.time()
            temporary = args.event.with_suffix('.tmp')
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
            temporary.replace(args.event)
            if record['verified_stopped']:
                return
            time.sleep(2.)
            continue
        time.sleep(min(1., max(.05, args.monotonic_deadline - time.monotonic())))


if __name__ == '__main__':
    main()
