"""CPU-only incremental imports of captures from verified completed launches.

This never launches a workload, Docker, profiler collection, or finalization.
Raw captures must have a clean matching completed launch and remain unchanged
across two polls. Imports share .postprocess.lock with the ordinary helper.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


DEFAULT_JOBS = ('jobs-resume.json', 'jobs-replay.json', 'jobs-trace.json', 'jobs-ncu.json')
TERMINAL = {'completed', 'failed', 'timeout', 'interrupted', 'gpu_monitor_unavailable',
            'blocked_foreign_gpu_workload', 'interrupted_foreign_gpu_workload'}
REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('workflow_postprocess_helper', REPO / 'docker/postprocess-workflow-study.py')
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def read_document(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def option(command, names):
    for index, item in enumerate(command):
        for name in names:
            if item == name and index + 1 < len(command):
                return command[index + 1]
            if item.startswith(name + '='):
                return item[len(name) + 1:]
    return None


def capture_path(root, command):
    profiler = next((i for i, item in enumerate(command) if Path(item).name in {'nsys', 'ncu'}), None)
    if profiler is None:
        return None
    kind = Path(command[profiler]).name
    module = command.index('-m') if '-m' in command else len(command)
    value = option(command[profiler + 1:module - 1], ('--output', '-o') if kind == 'nsys' else ('--export', '-o'))
    if value is None:
        raise ValueError('profiler command lacks a capture output')
    path = Path(value)
    if '..' in path.parts:
        raise ValueError('capture output contains parent traversal')
    container = Path('/data/workflow') / root.name
    if path.is_relative_to(container):
        path = root / path.relative_to(container)
    elif not path.is_absolute():
        path = root / path
    suffix = '.nsys-rep' if kind == 'nsys' else '.ncu-rep'
    if not str(path).endswith(suffix):
        path = Path(str(path) + suffix)
    expected = root / ('traces' if kind == 'nsys' else 'ncu')
    if path.parent != expected or 'probe' in path.name.lower():
        raise ValueError('capture must be a non-probe file directly in traces/ or ncu/')
    return kind, path


def load_jobs(root, manifests):
    jobs, bindings = {}, []
    for path in manifests:
        document, digest = read_document(path)
        if Path(document['root']).resolve() != root:
            raise ValueError('job manifest root differs from watcher root: ' + str(path))
        bindings.append(dict(path=str(path), sha256=digest))
        for value in document['jobs']:
            job = dict(value, source=value.get('source', document.get('source')))
            identifier = job['id']
            if not identifier.replace('-', '').replace('_', '').isalnum():
                raise ValueError('unsafe job identity')
            if identifier in jobs and jobs[identifier] != job:
                raise ValueError('conflicting job definitions: ' + identifier)
            jobs[identifier] = job
    return jobs, bindings


def launch_state(root, job):
    path = root / 'launches' / job['id'] / 'launch.json'
    if not path.exists():
        return dict(id=job['id'], status='not_started', ledger=str(path))
    try:
        value, digest = read_document(path)
    except (OSError, ValueError) as error:
        return dict(id=job['id'], status='unreadable', ledger=str(path), error=str(error))
    return dict(id=job['id'], status=value.get('status', 'unknown'), ledger=str(path),
                ledger_sha256=digest, data=value)


def completed_capture(root, job, state):
    """Fail closed: exact command/source, clean monitor, and capture time window."""
    target = capture_path(root, job['command'])
    if target is None:
        return None, 'not a profiler job'
    kind, path = target
    value = state.get('data', {})
    if state['status'] != 'completed' or value.get('returncode') != 0:
        return None, 'launch is not completed with exit code 0'
    command = value.get('command', [])
    if command[-len(job['command']):] != job['command'] or value.get('source') != job.get('source'):
        return None, 'launch command/source does not match current job'
    monitor = value.get('gpu_monitor')
    if not isinstance(monitor, dict) or monitor.get('admission') != 'passed' or monitor.get('conflicts') or monitor.get('errors'):
        return None, 'GPU monitor did not establish a clean launch'
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        return None, 'symlink capture paths are not eligible'
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None, 'completed launch has no capture file'
    start, finish = value.get('started_unix_ns'), value.get('finished_unix_ns')
    if not path.is_file() or not isinstance(start, int) or not isinstance(finish, int) or not start <= stat.st_mtime_ns <= finish:
        return None, 'capture timestamp is outside matching launch window'
    fingerprint = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, state['ledger_sha256']]
    return dict(kind=kind, path=str(path), job_id=job['id'], fingerprint=fingerprint,
                ledger=state['ledger'], ledger_sha256=state['ledger_sha256']), None


def process_identity(pid):
    try:
        directory = Path('/proc') / str(pid)
        stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        command = [item.decode(errors='replace') for item in (directory / 'cmdline').read_bytes().split(b'\0') if item]
        if stat[0] == 'Z' or not command:
            return None
        return dict(pid=int(pid), start_ticks=int(stat[19]), command=command,
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError, ValueError, IndexError):
        return None


def active_runners(root, manifests, states, *, identities=None, now_ns=None):
    """Current command identities dominate historical failed launch statuses."""
    if identities is None:
        identities = [value for path in Path('/proc').iterdir() if path.name.isdigit()
                      for value in [process_identity(int(path.name))] if value]
    manifest_names = {str(path) for path in manifests}
    resume = str(root / 'resume-after-reboot.sh')
    found = {}
    now_ns = time.time_ns() if now_ns is None else now_ns
    for identity in identities:
        command = identity['command']
        if resume in command or (any(Path(arg).name == 'run-workflow-suite.py' for arg in command)
                                 and option(command, ('--jobs',)) in manifest_names):
            found[identity['pid']] = identity
    by_pid = {value['pid']: value for value in identities}
    boot_seconds = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
    for state in states:
        if state['status'] != 'running':
            continue
        data = state.get('data', {})
        identity = by_pid.get(data.get('launcher_pid'))
        start = data.get('started_unix_ns', 0)
        if identity:
            process_start = (boot_seconds + identity['start_ticks'] / os.sysconf('SC_CLK_TCK')) * 1e9
            command_text = ' '.join(identity['command'])
            if abs(process_start - start) < 10e9 and ('run-workflow-profile.sh' in command_text or 'docker' in command_text):
                found[identity['pid']] = identity
        elif not data.get('launcher_pid') and 0 <= now_ns - start < 30e9:
            found[-1] = dict(pid=None, reason='new admission ledger within process creation grace')
    return list(found.values())


def terminal_status(states, runners, idle_since, now, idle_grace):
    if runners:
        return None, None
    idle_since = now if idle_since is None else idle_since
    if now - idle_since < idle_grace:
        return None, idle_since
    if states and all(state['status'] in TERMINAL for state in states):
        return 'pipeline_terminal', idle_since
    return 'pipeline_stopped_incomplete', idle_since


def validate_candidate(root, manifests, candidate):
    jobs, _ = load_jobs(root, manifests)
    job = jobs[candidate['job_id']]
    current, reason = completed_capture(root, job, launch_state(root, job))
    if current is None or current['fingerprint'] != candidate['fingerprint']:
        raise RuntimeError('completed capture changed or launch became ineligible: ' + str(reason))
    # A distinct retry/job must not be writing the same output prefix.
    for other in jobs.values():
        if other['id'] == job['id']:
            continue
        target = capture_path(root, other['command'])
        if target and str(target[1]) == candidate['path'] and launch_state(root, other)['status'] != 'completed':
            raise RuntimeError('another unfinished job shares capture output')


def primary_waiting(root):
    try:
        value, _ = read_document(root / '.postprocess-primary-waiting.json')
        identity = process_identity(value['pid'])
        return bool(identity and identity['start_ticks'] == value['start_ticks']
                    and time.time_ns() <= value['deadline_unix_ns'])
    except (OSError, ValueError, KeyError):
        return False


def run_child(command, stdout, stderr, *, timeout, stop, validate=None, environment=None, csv_target=None):
    """CPU command only; finish the current import before yielding shared locks."""
    if stop.is_set():
        raise RuntimeError('watcher stop requested')
    if validate:
        validate()
    output = csv_target.open('x') if csv_target else stdout
    try:
        result = subprocess.run(list(map(str, command)), stdout=output, stderr=stderr,
                                env=environment, cwd=REPO, timeout=timeout, check=False)
        if result.returncode:
            raise RuntimeError('CPU postprocess command returned ' + str(result.returncode))
        if validate:
            validate()
    finally:
        if csv_target:
            output.close()
    return list(map(str, command))


def process_pass(args, candidates, stop, *, preview=False):
    """Nonblocking shared lock, held only while exporting/parsing/previewing."""
    root = args.root
    with (root / '.postprocess.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='lock_busy', processed=[])
        if primary_waiting(root):
            return dict(status='yielded_primary', processed=[])
        path = root / 'postprocess.json'
        ledger = json.loads(path.read_text()) if path.exists() else dict(
            schema='ditserve.workflow.postprocess.v1', root=str(root), cpu_only=True, captures=[])
        entries = {row['capture']: row for row in ledger.get('captures', [])}
        processed = []
        yielded = False
        for candidate in candidates:
            if stop.is_set():
                break
            if primary_waiting(root):
                yielded = True
                break
            validate = lambda: validate_candidate(root, args.jobs, candidate)
            try:
                validate()
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                processed.append(dict(capture=candidate['path'], status='ineligible', error=str(error)))
                continue
            def logged(command, stdout_log, stderr_log, *, timeout, csv_target=None):
                with stdout_log.open('a') as stdout, stderr_log.open('a') as stderr:
                    stdout.write('\n[watch CPU import] ' + json.dumps(list(map(str, command))) + '\n')
                    stdout.flush()
                    return run_child(command, stdout, stderr, timeout=timeout, stop=stop,
                                     validate=validate, csv_target=csv_target)
            original = HELPER.run_logged
            HELPER.run_logged = logged
            try:
                entry = HELPER.process_capture(candidate['kind'], Path(candidate['path']), args,
                                               entries.get(candidate['path']), validate=validate)
            finally:
                HELPER.run_logged = original
            entry['completed_launch'] = {key: candidate[key] for key in ('job_id', 'ledger', 'ledger_sha256')}
            entries[candidate['path']] = entry
            processed.append(entry)
            ledger.update(captures=list(entries.values()), status='incremental_running', cpu_only=True,
                          updated_unix_ns=time.time_ns(), writer='completed_capture_watcher')
            HELPER.atomic_json(path, ledger)
        yielded = yielded or primary_waiting(root)
        result = dict(status='yielded_primary' if yielded else 'completed', processed=processed)
        if preview and not stop.is_set() and not yielded:
            env = dict(os.environ, PYTHONPATH=str(REPO / 'workers'))
            command = [sys.executable, '-m', 'ditserve.evaluation.workflow_report', '--root', str(root), '--preview']
            try:
                with (root / 'watch-preview.stdout.log').open('a') as stdout, (root / 'watch-preview.stderr.log').open('a') as stderr:
                    run_child(command, stdout, stderr, timeout=args.timeout, stop=stop, environment=env)
                result['preview'] = dict(status='completed', command=command)
            except Exception as error:
                result['preview'] = dict(status='failed', error=str(error))
        failures = sum(row['status'] == 'failed' for row in entries.values())
        ledger.update(status='failed' if failures else 'completed' if entries else 'no_captures', failures=failures,
                      captures=list(entries.values()), updated_unix_ns=time.time_ns(), writer='completed_capture_watcher')
        if 'preview' in result:
            ledger['preview'] = result['preview']
        HELPER.atomic_json(path, ledger)
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--jobs', type=Path, nargs='+', help='Default: resume, replay, trace and NCU job manifests under root')
    parser.add_argument('--interval', type=float, default=45, help='Poll interval, 30–60 seconds')
    parser.add_argument('--idle-grace', type=float, default=90, help='No active runner grace before stopping, seconds')
    parser.add_argument('--max-hours', type=float, default=48, help='Bounded watcher lifetime')
    parser.add_argument('--cpu', type=int, default=10)
    parser.add_argument('--nice', type=int, default=19)
    parser.add_argument('--timeout', type=int, default=1800, help='Bound each CPU export/import/preview command')
    parser.add_argument('--nsys', type=Path, default=HELPER.DEFAULT_NSYS)
    parser.add_argument('--ncu', type=Path, default=HELPER.DEFAULT_NCU)
    parser.add_argument('--once', action='store_true', help='Observe once, without waiting for a second stability observation')
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.jobs = [path.resolve() for path in args.jobs] if args.jobs else [args.root / name for name in DEFAULT_JOBS]
    args.reparse = False
    if (not args.root.is_dir() or not all(math.isfinite(v) for v in (args.interval, args.idle_grace, args.max_hours))
            or not 30 <= args.interval <= 60 or args.idle_grace < args.interval or args.max_hours <= 0 or args.timeout < 1):
        parser.error('existing root, interval 30–60, idle grace >= interval, and positive limits required')
    if not 0 <= args.nice <= 19:
        parser.error('nice must be in 0..19')
    return args


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(REPO / 'workers'))
    if args.cpu not in os.sched_getaffinity(0):
        raise RuntimeError('requested spare CPU is outside allowed affinity')
    os.sched_setaffinity(0, {args.cpu})
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < args.nice:
        os.nice(args.nice - current_nice)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    started = time.monotonic()
    state_path = args.root / 'watch-postprocess.json'
    stop_path = args.root / 'watch-postprocess.stop'
    with (args.root / '.watch-postprocess.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(schema='ditserve.workflow.postprocess_watch.v1', status='running', cpu_only=True,
                     identity=process_identity(os.getpid()), started_unix_ns=time.time_ns(), cpu=args.cpu,
                     nice=os.getpriority(os.PRIO_PROCESS, 0), interval_s=args.interval, jobs=[str(p) for p in args.jobs],
                     max_hours=args.max_hours, idle_grace_s=args.idle_grace,
                     stop_file=str(stop_path), passes=[], errors=[])
        HELPER.atomic_json(state_path, state)
        observations, attempted, last_preview, idle_since = {}, {}, None, None
        try:
            while not stop.is_set():
                if stop_path.exists():
                    state['status'] = 'stop_requested'
                    break
                if time.monotonic() - started >= args.max_hours * 3600:
                    state['status'] = 'lifetime_limit'
                    break
                jobs, bindings = load_jobs(args.root, args.jobs)
                launches = [launch_state(args.root, job) for job in jobs.values()]
                states = {row['id']: row for row in launches}
                runners = active_runners(args.root, args.jobs, launches)
                candidates, excluded, observed = [], [], {}
                for job in jobs.values():
                    candidate, reason = completed_capture(args.root, job, states[job['id']])
                    if candidate is None:
                        if capture_path(args.root, job['command']) is not None:
                            excluded.append(dict(job_id=job['id'], reason=reason))
                        continue
                    path, fingerprint = candidate['path'], candidate['fingerprint']
                    observed[path] = fingerprint
                    if observations.get(path) == fingerprint and attempted.get(path) != fingerprint:
                        candidates.append(candidate)
                observations = observed
                evidence = hashlib.sha256(json.dumps([(row['id'], row.get('ledger_sha256')) for row in launches
                                                     if row['status'] in TERMINAL], sort_keys=True).encode()).hexdigest()
                preview = evidence != last_preview or bool(candidates)
                action = process_pass(args, candidates, stop, preview=preview) if candidates or preview else dict(status='idle', processed=[])
                if action['status'] != 'lock_busy':
                    for candidate in candidates:
                        if any(row.get('capture') == candidate['path'] and row['status'] != 'ineligible' for row in action['processed']):
                            attempted[candidate['path']] = candidate['fingerprint']
                    if action.get('preview', {}).get('status') == 'completed':
                        last_preview = evidence
                terminal, idle_since = terminal_status(launches, runners, idle_since, time.monotonic(), args.idle_grace)
                event = dict(unix_ns=time.time_ns(), action=action, active_runners=runners,
                             jobs=[{k: v for k, v in row.items() if k != 'data'} for row in launches],
                             eligible_capture_count=len(observed), excluded=excluded)
                state.update(last_poll=event, manifest_bindings=bindings, updated_unix_ns=time.time_ns())
                postprocess = args.root / 'postprocess.json'
                if postprocess.exists():
                    imported = json.loads(postprocess.read_text())
                    state['postprocess_failures'] = sum(row.get('status') == 'failed' for row in imported.get('captures', []))
                    state['preview_status'] = imported.get('preview', {}).get('status')
                state['passes'].append(dict(unix_ns=event['unix_ns'], status=action['status'],
                                            processed=[dict(capture=r['capture'], status=r['status']) for r in action['processed']]))
                with (args.root / 'watch-postprocess.log').open('a') as log:
                    log.write(json.dumps(event, separators=(',', ':')) + '\n')
                HELPER.atomic_json(state_path, state)
                if terminal and action['status'] not in {'lock_busy', 'yielded_primary'}:
                    state['status'] = terminal
                    break
                if args.once:
                    state['status'] = 'observed_once'
                    break
                stop.wait(args.interval)
            if stop.is_set():
                state['status'] = 'stop_requested'
        except Exception as error:
            state['status'] = 'failed'
            state['errors'].append(dict(type=type(error).__name__, message=str(error)))
        finally:
            state['finished_unix_ns'] = time.time_ns()
            HELPER.atomic_json(state_path, state)
        print(json.dumps(dict(status=state['status'], state=str(state_path), pid=os.getpid())), flush=True)
        return int(state['status'] == 'failed' or bool(state.get('postprocess_failures')) or state.get('preview_status') == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
