"""Settle an abandoned study attempt only after exclusive, verified cleanup.

The driver may call recover_study(root, driver_lock_held=True) while holding its
driver.lock. The CLI acquires that lock itself. Budget accounting remains owned
by StudyBudget; its recovery callback holds budget.lock throughout verification.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'workers'))
from ditserve.evaluation.workflow_budget import StudyBudget, atomic_json


class RecoveryBlocked(RuntimeError):
    pass


def process_rows(proc_root=Path('/proc')):
    """Read stable identities for this user's live processes, without signaling."""
    rows = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            before = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            command = (directory / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            after = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if before[19] != after[19] or after[0] in ('Z', 'X', 'x'):
                continue
            rows.append(dict(pid=int(directory.name), start_ticks=int(after[19]),
                             command=[value for value in command if value]))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, IndexError) as error:
            raise RecoveryBlocked('cannot verify process identity: ' + str(directory)) from error
    return rows


def option(command, name):
    for index, value in enumerate(command):
        if value == name:
            return command[index + 1] if index + 1 < len(command) else None
        if value.startswith(name + '='):
            return value.split('=', 1)[1]
    return None


def study_processes(rows, root, source, caller_pid):
    root, source = Path(root).resolve(), Path(source).resolve()
    conflicts = []
    for row in rows:
        if row['pid'] == caller_pid:
            continue
        command = row['command']
        scripts = [Path(value) for value in command
                   if Path(value).name in ('run-serving-study.py', 'run-next-study.py', 'run-workflow-suite.py')]
        for script in scripts:
            selected = option(command, '--jobs' if script.name == 'run-workflow-suite.py' else '--root')
            matched = selected is not None and (Path(selected).resolve() == root or
                Path(selected).resolve().is_relative_to(root / 'jobs'))
            # Frozen script paths identify drivers started with implicit roots.
            matched |= script.is_absolute() and script.resolve().is_relative_to(source)
            if script.name in ('run-serving-study.py', 'run-next-study.py') and not selected:
                run_id = option(command, '--run-id') or ('run03' if script.name == 'run-next-study.py' else 'run02')
                matched |= root == Path('/mnt/ssd0/zyd/ditserve-data/workflow') / run_id
            if matched:
                conflicts.append(row)
                break
    return conflicts


def docker(arguments):
    command = ['docker', '--host', 'unix:///run/ditserve-docker/docker.sock', *arguments]
    return subprocess.run(['sg', 'ditserve-docker', '-c', shlex.join(command)],
                          capture_output=True, text=True, timeout=30)


def inspect_container(target, execute=docker):
    response = execute(['inspect', '--format', '{{.Id}} {{.State.Running}}', target])
    if response.returncode:
        if 'no such object' in response.stderr.lower() or 'no such container' in response.stderr.lower():
            return None
        raise RecoveryBlocked('Docker inspection failed: ' + response.stderr.strip())
    fields = response.stdout.strip().split()
    if len(fields) != 2 or not re.fullmatch('[0-9a-f]{64}', fields[0]) or fields[1] not in ('true', 'false'):
        raise RecoveryBlocked('unexpected Docker container identity')
    return dict(id=fields[0], running=fields[1] == 'true')


def verify_cleanup(active, study, caller_pid, audit, *, processes=process_rows, execute=docker):
    root = Path(study['root'])
    live = study_processes(processes(), root, study['source'], caller_pid)
    if live:
        audit['live_study_processes'] = live
        raise RecoveryBlocked('a previous study driver or suite is still running')
    identifier = active['job_id']
    if not re.fullmatch('[A-Za-z0-9_-]+', identifier):
        raise RecoveryBlocked('unsafe active job identifier')
    ledger_path = root / 'launches' / identifier / 'launch.json'
    if not ledger_path.is_file():
        raise RecoveryBlocked('active attempt has no launch ledger; ownership cannot be verified')
    ledger = json.loads(ledger_path.read_text())
    if ledger.get('budget_attempt_id') != active['id'] or ledger.get('id') != identifier:
        raise RecoveryBlocked('active attempt and launch ledger identities differ')
    name = ledger.get('container', '')
    if not re.fullmatch('ditserve-workflow-' + re.escape(identifier.lower()) + '-[0-9a-f]{12}', name):
        raise RecoveryBlocked('launch ledger lacks an exact generated container name')
    expected = ledger.get('gpu_monitor', {}).get('container_id')
    if expected is not None and not re.fullmatch('[0-9a-f]{64}', expected):
        raise RecoveryBlocked('invalid recorded container ID')
    state = inspect_container(name, execute)
    if state is not None and expected is not None and state['id'] != expected:
        raise RecoveryBlocked('container name now refers to a different immutable ID')
    target = state['id'] if state is not None else expected or name
    audit.update(job_id=identifier, attempt_id=active['id'], container=name,
                 target=target, initial_container=state, commands=[])
    if state is not None and state['running']:
        for arguments in (['stop', '--time', '10', target], ['kill', target]):
            try:
                response = execute(arguments)
                audit['commands'].append(dict(command=arguments, returncode=response.returncode,
                    stdout=response.stdout, stderr=response.stderr))
            except (OSError, subprocess.TimeoutExpired) as error:
                audit['commands'].append(dict(command=arguments, error=str(error)))
            state = inspect_container(target, execute)
            if state is None or not state['running']:
                break
    if state is not None and state['running']:
        raise RecoveryBlocked('experiment container remains running after cleanup')
    # Docker clients still creating a container would evade an absent-name test.
    # The unique name is retained in their argv by docker run / sg wrappers.
    clients = [row for row in processes() if row['pid'] != caller_pid and
               any(name in value for value in row['command'])]
    if clients:
        audit['live_container_clients'] = clients
        raise RecoveryBlocked('recorded container still has a live launcher or watchdog; retry after it settles')
    audit.update(verified_stopped=True, final_container=state)


def recover_study(root, *, caller_pid=None, driver_lock_held=False,
                  processes=process_rows, execute=docker):
    root = Path(root).resolve()
    caller_pid = os.getpid() if caller_pid is None else caller_pid
    if caller_pid != os.getpid():
        raise RecoveryBlocked('only the current process may be exempted from driver identity checks')
    lock = None
    audit = dict(schema='ditserve.workflow.recovery.v1', status='checking', root=str(root),
                 started_unix_ns=time.time_ns(), caller_pid=caller_pid)
    try:
        if not driver_lock_held:
            lock = (root / 'driver.lock').open('a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = json.loads((root / 'study.json').read_text())
        if Path(study['root']).resolve() != root:
            raise RecoveryBlocked('study root binding differs from requested recovery directory')
        def check(active):
            verify_cleanup(active, study, caller_pid, audit, processes=processes, execute=execute)
        with StudyBudget(root, study['budget_seconds'], recovery_check=check) as budget:
            audit.update(status='recovered' if audit.get('verified_stopped') else 'no_active_attempt',
                         charged_seconds=budget.record['charged_seconds'], remaining_seconds=budget.remaining_seconds)
            if audit.get('verified_stopped'):
                audit['accounting'] = budget.record['attempts'][-1]
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        audit.update(status='blocked', error=str(error))
    finally:
        audit['finished_unix_ns'] = time.time_ns()
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
    path = root / 'recoveries' / ('%d-%s.json' % (audit['started_unix_ns'], uuid.uuid4().hex[:8]))
    atomic_json(path, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    result = recover_study(args.root)
    print(json.dumps(result, ensure_ascii=False))
    return int(result['status'] == 'blocked')


if __name__ == '__main__':
    raise SystemExit(main())
