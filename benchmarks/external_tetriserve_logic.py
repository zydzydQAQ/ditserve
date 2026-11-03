"""Run source-bound TetriServe scheduling and real CPU Gloo diagnostics.

This entry point is intended for external_task_runner's fixed-image allocation
and storage watchdog. It loads no model and launches no server. Each independent
path records its actual exit status; a skipped or uncovered check never passes
its gate. All writes remain in a new output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import external_tetriserve_port as port
from external_sd35_round import verify_shared_sources, FROZEN_WORKER_SHA256

BRIDGE_TEST_NAME = 'test_20261103_external_tetriserve_bridge.py'
CRITICAL_TESTS = {
    'test_real_gloo_sp1_sp2_rank_switch_without_new_groups',
    'test_real_cpu_tensors_reject_mixed_schedules_and_broken_cfg_pairs',
}

# Explicit argv/sys.path, no import stubs. Keep child processes in the parent's
# process group so the outer storage watchdog kills all descendants together.
SCHEDULER_CHILD = r'''
import runpy, signal, sys
package, shared, timeout, *args = sys.argv[1:]
sys.path[:0] = [package, shared]
def expired(*unused):
    raise TimeoutError('native scheduler diagnostic deadline exceeded')
signal.signal(signal.SIGALRM, expired)
signal.alarm(int(timeout))
sys.argv = [package + '/external_tetriserve_port.py', *args]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
GLOO_CHILD = r'''
import hashlib, importlib.util, io, json, os, signal, sys, unittest
package, shared, test_file, test_sha, timeout = sys.argv[1:]
sys.path[:0] = [package, shared, os.path.dirname(test_file)]
os.environ['DITSERVE_SHARED_PYTHON_ROOT'] = shared
with open(test_file, 'rb') as stream:
    if hashlib.sha256(stream.read()).hexdigest() != test_sha:
        raise ValueError('bridge test source changed before execution')
spec = importlib.util.spec_from_file_location('test_20261103_external_tetriserve_bridge', test_file)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
def ids(node):
    if isinstance(node, unittest.TestSuite):
        return [name for child in node for name in ids(child)]
    return [node.id()]
selected = ids(suite)
def expired(*unused):
    raise TimeoutError('real CPU Gloo diagnostic deadline exceeded')
signal.signal(signal.SIGALRM, expired)
signal.alarm(int(timeout))
log = io.StringIO()
result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
record = dict(schema='ditserve.tetriserve-bridge-tests.v1', selected=selected,
    tests_run=result.testsRun, successful=result.wasSuccessful(),
    skipped=[dict(test=test.id(), reason=why) for test, why in result.skipped],
    failures=[dict(test=test.id(), traceback=why) for test, why in result.failures],
    errors=[dict(test=test.id(), traceback=why) for test, why in result.errors],
    log_tail=log.getvalue()[-16000:], qualification_eligible=False, allows=[])
print('DITSERVE_TEST_RESULT=' + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
sys.exit(0 if result.wasSuccessful() and not result.skipped and result.testsRun == 5 else 1)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def stop_owned_tree(proc):
    """Kill only this subprocess and its observed descendants on local timeout."""
    parents = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm may contain spaces/parentheses; parent PID follows final ')'.
            tail = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            parents[int(entry.name)] = int(tail[1])
        except (OSError, ValueError, IndexError):
            continue
    owned = {proc.pid}
    while True:
        more = {pid for pid, parent in parents.items() if parent in owned}
        if more <= owned:
            break
        owned |= more
    # Stop the direct producer first so it cannot create additional children.
    for pid in [proc.pid, *sorted(owned - {proc.pid}, reverse=True)]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_child(argv, env, timeout):
    start = time.monotonic()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env, start_new_session=False)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout + 15)
    except subprocess.TimeoutExpired:
        timed_out = True
        stop_owned_tree(proc)
        out, err = proc.communicate(timeout=10)
    except BaseException:
        stop_owned_tree(proc)
        proc.communicate(timeout=10)
        raise
    result = dict(argv=argv, returncode=proc.returncode, timed_out=timed_out,
                  seconds=time.monotonic() - start, stdout_tail=out[-16000:], stderr_tail=err[-16000:])
    for line in reversed(out.splitlines()):
        if line.startswith('DITSERVE_TEST_RESULT='):
            try:
                result['tests'] = json.loads(line.partition('=')[2])
            except ValueError as exc:
                result['result_parse_error'] = str(exc)
            break
    return result


def assess_scheduler(execution, evidence):
    issues = []
    if execution.get('returncode') != 0 or execution.get('timed_out'):
        issues.append('native scheduler process failed or timed out')
    if not isinstance(evidence, dict) or evidence.get('schema') != 'ditserve.tetriserve-native-scheduler-logic.v1':
        return dict(passed=False, covered=False, issues=issues + ['native scheduler summary missing or invalid'])
    if evidence.get('costs_are_measurements') is not False:
        issues.append('synthetic cost fixture was not explicitly marked')
    records = evidence.get('records', [])
    covered = len(records) == 2 and {row.get('batch') for row in records} == {1, 2}
    for record in records:
        rounds = record.get('rounds', [])
        if len(rounds) < 3:
            covered = False
            issues.append('three native rounds through terminal completion were not covered')
        cursor = {sid: 0 for sid in range(1, record.get('batch', 0) + 1)}
        prompts = {1: 'fox', 2: 'hare'}
        for rows in rounds:
            if sorted(sid for row in rows for sid in row.get('sub_req_ids', [])) != sorted(cursor):
                issues.append('a native round omitted or duplicated an offered request')
            for row in rows:
                if row.get('prompts') != [prompts.get(sid) for sid in row.get('sub_req_ids', [])]:
                    issues.append('split prompt identity differs from its subrequest')
                for sid in row.get('sub_req_ids', []):
                    if row.get('start') != cursor.get(sid):
                        issues.append('native round sequence is not contiguous')
                    cursor[sid] = row.get('end')
        if (not rounds or any(step != 28 for step in cursor.values())
                or record.get('terminal_items') != record.get('batch')
                or record.get('tracker_remaining') != 0 or record.get('complete_steps_covered') is not True):
            issues.append('actual RequestTracker did not finish every original 28-step request')
        first, second = record.get('first', []), record.get('second', [])
        completed = record.get('completed_steps', {})
        if not first or not second or not completed:
            covered = False
            issues.append('no continuation interval was produced')
            continue
        offered = set(range(1, record.get('batch', 0) + 1))
        if rounds[:2] != [first, second]:
            issues.append('first/second views differ from the complete round sequence')
        for ordinal, rows in enumerate(rounds):
            seen = [sid for row in rows for sid in row.get('sub_req_ids', [])]
            if sorted(seen) != sorted(offered):
                covered = False
                issues.append('continuation does not cover every offered subrequest exactly once')
            for row in rows:
                count = row.get('batch')
                if any(len(row.get(field, [])) != count for field in ('sub_req_ids', 'prompts', 'origin_steps', 'final')):
                    issues.append('native batch metadata is not aligned')
                if (type(row.get('start')) is not int or type(row.get('end')) is not int
                        or not 0 <= row['start'] < row['end'] <= 28
                        or row.get('steps') != row['end'] - row['start']):
                    issues.append('native step interval is invalid')
                if ordinal == 0 and row.get('start') != 0:
                    issues.append('first native interval does not begin at zero')
                if any(total != 28 for total in row.get('origin_steps', [])):
                    issues.append('original step count changed')
                if any(type(final) is not bool or final != (row.get('end') == total)
                       for total, final in zip(row.get('origin_steps', []), row.get('final', []))):
                    issues.append('native terminal marker differs from its original step count')
        if {int(sid) for sid in completed} != offered:
            covered = False
            issues.append('continuation does not cover every offered subrequest')
        for row in second:
            if not (type(row.get('start')) is int and type(row.get('end')) is int
                    and 0 < row['start'] < row['end'] <= 28):
                issues.append('continuation absolute interval is invalid')
            for sid in row.get('sub_req_ids', []):
                if row.get('start') != completed.get(str(sid), completed.get(sid)):
                    issues.append('native continuation repeats or skips completed steps')
        if record.get('cursor_contiguous') is not True:
            issues.append('native cursor continuity failed')
        if record.get('batch_metadata_aligned') is not True:
            issues.append('native batch metadata is not aligned')
    if not covered:
        issues.append('both batch-one and batch-two continuations were not covered')
    return dict(passed=covered and not issues, covered=covered, issues=sorted(set(issues)))


def assess_gloo(execution):
    tests = execution.get('tests', {})
    names = {name.rsplit('.', 1)[-1] for name in tests.get('selected', [])}
    covered = CRITICAL_TESTS <= names and tests.get('tests_run') == 5 and not tests.get('skipped', [1])
    passed = (covered and execution.get('returncode') == 0 and not execution.get('timed_out')
              and tests.get('successful') is True and not tests.get('errors', [1])
              and not tests.get('failures', [1]))
    return dict(passed=passed, covered=covered,
                issues=[] if passed else ['real tensor/Gloo checks failed, were skipped, or were not covered'])


def execute(args):
    output = args.output.resolve()
    package = Path(__file__).resolve().parent
    for source in (args.source_root.resolve(), args.shared_python_root.resolve(), package, args.bridge_test_file.resolve().parent):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('diagnostic output must be disjoint from bound source trees')
    # Preserve previous evidence: an existing output is never opened for writing.
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='ditserve.tetriserve-logic-check.v1', status='failed',
        scheduler=dict(status='blocked_source_validation'), gloo=dict(status='blocked_source_validation'),
        logic_gate_passed=False, eligible_for_next_gpu_migration_check=False,
        model_loaded=False, serving_executed=False, gpu_migration_verified=False,
        qualification_eligible=False, allows=[])
    try:
        if Path(port.__file__).resolve() != package / 'external_tetriserve_port.py':
            raise ValueError('port implementation imported from a different source package')
        test_file = args.bridge_test_file.resolve(strict=True)
        if test_file.name != BRIDGE_TEST_NAME or sha(test_file) != args.bridge_test_sha256:
            raise ValueError('bridge test identity or SHA256 differs')
        shared = verify_shared_sources(args.shared_python_root)
        derived = port.apply(args.source_root, args.source_manifest, args.source_manifest_sha256,
                             output / 'derived', adapter_root=package)
        result['binding'] = dict(upstream_commit=port.upstream.COMMIT,
            upstream_manifest_sha256=args.source_manifest_sha256,
            shared_worker_sha256=FROZEN_WORKER_SHA256,
            bridge_test_sha256=sha(test_file), logic_sha256=sha(__file__),
            port_sha256=sha(package / 'external_tetriserve_port.py'),
            bridge_sha256=sha(package / 'external_tetriserve_bridge.py'),
            round_sha256=sha(package / 'external_sd35_round.py'),
            derived_manifest_sha256=derived['manifest_sha256'])
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join([str(package), str(shared)]),
                   DITSERVE_SHARED_PYTHON_ROOT=str(shared), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        scheduler_argv = [sys.executable, '-I', '-B', '-c', SCHEDULER_CHILD,
            str(package), str(shared), str(args.timeout_seconds), 'scheduler-check',
            '--derived-root', derived['root'], '--derived-manifest-sha256', derived['manifest_sha256'],
            '--output', str(output / 'scheduler')]
        gloo_argv = [sys.executable, '-I', '-B', '-c', GLOO_CHILD,
            str(package), str(shared), str(test_file), args.bridge_test_sha256, str(args.timeout_seconds)]
        # A failed import/logic result in one path never prevents running the other.
        for name, argv in [('scheduler', scheduler_argv), ('gloo', gloo_argv)]:
            try:
                execution = run_child(argv, env, args.timeout_seconds)
                result[name] = dict(status='executed', execution=execution)
            except Exception as exc:
                result[name] = dict(status='execution_error', error=repr(exc), traceback=traceback.format_exc()[-16000:])
        scheduler_path = output / 'scheduler/summary.json'
        evidence = None
        if scheduler_path.is_file():
            evidence = json.loads(scheduler_path.read_bytes())
            if evidence.get('derived_manifest_sha256') != derived['manifest_sha256']:
                raise ValueError('native scheduler evidence source binding differs')
            result['scheduler'].update(evidence=evidence, summary_sha256=sha(scheduler_path))
        result['scheduler']['assessment'] = assess_scheduler(result['scheduler'].get('execution', {}), evidence)
        result['gloo']['assessment'] = assess_gloo(result['gloo'].get('execution', {}))
        passed = all(result[name]['assessment']['passed'] for name in ('scheduler', 'gloo'))
        result.update(status='logic_passed' if passed else 'logic_failed_or_uncovered',
                      logic_gate_passed=passed, eligible_for_next_gpu_migration_check=passed)
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc()[-16000:])
    write_json(output / 'summary.json', result)
    print(json.dumps(dict(summary=str(output / 'summary.json'), sha256=sha(output / 'summary.json'),
                          status=result['status'], logic_gate_passed=result['logic_gate_passed'],
                          qualification_eligible=False), sort_keys=True), flush=True)
    return 0 if result['logic_gate_passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--shared-python-root', type=Path, required=True)
    parser.add_argument('--bridge-test-file', type=Path, required=True)
    parser.add_argument('--bridge-test-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=240)
    args = parser.parse_args(argv)
    if not 30 <= args.timeout_seconds <= 600:
        parser.error('per-path timeout must be between 30 and 600 seconds')
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
