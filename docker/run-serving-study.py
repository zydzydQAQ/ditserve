"""Prepare or resume the bounded run02 study; never launch concurrent GPU jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'workers'))
from ditserve.evaluation.workflow_budget import atomic_json
from ditserve.evaluation import workflow_serving_study as study_api

CONTROLLER_REVISION = None


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def verify_revision(revision):
    if not revision:
        return
    source = Path(revision['source'])
    actual = [study_api.binding(path) for path in sorted(source.rglob('*'))
              if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc']
    for row in actual:
        row['path'] = str(Path(row['path']).relative_to(source))
    if actual != revision['files'] or study_api.digest(actual) != revision['source_digest']:
        raise RuntimeError('controller/analysis revision binding changed')


def controller_source(study):
    if read(Path(study['root']) / 'analysis-source.json') != CONTROLLER_REVISION:
        raise RuntimeError('analysis revision pointer changed during execution')
    verify_revision(CONTROLLER_REVISION)
    return CONTROLLER_REVISION['source'] if CONTROLLER_REVISION else study['source']


def result(study, identifier):
    job = next(value for value in reversed(study['jobs']) if value.get('logical_id', value['id']) == identifier and not value.get('superseded'))
    ledger = read(Path(study['root']) / 'launches' / job['id'] / 'launch.json', {})
    if ledger.get('status') != 'completed':
        return None
    value = read(Path(job['output']) / 'run.json')
    return value if value and value.get('status') in ('completed', 'qualification_failed') else None


def family_spent(study, family):
    jobs = {row['id']: row for row in study['jobs']}
    budget = read(Path(study['root']) / 'budget.json', {})
    return sum(row['charged_seconds'] for row in budget.get('attempts', [])
               if jobs.get(row['job_id'], {}).get('family') == family)


def family_available(study, family):
    order = list(study['family_caps_seconds'])
    prefix = order[:order.index(family) + 1]
    return sum(study['family_caps_seconds'][name] - family_spent(study, name) for name in prefix)


def estimated_seconds(item):
    config = read(item['input'], {})
    if config.get('observations') and config.get('nominal_rate'):
        obs = config['observations'][0]
        # Frozen arrivals plus a conservative service tail, model load and warmup.
        return obs['requests'][-1]['arrival_ms'] / 1000 + max(row['slo_ms'] for row in obs['requests']) / 1000 + 120
    if item['family'] == 'sp':
        return 600
    return {'calibration': 1500, 'quality': 1500, 'overlap': 1800}.get(item['id'], 600)


def run_blocks(study, jobs):
    groups = {}
    for item in jobs:
        groups.setdefault(item['block'], []).append(item)
    for key, group in groups.items():
        todo = [item for item in group if read(Path(study['root']) / 'launches' / item['id'] / 'launch.json', {}).get('status') != 'completed']
        required = sum(estimated_seconds(item) for item in todo) * 1.2 + 90
        if todo and (remaining(study) < required or family_available(study, group[0]['family']) < required):
            study['decisions'].append(dict(block=key, decision='complete_block_budget_deferred',
                estimated_seconds=required, remaining_seconds=remaining(study)))
            atomic_json(Path(study['root']) / 'study.json', study)
            continue
        for item in group:
            if not run_job(study, item):
                break


def remaining(study):
    budget = read(Path(study['root']) / 'budget.json', {})
    return study['budget_seconds'] - budget.get('charged_seconds', 0.)


def run_job(study, item):
    root = Path(study['root'])
    ledger = read(root / 'launches' / item['id'] / 'launch.json', {})
    if ledger.get('status') == 'completed':
        # The suite still validates the complete source/config/image fingerprint.
        pass
    elif ledger:
        study['decisions'].append(dict(job=item['id'], decision='incomplete_attempt_requires_new_block',
                                       prior_status=ledger['status']))
        atomic_json(root / 'study.json', study)
        return False
    elif family_available(study, item['family']) <= 90:
        study['decisions'].append(dict(job=item['id'], decision='family_budget_deferred'))
        atomic_json(root / 'study.json', study)
        return False
    elif remaining(study) < item.get('minimum_budget_seconds', 120):
        return False
    bundle = {key: study[key] for key in ('root', 'source', 'source_digest', 'image', 'budget_seconds')}
    bundle.update(strict_resume=True, jobs=[item])
    path = root / 'jobs' / (item['id'] + '.json')
    atomic_json(path, bundle)
    study.update(status='running', active_job=item['id'], updated_unix=time.time())
    atomic_json(root / 'study.json', study)
    print(json.dumps(dict(job=item['id'], family=item['family'], remaining_seconds=remaining(study))), flush=True)
    command = [sys.executable, str(Path(study['source']) / 'docker/run-workflow-suite.py'), '--jobs', str(path)]
    completed = subprocess.run(command, check=False)
    ledger = read(root / 'launches' / item['id'] / 'launch.json', {})
    study.update(active_job=None, updated_unix=time.time())
    atomic_json(root / 'study.json', study)
    preview(study)
    if ledger.get('status') in ('blocked_foreign_gpu_workload', 'interrupted_foreign_gpu_workload',
                              'gpu_monitor_unavailable', 'interrupted', 'cleanup_failed'):
        raise RuntimeError('study stopped: ' + ledger['status'])
    return completed.returncode == 0 and ledger.get('status') == 'completed'


def preview(study, final=False):
    if study.get('status') != 'prepared':
        controller_source(study)
    from ditserve.evaluation.workflow_serving_report import build_report
    build_report(Path(study['root']), final=final)


def archive(study, *, keep_raw=False):
    """Use the installed openpyxl in a CPU-only container, keeping host paths."""
    root = Path(study['root'])
    data = root.parent.parent
    destination = data / 'results/workflow' / root.name
    code = '''import os,json,pathlib
for p in (pathlib.Path('/mnt/ssd0'),pathlib.Path('/mnt/ssd0/zyd')):
    s=p.stat()
    if s.st_uid!=0 or s.st_dev!=pathlib.Path('/').stat().st_dev:
        raise RuntimeError('refusing to chmod a host-mounted parent')
    p.chmod(0o755)
os.setgroups([GID]);os.setgid(GID);os.setuid(UID)
from ditserve.results import load,verify
from ditserve.evaluation.workflow_retention import finalize_workflow
from ditserve.evaluation.workflow_serving_report import generate_quality_gallery
work=pathlib.Path(WORK);destination=pathlib.Path(DEST)
if (work/'measurements/quality/run.json').is_file():
    print(json.dumps(dict(quality_gallery=generate_quality_gallery(work))))
outcome=finalize_workflow(work,destination,keep_raw=KEEP_RAW)
assert outcome.get('status')=='completed',outcome
expected=json.loads((work/'run.json').read_text())
assert load(destination)['run']==expected,'canonical workbook roundtrip mismatch'
print(json.dumps(dict(finalization=outcome,verification=verify(destination))))
'''.replace('GID', str(os.getgid())).replace('UID', str(os.getuid()))
    code = code.replace('WORK', repr(str(root))).replace('DEST', repr(str(destination))).replace('KEEP_RAW', repr(keep_raw))
    report_source = controller_source(study)
    command = ['docker', '--host', 'unix:///run/ditserve-docker/docker.sock', 'run', '--rm',
               '--runtime=runc', '--network=none', '--cpuset-cpus=58', '--user=0:0',
               '-e', 'PYTHONPATH=/workspace/workers',
               '-v', report_source + ':/workspace:ro', '-v', str(data) + ':' + str(data),
               study['image'], '/opt/venv/bin/python', '-c', code]
    with (root / 'archive.stdout.log').open('a') as stdout, (root / 'archive.stderr.log').open('a') as stderr:
        outcome = subprocess.run(['sg', 'ditserve-docker', '-c', shlex.join(command)],
                                 stdout=stdout, stderr=stderr, check=False, timeout=600)
    if outcome.returncode:
        raise RuntimeError('archive verification failed; see archive.stderr.log and .finalization.json')
    print(json.dumps(dict(status='archive_verified', destination=str(destination))), flush=True)


def micro_alternative(study):
    values = {}
    for cores in (2, 4, 8, 16):
        value = result(study, f'cpu-micro-{cores}')
        if not value:
            continue
        samples = value.get('samples', value.get('requests', []))
        durations = [row.get('encode_ms', row.get('wall_ms')) for row in samples
                     if not row.get('warmup', False)]
        durations = [float(value) for value in durations if isinstance(value, (float, int))]
        if durations:
            values[cores] = statistics.median(durations)
    if 8 not in values or len(values) < 2:
        return None
    return min((c for c in values if c != 8), key=lambda c: (-abs(values[c] - values[8]), c))


def execute(study, retry_block=None, *, keep_raw=False):
    root = Path(study['root'])
    lock = (root / 'driver.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        controller_source(study)
        import importlib.util
        helper_path = Path(study['source']) / 'docker/recover-serving-study.py'
        recovery_spec = importlib.util.spec_from_file_location('serving_recovery', helper_path)
        recovery = importlib.util.module_from_spec(recovery_spec)
        recovery_spec.loader.exec_module(recovery)
        recovered = recovery.recover_study(root, caller_pid=os.getpid(), driver_lock_held=True)
        if recovered.get('status') == 'blocked':
            raise RuntimeError('prior attempt cannot be safely recovered: ' + json.dumps(recovered))
        if recovered.get('status') == 'recovered' or study.get('active_job'):
            study.update(active_job=None, status='interrupted')
            atomic_json(root / 'study.json', study)
        if retry_block:
            study_api.retry_block(study, retry_block)
        study['decisions'].append(dict(decision='controller_started', pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
            measurement_source_digest=study['source_digest'],
            controller_source_digest=CONTROLLER_REVISION['source_digest'] if CONTROLLER_REVISION else study['source_digest'],
            charged_seconds=study['budget_seconds'] - remaining(study), unix=time.time()))
        by_id = lambda identifier: next(row for row in reversed(study['jobs']) if row.get('logical_id', row['id']) == identifier and not row.get('superseded'))
        for placement in ('dual_cpu', 'dedicated_fp32'):
            if not run_job(study, by_id('smoke-' + placement)):
                raise RuntimeError('real-model smoke test failed: ' + placement)
        if not run_job(study, by_id('calibration')):
            raise RuntimeError('valid calibration is required before online comparisons')
        calibrated = study_api.calibration_from_result(result(study, 'calibration'))
        study_api.online_jobs(study, calibrated)
        run_job(study, by_id('quality'))
        for family in ('placement', 'online', 'cpu', 'overlap', 'sp'):
            jobs = [row for row in study['jobs'] if row['family'] == family and not row.get('superseded')]
            run_blocks(study, jobs)
            if family == 'cpu':
                alternative = micro_alternative(study)
                if alternative is not None:
                    study_api.prefetch_jobs(study, alternative)
                    run_blocks(study, [row for row in study['jobs'] if row['id'].startswith('prefetch-') and not row.get('superseded')])
        overlap = result(study, 'overlap')
        if overlap and overlap.get('qualification', {}).get('passed'):
            run_blocks(study, study_api.trace_jobs(study))
        else:
            study['decisions'].append(dict(job='trace-overlap', decision='normal_overlap_not_qualified'))
        if list((root / 'traces').glob('*.nsys-rep')):
            command = ['taskset', '-c', '58', sys.executable,
                       str(Path(controller_source(study)) / 'docker/postprocess-workflow-study.py'),
                       '--root', str(root), '--kind', 'nsys']
            subprocess.run(command, check=True)
        # A separate, deterministic selector only admits fresh confirmation traces.
        from ditserve.evaluation.workflow_serving_report import select_extension
        extension = select_extension(root)
        if extension:
            from ditserve.evaluation.workflow_serving_study import extension_jobs
            extension_jobs(study, calibrated, extension)
            run_blocks(study, [row for row in study['jobs'] if row['family'] == 'extension' and not row.get('profiling') and not row.get('superseded')])
        study.update(status='settled', active_job=None, finished_unix=time.time())
        atomic_json(root / 'study.json', study)
        preview(study, final=True)
    except BaseException as error:
        study.update(status='interrupted', active_job=None, error=str(error), updated_unix=time.time())
        atomic_json(root / 'study.json', study)
        preview(study)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    # Retention takes the driver lock itself so it can verify the study is idle.
    archive(study, keep_raw=keep_raw)


def main():
    global CONTROLLER_REVISION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='run02')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--freeze-analysis', action='store_true',
                        help='Version current controller/report code; keep measurement source unchanged')
    parser.add_argument('--image', default='ditserve:rewrite-runtime')
    parser.add_argument('--retry-block', help='Retry all arms of this settled block into fresh attempt directories')
    parser.add_argument('--keep-raw', action='store_true', help='Archive without pruning redundant raw artifacts')
    args = parser.parse_args()
    if not args.run_id.replace('-', '').replace('_', '').isalnum():
        parser.error('unsafe run ID')
    root = args.root or study_api.DEFAULT_ROOT / args.run_id
    if args.prepare:
        command = ['docker', '--host', 'unix:///run/ditserve-docker/docker.sock',
                   'image', 'inspect', args.image, '--format', '{{.Id}}']
        image = subprocess.check_output(['sg', 'ditserve-docker', '-c', shlex.join(command)], text=True).strip()
        if not image.startswith('sha256:') or len(image) != 71:
            raise ValueError('runtime image did not resolve to an immutable digest')
        study = study_api.freeze(REPO, root, image)
        study_api.initial_jobs(study)
        preview(study)
        print(json.dumps(dict(status='prepared', root=str(root), jobs=len(study['jobs']))), flush=True)
    else:
        study = read(root / 'study.json')
        if not study:
            parser.error('prepare this study before running it')
    if args.freeze_analysis:
        revision = study_api.freeze_analysis(REPO, study)
        print(json.dumps(dict(status='analysis_revision_frozen', source=revision['source'],
                              source_digest=revision['source_digest'])), flush=True)
    if args.run or args.preview:
        revision = read(root / 'analysis-source.json')
        if revision:
            verify_revision(revision)
            frozen_driver = Path(revision['source']) / 'docker/run-serving-study.py'
        else:
            frozen_driver = Path(study['source']) / 'docker/run-serving-study.py'
        CONTROLLER_REVISION = revision
        if Path(__file__).resolve() != frozen_driver.resolve():
            command = [sys.executable, str(frozen_driver), '--root', str(root),
                       '--run' if args.run else '--preview']
            if args.retry_block:
                command += ['--retry-block', args.retry_block]
            if args.keep_raw:
                command += ['--keep-raw']
            os.execv(sys.executable, command)
        if args.run:
            execute(study, args.retry_block, keep_raw=args.keep_raw)
        else:
            preview(study)


if __name__ == '__main__':
    main()
