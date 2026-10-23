"""Two-GPU lifecycle diagnostics under one outer suite owner; never performance evidence."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

from . import workflow_overlap_isolated as isolated
from .workflow_budget import atomic_json

SCHEMA = 'ditserve.workflow.overlap-bundle.v1'
LAYOUT = (('a17-b33', 0, [17, 33]), ('a33-b17', 1, [33, 17]),
          ('a33-b33', 0, [33, 33]), ('a17-b17', 1, [17, 17]))
WAVES = [[row[0] for row in LAYOUT[:2]], [row[0] for row in LAYOUT[2:]]]
MASKS = {0: list(range(8)), 1: list(range(60, 68))}
SHA = re.compile('[0-9a-f]{64}')


class CleanupError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def normalize_config(config):
    if not isinstance(config, dict) or set(config) - {'cases', 'waves', 'case_timeout_seconds', 'bundle_timeout_seconds'}:
        raise ValueError('unknown bundle configuration fields')
    value = dict(waves=WAVES, case_timeout_seconds=2700, bundle_timeout_seconds=6000)
    value.update(config)
    if (value['waves'] != WAVES or type(value['case_timeout_seconds']) is not int
            or value['case_timeout_seconds'] != 2700 or type(value['bundle_timeout_seconds']) is not int
            or value['bundle_timeout_seconds'] != 6000 or not isinstance(value.get('cases'), list)
            or len(value['cases']) != 4):
        raise ValueError('bundle requires the fixed two waves and finite 2700/6000-second deadlines')
    uuids, cases = {}, []
    for case, (name, ordinal, pair) in zip(value['cases'], LAYOUT):
        if (not isinstance(case, dict) or set(case) != {'id', 'device_ordinal', 'gpu_uuid', 'config'}
                or case['id'] != name or type(case['device_ordinal']) is not int or case['device_ordinal'] != ordinal):
            raise ValueError('cases must retain the four ordered frame/device assignments')
        child = isolated.normalize_config(case['config'])
        if (child['device'] != 'cuda:0' or child['gpu_uuid'] != case['gpu_uuid']
                or child['cpu_threads'] != 8 or child['cpu_masks'] != {'cuda:0': MASKS[ordinal]}
                or child['frame_pairs'] != [pair] or child['warmup_pairs'] != 2 or child['lifecycle_pairs'] != 3
                or child['height'] != 480 or child['width'] != 832 or child['steps'] != 20
                or child['guidance_scale'] != 5.0 or not 1 <= child['arm_timeout_seconds'] <= 900
                or child['save_videos'] or 'selected_arm' in child):
            raise ValueError('case differs from its UUID/CPU/shape/three-arm lifecycle controls')
        if ordinal in uuids and uuids[ordinal] != case['gpu_uuid']:
            raise ValueError('physical GPU UUID changed between waves')
        uuids[ordinal] = case['gpu_uuid']
        cases.append(dict(case, config=child))
    if len(set(uuids.values())) != 2:
        raise ValueError('the two physical GPUs must have distinct UUIDs')
    value['cases'] = cases
    return json.loads(json.dumps(value, allow_nan=False))


def _identity(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), pgid=int(fields[2]), sid=int(fields[3]))


def _group_members(pgid):
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ('Z', 'X', 'x'):
                members.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return members


def _stop(process):
    """Only the session created for this case, including its arm grandchildren."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10 if sig == signal.SIGTERM else 5)
        except subprocess.TimeoutExpired:
            pass
        if process.poll() is not None and not _group_members(process.pid):
            return True
    return process.poll() is not None and not _group_members(process.pid)


def _start(case, directory):
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'config.json'
    atomic_json(path, case['config'])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=case['gpu_uuid'],
                       OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    command = ['taskset', '-c', ','.join(map(str, MASKS[case['device_ordinal']])),
               sys.executable, '-m', 'ditserve.evaluation.workflow_overlap_isolated',
               '--config', str(path), '--out', str(directory)]
    started, monotonic = time.time_ns(), time.monotonic()
    with (directory/'stdout.log').open('x') as stdout, (directory/'stderr.log').open('x') as stderr:
        process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
    try:
        identity = _identity(process.pid)
        if identity['pgid'] != process.pid or identity['sid'] != process.pid:
            raise RuntimeError('case must own its fresh process group and session')
    except BaseException:
        if not _stop(process):
            raise CleanupError('new case process could not be cleaned up')
        raise
    launch = dict(**identity, command=command, started_unix_ns=started,
        started_monotonic=monotonic, configuration_path=str(path), configuration_bytes=path.stat().st_size,
        configuration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        configuration_payload_sha256=digest(case['config']), gpu_uuid=case['gpu_uuid'],
        device_ordinal=case['device_ordinal'], cpu_mask=MASKS[case['device_ordinal']],
        timeout_seconds=2700, container_owner_pid=os.getpid(), reaped=False, timed_out=False)
    atomic_json(directory/'case-launch.json', launch)
    return process, launch


def _finish(process, launch, directory, timed_out):
    launch['timed_out'] = timed_out
    if timed_out:
        clean = _stop(process)
    else:
        process.wait(timeout=5)
        clean = not _group_members(process.pid)
        if not clean:
            launch['unexpected_live_descendants'] = True
            clean = _stop(process)
    launch.update(returncode=process.returncode, reaped=process.poll() is not None,
                  cleanup_verified=clean, finished_unix_ns=time.time_ns(), finished_monotonic=time.monotonic())
    path = directory/'run.json'
    child = {}
    if path.is_file():
        payload = path.read_bytes()
        launch.update(result_path=str(path), result_bytes=len(payload), result_sha256=hashlib.sha256(payload).hexdigest())
        try:
            child = json.loads(payload)
            launch['result_payload_sha256'] = digest(child)
            launch['required_arm_evidence'] = {}
            for arm in child.get('arms', []):
                if arm['id'] not in isolated.REQUIRED_ARMS:
                    continue
                raw_path = Path(arm['launch']['result_path'])
                _require(raw_path.resolve().is_relative_to(directory.resolve()), 'arm result escapes its case')
                raw_bytes = raw_path.read_bytes()
                raw_sha = hashlib.sha256(raw_bytes).hexdigest()
                _require(raw_sha == arm['launch']['result_sha256'], 'raw arm SHA differs from child launch')
                raw = json.loads(raw_bytes)
                _require(isolated._compact_child(raw) == arm['result'], 'raw arm differs from compact evidence')
                launch['required_arm_evidence'][arm['id']] = dict(path=str(raw_path), bytes=len(raw_bytes),
                    sha256=raw_sha, payload_sha256=digest(raw), result=raw)
        except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
            launch['result_error'] = str(error)
    else:
        launch['result_error'] = 'missing child result'
    atomic_json(directory/'case-launch.json', launch)
    return child


def _required_arm_replay(arm, evidence, case):
    """Recheck raw lifecycle evidence rather than trusting an arm's passed flags."""
    raw, config = evidence['result'], case['config']
    spec = next(row for row in isolated.ARMS if row['id'] == arm['id'])
    expected_config = isolated.child_config(config, spec, config['frame_pairs'][0])
    _require(evidence['sha256'] == arm['launch']['result_sha256'] and SHA.fullmatch(evidence['sha256'])
             and evidence['bytes'] > 0 and evidence['path'] == arm['launch']['result_path'], 'raw arm binding differs')
    _require(digest(raw) == evidence['payload_sha256'], 'raw arm payload hash differs')
    _require(raw['schema'] == isolated.SCHEMA and raw['configuration'] == expected_config,
             'raw arm configuration differs')
    _require(raw['process']['pid'] == arm['launch']['pid'] and raw['process']['start_ticks'] > 0,
             'raw arm process differs')
    _require(raw['performance_evidence'] is False and not raw.get('error') and not raw.get('stop_reason'),
             'raw arm failed or claimed performance')
    _require(isolated._compact_child(raw) == arm['result'], 'raw arm compact replay differs')
    recomputed = copy.deepcopy(raw)
    isolated._finish_child_result(recomputed, expected_config)
    _require(isolated._compact_child(recomputed) == arm['result'], 'raw qualification/growth replay differs')
    _require(raw['startup_cache_trims'] == 1, 'startup cache trim count differs')
    stream_ids = raw['stream_ids']
    _require(len(stream_ids) == 2 and all(type(x) is int and x >= 0 for x in stream_ids), 'invalid stream identities')
    _require((stream_ids[0] == stream_ids[1]) == (spec['mode'] == 'serial'), 'stream mode differs')
    references = {(row['role'], row['frames']): row for row in raw['references']}
    expected_refs = {('A', config['frame_pairs'][0][0]), ('B', config['frame_pairs'][0][1])}
    _require(len(raw['references']) == 2 and set(references) == expected_refs, 'reference role/frame differs')
    outputs = {role.lower()+'_'+kind: references[(role, frames)][kind]
               for role, frames in expected_refs for kind in ('latent', 'video')}
    rounds = raw['rounds']
    _require(len(rounds) == 5, 'missing raw lifecycle rounds')
    boundaries = [raw['baseline']] + [row['after_release'] for row in rounds]
    for boundary in boundaries:
        _require(isolated.boundary_stop_reason(boundary, config) is None, 'raw release/NVML boundary failed')
        sample = boundary['nvml']
        _require(sample['uuid'] == case['gpu_uuid'], 'boundary GPU UUID differs')
        _require(all(type(sample[k]) is int and sample[k] >= 0 for k in
                     ('memory_used_bytes', 'memory_free_bytes', 'memory_total_bytes'))
                 and sample['memory_total_bytes'] > 0
                 and sample['memory_used_bytes'] <= sample['memory_total_bytes']
                 and sample['memory_free_bytes'] <= sample['memory_total_bytes'], 'invalid boundary memory values')
    previous_end = None
    for index, row in enumerate(rounds):
        _require(row['index'] == index and row['phase'] == ('warmup' if index < 2 else 'measured')
                 and row['measured_index'] == (None if index < 2 else index-2), 'raw lifecycle indices differ')
        measured = row['measurement']
        _require(row['stream_ids'] == stream_ids, 'stream pool changed')
        _require(isolated._hashes(measured) == outputs, 'raw output differs from independent references')
        _require(measured['mode'] == spec['mode'] and measured['repeat'] == index
                 and measured['purpose'] == ('lifecycle_warmup' if index < 2 else 'lifecycle'), 'raw arm execution differs')
        _require(measured['a_decode_stream'] == ('b_stream' if spec['decode_a_on_b_stream'] else 'current_stream'),
                 'raw decode stream differs')
        _require(measured['performance_evidence'] is False, 'lifecycle measurement claimed performance')
        start, end = measured['started_monotonic_ns'], measured['finished_monotonic_ns']
        _require(type(start) is int and type(end) is int and end > start, 'invalid hot window')
        _require(previous_end is None or start > previous_end, 'raw lifecycle windows reused or overlapped')
        _require(measured['sample_interval_seconds'] == config['sample_interval_seconds'], 'sample interval differs')
        previous_end = end
        samples = measured['nvml_samples']
        _require(samples and all(sample['uuid'] == case['gpu_uuid'] for sample in samples), 'hot sample GPU differs')
        gate = isolated.overlap.memory_margin_gate(samples, measured['nvml']['errors'], start, end)
        _require(gate == measured['memory_margin_gate'] and gate['eligible'] is True, 'raw hot capacity gate failed')


def _case_audit(row, case, owner_pid):
    errors, required = [], dict(passed=False)
    try:
        launch, child = row['launch'], row['result']
        _require(launch['container_owner_pid'] == owner_pid, 'outer owner differs')
        _require(row['id'] == case['id'] and row['configuration'] == case['config'], 'case configuration changed')
        _require(launch['gpu_uuid'] == case['gpu_uuid'] and launch['device_ordinal'] == case['device_ordinal'], 'GPU binding changed')
        _require(launch['cpu_mask'] == MASKS[case['device_ordinal']], 'CPU binding changed')
        _require(launch['configuration_payload_sha256'] == digest(case['config']), 'config payload hash differs')
        _require(launch['result_payload_sha256'] == digest(child), 'result payload hash differs')
        _require(all(type(launch[k]) is str and SHA.fullmatch(launch[k]) for k in ('configuration_sha256', 'result_sha256')), 'missing file hashes')
        _require(launch['configuration_bytes'] > 0 and launch['result_bytes'] > 0, 'missing file sizes')
        _require(launch['pid'] == launch['pgid'] == launch['sid'] and launch['start_ticks'] > 0, 'case ownership differs')
        _require(launch['reaped'] is True and launch['cleanup_verified'] is True, 'case process not reaped/clean')
        _require(launch['returncode'] == 0 and launch['timed_out'] is False and not launch.get('unexpected_live_descendants'), 'case launch failed')
        _require(launch['finished_monotonic'] >= launch['started_monotonic'], 'invalid wait timing')
        _require(child['schema'] == isolated.SCHEMA and child['configuration'] == case['config'], 'isolated child config/schema differs')
        _require(child['performance_evidence'] is False, 'lifecycle cannot claim performance')
        arms = child['arms']
        _require([a['id'] for a in arms] == [a['id'] for a in isolated.ARMS], 'missing/reordered causal arms')
        _require(all(a['frame_pair'] == case['config']['frame_pairs'][0] for a in arms), 'frame pair differs')
        required = isolated.aggregate(arms, case['config'])
        _require(child['qualification'] == required, 'child qualification does not replay')
        _require(child['status'] == ('completed' if required['passed'] else 'qualification_failed'), 'child status contradicts qualification')
        for arm in arms:
            if arm['id'] not in isolated.REQUIRED_ARMS:
                continue
            _required_arm_replay(arm, launch['required_arm_evidence'][arm['id']], case)
            result = arm['result']
            _require(result['cpu_affinity']['bound_cpus'] == MASKS[case['device_ordinal']], 'actual CPU mask differs')
            _require(result['completed_warmup_pairs'] == 2 and result['completed_measured_pairs'] == 3, 'incomplete warmup/lifecycle')
            hashes = result['output_hashes']
            _require(len(hashes) == 5, 'missing required outputs')
            for outputs in hashes:
                _require(set(outputs) == set(isolated.OUTPUTS), 'output fields differ')
                _require(all(x.get('finite') is True and isinstance(x.get('sha256'), str) and SHA.fullmatch(x['sha256'])
                           and x.get('shape') and x.get('dtype') for x in outputs.values()), 'invalid output hash/finite metadata')
    except (AssertionError, KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        errors.append(str(error))
    return dict(passed=not errors and required['passed'] is True, structure_valid=not errors,
                errors=errors, required_scope=required, performance_evidence=False,
                diagnostic_current_decode_retained=True)


def validate_result(result, config=None):
    checked = dict(passed=False, structure_valid=False, errors=[], cases={}, performance_evidence=False)
    try:
        config = normalize_config(result['configuration'] if config is None else config)
        _require(result['schema'] == SCHEMA and result['configuration'] == config, 'bundle configuration/schema differs')
        _require(result['configuration_sha256'] == digest(config), 'bundle config digest differs')
        _require(result['performance_evidence'] is False, 'bundle timing is not performance evidence')
        _require(result['status'] in ('completed', 'qualification_failed'), 'bundle did not settle')
        rows = result['cases']
        _require([r['id'] for r in rows] == [c['id'] for c in config['cases']], 'missing/reordered cases')
        _require(len(result['waves']) == 2, 'missing wave')
        for index, wave in enumerate(result['waves']):
            _require(wave['case_ids'] == WAVES[index] and wave['finished_monotonic'] >= wave['started_monotonic'], 'wave differs')
            if index:
                _require(wave['started_monotonic'] >= result['waves'][index-1]['finished_monotonic'], 'waves overlapped')
        for row, case in zip(rows, config['cases']):
            checked['cases'][case['id']] = _case_audit(row, case, result['process']['pid'])
        identities = [r.get('launch', {}).get('pid') for r in rows if type(r.get('launch', {}).get('pid')) is int]
        _require(len(identities) == len(set(identities)), 'case parent identity reused')
        for index, wave in enumerate(result['waves']):
            for row in rows[index*2:index*2+2]:
                launch = row['launch']
                case_audit = checked['cases'][row['id']]
                if case_audit['structure_valid'] and not (
                        wave['started_monotonic'] <= launch['started_monotonic'] <= launch['finished_monotonic'] <= wave['finished_monotonic']):
                    case_audit.update(passed=False, structure_valid=False)
                    case_audit['errors'].append('case wait lies outside its wave')
        checked['structure_valid'] = True
        checked['passed'] = all(r['passed'] for r in checked['cases'].values())
    except (AssertionError, KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        checked['errors'].append(str(error))
    return checked


def run_bundle(config, out):
    config, out = normalize_config(config), Path(out)
    if 'torch' in sys.modules:
        raise RuntimeError('bundle parent must not import torch or initialize CUDA')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'run.json').exists():
        raise FileExistsError(out/'run.json')
    result = dict(schema=SCHEMA, configuration=config, configuration_sha256=digest(config),
        status='running', process=_identity(os.getpid()), cases=[], waves=[], performance_evidence=False,
        started_unix_ns=time.time_ns(), budget_owner='outer guarded suite; one shared attempt for both GPUs',
        scope='two independent GPU lifecycle diagnostics per wave; serial/shared subset only; no performance claim')
    atomic_json(out/'run.json', result)
    pending = []
    deadline = time.monotonic() + config['bundle_timeout_seconds']
    try:
        for names in WAVES:
            wave = dict(case_ids=names, started_monotonic=time.monotonic())
            result['waves'].append(wave)
            if time.monotonic() >= deadline:
                raise RuntimeError('bundle deadline reached before next wave')
            for case in [c for c in config['cases'] if c['id'] in names]:
                row = dict(id=case['id'], configuration=case['config'], launch={}, result={})
                result['cases'].append(row)
                try:
                    process, row['launch'] = _start(case, out/case['id'])
                    pending.append((process, row, out/case['id']))
                except CleanupError:
                    raise
                except Exception:
                    row['launch_error'] = traceback.format_exc()
                atomic_json(out/'run.json', result)
            while pending:
                for process, row, directory in list(pending):
                    expired = time.monotonic() >= min(deadline, row['launch']['started_monotonic'] + config['case_timeout_seconds'])
                    if process.poll() is None and not expired:
                        continue
                    row['result'] = _finish(process, row['launch'], directory, expired)
                    pending.remove((process, row, directory))
                    if not row['launch']['cleanup_verified']:
                        raise RuntimeError('case descendants remain; refusing next wave')
                    atomic_json(out/'run.json', result)
                if pending:
                    time.sleep(.2)
            wave['finished_monotonic'] = time.monotonic()
        result['status'] = 'completed'
        result['qualification'] = validate_result(result, config)
        for row in result['cases']:
            row['required_scope'] = result['qualification']['cases'].get(row['id'], {}).get('required_scope', dict(passed=False))
        if not result['qualification']['passed']:
            result['status'] = 'qualification_failed'
    except BaseException:
        result.update(status='failed', error=traceback.format_exc())
    finally:
        for process, row, directory in pending:
            try:
                row['result'] = _finish(process, row['launch'], directory, True)
            except BaseException:
                row['cleanup_error'] = traceback.format_exc()
        result['finished_unix_ns'] = time.time_ns()
        atomic_json(out/'run.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_bundle(json.loads(args.config.read_text()), args.out)
    print(json.dumps(dict(status=result['status'], manifest=str(args.out/'run.json'))))
    return int(result['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
