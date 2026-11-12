#!/usr/bin/env python3
"""Read hash-bound isolated ABBA runs; never promote them to formal P1.

Manifest schema ditserve.p1-isolated-review-inputs.v1 binds protocol, pair_plan
and cases [{pair_id, inputs, config, run, attempt}]. File references are
{path, sha256}; run may be null. Missing formal pairs remain in the four-pair
denominator. Inputs is the existing isolated compiler's complete manifest.
Only complete exact, zero-stop four-cell seeds contribute paired timing.
No GPU execution, downloads, raw rewriting or dependency closure occurs here.
"""

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

import formal_benchmark as formal
import p1_isolated_batch as isolated
from native_exact import checked_digest, distribution, stop_samples, validate_pair_identity
from p1_benchmark import config_identity, match_native_config, source_file, validate_plan
from p1_qualification import finite, seed_interval
from p2_benefit import expanded, sha
from ditserve.evaluation.common import canonical_hash, seal


def _cost_number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid controller timestamp/duration')
    return value


def _cost_duration(start, finish):
    value = _cost_number(finish) - _cost_number(start)
    if value < -1e-6:
        raise ValueError('controller interval is reversed')
    return max(0, value)


def _cost_dependencies(event):
    stages = event.get('stages', [])
    if 'transfer_targets' in event and (type(event['transfer_targets']) is not int or event['transfer_targets'] < 0):
        raise ValueError('invalid transfer target count')
    by_name = defaultdict(list)
    members = {event['request_id']}
    if event.get('pair_request_id'):
        members.add(event['pair_request_id'])
    for stage in stages:
        by_name[stage['stage']].append(stage)
        if stage.get('request_id') is not None and stage['request_id'] not in members:
            raise ValueError('stage belongs to a different task request')
        if event['status'] == 'completed' and stage['status'] == 'failed':
            raise ValueError('completed task contains a failed stage')

    def parent_status(parent, children):
        if not children:
            raise ValueError('fanout has no corresponding rank RPCs')
        for child in children:
            if child['start_ms'] < parent['start_ms'] - 1e-6 or child['finish_ms'] > parent['finish_ms'] + 1e-6:
                raise ValueError('rank RPC lies outside its fanout interval')
        expected = 'failed' if any(child['status'] == 'failed' for child in children) else 'completed'
        if parent['status'] != expected:
            raise ValueError('fanout status differs from its rank RPCs')

    parents = {}
    for name in ('export', 'imports_all_ready'):
        parents[name] = {}
        for stage in by_name[name]:
            rid = stage.get('request_id', event['request_id'])
            if rid in parents[name]:
                raise ValueError('ambiguous repeated transfer for one task request')
            parents[name][rid] = stage
    imports = defaultdict(list)
    for stage in by_name['import']:
        imports[stage.get('request_id', event['request_id'])].append(stage)
    for prepared in by_name['prepare']:
        rid = prepared.get('request_id', event['request_id'])
        exported = parents['export'].get(rid)
        followers = [exported] if exported is not None else [stage for stage in by_name['advance']
                    if stage.get('worker') == prepared.get('worker')]
        for follower in followers:
            if prepared['status'] != 'completed':
                raise ValueError('failed preparation followed by execution')
            _cost_duration(prepared['finish_ms'], follower['start_ms'])
    for rid in set(imports) | set(parents['imports_all_ready']):
        exported = parents['export'].get(rid)
        parent, children = parents['imports_all_ready'].get(rid), imports[rid]
        if exported is None or exported['status'] != 'completed':
            raise ValueError('imports have no completed source export')
        if len({child.get('worker') for child in children}) != len(children) or any(not child.get('worker') for child in children):
            raise ValueError('import targets must have distinct worker identities')
        for child in children:
            _cost_duration(exported['finish_ms'], child['start_ms'])
        if parent is None:
            if event['status'] != 'failed':
                raise ValueError('completed imports lack an all-ready interval')
            continue
        _cost_duration(exported['finish_ms'], parent['start_ms'])
        parent_status(parent, children)
        if rid == event['request_id'] and 'transfer_targets' in event and len(children) != event['transfer_targets']:
            raise ValueError('import target count differs from planned fanout')
    if by_name['advance']:
        for rid, exported in parents['export'].items():
            parent = parents['imports_all_ready'].get(rid)
            if exported['status'] != 'completed' or parent is None or parent['status'] != 'completed':
                raise ValueError('advance precedes successful transfer readiness')
            for advance in by_name['advance']:
                _cost_duration(parent['finish_ms'], advance['start_ms'])
    for name in ('advances_all_done', 'prepare_and_advances_all_done'):
        if len(by_name[name]) > 1:
            raise ValueError('duplicate advance fanout')
        if by_name[name]:
            children = by_name['advance'] + (by_name['prepare'] if name.startswith('prepare_') else [])
            parent_status(by_name[name][0], children)
            workers = [child.get('worker') for child in children]
            if any(not worker for worker in workers) or (name == 'advances_all_done' and len(set(workers)) != len(workers)):
                raise ValueError('advance fanout has missing or duplicate rank identities')
            if 'group_after' in event and len(set(workers)) != len(event['group_after']):
                raise ValueError('advance fanout does not cover its complete group')
    if by_name['advance'] and not (by_name['advances_all_done'] or by_name['prepare_and_advances_all_done']) and event['status'] != 'failed':
        raise ValueError('completed advance lacks its fanout interval')


def _validate_cost_links(events, stops, identities):
    for sid, stop in stops.items():
        previous = events.get(stop.get('task_event_id'))
        if (previous is None or 'start_ms' not in previous or previous['status'] != 'completed'
                or identities[previous['event_id']] != stop['request_identity']
                or sid != previous['event_id'] + '/stop'):
            raise ValueError('stop does not bind its stopped task')
        _cost_duration(previous['start_ms'], stop['ring_ms'])
        _cost_duration(stop['ring_ms'], previous['all_done_ms'])
        _cost_duration(previous['all_done_ms'], stop['observed_ms'])
        _cost_duration(stop['observed_ms'], previous['observed_ms'])
        if 'group' in stop and stop['group'] != previous.get('group_after'):
            raise ValueError('stop group differs from stopped task')
    used_stops, used_restores = set(), set()
    for eid, event in events.items():
        link = event.get('resume_from_stop_event_id')
        if link:
            stop = stops.get(link)
            if ('start_ms' not in event or stop is None or stop['request_identity'] != identities[eid]
                    or link in used_stops):
                raise ValueError('resume has no unique same-request stop evidence')
            used_stops.add(link)
            _cost_duration(stop['observed_ms'], event['start_ms'])
            following = [other for other in events.values() if 'start_ms' in other
                         and identities[other['event_id']] == identities[eid]
                         and other['start_ms'] >= stop['observed_ms']]
            if min(following, key=lambda other: other['start_ms'])['event_id'] != eid:
                raise ValueError('stop link skips an earlier request dispatch')
        link = event.get('restore_event_id')
        if link:
            restored = events.get(link)
            if ('start_ms' not in event or restored is None or identities[link] != identities[eid]
                    or restored['transition'] != 'tiered_restore' or restored['status'] != 'completed'
                    or link in used_restores):
                raise ValueError('resume has no unique same-request completed restore')
            used_restores.add(link)
            _cost_duration(restored['observed_ms'], event['start_ms'])


def _summarize_controller_costs(record):
    if record.get('schema') != 'ditserve.run.v1' or not record.get('run_id'):
        raise ValueError('expected a native run identity')
    source = record.get('source', {})
    if not sha(source.get('core_sha256')) or not source.get('files') or not all(sha(v) for v in source['files'].values()):
        raise ValueError('missing native/Python source hashes')
    native = source.get('native_files')
    if native is not None and (not isinstance(native, dict) or not all(sha(v) for v in native.values())):
        raise ValueError('invalid native source inventory')
    source = dict(core_sha256=source['core_sha256'], python_sources_sha256=canonical_hash(source['files']),
                  native_sources_sha256=canonical_hash(native) if native else None)
    rows = record.get('requests', [])
    events, stops, identities = {}, {}, {}
    for row in rows:
        identity = tuple(row.get(k) for k in ('phase', 'attempt', 'cell', 'order', 'request_id'))
        for event in row.get('controller_timing', []):
            eid = event['event_id']
            if (not isinstance(eid, str) or not eid or eid in events
                    or event.get('schema') != 'ditserve.controller-timing.v1'
                    or event.get('clock') != 'controller_replay_monotonic_ms'
                    or tuple(event.get(k) for k in ('phase', 'attempt', 'cell', 'order', 'request_id')) != identity):
                raise ValueError('duplicate or inconsistent task timing identity')
            events[eid], identities[eid] = event, identity
        for stop in row.get('preemption_stop_samples', []):
            if 'event_id' not in stop:
                continue
            if stop['event_id'] in stops:
                raise ValueError('duplicate stop event')
            stop = dict(stop, request_identity=identity)
            if not math.isclose(_cost_duration(stop['ring_ms'], stop['observed_ms']), _cost_number(stop['stop_ms']), abs_tol=1e-6):
                raise ValueError('stop interval differs from duration')
            stops[stop['event_id']] = stop
    _validate_cost_links(events, stops, identities)
    groups = defaultdict(lambda: defaultdict(lambda: dict(counts=Counter(), wall=defaultdict(list), rank_rpc=defaultdict(list))))
    for eid, event in events.items():
        group = groups[identities[eid][:4]][event['transition']]
        status = event.get('status')
        if status not in ('completed', 'failed'):
            raise ValueError('unsettled task cannot be reported as measured timing')
        group['counts']['events'] += 1
        group['counts']['async_' + status] += 1
        task = 'start_ms' in event
        observed = _cost_number(event['observed_ms'])
        if task:
            if event.get('controller_status') not in ('completed', 'aborted'):
                raise ValueError('missing controller consumption status')
            group['counts']['controller_' + event['controller_status']] += 1
            _cost_duration(event['selected_ms'], event['start_ms'])
            group['wall']['async_launch_delay_ms'].append(_cost_duration(event['start_ms'], event['async_begin_ms']))
            _cost_duration(event['async_begin_ms'], event['all_done_ms'])
            group['wall']['controller_observation_lag_ms'].append(_cost_duration(event['all_done_ms'], observed))
            if status == 'completed' and event['controller_status'] == 'completed':
                group['wall']['task_to_progress_including_compute_ms'].append(_cost_duration(event['start_ms'], event['all_done_ms']))
        elif event['transition'] not in ('tiered_offload', 'tiered_restore'):
            raise ValueError('non-task timing must identify a tiered RPC')
        stages = event.get('stages', [])
        exports, ready = {}, {}
        for stage in stages:
            elapsed = _cost_duration(stage['start_ms'], stage['finish_ms'])
            if not math.isclose(elapsed, _cost_number(stage['elapsed_ms']), abs_tol=1e-6):
                raise ValueError('stage endpoints differ from elapsed duration')
            if stage['status'] not in ('completed', 'failed'):
                raise ValueError('unsettled stage')
            if stage['finish_ms'] > observed + 1e-6 or task and (
                    stage['start_ms'] < event['async_begin_ms'] - 1e-6
                    or stage['finish_ms'] > event['all_done_ms'] + 1e-6):
                raise ValueError('stage lies outside its controller task')
            name = stage['stage']
            group['counts']['stage_' + name + '_' + stage['status']] += 1
            if stage['status'] == 'completed':
                destination = group['rank_rpc'] if 'worker' in stage else group['wall']
                destination[name + '_ms'].append(elapsed)
                if name in ('export', 'imports_all_ready'):
                    rid = stage.get('request_id') or event['request_id']
                    index = exports if name == 'export' else ready
                    if rid in index:
                        raise ValueError('ambiguous repeated transfer for one task request')
                    index[rid] = stage
        _cost_dependencies(event)
        for rid, end in ready.items():
            if rid not in exports:
                raise ValueError('imports-ready has no corresponding source export')
            _cost_duration(exports[rid]['finish_ms'], end['start_ms'])
            group['wall']['export_to_targets_ready_ms'].append(_cost_duration(exports[rid]['start_ms'], end['finish_ms']))
        link = event.get('resume_from_stop_event_id')
        if link:
            stop = stops.get(link)
            if not task or stop is None or stop['request_identity'] != identities[eid]:
                raise ValueError('resume has no same-request stop evidence')
            group['wall']['stopped_to_next_dispatch_ms'].append(_cost_duration(stop['observed_ms'], event['start_ms']))
            if status == 'completed' and event['controller_status'] == 'completed':
                group['wall']['stopped_to_next_progress_including_other_work_ms'].append(_cost_duration(stop['observed_ms'], event['all_done_ms']))
    planning = record.get('controller_planning', [])
    for item in planning:
        for field in ('calls', 'failed', 'total_ms', 'max_ms'):
            _cost_number(item[field])
        if item['failed'] > item['calls'] or item['max_ms'] > item['total_ms'] + 1e-6:
            raise ValueError('inconsistent planner-only aggregate')
    cells = []
    for selector, transitions in groups.items():
        selected = [row for row in rows if tuple(row.get(k) for k in ('phase', 'attempt', 'cell', 'order')) == selector]
        cells.append(dict(zip(('phase', 'attempt', 'cell', 'order'), selector),
            observed_requests=len(selected), request_status_counts=dict(Counter(row.get('status', 'unknown') for row in selected)),
            transitions={name: dict(counts=dict(value['counts']),
                controller_wall_ms={key: distribution(samples) for key, samples in value['wall'].items()},
                per_rank_rpc_ms={key: distribution(samples) for key, samples in value['rank_rpc'].items()}) for name, value in transitions.items()}))
    return dict(run_id=record['run_id'], run_status=record.get('status'), source=source,
        measurement_status='measured_prefix' if events else 'not_measured',
        observed_requests=len(rows), request_status_counts=dict(Counter(row.get('status', 'unknown') for row in rows)),
        requests_with_controller_events=sum(bool(row.get('controller_timing')) for row in rows),
        shared_pair_events=sum(bool(event.get('pair_request_id')) for event in events.values()),
        config_planned_policy_requests_per_attempt=sum(len(cell.get('workload', [])) * (2 if phase.get('paired') else 1)
            for phase in record.get('config', {}).get('phases', []) if phase.get('kind') == 'cluster' for cell in expanded(phase)),
        cells=cells,
        planner_only=planning)

INPUT_SCHEMA = 'ditserve.p1-isolated-review-inputs.v1'
SCHEMA = 'ditserve.p1-isolated-review.v1'
ERRORS = (ValueError, KeyError, TypeError, IndexError, OSError, AttributeError)
EXPECTED_REQUESTS = len(isolated.SEEDS) * 4 * isolated.COUNT
TIMING_METRICS = ('batch_last_completion_ms', 'mean_request_latency_ms')


def integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('invalid integer ' + name)
    return value


def bound_json(base, reference):
    def reject(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(source_file(base, reference), parse_constant=reject)


def empty_case(pair, status, reason):
    cells = [dict(phase=f'seed-{seed}', position=position, cell=cell, order=order,
                  policy=isolated.POLICIES[cell], expected_requests=8,
                  status='unverified', execution_status='unknown', errors=[reason])
             for seed in isolated.SEEDS for position, (cell, order) in enumerate(
                 ((0, 'forward'), (1, 'forward'), (1, 'reverse'), (0, 'reverse')))]
    return dict(pair_id=pair['id'], model=pair['model'], status=status, reason=reason,
        expected_requests=EXPECTED_REQUESTS, expected_cells=12, expected_output_quartets=24,
        coverage=None, cells=cells, seeds=[], intervals={}, timing_eligible_requests=0,
        verified_output_quartets=0, qualification_eligible=False,
        diagnostic_only=True)


def session_for(record, phase, config, inputs):
    sessions = [s for s in record.get('budget', {}).get('sessions', [])
                if s.get('session_id') == phase.get('session_id')]
    if not phase.get('session_id') or len(sessions) != 1:
        raise ValueError('phase lacks a unique allocation session')
    session = sessions[0]
    config_binding = config_identity(record, config)
    match_native_config(config_binding, session.get('identity', {}))
    if 'identity' in phase.get('charge', {}):
        match_native_config(config_binding, phase['charge']['identity'])
    expected = dict(run_id=record['run_id'], attempt=phase['attempt'],
        gpu_uuids=inputs['gpu_uuids'],
        source_sha256=canonical_hash(record['source']))
    if (session.get('phase') != phase['id'] or session.get('devices') != 2
            or any(session.get('identity', {}).get(k) != v for k, v in expected.items())):
        raise ValueError('phase session source/config/run/attempt/two-GPU identity differs')
    if phase.get('status') != 'completed' or session.get('status') != 'completed' or session.get('cleanup_verified') is not True:
        raise ValueError('phase/session is incomplete or cleanup is unverified')


def row_output(row, planned, inputs, config):
    """Check diffusion placement, while permitting single-rank decode/finish."""
    if len(row.get('outputs', [])) != 1 or row.get('forms') != ['cfg2']:
        raise ValueError('completed isolated video needs one exact CFG2 output')
    output = row['outputs'][0]
    if output.get('backend') != 'wan' or output.get('measured') is not True:
        raise ValueError('output is not measured native Wan evidence')
    generation, artifact = output['generation'], output['output']
    identity = validate_pair_identity(row, row)
    if not all(identity.values()):
        raise ValueError('output runtime/weights are unverified')
    model = inputs['pair_id'].split('-')[0]
    spec = planned['spec']
    binding = config['workers']['wan0']['config']
    if (generation.get('model_manifest_sha256') != inputs['model_manifests'][model]['sha256']
            or generation.get('model_path') != binding['model_path']
            or canonical_hash(generation['runtime_identity']) != inputs['identity']['runtime_identity_sha256']
            or any(generation.get(k) != v for k, v in spec.items() if k != 'request_id')
            or any(generation.get(k) != binding[k] for k in ('revision', 'dtype', 'vae_dtype',
                'text_encoder_dtype', 'attention') if k in binding)
            or generation.get('cfg_folded_steps', 0) != 0 or generation.get('cache_enabled', False) is not False
            or row.get('reused_forwards', 0) != 0):
        raise ValueError('executed input/model/runtime or exact mode differs')
    shape = [1, spec['frames'], spec['height'], spec['width'], 3]
    if artifact.get('shape') != shape or artifact.get('dtype') != 'float32':
        raise ValueError('decoded dimensions/dtype differ from the formal shape')
    digest = checked_digest(row)
    progress = output.get('progress', {})
    if progress.get('done') is not True or progress.get('step') != spec['steps'] or progress.get('steps') != spec['steps']:
        raise ValueError('decoded output lacks complete solver progress')
    executions, samples = row.get('execution'), row.get('advance_samples')
    if not executions or not samples:
        raise ValueError('actual diffusion form/group evidence is missing')
    for event in executions:
        applied = event.get('applied', {})
        if (event.get('planned_form') != 'cfg2' or event.get('group') != [0, 1]
                or event.get('planned_cache_threshold') != 0 or event.get('planned_cfg_folded') is not False
                or any(applied.get(k) != v for k, v in dict(cfg=2, degree=2, ring=1,
                    cache_threshold=0., cfg_folded=False, reused_forwards=0).items())):
            raise ValueError('actual diffusion did not execute exact CFG2 on allocated ranks')
    for sample in samples:
        if sample.get('form') != 'cfg2' or sample.get('group') != [0, 1] or sample.get('cfg_folded') is not False:
            raise ValueError('diffusion advance form/group/folding differs')
    # degrees includes the rank-0 finish/decode; it is not a diffusion-only list.
    degrees = row.get('degrees')
    if (not isinstance(degrees, list) or 2 not in degrees
            or any(type(degree) is not int or degree not in (1, 2) for degree in degrees)):
        raise ValueError('execution degrees exceed the two-GPU allocation or lack CFG2')
    if stop_samples([row]) or row.get('preemption_reason') == 'block_preempt':
        raise ValueError('isolated video recorded a preemption stop')
    return digest


def cell_metrics(rows, cell):
    elapsed = finite(cell.get('elapsed_ms'), 'replay elapsed_ms', positive=True)
    latencies, finishes = [], []
    for row in rows:
        arrival, start, finish = (finite(row.get(k), k) for k in ('arrival_ms', 'start_ms', 'finish_ms'))
        if arrival != 0 or start < arrival or finish < start or finish > elapsed + 1e-6:
            raise ValueError('request timestamps lie outside the finite replay cell')
        latencies.append(finish - arrival); finishes.append(finish)
    result = dict(batch_last_completion_ms=max(finishes), mean_request_latency_ms=statistics.fmean(latencies),
                  cell_replay_elapsed_ms=elapsed)
    for field in ('encode_model_ms', 'model_ms'):
        values = [finite(row[field], field) for row in rows if field in row]
        result[field] = dict(measured_requests=len(values), expected_requests=len(rows),
                            mean_ms=statistics.fmean(values) if values else None)
    result['warmup'] = [dict(worker=item.get('worker'), elapsed_ms=finite(item['elapsed_ms'], 'warmup elapsed'))
                        for item in cell.get('warmup', [])]
    return result


def review_case(inputs, config, record, *, attempt=1):
    pair = isolated.validate_sources(inputs)
    planned = isolated.audit_config(config, inputs)
    result = empty_case(pair, 'incomplete', 'missing, failed or unverified seed/cell evidence')
    result['coverage'] = isolated.coverage(config, record, attempt=attempt)
    result['attempt'] = attempt
    if record is None:
        result.update(status='missing', reason='no raw run supplied')
        return result
    if record.get('schema') != 'ditserve.run.v1' or not record.get('run_id'):
        raise ValueError('raw run has no native identity')
    source = record['source']
    if source.get('native_files') is not None and (not isinstance(source['native_files'], dict)
            or not all(isinstance(v, str) and formal.HEX64.fullmatch(v) for v in source['native_files'].values())):
        raise ValueError('invalid native source inventory')
    result['source'] = dict(core_sha256=source['core_sha256'], python_sources_sha256=canonical_hash(source['files']),
        native_sources_sha256=canonical_hash(source['native_files']) if source.get('native_files') else None,
        runtime_identity_sha256=inputs['identity']['runtime_identity_sha256'], gpu_uuids=inputs['gpu_uuids'],
        model_manifests_sha256=canonical_hash(inputs['model_manifests']),
        calibration_bundle_sha256=inputs['bundle']['sha256'], inputs_manifest_sha256=inputs['inputs_manifest_sha256'])
    result['run_id'] = record['run_id']
    result['execution_status'] = record.get('status', 'unknown')
    result['failures'] = copy.deepcopy(record.get('failures', []))
    phases = {p['id']: p for p in record.get('phases', []) if p.get('attempt') == attempt}
    result['cells'] = []
    verified_rows, cells_by_seed = {}, {}
    for plan, covered in zip(planned, result['coverage']['cells']):
        selected = dict(covered, status='unverified', errors=[], zero_stop_status='unknown')
        result['cells'].append(selected)
        seed = int(plan['phase'].removeprefix('seed-'))
        cells_by_seed.setdefault(seed, []).append(selected)
        try:
            phase = phases[plan['phase']]
            cell = phase['cells'][plan['position']]
            rows = [r for r in record['requests'] if r.get('attempt') == attempt
                    and r.get('phase') == plan['phase'] and r.get('cell') == plan['cell']
                    and r.get('order') == plan['order']
                    and r.get('observation_id') == f"{plan['phase']}/{attempt}/{plan['position']}"]
            selected['observed_stop_events'] = len(stop_samples(rows))
            bell = cell.get('doorbell', {})
            selected['doorbell'] = copy.deepcopy(bell)
            if type(bell.get('rings')) is int and bell['rings'] >= 0:
                selected['zero_stop_status'] = ('nonzero' if bell['rings'] or selected['observed_stop_events'] else
                    'zero' if bell.get('enabled') is True else 'unknown')
            if cell.get('status') != 'completed' or cell.get('gpus') != 2:
                raise ValueError('cell is not completed on exactly two GPUs')
            session_for(record, phase, config, inputs)
            if bell.get('enabled') is not True or integer(bell.get('rings'), 'doorbell rings') != 0:
                raise ValueError('zero observed doorbell rings were not established')
            continuous = integer(bell.get('continuous_quanta'), 'continuous quanta')
            if (plan['policy'] == 'block-cfg2' and continuous == 0) or (plan['policy'] == 'edf-cfg2' and continuous != 0):
                raise ValueError('actual whole-step/continuous-block execution differs')
            workload = next(p for p in config['phases'] if p['id'] == plan['phase'])['matrix']['/workload'][0]
            expected = {r['request_id']: r for r in workload}
            for row in rows:
                row_output(row, expected[row['request_id']], inputs, config)
            selected.update(status='verified', metrics=cell_metrics(rows, cell), validated_requests=len(rows), zero_stops=True)
            verified_rows[plan['phase'], plan['position']] = {r['request_id']: r for r in rows}
        except ERRORS as error:
            selected['errors'].append(str(error))
    for seed in isolated.SEEDS:
        cells = cells_by_seed[seed]
        item = dict(seed=seed, expected_cells=4, expected_requests=32, expected_output_quartets=8,
                    verified_output_quartets=0, status='pending', errors=[], comparisons=[], metrics={})
        result['seeds'].append(item)
        if any(c['status'] != 'verified' for c in cells):
            item['errors'].append('all four exact zero-stop cells are required for paired timing')
            continue
        rows = [verified_rows[f'seed-{seed}', position] for position in range(4)]
        for request_id in rows[0]:
            try:
                anchor = rows[0][request_id]
                for other in rows[1:]:
                    candidate = other[request_id]
                    if not all(validate_pair_identity(anchor, candidate).values()) or checked_digest(anchor) != checked_digest(candidate):
                        raise ValueError('ABBA output identity/digest differs')
                item['verified_output_quartets'] += 1
            except ERRORS as error:
                item['errors'].append(request_id + ': ' + str(error))
        if item['verified_output_quartets'] != 8:
            continue
        for order, a, b in (('forward', 0, 1), ('reverse', 3, 2)):
            item['comparisons'].append(dict(order=order, reference_position=a, candidate_position=b,
                deltas={name: cells[b]['metrics'][name] - cells[a]['metrics'][name] for name in TIMING_METRICS}))
        item['metrics'] = {name.removesuffix('_ms') + '_delta_ms': statistics.fmean(row['deltas'][name] for row in item['comparisons'])
                           for name in TIMING_METRICS}
        item['status'] = 'measured_diagnostic'
    settings = inputs['pair_plan']['statistics']
    valid = [seed for seed in result['seeds'] if seed['status'] == 'measured_diagnostic']
    result['allocation_settled'] = ('active' in record.get('budget', {})
                                    and record['budget']['active'] is None)
    complete = (len(valid) == len(isolated.SEEDS) and record.get('status') == 'completed'
                and not result['failures'] and result['allocation_settled'])
    for name in TIMING_METRICS:
        key = name.removesuffix('_ms') + '_delta_ms'
        interval = seed_interval([row['metrics'][key] for row in valid], settings)
        if not complete:
            interval.update(mean=None, ci95=None, status='unknown_incomplete_seed_set_or_run')
        result['intervals'][key] = dict(interval, expected_seeds=3, timing_repeats_per_seed=2,
            scope='descriptive small-sample CI at this one allocation; no hardware generalization',
            sign='candidate minus reference; positive is overhead, negative is lower elapsed time')
    result['timing_eligible_requests'] = len(valid) * 32
    result['verified_output_quartets'] = sum(row['verified_output_quartets'] for row in result['seeds'])
    if complete:
        result.update(status='measured_diagnostic', reason=None)
    elif not result['allocation_settled']:
        result['reason'] = 'allocation settlement is missing or an active session remains; aggregate timing is unknown'
    try:
        result['controller_costs'] = _summarize_controller_costs(record)
    except ERRORS as error:
        result['controller_costs'] = dict(measurement_status='unknown', reason=str(error))
    return result


def evaluate(manifest, base=Path('.')):
    if manifest.get('schema') != INPUT_SCHEMA:
        raise ValueError('unknown isolated review manifest')
    protocol, plan = (bound_json(base, manifest[key]) for key in ('protocol', 'pair_plan'))
    validate_plan(plan, protocol)
    if plan['seeds'] != isolated.SEEDS or len(plan['shape_pairs']) != 4:
        raise ValueError('review requires the complete frozen four-pair, three-seed plan')
    expected = {pair['id']: pair for pair in plan['shape_pairs']}
    cases = {}
    for entry in manifest.get('cases', []):
        key = entry.get('pair_id')
        if key not in expected or key in cases:
            raise ValueError('unplanned or duplicate isolated pair case')
        cases[key] = entry
    results = []
    for key, pair in expected.items():
        entry = cases.get(key)
        if entry is None:
            results.append(empty_case(pair, 'missing', 'no pair evidence supplied')); continue
        try:
            reference = entry['inputs']
            inputs_path = Path(reference['path'])
            inputs = isolated.load_inputs(inputs_path if inputs_path.is_absolute() else base / inputs_path, reference['sha256'])
            if inputs['pair_id'] != key or inputs['pair_plan'] != plan or inputs['protocol'] != protocol:
                raise ValueError('case inputs bind another pair/plan/protocol')
            config = bound_json(base, entry['config'])
            record = bound_json(base, entry['run']) if entry.get('run') is not None else None
            value = review_case(inputs, config, record, attempt=entry.get('attempt', 1))
        except ERRORS as error:
            value = empty_case(pair, 'invalid', str(error))
        value['file_sources'] = copy.deepcopy(entry)
        results.append(value)
    return seal(dict(schema=SCHEMA, status='measured_diagnostic' if all(r['status'] == 'measured_diagnostic' for r in results) else 'incomplete',
        diagnostic_only=True, qualification_eligible=False, allows=[], stable_capacity_claim=False,
        expected_shape_pairs=4, expected_requests=4 * EXPECTED_REQUESTS, expected_cells=48,
        cases=results, protocol_sha256=canonical_hash(protocol), pair_plan_sha256=canonical_hash(plan),
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        retention=dict(dependencies_closed=False, deletion_authorized=False),
        timing_scope='finite offline video batch; controller replay starts after worker/model warmup, includes CPU encode and queueing; not offered-rate capacity',
        statistics_scope='AB and BA differences averaged within each seed; three workload seeds are the resampling units, at one allocation; do not pool requests/repeats/shapes/models or generalize to hardware populations',
        cost_scope='model_ms is worker-reported model time, not a GPU-kernel measurement; CPU encode, replay, warmup and nested RPC spans are separate and must not be summed',
        limitations=['same fixed CFG2 digest agreement only; not FP32, cross-form or semantic quality',
                     'missing/failed/invalid cells cannot enter timing comparisons; incomplete seed sets have no CI',
                     'P0 eligibility and full P1 qualification remain separate; raw evidence stays retained']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = bound_json(args.manifest.parent, dict(path=args.manifest.name, sha256=args.manifest_sha256))
    result = evaluate(manifest, args.manifest.parent)
    result['source_manifest'] = dict(path=str(args.manifest), sha256=args.manifest_sha256)
    formal.write_new(args.output, seal(result))
    print(json.dumps(dict(status=result['status'], expected_requests=result['expected_requests'], qualification_eligible=False)))


if __name__ == '__main__':
    main()
