#!/usr/bin/env python3
"""Summarize hash-bound controller spans without adding overlapping costs.

Input: {schema: ditserve.p1-cost-evidence.v1, runs: [{path, sha256}]}.
These are timing diagnostics, never a P1 receipt, kernel profile or quality gate.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from native_metrics import distribution
from p2_benefit import expanded, sha
from stage_gates import canonical_hash, read_json, source_file


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid controller timestamp/duration')
    return value


def duration(start, finish):
    value = number(finish) - number(start)
    if value < -1e-6:
        raise ValueError('controller interval is reversed')
    return max(0, value)


def contains(parent, child):
    if child['start_ms'] < parent['start_ms'] - 1e-6 or child['finish_ms'] > parent['finish_ms'] + 1e-6:
        raise ValueError('rank RPC lies outside its fanout interval')


def dependencies(event):
    """Check completed fanouts against their actual child RPCs, including failures."""
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
            contains(parent, child)
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
            duration(prepared['finish_ms'], follower['start_ms'])
    for rid in set(imports) | set(parents['imports_all_ready']):
        exported = parents['export'].get(rid)
        parent, children = parents['imports_all_ready'].get(rid), imports[rid]
        if exported is None or exported['status'] != 'completed':
            raise ValueError('imports have no completed source export')
        if len({child.get('worker') for child in children}) != len(children) or any(not child.get('worker') for child in children):
            raise ValueError('import targets must have distinct worker identities')
        for child in children:
            duration(exported['finish_ms'], child['start_ms'])
        if parent is None:
            if event['status'] != 'failed':
                raise ValueError('completed imports lack an all-ready interval')
            continue  # A launch failure may leave a measured, incomplete prefix.
        duration(exported['finish_ms'], parent['start_ms'])
        parent_status(parent, children)
        if rid == event['request_id'] and 'transfer_targets' in event and len(children) != event['transfer_targets']:
            raise ValueError('import target count differs from planned fanout')
    if by_name['advance']:
        for rid, exported in parents['export'].items():
            parent = parents['imports_all_ready'].get(rid)
            if exported['status'] != 'completed' or parent is None or parent['status'] != 'completed':
                raise ValueError('advance precedes successful transfer readiness')
            for advance in by_name['advance']:
                duration(parent['finish_ms'], advance['start_ms'])
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


def validate_links(events, stops, identities):
    for sid, stop in stops.items():
        previous = events.get(stop.get('task_event_id'))
        if (previous is None or 'start_ms' not in previous or previous['status'] != 'completed'
                or identities[previous['event_id']] != stop['request_identity']
                or sid != previous['event_id'] + '/stop'):
            raise ValueError('stop does not bind its stopped task')
        duration(previous['start_ms'], stop['ring_ms'])
        duration(stop['ring_ms'], previous['all_done_ms'])
        duration(previous['all_done_ms'], stop['observed_ms'])
        duration(stop['observed_ms'], previous['observed_ms'])
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
            duration(stop['observed_ms'], event['start_ms'])
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
            duration(restored['observed_ms'], event['start_ms'])


def summarize(record):
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
                continue  # Legacy duration-only evidence has no causal link.
            if stop['event_id'] in stops:
                raise ValueError('duplicate stop event')
            stop = dict(stop, request_identity=identity)
            if not math.isclose(duration(stop['ring_ms'], stop['observed_ms']), number(stop['stop_ms']), abs_tol=1e-6):
                raise ValueError('stop interval differs from duration')
            stops[stop['event_id']] = stop
    validate_links(events, stops, identities)
    groups = defaultdict(lambda: defaultdict(lambda: dict(counts=Counter(), wall=defaultdict(list), rank_rpc=defaultdict(list))))
    for eid, event in events.items():
        group = groups[identities[eid][:4]][event['transition']]
        status = event.get('status')
        if status not in ('completed', 'failed'):
            raise ValueError('unsettled task cannot be reported as measured timing')
        group['counts']['events'] += 1
        group['counts']['async_' + status] += 1
        task = 'start_ms' in event
        observed = number(event['observed_ms'])
        if task:
            if event.get('controller_status') not in ('completed', 'aborted'):
                raise ValueError('missing controller consumption status')
            group['counts']['controller_' + event['controller_status']] += 1
            duration(event['selected_ms'], event['start_ms'])
            group['wall']['async_launch_delay_ms'].append(duration(event['start_ms'], event['async_begin_ms']))
            duration(event['async_begin_ms'], event['all_done_ms'])
            group['wall']['controller_observation_lag_ms'].append(duration(event['all_done_ms'], observed))
            if status == 'completed' and event['controller_status'] == 'completed':
                group['wall']['task_to_progress_including_compute_ms'].append(duration(event['start_ms'], event['all_done_ms']))
        elif event['transition'] not in ('tiered_offload', 'tiered_restore'):
            raise ValueError('non-task timing must identify a tiered RPC')
        stages = event.get('stages', [])
        exports, ready = {}, {}
        for stage in stages:
            elapsed = duration(stage['start_ms'], stage['finish_ms'])
            if not math.isclose(elapsed, number(stage['elapsed_ms']), abs_tol=1e-6):
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
        dependencies(event)
        # Separate paired request transfers; never sum their rank work or the
        # whole pair's shared advance interval as per-request wall time.
        for rid, end in ready.items():
            if rid not in exports:
                raise ValueError('imports-ready has no corresponding source export')
            duration(exports[rid]['finish_ms'], end['start_ms'])
            group['wall']['export_to_targets_ready_ms'].append(duration(exports[rid]['start_ms'], end['finish_ms']))
        link = event.get('resume_from_stop_event_id')
        if link:
            stop = stops.get(link)
            if not task or stop is None or stop['request_identity'] != identities[eid]:
                raise ValueError('resume has no same-request stop evidence')
            group['wall']['stopped_to_next_dispatch_ms'].append(duration(stop['observed_ms'], event['start_ms']))
            if status == 'completed' and event['controller_status'] == 'completed':
                group['wall']['stopped_to_next_progress_including_other_work_ms'].append(duration(stop['observed_ms'], event['all_done_ms']))
    planning = record.get('controller_planning', [])
    for item in planning:
        for field in ('calls', 'failed', 'total_ms', 'max_ms'):
            number(item[field])
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


def evaluate(manifest, base):
    if manifest.get('schema') != 'ditserve.p1-cost-evidence.v1':
        raise ValueError('unknown cost evidence schema')
    results, identities = [], set()
    for reference in manifest['runs']:
        record = read_json(source_file(base, reference))
        if record['run_id'] in identities:
            raise ValueError('duplicate run evidence')
        identities.add(record['run_id'])
        results.append(dict(summarize(record), raw_reference=reference))
    return dict(schema='ditserve.p1-cost-review.v1', status='diagnostic_only', allows=[], runs=results,
        scope='controller wall-time diagnostics; CPU/mock and GPU qualification require separate evidence',
        grouping='each run is separated by phase, attempt, cell and order before transition distributions; no policy pooling',
        interpretation=dict(export_to_targets_ready='source export through all new targets ready; not full stale-copy cleanup',
            stopped_to_next_dispatch='waiting and intervening work after stop; not restore overhead',
            task_to_progress='includes useful model computation, RPC and dispatch',
            per_rank_rpc='parallel rank intervals include worker mutex/serialization; do not sum as wall time'),
        limitations=['nested spans and stop/advance intervals overlap; never add them as a cost decomposition',
            'failed events retain observed prefixes, not hypothetical completion costs',
            'shared pair events count once; no per-request attribution of shared computation',
            'no device-kernel or independent-seed confidence claim; no P1 promotion'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    result = evaluate(read_json(raw), args.manifest.resolve().parent)
    result['source_manifest'] = dict(path=str(args.manifest.resolve()), sha256=hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], runs=len(result['runs']))))


if __name__ == '__main__':
    main()
