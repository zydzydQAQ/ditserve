#!/usr/bin/env python3
"""Review finite windows of native P3 sustained-arrival replays.

Evidence contains protocol/plan hash references, configs [{seed,config:ref}],
and runs [{seed,attempt,record:ref}]. Every configured policy/order is expected;
missing cells and runs remain visible. Only aggregate queue bins are emitted.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from native_metrics import percentile
from native_config_identity import config_identity, match_native_config
from p2_benefit import audit_arm, counts, expanded, finite, interval, planned_arm, sha
from p2_contrasts import AUDITED_NATIVE_BUNDLES
from p3_stream import arrivals, execution_settings, prompt_hash, validate_requests, validate_structure
from stage_gates import canonical_hash, read_json, source_file


# This opt-in path has its own reviewed source pins. It must not silently bless
# whichever source happens to be installed on the review machine. These pins
# cover the separately reviewed cutoff, deadline, cleanup and settlement paths.
CUTOFF_NATIVE_SOURCES = {path: (digest,) for path, digest in AUDITED_NATIVE_BUNDLES['97f0713'].items()}
TERMINAL_STATES = ('completed', 'failed', 'censored', 'not_executed', 'interrupted')


def cutoff_receipt(record, arm, cell, rows):
    """Verify the original evidence before any compatibility projection."""
    expected = arm['phase']['replay_cutoff']
    receipt = cell.get('replay_cutoff')
    if not isinstance(receipt, dict):
        raise ValueError('runtime cutoff lacks its observation receipt')
    config_binding = config_identity(record, arm['config'])
    match_native_config(config_binding, receipt)
    observation = f"{arm['selector']['phase']}/{arm['selector']['attempt']}/{arm['position']}"
    if (any(receipt.get(key) != value for key, value in expected.items())
            or receipt.get('observation_id') != observation
            or receipt.get('policy') != arm['phase']['policy']
            or cell.get('policy') != arm['phase']['policy']
            or receipt.get('clock') != 'controller_replay_monotonic_ms'
            or receipt.get('physical_stop_bound') is not False):
        raise ValueError('runtime cutoff receipt does not bind the frozen controller observation')
    if (receipt.get('cleanup_verified') is not True or cell.get('cleanup_verified') is not True):
        raise ValueError('runtime cutoff cleanup is not verified')
    phases = [phase for phase in record.get('phases', [])
              if phase.get('id') == arm['selector']['phase'] and phase.get('attempt') == arm['selector']['attempt']]
    if len(phases) != 1:
        raise ValueError('runtime cutoff needs one selected attempt')
    phase = phases[0]
    sessions = [item for item in record.get('budget', {}).get('sessions', [])
                if item.get('session_id') == phase.get('session_id')]
    charge = phase.get('charge', {})
    if (len(sessions) != 1 or sessions[0].get('cleanup_verified') is not True
            or charge.get('cleanup_verified') is not True
            or charge.get('session_id') != phase.get('session_id')
            or ('identity' in charge and charge['identity'] != sessions[0].get('identity'))):
        raise ValueError('runtime cutoff lacks settled session cleanup')
    match_native_config(config_binding, sessions[0].get('identity', {}))
    if 'identity' in charge:
        match_native_config(config_binding, charge['identity'])
    for owner in (cell, phase):
        occupancy = owner.get('final_occupancy', {})
        if (occupancy.get('status') not in ('exclusive_observed', 'busy')
                or occupancy.get('owned') != [] or occupancy.get('foreign') != []):
            raise ValueError('runtime cutoff lacks an empty verified GPU cleanup observation')
    if phase.get('unverified_workers', []) != []:
        raise ValueError('runtime cutoff has unverified worker process cleanup')
    horizon = finite(receipt.get('observation_horizon_ms'), 'cutoff observation horizon')
    observed = finite(receipt.get('observed_at_ms'), 'cutoff observed time')
    terminated = finite(receipt.get('termination_finished_ms'), 'cutoff termination time')
    after = finite(receipt.get('after_ms'), 'cutoff after_ms', positive=True)
    origin = finite(receipt.get('replay_origin_monotonic_seconds'), 'cutoff replay origin')
    if horizon > after or observed < horizon or terminated < observed:
        raise ValueError('runtime cutoff controller timestamps are inconsistent')
    for previous in phase['cells'][:arm['position']]:
        earlier = previous.get('replay_cutoff', {})
        if previous.get('status') not in ('completed', 'censored') or not earlier:
            raise ValueError('a later cutoff replay follows a failed or unstarted cell')
        match_native_config(config_binding, earlier)
        earlier_end = (finite(earlier.get('replay_origin_monotonic_seconds'), 'earlier replay origin')
                       + finite(earlier.get('termination_finished_ms'), 'earlier termination time') / 1000)
        if origin + 1e-6 < earlier_end:
            raise ValueError('a later policy replay starts before previous cell cleanup finishes')
    trigger = receipt.get('status')
    if trigger == 'triggered':
        if horizon != after or observed < after or cell.get('status') not in ('censored', 'failed'):
            raise ValueError('triggered cutoff must retain its censored observation window')
    elif trigger == 'not_reached':
        if cell.get('status') not in ('completed', 'failed'):
            raise ValueError('censoring requires an actual cutoff trigger')
    else:
        raise ValueError('unknown runtime cutoff outcome')
    expected_rows = {row['request_id']: row for row in arm['workload']}
    actual = {row['request_id']: row for row in rows}
    if len(rows) != len(actual) or set(actual) != set(expected_rows):
        raise ValueError('runtime cutoff omits frozen request denominators')
    observed_counts = dict(planned=len(expected_rows), **{status: 0 for status in TERMINAL_STATES})
    for row in rows:
        status = row.get('status')
        if status not in TERMINAL_STATES or row.get('observation_id') != observation:
            raise ValueError('runtime cutoff request has an unknown outcome or observation identity')
        observed_counts[status] += 1
        if 'finish_ms' in row and finite(row['finish_ms'], 'request finish_ms') > horizon:
            raise ValueError('a post-cutoff result entered the observed prefix')
        if finite(row.get('observation_ms'), 'request observation_ms') != horizon:
            raise ValueError('request observation differs from its cutoff horizon')
        if status == 'completed' and 'finish_ms' not in row:
            raise ValueError('completed cutoff request lacks a delivered finish time')
        if status == 'censored' and trigger != 'triggered':
            raise ValueError('ordinary failure cannot be relabelled as cutoff censoring')
        if status in ('censored', 'not_executed') and 'finish_ms' in row:
            raise ValueError('an unfinished cutoff request cannot have a delivered finish time')
        if ((status == 'not_executed' and row['arrival_ms'] <= horizon)
                or (status == 'censored' and row['arrival_ms'] > horizon)):
            raise ValueError('cutoff censoring does not match the request arrival window')
    raw_counts = receipt.get('counts')
    if (not isinstance(raw_counts, dict) or raw_counts != observed_counts
            or any(type(value) is not int for value in raw_counts.values())):
        raise ValueError('runtime cutoff outcome counts do not partition the frozen requests')
    if cell.get('status') == 'completed' and observed_counts['completed'] != len(rows):
        raise ValueError('completed cutoff cell contains unfinished requests')
    if cell.get('status') == 'censored' and (observed_counts['failed'] or observed_counts['interrupted']):
        raise ValueError('genuine failures cannot be hidden as ordinary cutoff censoring')
    return receipt


def audit_replay(record, arm, protocol):
    """Reuse P2 identity checks without changing any original measurements."""
    if 'replay_cutoff' not in arm['phase']:
        return audit_arm(record, arm, protocol)
    base = audit_arm(None, arm, protocol)
    if record is None:
        return base
    rows = [row for row in record.get('requests', []) if all(row.get(key) == arm['selector'][key]
            for key in ('phase', 'attempt', 'cell', 'order'))]
    try:
        config_identity(record, arm['config'])
        source = record.get('source', {})
        if (not sha(source.get('core_sha256')) or not source.get('files')
                or not all(sha(value) for value in source['files'].values())
                or any(source.get('native_files', {}).get(path) not in hashes for path, hashes in CUTOFF_NATIVE_SOURCES.items())):
            raise ValueError('cutoff evidence lacks its frozen configuration or reviewed source identity')
        phases = [phase for phase in record.get('phases', []) if phase.get('id') == arm['selector']['phase']
                  and phase.get('attempt') == arm['selector']['attempt']]
        if len(phases) != 1 or len(phases[0].get('cells', [])) <= arm['position']:
            return audit_arm(record, arm, protocol, audited_native_sources=CUTOFF_NATIVE_SOURCES)
        phase = phases[0]; cell = phase['cells'][arm['position']]
        status = cell.get('status')
        if cell.get('policy') != arm['phase']['policy']:
            raise ValueError('cutoff cell policy differs from its frozen planned identity')
        observation = f"{arm['selector']['phase']}/{arm['selector']['attempt']}/{arm['position']}"
        if any(row.get('observation_id') != observation for row in rows):
            raise ValueError('cutoff request observation identity differs from its frozen cell')
        if status == 'not_executed':
            expected = {row['request_id']: row for row in arm['workload']}
            if (len(rows) != len(expected) or {row['request_id'] for row in rows} != set(expected)
                    or any(row.get('status') != 'not_executed' for row in rows)):
                raise ValueError('unexecuted cell omits frozen request denominators')
            receipt = None
        elif status == 'failed' and 'replay_cutoff' not in cell and rows and all(
                row.get('status') == 'not_executed' for row in rows):
            # Loading/warmup failed before the replay clock existed. It cannot
            # demonstrate a cutoff or produce a performance interval.
            receipt = None
        else:
            receipt = cutoff_receipt(record, arm, cell, rows)
        for row in rows:
            if row.get('status') == 'not_executed' and (any(key in row for key in ('start_ms', 'finish_ms'))
                    or row.get('outputs') or row.get('model_ms', 0) != 0 or row.get('quanta', 0) != 0):
                raise ValueError('unexecuted request contains actual execution or delivered output')
        projection = dict(record)
        projection['phases'] = [dict(item) if item is phase else item for item in record['phases']]
        projected_phase = next(item for item in projection['phases'] if item is not phase
                               and item.get('id') == phase['id'] and item.get('attempt') == phase['attempt'])
        projected_phase['cells'] = list(phase['cells'])
        projected_cell = dict(cell)
        projected_phase['cells'][arm['position']] = projected_cell
        if status in ('censored', 'not_executed'):
            projected_cell['status'] = 'failed'
        if status == 'not_executed':
            projected_cell.pop('elapsed_ms', None)
        audited = audit_arm(projection, arm, protocol, audited_native_sources=CUTOFF_NATIVE_SOURCES)
        if audited['status'] == 'invalid':
            return audited
        audited['status'] = status
        audited['cutoff_receipt'] = receipt
        audited['replay_started'] = receipt is not None
        if receipt is not None:
            audited['elapsed_ms'] = receipt['observation_horizon_ms']
        else:
            audited['elapsed_ms'] = None
            audited['metrics'] = {}
        for kind, values in audited['by_kind'].items():
            observed = [row for row in rows if row['kind'] == kind]
            for state in ('censored', 'not_executed', 'interrupted'):
                values[state] = sum(row.get('status') == state for row in observed)
        return audited
    except (ValueError, KeyError, TypeError, IndexError, StopIteration) as error:
        base.update(status='invalid', reason=str(error), by_kind=counts(arm['workload'], rows, 'invalid'))
        return base


def window_summary(workload, rows, audit, windows):
    warm = windows['traffic_warmup_ms']; end = warm + windows['measurement_ms']
    cutoff = end + windows['drain_ms']
    indexed = {row['request_id']: row for row in rows}
    result = dict(status=audit['status'], by_kind={}, metrics={}, queue={})
    usable = audit['status'] in ('completed', 'failed', 'censored') and audit.get('replay_started', True)
    receipt = audit.get('cutoff_receipt')
    horizon = audit.get('elapsed_ms')
    horizon_source = 'cell_elapsed_ms'
    if horizon is None and usable:
        observations = [finite(row['observation_ms'], 'request observation_ms') for row in rows if 'observation_ms' in row]
        if not observations:
            observations = [finite(row['finish_ms'], 'completed finish_ms') for row in rows if row.get('status') == 'completed']
            horizon_source = 'settled_completion_lower_bound'
        else:
            horizon_source = 'request_observation_lower_bound'
        horizon = max(observations) if observations else None
    horizon = min(finite(horizon, 'observation horizon'), cutoff) if horizon is not None else None
    complete_tail = usable and audit['status'] == 'completed' and horizon is not None and all(
        row.get('status') == 'completed' and row['finish_ms'] <= horizon for row in rows) and len(rows) == len(workload)
    result.update(observed_horizon_ms=horizon, horizon_source=horizon_source if horizon is not None else None,
                  declared_drain_end_ms=cutoff,
                  runtime_cutoff_verified=receipt is not None,
                  actual_drain_cutoff_enforced=receipt is not None and receipt['status'] == 'triggered',
                  physical_stop_bound=False,
                  empty_tail_inferred=bool(complete_tail and horizon < end),
                  measurement_class_counts=dict(Counter(row['class'] for row in workload
                      if warm <= row['arrival_ms'] < end and 'class' in row)))
    for kind in ('image', 'video'):
        cohort = [row for row in workload if row['kind'] == kind and warm <= row['arrival_ms'] < end]
        counts = dict(planned=len(cohort), observed_arrivals=0, completed_by_drain=0, ontime=0,
                      known_misses=0, failed=0, interrupted=0, censored=0, not_executed=0, right_censored=0)
        latencies = []
        for expected in cohort:
            row = indexed.get(expected['request_id'])
            if not usable or horizon is None or expected['arrival_ms'] > horizon:
                counts['not_executed'] += 1
                continue
            counts['observed_arrivals'] += 1
            status = row.get('status') if row else None
            counts['failed'] += status == 'failed'
            for state in ('interrupted', 'censored', 'not_executed'):
                counts[state] += status == state
            finished = status == 'completed' and row['finish_ms'] <= horizon
            if finished:
                counts['completed_by_drain'] += 1
                latencies.append(row['finish_ms'] - expected['arrival_ms'])
            if finished and row['finish_ms'] <= expected['deadline_ms']:
                counts['ontime'] += 1
            elif expected['deadline_ms'] <= horizon or status == 'failed':
                counts['known_misses'] += 1
            else:
                counts['right_censored'] += 1
        result['by_kind'][kind] = counts
        n = counts['planned']
        if usable and horizon is not None and n:
            unresolved = n - counts['ontime'] - counts['known_misses']
            result['metrics'][kind] = dict(attainment_lower=counts['ontime'] / n,
                attainment_upper=(counts['ontime'] + unresolved) / n,
                completion_by_drain=counts['completed_by_drain'] / n,
                latency_p95_ms=percentile(latencies, .95) if len(latencies) == n else None,
                completion_rps_during_measurement=(sum(row.get('status') == 'completed' and row['kind'] == kind
                    and warm <= row['finish_ms'] < end for row in rows) * 1000 / windows['measurement_ms']
                    if horizon >= end or complete_tail else None))
        # This reconstructs outstanding requests, not the controller ready
        # queue: first-start timestamps cannot expose later preemption waits.
        points = []
        bins = math.ceil(windows['measurement_ms'] / windows['bin_ms'])
        for index in range(bins + 1):
            timestamp = min(end, warm + index * windows['bin_ms'])
            value = None
            if usable and horizon is not None and (timestamp <= horizon or complete_tail):
                value = sum(row['kind'] == kind and row['arrival_ms'] <= timestamp and not (
                    indexed.get(row['request_id'], {}).get('status') == 'completed'
                    and indexed[row['request_id']]['finish_ms'] <= timestamp) for row in workload)
            points.append(dict(at_ms=timestamp, unfinished=value))
        known = [point['unfinished'] for point in points]
        slope = None
        if all(value is not None for value in known) and audit['status'] in ('completed', 'censored'):
            times = [(point['at_ms'] - warm) / 1000 for point in points]
            mean_t, mean_q = sum(times) / len(times), sum(known) / len(known)
            denominator = sum((value - mean_t) ** 2 for value in times)
            slope = sum((t - mean_t) * (q - mean_q) for t, q in zip(times, known)) / denominator if denominator else None
        result['queue'][kind] = dict(bins=points, growth_requests_per_second=slope,
            delta=known[-1] - known[0] if known[0] is not None and known[-1] is not None else None,
            scope='arrived minus delivered-completed requests, including traffic warmup backlog; not ready-queue length')
        if kind in result['metrics']:
            result['metrics'][kind]['backlog_slope_requests_per_second'] = slope
    result['scope'] = ('arrival cohorts use [warmup,end); late drain completions remain in their original cohort; '
                       'throughput during measurement excludes drain; failed/unobserved outcomes are never zero-latency samples')
    if receipt is not None:
        result['runtime_cutoff'] = dict(receipt)
        result['horizon_source'] = 'verified_controller_cutoff_receipt'
        result['cleanup_after_observation_ms'] = receipt['termination_finished_ms'] - receipt['observation_horizon_ms']
    return result


def evaluate(manifest, base):
    if manifest.get('schema') != 'ditserve.p3-evidence.v1':
        raise ValueError('unknown P3 evidence schema')
    plan = read_json(source_file(base, manifest['plan']))
    protocol = read_json(source_file(base, manifest['protocol']))
    validate_structure(plan, protocol)
    settings = plan['statistics']
    configs, supplied = {}, {}
    for entry in manifest['configs']:
        if entry['seed'] in configs:
            raise ValueError('duplicate frozen seed configuration')
        configs[entry['seed']] = read_json(source_file(base, entry['config']))
    if set(configs) != set(plan['seeds']) or len(plan['seeds']) != len(set(plan['seeds'])) or len(plan['seeds']) < 3:
        raise ValueError('all planned independent seed configurations required')
    for entry in manifest.get('runs', []):
        if entry['seed'] not in configs or entry['seed'] in supplied or entry.get('attempt') != 1:
            raise ValueError('duplicate/unplanned seed or attempt; retries require a declared experiment')
        supplied[entry['seed']] = (read_json(source_file(base, entry['record'])), entry['record'])
    cases, groups, traces, runtimes = [], {}, set(), {}
    prompts = {kind: set() for kind in ('image', 'video')}
    configurations = None
    for seed, config in configs.items():
        validate_structure(plan, protocol, config)
        binding = config.get('benchmark', {}).get('p3', {})
        if (binding.get('plan_sha256') != canonical_hash(plan) or binding.get('windows') != plan['windows']
                or binding.get('arrival') != plan['arrival'] or config['benchmark']['formal_protocol']['role'] != 'p3_sustained'):
            raise ValueError('configuration does not bind the frozen sustained-arrival plan')
        phase = config['phases'][0]
        signature = {key: value for key, value in phase.items() if key != 'matrix'}
        signature.update(workers=config['workers'], models=config['benchmark']['model_manifests'],
                         policies=sorted(phase['matrix']['/policy']))
        if configurations is not None and configurations != signature:
            raise ValueError('worker/profile/GPU/control configuration changes across seeds')
        configurations = signature
        trace = phase['matrix']['/trace_provenance'][0]
        if trace['sha256'] in traces:
            raise ValueError('same trace reused as independent seeds')
        traces.add(trace['sha256'])
        if (trace['manifest'].get('generator') != 'poisson-until-fixed-horizon-v1'
                or trace['manifest'].get('normalization') != 'none'):
            raise ValueError('conditioned/renormalized trace is not P3 sustained evidence')
        workload = phase['matrix']['/workload'][0]
        validate_requests(workload, protocol, phase['profile'], plan['shape_ids'])
        expected_events = arrivals(plan, seed, {kind: len(workload) + 1 for kind in ('image', 'video')})
        if [(row['arrival_ms'], row['kind'], row['class']) for row in workload] != expected_events:
            raise ValueError('actual arrivals/shapes differ from the frozen unconditioned Poisson process')
        for row in workload:
            digest = prompt_hash(row['spec']['prompt'])
            if digest in prompts[row['kind']]:
                raise ValueError('independent prompts reused within/across planned seeds')
            prompts[row['kind']].add(digest)
        record, reference = supplied.get(seed, (None, None))
        cells = expanded(phase)
        sequence = [(i, 'forward') for i in range(len(cells))]
        if phase.get('paired', False):
            sequence += [(i, 'reverse') for i in reversed(range(len(cells)))]
        for index, order in sequence:
            selector = dict(phase=phase['id'], attempt=1, cell=index, order=order)
            arm = planned_arm(config, selector, protocol, seed)
            audit = audit_replay(record, arm, protocol)
            rows = [] if record is None else [row for row in record.get('requests', []) if all(
                row.get(field) == selector[field] for field in ('phase', 'attempt', 'cell', 'order'))]
            review = window_summary(arm['workload'], rows, audit, plan['windows'])
            expected_models = {protocol['shapes'][row['class']]['model'] for row in arm['workload']}
            if audit['status'] in ('completed', 'failed', 'censored') and not expected_models <= audit['identities'].keys():
                review['status'] = 'partial_identity'; review['metrics'] = {}
                review['identity_missing_models'] = sorted(expected_models - audit['identities'].keys())
            if audit['status'] in ('completed', 'failed', 'censored'):
                source = audit['source']
                if runtimes.setdefault('source', source) != source:
                    raise ValueError('source runtime changes across seeds/policies')
                for model, identity in audit['identities'].items():
                    if runtimes.setdefault(model, identity) != identity:
                        raise ValueError('executed model/runtime identity changes')
            if audit.get('reason'):
                review['reason'] = audit['reason']
            review.update(seed=seed, policy=cells[index]['policy'], order=order, cell=index, run=reference,
                          all_arrival_counts=audit['by_kind'])
            cases.append(review)
            group = groups.setdefault((cells[index]['policy'], order), {})
            if seed in group:
                raise ValueError('duplicate policy observation for one seed/order')
            group[seed] = review
    repeated = []
    for (policy, order), seeds in groups.items():
        if set(seeds) != set(plan['seeds']):
            raise ValueError('policy coverage changes across planned seeds')
        intervals = {}
        for kind in ('image', 'video'):
            names = {name for row in seeds.values() for name in row['metrics'].get(kind, {})}
            intervals[kind] = {}
            for name in sorted(names):
                values = [row['metrics'].get(kind, {}).get(name) for row in seeds.values()]
                intervals[kind][name] = interval([value for value in values if value is not None], len(plan['seeds']), settings)
        repeated.append(dict(policy=policy, order=order, seeds=sorted(seeds), intervals=intervals))
    cutoff_requested = bool(execution_settings(plan))
    cutoff_verified = cutoff_requested and all(row['runtime_cutoff_verified'] for row in cases)
    remaining = ['independent P0/P1/P2 qualification', 'multiple declared durations and offered-rate brackets',
                 'complete cost/quality evidence; bins are correlated and not independent repetitions']
    if not cutoff_verified:
        remaining.append('verified controller replay cutoff receipts for every planned cell if runtime cutoff is required')
    return dict(schema='ditserve.p3-window-review.v1',
        status='reviewed' if all(row['status'] in ('completed', 'failed', 'censored') for row in cases) else 'incomplete',
        allows=[], stable_capacity_claim=False, expected_cases=len(cases), cases=cases, repetitions=repeated,
        runtime_cutoff_status='verified' if cutoff_verified else 'pending' if cutoff_requested else 'not_requested',
        model=plan['model'], split=plan['split'], shape_ids=plan['shape_ids'],
        arrival=plan['arrival'], windows=plan['windows'], slo=protocol['slo'],
        scope='finite windows at one frozen offered rate; no stage promotion, quality certificate or stable-capacity estimate',
        interval_limit='empirical paired-seed bootstrap; few seeds and all-one attainment can give degenerate intervals; correlated time bins are not repetitions',
        remaining_requirements=remaining,
        source=dict(plan_sha256=manifest['plan']['sha256'], protocol_sha256=canonical_hash(protocol)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    result = evaluate(read_json(raw), args.manifest.resolve().parent)
    result['source_manifest'] = dict(path=str(args.manifest.resolve()), sha256=hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'expected_cases', 'stable_capacity_claim')}))


if __name__ == '__main__':
    main()
