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
from p2_benefit import audit_arm, expanded, finite, interval, planned_arm
from p3_stream import arrivals, prompt_hash, validate_requests, validate_structure
from stage_gates import canonical_hash, read_json, source_file


def window_summary(workload, rows, audit, windows):
    warm = windows['traffic_warmup_ms']; end = warm + windows['measurement_ms']
    cutoff = end + windows['drain_ms']
    indexed = {row['request_id']: row for row in rows}
    result = dict(status=audit['status'], by_kind={}, metrics={}, queue={})
    usable = audit['status'] in ('completed', 'failed')
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
                  declared_drain_end_ms=cutoff, actual_drain_cutoff_enforced=False,
                  empty_tail_inferred=bool(complete_tail and horizon < end),
                  measurement_class_counts=dict(Counter(row['class'] for row in workload
                      if warm <= row['arrival_ms'] < end and 'class' in row)))
    for kind in ('image', 'video'):
        cohort = [row for row in workload if row['kind'] == kind and warm <= row['arrival_ms'] < end]
        counts = dict(planned=len(cohort), observed_arrivals=0, completed_by_drain=0, ontime=0,
                      known_misses=0, failed=0, interrupted=0, not_executed=0, right_censored=0)
        latencies = []
        for expected in cohort:
            row = indexed.get(expected['request_id'])
            if not usable or horizon is None or expected['arrival_ms'] > horizon:
                counts['not_executed'] += 1
                continue
            counts['observed_arrivals'] += 1
            status = row.get('status') if row else None
            counts['failed'] += status == 'failed'
            counts['interrupted'] += status not in (None, 'completed', 'failed')
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
        if all(value is not None for value in known) and audit['status'] == 'completed':
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
            audit = audit_arm(record, arm, protocol)
            rows = [] if record is None else [row for row in record.get('requests', []) if all(
                row.get(field) == selector[field] for field in ('phase', 'attempt', 'cell', 'order'))]
            review = window_summary(arm['workload'], rows, audit, plan['windows'])
            expected_models = {protocol['shapes'][row['class']]['model'] for row in arm['workload']}
            if audit['status'] in ('completed', 'failed') and not expected_models <= audit['identities'].keys():
                review['status'] = 'partial_identity'; review['metrics'] = {}
                review['identity_missing_models'] = sorted(expected_models - audit['identities'].keys())
            if audit['status'] in ('completed', 'failed'):
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
    return dict(schema='ditserve.p3-window-review.v1',
        status='reviewed' if all(row['status'] in ('completed', 'failed') for row in cases) else 'incomplete',
        allows=[], stable_capacity_claim=False, expected_cases=len(cases), cases=cases, repetitions=repeated,
        model=plan['model'], split=plan['split'], shape_ids=plan['shape_ids'],
        arrival=plan['arrival'], windows=plan['windows'], slo=protocol['slo'],
        scope='finite windows at one frozen offered rate; no stage promotion, quality certificate or stable-capacity estimate',
        interval_limit='empirical paired-seed bootstrap; few seeds and all-one attainment can give degenerate intervals; correlated time bins are not repetitions',
        remaining_requirements=['independent P0/P1/P2 qualification', 'multiple declared durations and offered-rate brackets',
            'strict replay-relative drain cutoff if required by the execution protocol',
            'complete cost/quality evidence; bins are correlated and not independent repetitions'],
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
