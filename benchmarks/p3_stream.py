#!/usr/bin/env python3
"""Freeze genuine fixed-horizon Poisson arrivals for the existing native replay.

Only prepares local files. Runtime warmup precedes replay time zero; traffic
warmup is real offered work. The declared drain horizon is an analysis cutoff,
not a claim that the current phase watchdog stops at that replay-relative time.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random

from formal_benchmark import shape_spec, split_pools
from native_serving import partition_image_gpus
from p2_benefit import expanded, finite, trace_hash
from stage_gates import canonical_hash, read_json


SCHEMA = 'ditserve.p3-stream-plan.v1'


def rng(seed, purpose):
    return random.Random(int(canonical_hash([seed, 'p3', purpose])[:16], 16))


def prompt_hash(text):
    return canonical_hash(' '.join(text.split()))


def validate_structure(plan, protocol, config=None):
    """Pool-independent rules shared by trace preparation and evidence review."""
    if plan.get('schema') != SCHEMA or plan.get('protocol_sha256') != canonical_hash(protocol):
        raise ValueError('P3 plan/protocol identity differs')
    if (plan.get('split') not in ('development', 'test') or plan['split'] not in protocol['seeds']
            or plan.get('model') not in {row['model'] for row in protocol['shapes'].values() if row['kind'] == 'video'}):
        raise ValueError('P3 requires a known video model and development/test split')
    expected = {sid for sid, shape in protocol['shapes'].items()
                if shape['scope'] == ('formal' if plan['split'] == 'test' else 'development')
                and shape['model'] in (plan['model'], 'sd35')}
    ids = plan.get('shape_ids')
    if (not isinstance(ids, list) or not all(isinstance(sid, str) for sid in ids)
            or len(ids) != len(set(ids)) or set(ids) != expected):
        raise ValueError('P3 may not omit a frozen model/resolution class')
    seeds = plan['seeds']
    if (not isinstance(seeds, list) or len(seeds) < 3 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds)
            or not set(seeds) <= set(protocol['seeds'][plan['split']])):
        raise ValueError('P3 needs at least three distinct protocol seeds')
    if plan.get('prompt_reuse') != 'paired_policies_only' or plan.get('policy_order') != 'seed_randomized':
        raise ValueError('freeze unique prompts across seeds and randomized policy order')
    windows, arrival = plan['windows'], plan['arrival']
    for key in ('traffic_warmup_ms', 'measurement_ms', 'drain_ms', 'bin_ms'):
        finite(windows[key], key, positive=key in ('measurement_ms', 'bin_ms'))
    if windows['bin_ms'] > windows['measurement_ms'] or windows['measurement_ms'] / windows['bin_ms'] > 1000:
        raise ValueError('measurement needs between one and 1000 aggregate time bins')
    if arrival.get('kind') != 'poisson':
        raise ValueError('this P3 generator implements unconditioned Poisson only')
    finite(arrival['rate_rps'], 'offered rate_rps', positive=True)
    if not 0 < finite(arrival['video_probability'], 'video probability') < 1:
        raise ValueError('mixed traffic needs positive probability for both modalities')
    if set(arrival['shape_probabilities']) != {'image', 'video'}:
        raise ValueError('shape probabilities must cover both modalities only')
    for kind in ('image', 'video'):
        weights = arrival['shape_probabilities'][kind]
        if set(weights) != {sid for sid in expected if protocol['shapes'][sid]['kind'] == kind}:
            raise ValueError('shape probabilities must cover each frozen class')
        for weight in weights.values():
            finite(weight, 'shape probability', positive=True)
        if not math.isclose(sum(weights.values()), 1., rel_tol=0, abs_tol=1e-12):
            raise ValueError('within-modality shape probabilities must sum to one')
    finite(plan['startup_allowance_seconds'], 'startup allowance', positive=True)
    statistics = plan['statistics']
    if (statistics.get('confidence') != .95 or statistics.get('resampling_unit') != 'seed'
            or statistics.get('minimum_independent_seeds') != 3
            or type(statistics.get('bootstrap_samples')) is not int or not 1000 <= statistics['bootstrap_samples'] <= 100000
            or type(statistics.get('bootstrap_seed')) is not int):
        raise ValueError('freeze seed bootstrap settings')
    if config is None:
        return
    if len(config.get('phases', [])) != 1 or config.get('benchmark', {}).get('native_serving') is not True:
        raise ValueError('P3 needs one frozen native serving phase')
    formal = config['benchmark']['formal_protocol']
    if (formal['sha256'] != canonical_hash(protocol) or formal['split'] != plan['split']
            or formal['shape_ids'] != ids or formal.get('schema') != protocol['schema']
            or config['benchmark'].get('exact_mode') is not True):
        raise ValueError('base configuration lacks the planned exact protocol/split/shapes')
    phase = config['phases'][0]
    if (set(phase.get('matrix', {})) != {'/policy', '/trace_provenance', '/workload'}
            or len(phase['matrix']['/workload']) != 1 or len(phase['matrix']['/trace_provenance']) != 1):
        raise ValueError('P3 needs the native policy/single frozen trace matrix only')


def service_times(protocol, profile, sid):
    cost, shape = profile['classes'][sid], protocol['shapes'][sid]
    if cost['kind'] != shape['kind']:
        raise ValueError('profile modality differs from frozen shape')
    if cost['kind'] == 'video':
        gpu = (finite(cost.get('prepare_ms', 0), 'prepare_ms') + shape['steps'] * finite(cost['step_compute_ms'], 'step_compute_ms')
               + finite(cost['finish_ms'], 'finish_ms'))
        service = gpu + finite(cost.get('encode_ms', 0), 'encode_ms')
    else:
        gpu = service = finite(cost['service_ms'], 'service_ms', positive=True)
    return finite(gpu, 'GPU service', positive=True), finite(service, 'service', positive=True)


def validate_requests(rows, protocol, profile, shape_ids):
    """Derive SLOs from the profile, never from a rehashed trace's labels."""
    for row in rows:
        sid = row['class']
        if sid not in shape_ids or row['kind'] != protocol['shapes'][sid]['kind']:
            raise ValueError('P3 request is outside its frozen shape/modality')
        _, service = service_times(protocol, profile, sid)
        scale = protocol['slo']['video_scale' if row['kind'] == 'video' else 'image_scale']
        for key in ('arrival_ms', 'deadline_ms', 'effective_slo_scale', 'service_ms', 'slo_ms'):
            finite(row[key], key, positive=key not in ('arrival_ms', 'deadline_ms'))
        if (row['effective_slo_scale'] != scale or not math.isclose(row['service_ms'], service, rel_tol=1e-12, abs_tol=1e-9)
                or not math.isclose(row['slo_ms'], service * scale, rel_tol=1e-12, abs_tol=1e-9)
                or not math.isclose(row['deadline_ms'] - row['arrival_ms'], service * scale, rel_tol=1e-12, abs_tol=1e-9)
                or {k: v for k, v in row['spec'].items() if k not in ('prompt', 'seed', 'request_id')} != shape_spec(protocol, sid)):
            raise ValueError('P3 preserves formal per-shape SLOs and specs; stress overrides need a separate pilot protocol')


def validate_plan(plan, protocol, config, pools, usage):
    validate_structure(plan, protocol, config)
    if (usage.get('schema') != 'ditserve.prompt-usage.v1' or usage.get('protocol_sha256') != canonical_hash(protocol)
            or plan.get('prompt_usage_sha256') != canonical_hash(usage)
            or plan.get('prompt_pools_sha256') != {kind: pool['sha256'] for kind, pool in pools.items()}):
        raise ValueError('prompt pools or explicit prior-usage identity differs')
    for hashes in usage['normalized_prompt_sha256'].values():
        if not isinstance(hashes, list) or len(hashes) != len(set(hashes)) or any(
                not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value) for value in hashes):
            raise ValueError('invalid prior prompt usage inventory')


def arrivals(plan, seed, capacities):
    """Independent modality Poisson processes; no count conditioning/rescaling."""
    horizon = plan['windows']['traffic_warmup_ms'] + plan['windows']['measurement_ms']
    settings = plan['arrival']
    events = []
    for kind in ('image', 'video'):
        probability = settings['video_probability'] if kind == 'video' else 1 - settings['video_probability']
        rate = settings['rate_rps'] * probability / 1000
        arrival_rng, shape_rng = rng(seed, kind + '-arrival'), rng(seed, kind + '-shape')
        weights = settings['shape_probabilities'][kind]
        names = sorted(weights)
        timestamp, count = 0., 0
        while True:
            timestamp += arrival_rng.expovariate(rate)
            if timestamp >= horizon:
                break
            count += 1
            if count > capacities[kind]:
                raise ValueError(f'insufficient independent {kind} prompts: need at least {count}, available {capacities[kind]}; no cycling/truncation')
            shape = shape_rng.choices(names, weights=[weights[name] for name in names], k=1)[0]
            events.append((timestamp, kind, shape))
    return sorted(events)


def prepare(config, protocol, plan, pools, usage):
    validate_plan(plan, protocol, config, pools, usage)
    selected = split_pools(pools, protocol, plan['split'])
    available = {kind: [sample for sample in pool['samples']
                       if prompt_hash(sample['prompt']) not in set(usage['normalized_prompt_sha256'][kind])]
                 for kind, pool in selected.items()}
    for kind in available:
        available[kind].sort(key=lambda sample: canonical_hash([plan['seeds'], kind, sample['sample_id']]))
    remaining = {kind: len(rows) for kind, rows in available.items()}
    streams = {}
    for seed in plan['seeds']:
        streams[seed] = arrivals(plan, seed, remaining)
        for kind, count in Counter(event[1] for event in streams[seed]).items():
            remaining[kind] -= count
    phase = config['phases'][0]
    templates = {row['class']: row for row in phase['matrix']['/workload'][0]}
    if set(templates) != set(plan['shape_ids']):
        raise ValueError('base trace lacks a shape/SLO template')
    def gpu_service(sid):
        return service_times(protocol, phase['profile'], sid)[0]
    validate_requests(phase['matrix']['/workload'][0], protocol, phase['profile'], plan['shape_ids'])
    expected_work = {}
    for kind, weights in plan['arrival']['shape_probabilities'].items():
        expected_work[kind] = sum(weight * gpu_service(sid) for sid, weight in weights.items())
    probability = plan['arrival']['video_probability']
    image_share = partition_image_gpus(config['benchmark']['gpus'], probability,
        dict(steps=1, step_compute_ms=expected_work['video']), expected_work['image'])
    cursor = Counter()
    consumed = {kind: [] for kind in ('image', 'video')}
    results = {}
    warm = plan['windows']['traffic_warmup_ms']; end = warm + plan['windows']['measurement_ms']
    for seed, events in streams.items():
        if not events:
            raise ValueError('empty Poisson realization; no automatic reseeding or extra arrivals')
        rows = []
        for index, (timestamp, kind, sid) in enumerate(events):
            sample = available[kind][cursor[kind]]; cursor[kind] += 1
            consumed[kind].append(prompt_hash(sample['prompt']))
            row = copy.deepcopy(templates[sid])
            rid = f'p3-s{seed}-{index:06d}'
            row.update(request_id=rid, arrival_ms=timestamp, deadline_ms=timestamp + row['slo_ms'],
                       trace_seed=seed, dataset_sample_id=sample['sample_id'],
                       p3_cohort='traffic_warmup' if timestamp < warm else 'measurement')
            row['spec'].update(request_id=rid, prompt=sample['prompt'], seed=int(canonical_hash([seed, kind, index, 'content'])[:8], 16))
            rows.append(row)
        windows = []
        for label, left, right in [('traffic_warmup', 0., warm), ('measurement', warm, end)]:
            subset = [row for row in rows if left <= row['arrival_ms'] < right]
            windows.append(dict(label=label, start_ms=left, duration_ms=right - left,
                parameters=dict(kind='unconditioned_poisson', rate_rps=plan['arrival']['rate_rps'],
                    video_probability=plan['arrival']['video_probability'], duration_ms=right-left),
                count=len(subset), counts=dict(Counter(row['kind'] for row in subset))))
        manifest = dict(schema='ditserve.benchmark.mixed-workload.v1', generator='poisson-until-fixed-horizon-v1',
            normalization='none', seed=seed, slots=config['benchmark']['gpus'],
            total_duration_ms=end, offered_rate_rps=plan['arrival']['rate_rps'],
            realized_rate_rps=1000 * len(rows) / end, counts=dict(Counter(row['kind'] for row in rows)),
            class_counts=dict(Counter(row['class'] for row in rows)), windows=windows,
            sources={kind: {key: selected[kind].get(key) for key in ('kind', 'source', 'selection')} for kind in selected},
            workload_sha256=trace_hash(rows), p3_plan_sha256=canonical_hash(plan))
        value = copy.deepcopy(config)
        current = value['phases'][0]
        policies = list(current['matrix']['/policy']); rng(seed, 'policy-order').shuffle(policies)
        current['matrix']['/policy'] = policies
        current['options']['partition_image_gpus'] = image_share
        provenance = dict(path='generated:p3-fixed-horizon', sha256=manifest['workload_sha256'], manifest=manifest)
        current['matrix']['/workload'] = [rows]; current['matrix']['/trace_provenance'] = [provenance]
        parent = copy.deepcopy(value['benchmark']['formal_protocol'])
        value['benchmark']['formal_protocol']['role'] = 'p3_sustained'
        value['benchmark']['formal_protocol']['dataset_split_sha256'] = {kind: pool['sha256'] for kind, pool in selected.items()}
        for field in ('p0_diagnostic_controls', 'p1', 'p2', 'preemption_diagnostic', 'time_planning', 'load', 'cv'):
            value['benchmark'].pop(field, None)
        value['benchmark'].update(seed=seed, count=len(rows), trace=provenance, pattern='unconditioned_poisson',
            policies=policies, requested_policies=policies, quality_qualified=False, performance_qualified=False,
            video_fraction=sum(row['kind'] == 'video' for row in rows) / len(rows),
            scope='P3 finite observation of sustained offered arrivals; stable capacity is not established',
            p3=dict(plan_sha256=canonical_hash(plan), parent_formal_protocol=parent,
                    source_config_sha256=canonical_hash(config), windows=plan['windows'],
                    arrival=plan['arrival'], prompt_reuse=plan['prompt_reuse'],
                    generator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    predicted_gpu_work_load=plan['arrival']['rate_rps'] * (probability * expected_work['video']
                        + (1-probability) * expected_work['image']) / 1000 / value['benchmark']['gpus']))
        value['benchmark']['partition_image_gpus'] = dict(value=image_share, rule='expected calibrated GPU work under frozen P3 probabilities')
        n_cells = len(expanded(current)) * (2 if current.get('paired', False) else 1)
        bound = math.ceil(n_cells * (plan['startup_allowance_seconds'] + (end + plan['windows']['drain_ms']) / 1000))
        current['watchdog_seconds'] = bound
        value['budget']['watchdog_seconds'] = bound
        value['budget']['limit_gpu_seconds'] = bound * value['benchmark']['gpus']
        value['benchmark']['p3']['budget_scope'] = 'phase wall-time bound includes startup; not a replay-relative drain cutoff or completion guarantee'
        value['experiment'] = f'p3-{plan["model"]}-{plan["split"]}-s{seed}'
        results[seed] = value
    reservation = dict(schema='ditserve.prompt-usage.v1', protocol_sha256=canonical_hash(protocol),
        parent_usage_sha256=canonical_hash(usage), p3_plan_sha256=canonical_hash(plan),
        normalized_prompt_sha256={kind: sorted(set(usage['normalized_prompt_sha256'][kind]) | set(consumed[kind])) for kind in consumed},
        remaining_unique_prompts=remaining, reserved_prompts={kind: len(values) for kind, values in consumed.items()},
        scope='reservation for these frozen traces; does not claim they have executed')
    return results, reservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'protocol', 'plan', 'image_pool', 'video_pool', 'usage', 'output_dir'):
        parser.add_argument('--' + key.replace('_', '-'), required=True, type=Path)
    args = parser.parse_args()
    protocol, plan = (read_json(path.read_bytes()) for path in (args.protocol, args.plan))
    configs, reservation = prepare(read_json(args.config.read_bytes()), protocol, plan,
        {kind: read_json(getattr(args, kind + '_pool').read_bytes()) for kind in ('image', 'video')},
        read_json(args.usage.read_bytes()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        with (args.output_dir / name).open('xb') as stream:
            stream.write(raw)
        return dict(path=name, sha256=hashlib.sha256(raw).hexdigest())
    evidence = dict(schema='ditserve.p3-evidence.v1', protocol=write('protocol.json', protocol),
        plan=write('plan.json', plan), configs=[dict(seed=seed, config=write(f'seed-{seed}.json', config)) for seed, config in configs.items()], runs=[])
    write('prompt-reservation.json', reservation); write('evidence.json', evidence)
    print(json.dumps(dict(status='prepared_only', seeds=len(configs), reserved_prompts=reservation['reserved_prompts'])))


if __name__ == '__main__':
    main()
