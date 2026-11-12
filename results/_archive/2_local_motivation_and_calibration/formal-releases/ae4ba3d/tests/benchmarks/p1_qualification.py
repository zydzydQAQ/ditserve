#!/usr/bin/env python3
"""Review raw P1 fixed-form evidence; never issue a stage-promotion receipt.

The manifest binds a verified P0 receipt, the formal protocol, a frozen P1
plan, and explicit run references. Missing or failed cases remain denominators.
Confidence intervals resample independent seeds, never individual requests.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from native_exact import checked_digest, compare, distribution, percentile, same_policy, stop_samples, validate_run
from p1_benchmark import canonical_hash, p0_controls, read_json, source_file, validate_plan, verify_receipt


def finite(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or positive and value == 0:
        raise ValueError("missing or invalid measured " + name)
    return value


def seed_interval(values, settings):
    """One observation per verified seed; incomplete repetitions have no CI."""
    minimum = settings['minimum_independent_seeds']
    result = dict(seed_count=len(values), unit='independent workload seed',
                  mean=statistics.fmean(values) if values else None, ci95=None,
                  status='pending' if len(values) < minimum else 'measured')
    if len(values) >= minimum:
        rng = random.Random(settings['bootstrap_seed'])
        means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(settings['bootstrap_samples'])]
        tail = (1 - settings['confidence']) / 2
        result['ci95'] = [percentile(means, tail), percentile(means, 1 - tail)]
    return result


def p0_identity(base, reference):
    raw = source_file(base, reference)
    path = (base / reference['path']).resolve()
    verified = verify_receipt(path, required_stage='P0', action='p1_diagnostics')
    receipt = read_json(raw)
    manifest_path = (path.parent / receipt['source_manifest']['path']).resolve()
    manifest = read_json(source_file(path.parent, receipt['source_manifest']))
    smoke = read_json(source_file(manifest_path.parent, manifest['contract']['runtime_smoke']))
    manifests = {}
    for item in manifest['runs']:
        run = read_json(source_file(manifest_path.parent, {key: item[key] for key in ('path', 'sha256')}))
        for model, binding in run['config']['benchmark']['model_manifests'].items():
            digest = binding['sha256']
            if model in manifests and manifests[model] != digest:
                raise ValueError('P0 runs disagree on model manifest identity')
            manifests[model] = digest
    return verified, smoke['python_sources'], manifests


def image_pair(reference, candidate):
    required_input = {'prompt', 'negative_prompt', 'seed', 'height', 'width', 'frames', 'steps', 'guidance_scale'}
    required_identity = {'model_path', 'model_manifest_sha256', 'model_integrity', 'runtime_identity',
                         'dtype', 'text_encoder_dtype', 'attention', 'engine', 'seed_protocol', 'batch_protocol'}
    generations = []
    for row in (reference, candidate):
        spec = row.get('spec', {})
        generation = row['outputs'][-1].get('generation', {})
        if not required_input <= spec.keys() or not required_identity <= generation.keys():
            raise ValueError('SD35 output lacks complete input or runtime identity')
        if (not generation['runtime_identity'] or not generation['engine']
                or generation['model_integrity'] != 'manifest_verified'
                or any(generation.get(key) != spec[key] for key in required_input)):
            raise ValueError('SD35 generation does not bind its executed inputs and runtime')
        generations.append({key: value for key, value in generation.items() if key != 'request_id'})
    if generations[0] != generations[1]:
        raise ValueError('SD35 paired execution identities differ')
    for key in ('shape', 'dtype'):
        if (key not in reference['outputs'][-1]['output'] or key not in candidate['outputs'][-1]['output']
                or reference['outputs'][-1]['output'][key] != candidate['outputs'][-1]['output'][key]):
            raise ValueError('SD35 paired output identity differs')
    if checked_digest(reference) != checked_digest(candidate):
        raise ValueError('SD35 paired finite output digests differ')


def audit_run(record, pair, role, seed, protocol, plan, identity):
    controls = plan['roles'][role]
    rows = record.get('requests', [])
    result = dict(pair_id=pair['id'], role=role, seed=seed, run_id=record.get('run_id'),
        expected_requests=2 * controls['count'], observed_requests=len(rows),
        completed_requests=sum(row.get('status') == 'completed' for row in rows),
        failed_requests=sum(row.get('status') != 'completed' for row in rows), status='blocked')
    try:
        qualified, python_sources, model_manifests = identity
        source = record.get('source', {})
        if source.get('core_sha256') != qualified['build']['core_sha256'] or source.get('files') != python_sources:
            raise ValueError('P1 C++ or frozen Python identity differs from verified P0')
        benchmark = validate_run(record)
        if record.get('failures') or any(row.get('status') != 'completed' for row in rows):
            raise ValueError('P1 contains failed/incomplete requests; failures remain in the denominator')
        expected = dict(plan_sha256=canonical_hash(plan), case_id=f"{pair['id']}-{role}-s{seed}",
            pair_id=pair['id'], model=pair['model'], shape_ids=[pair['video_shape'], pair['image_shape']],
            seed=seed, role=role, controls=controls)
        if ({key: benchmark.get('p1', {}).get(key) for key in expected} != expected
                or benchmark.get('seed') != seed):
            raise ValueError('run does not bind the prescribed P1 case and seed')
        formal = benchmark.get('formal_protocol', {})
        if (formal.get('schema') != protocol['schema'] or formal.get('sha256') != canonical_hash(protocol)
                or formal.get('shape_ids') != expected['shape_ids'] or formal.get('split') != plan['prompt_split']
                or formal.get('role') != 'p1_' + role):
            raise ValueError('run lacks formal-shape P1 protocol identity')
        if benchmark['trace']['manifest'].get('seed') != seed:
            raise ValueError('trace seed differs from the independent repetition')
        config_phase = record['config']['phases'][0]
        workers = record['config']['workers']
        uuids = []
        for slot in config_phase['gpus']:
            video_uuid, image_uuid = (workers[slot[kind]]['gpu_uuid'] for kind in ('video', 'image'))
            if video_uuid != image_uuid or video_uuid in uuids:
                raise ValueError('P1 slot does not identify a unique co-resident GPU')
            uuids.append(video_uuid)
        sessions = record.get('budget', {}).get('sessions', [])
        if (not uuids or not sessions or any(session['identity']['gpu_uuids'] != uuids for session in sessions)
                or config_phase['profile']['source']['gpu_uuids'] != uuids
                or benchmark.get('gpus') != len(uuids)
                or benchmark.get('profile_placement', {}).get('matches') is not True
                or any(cell.get('gpus') != len(uuids) for cell in record['phases'][0]['cells'])):
            raise ValueError('P1 measured allocation, profile or policy GPU budgets differ')
        for model in (pair['model'], 'sd35'):
            if benchmark['model_manifests'][model]['sha256'] != model_manifests.get(model):
                raise ValueError('model weights differ from verified P0: ' + model)
        for row in rows:
            for field in ('arrival_ms', 'deadline_ms', 'finish_ms'):
                finite(row.get(field), field)
            if row['finish_ms'] < row['arrival_ms']:
                raise ValueError('request finishes before arrival')
            sid = pair['video_shape'] if row['kind'] == 'video' else pair['image_shape']
            shape = protocol['shapes'][sid]
            expected_shape = ([1, shape['frames'], shape['height'], shape['width'], 3]
                              if row['kind'] == 'video' else [1, shape['height'], shape['width'], 3])
            if row['class'] != sid or row['outputs'][-1]['output'].get('shape') != expected_shape:
                raise ValueError('P1 executed class or decoded output dimensions differ from the declared pair')
        # Reuse the execution-input validator with the explicit P1 controls.
        # This private validation view never modifies or republishes raw data,
        # and does not call the P0 stage evaluator or relabel measured evidence.
        view = dict(record, config=dict(record['config'], benchmark=dict(benchmark,
            p0_diagnostic_controls=benchmark['p1']['controls'])))
        input_protocol = dict(protocol, p0_diagnostics=plan['roles'])
        p0_controls(view, dict(model=pair['model'], role='preemption' if role == 'contention' else role), input_protocol)
        exact = compare(record, require_preemption=role == 'contention')
        if not exact['qualification_eligible']:
            raise ValueError('fixed-CFG2 digest/runtime/weights or observed-stop requirement failed')
        phase = record['phases'][0]
        arms = {}
        for policy in plan['policies']:
            cells = [cell for cell in phase['cells'] if same_policy(cell['policy'], policy)]
            if len(cells) != 1:
                raise ValueError('missing or duplicated controlled policy cell')
            cell = cells[0]
            finite(cell.get('elapsed_ms'), 'cell elapsed_ms', positive=True)
            arms[policy] = {row['request_id']: row for row in rows if row['cell'] == cell['cell']}
        reference, candidate = (arms[policy] for policy in plan['policies'])
        metrics, by_kind = {}, {}
        for kind in ('video', 'image'):
            ids = [key for key, row in reference.items() if row['kind'] == kind]
            if not ids:
                continue
            if kind == 'image':
                for key in ids:
                    image_pair(reference[key], candidate[key])
            times = [[arm[key]['finish_ms'] - arm[key]['arrival_ms'] for key in ids] for arm in (reference, candidate)]
            ref_mean = finite(statistics.fmean(times[0]), 'reference mean latency', positive=True)
            attainment = [sum(arm[key]['finish_ms'] <= arm[key]['deadline_ms'] for key in ids) / len(ids)
                          for arm in (reference, candidate)]
            metrics[kind + '_mean_latency_ratio'] = statistics.fmean(times[1]) / ref_mean
            metrics[kind + '_attainment_delta'] = attainment[1] - attainment[0]
            by_kind[kind] = dict(paired_requests=len(ids), reference_latency_ms=distribution(times[0]),
                candidate_latency_ms=distribution(times[1]), reference_attainment=attainment[0], candidate_attainment=attainment[1])
        stops = stop_samples(candidate.values())
        if stops:
            metrics['stop_p95_ms'] = percentile(stops, .95)
        cost_fields = ('dispatch_gap_ms', 'worker_operation_ms', 'timed_quanta', 'transfer_export_ms',
                       'transfers', 'transfer_bytes', 'resume_extra_ms', 'resumes')
        for row in candidate.values():
            for field in cost_fields:
                if field in row:
                    if field == 'resume_extra_ms':
                        if type(row[field]) not in (int, float) or not math.isfinite(row[field]):
                            raise ValueError('invalid signed resume excess estimate')
                    else:
                        finite(row[field], field)
        costs = {field: dict(requests_with_measurement=sum(field in row for row in candidate.values()),
                    total=sum(row[field] for row in candidate.values() if field in row)) for field in cost_fields}
        result.update(status='verified_fixed_form', metrics=metrics, by_kind=by_kind,
            preemption_stop_ms=distribution(stops), partial_costs=costs,
            policy_order=[cell['policy'] for cell in sorted(phase['cells'], key=lambda cell: cell.get('order', cell['cell']))],
            shape_ids=expected['shape_ids'], fixed_form_numeric_scope='same runtime, same CFG2; finite paired video and image digests')
    except (KeyError, ValueError, TypeError, IndexError) as error:
        result['reason'] = str(error)
    return result


def evaluate(manifest, base):
    if manifest.get('schema') != 'ditserve.p1-evidence.v1':
        raise ValueError('expected ditserve.p1-evidence.v1')
    protocol = read_json(source_file(base, manifest['protocol']))
    plan = validate_plan(read_json(source_file(base, manifest['plan'])), protocol)
    identity = p0_identity(base, manifest['p0_receipt'])
    if identity[0]['protocol_sha256'] != canonical_hash(protocol):
        raise ValueError('P0 and P1 formal protocol identities differ')
    expected = {(pair['id'], role, seed): pair for pair in plan['shape_pairs']
                for role in plan['roles'] for seed in plan['seeds']}
    supplied, hashes, run_ids = {}, set(), set()
    for item in manifest.get('runs', []):
        key = (item['pair_id'], item['role'], item['seed'])
        if key not in expected or key in supplied:
            raise ValueError('unexpected or duplicate P1 case: ' + repr(key))
        record = read_json(source_file(base, {name: item[name] for name in ('path', 'sha256')}))
        if item['sha256'] in hashes or record.get('run_id') in run_ids:
            raise ValueError('a source run cannot masquerade as independent repetitions')
        hashes.add(item['sha256']); run_ids.add(record.get('run_id'))
        supplied[key] = audit_run(record, expected[key], key[1], key[2], protocol, plan, identity)
        supplied[key]['source_sha256'] = item['sha256']
    checks = []
    for key in sorted(expected):
        checks.append(supplied.get(key, dict(pair_id=key[0], role=key[1], seed=key[2], status='missing',
            expected_requests=2 * plan['roles'][key[1]]['count'], observed_requests=0, completed_requests=0, failed_requests=0)))
    repeated = []
    for pair in plan['shape_pairs']:
        for role in plan['roles']:
            cases = [row for row in checks if row['pair_id'] == pair['id'] and row['role'] == role]
            valid = [row for row in cases if row['status'] == 'verified_fixed_form']
            names = sorted({key for row in valid for key in row['metrics']})
            intervals = {name: seed_interval([row['metrics'][name] for row in valid if name in row['metrics']], plan['statistics'])
                         for name in names}
            repeated.append(dict(pair_id=pair['id'], role=role, expected_seeds=len(plan['seeds']),
                verified_seeds=len(valid), failed_seeds=sum(row['status'] == 'blocked' for row in cases),
                missing_seeds=sum(row['status'] == 'missing' for row in cases), intervals=intervals,
                policy_orders=sorted({tuple(row['policy_order']) for row in valid})))
    shapes = []
    for sid, shape in protocol['shapes'].items():
        if shape['scope'] != 'formal':
            continue
        relevant = [row for row in checks if sid in (
            expected[(row['pair_id'], row['role'], row['seed'])]['video_shape'],
            expected[(row['pair_id'], row['role'], row['seed'])]['image_shape'])
            and (shape['kind'] == 'video' or row['role'] == 'contention')]
        verified = sum(row['status'] == 'verified_fixed_form' for row in relevant)
        shapes.append(dict(shape_id=sid, model=shape['model'], expected_cases=len(relevant), verified_cases=verified,
            status='fixed_form_evidence_complete' if relevant and verified == len(relevant) else 'pending'))
    return dict(schema='ditserve.p1-review.v1', status='pending', allows=[],
        scope='offline fixed-CFG2 evidence review, never a P1 stage-promotion receipt',
        protocol_sha256=canonical_hash(protocol), plan_sha256=canonical_hash(plan), build=identity[0]['build'],
        expected_runs=len(expected), supplied_runs=len(supplied),
        expected_requests=sum(row['expected_requests'] for row in checks),
        observed_requests=sum(row['observed_requests'] for row in checks),
        completed_requests=sum(row['completed_requests'] for row in checks),
        failed_requests=sum(row['failed_requests'] for row in checks), checks=checks, repetitions=repeated, formal_shapes=shapes,
        performance_thresholds=dict(status='pending', thresholds=plan.get('thresholds', {}),
            reason='descriptive seed intervals only; no acceptance evaluator or post-hoc threshold selection'),
        cost_decomposition=dict(status='partial',
            available_scope={'preemption_stop_ms': 'ring to controller-confirmed worker stop; includes reply/poll delay',
                'dispatch_gap_ms': 'sum of elapsed minus slowest worker operation, only no-transfer/no-pair quanta',
                'transfer_export_ms': 'source export plus write only; not complete migration',
                'resume_extra_ms': 'signed resumed model-time excess over measured steady block average; not direct restore latency'},
            missing=['per-event import and all-ranks-ready timing', 'explicit resume start/ready timestamps',
                     'planner/RPC/queueing separation', 'non-preemptible encode/collective/decode phase coverage']),
        quality=dict(status='pending', scope='finite paired same-runtime digest equality only; no FP32, cross-form or semantic claim',
                     limitations=['SD35 actual VAE dtype is not independently emitted by its current generation metadata']),
        remaining_requirements=plan.get('unresolved_requirements', []),
        interval_limit='three seeds are a small workload sample; these are not request-level or same-input timing-repeatability intervals')


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
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'expected_runs', 'supplied_runs')}))
    if any(row['status'] == 'blocked' for row in result['checks']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
