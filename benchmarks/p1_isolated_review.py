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
import copy
import hashlib
import json
from native_config_identity import config_identity, match_native_config
from pathlib import Path
import statistics

import formal_benchmark as formal
import p1_costs
import p1_isolated_batch as isolated
from native_exact import checked_digest, stop_samples, validate_pair_identity
from p1_benchmark import validate_plan
from p1_qualification import finite, seed_interval
from stage_gates import source_file
from ditserve.evaluation.common import canonical_hash, seal

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
        result['controller_costs'] = p1_costs.summarize(record)
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
