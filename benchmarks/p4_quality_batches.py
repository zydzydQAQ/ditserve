#!/usr/bin/env python3
"""Merge immutable, source-bound P4 v2 evaluation batches without reading arrays.

Only original evaluator batches are accepted, never prior merges. Candidate
observations cannot repeat. Batches may share an identical verified exact
reference, counted once and recorded explicitly. Missing placeholders are not
observations. Raw sources remain required retained inputs;
hashes provide provenance/integrity, not independent authentication of an issuer.
No GPU execution, downloads, installations or retention cleanup occurs here.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import p4_stream_quality as stream
from ditserve.evaluation.common import canonical_hash, seal, verify_seal

SCHEMA = 'ditserve.p4-quality-merge.v1'
INPUT_SCHEMA = 'ditserve.p4-quality-batch-inputs.v1'


def finite(value, name, *, minimum=None, maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or minimum is not None and value < minimum or maximum is not None and value > maximum):
        raise ValueError('invalid finite metric: ' + name)
    return value


def planned_row(plan, pair, arm):
    return dict(pair_id=pair['id'], arm_id=arm['id'], shape_id=pair['shape_id'],
        prompt_sha256=plan['prompts'][pair['prompt_index']]['normalized_prompt_sha256'],
        generation_seed=pair['generation_seed'], spec_sha256=canonical_hash(pair['spec']))


def validate_output(output, pair):
    if not isinstance(output, dict) or set(output) != {
            'artifact_sha256', 'array_sha256', 'shape', 'dtype', 'generation_sha256', 'path_history_sha256'}:
        raise ValueError('missing complete verified output receipt')
    if any(not stream.hex_hash(output[key]) for key in
           ('artifact_sha256', 'array_sha256', 'generation_sha256', 'path_history_sha256')):
        raise ValueError('invalid verified output hashes')
    spec = pair['spec']
    if output['shape'] != [1, spec['frames'], spec['height'], spec['width'], 3] or output['dtype'] != 'float32':
        raise ValueError('verified output differs from frozen shape/dtype')


def validate_metrics(metrics, pair, *, lpips_ready):
    mse = finite(metrics['mse'], 'mse', minimum=0., maximum=1.)
    finite(metrics['ssim'], 'ssim', minimum=-1.000001, maximum=1.000001)
    exact = mse == 0.
    if metrics.get('psnr_is_infinite') is not exact or metrics.get('exact_match') is not exact:
        raise ValueError('MSE/exact/infinite PSNR relation differs')
    if exact:
        if metrics['psnr_db'] is not None:
            raise ValueError('infinite PSNR must use null plus explicit flag')
    elif not math.isclose(finite(metrics['psnr_db'], 'PSNR'), -10 * math.log10(mse), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('PSNR differs from MSE')
    if pair['spec']['frames'] == 1:
        if metrics['temporal_delta_rmse'] is not None:
            raise ValueError('one frame cannot have temporal differences')
    else:
        finite(metrics['temporal_delta_rmse'], 'temporal RMSE', minimum=0., maximum=2.)
    status = metrics.get('lpips_status')
    if status == 'measured':
        if (not lpips_ready or type(metrics.get('lpips_frames')) is not int
                or metrics['lpips_frames'] != pair['spec']['frames'] or metrics.get('lpips_reason') is not None):
            raise ValueError('LPIPS requires verified selftest and every planned frame')
        finite(metrics['lpips_mean'], 'LPIPS', minimum=0.)
    elif (status != 'unknown' or metrics.get('lpips_mean') is not None or metrics.get('lpips_frames') != 0
            or not isinstance(metrics.get('lpips_reason'), str) or not metrics['lpips_reason']):
        raise ValueError('unavailable/partial LPIPS must remain unknown without a score')


def validate_batch(plan, protocol, batch):
    verify_seal(batch)
    if (batch.get('schema') != stream.SCHEMA or batch.get('plan_sha256') != plan['sha256']
            or batch.get('protocol_sha256') != plan['protocol_sha256']):
        raise ValueError('batch schema or frozen plan/protocol differs')
    for flag in ('qualification_eligible', 'superiority_qualified'):
        if batch.get(flag) is not False:
            raise ValueError('an offline batch cannot grant P4 qualification')
    retention = batch.get('retention', {})
    if retention.get('dependencies_closed') is not False or retention.get('deletion_authorized') is not False:
        raise ValueError('raw retention dependencies must remain open')
    state = batch['lpips']['adapter']
    if type(state.get('requested')) is not bool or state.get('status') not in ('unknown', 'ready'):
        raise ValueError('invalid LPIPS adapter state')
    if state['status'] == 'ready':
        if (not state['requested'] or state.get('reason') is not None
                or stream.lpips_receipt_identity(state['receipt']) != state['metric_identity']):
            raise ValueError('LPIPS receipt/metric identity differs')
    elif (state.get('receipt') is not None or state.get('metric_identity') is not None
            or not isinstance(state.get('reason'), str) or not state['reason']):
        raise ValueError('unknown LPIPS cannot carry a ready metric identity')
    identity = stream.metric_identity(state)
    if batch['metric_identity'] != identity or batch['implementation'] != identity['implementation']:
        raise ValueError('batch metric implementation/transform/runtime identity differs')
    evidence = batch['evidence']
    index = stream.index_evidence(evidence, plan)
    if (batch.get('evidence_sha256') != evidence['sha256']
            or batch.get('source_binding') != stream.source_binding(plan, evidence, identity)):
        raise ValueError('batch source/core/model/metric binding differs')
    by_key = {}
    for row in batch['pair_results']:
        key = row['pair_id'], row['arm_id']
        if key in by_key:
            raise ValueError('duplicate pair/arm within batch')
        by_key[key] = row
    planned = {(pair['id'], arm['id']) for pair in plan['pairs'] for arm in plan['arms']}
    if set(by_key) != planned:
        raise ValueError('batch must preserve the entire planned denominator')
    for pair in plan['pairs']:
        reference_key = pair['id'], 'exact'
        reference = by_key[reference_key]
        for arm in plan['arms']:
            key = pair['id'], arm['id']
            row = by_key[key]
            if any(row.get(k) != v for k, v in planned_row(plan, pair, arm).items()):
                raise ValueError('batch prompt/seed/shape/arm identity differs')
            if (row.get('submitted_source') != stream.source_entry(index.get(key))
                    or row.get('reference_source') != stream.source_entry(index.get(reference_key))
                    or row.get('evidence_sha256') != evidence['sha256']):
                raise ValueError('observation source differs from sealed evidence')
            status = row.get('status')
            if key not in index:
                if status != 'missing':
                    raise ValueError('an unsubmitted arm cannot be an observation')
            elif status not in ('blocked', 'failed', 'measured'):
                raise ValueError('submitted failures cannot disappear from the denominator')
            if status == 'blocked' and reference_key in index:
                raise ValueError('blocked exact-reference reason contradicts evidence')
            if status != 'measured':
                if any(k in row for k in ('metrics', 'actual_fold_count', 'verified_output', 'reference_output')):
                    raise ValueError('failed/missing observations cannot retain measured values')
                continue
            count = row.get('actual_fold_count')
            if (type(count) is not int or not 0 <= count <= plan['rule']['max_folded_steps']
                    or arm['id'] == 'exact' and count != 0
                    or arm['id'].startswith('fixed-') and count != int(arm['id'][6:])):
                raise ValueError('measured actual fold count violates its frozen arm')
            validate_output(row['verified_output'], pair)
            validate_output(row['reference_output'], pair)
            if (reference['status'] != 'measured' or row['reference_output'] != reference['verified_output']
                    or row['run_sha256'] != row['submitted_source']['run_sha256']
                    or row['artifact_sha256'] != row['verified_output']['artifact_sha256']):
                raise ValueError('paired result uses a conflicting exact reference or output')
            validate_metrics(row['metrics'], pair, lpips_ready=state['status'] == 'ready')
            if (arm['id'] == 'exact' or row['verified_output']['array_sha256'] == row['reference_output']['array_sha256']):
                if not row['metrics']['exact_match'] or row['metrics']['ssim'] != 1.:
                    raise ValueError('identity output contradicts metric values')
                if row['metrics']['lpips_status'] == 'measured' and row['metrics']['lpips_mean'] != 0.:
                    raise ValueError('identical inputs must have zero LPIPS distance')
    recomputed = stream.summarize_results(plan, by_key, state)
    for key, value in recomputed.items():
        if batch.get(key) != value:
            raise ValueError('batch summary differs from its observations: ' + key)
    return by_key


def reference_identity(row):
    """All verified observation identity, excluding the batch manifest itself."""
    if row['arm_id'] != 'exact' or row['status'] != 'measured' or row['actual_fold_count'] != 0:
        raise ValueError('only a measured exact reference can be shared between batches')
    keys = ('pair_id', 'arm_id', 'shape_id', 'prompt_sha256', 'generation_seed', 'spec_sha256',
            'status', 'actual_fold_count', 'submitted_source', 'reference_source', 'run_sha256',
            'artifact_sha256', 'verified_output', 'reference_output', 'metrics')
    return {key: row[key] for key in keys}


def merge_results(plan, protocol, batches):
    """All summaries/cluster bootstrap are recomputed; no batch CI averaging."""
    stream.validate_plan(plan, protocol)
    results = {(pair['id'], arm['id']): dict(planned_row(plan, pair, arm),
        status='missing', reason='no submitted run in supplied batches', submitted_source=None,
        reference_source=None, evidence_sha256=None) for pair in plan['pairs'] for arm in plan['arms']}
    binding, state, identity, lineage = None, None, None, []
    state_batch = None
    seen_batches, observed, references = set(), {}, {}
    for batch in batches:
        rows = validate_batch(plan, protocol, batch)
        if batch['sha256'] in seen_batches:
            raise ValueError('duplicate batch receipt')
        seen_batches.add(batch['sha256'])
        if binding is None:
            binding, identity = batch['source_binding'], batch['metric_identity']
        elif batch['source_binding'] != binding or batch['metric_identity'] != identity:
            raise ValueError('batches have different core/source/runtime/model/metric identity')
        # Equivalent metric identities can have different local weight paths,
        # selftest rounding or unavailable reasons. Select a stable representative
        # while retaining each original selftest receipt in the lineage.
        if state_batch is None or batch['sha256'] < state_batch:
            state_batch, state = batch['sha256'], batch['lpips']['adapter']
        for key, row in rows.items():
            if row['submitted_source'] is None:
                continue
            if key in observed:
                if key[1] != 'exact' or reference_identity(results[key]) != reference_identity(row):
                    raise ValueError('duplicate or conflicting pair/arm observation across batches: ' + str(key))
                references[key].append(dict(batch_sha256=batch['sha256'], evidence_sha256=batch['evidence_sha256']))
                if batch['sha256'] < observed[key]:
                    observed[key], results[key] = batch['sha256'], copy.deepcopy(row)
                continue
            observed[key] = batch['sha256']
            results[key] = copy.deepcopy(row)
            if key[1] == 'exact':
                references[key] = [dict(batch_sha256=batch['sha256'], evidence_sha256=batch['evidence_sha256'])]
        lineage.append(dict(batch_sha256=batch['sha256'], evidence_sha256=batch['evidence_sha256'],
                            lpips_availability=dict(status=batch['lpips']['adapter']['status'],
                                                    reason=batch['lpips']['adapter']['reason']),
                            lpips_selftest_receipt=copy.deepcopy(batch['lpips']['adapter']['receipt'])))
    if binding is None:
        raise ValueError('at least one immutable evaluator batch is required')
    shared = [dict(pair_id=key[0], observation_sha256=canonical_hash(reference_identity(results[key])),
                   sources=sorted(sources, key=lambda source: source['batch_sha256']))
              for key, sources in sorted(references.items()) if len(sources) > 1]
    result = stream.summarize_results(plan, results, state)
    result.update(schema=SCHEMA, plan_sha256=plan['sha256'], protocol_sha256=plan['protocol_sha256'],
        source_binding=copy.deepcopy(binding), metric_identity=copy.deepcopy(identity),
        implementation=identity['implementation'], merge_implementation_sha256=stream.file_hash(Path(__file__)),
        batches=sorted(lineage, key=lambda row: row['batch_sha256']),
        reference_reuse=dict(shared_exact_anchors=len(shared),
            repeated_reference_observations_not_counted=sum(len(row['sources']) - 1 for row in shared),
            anchors=shared, scope='only identical verified measured exact references; each planned exact arm counted once'),
        merge_scope='original immutable batches only; candidate duplicates rejected; identical measured exact references may be shared; prompt clusters retain all seeds/shapes; no raw cleanup')
    return seal(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=stream.formal.DEFAULT_PROTOCOL)
    parser.add_argument('--batches', required=True, type=Path, help='sealed manifest of explicit {path,sha256} byte-bound batches')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = {name: stream.bound_json(getattr(args, name)) for name in ('plan', 'protocol', 'batches')}
    plan, protocol, manifest = (inputs[name][0] for name in ('plan', 'protocol', 'batches'))
    verify_seal(manifest)
    if manifest.get('schema') != INPUT_SCHEMA or manifest.get('plan_sha256') != plan['sha256']:
        raise ValueError('batch input manifest does not bind the plan')
    sources = []
    def batches():
        for entry in manifest['batches']:
            if set(entry) != {'path', 'sha256'} or not stream.hex_hash(entry['sha256']):
                raise ValueError('each batch requires an explicit path and byte SHA256')
            path = stream.local_path(entry['path'], args.batches.parent)
            value, digest, before = stream.bound_json(path)
            if digest != entry['sha256']:
                raise ValueError('batch file byte SHA256 differs')
            sources.append((path, digest, before))
            yield value
    result = merge_results(plan, protocol, batches())
    if any(stream.fingerprint(path) != before for path, _, before in sources) or any(
            stream.fingerprint(getattr(args, name)) != value[2] for name, value in inputs.items()):
        raise ValueError('input changed during merge')
    result['input_file_sha256'] = {name: value[1] for name, value in inputs.items()}
    result['batch_file_sources'] = [dict(path=str(path), sha256=digest) for path, digest, _ in sources]
    stream.formal.write_new(args.output, seal(result))
    print(json.dumps(dict(status=result['status'], qualification_eligible=False, output=str(args.output))))


if __name__ == '__main__':
    main()
