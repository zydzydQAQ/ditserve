#!/usr/bin/env python3
"""Review paired P2 measurements, including failed and unstarted planned cells.

This is a local descriptive review, not a stage receipt or a quality certificate.
The evidence manifest has schema ditserve.p2-evidence.v1, hash references to
protocol and plan, and runs [{case_id, arm, record:{path,sha256}}]. The plan has
schema ditserve.p2-plan.v1, protocol_sha256, statistics, and cases [{id,
workload_id, contrast, seed, reference, candidate}]. Each arm names a frozen
config reference and explicit phase, attempt, cell, order. Config references
are relative to the plan; run references are relative to the evidence manifest.
Missing run entries and cells stay in the planned denominator; they never
become measured zero-throughput counterfactuals. Repetitions resample seeds.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

from formal_benchmark import input_shape, shape_spec
from native_exact import checked_digest, stop_samples, validate_pair_identity
from native_metrics import percentile
from p2_contrasts import CONTRASTS, audit_native_sources, canonical_policy, validate_contrast
from stage_gates import canonical_hash, read_json, source_file


def finite(value, name, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or positive and value == 0:
        raise ValueError('invalid measured ' + name)
    return value


def sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def trace_hash(rows):
    # Frozen workload generator uses ASCII JSON, unlike protocol canonical JSON.
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def expanded(phase):
    """Match native expand_matrix's lexicographic JSON-pointer Cartesian order."""
    rows = [copy.deepcopy({key: value for key, value in phase.items() if key != 'matrix'})]
    for pointer, choices in sorted(phase.get('matrix', {}).items()):
        if not pointer.startswith('/') or not isinstance(choices, list) or not choices:
            raise ValueError('invalid frozen phase matrix')
        parts = [part.replace('~1', '/').replace('~0', '~') for part in pointer[1:].split('/')]
        next_rows = []
        for row in rows:
            for value in choices:
                item = copy.deepcopy(row)
                target = item
                for part in parts[:-1]:
                    target = target[int(part)] if isinstance(target, list) else target.setdefault(part, {})
                target[int(parts[-1]) if isinstance(target, list) else parts[-1]] = copy.deepcopy(value)
                next_rows.append(item)
                if len(next_rows) > 10000:
                    raise ValueError('matrix exceeds native cell limit')
        rows = next_rows
    return rows


def planned_arm(config, selector, protocol, seed):
    phases = [phase for phase in config.get('phases', []) if phase.get('id') == selector['phase']]
    if len(phases) != 1 or type(selector['attempt']) is not int or selector['attempt'] < 1:
        raise ValueError('explicit unique phase and positive attempt required')
    original = phases[0]
    cells = expanded(original)
    index = selector['cell']
    if type(index) is not int or not 0 <= index < len(cells):
        raise ValueError('selected cell is not in the frozen configuration')
    order = selector['order']
    if order not in ('forward', 'reverse') or order == 'reverse' and not original.get('paired', False):
        raise ValueError('selected order is not planned')
    sequence = [(i, 'forward') for i in range(len(cells))]
    if original.get('paired', False):
        sequence += [(i, 'reverse') for i in reversed(range(len(cells)))]
    phase = cells[index]
    benchmark = config.get('benchmark', {})
    formal = benchmark.get('formal_protocol', {})
    if (phase.get('kind') != 'cluster' or benchmark.get('native_serving') is not True
            or benchmark.get('exact_mode') is not True or benchmark.get('seed') != seed
            or formal.get('sha256') != canonical_hash(protocol) or formal.get('schema') != protocol.get('schema')):
        raise ValueError('configuration lacks frozen formal native/exact/seed identity')
    split, role = formal.get('split'), formal.get('role')
    if (split not in ('development', 'test') or type(seed) is not int
            or seed not in protocol.get('seeds', {}).get(split, [])):
        raise ValueError('seed/split is outside the frozen formal protocol')
    if role == 'serving':
        shape_scope = 'development' if split == 'development' else 'formal'
    elif (role == 'p3_sustained' and isinstance(benchmark.get('p3'), dict)
          and sha(benchmark['p3'].get('plan_sha256'))):
        # P3's caller separately verifies the bound plan, windows and arrivals.
        shape_scope = 'development' if split == 'development' else 'formal'
    elif role in ('preemption', 'no_preemption', 'p1_contention', 'p1_no_preemption') and split == 'development':
        shape_scope = 'formal' if role.startswith('p1_') else 'development'
    else:
        raise ValueError('formal role is inconsistent with its prompt split')
    if 'p2' in benchmark and benchmark['p2'].get('parent_role') != role:
        raise ValueError('formal role differs from the frozen parent role')
    shape_ids = formal.get('shape_ids')
    shapes = protocol.get('shapes', {})
    if (not isinstance(shape_ids, list) or not shape_ids or not all(isinstance(sid, str) for sid in shape_ids)
            or len(set(shape_ids)) != len(shape_ids)
            or any(sid not in shapes or shapes[sid].get('scope') != shape_scope for sid in shape_ids)):
        raise ValueError('formal shape scope is inconsistent with its split/role')
    bundle_hash = formal.get('profile_bundle_sha256')
    if not sha(bundle_hash) or phase.get('profile', {}).get('source', {}).get('bundle_sha256') != bundle_hash:
        raise ValueError('formal/profile bundle hash bindings differ')
    workload = phase.get('workload')
    provenance = phase.get('trace_provenance', {})
    if (not isinstance(workload, list) or not workload or benchmark.get('trace') != provenance
            or provenance.get('sha256') != trace_hash(workload)
            or provenance.get('manifest', {}).get('workload_sha256') != trace_hash(workload)
            or provenance['manifest'].get('seed') != seed):
        raise ValueError('trace inputs do not match their frozen hash and seed')
    ids = [row['request_id'] for row in workload]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate planned request identity')
    for row in workload:
        if row['kind'] not in ('image', 'video') or row['class'] not in formal.get('shape_ids', []):
            raise ValueError('planned modality or shape is outside formal identity')
        if input_shape(row['spec']) != shape_spec(protocol, row['class']):
            raise ValueError('planned request differs from its frozen shape')
        finite(row['arrival_ms'], 'arrival_ms'); finite(row['deadline_ms'], 'deadline_ms')
        if row['deadline_ms'] <= row['arrival_ms']:
            raise ValueError('nonpositive planned request SLO')
    workers = config['workers']
    uuids = []
    for slot in phase['gpus']:
        video_uuid, image_uuid = (workers[slot[kind]]['gpu_uuid'] for kind in ('video', 'image'))
        if not video_uuid or video_uuid != image_uuid or video_uuid in uuids:
            raise ValueError('slots must bind distinct co-resident video/image GPU UUIDs')
        uuids.append(video_uuid)
    if not uuids or phase['profile']['source'].get('gpu_uuids') != uuids or benchmark.get('gpus') != len(uuids):
        raise ValueError('profile and worker allocations differ')
    return dict(config=config, selector=selector, phase=phase, sequence=sequence,
                position=sequence.index((index, order)), workload=workload, gpu_uuids=uuids)


def counts(planned, rows, state):
    result = {}
    for kind in ('image', 'video'):
        expected = [row for row in planned if row['kind'] == kind]
        observed = [row for row in rows if row.get('kind') == kind]
        complete = [row for row in observed if row.get('status') == 'completed']
        result[kind] = dict(planned=len(expected), observed=len(observed), completed=len(complete),
            failed=sum(row.get('status') == 'failed' for row in observed),
            interrupted=sum(row.get('status') not in ('completed', 'failed') for row in observed),
            not_started=len(expected) if state == 'not_started' else 0,
            unobserved_after_failure=max(0, len(expected) - len(observed)) if state == 'failed' else 0)
    return result


def produced_identity(row, planned, arm, protocol):
    if row.get('status') != 'completed':
        return None
    if not row.get('outputs'):
        raise ValueError('completed request has no measured output')
    output = row['outputs'][-1]
    generation, artifact = output.get('generation', {}), output.get('output', {})
    spec = planned['spec']
    shape = protocol['shapes'][planned['class']]
    wanted = ([1, shape['frames'], shape['height'], shape['width'], 3] if row['kind'] == 'video'
              else [1, shape['height'], shape['width'], 3])
    required = {'model_path', 'model_manifest_sha256', 'model_integrity', 'runtime_identity',
                'dtype', 'text_encoder_dtype', 'attention', 'engine', 'revision'}
    if row['kind'] == 'video':
        required |= {'vae_dtype', 'xdit_revision'}
        # Reuse native exact's full video input/runtime validation, without
        # asserting cross-SP hash equivalence from an exact_mode label.
        validate_pair_identity(row, row)
    else:
        required |= {'seed_protocol', 'batch_protocol'}
    if (not required <= generation.keys() or generation.get('approximate') is not False
            or not generation.get('runtime_identity') or not generation.get('engine')
            or generation.get('model_integrity') != 'manifest_verified'):
        raise ValueError('missing verified model/runtime/input identity')
    for field in ('prompt', 'negative_prompt', 'seed', 'steps', 'frames', 'height', 'width', 'guidance_scale'):
        if field not in spec or generation.get(field) != spec[field]:
            raise ValueError('measured generation differs from frozen input: ' + field)
    model = shape['model']
    binding = arm['config']['benchmark']['model_manifests'][model]
    if not sha(binding.get('sha256')) or generation['model_manifest_sha256'] != binding['sha256']:
        raise ValueError('measured weights differ from configured manifest')
    workers = arm['config']['workers']
    for slot in arm['phase']['gpus']:
        worker = workers[slot[row['kind']]]['config']
        if (worker.get('model_manifest_sha256') != binding['sha256']
                or worker.get('model_manifest_path') != binding['path']
                or worker.get('model_path') != generation['model_path']):
            raise ValueError('worker model binding differs from measured output')
    if (artifact.get('shape') != wanted or not isinstance(artifact.get('dtype'), str)
            or artifact.get('finite') is not True or artifact.get('output_mode') not in ('check', 'digest')):
        raise ValueError('decoded output has wrong requested dimensions or lacks finite verification')
    if artifact['output_mode'] != planned.get('output_mode', arm['phase'].get('output_mode', 'file')):
        raise ValueError('produced output mode differs from frozen request')
    digest = checked_digest(row) if artifact['output_mode'] == 'digest' else None
    identity = {key: generation[key] for key in required}
    return dict(model=model, identity=identity, digest=digest,
                output_dtype=artifact['dtype'],
                generation={key: value for key, value in generation.items() if key != 'request_id'})


def audit_arm(record, arm, protocol, *, audited_native_sources=None):
    """Retain frozen denominators; an internal caller may provide singleton pins."""
    selector, workload = arm['selector'], arm['workload']
    rows, result = [], dict(status='not_started', by_kind=counts(workload, [], 'not_started'),
                           metrics={}, identities={}, outputs={}, stops=0, forms=[])
    if record is None:
        return result
    result['run_id'] = record.get('run_id')
    try:
        if record.get('schema') != 'ditserve.run.v1' or record.get('config') != arm['config'] or not record.get('run_id'):
            raise ValueError('run differs from frozen planned configuration')
        config_hash = canonical_hash(arm['config'])
        # Older evidence may omit redundant hashes; a supplied hash must bind
        # the actual frozen config, rather than merely agree with another tag.
        if 'config_sha256' in record and record['config_sha256'] != config_hash:
            raise ValueError('run config hash differs from the frozen configuration')
        source = record.get('source', {})
        if not sha(source.get('core_sha256')) or not source.get('files') or not all(sha(h) for h in source['files'].values()):
            raise ValueError('run lacks C++ and Python source identity')
        result['native_source_audit'] = audit_native_sources(source.get('native_files'), audited_native_sources)
        phases = [phase for phase in record.get('phases', [])
                  if phase.get('id') == selector['phase'] and phase.get('attempt') == selector['attempt']]
        if len(phases) > 1:
            raise ValueError('duplicate selected attempt')
        if not phases:
            return result
        phase = phases[0]
        cells = phase.get('cells', [])
        if len(cells) > len(arm['sequence']):
            raise ValueError('extra unplanned cell observations')
        for position, cell in enumerate(cells):
            expected_index, expected_order = arm['sequence'][position]
            if (cell.get('cell') != expected_index or cell.get('order', expected_order) != expected_order
                    or 'order' not in cell and (cell.get('status') != 'failed' or position != len(cells) - 1)):
                raise ValueError('recorded cell order is not the frozen execution prefix')
        rows = [row for row in record.get('requests', []) if row.get('phase') == selector['phase']
                and row.get('attempt') == selector['attempt'] and row.get('cell') == selector['cell']
                and row.get('order') == selector['order']]
        if len(cells) <= arm['position']:
            if rows:
                raise ValueError('unstarted cell unexpectedly contains request observations')
            return result
        cell = cells[arm['position']]
        status = cell.get('status')
        if status not in ('completed', 'failed'):
            raise ValueError('cell has no terminal measured outcome')
        result['status'] = status
        result['by_kind'] = counts(workload, rows, status)
        if status == 'failed':
            result['error'] = cell.get('error', phase.get('error'))
        if cell.get('policy') is not None and canonical_policy(cell['policy']) != canonical_policy(arm['phase']['policy']):
            raise ValueError('executed policy differs from selected frozen cell')
        sessions = [session for session in record.get('budget', {}).get('sessions', [])
                    if session.get('session_id') == phase.get('session_id')]
        if len(sessions) != 1:
            raise ValueError('selected attempt has no unique measured allocation session')
        session = sessions[0]
        identity = session.get('identity', {})
        for owner in (identity, phase.get('charge', {}).get('identity', {})):
            if 'config_sha256' in owner and owner['config_sha256'] != config_hash:
                raise ValueError('session config hash differs from the frozen configuration')
        if (identity.get('gpu_uuids') != arm['gpu_uuids'] or identity.get('attempt') != selector['attempt']
                or identity.get('run_id') != record['run_id'] or session.get('phase') != selector['phase']
                or session.get('devices') != len(arm['gpu_uuids'])
                or status == 'completed' and cell.get('gpus') != len(arm['gpu_uuids'])):
            raise ValueError('measured GPU allocation/budget differs from frozen workers/profile')
        expected = {row['request_id']: row for row in workload}
        observed = {row['request_id']: row for row in rows}
        if len(observed) != len(rows) or not observed.keys() <= expected.keys():
            raise ValueError('duplicate or unexpected request observation')
        if status == 'completed' and (observed.keys() != expected.keys() or any(row.get('status') != 'completed' for row in rows)):
            raise ValueError('completed cell omits or fails planned requests')
        identities, outputs, forms = {}, {}, set()
        for row in rows:
            planned = expected[row['request_id']]
            if any(row.get(key) != value for key, value in planned.items()):
                raise ValueError('measured request changed a frozen input, SLO or dataset identity')
            if row.get('status') == 'completed':
                finish = finite(row.get('finish_ms'), 'finish_ms')
                if finish < planned['arrival_ms']:
                    raise ValueError('request finished before arrival')
                output = produced_identity(row, planned, arm, protocol)
                previous = identities.setdefault(output['model'], output['identity'])
                if previous != output['identity']:
                    raise ValueError('runtime or model identity changes within one cell')
                outputs[row['request_id']] = output
            forms.update(row.get('forms', []))
        result.update(identities=identities, outputs=outputs, stops=len(stop_samples(rows)), forms=sorted(forms))
        elapsed = cell.get('elapsed_ms')
        if elapsed is not None:
            finite(elapsed, 'cell elapsed_ms', positive=True)
            if any(row.get('finish_ms', 0) > elapsed + 1e-6 for row in rows):
                raise ValueError('cell elapsed time excludes the observed request tail')
        elif status == 'completed':
            raise ValueError('completed cell lacks measured elapsed time')
        result['elapsed_ms'] = elapsed
        result['serving_gpu_seconds'] = elapsed * len(arm['gpu_uuids']) / 1000 if elapsed is not None else None
        result['gpu_time_scope'] = 'replay elapsed times allocated GPUs; excludes initialization/warmup, not active-GPU utilization'
        for kind, count in result['by_kind'].items():
            complete = [row for row in rows if row['kind'] == kind and row.get('status') == 'completed']
            ontime = sum(row['finish_ms'] <= row['deadline_ms'] for row in complete)
            count['ontime'] = ontime
            n = count['planned']
            if not n:
                continue
            values = [row['finish_ms'] - row['arrival_ms'] for row in complete]
            result['metrics'][kind] = dict(completion=count['completed'] / n, attainment=ontime / n,
                ontime_rps=1000 * ontime / elapsed if elapsed is not None else None,
                ontime_per_gpu_second=ontime / result['serving_gpu_seconds'] if elapsed is not None else None,
                mean_latency_ms=statistics.fmean(values) if len(values) == n else None,
                p95_latency_ms=percentile(values, .95) if len(values) == n else None)
        result['source'] = source
    except (ValueError, KeyError, TypeError, IndexError) as error:
        result['status'] = 'invalid'
        result['reason'] = str(error)
        result['metrics'] = {}
        result['by_kind'] = counts(workload, rows, result.get('status'))
    return result


def paired(reference, candidate, expected_models):
    output = dict(status='pending', metrics={}, numeric_equivalence='pending', mismatched_digests=0)
    if reference['status'] not in ('completed', 'failed') or candidate['status'] not in ('completed', 'failed'):
        return output
    if reference['source'] != candidate['source']:
        raise ValueError('paired arms have different frozen C++/Python/native sources')
    common_models = reference['identities'].keys() & candidate['identities'].keys()
    if any(reference['identities'][model] != candidate['identities'][model] for model in common_models):
        raise ValueError('paired executed model/runtime identities differ')
    for request_id in reference['outputs'].keys() & candidate['outputs'].keys():
        left, right = reference['outputs'][request_id], candidate['outputs'][request_id]
        if left['generation'] != right['generation'] or left['output_dtype'] != right['output_dtype']:
            raise ValueError('paired executed generation identities differ')
        if left['digest'] is not None and right['digest'] is not None and left['digest'] != right['digest']:
            output['mismatched_digests'] += 1
    for kind in ('image', 'video'):
        if kind not in reference['metrics'] or kind not in candidate['metrics']:
            continue
        left, right = reference['metrics'][kind], candidate['metrics'][kind]
        metrics = {}
        for name in ('completion', 'attainment', 'ontime_rps', 'ontime_per_gpu_second'):
            if left[name] is not None and right[name] is not None:
                metrics[name + '_delta'] = right[name] - left[name]
        for name in ('ontime_rps', 'ontime_per_gpu_second', 'mean_latency_ms', 'p95_latency_ms'):
            metrics[name + '_ratio'] = right[name] / left[name] if left[name] and right[name] is not None else None
        output['metrics'][kind] = metrics
    # Exact digest agreement is reported only over complete measured outputs.
    all_outputs = list(reference['outputs'].values()) + list(candidate['outputs'].values())
    all_complete = reference['status'] == candidate['status'] == 'completed'
    if all_complete and all_outputs and all(row['digest'] is not None for row in all_outputs):
        output['numeric_equivalence'] = 'digest_mismatch' if output['mismatched_digests'] else 'paired_digest_agreement'
    output['metrics']['allocation'] = {}
    for name in ('elapsed_ms', 'serving_gpu_seconds'):
        left, right = reference.get(name), candidate.get(name)
        output['metrics']['allocation'][name + '_ratio'] = right / left if left and right is not None else None
    output['status'] = 'measured' if expected_models <= common_models else 'pending'
    output['runtime_identity_scope'] = dict(observed_models=sorted(common_models),
        reference_models_without_output=sorted(expected_models - reference['identities'].keys()),
        candidate_models_without_output=sorted(expected_models - candidate['identities'].keys()))
    return output


def interval(values, expected, settings):
    result = dict(seed_count=len(values), expected_seeds=expected, mean=statistics.fmean(values) if values else None,
                  ci95=None, status='pending', resampling_unit='independent paired seed')
    if len(values) == expected and len(values) >= settings['minimum_independent_seeds']:
        rng = random.Random(settings['bootstrap_seed'])
        means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(settings['bootstrap_samples'])]
        tail = (1 - settings['confidence']) / 2
        result.update(ci95=[percentile(means, tail), percentile(means, 1 - tail)], status='measured')
    return result


def measurement_scope(arm):
    benchmark = arm['config']['benchmark']
    formal = benchmark['formal_protocol']
    by_class = {}
    for row in arm['workload']:
        entry = by_class.setdefault(row['class'], dict(count=0, kind=row['kind'], effective_slo_scales=[], slo_ms=[]))
        entry['count'] += 1
        for field in ('effective_slo_scale', 'slo_ms'):
            target = 'effective_slo_scales' if field == 'effective_slo_scale' else field
            if row.get(field) not in entry[target]:
                entry[target].append(row.get(field))
    return dict(split=formal['split'], source_role=formal['role'],
        population='development_diagnostic' if formal['split'] == 'development' or formal['role'] != 'serving'
                   else 'test_serving_source',
        parent_controls=benchmark.get('p2', {}).get('parent_controls', benchmark.get('p0_diagnostic_controls',
            benchmark.get('p1', {}).get('controls'))),
        actual_classes=by_class, actual_windows=[row['parameters'] for row in arm['phase']['trace_provenance']['manifest']['windows']],
        actual_lazy_preemption=arm['phase'].get('options', {}).get('lazy_preemption', True),
        formal_performance_qualified=False,
        scope='source purpose and measured controls; no unseen-dataset, full-protocol, performance or quality certification')


def evaluate(manifest, base):
    if manifest.get('schema') != 'ditserve.p2-evidence.v1':
        raise ValueError('unknown P2 evidence schema')
    plan = read_json(source_file(base, manifest['plan']))
    plan_base = (base / manifest['plan']['path']).resolve().parent
    protocol = read_json(source_file(base, manifest['protocol']))
    if plan.get('schema') != 'ditserve.p2-plan.v1' or plan.get('protocol_sha256') != canonical_hash(protocol):
        raise ValueError('plan/protocol identity differs')
    settings = plan['statistics']
    if (settings.get('confidence') != .95 or settings.get('resampling_unit') != 'seed'
            or type(settings.get('minimum_independent_seeds')) is not int or settings['minimum_independent_seeds'] < 3
            or type(settings.get('bootstrap_samples')) is not int or not 1000 <= settings['bootstrap_samples'] <= 100000
            or type(settings.get('bootstrap_seed')) is not int):
        raise ValueError('P2 requires frozen seed-based confidence interval settings')
    cases = plan.get('cases', [])
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('plan must contain unique case IDs')
    supplied, supplied_sources = {}, {}
    for item in manifest.get('runs', []):
        key = (item['case_id'], item['arm'])
        if key in supplied or item['case_id'] not in {case['id'] for case in cases} or item['arm'] not in ('reference', 'candidate'):
            raise ValueError('duplicate or unplanned run reference')
        supplied[key] = read_json(source_file(base, item['record']))
        supplied_sources[key] = item['record']
    results, groups, signatures, observed_ids = [], {}, {}, {}
    runtime_sources, runtime_models, workload_seeds = {}, {}, {}
    for case in cases:
        seed = case['seed']
        if type(seed) is not int or case['contrast'] not in CONTRASTS:
            raise ValueError('invalid case seed/contrast')
        key = (case['workload_id'], case['contrast'])
        group = groups.setdefault(key, {})
        if seed in group:
            raise ValueError('duplicate seed within one workload contrast')
        arms = {arm: planned_arm(read_json(source_file(plan_base, case[arm]['config'])), case[arm], protocol, seed)
                for arm in ('reference', 'candidate')}
        resolved = validate_contrast(case['contrast'], arms['reference']['phase'], arms['candidate']['phase'])
        workload_identity = (key, trace_hash(arms['reference']['workload']))
        if workload_identity in workload_seeds and workload_seeds[workload_identity] != seed:
            raise ValueError('same frozen trace relabelled as independent seeds')
        workload_seeds[workload_identity] = seed
        left, right = (arms[arm]['config'] for arm in ('reference', 'candidate'))
        for field in ('workers',):
            if left[field] != right[field]:
                raise ValueError('paired worker configuration or GPU identity differs')
        for field in ('formal_protocol', 'model_manifests'):
            if left['benchmark'].get(field) != right['benchmark'].get(field):
                raise ValueError('paired protocol or model-manifest identity differs')
        # Freeze non-random conditions across seeds rather than averaging
        # unrelated loads, shapes, budgets or software into one interval.
        signature = []
        for arm in ('reference', 'candidate'):
            item = arms[arm]
            phase = {k: v for k, v in item['phase'].items() if k not in ('workload', 'trace_provenance')}
            trace = item['phase']['trace_provenance']['manifest']
            signature.append(dict(phase=phase, workers=item['config']['workers'],
                protocol=item['config']['benchmark']['formal_protocol'],
                weights=item['config']['benchmark']['model_manifests'], counts=trace['counts'],
                class_counts=trace['class_counts'], windows=[row['parameters'] for row in trace['windows']]))
        if key in signatures and signatures[key] != signature:
            raise ValueError('independent seeds change workload controls, shapes, GPU allocation or weights')
        signatures[key] = signature
        audits = {}
        for arm in ('reference', 'candidate'):
            record = supplied.get((case['id'], arm))
            if record is not None:
                locator = (record.get('run_id'), case[arm]['phase'], case[arm]['attempt'], case[arm]['cell'], case[arm]['order'])
                if locator in observed_ids and observed_ids[locator] != seed:
                    raise ValueError('one observation reused as independent seeds')
                observed_ids[locator] = seed
            audits[arm] = audit_arm(record, arms[arm], protocol)
            audits[arm]['record'] = supplied_sources.get((case['id'], arm))
        try:
            expected_models = {protocol['shapes'][row['class']]['model'] for row in arms['reference']['workload']}
            comparison = paired(audits['reference'], audits['candidate'], expected_models)
        except (ValueError, KeyError, TypeError) as error:
            comparison = dict(status='invalid', reason=str(error), metrics={}, numeric_equivalence='pending')
        if comparison['status'] == 'measured':
            audit = audits['reference']
            if runtime_sources.setdefault(key, audit['source']) != audit['source']:
                raise ValueError('independent seeds change frozen C++/Python/native sources')
            for model, identity in audit['identities'].items():
                if runtime_models.setdefault((key, model), identity) != identity:
                    raise ValueError('independent seeds change measured model/runtime identity')
        row = dict(case_id=case['id'], workload_id=case['workload_id'], contrast=case['contrast'], seed=seed,
                   resolved_options=resolved, comparison=comparison, arms=audits,
                   measurement_scope={arm: measurement_scope(arms[arm]) for arm in ('reference', 'candidate')},
                   activation=dict(reference_stops=audits['reference']['stops'], candidate_stops=audits['candidate']['stops'],
                       reference_forms=audits['reference']['forms'], candidate_forms=audits['candidate']['forms'],
                       scope='observed stops/forms; planner decision-cost attribution requires separate evidence'))
        for audit in audits.values():
            audit.pop('outputs', None)
            source = audit.pop('source', None)
            if source is not None:
                audit['source_identity'] = dict(core_sha256=source['core_sha256'],
                    python_sources_sha256=canonical_hash(source['files']),
                    native_sources_sha256=canonical_hash(source['native_files']))
        results.append(row); group[seed] = row
    repetitions = []
    for (workload, contrast), seeds in groups.items():
        if len(seeds) < settings['minimum_independent_seeds']:
            raise ValueError('plan needs at least three independent seeds per workload/contrast; supplied runs may be missing')
        metrics = {}
        for kind in ('image', 'video', 'allocation'):
            names = {name for row in seeds.values() for name in row['comparison']['metrics'].get(kind, {})}
            metrics[kind] = {}
            for name in sorted(names):
                values = [row['comparison']['metrics'].get(kind, {}).get(name)
                          if row['comparison']['status'] == 'measured' else None for row in seeds.values()]
                metrics[kind][name] = interval([value for value in values if value is not None], len(seeds), settings)
        repetitions.append(dict(workload_id=workload, contrast=contrast, seeds=sorted(seeds), intervals=metrics))
    totals = {arm: {kind: {} for kind in ('image', 'video')} for arm in ('reference', 'candidate')}
    for row in results:
        for arm, audit in row['arms'].items():
            for kind, counts_ in audit['by_kind'].items():
                for field, count in counts_.items():
                    totals[arm][kind][field] = totals[arm][kind].get(field, 0) + count
    complete = all(row['comparison']['status'] == 'measured' for row in results)
    return dict(schema='ditserve.p2-review.v1', status='measured' if complete else 'incomplete',
        scope='paired descriptive performance; neither stage promotion nor qualified/semantic goodput',
        denominator_scope='planned contrast-arm request slots; a reused arm counts within each declared contrast',
        allows=[], expected_cases=len(cases), expected_requests=sum(v['planned'] for a in totals.values() for v in a.values()),
        totals=totals, cases=results, repetitions=repetitions,
        quality=dict(status='pending', reason='independent fixed/cross-form numerical and semantic qualification required'),
        interval_limit='seed bootstrap describes workload variation; three seeds do not establish timing repeatability or broad generalization',
        source=dict(plan_sha256=manifest['plan']['sha256'], protocol_sha256=canonical_hash(protocol)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(read_json(args.manifest.read_bytes()), args.manifest.resolve().parent)
    result['source_manifest'] = dict(path=str(args.manifest.resolve()), sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'expected_cases', 'expected_requests')}))


if __name__ == '__main__':
    main()
