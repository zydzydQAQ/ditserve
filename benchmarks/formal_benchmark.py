#!/usr/bin/env python3
"""Generate frozen mixed-resolution studies from independently measured shapes.

No downloads, submission, synthetic performance, or GPU qualification occurs here.
"""
import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

from cluster_config import MODELS, allocated_uuids, make_workers
from native_study import build_calibration_config
from native_serving import (
    IN_FRAMEWORK_COMPARISON_DECLARATION,
    IN_FRAMEWORK_MECHANISM_DISPLAY,
    STATIC_POLICY,
    SUITES,
    build_serving_config,
    in_framework_main_policies,
    partition_image_gpus,
)
from fit_native_profile import build_profile
from ditserve.evaluation.common import canonical_hash, read_json, seal, verify_seal
from ditserve.evaluation.workloads import build_trace, validate_trace

DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / 'configs/benchmarks/formal-mixed-v1.json'
DEFAULT_V2_PROTOCOL = Path(__file__).resolve().parents[1] / 'configs/benchmarks/formal-mixed-v2.json'
SUPPORTED_SCHEMAS = ('ditserve.formal-protocol.v1', 'ditserve.formal-protocol.v2')
BUNDLE_SCHEMA = 'ditserve.shape-profiles.v1'
HEX64 = re.compile(r'[0-9a-f]{64}')
TRACE_BUDGET_RULE = dict(schema='ditserve.trace-budget.v1', per_cell_startup_seconds=600,
    timing_margin=1.5,
    drain_basis='sum of per-request calibrated SP1 service; conservative planning estimate, not a measured upper bound')


def apply_trace_budget(config, rule=TRACE_BUDGET_RULE):
    """Keep long, low-load traces inside the budget without altering arrivals."""
    phase = config['phases'][0]
    trace = phase['matrix']['/workload'][0]
    horizon = max(row['arrival_ms'] for row in trace)
    serial_work = sum(row['service_ms'] for row in trace)
    per_cell_seconds = (horizon + serial_work) / 1000 * rule['timing_margin'] + rule['per_cell_startup_seconds']
    wall_seconds = math.ceil(per_cell_seconds * len(phase['matrix']['/policy']))
    config['budget']['watchdog_seconds'] = max(config['budget']['watchdog_seconds'], wall_seconds)
    config['budget']['limit_gpu_seconds'] = config['budget']['watchdog_seconds'] * config['benchmark']['gpus']
    planning = dict(trace_horizon_ms=horizon, serial_service_estimate_ms=serial_work,
        rule=copy.deepcopy(rule), watchdog_seconds=config['budget']['watchdog_seconds'],
        allocated_gpu_hour_limit=config['budget']['limit_gpu_seconds'] / 3600,
        scope='calibration-based execution budget; not measured performance or guaranteed completion')
    config['benchmark']['time_planning'] = planning
    return planning


def protocol(path=DEFAULT_PROTOCOL):
    value = read_json(path)
    if value.get('schema') not in SUPPORTED_SCHEMAS:
        raise ValueError('unsupported formal protocol')
    return value


def genserve_sweep_cells(p):
    """Return the 9-cell GenServe-style anchor + single-factor sweep configurations for v2."""
    if p.get('schema') != 'ditserve.formal-protocol.v2':
        raise ValueError('genserve_sweep_cells requires ditserve.formal-protocol.v2')
    anchor = copy.deepcopy(p['anchor'])
    cells = [dict(anchor, sweep_dimension='anchor')]
    for scale in p['slo']['scales']:
        if scale != anchor['slo_scale']:
            cells.append(dict(anchor, slo_scale=scale, sweep_dimension='slo_scale'))
    for load_val in p['loads']:
        if load_val != anchor['load']:
            cells.append(dict(anchor, load=load_val, sweep_dimension='rate_scale'))
    for vfrac in p['video_fractions']:
        if vfrac != anchor['video_fraction']:
            cells.append(dict(anchor, video_fraction=vfrac, sweep_dimension='video_fraction'))
    cells.append(dict(anchor, resolution_distribution='skewed', sweep_dimension='resolution_distribution'))
    cells.append(dict(anchor, pattern='bursty', sweep_dimension='pattern'))
    return cells


def shape_spec(p, shape_id):
    row = p['shapes'][shape_id]
    spec = {k: row[k] for k in ('height', 'width', 'frames', 'steps', 'guidance_scale')}
    if row['kind'] == 'video':
        spec.update(cache=False, threshold=0.0, max_continuous=1, negative_prompt='')
    else:
        spec.update(batch_size=1, max_sequence_length=256, negative_prompt='blurry, low quality')
    return spec


def input_shape(spec):
    return {k: v for k, v in spec.items() if k not in ('prompt', 'seed', 'request_id')}


def split_pool(pool, p, kind, split):
    verify_seal(pool)
    expected = p['datasets'][kind]
    if pool.get('kind') != expected['kind']:
        raise ValueError('dataset kind differs from protocol')
    for key in ('revision', 'sha256'):
        if pool.get('source', {}).get(key) != expected[key]:
            raise ValueError('dataset source differs from frozen protocol: ' + key)
    rule = p['dataset_split']
    lo, hi = rule['buckets'][split]
    prior = p['prior_exposure']
    if Path(prior['manifest_file']).name != prior['manifest_file']:
        raise ValueError('exclusion manifest must be a protocol-local filename')
    exclusions = verify_seal(read_json(DEFAULT_PROTOCOL.parent / prior['manifest_file']))
    if exclusions['sha256'] != prior['manifest_sha256']:
        raise ValueError('prior exposure exclusion manifest differs from protocol')
    excluded = set(exclusions['normalized_prompt_sha256'])
    legacy_roles = set(prior['also_exclude_existing_pool_roles'])
    if pool.get('selection', {}).get('dev_distinct_prompts', 0) and any(
            not ('split' in row or 'service_split' in row) for row in pool['samples']):
        raise ValueError('historical development roles missing from source pool')
    # Collect first, so a whitespace duplicate marked dev cannot later slip
    # into test through another row whose role is test.
    excluded.update(canonical_hash(' '.join(row['prompt'].split())) for row in pool['samples']
                    if row.get('service_split', row.get('split')) in legacy_roles)
    samples, seen = [], set()
    # Normalize only for membership/deduplication; generation receives original text.
    for row in sorted(pool['samples'], key=lambda r: str(r['sample_id'])):
        normalized = ' '.join(row['prompt'].split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        known = canonical_hash(normalized) in excluded
        if known and split != 'development':
            continue
        bucket = int(canonical_hash([rule['salt'], normalized]), 16) % 100
        if known and split == 'development' or lo <= bucket < hi:
            samples.append(copy.deepcopy(row))
    if not samples:
        raise ValueError('empty prompt split: ' + split)
    return seal(dict(kind=pool['kind'], source=pool['source'], samples=samples,
                     selection=dict(method=rule['method'], salt=rule['salt'], split=split,
                                    bucket_range=[lo, hi], parent_pool_sha256=pool['sha256'],
                                    prior_exposure_sha256=exclusions['sha256'],
                                    sample_count=len(samples), prompt_transform='none')))


def split_pools(pools, p, split):
    return {kind: split_pool(pools[kind], p, kind, split) for kind in ('image', 'video')}


def binding(p, split, role, ids, pools, bundle=None):
    return dict(schema=p['schema'], id=p['id'], sha256=canonical_hash(p), split=split, role=role,
                shape_ids=list(ids), profile_bundle_sha256=None if bundle is None else bundle['sha256'],
                dataset_split_sha256={kind: pool['sha256'] for kind, pool in pools.items()})


def bind_manifests(config, model, manifests):
    for name, worker in config['workers'].items():
        key = model if worker['backend'] == 'wan' else 'sd35'
        entry = manifests.get(key, {})
        path, digest = entry.get('path'), entry.get('sha256')
        if not isinstance(path, str) or not path.startswith('/shared/data/DITSERVE/models/') or '..' in Path(path).parts:
            raise ValueError('model manifest must be in shared DITSERVE model directory: ' + key)
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise ValueError('missing model manifest SHA256: ' + key)
        if 'model_path' in entry:
            model_path = entry['model_path']
            if (not isinstance(model_path, str) or not model_path.startswith('/shared/data/DITSERVE/models/')
                    or '..' in Path(model_path).parts):
                raise ValueError('explicit model snapshot must be inside shared DITSERVE models: ' + key)
            worker['config']['model_path'] = model_path
        worker['config'].update(model_manifest_path=path, model_manifest_sha256=digest)
    config['benchmark']['model_manifests'] = copy.deepcopy(manifests)


def preflight(p, model, pools, manifests, scope):
    """Validate input capacity on CPU before expensive allocation/calibration."""
    if model not in MODELS or scope not in ('development', 'formal'):
        raise ValueError('invalid input preflight model/scope')
    declared = {'workers': {'video': {'backend': 'wan', 'config': {}},
                            'image': {'backend': 'sd35', 'config': {}}}, 'benchmark': {}}
    bind_manifests(declared, model, manifests)
    split = 'test' if scope == 'formal' else 'development'
    if scope == 'formal':
        count = p['requests_per_cell']['test']
        windows = [(count, fraction) for fraction in p['video_fractions']]
    else:
        windows = [(p['p0_diagnostics'][name]['count'], p['p0_diagnostics'][name]['video_fraction'])
                   for name in ('contention', 'no_preemption')]
    needs = {'video': max(round(count * fraction) for count, fraction in windows),
             'image': max(count - round(count * fraction) for count, fraction in windows)}
    checked = {}
    for selected_split, required in (('calibration', {'video': 1, 'image': 1}), (split, needs)):
        checked[selected_split] = {}
        for kind in ('video', 'image'):
            selected = split_pool(pools[kind], p, kind, selected_split)
            available = len(selected['samples'])
            if available < required[kind]:
                raise ValueError(f'insufficient distinct {kind} prompts in {selected_split}: '
                                 f'{available} available, {required[kind]} required after exclusions')
            checked[selected_split][kind] = dict(available=available, required=required[kind],
                split_sha256=selected['sha256'], parent_pool_sha256=pools[kind]['sha256'])
    return dict(schema='ditserve.formal-preflight.v1', status='passed', model=model, scope=scope,
                protocol_sha256=canonical_hash(p), pools=checked, model_manifests=copy.deepcopy(manifests),
                scientific_qualification='pending; input counts and declarations only, no GPU or weight verification')


def calibration(p, model, uuids, pools, manifests, video_shape, image_shape):
    ids = [video_shape, image_shape]
    if p['shapes'][video_shape]['model'] != model or p['shapes'][image_shape]['kind'] != 'image':
        raise ValueError('calibration shape/model mismatch')
    selected = split_pools(pools, p, 'calibration')
    specs = [dict(shape_spec(p, sid), prompt=selected[kind]['samples'][0]['prompt'], seed=101)
             for sid, kind in zip(ids, ('video', 'image'))]
    config = build_calibration_config(model, uuids, video_spec_override=specs[0], image_spec_override=specs[1])
    config['benchmark']['formal_protocol'] = binding(p, 'calibration', 'calibration', ids, selected)
    # Preserve legacy experiment parsing in fit_native_profile; unique run IDs
    # and protocol shape identity distinguish each independently fitted run.
    bind_manifests(config, model, manifests)
    return config


def assemble(p, model, records, runtime_facts):
    """Assemble independent calibration outputs without refitting across shapes."""
    entries, uuids, manifests = {}, None, None
    for record in records:
        meta = record.get('config', {}).get('benchmark', {})
        identity = meta.get('formal_protocol', {})
        if identity.get('sha256') != canonical_hash(p) or identity.get('role') != 'calibration' or identity.get('split') != 'calibration':
            raise ValueError('calibration lacks this frozen protocol identity')
        ids = identity['shape_ids']
        if len(ids) != 2 or p['shapes'][ids[0]]['model'] != model:
            raise ValueError('calibration shape/model mismatch')
        core = record.get('source', {}).get('core_sha256', '')
        if not HEX64.fullmatch(core):
            raise ValueError('calibration requires frozen core SHA256')
        current_manifests = meta.get('model_manifests')
        if not current_manifests or manifests is not None and current_manifests != manifests:
            raise ValueError('calibration model manifests are missing or changed')
        manifests = current_manifests
        for row in record['requests']:
            sid = ids[0] if row['kind'] == 'video' else ids[1]
            if input_shape(row['spec']) != shape_spec(p, sid):
                raise ValueError('measured request shape differs from protocol')
        fitted = build_profile(record, runtime_facts, MODELS[model]['name'])
        current = fitted['source']['gpu_uuids']
        if not current or uuids is not None and current != uuids:
            raise ValueError('calibration GPU placement differs')
        uuids = current
        for sid, klass in zip(ids, (model + '-preflight', 'sd35-preflight')):
            if sid in entries:
                raise ValueError('duplicate shape calibration; aggregate repetitions explicitly before assembly')
            cost = copy.deepcopy(fitted['classes'][klass])
            if cost['kind'] == 'video':
                cost['links'] = copy.deepcopy(fitted['links'])
            entries[sid] = dict(spec=shape_spec(p, sid), cost=cost,
                source=dict(run_sha256=fitted['source']['run_sha256'], run_id=record['run_id'],
                            core_sha256=core, gpu_uuids=current, fit=fitted['source'],
                            resize_ms=fitted['resize_ms'], quantum_overhead_ms=fitted['quantum_overhead_ms']))
    if not entries:
        raise ValueError('no completed calibration records')
    if len({entry['source']['core_sha256'] for entry in entries.values()}) != 1:
        raise ValueError('calibration runtime core changed between shapes')
    return seal(dict(schema=BUNDLE_SCHEMA, model=model, protocol_sha256=canonical_hash(p),
                     gpu_uuids=uuids, classes=entries, model_manifests=manifests,
                     qualification='measured calibration only; numerical, quality and serving qualification pending'))


def serving(p, model, uuids, bundle, pools, manifests, *, split='test', seed=301,
            load=None, video_fraction=None, pattern='poisson', suite='main', diagnostic=None,
            p1_plan=None, p1_pair=None, slo_scale=None, resolution_distribution=None):
    is_v2 = p.get('schema') == 'ditserve.formal-protocol.v2'
    if load is None:
        load = p['anchor']['load'] if is_v2 and 'anchor' in p else 0.6
    if video_fraction is None:
        video_fraction = p['anchor']['video_fraction'] if is_v2 and 'anchor' in p else 0.2
    verify_seal(bundle)
    if bundle.get('schema') != BUNDLE_SCHEMA or bundle.get('model') != model or bundle.get('protocol_sha256') != canonical_hash(p):
        raise ValueError('profile bundle does not match model/protocol')
    if bundle.get('gpu_uuids') != list(uuids):
        raise ValueError('profile bundle GPU UUIDs differ from exact allocation')
    if bundle.get('model_manifests') != manifests:
        raise ValueError('profile model identity differs from serving')
    if split not in ('development', 'test') or seed not in p['seeds'][split]:
        raise ValueError('seed/split is outside frozen protocol')
    if load not in p['loads'] or video_fraction not in p['video_fractions']:
        raise ValueError('load/mix is outside frozen protocol')
    patterns = {x['name']: x for x in p['patterns']}
    if pattern not in patterns:
        raise ValueError('arrival pattern is outside frozen protocol')
    if diagnostic not in (None, 'contention', 'no_preemption'):
        raise ValueError('unknown preemption diagnostic')
    if diagnostic and split != 'development':
        raise ValueError('P0 diagnostics use development shapes; formal qualification must be a separate stage')
    diagnostic_controls = p['p0_diagnostics'][diagnostic] if diagnostic else None
    pair = None
    if p1_plan is not None:
        from p1_benchmark import validate_plan
        validate_plan(p1_plan, p)
        pair = next((item for item in p1_plan['shape_pairs'] if item['id'] == p1_pair), None)
        if (pair is None or pair['model'] != model or split != p1_plan['prompt_split']
                or seed not in p1_plan['seeds'] or diagnostic not in p1_plan['roles']):
            raise ValueError('P1 pair/model/split/seed/role differs from frozen plan')
        diagnostic_controls = p1_plan['roles'][diagnostic]
    elif p1_pair is not None:
        raise ValueError('P1 pair requires a frozen plan')
    if diagnostic:
        image_scale = diagnostic_controls['image_slo_scale']
        video_scale = diagnostic_controls['video_slo_scale']
        effective_slo_scale = None
    elif slo_scale is not None or is_v2:
        allowed_scales = p['slo'].get('scales', [p['slo']['video_scale']])
        effective_slo_scale = float(p['slo'].get('default_scale', p['slo']['video_scale'])
                                    if slo_scale is None else slo_scale)
        if effective_slo_scale not in allowed_scales:
            raise ValueError('slo_scale is outside frozen protocol')
        image_scale = video_scale = effective_slo_scale
    else:
        image_scale = p['slo']['image_scale']
        video_scale = p['slo']['video_scale']
        effective_slo_scale = None
    dist_map = {d['name']: d for d in p.get('resolution_distributions', [{'name': 'uniform', 'alpha': 0.0}])}
    dist_name = ('uniform' if resolution_distribution is None else resolution_distribution)
    if dist_name not in dist_map:
        raise ValueError('resolution_distribution is outside frozen protocol')
    scope = 'development' if split == 'development' else 'formal'
    ids = [sid for sid, shape in p['shapes'].items() if shape['scope'] == scope and shape['model'] in (model, 'sd35')]
    if pair is not None:
        ids = [pair['video_shape'], pair['image_shape']]
    if any(sid not in bundle['classes'] for sid in ids):
        raise ValueError('every frozen shape needs its own measured calibration; no shape may be silently omitted')
    selected = split_pools(pools, p, split)
    classes, cost_classes = {}, {}
    for sid in ids:
        item = bundle['classes'][sid]
        spec = shape_spec(p, sid)
        if item['spec'] != spec:
            raise ValueError('profile shape identity mismatch: ' + sid)
        cost = copy.deepcopy(item['cost'])
        if cost['kind'] == 'video':
            gpu_ms = cost.get('prepare_ms', 0.0) + spec['steps'] * cost['step_compute_ms'] + cost['finish_ms']
            service = gpu_ms + cost.get('encode_ms', 0.0)
            scale = video_scale
        else:
            gpu_ms = service = cost['service_ms']
            scale = image_scale
        if not math.isfinite(service) or service <= 0:
            raise ValueError('nonpositive or nonfinite measured service time')
        classes[sid] = dict(cost, **{'class': sid, 'spec': spec, 'service_ms': service,
                                    'gpu_service_ms': gpu_ms, 'slo_scale': scale})
        cost_classes[sid] = cost
    videos = [sid for sid in ids if classes[sid]['kind'] == 'video']
    images = [sid for sid in ids if classes[sid]['kind'] == 'image']
    profile = dict(provenance='mixed', classes=cost_classes,
        links=copy.deepcopy(cost_classes[videos[0]]['links']),
        resize_ms=max(bundle['classes'][sid]['source']['resize_ms'] for sid in ids),
        quantum_overhead_ms=max(bundle['classes'][sid]['source']['quantum_overhead_ms'] for sid in ids),
        block_overhead_ms=0.0, source=dict(gpu_uuids=list(uuids), shape=shape_spec(p, videos[0]),
        bundle_sha256=bundle['sha256'], overhead_aggregation='conservative maximum across shape calibrations; not a direct global measurement',
        links_scope='each video class owns its independently fitted links; global links are legacy compatibility only'))
    # Reuse runtime defaults and policy validation, then replace all singleton
    # profile/trace/warmup data with the exact mixed-shape equivalents.
    singleton = copy.deepcopy(profile)
    singleton['classes'] = {model + '-preflight': cost_classes[videos[0]], 'sd35-preflight': cost_classes[images[0]]}
    policies = SUITES['preempt'] if diagnostic else SUITES[suite]
    count = diagnostic_controls['count'] if diagnostic else p['requests_per_cell'][split]
    actual_fraction = diagnostic_controls['video_fraction'] if diagnostic else video_fraction
    actual_load = diagnostic_controls['load'] if diagnostic else load
    actual_pattern = diagnostic_controls['pattern'] if diagnostic else pattern
    mode = 'digest' if diagnostic else 'check'
    cv = patterns[actual_pattern].get('cv', 1.0)
    config = build_serving_config(model, uuids, singleton, selected, seed=seed, load=actual_load,
        pattern=actual_pattern, cv=cv, video_fraction=actual_fraction, count=count, policies=policies,
        image_slo_scale=image_scale, video_slo_scale=video_scale,
        require_profile_match=True, forms=['cfg2'] if diagnostic else None, output_mode=mode,
        doorbell=True, image_batch=p['batch_limit'], policy_order='random')
    window = dict(count=count, video_fraction=actual_fraction, pattern=actual_pattern, load=actual_load)
    window.update({k: v for k, v in patterns[actual_pattern].items() if k != 'name'})
    if not diagnostic and (is_v2 or resolution_distribution is not None):
        window['resolution_distribution'] = copy.deepcopy(dist_map[dist_name])
    trace = build_trace(classes, selected, [window], slots=len(uuids), seed=seed, output_mode=mode)
    validate_trace(trace)
    provenance = dict(path='generated:frozen-formal-split', sha256=trace['manifest']['workload_sha256'], manifest=trace['manifest'])
    phase = config['phases'][0]
    if diagnostic:
        phase['options']['lazy_preemption'] = diagnostic_controls['lazy_preemption']
    phase['profile'] = profile
    phase['matrix']['/workload'] = [trace['workload']]
    phase['matrix']['/trace_provenance'] = [provenance]
    phase['warmup'] = []
    for sid in ids:
        for index in range(len(uuids)):
            phase['warmup'].append(dict(worker=('wan' if classes[sid]['kind']=='video' else 'image')+str(index),
                spec=dict(classes[sid]['spec'], steps=1, prompt='warmup', seed=7), quantum='step'))
    class_counts = trace['manifest']['class_counts']
    vid_total = sum(class_counts.get(sid, 0) for sid in videos)
    img_total = sum(class_counts.get(sid, 0) for sid in images)
    if is_v2 and vid_total > 0 and img_total > 0:
        video_mean_ms = sum(classes[sid]['gpu_service_ms'] * class_counts.get(sid, 0) for sid in videos) / vid_total
        image_mean_ms = sum(classes[sid]['service_ms'] * class_counts.get(sid, 0) for sid in images) / img_total
        share_rule = 'mean measured GPU work weighted by resolution distribution class counts'
    else:
        video_mean_ms = sum(classes[sid]['gpu_service_ms'] for sid in videos) / len(videos)
        image_mean_ms = sum(classes[sid]['service_ms'] for sid in images) / len(images)
        share_rule = 'mean measured GPU work across equally weighted shape classes'
    video_average = {'steps': 1, 'step_compute_ms': video_mean_ms}
    image_average = image_mean_ms
    image_share = partition_image_gpus(len(uuids), actual_fraction, video_average, image_average)
    phase['options']['partition_image_gpus'] = image_share
    # Suite keys select policies above; only their result-directory suffix is encoded.
    experiment_suite = suite.replace('+', '-')
    extra_tags = ''
    if not diagnostic and (is_v2 or slo_scale is not None or resolution_distribution is not None):
        if effective_slo_scale is not None:
            extra_tags += f"-slo{str(effective_slo_scale).replace('.', 'p')}"
        if dist_name != 'uniform':
            extra_tags += f"-{dist_name.replace('_', '-')}"
    config['experiment'] = (
        f"formal-{model}-{split}-{actual_pattern}-s{seed}-l{str(actual_load).replace('.', 'p')}"
        f"-v{str(actual_fraction).replace('.', 'p')}{extra_tags}"
        + (f'-{diagnostic}' if diagnostic else f'-{experiment_suite}')
    )
    role = 'preemption' if diagnostic == 'contention' else diagnostic or 'serving'
    config['benchmark'].update(trace=provenance, formal_protocol=binding(p, split, role, ids, selected, bundle),
        preemption_diagnostic=diagnostic, mixed_resolution=True,
        partition_image_gpus=dict(value=image_share, rule=share_rule),
        scope='finite public-prompt trace; not sustained-capacity or semantic qualification')
    if not diagnostic and (is_v2 or slo_scale is not None or resolution_distribution is not None):
        config['benchmark']['slo_scale'] = effective_slo_scale
        config['benchmark']['resolution_distribution'] = copy.deepcopy(dist_map[dist_name])
    if is_v2 or 'baseline_comparison_contract' in p or 'in_framework_policy_comparison' in p:
        comparison_meta = copy.deepcopy(config['benchmark'].get('in_framework_policy_comparison', {}))
        for key in ('in_framework_policy_comparison', 'baseline_comparison_contract'):
            if isinstance(p.get(key), dict):
                comparison_meta.update(copy.deepcopy(p[key]))
                comparison_meta[key] = copy.deepcopy(p[key])
        config['benchmark']['in_framework_policy_comparison'] = comparison_meta
    if 'quality_acceptance_contract' in p:
        config['benchmark']['quality_acceptance_contract'] = copy.deepcopy(p['quality_acceptance_contract'])
    if diagnostic and p1_plan is None:
        config['benchmark']['p0_diagnostic_controls'] = copy.deepcopy(diagnostic_controls)
        config['benchmark']['scope'] = p['p0_diagnostics']['scope']
        config['benchmark']['requested_workload_before_diagnostic_override'] = dict(
            load=load, video_fraction=video_fraction, pattern=pattern)
    if p1_plan is not None:
        case_id = f'{p1_pair}-{diagnostic}-s{seed}'
        config['experiment'] = 'p1-' + case_id
        config['benchmark']['formal_protocol']['role'] = 'p1_' + diagnostic
        config['benchmark']['mixed_resolution'] = False
        config['benchmark']['scope'] = p1_plan['scope']
        config['benchmark']['p1'] = dict(plan_sha256=canonical_hash(p1_plan), case_id=case_id,
            pair_id=p1_pair, model=model, shape_ids=ids, seed=seed, role=diagnostic,
            controls=copy.deepcopy(diagnostic_controls))
    bind_manifests(config, model, manifests)
    apply_trace_budget(config)
    return config


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'calibrate', 'assemble', 'serve'):
        ap = sub.add_parser(name)
        ap.add_argument('--model', choices=sorted(MODELS), required=True)
        ap.add_argument('--output', type=Path, required=True)
        if name == 'assemble':
            ap.add_argument('--runs', type=Path, nargs='+', required=True)
            ap.add_argument('--runtime-facts', type=Path, required=True)
        else:
            ap.add_argument('--datasets', type=Path, required=True)
            ap.add_argument('--model-manifests', type=Path, required=True)
            if name != 'preflight':
                ap.add_argument('--gpu-uuids')
        if name == 'preflight':
            ap.add_argument('--scope', choices=['development', 'formal'], required=True)
        if name == 'calibrate':
            ap.add_argument('--video-shape', required=True)
            ap.add_argument('--image-shape', required=True)
        elif name == 'serve':
            ap.add_argument('--bundle', type=Path, required=True)
            ap.add_argument('--split', choices=['development', 'test'], default='test')
            ap.add_argument('--seed', type=int, default=301)
            ap.add_argument('--load', type=float, default=None)
            ap.add_argument('--video-fraction', type=float, default=None)
            ap.add_argument('--slo-scale', type=float, default=None)
            ap.add_argument('--resolution-distribution', default=None)
            ap.add_argument('--pattern', choices=['poisson', 'gamma', 'bursty'], default='poisson')
            ap.add_argument('--suite', choices=sorted(SUITES), default='main')
            ap.add_argument('--diagnostic', choices=['contention', 'no_preemption'])
    args = parser.parse_args()
    p = protocol(args.protocol)
    if args.command == 'assemble':
        result = assemble(p, args.model, [read_json(path) for path in args.runs], read_json(args.runtime_facts))
    else:
        pools = {kind: read_json(args.datasets / (p['datasets'][kind]['kind'] + '.json')) for kind in ('image', 'video')}
        manifests = read_json(args.model_manifests)
        if args.command == 'preflight':
            result = preflight(p, args.model, pools, manifests, args.scope)
        elif args.command == 'calibrate':
            uuids = args.gpu_uuids.split(',') if args.gpu_uuids else allocated_uuids()
            result = calibration(p, args.model, uuids, pools, manifests, args.video_shape, args.image_shape)
        else:
            uuids = args.gpu_uuids.split(',') if args.gpu_uuids else allocated_uuids()
            result = serving(p, args.model, uuids, read_json(args.bundle), pools, manifests,
                split=args.split, seed=args.seed, load=args.load, video_fraction=args.video_fraction,
                pattern=args.pattern, suite=args.suite, diagnostic=args.diagnostic,
                slo_scale=args.slo_scale, resolution_distribution=args.resolution_distribution)
    write_new(args.output, result)
    print(json.dumps(dict(output=str(args.output), sha256=canonical_hash(result), scientific_qualification='pending')))


if __name__ == '__main__':
    main()
