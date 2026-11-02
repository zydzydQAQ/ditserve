#!/usr/bin/env python3
"""Generate one preregistered P1 case without submitting it or granting eligibility."""

import argparse
import math
from pathlib import Path

import formal_benchmark as formal
from cluster_config import allocated_uuids
from ditserve.evaluation.common import canonical_hash, read_json

DEFAULT_PLAN = formal.DEFAULT_PROTOCOL.parent / 'p1-qualification-v1.json'


def validate_plan(plan, protocol):
    if (plan.get('schema') != 'ditserve.p1-plan.v1'
            or plan.get('protocol_sha256') != canonical_hash(protocol)):
        raise ValueError('P1 plan must bind the exact formal protocol')
    if (plan.get('prompt_split') != 'development' or plan.get('policies') != ['edf-cfg2', 'block-cfg2']
            or plan.get('forms') != ['cfg2'] or plan.get('output_mode') != 'digest'):
        raise ValueError('P1 requires development prompts and the fixed CFG2 digest pair')
    seeds = plan.get('seeds', [])
    if (len(seeds) < 3 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed not in protocol['seeds']['development'] for seed in seeds)):
        raise ValueError('P1 requires at least three distinct development seeds')
    required = {sid for sid, shape in protocol['shapes'].items() if shape['scope'] == 'formal'}
    covered, ids, videos = set(), set(), set()
    for pair in plan.get('shape_pairs', []):
        if set(pair) != {'id', 'model', 'video_shape', 'image_shape'}:
            raise ValueError('P1 pair needs explicit id, model and both shape IDs')
        if not isinstance(pair['id'], str) or not pair['id'] or pair['id'] in ids:
            raise ValueError('P1 pair IDs must be nonempty and unique')
        video, image = pair['video_shape'], pair['image_shape']
        if video in videos or video not in required or image not in required:
            raise ValueError('P1 requires unique formal video shapes and formal image shapes')
        v, i = protocol['shapes'][video], protocol['shapes'][image]
        if v['kind'] != 'video' or v['model'] != pair['model'] or i['kind'] != 'image' or i['model'] != 'sd35':
            raise ValueError('P1 pair model/kind differs from formal shapes')
        ids.add(pair['id']); videos.add(video); covered.update((video, image))
    if covered != required:
        raise ValueError('P1 plan must retain every formal shape, including unsuccessful shapes')
    if set(plan.get('roles', {})) != {'contention', 'no_preemption'}:
        raise ValueError('P1 requires contention and no_preemption controls')
    patterns = {item['name'] for item in protocol['patterns']}
    for role, controls in plan['roles'].items():
        for name in ('load', 'video_fraction', 'video_slo_scale', 'image_slo_scale'):
            value = controls.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('P1 controls must be finite and positive: ' + name)
        count, fraction = controls.get('count'), controls['video_fraction']
        if (type(count) is not int or count <= 0 or fraction > 1 or count * fraction != round(count * fraction)
                or controls.get('pattern') not in patterns):
            raise ValueError('P1 requires integer modality counts and a declared arrival pattern')
        if (controls.get('lazy_preemption') is not True
                or controls['video_slo_scale'] != protocol['slo']['video_scale']
                or controls['image_slo_scale'] != protocol['slo']['image_scale']):
            raise ValueError('P1 must retain lazy filtering and formal SLOs')
        if (controls.get('require_observed_stops') is not (role == 'contention')
                or role == 'contention' and not 0 < fraction < 1
                or role == 'no_preemption' and fraction != 1):
            raise ValueError('P1 role must match its actual stop and modality requirements')
    stats = plan.get('statistics', {})
    if (stats.get('confidence') != 0.95 or stats.get('resampling_unit') != 'seed'
            or type(stats.get('bootstrap_samples')) is not int or stats['bootstrap_samples'] < 1000
            or type(stats.get('bootstrap_seed')) is not int
            or stats.get('minimum_independent_seeds') != 3):
        raise ValueError('P1 requires explicit 95% seed-level uncertainty settings')
    rule = plan.get('budget_rule', {})
    if (rule.get('schema') != 'ditserve.trace-budget.v1'
            or type(rule.get('per_cell_startup_seconds')) not in (int, float)
            or not math.isfinite(rule['per_cell_startup_seconds']) or rule['per_cell_startup_seconds'] <= 0
            or type(rule.get('timing_margin')) not in (int, float)
            or not math.isfinite(rule['timing_margin']) or rule['timing_margin'] < 1):
        raise ValueError('P1 requires a finite explicit trace-derived time budget')
    return plan


def case_config(protocol, plan, pair_id, role, seed, uuids, bundle, pools, manifests):
    validate_plan(plan, protocol)
    pair = next((item for item in plan['shape_pairs'] if item['id'] == pair_id), None)
    if pair is None:
        raise ValueError('unknown P1 pair')
    config = formal.serving(protocol, pair['model'], uuids, bundle, pools, manifests,
        split=plan['prompt_split'], seed=seed, diagnostic=role, p1_plan=plan, p1_pair=pair_id)
    config['benchmark']['p1']['time_planning'] = formal.apply_trace_budget(config, plan['budget_rule'])
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=formal.DEFAULT_PROTOCOL)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--pair', required=True)
    parser.add_argument('--role', choices=['contention', 'no_preemption'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--datasets', type=Path, required=True)
    parser.add_argument('--model-manifests', type=Path, required=True)
    parser.add_argument('--gpu-uuids')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = formal.protocol(args.protocol)
    pools = {kind: read_json(args.datasets / (protocol['datasets'][kind]['kind'] + '.json'))
             for kind in ('image', 'video')}
    uuids = args.gpu_uuids.split(',') if args.gpu_uuids else allocated_uuids()
    config = case_config(protocol, read_json(args.plan), args.pair, args.role, args.seed,
                         uuids, read_json(args.bundle), pools, read_json(args.model_manifests))
    formal.write_new(args.output, config)
    print(config['experiment'] + ': configuration only; P0 gate required before execution')


if __name__ == '__main__':
    main()
