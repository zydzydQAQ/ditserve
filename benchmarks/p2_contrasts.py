#!/usr/bin/env python3
"""Prepare explicit exact-serving contrasts without changing runtime code.

Resolved defaults below are audited against the pinned native sources. A source
change requires reviewing this resolver, rather than silently trusting labels.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from native_serving import FORM_DEGREES
from stage_gates import canonical_hash, read_json


AUDITED_NATIVE_SOURCES = {
    'src/ditserve/control/policy.cpp': ('6a52ff3a2caabd73ccf076a9158050fa8765d772ffedad498ffe4a18f1e1d2fb',),
    'include/ditserve/control/policy.hpp': ('10d5ed4df6da5c8a9151a3adcbea76d16bf5a435b8daaf99968ec4114f51b9c8',),
    'src/ditserve/control/study.cpp': ('34582d968cd1558e826849c86d408f958b9217825b7486f942d2b4e1ecc10374',),
    # The ownership fix changes correctness, not option resolution. Keeping
    # the prior hash permits honest analysis of its failed runs, not promotion.
    'src/ditserve/control/cluster.cpp': ('6cf1044cc837e26ba5bfb4e1b83870a10709e6e989cb58662e8bbf4d46287d0e',
                                      '5c2054931ec6d1b8c5dfa3c50f781e0f57594783633494b64f409adad7a5ed54'),
}

CONTRASTS = {
    'block-fixed': dict(reference='edf-cfg2', candidate='block-cfg2', mechanism='block preemption', field='blocks'),
    'block-elastic': dict(reference='cfg2-first', candidate='ditserve', mechanism='block preemption', field='blocks'),
    'pruning': dict(reference='abl-no-prune', candidate='ditserve', mechanism='form pruning', field='prune_dominated'),
    'minimum-pair': dict(reference='abl-no-mindeg', candidate='ditserve', mechanism='minimum CFG-pair admission', field='min_video_degree'),
    'placement': dict(reference='abl-first-fit', candidate='ditserve', mechanism='pair-fit image placement', field='image_placement'),
    'lazy': dict(reference='ditserve', candidate='ditserve', mechanism='deadline utility filter',
                 option='lazy_preemption', values=[False, True], field='lazy_preemption'),
    'remaining-work': dict(reference='ditserve', candidate='ditserve', mechanism='remaining-work widening',
                           option='widen_remaining_work', values=[False, True], field='widen_remaining_work'),
    'doorbell': dict(reference='ditserve', candidate='ditserve', mechanism='continuous doorbell execution',
                     option='doorbell', values=[False, True], field='doorbell'),
}
for _policy in ('xdit', 'tetriserve', 'sbp-esp', 'edf-bestfit', 'edf-cfg2', 'cfg2-first'):
    CONTRASTS['system-' + _policy] = dict(reference=_policy, candidate='ditserve', mechanism='system comparison', system=True)


def canonical_policy(policy):
    return 'xdit-cfg2' if policy == 'xdit' else policy


def resolved_options(policy, options, gpus):
    if (options.get('sp_transport') != 'export' or options.get('pin_started', False)
            or options.get('tiered_memory', False) or gpus not in (2, 4, 6, 8)):
        raise ValueError('unsupported: P2 resolver covers unpinned export transport on 2/4/6/8 GPUs only')
    policy = canonical_policy(policy)
    known = {'ditserve', 'cfg2-first', 'abl-no-block', 'abl-no-prune', 'abl-no-mindeg', 'abl-first-fit',
             'edf-cfg2', 'block-cfg2', 'xdit-cfg2', 'tetriserve', 'sbp-esp', 'edf-bestfit'}
    if policy not in known:
        raise ValueError('policy has no audited exact resolver: ' + policy)
    ours = policy in {'ditserve', 'cfg2-first', 'abl-no-block', 'abl-no-prune', 'abl-no-mindeg', 'abl-first-fit'}
    fixed = policy in {'edf-cfg2', 'block-cfg2', 'xdit-cfg2'}
    blocks = policy in {'ditserve', 'abl-no-prune', 'abl-no-mindeg', 'abl-first-fit', 'block-cfg2'}
    default = dict(elastic=not fixed, blocks=blocks, fixed_form='cfg2' if fixed else None,
        prune_dominated=ours and policy != 'abl-no-prune', min_video_degree=2 if ours and policy != 'abl-no-mindeg' else 1,
        widen_efficiency=.9 if ours else 0., image_placement='pair_fit' if ours and policy != 'abl-first-fit' else 'first',
        order='fcfs' if policy == 'xdit-cfg2' else 'edf', hold_diffuse=policy == 'xdit-cfg2',
        demote_late=policy != 'xdit-cfg2', rounds=policy == 'tetriserve',
        deadline_dp=policy in {'tetriserve', 'sbp-esp'}, best_fit=policy == 'edf-bestfit',
        lazy_preemption=True, widen_remaining_work=True, doorbell=False, check_blocks=1,
        quantum_blocks=4, quantum_ms=0., pin_started=False, round_steps=5,
        image_batching=policy == 'sbp-esp')
    overridable = {'prune_dominated', 'min_video_degree', 'widen_efficiency', 'image_placement', 'order',
                   'hold_diffuse', 'demote_late', 'lazy_preemption', 'widen_remaining_work', 'doorbell',
                   'check_blocks', 'quantum_blocks', 'quantum_ms', 'pin_started', 'round_steps'}
    default.update({key: options[key] for key in overridable if key in options})
    default['forms'] = ['cfg2'] if fixed else options.get('forms')
    if (not isinstance(default['forms'], list) or not default['forms']
            or any(form not in FORM_DEGREES for form in default['forms'])):
        raise ValueError('unsupported or implicit parallel form space')
    default['reserve_image_gpus'] = options.get('reserve_image_gpus',
        options.get('partition_image_gpus', max(1, gpus // 4)) if policy == 'xdit-cfg2' else 0)
    default['strict_partition'] = options.get('strict_partition', policy == 'xdit-cfg2')
    reserve = default['reserve_image_gpus']
    if type(reserve) is not int or reserve < 0 or reserve >= gpus or fixed and gpus - reserve < 2:
        raise ValueError('unsupported baseline: image partition leaves no complete fixed CFG2 video group')
    default['doorbell_effective'] = blocks and default['doorbell']
    default['image_batching_effective'] = default['image_batching'] and 'image_batch' not in options
    default['shared_options'] = {key: value for key, value in options.items() if key not in overridable
                                 and key not in {'forms', 'reserve_image_gpus', 'partition_image_gpus', 'strict_partition'}}
    return default


def validate_contrast(name, reference, candidate):
    definition = CONTRASTS[name]
    for arm, phase in (('reference', reference), ('candidate', candidate)):
        if canonical_policy(phase['policy']) != canonical_policy(definition[arm]):
            raise ValueError('selected policy differs from contrast ' + arm)
    normalized = [copy.deepcopy(phase) for phase in (reference, candidate)]
    for phase in normalized:
        phase.pop('policy', None)
    if 'option' in definition:
        key = definition['option']
        for index, phase in enumerate(normalized):
            if phase.get('options', {}).get(key) is not definition['values'][index]:
                raise ValueError('both contrast option values must be explicit and different')
            phase['options'].pop(key)
    if normalized[0] != normalized[1]:
        raise ValueError('contrast changes inputs or settings beyond its declared mechanism/policy')
    resolved = [resolved_options(phase['policy'], phase.get('options', {}), len(phase['gpus']))
                for phase in (reference, candidate)]
    for phase, options in zip((reference, candidate), resolved):
        available = len(phase['gpus']) - options['reserve_image_gpus']
        if any(row.get('kind') == 'video' and row.get('min_degree', 1) > available
               for row in phase.get('profile', {}).get('classes', {}).values()):
            raise ValueError('unsupported model: partition cannot accommodate its minimum degree')
        if options['fixed_form'] and any(row.get('kind') == 'video' and row.get('min_degree', 1) > 2
                                        for row in phase.get('profile', {}).get('classes', {}).values()):
            raise ValueError('unsupported: class minimum would replace the declared fixed CFG2 form with SP')
    differing = {key for key in resolved[0] if resolved[0][key] != resolved[1][key]}
    if not differing:
        raise ValueError('policy/options overrides make the two supposedly different arms identical')
    if not definition.get('system'):
        allowed = {definition['field']}
        if definition['field'] in {'blocks', 'doorbell'}:
            allowed.add('doorbell_effective')
        if not differing <= allowed or definition['field'] not in differing:
            raise ValueError('resolved options do not isolate the declared mechanism')
    if name == 'lazy' and not all(item['doorbell_effective'] for item in resolved):
        raise ValueError('lazy doorbell contrast is disabled by the resolved runtime configuration')
    if name == 'remaining-work' and not all(item['widen_efficiency'] > 0 for item in resolved):
        raise ValueError('remaining-work widening is disabled by the resolved runtime configuration')
    if name == 'minimum-pair' and any('sp1' not in item['forms'] for item in resolved):
        raise ValueError('minimum-pair ablation needs an actually available SP1 form')
    if name == 'pruning' and any(len(item['forms']) < 2 for item in resolved):
        raise ValueError('pruning ablation needs multiple candidate forms')
    if name == 'remaining-work' and any(len({FORM_DEGREES[form] for form in item['forms']}) < 2 for item in resolved):
        raise ValueError('remaining-work widening needs multiple candidate degrees')
    if name == 'placement' and len(reference['gpus']) < 4:
        raise ValueError('placement ablation needs at least two CFG pairs')
    video_classes = [row for row in reference.get('profile', {}).get('classes', {}).values() if row.get('kind') == 'video']
    if name == 'minimum-pair' and video_classes and all(row.get('min_degree', 1) >= 2 for row in video_classes):
        raise ValueError('model minimum degree makes the minimum-pair option ineffective')
    if name in ('pruning', 'remaining-work') and video_classes:
        allowed = [[form for form in resolved[1]['forms'] if FORM_DEGREES[form] <= len(reference['gpus'])
                    and FORM_DEGREES[form] >= max(row.get('min_degree', 1), resolved[1]['min_video_degree'])]
                   for row in video_classes]
        if not any(len(forms if name == 'pruning' else {FORM_DEGREES[form] for form in forms}) > 1 for forms in allowed):
            raise ValueError('model minimum/form constraints disable the declared mechanism')
    if name == 'doorbell':
        if any(item['quantum_ms'] != 0 or item['quantum_blocks'] != item['check_blocks'] for item in resolved):
            raise ValueError('doorbell contrast requires the same explicit block-check spacing')
    return dict(reference=resolved[0], candidate=resolved[1], differing_fields=sorted(differing),
                scope='audited registry/options resolution; actual execution evidence is reviewed separately')


def make_configs(config, contrast):
    if len(config.get('phases', [])) != 1 or config.get('benchmark', {}).get('native_serving') is not True:
        raise ValueError('contrast input must be one frozen native serving phase')
    if config['benchmark'].get('exact_mode') is not True:
        raise ValueError('P2 exact contrasts do not accept approximate arms')
    definition = CONTRASTS[contrast]
    result = {}
    for index, arm in enumerate(('reference', 'candidate')):
        value = copy.deepcopy(config)
        phase = value['phases'][0]
        if set(phase.get('matrix', {})) != {'/policy', '/workload', '/trace_provenance'}:
            raise ValueError('start from a single frozen formal workload without other matrix axes')
        if len(phase['matrix']['/workload']) != 1 or len(phase['matrix']['/trace_provenance']) != 1:
            raise ValueError('one frozen trace per contrast configuration is required')
        phase['policy'] = definition[arm]
        phase['matrix']['/policy'] = [definition[arm]]
        if 'option' in definition:
            phase['options'][definition['option']] = definition['values'][index]
        if contrast == 'doorbell':
            phase['options']['check_blocks'] = phase['options']['quantum_blocks']
            phase['options']['quantum_ms'] = 0
        value['benchmark']['policies'] = [definition[arm]]
        value['benchmark']['requested_policies'] = [definition[arm]]
        for field in ('p1', 'p0_diagnostic_controls', 'preemption_diagnostic', 'policy_display', 'policy_order'):
            value['benchmark'].pop(field, None)
        value['benchmark']['quality_qualified'] = False
        value['benchmark']['performance_qualified'] = False
        value['benchmark']['scope'] = 'prepared P2 exact-serving contrast; independent qualification required'
        value['benchmark']['p2'] = dict(contrast=contrast, arm=arm, mechanism=definition['mechanism'],
            source_config_sha256=canonical_hash(config),
            parent_role=config['benchmark']['formal_protocol']['role'],
            parent_controls=copy.deepcopy(config['benchmark'].get('p0_diagnostic_controls',
                config['benchmark'].get('p1', {}).get('controls'))),
            parent_diagnostic=config['benchmark'].get('preemption_diagnostic'),
            source_benchmark_controls={key: copy.deepcopy(config['benchmark'][key]) for key in
                ('count', 'seed', 'pattern', 'load', 'video_fraction', 'image_slo_scale', 'video_slo_scale')
                if key in config['benchmark']},
            budget_scope='inherited source-suite conservative upper bound; not a single-arm time prediction')
        value['experiment'] += '-p2-' + contrast + '-' + arm
        result[arm] = value
    phases = [copy.deepcopy(result[arm]['phases'][0]) for arm in ('reference', 'candidate')]
    for phase in phases:
        phase['matrix'].pop('/policy')
    validate_contrast(contrast, *phases)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, action='append', type=Path, help='one frozen configuration per independent seed')
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--workload-id', required=True, help='shared workload/control identity across independent seeds')
    parser.add_argument('--contrast', required=True, action='append', choices=sorted(CONTRASTS))
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    protocol = read_json(args.protocol.read_bytes())
    configs = [read_json(path.read_bytes()) for path in args.config]
    seeds = [config['benchmark']['seed'] for config in configs]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        parser.error('freeze at least three distinct independent seed configurations')
    if len(args.contrast) != len(set(args.contrast)):
        parser.error('contrasts must be unique')
    if any(config['benchmark']['formal_protocol']['sha256'] != canonical_hash(protocol) for config in configs):
        parser.error('source configurations do not bind the supplied protocol')
    files, cases = {}, []
    def reference(name, value):
        raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
        files[name] = raw
        return dict(path=name, sha256=hashlib.sha256(raw).hexdigest())
    for contrast in args.contrast:
        for config in configs:
            seed = config['benchmark']['seed']
            case = dict(id=f'{contrast}-s{seed}', workload_id=args.workload_id, contrast=contrast, seed=seed)
            for arm, value in make_configs(config, contrast).items():
                case[arm] = dict(config=reference(f'{contrast}-s{seed}-{arm}.json', value),
                    phase=value['phases'][0]['id'], attempt=1, cell=0, order='forward')
            cases.append(case)
    plan = dict(schema='ditserve.p2-plan.v1', protocol_sha256=canonical_hash(protocol), cases=cases,
        statistics=dict(confidence=.95, resampling_unit='seed', minimum_independent_seeds=3,
                        bootstrap_samples=10000, bootstrap_seed=20261103))
    evidence = dict(schema='ditserve.p2-evidence.v1', protocol=reference('protocol.json', protocol),
                    plan=reference('plan.json', plan), runs=[])
    reference('evidence.json', evidence)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        with (args.output_dir / name).open('xb') as stream:
            stream.write(raw)
    print('prepared configs/plan/evidence template only; stage eligibility and numeric quality require separate evidence')


if __name__ == '__main__':
    main()
