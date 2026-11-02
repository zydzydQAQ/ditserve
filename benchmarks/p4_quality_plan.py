#!/usr/bin/env python3
"""Freeze P4 paired quality inputs and storage requirements; never execute jobs.

This is a plan, not a serving configuration or an eligibility receipt. Fold
parameters must come from a separately frozen development decision. No assets
are downloaded, GPU allocation queried, artifacts deleted, or gates passed.
"""

import argparse
import copy
import math
import re
from pathlib import Path

import formal_benchmark as formal
from cluster_config import MODELS, SD35_PATH
from ditserve.evaluation.common import canonical_hash, read_json, seal, verify_seal

SCHEMA = 'ditserve.p4-quality-plan.v1'
GIB = 1024 ** 3


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return value


def finite(value, name, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(name + ' must be finite and >= ' + str(minimum))
    return value


def shape_ids(protocol, model):
    if model not in MODELS:
        raise ValueError('P4 folding applies to a declared Wan video model')
    ids = [sid for sid, row in protocol['shapes'].items()
           if row['model'] == model and row['scope'] == 'formal' and row['kind'] == 'video']
    if not ids or any(protocol['shapes'][sid]['steps'] != 50 for sid in ids):
        raise ValueError('this P4 rule requires explicit formal 50-step video shapes')
    return ids


def validate_rule(rule):
    if (rule.get('schema') != 'ditserve.p4-fold-rule.v1' or rule.get('steps') != 50
            or rule.get('selected_without_test_outputs') is not True
            or not isinstance(rule.get('development_evidence_sha256'), str)
            or not formal.HEX64.fullmatch(rule['development_evidence_sha256'])):
        raise ValueError('freeze a 50-step rule and development evidence before test outputs')
    window = rule.get('eligible_steps')
    if (not isinstance(window, list) or not window or any(type(i) is not int for i in window)
            or window[0] < 25 or window[-1] > 48
            or window != list(range(window[0], window[-1] + 1))):
        raise ValueError('late eligible_steps must be an explicit contiguous subset of 25..48')
    if integer(rule.get('max_folded_steps'), 'max_folded_steps', 1) > len(window):
        raise ValueError('actual per-request fold budget exceeds the eligible window')
    for key in ('curvature_threshold', 'cfg_damping', 'cfg_vel_alpha', 'max_curvature_guard'):
        finite(rule.get(key), key)
    if rule['cfg_damping'] > 1 or rule['cfg_vel_alpha'] > 1 or rule['max_curvature_guard'] <= 0:
        raise ValueError('invalid CFG predictor/guard parameters')
    thresholds = rule.get('numeric_thresholds', {})
    if set(thresholds) != {'min_psnr', 'min_ssim', 'max_lpips'}:
        raise ValueError('freeze all three numerical thresholds on development data')
    for key, value in thresholds.items():
        finite(value, key)
    if thresholds['min_ssim'] > 1:
        raise ValueError('SSIM threshold exceeds one')
    return rule


def quality_arms(rule, form):
    """All controls hold form/predictor fixed; fixed-0 aliases the exact output."""
    validate_rule(rule)
    common = {key: rule[key] for key in ('cfg_damping', 'cfg_vel_alpha', 'max_curvature_guard')}
    common.update(final_step_exact_anchor=True, force_exact_cond_on_fold=True,
                  min_full_cfg_steps=rule['eligible_steps'][0])
    arms = [dict(id='exact', policy='block-' + form, cfg_fold=None,
                 expected_actual_fold_count=0)]
    for count in range(1, rule['max_folded_steps'] + 1):
        arms.append(dict(id='fixed-' + str(count), policy='fold-' + form,
            expected_actual_fold_count=count, cfg_fold=dict(common,
                folded_steps=rule['eligible_steps'][:count], max_folded_steps=count)))
    arms.append(dict(id='adaptive', policy='fold-' + form,
        expected_actual_fold_count=None, actual_fold_count_bounds=[0, rule['max_folded_steps']],
        cfg_fold=dict(common, folded_steps=rule['eligible_steps'],
            max_folded_steps=rule['max_folded_steps'], curvature_threshold=rule['curvature_threshold'])))
    return arms


def storage_budget(protocol, model, rule, snapshot, *, reserve_bytes=2 * GIB,
                   metadata_allowance_bytes=512 * 1024 ** 2):
    """Bound float32 NPY disk payload, a full-size temporary, and stop-line margin.

    A df snapshot is evidence supplied by the operator, never a live guarantee.
    GNU df rounds utilization upward, hence 79% of (used + available) is used
    conservatively for the >=80% stop rule. Other writers can consume this margin.
    """
    validate_rule(rule)
    ids = shape_ids(protocol, model)
    if snapshot.get('filesystem') != '/data' or not snapshot.get('observed_at_utc'):
        raise ValueError('storage needs a timestamped /data df snapshot')
    total, used, available = [integer(snapshot.get(key), key, 0 if key == 'used_bytes' else 1)
                              for key in ('total_bytes', 'used_bytes', 'available_bytes')]
    percentage = integer(snapshot.get('use_percent'), 'use_percent')
    if used + available > total or percentage != math.ceil(100 * used / (used + available)):
        raise ValueError('inconsistent df byte counts or rounded utilization')
    integer(reserve_bytes, 'reserve_bytes'); integer(metadata_allowance_bytes, 'metadata allowance')
    margin = max(0, min(790 * GIB - used, 79 * (used + available) // 100 - used, available))
    stopped = used > 790 * GIB or percentage >= 80
    usable = max(0, margin - reserve_bytes)
    count = max(32, integer(protocol['quality']['paired_prompts_per_model_minimum'], 'prompt minimum', 1))
    seeds = protocol['quality']['paired_seeds']
    pairs = count * len(seeds)
    arms = rule['max_folded_steps'] + 2  # exact plus fixed 1..K plus adaptive
    shapes = {}
    for sid in ids:
        row = protocol['shapes'][sid]
        payload = math.prod(row[key] for key in ('height', 'width', 'frames')) * 3 * 4
        output = payload + 4096  # conservative NPY header allowance; no lossy encoding
        shapes[sid] = dict(pairs=pairs, artifacts=pairs * arms, array_payload_bytes=payload,
            artifact_bound_bytes=output, all_arms_artifact_bytes=output * pairs * arms)
    largest = max(row['artifact_bound_bytes'] for row in shapes.values())
    full = sum(row['all_arms_artifact_bytes'] for row in shapes.values())
    full_peak = full + largest + metadata_allowance_bytes
    pair_peak = 3 * largest + metadata_allowance_bytes
    return dict(schema='ditserve.p4-storage-budget.v1', snapshot=copy.deepcopy(snapshot),
        stop_rule=dict(used_bytes_exceeds=790 * GIB, use_percent_at_least=80),
        stopped=stopped, stop_line_headroom_bytes=margin, reserve_bytes=reserve_bytes,
        usable_headroom_bytes=usable, metadata_allowance_bytes=metadata_allowance_bytes,
        shapes=shapes, generated_arms=arms, fixed_zero='reuse exact artifact; no extra generation',
        full_artifact_bytes=full, full_temporary_peak_bytes=full_peak,
        full_retention_fits=not stopped and full_peak <= usable,
        serial_pair_temporary_peak_bytes=pair_peak,
        serial_pair_fits=not stopped and pair_peak <= usable,
        concurrency_limit=1,
        accounting='float32 decoded arrays plus NPY header, one full-size write temporary and metadata; '
                   'model inputs already present; no checkpoint/export copies, VBench exports or retries included',
        recheck='df -h /data and byte snapshot before every write and at most 60s during work; '
                'stop all project tasks immediately if either stop line is crossed',
        retention=dict(automatic_deletion=False, execution_ready=False,
            live_arrays=['exact reference', 'one candidate', 'one full-size atomic-write temporary'],
            order=['Generate exact once per prompt/seed/shape and preserve its verified artifact.',
                   'Generate adaptive and each fixed-N arm serially; verify runtime/model/output hashes and full step history.',
                   'Measure each arm against the retained exact reference; persist hash-bound per-pair metrics.',
                   'Retain pair metrics for all fixed-N controls, actual-N matching and prompt-cluster bootstrap.',
                   'Before removing any raw output, inventory and close ALL declared consumers including semantic/replay uses.',
                   'Only a separately reviewed exact cleanup manifest may reclaim closed raw artifacts; verify deletion.',
                   'Keep the exact reference until every candidate and all other consumers have closed; then proceed to next pair.'],
            blocked_by='No streaming evaluator or verified dependency-closure/cleanup manifest is produced by this plan; '
                       'normal completed-run cleanup follows the project retention workflow after workers/finalizers stop. '
                       'Unclosed dependencies must remain and can exceed the serial-pair estimate.'))


def prepare(protocol, model, uuids, bundle, pools, manifests, rule, snapshot, *,
            prompt_selection_seed, form='cfg2'):
    ids = shape_ids(protocol, model)
    integer(prompt_selection_seed, 'prompt_selection_seed')
    seeds = protocol['quality']['paired_seeds']
    if seeds != [301, 302, 303]:
        raise ValueError('P4 requires the frozen literal generation seeds 301,302,303')
    if (not uuids or any(not isinstance(uuid, str) or not uuid for uuid in uuids)
            or len(set(uuids)) != len(uuids)):
        raise ValueError('explicit unique allocation GPU UUIDs are required; no GPU discovery')
    match = re.fullmatch(r'cfg2(?:sp([2-4]))?', form)
    if not match or 2 * int(match.group(1) or 1) > len(uuids):
        raise ValueError('quality form must be an explicit CFG2 form fitting this allocation')
    verify_seal(bundle)
    if (bundle.get('schema') != formal.BUNDLE_SCHEMA or bundle.get('model') != model
            or bundle.get('protocol_sha256') != canonical_hash(protocol)
            or bundle.get('gpu_uuids') != list(uuids)
            or bundle.get('model_manifests') != manifests):
        raise ValueError('calibration must bind the same model/protocol/allocation/model mapping')
    declared = {'workers': {'video': {'backend': 'wan', 'config': {'model_path': MODELS[model]['path']}},
                            'image': {'backend': 'sd35', 'config': {'model_path': SD35_PATH}}}, 'benchmark': {}}
    if model == 'wan22' and not manifests.get(model, {}).get('model_path'):
        raise ValueError('Wan22 requires an explicit official snapshot; legacy composite default forbidden')
    formal.bind_manifests(declared, model, manifests)
    if model == 'wan22' and declared['workers']['video']['config']['model_path'] == MODELS[model]['path']:
        raise ValueError('legacy composite Wan22 snapshot cannot be planned as official')
    sources, services = {}, {}
    for sid in ids:
        entry = bundle.get('classes', {}).get(sid, {})
        if entry.get('spec') != formal.shape_spec(protocol, sid):
            raise ValueError('every formal video shape needs its exact measured calibration: ' + sid)
        source, cost = entry.get('source', {}), entry.get('cost', {})
        if (source.get('gpu_uuids') != list(uuids) or cost.get('kind') != 'video'
                or any(not isinstance(source.get(k), str) or not formal.HEX64.fullmatch(source[k])
                       for k in ('run_sha256', 'core_sha256'))):
            raise ValueError('calibration lacks source hashes and identical GPU placement')
        service = sum(finite(cost.get(k, 0), k) for k in ('prepare_ms', 'encode_ms', 'finish_ms'))
        service += 50 * finite(cost.get('step_compute_ms'), 'step_compute_ms')
        if service <= 0:
            raise ValueError('calibrated service must be positive')
        sources[sid], services[sid] = copy.deepcopy(source), service
    if len({source['core_sha256'] for source in sources.values()}) != 1:
        raise ValueError('calibration core changed between formal shapes')
    selected = formal.split_pool(pools['video'], protocol, 'video', 'test')
    count = max(32, protocol['quality']['paired_prompts_per_model_minimum'])
    samples = sorted(selected['samples'], key=lambda row: (
        canonical_hash([prompt_selection_seed, model, ' '.join(row['prompt'].split())]), str(row['sample_id'])))[:count]
    if len(samples) != count or len({str(row['sample_id']) for row in samples}) != count:
        raise ValueError('insufficient distinct held-out prompts or duplicate dataset sample IDs')
    prompts = [dict(dataset_sample_id=row['sample_id'], prompt=row['prompt'],
                    normalized_prompt_sha256=canonical_hash(' '.join(row['prompt'].split()))) for row in samples]
    arms = quality_arms(rule, form)
    pairs = []
    for sid in ids:
        for index, row in enumerate(prompts):
            for seed in seeds:
                pairs.append(dict(id=f'{sid}-p{index:03d}-s{seed}', shape_id=sid,
                    dataset_sample_id=row['dataset_sample_id'], prompt_index=index,
                    generation_seed=seed, spec=dict(formal.shape_spec(protocol, sid), prompt=row['prompt'], seed=seed)))
    storage = storage_budget(protocol, model, rule, snapshot)
    blockers = ['GPU numerical/form qualification and current full model integrity remain prerequisites.',
        'Development rule evidence and threshold decision need verification; a supplied hash is not an acceptance receipt.',
        'Native quality helper alone does not enforce all identity/weight gates, complete planned denominator, '
        'hash-bound fixed-control files or prompt-cluster bootstrap.',
        'Streaming pair evaluator and dependency-closure retention receipts are not implemented here.',
        'LPIPS package/checkpoint SHA and offline availability must be verified without downloads.',
        'Full pinned VBench protocol and native-output adapter remain separate semantic prerequisites.',
        'Fission release/beneficiary lifecycle evidence and controlled reuse runs are still required.']
    if storage['stopped']:
        blockers.insert(0, 'Shared storage stop line reached: stop all project work; no writes or jobs.')
    elif not storage['full_retention_fits']:
        blockers.insert(0, 'Full matrix artifacts exceed stop-line headroom; do not launch all-at-once retention.')
    if not storage['serial_pair_fits']:
        blockers.insert(0, 'Even the serial-pair temporary peak exceeds the reserved headroom.')
    return seal(dict(schema=SCHEMA, status='blocked', execution_ready=False, qualification_eligible=False,
        scope='Reference-relative paired VIDEO quality plan; not official VBench or serving-performance evidence. '
              'SD3.5 is an identity-bound mixed-serving beneficiary, not a CFG-fold quality candidate.',
        protocol_sha256=canonical_hash(protocol), model=model, shape_ids=ids, form=form,
        gpu_uuids=list(uuids), calibration_bundle_sha256=bundle['sha256'], calibration_sources=sources,
        model_manifests=copy.deepcopy(manifests), model_mapping_canonical_sha256=canonical_hash(manifests),
        worker_model_bindings={key: value['config'] for key, value in declared['workers'].items()},
        split='test', parent_pool_sha256=pools['video']['sha256'], test_pool_sha256=selected['sha256'],
        prompt_selection_seed=prompt_selection_seed, generation_seeds=seeds, arrival_seed=None,
        arrival_scope='isolated quality pairs have no mixed traffic; reuse arms require a separately frozen paired trace',
        prompts=prompts, prompts_per_model=count, prompts_per_shape=count,
        pairing='the same selected prompts and literal generation seeds in every shape and arm',
        rule=copy.deepcopy(rule), rule_sha256=canonical_hash(rule), arms=arms, pairs=pairs,
        fixed_zero_control=dict(arm='exact', expected_actual_fold_count=0,
            interpretation='zero-fold/no-compute-reduction self-control; never adaptive superiority'),
        expected=dict(pair_count=len(pairs), generated_artifact_count=len(pairs) * len(arms),
            denominator='all planned pair x arm identities, including failures, missing outputs and unqualified shapes'),
        compute_controls=dict(actual_fold_strata=list(range(rule['max_folded_steps'] + 1)),
            require='ordered complete per-step history, worker full+folded=50, matching actual folded indices, '
                    'saved unconditional forwards, final step 49 full, fixed-N actual count exactly N',
            scope='matching N equalizes logical unconditional-forward count at fixed shape/form; '
                  'it does not equalize measured GPU time',
            measured_costs=['model_ms', 'worker_operation_ms', 'curvature_gate_ms', 'transfer_ms', 'allocated_gpu_seconds'],
            disabled=['Taylor cache', 'DBCache', 'int8 state', 'CFG fission in isolated quality arms']),
        statistics=dict(confidence=.95, bootstrap_samples=2000, bootstrap_seed=20261103,
            resampling_unit='prompt cluster; keep all three seeds and shapes attached',
            strata='report each model/shape and actual-N stratum; do not treat 96 seed pairs as 96 independent prompts',
            psnr='infinite PSNR is an exact-match relation, never infinity-minus-infinity; '
                 'keep SSIM/LPIPS decisions independent and zero-fold strata non-superiority'),
        acceptance=dict(raw_evidence='bind raw run/control JSON hashes, model/runtime/build/allocation identities, '
            'artifact bytes/array hashes, all planned case IDs and failure counts; do not trust exit status alone',
            numeric_thresholds=copy.deepcopy(rule['numeric_thresholds']),
            semantic=copy.deepcopy(protocol['quality']),
            reuse=dict(existing_metric='placement/window overlap proxy only; interval unions do not establish release ACK or CUDA busy time',
                controls=['same fold schedule/budget with fold-only', 'forced fission', 'demand fission',
                          'no pending beneficiary', 'compatible pending beneficiary', 'incompatible pending beneficiary'],
                events=['request/step/epoch/lease identity', 'old/new/released physical GPU UUIDs',
                        'release ACK timestamp', 'reacquire/restore timestamp',
                        'beneficiary dispatch and actual execution intervals with completed finite output',
                        'export/import bytes and latency, group initialization and restore costs'],
                accounting='union intervals per physical GPU for release and beneficiary execution; '
                           'reused GPU-ms <= released GPU-ms; include video beneficiaries; '
                           'wall reservation overlap alone is not CUDA busy time',
                utility='same frozen mixed trace and fold decisions; compare per-modality SLO goodput and net overhead '
                        'against fold-only; queue suffix nonempty is not feasibility evidence')),
        time_planning=dict(serial_sp1_reference_ms=sum(services[sid] for sid in ids) * count * len(seeds) * len(arms),
            scope='calibrated SP1 work planning only; no assumed folding or multi-GPU speedup'),
        storage=storage, blockers=blockers))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['storage', 'prepare'])
    parser.add_argument('--protocol', type=Path, default=formal.DEFAULT_PROTOCOL)
    parser.add_argument('--model', choices=sorted(MODELS), required=True)
    parser.add_argument('--rule', type=Path, required=True)
    parser.add_argument('--storage-snapshot', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--datasets', type=Path)
    parser.add_argument('--model-manifests', type=Path)
    parser.add_argument('--gpu-uuids')
    parser.add_argument('--prompt-selection-seed', type=int)
    parser.add_argument('--form', default='cfg2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol, rule, snapshot = formal.protocol(args.protocol), read_json(args.rule), read_json(args.storage_snapshot)
    if args.mode == 'storage':
        result = seal(dict(protocol_sha256=canonical_hash(protocol), model=args.model,
                           **storage_budget(protocol, args.model, rule, snapshot)))
    else:
        required = ('bundle', 'datasets', 'model_manifests', 'gpu_uuids', 'prompt_selection_seed')
        if any(getattr(args, key) is None for key in required):
            parser.error('prepare requires bundle, datasets, model-manifests, gpu-uuids and prompt-selection-seed')
        pools = {'video': read_json(args.datasets / (protocol['datasets']['video']['kind'] + '.json'))}
        result = prepare(protocol, args.model, args.gpu_uuids.split(','), read_json(args.bundle), pools,
                         read_json(args.model_manifests), rule, snapshot,
                         prompt_selection_seed=args.prompt_selection_seed, form=args.form)
    formal.write_new(args.output, result)
    print(str(args.output) + ': CPU planning only; no execution or qualification authorized')


if __name__ == '__main__':
    main()
