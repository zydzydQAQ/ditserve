#!/usr/bin/env python3
"""Compile/review an independent finite offline CFG2 diagnostic on CPU.

The explicit inputs manifest binds local files by byte SHA256. Its identity
file freezes the expected core, Python inventory and official-model mapping;
it is an operator input, not a substitute for a verified P0 submission gate.
No downloads, GPU probes, execution, stage promotion or overwrite occurs here.

Inputs: {schema: ditserve.p1-isolated-inputs.v1, pair_id, gpu_uuids,
protocol, pair_plan, bundle, model_manifests, identity, runtime_facts,
pools: {video, image}, calibrations: [...]}. Every source is {path, sha256}.
Identity: {schema: ditserve.p1-isolated-identity.v1, core_sha256,
python_sources: {relative_python_path: sha256}, runtime_identity_sha256,
model_manifests_sha256}. The latter two are canonical JSON hashes.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import formal_benchmark as formal
from native_serving import FORM_DEGREES, build_serving_config
from p1_benchmark import config_identity, match_native_config, source_file, validate_plan
from ditserve.evaluation.common import canonical_hash, read_json, verify_seal
from ditserve.evaluation.workloads import build_trace, validate_trace

ID = 'p1-isolated-batch-v1'
SCHEMA = 'ditserve.p1-isolated-inputs.v1'
POLICIES = ['edf-cfg2', 'block-cfg2']
SEEDS = [201, 202, 203]
COUNT = 8
SCOPE = ('Development prompts, formal shapes, two resident GPUs, fixed CFG2, finite offline batch. '
         'Diagnostic only: not low-load Poisson, stable capacity, semantic quality or P1 qualification. '
         'Two timing repeats are nested within each seed; retain all failed/unexecuted requests.')


def load_inputs(path, expected_sha256):
    """Resolve only explicitly supplied sources; never infer a latest result."""
    path = Path(path)
    raw = source_file(path.parent, dict(path=path.name, sha256=expected_sha256))
    manifest = json.loads(raw)
    if manifest.get('schema') != SCHEMA:
        raise ValueError('unsupported isolated-batch inputs manifest')
    refs = ('protocol', 'pair_plan', 'bundle', 'model_manifests', 'identity', 'runtime_facts')
    values = {key: json.loads(source_file(path.parent, manifest[key])) for key in refs}
    values['pools'] = {kind: json.loads(source_file(path.parent, manifest['pools'][kind]))
                       for kind in ('video', 'image')}
    values['calibrations'] = [json.loads(source_file(path.parent, ref))
                              for ref in manifest['calibrations']]
    values.update(pair_id=manifest['pair_id'], gpu_uuids=manifest['gpu_uuids'],
                  inputs_manifest_sha256=hashlib.sha256(raw).hexdigest())
    return values



def settled_ledger(record, binding, uuids):
    """Bind current phases, retaining and checking the cumulative allocation history.

    Older runs keep their own native config digest. Their charges are not
    evidence for this run; callers with raw predecessors must bind the prefix.
    """
    budget = record.get('budget', {})
    sessions = budget.get('sessions')
    if (budget.get('schema_version') != 'ditserve.budget.v1'
            or not budget.get('allocation_id') or budget.get('active', 'missing') is not None
            or not isinstance(sessions, list)):
        raise ValueError('allocation ledger is missing or unsettled')
    phases = record.get('phases', [])
    selected = {}
    for phase in phases:
        sid = phase.get('session_id')
        if not sid or sid in selected:
            raise ValueError('phase lacks a unique allocation session')
        selected[sid] = phase
    seen, current, history = set(), set(), []
    regular, repair, usage = 0., 0., {}
    previous_end = None
    statuses = {'completed', 'failed', 'interrupted', 'timed_out', 'cancelled', 'incomplete', 'censored'}

    def number(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid allocation charge/time')
        return value

    def same(left, right):
        return math.isclose(number(left), number(right), rel_tol=1e-12, abs_tol=1e-6)

    for session in sessions:
        sid, owner = session.get('session_id'), session.get('identity', {})
        if not sid or sid in seen:
            raise ValueError('allocation session identity is missing or duplicate')
        seen.add(sid)
        if (session.get('cleanup_verified') is not True or session.get('status') not in statuses
                or type(session.get('repair')) is not bool):
            raise ValueError('allocation history is not fully settled')
        if (owner.get('gpu_uuids') != uuids or session.get('devices') != len(uuids)
                or owner.get('source_sha256') != canonical_hash(record.get('source', {}))
                or not owner.get('run_id') or type(owner.get('attempt')) is not int or owner['attempt'] < 1
                or not formal.HEX64.fullmatch(owner.get('config_sha256', ''))):
            raise ValueError('allocation history source/GPU/native config hash identity differs')
        start, end = (number(session.get(k)) for k in ('started_monotonic', 'ended_monotonic'))
        elapsed, charge = number(session.get('elapsed_seconds')), number(session.get('gpu_seconds'))
        if (end < start or previous_end is not None and start < previous_end
                or not same(end - start, elapsed) or not same(charge, elapsed * len(uuids))):
            raise ValueError('allocation history time/device-seconds differ')
        previous_end = end
        if session['repair']:
            repair += charge
        else:
            regular += charge
            pid = session.get('phase')
            if not isinstance(pid, str) or not pid:
                raise ValueError('allocation history phase is missing')
            usage[pid] = usage.get(pid, 0.) + charge
        if sid in selected:
            phase = selected[sid]
            match_native_config(binding, owner)
            if (owner['run_id'] != record.get('run_id') or owner['attempt'] != phase.get('attempt')
                    or session.get('phase') != phase.get('id') or session['status'] != phase.get('status')
                    or phase.get('charge') != session):
                raise ValueError('current phase/session run/attempt/source/charge identity differs')
            current.add(sid)
        else:
            if current or owner['run_id'] == record.get('run_id'):
                raise ValueError('allocation history is not a preceding run prefix')
            history.append(session)
    if current != set(selected) or record.get('requests') and not current:
        raise ValueError('phase lacks a unique allocation session')
    actual_usage = budget.get('phase_usage', {})
    if (not same(budget.get('used_regular_gpu_seconds'), regular)
            or not same(budget.get('used_repair_gpu_seconds'), repair)
            or not same(budget.get('used_gpu_seconds'), regular + repair)
            or set(actual_usage) != set(usage) or any(not same(actual_usage[k], v) for k, v in usage.items())):
        raise ValueError('allocation cumulative charge history differs')
    return history


def calibration_history(record, binding, uuids, records):
    """Authenticate a cumulative prefix against independently supplied raw runs."""
    history = settled_ledger(record, binding, uuids)
    if history and not any(other is not record
            and other.get('budget', {}).get('allocation_id') == record['budget']['allocation_id']
            and other['budget'].get('limits') == record['budget'].get('limits')
            and other['budget'].get('sessions') == history for other in records):
        raise ValueError('calibration history lacks its exact raw predecessor prefix')


def validate_sources(inputs):
    p, plan, bundle, identity = (inputs[k] for k in ('protocol', 'pair_plan', 'bundle', 'identity'))
    if not formal.HEX64.fullmatch(inputs.get('inputs_manifest_sha256', '')):
        raise ValueError('explicit frozen inputs manifest SHA256 required')
    validate_plan(plan, p)
    if plan['seeds'] != SEEDS or len(plan['shape_pairs']) != 4:
        raise ValueError('isolated v1 requires the four frozen pairs and seeds201/202/203')
    pair = next((v for v in plan['shape_pairs'] if v['id'] == inputs['pair_id']), None)
    if pair is None:
        raise ValueError('unknown formal shape pair')
    uuids, manifests = inputs['gpu_uuids'], inputs['model_manifests']
    if len(uuids) != 2 or len(set(uuids)) != 2 or any(not isinstance(v, str) or not v for v in uuids):
        raise ValueError('isolated v1 requires exactly two distinct allocated GPU UUIDs')
    files = identity.get('python_sources')
    if (identity.get('schema') != 'ditserve.p1-isolated-identity.v1'
            or not formal.HEX64.fullmatch(identity.get('core_sha256', ''))
            or not formal.HEX64.fullmatch(identity.get('runtime_identity_sha256', ''))
            or identity.get('model_manifests_sha256') != canonical_hash(manifests)
            or not isinstance(files, dict) or not files):
        raise ValueError('frozen runtime or official model mapping identity differs')
    for name, digest in files.items():
        path = PurePosixPath(name)
        if (str(path) != name or path.parts[0] not in ('ditserve', 'xfuser') or '..' in path.parts
                or '\\' in name or path.suffix != '.py' or not formal.HEX64.fullmatch(digest)):
            raise ValueError('invalid frozen Python source inventory')
    verify_seal(bundle)
    if (bundle.get('model') != pair['model'] or bundle.get('gpu_uuids') != uuids
            or bundle.get('protocol_sha256') != canonical_hash(p)
            or bundle.get('model_manifests') != manifests):
        raise ValueError('bundle model, protocol, allocation or model manifest differs')
    records = inputs['calibrations']
    if not records:
        raise ValueError('raw calibration source records required')
    for record in records:
        if (record.get('status') != 'completed' or record.get('failures')
                or record.get('source', {}).get('core_sha256') != identity['core_sha256']
                or record['source'].get('files') != files):
            raise ValueError('calibration runtime/source/config differs or did not complete')
        ids = record['config']['benchmark'].get('formal_protocol', {}).get('shape_ids', [])
        if len(ids) != 2 or any(p['shapes'][sid]['scope'] != 'formal' for sid in ids):
            raise ValueError('calibration must execute formal pair shapes')
        expected_config = formal.calibration(p, pair['model'], uuids, inputs['pools'], manifests, *ids)
        config_binding = config_identity(record, expected_config)
        calibration_history(record, config_binding, uuids, records)
        cal_phase = expected_config['phases'][0]
        phases = record.get('phases', [])
        if (len(phases) != 1 or phases[0].get('id') != cal_phase['id']
                or phases[0].get('status') != 'completed'
                or type(phases[0].get('attempt')) is not int or phases[0]['attempt'] < 1):
            raise ValueError('calibration phase execution identity/status differs')
        executed = phases[0]
        policies = cal_phase['matrix']['/policy']
        if len(executed.get('cells', [])) != len(policies):
            raise ValueError('calibration execution cells are incomplete')
        for index, (cell, policy) in enumerate(zip(executed['cells'], policies)):
            if (cell.get('cell') != index or cell.get('order') != 'forward'
                    or cell.get('policy') != policy or cell.get('status') != 'completed'):
                raise ValueError('calibration cell policy/order/status differs')
        wanted = {(i, r['request_id']) for i, _ in enumerate(cal_phase['matrix']['/policy'])
                  for r in cal_phase['workload']}
        actual = {(r['cell'], r['request_id']) for r in record['requests']}
        if actual != wanted or len(record['requests']) != len(wanted):
            raise ValueError('calibration lost requests or policy denominators')
        sessions = record.get('budget', {}).get('sessions', [])
        if not sessions or any(s.get('identity', {}).get('gpu_uuids') != uuids for s in sessions):
            raise ValueError('calibration GPU allocation differs')
        for row in record['requests']:
            planned = next(r for r in cal_phase['workload'] if r['request_id'] == row['request_id'])
            index = row['cell']
            policy = policies[index]
            normalize = lambda spec: {k: v for k, v in spec.items() if k != 'request_id'}
            if (row.get('phase') != cal_phase['id'] or row.get('attempt') != executed['attempt']
                    or row.get('order') != 'forward'
                    or row.get('observation_id') != f"{cal_phase['id']}/{executed['attempt']}/{index}"
                    or row.get('policy', policy) != policy
                    or any(row.get(k) != planned[k] for k in ('kind', 'class', 'arrival_ms', 'deadline_ms'))
                    or normalize(row['spec']) != normalize(planned['spec'])):
                raise ValueError('calibration row differs from planned input/cell execution')
            if row['kind'] == 'video':
                form = policy.removeprefix('static_')
                samples = row.get('advance_samples', [])
                group = list(range(FORM_DEGREES[form]))
                if (row.get('forms') != [form] or not samples or any(
                        s.get('form') != form or s.get('group') != group for s in samples)):
                    raise ValueError('calibration executed form/group differs from its policy')
            generation = row.get('outputs', [{}])[-1].get('generation', {})
            artifact = row.get('outputs', [{}])[-1].get('output', {})
            model = pair['model'] if row['kind'] == 'video' else 'sd35'
            spec = row['spec']
            shape = [1] + ([spec['frames']] if row['kind'] == 'video' else []) + [spec['height'], spec['width'], 3]
            if (row['status'] != 'completed' or artifact.get('finite') is not True or artifact.get('shape') != shape
                    or generation.get('model_manifest_sha256') != manifests[model]['sha256']
                    or generation.get('model_integrity') != 'manifest_verified'
                    or not generation.get('runtime_identity')
                    or canonical_hash(generation['runtime_identity']) != identity['runtime_identity_sha256']
                    or any(generation.get(k) != v for k, v in spec.items() if k != 'request_id')):
                raise ValueError('calibration output lacks executed input/model/finite-shape evidence')
            if 'model_path' in manifests[model] and generation.get('model_path') != manifests[model]['model_path']:
                raise ValueError('calibration output model snapshot differs')
    # A resealed edited cost table must not authenticate itself. Refit the
    # explicitly hash-bound calibration records using the bound architecture.
    reconstructed = formal.assemble(p, pair['model'], records, inputs['runtime_facts'])
    if reconstructed != bundle:
        raise ValueError('bundle differs from its raw calibration sources')
    for sid in (pair['video_shape'], pair['image_shape']):
        if sid not in bundle['classes'] or bundle['classes'][sid]['spec'] != formal.shape_spec(p, sid):
            raise ValueError('each exact formal pair shape requires its own calibration')
    return pair


def compile_case(inputs):
    """One pair, three seeds, four actual cells per seed (A,B,B,A)."""
    pair = validate_sources(inputs)
    p, uuids, bundle = inputs['protocol'], inputs['gpu_uuids'], inputs['bundle']
    selected = formal.split_pools(inputs['pools'], p, 'development')
    if len(selected['video']['samples']) < COUNT:
        raise ValueError('insufficient distinct development video prompts; no recycling')
    ids = [pair['video_shape'], pair['image_shape']]
    entries = {sid: bundle['classes'][sid] for sid in ids}
    classes, costs = {}, {}
    for sid, entry in entries.items():
        cost, spec = copy.deepcopy(entry['cost']), formal.shape_spec(p, sid)
        gpu_ms = (cost.get('prepare_ms', 0) + spec['steps'] * cost['step_compute_ms'] + cost['finish_ms']
                  if cost['kind'] == 'video' else cost['service_ms'])
        service = gpu_ms + (cost.get('encode_ms', 0) if cost['kind'] == 'video' else 0)
        if not math.isfinite(service) or service <= 0:
            raise ValueError('calibrated service must be finite and positive')
        costs[sid] = cost
        classes[sid] = dict(cost, **{'class': sid, 'spec': spec, 'service_ms': service,
            'gpu_service_ms': gpu_ms, 'slo_scale': p['slo'][cost['kind'] + '_scale']})
    profile = dict(provenance='mixed', classes=costs, links=copy.deepcopy(costs[ids[0]]['links']),
        resize_ms=max(e['source']['resize_ms'] for e in entries.values()),
        quantum_overhead_ms=max(e['source']['quantum_overhead_ms'] for e in entries.values()),
        block_overhead_ms=0., source=dict(gpu_uuids=uuids, shape=formal.shape_spec(p, ids[0]),
                                        bundle_sha256=bundle['sha256']))
    singleton = copy.deepcopy(profile)
    singleton['classes'] = {pair['model'] + '-preflight': costs[ids[0]], 'sd35-preflight': costs[ids[1]]}
    config = build_serving_config(pair['model'], uuids, singleton, selected, seed=SEEDS[0], load=1.,
        pattern='offline_all', video_fraction=1., count=COUNT, policies=POLICIES, forms=['cfg2'],
        doorbell=True, output_mode='digest', require_profile_match=True, policy_order='forward',
        video_slo_scale=p['slo']['video_scale'], image_slo_scale=p['slo']['image_scale'])
    template = config['phases'][0]
    template.update(profile=profile, paired=True, output_mode='digest')
    template['options']['lazy_preemption'] = True
    # Retain the same image residency/warmup as P1, but no image measurements.
    template['warmup'] = [dict(worker=('wan' if classes[sid]['kind'] == 'video' else 'image') + str(i),
        spec=dict(formal.shape_spec(p, sid), steps=1, prompt='warmup', seed=7), quantum='step')
        for sid in ids for i in range(2)]
    phases, traces = [], {}
    for seed in SEEDS:
        trace = build_trace(classes, selected, [dict(count=COUNT, video_fraction=1., pattern='offline_all')],
                            slots=2, seed=seed, output_mode='digest')
        # Offline batches have no offered arrival rate or normalized load.
        trace['manifest']['normalized_gpu_work'] = None
        for window in trace['manifest']['windows']:
            window['load'] = None
        validate_trace(trace)
        provenance = dict(path='generated:' + ID, sha256=trace['manifest']['workload_sha256'], manifest=trace['manifest'])
        phase = copy.deepcopy(template)
        phase.update(id=f'seed-{seed}', watchdog_seconds=math.ceil(4 * (
            600 + 1.5 * sum(r['service_ms'] for r in trace['workload']) / 1000)))
        phase['matrix'] = {'/policy': POLICIES[:], '/workload': [trace['workload']], '/trace_provenance': [provenance]}
        phases.append(phase)
        traces[str(seed)] = provenance
    config['experiment'] = ID + '-' + pair['id']
    config['phases'] = phases
    config['budget'].update(watchdog_seconds=max(v['watchdog_seconds'] for v in phases),
        limit_gpu_seconds=sum(v['watchdog_seconds'] * 2 for v in phases),
        stage_limits={v['id']: v['watchdog_seconds'] * 2 for v in phases})
    meta = config['benchmark']
    for key in ('load', 'seed', 'trace', 'policy_order'):
        meta.pop(key, None)
    meta.update(diagnostic_only=True, qualification_eligible=False, scope=SCOPE, mixed_resolution=False,
        formal_protocol=formal.binding(p, 'development', 'p1_isolated_batch', ids, selected, bundle),
        isolated_batch=dict(id=ID, pair=copy.deepcopy(pair), pair_plan_sha256=canonical_hash(inputs['pair_plan']),
            seeds=SEEDS[:], timing_repeats=2, order='AB/BA via native paired=true', requests_per_cell=COUNT,
            expected_cells=12, expected_policy_requests=96, expected_image_requests=0, require_zero_stops=True,
            study_shape_pairs=copy.deepcopy(inputs['pair_plan']['shape_pairs']),
            study_expected_cells=48, study_expected_policy_requests=384,
            requires_stage='P0', submission_action='p1_diagnostics', allows=[], traces=traces,
            identity=copy.deepcopy(inputs['identity']), inputs_manifest_sha256=inputs['inputs_manifest_sha256'],
            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            budget_scope='4*(600s startup + 1.5*sum SP1 end-to-end service) per phase; not a measured bound'))
    formal.bind_manifests(config, pair['model'], inputs['model_manifests'])
    return config


def audit_config(config, inputs):
    """Recompile all settings, not just self-reported hashes or labels."""
    if config != compile_case(inputs):
        raise ValueError('configuration differs from the frozen isolated-batch inputs')
    return planned_cells(config)


def planned_cells(config):
    """Enumerate the full failure denominator before any cells start."""
    cells = []
    for phase in config['phases']:
        for repeat, order, indices in ((0, 'forward', (0, 1)), (1, 'reverse', (1, 0))):
            for offset, cell in enumerate(indices):
                cells.append(dict(phase=phase['id'], cell=cell, order=order, timing_repeat=repeat,
                    position=repeat * 2 + offset, policy=phase['matrix']['/policy'][cell], expected_requests=COUNT))
    return cells


def coverage(config, record=None, *, attempt=1):
    """Coverage only; completion is never numerical or scientific acceptance."""
    if type(attempt) is not int or attempt < 1:
        raise ValueError('select one explicit positive attempt')
    if record is not None:
        config_binding = config_identity(record, config)
        identity = config['benchmark']['isolated_batch']['identity']
        source = record.get('source', {})
        if (source.get('core_sha256') != identity['core_sha256']
                or source.get('files') != identity['python_sources']):
            raise ValueError('executed runtime source differs from frozen inputs')
        settled_ledger(record, config_binding, config['benchmark']['profile_placement']['gpu_uuids'])
    expected = {(c['phase'], c['cell'], c['order']): c for c in planned_cells(config)}
    # Native cell numbers repeat on the reverse pass. Only their position in
    # one explicit phase attempt, bound by each row's observation_id, identifies
    # the second execution. Failed native cells may omit policy/order; their
    # materialized prefix remains visible but is never a completed ABBA cell.
    executed, phase_keys = {}, set()
    for phase in (record or {}).get('phases', []):
        phase_id, phase_attempt = phase.get('id'), phase.get('attempt')
        if (phase_id not in {p['id'] for p in config['phases']}
                or type(phase_attempt) is not int or phase_attempt < 1
                or (phase_id, phase_attempt) in phase_keys):
            raise ValueError('invalid or duplicate executed phase attempt')
        phase_keys.add((phase_id, phase_attempt))
        if phase_attempt != attempt:
            continue
        planned = [v for v in expected.values() if v['phase'] == phase_id]
        actual = phase.get('cells', [])
        if len(actual) > len(planned):
            raise ValueError('unexpected executed cell position')
        for position, cell in enumerate(actual):
            plan = planned[position]
            complete = cell.get('status') == 'completed'
            if (cell.get('status') not in ('completed', 'failed')
                    or cell.get('cell') != plan['cell']
                    or cell.get('policy', plan['policy'] if not complete else None) != plan['policy']
                    or cell.get('order', plan['order'] if not complete else None) != plan['order']
                    or cell.get('attempt', attempt) != attempt
                    or cell.get('observation_id', f'{phase_id}/{attempt}/{position}') != f'{phase_id}/{attempt}/{position}'
                    or not complete and position != len(actual) - 1):
                raise ValueError('executed cell policy/order/attempt/position differs')
            executed[phase_id, plan['cell'], plan['order']] = cell
        if phase.get('status') == 'completed' and (len(actual) != 4 or any(c['status'] != 'completed' for c in actual)):
            raise ValueError('completed phase lacks four completed ABBA cells')
    observed = {}
    for row in (record or {}).get('requests', []):
        if type(row.get('attempt')) is not int or row['attempt'] < 1:
            raise ValueError('request lacks an explicit native attempt')
        if row.get('attempt') != attempt:
            continue
        key = (row.get('phase'), row.get('cell'), row.get('order'))
        if key not in executed:
            raise ValueError('unexpected executed phase/cell/order')
        plan = expected[key]
        if (row.get('observation_id') != f"{plan['phase']}/{attempt}/{plan['position']}"
                or row.get('policy', plan['policy']) != plan['policy']):
            raise ValueError('request observation_id/policy differs from its native cell')
        if executed[key]['status'] == 'completed' and row.get('status') != 'completed':
            raise ValueError('completed cell contains an incomplete request')
        phase = next(p for p in config['phases'] if p['id'] == key[0])
        planned = {r['request_id']: r for r in phase['matrix']['/workload'][0]}
        rid = row.get('request_id')
        if rid not in planned or (key, rid) in observed:
            raise ValueError('unexpected or duplicate request identity')
        if any(row.get(k) != planned[rid][k] for k in ('kind', 'class', 'spec', 'arrival_ms', 'deadline_ms', 'dataset_sample_id')):
            raise ValueError('executed request input/deadline identity differs')
        observed[key, rid] = row
    cells = []
    for key, plan in expected.items():
        rows = [r for (selector, _), r in observed.items() if selector == key]
        status = executed.get(key, {}).get('status', 'unexecuted')
        if status == 'completed' and len(rows) != COUNT:
            raise ValueError('completed cell lacks its complete request denominator')
        complete = sum(r.get('status') == 'completed' for r in rows)
        cells.append(dict(plan, execution_status=status, observed=len(rows), completed=complete,
                          failed_or_incomplete=len(rows)-complete, unexecuted=COUNT-len(rows)))
    return dict(schema='ditserve.p1-isolated-coverage.v1', diagnostic_only=True, qualification_eligible=False,
        scope='coverage only; outputs, zero stops, timing and P0 submission eligibility are not accepted here',
        config_identity=config_binding if record is not None else None,
        attempt=attempt, cells=cells, failures=copy.deepcopy((record or {}).get('failures', [])),
        abba_cells_completed=sum(c['execution_status'] == 'completed' for c in cells),
        expected_requests=96, completed=sum(c['completed'] for c in cells),
        failed_or_incomplete=sum(c['failed_or_incomplete'] for c in cells), unexecuted=sum(c['unexecuted'] for c in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='review this existing config instead of compiling')
    parser.add_argument('--run', type=Path, help='optional coverage only; requires --config')
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.run and not args.config:
        parser.error('--run requires --config')
    inputs = load_inputs(args.inputs, args.inputs_sha256)
    if args.config:
        config = read_json(args.config)
        audit_config(config, inputs)
        result = coverage(config, read_json(args.run) if args.run else None, attempt=args.attempt)
    else:
        result = compile_case(inputs)
    formal.write_new(args.output, result)
    print(ID + ': CPU preparation/review only; verified P0 required before submission')


if __name__ == '__main__':
    main()
