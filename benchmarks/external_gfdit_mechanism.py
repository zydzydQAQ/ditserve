"""GF-DiT's original EDF decision core over a sealed common workload.

This is a mechanism reference, not a worker, simulator, or full GF-DiT runtime.
It consumes explicit submit/launch/end events; costs NEVER generate events or
model outputs. Original policy/SP selection and original chunked-DiT DAG code
are imported from the source-verified namespace-only port. Two explicit bridges
provide absolute client-clock metadata and request-specific cost lookup. This
avoids native submission-time deadline renewal and cross-model cost aliasing.
The host event adapter publishes DAG dependencies on launch_end, like the
original scheduler, but has no artifacts, GFC, CUDA, output fetch or transport.
Every offered request remains in the accounting; no SLO/throughput is computed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import external_comparison as common
import external_gfdit_shared as source

SCHEMA = 'ditserve.gfdit-mechanism-reference.v1'
COST_SCHEMA = 'ditserve.gfdit-mechanism-costs.v1'
EVENT_SCHEMA = 'ditserve.gfdit-mechanism-events.v1'
POLICIES = ('edf_best_fit', 'edf_greedy')
STAGES = ('text_encode', 'dit_prepare', 'timestep_prepare', 'dit_step_chunk_cfg',
          'dit_step_chunk_nocfg', 'vae_decode', 'finalize')


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def generation_profile(request):
    return {k: v for k, v in request['spec'].items() if k not in ('prompt', 'seed', 'request_id')}


def evidence_bindings(provenance, *, allow_fixture=False):
    """Check named evidence bytes, not calibration quality or GPU execution."""
    if not isinstance(provenance, dict):
        raise ValueError('explicit input provenance is required')
    kind = provenance.get('kind')
    if kind == 'fixture':
        if not allow_fixture or not provenance.get('description'):
            raise ValueError('fixture input needs explicit opt-in and description')
        return dict(kind='fixture', verified_files=[], measurement_claim=False)
    if kind != 'external_observation' or not provenance.get('files'):
        raise ValueError('external observations need hash-bound evidence files')
    checked = []
    for row in provenance['files']:
        path = Path(row['path'])
        if path.is_symlink() or not path.is_file() or not common.hex64(row.get('sha256')):
            raise ValueError('invalid observed-evidence file')
        if common.file_sha(path.read_bytes()) != row['sha256']:
            raise ValueError('observed-evidence SHA256 differs')
        checked.append(dict(path=str(path.resolve()), sha256=row['sha256']))
    return dict(kind=kind, verified_files=checked,
        measurement_claim=False, scope='byte provenance only; calibration/execution not certified')


class CostBook:
    """Strict per-model/shape native polynomial costs; no missing-stage fallback."""
    def __init__(self, control, workload, document, allowed_sp, *, allow_fixture=False):
        if document.get('schema') != COST_SCHEMA or document.get('workload_sha256') != workload['sha256']:
            raise ValueError('cost input is not bound to this workload')
        if not common.hex64(document.get('backend_profile_sha256')):
            raise ValueError('costs need an explicit backend profile identity')
        self.provenance = evidence_bindings(document.get('provenance'), allow_fixture=allow_fixture)
        self.document = copy.deepcopy(document)
        self.requests = {r['request_id']: r for r in copy.deepcopy(workload['requests'])}
        self.profiles, self.models, self.context = {}, {}, None
        for profile in document.get('profiles', []):
            key = (profile['model'], profile['shape_id'])
            if key in self.profiles:
                raise ValueError('duplicate model/shape cost profile')
            rows = [r for r in workload['requests'] if (r['model'], r['shape_id']) == key]
            if not rows or any(profile.get('generation_profile') != generation_profile(r)
                               or profile.get('model_manifest_sha256') != r['model_manifest']['sha256'] for r in rows):
                raise ValueError('cost model/shape/generation identity differs')
            for name in ('latent_seq_len', 'text_seq_len'):
                positive_int(profile.get(name), name)
            schedule = profile.get('dit_step_do_true_cfg')
            if (not isinstance(schedule, list) or len(schedule) != rows[0]['spec']['steps']
                    or any(type(x) is not bool for x in schedule)):
                raise ValueError('explicit per-step CFG schedule required')
            found = set()
            for record in profile.get('models', []):
                parallel = record.get('parallelism', {})
                sp = parallel.get('sp')
                if type(sp) is not int or sp not in allowed_sp or sp in found or parallel != dict(tp=1, sp=sp, ulysses=sp, ring=1, cfg=1):
                    raise ValueError('duplicate/unsupported SP cost identity')
                found.add(sp)
                stages = record.get('stages', {})
                required = {'dit_prepare', 'timestep_prepare'} | {('dit_step_chunk_cfg' if x else 'dit_step_chunk_nocfg') for x in schedule}
                if sp == 1:
                    required |= {'text_encode', 'vae_decode', 'finalize'}
                if not required <= set(stages) or set(stages) - set(STAGES):
                    raise ValueError('missing or unsupported explicit stage cost')
                for stage, value in stages.items():
                    coeffs = value.get('coeffs', {})
                    if set(coeffs) != {'a', 'b', 'c'} or any(type(v) not in (int, float) or not math.isfinite(v) for v in coeffs.values()):
                        raise ValueError('all finite polynomial coefficients must be explicit')
                native_key = control.cost.CostModelKey(tp=1, sp=sp, ulysses=sp, ring=1, cfg=1)
                self.models[key + (sp,)] = control.cost.StageCostModel(Path('<bound-mechanism-costs>'), native_key, copy.deepcopy(stages))
            if found != set(allowed_sp):
                raise ValueError('missing SP cost coverage')
            self.profiles[key] = copy.deepcopy(profile)
        expected = {(r['model'], r['shape_id']) for r in workload['requests']}
        if set(self.profiles) != expected:
            raise ValueError('missing model/shape cost coverage')

    def profile(self, request_id):
        request = self.requests[request_id]
        return self.profiles[(request['model'], request['shape_id'])]

    @contextmanager
    def for_request(self, request_id):
        if self.context is not None:
            raise ValueError('nested or concurrent cost context is unsupported')
        self.context = request_id
        try:
            yield
        finally:
            self.context = None

    def estimate(self, spec, x, stage):
        if self.context is None or (spec.tp, spec.cfg) != (1, 1):
            raise ValueError('cost request context or parallel identity is missing')
        request = self.requests[self.context]
        model = self.models[(request['model'], request['shape_id'], spec.sp)]
        if stage not in model.stages:
            raise ValueError('no implicit missing-stage or CFG cost fallback')
        value = model.estimate_ms(stage, x)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('stage cost must be finite and strictly positive')
        return value


def compile_plan(control, request, profile, chunk_size):
    """Preserve the entire request; token metadata is supplied by the bound profile."""
    spec = request['spec']
    metadata = dict(frozen_request=copy.deepcopy(request), latent_seq_len=profile['latent_seq_len'],
        text_seq_len=profile['text_seq_len'], voxels=spec['height'] * spec['width'] * spec['frames'],
        num_steps=spec['steps'], denoise_chunk_size=chunk_size)
    value = SimpleNamespace(request_id=request['request_id'], runtime_v2_arrival_ms=request['arrival_ms'],
        runtime_v2_deadline_ms=request['deadline_ms'], runtime_v2_priority=0)
    return control.common.build_chunked_dit_plan(request_id=request['request_id'], request_value=value,
        request_type='common-workload-mechanism-only', num_steps=spec['steps'], chunk_size=chunk_size,
        priority=0, group_id=None, text_seq_len=profile['text_seq_len'], latent_seq_len=profile['latent_seq_len'],
        state_codec_id='mechanism-no-tensor-codec', decoded_codec_id=None, metadata=metadata,
        dit_step_do_true_cfg=tuple(profile['dit_step_do_true_cfg']))


def make_policy(control, topology, costs, allowed_sp, name):
    if name not in POLICIES:
        raise ValueError('unsupported original policy')
    module = importlib.import_module(source.NAMESPACE + '.policies.' + name)
    base = module.EdfBestFitSchedulerPolicy if name == 'edf_best_fit' else module.EdfGreedySchedulerPolicy
    greedy = importlib.import_module(source.NAMESPACE + '.policies.edf_greedy')

    class AbsoluteClockPolicy(base):
        # These are the entire executable changes relative to original policy:
        # frozen metadata clock, and request context around original cost calls.
        def _meta_from_plan(self, plan, order):
            request = plan.metadata['frozen_request']
            fields = dict(deadline_ms=request['deadline_ms'], arrival_ms=request['arrival_ms'], priority=0, order=order)
            if name == 'edf_best_fit':
                fields.update({k: plan.metadata[k] for k in ('latent_seq_len', 'voxels', 'text_seq_len', 'num_steps')})
                fields['chunk_steps'] = plan.metadata['denoise_chunk_size']
                return module._EdfBestFitRequestMeta(**fields)
            return greedy._EdfGreedyRequestMeta(**fields)

        def _estimate_finish_ms(self, request_id, sp_size):
            with costs.for_request(request_id):
                return super()._estimate_finish_ms(request_id, sp_size)

        def _select_dit_group(self, task, free_ranks):
            with costs.for_request(task.request_id):
                return super()._select_dit_group(task, free_ranks)

    return AbsoluteClockPolicy(topology, costs.estimate, allowed_sp_sizes=allowed_sp)


class DecisionDriver:
    """Single-threaded decision API for an eventual real executor integration.

    submit(request_id, now_ms), event(task_id, kind, now_ms), hold(id, ranks,
    begin, now_ms), and poll(now_ms) return decisions, never model completions.
    Caller timestamps are absolute offsets from the common origin. A future
    executor must attest actual task/transfer completion and decoded results;
    this adapter cannot do that. Migration holds here are metadata only.
    """
    def __init__(self, control, workload, cost_document, *, world_size, allowed_sp=(1, 2), chunk_size=2,
                 policy='edf_best_fit', allow_fixture=False):
        common.verify_export(workload)
        positive_int(world_size, 'world_size'); positive_int(chunk_size, 'chunk_size')
        if (not allowed_sp or any(type(s) is not int or not 1 <= s <= world_size for s in allowed_sp)
                or len(set(allowed_sp)) != len(allowed_sp) or 1 not in allowed_sp):
            raise ValueError('allowed SP sizes must be unique, include SP1, and fit the world')
        self.control, self.p = control, control.protocol
        self.workload, self.policy_name = copy.deepcopy(workload), policy
        self.requests = {r['request_id']: r for r in self.workload['requests']}
        self.costs = CostBook(control, workload, cost_document, tuple(allowed_sp), allow_fixture=allow_fixture)
        self.plans = {rid: compile_plan(control, r, self.costs.profile(rid), chunk_size) for rid, r in self.requests.items()}
        self.tasks = {tid: t for plan in self.plans.values() for tid, t in plan.tasks.items()}
        self.topology = control.topology.RuntimeTopology(
            tuple(control.topology.WorkerSpec(rank, rank) for rank in range(world_size)),
            tuple(self.p.ExecutionGroupSpec('initial_' + str(rank), (rank,), self.p.ParallelSpec(), control.common.STANDARD_TASK_KINDS)
                  for rank in range(world_size)))
        # Preflight every expression the native policy may use, before its
        # _safe_estimate can convert a missing/invalid estimate to zero.
        resolver = importlib.import_module(source.NAMESPACE + '.policies.common')
        for rid, plan in self.plans.items():
            m = plan.metadata
            with self.costs.for_request(rid):
                for sp in allowed_sp:
                    native = self.p.ParallelSpec(sp=sp)
                    for task in plan.tasks.values():
                        stage = resolver._resolve_request_stage(task)
                        candidates = {resolver._resolve_seq_len(task, plan)}
                        candidates.add(m['num_steps'] if stage == 'timestep_prepare' else m['voxels'] if stage == 'vae_decode'
                                       else m['text_seq_len'] if stage == 'text_encode' else 1 if stage == 'finalize' else m['latent_seq_len'])
                        actual = self.p.ParallelSpec() if task.kind in (self.p.TaskKind.TEXT_ENCODE, self.p.TaskKind.VAE_DECODE, self.p.TaskKind.FINALIZE) else native
                        for x in candidates:
                            self.costs.estimate(actual, x, stage)
        self.policy = make_policy(control, self.topology, self.costs, tuple(allowed_sp), policy)
        self.status = {rid: 'not_submitted' for rid in self.requests}
        self.active, self.launched, self.holds, self.decisions = {}, set(), {}, []
        self.now_ms, self.event_count = 0., 0

    def _time(self, now_ms):
        common.finite(now_ms, 'event time')
        if now_ms < self.now_ms:
            raise ValueError('event clock moved backwards')
        self.now_ms = float(now_ms)
        self.policy.now_ms = self.now_ms

    def _record(self, tasks):
        rows = []
        occupied = {rank for row in self.active.values() for rank in row['ranks']}
        held = {rank for ranks in self.holds.values() for rank in ranks}
        for task in tasks:
            group = self.topology.get_group(task.group_id)
            ranks = list(group.ranks)
            if (task.task_id in self.active or occupied.intersection(ranks) or held.intersection(ranks)
                    or any(row['request_id'] == task.request_id for row in self.active.values())):
                raise ValueError('native decision violated rank or per-request exclusivity')
            if not all(dependency in self.launched for dependency in task.dependencies):
                raise ValueError('decision precedes original DAG dependency publication')
            request = self.requests[task.request_id]
            previous = self.policy.request_last_group.get(task.request_id)
            row = dict(sequence=len(self.decisions), task_id=task.task_id, request_id=task.request_id,
                kind=task.kind.value, step_range=([task.step_range.start, task.step_range.end] if task.step_range else None),
                at_ms=self.now_ms, group_id=group.group_id, ranks=ranks, sp=group.parallel_spec.sp,
                source_request_sha256=request['source_request_sha256'], spec_sha256=request['spec_sha256'],
                model_manifest_sha256=request['model_manifest']['sha256'],
                original_arrival_ms=request['arrival_ms'], original_deadline_ms=request['deadline_ms'],
                remaining_budget_ms=max(0., request['deadline_ms'] - self.now_ms),
                previous_group_id=previous, group_changed=previous is not None and previous != group.group_id,
                migration_executed=False)
            self.active[task.task_id] = row; self.decisions.append(row); rows.append(copy.deepcopy(row))
            occupied.update(ranks)
        return rows

    def submit(self, request_id, now_ms):
        self._time(now_ms)
        if request_id not in self.requests or self.status[request_id] != 'not_submitted':
            raise ValueError('unknown or duplicate offered request')
        if now_ms < self.requests[request_id]['arrival_ms']:
            raise ValueError('request submitted before frozen arrival')
        self.status[request_id] = 'submitted'
        return self._record(self.policy.on_request_submitted(self.plans[request_id]))

    def _native_event(self, task, kind):
        self.event_count += 1
        return self.p.WorkerEvent(str(self.event_count), task.task_id, task.request_id, task.group_id,
            self.topology.get_group(task.group_id).ranks[0], kind, round(self.now_ms * 1_000_000))

    def event(self, task_id, kind, now_ms):
        self._time(now_ms)
        if task_id not in self.active or kind not in ('launch_end', 'exec_end', 'failed'):
            raise ValueError('event is not for an active dispatched task')
        task = self.tasks[task_id]; plan = self.plans[task.request_id]
        if kind == 'launch_end':
            if task_id in self.launched:
                raise ValueError('duplicate launch-end event')
            self.launched.add(task_id)
            out = self._record(self.policy.on_worker_event(self._native_event(task, self.p.WorkerEventKind.TASK_LAUNCH_END)))
            runnable = [t for t in plan.tasks.values() if t.status == self.p.TaskStatus.PENDING
                        and t.dependencies and all(d in self.launched for d in t.dependencies)]
            return out + self._record(self.policy.on_tasks_runnable(runnable))
        if kind == 'exec_end' and task_id not in self.launched:
            raise ValueError('completion without original launch-end dependency event')
        del self.active[task_id]
        task.status = self.p.TaskStatus.FAILED if kind == 'failed' else self.p.TaskStatus.FINISHED
        # Do not dispatch already-published successors of a failed task. This
        # fail-closed host boundary is explicit; it is not a native runtime fix.
        if kind == 'failed':
            self.policy.ready_by_request.pop(task.request_id, None)
        event_kind = self.p.WorkerEventKind.TASK_FAILED if kind == 'failed' else self.p.WorkerEventKind.TASK_EXEC_END
        out = self._record(self.policy.on_worker_event(self._native_event(task, event_kind)))
        if kind == 'failed' or all(self.tasks[tid].status == self.p.TaskStatus.FINISHED for tid in plan.terminal_task_ids):
            self.status[task.request_id] = 'failed_event_observed' if kind == 'failed' else 'terminal_event_observed'
            request_kind = self.p.WorkerEventKind.REQUEST_FAILED if kind == 'failed' else self.p.WorkerEventKind.REQUEST_FINISHED
            out += self._record(self.policy.on_worker_event(self._native_event(task, request_kind)))
        return out

    def hold(self, identity, ranks, begin, now_ms):
        self._time(now_ms)
        if not isinstance(identity, str) or not identity or type(begin) is not bool:
            raise ValueError('explicit migration-hold identity required')
        if begin:
            if identity in self.holds or not ranks or len(set(ranks)) != len(ranks) or any(type(r) is not int or r < 0 or r >= len(self.topology.workers) for r in ranks):
                raise ValueError('invalid or duplicate migration hold')
            self.holds[identity] = tuple(ranks)
            self.policy.acquire_migration_hold(tuple(ranks))
        else:
            if identity not in self.holds or tuple(ranks) != self.holds[identity]:
                raise ValueError('unknown or mismatched migration hold release')
            self.policy.release_migration_hold(self.holds.pop(identity))
        return self.poll(now_ms)

    def poll(self, now_ms):
        self._time(now_ms)
        return self._record(self.policy.on_tasks_runnable(()))

    def summary(self):
        return dict(schema=SCHEMA, scope='original GF-DiT policy decision reference; no worker/GFC/model execution',
            workload_sha256=self.workload['sha256'], policy=self.policy_name,
            cost_input_sha256=common.canonical(self.costs.document), cost_provenance=self.costs.provenance,
            offered_requests=len(self.requests), status_counts=dict(Counter(self.status.values())),
            requests=[dict(request_id=rid, status=self.status[rid], frozen_request=copy.deepcopy(request)) for rid, request in self.requests.items()],
            decisions=copy.deepcopy(self.decisions), active_task_ids=sorted(self.active),
            outstanding_migration_holds={k:list(v) for k,v in self.holds.items()}, observed_until_ms=self.now_ms,
            elapsed_times_are_supplied_events=True, model_execution_performed=False, native_global_scheduler_executed=False,
            native_gfc_executed=False, decoded_outputs_verified=0, qualification_eligible=False, allows=[],
            explicit_differences=['namespace/logging relocation verified by existing source port',
                'absolute client-offset metadata replaces original submit-time deadline reanchoring',
                'request context supplies model/shape-specific costs to original estimates',
                'host launch/end event adapter has no artifacts, worker or output fetch',
                'failed-task successors fail closed in host event adapter'])


def replay(driver, document, *, allow_fixture=False):
    if document.get('schema') != EVENT_SCHEMA or document.get('workload_sha256') != driver.workload['sha256']:
        raise ValueError('event ledger workload binding differs')
    provenance = evidence_bindings(document.get('provenance'), allow_fixture=allow_fixture)
    error = None
    try:
        for event in document['events']:
            action, now = event['action'], event['at_ms']
            if action == 'submit': driver.submit(event['request_id'], now)
            elif action == 'task': driver.event(event['task_id'], event['kind'], now)
            elif action == 'hold': driver.hold(event['id'], event['ranks'], event['begin'], now)
            elif action == 'poll': driver.poll(now)
            else: raise ValueError('unknown event action')
    except (ValueError, KeyError, TypeError) as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    return dict(**driver.summary(), event_provenance=provenance, event_input_sha256=common.canonical(document),
                replay_completed=error is None, error=error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'source-manifest', 'source-manifest-sha256', 'workload', 'workload-sha256',
                 'costs', 'costs-sha256', 'events', 'events-sha256', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--allowed-sp', type=int, nargs='+', required=True)
    parser.add_argument('--chunk-size', type=int, default=2)
    parser.add_argument('--policy', choices=POLICIES, default='edf_best_fit')
    parser.add_argument('--allow-fixture-inputs', action='store_true')
    args = parser.parse_args(argv)
    workload, wb = common.load_bound(args.workload, args.workload_sha256)
    costs, cb = common.load_bound(args.costs, args.costs_sha256)
    events, eb = common.load_bound(args.events, args.events_sha256)
    common.verify_export(workload)
    output = Path(args.output_dir).absolute()
    inputs = [Path(p).resolve() for p in (args.source_root, args.source_manifest, args.workload, args.costs, args.events)]
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):
        raise ValueError('output must be new and disjoint from bound inputs')
    derived = source.apply(args.source_root, args.source_manifest, args.source_manifest_sha256, output / 'control')
    control = source.load_control(derived['root'], derived['manifest_sha256'])
    driver = DecisionDriver(control, workload, costs, world_size=args.world_size, allowed_sp=tuple(args.allowed_sp),
        chunk_size=args.chunk_size, policy=args.policy, allow_fixture=args.allow_fixture_inputs)
    result = replay(driver, events, allow_fixture=args.allow_fixture_inputs)
    result.update(upstream_commit=source.upstream.COMMIT, upstream_manifest_sha256=args.source_manifest_sha256,
        derived_manifest_sha256=derived['manifest_sha256'], adapter_sha256=common.file_sha(Path(__file__).read_bytes()),
        helper_source_sha256={name:common.file_sha(Path(module.__file__).read_bytes()) for name,module in
            [('external_comparison',common),('external_gfdit_shared',source),('external_gfdit',source.upstream),
             ('external_sd35_round',source.round_adapter)]},
        source_bindings=dict(workload=wb, costs=cb, events=eb), algorithm_sources=derived['manifest']['imports_relocated'])
    with (output / 'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(replay_completed=result['replay_completed'], offered_requests=result['offered_requests'],
                         qualification_eligible=False)))
    return 0 if result['replay_completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
