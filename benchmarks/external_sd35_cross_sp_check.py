"""Three observed first steps: native SP1, shared-round SP1, shared-round SP2.

All use the retained request's original 28-step schedule and raw Euler solver.
Only the first step runs; no arrays or performance qualification are retained.
The SP2 local outputs must reproduce the immutable r5 first-step observation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import timedelta
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import external_tetriserve_sp2_diagnostic as base
import external_sd35_trace as trace

R5_SHA256 = 'a716aed06acdbc54b661d945bb5bd188ff3e38b8b6aa6ef31a4a9dd8a39425b1'
ROUTES = ('sp1_native', 'sp1_round', 'sp2_round')
INPUTS = ('hidden_states', 'encoder_hidden_states', 'pooled_projections', 'timestep')
BOUNDARIES = INPUTS + ('transformer_prediction', 'cfg_prediction', 'scheduler_sample',
                      'scheduler_timestep', 'scheduler_output')
SPATIAL = frozenset(('hidden_states', 'transformer_prediction', 'cfg_prediction',
                     'scheduler_sample', 'scheduler_output'))
STAGES = (('input', INPUTS), ('transformer_prediction', ('transformer_prediction',)),
          ('cfg_prediction', ('cfg_prediction',)),
          ('scheduler_input', ('scheduler_sample', 'scheduler_timestep')),
          ('scheduler_output', ('scheduler_output',)))
MAX_RECEIPT = 256 * 1024


@contextmanager
def override(obj, name, replacement):
    """Restore the exact instance override, including its original absence."""
    present, value = name in vars(obj), vars(obj).get(name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if present:
            setattr(obj, name, value)
        else:
            delattr(obj, name)


class FirstStepComplete(Exception):
    """Private control flow raised only after an observed native solver return."""


class FirstStepObserver:
    """Observe one original transformer and solver call without replacing tensors."""
    def __init__(self, pipe, scheduler, *, native, metadata, snapshot=None):
        self.pipe, self.scheduler, self.native = pipe, scheduler, native
        self.metadata = metadata
        self.snapshot = snapshot or (lambda value: value.detach().clone())
        self.tensors = {}
        self.counts = dict(transformer_enter=0, transformer_return=0, solver_enter=0,
                           solver_return=0, naive=0, sync=0, async_=0)
        self.stop = FirstStepComplete('observed the original native first solver return')
        self.stopped = False
        self.meta = None

    def save(self, key, tensor):
        if key in self.tensors:
            raise ValueError('repeated first-step boundary: ' + key)
        self.tensors[key] = self.snapshot(tensor)

    def run_native(self, call):
        if not self.native:
            raise ValueError('native stop catcher used for a round route')
        try:
            call()
        except FirstStepComplete as exc:
            if exc is not self.stop or self.counts['solver_return'] != 1:
                raise
            self.stopped = True
        else:
            raise ValueError('native invocation did not reach the intentional first-step stop')

    @contextmanager
    def observe(self):
        with ExitStack() as stack:
            for name, key in (('_naive_forward', 'naive'), ('_sync_pipeline', 'sync'),
                              ('_async_pipeline', 'async_')):
                original = getattr(self.pipe, name)
                def counted(*args, _original=original, _key=key, **kwargs):
                    self.counts[_key] += 1
                    return _original(*args, **kwargs)
                stack.enter_context(override(self.pipe, name, counted))
            forward = self.pipe.transformer.forward
            signature = inspect.signature(forward)
            def before(module, args, kwargs):
                self.counts['transformer_enter'] += 1
                values = signature.bind(*args, **kwargs).arguments
                for key in INPUTS:
                    self.save(key, values[key])
                scheduler = self.scheduler()
                self.save('timesteps', scheduler.timesteps)
                self.save('sigmas', scheduler.sigmas)
                self.meta = self.metadata(scheduler)
            def after(module, args, kwargs, output):
                self.counts['transformer_return'] += 1
                if not isinstance(output, tuple) or len(output) != 1:
                    raise ValueError('expected the original transformer tuple output')
                self.save('transformer_prediction', output[0])
                # A forward hook returning None leaves the exact output untouched.
            handle = self.pipe.transformer.register_forward_pre_hook(before, with_kwargs=True)
            stack.callback(handle.remove)
            handle = self.pipe.transformer.register_forward_hook(after, with_kwargs=True)
            stack.callback(handle.remove)
            target, method = (self.scheduler(), 'step') if self.native else (self.pipe, '_scheduler_step')
            original = getattr(target, method)
            signature_step = inspect.signature(original)
            def step(*args, **kwargs):
                self.counts['solver_enter'] += 1
                values = signature_step.bind(*args, **kwargs).arguments
                names = ('model_output', 'sample', 'timestep') if self.native else ('noise_pred', 'latents', 't')
                for key, name in zip(('cfg_prediction', 'scheduler_sample', 'scheduler_timestep'), names):
                    self.save(key, values[name])
                output = original(*args, **kwargs)
                self.counts['solver_return'] += 1
                if self.native and (not isinstance(output, tuple) or len(output) != 1):
                    raise ValueError('expected native raw scheduler tuple output')
                self.save('scheduler_output', output[0] if self.native else output)
                if self.native:
                    raise self.stop
                return output
            stack.enter_context(override(target, method, step))
            yield self
        expected = dict(transformer_enter=1, transformer_return=1, solver_enter=1,
                        solver_return=1, naive=int(self.native), sync=int(not self.native), async_=0)
        if self.counts != expected or (self.native and not self.stopped):
            raise ValueError('first-step execution coverage or route differs')
        if set(self.tensors) != set(BOUNDARIES) | {'timesteps', 'sigmas'}:
            raise ValueError('first-step observation is incomplete')


def validate_rows(ranges, height, local_height):
    if (type(height) is not int or type(local_height) is not int or height < 1
            or local_height < 1 or not isinstance(ranges, list) or not ranges):
        raise ValueError('invalid row metadata')
    previous, count = 0, 0
    for pair in ranges:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(x) is not int for x in pair)
                or not previous <= pair[0] < pair[1] <= height):
            raise ValueError('row ranges must be ordered disjoint intervals')
        previous = pair[1]
        count += pair[1] - pair[0]
    if count != local_height:
        raise ValueError('runtime rows do not match the actual tensor height')
    return ranges


def validate_partition(rows, height=128):
    if len(rows) != 2 or {r.get('rank') for r in rows} != {0, 1}:
        raise ValueError('two distinct rank row maps are required')
    spans = []
    for row in rows:
        spans.extend(validate_rows(row['ranges'], height, row['local_height']))
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            raise ValueError('SP2 row maps overlap or leave a gap')
        cursor = end
    if cursor != height:
        raise ValueError('SP2 row maps do not cover the full latent')


def slice_rows(tensor, ranges):
    import torch
    validate_rows(ranges, tensor.shape[2], sum(b-a for a,b in ranges))
    return torch.cat([tensor[:, :, a:b, :] for a,b in ranges], dim=2)


def first_differences(comparisons):
    if set(comparisons) != set(BOUNDARIES):
        raise ValueError('all first-step boundaries are required')
    result = dict(first_bitwise=None, first_tolerance=None)
    for stage, keys in STAGES:
        for field, flag in (('first_bitwise', 'bitwise_equal'), ('first_tolerance', 'passed')):
            if any(type(comparisons[key].get(flag)) is not bool for key in keys):
                raise ValueError('comparison flags must be booleans')
            if result[field] is None and any(not comparisons[key][flag] for key in keys):
                result[field] = stage
    return result


def bind_reference(summary, expected, paths, binding):
    if expected != R5_SHA256 or len(paths) != 2:
        raise ValueError('cross-SP observation requires the fixed r5 child and two traces')
    raw = trace.bounded_read(summary, trace.REFERENCE_MAX_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('r5 child SHA256 differs')
    child = json.loads(raw)
    if (child.get('binding') != binding or child.get('diagnostic_completed') is not True
            or child.get('observation_admissible') is not True
            or child.get('observer_reproduced') is not True
            or child.get('solver_counterfactual_enabled') is not True
            or child.get('qualification_eligible') is not False
            or not base.assess(child.get('ranks', []))):
        raise ValueError('r5 child is not the completed source-bound counterfactual observation')
    references = []
    for rank, path in enumerate(paths):
        row = next(r for r in child['ranks'] if r['rank'] == rank)
        receipt = row['trace']
        raw = trace.bounded_read(path, trace.MAX_BYTES)
        record = json.loads(raw)
        if (len(raw) != receipt['bytes'] or hashlib.sha256(raw).hexdigest() != receipt['sha256']
                or receipt['file'] != 'rank-'+str(rank)+'.trace.json'
                or record.get('rank') != rank or not trace.valid_trace(record)):
            raise ValueError('r5 trace binding or complete observation differs')
        first = record['trajectories']['round_sp2_full']
        references.append(dict(rank=rank, trace_sha256=receipt['sha256'],
            initial_five_tensors=row['initial_five_tensors'],
            inputs=first['scheduler'][0]['inputs'], cfg_prediction=first['backbone'][0]['output']['noise_pred'],
            scheduler_output=first['scheduler'][0]['output'], runtime=first['backbone'][0]['runtime']))
    return references


def processor_observer(transformer, stack):
    """Count attention calls without storing any block tensors or changing processors."""
    from xfuser.model_executor.layers import attention_processor as processors
    from xfuser.model_executor.models.transformers.transformer_sd3 import sd3_attn_modules
    rows = []
    names = {id(module): 'transformer_blocks.'+str(index)+'.'+name
             for index,block in enumerate(transformer.transformer_blocks)
             for name in ('attn','attn2') if (module:=getattr(block,name,None)) is not None}
    # Use the same inventory helper as the frozen backend, including attn2.
    for index, attention in enumerate(sd3_attn_modules(transformer)):
        processor = attention.processor
        row = dict(index=index, name=names[id(attention)], attention_type=type(attention).__module__+'.'+type(attention).__qualname__,
            processor_type=type(processor).__module__+'.'+type(processor).__qualname__,
            use_long_ctx_attn_kvcache=getattr(processor, 'use_long_ctx_attn_kvcache', None), calls=0)
        def count(*args, _row=row):
            _row['calls'] += 1
        handle = attention.register_forward_pre_hook(count)
        stack.callback(handle.remove)
        rows.append(row)
    if not rows:
        raise ValueError('actual attention inventory is empty')
    return dict(HAS_LONG_CTX_ATTN=processors.HAS_LONG_CTX_ATTN, modules=rows)


def valid_runtime(runtime, route):
    if route not in ROUTES or not isinstance(runtime,dict):
        return False
    expected = dict(pp_degree=1,cfg_degree=1,dp_degree=1,sp_degree=2 if route=='sp2_round' else 1)
    return (runtime.get('parallel') == expected
        and all(type(v) is int for v in runtime['parallel'].values())
        and runtime.get('patch_mode') is False
        and runtime.get('split_text_embed_in_sp') is (route=='sp1_round')
        and runtime.get('split_latents_by_rows') is (route!='sp1_native')
        and runtime.get('text_embed_sp_pad') == 0)


def describe_observation(observer, initial, route, processors):
    import torch
    tensors = observer.tensors
    descriptors = {name: trace.tensor_description(value) for name,value in tensors.items()}
    runtime = observer.meta['runtime']
    ranges = runtime['pp_patches_start_end_idx_global']
    validate_rows(ranges, 128, tensors['scheduler_sample'].shape[2])
    if route != 'sp2_round' and ranges != [[0,128]]:
        raise ValueError('SP1 route must cover the complete original latent')
    expected = dict(hidden_states=torch.cat([slice_rows(initial.latents,ranges)]*2),
        encoder_hidden_states=initial.prompt_embeds, pooled_projections=initial.pooled_prompt_embeds,
        timestep=initial.timesteps[0].expand(2), scheduler_sample=slice_rows(initial.latents,ranges),
        scheduler_timestep=initial.timesteps[0],
        timesteps=initial.timesteps if route=='sp1_native' else initial.timesteps[:1],
        sigmas=initial.sigmas if route=='sp1_native' else initial.sigmas[:2])
    matches = {key: descriptors[key] == trace.tensor_description(value) for key,value in expected.items()}
    if not all(matches.values()):
        raise ValueError('actual first-step inputs differ from the shared original five tensors: '+str(matches))
    if not valid_runtime(runtime, route):
        raise ValueError('unexpected parallel/patch/text runtime')
    if any(row['calls'] != 1 for row in processors['modules']):
        raise ValueError('attention inventory did not execute exactly once at the first step')
    return dict(route=route, logical_step=0, original_schedule_steps=28, counts=observer.counts,
        intentional_native_stop=observer.stopped, actual_inputs_match_initial=matches,
        tensors=descriptors, metadata=observer.meta, processors=processors,
        actual_row_ranges=ranges, local_height=tensors['scheduler_sample'].shape[2],
        boundary_owner='actual local computation before final pipeline output collection',
        hooks_restored=True, arrays_retained=False)


def observe_route(pipeline, initial, request, route):
    import torch
    from external_sd35_round import RoundState
    inspector = trace.TraceObserver(pipeline.pipe, lambda:pipeline.runtime.state, 0)
    raw = vars(pipeline.raw.scheduler).get('module', pipeline.raw.scheduler)
    # Capture native source before installing the one-call step observer.
    native_solver = inspector._scheduler(raw) if route=='sp1_native' else None
    def metadata(scheduler):
        state = native_solver if route=='sp1_native' else inspector._scheduler(scheduler)
        item = inspector.record['schedulers'][state['identity']]
        if item['bound_step']['file_sha256'] != trace.RAW_SOLVER_SHA256 or 'inner' in item:
            raise ValueError('all cross-SP routes require the fixed raw diffusers step')
        return dict(runtime=inspector._runtime(), scheduler=item,
            step_index=getattr(scheduler,'step_index',None), begin_index=getattr(scheduler,'begin_index',None))
    getter = (lambda:raw) if route=='sp1_native' else (lambda:pipeline.pipe.scheduler)
    observer = FirstStepObserver(pipeline.pipe, getter, native=route=='sp1_native', metadata=metadata)
    with ExitStack() as stack, torch.inference_mode():
        processors = processor_observer(pipeline.pipe.transformer, stack)
        with observer.observe():
            if route=='sp1_native':
                observer.run_native(lambda:pipeline.pipe(prompt=request.prompt,negative_prompt=request.negative_prompt,
                    height=request.height,width=request.width,num_inference_steps=request.steps,
                    guidance_scale=request.guidance_scale,max_sequence_length=request.max_sequence_length,
                    generator=torch.Generator(device='cpu').manual_seed(request.seed),output_type='latent'))
            else:
                state = RoundState(request, *(t.detach().clone() for t in initial.transferred_tensors()))
                pipeline.advance(state,1)
    receipt = describe_observation(observer, initial, route, processors)
    return observer.tensors, receipt


def compare_boundaries(left, right, ranges=None):
    from external_sd35_round_check import compare_tensors
    rows = {}
    for key in BOUNDARIES:
        reference = slice_rows(left[key], ranges) if ranges is not None and key in SPATIAL else left[key]
        rows[key] = dict(compare_tensors(reference,right[key]), reference=trace.tensor_description(reference),
                         actual=trace.tensor_description(right[key]))
    return dict(boundaries=rows, **first_differences(rows))


def broadcast_boundaries(values, rank, control, group):
    """Post-model diagnostic transfer over existing groups, in a fixed key order."""
    import torch
    import torch.distributed as dist
    payload = [{key:dict(shape=list(values[key].shape),dtype=str(values[key].dtype)) for key in BOUNDARIES}
               if rank==0 else None]
    dist.broadcast_object_list(payload,src=0,group=control)
    descriptions = payload[0]
    if set(descriptions) != set(BOUNDARIES):
        raise ValueError('broadcast boundary inventory differs')
    received = {}
    for key in BOUNDARIES:
        row = descriptions[key]
        if (row['dtype'] not in ('torch.bfloat16','torch.float32')
                or any(type(x) is not int or x<1 for x in row['shape'])
                or math.prod(row['shape']) > 4*1024*1024):
            raise ValueError('unbounded diagnostic tensor transfer')
        tensor = values[key].contiguous() if rank==0 else torch.empty(row['shape'],
            dtype=getattr(torch,row['dtype'].split('.')[-1]),device='cuda:'+str(rank))
        dist.broadcast(tensor,src=0,group=group)
        received[key] = tensor
    return received


def reproduced(receipt, reference):
    values = receipt['tensors']
    return (values['cfg_prediction']==reference['cfg_prediction']
        and values['scheduler_output']==reference['scheduler_output']
        and values['scheduler_sample']==reference['inputs']['latents']
        and values['scheduler_timestep']==reference['inputs']['t']
        and receipt['metadata']['runtime']==reference['runtime'])


def worker(rank, args, binding, references):
    result = dict(rank=rank,status='failed',distributed_closed=False,qualification_eligible=False)
    pipeline = progress = None
    try:
        progress = base.RankProgress(args.output_dir,rank,binding)
        progress.update('source_binding_enter')
        base.migration.verify_derived(root=binding['derived']['root'],expected=binding['derived']['manifest_sha256'])
        if base.sha(binding['config_path']) != binding['config_sha256']:
            raise ValueError('prior config changed before spawn')
        base.numerics.service.bind_derived_adapters(binding['derived'])
        sys.path.insert(0,binding['config']['shared_python_root'])
        os.environ.update(DITSERVE_TETRI_PORT_CONFIG=binding['config_path'],
                          DITSERVE_TETRI_PORT_CONFIG_SHA256=binding['config_sha256'])
        import torch
        import torch.distributed as dist
        from external_sd35_round import ImageRequest
        from external_tetriserve_bridge import TetriSharedSD35Pipeline
        from tetriserve.utils import set_device
        from tetriserve.distributed.globals import PROCESS_GROUP, set_seq_parallel_pg
        if torch.cuda.device_count()!=2:
            raise ValueError('exactly two GPUs are required')
        torch.cuda.set_device(rank); set_device(gpu=True)
        dist.init_process_group('nccl',init_method='file://'+str(args.output_dir/'rendezvous'),rank=rank,
                                world_size=2,timeout=timedelta(seconds=180))
        PROCESS_GROUP.generate_all_process_groups(2,skip_latent_transfer=False)
        control = PROCESS_GROUP.get_process_group([0,1],device='cpu')
        group = PROCESS_GROUP.get_process_group([0,1],device='cuda')
        progress.update('model_load_enter')
        pipeline = TetriSharedSD35Pipeline.from_pretrained('stabilityai/stable-diffusion-3.5-medium',
            base.migration.upstream_engine_config('stabilityai/stable-diffusion-3.5-medium'),torch.bfloat16)
        result['provenance'] = pipeline.provenance
        result['gpu_uuid'] = str(torch.cuda.get_device_properties(rank).uuid)
        # Match r5 construction/prologue, then switch only active existing groups.
        set_seq_parallel_pg(2,1,rank,[0,1],2)
        pipeline.runtime.pool.activate((0,1),pipeline.runtime.state,ulysses_degree=2,ring_degree=1)
        pipeline.runtime.bind_global()
        request = ImageRequest(**binding['request'])
        initial = pipeline.prepare(request)
        initial_summary = {name:trace.tensor_description(t) for name,t in zip(base.migration.NAMES,initial.transferred_tensors())}
        if initial_summary != references[rank]['initial_five_tensors']:
            raise ValueError('original five tensors do not reproduce r5')
        agreed = [None,None]
        dist.all_gather_object(agreed,initial_summary,group=control)
        if agreed[0]!=agreed[1]:
            raise ValueError('initial tensors differ across ranks')
        result['initial_five_tensors'] = initial_summary
        result['routes'] = {}
        snapshots = {}
        pipeline.runtime.pool.activate((rank,),pipeline.runtime.state,ulysses_degree=1,ring_degree=1)
        pipeline.runtime.bind_global()
        progress.update('sp1_native_enter' if rank==0 else 'waiting_for_rank0_sp1')
        if rank==0:
            for route in ROUTES[:2]:
                progress.update(route+'_enter')
                snapshots[route],result['routes'][route] = observe_route(pipeline,initial,request,route)
                progress.update(route+'_return')
            result['sp1_native_vs_sp1_round'] = compare_boundaries(snapshots[ROUTES[0]],snapshots[ROUTES[1]])
        dist.barrier(group=control)
        pipeline.runtime.pool.activate((0,1),pipeline.runtime.state,ulysses_degree=2,ring_degree=1)
        pipeline.runtime.bind_global()
        progress.update('sp2_round_enter')
        snapshots['sp2_round'],result['routes']['sp2_round'] = observe_route(pipeline,initial,request,'sp2_round')
        progress.update('sp2_round_return')
        receipt = result['routes']['sp2_round']
        result['sp2_step0_reproduced'] = reproduced(receipt,references[rank])
        maps = [None,None]
        dist.all_gather_object(maps,dict(rank=rank,ranges=receipt['actual_row_ranges'],local_height=receipt['local_height']),group=control)
        validate_partition(maps)
        result['row_partition'] = maps
        progress.update('boundary_comparison_enter')
        for route in ROUTES[:2]:
            reference = broadcast_boundaries(snapshots.get(route),rank,control,group)
            result[route+'_vs_sp2_round'] = compare_boundaries(reference,snapshots['sp2_round'],receipt['actual_row_ranges'])
        result['status'] = 'completed'
        progress.update('boundary_comparison_return')
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc()[-10000:])
        if progress is not None:
            try: progress.fail(exc)
            except BaseException as error: result['progress_error']=repr(error)
    finally:
        try:
            if pipeline is not None: pipeline.runtime.close()
            import torch.distributed as dist
            if dist.is_initialized(): dist.destroy_process_group()
            result['distributed_closed']=not dist.is_initialized()
        except BaseException as exc:
            result.update(status='failed',cleanup_error=repr(exc))
        if len(trace.encoded(result)) > MAX_RECEIPT:
            raise ValueError('rank receipt exceeds bounded metadata budget')
        base.probe.storage()
        base.write_json(args.output_dir/('rank-'+str(rank)+'.json'),result)
    if result['status']!='completed':
        raise RuntimeError('cross-SP first-step worker failed; see rank receipt')


def valid_pair(pair):
    if not isinstance(pair,dict): return False
    rows = pair.get('boundaries',{})
    try:
        if {k:pair.get(k) for k in ('first_bitwise','first_tolerance')} != first_differences(rows): return False
        for row in rows.values():
            a,b = row['reference'],row['actual']
            if (not trace.valid_tensor(a) or not trace.valid_tensor(b) or a['shape']!=b['shape']
                    or a['dtype']!=b['dtype'] or type(row['elements']) is not int or row['elements']!=math.prod(a['shape'])
                    or row['atol']!=1e-5 or row['rtol']!=1e-5
                    or row['bitwise_equal']!=(a['sha256']==b['sha256'])): return False
            for key in ('max_abs_error','rmse','relative_l2_error','cosine_similarity'):
                if type(row.get(key)) not in (int,float) or not math.isfinite(row[key]): return False
            if not 0<=row['rmse']<=row['max_abs_error'] or row['relative_l2_error']<0: return False
            if not -1.000000000001<=row['cosine_similarity']<=1.000000000001: return False
            if row['passed'] and row['max_abs_error']>1e-5+1e-5*max(abs(b['minimum']),abs(b['maximum'])): return False
            if row['bitwise_equal'] and (not row['passed'] or row['max_abs_error']!=0): return False
        return True
    except (KeyError,TypeError,ValueError): return False


def assess(rows, references):
    try:
        if len(rows)!=2 or {r['rank'] for r in rows}!={0,1}: return False
        for row in rows:
            rank=row['rank']; routes=row['routes']
            if (row['status']!='completed' or row['distributed_closed'] is not True
                    or row.get('qualification_eligible') is not False
                    or row.get('sp2_step0_reproduced') is not True
                    or row['initial_five_tensors']!=references[rank]['initial_five_tensors']
                    or not reproduced(routes['sp2_round'],references[rank])
                    or set(routes)!=(set(ROUTES) if rank==0 else {'sp2_round'})): return False
            validate_partition(row['row_partition'])
            for name,route in routes.items():
                native=name=='sp1_native'
                expected=dict(transformer_enter=1,transformer_return=1,solver_enter=1,solver_return=1,
                              naive=int(native),sync=int(not native),async_=0)
                if (route['counts']!=expected or any(type(v) is not int for v in route['counts'].values())
                        or route['route']!=name or route['intentional_native_stop'] is not native
                        or route['hooks_restored'] is not True or route['arrays_retained'] is not False
                        or route['logical_step']!=0 or route['original_schedule_steps']!=28
                        or set(route['tensors'])!=set(BOUNDARIES)|{'timesteps','sigmas'}
                        or not all(trace.valid_tensor(v) for v in route['tensors'].values())
                        or set(route['actual_inputs_match_initial'])!=set(INPUTS)|{'scheduler_sample','scheduler_timestep','timesteps','sigmas'}
                        or not all(v is True for v in route['actual_inputs_match_initial'].values())
                        or not valid_runtime(route['metadata']['runtime'],name)
                        or route['metadata']['scheduler']['type']!=base.common_solver.RAW_TYPE
                        or route['metadata']['scheduler']['bound_step']['file_sha256']!=trace.RAW_SOLVER_SHA256
                        or type(route['processors']['HAS_LONG_CTX_ATTN']) is not bool
                        or not route['processors']['modules']
                        or any(type(x['calls']) is not int or x['calls']!=1
                               or not isinstance(x['name'],str) or not x['name'].endswith(('.attn','.attn2'))
                               or not isinstance(x['processor_type'],str)
                               for x in route['processors']['modules'])): return False
                validate_rows(route['actual_row_ranges'],128,route['local_height'])
                if route['actual_row_ranges']!=route['metadata']['runtime']['pp_patches_start_end_idx_global']: return False
                if name!='sp2_round' and route['actual_row_ranges']!=[[0,128]]: return False
                if route['tensors']['timesteps']['shape']!=([28] if native else [1]): return False
                if route['tensors']['sigmas']['shape']!=([29] if native else [2]): return False
            for pair in ('sp1_native_vs_sp2_round','sp1_round_vs_sp2_round'):
                if not valid_pair(row[pair]): return False
                if any(metric['actual']!=routes['sp2_round']['tensors'][key]
                       for key,metric in row[pair]['boundaries'].items()): return False
            if rank==0:
                if not valid_pair(row['sp1_native_vs_sp1_round']): return False
                if any(metric['reference']!=routes['sp1_native']['tensors'][key]
                       or metric['actual']!=routes['sp1_round']['tensors'][key]
                       for key,metric in row['sp1_native_vs_sp1_round']['boundaries'].items()): return False
        return True
    except (KeyError,TypeError,ValueError): return False


def execute(args,binding,references):
    import torch.multiprocessing as mp
    children=None; error=None; start=time.monotonic()
    try:
        children=mp.spawn(worker,args=(args,binding,references),nprocs=2,join=False)
        while not children.join(timeout=1):
            base.probe.storage()
            if time.monotonic()-start>args.timeout_seconds: raise TimeoutError('cross-SP diagnostic timeout')
    except BaseException as exc: error=repr(exc)
    finally:
        if children:
            for child in children.processes:
                if child.is_alive(): child.terminate()
            for child in children.processes:
                child.join(timeout=5)
                if child.is_alive(): child.kill(); child.join(timeout=5)
    return base.read_rank_results(args.output_dir),error


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','run'])
    parser.add_argument('--prior-result',type=Path,required=True)
    parser.add_argument('--prior-summary-sha256',required=True)
    parser.add_argument('--prior-numerics-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    parser.add_argument('--sp2-reference-summary',type=Path,required=True)
    parser.add_argument('--sp2-reference-sha256',default=R5_SHA256)
    parser.add_argument('--sp2-reference-traces',type=Path,nargs=2,required=True)
    args=parser.parse_args(argv)
    args.output_dir=args.output_dir.absolute(); args.prior_result=args.prior_result.absolute()
    result=dict(schema='ditserve.sd35-cross-sp-first-step.v1',status='failed',diagnostic_completed=False,
        qualification_eligible=False,performance_measurements=False,arrays_retained=False,
        original_schedule_steps=28,executed_steps_per_route=1,routes=list(ROUTES),
        scope='first differing observed boundary, not an attention root cause or 28-step correctness result',
        tolerance=dict(atol=1e-5,rtol=1e-5),hashing_may_synchronize_cuda=True,adapter_sha256=base.sha(__file__))
    created=False
    try:
        if not 120<=args.timeout_seconds<=3600: raise ValueError('invalid diagnostic timeout')
        binding=base.bind_prior(args.prior_result,args.prior_summary_sha256,args.prior_numerics_sha256)
        references=bind_reference(args.sp2_reference_summary,args.sp2_reference_sha256,args.sp2_reference_traces,binding)
        result.update(binding=binding,sp2_reference_sha256=args.sp2_reference_sha256,sp2_references=references)
        if args.command=='plan':
            result['status']='planned'; print(json.dumps(result,sort_keys=True,allow_nan=False)); return 0
        if (not args.output_dir.is_relative_to('/shared/output/DITSERVE')
                or args.output_dir.resolve()!=args.output_dir
                or args.output_dir.is_relative_to(args.prior_result) or args.prior_result.is_relative_to(args.output_dir)):
            raise ValueError('output must be new, unlinked, owned and disjoint from protected prior evidence')
        base.probe.storage(); args.output_dir.mkdir(parents=True,exist_ok=False); created=True
        rows,error=execute(args,binding,references)
        base.retain_execution(result,rows,error,args.output_dir)
        complete=error is None and assess(rows,references)
        result.update(diagnostic_completed=complete,status='diagnostic_completed' if complete else 'failed')
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc()[-10000:])
    if created:
        if len(trace.encoded(result))>3*MAX_RECEIPT: raise ValueError('summary exceeds bounded metadata budget')
        base.probe.storage(); base.write_json(args.output_dir/'summary.json',result)
    print(json.dumps(dict(status=result['status'],diagnostic_completed=result['diagnostic_completed'],qualification_eligible=False)),flush=True)
    return 0 if result['diagnostic_completed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
