"""Observe the original Wan22 first step, preserving the formal 50-step schedule.

The original runner owns initialization, CFG, transformer, scheduler and SP.
Stop before the next transformer invocation, after the first scheduler returned.
No completed request, decoded output, throughput or quality qualification follows.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, redirect_stdout
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback

import external_xdit_native as native
import external_xdit_distributed as distributed

BUDGET = 256 * 1024**2
ATOL = RTOL = 1e-5
SCHEMA = 'ditserve.wan-first-step.v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def source_of(method):
    function = getattr(method, '__func__', method)
    path = Path(inspect.getsourcefile(function)).resolve(strict=True)
    return dict(qualified_name=function.__module__+'.'+function.__qualname__,
                file=str(path), sha256=sha(path.read_bytes()))


def method_sources(method):
    return dict(bound=source_of(method), unwrapped=source_of(inspect.unwrap(method)))


def safe_artifact(root, name):
    if not isinstance(name, str) or not re.fullmatch(r'boundary-[0-9]{3}\.safetensors', name):
        raise ValueError('invalid bounded artifact filename')
    path = Path(root)/name
    if path.resolve() != path or not path.is_relative_to(Path('/shared/output/DITSERVE')):
        raise ValueError('artifact must remain in the project output without symlinks')
    return path


def budget_check(used, additional):
    if type(used) is not int or type(additional) is not int or min(used, additional) < 0 or used+additional > BUDGET:
        raise ValueError('first-step protected artifact budget exceeded')


def compare_arrays(actual, reference):
    """All elements, reference-relative fixed tolerances, bounded fp64 temporaries."""
    import numpy as np
    if actual.shape != reference.shape or actual.size == 0:
        raise ValueError('boundary shape differs or is empty')
    a, b = actual.reshape(-1), reference.reshape(-1)
    total, failed, absolute, squares, reference_squares, maximum = 0, 0, 0., 0., 0., 0.
    for start in range(0, a.size, 262144):
        x, y = a[start:start+262144].astype(np.float64), b[start:start+262144].astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('nonfinite first-step boundary')
        error = np.abs(x-y)
        total += int(x.size); failed += int(np.count_nonzero(error > ATOL+RTOL*np.abs(y)))
        absolute += float(error.sum()); squares += float(np.square(error).sum())
        reference_squares += float(np.square(y).sum()); maximum = max(maximum, float(error.max()))
    return dict(total_elements=total, failed_elements=failed, max_abs=maximum,
        mean_abs=absolute/total, rmse=math.sqrt(squares/total),
        relative_l2=math.sqrt(squares/reference_squares) if reference_squares else (0. if squares == 0 else None),
        reference_l2=math.sqrt(reference_squares), zero_reference_norm=reference_squares == 0,
        atol=ATOL, rtol=RTOL, within_tolerance=failed == 0)


def tensor_descriptor(tensor):
    import torch
    value = tensor.detach().to('cpu').contiguous()
    if value.numel() == 0 or not torch.isfinite(value).all().item():
        raise ValueError('empty or nonfinite observed tensor')
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    meta = dict(shape=list(value.shape), dtype=str(value.dtype))
    return value, dict(**meta, bytes=len(raw), sha256=sha(canonical(meta)+raw), finite=True)


def read_bound(path, expected, maximum=1024**2):
    path = Path(path)
    if path.resolve() != path or not path.is_file():
        raise ValueError('reference path is not a regular resolved file')
    before = path.stat()
    if not 0 < before.st_size <= maximum: raise ValueError('reference exceeds predeclared read budget')
    raw = path.read_bytes(); after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or sha(raw) != expected:
        raise ValueError('reference file changed or hash differs')
    return raw


def reference_binding(result):
    identity = {k:v for k,v in result['model_identity'].items() if k != 'snapshot_fingerprint_sha256'}
    return dict(request=result['request'], source=result['source'], model_identity=identity,
                protocol_sha256=result['protocol_sha256'], adapter=result['adapter'])


def load_reference(directory, expected, binding):
    root = Path(directory)
    if root.resolve() != root or not root.is_relative_to(Path('/shared/output/DITSERVE')):
        raise ValueError('reference root must remain in project output')
    summary = native.source.read_json(read_bound(root/'summary.json', expected))
    if (summary.get('schema') != SCHEMA or summary.get('status') != 'first_step_observed'
            or summary.get('world_size') != 1 or summary.get('completed_requests') != 0
            or summary.get('qualification_eligible') is not False
            or summary.get('binding') != binding or len(summary.get('ranks', [])) != 1):
        raise ValueError('reference is not the matching bound SP1 first-step oracle')
    row = summary['ranks'][0]
    validate_rank(row)
    protection = native.source.read_json(read_bound(root/'protection.json', summary['protection_sha256'], 65536))
    if protection != dict(schema='ditserve.wan-first-step-protection.v1', dependency='SP1-SP4 first-step numerical comparison',
            protected=True, cleanup_allowed=False, automatic_cleanup=False, budget_bytes=BUDGET, binding=binding):
        raise ValueError('reference dependency protection differs')
    inventory = row['boundaries']; total = 0
    if not 1 <= len(inventory) <= 128: raise ValueError('unbounded reference inventory')
    for item in inventory:
        declared = item.get('artifact', {}).get('bytes')
        if type(declared) is not int or declared <= 0: raise ValueError('invalid artifact byte budget')
        budget_check(total, declared); total += declared
    budget_check(total, len(canonical(protection))+len(canonical(summary))+len(canonical(row))+4096)
    total = 0
    for item in inventory:
        artifact = item.get('artifact', {}); path = safe_artifact(root, artifact.get('file'))
        raw = read_bound(path, artifact['sha256'], artifact['bytes'])
        if len(raw) != artifact['bytes']: raise ValueError('reference artifact size differs')
        total += len(raw)
    budget_check(0, total+len(canonical(protection))+len(canonical(summary)))
    if total != row['artifact_bytes']: raise ValueError('reference artifact inventory incomplete')
    return dict(root=root, summary=summary, summary_sha256=expected,
                boundaries={r['name']:r for r in inventory})


class Recorder:
    def __init__(self, output, rank, reference=None):
        self.output, self.rank, self.reference = Path(output), rank, reference
        self.rows = []; self.bytes = 0
    def capture(self, name, value):
        import torch
        if isinstance(value, torch.Tensor):
            tensor, descriptor = tensor_descriptor(value)
            if any(r['name'] == name for r in self.rows): raise ValueError('duplicate boundary')
            row = dict(name=name, tensor=descriptor)
            if self.rank == 0 and self.reference is None:
                from safetensors.torch import save
                # Header reserve is checked before serialization and writing.
                budget_check(self.bytes, descriptor['bytes']+4096)
                raw = save({'tensor':tensor}); budget_check(self.bytes, len(raw)+65536)
                path = safe_artifact(self.output, f'boundary-{len(self.rows):03d}.safetensors')
                native.check_storage()
                with path.open('xb') as stream: stream.write(raw)
                row['artifact'] = dict(file=path.name, bytes=len(raw), sha256=sha(raw))
                self.bytes += len(raw)
            elif self.rank == 0:
                from safetensors.torch import load
                original = self.reference['boundaries'].get(name)
                if original is None: raise ValueError('unexpected boundary absent in oracle')
                artifact = original['artifact']
                raw = read_bound(safe_artifact(self.reference['root'], artifact['file']), artifact['sha256'], artifact['bytes'])
                tensors = load(raw)
                if set(tensors) != {'tensor'}: raise ValueError('unexpected safetensor keys')
                oracle, described = tensor_descriptor(tensors['tensor'])
                if described != original['tensor'] or described['dtype'] != descriptor['dtype']:
                    raise ValueError('oracle tensor or dtype differs from protected receipt')
                row['comparison'] = compare_arrays(tensor.double().numpy(), oracle.double().numpy())
                row['comparison']['bitwise_equal'] = descriptor == described
            self.rows.append(row)
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value): self.capture(name+'.'+str(i), item)
        elif isinstance(value, dict):
            for key in sorted(value): self.capture(name+'.'+key, value[key])
        elif value is not None and hasattr(value, 'sample'):
            self.capture(name+'.sample', value.sample)


class PrefixComplete(Exception):
    """Expected stop before transformer invocation for the second logical step."""


@contextmanager
def observe(pipe, model, recorder):
    """Method wrappers preserve original call/return identity and instance state."""
    counts = dict(encode_prompt=0, prepare_latents=0, generator=0, transformer=0, scheduler=0)
    row = dict(counts=counts, stopped_before_step1=False, restored=False, methods={}, generator=None)
    patches = []
    def patch(obj, name, kind):
        present, saved = name in vars(obj), vars(obj).get(name)
        original = getattr(obj, name); signature = inspect.signature(original)
        row['methods'][kind] = method_sources(original)
        def wrapped(*args, **kwargs):
            if kind == 'transformer' and counts['scheduler'] == 1:
                row['stopped_before_step1'] = True
                raise PrefixComplete()
            call = counts[kind]
            if call >= (2 if kind == 'transformer' else 1):
                raise ValueError('unexpected additional first-step call: '+kind)
            bound = signature.bind(*args, **kwargs)
            if kind in ('transformer', 'scheduler'):
                recorder.capture(kind+'.'+str(call)+'.in', dict(bound.arguments))
            if kind == 'scheduler':
                scheduler = pipe.scheduler
                recorder.capture('scheduler.0.schedule.timesteps', scheduler.timesteps)
                recorder.capture('scheduler.0.schedule.sigmas', scheduler.sigmas)
                if len(scheduler.timesteps) != 50 or len(scheduler.sigmas) != 51:
                    raise ValueError('original formal 50-step schedule changed')
                row['scheduler_before'] = dict(config=dict(scheduler.config),
                    step_index=getattr(scheduler, 'step_index', None),
                    begin_index=getattr(scheduler, 'begin_index', None),
                    type=type(scheduler).__module__+'.'+type(scheduler).__qualname__)
            output = original(*args, **kwargs)
            counts[kind] += 1
            if kind == 'generator':
                row['generator'] = dict(device=str(output.device), initial_seed=output.initial_seed())
                recorder.capture('generator.state', output.get_state())
            else:
                recorder.capture(kind+'.'+str(call)+'.out', output)
            return output
        setattr(obj, name, wrapped); patches.append((obj, name, present, saved, wrapped))
    try:
        patch(pipe, 'encode_prompt', 'encode_prompt'); patch(pipe, 'prepare_latents', 'prepare_latents')
        patch(model, '_make_generator', 'generator'); patch(pipe.transformer, 'forward', 'transformer')
        patch(pipe.scheduler, 'step', 'scheduler')
        yield row
    finally:
        clean = all(vars(obj).get(name) is wrapper for obj,name,_,_,wrapper in patches)
        for obj,name,present,saved,_ in reversed(patches):
            if present: setattr(obj,name,saved)
            else: delattr(obj,name)
        row['restored'] = clean and all((name in vars(obj)) == present and (not present or vars(obj)[name] is saved) for obj,name,present,saved,_ in patches)


def validate_rank(row):
    observation = row['observation']
    if (row.get('status') != 'first_step_observed' or row.get('cleanup_completed') is not True
            or observation.get('restored') is not True or observation.get('stopped_before_step1') is not True
            or observation['counts'] != dict(encode_prompt=1, prepare_latents=1, generator=1, transformer=2, scheduler=1)
            or row.get('completed_requests') != 0 or row.get('qualification_eligible') is not False):
        raise ValueError('first-step execution/cleanup counts incomplete')
    boundaries = row['boundaries']; names = [x['name'] for x in boundaries]
    if len(names) != len(set(names)) or not names: raise ValueError('duplicate or missing boundary inventory')
    required = {'generator.state', 'prepare_latents.0.out', 'encode_prompt.0.out.0', 'encode_prompt.0.out.1',
        'scheduler.0.in.model_output', 'scheduler.0.in.sample', 'scheduler.0.in.timestep',
        'scheduler.0.out.0', 'scheduler.0.schedule.timesteps', 'scheduler.0.schedule.sigmas'}
    for branch in (0,1):
        required.update({f'transformer.{branch}.in.hidden_states',f'transformer.{branch}.in.encoder_hidden_states',
            f'transformer.{branch}.in.timestep',f'transformer.{branch}.out.0'})
    if not required <= set(names): raise ValueError('missing required true model boundary')
    for item in boundaries:
        desc=item['tensor']
        if desc.get('finite') is not True or not re.fullmatch('[a-f0-9]{64}',desc.get('sha256','')) or desc['bytes'] <= 0:
            raise ValueError('invalid tensor descriptor')


def aggregate(rows, binding, world):
    if len(rows) != world or {r['rank'] for r in rows} != set(range(world)):
        raise ValueError('missing or duplicate final ranks')
    for row in rows:
        validate_rank(row)
        if row['binding'] != binding or row['world_size'] != world: raise ValueError('rank source or world differs')
    uuids = [distributed.normalize_uuid(r['runtime']['gpu_uuid']) for r in rows]
    if len(set(uuids)) != world: raise ValueError('actual GPU UUIDs are not distinct')
    first = sorted(rows,key=lambda r:r['rank'])[0]
    inventory = first.get('loaded_source_sha256', {})
    required_sources = {'xfuser/runner.py', 'xfuser/model_executor/models/runner_models/wan.py',
        'xfuser/model_executor/models/transformers/transformer_wan.py'}
    if not required_sources <= set(inventory) or any(not re.fullmatch('[a-f0-9]{64}',v) for v in inventory.values()):
        raise ValueError('actual loaded-source inventory missing or invalid')
    for row in rows:
        if row.get('loaded_source_sha256') != inventory or row['runtime'].get('sp') != world or row['runtime'].get('dp') != 1:
            raise ValueError('actual rank source inventory or topology differs')
    descriptors = lambda row:[dict(name=x['name'],tensor=x['tensor']) for x in row['boundaries']]
    same = all(descriptors(row)==descriptors(first) for row in rows)
    comparisons = [r['comparison'] for r in first['boundaries'] if 'comparison' in r]
    if world == 4 and len(comparisons) != len(first['boundaries']): raise ValueError('not all global boundaries compared')
    first_bitwise = next((r['name'] for r in first['boundaries'] if 'comparison' in r and not r['comparison']['bitwise_equal']), None)
    first_tolerance = next((r['name'] for r in first['boundaries'] if 'comparison' in r and not r['comparison']['within_tolerance']), None)
    return dict(schema=SCHEMA,status='first_step_observed',world_size=world,binding=binding,
        first_bitwise_difference=first_bitwise, first_tolerance_difference=first_tolerance,
        ranks=sorted(rows,key=lambda r:r['rank']), cross_rank_full_boundary_hashes_equal=same,
        all_boundaries_within_tolerance=all(r['within_tolerance'] for r in comparisons) if comparisons else None,
        completed_requests=0, requested_requests=1, qualification_eligible=False, performance_eligible=False,
        serving_executed=False, quality_evaluated=False, formal_benchmark_executed=False,
        scope='Original 50-step schedule, first scheduler step only; controlled stop before step1 transformer; no VAE or completed output')


def execute(args, result, config, identity, manifest, recorder):
    if any(n == 'xfuser' or n.startswith('xfuser.') for n in sys.modules): raise ValueError('fresh xfuser interpreter required')
    sys.path.insert(0,str(args.source_root.resolve()))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    import torch
    from xfuser.runner import xFuserModelRunner
    from xfuser.core.distributed import get_sequence_parallel_world_size, get_data_parallel_world_size
    if torch.cuda.device_count() != args.world_size: raise ValueError('actual allocated CUDA device count differs')
    torch.cuda.set_device(result['rank'])
    native.loaded_sources(args.source_root, manifest)
    class LocalCheckpointRunner(xFuserModelRunner):
        def _select_model(self, name, upstream):
            model=super()._select_model(name,upstream)
            if type(model).__name__ != native.CLASSES['wan22'] or model.settings.model_name != identity['model_id']:
                raise ValueError('unexpected original model class or checkpoint')
            model.settings.model_name=identity['root'];return model
    runner=None; initialized=False
    try:
        runner=LocalCheckpointRunner(config)
        effective=runner.preprocess_args(dict(vars(runner.config)));native.check_effective_request(result['request'],effective)
        runner.initialize(effective); initialized=True; pipe=runner.model.pipe
        if type(pipe).__name__ != 'WanPipeline' or getattr(pipe,'transformer_2',None) is not None:
            raise ValueError('expected original single-transformer WanPipeline')
        if torch.distributed.get_rank()!=result['rank'] or torch.distributed.get_world_size()!=args.world_size:
            raise ValueError('actual rank/world differs')
        if get_sequence_parallel_world_size()!=args.world_size or get_data_parallel_world_size()!=1:
            raise ValueError('actual SP/DP layout differs')
        result.update(runtime=dict(gpu=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(result['rank']).uuid),
            torch=torch.__version__,cuda=torch.version.cuda,sp=args.world_size,dp=1),
            pipeline_type=type(pipe).__module__+'.'+type(pipe).__qualname__,
            transformer_type=type(pipe.transformer).__module__+'.'+type(pipe.transformer).__qualname__,
            scheduler_type=type(pipe.scheduler).__module__+'.'+type(pipe.scheduler).__qualname__,
            runner_run_source=method_sources(runner.run), model_run_source=method_sources(runner.model._run_pipe),
            pipeline_call_source=method_sources(pipe.__call__),
            components={k:dict(type=type(v).__module__+'.'+type(v).__qualname__,dtype=str(getattr(v,'dtype',None))) for k,v in pipe.components.items()})
        for key in ('transformer','text_encoder','vae'):
            if result['components'][key]['dtype']!='torch.bfloat16':raise ValueError('original bf16 component precision changed')
        with observe(pipe,runner.model,recorder) as observation:
            result['observation']=observation
            try: runner.run(effective)
            except PrefixComplete: pass
            else: raise ValueError('original runner completed without expected prefix stop')
        torch.cuda.synchronize()
        result.update(boundaries=recorder.rows,artifact_bytes=recorder.bytes,
            attention_backend_observation=native.attention_observation(pipe,runner.config.attention_backend),
            loaded_source_sha256=native.loaded_sources(args.source_root,manifest))
        if recorder.reference is not None and recorder.rank==0 and {x['name'] for x in recorder.rows} != set(recorder.reference['boundaries']):
            raise ValueError('oracle/candidate boundary denominator differs')
        if recorder.reference is not None and recorder.rank == 0:
            oracle = recorder.reference['summary']['ranks'][0]
            result['oracle_path_comparison'] = {key: result[key] == oracle[key] for key in
                ('pipeline_type','transformer_type','scheduler_type','runner_run_source','model_run_source','pipeline_call_source')}
            result['oracle_path_comparison']['bound_methods'] = result['observation']['methods'] == oracle['observation']['methods']
            result['oracle_path_comparison']['scheduler_config'] = result['observation']['scheduler_before']['config'] == oracle['observation']['scheduler_before']['config']
        result['status']='first_step_observed'
    finally:
        if initialized:runner.cleanup()
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        result['cleanup_completed']=True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('plan','run'))
    p.add_argument('--world-size',type=int,choices=(1,4),required=True)
    for name in ('source-root','source-manifest','model-root','protocol','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('source-manifest-sha256','protocol-sha256','prompt'):p.add_argument('--'+name,required=True)
    p.add_argument('--models',type=Path,default=native.DEFAULT_MODELS);p.add_argument('--models-sha256',default=native.DEFAULT_MODELS_SHA256)
    p.add_argument('--model',choices=('wan22',),required=True);p.add_argument('--shape',choices=('wan22-small',),required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--timeout-seconds',type=int,default=900)
    p.add_argument('--reference-dir',type=Path);p.add_argument('--reference-summary-sha256')
    args=p.parse_args(argv)
    if not 30<=args.timeout_seconds<=1200:p.error('timeout must be 30..1200 seconds')
    if (args.world_size==4)!=bool(args.reference_dir) or bool(args.reference_dir)!=bool(args.reference_summary_sha256):p.error('only SP4 requires an explicit bound SP1 reference')
    output=args.output_dir
    if output.resolve()!=output or not output.is_relative_to(Path('/shared/output/DITSERVE')):raise ValueError('resolved project output required')
    for path in (args.source_root,args.model_root,args.reference_dir):
        if path is not None and (output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output)):raise ValueError('output must be disjoint from inputs')
    rank=distributed.topology(args.world_size,os.environ) if args.world_size==4 and args.command=='run' else 0
    deadline=time.monotonic()+args.timeout_seconds;stop=threading.Event();created=False
    result=dict(schema=SCHEMA,rank=rank,world_size=args.world_size,status='failed',cleanup_completed=False,
        completed_requests=0,qualification_eligible=False,performance_eligible=False)
    try:
        if args.command=='run':
            native.check_storage();output.mkdir(parents=True,exist_ok=True);rankdir=output/f'rank-{rank}';rankdir.mkdir(exist_ok=False);created=True
            def guard():
                while not stop.wait(5):
                    try:
                        native.check_storage()
                        if time.monotonic()>=deadline:raise TimeoutError('first-step deadline')
                    except Exception:os._exit(75)
            threading.Thread(target=guard,daemon=True).start()
        manifest=native.source.verify_source(args.source_root,args.source_manifest,args.source_manifest_sha256)
        request=native.request_plan(native.bound_json(args.protocol,args.protocol_sha256),'wan22','wan22-small','formal',args.prompt,args.seed)
        identity=native.verify_model('wan22',args.model_root,args.models,args.models_sha256)
        result.update(request=request,model_identity=identity,protocol_sha256=args.protocol_sha256,
            source=dict(commit=native.source.COMMIT,manifest_sha256=args.source_manifest_sha256,archive_sha256=native.source.ARCHIVE_SHA256),
            adapter={Path(m.__file__).name:sha(Path(m.__file__).read_bytes()) for m in (native,distributed,native.source)})
        result['adapter'][Path(__file__).name]=sha(Path(__file__).read_bytes());binding=reference_binding(result);result['binding']=binding
        config=native.runner_config('wan22',request,identity,output) if args.world_size==1 else distributed.distributed_config('wan22',request,identity,output,args.world_size)
        result['runner_config']=config
        reference=load_reference(args.reference_dir,args.reference_summary_sha256,binding) if args.reference_dir and rank == 0 else None
        if args.command=='plan':print(json.dumps(dict(result,status='planned'),sort_keys=True,allow_nan=False));return 0
        if args.world_size==1:
            protection=dict(schema='ditserve.wan-first-step-protection.v1',dependency='SP1-SP4 first-step numerical comparison',protected=True,
                cleanup_allowed=False,automatic_cleanup=False,budget_bytes=BUDGET,binding=binding)
            native.check_storage();distributed.publish_json(output/'protection.json',protection)
        recorder=Recorder(output,rank,reference)
        with redirect_stdout(sys.stderr):execute(args,result,config,identity,manifest,recorder)
        native.source.verify_source(args.source_root,args.source_manifest,args.source_manifest_sha256)
        validate_rank(result)
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()[-8000:])
    if created:
        if args.world_size == 1 and result['status'] == 'first_step_observed':
            budget_check(result['artifact_bytes'], 3*len(canonical(result))+65536)
        native.check_storage();distributed.publish_json(rankdir/'summary.json',result)
        if rank==0:
            try:
                paths=[output/f'rank-{i}'/'summary.json' for i in range(args.world_size)]
                if result['status'] != 'first_step_observed':
                    raise RuntimeError('rank0 failed; do not wait while peers may be in collectives')
                while not all(x.exists() for x in paths):
                    if time.monotonic()>=deadline:raise TimeoutError('missing final rank receipt')
                    native.check_storage();time.sleep(.5)
                rows=[native.source.read_json(x.read_bytes()) for x in paths];summary=aggregate(rows,result['binding'],args.world_size)
                if args.world_size==1:summary['protection_sha256']=sha((output/'protection.json').read_bytes())
                else:summary['reference_summary_sha256']=args.reference_summary_sha256
            except Exception as exc:
                summary=dict(schema=SCHEMA,status='failed',error=repr(exc),qualification_eligible=False,
                    completed_requests=0, ranks=[result])
            native.check_storage();distributed.publish_json(output/'summary.json',summary)
            print(json.dumps(dict(summary=str(output/'summary.json'),status=summary['status'],sha256=sha((output/'summary.json').read_bytes()))))
            if summary['status']=='failed':result['status']='failed'
    stop.set();return 0 if result['status']=='first_step_observed' else 1


if __name__=='__main__':raise SystemExit(main())
