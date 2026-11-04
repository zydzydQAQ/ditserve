"""Source-bound SD3.5 SP2 diagnosis, reusing protected Tetri r2 reference tensors.

Three real 28-step trajectories share one two-GPU allocation: normal xfuser,
Tetri round full28, and Tetri round with the observed r2 chunk boundaries. All
use fixed SP2, unchanged BF16/math attention/CPU RNG. No scheduler, migration,
new reference inference or formal performance claim is made by this diagnostic.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import external_tetriserve_migration as migration
import external_tetriserve_service_numerics as numerics
import external_runtime_probe as probe

sha = migration.sha
write_json = migration.write_json
TRAJECTORIES = ('normal_sp2', 'round_sp2_full', 'round_sp2_chunked')
PAIRS = ('sp1_reference_vs_normal_sp2', 'normal_sp2_vs_round_sp2_full',
         'round_sp2_full_vs_round_sp2_chunked', 'sp1_reference_vs_round_sp2_full',
         'sp1_reference_vs_round_sp2_chunked', 'service_vs_round_sp2_chunked')
EXPECTED_ENDS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]


class RankProgress:
    """One atomically replaced small receipt per owned rank; no CUDA operations."""
    def __init__(self, directory, rank, binding):
        self.directory = Path(directory).absolute()
        if (type(rank) is not int or rank not in (0,1) or self.directory.resolve() != self.directory
                or not self.directory.is_dir()):
            raise ValueError('progress requires one owned rank and an existing unlinked output directory')
        self.path = self.directory / ('rank-'+str(rank)+'.progress.json')
        self.pending = self.path.with_name(self.path.name+'.pending')
        if self.path.exists() or self.pending.exists():
            raise FileExistsError('previous progress must not be overwritten by another worker')
        self.record = dict(schema='ditserve.tetriserve-sp2-progress.v1',rank=rank,pid=os.getpid(),
            adapter_sha256=sha(__file__),prior_summary_sha256=binding['prior_summary_sha256'],
            request_sha256=numerics.canonical(binding['request']),observation_only=True,
            call_count_scope='Python forward entry/return; no new CUDA synchronization',
            qualification_eligible=False,diagnostic_completed=False,started_calls=0,returned_calls=0,
            trajectory=None,cursor=None,error=None,failed_stage=None)

    def update(self, stage, *, trajectory=None, started=None, returned=None, cursor=None):
        if not isinstance(stage,str) or not re.fullmatch('[a-z0-9_]{1,80}',stage):
            raise ValueError('invalid progress stage')
        if trajectory is not None and trajectory not in TRAJECTORIES:
            raise ValueError('invalid progress trajectory')
        for value in (started,returned,cursor):
            if value is not None and (type(value) is not int or not 0<=value<=28):
                raise ValueError('invalid progress call/cursor count')
        row=dict(self.record,stage=stage,monotonic_ns=time.monotonic_ns())
        if trajectory is not None: row['trajectory']=trajectory
        if started is not None: row['started_calls']=started
        if returned is not None: row['returned_calls']=returned
        if cursor is not None: row['cursor']=cursor
        if row['returned_calls']>row['started_calls']:
            raise ValueError('returned calls cannot exceed started calls')
        raw=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode()
        if len(raw)>8192: raise ValueError('progress exceeds fixed small receipt budget')
        if any(path.is_symlink() or path.resolve()!=path for path in (self.path,self.pending)):
            raise ValueError('linked progress path')
        probe.storage()
        with self.pending.open('xb') as stream:
            stream.write(raw)
            stream.flush()
        os.replace(self.pending,self.path)
        self.record=row

    def fail(self, error):
        self.record.update(error=repr(error)[:1000],failed_stage=self.record.get('stage'))
        self.update('worker_failed')


def read_progress(directory):
    records=[]
    for rank in (0,1):
        path=Path(directory)/('rank-'+str(rank)+'.progress.json')
        if not path.exists(): continue
        if path.is_symlink() or path.resolve()!=path.absolute() or path.stat().st_size>8192:
            raise ValueError('invalid final progress path or size')
        row=json.loads(path.read_bytes())
        if (row.get('schema')!='ditserve.tetriserve-sp2-progress.v1' or row.get('rank')!=rank
                or row.get('observation_only') is not True or row.get('diagnostic_completed') is not False):
            raise ValueError('invalid final progress identity')
        records.append(row)
    return records


def intervals_from_dispatches(dispatches, native_id):
    rows = [r for r in dispatches if r['request_ids'] == [native_id]]
    cursor, ends = 0, []
    for row in rows:
        if (row['start'] != cursor or type(row['end']) is not int or row['end'] <= cursor
                or row['height'] != 1024 or row['width'] != 1024 or row['batch_size'] != 1
                or row['ranks'] != ([0] if cursor < 10 else [0, 1])):
            raise ValueError('observed 1024 SP1-to-SP2 dispatch sequence differs')
        cursor = row['end']; ends.append(cursor)
    if ends != EXPECTED_ENDS:
        raise ValueError('observed complete 28-step chunk boundaries differ')
    return ends


def bind_prior(root, child_sha, numerical_sha):
    root = Path(root).absolute()
    if root.resolve(strict=True) != root:
        raise ValueError('prior evidence root must not be linked')
    child_path, numeric_path = root/'summary.json', root/'numerics/summary.json'
    if sha(child_path) != child_sha or sha(numeric_path) != numerical_sha:
        raise ValueError('prior immutable summaries differ')
    child, numeric = json.loads(child_path.read_bytes()), json.loads(numeric_path.read_bytes())
    if (child.get('status') != 'numerical_failed' or child.get('worker_exitcodes') != [0, 0]
            or child.get('counts', {}).get('requests_completed') != 2
            or numeric.get('status') != 'numerical_failed' or numeric.get('captured_requests') != 2
            or numeric.get('raw_artifacts_protected') is not True):
        raise ValueError('expected completed service with preserved failed numerical evidence')
    plan_sha = child['numerics_plan_sha256']
    plan = numerics.load_plan(root/'numerics/numerics-plan.json', plan_sha)
    service = numerics.closed_service(root/'service-completion.json', numeric['service_summary_sha256'], plan)
    captures = numerics.bound_captures(root/'numerics', plan, plan_sha, service)
    offered = [r for r in plan['requests'] if r['request_id'] == 101]
    if len(offered) != 1:
        raise ValueError('expected exact retained 1024 request')
    request = offered[0]['request']
    expected = dict(prompt='A white rabbit in a field of blue flowers.', negative_prompt='blurry, low quality',
        height=1024, width=1024, steps=28, seed=102, guidance_scale=4.5, max_sequence_length=256)
    if request != expected:
        raise ValueError('original request scientific settings differ')
    reference_rows = [r for r in numeric['comparisons'] if r['request_id'] == 101]
    if len(reference_rows) != 1:
        raise ValueError('one actual full reference is required')
    row = reference_rows[0]
    if (row['reference_transformer_calls'] != 28 or row['generator_device'] != 'cpu'
            or row['normal_pipeline_output_type'] != 'latent'
            or row['request_sha256'] != numerics.canonical(request)
            or row['service_file_sha256'] != captures[101]['tensor_file']['sha256']
            or row['reference_file']['file'] != 'request-101.safetensors'):
        raise ValueError('reference identity/execution differs')
    reference = root/'numerics/reference/request-101.safetensors'
    descriptors = numerics.inspect_safetensors(reference, request, numerics.canonical(request), plan_sha)
    if (sha(reference) != row['reference_file']['sha256']
            or descriptors != row['reference_file']['tensors']
            or reference.stat().st_size != row['reference_file']['bytes']):
        raise ValueError('protected full reference changed')
    ends = intervals_from_dispatches(service['dispatches'], captures[101]['native_request_id'])
    derived = root/'derived'
    manifest = migration.verify_derived(derived, child['derived_manifest_sha256'])
    # A diagnosis must execute the same adapter implementation as the failed run.
    package = Path(__file__).resolve().parent
    for name in ('external_sd35_round.py', 'external_tetriserve_bridge.py'):
        if sha(package/name) != manifest['files'][name]:
            raise ValueError('diagnostic adapter differs from failed run: '+name)
    config_path = root/'service-port-config.json'
    config = json.loads(config_path.read_bytes())
    if (config['derived_manifest_sha256'] != child['derived_manifest_sha256']
            or config['model_manifest_sha256'] != child['model_manifest_sha256']
            or config['attention'] != 'math' or config['guidance_scale'] != 4.5
            or config['max_sequence_length'] != 256
            or Path(config['derived_manifest']) != derived/'PORT_MANIFEST.json'):
        raise ValueError('prior model/derived/profile binding differs')
    return dict(request=request, chunk_ends=ends, reference=str(reference),
        service_raw=str(root/'numerics/raw/request-101.safetensors'), reference_sha256=sha(reference),
        service_raw_sha256=captures[101]['tensor_file']['sha256'], prior_summary_sha256=child_sha,
        prior_numerics_sha256=numerical_sha, plan_sha256=plan_sha, sources=plan['sources'],
        derived=dict(root=str(derived), manifest_sha256=child['derived_manifest_sha256']),
        config_path=str(config_path), config_sha256=sha(config_path), config=config)


def validate_comparison(row, count, reference, actual):
    if (row.get('elements') != count or row.get('atol') != 1e-5 or row.get('rtol') != 1e-5
            or type(row.get('passed')) is not bool or type(row.get('bitwise_equal')) is not bool):
        return False
    fields = ('max_abs_error', 'rmse', 'relative_l2_error', 'cosine_similarity')
    if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) for k in fields):
        return False
    if not 0 <= row['rmse'] <= row['max_abs_error'] or row['relative_l2_error'] < 0:
        return False
    if not -1.000000000001 <= row['cosine_similarity'] <= 1.000000000001:
        return False
    if row['passed'] and row['max_abs_error'] > 1e-5 + 1e-5 * max(abs(reference['minimum']), abs(reference['maximum'])):
        return False
    if row['bitwise_equal'] != (reference['sha256'] == actual['sha256']):
        return False
    if row['bitwise_equal'] and (not row['passed'] or row['max_abs_error'] != 0 or row['relative_l2_error'] != 0):
        return False
    return True


def valid_initial(values):
    if set(values) != set(migration.NAMES):
        return False
    for name,value in values.items():
        shape=value.get('shape')
        if (not isinstance(shape,list) or any(type(x) is not int or x<1 for x in shape)
                or value.get('finite') is not True
                or not re.fullmatch('[0-9a-f]{64}',value.get('sha256',''))
                or any(type(value.get(k)) not in (int,float) or not math.isfinite(value[k]) for k in ('minimum','maximum'))
                or value['minimum']>value['maximum']):
            return False
        if name in ('timesteps','sigmas'):
            if shape != ([28] if name=='timesteps' else [29]) or value.get('dtype')!='torch.float32':
                return False
        elif value.get('dtype')!='torch.bfloat16':
            return False
        elif name=='latents' and shape != [1,16,128,128]:
            return False
        elif name=='prompt_embeds' and (len(shape)!=3 or shape[0]!=2):
            return False
        elif name=='pooled_prompt_embeds' and (len(shape)!=2 or shape[0]!=2):
            return False
    return True


def assess(records):
    """Completion is independent from numerical agreement; differences remain explicit."""
    if len(records) != 2 or {r.get('rank') for r in records} != {0, 1}:
        return False
    rows = sorted(records, key=lambda r:r['rank'])
    if len({r.get('gpu_uuid') for r in rows}) != 2 or any(not r.get('gpu_uuid') for r in rows):
        return False
    for row in rows:
        if (row.get('status') != 'completed' or row.get('distributed_closed') is not True
                or row.get('groups') != [0,1] or row.get('actual_sp') != 2 or row.get('actual_dp') != 1
                or row.get('chunk_ends') != EXPECTED_ENDS
                or row.get('transformer_calls') != {name:28 for name in TRAJECTORIES}
                or set(row.get('outputs', {})) != set(TRAJECTORIES)
                or not valid_initial(row.get('initial_five_tensors', {}))):
            return False
        for trajectory in row['outputs'].values():
            if set(trajectory) != {'latents','decoded'}:
                return False
            for name, shape in numerics.raw_shapes(dict(height=1024,width=1024)).items():
                value = trajectory[name]
                if (value.get('shape') != shape or value.get('dtype') != 'torch.bfloat16'
                        or value.get('finite') is not True
                        or not re.fullmatch('[0-9a-f]{64}', value.get('sha256',''))):
                    return False
    if (rows[0]['initial_five_tensors'] != rows[1]['initial_five_tensors']
            or rows[0]['outputs'] != rows[1]['outputs']):
        return False
    comparisons = rows[0].get('comparisons', {})
    inputs = dict(rows[0]['outputs'], **rows[0].get('comparison_inputs', {}))
    if set(inputs) != set(TRAJECTORIES) | {'sp1_reference', 'service'}:
        return False
    for tensors in inputs.values():
        if set(tensors) != {'latents', 'decoded'}:
            return False
        for name, value in tensors.items():
            if (value.get('finite') is not True or value.get('dtype') != 'torch.bfloat16'
                    or value.get('shape') != numerics.raw_shapes(dict(height=1024,width=1024))[name]
                    or not re.fullmatch('[0-9a-f]{64}',value.get('sha256',''))
                    or any(type(value.get(k)) not in (int,float) or not math.isfinite(value[k]) for k in ('minimum','maximum'))
                    or value['minimum'] > value['maximum']):
                return False
    if set(comparisons) != set(PAIRS):
        return False
    for key,pair in comparisons.items():
        left,right=key.split('_vs_')
        if set(pair) != {'latents','decoded'} or not all(validate_comparison(pair[k],math.prod(shape),
                inputs[left][k],inputs[right][k]) for k,shape in numerics.raw_shapes(dict(height=1024,width=1024)).items()):
            return False
    return True


def worker(rank, args, binding):
    result = dict(rank=rank,status='failed',stage='binding',distributed_closed=False)
    pipeline = None
    progress = None
    try:
        progress = RankProgress(args.output_dir,rank,binding)
        progress.update('source_binding_enter')
        migration.verify_derived(**{'root':binding['derived']['root'], 'expected':binding['derived']['manifest_sha256']})
        if sha(binding['config_path']) != binding['config_sha256']:
            raise ValueError('prior port config changed before spawn')
        # Clear only the pure planning adapter; execute the exact bound derived copy.
        numerics.service.bind_derived_adapters(binding['derived'])
        sys.path.insert(0,binding['config']['shared_python_root'])
        os.environ.update(DITSERVE_TETRI_PORT_CONFIG=binding['config_path'],
            DITSERVE_TETRI_PORT_CONFIG_SHA256=binding['config_sha256'])
        progress.update('source_binding_return')
        import torch
        import torch.distributed as dist
        from safetensors.torch import load_file
        from external_sd35_round import ImageRequest, RoundState
        from external_sd35_round_check import compare_tensors, tensor_summary
        from external_tetriserve_bridge import TetriSharedSD35Pipeline
        from tetriserve.utils import set_device
        from tetriserve.distributed.globals import PROCESS_GROUP, set_seq_parallel_pg
        if torch.cuda.device_count() != 2:
            raise ValueError('exactly two GPUs required')
        torch.cuda.set_device(rank); set_device(gpu=True)
        result['gpu_uuid'] = str(torch.cuda.get_device_properties(rank).uuid)
        progress.update('default_group_enter')
        dist.init_process_group('nccl',init_method='file://'+str(args.output_dir/'rendezvous'),
            rank=rank,world_size=2,timeout=timedelta(seconds=180))
        progress.update('default_group_return')
        progress.update('native_groups_enter')
        PROCESS_GROUP.generate_all_process_groups(2,skip_latent_transfer=False)
        progress.update('native_groups_return')
        control = PROCESS_GROUP.get_process_group([0,1],device='cpu')
        result['stage']='load_same_verified_model'
        progress.update('model_load_enter')
        pipeline = TetriSharedSD35Pipeline.from_pretrained('stabilityai/stable-diffusion-3.5-medium',
            migration.upstream_engine_config('stabilityai/stable-diffusion-3.5-medium'),torch.bfloat16)
        progress.update('model_load_return')
        progress.update('sp2_activation_enter')
        set_seq_parallel_pg(2,1,rank,[0,1],2)
        pipeline.runtime.pool.activate((0,1),pipeline.runtime.state,ulysses_degree=2,ring_degree=1)
        pipeline.runtime.bind_global()
        from xfuser.core.distributed import get_sequence_parallel_world_size, get_data_parallel_world_size
        actual_sp,actual_dp=get_sequence_parallel_world_size(),get_data_parallel_world_size()
        if pipeline.runtime.pool.current_ranks!=(0,1) or actual_sp!=2 or actual_dp!=1:
            raise ValueError('actual shared SP2/DP1 group activation differs')
        result.update(groups=list(pipeline.runtime.pool.current_ranks),actual_sp=actual_sp,actual_dp=actual_dp,
                      chunk_ends=binding['chunk_ends'],transformer_calls={},outputs={},
                      provenance=pipeline.provenance)
        progress.update('sp2_activation_return')
        request = ImageRequest(**binding['request'])
        progress.update('prepare_enter')
        initial = pipeline.prepare(request)
        progress.update('prepare_return')
        result['initial_five_tensors'] = {name:tensor_summary(t) for name,t in zip(migration.NAMES,initial.transferred_tensors())}
        outputs = {}
        for name in TRAJECTORIES:
            result['stage']=name
            calls=[0]; returned=[0]
            progress.update('trajectory_enter',trajectory=name,started=0,returned=0,cursor=0)
            def count_forward(*unused):
                calls[0]+=1
                progress.update('transformer_enter',trajectory=name,started=calls[0],returned=returned[0])
            def count_return(*unused):
                returned[0]+=1
                progress.update('transformer_return',trajectory=name,started=calls[0],returned=returned[0])
            handle=pipeline.pipe.transformer.register_forward_pre_hook(count_forward)
            return_handle=pipeline.pipe.transformer.register_forward_hook(count_return)
            try:
                with torch.inference_mode():
                    if name == 'normal_sp2':
                        latents = pipeline.pipe(prompt=request.prompt,negative_prompt=request.negative_prompt,
                            height=request.height,width=request.width,num_inference_steps=28,
                            guidance_scale=4.5,max_sequence_length=256,
                            generator=torch.Generator(device='cpu').manual_seed(102),output_type='latent').images
                    else:
                        # Clone all five actual tensors; no mutation crosses trajectories.
                        state=RoundState(request,*(t.detach().clone() for t in initial.transferred_tensors()))
                        boundaries=[28] if name=='round_sp2_full' else binding['chunk_ends']
                        for end in boundaries:
                            progress.update('round_enter',trajectory=name,cursor=state.cursor)
                            state=pipeline.advance(state,end)
                            progress.update('round_return',trajectory=name,cursor=state.cursor)
                            # Exactly the transferred protocol, with explicit absolute cursor.
                            state=RoundState(request,*(t.detach().clone() for t in state.transferred_tensors()),cursor=end)
                        latents=state.latents
                    progress.update('decode_enter',trajectory=name)
                    decoded=pipeline.raw.vae.decode(latents/pipeline.raw.vae.config.scaling_factor
                        +pipeline.raw.vae.config.shift_factor,return_dict=False)[0]
                    torch.cuda.synchronize()
                    progress.update('decode_return',trajectory=name)
                    outputs[name]={'latents':latents.detach().cpu(),'decoded':decoded.detach().cpu()}
                    result['outputs'][name]={k:tensor_summary(v) for k,v in outputs[name].items()}
                    result['transformer_calls'][name]=calls[0]
                    if calls[0] != 28: raise ValueError('trajectory must execute exactly 28 transformer calls')
            finally:
                handle.remove()
                return_handle.remove()
            progress.update('trajectory_return',trajectory=name,cursor=28)
            progress.update('trajectory_barrier_enter',trajectory=name)
            dist.barrier(group=control)
            progress.update('trajectory_barrier_return',trajectory=name)
        if rank==0:
            progress.update('comparison_enter')
            result['stage']='compare_existing_reference_without_inference'
            for path,key in ((binding['reference'],'reference_sha256'),(binding['service_raw'],'service_raw_sha256')):
                if sha(path)!=binding[key]: raise ValueError('prior raw changed before comparison')
            outputs['sp1_reference']=load_file(binding['reference'],device='cpu')
            outputs['service']=load_file(binding['service_raw'],device='cpu')
            result['comparison_inputs']={name:{k:tensor_summary(v) for k,v in outputs[name].items()}
                for name in ('sp1_reference','service')}
            result['comparisons']={}
            for pair in PAIRS:
                left,right=pair.split('_vs_')
                result['comparisons'][pair]={k:compare_tensors(outputs[left][k],outputs[right][k])
                    for k in ('latents','decoded')}
            progress.update('comparison_return')
        progress.update('final_barrier_enter')
        dist.barrier(group=control)
        progress.update('final_barrier_return')
        result.update(status='completed',stage='cleanup')
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc()[-14000:])
        if progress is not None:
            try: progress.fail(exc)
            except BaseException as progress_error: result['progress_error']=repr(progress_error)
    finally:
        def observe_cleanup(stage):
            if progress is not None:
                try: progress.update(stage)
                except BaseException as exc: result.update(status='failed',progress_error=repr(exc))
        try:
            observe_cleanup('runtime_cleanup_enter')
            if pipeline is not None: pipeline.runtime.close()
            observe_cleanup('runtime_cleanup_return')
            import torch.distributed as dist
            observe_cleanup('distributed_cleanup_enter')
            if dist.is_initialized(): dist.destroy_process_group()
            observe_cleanup('distributed_cleanup_return')
            result['distributed_closed']=not dist.is_initialized()
        except BaseException as exc:
            result.update(status='failed',cleanup_error=repr(exc))
        probe.storage()
        write_json(args.output_dir/('rank-'+str(rank)+'.json'),result)
    if result['status']!='completed':
        raise RuntimeError('SP2 diagnosis worker failed; see preserved rank summary')


def read_rank_results(directory):
    # Fixed progress files must never become completion/qualification receipts.
    paths=[Path(directory)/('rank-'+str(rank)+'.json') for rank in (0,1)]
    return [json.loads(path.read_bytes()) for path in paths if path.is_file()]


def execute(args,binding):
    import torch.multiprocessing as mp
    children=None; error=None
    start=time.monotonic()
    try:
        children=mp.spawn(worker,args=(args,binding),nprocs=2,join=False)
        while not children.join(timeout=1):
            probe.storage()
            if time.monotonic()-start>args.timeout_seconds:
                raise TimeoutError('SP2 diagnostic timeout')
    except BaseException as exc:
        error=repr(exc)
    finally:
        if children:
            for child in children.processes:
                if child.is_alive(): child.terminate()
            for child in children.processes:
                child.join(timeout=5)
                if child.is_alive(): child.kill(); child.join(timeout=5)
    rows=read_rank_results(args.output_dir)
    return rows,error



def retain_execution(result, rows, error, directory):
    # Preserve the causal timeout/worker failure even if an observation is corrupt.
    result['ranks']=rows
    if error: result['worker_error']=error
    result['last_progress']=read_progress(directory)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','run'])
    parser.add_argument('--prior-result',type=Path,required=True)
    parser.add_argument('--prior-summary-sha256',required=True)
    parser.add_argument('--prior-numerics-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    args=parser.parse_args(argv)
    args.output_dir=args.output_dir.absolute()
    args.prior_result=args.prior_result.absolute()
    result=dict(schema='ditserve.tetriserve-sp2-diagnostic.v1',status='failed',diagnostic_completed=False,
        numerical_check_passed=False,qualification_eligible=False,allows=[],serving_executed=False,
        migration_executed=False,performance_measurements=False,reference_inference_executed=False,
        retained_new_arrays=False,trajectories=list(TRAJECTORIES),tolerance=dict(atol=1e-5,rtol=1e-5))
    created=False
    try:
        if not 120<=args.timeout_seconds<=3600: raise ValueError('invalid diagnostic timeout')
        if args.command=='run': probe.storage()
        binding=bind_prior(args.prior_result,args.prior_summary_sha256,args.prior_numerics_sha256)
        result['binding']=binding
        if args.command=='plan':
            result['status']='planned'; print(json.dumps(result,sort_keys=True,allow_nan=False)); return 0
        root=Path('/shared/output/DITSERVE')
        if (not args.output_dir.is_relative_to(root) or args.output_dir.resolve()!=args.output_dir
                or args.output_dir.is_relative_to(args.prior_result) or args.prior_result.is_relative_to(args.output_dir)):
            raise ValueError('new output must be disjoint from protected prior artifacts')
        args.output_dir.mkdir(parents=True,exist_ok=False); created=True
        rows,error=execute(args,binding)
        retain_execution(result,rows,error,args.output_dir)
        complete=error is None and assess(rows)
        comparisons=next((r.get('comparisons',{}) for r in rows if r.get('rank')==0),{})
        result.update(diagnostic_completed=complete,status='diagnostic_completed' if complete else 'failed',
            numerical_check_passed=complete and all(c['passed'] for pair in comparisons.values() for c in pair.values()))
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc()[-14000:])
    if created:
        probe.storage(); write_json(args.output_dir/'summary.json',result)
    print(json.dumps(dict(status=result['status'],diagnostic_completed=result['diagnostic_completed'],
        numerical_check_passed=result['numerical_check_passed'],qualification_eligible=False),allow_nan=False),flush=True)
    return 0 if result['diagnostic_completed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
