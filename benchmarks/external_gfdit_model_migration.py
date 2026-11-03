"""Actual SD3.5 step-2 state migration through original GF-DiT/GFC, then resume.

Rank 0 executes two real denoising steps. Original GF-DiT migration carries all
five CUDA tensors and CPU request/cursor metadata to rank 1. Rank 1 executes the
remaining 26 steps and independently runs the normal shared 28-step pipeline.
This verifies SP1-to-SP1 model continuation, not EDF decisions, the native vLLM
worker pool, SP2 reshaping, serving performance or paper quality qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import threading
import traceback

import external_gfdit_migration as migration
import external_sd35_round as adapter
from external_sd35_round_check import (
    ATOL, RTOL, DEFAULT_MODEL_MANIFEST_SHA256, compare_tensors, request_plan, tensor_summary,
)

sys.dont_write_bytecode = True
TENSOR_FIELDS = ('prompt_embeds','pooled_prompt_embeds','latents','timesteps','sigmas')
MID = 'sd35-real-state-rank0-to-rank1-step2'
REQUEST_ID = 'sd35-real-state'
STATE_SCHEMA = 'ditserve.sd35-five-tensor-continuation.v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pure_request(value):
    request = adapter.ImageRequest(**value)
    if asdict(request) != value:
        raise ValueError('continuation request must explicitly retain every immutable field')
    if (request.height,request.width,request.steps,request.guidance_scale,request.max_sequence_length)!=(512,512,28,4.5,256):
        raise ValueError('this migration check is bound to the formal SD3.5 512 profile')
    return request


def validate_metadata(value, expected_request):
    if value.get('schema') != STATE_SCHEMA or type(value.get('cursor')) is not int or value['cursor'] != 2:
        raise ValueError('actual continuation must be exactly after step 2')
    pure_request(value.get('request',{}))
    if value['request'] != asdict(expected_request):
        raise ValueError('continuation changed prompt, seed, resolution or solver profile')


def validate_descriptor(descriptor, request):
    validate_metadata(descriptor,request)
    fields=descriptor.get('tensor_fields',{})
    if set(fields)!=set(TENSOR_FIELDS):
        raise ValueError('all five original continuation tensors are required')
    for name in TENSOR_FIELDS:
        item=fields[name]
        shape=item.get('shape',[])
        dtype='torch.float32' if name in ('timesteps','sigmas') else 'torch.bfloat16'
        if (not shape or any(type(d) is not int or d<1 for d in shape)
                or item.get('dtype')!=dtype or not re.fullmatch('[0-9a-f]{64}',item.get('sha256',''))
                or type(item.get('bytes')) is not int or item['bytes']!=math.prod(shape)*(4 if dtype=='torch.float32' else 2)):
            raise ValueError('invalid actual continuation tensor descriptor: '+name)
    expected={'latents':[1,16,64,64],'timesteps':[28],'sigmas':[29]}
    if any(fields[name]['shape']!=shape for name,shape in expected.items()):
        raise ValueError('continuation latent/schedule geometry differs')
    if len(fields['prompt_embeds']['shape'])!=3 or fields['prompt_embeds']['shape'][:2]!=[2,333]:
        raise ValueError('original CFG prompt embedding pair/text length differs')
    if len(fields['pooled_prompt_embeds']['shape'])!=2 or fields['pooled_prompt_embeds']['shape'][0]!=2:
        raise ValueError('original CFG pooled embedding pair differs')
    migration.no_tensors(descriptor)
    return descriptor


def payload_from_state(state):
    if not isinstance(state,adapter.RoundState) or state.cursor!=2:
        raise ValueError('requires a real step-2 round state')
    return dict(schema=STATE_SCHEMA,request=asdict(state.request),cursor=state.cursor,
                **{name:getattr(state,name) for name in TENSOR_FIELDS})


def describe_payload(value,request,device):
    import torch
    validate_metadata(value,request)
    if set(value)!=set(TENSOR_FIELDS)|{'schema','request','cursor'}:
        raise ValueError('unexpected continuation field could escape tensor migration')
    fields={}
    for name in TENSOR_FIELDS:
        tensor=value[name]
        if not isinstance(tensor,torch.Tensor) or tensor.device!=device:
            raise ValueError('continuation tensor is missing or on the wrong actual GPU: '+name)
        details=tensor_summary(tensor)
        fields[name]={key:details[key] for key in ('shape','dtype','sha256')}
        fields[name]['bytes']=tensor.numel()*tensor.element_size()
    return validate_descriptor(dict(schema=value['schema'],request=value['request'],cursor=value['cursor'],
                                    tensor_fields=fields),request)


def make_codec(control,request):
    base=control.codec.ArtifactLayoutCodec
    class Codec(base):
        codec_id='shared-sd35-five-tensors-v1'
        def describe_output(self,*,group,group_rank,request_metadata,artifact):
            validate_descriptor(request_metadata,request)
            if len(group.ranks)!=1 or group_rank!=0 or group.parallel_spec!=control.protocol.ParallelSpec():
                raise ValueError('real model migration is restricted to SP1 singletons')
            tensors=tuple(control.common.replicated_tensor_field(name,tuple(request_metadata['tensor_fields'][name]['shape']),
                request_metadata['tensor_fields'][name]['dtype']) for name in TENSOR_FIELDS)
            return control.protocol.OutputArtifactLayout(artifact,tensors)
        def pack_metadata(self,*,value,layout):
            validate_metadata(value,request)
            if set(value)!=set(TENSOR_FIELDS)|{'schema','request','cursor'}:
                raise ValueError('undeclared model continuation field')
            raw=super().pack_metadata(value=value,layout=layout)
            skeleton=pickle.loads(raw)
            migration.no_tensors(skeleton)
            if any(skeleton[name] is not None for name in TENSOR_FIELDS) or len(raw)>4096:
                raise ValueError('model tensor leaked into control metadata')
            return raw
        def assemble(self,*,metadata_bytes,tensors,layout,device=None):
            value=super().assemble(metadata_bytes=metadata_bytes,tensors=tensors,layout=layout,device=device)
            validate_metadata(value,request)
            return value
    return Codec()


def build_plan(control,descriptor):
    request=pure_request(descriptor['request'])
    validate_descriptor(descriptor,request)
    p=control.protocol
    specs={name:p.ExecutionGroupSpec(name,(rank,),p.ParallelSpec(),(p.TaskKind.DIT_STEP_CHUNK,))
           for rank,name in enumerate(('model-rank0','model-rank1'))}
    source=p.ArtifactHandle(REQUEST_ID,'actual-state-rank0-step2',p.ArtifactKind.REQUEST_STATE,p.ArtifactLayout.WORKER_LOCAL,
                            codec_id='shared-sd35-five-tensors-v1')
    destination=p.ArtifactHandle(REQUEST_ID,'actual-state-rank1-step2',p.ArtifactKind.REQUEST_STATE,p.ArtifactLayout.WORKER_LOCAL,
                                 codec_id=source.codec_id)
    plan=p.MigrateArtifactsPlan(REQUEST_ID,'model-rank0','model-rank1',
                                (p.MigrateArtifactSpec(destination,source),),descriptor)
    return specs,plan


def restore_state(value,request,descriptor,device):
    actual=describe_payload(value,request,device)
    if actual!=descriptor:
        raise ValueError('actual migrated five tensors/request/cursor differ from source')
    return adapter.RoundState(request,*(value[name] for name in TENSOR_FIELDS),cursor=value['cursor'])


def expected_p2p(descriptor,rank):
    fields=descriptor['tensor_fields'].values()
    byte_count=sum(item['bytes'] for item in fields)
    calls=sum((item['bytes']+migration.BUFFER_BYTES-1)//migration.BUFFER_BYTES for item in fields)
    return dict(send_calls=calls if rank==0 else 0,recv_calls=calls if rank==1 else 0,
                send_bytes=byte_count if rank==0 else 0,recv_bytes=byte_count if rank==1 else 0)


def expected_trace():
    return [dict(migrate_id=MID,phase=phase,targets=[0] if phase=='build_schedule' else [0,1]) for phase in migration.PHASES]


def finite_number(value):
    return type(value) in (int,float) and math.isfinite(value)


def assess(records,descriptor):
    validate_descriptor(descriptor,pure_request(descriptor['request']))
    if len(records)!=2 or sorted(row.get('rank') for row in records)!=[0,1]:
        raise ValueError('two final real rank records required')
    by_rank={row['rank']:row for row in records}
    uuids={row.get('gpu_uuid') for row in records}
    if len(uuids)!=2 or None in uuids or 'unavailable' in uuids:
        raise ValueError('actual GPU identities must be distinct')
    bindings={row.get('source_binding_sha256') for row in records}
    if len(bindings)!=1 or not next(iter(bindings)):
        raise ValueError('rank source/model binding differs')
    for rank,row in by_rank.items():
        if row.get('status')!='passed' or row.get('cleanup_verified') is not True or row.get('model_weights_loaded') is not True:
            raise ValueError('rank did not complete real model execution and cleanup')
        if (type(row.get('phase_executions')) is not int or row['phase_executions']!=(4 if rank==0 else 3)
                or type(row.get('migrations')) is not int or row['migrations']!=1):
            raise ValueError('original migration lifecycle denominator differs')
        if row.get('descriptor')!=descriptor:
            raise ValueError('rank model continuation descriptors differ')
        if row.get('phase_trace')!=(expected_trace() if rank==0 else []):
            raise ValueError('original migration phase order differs')
        if row.get('p2p')!=expected_p2p(descriptor,rank) or row.get('expected_p2p')!=row['p2p']:
            raise ValueError('five actual tensors did not use the exact native GFC payload path')
        if any(type(value) is not int for value in row['p2p'].values()):
            raise ValueError('actual GFC byte/call counts must be integers')
        if row.get('model_sp_ranks')!=[rank] or row.get('model_pool_rank')!=rank:
            raise ValueError('actual model SP1 rank binding differs')
    if by_rank[0].get('source_transformer_calls')!=2 or by_rank[0].get('source_interval')!=[0,2]:
        raise ValueError('actual source transformer did not execute exactly two steps')
    destination=by_rank[1]
    if (destination.get('resume_transformer_calls')!=26 or destination.get('reference_transformer_calls')!=28
            or destination.get('resume_interval')!=[2,28] or destination.get('received_tensor_hashes_match') is not True):
        raise ValueError('true migrated continuation or independent full reference is incomplete')
    comparisons=destination.get('comparisons',{})
    if set(comparisons)!={'latents','images'}:
        raise ValueError('latent and decoded-image comparisons are both required')
    outputs=destination.get('outputs',{})
    shapes={'reference_latents':[1,16,64,64],'actual_latents':[1,16,64,64],
            'reference_image':[1,3,512,512],'actual_image':[1,3,512,512]}
    if set(outputs)!=set(shapes) or destination.get('raw_decodes_finite') is not True:
        raise ValueError('all actual/reference tensors and finite preclamp decodes are required')
    for name,shape in shapes.items():
        item=outputs[name]
        if (item.get('shape')!=shape or item.get('dtype')!='torch.bfloat16' or item.get('finite') is not True
                or not re.fullmatch('[0-9a-f]{64}',item.get('sha256',''))
                or any(not finite_number(item.get(key)) for key in ('minimum','maximum'))
                or item['minimum']>item['maximum'] or item.get('retained_array') is not False):
            raise ValueError('invalid complete output tensor evidence: '+name)
    for name,item in comparisons.items():
        if item.get('passed') is not True or item.get('atol')!=ATOL or item.get('rtol')!=RTOL:
            raise ValueError('actual resumed model differs from complete shared reference')
        if type(item.get('elements')) is not int or item['elements']!={'latents':65536,'images':786432}[name]:
            raise ValueError('complete latent/image comparison denominator differs')
        errors=[item.get(key) for key in ('max_abs_error','rmse','relative_l2_error')]
        cosine=item.get('cosine_similarity')
        if (any(not finite_number(value) or value<0 for value in errors)
                or not finite_number(cosine) or not -1.000001<=cosine<=1.000001
                or type(item.get('bitwise_equal')) is not bool):
            raise ValueError('nonfinite or invalid numerical comparison evidence')
        suffix='latents' if name=='latents' else 'image'
        reference,actual=outputs['reference_'+suffix],outputs['actual_'+suffix]
        if item['bitwise_equal'] and (any(errors) or reference['sha256']!=actual['sha256']):
            raise ValueError('bitwise equality contradicts output hashes or numerical errors')
        if item['max_abs_error']>ATOL+RTOL*max(abs(reference['minimum']),abs(reference['maximum'])):
            raise ValueError('reported maximum error contradicts tolerance')
    return dict(schema='ditserve.gfdit-sd35-model-migration-check.v1',status='numerical_passed',
        source_interval=[0,2],destination_interval=[2,28],reference_steps=28,
        model_transformer_calls=56,migrations=1,migrated_tensors=5,descriptor=descriptor,ranks=records,
        comparisons=comparisons,real_model_continuation_verified=True,
        original_migration_engine_executed=True,original_data_plane_executed=True,
        edf_decisions_tested=False,native_multiproc_worker_reproduced=False,sp2_reshaping_tested=False,
        serving_executed=False,performance_measurements=False,qualification_eligible=False,allows=[],
        cleanup_scope='GFC, model process groups and artifact handles; require platform Succeeded plus final rank files')


def model_groups(dist,rank):
    """Actual common model groups; created before entering the no-fallback gate."""
    if dist.get_world_size()!=2 or dist.get_rank()!=rank:
        raise ValueError('actual model world/rank differs')
    mapping={ranks:dist.new_group(list(ranks),backend='nccl',timeout=timedelta(seconds=600)) for ranks in ((0,),(1,))}
    mapping[(0,1)]=dist.group.WORLD
    class Groups:
        backend='nccl'
        timeout_seconds=600
        def __init__(self):
            self.rank=rank
            self.groups=mapping
        def group(self,ranks):
            return mapping[tuple(ranks)]
        def close(self):
            for ranks,group in mapping.items():
                if len(ranks)==1 and rank in ranks:
                    dist.destroy_process_group(group)
    return Groups()


def count_transformer_calls(pipeline,expected,function):
    calls=0
    def observe(*unused):
        nonlocal calls
        calls+=1
    hook=pipeline.pipe.transformer.register_forward_pre_hook(observe)
    try:
        result=function()
        pipeline.torch.cuda.synchronize()
    finally:
        hook.remove()
    if calls!=expected:
        raise ValueError(f'actual transformer call count differs: expected {expected}, observed {calls}')
    return result,calls


def decode(pipeline,latents):
    with pipeline._execution():
        image=pipeline.raw.vae.decode(latents/pipeline.raw.vae.config.scaling_factor
                                      +pipeline.raw.vae.config.shift_factor,return_dict=False)[0]
        if not pipeline.torch.isfinite(image).all().item():
            raise ValueError('nonfinite raw VAE image before postprocess/clamp')
        return pipeline.raw.image_processor.postprocess(image,output_type='pt')


def storage_watchdog(stop):
    while not stop.wait(5):
        try: migration.gfc_check.storage_guard()
        except BaseException: os._exit(75)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','source-manifest','gfc-source-root','gfc-source-manifest','shared-python-root',
                 'model-root','model-manifest','protocol','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('source-manifest-sha256','gfc-source-manifest-sha256','protocol-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--model-manifest-sha256',default=DEFAULT_MODEL_MANIFEST_SHA256)
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--seed',type=int,default=101)
    args=parser.parse_args(argv)
    rank=int(os.environ.get('RANK','-1'))
    world=int(os.environ.get('WORLD_SIZE','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','-1'))
    output=args.output_dir.resolve()
    record=dict(rank=rank,status='started',stage='binding',phase_executions=0,phase_trace=[],migrations=0,
        model_weights_loaded=False,cleanup_verified=False,qualification_eligible=False,allows=[],
        p2p=dict(send_calls=0,recv_calls=0,send_bytes=0,recv_bytes=0))
    owned=False
    stop=threading.Event()
    def checkpoint():
        migration.gfc_check.storage_guard()
        temporary=output/f'rank-{rank}.{os.getpid()}.tmp'
        with temporary.open('xb') as stream: stream.write(migration.encoded(record))
        os.replace(temporary,output/f'rank-{rank}.json')
    try:
        if world!=2 or rank not in (0,1) or local_rank!=rank:
            raise ValueError('requires single-node torchrun --nproc_per_node=2')
        if not output.is_relative_to(Path('/shared/output/DITSERVE')):
            raise ValueError('outputs must stay under /shared/output/DITSERVE')
        if any(name.startswith('SYMM_COLL_') for name in os.environ):
            raise ValueError('uncontrolled GFC environment override')
        for name,value in {'VLLM_RUNTIME_V2_MIGRATE_GFC_P2P':'1','VLLM_RUNTIME_V2_MIGRATE_TRACE':'0',
                           'VLLM_RUNTIME_V2_MIGRATE_NSYS':'0'}.items():
            if name in os.environ and os.environ[name]!=value: raise ValueError('uncontrolled migration option '+name)
            os.environ[name]=value
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        migration.gfc_check.storage_guard()
        threading.Thread(target=storage_watchdog,args=(stop,),daemon=True).start()
        gfc_source=migration.gfc_check.verify_sources(args.gfc_source_root,args.gfc_source_manifest,
            args.gfc_source_manifest_sha256,args.shared_python_root)
        migration.upstream.verify_source(args.source_root,args.source_manifest,args.source_manifest_sha256)
        adapter.verify_shared_sources(args.shared_python_root)
        if sha(args.protocol.read_bytes())!=args.protocol_sha256:
            raise ValueError('formal protocol SHA256 differs')
        request=request_plan(migration.upstream.read_json(args.protocol.read_bytes()),args.prompt,args.seed)
        pure_request(asdict(request))
        sys.path.insert(0,str(args.shared_python_root.resolve()))
        import torch
        import torch.distributed as dist
        import diffusers
        import gfc
        if torch.cuda.device_count()!=2 or diffusers.__version__!='0.36.0':
            raise ValueError('requires fixed-image diffusers 0.36.0 and two actual GPUs')
        if Path(gfc.__file__).resolve()!=args.shared_python_root.resolve()/'gfc/__init__.py':
            raise ValueError('actual GFC source differs')
        torch.cuda.set_device(rank)
        device=torch.device('cuda',rank)
        if not torch.cuda.can_device_access_peer(rank,1-rank): raise ValueError('peer access required; no fallback')
        record.update(gpu_uuid=str(torch.cuda.get_device_properties(rank).uuid),torch=torch.__version__,
            cuda=torch.version.cuda,diffusers=diffusers.__version__,request=asdict(request))
        if rank==0:
            migration.gfc_check.storage_guard()
            output.mkdir(parents=True,exist_ok=False)
            derived=migration.apply(args.source_root,args.source_manifest,args.source_manifest_sha256,output/'derived')
        dist.init_process_group('nccl',device_id=device,timeout=timedelta(seconds=600))
        bound=[derived if rank==0 else None]
        dist.broadcast_object_list(bound,src=0)
        derived=bound[0]
        owned=True
        control=migration.load(derived['root'],derived['manifest_sha256'],gpu=True)
        binding=dict(gfdit_manifest_sha256=args.source_manifest_sha256,gfc=gfc_source,
            derived_manifest_sha256=derived['manifest_sha256'],model_manifest_sha256=args.model_manifest_sha256,
            protocol_sha256=args.protocol_sha256,shared_worker_sha256=adapter.FROZEN_WORKER_SHA256,
            model_checker_sha256=sha(Path(__file__).read_bytes()),migration_sha256=sha(Path(migration.__file__).read_bytes()),
            round_sha256=sha(Path(adapter.__file__).read_bytes()))
        record.update(source=binding,source_binding_sha256=sha(migration.encoded(binding)),stage='load_real_model')
        checkpoint()
        owned_groups=model_groups(dist,rank)
        pipeline=adapter.SD35RoundPipeline.from_pretrained(args.model_root,shared_python_root=args.shared_python_root,
            model_manifest_path=args.model_manifest,model_manifest_sha256=args.model_manifest_sha256,
            device=device,groups=owned_groups,attention='math',guidance_scale=4.5,max_sequence_length=256)
        pipeline.runtime.pool.activate((rank,),pipeline.runtime.state)
        if pipeline.runtime.pool.rank!=rank or pipeline.runtime.pool.current_ranks!=(rank,):
            raise ValueError('model SP1 group does not bind the actual rank')
        for group in (pipeline.runtime.pool._device_group((rank,)),pipeline.runtime.pool._cpu_group((rank,))):
            if group is None or tuple(dist.get_process_group_ranks(group))!=(rank,):
                raise ValueError('model singleton is not a real process group')
        record.update(model_weights_loaded=True,model_provenance=pipeline.provenance,
            model_sp_ranks=list(pipeline.runtime.pool.current_ranks),model_pool_rank=pipeline.runtime.pool.rank,
            backend=pipeline.runtime.info(),stage='source_first_two_steps')
        checkpoint()
        dist.barrier()
        with migration.forbid_payload_fallback(dist):
            runtime=control.collective_runtime.init_runtime_v2_collective_runtime(
                backend='gfc',device=device,max_group_size=2,max_collective_bytes=migration.BUFFER_BYTES)
            gfc_runtime=control.collective_runtime.get_gfc_migrate_runtime()
            if runtime is gfc_runtime or runtime.stream==gfc_runtime.stream:
                raise ValueError('original migration requires an independent GFC runtime/stream')
            if not control.data_plane._MIGRATE_GFC_P2P or not control.collective_runtime.runtime_v2_uses_gfc():
                raise ValueError('actual data plane did not select GFC')
            record['gfc_config']=asdict(gfc_runtime.config)
            if (gfc_runtime.config.max_collective_bytes!=migration.BUFFER_BYTES or gfc_runtime.config.use_tma
                    or gfc_runtime.config.use_copy_engine or gfc_runtime.config.enable_fused_path):
                raise ValueError('unexpected actual GFC configuration')
            migration.instrument_p2p(gfc_runtime,record['p2p'])
            descriptor=[None]
            if rank==0:
                initial=pipeline.prepare(request)
                prefix,calls=count_transformer_calls(pipeline,2,lambda:pipeline.advance(initial,2))
                value=payload_from_state(prefix)
                descriptor[0]=describe_payload(value,request,device)
                record.update(source_transformer_calls=calls,source_interval=[0,2])
            dist.broadcast_object_list(descriptor,src=0)  # CPU shapes/hashes/request only.
            descriptor=validate_descriptor(descriptor[0],request)
            specs,plan=build_plan(control,descriptor)
            worker=migration.make_worker(control,rank,specs,make_codec(control,request))
            if rank==0:
                source=plan.artifacts[0].effective_source_handle
                worker.local_artifacts[(REQUEST_ID,plan.src_group_id,source.artifact_id)]=control.protocol.ArtifactValue(source,value)
            record.update(stage='native_migrate_real_five_tensors',descriptor=descriptor)
            checkpoint()
            schedule,_=migration.migrate_plan(control,worker,plan,specs,MID,dist,rank,record)
            record.update(migrations=1,expected_p2p=migration.edge_counts(schedule,rank))
            if record['p2p']!=record['expected_p2p'] or record['p2p']!=expected_p2p(descriptor,rank):
                raise ValueError('actual GFC did not transport the five tensors exactly')
            if rank==1:
                destination=plan.artifacts[0].handle
                value=worker.local_artifacts[(REQUEST_ID,plan.dst_group_id,destination.artifact_id)].value
                restored=restore_state(value,request,descriptor,device)
                record.update(received_tensor_hashes_match=True,stage='resume_remaining_26_steps')
                checkpoint()
                # New adapter object on rank 1: the only request state received
                # from rank 0 is the five tensors plus explicit CPU metadata.
                resumed_pipeline=adapter.SD35RoundPipeline(pipeline.runtime,guidance_scale=4.5,
                    max_sequence_length=256,provenance=pipeline.provenance)
                resumed,calls=count_transformer_calls(resumed_pipeline,26,lambda:resumed_pipeline.advance(restored,28))
                actual_latents=resumed.latents
                actual_image=decode(resumed_pipeline,actual_latents)
                record.update(resume_transformer_calls=calls,resume_interval=[2,28],stage='independent_full_28_step_reference')
                checkpoint()
                def reference_call():
                    with torch.inference_mode():
                        return pipeline.pipe(prompt=request.prompt,negative_prompt=request.negative_prompt,
                            height=request.height,width=request.width,num_inference_steps=28,guidance_scale=4.5,
                            max_sequence_length=256,generator=torch.Generator(device='cpu').manual_seed(request.seed),
                            output_type='latent').images
                reference_latents,calls=count_transformer_calls(pipeline,28,reference_call)
                reference_image=decode(pipeline,reference_latents)
                record.update(reference_transformer_calls=calls,raw_decodes_finite=True,
                    comparisons=dict(latents=compare_tensors(reference_latents,actual_latents),
                                     images=compare_tensors(reference_image,actual_image)),
                    outputs=dict(reference_latents=tensor_summary(reference_latents),actual_latents=tensor_summary(actual_latents),
                                 reference_image=tensor_summary(reference_image),actual_image=tensor_summary(actual_image)))
                if not all(item['passed'] for item in record['comparisons'].values()):
                    raise ValueError('actual cross-rank continuation differs from the full shared reference')
            dist.barrier()
            worker.local_artifacts.clear()
            if worker._migration_schedules or worker.results:
                raise ValueError('native worker migration lifecycle did not settle')
            control.collective_runtime.shutdown_runtime_v2_collective_runtime()
        pipeline.runtime.close()
        owned_groups.close()
        torch.cuda.synchronize()
        adapter.verify_shared_sources(args.shared_python_root)
        migration.load(derived['root'],derived['manifest_sha256'],gpu=True)
        if migration.gfc_check.verify_sources(args.gfc_source_root,args.gfc_source_manifest,
                args.gfc_source_manifest_sha256,args.shared_python_root)!=gfc_source:
            raise ValueError('GFC source changed during model migration')
        migration.upstream.verify_source(args.source_root,args.source_manifest,args.source_manifest_sha256)
        record.update(status='passed',stage='completed',cleanup_verified=True)
        checkpoint()
        records=[None,None]
        dist.all_gather_object(records,record)
        summary=assess(records,descriptor)
        if rank==0:
            migration.gfc_check.storage_guard()
            with (output/'summary.json').open('xb') as stream: stream.write(migration.encoded(summary))
        dist.barrier()
        dist.destroy_process_group()
        stop.set()
        print(json.dumps(dict(rank=rank,status='passed',qualification_eligible=False)))
        return 0
    except Exception as exc:
        record.update(status='failed',error=repr(exc),traceback=traceback.format_exc()[-16000:],cleanup_verified=False)
        if owned and not isinstance(exc,migration.gfc_check.StorageStop):
            try: checkpoint()
            except migration.gfc_check.StorageStop: pass
        stop.set()
        print(json.dumps(record,sort_keys=True,allow_nan=False),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
