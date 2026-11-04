"""Experimental in-process completion of frozen SD3 parallel layer wrapping.

The conversion mutates only this temporary diagnostic model. It is intentionally
not rolled back, and is neither a production patch nor full trajectory evidence.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from contextlib import contextmanager, ExitStack
import hashlib
import inspect
import json
import re
from pathlib import Path

import external_sd35_trace as trace

R7_SHA256 = '8cbf276c618e95bb0139b9e7e39e48d816930d1569055782cd5634333c7fc3d1'
SOURCE_HASHES = dict(
    wrap='7d6439b3bc90bec49c29a1339603b7e4beee962cb0b6689515215ef772003ad8',
    cache='7d6439b3bc90bec49c29a1339603b7e4beee962cb0b6689515215ef772003ad8',
    transformer='9637fc6061b0384e0322a0eacbbbe62252bb199c3728cc04f719f1183f058ad1',
    attention='5497073287924ede46aa207a131978c165394c4b579628f297ef57bf9e8c52bf',
    patch='0d0ae74c2b628470647c97d8dde69852ac732b6ce8d35bef2d5e245a43be4812',
    conv='3bf3a621274fa2b7cfc37c1c24a6c41405a76e69cba9213558f385af2e703900')
ATTENTION_TYPE='xfuser.model_executor.layers.attention_processor.xFuserAttentionWrapper'
PROCESSOR_TYPE='xfuser.model_executor.layers.attention_processor.xFuserJointAttnProcessor2_0'
PATCH_TYPE='xfuser.model_executor.layers.embeddings.xFuserPatchEmbedWrapper'
CONV_TYPE='xfuser.model_executor.layers.conv.xFuserConv2dWrapper'


def qualified(value):
    cls = value if isinstance(value,type) else type(value)
    return cls.__module__+'.'+cls.__qualname__


def digest(value):
    return hashlib.sha256(trace.encoded(value)).hexdigest()


def source(value):
    path=Path(inspect.getsourcefile(value)).resolve(strict=True)
    return dict(file=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def factories():
    import torch
    from diffusers.models.embeddings import PatchEmbed
    from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
    from xfuser.model_executor.models.transformers.transformer_sd3 import xFuserSD3Transformer2DWrapper
    from xfuser.model_executor.layers.attention_processor import xFuserAttentionWrapper, xFuserJointAttnProcessor2_0
    from xfuser.model_executor.layers.embeddings import xFuserPatchEmbedWrapper
    from xfuser.model_executor.layers.conv import xFuserConv2dWrapper
    return dict(transformer=xFuserSD3Transformer2DWrapper,attention=xFuserAttentionWrapper,
        processor=xFuserJointAttnProcessor2_0,patch=xFuserPatchEmbedWrapper,conv=xFuserConv2dWrapper,
        raw_attention=Attention,raw_processor=JointAttnProcessor2_0,raw_patch=PatchEmbed,raw_conv=torch.nn.Conv2d)


def tensor_identity(value):
    return (id(value),value.data_ptr(),str(value.dtype),tuple(value.shape),value.storage_offset(),value._version)


def identities(transformer):
    # No tensor values are copied/hashed. keep_vars preserves parameter/buffer identity.
    parameters={name:tensor_identity(tensor) for name,tensor in transformer.named_parameters()}
    state={name:tensor_identity(tensor) for name,tensor in transformer.state_dict(keep_vars=True).items()}
    return dict(parameters=parameters,state_dict=state)


def preserved(before, after):
    return all(Counter(before[key].values())==Counter(after[key].values()) for key in ('parameters','state_dict'))


def identity_receipt(before,after):
    if not preserved(before,after):
        raise ValueError('parallel wrapping changed parameter or state tensor identity/version/storage')
    result=dict(preserved=True,values_rehashed=False,
        scope='tensor object, data_ptr, dtype, shape, storage_offset and version; not a new weight-content SHA256')
    for key in ('parameters','state_dict'):
        reverse={value:[] for value in after[key].values()}
        for name,value in after[key].items():reverse[value].append(name)
        mapping={name:sorted(reverse[value]) for name,value in before[key].items()}
        result[key]=dict(count=len(before[key]),before_keys_sha256=digest(sorted(before[key])),
            after_keys_sha256=digest(sorted(after[key])),key_mapping_sha256=digest(mapping),
            before_identity_sha256=digest(sorted(before[key].values())),
            after_identity_sha256=digest(sorted(after[key].values())))
    return result


def modules(transformer):
    result=[]
    for index,block in enumerate(transformer.transformer_blocks):
        for name in ('attn','attn2'):
            value=getattr(block,name,None)
            if value is not None:result.append(('transformer_blocks.'+str(index)+'.'+name,value))
    result.append(('pos_embed',transformer.pos_embed))
    result.append(('pos_embed.proj',transformer.pos_embed.proj))
    return result


def inventory(transformer):
    return [dict(name=name,type=qualified(module),processor_type=qualified(module.processor) if hasattr(module,'processor') else None)
            for name,module in modules(transformer)]


def convert(transformer,runtime, *, types=None, describe_source=source):
    types=factories() if types is None else types
    parallel=runtime.parallel_config
    if (any(type(getattr(parallel,k)) is not int or getattr(parallel,k)!=v
            for k,v in dict(pp_degree=1,cfg_degree=1,dp_degree=1,sp_degree=2).items())
            or runtime.patch_mode is not False or runtime.attention_backend.name!='SDPA_MATH'):
        raise ValueError('conversion requires actual SP2/PP1/CFG1/DP1, no patches, configured math')
    if type(transformer) is not types['transformer'] or transformer.wrapped_layers!=[]:
        raise ValueError('only the original unwrapped frozen SD3 model may be converted once')
    targets=dict(wrap=transformer._wrap_layers,cache=transformer._register_cache,
                 **{key:types[key] for key in ('transformer','attention','patch','conv')})
    sources={key:describe_source(value) for key,value in targets.items()}
    if any(sources[key]['file_sha256']!=expected for key,expected in SOURCE_HASHES.items()):
        raise ValueError('parallel conversion implementation differs from frozen 51740c0')
    old=modules(transformer)
    attention=[(name,module) for name,module in old if name.endswith(('.attn','.attn2'))]
    if (len(attention)!=37 or sum(name.endswith('.attn2') for name,_ in attention)!=13
            or any(type(m) is not types['raw_attention'] or type(m.processor) is not types['raw_processor'] for _,m in attention)
            or type(transformer.pos_embed) is not types['raw_patch']
            or type(transformer.pos_embed.proj) is not types['raw_conv']):
        raise ValueError('the complete original 37-attention/patch inventory is required')
    before=identities(transformer); original=transformer.module
    receipt=dict(recipe='complete frozen SD3 _wrap_layers + _register_cache under active SP2',
        scope='irreversible object mutation confined to this temporary diagnostic process',
        production_default_changed=False,qualification_eligible=False,sources=sources,
        before=inventory(transformer),wrapped_layers_before=0,complete=False)
    converted=transformer._wrap_layers(model=original,
        submodule_classes_to_wrap=[types['raw_conv'],types['raw_patch']],
        submodule_name_to_wrap=['attn','attn2'],submodule_addition_args={})
    if converted is not original or transformer.module is not original:
        raise ValueError('frozen wrapping must preserve the original raw transformer object')
    transformer._register_cache()
    actual=modules(transformer)
    if [n for n,_ in actual]!=[n for n,_ in old]:
        raise ValueError('conversion changed the module inventory')
    for (name,raw),(_,wrapped) in zip(old,actual):
        kind='attention' if name.endswith(('.attn','.attn2')) else 'patch' if name=='pos_embed' else 'conv'
        if type(wrapped) is not types[kind] or wrapped.module is not raw:
            raise ValueError('conversion did not wrap the exact original object: '+name)
        if kind=='attention' and type(wrapped.processor) is not types['processor']:
            raise ValueError('actual parallel attention processor differs')
    if len(transformer.wrapped_layers)!=39 or {id(m) for m in transformer.wrapped_layers}!={id(m) for _,m in actual}:
        raise ValueError('actual registered parallel layer inventory differs')
    receipt.update(after=inventory(transformer),wrapped_layers_after=39,
                   weights=identity_receipt(before,identities(transformer)),complete=True)
    return receipt


@contextmanager
def observe_modules(transformer):
    rows=[]
    with ExitStack() as stack:
        for name,module in modules(transformer):
            row=dict(name=name,type=qualified(module),calls=0)
            def count(*args,_row=row):_row['calls']+=1
            handle=module.register_forward_pre_hook(count);stack.callback(handle.remove);rows.append(row)
        yield rows
    if len(rows)!=39 or any(row['calls']!=1 for row in rows):
        raise ValueError('every converted attention/patch/conv must execute once')


def backend_metadata(runtime):
    import torch
    return dict(configured=runtime.attention_backend.name,torch_flash_sdp_enabled=torch.backends.cuda.flash_sdp_enabled(),
        torch_math_sdp_enabled=torch.backends.cuda.math_sdp_enabled(),
        torch_mem_efficient_sdp_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
        native_kernel_identity='unobserved PyTorch SDPA dispatcher; configuration alone is not kernel proof',
        converted_dispatch_inference='frozen USP SDPA_MATH -> aten._scaled_dot_product_attention_math',
        kernel_dispatch_verified=False)


def bind_reference(path, binding, references, cross):
    raw=trace.bounded_read(path,256*1024)
    if hashlib.sha256(raw).hexdigest()!=R7_SHA256:raise ValueError('candidate requires immutable r7 child SHA256')
    row=json.loads(raw)
    if (row.get('binding')!=binding or row.get('diagnostic_completed') is not True
            or row.get('qualification_eligible') is not False or not cross.assess(row.get('ranks',[]),references)):
        raise ValueError('candidate reference must be the complete source-bound r7 observation')
    return row


def normalized_routes(routes):
    # Diffusers serializes this default-key set as a hash-seed-dependent list.
    # Normalize only that known non-semantic set; every other field stays exact.
    result=deepcopy(routes)
    for route in result.values():
        config=route.get('metadata',{}).get('scheduler',{}).get('config',{})
        if '_use_default_values' in config:
            keys=config['_use_default_values']
            if not isinstance(keys,list) or any(not isinstance(k,str) for k in keys) or len(set(keys))!=len(keys):
                raise ValueError('scheduler default-key metadata must be a unique string list')
            config['_use_default_values']=sorted(keys)
    return result


def original_reproduced(rows,reference):
    try:
        return (len(rows)==2 and {r.get('rank') for r in rows}=={0,1}
            and all(normalized_routes(row['routes'])==normalized_routes(next(r['routes'] for r in reference['ranks'] if r['rank']==row['rank']))
                    for row in rows))
    except (KeyError,TypeError,ValueError,StopIteration):return False


def valid_weights(weights):
    try:
        if weights['preserved'] is not True or weights['values_rehashed'] is not False:return False
        for key in ('parameters','state_dict'):
            value=weights[key]
            if type(value['count']) is not int or value['count']<1:return False
            fields=('before_keys_sha256','after_keys_sha256','key_mapping_sha256','before_identity_sha256','after_identity_sha256')
            if any(not isinstance(value[field],str) or not re.fullmatch('[0-9a-f]{64}',value[field]) for field in fields):return False
            if value['before_identity_sha256']!=value['after_identity_sha256']:return False
        return True
    except (KeyError,TypeError):return False


def valid_conversion(receipt):
    try:
        if (receipt['complete'] is not True or receipt['production_default_changed'] is not False
                or receipt['qualification_eligible'] is not False or receipt['wrapped_layers_before']!=0
                or receipt['wrapped_layers_after']!=39
                or any(receipt['sources'][key]['file_sha256']!=value for key,value in SOURCE_HASHES.items())
                or not valid_weights(receipt['weights'])):return False
        for key in ('parameters','state_dict'):
            value=receipt['weights'][key]
            if (type(value['count']) is not int or value['count']<1
                    or value['before_identity_sha256']!=value['after_identity_sha256']):return False
        before,after=receipt['before'],receipt['after']
        if len(before)!=39 or len(after)!=39 or [v['name'] for v in before]!=[v['name'] for v in after]:return False
        if len({v['name'] for v in after})!=39:return False
        for row in after:
            expected=ATTENTION_TYPE if row['name'].endswith(('.attn','.attn2')) else PATCH_TYPE if row['name']=='pos_embed' else CONV_TYPE if row['name']=='pos_embed.proj' else None
            if row['type']!=expected or (expected==ATTENTION_TYPE and row['processor_type']!=PROCESSOR_TYPE):return False
        return sum(v['name'].endswith('.attn2') for v in after)==13
    except (KeyError,TypeError):return False


def assess(rows,reference,cross):
    result=dict(candidate_diagnostic_completed=False,candidate_equivalence_passed=False,
        original_observer_reproduced=original_reproduced(rows,reference),qualification_eligible=False)
    try:
        if not result['original_observer_reproduced']:return result
        for row in rows:
            candidate=row['parallel_wrapper_candidate'];rank=row['rank']
            if (not valid_conversion(candidate['conversion']) or candidate['backend']['configured']!='SDPA_MATH'
                    or not valid_weights(candidate['post_inference_weights'])):return result
            for key in ('parameters','state_dict'):
                if candidate['post_inference_weights'][key]['before_identity_sha256']!=candidate['conversion']['weights'][key]['after_identity_sha256']:return result
            cross.validate_partition(candidate['row_partition'])
            expected={'sp1_round','sp2_round'} if rank==0 else {'sp2_round'}
            if set(candidate['routes'])!=expected:return result
            for name,route in candidate['routes'].items():
                if (not cross.valid_runtime(route['metadata']['runtime'],name)
                        or route['metadata']['scheduler']['bound_step']['file_sha256']!=trace.RAW_SOLVER_SHA256
                        or route['counts']!=dict(transformer_enter=1,transformer_return=1,solver_enter=1,solver_return=1,naive=0,sync=1,async_=0)
                        or any(type(v) is not int for v in route['counts'].values())
                        or route['route']!=name or route['logical_step']!=0 or route['original_schedule_steps']!=28
                        or route['intentional_native_stop'] is not False or route['arrays_retained'] is not False
                        or route['hooks_restored'] is not True
                        or set(route['tensors'])!=set(cross.BOUNDARIES)|{'timesteps','sigmas'}
                        or not all(trace.valid_tensor(v) for v in route['tensors'].values())
                        or set(route['actual_inputs_match_initial'])!=set(cross.INPUTS)|{'scheduler_sample','scheduler_timestep','timesteps','sigmas'}
                        or not all(v is True for v in route['actual_inputs_match_initial'].values())
                        or route['tensors']['timesteps']['shape']!=[1] or route['tensors']['sigmas']['shape']!=[2]):return result
                cross.validate_rows(route['actual_row_ranges'],128,route['local_height'])
                if route['actual_row_ranges']!=route['metadata']['runtime']['pp_patches_start_end_idx_global']:return result
                if name=='sp1_round' and route['actual_row_ranges']!=[[0,128]]:return result
                calls=candidate['module_calls'][name]
                if len(calls)!=39 or {r['name'] for r in calls}!={r['name'] for r in candidate['conversion']['after']}:return result
                if any(type(r['calls']) is not int or r['calls']!=1 for r in calls):return result
                if any(r['type']!=next(m['type'] for m in candidate['conversion']['after'] if m['name']==r['name']) for r in calls):return result
                if len(route['processors']['modules'])!=37 or any(x['processor_type']!=PROCESSOR_TYPE or x['calls']!=1 for x in route['processors']['modules']):return result
            for name in ('original_sp1_vs_converted_sp2','converted_sp1_vs_converted_sp2'):
                pair=candidate[name]
                if not cross.valid_pair(pair) or any(m['actual']!=candidate['routes']['sp2_round']['tensors'][k] for k,m in pair['boundaries'].items()):return result
            if rank==0:
                pair=candidate['original_sp1_vs_converted_sp1']
                if not cross.valid_pair(pair) or any(m['reference']!=row['routes']['sp1_native']['tensors'][k]
                    or m['actual']!=candidate['routes']['sp1_round']['tensors'][k] for k,m in pair['boundaries'].items()):return result
        result['candidate_diagnostic_completed']=True
        result['candidate_equivalence_passed']=all(all(m['passed'] for m in row['parallel_wrapper_candidate']['converted_sp1_vs_converted_sp2']['boundaries'].values()) for row in rows)
        return result
    except (KeyError,TypeError,ValueError):return result
