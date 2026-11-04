"""CPU contracts for candidate-only frozen layer wrapping and weight preservation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import external_sd35_parallel_wrapper_check as candidate


class Tensor:
    def __init__(self):self.dtype='torch.bfloat16';self.shape=(2,2);self._version=0
    def data_ptr(self):return id(self)+8
    def storage_offset(self):return 0


class Hook:
    def __init__(self,items,call):self.items=items;self.call=call
    def remove(self):self.items.remove(self.call)


class Module:
    def __init__(self):self.pre=[]
    def register_forward_pre_hook(self,fn):self.pre.append(fn);return Hook(self.pre,fn)
    def __call__(self,value):
        for fn in self.pre:fn(self,(value,))
        return value


class RawProcessor:pass
class Processor:pass
class RawAttention(Module):
    def __init__(self):super().__init__();self.processor=RawProcessor();self.weight=Tensor()
class RawConv(Module):
    def __init__(self):super().__init__();self.weight=Tensor()
class RawPatch(Module):
    def __init__(self):super().__init__();self.proj=RawConv();self.position=Tensor()
class Wrapped(Module):
    def __init__(self,module):super().__init__();self.module=module
    def __getattr__(self,name):return getattr(self.module,name)
class Attention(Wrapped):
    def __init__(self,module):super().__init__(module);self.processor=Processor()
class Conv(Wrapped):pass
class Patch(Wrapped):pass

for cls,path in ((Attention,candidate.ATTENTION_TYPE),(Patch,candidate.PATCH_TYPE),
                 (Conv,candidate.CONV_TYPE),(Processor,candidate.PROCESSOR_TYPE)):
    cls.__module__,cls.__qualname__=path.rsplit('.',1)


class Transformer:
    def __init__(self):
        self.module=object();self.wrapped_layers=[];self.wrap_calls=0;self.cache_calls=0;self.change_weight=False
        self.transformer_blocks=[SimpleNamespace(attn=RawAttention(),**({'attn2':RawAttention()} if i<13 else {})) for i in range(24)]
        self.pos_embed=RawPatch()
    def _wrap_layers(self,*,model,submodule_classes_to_wrap,submodule_name_to_wrap,submodule_addition_args):
        assert model is self.module
        assert submodule_classes_to_wrap==[RawConv,RawPatch]
        assert submodule_name_to_wrap==['attn','attn2']
        assert submodule_addition_args=={}
        self.wrap_calls+=1
        for block in self.transformer_blocks:
            for name in ('attn','attn2'):
                if hasattr(block,name):setattr(block,name,Attention(getattr(block,name)))
        self.pos_embed=Patch(self.pos_embed);self.pos_embed.module.proj=Conv(self.pos_embed.module.proj)
        self.wrapped_layers=[m for _,m in candidate.modules(self)]
        if self.change_weight:self.transformer_blocks[0].attn.module.weight._version+=1
        return model
    def _register_cache(self):self.cache_calls+=1
    def named_parameters(self):
        result=[]
        for name,module in candidate.modules(self):
            if name=='pos_embed':continue
            wrapped=isinstance(module,Wrapped);raw=module.module if wrapped else module
            result.append((name+('.module' if wrapped else '')+'.weight',raw.weight))
        return result
    def state_dict(self,*,keep_vars):
        assert keep_vars
        result=dict(self.named_parameters());wrapped=isinstance(self.pos_embed,Wrapped)
        raw=self.pos_embed.module if wrapped else self.pos_embed
        result['pos_embed'+('.module' if wrapped else '')+'.position']=raw.position
        return result


def types():
    return dict(transformer=Transformer,attention=Attention,processor=Processor,patch=Patch,conv=Conv,
                raw_attention=RawAttention,raw_processor=RawProcessor,raw_patch=RawPatch,raw_conv=RawConv)


def sources(value):
    if getattr(value,'__name__',None)=='_wrap_layers':key='wrap'
    elif getattr(value,'__name__',None)=='_register_cache':key='cache'
    else:key=next(k for k in ('transformer','attention','patch','conv') if types()[k] is value)
    return dict(file='/frozen/'+key+'.py',file_sha256=candidate.SOURCE_HASHES[key])


def runtime():
    return SimpleNamespace(parallel_config=SimpleNamespace(pp_degree=1,cfg_degree=1,dp_degree=1,sp_degree=2),
                           patch_mode=False,attention_backend=SimpleNamespace(name='SDPA_MATH'))


def convert(model):return candidate.convert(model,runtime(),types=types(),describe_source=sources)


class ParallelWrapperTests(unittest.TestCase):
    def test_wraps_exact_original_39_objects_once_without_replacing_parameters(self):
        model=Transformer();original=candidate.modules(model);before=candidate.identities(model)
        receipt=convert(model)
        self.assertEqual((model.wrap_calls,model.cache_calls),(1,1))
        self.assertEqual(len(model.wrapped_layers),39)
        for (name,old),(new_name,new) in zip(original,candidate.modules(model)):
            self.assertEqual(name,new_name);self.assertIs(new.module,old)
        self.assertTrue(candidate.preserved(before,candidate.identities(model)))
        self.assertTrue(candidate.valid_conversion(receipt))
        self.assertFalse(receipt['weights']['values_rehashed'])
        self.assertNotEqual(receipt['weights']['parameters']['before_keys_sha256'],receipt['weights']['parameters']['after_keys_sha256'])

    def test_second_conversion_is_rejected_before_mutating(self):
        model=Transformer();convert(model)
        with self.assertRaisesRegex(ValueError,'converted once'):convert(model)
        self.assertEqual((model.wrap_calls,model.cache_calls),(1,1))

    def test_source_runtime_and_missing_attn2_fail_before_wrap(self):
        for failure in ('source','sp','patch','backend','attn2'):
            model=Transformer();state=runtime();source=sources
            if failure=='source':source=lambda value:dict(file='/other',file_sha256='0'*64)
            if failure=='sp':state.parallel_config.sp_degree=1
            if failure=='patch':state.patch_mode=True
            if failure=='backend':state.attention_backend.name='SDPA_FLASH'
            if failure=='attn2':del model.transformer_blocks[0].attn2
            with self.assertRaises(ValueError,msg=failure):candidate.convert(model,state,types=types(),describe_source=source)
            self.assertEqual(model.wrap_calls,0)

    def test_write_to_existing_parameter_is_detected_without_claiming_rollback(self):
        model=Transformer();model.change_weight=True
        with self.assertRaisesRegex(ValueError,'identity/version/storage'):convert(model)
        self.assertEqual(model.wrap_calls,1)
        self.assertEqual(len(model.wrapped_layers),39) # Mutation is confined to this diagnostic model.

    def test_identity_check_detects_replaced_storage_dtype_shape_offset_version_and_missing_key(self):
        before=candidate.identities(Transformer())
        for position,value in enumerate((1,2,'torch.float32',(1,4),3,4)):
            after=deepcopy(before);key=next(iter(after['parameters']));identity=list(after['parameters'][key]);identity[position]=value;after['parameters'][key]=tuple(identity)
            self.assertFalse(candidate.preserved(before,after))
        after=deepcopy(before);del after['state_dict'][next(iter(after['state_dict']))]
        self.assertFalse(candidate.preserved(before,after))

    def test_every_wrapper_call_is_observed_and_hooks_restore_on_failure(self):
        model=Transformer();convert(model);sentinel=object()
        with candidate.observe_modules(model) as counts:
            for _,module in candidate.modules(model):self.assertIs(module(sentinel),sentinel)
        self.assertEqual(len(counts),39);self.assertTrue(all(row['calls']==1 for row in counts))
        self.assertTrue(all(not m.pre for _,m in candidate.modules(model)))
        with self.assertRaisesRegex(RuntimeError,'actual forward'):
            with candidate.observe_modules(model):raise RuntimeError('actual forward')
        self.assertTrue(all(not m.pre for _,m in candidate.modules(model)))
        with self.assertRaisesRegex(ValueError,'execute once'):
            with candidate.observe_modules(model):pass

    def test_persisted_conversion_rejects_missing_layers_sources_and_weight_drift(self):
        good=convert(Transformer());self.assertTrue(candidate.valid_conversion(good))
        for failure in ('source','weight','type','processor','attn2','count','qual'):
            row=deepcopy(good)
            if failure=='source':row['sources']['wrap']['file_sha256']='0'*64
            if failure=='weight':row['weights']['parameters']['after_identity_sha256']='0'*64
            if failure=='type':row['after'][0]['type']='diffusers.Attention'
            if failure=='processor':row['after'][0]['processor_type']='diffusers.JointAttnProcessor2_0'
            if failure=='attn2':row['after'][1]['name']='transformer_blocks.0.attn'
            if failure=='count':row['wrapped_layers_after']=38
            if failure=='qual':row['qualification_eligible']=True
            self.assertFalse(candidate.valid_conversion(row),failure)

    def test_only_valid_default_key_set_order_is_normalized(self):
        route=dict(metadata=dict(scheduler=dict(config=dict(_use_default_values=['shift','invert_sigmas'],shift=3.))),
                   tensors={'output':{'sha256':'a'*64}},runtime={'sp':2})
        rows=[dict(rank=i,routes={'sp2_round':deepcopy(route)}) for i in (0,1)]
        reference=dict(ranks=deepcopy(rows))
        rows[0]['routes']['sp2_round']['metadata']['scheduler']['config']['_use_default_values'].reverse()
        self.assertTrue(candidate.original_reproduced(rows,reference))
        for change in ('duplicate','nonstring','tuple','realconfig','tensor','runtime'):
            bad=deepcopy(rows);value=bad[0]['routes']['sp2_round'];config=value['metadata']['scheduler']['config']
            if change=='duplicate':config['_use_default_values']=['shift','shift']
            if change=='nonstring':config['_use_default_values']=['shift',1]
            if change=='tuple':config['_use_default_values']=('shift','invert_sigmas')
            if change=='realconfig':config['shift']=4.
            if change=='tensor':value['tensors']['output']['sha256']='b'*64
            if change=='runtime':value['runtime']['sp']=1
            self.assertFalse(candidate.original_reproduced(bad,reference),change)

    def test_original_r7_reproduction_does_not_compare_candidate_with_old_wrong_output(self):
        rows=[dict(rank=i,routes={'sp2_round':{'output':'original'}},parallel_wrapper_candidate={'output':'changed'}) for i in (0,1)]
        reference=dict(ranks=[dict(rank=i,routes={'sp2_round':{'output':'original'}}) for i in (0,1)])
        self.assertTrue(candidate.original_reproduced(rows,reference))
        rows[1]['parallel_wrapper_candidate']['output']='another candidate'
        self.assertTrue(candidate.original_reproduced(rows,reference))
        rows[1]['routes']['sp2_round']['output']='observer drift'
        self.assertFalse(candidate.original_reproduced(rows,reference))


if __name__=='__main__':unittest.main()
