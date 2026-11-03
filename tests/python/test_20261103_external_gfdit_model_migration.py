"""Continuation integrity and evidence gates; actual inference is GPU-only."""
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import external_gfdit_model_migration as check

SOURCE=Path(__file__).resolve().parents[3]/'.local/ditserve-kairos-rewrite-runtime/external-systems/20261103/GF-DiT'
MANIFEST=SOURCE.with_suffix('.source.json')
MANIFEST_SHA='e1b54cf09241ff70fc27e9d1666a44cbe38e00eaf1713b57cd5c9ade9c9db3f4'


def request():
    return check.adapter.ImageRequest('A fox.',negative_prompt='blurry, low quality')


def descriptor():
    # This is only a CPU schema fixture, never a claimed GPU tensor or output.
    shapes={'prompt_embeds':[2,333,4096],'pooled_prompt_embeds':[2,2048],
            'latents':[1,16,64,64],'timesteps':[28],'sigmas':[29]}
    return dict(schema=check.STATE_SCHEMA,request=asdict(request()),cursor=2,
        tensor_fields={name:dict(shape=shape,dtype='torch.float32' if name in ('timesteps','sigmas') else 'torch.bfloat16',
            bytes=math.prod(shape)*(4 if name in ('timesteps','sigmas') else 2),sha256=str(index)*64)
            for index,(name,shape) in enumerate(shapes.items())})


def records():
    value=descriptor()
    result=[]
    for rank in (0,1):
        row=dict(rank=rank,status='passed',cleanup_verified=True,model_weights_loaded=True,gpu_uuid='unit-'+str(rank),
            source_binding_sha256='unit-source-binding',phase_executions=4 if rank==0 else 3,migrations=1,
            phase_trace=check.expected_trace() if rank==0 else [],descriptor=deepcopy(value),
            p2p=check.expected_p2p(value,rank),expected_p2p=check.expected_p2p(value,rank),
            model_sp_ranks=[rank],model_pool_rank=rank)
        if rank==0:
            row.update(source_transformer_calls=2,source_interval=[0,2])
        else:
            row.update(resume_transformer_calls=26,reference_transformer_calls=28,resume_interval=[2,28],
                received_tensor_hashes_match=True,raw_decodes_finite=True,
                comparisons={name:dict(passed=True,bitwise_equal=True,max_abs_error=0.0,rmse=0.0,
                    relative_l2_error=0.0,cosine_similarity=1.0,atol=check.ATOL,rtol=check.RTOL,elements=elements)
                    for name,elements in [('latents',65536),('images',786432)]},
                outputs={name:dict(shape=[1,16,64,64] if name.endswith('latents') else [1,3,512,512],
                    dtype='torch.bfloat16',finite=True,minimum=-1.0,maximum=1.0,sha256='0'*64,retained_array=False)
                    for name in ('reference_latents','actual_latents','reference_image','actual_image')})
        result.append(row)
    return result


class GFModelMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_dir(): raise unittest.SkipTest('pinned official GF-DiT source required')
        cls.directory=TemporaryDirectory()
        cls.derived=check.migration.apply(SOURCE,MANIFEST,MANIFEST_SHA,Path(cls.directory.name)/'derived')
        cls.control=check.migration.load(cls.derived['root'],cls.derived['manifest_sha256'])

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name==check.migration.NAMESPACE or name.startswith(check.migration.NAMESPACE+'.'):
                del sys.modules[name]
        cls.directory.cleanup()

    def test_all_five_tensor_shape_dtype_hash_and_byte_descriptors_are_required(self):
        check.validate_descriptor(descriptor(),request())
        for mutation in ('missing','dtype','hash','bytes','cfg','time','sigma','latent'):
            value=descriptor()
            fields=value['tensor_fields']
            if mutation=='missing': del fields['sigmas']
            if mutation=='dtype': fields['latents']['dtype']='torch.float32'
            if mutation=='hash': fields['latents']['sha256']='not-a-hash'
            if mutation=='bytes': fields['latents']['bytes']-=1
            if mutation=='cfg': fields['prompt_embeds']['shape'][0]=1
            if mutation=='time': fields['timesteps']['shape']=[26]
            if mutation=='sigma': fields['sigmas']['shape']=[27]
            if mutation=='latent': fields['latents']['shape']=[1,16,32,32]
            with self.assertRaises(ValueError,msg=mutation): check.validate_descriptor(value,request())

    def test_cursor_and_full_immutable_request_cannot_be_reset_or_changed(self):
        for cursor in (True,False,0,1,3,28,2.0):
            value=descriptor();value['cursor']=cursor
            with self.assertRaises(ValueError): check.validate_metadata(value,request())
        for key,new in [('prompt','different'),('negative_prompt',''),('seed',102),('steps',26),
                        ('height',768),('guidance_scale',7.0),('max_sequence_length',128)]:
            value=descriptor();value['request'][key]=new
            with self.assertRaises(ValueError,msg=key): check.validate_metadata(value,request())
        value=descriptor();del value['request']['negative_prompt']
        with self.assertRaises(ValueError): check.validate_metadata(value,request())

    def test_actual_original_plan_preserves_artifact_identity_and_singleton_layouts(self):
        value=descriptor()
        specs,plan=check.build_plan(self.control,value)
        self.assertEqual(specs['model-rank0'].ranks,(0,))
        self.assertEqual(specs['model-rank1'].ranks,(1,))
        self.assertEqual(plan.request_metadata,value)
        self.assertNotEqual(plan.artifacts[0].handle.artifact_id,plan.artifacts[0].effective_source_handle.artifact_id)
        codec=check.make_codec(self.control,request())
        for name in specs:
            layout=codec.describe_output(group=specs[name],group_rank=0,request_metadata=value,artifact=plan.artifacts[0].handle)
            self.assertEqual([field.field_path for field in layout.tensors],[(name,) for name in check.TENSOR_FIELDS])
            self.assertTrue(all(field.layout.kind==self.control.protocol.TensorLayoutKind.REPLICATED for field in layout.tensors))
        bad_group=self.control.protocol.ExecutionGroupSpec('sp2',(0,1),self.control.protocol.ParallelSpec(sp=2),
                                                         (self.control.protocol.TaskKind.DIT_STEP_CHUNK,))
        with self.assertRaisesRegex(ValueError,'SP1'):
            codec.describe_output(group=bad_group,group_rank=0,request_metadata=value,artifact=plan.artifacts[0].handle)

    def test_original_codec_strips_all_declared_fields_from_control_metadata(self):
        value=descriptor()
        specs,plan=check.build_plan(self.control,value)
        codec=check.make_codec(self.control,request())
        layout=codec.describe_output(group=specs['model-rank0'],group_rank=0,request_metadata=value,
                                     artifact=plan.artifacts[0].effective_source_handle)
        # CPU sentinel values test only metadata stripping. Real CUDA bytes are
        # exclusively validated by the source-bound two-rank GPU entry point.
        payload=dict(schema=check.STATE_SCHEMA,request=asdict(request()),cursor=2,
                     **{name:'explicit CPU unit sentinel' for name in check.TENSOR_FIELDS})
        raw=codec.pack_metadata(value=payload,layout=layout)
        skeleton=pickle.loads(raw)
        self.assertTrue(all(skeleton[name] is None for name in check.TENSOR_FIELDS))
        self.assertEqual(skeleton['request'],asdict(request()))
        self.assertEqual(skeleton['cursor'],2)
        self.assertLess(len(raw),4096)
        self.assertTrue(all(payload[name]=='explicit CPU unit sentinel' for name in check.TENSOR_FIELDS))
        payload['hidden_extra_field']='reject rather than silently pickle'
        with self.assertRaisesRegex(ValueError,'undeclared'): codec.pack_metadata(value=payload,layout=layout)

    def test_expected_p2p_covers_full_large_prompt_tensor_and_five_chunks_streams(self):
        value=descriptor()
        source=check.expected_p2p(value,0);destination=check.expected_p2p(value,1)
        self.assertGreater(source['send_bytes'],5*1024*1024)
        self.assertGreater(source['send_calls'],80)
        self.assertEqual(source['send_bytes'],destination['recv_bytes'])
        self.assertEqual(source['send_calls'],destination['recv_calls'])
        self.assertEqual(source['recv_bytes'],0)
        self.assertEqual(destination['send_bytes'],0)

    def test_complete_model_and_migration_evidence_never_grants_serving_qualification(self):
        result=check.assess(records(),descriptor())
        self.assertEqual(result['model_transformer_calls'],56)
        self.assertFalse(result['qualification_eligible'])
        self.assertFalse(result['edf_decisions_tested'])
        self.assertFalse(result['native_multiproc_worker_reproduced'])
        self.assertFalse(result['sp2_reshaping_tested'])
        self.assertFalse(result['serving_executed'])
        self.assertEqual(result['allows'],[])

    def test_missing_steps_reference_real_rank_or_tensor_evidence_fails_closed(self):
        for mutation in ('missing','uuid','source','model','cleanup','status','phase','trace','migration',
                         'descriptor','rank','singleton','first','resume','reference','hash','p2p','p2p_bool','comparison','elements',
                         'outputs','shape','finite','preclamp','nan_error','negative_error','bad_cosine','bad_hash',
                         'nan_extrema','bitwise_error','tolerance'):
            value=records()
            if mutation=='missing': value.pop()
            elif mutation=='uuid': value[1]['gpu_uuid']=value[0]['gpu_uuid']
            elif mutation=='source': value[1]['source_binding_sha256']='other'
            elif mutation=='model': value[1]['model_weights_loaded']=False
            elif mutation=='cleanup': value[1]['cleanup_verified']=False
            elif mutation=='status': value[1]['status']='failed'
            elif mutation=='phase': value[1]['phase_executions']=2
            elif mutation=='trace': value[0]['phase_trace'].pop()
            elif mutation=='migration': value[1]['migrations']=True
            elif mutation=='descriptor': value[1]['descriptor']['cursor']=0
            elif mutation=='rank': value[1]['model_pool_rank']=0
            elif mutation=='singleton': value[1]['model_sp_ranks']=[0,1]
            elif mutation=='first': value[0]['source_transformer_calls']=1
            elif mutation=='resume': value[1]['resume_transformer_calls']=25
            elif mutation=='reference': value[1]['reference_transformer_calls']=0
            elif mutation=='hash': value[1]['received_tensor_hashes_match']=False
            elif mutation=='p2p': value[1]['p2p']['recv_bytes']-=1
            elif mutation=='p2p_bool': value[1]['p2p']['send_bytes']=False
            elif mutation=='comparison': value[1]['comparisons']['latents']['passed']=False
            elif mutation=='elements': value[1]['comparisons']['images']['elements']=1
            elif mutation=='outputs': value[1]['outputs'].pop('actual_image')
            elif mutation=='shape': value[1]['outputs']['actual_image']['shape']=[1,3,256,256]
            elif mutation=='finite': value[1]['outputs']['actual_latents']['finite']=False
            elif mutation=='preclamp': value[1]['raw_decodes_finite']=False
            elif mutation=='nan_error': value[1]['comparisons']['latents']['rmse']=float('nan')
            elif mutation=='negative_error': value[1]['comparisons']['latents']['relative_l2_error']=-0.1
            elif mutation=='bad_cosine': value[1]['comparisons']['latents']['cosine_similarity']=2.0
            elif mutation=='bad_hash': value[1]['outputs']['actual_latents']['sha256']='1'*64
            elif mutation=='nan_extrema': value[1]['outputs']['actual_latents']['maximum']=float('nan')
            elif mutation=='bitwise_error': value[1]['comparisons']['latents']['max_abs_error']=1e-7
            elif mutation=='tolerance': value[1]['comparisons']['latents'].update(bitwise_equal=False,max_abs_error=1.0)
            with self.assertRaises(ValueError,msg=mutation): check.assess(value,descriptor())


if __name__=='__main__': unittest.main()
