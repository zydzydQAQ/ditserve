"""Metadata/source/completion guards only; these tests cannot qualify GPU migration."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import external_tetriserve_migration as migration


def packet():
    descriptors = {name:dict(shape=[2,3],dtype='torch.float32',bytes=24,sha256='a'*64)
                   for name in migration.NAMES}
    future = SimpleNamespace(sub_req_id=1,src_rank=0,step_range=(0,2),
        tensors_info={name:SimpleNamespace(shape=(2,3),dtype='torch.float32') for name in migration.NAMES})
    return future,descriptors


def completion():
    _,wire=packet()
    output, comparisons = {}, {}
    for kind, shape in (('latent', [1,16,64,64]), ('decoded', [1,3,512,512])):
        for prefix in ('reference_', 'resumed_'):
            output[prefix+kind] = dict(shape=shape, dtype='torch.bfloat16', finite=True,
                retained_array=False, sha256='a'*64, minimum=-1., maximum=1.)
        comparisons[kind] = dict(passed=True, bitwise_equal=True, elements=math.prod(shape),
            max_abs_error=0., rmse=0., relative_l2_error=0., cosine_similarity=1., atol=1e-5, rtol=1e-5)
    return [dict(rank=0,status='passed',sender_thread_stopped=True,distributed_closed=True,
                 gpu=dict(uuid='GPU-0'),executed_interval=[0,2],source_payload_released=True,wire_sent=wire),
            dict(rank=1,status='passed',sender_thread_stopped=True,distributed_closed=True,
                 gpu=dict(uuid='GPU-1'),executed_interval=[2,28],wire_bytes_exact=True,wire_received=deepcopy(wire),
                 reference_steps=28,resumed_steps=26,received_cursor=2,numerical_check_passed=True,
                 comparisons=comparisons,output=output)]


class TetriMigrationTests(unittest.TestCase):
    def test_native_future_binds_request_identity_absolute_cursor_and_tensor_order(self):
        future,wire=packet()
        request=dict(prompt='fox',seed=101,steps=28,guidance_scale=4.5)
        migration.validate_packet(future,request,request,wire)
        for field,value in [('src_rank',1),('sub_req_id',2),('step_range',(0,3))]:
            bad=deepcopy(future);setattr(bad,field,value)
            with self.assertRaises(ValueError):migration.validate_packet(bad,request,request,wire)
        with self.assertRaises(ValueError):
            migration.validate_packet(future,dict(request,seed=102),request,wire)
        reordered=dict(reversed(list(wire.items())))
        with self.assertRaises(ValueError):migration.validate_packet(future,request,request,reordered)
        bad=deepcopy(wire);bad['latents']['shape']=[6]
        with self.assertRaises(ValueError):migration.validate_packet(future,request,request,bad)

    def test_missing_rank_wrong_wire_cleanup_or_numerics_cannot_pass(self):
        self.assertTrue(migration.assess(completion()))
        self.assertFalse(migration.assess(completion()[:1]))
        for mutation in ('uuid','wire','cursor','reference','numerics','thread','distributed','source_release'):
            rows=completion()
            if mutation=='uuid':rows[1]['gpu']['uuid']='GPU-0'
            if mutation=='wire':rows[1]['wire_received']['latents']['sha256']='b'*64
            if mutation=='cursor':rows[1]['executed_interval']=[0,26]
            if mutation=='reference':rows[1]['reference_steps']=2
            if mutation=='numerics':rows[1]['numerical_check_passed']=False
            if mutation=='thread':rows[0]['sender_thread_stopped']=False
            if mutation=='distributed':rows[1]['distributed_closed']=False
            if mutation=='source_release':rows[0]['source_payload_released']=False
            self.assertFalse(migration.assess(rows),mutation)

    def test_numerical_receipt_requires_actual_shapes_finite_errors_and_both_comparisons(self):
        for mutation in ('received_cursor', 'missing_comparison', 'failed_comparison', 'nan_error',
                         'large_error', 'wrong_tolerance', 'wrong_elements', 'output_nonfinite',
                         'output_shape', 'output_dtype', 'output_hash', 'output_extrema', 'inconsistent_exact'):
            rows = completion(); row = rows[1]
            if mutation == 'received_cursor': row['received_cursor'] = 0
            if mutation == 'missing_comparison': del row['comparisons']['decoded']
            if mutation == 'failed_comparison': row['comparisons']['latent']['passed'] = False
            if mutation == 'nan_error': row['comparisons']['decoded']['rmse'] = float('nan')
            if mutation == 'large_error': row['comparisons']['latent']['max_abs_error'] = 1.
            if mutation == 'wrong_tolerance': row['comparisons']['latent']['atol'] = 1.
            if mutation == 'wrong_elements': row['comparisons']['latent']['elements'] = 1
            if mutation == 'output_nonfinite': row['output']['resumed_latent']['finite'] = False
            if mutation == 'output_shape': row['output']['reference_decoded']['shape'] = [1,3,1,1]
            if mutation == 'output_dtype': row['output']['reference_latent']['dtype'] = 'torch.float32'
            if mutation == 'output_hash': row['output']['reference_decoded']['sha256'] = 'z'*64
            if mutation == 'output_extrema': row['output']['reference_decoded']['minimum'] = float('inf')
            if mutation == 'inconsistent_exact': row['output']['resumed_latent']['sha256'] = 'b'*64
            self.assertFalse(migration.assess(rows), mutation)

    def test_bound_derived_transport_cannot_be_modified_or_extended(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            files={}
            for name in migration.TRANSPORT:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('# exact transport fixture\n')
                files[name]=migration.sha(path)
            manifest=dict(schema=migration.port.PORT_SCHEMA,upstream_commit=migration.port.upstream.COMMIT,
                          files=files,preserved_native_mechanisms=dict(files))
            target=root/'PORT_MANIFEST.json';migration.write_json(target,manifest)
            expected=migration.sha(target)
            migration.verify_derived(root,expected)
            extra=root/'unexpected.py';extra.write_text('raise RuntimeError()\n')
            with self.assertRaisesRegex(ValueError,'inventory'):migration.verify_derived(root,expected)
            extra.unlink()
            changed=root/migration.TRANSPORT[0];changed.write_text('# changed transport\n')
            with self.assertRaisesRegex(ValueError,'bytes or path'):migration.verify_derived(root,expected)

    def test_plan_does_not_spawn_load_model_or_create_output(self):
        import external_sd35_round as adapter
        import external_sd35_round_check as check
        with TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'model.json'
            manifest.write_text(json.dumps(dict(model=adapter.SD35_MODEL,revision=adapter.SD35_REVISION)))
            output=root/'output'
            argv=['plan','--source-root',str(root/'source'),'--source-manifest',str(root/'source.json'),
                '--source-manifest-sha256','0'*64,'--shared-python-root',str(root/'shared'),
                '--model-root',str(root/'weights'),'--model-manifest',str(manifest),
                '--model-manifest-sha256',migration.sha(manifest),'--protocol',str(check.DEFAULT_PROTOCOL),
                '--protocol-sha256',migration.sha(check.DEFAULT_PROTOCOL),'--prompt','A fox.', '--output-dir',str(output)]
            source_plan=dict(files={name:'1'*64 for name in migration.TRANSPORT})
            capture=io.StringIO()
            with patch.object(adapter,'verify_shared_sources',return_value=root/'shared'), \
                 patch.object(migration.port,'plan',return_value=(source_plan,{},{})), \
                 patch.object(migration,'run_workers',side_effect=AssertionError('must not launch')) as run, \
                 patch.object(check,'check_storage',side_effect=AssertionError('plan must remain local/read-only')), \
                 redirect_stdout(capture):
                self.assertEqual(migration.main(argv),0)
            run.assert_not_called()
            result=json.loads(capture.getvalue())
            self.assertEqual(result['status'],'planned')
            self.assertFalse(result['model_loaded'])
            self.assertFalse(result['native_transport_executed'])
            self.assertFalse(result['cross_rank_sp1_verified'])
            self.assertFalse(result['sp2_verified'])
            self.assertFalse(result['serving_executed'])
            self.assertFalse(result['qualification_eligible'])
            self.assertEqual(result['allows'],[])
            self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
