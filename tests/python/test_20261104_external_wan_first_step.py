"""CPU observer/provenance contracts; these fixtures are not GPU evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import external_wan_first_step as w


class Recorder:
    def __init__(self):self.values=[]
    def capture(self,name,value):self.values.append((name,value))


class Transformer:
    def forward(self,hidden_states,timestep,encoder_hidden_states,return_dict=False):
        self.calls=getattr(self,'calls',0)+1
        if getattr(self,'fail',False):raise RuntimeError('actual transformer failure')
        return (hidden_states,)


class Scheduler:
    def __init__(self):self.timesteps=list(range(50));self.sigmas=list(range(51));self.config={'solver_order':2};self.calls=0
    def step(self,model_output,timestep,sample,return_dict=False):
        self.calls+=1;return (sample,)


class Model:
    def _make_generator(self,seed):return SimpleNamespace(device='cpu',initial_seed=lambda:seed,get_state=lambda:b'state')


class Pipe:
    def __init__(self):self.transformer=Transformer();self.scheduler=Scheduler()
    def encode_prompt(self,prompt):return (prompt,prompt)
    def prepare_latents(self,value):return value


def complete_fixture():
    required={'generator.state','prepare_latents.0.out','encode_prompt.0.out.0','encode_prompt.0.out.1',
        'scheduler.0.in.model_output','scheduler.0.in.sample','scheduler.0.in.timestep','scheduler.0.out.0',
        'scheduler.0.schedule.timesteps','scheduler.0.schedule.sigmas'}
    for i in (0,1):required.update({f'transformer.{i}.in.hidden_states',f'transformer.{i}.in.encoder_hidden_states',f'transformer.{i}.in.timestep',f'transformer.{i}.out.0'})
    return dict(rank=0,world_size=1,binding={'x':'source'},status='first_step_observed',cleanup_completed=True,
        runtime=dict(gpu_uuid='00000000-0000-0000-0000-000000000001',sp=1,dp=1),
        loaded_source_sha256={k:'a'*64 for k in ('xfuser/runner.py','xfuser/model_executor/models/runner_models/wan.py','xfuser/model_executor/models/transformers/transformer_wan.py')},
        completed_requests=0,qualification_eligible=False,observation=dict(restored=True,stopped_before_step1=True,
        counts=dict(encode_prompt=1,prepare_latents=1,generator=1,transformer=2,scheduler=1)),
        boundaries=[dict(name=n,tensor=dict(shape=[2],dtype='torch.float32',finite=True,sha256='a'*64,bytes=8)) for n in sorted(required)])


class FirstStepTests(unittest.TestCase):
    def test_wrappers_preserve_identity_and_stop_before_next_original_call(self):
        pipe,model,recorder=Pipe(),Model(),Recorder();latent=object();text=object()
        with w.observe(pipe,model,recorder) as row:
            generator=model._make_generator(42);self.assertEqual(generator.initial_seed(),42)
            encoded=pipe.encode_prompt(text);self.assertIs(encoded[0],text)
            self.assertIs(pipe.prepare_latents(latent),latent)
            for _ in range(2):self.assertIs(pipe.transformer.forward(latent,999,text)[0],latent)
            self.assertIs(pipe.scheduler.step(latent,999,latent)[0],latent)
            with self.assertRaises(w.PrefixComplete):pipe.transformer.forward(latent,998,text)
        self.assertEqual(pipe.transformer.calls,2);self.assertEqual(pipe.scheduler.calls,1)
        self.assertTrue(row['restored']);self.assertTrue(row['stopped_before_step1'])
        for obj,key in ((pipe,'encode_prompt'),(pipe,'prepare_latents'),(model,'_make_generator'),(pipe.transformer,'forward'),(pipe.scheduler,'step')):self.assertNotIn(key,vars(obj))
        self.assertIs(next(x[1] for x in recorder.values if x[0]=='transformer.0.in')['hidden_states'],latent)

    def test_actual_exception_propagates_and_attributes_restore(self):
        pipe,model=Pipe(),Model();pipe.transformer.fail=True
        with self.assertRaisesRegex(RuntimeError,'actual transformer'):
            with w.observe(pipe,model,Recorder()) as row:pipe.transformer.forward(object(),999,object())
        self.assertTrue(row['restored']);self.assertEqual(pipe.transformer.calls,1)
        self.assertEqual(row['counts']['transformer'],0);self.assertNotIn('forward',vars(pipe.transformer))

    def test_existing_instance_override_restored_exactly(self):
        pipe,model=Pipe(),Model();saved=pipe.prepare_latents;pipe.prepare_latents=saved
        with w.observe(pipe,model,Recorder()) as row:pass
        self.assertIs(vars(pipe)['prepare_latents'],saved);self.assertTrue(row['restored'])

    def test_one_step_reschedule_is_rejected_before_scheduler_execution(self):
        pipe,model=Pipe(),Model();pipe.scheduler.timesteps=[999]
        with self.assertRaisesRegex(ValueError,'50-step'):
            with w.observe(pipe,model,Recorder()) as row:pipe.scheduler.step(object(),999,object())
        self.assertEqual(pipe.scheduler.calls,0);self.assertTrue(row['restored'])

    def test_extra_cfg_call_rejected(self):
        pipe,model=Pipe(),Model()
        with w.observe(pipe,model,Recorder()):
            for _ in range(2):pipe.transformer.forward(object(),999,object())
            with self.assertRaisesRegex(ValueError,'additional'):pipe.transformer.forward(object(),999,object())

    def test_comparison_counts_every_element_with_reference_relative_tolerance(self):
        import numpy as np
        ref=np.array([0.,100.,0.,1.],dtype=np.float64);actual=np.array([1.5e-5,100.,0.,1.])
        row=w.compare_arrays(actual,ref)
        self.assertEqual(row['total_elements'],4);self.assertEqual(row['failed_elements'],1)
        self.assertFalse(row['within_tolerance']);self.assertAlmostEqual(row['rmse'],7.5e-6)
        self.assertEqual(row['atol'],1e-5);self.assertEqual(row['rtol'],1e-5)
        tiny=np.array([1.+1e-10]);self.assertGreater(w.compare_arrays(tiny,np.array([1.]))['max_abs'],0.)

    def test_comparison_rejects_nonfinite_shape_and_accounts_zero_reference(self):
        import numpy as np
        for a,b in ((np.array([np.inf]),np.array([0.])),(np.zeros(2),np.zeros(3)),(np.array([]),np.array([]))):
            with self.assertRaises(ValueError):w.compare_arrays(a,b)
        row=w.compare_arrays(np.ones(3),np.zeros(3));self.assertIsNone(row['relative_l2']);self.assertEqual(row['failed_elements'],3)

    def test_budget_and_path_fail_closed(self):
        w.budget_check(0,w.BUDGET)
        for a,b in ((w.BUDGET,1),(-1,1),(True,1),(0,1.)):
            with self.assertRaises(ValueError):w.budget_check(a,b)
        for root,name in (('/tmp','boundary-000.safetensors'),('/shared/output/DITSERVE/example','../other'),('/shared/output/DITSERVE/example','boundary-0.safetensors')):
            with self.assertRaises(ValueError):w.safe_artifact(root,name)

    def test_source_binding_ignores_only_cross_node_stat_fingerprint(self):
        row=dict(request={'steps':50},source={'sha':'frozen'},model_identity={'manifest_sha256':'model','snapshot_fingerprint_sha256':'node-a'},protocol_sha256='protocol',adapter={'code':'commit'})
        other=deepcopy(row);other['model_identity']['snapshot_fingerprint_sha256']='node-b';self.assertEqual(w.reference_binding(row),w.reference_binding(other))
        other['request']['steps']=1;self.assertNotEqual(w.reference_binding(row),w.reference_binding(other))

    def test_full_boundary_denominator_counts_cleanup_and_first_divergence(self):
        row=complete_fixture();w.validate_rank(row)
        for mutate in (lambda r:r['observation']['counts'].update(transformer=1),lambda r:r.update(cleanup_completed=False),lambda r:r['boundaries'].pop()):
            damaged=deepcopy(row);mutate(damaged)
            with self.assertRaises(ValueError):w.validate_rank(damaged)
        rows=[]
        for rank in range(4):
            r=deepcopy(row);r.update(rank=rank,world_size=4)
            r['runtime'].update(sp=4,gpu_uuid='00000000-0000-0000-0000-'+str(rank+1).zfill(12))
            if rank==0:
                for i,b in enumerate(r['boundaries']):b['comparison']=dict(bitwise_equal=i!=2,within_tolerance=i!=3)
            rows.append(r)
        summary=w.aggregate(rows,row['binding'],4)
        self.assertEqual(summary['first_bitwise_difference'],rows[0]['boundaries'][2]['name'])
        self.assertEqual(summary['first_tolerance_difference'],rows[0]['boundaries'][3]['name'])
        self.assertEqual(summary['completed_requests'],0);self.assertFalse(summary['qualification_eligible'])
        with self.assertRaises(ValueError):w.aggregate(rows[:-1],row['binding'],4)
        bad=deepcopy(rows);bad[1]['runtime']['gpu_uuid']=bad[0]['runtime']['gpu_uuid']
        with self.assertRaisesRegex(ValueError,'UUID'):w.aggregate(bad,row['binding'],4)
        bad=deepcopy(rows);bad[1]['loaded_source_sha256']['xfuser/runner.py']='b'*64
        with self.assertRaisesRegex(ValueError,'inventory'):w.aggregate(bad,row['binding'],4)

    def test_read_limit_checked_before_reading_payload(self):
        with patch.object(Path,'resolve',lambda p:p),patch.object(Path,'is_file',return_value=True),patch.object(Path,'stat',return_value=SimpleNamespace(st_size=1025)),patch.object(Path,'read_bytes',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'read budget'):w.read_bound(Path('/shared/output/DITSERVE/a'),'a'*64,1024)


if __name__=='__main__':unittest.main()
