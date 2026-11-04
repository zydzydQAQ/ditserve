"""CPU contracts for first-step route observation and cross-SP evidence gates."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import hashlib
import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import external_sd35_cross_sp_check as cross


class Handle:
    def __init__(self, hooks, fn): self.hooks=hooks; self.fn=fn
    def remove(self): self.hooks.remove(self.fn)


class Transformer:
    def __init__(self):
        self.pre=[]; self.post=[]; self.output=(object(),); self.calls=0; self.fail=False
    def register_forward_pre_hook(self, fn, *, with_kwargs=False):
        assert with_kwargs; self.pre.append(fn); return Handle(self.pre,fn)
    def register_forward_hook(self, fn, *, with_kwargs=False):
        assert with_kwargs; self.post.append(fn); return Handle(self.post,fn)
    def forward(self, hidden_states, encoder_hidden_states, pooled_projections, timestep, return_dict=False):
        self.calls+=1
        if self.fail: raise RuntimeError('real transformer failed')
        return self.output
    def __call__(self, *args, **kwargs):
        for fn in self.pre: self.unused=fn(self,args,kwargs)
        output=self.forward(*args,**kwargs)
        for fn in self.post:
            self.assert_none=fn(self,args,kwargs,output)
            if self.assert_none is not None: output=self.assert_none
        return output


class Scheduler:
    def __init__(self):
        self.timesteps=object(); self.sigmas=object(); self.output=(object(),); self.calls=0; self.fail=False
    def step(self, model_output, timestep, sample, return_dict=False):
        self.calls+=1
        if self.fail: raise RuntimeError('real scheduler failed')
        return self.output


class Pipe:
    def __init__(self):
        self.transformer=Transformer(); self.scheduler=Scheduler(); self.inputs=[object() for _ in range(4)]
    def _naive_forward(self):
        out=self.transformer(*self.inputs,return_dict=False)
        assert out is self.transformer.output
        return self.scheduler.step(out[0],self.inputs[3],self.inputs[0],return_dict=False)
    def _sync_pipeline(self):
        out=self.transformer(hidden_states=self.inputs[0],encoder_hidden_states=self.inputs[1],
            pooled_projections=self.inputs[2],timestep=self.inputs[3],return_dict=False)
        assert out is self.transformer.output
        return self._scheduler_step(noise_pred=out[0],latents=self.inputs[0],t=self.inputs[3])
    def _scheduler_step(self, noise_pred, latents, t):
        return self.scheduler.step(noise_pred,t,latents,return_dict=False)[0]
    def _async_pipeline(self): raise AssertionError('async route must not run')


def observe(pipe, native):
    return cross.FirstStepObserver(pipe,lambda:pipe.scheduler,native=native,
                                   metadata=lambda scheduler:{'actual':scheduler},snapshot=lambda x:x)


def descriptor(shape=None):
    return dict(shape=[1] if shape is None else shape,dtype='torch.bfloat16',finite=True,
                retained_array=False,minimum=-1.,maximum=1.,sha256='a'*64)


def pair():
    metrics=dict(passed=True,bitwise_equal=True,max_abs_error=0.,rmse=0.,relative_l2_error=0.,
                 cosine_similarity=1.,atol=1e-5,rtol=1e-5,elements=1,
                 reference=descriptor(),actual=descriptor())
    return dict(boundaries={key:deepcopy(metrics) for key in cross.BOUNDARIES},
                first_bitwise=None,first_tolerance=None)


class CrossSPTests(unittest.TestCase):
    def test_round_calls_original_once_returns_exact_object_and_restores_absence(self):
        pipe=Pipe(); observer=observe(pipe,False)
        with observer.observe():
            result=pipe._sync_pipeline()
        self.assertIs(result,pipe.scheduler.output[0])
        self.assertIs(observer.tensors['transformer_prediction'],pipe.transformer.output[0])
        self.assertIs(observer.tensors['scheduler_output'],result)
        self.assertEqual((pipe.transformer.calls,pipe.scheduler.calls),(1,1))
        self.assertEqual((pipe.transformer.pre,pipe.transformer.post),([],[]))
        for name in ('_naive_forward','_sync_pipeline','_scheduler_step','_async_pipeline'):
            self.assertNotIn(name,vars(pipe))

    def test_native_stops_only_after_real_solver_return_and_restores_instance_override(self):
        pipe=Pipe(); original=pipe.scheduler.step; pipe.scheduler.step=original
        saved_sync=pipe._sync_pipeline; pipe._sync_pipeline=saved_sync
        observer=observe(pipe,True)
        with observer.observe(): observer.run_native(pipe._naive_forward)
        self.assertTrue(observer.stopped)
        self.assertEqual((pipe.transformer.calls,pipe.scheduler.calls),(1,1))
        self.assertIs(pipe.scheduler.step,original)
        self.assertIs(pipe._sync_pipeline,saved_sync)
        self.assertIs(observer.tensors['scheduler_output'],pipe.scheduler.output[0])
        self.assertEqual(observer.counts['solver_return'],1)

    def test_real_forward_and_step_errors_propagate_and_all_hooks_restore(self):
        for target in ('transformer','scheduler'):
            for native in (False,True):
                pipe=Pipe();getattr(pipe,target).fail=True;observer=observe(pipe,native)
                with self.assertRaisesRegex(RuntimeError,'real '+target+' failed'):
                    with observer.observe():
                        if native: observer.run_native(pipe._naive_forward)
                        else: pipe._sync_pipeline()
                self.assertEqual((pipe.transformer.pre,pipe.transformer.post),([],[]))
                self.assertNotIn('step',vars(pipe.scheduler))
                self.assertNotIn('_scheduler_step',vars(pipe))
                self.assertFalse(observer.stopped)

    def test_unrelated_control_exception_is_not_swallowed(self):
        pipe=Pipe();observer=observe(pipe,True);unrelated=cross.FirstStepComplete('other')
        def fail(): raise unrelated
        with self.assertRaises(cross.FirstStepComplete) as raised:
            with observer.observe(): observer.run_native(fail)
        self.assertIs(raised.exception,unrelated)
        self.assertEqual(observer.counts['solver_return'],0)
        self.assertNotIn('step',vars(pipe.scheduler))

    def test_missing_repeated_or_wrong_route_cannot_complete(self):
        pipe=Pipe();observer=observe(pipe,False)
        with self.assertRaisesRegex(ValueError,'coverage'):
            with observer.observe(): pass
        pipe=Pipe();observer=observe(pipe,False)
        with self.assertRaisesRegex(ValueError,'repeated first-step boundary'):
            with observer.observe():
                pipe._sync_pipeline();pipe._sync_pipeline()
        pipe=Pipe();observer=observe(pipe,False)
        with self.assertRaisesRegex(ValueError,'coverage'):
            with observer.observe(): pipe._naive_forward()
        self.assertEqual((pipe.transformer.pre,pipe.transformer.post),([],[]))

    def test_partition_uses_actual_rows_and_rejects_overlap_gap_and_boolean(self):
        rows=[dict(rank=0,ranges=[[0,32],[64,96]],local_height=64),
              dict(rank=1,ranges=[[32,64],[96,128]],local_height=64)]
        cross.validate_partition(rows)
        for bad in ([[0,65]],[[1,65]],[[False,64]],[[64,128],[0,64]],[[0,63]]):
            modified=deepcopy(rows);modified[0]['ranges']=bad
            with self.assertRaises(ValueError):cross.validate_partition(modified)
        with self.assertRaises(ValueError):cross.validate_partition([rows[0],rows[0]])

    def test_actual_spatial_slicing_preserves_global_row_order(self):
        class Tensor:
            shape=(2,16,128,128)
            def __getitem__(self,key):
                self_keys=(key[0],key[1],key[3])
                if self_keys!=(slice(None),slice(None),slice(None)):
                    raise AssertionError('only latent row axis may be sliced')
                return list(range(key[2].start,key[2].stop))
        calls=[]
        def cat(chunks,dim):
            calls.append(dim)
            return [value for chunk in chunks for value in chunk]
        with patch.dict(sys.modules,torch=SimpleNamespace(cat=cat)):
            actual=cross.slice_rows(Tensor(),[[0,32],[64,96]])
        self.assertEqual(actual,list(range(32))+list(range(64,96)))
        self.assertEqual(calls,[2])

    def test_scalar_description_allowed_and_bitwise_distinct_from_tolerance(self):
        row=pair(); row['boundaries']['timestep'].update(reference=descriptor([]),actual=descriptor([]))
        self.assertTrue(cross.valid_pair(row))
        metric=row['boundaries']['transformer_prediction']
        metric.update(bitwise_equal=False,max_abs_error=1e-6,rmse=1e-7,relative_l2_error=1e-7)
        metric['actual']['sha256']='b'*64
        row.update(cross.first_differences(row['boundaries']))
        self.assertEqual(row['first_bitwise'],'transformer_prediction')
        self.assertIsNone(row['first_tolerance'])
        self.assertTrue(cross.valid_pair(row))
        metric.update(passed=False,max_abs_error=.1,rmse=.01,relative_l2_error=.01)
        row.update(cross.first_differences(row['boundaries']))
        self.assertEqual(row['first_tolerance'],'transformer_prediction')
        self.assertTrue(cross.valid_pair(row))

    def test_persisted_metrics_reject_missing_nonfinite_and_conflicting_flags(self):
        for change in ('missing','nan','sha','first','bound','shape','bool'):
            row=pair(); metric=row['boundaries']['scheduler_output']
            if change=='missing':del row['boundaries']['hidden_states']
            if change=='nan':metric['rmse']=float('nan')
            if change=='sha':metric['actual']['sha256']='b'*64
            if change=='first':row['first_bitwise']='scheduler_output'
            if change=='bound':metric['max_abs_error']=.2
            if change=='shape':metric['actual']['shape']=[2]
            if change=='bool':metric['passed']=1
            self.assertFalse(cross.valid_pair(row),change)

    def test_r5_reproduction_requires_all_local_input_output_and_runtime_descriptors(self):
        ref=dict(cfg_prediction=descriptor(),scheduler_output=descriptor(),
                 inputs=dict(latents=descriptor(),t=descriptor([])),runtime={'parallel':{'sp_degree':2}})
        receipt=dict(tensors=dict(cfg_prediction=descriptor(),scheduler_output=descriptor(),
            scheduler_sample=descriptor(),scheduler_timestep=descriptor([])),metadata=dict(runtime=deepcopy(ref['runtime'])))
        self.assertTrue(cross.reproduced(receipt,ref))
        for field in ('cfg_prediction','scheduler_output','scheduler_sample','scheduler_timestep'):
            bad=deepcopy(receipt);bad['tensors'][field]['sha256']='b'*64
            self.assertFalse(cross.reproduced(bad,ref),field)
        bad=deepcopy(receipt);bad['metadata']['runtime']['parallel']['sp_degree']=1
        self.assertFalse(cross.reproduced(bad,ref))

    def test_runtime_text_flags_follow_actual_native_and_round_dispatch(self):
        for route in cross.ROUTES:
            runtime=dict(parallel=dict(pp_degree=1,cfg_degree=1,dp_degree=1,
                sp_degree=2 if route=='sp2_round' else 1),patch_mode=False,text_embed_sp_pad=0,
                split_latents_by_rows=route!='sp1_native',split_text_embed_in_sp=route=='sp1_round')
            self.assertTrue(cross.valid_runtime(runtime,route))
            for field in ('split_text_embed_in_sp','split_latents_by_rows','patch_mode'):
                bad=deepcopy(runtime);bad[field]=not bad[field]
                self.assertFalse(cross.valid_runtime(bad,route), (route,field))
            bad=deepcopy(runtime);bad['parallel']['sp_degree']=True
            self.assertFalse(cross.valid_runtime(bad,route))

    def test_native_missing_stop_and_unexpected_scheduler_tuple_are_rejected(self):
        pipe=Pipe();observer=observe(pipe,True)
        with self.assertRaisesRegex(ValueError,'did not reach'):
            with observer.observe(): observer.run_native(lambda: None)
        pipe=Pipe();pipe.scheduler.output=(object(),object());observer=observe(pipe,True)
        with self.assertRaisesRegex(ValueError,'scheduler tuple'):
            with observer.observe():observer.run_native(pipe._naive_forward)
        self.assertEqual(pipe.scheduler.calls,1)
        self.assertFalse(observer.stopped)
        self.assertNotIn('step',vars(pipe.scheduler))

    def test_observer_metadata_failure_does_not_run_model_or_leave_hooks(self):
        pipe=Pipe();observer=observe(pipe,False)
        def fail(scheduler): raise RuntimeError('metadata source failure')
        observer.metadata=fail
        with self.assertRaisesRegex(RuntimeError,'metadata source failure'):
            with observer.observe():pipe._sync_pipeline()
        self.assertEqual(pipe.transformer.calls,0)
        self.assertEqual((pipe.transformer.pre,pipe.transformer.post),([],[]))
        self.assertNotIn('_scheduler_step',vars(pipe))

    def test_reference_loader_binds_rank_size_sha_and_validator(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); binding={'fixed':'prior'}; ranks=[];paths=[]
            for rank in (0,1):
                record=dict(rank=rank,trajectories={'round_sp2_full':dict(
                    backbone=[dict(output={'noise_pred':descriptor()},runtime={})],
                    scheduler=[dict(inputs={},output=descriptor())])})
                raw=json.dumps(record).encode();path=root/('trace-'+str(rank)+'.json');path.write_bytes(raw);paths.append(path)
                ranks.append(dict(rank=rank,initial_five_tensors={},trace=dict(bytes=len(raw),
                    file='rank-'+str(rank)+'.trace.json',sha256=hashlib.sha256(raw).hexdigest())))
            child=dict(binding=binding,diagnostic_completed=True,observation_admissible=True,
                observer_reproduced=True,solver_counterfactual_enabled=True,qualification_eligible=False,ranks=ranks)
            raw=json.dumps(child).encode();summary=root/'child.json';summary.write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
            with patch.object(cross,'R5_SHA256',sha),patch.object(cross.base,'assess',return_value=True),patch.object(cross.trace,'valid_trace',return_value=True):
                self.assertEqual(len(cross.bind_reference(summary,sha,paths,binding)),2)
                with self.assertRaises(ValueError):cross.bind_reference(summary,sha,list(reversed(paths)),binding)
                with self.assertRaises(ValueError):cross.bind_reference(summary,sha,paths,{'other':1})
                with patch.object(cross.trace,'valid_trace',return_value=False):
                    with self.assertRaises(ValueError):cross.bind_reference(summary,sha,paths,binding)
                paths[0].write_bytes(paths[0].read_bytes()+b' ')
                with self.assertRaises(ValueError):cross.bind_reference(summary,sha,paths,binding)
                summary.write_bytes(raw+b' ')
                with self.assertRaises(ValueError):cross.bind_reference(summary,sha,paths,binding)


if __name__=='__main__': unittest.main()
