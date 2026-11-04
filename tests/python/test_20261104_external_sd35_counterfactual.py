"""CPU isolation and evidence gates; actual solver arithmetic requires the GPU run."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import external_sd35_trace as trace


class Device:
    def __init__(self, value):
        self.type, _, index = value.partition(':')
        self.index = int(index) if index else None
    def __str__(self):
        return self.type + (':' + str(self.index) if self.index is not None else '')


class Tensor:
    def __init__(self, value, shape=(1, 16, 64, 128), device='cuda:0'):
        self.value, self.shape, self.device = value, shape, Device(device)
    def detach(self): return self
    def clone(self): return deepcopy(self)
    def to(self, device):
        out = self.clone(); out.device = Device(str(device)); return out


def summary(value):
    return dict(shape=list(value.shape), dtype='torch.bfloat16',
        sha256=hashlib.sha256(json.dumps([value.value, value.shape]).encode()).hexdigest(),
        finite=True, minimum=float(value.value), maximum=float(value.value), retained_array=False)


def compare(a, b):
    difference = abs(a.value - b.value)
    return dict(elements=math.prod(a.shape), atol=1e-5, rtol=1e-5, passed=difference <= 1e-5 + 1e-5*abs(b.value),
        bitwise_equal=summary(a) == summary(b), max_abs_error=difference, rmse=difference,
        relative_l2_error=difference/max(abs(a.value),1e-12), cosine_similarity=1.)


class RNG:
    def __init__(self): self.cpu, self.gpu = 5, 7
    def consume(self): self.cpu += 1; self.gpu += 2
    def get_rng_state(self): return self.cpu
    @property
    def cuda(self):
        parent = self
        class CUDA:
            def get_rng_state(self, device): return parent.gpu
        return CUDA()
    @property
    def random(self): return self
    @contextmanager
    def fork_rng(self, devices):
        assert devices == [0]
        original = self.cpu, self.gpu
        try: yield
        finally: self.cpu, self.gpu = original
    @staticmethod
    def equal(a, b): return a == b


RANDOM = RNG()


class RawScheduler:
    fail = False
    def __init__(self):
        self._step_index = None; self._begin_index = None
        self.num_inference_steps = 28
        self.timesteps = Tensor(1000., (28,))
        self.sigmas = Tensor(1., (29,), device='cpu')
    @property
    def step_index(self): return self._step_index
    @property
    def begin_index(self): return self._begin_index
    def set_begin_index(self, value): self._begin_index = value
    def step(self, noise, t, sample, return_dict=False):
        RANDOM.consume()
        if self.fail: raise RuntimeError('counterfactual branch failed')
        self._step_index = 1 if self._step_index is None else self._step_index + 1
        out = sample.clone(); out.value += noise.value + (0.125 if self.sigmas.device.type == 'cuda' else 0.)
        # Deliberate mutation must affect only this branch's independently cloned input.
        sample.value = -999.
        return (out,)


class NativeScheduler:
    def __init__(self, module):
        self.module = module; self.module_type = type(module)
    def __getattr__(self, name):
        # Same missing-module recursion hazard as the frozen wrapper.
        return getattr(self.module, name)
    def __setattr__(self, name, value):
        if name == 'module': object.__setattr__(self, name, value)
        elif hasattr(self, 'module') and hasattr(self.module, name): setattr(self.module, name, value)
        else: object.__setattr__(self, name, value)
    def step(self, noise, t, sample, return_dict=False):
        RANDOM.consume()
        self._step_index = 1 if self._step_index is None else self._step_index + 1
        out = sample.clone(); out.value += noise.value + 1.
        return (out,)


def describe(scheduler):
    return dict(identity='native' if type(scheduler) is NativeScheduler else 'raw')


def fixture():
    scheduler = NativeScheduler(RawScheduler())
    values = dict(noise_pred=Tensor(2.), latents=Tensor(1.), t=Tensor(1000., ()))
    counter = trace.SolverCounterfactual(scheduler, values, describe, summarize=summary, compare=compare, torch_module=RANDOM)
    actual = scheduler.step(values['noise_pred'], values['t'], values['latents'])[0]
    return scheduler, values, counter, actual


def metadata():
    return dict(type='xfuser.model_executor.schedulers.scheduling_flow_match_euler_discrete.xFuserFlowMatchEulerDiscreteSchedulerWrapper',
        unwrapped_step=dict(file_sha256=trace.NATIVE_SOLVER_SHA256),
        bound_step=dict(file_sha256='570aa9cc5806b6dfc01eb66228deed5c41ffb0465947e27cfd42534493a58897'),
        config=dict(stochastic_sampling=False), inner=dict(
            type='diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler',
            bound_step=dict(file_sha256=trace.RAW_SOLVER_SHA256), config=dict(stochastic_sampling=False)))


def valid_record():
    scheduler, values, counter, actual = fixture(); row = counter.run(actual)
    native = row['variants']['native_replay']; raw = row['variants']['raw_round_device']
    record = dict(rank=0, solver_counterfactual_enabled=True, solver_counterfactual=row,
        schedulers=dict(native=metadata(), raw=dict(bound_step=dict(file_sha256=trace.RAW_SOLVER_SHA256))),
        trajectories=dict(normal_sp2=dict(scheduler=[dict(inputs=row['inputs'], output=row['actual_output'],
            solver_state_before=native['state_before'], scheduler_before=dict(identity='native'))]),
            round_sp2_full=dict(scheduler=[dict(inputs=deepcopy(row['inputs']), output=raw['output'],
                solver_state_before=raw['state_before'], scheduler_before=dict(identity='raw'))])))
    assert trace.valid_counterfactual(record)
    return record


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        RANDOM.cpu, RANDOM.gpu = 5, 7
        RawScheduler.fail = False

    def test_clone_avoids_wrapper_deepcopy_recursion_and_rejects_unknown_state(self):
        scheduler = NativeScheduler(RawScheduler())
        with self.assertRaises(RecursionError): deepcopy(scheduler)
        clone = trace.clone_native_scheduler(scheduler)
        self.assertIsNot(clone.module, scheduler.module)
        self.assertIsNot(clone.sigmas, scheduler.sigmas)
        clone.sigmas.value = 9.
        self.assertEqual(scheduler.sigmas.value, 1.)
        scheduler.extra = 'unverified'
        with self.assertRaisesRegex(ValueError, 'instance state'): trace.clone_native_scheduler(scheduler)

    def test_real_call_output_inputs_state_and_rng_are_preserved(self):
        scheduler, values, counter, actual = fixture()
        rng = RANDOM.cpu, RANDOM.gpu
        state = trace.solver_state(scheduler, summary)
        original_inputs = {k:summary(v) for k,v in values.items()}
        original_output = summary(actual)
        row = counter.run(actual)
        self.assertEqual((RANDOM.cpu, RANDOM.gpu), rng)
        self.assertEqual(trace.solver_state(scheduler, summary), state)
        self.assertEqual({k:summary(v) for k,v in values.items()}, original_inputs)
        self.assertEqual(summary(actual), original_output)
        self.assertEqual(row['actual_output'], row['variants']['native_replay']['output'])
        self.assertFalse(row['output_replaced'])
        self.assertEqual(set(row['variants']), set(trace.COUNTERFACTUAL_VARIANTS))
        self.assertEqual(len(row['comparisons']), 7)
        self.assertEqual({v['elements'] for v in row['comparisons'].values()}, {131072})

    def test_neighboring_branches_change_only_the_declared_factor(self):
        record = valid_record(); variants = record['solver_counterfactual']['variants']
        a,b,c,d = [variants[name] for name in trace.COUNTERFACTUAL_VARIANTS]
        self.assertEqual(a['state_before'], b['state_before'])
        self.assertNotEqual(a['implementation'], b['implementation'])
        self.assertEqual(c['state_before'], dict(b['state_before'],step_index=0,begin_index=0))
        self.assertEqual(c['implementation'], b['implementation'])
        self.assertEqual(d['state_before'],dict(c['state_before'],sigmas_device='cuda:0'))
        self.assertEqual(d['implementation'], c['implementation'])

    def test_branch_exception_restores_rng_and_leaves_actual_state(self):
        scheduler, values, counter, actual = fixture()
        rng = RANDOM.cpu, RANDOM.gpu; state = trace.solver_state(scheduler,summary)
        RawScheduler.fail = True
        with self.assertRaisesRegex(RuntimeError,'counterfactual branch failed'): counter.run(actual)
        self.assertEqual((RANDOM.cpu,RANDOM.gpu),rng)
        self.assertEqual(trace.solver_state(scheduler,summary),state)
        self.assertEqual(values['latents'].value,1.)

    def test_native_replay_drift_cannot_be_accepted(self):
        scheduler, values, counter, actual = fixture()
        actual.value += 1.
        with self.assertRaisesRegex(ValueError,'native replay'): counter.run(actual)

    def test_corrupt_state_source_metric_or_round_target_is_rejected(self):
        original = valid_record()
        for mutation in ('missing','input','target','native','state','schedule','identity','source','rng',
                         'elements','nan','tol','falsepass','bitwise','missing_pair','output_replaced'):
            record = deepcopy(original); row = record['solver_counterfactual']
            if mutation == 'missing': row['variants'].pop('native_replay')
            if mutation == 'input': record['trajectories']['round_sp2_full']['scheduler'][0]['inputs']['latents']['sha256']='0'*64
            if mutation == 'target': row['variants']['raw_round_device']['output']=dict(row['variants']['raw_round_device']['output'],sha256='0'*64)
            if mutation == 'native': row['variants']['native_replay']['output']=dict(row['actual_output'],sha256='0'*64)
            if mutation == 'state': row['variants']['raw_same_state']['state_before']['begin_index']=0
            if mutation == 'schedule': row['variants']['raw_round_device']['state_before']['sigmas']=dict(row['variants']['raw_round_device']['state_before']['sigmas'],sha256='0'*64)
            if mutation == 'identity': row['variants']['raw_same_state']['implementation']='native'
            if mutation == 'source': record['schedulers']['raw']['bound_step']['file_sha256']='0'*64
            if mutation == 'rng': row['cpu_cuda_rng_restored']=False
            if mutation == 'output_replaced': row['output_replaced']=True
            comparison=row['comparisons']['actual_vs_raw_same_state']
            if mutation == 'elements': comparison['elements']=1
            if mutation == 'nan': comparison['rmse']=float('nan')
            if mutation == 'tol': comparison['atol']=1.
            if mutation == 'falsepass': comparison['passed']=True
            if mutation == 'bitwise': comparison['bitwise_equal']=True
            if mutation == 'missing_pair': row['comparisons'].pop('native_replay_vs_raw_same_state')
            self.assertFalse(trace.valid_counterfactual(record),mutation)

    def test_observer_runs_sidecar_only_at_normal_step0_without_replacing_return(self):
        from test_20261104_external_sd35_trace import Pipe, runtime, summarize, invoke
        pipe = Pipe(); calls = []
        class Sidecar:
            def __init__(self, scheduler, values, describe_scheduler, **kwargs):
                calls.append(('construct', values['latents']))
            def run(self, output):
                self_output = calls[0][1]
                self.assertion = output is self_output
                if not self.assertion: raise AssertionError('real output object was replaced')
                calls.append(('run', output))
                return dict(complete=False)
        observer = trace.TraceObserver(pipe, runtime, 0, summarize=summarize, solver_counterfactual=True)
        with patch.object(trace,'require_counterfactual_source'), patch.object(trace,'solver_state',return_value={}), patch.object(trace,'SolverCounterfactual',Sidecar):
            for name in trace.TRAJECTORIES:
                with observer.trajectory(name):
                    cursor = 0
                    for end in (trace.CHUNK_ENDS if name=='round_sp2_chunked' else [28]):
                        if name != 'normal_sp2':
                            pipe.scheduler.sigmas.shape = (end-cursor+1,)
                        invoke(pipe,end-cursor);cursor=end
        self.assertEqual([name for name, unused in calls], ['construct','run'])
        self.assertEqual(pipe.calls,dict(sync=15,backbone=84,scheduler=84,naive=0,async_=0))
        self.assertFalse(any(name in vars(pipe) for name in trace.METHODS))

    def test_allclose_relative_tolerance_uses_the_actual_second_operand(self):
        record = valid_record(); row = record['solver_counterfactual']
        # Near the asymmetric allclose boundary: exceeds the left-reference
        # bound but remains within the actual second-operand bound.
        value = 4.0000500003
        for name in ('raw_same_state','raw_explicit_index'):
            row['variants'][name]['output'] = summary(Tensor(value))
        descriptions = dict(actual=row['actual_output'], **{name:v['output'] for name,v in row['variants'].items()})
        for name in row['comparisons']:
            a,b = name.split('_vs_')
            row['comparisons'][name] = compare(Tensor(descriptions[a]['minimum']),Tensor(descriptions[b]['minimum']))
        metric = row['comparisons']['actual_vs_raw_same_state']
        self.assertTrue(metric['passed'])
        self.assertGreater(metric['max_abs_error'],1e-5 + 1e-5*abs(row['actual_output']['maximum']))
        self.assertTrue(trace.valid_counterfactual(record))

    def test_source_and_pre_step_state_gates_fail_closed(self):
        value=metadata();trace.require_counterfactual_source(value)
        value['config']['stochastic_sampling']=True
        with self.assertRaisesRegex(ValueError,'frozen deterministic'):trace.require_counterfactual_source(value)
        scheduler=NativeScheduler(RawScheduler());scheduler._step_index=0
        values=dict(noise_pred=Tensor(2.),latents=Tensor(1.),t=Tensor(1000.,()))
        with self.assertRaisesRegex(ValueError,'before scheduler initialization'):
            trace.SolverCounterfactual(scheduler,values,describe,summarize=summary,compare=compare,torch_module=RANDOM)


if __name__=='__main__': unittest.main()
