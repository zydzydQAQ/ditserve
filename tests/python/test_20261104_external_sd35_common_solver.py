"""CPU contracts for the experimental normal-only common-solver path."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import external_sd35_common_solver_check as common
import external_tetriserve_sp2_diagnostic as diagnostic


class Raw:
    def __init__(self):
        self.config = dict(stochastic_sampling=False,shift=3.)
        self.calls = 0; self.index = None; self.fail = False
    def step(self, value, *, return_dict=False):
        self.calls += 1
        if self.fail: raise RuntimeError('real step failed')
        self.index = self.calls
        return value


class Wrapper:
    def __init__(self, module): self.module = module
    @property
    def config(self): return self.module.config
    def step(self, *args, **kwargs): raise AssertionError('wrapper must not run in experimental normal invocation')


def metadata():
    config = dict(stochastic_sampling=False,shift=3.)
    return dict(previous_type=common.WRAPPER_TYPE,selected_type=common.RAW_TYPE,
        previous_bound_step=dict(file_sha256=common.BOUND_WRAPPER_SHA256),
        previous_unwrapped_step=dict(file_sha256=common.NATIVE_SOLVER_SHA256),
        selected_step=dict(file_sha256=common.RAW_SOLVER_SHA256),configuration=config,
        configuration_sha256=common.canonical(config))


def runtime():
    return SimpleNamespace(parallel_config=SimpleNamespace(pp_degree=1,cfg_degree=1,sp_degree=2,dp_degree=1),patch_mode=False)


def make_pipe(inherited=True):
    wrapper=Wrapper(Raw())
    class Pipe:
        scheduler=wrapper
    pipe=Pipe()
    if not inherited: pipe.scheduler=wrapper
    return pipe,wrapper,wrapper.module


def completed_receipt():
    pipe,wrapper,raw=make_pipe()
    with patch.object(common,'describe_solver',return_value=metadata()):
        with common.select(pipe,runtime) as row:
            for unused in range(28): pipe.scheduler.step(object())
    return row


class CommonSolverTests(unittest.TestCase):
    def test_selects_actual_module_counts_once_and_restores_attribute_presence(self):
        for inherited in (False,True):
            pipe,wrapper,raw=make_pipe(inherited)
            value=object()
            with patch.object(common,'describe_solver',return_value=metadata()):
                with common.select(pipe,runtime) as row:
                    self.assertIs(pipe.scheduler,raw)
                    for unused in range(28): self.assertIs(pipe.scheduler.step(value,return_dict=False),value)
            self.assertIs(pipe.scheduler,wrapper)
            self.assertEqual('scheduler' in vars(pipe),not inherited)
            self.assertNotIn('step',vars(raw))
            self.assertEqual(raw.calls,28)
            self.assertEqual(raw.index,28)  # Ordinary inference state is deliberately retained.
            self.assertTrue(common.valid_receipt(row))
            self.assertFalse(row['scheduler_state_restored'])

    def test_exact_instance_step_override_restored_on_success_and_error(self):
        pipe,wrapper,raw=make_pipe(False)
        original=raw.step;raw.step=original
        with patch.object(common,'describe_solver',return_value=metadata()):
            with common.select(pipe,runtime) as row:
                for unused in range(28):raw.step(object())
        self.assertIs(vars(raw)['step'],original)
        raw.fail=True
        with patch.object(common,'describe_solver',return_value=metadata()):
            with self.assertRaisesRegex(RuntimeError,'real step failed'):
                with common.select(pipe,runtime) as failed: raw.step(object())
        self.assertIs(pipe.scheduler,wrapper)
        self.assertIs(vars(raw)['step'],original)
        self.assertFalse(failed['complete'])
        self.assertEqual((failed['started_calls'],failed['returned_calls']),(1,0))
        self.assertTrue(failed['pipeline_scheduler_identity_restored'])

    def test_wrong_runtime_or_call_count_cannot_complete(self):
        for field,value in [('pp_degree',2),('cfg_degree',2),('sp_degree',1),('dp_degree',2),('pp_degree',True)]:
            state=runtime();setattr(state.parallel_config,field,value)
            with self.assertRaises(ValueError,msg=field):common.runtime_description(state)
        for patched in (True,0):
            state=runtime();state.patch_mode=patched
            with self.assertRaises(ValueError):common.runtime_description(state)
        for count in (0,27,29):
            pipe,wrapper,raw=make_pipe()
            with patch.object(common,'describe_solver',return_value=metadata()):
                with self.assertRaisesRegex(ValueError,'exactly 28'):
                    with common.select(pipe,runtime) as row:
                        for unused in range(count):raw.step(object())
            self.assertIs(pipe.scheduler,wrapper);self.assertFalse(row['complete'])

    def test_sources_config_and_runtime_receipts_fail_closed(self):
        self.assertTrue(common.valid_description(metadata()))
        for change in ('source','config','type','calls','restore','state','runtime'):
            row=deepcopy(completed_receipt())
            if change=='source':row['solver']['selected_step']['file_sha256']='0'*64
            if change=='config':row['solver']['configuration']['shift']=4.
            if change=='type':row['solver']['selected_type']=common.WRAPPER_TYPE
            if change=='calls':row['returned_calls']=27
            if change=='restore':row['pipeline_scheduler_identity_restored']=False
            if change=='state':row['scheduler_state_restored']=True
            if change=='runtime':row['runtime']['patch_mode']=True
            self.assertFalse(common.valid_receipt(row),change)

    def test_only_completed_within_sp2_equivalence_is_reported(self):
        from test_20261104_external_tetriserve_sp2_diagnostic import receipt
        records=receipt()
        for row in records:row['common_solver']=completed_receipt()
        # Deliberately retain a real-shaped SP1 mismatch; it does not invalidate
        # the narrowly named normal-common/full/chunk equality claim.
        records[0]['comparison_inputs']['sp1_reference']['latents']['sha256']='b'*64
        for name,pair in records[0]['comparisons'].items():
            if name.startswith('sp1_reference_'):
                pair['latents'].update(passed=False,bitwise_equal=False,max_abs_error=2.,rmse=.2,relative_l2_error=.1)
        self.assertTrue(diagnostic.assess(records))
        result=common.assess(records,diagnostic.assess(records))
        self.assertTrue(result['common_solver_equivalence_passed'])
        self.assertFalse(result['qualification_eligible'])
        self.assertFalse(all(v['passed'] for p in records[0]['comparisons'].values() for v in p.values()))
        for change in ('incomplete','missing_receipt','output','missing_output'):
            rows=deepcopy(records);completed=True
            if change=='incomplete':completed=False
            if change=='missing_receipt':del rows[1]['common_solver']
            if change=='output':rows[1]['outputs']['normal_sp2']['latents']['sha256']='c'*64
            if change=='missing_output':del rows[0]['outputs']
            self.assertFalse(common.assess(rows,completed)['common_solver_equivalence_passed'],change)

    def test_experimental_flag_rejects_old_output_observer_modes_before_binding(self):
        required=['plan','--prior-result','/unused','--prior-summary-sha256','a'*64,
            '--prior-numerics-sha256','b'*64,'--output-dir','/unused-output','--normal-sp2-common-solver']
        for extra in (['--trace-reference-summary','/unused-reference'],['--solver-counterfactual']):
            with patch.object(diagnostic,'bind_prior') as binding,redirect_stdout(StringIO()):
                self.assertEqual(diagnostic.main(required+extra),1)
            binding.assert_not_called()


if __name__=='__main__':unittest.main()
