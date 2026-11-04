"""CPU contracts for the source-bound, candidate-only attn2 local initialization."""
import ast
from contextlib import ExitStack, redirect_stdout, contextmanager
from copy import deepcopy
import hashlib
from io import StringIO
import inspect
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import external_sd35_parallel_wrapper_check as candidate
import external_sd35_cross_sp_check as cross

FAKE_SOURCE='''class xFuserJointAttnProcessor2_0:
    def __call__(self, encoder_hidden_states=None):
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            attention_kwargs = None
            if encoder_hidden_states is not None:
                joint_strategy = "rear"
            return USP(encoder_hidden_states, joint_strategy=joint_strategy, attention_kwargs=attention_kwargs)
        return USP(encoder_hidden_states, joint_strategy="sp1")
'''


class Fixture:
    def __init__(self,directory):
        self.path=Path(directory)/'processor.py';self.path.write_text(FAKE_SOURCE)
        self.state=dict(sp=2,fail=False);self.calls=[];self.output=object()
        def usp(*args,**kwargs):
            self.calls.append((args,kwargs))
            if self.state['fail']:raise RuntimeError('actual USP failure')
            return self.output
        namespace=dict(__name__=candidate.PROCESSOR_TYPE.rsplit('.',1)[0],HAS_LONG_CTX_ATTN=True,
                       get_sequence_parallel_world_size=lambda:self.state['sp'],USP=usp)
        exec(compile(FAKE_SOURCE,str(self.path),'exec'),namespace)
        self.cls=namespace['xFuserJointAttnProcessor2_0'];self.original=self.cls.__call__
        self.text=textwrap.dedent(inspect.getsource(self.original))
        self.derived=self.text.replace('        attention_kwargs = None\n','        joint_strategy = None\n        attention_kwargs = None\n')
    def gates(self):
        stack=ExitStack()
        stack.enter_context(patch.dict(candidate.SOURCE_HASHES,attention=hashlib.sha256(FAKE_SOURCE.encode()).hexdigest()))
        stack.enter_context(patch.object(candidate,'JOINT_METHOD_SHA256',hashlib.sha256(self.text.encode()).hexdigest()))
        stack.enter_context(patch.object(candidate,'JOINT_DERIVED_SHA256',hashlib.sha256(self.derived.encode()).hexdigest()))
        return stack


def receipt():
    return dict(original_source=dict(file_sha256=candidate.SOURCE_HASHES['attention']),
        derived_source='<ditserve-sd35-joint-strategy-default:'+candidate.JOINT_DERIVED_SHA256+'>',
        original_method_sha256=candidate.JOINT_METHOD_SHA256,derived_method_sha256=candidate.JOINT_DERIVED_SHA256,
        original_ast_sha256=candidate.JOINT_ORIGINAL_AST_SHA256,restored_ast_sha256=candidate.JOINT_ORIGINAL_AST_SHA256,
        derived_ast_sha256=candidate.JOINT_DERIVED_AST_SHA256,insertion_line_in_method=74,
        inserted_statement='joint_strategy = None',ast_statements_inserted=1,ast_statements_removed=0,
        other_ast_changes=False,complete=True,method_identity_restored=True,source_files_modified=False)


class JointStrategyTests(unittest.TestCase):
    def test_real_unbound_local_path_is_fixed_with_exact_once_usp_and_finally_restoration(self):
        with TemporaryDirectory() as directory:
            f=Fixture(directory);processor=f.cls()
            with self.assertRaises(UnboundLocalError):processor()
            self.assertEqual(f.calls,[])
            with f.gates():
                with candidate.joint_strategy_default(processor_class=f.cls) as row:
                    self.assertIs(processor(),f.output)
                    self.assertIsNone(f.calls[0][1]['joint_strategy'])
                    self.assertEqual(len(f.calls),1)
                    self.assertIsNot(f.cls.__call__,f.original)
            self.assertIs(f.cls.__call__,f.original)
            self.assertTrue(row['complete']);self.assertTrue(row['method_identity_restored'])
            self.assertFalse(row['source_files_modified'])
            self.assertEqual(f.path.read_text(),FAKE_SOURCE)
            with self.assertRaises(UnboundLocalError):processor()

    def test_text_joint_and_sp1_branches_preserve_arguments_and_exact_return_identity(self):
        with TemporaryDirectory() as directory:
            f=Fixture(directory);processor=f.cls();text=object()
            for sp,encoder in ((2,text),(1,None),(1,text)):
                f.state['sp']=sp;f.calls.clear()
                expected=processor(encoder);original_call=f.calls.pop()
                with f.gates(),candidate.joint_strategy_default(processor_class=f.cls):
                    self.assertIs(processor(encoder),expected)
                    self.assertEqual(f.calls,[original_call])
                self.assertIs(f.cls.__call__,f.original)

    def test_actual_usp_exception_propagates_and_restores_exact_class_attribute(self):
        with TemporaryDirectory() as directory:
            f=Fixture(directory);f.state['fail']=True
            with f.gates():
                with self.assertRaisesRegex(RuntimeError,'actual USP failure'):
                    with candidate.joint_strategy_default(processor_class=f.cls) as row:f.cls()()
            self.assertIs(f.cls.__call__,f.original)
            self.assertFalse(row['complete']);self.assertTrue(row['method_identity_restored'])
            self.assertEqual(len(f.calls),1)

    def test_exact_one_statement_ast_diff_and_text_hashes(self):
        with TemporaryDirectory() as directory:
            f=Fixture(directory)
            with f.gates():derived,row=candidate.derive_joint_method(f.text)
            self.assertEqual(derived,f.derived)
            tree=ast.parse(derived);del candidate.joint_branch(tree).body[0]
            self.assertEqual(candidate.canonical_ast(tree),candidate.canonical_ast(ast.parse(f.text)))
            self.assertEqual(row['ast_statements_inserted'],1)
            self.assertEqual(row['original_ast_sha256'],row['restored_ast_sha256'])
            self.assertNotEqual(row['original_ast_sha256'],row['derived_ast_sha256'])
            self.assertEqual(row['inserted_statement'],'joint_strategy = None')

    def test_wrong_source_method_or_derived_text_fails_before_installation(self):
        with TemporaryDirectory() as directory:
            f=Fixture(directory)
            for failure in ('file','method','derived'):
                with f.gates(),ExitStack() as stack:
                    if failure=='file':stack.enter_context(patch.dict(candidate.SOURCE_HASHES,attention='0'*64))
                    if failure=='method':stack.enter_context(patch.object(candidate,'JOINT_METHOD_SHA256','0'*64))
                    if failure=='derived':stack.enter_context(patch.object(candidate,'JOINT_DERIVED_SHA256','0'*64))
                    with self.assertRaises(ValueError):
                        with candidate.joint_strategy_default(processor_class=f.cls):self.fail('must not enter')
                self.assertIs(f.cls.__call__,f.original)
                self.assertEqual(f.calls,[])

    def test_reentrant_patch_is_rejected_and_outer_patch_restored(self):
        with TemporaryDirectory() as directory:
            f=Fixture(directory)
            with f.gates():
                with candidate.joint_strategy_default(processor_class=f.cls):
                    with self.assertRaises((OSError,TypeError,ValueError)):
                        with candidate.joint_strategy_default(processor_class=f.cls):self.fail('no nested adaptation')
            self.assertIs(f.cls.__call__,f.original)

    def test_canonical_ast_ignores_only_empty_type_parameters(self):
        first=ast.parse('def f(x): return x')
        second=deepcopy(first);second.body[0].type_params=[]
        self.assertEqual(candidate.canonical_ast(first),candidate.canonical_ast(second))
        second.body[0].body[0].value=ast.Constant(value=None)
        self.assertNotEqual(candidate.canonical_ast(first),candidate.canonical_ast(second))

    def test_persisted_adaptation_receipt_rejects_incomplete_or_other_changes(self):
        self.assertTrue(candidate.valid_joint_strategy(receipt()))
        for key,value in [('original_method_sha256','0'*64),('derived_method_sha256','0'*64),
                          ('original_ast_sha256','0'*64),('derived_ast_sha256','0'*64),
                          ('restored_ast_sha256','0'*64),('insertion_line_in_method',75),
                          ('ast_statements_inserted',2),('ast_statements_inserted',True),('other_ast_changes',True),
                          ('complete',False),('method_identity_restored',False),('source_files_modified',True)]:
            row=receipt();row[key]=value
            self.assertFalse(candidate.valid_joint_strategy(row),key)

    def test_candidate_first_error_preserves_completed_stage_and_active_calls(self):
        distributed=SimpleNamespace(barrier=lambda **kwargs:None)
        pipeline=SimpleNamespace(pipe=SimpleNamespace(transformer=object()),runtime=SimpleNamespace(
            pool=SimpleNamespace(activate=lambda *args,**kwargs:None),state=object(),bind_global=lambda:None))
        progress=SimpleNamespace(update=lambda stage:None)
        @contextmanager
        def observe(module):
            yield [dict(name='actual_wrapper',calls=1)]
        record={}
        with patch.dict(sys.modules,{'torch':SimpleNamespace(distributed=distributed),'torch.distributed':distributed}),ExitStack() as stack:
            stack.enter_context(patch.object(candidate,'backend_metadata',return_value={'configured':'SDPA_MATH'}))
            stack.enter_context(patch.object(candidate,'convert',return_value={'complete':True}))
            stack.enter_context(patch.object(candidate,'identities',return_value={}))
            stack.enter_context(patch.object(candidate,'observe_modules',side_effect=observe))
            stack.enter_context(patch.object(cross,'observe_route',side_effect=[({'tensor':'D'},{'route':'sp1_round'}),RuntimeError('actual E failure')]))
            stack.enter_context(patch.object(cross,'compare_boundaries',return_value={'comparison':'D'}))
            with self.assertRaisesRegex(RuntimeError,'actual E failure'):
                cross.run_parallel_candidate(pipeline,None,None,0,None,None,{'sp1_native':{}},progress,result=record)
        self.assertEqual(record['stage'],'converted_sp2_enter')
        self.assertEqual(record['routes'],{'sp1_round':{'route':'sp1_round'}})
        self.assertEqual(record['original_sp1_vs_converted_sp1'],{'comparison':'D'})
        self.assertEqual(set(record['module_calls']),{'sp1_round','sp2_round'})
        self.assertEqual(record['conversion'],{'complete':True})
        self.assertNotIn('post_inference_weights',record)

    def test_cli_cannot_enable_adaptation_outside_candidate(self):
        argv=['plan','--prior-result','/unused','--prior-summary-sha256','a'*64,
              '--prior-numerics-sha256','b'*64,'--output-dir','/unused-output',
              '--sp2-reference-summary','/unused-r5','--sp2-reference-traces','/rank0','/rank1',
              '--joint-strategy-default']
        with patch.object(cross.base,'bind_prior') as binding,redirect_stdout(StringIO()):
            self.assertEqual(cross.main(argv),1)
        binding.assert_not_called()


if __name__=='__main__':unittest.main()
