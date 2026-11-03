"""Gate actual diagnostics on completed evidence, never process success alone."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import external_tetriserve_logic as logic


def evidence():
    records = []
    for batch in (1, 2):
        row = dict(sub_req_ids=list(range(1, batch + 1)), ranks=[0, 1], start=10, end=20,
                   steps=10, batch=batch, prompts=['fox','hare'][:batch], origin_steps=[28]*batch,
                   final=[False]*batch)
        records.append(dict(batch=batch, first=[dict(row, start=0, end=10)], second=[row],
                            rounds=[[dict(row,start=0,end=10)],[row],[dict(row,start=20,end=28,steps=8,final=[True]*batch)]],
                            terminal_items=batch,tracker_remaining=0,complete_steps_covered=True,
                            completed_steps={str(sid):10 for sid in row['sub_req_ids']},
                            cursor_contiguous=True, batch_metadata_aligned=True))
    return dict(schema='ditserve.tetriserve-native-scheduler-logic.v1',
                costs_are_measurements=False, records=records)


def gloo_execution():
    return dict(returncode=0, timed_out=False, tests=dict(
        selected=['module.Class.' + name for name in logic.CRITICAL_TESTS] + ['unit.a','unit.b','unit.c'],
        tests_run=5, successful=True, skipped=[], errors=[], failures=[]))


class TetriLogicTests(unittest.TestCase):
    def test_zero_exit_without_real_continuation_or_with_bad_cursor_cannot_pass(self):
        execution = dict(returncode=0, timed_out=False)
        self.assertTrue(logic.assess_scheduler(execution,evidence())['passed'])
        self.assertFalse(logic.assess_scheduler(execution,None)['passed'])
        for mutation in ('missing','replay','omitted','batch','bad_prompts','unfinished','wrong_split_prompt','third_origin','third_steps','third_final','costs'):
            value = evidence()
            if mutation == 'third_origin': value['records'][0]['rounds'][2][0]['origin_steps'] = [27]
            if mutation == 'third_steps': value['records'][0]['rounds'][2][0]['steps'] = 7
            if mutation == 'third_final': value['records'][0]['rounds'][2][0]['final'] = [False]
            if mutation == 'unfinished': value['records'][0]['rounds'] = value['records'][0]['rounds'][:2]
            if mutation == 'wrong_split_prompt': value['records'][1]['rounds'][2][0]['prompts'] = ['fox','fox']
            if mutation == 'missing': value['records'][0]['second'] = []
            if mutation == 'replay': value['records'][0]['second'][0]['start'] = 0
            if mutation == 'omitted': value['records'][1]['second'][0]['sub_req_ids'] = [1]
            if mutation == 'bad_prompts': value['records'][1]['second'][0]['prompts'] = ['one']
            if mutation == 'batch': value['records'][1]['batch_metadata_aligned'] = False
            if mutation == 'costs': value['costs_are_measurements'] = True
            self.assertFalse(logic.assess_scheduler(execution,value)['passed'],mutation)

    def test_gloo_skips_failures_or_missing_required_test_never_pass(self):
        self.assertTrue(logic.assess_gloo(gloo_execution())['passed'])
        for mutation in ('skip','failure','error','missing','exit','timeout'):
            value = gloo_execution()
            if mutation == 'skip': value['tests']['skipped'] = [{'test':'gloo','reason':'no Torch'}]
            if mutation == 'failure': value['tests']['failures'] = [{'test':'gloo','traceback':'failed'}]
            if mutation == 'error': value['tests']['errors'] = [{'test':'gloo','traceback':'missing package'}]
            if mutation == 'third_origin': value['records'][0]['rounds'][2][0]['origin_steps'] = [27]
            if mutation == 'third_steps': value['records'][0]['rounds'][2][0]['steps'] = 7
            if mutation == 'third_final': value['records'][0]['rounds'][2][0]['final'] = [False]
            if mutation == 'unfinished': value['records'][0]['rounds'] = value['records'][0]['rounds'][:2]
            if mutation == 'wrong_split_prompt': value['records'][1]['rounds'][2][0]['prompts'] = ['fox','fox']
            if mutation == 'missing': value['tests']['selected'] = ['unrelated']*5
            if mutation == 'exit': value['returncode'] = 1
            if mutation == 'timeout': value['timed_out'] = True
            self.assertFalse(logic.assess_gloo(value)['passed'],mutation)

    def test_real_subprocess_failure_and_structured_output_are_preserved(self):
        result = logic.run_child([sys.executable,'-B','-c',
            'import sys; print(\'DITSERVE_TEST_RESULT={"successful": false}\'); print("actual error",file=sys.stderr); sys.exit(2)'],
            os.environ.copy(),30)
        self.assertEqual(result['returncode'],2)
        self.assertIn('actual error',result['stderr_tail'])
        self.assertFalse(result['tests']['successful'])
        self.assertFalse(result['timed_out'])

    def test_source_mismatch_writes_failure_and_preserves_previous_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, shared, tests = [root/name for name in ('source','shared','tests')]
            for path in (source,shared,tests): path.mkdir()
            test = tests/logic.BRIDGE_TEST_NAME
            test.write_text('# source fixture\n')
            args = SimpleNamespace(output=root/'result',source_root=source,source_manifest=root/'manifest.json',
                source_manifest_sha256='0'*64, shared_python_root=shared,
                bridge_test_file=test,bridge_test_sha256='0'*64,timeout_seconds=30)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(logic.execute(args),1)
            summary = json.loads((args.output/'summary.json').read_bytes())
            self.assertFalse(summary['logic_gate_passed'])
            self.assertFalse(summary['qualification_eligible'])
            self.assertEqual(summary['scheduler']['status'],'blocked_source_validation')
            before = (args.output/'summary.json').read_bytes()
            with self.assertRaises(FileExistsError): logic.execute(args)
            self.assertEqual(before,(args.output/'summary.json').read_bytes())
            args.output = source/'must-not-exist'
            with self.assertRaisesRegex(ValueError,'disjoint'): logic.execute(args)
            self.assertFalse(args.output.exists())

    def test_scheduler_failure_does_not_prevent_independent_gloo_attempt(self):
        # Unit-test orchestration only. No fabricated test result is presented as
        # an actual Gloo/model run; the production path always uses real children.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('source','shared','tests'): (root/name).mkdir()
            test = root/'tests'/logic.BRIDGE_TEST_NAME
            test.write_text('# explicit unit fixture\n')
            args = SimpleNamespace(output=root/'result',source_root=root/'source',source_manifest=root/'manifest.json',
                source_manifest_sha256='0'*64,shared_python_root=root/'shared',
                bridge_test_file=test,bridge_test_sha256=logic.sha(test),timeout_seconds=30)
            derived = dict(root=str(args.output/'derived'),manifest_sha256='1'*64)
            calls = []
            def run(argv,env,timeout):
                calls.append(argv)
                return dict(returncode=1,timed_out=False,stderr_tail='real path unavailable')
            with patch.object(logic,'verify_shared_sources',return_value=root/'shared'), \
                 patch.object(logic.port,'apply',return_value=derived), \
                 patch.object(logic,'run_child',side_effect=run), redirect_stdout(io.StringIO()):
                self.assertEqual(logic.execute(args),1)
            self.assertEqual(len(calls),2)
            result = json.loads((args.output/'summary.json').read_bytes())
            self.assertEqual(result['scheduler']['status'],'executed')
            self.assertEqual(result['gloo']['status'],'executed')
            self.assertFalse(result['logic_gate_passed'])
            self.assertFalse(result['eligible_for_next_gpu_migration_check'])


if __name__ == '__main__':
    unittest.main()
