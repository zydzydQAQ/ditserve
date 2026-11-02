"""CUDA success alone must never publish a full runtime contract receipt."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import stage_gates

LOCAL = Path(os.environ.get('DITSERVE_KAIROS_RUNTIME_ROOT',
    Path(__file__).resolve().parents[3] / '.local/ditserve-kairos-rewrite-runtime'))


@unittest.skipUnless((LOCAL / 'scripts/kairos_contract_receipt.py').is_file(), 'local operator scripts')
class ContractReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('contract_receipt_test', LOCAL / 'scripts/kairos_contract_receipt.py')
        cls.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helper)

    def prepare(self, root, log):
        build = dict(commit='a'*40, core_sha256='c'*64)
        protocol = dict(schema='ditserve.formal-protocol.v1', p0_required_cases=[dict(model='wan21', shape_id='development')])
        job = dict(name='contract-fixture', image='registry/ditserve@sha256:'+'d'*64,
            workers=1, gpus_per_worker=1, mount_shared=True,
            env=[dict(name='DITSERVE_BUILD', value=build['commit']),
                 dict(name='DITSERVE_TEST_REVISION', value=build['commit']),
                 dict(name='DITSERVE_PROTOCOL_SHA256', value=stage_gates.canonical_hash(protocol))])
        def write(name, data):
            (root/name).write_text(json.dumps(data))
        write('BUILD_INFO.json', build)
        write('protocol.json', protocol)
        write('job.json', job)
        smoke = dict(schema='ditserve.runtime-smoke.v1', status='passed', **build,
            python_sources={'ditserve/runtime/worker.py': 'e'*64, 'xfuser/__init__.py': 'f'*64},
            test_revision=build['commit'], devices=1, sum=524800, matmul_value=32,
            run_id=job['name'], execution_image=job['image'], image_digest='sha256:'+'d'*64,
            image_identity_source='submitted_job_spec_digest_pin',
            image_job_spec_sha256=hashlib.sha256((root/'job.json').read_bytes()).hexdigest())
        write('runtime-smoke.json', smoke)
        (root/'contract.log').write_text(log)

    def test_complete_contract_enables_calibration_but_not_p0_or_main(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root, 'Ran 120 tests in 1.0s\n\nOK\nRan 40 tests in 1.0s\n\nOK\n')
            result = self.helper.finalize(root, root/'protocol.json', root/'job.json')
            self.assertEqual(result['stage'], 'CONTRACT')
            receipt = stage_gates.verify_receipt(result['receipt'], required_stage='CONTRACT', action='shape_calibration')
            self.assertEqual(receipt['stages']['P0']['status'], 'blocked')
            with self.assertRaises(ValueError):
                stage_gates.verify_receipt(result['receipt'], required_stage='P0')
            with self.assertRaises(FileExistsError):
                self.helper.finalize(root, root/'protocol.json', root/'job.json')

    def test_cuda_only_partial_skipped_and_failed_tests_never_publish(self):
        logs = ['CUDA passed\n', 'Ran 120 tests in 1.0s\n\nOK\n',
                'Ran 120 tests in 1.0s\n\nOK\nRan 40 tests in 1.0s\n\nOK (skipped=1)\n',
                'Ran 120 tests in 1.0s\n\nOK\nRan 40 tests in 1.0s\n\nFAILED (errors=1)\n']
        for log in logs:
            with self.subTest(log=log), TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.prepare(root, log)
                with self.assertRaisesRegex(ValueError, 'contract did not qualify'):
                    self.helper.finalize(root, root/'protocol.json', root/'job.json')
                self.assertFalse((root/'contract-receipt.json').exists())
                self.assertFalse((root/'contract-evidence.json').exists())

    def test_changed_job_image_cannot_be_published(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root, 'Ran 1 test in 1s\n\nOK\nRan 1 test in 1s\n\nOK\n')
            spec = json.loads((root/'job.json').read_text())
            spec['image'] = 'registry/ditserve:mutable-tag'
            (root/'job.json').write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'digest-pinned'):
                self.helper.finalize(root, root/'protocol.json', root/'job.json')


if __name__ == '__main__':
    unittest.main()
