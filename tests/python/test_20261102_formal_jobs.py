"""Formal operator specs pin scientific inputs before platform submission."""
import importlib.util
import json
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(os.environ.get('DITSERVE_KAIROS_RUNTIME_ROOT', ROOT.parent / '.local/ditserve-kairos-rewrite-runtime'))
SOURCE = LOCAL / 'tools/generate_formal_batch.py'


@unittest.skipUnless(SOURCE.is_file(), 'operator tooling not installed')
class FormalJobTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('formal_jobs_test', SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.protocol = json.loads((ROOT / 'configs/benchmarks/formal-mixed-v1.json').read_text())

    def test_contract_pins_runtime_and_full_test_commit(self):
        jobs = self.module.prepare('a'*40, self.protocol, 'contract', date='20261102')
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        env = {item['name']: item['value'] for item in job['env']}
        self.assertEqual(job['gpus_per_worker'], 1)
        self.assertEqual(env['DITSERVE_EXECUTION_IMAGE'], job['image'])
        self.assertIn('@sha256:', job['image'])
        self.assertEqual(env['DITSERVE_TEST_REVISION'], 'a'*40)
        self.assertTrue(env['DITSERVE_JOB_SPEC_PATH'].endswith('/'+job['name']+'.json'))
        self.assertEqual(env['DITSERVE_KAIROS_RUNTIME_ROOT'], '/shared/data/DITSERVE/scripts/releases/aaaaaaa')
        self.assertEqual(job['command'], 'bash ' + env['DITSERVE_KAIROS_RUNTIME_ROOT'] + '/scripts/kairos_targeted_tests.sh')

    def test_p0_parallel_jobs_have_independent_identity_and_require_manifests(self):
        with self.assertRaises(ValueError):
            self.module.prepare('a'*40, self.protocol, 'p0', date='20261102')
        jobs = self.module.prepare('a'*40, self.protocol, 'p0', date='20261102', gpus=4,
                                   model_manifests='/shared/data/DITSERVE/manifests/models.json')
        self.assertEqual(len({j['name'] for j in jobs}), 2)
        for job in jobs:
            env = {item['name']: item['value'] for item in job['env']}
            self.assertEqual(env['DITSERVE_BENCHMARK_MODE'], 'formal')
            self.assertEqual(env['DITSERVE_STAGE_ACTION'], 'preemption')
            self.assertEqual(job['gpus_per_worker'], 4)
            self.assertLessEqual(len(job['name']), 52)

    def test_manifest_path_cannot_escape_project(self):
        for path in ('relative.json', '/shared/data/OTHER/models.json', '/shared/data/DITSERVE/../OTHER/models.json'):
            with self.assertRaises(ValueError):
                self.module.prepare('a'*40, self.protocol, 'p0', date='20261102', model_manifests=path)

    def test_p0_cases_can_fill_two_independent_two_gpu_allocations(self):
        jobs = [self.module.prepare('a'*40, self.protocol, 'p0', date='20261103', gpus=2,
                model='wan21', p0_case=case, model_manifests='/shared/data/DITSERVE/models/manifests.json')[0]
                for case in ('contention', 'no_preemption')]
        self.assertEqual(len({job['name'] for job in jobs}), 2)
        for job, case in zip(jobs, ('contention', 'no_preemption')):
            env = {item['name']: item['value'] for item in job['env']}
            self.assertEqual(env['DITSERVE_P0_CASE'], case)
            self.assertEqual(job['gpus_per_worker'], 2)
            self.assertEqual(env['DITSERVE_STAGE_ACTION'], 'preemption' if case == 'contention' else 'no_preemption')


if __name__ == '__main__':
    unittest.main()
