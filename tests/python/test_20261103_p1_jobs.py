"""P1 jobs retain immutable identities and require the P0 action gate."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path(os.environ.get('DITSERVE_KAIROS_RUNTIME_ROOT', ROOT.parent / '.local/ditserve-kairos-rewrite-runtime'))
SOURCE = RUNTIME / 'tools/generate_p1_batch.py'


@unittest.skipUnless(SOURCE.is_file(), 'operator tooling not installed')
class P1JobsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SOURCE.parent))
        try:
            spec = importlib.util.spec_from_file_location('p1_jobs_20261103', SOURCE)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        finally:
            sys.path.pop(0)
        cls.protocol = json.loads((ROOT / 'configs/benchmarks/formal-mixed-v1.json').read_bytes())
        cls.plan = json.loads((ROOT / 'configs/benchmarks/p1-qualification-v1.json').read_bytes())

    def prepare(self, **kwargs):
        return self.module.prepare_p1('a' * 40, self.protocol, self.plan, date='20261103', gpus=2,
            model_manifests='/shared/data/DITSERVE/models/frozen-mapping.json', **kwargs)

    def test_all_pairs_and_roles_are_distinct_p0_gated_jobs(self):
        specs = self.prepare()
        self.assertEqual(len(specs), 8)
        self.assertEqual(len({spec['name'] for spec in specs}), 8)
        for spec in specs:
            env = {item['name']: item['value'] for item in spec['env']}
            self.assertLessEqual(len(spec['name']), 52)
            self.assertEqual(env['DITSERVE_STAGE_ACTION'], 'p1_diagnostics')
            self.assertEqual(env['DITSERVE_FORMAL_ACTION'], 'p1')
            self.assertEqual(env['DITSERVE_P1_PLAN_SHA256'], self.module.canonical_hash(self.plan))
            self.assertEqual(env['DITSERVE_RUN_ID'], spec['name'])
            self.assertIn('/releases/aaaaaaa/', spec['command'])
            self.assertEqual(spec['gpus_per_worker'], 2)

    def test_selection_cannot_silently_replace_unknown_pair_or_role(self):
        self.assertEqual(len(self.prepare(pair_id='wan22-1', role='no_preemption')), 1)
        for arguments in ({'pair_id': 'wan22-dev'}, {'role': 'calibration'}):
            with self.assertRaisesRegex(ValueError, 'unknown P1'):
                self.prepare(**arguments)


if __name__ == '__main__':
    unittest.main()
