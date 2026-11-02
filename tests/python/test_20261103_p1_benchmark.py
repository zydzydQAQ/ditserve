"""P1 uses formal shapes without changing P0 or leaking test prompts."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import formal_benchmark as formal
import p1_benchmark as p1
import test_20261102_formal_benchmark as fixtures
from ditserve.evaluation.common import canonical_hash, read_json, seal


class P1BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.FormalBenchmarkTest.setUpClass()
        cls.fixture = fixtures.FormalBenchmarkTest()
        cls.plan = read_json(p1.DEFAULT_PLAN)

    def build(self, role='contention', **kwargs):
        f = self.fixture
        return p1.case_config(f.p, kwargs.pop('plan', self.plan), 'wan21-1', role, 202,
            f.uuids, kwargs.pop('bundle', f.bundle()), f.pools, f.manifests, **kwargs)

    def test_formal_shapes_lazy_and_slos_with_development_prompts(self):
        cfg = self.build()
        phase, meta = cfg['phases'][0], cfg['benchmark']
        trace = phase['matrix']['/workload'][0]
        self.assertEqual(len(trace), 40)
        self.assertEqual({row['class'] for row in trace}, {'wan21-720p', 'sd35-1024'})
        self.assertEqual(meta['formal_protocol']['role'], 'p1_contention')
        self.assertEqual(meta['p1']['plan_sha256'], canonical_hash(self.plan))
        self.assertNotIn('p0_diagnostic_controls', meta)
        self.assertEqual(phase['options']['forms'], ['cfg2'])
        self.assertIs(phase['options']['lazy_preemption'], True)
        for row in trace:
            self.assertEqual(formal.input_shape(row['spec']), formal.shape_spec(self.fixture.p, row['class']))
            self.assertEqual(row['spec']['output_mode'] if 'output_mode' in row['spec'] else row.get('output_mode'), 'digest')
            self.assertAlmostEqual(row['deadline_ms'] - row['arrival_ms'], row['service_ms'] * (1.5 if row['kind'] == 'video' else 2.5))
            dev = formal.split_pool(self.fixture.pools[row['kind']], self.fixture.p, row['kind'], 'development')
            test = formal.split_pool(self.fixture.pools[row['kind']], self.fixture.p, row['kind'], 'test')
            self.assertIn(row['spec']['prompt'], {item['prompt'] for item in dev['samples']})
            self.assertNotIn(row['spec']['prompt'], {item['prompt'] for item in test['samples']})

    def test_idle_uses_same_shape_and_preserves_zero_stop_requirement(self):
        cfg = self.build('no_preemption')
        rows = cfg['phases'][0]['matrix']['/workload'][0]
        self.assertEqual(len(rows), 40)
        self.assertEqual({row['class'] for row in rows}, {'wan21-720p'})
        self.assertEqual(cfg['benchmark']['load'], 0.1)
        self.assertIs(cfg['benchmark']['p1']['controls']['require_observed_stops'], False)

    def test_cannot_remove_hard_shapes_change_slo_or_disable_lazy(self):
        mutations = [lambda p: p['shape_pairs'].pop(),
                     lambda p: p['roles']['contention'].__setitem__('image_slo_scale', 1.1),
                     lambda p: p['roles']['contention'].__setitem__('lazy_preemption', False),
                     lambda p: p.__setitem__('prompt_split', 'test'),
                     lambda p: p.__setitem__('seeds', [201, 201, 202]),
                     lambda p: p.__setitem__('protocol_sha256', '0' * 64)]
        for mutate in mutations:
            plan = copy.deepcopy(self.plan)
            mutate(plan)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.build(plan=plan)

    def test_formal_pair_requires_its_own_exact_allocation_profile(self):
        bundle = self.fixture.bundle()
        bundle['classes'].pop('wan21-720p')
        with self.assertRaisesRegex(ValueError, 'own measured calibration'):
            self.build(bundle=seal(bundle))
        bundle = self.fixture.bundle()
        bundle['gpu_uuids'].reverse()
        with self.assertRaisesRegex(ValueError, 'exact allocation'):
            self.build(bundle=seal(bundle))

    def test_configuration_generation_does_not_mutate_inputs(self):
        before = copy.deepcopy(self.plan)
        first = self.build()
        self.assertEqual(first, self.build())
        self.assertEqual(self.plan, before)

    def test_formal_sparse_trace_is_not_truncated_by_development_watchdog(self):
        bundle = self.fixture.bundle()
        bundle['classes']['wan21-720p']['cost']['step_compute_ms'] = 10000
        cfg = self.build('no_preemption', bundle=seal(bundle))
        planning = cfg['benchmark']['p1']['time_planning']
        self.assertGreater(cfg['budget']['watchdog_seconds'], 14400)
        self.assertGreater(cfg['budget']['watchdog_seconds'], 2 * planning['trace_horizon_ms'] / 1000)
        self.assertEqual(cfg['budget']['limit_gpu_seconds'], cfg['budget']['watchdog_seconds'] * len(self.fixture.uuids))


if __name__ == '__main__':
    unittest.main()
