import copy
import importlib.util
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import formal_benchmark as formal
from ditserve.evaluation.common import canonical_hash, seal


class FormalBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = formal.protocol()
        cls.uuids = [f'GPU-{i}' for i in range(8)]
        cls.pools = {kind: seal(dict(kind=expected['kind'], source=expected,
            samples=[dict(sample_id=f'{kind}-{i}', prompt=f'{kind} sample {i}') for i in range(1200)]))
            for kind, expected in cls.p['datasets'].items()}
        cls.manifests = {model: dict(path=f'/shared/data/DITSERVE/models/{model}/manifest.json', sha256='a'*64)
                         for model in ['wan21', 'sd35']}

    def bundle(self, split='test'):
        entries = {}
        scope = 'formal' if split == 'test' else 'development'
        for sid, shape in self.p['shapes'].items():
            if shape['scope'] != scope or shape['model'] not in ('wan21', 'sd35'):
                continue
            if shape['kind'] == 'video':
                cost = dict(kind='video', steps=shape['steps'], blocks_per_step=60, prepare_ms=0,
                    encode_ms=20, step_compute_ms=1000, finish_ms=100, comm_mb_per_step=100,
                    collectives_per_step=240, cfg_exchange_mb=1, decode_chunks=21,
                    links={str(d): dict(gbps=5, latency_ms=.1) for d in [2, 3, 4, 6, 8]})
            else:
                cost = dict(kind='image', service_ms=250*(shape['height']/512)**2)
            entries[sid] = dict(spec=formal.shape_spec(self.p, sid), cost=cost,
                source=dict(run_sha256='b'*64, core_sha256='c'*64,
                            resize_ms=10, quantum_overhead_ms=1, gpu_uuids=self.uuids))
        return seal(dict(schema=formal.BUNDLE_SCHEMA, model='wan21', protocol_sha256=canonical_hash(self.p),
                         gpu_uuids=self.uuids, classes=entries, model_manifests=self.manifests))

    def build(self, **kwargs):
        split = kwargs.get('split', 'test')
        return formal.serving(self.p, 'wan21', self.uuids, kwargs.pop('bundle', self.bundle(split)),
                              self.pools, self.manifests, **kwargs)

    def test_split_is_disjoint_stable_and_preserves_text(self):
        for kind in self.pools:
            parts = [formal.split_pool(self.pools[kind], self.p, kind, split)
                     for split in ('development', 'calibration', 'test')]
            ids = [set(row['sample_id'] for row in p['samples']) for p in parts]
            self.assertEqual(len(set.union(*ids)), 1200)
            self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
            reversed_pool = copy.deepcopy(self.pools[kind]);reversed_pool['samples'].reverse()
            self.assertEqual(parts[2]['samples'], formal.split_pool(seal(reversed_pool), self.p, kind, 'test')['samples'])
        tampered = copy.deepcopy(self.pools['image']);tampered['source']['revision'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'source differs'):
            formal.split_pool(seal(tampered), self.p, 'image', 'test')

    def test_preflight_counts_excluded_pool_capacity_before_gpu_use(self):
        result = formal.preflight(self.p, 'wan21', self.pools, self.manifests, 'formal')
        self.assertEqual(result['pools']['test']['video']['required'], 100)
        self.assertEqual(result['pools']['test']['image']['required'], 160)
        dev = formal.preflight(self.p, 'wan21', self.pools, self.manifests, 'development')
        self.assertEqual(dev['pools']['development']['video']['required'], 40)
        self.assertEqual(dev['pools']['development']['image']['required'], 20)
        selected = formal.split_pool(self.pools['image'], self.p, 'image', 'test')['samples']
        pools = copy.deepcopy(self.pools)
        pools['image']['samples'] = (selected[:159] + formal.split_pool(
            self.pools['image'], self.p, 'image', 'calibration')['samples'][:1])
        pools['image'] = seal(pools['image'])
        with self.assertRaisesRegex(ValueError, '159 available, 160 required'):
            formal.preflight(self.p, 'wan21', pools, self.manifests, 'formal')
        manifests = copy.deepcopy(self.manifests)
        manifests['wan21']['sha256'] = 'not-a-hash'
        with self.assertRaisesRegex(ValueError, 'manifest SHA256'):
            formal.preflight(self.p, 'wan21', self.pools, manifests, 'formal')

    def test_preflight_cli_is_cpu_only(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind, pool in self.pools.items():
                (root / (pool['kind'] + '.json')).write_text(json.dumps(pool))
            (root / 'models.json').write_text(json.dumps(self.manifests))
            output = root / 'preflight.json'
            args = ['formal_benchmark.py', 'preflight', '--model', 'wan21', '--scope', 'formal',
                    '--datasets', str(root), '--model-manifests', str(root / 'models.json'),
                    '--output', str(output)]
            with patch.object(sys, 'argv', args), patch.object(formal, 'allocated_uuids',
                    side_effect=AssertionError('CPU preflight must not query GPUs')):
                formal.main()
            self.assertEqual(json.loads(output.read_text())['status'], 'passed')

    def test_manifest_mapping_selects_explicit_verified_snapshot(self):
        manifests = copy.deepcopy(self.manifests)
        manifests['wan21']['model_path'] = '/shared/data/DITSERVE/models/wan21-official'
        cfg = formal.calibration(self.p, 'wan21', self.uuids, self.pools, manifests, 'wan21-dev', 'sd35-dev')
        self.assertTrue(all(worker['config']['model_path'] == manifests['wan21']['model_path']
                            for worker in cfg['workers'].values() if worker['backend'] == 'wan'))
        manifests['wan21']['model_path'] = '/shared/data/DITSERVE/models/../other'
        with self.assertRaisesRegex(ValueError, 'snapshot must be inside'):
            formal.preflight(self.p, 'wan21', self.pools, manifests, 'development')

    def test_mixed_resolution_trace_and_policy_pairing(self):
        cfg = self.build(video_fraction=.2)
        phase = cfg['phases'][0];trace = phase['matrix']['/workload'][0]
        self.assertEqual(len(trace), 200)
        self.assertEqual(cfg['benchmark']['trace']['manifest']['class_counts'],
            {'wan21-480p': 20, 'wan21-720p': 20, 'sd35-512': 80, 'sd35-1024': 80})
        for row in trace:
            self.assertEqual(formal.input_shape(row['spec']), formal.shape_spec(self.p, row['class']))
            self.assertAlmostEqual(row['deadline_ms']-row['arrival_ms'], row['service_ms']*(1.5 if row['kind']=='video' else 2.5))
        self.assertEqual(len(phase['matrix']['/workload']), 1)
        self.assertTrue(all(worker['config']['model_manifest_sha256']=='a'*64 for worker in cfg['workers'].values()))
        self.assertEqual(cfg['benchmark']['formal_protocol']['sha256'], canonical_hash(self.p))
        again = self.build(video_fraction=.2)
        self.assertEqual(cfg, again)

    def test_previous_exploration_is_not_relabelled_unseen_by_new_hash(self):
        pool = copy.deepcopy(self.pools['image'])
        candidate = formal.split_pool(pool, self.p, 'image', 'test')['samples'][0]
        for row in pool['samples']:
            row['split'] = 'dev' if row['sample_id'] == candidate['sample_id'] else 'test'
        pool['samples'].append(dict(sample_id='whitespace-duplicate',
            prompt='  ' + candidate['prompt'].replace(' ', '   ') + '  ', split='test'))
        pool['samples'].append(dict(sample_id='known-warmup', prompt='warmup', split='test'))
        pool['selection'] = {'dev_distinct_prompts': 1}
        pool = seal(pool)
        for split in ('calibration', 'test'):
            samples = formal.split_pool(pool, self.p, 'image', split)['samples']
            normalized = {' '.join(row['prompt'].split()) for row in samples}
            self.assertNotIn(candidate['prompt'], normalized)
            self.assertNotIn('warmup', normalized)
        development = formal.split_pool(pool, self.p, 'image', 'development')['samples']
        self.assertIn('warmup', [row['prompt'] for row in development])
        pool['samples'][0].pop('split')
        with self.assertRaisesRegex(ValueError, 'roles missing'):
            formal.split_pool(seal(pool), self.p, 'image', 'test')

    def test_each_shape_requires_own_profile_and_exact_allocation(self):
        bundle = self.bundle();bundle['classes'].pop('wan21-720p')
        with self.assertRaisesRegex(ValueError, 'every frozen shape'):
            self.build(bundle=seal(bundle))
        bundle = self.bundle();bundle['classes']['wan21-720p']['spec']['height'] = 480
        with self.assertRaisesRegex(ValueError, 'shape identity'):
            self.build(bundle=seal(bundle))
        bundle = self.bundle();bundle['gpu_uuids'].reverse()
        with self.assertRaisesRegex(ValueError, 'GPU UUIDs'):
            self.build(bundle=seal(bundle))
        with self.assertRaisesRegex(ValueError, 'seed/split'):
            self.build(seed=201)

    def test_multiclass_gamma_and_burst_preserve_counts(self):
        for pattern in ('gamma', 'bursty'):
            cfg = self.build(pattern=pattern, video_fraction=.5)
            trace = cfg['benchmark']['trace']['manifest']
            self.assertEqual(trace['counts'], {'video':100, 'image':100})
            self.assertEqual(set(trace['class_counts'].values()), {50})
            self.assertTrue(trace['total_duration_ms'] > 0)
            if pattern=='gamma':self.assertEqual(trace['windows'][0]['cv'], 2)

    def test_p0_modes_are_digest_bound_and_separate(self):
        for mode in ('contention', 'no_preemption'):
            cfg = self.build(split='development', seed=201, diagnostic=mode)
            phase = cfg['phases'][0]
            self.assertEqual(set(phase['matrix']['/policy']), {'edf-cfg2','block-cfg2'})
            self.assertEqual(phase['options']['forms'], ['cfg2'])
            self.assertTrue(all(row['output_mode']=='digest' for row in phase['matrix']['/workload'][0]))
            self.assertEqual(cfg['benchmark']['formal_protocol']['role'], 'preemption' if mode == 'contention' else mode)
            self.assertEqual(len(phase['matrix']['/workload']), 1)
            self.assertEqual(len(phase['matrix']['/trace_provenance']), 1)
            self.assertEqual(cfg['benchmark']['p0_diagnostic_controls'], self.p['p0_diagnostics'][mode])
            if mode=='no_preemption':
                self.assertEqual(cfg['benchmark']['trace']['manifest']['counts'], {'video':40})
                self.assertTrue(phase['options']['lazy_preemption'])
                self.assertEqual(cfg['benchmark']['load'], .1)
            else:
                self.assertFalse(phase['options']['lazy_preemption'])
                self.assertEqual(cfg['benchmark']['trace']['manifest']['counts'], {'video':20, 'image':20})
                self.assertEqual(cfg['benchmark']['pattern'], 'bursty')
                self.assertEqual(cfg['benchmark']['load'], 1.2)
                for row in phase['matrix']['/workload'][0]:
                    self.assertEqual(row['effective_slo_scale'], 1.1 if row['kind']=='image' else 1.5)
                self.assertTrue(cfg['benchmark']['p0_diagnostic_controls']['require_observed_stops'])
        # The stress overrides cannot leak into the actual formal main table.
        main = self.build()
        self.assertNotIn('p0_diagnostic_controls', main['benchmark'])
        self.assertEqual(main['benchmark']['image_slo_scale'], 2.5)
        self.assertTrue(main['phases'][0]['options'].get('lazy_preemption', True))

    def test_exclusion_manifest_resolves_from_installed_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            installed = Path(temporary) / 'share/ditserve'
            (installed / 'benchmarks').mkdir(parents=True)
            shutil.copy2(ROOT / 'benchmarks/formal_benchmark.py', installed / 'benchmarks/formal_benchmark.py')
            shutil.copytree(ROOT / 'configs/benchmarks', installed / 'configs/benchmarks')
            spec = importlib.util.spec_from_file_location('installed_formal', installed / 'benchmarks/formal_benchmark.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            p = module.protocol()
            self.assertEqual(module.DEFAULT_PROTOCOL, installed / 'configs/benchmarks/formal-mixed-v1.json')
            self.assertEqual(module.split_pool(self.pools['image'], p, 'image', 'test'),
                             formal.split_pool(self.pools['image'], self.p, 'image', 'test'))
            (installed / 'configs/benchmarks' / p['prior_exposure']['manifest_file']).unlink()
            with self.assertRaises(FileNotFoundError):
                module.split_pool(self.pools['image'], p, 'image', 'test')

    def test_calibration_uses_declared_shapes_and_calibration_split(self):
        cfg = formal.calibration(self.p,'wan21',self.uuids,self.pools,self.manifests,'wan21-720p','sd35-512')
        self.assertEqual(cfg['phases'][0]['workload'][0]['spec']['height'],720)
        self.assertEqual(cfg['phases'][0]['workload'][1]['spec']['height'],512)
        self.assertEqual(cfg['phases'][0]['profile']['classes']['wan21-preflight']['steps'],50)
        self.assertEqual(cfg['benchmark']['formal_protocol']['split'],'calibration')
        with self.assertRaisesRegex(ValueError,'model manifest'):
            formal.calibration(self.p,'wan21',self.uuids,self.pools,{},'wan21-dev','sd35-dev')

    def test_old_shape_records_cannot_be_relabelled_formal(self):
        with self.assertRaisesRegex(ValueError,'frozen protocol identity'):
            formal.assemble(self.p,'wan21',[{'config':{'benchmark':{}}}],{})


if __name__ == '__main__':unittest.main()
