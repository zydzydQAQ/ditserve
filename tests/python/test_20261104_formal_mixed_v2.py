"""GenServe/TetriServe-aligned formal-mixed-v2 protocol and workload unit tests."""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get('DITSERVE_CORE', ROOT / '.cache/build/ditserve'))
sys.path.insert(0, str(ROOT / 'benchmarks'))
import challenge1_contrasts as c1
import formal_benchmark as formal
import native_serving
import paper_results
from ditserve.evaluation.common import canonical_hash, seal
from ditserve.evaluation.workloads import (
    build_trace,
    latent_length,
    resolution_class_quotas,
    resolution_probabilities,
    validate_trace,
)


class FormalMixedV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = formal.protocol(formal.DEFAULT_V2_PROTOCOL)
        cls.uuids = [f'GPU-{i}' for i in range(8)]
        cls.pools = {
            kind: seal(dict(
                kind=expected['kind'],
                source=expected,
                samples=[dict(sample_id=f'{kind}-{i}', prompt=f'{kind} sample {i}') for i in range(1200)],
            ))
            for kind, expected in cls.p['datasets'].items()
        }
        cls.manifests = {
            model: dict(path=f'/shared/data/DITSERVE/models/{model}/manifest.json', sha256='a' * 64)
            for model in ('wan22', 'wan21', 'sd35')
        }

    def bundle(self, model='wan22', split='test'):
        entries = {}
        scope = 'formal' if split == 'test' else 'development'
        for sid, shape in self.p['shapes'].items():
            if shape['scope'] != scope or shape['model'] not in (model, 'sd35'):
                continue
            if shape['kind'] == 'video':
                scale = (shape['height'] * shape['width']) / (512 * 896)
                cost = dict(
                    kind='video',
                    steps=shape['steps'],
                    blocks_per_step=60,
                    prepare_ms=100.0,
                    encode_ms=15000.0,
                    step_compute_ms=3000.0 * scale,
                    finish_ms=1200.0 * scale,
                    comm_mb_per_step=100.0 * scale,
                    collectives_per_step=240,
                    cfg_exchange_mb=2.0 * scale,
                    decode_chunks=21,
                    links={str(d): dict(gbps=5.0, latency_ms=0.1) for d in (2, 3, 4, 6, 8)},
                )
            else:
                scale = (shape['height'] / 512.0) ** 2
                cost = dict(kind='image', service_ms=1800.0 * scale)
            entries[sid] = dict(
                spec=formal.shape_spec(self.p, sid),
                cost=cost,
                source=dict(
                    run_sha256='b' * 64,
                    core_sha256='c' * 64,
                    resize_ms=10.0,
                    quantum_overhead_ms=1.0,
                    gpu_uuids=self.uuids,
                ),
            )
        return seal(dict(
            schema=formal.BUNDLE_SCHEMA,
            model=model,
            protocol_sha256=canonical_hash(self.p),
            gpu_uuids=self.uuids,
            classes=entries,
            model_manifests=self.manifests,
        ))

    def test_v2_protocol_declares_confirmed_parameters(self):
        self.assertEqual(self.p['schema'], 'ditserve.formal-protocol.v2')
        self.assertEqual(self.p['primary_model'], 'wan22')
        self.assertEqual(self.p['sensitivity_models'], ['wan21'])
        self.assertEqual(self.p['requests_per_cell']['test'], 100)
        self.assertEqual(self.p['slo']['scales'], [1.0, 1.5, 2.0])
        self.assertEqual(self.p['slo']['default_scale'], 1.5)
        self.assertEqual(self.p['loads'], [0.5, 1.0, 1.5])
        self.assertEqual(self.p['video_fractions'], [0.2, 0.5, 0.8])
        self.assertEqual([x['name'] for x in self.p['patterns']], ['poisson', 'gamma'])
        gamma_pattern = next(x for x in self.p['patterns'] if x['name'] == 'gamma')
        self.assertEqual(gamma_pattern['cv'], 2.0)
        self.assertEqual(gamma_pattern['cvs'], [2.0, 4.0, 6.0])
        wan22_shapes = [k for k, v in self.p['shapes'].items() if v['scope'] == 'formal' and v['model'] == 'wan22']
        sd35_shapes = [k for k, v in self.p['shapes'].items() if v['scope'] == 'formal' and v['model'] == 'sd35']
        self.assertEqual(wan22_shapes, ['wan22-256p', 'wan22-480p', 'wan22-720p'])
        self.assertEqual(sd35_shapes, ['sd35-512', 'sd35-1024', 'sd35-1440'])
        for sid in wan22_shapes:
            s = self.p['shapes'][sid]
            self.assertEqual(s['frames'], 81)
            self.assertEqual(s['steps'], 50)
            self.assertEqual(s['height'] % 32, 0)
            self.assertEqual(s['width'] % 32, 0)
            tokens = ((s['frames'] - 1) // 4 + 1) * (s['height'] // 32) * (s['width'] // 32)
            for sp in (1, 2, 3, 4, 6, 8):
                self.assertEqual(tokens % sp, 0, f'{sid} tokens={tokens} not divisible by SP={sp}')
        self.assertEqual(self.p['shapes']['sd35-1440']['height'], 1440)
        self.assertEqual(self.p['shapes']['sd35-1440']['width'], 1440)

    def test_tetriserve_exponential_skewed_distribution(self):
        img_names = ['sd35-512', 'sd35-1024', 'sd35-1440']
        img_classes = {sid: dict(spec=formal.shape_spec(self.p, sid)) for sid in img_names}
        self.assertEqual(latent_length(img_classes['sd35-512']), 1024.0)
        self.assertEqual(latent_length(img_classes['sd35-1024']), 4096.0)
        self.assertEqual(latent_length(img_classes['sd35-1440']), 8100.0)
        probs = resolution_probabilities(img_names, img_classes, 'skewed')
        self.assertLess(probs['sd35-512'], probs['sd35-1024'])
        self.assertLess(probs['sd35-1024'], probs['sd35-1440'])
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        quotas_50 = resolution_class_quotas(50, img_names, img_classes, 'skewed')
        self.assertEqual(quotas_50, [10, 15, 25])
        self.assertEqual(sum(quotas_50), 50)
        uniform_50 = resolution_class_quotas(50, img_names, img_classes, 'uniform')
        self.assertEqual(uniform_50, [17, 17, 16])

    def test_anchor_and_sweeps_preserve_unoptimized_sp1_slo_and_rate_rpm(self):
        b = self.bundle('wan22')
        anchor_cfg = formal.serving(self.p, 'wan22', self.uuids, b, self.pools, self.manifests)
        trace = anchor_cfg['phases'][0]['matrix']['/workload'][0]
        manifest = anchor_cfg['benchmark']['trace']['manifest']
        self.assertEqual(len(trace), 100)
        self.assertEqual(manifest['counts'], {'video': 50, 'image': 50})
        self.assertEqual(
            manifest['class_counts'],
            {
                'wan22-256p': 17, 'wan22-480p': 17, 'wan22-720p': 16,
                'sd35-512': 17, 'sd35-1024': 17, 'sd35-1440': 16,
            },
        )
        self.assertIn('r0_rpm', manifest)
        self.assertIn('rate_rpm', manifest)
        self.assertAlmostEqual(manifest['rate_rpm'], manifest['r0_rpm'])
        for row in trace:
            self.assertAlmostEqual(row['effective_slo_scale'], 1.5)
            self.assertAlmostEqual(row['deadline_ms'] - row['arrival_ms'], row['service_ms'] * 1.5)
        for scale in (1.0, 1.5, 2.0):
            cfg = formal.serving(self.p, 'wan22', self.uuids, b, self.pools, self.manifests, slo_scale=scale)
            for row in cfg['phases'][0]['matrix']['/workload'][0]:
                self.assertAlmostEqual(row['effective_slo_scale'], scale)
        for load_val in (0.5, 1.0, 1.5):
            cfg = formal.serving(self.p, 'wan22', self.uuids, b, self.pools, self.manifests, load=load_val)
            m = cfg['benchmark']['trace']['manifest']
            self.assertAlmostEqual(m['rate_rpm'], m['r0_rpm'] * load_val)
        skewed_cfg = formal.serving(
            self.p, 'wan22', self.uuids, b, self.pools, self.manifests, resolution_distribution='skewed'
        )
        self.assertEqual(
            skewed_cfg['benchmark']['trace']['manifest']['class_counts'],
            {
                'wan22-256p': 10, 'wan22-480p': 15, 'wan22-720p': 25,
                'sd35-512': 10, 'sd35-1024': 15, 'sd35-1440': 25,
            },
        )
        cells = formal.genserve_sweep_cells(self.p)
        self.assertEqual(len(cells), 11)
        gamma_cells = [c for c in cells if c['pattern'] == 'gamma']
        self.assertEqual([c['cv'] for c in gamma_cells], [2.0, 4.0, 6.0])
        for bad_cv in (0.5, 1.0):
            with self.assertRaises(ValueError):
                formal.serving(self.p, 'wan22', self.uuids, b, self.pools, self.manifests, pattern='gamma', cv=bad_cv)

    def test_in_framework_policy_comparison_and_quality_contracts(self):
        expected_policies = [
            'static-cfg2sp4',
            'tetriserve',
            'sbp-esp',
            'edf-bestfit',
            'ditserve',
        ]
        self.assertEqual(formal.SUITES['in-framework-main'], expected_policies)
        self.assertEqual(formal.in_framework_main_policies(), expected_policies)
        self.assertEqual(
            formal.in_framework_main_policies('static-cfg2sp3'),
            ['static-cfg2sp3', 'tetriserve', 'sbp-esp', 'edf-bestfit', 'ditserve'],
        )
        with self.assertRaises(ValueError):
            formal.in_framework_main_policies('invalid-policy')

        contract = self.p['baseline_comparison_contract']
        self.assertEqual(contract['mode'], 'in_framework_policy_comparison')
        self.assertTrue(contract['shared_executor'])
        self.assertEqual(contract['main_table_policies'], expected_policies)
        self.assertEqual(contract['ablation_only_policies'], ['edf-cfg2', 'cfg2-first'])
        for suite in ('main', 'in-framework-main', 'sweep'):
            for heuristic in ('edf-cfg2', 'cfg2-first'):
                self.assertNotIn(heuristic, formal.SUITES[suite])
        for heuristic in ('edf-cfg2', 'cfg2-first'):
            self.assertIn(heuristic, formal.SUITES['ablation'])
        self.assertIn('static-cfg2sp4', contract['best_static_form_rule'])
        self.assertEqual(contract['mechanism_labels']['tetriserve'], 'TetriServe-policy')
        self.assertEqual(contract['mechanism_labels']['static-cfg2sp4'], 'Static-<Form>')
        self.assertIn('tetriserve', contract['external_native_intersection'])
        self.assertIn('xdit', contract['external_native_intersection'])

        q_contract = self.p['quality_acceptance_contract']
        self.assertEqual(q_contract['same_parallelism_segmentation_gate']['atol'], 1e-5)
        self.assertEqual(q_contract['same_parallelism_segmentation_gate']['rtol'], 1e-5)
        self.assertEqual(q_contract['cross_sp_acceptance_gate']['mode'], 'semantic_quality_equivalence')
        self.assertIn('PSNR', q_contract['cross_sp_acceptance_gate']['metrics'])
        self.assertIn('VBench', q_contract['cross_sp_acceptance_gate']['metrics'])

        b = self.bundle('wan22')
        cfg = formal.serving(
            self.p, 'wan22', self.uuids, b, self.pools, self.manifests, suite='in-framework-main'
        )
        self.assertEqual(cfg['benchmark']['requested_policies'], expected_policies)
        self.assertCountEqual(cfg['phases'][0]['matrix']['/policy'], expected_policies)
        in_fw = cfg['benchmark']['in_framework_policy_comparison']
        self.assertTrue(in_fw['enabled'])
        self.assertEqual(in_fw['comparison_mode'], 'same_executor_mechanism_comparison')
        self.assertEqual(in_fw['canonical_mechanisms'], expected_policies)
        self.assertEqual(in_fw['mechanism_display']['tetriserve'], 'TetriServe-policy')
        self.assertEqual(in_fw['mechanism_display']['static-cfg2sp4'], 'Static-CFG2SP4')
        self.assertEqual(in_fw['baseline_comparison_contract']['main_table_policies'], expected_policies)
        self.assertEqual(cfg['benchmark']['quality_acceptance_contract'], q_contract)

        # Verify Challenge 1 contrasts and paper_results exports
        self.assertEqual(tuple(c1.IN_FRAMEWORK_MAIN_ARMS), tuple(expected_policies))
        for arm in c1.IN_FRAMEWORK_MAIN_ARMS:
            self.assertIn(arm, c1.ARM_SPECS)
            resolved = c1.resolve_arm_config(arm, 'wan22', 8)
            self.assertEqual(resolved['policy'], arm)
        for static_pol in ('static-cfg2sp2', 'static-cfg2sp3', 'static-cfg2sp4'):
            self.assertIn(static_pol, paper_results.POLICY_ORDER)
            self.assertIn(static_pol, paper_results.MECHANISM_DISPLAY)
        self.assertEqual(paper_results.MECHANISM_DISPLAY['tetriserve'], 'TetriServe-policy')
        self.assertEqual(paper_results.mechanism_label('static-cfg2sp4', generic_static=True), 'Static-<Form>')
        self.assertEqual(native_serving.POLICY_DISPLAY['tetriserve'], 'TetriServe (re-impl.)')

    @unittest.skipUnless(BINARY.is_file(), 'build the native ditserve executable first')
    def test_native_cli_validates_all_nine_genserve_sweep_cells(self):
        b = self.bundle('wan22')
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES='',
            HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',
            PYTHONDONTWRITEBYTECODE='1',
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for idx, cell in enumerate(formal.genserve_sweep_cells(self.p)):
                with self.subTest(cell=cell):
                    cfg = formal.serving(
                        self.p,
                        cell['model'],
                        self.uuids,
                        b,
                        self.pools,
                        self.manifests,
                        load=cell['load'],
                        video_fraction=cell['video_fraction'],
                        pattern=cell['pattern'],
                        slo_scale=cell['slo_scale'],
                        resolution_distribution=cell['resolution_distribution'],
                        cv=cell.get('cv'),
                    )
                    path = root / f'cell-{idx}.json'
                    path.write_text(json.dumps(cfg, allow_nan=False), encoding='utf-8')
                    command = [
                        str(BINARY),
                        '--root', str(root),
                        '--package-root', str(ROOT / 'workers'),
                        '--python', sys.executable,
                        'run',
                        '--config', str(path),
                        '--run-id', f'cell-{idx}',
                        '--resume',
                    ]
                    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=30)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn('resume needs an existing run', result.stderr)


if __name__ == '__main__':
    unittest.main()
