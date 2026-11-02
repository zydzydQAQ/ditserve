import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import formal_benchmark as formal
import native_quality
import p4_quality_plan as p4
from ditserve.evaluation.common import canonical_hash, seal, verify_seal


class P4PlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = formal.protocol()
        cls.uuids = ['GPU-first', 'GPU-second']
        expected = cls.protocol['datasets']['video']
        cls.pools = {'video': seal(dict(kind=expected['kind'], source=expected,
            samples=[dict(sample_id=f'v{i:04}', prompt=f'quality held out prompt {i}') for i in range(300)]))}
        cls.manifests = {key: dict(path=f'/shared/data/DITSERVE/models/formal-manifests/{key}.json', sha256='a' * 64)
                         for key in ('wan21', 'wan22', 'sd35')}
        cls.manifests['wan22']['model_path'] = '/shared/data/DITSERVE/models/wan2.2-ti2v-5b-official-b8fff731'
        cls.rule = dict(schema='ditserve.p4-fold-rule.v1', steps=50,
            selected_without_test_outputs=True, development_evidence_sha256='d' * 64,
            eligible_steps=list(range(30, 49)), max_folded_steps=7,
            curvature_threshold=.02, cfg_damping=.96, cfg_vel_alpha=0., max_curvature_guard=1.65,
            numeric_thresholds=dict(min_psnr=30.5, min_ssim=.94, max_lpips=.1))
        cls.snapshot = dict(filesystem='/data', observed_at_utc='2026-11-03T00:00:00Z',
            total_bytes=1025 * p4.GIB, used_bytes=770 * p4.GIB,
            available_bytes=255 * p4.GIB, use_percent=76)

    def bundle(self, model='wan22'):
        classes = {}
        for sid in p4.shape_ids(self.protocol, model):
            classes[sid] = dict(spec=formal.shape_spec(self.protocol, sid),
                cost=dict(kind='video', prepare_ms=5, encode_ms=10, step_compute_ms=1000, finish_ms=100),
                source=dict(run_sha256='b' * 64, core_sha256='c' * 64, gpu_uuids=self.uuids))
        return seal(dict(schema=formal.BUNDLE_SCHEMA, model=model, protocol_sha256=canonical_hash(self.protocol),
            gpu_uuids=self.uuids, classes=classes, model_manifests=self.manifests))

    def prepare(self, **kwargs):
        model = kwargs.pop('model', 'wan22')
        return p4.prepare(self.protocol, model, kwargs.pop('uuids', self.uuids),
            kwargs.pop('bundle', self.bundle(model)), kwargs.pop('pools', self.pools),
            kwargs.pop('manifests', self.manifests), kwargs.pop('rule', self.rule), self.snapshot,
            prompt_selection_seed=kwargs.pop('prompt_selection_seed', 1234), **kwargs)

    def test_fixed_prompt_grid_literal_seeds_and_official_mapping_on_two_gpus(self):
        plan = self.prepare()
        verify_seal(plan)
        self.assertEqual(plan['form'], 'cfg2')
        self.assertEqual(plan['generation_seeds'], [301, 302, 303])
        self.assertEqual(plan['expected'], dict(pair_count=192, generated_artifact_count=1728,
            denominator='all planned pair x arm identities, including failures, missing outputs and unqualified shapes'))
        self.assertEqual(plan['worker_model_bindings']['video']['model_path'], self.manifests['wan22']['model_path'])
        for sid in plan['shape_ids']:
            pairs = [row for row in plan['pairs'] if row['shape_id'] == sid]
            self.assertEqual(len(pairs), 96)
            for seed in [301, 302, 303]:
                self.assertEqual([row['dataset_sample_id'] for row in pairs if row['generation_seed'] == seed],
                                 [row['dataset_sample_id'] for row in plan['prompts']])
            for row in pairs:
                self.assertEqual(row['spec']['seed'], row['generation_seed'])
                self.assertEqual(row['spec']['steps'], 50)
                self.assertFalse(row['spec']['cache'])
        self.assertFalse(plan['execution_ready'])
        self.assertFalse(plan['qualification_eligible'])
        self.assertEqual(plan['statistics']['resampling_unit'], 'prompt cluster; keep all three seeds and shapes attached')
        self.assertIn('release ACK timestamp', plan['acceptance']['reuse']['events'])

    def test_prompt_selection_is_independent_of_order_and_generation_seeds(self):
        pools = copy.deepcopy(self.pools)
        pools['video']['samples'].reverse(); pools['video'] = seal(pools['video'])
        first = self.prepare(); second = self.prepare(pools=pools)
        self.assertEqual(first['prompts'], second['prompts'])
        self.assertEqual(first['pairs'], second['pairs'])
        self.assertNotEqual(first['prompts'], self.prepare(prompt_selection_seed=9876)['prompts'])
        held_out = formal.split_pool(self.pools['video'], self.protocol, 'video', 'test')
        self.assertTrue({row['dataset_sample_id'] for row in first['prompts']}
                        <= {row['sample_id'] for row in held_out['samples']})

    def test_calibration_identity_shape_and_allocation_fail_closed(self):
        for mutation, message in [
                (lambda b: b['gpu_uuids'].reverse(), 'same model/protocol/allocation'),
                (lambda b: b['classes'].pop('wan22-720p'), 'every formal video shape'),
                (lambda b: b['classes']['wan22-720p']['spec'].update(frames=81), 'exact measured calibration'),
                (lambda b: b['classes']['wan22-720p']['source'].update(core_sha256='f' * 64), 'core changed'),
                (lambda b: b['classes']['wan22-720p']['source'].update(gpu_uuids=['other']), 'identical GPU placement')]:
            with self.subTest(message=message):
                bundle = copy.deepcopy(self.bundle()); mutation(bundle)
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare(bundle=seal(bundle))
        with self.assertRaisesRegex(ValueError, 'fitting this allocation'):
            self.prepare(form='cfg2sp3')
        for path in (None, '/shared/data/DITSERVE/models/wan2.2-ti2v-5b'):
            manifests = copy.deepcopy(self.manifests)
            if path is None:
                manifests['wan22'].pop('model_path')
            else:
                manifests['wan22']['model_path'] = path
            bundle = self.bundle(); bundle['model_manifests'] = manifests
            with self.assertRaisesRegex(ValueError, 'composite'):
                self.prepare(bundle=seal(bundle), manifests=manifests)

    def test_late_window_budget_controls_and_no_implicit_pilot_schedule(self):
        plan = self.prepare()
        for count, arm in enumerate(plan['arms'][:-1]):
            self.assertEqual(arm['expected_actual_fold_count'], count)
            if count:
                self.assertEqual(arm['cfg_fold']['folded_steps'], list(range(30, 30 + count)))
                self.assertEqual(arm['cfg_fold']['max_folded_steps'], count)
                self.assertTrue(arm['cfg_fold']['final_step_exact_anchor'])
        self.assertEqual(plan['arms'][-1]['cfg_fold']['folded_steps'], list(range(30, 49)))
        self.assertEqual(plan['fixed_zero_control']['arm'], 'exact')
        for fields in ({'eligible_steps': list(range(12, 19))}, {'steps': 20},
                       {'eligible_steps': [48, 49]}, {'max_folded_steps': 20},
                       {'selected_without_test_outputs': False}, {'curvature_threshold': float('nan')}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.prepare(rule=dict(self.rule, **fields))

    def test_stop_line_budget_detects_full_matrix_and_requires_retention_closure(self):
        result = self.prepare()['storage']
        self.assertEqual(result['stop_line_headroom_bytes'], 20 * p4.GIB)
        self.assertEqual(result['shapes']['wan22-720p']['array_payload_bytes'], 1308426240)
        self.assertGreater(result['full_artifact_bytes'], 1500 * p4.GIB)
        self.assertFalse(result['full_retention_fits'])
        self.assertTrue(result['serial_pair_fits'])
        self.assertFalse(result['retention']['automatic_deletion'])
        self.assertFalse(result['retention']['execution_ready'])
        self.assertLess(result['serial_pair_temporary_peak_bytes'], 5 * p4.GIB)
        for total, used, available, percent in [(1025, 791, 234, 78), (900, 720, 180, 80)]:
            snap = dict(self.snapshot, total_bytes=total * p4.GIB, used_bytes=used * p4.GIB,
                        available_bytes=available * p4.GIB, use_percent=percent)
            stopped = p4.storage_budget(self.protocol, 'wan22', self.rule, snap)
            self.assertTrue(stopped['stopped'])
            self.assertFalse(stopped['serial_pair_fits'])
        tiny = dict(self.snapshot, total_bytes=1025 * p4.GIB, used_bytes=789 * p4.GIB,
                    available_bytes=236 * p4.GIB, use_percent=77)
        self.assertFalse(p4.storage_budget(self.protocol, 'wan22', self.rule, tiny)['serial_pair_fits'])

    def test_storage_cli_requires_no_gpu_or_calibration_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'rule.json').write_text(json.dumps(self.rule))
            (path / 'df.json').write_text(json.dumps(self.snapshot))
            output = path / 'storage.json'
            args = ['p4_quality_plan.py', 'storage', '--model', 'wan22', '--rule', str(path / 'rule.json'),
                    '--storage-snapshot', str(path / 'df.json'), '--output', str(output)]
            with patch.object(sys, 'argv', args), patch.object(p4, 'prepare', side_effect=AssertionError('no preparation')):
                p4.main()
                self.assertFalse(json.loads(output.read_text())['full_retention_fits'])
                with self.assertRaises(FileExistsError):
                    p4.main()


class P4InfinitePSNRTest(unittest.TestCase):
    def quality(self, *, count=0, exact=True, psnr=None, lpips=0.):
        return dict(status='completed', source=dict(reference_run_sha256='a' * 64), pairs=[dict(
            dataset_sample_id='p', seed=301, prompt='prompt', shape=[121, 704, 1280, 50],
            candidate_cfg_folded_steps=count, psnr_db=psnr, psnr_is_infinite=exact,
            exact_match=exact, ssim=1. if exact else .9, lpips_mean=lpips)])

    def test_exact_zero_fold_self_control_does_not_invent_infinity_delta(self):
        exact = self.quality()
        result = native_quality.matched_fold_bootstrap(exact, {0: exact}, resamples=100)
        self.assertEqual(result['status'], 'measured')
        self.assertFalse(result['superiority_qualified'])
        stratum = result['strata']['0']
        self.assertTrue(stratum['zero_fold_exact_self_control'])
        psnr = stratum['metrics']['psnr_db']
        self.assertEqual(psnr['relations']['both_exact'], 1)
        self.assertIsNone(psnr['mean_difference'])
        self.assertIsNone(psnr['ci95'])
        self.assertEqual(stratum['metrics']['ssim']['mean_difference'], 0.)
        self.assertEqual(stratum['metrics']['lpips_mean']['mean_difference'], 0.)
        json.dumps(result, allow_nan=False)

    def test_one_sided_infinity_is_relation_and_other_metrics_remain_independent(self):
        candidate = self.quality(count=2)
        control = self.quality(count=2, exact=False, psnr=30., lpips=.1)
        result = native_quality.matched_fold_bootstrap(candidate, {2: control}, resamples=100)
        self.assertEqual(result['status'], 'measured')
        metrics = result['strata']['2']['metrics']
        self.assertEqual(metrics['psnr_db']['relations']['candidate_exact_control_finite'], 1)
        self.assertIsNone(metrics['psnr_db']['mean_difference'])
        self.assertAlmostEqual(metrics['lpips_mean']['mean_difference'], -.1)
        self.assertFalse(result['strata']['2']['zero_fold_exact_self_control'])
        self.assertFalse(result['superiority_qualified'])

    def test_missing_or_inconsistent_metrics_still_fail_closed(self):
        exact = self.quality()
        for field, value in [('exact_match', False), ('psnr_is_infinite', False), ('lpips_mean', None)]:
            invalid = copy.deepcopy(exact); invalid['pairs'][0][field] = value
            result = native_quality.matched_fold_bootstrap(invalid, {0: exact}, resamples=100)
            self.assertEqual(result['status'], 'unverified')
        self.assertEqual(native_quality.matched_fold_bootstrap(exact, {}, resamples=100)['status'], 'unverified')

    def test_finite_psnr_difference_preserves_existing_bootstrap(self):
        candidate = self.quality(count=1, exact=False, psnr=32., lpips=.05)
        control = self.quality(count=1, exact=False, psnr=30., lpips=.1)
        result = native_quality.matched_fold_bootstrap(candidate, {1: control}, resamples=100)
        metric = result['strata']['1']['metrics']['psnr_db']
        self.assertEqual(metric['status'], 'measured')
        self.assertEqual(metric['mean_difference'], 2.)
        self.assertEqual(metric['ci95'], [2., 2.])


if __name__ == '__main__':
    unittest.main()
