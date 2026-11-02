import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import formal_benchmark as formal
import p4_quality_plan as planner
import p4_stream_quality as stream
from ditserve.evaluation.common import canonical_hash, seal, verify_seal


class StreamFixture:
    def __init__(self, root):
        self.root = root
        self.protocol = copy.deepcopy(formal.protocol())
        # Tiny arrays in a distinct sealed TEST protocol; never relabel real runs.
        for sid, shape in self.protocol['shapes'].items():
            if shape['model'] == 'wan21' and shape['scope'] == 'formal':
                shape.update(height=4, width=4, frames=2)
        self.uuids = ['GPU-a', 'GPU-b']
        self.mapping = {name: dict(path=f'/shared/data/DITSERVE/models/{name}.json', sha256='d' * 64)
                        for name in ('wan21', 'sd35')}
        expected = self.protocol['datasets']['video']
        self.pools = {'video': seal(dict(kind=expected['kind'], source=expected,
            samples=[dict(sample_id=str(i), prompt=f'offline quality fixture prompt {i}') for i in range(300)]))}
        classes = {sid: dict(spec=formal.shape_spec(self.protocol, sid),
            cost=dict(kind='video', step_compute_ms=1000, finish_ms=10),
            source=dict(run_sha256='b' * 64, core_sha256='c' * 64, gpu_uuids=self.uuids))
            for sid in planner.shape_ids(self.protocol, 'wan21')}
        bundle = seal(dict(schema=formal.BUNDLE_SCHEMA, model='wan21', protocol_sha256=canonical_hash(self.protocol),
            gpu_uuids=self.uuids, classes=classes, model_manifests=self.mapping))
        rule = dict(schema='ditserve.p4-fold-rule.v1', steps=50, selected_without_test_outputs=True,
            development_evidence_sha256='f' * 64, eligible_steps=list(range(30, 49)), max_folded_steps=1,
            cfg_damping=.96, cfg_vel_alpha=0., curvature_threshold=.02, max_curvature_guard=1.65,
            numeric_thresholds=dict(min_psnr=30., min_ssim=.9, max_lpips=.1))
        df = dict(filesystem='/data', observed_at_utc='2026-11-03T00:00:00Z', total_bytes=1025 * planner.GIB,
                  used_bytes=770 * planner.GIB, available_bytes=255 * planner.GIB, use_percent=76)
        self.plan = planner.prepare(self.protocol, 'wan21', self.uuids, bundle, self.pools, self.mapping,
                                   rule, df, prompt_selection_seed=123)
        self.plan['statistics']['bootstrap_samples'] = 100
        self.plan = seal(self.plan)
        self.source = dict(core_sha256='c' * 64, files={'ditserve/runtime.py': 'e' * 64})
        self.runtime = dict(python='fixture', packages={'numpy': 'fixture'})
        self.entries = []

    def produce(self, pair, arm, *, value=.5, adaptive_folds=0):
        prefix = pair['id'] + '-' + arm['id']
        array = np.full((1, 2, 4, 4, 3), value, dtype=np.float32)
        artifact_path = self.root / (prefix + '.npy')
        np.save(artifact_path, array, allow_pickle=False)
        artifact = dict(path='/original/output/' + artifact_path.name, sha256=stream.file_hash(artifact_path),
            array_sha256=hashlib.sha256(array.tobytes()).hexdigest(), dtype='float32', shape=list(array.shape))
        count = (0 if arm['id'] == 'exact' else 1 if arm['id'] == 'fixed-1' else adaptive_folds)
        indices = list(range(30, 30 + count))
        generation = dict(pair['spec'], model_path='/shared/data/DITSERVE/models/wan', revision='official-revision',
            model_manifest_sha256='d' * 64, model_integrity='manifest_verified', dtype='float16',
            vae_dtype='bfloat16', text_encoder_dtype='float32', attention='flash', engine='xdit',
            xdit_revision='frozen-xdit', runtime_identity=self.runtime, approximate=count > 0, cfg_folded_steps=count)
        row = dict(kind='video', status='completed', request_id='request', phase='cluster', cell=0,
            dataset_sample_id=pair['dataset_sample_id'], spec=pair['spec'], forms=[self.plan['form']],
            outputs=[dict(generation=generation, output=artifact,
                path_history=[dict(step_index=i, cfg_folded=i in indices) for i in range(50)])])
        if arm['id'] != 'exact':
            row['cfg_fold'] = dict(full_cfg_steps=50 - count, folded_cfg_steps=count,
                                  folded_step_indices=indices, saved_uncond_forwards=count)
        workers = {f'wan{i}': dict(backend='wan', gpu_uuid=uuid,
            config=dict(self.plan['worker_model_bindings']['video'], options={'cfg_fold': arm['cfg_fold']}))
            for i, uuid in enumerate(self.uuids)}
        phase = dict(id='cluster', kind='cluster', policy=arm['policy'],
            gpus=[dict(video=f'wan{i}') for i in range(len(self.uuids))],
            options=dict(forms=[self.plan['form']], sp_transport='export', tiered_memory=False),
            video_worker_options=dict(negative_prompt_cache_entries=2),
            matrix={'/policy': [arm['policy']]},
            workload=[dict(request_id='request', kind='video', dataset_sample_id=pair['dataset_sample_id'], spec=pair['spec'])])
        run = dict(schema='ditserve.run.v1', status='completed', experiment='fixture', run_id=prefix,
            source=self.source, requests=[row],
            budget=dict(sessions=[dict(status='completed', identity={'gpu_uuids': self.uuids})]),
            config=dict(workers=workers, phases=[phase], benchmark=dict(native_serving=True, model_manifests=self.mapping,
                formal_protocol=dict(sha256=self.plan['protocol_sha256'], split='test', role='p4_quality', shape_ids=self.plan['shape_ids']),
                p4_quality=dict(plan_sha256=self.plan['sha256'], arm_id=arm['id']))),
            phases=[dict(id='cluster', cells=[dict(cell=0, status='completed', policy=arm['policy'], gpus=2)])])
        run_path = self.root / (prefix + '.json')
        run_path.write_text(json.dumps(run))
        entry = dict(pair_id=pair['id'], arm_id=arm['id'], request_id='request', path=run_path.name,
                     sha256=stream.file_hash(run_path), artifact_path=artifact_path.name)
        self.entries.append(entry)
        return entry

    def one_pair(self):
        for arm in self.plan['arms']:
            self.produce(self.plan['pairs'][0], arm)

    def evidence(self):
        return seal(dict(schema=stream.EVIDENCE_SCHEMA, plan_sha256=self.plan['sha256'],
            runtime_source_sha256=canonical_hash(self.source), runtime_identity_sha256=canonical_hash(self.runtime), runs=self.entries))

    def evaluate(self):
        return stream.evaluate(self.plan, self.protocol, self.evidence(), root=self.root)


class StreamQualityTests(unittest.TestCase):
    def test_complete_grid_is_measured_but_missing_lpips_and_vbench_never_qualify(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory))
            for pair in fixture.plan['pairs']:
                for arm in fixture.plan['arms']:
                    fixture.produce(pair, arm)
            before = {path.name: stream.file_hash(path) for path in fixture.root.iterdir()}
            with patch('ditserve.evaluation.quality._lpips_model', side_effect=AssertionError('must not construct/download')):
                result = fixture.evaluate()
            verify_seal(result)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['denominator']['planned_pair_arms'], 576)
            self.assertEqual(result['denominator']['statuses'], {'measured': 576})
            self.assertFalse(result['qualification_eligible'])
            self.assertFalse(result['retention']['dependencies_closed'])
            self.assertEqual(result['lpips']['status'], 'unknown')
            self.assertEqual(result['vbench']['status'], 'unknown')
            self.assertTrue(all(row['numeric_qualification'] == 'unknown' for row in result['by_arm'].values()))
            matched = result['matched_fold_comparison']
            self.assertEqual(matched['matched_pairs'], 192)
            self.assertEqual(matched['zero_fold_exact_self_controls'], 192)
            self.assertEqual(matched['psnr']['relations'], {'both_exact': 192})
            self.assertIsNone(matched['psnr']['mean_difference'])
            self.assertFalse(matched['superiority_qualified'])
            summary = matched['groups']['all']['mse']
            self.assertEqual(summary['independent_prompt_clusters'], 32)
            self.assertEqual(summary['observed_pairs'], 192)
            self.assertEqual(summary['ci95'], [0., 0.])
            self.assertEqual(before, {path.name: stream.file_hash(path) for path in fixture.root.iterdir()})
            json.dumps(result, allow_nan=False)

    def test_missing_and_failed_arms_remain_in_the_full_planned_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); fixture.one_pair()
            candidate = next(row for row in fixture.entries if row['arm_id'] == 'fixed-1')
            with (fixture.root / candidate['artifact_path']).open('ab') as stream_file:
                stream_file.write(b'corruption')
            result = fixture.evaluate()
            counts = result['denominator']
            self.assertEqual(counts['planned_pair_arms'], 576)
            self.assertEqual(counts['statuses'], {'measured': 2, 'failed': 1, 'missing': 573})
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(result['by_arm']['fixed-1']['missing_or_failed_pairs'], 192)
            fixture.entries = [row for row in fixture.entries if row['arm_id'] != 'exact']
            missing_ref = fixture.evaluate()
            self.assertEqual(missing_ref['denominator']['statuses'], {'missing': 574, 'blocked': 2})

    def test_rehashing_a_changed_identity_or_rule_cannot_bypass_pair_binding(self):
        mutations = [
            lambda r: r['requests'][0]['spec'].update(seed=999),
            lambda r: r['requests'][0].update(dataset_sample_id='swapped'),
            lambda r: r['source'].update(core_sha256='0' * 64),
            lambda r: r['budget']['sessions'][0]['identity'].update(gpu_uuids=['wrong']),
            lambda r: r['config']['workers']['wan0']['config']['options']['cfg_fold'].update(cfg_damping=.9),
            lambda r: r['requests'][0]['outputs'][0]['output'].update(array_sha256='0' * 64),
            lambda r: r['requests'][0]['outputs'][0]['generation'].update(model_integrity='unverified'),
            lambda r: r['config']['benchmark'].pop('p4_quality'),
            lambda r: r['requests'][0]['outputs'][0]['path_history'][0].update(reused_forwards=1),
            lambda r: r['requests'][0]['outputs'][0]['generation'].update(approximate=True),
            lambda r: r.update(status='failed'),
            lambda r: r['source'].update(files=[]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                fixture = StreamFixture(Path(directory)); fixture.one_pair()
                entry = next(row for row in fixture.entries if row['arm_id'] == 'adaptive')
                path = fixture.root / entry['path']; raw = json.loads(path.read_bytes()); mutate(raw)
                path.write_text(json.dumps(raw)); entry['sha256'] = stream.file_hash(path)
                result = fixture.evaluate()
                row = next(row for row in result['pair_results'] if row['pair_id'] == entry['pair_id'] and row['arm_id'] == 'adaptive')
                self.assertEqual(row['status'], 'failed', row)
                self.assertEqual(result['by_arm']['adaptive']['missing_or_failed_pairs'], 192)

    def test_raw_hash_duplicate_and_plan_denominator_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); fixture.one_pair()
            fixture.entries[1]['sha256'] = '0' * 64
            result = fixture.evaluate()
            self.assertEqual(result['denominator']['statuses']['failed'], 1)
            fixture.entries.append(copy.deepcopy(fixture.entries[0]))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                fixture.evaluate()
            fixture.entries.pop()
            fixture.plan['pairs'].pop(); fixture.plan = seal(fixture.plan)
            with self.assertRaisesRegex(ValueError, 'denominator'):
                fixture.evaluate()

    def test_effective_phase_worker_cell_and_matrix_overrides_cannot_hide_wrong_curvature_rule(self):
        overrides = [
            lambda r: r['config']['phases'][0].update(video_worker_options={'cfg_fold': {'curvature_threshold': .5}}),
            lambda r: r['config']['phases'][0].update(worker='wan0', worker_config={'options': {'cfg_fold': {'curvature_threshold': .5}}}),
            lambda r: r['config']['phases'][0]['matrix'].update({'/video_worker_options/cfg_fold/curvature_threshold': [.5]}),
            lambda r: r['config']['phases'][0]['matrix'].update({'/worker': ['wan0'], '/worker_config/options/cfg_fold/curvature_threshold': [.5]}),
            lambda r: r['phases'][0]['cells'][0].update(config={'video_worker_options': {'cfg_fold': {'curvature_threshold': .5}}}),
            lambda r: r['config']['phases'][0].update(video_worker_options={'cfg_fold': None}),
            lambda r: r['config']['phases'][0]['gpus'].reverse(),
            lambda r: r['config']['phases'][0]['workload'][0]['spec'].update(seed=999),
        ]
        for mutate in overrides:
            with self.subTest(override=mutate), tempfile.TemporaryDirectory() as directory:
                fixture = StreamFixture(Path(directory)); fixture.one_pair()
                entry = next(row for row in fixture.entries if row['arm_id'] == 'adaptive')
                path = fixture.root / entry['path']; raw = json.loads(path.read_bytes()); mutate(raw)
                path.write_text(json.dumps(raw)); entry['sha256'] = stream.file_hash(path)
                result = fixture.evaluate()
                row = next(row for row in result['pair_results'] if row['pair_id'] == entry['pair_id'] and row['arm_id'] == 'adaptive')
                self.assertEqual(row['status'], 'failed', row)

    def test_known_merge_order_can_restore_the_frozen_rule_and_null_removes_keys(self):
        self.assertEqual(stream.merge_patch({'a': {'b': 1, 'c': 2}}, {'a': {'b': None, 'd': 3}}), {'a': {'c': 2, 'd': 3}})
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); fixture.one_pair()
            entry = next(row for row in fixture.entries if row['arm_id'] == 'adaptive')
            path = fixture.root / entry['path']; raw = json.loads(path.read_bytes())
            phase = raw['config']['phases'][0]
            phase.update(worker='wan0', worker_config={'options': {'cfg_fold': {'curvature_threshold': .5}}},
                         video_worker_options={'cfg_fold': {'curvature_threshold': .02}})
            path.write_text(json.dumps(raw)); entry['sha256'] = stream.file_hash(path)
            result = fixture.evaluate()
            row = next(row for row in result['pair_results'] if row['pair_id'] == entry['pair_id'] and row['arm_id'] == 'adaptive')
            self.assertEqual(row['status'], 'measured', row)

    def test_resealed_invalid_rule_and_calibration_bindings_are_rejected(self):
        mutations = [
            lambda p: p['rule'].update(eligible_steps=list(range(12, 19))),
            lambda p: p['rule'].update(curvature_threshold=True),
            lambda p: p['rule']['numeric_thresholds'].update(min_ssim=2),
            lambda p: p['calibration_sources'][p['shape_ids'][0]].update(core_sha256='bad'),
            lambda p: p['calibration_sources'][p['shape_ids'][0]].update(core_sha256='0' * 64),
            lambda p: p['calibration_sources'][p['shape_ids'][0]]['gpu_uuids'].reverse(),
            lambda p: p.update(calibration_bundle_sha256='bad'),
            lambda p: p['worker_model_bindings']['video'].update(model_path='/other/model'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory))
            for mutate in mutations:
                with self.subTest(mutation=mutate):
                    plan = copy.deepcopy(fixture.plan); mutate(plan)
                    plan['rule_sha256'] = canonical_hash(plan['rule']); plan = seal(plan)
                    with self.assertRaises(ValueError):
                        stream.validate_plan(plan, fixture.protocol)

    def test_frame_metrics_match_global_mse_and_ssim_without_full_video_copy(self):
        rng = np.random.default_rng(42)
        ref = rng.random((1, 3, 4, 4, 3), dtype=np.float32)
        other = ref.copy(); other[0, 1, 0, 0, 0] *= .5
        metrics = stream.frame_metrics(ref, other)
        delta = other.astype(np.float64) - ref.astype(np.float64)
        expected = float(np.mean(delta * delta))
        self.assertAlmostEqual(metrics['mse'], expected)
        self.assertAlmostEqual(metrics['psnr_db'], -10 * math.log10(expected))
        self.assertAlmostEqual(metrics['temporal_delta_rmse'], float(np.sqrt(np.mean(np.diff(delta, axis=1) ** 2))))
        self.assertAlmostEqual(metrics['ssim'], stream.ssim(ref, other))
        self.assertIsNone(metrics['lpips_mean'])
        for invalid in (np.full_like(ref, float('nan')), np.full_like(ref, 1.1)):
            with self.assertRaisesRegex(ValueError, 'finite and in'):
                stream.frame_metrics(ref, invalid)

    def test_array_reader_requires_readonly_memmap_and_closes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); fixture.one_pair()
            entry = fixture.entries[0]; pair = fixture.plan['pairs'][0]
            raw = json.loads((fixture.root / entry['path']).read_bytes())
            row = raw['requests'][0]
            with patch.object(stream.np, 'load', wraps=np.load) as loader:
                with stream.checked_array(row, pair, entry, fixture.root) as array:
                    self.assertIsInstance(array, np.memmap)
                    self.assertFalse(array.flags.writeable)
                self.assertTrue(array._mmap.closed)
                self.assertEqual(loader.call_args.kwargs, dict(mmap_mode='r', allow_pickle=False))

    def test_cli_records_exact_input_hashes_and_never_overwrites_existing_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); fixture.one_pair()
            for name, value in [('plan', fixture.plan), ('protocol', fixture.protocol), ('evidence', fixture.evidence())]:
                (fixture.root / (name + '.json')).write_text(json.dumps(value))
            output = fixture.root / 'summary.json'
            argv = ['p4_stream_quality.py', '--plan', str(fixture.root / 'plan.json'),
                '--protocol', str(fixture.root / 'protocol.json'), '--evidence', str(fixture.root / 'evidence.json'),
                '--output', str(output)]
            with patch.object(sys, 'argv', argv):
                stream.main()
                result = json.loads(output.read_bytes()); verify_seal(result)
                self.assertEqual(result['input_file_sha256'], {name: stream.file_hash(fixture.root / (name + '.json'))
                                 for name in ('plan', 'protocol', 'evidence')})
                original = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    stream.main()
                self.assertEqual(output.read_bytes(), original)


class PromptClusterBootstrapTests(unittest.TestCase):
    def test_adding_correlated_seeds_and_shapes_does_not_increase_independent_n(self):
        independent = [dict(prompt_sha256=str(i), difference=float(i)) for i in range(4)]
        repeated = [copy.deepcopy(row) for row in independent for _ in range(6)]
        one = stream.cluster_bootstrap(independent, samples=200, seed=7)
        six = stream.cluster_bootstrap(repeated, samples=200, seed=7)
        self.assertEqual(one['ci95'], six['ci95'])
        self.assertEqual(six['independent_prompt_clusters'], 4)
        self.assertEqual(six['observed_pairs'], 24)
        self.assertEqual(one['mean_difference'], six['mean_difference'])
        alone = stream.cluster_bootstrap(repeated[:6], samples=200, seed=7)
        self.assertEqual(alone['status'], 'unknown')
        self.assertIsNone(alone['ci95'])

    def test_one_sided_infinity_uses_finite_mse_domain_without_numeric_psnr_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); pair = fixture.plan['pairs'][0]
            for arm in fixture.plan['arms']:
                fixture.produce(pair, arm, value=.6 if arm['id'] == 'fixed-1' else .5, adaptive_folds=1)
            result = fixture.evaluate()['matched_fold_comparison']
            self.assertEqual(result['psnr']['relations'], {'candidate_exact_control_finite': 1})
            self.assertIsNone(result['psnr']['mean_difference'])
            self.assertLess(result['groups']['all']['mse']['mean_difference'], 0)
            self.assertEqual(result['zero_fold_exact_self_controls'], 0)
            self.assertFalse(result['superiority_qualified'])


if __name__ == '__main__':
    unittest.main()
