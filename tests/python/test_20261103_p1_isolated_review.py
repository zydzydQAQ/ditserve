"""Offline evidence fixtures, not GPU or scientific qualification results."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import p1_isolated_batch as isolated
import p1_isolated_review as review
import test_20261103_p1_isolated_batch as fixtures
from ditserve.evaluation.common import canonical_hash


class IsolatedReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.IsolatedBatchTest.setUpClass()
        cls.inputs = fixtures.IsolatedBatchTest().inputs()
        cls.config = isolated.compile_case(cls.inputs)

    @staticmethod
    def record(inputs, config):
        """Match native retained raw layout; all numeric/digest values synthetic."""
        source = dict(core_sha256=inputs['identity']['core_sha256'],
                      files=inputs['identity']['python_sources'], native_files={'src/control.cpp': 'f'*64})
        record = dict(schema='ditserve.run.v1', experiment=config['experiment'], run_id='review-fixture',
                      status='completed', failures=[], config=config, config_sha256=canonical_hash(config),
                      source=source, phases=[], requests=[], budget=dict(sessions=[], active=None))
        runtime = inputs['calibrations'][0]['requests'][0]['outputs'][0]['generation']['runtime_identity']
        binding = config['workers']['wan0']['config']
        for phase_index, phase in enumerate(config['phases']):
            pid = phase['id']
            session_id = 'session-' + pid
            record['budget']['sessions'].append(dict(session_id=session_id, phase=pid, devices=2,
                status='completed', cleanup_verified=True, identity=dict(run_id=record['run_id'], attempt=1,
                    gpu_uuids=inputs['gpu_uuids'], config_sha256=canonical_hash(config),
                    source_sha256=canonical_hash(source))))
            native_phase = dict(id=pid, attempt=1, session_id=session_id, status='completed', cells=[])
            record['phases'].append(native_phase)
            for plan in (c for c in isolated.planned_cells(config) if c['phase'] == pid):
                pos = plan['position']
                cell = dict(cell=plan['cell'], order=plan['order'], policy=plan['policy'],
                    gpus=2, status='completed', elapsed_ms=600,
                    doorbell=dict(enabled=True, rings=0, continuous_quanta=400 if plan['cell'] else 0),
                    warmup=[dict(worker='wan0', elapsed_ms=8000), dict(worker='wan1', elapsed_ms=7900)])
                native_phase['cells'].append(cell)
                for index, planned in enumerate(phase['matrix']['/workload'][0]):
                    row = copy.deepcopy(planned)
                    spec = row['spec']
                    generation = dict(spec, model_path=binding['model_path'], revision=None,
                        dtype='bfloat16', vae_dtype='bfloat16', text_encoder_dtype='bfloat16',
                        attention='flash', engine='xfuser', xdit_revision='fixture', runtime_identity=runtime,
                        model_manifest_sha256=binding['model_manifest_sha256'], model_integrity='manifest_verified',
                        approximate=False, cfg_folded_steps=0, cache_enabled=False)
                    artifact = dict(output_mode='digest', finite=True, dtype='float32',
                        shape=[1, spec['frames'], spec['height'], spec['width'], 3],
                        array_sha256=canonical_hash(dict(seed=pid, request=planned['request_id'])))
                    # Linear order drift cancels when AB and BA differences are averaged.
                    finish = 400 + index*10 + pos*5 + (-20, 10, 40)[phase_index]*plan['cell']
                    row.update(phase=pid, cell=plan['cell'], order=plan['order'], attempt=1,
                        observation_id=f'{pid}/1/{pos}', status='completed', policy=plan['policy'],
                        forms=['cfg2'], degrees=[2, 1], resumes=49, start_ms=100, finish_ms=finish,
                        encode_model_ms=80, model_ms=200, reused_forwards=0,
                        outputs=[dict(backend='wan', measured=True, generation=generation, output=artifact,
                            progress=dict(done=True, step=spec['steps'], steps=spec['steps']))],
                        execution=[dict(planned_form='cfg2', group=[0, 1], planned_cache_threshold=0,
                            planned_cfg_folded=False, applied=dict(cfg=2, degree=2, ring=1,
                                cache_threshold=0, cfg_folded=False, reused_forwards=0))],
                        advance_samples=[dict(form='cfg2', group=[0, 1], cfg_folded=False)])
                    record['requests'].append(row)
        return record

    def fresh(self):
        return self.record(self.inputs, self.config)

    def run_review(self, record):
        return review.review_case(self.inputs, self.config, record)

    def test_abba_cancels_order_drift_and_preserves_negative_effects_and_seed_unit(self):
        value = self.run_review(self.fresh())
        self.assertEqual(value['status'], 'measured_diagnostic')
        self.assertEqual(value['coverage']['completed'], 96)
        self.assertEqual(value['verified_output_quartets'], 24)
        self.assertEqual(value['timing_eligible_requests'], 96)
        for metric in ('batch_last_completion_delta_ms', 'mean_request_latency_delta_ms'):
            self.assertEqual([s['metrics'][metric] for s in value['seeds']], [-20, 10, 40])
            interval = value['intervals'][metric]
            self.assertEqual((interval['seed_count'], interval['mean']), (3, 10))
            self.assertEqual(interval['timing_repeats_per_seed'], 2)
            self.assertIsNotNone(interval['ci95'])
        self.assertEqual(value['seeds'][0]['comparisons'][0]['deltas']['batch_last_completion_ms'], -15)
        self.assertEqual(value['seeds'][0]['comparisons'][1]['deltas']['batch_last_completion_ms'], -25)
        self.assertEqual(value['cells'][0]['metrics']['cell_replay_elapsed_ms'], 600)
        self.assertEqual(value['cells'][0]['metrics']['batch_last_completion_ms'], 470)
        self.assertEqual(len(value['cells'][0]['metrics']['warmup']), 2)
        self.assertFalse(value['qualification_eligible'])

    def test_same_policy_repeat_and_candidate_mismatch_block_whole_seed(self):
        for position in (1, 2, 3):
            record = self.fresh()
            record['requests'][position*8]['outputs'][0]['output']['array_sha256'] = 'a'*64
            value = self.run_review(record)
            with self.subTest(position=position):
                self.assertEqual(value['status'], 'incomplete')
                self.assertEqual(value['seeds'][0]['verified_output_quartets'], 7)
                self.assertEqual(value['timing_eligible_requests'], 64)
                for interval in value['intervals'].values():
                    self.assertEqual(interval['seed_count'], 2)
                    self.assertIsNone(interval['mean'])
                    self.assertIsNone(interval['ci95'])
                self.assertEqual(value['seeds'][1]['status'], 'measured_diagnostic')

    def test_single_rank_finish_is_valid_but_wrong_diffusion_form_group_or_fold_is_not(self):
        changes = [lambda r: r['requests'][0]['execution'][0]['group'].__setitem__(1, 2),
                   lambda r: r['requests'][0]['execution'][0]['applied'].__setitem__('cfg', 1),
                   lambda r: r['requests'][0]['advance_samples'][0].__setitem__('form', 'sp2'),
                   lambda r: r['requests'][0]['advance_samples'][0].__setitem__('cfg_folded', True),
                   lambda r: r['requests'][0].__setitem__('advance_samples', []),
                   lambda r: r['requests'][0]['outputs'][0]['progress'].__setitem__('step', 49)]
        for change in changes:
            record = self.fresh(); change(record)
            value = self.run_review(record)
            with self.subTest(change=change):
                self.assertEqual(value['cells'][0]['status'], 'unverified')
                self.assertEqual(value['timing_eligible_requests'], 64)
        self.assertEqual(self.run_review(self.fresh())['timing_eligible_requests'], 96)

    def test_degree_trace_permits_decode_one_but_not_out_of_allocation_or_missing_evidence(self):
        for degrees in ([2, 4], [], [1], None, [True, 2], ['2']):
            record = self.fresh(); record['requests'][0]['degrees'] = degrees
            with self.subTest(degrees=degrees):
                self.assertEqual(self.run_review(record)['timing_eligible_requests'], 64)

    def test_zero_duration_stop_and_missing_counter_are_not_zero_stop_evidence(self):
        record = self.fresh()
        record['requests'][0]['preemption_stop_samples'] = [dict(stop_ms=0)]
        value = self.run_review(record)
        self.assertEqual(value['cells'][0]['observed_stop_events'], 1)
        self.assertEqual(value['cells'][0]['zero_stop_status'], 'nonzero')
        self.assertEqual(value['timing_eligible_requests'], 64)
        record = self.fresh(); del record['phases'][0]['cells'][0]['doorbell']['rings']
        value = self.run_review(record)
        self.assertEqual(value['cells'][0]['zero_stop_status'], 'unknown')
        self.assertEqual(value['timing_eligible_requests'], 64)
        record = self.fresh(); record['phases'][0]['cells'][1]['doorbell']['rings'] = 1
        value = self.run_review(record)
        self.assertEqual(value['cells'][1]['zero_stop_status'], 'nonzero')

    def test_missing_block_path_and_false_whole_step_path_reject_timing(self):
        for position, value in ((0, 1), (1, 0)):
            record = self.fresh()
            record['phases'][0]['cells'][position]['doorbell']['continuous_quanta'] = value
            result = self.run_review(record)
            self.assertEqual(result['cells'][position]['status'], 'unverified')

    def test_source_runtime_weights_shape_generation_are_bound_even_if_all_four_copies_agree(self):
        def generation_change(record, key, value):
            for row in record['requests']:
                row['outputs'][0]['generation'][key] = value
        changes = [lambda r: generation_change(r, 'runtime_identity', dict(python='different')),
                   lambda r: generation_change(r, 'model_manifest_sha256', 'b'*64),
                   lambda r: generation_change(r, 'model_path', '/different'),
                   lambda r: generation_change(r, 'prompt', 'same wrong prompt'),
                   lambda r: generation_change(r, 'attention', 'different'),
                   lambda r: generation_change(r, 'approximate', True),
                   lambda r: [row['outputs'][0]['output']['shape'].__setitem__(1, 33) for row in r['requests']]]
        for change in changes:
            record = self.fresh(); change(record)
            with self.subTest(change=change):
                self.assertEqual(self.run_review(record)['timing_eligible_requests'], 0)
        for key in ('core_sha256', 'files'):
            record = self.fresh(); record['source'][key] = 'b'*64 if key == 'core_sha256' else {}
            with self.subTest(source=key), self.assertRaises(ValueError):
                self.run_review(record)

    def test_same_allocation_session_binds_config_source_phase_and_actual_two_devices(self):
        mutations = [lambda s: s.__setitem__('devices', 4),
                     lambda s: s.__setitem__('cleanup_verified', False),
                     lambda s: s.__setitem__('phase', 'other'),
                     lambda s: s['identity'].__setitem__('source_sha256', 'b'*64),
                     lambda s: s['identity'].__setitem__('config_sha256', 'b'*64),
                     lambda s: s['identity'].__setitem__('run_id', 'other')]
        for mutate in mutations:
            record = self.fresh(); mutate(record['budget']['sessions'][0])
            with self.subTest(mutate=mutate):
                self.assertEqual(self.run_review(record)['timing_eligible_requests'], 64)
        record = self.fresh(); record['budget']['sessions'][0]['identity']['gpu_uuids'] = ['GPU-c', 'GPU-d']
        with self.assertRaises(ValueError):
            self.run_review(record)

    def test_failed_prefix_preserves_all_96_and_excludes_incomplete_seed_estimates(self):
        record = self.fresh()
        record.update(status='failed', failures=[dict(phase='seed-201', error='injected')])
        record['phases'] = record['phases'][:1]
        record['phases'][0].update(status='failed', cells=[dict(cell=0, status='failed')])
        record['budget']['sessions'] = record['budget']['sessions'][:1]
        record['budget']['sessions'][0]['status'] = 'failed'
        record['requests'] = record['requests'][:8]
        for row in record['requests']:
            row['status'] = 'failed'; row['outputs'] = []
        value = self.run_review(record)
        self.assertEqual(value['expected_requests'], 96)
        self.assertEqual(value['coverage']['failed_or_incomplete'], 8)
        self.assertEqual(value['coverage']['unexecuted'], 88)
        self.assertEqual(value['failures'], record['failures'])
        self.assertEqual(len(value['cells']), 12)
        self.assertEqual(value['timing_eligible_requests'], 0)
        self.assertTrue(all(x['mean'] is None and x['ci95'] is None for x in value['intervals'].values()))

    def test_other_attempt_cannot_pollute_selected_cell_or_its_timing(self):
        record = self.fresh()
        row = copy.deepcopy(record['requests'][0]); row['attempt'] = 2
        # This other attempt is ignored by coverage; it must also be ignored by review.
        row['finish_ms'] = 999999
        record['requests'].append(row)
        self.assertEqual(self.run_review(record)['status'], 'measured_diagnostic')
        record = self.fresh(); record['requests'].append(copy.deepcopy(record['requests'][0]))
        with self.assertRaises(ValueError):
            self.run_review(record)

    def test_invalid_or_missing_time_is_not_inferred_from_model_or_elapsed(self):
        for value in (float('nan'), -1, 800, True, None):
            record = self.fresh(); record['requests'][0]['finish_ms'] = value
            with self.subTest(value=value):
                self.assertEqual(self.run_review(record)['timing_eligible_requests'], 64)

    def test_failed_run_cannot_emit_aggregate_interval_even_with_complete_cells(self):
        record = self.fresh()
        record.update(status='failed', failures=[dict(error='publication failure')])
        value = self.run_review(record)
        self.assertEqual(value['status'], 'incomplete')
        self.assertTrue(all(s['status'] == 'measured_diagnostic' for s in value['seeds']))
        self.assertTrue(all(x['mean'] is None and x['ci95'] is None for x in value['intervals'].values()))

    def test_active_allocation_blocks_aggregate_even_when_all_cells_completed(self):
        record = self.fresh()
        record['budget']['active'] = dict(session_id='unsettled-session', identity=dict(run_id=record['run_id']))
        value = self.run_review(record)
        self.assertEqual(value['status'], 'incomplete')
        self.assertFalse(value['allocation_settled'])
        self.assertTrue(all(s['status'] == 'measured_diagnostic' for s in value['seeds']))
        self.assertTrue(all(x['mean'] is None and x['ci95'] is None for x in value['intervals'].values()))
        del record['budget']['active']
        value = self.run_review(record)
        self.assertEqual(value['status'], 'incomplete')
        self.assertFalse(value['allocation_settled'])
        self.assertTrue(all(x['mean'] is None and x['ci95'] is None for x in value['intervals'].values()))

    @staticmethod
    def write(root, name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def manifest_fixture(self, root):
        inputs = copy.deepcopy(self.inputs)
        source = dict(schema=isolated.SCHEMA, pair_id=inputs['pair_id'], gpu_uuids=inputs['gpu_uuids'])
        for key in ('protocol', 'pair_plan', 'bundle', 'model_manifests', 'identity', 'runtime_facts'):
            source[key] = self.write(root, key+'.json', inputs[key])
        source['pools'] = {key: self.write(root, key+'.json', value) for key, value in inputs['pools'].items()}
        source['calibrations'] = [self.write(root, f'cal-{i}.json', value) for i, value in enumerate(inputs['calibrations'])]
        source_ref = self.write(root, 'inputs.json', source)
        loaded = isolated.load_inputs(Path(source_ref['path']), source_ref['sha256'])
        config = isolated.compile_case(loaded)
        run = self.record(loaded, config)
        entry = dict(pair_id=inputs['pair_id'], inputs=source_ref,
                     config=self.write(root, 'config.json', config), run=self.write(root, 'run.json', run))
        manifest = dict(schema=review.INPUT_SCHEMA, protocol=source['protocol'], pair_plan=source['pair_plan'], cases=[entry])
        return manifest, run, config

    def test_manifest_hash_binding_and_missing_pairs_never_reduce_global_denominator(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, run, config = self.manifest_fixture(root)
            value = review.evaluate(manifest, root)
            self.assertEqual(value['expected_requests'], 384)
            self.assertEqual(value['expected_cells'], 48)
            self.assertEqual([c['status'] for c in value['cases']], ['measured_diagnostic', 'missing', 'missing', 'missing'])
            self.assertEqual(sum(len(c['cells']) for c in value['cases']), 48)
            self.assertEqual(value['allows'], [])
            self.assertFalse(value['qualification_eligible'])
            self.assertFalse(value['stable_capacity_claim'])
            self.assertFalse(value['retention']['dependencies_closed'])
            self.assertFalse(value['retention']['deletion_authorized'])
            # Even byte-only formatting edits are rejected by the frozen reference.
            with Path(manifest['cases'][0]['run']['path']).open('a') as stream:
                stream.write('\n')
            value = review.evaluate(manifest, root)
            self.assertEqual(value['cases'][0]['status'], 'invalid')
            self.assertEqual(value['expected_requests'], 384)
            self.assertEqual(value['cases'][0]['expected_requests'], 96)
            self.assertEqual(len(value['cases'][0]['cells']), 12)

    def test_rehashed_relabelled_inputs_and_config_cannot_impersonate_the_frozen_case(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, run, config = self.manifest_fixture(root)
            config['benchmark']['preemption_diagnostic'] = 'no_preemption'
            run['config'] = config; run['config_sha256'] = canonical_hash(config)
            manifest['cases'][0]['config'] = self.write(root, 'changed-config.json', config)
            manifest['cases'][0]['run'] = self.write(root, 'changed-run.json', run)
            value = review.evaluate(manifest, root)
            self.assertEqual(value['cases'][0]['status'], 'invalid')
            self.assertEqual(value['expected_requests'], 384)
            manifest['cases'].append(copy.deepcopy(manifest['cases'][0]))
            with self.assertRaises(ValueError):
                review.evaluate(manifest, root)

    def test_all_formal_pairs_keep_shape_and_model_binding(self):
        for pair, inputs in fixtures.IsolatedBatchTest.examples.items():
            config = isolated.compile_case(inputs)
            with self.subTest(pair=pair):
                value = review.review_case(inputs, config, self.record(inputs, config))
                self.assertEqual((value['status'], value['expected_requests']), ('measured_diagnostic', 96))

    def test_cli_writes_once_and_binds_manifest_bytes_without_changing_raw(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _, _ = self.manifest_fixture(root)
            source = self.write(root, 'review-inputs.json', manifest)
            before = Path(manifest['cases'][0]['run']['path']).read_bytes()
            args = ['--manifest', source['path'], '--manifest-sha256', source['sha256'], '--output', str(root/'summary.json')]
            review.main(args)
            value = json.loads((root/'summary.json').read_text())
            self.assertEqual(value['source_manifest']['sha256'], source['sha256'])
            self.assertEqual(before, Path(manifest['cases'][0]['run']['path']).read_bytes())
            with self.assertRaises((ValueError, FileExistsError)):
                review.main(args)


if __name__ == '__main__':
    unittest.main()
