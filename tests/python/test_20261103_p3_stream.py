"""Fixed-horizon Poisson, distinct input budgets and censored native windows."""

import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import p3_stream as p3
import p3_window_review as review
import test_20261102_formal_benchmark as formal_fixtures
import test_20261103_p2_benefit as p2_fixtures
from formal_benchmark import split_pools
from p2_contrasts import AUDITED_NATIVE_SOURCES
from stage_gates import canonical_hash
from ditserve.runtime.client import CoreBridge


class P3StreamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        formal_fixtures.FormalBenchmarkTest.setUpClass()
        cls.f = formal_fixtures.FormalBenchmarkTest()

    def inputs(self):
        base = self.f.build()
        base['phases'][0]['matrix']['/policy'] = ['edf-cfg2', 'block-cfg2']
        usage = dict(schema='ditserve.prompt-usage.v1', protocol_sha256=canonical_hash(self.f.p),
                     normalized_prompt_sha256=dict(image=[], video=[]))
        shapes = base['benchmark']['formal_protocol']['shape_ids']
        weights = {kind: {sid: .5 for sid in shapes if self.f.p['shapes'][sid]['kind'] == kind}
                   for kind in ('image', 'video')}
        plan = dict(schema=p3.SCHEMA, protocol_sha256=canonical_hash(self.f.p), model='wan21', split='test',
            shape_ids=shapes, seeds=[301, 302, 303], prompt_reuse='paired_policies_only', policy_order='seed_randomized',
            prompt_usage_sha256=canonical_hash(usage), prompt_pools_sha256={kind: pool['sha256'] for kind, pool in self.f.pools.items()},
            arrival=dict(kind='poisson', rate_rps=4., video_probability=.2, shape_probabilities=weights),
            windows=dict(traffic_warmup_ms=10000, measurement_ms=20000, drain_ms=5000, bin_ms=5000),
            startup_allowance_seconds=600,
            statistics=dict(confidence=.95, resampling_unit='seed', minimum_independent_seeds=3,
                            bootstrap_samples=1000, bootstrap_seed=20261103))
        return base, plan, usage

    def prepared(self):
        base, plan, usage = self.inputs()
        configs, reserved = p3.prepare(base, self.f.p, plan, self.f.pools, usage)
        return configs, reserved, plan

    def test_poisson_has_random_count_and_prefix_is_never_time_compressed(self):
        base, plan, usage = self.inputs()
        caps = dict(image=10000, video=10000)
        streams = [p3.arrivals(plan, seed, caps) for seed in plan['seeds']]
        self.assertGreater(len({len(events) for events in streams}), 1)
        extended = copy.deepcopy(plan); extended['windows']['measurement_ms'] += 10000
        longer = p3.arrivals(extended, 301, caps)
        self.assertEqual([event for event in longer if event[0] < 30000], streams[0])
        self.assertGreater(streams[0][0][0], 0)
        self.assertLess(streams[0][-1][0], 30000)
        doubled = copy.deepcopy(plan); doubled['arrival']['rate_rps'] *= 2
        fast = p3.arrivals(doubled, 301, caps)
        for kind in ('image', 'video'):
            left = [event[0] for event in streams[0] if event[1] == kind][:5]
            right = [event[0] for event in fast if event[1] == kind][:5]
            for a, b in zip(left, right):
                self.assertAlmostEqual(a, 2 * b)

    def test_generated_native_configs_keep_arrivals_inputs_and_phase_boundaries(self):
        configs, reservation, plan = self.prepared()
        seen = dict(image=set(), video=set())
        for seed, config in configs.items():
            phase = config['phases'][0]; rows = phase['matrix']['/workload'][0]
            expected = p3.arrivals(plan, seed, dict(image=10000, video=10000))
            self.assertEqual([(row['arrival_ms'], row['kind'], row['class']) for row in rows], expected)
            self.assertEqual(config['benchmark']['p3']['plan_sha256'], canonical_hash(plan))
            self.assertEqual(config['benchmark']['formal_protocol']['role'], 'p3_sustained')
            self.assertTrue(phase['warmup'])
            self.assertNotIn('load', config['benchmark'])
            for row in rows:
                self.assertEqual(row['deadline_ms'], row['arrival_ms'] + row['slo_ms'])
                self.assertEqual(row['p3_cohort'], 'traffic_warmup' if row['arrival_ms'] < 10000 else 'measurement')
                digest = p3.prompt_hash(row['spec']['prompt'])
                self.assertNotIn(digest, seen[row['kind']]); seen[row['kind']].add(digest)
        self.assertEqual(reservation['reserved_prompts'], {kind: len(values) for kind, values in seen.items()})

    @unittest.skipUnless(CoreBridge().available(), 'build the C++ core first')
    def test_native_workload_parser_preserves_sustained_replay_timestamps(self):
        configs, _, _ = self.prepared()
        phase = review.expanded(configs[301]['phases'][0])[0]
        parsed = CoreBridge().call('control', dict(op='workload', phase=phase))
        expected = {row['request_id']: row for row in phase['workload']}
        self.assertEqual(len(parsed), len(expected))
        for row in parsed:
            for field in ('arrival_ms', 'deadline_ms', 'p3_cohort', 'spec', 'dataset_sample_id'):
                self.assertEqual(row[field], expected[row['request_id']][field])

    def test_entire_plan_prompt_shortage_rejects_without_recycling_or_shortening(self):
        base, plan, usage = self.inputs()
        selected = split_pools(self.f.pools, self.f.p, 'test')
        needed = sum(sum(event[1] == 'image' for event in p3.arrivals(plan, seed, dict(image=10000, video=10000)))
                     for seed in plan['seeds'])
        samples = selected['image']['samples']
        usage['normalized_prompt_sha256']['image'] = [p3.prompt_hash(sample['prompt']) for sample in samples[needed-1:]]
        plan['prompt_usage_sha256'] = canonical_hash(usage)
        before = copy.deepcopy((base, plan, self.f.pools, usage))
        with self.assertRaisesRegex(ValueError, 'insufficient independent image prompts'):
            p3.prepare(base, self.f.p, plan, self.f.pools, usage)
        self.assertEqual((base, plan, self.f.pools, usage), before)

    def test_drain_does_not_resample_arrivals_and_stress_slo_cannot_be_relabelled(self):
        base, plan, usage = self.inputs()
        first, _ = p3.prepare(base, self.f.p, plan, self.f.pools, usage)
        plan['windows']['drain_ms'] *= 2
        second, _ = p3.prepare(base, self.f.p, plan, self.f.pools, usage)
        self.assertEqual(first[301]['phases'][0]['matrix']['/workload'], second[301]['phases'][0]['matrix']['/workload'])
        base['phases'][0]['matrix']['/workload'][0][0]['effective_slo_scale'] = 1.1
        with self.assertRaisesRegex(ValueError, 'formal per-shape SLOs'):
            p3.prepare(base, self.f.p, plan, self.f.pools, usage)

    def record(self, config):
        phase = config['phases'][0]; seed = config['benchmark']['seed']
        run_id, session = f'p3-{seed}', f'p3-session-{seed}'
        record = dict(schema='ditserve.run.v1', run_id=run_id, status='completed', config=config,
            source=dict(core_sha256='c'*64, files={'ditserve/runtime/worker.py': 'b'*64},
                        native_files={path: hashes[0] for path, hashes in AUDITED_NATIVE_SOURCES.items()}),
            budget=dict(sessions=[dict(session_id=session, phase=phase['id'], devices=len(self.f.uuids),
                identity=dict(attempt=1, run_id=run_id, gpu_uuids=self.f.uuids))]),
            phases=[dict(id=phase['id'], attempt=1, session_id=session, cells=[])], requests=[])
        for cell, policy in enumerate(phase['matrix']['/policy']):
            rows = []
            for expected in phase['matrix']['/workload'][0]:
                row = copy.deepcopy(expected); video = row['kind'] == 'video'
                worker = config['workers']['wan0' if video else 'image0']['config']
                generation = dict(row['spec'], model_path=worker['model_path'], revision='fixture',
                    dtype='float16' if video else 'bfloat16', vae_dtype='bfloat16', text_encoder_dtype='float32',
                    attention='flash', engine='fixture', xdit_revision='fixture', approximate=False,
                    runtime_identity={'python': 'fixture'}, model_manifest_sha256='a'*64,
                    model_integrity='manifest_verified', seed_protocol='per-request', batch_protocol='single')
                row.update(phase=phase['id'], attempt=1, cell=cell, order='forward', status='completed',
                    start_ms=row['arrival_ms'] + 10, finish_ms=row['arrival_ms'] + 200,
                    forms=['cfg2'] if video else [], outputs=[dict(generation=generation, output=dict(
                        output_mode='check', finite=True, dtype='float32', shape=([1, row['spec']['frames'], row['spec']['height'], row['spec']['width'], 3]
                            if video else [1, row['spec']['height'], row['spec']['width'], 3])))])
                rows.append(row)
            record['requests'] += rows
            record['phases'][0]['cells'].append(dict(cell=cell, order='forward', policy=policy, status='completed',
                gpus=len(self.f.uuids), elapsed_ms=max(row['finish_ms'] for row in rows) + 1))
        return record

    def evidence(self, root):
        configs, reservation, plan = self.prepared()
        write = p2_fixtures.write
        manifest = dict(schema='ditserve.p3-evidence.v1', plan=write(root, 'plan.json', plan),
            protocol=write(root, 'protocol.json', self.f.p), configs=[], runs=[])
        for seed, config in configs.items():
            manifest['configs'].append(dict(seed=seed, config=write(root, f'config-{seed}.json', config)))
            manifest['runs'].append(dict(seed=seed, attempt=1, record=write(root, f'run-{seed}.json', self.record(config))))
        return manifest

    def test_raw_window_review_uses_seed_intervals_and_never_asserts_stable_capacity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root)
            result = review.evaluate(manifest, root)
            self.assertEqual(result['status'], 'reviewed', [(row['status'], row.get('reason')) for row in result['cases']])
            self.assertEqual(result['expected_cases'], 6)
            self.assertFalse(result['stable_capacity_claim'])
            self.assertEqual(result['allows'], [])
            for repeated in result['repetitions']:
                for kind in ('image', 'video'):
                    self.assertEqual(repeated['intervals'][kind]['attainment_lower']['seed_count'], 3)
                    self.assertEqual(repeated['intervals'][kind]['attainment_lower']['ci95'], [1., 1.])

    def test_failed_first_cell_and_unstarted_second_cell_retain_arrival_cohorts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root)
            entry = manifest['runs'][0]; path = entry['record']['path']
            record = json.loads((root/path).read_bytes()); record['status'] = 'failed'
            record['phases'][0]['cells'] = [dict(cell=0, status='failed', error='fixture abort')]
            record['requests'] = [row for row in record['requests'] if row['cell'] == 0]
            for row in record['requests']:
                row['observation_ms'] = 20000
                if row['finish_ms'] > 20000:
                    row['status'] = 'interrupted'; row['finish_ms'] = 20000; row.pop('outputs')
            entry['record'] = p2_fixtures.write(root, path, record)
            result = review.evaluate(manifest, root)
            first, missing = result['cases'][:2]
            self.assertEqual(first['status'], 'failed')
            self.assertEqual(first['observed_horizon_ms'], 20000)
            self.assertIsNone(first['queue']['image']['growth_requests_per_second'])
            self.assertEqual(missing['status'], 'not_started')
            self.assertEqual(first['by_kind']['image']['planned'], missing['by_kind']['image']['planned'])
            self.assertGreater(missing['by_kind']['image']['not_executed'], 0)
            self.assertTrue(any(row['by_kind']['image']['right_censored'] > 0 for row in result['cases']))

    def test_rehashed_normalized_arrivals_are_rejected_and_missing_runs_stay_pending(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root)
            empty = copy.deepcopy(manifest); empty['runs'] = []
            result = review.evaluate(empty, root)
            self.assertEqual(result['status'], 'incomplete')
            self.assertTrue(all(row['status'] == 'not_started' for row in result['cases']))
            entry = manifest['configs'][0]; config = json.loads((root/entry['config']['path']).read_bytes())
            rows = config['phases'][0]['matrix']['/workload'][0]
            for row in rows:
                row['arrival_ms'] *= .5; row['deadline_ms'] = row['arrival_ms'] + row['slo_ms']
            trace = config['phases'][0]['matrix']['/trace_provenance'][0]
            trace['sha256'] = trace['manifest']['workload_sha256'] = p3.trace_hash(rows)
            config['benchmark']['trace'] = trace
            entry['config'] = p2_fixtures.write(root, entry['config']['path'], config)
            with self.assertRaisesRegex(ValueError, 'actual arrivals'):
                review.evaluate(manifest, root)

    def test_prepare_and_review_reject_the_same_rehashed_invalid_plans(self):
        def change(plan, field, value):
            target = plan
            for part in field[:-1]:
                target = target[part]
            target[field[-1]] = value
        mutations = [
            (('split',), 'calibration'), (('model',), 'unknown'),
            (('seeds',), [301, 302, 999]), (('seeds',), [301, 302, True]),
            (('shape_ids',), ['wan21-small', 'sd35-512']),
            (('arrival', 'shape_probabilities', 'video'), {'wan21-small': 1.}),
            (('arrival', 'video_probability'), True),
            (('arrival', 'rate_rps'), -1), (('windows', 'drain_ms'), -1),
            (('statistics', 'bootstrap_samples'), 1000.5),
            (('statistics', 'bootstrap_seed'), 'undeclared'),
            (('prompt_reuse',), 'cycle')]
        for field, value in mutations:
            with self.subTest(field=field), TemporaryDirectory() as directory:
                base, plan, usage = self.inputs()
                change(plan, field, value)
                with self.assertRaises(ValueError):
                    p3.prepare(base, self.f.p, plan, self.f.pools, usage)
                root = Path(directory); manifest = self.evidence(root)
                manifest['plan'] = p2_fixtures.write(root, 'tampered-plan.json', plan)
                # All hashes are refreshed: rejection must follow the protocol rules.
                for entry in manifest['configs']:
                    config = json.loads((root/entry['config']['path']).read_bytes())
                    config['benchmark']['p3'].update(plan_sha256=canonical_hash(plan),
                        windows=plan['windows'], arrival=plan['arrival'])
                    entry['config'] = p2_fixtures.write(root, entry['config']['path'], config)
                with self.assertRaises(ValueError):
                    review.evaluate(manifest, root)

    def test_review_rederives_slo_and_spec_after_all_trace_hashes_are_refreshed(self):
        for field in ('deadline_ms', 'service_ms', 'slo_ms', 'effective_slo_scale', 'consistent_slo', 'spec'):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                root = Path(directory); manifest = self.evidence(root)
                manifest['runs'] = []  # Validation cannot depend on having a completed run.
                entry = manifest['configs'][0]
                config = json.loads((root/entry['config']['path']).read_bytes())
                phase = config['phases'][0]; rows = phase['matrix']['/workload'][0]
                row = rows[0]
                if field == 'consistent_slo':
                    row['service_ms'] *= 2; row['slo_ms'] *= 2
                    row['deadline_ms'] = row['arrival_ms'] + row['slo_ms']
                elif field == 'spec':
                    row['spec']['height'] *= 2
                else:
                    row[field] *= 2
                trace = phase['matrix']['/trace_provenance'][0]
                trace['sha256'] = trace['manifest']['workload_sha256'] = p3.trace_hash(rows)
                config['benchmark']['trace'] = trace
                entry['config'] = p2_fixtures.write(root, entry['config']['path'], config)
                with self.assertRaisesRegex(ValueError, 'formal per-shape SLOs and specs'):
                    review.evaluate(manifest, root)

    def test_review_rejects_config_split_and_shape_scope_even_with_valid_plan(self):
        for field, value in [('split', 'development'), ('shape_ids', ['wan21-dev', 'sd35-dev'])]:
            with self.subTest(field=field), TemporaryDirectory() as directory:
                root = Path(directory); manifest = self.evidence(root)
                entry = manifest['configs'][0]
                config = json.loads((root/entry['config']['path']).read_bytes())
                config['benchmark']['formal_protocol'][field] = value
                entry['config'] = p2_fixtures.write(root, entry['config']['path'], config)
                with self.assertRaisesRegex(ValueError, 'protocol/split/shapes'):
                    review.evaluate(manifest, root)

    def test_cli_writes_replay_configs_reservation_and_review_without_overwrite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); base, plan, usage = self.inputs()
            objects = dict(config=base, protocol=self.f.p, plan=plan, usage=usage,
                           image_pool=self.f.pools['image'], video_pool=self.f.pools['video'])
            args = ['p3_stream.py', '--output-dir', str(root/'prepared')]
            for key, value in objects.items():
                path = root/(key+'.json'); p2_fixtures.write(root, path.name, value)
                args += ['--'+key.replace('_','-'), str(path)]
            with patch.object(sys, 'argv', args):
                p3.main()
            reservation = json.loads((root/'prepared/prompt-reservation.json').read_bytes())
            self.assertGreater(reservation['reserved_prompts']['image'], 0)
            self.assertIn('reservation', reservation['scope'])
            with patch.object(sys, 'argv', args), self.assertRaises(FileExistsError):
                p3.main()
            args = ['p3_window_review.py', '--manifest', str(root/'prepared/evidence.json'), '--output', str(root/'review.json')]
            with patch.object(sys, 'argv', args):
                review.main()
            self.assertEqual(json.loads((root/'review.json').read_bytes())['status'], 'incomplete')
            with patch.object(sys, 'argv', args), self.assertRaises(FileExistsError):
                review.main()


class WindowAccountingTest(unittest.TestCase):
    def test_arrival_cohort_drain_and_completion_window_are_distinct(self):
        workload = [dict(request_id='warm', kind='video', arrival_ms=9000, deadline_ms=18000),
                    dict(request_id='measured', kind='video', arrival_ms=19000, deadline_ms=24000)]
        rows = [dict(workload[0], status='completed', finish_ms=17000),
                dict(workload[1], status='completed', finish_ms=22000)]
        windows = dict(traffic_warmup_ms=10000, measurement_ms=10000, drain_ms=5000, bin_ms=5000)
        result = review.window_summary(workload, rows, dict(status='completed', elapsed_ms=22000), windows)
        self.assertEqual(result['by_kind']['video']['planned'], 1)
        self.assertEqual(result['by_kind']['video']['ontime'], 1)
        self.assertEqual(result['metrics']['video']['latency_p95_ms'], 3000)
        self.assertEqual(result['metrics']['video']['completion_rps_during_measurement'], .1)
        self.assertEqual(result['queue']['video']['bins'][0]['unfinished'], 1)
        self.assertEqual(result['queue']['video']['bins'][-1]['unfinished'], 1)
        self.assertFalse(result['actual_drain_cutoff_enforced'])
        short = review.window_summary(workload, rows, dict(status='completed', elapsed_ms=22000), dict(windows, drain_ms=0))
        self.assertEqual(short['by_kind']['video']['right_censored'], 1)
        self.assertEqual(short['metrics']['video']['attainment_lower'], 0)
        self.assertEqual(short['metrics']['video']['attainment_upper'], 1)
        self.assertIsNone(short['metrics']['video']['latency_p95_ms'])


if __name__ == '__main__':
    unittest.main()
