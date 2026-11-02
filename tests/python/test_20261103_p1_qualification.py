"""P1 raw-evidence review cannot inherit P0 stress or manufacture promotion."""

import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import p1_benchmark
import p1_qualification as p1
import stage_gates
from ditserve.evaluation.common import canonical_hash, read_json, seal
import test_20261102_formal_benchmark as formal_fixtures
from test_20261102_stage_gates import fixture as p0_fixture, write


class P1QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        formal_fixtures.FormalBenchmarkTest.setUpClass()
        cls.f = formal_fixtures.FormalBenchmarkTest()
        cls.plan = read_json(p1_benchmark.DEFAULT_PLAN)
        cls.python = {'ditserve/runtime/worker.py': 'b' * 64}
        cls.identity = ({'build': {'commit': 'a' * 40, 'core_sha256': 'c' * 64},
                         'protocol_sha256': canonical_hash(cls.f.p)}, cls.python,
                        {'wan21': 'a' * 64, 'sd35': 'a' * 64})

    def run_record(self, role='contention', seed=201, *, protocol=None, plan=None):
        protocol, plan = protocol or self.f.p, plan or self.plan
        bundle = copy.deepcopy(self.f.bundle())
        bundle['protocol_sha256'] = canonical_hash(protocol)
        config = p1_benchmark.case_config(protocol, plan, 'wan21-0', role, seed, self.f.uuids,
            seal(bundle), self.f.pools, self.f.manifests)
        record = dict(schema='ditserve.run.v1', run_id=f'p1-{role}-{seed}', status='completed',
            source=dict(core_sha256='c' * 64, files=self.python), config=config, requests=[],
            budget=dict(sessions=[dict(identity=dict(gpu_uuids=list(self.f.uuids)))]),
            phases=[dict(id='cluster', cells=[])])
        for cell, policy in enumerate(config['phases'][0]['matrix']['/policy']):
            record['phases'][0]['cells'].append(dict(cell=cell, order=cell, policy=policy,
                                                    status='completed', elapsed_ms=200000, gpus=len(self.f.uuids)))
            for expected in config['phases'][0]['matrix']['/workload'][0]:
                row = copy.deepcopy(expected)
                is_video = row['kind'] == 'video'
                worker = config['workers']['wan0' if is_video else 'image0']['config']
                generation = dict(row['spec'], model_path=worker['model_path'], revision='frozen',
                    dtype='float16' if is_video else 'bfloat16', vae_dtype='bfloat16',
                    text_encoder_dtype='float32' if is_video else 'bfloat16', attention='flash',
                    engine='xdit' if is_video else {'revision': 'xdit-revision'}, xdit_revision='xdit-revision',
                    runtime_identity={'python': 'fixture', 'packages': {'torch': 'fixture'}},
                    approximate=False, model_manifest_sha256='a' * 64, model_integrity='manifest_verified',
                    seed_protocol='per-request', batch_protocol='single')
                latency = 1000 if policy == 'edf-cfg2' else 1100 + (seed - 201) * 10
                row.update(cell=cell, phase='cluster', status='completed', forms=['cfg2'] if is_video else [],
                    start_ms=row['arrival_ms'] + 100, finish_ms=row['arrival_ms'] + latency,
                    outputs=[dict(generation=generation, output=dict(output_mode='digest', finite=True,
                        array_sha256='d' * 64, dtype='float32', shape=([1, row['spec']['frames'], row['spec']['height'], row['spec']['width'], 3]
                            if is_video else [1, row['spec']['height'], row['spec']['width'], 3])))])
                if is_video and role == 'contention' and policy == 'block-cfg2':
                    row['preemption_stop_samples'] = [{'stop_ms': 10}]
                record['requests'].append(row)
        return record

    def audit(self, record, role='contention', seed=201):
        return p1.audit_run(record, self.plan['shape_pairs'][0], role, seed, self.f.p, self.plan, self.identity)

    def test_raw_formal_pair_counts_and_negative_latency_effect_are_reported(self):
        value = self.audit(self.run_record())
        self.assertEqual(value['status'], 'verified_fixed_form', value.get('reason'))
        self.assertEqual(value['expected_requests'], 80)
        self.assertEqual(value['completed_requests'], 80)
        self.assertEqual(value['by_kind']['image']['paired_requests'], 20)
        self.assertGreater(value['metrics']['video_mean_latency_ratio'], 1)
        self.assertEqual(value['partial_costs']['transfer_export_ms']['requests_with_measurement'], 0)

    def test_p0_pressure_and_summary_only_evidence_cannot_replace_raw_p1(self):
        mutations = [lambda r: r['config']['phases'][0]['options'].__setitem__('lazy_preemption', False),
                     lambda r: r['config']['benchmark']['p1']['controls'].__setitem__('image_slo_scale', 1.1),
                     lambda r: r['source']['files'].__setitem__('unverified.py', 'f' * 64),
                     lambda r: r.__setitem__('requests', []),
                     lambda r: [row.pop('preemption_stop_samples', None) for row in r['requests']]]
        for mutate in mutations:
            record = copy.deepcopy(self.run_record())
            mutate(record)
            with self.subTest(mutate=mutate):
                self.assertEqual(self.audit(record)['status'], 'blocked')

    def test_image_digest_identity_and_incomplete_inputs_fail_closed(self):
        for field, value in [('array_sha256', 'e' * 64), ('finite', False)]:
            record = self.run_record()
            image = next(row for row in record['requests'] if row['kind'] == 'image')
            image['outputs'][-1]['output'][field] = value
            self.assertEqual(self.audit(record)['status'], 'blocked')
        record = self.run_record()
        image = next(row for row in record['requests'] if row['kind'] == 'image')
        image['outputs'][-1]['generation'].pop('attention')
        self.assertEqual(self.audit(record)['status'], 'blocked')

    def test_allocation_and_same_wrong_output_dimensions_cannot_pass(self):
        record = self.run_record()
        record['budget']['sessions'][0]['identity']['gpu_uuids'].reverse()
        self.assertEqual(self.audit(record)['status'], 'blocked')
        record = self.run_record()
        record['phases'][0]['cells'][0]['gpus'] = 2
        self.assertEqual(self.audit(record)['status'], 'blocked')
        record = self.run_record()
        for row in record['requests']:
            row['outputs'][-1]['output']['shape'][-2] += 8
        self.assertEqual(self.audit(record)['status'], 'blocked')

    def test_failed_requests_stay_in_denominator_and_zero_stop_is_role_specific(self):
        record = self.run_record()
        record['status'] = 'failed'
        record['requests'][0]['status'] = 'failed'
        result = self.audit(record)
        self.assertEqual(result['expected_requests'], 80)
        self.assertEqual(result['completed_requests'], 79)
        self.assertEqual(result['failed_requests'], 1)
        self.assertEqual(result['status'], 'blocked')
        result = self.audit(self.run_record('no_preemption'), 'no_preemption')
        self.assertEqual(result['status'], 'verified_fixed_form', result.get('reason'))
        record = self.run_record('no_preemption')
        record['requests'][0]['preemption_stop_samples'] = [{'stop_ms': 1}]
        self.assertEqual(self.audit(record, 'no_preemption')['status'], 'blocked')

    def test_seed_intervals_are_deterministic_and_need_three_independent_seeds(self):
        settings = self.plan['statistics']
        self.assertIsNone(p1.seed_interval([1, 1.1], settings)['ci95'])
        measured = p1.seed_interval([1.1, 1.11, 1.12], settings)
        self.assertEqual(measured, p1.seed_interval([1.1, 1.11, 1.12], settings))
        self.assertEqual(measured['seed_count'], 3)
        self.assertGreater(measured['ci95'][0], 1)

    def evidence(self, root):
        old = p0_fixture(root)
        tiny_protocol = json.loads((root / 'protocol.json').read_bytes())
        protocol = copy.deepcopy(self.f.p)
        # A small explicit P0 fixture, not a claim of real 40-request execution.
        protocol['p0_required_cases'] = tiny_protocol['p0_required_cases']
        protocol['p0_diagnostics'] = tiny_protocol['p0_diagnostics']
        old['protocol'] = write(root, 'protocol.json', protocol)
        for entry in old['runs']:
            record = json.loads((root / entry['path']).read_bytes())
            record['config']['benchmark']['formal_protocol']['sha256'] = canonical_hash(protocol)
            record['config']['benchmark']['model_manifests'] = self.f.manifests
            record['config']['workers']['wan0']['config'].update(
                model_manifest_sha256='a' * 64, model_manifest_path=self.f.manifests['wan21']['path'])
            for row in record['requests']:
                row['outputs'][-1]['generation']['model_manifest_sha256'] = 'a' * 64
            entry.update(write(root, entry['path'], record))
        receipt = stage_gates.evaluate(old, root)
        self.assertEqual(receipt['stages']['P0']['status'], 'passed')
        receipt['source_manifest'] = write(root, 'p0-evidence.json', old)
        plan = copy.deepcopy(self.plan)
        plan['protocol_sha256'] = canonical_hash(protocol)
        manifest = dict(schema='ditserve.p1-evidence.v1', protocol=old['protocol'],
            p0_receipt=write(root, 'p0-receipt.json', receipt), plan=write(root, 'p1-plan.json', plan), runs=[])
        for seed in plan['seeds']:
            record = self.run_record('no_preemption', seed, protocol=protocol, plan=plan)
            manifest['runs'].append(dict(pair_id='wan21-0', role='no_preemption', seed=seed,
                **write(root, f'p1-{seed}.json', record)))
        return manifest

    def test_manifest_review_preserves_missing_cases_and_never_promotes_p1(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.evidence(root)
            result = p1.evaluate(manifest, root)
            self.assertEqual(result['status'], 'pending')
            self.assertEqual(result['allows'], [])
            self.assertEqual(result['expected_runs'], 24)
            self.assertEqual(result['expected_requests'], 1920)
            self.assertEqual(result['supplied_runs'], 3)
            repeated = next(row for row in result['repetitions'] if row['pair_id'] == 'wan21-0' and row['role'] == 'no_preemption')
            self.assertEqual(repeated['verified_seeds'], 3, result['checks'])
            self.assertGreater(repeated['intervals']['video_mean_latency_ratio']['ci95'][0], 1)
            self.assertTrue(all(row['status'] == 'pending' for row in result['formal_shapes']))
            self.assertEqual(result['quality']['status'], 'pending')
            self.assertEqual(result['performance_thresholds']['status'], 'pending')
            self.assertEqual(result['cost_decomposition']['status'], 'partial')
            manifest['runs'].append(dict(manifest['runs'][0]))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                p1.evaluate(manifest, root)

    def test_corrupt_run_source_or_unpassed_p0_cannot_enter_review(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.evidence(root)
            (root / manifest['runs'][0]['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                p1.evaluate(manifest, root)
            (root / 'p0-receipt.json').write_text('{}')
            with self.assertRaises(ValueError):
                p1.evaluate(manifest, root)


if __name__ == '__main__':
    unittest.main()
