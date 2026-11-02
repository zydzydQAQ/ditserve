"""P2 pairs real identities and keeps failed/unstarted native cell denominators."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import p2_benefit as p2
import p2_contrasts as contrasts
import test_20261103_p1_qualification as fixtures
from stage_gates import canonical_hash


def write(root, name, value):
    raw = (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()
    (root / name).write_bytes(raw)
    return dict(path=name, sha256=hashlib.sha256(raw).hexdigest())


class P2BenefitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.P1QualificationTests.setUpClass()
        cls.f = fixtures.P1QualificationTests()

    def records(self, seed=201):
        old = self.f.run_record(seed=seed)
        configs = contrasts.make_configs(old['config'], 'block-fixed')
        result = {}
        for arm, policy in [('reference', 'edf-cfg2'), ('candidate', 'block-cfg2')]:
            cell = next(cell['cell'] for cell in old['phases'][0]['cells'] if cell['policy'] == policy)
            rows = copy.deepcopy([row for row in old['requests'] if row['cell'] == cell])
            for row in rows:
                row.update(phase='cluster', attempt=1, cell=0, order='forward')
            run_id = f'p2-{seed}-{arm}'
            session_id = run_id + '-session'
            result[arm] = dict(schema='ditserve.run.v1', status='completed', run_id=run_id,
                config=configs[arm], requests=rows,
                source=dict(core_sha256='c' * 64, files=self.f.python,
                            native_files={path: hashes[0] for path, hashes in contrasts.AUDITED_NATIVE_SOURCES.items()}),
                budget=dict(sessions=[dict(session_id=session_id, phase='cluster', devices=len(self.f.f.uuids),
                    identity=dict(gpu_uuids=self.f.f.uuids, attempt=1, run_id=run_id))]),
                phases=[dict(id='cluster', attempt=1, session_id=session_id, status='completed', cells=[dict(
                    cell=0, order='forward', policy=policy, status='completed', gpus=len(self.f.f.uuids),
                    elapsed_ms=max(row['finish_ms'] for row in rows) + 10)])])
        return result

    def evidence(self, root, *, mutate=None):
        cases, runs = [], []
        for seed in (201, 202, 203):
            records = self.records(seed)
            if mutate:
                mutate(records, seed)
            case = dict(id=f's{seed}', workload_id='wan21-formal-contention', contrast='block-fixed', seed=seed)
            for arm in ('reference', 'candidate'):
                record = records[arm]
                case[arm] = dict(config=write(root, f'{seed}-{arm}-config.json', record['config']),
                    phase='cluster', attempt=1, cell=0, order='forward')
                runs.append(dict(case_id=case['id'], arm=arm,
                                 record=write(root, f'{seed}-{arm}-run.json', record)))
            cases.append(case)
        plan = dict(schema='ditserve.p2-plan.v1', protocol_sha256=canonical_hash(self.f.f.p),
                    cases=cases, statistics=dict(self.f.plan['statistics'], bootstrap_samples=1000))
        return dict(schema='ditserve.p2-evidence.v1', protocol=write(root, 'protocol.json', self.f.f.p),
                    plan=write(root, 'plan.json', plan), runs=runs)

    @staticmethod
    def mutate_run(root, manifest, arm, mutate, seed=201):
        entry = next(item for item in manifest['runs'] if item['arm'] == arm and item['case_id'] == f's{seed}')
        path = entry['record']['path']
        record = json.loads((root / path).read_bytes())
        mutate(record)
        entry['record'] = write(root, path, record)

    def test_paired_per_modality_seed_intervals_and_quality_scope(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root)
            result = p2.evaluate(manifest, root)
            self.assertEqual(result['status'], 'measured', result['cases'][0]['arms'])
            self.assertEqual(result['expected_requests'], 240)
            self.assertEqual(result['totals']['reference']['video']['planned'], 60)
            interval = result['repetitions'][0]['intervals']['video']['mean_latency_ms_ratio']
            self.assertEqual(interval['seed_count'], 3)
            self.assertGreater(interval['ci95'][0], 1)
            self.assertEqual(result['cases'][0]['comparison']['numeric_equivalence'], 'paired_digest_agreement')
            self.assertEqual(result['quality']['status'], 'pending')
            self.assertEqual(result['allows'], [])
            self.assertEqual(result, p2.evaluate(manifest, root))

    def test_unstarted_second_policy_remains_forty_planned_requests(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root)
            records = self.records()
            raw = records['candidate']
            config = raw['config']
            config['phases'][0]['matrix']['/policy'] = ['block-cfg2', 'edf-cfg2']
            raw['status'] = 'failed'
            raw['phases'][0]['status'] = 'failed'
            raw['phases'][0]['cells'] = [dict(cell=0, status='failed', error='solver-step owner conflict')]
            for row in raw['requests'][30:]:
                row['status'] = 'interrupted'; row.pop('outputs')
            frozen = write(root, 'failed-config.json', config)
            failed = write(root, 'failed-run.json', raw)
            plan = json.loads((root / 'plan.json').read_bytes())
            case = plan['cases'][0]
            for arm, cell in [('reference', 1), ('candidate', 0)]:
                case[arm].update(config=frozen, cell=cell)
                entry = next(item for item in manifest['runs'] if item['arm'] == arm and item['case_id'] == 's201')
                entry['record'] = failed
            manifest['plan'] = write(root, 'plan.json', plan)
            result = p2.evaluate(manifest, root)
            case = result['cases'][0]
            self.assertEqual(result['expected_requests'], 240)
            self.assertEqual(case['arms']['reference']['status'], 'not_started')
            self.assertEqual(sum(c['not_started'] for c in case['arms']['reference']['by_kind'].values()), 40)
            self.assertEqual(sum(c['completed'] for c in case['arms']['candidate']['by_kind'].values()), 30)
            self.assertEqual(sum(c['interrupted'] for c in case['arms']['candidate']['by_kind'].values()), 10)
            self.assertIsNone(case['arms']['candidate']['elapsed_ms'])
            self.assertEqual(case['comparison']['status'], 'pending')
            self.assertIsNone(result['repetitions'][0]['intervals']['video']['completion_delta']['ci95'])

    def test_failed_cells_count_negative_completion_without_inventing_duration(self):
        def fail(records, seed):
            raw = records['candidate']; raw['status'] = 'failed'
            raw['phases'][0]['cells'][0] = dict(cell=0, status='failed', error='OOM')
            for kind in ('image', 'video'):
                row = next(row for row in raw['requests'] if row['kind'] == kind)
                row['status'] = 'failed'; row.pop('outputs')
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root, mutate=fail)
            result = p2.evaluate(manifest, root)
            self.assertEqual(result['status'], 'measured')
            for kind in ('image', 'video'):
                self.assertEqual(result['totals']['candidate'][kind]['failed'], 3)
                interval = result['repetitions'][0]['intervals'][kind]['completion_delta']
                self.assertLess(interval['ci95'][1], 0)
                self.assertIsNone(result['cases'][0]['arms']['candidate']['metrics'][kind]['ontime_rps'])
                self.assertIsNone(result['repetitions'][0]['intervals'][kind]['p95_latency_ms_ratio']['ci95'])

    def test_same_wrong_shape_weights_or_nonfinite_outputs_cannot_pass(self):
        mutations = [lambda row: row['outputs'][-1]['output']['shape'].__setitem__(-2, 17),
                     lambda row: row['outputs'][-1]['output'].__setitem__('finite', False),
                     lambda row: row['outputs'][-1]['generation'].__setitem__('model_manifest_sha256', 'e' * 64)]
        for mutate in mutations:
            with self.subTest(mutate=mutate), TemporaryDirectory() as directory:
                root = Path(directory); manifest = self.evidence(root)
                for arm in ('reference', 'candidate'):
                    self.mutate_run(root, manifest, arm, lambda record: mutate(record['requests'][0]))
                result = p2.evaluate(manifest, root)
                self.assertEqual(result['cases'][0]['arms']['reference']['status'], 'invalid')
                self.assertEqual(result['cases'][0]['comparison']['status'], 'pending')
                self.assertEqual(result['expected_requests'], 240)

    def test_measured_gpu_runtime_and_source_must_match(self):
        mutations = [lambda r: r['budget']['sessions'][0]['identity']['gpu_uuids'].reverse(),
                     lambda r: r['source'].__setitem__('core_sha256', 'f' * 64),
                     lambda r: r['source']['native_files'].__setitem__('src/ditserve/control/policy.cpp', 'f' * 64),
                     lambda r: [row['outputs'][-1]['generation']['runtime_identity'].__setitem__('python', 'changed') for row in r['requests']]]
        for mutate in mutations:
            with self.subTest(mutate=mutate), TemporaryDirectory() as directory:
                root = Path(directory); manifest = self.evidence(root)
                self.mutate_run(root, manifest, 'candidate', mutate)
                result = p2.evaluate(manifest, root)
                self.assertNotEqual(result['cases'][0]['comparison']['status'], 'measured')

    def test_missing_completed_rows_or_changed_actual_inputs_are_invalid(self):
        mutations = [lambda r: r['requests'].pop(),
                     lambda r: r['requests'][0]['spec'].__setitem__('seed', 999),
                     lambda r: r['phases'][0]['cells'][0].__setitem__('elapsed_ms', 1)]
        for mutate in mutations:
            with self.subTest(mutate=mutate), TemporaryDirectory() as directory:
                root = Path(directory); manifest = self.evidence(root)
                self.mutate_run(root, manifest, 'candidate', mutate)
                result = p2.evaluate(manifest, root)
                self.assertEqual(result['cases'][0]['arms']['candidate']['status'], 'invalid')

    def test_hash_mismatch_duplicate_seeds_and_duplicate_run_references_reject(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root)
            duplicate = copy.deepcopy(manifest); duplicate['runs'].append(duplicate['runs'][0])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                p2.evaluate(duplicate, root)
            plan = json.loads((root / 'plan.json').read_bytes()); plan['cases'][1]['seed'] = 201
            duplicate = copy.deepcopy(manifest); duplicate['plan'] = write(root, 'bad-plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'duplicate seed'):
                p2.evaluate(duplicate, root)
            (root / manifest['runs'][0]['record']['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                p2.evaluate(manifest, root)

    def test_check_mode_and_cross_form_digest_mismatch_do_not_claim_quality(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root)
            self.mutate_run(root, manifest, 'candidate', lambda r: r['requests'][0]['outputs'][-1]['output'].__setitem__('array_sha256', 'e' * 64))
            result = p2.evaluate(manifest, root)
            self.assertEqual(result['cases'][0]['comparison']['numeric_equivalence'], 'digest_mismatch')
            self.assertEqual(result['quality']['status'], 'pending')
            self.assertEqual(result['cases'][0]['comparison']['mismatched_digests'], 1)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            def check_mode(records, seed):
                for record in records.values():
                    phase = record['config']['phases'][0]
                    phase['output_mode'] = record['config']['benchmark']['output_mode'] = 'check'
                    for row in phase['matrix']['/workload'][0] + record['requests']:
                        row['output_mode'] = 'check'
                        if row.get('outputs'):
                            row['outputs'][-1]['output']['output_mode'] = 'check'
                    provenance = record['config']['benchmark']['trace']
                    provenance['sha256'] = provenance['manifest']['workload_sha256'] = p2.trace_hash(phase['matrix']['/workload'][0])
                    phase['matrix']['/trace_provenance'] = [provenance]
            manifest = self.evidence(root, mutate=check_mode)
            result = p2.evaluate(manifest, root)
            self.assertEqual(result['cases'][0]['comparison']['status'], 'measured')
            self.assertEqual(result['cases'][0]['comparison']['numeric_equivalence'], 'pending')
            self.assertEqual(result['quality']['status'], 'pending')

    def test_independent_seeds_cannot_change_runtime_or_drop_all_failed_modality_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root)
            for arm in ('reference', 'candidate'):
                self.mutate_run(root, manifest, arm, lambda r: r['source'].__setitem__('core_sha256', 'f' * 64), seed=203)
            with self.assertRaisesRegex(ValueError, 'independent seeds change'):
                p2.evaluate(manifest, root)
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root)
            def fail_video(record):
                record['phases'][0]['cells'][0] = dict(cell=0, status='failed', error='video load failed')
                for row in record['requests']:
                    if row['kind'] == 'video':
                        row['status'] = 'failed'; row.pop('outputs')
            self.mutate_run(root, manifest, 'candidate', fail_video)
            result = p2.evaluate(manifest, root)
            case = result['cases'][0]
            self.assertEqual(case['arms']['candidate']['by_kind']['video']['failed'], 20)
            self.assertEqual(case['comparison']['status'], 'pending')
            self.assertEqual(case['comparison']['runtime_identity_scope']['candidate_models_without_output'], ['wan21'])

    def test_relabelled_trace_seeds_and_reverse_order_omissions_are_not_repetitions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root)
            plan = json.loads((root / 'plan.json').read_bytes())
            for arm in ('reference', 'candidate'):
                first = json.loads((root / plan['cases'][0][arm]['config']['path']).read_bytes())
                second = json.loads((root / plan['cases'][1][arm]['config']['path']).read_bytes())
                second['phases'][0]['matrix']['/workload'] = first['phases'][0]['matrix']['/workload']
                trace = copy.deepcopy(first['benchmark']['trace']); trace['manifest']['seed'] = 202
                second['benchmark']['trace'] = trace
                second['phases'][0]['matrix']['/trace_provenance'] = [trace]
                plan['cases'][1][arm]['config'] = write(root, f'relabelled-{arm}.json', second)
            manifest['plan'] = write(root, 'plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'same frozen trace'):
                p2.evaluate(manifest, root)
        records = self.records()
        config = records['reference']['config']; config['phases'][0]['paired'] = True
        selector = dict(phase='cluster', attempt=1, cell=0, order='reverse')
        arm = p2.planned_arm(config, selector, self.f.f.p, 201)
        result = p2.audit_arm(records['reference'], arm, self.f.f.p)
        self.assertEqual(result['status'], 'not_started')
        self.assertEqual(sum(row['not_started'] for row in result['by_kind'].values()), 40)


class P2ContrastTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.P1QualificationTests.setUpClass()
        cls.f = fixtures.P1QualificationTests()

    def test_single_mechanism_changes_resolved_options_and_fixed_registry_wins(self):
        base = self.f.run_record()['config']
        base['phases'][0]['options']['forms'] = ['sp1', 'cfg2', 'sp2', 'cfg2sp2']
        original = copy.deepcopy(base)
        for name in contrasts.CONTRASTS:
            configs = contrasts.make_configs(base, name)
            arms = [p2.expanded(configs[arm]['phases'][0])[0] for arm in ('reference', 'candidate')]
            resolved = contrasts.validate_contrast(name, *arms)
            self.assertTrue(resolved['differing_fields'])
            self.assertNotIn('p1', configs['candidate']['benchmark'])
        self.assertEqual(base, original)
        resolved = contrasts.resolved_options('edf-cfg2', {'forms': ['sp1', 'sp8'], 'sp_transport': 'export'}, 8)
        self.assertEqual(resolved['forms'], ['cfg2'])

    def test_policy_option_masks_and_unequal_doorbell_checks_reject(self):
        base = self.f.run_record()['config']
        for name, option, value in [('pruning', 'prune_dominated', False),
                                    ('placement', 'image_placement', 'first'),
                                    ('minimum-pair', 'min_video_degree', 2)]:
            config = copy.deepcopy(base); config['phases'][0]['options'][option] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'identical'):
                contrasts.make_configs(config, name)
        configs = contrasts.make_configs(base, 'doorbell')
        arms = [p2.expanded(configs[arm]['phases'][0])[0] for arm in ('reference', 'candidate')]
        for arm in arms:
            self.assertEqual(arm['options']['quantum_blocks'], arm['options']['check_blocks'])
            arm['options']['check_blocks'] = 99
        with self.assertRaisesRegex(ValueError, 'block-check spacing'):
            contrasts.validate_contrast('doorbell', *arms)

    def test_unsupported_transport_small_xdit_and_inoperative_forms_reject(self):
        base = self.f.run_record()['config']
        for transport in ('pinned', 'resume'):
            config = copy.deepcopy(base); config['phases'][0]['options']['sp_transport'] = transport
            with self.subTest(transport=transport), self.assertRaisesRegex(ValueError, 'unpinned export'):
                contrasts.make_configs(config, 'block-fixed')
        config = copy.deepcopy(base); config['phases'][0]['gpus'] = config['phases'][0]['gpus'][:2]
        config['phases'][0]['options']['partition_image_gpus'] = 1
        with self.assertRaisesRegex(ValueError, 'no complete fixed CFG2'):
            contrasts.make_configs(config, 'system-xdit')
        for name in ('minimum-pair', 'pruning', 'remaining-work'):
            with self.subTest(contrast=name), self.assertRaises(ValueError):
                contrasts.make_configs(base, name)
        config = copy.deepcopy(base); config['phases'][0]['options']['forms'] = ['sp1', 'cfg2']
        for row in config['phases'][0]['profile']['classes'].values():
            if row['kind'] == 'video':
                row['min_degree'] = 2
        with self.assertRaisesRegex(ValueError, 'minimum degree'):
            contrasts.make_configs(config, 'minimum-pair')

    def test_p0_diagnostic_controls_and_population_are_preserved(self):
        config = self.f.run_record()['config']
        config['benchmark']['formal_protocol']['role'] = 'preemption'
        controls = dict(config['benchmark']['p1']['controls'], image_slo_scale=1.1, lazy_preemption=False)
        config['benchmark']['p0_diagnostic_controls'] = controls
        config['phases'][0]['options']['lazy_preemption'] = False
        generated = contrasts.make_configs(config, 'block-fixed')['candidate']
        self.assertEqual(generated['benchmark']['formal_protocol']['role'], 'preemption')
        self.assertEqual(generated['benchmark']['p2']['parent_controls'], controls)
        arm = p2.planned_arm(generated, dict(phase='cluster', attempt=1, cell=0, order='forward'), self.f.f.p, 201)
        scope = p2.measurement_scope(arm)
        self.assertEqual(scope['population'], 'development_diagnostic')
        self.assertEqual(scope['parent_controls']['image_slo_scale'], 1.1)
        self.assertFalse(scope['actual_lazy_preemption'])
        self.assertFalse(scope['formal_performance_qualified'])

    def test_cli_freezes_three_seed_plan_and_missing_evidence_template(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / 'input-protocol.json'; write(root, protocol.name, self.f.f.p)
            args = ['p2_contrasts.py', '--protocol', str(protocol), '--workload-id', 'wan21-contention',
                    '--contrast', 'block-fixed', '--output-dir', str(root / 'prepared')]
            for seed in (201, 202, 203):
                path = root / f'input-{seed}.json'; write(root, path.name, self.f.run_record(seed=seed)['config'])
                args += ['--config', str(path)]
            with patch.object(sys, 'argv', args):
                contrasts.main()
            evidence = json.loads((root / 'prepared/evidence.json').read_bytes())
            review = p2.evaluate(evidence, root / 'prepared')
            self.assertEqual(review['status'], 'incomplete')
            self.assertEqual(review['expected_requests'], 240)
            self.assertEqual(sum(kind['not_started'] for arm in review['totals'].values() for kind in arm.values()), 240)
            with patch.object(sys, 'argv', args), self.assertRaises(FileExistsError):
                contrasts.main()


if __name__ == '__main__':
    unittest.main()
