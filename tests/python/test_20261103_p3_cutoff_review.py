"""Plan-bound controller cutoffs; synthetic identities are not GPU evidence."""

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
import test_20261103_p3_stream as fixtures
import test_20261103_p2_benefit as p2_fixtures
from stage_gates import canonical_hash


class P3CutoffReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.P3StreamTest.setUpClass()
        cls.f = fixtures.P3StreamTest()

    def setUp(self):
        # The strict loader checks an explicit reviewed allowlist. This is a
        # deliberately synthetic source pin used only by these CPU fixtures.
        pins = patch.dict(review.CUTOFF_NATIVE_SOURCES,
                          {'src/ditserve/control/study.cpp': ('9' * 64,)})
        pins.start(); self.addCleanup(pins.stop)

    def prepared(self, enabled=True):
        base, plan, usage = self.f.inputs()
        if enabled:
            plan['execution'] = dict(drain_cutoff='cancel_and_censor', cleanup_allowance_seconds=30)
        configs, _ = p3.prepare(base, self.f.f.p, plan, self.f.f.pools, usage)
        return configs, plan

    def record(self, config, *, censored=True):
        record = self.f.record(config)
        record['config_sha256'] = canonical_hash(config)
        record['source']['native_files'] = {path: hashes[0] for path, hashes in review.CUTOFF_NATIVE_SOURCES.items()}
        phase = record['phases'][0]
        session = record['budget']['sessions'][0]
        session['cleanup_verified'] = True
        session['identity']['config_sha256'] = record['config_sha256']
        phase['charge'] = copy.deepcopy(session)
        phase['final_occupancy'] = dict(status='exclusive_observed', owned=[], foreign=[])
        control = config['phases'][0]['replay_cutoff']
        for position, cell in enumerate(phase['cells']):
            rows = [row for row in record['requests'] if row['cell'] == cell['cell']]
            horizon = control['after_ms'] if censored else cell['elapsed_ms']
            observation = f"{phase['id']}/1/{position}"
            for row in rows:
                row['observation_id'] = observation
                row['observation_ms'] = horizon
            if censored:
                row = next(row for row in reversed(rows) if row['kind'] == 'video')
                row['status'] = 'censored'
                row.pop('finish_ms'); row.pop('outputs')
            cell.update(status='censored' if censored else 'completed', elapsed_ms=horizon,
                        cleanup_verified=True, final_occupancy=copy.deepcopy(phase['final_occupancy']))
            cell['replay_cutoff'] = dict(control,
                config_sha256=record['config_sha256'], observation_id=observation, policy=cell['policy'],
                clock='controller_replay_monotonic_ms', physical_stop_bound=False,
                status='triggered' if censored else 'not_reached', observation_horizon_ms=horizon,
                observed_at_ms=horizon + 1, termination_finished_ms=horizon + 150,
                replay_origin_monotonic_seconds=1000 + position * 100,
                cleanup_verified=True,
                counts=dict(planned=len(rows), **{state: sum(row['status'] == state for row in rows)
                                                 for state in review.TERMINAL_STATES}))
        phase['status'] = record['status'] = 'censored' if censored else 'completed'
        return record

    def evidence(self, root, *, censored=True):
        configs, plan = self.prepared()
        write = p2_fixtures.write
        manifest = dict(schema='ditserve.p3-evidence.v1', plan=write(root, 'plan.json', plan),
                        protocol=write(root, 'protocol.json', self.f.f.p), configs=[], runs=[])
        for seed, config in configs.items():
            manifest['configs'].append(dict(seed=seed, config=write(root, f'config-{seed}.json', config)))
            manifest['runs'].append(dict(seed=seed, attempt=1,
                record=write(root, f'run-{seed}.json', self.record(config, censored=censored))))
        return manifest

    def test_cutoff_is_explicit_plan_bound_and_adds_only_cleanup_budget(self):
        before, old_plan = self.prepared(False)
        after, plan = self.prepared()
        for seed, config in after.items():
            old, phase = before[seed]['phases'][0], config['phases'][0]
            self.assertNotIn('replay_cutoff', old)
            self.assertEqual(phase['replay_cutoff'], dict(schema='ditserve.replay-cutoff.v1',
                                                        mode='cancel_and_censor', after_ms=35000))
            self.assertEqual(phase['matrix']['/workload'], old['matrix']['/workload'])
            self.assertEqual(phase['watchdog_seconds'] - old['watchdog_seconds'], 60)
            self.assertEqual(config['benchmark']['p3']['execution'], plan['execution'])
            self.assertEqual(config['benchmark']['p3']['plan_sha256'], canonical_hash(plan))
        base, plan, usage = self.f.inputs()
        for execution in (None, True, 'cancel_and_censor', {'drain_cutoff': 'watchdog'},
                          {'drain_cutoff': 'cancel_and_censor', 'cleanup_allowance_seconds': 0},
                          {'drain_cutoff': 'cancel_and_censor', 'cleanup_allowance_seconds': True},
                          {'drain_cutoff': 'cancel_and_censor', 'cleanup_allowance_seconds': 30, 'extra': 1}):
            with self.subTest(execution=execution), self.assertRaises(ValueError):
                p3.validate_structure(dict(plan, execution=execution), self.f.f.p)
        plan['execution'] = dict(drain_cutoff='cancel_and_censor', cleanup_allowance_seconds=30)
        for controls in (dict(mps={'server_pid': 42}), dict(actions=[{'op': 'advance'}]), dict(kind='online')):
            changed = copy.deepcopy(base); changed['phases'][0].update(controls)
            with self.subTest(controls=controls), self.assertRaisesRegex(ValueError, 'without scripted actions or shared MPS'):
                p3.prepare(changed, self.f.f.p, plan, self.f.f.pools, usage)

    def test_rehashed_phase_cannot_change_or_synthesize_cutoff(self):
        for enabled in (False, True):
            configs, plan = self.prepared(enabled)
            config = copy.deepcopy(configs[301])
            config['phases'][0]['replay_cutoff'] = dict(schema='ditserve.replay-cutoff.v1',
                after_ms=35001, mode='cancel_and_censor')
            with self.subTest(enabled=enabled), self.assertRaisesRegex(ValueError, 'cutoff'):
                p3.validate_structure(plan, self.f.f.p, config)
        configs, plan = self.prepared()
        config = copy.deepcopy(configs[301]); config['benchmark']['p3']['execution']['cleanup_allowance_seconds'] = 1
        with self.assertRaisesRegex(ValueError, 'execution controls'):
            p3.validate_structure(plan, self.f.f.p, config)
        config = copy.deepcopy(configs[301]); config['phases'][0]['watchdog_seconds'] -= 1
        with self.assertRaisesRegex(ValueError, 'cleanup budget'):
            p3.validate_structure(plan, self.f.f.p, config)

    def test_verified_censoring_retains_denominators_and_never_claims_capacity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root)
            result = review.evaluate(manifest, root)
            self.assertEqual(result['status'], 'reviewed', result['cases'])
            self.assertFalse(result['stable_capacity_claim']); self.assertEqual(result['allows'], [])
            self.assertEqual(result['runtime_cutoff_status'], 'verified')
            for case in result['cases']:
                self.assertEqual(case['status'], 'censored')
                self.assertTrue(case['actual_drain_cutoff_enforced'])
                self.assertTrue(case['runtime_cutoff_verified'])
                self.assertFalse(case['physical_stop_bound'])
                self.assertEqual(case['observed_horizon_ms'], 35000)
                self.assertEqual(case['cleanup_after_observation_ms'], 150)
                self.assertEqual(case['all_arrival_counts']['video']['censored'], 1)
                self.assertGreater(case['by_kind']['video']['censored'], 0)
                self.assertIsNone(case['metrics']['video']['latency_p95_ms'])
                self.assertLess(case['metrics']['video']['completion_by_drain'], 1)
                self.assertIsNotNone(case['queue']['video']['growth_requests_per_second'])

    def test_early_completion_does_not_claim_a_triggered_cutoff(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); result = review.evaluate(self.evidence(root, censored=False), root)
            self.assertEqual(result['status'], 'reviewed', result['cases'])
            for case in result['cases']:
                self.assertTrue(case['runtime_cutoff_verified'])
                self.assertFalse(case['actual_drain_cutoff_enforced'])
                self.assertEqual(case['status'], 'completed')

    def test_receipt_identity_cleanup_counts_and_late_outputs_fail_closed(self):
        def row(record):
            return next(row for row in record['requests'] if row['cell'] == 0 and row['status'] == 'completed')
        def cell(record):
            return record['phases'][0]['cells'][0]
        mutations = [
            lambda r: cell(r)['replay_cutoff'].update(config_sha256='0'*64),
            lambda r: cell(r)['replay_cutoff'].update(observation_id='cluster/1/99'),
            lambda r: cell(r)['replay_cutoff'].update(status='not_reached'),
            lambda r: cell(r)['replay_cutoff'].update(after_ms=35001),
            lambda r: cell(r)['replay_cutoff']['counts'].update(censored=0),
            lambda r: cell(r).update(cleanup_verified=False),
            lambda r: r['budget']['sessions'][0].update(cleanup_verified=False),
            lambda r: r['phases'][0]['charge'].update(cleanup_verified=False),
            lambda r: r['phases'][0]['charge'].update(session_id='another-session'),
            lambda r: r['phases'][0]['charge']['identity'].update(run_id='another-run'),
            lambda r: r['phases'][0].update(unverified_workers=[dict(cell=0, workers=[dict(worker='wan0', pid=42)])]),
            lambda r: cell(r)['final_occupancy'].update(owned=[42]),
            lambda r: cell(r)['final_occupancy'].update(foreign=[42]),
            lambda r: r['phases'][0]['final_occupancy'].update(status='unknown'),
            lambda r: row(r).update(finish_ms=35001),
            lambda r: row(r).pop('observation_ms'),
            lambda r: next(row for row in r['requests'] if row['cell'] == 0 and row['status'] == 'censored').update(finish_ms=34000),
            lambda r: row(r).update(observation_id='cluster/1/1'),
            lambda r: row(r)['spec'].update(prompt='changed after freezing'),
            lambda r: row(r)['outputs'][0]['generation'].update(model_manifest_sha256='f'*64),
            lambda r: r['source']['native_files'].update({'src/ditserve/control/study.cpp': '0'*64}),
            lambda r: r['requests'].remove(row(r)),
            lambda r: cell(r).pop('replay_cutoff'),
        ]
        configs, _ = self.prepared(); config = configs[301]
        arm = review.planned_arm(config, dict(phase=config['phases'][0]['id'], attempt=1, cell=0,
                                             order='forward'), self.f.f.p, 301)
        for mutate in mutations:
            record = self.record(config); mutate(record)
            before = copy.deepcopy(record)
            with self.subTest(mutation=mutate):
                result = review.audit_replay(record, arm, self.f.f.p)
                self.assertEqual(result['status'], 'invalid', result)
                self.assertEqual(record, before, 'review must not edit its evidence')
                self.assertEqual(sum(v['planned'] for v in result['by_kind'].values()), len(arm['workload']))

    def test_each_policy_needs_a_distinct_origin_after_verified_previous_cleanup(self):
        configs, _ = self.prepared(); config = configs[301]
        arm = review.planned_arm(config, dict(phase=config['phases'][0]['id'], attempt=1, cell=1,
                                             order='forward'), self.f.f.p, 301)
        for origin in (1000, 1035):
            record = self.record(config)
            record['phases'][0]['cells'][1]['replay_cutoff']['replay_origin_monotonic_seconds'] = origin
            with self.subTest(origin=origin):
                audited = review.audit_replay(record, arm, self.f.f.p)
                self.assertEqual(audited['status'], 'invalid', audited)
                self.assertIn('before previous cell cleanup', audited['reason'])
        record = self.record(config)
        record['phases'][0]['cells'][1]['replay_cutoff']['replay_origin_monotonic_seconds'] = 1035.15
        self.assertEqual(review.audit_replay(record, arm, self.f.f.p)['status'], 'censored')
        record['phases'][0]['cells'][0]['status'] = 'failed'
        self.assertEqual(review.audit_replay(record, arm, self.f.f.p)['status'], 'invalid')

    def test_real_failure_before_cutoff_remains_failure_with_its_observed_prefix(self):
        configs, plan = self.prepared(); config = configs[301]
        arm = review.planned_arm(config, dict(phase=config['phases'][0]['id'], attempt=1, cell=0,
                                             order='forward'), self.f.f.p, 301)
        record = self.record(config, censored=False)
        rows = [row for row in record['requests'] if row['cell'] == 0]
        horizon = 20000
        for row in rows:
            row['observation_ms'] = horizon
            if row['finish_ms'] > horizon:
                row['status'] = 'interrupted' if row['arrival_ms'] <= horizon else 'not_executed'
                for field in ('start_ms', 'finish_ms', 'outputs'):
                    row.pop(field, None)
        failed = next(row for row in rows if row['status'] == 'completed' and row['arrival_ms'] >= 10000)
        failed['status'] = 'failed'; failed.pop('outputs')
        cell = record['phases'][0]['cells'][0]
        cell.update(status='failed', elapsed_ms=horizon, error='model failure before drain')
        receipt = cell['replay_cutoff']
        receipt.update(observation_horizon_ms=horizon, observed_at_ms=horizon + 1,
                       termination_finished_ms=36000,
                       counts=dict(planned=len(rows), **{state: sum(row['status'] == state for row in rows)
                                                       for state in review.TERMINAL_STATES}))
        audited = review.audit_replay(record, arm, self.f.f.p)
        self.assertEqual(audited['status'], 'failed', audited)
        window = review.window_summary(arm['workload'], rows, audited, plan['windows'])
        self.assertFalse(window['actual_drain_cutoff_enforced'])
        self.assertEqual(sum(values['failed'] for values in window['by_kind'].values()), 1)
        self.assertEqual(sum(values['censored'] for values in window['by_kind'].values()), 0)
        self.assertEqual(window['observed_horizon_ms'], horizon)
        # Relabelling this genuine error as an ordinary cutoff is not allowed.
        cell['status'] = 'censored'
        receipt.update(status='triggered', observation_horizon_ms=35000, observed_at_ms=35001)
        for row in rows:
            row['observation_ms'] = 35000
        self.assertEqual(review.audit_replay(record, arm, self.f.f.p)['status'], 'invalid')

    def test_failed_start_and_later_unexecuted_policy_remain_visible(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.evidence(root)
            reference = manifest['runs'][0]['record']; record = json.loads((root/reference['path']).read_text())
            record['status'] = record['phases'][0]['status'] = 'failed'
            for index, cell in enumerate(record['phases'][0]['cells']):
                cell['status'] = 'failed' if index == 0 else 'not_executed'
                cell['error'] = 'warmup failed' if index == 0 else 'previous cell failed'
                cell.pop('replay_cutoff'); cell.pop('elapsed_ms')
            for row in record['requests']:
                row['status'] = 'not_executed'
                for field in ('start_ms', 'finish_ms', 'observation_ms', 'outputs'):
                    row.pop(field, None)
            manifest['runs'][0]['record'] = p2_fixtures.write(root, reference['path'], record)
            result = review.evaluate(manifest, root)
            first, second = result['cases'][:2]
            self.assertEqual(second['status'], 'not_executed', second)
            for case in (first, second):
                self.assertFalse(case['actual_drain_cutoff_enforced'])
                self.assertEqual(case['metrics'], {})
                self.assertEqual(sum(v['not_executed'] for v in case['all_arrival_counts'].values()),
                                 sum(v['planned'] for v in case['all_arrival_counts'].values()))
            arm = review.planned_arm(record['config'], dict(phase=record['phases'][0]['id'], attempt=1,
                cell=1, order='forward'), self.f.f.p, manifest['runs'][0]['seed'])
            next(row for row in record['requests'] if row['cell'] == 1)['start_ms'] = 0
            self.assertEqual(review.audit_replay(record, arm, self.f.f.p)['status'], 'invalid')


if __name__ == '__main__':
    unittest.main()
