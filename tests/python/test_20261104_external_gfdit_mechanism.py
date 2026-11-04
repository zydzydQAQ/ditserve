"""CPU fixtures for original GF-DiT policy decisions; never measured performance."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import external_gfdit_mechanism as mechanism
import test_20261103_external_comparison as common_fixture

SOURCE = ROOT.parent / '.local/ditserve-kairos-rewrite-runtime/external-systems/20261103/GF-DiT'
MANIFEST = SOURCE.with_suffix('.source.json')
MANIFEST_SHA = 'e1b54cf09241ff70fc27e9d1666a44cbe38e00eaf1713b57cd5c9ade9c9db3f4'


def fixture(count=4):
    workload = common_fixture.ExternalComparisonTests().exported(count)
    profiles = []
    for request in workload['requests']:
        if any(p['shape_id'] == request['shape_id'] for p in profiles):
            continue
        spec = request['spec']
        # These explicit token/latency values are only CPU logic fixtures.
        profiles.append(dict(model=request['model'], shape_id=request['shape_id'],
            generation_profile=mechanism.generation_profile(request),
            model_manifest_sha256=request['model_manifest']['sha256'],
            latent_seq_len=123, text_seq_len=77,
            dit_step_do_true_cfg=[True] * spec['steps'],
            models=[dict(parallelism=dict(tp=1, sp=sp, ulysses=sp, ring=1, cfg=1),
                stages={stage:dict(coeffs=dict(a=0., b=0., c=cost)) for stage in mechanism.STAGES})
                for sp, cost in [(1, 1.), (2, .6), (4, .8)]]))
    costs = dict(schema=mechanism.COST_SCHEMA, workload_sha256=workload['sha256'],
        backend_profile_sha256='a'*64, provenance=dict(kind='fixture', description='CPU contract only; no measurements'),
        profiles=profiles)
    return workload, costs


def costs_for(costs, sizes):
    value = copy.deepcopy(costs)
    for profile in value['profiles']:
        profile['models'] = [m for m in profile['models'] if m['parallelism']['sp'] in sizes]
    return value


class GFMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_dir():
            raise unittest.SkipTest('requires exact official source; no replacement')
        cls.directory = TemporaryDirectory()
        cls.derived = mechanism.source.apply(SOURCE, MANIFEST, MANIFEST_SHA, Path(cls.directory.name) / 'control')
        cls.control = mechanism.source.load_control(cls.derived['root'], cls.derived['manifest_sha256'])

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == mechanism.source.NAMESPACE or name.startswith(mechanism.source.NAMESPACE + '.'):
                del sys.modules[name]
        cls.directory.cleanup()

    def driver(self, *, world=4, sizes=(1, 2, 4), count=4, costs=None, policy='edf_best_fit'):
        workload, default = fixture(count)
        return mechanism.DecisionDriver(self.control, workload, costs_for(costs or default, sizes),
            world_size=world, allowed_sp=sizes, policy=policy, allow_fixture=True)

    def finish(self, driver, row, now):
        driver.event(row['task_id'], 'launch_end', now)
        return driver.event(row['task_id'], 'exec_end', now)

    def test_original_policy_and_dag_sources_remain_import_only(self):
        driver = self.driver()
        native = importlib.import_module(mechanism.source.NAMESPACE + '.policies.edf_best_fit')
        self.assertIsInstance(driver.policy, native.EdfBestFitSchedulerPolicy)
        self.assertIs(type(driver.policy)._take_dispatchable_tasks, native.EdfBestFitSchedulerPolicy._take_dispatchable_tasks)
        self.assertIs(type(driver.policy)._best_fit_sp, native.EdfBestFitSchedulerPolicy._best_fit_sp)
        for target, record in self.derived['manifest']['imports_relocated'].items():
            raw = (SOURCE / record['upstream_path']).read_bytes()
            relocated, ast_sha = mechanism.source.relocate_source(raw)
            self.assertEqual(relocated, (Path(self.derived['root']) / target).read_bytes())
            self.assertEqual(ast_sha, record['algorithm_ast_sha256'])
        for rid, plan in driver.plans.items():
            spec = driver.requests[rid]['spec']
            intervals = sorted((t.step_range.start, t.step_range.end) for t in plan.tasks.values() if t.step_range)
            self.assertEqual(intervals[0][0], 0)
            self.assertEqual(intervals[-1][1], spec['steps'])
            self.assertTrue(all(a[1] == b[0] for a, b in zip(intervals, intervals[1:])))
            self.assertEqual(plan.metadata['frozen_request'], driver.requests[rid])

    def test_absolute_deadline_survives_late_submission_without_budget_renewal(self):
        driver = self.driver()
        row = driver.submit('r0', 140.)[0]
        meta = driver.policy.request_meta['r0']
        self.assertEqual((meta.arrival_ms, meta.deadline_ms), (0., 100.))
        self.assertEqual(row['remaining_budget_ms'], 0.)
        self.assertEqual(row['original_deadline_ms'], 100.)
        self.assertEqual(driver.summary()['offered_requests'], 4)
        self.assertEqual(driver.summary()['status_counts']['not_submitted'], 3)
        with self.assertRaisesRegex(ValueError, 'duplicate'): driver.submit('r0', 141.)
        with self.assertRaisesRegex(ValueError, 'backwards'): driver.poll(139.)
        early = self.driver()
        with self.assertRaisesRegex(ValueError, 'before frozen'): early.submit('r1', 1.)

    def test_every_shape_sp_stage_and_identity_is_preflighted_without_fallback(self):
        workload, costs = fixture()
        mutations = {
            'shape': lambda d: d['profiles'].pop(),
            'sp': lambda d: d['profiles'][0]['models'].pop(),
            'stage': lambda d: d['profiles'][0]['models'][0]['stages'].pop('dit_step_chunk_cfg'),
            'zero': lambda d: d['profiles'][0]['models'][0]['stages']['dit_step_chunk_cfg']['coeffs'].update(c=0.),
            'nan': lambda d: d['profiles'][0]['models'][0]['stages']['dit_step_chunk_cfg']['coeffs'].update(c=float('nan')),
            'model': lambda d: d['profiles'][0].update(model_manifest_sha256='b'*64),
            'shape_spec': lambda d: d['profiles'][0]['generation_profile'].update(frames=1),
            'cfg_schedule': lambda d: d['profiles'][0]['dit_step_do_true_cfg'].pop(),
            'workload': lambda d: d.update(workload_sha256='b'*64),
        }
        for name, mutate in mutations.items():
            value = copy.deepcopy(costs); mutate(value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                mechanism.DecisionDriver(self.control, workload, value, world_size=4, allowed_sp=(1,2,4), allow_fixture=True)
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            mechanism.DecisionDriver(self.control, workload, costs, world_size=4, allowed_sp=(1,2,4))
        with self.assertRaises(ValueError): self.driver(sizes=(2,))

    def test_external_observation_hashes_are_not_silently_promoted_to_measurements(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'; path.write_text('{"cpu_fixture":true}')
            provenance = dict(kind='external_observation', files=[dict(path=str(path), sha256=mechanism.common.file_sha(path.read_bytes()))])
            checked = mechanism.evidence_bindings(provenance)
            self.assertFalse(checked['measurement_claim'])
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA256'): mechanism.evidence_bindings(provenance)
        with self.assertRaises(ValueError): mechanism.evidence_bindings(dict(kind='external_observation',files=[]))

    def test_best_fit_uses_smallest_feasible_then_only_faster_surplus(self):
        driver = self.driver()
        text = driver.submit('r0', 0.)[0]
        # Original best-fit minimum is SP1; native pass 2 upgrades to faster SP2
        # but never SP4, whose supplied fixture estimate is slower.
        pending = driver.tasks['r0:dit_prepare:0']
        self.assertEqual(driver.policy._best_fit_sp('r0', pending, 4), 1)
        rows = self.finish(driver, text, 1.)
        self.assertEqual(rows[0]['kind'], 'dit_prepare')
        self.assertEqual(rows[0]['sp'], 2)
        self.assertLess(driver.policy._estimate_finish_ms('r0', 2), driver.policy._estimate_finish_ms('r0', 4))

    def test_same_token_count_different_model_shape_costs_cannot_alias(self):
        workload, costs = fixture()
        for profile in costs['profiles']:
            if profile['model'] == 'sd35':
                for model in profile['models']:
                    for stage in model['stages'].values(): stage['coeffs']['c'] = float(model['parallelism']['sp'])
        driver = mechanism.DecisionDriver(self.control, workload, costs, world_size=4,
            allowed_sp=(1,2,4), policy='edf_greedy', allow_fixture=True)
        # Caller-owned inputs may be reused; changing them cannot change the
        # driver's frozen request/cost identity or trigger native zero fallback.
        workload['requests'][2]['model'] = 'caller-mutation'
        costs['profiles'][2]['models'][0]['stages']['dit_prepare']['coeffs']['c'] = 9999.
        first = driver.submit('r2', 20.)[0]
        self.assertEqual(self.finish(driver, first, 21.)[0]['sp'], 1)
        video = self.driver(costs=costs, policy='edf_greedy')
        first = video.submit('r0', 0.)[0]
        self.assertEqual(self.finish(video, first, 1.)[0]['sp'], 2)
        with self.assertRaisesRegex(ValueError, 'context'):
            driver.costs.estimate(driver.p.ParallelSpec(), 123, 'dit_prepare')

    def test_edf_queued_order_and_all_offered_failure_denominator(self):
        driver = self.driver(world=2, sizes=(1,2))
        driver.hold('blocked', [0,1], True, 30.)
        for rid in ('r2','r1','r0'): self.assertEqual(driver.submit(rid, 30.), [])
        rows = driver.hold('blocked', [0,1], False, 31.)
        self.assertEqual([r['request_id'] for r in rows], ['r0','r1'])
        driver.event(rows[0]['task_id'], 'failed', 32.)
        summary = driver.summary()
        self.assertEqual(summary['offered_requests'], 4)
        self.assertEqual(summary['status_counts']['failed_event_observed'], 1)
        self.assertEqual(summary['status_counts']['not_submitted'], 1)
        self.assertEqual(summary['decoded_outputs_verified'], 0)
        self.assertFalse(summary['qualification_eligible'])
        self.assertNotIn('throughput_rps', summary)

    def test_original_hold_and_elastic_sp_one_two_one_without_gpu_migration_claim(self):
        driver = self.driver(world=2, sizes=(1,2))
        driver.hold('rank1-held', [1], True, 0.)
        text = driver.submit('r0', 0.)[0]
        prepare = self.finish(driver, text, 1.)[0]
        self.assertEqual(prepare['sp'], 1)
        driver.hold('rank1-held', [1], False, 2.)
        timestep = self.finish(driver, prepare, 3.)[0]
        self.assertEqual(timestep['sp'], 2)
        driver.hold('rank1-held-again', [1], True, 4.)
        chunk = self.finish(driver, timestep, 5.)[0]
        self.assertEqual(chunk['sp'], 1)
        self.assertTrue(chunk['group_changed'])
        self.assertFalse(chunk['migration_executed'])
        self.assertEqual(driver.policy.reshard_holds_by_rank, {1:1})
        with self.assertRaises(ValueError): driver.hold('rank1-held-again', [0], False, 6.)
        with self.assertRaises(ValueError): driver.hold('bad', [-1], True, 6.)

    def test_native_sticky_rank_preference_when_same_width_is_free(self):
        workload, costs = fixture()
        for profile in costs['profiles']:
            for model in profile['models']:
                for stage in model['stages'].values(): stage['coeffs']['c'] = float(model['parallelism']['sp'])
        driver = self.driver(world=2,sizes=(1,2),costs=costs)
        driver.hold('r0held', [0], True, 0.)
        text = driver.submit('r0', 0.)[0]
        self.assertEqual(text['ranks'], [1])
        driver.hold('r0held', [0], False, 1.)
        prepare = self.finish(driver, text, 2.)[0]
        self.assertEqual(prepare['ranks'], [1])

    def test_launch_publication_dependency_and_duplicate_events_fail_closed(self):
        driver = self.driver()
        text = driver.submit('r0', 0.)[0]
        with self.assertRaisesRegex(ValueError, 'without original'): driver.event(text['task_id'], 'exec_end', 1.)
        with self.assertRaisesRegex(ValueError, 'active dispatched'): driver.event('r0:finalize:0','launch_end',1.)
        self.assertEqual(driver.event(text['task_id'],'launch_end',2.), [])
        with self.assertRaisesRegex(ValueError, 'duplicate launch'): driver.event(text['task_id'],'launch_end',2.)
        driver.event(text['task_id'], 'failed', 3.)
        self.assertFalse(any(r['kind']=='dit_prepare' for r in driver.decisions))
        self.assertEqual(driver.status['r0'], 'failed_event_observed')

    def test_external_events_can_finish_entire_offered_dag_without_invented_output(self):
        for policy in mechanism.POLICIES:
            driver = self.driver(policy=policy)
            for rid in driver.requests: driver.submit(rid, 30.)
            now = 31.
            while driver.active:
                row = next(iter(driver.active.values()))
                self.finish(driver,row,now); now += 1.
            summary = driver.summary()
            self.assertEqual(summary['status_counts'], {'terminal_event_observed':4})
            self.assertEqual(len(summary['decisions']), sum(len(p.tasks) for p in driver.plans.values()))
            self.assertFalse(summary['model_execution_performed'])
            self.assertFalse(summary['native_global_scheduler_executed'])
            self.assertFalse(summary['native_gfc_executed'])
            self.assertEqual(summary['decoded_outputs_verified'], 0)
            self.assertEqual(summary['requests'][0]['frozen_request']['spec']['prompt'], fixture()[0]['requests'][0]['spec']['prompt'])

    def test_invalid_event_replay_retains_every_offered_request_and_cli_help(self):
        driver = self.driver()
        events = dict(schema=mechanism.EVENT_SCHEMA, workload_sha256=driver.workload['sha256'],
            provenance=dict(kind='fixture',description='CPU event fixtures'),
            events=[dict(action='submit',request_id='r0',at_ms=0.),dict(action='task',task_id='unknown',kind='exec_end',at_ms=1.)])
        result = mechanism.replay(driver,events,allow_fixture=True)
        self.assertFalse(result['replay_completed'])
        self.assertEqual(len(result['requests']),4)
        self.assertEqual(result['offered_requests'],4)
        self.assertFalse(result['qualification_eligible'])
        process = subprocess.run([sys.executable,'-B',str(ROOT/'benchmarks/external_gfdit_mechanism.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertIn('--allow-fixture-inputs',process.stdout)


if __name__ == '__main__':
    unittest.main()
