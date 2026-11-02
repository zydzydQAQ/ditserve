"""Opt-in bounded decisions, real native mock execution; no GPU qualification."""
import json
import unittest

import test_study as fixtures
from ditserve.runtime.client import CoreBridge, CoreError


@unittest.skipUnless(CoreBridge().available(), 'build the candidate C++ core first')
class ControllerDecisionsTest(unittest.TestCase):
    setUpClass = classmethod(fixtures.NativeStudyTest.setUpClass.__func__)
    setUp = fixtures.NativeStudyTest.setUp
    config = fixtures.NativeStudyTest.config
    cluster_phase = fixtures.NativeStudyTest.cluster_phase
    cluster_workers = fixtures.NativeStudyTest.cluster_workers
    run_study = fixtures.NativeStudyTest.run_study
    assert_completed = fixtures.NativeStudyTest.assert_completed

    def fixture(self, telemetry=None):
        phase = self.cluster_phase('block-cfg2', gpus=2, text_prefetch=0,
                                   output_mode='digest', options=dict(quantum_blocks=4))
        phase['workload'] = phase['workload'][:2]
        phase['profile']['quantum_overhead_ms'] = 0.5
        phase['profile']['classes']['video'].update(step_compute_ms=40, finish_ms=0)
        for row in phase['workload']:
            row.update(slo_ms=1000000)
            row['spec'].update(height=32, width=48, num_frames=3)
        workers = self.cluster_workers(2)
        for worker in workers.values():
            worker['config']['options']['costs'] = dict(step_ms=40, finish_ms=0)
        if telemetry is not None:
            phase['controller_decisions'] = telemetry
        return phase, workers

    def run_fixture(self, telemetry=None, name='run'):
        phase, workers = self.fixture(telemetry)
        return self.run_study(self.config([phase], workers=workers), name=name)

    def descriptors(self, run):
        return [(r['request_id'], r['status'], r['forms'], r['degrees'],
                 [x['output']['sha256'] for x in r['outputs']],
                 [(e['step_before'], e['block_before'], e['group_after'], e['quantum'])
                  for e in r['controller_timing']]) for r in run['requests']]

    def assert_accounting(self, log):
        self.assertLessEqual(len(log['events']), log['limits']['max_events'])
        self.assertLessEqual(len(json.dumps(log, separators=(',', ':'), ensure_ascii=False).encode()), log['limits']['max_bytes'])
        self.assertEqual(log['plan_calls'], len(log['events']) + log['dropped_plan_calls'] + log['deduplicated_idle_calls'] + log['failed_plan_calls'] + log['filtered_plan_calls'])
        for event in log['events']:
            self.assertLessEqual(len(event['rows']), log['limits']['max_rows_per_event'])
            self.assertEqual(event['candidate_count'], len(event['rows']) + event['dropped_rows'] + event['filtered_rows'])

    def test_enabled_same_dispatch_and_digest_with_actual_decision_links(self):
        baseline = self.run_fixture(name='disabled')
        explicit = self.run_fixture(dict(enabled=False), name='explicit-disabled')
        observed = self.run_fixture(dict(enabled=True), name='enabled')
        for run in (baseline, explicit, observed):
            self.assert_completed(run)
        self.assertNotIn('controller_decisions', baseline)
        self.assertNotIn('controller_decisions', explicit)
        self.assertTrue(all('decision_id' not in e for r in baseline['requests'] for e in r['controller_timing']))
        self.assertEqual(self.descriptors(baseline), self.descriptors(explicit))
        self.assertEqual(self.descriptors(baseline), self.descriptors(observed))
        log, = observed['controller_decisions']
        self.assert_accounting(log)
        self.assertFalse(log['truncated'])
        self.assertFalse(log['exhaustive'])
        self.assertEqual(log['config_sha256'], observed['config_sha256'])
        events = {e['decision_id']: e for e in log['events']}
        for row in observed['requests']:
            for dispatch in row['controller_timing']:
                decision = events[dispatch['decision_id']]
                selected = next(a for a in decision['assignments'] if a['request_id'] == row['request_id'])
                self.assertEqual(selected['group'], dispatch['group_after'])
                self.assertEqual(selected['quantum'], dispatch['quantum'])
                self.assertLessEqual(decision['now_ms'], dispatch['selected_ms'])
        for decision in log['events']:
            for row in decision['rows']:
                self.assertEqual(row['generation_shape']['height'], 32)
                self.assertEqual(row['estimate_group'], None if row['blocks_done'] == 0 else row['resident_group'])
                self.assertAlmostEqual(row['estimated_remaining_ms'], row['remaining_without_dispatch_ms'] + row['modeled_dispatch_quanta'] * row['quantum_overhead_ms'])
                self.assertEqual(row['estimated_feasible'], decision['now_ms'] + row['estimated_remaining_ms'] <= row['deadline_ms'])

    def test_idle_dedup_retains_controller_observed_owner_and_rank(self):
        phase, workers = self.fixture(dict(enabled=True, focus=['i0']))
        video = phase['workload'][0]
        image = dict(request_id='i0', kind='image', worker='i0', arrival_ms=0,
                     slo_ms=1000, spec=dict(steps=1), **{'class': 'image'})
        phase['workload'] = [image, video]
        workers['i0']['config']['options']['costs'] = dict(step_ms=0, finish_ms=200)
        workers['i1']['config']['options']['costs'] = dict(step_ms=0, finish_ms=200)
        run = self.run_study(self.config([phase], workers=workers), name='idle')
        self.assert_completed(run)
        log, = run['controller_decisions']
        self.assert_accounting(log)
        self.assertGreater(log['deduplicated_idle_calls'], 0)
        idle = [e for e in log['events'] if not e['assignments'] and e['running_tasks']]
        self.assertTrue(idle)
        self.assertTrue(any(t['request_id'] == 'i0' for e in idle for t in e['running_tasks']))
        self.assertTrue(all(any(e['busy']) for e in idle))
        self.assertTrue(all(not row['selected'] for e in idle for row in e['rows']))

    def test_count_and_byte_caps_preserve_request_denominator(self):
        for name, config in [('events', dict(enabled=True, max_events=1)),
                             ('bytes', dict(enabled=True, max_bytes=4096))]:
            with self.subTest(name=name):
                run = self.run_fixture(config, name=name)
                self.assert_completed(run)
                self.assertEqual(len(run['requests']), 2)
                self.assertTrue(all(r['status'] == 'completed' for r in run['requests']))
                log, = run['controller_decisions']
                self.assert_accounting(log)
                self.assertTrue(log['truncated'])
                self.assertGreater(log['dropped_plan_calls'], 0)
                links = {e['decision_id'] for e in log['events']}
                for row in run['requests']:
                    for timing in row['controller_timing']:
                        if 'decision_id' in timing:
                            self.assertIn(timing['decision_id'], links)

    def test_focus_has_explicit_filtered_denominator(self):
        run = self.run_fixture(dict(enabled=True, focus=['v1']), name='focused')
        self.assert_completed(run)
        log, = run['controller_decisions']
        self.assert_accounting(log)
        self.assertGreater(log['filtered_rows'], 0)
        self.assertEqual({r['request_id'] for e in log['events'] for r in e['rows']}, {'v1'})

    def test_oversized_ids_and_shapes_drop_diagnostics_without_losing_requests(self):
        for name, change in [('id', 'request_id'), ('shape', 'shape')]:
            phase, workers = self.fixture(dict(enabled=True, focus=['v0'] if change == 'shape' else []))
            phase['workload'] = phase['workload'][:1]
            if change == 'request_id':
                phase['workload'][0]['request_id'] = 'v' * 70000
            else:
                # The mock does not interpret this shape field; exercise the
                # diagnostic input boundary without allocating a huge array.
                phase['workload'][0]['spec']['height'] = 'h' * 70000
            run = self.run_study(self.config([phase], workers=workers), name='oversized-' + name)
            self.assert_completed(run)
            self.assertEqual(len(run['requests']), 1)
            log, = run['controller_decisions']
            self.assert_accounting(log)
            self.assertTrue(log['truncated'])
            self.assertGreater(log['dropped_rows'], 0)
            if change == 'request_id':
                self.assertGreater(log['dropped_plan_calls'], 0)
            else:
                self.assertTrue(log['events'])
                self.assertTrue(all(not e['rows'] for e in log['events']))

    def test_late_focus_survives_more_unrelated_dispatches_than_event_budget(self):
        phase, workers = self.fixture(dict(enabled=True, focus=['v1'], max_events=2))
        phase['workload'][1]['arrival_ms'] = 800
        run = self.run_study(self.config([phase], workers=workers), name='late-focus')
        self.assert_completed(run)
        earlier = next(r for r in run['requests'] if r['request_id'] == 'v0')
        self.assertLess(earlier['finish_ms'], 800)
        self.assertGreater(len(earlier['controller_timing']), 2)
        self.assertTrue(all('decision_id' not in e for e in earlier['controller_timing']))
        log, = run['controller_decisions']
        self.assert_accounting(log)
        self.assertGreaterEqual(log['filtered_plan_calls'], len(earlier['controller_timing']))
        self.assertEqual(len(log['events']), 2)
        self.assertTrue(all(e['now_ms'] >= 800 for e in log['events']))
        self.assertTrue(all(any(r['request_id'] == 'v1' for r in e['rows']) for e in log['events']))

    def test_invalid_limits_rejected_before_worker_or_budget_start(self):
        invalid = [dict(enabled='yes'), dict(enabled=True, max_events=257), dict(max_events=0),
                   dict(max_rows=65), dict(max_bytes=4194305), dict(max_bytes=4095),
                   dict(focus=['v'] * 2), dict(focus=['a', 'b', 'c', 'd', 'e']), dict(unknown=True)]
        for i, diagnostic in enumerate(invalid):
            with self.subTest(diagnostic=diagnostic):
                phase, workers = self.fixture(diagnostic)
                with self.assertRaises(CoreError):
                    self.run_study(self.config([phase], workers=workers), name=f'invalid-{i}')
                self.assertFalse((self.root / f'invalid-{i}').exists())

    def test_failed_dispatch_retains_decision_and_requests(self):
        phase, workers = self.fixture(dict(enabled=True))
        workers['v0']['config']['options']['fail_ops'] = ['advance']
        run = self.run_study(self.config([phase], workers=workers), name='failed')
        self.assertEqual(run['status'], 'failed')
        log, = run['controller_decisions']
        self.assert_accounting(log)
        ids = {e['decision_id'] for e in log['events']}
        self.assertEqual(len(run['requests']), 2)
        failed = [t for r in run['requests'] for t in r.get('controller_timing', []) if t.get('controller_status') == 'aborted']
        self.assertTrue(failed)
        self.assertTrue(all(t['decision_id'] in ids for t in failed))


if __name__ == '__main__':
    unittest.main()
