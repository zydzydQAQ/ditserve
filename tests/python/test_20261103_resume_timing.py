"""Controller critical-path telemetry: concurrent RPCs, stop/resume and failures."""

import copy
import unittest

import test_study as fixtures
from ditserve.runtime.client import CoreBridge


@unittest.skipUnless(CoreBridge().available(), 'build the C++ core first')
class ResumeTimingTest(unittest.TestCase):
    setUpClass = classmethod(fixtures.NativeStudyTest.setUpClass.__func__)
    setUp = fixtures.NativeStudyTest.setUp
    config = fixtures.NativeStudyTest.config
    cluster_phase = fixtures.NativeStudyTest.cluster_phase
    cluster_workers = fixtures.NativeStudyTest.cluster_workers
    run_study = fixtures.NativeStudyTest.run_study
    assert_completed = fixtures.NativeStudyTest.assert_completed

    def assert_task(self, event):
        self.assertEqual(event['schema'], 'ditserve.controller-timing.v1')
        self.assertEqual(event['clock'], 'controller_replay_monotonic_ms')
        self.assertLessEqual(event['selected_ms'], event['start_ms'])
        # Scheduling retains its original post-launch clock anchor, while
        # observation covers launch itself. Async begin may fall on either
        # side of the scheduling anchor, so do not impose an ordering there.
        self.assertLessEqual(event['start_ms'], event['scheduling_start_ms'])
        self.assertLessEqual(event['scheduling_start_ms'], event['observed_ms'])
        self.assertLessEqual(event['start_ms'], event['async_begin_ms'])
        self.assertLessEqual(event['async_begin_ms'], event['all_done_ms'])
        self.assertLessEqual(event['all_done_ms'], event['observed_ms'])
        for stage in event['stages']:
            self.assertLessEqual(event['async_begin_ms'], stage['start_ms'])
            self.assertLessEqual(stage['start_ms'], stage['finish_ms'])
            self.assertLessEqual(stage['finish_ms'], event['all_done_ms'])
            self.assertAlmostEqual(stage['elapsed_ms'], stage['finish_ms'] - stage['start_ms'])

    def transfer_fixture(self):
        phase = self.cluster_phase('block-sp4', text_prefetch=0, options=dict(
            sp_transport='export', transfer_dir=str(self.root/'transfer')))
        phase['workload'] = [phase['workload'][0]]
        phase['workload'][0]['spec'].update(steps=1, blocks_per_step=1)
        phase['profile']['classes']['video'].update(steps=1, blocks_per_step=1, step_compute_ms=20)
        workers = self.cluster_workers()
        for index, delay in enumerate((20, 80, 160, 240)):
            workers[f'v{index}']['config']['options']['delay_ms'] = delay
            workers[f'v{index}']['config']['options']['costs'] = dict(step_ms=20, finish_ms=0)
        return phase, workers

    def test_import_fanout_measures_concurrent_wallclock_and_orders_advance(self):
        phase, workers = self.transfer_fixture()
        value = self.run_study(self.config([phase], workers=workers), name='timing-parallel')
        self.assert_completed(value)
        cell = value['phases'][0]['cells'][0]
        event = next(item for item in value['requests'][0]['controller_timing'] if item['transition'] == 'initial_distribution')
        self.assert_task(event)
        self.assertEqual(event['group_before'], [])
        self.assertEqual(event['group_after'], [0, 1, 2, 3])
        self.assertEqual(event['transfer_targets'], 3)
        stages = event['stages']
        prepare, export = [next(item for item in stages if item['stage'] == label) for label in ('prepare', 'export')]
        imports = [item for item in stages if item['stage'] == 'import']
        ready = next(item for item in stages if item['stage'] == 'imports_all_ready')
        advances = [item for item in stages if item['stage'] == 'advance']
        done = next(item for item in stages if item['stage'] == 'advances_all_done')
        self.assertEqual((len(imports), len(advances)), (3, 4))
        self.assertLessEqual(prepare['finish_ms'], export['start_ms'])
        self.assertLessEqual(export['finish_ms'], ready['start_ms'])
        for rank in imports:
            self.assertLessEqual(ready['start_ms'], rank['start_ms'])
            self.assertLessEqual(rank['finish_ms'], ready['finish_ms'])
        self.assertTrue(any(a['start_ms'] < b['finish_ms'] and b['start_ms'] < a['finish_ms']
                            for i, a in enumerate(imports) for b in imports[i+1:]))
        self.assertGreaterEqual(ready['elapsed_ms'], max(item['elapsed_ms'] for item in imports))
        # RPCs are overlapping intervals, so their sum is explicitly not the fanout latency.
        self.assertLess(ready['elapsed_ms'], sum(item['elapsed_ms'] for item in imports))
        for rank in advances:
            self.assertLessEqual(ready['finish_ms'], rank['start_ms'])
            self.assertLessEqual(rank['finish_ms'], done['finish_ms'])
            self.assertGreaterEqual(rank['worker_details']['worker_operation_ms'], 0)
        planning = cell['controller_planning']
        self.assertGreater(planning['calls'], 0)
        self.assertEqual(planning['failed'], 0)
        self.assertGreaterEqual(planning['total_ms'], planning['max_ms'])
        self.assertEqual(value['controller_planning'][0]['total_ms'], planning['total_ms'])

    def test_import_failure_preserves_export_and_every_joined_rank_without_fake_ready(self):
        phase, workers = self.transfer_fixture()
        workers['v1']['config']['options']['fail_ops'] = ['block_import']
        value = self.run_study(self.config([phase], workers=workers), name='timing-failed')
        self.assertEqual(value['status'], 'failed')
        event = value['requests'][0]['controller_timing'][0]
        self.assert_task(event)
        self.assertEqual((event['status'], event['controller_status']), ('failed', 'aborted'))
        stages = event['stages']
        self.assertEqual(next(item for item in stages if item['stage'] == 'export')['status'], 'completed')
        imports = [item for item in stages if item['stage'] == 'import']
        self.assertEqual({item['worker']: item['status'] for item in imports},
                         {'v1': 'failed', 'v2': 'completed', 'v3': 'completed'})
        self.assertEqual(next(item for item in stages if item['stage'] == 'imports_all_ready')['status'], 'failed')
        self.assertFalse(any(item['stage'] == 'advance' for item in stages))
        self.assertNotIn('step_after', event)
        self.assertEqual(len(value['controller_planning']), 1)

    def test_stop_links_next_resident_resume_and_preserves_numerical_output(self):
        phase = self.cluster_phase('block-cfg2', gpus=2, text_prefetch=0, output_mode='digest', options=dict(
            sp_transport='export', doorbell=True, lazy_preemption=False,
            transfer_dir=str(self.root/'stop')))
        phase['workload'] = [phase['workload'][0], phase['workload'][2]]
        phase['workload'][1].update(arrival_ms=75, slo_ms=80)
        workers = self.cluster_workers(2)
        value = self.run_study(self.config([phase], workers=workers), name='timing-stop')
        self.assert_completed(value)
        video = next(row for row in value['requests'] if row['kind'] == 'video')
        image = next(row for row in value['requests'] if row['kind'] == 'image')
        stops = video['preemption_stop_samples']
        self.assertTrue(stops)
        events = {item['event_id']: item for item in video['controller_timing']}
        for stop in stops:
            self.assertAlmostEqual(stop['stop_ms'], stop['observed_ms'] - stop['ring_ms'])
            owner = events[stop['task_event_id']]
            self.assertLessEqual(owner['all_done_ms'], stop['observed_ms'])
            resume = next(item for item in events.values() if item.get('resume_from_stop_event_id') == stop['event_id'])
            self.assert_task(resume)
            self.assertEqual(resume['transition'], 'resident_resume')
            self.assertEqual(resume['group_before'], resume['group_after'])
            self.assertEqual((resume['step_before'], resume['block_before']), (owner['step_after'], owner['block_after']))
            self.assertFalse(any(item['stage'] in ('prepare', 'export', 'import') for item in resume['stages']))
            self.assertLessEqual(stop['observed_ms'], image['start_ms'])
            self.assertLessEqual(image['finish_ms'], resume['start_ms'])
        for sample in video['advance_samples']:
            self.assertEqual(sample['start_ms'], events[sample['event_id']]['start_ms'])
        reference = copy.deepcopy(phase)
        reference.update(policy='edf-cfg2', options=dict(sp_transport='export', transfer_dir=str(self.root/'reference')))
        baseline = self.run_study(self.config([reference], workers=workers), name='timing-output-reference')
        self.assert_completed(baseline)
        expected = next(row for row in baseline['requests'] if row['kind'] == 'video')
        self.assertEqual(video['outputs'][-1]['output']['sha256'], expected['outputs'][-1]['output']['sha256'])

    def test_regroup_is_distinct_from_first_distribution_and_tracks_positions(self):
        phase = self.cluster_phase('step_elastic', text_prefetch=0, options=dict(
            sp_transport='export', forms=['sp2', 'sp4'], transfer_dir=str(self.root/'regroup')))
        phase['workload'] = [phase['workload'][0], phase['workload'][2]]
        phase['workload'][1].update(arrival_ms=20, slo_ms=80)
        phase['profile']['classes']['image']['service_ms'] = 300
        workers = self.cluster_workers()
        for name, worker in workers.items():
            if name.startswith('i'):
                worker['config']['options']['costs']['finish_ms'] = 300
        value = self.run_study(self.config([phase], workers=workers), name='timing-regroup')
        self.assert_completed(value)
        row = next(row for row in value['requests'] if row['kind'] == 'video')
        regroups = [event for event in row['controller_timing'] if event['transition'] == 'regroup']
        self.assertTrue(regroups)
        for event in regroups:
            self.assert_task(event)
            self.assertFalse(event['first'])
            self.assertGreater(event['step_before'], 0)
            self.assertEqual(event['block_before'], 0)
            self.assertNotEqual(event['group_before'], event['group_after'])
            self.assertGreater(event['epoch_after'], event['epoch_before'])

    def test_tiered_restore_has_its_own_rpc_and_links_the_following_task(self):
        phase = self.cluster_phase('heterodit_exact', gpus=1,
            options=dict(sp_transport='pinned', tiered_memory=True, quantum_ms=100))
        phase['workload'] = [phase['workload'][0], phase['workload'][2]]
        phase['workload'][1].update(arrival_ms=200, slo_ms=1000)
        phase['warmup'] = [dict(worker=name, spec=dict(prompt='warmup', steps=1)) for name in ('v0', 'i0')]
        value = self.run_study(self.config([phase], workers=self.cluster_workers(1)), name='timing-tiered')
        self.assert_completed(value)
        row = next(row for row in value['requests'] if row['kind'] == 'video')
        restored = next(event for event in row['controller_timing'] if event['transition'] == 'tiered_restore')
        resumed = next(event for event in row['controller_timing'] if event.get('restore_event_id') == restored['event_id'])
        self.assertEqual(restored['stages'][0]['stage'], 'tiered_prefetch')
        self.assertLessEqual(restored['stages'][0]['finish_ms'], restored['observed_ms'])
        self.assertLessEqual(restored['observed_ms'], resumed['start_ms'])
        self.assert_task(resumed)


if __name__ == '__main__':
    unittest.main()
