"""Controller fanout is a critical path, not the sum of rank RPC durations."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from p1_costs import summarize
from stage_gates import canonical_hash


def stage(name, start, end, worker=None, status='completed'):
    row = dict(stage=name, start_ms=start, finish_ms=end, elapsed_ms=end-start,
               status=status, request_id='v')
    if worker:
        row['worker'] = worker
    return row


def event(eid='task1', **changes):
    value = dict(schema='ditserve.controller-timing.v1', clock='controller_replay_monotonic_ms',
        event_id=eid, phase='c', attempt=1, cell=0, order='forward', request_id='v',
        selected_ms=0, start_ms=1, async_begin_ms=2, all_done_ms=25, observed_ms=27,
        status='completed', controller_status='completed', transition='regroup',
        group_before=[0], group_after=[1, 2], form_before='sp1', form_after='sp2',
        stages=[stage('export', 3, 5, 'w0'), stage('import', 5, 15, 'w1'),
                stage('import', 5, 20, 'w2'), stage('imports_all_ready', 5, 20),
                stage('advance', 20, 25, 'w1'), stage('advance', 20, 25, 'w2'),
                stage('advances_all_done', 20, 25)])
    value.update(changes)
    return value


def record(events=None):
    return dict(schema='ditserve.run.v1', run_id='fixture', status='completed',
        source=dict(core_sha256='c'*64, files={'worker.py': 'a'*64}), config=dict(phases=[]),
        requests=[dict(phase='c', attempt=1, cell=0, order='forward', request_id='v',
                       status='completed', controller_timing=[event()] if events is None else events)])


class ControllerCostReviewTest(unittest.TestCase):
    def test_imports_use_all_ready_wallclock_not_rank_sum(self):
        run = record(); run['requests'][0]['controller_timing'][0]['pair_request_id'] = 'v2'
        result = summarize(run)
        group = result['cells'][0]['transitions']['regroup']
        self.assertEqual(group['controller_wall_ms']['export_to_targets_ready_ms']['p50'], 17)
        self.assertEqual(group['controller_wall_ms']['imports_all_ready_ms']['p50'], 15)
        self.assertEqual(group['per_rank_rpc_ms']['import_ms']['count'], 2)
        self.assertEqual(group['controller_wall_ms']['task_to_progress_including_compute_ms']['p50'], 24)
        self.assertEqual(result['shared_pair_events'], 1)

    def test_failed_import_retains_export_without_ready_or_progress_credit(self):
        run = record([event(status='failed', controller_status='aborted', all_done_ms=21,
            stages=[stage('export', 3, 5, 'w0'), stage('import', 5, 15, 'w1'),
                    stage('import', 5, 20, 'w2', 'failed'), stage('imports_all_ready', 5, 20, status='failed')])])
        run['status'] = 'failed'; run['requests'][0]['status'] = 'interrupted'
        group = summarize(run)['cells'][0]['transitions']['regroup']
        self.assertEqual(group['counts']['stage_import_failed'], 1)
        self.assertEqual(group['per_rank_rpc_ms']['export_ms']['p50'], 2)
        self.assertNotIn('export_to_targets_ready_ms', group['controller_wall_ms'])
        self.assertNotIn('task_to_progress_including_compute_ms', group['controller_wall_ms'])

    def test_stop_wait_includes_intervening_work_and_requires_same_request_link(self):
        first = event()
        second = event('task2', transition='resident_resume', selected_ms=50, start_ms=51,
            async_begin_ms=52, all_done_ms=60, observed_ms=62, stages=[], resume_from_stop_event_id='task1/stop')
        run = record([first, second])
        run['requests'][0]['preemption_stop_samples'] = [dict(event_id='task1/stop', task_event_id='task1',
                                                            ring_ms=20, observed_ms=27, stop_ms=7)]
        group = summarize(run)['cells'][0]['transitions']['resident_resume']['controller_wall_ms']
        self.assertEqual(group['stopped_to_next_dispatch_ms']['p50'], 24)
        self.assertEqual(group['stopped_to_next_progress_including_other_work_ms']['p50'], 33)
        self.assertNotIn('restore_ms', group)
        second['resume_from_stop_event_id'] = 'missing/stop'
        with self.assertRaisesRegex(ValueError, 'same-request stop'):
            summarize(run)

    def test_legacy_missing_costs_and_non_task_tiered_rpc(self):
        self.assertEqual(summarize(record([]))['measurement_status'], 'not_measured')
        empty_native = record([]); empty_native['source']['native_files'] = {}
        self.assertIsNone(summarize(empty_native)['source']['native_sources_sha256'])
        tiered = event('restore', transition='tiered_restore', stages=[stage('tiered_prefetch', 5, 15, 'w0')])
        for field in ('start_ms', 'selected_ms', 'async_begin_ms', 'all_done_ms', 'controller_status'):
            tiered.pop(field)
        group = summarize(record([tiered]))['cells'][0]['transitions']['tiered_restore']
        self.assertEqual(group['per_rank_rpc_ms']['tiered_prefetch_ms']['p50'], 10)
        self.assertEqual(group['controller_wall_ms'], {})

    def test_conflicting_timing_and_duplicate_event_fail_closed(self):
        mutations = [lambda e: e.__setitem__('all_done_ms', 19),
                     lambda e: e['stages'][0].__setitem__('elapsed_ms', 99),
                     lambda e: e.__setitem__('observed_ms', float('nan')),
                     lambda e: e.__setitem__('request_id', 'another')]
        for mutate in mutations:
            run = record(); mutate(run['requests'][0]['controller_timing'][0])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                summarize(run)
        run = record(); run['requests'][0]['controller_timing'].append(copy.deepcopy(event()))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            summarize(run)

    def test_rank_fanout_success_requires_imports_and_correct_dependency_order(self):
        def missing_imports(e):
            e['stages'] = [s for s in e['stages'] if s['stage'] != 'import']
        def failed_import(e):
            e['status'] = 'failed'; e['controller_status'] = 'aborted'
            e['stages'][1]['status'] = 'failed'
        def early_advance(e):
            e['stages'][-3].update(start_ms=10, finish_ms=15, elapsed_ms=5)
        def import_outside_fanout(e):
            e['stages'][1].update(start_ms=4, finish_ms=14, elapsed_ms=10)
        def missing_target(e):
            e['transfer_targets'] = 3
        def missing_advance_rank(e):
            e['stages'].pop(-2)
        for change in (missing_imports, failed_import, early_advance, import_outside_fanout, missing_target, missing_advance_rank):
            with self.subTest(change=change):
                current = event(); change(current)
                with self.assertRaises(ValueError):
                    summarize(record([current]))

    def test_every_stop_is_bound_even_without_a_following_resume(self):
        for owner, ring, observed in [('missing', 20, 27), ('task1', 26, 27), ('task1', 20, 30), ('task1', 0, 27)]:
            with self.subTest(owner=owner, ring=ring, observed=observed):
                run = record()
                run['requests'][0]['preemption_stop_samples'] = [dict(event_id=owner+'/stop', task_event_id=owner,
                    ring_ms=ring, observed_ms=observed, stop_ms=observed-ring)]
                with self.assertRaises(ValueError):
                    summarize(run)

    def test_tiered_restore_link_requires_completed_same_request_and_precedes_dispatch(self):
        restored = event('restore', transition='tiered_restore', observed_ms=10,
                         stages=[stage('tiered_prefetch', 5, 9, 'w0')])
        for field in ('start_ms', 'selected_ms', 'async_begin_ms', 'all_done_ms', 'controller_status'):
            restored.pop(field)
        resumed = event('next', transition='resident_resume', selected_ms=11, start_ms=12,
                        async_begin_ms=13, all_done_ms=25, stages=[], restore_event_id='restore')
        valid = record([restored, resumed])
        self.assertEqual(len(summarize(valid)['cells'][0]['transitions']), 2)
        for field, value in [('restore_event_id', 'unknown'), ('start_ms', 9)]:
            run = copy.deepcopy(valid); run['requests'][0]['controller_timing'][1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(run)
        for status in ('failed',):
            run = copy.deepcopy(valid); run['requests'][0]['controller_timing'][0]['status'] = status
            with self.assertRaisesRegex(ValueError, 'completed restore'):
                summarize(run)
        run = copy.deepcopy(valid)
        restored = run['requests'][0]['controller_timing'].pop(0)
        restored['request_id'] = 'other'; restored['stages'][0]['request_id'] = 'other'
        run['requests'].append(dict(phase='c', attempt=1, cell=0, order='forward', request_id='other',
                                   status='completed', controller_timing=[restored]))
        with self.assertRaisesRegex(ValueError, 'same-request'):
            summarize(run)

    def test_cells_orders_and_attempts_are_not_pooled_and_source_inventories_are_compact(self):
        run = record(); run['requests'] = []
        for attempt, cell, order in [(1, 0, 'forward'), (1, 0, 'reverse'), (1, 1, 'forward'), (2, 0, 'forward')]:
            row = copy.deepcopy(record()['requests'][0]); current = row['controller_timing'][0]
            row.update(attempt=attempt, cell=cell, order=order)
            current.update(event_id=f'{attempt}-{cell}-{order}', attempt=attempt, cell=cell, order=order)
            run['requests'].append(row)
        run['source']['native_files'] = {'src/study.cpp': 'b'*64}
        result = summarize(run)
        self.assertNotIn('transitions', result)
        self.assertEqual(len(result['cells']), 4)
        self.assertTrue(all(cell['transitions']['regroup']['counts']['events'] == 1 for cell in result['cells']))
        self.assertEqual(result['source'], dict(core_sha256='c'*64,
            python_sources_sha256=canonical_hash(run['source']['files']),
            native_sources_sha256=canonical_hash(run['source']['native_files'])))


if __name__ == '__main__':
    unittest.main()
