"""GPU-time accounting must not count batch members as separate busy intervals."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from native_metrics import approximation_summary, summarize_run


def video():
    return dict(request_id='v', kind='video', status='completed', spec=dict(steps=20),
        cfg_fold=dict(full_cfg_steps=19, folded_cfg_steps=1, folded_step_indices=[12],
                      final_step_exact_anchor=True), advance_samples=[
            dict(form='cfg2sp3', group=[0, 1, 2, 3, 4, 5], cfg_folded=False,
                 start_ms=0, finish_ms=10, elapsed_ms=10),
            dict(form='sp3', group=[0, 1, 2], cfg_folded=True,
                 start_ms=10, finish_ms=30, elapsed_ms=20)])


def image(rid, gpu=4, start=15, finish=25):
    return dict(request_id=rid, kind='image', status='completed', gpu_group=[gpu],
                arrival_ms=0, start_ms=start, finish_ms=finish, deadline_ms=100)


class ReleasedCapacityTest(unittest.TestCase):
    def test_batch_members_share_gpu_time_but_remain_separate_requests(self):
        result = approximation_summary([video()] + [image(f'i{i}') for i in range(7)])
        self.assertEqual(result['released_gpu_ms_capacity'], 60)
        self.assertEqual(result['image_overlap_on_released_gpu_ms'], 10)
        self.assertEqual(result['image_requests_on_released_gpus'], 7)
        self.assertFalse(result['reuse_qualified'])

    def test_missing_batch_member_placement_remains_visible(self):
        member = image('unplaced'); member.pop('gpu_group')
        result = approximation_summary([video(), image('placed'), member])
        self.assertEqual(result['image_requests_missing_placement'], 1)
        self.assertFalse(result['overlap_attribution_complete'])
        self.assertEqual(result['image_overlap_on_released_gpu_ms'], 10)

    def test_release_and_image_unions_are_per_gpu_not_per_request(self):
        # Even overlapping inferred release claims must not double-count time.
        other = video(); other['request_id'] = 'v2'
        result = approximation_summary([video(), other, image('i0', 3, 0, 20),
                                        image('i1', 3, 15, 40), image('i2', 4, 0, 100)])
        self.assertEqual(result['released_gpu_ms_capacity'], 60)
        self.assertEqual(result['image_overlap_on_released_gpu_ms'], 40)
        self.assertLessEqual(result['image_overlap_on_released_gpu_ms'], result['released_gpu_ms_capacity'])

    def test_latest_full_group_controls_the_contraction_proxy(self):
        row = video()
        moved = copy.deepcopy(row['advance_samples'][0])
        moved.update(group=[2, 3, 4, 5, 6, 7], start_ms=10, finish_ms=20)
        row['advance_samples'].insert(1, moved)
        row['advance_samples'][2].update(group=[2, 3, 4], start_ms=20, finish_ms=40)
        result = approximation_summary([row, image('i0', 6, 22, 32), image('i1', 1, 22, 32)])
        self.assertEqual(result['released_gpu_ms_capacity'], 60)
        self.assertEqual(result['image_overlap_on_released_gpu_ms'], 10)
        self.assertEqual(result['image_requests_on_released_gpus'], 1)

    def test_unknown_layout_is_not_assigned_release_credit(self):
        row = video(); row['advance_samples'][1]['group'] = [6, 7, 8]
        result = approximation_summary([row, image('i0')])
        self.assertEqual(result['unattributed_folded_quanta'], 1)
        self.assertEqual(result['released_gpu_ms_capacity'], 0)
        self.assertEqual(result['image_overlap_on_released_gpu_ms'], 0)
        row['advance_samples'].reverse()
        with self.assertRaisesRegex(ValueError, 'prior full-CFG'):
            approximation_summary([row])

    def test_prior_exact_contraction_does_not_earn_fold_release_credit(self):
        row = video()
        exact = dict(row['advance_samples'][1], cfg_folded=False, start_ms=10, finish_ms=20)
        row['advance_samples'].insert(1, exact)
        row['advance_samples'][2].update(start_ms=20, finish_ms=30)
        result = approximation_summary([row, image('i', 4, 20, 30)])
        self.assertEqual(result['unattributed_folded_quanta'], 1)
        self.assertEqual(result['released_gpu_ms_capacity'], 0)
        self.assertEqual(result['image_overlap_on_released_gpu_ms'], 0)

    def test_inverted_or_nonfinite_intervals_fail(self):
        for finish in (9, float('nan'), float('inf')):
            row = video(); row['advance_samples'][1]['finish_ms'] = finish
            with self.subTest(finish=finish), self.assertRaisesRegex(ValueError, 'timed execution'):
                approximation_summary([row])

    def test_forward_reverse_and_retry_cells_keep_separate_denominators(self):
        record = dict(experiment='fixture', run_id='test', status='completed', metrics={},
                      budget={}, phases=[], requests=[])
        for attempt in (1, 2):
            phase = dict(id='cluster', attempt=attempt, cells=[])
            for order in ('forward', 'reverse'):
                phase['cells'].append(dict(cell=0, order=order, status='completed',
                                           policy='block-cfg2', elapsed_ms=100, gpus=2))
                row = image(f'{attempt}-{order}')
                row.update(phase='cluster', attempt=attempt, cell=0, order=order)
                record['requests'].append(row)
            record['phases'].append(phase)
        summaries = summarize_run(record)['cells']
        self.assertEqual(len(summaries), 4)
        self.assertEqual([cell['by_kind']['image']['requests'] for cell in summaries], [1] * 4)
        unique_retry = copy.deepcopy(record)
        unique_retry['phases'] = unique_retry['phases'][1:]
        unique_retry['requests'] = unique_retry['requests'][2:]
        for row in unique_retry['requests']:
            row.pop('attempt')
        self.assertEqual([cell['by_kind']['image']['requests'] for cell in summarize_run(unique_retry)['cells']], [1, 1])
        record['requests'][0].pop('order')
        with self.assertRaisesRegex(ValueError, 'unambiguous attempt/order'):
            summarize_run(record)


if __name__ == '__main__':
    unittest.main()
