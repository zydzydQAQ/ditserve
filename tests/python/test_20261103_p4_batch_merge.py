"""CPU/NumPy/mock contract tests, never a real LPIPS or P4 qualification."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import p4_quality_batches as batches
import p4_stream_quality as stream
from ditserve.evaluation.common import canonical_hash, seal, verify_seal
from test_20261103_p4_stream_quality import StreamFixture


def synthetic_adapter_receipt():
    """Only returned by explicitly mocked adapter construction in these tests."""
    offline = stream.offline_lpips
    value = dict(schema=offline.SCHEMA, status='ready', metric='lpips-alex-v0.1', device='cpu',
        preprocessing=stream.LPIPS_PREPROCESSING,
        adapter_sha256=stream.file_hash(Path(offline.__file__)), p4_qualification_eligible=False,
        retention_dependencies_closed=False,
        runtime=dict(python='fixture-not-real-lpips', packages=dict(offline.PACKAGES, numpy=np.__version__),
            sources={member: dict(path='/fixture/' + member, bytes=expected.size, sha256=expected.sha256)
                     for (_, member), expected in offline.SOURCES.items()}),
        weights={key: dict(path='/fixture/' + key, bytes=expected.size, sha256=expected.sha256)
                 for key, expected in (('backbone', offline.BACKBONE), ('calibrated_head', offline.HEAD))},
        selftest=dict(status='passed_real_cpu_weights', exact_distance=0., perturbed_distance=.1,
            symmetric_distance=.1, independent_reference_distance=.1, network_attempts=0,
            input_sha256=stream.LPIPS_SELFTEST_INPUT_SHA256))
    value['sha256'] = offline.canonical_hash(value)
    return value


class MockOfflineLPIPS:
    instances = []

    def __init__(self, backbone, head):
        self.tested = False
        self.calls = 0
        self.instances.append(self)

    def selftest(self):
        self.tested = True
        return self.receipt()

    def receipt(self):
        if not self.tested:
            raise AssertionError('must selftest before any metric receipt')
        return synthetic_adapter_receipt()

    def score_frame(self, first, second, *, input_range):
        if not self.tested or input_range != 'zero_one':
            raise AssertionError('selftest/transform contract bypassed')
        self.calls += 1
        return float(np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2))


def selected_evidence(fixture, pair_ids):
    evidence = fixture.evidence()
    evidence['runs'] = [row for row in evidence['runs'] if row['pair_id'] in pair_ids]
    return seal(evidence)


def report(fixture, pair_ids, **kwargs):
    return stream.evaluate(fixture.plan, fixture.protocol, selected_evidence(fixture, pair_ids),
                           root=fixture.root, **kwargs)


def produce(fixture, pairs):
    for pair in pairs:
        for arm in fixture.plan['arms']:
            value = .5 if arm['id'] == 'exact' else .51 if arm['id'] == 'fixed-1' else .52 + pair['prompt_index'] * .005
            fixture.produce(pair, arm, value=value, adaptive_folds=1)


class LPIPSIntegrationTests(unittest.TestCase):
    def test_complete_mock_numeric_grid_still_cannot_grant_p4_or_close_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory))
            for pair in fixture.plan['pairs']:
                for arm in fixture.plan['arms']:
                    fixture.produce(pair, arm)
            with patch.object(stream.offline_lpips, 'OfflineLPIPS', MockOfflineLPIPS):
                result = report(fixture, {pair['id'] for pair in fixture.plan['pairs']}, lpips=True)
            merged = batches.merge_results(fixture.plan, fixture.protocol, [result])
            self.assertEqual(merged['status'], 'complete')
            self.assertEqual(merged['lpips']['status'], 'measured')
            self.assertTrue(all(row['numeric_qualification'] == 'passed_numeric_only' for row in merged['by_arm'].values()))
            self.assertEqual(merged['matched_fold_comparison']['zero_fold_exact_self_controls'], 192)
            self.assertEqual(merged['matched_fold_comparison']['groups']['all']['lpips']['ci95'], [0., 0.])
            self.assertEqual(merged['vbench']['status'], 'unknown')
            self.assertFalse(merged['qualification_eligible'])
            self.assertFalse(merged['superiority_qualified'])
            self.assertFalse(merged['retention']['dependencies_closed'])

    def test_only_selftested_explicit_offline_adapter_can_supply_frame_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); fixture.one_pair()
            with patch.object(stream.offline_lpips, 'OfflineLPIPS', MockOfflineLPIPS):
                value = fixture.evaluate()
                self.assertEqual(value['lpips']['adapter']['requested'], False)
                enabled = report(fixture, {fixture.plan['pairs'][0]['id']}, lpips=True)
            self.assertEqual(enabled['lpips']['measured_pair_arms'], 3)
            self.assertEqual(enabled['lpips']['status'], 'partial')
            self.assertTrue(MockOfflineLPIPS.instances[-1].tested)
            self.assertEqual(MockOfflineLPIPS.instances[-1].calls, 6)
            self.assertEqual(enabled['lpips']['transform'], stream.LPIPS_TRANSFORM)
            self.assertEqual(enabled['metric_identity']['lpips']['identity']['weights']['backbone']['sha256'],
                             stream.offline_lpips.BACKBONE.sha256)
            for row in enabled['pair_results']:
                if row['status'] == 'measured':
                    self.assertEqual(row['metrics']['lpips_mean'], 0.)
                    self.assertEqual(row['metrics']['lpips_frames'], 2)
                    self.assertIsNone(row['metrics']['psnr_db'])
            self.assertFalse(enabled['qualification_eligible'])
            self.assertFalse(enabled['retention']['dependencies_closed'])
            batches.validate_batch(fixture.plan, fixture.protocol, enabled)

    def test_missing_dependencies_failed_selftest_or_bad_weight_receipt_remain_unknown(self):
        class FailedSelftest(MockOfflineLPIPS):
            def selftest(self):
                raise RuntimeError('selftest failed')
        class BadWeights(MockOfflineLPIPS):
            def receipt(self):
                value = synthetic_adapter_receipt()
                value['weights']['backbone']['sha256'] = '0' * 64
                value.pop('sha256'); value['sha256'] = stream.offline_lpips.canonical_hash(value)
                return value
        for factory in (FailedSelftest, BadWeights):
            with self.subTest(factory=factory), tempfile.TemporaryDirectory() as directory:
                fixture = StreamFixture(Path(directory)); fixture.one_pair()
                with patch.object(stream.offline_lpips, 'OfflineLPIPS', factory):
                    result = report(fixture, {fixture.plan['pairs'][0]['id']}, lpips=True)
                self.assertEqual(result['lpips']['adapter']['status'], 'unknown')
                self.assertEqual(result['lpips']['measured_pair_arms'], 0)
                self.assertTrue(all(row['metrics']['lpips_mean'] is None for row in result['pair_results'] if row['status'] == 'measured'))
        with patch.object(stream.offline_lpips, 'OfflineLPIPS', side_effect=ImportError('fixture dependency absent')):
            model, state = stream.prepare_lpips(True)
        self.assertIsNone(model)
        self.assertIn('dependency absent', state['reason'])

    def test_frame_failure_never_averages_only_the_successful_subset(self):
        class FailsOnSecondFrame:
            calls = 0
            def score_frame(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError('fixture per-frame failure')
                return .1
        first = np.zeros((1, 3, 4, 4, 3), dtype=np.float32)
        value = stream.frame_metrics(first, first + .1, lpips_adapter=FailsOnSecondFrame())
        self.assertGreater(value['mse'], 0.)
        self.assertEqual((value['lpips_status'], value['lpips_mean'], value['lpips_frames']), ('unknown', None, 0))

    def test_resealed_wrong_selftest_input_or_transform_is_not_a_valid_receipt(self):
        for field, value in (('preprocessing', 'resize to 64x64'), ('input_sha256', 'a' * 64)):
            receipt = synthetic_adapter_receipt()
            target = receipt['selftest'] if field == 'input_sha256' else receipt
            target[field] = value
            receipt.pop('sha256'); receipt['sha256'] = stream.offline_lpips.canonical_hash(receipt)
            with self.subTest(field=field), self.assertRaises(ValueError):
                stream.lpips_receipt_identity(receipt)
        with self.assertRaises(ValueError):
            stream.prepare_lpips(False, backbone='ignored-weight-path')
        with self.assertRaises(ValueError):
            stream.prepare_lpips('true')


class BatchMergeTests(unittest.TestCase):
    def test_candidate_batches_share_identical_exact_once_and_equal_joint_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory))
            pairs = [pair for pair in fixture.plan['pairs'] if pair['prompt_index'] < 2]
            produce(fixture, pairs)
            ids = {pair['id'] for pair in pairs}
            values = []
            with patch.object(stream.offline_lpips, 'OfflineLPIPS', MockOfflineLPIPS):
                for candidate in ('fixed-1', 'adaptive'):
                    evidence = selected_evidence(fixture, ids)
                    evidence['runs'] = [row for row in evidence['runs'] if row['arm_id'] in ('exact', candidate)]
                    values.append(stream.evaluate(fixture.plan, fixture.protocol, seal(evidence), root=fixture.root, lpips=True))
                joint = report(fixture, ids, lpips=True)
            with patch.object(stream, 'checked_array', side_effect=AssertionError('merge must not read arrays')):
                merged = batches.merge_results(fixture.plan, fixture.protocol, values)
                reverse = batches.merge_results(fixture.plan, fixture.protocol, values[::-1])
            self.assertEqual(merged, reverse)
            for key in ('denominator', 'by_arm', 'matched_fold_comparison'):
                self.assertEqual(merged[key], joint[key])
            self.assertEqual(merged['denominator']['submitted_pair_arms'], 36)
            reuse = merged['reference_reuse']
            self.assertEqual((reuse['shared_exact_anchors'], reuse['repeated_reference_observations_not_counted']), (12, 12))
            self.assertTrue(all(len(row['sources']) == 2 for row in reuse['anchors']))
            self.assertFalse(merged['retention']['dependencies_closed'])

    def test_different_exact_anchor_and_duplicate_candidate_remain_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); fixture.one_pair()
            pair_id = fixture.plan['pairs'][0]['id']
            values = []
            for candidate in ('fixed-1', 'adaptive'):
                evidence = selected_evidence(fixture, {pair_id})
                evidence['runs'] = [row for row in evidence['runs'] if row['arm_id'] in ('exact', candidate)]
                values.append(stream.evaluate(fixture.plan, fixture.protocol, seal(evidence), root=fixture.root))
            second = values[1]
            # A new, individually consistent anchor receipt must still conflict.
            for field in ('artifact_sha256', 'array_sha256', 'generation_sha256', 'path_history_sha256'):
                changed = copy.deepcopy(second)
                for row in changed['pair_results']:
                    if row['status'] != 'measured':
                        continue
                    row['reference_output'][field] = 'a' * 64
                    if row['arm_id'] == 'exact':
                        row['verified_output'][field] = 'a' * 64
                        if field == 'artifact_sha256':
                            row['artifact_sha256'] = 'a' * 64
                changed = seal(changed)
                batches.validate_batch(fixture.plan, fixture.protocol, changed)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'duplicate or conflicting'):
                    batches.merge_results(fixture.plan, fixture.protocol, [values[0], changed])
            # Re-evidence of a candidate is not deduplicated like an exact anchor.
            evidence = copy.deepcopy(values[1]['evidence']); evidence['note'] = 'duplicate adaptive'
            duplicate = stream.evaluate(fixture.plan, fixture.protocol, seal(evidence), root=fixture.root)
            with self.assertRaisesRegex(ValueError, 'duplicate or conflicting'):
                batches.merge_results(fixture.plan, fixture.protocol, [values[0], values[1], duplicate])

    def test_equivalent_receipt_paths_keep_every_receipt_and_merge_order_is_stable(self):
        class RelocatedWeights(MockOfflineLPIPS):
            def receipt(self):
                value = super().receipt()
                value['weights']['backbone']['path'] = '/other-allowed-project-path/alex.pth'
                value['selftest']['perturbed_distance'] += 1e-9
                value.pop('sha256'); value['sha256'] = stream.offline_lpips.canonical_hash(value)
                return value
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); produce(fixture, fixture.plan['pairs'][:2])
            with patch.object(stream.offline_lpips, 'OfflineLPIPS', MockOfflineLPIPS):
                first = report(fixture, {fixture.plan['pairs'][0]['id']}, lpips=True)
            with patch.object(stream.offline_lpips, 'OfflineLPIPS', RelocatedWeights):
                second = report(fixture, {fixture.plan['pairs'][1]['id']}, lpips=True)
            forward = batches.merge_results(fixture.plan, fixture.protocol, [first, second])
            backward = batches.merge_results(fixture.plan, fixture.protocol, [second, first])
            self.assertEqual(forward, backward)
            self.assertEqual({row['lpips_selftest_receipt']['sha256'] for row in forward['batches']},
                             {value['lpips']['adapter']['receipt']['sha256'] for value in (first, second)})

    def test_disjoint_batches_equal_joint_statistics_and_keep_prompt_clusters(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory))
            pairs = [pair for pair in fixture.plan['pairs'] if pair['prompt_index'] < 2]
            produce(fixture, pairs)
            groups = [{p['id'] for p in pairs if p['prompt_index'] == index} for index in (0, 1)]
            with patch.object(stream.offline_lpips, 'OfflineLPIPS', MockOfflineLPIPS):
                first, second = (report(fixture, group, lpips=True) for group in groups)
                joint = report(fixture, set.union(*groups), lpips=True)
            before = {p.name: stream.file_hash(p) for p in fixture.root.iterdir()}
            merged = batches.merge_results(fixture.plan, fixture.protocol, iter([first, second]))
            reversed_order = batches.merge_results(fixture.plan, fixture.protocol, [second, first])
            self.assertEqual(merged, reversed_order)
            verify_seal(merged)
            for key in ('denominator', 'by_arm', 'matched_fold_comparison'):
                self.assertEqual(merged[key], joint[key])
            stats = merged['matched_fold_comparison']['groups']['all']['lpips']
            self.assertEqual((stats['independent_prompt_clusters'], stats['observed_pairs']), (2, 12))
            self.assertEqual(merged['denominator']['planned_pair_arms'], 576)
            self.assertEqual(merged['denominator']['statuses'], {'measured': 36, 'missing': 540})
            self.assertFalse(merged['qualification_eligible'])
            self.assertEqual(before, {p.name: stream.file_hash(p) for p in fixture.root.iterdir()})

    def test_duplicate_same_conflicting_and_already_merged_observations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); fixture.one_pair()
            first = fixture.evaluate()
            for second in (first, copy.deepcopy(first)):
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    batches.merge_results(fixture.plan, fixture.protocol, [first, second])
            changed_evidence = copy.deepcopy(first['evidence'])
            changed_evidence['note'] = 'same observations through another manifest'
            changed_evidence = seal(changed_evidence)
            second = stream.evaluate(fixture.plan, fixture.protocol, changed_evidence, root=fixture.root)
            with self.assertRaisesRegex(ValueError, 'duplicate or conflicting'):
                batches.merge_results(fixture.plan, fixture.protocol, [first, second])
            merged = batches.merge_results(fixture.plan, fixture.protocol, [first])
            with self.assertRaisesRegex(ValueError, 'batch schema'):
                batches.merge_results(fixture.plan, fixture.protocol, [merged])

    def test_failure_and_blocked_denominators_cannot_be_healed_by_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory))
            pairs = fixture.plan['pairs'][:2]
            produce(fixture, pairs)
            entry = next(row for row in fixture.entries if row['pair_id'] == pairs[0]['id'] and row['arm_id'] == 'fixed-1')
            with (fixture.root / entry['artifact_path']).open('ab') as file:
                file.write(b'corrupt fixture')
            first = report(fixture, {pairs[0]['id']})
            evidence = selected_evidence(fixture, {pairs[1]['id']})
            evidence['runs'] = [row for row in evidence['runs'] if row['arm_id'] == 'adaptive']
            second = stream.evaluate(fixture.plan, fixture.protocol, seal(evidence), root=fixture.root)
            merged = batches.merge_results(fixture.plan, fixture.protocol, [first, second])
            self.assertEqual(merged['denominator']['submitted_pair_arms'], 4)
            self.assertEqual(merged['denominator']['statuses'], {'measured': 2, 'failed': 1, 'missing': 572, 'blocked': 1})
            self.assertFalse(merged['retention']['dependencies_closed'])

    def test_resealed_plan_core_model_metric_shape_reference_and_summary_tampering_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); fixture.one_pair()
            good = fixture.evaluate()
            mutations = [lambda r: r.update(plan_sha256='f' * 64),
                lambda r: r['source_binding'].update(core_sha256='f' * 64),
                lambda r: r['source_binding'].update(model_mapping_sha256='f' * 64),
                lambda r: r['metric_identity']['pixels'].update(range=[-1., 1.]),
                lambda r: r['pair_results'][0].update(generation_seed=999),
                lambda r: r['pair_results'][0]['verified_output'].update(shape=[1, 1, 4, 4, 3]),
                lambda r: r['pair_results'][1]['reference_output'].update(array_sha256='f' * 64),
                lambda r: r['pair_results'][0]['metrics'].update(psnr_db=100.),
                lambda r: r['pair_results'][0]['metrics'].update(lpips_status='measured', lpips_mean=0., lpips_frames=2, lpips_reason=None),
                lambda r: r['denominator'].update(planned_pair_arms=3),
                lambda r: r['by_arm']['exact'].update(numeric_qualification='passed_numeric_only'),
                lambda r: r.update(qualification_eligible=True),
                lambda r: r['retention'].update(dependencies_closed=True),
                lambda r: r['pair_results'].pop()]
            for mutation in mutations:
                bad = copy.deepcopy(good); mutation(bad); bad = seal(bad)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    batches.merge_results(fixture.plan, fixture.protocol, [bad])

    def test_different_runtime_sources_or_lpips_metric_modes_cannot_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); produce(fixture, fixture.plan['pairs'][:2])
            first = report(fixture, {fixture.plan['pairs'][0]['id']})
            with patch.object(stream.offline_lpips, 'OfflineLPIPS', MockOfflineLPIPS):
                second = report(fixture, {fixture.plan['pairs'][1]['id']}, lpips=True)
            with self.assertRaisesRegex(ValueError, 'different core/source/runtime/model/metric'):
                batches.merge_results(fixture.plan, fixture.protocol, [first, second])
            evidence = selected_evidence(fixture, {fixture.plan['pairs'][1]['id']})
            evidence['runtime_source_sha256'] = 'f' * 64
            second = stream.evaluate(fixture.plan, fixture.protocol, seal(evidence), root=fixture.root)
            self.assertTrue(any(row['status'] == 'failed' for row in second['pair_results']))
            with self.assertRaisesRegex(ValueError, 'different core/source/runtime/model/metric'):
                batches.merge_results(fixture.plan, fixture.protocol, [first, second])

    def test_cli_requires_byte_hashes_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StreamFixture(Path(directory)); fixture.one_pair()
            value = fixture.evaluate()
            for name, document in (('plan', fixture.plan), ('protocol', fixture.protocol), ('batch', value)):
                (fixture.root / (name + '.json')).write_text(json.dumps(document))
            manifest = seal(dict(schema=batches.INPUT_SCHEMA, plan_sha256=fixture.plan['sha256'],
                batches=[dict(path='batch.json', sha256=stream.file_hash(fixture.root / 'batch.json'))]))
            (fixture.root / 'inputs.json').write_text(json.dumps(manifest))
            args = ['--plan', str(fixture.root / 'plan.json'), '--protocol', str(fixture.root / 'protocol.json'),
                    '--batches', str(fixture.root / 'inputs.json'), '--output', str(fixture.root / 'merged.json')]
            with patch('sys.stdout', io.StringIO()):
                batches.main(args)
                with self.assertRaises(FileExistsError):
                    batches.main(args)
            result = json.loads((fixture.root / 'merged.json').read_bytes())
            self.assertEqual(result['batch_file_sources'][0]['sha256'], manifest['batches'][0]['sha256'])
            with (fixture.root / 'batch.json').open('a') as file:
                file.write(' ')
            args[-1] = str(fixture.root / 'not-written.json')
            with self.assertRaisesRegex(ValueError, 'byte SHA256'):
                batches.main(args)
            self.assertFalse((fixture.root / 'not-written.json').exists())


if __name__ == '__main__':
    unittest.main()
