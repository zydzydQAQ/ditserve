"""Source-integrity CPU fixtures; never evidence of native/GPU qualification."""

import copy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import p2_benefit as p2
import p2_contrasts as contrasts
import p3_window_review as p3
import test_20261103_p2_benefit as fixtures


class NativeSourceBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.P2BenefitTest.setUpClass()
        cls.fixture = fixtures.P2BenefitTest()
        cls.protocol = cls.fixture.f.f.p

    def record(self, bundle='c4b9e26-51740c0'):
        record = self.fixture.records()['candidate']
        record['source']['native_files'] = copy.deepcopy(contrasts.AUDITED_NATIVE_BUNDLES[bundle])
        return record

    def arm(self, record):
        return p2.planned_arm(record['config'], dict(phase='cluster', attempt=1, cell=0, order='forward'),
                              self.protocol, 201)

    def audit(self, record, **kwargs):
        return p2.audit_arm(record, self.arm(record), self.protocol, **kwargs)

    def assert_invalid_with_full_denominator(self, record, **kwargs):
        result = self.audit(record, **kwargs)
        self.assertEqual(result['status'], 'invalid', result)
        self.assertEqual(sum(row['planned'] for row in result['by_kind'].values()), 40)
        self.assertEqual(result['metrics'], {})

    def test_each_reviewed_bundle_is_atomic_and_retains_source_only_scope(self):
        for name in contrasts.AUDITED_NATIVE_BUNDLES:
            with self.subTest(bundle=name):
                result = self.audit(self.record(name))
                self.assertEqual(result['status'], 'completed', result)
                self.assertEqual(result['native_source_audit']['bundle'], name)
                self.assertFalse(result['native_source_audit']['qualification_eligible'])
                self.assertEqual(sum(row['planned'] for row in result['by_kind'].values()), 40)

    def test_reviewed_file_hashes_cannot_be_cross_combined(self):
        pairs = [('12b0918', '92b72ff', 'src/ditserve/control/study.cpp'),
                 ('92b72ff', 'c4b9e26-51740c0', 'src/ditserve/control/study.cpp'),
                 ('c4b9e26-51740c0', '92b72ff', 'src/ditserve/control/cluster.cpp'),
                 ('c4b9e26-51740c0', '97f0713', 'src/ditserve/runtime/process.cpp'),
                 ('97f0713', 'c4b9e26-51740c0', 'include/ditserve/runtime/process.hpp'),
                 ('97f0713', 'c4b9e26-51740c0', 'src/ditserve/control/budget.cpp'),
                 ('97f0713', 'c4b9e26-51740c0', 'src/ditserve/control/dispatch.cpp'),
                 ('97f0713', 'c4b9e26-51740c0', 'src/ditserve/interface/command.cpp')]
        for base, other, path in pairs:
            with self.subTest(base=base, other=other, path=path):
                record = self.record(base)
                record['source']['native_files'][path] = contrasts.AUDITED_NATIVE_BUNDLES[other][path]
                self.assert_invalid_with_full_denominator(record)

    def test_missing_changed_and_old_cluster_header_fail_closed(self):
        path = 'include/ditserve/control/cluster.hpp'
        for value in (None, 'f' * 64, contrasts.AUDITED_NATIVE_BUNDLES['92b72ff'][path]):
            record = self.record()
            if value is None:
                record['source']['native_files'].pop(path)
            else:
                record['source']['native_files'][path] = value
            with self.subTest(header=value):
                self.assert_invalid_with_full_denominator(record)

    def test_explicit_p3_pins_are_a_complete_singleton_snapshot(self):
        expected = contrasts.AUDITED_NATIVE_BUNDLES['97f0713']
        self.assertEqual({path: values[0] for path, values in p3.CUTOFF_NATIVE_SOURCES.items()}, expected)
        self.assertTrue(all(len(values) == 1 for values in p3.CUTOFF_NATIVE_SOURCES.values()))
        record = self.record('97f0713')
        result = self.audit(record, audited_native_sources=p3.CUTOFF_NATIVE_SOURCES)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['native_source_audit']['bundle'], 'explicit-singleton-map')
        self.assertFalse(result['native_source_audit']['qualification_eligible'])
        self.assert_invalid_with_full_denominator(self.record(), audited_native_sources=p3.CUTOFF_NATIVE_SOURCES)

    def test_multivalue_or_malformed_explicit_maps_cannot_restore_cartesian_acceptance(self):
        record = self.record('97f0713')
        key = 'src/ditserve/control/study.cpp'
        current = p3.CUTOFF_NATIVE_SOURCES[key][0]
        alternatives = [(current, contrasts.AUDITED_NATIVE_BUNDLES['92b72ff'][key]),
                        (current, current), [current, current], (), current, {current}, ('not-a-hash',)]
        for values in alternatives:
            pins = copy.deepcopy(p3.CUTOFF_NATIVE_SOURCES); pins[key] = values
            with self.subTest(values=values):
                self.assert_invalid_with_full_denominator(record, audited_native_sources=pins)
        pins = copy.deepcopy(p3.CUTOFF_NATIVE_SOURCES); pins.pop('include/ditserve/control/cluster.hpp')
        self.assert_invalid_with_full_denominator(record, audited_native_sources=pins)
        for pins in ({}, [], True):
            with self.subTest(pins=pins):
                self.assert_invalid_with_full_denominator(record, audited_native_sources=pins)

    def test_explicit_extra_pins_are_checked_without_broadening_default_bundles(self):
        record = self.record('97f0713')
        pins = copy.deepcopy(p3.CUTOFF_NATIVE_SOURCES)
        pins['src/extra-reviewed.cpp'] = ('e' * 64,)
        self.assert_invalid_with_full_denominator(record, audited_native_sources=pins)
        record['source']['native_files']['src/extra-reviewed.cpp'] = 'e' * 64
        self.assertEqual(self.audit(record, audited_native_sources=pins)['status'], 'completed')

    def test_historical_failed_record_stays_failed_with_all_forty_requests(self):
        record = self.record('12b0918')
        record['status'] = 'failed'
        record['phases'][0]['status'] = 'failed'
        record['phases'][0]['cells'] = [dict(cell=0, status='failed', error='solver-step owner conflict')]
        for row in record['requests'][30:]:
            row['status'] = 'interrupted'; row.pop('outputs')
        result = self.audit(record)
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(sum(row['planned'] for row in result['by_kind'].values()), 40)
        self.assertEqual(sum(row['completed'] for row in result['by_kind'].values()), 30)
        self.assertEqual(sum(row['interrupted'] for row in result['by_kind'].values()), 10)
        self.assertIsNone(result['elapsed_ms'])
        self.assertFalse(result['native_source_audit']['qualification_eligible'])

    def test_same_source_pairs_are_analyzable_without_transferring_qualification(self):
        for bundle in ('c4b9e26-51740c0', '97f0713'):
            with self.subTest(bundle=bundle), TemporaryDirectory() as directory:
                root = Path(directory)
                def use_bundle(records, seed):
                    for record in records.values():
                        record['source']['native_files'] = copy.deepcopy(contrasts.AUDITED_NATIVE_BUNDLES[bundle])
                evidence = self.fixture.evidence(root, mutate=use_bundle)
                result = p2.evaluate(evidence, root)
                self.assertEqual(result['status'], 'measured')
                self.assertEqual(result['expected_requests'], 240)
                self.assertEqual(result['quality']['status'], 'pending')
                self.assertEqual(result['allows'], [])
                self.fixture.mutate_run(root, evidence, 'candidate',
                    lambda record: record['source'].update(core_sha256='f' * 64))
                mixed = p2.evaluate(evidence, root)
                self.assertEqual(mixed['cases'][0]['comparison']['status'], 'invalid')
                self.assertEqual(mixed['expected_requests'], 240)
                self.assertEqual(mixed['allows'], [])


if __name__ == '__main__':
    unittest.main()
