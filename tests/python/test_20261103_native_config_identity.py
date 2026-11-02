"""Native float spelling, frozen parsed inputs and record/session/receipt links."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from native_config_identity import config_identity, match_native_config
import p1_isolated_batch as isolated
import p1_isolated_review as isolated_review
import p2_benefit as p2
import p3_window_review as p3
import formal_benchmark as formal
from stage_gates import canonical_hash, source_file
import test_20261103_p1_isolated_review as p1_fixtures
import test_20261103_p2_benefit as p2_fixtures
import test_20261103_p3_cutoff_review as p3_fixtures


# Reproduced with project nlohmann/json 3.11.3:
# json j = json::parse(R"({"arrival_ms":521340.85572874843})");
# std::cout << j.dump();
# The same spelling occurs in the real 51740c0 Wan21 native config. CPython
# chooses 521340.8557287484; both round-trip to the same IEEE-754 double.
NATIVE_BYTES = b'{"arrival_ms":521340.85572874843}'
NATIVE_SHA = hashlib.sha256(NATIVE_BYTES).hexdigest()


def bind_native(record, digest=NATIVE_SHA):
    """Synthetic native identity: tests consistency, not digest authentication."""
    record['config_sha256'] = digest
    for session in record.get('budget', {}).get('sessions', []):
        session.setdefault('identity', {})['config_sha256'] = digest
    for phase in record.get('phases', []):
        if 'identity' in phase.get('charge', {}):
            phase['charge']['identity']['config_sha256'] = digest
        for cell in phase.get('cells', []):
            if 'replay_cutoff' in cell:
                cell['replay_cutoff']['config_sha256'] = digest


class NativeConfigIdentityTest(unittest.TestCase):
    def test_real_native_float_spelling_is_not_python_canonical_bytes(self):
        config = json.loads(NATIVE_BYTES)
        self.assertEqual(config['arrival_ms'], 521340.8557287484)
        self.assertNotEqual(canonical_hash(config), NATIVE_SHA)
        binding = config_identity(dict(config=config, config_sha256=NATIVE_SHA), config)
        self.assertEqual(binding['native_config_sha256'], NATIVE_SHA)
        self.assertEqual(binding['parsed_config_sha256'], canonical_hash(config))
        self.assertFalse(binding['native_digest_recomputed'])
        match_native_config(binding, dict(config_sha256=NATIVE_SHA))

    def test_parsed_tamper_and_invalid_digest_are_rejected(self):
        expected = json.loads(NATIVE_BYTES)
        for digest in (None, '', 'g' * 64, 'A' * 64, 'a' * 63, True):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                config_identity(dict(config=expected, config_sha256=digest), expected)
        actual = dict(expected, arrival_ms=expected['arrival_ms'] + 1)
        with self.assertRaisesRegex(ValueError, 'parsed configuration'):
            config_identity(dict(config=actual, config_sha256=NATIVE_SHA), expected)
        with self.assertRaisesRegex(ValueError, 'parsed configuration'):
            config_identity(dict(config={'enabled': 1}, config_sha256=NATIVE_SHA), {'enabled': True})

    def test_legacy_missing_is_explicit_and_cannot_accept_orphan_or_missing_owner_hash(self):
        config = json.loads(NATIVE_BYTES)
        legacy = config_identity(dict(config=config), config, require_native=False)
        self.assertIsNone(legacy['native_config_sha256'])
        match_native_config(legacy, {}, required=False)
        with self.assertRaises(ValueError):
            match_native_config(legacy, dict(config_sha256=NATIVE_SHA), required=False)
        native = config_identity(dict(config=config, config_sha256=NATIVE_SHA), config)
        for owner in ({}, dict(config_sha256='a' * 64), dict(config_sha256=None)):
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                match_native_config(native, owner)

    def test_raw_reference_sha_still_rejects_even_equal_parsed_content(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_bytes(NATIVE_BYTES)
            reference = dict(path='config.json', sha256=NATIVE_SHA)
            self.assertEqual(source_file(root, reference), NATIVE_BYTES)
            python_bytes = json.dumps(json.loads(NATIVE_BYTES), sort_keys=True, separators=(',', ':')).encode()
            self.assertEqual(json.loads(python_bytes), json.loads(NATIVE_BYTES))
            (root / 'config.json').write_bytes(python_bytes)
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                source_file(root, reference)


class NativeConfigConsumerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p1_fixtures.IsolatedReviewTest.setUpClass()
        cls.one = p1_fixtures.IsolatedReviewTest()
        p2_fixtures.P2BenefitTest.setUpClass()
        cls.two = p2_fixtures.P2BenefitTest()
        p3_fixtures.P3CutoffReviewTest.setUpClass()
        cls.three = p3_fixtures.P3CutoffReviewTest()

    def test_p1_calibration_compile_accepts_native_identity_and_rejects_session_mismatch(self):
        inputs = copy.deepcopy(self.one.inputs)
        for record in inputs['calibrations']:
            bind_native(record)
        model = inputs['pair_id'].split('-')[0]
        inputs['bundle'] = formal.assemble(inputs['protocol'], model, inputs['calibrations'], inputs['runtime_facts'])
        config = isolated.compile_case(inputs)
        self.assertEqual(isolated.coverage(config)['expected_requests'], 96)
        for missing in (False, True):
            changed = copy.deepcopy(inputs)
            owner = changed['calibrations'][0]['budget']['sessions'][0]['identity']
            if missing:
                del owner['config_sha256']
            else:
                owner['config_sha256'] = 'b' * 64
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'native config hash'):
                isolated.compile_case(changed)

    def test_p1_replay_coverage_and_review_use_native_session_hash(self):
        record = self.one.fresh()
        bind_native(record)
        self.assertEqual(isolated.coverage(self.one.config, record)['completed'], 96)
        result = self.one.run_review(record)
        self.assertEqual(result['status'], 'measured_diagnostic')
        self.assertEqual(result['timing_eligible_requests'], 96)
        self.assertFalse(result['qualification_eligible'])
        changed = copy.deepcopy(record)
        changed['config'] = copy.deepcopy(record['config'])
        changed['config']['benchmark']['isolated_batch']['count'] = 7
        with self.assertRaisesRegex(ValueError, 'parsed configuration'):
            isolated.coverage(self.one.config, changed)
        for owner in (record['budget']['sessions'][0]['identity'],):
            owner['config_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'native config hash'):
            isolated.coverage(self.one.config, record)

    def test_p2_selected_session_and_charge_bind_native_not_python(self):
        record = self.two.records()['candidate']
        selector = dict(phase='cluster', attempt=1, cell=0, order='forward')
        arm = p2.planned_arm(record['config'], selector, self.two.f.f.p, 201)
        bind_native(record)
        result = p2.audit_arm(record, arm, self.two.f.f.p)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['config_identity']['native_config_sha256'], NATIVE_SHA)
        self.assertFalse(result['config_identity']['native_digest_recomputed'])
        for kind in ('missing_session_hash', 'charge_mismatch', 'changed_config'):
            changed = copy.deepcopy(record)
            if kind == 'missing_session_hash':
                del changed['budget']['sessions'][0]['identity']['config_sha256']
            elif kind == 'charge_mismatch':
                changed['phases'][0]['charge'] = dict(identity=dict(config_sha256='b' * 64))
            else:
                changed['config']['benchmark']['seed'] = 999
            with self.subTest(kind=kind):
                result = p2.audit_arm(changed, arm, self.two.f.f.p)
                self.assertEqual(result['status'], 'invalid')
                self.assertEqual(sum(r['planned'] for r in result['by_kind'].values()), 40)

    def test_p3_receipt_is_checked_before_projection_with_native_identity(self):
        configs, _ = self.three.prepared()
        config = configs[301]
        record = self.three.record(config)
        selector = dict(phase='cluster', attempt=1, cell=0, order='forward')
        arm = p2.planned_arm(config, selector, self.three.f.f.p, 301)
        bind_native(record)
        result = p3.audit_replay(record, arm, self.three.f.f.p)
        self.assertEqual(result['status'], 'censored', result)
        self.assertEqual(result['cutoff_receipt']['config_sha256'], NATIVE_SHA)
        second_arm = p2.planned_arm(config, dict(selector, cell=1), self.three.f.f.p, 301)
        self.assertEqual(p3.audit_replay(record, second_arm, self.three.f.f.p)['status'], 'censored')
        changed = copy.deepcopy(record)
        changed['phases'][0]['cells'][0]['replay_cutoff']['config_sha256'] = 'b' * 64
        self.assertEqual(p3.audit_replay(changed, second_arm, self.three.f.f.p)['status'], 'invalid')
        for target in ('receipt', 'session', 'charge', 'missing_record'):
            changed = copy.deepcopy(record)
            if target == 'receipt':
                changed['phases'][0]['cells'][0]['replay_cutoff']['config_sha256'] = 'b' * 64
            elif target == 'session':
                changed['budget']['sessions'][0]['identity']['config_sha256'] = 'b' * 64
            elif target == 'charge':
                changed['phases'][0]['charge']['identity']['config_sha256'] = 'b' * 64
            else:
                del changed['config_sha256']
            with self.subTest(target=target):
                self.assertEqual(p3.audit_replay(changed, arm, self.three.f.f.p)['status'], 'invalid')


if __name__ == '__main__':
    unittest.main()
