"""CPU guards for the actual distributed GFC smoke; no simulated GPU pass."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('external_gfc_smoke', ROOT / 'benchmarks/external_gfc_smoke.py')
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class ExternalGFCSmokeTests(unittest.TestCase):
    def test_two_and_eight_rank_plans_cover_full_denominators(self):
        two = smoke.cases(2)
        self.assertEqual(len(two), 88)
        self.assertEqual(smoke.expected_counts(two, 2), [dict(rank=r, executions=88, byte_checks=80) for r in range(2)])
        eight = smoke.cases(8)
        counts = smoke.expected_counts(eight, 8)
        self.assertEqual(len(eight), 232)
        self.assertEqual(sum(c['executions'] for c in counts), 1016)
        self.assertEqual(sum(c['byte_checks'] for c in counts), 952)
        groups = {tuple(c['ranks']) for c in eight if c['op'] == 'all2all'}
        self.assertIn((1, 2, 4), groups)
        self.assertIn((0, 7), groups)
        self.assertIn(tuple(reversed(range(8))), groups)
        self.assertEqual({c['dtype'] for c in eight}, set(smoke.DTYPES))
        self.assertEqual({c['bytes_per_peer'] for c in eight}, set(smoke.SIZES))
        self.assertEqual({c['op'] for c in eight}, {'all_gather', 'all2all', 'p2p_put', 'p2p_get'})

    def test_one_rank_never_claims_collective_execution(self):
        self.assertEqual(smoke.cases(1), [])
        records = [dict(rank=0, world_size=1, status='passed', plan_sha256='plan',
                        source_binding_sha256='source', executions=0, byte_checks=0,
                        cleanup_verified=True, subgroup_process_groups_created=0,
                        runtime={'gpu_uuid': 'GPU-0'})]
        result = smoke.summarize(records, 1, 'plan', smoke.expected_counts([], 1), 'source')
        self.assertFalse(result['collectives_executed'])
        self.assertFalse(result['serving_executed'])
        self.assertFalse(result['qualification_eligible'])
        self.assertEqual(result['measurements'], [])

    def test_partial_failed_mixed_source_and_boolean_counts_are_rejected(self):
        expected = smoke.expected_counts(smoke.cases(2), 2)
        records = [dict(rank=r, world_size=2, status='passed', plan_sha256='plan',
                        source_binding_sha256='source', executions=88, byte_checks=80,
                        cleanup_verified=True, subgroup_process_groups_created=0,
                        runtime={'gpu_uuid': 'GPU-' + str(r)}) for r in range(2)]
        smoke.summarize(records, 2, 'plan', expected, 'source')
        with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
            smoke.summarize(records[:1], 2, 'plan', expected, 'source')
        for field, value in [('executions', 87), ('byte_checks', 79), ('executions', True),
                             ('cleanup_verified', False), ('status', 'failed'),
                             ('source_binding_sha256', 'different'), ('plan_sha256', 'different'),
                             ('subgroup_process_groups_created', 1)]:
            bad = copy.deepcopy(records)
            bad[1][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'inconsistent'):
                smoke.summarize(bad, 2, 'plan', expected, 'source')
        bad = copy.deepcopy(records)
        bad[1]['runtime']['gpu_uuid'] = bad[0]['runtime']['gpu_uuid']
        with self.assertRaisesRegex(ValueError, 'GPU UUIDs'):
            smoke.summarize(bad, 2, 'plan', expected, 'source')

    def test_storage_guard_enforces_both_stop_lines_and_unknown_state(self):
        def output(used, percent):
            return subprocess.CompletedProcess([], 0,
                'Filesystem 1B-blocks Used Available Use% Mounted on\n'
                f'/dev/sdb 1099511627776 {used} 200000000000 {percent}% /shared\n', '')
        with mock.patch.object(smoke.subprocess, 'run', return_value=output(790 * 1024**3, 79)):
            self.assertEqual(smoke.storage_guard()['used_percent'], 79)
        for used, percent in [(790 * 1024**3 + 1, 79), (700 * 1024**3, 80)]:
            with mock.patch.object(smoke.subprocess, 'run', return_value=output(used, percent)), self.assertRaises(smoke.StorageStop):
                smoke.storage_guard()
        for result in (subprocess.CompletedProcess([], 0, 'invalid', ''),):
            with mock.patch.object(smoke.subprocess, 'run', return_value=result), self.assertRaises(smoke.StorageStop):
                smoke.storage_guard()
        with mock.patch.object(smoke.subprocess, 'run', side_effect=OSError('unavailable')), self.assertRaises(smoke.StorageStop):
            smoke.storage_guard()

    def test_process_group_fallback_guard_restores_real_method(self):
        original = lambda: 'original'
        dist = SimpleNamespace(new_group=original)
        with smoke.forbid_subgroups(dist):
            with self.assertRaisesRegex(RuntimeError, 'forbidden'):
                dist.new_group([0, 1])
        self.assertIs(dist.new_group, original)

    def test_invalid_world_and_iterations_are_rejected(self):
        for world in (True, 0, 3, 4, 16, '2'):
            with self.subTest(world=world), self.assertRaises(ValueError):
                smoke.cases(world)
        for iterations in (True, 0, -1, 11, 2.0):
            with self.subTest(iterations=iterations), self.assertRaises(ValueError):
                smoke.cases(2, iterations)

    def test_source_requires_exact_pinned_package_and_rejects_injected_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, shared = root / 'source', root / 'shared'
            for where in (source, shared):
                (where / 'gfc').mkdir(parents=True)
                (where / 'gfc/__init__.py').write_text('# inventory fixture only\n')
            files = {'gfc/__init__.py': smoke.digest((source / 'gfc/__init__.py').read_bytes())}
            manifest = root / 'source.json'
            manifest.write_text(json.dumps({'schema': 'ditserve.official-source-archive.v1',
                'repository': smoke.REPOSITORY, 'commit': smoke.COMMIT,
                'archive_sha256': smoke.ARCHIVE_SHA256, 'files': files}))
            sha = smoke.digest(manifest.read_bytes())
            result = smoke.verify_sources(source, manifest, sha, shared)
            self.assertEqual(result['verified_package_files'], files)
            for name in ('extra.py', 'extra.so', 'extra.pyc'):
                extra = shared / 'gfc' / name
                extra.write_bytes(b'injected')
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'importable files differ'):
                    smoke.verify_sources(source, manifest, sha, shared)
                extra.unlink()
            (shared / 'gfc/__init__.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'importable files differ'):
                smoke.verify_sources(source, manifest, sha, shared)
            with self.assertRaisesRegex(ValueError, 'manifest SHA256'):
                smoke.verify_sources(source, manifest, '0' * 64, shared)


if __name__ == '__main__':
    unittest.main()
