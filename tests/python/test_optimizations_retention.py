"""Tiny stdlib-only finalization fixtures; never touch real experiments."""

import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve import compact_results as compact
from ditserve.optimizations import retention


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.output = base / 'results/optimization-screen/run01'
        self.cache = base / '.cache/optimization-screen/run01'
        self.ledger = write(self.output.parent / 'gpu-budget.json', {'active': None})
        for name, value in [('OUTPUT', self.output), ('CACHE', self.cache), ('LEDGER', self.ledger)]:
            patcher = patch.object(retention, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for phase in retention.PHASES:
            write(self.output / phase / 'terminal.json',
                  {'status': 'stopped', 'gpu_work_started': False})
        self.media = self.output / 'archive-previews/key.png'
        self.media.parent.mkdir(parents=True)
        self.media.write_bytes(b'tiny retained preview')
        stage = self.output / 'archive-metadata'
        plan = write(stage / 'raw-retention-plan.json',
                     {'previews': [{'path': str(self.media), 'sha256': digest(self.media)}], 'videos': []})
        self.primary = compact.export(stage, self.output / 'excel')
        self.primary.update(verification=compact.verify(self.output / 'excel', check_sources=True),
                            retention_plan=str(plan))
        self.required_media = [{'path': str(self.media), 'sha256': digest(self.media)}]

    def artifact(self, *, cached=False, name='image.npy'):
        leaf = self.cache / 'vae/attempt00' if cached else self.output / 'sana_compile/attempt00/images'
        leaf.mkdir(parents=True, exist_ok=True)
        raw = leaf / name
        raw.write_bytes(b'tiny raw image ' + name.encode())
        source = write(self.output / 'validation' / (name + '.json'),
                       {'rows': [{'artifact': {'path': str(raw), 'sha256': digest(raw)}}]})
        return {'path': str(raw), 'sha256': digest(raw),
                'source': {'path': str(source), 'sha256': digest(source), 'pointer': '/rows/0/artifact'},
                'dependencies_closed': True, 'closure_reason': 'quality and continuation completed; key preview retained'}

    def finalize(self, artifacts, **kwargs):
        return retention.finalize(self.output, self.ledger, artifacts,
                                  primary_archive=self.primary, required_media=self.required_media, **kwargs)

    def test_real_leaf_archive_readback_deletes_only_explicit_array_and_is_idempotent(self):
        item = self.artifact()
        keep = self.artifact(name='pending.npy')
        leaf = Path(item['path']).parent
        metadata = write(leaf / 'metadata.json', {'retain': True})
        video = leaf / 'keep.mp4'
        video.write_bytes(b'keep')
        original_guard = compact._guard
        result = self.finalize([item])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['deleted_files'], 1)
        self.assertEqual(result['deleted_bytes'], len(b'tiny raw image image.npy'))
        self.assertFalse(Path(item['path']).exists())
        self.assertTrue(Path(keep['path']).exists())
        self.assertTrue(metadata.exists() and video.exists() and self.media.exists())
        self.assertIs(compact._guard, original_guard)
        archive = Path(result['groups'][0]['archive'])
        self.assertEqual(compact.verify(archive)['status'], 'verified')
        receipt = json.loads((archive / 'pruned.json').read_text())
        self.assertEqual(set(receipt['deleted']), {'image.npy'})
        repeated = self.finalize([item])
        self.assertEqual(repeated['deleted_files'], 1)
        self.assertEqual(repeated['groups'][0]['newly_deleted'], [])

    def test_cache_leaf_holds_real_results_ledger_lock_through_prune(self):
        item = self.artifact(cached=True)
        original = compact.prune
        def checked(*args, **kwargs):
            with self.ledger.with_suffix('.lock').open('a+') as handle:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return original(*args, **kwargs)
        with patch.object(compact, 'prune', checked):
            self.assertEqual(self.finalize([item])['status'], 'completed')

    def test_active_ledger_and_unclosed_dependencies_never_delete(self):
        item = self.artifact()
        item['dependencies_closed'] = False
        with self.assertRaisesRegex(ValueError, 'closure'):
            self.finalize([item])
        item['dependencies_closed'] = True
        write(self.ledger, {'active': {'stage': 'vae'}})
        with self.assertRaisesRegex(ValueError, 'active'):
            self.finalize([item])
        self.assertTrue(Path(item['path']).exists())

    def test_source_pointer_seed_record_and_hash_must_match(self):
        item = self.artifact()
        item['source']['pointer'] = '/rows/0'
        with self.assertRaises((ValueError, KeyError)):
            self.finalize([item])
        item['source']['pointer'] = '/rows/0/artifact'
        item['source']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'JSON hash'):
            self.finalize([item])
        self.assertTrue(Path(item['path']).exists())

    def test_media_or_workbook_tamper_blocks_cleanup(self):
        item = self.artifact()
        self.media.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'media hash'):
            self.finalize([item])
        self.media.write_bytes(b'tiny retained preview')
        with Path(self.primary['workbook']).open('ab') as stream:
            stream.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'archive hash'):
            self.finalize([item])
        self.assertTrue(Path(item['path']).exists())

    def test_source_change_at_prune_boundary_is_checked_under_guard(self):
        item = self.artifact()
        original, original_guard = compact.prune, compact._guard
        def changed(*args, **kwargs):
            write(Path(item['source']['path']), {'changed': True})
            return original(*args, **kwargs)
        with patch.object(compact, 'prune', changed):
            with self.assertRaisesRegex(ValueError, 'bound source changed'):
                self.finalize([item])
        self.assertTrue(Path(item['path']).exists())
        self.assertIs(compact._guard, original_guard)
        receipt = json.loads((self.output / 'raw-finalization/receipt.json').read_text())
        self.assertEqual(receipt['status'], 'incomplete')

    def test_symlink_outside_root_and_non_npy_are_rejected(self):
        item = self.artifact()
        for bad in (self.output.parent / 'run00/old.npy', self.output / 'keep.mp4'):
            bad.parent.mkdir(parents=True, exist_ok=True)
            bad.write_bytes(b'keep')
            with self.subTest(path=str(bad)), self.assertRaisesRegex(ValueError, 'NPY'):
                self.finalize([dict(item, path=str(bad))])
        target = Path(item['path'])
        link = target.with_name('link.npy')
        link.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.finalize([dict(item, path=str(link))])
        self.assertTrue(target.exists())

    def test_raw_hash_change_prevents_prune_and_guard_is_restored(self):
        item = self.artifact()
        Path(item['path']).write_bytes(b'changed raw')
        original_guard = compact._guard
        with self.assertRaisesRegex(ValueError, 'expected artifact'):
            self.finalize([item])
        self.assertTrue(Path(item['path']).exists())
        self.assertIs(compact._guard, original_guard)

    def test_parent_tree_and_missing_terminal_cannot_be_finalized(self):
        item = self.artifact()
        (Path(item['path']).parent / 'compiler-cache').mkdir()
        with self.assertRaisesRegex(ValueError, 'flat artifact leaf'):
            self.finalize([item])
        (Path(item['path']).parent / 'compiler-cache').rmdir()
        (self.output / retention.PHASES[-1] / 'terminal.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.finalize([item])
        self.assertTrue(Path(item['path']).exists())

    def test_new_active_ledger_at_delete_boundary_blocks_the_whole_leaf(self):
        first, second = self.artifact(), self.artifact(name='second.npy')
        original = compact.prune
        def activated(*args, **kwargs):
            write(self.ledger, {'active': {'stage': 'new_attempt'}})
            return original(*args, **kwargs)
        with patch.object(compact, 'prune', activated):
            with self.assertRaisesRegex(ValueError, 'active'):
                self.finalize([first, second])
        self.assertTrue(Path(first['path']).exists() and Path(second['path']).exists())

    def test_interrupt_after_unlink_can_resume_using_compact_durable_intent(self):
        item = self.artifact()
        original_unlink, original_guard = Path.unlink, compact._guard
        def interrupted(path, *args, **kwargs):
            value = original_unlink(path, *args, **kwargs)
            if str(path) == item['path']:
                raise KeyboardInterrupt('after unlink')
            return value
        with patch.object(Path, 'unlink', interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.finalize([item])
        self.assertIs(compact._guard, original_guard)
        self.assertFalse(Path(item['path']).exists())
        result = self.finalize([item])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['deleted_files'], 1)


if __name__ == '__main__':
    unittest.main()
