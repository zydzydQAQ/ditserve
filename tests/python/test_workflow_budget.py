"""Budget recovery must never turn failed or interrupted measurements into free time."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from ditserve.evaluation.workflow_budget import StudyBudget, BudgetExhausted


class BudgetTests(unittest.TestCase):
    def test_explicit_unlimited_conversion_preserves_ledger_snapshot_and_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [1000.]
            with StudyBudget(directory, 300, clock=lambda: now[0], monotonic=lambda: now[0]) as budget:
                budget.start('old-failed')
                now[0] += 250
                budget.finish('failed', evidence='preserved')
                old = (Path(directory) / 'budget.json').read_bytes()
                authorization = '取消预算，GPU预算无限，继续运行。'
                event = budget.authorize_unlimited(authorization, authorized_unix=1250.)
                snapshot = Path(directory) / event['snapshot']
                self.assertEqual(snapshot.read_bytes(), old)
                self.assertEqual(event['snapshot_sha256'], hashlib.sha256(old).hexdigest())
                self.assertEqual(event['authorization'], authorization)
                self.assertEqual(event['authorized_unix'], 1250.)
                self.assertEqual(event['old_limit_seconds'], 300.)
                self.assertIsNone(event['new_limit_seconds'])
                self.assertIsNone(budget.limit)
                self.assertIsNone(budget.remaining_seconds)
                self.assertEqual(budget.record['attempts'], json.loads(old)['attempts'])
                lease = budget.start('after-conversion', minimum_seconds=100000, timeout_seconds=3600)
                self.assertEqual(lease['stop_at_unix'], 4850.)
                self.assertEqual(lease['deadline_reason'], 'task_timeout')
                self.assertIsNone(lease['remaining_at_start'])
                now[0] += 3700
                budget.finish('timeout')
                self.assertEqual(budget.record['charged_seconds'], 3950.)
                self.assertEqual(budget.record['overrun_seconds'], 0.)
            with StudyBudget(directory, None) as budget:
                self.assertEqual(budget.record['authorization_history'], [event])
                self.assertEqual(len(budget.record['attempts']), 2)
            text = (Path(directory) / 'budget.json').read_text()
            json.loads(text, parse_constant=lambda value: self.fail('nonfinite JSON: ' + value))
            self.assertNotIn('Infinity', text)

    def test_unlimited_resume_never_implicitly_changes_existing_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            with StudyBudget(directory, 300):
                pass
            before = (Path(directory) / 'budget.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'cannot change'):
                with StudyBudget(directory, None):
                    pass
            self.assertEqual((Path(directory) / 'budget.json').read_bytes(), before)
            with StudyBudget(directory, 300) as budget:
                budget.authorize_unlimited('Explicit unlimited authorization')
            with self.assertRaisesRegex(ValueError, 'cannot change'):
                with StudyBudget(directory, 300):
                    pass

    def test_authorization_requires_text_lock_and_recovered_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = StudyBudget(directory, 300)
            with self.assertRaisesRegex(RuntimeError, 'held budget lock'):
                budget.authorize_unlimited('authorized')
            with budget:
                for authorization in ('', '  ', None):
                    with self.assertRaises(ValueError):
                        budget.authorize_unlimited(authorization)
                for timestamp in (float('nan'), float('inf'), -1, True):
                    with self.assertRaises(ValueError):
                        budget.authorize_unlimited('authorized', authorized_unix=timestamp)
                budget.start('still-active')
                with self.assertRaisesRegex(RuntimeError, 'active attempt'):
                    budget.authorize_unlimited('authorized')
                self.assertEqual(budget.limit, 300.)
            with self.assertRaisesRegex(RuntimeError, 'held budget lock'):
                budget.authorize_unlimited('authorized')

    def test_unlimited_crash_recovery_charges_full_elapsed_and_requires_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [1000.]
            kwargs = dict(clock=lambda: now[0], monotonic=lambda: now[0])
            with StudyBudget(directory, None, **kwargs) as budget:
                budget.start('interrupted', timeout_seconds=120)
            now[0] += 5000
            with self.assertRaisesRegex(RuntimeError, 'verified recovery'):
                with StudyBudget(directory, None, **kwargs):
                    pass
            checked = []
            with StudyBudget(directory, None, recovery_check=lambda row: checked.append(row), **kwargs) as budget:
                self.assertEqual(len(checked), 1)
                self.assertEqual(budget.record['charged_seconds'], 5000.)
                self.assertEqual(budget.record['attempts'][0]['status'], 'recovered_uncertain')
                self.assertIsNone(budget.remaining_seconds)
                budget.start('next', timeout_seconds=120)
                budget.finish('completed')

    def test_unlimited_reboot_clock_rollback_charges_bounded_authorized_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [1000.]
            kwargs = dict(clock=lambda: now[0], monotonic=lambda: now[0])
            with StudyBudget(directory, None, **kwargs) as budget:
                budget.start('interrupted', timeout_seconds=120, cleanup_seconds=90)
                budget.record['active']['boot_id'] = 'previous-boot'
                budget.save()
            now[0] = 800.
            with StudyBudget(directory, None, recovery_check=lambda _: None, **kwargs) as budget:
                self.assertEqual(budget.record['charged_seconds'], 210.)
                self.assertIn('clock rollback', budget.record['attempts'][0]['accounting'])
                self.assertIsNone(budget.record['active'])

    def test_malformed_budget_record_releases_lock_without_rewriting_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'budget.json'
            path.write_text('{}\n')
            budget = StudyBudget(directory, None)
            with self.assertRaises(KeyError):
                budget.__enter__()
            self.assertIsNone(budget.handle)
            self.assertEqual(path.read_text(), '{}\n')
            # A second holder reaches the same parse failure, not lock contention.
            with self.assertRaises(KeyError):
                with StudyBudget(directory, None):
                    pass

    def test_nonfinite_limits_and_unbounded_task_timeout_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for limit in (float('inf'), float('nan'), -float('inf')):
                with self.assertRaises(ValueError):
                    StudyBudget(directory, limit)
            with StudyBudget(directory, None) as budget:
                for name in ('timeout_seconds', 'minimum_seconds', 'cleanup_seconds'):
                    for value in (float('inf'), float('nan'), None, -1, True):
                        with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                            budget.start('invalid', **{name: value})
                        self.assertIsNone(budget.record['active'])
                with self.assertRaises(ValueError):
                    budget.start('invalid', timeout_seconds=0)

    def test_resume_and_failed_attempts_charge_elapsed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [1000.]
            clock = lambda: now[0]
            with StudyBudget(directory, 300, clock=clock, monotonic=clock) as budget:
                lease = budget.start('first')
                self.assertEqual(lease['stop_at_unix'], 1210.)
                now[0] += 150
                budget.finish('failed')
            with StudyBudget(directory, 300, clock=clock, monotonic=clock) as budget:
                self.assertEqual(budget.remaining_seconds, 150)
                budget.start('retry')
                now[0] += 70
                budget.finish('completed')
                with self.assertRaises(BudgetExhausted):
                    budget.start('next')
                self.assertEqual(len(budget.record['attempts']), 2)

    def test_crash_recovery_is_explicit_and_conservative(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [1000.]
            clock = lambda: now[0]
            with StudyBudget(directory, 300, clock=clock, monotonic=clock) as budget:
                budget.start('interrupted')
            now[0] += 80
            with StudyBudget(directory, 300, clock=clock, monotonic=clock,
                             recovery_check=lambda active: None) as budget:
                self.assertEqual(budget.remaining_seconds, 220)
                self.assertEqual(budget.record['attempts'][0]['status'], 'recovered_uncertain')
                self.assertIsNone(budget.record['active'])

    def test_second_launcher_and_budget_increase_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with StudyBudget(directory, 300):
                with self.assertRaisesRegex(RuntimeError, 'another launcher'):
                    with StudyBudget(directory, 300):
                        pass
            with self.assertRaisesRegex(ValueError, 'cannot change'):
                with StudyBudget(directory, 600):
                    pass

    def test_complete_block_admission_reserves_minimum_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            with StudyBudget(directory, 300) as budget:
                with self.assertRaises(BudgetExhausted):
                    budget.start('too-large', minimum_seconds=400)
                self.assertIsNone(budget.record['active'])

    def test_real_overrun_is_reported_not_clamped(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [0.]
            clock = lambda: now[0]
            with StudyBudget(directory, 100, clock=clock, monotonic=clock) as budget:
                budget.start('job')
                now[0] = 110.
                budget.finish('cleanup_delayed')
                self.assertEqual(budget.record['overrun_seconds'], 10.)

    def test_wall_clock_rollback_does_not_extend_deadline_or_reduce_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            wall, monotonic = [1000.], [50.]
            with StudyBudget(directory, 300, clock=lambda: wall[0],
                             monotonic=lambda: monotonic[0]) as budget:
                lease = budget.start('clock-adjusted')
                self.assertEqual(lease['stop_at_monotonic'], 260.)
                wall[0], monotonic[0] = 800., 270.
                self.assertLess(wall[0], lease['stop_at_unix'])
                self.assertGreater(monotonic[0], lease['stop_at_monotonic'])
                budget.finish('budget_exhausted')
                self.assertEqual(budget.record['charged_seconds'], 220.)
                self.assertEqual(budget.remaining_seconds, 80.)

    def test_same_boot_recovery_uses_elapsed_monotonic_despite_wall_clock_jump(self):
        for changed_wall in (800., 10000.):
            with self.subTest(changed_wall=changed_wall), tempfile.TemporaryDirectory() as directory:
                wall, monotonic = [1000.], [50.]
                kwargs = dict(clock=lambda: wall[0], monotonic=lambda: monotonic[0])
                with StudyBudget(directory, 300, **kwargs) as budget:
                    budget.start('interrupted')
                wall[0], monotonic[0] = changed_wall, 130.
                with StudyBudget(directory, 300, recovery_check=lambda active: None, **kwargs) as budget:
                    self.assertEqual(budget.record['charged_seconds'], 80.)
                    self.assertEqual(budget.remaining_seconds, 220.)
                    self.assertEqual(budget.record['attempts'][0]['status'], 'recovered_uncertain')

    def test_reboot_with_rollback_cannot_make_interrupted_work_free(self):
        with tempfile.TemporaryDirectory() as directory:
            wall, monotonic = [1000.], [50.]
            kwargs = dict(clock=lambda: wall[0], monotonic=lambda: monotonic[0])
            with StudyBudget(directory, 300, **kwargs) as budget:
                budget.start('interrupted')
                budget.record['active']['boot_id'] = 'previous-boot'
                budget.save()
            wall[0], monotonic[0] = 800., 10.
            with StudyBudget(directory, 300, recovery_check=lambda active: None, **kwargs) as budget:
                self.assertEqual(budget.remaining_seconds, 0.)
                self.assertEqual(budget.record['attempts'][0]['charged_seconds'], 300.)
                with self.assertRaises(BudgetExhausted):
                    budget.start('must-not-run')

    def test_unverified_active_attempt_is_preserved_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            with StudyBudget(directory, 300) as budget:
                budget.start('interrupted')
            path = Path(directory) / 'budget.json'
            before = path.read_bytes()
            for callback in (None, lambda active: (_ for _ in ()).throw(RuntimeError('container still active'))):
                with self.subTest(verified_callback=callback is not None):
                    with self.assertRaises(RuntimeError):
                        with StudyBudget(directory, 300, recovery_check=callback):
                            self.fail('unverified recovery must not enter')
                    self.assertEqual(path.read_bytes(), before)
            with StudyBudget(directory, 300, recovery_check=lambda active: None) as budget:
                self.assertIsNone(budget.record['active'])

    def test_recovery_callback_runs_while_budget_lock_is_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            with StudyBudget(directory, 300) as budget:
                original = budget.start('interrupted')
            def check(active):
                self.assertEqual(active['id'], original['id'])
                with self.assertRaisesRegex(RuntimeError, 'another launcher'):
                    with StudyBudget(directory, 300):
                        self.fail('recovery verification must hold budget lock')
            with StudyBudget(directory, 300, recovery_check=check):
                pass


if __name__ == '__main__':
    unittest.main()
