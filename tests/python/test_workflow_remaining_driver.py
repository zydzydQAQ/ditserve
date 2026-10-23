"""A continuous queue keeps admission/accounting and defers only full reports."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ditserve.evaluation as evaluation
from ditserve.evaluation import workflow_next_study as api

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('remaining_driver', REPO / 'docker/run-next-study.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class RemainingDriverTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        rows = [dict(id=name, logical_id=name, source=str(REPO), source_digest='fixture', family='extension',
                     requires=requires, estimated_seconds=1)
                for name, requires in [('first', []), ('dependent', ['first']), ('independent', [])]]
        study = dict(root=str(root), source=str(REPO), source_digest='fixture', image='fixture', budget_seconds=None,
                     source_revisions=[dict(source=str(REPO))], active_job=None, decisions=[], selections={}, jobs=rows)
        api.atomic_json(root / 'budget.json', dict(charged_seconds=123., active=None))
        return root, study, rows

    def run_queue(self, *, technical=None, launch_status=None, affinity={58}, missing_launch=False,
                  leave_active=False, infrastructure_exit=False, capacity=None, gate_override=None):
        root, study, rows = self.fixture()
        technical = technical or {}
        launch_status = launch_status or {}
        capacity = capacity or {}
        gate_override = gate_override or {}
        launched = []
        def launch(command, **kwargs):
            bundle = api.read(Path(command[-1]))
            name = bundle['jobs'][0]['id']
            launched.append(name)
            status = launch_status.get(name, 'completed')
            if not missing_launch:
                api.atomic_json(root / 'launches' / name / 'launch.json', dict(id=name, status=status))
            if leave_active:
                api.atomic_json(root / 'budget.json', dict(charged_seconds=123., active=dict(job_id=name)))
            return SimpleNamespace(returncode=1 if infrastructure_exit or status != 'completed' else 0)
        def evidence(current, job):
            settled = api.read(root / 'launches' / job['id'] / 'launch.json', {}).get('status') == 'completed'
            return dict(qualified=settled and technical.get(job['id'], True),
                        capacity=settled and capacity.get(job['id'], True))
        def gate(current, job):
            failed = [name for name in job['requires'] if not all(evidence(current, driver.report.current_job(current, name)).values())]
            return dict(passed=gate_override.get(job['id'], not failed), reasons=failed)
        recovery = SimpleNamespace(recover_study=lambda *a, **kw: dict(status='clean'))
        registered = SimpleNamespace(register=lambda current: rows, postprocess=Mock())
        summary = SimpleNamespace(prerequisite_gate=gate,
            summarize=lambda current: dict(qualified=False, scope='descriptive',
                                           postprocess_completed=registered.postprocess.call_count == 1))
        with ExitStack() as stack:
            stack.enter_context(patch.object(evaluation, 'workflow_remaining_study', registered, create=True))
            stack.enter_context(patch.object(evaluation, 'workflow_remaining_report', summary, create=True))
            stack.enter_context(patch.object(driver, 'verify_revision'))
            stack.enter_context(patch.object(driver, 'helper', return_value=recovery))
            stack.enter_context(patch.object(driver.os, 'sched_getaffinity', return_value=affinity))
            stack.enter_context(patch.object(driver.subprocess, 'run', side_effect=launch))
            stack.enter_context(patch.object(driver.report, 'evidence', side_effect=evidence))
            build = stack.enter_context(patch.object(driver.report, 'build_report'))
            archive = stack.enter_context(patch.object(driver, 'archive'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            error = None
            try:
                driver.execute(study, validate_remaining=True)
            except RuntimeError as exc:
                error = exc
        return root, study, rows, launched, build, archive, error

    def test_all_jobs_continuous_with_one_final_report_and_archive(self):
        _, study, rows, launched, build, archive, error = self.run_queue()
        self.assertIsNone(error)
        self.assertEqual(launched, [row['id'] for row in rows])
        build.assert_called_once_with(Path(study['root']), final=True)
        archive.assert_called_once_with(study)
        self.assertEqual(study['status'], 'settled')
        self.assertTrue(study['selections']['remaining_validation']['postprocess_completed'])
        self.assertEqual([r['technical_qualified'] for r in study['selections']['remaining_validation_progress']], [True]*3)
        self.assertTrue(all('decision' not in row for row in rows))

    def test_technical_failure_skips_only_its_dependent(self):
        _, study, rows, launched, build, archive, error = self.run_queue(technical={'first': False})
        self.assertIsNone(error)
        self.assertEqual(launched, ['first', 'independent'])
        self.assertNotIn('decision', rows[1])
        self.assertTrue(any(r['decision'] == 'remaining_prerequisite_not_qualified' for r in study['decisions']))
        self.assertEqual([r['technical_qualified'] for r in study['selections']['remaining_validation_progress']], [False, False, True])
        build.assert_called_once_with(Path(study['root']), final=True)
        archive.assert_called_once()

    def test_performance_selection_is_not_an_admission_gate(self):
        _, study, _, launched, _, archive, error = self.run_queue()
        self.assertIsNone(error)
        self.assertFalse(study['selections']['remaining_validation']['qualified'])
        self.assertEqual(len(launched), 3)
        archive.assert_called_once()

    def test_cleanup_verified_timeout_continues_independent(self):
        _, _, rows, launched, _, archive, error = self.run_queue(launch_status={'first': 'timeout'})
        self.assertIsNone(error)
        self.assertEqual(launched, ['first', 'independent'])
        self.assertNotIn('decision', rows[1])
        archive.assert_called_once()

    def test_infrastructure_failures_stop_queue_without_archive(self):
        for status in driver.STOP_STATUSES | {'budget_exhausted'}:
            with self.subTest(status=status):
                _, study, _, launched, build, archive, error = self.run_queue(launch_status={'first': status})
                self.assertIsNotNone(error)
                self.assertEqual(launched, ['first'])
                self.assertEqual(study['status'], 'interrupted')
                build.assert_not_called()
                archive.assert_not_called()

    def test_prerequisite_capacity_failure_skips_dependent(self):
        _, study, _, launched, _, _, error = self.run_queue(capacity={'first': False})
        self.assertIsNone(error)
        self.assertEqual(launched, ['first', 'independent'])
        progress = study['selections']['remaining_validation_progress']
        self.assertFalse(progress[1]['prerequisite_gate']['passed'])

    def test_explicit_subset_gate_is_not_rejected_again_by_legacy_gate(self):
        _, _, _, launched, _, _, error = self.run_queue(technical={'first': False}, gate_override={'dependent': True})
        self.assertIsNone(error)
        self.assertEqual(launched, ['first', 'dependent', 'independent'])

    def test_unsettled_attempt_or_missing_launch_stops(self):
        for controls in (dict(leave_active=True), dict(missing_launch=True, infrastructure_exit=True)):
            with self.subTest(controls=controls):
                _, _, _, launched, _, archive, error = self.run_queue(**controls)
                self.assertRegex(str(error), 'launch/accounting did not settle')
                self.assertEqual(launched, ['first'])
                archive.assert_not_called()

    def test_nonzero_suite_exit_cannot_reuse_old_completed_evidence(self):
        _, study, _, launched, build, archive, error = self.run_queue(infrastructure_exit=True)
        self.assertRegex(str(error), 'refusing stale evidence')
        self.assertEqual(launched, ['first'])
        self.assertEqual(study['status'], 'interrupted')
        build.assert_not_called()
        archive.assert_not_called()

    def test_wrong_cpu_affinity_stops_before_launch(self):
        _, _, _, launched, _, archive, error = self.run_queue(affinity={0, 58})
        self.assertRegex(str(error), 'CPU 58')
        self.assertEqual(launched, [])
        archive.assert_not_called()

    def test_run_job_default_still_refreshes_full_report(self):
        root, study, rows = self.fixture()
        row = rows[0]
        api.atomic_json(root / 'launches' / row['id'] / 'launch.json', dict(status='completed'))
        with patch.object(driver.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
             patch.object(driver.report, 'evidence', return_value=dict(qualified=True)), \
             patch.object(driver.report, 'build_report') as build, redirect_stdout(io.StringIO()):
            self.assertTrue(driver.run_job(study, row))
        build.assert_called_once_with(root)

    def test_execution_modes_are_exclusive(self):
        for flag in ('retry', 'only', 'diagnose_sp', 'optimize_followup', 'accept_sp2', 'optimize_runtime',
                     'optimize_image_cache', 'optimize_encoding', 'optimize_dispatch_refresh',
                     'optimize_occupancy', 'optimize_cache_combination'):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'cannot be combined'):
                driver.execute({}, validate_remaining=True, **{flag: True})

    def test_cli_rejects_missing_run_and_combined_modes(self):
        for arguments in (['--validate-remaining'], ['--run', '--validate-remaining', '--only', 'first'],
                          ['--run', '--validate-remaining', '--optimize-cache-combination']):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                driver.main(arguments)
            self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
