"""Controller/report revisions retain immutable measurement provenance."""
import importlib.util
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ditserve.evaluation.workflow_serving_study import binding, digest, freeze_analysis

_path = Path(__file__).resolve().parents[2] / 'docker/run-serving-study.py'
_spec = importlib.util.spec_from_file_location('serving_controller', _path)
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)


class ControllerRevisionTests(unittest.TestCase):
    def test_archive_runs_after_driver_lock_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'docker').mkdir()
            (root / 'docker/recover-serving-study.py').write_text(
                "def recover_study(*args, **kwargs):\n    return {'status': 'idle'}\n")
            jobs = [dict(id=identifier, family='calibration_quality') for identifier in
                    ('smoke-dual_cpu', 'smoke-dedicated_fp32', 'calibration', 'quality')]
            study = dict(root=str(root), source=str(root), source_digest='original',
                         jobs=jobs, decisions=[], budget_seconds=100)
            def archive_after_unlock(value, *, keep_raw):
                self.assertTrue(keep_raw)
                self.assertEqual(value['status'], 'settled')
                with (root / 'driver.lock').open('a+') as other:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(other, fcntl.LOCK_UN)
            with mock.patch.object(driver, 'controller_source', return_value=str(root)), \
                 mock.patch.object(driver, 'run_job', return_value=True), \
                 mock.patch.object(driver, 'result', return_value=None), \
                 mock.patch.object(driver, 'run_blocks'), \
                 mock.patch.object(driver, 'micro_alternative', return_value=None), \
                 mock.patch.object(driver, 'preview'), \
                 mock.patch.object(driver.study_api, 'calibration_from_result', return_value={}), \
                 mock.patch.object(driver.study_api, 'online_jobs'), \
                 mock.patch('ditserve.evaluation.workflow_serving_report.select_extension', return_value=None), \
                 mock.patch.object(driver, 'archive', side_effect=archive_after_unlock) as archive:
                driver.execute(study, keep_raw=True)
            archive.assert_called_once_with(study, keep_raw=True)

    def test_revision_preserves_measurement_source_and_detects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            repo = Path(directory) / 'repo'
            for name in ('workers', 'configs', 'docker', 'tests/python'):
                (repo / name).mkdir(parents=True)
                (repo / name / 'example.py').write_text('old = 1\n')
            study = dict(root=str(root), source=str(root / 'sources/frozen'),
                         source_digest='original', source_files=[])
            revision = freeze_analysis(repo, study)
            self.assertEqual(study['source_digest'], 'original')
            self.assertEqual(revision['measurement_source'], study['source'])
            driver.verify_revision(revision)
            driver.CONTROLLER_REVISION = revision
            self.assertEqual(driver.controller_source(study), revision['source'])
            (Path(revision['source']) / 'workers/example.py').write_text('changed = 1\n')
            with self.assertRaisesRegex(RuntimeError, 'binding changed'):
                driver.controller_source(study)

    def test_running_controller_rejects_pointer_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            driver.CONTROLLER_REVISION = None
            (root / 'analysis-source.json').write_text(json.dumps({'source': 'different'}))
            with self.assertRaisesRegex(RuntimeError, 'pointer changed'):
                driver.controller_source(dict(root=str(root), source='original'))

    def tearDown(self):
        driver.CONTROLLER_REVISION = None


if __name__ == '__main__':
    unittest.main()
