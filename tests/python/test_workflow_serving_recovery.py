"""Recovery must prove exclusive container cleanup before charging stale leases."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ditserve.evaluation.workflow_budget import StudyBudget

SPEC = importlib.util.spec_from_file_location('serving_recovery',
    Path(__file__).resolve().parents[2] / 'docker/recover-serving-study.py')
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)
CONTAINER_ID = 'a' * 64
CONTAINER_NAME = 'ditserve-workflow-smoke-dual_cpu-0123456789ab'


def response(code=0, stdout='', stderr=''):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.study = dict(root=str(self.root), source=str(self.root / 'sources/frozen'), budget_seconds=300)
        (self.root / 'study.json').write_text(json.dumps(self.study))
        with StudyBudget(self.root, 300) as budget:
            self.active = budget.start('smoke-dual_cpu')
        self.budget_path = self.root / 'budget.json'
        self.before = self.budget_path.read_bytes()
        self.ledger = dict(id='smoke-dual_cpu', budget_attempt_id=self.active['id'],
            container=CONTAINER_NAME, gpu_monitor=dict(container_id=CONTAINER_ID))
        self.ledger_path = self.root / 'launches/smoke-dual_cpu/launch.json'
        self.ledger_path.parent.mkdir(parents=True)
        self.save_ledger()

    def save_ledger(self):
        self.ledger_path.write_text(json.dumps(self.ledger))

    def test_absent_container_recovery_charges_without_stopping_other_processes(self):
        commands = []
        def execute(arguments):
            commands.append(arguments)
            return response(1, stderr='Error: No such container: owned')
        result = recovery.recover_study(self.root, processes=lambda: [], execute=execute)
        self.assertEqual(result['status'], 'recovered')
        self.assertTrue(result['verified_stopped'])
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][0], 'inspect')
        budget = json.loads(self.budget_path.read_text())
        self.assertIsNone(budget['active'])
        self.assertEqual(budget['attempts'][0]['status'], 'recovered_uncertain')
        self.assertTrue(list((self.root / 'recoveries').glob('*.json')))

    def test_running_container_stop_failure_requires_kill_and_inspection(self):
        commands = []
        results = iter([response(stdout=CONTAINER_ID+' true'), response(1, stderr='stop failed'),
                        response(stdout=CONTAINER_ID+' true'), response(),
                        response(stdout=CONTAINER_ID+' false')])
        def execute(arguments):
            commands.append(arguments)
            return next(results)
        result = recovery.recover_study(self.root, processes=lambda: [], execute=execute)
        self.assertEqual(result['status'], 'recovered')
        self.assertEqual([row[0] for row in commands], ['inspect', 'stop', 'inspect', 'kill', 'inspect'])
        self.assertEqual(commands[1][-1], CONTAINER_ID)
        self.assertEqual(commands[3][-1], CONTAINER_ID)

    def test_failed_cleanup_preserves_active_lease(self):
        def execute(arguments):
            return response(stdout=CONTAINER_ID+' true') if arguments[0] == 'inspect' else response(1)
        result = recovery.recover_study(self.root, processes=lambda: [], execute=execute)
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('remains running', result['error'])
        self.assertEqual(self.budget_path.read_bytes(), self.before)

    def test_reused_container_name_never_gets_signaled(self):
        commands = []
        def execute(arguments):
            commands.append(arguments)
            return response(stdout='b'*64+' true')
        result = recovery.recover_study(self.root, processes=lambda: [], execute=execute)
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual([row[0] for row in commands], ['inspect'])
        self.assertEqual(self.budget_path.read_bytes(), self.before)

    def test_live_previous_suite_or_driver_blocks_before_docker(self):
        for script, arguments in [('run-workflow-suite.py', ['--jobs', str(self.root/'jobs/smoke.json')]),
                                  ('run-serving-study.py', ['--root', str(self.root)])]:
            with self.subTest(script=script):
                rows = [dict(pid=os.getpid()+100000, start_ticks=1,
                    command=['python', str(Path(self.study['source'])/'docker'/script), *arguments])]
                result = recovery.recover_study(self.root, processes=lambda: rows,
                    execute=lambda _: self.fail('live study must block before Docker'))
                self.assertEqual(result['status'], 'blocked')
                self.assertIn('still running', result['error'])
                self.assertEqual(self.budget_path.read_bytes(), self.before)

    def test_live_container_create_client_blocks_absent_container_recovery(self):
        rows = [dict(pid=os.getpid()+100000, start_ticks=1,
                     command=['docker', 'run', '--name', CONTAINER_NAME])]
        result = recovery.recover_study(self.root, processes=lambda: rows,
            execute=lambda _: response(1, stderr='no such container'))
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('live launcher', result['error'])
        self.assertEqual(self.budget_path.read_bytes(), self.before)

    def test_wrong_attempt_or_unsafe_container_cannot_be_recovered(self):
        for key, value in [('budget_attempt_id', 'different'), ('container', 'some-other-user-container')]:
            with self.subTest(key=key):
                previous = self.ledger[key]
                self.ledger[key] = value
                self.save_ledger()
                result = recovery.recover_study(self.root, processes=lambda: [],
                    execute=lambda _: self.fail('invalid ownership must block before Docker'))
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(self.budget_path.read_bytes(), self.before)
                self.ledger[key] = previous

    def test_existing_budget_owner_prevents_recovery(self):
        # Simulate a live launcher holding the same budget lock without consuming
        # its lease; an advisory lock is stronger than a momentary /proc snapshot.
        import fcntl
        with (self.root/'budget.lock').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = recovery.recover_study(self.root, processes=lambda: [],
                execute=lambda _: self.fail('owned budget must block before Docker'))
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('another launcher', result['error'])
        self.assertEqual(self.budget_path.read_bytes(), self.before)


if __name__ == '__main__':
    unittest.main()
