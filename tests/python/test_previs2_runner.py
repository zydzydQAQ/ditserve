"""CPU supervisor tests: all worker processes/GPU queries are test doubles."""
import contextlib
import copy
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from ditserve.experiments import run_previs2 as runner

CONFIG = json.loads((runner.CONFIGS / 'previs2.json').read_text())


def identity():
    value = dict(worker_sha256='a' * 64, source_hashes={'gpu/worker.py': 'a' * 64,
        'previs2_experiment.py': 'b' * 64, 'previs2_policy.py': 'c' * 64, 'run_previs2.py': 'd' * 64},
        packages=runner.EXPECTED_PACKAGES, entrypoint_sha256='e' * 64)
    value['sha256'] = runner.fingerprint(value)
    return value


def correct_rows():
    names = ['official-none', 'official-cache', 'cold-policy-interleave-16']
    names.extend('%s-%s-%s-%d' % (cache, mode, inter, progress)
                 for cache in ('none', 'cache') for mode, inter, progress in
                 (('keep', 'pause', 25), ('hot', 'pause', 50), ('hot', 'interleave', 75)))
    return [dict(type='correctness', case_id=name, status='completed', output_exact_match=True,
                 latent_exact_match=True, path_exact_match=True, trajectory_verified=True,
                 memory_release_verified=True) for name in names]


class RunnerFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='previs2-cpu-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.suite = self.root / 'results/previs/run02'
        self.old = self.root / 'results/previs/run01'
        self.old.mkdir(parents=True)
        for name in ('manifest.json', 'delivery-manifest.json'):
            (self.old / name).write_text('{"fixture": true}\n')
        self.config = copy.deepcopy(CONFIG)
        self.config_path = self.root / 'protocol.json'
        self.config_path.write_text(json.dumps(self.config))
        self.identity = identity()
        self.clock = [100.0]
        self.calls, self.commands, self.behaviors = [], [], {}
        for patch in (
            mock.patch.object(runner, 'ROOT', self.root),
            mock.patch.object(runner, 'source_identity', return_value=self.identity),
            mock.patch.object(runner, 'archive_sources', return_value='fixture-archive'),
            mock.patch.object(runner, 'gpu_idle', return_value={'name': 'CPU double'}),
            mock.patch.object(runner, 'wait_gpu_release', side_effect=self.cleanup),
            mock.patch.object(runner.time, 'monotonic', side_effect=lambda: self.clock[0]),
            mock.patch.object(runner.subprocess, 'Popen', side_effect=self.spawn)):
            patch.start()
            self.addCleanup(patch.stop)

    def cleanup(self, uuid, timeout):
        self.clock[0] += 2
        return {'compute_pids': [], 'memory_used_mib': 0}

    def env(self, device):
        return dict(type='environment', status='completed', versions=self.identity['packages'],
                    source_hashes=self.identity['source_hashes'], uuid=self.config['devices'][device], protocol=self.config)

    def rows(self, job):
        result = [self.env(job['device'])]
        if job['stage'] == 'correctness':
            result += correct_rows()
        if job['stage'] == 'probe':
            result += [dict(type='shape_probe', status='completed', safe_to_run=True, frames=job['frame_probe'])]
        return result

    def spawn(self, command, **kwargs):
        self.commands.append(command)
        job = json.loads(Path(command[-1]).read_text())
        self.calls.append(job)
        behavior = self.behaviors.get(runner.job_key(job), self.behaviors.get(job['stage'], {}))
        self.write_rows(Path(job['output']), behavior.get('rows', self.rows(job)))
        (Path(job['output']) / 'outcome.json').write_text(json.dumps({'status': behavior.get('status', 'completed')}))
        if job['stage'] == 'mechanism':
            self.write_gate()
        if behavior.get('mutate_old'):
            (self.old / 'manifest.json').write_text('{"changed":true}')
        outer = self
        class Process:
            pid = 987654
            waits = 0
            running = True
            def wait(self, timeout=None):
                self.waits += 1
                if self.waits == 1:
                    outer.clock[0] += behavior.get('elapsed', 7)
                    if behavior.get('timeout'):
                        raise subprocess.TimeoutExpired(command, timeout)
                else:
                    outer.clock[0] += behavior.get('stop_elapsed', 3)
                self.running = False
                return behavior.get('returncode', 0)
            def poll(self):
                return None if self.running else 0
        return Process()

    def write_rows(self, output, rows):
        output.mkdir(exist_ok=True, parents=True)
        (output / 'records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')

    def prior(self, stage, device='a100', elapsed=10, status='completed', frame=None):
        item = dict(stage=stage, device=device)
        if frame is not None:
            item['frame_probe'] = frame
        output = self.suite / ('prior-%s-%s-%s' % (stage, device, frame))
        self.write_rows(output, self.rows(item))
        return dict(item, output=str(output), status=status, elapsed_seconds=elapsed, charge_closed=True,
                    source_identity=self.identity)

    def manifest(self, records=(), limit=43200):
        self.suite.mkdir(parents=True, exist_ok=True)
        value = dict(schema_version=runner.SCHEMA, protocol=self.config, protocol_sha256=runner.fingerprint(self.config),
                     records=list(records), used_seconds=runner.charged_seconds(records), limit_seconds=limit,
                     current_source_identity=self.identity, run01_protection=runner.protected_run01())
        (self.suite / 'manifest.json').write_text(json.dumps(value))
        return value

    def gates(self):
        return [self.prior('correctness', device) for device in ('a100', 'v100')]

    def write_gate(self, **changes):
        gate = dict(schema_version='previs2-gate.v1', status='passed', source_identity_sha256=self.identity['sha256'],
                    protocol_sha256=runner.fingerprint(self.config), natural_resource_gate=True, mechanism_gate=True, reasons=[])
        gate.update(changes)
        (self.suite / 'gate.json').write_text(json.dumps(gate))

    def run_main(self, stage='correctness', gpu='both', extra=()):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.main(['--config', str(self.config_path), '--output', str(self.suite),
                                '--stage', stage, '--gpu', gpu] + list(extra))

    def saved(self):
        return json.loads((self.suite / 'manifest.json').read_text())


class BudgetAndProcessTests(RunnerFixture):
    def test_all_jobs_serial_budget_and_no_completed_repeats(self):
        self.assertEqual(self.run_main('all'), 0)
        self.assertEqual(len(self.calls), 9)
        m = self.saved()
        self.assertEqual(m['status'], 'completed')
        self.assertEqual(m['used_seconds'], 81)
        self.assertEqual(self.run_main('all'), 0)
        self.assertEqual(len(self.calls), 9)
        self.assertEqual([x.get('frame_probe') for x in self.calls if x['stage'] == 'probe'], [17, 33, 49])
        self.assertTrue(all(cmd[1:4] == ['-m', 'ditserve.experiments.previs2_experiment', '--job'] for cmd in self.commands))

    def test_completed_selection_does_not_claim_full_suite(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.saved()['status'], 'partial')
        self.assertEqual(self.saved()['selection_status'], 'completed')

    def test_active_manifest_running_before_each_launch(self):
        original = self.spawn
        states = []
        def observe(*args, **kwargs):
            value = self.saved()
            states.append((value['status'], value['records'][-1]['status']))
            return original(*args, **kwargs)
        with mock.patch.object(runner.subprocess, 'Popen', side_effect=observe):
            self.run_main()
        self.assertEqual(states, [('running', 'running'), ('running', 'running')])

    def test_timeout_cleanup_fully_charged_only_owned_group_signalled(self):
        self.behaviors['correctness'] = dict(timeout=True, elapsed=65, stop_elapsed=4)
        with mock.patch.object(runner.os, 'killpg') as kill:
            self.assertEqual(self.run_main(gpu='a100', extra=['--job-seconds', '130']), 2)
        kill.assert_called_once_with(987654, runner.signal.SIGTERM)
        m = self.saved()
        self.assertEqual(m['used_seconds'], 71)
        self.assertEqual(m['records'][0]['status'], 'budget_exhausted')
        self.assertTrue(m['records'][0]['charge_closed'])
        self.assertEqual(self.calls[0]['allowed_seconds'], 65)

    def test_failed_probe_does_not_block_next_independent_shape(self):
        self.manifest(self.gates())
        self.behaviors[('probe', 'a100', 33)] = dict(status='failed', returncode=2)
        self.assertEqual(self.run_main('probe', 'a100'), 2)
        self.assertEqual([j['frame_probe'] for j in self.calls], [17, 33, 49])
        self.assertEqual(self.saved()['used_seconds'], 47)
        self.assertEqual(self.saved()['records'][-2]['status'], 'failed')

    def test_unused_early_budget_rolls_forward(self):
        self.manifest(self.gates())
        self.run_main('probe', 'a100')
        self.assertEqual(self.calls[0]['allowed_seconds'], 5400 - 20 - 65)
        self.run_main('calibration', 'a100')
        self.assertEqual(self.calls[-1]['allowed_seconds'], 21600 - 47 - 65)

    def test_limit_cannot_increase_or_reset_across_calls(self):
        self.manifest(self.gates(), limit=100)
        self.assertEqual(self.run_main('probe', 'a100'), 2)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.saved()['limit_seconds'], 100)
        self.assertEqual(self.saved()['used_seconds'], 20)

    def test_failure_needs_explicit_retry_without_erasing_charge(self):
        self.manifest([self.prior('correctness', status='failed', elapsed=13)])
        self.assertEqual(self.run_main(gpu='a100'), 2)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.run_main(gpu='a100', extra=['--retry-failed']), 0)
        self.assertEqual(self.saved()['used_seconds'], 22)
        self.assertEqual(self.saved()['records'][0]['status'], 'failed')

    def test_charge_corruption_and_open_charge_reject_resume(self):
        value = self.manifest(self.gates())
        value['used_seconds'] += 1
        (self.suite / 'manifest.json').write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.run_main()
        value['used_seconds'] -= 1
        value['records'][0]['charge_closed'] = False
        (self.suite / 'manifest.json').write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            self.run_main()
        self.assertEqual(self.calls, [])

    def test_cleanup_failure_retains_unclosed_charge_and_blocks_resume(self):
        with mock.patch.object(runner, 'wait_gpu_release', side_effect=RuntimeError('fixture cleanup')):
            with self.assertRaisesRegex(RuntimeError, 'cleanup unconfirmed'):
                self.run_main(gpu='a100')
        record = self.saved()['records'][0]
        self.assertEqual(record['status'], 'cleanup_unconfirmed')
        self.assertFalse(record['charge_closed'])
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            self.run_main()

    def test_real_overshoot_never_clamped_or_called_success(self):
        self.manifest([], limit=100)
        self.behaviors['correctness'] = dict(elapsed=102)
        self.assertEqual(self.run_main(gpu='a100'), 2)
        self.assertEqual(self.saved()['used_seconds'], 104)
        self.assertEqual(self.saved()['status'], 'budget_exceeded')
        with self.assertRaisesRegex(ValueError, 'overshoot'):
            self.run_main()

    def test_double_supervisor_lock_rejected(self):
        self.suite.mkdir(parents=True)
        with (self.suite / 'suite.lock').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_main()
        self.assertEqual(self.calls, [])

    def test_busy_gpu_never_signalled(self):
        with mock.patch.object(runner, 'gpu_idle', side_effect=RuntimeError('occupied')), mock.patch.object(runner.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                self.run_main()
        kill.assert_not_called()
        self.assertEqual(self.calls, [])


class GuardTests(RunnerFixture):
    def test_both_correctness_gates_required(self):
        self.manifest([self.prior('correctness')])
        self.assertEqual(self.run_main('probe', 'a100'), 2)
        self.assertEqual(self.calls, [])
        self.assertTrue(all('both_current_source' in r['reason'] for r in self.saved()['records'][1:]))

    def test_missing_or_false_ninth_trajectory_fails(self):
        rows = [self.env('a100')] + correct_rows()
        rows[3]['trajectory_verified'] = False
        self.behaviors['correctness'] = dict(rows=rows)
        self.assertEqual(self.run_main(gpu='a100'), 2)
        self.assertEqual(self.saved()['records'][0]['status'], 'failed')

    def test_false_source_environment_not_accepted(self):
        rows = [self.env('a100')] + correct_rows()
        rows[0]['source_hashes'] = {}
        self.behaviors['correctness'] = dict(rows=rows)
        self.assertEqual(self.run_main(gpu='a100'), 2)
        self.assertEqual(self.saved()['records'][0]['status'], 'failed')

    def test_online_gate_binds_boolean_gates_and_both_hashes(self):
        self.manifest(self.gates())
        for change in ({'natural_resource_gate': False}, {'mechanism_gate': False}, {'source_identity_sha256': 'f' * 64},
                       {'protocol_sha256': 'f' * 64}, {'status': 'failed'}, {'natural_resource_gate': 1}):
            self.write_gate(**change)
            self.assertEqual(self.run_main('online', 'a100'), 2)
        self.assertEqual(self.calls, [])
        self.write_gate()
        self.assertEqual(self.run_main('online', 'a100'), 0)
        self.assertIn('online_gate', self.calls[-1])

    def test_no_safe_probe_blocks_calibration(self):
        self.manifest(self.gates())
        self.assertEqual(self.run_main('calibration', 'a100'), 2)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.saved()['records'][-1]['reason'], 'no_current_source_safe_shape_probe')

    def test_run01_protection_checked_on_resume_and_after_job(self):
        self.manifest(self.gates())
        (self.old / 'delivery-manifest.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'protected run01'):
            self.run_main()
        self.assertEqual(self.calls, [])

    def test_worker_old_suite_mutation_stops_next_launch(self):
        self.behaviors['correctness'] = dict(mutate_old=True)
        with self.assertRaisesRegex(ValueError, 'protected run01'):
            self.run_main()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.saved()['status'], 'partial')
        self.assertIn('supervisor_error', self.saved())

    def test_changed_source_cannot_mix_or_reset(self):
        self.manifest(self.gates())
        new_identity = copy.deepcopy(self.identity)
        new_identity['source_hashes']['new.py'] = 'f' * 64
        with mock.patch.object(runner, 'source_identity', return_value=new_identity):
            with self.assertRaisesRegex(ValueError, 'cannot mix'):
                self.run_main()
        self.assertEqual(self.calls, [])

    def test_dry_run_creates_nothing_queries_nothing(self):
        prior_torch = sys.modules.get('torch')
        with mock.patch.object(runner, 'source_identity') as source, mock.patch.object(runner, 'gpu_idle') as gpu:
            self.assertEqual(self.run_main('all', extra=['--dry-run']), 0)
        source.assert_not_called()
        gpu.assert_not_called()
        self.assertFalse(self.suite.exists())
        self.assertIs(sys.modules.get('torch'), prior_torch)

    def test_disjoint_output_and_budget_ceiling(self):
        for extra in (['--gpu-hours', '12.1'], ['--gpu-hours', 'nan'], ['--output', str(self.old / 'bad')]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_main(extra=extra)
        self.assertFalse(self.suite.exists())

    def test_venv_python_symlink_is_not_resolved(self):
        python = self.root / 'python'
        python.symlink_to(sys.executable)
        self.run_main(gpu='a100', extra=['--python', str(python)])
        self.assertEqual(self.commands[0][0], str(python))


class PureTests(unittest.TestCase):
    def test_protocol_new_prompts_fixed_shapes_and_disjoint_seeds(self):
        runner.validate_config(CONFIG)
        self.assertEqual(CONFIG['shapes']['short'], dict(height=480, width=832, frames=9, steps=32))
        self.assertEqual(CONFIG['dev_seeds'], [101, 102])
        self.assertEqual(CONFIG['test_seeds'], [201, 202])
        self.assertEqual(CONFIG['pause_steps'], [8, 16, 24])
        self.assertEqual(CONFIG['requests_per_trace'], 8)

    def test_all_status_charges_including_failures_and_invalid_rejected(self):
        self.assertEqual(runner.charged_seconds([dict(status=s, elapsed_seconds=2) for s in
                         ('completed', 'failed', 'interrupted', 'partial', 'budget_exhausted')]), 10)
        for value in (-1, True, float('nan'), float('inf'), None):
            with self.assertRaises(ValueError):
                runner.charged_seconds([dict(status='failed', elapsed_seconds=value)])

    def test_nvidia_query_strict_pid_and_uuid_filters(self):
        uuid = CONFIG['devices']['a100']
        with mock.patch.object(runner.subprocess, 'check_output', side_effect=[
                uuid + ', A100, 40536, 0, 0\n', uuid + ', 123\n' + CONFIG['devices']['v100'] + ', 999\n']):
            state = runner.gpu_state(uuid)
        self.assertEqual(state['compute_pids'], [123])
        with mock.patch.object(runner.subprocess, 'check_output', side_effect=[uuid + ', A100, 40536, N/A, 0\n']):
            with self.assertRaises(ValueError):
                runner.gpu_state(uuid)

    def test_source_identity_self_digest_and_fixed_versions(self):
        self.assertTrue(runner.validate_source_identity(identity()))
        changed = identity()
        changed['packages'] = dict(changed['packages'], torch='other')
        self.assertFalse(runner.validate_source_identity(changed))

class ArchiveAndTerminationTests(unittest.TestCase):
    def test_copy_archive_binds_module_entrypoint_and_protocol(self):
        with tempfile.TemporaryDirectory(prefix='previs2-archive-test-') as temporary:
            root = Path(temporary)
            package = root / 'python/ditserve/experiments'
            gpu = root / 'python/ditserve/gpu'
            package.mkdir(parents=True)
            gpu.mkdir(parents=True)
            suite = root / 'suite'
            suite.mkdir()
            modules = ('previs2_experiment.py', 'previs2_policy.py', 'run_previs2.py')
            for name in modules:
                (package / name).write_text('# fixture ' + name + '\n')
            (gpu / 'worker.py').write_text('# fixture worker\n')
            with mock.patch.object(runner, '__file__', str(package / 'run_previs2.py')), mock.patch.object(runner, 'ROOT', root):
                hashes = runner.source_hashes()
                ident = dict(worker_sha256=hashes['gpu/worker.py'], source_hashes=hashes,
                             entrypoint_sha256=runner.file_sha(package / 'run_previs2.py'), packages=runner.EXPECTED_PACKAGES)
                ident['sha256'] = runner.fingerprint(ident)
                archive = Path(runner.archive_sources(suite, ident, CONFIG))
                for name in modules:
                    self.assertEqual((archive / name).read_bytes(), (package / name).read_bytes())
                self.assertEqual((archive / 'gpu/worker.py').read_bytes(), (gpu / 'worker.py').read_bytes())
                descriptor = json.loads((archive / 'archive.json').read_text())
                self.assertEqual(descriptor['protocol_sha256'], runner.fingerprint(CONFIG))
                self.assertEqual(runner.archive_sources(suite, ident, CONFIG), str(archive))
                (archive / 'gpu/worker.py').write_text('# corrupt')
                with self.assertRaisesRegex(ValueError, 'archive differs'):
                    runner.archive_sources(suite, ident, CONFIG)
                (package / 'later.py').write_text('# new frozen dependency')
                self.assertNotEqual(runner.source_hashes(), ident['source_hashes'])

    def test_missing_new_source_refuses_freeze(self):
        with tempfile.TemporaryDirectory(prefix='previs2-source-test-') as directory:
            path = Path(directory) / 'run_previs2.py'
            path.write_text('# fixture')
            with mock.patch.object(runner, '__file__', str(path)), self.assertRaisesRegex(ValueError, 'finish all'):
                runner.source_hashes()

    def test_term_timeout_escalates_only_owned_child_group(self):
        child = mock.Mock()
        child.pid = 543210
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('fixture', 10), 0]
        with mock.patch.object(runner.os, 'killpg') as kill:
            runner.stop_owned(child)
        self.assertEqual(kill.call_args_list, [mock.call(543210, runner.signal.SIGTERM), mock.call(543210, runner.signal.SIGKILL)])
        self.assertEqual(child.wait.call_args_list, [mock.call(timeout=10), mock.call(timeout=20)])

    def test_dead_child_is_never_signalled(self):
        child = mock.Mock()
        child.poll.return_value = 0
        with mock.patch.object(runner.os, 'killpg') as kill:
            runner.stop_owned(child)
        kill.assert_not_called()

    def test_budget_roundoff_tolerance_does_not_hide_real_excess(self):
        manifest = dict(limit_seconds=10., used_seconds=10. + 1e-8,
                        records=[dict(status='completed', elapsed_seconds=10. + 1e-8)])
        self.assertGreater(runner.validate_ledger(manifest), 10)
        manifest['used_seconds'] = manifest['records'][0]['elapsed_seconds'] = 10.01
        with self.assertRaisesRegex(ValueError, 'overshoot'):
            runner.validate_ledger(manifest)


if __name__ == '__main__':
    unittest.main()
