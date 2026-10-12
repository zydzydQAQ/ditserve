"""Run02 CPU integration contracts; synthetic arrays and worker/torch doubles only.

These tests exercise orchestration and evidence accounting, not model numerics.
They never load model weights, inspect real experiment arrays, or launch CUDA.
"""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from ditserve.experiments import previs2_experiment as experiment
from ditserve.experiments import previs2_storage as storage
from ditserve.experiments import previs_storage as resolver
from ditserve.gpu import quality as previs_quality
from ditserve.gpu import worker as previs_worker
from ditserve.experiments.run_previs2 import EXPECTED_PACKAGES, fingerprint

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / 'configs/previs2.json').read_text())
SOURCE = 'a' * 64
MIB = 1024 ** 2


class Tensor:
    def __init__(self, values=None):
        self.values = np.zeros((1, 2), dtype=np.float32) if values is None else values
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return Tensor(self.values.copy())
    def numpy(self): return self.values.copy()


def fake_torch():
    return types.SimpleNamespace(set_num_threads=mock.Mock(),
        equal=lambda a, b: np.array_equal(a.values, b.values),
        cuda=types.SimpleNamespace(synchronize=mock.Mock(), reset_peak_memory_stats=mock.Mock(),
            max_memory_allocated=lambda: 200 * MIB, max_memory_reserved=lambda: 256 * MIB,
            get_device_name=lambda device: 'CPU fixture device', empty_cache=mock.Mock()))


def video():
    return np.zeros((1, 1, 2, 2, 3), dtype=np.float32)


class Worker:
    """Logical step clock; no numerical simulation of the diffusion model."""
    def __init__(self, clock):
        self.clock = clock
        self.prepared, self.released, self.step_calls = [], [], []
        self.fail_step = None
        self.prepare_delay, self.step_delay = .001, .010
        self.states = []
    def _memory(self):
        return dict(device_total_bytes=8 * 1024 * MIB, driver_free_bytes=4 * 1024 * MIB,
                    allocated_bytes=2 * 1024 * MIB, reserved_bytes=3 * 1024 * MIB)
    def _inventory(self, state):
        return dict(cuda_bytes=32 * MIB, storage_bytes=96 * MIB, total_bytes=96 * MIB)
    def prepare(self, spec):
        self.prepared.append(copy.deepcopy(spec))
        self.clock[0] += self.prepare_delay
        state = types.SimpleNamespace(spec=dict(spec), request_id=spec['request_id'], step_index=0,
            latents=Tensor(), residency='gpu', events=[dict(kind='prepare', ms=1.)])
        self.states.append(state)
        return state
    def step(self, state):
        self.clock[0] += self.step_delay
        self.step_calls.append((state.request_id, state.step_index))
        if self.fail_step is not None:
            failure = self.fail_step(state)
            if failure is not None:
                raise failure
        state.events.append(dict(kind='step', step_index=state.step_index, ms=self.step_delay * 1000,
            path='compute', finite=True, reused_forwards=0,
            threshold=state.spec['threshold'], max_continuous=state.spec['max_continuous']))
        state.step_index += 1
        return state.events[-1]
    def finish(self, state):
        self.clock[0] += .001
        state.events.append(dict(kind='decode', ms=1.))
        return video()
    def release(self, state):
        self.released.append(state.request_id)
    def state_summary(self, state):
        return dict(step_index=state.step_index)
    def snapshot(self, state, mode):
        self.clock[0] += .001
        event = dict(kind='snapshot', mode=mode, ms=1., state_bytes={'cuda_bytes': 32 * MIB},
                     memory_before={'allocated_bytes': 64 * MIB},
                     memory_after={'allocated_bytes': (64 if mode == 'keep' else 32) * MIB})
        state.events.append(event)
        state.residency = 'gpu' if mode == 'keep' else 'cpu'
        return event
    def restore(self, state):
        self.clock[0] += .001
        last = next(e for e in reversed(state.events) if e['kind'] == 'snapshot')
        event = dict(kind='restore', mode=last['mode'], ms=1.)
        state.residency = 'gpu'
        state.events.append(event)
        return event


class Fixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='previs2-experiment-test-')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.suite = self.base / 'suite'
        self.stage = self.suite / '00-stage'
        self.ssd = self.base / 'ssd'
        self.config = copy.deepcopy(CONFIG)
        self.config['raw_storage_root'] = str(self.ssd)
        self.job = dict(config=self.config, output=str(self.stage), suite_root=str(self.suite),
            allowed_seconds=1000, stage='online', device='a100', uuid=self.config['devices']['a100'],
            source_identity={'sha256': SOURCE, 'source_hashes': {'fixture.py': 'b' * 64}},
            protocol_sha256=fingerprint(self.config))
        self.clock = [100.]
        self.exp = experiment.Experiment(self.job)
        self.exp.started = self.clock[0]
        self.exp.deadline = self.clock[0] + 1000
        self.worker = Worker(self.clock)
        self.exp.worker = self.worker
        self.torch = fake_torch()
        for patch in (mock.patch.object(experiment.time, 'monotonic', side_effect=lambda: self.clock[0]),
                      mock.patch.dict(sys.modules, {'torch': self.torch})):
            patch.start()
            self.addCleanup(patch.stop)

    def record_rows(self):
        p = self.stage / 'records.jsonl'
        return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []

    def controllers(self):
        initial = dict(config=dict(experiment.FULL), predicted_trajectory_lpips=0.,
                       remaining_by_completed_step_ms={str(i): 100. for i in range(33)})
        recovery = dict(status='unavailable', actions=[], prefix_config=dict(experiment.FULL),
                        device='a100', shape=self.config['shapes']['long'])
        controller = dict(initial=initial, recovery=recovery, fixed_state={'mode': 'hot'})
        return {'short': copy.deepcopy(controller), 'long': copy.deepcopy(controller)}

    def trace(self, arrivals=(0, 5, 50)):
        return [dict(request_id='r%d' % i, kind='long', prompt_id='fixture%d' % i, seed=201 + i,
            prompt='fixture %d' % i, arrival_ms=float(arrival), deadline_ms=1000. + arrival,
            reference={'path': str(self.base / 'unused-reference.npy')},
            initial_cache_config=dict(experiment.FULL)) for i, arrival in enumerate(arrivals)]


class ArrivalAndFailureTests(Fixture):
    def test_terminal_arrivals_include_inside_failed_step_and_boundary(self):
        trace = self.trace((0, 5, 10, 10.001))
        result = experiment.arrival_accounting(trace, 10, ['r0'])
        self.assertEqual(result, dict(offered=3, scheduled=4, completed=1,
            offered_unfinished=['r1', 'r2'], not_yet_arrived=['r3']))

    def test_failed_first_step_preserves_offered_denominator_and_unarrived(self):
        self.worker.step_delay = .025
        self.worker.fail_step = lambda state: RuntimeError('fixture failed step')
        trace = self.trace((0, 12, 40))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'fixture failed step'):
            self.exp.run_trace(trace, 'joint_state_qc', 0, self.controllers(), {'long': 64 * MIB, 'short': 64 * MIB})
        summary = self.record_rows()[-1]
        self.assertEqual(summary['type'], 'online_summary')
        self.assertEqual(summary['status'], 'partial')
        self.assertEqual(summary['offered'], 2)
        self.assertEqual(summary['arrival_accounting']['offered_unfinished'], ['r0', 'r1'])
        self.assertEqual([row['status'] for row in summary['requests']], ['aborted', 'aborted', 'not_arrived'])
        self.assertEqual(len(self.worker.released), 1)
        self.assertAlmostEqual(summary['horizon_ms'], 26.)
        self.assertEqual(summary['source_identity_sha256'], SOURCE)
        self.assertEqual(summary['protocol_sha256'], self.job['protocol_sha256'])

    def test_budget_stop_in_trace_is_partial_and_outer_outcome_not_completed(self):
        self.worker.step_delay = .025
        self.worker.fail_step = lambda state: experiment.BudgetStop('fixture budget')
        self.exp.setup = lambda: None
        self.exp.online = lambda: self.exp.run_trace(self.trace((0, 12, 40)), 'fixed_state_qc', 2,
            self.controllers(), {'long': 64 * MIB, 'short': 64 * MIB})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.exp.run(), 2)
        outcome = json.loads((self.stage / 'outcome.json').read_text())
        rows = self.record_rows()
        self.assertEqual(outcome['status'], 'budget_exhausted')
        self.assertEqual(rows[-1]['type'], 'unfinished')
        summary = next(r for r in rows if r['type'] == 'online_summary')
        self.assertEqual(summary['status'], 'partial')
        self.assertEqual(summary['offered'], 2)
        self.assertEqual(len(self.worker.released), 1)

    def test_generate_releases_state_on_step_failure(self):
        self.worker.fail_step = lambda state: RuntimeError('fixture failure')
        spec = self.exp.spec('fixture-generate')
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            self.exp.generate(spec)
        self.assertEqual(self.worker.released, ['fixture-generate'])

    def test_generate_applies_full_compute_suffix_only_after_restore(self):
        spec = self.exp.spec('fixture-suffix', cache=True, threshold=.08)
        result = self.exp.generate(spec, pause=16, mode='cold', resume_config=dict(experiment.FULL))
        steps = [e for e in result['events'] if e['kind'] == 'step']
        self.assertEqual([e['step_index'] for e in steps], list(range(32)))
        self.assertEqual([e['threshold'] for e in steps], [.08] * 16 + [0.] * 16)
        self.assertTrue(all(e['max_continuous'] == 1 for e in steps))
        self.assertEqual([e['mode'] for e in result['switches']], ['cold', 'cold'])
        self.assertEqual(self.worker.released, ['fixture-suffix'])

    def test_failed_prepare_never_reports_completed_output(self):
        self.worker.prepare = mock.Mock(side_effect=RuntimeError('prepare failed'))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'prepare failed'):
            self.exp.run_trace(self.trace(), 'fixed_state_qc', 0, self.controllers(), {'long': 64 * MIB, 'short': 64 * MIB})
        summary = self.record_rows()[-1]
        self.assertEqual(summary['status'], 'partial')
        self.assertEqual(summary['arrival_accounting']['completed'], 0)
        self.assertFalse(self.worker.released)


class ProtocolAndCorrectnessTests(Fixture):
    def test_spec_uses_new_seeds_m1_and_uncached_zero_threshold(self):
        self.assertEqual(self.exp.spec('full')['max_continuous'], 1)
        self.assertEqual(self.exp.spec('full')['threshold'], 0)
        self.assertEqual(self.exp.spec('cached', cache=True)['seed'], 101)
        self.assertEqual(self.exp.spec('cached', cache=True)['threshold'], .08)
        with self.assertRaisesRegex(ValueError, 'm=1'):
            self.exp.spec('wrong', max_continuous=3)

    def test_correctness_shape_copy_keeps_original_job_and_environment_protocol(self):
        original = copy.deepcopy(self.config)
        with mock.patch.object(previs_worker, 'WanStepWorker', return_value=self.worker), \
             mock.patch.object(experiment.os, 'sched_getaffinity', return_value={0}), \
             mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': self.job['uuid']}), \
             mock.patch.object(experiment.importlib.metadata, 'version', side_effect=lambda name: EXPECTED_PACKAGES[name]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.exp.setup()
        observed = []
        def inherited(exp):
            observed.append(copy.deepcopy(exp.config['shapes']))
            self.assertEqual(exp.spec('cache', cache=True)['max_continuous'], 1)
        with mock.patch.object(experiment.LegacyExperiment, 'correctness', autospec=True, side_effect=inherited):
            self.exp.correctness()
        self.assertEqual(observed[0]['long'], self.config['correctness_shape'])
        self.assertEqual(observed[0]['short']['frames'], 9)
        self.assertEqual(self.job['config'], original)
        environment = self.record_rows()[0]
        self.assertEqual(environment['protocol'], original)
        self.assertEqual(self.exp.records[0]['protocol'], original)
        self.assertIsNot(self.exp.config, self.job['config'])

    def test_actual_inherited_eight_correctness_cases_request_m1(self):
        seen = []
        def run(spec, pause=None, mode='keep', **kwargs):
            seen.append(copy.deepcopy(spec))
            events = [dict(kind='step', step_index=i, path='compute') for i in range(spec['steps'])]
            snapshot = dict(kind='snapshot', mode=mode, ms=1., memory_before={'allocated_bytes': 64},
                            memory_after={'allocated_bytes': 32 if mode != 'keep' else 64}, state_bytes={'cuda_bytes': 32})
            return dict(output=video(), latents=Tensor(), events=events, elapsed_ms=10.,
                        switches=[] if pause is None else [snapshot, dict(kind='restore', mode=mode, ms=1.)])
        self.worker.reference = lambda spec: run(spec)
        self.exp.generate = run
        self.exp.save_array = lambda case, array, label='output': dict(path='fixture-' + case, shape=list(array.shape), dtype=str(array.dtype))
        self.exp._cold_policy_interleave_correctness = mock.Mock()
        with contextlib.redirect_stdout(io.StringIO()):
            self.exp.correctness()
        self.assertTrue(all(spec['max_continuous'] == 1 for spec in seen))
        self.assertTrue(all(spec['steps'] == 32 and spec['height'] == 256 and spec['frames'] == 17 for spec in seen))
        self.assertEqual(len(self.exp.records), 8)
        self.assertTrue(all(row['status'] == 'completed' for row in self.exp.records))
        self.exp._cold_policy_interleave_correctness.assert_called_once_with()

    def test_cold_ninth_uses_same_cold_prefix_full_suffix_trajectory(self):
        calls = []
        def run(spec, **kwargs):
            calls.append((copy.deepcopy(spec), copy.deepcopy(kwargs)))
            events = [dict(kind='step', step_index=i, path='compute', finite=True, max_continuous=1,
                           threshold=.08 if i < 16 else 0., reused_forwards=0) for i in range(32)]
            return dict(output=video(), latents=Tensor(), elapsed_ms=10., events=events,
                switches=[dict(kind='snapshot', mode='cold', ms=1., memory_before={'allocated_bytes': 64},
                    memory_after={'allocated_bytes': 32}, state_bytes={'cuda_bytes': 32}), dict(kind='restore', mode='cold', ms=1.)])
        self.exp.generate = run
        self.exp.save_array = lambda case, array, label='output': {'path': 'fixture-' + case}
        with contextlib.redirect_stdout(io.StringIO()):
            self.exp._cold_policy_interleave_correctness()
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(spec['cache'] and spec['threshold'] == .08 and spec['max_continuous'] == 1 for spec, _ in calls))
        self.assertTrue(all(kw['pause'] == 16 and kw['mode'] == 'cold' and kw['resume_config'] == experiment.FULL for _, kw in calls))
        self.assertNotIn('interleave', calls[0][1])
        self.assertIs(calls[1][1]['interleave'], True)
        row = self.record_rows()[-1]
        self.assertTrue(row['trajectory_verified'])
        self.assertEqual(row['baseline_scope'], 'same_cold_prefix_suffix_trajectory')
        self.assertEqual(row['suffix_config'], experiment.FULL)

    def test_adopt_shape_selects_largest_safe_current_source_and_keeps_protocol(self):
        rows = [dict(type='shape_probe', status='completed', device='a100', safe_to_run=safe,
                     shape=dict(self.config['shapes']['long'], frames=frames)) for frames, safe in ((17, True), (33, True), (49, False))]
        self.exp.previous_records = lambda: rows
        before = copy.deepcopy(self.job['config'])
        selected = self.exp.adopt_shape()
        self.assertEqual(selected['shape']['frames'], 33)
        self.assertEqual(self.exp.config['shapes']['long']['frames'], 33)
        self.assertEqual(self.job['config'], before)
        with self.assertRaisesRegex(RuntimeError, 'no safe'):
            experiment.choose_shape([dict(r, safe_to_run=False) for r in rows])

    def test_previous_records_reject_other_protocol_and_source(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.exp.record({'type': 'fixture', 'status': 'completed', 'case_id': 'matching'})
        with (self.stage / 'records.jsonl').open('a') as handle:
            for identity, protocol in ((SOURCE, 'x' * 64), ('x' * 64, self.job['protocol_sha256'])):
                handle.write(json.dumps(dict(type='fixture', case_id='wrong', source_identity_sha256=identity,
                                            protocol_sha256=protocol)) + '\n')
        self.assertEqual([r['case_id'] for r in self.exp.previous_records()], ['matching'])


class CapacityHookTests(Fixture):
    def test_capacity_uses_cuda_storage_and_incremental_peak_not_total_state(self):
        state = types.SimpleNamespace(spec=self.exp.spec('capacity'), step_index=8)
        with mock.patch.object(experiment, 'capacity_feasibility', return_value={'fixture': True}) as function:
            self.assertEqual(self.exp.capacity(state, 128 * MIB), {'fixture': True})
        args = function.call_args.kwargs
        self.assertEqual(args['victim_storage_bytes'], 32 * MIB)
        self.assertNotEqual(args['victim_storage_bytes'], 96 * MIB)
        self.assertEqual(args['next_peak_bytes'], 128 * MIB)
        self.assertIsNone(args['logical_budget_bytes'])
        self.assertIsNone(args['logical_live_bytes'])

    def test_capacity_natural_release_does_not_claim_driver_free_increase(self):
        state = types.SimpleNamespace(spec=self.exp.spec('capacity'), step_index=8)
        capacity = self.exp.capacity(state, 64 * MIB)
        self.assertIs(capacity['diagnostics']['artificial_logical_pressure'], False)
        self.assertIsNone(capacity['modes']['hot']['expected_driver_free_after_release_bytes'])
        self.assertEqual(capacity['modes']['hot']['expected_allocator_reusable_release_bytes'], 32 * MIB)
        self.assertTrue(capacity['modes']['hot']['requires_runtime_capacity_check'])


class RegisteredStorageTests(Fixture):
    def setUp(self):
        super().setUp()
        (self.stage / 'records.jsonl').write_text('{"type":"fixture"}\n')

    def save(self, case='fixture', array=None, root=None):
        return storage.save_array(self.suite, self.stage, case, video() if array is None else array,
                                  storage_root=self.ssd if root is None else root)

    def test_registered_ssd_array_resolves_and_both_hash_semantics_match(self):
        array = np.arange(24, dtype=np.float32).reshape(1, 2, 2, 2, 3) / 24
        descriptor = self.save(array=array)
        path = Path(descriptor['path'])
        self.assertTrue(path.is_symlink())
        resolved = resolver.resolve_candidates([path], self.suite)
        self.assertEqual(resolved.path, self.ssd / '00-stage/arrays/fixture-output.npy')
        index = resolver.load_index(self.suite)
        entry = index['entries']['00-stage/arrays/fixture-output.npy']
        self.assertEqual(entry['file_sha256'], hashlib.sha256(resolved.path.read_bytes()).hexdigest())
        self.assertEqual(descriptor['sha256'], hashlib.sha256(array.tobytes(order='C')).hexdigest())
        self.assertNotEqual(entry['file_sha256'], descriptor['sha256'])
        self.assertTrue(np.array_equal(np.load(resolved.path, allow_pickle=False), array))

    def test_append_only_rejects_existing_case_without_overwrite_or_index_change(self):
        descriptor = self.save()
        index = (self.suite / 'artifact-storage.json').read_bytes()
        data = Path(descriptor['path']).read_bytes()
        with self.assertRaises(FileExistsError):
            self.save(array=np.ones_like(video()))
        self.assertEqual((self.suite / 'artifact-storage.json').read_bytes(), index)
        self.assertEqual(Path(descriptor['path']).read_bytes(), data)

    def test_preexisting_physical_file_not_overwritten_or_registered(self):
        path = self.ssd / '00-stage/arrays/fixture-output.npy'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'preexisting data')
        with self.assertRaises(FileExistsError):
            self.save()
        self.assertEqual(path.read_bytes(), b'preexisting data')
        self.assertFalse((self.suite / 'artifact-storage.json').exists())

    def test_mutated_payload_is_rejected_by_registered_file_hash(self):
        descriptor = self.save()
        physical = Path(descriptor['path']).resolve()
        raw = bytearray(physical.read_bytes())
        raw[-1] ^= 1
        physical.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'sha256'):
            resolver.resolve_candidates([Path(descriptor['path'])], self.suite)

    def test_unregistered_link_and_direct_external_path_are_not_authorized(self):
        descriptor = self.save()
        logical = Path(descriptor['path'])
        alias = logical.with_name('alias.npy')
        alias.symlink_to(logical.resolve())
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            resolver.resolve_candidates([alias], self.suite)
        with self.assertRaisesRegex(ValueError, 'outside'):
            resolver.resolve_candidates([logical.resolve()], self.suite)

    def test_external_root_ancestor_of_suite_rejected_before_array_write(self):
        with self.assertRaises(ValueError):
            self.save(root=self.base)
        self.assertFalse((self.base / '00-stage/arrays/fixture-output.npy').exists())

    def test_external_stage_symlink_escape_rejected_before_array_write(self):
        self.ssd.mkdir()
        outside = self.base / 'unrelated'
        outside.mkdir()
        (self.ssd / '00-stage').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.save()
        self.assertFalse((outside / 'arrays/fixture-output.npy').exists())

    def test_failed_index_publish_preserves_unique_data_without_overwrite(self):
        with mock.patch.object(storage, 'atomic_json', side_effect=OSError('fixture index write failed')):
            with self.assertRaisesRegex(OSError, 'index write failed'):
                self.save()
        physical = self.ssd / '00-stage/arrays/fixture-output.npy'
        self.assertTrue(physical.is_file())
        data = physical.read_bytes()
        self.assertFalse((self.stage / 'arrays/fixture-output.npy').exists())
        with self.assertRaises(FileExistsError):
            self.save()
        self.assertEqual(physical.read_bytes(), data)

class PersistenceAndScoringTests(Fixture):
    def setUp(self):
        super().setUp()
        # One toy solver update; completion/scoring ordering is the contract.
        self.exp.config = copy.deepcopy(self.exp.config)
        self.exp.config['shapes']['long']['steps'] = 1
        self.trace_one = self.trace((0,))

    def invoke(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.exp.run_trace(self.trace_one, 'fixed_state_qc', 0,
                self.controllers(), {'long': 64 * MIB, 'short': 64 * MIB})

    def test_array_write_failure_still_records_completed_generation_and_partial_trace(self):
        self.exp.save_array = mock.Mock(side_effect=OSError('fixture disk full'))
        with self.assertRaisesRegex(OSError, 'disk full'):
            self.invoke()
        summary = self.record_rows()[-1]
        self.assertEqual(summary['status'], 'partial')
        self.assertEqual(summary['arrival_accounting']['completed'], 1)
        self.assertEqual(summary['requests'][0]['status'], 'completed')
        self.assertEqual(summary['requests'][0]['quality']['status'], 'unknown')
        self.assertIn('disk full', summary['requests'][0]['artifact_error'])
        self.assertNotIn('_output', summary['requests'][0])
        self.assertNotIn('_latent', summary['requests'][0])

    def test_scoring_failure_cannot_erase_completed_trace_denominator(self):
        self.exp.save_array = lambda *args: {'path': 'fixture-array'}
        with mock.patch.object(np, 'load', return_value=video()), \
             mock.patch.object(previs_quality, 'evaluate_quality', side_effect=RuntimeError('fixture metric failure')):
            with self.assertRaisesRegex(RuntimeError, 'metric failure'):
                self.invoke()
        summary = self.record_rows()[-1]
        self.assertEqual(summary['offered'], 1)
        self.assertEqual(summary['status'], 'partial')
        self.assertEqual(summary['requests'][0]['quality']['reason'], 'artifact_or_quality_failure')

    def test_budget_before_scoring_keeps_unknown_quality_not_zero(self):
        self.exp.save_array = lambda *args: {'path': 'fixture-array'}
        def budget(estimate=5):
            if self.worker.released:
                raise experiment.BudgetStop('no scoring budget')
        self.exp.check_budget = budget
        with mock.patch.object(previs_quality, 'evaluate_quality') as metric:
            self.invoke()
        metric.assert_not_called()
        summary = self.record_rows()[-1]
        self.assertEqual(summary['arrival_accounting']['completed'], 1)
        self.assertEqual(summary['requests'][0]['quality'],
                         dict(status='unknown', reason='budget_ended_before_scoring'))
        self.assertNotIn('lpips_mean', summary['requests'][0]['quality'])

    def test_scoring_wall_time_excluded_from_arrival_horizon(self):
        self.exp.save_array = lambda *args: {'path': 'fixture-array'}
        def score(reference, candidate):
            self.clock[0] += 99
            return dict(status='ok', finite=True, range_valid=True, lpips_mean=.01)
        with mock.patch.object(np, 'load', return_value=video()), mock.patch.object(previs_quality, 'evaluate_quality', side_effect=score):
            self.invoke()
        summary = self.record_rows()[-1]
        self.assertAlmostEqual(summary['horizon_ms'], 12.)
        self.assertAlmostEqual(summary['requests'][0]['e2e_ms'], 12.)
        self.assertGreater(self.clock[0] - self.exp.started, 99)


class MechanismOrderingTests(Fixture):
    def test_sync_snapshot_finishes_before_urgent_and_response_includes_copy(self):
        order = []
        original_snapshot, original_restore = self.worker.snapshot, self.worker.restore
        def snapshot(state, mode):
            result = original_snapshot(state, mode)
            order.append('snapshot_complete')
            return result
        def restore(state):
            order.append('restore')
            return original_restore(state)
        self.worker.snapshot, self.worker.restore = snapshot, restore
        def urgent(spec):
            self.assertEqual(order[-1], 'snapshot_complete')
            order.append('urgent_complete')
            self.clock[0] += .100
            return dict(output=video(), elapsed_ms=100.)
        self.exp.generate = urgent
        self.exp.save_array = lambda *args: {'path': 'fixture-array'}
        action = dict(mode='hot', suffix='continue', config=dict(experiment.FULL), estimated_total_ms=243., predicted_trajectory_lpips=0.)
        self.exp.enumerate = lambda *args: {'actions': [action], 'excluded': []}
        (self.suite / 'short-profile.json').write_text(json.dumps({'service_ms': 100.}))
        with mock.patch.object(np, 'load', return_value=video()), \
             mock.patch.object(previs_quality, 'evaluate_quality', return_value=dict(status='ok', lpips_mean=.01)), \
             contextlib.redirect_stdout(io.StringIO()):
            row = self.exp.mechanism_case('mechanism-fixture', self.config['prompts'][0], dict(experiment.FULL),
                8, 'hot', self.controllers()['long'], 64 * MIB, {'path': 'fixture-reference'}, 0)
        self.assertEqual(order, ['snapshot_complete', 'urgent_complete', 'restore'])
        self.assertTrue(row['capacity_reusable_verified'])
        self.assertTrue(row['next_request_allocation_and_completion_verified'])
        self.assertAlmostEqual(row['urgent_e2e_ms'], 101.)
        self.assertAlmostEqual(row['urgent_service_ms'], 100.)
        self.assertAlmostEqual(row['resume_critical_ms'], 243.)
        self.assertEqual(self.worker.released, ['mechanism-fixture'])

    def test_committed_suffix_prediction_and_remaining_curve_do_not_revert_to_initial(self):
        state = types.SimpleNamespace(spec=self.exp.spec('suffix'), step_index=16)
        controller = self.controllers()['long']
        controller['initial']['config'] = dict(cache=True, threshold=.08, max_continuous=1)
        controller['initial']['predicted_trajectory_lpips'] = .01
        committed = dict(predicted_trajectory_lpips=.04, remaining_by_completed_step_ms={'16': 9.})
        with mock.patch.object(experiment, 'enumerate_recovery_actions', return_value={'actions': []}) as function:
            self.exp.enumerate(state, controller, {}, changes=1, committed_action=committed)
        request = function.call_args.args[0]
        self.assertEqual(request['predicted_trajectory_lpips'], .04)
        self.assertEqual(request['remaining_ms'], 9.)
        self.assertEqual(request['policy_change_count'], 1)
        self.assertEqual(request['lossy_choices_used'], 1)
        self.assertEqual(request['selected_cache_config'], experiment.FULL)

class AllocationCheckTests(Fixture):
    def test_actual_headroom_includes_reusable_reserved_bytes_and_margin(self):
        # 4 GiB driver free + 1 GiB reusable reservation - 512 MiB margin.
        available = (4 * 1024 + 1024 - 512) * MIB
        exact = self.exp.allocation_check(available)
        self.assertTrue(exact['feasible'])
        self.assertEqual(exact['available_bytes'], available)
        self.assertEqual(exact['margin_bytes'], 512 * MIB)
        self.assertEqual(exact['required_incremental_bytes'], available)
        self.assertFalse(self.exp.allocation_check(available + 1)['feasible'])

    def test_resident_state_subtracted_once_and_large_device_margin_applied(self):
        check = self.exp.allocation_check(128 * MIB, already_resident=32 * MIB)
        self.assertEqual(check['required_incremental_bytes'], 96 * MIB)
        self.assertEqual(self.exp.allocation_check(16 * MIB, already_resident=32 * MIB)['required_incremental_bytes'], 0)
        memory = self.worker._memory()
        memory['device_total_bytes'] = 40 * 1024 * MIB
        with mock.patch.object(self.worker, '_memory', return_value=memory):
            self.assertEqual(self.exp.allocation_check(1)['margin_bytes'], 2 * 1024 * MIB)

    def test_actual_memory_is_reread_each_call_not_cached_from_planning(self):
        good = self.worker._memory()
        poor = dict(good, driver_free_bytes=0, reserved_bytes=good['allocated_bytes'])
        with mock.patch.object(self.worker, '_memory', side_effect=[good, poor]) as memory:
            self.assertTrue(self.exp.allocation_check(64 * MIB)['feasible'])
            self.assertFalse(self.exp.allocation_check(64 * MIB)['feasible'])
        self.assertEqual(memory.call_count, 2)

    def test_mechanism_post_snapshot_check_prevents_urgent_allocation(self):
        order = []
        original_snapshot = self.worker.snapshot
        def snapshot(state, mode):
            result = original_snapshot(state, mode)
            order.append('snapshot_complete')
            return result
        def check(peak, already_resident=0):
            self.assertEqual(order, ['snapshot_complete'])
            order.append('actual_headroom_check')
            return dict(feasible=False, reason='actual_headroom_insufficient')
        self.worker.snapshot = snapshot
        self.exp.allocation_check = check
        self.exp.generate = mock.Mock()
        action = dict(mode='hot', suffix='continue', config=dict(experiment.FULL),
                      estimated_total_ms=10., predicted_trajectory_lpips=0.)
        self.exp.enumerate = lambda *args: {'actions': [action], 'excluded': []}
        with contextlib.redirect_stdout(io.StringIO()):
            row = self.exp.mechanism_case('post-snapshot-reject', self.config['prompts'][0], dict(experiment.FULL),
                8, 'hot', self.controllers()['long'], 64 * MIB, {'path': 'unused'}, 0)
        self.assertEqual(row['status'], 'infeasible')
        self.assertEqual(row['reason'], 'post_snapshot_headroom_insufficient')
        self.assertEqual(order, ['snapshot_complete', 'actual_headroom_check'])
        self.exp.generate.assert_not_called()
        self.assertEqual(self.worker.released, ['post-snapshot-reject'])
        self.assertNotIn('quality', row)

    def test_nested_generate_restores_active_case_before_victim_suffix_failure(self):
        def fail(state):
            if state.request_id == 'victim' and state.step_index == 16:
                return experiment.BudgetStop('victim suffix budget')
            return None
        self.worker.fail_step = fail
        with self.assertRaisesRegex(experiment.BudgetStop, 'victim suffix'):
            self.exp.generate(self.exp.spec('victim'), pause=16, mode='hot', interleave=True)
        self.assertEqual(self.exp.active_case, 'victim')
        self.assertEqual(self.worker.released, ['victim-interloper', 'victim'])


class ResourceRejectionTests(Fixture):
    def setUp(self):
        super().setUp()
        self.exp.config = copy.deepcopy(self.exp.config)
        self.exp.config['shapes']['long']['steps'] = 1
        self.exp.config['shapes']['short']['steps'] = 1
        self.exp.save_array = mock.Mock(return_value={'path': 'synthetic-array'})

    def invoke(self, trace, choices):
        self.exp.allocation_check = mock.Mock(side_effect=[dict(feasible=value,
            reason='actual_headroom_sufficient' if value else 'actual_headroom_insufficient') for value in choices])
        with mock.patch.object(np, 'load', return_value=video()), \
             mock.patch.object(previs_quality, 'evaluate_quality', return_value=dict(status='ok', lpips_mean=.01)) as metric, \
             contextlib.redirect_stdout(io.StringIO()):
            self.exp.run_trace(trace, 'joint_state_qc', 0, self.controllers(), {'long': 64 * MIB, 'short': 64 * MIB})
        summary = self.record_rows()[-1]
        return summary, metric.call_count

    def test_rejected_arrival_is_uncompleted_unscored_then_next_request_runs(self):
        summary, metric_calls = self.invoke(self.trace((0, 0)), [False, True])
        self.assertEqual(summary['status'], 'completed')  # Trace processing completed, not both generations.
        self.assertEqual(summary['arrival_accounting']['completed'], 1)
        self.assertEqual(summary['arrival_accounting']['resource_rejected'], 1)
        self.assertEqual(summary['offered'], 2)
        self.assertEqual(summary['arrival_accounting']['offered_unfinished'], ['r0'])
        rows = {r['request_id']: r for r in summary['requests']}
        self.assertEqual(rows['r0']['status'], 'resource_rejected')
        self.assertFalse(rows['r0']['deadline_met'])
        self.assertEqual(rows['r0']['quality'], dict(status='unknown', reason='actual_capacity_insufficient'))
        self.assertNotIn('output', rows['r0'])
        self.assertNotIn('lpips_mean', rows['r0']['quality'])
        self.assertEqual(rows['r1']['status'], 'completed')
        self.assertEqual(metric_calls, 1)
        self.assertEqual(self.exp.save_array.call_count, 2)  # Only completed output and latent.
        self.assertEqual(len(self.worker.prepared), 1)
        self.assertTrue(self.worker.prepared[0]['request_id'].endswith('-r1'))

    def test_all_rejected_requests_never_prepare_score_or_save(self):
        summary, metric_calls = self.invoke(self.trace((0, 0)), [False, False])
        self.assertEqual(summary['arrival_accounting']['completed'], 0)
        self.assertEqual(summary['arrival_accounting']['resource_rejected'], 2)
        self.assertEqual(summary['offered'], 2)
        self.assertEqual(metric_calls, 0)
        self.assertFalse(self.worker.prepared)
        self.exp.save_array.assert_not_called()
        self.assertTrue(all(row['status'] == 'resource_rejected' for row in summary['requests']))

    def keep_action(self):
        action = dict(mode='keep', suffix='continue', config=dict(experiment.FULL),
                      estimated_total_ms=10., predicted_trajectory_lpips=0., consumes_recovery_choice=False)
        self.exp.enumerate = lambda *args: {'actions': [action], 'excluded': []}

    def test_rejected_urgent_after_snapshot_allows_victim_to_resume(self):
        self.exp.config['shapes']['long']['steps'] = 2
        trace = self.trace((0, 5))
        trace[1]['deadline_ms'] = 100.
        self.keep_action()
        with mock.patch.object(self.worker, 'restore', wraps=self.worker.restore) as restore:
            summary, metric_calls = self.invoke(trace, [True, False, True])
        rows = {r['request_id']: r for r in summary['requests']}
        self.assertEqual(rows['r0']['status'], 'completed')
        self.assertEqual(rows['r1']['status'], 'resource_rejected')
        self.assertEqual(len(self.worker.prepared), 1)
        self.assertEqual(restore.call_count, 1)
        self.assertEqual(self.exp.allocation_check.call_args_list, [
            mock.call(64 * MIB, 0), mock.call(64 * MIB, 0), mock.call(64 * MIB, 32 * MIB)])
        self.assertEqual(summary['arrival_accounting']['completed'], 1)
        self.assertEqual(summary['arrival_accounting']['resource_rejected'], 1)
        self.assertEqual(metric_calls, 1)

    def test_rejected_previously_resident_victim_releases_state_without_restore(self):
        self.exp.config['shapes']['long']['steps'] = 2
        trace = self.trace((0, 5))
        trace[1].update(kind='short', deadline_ms=100.)
        self.keep_action()
        with mock.patch.object(self.worker, 'restore', wraps=self.worker.restore) as restore:
            summary, metric_calls = self.invoke(trace, [True, True, False])
        rows = {r['request_id']: r for r in summary['requests']}
        self.assertEqual(rows['r0']['status'], 'resource_rejected')
        self.assertEqual(rows['r1']['status'], 'completed')
        restore.assert_not_called()
        self.assertEqual(len(self.worker.released), 2)
        self.assertEqual(len(set(self.worker.released)), 2)
        self.assertEqual(self.exp.allocation_check.call_args_list[-1], mock.call(64 * MIB, 32 * MIB))
        self.assertEqual(metric_calls, 1)
        self.assertEqual(summary['arrival_accounting']['completed'], 1)
        self.assertEqual(summary['arrival_accounting']['resource_rejected'], 1)


if __name__ == '__main__':
    unittest.main()
