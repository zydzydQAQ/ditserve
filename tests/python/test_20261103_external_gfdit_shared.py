"""Original GF-DiT control-plane tests; model tests require the actual GPU CLI.

The small executor below is only a unit fixture for original scheduling events.
It is never selectable by the production CLI or reported as model execution.
"""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import external_gfdit_shared as port

SOURCE = Path(__file__).resolve().parents[3] / '.local/ditserve-kairos-rewrite-runtime/external-systems/20261103/GF-DiT'
MANIFEST = SOURCE.with_suffix('.source.json')
MANIFEST_SHA = 'e1b54cf09241ff70fc27e9d1666a44cbe38e00eaf1713b57cd5c9ade9c9db3f4'


class UnitExecutor:
    def __init__(self, control, fail_stage=None):
        self.p, self.fail_stage = control.protocol, fail_stage

    def execute(self, task, inputs):
        if task.kind.value == self.fail_stage:
            raise ValueError('explicit unit-executor failure')
        value = next(iter(inputs.values()))
        if task.kind == self.p.TaskKind.TEXT_ENCODE:
            value = dict(cursor=0, total=value.image['steps'], kinds=[])
        else:
            value = deepcopy(value)
        value['kinds'].append(task.kind.value)
        if task.step_range is not None:
            if task.step_range.start != value['cursor']:
                raise AssertionError('original scheduler repeated/skipped cursor')
            value['cursor'] = task.step_range.end
        return (self.p.ArtifactValue(task.outputs[0], value),)


class GFSharedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_dir():
            raise unittest.SkipTest('pinned official GF-DiT source required; no fake source fallback')
        cls.directory = TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.derived = port.apply(SOURCE, MANIFEST, MANIFEST_SHA, cls.root / 'control')
        cls.control = port.load_control(cls.derived['root'], cls.derived['manifest_sha256'])

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == port.NAMESPACE or name.startswith(port.NAMESPACE + '.'):
                del sys.modules[name]
        cls.directory.cleanup()

    def request(self, chunk=2, identity='unit-request'):
        return port.make_request(identity, port.round_adapter.ImageRequest('A fox.'), chunk_size=chunk)

    def test_real_official_algorithms_have_only_import_changes(self):
        manifest = self.derived['manifest']
        self.assertEqual(len(manifest['imports_relocated']), len(port.CONTROL_FILES))
        for target, row in manifest['imports_relocated'].items():
            original = (SOURCE / row['upstream_path']).read_bytes()
            content, algorithm_sha = port.relocate_source(original)
            self.assertEqual(content, (Path(self.derived['root']) / target).read_bytes())
            self.assertEqual(algorithm_sha, row['algorithm_ast_sha256'])
            self.assertEqual(port.sha(original), row['upstream_sha256'])
        self.assertFalse(manifest['migration_implemented'])
        self.assertFalse(manifest['native_vllm_worker_implemented'])
        self.assertFalse(manifest['qualification_eligible'])
        with self.assertRaisesRegex(ValueError, 'execution dependency'):
            port.relocate_source(b'from vllm.distributed import get_world_group\n')
        with self.assertRaisesRegex(ValueError, 'unexpected upstream logging'):
            port.relocate_source(b'from vllm.logger import other_logger\n')
        with self.assertRaisesRegex(ValueError, 'real worker dependency'):
            port.relocate_source(b'from vllm_omni.diffusion.runtime_v2.multiproc_worker import MultiprocWorkerPool\n')

    def test_relocated_official_worker_death_message_still_fails_the_scheduler(self):
        scheduler, worker, _ = port.build_runtime(self.control, UnitExecutor(self.control))
        message_module = port.importlib.import_module(port.NAMESPACE + '.transport_protocol')
        message = message_module.WorkerReadyMessage(0, status='failed', message='real transport death signal')
        with self.assertRaisesRegex(RuntimeError, 'real transport death signal'):
            scheduler._handle_worker_event(message)
        self.assertFalse(worker.trace)

    def test_original_graph_full_denominator_and_request_validation(self):
        for chunk in (1, 2, 5, 28):
            summary = port.graph_summary(self.control, self.request(chunk))
            self.assertEqual(summary['original_steps'], 28)
            self.assertEqual(summary['task_count'], 5 + (28 + chunk - 1) // chunk)
            self.assertEqual(sum(end-start for start,end in summary['intervals']), 28)
        for bad in (True, 0, -1, 29, 2.0, '2'):
            with self.assertRaises(ValueError): self.request(bad)
        for mutation in ('arrival', 'parallel', 'identity', 'deadline'):
            value = self.request()
            if mutation == 'arrival': value['runtime_v2_arrival_ms'] = 1.
            if mutation == 'parallel': value['sp'] = 2
            if mutation == 'identity': value['request_id'] = '../unsafe'
            if mutation == 'deadline': value['runtime_v2_deadline_ms'] = float('nan')
            with self.assertRaises(ValueError): port.SD35Compiler(self.control).compile_request(value)

    def test_original_scheduler_edf_cost_dag_events_and_artifact_eviction(self):
        scheduler, worker, cost = port.build_runtime(self.control, UnitExecutor(self.control))
        self.assertEqual(type(scheduler).__module__, port.NAMESPACE + '.scheduler')
        self.assertEqual(type(scheduler.policy).__name__, 'EdfBestFitSchedulerPolicy')
        self.assertEqual(cost.available_sp_sizes(), (1,))
        scheduler.start()
        try:
            output = scheduler.execute(self.request(), timeout_s=2.)
            self.assertEqual(output['cursor'], 28)
            self.assertEqual(len(output['kinds']), 19)
            self.assertEqual(len(worker.trace), 19)
            self.assertTrue(all(row['status'] == 'completed' for row in worker.trace))
            self.assertEqual(worker.cleaned, ['unit-request'])
            self.assertFalse(worker.values)
            self.assertFalse(worker.handles)
            self.assertFalse(worker.has_pending_migrations())
        finally:
            scheduler.shutdown()

    def test_actual_scheduler_propagates_worker_failure_without_terminal_success(self):
        scheduler, worker, _ = port.build_runtime(self.control, UnitExecutor(self.control, 'dit_step_chunk'))
        scheduler.start()
        try:
            with self.assertRaisesRegex(RuntimeError, 'explicit unit-executor failure'):
                scheduler.execute(self.request(), timeout_s=2.)
            self.assertEqual(worker.trace[-1]['status'], 'failed')
            self.assertFalse(any(row['kind'] in ('vae_decode','finalize') for row in worker.trace))
            self.assertFalse(worker.values)
        finally:
            scheduler.shutdown()

    def test_native_policy_receives_finite_deadlines_and_reorders_queued_requests(self):
        scheduler, worker, _ = port.build_runtime(self.control, UnitExecutor(self.control))
        scheduler.start()
        try:
            for name, budget in [('blocker',100.), ('late',60000.), ('early',1000.)]:
                request = self.request(28, name)
                request['runtime_v2_deadline_ms'] = budget
                before = time.monotonic_ns() / 1e6
                scheduler.submit_request(request)
                after = time.monotonic_ns() / 1e6
                meta = scheduler.policy.request_meta[name]
                self.assertTrue(math.isfinite(meta.deadline_ms))
                self.assertGreaterEqual(meta.deadline_ms, before + budget)
                self.assertLessEqual(meta.deadline_ms, after + budget)
                self.assertEqual(meta.arrival_ms, 0.)
            deadline = time.monotonic() + 2.
            while len(worker.cleaned) < 3:
                if time.monotonic() > deadline:
                    self.fail('actual original EDF did not complete all unit requests')
                scheduler.poll_once()
                for name in ('blocker','late','early'):
                    scheduler.get_request_status(name)
            self.assertEqual([row['request_id'] for row in worker.trace if row['kind']=='finalize'],
                             ['blocker','early','late'])
        finally:
            scheduler.shutdown()

    def test_migration_and_nonlocal_parallel_forms_cannot_fall_back(self):
        scheduler, worker, _ = port.build_runtime(self.control, UnitExecutor(self.control))
        with self.assertRaisesRegex(NotImplementedError, 'no host/NCCL fallback'):
            worker.begin_migration(object(), on_done=lambda: None, on_error=lambda: None)
        p = self.control.protocol
        remote = p.ExecutionGroupSpec('wrong-sp', (0,), p.ParallelSpec(sp=2), self.control.common.STANDARD_TASK_KINDS)
        scheduler.topology.ensure_group(remote)
        task = next(iter(port.SD35Compiler(self.control).compile_request(self.request()).tasks.values()))
        task.group_id = remote.group_id
        worker.start()
        with self.assertRaisesRegex(ValueError, 'parallel form'):
            worker.dispatch(task=task, inline_inputs=(), release_after_exec_artifact_ids=())
        worker.shutdown()

    def test_derived_mutation_extra_file_and_reused_output_fail_closed(self):
        root = Path(self.derived['root'])
        with self.assertRaises(FileExistsError):
            port.apply(SOURCE, MANIFEST, MANIFEST_SHA, root)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            port.apply(SOURCE, MANIFEST, MANIFEST_SHA, SOURCE / 'must-not-create')
        with self.assertRaisesRegex(ValueError, 'manifest SHA256'):
            port.load_control(root, '0'*64)
        extra = root / 'injected.so'
        try:
            extra.write_bytes(b'not part of bound source')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                port.load_control(root, self.derived['manifest_sha256'])
        finally:
            extra.unlink()
        selected = root / port.NAMESPACE / 'scheduler.py'
        original = selected.read_bytes()
        try:
            selected.write_bytes(original + b'\n# altered source\n')
            with self.assertRaisesRegex(ValueError, 'source file differs'):
                port.load_control(root, self.derived['manifest_sha256'])
        finally:
            selected.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
