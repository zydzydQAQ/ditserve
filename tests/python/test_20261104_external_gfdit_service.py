"""CPU contracts for the real GF-DiT service adapter; no GPU/model success claim."""
from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
import math
from pathlib import Path
import pickle
import sys
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import external_gfdit_service as service

SOURCE = ROOT.parent / '.local/ditserve-kairos-rewrite-runtime/external-systems/20261103/GF-DiT'
MANIFEST = SOURCE.with_suffix('.source.json')
MANIFEST_SHA = 'e1b54cf09241ff70fc27e9d1666a44cbe38e00eaf1713b57cd5c9ade9c9db3f4'


def request(size=512, request_id='unit-request'):
    image = service.rounds.ImageRequest('A fox.', negative_prompt='blurry, low quality', height=size, width=size)
    return service.ServiceRequest(request_id, asdict(image), 2, 125., 60125.)


def descriptor(size):
    return dict(shape=[1, 3, size, size], dtype='torch.bfloat16', finite=True,
                minimum=0., maximum=1., sha256='1' * 64, retained_array=False)


def receipt(native):
    requests = [request(512, 'small'), request(1024, 'large')]
    tasks = [service.task_record(task) for req in requests for task in service.Compiler(native).compile_request(req).tasks.values()]
    dispatches = [dict(**row, group_id='unit', ranks=[0, 1], parallel_spec=dict(tp=1, sp=2, cfg=1)) for row in tasks]
    return dict(expected_requests=2, completed_requests=2, qualification_eligible=False, allows=[],
        costs_are_measurements=False, upstream_patches=[], requests=[asdict(x) for x in requests],
        native_pool_started=True, native_workers_exited=True, native_worker_pids={'0': 123, '1': 456},
        native_worker_exitcodes=[0, 0], planned_tasks=tasks, dispatches=dispatches, migrations=[],
        events=dict(task_events=[dict(task_id=row['task_id'], request_id=row['request_id'], group_id='unit', worker_rank=rank, kind='task_exec_end') for row in tasks for rank in (0, 1)]),
        completed=[dict(request_id=req.request_id, output=descriptor(req.image['height'])) for req in requests],
        comparisons=[dict(request_id=req.request_id, reference=descriptor(req.image['height']),
            comparison=dict(passed=True, bitwise_equal=True, atol=1e-5, rtol=1e-5, elements=3*req.image['height']**2,
                            max_abs_error=0., rmse=0., relative_l2_error=0.)) for req in requests])


class GFServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_dir():
            raise unittest.SkipTest('pinned official GF-DiT source required; no replacement source')
        cls.directory = TemporaryDirectory()
        # CPU contract imports use the existing source-verified namespace-only
        # relocation, not mocked vLLM. Production service imports original names.
        cls.derived = service.staged.apply(SOURCE, MANIFEST, MANIFEST_SHA, Path(cls.directory.name) / 'control')
        cls.native = service.staged.load_control(cls.derived['root'], cls.derived['manifest_sha256'])
        cls.native.codec = importlib.import_module(service.staged.NAMESPACE + '.codec')
        cls.native.interfaces = importlib.import_module(service.staged.NAMESPACE + '.interfaces')

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == service.staged.NAMESPACE or name.startswith(service.staged.NAMESPACE + '.'):
                del sys.modules[name]
        cls.directory.cleanup()

    def test_two_formal_requests_preserve_all_immutable_inputs(self):
        protocol = json.loads((ROOT / 'configs/benchmarks/formal-mixed-v1.json').read_bytes())
        requests = service.two_requests(protocol)
        self.assertEqual([row.image['height'] for row in requests], [512, 1024])
        self.assertEqual([row.image['seed'] for row in requests], [42, 43])
        self.assertEqual(len({row.request_id for row in requests}), 2)
        for row in requests:
            service.validate_request(row)
            self.assertEqual(row.image['steps'], 28)
            self.assertEqual(row.image['negative_prompt'], 'blurry, low quality')
            self.assertEqual(row.image['guidance_scale'], 4.5)
        for field, value in [('chunk_size', True), ('chunk_size', 0), ('chunk_size', 29),
                             ('runtime_v2_arrival_ms', float('nan')), ('runtime_v2_deadline_ms', 0),
                             ('request_id', '../escape')]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                service.validate_request(replace(requests[0], **{field: value}))
        for field, value in [('steps', 27), ('negative_prompt', ''), ('height', 768),
                             ('guidance_scale', 1.), ('max_sequence_length', 77)]:
            image = dict(requests[0].image, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                service.validate_request(replace(requests[0], image=image))

    def test_actual_original_compiler_covers_every_step_and_native_deadline_attributes(self):
        for size in (512, 1024):
            req = request(size)
            plan = service.Compiler(self.native).compile_request(req)
            self.assertEqual(len(plan.tasks), 19)
            intervals = sorted((task.step_range.start, task.step_range.end) for task in plan.tasks.values() if task.step_range)
            self.assertEqual(intervals, [(i, i+2) for i in range(0, 28, 2)])
            initial = plan.initial_artifacts[0].value
            self.assertIsInstance(initial, service.staged.RuntimeRequest)
            self.assertEqual(initial.runtime_v2_deadline_ms-initial.runtime_v2_arrival_ms, 60000.)
            self.assertEqual(plan.metadata['image'], req.image)
            self.assertEqual(plan.metadata['latent_seq_len'], (size//16)**2)
            self.assertEqual(plan.metadata['voxels'], size**2)
            self.assertEqual(plan.metadata['request_sha256'], service.canonical(req.image))
            all_ids = set(plan.tasks)
            for task in plan.tasks.values():
                self.assertTrue(set(task.dependencies).issubset(all_ids))
                self.assertEqual(task.payload['request_metadata']['image'], req.image)
                for handle in task.outputs:
                    if handle.artifact_id.startswith('state_'):
                        self.assertEqual(handle.codec_id, service.STATE_CODEC)
                    if handle.artifact_id == 'decoded':
                        self.assertEqual(handle.codec_id, service.DECODED_CODEC)

    def test_original_edf_converts_workload_deadline_to_submit_monotonic_time(self):
        p = self.native.protocol
        topology = self.native.topology.RuntimeTopology(
            tuple(self.native.topology.WorkerSpec(rank, rank, 'unit-'+str(rank)) for rank in (0, 1)),
            tuple(p.ExecutionGroupSpec('r'+str(rank), (rank,), p.ParallelSpec(), self.native.common.STANDARD_TASK_KINDS) for rank in (0, 1)))
        policy = self.native.scheduler.EdfBestFitSchedulerPolicy(topology, service.synthetic_estimator, allowed_sp_sizes=(1,))
        plan = service.Compiler(self.native).compile_request(request())
        before = time.monotonic_ns() / 1e6
        policy.on_request_submitted(plan)
        after = time.monotonic_ns() / 1e6
        meta = policy.request_meta[plan.request_id]
        self.assertGreaterEqual(meta.deadline_ms, before + 60000.)
        self.assertLessEqual(meta.deadline_ms, after + 60000.)
        self.assertAlmostEqual(meta.deadline_ms-policy.now_ms, 60000., delta=1.)
        self.assertEqual(meta.arrival_ms, 125.)
        self.assertGreater(meta.deadline_ms, policy.now_ms)

    def test_original_codec_describes_every_real_stage_in_sp1_and_sp2(self):
        codecs = service.codecs(self.native)
        p = self.native.protocol
        for size in (512, 1024):
            req = request(size); plan = service.Compiler(self.native).compile_request(req)
            for task in plan.tasks.values():
                for handle in task.outputs:
                    if not handle.codec_id:
                        continue
                    for ranks in ((0,), (1,), (0, 1), (1, 0)):
                        group = p.ExecutionGroupSpec('unit', ranks, p.ParallelSpec(sp=len(ranks)), self.native.common.STANDARD_TASK_KINDS)
                        layout = codecs[handle.codec_id].describe_output(group=group, group_rank=0,
                            request_metadata=plan.metadata, artifact=handle)
                        wanted = service.artifact_fields(handle.artifact_id, service.rounds.ImageRequest(**req.image))
                        self.assertEqual(layout.handle, handle)
                        self.assertEqual({field.field_path[0]: (field.local_shape, field.dtype) for field in layout.tensors},
                                         {name: (item['shape'], item['dtype']) for name, item in wanted.items()})
                        self.assertTrue(all(field.layout.kind == p.TensorLayoutKind.REPLICATED for field in layout.tensors))
                        self.assertEqual([field.global_offset for field in layout.tensors],
                                         [tuple(0 for _ in field.local_shape) for field in layout.tensors])

    def test_metadata_transport_contains_no_five_tensor_payload_and_rejects_leakage(self):
        req = request(); plan = service.Compiler(self.native).compile_request(req)
        handle = next(task.outputs[0] for task in plan.tasks.values() if task.step_range and task.step_range.end == 2)
        group = self.native.protocol.ExecutionGroupSpec('unit', (0,), self.native.protocol.ParallelSpec(), self.native.common.STANDARD_TASK_KINDS)
        codec = service.codecs(self.native)[service.STATE_CODEC]
        layout = codec.describe_output(group=group, group_rank=0, request_metadata=plan.metadata, artifact=handle)
        payload = dict(schema=service.STATE_SCHEMA, request_id=req.request_id, image=req.image,
                       request_sha256=service.canonical(req.image), cursor=2,
                       **{name: 'CPU-only metadata sentinel' for name in service.TENSORS})
        # This test exercises only the real codec's metadata split; production
        # validates actual CUDA tensors on both send and receive. No GPU claim.
        with patch.object(service, 'validate_model_tensors', side_effect=service.validate_state_metadata):
            raw = codec.pack_metadata(value=payload, layout=layout)
        skeleton = pickle.loads(raw)
        self.assertTrue(all(skeleton[name] is None for name in service.TENSORS))
        self.assertEqual(skeleton['cursor'], 2)
        self.assertEqual(skeleton['image'], req.image)
        self.assertTrue(all(payload[name] == 'CPU-only metadata sentinel' for name in service.TENSORS))
        for mutation in ('request_id', 'hash', 'field', 'cursor', 'missing', 'layout'):
            value = deepcopy(payload); selected_layout = layout
            if mutation == 'request_id': value['request_id'] = 'another-request'
            if mutation == 'hash': value['image']['seed'] += 1
            if mutation == 'field': value['undeclared'] = 'payload'
            if mutation == 'cursor': value['cursor'] = True
            if mutation == 'missing': del value['sigmas']
            if mutation == 'layout': selected_layout = replace(layout, tensors=layout.tensors[:-1])
            with self.subTest(mutation=mutation), patch.object(service, 'validate_model_tensors', side_effect=service.validate_state_metadata), self.assertRaises(ValueError):
                codec.pack_metadata(value=value, layout=selected_layout)

    def test_observers_preserve_actual_event_and_migration_call_return_values(self):
        calls = []
        def consume(event):
            calls.append(('event', event)); return ['original result']
        def begin(plan, *, on_done, on_error, timeout_s):
            calls.append(('migration', plan, timeout_s))
            on_done(SimpleNamespace(edges=(SimpleNamespace(local_copy=False),)))
            return 'original-migrate-id'
        pool = SimpleNamespace(_consume_event=consume, begin_migration=begin)
        events, migrations = [], []
        service.record_events(pool, events); service.record_migrations(pool, migrations)
        event = SimpleNamespace(kind=SimpleNamespace(value='task_exec_end'), task_id='x', worker_rank=0)
        self.assertEqual(pool._consume_event(event), ['original result'])
        handle = SimpleNamespace(artifact_id='state')
        plan = SimpleNamespace(request_id='r', src_group_id='a', dst_group_id='b', artifacts=[SimpleNamespace(handle=handle)])
        done = []
        self.assertEqual(pool.begin_migration(plan, on_done=done.append, on_error=lambda error: self.fail(str(error)), timeout_s=37), 'original-migrate-id')
        self.assertIs(calls[0][1], event); self.assertIs(calls[1][1], plan)
        self.assertEqual(calls[1][2], 37)
        self.assertEqual(len(done), 1); self.assertEqual(migrations[0]['remote_edges'], 1)
        self.assertEqual(migrations[0]['status'], 'completed')

    def test_all_offered_gate_preserves_failures_and_zero_migration_scope(self):
        good = receipt(self.native)
        self.assertTrue(service.assess(good))
        self.assertEqual(good['migrations'], [])
        for mutation in ('missing_request', 'missing_comparison', 'numerical_error', 'nonfinite_error',
                         'tolerance', 'shape', 'failure_event', 'missing_task', 'pid', 'exit',
                         'cost_scope', 'qualified', 'migration', 'huge_error', 'rmse_bound', 'bitwise', 'missing_rank', 'duplicate_rank', 'extra_task', 'changed_interval', 'extra_output', 'extra_reference'):
            value = deepcopy(good)
            if mutation == 'missing_request': value['completed_requests'] = 1
            if mutation == 'missing_comparison': value['comparisons'].pop()
            if mutation == 'numerical_error': value['comparisons'][0]['comparison']['passed'] = False
            if mutation == 'nonfinite_error': value['comparisons'][0]['comparison']['rmse'] = float('nan')
            if mutation == 'tolerance': value['comparisons'][0]['comparison']['atol'] = 1.
            if mutation == 'shape': value['completed'][0]['output']['shape'] = [1, 3, 1, 1]
            if mutation == 'failure_event': value['events']['task_events'].append(dict(kind='task_failed'))
            if mutation == 'missing_task': value['events']['task_events'].pop()
            if mutation == 'pid': value['native_worker_pids']['1'] = value['native_worker_pids']['0']
            if mutation == 'exit': value['native_worker_exitcodes'][1] = -15
            if mutation == 'cost_scope': value['costs_are_measurements'] = True
            if mutation == 'qualified': value['qualification_eligible'] = True
            if mutation == 'migration': value['migrations'] = [dict(status='failed')]
            if mutation == 'huge_error': value['comparisons'][0]['comparison']['max_abs_error'] = 1e9
            if mutation == 'rmse_bound': value['comparisons'][0]['comparison']['rmse'] = 1e-7
            if mutation == 'bitwise': value['comparisons'][0]['comparison']['bitwise_equal'] = False
            if mutation == 'missing_rank': value['events']['task_events'] = [row for row in value['events']['task_events'] if row['worker_rank'] == 0]
            if mutation == 'duplicate_rank': value['events']['task_events'].append(deepcopy(value['events']['task_events'][0]))
            if mutation == 'extra_task': value['dispatches'].append(deepcopy(value['dispatches'][0]))
            if mutation == 'changed_interval': next(row for row in value['dispatches'] if row['step_range'])['step_range']['end'] = 3
            if mutation == 'extra_output': value['completed'].append(dict(request_id='unexpected'))
            if mutation == 'extra_reference': value['comparisons'].append(dict(request_id='unexpected'))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                service.assess(value)

    def test_actual_torch_uuid_format_accepts_bare_or_prefixed_and_rejects_placeholders(self):
        raw = 'bf6392ab-b12b-746c-58c7-1115d4730ab3'
        self.assertEqual(service.normalized_uuid(raw), 'GPU-' + raw)
        self.assertEqual(service.normalized_uuid('GPU-' + raw), 'GPU-' + raw)
        for value in ('GPU-0', 'gpu-' + raw, 'none', '', None, raw.replace('-', '')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                service.normalized_uuid(value)

    def test_storage_uses_new_strict_stop_line_and_never_stops_other_jobs(self):
        for used, percent, accepted in [(800, 79, True), (801, 79, False), (777, 80, False)]:
            result = SimpleNamespace(returncode=0, stdout=f'Filesystem 1G-blocks Used Available Use% Mounted\n/dev/test 1024G {used}G 224G {percent}% /shared\n')
            with patch.object(service.subprocess, 'run', return_value=result) as run:
                if accepted:
                    self.assertEqual(service.check_storage()['used_gib'], used)
                else:
                    with self.assertRaises(RuntimeError): service.check_storage()
                self.assertEqual(run.call_args.args[0], ['df', '-BG', '/shared'])


if __name__ == '__main__':
    unittest.main()
