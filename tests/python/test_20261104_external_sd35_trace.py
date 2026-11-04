"""CPU observer contract tests; fake tensors do not claim real GPU equivalence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import external_sd35_trace as trace


class Tensor:
    def __init__(self, shape=(1, 16, 64, 128), name='tensor'):
        self.shape, self.name = shape, name
    def __len__(self):
        return self.shape[0]
    def __iter__(self):
        return iter(Tensor((), str(i)) for i in range(len(self)))


def summarize(value):
    return dict(shape=list(value.shape), dtype='torch.float32' if value.shape == () else 'torch.bfloat16',
        sha256=hashlib.sha256(value.name.encode()).hexdigest(), finite=True,
        minimum=0., maximum=1., retained_array=False)


class Scheduler:
    def __init__(self):
        self.config = dict(shift=1.)
        self.step_index = None
        self.begin_index = None
        self.sigmas = Tensor((29,), 'sigmas')
    def step(self, noise_pred, t, latents):
        self.step_index = 1 if self.step_index is None else self.step_index + 1
        return latents


class SchedulerWrapper(Scheduler):
    def __init__(self):
        super().__init__()
        self.module = Scheduler()


class Pipe:
    def __init__(self):
        self.scheduler = SchedulerWrapper()
        self.calls = dict(sync=0, backbone=0, scheduler=0, naive=0, async_=0)
        self.identities = []
        self.failure = False
    def _sync_pipeline(self, latents, prompt_embeds, pooled_prompt_embeds, timesteps,
                       num_warmup_steps=0, progress_bar=None, sync_only=False):
        self.calls['sync'] += 1
        for t in timesteps:
            output = self._backbone_forward(latents, prompt_embeds, pooled_prompt_embeds, t)
            assert output is self.identities[-1]
            updated = self._scheduler_step(output[0], latents, t)
            assert updated is latents
        return latents
    def _backbone_forward(self, latents, encoder_hidden_states, pooled_prompt_embeds, t):
        self.calls['backbone'] += 1
        if self.failure:
            raise RuntimeError('actual forward failure')
        output = (latents, None)
        self.identities.append(output)
        return output
    def _scheduler_step(self, noise_pred, latents, t):
        self.calls['scheduler'] += 1
        return self.scheduler.step(noise_pred, t, latents)
    def _naive_forward(self, *args, **kwargs):
        self.calls['naive'] += 1
        return self
    def _async_pipeline(self, *args, **kwargs):
        self.calls['async_'] += 1
        return self


def runtime():
    values = {name: None for name in trace.RUNTIME_FIELDS}
    values.update(split_text_embed_in_sp=False, split_latents_by_rows=True,
        text_embed_sp_pad=0, patch_mode=False, pipeline_patch_idx=0, num_pipeline_patch=1,
        pp_patches_start_end_idx_global=[[0, 64]], pp_patches_height=[64])
    return SimpleNamespace(**values, parallel_config=SimpleNamespace(pp_degree=1, cfg_degree=1, sp_degree=2, dp_degree=1))


def invoke(pipe, length=28):
    value = Tensor()
    result = pipe._sync_pipeline(value, Tensor((2, 333, 4096)), Tensor((2, 2048)),
        Tensor((length,)), sync_only=True)
    assert result is value
    return value


def complete_observer(rank=0):
    pipe = Pipe()
    observer = trace.TraceObserver(pipe, runtime, rank, summarize=summarize)
    for name in trace.TRAJECTORIES:
        with observer.trajectory(name):
            ends = trace.CHUNK_ENDS if name == 'round_sp2_chunked' else [28]
            cursor = 0
            for end in ends:
                if name != 'normal_sp2':
                    pipe.scheduler = Scheduler()
                    pipe.scheduler.step_index = 0
                    pipe.scheduler.sigmas = Tensor((end - cursor + 1,), 'sigmas')
                invoke(pipe, end - cursor)
                cursor = end
    assert observer.finish()
    return observer, pipe


class TraceTests(unittest.TestCase):
    def test_scalar_hash_keeps_empty_shape_and_uses_flat_byte_view(self):
        raw = struct.pack('<f', 3.5)
        class Scalar:
            ndim = 0
            dtype = 'torch.float32'
            def detach(self): return self
            def contiguous(self): return self
            def cpu(self): return self
            def reshape(self, size):
                self.asserted_size = size
                return SimpleNamespace(view=lambda dtype: SimpleNamespace(numpy=lambda: SimpleNamespace(tobytes=lambda: raw)))
            def view(self, dtype): raise AssertionError('unsafe scalar view')
            def __float__(self): return 3.5
        tensor = Scalar()
        fake_torch = SimpleNamespace(uint8=object(), isfinite=lambda value: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)))
        with patch.dict(sys.modules, torch=fake_torch):
            row = trace.tensor_description(tensor)
        digest = hashlib.sha256(json.dumps(dict(shape=[], dtype='torch.float32'), sort_keys=True).encode())
        digest.update(raw)
        self.assertEqual(row['shape'], [])
        self.assertEqual(row['sha256'], digest.hexdigest())
        self.assertEqual(tensor.asserted_size, -1)

    def test_scheduler_config_changes_create_a_new_actual_identity(self):
        pipe = Pipe()
        observer = trace.TraceObserver(pipe, runtime, 0, summarize=summarize)
        first = observer._scheduler()
        pipe.scheduler.config['shift'] = 2.
        second = observer._scheduler()
        self.assertNotEqual(first['identity'], second['identity'])
        self.assertEqual(observer.record['schedulers'][first['identity']]['config']['shift'], 1.)
        self.assertEqual(observer.record['schedulers'][second['identity']]['config']['shift'], 2.)

    def test_exact_once_identity_and_inherited_method_restoration(self):
        observer, pipe = complete_observer()
        self.assertEqual(pipe.calls, dict(sync=15, backbone=84, scheduler=84, naive=0, async_=0))
        self.assertFalse(any(method in vars(pipe) for method in trace.METHODS))
        self.assertTrue(observer.record['complete'])
        self.assertLess(len(trace.encoded(observer.record)), trace.MAX_BYTES)
        self.assertEqual(len(observer.record['schedulers']), 2)
        self.assertTrue(any('inner' in item for item in observer.record['schedulers'].values()))
        self.assertTrue(all(row['inputs']['t']['shape'] == []
            for item in observer.record['trajectories'].values() for row in item['backbone']))

    def test_instance_override_and_original_error_restore_exact_attribute(self):
        pipe = Pipe()
        original = pipe._backbone_forward
        pipe._backbone_forward = original
        observer = trace.TraceObserver(pipe, runtime, 0, summarize=summarize)
        with observer.trajectory('normal_sp2'):
            invoke(pipe)
        self.assertIs(pipe.__dict__['_backbone_forward'], original)
        pipe.failure = True
        observer = trace.TraceObserver(pipe, runtime, 0, summarize=summarize)
        with self.assertRaisesRegex(RuntimeError, 'actual forward failure'):
            with observer.trajectory('normal_sp2'):
                invoke(pipe)
        self.assertIs(pipe.__dict__['_backbone_forward'], original)
        self.assertFalse(any(method in vars(pipe) for method in trace.METHODS if method != '_backbone_forward'))
        event = observer.record['trajectories']['normal_sp2']['backbone'][0]
        self.assertFalse(event['returned'])
        self.assertFalse(observer.finish())
        self.assertEqual(pipe.calls['backbone'], 29)
        self.assertEqual(pipe.calls['scheduler'], 28)

    def test_chunks_use_absolute_cursor_despite_scheduler_index_resets(self):
        observer, unused = complete_observer()
        chunked = observer.record['trajectories']['round_sp2_chunked']
        self.assertEqual([r['logical_step'] for r in chunked['scheduler']], list(range(28)))
        self.assertEqual([r['chunk_start'] for r in chunked['sync']], [0] + trace.CHUNK_ENDS[:-1])
        for row in chunked['scheduler']:
            self.assertEqual(row['scheduler_before']['step_index'], row['local_index'])
        for mutation in ('truncated', 'logical', 'extra', 'sync', 'returned', 'missing_output', 'time', 'noise', 'sigmas'):
            changed = deepcopy(chunked)
            if mutation == 'truncated': changed['scheduler'].pop()
            if mutation == 'logical': changed['scheduler'][5]['logical_step'] = 1
            if mutation == 'extra': changed['backbone'].append(deepcopy(changed['backbone'][-1]))
            if mutation == 'sync': changed['sync'].pop()
            if mutation == 'returned': changed['backbone'][2]['returned'] = False
            if mutation == 'missing_output': del changed['scheduler'][3]['output']
            if mutation == 'time': changed['scheduler'][3]['inputs']['t']['sha256'] = '0' * 64
            if mutation == 'noise': changed['scheduler'][3]['inputs']['noise_pred']['sha256'] = '0' * 64
            if mutation == 'sigmas': changed['sync'][0]['inputs']['sigmas']['shape'] = [29]
            self.assertFalse(trace.TraceObserver._coverage('round_sp2_chunked', changed), mutation)

    def test_naive_async_or_wrong_interval_cannot_claim_complete(self):
        for method in ('_naive_forward', '_async_pipeline'):
            pipe = Pipe()
            observer = trace.TraceObserver(pipe, runtime, 0, summarize=summarize)
            with self.assertRaisesRegex(ValueError, 'unexpected naive/async'):
                with observer.trajectory('normal_sp2'):
                    self.assertIs(getattr(pipe, method)(), pipe)
            self.assertEqual(sum(pipe.calls.values()), 1)
            self.assertFalse(observer.finish())
        pipe = Pipe()
        observer = trace.TraceObserver(pipe, runtime, 0, summarize=summarize)
        with self.assertRaisesRegex(ValueError, 'interval differs'):
            with observer.trajectory('normal_sp2'):
                invoke(pipe, 27)
        self.assertFalse(observer.finish())

    def test_storage_gate_and_exclusive_bounded_trace_write(self):
        observer, unused = complete_observer()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'rank-0.trace.json'
            with self.assertRaisesRegex(RuntimeError, 'storage stop'):
                observer.write(directory, lambda: (_ for _ in ()).throw(RuntimeError('storage stop')))
            self.assertFalse(path.exists())
            row = observer.write(directory, lambda: None)
            self.assertEqual(row['bytes'], path.stat().st_size)
            self.assertEqual(row['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError): observer.write(directory, lambda: None)
        observer.record['unexpected'] = 'x' * trace.MAX_BYTES
        with self.assertRaisesRegex(ValueError, 'one MiB'): observer.finish()

    def test_r3_reference_digest_binding_and_completion_fail_closed(self):
        binding = dict(request=dict(seed=102))
        reference = dict(schema='ditserve.tetriserve-sp2-diagnostic.v1', status='diagnostic_completed',
            diagnostic_completed=True, numerical_check_passed=False, qualification_eligible=False,
            binding=binding, ranks=[dict(rank=0), dict(rank=1)])
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.json'
            raw = trace.encoded(reference)
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            with patch.object(trace, 'REFERENCE_SHA256', digest):
                self.assertEqual(trace.bind_reference(path, digest, binding, lambda rows: len(rows) == 2), reference)
                with self.assertRaisesRegex(ValueError, 'fixed completed r3'): trace.bind_reference(path, '0'*64, binding, lambda rows: True)
                with self.assertRaisesRegex(ValueError, 'identical prior bindings'): trace.bind_reference(path, digest, {}, lambda rows: True)
                with self.assertRaisesRegex(ValueError, 'completed r3'): trace.bind_reference(path, digest, binding, lambda rows: False)
                path.write_bytes(raw + b' ')
                with self.assertRaisesRegex(ValueError, 'SHA256 differs'): trace.bind_reference(path, digest, binding, lambda rows: True)
                path.write_bytes(b'x' * (trace.REFERENCE_MAX_BYTES + 1))
                with self.assertRaisesRegex(ValueError, 'bounded regular'): trace.bind_reference(path, digest, binding, lambda rows: True)

    def test_every_final_descriptor_and_complete_trace_are_mandatory(self):
        outputs = {name:dict(latents=summarize(Tensor()), decoded=summarize(Tensor((1,3,1024,1024))))
                   for name in trace.TRAJECTORIES}
        reference = dict(ranks=[dict(rank=i, outputs=deepcopy(outputs)) for i in (0,1)])
        with TemporaryDirectory() as directory:
            rows = []
            for rank in (0,1):
                observer, unused = complete_observer(rank)
                rows.append(dict(rank=rank, outputs=deepcopy(outputs), trace=observer.write(directory, lambda: None)))
            self.assertTrue(trace.assess_observation(rows, directory, reference)['observation_admissible'])
            for rank in (0,1):
                for name in trace.TRAJECTORIES:
                    for field in ('latents','decoded'):
                        changed = deepcopy(rows)
                        changed[rank]['outputs'][name][field]['sha256'] = '0' * 64
                        result = trace.assess_observation(changed, directory, reference)
                        self.assertFalse(result['observer_reproduced'])
                        self.assertFalse(result['observation_admissible'])
            path = Path(directory) / 'rank-1.trace.json'
            original = path.read_bytes()
            path.write_bytes(original[:-10])
            result = trace.assess_observation(rows, directory, reference)
            self.assertTrue(result['observer_reproduced'])
            self.assertFalse(result['observation_admissible'])
            changed = json.loads(original)
            changed['trajectories']['normal_sp2']['scheduler'].pop()
            raw = trace.encoded(changed)
            path.write_bytes(raw)
            rows[1]['trace'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            self.assertFalse(trace.assess_observation(rows, directory, reference)['observation_admissible'])
            for mutation in ('metadata', 'identity', 'runtime'):
                changed = json.loads(original)
                if mutation == 'metadata': changed['schedulers'] = {}
                if mutation == 'identity': changed['trajectories']['normal_sp2']['scheduler'][0]['scheduler_before']['identity'] = '0' * 64
                if mutation == 'runtime': del changed['trajectories']['normal_sp2']['backbone'][0]['runtime']
                raw = trace.encoded(changed)
                path.write_bytes(raw)
                rows[1]['trace'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                self.assertFalse(trace.assess_observation(rows, directory, reference)['observation_admissible'], mutation)


if __name__ == '__main__':
    unittest.main()
