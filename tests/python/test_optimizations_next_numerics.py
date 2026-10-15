"""Tiny CPU pipeline fixtures; no model downloads, compilation, or GPU calls."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ditserve.directions.budget import BudgetStop
from ditserve.optimizations import next_numerics as n


class Transformer:
    dtype = torch.float32
    config = SimpleNamespace(in_channels=1, guidance_embeds_scale=1.)

    def __init__(self, adapter, compiled=False):
        self.adapter, self.compiled = adapter, compiled

    def __call__(self, hidden_states, **kwargs):
        if self.compiled and self.adapter.fail_compile:
            raise RuntimeError("fixture compile failure")
        prediction = hidden_states * .1 + kwargs['encoder_hidden_states'].reshape(-1, 1, 1, 1) * .01
        if self.compiled:
            prediction += .03
        return (prediction,)


class Scheduler:
    def __init__(self, adapter):
        self.adapter = adapter

    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        streams = generator if isinstance(generator, list) else [generator]
        noise = torch.cat([torch.randn((1, 1, 2, 2), generator=g) for g in streams])
        original = sample - model_output
        previous = original * .9 + noise * .1
        return (previous, original) if not return_dict else SimpleNamespace(prev_sample=previous, pred_original_sample=original)


class Pipeline:
    def __init__(self, adapter):
        self.adapter = adapter
        self.transformer = Transformer(adapter)
        self.scheduler = Scheduler(adapter)
        self.config = {'transformer': ('fixture', 'Transformer')}

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        streams = generator if isinstance(generator, list) else [generator]
        return torch.cat([torch.randn((1, 1, 2, 2), generator=g) for g in streams])

    def __call__(self, prompts, seeds, check_budget):
        streams = [torch.Generator(device='cpu').manual_seed(seed) for seed in seeds]
        generator = streams[0] if len(streams) == 1 else streams
        latents = self.prepare_latents(len(streams), 1, 2, 2, torch.float32, 'cpu', generator)
        # Match official signature-driven generator forwarding, the actual bug
        # an untyped *args/**kwargs wrapper would introduce.
        kwargs = {'generator': generator} if 'generator' in inspect.signature(self.scheduler.step).parameters else {}
        embedding = torch.tensor([float(len(p)) for p in prompts]).reshape(-1, 1)
        if len(prompts) == 2:
            embedding = embedding + .01  # Simulate batch-sensitive text encoder arithmetic.
        for index in range(2):
            time = torch.full((len(streams),), float(index))
            output = self.transformer(latents, encoder_hidden_states=embedding,
                encoder_attention_mask=torch.ones((len(streams), 1), dtype=torch.bool),
                guidance=torch.ones(len(streams)), timestep=time, return_dict=False, attention_kwargs=None)[0]
            latents, denoised = self.scheduler.step(output, time, latents, **kwargs, return_dict=False)
            check_budget()
        return denoised.sigmoid().permute(0, 2, 3, 1).repeat(1, 1, 1, 3).numpy()


class Adapter:
    def __init__(self):
        self.torch = torch
        self.backend, self._busy, self.in_call, self.fail_compile = 'eager', False, False, False
        self.pipe = Pipeline(self)
        self.eager = self.pipe.transformer
        self.compiled = Transformer(self, compiled=True)
        self.calls = []

    def set_backend(self, backend):
        if self._busy:
            raise RuntimeError('busy')
        self.backend = backend
        self.pipe.transformer = self.eager if backend == 'eager' else self.compiled

    def compilation_summary(self):
        return dict(graph_compile_count=int(any(c[0] == 'compiled' for c in self.calls)))

    def generate_batch(self, prompts, *, seeds, check_budget):
        self.calls.append((self.backend, list(prompts), list(seeds)))
        self._busy = self.in_call = True
        try:
            return self.pipe(prompts, seeds, check_budget), dict(backend=self.backend, compile_delta=0)
        finally:
            self._busy = self.in_call = False


def quality(reference, candidate, lpips_device):
    value = float(np.abs(reference - candidate).mean())
    return dict(status='ok', finite=True, range_valid=True, lpips_mean=value, lpips_p95=value,
                lpips_per_frame=[value], lpips_status='ok')


class NumericsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.adapter = Adapter()
        self.context = SimpleNamespace(sana=self.adapter, output=Path(self.temp.name),
            config=dict(direction1=dict(prompts=['a', 'longer']), quality=dict(rtol=1e-5, atol=1e-6, image_lpips=.01)),
            check_budget=lambda: None, quality_evaluator=quality)

    def tearDown(self):
        self.temp.cleanup()

    def test_singleton_zero_stride_summary_preserves_float_and_bfloat_bits(self):
        cases=((torch.float32,1.5,np.array([1.5],dtype=np.float32).tobytes()),
               (torch.bfloat16,1.5,np.array([0x3fc0],dtype=np.uint16).tobytes()),
               (torch.bfloat16,-0.,np.array([0x8000],dtype=np.uint16).tobytes()))
        for dtype,number,expected in cases:
            with self.subTest(dtype=dtype,number=number):
                value=torch.tensor(number,dtype=dtype).expand(1)
                self.assertTrue(value.is_contiguous())
                self.assertEqual(value.stride(),(0,))
                self.assertEqual(value.contiguous().stride(),(0,))
                before=(value.data_ptr(),value.stride(),value.storage_offset())
                result=n._tensor_summary(value)
                self.assertEqual(result['shape'],[1])
                self.assertEqual(result['dtype'],str(dtype))
                self.assertEqual(result['bytes'],len(expected))
                self.assertEqual(result['sha256'],hashlib.sha256(expected).hexdigest())
                self.assertTrue(result['finite'])
                self.assertEqual((result['min'],result['max'],result['mean']),(number,number,number))
                self.assertEqual((value.data_ptr(),value.stride(),value.storage_offset()),before)

    def test_scalar_empty_and_noncontiguous_summary_keeps_original_shape(self):
        for dtype in (torch.float32,torch.bfloat16):
            scalar=torch.tensor(3.25,dtype=dtype)
            result=n._tensor_summary(scalar)
            dense=torch.tensor([3.25],dtype=dtype)
            self.assertEqual(result['shape'],[])
            self.assertEqual(result['sha256'],hashlib.sha256(dense.view(torch.uint8).numpy().tobytes()).hexdigest())
            self.assertEqual(result['mean'],3.25)
            for shape in ((0,),(2,0,1)):
                with self.subTest(dtype=dtype,shape=shape):
                    value=torch.empty(shape,dtype=dtype)
                    result=n._tensor_summary(value)
                    self.assertEqual(result['shape'],list(shape))
                    self.assertEqual(result['bytes'],0)
                    self.assertEqual(result['sha256'],hashlib.sha256(b'').hexdigest())
                    self.assertTrue(result['finite'])
                    self.assertEqual((result['min'],result['max'],result['mean']),(None,None,None))
            source=torch.arange(12,dtype=dtype).reshape(3,4)
            value=source[1:].t()
            self.assertFalse(value.is_contiguous())
            self.assertGreater(value.storage_offset(),0)
            self.assertEqual(n._tensor_summary(value),n._tensor_summary(value.contiguous()))
            self.assertEqual(n._tensor_summary(value)['shape'],[4,2])
            self.assertTrue(torch.equal(source,torch.arange(12,dtype=dtype).reshape(3,4)))

    def test_summary_keeps_nonfinite_bit_patterns_without_claiming_finite_statistics(self):
        raw=np.array([0x7fc1],dtype=np.uint16)
        value=torch.from_numpy(raw.copy()).view(torch.bfloat16)[0].expand(1)
        result=n._tensor_summary(value)
        self.assertFalse(result['finite'])
        self.assertEqual(result['sha256'],hashlib.sha256(raw.tobytes()).hexdigest())
        self.assertEqual((result['min'],result['max'],result['mean']),(None,None,None))

    def test_complete_diagnosis_handles_real_expanded_singleton_timestep_boundaries(self):
        original=Pipeline.__call__
        original_add=n._Capture.add
        seen=[]
        def expanded(capture,name,tensor):
            if name.endswith('input.timestep') and tensor.numel()==1:
                tensor=tensor.reshape(())[()].expand(1)
                self.assertEqual(tensor.stride(),(0,))
                seen.append(name)
            return original_add(capture,name,tensor)
        with patch.object(n._Capture,'add',expanded):
            result=n.run_numerics(self.context)
        self.assertGreaterEqual(len(seen),16)
        self.assertTrue(result['diagnostic_complete'])
        self.assertEqual(result['completed_units'],12)
        self.assertEqual(result['failed_units'],0)
        self.assertEqual(result['compared_images'],12)
        self.assertFalse(result['local_qualified'])
        self.assertFalse(result['changes_serving_default'])
        self.assertIs(Pipeline.__call__,original)

    def test_complete_fixed_matrix_attributes_first_drift_without_changing_defaults(self):
        original_forward = Transformer.__call__
        original_config = copy.deepcopy(self.adapter.pipe.config)
        result = n.run_numerics(self.context)
        self.assertEqual(len(self.adapter.calls), 12)
        self.assertEqual(sum(len(c[1]) for c in self.adapter.calls), 16)
        self.assertTrue(result['diagnostic_complete'])
        self.assertFalse(result['local_qualified'])
        self.assertFalse(result['changes_serving_default'])
        self.assertEqual(result['compared_images'], 12)
        self.assertEqual(result['failed_units'], 0)
        for row in result['rows']:
            self.assertTrue(row['execution']['same_cpu_generators_through_scm'])
            for comparison in row['comparisons']:
                expected = ('step0.transformer.input.encoder_hidden_states' if 'batch2' in row['arm']
                            else 'step0.transformer.prediction')
                self.assertEqual(comparison['trajectory']['first_strict_drift'], expected)
                rng = [c for c in comparison['trajectory']['components'] if c['boundary'].endswith('generator_state')]
                self.assertTrue(all(c['numerical']['exact'] for c in rng))
        self.assertEqual(self.adapter.pipe.config, original_config)
        self.assertIs(Transformer.__call__, original_forward)
        self.assertIs(self.adapter.pipe.transformer, self.adapter.eager)
        self.assertNotIn('step', vars(self.adapter.pipe.scheduler))
        self.assertNotIn('prepare_latents', vars(self.adapter.pipe))
        self.assertTrue(result['backend_restored'])

    def test_cpu_snapshot_transfer_happens_only_after_full_generation(self):
        original = n._to_cpu
        count = []
        def transfer(value):
            self.assertFalse(self.adapter.in_call)
            count.append(1)
            return original(value)
        with patch.object(n, '_to_cpu', side_effect=transfer):
            n.run_numerics(self.context)
        self.assertGreater(len(count), 20)

    def test_reverse_batch_pairs_by_input_identity_and_same_rng(self):
        result = n.run_numerics(self.context)
        rows = result['rows']
        for forward, reverse in ((rows[4], rows[5]), (rows[6], rows[7])):
            self.assertEqual(forward['input_ids'], reverse['input_ids'][::-1])
            a = {c['input_id']: c for c in forward['comparisons']}
            b = {c['input_id']: c for c in reverse['comparisons']}
            for name in a:
                self.assertEqual(a[name]['output']['candidate_sha256'], b[name]['output']['candidate_sha256'])

    def test_compile_failure_is_retained_four_times_and_hooks_restored(self):
        self.adapter.fail_compile = True
        result = n.run_numerics(self.context)
        self.assertEqual(result['failed_units'], 4)
        self.assertEqual(len(result['rows']), 12)
        self.assertEqual(result['status'], 'passed')  # The diagnosis completed, backend is not qualified.
        self.assertFalse(result['local_qualified'])
        self.assertTrue(all('fixture compile failure' in r['error'] for r in result['rows'][-4:]))
        self.assertNotIn('step', vars(self.adapter.pipe.scheduler))
        self.assertNotIn('prepare_latents', vars(self.adapter.pipe))
        self.assertIs(self.adapter.pipe.transformer, self.adapter.eager)

    def test_quality_failure_does_not_omit_remaining_fixed_units(self):
        def rejected(*args, **kwargs):
            return dict(status='ok', finite=True, range_valid=True, lpips_mean=.02, lpips_p95=.02)
        self.context.quality_evaluator = rejected
        result = n.run_numerics(self.context)
        self.assertEqual(result['compared_images'], 12)
        self.assertEqual(result['lpips_passed_images'], 0)
        self.assertEqual(result['completed_units'], 12)

    def test_budget_stop_inside_step_restores_hooks_backend_and_preserves_partial(self):
        def guard():
            if self.adapter.in_call:
                raise BudgetStop('fixture budget stop')
        self.context.check_budget = guard
        with self.assertRaises(BudgetStop):
            n.run_numerics(self.context)
        result = json.loads((self.context.output / 'sana-numerics.json').read_text())
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(result['rows']), 1)
        self.assertEqual(result['rows'][0]['status'], 'incomplete')
        self.assertIsNone(result['active_unit'])
        self.assertTrue(result['backend_restored'])
        self.assertNotIn('step', vars(self.adapter.pipe.scheduler))
        self.assertNotIn('prepare_latents', vars(self.adapter.pipe))

    def test_between_units_interrupt_does_not_rewrite_completed_row(self):
        def guard():
            if len(self.adapter.calls) == 1 and not self.adapter.in_call:
                # First post-generation member processing still belongs to unit
                # zero, so stop only after its row has been persisted complete.
                p = self.context.output / 'sana-numerics.json'
                if p.exists() and len(json.loads(p.read_text())['rows']) == 1:
                    raise BudgetStop('between units')
        self.context.check_budget = guard
        with self.assertRaises(BudgetStop):
            n.run_numerics(self.context)
        result = json.loads((self.context.output / 'sana-numerics.json').read_text())
        self.assertEqual(result['rows'][0]['status'], 'completed')
        self.assertIsNone(result['active_unit'])

    def test_scheduler_hook_preserves_signature_and_existing_instance_hooks(self):
        pipe = self.adapter.pipe
        original = pipe.scheduler.step
        def existing(model_output, timestep, sample, generator=None, return_dict=True):
            return original(model_output, timestep, sample, generator=generator, return_dict=return_dict)
        pipe.scheduler.step = existing
        capture = n._Capture(self.adapter, 1)
        with capture.installed():
            self.assertIn('generator', inspect.signature(pipe.scheduler.step).parameters)
        self.assertIs(pipe.scheduler.step, existing)

    def test_wrong_or_shared_scm_rng_is_rejected_without_sampling(self):
        capture = n._Capture(self.adapter, 2)
        a, b = torch.Generator().manual_seed(101), torch.Generator().manual_seed(102)
        with self.assertRaises(ValueError):
            capture.generators('initial', [a, a])
        capture.generators('initial', [a, b])
        with self.assertRaises(ValueError):
            capture.generators('scheduler', [b, a])

    def test_snapshot_bound_and_recipe_guard_never_silently_reduce_scope(self):
        capture = n._Capture(self.adapter, 1)
        with patch.object(n, 'MAX_CAPTURE_BYTES', 1), self.assertRaises(RuntimeError):
            capture.add('x', torch.ones(2))
        self.context.config['quality']['image_lpips'] = .02
        with self.assertRaises(ValueError):
            n.run_numerics(self.context)
        self.assertEqual(self.adapter.calls, [])


if __name__ == '__main__':
    unittest.main()
