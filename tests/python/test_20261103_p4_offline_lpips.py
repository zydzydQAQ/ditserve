"""Offline contract/math tests; these mocks do not qualify real LPIPS scores."""
from contextlib import nullcontext
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import p4_offline_lpips as metric


class TensorFixture:
    def __init__(self, shape, *, finite=True, dtype='float32', device='cpu'):
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)
        self.finite = finite


FAKE_TORCH = SimpleNamespace(
    float32='float32', is_tensor=lambda value: isinstance(value, TensorFixture),
    isfinite=lambda value: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: value.finite)))


def checkpoint_fixture(shapes):
    return {name: TensorFixture(shape) for name, shape in shapes.items()}


class ModuleFixture:
    def __init__(self, keys):
        self.keys = set(keys)
        self.calls = []

    def state_dict(self):
        return dict.fromkeys(self.keys)

    def load_state_dict(self, state, *, strict):
        if not strict or set(state) != self.keys:
            raise AssertionError('non-strict fixture load')
        self.calls.append(state)


def model_fixture():
    model = SimpleNamespace(net=ModuleFixture(metric.TRUNK_MAPPING.values()), lins=[])
    for i in range(5):
        head = ModuleFixture({'model.1.weight'})
        setattr(model, f'lin{i}', head)
        model.lins.append(head)
    names = {'net.' + name for name in metric.TRUNK_MAPPING.values()} | set(metric.HEAD_SHAPES)
    model.named_parameters = lambda: [(name, None) for name in names]
    return model


class OfflineLPIPSContractTests(unittest.TestCase):
    def test_exact_measured_weight_and_source_identities_are_frozen(self):
        self.assertEqual(metric.BACKBONE.size + metric.HEAD.size, 244414920)
        self.assertEqual(metric.BACKBONE.sha256[:8], '7be5be79')
        self.assertEqual(metric.PACKAGES, {'torch': '2.10.0+cu126', 'torchvision': '0.25.0+cu126', 'lpips': '0.1.4'})
        self.assertEqual(len(metric.SOURCES), 4)
        self.assertEqual(len(metric.BACKBONE_SHAPES), 16)
        self.assertEqual(len(metric.TRUNK_MAPPING), 10)
        self.assertEqual(len(metric.HEAD_SHAPES), 5)

    def test_local_file_hash_size_and_stability_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small-checkpoint'
            path.write_bytes(b'known bytes')
            expected = metric.FileIdentity(11, hashlib.sha256(b'known bytes').hexdigest())
            with metric.verified_file(path, expected) as (stream, receipt):
                self.assertEqual(stream.read(), b'known bytes')
                self.assertEqual(receipt['sha256'], expected.sha256)
            for wrong in (metric.FileIdentity(12, expected.sha256), metric.FileIdentity(11, '0' * 64)):
                with self.assertRaises(metric.OfflineLPIPSError), metric.verified_file(path, wrong):
                    self.fail('bad checkpoint accepted')
            with self.assertRaisesRegex(metric.OfflineLPIPSError, 'changed during use'):
                with metric.verified_file(path, expected):
                    path.write_bytes(b'known bytes!')
            with self.assertRaises(metric.OfflineLPIPSError), metric.verified_file('https://example.invalid/model', expected):
                self.fail('remote URL accepted')

    def test_deserializer_uses_verified_descriptor_cpu_and_weights_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small-checkpoint'
            path.write_bytes(b'fixture')
            identity = metric.FileIdentity(7, hashlib.sha256(b'fixture').hexdigest())
            state = checkpoint_fixture(metric.HEAD_SHAPES)
            loader = Mock(return_value=state)
            torch = SimpleNamespace(**vars(FAKE_TORCH), load=loader)
            loaded, receipt = metric._checkpoint(torch, path, identity, metric.HEAD_SHAPES)
            self.assertIs(loaded, state)
            args, kwargs = loader.call_args
            self.assertEqual(kwargs, {'map_location': 'cpu', 'weights_only': True})
            self.assertTrue(args[0].closed)
            self.assertEqual(receipt['bytes'], 7)

    def test_trunk_mapping_and_all_five_heads_are_loaded_strictly(self):
        backbone = checkpoint_fixture(metric.BACKBONE_SHAPES)
        heads = checkpoint_fixture(metric.HEAD_SHAPES)
        model = model_fixture()
        metric._install_weights(model, backbone, heads, FAKE_TORCH)
        for source, destination in metric.TRUNK_MAPPING.items():
            self.assertIs(model.net.calls[0][destination], backbone[source])
        for i in range(5):
            self.assertIs(model.lins[i].calls[0]['model.1.weight'], heads[f'lin{i}.model.1.weight'])
        self.assertFalse(any('classifier' in name for name in model.net.calls[0]))

    def test_corrupt_missing_unexpected_tensor_and_architecture_are_rejected(self):
        mutations = [
            lambda values: values.pop(next(iter(values))),
            lambda values: values.update(unexpected=TensorFixture((1,))),
            lambda values: values.update({next(iter(values)): TensorFixture((1,))}),
            lambda values: setattr(next(iter(values.values())), 'finite', False),
            lambda values: setattr(next(iter(values.values())), 'dtype', 'float16'),
            lambda values: setattr(next(iter(values.values())), 'device', SimpleNamespace(type='cuda')),
        ]
        for shapes in (metric.BACKBONE_SHAPES, metric.HEAD_SHAPES):
            for mutate in mutations:
                with self.subTest(shapes=len(shapes), mutation=mutate):
                    state = checkpoint_fixture(shapes)
                    mutate(state)
                    with self.assertRaises(metric.OfflineLPIPSError):
                        metric._validate_state(state, shapes, FAKE_TORCH)
        for mutate in (lambda model: model.net.keys.add('random.weight'),
                       lambda model: model.lins.__setitem__(0, ModuleFixture({'model.1.weight'})),
                       lambda model: setattr(model, 'named_parameters', lambda: [('unloaded.weight', None)])):
            model = model_fixture()
            mutate(model)
            with self.assertRaises(metric.OfflineLPIPSError):
                metric._install_weights(model, checkpoint_fixture(metric.BACKBONE_SHAPES),
                                        checkpoint_fixture(metric.HEAD_SHAPES), FAKE_TORCH)

    def test_constructor_disables_both_automatic_weight_loaders(self):
        model = Mock()
        model.to.return_value = model
        constructor = Mock(return_value=model)
        torch = SimpleNamespace(device=lambda name: nullcontext(), float32='float32')
        self.assertIs(metric._construct_model(torch, SimpleNamespace(LPIPS=constructor)), model)
        kwargs = constructor.call_args.kwargs
        self.assertIs(kwargs['pretrained'], False)
        self.assertIs(kwargs['pnet_rand'], True)
        self.assertEqual(kwargs['version'], '0.1')
        self.assertTrue(kwargs['eval_mode'])
        model.to.assert_called_once_with(device='cpu', dtype='float32')

    def test_guard_blocks_network_download_and_process_without_contact(self):
        original_urlopen, original_connect, original_popen = urllib.request.urlopen, socket.socket.connect, subprocess.Popen
        for call in (lambda: urllib.request.urlopen('https://example.invalid'),
                     lambda: socket.getaddrinfo('example.invalid', 443),
                     lambda: socket.socket.sendall(None, b'blocked'),
                     lambda: ssl.SSLSocket.write(None, b'blocked'),
                     lambda: subprocess.Popen(['must-never-execute'])):
            with self.assertRaisesRegex(metric.OfflineLPIPSError, 'rejected a network'):
                with metric.offline_guard():
                    try:
                        call()
                    except metric.OfflineLPIPSError:
                        pass  # Dependency swallowing the exception must still fail.
        self.assertIs(urllib.request.urlopen, original_urlopen)
        self.assertIs(socket.socket.connect, original_connect)
        self.assertIs(subprocess.Popen, original_popen)
        hub = SimpleNamespace(load_state_dict_from_url=Mock(), download_url_to_file=Mock())
        with self.assertRaises(metric.OfflineLPIPSError), metric.offline_guard(SimpleNamespace(hub=hub)):
            hub.load_state_dict_from_url('file:///also-blocked')
        hub.load_state_dict_from_url.assert_not_called()
        with patch.object(metric.threading, 'active_count', return_value=2):
            with self.assertRaisesRegex(metric.OfflineLPIPSError, 'single Python-thread'), metric.offline_guard():
                self.fail('global monkeypatching while other Python threads run')

    def test_missing_or_incompatible_packages_fail_before_import(self):
        for value in (importlib.metadata.PackageNotFoundError('torch'), '2.10.0+cu128'):
            kwargs = {'side_effect': value} if isinstance(value, Exception) else {'return_value': value}
            with patch.object(metric.importlib.metadata, 'version', **kwargs), \
                    patch.object(metric.importlib, 'import_module') as importing:
                with self.assertRaises(metric.OfflineLPIPSError):
                    metric._load_dependencies()
                importing.assert_not_called()

    def test_changed_source_fails_before_dependency_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed-source.py'
            path.write_bytes(b'pass\n')
            distribution = SimpleNamespace(locate_file=lambda member: path)
            with patch.object(metric.importlib.metadata, 'version', side_effect=metric.PACKAGES.__getitem__), \
                    patch.object(metric.importlib.metadata, 'distribution', return_value=distribution), \
                    patch.object(metric.importlib, 'import_module') as importing:
                with self.assertRaisesRegex(metric.OfflineLPIPSError, 'file size differs'):
                    metric._load_dependencies()
                importing.assert_not_called()

    def test_frame_conversion_preserves_rgb_axes_pixels_and_source(self):
        original = np.zeros((32, 40, 3), dtype=np.float64)
        original[..., 1] = .5
        original[..., 2] = 1.
        before = original.copy()
        converted = metric.prepare_frame(original)
        self.assertEqual(converted.shape, (1, 3, 32, 40))
        self.assertEqual(converted.dtype, np.float32)
        self.assertTrue(converted.flags.c_contiguous)
        np.testing.assert_array_equal(converted[0, :, 0, 0], [-1., 0., 1.])
        np.testing.assert_array_equal(original, before)
        np.testing.assert_array_equal(converted, metric.prepare_frame(original * 2. - 1., 'minus_one_one'))

    def test_frame_range_finiteness_layout_and_encoding_are_not_guessed(self):
        base = np.zeros((32, 32, 3), dtype=np.float32)
        cases = [(base.astype('uint8'), 'zero_one'), (base + np.nan, 'zero_one'),
                 (base + np.inf, 'zero_one'), (base - .01, 'zero_one'),
                 (base + 1.01, 'zero_one'), (base - 1.01, 'minus_one_one'),
                 (base[None], 'zero_one'), (base[:31], 'zero_one'), (base, 'auto')]
        for value, declared in cases:
            with self.subTest(shape=value.shape, declared=declared), self.assertRaises(metric.OfflineLPIPSError):
                metric.prepare_frame(value, declared)

    def test_scoring_cannot_bypass_real_selftest_and_receipt_never_qualifies_p4(self):
        adapter = metric.OfflineLPIPS.__new__(metric.OfflineLPIPS)
        adapter._selftest = None
        adapter._identity = dict(schema=metric.SCHEMA, metric='unexecuted fixture')
        with self.assertRaisesRegex(metric.OfflineLPIPSError, 'real checkpoint CPU selftest'):
            adapter.score_frame(None, None)
        receipt = adapter.receipt()
        self.assertEqual(receipt['status'], 'selftest_required')
        self.assertFalse(receipt['p4_qualification_eligible'])
        self.assertFalse(receipt['retention_dependencies_closed'])
        self.assertEqual(receipt['vbench']['status'], 'unknown')
        digest = receipt.pop('sha256')
        self.assertEqual(digest, metric.canonical_hash(receipt))

    def test_cli_unavailable_dependency_is_unknown_and_nonzero(self):
        output = io.StringIO()
        with patch.object(metric, 'OfflineLPIPS', side_effect=metric.OfflineLPIPSError('no matching dependencies')), \
                patch('sys.stdout', output):
            self.assertEqual(metric.main(['selftest']), 2)
        report = json.loads(output.getvalue())
        self.assertEqual(report['status'], 'unknown')
        self.assertFalse(report['p4_qualification_eligible'])
        self.assertNotIn('selftest', report)

    def test_nonfinite_negative_or_nonscalar_scores_fail(self):
        for value, count, device in ((float('nan'), 1, 'cpu'), (float('inf'), 1, 'cpu'),
                                      (-.1, 1, 'cpu'), (0., 2, 'cpu'), (0., 1, 'cuda')):
            tensor = SimpleNamespace(numel=lambda: count, item=lambda: value, device=SimpleNamespace(type=device))
            with self.assertRaises(metric.OfflineLPIPSError):
                metric._scalar(tensor)
        zero = SimpleNamespace(numel=lambda: 1, item=lambda: 0., device=SimpleNamespace(type='cpu'))
        self.assertEqual(metric._scalar(zero), 0.)

    def test_frame_score_passes_explicit_normalization_and_rejects_shape_change(self):
        adapter = metric.OfflineLPIPS.__new__(metric.OfflineLPIPS)
        adapter.torch = SimpleNamespace(inference_mode=nullcontext, from_numpy=lambda value: value)
        scalar = SimpleNamespace(numel=lambda: 1, item=lambda: .125, device=SimpleNamespace(type='cpu'))
        adapter.model = Mock(return_value=scalar)
        first = np.zeros((32, 40, 3), dtype=np.float32)
        second = first + 1.
        self.assertEqual(adapter._score_frame(first, second, 'zero_one'), .125)
        args, kwargs = adapter.model.call_args
        self.assertEqual(kwargs, {'normalize': False})
        np.testing.assert_array_equal(args[0], -np.ones((1, 3, 32, 40), dtype=np.float32))
        np.testing.assert_array_equal(args[1], np.ones((1, 3, 32, 40), dtype=np.float32))
        adapter.model.reset_mock()
        with self.assertRaisesRegex(metric.OfflineLPIPSError, 'shapes differ'):
            adapter._score_frame(first, second[:, :39], 'zero_one')
        adapter.model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
