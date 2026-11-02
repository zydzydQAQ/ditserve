#!/usr/bin/env python3
"""Strict, CPU-only LPIPS-Alex v0.1 for the existing DitServe image.

No installation, download, GPU work, artifact deletion or P4 qualification.
Dependencies are lazy: --help and unavailable-dependency receipts need no torch.
Use in a dedicated, single Python-thread process. The Python network/process
guard prevents accidental library downloads; it is not an OS security sandbox.
After selftest(), score_frame() reads one RGB HWC float frame pair at a time.
Callers still own source/prompt/seed binding, missing denominators and statistics.
"""

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import ssl
import stat
import subprocess
import sys
import threading
import urllib.request
from unittest.mock import patch

SCHEMA = 'ditserve.p4-offline-lpips.v1'
CHUNK_BYTES = 256 * 1024
PACKAGES = {'torch': '2.10.0+cu126', 'torchvision': '0.25.0+cu126', 'lpips': '0.1.4'}
DEFAULT_BACKBONE = '/shared/data/DITSERVE/models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'


class OfflineLPIPSError(ValueError):
    """Unavailable or unverified metric, never a successful qualification."""


@dataclass(frozen=True)
class FileIdentity:
    size: int
    sha256: str


BACKBONE = FileIdentity(244408911, '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02')
HEAD = FileIdentity(6009, 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0')
SOURCES = {
    ('lpips', 'lpips/__init__.py'): FileIdentity(4846, '36ee004a45e2cc2c5ab47fa3955f00c0e41f6a151ef0249fb6f362533ed58b22'),
    ('lpips', 'lpips/lpips.py'): FileIdentity(10101, '780d09b907cb9b661e0ae28b2d163ddfba92f9e870d7feba34d4790cc6590658'),
    ('lpips', 'lpips/pretrained_networks.py'): FileIdentity(6507, '6a27f714c51796db466e86bebba6a617c1bfc4d566f3a1756497629c1248686e'),
    ('torchvision', 'torchvision/models/alexnet.py'): FileIdentity(4488, '76f0592d51fad133931c234d9168344c52edb89e46861ff15cdef7af7572ab9f'),
}
CONVS = ((0, (64, 3, 11, 11)), (3, (192, 64, 5, 5)), (6, (384, 192, 3, 3)),
         (8, (256, 384, 3, 3)), (10, (256, 256, 3, 3)))
BACKBONE_SHAPES = {f'features.{i}.{suffix}': shape if suffix == 'weight' else (shape[0],)
                   for i, shape in CONVS for suffix in ('weight', 'bias')}
BACKBONE_SHAPES.update({f'classifier.{i}.{suffix}': shape if suffix == 'weight' else (shape[0],)
                        for i, shape in ((1, (4096, 9216)), (4, (4096, 4096)), (6, (1000, 4096)))
                        for suffix in ('weight', 'bias')})
HEAD_SHAPES = {f'lin{i}.model.1.weight': (1, channels, 1, 1)
               for i, channels in enumerate((64, 192, 384, 256, 256))}
TRUNK_MAPPING = {f'features.{index}.{suffix}': f'slice{layer}.{index}.{suffix}'
                 for layer, (index, _) in enumerate(CONVS, 1) for suffix in ('weight', 'bias')}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _fingerprint(stream):
    value = os.fstat(stream.fileno())
    if not stat.S_ISREG(value.st_mode):
        raise OfflineLPIPSError('a regular local file is required')
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@contextmanager
def verified_file(path, expected):
    """Hash and deserialize through the same read-only descriptor; detect edits."""
    if not isinstance(path, (str, Path)) or not str(path) or '://' in str(path):
        raise OfflineLPIPSError('an explicit local file path is required')
    # O_NONBLOCK lets fstat reject a FIFO without hanging on its initial open.
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        before = _fingerprint(stream)
        if before[2] != expected.size:
            raise OfflineLPIPSError('file size differs: ' + str(path))
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(CHUNK_BYTES), b''):
            digest.update(block)
        if digest.hexdigest() != expected.sha256 or _fingerprint(stream) != before:
            raise OfflineLPIPSError('file SHA256 or stable identity differs: ' + str(path))
        stream.seek(0)
        yield stream, dict(path=str(Path(path).resolve()), bytes=before[2], sha256=digest.hexdigest())
        if _fingerprint(stream) != before:
            raise OfflineLPIPSError('file changed during use: ' + str(path))


@contextmanager
def offline_guard(torch=None):
    """Fail even if a dependency catches a denied networking/download attempt."""
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise OfflineLPIPSError('offline guard requires a dedicated single Python-thread process')
    attempts = []

    def denied(*args, **kwargs):
        attempts.append('blocked')
        raise OfflineLPIPSError('network, download and child-process entrypoints are disabled')

    targets = [(socket, name) for name in ('create_connection', 'getaddrinfo', 'gethostbyname', 'gethostbyname_ex')]
    operations = ('connect', 'connect_ex', 'send', 'sendall', 'sendto', 'sendmsg',
                  'recv', 'recv_into', 'recvfrom', 'recvfrom_into', 'recvmsg', 'recvmsg_into')
    targets += [(socket.socket, name) for name in operations if hasattr(socket.socket, name)]
    # TLS methods can bypass socket.socket.send/recv on an existing connection.
    targets += [(ssl.SSLSocket, name) for name in operations + ('read', 'write') if hasattr(ssl.SSLSocket, name)]
    targets += [(urllib.request, 'urlopen'), (urllib.request, 'urlretrieve'),
                (urllib.request.OpenerDirector, 'open'), (subprocess, 'Popen'), (os, 'system')]
    targets += [(os, name) for name in ('posix_spawn', 'posix_spawnp') if hasattr(os, name)]
    if torch is not None:
        targets += [(torch.hub, 'load_state_dict_from_url'), (torch.hub, 'download_url_to_file')]
        for name in ('torchvision.models._api', 'torchvision._internally_replaced_utils'):
            module = sys.modules.get(name)
            if module is not None and hasattr(module, 'load_state_dict_from_url'):
                targets.append((module, 'load_state_dict_from_url'))
    with ExitStack() as stack:
        for target, name in targets:
            stack.enter_context(patch.object(target, name, denied))
        try:
            yield
        finally:
            if attempts:
                raise OfflineLPIPSError('offline execution rejected a network/download/process attempt')


def _load_dependencies():
    identities, sources = {}, {}
    for name, expected in PACKAGES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise OfflineLPIPSError('required package unavailable: ' + name) from error
        if actual != expected:
            raise OfflineLPIPSError(f'package identity differs: {name}={actual}, expected {expected}')
        identities[name] = actual
    for (distribution, member), expected in SOURCES.items():
        path = importlib.metadata.distribution(distribution).locate_file(member)
        with verified_file(path, expected) as (_, receipt):
            sources[member] = receipt
    torch, torchvision, lpips = (importlib.import_module(name) for name in PACKAGES)
    for name, module in (('torch', torch), ('torchvision', torchvision)):
        if str(module.__version__) != PACKAGES[name]:
            raise OfflineLPIPSError('imported module and distribution disagree: ' + name)
    for name, member in (('lpips', 'lpips/__init__.py'), ('lpips.lpips', 'lpips/lpips.py'),
                         ('lpips.pretrained_networks', 'lpips/pretrained_networks.py'),
                         ('torchvision.models.alexnet', 'torchvision/models/alexnet.py')):
        module = importlib.import_module(name)
        if str(Path(module.__file__).resolve()) != sources[member]['path']:
            raise OfflineLPIPSError('imported source differs: ' + name)
    for (distribution, member), expected in SOURCES.items():
        with verified_file(sources[member]['path'], expected):
            pass  # Reject sources changed while the imports executed.
    # These versions are evidence, while the metric/model APIs above are pinned.
    for name in ('numpy', 'scipy', 'Pillow', 'tqdm'):
        identities[name] = importlib.metadata.version(name)
    return torch, torchvision, lpips, dict(packages=identities, sources=sources, python=platform.python_version())


def _validate_state(state, shapes, torch):
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise OfflineLPIPSError('checkpoint keys are incomplete or unexpected')
    for name, shape in shapes.items():
        value = state[name]
        if (not torch.is_tensor(value) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.device.type != 'cpu'
                or not torch.isfinite(value).all().item()):
            raise OfflineLPIPSError('checkpoint tensor shape/dtype/device/finite check failed: ' + name)
    return state


def _checkpoint(torch, path, expected, shapes):
    with verified_file(path, expected) as (stream, receipt):
        state = torch.load(stream, map_location='cpu', weights_only=True)
        _validate_state(state, shapes, torch)
    return state, receipt


def _install_weights(model, backbone, heads, torch):
    _validate_state(backbone, BACKBONE_SHAPES, torch)
    _validate_state(heads, HEAD_SHAPES, torch)
    trunk = {destination: backbone[source] for source, destination in TRUNK_MAPPING.items()}
    if set(model.net.state_dict()) != set(trunk):
        raise OfflineLPIPSError('LPIPS trunk architecture differs')
    model.net.load_state_dict(trunk, strict=True)
    for i in range(5):
        layer = getattr(model, f'lin{i}')
        if model.lins[i] is not layer or set(layer.state_dict()) != {'model.1.weight'}:
            raise OfflineLPIPSError('LPIPS calibrated head architecture differs')
        layer.load_state_dict({'model.1.weight': heads[f'lin{i}.model.1.weight']}, strict=True)
    expected_parameters = {'net.' + value for value in TRUNK_MAPPING.values()} | set(HEAD_SHAPES)
    if {name for name, _ in model.named_parameters()} != expected_parameters:
        raise OfflineLPIPSError('unaccounted or missing LPIPS parameters')


def _construct_model(torch, lpips):
    with torch.device('cpu'):
        return lpips.LPIPS(net='alex', version='0.1', pretrained=False, pnet_rand=True,
            pnet_tune=False, lpips=True, spatial=False, use_dropout=True,
            eval_mode=True, verbose=False).to(device='cpu', dtype=torch.float32)


def prepare_frame(frame, input_range='zero_one'):
    """Explicit HWC RGB float -> CPU NCHW float32 [-1,1], without cropping."""
    import numpy as np
    value = np.asarray(frame)
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 32:
        raise OfflineLPIPSError('frame must be HWC RGB, height and width at least 32')
    if value.dtype.kind != 'f' or not np.isfinite(value).all():
        raise OfflineLPIPSError('frame must contain finite floating-point RGB values')
    bounds = {'zero_one': (0., 1.), 'minus_one_one': (-1., 1.)}
    if input_range not in bounds:
        raise OfflineLPIPSError('input_range must be explicit zero_one or minus_one_one')
    low, high = bounds[input_range]
    if value.min() < low or value.max() > high:
        raise OfflineLPIPSError('frame values are outside the declared input range')
    value = np.array(value, dtype=np.float32, copy=True)
    if input_range == 'zero_one':
        value *= 2.
        value -= 1.
    return np.ascontiguousarray(value.transpose(2, 0, 1)[None])


def _scalar(value):
    if value.numel() != 1 or value.device.type != 'cpu':
        raise OfflineLPIPSError('metric must return one CPU scalar')
    result = float(value.item())
    if not math.isfinite(result) or result < 0:
        raise OfflineLPIPSError('metric returned a nonfinite or negative distance')
    return result


class OfflineLPIPS:
    def __init__(self, backbone_path=DEFAULT_BACKBONE, head_path=None):
        self._selftest = None
        with offline_guard():
            self.torch, self.tv, lpips, identity = _load_dependencies()
            self.backbone_path = backbone_path
            self.head_path = (head_path if head_path is not None else
                importlib.metadata.distribution('lpips').locate_file('lpips/weights/v0.1/alex.pth'))
            with offline_guard(self.torch):
                backbone, backbone_receipt = _checkpoint(self.torch, self.backbone_path, BACKBONE, BACKBONE_SHAPES)
                heads, head_receipt = _checkpoint(self.torch, self.head_path, HEAD, HEAD_SHAPES)
                self.model = _construct_model(self.torch, lpips)
                _install_weights(self.model, backbone, heads, self.torch)
                self.model.eval().requires_grad_(False)
                for value in list(self.model.parameters()) + list(self.model.buffers()):
                    if value.device.type != 'cpu' or value.dtype != self.torch.float32 or value.requires_grad:
                        raise OfflineLPIPSError('model must be frozen float32 on CPU')
                if any(module.training for module in self.model.modules()):
                    raise OfflineLPIPSError('model contains training-mode modules')
        self._identity = dict(schema=SCHEMA, metric='lpips-alex-v0.1', device='cpu', runtime=identity,
            weights=dict(backbone=backbone_receipt, calibrated_head=head_receipt),
            provenance=dict(backbone_full_sha256='observed existing project checkpoint; upstream publishes 7be5be79 prefix',
                backbone_url='https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
                head_official_revision='082bb24f84c091ea94de2867d34c4544f68e0963',
                head_git_blob_sha1='1df9dfe62abb1fc89cc7f82b4e5fe886c979708e'),
            preprocessing='RGB HWC float, explicit range -> NCHW float32 [-1,1]; no resize/crop',
            guard='Python socket/urllib/download/child-process denial; dedicated process, not OS sandbox',
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def receipt(self):
        result = dict(self._identity, status='ready' if self._selftest else 'selftest_required',
            selftest=self._selftest, p4_qualification_eligible=False, retention_dependencies_closed=False,
            vbench={'status': 'unknown'})
        result['sha256'] = canonical_hash(result)
        return json.loads(json.dumps(result, allow_nan=False))

    def _score_frame(self, reference, candidate, input_range):
        first, second = prepare_frame(reference, input_range), prepare_frame(candidate, input_range)
        if first.shape != second.shape:
            raise OfflineLPIPSError('paired frame shapes differ')
        torch = self.torch
        with torch.inference_mode():
            return _scalar(self.model(torch.from_numpy(first), torch.from_numpy(second), normalize=False))

    def score_frame(self, reference, candidate, *, input_range='zero_one'):
        if self._selftest is None:
            raise OfflineLPIPSError('real checkpoint CPU selftest must pass before scoring')
        with offline_guard(self.torch):
            return self._score_frame(reference, candidate, input_range)

    def selftest(self):
        """Real weights only: independent full torchvision trunk plus LPIPS math."""
        import numpy as np
        self._selftest = None
        torch = self.torch
        with offline_guard(torch), torch.inference_mode():
            pixels = np.arange(64 * 64 * 3, dtype=np.float32).reshape(64, 64, 3)
            first = (pixels % 251) / np.float32(250.)
            second = np.ascontiguousarray(np.roll(first, 3, axis=1))
            exact = self._score_frame(first, first, 'zero_one')
            actual = self._score_frame(first, second, 'zero_one')
            reverse = self._score_frame(second, first, 'zero_one')
            if exact != 0. or actual <= 0. or not math.isclose(actual, reverse, rel_tol=1e-6, abs_tol=1e-8):
                raise OfflineLPIPSError('identity/perturbation/symmetry CPU selftest failed')
            # A separate full AlexNet strict load independently tests slice mapping.
            state, _ = _checkpoint(torch, self.backbone_path, BACKBONE, BACKBONE_SHAPES)
            with torch.device('cpu'):
                reference = self.tv.models.alexnet(weights=None).to(device='cpu', dtype=torch.float32)
            reference.load_state_dict(state, strict=True)
            reference.eval().requires_grad_(False)
            features = reference.features
            del state, reference  # Discard the unused classifier before frame evaluation.
            heads, _ = _checkpoint(torch, self.head_path, HEAD, HEAD_SHAPES)
            left, right = (torch.from_numpy(prepare_frame(frame)) for frame in (first, second))
            shift = torch.tensor([-.030, -.088, -.188], device='cpu', dtype=torch.float32).reshape(1, 3, 1, 1)
            scale = torch.tensor([.458, .448, .450], device='cpu', dtype=torch.float32).reshape(1, 3, 1, 1)
            left, right = (left - shift) / scale, (right - shift) / scale
            expected, head_index = 0., 0
            for index, layer in enumerate(features):
                if index > 11:
                    break
                left, right = layer(left), layer(right)
                if index in (1, 4, 7, 9, 11):
                    normalize = lambda value: value / (torch.sqrt((value ** 2).sum(dim=1, keepdim=True)) + 1e-10)
                    difference = (normalize(left) - normalize(right)) ** 2
                    term = torch.nn.functional.conv2d(difference, heads[f'lin{head_index}.model.1.weight']).mean()
                    expected += float(term.item())
                    head_index += 1
            if head_index != 5 or not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-7):
                raise OfflineLPIPSError('independent torchvision/LPIPS computation differs')
            result = dict(status='passed_real_cpu_weights', exact_distance=exact, perturbed_distance=actual,
                symmetric_distance=reverse, independent_reference_distance=expected,
                input_sha256=hashlib.sha256(first.tobytes() + second.tobytes()).hexdigest(),
                network_attempts=0, reference='full torchvision strict checkpoint + independent LPIPS v0.1 math')
        self._selftest = result  # Guard must also exit successfully before enabling scores.
        return self.receipt()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['selftest'])
    parser.add_argument('--backbone', default=DEFAULT_BACKBONE)
    parser.add_argument('--head', default=None)
    args = parser.parse_args(argv)
    try:
        result = OfflineLPIPS(args.backbone, args.head).selftest()
        code = 0
    except (OfflineLPIPSError, ImportError, OSError, RuntimeError) as error:
        result = dict(schema=SCHEMA, status='unknown', reason=f'{type(error).__name__}: {error}',
            p4_qualification_eligible=False, retention_dependencies_closed=False, vbench={'status': 'unknown'})
        code = 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
