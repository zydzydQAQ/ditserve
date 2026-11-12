"""Validate the delivered r3 image on one actual A100 before service trials.

Runs the image's hash-bound CUDA smoke, real cuMem allocation/sleep/wake,
installed dependency consistency and the original GF-DiT worker imports.
A passing result grants runtime readiness only, never benchmark qualification.
"""
from __future__ import annotations
import argparse
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

import external_runtime_probe as runtime

IMAGE_DIGEST = 'sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05'
SMOKE = Path('/opt/ditserve-image-deps/20261104-vllm-cu126-r3/smoke_kernels.py')
SMOKE_SHA256 = 'cd0863afb1f72dccdc544c085ed2ea281ccce70721460dc2ac8fe9ef132c7f13'
MARKER = 'DITSERVE_KERNEL_SMOKE_JSON='
CASE_MARKER = 'DITSERVE_IMAGE_CASE_JSON='
KERNELS = {'torch.cuda_arithmetic', 'vllm._C.rms_norm', 'vllm._C.silu_and_mul',
           'vllm._moe_C.moe_sum', 'vllm.FA2.varlen_attention',
           'flashinfer.single_decode_with_kv_cache'}
GF_ROOT = Path('/shared/data/DITSERVE/external-systems/20261103-source-v1/GF-DiT')
GF_MANIFEST = GF_ROOT.with_name('GF-DiT.source.json')
GF_MANIFEST_SHA256 = 'e1b54cf09241ff70fc27e9d1666a44cbe38e00eaf1713b57cd5c9ade9c9db3f4'


def parse_marked(result, marker):
    lines = result.get('stdout_tail', '').splitlines()
    matches = [line[len(marker):] for line in lines if line.startswith(marker)]
    if len(matches) != 1:
        raise ValueError('missing or duplicate structured case result')
    value = json.loads(matches[0])
    if not isinstance(value, dict):
        raise ValueError('case result must be an object')
    return value


def kernel_pass(result):
    if result.get('returncode') != 0 or result.get('timed_out'):
        return False
    value = parse_marked(result, MARKER)
    checks = value.get('checks', [])
    return (value.get('status') == 'passed' and value.get('check_count') == 6
            and len(checks) == 6 and {x.get('name') for x in checks} == KERNELS
            and all(x.get('status') == 'passed' and x.get('all_finite') is True
                    and x.get('failed_elements') == 0 for x in checks))


def module_identity(module, *, native=False):
    path = Path(module.__file__).resolve(strict=True)
    if native and not isinstance(module.__spec__.loader, importlib.machinery.ExtensionFileLoader):
        raise ValueError('actual native extension required: '+module.__name__)
    return {'path': str(path), 'sha256': runtime.sha(path), 'native': native}


def child(case):
    if case == 'cumem':
        import torch
        import vllm.device_allocator.cumem as cumem
        native = importlib.import_module('vllm.cumem_allocator')
        identity = module_identity(native, native=True)
        if not cumem.cumem_available:
            raise RuntimeError('real cuMem allocator unavailable')
        torch.cuda.set_device(0)
        allocator = cumem.CuMemAllocator.get_instance()
        tag = 'ditserve_image_probe'
        usage_before = allocator.get_current_usage()
        expected = torch.arange(4096, device='cpu', dtype=torch.float32)
        with allocator.use_memory_pool(tag=tag):
            tensor = torch.arange(4096, device='cuda', dtype=torch.float32)
        torch.cuda.synchronize()
        pointer, tensor_bytes = tensor.data_ptr(), tensor.numel() * tensor.element_size()
        # PyTorch may suballocate inside the native cuMem segment. Bind the
        # complete tensor range to its unique real allocation, not only a tag.
        matches = [(base, data) for base, data in allocator.pointer_to_data.items()
                   if base <= pointer and pointer + tensor_bytes <= base + data.handle[1]]
        if len(matches) != 1:
            raise ValueError('tensor is not uniquely tracked by native cuMem')
        allocation_base, allocation = matches[0]
        handle = tuple(allocation.handle)
        usage_allocated = allocator.get_current_usage()
        if (allocation.tag != tag or handle[2] != allocation_base or handle[0] != 0
                or allocation.cpu_backup_tensor is not None
                or usage_allocated - usage_before < handle[1]):
            raise ValueError('cuMem tracked allocation identity or usage differs')
        if not torch.equal(tensor.cpu(), expected):
            raise ValueError('cuMem allocation values differ')
        allocator.sleep(offload_tags=(tag,))
        backup = allocation.cpu_backup_tensor
        if (allocator.pointer_to_data.get(allocation_base) is not allocation
                or backup is None or backup.device.type != 'cpu'
                or backup.dtype != torch.uint8 or backup.numel() != handle[1]):
            raise ValueError('cuMem sleep did not retain the complete tracked CPU backup')
        backup_bytes = backup.numel() * backup.element_size()
        usage_asleep = allocator.get_current_usage()
        allocator.wake_up(tags=[tag])
        torch.cuda.synchronize()
        if (allocator.pointer_to_data.get(allocation_base) is not allocation
                or tuple(allocation.handle) != handle or allocation.cpu_backup_tensor is not None):
            raise ValueError('cuMem wake did not restore the same tracked allocation')
        if not torch.equal(tensor.cpu(), expected):
            raise ValueError('cuMem sleep/wake did not preserve values')
        return {'status': 'passed', 'elements': 4096, 'sleep_wake_exact': True,
                'native_extension': identity, 'allocation_tracked': True,
                'allocation_tag': tag, 'tensor_pointer': pointer, 'tensor_bytes': tensor_bytes,
                'allocation_base': allocation_base, 'allocation_bytes': handle[1],
                'allocation_device': handle[0], 'cpu_backup_bytes': backup_bytes,
                'cpu_backup_observed_after_sleep': True, 'cpu_backup_cleared_after_wake': True,
                'usage_bytes': {'before': usage_before, 'allocated': usage_allocated,
                                'asleep': usage_asleep, 'awake': allocator.get_current_usage()},
                'usage_scope': 'Tracked reserved bytes, not physical resident bytes while asleep.'}
    if case == 'gfdit_imports':
        if runtime.sha(GF_MANIFEST) != GF_MANIFEST_SHA256:
            raise ValueError('official source manifest SHA256 differs')
        manifest = json.loads(GF_MANIFEST.read_text(encoding='utf-8'))
        sys.path.insert(0, str(GF_ROOT))
        names = [
            'vllm', 'vllm_omni.diffusion.runtime_v2.runner',
            'vllm_omni.diffusion.runtime_v2.multiproc_worker',
            'vllm_omni.diffusion.worker.diffusion_worker',
            'vllm_omni.diffusion.distributed.collective_runtime',
        ]
        identities = {}
        for name in names:
            module = importlib.import_module(name)
            identities[name] = module_identity(module)
            if name.startswith('vllm_omni.') and not Path(module.__file__).resolve().is_relative_to(GF_ROOT):
                raise ValueError('original GF-DiT module shadowed: '+name)
        return {'status': 'passed', 'official_commit': manifest['commit'],
                'source_files': len(manifest['files']), 'modules': identities}
    raise ValueError('unknown validation case')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['cumem', 'gfdit_imports'])
    args = parser.parse_args()
    if args.case:
        try:
            value = child(args.case)
        except Exception as error:
            traceback.print_exc()
            print(CASE_MARKER+json.dumps({'status': 'failed', 'error_type': type(error).__name__,
                                        'error': str(error)[-4000:]}), flush=True)
            raise SystemExit(1)
        print(CASE_MARKER+json.dumps(value, allow_nan=False), flush=True)
        return
    storage_before = runtime.storage()
    root, shared, plan, binding = runtime.bind_sources(
        'external_image_validation.py', expected_image_digest=IMAGE_DIGEST)
    print('build=51740c0 core='+binding['core_sha256']+' image='+binding['image'], flush=True)
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('image validation requires exactly one actual visible CUDA device')
    if plan != {'schema': 'ditserve.external-image-validation.v1', 'image_digest': IMAGE_DIGEST}:
        raise ValueError('unexpected image validation plan')
    if runtime.sha(SMOKE) != SMOKE_SHA256:
        raise ValueError('delivered image kernel smoke differs')
    output = runtime.OUTPUT/'external-probes'/os.environ['DITSERVE_RUN_ID']
    output.mkdir(parents=True, exist_ok=False)
    os.umask(0o022)
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(root), str(shared)]), PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='1', MAX_JOBS='2', CUDA_VISIBLE_DEVICES='0')
    for key, suffix in {'TMPDIR':'tmp', 'HF_HOME':'hf', 'XDG_CACHE_HOME':'xdg',
        'TORCH_HOME':'torch', 'TORCH_EXTENSIONS_DIR':'extensions',
        'TORCHINDUCTOR_CACHE_DIR':'inductor', 'TRITON_CACHE_DIR':'triton',
        'FLASHINFER_WORKSPACE_BASE':'flashinfer', 'FLASHINFER_CUBIN_DIR':'cubins',
        'MPLCONFIGDIR':'matplotlib'}.items():
        path = output/'cache'/suffix
        path.mkdir(parents=True, exist_ok=False)
        env[key] = str(path)
    versions = {}
    for name in runtime.PACKAGES:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    cases = [
        ('pip_check', [sys.executable, '-m', 'pip', 'check'], 120),
        ('native_kernels', [sys.executable, '-B', str(SMOKE), '--device', '0'], 900),
        ('cumem', [sys.executable, '-B', str(root/'external_image_validation.py'), '--case', 'cumem'], 180),
        ('gfdit_imports', [sys.executable, '-B', str(root/'external_image_validation.py'), '--case', 'gfdit_imports'], 180),
    ]
    results = []
    for name, argv, timeout in cases:
        runtime.storage()
        item = runtime.run_command(argv, timeout, env)
        item['name'] = name
        try:
            item['passed'] = item['returncode'] == 0 and not item['timed_out']
            if name == 'native_kernels':
                item['passed'] = kernel_pass(item)
                item['structured'] = parse_marked(item, MARKER)
            elif name != 'pip_check':
                item['structured'] = parse_marked(item, CASE_MARKER)
                item['passed'] = item['passed'] and item['structured'].get('status') == 'passed'
        except (ValueError, KeyError, TypeError) as error:
            item['passed'] = False
            item['validation_error'] = str(error)
        results.append(item)
        print(json.dumps({'case': name, 'passed': item['passed'], 'returncode': item['returncode']}), flush=True)
    result = {'schema': 'ditserve.external-image-validation-result.v1',
        'run_id': os.environ['DITSERVE_RUN_ID'], 'binding': binding, 'packages': versions,
        'image_kernel_smoke_sha256': SMOKE_SHA256, 'cases': results,
        'status': 'passed' if all(x['passed'] for x in results) else 'failed',
        'storage_before': storage_before, 'storage_after': runtime.storage(),
        'qualification_eligible': False, 'performance_measurements': False,
        'scope': 'New-image kernels, allocator and original worker import readiness only; full service and models pending.'}
    target = output/'summary.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    target.chmod(0o644)
    print(json.dumps({'summary': str(target), 'sha256': runtime.sha(target), 'status': result['status']}), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
