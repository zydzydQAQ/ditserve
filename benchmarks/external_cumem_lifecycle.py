"""Diagnose r3 cuMem shutdown using explicit pool-before-allocator destruction.

The original r1 completed allocation/sleep/wake checks, printed its receipt and
then exited with SIGSEGV. A pool/allocator finalization-order defect is a strong
hypothesis, not an established native stack diagnosis. This new case retains
that failed evidence and changes only call-site resource cleanup. It neither
patches installed packages nor bypasses interpreter shutdown with os._exit.

Current upstream describes deterministic pool-before-wrapper release:
https://github.com/vllm-project/vllm/blob/main/vllm/device_allocator/cumem.py
The delivered v0.17 source below is pinned independently; upstream main is only
rationale for the cleanup experiment, never substituted as the executed source.
A child summary alone cannot pass this diagnostic: the outer process must also
observe normal exit 0. GF imports/full-image readiness remain separate gates.
"""
from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
from pathlib import Path
import time
import traceback

import external_image_validation as original
import external_runtime_probe as runtime

CUMEM_PY_SHA256 = 'e3301b98d8d9c6438584afffc8a2ea2d4b8233894c191ab10c3c25799d618512'
CUMEM_NATIVE_SHA256 = '49b6f9e4cc443732727aa842e124309b57e7e361d53a3091d0da96a6f5a48c96'
TAG = 'ditserve_image_probe'


def mark(stage):
    print('DITSERVE_CUMEM_STAGE=' + json.dumps(dict(stage=stage, monotonic_ns=time.monotonic_ns())), flush=True)


def release_owned_pools(allocator, *, collect=gc.collect, phase=mark):
    """Release this probe's inactive pool while wrappers/callbacks remain alive.

The delivered API keeps one (MemPool, CUDAPluggableAllocator) tuple per tag.
No allocations or dictionary entries are manually unmap/freed: the real MemPool
and its registered native callback own reclamation. Unexpected/live allocations
fail closed. This helper does not support model-owned pools or CUDA graphs.
"""
    kept = allocator.allocator_and_pools
    if not isinstance(kept, dict) or set(kept) != {TAG}:
        raise ValueError('only the dedicated probe pool may be released')
    entries = list(kept.values())
    if any(type(entry) is not tuple or len(entry) != 2 for entry in entries):
        raise ValueError('delivered cuMem pool ownership layout differs')
    pools = [entry[0] for entry in entries]
    wrappers = [entry[1] for entry in entries]
    snapshots = [pool.snapshot() for pool in pools]
    if any(not isinstance(snapshot, list) or not snapshot for snapshot in snapshots):
        raise ValueError('probe pool allocation snapshot is missing')
    for snapshot in snapshots:
        for segment in snapshot:
            if type(segment.get('allocated_size')) is not int or segment['allocated_size'] != 0:
                raise ValueError('probe still owns a live tensor allocation')
    before = allocator.get_current_usage()
    tracked_before = len(allocator.pointer_to_data)
    if type(before) is not int or before <= 0 or tracked_before != 1:
        raise ValueError('expected the one tracked native probe allocation before cleanup')
    phase('release_mempools_keep_allocator_wrappers')
    kept.clear()
    entries.clear()
    pools.clear()
    collect()
    # Native my_free must have called the real Python free callback and retired
    # the tracking entry. Never clear pointer_to_data to manufacture this proof.
    after = allocator.get_current_usage()
    if allocator.pointer_to_data or after != 0:
        raise ValueError('native MemPool teardown did not reclaim the tracked allocation')
    phase('native_allocation_reclaimed_release_allocator_wrappers')
    wrappers.clear()
    collect()
    if allocator.allocator_and_pools or allocator.pointer_to_data or allocator.get_current_usage() != 0:
        raise ValueError('allocator state changed after wrapper release')
    return dict(pool_count=len(snapshots), tracked_allocations_before=tracked_before,
                tracked_allocations_after=0, usage_before_bytes=before, usage_after_bytes=0,
                inactive_segments_before=sum(len(snapshot) for snapshot in snapshots),
                pool_released_before_allocator=True, native_free_callback_observed=True,
                package_sources_modified=False)


def execute(result):
    mark('import_actual_r3_allocator')
    import torch
    import vllm.device_allocator.cumem as cumem
    if torch.cuda.device_count() != 1 or torch.__version__ != '2.10.0+cu126':
        raise ValueError('requires the delivered Torch2.10/cu126 image and one actual GPU')
    if runtime.sha(Path(cumem.__file__)) != CUMEM_PY_SHA256:
        raise ValueError('delivered cuMem Python source differs')
    result['allocator_source_sha256'] = CUMEM_PY_SHA256
    result['stage'] = 'original_allocation_sleep_wake'
    mark(result['stage'])
    # All tensors created by the unchanged child are local to that function.
    operations = original.child('cumem')
    if (operations.get('status') != 'passed' or operations.get('sleep_wake_exact') is not True
            or operations.get('native_extension', {}).get('sha256') != CUMEM_NATIVE_SHA256):
        raise ValueError('original native allocator check did not pass on the pinned extension')
    result['operations'] = operations
    result['stage'] = 'drop_tensor_references_and_synchronize'
    mark(result['stage'])
    gc.collect()
    torch.cuda.synchronize()
    allocator = cumem.CuMemAllocator.get_instance()
    result['stage'] = 'explicit_two_phase_cleanup'
    result['cleanup'] = release_owned_pools(allocator)
    torch.cuda.synchronize()
    result.update(status='cleanup_completed', stage='normal_interpreter_exit_pending',
                  normal_process_exit_requires_outer_receipt=True)
    mark(result['stage'])


def main(argv=None):
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    initial_storage = runtime.storage()
    root, shared, plan, binding = runtime.bind_sources(
        'external_task_runner.py', 1, expected_image_digest=original.IMAGE_DIGEST)
    if (Path(__file__).resolve() != root/'external_cumem_lifecycle.py'
            or Path(original.__file__).resolve() != root/'external_image_validation.py'):
        raise ValueError('lifecycle and original check must use the same bound package')
    expected = runtime.OUTPUT/'external-tasks'/os.environ['DITSERVE_RUN_ID']/'result'
    output = args.output_dir.resolve()
    if output != expected or output.exists():
        raise ValueError('requires a new result directory owned by this wrapper job')
    output.mkdir(parents=True, exist_ok=False)
    print('build=51740c0 core='+binding['core_sha256']+' image='+binding['image'], flush=True)
    result = dict(schema='ditserve.external-cumem-lifecycle.v1', status='failed', stage='source_bound',
        binding=binding, initial_storage=initial_storage, qualification_eligible=False, allows=[],
        full_image_validation_passed=False, performance_measurements=False,
        hypothesis='late pool/allocator finalization order; no native fault stack from original r1',
        scope='allocation/sleep/wake plus explicit native pool cleanup only; normal process exit requires the outer receipt; GF dependencies remain unvalidated')
    code = 1
    try:
        execute(result)
        code = 0
    except Exception as exc:
        result.update(status='failed', error=type(exc).__name__+': '+str(exc), failed_stage=result['stage'])
        traceback.print_exc()
    result['final_storage'] = runtime.storage()
    target = output/'summary.json'
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(summary=str(target), sha256=runtime.sha(target), status=result['status'],
                         normal_exit_not_yet_observed=True)), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
