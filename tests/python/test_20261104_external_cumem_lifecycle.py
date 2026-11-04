"""CPU-only ownership contracts; no cuMem/GPU execution or native exit claim."""
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import weakref

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import external_cumem_lifecycle as lifecycle


class Allocator:
    def __init__(self):
        self.pointer_to_data = {123:2097152}
        self.allocator_and_pools = {}
    def get_current_usage(self):
        return sum(self.pointer_to_data.values())


def fixture(*, live=False, reclaim=True):
    owner = Allocator(); log=[]
    class Wrapper:
        def __del__(self): log.append('wrapper_destroyed')
    class Pool:
        def __init__(self,wrapper): self.wrapper=weakref.ref(wrapper)
        def snapshot(self): return [dict(allocated_size=16384 if live else 0)]
        def __del__(self):
            log.append('pool_destroyed_with_wrapper' if self.wrapper() is not None else 'pool_destroyed_without_wrapper')
            if reclaim: owner.pointer_to_data.pop(123,None)
    wrapper=Wrapper()
    owner.allocator_and_pools[lifecycle.TAG]=(Pool(wrapper),wrapper)
    return owner,log


class CumemLifecycleTests(unittest.TestCase):
    def test_real_ownership_order_keeps_wrapper_alive_until_pool_destruction(self):
        owner,log=fixture(); stages=[]
        result=lifecycle.release_owned_pools(owner,phase=stages.append)
        self.assertEqual(log,['pool_destroyed_with_wrapper','wrapper_destroyed'])
        self.assertEqual(owner.pointer_to_data,{})
        self.assertEqual(owner.allocator_and_pools,{})
        self.assertTrue(result['native_free_callback_observed'])
        self.assertEqual(result['usage_before_bytes'],2097152)
        self.assertEqual(result['usage_after_bytes'],0)
        self.assertEqual(stages,['release_mempools_keep_allocator_wrappers',
                                 'native_allocation_reclaimed_release_allocator_wrappers'])
        self.assertFalse(result['package_sources_modified'])

    def test_live_tensor_and_foreign_pool_are_rejected_before_mutation(self):
        for mutation in ('live','foreign','layout','empty_snapshot','extra_allocation'):
            owner,log=fixture(live=mutation=='live')
            if mutation=='foreign': owner.allocator_and_pools['model_weights']=object()
            if mutation=='layout': owner.allocator_and_pools[lifecycle.TAG]=list(owner.allocator_and_pools[lifecycle.TAG])
            if mutation=='empty_snapshot': owner.allocator_and_pools[lifecycle.TAG][0].snapshot=lambda:[]
            if mutation=='extra_allocation': owner.pointer_to_data[456]=2097152
            before=dict(owner.allocator_and_pools)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                lifecycle.release_owned_pools(owner,phase=lambda stage:None)
            self.assertEqual(owner.allocator_and_pools,before)
            self.assertEqual(log,[])
            # CPU-only fixture references cleaned after assertions.
            owner.allocator_and_pools.clear();before.clear();gc.collect()

    def test_failure_to_run_native_free_callback_cannot_manufacture_zero_usage(self):
        owner,log=fixture(reclaim=False)
        with self.assertRaisesRegex(ValueError,'did not reclaim'):
            lifecycle.release_owned_pools(owner,phase=lambda stage:None)
        self.assertEqual(owner.pointer_to_data,{123:2097152})
        self.assertEqual(log[0],'pool_destroyed_with_wrapper')

    def test_allocators_stay_alive_during_gc_phase(self):
        owner,log=fixture(); observed=[]
        def collect():
            observed.append(tuple(log));return gc.collect()
        lifecycle.release_owned_pools(owner,collect=collect,phase=lambda stage:None)
        self.assertEqual(observed[0],('pool_destroyed_with_wrapper',))
        self.assertEqual(observed[1],('pool_destroyed_with_wrapper','wrapper_destroyed'))

    def test_pinned_sources_and_exit_scope_do_not_modify_original_probe(self):
        self.assertEqual(lifecycle.CUMEM_PY_SHA256,'e3301b98d8d9c6438584afffc8a2ea2d4b8233894c191ab10c3c25799d618512')
        self.assertEqual(lifecycle.CUMEM_NATIVE_SHA256,'49b6f9e4cc443732727aa842e124309b57e7e361d53a3091d0da96a6f5a48c96')
        self.assertEqual(lifecycle.original.IMAGE_DIGEST,'sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05')


if __name__=='__main__': unittest.main()
