"""A runtime pass requires every native kernel, never a partial successful log."""
import copy
import contextlib
import types
from unittest import mock
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'))
import external_image_validation as validation
import external_runtime_probe as runtime


def success():
    value={'status':'passed','check_count':6,'checks':[
        {'name':name,'status':'passed','all_finite':True,'failed_elements':0}
        for name in sorted(validation.KERNELS)]}
    return {'returncode':0,'timed_out':False,'stdout_tail':'other import output\n'+validation.MARKER+json.dumps(value)}


@contextlib.contextmanager
def fake_cumem_contract(mode='ok', offset=0):
    """CPU-only call/receipt contract. Never represents CUDA execution evidence."""
    torch = types.ModuleType('torch')
    torch.float32, torch.uint8 = 'float32', 'uint8'
    torch.cuda = types.SimpleNamespace(set_device=lambda index: None, synchronize=lambda: None)
    allocator = types.SimpleNamespace(pointer_to_data={})
    base, allocation_bytes = 100000, 65536
    expected_tag = 'ditserve_image_probe'

    class Tensor:
        def data_ptr(self): return base + offset
        def numel(self): return 4096
        def element_size(self): return 4
        def cpu(self): return self

    torch.arange = lambda count, *, device, dtype: Tensor()
    calls = []
    def equal(left, right):
        calls.append(True)
        return not (mode == 'wrong_values' and len(calls) == 2)
    torch.equal = equal

    @contextlib.contextmanager
    def pool(*, tag):
        if mode != 'untracked':
            allocator.pointer_to_data[base] = types.SimpleNamespace(
                handle=(1 if mode == 'wrong_device' else 0, allocation_bytes, base, 7),
                tag='different' if mode == 'wrong_tag' else tag, cpu_backup_tensor=None)
        yield
    allocator.use_memory_pool = pool
    allocator.get_current_usage = lambda: (0 if mode == 'zero_usage' else
        sum(data.handle[1] for data in allocator.pointer_to_data.values()))
    def sleep(*, offload_tags):
        assert offload_tags == (expected_tag,)
        if mode != 'no_backup':
            allocator.pointer_to_data[base].cpu_backup_tensor = types.SimpleNamespace(
                device=types.SimpleNamespace(type='cuda' if mode == 'gpu_backup' else 'cpu'),
                dtype='float32' if mode == 'wrong_backup_dtype' else torch.uint8,
                numel=lambda: allocation_bytes - (1 if mode == 'short_backup' else 0),
                element_size=lambda: 1)
    allocator.sleep = sleep
    def wake_up(*, tags):
        assert tags == [expected_tag]
        if mode != 'stale_backup': allocator.pointer_to_data[base].cpu_backup_tensor = None
    allocator.wake_up = wake_up
    vllm = types.ModuleType('vllm')
    device_allocator = types.ModuleType('vllm.device_allocator')
    cumem = types.ModuleType('vllm.device_allocator.cumem')
    native = types.ModuleType('vllm.cumem_allocator')
    cumem.cumem_available = True
    cumem.CuMemAllocator = types.SimpleNamespace(get_instance=lambda: allocator)
    vllm.device_allocator, device_allocator.cumem = device_allocator, cumem
    modules = {'torch': torch, 'vllm': vllm, 'vllm.device_allocator': device_allocator,
               'vllm.device_allocator.cumem': cumem, 'vllm.cumem_allocator': native}
    with mock.patch.dict(sys.modules, modules), mock.patch.object(validation, 'module_identity',
            return_value={'path': 'CPU_contract_fixture_only', 'sha256': '0'*64, 'native': True}):
        yield allocator


class ImageValidationTest(unittest.TestCase):
    def test_complete_native_kernel_set(self):
        self.assertTrue(validation.kernel_pass(success()))

    def test_failed_process_and_timeout_override_success_marker(self):
        for field,value in [('returncode',1),('timed_out',True)]:
            result=success();result[field]=value
            self.assertFalse(validation.kernel_pass(result))

    def test_missing_duplicate_nonfinite_or_error_kernel_is_not_pass(self):
        for mode in ('missing','duplicate','nonfinite','numerical_failure'):
            result=success();value=validation.parse_marked(result,validation.MARKER)
            if mode=='missing':value['checks'].pop()
            elif mode=='duplicate':value['checks'][0]=copy.deepcopy(value['checks'][1])
            elif mode=='nonfinite':value['checks'][0]['all_finite']=False
            else:value['checks'][0]['failed_elements']=1
            result['stdout_tail']=validation.MARKER+json.dumps(value)
            self.assertFalse(validation.kernel_pass(result),mode)

    def test_duplicate_or_missing_result_marker_rejected(self):
        for output in ('{}',success()['stdout_tail']+'\n'+success()['stdout_tail']):
            with self.assertRaisesRegex(ValueError,'missing or duplicate'):
                validation.parse_marked({'stdout_tail':output},validation.MARKER)

    def test_malformed_image_digest_rejected_before_environment_access(self):
        for digest in ('latest',validation.IMAGE_DIGEST+'x','sha256:'+'g'*64):
            with self.assertRaisesRegex(ValueError,'pinned image digest'):
                runtime.bind_sources(expected_image_digest=digest)


    def test_cumem_tracks_whole_tensor_and_real_backup_contract(self):
        for offset in (0, 128):
            with self.subTest(offset=offset), fake_cumem_contract(offset=offset):
                value = validation.child('cumem')
                self.assertTrue(value['allocation_tracked'])
                self.assertTrue(value['cpu_backup_observed_after_sleep'])
                self.assertTrue(value['cpu_backup_cleared_after_wake'])
                self.assertEqual(value['tensor_bytes'], 4096 * 4)
                self.assertEqual(value['cpu_backup_bytes'], 65536)
                self.assertEqual(value['tensor_pointer'] - value['allocation_base'], offset)
                self.assertEqual(value['usage_bytes'],
                    {'before': 0, 'allocated': 65536, 'asleep': 65536, 'awake': 65536})

    def test_cumem_untracked_wrong_allocation_or_usage_rejected(self):
        for mode in ('untracked', 'wrong_tag', 'wrong_device', 'zero_usage'):
            with self.subTest(mode=mode), fake_cumem_contract(mode):
                with self.assertRaisesRegex(ValueError, 'tracked|usage'):
                    validation.child('cumem')
        with fake_cumem_contract(offset=65536):
            with self.assertRaisesRegex(ValueError, 'uniquely tracked'):
                validation.child('cumem')

    def test_cumem_sleep_wake_noop_or_corrupt_backup_rejected(self):
        for mode in ('no_backup', 'gpu_backup', 'wrong_backup_dtype', 'short_backup',
                     'stale_backup', 'wrong_values'):
            with self.subTest(mode=mode), fake_cumem_contract(mode):
                with self.assertRaisesRegex(ValueError, 'backup|restore|preserve'):
                    validation.child('cumem')

    def test_kernel_marker_survives_prior_logs_but_not_truncation(self):
        value = validation.parse_marked(success(), validation.MARKER)
        value['scope'] = 'CPU parser fixture only'
        value['fixture_padding'] = 'x' * 8000
        line = validation.MARKER + json.dumps(value)
        self.assertLess(len(line), 12000)
        result = {'returncode': 0, 'timed_out': False,
                  'stdout_tail': ('prior output\n' * 2000 + line)[-12000:]}
        self.assertTrue(validation.kernel_pass(result))
        result['stdout_tail'] = line[100:]
        with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
            validation.kernel_pass(result)

if __name__=='__main__':unittest.main()
