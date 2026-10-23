"""NVML ABI, fail-closed ownership and resident lifetime, entirely on CPU."""

import ctypes
import io
import json
import subprocess
import unittest
from unittest.mock import Mock, patch

from ditserve.runtime.occupancy_nvml import NvmlOccupancy, _ProcessInfo
from ditserve.runtime.worker import Worker, occupancy, serve


GPU0 = "GPU-01234567-89ab-cdef-0123-456789abcdef"
GPU1 = "GPU-11234567-89ab-cdef-0123-456789abcdef"


class Function:
    def __init__(self, callback):
        self.callback = callback
        self.calls = 0

    def __call__(self, *arguments):
        self.calls += 1
        return self.callback(*arguments)


class Library:
    def __init__(self, processes=None):
        self.uuids = [GPU0, GPU1]
        self.processes = processes or {}
        self.codes = []
        self.init_code = self.shutdown_code = 0
        self.nvmlInit_v2 = Function(lambda: self.init_code)
        self.nvmlShutdown = Function(lambda: self.shutdown_code)
        self.nvmlDeviceGetHandleByUUID = Function(self.handle)
        self.nvmlDeviceGetUUID = Function(self.uuid)
        self.nvmlDeviceGetName = Function(self.name)
        self.nvmlDeviceGetComputeRunningProcesses_v3 = Function(self.query)
        self.nvmlDeviceGetComputeRunningProcesses_v2 = Function(self.query)

    def handle(self, uuid, output):
        if uuid.decode() not in self.uuids:
            return 6
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = self.uuids.index(uuid.decode()) + 1
        return 0

    def uuid(self, handle, output, size):
        output.value = self.uuids[handle.value - 1].encode()
        return 0

    def name(self, handle, output, size):
        output.value = b"CPU fake GPU"
        return 0

    def query(self, handle, count_pointer, output):
        count = ctypes.cast(count_pointer, ctypes.POINTER(ctypes.c_uint))
        capacity = count[0]
        if self.codes:
            code, required = self.codes.pop(0)
            count[0] = required
            return code
        rows = self.processes.get(self.uuids[handle.value - 1], [])
        count[0] = len(rows)
        if len(rows) > capacity:
            return 7
        for index, (pid, memory) in enumerate(rows):
            output[index].pid = pid
            output[index].usedGpuMemory = memory
            output[index].gpuInstanceId = 0xffffffff
            output[index].computeInstanceId = 0xffffffff
        return 0


class OccupancyNvmlTests(unittest.TestCase):
    def setUp(self):
        self.library = Library()
        self.loader = patch("ditserve.runtime.occupancy_nvml.ctypes.CDLL", return_value=self.library).start()
        self.addCleanup(patch.stopall)
        self.probe = NvmlOccupancy()
        self.addCleanup(self.probe.close)

    def observe(self, **fields):
        return self.probe.observe(dict(gpu_uuids=[GPU0, GPU1], **fields))

    def test_native_structure_layout_and_versioned_signatures(self):
        self.assertEqual(ctypes.sizeof(_ProcessInfo), 24)
        self.assertEqual([getattr(_ProcessInfo, field).offset for field, unused in _ProcessInfo._fields_],
                         [0, 8, 16, 20])
        self.assertEqual(self.observe()["status"], "exclusive_observed")
        self.assertEqual(self.library.nvmlDeviceGetComputeRunningProcesses_v3.argtypes[-1],
                         ctypes.POINTER(_ProcessInfo))
        self.assertEqual(self.library.nvmlDeviceGetComputeRunningProcesses_v2.calls, 0)

    def test_resident_probe_initializes_once_but_reads_fresh_uuid_and_processes(self):
        first = self.observe()
        self.library.processes[GPU1] = [(501, 2 << 20)]
        with patch("os.getpgid", return_value=600):
            second = self.observe()
        self.assertEqual(first["status"], "exclusive_observed")
        self.assertEqual(second["status"], "busy")
        self.assertEqual(second["foreign"][0]["gpu_uuid"], GPU1)
        self.assertEqual(second["foreign"][0]["memory_mib"], "2.000000")
        self.assertFalse(second["cuda_initialized"])
        self.assertEqual(second["occupancy_backend"], "nvml")
        self.assertEqual(self.library.nvmlInit_v2.calls, 1)
        self.assertEqual(self.library.nvmlDeviceGetUUID.calls, 4)
        self.assertEqual(self.library.nvmlDeviceGetComputeRunningProcesses_v3.calls, 4)
        self.probe.close()
        self.probe.close()
        self.assertEqual(self.library.nvmlShutdown.calls, 1)
        self.assertEqual(self.observe()["status"], "unknown")
        self.assertEqual(self.library.nvmlInit_v2.calls, 1)

    def test_owned_pids_groups_disappeared_pids_and_unknown_memory_are_preserved(self):
        self.library.processes[GPU0] = [(11, 1024), (12, 2048), (13, (1 << 64) - 1), (14, 0)]
        with patch("os.getpgid", side_effect=[ProcessLookupError(), 22, ProcessLookupError(), 24]):
            result = self.observe(allowed_pids=[11], allowed_pgids=[22])
        self.assertEqual(result["status"], "busy")
        self.assertEqual([row["pid"] for row in result["owned"]], [11, 12])
        self.assertEqual([row["pid"] for row in result["foreign"]], [13, 14])
        self.assertIsNone(result["foreign"][0]["pgid"])
        self.assertEqual(result["foreign"][0]["memory_mib"], "N/A")

    def test_invalid_or_partial_uuid_and_allowed_identity_never_initialize(self):
        cases = [dict(gpu_uuids=[]), dict(gpu_uuids=[GPU0, GPU0]), dict(gpu_uuids=["GPU-01234567"]),
                 dict(gpu_uuids=["0"]), dict(gpu_uuids=GPU0), dict(gpu_uuids=[GPU0 + "\n"]),
                 dict(gpu_uuids=[GPU0], allowed_pids=[True]), dict(gpu_uuids=[GPU0], allowed_pgids=[0])]
        for request in cases:
            with self.subTest(request=request):
                result = NvmlOccupancy().observe(request)
                self.assertEqual(result["status"], "unknown")
        self.loader.assert_not_called()

    def test_full_uuid_is_verified_after_resolution_and_every_gpu_is_required(self):
        self.library.nvmlDeviceGetUUID = Function(lambda handle, output, size: setattr(output, "value", GPU1.encode()) or 0)
        result = self.observe()
        self.assertEqual(result["status"], "unknown")
        self.assertIn("does not match", result["reason"])
        self.assertEqual(self.library.nvmlShutdown.calls, 1)
        self.library = Library()
        self.library.uuids = [GPU0]
        self.loader.return_value = self.library
        result = NvmlOccupancy().observe(dict(gpu_uuids=[GPU0, GPU1]))
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["devices"], [])

    def test_process_list_growth_retries_with_new_capacity_without_dropping_rows(self):
        self.library.codes = [(7, 20), (7, 70)]
        self.library.processes[GPU0] = [(pid, 0) for pid in range(1, 71)]
        with patch("os.getpgid", return_value=900):
            result = self.observe()
        self.assertEqual(result["status"], "busy")
        self.assertEqual(len(result["foreign"]), 70)
        self.assertEqual(self.library.nvmlDeviceGetComputeRunningProcesses_v3.calls, 4)

    def test_repeated_growth_absurd_capacity_and_invalid_success_count_fail_closed(self):
        for codes in ([(7, 20), (7, 70), (7, 100)], [(7, 1000000)], [(0, 100)]):
            with self.subTest(codes=codes):
                library = Library()
                library.codes = list(codes)
                self.loader.return_value = library
                result = NvmlOccupancy().observe(dict(gpu_uuids=[GPU0]))
                self.assertEqual(result["status"], "unknown")
                self.assertEqual(library.nvmlShutdown.calls, 1)
                self.assertEqual(library.nvmlDeviceGetComputeRunningProcesses_v2.calls, 0)

    def test_missing_or_explicitly_unsupported_v3_uses_only_known_v2_abi(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                library = Library()
                if missing:
                    del library.nvmlDeviceGetComputeRunningProcesses_v3
                else:
                    library.codes = [(3, 0)]
                self.loader.return_value = library
                probe = NvmlOccupancy()
                result = probe.observe(dict(gpu_uuids=[GPU0]))
                probe.close()
                self.assertEqual(result["status"], "exclusive_observed")
                self.assertEqual(result["devices"][0]["process_api"], "nvmlDeviceGetComputeRunningProcesses_v2")
                self.assertEqual(library.nvmlDeviceGetComputeRunningProcesses_v2.calls, 1)

    def test_permission_lost_device_and_abi_errors_never_fallback(self):
        for code in (4, 9, 13, 25, 999):
            with self.subTest(code=code):
                library = Library()
                library.codes = [(code, 0)]
                self.loader.return_value = library
                result = NvmlOccupancy().observe(dict(gpu_uuids=[GPU0]))
                self.assertEqual(result["status"], "unknown")
                self.assertEqual(library.nvmlDeviceGetComputeRunningProcesses_v2.calls, 0)
                self.assertEqual(library.nvmlShutdown.calls, 1)

    def test_unversioned_api_is_never_used(self):
        del self.library.nvmlDeviceGetComputeRunningProcesses_v3
        del self.library.nvmlDeviceGetComputeRunningProcesses_v2
        self.library.nvmlDeviceGetComputeRunningProcesses = Mock()
        self.assertEqual(self.observe()["status"], "unknown")
        self.library.nvmlDeviceGetComputeRunningProcesses.assert_not_called()

    def test_library_init_pid_lookup_and_shutdown_failures_are_visible(self):
        self.loader.side_effect = OSError("missing library")
        self.assertEqual(self.observe()["status"], "unknown")
        self.loader.side_effect = None
        self.library.init_code = 4
        self.assertEqual(NvmlOccupancy().observe(dict(gpu_uuids=[GPU0]))["status"], "unknown")
        self.assertEqual(self.library.nvmlShutdown.calls, 0)
        self.library.init_code = 0
        self.library.processes[GPU0] = [(1, 0)]
        self.library.shutdown_code = 999
        with patch("os.getpgid", side_effect=PermissionError("no PID access")):
            result = NvmlOccupancy().observe(dict(gpu_uuids=[GPU0]))
        self.assertEqual(result["status"], "unknown")
        self.assertIn("no PID access", result["reason"])
        self.assertIn("close: nvmlShutdown", result["reason"])

    def test_worker_default_smi_and_explicit_nvml_have_independent_lifetimes(self):
        worker = Worker("mock", {})
        self.addCleanup(worker.close)
        request = dict(version=1, id=1, op="occupancy", gpu_uuids=[GPU0])
        hardware = subprocess.CompletedProcess([], 0, GPU0 + ", CPU fake GPU\n", "")
        apps = subprocess.CompletedProcess([], 0, "", "")
        with patch("ditserve.runtime.worker.subprocess.run", side_effect=[hardware, apps]):
            self.assertEqual(worker.dispatch(request)["occupancy_backend"], "nvidia_smi")
        self.loader.assert_not_called()
        self.assertIsNone(worker.backend)
        with patch("ditserve.runtime.worker.subprocess.run", side_effect=AssertionError("NVML spawned process")):
            for index in range(2):
                self.assertEqual(worker.dispatch(dict(request, id=index + 2, occupancy_backend="nvml"))["status"],
                                 "exclusive_observed")
        self.assertEqual(self.library.nvmlInit_v2.calls, 1)
        worker.dispatch(dict(version=1, id=4, op="shutdown"))
        self.assertEqual(self.library.nvmlShutdown.calls, 1)
        self.assertIsNone(worker.backend)

    def test_worker_eof_and_backend_close_exception_always_close_nvml(self):
        worker = Worker("mock", {})
        request = dict(version=1, id=1, op="occupancy", gpu_uuids=[GPU0], occupancy_backend="nvml")
        output = io.StringIO()
        serve(worker, io.StringIO(json.dumps(request) + "\n"), output)
        self.assertTrue(json.loads(output.getvalue())["ok"])
        self.assertTrue(worker.closed)
        self.assertEqual(self.library.nvmlShutdown.calls, 1)
        worker = Worker("mock", {})
        worker.dispatch(request)
        worker.backend = Mock()
        worker.backend.close.side_effect = RuntimeError("model close failed")
        with self.assertRaisesRegex(RuntimeError, "model close failed"):
            worker.close()
        self.assertTrue(worker.closed)
        self.assertEqual(self.library.nvmlShutdown.calls, 2)

    def test_explicit_invalid_backend_and_one_shot_nvml(self):
        for backend in (None, "smi", "auto", True):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                occupancy(dict(occupancy_backend=backend))
        self.assertEqual(occupancy({}, mock=True)["occupancy_backend"], "nvidia_smi")
        self.assertEqual(occupancy(dict(gpu_uuids=[GPU0], occupancy_backend="nvml"))["status"],
                         "exclusive_observed")
        self.assertEqual(self.library.nvmlInit_v2.calls, 1)
        self.assertEqual(self.library.nvmlShutdown.calls, 1)


if __name__ == "__main__":
    unittest.main()
