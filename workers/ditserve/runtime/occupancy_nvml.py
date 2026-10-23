"""Synchronous, fail-closed NVML occupancy checks without a CUDA context.

The resident probe owns one NVML initialization. Every observation re-resolves
and verifies each full UUID and reads a fresh compute-process list. There is no
sampling thread or cached successful observation. This intentionally matches
the existing nvidia-smi compute-process guard; the outer launcher separately
checks graphics processes.
"""

import ctypes
import os
import re


_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
_NOT_SUPPORTED = 3
_INSUFFICIENT_SIZE = 7
_MAX_PROCESSES = 65536


class _ProcessInfo(ctypes.Structure):
    # NVML API 12 nvml.h: nvmlProcessInfo_v2_t == nvmlProcessInfo_t.
    # Both GetComputeRunningProcesses_v2 and _v3 use this 24-byte ABI on
    # 64-bit Linux. The unversioned API uses a different (v1) structure.
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class NvmlError(RuntimeError):
    pass


class NvmlOccupancy:
    """One lazily initialized NVML owner; close is idempotent, errors are fatal."""

    def __init__(self):
        self._library = None
        self._initialized = False
        self._closed = False
        self._functions = {}

    def _function(self, name, arguments):
        if name not in self._functions:
            function = getattr(self._library, name)
            function.restype, function.argtypes = ctypes.c_int, arguments
            self._functions[name] = function
        return self._functions[name]

    @staticmethod
    def _check(code, operation):
        if code != 0:
            raise NvmlError("%s: NVML status %d" % (operation, code))

    def _open(self):
        if self._closed:
            raise NvmlError("NVML occupancy probe is closed")
        if self._initialized:
            return
        if (ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(_ProcessInfo) != 24
                or _ProcessInfo.usedGpuMemory.offset != 8 or _ProcessInfo.gpuInstanceId.offset != 16):
            raise NvmlError("unsupported NVML process structure ABI")
        self._library = ctypes.CDLL("libnvidia-ml.so.1")
        # Resolve shutdown first so every successful initialization has an owner.
        self._function("nvmlShutdown", [])
        self._check(self._function("nvmlInit_v2", [])(), "nvmlInit_v2")
        self._initialized = True

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._initialized:
            self._initialized = False
            self._check(self._functions["nvmlShutdown"](), "nvmlShutdown")

    def _string(self, name, handle):
        output = ctypes.create_string_buffer(96)
        self._check(self._function(name, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint])(
            handle, output, len(output)), name)
        value = output.value.decode("utf-8")
        if not value or b"\0" not in output.raw:
            raise NvmlError(name + ": empty or unterminated value")
        return value

    def _processes(self, handle):
        for version in ("_v3", "_v2"):
            name = "nvmlDeviceGetComputeRunningProcesses" + version
            try:
                function = self._function(name, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint),
                                                 ctypes.POINTER(_ProcessInfo)])
            except AttributeError:
                continue
            capacity = 16
            for unused in range(3):
                count, rows = ctypes.c_uint(capacity), (_ProcessInfo * capacity)()
                code = function(handle, ctypes.byref(count), rows)
                if code == _NOT_SUPPORTED:
                    # Only an explicit unsupported API may try the older,
                    # same-layout v2. Permission/device/unknown errors stop.
                    break
                if code == _INSUFFICIENT_SIZE:
                    capacity = max(capacity * 2, count.value + 16)
                    if capacity > _MAX_PROCESSES:
                        raise NvmlError(name + ": process list exceeds bounded capacity")
                    continue
                self._check(code, name)
                if count.value > capacity:
                    raise NvmlError(name + ": successful response exceeds buffer capacity")
                result = []
                for row in rows[:count.value]:
                    if row.pid <= 0:
                        raise NvmlError(name + ": invalid process PID")
                    result.append((int(row.pid), int(row.usedGpuMemory)))
                return result, name
            else:
                raise NvmlError(name + ": process list changed repeatedly")
        raise NvmlError("no supported versioned NVML compute-process API")

    @staticmethod
    def _allowed(request, key):
        values = request.get(key, [])
        if not isinstance(values, list) or any(type(value) is not int or value <= 0 for value in values):
            raise ValueError(key + " must contain positive integer process identities")
        return set(values)

    def observe(self, request):
        empty = dict(status="unknown", owned=[], foreign=[], devices=[],
                     cuda_initialized=False, occupancy_backend="nvml")
        try:
            requested = request.get("gpu_uuids", [])
            if (not isinstance(requested, list) or not requested
                    or any(not isinstance(value, str) or not _UUID.fullmatch(value) for value in requested)
                    or len(set(requested)) != len(requested)):
                raise ValueError("explicit unique full GPU UUIDs are required")
            allowed, groups = self._allowed(request, "allowed_pids"), self._allowed(request, "allowed_pgids")
            self._open()
            devices, owned, foreign = [], [], []
            for uuid in requested:
                handle = ctypes.c_void_p()
                self._check(self._function("nvmlDeviceGetHandleByUUID", [ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_void_p)])(uuid.encode(), ctypes.byref(handle)),
                    "nvmlDeviceGetHandleByUUID")
                if not handle.value or self._string("nvmlDeviceGetUUID", handle) != uuid:
                    raise NvmlError("requested full GPU UUID does not match resolved device")
                name = self._string("nvmlDeviceGetName", handle)
                processes, api = self._processes(handle)
                devices.append(dict(uuid=uuid, name=name, process_api=api))
                for pid, memory in processes:
                    try:
                        pgid = os.getpgid(pid)
                    except ProcessLookupError:
                        # Preserve the row even when NVML outlives the PID or
                        # the namespace cannot resolve it, as the smi path does.
                        pgid = None
                    row = dict(pid=pid, pgid=pgid, gpu_uuid=uuid,
                        memory_mib="N/A" if memory == (1 << 64) - 1 else format(memory / (1 << 20), ".6f"))
                    (owned if pid in allowed or pgid in groups else foreign).append(row)
            return dict(empty, status="busy" if foreign else "exclusive_observed",
                        owned=owned, foreign=foreign, devices=devices)
        except (OSError, AttributeError, ValueError, TypeError, NvmlError) as error:
            reason = str(error)
            try:
                self.close()
            except (OSError, AttributeError, ValueError, TypeError, NvmlError) as close_error:
                reason += "; close: " + str(close_error)
            return dict(empty, reason=reason)
