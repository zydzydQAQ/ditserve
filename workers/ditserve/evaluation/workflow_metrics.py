"""Opt-in video workflow measurements; importing this module never initializes CUDA.

Wall intervals are host intervals unless ``synchronize=True``. NVML GPU busy
and memory-I/O busy are *not* SM Active, occupancy or DRAM bandwidth utilization.
Those require the separate Nsight counter collection described by the study.
"""

from contextlib import contextmanager
import ctypes
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time


def interval_union(intervals):
    """Return disjoint half-open intervals; input and output units are unchanged."""
    values = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("an interval must contain start and end")
        start, end = interval
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (start, end)) or end < start:
            raise ValueError("interval endpoints must be finite and end must not precede start")
        if end > start:
            values.append((start, end))
    merged = []
    for start, end in sorted(values):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_intersection(left, right):
    """Intersect unions, never double-counting overlapping events on a device."""
    left, right = interval_union(left), interval_union(right)
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def overlap_summary(compute, communication):
    """Summarize nanosecond device intervals from the SAME device/time domain.

    Communication kinds (H2D, D2H, NCCL) should be summarized separately.
    Missing communication yields a null coverage fraction, not perfect overlap.
    CUDA-event intervals cannot be mixed directly with host monotonic timestamps.
    """
    compute, communication = interval_union(compute), interval_union(communication)
    overlap = interval_intersection(compute, communication)
    duration = lambda rows: sum(end - start for start, end in rows)
    compute_ns, communication_ns, overlap_ns = map(duration, (compute, communication, overlap))
    return dict(compute_ns=compute_ns, communication_ns=communication_ns, overlap_ns=overlap_ns,
                exposed_communication_ns=communication_ns - overlap_ns,
                compute_ms=compute_ns / 1e6, communication_ms=communication_ns / 1e6,
                overlap_ms=overlap_ns / 1e6,
                exposed_communication_ms=(communication_ns - overlap_ns) / 1e6,
                communication_coverage=overlap_ns / communication_ns if communication_ns else None,
                compute_coverage=overlap_ns / compute_ns if compute_ns else None)


def process_snapshot(pid=None, include_children=True):
    """Linux process-tree counters, including CPU time of already waited children.

    RSS is the sum of resident sets, so shared pages may appear more than once.
    Tree CPU is a process-window measurement, not attribution to a calling thread.
    """
    timestamp_ns = time.perf_counter_ns()
    pid = os.getpid() if pid is None else int(pid)
    tick_s = 1. / os.sysconf("SC_CLK_TCK")
    page_bytes = os.sysconf("SC_PAGE_SIZE")
    queue, seen, rows, errors = [pid], set(), [], []
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        root = Path("/proc") / str(current)
        try:
            # comm can contain spaces and parentheses. Fields below start at state (3).
            stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
            rows.append(dict(pid=current, cpu_s=(int(stat[11]) + int(stat[12])) * tick_s,
                             waited_children_cpu_s=(int(stat[13]) + int(stat[14])) * tick_s,
                             rss_bytes=int(stat[21]) * page_bytes, threads=int(stat[17])))
            if include_children:
                # A child can have been forked by any thread, not only the main thread.
                for task in (root / "task").iterdir():
                    try:
                        queue.extend(int(value) for value in (task / "children").read_text().split())
                    except (FileNotFoundError, ProcessLookupError):
                        pass
        except (OSError, ValueError, IndexError) as error:
            errors.append("pid %d: %s" % (current, error))
    current = next((row for row in rows if row["pid"] == pid), {})
    return dict(pid=pid, timestamp_ns=timestamp_ns, query_duration_ns=time.perf_counter_ns() - timestamp_ns,
                process_cpu_s=current.get("cpu_s"),
                tree_cpu_s=sum(row["cpu_s"] + row["waited_children_cpu_s"] for row in rows),
                rss_bytes=current.get("rss_bytes"), tree_rss_bytes=sum(row["rss_bytes"] for row in rows),
                threads=current.get("threads"), tree_threads=sum(row["threads"] for row in rows),
                pids=sorted(row["pid"] for row in rows), errors=errors)


def torch_snapshot(device=None):
    """Allocator/device snapshot without importing torch or initializing CUDA."""
    torch = sys.modules.get("torch")
    if (torch is None or str(device) == "cpu" or not torch.cuda.is_initialized()):
        return dict(cuda_initialized=False)
    try:
        free, total = torch.cuda.mem_get_info(device)
        return dict(cuda_initialized=True, device=str(device if device is not None else torch.cuda.current_device()),
                    allocated_bytes=int(torch.cuda.memory_allocated(device)),
                    reserved_bytes=int(torch.cuda.memory_reserved(device)),
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                    peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                    device_total_bytes=int(total), device_free_bytes=int(free),
                    device_used_bytes=int(total - free))
    except Exception as error:
        return dict(cuda_initialized=True, error=str(error))


class PhaseRecorder:
    """NVTX + CPU/allocator windows, with no synchronization in the default mode.

    Enable synchronize/reset_peaks only for isolated diagnostic measurements.
    CUDA allocator peaks are process/device global and cannot be attributed to
    overlapping phases. Metadata is nested to prevent overriding measured fields.
    """

    def __init__(self, device=None, *, synchronize=False, reset_peaks=False, nvtx=True):
        if reset_peaks and not synchronize:
            raise ValueError("reset_peaks requires isolated synchronized phase measurements")
        self.device, self.synchronize, self.reset_peaks, self.nvtx = device, synchronize, reset_peaks, nvtx
        self.records, self._active, self._lock = [], 0, threading.Lock()

    def _cuda(self):
        torch = sys.modules.get("torch")
        return torch if (torch is not None and str(self.device) != "cpu" and torch.cuda.is_initialized()) else None

    @contextmanager
    def phase(self, name, **metadata):
        if any(not isinstance(value, (str, int, float, bool, type(None))) for value in metadata.values()):
            raise TypeError("phase metadata values must be JSON primitive types")
        metadata_json = json.dumps(metadata, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        nvtx_label = "ditserve.workflow/" + str(name) + "|" + metadata_json
        with self._lock:
            if self.reset_peaks and self._active:
                raise RuntimeError("peak resets cannot be nested or concurrent")
            self._active += 1
        record = dict(name=str(name), metadata=metadata, nvtx_label=nvtx_label, pid=os.getpid(), thread_id=threading.get_ident(),
                      timing_scope="completed_device_work" if self.synchronize else "host_interval",
                      cpu_scope="process_tree_window", peak_scope="phase" if self.reset_peaks else "since_external_reset",
                      errors=[])
        pushed = False
        try:
            cuda = self._cuda()
            if cuda is not None and self.synchronize:
                cuda.cuda.synchronize(self.device)
            if cuda is not None and self.reset_peaks:
                cuda.cuda.reset_peak_memory_stats(self.device)
            record["process_before"] = process_snapshot()
            record["torch_before"] = torch_snapshot(self.device)
            nvtx_torch = sys.modules.get("torch")
            if nvtx_torch is not None and self.nvtx:
                try:
                    nvtx_torch.cuda.nvtx.range_push(nvtx_label)
                    pushed = True
                except Exception as error:
                    record["errors"].append("NVTX: " + str(error))
            record["start_ns"] = time.perf_counter_ns()
            record["start_unix_ns"] = time.time_ns()
            thread_start = time.thread_time_ns()
            process_start = time.process_time_ns()
            try:
                yield record
            except BaseException as error:
                record["status"], record["exception_type"] = "failed", type(error).__name__
                raise
            finally:
                try:
                    cuda = self._cuda()
                    if cuda is not None and self.synchronize:
                        cuda.cuda.synchronize(self.device)
                except Exception as error:
                    record["errors"].append("CUDA synchronize: " + str(error))
                record["end_ns"] = time.perf_counter_ns()
                record["wall_ms"] = (record["end_ns"] - record["start_ns"]) / 1e6
                record["thread_cpu_ms"] = (time.thread_time_ns() - thread_start) / 1e6
                record["process_cpu_ms"] = (time.process_time_ns() - process_start) / 1e6
                if pushed:
                    try:
                        nvtx_torch.cuda.nvtx.range_pop()
                    except Exception as error:
                        record["errors"].append("NVTX pop: " + str(error))
                record["process_after"] = process_snapshot()
                record["torch_after"] = torch_snapshot(self.device)
                tree_s = record["process_after"]["tree_cpu_s"] - record["process_before"]["tree_cpu_s"]
                record["tree_cpu_ms"] = tree_s * 1000 if tree_s >= 0 else None
                tree_window_ms = (record["process_after"]["timestamp_ns"] - record["process_before"]["timestamp_ns"]) / 1e6
                record["tree_cpu_window_ms"] = tree_window_ms
                wall = record["wall_ms"]
                record["process_average_cores"] = record["process_cpu_ms"] / wall if wall else None
                record["tree_average_cores"] = tree_s * 1000 / tree_window_ms if tree_window_ms and tree_s >= 0 else None
                before, after = record["torch_before"], record["torch_after"]
                if self.reset_peaks and "allocated_bytes" in before and "peak_allocated_bytes" in after:
                    record["peak_allocated_increment_bytes"] = max(0, after["peak_allocated_bytes"] - before["allocated_bytes"])
                    record["peak_reserved_increment_bytes"] = max(0, after["peak_reserved_bytes"] - before["reserved_bytes"])
                if "device_total_bytes" in after:
                    total = after["device_total_bytes"]
                    record["peak_allocated_capacity_fraction"] = after["peak_allocated_bytes"] / total
                    record["peak_reserved_capacity_fraction"] = after["peak_reserved_bytes"] / total
                record.setdefault("status", "completed")
                self.records.append(record)
        finally:
            with self._lock:
                self._active -= 1


def summarize_phases(records):
    """Per-label distributions; concurrent windows must not be summed as wall time."""
    groups = {}
    for row in records:
        groups.setdefault(row["name"], []).append(row)
    result = {}
    for name, rows in groups.items():
        summary = dict(samples=len(rows), failed=sum(row.get("status") != "completed" for row in rows))
        for key in ("wall_ms", "process_cpu_ms", "tree_cpu_ms", "thread_cpu_ms", "process_average_cores",
                    "tree_average_cores", "peak_allocated_increment_bytes", "peak_reserved_increment_bytes"):
            values = [row[key] for row in rows if row.get(key) is not None]
            if values:
                summary[key] = dict(n=len(values), min=min(values), median=statistics.median(values),
                                    mean=statistics.mean(values), max=max(values))
        result[name] = summary
    return result


class _Memory(ctypes.Structure):
    _fields_ = [(key, ctypes.c_ulonglong) for key in ("total", "free", "used")]


class _Utilization(ctypes.Structure):
    _fields_ = [(key, ctypes.c_uint) for key in ("gpu", "memory")]


class _Process(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class NvmlSampler:
    """Read-only ctypes NVML sampling, usable without torch/pynvml/CUDA context.

    Devices are physical NVML indices or GPU UUIDs, never CUDA-visible ordinals.
    Driver utilization samples have their own sampling period: polling faster
    does not create higher-resolution measurements. No SM/occupancy claims are made.
    """

    def __init__(self, devices=(0, 1), interval_s=0.05):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("interval_s must be positive and finite")
        self.devices, self.interval_s = tuple(devices), float(interval_s)
        self.samples, self.errors = [], []
        self._library, self._handles, self._thread = None, [], None
        self._stop = threading.Event()
        self._initialized = False

    def _error(self, message):
        if message not in self.errors:
            self.errors.append(message)

    def _function(self, name, argtypes):
        function = getattr(self._library, name)
        function.restype, function.argtypes = ctypes.c_int, argtypes
        return function

    def _check(self, code, operation):
        if code:
            self._error("%s: NVML status %d" % (operation, code))
            return False
        return True

    def start(self):
        if self._initialized or self._thread is not None:
            raise RuntimeError("sampler already started")
        self._stop.clear()
        try:
            self._library = ctypes.CDLL("libnvidia-ml.so.1")
            if not self._check(self._function("nvmlInit_v2", [])(), "nvmlInit_v2"):
                return self
            self._initialized = True
            for device in self.devices:
                handle = ctypes.c_void_p()
                if isinstance(device, str) and device.startswith("GPU-"):
                    code = self._function("nvmlDeviceGetHandleByUUID", [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)])(
                        device.encode(), ctypes.byref(handle))
                else:
                    code = self._function("nvmlDeviceGetHandleByIndex_v2", [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)])(
                        int(device), ctypes.byref(handle))
                if self._check(code, "device " + str(device)):
                    uuid, name = ctypes.create_string_buffer(96), ctypes.create_string_buffer(96)
                    for key, output in (("UUID", uuid), ("Name", name)):
                        self._check(self._function("nvmlDeviceGet" + key, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint])(
                            handle, output, len(output)), key)
                    self._handles.append((device, handle, uuid.value.decode(), name.value.decode()))
            self.sample_once()
            self._thread = threading.Thread(target=self._run, name="workflow-nvml", daemon=True)
            self._thread.start()
        except (OSError, AttributeError, ValueError) as error:
            self._error(str(error))
        return self

    def _processes(self, handle, kind):
        name = "nvmlDeviceGet" + kind + "RunningProcesses"
        for version in ("_v3", "_v2"):
            try:
                function = self._function(name + version, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(_Process)])
                break
            except AttributeError:
                function = None
        if function is None:
            self._error(name + ": no v2/v3 symbol")
            return None
        capacity = 16
        for _ in range(3):
            count, rows = ctypes.c_uint(capacity), (_Process * capacity)()
            code = function(handle, ctypes.byref(count), rows)
            if code == 7:  # NVML_ERROR_INSUFFICIENT_SIZE; process sets can change between reads.
                capacity = count.value + 16
                continue
            if not self._check(code, name):
                return None
            return [dict(pid=int(row.pid), used_gpu_memory_bytes=None if row.usedGpuMemory == (1 << 64) - 1
                         else int(row.usedGpuMemory)) for row in rows[:count.value]]
        self._error(name + ": process list changed repeatedly")
        return None

    def sample_once(self):
        """Append one sample per resolved device; never initializes NVML itself."""
        result = []
        for device, handle, uuid, name in self._handles:
            row = dict(timestamp_ns=time.perf_counter_ns(), unix_ns=time.time_ns(),
                       device=device, uuid=uuid, name=name)
            for function_name, structure, fields in (
                ("nvmlDeviceGetMemoryInfo", _Memory, {"total": "memory_total_bytes", "free": "memory_free_bytes", "used": "memory_used_bytes"}),
                ("nvmlDeviceGetUtilizationRates", _Utilization, {"gpu": "gpu_busy_percent", "memory": "memory_io_busy_percent"}),
            ):
                value = structure()
                code = self._function(function_name, [ctypes.c_void_p, ctypes.POINTER(structure)])(handle, ctypes.byref(value))
                if self._check(code, function_name):
                    row.update({output: int(getattr(value, key)) for key, output in fields.items()})
            for key, function_name, arguments in (
                ("power_mw", "nvmlDeviceGetPowerUsage", []),
                ("temperature_c", "nvmlDeviceGetTemperature", [0]),
                ("pcie_tx_kb_s", "nvmlDeviceGetPcieThroughput", [0]),
                ("pcie_rx_kb_s", "nvmlDeviceGetPcieThroughput", [1]),
            ):
                try:
                    value = ctypes.c_uint()
                    function = self._function(function_name, [ctypes.c_void_p] + [ctypes.c_uint] * len(arguments) + [ctypes.POINTER(ctypes.c_uint)])
                    if self._check(function(handle, *arguments, ctypes.byref(value)), function_name):
                        row[key] = int(value.value)
                except AttributeError:
                    self._error(function_name + ": unavailable")
            row["compute_processes"] = self._processes(handle, "Compute")
            row["graphics_processes"] = self._processes(handle, "Graphics")
            row["query_duration_ns"] = time.perf_counter_ns() - row["timestamp_ns"]
            self.samples.append(row)
            result.append(row)
        return result

    def _run(self):
        while not self._stop.wait(self.interval_s):
            try:
                self.sample_once()
            except Exception as error:
                self._error("sampling: " + str(error))

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._initialized:
            try:
                self.sample_once()
            except Exception as error:
                self._error("final sampling: " + str(error))
            finally:
                self._check(self._function("nvmlShutdown", [])(), "nvmlShutdown")
                self._initialized = False
                self._handles = []
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *unused):
        self.stop()

    def summary(self, start_ns=None, end_ns=None):
        return dict(devices=summarize_nvml_samples(self.samples, start_ns, end_ns), errors=list(self.errors),
                    interpretation="NVML GPU busy and memory-I/O busy are not SM Active, occupancy or DRAM bandwidth utilization")


def summarize_nvml_samples(samples, start_ns=None, end_ns=None):
    """Zero-order-hold time weighting; no extrapolation before/after observations."""
    groups = {}
    for row in samples:
        groups.setdefault(row["uuid"], []).append(row)
    result = {}
    for uuid, rows in groups.items():
        rows = sorted(rows, key=lambda row: row["timestamp_ns"])
        start = rows[0]["timestamp_ns"] if start_ns is None else start_ns
        end = rows[-1]["timestamp_ns"] if end_ns is None else end_ns
        if end < start:
            raise ValueError("end_ns must not precede start_ns")
        inside = [row for row in rows if start <= row["timestamp_ns"] <= end]
        summary = dict(sample_count=len(inside), window_ns=end - start,
                       covered_ns=max(0, min(end, rows[-1]["timestamp_ns"]) - max(start, rows[0]["timestamp_ns"])))
        for key in ("gpu_busy_percent", "memory_io_busy_percent", "power_mw", "pcie_tx_kb_s", "pcie_rx_kb_s"):
            weighted, observed = 0., 0
            for left, right in zip(rows, rows[1:]):
                duration = max(0, min(end, right["timestamp_ns"]) - max(start, left["timestamp_ns"]))
                if key in left and duration:
                    weighted += left[key] * duration
                    observed += duration
            summary[key + "_time_mean"] = weighted / observed if observed else None
            summary[key + "_observed_ns"] = observed
        memory = [row["memory_used_bytes"] for row in inside if "memory_used_bytes" in row]
        summary["memory_used_bytes_observed_peak"] = max(memory) if memory else None
        total = next((row["memory_total_bytes"] for row in rows if row.get("memory_total_bytes")), None)
        summary["memory_total_bytes"] = total
        summary["memory_capacity_fraction_observed_peak"] = max(memory) / total if memory and total else None
        for kind in ("compute", "graphics"):
            summary[kind + "_pids_seen"] = sorted({process["pid"] for row in inside
                                                   for process in row.get(kind + "_processes") or []})
            summary[kind + "_process_query_missing_samples"] = sum(row.get(kind + "_processes") is None for row in inside)
        result[uuid] = summary
    return result
