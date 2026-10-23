"""Read-only Nsight SQLite analysis for workflow stages and actual device overlap.

    python -m ditserve.evaluation.workflow_trace --sqlite capture.sqlite --out trace.json

NVTX host ranges and CUDA launch-correlated GPU projections are reported separately.
Hardware counters remain device-wide, even when averaged over a particular stage.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sqlite3

from .workflow_metrics import interval_intersection, interval_union, overlap_summary


METRICS = {
    "SM Active": "sm_active_percent",
    "SMs Active": "sm_active_percent",  # Nsight Systems 2024.6 renamed this label.
    "SM Issue": "sm_issue_percent",
    "Tensor Active": "tensor_active_percent",
    "Compute Warps in Flight": "compute_warp_slots_percent_elapsed",
    "Unallocated Warps in Active SMs": "unallocated_warp_slots_percent_elapsed",
    "DRAM Read Bandwidth": "dram_read_throughput_percent",
    "DRAM Write Bandwidth": "dram_write_throughput_percent",
    "PCIe RX Throughput": "pcie_rx_throughput_percent",
    "PCIe TX Throughput": "pcie_tx_throughput_percent",
    "GPC Clock Frequency": "gpc_clock_mhz",
    "SYS Clock Frequency": "sys_clock_mhz",
}


def metric_definition(name):
    """ad10x.config: percentage multiplier is 1; exported clock values are Hz."""
    base = name.split(" [", 1)[0]
    if base not in METRICS:
        return None
    if "Clock" in base:
        return dict(key=METRICS[base], scale=1e-6, unit="MHz", raw_unit="Hz")
    if "[" in name and "Throughput %" not in name:
        return None  # Do not confuse average warp counts with percentage of slots.
    return dict(key=METRICS[base], scale=1., unit="percent", raw_unit="percent")


def normalized_metric(value, definition):
    value = float(value)
    if not math.isfinite(value):
        return None
    # Nsight 2023.4's integer SQLite export wraps positive clock rates at 2^31.
    if definition["raw_unit"] == "Hz" and -(2 ** 31) <= value < 0 and value.is_integer():
        value += 2 ** 32
    return value * definition["scale"]


class MetricSeries:
    """Sample i summarizes the interval ending at its timestamp; no extrapolation.

    Hardware counters are differences between samples. A gap larger than four
    median sample periods is excluded rather than assigning one sample to a gap.
    """
    def __init__(self, samples):
        by_time = {int(timestamp): value for timestamp, value in samples if value is not None}
        self.times = sorted(by_time)
        self.values = [by_time[timestamp] for timestamp in self.times]
        periods = sorted(b - a for a, b in zip(self.times, self.times[1:]) if b > a)
        self.period_ns = periods[len(periods) // 2] if periods else 0

    def summary(self, start, end):
        if end < start:
            raise ValueError("end must not precede start")
        weighted = []
        first = max(1, bisect_right(self.times, start))
        last = min(len(self.times), bisect_left(self.times, end) + 1)
        for index in range(first, last):
            left, right = self.times[index - 1], self.times[index]
            if right - left > 4 * self.period_ns:
                continue
            duration = max(0, min(end, right) - max(start, left))
            if duration:
                weighted.append((self.values[index], duration))
        covered = sum(duration for _, duration in weighted)
        result = dict(sample_intervals=len(weighted), covered_ns=covered, window_ns=end - start,
                      coverage_fraction=covered / (end - start) if end > start else None,
                      nominal_period_ns=self.period_ns, mean=None, p05=None, p50=None, p95=None,
                      min=None, max=None)
        if covered:
            result["mean"] = sum(value * duration for value, duration in weighted) / covered
            ordered = sorted(weighted)
            result["min"], result["max"] = ordered[0][0], ordered[-1][0]
            for quantile, name in ((.05, "p05"), (.5, "p50"), (.95, "p95")):
                accumulated = 0
                for value, duration in ordered:
                    accumulated += duration
                    if accumulated >= quantile * covered:
                        result[name] = value
                        break
        return result


def parse_label(label):
    name, separator, payload = label.partition("|")
    metadata = {}
    error = None
    if separator:
        try:
            metadata = json.loads(payload)
            if not isinstance(metadata, dict):
                raise ValueError("metadata is not an object")
        except (ValueError, TypeError) as exception:
            metadata, error = {}, str(exception)
    for prefix in ("ditserve.workflow/", "workflow/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Frozen encoder runners put frame/mode metadata on GPU stages but omit
    # it on the encoder stages of the same request. Recover it from their
    # stable request IDs so both devices belong to one queue arm, while serial
    # and prefetch arms remain separate. Explicit metadata stays authoritative.
    request = re.fullmatch(
        r"(?P<variant>T[0-5])-(?:q-(?P<mode>serial|prefetch)-)?f(?P<frames>\d+)"
        r"-r(?P<repeat>-?\d+)(?:-i\d+)?", str(metadata.get("request_id", "")))
    if request and request["variant"] == metadata.get("variant"):
        metadata.setdefault("frames", int(request["frames"]))
        metadata.setdefault("repeat", int(request["repeat"]))
        if request["mode"]:
            metadata.setdefault("mode", request["mode"])
    purpose = str(metadata.get("purpose", ""))
    tokens = name.split("/")
    reasons = []
    if (any(re.search(r"warm|qualification|validation|numeric|correctness|cold|setup", token, re.I)
            for token in tokens + [purpose]) or any(re.fullmatch(r"input\.\d+", token) for token in tokens)):
        reasons.append("warmup, setup, cold load or numerical qualification")
    repeat = metadata.get("repeat")
    if (isinstance(repeat, (int, float)) and repeat < 0) or any(re.fullmatch(r"r-\d+", token) for token in tokens):
        reasons.append("negative warmup repeat")
    arm = None
    if tokens and re.fullmatch(r"E[1-6]", tokens[0]):
        for index, token in enumerate(tokens):
            if re.fullmatch(r"r\d+", token):
                limit = index + 1 if tokens[0] == "E6" else index + 2
                arm = "/".join(tokens[:limit])
                break
    elif metadata.get("variant"):
        parts = [metadata["variant"], purpose or "unspecified"]
        if metadata.get("mode"):
            parts.append(metadata["mode"])
        parts.append("frames." + str(metadata.get("frames", "na")))
        if "peer_frames" in metadata:
            parts.append("peer_frames." + str(metadata["peer_frames"]))
        parts.append("r" + str(metadata.get("repeat", "na")))
        arm = "/".join(str(value) for value in parts)
    return dict(name=name, metadata=metadata, included=not reasons,
                exclusion_reasons=reasons, arm=arm, metadata_error=error)


def _duration(intervals):
    return sum(end - start for start, end in interval_union(intervals))


def _group_operations(operations):
    groups = defaultdict(list)
    for operation in operations:
        groups[operation["kind"]].append(operation)
    compute = [(op["start"], op["end"]) for op in groups["compute"]]
    result = dict(operation_count=len(operations), compute_kernel_count=len(groups["compute"]),
                  compute_union_ms=_duration(compute) / 1e6, communication={},
                  memset_count=len(groups["memset"]),
                  memset_union_ms=_duration([(op["start"], op["end"]) for op in groups["memset"]]) / 1e6,
                  overlap_interval_basis="device execution, not host API duration")
    for kind in ("nccl", "h2d", "d2h", "d2d", "p2p", "other_copy"):
        rows = groups[kind]
        intervals = [(op["start"], op["end"]) for op in rows]
        result["communication"][kind] = dict(count=len(rows), bytes=None if kind == "nccl" and rows else sum(op.get("bytes", 0) for op in rows),
                                              **overlap_summary(compute, intervals))
    return result


def _tables(connection):
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _rows(connection, tables, name):
    return connection.execute('SELECT * FROM "' + name + '"') if name in tables else []


def analyze_trace(sqlite_path):
    path = Path(sqlite_path).resolve()
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return _analyze(connection, path)
    finally:
        connection.close()


def _analyze(connection, path):
    tables = _tables(connection)
    warnings = []
    strings = {row["id"]: row["value"] for row in _rows(connection, tables, "StringIds")}
    devices = {int(row["id"]): dict(row) for row in _rows(connection, tables, "TARGET_INFO_GPU")}
    export = {row["name"]: row["value"] for row in _rows(connection, tables, "META_DATA_EXPORT")}
    definitions, catalog = {}, []
    for row in _rows(connection, tables, "TARGET_INFO_GPU_METRICS"):
        definition = metric_definition(row["metricName"])
        if definition:
            definitions[(row["typeId"], row["metricId"])] = definition
            catalog.append(dict(row, **definition, device_id=row["typeId"] & 0xFF))
    samples = defaultdict(list)
    for row in _rows(connection, tables, "GPU_METRICS"):
        definition = definitions.get((row["typeId"], row["metricId"]))
        if definition:
            samples[(row["typeId"] & 0xFF, definition["key"])].append(
                (row["timestamp"], normalized_metric(row["value"], definition)))
    series = {key: MetricSeries(rows) for key, rows in samples.items()}
    for gpu in {device for device, _ in series}:
        read = dict(samples.get((gpu, "dram_read_throughput_percent"), []))
        write = dict(samples.get((gpu, "dram_write_throughput_percent"), []))
        common = sorted(read.keys() & write.keys())
        if common:
            series[(gpu, "dram_total_throughput_percent")] = MetricSeries(
                [(timestamp, read[timestamp] + write[timestamp]) for timestamp in common
                 if read[timestamp] is not None and write[timestamp] is not None])
    if not series:
        warnings.append("No GPU metric samples; SM and DRAM metrics are unavailable, not zero.")
    metric_devices = sorted({device for device, _ in series})

    def counters(device, start, end):
        return {key: values.summary(start, end) for (gpu, key), values in series.items() if gpu == device}

    phases = []
    for row in _rows(connection, tables, "NVTX_EVENTS"):
        raw = dict(row)
        label = raw.get("text") or strings.get(raw.get("textId"), "")
        if not (label.startswith(("ditserve.workflow/", "workflow/")) or re.match(r"E[1-6]/", label)):
            continue
        if raw.get("end") is None or raw["end"] <= raw["start"]:
            warnings.append("Ignored incomplete/empty NVTX range: " + label)
            continue
        parsed = parse_label(label)
        if parsed["metadata_error"]:
            warnings.append("Malformed NVTX metadata: " + label)
        tid = raw.get("globalTid")
        phases.append(dict(id=len(phases), label=label, start_ns=raw["start"], end_ns=raw["end"],
                           wall_ms=(raw["end"] - raw["start"]) / 1e6, global_tid=tid,
                           pid=((tid >> 24) & 0xFFFFFF) if tid is not None else None,
                           cross_thread=raw.get("endGlobalTid") not in (None, tid), **parsed))
    if not phases:
        warnings.append("No workflow NVTX ranges were found.")

    operations, by_correlation = [], defaultdict(list)
    kernel_tables = [name for name in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL") if name in tables]
    for table in kernel_tables + [name for name in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET") if name in tables]:
        for source in _rows(connection, tables, table):
            row = dict(source)
            if row.get("deviceId") is None or row.get("end", 0) <= row.get("start", 0):
                continue
            operation = dict(id=len(operations), start=row["start"], end=row["end"], device_id=row["deviceId"],
                             global_pid=row.get("globalPid"), correlation_id=row.get("correlationId"),
                             stream_id=row.get("streamId"), kind="compute")
            if "KERNEL" in table:
                name = next((strings[row[key]] for key in ("demangledName", "shortName", "mangledName")
                             if row.get(key) in strings), "")
                operation["kernel_name"] = name
                if "nccl" in name.lower():
                    operation["kind"] = "nccl"
            elif "MEMCPY" in table:
                operation.update(kind={1: "h2d", 3: "h2d", 11: "h2d", 2: "d2h", 4: "d2h", 12: "d2h",
                                       5: "d2d", 6: "d2d", 7: "d2d", 8: "d2d", 13: "d2d", 10: "p2p"}.get(
                                           row.get("copyKind"), "other_copy"), bytes=row.get("bytes", 0))
            else:
                operation.update(kind="memset", bytes=row.get("bytes", 0))
            operations.append(operation)
            if operation["correlation_id"] is not None and operation["global_pid"] is not None:
                by_correlation[(operation["global_pid"], operation["correlation_id"])].append(operation["id"])
    if not kernel_tables:
        warnings.append("No CUDA kernel table; kernel/communication overlap is unavailable, not zero.")
    runtimes = defaultdict(list)
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        for row in _rows(connection, tables, table):
            if row["globalTid"] is not None and row["correlationId"] is not None:
                tid = row["globalTid"]
                ids = by_correlation.get((tid & 0xFFFFFFFFFF000000, row["correlationId"]))
                if ids:
                    runtimes[tid].append((row["start"], row["end"], ids))
    for tid in runtimes:
        runtimes[tid].sort()
    runtime_starts = {tid: [row[0] for row in rows] for tid, rows in runtimes.items()}
    phase_ops, assigned = {}, set()
    for phase in phases:
        ids = set()
        tid, start, end = phase["global_tid"], phase["start_ns"], phase["end_ns"]
        if not phase["cross_thread"] and tid in runtimes:
            first, last = bisect_left(runtime_starts[tid], start), bisect_right(runtime_starts[tid], end)
            for api_start, api_end, found in runtimes[tid][first:last]:
                if api_end <= end:
                    ids.update(found)
        phase_ops[phase["id"]] = ids
        assigned.update(ids)
        per_device = {}
        for gpu in sorted(set(metric_devices) | {operations[index]["device_id"] for index in ids}):
            owned = [operations[index] for index in ids if operations[index]["device_id"] == gpu]
            projection = None
            if owned:
                gpu_start, gpu_end = min(op["start"] for op in owned), max(op["end"] for op in owned)
                projection = dict(start_ns=gpu_start, end_ns=gpu_end, span_ms=(gpu_end - gpu_start) / 1e6,
                                  metrics=counters(gpu, gpu_start, gpu_end), **_group_operations(owned))
            per_device[str(gpu)] = dict(host_window_metrics=counters(gpu, start, end), gpu_projection=projection)
        phase["devices"] = per_device

    grouped = defaultdict(list)
    for phase in phases:
        if phase["included"] and phase["arm"]:
            grouped[phase["arm"]].append(phase)
    arms = []
    for name, members in sorted(grouped.items()):
        start, end = min(row["start_ns"] for row in members), max(row["end_ns"] for row in members)
        ids = set().union(*(phase_ops[row["id"]] for row in members))
        owned = [operations[index] for index in ids]
        # Asynchronous streams can outlive their launch-side NVTX ranges. Keep
        # the host envelope intact, but include every correlated GPU operation
        # when defining the execution window used for busy-time percentages.
        execution_start = min(start, min((op["start"] for op in owned), default=start))
        execution_end = max(end, max((op["end"] for op in owned), default=end))
        gpu_ids = sorted(set(metric_devices) | {op["device_id"] for op in owned})
        per_device, compute_by_device = {}, {}
        for gpu in gpu_ids:
            local = [op for op in owned if op["device_id"] == gpu]
            compute = [(op["start"], op["end"]) for op in local if op["kind"] == "compute"]
            compute_by_device[gpu] = compute
            cpu_encode = [(row["start_ns"], row["end_ns"]) for row in members
                          if ("cpu_encode" in row["name"] or row["name"] == "text_encoder")
                          and not phase_ops[row["id"]]]
            models = {}
            # Match the operation leaf, not experiment/arm path components.
            # E5/vae/.../sana.0 belongs only to Sana; matching '/vae' anywhere
            # would count the same Sana kernel as both models' work.
            for model, pattern in (("sana", r"sana\.\d+"),
                                   ("wan", r"dit\.step\.\d+|vae(?:\.chunk\.\d+)?")):
                model_ids = set().union(*(phase_ops[row["id"]] for row in members
                                          if re.fullmatch(pattern, row["name"].rsplit("/", 1)[-1])))
                models[model] = [(op["start"], op["end"]) for op in local if op["id"] in model_ids and op["kind"] == "compute"]
            per_device[str(gpu)] = dict(metrics=counters(gpu, start, end),
                                       execution_window_metrics=counters(gpu, execution_start, execution_end),
                                       **_group_operations(local),
                                       cpu_encode_gpu_compute_overlap_ms=_duration(interval_intersection(cpu_encode, compute)) / 1e6,
                                       wan_sana_kernel_overlap_ms=_duration(interval_intersection(models["wan"], models["sana"])) / 1e6)
            if any(row["metadata"].get("variant") == "P4" for row in members):
                # The stage_pair parent inclusively owns both streams and
                # prefix_b_dit_step is B's earlier denoising. Neither is an A
                # DiT leaf. Only exact leaf+role pairs identify this overlap.
                role_intervals = {}
                for key, leaf, role in (("dit_a", "dit_step", "A"), ("vae_b", "vae_decode", "B")):
                    selected_ids = set().union(*(phase_ops[row["id"]] for row in members
                        if row["name"].rsplit("/", 1)[-1] == leaf and
                        row["metadata"].get("role") == role))
                    role_intervals[key] = [(op["start"], op["end"]) for op in local
                                           if op["id"] in selected_ids and op["kind"] == "compute"]
                dit, vae = role_intervals["dit_a"], role_intervals["vae_b"]
                available = bool(kernel_tables and dit and vae)
                per_device[str(gpu)].update(
                    dit_vae_kernel_overlap_ms=(_duration(interval_intersection(dit, vae)) / 1e6
                                               if available else None),
                    dit_a_kernel_union_ms=_duration(dit) / 1e6 if dit else None,
                    vae_b_kernel_union_ms=_duration(vae) / 1e6 if vae else None,
                    dit_vae_kernel_overlap_available=available,
                    dit_vae_kernel_overlap_basis="same-device launch-correlated compute kernel interval intersection; exact dit_step/A and vae_decode/B leaves")
        pair_overlap = []
        for index, left in enumerate(gpu_ids):
            for right in gpu_ids[index + 1:]:
                pair_overlap.append(dict(device_ids=[left, right], simultaneous_compute_ms=_duration(
                    interval_intersection(compute_by_device[left], compute_by_device[right])) / 1e6))
        arms.append(dict(name=name, host_start_ns=start, host_end_ns=end, host_span_ms=(end - start) / 1e6,
                         execution_start_ns=execution_start, execution_end_ns=execution_end,
                         execution_span_ms=(execution_end - execution_start) / 1e6,
                         phase_ids=[row["id"] for row in members], devices=per_device, dual_gpu_overlap=pair_overlap,
                         boundary_basis="envelope of included NVTX phases, not replacement for runner end-to-end timer",
                         execution_boundary_basis="envelope of included NVTX phases and their launch-correlated GPU operations; excludes uncorrelated work"))
    return dict(schema_version="ditserve.workflow_trace.v1", source=str(path), export=export,
                status="partial" if warnings else "completed", warnings=warnings,
                availability=dict(gpu_metrics=bool(series), cuda_kernels=bool(kernel_tables),
                                  cuda_memcpy="CUPTI_ACTIVITY_KIND_MEMCPY" in tables, workflow_nvtx=bool(phases)),
                devices=devices, metric_catalog=catalog, phases=phases, arms=arms,
                counts=dict(cuda_operations=len(operations), cuda_operations_correlated_to_workflow=len(assigned),
                            phases=len(phases), phases_included=sum(row["included"] for row in phases)),
                interpretation=[
                    "Metric percentages are exported percentage points (0-100), not fractions. No artificial clamping is applied.",
                    "GPU counters are device-wide and include concurrent workloads; stage windows alone do not establish attribution.",
                    "Compute warp slots are relative to peak over elapsed cycles; this is not NCU achieved occupancy conditional on active cycles.",
                    "DRAM throughput is sampled interface activity relative to peak; exact bytes/s are not inferred from theoretical bandwidth.",
                    "Clock values use ad10x.config multiplier 1e-6; old signed-int32 export wrapping is repaired.",
                    "Metric samples cover preceding intervals; gaps over four median sample periods are excluded; no endpoint extrapolation.",
                    "CUDA projection associates GPU operations through same-process correlation IDs and API calls enclosed by same-thread NVTX ranges.",
                    "Arm host_span_ms covers NVTX host ranges; execution_span_ms also includes their correlated GPU operations, which may finish after host ranges close.",
                    "Arm device metrics retain the host window; execution_window_metrics use the execution envelope. Neither envelope replaces runner end-to-end timing.",
                    "Nested NVTX ranges are inclusive and must not be summed as exclusive stage time.",
                    "P4 DiT/VAE overlap matches only dit_step role A and vae_decode role B kernels on the same device; prefix B DiT, ancestor ranges and memcpy are excluded. Missing role kernels yield null, not zero.",
                    "NCCL intervals identify CUDA kernels containing nccl in their symbol name; they can include device waiting.",
                    "Communication bytes count actual traced copies; host wait ranges are not memcpy or NCCL execution.",
                    "NCCL byte counts cannot be recovered from kernel intervals alone and are null when NCCL kernels are present.",
                    "Missing CUDA tables make overlap unavailable even if summary interval lists are empty.",
                ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_trace(args.sqlite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(out=str(args.out), status=report["status"], counts=report["counts"], warnings=report["warnings"])))


if __name__ == "__main__":
    main()
