"""Read Nsight Systems GPU sampling from the documented 2023.4 SQLite schema.

This module never launches a profiler. Percentages are device-wide Nsight
Systems samples, not per-kernel Nsight Compute measurements. Sample timestamps
do not supply interval endpoints in this schema: integration below explicitly
uses linear interpolation between adjacent samples, without extrapolation.
"""

from contextlib import closing
import math
from pathlib import Path
import re
import sqlite3

from .common import file_hash as _sha256


_DOC = "https://docs.nvidia.com/nsight-systems/2023.4/UserGuide/index.html"
_LOCAL_DOC = "/opt/nvidia/nsight-systems/2023.4.4/documentation/UserGuide/index.html"
_REQUIRED = {
    "GPU_METRICS": {"timestamp", "typeId", "metricId", "value"},
    "TARGET_INFO_GPU_METRICS": {"typeId", "sourceId", "typeName", "metricId", "metricName"},
    "NVTX_EVENTS": {"start", "end", "eventType", "text", "textId", "globalTid"},
}
_RANGE_PATTERN = r"^(?:wan_profile/)?(?:request|denoise)(?:_\d+)?$"
_RANGE_EVENT_TYPES = {59, 60, 70, 71}  # Documented push/pop and start/end NVTX ranges.
_DEFINITIONS = {
    "sm_active_pct": ("SM active", "SMs with at least one warp in flight during sampled cycles; not compute throughput."),
    "tensor_active_pct": ("Tensor active", "Tensor pipe activity during the sample period; not achieved model FLOP/s."),
    "dram_read_active_pct": ("DRAM read bandwidth", "Ratio of sampled cycles the DRAM interface was active reading data."),
    "dram_write_active_pct": ("DRAM write bandwidth", "Ratio of sampled cycles the DRAM interface was active writing data."),
}
_ALIASES = {
    "SM Active": "sm_active_pct", "SMs Active": "sm_active_pct",
    "Tensor Active": "tensor_active_pct",
    "DRAM Read Bandwidth": "dram_read_active_pct", "VRAM Read Bandwidth": "dram_read_active_pct",
    "DRAM Write Bandwidth": "dram_write_active_pct", "VRAM Write Bandwidth": "dram_write_active_pct",
}


def _invalidate(result, reason):
    result.update(status="unknown", measured=False)
    result["reason_codes"].append(reason)
    for interval in result["ranges"]:
        interval["status"] = "unknown"
        for metric in interval["metrics"]:
            metric.update(status="unknown", measured=False, time_weighted_mean_pct=None)
            metric["reason_codes"].append(reason)


def _metric_kind(name):
    if not isinstance(name, str):
        return None
    # Percent suffixes are labels only; arbitrary suffixes and byte/s units are rejected.
    name = re.sub(r"\s*(?:\[%\]|\(%\))$", "", name).strip()
    return _ALIASES.get(name)


def _unknown_metric(key, reasons, provenance=None):
    label, definition = _DEFINITIONS[key]
    return {"id": key, "label": label, "definition": definition, "unit": "%",
            "status": "unknown", "measured": False, "reason_codes": list(reasons),
            "time_weighted_mean_pct": None, "covered_time_mean_estimate_pct": None,
            "sample_min_pct": None, "sample_max_pct": None, "sample_count": 0,
            "coverage_fraction": 0.0, "covered_duration_ns": 0, "missing_duration_ns": None,
            "provenance": provenance}


def _series(connection, type_id, metric_id, maximum_samples):
    samples = []
    invalid = duplicates = conflicts = rows_seen = 0
    for timestamp, value in connection.execute(
            "SELECT timestamp,value FROM GPU_METRICS WHERE typeId=? AND metricId=? ORDER BY timestamp",
            (type_id, metric_id)):
        rows_seen += 1
        if rows_seen > maximum_samples:
            return [], {"rows_seen": rows_seen, "row_limit_exceeded": True,
                        "invalid_rows": invalid, "duplicate_rows": duplicates, "conflicting_duplicates": conflicts}
        if not isinstance(timestamp, int):
            invalid += 1
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            invalid += 1
            value = None
        if samples and timestamp == samples[-1][0]:
            duplicates += 1
            if value != samples[-1][1]:
                conflicts += 1
                samples[-1] = (timestamp, None)
        else:
            samples.append((timestamp, value))
    return samples, {"rows_seen": rows_seen, "row_limit_exceeded": False,
                     "invalid_rows": invalid, "duplicate_rows": duplicates, "conflicting_duplicates": conflicts}


def _summarize_series(samples, start, end, *, period_ns, max_gap_periods, minimum_coverage, minimum_samples):
    covered = 0
    weighted = 0.0
    gaps = 0
    observed = [value for timestamp, value in samples if start <= timestamp <= end and value is not None]
    for (left_t, left_v), (right_t, right_v) in zip(samples, samples[1:]):
        left, right = max(start, left_t), min(end, right_t)
        if right <= left:
            continue
        span = right_t - left_t
        if left_v is None or right_v is None or span > period_ns * max_gap_periods:
            gaps += 1
            continue
        # Clipping an interval also clips the linear interpolation at both ends.
        value_left = left_v + (right_v - left_v) * ((left - left_t) / span)
        value_right = left_v + (right_v - left_v) * ((right - left_t) / span)
        duration = right - left
        covered += duration
        weighted += duration * ((value_left + value_right) / 2.0)
    coverage = covered / (end - start)
    reasons = []
    if coverage < minimum_coverage:
        reasons.append("insufficient_sample_coverage")
    if len(observed) < minimum_samples:
        reasons.append("insufficient_sample_count")
    estimate = weighted / covered if covered else None
    return {"status": "sampled" if not reasons else "unknown", "measured": not reasons,
            "reason_codes": reasons, "time_weighted_mean_pct": estimate if not reasons else None,
            "covered_time_mean_estimate_pct": estimate,
            "sample_min_pct": min(observed) if observed else None,
            "sample_max_pct": max(observed) if observed else None,
            "sample_count": len(observed), "coverage_fraction": coverage,
            "covered_duration_ns": covered, "missing_duration_ns": end - start - covered,
            "excluded_gap_count": gaps}


def _ranges(connection, table_columns, pattern):
    strings = "StringIds" in table_columns and {"id", "value"} <= table_columns["StringIds"]
    expression = "COALESCE(s.value,n.text)" if strings else "n.text"
    join = "LEFT JOIN StringIds s ON s.id=n.textId" if strings else ""
    query = ("SELECT n.rowid,n.start,n.end,n.eventType," + expression +
             ",n.globalTid,n.textId FROM NVTX_EVENTS n " + join + " ORDER BY n.start,n.rowid")
    result = []
    for row_id, start, end, event_type, name, tid, text_id in connection.execute(query):
        if event_type not in _RANGE_EVENT_TYPES or not isinstance(name, str) or not pattern.fullmatch(name):
            continue
        valid = isinstance(start, int) and isinstance(end, int) and end > start
        reason = None if valid else "incomplete_or_invalid_nvtx_range"
        if text_id is not None and not strings:
            valid, reason = False, "registered_nvtx_string_table_missing"
        frames = re.search(r"_(\d+)$", name)
        result.append({"range_rowid": row_id, "range_name": name, "frames": int(frames.group(1)) if frames else None,
                       "scope": "denoise" if "denoise" in name else "request", "start_ns": start,
                       "end_ns": end, "duration_ns": end - start if valid else None,
                       "global_tid": tid, "valid": valid, "reason": reason})
    return result


def summarize_nsys_metrics(sqlite_path, *, expected_sample_hz=1000.0, gpu_type_id=None,
                           range_pattern=_RANGE_PATTERN, minimum_coverage=0.95,
                           minimum_samples=10, max_gap_periods=2.5, maximum_samples=2000000):
    """Summarize supported percentage metrics within completed NVTX request ranges.

    Export using ``nsys export --type=sqlite --lazy=false --output=output.sqlite
    input.nsys-rep``. Supply the *configured* sampling frequency. A single metric
    stream must exist, or the caller must select its exact ``typeId`` from
    ``TARGET_INFO_GPU_METRICS``. IDs are never guessed from CUDA device indices.

    The source must be a closed, standalone SQLite export. Unknown schemas,
    incomplete ranges, inadequate coverage, invalid samples and absent metrics
    remain unknown, with null primary means rather than zeros. The partial
    covered-time estimate is explicitly diagnostic, not the full-range result.
    """
    if not isinstance(expected_sample_hz, (int, float)) or isinstance(expected_sample_hz, bool) or \
            not math.isfinite(expected_sample_hz) or expected_sample_hz <= 0:
        raise ValueError("expected_sample_hz must be positive and finite")
    if not 0 < minimum_coverage <= 1 or not isinstance(minimum_samples, int) or minimum_samples < 2:
        raise ValueError("coverage must be in (0,1] and minimum_samples at least 2")
    if not math.isfinite(max_gap_periods) or max_gap_periods < 1 or maximum_samples < minimum_samples:
        raise ValueError("invalid sample gap or row limit")
    pattern = re.compile(range_pattern)
    path = Path(sqlite_path).resolve()
    result = {"schema": "ditserve.nsys_gpu_metrics.v1", "status": "unknown", "measured": False,
              "reason_codes": [], "ranges": [], "available_streams": [], "unrecognized_metrics": [],
              "provenance": {"path": str(path), "sha256": None, "bytes": None,
                             "schema_reference": _DOC + "#sqlite-schema-reference",
                             "metric_reference": _DOC + "#gpu-metrics", "verified_local_documentation": _LOCAL_DOC,
                             "backend": "Nsight Systems periodic GPU Metrics; device scope",
                             "timestamp_unit": "ns", "value_unit": "%",
                             "integration": "piecewise linear between adjacent sample timestamps, clipped to NVTX range; no extrapolation",
                             "scope_limitation": "sampling estimates, not per-kernel NCU counters or exact continuous hardware utilization",
                             "expected_sample_hz": expected_sample_hz, "max_gap_periods": max_gap_periods,
                             "minimum_coverage": minimum_coverage, "minimum_samples": minimum_samples,
                             "maximum_samples_per_metric": maximum_samples}}
    if not path.is_file():
        result["reason_codes"].append("sqlite_export_missing")
        return result
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-journal")):
        result["reason_codes"].append("sqlite_export_has_live_sidecar")
        return result
    try:
        result["provenance"].update(sha256=_sha256(path), bytes=path.stat().st_size)
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            tables = {name for name, in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            columns = {table: {row[1] for row in connection.execute('PRAGMA table_info("' + table + '")')}
                       for table in (*_REQUIRED, "StringIds") if table in tables}
            missing = {table: sorted(required - columns.get(table, set())) for table, required in _REQUIRED.items()
                       if not required <= columns.get(table, set())}
            if missing:
                result["reason_codes"].append("unsupported_or_missing_sqlite_schema")
                result["missing_columns"] = missing
                return result
            ranges = _ranges(connection, columns, pattern)
            if not ranges:
                result["reason_codes"].append("no_matching_nvtx_ranges")
                return result
            metadata = list(connection.execute(
                "SELECT typeId,sourceId,typeName,metricId,metricName FROM TARGET_INFO_GPU_METRICS"))
            if any(not all(isinstance(row[index], int) for index in (0, 1, 3)) or
                   not all(isinstance(row[index], str) for index in (2, 4)) for row in metadata):
                result["reason_codes"].append("invalid_gpu_metric_metadata")
                return result
            streams = sorted({(row[0], row[1], row[2]) for row in metadata if _metric_kind(row[4])})
            result["available_streams"] = [{"type_id": t, "source_id": s, "type_name": n} for t, s, n in streams]
            type_ids = {row[0] for row in streams}
            selection_error = None
            if gpu_type_id is None:
                if len(type_ids) == 1:
                    gpu_type_id = next(iter(type_ids))
                else:
                    selection_error = "ambiguous_gpu_metric_stream" if type_ids else "gpu_metrics_not_collected"
            elif gpu_type_id not in type_ids:
                selection_error = "requested_gpu_metric_stream_missing"
            result["selected_type_id"] = gpu_type_id
            selected = [row for row in metadata if row[0] == gpu_type_id]
            result["unrecognized_metrics"] = [row[4] for row in selected if not _metric_kind(row[4])]
            definitions = {}
            for row in selected:
                kind = _metric_kind(row[4])
                if kind:
                    definitions.setdefault(kind, []).append(row)
            data = {}
            for kind, records in definitions.items():
                if len(records) == 1:
                    data[kind] = _series(connection, gpu_type_id, records[0][3], maximum_samples)
            for interval in ranges:
                metrics = []
                for kind in _DEFINITIONS:
                    records = definitions.get(kind, [])
                    provenance = None
                    if records:
                        provenance = [{"type_id": r[0], "source_id": r[1], "type_name": r[2],
                                       "metric_id": r[3], "metric_name": r[4], "unit": "%"} for r in records]
                    reasons = [selection_error] if selection_error else []
                    if not interval["valid"]:
                        reasons.append(interval["reason"])
                    if not records:
                        reasons.append("metric_not_collected")
                    elif len(records) != 1:
                        reasons.append("ambiguous_metric_definition")
                    metric = _unknown_metric(kind, reasons, provenance)
                    if not reasons:
                        samples, audit = data[kind]
                        metric["source_rows"] = audit
                        if audit["row_limit_exceeded"]:
                            metric["reason_codes"].append("source_sample_limit_exceeded")
                        else:
                            metric.update(_summarize_series(samples, interval["start_ns"], interval["end_ns"],
                                period_ns=1e9 / expected_sample_hz, max_gap_periods=max_gap_periods,
                                minimum_coverage=minimum_coverage, minimum_samples=minimum_samples))
                            if audit["invalid_rows"] or audit["conflicting_duplicates"]:
                                metric["reason_codes"].append("invalid_or_conflicting_source_samples")
                                metric.update(status="unknown", measured=False, time_weighted_mean_pct=None)
                    metrics.append(metric)
                interval["metrics"] = metrics
                interval["status"] = "sampled" if all(m["measured"] for m in metrics) else "partial" if any(
                    m["measured"] for m in metrics) else "unknown"
            result["ranges"] = ranges
            if selection_error:
                result["reason_codes"].append(selection_error)
            valid_metrics = [m["measured"] for interval in ranges for m in interval["metrics"]]
            result["measured"] = bool(valid_metrics) and all(valid_metrics)
            result["status"] = "sampled" if result["measured"] else "partial" if any(valid_metrics) else "unknown"
        if _sha256(path) != result["provenance"]["sha256"] or any(
                Path(str(path) + suffix).exists() for suffix in ("-wal", "-journal")):
            _invalidate(result, "sqlite_export_changed_during_read")
    except (OSError, sqlite3.Error) as error:
        _invalidate(result, "sqlite_read_failed")
        result["error"] = type(error).__name__ + ": " + str(error)
    return result
