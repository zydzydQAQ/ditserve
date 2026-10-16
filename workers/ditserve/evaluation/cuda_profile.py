"""Aggregate *real* Kineto GPU activities without counting synthetic ranges.

Chrome trace ``ts``/``dur`` are microseconds, even when ``displayTimeUnit`` is
``ms``.  Kineto emits CONCURRENT_KERNEL as ``kernel`` and synthesizes
``gpu_user_annotation`` ranges from associated activities.  The latter must
never be counted as kernels.  See the upstream source, inspected for this fix:
https://github.com/pytorch/kineto/blob/main/libkineto/src/output_json.cpp
https://github.com/pytorch/kineto/blob/main/libkineto/include/ActivityType.h

GPU External id links to a CPU operation; correlation links to CUDA runtime or
driver calls.  Only the linked CPU event's same-thread annotation ancestry is
used.  GPU timestamps are NOT compared with CPU annotations for attribution,
because asynchronous execution makes that inference invalid.  Every actual
GPU event is assigned once, independent of how many links/ancestors it has.

Only scalar summaries and bounded kernel-name aggregates are returned.  The
caller owns trace retention, hashing, and archival.  This module imports no
torch/CUDA package and is suitable for CPU-only archival and tests.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .common import file_hash


_GPU_TYPES = {"kernel": "kernel", "gpu_memcpy": "memcpy", "gpu_memset": "memset"}
_CPU_TYPES = {"cpu_op", "user_annotation"}
_RUNTIME_TYPES = {"cuda_runtime", "cuda_driver"}
_CATEGORIES = (
    "self_attention", "cross_attention", "ffn", "transformer_other", "vae",
    "text_encode", "other", "unattributed",
)
_TOLERANCE_US = 0.01  # Export rounding; never a cross-thread/time heuristic.


def _interval(event: Mapping[str, Any]) -> tuple[float, float] | None:
    if event.get("ph") != "X":
        return None
    try:
        start, duration = float(event["ts"]), float(event["dur"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
        return None
    end = start + duration
    return (start, end) if math.isfinite(end) else None


def _identifier(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and (not value.is_integer() or not math.isfinite(value)):
        return None
    result = str(int(value)) if isinstance(value, (int, float)) else str(value)
    return None if result in ("", "0", "-1") else result


def _args(event: Mapping[str, Any]) -> Mapping[str, Any]:
    result = event.get("args", {})
    return result if isinstance(result, Mapping) else {}


def _external(event: Mapping[str, Any]) -> str | None:
    return _identifier(_args(event).get("External id"))


def _correlation(event: Mapping[str, Any]) -> str | None:
    return _identifier(_args(event).get("correlation"))


def _thread(event: Mapping[str, Any]) -> tuple[str, str] | None:
    if "pid" not in event or "tid" not in event:
        return None
    return str(event["pid"]), str(event["tid"])


def _union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _length(intervals: Iterable[tuple[float, float]]) -> float:
    return math.fsum(end - start for start, end in _union(intervals))


def _percent(numerator: float, denominator: float) -> float | None:
    return numerator * 100.0 / denominator if denominator > 0 else None


@dataclass(frozen=True, slots=True)
class _Annotation:
    index: int
    name: str
    start: float
    end: float
    thread: tuple[str, str]


class _IntervalIndex:
    """Balanced interval search; avoids O(kernel count * annotation count)."""

    __slots__ = ("annotation", "left", "right", "max_end")

    def __init__(self, ordered: list[_Annotation], begin: int = 0, end: int | None = None):
        end = len(ordered) if end is None else end
        middle = (begin + end) // 2
        self.annotation = ordered[middle]
        self.left = _IntervalIndex(ordered, begin, middle) if begin < middle else None
        self.right = _IntervalIndex(ordered, middle + 1, end) if middle + 1 < end else None
        self.max_end = max(self.annotation.end,
                           self.left.max_end if self.left else -math.inf,
                           self.right.max_end if self.right else -math.inf)

    def containing(self, start: float, end: float, found: list[_Annotation]) -> None:
        if self.max_end + _TOLERANCE_US < end:
            return
        if self.left:
            self.left.containing(start, end, found)
        current = self.annotation
        if current.start > start + _TOLERANCE_US:
            return
        if current.end + _TOLERANCE_US >= end:
            found.append(current)
        if self.right:
            self.right.containing(start, end, found)


def _signature(event: Mapping[str, Any]) -> tuple[Any, ...]:
    """Deduplicate links to one event, never deduplicate GPU work by name/id."""
    return (event.get("cat"), str(event.get("name", "")), _thread(event),
            _interval(event), _external(event), _correlation(event))


def _unique_events(events: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen = set()
    answer = []
    for event in events:
        key = _signature(event)
        if key not in seen:
            seen.add(key)
            answer.append(event)
    return answer


def _chain(event: Mapping[str, Any], indices: Mapping[Any, _IntervalIndex]) -> tuple[list[_Annotation], bool]:
    interval = _interval(event)
    index = indices.get(_thread(event))
    if interval is None or index is None:
        return [], False
    found: list[_Annotation] = []
    index.containing(*interval, found)
    found.sort(key=lambda a: (a.start, -a.end, a.index))
    # Incomparable intersecting scopes and equally bounded different scopes
    # cannot be ordered by timestamps.  Refuse an arbitrary lexical winner.
    for left, right in zip(found, found[1:]):
        if right.end > left.end + _TOLERANCE_US:
            return [], True
        if (abs(left.start - right.start) < _TOLERANCE_US
                and abs(left.end - right.end) < _TOLERANCE_US
                and left.name != right.name):
            return [], True
    return found, False


def _attribute(event: Mapping[str, Any], external: Mapping[str, list],
               runtime: Mapping[str, list], indices: Mapping[Any, _IntervalIndex]) -> tuple[list[_Annotation], str, int]:
    """Resolve all exact links, then choose their compatible deepest ancestry."""
    candidates = list(external.get(_external(event), ()))
    runtimes = _unique_events(runtime.get(_correlation(event), ()))
    for call in runtimes:
        candidates.append(call)
        candidates.extend(external.get(_external(call), ()))
    candidates = _unique_events(candidates)
    chains = []
    ambiguous = False
    for candidate in candidates:
        chain, conflict = _chain(candidate, indices)
        ambiguous |= conflict
        if chain:
            chains.append(chain)
    if ambiguous:
        return [], "ambiguous_annotation", len(runtimes)
    if not chains:
        return [], "unattributed" if candidates else "missing_link", len(runtimes)
    longest = max(chains, key=len)
    ids = {a.index for a in longest}
    if any(not {a.index for a in chain}.issubset(ids) for chain in chains):
        return [], "conflicting_links", len(runtimes)
    method = "external_and_runtime" if _external(event) in external and runtimes else (
        "external" if _external(event) in external else "runtime")
    return longest, method, len(runtimes)


def _category(chain: list[_Annotation]) -> str:
    labels = [a.name for a in chain]
    for label in reversed(labels):
        if label in ("self_attention", "cross_attention", "ffn", "vae", "text_encode"):
            return label
    if "transformer" in labels:
        return "transformer_other"
    return "other" if "request" in labels else "unattributed"


def _exclusive(intervals_by_category: Mapping[str, list[tuple[float, float]]]) -> tuple[dict[str, float], float]:
    """Partition kernel union into category-only time and explicit overlap."""
    changes: dict[float, list[tuple[str, int]]] = defaultdict(list)
    for category, intervals in intervals_by_category.items():
        for start, end in _union(intervals):
            changes[start].append((category, 1))
            changes[end].append((category, -1))
    active: Counter = Counter()
    exclusive: dict[str, float] = defaultdict(float)
    overlap = 0.0
    previous = None
    for moment, entries in sorted(changes.items()):
        if previous is not None:
            elapsed = moment - previous
            occupied = [category for category, count in active.items() if count > 0]
            if len(occupied) == 1:
                exclusive[occupied[0]] += elapsed
            elif len(occupied) > 1:
                overlap += elapsed
        for category, delta in entries:
            active[category] += delta
        previous = moment
    return dict(exclusive), overlap


def summarize_trace(source: str | Path | Mapping[str, Any], *, annotation_prefix: str = "wan_profile/",
                    top_kernel_limit: int = 40) -> dict[str, Any]:
    """Read one Chrome/Kineto trace and return bounded, JSON-safe evidence.

    ``source`` can be a file or a parsed trace mapping.  Files are parsed once
    with stdlib JSON; all event-level data is released on return.  A SHA256 and
    size are included for files so callers can archive then remove them safely.
    Exactly one annotated request and one GPU are required for verified rates.
    Missing/ambiguous attribution yields ``unverified``, never a fabricated 0.
    """
    provenance = None
    if isinstance(source, Mapping):
        trace = source
    else:
        path = Path(source)
        digest = file_hash(path)
        provenance = {"path": str(path.resolve()), "bytes": path.stat().st_size,
                      "sha256": digest}
        with path.open("r", encoding="utf-8") as stream:
            trace = json.load(stream)
    if not isinstance(trace, Mapping) or not isinstance(trace.get("traceEvents"), list):
        raise ValueError("Expected a Chrome trace object containing traceEvents")
    summary = summarize_trace_events(trace["traceEvents"], annotation_prefix=annotation_prefix,
                                     top_kernel_limit=top_kernel_limit)
    if provenance is not None:
        summary["trace_source"] = provenance
    return summary


def summarize_trace_events(events: Iterable[Mapping[str, Any]], *, annotation_prefix: str = "wan_profile/",
                           top_kernel_limit: int = 40) -> dict[str, Any]:
    """Aggregate complete events; auxiliary/metadata/flow rows are never work."""
    if not isinstance(top_kernel_limit, int) or isinstance(top_kernel_limit, bool) or top_kernel_limit < 0:
        raise ValueError("top_kernel_limit must be a nonnegative integer")
    if not annotation_prefix:
        raise ValueError("annotation_prefix must not be empty")
    annotations_by_thread: dict[Any, list[_Annotation]] = defaultdict(list)
    annotations = []
    gpu_events = []
    external: dict[str, list] = defaultdict(list)
    runtime: dict[str, list] = defaultdict(list)
    counts: Counter = Counter()
    category_counts: Counter = Counter()
    unknown_gpu_categories: Counter = Counter()
    for index, event in enumerate(events):
        counts["trace_event_count"] += 1
        if not isinstance(event, Mapping):
            counts["malformed_event_count"] += 1
            continue
        cat = str(event.get("cat", ""))
        category_counts[cat] += 1
        if cat == "gpu_user_annotation":
            counts["excluded_gpu_annotation_count"] += 1
            continue
        interval = _interval(event)
        if cat in _GPU_TYPES:
            if interval is None:
                counts["invalid_gpu_timeline_count"] += 1
            elif interval[1] <= interval[0]:
                counts["zero_duration_gpu_event_count"] += 1
                gpu_events.append(event)
            else:
                gpu_events.append(event)
            continue
        if cat.startswith("gpu_") and cat not in {"gpu_user_annotation"}:
            unknown_gpu_categories[cat] += 1
        if interval is None:
            continue
        if cat in _CPU_TYPES and _external(event) is not None:
            external[_external(event)].append(event)
        if cat in _RUNTIME_TYPES and _correlation(event) is not None:
            runtime[_correlation(event)].append(event)
        name = str(event.get("name", ""))
        if cat == "user_annotation" and name.startswith(annotation_prefix):
            thread = _thread(event)
            if thread is None:
                counts["invalid_annotation_thread_count"] += 1
                continue
            annotation = _Annotation(index, name[len(annotation_prefix):], *interval, thread)
            annotations.append(annotation)
            annotations_by_thread[thread].append(annotation)
    indices = {thread: _IntervalIndex(sorted(rows, key=lambda a: (a.start, -a.end, a.index)))
               for thread, rows in annotations_by_thread.items()}
    requests = [a for a in annotations if a.name == "request"]
    request_annotation = requests[0] if len(requests) == 1 else None
    counts["request_annotation_count"] = len(requests)
    counts["annotation_count"] = len(annotations)
    counts["real_gpu_event_count_in_trace"] = len(gpu_events)
    request_interval = (request_annotation.start, request_annotation.end) if request_annotation else None

    kernel_intervals = []
    activity_intervals = []
    type_intervals: dict[str, list] = defaultdict(list)
    category_intervals: dict[str, list] = defaultdict(list)
    category_durations: dict[str, list] = defaultdict(list)
    category_kernel_counts: Counter = Counter()
    transformer_durations: dict[str, list] = defaultdict(list)
    transformer_counts: Counter = Counter()
    instance_intervals: dict[int, list] = defaultdict(list)
    instance_kernel_intervals: dict[int, list] = defaultdict(list)
    name_aggregate: dict[str, dict[str, Any]] = {}
    devices, streams = set(), set()
    link_counts: Counter = Counter()
    status_counts: Counter = Counter()
    kernel_duration_parts = []
    for event in gpu_events:
        start, end = _interval(event)
        if request_interval:
            request_start, request_end = request_interval
            if end <= request_start or start >= request_end:
                counts["gpu_events_outside_request"] += 1
                continue
            if start < request_start - _TOLERANCE_US or end > request_end + _TOLERANCE_US:
                counts["gpu_events_crossing_request_boundary"] += 1
            # A known window denominator uses the clipped union.  Full kernel
            # durations are retained separately and boundary crossing invalidates
            # verified status rather than silently presenting clipped shares.
            clipped = max(start, request_start), min(end, request_end)
        else:
            clipped = start, end
        activity_type = _GPU_TYPES[str(event["cat"])]
        args = _args(event)
        device = args.get("device", event.get("pid"))
        stream = args.get("stream", event.get("tid"))
        if device is None:
            counts["missing_gpu_device_count"] += 1
        else:
            devices.add(str(device))
        if stream is None:
            counts["missing_gpu_stream_count"] += 1
        else:
            streams.add((str(device), str(stream)))
        chain, method, runtime_count = _attribute(event, external, runtime, indices)
        link_counts[method] += 1
        if runtime_count > 1:
            counts["gpu_events_with_multiple_runtime_links"] += 1
        if request_annotation and chain and request_annotation.index not in {a.index for a in chain}:
            chain, method = [], "outside_request_link"
        category = _category(chain)
        if method in {"ambiguous_annotation", "conflicting_links", "outside_request_link"}:
            counts["ambiguous_gpu_event_count"] += 1
        if category == "unattributed":
            counts["unattributed_gpu_event_count"] += 1
        activity_intervals.append(clipped)
        type_intervals[activity_type].append(clipped)
        counts[activity_type + "_count"] += 1
        for annotation in chain:
            instance_intervals[annotation.index].append(clipped)
            if activity_type == "kernel":
                instance_kernel_intervals[annotation.index].append(clipped)
        if activity_type != "kernel":
            continue
        status_counts[method] += 1
        if category == "unattributed":
            counts["unattributed_kernel_count"] += 1
        if method in {"ambiguous_annotation", "conflicting_links", "outside_request_link"}:
            counts["ambiguous_kernel_count"] += 1
        duration = end - start
        kernel_duration_parts.append(duration)
        kernel_intervals.append(clipped)
        category_intervals[category].append(clipped)
        category_durations[category].append(duration)
        category_kernel_counts[category] += 1
        if any(annotation.name == "transformer" for annotation in chain):
            transformer_durations[category].append(duration)
            transformer_counts[category] += 1
        name = str(event.get("name", "<unnamed>"))
        item = name_aggregate.setdefault(name, {"name": name, "kernel_count": 0,
                                                "durations": [], "categories": Counter()})
        item["kernel_count"] += 1
        item["durations"].append(duration)
        item["categories"][category] += 1

    kernel_duration = math.fsum(kernel_duration_parts)
    kernel_union = _length(kernel_intervals)
    activity_union = _length(activity_intervals)
    request_duration = request_interval[1] - request_interval[0] if request_interval else 0.0
    gpu_span = ((min(x[0] for x in activity_intervals), max(x[1] for x in activity_intervals))
                if activity_intervals else None)
    gpu_duration = gpu_span[1] - gpu_span[0] if gpu_span else 0.0
    exclusive, cross_category_overlap = _exclusive(category_intervals)
    reasons = []
    if len(requests) != 1 or request_duration <= 0:
        reasons.append("requires_one_positive_duration_request_annotation")
    if not kernel_intervals or kernel_duration <= 0:
        reasons.append("no_real_kernel_timeline")
    if len(devices) != 1:
        reasons.append("requires_one_gpu")
    for key in ("invalid_gpu_timeline_count", "zero_duration_gpu_event_count", "missing_gpu_device_count", "missing_gpu_stream_count",
                "gpu_events_crossing_request_boundary", "invalid_annotation_thread_count"):
        if counts[key]:
            reasons.append(key)
    if unknown_gpu_categories:
        reasons.append("unknown_gpu_activity_categories")
    timeline_verified = not reasons
    attribution_reasons = []
    if counts["unattributed_kernel_count"]:
        attribution_reasons.append("unattributed_kernels")
    if counts["ambiguous_kernel_count"]:
        attribution_reasons.append("ambiguous_kernel_links_or_annotations")
    if not kernel_intervals:
        attribution_reasons.append("no_real_kernels")
    attribution_verified = timeline_verified and not attribution_reasons
    activity_attribution_reasons = []
    if counts["unattributed_gpu_event_count"]:
        activity_attribution_reasons.append("unattributed_gpu_activities")
    if counts["ambiguous_gpu_event_count"]:
        activity_attribution_reasons.append("ambiguous_gpu_links_or_annotations")
    if not activity_intervals:
        activity_attribution_reasons.append("no_real_gpu_activities")
    activity_attribution_verified = timeline_verified and not activity_attribution_reasons
    all_reasons = reasons + attribution_reasons + activity_attribution_reasons
    kernel_categories = []
    for category in _CATEGORIES:
        duration = math.fsum(category_durations[category])
        category_union = _length(category_intervals[category])
        kernel_categories.append({
            "name": category, "kernel_count": category_kernel_counts[category],
            "kernel_duration_ms": duration / 1000.0,
            "share_pct": _percent(duration, kernel_duration),
            "union_ms": category_union / 1000.0,
            "exclusive_union_ms": exclusive.get(category, 0.0) / 1000.0,
            "exclusive_union_share_pct": _percent(exclusive.get(category, 0.0), kernel_union),
        })
    transformer_duration = math.fsum(math.fsum(values) for values in transformer_durations.values())
    transformer_categories = [
        {"name": category, "kernel_count": transformer_counts[category],
         "kernel_duration_ms": math.fsum(transformer_durations[category]) / 1000.0,
         "share_pct": _percent(math.fsum(transformer_durations[category]), transformer_duration)}
        for category in _CATEGORIES
    ]
    stage_intervals: dict[str, list] = defaultdict(list)
    stage_kernels: dict[str, list] = defaultdict(list)
    stage_windows: dict[str, list] = defaultdict(list)
    stage_cpu: dict[str, list] = defaultdict(list)
    stage_instances: Counter = Counter()
    for annotation in annotations:
        if request_interval and (annotation.start < request_interval[0] or annotation.end > request_interval[1]):
            continue
        label = annotation.name
        stage_cpu[label].append((annotation.start, annotation.end))
        stage_instances[label] += 1
        intervals = instance_intervals[annotation.index]
        stage_intervals[label].extend(intervals)
        stage_kernels[label].extend(instance_kernel_intervals[annotation.index])
        if intervals:
            stage_windows[label].append((min(x[0] for x in intervals), max(x[1] for x in intervals)))
    stages = []
    for label in sorted(stage_cpu):
        window = _length(stage_windows[label])
        kernel_busy = _length(stage_kernels[label])
        activity_busy = _length(stage_intervals[label])
        spans = stage_windows[label]
        span = max(x[1] for x in spans) - min(x[0] for x in spans) if spans else None
        stages.append({
            "name": label, "annotation_count": stage_instances[label],
            "status": "measured" if attribution_verified and activity_attribution_verified and spans else "unverified",
            "kernel_attribution_status": "measured" if attribution_verified and spans else "unverified",
            "activity_attribution_status": "measured" if activity_attribution_verified and spans else "unverified",
            "window_status": "measured" if activity_attribution_verified and spans else "unverified",
            "cpu_annotation_union_ms": _length(stage_cpu[label]) / 1000.0,
            "window_ms": window / 1000.0 if spans else None,
            "gpu_first_to_last_span_ms": span / 1000.0 if span is not None else None,
            "kernel_union_ms": kernel_busy / 1000.0,
            "activity_union_ms": activity_busy / 1000.0,
            "kernel_active_pct": _percent(kernel_busy, window) if attribution_verified and activity_attribution_verified else None,
            "activity_active_pct": _percent(activity_busy, window) if activity_attribution_verified else None,
            "window_definition": "union of each CPU annotation instance's first-to-last linked GPU activity window",
        })
    top = []
    for item in name_aggregate.values():
        duration = math.fsum(item.pop("durations"))
        item["kernel_duration_ms"] = duration / 1000.0
        item["share_pct"] = _percent(duration, kernel_duration)
        item["categories"] = dict(item["categories"])
        top.append(item)
    top.sort(key=lambda row: (-row["kernel_duration_ms"], row["name"]))
    omitted = top[top_kernel_limit:]
    top = top[:top_kernel_limit]
    # Add known zero counts explicitly; absence must not be confused with an
    # unavailable metric in an exported sheet.
    for key in ("kernel_count", "memcpy_count", "memset_count", "unattributed_kernel_count",
                "ambiguous_kernel_count", "unattributed_gpu_event_count", "ambiguous_gpu_event_count",
                "gpu_events_outside_request", "gpu_events_crossing_request_boundary",
                "excluded_gpu_annotation_count", "invalid_gpu_timeline_count"):
        counts.setdefault(key, 0)
    return {
        "schema": "ditserve.cuda_profile.v1",
        "status": "measured" if attribution_verified and activity_attribution_verified else "unverified",
        "timeline_status": "measured" if timeline_verified else "unverified",
        "attribution_status": "measured" if attribution_verified else "unverified",
        "kernel_attribution_status": "measured" if attribution_verified else "unverified",
        "activity_attribution_status": "measured" if activity_attribution_verified else "unverified",
        "reasons": all_reasons,
        "request": {
            "wall_ms": request_duration / 1000.0 if request_interval else None,
            "kernel_union_ms": kernel_union / 1000.0,
            "activity_union_ms": activity_union / 1000.0,
            "kernel_active_pct": _percent(kernel_union, request_duration) if timeline_verified else None,
            "activity_active_pct": _percent(activity_union, request_duration) if timeline_verified else None,
            "denominator": "CPU wan_profile/request annotation wall interval; clipped GPU interval union",
        },
        "gpu_window": {
            "window_ms": gpu_duration / 1000.0 if gpu_span else None,
            "kernel_active_pct": _percent(kernel_union, gpu_duration) if timeline_verified else None,
            "activity_active_pct": _percent(activity_union, gpu_duration) if timeline_verified else None,
            "denominator": "first-to-last real GPU activity within request, including CPU launch gaps",
        },
        "kernel_duration_ms": kernel_duration / 1000.0,
        "kernel_duration_share_denominator": "sum of real kernel event durations; concurrent streams count separately, not exclusive wall time",
        "kernel_categories": kernel_categories,
        "transformer_kernel_duration_ms": transformer_duration / 1000.0,
        "transformer_kernel_categories": transformer_categories,
        "transformer_share_denominator": "sum of real kernels whose linked CPU annotation ancestry includes transformer",
        "kernel_duration_share_sum_pct": math.fsum(row["share_pct"] or 0.0 for row in kernel_categories) if kernel_duration else None,
        "kernel_union_partition": {
            "kernel_union_ms": kernel_union / 1000.0,
            "sum_minus_union_ms": (kernel_duration - kernel_union) / 1000.0,
            "cross_category_overlap_ms": cross_category_overlap / 1000.0,
            "cross_category_overlap_share_pct": _percent(cross_category_overlap, kernel_union),
            "exclusive_category_share_sum_pct": math.fsum(row["exclusive_union_share_pct"] or 0.0 for row in kernel_categories) if kernel_union else None,
            "definition": "category-only union time plus explicit cross-category overlap partitions kernel union",
        },
        "memory_activities": [
            {"name": name, "event_count": counts[name + "_count"],
             "duration_sum_ms": math.fsum(end - start for start, end in type_intervals[name]) / 1000.0,
             "union_ms": _length(type_intervals[name]) / 1000.0}
            for name in ("memcpy", "memset")
        ],
        "stages": stages,
        "top_kernels": top,
        "top_kernels_retention": {
            "limit": top_kernel_limit, "sort": "descending summed real kernel duration",
            "distinct_kernel_names": len(name_aggregate), "omitted_name_count": len(omitted),
            "omitted_kernel_count": sum(row["kernel_count"] for row in omitted),
            "omitted_kernel_duration_ms": math.fsum(row["kernel_duration_ms"] for row in omitted),
            "event_details_retained": False,
        },
        "diagnostics": {**dict(counts), "devices": sorted(devices),
                        "streams": [list(value) for value in sorted(streams)],
                        "multistream": len(streams) > 1,
                        "trace_categories": dict(sorted(category_counts.items())),
                        "unknown_gpu_categories": dict(unknown_gpu_categories),
                        "gpu_link_methods": dict(link_counts), "kernel_link_methods": dict(status_counts)},
        "limitations": [
            "GPU activity coverage measures time with kernels/memory operations, not SM/Tensor Core/DRAM throughput utilization.",
            "Attention categories include their projection/normalization kernels enclosed by leaf annotations; transformer_other stays explicit.",
            "Nested stage rows overlap by design and must not be summed; mutually exclusive kernel_categories are the duration partition.",
            "Any unlinked memory activity makes stage activity rates unavailable, because its stage cannot be inferred; global timeline coverage remains independently valid.",
            "Multi-stream summed duration shares are work-time shares. Use exclusive union and overlap rows for elapsed-time attribution.",
            "Only one GPU and one complete request annotation are supported for verified rates.",
        ],
    }
