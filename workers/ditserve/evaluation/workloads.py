"""Deterministic mixed-modality serving traces with independent arrival streams."""

import copy
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

SCHEMA = "ditserve.benchmark.mixed-workload.v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _rng(seed, *parts):
    return random.Random(int(_digest([seed, *parts])[:16], 16))


def gpu_service_ms(klass):
    if "gpu_service_ms" in klass:
        return float(klass["gpu_service_ms"])
    if klass["kind"] == "image":
        return float(klass["service_ms"])
    return float(klass["prepare_ms"] + klass["steps"] * klass["step_compute_ms"] + klass["finish_ms"])


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive")
    return value


def _quota(total, parts):
    if type(parts) is not int or parts < 1:
        raise ValueError("quota parts must be positive integers")
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def _ratio_count(total, fraction):
    value = total * fraction
    count = int(round(value))
    if count < 0 or count > total or abs(value - count) > 1e-9:
        raise ValueError("ratio does not produce an exact integer class count")
    return count


def _prompt_choices(pool, count, seed, kind):
    if not isinstance(pool, dict) or not isinstance(pool.get("samples"), list) or not pool["samples"]:
        raise ValueError("prompt pool must be a nonempty sealed dataset")
    unique = {}
    for sample in pool["samples"]:
        prompt = sample.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt pool contains empty text")
        unique.setdefault(prompt, sample)
    if len(unique) < count:
        raise ValueError("not enough distinct prompts for frozen workload")
    ordered = sorted(unique.values(), key=lambda sample: _digest([seed, kind, sample["prompt"]]))
    return ordered[:count]


def _pool_manifest(pool):
    return dict(kind=pool.get("kind"), source=pool.get("source"), selection=pool.get("selection"))


def build_trace(classes, prompt_pools, windows, *, slots, seed, quality_per_kind=0):
    if type(slots) is not int or slots < 1:
        raise ValueError("slots must be a positive integer")
    if type(seed) is not int or type(quality_per_kind) is not int or quality_per_kind < 0:
        raise ValueError("seed must be an integer and quality_per_kind nonnegative")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("classes must be a nonempty mapping")
    for name, klass in classes.items():
        if klass.get("class", name) != name or klass.get("kind") not in ("image", "video"):
            raise ValueError("class identity or kind is invalid")
        _positive(klass["service_ms"], "service_ms")
        _positive(gpu_service_ms(klass), "gpu_service_ms")
        if not isinstance(klass.get("spec"), dict):
            raise ValueError("class spec must be an object")
    kinds = {klass["kind"] for klass in classes.values()}
    if kinds != {"image", "video"}:
        raise ValueError("mixed workload requires image and video classes")
    if not isinstance(windows, list) or not windows:
        raise ValueError("windows must be a nonempty list")
    if len(windows) > 1 and any(window.get("pattern", "poisson") in ("offline", "offline_all") for window in windows):
        raise ValueError("ratio(t) requires online arrival patterns")
    video_classes = [name for name, klass in classes.items() if klass["kind"] == "video"]
    total_by_kind = {kind: 0 for kind in kinds}
    for window in windows:
        count = window["count"]
        if type(count) is not int or count < 1:
            raise ValueError("window count must be a positive integer")
        video_count = _ratio_count(count, window.get("video_fraction", 0.5))
        total_by_kind["video"] += video_count
        total_by_kind["image"] += count - video_count
    prompts = {kind: _prompt_choices(prompt_pools[kind], total_by_kind[kind], seed, kind) for kind in kinds}
    prompt_index = {kind: 0 for kind in kinds}
    quality_seen = {kind: 0 for kind in kinds}
    rows, manifest_windows = [], []
    start = 0.0
    for window_index, window in enumerate(windows):
        count = window["count"]
        if type(count) is not int or count < 1:
            raise ValueError("window count must be a positive integer")
        video_fraction = window.get("video_fraction", 0.5)
        video_count = _ratio_count(count, video_fraction)
        image_count = count - video_count
        pattern = window.get("pattern", "poisson")
        if pattern not in ("offline", "offline_all", "poisson", "bursty"):
            raise ValueError("unknown arrival pattern")
        load = _positive(window.get("load", 1.0), "load")
        class_counts = {"video": _quota(video_count, len(video_classes)) if video_classes else [],
                        "image": [image_count]}
        class_names = {"video": video_classes, "image": [name for name, klass in classes.items() if klass["kind"] == "image"]}
        if any(len(class_names[kind]) != len(class_counts[kind]) for kind in ("image", "video")):
            raise ValueError("each modality needs its own class quota")
        events = []
        gpu_by_kind = {}
        for kind in ("image", "video"):
            names = class_names[kind]
            kind_events = []
            for name, amount in zip(names, class_counts[kind]):
                kind_events.extend([name] * amount)
            _rng(seed, window_index, "classes", kind).shuffle(kind_events)
            events.extend((kind, name) for name in kind_events)
            gpu_by_kind[kind] = sum(classes[name]["service_ms"] * 0 + gpu_service_ms(classes[name]) * amount
                                    for name, amount in zip(names, class_counts[kind]))
        exposure = sum(gpu_by_kind.values()) / slots / load
        if not math.isfinite(exposure) or exposure <= 0:
            raise ValueError("window GPU exposure must be positive")
        bursts = window.get("bursts", 2) if pattern == "bursty" else 1
        fraction = float(window.get("burst_active_fraction", 0.25)) if pattern == "bursty" else 1.0
        if not 0 < fraction <= 1 or type(bursts) is not int or bursts < 1:
            raise ValueError("invalid burst count or active fraction")
        active_duration = exposure if pattern != "bursty" else exposure / bursts
        period = active_duration if pattern != "bursty" else active_duration / fraction
        active_windows = [] if pattern in ("offline", "offline_all") else [(start + index * period, active_duration)
                                                                          for index in range(bursts)]
        duration = 0.0 if pattern in ("offline", "offline_all") else bursts * period
        assignments = []
        for kind in ("image", "video"):
            quotas = _quota(class_counts[kind][0] if kind == "image" else video_count, bursts)
            for period_index, amount in enumerate(quotas):
                assignments.append((kind, period_index, amount))
        period_events = {(kind, index): [] for kind in kinds for index in range(bursts)}
        class_cursors = {kind: 0 for kind in kinds}
        grouped_events = {kind: [name for current, name in events if current == kind] for kind in kinds}
        for kind, period_index, amount in assignments:
            period_events[(kind, period_index)] = grouped_events[kind][class_cursors[kind]:class_cursors[kind] + amount]
            class_cursors[kind] += amount
        window_rows = []
        for kind, period_index, amount in assignments:
            selected = period_events[(kind, period_index)]
            if len(selected) != amount:
                raise RuntimeError("internal class quota mismatch")
            if pattern in ("offline", "offline_all"):
                timestamps = [0.0] * amount
            else:
                active_start, active_length = active_windows[period_index]
                arrival_rng = _rng(seed, window_index, period_index, "arrival", kind)
                timestamps = sorted(arrival_rng.random() * active_length for unused in range(amount))
            for name, timestamp in zip(selected, timestamps):
                klass = classes[name]
                sample = prompts[kind][prompt_index[kind]]
                prompt_index[kind] += 1
                request_id = "w%d-%s%04d" % (window_index, "v" if kind == "video" else "i", prompt_index[kind] - 1)
                arrival = 0.0 if pattern in ("offline", "offline_all") else active_windows[period_index][0] + timestamp
                service = float(klass["service_ms"])
                slo = float(klass["slo_ms"]) if "slo_ms" in klass else service * float(klass.get("slo_scale", 2.5))
                spec = copy.deepcopy(klass["spec"])
                spec.update(prompt=sample["prompt"], seed=int(_digest([seed, window_index, request_id, "content"])[:8], 16),
                            request_id=request_id)
                quality = quality_seen[kind] < quality_per_kind
                quality_seen[kind] += int(quality)
                row = dict(request_id=request_id, arrival_ms=arrival, service_ms=service, slo_ms=slo,
                           deadline_ms=arrival + slo, kind=kind, **{"class": name}, spec=spec,
                           dataset=prompt_pools[kind].get("kind"), dataset_sample_id=sample["sample_id"],
                           output_mode="file" if quality else "check", trace_seed=seed)
                if "worker" in klass:
                    row["worker"] = klass["worker"]
                window_rows.append(row)
        rows.extend(window_rows)
        counts = Counter(row["kind"] for row in window_rows)
        class_totals = Counter(row["class"] for row in window_rows)
        manifest_windows.append(dict(index=window_index, pattern=pattern, start_ms=start, duration_ms=duration,
                                     load=load, count=count, video_fraction=video_count / count,
                                     counts=dict(counts), class_counts=dict(class_totals),
                                     active_windows=[dict(start_ms=active_start, duration_ms=active_length)
                                                     for active_start, active_length in active_windows],
                                     rate_rps=None if pattern in ("offline", "offline_all") else count * 1000 / duration,
                                     gpu_ms=dict(gpu_by_kind)))
        start += duration
    rows.sort(key=lambda row: (row["arrival_ms"], row["kind"], row["request_id"]))
    counts = Counter(row["kind"] for row in rows)
    class_counts = Counter(row["class"] for row in rows)
    gpu_counts = {kind: sum(gpu_service_ms(classes[row["class"]]) for row in rows if row["kind"] == kind)
                  for kind in kinds}
    gpu_total = sum(gpu_counts.values())
    duration = start
    manifest = dict(schema=SCHEMA, seed=seed, slots=slots,
                    patterns=sorted({window["pattern"] for window in manifest_windows}),
                    counts=dict(counts), class_counts=dict(class_counts), total_duration_ms=duration,
                    rate_rps=None if duration == 0 else len(rows) * 1000 / duration,
                    gpu_ms=dict(gpu_counts),
                    gpu_time_fraction={kind: value / gpu_total for kind, value in gpu_counts.items()},
                    normalized_gpu_work=gpu_total / slots / duration if duration else manifest_windows[0]["load"],
                    quality_requests=dict(quality_seen), sources={kind: _pool_manifest(prompt_pools[kind]) for kind in kinds},
                    windows=manifest_windows)
    manifest["workload_sha256"] = _digest(rows)
    return dict(manifest=manifest, workload=rows)


def validate_trace(trace):
    if not isinstance(trace, dict) or trace.get("manifest", {}).get("schema") != SCHEMA:
        raise ValueError("unrecognized mixed workload schema")
    manifest, rows = trace["manifest"], trace["workload"]
    if manifest["workload_sha256"] != _digest(rows):
        raise ValueError("workload content differs from its frozen hash")
    ids = set()
    for row in rows:
        if row["request_id"] in ids or not row["spec"].get("prompt"):
            raise ValueError("duplicate request ID or empty prompt")
        ids.add(row["request_id"])
        if row["arrival_ms"] < 0 or row["deadline_ms"] < row["arrival_ms"]:
            raise ValueError("invalid arrival or deadline")
        if row["dataset_sample_id"] is None or row["spec"].get("request_id") != row["request_id"]:
            raise ValueError("request provenance differs from its identity")
    if Counter(row["kind"] for row in rows) != Counter(manifest["counts"]):
        raise ValueError("manifest modality counts differ")
    if Counter(row["class"] for row in rows) != Counter(manifest["class_counts"]):
        raise ValueError("manifest class counts differ")
    return True


def write_trace(trace, path):
    validate_trace(trace)
    target = Path(path)
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) == trace:
            return target
        raise ValueError("cannot overwrite a different frozen workload")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(trace, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return target


def load_trace(path):
    trace = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_trace(trace)
    return trace
