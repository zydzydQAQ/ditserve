"""Frozen request content, independent arrival processes and a C++ v2 adapter."""

import copy
import datetime
import math
import platform
import random
import re
from pathlib import Path

from .common import canonical_hash, positive, read_json, seal, verify_seal


SCHEMA = "ditserve.benchmark.trace.v1"
PATTERNS = ("poisson", "gamma", "foreground_burst", "offline_all", "replay", "batch_burst")
CORE_FIELDS = ("request_id", "arrival_ms", "deadline_ms", "model", "height", "width", "frames",
               "steps", "seed", "cache_threshold", "cache_enabled")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("%s must be an integer >= %s" % (name, minimum))
    return value


def _stream(seed, name):
    return random.Random(int(canonical_hash({"seed": seed, "stream": name}), 16))


def _checked_shapes(config, profile):
    if profile.get("schema_version") != "ditserve.benchmark.profile.v1":
        raise ValueError("unsupported service profile schema")
    if profile.get("status", "completed") != "completed":
        raise ValueError("trace requires a completed standalone service profile")
    shapes = config.get("shapes")
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError("config.shapes must be a nonempty mapping")
    profiles = profile.get("shapes", {})
    for shape_id, shape in shapes.items():
        if not isinstance(shape_id, str) or not shape_id or not isinstance(shape, dict):
            raise ValueError("shape IDs must be nonempty strings with shape dictionaries")
        positive(shape.get("weight", 1.0), "shape weight")
        measured = profiles.get(shape_id)
        if not isinstance(measured, dict):
            raise ValueError("missing service profile for shape %s" % shape_id)
        positive(measured.get("service_ms"), "profile service_ms")
        for key in ("height", "width", "frames", "steps"):
            _integer(shape.get(key), "shape " + key)
            if measured.get(key) != shape[key]:
                raise ValueError("profile dimensions do not match shape %s: %s" % (shape_id, key))
    return shapes


def _execution_settings(config):
    cache = config.get("cache", {})
    return {"model": config.get("model"),
            "cache_enabled": cache.get("enabled", False),
            "cache_threshold": cache.get("threshold", .08),
            "max_continuous": cache.get("max_continuous", 3),
            "guidance_scale": config.get("guidance_scale", 5.0),
            "negative_prompt": config.get("negative_prompt", "")}


def _profile_compatible(config, profile):
    if "config" in profile and _execution_settings(config) != _execution_settings(profile["config"]):
        raise ValueError("service profile model, cache or generation settings differ from trace config")


def _replay_timezone(value):
    if value == "UTC":
        return datetime.timezone.utc
    match = re.fullmatch(r"([+-])(\d{2}):(\d{2})", str(value))
    if not match or int(match.group(2)) > 23 or int(match.group(3)) > 59:
        raise ValueError("replay_timezone must be UTC or an explicit offset such as +08:00")
    minutes = int(match.group(2)) * 60 + int(match.group(3))
    return datetime.timezone(datetime.timedelta(minutes=minutes * (1 if match.group(1) == "+" else -1)))


def _timestamp_ms(value, timezone="UTC"):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return positive(value, "timestamp in seconds", allow_zero=True) * 1000
    if not isinstance(value, str):
        raise ValueError("timestamp must be ISO-8601, native VidProM time or Unix seconds")
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.datetime.strptime(value, "%a %b %d %H:%M:%S %Y")
        except ValueError:
            try:
                return positive(float(value), "timestamp in seconds", allow_zero=True) * 1000
            except ValueError:
                raise ValueError("timestamp must be ISO-8601, native VidProM time or Unix seconds")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_replay_timezone(timezone))
    return positive(parsed.timestamp(), "timestamp in seconds", allow_zero=True) * 1000


def _replay_rows(replay, samples, timezone="UTC"):
    if replay is None:
        rows = [{"sample_id": sample["sample_id"], "timestamp": sample.get("timestamp")} for sample in samples]
        source = "dataset_collection_timestamps"
    else:
        rows = read_json(replay) if isinstance(replay, (str, Path)) else copy.deepcopy(replay)
        if isinstance(rows, dict):
            if "sha256" in rows:
                verify_seal(rows)
            rows = rows.get("requests", rows.get("arrivals"))
        source = "provided_replay_timestamps"
    if not isinstance(rows, list) or not rows:
        raise ValueError("replay requires a nonempty list of timestamp records")
    prepared = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each replay row must be a dictionary")
        time = (positive(row["arrival_ms"], "replay arrival_ms", allow_zero=True)
                if "arrival_ms" in row else _timestamp_ms(row.get("timestamp"), timezone))
        prepared.append((time, row))
    # Stable sort preserves the source order of simultaneous arrivals.
    prepared.sort(key=lambda item: item[0])
    origin = prepared[0][0]
    return [(time - origin, row) for time, row in prepared], source


def build_trace(config, dataset, profile, pattern="poisson", rate_rps=None, load=1.0,
                cv=2.0, seed=20261012, count=None, replay=None, time_scale=1.0):
    """Build a sealed, open-loop trace without scheduling or GPU execution.

    All random streams are independent. Changing rate, load or arrival pattern
    leaves request contents unchanged. ``load`` is relative to the weighted
    mean standalone service profile (one server), not a measured achieved load.
    Replay changes timestamps only unless rows explicitly identify sample IDs.
    """
    if pattern not in PATTERNS:
        raise ValueError("unknown arrival pattern: %s" % pattern)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    verify_seal(dataset)
    verify_seal(profile)
    if dataset.get("schema_version") != "ditserve.benchmark.dataset.v1":
        raise ValueError("unsupported prompt dataset schema")
    if profile.get("dataset_sha256", dataset["sha256"]) != dataset["sha256"]:
        raise ValueError("service profile belongs to a different dataset snapshot")
    if "sha256" in config:
        verify_seal(config)
    shapes = _checked_shapes(config, profile)
    _profile_compatible(config, profile)
    model = config.get("model", {})
    if not isinstance(model, dict) or not model.get("id") or not model.get("revision"):
        raise ValueError("config.model requires id and a pinned revision")
    profile_model = profile.get("config", {}).get("model")
    if profile_model is not None and profile_model != model:
        raise ValueError("service profile belongs to a different model revision")
    samples = [sample for sample in dataset.get("samples", [])
               if sample.get("service_split", sample.get("split")) == "test"]
    if not samples:
        raise ValueError("dataset has no held-out service test prompts")
    sample_map = {sample["sample_id"]: sample for sample in samples}
    if len(sample_map) != len(samples):
        raise ValueError("dataset contains duplicate sample IDs")
    for sample in samples:
        if not isinstance(sample.get("prompt"), str) or not sample["prompt"].strip():
            raise ValueError("dataset prompt must be nonempty text")
    requested_count = config.get("count", 300) if count is None else count
    _integer(requested_count, "count")
    scale = positive(config.get("slo_scale", 2.5), "slo_scale")
    positive(load, "load")
    positive(cv, "cv")
    positive(time_scale, "time_scale")
    if rate_rps is not None:
        positive(rate_rps, "rate_rps")
    if pattern != "replay" and replay is not None:
        raise ValueError("replay data requires pattern=replay")
    if pattern != "replay" and time_scale != 1.0:
        raise ValueError("time_scale is only supported for replay")
    if pattern in ("offline_all", "replay") and rate_rps is not None:
        raise ValueError("rate_rps does not apply to offline_all or replay")
    cache = config.get("cache", {})
    enabled = cache.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("cache.enabled must be boolean")
    threshold = positive(cache.get("threshold", 0.08), "cache.threshold", allow_zero=True)
    maximum = _integer(cache.get("max_continuous", 3), "cache.max_continuous")
    guidance = positive(config.get("guidance_scale", 5.0), "guidance_scale", allow_zero=True)
    negative_prompt = config.get("negative_prompt", "")
    if not isinstance(negative_prompt, str):
        raise ValueError("negative_prompt must be text")
    shape_ids = sorted(shapes)
    weights = [float(shapes[key].get("weight", 1)) for key in shape_ids]
    total_weight = sum(weights)
    positive(total_weight, "total shape weight")
    mean_service = sum(weight * profile["shapes"][key]["service_ms"]
                       for key, weight in zip(shape_ids, weights)) / total_weight
    if pattern == "foreground_burst":
        if not {"short", "long"}.issubset(shapes):
            raise ValueError("foreground_burst requires short and long shape IDs")
        mean_service = (profile["shapes"]["long"]["service_ms"] +
                        3 * profile["shapes"]["short"]["service_ms"]) / 4
    positive(mean_service, "weighted standalone service time")
    effective_rate = float(rate_rps) if rate_rps is not None else load * 1000 / mean_service
    positive(effective_rate, "effective rate_rps")
    arrivals_rng = _stream(seed, "arrivals")
    shape_rng = _stream(seed, "shapes")
    prompt_rng = _stream(seed, "prompts")
    generation_rng = _stream(seed, "generation_seeds")
    burst_size = _integer(config.get("batch_burst_size", 3), "batch_burst_size")
    timezone = config.get("replay_timezone", "UTC")
    replay_rows, replay_source = (None, None)
    if pattern == "replay":
        _replay_timezone(timezone)
        replay_rows, replay_source = _replay_rows(replay, samples, timezone)
        if requested_count > len(replay_rows):
            raise ValueError("replay has fewer timestamps than the requested count")
        replay_rows = replay_rows[:requested_count]
    requests = []
    arrival = 0.0
    for index in range(requested_count):
        shape_id = shape_rng.choices(shape_ids, weights=weights, k=1)[0]
        if pattern == "foreground_burst":
            shape_id = "long" if index % 4 == 0 else "short"
        shape = shapes[shape_id]
        sample = samples[prompt_rng.randrange(len(samples))]
        generation_seed = generation_rng.randrange(2 ** 31)
        if pattern == "replay":
            arrival = replay_rows[index][0] * time_scale
            sample_id = replay_rows[index][1].get("sample_id")
            if sample_id is not None:
                if sample_id not in sample_map:
                    raise ValueError("replay sample_id is missing or belongs to development prompts")
                sample = sample_map[sample_id]
        elif pattern == "offline_all":
            arrival = 0.0
        elif pattern == "foreground_burst":
            cycle_ms = 4000 / effective_rate
            arrival = (index // 4) * cycle_ms + (0 if index % 4 == 0 else cycle_ms / 4)
        elif pattern == "batch_burst":
            if index % burst_size == 0:
                arrival += arrivals_rng.expovariate(effective_rate / (1000 * burst_size))
        elif pattern == "gamma":
            arrival += arrivals_rng.gammavariate(1 / (cv * cv), 1000 * cv * cv / effective_rate)
        else:
            arrival += arrivals_rng.expovariate(effective_rate / 1000)
        requests.append({"request_id": "request-%06d" % index, "sample_id": sample["sample_id"],
                         "dataset": dataset["kind"], "prompt": sample["prompt"], "shape_id": shape_id,
                         "height": shape["height"], "width": shape["width"], "frames": shape["frames"],
                         "steps": shape["steps"], "seed": generation_seed, "model": model["id"],
                         "revision": model["revision"], "cache_enabled": enabled,
                         "cache_threshold": threshold, "max_continuous": maximum,
                         "guidance_scale": guidance, "negative_prompt": negative_prompt,
                         "arrival_ms": arrival,
                         "deadline_ms": arrival + scale * profile["shapes"][shape_id]["service_ms"]})
    arrival_config = {"pattern": pattern, "seed": seed, "count": requested_count,
                      "rate_rps": effective_rate if pattern not in ("offline_all", "replay") else None,
                      "nominal_load": load if rate_rps is None and pattern not in ("offline_all", "replay") else None,
                      "mean_standalone_service_ms": mean_service,
                      "load_semantics": "weighted standalone service time on one server; not achieved load",
                      "cv": cv if pattern == "gamma" else None,
                      "burst_size": burst_size if pattern == "batch_burst" else None,
                      "time_scale": time_scale if pattern == "replay" else None,
                      "replay_source": replay_source,
                      "replay_timezone": timezone if pattern == "replay" else None,
                      "naive_timestamp_semantics": "Assume the recorded replay_timezone; source did not specify a timezone." if pattern == "replay" else None,
                      "replay_sha256": canonical_hash(replay_rows) if replay_rows is not None else None,
                      "foreground_burst_protocol": "cycle=4/rate seconds; one long at cycle start, three short at cycle quarter; truncate last cycle" if pattern == "foreground_burst" else None,
                      "first_arrival": "zero" if pattern in ("offline_all", "replay", "foreground_burst") else "sampled_gap",
                      "rng": "python.random.mt19937.independent-sha256-streams.v1",
                      "python_version": platform.python_version(),
                      "open_loop": True}
    trace = seal({"schema_version": SCHEMA, "config": copy.deepcopy(config),
                  "purpose": "offline_diagnostic" if pattern == "offline_all" else "online_serving",
                  "serving_performance_evidence": pattern != "offline_all",
                  "dataset_sha256": dataset["sha256"], "profile_sha256": profile["sha256"],
                  "dataset": {key: copy.deepcopy(dataset[key]) for key in
                              ("kind", "source", "sha256", "selection", "split_protocol") if key in dataset},
                  "profile": copy.deepcopy(profile), "arrival": arrival_config,
                  "requests": requests})
    validate_trace(trace)
    return trace


def validate_trace(trace):
    verify_seal(trace)
    if trace.get("schema_version") != SCHEMA:
        raise ValueError("unsupported benchmark trace schema")
    config, profile = trace.get("config", {}), trace.get("profile", {})
    verify_seal(profile)
    if trace.get("profile_sha256") != profile["sha256"]:
        raise ValueError("trace profile identity mismatch")
    if trace.get("dataset_sha256") != trace.get("dataset", {}).get("sha256"):
        raise ValueError("trace dataset identity mismatch")
    shapes = _checked_shapes(config, profile)
    _profile_compatible(config, profile)
    requests = trace.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("trace requires a nonempty request list")
    ids = set()
    previous = -1.0
    scale = positive(config.get("slo_scale", 2.5), "slo_scale")
    model = config.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("id"), str) or not isinstance(model.get("revision"), str):
        raise ValueError("frozen model requires string id and revision")
    dev_ids = set(profile.get("dev_sample_ids", []))
    quality_protocol = "quality_protocol" in trace
    if quality_protocol:
        from .protocols import validate_vbench_trace_protocol
        validate_vbench_trace_protocol(trace)
    elif trace.get("purpose") == "independent_quality_generation":
        raise ValueError("independent quality generation requires a complete validated protocol")
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("request must be a dictionary")
        identity = request.get("request_id")
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("request IDs must be nonempty and unique")
        ids.add(identity)
        for field in ("sample_id", "dataset", "prompt", "model", "revision"):
            if not isinstance(request.get(field), str) or not request[field].strip():
                raise ValueError("request %s must be nonempty text" % field)
        if request["sample_id"] in dev_ids and not quality_protocol:
            raise ValueError("development profile sample leaked into serving test workload")
        shape_id = request.get("shape_id")
        if shape_id not in shapes:
            raise ValueError("request references an unknown shape")
        for field in ("height", "width", "frames", "steps"):
            _integer(request.get(field), field)
            if request[field] != shapes[shape_id][field]:
                raise ValueError("request differs from frozen shape")
        if request["model"] != config["model"]["id"] or request["revision"] != config["model"]["revision"]:
            raise ValueError("request differs from frozen model revision")
        _integer(request.get("seed"), "request seed", minimum=0)
        _integer(request.get("max_continuous"), "max_continuous")
        if not isinstance(request.get("cache_enabled"), bool):
            raise ValueError("cache_enabled must be boolean")
        positive(request.get("cache_threshold"), "cache_threshold", allow_zero=True)
        positive(request.get("guidance_scale"), "guidance_scale", allow_zero=True)
        if not isinstance(request.get("negative_prompt"), str):
            raise ValueError("negative_prompt must be text")
        for field, expected in _execution_settings(config).items():
            if field != "model" and request[field] != expected:
                raise ValueError("request generation settings differ from frozen config: " + field)
        arrival = positive(request.get("arrival_ms"), "arrival_ms", allow_zero=True)
        deadline = positive(request.get("deadline_ms"), "deadline_ms", allow_zero=True)
        if arrival < previous or deadline < arrival:
            raise ValueError("arrivals must be ordered and deadlines must follow arrivals")
        expected = arrival + scale * profile["shapes"][shape_id]["service_ms"]
        if not math.isclose(deadline, expected, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("deadline differs from frozen standalone SLO budget")
        previous = arrival
    if trace.get("arrival", {}).get("count") != len(requests):
        raise ValueError("trace count differs from arrival configuration")
    return trace


def to_core_trace(trace):
    """Project metadata-rich requests onto the strict historical C++ v2 schema."""
    validate_trace(trace)
    return [{field: request[field] for field in CORE_FIELDS} for request in trace["requests"]]
