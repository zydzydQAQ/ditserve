"""Reproducible CPU-only request traces for the D1 dispatch comparison."""

import copy
import math

from ..directions.direction1 import frozen_requests


SCENARIOS = {
    "original_burst": {"load": .9, "pattern": "burst", "video_deadline_multiplier": 2.5},
    "tight_deadline": {"load": .9, "pattern": "ordinary", "video_deadline_multiplier": 1.25},
    "original_ordinary": {"load": .5, "pattern": "ordinary", "video_deadline_multiplier": 2.5},
    "video_contention": {"load": .9, "pattern": "burst", "video_deadline_multiplier": 1.5},
}


def _service(cost_model, key):
    value = cost_model[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(key + " must be a finite positive independent service time")
    return float(value)


def _prompts(config, kind):
    prompts = config["prompts"]
    if isinstance(prompts, dict):
        prompts = prompts[kind]
    if not isinstance(prompts, (tuple, list)) or not prompts or any(not isinstance(p, str) or not p for p in prompts):
        raise ValueError("prompts must be a nonempty list, or lists indexed by modality")
    return prompts


def make_trace(config, cost_model, scenario, seed):
    """Freeze 24 arrivals without consulting a scheduler or runtime outcome.

    Contention keeps the original burst trace's arrivals byte-for-byte in the
    JSON representation.  It converts r006 after generation, so extra video
    work does not lower the arrival rate through a changed mean service cost.
    It therefore has the *base trace's* load label, not a claim of load 0.9
    after the conversion.
    """
    if scenario not in SCENARIOS:
        raise ValueError("unknown scheduling scenario: " + str(scenario))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("trace seed must be an integer")
    shape = config["shape"]
    if not isinstance(shape, dict) or any(type(shape.get(key)) is not int or shape[key] <= 0
                                          for key in ("height", "width", "frames")):
        raise ValueError("shape requires positive integer height, width and frames")
    video_ms, image_ms = _service(cost_model, "video_service_ms"), _service(cost_model, "image_service_ms")
    spec = SCENARIOS[scenario]
    rows = frozen_requests(count=24, seed=seed, load=spec["load"], pattern=spec["pattern"],
                           video_ms=video_ms, image_ms=image_ms)
    if scenario == "video_contention":
        rows[6]["kind"] = "video"
        rows[6]["service_ms"] = video_ms
    counts = {"video": 0, "image": 0}
    for row in rows:
        kind = row["kind"]
        multiplier = spec["video_deadline_multiplier"] if kind == "video" else 2.5
        row["deadline_ms"] = row["arrival_ms"] + multiplier * row["service_ms"]
        prompts = _prompts(config, kind)
        row.update(prompt=prompts[counts[kind] % len(prompts)], shape=copy.deepcopy(shape),
                   scenario=scenario, trace_seed=seed)
        counts[kind] += 1
    return rows
