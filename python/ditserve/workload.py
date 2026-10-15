"""Workload inputs for the control plane: generator payloads and trace files."""

import json
from pathlib import Path


def generator_payload(count=64, seed=7, arrival_rate=8.0, pattern="steady", slo_scale=2.0,
                      shapes=((512, 512, 1), (768, 768, 1)), step_choices=(20, 30), cache_threshold=0.15):
    """Keyword arguments mirror the schema-v2 `generator` object of ditserve-core."""
    if pattern not in ("steady", "bursty"):
        raise ValueError("pattern must be steady or bursty")
    return {"count": count, "seed": seed, "arrival_rate": arrival_rate, "pattern": pattern,
            "slo_scale": slo_scale, "shapes": [list(shape) for shape in shapes],
            "step_choices": list(step_choices), "cache_threshold": cache_threshold}


def load_trace(path):
    """A trace is a JSON array of RequestSpec objects with absolute times in ms."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("trace must be a JSON array of RequestSpec objects")
    return value
