"""Bridge the separate VBench generation protocol to frozen executable traces.

These all-at-once requests exist to produce quality artifacts. They are not a
held-out serving workload, and their timing cannot support serving comparisons.
"""

import copy

from .common import canonical_hash, positive, seal, verify_seal
from .evaluation import (VBENCH_DIMENSIONS, VBENCH_REPOSITORY, VBENCH_REVISION,
                         VBENCH_VERSION)


PURPOSE = "independent_quality_generation"
PROTOCOL_SCHEMA = "ditserve.benchmark.quality_protocol.v1"
MANIFEST_SCHEMA = "ditserve.benchmark.vbench_generation.v1"


def _integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def generation_settings(config, shape_id):
    """Freeze the flat generation fields understood by the VBench exporter."""
    if not isinstance(config, dict):
        raise ValueError("generation config must be a dictionary")
    if "sha256" in config:
        verify_seal(config)
    model = config.get("model", {})
    if not isinstance(model, dict) or any(not isinstance(model.get(key), str) or
                                          not model[key].strip() for key in ("id", "revision")):
        raise ValueError("generation settings require a frozen model id and revision")
    shape = config.get("shapes", {}).get(shape_id)
    if not isinstance(shape, dict):
        raise ValueError("unknown VBench generation shape")
    cache = config.get("cache", {})
    if not isinstance(cache, dict) or type(cache.get("enabled", False)) is not bool:
        raise ValueError("cache.enabled must be boolean")
    negative = config.get("negative_prompt", "")
    if not isinstance(negative, str):
        raise ValueError("negative_prompt must be text")
    result = {"model": model["id"], "revision": model["revision"],
              "cache_enabled": cache.get("enabled", False),
              "cache_threshold": positive(cache.get("threshold", .08), "cache threshold", allow_zero=True),
              "max_continuous": _integer(cache.get("max_continuous", 3), "max_continuous"),
              "guidance_scale": positive(config.get("guidance_scale", 5.0), "guidance_scale", allow_zero=True),
              "negative_prompt": negative, "fps": 16}
    for field in ("height", "width", "frames", "steps"):
        result[field] = _integer(shape.get(field), field)
    return result


def _manifest_inputs(config, dataset, manifest):
    """Verify the complete source-derived protocol, including resealed inputs."""
    verify_seal(dataset)
    verify_seal(manifest)
    if dataset.get("schema_version") != "ditserve.benchmark.dataset.v1" or dataset.get("kind") != "vbench":
        raise ValueError("VBench generation requires a prepared VBench dataset")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported VBench generation manifest schema")
    for field, expected in (("repository", VBENCH_REPOSITORY), ("revision", VBENCH_REVISION),
                            ("version", VBENCH_VERSION), ("seeds_per_prompt", 5),
                            ("flickering_seeds_per_prompt", 25)):
        if manifest.get(field) != expected:
            raise ValueError("VBench manifest differs from the pinned protocol: " + field)
    source = dataset.get("source", {})
    digest = source.get("sha256")
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(char not in "0123456789abcdef" for char in digest) or
            manifest.get("full_info_sha256") != digest):
        raise ValueError("VBench manifest and dataset source file hash differ")
    if source.get("repository") != VBENCH_REPOSITORY or source.get("revision") not in (None, VBENCH_REVISION):
        raise ValueError("VBench dataset does not match the pinned source revision")
    dimensions = manifest.get("dimensions")
    if (not isinstance(dimensions, list) or not dimensions or
            any(not isinstance(item, str) for item in dimensions) or
            len(dimensions) != len(set(dimensions)) or set(dimensions) - set(VBENCH_DIMENSIONS)):
        raise ValueError("invalid VBench dimension selection")
    scope = "full_dimension_suite" if set(dimensions) == set(VBENCH_DIMENSIONS) else "dimension_subset"
    if manifest.get("scope") != scope:
        raise ValueError("VBench manifest scope differs from its dimensions")
    settings = manifest.get("generation_settings")
    matches = [shape_id for shape_id in config.get("shapes", {})
               if generation_settings(config, shape_id) == settings]
    if len(matches) != 1:
        raise ValueError("VBench generation settings must match exactly one configured shape")
    shape_id = matches[0]
    samples = dataset.get("samples")
    selection = dataset.get("selection", {})
    if (not isinstance(samples, list) or not samples or
            selection.get("method") != "full_official_list" or
            selection.get("rows_seen") != len(samples) or selection.get("samples_retained") != len(samples)):
        raise ValueError("VBench generation requires the complete prepared metadata list")
    expected, seen, sample_ids, by_prompt = [], set(), set(), {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("invalid VBench sample")
        metadata = sample.get("metadata", {})
        prompt = sample.get("prompt")
        identity = sample.get("sample_id")
        if (not isinstance(prompt, str) or not prompt.strip() or
                any(char in prompt for char in ("/", "\\", "\0")) or
                not isinstance(metadata, dict) or metadata.get("prompt_en") != prompt):
            raise ValueError("VBench prompt differs from its official metadata or is an unsafe filename")
        if not isinstance(identity, str) or not identity or identity in sample_ids:
            raise ValueError("VBench sample IDs must be nonempty and unique")
        sample_ids.add(identity)
        if sample.get("split") != "evaluation" or sample.get("service_split") not in ("dev", "test"):
            raise ValueError("VBench quality entries must retain their independent evaluation split")
        tags = metadata.get("dimension")
        if (not isinstance(tags, list) or not tags or any(tag not in VBENCH_DIMENSIONS for tag in tags) or
                sample.get("dimensions") != tags):
            raise ValueError("VBench sample dimensions differ from official metadata")
        by_prompt.setdefault(prompt, sample)
        for dimension in tags:
            if dimension not in dimensions:
                continue
            if (dimension, prompt) in seen:
                raise ValueError("duplicate official VBench dimension/prompt")
            seen.add((dimension, prompt))
            for seed in range(25 if dimension == "temporal_flickering" else 5):
                expected.append({"dimension": dimension, "prompt": prompt, "seed": seed,
                                 "video_path": "%s/%s-%d.mp4" % (dimension, prompt, seed)})
    if set(dimensions) != {row["dimension"] for row in expected}:
        raise ValueError("VBench dataset does not cover the requested dimensions")
    if canonical_hash(manifest.get("requests")) != canonical_hash(expected):
        raise ValueError("VBench manifest must contain every source prompt/dimension/seed exactly once")
    unique = {}
    for row in expected:
        key = (row["prompt"], row["seed"])
        if key not in unique:
            unique[key] = {"sample_id": by_prompt[row["prompt"]]["sample_id"],
                           "prompt": row["prompt"], "seed": row["seed"]}
    return shape_id, settings, list(unique.values())


def validate_vbench_trace_protocol(trace):
    """Validate the sole explicit exception to serving dev/test separation.

    Called by workload.validate_trace; do not call that validator here, because
    the remaining request/model/profile checks are owned by the caller.
    """
    protocol = trace.get("quality_protocol")
    if (not isinstance(protocol, dict) or protocol.get("schema_version") != PROTOCOL_SCHEMA or
            protocol.get("kind") != "vbench" or trace.get("purpose") != PURPOSE or
            trace.get("serving_performance_evidence") is not False or
            protocol.get("serving_performance_evidence") is not False):
        raise ValueError("invalid independent VBench quality protocol")
    dataset, manifest = protocol.get("dataset"), protocol.get("manifest")
    shape_id, settings, expected = _manifest_inputs(trace.get("config", {}), dataset, manifest)
    if (protocol.get("manifest_sha256") != manifest["sha256"] or
            protocol.get("full_info_sha256") != manifest["full_info_sha256"] or
            protocol.get("scope") != manifest["scope"] or
            protocol.get("expected_videos") != len(manifest["requests"]) or
            protocol.get("unique_generations") != len(expected) or
            protocol.get("shape_id") != shape_id or
            protocol.get("dimensions") != manifest["dimensions"]):
        raise ValueError("quality protocol identity or counts differ from its manifest")
    if (trace.get("dataset_sha256") != dataset["sha256"] or
            trace.get("dataset", {}).get("source") != dataset["source"]):
        raise ValueError("quality protocol dataset differs from trace provenance")
    arrival = trace.get("arrival", {})
    if arrival.get("pattern") != "offline_all" or arrival.get("count") != len(expected):
        raise ValueError("independent VBench generation must be a complete offline_all trace")
    requests = trace.get("requests")
    if not isinstance(requests, list) or len(requests) != len(expected):
        raise ValueError("quality trace has missing or duplicate generations")
    for request, row in zip(requests, expected):
        if not isinstance(request, dict) or any(request.get(key) != value for key, value in row.items()):
            raise ValueError("quality trace prompt, sample or seed differs from its manifest")
        if (request.get("shape_id") != shape_id or request.get("dataset") != "vbench" or
                request.get("arrival_ms") != 0 or
                any(request.get(key) != value for key, value in settings.items())):
            raise ValueError("quality trace generation settings differ from its manifest")
    return trace


def build_vbench_trace(config, dataset, profile, manifest):
    """Generate once per unique prompt/seed and reuse outputs across dimensions."""
    # Local import lets workload call the independent protocol validator without
    # a module-import cycle.
    from .workload import build_trace, validate_trace

    shape_id, settings, rows = _manifest_inputs(config, dataset, manifest)
    seeds = config.get("trace_seeds", [20261012])
    if not isinstance(seeds, list) or not seeds or type(seeds[0]) is not int:
        raise ValueError("trace_seeds must contain an integer reproducibility seed")
    trace = build_trace(config, dataset, profile, pattern="offline_all", seed=seeds[0], count=len(rows))
    service = profile["shapes"][shape_id]["service_ms"]
    deadline = positive(config.get("slo_scale", 2.5), "slo_scale") * service
    template = trace["requests"][0]
    requests = []
    for index, row in enumerate(rows):
        request = dict(template, **settings)
        request.update(row)
        request.update(request_id="vbench-%06d" % index, shape_id=shape_id,
                       arrival_ms=0.0, deadline_ms=deadline)
        requests.append(request)
    trace["requests"] = requests
    trace["arrival"]["mean_standalone_service_ms"] = service
    trace["arrival"]["rng"] = "official_vbench_prompt_seeds.v1"
    trace["arrival"]["load_semantics"] = "independent quality generation; no serving load or latency claim"
    trace["purpose"] = PURPOSE
    trace["serving_performance_evidence"] = False
    trace["quality_protocol"] = {
        "schema_version": PROTOCOL_SCHEMA, "kind": "vbench", "shape_id": shape_id,
        "manifest_sha256": manifest["sha256"], "full_info_sha256": manifest["full_info_sha256"],
        "scope": manifest["scope"], "dimensions": copy.deepcopy(manifest["dimensions"]),
        "expected_videos": len(manifest["requests"]), "unique_generations": len(requests),
        "serving_performance_evidence": False,
        "split_semantics": "Full official quality prompts may include service dev prompts; not held-out serving evidence.",
        "reuse_semantics": "Identical prompt, seed and generation settings share one generated artifact across dimensions.",
        "manifest": copy.deepcopy(manifest), "dataset": copy.deepcopy(dataset)}
    trace = seal(trace)
    validate_trace(trace)
    return trace
