"""Bounded, paired serving ablations and evidence-based follow-up decisions.

Planning and summarizing are CPU-only. A shared trace profile freezes load/SLOs;
each execution profile supplies its own remaining costs and memory requirements.
Neither an estimate nor a generated plan starts or authorizes a GPU run.
"""

import copy
import math
from collections import defaultdict
from pathlib import Path

from .common import atomic_json, canonical_hash, file_hash, positive, read_json, seal, verify_seal
from .evaluation import generation_identity, resolve_artifact_path
from .reporting import _load_overlay, summarize_requests
from .workload import _checked_shapes, _profile_compatible, build_trace, validate_trace


VARIANTS = ("math_serial", "flash_serial", "flash_prefetch")
PAIRS = (("math_serial", "flash_serial"), ("flash_serial", "flash_prefetch"))
IDENTITY_FIELDS = ("gpu", "precision", "text_encoder_device", "scheduler", "model_id",
                   "revision", "torch_version", "cuda_runtime_version", "driver_version")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _attention(identity):
    attention = identity.get("attention", {})
    name = attention.get("sdpa_backend") if isinstance(attention, dict) else None
    return {"MATH": "math", "FLASH": "flash", "FLASH_ATTENTION": "flash"}.get(name)


def _check_identity(profile, variant, arguments, target_gpu=None):
    if profile.get("backend") != "wan" or profile.get("measurement_kind") != "measured_gpu":
        raise ValueError("optimization requires measured Wan profiles: " + variant)
    if profile.get("status") != "completed":
        raise ValueError("optimization requires completed profiles: " + variant)
    identity = profile.get("execution_identity", {})
    gpu = identity.get("gpu", {})
    if not gpu.get("uuid") or not gpu.get("name"):
        raise ValueError("profile lacks a physical GPU identity: " + variant)
    if target_gpu and target_gpu.lower() not in gpu["name"].lower():
        raise ValueError("profile is not from target GPU " + target_gpu)
    if _attention(identity) != arguments.get("attention"):
        raise ValueError("profile attention differs from variant " + variant)
    return identity


def execution_profile(trace):
    """Return validated variant costs without changing frozen arrival/SLO costs.

    Ordinary traces keep their existing profile. Optimization traces bind an
    independent profile, its hash, variant arguments, and shared request hash.
    Callers must still compare this identity with the actual execution adapter.
    """
    base = trace["profile"]
    if "execution_profile" not in trace:
        if "optimization" in trace:
            raise ValueError("optimization trace is missing its execution profile")
        return base
    profile = verify_seal(trace["execution_profile"])
    if trace.get("execution_profile_sha256") != profile["sha256"]:
        raise ValueError("execution profile hash mismatch")
    _checked_shapes(trace["config"], profile)
    _profile_compatible(trace["config"], profile)
    if profile.get("dataset_sha256") != trace.get("dataset_sha256"):
        raise ValueError("execution profile belongs to a different dataset")
    metadata = trace.get("optimization")
    if not isinstance(metadata, dict) or metadata.get("variant") not in VARIANTS:
        raise ValueError("execution profile requires optimization variant metadata")
    if metadata.get("shared_requests_sha256") != canonical_hash(trace["requests"]):
        raise ValueError("optimization request content, arrival or deadline changed")
    identity = _check_identity(profile, metadata["variant"], metadata.get("arguments", {}),
                               metadata.get("target_gpu"))
    reference = base.get("execution_identity", {})
    if any(identity.get(key) != reference.get(key) for key in IDENTITY_FIELDS):
        raise ValueError("variant profiles must use the same physical GPU and execution recipe")
    if set(profile.get("dev_sample_ids", [])) != set(base.get("dev_sample_ids", [])):
        raise ValueError("variant profiles must use the same development split")
    return profile


def build_plan(config, dataset, profiles, *, stage="pilot", remaining_gpu_seconds=None):
    """Freeze paired traces; ``pilot`` is intentionally one small group by default.

    ``profiles`` maps variant names to sealed measured profiles. Omitting
    flash_prefetch explicitly aliases flash_serial's *standalone* profile.
    Expanded plans contain the requested matrix but never execute it.
    """
    if stage not in ("pilot", "expanded"):
        raise ValueError("stage must be pilot or expanded")
    if remaining_gpu_seconds is not None:
        positive(remaining_gpu_seconds, "remaining_gpu_seconds", allow_zero=True)
    settings = config.get("optimization", {})
    variants = settings.get("variants", {})
    if set(variants) != set(VARIANTS):
        raise ValueError("optimization requires math_serial, flash_serial and flash_prefetch")
    expected = {"math_serial": ("math", 0), "flash_serial": ("flash", 0), "flash_prefetch": ("flash", 2)}
    for name, (attention, prefetch) in expected.items():
        if variants[name] != {"attention": attention, "text_prefetch": prefetch}:
            raise ValueError("unsupported ablation arguments for " + name)
    profiles = dict(profiles)
    aliases = {}
    if "flash_prefetch" not in profiles and "flash_serial" in profiles:
        profiles["flash_prefetch"] = profiles["flash_serial"]
        aliases["flash_prefetch"] = {"profile_alias_of": "flash_serial",
            "reason": "same isolated execution recipe; online prefetch does not change standalone costs"}
    reference_name = settings.get("load_reference_variant", "flash_serial")
    if reference_name not in VARIANTS:
        raise ValueError("invalid load_reference_variant")
    target_gpu = settings.get("target_gpu", "A100")
    for name in VARIANTS:
        if name not in profiles:
            raise ValueError("missing standalone profile: " + name)
        verify_seal(profiles[name])
        _check_identity(profiles[name], name, variants[name], target_gpu)
    seeds = config.get("trace_seeds", [])
    if len(seeds) != 3 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 3:
        raise ValueError("optimization requires three distinct integer trace seeds")
    policies = settings.get("policies", ["fcfs"])
    if not policies or len(set(policies)) != len(policies) or any(p not in ("fcfs", "edf", "srtf") for p in policies):
        raise ValueError("invalid optimization policies")
    scenarios = settings.get("scenarios", [])
    names = [item.get("name") for item in scenarios]
    if set(names) != {"under", "near", "over", "burst"} or len(names) != 4:
        raise ValueError("optimization requires distinct under, near, over and burst scenarios")
    pilot_names = settings.get("pilot_scenarios", ["near"])
    if not pilot_names or any(name not in names for name in pilot_names):
        raise ValueError("pilot_scenarios must select known scenarios")
    count = settings.get("pilot_count", 3) if stage == "pilot" else config.get("count", 6)
    if type(count) is not int or count < 2:
        raise ValueError("optimization trace count must be at least two")
    groups, deferred = [], []
    for seed in seeds:
        for scenario in scenarios:
            group_id = "%s-seed-%d" % (scenario["name"], seed)
            if stage == "pilot" and (seed != seeds[0] or scenario["name"] not in pilot_names):
                deferred.append({"group_id": group_id, "scenario": copy.deepcopy(scenario), "seed": seed,
                                 "reason": "outside_small_pilot; requires a new explicit plan and budget check"})
                continue
            shared = build_trace(config, dataset, profiles[reference_name], pattern=scenario["pattern"],
                                 load=scenario["load"], seed=seed, count=count)
            shared_hash = canonical_hash(shared["requests"])
            group = {"group_id": group_id, "scenario": copy.deepcopy(scenario), "seed": seed,
                     "shared_requests_sha256": shared_hash, "traces": {}, "variants": {}, "estimated_gpu_seconds": 0.}
            for name in VARIANTS:
                profile = profiles[name]
                trace = copy.deepcopy(shared)
                trace.update(execution_profile=copy.deepcopy(profile), execution_profile_sha256=profile["sha256"],
                    optimization={"group_id": group_id, "variant": name, "arguments": copy.deepcopy(variants[name]),
                        "shared_requests_sha256": shared_hash, "load_reference_variant": reference_name,
                        "target_gpu": target_gpu, "scenario": scenario["name"], "seed": seed,
                        **aliases.get(name, {})})
                trace = seal(trace)
                validate_trace(trace)
                execution_profile(trace)
                estimate = sum(profile["shapes"][r["shape_id"]]["service_ms"] for r in trace["requests"])
                estimate = (estimate * (len(policies) + 1) + max(r["arrival_ms"] for r in trace["requests"]) * len(policies)) / 1000. + 300.
                group["traces"][name] = trace
                group["variants"][name] = {"arguments": copy.deepcopy(variants[name]), "policies": list(policies),
                    "profile_sha256": profile["sha256"], "estimated_gpu_seconds": estimate, **aliases.get(name, {})}
                group["estimated_gpu_seconds"] += estimate
            groups.append(group)
    estimate = sum(group["estimated_gpu_seconds"] for group in groups)
    return seal({"schema_version": "ditserve.benchmark.optimization_plan.v1", "status": "completed",
        "stage": stage, "executes_gpu_work": False, "groups": groups, "deferred_groups": deferred,
        "config": copy.deepcopy(config), "dataset_sha256": dataset["sha256"],
        "pairing": "Identical request content, generation seeds, arrivals and deadlines; per-variant execution cost profiles.",
        "load_reference_variant": reference_name,
        "budget": {"estimated_gpu_seconds": estimate, "remaining_gpu_seconds": remaining_gpu_seconds,
                   "fits_estimate": None if remaining_gpu_seconds is None else estimate <= remaining_gpu_seconds,
                   "scope": "references, timed policies, idle arrival spans, 300 seconds per variant setup; prior profiling excluded",
                   "authoritative_limit": "existing GPU ledger; estimates do not reserve GPU time or guarantee completion"}})


def _stage_metrics(policy):
    rows = policy.get("requests", [])
    online = lambda row: [event for event in row.get("events", []) if event.get("outside_policy_horizon") is not True]
    events = [(row, event) for row in rows for event in online(row)]
    stages = {}
    for kind in ("prepare", "cpu_prompt_encode", "step", "decode", "snapshot", "restore", "artifact_enqueue"):
        values = [event["ms"] for _, event in events if event.get("kind") == kind and _finite(event.get("ms"))]
        stages[kind + "_ms"] = sum(values) if values else None
    # Serial preparation records CPU work on its prepare event; async encoders
    # may expose their own event. Avoid counting both for a single request.
    cpu, cpu_rows = [], 0
    for row in rows:
        separate = [e["ms"] for e in online(row) if e.get("kind") in ("cpu_prompt_encode", "text_encode") and _finite(e.get("ms"))]
        nested = [e["cpu_prompt_encode_ms"] for e in online(row) if e.get("kind") == "prepare" and _finite(e.get("cpu_prompt_encode_ms"))]
        values = separate or nested
        if values:
            cpu_rows += 1
            cpu.extend(values)
    cpu_ms = sum(cpu) if cpu else None
    horizon = policy.get("horizon_ms")
    active = sum(stages[key] for key in ("prepare_ms", "step_ms", "decode_ms") if stages[key] is not None)
    stages.update(cpu_prompt_encode_ms=cpu_ms, cpu_prompt_encode_request_samples=cpu_rows,
        cpu_work_fraction_of_horizon=cpu_ms / horizon if cpu_ms is not None and _finite(horizon) and horizon > 0 else None,
        gpu_step_fraction_of_serial_stages=stages["step_ms"] / active if stages["step_ms"] is not None and active > 0 else None,
        fraction_scope="CPU work can overlap GPU and idle; fractions are diagnostic, not mutually exclusive utilization.")
    complete_events = bool(rows) and all(
        sorted(e.get("step_index", -1) for e in online(r) if e.get("kind") == "step") ==
        list(range(r.get("generation", {}).get("steps", 0))) and
        any(e.get("kind") == "decode" for e in online(r)) for r in rows)
    observations = policy.get("observations", {})
    for metric, kind in (("snapshot_count", "snapshot"), ("restore_count", "restore")):
        value = observations.get(metric)
        stages[metric] = value if type(value) is int and value >= 0 else (sum(e.get("kind") == kind for _, e in events) if complete_events else None)
    max_ready = observations.get("max_gpu_ready")
    stages["max_gpu_ready"] = max_ready if type(max_ready) is int and max_ready >= 0 else None
    preemptions = observations.get("step_preemptions")
    if type(preemptions) is int and preemptions >= 0:
        stages["step_preemptions"] = preemptions
    else:
        timeline = [(event["start_ms"], row, event) for row, event in events
                    if event.get("kind") in ("prepare", "step", "decode", "restore") and _finite(event.get("start_ms"))]
        expected = sum(e.get("kind") in ("prepare", "step", "decode", "restore") for _, e in events)
        if not complete_events or len(timeline) != expected:
            stages["step_preemptions"] = None
        else:
            timeline.sort(key=lambda item: (item[0], item[1]["request_id"]))
            stages["step_preemptions"] = sum(
                before[2].get("kind") == "step" and before[1]["request_id"] != after[1]["request_id"] and
                type(before[2].get("step_index")) is int and
                before[2]["step_index"] + 1 < before[1].get("generation", {}).get("steps", 0)
                for before, after in zip(timeline, timeline[1:]))
    stages["preemption_scope"] = "different request dispatched after a solver step with denoising steps remaining"
    return stages


def _backend_quality(run, path, trace):
    if trace["optimization"]["arguments"]["attention"] == "math":
        return {"status": "not_applicable", "reason": "MATH is the attention quality reference"}
    gate = run.get("backend_quality_gate")
    if gate is None:
        return {"status": "unknown", "reason": "cross_attention_quality_gate_missing"}
    verify_seal(gate)
    result = {"status": "unknown", "gate_sha256": gate["sha256"],
              "scope": "held-out cross-attention LPIPS; not absolute quality or VBench"}
    if gate.get("schema_version") != "ditserve.benchmark.backend_quality.v1":
        return dict(result, reason="unsupported_backend_quality_schema")
    if gate.get("status") != "passed":
        return dict(result, status="failed", reason="backend_quality_gate_not_passed")
    epsilon = gate.get("epsilon")
    if not _finite(epsilon) or not 0 <= epsilon <= .05:
        return dict(result, reason="missing_or_relaxed_lpips_threshold")
    try:
        reference_path = Path(gate["reference_run_path"])
        if not reference_path.is_absolute():
            reference_path = path.parent / reference_path
        baseline = verify_seal(read_json(reference_path))
        if (baseline.get("sha256") != gate.get("reference_run_sha256") or
                file_hash(reference_path) != gate.get("reference_run_file_sha256")):
            return dict(result, reason="backend_quality_reference_run_changed")
        if (baseline.get("status") != "completed" or baseline.get("measured") is not True or
                baseline.get("backend") != "wan" or _attention(baseline.get("execution_identity", {})) != "math"):
            return dict(result, reason="backend_quality_reference_is_not_completed_measured_math")
        ids = {request["request_id"] for request in trace["requests"]}
        if set(gate.get("requests", {})) != ids:
            return dict(result, reason="backend_quality_request_coverage_mismatch")
        for rid in ids:
            row = gate["requests"][rid]
            candidate = run["references"][rid]
            reference = baseline["references"][rid]
            metrics = row.get("metrics", {})
            lpips = metrics.get("lpips_mean")
            if (row.get("status") != "passed" or metrics.get("status") != "ok" or
                    metrics.get("lpips_status") != "ok" or metrics.get("finite") is not True or
                    metrics.get("range_valid") is not True or not _finite(lpips) or not 0 <= lpips <= epsilon):
                return dict(result, status="failed", reason="backend_quality_request_not_passed")
            if (row.get("candidate_sha256") != candidate.get("output_sha256") or
                    row.get("reference_sha256") != reference.get("output_sha256") or
                    generation_identity(candidate.get("generation", {})) != generation_identity(reference.get("generation", {}))):
                return dict(result, reason="backend_quality_artifact_identity_mismatch")
            resolve_artifact_path(candidate["output_path"], candidate["output_sha256"], path.parent)
            resolve_artifact_path(reference["output_path"], reference["output_sha256"], reference_path.parent)
        return dict(result, status="passed", epsilon=epsilon, requests=len(ids), reference_run_sha256=baseline["sha256"])
    except (KeyError, LookupError, OSError, ValueError) as error:
        return dict(result, reason="backend_quality_evidence_unavailable: " + str(error))


def _comparison(left, right):
    result = {"baseline": left["variant"], "candidate": right["variant"], "policy": left["policy"],
              "group_id": left["group_id"], "status": "excluded", "reason": None,
              "mean_latency_improvement_fraction": None, "quality_goodput_improvement_fraction": None}
    if left["eligibility_reasons"] or right["eligibility_reasons"]:
        result["reason"] = "incomplete_or_unqualified_attempt"
    elif left["requests_sha256"] != right["requests_sha256"]:
        result["reason"] = "request_content_arrival_or_deadline_mismatch"
    elif left["physical_recipe_sha256"] != right["physical_recipe_sha256"]:
        result["reason"] = "physical_gpu_or_execution_recipe_mismatch"
    elif (left["variant"] == "math_serial" and
          right["backend_quality"]["reference_run_sha256"] != left["run_sha256"]):
        result["reason"] = "cross_attention_quality_uses_different_math_attempt"
    elif (left["variant"] == "flash_serial" and
          right["backend_quality"]["reference_run_sha256"] != left["backend_quality"]["reference_run_sha256"]):
        result["reason"] = "cross_attention_quality_reference_mismatch"
    else:
        lmean = left["summary"]["completed_latency_ms"]["mean"]
        rmean = right["summary"]["completed_latency_ms"]["mean"]
        lgood = left["summary"]["quality_qualified_goodput_per_s"]
        rgood = right["summary"]["quality_qualified_goodput_per_s"]
        result.update(status="included", paired_requests=left["summary"]["completed"],
            mean_latency_improvement_fraction=(lmean - rmean) / lmean if lmean and rmean is not None else None,
            quality_goodput_improvement_fraction=(rgood - lgood) / lgood if lgood and rgood is not None else None,
            quality_scope="within-variant reference parity plus same-input MATH-relative LPIPS <= 0.05; not absolute quality")
    return result


def _gates(attempts, pairs):
    eligible = [a for a in attempts if not a["eligibility_reasons"]]
    optimized = [a for a in eligible if a["variant"] == "flash_prefetch"]
    measured_cpu = [a["stages"]["cpu_work_fraction_of_horizon"] for a in optimized
                    if a["stages"]["cpu_work_fraction_of_horizon"] is not None]
    cpu_busy = bool(measured_cpu) and any(value >= .3 for value in measured_cpu)
    pressure = [a for a in optimized if (a["stages"]["snapshot_count"] or 0) > 0]
    batch_evidence = [a for a in optimized if a["stages"]["max_gpu_ready"] is not None and
        a["stages"]["max_gpu_ready"] >= 2 and a["stages"]["gpu_step_fraction_of_serial_stages"] is not None and
        a["stages"]["gpu_step_fraction_of_serial_stages"] >= .5]
    included = defaultdict(set)
    for pair in pairs:
        if pair["status"] == "included":
            included[(pair["group_id"], pair["policy"])].add((pair["baseline"], pair["candidate"]))
    complete_groups = {group for (group, _), values in included.items() if values == set(PAIRS)}
    flash = [a for a in attempts if a["variant"] != "math_serial"]
    quality_passed = bool(flash) and all(a["backend_quality"]["status"] == "passed" for a in flash)
    return {
        "paired_ablation": {"status": "measured" if complete_groups else "needs_measurement",
            "included_pair_count": sum(p["status"] == "included" for p in pairs),
            "complete_group_ids": sorted(complete_groups),
            "next_action": "expand to the remaining seeds and burst/load conditions only after budget review" if complete_groups else "complete one shared pilot group and its reference evaluations"},
        "cpu_pipeline": {"status": "profile_further" if cpu_busy else "measured_below_threshold" if measured_cpu else "unmeasured",
            "cpu_work_fraction_samples": measured_cpu, "threshold": .3,
            "next_action": "measure CPU encode concurrency and ready-queue starvation" if cpu_busy else "retain measured stage breakdown; do not infer zero CPU cost from missing events"},
        "batch_2": {"status": "profile_feasibility" if batch_evidence else "insufficient_evidence",
            "supporting_attempts": [a["run_path"] for a in batch_evidence],
            "next_action": "profile same-shape batch=2 memory, latency, throughput and per-request quality before implementation" if batch_evidence else "require observed multiple GPU-ready requests and GPU-step-dominated service first",
            "batch_2_capacity_proven": False},
        "state_policy": {"status": "profile_switch_costs" if pressure else "insufficient_evidence",
            "supporting_attempts": [a["run_path"] for a in pressure],
            "next_action": "compare transfer-aware simple scheduling against hot/keep using the observed pressure cases" if pressure else "require actual GPU state offload or capacity rejection; a synthetic mechanism is insufficient",
            "cold_quality_proven": False},
        "quality_scope": {"status": "cross_attention_gate_passed" if quality_passed else "separate_validation_required",
            "next_action": "cross-attention LPIPS passed only for the measured shape and seeds; absolute quality and VBench remain separate" if quality_passed else "validate Flash versus MATH on the target shape and generation seeds; own-reference parity does not establish perceptual equivalence or full VBench quality"}}


def summarize_runs(run_paths, output_path=None):
    """Summarize saved measured runs, rejecting incomplete/unbound comparisons.

    Missing timings remain null. Evaluation overlays and their persisted arrays
    are hash-verified through the same reader as the ordinary benchmark report.
    Invalid source artifacts raise; valid failed attempts remain explicit rows.
    """
    attempts, grouped = [], defaultdict(dict)
    for source in run_paths:
        path = Path(source).resolve()
        if path.is_dir():
            path /= "run.json"
        run = verify_seal(read_json(path))
        trace = verify_seal(read_json(path.parent / "trace.json"))
        validate_trace(trace)
        profile = execution_profile(trace)
        if run.get("trace_sha256") != trace["sha256"]:
            raise ValueError("run belongs to a different frozen trace")
        metadata = trace.get("optimization", {})
        if metadata.get("variant") not in VARIANTS:
            raise ValueError("run is not bound to an optimization plan")
        qualities, evaluation = _load_overlay(run, path)
        requests = trace["requests"]
        expected_ids = {r["request_id"] for r in requests}
        identity = run.get("execution_identity", {})
        backend_quality = _backend_quality(run, path, trace)
        for policy_name, policy in run.get("policies", {}).items():
            reasons = []
            if run.get("backend") != "wan" or run.get("measured") is not True or run.get("measurement_kind") != "measured_gpu":
                reasons.append("not_measured_gpu")
            if run.get("serving_performance_evidence") is not True or run.get("purpose") != "online_serving":
                reasons.append("not_online_serving_evidence")
            if run.get("status") != "completed" or policy.get("status") != "completed":
                reasons.append("incomplete_attempt")
            if identity != profile.get("execution_identity"):
                reasons.append("execution_profile_identity_mismatch")
            if run.get("execution_options") != metadata["arguments"]:
                reasons.append("execution_arguments_mismatch")
            if run.get("execution_profile_sha256", profile["sha256"]) != profile["sha256"]:
                reasons.append("execution_profile_hash_mismatch")
            if metadata["variant"] != "math_serial" and backend_quality["status"] != "passed":
                reasons.append("cross_attention_quality_not_passed")
            rows = policy.get("requests", [])
            ids = [r.get("request_id") for r in rows]
            if len(set(ids)) != len(ids) or set(ids) != expected_ids:
                reasons.append("incomplete_trace_coverage")
            frozen = {r["request_id"]: r for r in requests}
            if any(row.get("generation") != frozen.get(row.get("request_id")) or
                   row.get("arrival_ms") != frozen.get(row.get("request_id"), {}).get("arrival_ms") or
                   row.get("deadline_ms") != frozen.get(row.get("request_id"), {}).get("deadline_ms") for row in rows):
                reasons.append("request_content_arrival_or_deadline_mismatch")
            if not rows or any(row.get("status") != "completed" for row in rows):
                reasons.append("not_all_requests_completed")
            summary = summarize_requests(rows, policy.get("horizon_ms"), qualities.get(policy_name))
            if summary["quality_qualified"] != len(requests):
                reasons.append("reference_quality_not_all_passed")
            if summary["completed_missing_latency"] or summary["horizon_ms"] is None:
                reasons.append("missing_latency_or_horizon")
            attempt = {"run_path": str(path), "run_sha256": run["sha256"], "group_id": metadata["group_id"],
                "variant": metadata["variant"], "arguments": metadata["arguments"], "policy": policy_name,
                "scenario": metadata.get("scenario"), "seed": metadata.get("seed"),
                "requests_sha256": canonical_hash(requests), "execution_profile_sha256": profile["sha256"],
                "physical_recipe_sha256": canonical_hash({key: identity.get(key) for key in IDENTITY_FIELDS}),
                "eligibility_reasons": reasons, "summary": summary, "stages": _stage_metrics(policy),
                "backend_quality": backend_quality, "evaluation": evaluation}
            key = (metadata["group_id"], policy_name)
            if metadata["variant"] in grouped[key]:
                raise ValueError("multiple attempts for one group/variant/policy; choose explicit paired attempts")
            attempts.append(attempt)
            grouped[key][metadata["variant"]] = attempt
    pairs = []
    for (group_id, policy), arms in sorted(grouped.items()):
        for baseline, candidate in PAIRS:
            if baseline in arms and candidate in arms:
                pairs.append(_comparison(arms[baseline], arms[candidate]))
            else:
                pairs.append({"group_id": group_id, "policy": policy, "baseline": baseline, "candidate": candidate,
                              "status": "excluded", "reason": "missing_variant", "mean_latency_improvement_fraction": None,
                              "quality_goodput_improvement_fraction": None})
    result = seal({"schema_version": "ditserve.benchmark.optimization_summary.v1", "status": "completed",
        "attempts": attempts, "pairs": pairs, "gates": _gates(attempts, pairs),
        "scope": "Small measured serving ablations; missing timings stay unknown, incomplete or unqualified pairs cannot claim improvement."})
    if output_path is not None:
        if Path(output_path).exists():
            raise ValueError("refusing to overwrite optimization summary")
        atomic_json(output_path, result)
    return result
