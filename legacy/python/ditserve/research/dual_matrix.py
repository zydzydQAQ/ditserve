"""Finite two-device matrix and conservative, development-only decisions."""

import math
import re
import statistics

from .common import digest

ACTIONS = ("stay", "full_migrate", "cold_migrate", "certified_omit")


def seal(value):
    value = dict(value)
    value["sha256"] = digest(value)
    return value


def inputs(config, split="dev"):
    if split not in ("dev", "holdout"):
        raise ValueError("only explicit development and held-out splits are supported")
    seeds = config["dev_seeds" if split == "dev" else "holdout_seeds"]
    return [dict(prompt_id=p["id"], prompt=p["text"], seed=s, split=split)
            for p in config["prompts"] if p["split"] == split for s in seeds]


def known_quality(value, epsilon=.05, expected_frames=17):
    frames = value.get("lpips_per_frame", [])
    return (value.get("status") == "ok" and value.get("finite") is True
            and value.get("range_valid") is True and bool(frames)
            and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in frames)
            and type(value.get("frame_count")) is int
            and value["frame_count"] == len(frames) == expected_frames
            and type(value.get("lpips_mean")) in (int, float)
            and math.isclose(value["lpips_mean"], statistics.mean(frames), abs_tol=1e-6)
            and value["lpips_mean"] <= epsilon)


def bound_quality(value, candidate, reference):
    """Accept a metric only for its exact full outputs and generating identities."""
    if not known_quality(value) or not isinstance(candidate, dict) or not isinstance(reference, dict):
        return False
    for label, run in (("candidate", candidate), ("reference", reference)):
        output_hash, identity = run.get("output_sha256"), run.get("identity")
        if (not isinstance(identity, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(output_hash))
                or value.get(label + "_output_sha256") != output_hash
                or value.get(label + "_identity_sha256") != digest(identity)):
            return False
    return True


def _trajectory_matches(run, item, context, *, role, recipe, action, path, boundary=16):
    if not isinstance(run, dict):
        return False
    identity = run.get("identity", {})
    if not isinstance(identity, dict):
        return False
    expected = dict(status="completed", device_role=role, recipe_id=recipe, action=action,
                    action_step=boundary, prompt_id=item["prompt_id"], seed=item["seed"], split=item["split"])
    identity_fields = dict(context_sha256=context, input=item, recipe=recipe, action=action,
                           path=path, boundary=boundary)
    attention = identity.get("attention_path")
    return (all(run.get(k) == v for k, v in expected.items())
            and all(identity.get(k) == v for k, v in identity_fields.items())
            and isinstance(attention, list) and len(attention) == len(path)
            and all(backend == "flash" if device == "a100" else backend in ("math", "efficient")
                    for device, backend in zip(path, attention)))


def _cache_row_bound(row, item, context, device, boundary):
    candidate, reference = row.get("candidate"), row.get("reference")
    return (_trajectory_matches(candidate, item, context, role=device, recipe="t008", action="stay", path=[device], boundary=boundary)
            and _trajectory_matches(reference, item, context, role=device, recipe="off", action="stay", path=[device], boundary=boundary)
            and candidate["identity"]["attention_path"] == reference["identity"]["attention_path"]
            and bound_quality(row.get("quality", {}), candidate, reference))


def _migration_row_bound(row, item, context, selected, boundary):
    candidate, common, path = (row.get(k) for k in ("candidate", "common_reference", "path_reference"))
    return (_trajectory_matches(candidate, item, context, role="v100", recipe="t008", action=selected,
                                path=["v100", "a100"], boundary=boundary)
            and _trajectory_matches(common, item, context, role="a100", recipe="off", action="stay",
                                    path=["a100"], boundary=boundary)
            and _trajectory_matches(path, item, context, role="v100", recipe="off", action="full_migrate",
                                    path=["v100", "a100"], boundary=boundary)
            and candidate["identity"]["attention_path"] == path["identity"]["attention_path"]
            and bound_quality(row.get("path_reference_quality", {}), path, common)
            and bound_quality(row.get("canonical_quality", {}), candidate, common)
            and bound_quality(row.get("path_quality", {}), candidate, path))


def select_backend(pairs, *, shape, context_sha256):
    """Two full forward/reverse pairs, including quality, or retain MATH."""
    valid = len(pairs) == 2 and [p.get("order") for p in pairs] == [
        ["math", "efficient"], ["efficient", "math"]]
    gains = []
    for pair in pairs:
        valid &= (pair.get("shape") == shape and pair.get("context_sha256") == context_sha256
                  and pair.get("performance_valid") is True
                  and known_quality(pair.get("quality", {})))
        costs = pair.get("costs_ms", {})
        if not all(type(costs.get(k)) in (int, float) and math.isfinite(costs[k]) and costs[k] > 0
                   for k in ("math", "efficient")):
            valid = False
            continue
        gains.append(1 - costs["efficient"] / costs["math"])
    selected = "efficient" if valid and len(gains) == 2 and min(gains) >= .05 else "math"
    return seal(dict(schema_version="ditserve.dual.backend.v1", selected=selected, shape=shape,
                     context_sha256=context_sha256, gains=gains, locked=True,
                     reason="two_quality_qualified_opposite_order_gains" if selected == "efficient"
                     else "EFFICIENT_not_qualified_use_verified_MATH"))


def estimate_matrix(costs, *, safety=1.3, setup_gpu_seconds=600., evaluation_gpu_seconds=600.,
                    cleanup_gpu_seconds=120.):
    """Budget every predeclared sample; no sample-count reduction on failure.

    Costs are full measured generation seconds. Migration is conservatively
    charged on both resident devices, including target-idle source work.
    The queue bound includes real decoded requests, never artificial waiting.
    """
    if safety < 1.3 or set(costs) != {"a100", "v100"}:
        raise ValueError("both measured devices and >=1.3 safety factor are required")
    if any(not math.isfinite(v) or v <= 0 for v in costs.values()):
        raise ValueError("positive measured full-trajectory costs required")
    a, v = costs["a100"], costs["v100"]
    migration = v / 2 + a / 2
    raw = dict(calibration=16 * (a + v),
               # Screen + eight development + four held-out, both references.
               migration=2 * (8 * migration + 12 * (2 * migration + a)),
               # Idle/busy x two orders x <=3 actions, plus up to eight queued jobs.
               mechanism=2 * (12 * migration + 6 * 8 * a + 15 * a),
               capacity=3 * (1 + 2 + 4 + 8 + 4) * a)
    phases = {key: value * safety for key, value in raw.items()}
    phases["reserve"] = setup_gpu_seconds + evaluation_gpu_seconds + cleanup_gpu_seconds
    required = [stage for stage in phases if stage != "capacity"]
    return seal(dict(schema_version="ditserve.dual.cost_plan.v1", phases_gpu_seconds=phases,
                     required_phases=required, optional_phases=["capacity"],
                     required_gpu_seconds=sum(phases[stage] for stage in required),
                     planned_with_optional_gpu_seconds=sum(phases.values()), safety_factor=safety,
                     measured_costs_seconds=costs, quality_samples_reduced=False,
                     assumptions="bounded complete matrix; resident idle GPUs charged; no four-card work"))


def fits_plan(plan, remaining, phase_remaining=None):
    if plan["required_gpu_seconds"] > remaining:
        return False
    return phase_remaining is None or all(
        plan["phases_gpu_seconds"][stage] <= phase_remaining.get(stage, 0.)
        for stage in plan["required_phases"])


def cache_gate(rows, config, device, context_sha256):
    if device not in ("a100", "v100"):
        raise ValueError("cache qualification requires the declared device role")
    expected = {(x["prompt_id"], x["seed"]): x for x in inputs(config)}
    selected = [r for r in rows if r.get("device_role") == device and r.get("recipe_id") == "t008"]
    keys = [(r.get("prompt_id"), r.get("seed")) for r in selected]
    complete = bool(expected) and len(keys) == len(expected) and set(keys) == set(expected)
    valid = complete and all(r.get("split") == "dev" and r.get("status") == "completed"
        and r.get("performance_valid") is True
        and r.get("context_sha256") == context_sha256
        and _cache_row_bound(r, expected[(r["prompt_id"], r["seed"])], context_sha256,
                             device, config.get("transition_step", 16)) for r in selected)
    if valid:
        valid = len({tuple(r["candidate"]["identity"]["attention_path"]) for r in selected}) == 1
    return seal(dict(device=device, status="passed" if valid else "failed" if complete else "unverified",
                     eligible=valid, observed=len(keys), required=len(expected),
                     context_sha256=context_sha256))


def select_migration(rows, context_sha256):
    expected = {("kettle", 101), ("pendulum", 101)}
    candidates, missing, ineligible = [], [], []
    for action in ACTIONS[1:]:
        group = [r for r in rows if r.get("action") == action]
        keys = [(r.get("prompt_id"), r.get("seed")) for r in group]
        if len(keys) != 2 or set(keys) != expected:
            missing.append(action)
            continue
        if any(r.get("split") != "dev" or r.get("context_sha256") != context_sha256 for r in group):
            raise ValueError("screen uses wrong context or held-out data")
        if action == "certified_omit" and any(r.get("status") == "ineligible" for r in group):
            ineligible.append(action)
            continue
        if not all(r.get("status") == "completed" and r.get("performance_valid") is True
                   and type(r.get("elapsed_ms")) in (int, float)
                   and math.isfinite(r["elapsed_ms"]) and r["elapsed_ms"] > 0 for r in group):
            missing.append(action)
            continue
        candidates.append(dict(action=action, mean_cost_ms=statistics.mean(r["elapsed_ms"] for r in group)))
    candidates.sort(key=lambda x: (x["mean_cost_ms"], x["action"]))
    return seal(dict(status="selected" if candidates and not missing else "unverified",
                     selected=candidates[0]["action"] if candidates and not missing else None,
                     candidates=candidates, missing=missing, ineligible=ineligible,
                     context_sha256=context_sha256, selection_split="dev", holdout_used=False))


def migration_quality_gate(rows, config, split, selected, context_sha256):
    if selected not in ACTIONS[1:]:
        raise ValueError("migration quality requires a declared migration action")
    expected = {(x["prompt_id"], x["seed"]): x for x in inputs(config, split)}
    keys = [(r.get("prompt_id"), r.get("seed")) for r in rows]
    complete = bool(expected) and len(keys) == len(expected) and set(keys) == set(expected)
    passed = complete and all(r.get("action") == selected and r.get("split") == split
        and r.get("context_sha256") == context_sha256 and r.get("status") == "completed"
        and r.get("performance_valid") is True
        and _migration_row_bound(r, expected[(r["prompt_id"], r["seed"])], context_sha256,
                                 selected, config.get("transition_step", 16))
        for r in rows)
    if passed:
        passed = len({tuple(r["candidate"]["identity"]["attention_path"]) for r in rows}) == 1
    return seal(dict(status="passed" if passed else "failed" if complete else "unverified",
                     eligible=passed, split=split, selected=selected, context_sha256=context_sha256,
                     required=len(expected), observed=len(keys), error_composition="none"))


def choose_queue(observations, break_even_wait_ms):
    """First measured real queue reaching the threshold, never interpolate."""
    if not math.isfinite(break_even_wait_ms) or break_even_wait_ms <= 0:
        return None
    for length in (1, 2, 4, 8):
        matches = [r for r in observations if r.get("queue_length") == length
                   and r.get("all_requests_decoded") is True and r.get("performance_valid") is True]
        if len(matches) != 1:
            return None
        wait = matches[0].get("wait_ms")
        if type(wait) not in (int, float) or not math.isfinite(wait) or wait < 0:
            return None
        if wait >= 1.25 * break_even_wait_ms:
            return length
    return None


def mechanism_gate(pairs):
    """Two reversed full-makespan pairs are engineering evidence, not a p-value."""
    if (len(pairs) != 2 or pairs[0].get("order") != list(reversed(pairs[1].get("order", [])))
            or pairs[0].get("order") not in (["stay", "candidate"], ["candidate", "stay"])):
        return dict(status="unverified", supported=False)
    gains = []
    for p in pairs:
        costs = p.get("costs_ms", {})
        if (p.get("performance_valid") is not True or p.get("quality_passed") is not True
                or p.get("all_requests_decoded") is not True
                or any(type(costs.get(name)) not in (int, float) or not math.isfinite(costs[name])
                       or costs[name] <= 0 for name in ("stay", "candidate"))):
            return dict(status="unverified", supported=False)
        gains.append(1 - costs["candidate"] / costs["stay"])
    return dict(status="supported" if min(gains) >= .05 else "not_observed",
                supported=min(gains) >= .05, gains=gains,
                scope="engineering_gate_not_statistical_significance_or_cache_joint_innovation")
