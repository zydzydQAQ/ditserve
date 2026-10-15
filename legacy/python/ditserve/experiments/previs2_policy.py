"""Pure-CPU, evidence-bound policies for the second PreViS experiment.

Costs are instrumented phase estimates, LPIPS bounds are development maxima,
and capacity estimates require a post-release check before admitting work.
This module never imports a model, reads files, or queries a GPU.
"""

import hashlib
import json
import math
import statistics
from collections import defaultdict


MODES = ("keep", "hot", "cold")
SHAPE_KEYS = ("height", "width", "frames", "steps")
FULL = {"cache": False, "threshold": 0.0, "max_continuous": 1}
MIB = 1024 ** 2


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bytes(value):
    return type(value) is int and value >= 0


def _safe(value):
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _config(value):
    if not isinstance(value, dict) or type(value.get("cache")) is not bool:
        return None
    threshold, maximum = value.get("threshold"), value.get("max_continuous")
    if not _finite(threshold) or threshold < 0 or type(maximum) is not int or maximum != 1:
        return None
    if value["cache"] != (threshold > 0):
        return None
    return dict(cache=value["cache"], threshold=float(threshold), max_continuous=1)


def _shape(value):
    return (isinstance(value, dict) and all(type(value.get(k)) is int and value[k] > 0 for k in SHAPE_KEYS))


def _loss(row):
    q = row.get("quality", {})
    if (q.get("status") == "ok" and q.get("finite") is True and q.get("range_valid") is True
            and _finite(q.get("lpips_mean")) and q["lpips_mean"] >= 0):
        return q["lpips_mean"]
    return None


def _phases(row, steps):
    events = row.get("events")
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        return None
    kinds = {kind: [e for e in events if e.get("kind") == kind] for kind in ("prepare", "step", "decode")}
    if (len(kinds["prepare"]) != 1 or len(kinds["decode"]) != 1 or len(kinds["step"]) != steps
            or any(type(e.get("step_index")) is not int for e in kinds["step"])
            or [e["step_index"] for e in kinds["step"]] != list(range(steps))
            or any(not _finite(e.get("ms")) or e["ms"] < 0 for values in kinds.values() for e in values)):
        return None
    ordered = [e.get("kind") for e in events if e.get("kind") in kinds]
    if ordered != ["prepare"] + ["step"] * steps + ["decode"]:
        return None
    return dict(prepare=kinds["prepare"][0]["ms"], steps=[e["ms"] for e in kinds["step"]],
                decode=kinds["decode"][0]["ms"])


def _median_phases(values, common_prepare=None):
    steps = [statistics.median(row["steps"][i] for row in values) for i in range(len(values[0]["steps"]))]
    prepare = common_prepare if common_prepare is not None else statistics.median(row["prepare"] for row in values)
    decode = statistics.median(row["decode"] for row in values)
    return dict(prepare=prepare, steps=sum(steps), decode=decode, total=prepare + sum(steps) + decode,
                per_step=steps)


def _curve(phases, start=0):
    return {str(i): sum(phases["per_step"][i:]) + phases["decode"]
            for i in range(start, len(phases["per_step"]) + 1)}


def _evidence(rows):
    text = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return dict(sha256=hashlib.sha256(text.encode()).hexdigest(), samples=len(rows),
                case_ids=[row.get("case_id") for row in rows],
                prompt_seed_pairs=sorted([[row["prompt_id"], row["seed"]] for row in rows]))


def _matches(row, device, shape, model_config):
    return (isinstance(row, dict) and row.get("status") == "completed" and row.get("split") == "dev"
            and row.get("device") == device and row.get("shape") == shape
            and (model_config is None or row.get("model_config", row.get("config")) == model_config))


def _coverage(rows, expected):
    pairs = [(r.get("prompt_id"), r.get("seed")) for r in rows]
    return (all(isinstance(p, str) and type(s) is int for p, s in pairs)
            and len(pairs) == len(expected) and set(pairs) == expected and len(set(pairs)) == len(pairs))


def _transfer(row, kind):
    value = row.get(kind)
    if not isinstance(value, dict):
        values = [e for e in row.get("events", []) if isinstance(e, dict) and e.get("kind") == kind]
        if not values:
            values = [e for e in row.get("switches", []) if isinstance(e, dict) and e.get("kind") == kind]
        value = values[0] if len(values) == 1 else {}
    if value.get("mode", row.get("mode")) != row.get("mode"):
        return None
    return value.get("ms")


def _recovery_path(row, step, mode, prefix, target):
    events = row.get("events", [])
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        return False
    snapshots = [i for i, e in enumerate(events) if e.get("kind") == "snapshot"]
    restores = [i for i, e in enumerate(events) if e.get("kind") == "restore"]
    if len(snapshots) != 1 or len(restores) != 1 or snapshots[0] >= restores[0]:
        return False
    before, after = snapshots[0], restores[0]
    if (any(events[i].get("mode") != mode for i in (before, after))
            or sum(e.get("kind") == "step" for e in events[:before]) != step
            or any(e.get("kind") == "step" for e in events[before:after])):
        return False
    for e in events:
        if e.get("kind") == "step":
            config = prefix if e.get("step_index", -1) < step else target
            if any(key in e and e[key] != config[key] for key in ("threshold", "max_continuous")):
                return False
    return True


def select_initial_policy(records, *, device, shape, dev_prompt_ids, seeds=(0, 1), epsilon=.05, model_config=None):
    """Pick the fastest qualified m1 configuration, sharing uncached prepare cost.

    Eight distinct development prompt/seed pairs are required for every
    configuration. Uncached quality_reference rows need no self-LPIPS metric.
    Unknown coverage permits full compute, but never fabricates a cost.
    """
    if (not _shape(shape) or len(set(dev_prompt_ids)) != 4 or len(dev_prompt_ids) != 4
            or len(seeds) != 2 or len(set(seeds)) != 2 or any(type(s) is not int or s < 0 for s in seeds)
            or not _finite(epsilon) or epsilon < 0):
        raise ValueError("exact shape, four prompts, two seeds and finite epsilon required")
    expected = {(p, s) for p in dev_prompt_ids for s in seeds}
    grouped, excluded = defaultdict(list), []
    for row in records:
        if not _matches(row, device, shape, model_config) or row.get("type") not in ("quality", "quality_reference"):
            continue
        if row.get("quality_scope", "whole_request_fixed_policy") != "whole_request_fixed_policy":
            continue
        config = _config(row.get("cache_config"))
        if config is None:
            excluded.append(dict(case_id=row.get("case_id"), reason="invalid_m1_cache_config")); continue
        grouped[tuple(config.values())].append(row)
    candidates = []
    for key, rows in grouped.items():
        config = dict(zip(("cache", "threshold", "max_continuous"), key))
        reason = None
        phases = [_phases(r, shape["steps"]) for r in rows]
        losses = [0.0 if not config["cache"] and r.get("type") == "quality_reference" and "quality" not in r
                  else _loss(r) for r in rows]
        if not _coverage(rows, expected): reason = "incomplete_or_duplicate_development_grid"
        elif any(p is None for p in phases): reason = "incomplete_or_invalid_phase_events"
        elif any(v is None for v in losses): reason = "unknown_quality"
        elif max(losses) > epsilon: reason = "development_quality_exceeds_epsilon"
        if reason:
            excluded.append(dict(config=config, reason=reason)); continue
        candidates.append(dict(config=config, predicted_trajectory_lpips=max(losses),
                               _phases=phases, evidence=_evidence(rows)))
    reference = next((a for a in candidates if a["config"] == FULL), None)
    if reference is None:
        return dict(status="fallback_unknown_cost", config=dict(FULL), predicted_trajectory_lpips=0.0,
                    phase_cost_ms=None, remaining_by_completed_step_ms=None, evidence=None,
                    excluded=excluded, candidates=[], reason="complete_uncached_phase_reference_required",
                    quality_is_guarantee=False)
    prepare = statistics.median(p["prepare"] for p in reference["_phases"])
    for action in candidates:
        phases = _median_phases(action.pop("_phases"), prepare)
        action.update(phase_cost_ms=phases, remaining_by_completed_step_ms=_curve(phases))
    winner = min(candidates, key=lambda a: (a["phase_cost_ms"]["total"], a["predicted_trajectory_lpips"],
                                           a["config"]["cache"], a["config"]["threshold"]))
    return dict(winner, status="selected", candidates=candidates, excluded=excluded,
                quality_is_guarantee=False, quality_scope="whole_trajectory_development_max",
                timing_scope="common_uncached_prepare_plus_configuration_step_and_decode_medians")


def capacity_feasibility(*, total_bytes, driver_free_bytes, allocated_bytes, reserved_bytes,
                         victim_storage_bytes, next_peak_bytes, logical_budget_bytes=None,
                         logical_live_bytes=None, next_state_bytes=None):
    """Conservative allocator-headroom estimate; never equate freeing tensors with driver return.

    next_peak_bytes is the measured additional allocation peak, not a total
    process peak. Artificial logical capacity is an explicit separate gate.
    Fragmentation and external allocations require rechecking after release.
    """
    values = (total_bytes, driver_free_bytes, allocated_bytes, reserved_bytes, victim_storage_bytes, next_peak_bytes)
    valid = (all(_bytes(v) for v in values) and total_bytes > 0 and allocated_bytes <= reserved_bytes
             and victim_storage_bytes <= allocated_bytes and driver_free_bytes + reserved_bytes <= total_bytes)
    margin = max(512 * MIB, math.ceil(.05 * total_bytes)) if _bytes(total_bytes) and total_bytes else None
    artificial = logical_budget_bytes is not None
    logical_valid = (not artificial or all(_bytes(v) for v in (logical_budget_bytes, logical_live_bytes, next_state_bytes))
                     and logical_live_bytes >= victim_storage_bytes)
    modes = {}
    for mode in MODES:
        released = 0 if mode == "keep" else victim_storage_bytes
        available = driver_free_bytes + reserved_bytes - allocated_bytes + released - margin if valid else None
        physical = valid and available >= next_peak_bytes
        logical_required = logical_live_bytes - released + next_state_bytes if artificial and logical_valid else None
        logical = logical_valid and (not artificial or logical_required <= logical_budget_bytes)
        reason = ("unknown_or_inconsistent_memory_evidence" if not valid else
                  "unknown_or_inconsistent_logical_budget_evidence" if not logical_valid else
                  "physical_headroom_insufficient" if not physical else
                  "explicit_logical_state_budget_exceeded" if not logical else "estimated_feasible_requires_runtime_check")
        modes[mode] = dict(feasible=bool(physical and logical), reason=reason,
                           allocatable_after_margin_bytes=available, logical_required_bytes=logical_required,
                           expected_allocator_reusable_release_bytes=released if valid else None,
                           expected_driver_free_after_release_bytes=None, requires_post_release_check=mode != "keep",
                           requires_runtime_capacity_check=True)
    return _safe(dict(modes=modes, margin_bytes=margin, memory_evidence_known=valid,
                diagnostics=dict(artificial_logical_pressure=artificial, logical_budget_bytes=logical_budget_bytes,
                    logical_live_bytes=logical_live_bytes, next_state_bytes=next_state_bytes,
                    driver_free_bytes=driver_free_bytes, allocated_bytes=allocated_bytes, reserved_bytes=reserved_bytes,
                    victim_storage_bytes=victim_storage_bytes, next_peak_bytes=next_peak_bytes,
                    next_peak_scope="additional_peak_allocated_minus_pre_request_allocated_including_no_resident_weights_again",
                    released_allocated_is_not_driver_free=True, fragmentation_or_external_growth_not_guaranteed=True)))


def build_recovery_table(records, *, device, shape, prefix_config, dev_prompt_ids,
                         first_seed=0, epsilon=.05, model_config=None):
    """Calibrate full prefix/suffix trajectories; the calibration seed is not a runtime seed restriction."""
    prefix = _config(prefix_config)
    if (not _shape(shape) or prefix is None or len(dev_prompt_ids) != 4 or len(set(dev_prompt_ids)) != 4
            or type(first_seed) is not int or first_seed < 0 or not _finite(epsilon) or epsilon < 0):
        raise ValueError("valid m1 prefix, shape and four development prompts required")
    expected = {(p, first_seed) for p in dev_prompt_ids}
    grouped, excluded, actions = defaultdict(list), [], []
    for r in records:
        if (not _matches(r, device, shape, model_config) or r.get("type") != "quality"
                or r.get("quality_scope") != "resume_policy" or _config(r.get("prefix_config")) != prefix):
            continue
        key = (r.get("pause_step"), r.get("mode"), r.get("suffix"))
        if type(key[0]) is not int or key[0] not in (8, 16, 24) or not 0 < key[0] < shape["steps"] or key[1] not in MODES or key[2] not in ("continue", "full_compute"):
            excluded.append(dict(case_id=r.get("case_id"), reason="invalid_recovery_action_identity")); continue
        grouped[key].append(r)
    for (step, mode, suffix), rows in grouped.items():
        reason = None; losses = [_loss(r) for r in rows]; phases = [_phases(r, shape["steps"]) for r in rows]
        target = prefix if suffix == "continue" else FULL
        transfers = [(_transfer(r, "snapshot"), _transfer(r, "restore")) for r in rows]
        if not _coverage(rows, expected): reason = "incomplete_or_duplicate_development_grid"
        elif any(_config(r.get("cache_config")) != target for r in rows): reason = "suffix_configuration_mismatch"
        elif any(v is None for v in losses): reason = "unknown_whole_trajectory_quality"
        elif max(losses) > epsilon: reason = "whole_trajectory_quality_exceeds_epsilon"
        elif any(p is None for p in phases): reason = "incomplete_or_invalid_phase_events"
        elif any(not _recovery_path(r, step, mode, prefix, target) for r in rows): reason = "observed_pause_mode_or_step_policy_mismatch"
        elif any(not _finite(v) or v < 0 for pair in transfers for v in pair): reason = "unknown_transfer_cost"
        if reason:
            excluded.append(dict(step_index=step, mode=mode, suffix=suffix, reason=reason)); continue
        costs = _median_phases(phases); curve = _curve(costs, step)
        snapshot = statistics.median(p[0] for p in transfers); restore = statistics.median(p[1] for p in transfers)
        actions.append(dict(step_index=step, mode=mode, suffix=suffix, config=dict(target),
            predicted_trajectory_lpips=max(losses), remaining_ms=curve[str(step)],
            remaining_by_completed_step_ms=curve, snapshot_ms=snapshot, restore_ms=restore,
            estimated_total_ms=snapshot + restore + curve[str(step)], evidence=_evidence(rows),
            consumes_recovery_choice=mode == "cold" or suffix != "continue",
            quality_scope="whole_prefix_plus_suffix_development_max", quality_is_additive=False))
    return dict(status="ready" if actions else "unavailable", device=device, shape=dict(shape),
                model_config=model_config, prefix_config=prefix, actions=actions, excluded=excluded,
                calibration_seed=first_seed, runtime_seed_restriction=None, quality_is_guarantee=False)


def enumerate_recovery_actions(request, table, capacity, epsilon=.05):
    """Enumerate capacity-safe choices; uncovered keep/hot continuation preserves prior loss.

    Caller must supply measured fallback state_cost_ms and the current-policy
    remaining_ms. Unknown costs remain unknown rather than becoming zero.
    """
    if not _finite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    actions, excluded = [], []
    current = _config(request.get("selected_cache_config")); initial = _config(request.get("initial_cache_config"))
    loss = request.get("predicted_trajectory_lpips"); loss = loss if _finite(loss) and loss >= 0 else None
    remaining = request.get("remaining_ms"); remaining = remaining if _finite(remaining) and remaining >= 0 else None
    for mode in MODES:
        if capacity.get("modes", {}).get(mode, {}).get("feasible") is not True:
            excluded.append(dict(mode=mode, reason="capacity_not_proven_feasible")); continue
        if mode != "cold" and current is not None:
            transfer = request.get("state_cost_ms", {}).get(mode, {})
            snapshot, restore = transfer.get("snapshot_ms"), transfer.get("restore_ms")
            snapshot = snapshot if _finite(snapshot) and snapshot >= 0 else None
            restore = restore if _finite(restore) and restore >= 0 else None
            known = all(_finite(x) and x >= 0 for x in (snapshot, restore, remaining))
            actions.append(dict(mode=mode, suffix="continue", config=dict(current),
                predicted_trajectory_lpips=loss, remaining_ms=remaining, snapshot_ms=snapshot, restore_ms=restore,
                estimated_total_ms=snapshot + restore + remaining if known else None,
                evidence=None, calibration="exact_state_continuation", consumes_recovery_choice=False,
                quality_scope="unchanged_prior_whole_trajectory_prediction", quality_is_additive=False))
    matches = (table.get("status") == "ready" and request.get("device") == table.get("device")
        and request.get("shape") == table.get("shape") and current == initial == table.get("prefix_config")
        and type(request.get("policy_change_count")) is int and request["policy_change_count"] == 0
        and type(request.get("lossy_choices_used")) is int and request["lossy_choices_used"] == 0
        and request.get("recovery_action_used", False) is False
        and (table.get("model_config") is None or request.get("model_config") == table["model_config"]))
    if not matches:
        excluded.append(dict(reason="uncovered_or_already_changed_prefix_continue_only"))
    else:
        for action in table.get("actions", []):
            reason = None
            if action["step_index"] != request.get("step_index"): reason = "uncovered_progress"
            elif capacity.get("modes", {}).get(action["mode"], {}).get("feasible") is not True: reason = "capacity_not_proven_feasible"
            elif action["predicted_trajectory_lpips"] > epsilon: reason = "whole_trajectory_quality_exceeds_epsilon"
            if reason:
                excluded.append(dict(mode=action["mode"], suffix=action["suffix"], reason=reason)); continue
            candidate = dict(action, calibration="complete_development_trajectory")
            if action["mode"] in ("keep", "hot") and action["suffix"] == "continue":
                # Exact continuation cannot erase a larger previously known
                # whole-trajectory maximum, nor turn unknown prior loss into a
                # guarantee using a smaller recovery calibration cohort.
                candidate["predicted_trajectory_lpips"] = max(loss, action["predicted_trajectory_lpips"]) if loss is not None else None
                candidate["quality_scope"] = "unchanged_trajectory_max_of_prior_and_matching_development_evidence"
                # A measured matching transfer/suffix profile supersedes its
                # fallback estimate; do not pick the cheaper of two models for
                # the very same state/configuration action.
                actions = [a for a in actions if not (a["mode"] == action["mode"] and a["suffix"] == "continue")]
            actions.append(candidate)
    return dict(actions=actions, excluded=excluded, quality_is_guarantee=False)


def select_fixed_state(table):
    """Pick a fixed evicted mode on shared complete development action coverage."""
    by_mode = {m: {a["step_index"]: a for a in table.get("actions", [])
                   if a["mode"] == m and a["suffix"] == "continue"} for m in ("hot", "cold")}
    shared = sorted(set(by_mode["hot"]) & set(by_mode["cold"]))
    if not shared:
        return dict(mode="hot", status="exact_hot_fallback", reason="no_shared_development_comparison")
    estimates = {m: statistics.median(by_mode[m][s]["estimated_total_ms"] for s in shared) for m in by_mode}
    mode = min(estimates, key=lambda m: (estimates[m], m != "hot"))
    return dict(mode=mode, status="development_selected", compared_steps=shared, estimated_total_ms=estimates)


def choose_action(policy, actions, *, fixed_state="hot"):
    """All policies use the same external EDF order; this chooses only victim state/suffix."""
    if policy not in ("edf", "fixed_state_qc", "joint_state_qc") or fixed_state not in ("hot", "cold"):
        raise ValueError("unknown policy or fixed eviction mode")
    candidates = list(actions); fallback_reason = None
    if policy == "edf":
        candidates = [a for a in candidates if a["suffix"] == "continue" and a["mode"] in ("keep", "hot")]
    if policy != "joint_state_qc":
        mode = "keep" if any(a["mode"] == "keep" for a in candidates) else "hot" if policy == "edf" else fixed_state
        selected_mode = [a for a in candidates if a["mode"] == mode]
        if not selected_mode and policy == "fixed_state_qc" and mode == "cold":
            selected_mode = [a for a in candidates if a["mode"] == "hot" and a["suffix"] == "continue"
                             and a.get("calibration") == "exact_state_continuation"]
            if selected_mode:
                fallback_reason = "fixed_cold_uncovered_fallback_exact_hot_continuation"
        candidates = selected_mode
    known = [a for a in candidates if _finite(a.get("estimated_total_ms")) and a["estimated_total_ms"] >= 0]
    if not known:
        fallback = next((a for mode in ("keep", "hot") for a in candidates
                         if a["mode"] == mode and a["suffix"] == "continue"
                         and a.get("calibration") == "exact_state_continuation"), None)
        return dict(status="fallback_unknown_cost" if fallback else "no_feasible_action", action=fallback,
                    fallback_reason=fallback_reason)
    winner = min(known, key=lambda a: (a["estimated_total_ms"],
        a["predicted_trajectory_lpips"] is None,
        a["predicted_trajectory_lpips"] if a["predicted_trajectory_lpips"] is not None else 0,
        a["suffix"] != "continue", MODES.index(a["mode"])))
    return dict(status="selected", action=winner, fallback_reason=fallback_reason,
                selection_scope="shared_EDF_resource_and_remaining_compute_cost")
