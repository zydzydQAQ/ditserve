"""Pure evidence contracts for bounded DiT research experiments.

These helpers never execute a model and never turn a plan or a CPU fixture into
GPU evidence. Quality belongs to one complete trajectory, not to a sum of
prefix/suffix losses. Empirical LPIPS qualification is not a quality guarantee.
"""

import copy
import math
import statistics
from collections import defaultdict

from ..benchmark.common import canonical_hash, seal, verify_seal


IDENTITY_FIELDS = ("model_id", "revision", "backend", "precision", "input", "spec",
                   "prefix", "action", "suffix", "layout_history", "protocol_sha256", "source_sha256")
_REFERENCE_SHARED = ("model_id", "revision", "backend", "precision", "input", "spec", "protocol_sha256", "source_sha256")
_SCHEMA = "ditserve.research.trajectory_evidence.v1"


def _number(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and (value > 0 if positive else value >= 0))


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _recipe(value):
    if not isinstance(value, dict) or type(value.get("cache")) is not bool:
        raise ValueError("cache recipe must explicitly contain cache, threshold and max_continuous")
    if not _number(value.get("threshold")) or type(value.get("max_continuous")) is not int or value["max_continuous"] != 1:
        raise ValueError("research recipes require a finite threshold and max_continuous=1")
    if not value["cache"] and value["threshold"] != 0:
        raise ValueError("cache-off threshold must be zero")


def trajectory_identity(identity):
    """Validate and copy the entire identity; extra fields also affect its hash.

    ``input`` contains prompt, negative_prompt and seed; ``spec`` contains
    height, width, frames, steps and guidance_scale. ``action`` explicitly gives
    kind and at_step. Layout history is a nonempty list of layout decisions.
    """
    if not isinstance(identity, dict) or any(k not in identity for k in IDENTITY_FIELDS):
        raise ValueError("incomplete full-trajectory identity")
    for key in ("model_id", "revision"):
        if not isinstance(identity[key], str) or not identity[key]:
            raise ValueError("missing " + key)
    if not isinstance(identity["backend"], (str, dict)) or not identity["backend"]:
        raise ValueError("backend identity is required")
    if not isinstance(identity["precision"], dict) or not identity["precision"]:
        raise ValueError("explicit precision identity is required")
    inputs, spec, action = identity["input"], identity["spec"], identity["action"]
    if (not isinstance(inputs, dict) or not isinstance(inputs.get("prompt"), str) or
            not isinstance(inputs.get("negative_prompt"), str) or type(inputs.get("seed")) is not int):
        raise ValueError("input requires exact prompt, negative_prompt and integer seed")
    if not isinstance(spec, dict) or any(type(spec.get(k)) is not int or spec[k] <= 0
                                        for k in ("height", "width", "frames", "steps")):
        raise ValueError("spec requires positive integer height, width, frames and steps")
    if not _number(spec.get("guidance_scale")):
        raise ValueError("spec requires guidance_scale")
    _recipe(identity["prefix"])
    _recipe(identity["suffix"])
    if (not isinstance(action, dict) or action.get("kind") not in ("none", "keep", "hot", "exact_refresh", "cold") or
            type(action.get("at_step")) is not int or not 0 <= action["at_step"] < spec["steps"]):
        raise ValueError("action requires a supported kind and completed-step boundary")
    history = identity["layout_history"]
    if not isinstance(history, list) or not history or any(not isinstance(item, dict) or not item for item in history):
        raise ValueError("complete nonempty layout history is required")
    if not all(_hash(identity[key]) for key in ("protocol_sha256", "source_sha256")):
        raise ValueError("protocol and source SHA256 identities are required")
    canonical_hash(identity)  # Reject non-JSON and nonfinite metadata.
    return copy.deepcopy(identity)


def trajectory_fingerprint(identity):
    return canonical_hash(trajectory_identity(identity))


def _percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _quality_reasons(metrics, frames):
    if not isinstance(metrics, dict):
        return ["missing_quality_metrics"]
    reasons = []
    if metrics.get("status") != "ok" or metrics.get("lpips_status") != "ok":
        reasons.append("unknown_quality")
    if metrics.get("finite") is not True or metrics.get("range_valid") is not True:
        reasons.append("invalid_output_values")
    values = metrics.get("lpips_per_frame")
    if not isinstance(values, list) or len(values) != frames or not all(_number(v) for v in values):
        reasons.append("incomplete_full_trajectory_lpips")
    else:
        for key, expected in (("lpips_mean", statistics.mean(values)), ("lpips_p95", _percentile(values, .95))):
            actual = metrics.get(key)
            if not _number(actual) or not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9):
                reasons.append("inconsistent_" + key)
    if metrics.get("frame_count") != frames:
        reasons.append("frame_count_mismatch")
    if frames > 1 and not _number(metrics.get("temporal_delta_rmse")):
        reasons.append("missing_temporal_error")
    if metrics.get("aggregation_scope", "full_trajectory") != "full_trajectory":
        reasons.append("prefix_suffix_quality_cannot_be_added")
    return reasons


def trajectory_evidence(identity, metrics, *, correctness, completed_steps,
                        reference_output_sha256=None, candidate_output_sha256=None,
                        reference_identity=None, measurement_kind="measured_gpu"):
    """Seal one measured final video and its full-trajectory reference metrics.

    Metrics use ``ditserve.gpu.quality.evaluate_quality`` field names. Missing
    reference identity, arrays, complete steps or metrics remain insufficient.
    Correctness is a status string or a record containing ``status``.
    """
    identity = trajectory_identity(identity)
    reference = None if reference_identity is None else trajectory_identity(reference_identity)
    payload = dict(schema_version=_SCHEMA, identity=identity,
        trajectory_sha256=canonical_hash(identity), reference_identity=reference,
        reference_trajectory_sha256=None if reference is None else canonical_hash(reference),
        reference_output_sha256=reference_output_sha256, candidate_output_sha256=candidate_output_sha256,
        measurement_kind=measurement_kind, completed_steps=completed_steps,
        correctness=copy.deepcopy(correctness), metrics=copy.deepcopy(metrics),
        quality_scope="full_trajectory_vs_cache_off_reference", additive_quality_budget=False)
    # Status is recomputed on reading, never trusted as qualification on its own.
    return seal(payload)


def qualify_trajectory(evidence, identity, *, epsilon=.05):
    """Return qualified/failed/insufficient for this exact identity only."""
    if not _number(epsilon):
        raise ValueError("epsilon must be finite and nonnegative")
    expected = trajectory_identity(identity)
    reasons = []
    try:
        verify_seal(evidence)
    except (ValueError, TypeError):
        return dict(status="insufficient", qualified=False, reasons=["missing_or_changed_evidence"], epsilon=epsilon)
    if evidence.get("schema_version") != _SCHEMA:
        reasons.append("unsupported_evidence_schema")
    if evidence.get("identity") != expected or evidence.get("trajectory_sha256") != canonical_hash(expected):
        reasons.append("trajectory_identity_mismatch")
    if evidence.get("measurement_kind") != "measured_gpu":
        reasons.append("not_measured_gpu")
    correctness = evidence.get("correctness")
    correctness = correctness.get("status") if isinstance(correctness, dict) else correctness
    if correctness != "passed":
        reasons.append("correctness_not_passed")
    if type(evidence.get("completed_steps")) is not int or evidence["completed_steps"] != expected["spec"]["steps"]:
        reasons.append("incomplete_solver_trajectory")
    if not all(_hash(evidence.get(key)) for key in ("reference_output_sha256", "candidate_output_sha256")):
        reasons.append("missing_output_hashes")
    reference = evidence.get("reference_identity")
    try:
        reference = trajectory_identity(reference)
    except (ValueError, TypeError):
        reasons.append("missing_reference_identity")
    else:
        if evidence.get("reference_trajectory_sha256") != canonical_hash(reference):
            reasons.append("reference_identity_hash_mismatch")
        if any(reference[key] != expected[key] for key in _REFERENCE_SHARED):
            reasons.append("reference_input_recipe_or_source_mismatch")
        if reference.get("text_encoding") != expected.get("text_encoding"):
            reasons.append("reference_text_encoding_mismatch")
        if reference["prefix"]["cache"] or reference["suffix"]["cache"] or reference["action"]["kind"] != "none":
            reasons.append("reference_is_not_unmodified_cache_off")
    if evidence.get("quality_scope") != "full_trajectory_vs_cache_off_reference" or evidence.get("additive_quality_budget") is not False:
        reasons.append("not_full_trajectory_quality")
    metrics = evidence.get("metrics")
    reasons.extend(_quality_reasons(metrics, expected["spec"]["frames"]))
    if reasons:
        return dict(status="insufficient", qualified=False, reasons=sorted(set(reasons)), epsilon=epsilon)
    passed = metrics["lpips_mean"] <= epsilon
    return dict(status="qualified" if passed else "failed", qualified=passed,
                reasons=[] if passed else ["video_mean_lpips_exceeds_epsilon"], epsilon=epsilon,
                trajectory_sha256=evidence["trajectory_sha256"], evidence_sha256=evidence["sha256"],
                lpips_mean=metrics["lpips_mean"], lpips_p95=metrics["lpips_p95"],
                lpips_per_frame=copy.deepcopy(metrics["lpips_per_frame"]),
                temporal_delta_rmse=metrics.get("temporal_delta_rmse"),
                scope="empirical_full_video_mean_not_per_frame_or_quality_guarantee")


def _arm_quality(arm, epsilon):
    if not isinstance(arm, dict):
        return False
    try:
        return qualify_trajectory(arm.get("evidence"), arm.get("identity"), epsilon=epsilon)["qualified"]
    except (ValueError, TypeError):
        return False


def _paired_costs(trials, *, epsilon, minimum_gain, sp=False):
    """Every row is one independently repeated, complete two-arm comparison.

    Row fields: condition_id, pair_id, order=[A,B], status, measurement_kind,
    arms={A:{cost_ms, identity, evidence}, B:{...}}. State arms additionally need
    cost_scope='full_critical_path' and urgent={identity,evidence}, so a local
    restore improvement or an unqualified urgent video cannot pass. SP rows require phase
    compute/reuse, and each arm cost_scope='net_layout_window', with measured
    collectives_included/transitions_included and observed_path matching phase.
    """
    groups, rejected = defaultdict(list), []
    seen = set()
    for row in trials:
        reason = None
        order = row.get("order", [])
        arms = row.get("arms", {})
        key = (row.get("condition_id"), row.get("pair_id"), row.get("phase") if sp else None)
        if (not all(isinstance(value, str) and value for value in key[:2]) or key in seen or
                not isinstance(order, list) or len(order) != 2 or len(set(order)) != 2 or set(order) != set(arms)):
            reason = "invalid_or_duplicate_pair"
        seen.add(key)
        if row.get("status") != "completed" or row.get("measurement_kind") != "measured_gpu":
            reason = "incomplete_or_unmeasured_pair"
        if not reason and not all(_number(arms[name].get("cost_ms"), positive=True) and _arm_quality(arms[name], epsilon) for name in order):
            reason = "missing_cost_correctness_or_full_trajectory_quality"
        if not reason:
            left, right = (arms[name]["identity"] for name in order)
            if any(left[field] != right[field] for field in _REFERENCE_SHARED):
                reason = "unpaired_inputs_or_recipe"
            proofs = [arms[name]["evidence"] for name in order]
            if any(proofs[0].get(key) != proofs[1].get(key) for key in
                   ("reference_trajectory_sha256", "reference_output_sha256")):
                reason = "different_quality_reference_in_paired_comparison"
        if not sp and not reason:
            if not all(arms[name].get("cost_scope") == "full_critical_path" and
                       _arm_quality(arms[name].get("urgent"), epsilon) for name in order):
                reason = "missing_full_critical_path_or_urgent_quality"
            else:
                urgent_left, urgent_right = (arms[name]["urgent"]["identity"] for name in order)
                if any(urgent_left[field] != urgent_right[field] for field in _REFERENCE_SHARED):
                    reason = "unpaired_urgent_request"
        if sp and not reason:
            phase = row.get("phase")
            if phase not in ("compute", "reuse") or not all(
                    arms[name].get("cost_scope") == "net_layout_window" and
                    arms[name].get("collectives_included") is True and
                    arms[name].get("transitions_included") is True and
                    arms[name].get("observed_path") == phase for name in order):
                reason = "missing_measured_path_or_net_layout_cost"
            elif left["layout_history"] == right["layout_history"]:
                reason = "same_layout_cannot_prove_layout_selection"
            elif phase == "reuse" and not all(arms[name]["identity"]["prefix"]["cache"] or
                                               arms[name]["identity"]["suffix"]["cache"] for name in order):
                reason = "cache_off_trajectory_cannot_supply_reuse_path"
        if reason:
            rejected.append(dict(condition_id=row.get("condition_id"), pair_id=row.get("pair_id"), reason=reason))
            continue
        comparison = {name: {key: value for key, value in arms[name]["identity"].items()
                             if key not in ("input", "spec")} for name in sorted(order)}
        group_key = (row["condition_id"], tuple(sorted(order)), row.get("phase") if sp else None,
                     canonical_hash(comparison))
        groups[group_key].append(row)
    conditions = []
    for (condition, names, phase, comparison_hash), rows in sorted(groups.items()):
        order_set = {tuple(row["order"]) for row in rows}
        balanced = len(rows) >= 2 and set((names, names[::-1])).issubset(order_set)
        winners, margins = [], []
        for row in rows:
            winner = min(names, key=lambda name: (row["arms"][name]["cost_ms"], name))
            loser = next(name for name in names if name != winner)
            margin = 1. - row["arms"][winner]["cost_ms"] / row["arms"][loser]["cost_ms"]
            winners.append(winner)
            margins.append(margin)
        replicated = balanced and len(set(winners)) == 1 and all(value + 1e-12 >= minimum_gain for value in margins)
        conditions.append(dict(condition_id=condition, comparison=list(names), phase=phase,
            comparison_sha256=comparison_hash,
            pairs=len(rows), counterbalanced=balanced, replicated_gain=replicated,
            winner=winners[0] if replicated else None, improvement_fractions=margins,
            pair_ids=[row["pair_id"] for row in rows]))
    return conditions, rejected


def state_crossover_gate(trials, *, epsilon=.05, minimum_gain=.05):
    """Require counterbalanced >=5% effects with different winning conditions."""
    conditions, rejected = _paired_costs(trials, epsilon=epsilon, minimum_gain=minimum_gain)
    supported = any(a["replicated_gain"] and b["replicated_gain"] and a["comparison_sha256"] == b["comparison_sha256"] and
                    a["condition_id"] != b["condition_id"] and a["winner"] != b["winner"]
                    for a in conditions for b in conditions)
    comparable_conditions = any(a["comparison_sha256"] == b["comparison_sha256"] and
                                a["condition_id"] != b["condition_id"]
                                for a in conditions for b in conditions)
    complete = comparable_conditions and all(row["counterbalanced"] for row in conditions) and not rejected
    return dict(status="supported" if supported else "not_observed" if complete else "insufficient_evidence",
                supported=supported, minimum_gain=minimum_gain, conditions=conditions, rejected=rejected,
                scope="paired_replicated_condition_dependent_cost_not_statistical_significance")


def sp_path_gate(trials, *, epsilon=.05, minimum_gain=.05):
    """Require different compute/reuse layout winners after communication costs."""
    conditions, rejected = _paired_costs(trials, epsilon=epsilon, minimum_gain=minimum_gain, sp=True)
    supported = any(a["replicated_gain"] and b["replicated_gain"] and a["comparison_sha256"] == b["comparison_sha256"] and
                    a["condition_id"] == b["condition_id"] and a["phase"] == "compute" and b["phase"] == "reuse" and
                    a["winner"] != b["winner"] for a in conditions for b in conditions)
    complete = any(a["condition_id"] == b["condition_id"] and a["comparison_sha256"] == b["comparison_sha256"] and
                   a["phase"] == "compute" and b["phase"] == "reuse" and a["counterbalanced"] and b["counterbalanced"]
                   for a in conditions for b in conditions) and not rejected
    return dict(status="supported" if supported else "not_observed" if complete else "insufficient_evidence",
                supported=supported, minimum_gain=minimum_gain, conditions=conditions, rejected=rejected,
                scope="measured_compute_reuse_path_net_cost_including_collectives_and_layout_transitions")


def memory_admission_gate(trials, *, epsilon=.05):
    """Accept measured release enabling a real second request, not byte estimates.

    Rows contain a measured keep rejection, before/after memory observations,
    synchronized transfer, and a replacement arm with allocation, compute and
    decode completed plus its qualified complete-trajectory evidence.
    """
    accepted, rejected = [], []
    for row in trials:
        reasons = []
        if row.get("artificially_filled_memory") is True:
            reasons.append("artificial_pressure_is_not_natural_admission_evidence")
        if row.get("single_request_oom") is True:
            reasons.append("single_request_oom_is_not_state_admission_evidence")
        if row.get("status") != "completed" or row.get("measurement_kind") != "measured_gpu":
            reasons.append("unmeasured_or_incomplete_trial")
        if row.get("keep_rejected") is not True or row.get("keep_rejection_reason") != "measured_capacity":
            reasons.append("no_measured_keep_capacity_rejection")
        if row.get("released_mode") not in ("hot", "exact_refresh", "cold") or row.get("transfer_complete") is not True:
            reasons.append("no_completed_state_release")
        for name in ("before", "after"):
            memory = row.get(name, {})
            if (memory.get("measurement_kind") != "measured_gpu" or
                    not all(_number(memory.get(key)) for key in ("allocated_bytes", "reserved_bytes", "driver_free_bytes"))):
                reasons.append("missing_" + name + "_memory_observation")
        if not reasons and row["after"]["allocated_bytes"] >= row["before"]["allocated_bytes"]:
            reasons.append("no_observed_live_allocation_release")
        admitted = row.get("admitted", {})
        try:
            rejected_identity = trajectory_identity(row.get("keep_request_identity"))
        except (ValueError, TypeError):
            reasons.append("missing_identity_of_capacity_rejected_request")
        else:
            if rejected_identity != admitted.get("identity"):
                reasons.append("admitted_request_differs_from_capacity_rejected_request")
        if not all(admitted.get(key) is True for key in ("allocation_succeeded", "compute_completed", "decode_completed")):
            reasons.append("second_request_did_not_execute")
        if not _arm_quality(admitted, epsilon):
            reasons.append("second_request_correctness_or_quality_not_qualified")
        if not _arm_quality(row.get("resumed", {}), epsilon):
            reasons.append("resumed_request_correctness_or_quality_not_qualified")
        elif row["resumed"]["identity"]["action"]["kind"] != row.get("released_mode"):
            reasons.append("resumed_evidence_is_for_a_different_state_action")
        if reasons:
            rejected.append(dict(trial_id=row.get("trial_id"), reasons=reasons))
        else:
            accepted.append(row.get("trial_id"))
    return dict(status="supported" if accepted else "insufficient_evidence", supported=bool(accepted),
                accepted_trial_ids=accepted, rejected=rejected,
                scope="observed_release_followed_by_real_allocation_compute_decode_and_qualified_resume")


def research_gates(*, memory_trials=(), state_trials=(), sp_trials=(), epsilon=.05):
    return dict(memory_admission=memory_admission_gate(memory_trials, epsilon=epsilon),
                state_crossover=state_crossover_gate(state_trials, epsilon=epsilon),
                sp_path=sp_path_gate(sp_trials, epsilon=epsilon),
                missing_evidence_semantics="insufficient_evidence_is_not_a_negative_research_result")
