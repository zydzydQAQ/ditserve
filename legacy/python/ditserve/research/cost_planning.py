"""Conservative development-only planning for the bounded research pilot.

This module performs no measurements and changes no protocol or experiment.
Recorded whole-generation time must include artifact writes. CPU encoding must
either be included or explicitly covered by the frozen-text setup reservation;
quality time is the measured whole-video CPU LPIPS evaluation time. Unknown
transfer time may use a separately labelled configuration reservation. Unknown
generation or quality time can never be replaced by zero or an extrapolation.
"""

import copy
import math

from ..benchmark.common import canonical_hash, seal, verify_seal
from .common import read_json


CONDITION_SLOTS = 2
MIN_FIXED_RESERVE_SECONDS = 600.
MIN_SAFETY_MULTIPLIER = 1.3
FROZEN_ENCODING_MODE = "frozen_text_stage_setup_reserved_separately"


def read_bound_cost_plan(root, frozen):
    """Only this protocol's completed probe can set future execution guards."""
    from pathlib import Path
    root = Path(root)
    plan = verify_seal(read_json(root / "cost-plan.json"))
    probe = read_json(root / "probe.json")
    if (plan.get("schema_version") != "ditserve.research.cost_plan.v1" or plan.get("status") != "planned"
            or plan.get("protocol_sha256") != frozen["sha256"]
            or plan.get("source_sha256") != frozen["source_sha256"]
            or probe.get("protocol_sha256") != frozen["sha256"]
            or probe.get("status") != "completed" or probe.get("cost_plan") != plan
            or probe.get("selected_shape") != plan.get("selected_shape")
            or type(plan.get("candidate_limit")) is not int or not 1 <= plan["candidate_limit"] <= 2
            or any(not _number(plan.get("unit_estimates", {}).get(stage), positive=True)
                   for stage in ("calibrate", "mechanism"))):
        raise ValueError("cost plan is not bound to this completed frozen probe")
    return plan


def _number(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def work_counts(candidate_count, screen_count=12):
    """Upper bounds for the current fixed scientific matrix, with two conditions.

    G=48+2S+30C+2L+8CL complete generations; Q=24+26C+8CL full-video
    quality calls; A=S+14C+4CL state actions. Each state action includes both
    snapshot and restore overhead. Up to C natural capacity trials are reserved
    even though successful keep or admission can stop those trials early.
    """
    if type(candidate_count) is not int or not 1 <= candidate_count <= 2:
        raise ValueError("the scientific plan permits one or two joint candidates")
    if type(screen_count) is not int or not 1 <= screen_count <= 12:
        raise ValueError("screen_count must describe 1..12 predeclared candidates")
    c, s, l = candidate_count, screen_count, CONDITION_SLOTS
    stages = dict(
        initial_calibration=dict(generations=24, quality_calls=24, state_actions=0),
        cost_screen=dict(generations=2*s, quality_calls=0, state_actions=s),
        development_and_holdout=dict(generations=24+24*c, quality_calls=24*c, state_actions=12*c),
        counterbalanced_conditions=dict(generations=l*(2+8*c), quality_calls=8*c*l, state_actions=4*c*l),
        natural_capacity=dict(generations=6*c, quality_calls=2*c, state_actions=2*c))
    return dict(candidate_count=c, screen_count=s, budgeted_condition_slots=l,
                generations=sum(row["generations"] for row in stages.values()),
                quality_calls=sum(row["quality_calls"] for row in stages.values()),
                state_actions=sum(row["state_actions"] for row in stages.values()), stages=stages)


def _validate_measurement(row, transfer_reserve_seconds):
    reasons = []
    shape = row.get("shape", {})
    if not isinstance(shape, dict) or any(type(shape.get(key)) is not int or shape[key] <= 0
                                        for key in ("height", "width", "frames", "steps")):
        reasons.append("missing_complete_shape")
    elif shape["steps"] != 32 or shape["frames"] not in (17, 33, 49, 81):
        reasons.append("shape_outside_frozen_pilot")
    if row.get("measurement_kind") != "measured_gpu" or row.get("status", "completed") != "completed":
        reasons.append("not_completed_gpu_probe")
    if row.get("split", "dev") != "dev":
        reasons.append("holdout_cannot_select_shape_or_candidate_count")
    if not _sha256(row.get("source_sha256")):
        reasons.append("missing_source_sha256")
    for name in ("generation_wall_seconds", "quality_wall_seconds"):
        if not _number(row.get(name), positive=True):
            reasons.append("missing_measured_" + name)
    scope = row.get("timing_scope", {})
    encoding_covered = (isinstance(scope, dict) and (
        scope.get("encoding_included") is True or (
            scope.get("encoding_included") is False and scope.get("encoding_mode") == FROZEN_ENCODING_MODE)))
    if not encoding_covered or scope.get("archive_included") is not True:
        reasons.append("generation_timing_must_include_encoding_and_archive")
    if row.get("generation_statistic", "maximum") != "maximum":
        reasons.append("generation_cost_must_be_observed_maximum")
    if row.get("quality_scope", "whole_video") != "whole_video":
        reasons.append("quality_cost_must_cover_full_video")
    transfer = row.get("transfer_wall_seconds")
    reservation = row.get("transfer_reservation_seconds")
    if transfer is not None and not _number(transfer, positive=True):
        reasons.append("invalid_measured_transfer_cost")
    if reservation is not None and not _number(reservation, positive=True):
        reasons.append("invalid_configured_transfer_reservation")
    if transfer is None and reservation is None and transfer_reserve_seconds is None:
        reasons.append("transfer_cost_unmeasured_without_explicit_reservation")
    return reasons


def choose_cost_plan(probes, *, remaining_calibrate_seconds, screen_count=12,
                     max_candidates=2, transfer_reserve_seconds=None,
                     transfer_reservation_seconds=None,
                     fixed_reserve_seconds=MIN_FIXED_RESERVE_SECONDS,
                     safety_multiplier=MIN_SAFETY_MULTIPLIER):
    """Choose the largest measured affordable shape for at least one candidate.

    Required probe fields are ``shape``, ``measurement_kind='measured_gpu'``,
    ``source_sha256``, ``generation_wall_seconds`` (maximum whole-generation
    wall time), ``quality_wall_seconds`` (whole-video evaluation time), and
    ``timing_scope={encoding_included: True, archive_included: True}``. Encoding
    may be excluded ONLY with ``encoding_mode`` equal to
    ``frozen_text_stage_setup_reserved_separately``; the fixed 600-second minimum
    then covers text-bank preparation, lifecycle and control costs separately.
    ``transfer_wall_seconds`` is the measured extra time of a complete state
    action; if absent, callers must explicitly reserve its cost through
    ``transfer_reserve_seconds`` (alias ``transfer_reservation_seconds``), or a
    per-probe ``transfer_reservation_seconds`` configuration reservation. These
    are per action, not a session total, and are never labelled measurements.

    A second candidate is admitted only if the SAME selected shape still fits.
    Cost conditions use the smallest eligible measured shape and selected
    shape; a larger probed shape is never silently introduced. A single unique
    shape is permitted, but cannot establish a condition-dependent crossover.
    Two condition slots remain reserved conservatively even in that case.

    This estimates the entire calibration + mechanism matrix from scratch.
    Callers must not subtract already-completed work implicitly; a later plan
    is intentionally conservative. Freeze the resulting count before holdout.
    """
    work_counts(max_candidates, screen_count)
    if transfer_reservation_seconds is not None:
        if transfer_reserve_seconds is not None:
            raise ValueError("use only one transfer reservation parameter spelling")
        transfer_reserve_seconds = transfer_reservation_seconds
    if not _number(remaining_calibrate_seconds):
        raise ValueError("remaining calibration bucket must be finite and nonnegative")
    if not _number(fixed_reserve_seconds) or fixed_reserve_seconds < MIN_FIXED_RESERVE_SECONDS:
        raise ValueError("fixed encoding/setup/lifecycle reserve must be at least 600 seconds")
    if not _number(safety_multiplier) or safety_multiplier < MIN_SAFETY_MULTIPLIER:
        raise ValueError("safety multiplier cannot be less than 1.3")
    if transfer_reserve_seconds is not None and not _number(transfer_reserve_seconds, positive=True):
        raise ValueError("an explicit transfer reservation must be positive; missing is not zero")
    probes = copy.deepcopy(list(probes))
    if any(not isinstance(row, dict) for row in probes):
        raise ValueError("probe measurements must be objects")
    # Reject nonfinite/unserializable provenance rather than losing the exact
    # inputs or writing a JSON artifact containing NaN/Infinity.
    canonical_hash(probes)
    rejected, valid = [], []
    for index, row in enumerate(probes):
        reasons = _validate_measurement(row, transfer_reserve_seconds)
        if reasons:
            rejected.append(dict(measurement_index=index, shape=row.get("shape"), reasons=reasons))
        else:
            valid.append(dict(index=index, record=row, sha256=canonical_hash(row)))
    result = dict(schema_version="ditserve.research.cost_plan.v1", status="insufficient_evidence", choice=None,
        selected_shape=None, candidate_limit=0, paired_conditions=[], selected_cost_observation=None, unit_estimates={},
        executes_gpu_work=False, measurement_records=probes, rejected_measurements=rejected,
        budget=dict(bucket="calibrate", remaining_seconds=remaining_calibrate_seconds,
                    fixed_reserve_seconds=fixed_reserve_seconds, safety_multiplier=safety_multiplier),
        formula=dict(total="fixed_reserve + safety_multiplier * (G * max_generation_wall + Q * max_quality_wall + A * extra_state_action_wall)",
                     G="48 + 2*S + 30*C + 2*L + 8*C*L", Q="24 + 26*C + 8*C*L", A="S + 14*C + 4*C*L",
                     S=screen_count, L=CONDITION_SLOTS,
                     timing="generation includes GPU work, decode and artifact I/O; CPU encoding is included or reserved as frozen text stage setup; quality covers full video",
                     lifecycle="fixed reserve includes frozen text preparation for calibration and mechanism, model setup/cleanup and control overhead; all held-GPU wall time remains charged by external ledger"),
        setup_reservation=dict(seconds=fixed_reserve_seconds, provenance="configuration", measured=False,
                              includes=["frozen_text_stage_setup", "model_setup_cleanup", "control_overhead"]),
        transfer_reservation=None if transfer_reserve_seconds is None else dict(
            per_action_seconds=transfer_reserve_seconds, provenance="configuration", measured=False),
        candidates_evaluated=[], selection_split="dev", holdout_used=False,
        decision_rule="largest affordable shape with one candidate, then at most two at the same shape if affordable",
        scientific_scope_changed=False,
        uncertainty="engineering reservation, not a statistical upper confidence bound or a guarantee against timeout")
    if not valid:
        result["reason"] = "no_complete_measured_generation_and_quality_cost_basis"
        return seal(result)
    # Changing shape is the intended probe dimension. Runtime/source identity
    # and spatial resolution must otherwise agree wherever supplied.
    context_fields = ("source_sha256", "protocol_sha256", "execution_identity")
    contexts = {canonical_hash({key: entry["record"].get(key) for key in context_fields}) for entry in valid}
    spatial = {(entry["record"]["shape"]["height"], entry["record"]["shape"]["width"], entry["record"]["shape"]["steps"])
               for entry in valid}
    shape_ids = [canonical_hash(entry["record"]["shape"]) for entry in valid]
    if len(contexts) != 1 or len(spatial) != 1 or len(set(shape_ids)) != len(shape_ids):
        result["reason"] = "mixed_probe_context_or_duplicate_shape_maxima"
        return seal(result)
    valid.sort(key=lambda entry: entry["record"]["shape"]["frames"])
    smallest = valid[0]
    choices = []
    for entry in valid:
        bounds = [smallest] if entry is smallest else [smallest, entry]
        generation = max(item["record"]["generation_wall_seconds"] for item in bounds)
        quality = max(item["record"]["quality_wall_seconds"] for item in bounds)
        transfers = []
        for item in bounds:
            observed = item["record"].get("transfer_wall_seconds")
            configured = [value for value in (transfer_reserve_seconds, item["record"].get("transfer_reservation_seconds")) if value is not None]
            reservation = max(configured) if configured else None
            value = max(v for v in (observed, reservation) if v is not None)
            transfers.append(dict(measurement_index=item["index"], seconds=value,
                provenance="configuration" if observed is None or (reservation is not None and reservation > observed) else "measurement",
                observed_seconds=observed, reserved_seconds=reservation))
        transfer = max(item["seconds"] for item in transfers)
        for count in range(1, max_candidates + 1):
            work = work_counts(count, screen_count)
            components = dict(generation_seconds=work["generations"] * generation,
                              quality_seconds=work["quality_calls"] * quality,
                              extra_state_action_seconds=work["state_actions"] * transfer)
            subtotal = sum(components.values())
            total = fixed_reserve_seconds + safety_multiplier * subtotal
            choice = dict(shape=copy.deepcopy(entry["record"]["shape"]), candidate_count=count,
                paired_conditions=[copy.deepcopy(item["record"]["shape"]) for item in bounds],
                distinct_condition_count=len(bounds), crossover_design_complete=len(bounds) == 2,
                measurement_indices=[item["index"] for item in bounds],
                measurement_sha256=[item["sha256"] for item in bounds], work_counts=work,
                cost_bounds=dict(generation_wall_seconds=generation, quality_wall_seconds=quality,
                                 extra_state_action_wall_seconds=transfer, transfers=transfers),
                unscaled_components=components, unscaled_work_seconds=subtotal,
                estimate_seconds=total, fits_budget=total < remaining_calibrate_seconds,
                planned_headroom_seconds=remaining_calibrate_seconds-total)
            choices.append(choice)
    result["candidates_evaluated"] = choices
    one_candidate = [row for row in choices if row["candidate_count"] == 1 and row["fits_budget"]]
    if not one_candidate:
        result.update(status="insufficient_budget", reason="no_measured_shape_fits_one_complete_candidate_matrix")
        return seal(result)
    selected_shape = max(one_candidate, key=lambda row: row["shape"]["frames"])["shape"]
    selected = max((row for row in choices if row["shape"] == selected_shape and row["fits_budget"]),
                   key=lambda row: row["candidate_count"])
    result.update(status="planned", choice=selected,
        selected_shape=copy.deepcopy(selected_shape), candidate_limit=selected["candidate_count"],
        paired_conditions=copy.deepcopy(selected["paired_conditions"]),
        selected_cost_observation=dict(
            generation_wall_seconds=selected["cost_bounds"]["generation_wall_seconds"],
            quality_wall_seconds=selected["cost_bounds"]["quality_wall_seconds"],
            transfer_reservation_seconds=selected["cost_bounds"]["extra_state_action_wall_seconds"],
            transfer_provenance=copy.deepcopy(selected["cost_bounds"]["transfers"]),
            measurement_indices=selected["measurement_indices"],
            timing_scope="maximum_of_smallest_and_selected_probe; encoding_included_or_setup_reserved"),
        reason="development_cost_only_choice; lock candidate count before holdout and keep all required cells",
        continuation="external runner must recheck actual ledger at complete-unit boundaries; do not borrow the unavailable-SP bucket")
    g = selected["cost_bounds"]["generation_wall_seconds"]
    q = selected["cost_bounds"]["quality_wall_seconds"]
    x = selected["cost_bounds"]["extra_state_action_wall_seconds"]
    result["unit_estimates"] = dict(calibrate=safety_multiplier*(3*g+3*q)+90,
                                    mechanism=safety_multiplier*(6*g+4*q+2*x)+90)
    result["formula"]["unit_estimates"] = dict(
        calibrate="safety_multiplier * (3*max_generation_wall + 3*max_quality_wall) + 90",
        mechanism="safety_multiplier * (6*max_generation_wall + 4*max_quality_wall + 2*extra_state_action_wall) + 90",
        scope="complete-unit guard only; 90 seconds is covered by the total setup/margin reservation, not multiplied into total work")
    return seal(result)
