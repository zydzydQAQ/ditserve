"""CPU-only, output-bound gates for the finite CFG policy experiment.

Oracle measurements may reject an unpromising implementation; they cannot
qualify performance. Every production cost decision uses fresh, uninstrumented
full-trajectory pairs and the sum of costs, never an average of sample gains.
"""

import math
import re
import statistics

from .common import digest
from .dual_matrix import bound_quality, inputs, known_quality, seal, _trajectory_matches

CANDIDATES = ("independent_004", "paired_008")
DEVICES = ("a100", "v100")
MIGRATION_ACTIONS = ("full_migrate", "cold_migrate")
SCREEN_INPUTS = (("origami", 101), ("origami", 102), ("kettle", 101))
BOUND_METHOD = "oracle_excluding_snapshot_and_discarded_trial"


def _number(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _candidate(value):
    if value not in CANDIDATES:
        raise ValueError("only the two frozen CFG candidate policies are eligible")
    return value


def _expected(config, split):
    result = {(x["prompt_id"], x["seed"]): x for x in inputs(config, split)}
    if len(result) != (8 if split == "dev" else 4):
        raise ValueError("the frozen matrix requires eight development and four held-out inputs")
    return result


def _policy_matches(run, config, recipe, implementation):
    from .cfg_policy import resolve_policy
    policy = resolve_policy(config, recipe, implementation=implementation)
    identity = run.get("identity", {})
    return (identity.get("policy_sha256") == policy["sha256"]
            and identity.get("implementation") == policy["implementation"])


def _trajectory(run, item, context, config, *, role, recipe, action="stay", phase="dev"):
    path = [role] if action == "stay" else ["v100", "a100"]
    implementation = "oracle" if recipe == "paired_008" and phase == "stage_a" else (
        "split_prefix" if recipe == "paired_008" else "native")
    return (_trajectory_matches(run, item, context, role=role, recipe=recipe, action=action,
                path=path, boundary=config.get("transition_step", 16))
            and run["identity"]["attention_path"] == [config["attention"][device.upper()] for device in path]
            and _policy_matches(run, config, recipe, implementation))


def _quality_status(metric, candidate, reference):
    """Separate observed quality failure from absent/misbound evidence."""
    if not isinstance(metric, dict) or not known_quality(metric, epsilon=float("inf")):
        return "unverified"
    for label, run in (("candidate", candidate), ("reference", reference)):
        if not isinstance(run, dict) or not isinstance(run.get("identity"), dict):
            return "unverified"
        output_hash = run.get("output_sha256")
        if (not re.fullmatch(r"[0-9a-f]{64}", str(output_hash))
                or metric.get(label + "_output_sha256") != output_hash
                or metric.get(label + "_identity_sha256") != digest(run["identity"])):
            return "unverified"
    return "passed" if bound_quality(metric, candidate, reference) else "failed"


def _timing_eligible(row):
    runs = [row[k] for k in ("candidate", "reference", "run") if isinstance(row.get(k), dict)]
    if isinstance(row.get("runs"), dict):
        runs += [r for r in row["runs"].values() if isinstance(r, dict)]
    clean_runs = all(r.get("diagnostic", False) is False and r.get("instrumented", False) is False
                     and r.get("identity", {}).get("implementation") != "oracle" for r in runs)
    return (clean_runs and row.get("performance_valid") is True and row.get("timing_eligible") is True
            and row.get("measurement_kind") not in ("oracle", "diagnostic")
            and row.get("diagnostic", False) is False
            and row.get("instrumented", False) is False)


def _fresh(rows):
    pair_ids = [r.get("pair_id") for r in rows]
    generations = [r.get(k, {}).get("generation_id") for r in rows for k in ("candidate", "reference")]
    return (bool(rows) and all(r.get("fresh_pair") is True for r in rows)
            and all(isinstance(x, str) and x for x in pair_ids + generations)
            and len(set(pair_ids)) == len(pair_ids)
            and len(set(generations)) == len(generations))


def _cache_quality(row, item, context, config, recipe, phase):
    candidate, reference = row.get("candidate", {}), row.get("reference", {})
    role = row.get("device_role")
    if (row.get("status") != "completed" or row.get("performance_valid") is not True
            or row.get("context_sha256") != context or row.get("split") != item["split"]
            or not _trajectory(candidate, item, context, config, role=role, recipe=recipe, phase=phase)
            or not _trajectory(reference, item, context, config, role=role, recipe="off", phase=phase)
            or candidate["identity"]["attention_path"] != reference["identity"]["attention_path"]):
        return "unverified"
    return _quality_status(row.get("quality", {}), candidate, reference)


def _locked_backends(rows):
    return all(len({tuple(r.get("candidate", {}).get("identity", {}).get("attention_path", []))
                    for r in rows if r.get("device_role") == role}) == 1 for role in DEVICES)


def _status(complete, quality, *, additional=True):
    if not complete or "unverified" in quality:
        return "unverified"
    return "passed" if quality and all(q == "passed" for q in quality) and additional else "failed"


def optimistic_cost(candidate):
    """Only subtract separately measured snapshot and discarded-trial cost."""
    bound = candidate.get("optimistic_bound", {})
    names = ("elapsed_ms", "snapshot_ms", "discarded_trial_ms", "optimistic_ms")
    if (bound.get("method") != BOUND_METHOD or not all(_number(bound.get(k)) for k in names)
            or not _number(candidate.get("elapsed_ms"), True)
            or not math.isclose(bound["elapsed_ms"], candidate["elapsed_ms"], rel_tol=1e-7, abs_tol=1e-5)):
        return None
    estimated = bound["elapsed_ms"] - bound["snapshot_ms"] - bound["discarded_trial_ms"]
    if (estimated <= 0 or not math.isclose(bound["optimistic_ms"], estimated, rel_tol=1e-7, abs_tol=1e-5)):
        return None
    return estimated


def stage_a_gate(rows, config, candidate, context_sha256):
    candidate = _candidate(candidate)
    expected = _expected(config, "dev")
    expected = {k: expected[k] for k in SCREEN_INPUTS}
    selected = [r for r in rows if r.get("type") == "cfg_screen" and r.get("phase") == "stage_a"
                and r.get("recipe_id") == candidate]
    keys = [(r.get("device_role"), r.get("prompt_id"), r.get("seed")) for r in selected]
    required = {(role, *key) for role in DEVICES for key in expected}
    complete = len(keys) == len(required) and set(keys) == required
    qualities = [_cache_quality(r, expected[(r["prompt_id"], r["seed"])], context_sha256, config,
                                candidate, "stage_a") for r in selected] if complete else []
    complete = complete and _fresh(selected) and _locked_backends(selected)
    bounds = {}
    if candidate == "paired_008" and complete:
        for role in DEVICES:
            group = [r for r in selected if r["device_role"] == role]
            costs = [optimistic_cost(r["candidate"]) for r in group]
            refs = [r["reference"].get("elapsed_ms") for r in group]
            if None in costs or not all(_number(x, True) for x in refs):
                complete = False
                bounds[role] = dict(status="unverified", reason="recorded oracle overhead decomposition is missing or inconsistent")
            else:
                ratio = sum(costs) / sum(refs)
                bounds[role] = dict(status="passed" if ratio <= .95 else "failed", normalized_ratio=ratio,
                                    optimistic_gain=1 - ratio, optimistic_ms=sum(costs), reference_ms=sum(refs),
                                    method=BOUND_METHOD, performance_qualified=False)
    passed_bound = candidate != "paired_008" or len(bounds) == 2 and all(b["status"] == "passed" for b in bounds.values())
    status = _status(complete, qualities, additional=passed_bound)
    return seal(dict(schema_version="ditserve.cfg.stage_a_gate.v1", candidate=candidate, status=status,
                     eligible=status == "passed", observed=len(keys), required=len(required),
                     quality_statuses=qualities, optimistic_bounds=bounds, context_sha256=context_sha256,
                     performance_qualified=False, scope="screen only; oracle costs never qualify performance"))


def development_gate(rows, config, candidate, context_sha256):
    candidate = _candidate(candidate)
    expected = _expected(config, "dev")
    orders = (("off", candidate), (candidate, "off"))
    selected = [r for r in rows if r.get("type") == "cfg_calibration" and r.get("phase") == "dev"
                and r.get("recipe_id") == candidate]
    keys = [(r.get("device_role"), tuple(r.get("order", [])), r.get("prompt_id"), r.get("seed")) for r in selected]
    required = {(role, order, *key) for role in DEVICES for order in orders for key in expected}
    complete = len(keys) == len(required) and set(keys) == required
    qualities = [_cache_quality(r, expected[(r["prompt_id"], r["seed"])], context_sha256, config,
                                candidate, "dev") for r in selected] if complete else []
    valid = (complete and _fresh(selected) and _locked_backends(selected)
             and all(_timing_eligible(r) and _number(r["candidate"].get("elapsed_ms"), True)
                     and _number(r["reference"].get("elapsed_ms"), True) for r in selected))
    costs = []
    if valid:
        for role in DEVICES:
            for order in orders:
                group = [r for r in selected if r["device_role"] == role and tuple(r["order"]) == order]
                candidate_ms = sum(r["candidate"]["elapsed_ms"] for r in group)
                reference_ms = sum(r["reference"]["elapsed_ms"] for r in group)
                ratio = candidate_ms / reference_ms
                costs.append(dict(device_role=role, order=list(order), candidate_ms=candidate_ms,
                                  reference_ms=reference_ms, normalized_ratio=ratio, gain=1 - ratio))
    status = _status(valid, qualities, additional=bool(costs) and all(c["normalized_ratio"] <= .95 for c in costs))
    return seal(dict(schema_version="ditserve.cfg.development_gate.v1", candidate=candidate,
                     status=status, eligible=status == "passed", observed=len(keys), required=len(required),
                     quality_statuses=qualities, device_orders=costs,
                     mean_normalized_ratio=statistics.mean(c["normalized_ratio"] for c in costs) if costs else None,
                     context_sha256=context_sha256, performance_qualified=status == "passed",
                     selection_split="dev", holdout_used=False))


def select_candidate(rows, config, context_sha256, stage_a=None, *, frozen=None):
    if frozen is None:
        screens = {name: stage_a_gate(rows, config, name, context_sha256) for name in CANDIDATES}
    else:
        if frozen.get("config") != config:
            raise ValueError("candidate selection config differs from its frozen protocol")
        from .cfg_continuation import screening_decisions
        screens = screening_decisions(frozen, rows, context_sha256)
    if stage_a is not None and stage_a != screens:
        raise ValueError("stage A eligibility must be recomputed from the bound evidence")
    active = list(CANDIDATES)
    if frozen is not None and frozen.get("implementation_refinement"):
        from .cfg_refinement import development_candidates
        active = development_candidates(frozen, screens)
    development = {name: development_gate(rows, config, name, context_sha256)
                   for name in active if screens[name]["eligible"]}
    missing = [name for name in active if screens[name]["status"] == "unverified"
               or name in development and development[name]["status"] == "unverified"]
    qualified = [g for g in development.values() if g["eligible"]]
    formal_rows = [r for r in rows if r.get("type") == "cfg_calibration" and r.get("phase") == "dev"
                   and r.get("recipe_id") in development]
    if frozen is not None and frozen.get("implementation_refinement"):
        from .cfg_refinement import guarded_split_trace
        if any(not guarded_split_trace(r.get("candidate", {})) for r in formal_rows):
            missing.append("guarded_split_step_evidence")
    if development and formal_rows and not _fresh(formal_rows):
        missing.append("fresh_pairs_across_candidates")
    qualified.sort(key=lambda g: (g["mean_normalized_ratio"], CANDIDATES.index(g["candidate"])))
    selected = qualified[0]["candidate"] if qualified and not missing else None
    return seal(dict(schema_version="ditserve.cfg.selection.v1", status="selected" if selected else
                     "unverified" if missing else "direction_stopped", selected=selected,
                     stage_a=screens, development=development, missing=missing,
                     context_sha256=context_sha256, selection_split="dev", holdout_used=False,
                     tie_break="independent_004", criterion="mean of four device/order total-cost ratios"))


def holdout_gate(rows, config, selected, context_sha256):
    selected = _candidate(selected)
    expected = _expected(config, "holdout")
    selected_rows = [r for r in rows if r.get("type") == "cfg_calibration" and r.get("phase") == "holdout"
                     and r.get("recipe_id") == selected]
    keys = [(r.get("device_role"), r.get("prompt_id"), r.get("seed")) for r in selected_rows]
    required = {(role, *key) for role in DEVICES for key in expected}
    complete = (len(keys) == len(required) and set(keys) == required and _fresh(selected_rows)
                and _locked_backends(selected_rows)
                and all(r.get("order") in (["off", selected], [selected, "off"]) for r in selected_rows))
    quality = [_cache_quality(r, expected[(r["prompt_id"], r["seed"])], context_sha256, config,
                             selected, "holdout") for r in selected_rows] if complete else []
    status = _status(complete, quality)
    return seal(dict(schema_version="ditserve.cfg.holdout_gate.v1", selected=selected, status=status,
                     eligible=status == "passed", observed=len(keys), required=len(required),
                     quality_statuses=quality, context_sha256=context_sha256, holdout_used_for_selection=False,
                     reselect_allowed=False, performance_qualified=False))


def migration_quality_gate(rows, config, split, selected, action, context_sha256):
    selected = _candidate(selected)
    if action not in MIGRATION_ACTIONS:
        raise ValueError("only full and explicit cold migration are declared")
    expected = _expected(config, split)
    selected_rows = [r for r in rows if r.get("type") == "cfg_migration_quality" and r.get("action") == action
                     and r.get("recipe_id") == selected and r.get("split") == split]
    keys = [(r.get("prompt_id"), r.get("seed")) for r in selected_rows]
    complete = len(keys) == len(expected) and set(keys) == set(expected)
    quality = []
    backends = set()
    for row in selected_rows if complete else []:
        item = expected[(row["prompt_id"], row["seed"])]
        candidate, common, path = (row.get(k, {}) for k in ("candidate", "common_reference", "path_reference"))
        if (row.get("status") != "completed" or row.get("performance_valid") is not True
                or row.get("context_sha256") != context_sha256
                or not _trajectory(candidate, item, context_sha256, config, role="v100", recipe=selected, action=action)
                or not _trajectory(common, item, context_sha256, config, role="a100", recipe="off")
                or not _trajectory(path, item, context_sha256, config, role="v100", recipe="off", action="full_migrate")
                or candidate["identity"]["attention_path"] != path["identity"]["attention_path"]):
            quality.append("unverified")
            continue
        backends.add(tuple(candidate["identity"]["attention_path"]))
        quality.extend((_quality_status(row.get("path_reference_quality", {}), path, common),
                        _quality_status(row.get("canonical_quality", {}), candidate, common),
                        _quality_status(row.get("path_quality", {}), candidate, path)))
    status = _status(complete and len(backends) == 1, quality)
    return seal(dict(schema_version="ditserve.cfg.migration_quality_gate.v1", selected=selected, action=action,
                     split=split, status=status, eligible=status == "passed", observed=len(keys), required=len(expected),
                     quality_statuses=quality, context_sha256=context_sha256, error_composition="none",
                     scope="complete trajectories vs both common and identical-device-path cache-off references"))


def mechanism_gate(pairs, *, condition, context_sha256, quality_passed=False):
    if condition not in ("idle", "busy"):
        raise ValueError("only measured idle and real-queue conditions are declared")
    if isinstance(quality_passed, dict):
        qualified_actions = {k for k in MIGRATION_ACTIONS if quality_passed.get(k) is True}
    else:
        qualified_actions = set(MIGRATION_ACTIONS) if quality_passed is True else set()
    names = {"stay", *qualified_actions}
    complete = (bool(qualified_actions) and len(pairs) == 2 and set(pairs[0].get("order", [])) == names
                and len(pairs[0]["order"]) == len(names)
                and pairs[0]["order"] == list(reversed(pairs[1].get("order", []))))
    evidence_valid = complete
    for row in pairs:
        costs = row.get("costs_ms", {})
        evidence_valid &= (_timing_eligible(row) and row.get("context_sha256") == context_sha256
                           and row.get("condition") == condition and row.get("status") == "completed"
                           and row.get("all_requests_decoded") is True
                           and row.get("quality_passed") is True
                           and row.get("cost_scope") == "all_requests_decode_complete"
                           and set(costs) == names and all(_number(costs.get(k), True) for k in names))
        if condition == "busy":
            evidence_valid &= (row.get("queue_coverage") is True and row.get("real_queue") is True
                               and row.get("queue_length") in (1, 2, 4, 8)
                               and _number(row.get("queue_wait_ms")) and _number(row.get("break_even_wait_ms"), True)
                               and row["queue_wait_ms"] >= 1.25 * row["break_even_wait_ms"])
    gains = {name: [1 - r["costs_ms"][name] / r["costs_ms"]["stay"] for r in pairs]
             for name in sorted(qualified_actions)} if evidence_valid else {}
    supported = [name for name, values in gains.items() if all(v >= .05 for v in values)]
    return seal(dict(schema_version="ditserve.cfg.mechanism_gate.v1", condition=condition,
                     status="supported" if supported else "not_observed" if gains else "unverified",
                     supported=bool(supported), supported_actions=supported, gains=gains,
                     context_sha256=context_sha256, queue_coverage_verified=bool(evidence_valid and condition == "busy"),
                     scope="engineering threshold; does not establish cache-joint-controller innovation"))
