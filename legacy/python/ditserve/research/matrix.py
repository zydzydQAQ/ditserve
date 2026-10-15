"""Bounded, development-only experiment planning and selection.

No filesystem, model, CUDA, or scheduling side effects occur here. A candidate
selected by cost screening still needs all eight complete development quality
trajectories. Holdout observations never select a recipe or a recovery policy.
"""

import copy
import math
import statistics
from collections import defaultdict

from ..benchmark.common import canonical_hash, seal, verify_seal
from .evidence import qualify_trajectory


RECIPES = (
    dict(id="off", cache=False, threshold=0., max_continuous=1),
    dict(id="t004", cache=True, threshold=.04, max_continuous=1),
    dict(id="t008", cache=True, threshold=.08, max_continuous=1),
)
ACTIONS = ("keep", "hot", "exact_refresh", "cold")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _config(recipe):
    return {key: recipe[key] for key in ("cache", "threshold", "max_continuous")}


def _protocol_hash(protocol):
    return verify_seal(protocol)["sha256"] if "sha256" in protocol else canonical_hash(protocol)


def _bound_protocol_hash(protocol, supplied):
    if supplied is None:
        return _protocol_hash(protocol)
    if not isinstance(supplied, str) or len(supplied) != 64 or any(c not in "0123456789abcdef" for c in supplied):
        raise ValueError("frozen protocol SHA256 must be supplied explicitly")
    return supplied


def _protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("schema_version") != "ditserve.research.protocol.v1":
        raise ValueError("expected research protocol v1")
    _protocol_hash(protocol)
    recipes = protocol.get("recipes", list(RECIPES))
    if len(recipes) != 3 or {canonical_hash(row) for row in recipes} != {canonical_hash(row) for row in RECIPES}:
        raise ValueError("freeze exactly off, t004/m1 and t008/m1 recipes")
    if tuple(protocol.get("dev_seeds", (101, 102))) != (101, 102) or tuple(protocol.get("holdout_seeds", (201, 202))) != (201, 202):
        raise ValueError("development and holdout seeds must remain disjoint and frozen")
    prompts = protocol.get("prompts", [])
    ids = [row.get("id") for row in prompts]
    if len(set(ids)) != len(ids) or any(not isinstance(row.get("id"), str) or not row["id"] or
                                      not isinstance(row.get("text"), str) or not row["text"] for row in prompts):
        raise ValueError("unique prompt IDs and exact nonempty text are required")
    dev = [row for row in prompts if row.get("split") == "dev"]
    holdout = [row for row in prompts if row.get("split") == "holdout"]
    if len(dev) != 4 or {row["id"] for row in holdout} != {"tram", "aquarium"} or len(prompts) != 6:
        raise ValueError("require four development prompts and tram/aquarium holdout only")
    if protocol.get("transition_step", 16) != 16 or protocol.get("steps", 32) != 32:
        raise ValueError("this bounded matrix requires one optional change at step 16 of 32")
    if protocol.get("max_joint_candidates", 2) != 2 or set(protocol.get("actions", ACTIONS)) != set(ACTIONS):
        raise ValueError("freeze all four state actions and at most two joint candidates")
    if protocol.get("epsilon", .05) != .05:
        raise ValueError("the predefined video-mean LPIPS threshold is .05")
    return dev, holdout, recipes


def build_matrix(protocol, selected_shape=None, *, protocol_sha256=None,
                 initial_recipe_id=None, allowed_actions=None):
    """Describe separate calibration, cost screen, joint and holdout stages.

    Initial calibration has 3 recipes x 8 development inputs. The inexpensive
    cost screen covers at most 24 deduplicated action/suffix candidates on the first
    frozen development input only. It confers no quality qualification. Only
    the selected <=2 candidates expand to 8 development/4 holdout inputs each.
    """
    dev, holdout, recipes = _protocol(protocol)
    if initial_recipe_id is not None and initial_recipe_id not in {row["id"] for row in recipes}:
        raise ValueError("initial recipe was not frozen in this protocol")
    actions = list(ACTIONS if allowed_actions is None else allowed_actions)
    if not actions or len(set(actions)) != len(actions) or not set(actions).issubset(ACTIONS) or "keep" not in actions:
        raise ValueError("screen actions must be a unique frozen subset including keep")
    if selected_shape is not None:
        if any(type(selected_shape.get(key)) is not int or selected_shape[key] <= 0
               for key in ("height", "width", "frames", "steps")) or selected_shape["steps"] != 32:
            raise ValueError("selected shape must describe a positive 32-step generation")
    shape = copy.deepcopy(selected_shape)
    def inputs(prompts, seeds):
        return [dict(prompt_id=p["id"], prompt=p["text"], split=p["split"], seed=seed)
                for p in prompts for seed in seeds]
    dev_inputs = inputs(dev, (101, 102))
    holdout_inputs = inputs(holdout, (201, 202))
    fixed = []
    for recipe in recipes:
        for item in dev_inputs:
            fixed.append(dict(job_id="initial-%s-%s-%d" % (recipe["id"], item["prompt_id"], item["seed"]),
                stage="initial_calibration", role="reference" if recipe["id"] == "off" else "candidate",
                recipe_id=recipe["id"], prefix=_config(recipe), suffix=_config(recipe),
                action=dict(kind="none", at_step=0), shape=copy.deepcopy(shape), **item))
    by_threshold = {row["threshold"]: row for row in recipes}
    candidates = []
    for prefix in recipes:
        if initial_recipe_id is not None and prefix["id"] != initial_recipe_id:
            continue
        suffixes = []
        for suffix in (prefix, by_threshold[.04], by_threshold[0.]):
            if suffix["threshold"] > prefix["threshold"]:
                continue
            if not any(_config(previous) == _config(suffix) for previous in suffixes):
                suffixes.append(suffix)
        for action in actions:
            for suffix in suffixes:
                candidates.append(dict(candidate_id="%s-%s-%s" % (prefix["id"], action, suffix["id"]),
                    prefix_recipe_id=prefix["id"], suffix_recipe_id=suffix["id"],
                    prefix=_config(prefix), suffix=_config(suffix), action=dict(kind=action, at_step=16),
                    policy_changes=int(_config(prefix) != _config(suffix)), max_policy_changes=1))
    screen_input = dev_inputs[0]
    screen_jobs = [dict(candidate, **screen_input, shape=copy.deepcopy(shape), stage="cost_screen",
                        job_id="screen-" + candidate["candidate_id"], quality_qualified=False) for candidate in candidates]
    return seal(dict(schema_version="ditserve.research.matrix.v1", protocol_sha256=_bound_protocol_hash(protocol, protocol_sha256),
        initial_recipe_id=initial_recipe_id, allowed_actions=actions,
        selected_shape=shape, requires_selected_shape=shape is None, executes_gpu_work=False,
        development_inputs=dev_inputs, holdout_inputs=holdout_inputs, initial_jobs=fixed,
        screen_inputs=[screen_input], screen_candidates=candidates, screen_jobs=screen_jobs,
        max_joint_candidates=2, joint_validation_jobs=[], holdout_jobs=[],
        maximum_joint_development_jobs=16, maximum_joint_holdout_jobs=8,
        holdout_selection_permitted=False,
        scheduling="stage by stage under external budget; no unconditional Cartesian execution"))


def _record_input(record, prompts, seeds):
    return record.get("prompt_id") in prompts and record.get("seed") in seeds


def _reject_holdout(records, protocol=None):
    for record in records:
        if (record.get("split") != "dev" or record.get("prompt_id") in ("tram", "aquarium") or
                record.get("seed") in (201, 202)):
            raise ValueError("holdout or unspecified-split observations cannot participate in development selection")


def _identity_context(identity):
    # Inputs vary within the declared development grid; every other part of a
    # fixed-policy trajectory must describe the same measured configuration.
    return {key: value for key, value in identity.items() if key not in ("input", "prefix", "suffix", "action")}


def select_initial_recipe(records, protocol, selected_shape=None, *, protocol_sha256=None):
    """Select on all 8 dev trajectories, excluding CPU prepare variability.

    Record fields: recipe_id, prompt_id, seed, split='dev', status='completed',
    measurement_kind='measured_gpu', evidence, and phase_cost_ms containing
    prepare/denoise/decode. Full-trajectory evidence must match the recipe,
    exact prompt/seed, protocol, model, shape, backend, source and layout.
    """
    records = list(records)
    dev, _, recipes = _protocol(protocol)
    protocol_sha256 = _bound_protocol_hash(protocol, protocol_sha256)
    _reject_holdout(records)
    prompts = {row["id"]: row["text"] for row in dev}
    expected = {(prompt, seed) for prompt in prompts for seed in (101, 102)}
    grouped = defaultdict(list)
    for row in records:
        grouped[row.get("recipe_id")].append(row)
    candidates, excluded = [], []
    global_contexts = set()
    valid_groups = {}
    measured_groups = set()
    input_contexts = defaultdict(set)
    for recipe in recipes:
        rows = grouped.get(recipe["id"], [])
        keys = [(row.get("prompt_id"), row.get("seed")) for row in rows]
        reasons, losses, costs, contexts = [], [], [], set()
        if len(rows) != 8 or len(set(keys)) != len(keys) or set(keys) != expected:
            reasons.append("incomplete_or_duplicate_development_grid")
        for row in rows:
            evidence = row.get("evidence", {})
            identity = evidence.get("identity", {})
            phase = row.get("phase_cost_ms", {})
            if row.get("status") != "completed" or row.get("measurement_kind") != "measured_gpu":
                reasons.append("not_completed_gpu_measurement")
            if not all(_number(phase.get(key)) for key in ("prepare", "denoise", "decode")):
                reasons.append("missing_phase_cost")
            elif phase["denoise"] + phase["decode"] <= 0:
                reasons.append("nonpositive_mechanism_cost")
            else:
                costs.append(phase["denoise"] + phase["decode"])
            try:
                qualification = qualify_trajectory(evidence, identity, epsilon=protocol.get("epsilon", .05))
            except (ValueError, TypeError):
                qualification = dict(qualified=False)
            if qualification.get("status") == "failed":
                reasons.append("development_video_mean_lpips_exceeds_epsilon")
            elif not qualification["qualified"]:
                reasons.append("full_development_quality_or_correctness_not_qualified")
            else:
                losses.append(qualification["lpips_mean"])
            if (identity.get("model_id") != protocol.get("model_id") or identity.get("revision") != protocol.get("revision") or
                    identity.get("protocol_sha256") != protocol_sha256 or
                    identity.get("prefix") != _config(recipe) or identity.get("suffix") != _config(recipe) or
                    identity.get("action") != dict(kind="none", at_step=0)):
                reasons.append("fixed_recipe_identity_mismatch")
            inputs = identity.get("input", {})
            input_contexts[(row.get("prompt_id"), row.get("seed"))].add(canonical_hash(dict(
                input=inputs, reference_trajectory=evidence.get("reference_trajectory_sha256"),
                reference_output=evidence.get("reference_output_sha256"))))
            if (inputs.get("prompt") != prompts.get(row.get("prompt_id")) or inputs.get("seed") != row.get("seed") or
                    inputs.get("negative_prompt") != protocol.get("negative_prompt", "") or
                    not _record_input(row, prompts, (101, 102))):
                reasons.append("development_input_identity_mismatch")
            if selected_shape is not None and any(identity.get("spec", {}).get(k) != selected_shape.get(k)
                                                  for k in ("height", "width", "frames", "steps")):
                reasons.append("selected_shape_mismatch")
            if identity:
                contexts.add(canonical_hash(_identity_context(identity)))
        if len(contexts) != 1:
            reasons.append("mixed_or_missing_shape_backend_source_layout")
        if not set(reasons) - {"development_video_mean_lpips_exceeds_epsilon"}:
            measured_groups.add(recipe["id"])
            global_contexts.update(contexts)
        if reasons:
            excluded.append(dict(recipe_id=recipe["id"], reasons=sorted(set(reasons))))
        else:
            global_contexts.update(contexts)
            valid_groups[recipe["id"]] = rows
            candidates.append(dict(recipe_id=recipe["id"], recipe=_config(recipe),
                maximum_development_video_mean_lpips=max(losses), median_denoise_decode_ms=statistics.median(costs),
                development_evidence_sha256=[row["evidence"]["sha256"] for row in rows]))
    unknown = set(grouped) - {recipe["id"] for recipe in recipes}
    if unknown:
        raise ValueError("record contains an unfrozen recipe")
    if any(len(values) != 1 for values in input_contexts.values()):
        return seal(dict(schema_version="ditserve.research.initial_selection.v1", status="insufficient_evidence",
            selected=None, candidates=candidates, excluded=excluded, reason="cross_recipe_input_or_reference_mismatch",
            protocol_sha256=protocol_sha256, selection_split="dev"))
    if len(global_contexts) > 1:
        return seal(dict(schema_version="ditserve.research.initial_selection.v1", status="insufficient_evidence",
            selected=None, candidates=candidates, excluded=excluded, reason="cross_recipe_shape_backend_source_layout_mismatch",
            protocol_sha256=protocol_sha256, selection_split="dev"))
    if "off" not in valid_groups:
        return seal(dict(schema_version="ditserve.research.initial_selection.v1", status="insufficient_evidence",
            selected=None, candidates=candidates, excluded=excluded, reason="complete_cache_off_reference_grid_required",
            protocol_sha256=protocol_sha256, selection_split="dev"))
    shared_prepare = statistics.median(row["phase_cost_ms"]["prepare"] for row in valid_groups["off"])
    for candidate in candidates:
        candidate["shared_prepare_ms"] = shared_prepare
        candidate["comparable_service_ms"] = shared_prepare + candidate["median_denoise_decode_ms"]
    # Do not silently fall back to the completed subset of a predeclared grid.
    complete = len(measured_groups) == len(recipes)
    selected = min(candidates, key=lambda row: (row["median_denoise_decode_ms"], row["recipe_id"])) if complete else None
    return seal(dict(schema_version="ditserve.research.initial_selection.v1",
        status="selected" if selected else "insufficient_evidence", selected=selected, candidates=candidates, excluded=excluded,
        protocol_sha256=protocol_sha256, selection_split="dev",
        cpu_prepare_cached=any(row.get("evidence", {}).get("identity", {}).get("input", {}).get("text_encoding", {}).get("mode")
                               == "frozen_cpu_embeddings_mechanism_only" for row in records),
        selection_cost_scope="measured_denoise_plus_decode; shared_off_prepare_reported_only",
        reason=None if selected else "complete_all_predeclared_recipe_measurements_before_selection"))


def select_joint_candidates(records, *, matrix, limit=2):
    """Rank a common development-only cost screen, never qualify its quality.

    Records contain candidate_id, prompt_id, seed, split, status,
    measurement_kind, correctness_status and cost_ms (including state action,
    denoising, decode). ``context_sha256`` binds the common shape/backend/source
    and ``protocol_sha256`` binds this matrix. Missing candidate cells prevent
    selection rather than making an unmeasured candidate look slow.
    """
    verify_seal(matrix)
    if type(limit) is not int or not 1 <= limit <= min(2, matrix["max_joint_candidates"]):
        raise ValueError("joint development selection is limited to one or two candidates")
    records = list(records)
    _reject_holdout(records)
    candidates = {row["candidate_id"]: row for row in matrix["screen_candidates"]}
    expected = {(row["prompt_id"], row["seed"]) for row in matrix["screen_inputs"]}
    grouped, rejected = defaultdict(list), []
    contexts = set()
    for row in records:
        if row.get("candidate_id") not in candidates:
            raise ValueError("cost screen contains an unfrozen joint candidate")
        grouped[row["candidate_id"]].append(row)
    ranked = []
    ineligible, baselines = [], []
    for candidate_id in candidates:
        rows = grouped[candidate_id]
        keys = [(row.get("prompt_id"), row.get("seed")) for row in rows]
        reasons = []
        if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
            reasons.append("incomplete_or_duplicate_common_screen_inputs")
        explicit_ineligible = bool(rows) and all(row.get("status") == "ineligible" and
            isinstance(row.get("reason"), str) and row["reason"] and row.get("correctness_status") == "passed"
            for row in rows)
        for row in rows:
            if not explicit_ineligible and (row.get("status") != "completed" or row.get("measurement_kind") != "measured_gpu" or
                    row.get("correctness_status") != "passed" or not _number(row.get("cost_ms")) or row["cost_ms"] <= 0):
                reasons.append("missing_measured_cost_or_correctness")
            if row.get("protocol_sha256") != matrix["protocol_sha256"]:
                reasons.append("protocol_identity_mismatch")
            context = row.get("context_sha256")
            if not isinstance(context, str) or len(context) != 64 or any(c not in "0123456789abcdef" for c in context):
                reasons.append("missing_bound_screen_context")
            else:
                contexts.add(context)
        if reasons:
            rejected.append(dict(candidate_id=candidate_id, reasons=sorted(set(reasons))))
        elif explicit_ineligible:
            ineligible.append(dict(candidate_id=candidate_id, reasons=[row["reason"] for row in rows]))
        elif candidates[candidate_id]["action"]["kind"] == "keep" and candidates[candidate_id]["prefix"] == candidates[candidate_id]["suffix"]:
            baselines.append(dict(candidate_id=candidate_id, median_cost_ms=statistics.median(row["cost_ms"] for row in rows)))
        else:
            ranked.append(dict(candidate=copy.deepcopy(candidates[candidate_id]), candidate_id=candidate_id,
                median_cost_ms=statistics.median(row["cost_ms"] for row in rows),
                screen_record_sha256=[canonical_hash(row) for row in rows]))
    complete = not rejected and len(contexts) == 1 and bool(ranked)
    ranked.sort(key=lambda row: (row["median_cost_ms"], row["candidate_id"]))
    selected = ranked[:limit] if complete else []
    return seal(dict(schema_version="ditserve.research.joint_selection.v1",
        status="selected_for_validation" if complete else "insufficient_evidence", selected=selected,
        ranked=ranked, rejected=rejected, ineligible=ineligible, baselines=baselines,
        context_sha256=next(iter(contexts)) if len(contexts) == 1 else None,
        protocol_sha256=matrix["protocol_sha256"], matrix_sha256=matrix["sha256"],
        selection_split="dev", quality_qualified=False, holdout_used=False,
        reason=None if complete else "complete_common_development_screen_required",
        required_next="complete_full_trajectory_quality_on_all_eight_development_inputs_before_holdout"))


def materialize_joint_jobs(matrix, selection, *, split="dev"):
    """Expand only a frozen <=2 selection; holdout jobs cannot change selection.

    This is a scheduling description, not an execution/quality authorization.
    The external runner must satisfy development quality before holdout runs.
    """
    verify_seal(matrix)
    verify_seal(selection)
    if split not in ("dev", "holdout"):
        raise ValueError("split must be dev or holdout")
    if (selection.get("status") != "selected_for_validation" or selection.get("selection_split") != "dev" or
            selection.get("holdout_used") is not False or selection.get("matrix_sha256") != matrix["sha256"] or
            selection.get("protocol_sha256") != matrix["protocol_sha256"]):
        raise ValueError("joint jobs require this matrix's frozen development-only selection")
    selected = selection.get("selected", [])
    candidates = {row["candidate_id"]: row for row in matrix["screen_candidates"]}
    if not 1 <= len(selected) <= 2 or len({row["candidate_id"] for row in selected}) != len(selected):
        raise ValueError("one or two distinct selected candidates are required")
    inputs = matrix["development_inputs" if split == "dev" else "holdout_inputs"]
    jobs = []
    for row in selected:
        candidate = row["candidate"]
        if candidate != candidates.get(row["candidate_id"]):
            raise ValueError("selected action differs from its frozen matrix candidate")
        for item in inputs:
            jobs.append(dict(candidate, **item, shape=copy.deepcopy(matrix["selected_shape"]),
                job_id="%s-%s-%s-%d" % (split, row["candidate_id"], item["prompt_id"], item["seed"]),
                stage="joint_validation" if split == "dev" else "holdout_confirmation",
                selection_sha256=selection["sha256"], selection_permitted=split == "dev",
                requires_development_quality=split == "holdout"))
    return jobs
