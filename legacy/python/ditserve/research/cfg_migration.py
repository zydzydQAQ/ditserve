"""Conditional full-trajectory migration and real-queue mechanisms.

These functions use an already budgeted ``CfgExperiment``. They never load a
worker, create a ledger, or manufacture waiting time. Eligibility can therefore
be checked by the supervisor before allocating either GPU.
"""

import math
import statistics

from .cfg_matrix import (MIGRATION_ACTIONS, holdout_gate, mechanism_gate,
                         migration_quality_gate, select_candidate)
from .common import digest, read_json
from .dual_evidence import read_records
from .dual_matrix import bound_quality, inputs, seal


def _rows(experiment):
    return read_records(experiment.root, experiment.frozen)


def eligibility(experiment):
    """Recompute current evidence and enforce the pre-holdout frozen choice."""
    rows = _rows(experiment)
    selection = select_candidate(rows, experiment.config, experiment.context(), frozen=experiment.frozen)
    path = experiment.root / "selection.json"
    frozen = read_json(path) if path.exists() else {}
    valid = (frozen.get("sha256") == digest({k: v for k, v in frozen.items() if k != "sha256"})
             and frozen.get("protocol_sha256") == experiment.frozen["sha256"]
             and frozen.get("source_sha256") == experiment.frozen["source_sha256"]
             and frozen.get("context_sha256") == experiment.context()
             and frozen.get("status") == "selected"
             and frozen.get("selected") == selection.get("selected")
             and selection.get("selected") is not None)
    if not valid:
        return seal(dict(eligible=False, status="unverified", selected=None, selection=selection,
                         reason="current development evidence and frozen pre-holdout selection required"))
    selected = selection["selected"]
    held = holdout_gate(rows, experiment.config, selected, experiment.context())
    return seal(dict(eligible=held["eligible"], status=held["status"], selected=selected,
                     selection=selection, holdout=held,
                     reason=None if held["eligible"] else "selected policy did not pass complete held-out quality"))


def _use_selected(experiment, selected):
    if selected == "paired_008":
        experiment.implementations[selected] = "split_prefix"
    policy = experiment.policy(selected)
    if policy["implementation"] == "oracle":
        raise ValueError("reference prototype cannot qualify migration performance")


def _estimate(experiment, *, a100=0, v100=0, queued=0):
    return ((a100 + queued) * experiment.costs["a100"] + v100 * experiment.costs["v100"]) * 1.3 + 10.


def _actions_gates(experiment, selected):
    rows = _rows(experiment)
    return {action: {split: migration_quality_gate(rows, experiment.config, split, selected,
                                                  action, experiment.context())
                     for split in ("dev", "holdout")} for action in MIGRATION_ACTIONS}


def migration(experiment):
    gate = eligibility(experiment)
    if not gate["eligible"]:
        return experiment.write("migration", dict(status="direction_stopped", eligibility=gate,
            qualified_actions=[], request_work_started=False, reason=gate["reason"]))
    selected = gate["selected"]
    _use_selected(experiment, selected)
    declared = [(x["prompt_id"], x["seed"]) for x in experiment.config["migration_screening_inputs"]]
    development = inputs(experiment.config)
    screening = [next(item for item in development if (item["prompt_id"], item["seed"]) == key)
                 for key in declared]
    for item in screening:
        with experiment.unit("migration-screen-" + item["prompt_id"],
                             _estimate(experiment, a100=2, v100=3)):
            for action in ("stay", *MIGRATION_ACTIONS):
                run = experiment.generate(item, selected, role="v100", action=action)
                experiment.record("cfg_migration_screen", action + "-" + item["prompt_id"],
                    status="completed", recipe_id=selected, action=action, split="dev",
                    prompt_id=item["prompt_id"], seed=item["seed"], run=run, timing_eligible=True)

    # Each input is one complete unit, with the shared references freshly run
    # in that unit. Neither action inherits the other's trajectory qualification.
    for split in ("dev", "holdout"):
        for item in inputs(experiment.config, split):
            case = "%s-%s-%s" % (split, item["prompt_id"], item["seed"])
            with experiment.unit("migration-quality-" + case,
                                 _estimate(experiment, a100=4, v100=3)):
                common = experiment.generate(item, "off", role="a100")
                path = experiment.generate(item, "off", role="v100", action="full_migrate")
                path_quality = experiment.quality(path, common)
                if not bound_quality(path_quality, path, common):
                    experiment.record("cfg_migration_reference", case, status="failed", split=split,
                        prompt_id=item["prompt_id"], seed=item["seed"], common_reference=common,
                        path_reference=path, path_reference_quality=path_quality, timing_eligible=False)
                    return experiment.write("migration", dict(status="quality_stopped", selected=selected,
                        qualified_actions=[], reason="cache-off device-path reference failed common reference",
                        failed_reference_case=case, remaining_matrix="unverified"))
                for action in MIGRATION_ACTIONS:
                    candidate = experiment.generate(item, selected, role="v100", action=action)
                    experiment.record("cfg_migration_quality", case + "-" + action,
                        status="completed", split=split, prompt_id=item["prompt_id"], seed=item["seed"],
                        recipe_id=selected, action=action, candidate=candidate,
                        common_reference=common, path_reference=path, path_reference_quality=path_quality,
                        canonical_quality=experiment.quality(candidate, common),
                        path_quality=experiment.quality(candidate, path), timing_eligible=False)
    gates = _actions_gates(experiment, selected)
    qualified = [action for action in MIGRATION_ACTIONS
                 if all(value["eligible"] for value in gates[action].values())]
    return experiment.write("migration", dict(status="completed" if qualified else "quality_stopped",
        selected=selected, policy_sha256=experiment.policy(selected)["sha256"], actions=gates,
        qualified_actions=qualified, quality_error_composition="none",
        reason=None if qualified else "neither migration action passed both complete quality splits"))


def _queue_quality(experiment, run, reference, expected_length):
    queued = run.get("queued_outputs", [])
    if len(queued) != expected_length or run.get("all_requests_decoded") is not True:
        return False
    generation_ids = [output.get("generation_id") for output in queued]
    if (any(not isinstance(value, str) or not value for value in generation_ids)
            or len(set(generation_ids)) != len(generation_ids)
            or any(value in (reference.get("generation_id"), run.get("generation_id"))
                   for value in generation_ids)):
        return False
    valid = True
    for output in queued:
        identity = output.get("identity", {})
        # Batch execution must emit its own known input/policy identity. It may
        # not be supplied by copying a reference after execution.
        if identity != reference["identity"] or output.get("status") != "completed":
            valid = False
            continue
        quality = experiment.quality(output, reference)
        output["quality"] = quality
        valid &= bound_quality(quality, output, reference)
    return valid


def _screen_costs(experiment, selected, actions):
    expected = {(x["prompt_id"], x["seed"]) for x in experiment.config["migration_screening_inputs"]}
    rows = [r for r in _rows(experiment) if r.get("type") == "cfg_migration_screen"
            and r.get("recipe_id") == selected and r.get("action") in actions]
    if len(rows) != len(expected) * len(actions):
        return None
    means = {}
    for action in actions:
        group = [r for r in rows if r["action"] == action]
        if ({(r.get("prompt_id"), r.get("seed")) for r in group} != expected
                or len(group) != len(expected)
                or any(r.get("performance_valid") is not True or r.get("status") != "completed"
                       or r.get("context_sha256") != experiment.context() for r in group)):
            return None
        waits = [r.get("run", {}).get("target_wait_ms", 0.) for r in group]
        if any(type(wait) not in (int, float) or not math.isfinite(wait) or wait < 0 for wait in waits):
            return None
        values = [r.get("run", {}).get("critical_path_ms") for r in group]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
            return None
        if any(r["run"].get("queued_outputs") for r in group):
            return None
        without_wait = [value - wait for value, wait in zip(values, waits)]
        if min(without_wait) <= 0:
            return None
        means[action] = statistics.mean(without_wait)
    return means


def mechanism(experiment):
    gate = eligibility(experiment)
    if not gate["eligible"]:
        return experiment.write("mechanism", dict(status="unverified", reason=gate["reason"],
            eligibility=gate, request_work_started=False))
    selected = gate["selected"]
    _use_selected(experiment, selected)
    quality_gates = _actions_gates(experiment, selected)
    qualified = {action: all(g["eligible"] for g in quality_gates[action].values())
                 for action in MIGRATION_ACTIONS}
    actions = ["stay"] + [action for action in MIGRATION_ACTIONS if qualified[action]]
    if len(actions) == 1:
        return experiment.write("mechanism", dict(status="unverified", selected=selected,
            reason="no migration action passed complete development and held-out quality",
            actions=quality_gates, request_work_started=False))
    means = _screen_costs(experiment, selected, actions)
    if means is None:
        return experiment.write("mechanism", dict(status="unverified", selected=selected,
            reason="complete current-source development costs required", request_work_started=False))
    break_evens = {action: means["stay"] - means[action] for action in actions if action != "stay"}
    break_even = max(break_evens.values())
    probe_action = min(actions[1:], key=lambda action: means[action])
    item = inputs(experiment.config)[0]  # also the declared input of every queued off request
    queue_length = None
    probes = []
    for length in experiment.config["queue_lengths"] if break_even > 0 else ():
        with experiment.unit("queue-%s" % length,
                             _estimate(experiment, a100=3, v100=2, queued=length)):
            common = experiment.generate(item, "off", role="a100")
            path = experiment.generate(item, "off", role="v100", action="full_migrate")
            candidate = experiment.generate(item, selected, role="v100", action=probe_action, queue=length)
            path_quality = experiment.quality(path, common)
            canonical = experiment.quality(candidate, common)
            path_metric = experiment.quality(candidate, path)
            quality = (bound_quality(path_quality, path, common)
                       and bound_quality(canonical, candidate, common)
                       and bound_quality(path_metric, candidate, path)
                       and _queue_quality(experiment, candidate, common, length))
            wait = candidate.get("target_wait_ms", 0.)
            coverage = type(wait) in (int, float) and math.isfinite(wait) and wait >= 1.25 * break_even
            row = experiment.record("cfg_queue_probe", str(length), status="completed", selected=selected,
                recipe_id=selected, action=probe_action, queue_length=length, run=candidate,
                common_reference=common, path_reference=path, path_reference_quality=path_quality,
                canonical_quality=canonical, path_quality=path_metric, quality_passed=bool(quality),
                queue_wait_ms=wait, break_even_wait_ms=break_even, queue_coverage=bool(coverage),
                real_queue=True, all_requests_decoded=candidate.get("all_requests_decoded") is True,
                timing_eligible=True)
            probes.append(row)
        sealed = [r for r in _rows(experiment) if r.get("type") == "cfg_queue_probe"
                  and r.get("case_id") == str(length)]
        if quality and coverage and len(sealed) == 1 and sealed[0].get("performance_valid") is True:
            queue_length = length
            break
    for condition, length in [("idle", 0)] + ([("busy", queue_length)] if queue_length is not None else []):
        for repetition, order in enumerate((actions, list(reversed(actions)))):
            with experiment.unit("mechanism-%s-%s" % (condition, repetition),
                                 _estimate(experiment, a100=2 + len(actions), v100=2 + len(actions),
                                           queued=length * len(actions))):
                common = experiment.generate(item, "off", role="a100")
                path = experiment.generate(item, "off", role="v100", action="full_migrate")
                stay_reference = experiment.generate(item, "off", role="v100")
                path_quality = experiment.quality(path, common)
                runs, qualities = {}, {}
                quality = bound_quality(path_quality, path, common)
                for action in order:
                    run = experiment.generate(item, selected, role="v100", action=action, queue=length)
                    runs[action] = run
                    if action == "stay":
                        metric = experiment.quality(run, stay_reference)
                        qualities[action] = dict(same_device=metric)
                        quality &= bound_quality(metric, run, stay_reference)
                    else:
                        canonical = experiment.quality(run, common)
                        path_metric = experiment.quality(run, path)
                        qualities[action] = dict(canonical=canonical, path=path_metric)
                        quality &= bound_quality(canonical, run, common) and bound_quality(path_metric, run, path)
                    quality &= _queue_quality(experiment, run, common, length)
                waits = [runs[action]["target_wait_ms"] for action in actions if action != "stay"]
                coverage = condition == "idle" or min(waits) >= 1.25 * break_even
                experiment.record("cfg_mechanism", "%s-%s" % (condition, repetition), status="completed",
                    condition=condition, recipe_id=selected, selected=selected, order=list(order),
                    queue_length=length, queue_wait_ms=min(waits), break_even_wait_ms=break_even,
                    queue_coverage=coverage, real_queue=condition == "busy", quality_passed=bool(quality),
                    quality=qualities, common_reference=common, path_reference=path,
                    stay_reference=stay_reference, path_reference_quality=path_quality, runs=runs,
                    all_requests_decoded=all(run.get("all_requests_decoded") is True for run in runs.values()),
                    costs_ms={action: run["critical_path_ms"] for action, run in runs.items()},
                    cost_scope="all_requests_decode_complete", timing_eligible=True,
                    simple_baseline="average_remaining_cost_plus_full_migration")
    pairs = [r for r in _rows(experiment) if r.get("type") == "cfg_mechanism"
             and r.get("recipe_id") == selected]
    gates = {condition: mechanism_gate([r for r in pairs if r.get("condition") == condition],
              condition=condition, context_sha256=experiment.context(), quality_passed=qualified)
             for condition in ("idle", "busy")}
    return experiment.write("mechanism", dict(status="completed", selected=selected,
        qualified_actions=actions[1:], actions=quality_gates, gates=gates, queue_length=queue_length,
        queue_probes=len(probes), development_remaining_costs_ms=means,
        break_even_wait_ms=break_even, per_action_break_even_wait_ms=break_evens,
        busy_status="measured" if queue_length is not None else "unverified_real_queue_coverage",
        innovation_status="not_established; assess action crossover against simple full-migration baseline"))
