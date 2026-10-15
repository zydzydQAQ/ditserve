"""CPU-only reports for sealed CFG policy, quality and migration evidence."""

from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics

from .common import atomic_json, digest, read_json
from .cfg_matrix import (CANDIDATES, MIGRATION_ACTIONS, _quality_status, _timing_eligible,
                         holdout_gate, mechanism_gate, migration_quality_gate,
                         optimistic_cost, select_candidate)
from .cfg_protocol import verify_protocol
from .dual_evidence import read_records
from .dual_report import budget_summary
from .retention import _artifacts, idle_guard, verify_reference


def _decision(status, reason, **details):
    return dict(status=status, reason=reason, **details)


def _number(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def evaluate_hypotheses(rows, config, *, frozen=None):
    """Recompute qualification; recorded policy selection is not proof itself."""
    contexts = {r.get("context_sha256") for r in rows if str(r.get("type", "")).startswith("cfg_")}
    context = next(iter(contexts)) if len(contexts) == 1 and None not in contexts else None
    unavailable = _decision("insufficient_evidence", "complete evidence for one frozen configuration is missing")
    if context is None:
        return dict(cache=unavailable, migration=unavailable, selection=None, holdout=None,
                    candidates={}, migration_actions={}, mechanism={},
                    cache_joint_control=_decision("insufficient_evidence", "no comparison against a simple full-migration controller"),
                    memory=_decision("insufficient_evidence", "state costs do not establish real admission improvement"),
                    sp=_decision("deferred", "two heterogeneous SP1 workers only"),
                    lp=_decision("deferred", "Latent Parallelism is outside this pre-experiment"))
    selection = select_candidate(rows, config, context, **({"frozen": frozen} if frozen is not None else {}))
    candidates = {name: dict(stage_a=selection["stage_a"][name],
                            development=selection["development"].get(name)) for name in CANDIDATES}
    selected = selection["selected"]
    holdout = holdout_gate(rows, config, selected, context) if selected else None
    if selected and holdout["eligible"]:
        cache = _decision("supported", "every development/held-out input passed quality; both orders on both cards met 5%",
                          selected=selected, scope="finite frozen-text mechanism, not online goodput or per-request quality assurance")
    elif selected and holdout["status"] == "failed":
        cache = _decision("falsified", "selected policy failed held-out quality; no reselection", selected=selected)
    elif selection["status"] == "direction_stopped":
        cache = _decision("falsified", "neither declared policy passed all required screening/development gates")
    else:
        cache = _decision("insufficient_evidence", "screening, both formal orders, or fixed-policy holdout is incomplete", selected=selected)
    actions = {}
    for action in MIGRATION_ACTIONS:
        gates = {split: migration_quality_gate(rows, config, split, selected, action, context)
                 for split in ("dev", "holdout")} if selected else {}
        actions[action] = dict(quality=gates, eligible=bool(holdout and holdout["eligible"] and gates
                                and all(g["eligible"] for g in gates.values())))
    mechanisms = {}
    for condition in ("idle", "busy"):
        pairs = [r for r in rows if r.get("type") == "cfg_mechanism" and r.get("condition") == condition
                 and r.get("recipe_id") == selected]
        mechanisms[condition] = mechanism_gate(pairs, condition=condition, context_sha256=context,
                                             quality_passed={a: v["eligible"] for a, v in actions.items()})
    if any(g["supported"] for g in mechanisms.values()):
        migration = _decision("supported", "dual-reference-qualified migration met 5% full decoded-request cost improvement in both orders",
                              scope="ordinary heterogeneous migration engineering evidence")
    elif all(g["status"] == "not_observed" for g in mechanisms.values()):
        migration = _decision("falsified", "measured idle and real-queue comparisons did not meet the gain threshold")
    elif any(g["status"] == "failed" for a in actions.values() for g in a["quality"].values()):
        migration = _decision("falsified", "a declared migration action failed full-trajectory dual-reference quality",
                              scope="consult per-action gates; failed actions are excluded independently")
    else:
        migration = _decision("insufficient_evidence", "complete migration quality or reversed full-cost comparisons are missing")
    return dict(cache=cache, migration=migration, selection=selection, holdout=holdout, candidates=candidates,
                migration_actions=actions, mechanism=mechanisms,
                cache_joint_control=_decision("insufficient_evidence", "a simple remaining-cost/full-migration baseline may explain all observed gain; no joint-control claim"),
                memory=_decision("insufficient_evidence", "state and peak-allocation measurements alone do not prove real admission improvement"),
                sp=_decision("deferred", "two heterogeneous SP1 workers only"),
                lp=_decision("deferred", "Latent Parallelism is outside this pre-experiment"))


def _runs(value):
    if isinstance(value, dict):
        if value.get("output_sha256") and isinstance(value.get("identity"), dict):
            yield value
        for child in value.values():
            yield from _runs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _runs(child)


def _step_events(run):
    return [event for event in run.get("events", []) if isinstance(event, dict) and event.get("kind") == "step"]


def _sum_observed(events, key):
    values = [event.get("cfg_research", {}).get(key) for event in events]
    return sum(values) if values and all(_number(value) for value in values) else None


def _split_probe_cost(events):
    values = [event.get("cfg_research", {}).get("split_prefix", {}).get("gate_probe_ms") for event in events]
    return sum(values) if values and all(_number(value) for value in values) else None


def descriptive_inputs(frozen, rows):
    """Import only verified parent prototype metrics for labelled descriptions.

    The returned rows must never be sent to development or migration gates.
    Parent raw arrays are neither loaded nor newly re-evaluated by this helper.
    """
    def annotate(values, origin, protocol, source, root=None):
        return [dict(row, descriptive_source=dict(origin=origin, protocol_sha256=protocol,
                     source_sha256=source, context_sha256=row.get("context_sha256"), run=root),
                     **({"timing_eligible": False} if origin == "parent_prototype" else {})) for row in values]
    result = annotate(rows, "current_run", frozen.get("sha256"), frozen.get("source_sha256"))
    proof = frozen.get("prototype_continuation")
    if not proof:
        return result
    from .cfg_continuation import verify_continuation
    verified = verify_continuation(proof, config=frozen["config"], ledger=frozen["ledger"])
    parent_root = Path(verified["parent_run"])
    parent = read_json(parent_root / "protocol.json")
    if (parent.get("sha256") != verified["parent_protocol_sha256"]
            or parent.get("source_sha256") != verified["parent_source_sha256"]):
        raise ValueError("descriptive parent differs from the verified continuation")
    parent_rows = [row for row in read_records(parent_root, parent)
                   if row.get("type") in ("cfg_screen", "cfg_diagnostic")
                   and row.get("context_sha256") == verified["parent_context_sha256"]]
    return result + annotate(parent_rows, "parent_prototype", parent["sha256"], parent["source_sha256"], str(parent_root))


def _source(row):
    return row.get("descriptive_source", dict(origin="current_run", protocol_sha256=row.get("protocol_sha256"),
        source_sha256=row.get("source_sha256"), context_sha256=row.get("context_sha256"), run=None))


def cost_memory_summary(rows):
    """Descriptive prototype costs and formal observations stay separate.

    Snapshot/discarded-trial costs form a disjoint decomposition only when the
    complete recorded oracle bound verifies. Accepted replay is useful compute
    in the final path and is never subtracted a second time.
    """
    samples, seen = [], set()
    for row in rows:
        if row.get("performance_valid") is not True:
            continue
        formal = _timing_eligible(row)
        screen = row.get("type") == "cfg_screen" and row.get("status") == "completed"
        if not formal and not screen:
            continue
        for run in _runs(row):
            generation = run.get("generation_id")
            key = (digest(_source(row)), generation)
            if not generation or key in seen or not _number(run.get("elapsed_ms"), True) or run.get("diagnostic", False):
                continue
            seen.add(key)
            implementation = run["identity"].get("implementation", "unknown")
            category = "formal_observation" if formal else "oracle_screen" if implementation == "oracle" else "native_screen"
            events = _step_events(run)
            estimated = optimistic_cost(run) if implementation == "oracle" else None
            bound = run.get("optimistic_bound", {}) if estimated is not None else {}
            sample = dict(case_id=row.get("case_id"), generation_id=generation, evidence_sha256=row.get("sha256"),
                source=_source(row),
                context_sha256=row.get("context_sha256"), device_role=run.get("device_role", row.get("device_role")),
                recipe_id=run.get("recipe_id"), implementation=implementation, category=category,
                elapsed_ms=run["elapsed_ms"], formal_cost_eligible=formal, performance_qualified=False,
                snapshot_ms=bound.get("snapshot_ms") if bound else _sum_observed(events, "snapshot_ms"),
                discarded_trial_ms=bound.get("discarded_trial_ms"),
                optimistic_ms=estimated, accepted_replay_ms=_sum_observed(events, "replay_ms"),
                gate_probe_ms=_split_probe_cost(events),
                allocated_peak_bytes=run.get("peak_memory_bytes", {}).get("allocated"),
                reserved_peak_bytes=run.get("peak_memory_bytes", {}).get("reserved"))
            for name in ("allocated_peak_bytes", "reserved_peak_bytes"):
                if not _number(sample[name]):
                    sample[name] = None
            samples.append(sample)
    grouped = defaultdict(list)
    for sample in samples:
        grouped[(digest(sample["source"]), sample["category"], sample["device_role"], sample["recipe_id"], sample["implementation"])].append(sample)
    groups = []
    for (_, category, role, recipe, implementation), group in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        result = dict(category=category, device_role=role, recipe_id=recipe, implementation=implementation,
                      source=group[0]["source"],
                      count=len(group), formal_cost_eligible=category == "formal_observation", performance_qualified=False)
        for metric in ("elapsed_ms", "snapshot_ms", "discarded_trial_ms", "optimistic_ms", "accepted_replay_ms",
                       "gate_probe_ms", "allocated_peak_bytes", "reserved_peak_bytes"):
            values = [s[metric] for s in group if _number(s[metric])]
            result[metric] = dict(observed=len(values), mean=statistics.mean(values) if values else None,
                                  max=max(values) if values else None)
        groups.append(result)
    return dict(samples=samples, groups=groups,
                formal_samples=sum(s["formal_cost_eligible"] for s in samples),
                prototype_samples=sum(not s["formal_cost_eligible"] for s in samples),
                scope="prototype bars are descriptive measurements; oracle optimistic bounds are not executed optimized serving costs",
                memory_scope="allocator peaks include weights/workspace; no releasable-cache or admission inference",
                missing_semantics="unrecorded overhead or peak is null, never imputed zero")


def _pair_path(event, key="native_attempt", reuse_key="native_reuse"):
    gates = event.get("cfg_research", {}).get(key, {}).get("gates", [])
    branches = {g.get("branch"): g.get(reuse_key) for g in gates if isinstance(g, dict)}
    if set(branches) == {"cond", "uncond"} and all(type(x) is bool for x in branches.values()):
        return "".join("R" if branches[name] else "C" for name in ("cond", "uncond"))
    return None


def asymmetry_evidence(rows, config):
    """Describe associations, preserving exact input, device and reference pairs."""
    from .cfg_policy import resolve_policy
    from .dual_matrix import inputs
    expected = {(x["prompt_id"], x["seed"], x["split"]): x for split in ("dev", "holdout") for x in inputs(config, split)}
    diagnostics, paired = [], defaultdict(list)
    ignored = Counter()
    def validated(row, recipe):
        run, reference = row.get("candidate", {}), row.get("reference", {})
        identity, ref_identity = run.get("identity", {}), reference.get("identity", {})
        item = expected.get((row.get("prompt_id"), row.get("seed"), row.get("split")))
        implementation = identity.get("implementation")
        if (row.get("performance_valid") is not True or row.get("status") != "completed" or item is None
                or _quality_status(row.get("quality", {}), run, reference) == "unverified"
                or identity.get("recipe") != recipe or ref_identity.get("recipe") != "off"
                or identity.get("input") != item or ref_identity.get("input") != item
                or identity.get("context_sha256") != row.get("context_sha256")
                or ref_identity.get("context_sha256") != row.get("context_sha256")
                or identity.get("path") != [row.get("device_role")]
                or ref_identity.get("path") != identity.get("path")
                or identity.get("attention_path") != ref_identity.get("attention_path")
                or implementation not in (("native",) if recipe == "independent_008" else ("oracle", "split_prefix"))):
            return None
        policy, off = resolve_policy(config, recipe, implementation), resolve_policy(config, "off")
        if identity.get("policy_sha256") != policy["sha256"] or ref_identity.get("policy_sha256") != off["sha256"]:
            return None
        key = (row["context_sha256"], row["device_role"], digest(item),
               reference.get("output_sha256"), digest(ref_identity))
        return key, run
    for row in rows:
        if row.get("type") == "cfg_screen" and row.get("recipe_id") == "paired_008":
            value = validated(row, "paired_008")
            if value:
                paired[value[0]].append(row)
            else:
                ignored["unbound_paired_quality"] += 1
    for row in rows:
        if row.get("type") != "cfg_diagnostic" or row.get("recipe_id") != "independent_008":
            continue
        value = validated(row, "independent_008")
        if value is None:
            ignored["unbound_independent_diagnostic"] += 1
            continue
        key, run = value
        paths, accepted, errors = Counter(), Counter(), defaultdict(list)
        per_step = []
        for event in _step_events(run):
            native = _pair_path(event)
            final = _pair_path(event, "accepted_pair", "accepted_reuse")
            if native:
                paths[native] += 1
            if final:
                accepted[final] += 1
            metric = event.get("cfg_research", {}).get("same_input_prediction_error") or {}
            guided = metric.get("cfg", {}).get("relative_l2")
            if native and _number(guided):
                errors[native].append(guided)
                per_step.append(dict(step=event.get("completed_steps"), native_path=native, guided_relative_l2=guided))
        mixed = errors["CR"] + errors["RC"]
        symmetric = errors["RR"]
        sample = dict(case_id=row.get("case_id"), evidence_sha256=row.get("sha256"), context_sha256=row["context_sha256"],
                      source=_source(row),
                      device_role=row["device_role"], prompt_id=row["prompt_id"], seed=row["seed"],
                      native_path_counts=dict(paths), accepted_path_counts=dict(accepted),
                      observed_cfg_steps=sum(paths.values()), expected_cfg_steps=config.get("steps", 32),
                      mixed_steps=paths["CR"] + paths["RC"] if paths else None, guided_error_samples=per_step,
                      mean_mixed_guided_relative_l2=statistics.mean(mixed) if mixed else None,
                      mean_rr_guided_relative_l2=statistics.mean(symmetric) if symmetric else None,
                      independent_lpips_mean=row["quality"]["lpips_mean"], paired_comparison=None)
        matches = paired.get(key, [])
        if len(matches) == 1:
            match = matches[0]
            sample["paired_comparison"] = dict(evidence_sha256=match.get("sha256"),
                independent_output_sha256=run["output_sha256"], paired_output_sha256=match["candidate"]["output_sha256"],
                reference_output_sha256=row["reference"]["output_sha256"],
                paired_lpips_mean=match["quality"]["lpips_mean"],
                lpips_reduction=row["quality"]["lpips_mean"] - match["quality"]["lpips_mean"],
                paired_implementation=match["candidate"]["identity"]["implementation"], performance_qualified=False)
        elif matches:
            ignored["ambiguous_paired_quality"] += 1
        diagnostics.append(sample)
    within = [s for s in diagnostics if s["mean_mixed_guided_relative_l2"] is not None and s["mean_rr_guided_relative_l2"] is not None]
    comparisons = [s for s in diagnostics if s["paired_comparison"] is not None]
    return dict(status="descriptive_evidence" if diagnostics else "insufficient_evidence", samples=diagnostics,
                independent_trajectories=len(diagnostics), paired_quality_comparisons=len(comparisons),
                trajectories_with_mixed_steps=sum(_number(s["mixed_steps"], True) for s in diagnostics),
                mixed_vs_rr_comparable_trajectories=len(within),
                mixed_error_higher_trajectories=sum(s["mean_mixed_guided_relative_l2"] > s["mean_rr_guided_relative_l2"] for s in within),
                paired_quality_improved_inputs=sum(s["paired_comparison"]["lpips_reduction"] > 0 for s in comparisons),
                ignored=dict(ignored), matched_cost_control=False, causal_identification=False, performance_qualified=False,
                scope="association only; steps within a trajectory are dependent, and no matched-cost control isolates a causal CFG effect",
                missing_semantics="same-latent shadow errors are measured only on sampled reuse steps; unsampled CC steps are not zero error")
def _diagnostic_steps(row):
    """Accept explicit diagnostic samples only, never infer error from LPIPS."""
    if row.get("type") != "cfg_diagnostic" or row.get("performance_valid") is not True:
        return []
    result = [sample for sample in row.get("diagnostic_steps", []) if isinstance(sample, dict)]
    def visit(value):
        if isinstance(value, dict):
            research = value.get("cfg_research", {})
            errors = research.get("same_input_prediction_error") if isinstance(research, dict) else None
            if isinstance(errors, dict):
                step = value.get("completed_steps")
                if step is None and type(value.get("step_index")) is int:
                    step = value["step_index"] + 1
                result.append(dict(step=step, branch_errors={label: errors.get(key, {}).get("relative_l2")
                    for key, label in (("cond", "conditional"), ("uncond", "unconditional"), ("cfg", "guided"))}))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(row)
    return result


def make_figures(root, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        return dict(status="unavailable", reason=str(error), paths=[])
    root = Path(root)
    directory = root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    figures = []
    source_records = {digest(_source(row)): _source(row) for row in rows}
    def save(name, title, samples, draw, scope):
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        ax.set_title(title)
        if samples:
            draw(ax)
        else:
            ax.text(.5, .5, "Insufficient completed-unit evidence\nMissing data are not zero error or zero gain",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        path = directory / (name + ".svg")
        parent_sources = [value for value in source_records.values() if value["origin"] == "parent_prototype"]
        if parent_sources:
            fig.supxlabel(" | ".join("Parent prototype: protocol %s; source %s; context %s (descriptive only)" %
                (str(s["protocol_sha256"])[:10], str(s["source_sha256"])[:10], str(s["context_sha256"])[:10])
                for s in parent_sources), fontsize=6)
        fig.savefig(path)
        plt.close(fig)
        figures.append(dict(path=str(path.relative_to(root)), measured=bool(samples), scope=scope,
                            sources=[source_records[key] for key in sorted(source_records)]))
    diagnostics = []
    for row in rows:
        for sample in _diagnostic_steps(row):
            errors = sample.get("branch_errors", {})
            for branch in ("conditional", "unconditional", "guided"):
                if _number(errors.get(branch)) and _number(sample.get("step")):
                    diagnostics.append(((_source(row)["origin"] + "/" + row.get("case_id", "diagnostic")), sample["step"], branch, errors[branch]))
    def draw_errors(ax):
        for case, branch in sorted({(p[0], p[2]) for p in diagnostics}):
            points = sorted((p[1], p[3]) for p in diagnostics if p[0] == case and p[2] == branch)
            ax.plot([p[0] for p in points], [p[1] for p in points], ".-", label=case + "/" + branch)
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("Recorded branch error (diagnostic metric)")
        ax.legend(fontsize=6)
    save("cfg-branch-error", "CFG branch errors (diagnostic, no performance qualification)", diagnostics,
         draw_errors, "recorded shadow errors only; not LPIPS or a performance measurement")
    quality = []
    for row in rows:
        if row.get("performance_valid") is not True or row.get("type") not in ("cfg_screen", "cfg_calibration"):
            continue
        cost = row.get("candidate", {}).get("elapsed_ms")
        loss = row.get("quality", {}).get("lpips_mean")
        if _number(cost, True) and _number(loss):
            phase = row.get("phase", "unknown")
            quality.append((_source(row)["origin"] + "/" + row.get("recipe_id", "unknown") + "/" + phase, cost / 1000, loss))
    def draw_quality(ax):
        for label in sorted({p[0] for p in quality}):
            group = [p for p in quality if p[0] == label]
            ax.scatter([p[1] for p in group], [p[2] for p in group], label=label)
        ax.axhline(.05, color="gray", linestyle="--")
        ax.set_xlabel("Recorded full trajectory seconds (screen oracle includes overhead)")
        ax.set_ylabel("Whole-video mean LPIPS")
        ax.legend(fontsize=7)
    save("quality-time", "Complete trajectory quality and time", quality, draw_quality,
         "screen/oracle timings are descriptive; only uninstrumented formal pairs qualify performance")
    costs = cost_memory_summary(rows)
    def draw_cost_memory(ax):
        fig = ax.figure
        ax.remove()
        fig.set_size_inches(10, 10)
        formal_ax, prototype_ax, memory_ax = fig.subplots(3, 1)
        formal = [g for g in costs["groups"] if g["category"] == "formal_observation"]
        prototype = [g for g in costs["groups"] if g["category"] != "formal_observation"]
        def label(group):
            return ("parent/" if group["source"]["origin"] == "parent_prototype" else "") + str(group["device_role"]) + "/" + str(group["recipe_id"])
        formal_ax.set_title("Formal uninstrumented observations (qualification is checked separately)")
        if formal:
            formal_ax.bar([label(g) for g in formal], [g["elapsed_ms"]["mean"] / 1000 for g in formal])
            formal_ax.set_ylabel("Full trajectory seconds")
            formal_ax.tick_params(axis="x", labelsize=7)
            overheads = [(metric, [(i, g[metric]["mean"] / 1000) for i, g in enumerate(formal) if g[metric]["observed"]])
                         for metric in ("gate_probe_ms", "snapshot_ms")]
            if any(points for _, points in overheads):
                other = formal_ax.twinx()
                for metric, points in overheads:
                    if points:
                        other.plot([p[0] for p in points], [p[1] for p in points],
                                   "ro" if metric == "gate_probe_ms" else "g^", label=metric)
                other.set_ylabel("Coordination seconds (included, not subtracted)", fontsize=7)
                other.legend(fontsize=6, loc="upper right")
        else:
            formal_ax.text(.5, .5, "Formal performance data missing", ha="center", va="center", transform=formal_ax.transAxes)
            formal_ax.set_xticks([])
            formal_ax.set_yticks([])
        prototype_ax.set_title("Prototype screening: descriptive costs only; no performance qualification")
        if prototype:
            labels = [label(g) + "/" + g["implementation"] for g in prototype]
            base, snapshots, trials = [], [], []
            for group in prototype:
                decomposed = (group["snapshot_ms"]["observed"] == group["count"]
                              and group["discarded_trial_ms"]["observed"] == group["count"])
                snap = group["snapshot_ms"]["mean"] / 1000 if decomposed else 0.
                trial = group["discarded_trial_ms"]["mean"] / 1000 if decomposed else 0.
                base.append(group["elapsed_ms"]["mean"] / 1000 - snap - trial)
                snapshots.append(snap)
                trials.append(trial)
            prototype_ax.bar(labels, base, label="Other recorded time (not optimized execution)")
            prototype_ax.bar(labels, snapshots, bottom=base, label="Measured snapshots")
            prototype_ax.bar(labels, trials, bottom=[a + b for a, b in zip(base, snapshots)], label="Measured discarded trials / restore")
            prototype_ax.set_ylabel("Actual full trajectory seconds")
            prototype_ax.tick_params(axis="x", labelsize=6, rotation=15)
            prototype_ax.legend(fontsize=7)
            prototype_ax.text(.01, .98, "Unrecorded overhead stays in other time; it is not assumed zero",
                              va="top", transform=prototype_ax.transAxes, fontsize=6)
        else:
            prototype_ax.text(.5, .5, "Prototype observations missing", ha="center", va="center", transform=prototype_ax.transAxes)
            prototype_ax.set_xticks([])
            prototype_ax.set_yticks([])
        memory = [g for g in costs["groups"] if g["allocated_peak_bytes"]["observed"]]
        memory_ax.set_title("Allocator peaks: formal and prototype observations kept separate")
        if memory:
            memory_ax.bar([("formal/" if g["formal_cost_eligible"] else "screen/") + label(g) for g in memory],
                          [g["allocated_peak_bytes"]["mean"] / 2**30 for g in memory])
            memory_ax.set_ylabel("Mean peak allocated GiB")
            memory_ax.tick_params(axis="x", rotation=20, labelsize=6)
        else:
            memory_ax.text(.5, .5, "Memory measurements missing", ha="center", va="center", transform=memory_ax.transAxes)
            memory_ax.set_xticks([])
            memory_ax.set_yticks([])
    save("cost-memory", "Cost, coordination overhead and allocator peaks", costs["groups"], draw_cost_memory,
         "formal observations and descriptive prototype costs are separate; accepted replay is not double-subtracted; peaks do not establish admission benefit")
    figures[-1].update(formal_samples=costs["formal_samples"], prototype_samples=costs["prototype_samples"],
                       performance_qualification="not conferred by this descriptive figure")
    migration = [r for r in rows if r.get("type") == "cfg_mechanism" and _timing_eligible(r)
                 and r.get("all_requests_decoded") is True
                 and r.get("cost_scope") == "all_requests_decode_complete"
                 and "stay" in r.get("costs_ms", {})
                 and set(r["costs_ms"]) <= {"stay", *MIGRATION_ACTIONS} and len(r["costs_ms"]) >= 2
                 and all(_number(v, True) for v in r["costs_ms"].values())]
    def draw_migration(ax):
        for action in ("stay", *MIGRATION_ACTIONS):
            points = [(i, r["costs_ms"][action] / 1000) for i, r in enumerate(migration) if action in r["costs_ms"]]
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], "o-", label=action)
        ax.set_xticks(range(len(migration)), [str(r.get("case_id", r.get("condition"))) for r in migration], rotation=20)
        ax.set_ylabel("All participating requests decoded (seconds)")
        ax.legend()
    save("migration-crossover", "Migration costs with real waiting and complete decoding", migration, draw_migration,
         "missing queue coverage prevents qualification even when descriptive costs exist")
    return dict(status="completed", paths=figures)


def report(root):
    """Read sealed metrics and verify archive/raw references only while idle."""
    root = Path(root).absolute()
    preliminary = read_json(root / "protocol.json")
    with idle_guard(root, [preliminary["ledger"]]):
        frozen = verify_protocol(root / "protocol.json", verify_current=False)
        rows = read_records(root, frozen)
        references = {}
        for row in rows:
            for item in _artifacts(row, root):
                if item.get("sha256"):
                    references[(item["path"], item["sha256"])] = verify_reference(root, item["path"], item["sha256"])
        availability = Counter(item["status"] for item in references.values())
        hypotheses = evaluate_hypotheses(rows, frozen["config"], frozen=frozen)
        descriptive_rows = descriptive_inputs(frozen, rows)
        asymmetry = asymmetry_evidence(descriptive_rows, frozen["config"])
        hypotheses["cfg_asymmetry"] = _decision(asymmetry["status"], asymmetry["scope"],
            causal_identification=False, matched_cost_control=False)
        continuation = frozen.get("prototype_continuation")
        result = dict(schema_version="ditserve.cfg.report.v1", status="completed",
                      protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"],
                      source_archive=frozen.get("source_archive"), measured_records=len(rows),
                      valid_complete_unit_records=sum(r.get("performance_valid") is True for r in rows),
                      hypotheses=hypotheses, budget=budget_summary(read_json(frozen["ledger"])),
                      figures=make_figures(root, descriptive_rows),
                      cost_memory=cost_memory_summary(descriptive_rows), cfg_asymmetry_evidence=asymmetry,
                      screening_provenance=dict(origin="verified_parent_prototype" if continuation else "current_run",
                          parent_run=continuation.get("parent_run") if continuation else None,
                          continuation_sha256=continuation.get("sha256") if continuation else None,
                          parent_protocol_sha256=continuation.get("parent_protocol_sha256") if continuation else None,
                          parent_source_sha256=continuation.get("parent_source_sha256") if continuation else None,
                          parent_context_sha256=continuation.get("parent_context_sha256") if continuation else None,
                          parent_raw_arrays_reverified=False,
                          scope="implementation continuation only; no inherited formal development, holdout or performance qualification"),
                      evidence_availability=dict(raw_available=availability["raw_available"],
                          retained_metrics_only=availability["retained_metrics_only"],
                          interpretation="retained_metrics_only is previously recorded evidence, not new raw-array re-evaluation"),
                      quality_scope="mean full-video LPIPS <= 0.05 is an empirical filter, not absolute or per-request quality assurance",
                      performance_scope="oracle and shadow diagnostics never qualify cost; full frozen-text pairs are not online E2E goodput",
                      selection_scope="development only; holdout failure stops the chosen policy without reselection")
        result["sha256"] = digest(result)
        atomic_json(root / "report.json", result)
        labels = {"supported": "支持", "falsified": "在已测范围内未通过", "insufficient_evidence": "证据不足", "deferred": "后置",
                  "descriptive_evidence": "仅观察关联"}
        lines = ["# CFG 双分支缓存预实验", "", "缺测不是零收益；oracle 与诊断运行不能提供正式性能资格。", "",
                 "| 方向 | 结论 | 依据 |", "|---|---|---|"]
        for name, key in (("缓存策略", "cache"), ("状态迁移", "migration"), ("缓存联合控制", "cache_joint_control"),
                          ("内存容量", "memory"), ("CFG 非对称误差关联", "cfg_asymmetry")):
            item = hypotheses[key]
            lines.append("| %s | %s | %s |" % (name, labels[item["status"]], item["reason"]))
        budget = result["budget"]
        lines += ["", "累计结算 %.3f / %.0f GPU·秒；活动会话估算 %.3f GPU·秒，单列且未结算。" %
                  (budget["settled_gpu_seconds"], budget["limit_gpu_seconds"], budget["active_estimated_gpu_seconds"]),
                  "", "原始证据可重新评价：%d 个引用；仅保留已有指标：%d 个引用。未声明的文件缺失会使报告失败。" %
                  (availability["raw_available"], availability["retained_metrics_only"]), "",
                  "LPIPS 是完整视频的经验筛选指标，不代表绝对质量保证。留出集不用于重新选参；SP、LP 与在线控制器后置。", ""]
        if continuation:
            lines += ["筛选依据来自已验证的父原型封印，仅允许继续实现；当前运行的开发集、留出集和性能资格均须重新测量。", "",
                      "图中 parent 仅沿用父运行已记录的诊断与原型指标，未重新评价其原始数组。父协议 `%s`，源码 `%s`，上下文 `%s`。" %
                      (continuation["parent_protocol_sha256"], continuation["parent_source_sha256"], continuation["parent_context_sha256"]), ""]
        lines += ["成本图将原型描述性时间、协调开销与正式观测分开。原型乐观界并非已执行的高效实现；缺失开销与显存值保留为未测。", ""]
        if asymmetry["samples"]:
            lines += ["| 输入／设备 | 已记录混合步数 | 混合步 guided 相对 L2 均值 | RR 步 guided 相对 L2 均值 | 独立 0.08 LPIPS | 成对 0.08 LPIPS |",
                      "|---|---:|---:|---:|---:|---:|"]
            def number(value):
                return "未测" if value is None else "%.6f" % value
            for sample in asymmetry["samples"]:
                comparison = sample["paired_comparison"] or {}
                lines.append("| %s/%s/%s | %s | %s | %s | %s | %s |" %
                    (sample["prompt_id"], sample["seed"], sample["device_role"],
                     "未测" if sample["mixed_steps"] is None else str(sample["mixed_steps"]),
                     number(sample["mean_mixed_guided_relative_l2"]), number(sample["mean_rr_guided_relative_l2"]),
                     number(sample["independent_lpips_mean"]), number(comparison.get("paired_lpips_mean"))))
            lines += ["", "这些是同输入、同设备、相同参考输出的关联观测；轨迹内各步不独立，也没有相近成本的控制组，不能识别 CFG 非对称的因果效应。未采样的 CC 步没有被记为零误差。", ""]
        for item in result["figures"].get("paths", []):
            lines.append("![%s](%s)" % (Path(item["path"]).stem, item["path"]))
        (root / "report.md").write_text("\n".join(lines) + "\n")
        return result
