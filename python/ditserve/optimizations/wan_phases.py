"""Finite Wan compilation and eager cache/K4 pilots; caller owns GPU admission.

Mechanism timing uses preencoded CPU prompts. This is deliberately separate
from the root runner's arrived-request online experiment and qualification.
"""

import hashlib
import itertools
from pathlib import Path
import statistics
import time

from ..directions.budget import BudgetStop
from ..directions.direction1 import numerical_error, solver_matches, quantile
from ..research.cfg_adapter import CfgResearchAdapter
from ..research.prompt_bank import FrozenPromptBank
from ..util import atomic_json
from .wan import BlockBackend, NativeWanExecutor, WanCompiledExecutor
from .cache_executor import PairedCacheExecutor


DEFAULTS = dict(wan_input_seeds=[101, 102], wan_input_prompts=2, wan_repeats=1,
                minimum_local_gain=.05, cache_screen_inputs=3,
                cache_full_qualification="complete_pairs_until_budget_limit",
                group4_requires_verified_host_dispatch_fraction=.05)
_IDS = itertools.count()


class _Evidence:
    def __init__(self, context, phase):
        self.context, self.path = context, Path(context.output) / (phase + ".json")
        if self.path.exists():
            raise FileExistsError("optimization evidence already exists: " + str(self.path))
        self.data = dict(schema_version="ditserve.optimizations.wan.v1", phase=phase,
            status="running", local_qualified=False, scientific_qualified=False,
            system_qualified=False, rows=[], active_unit=None,
            metadata=getattr(context, "metadata", {}), selected_backend="eager",
            timing_scope="preencoded CPU prompts; synchronized prepare+denoise+decode; evidence IO and quality outside service timing",
            quality_policy=dict(video_lpips=.05, same_backend_resume_rtol=1e-5, same_backend_resume_atol=1e-6))
        self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def begin(self, **unit):
        self.data["active_unit"] = dict(unit, started=False)
        self.save()
        self.context.check_budget()
        self.data["active_unit"]["started"] = True
        self.save()

    def append(self, row):
        self.data["rows"].append(row)
        self.data["active_unit"] = None
        self.save()

    def finish(self, **values):
        values.setdefault("status", "passed")
        self.data.update(active_unit=None, **values)
        self.save()
        return self.data

    def interrupted(self, error):
        self.data["local_gate_passed_before_interruption"] = bool(self.data.get("local_qualified"))
        self.data.update(status="stopped" if isinstance(error, BudgetStop) else "failed",
            local_qualified=False, scientific_qualified=False, system_qualified=False,
            error=type(error).__name__ + ": " + str(error), budget_stopped=isinstance(error, BudgetStop))
        self.save()


def _ensure(context, seconds):
    context.check_budget.ensure_unit(float(seconds))


def _inputs(context):
    prompts = context.config["direction1"]["prompts"]
    if len(prompts) < 2:
        raise ValueError("Wan pilot requires the two frozen development prompts")
    return [dict(input_id="p%d-s%d" % (i, seed), prompt=prompt, seed=seed, split="dev")
            for i, prompt in enumerate(prompts[:2]) for seed in (101, 102)]


def _cache_inputs(split=None):
    from ..directions.direction2 import _inputs as catalog, _screen_inputs
    items = _screen_inputs() if split is None else catalog(split)
    return [dict(item, input_id="%s-%s" % (item["prompt_id"], item["seed"])) for item in items]


def _spec(item, cache):
    return dict(request_id="optimization-%08d" % next(_IDS), prompt=item["prompt"],
                negative_prompt="", seed=item["seed"], height=480, width=832, frames=81,
                steps=32, guidance_scale=5., cache=cache,
                threshold=.08 if cache else 0., max_continuous=1)


def _validate(context):
    quality = context.config.get("quality", {})
    if (quality.get("video_lpips", .05) != .05 or quality.get("rtol", 1e-5) != 1e-5
            or quality.get("atol", 1e-6) != 1e-6):
        raise ValueError("Wan quality protocol differs from frozen pilot")
    if context.worker.attention != "flash":
        raise ValueError("Wan optimization pilot requires A100 Flash backend")


def _hash(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _artifact(context, name, array):
    import numpy as np
    root = Path(context.config.get("artifact_root", Path(context.output) / "artifacts"))
    relative_output = Path(context.output).resolve().relative_to(Path(context.run_root).resolve())
    root = root / Path(context.run_root).name / relative_output
    root.mkdir(parents=True, exist_ok=True)
    path = root / (name + ".npy")
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(),
                array_sha256=_hash(array), shape=list(array.shape), raw_available=True)


def _quality(reference, candidate):
    from ..gpu.quality import evaluate_quality, quality_pass
    value = evaluate_quality(reference, candidate, with_lpips=True, lpips_device="cpu")
    return dict(metrics=value, passed=quality_pass(value, .05))


def _cache_paths(events, expected_steps=32):
    """Observe committed paired-CFG paths; never infer full work from absent reuse."""
    sequence, counts = [], dict(full=0, reuse=0, mixed=0, unknown=0)
    for event in events:
        if event.get("kind") != "step":
            continue
        path, reused = event.get("path"), event.get("reused_forwards")
        outcome = ("full" if path == "compute" and reused == 0 else
                   "reuse" if path == "reuse" and reused == 2 else
                   "mixed" if path == "mixed" else "unknown")
        counts[outcome] += 1
        sequence.append(dict(step_index=event.get("step_index"),
            completed_steps=event.get("completed_steps"), path=path,
            reused_forwards=reused, outcome=outcome))
    ordered = (len(sequence) == expected_steps and
               all(row["step_index"] == index and row["completed_steps"] == index+1
                   for index, row in enumerate(sequence)))
    valid = ordered and counts["mixed"] == counts["unknown"] == 0
    return dict(path_sequence=sequence, counts=counts, required_steps=expected_steps,
                observed_steps=len(sequence), valid=valid,
                full_reuse_observed=valid and counts["full"] > 0 and counts["reuse"] > 0,
                scope="committed paired-CFG middle-block path; prefix always executes")


def _cache_resume_paths(left_events, right_events, interruptions, expected_steps=32):
    left, right = _cache_paths(left_events, expected_steps), _cache_paths(right_events, expected_steps)
    counts = dict(full=0, reuse=0, mixed=0, unknown=0)
    for interruption in interruptions:
        index = interruption["step"]
        matches = [row for row in right["path_sequence"]
                   if row["step_index"] == index and row["completed_steps"] == index+1]
        path = matches[0]["outcome"] if len(matches) == 1 else "unknown"
        interruption.update(resolved_step_path=path,
            path_scope="Sana inserted mid-prefix before cache decision; path resolved after this step committed")
        counts[path] += 1
    return dict(continuous=left, k4=right,
        valid=left["valid"] and right["valid"],
        paths_equal=left["path_sequence"] == right["path_sequence"],
        sana_interruptions=dict(counts=counts,
            unverified_paths=[path for path in ("full", "reuse") if not counts[path]],
            full_reuse_cross_product_required=False,
            scope="fixed step indices 0/8/16; missing insertion-path coverage does not invalidate whole-trajectory coverage"))


def _cache_coverage(rows, input_ids):
    """Aggregate only the fixed correctness trajectories, before speed screening."""
    samples, totals = [], {name: dict(full=0, reuse=0, mixed=0, unknown=0)
                          for name in ("original_atomic", "original_k4", "continuous", "resumed_k4")}
    insertion_counts = dict(full=0, reuse=0, mixed=0, unknown=0)
    valid = True
    for input_id in input_ids:
        original = [row for row in rows if row.get("input_id") == input_id and
                    row.get("kind") == "original_split_equivalence"]
        resumes = [row for row in rows if row.get("input_id") == input_id and
                   row.get("kind") == "same_backend_resume"]
        observations = {}
        for name, native in (("original_atomic", True), ("original_k4", False)):
            matches = [row for row in original if row.get("native") is native]
            observations[name] = matches[0]["service"].get("cache_paths") if len(matches) == 1 else None
        resume = resumes[0]["result"].get("cache_paths", {}) if len(resumes) == 1 else {}
        observations.update(continuous=resume.get("continuous"), resumed_k4=resume.get("k4"))
        sample_valid = all(value and value.get("valid") is True and value.get("required_steps") == 32
                           for value in observations.values())
        equality = (sample_valid and resume.get("paths_equal") is True and
                    observations["original_atomic"]["path_sequence"] == observations["original_k4"]["path_sequence"])
        valid = bool(valid and equality)
        for name, value in observations.items():
            if value:
                for path in totals[name]:
                    totals[name][path] += value["counts"][path]
        for path in insertion_counts:
            insertion_counts[path] += resume.get("sana_interruptions", {}).get("counts", {}).get(path, 0)
        samples.append(dict(input_id=input_id, valid=bool(sample_valid), paths_equal=bool(equality),
            counts={name: value["counts"] if value else None for name, value in observations.items()},
            sana_interruptions=resume.get("sana_interruptions")))
    missing = [path for path in ("full", "reuse") if not totals["resumed_k4"][path]]
    return dict(complete=bool(input_ids) and valid, passed=bool(input_ids) and valid and not missing,
        required_inputs=list(input_ids), samples=samples, counts=totals,
        missing_resumed_paths=missing,
        sana_interruptions=dict(counts=insertion_counts,
            unverified_paths=[path for path in ("full", "reuse") if not insertion_counts[path]],
            full_reuse_cross_product_required=False),
        scope="all fixed 32-step K4 recovery trajectories must collectively observe full and reuse; Sana insertion-path cross product is descriptive only")


def _generate(context, evidence, bank, item, executor, *, cache=False, native_cache=False):
    """A complete trajectory; interrupted progress is saved by the phase caller."""
    state = continuation = None
    worker = context.worker
    spec = _spec(item, cache)
    context.check_budget()
    bank.warm(spec)
    worker._sync()
    started = time.perf_counter()
    try:
        state = bank.prepare(spec)
        if native_cache:
            adapter = CfgResearchAdapter(worker)
            policy = dict(id="paired_008", threshold=.08, max_continuous=1,
                          cfg_mode="paired", implementation="split_prefix", cache=True)
        else:
            continuation = executor.attach(state)
        quanta = []
        while state.step_index < 32:
            context.check_budget()
            if native_cache:
                adapter.step(state, policy=policy)
            else:
                quanta.append(executor.advance(continuation, 4, check_budget=context.check_budget))
            evidence.data["active_unit"].update(completed_steps=state.step_index,
                continuation_phase=getattr(continuation, "phase", None),
                continuation_branch=getattr(continuation, "branch", None),
                continuation_next_block=getattr(continuation, "next_block", None))
        array = worker.finish(state)
        worker._sync()
        row = dict(ms=(time.perf_counter()-started)*1000, completed_steps=32,
                   output_sha256=_hash(array), generation_completed=True,
                   quantum_p95_ms=quantile([q["ms"] for q in quanta], .95),
                   peak_packet_bytes=max((q["extra_state_bytes"] for q in quanta), default=0),
                   decode_ms=next((e["ms"] for e in reversed(state.events) if e.get("kind")=="decode"), None),
                   reuse_steps=[e["completed_steps"] for e in state.events
                                if e.get("kind")=="step" and e.get("reused_forwards")==2],
                   cache=cache, native_cache=native_cache,
                   backend="paired_008_original" if native_cache else executor.backend.name)
        if cache:
            row["cache_paths"] = _cache_paths(state.events)
        return array, row
    finally:
        if continuation is not None and not continuation.closed:
            executor.close(continuation)
        if state is not None and state.residency != "released":
            worker.release(state)


def _cache_equal(torch, left, right):
    if (set(left.contexts) != set(right.contexts) or left.needs_anchor != right.needs_anchor
            or left.current_step_refreshed != right.current_step_refreshed):
        return False
    for name in left.contexts:
        a, b = left.contexts[name], right.contexts[name]
        names = ("executed_steps", "transformer_executed_steps", "cached_steps", "cfg_cached_steps",
                 "continuous_cached_steps", "cfg_continuous_cached_steps",
                 "accumulated_executed_steps", "accumulated_transformer_executed_steps",
                 "accumulated_residual_diff", "cfg_accumulated_residual_diff",
                 "residual_diffs", "cfg_residual_diffs")
        if any(getattr(a, key) != getattr(b, key) for key in names):
            return False
        if set(a.buffers) != set(b.buffers):
            return False
        for key, value in a.buffers.items():
            ref = b.buffers[key]
            if (value.shape != ref.shape or value.dtype != ref.dtype or
                    not bool(torch.isfinite(value).all()) or
                    not torch.allclose(value, ref, rtol=1e-5, atol=1e-6)):
                return False
    return True


def _resume_check(context, evidence, bank, item, executor, *, cache=False):
    """Lockstep same-backend continuous step versus K4 with actual Sana work."""
    worker, torch = context.worker, context.worker.torch
    left = right = a = b = None
    steps, interrupts, observations = [], [], []
    cache_paths = None
    try:
        left, right = bank.prepare(_spec(item, cache)), bank.prepare(_spec(item, cache))
        a, b = executor.attach(left), executor.attach(right)
        for step in range(32):
            context.check_budget()
            executor.advance(a, 60, check_budget=context.check_budget)
            injected = False
            while right.step_index == step:
                options = dict(observe=True, check_budget=context.check_budget)
                if getattr(executor, "supports_host_observation", False):
                    options["observe_host"] = True
                observation = executor.advance(b, 4, **options)
                observations.append(observation)
                if step in (0, 8, 16) and not injected and right.step_index == step:
                    _, service = context.sana.generate(item["prompt"], seed=item["seed"],
                                                       check_budget=context.check_budget)
                    interrupts.append(dict(step=step, branch=b.branch,
                                           phase=getattr(b, "phase", None), next_block=b.next_block,
                                           image_service=service))
                    injected = True
            latent = bool(torch.isfinite(left.latents).all() and torch.isfinite(right.latents).all()
                          and torch.allclose(left.latents, right.latents, rtol=1e-5, atol=1e-6))
            solver = solver_matches(torch, left.scheduler, right.scheduler)
            cache_state = not cache or _cache_equal(torch, left, right)
            rng = bool(torch.equal(left.generator.get_state(), right.generator.get_state()))
            row = dict(step=step+1, latent=latent, solver=solver, cache_state=cache_state, rng=rng,
                       passed=latent and solver and cache_state and rng)
            if cache:
                cache_paths = _cache_resume_paths(left.events, right.events, interrupts, step+1)
                for name, source in (("continuous_path", "continuous"), ("k4_path", "k4")):
                    matches = [path for path in cache_paths[source]["path_sequence"]
                               if path["step_index"] == step and path["completed_steps"] == step+1]
                    row[name] = matches[0] if len(matches) == 1 else None
                row["cache_paths_equal"] = cache_paths["valid"] and cache_paths["paths_equal"]
                evidence.data["active_unit"]["cache_paths"] = cache_paths
            steps.append(row)
            evidence.data["active_unit"].update(completed_steps=step+1, step_checks=steps,
                                                  interruptions=interrupts)
            # Correctness diagnostics are outside the performance service timer.
            # Persist each committed step so interruption has bounded evidence.
            evidence.save()
            if not row["passed"]:
                retained = []
                for label, value in (("continuous", left.latents), ("k4", right.latents)):
                    artifact = _artifact(context, "failed-latent-%s-%s-s%d-%s" %
                        (executor.backend.name, item["input_id"], step+1, label),
                        value.detach().float().cpu().numpy())
                    retained.append(dict(artifact, kind="latent_not_decoded_video"))
                result = dict(passed=False, steps=steps, interruptions=interrupts,
                              artifacts=retained, reason="same_backend_step_state_mismatch")
                if cache:
                    result["cache_paths"] = cache_paths
                return result
        retention = []
        first = worker.finish(left)
        retention.append(_artifact(context, "resume-%s-%s-continuous" % (executor.backend.name, item["input_id"]), first))
        evidence.data["active_unit"]["artifacts"] = retention
        evidence.save()
        context.check_budget()
        second = worker.finish(right)
        retention.append(_artifact(context, "resume-%s-%s-k4" % (executor.backend.name, item["input_id"]), second))
        evidence.data["active_unit"]["artifacts"] = retention
        numerical = numerical_error(second, first)
        wall_ms = sum(row["ms"] for row in observations)
        block_ms = sum(block["ms"] for row in observations for block in row["block_cuda_ms"])
        host_rows = [row for row in observations if row.get("pure_host_control_ms") is not None]
        host_ms = sum(row["pure_host_control_ms"] for row in host_rows)
        total_ms = sum(row["quantum_total_wall_ms"] for row in host_rows)
        result = dict(passed=numerical["passed"] and len(interrupts)==3, steps=steps,
                    interruptions=interrupts, final=numerical, artifacts=retention,
                    dispatch_observation=dict(quantum_wall_ms=wall_ms, block_cuda_ms=block_ms,
                        residual_ms=wall_ms-block_ms, host_dispatch_fraction=None,
                        pure_host_control_ms=host_ms if host_rows else None,
                        quantum_total_wall_ms=total_ms if host_rows else None,
                        host_control_fraction_lower_bound=host_ms/total_ms if total_ms else None,
                        host_scope="explicit pure-Python guard/control spans only; excludes tensor calls, CUDA APIs, sync, budget callback and IO; not total dispatch or nonoverlapped critical-path cost",
                        scope="instrumented correctness only; residual includes prelude, output, solver, sync and event overhead; not isolated host dispatch"),
                    scope="same backend uninterrupted step vs K4; real co-resident image work")
        if cache:
            result["cache_paths"] = cache_paths
        return result
    finally:
        for state, continuation in ((left,a),(right,b)):
            if continuation is not None and not continuation.closed:
                executor.close(continuation)
            if state is not None and state.residency != "released":
                worker.release(state)


def _performance(context, evidence, bank, inputs, candidate, baseline, *, cache=False,
                 repeats=1, baseline_native_cache=False, tag="performance",
                 quality_references=None, reference_sink=None):
    references, hashes, qualities, rows = dict(quality_references or {}), {}, [], []
    for order_id, order in enumerate((("eager", "candidate"), ("candidate", "eager"))):
        for item in inputs:
            for repeat in range(repeats):
                # Whole pair admission; estimate updated conservatively from
                # completed real trajectories, never shorten a frozen request.
                prior = [r["service"]["ms"]/1000 for r in rows]
                _ensure(context, 2*max(prior, default=180.)*1.25 + 60)
                pair = []
                for arm in order:
                    evidence.begin(kind=tag, order=order_id, input_id=item["input_id"],
                                   repeat=repeat, arm=arm, backend=candidate.backend.name)
                    ex = candidate if arm=="candidate" else baseline
                    array, service = _generate(context, evidence, bank, item, ex, cache=cache,
                        native_cache=baseline_native_cache and arm=="eager")
                    row = dict(kind=tag, order=order_id, input_id=item["input_id"], repeat=repeat,
                               arm=arm, service=service, candidate_backend=candidate.backend.name)
                    key = arm, item["input_id"]
                    previous = hashes.setdefault(key, service["output_sha256"])
                    row["repeat_exact"] = previous == service["output_sha256"]
                    if arm=="eager" and item["input_id"] not in references:
                        references[item["input_id"]] = array.copy()
                        if reference_sink is not None:
                            reference_sink[item["input_id"]] = references[item["input_id"]]
                    if repeat==0 or not row["repeat_exact"]:
                        row["artifact"] = _artifact(context,
                            "%s-%s-o%d-%s-%s-r%d" % (tag,candidate.backend.name,order_id,item["input_id"],arm,repeat),array)
                    # Publish the completed generation before CPU quality work.
                    evidence.append(row)
                    rows.append(row)
                    pair.append((row,array))
                for row,array in pair:
                    if repeat==0:
                        row["quality"] = _quality(references[item["input_id"]],array)
                        qualities.append(row["quality"]["passed"])
                    row["complete"] = True
                evidence.save()
    gains=[]
    for order_id in range(2):
        means={arm:statistics.mean(r["service"]["ms"] for r in rows
               if r["order"]==order_id and r["arm"]==arm) for arm in ("eager","candidate")}
        gains.append(1-means["candidate"]/means["eager"])
    # Hash equality across independent repetitions is descriptive. The frozen
    # numerical gate applies to same-backend resume; candidate quality is LPIPS.
    quality = len(qualities)==len(inputs)*4 and all(qualities)
    return dict(complete=True, distinct_inputs=len(inputs), repeats=repeats, orders=2,
                quality_passed=quality, quality_checked=len(qualities), gains=gains,
                performance_passed=all(g>=.05 for g in gains),
                baseline_backend=baseline.backend.name,candidate_backend=candidate.backend.name,
                quality_reference="original eager same-input outputs",
                n_per_arm=len(inputs)*repeats*2)


def run_wan(context):
    """Four-input single-block pilot; group4 needs independently isolated host cost."""
    evidence = _Evidence(context, "wan-compile")
    bank = FrozenPromptBank(context.worker)
    try:
        _validate(context)
        inputs = _inputs(context)
        evidence.data["inputs"] = inputs
        context.executors = {"eager": NativeWanExecutor(context.worker)}
        eager = context.executors["eager"]
        evidence.data["eager_baseline"] = eager.backend.report()
        backend = BlockBackend(context.worker, compiled=True, group_size=1)
        candidate = WanCompiledExecutor(context.worker, backend)
        context.executors[backend.name] = candidate
        # These complete checks also warm every actual block/branch layout.
        # Diagnose candidate compilation before repeating all native checks;
        # both retain all four inputs, and performance AB/BA is unchanged.
        evidence.data["resume_backend_order"] = [backend.name, "eager"]
        for ex in (candidate, eager):
            for item in inputs:
                _ensure(context, 600)
                evidence.begin(kind="same_backend_resume", backend=ex.backend.name,input_id=item["input_id"])
                check = _resume_check(context,evidence,bank,item,ex)
                evidence.append(dict(kind="same_backend_resume",backend=ex.backend.name,
                                     input_id=item["input_id"],result=check))
                if not check["passed"]:
                    return evidence.finish(status="failed",reason="same_backend_resume_failed",
                                           compilation=backend.report())
        backend.freeze_compilation()
        references = {}
        summary = _performance(context,evidence,bank,inputs,candidate,eager,reference_sink=references)
        local = summary["quality_passed"] and summary["performance_passed"]
        selected = backend.name if local else "eager"
        evidence.data.update(single_block=summary,selected_backend=selected,
                             compilation=backend.report(),local_qualified=local)
        evidence.save()
        # The first candidate correctness input includes cold capture; use the
        # remaining three warmed inputs for the explicit host-control lower bound.
        host_samples = [row["result"].get("dispatch_observation", {}) for row in evidence.data["rows"]
            if row["kind"]=="same_backend_resume" and row.get("backend")==backend.name][1:]
        known_host = (len(host_samples)==3 and all(sample.get("pure_host_control_ms") is not None
            and sample.get("quantum_total_wall_ms",0)>0 for sample in host_samples))
        host_fraction = (sum(sample["pure_host_control_ms"] for sample in host_samples)/
            sum(sample["quantum_total_wall_ms"] for sample in host_samples)) if known_host else None
        optional = dict(status="not_run", qualified=False, host_dispatch_fraction=None,
            host_control_fraction_lower_bound=host_fraction, required_host_control_fraction=.05,
            reason="single_block_gate_not_passed" if not local else
                   "host_control_fraction_not_measured" if host_fraction is None else
                   "host_control_lower_bound_below_trigger",
            scope="explicit pure Python control lower bound; CUDA-wall residual never used as host cost")
        if local and host_fraction is not None and host_fraction>=.05:
            prior = [row["service"]["ms"]/1000 for row in evidence.data["rows"] if "service" in row]
            expected = 24*max(prior,default=180.)*1.25+600
            remaining = context.check_budget.remaining()
            if remaining<expected:
                optional.update(status="not_measured_budget_limited",reason="complete_group4_matrix_does_not_fit",
                    required_seconds=expected,remaining_seconds=remaining)
            else:
                try:
                    grouped = WanCompiledExecutor(context.worker,
                        BlockBackend(context.worker,compiled=True,group_size=4))
                    context.executors[grouped.backend.name] = grouped
                    checks_passed = True
                    for item in inputs:
                        _ensure(context,600)
                        evidence.begin(kind="same_backend_resume",backend=grouped.backend.name,input_id=item["input_id"])
                        check = _resume_check(context,evidence,bank,item,grouped)
                        evidence.append(dict(kind="same_backend_resume",backend=grouped.backend.name,
                            input_id=item["input_id"],result=check))
                        if not check["passed"]:
                            checks_passed = False
                            optional.update(status="failed",reason="group4_same_backend_resume_failed")
                            break
                    if checks_passed:
                        grouped.backend.freeze_compilation()
                        grouped_summary = _performance(context,evidence,bank,inputs,grouped,candidate,
                            tag="group4_performance",quality_references=references)
                        qualified = grouped_summary["quality_passed"] and grouped_summary["performance_passed"]
                        optional.update(status="passed" if grouped_summary["quality_passed"] else "failed",
                            qualified=qualified,result=grouped_summary,
                            reason="complete_group4_ablation_against_single_block" if grouped_summary["quality_passed"] else "group4_quality_failed_primary_preserved")
                        if qualified:
                            selected = grouped.backend.name
                except Exception as error:
                    if isinstance(error,BudgetStop):
                        raise
                    optional.update(status="failed",qualified=False,
                        reason="group4_implementation_failed_primary_preserved",
                        error=type(error).__name__+": "+str(error),
                        active_unit=evidence.data.get("active_unit"))
                    evidence.data["active_unit"] = None
                    evidence.save()
        return evidence.finish(local_qualified=local,selected_backend=selected,
                               single_block=summary,optional_group4=optional,
                               compilation={key:ex.backend.report() for key,ex in context.executors.items()},
                               reason="complete local compilation pilot; online qualification separate")
    except BaseException as error:
        evidence.data["compilation"] = {key: ex.backend.report()
            for key, ex in getattr(context, "executors", {}).items()}
        evidence.interrupted(error)
        raise
    finally:
        bank.entries.clear()


def run_cache(context):
    """Eager-only screen, then admit qualification one complete pair at a time."""
    evidence = _Evidence(context,"cache-k4")
    bank=FrozenPromptBank(context.worker)
    try:
        _validate(context)
        inputs=_cache_inputs()
        executor=PairedCacheExecutor(context.worker,BlockBackend(context.worker,compiled=False))
        context.cache_executor=executor
        evidence.data.update(inputs=inputs,backend="eager",compiled=False,
            cache_quality_qualified=False,required_full_qualification="8dev x2orders +4heldout; not completed by this screen")
        # Reference arithmetic is the prior atomic split-prefix implementation.
        for item in inputs:
            _ensure(context,600)
            pair=[]
            for native in (True,False):
                evidence.begin(kind="original_split_equivalence",input_id=item["input_id"],native=native)
                array,service=_generate(context,evidence,bank,item,executor,cache=True,native_cache=native)
                row=dict(kind="original_split_equivalence",input_id=item["input_id"],native=native,
                         service=service,artifact=_artifact(context,"cache-original-%s-%s"%(item["input_id"],native),array))
                evidence.append(row)
                pair.append((row,array))
            comparison=numerical_error(pair[1][1],pair[0][1])
            paths=pair[0][0]["service"]["reuse_steps"]==pair[1][0]["service"]["reuse_steps"]
            observed=[row["service"].get("cache_paths", {}) for row,array in pair]
            paths_valid=all(value.get("valid") is True for value in observed)
            full_paths_equal=(observed[0]["path_sequence"]==observed[1]["path_sequence"]
                              if paths_valid else None)
            evidence.append(dict(kind="original_split_comparison",input_id=item["input_id"],
                                 numerical=comparison,cache_paths_equal=paths,
                                 path_records_valid=paths_valid,full_paths_equal=full_paths_equal,
                                 passed=comparison["passed"] and paths and full_paths_equal is not False))
            if not comparison["passed"] or not paths or full_paths_equal is False:
                return evidence.finish(status="failed",reason="new_continuation_differs_from_original_paired")
            _ensure(context,600)
            evidence.begin(kind="same_backend_resume",input_id=item["input_id"],backend="eager-paired")
            check=_resume_check(context,evidence,bank,item,executor,cache=True)
            evidence.append(dict(kind="same_backend_resume",input_id=item["input_id"],result=check))
            if not check["passed"]:
                return evidence.finish(status="failed",reason="cache_same_backend_resume_failed")
            evidence.data["cache_path_coverage"]=_cache_coverage(
                evidence.data["rows"], [value["input_id"] for value in inputs])
            evidence.save()
        coverage=evidence.data["cache_path_coverage"]
        if not coverage["passed"]:
            return evidence.finish(status="stopped",decision="未验证",local_qualified=False,
                cache_quality_qualified=False,screen_local_qualified=False,
                full_qualification_completed=False,
                reason="fixed_cache_correctness_trajectory_path_coverage_incomplete")
        # Speed/quality reference is complete cache-off K4, same shape and input.
        eager=NativeWanExecutor(context.worker)
        evidence.data["eager_baseline"] = eager.backend.report()
        summary=_cache_performance(context,evidence,bank,inputs,executor,eager)
        local=summary["quality_passed"] and summary["performance_passed"]
        evidence.data.update(screen=summary, screen_local_qualified=local, local_qualified=local)
        evidence.save()
        qualification = dict(status="not_run", complete=False, qualified=False,
                             reason="three_input_screen_gate_not_passed")
        if local:
            qualification = _cache_full_qualification(context,evidence,bank,executor,eager)
        qualified = qualification.get("qualified") is True
        complete_failure = qualification.get("complete") is True and not qualified
        failed_gates = [name for name in ("quality", "performance")
                        if qualification.get(name + "_passed") is False]
        effective_local = local and not complete_failure
        return evidence.finish(local_qualified=effective_local,
            selected_backend="eager-paired-k4" if effective_local else "eager",
            screen=summary,screen_local_qualified=local,cache_quality_qualified=qualified,scientific_qualified=False,
            full_qualification_completed=qualification.get("complete") is True,
            full_qualification=qualification,
            decision="当前配置无收益" if complete_failure or not local else "未验证",
            reason=("complete 20-pair cache qualification failed: " + ", ".join(failed_gates)
                    if complete_failure else "finite eager cache/K4 pilot; online qualification separate"))
    except BaseException as error:
        evidence.interrupted(error)
        raise
    finally:
        bank.entries.clear()


def _cache_performance(context,evidence,bank,inputs,candidate,baseline):
    """Separate cache flags for off reference and paired candidate."""
    refs,qualities,rows={},[],[]
    for order_id,order in enumerate((("off","paired"),("paired","off"))):
        for item in inputs:
            _ensure(context,510)
            pair=[]
            for arm in order:
                evidence.begin(kind="cache_screen",order=order_id,input_id=item["input_id"],arm=arm)
                array,service=_generate(context,evidence,bank,item,candidate if arm=="paired" else baseline,
                                        cache=arm=="paired")
                if arm=="off":refs[item["input_id"]]=array.copy()
                row=dict(kind="cache_screen",order=order_id,input_id=item["input_id"],arm=arm,service=service,
                         artifact=_artifact(context,"cache-screen-o%d-%s-%s"%(order_id,item["input_id"],arm),array))
                evidence.append(row);rows.append(row);pair.append((row,array))
            for row,array in pair:
                row["quality"]=_quality(refs[item["input_id"]],array)
                qualities.append(row["quality"]["passed"])
                row["complete"]=True
            evidence.save()
    gains=[]
    for order_id in range(2):
        total={arm:sum(r["service"]["ms"] for r in rows if r["order"]==order_id and r["arm"]==arm)
               for arm in ("off","paired")}
        gains.append(1-total["paired"]/total["off"])
    return dict(complete=True,distinct_development_inputs=len(inputs),heldout_inputs=0,
                quality_passed=len(qualities)==len(inputs)*4 and all(qualities),
                quality_checked=len(qualities), n_per_arm=len(inputs)*2,gains=gains,
                performance_passed=all(g>=.05 for g in gains),full_qualification_completed=False)



def _cache_full_qualification(context,evidence,bank,candidate,baseline):
    """Frozen 8dev x AB/BA + 4heldout, with no heldout-based selection."""
    dev, heldout = _cache_inputs("dev"), _cache_inputs("holdout")
    if len(dev)!=8 or len(heldout)!=4:
        raise ValueError("cache qualification catalog differs from frozen 8dev/4heldout")
    pairs=[]
    units=[(split,order_id,item)
           for split,order_id,items in (("dev",0,dev),("dev",1,dev),("holdout",0,heldout))
           for item in items]

    def remaining_at_boundary():
        try:
            return context.check_budget.remaining()
        except BudgetStop as error:
            # Only allocation exhaustion between units is a partial outcome.
            # Supervisor loss and interrupts inside a pair keep the hard-stop path.
            if str(error) != "phase allowance reached":
                raise
            return 0.

    def budget_partial(index, required, remaining):
        pending=[dict(split=split,order=order_id,input_id=item["input_id"],
                      arms=["off","paired"] if order_id==0 else ["paired","off"])
                 for split,order_id,item in units[index:]]
        return dict(status="partial" if pairs else "not_measured_budget_limited",
            complete=False,qualified=False,quality_passed=False,performance_passed=False,
            reason="remaining_budget_cannot_admit_complete_pair",budget_limited=True,
            required_seconds=required,remaining_seconds=remaining,pairs=len(pairs),required_pairs=20,
            dev_pairs=sum(p["split"]=="dev" for p in pairs),required_dev_pairs=16,
            heldout_pairs=sum(p["split"]=="holdout" for p in pairs),required_heldout_pairs=4,
            remaining_pairs=len(pending),remaining_units=pending,next_unit=pending[0],
            observed_quality_checked=len(pairs),
            observed_quality_passed=all(p["quality"]["passed"] for p in pairs) if pairs else None,
            heldout_selection=False,scope="partial eager paired_008 qualification; required denominator remains 20")

    for index,(split,order_id,item) in enumerate(units):
        order=("off","paired") if order_id==0 else ("paired","off")
        previous=[r["service"]["ms"]/1000 for r in evidence.data["rows"] if "service" in r]
        required=2*max(previous,default=180.)*1.25+60
        remaining=remaining_at_boundary()
        if remaining<required:
            return budget_partial(index,required,remaining)
        try:
            _ensure(context,required)
        except BudgetStop as error:
            if (str(error)!="phase allowance reached" and
                    not str(error).startswith("remaining budget cannot admit complete paired unit")):
                raise
            return budget_partial(index,required,remaining_at_boundary())
        samples={}
        for arm in order:
            evidence.begin(kind="cache_qualification",split=split,order=order_id,
                           input_id=item["input_id"],arm=arm)
            array,service=_generate(context,evidence,bank,item,candidate if arm=="paired" else baseline,
                                    cache=arm=="paired")
            row=dict(kind="cache_qualification",split=split,order=order_id,input_id=item["input_id"],
                arm=arm,service=service,artifact=_artifact(context,
                "cache-qualification-%s-o%d-%s-%s"%(split,order_id,item["input_id"],arm),array))
            evidence.append(row)
            samples[arm]=(row,array)
        context.check_budget()
        quality=_quality(samples["off"][1],samples["paired"][1])
        pair=dict(kind="cache_qualification_pair",split=split,order=order_id,input_id=item["input_id"],
            quality=quality,off_ms=samples["off"][0]["service"]["ms"],
            paired_ms=samples["paired"][0]["service"]["ms"],complete=True)
        for row,array in samples.values():row["complete"]=True
        evidence.append(pair)
        pairs.append(pair)
    gains=[]
    for order_id in range(2):
        rows=[p for p in pairs if p["split"]=="dev" and p["order"]==order_id]
        gains.append(1-sum(r["paired_ms"] for r in rows)/sum(r["off_ms"] for r in rows))
    quality_passed=len(pairs)==20 and all(p["quality"]["passed"] for p in pairs)
    performance_passed=all(g>=.05 for g in gains)
    return dict(status="passed" if quality_passed and performance_passed else "failed",
        complete=True,qualified=quality_passed and performance_passed,quality_passed=quality_passed,
        performance_passed=performance_passed,gains=gains,pairs=len(pairs),required_pairs=20,
        dev_pairs=16,heldout_pairs=4,remaining_pairs=0,remaining_units=[],
        heldout_selection=False,scope="eager paired_008 cache quality; no serving qualification")
