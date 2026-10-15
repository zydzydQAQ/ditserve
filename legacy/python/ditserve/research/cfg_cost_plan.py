"""Conservative whole-matrix planning from the sealed prototype run."""

from pathlib import Path

from .common import digest, finite, read_json
from .dual_evidence import read_records
from .dual_matrix import seal


def continuation_cost_plan(frozen, ledger):
    """Plan, never grant qualification or move a stage allowance."""
    if frozen.get("implementation_refinement"):
        return refinement_cost_plan(frozen, ledger)
    if ledger.get("active"):
        raise ValueError("planning requires a settled ledger")
    proof = frozen.get("prototype_continuation")
    if not proof:
        raise ValueError("whole continuation matrix requires a bound prototype")
    from .cfg_continuation import screening_decisions
    screens = screening_decisions(frozen, [], None)
    root = Path(proof["parent_run"])
    parent = read_json(root / "protocol.json")
    units = [read_json(p) for p in (root / "units").glob("*.json")]
    units = [u for u in units if u.get("status") == "completed" and u.get("performance_valid") is True
             and u.get("protocol_sha256") == parent["sha256"]]
    if any(u.get("sha256") != digest({k: v for k, v in u.items() if k != "sha256"}) for u in units):
        raise ValueError("planning units must retain their seals")
    sessions = [s for s in ledger["sessions"] if s.get("protocol_sha256") == parent["sha256"]
                and s.get("status") == "completed"]
    n_candidates = sum(g["eligible"] for g in screens.values())
    per_role = {}
    cleanup = frozen.get("config", {}).get("cleanup_wall_seconds", 30.)
    for role in ("a100", "v100"):
        # Match the recorded worker's actual UUID, never infer costs by index.
        job = read_json(root / ("screen-" + role + "-job.json"))
        uuid = job["physical_devices"][0]["uuid"]
        elapsed = lambda u: u["ended_at"] - u["started_at"]
        screen_units = [u for u in units if u["unit_id"].startswith("screen-" + role + "-")]
        pairs = [u for u in screen_units if "-stage_a-" in u["unit_id"]]
        corr_units = [u for u in units if u["unit_id"].startswith("correctness-" + role + "-")]
        corr = next(s["wall_seconds"] for s in sessions if s["stage"] == "correctness" and s["devices"] == [uuid])
        screen = next(s["wall_seconds"] for s in sessions if s["stage"] == "calibration" and s["devices"] == [uuid])
        paired = next(u for u in corr_units if u["unit_id"].endswith("-paired_008"))
        setup = max(0., screen - sum(map(elapsed, screen_units)), corr - sum(map(elapsed, corr_units)))
        # Extra oracle trajectory uses half the old complete two-trajectory
        # restore pair, including its instrumentation and artifact handling.
        correctness = (corr + elapsed(paired) / 2) * 1.3 + cleanup
        pairs_required = n_candidates * 8 * 2 + 4
        calibration = (max(map(elapsed, pairs)) * pairs_required + setup * 2) * 1.3 + cleanup * 2
        unit_seconds = {u["unit_id"].removeprefix("correctness-" + role + "-"): elapsed(u)
                        for u in corr_units}
        unit_seconds["correctness-" + role + "-paired_008"] = elapsed(paired) * 1.5
        per_role[role] = dict(correctness_gpu_seconds=correctness, calibration_gpu_seconds=calibration,
            physical_uuid=uuid, correctness_units_seconds=unit_seconds,
            measured_max_pair_wall_seconds=max(map(elapsed, pairs)), measured_setup_wall_seconds=setup,
            formal_dev_pairs=n_candidates * 8 * 2, held_out_pairs=4,
            startup_correctness_wall_seconds=correctness - cleanup)
    allocation = ledger["reallocations"][-1]
    remaining = {stage: max(0., allowance - (ledger["stage_used_gpu_seconds"].get(stage, 0.)
        - allocation["baseline_stage_used"].get(stage, 0.)))
        for stage, allowance in allocation["stage_allowances_gpu_seconds"].items()}
    estimates = {stage: sum(p[stage + "_gpu_seconds"] for p in per_role.values())
                 for stage in ("correctness", "calibration")}
    fits = all(estimates[s] < remaining[s] for s in estimates)
    return seal(dict(status="fits" if fits else "budget_stopped", safety_factor=1.3,
        estimates_gpu_seconds=estimates, remaining_stage_gpu_seconds=remaining, per_device=per_role,
        protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"],
        parent_protocol_sha256=parent["sha256"], ledger_used_gpu_seconds=ledger["used_gpu_seconds"],
        scope="full current-source correctness, both candidates' full development matrix and one selected holdout; planning only",
        later_stages="migration and queue units remain independently budget-gated"))


def refinement_cost_plan(frozen, ledger):
    """Use measured predecessor costs, with CPU inputs already materialized.

    No predicted benefit of the no-op context optimization is discounted. Five
    seconds per installation is an explicit estimate, not a measured speedup.
    """
    if ledger.get("active"):
        raise ValueError("refinement planning requires a settled ledger")
    from .cfg_refinement import verify_refinement, verify_cpu_binding
    proof = verify_refinement(frozen["implementation_refinement"], config=frozen["config"], ledger=frozen["ledger"])
    verify_cpu_binding(frozen.get("cpu_embeddings"), config=frozen["config"], model_manifest=frozen["model_manifest"])
    root = Path(proof["parent_run"])
    parent = read_json(root / "protocol.json")
    rows = read_records(root, parent)
    units = {row["unit_id"]: read_json(root / "units" / (row["unit_id"] + ".json")) for row in rows}
    elapsed = lambda u: finite(u["ended_at"] - u["started_at"], "measured complete unit", positive=True)
    cleanup = finite(frozen["config"]["cleanup_wall_seconds"], "cleanup reserve", positive=True)
    installation = 5.
    per_role = {}
    for role in ("a100", "v100"):
        job = read_json(root / ("calibrate-" + role + "-job.json"))
        uuid = job["physical_devices"][0]["uuid"]
        costs, sessions = {}, {}
        for kind in ("correctness", "calibrate"):
            phase = kind + "-" + role
            phase_job = read_json(root / (phase + "-job.json"))
            sessions[kind] = next(s for s in ledger["sessions"] if s["session_id"] == phase_job["session_id"])
            costs[kind] = sum(elapsed(u) for name, u in units.items() if name.startswith(phase + "-"))
        setup = max(0., *(sessions[k]["wall_seconds"] - costs[k] for k in costs))
        local = [r for r in rows if r.get("device_role") == role]
        paired = [r for r in local if r.get("type") == "cfg_calibration" and r.get("recipe_id") == "paired_008"]
        max_pair = max(elapsed(units[r["unit_id"]]) for r in paired)
        off = max(finite(r["reference"]["elapsed_ms"], "measured off trajectory", positive=True) / 1000 for r in paired)
        correct = next(r for r in local if r.get("type") == "cfg_correctness" and r.get("recipe_id") == "paired_008")
        diagnostic = next(r for r in local if r.get("type") == "cfg_diagnostic_neutrality")
        details = [e.get("cfg_research", {}) for e in diagnostic["instrumented"]["events"] if e.get("kind") == "step"]
        if len(details) != 32 or any("diagnostic_ms" not in e for e in details):
            raise ValueError("refinement shadow reserve requires 32 measured diagnostic steps")
        shadow = sum(finite(e["diagnostic_ms"], "measured shadow overhead") for e in details) / 1000
        trajectories = elapsed(units[correct["unit_id"]])
        unit_bound = (trajectories + shadow) * 1.3
        correctness = unit_bound + (setup + installation) * 1.3 + cleanup
        calibration = (max_pair * 20 + 2 * (setup + installation)) * 1.3 + 2 * cleanup
        per_role[role] = dict(physical_uuid=uuid, correctness_gpu_seconds=correctness,
            calibration_gpu_seconds=calibration, correctness_unit_estimate_wall_seconds=unit_bound,
            measured_three_trajectory_unit_wall_seconds=trajectories,
            measured_shadow_wall_seconds=shadow, measured_max_pair_wall_seconds=max_pair,
            measured_off_trajectory_wall_seconds=off, measured_setup_wall_seconds=setup,
            estimated_installation_wall_seconds=installation, planning_setup_wall_seconds=setup + installation,
            correctness_units_seconds={"refinement-correctness-" + role: trajectories + shadow},
            formal_dev_pairs=16, held_out_pairs=4, startup_correctness_wall_seconds=correctness - cleanup)
    allocation = ledger["reallocations"][-1]
    remaining = {stage: max(0., allowance - (ledger["stage_used_gpu_seconds"].get(stage, 0.)
        - allocation["baseline_stage_used"].get(stage, 0.)))
        for stage, allowance in allocation["stage_allowances_gpu_seconds"].items()}
    estimates = {stage: sum(p[stage + "_gpu_seconds"] for p in per_role.values())
                 for stage in ("correctness", "calibration")}
    lifetime = ledger.get("limit_gpu_seconds", frozen["config"]["budget_gpu_seconds"]) - ledger["used_gpu_seconds"]
    fits = all(estimates[s] < remaining[s] for s in estimates) and sum(estimates.values()) < lifetime
    return seal(dict(status="fits" if fits else "budget_stopped", safety_factor=1.3,
        estimates_gpu_seconds=estimates, remaining_stage_gpu_seconds=remaining, per_device=per_role,
        protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"],
        parent_protocol_sha256=parent["sha256"], refinement_sha256=proof["sha256"],
        cpu_embeddings_manifest_sha256=frozen["cpu_embeddings"]["manifest_sha256"],
        ledger_used_gpu_seconds=ledger["used_gpu_seconds"], active_candidates=["paired_008"],
        installation_reserve_kind="estimate: five wall seconds per device and worker phase; not a measurement",
        optimization_savings_assumed_seconds=0.,
        scope="fresh three-trajectory correctness, paired-only 16 development pairs and four held-out pairs per device",
        later_stages="migration and queue units remain independently budget-gated"))


def _sealed(value, frozen=None, statuses=("completed",)):
    return (isinstance(value, dict)
            and value.get("sha256") == digest({k: v for k, v in value.items() if k != "sha256"})
            and value.get("status") in statuses
            and (frozen is None or value.get("protocol_sha256") == frozen["sha256"]
                 and value.get("source_sha256") == frozen["source_sha256"]))


def _completed(root, frozen, phase, rows, candidates):
    """A stale summary or a partially completed unit never erases future work."""
    summary_path, attempt_path = root / (phase + ".json"), root / (phase + "-attempt.json")
    if not summary_path.exists() or not attempt_path.exists():
        return False
    summary, attempt = read_json(summary_path), read_json(attempt_path)
    if (not _sealed(summary, frozen) or not _sealed(attempt)
            or attempt.get("protocol_sha256") != frozen["sha256"] or attempt.get("returncode") != 0):
        return False
    role = phase.rsplit("-", 1)[-1]
    selected = [r for r in rows if r.get("unit_id", "").startswith(phase + "-")]
    if not selected or any(r.get("performance_valid") is not True or r.get("status") != "completed" for r in selected):
        return False
    if phase.startswith("correctness-"):
        if frozen.get("implementation_refinement"):
            from .cfg_refinement import correctness_complete
            return correctness_complete(selected, summary, role)
        expected = {"off", "independent_008", "paired_008"}
        controls = [r for r in selected if r.get("type") == "cfg_correctness"]
        equivalence = [r for r in selected if r.get("type") == "cfg_implementation_equivalence"]
        diagnostic = [r for r in selected if r.get("type") == "cfg_diagnostic_neutrality"]
        return (summary.get("checks_passed") is True and len(controls) == 3
                and {r.get("recipe_id") for r in controls} == expected
                and len(equivalence) == len(diagnostic) == 1
                and len(selected) == 5
                and all(r.get("equal") is True and r.get("device_role") == role for r in selected))
    from .dual_matrix import inputs
    split = "dev" if phase.startswith("calibrate-") else "holdout"
    recipes = candidates if split == "dev" else [summary.get("selected")]
    if not recipes or any(recipe not in candidates for recipe in recipes):
        return False
    orders = lambda recipe: (("off", recipe), (recipe, "off")) if split == "dev" else (("off", recipe),)
    expected = {(recipe, order, item["prompt_id"], item["seed"])
                for recipe in recipes for order in orders(recipe) for item in inputs(frozen["config"], split)}
    keys = [(r.get("recipe_id"), tuple(r.get("order", [])), r.get("prompt_id"), r.get("seed")) for r in selected]
    return (len(keys) == len(expected) and set(keys) == expected
            and all(r.get("type") == "cfg_calibration" and r.get("phase") == split
                    and r.get("device_role") == role and r.get("split") == split for r in selected))


def pending_cost_plan(frozen, root, ledger, original_plan):
    """Re-admit every unfinished correctness/dev/holdout phase with fresh limits.

    The supervisor calls this before admission and again under its ledger lock,
    passing an ``active=None`` copy before Popen. Elapsed reservation time is
    subsequently covered by ``session.check``; this helper never writes a ledger
    or grants scientific qualification. Startup values exclude cleanup only.
    """
    if ledger.get("active"):
        raise ValueError("pending planning requires settled totals or the supervisor's locked pre-launch view")
    if not _sealed(original_plan, frozen, ("fits",)):
        raise ValueError("pending planning requires the original current-source complete-matrix admission")
    from .cfg_refinement import development_candidates
    candidates = development_candidates(frozen)
    root = Path(root)
    rows = read_records(root, frozen)
    valid_rows = [r for r in rows if r.get("performance_valid") is True]
    unit_ids = {r["unit_id"] for r in valid_rows}
    units = {name: read_json(root / "units" / (name + ".json")) for name in unit_ids}
    durations = {}
    for name, unit in units.items():
        if not _sealed(unit, frozen) or unit.get("performance_valid") is not True:
            raise ValueError("current planning measurements require complete valid units")
        durations[name] = finite(unit["ended_at"] - unit["started_at"], "current complete-unit wall time", positive=True)
    cleanup = finite(frozen["config"]["cleanup_wall_seconds"], "cleanup reserve", positive=True)
    phase_estimates, startups, completed, measured = {}, {}, [], {}
    for role in ("a100", "v100"):
        baseline = original_plan["per_device"][role]
        pair = finite(baseline["measured_max_pair_wall_seconds"], "prototype paired unit", positive=True)
        setup = finite(baseline.get("planning_setup_wall_seconds", baseline["measured_setup_wall_seconds"]), "planned setup")
        current = [r for r in valid_rows if r.get("device_role") == role]
        # Instrumented correctness is deliberately an upper planning bound.
        # Diagnostic and oracle runs can never become production timing data.
        for row in current:
            if row.get("type") == "cfg_calibration" and row.get("phase") in ("dev", "holdout"):
                pair = max(pair, durations[row["unit_id"]])
            elif row.get("type") == "cfg_correctness" and not frozen.get("implementation_refinement"):
                for field in ("continuous", "restored"):
                    run = row.get(field, {})
                    if run.get("identity", {}).get("implementation") not in ("native", "split_prefix"):
                        continue
                    seconds = finite(run.get("elapsed_ms"), "current correctness trajectory", positive=True) / 1000
                    pair = max(pair, 2 * seconds)
        # A fully settled phase measures model load, encoding preparation and
        # cleanup outside units. Later phases reserve the largest such cost.
        uuid = baseline.get("physical_uuid")
        for phase_kind, stage in (("correctness", "correctness"), ("calibrate", "calibration"), ("holdout", "calibration")):
            phase = phase_kind + "-" + role
            attempt_path = root / (phase + "-attempt.json")
            attempt = read_json(attempt_path) if attempt_path.exists() else {}
            if _completed(root, frozen, phase, rows, candidates):
                completed.append(phase)
            sessions = [s for s in ledger.get("sessions", [])
                        if s.get("protocol_sha256") == frozen["sha256"] and s.get("stage") == stage
                        and s.get("devices") == [uuid] and s.get("status") == "completed"
                        and s.get("worker_pgid") == attempt.get("worker_pgid")]
            # Supervisor attempts historically omit worker_pgid. Job metadata
            # binds a phase to the ledger session without guessing by stage.
            job_path = root / (phase + "-job.json")
            if job_path.exists():
                job = read_json(job_path)
                sessions = [s for s in ledger.get("sessions", [])
                            if s.get("session_id") == job.get("session_id")
                            and s.get("protocol_sha256") == frozen["sha256"] and s.get("stage") == stage
                            and s.get("devices") == [uuid] and s.get("status") == "completed"]
            if len(sessions) == 1 and phase in completed:
                unit_wall = sum(value for name, value in durations.items() if name.startswith(phase + "-"))
                setup = max(setup, finite(sessions[0]["wall_seconds"], "settled phase cost") - unit_wall)
        correction_units = dict(baseline.get("correctness_units_seconds", {}))
        for name, value in durations.items():
            prefix = "correctness-" + role + "-"
            if name.startswith(prefix):
                label = name[len(prefix):]
                correction_units[label] = max(correction_units.get(label, 0.), value)
        correctness = finite(baseline["correctness_gpu_seconds"], "original complete correctness", positive=True)
        if correction_units:
            correctness = max(correctness, (sum(correction_units.values()) + setup) * 1.3 + cleanup)
        for kind, estimate in (("correctness", correctness),
                ("calibrate", (len(candidates) * 8 * 2 * pair + setup) * 1.3 + cleanup),
                ("holdout", (4 * pair + setup) * 1.3 + cleanup)):
            phase = kind + "-" + role
            phase_estimates[phase] = estimate
            startups[phase] = max(1e-6, estimate - cleanup)
        measured[role] = dict(pair_wall_seconds=pair, setup_wall_seconds=setup,
                              correctness_unit_seconds=correction_units, physical_uuid=uuid)
    allocation = ledger["reallocations"][-1]
    remaining = {stage: max(0., allowance - (ledger["stage_used_gpu_seconds"].get(stage, 0.)
        - allocation["baseline_stage_used"].get(stage, 0.)))
        for stage, allowance in allocation["stage_allowances_gpu_seconds"].items()}
    pending = {stage: sum(value for phase, value in phase_estimates.items() if phase not in completed
                         and (stage == "correctness") == phase.startswith("correctness-"))
               for stage in ("correctness", "calibration")}
    lifetime = ledger.get("limit_gpu_seconds", frozen["config"]["budget_gpu_seconds"]) - ledger["used_gpu_seconds"]
    fits = (all(value == 0 or value < remaining[stage] for stage, value in pending.items())
            and (sum(pending.values()) == 0 or sum(pending.values()) < lifetime))
    return seal(dict(status="fits" if fits else "budget_stopped", safety_factor=1.3,
        stage_remaining_gpu_seconds=remaining, pending_stage_estimates_gpu_seconds=pending,
        phase_estimates_gpu_seconds=phase_estimates, phase_startup_wall_seconds=startups,
        completed_phases=sorted(completed), measured_per_device=measured,
        protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"],
        original_plan_sha256=original_plan["sha256"], ledger_used_gpu_seconds=ledger["used_gpu_seconds"],
        lifetime_remaining_gpu_seconds=lifetime,
        scope="unfinished complete correctness, development and held-out matrices; no qualification inheritance"))
