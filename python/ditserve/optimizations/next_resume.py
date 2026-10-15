"""One fixed cross-model recovery regression, before a Sana candidate goes online.

The existing paired-cache resume implementation owns all numerical work. This
wrapper preserves the selected Sana candidate, bounds the evidence denominator,
and retains its diagnostic arrays for the final archive. It is not another full
cache-quality qualification or an online/performance measurement.
"""

import time

from . import wan_phases as phases


PHASE = "sana-wan-resume"
INPUT_ID = "kettle-101"
INTERRUPT_STEPS = (0, 8, 16)
VIDEO_SHAPE = [1, 81, 480, 832, 3]


def _gate(result, candidate):
    """Require the complete fixed regression, not merely a helper success bit."""
    steps = result.get("steps", [])
    interruptions = result.get("interruptions", [])
    paths = result.get("cache_paths", {})
    final = result.get("final", {})
    artifacts = result.get("artifacts", [])
    checks = dict(
        helper_passed=result.get("passed") is True,
        complete_steps=len(steps) == 32 and [r.get("step") for r in steps] == list(range(1, 33)),
        state_checks=bool(steps) and all(all(row.get(key) is True for key in
            ("latent", "solver", "cache_state", "rng", "passed", "cache_paths_equal")) for row in steps),
        actual_candidate_insertions=[r.get("step") for r in interruptions] == list(INTERRUPT_STEPS)
            and all(r.get("image_service", {}).get("candidate") == candidate
                    and r.get("image_service", {}).get("finite") is True for r in interruptions),
        final_video=final.get("passed") is True and final.get("finite") is True
            and final.get("rtol") == 1e-5 and final.get("atol") == 1e-6
            and final.get("shape") == VIDEO_SHAPE,
        cache_paths=paths.get("valid") is True and paths.get("paths_equal") is True
            and all(paths.get(name, {}).get("valid") is True
                    and paths[name].get("required_steps") == 32
                    and paths[name].get("observed_steps") == 32 for name in ("continuous", "k4")),
        retained_videos=len(artifacts) == 2 and all(a.get("shape") == VIDEO_SHAPE
            and a.get("raw_available") is True for a in artifacts),
    )
    return dict(passed=all(checks.values()), checks=checks,
        failed_checks=[name for name, passed in checks.items() if not passed],
        step_count=len(steps), real_sana_insertions=len(interruptions),
        scope="one fixed input; cache full/reuse paths are observed, not a new full-quality qualification")


def _cleanup(worker, executor, bank):
    """The caller admitted an empty worker, so only this regression owns states."""
    errors = []
    if executor is not None:
        for state in list(executor._states.values()):
            try:
                if not state.closed:
                    executor.close(state)
            except BaseException as error:
                errors.append("continuation: " + type(error).__name__ + ": " + str(error))
    # Also catches a successfully installed first state when preparing the
    # second fails before _resume_check's tuple assignment has completed.
    for state in list(worker._states.values()):
        try:
            if state.residency != "released":
                worker.release(state)
        except BaseException as error:
            errors.append("request: " + type(error).__name__ + ": " + str(error))
    if bank is not None:
        bank.entries.clear()
    return dict(passed=not errors and not worker._states
                and (executor is None or not executor._states), errors=errors,
        worker_live_states_after=len(worker._states),
        executor_live_states_after=len(executor._states) if executor is not None else 0,
        prompt_bank_entries_after=len(bank.entries) if bank is not None else 0)


def run_candidate_resume(context):
    """Run kettle seed 101 with the already-selected, already-warmed candidate.

    Caller owns model residency, GPU supervision/accounting and local Sana
    qualification. This function neither changes the candidate nor promotes an
    online/default backend. Exceptions preserve partial evidence and propagate.
    """
    evidence = phases._Evidence(context, PHASE)
    worker, adapter = context.worker, context.sana
    candidate = getattr(adapter, "candidate", None)
    bank = executor = None
    admitted = False
    error_in_flight = False
    started = time.perf_counter()
    evidence.data.update(schema_version="ditserve.optimizations.next_resume.v1",
        candidate=candidate, resume_qualified=False, eligible=False,
        online_qualified=False, cache_quality_qualified=False,
        fixed_protocol=dict(input_id=INPUT_ID, seed=101, video_shape=VIDEO_SHAPE,
            steps=32, policy="paired_008", threshold=.08, max_continuous=1,
            continuous_quantum=60, resumed_quantum=4, real_sana_steps=list(INTERRUPT_STEPS),
            rtol=1e-5, atol=1e-6),
        scope="one fixed cross-model interference/recovery regression; not full cache quality, speed or online qualification",
        retention_policy=dict(raw_arrays="retain_until_final_archive",
            may_delete_now=False, dependency="final source-bound archive and explicit retention decision"))
    evidence.save()
    try:
        if worker._states:
            raise ValueError("candidate resume requires no live request state")
        admitted = True
        if candidate not in ("vae_compile", "transformer_graph"):
            raise ValueError("candidate resume requires an already-selected new Sana candidate")
        phases._validate(context)
        context.check_budget()
        phases._ensure(context, 600.)
        matches = [item for item in phases._cache_inputs() if item["input_id"] == INPUT_ID]
        if len(matches) != 1 or matches[0]["seed"] != 101:
            raise ValueError("fixed kettle seed 101 input is absent or ambiguous")
        item = matches[0]
        bank = phases.FrozenPromptBank(worker)
        executor = phases.PairedCacheExecutor(worker, phases.BlockBackend(worker, compiled=False))
        evidence.data.update(input=item, candidate_before=adapter.candidate_summary(),
                             backend="eager-paired_008")
        evidence.begin(kind="same_backend_resume", input_id=INPUT_ID, backend="eager-paired_008",
                       candidate=candidate)
        # The candidate must have been warmed during its local qualification;
        # a new graph/compile here is an error, not hidden recovery setup work.
        with adapter.steady_context():
            result = phases._resume_check(context, evidence, bank, item, executor, cache=True)
        evidence.append(dict(kind="same_backend_resume", input_id=INPUT_ID, candidate=candidate, result=result))
        gate = _gate(result, candidate)
        unchanged = context.sana is adapter and adapter.candidate == candidate
        gate["checks"]["selected_candidate_unchanged"] = unchanged
        if not unchanged:
            gate["failed_checks"].append("selected_candidate_unchanged")
            gate["passed"] = False
        evidence.finish(status="passed" if gate["passed"] else "failed",
            passed=gate["passed"], resume_qualified=gate["passed"], eligible=gate["passed"],
            gate=gate, candidate_after=adapter.candidate_summary(),
            reason="fixed candidate interference regression passed" if gate["passed"] else
                   "fixed candidate interference regression failed or incomplete",
            retention=result.get("artifacts", []))
    except BaseException as error:
        error_in_flight = True
        evidence.interrupted(error)
        evidence.data.update(passed=False, resume_qualified=False, eligible=False)
        raise
    finally:
        manifest_error = None
        if bank is not None:
            try:
                evidence.data["prompt_bank"] = bank.manifest()
            except BaseException as error:
                manifest_error = error
                evidence.data["prompt_bank_error"] = type(error).__name__ + ": " + str(error)
        cleanup = (_cleanup(worker, executor, bank) if admitted else
                   dict(passed=False, skipped=True, reason="pre-existing states are not owned by this regression"))
        evidence.data.update(cleanup=cleanup, elapsed_ms=(time.perf_counter()-started)*1000,
            elapsed_scope="charged host wall for encoding, correctness diagnostics, decode, artifact IO and cleanup; not performance")
        if not cleanup["passed"] or manifest_error is not None:
            evidence.data.update(status="failed", passed=False, resume_qualified=False, eligible=False)
        evidence.save()
        if manifest_error is not None and not error_in_flight:
            raise manifest_error
        if admitted and not cleanup["passed"] and not error_in_flight:
            raise RuntimeError("candidate resume private-state cleanup failed")
    return evidence.data
