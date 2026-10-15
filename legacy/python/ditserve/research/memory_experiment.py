"""Natural two-request capacity trials, separate from nominal storage savings."""

import copy
import gc

from .budget import BudgetStop
from .evidence import memory_admission_gate, qualify_trajectory


def run_memory_admission(exp, prefix, shape, candidate):
    """Try one fixed A/B pair after quality selection, without artificial pressure.

    Cache-off A and B must finish alone. Keep is attempted first, and only an
    urgent-request OOM while A is paused admits a fresh release-action trial.
    An OOM traceback is never retained across that retry: only JSON observations
    leave each exception block, followed by collection before the next prepare.
    This is a capacity observation, not a throughput or statistical claim.
    """
    from .experiment import AdmissionCapacityError, IneligibleAction

    config = exp.config
    action = candidate["action"]["kind"]
    at_step = candidate["action"]["at_step"]
    if action not in ("hot", "exact_refresh", "cold") or at_step != config["transition_step"]:
        raise ValueError("memory admission requires a selected state-release action at the frozen step")
    holdout = [p for p in config["prompts"] if p["split"] == "holdout"]
    dev = [p for p in config["prompts"] if p["split"] == "dev"]
    prompt, seed = holdout[0], config["holdout_seeds"][0]
    urgent_prompt, urgent_seed = dev[1], config["dev_seeds"][1]
    off = next(r for r in config["recipes"] if r["id"] == "off")
    trial_id = "memory-%s-%dx%d-%df" % (candidate["candidate_id"], shape["height"], shape["width"], shape["frames"])
    trial = dict(type="memory_trial", case_id=trial_id, trial_id=trial_id,
                 candidate_id=candidate["candidate_id"], shape=copy.deepcopy(shape),
                 status="inconclusive", measurement_kind="measured_gpu", keep_rejected=False,
                 artificially_filled_memory=False, single_request_oom=False, released_mode=action,
                 input_pair=dict(main_prompt_id=prompt["id"], main_seed=seed,
                                 urgent_prompt_id=urgent_prompt["id"], urgent_seed=urgent_seed),
                 scope="one_natural_capacity_pair_full_reference_quality_not_a_statistical_claim")

    def finish(reason=None):
        if reason is not None:
            trial["reason"] = reason
        trial["gate"] = memory_admission_gate([trial], epsilon=config["epsilon"])
        trial["supported"] = trial["gate"]["supported"]
        return exp.record(trial)

    def attempt(name, status, **values):
        return exp.record(dict(type="memory_attempt", case_id=trial_id + "-" + name,
                               trial_id=trial_id, phase=name, status=status,
                               measurement_kind="measured_gpu", **values))

    try:
        # Reserve a complete unit before either reference, keep, or release arm.
        exp.check_unit()
    except BudgetStop:
        return finish("complete_capacity_unit_does_not_fit_budget")

    references = []
    for name, input_prompt, input_seed in (("main_reference", prompt, seed),
                                           ("urgent_reference", urgent_prompt, urgent_seed)):
        spec = exp.spec(input_prompt, input_seed, shape, off)
        failure = None
        try:
            run = exp.generate(spec)
        except exp.worker.torch.OutOfMemoryError as error:
            failure = dict(error=str(error))
        except BudgetStop as error:
            failure = dict(error=str(error), budget_stopped=True)
        if failure is not None:
            gc.collect()
            trial["single_request_oom"] = not failure.get("budget_stopped", False)
            attempt(name, "insufficient" if failure.get("budget_stopped") else "single_request_oom", **failure)
            return finish("standalone_reference_did_not_complete")
        if run.get("status") != "completed":
            attempt(name, "incomplete", run=exp.serializable(run))
            return finish("standalone_reference_did_not_complete")
        identity = exp.identity(spec, off)
        references.append((run, identity))
        attempt(name, "completed", identity=identity, run=exp.serializable(run))

    main_reference, urgent_reference = references
    keep_spec = exp.spec(prompt, seed, shape, prefix)
    urgent_spec = exp.spec(urgent_prompt, urgent_seed, shape, off)
    urgent_identity = exp.identity(urgent_spec, off)
    trial["keep_request_identity"] = urgent_identity
    observation, keep_failure = {}, None
    try:
        kept = exp.generate(keep_spec, action="keep", suffix=prefix, at_step=at_step,
                            urgent_spec=urgent_spec, admission=observation)
    except AdmissionCapacityError as error:
        keep_failure = dict(kind="urgent_capacity", observation=copy.deepcopy(error.observation))
    except exp.worker.torch.OutOfMemoryError as error:
        keep_failure = dict(kind="main_capacity", error=str(error))
    except (BudgetStop, IneligibleAction) as error:
        keep_failure = dict(kind="incomplete", error=str(error))
    if keep_failure is None:
        attempt("keep", "completed", observation=observation, run=exp.serializable(kept))
        if (kept.get("status") == "completed" and (kept.get("urgent") or {}).get("status") == "completed"
                and observation.get("main_completed") is True and observation.get("urgent_decode_completed") is True):
            trial.update(status="no_admission_difference", keep_observation=observation)
            return finish("keep_already_completes_both_real_requests")
        return finish("keep_execution_incomplete")
    gc.collect()  # No failed model frames/exception traceback survive the retry.
    attempt("keep", keep_failure["kind"], **{k: v for k, v in keep_failure.items() if k != "kind"})
    if keep_failure["kind"] != "urgent_capacity":
        trial["single_request_oom"] = keep_failure["kind"] == "main_capacity"
        return finish("keep_failure_is_not_an_urgent_capacity_rejection")
    keep_observation = keep_failure["observation"]
    if (keep_observation.get("phase") != "urgent" or keep_observation.get("at_step") != at_step
            or keep_observation.get("action") != "keep" or keep_observation.get("urgent_capacity_failed") is not True
            or keep_observation.get("transfer_complete") is not True
            or keep_observation.get("artificially_filled_memory") is not False
            or keep_observation.get("single_request_oom") is not False):
        return finish("keep_capacity_observation_is_incomplete_or_not_natural")
    trial.update(keep_rejected=True, keep_rejection_reason="measured_capacity", keep_observation=keep_observation)

    spec = exp.spec(prompt, seed, shape, prefix)
    urgent_spec = exp.spec(urgent_prompt, urgent_seed, shape, off)
    identity = exp.identity(spec, prefix, action, candidate["suffix"], at_step=at_step)
    admitted_identity = exp.identity(urgent_spec, off)
    observation, release_failure = {}, None
    try:
        released = exp.generate(spec, action=action, suffix=candidate["suffix"], at_step=at_step,
                                urgent_spec=urgent_spec, admission=observation)
    except AdmissionCapacityError as error:
        release_failure = dict(kind="urgent_capacity", observation=copy.deepcopy(error.observation))
    except exp.worker.torch.OutOfMemoryError as error:
        release_failure = dict(kind="main_capacity", error=str(error))
    except (BudgetStop, IneligibleAction) as error:
        release_failure = dict(kind="incomplete", error=str(error))
    if release_failure is not None:
        gc.collect()
        attempt(action, release_failure["kind"], **{k: v for k, v in release_failure.items() if k != "kind"})
        trial["single_request_oom"] = release_failure["kind"] == "main_capacity"
        return finish("release_arm_did_not_complete_both_requests")
    attempt(action, "completed", observation=observation, run=exp.serializable(released))
    if (released.get("status") != "completed" or (released.get("urgent") or {}).get("status") != "completed"
            or observation.get("main_completed") is not True):
        return finish("release_arm_did_not_complete_both_requests")
    if (observation.get("action") != action or observation.get("at_step") != at_step
            or observation.get("artificially_filled_memory") is not False
            or observation.get("single_request_oom") is not False):
        return finish("release_observation_does_not_match_the_natural_trial")
    main_evidence = exp.quality(released, main_reference[0], identity, main_reference[1])
    urgent_evidence = exp.quality(released["urgent"], urgent_reference[0], admitted_identity, urgent_reference[1])
    trial.update(status="completed", before=observation.get("before", {}), after=observation.get("after", {}),
                 transfer_complete=observation.get("transfer_complete") is True,
                 admitted=dict(identity=admitted_identity, evidence=urgent_evidence,
                               allocation_succeeded=observation.get("urgent_allocation_succeeded") is True,
                               compute_completed=observation.get("urgent_compute_completed") is True,
                               decode_completed=observation.get("urgent_decode_completed") is True),
                 resumed=dict(identity=identity, evidence=main_evidence),
                 quality=dict(main=qualify_trajectory(main_evidence, identity, epsilon=config["epsilon"]),
                              urgent=qualify_trajectory(urgent_evidence, admitted_identity, epsilon=config["epsilon"])))
    return finish()
