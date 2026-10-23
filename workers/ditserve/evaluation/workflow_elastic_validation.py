"""Opt-in two-GPU accuracy gates for Wan block pause and elastic ownership.

This is diagnostic work, never performance evidence. Ordinary SP1 references
are produced before attaching any distributed processor. Each dependent gate
requires its predecessor; a failed gate leaves later work explicitly unrun.
"""

import argparse
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import tempfile
import traceback

from .workflow_overlap import _hash_array, atomic_json
from .workflow_serving_overlap import normalize_config as overlap_config, request_spec
from .workflow_serving_sp import comparison_passes, _cpu_array


SCHEMA = "ditserve.workflow.elastic_validation.v1"
GATES = ("sp1_resume", "sp1_transfer", "fixed_sp2", "fixed_sp2_pause",
         "elastic_1_2_1", "sp2_sana_interleave")
STAGE_GATES = dict(all=GATES, transfer=("sp1_resume", "sp1_transfer"),
                   pause=("fixed_sp2", "fixed_sp2_pause"),
                   resize=("elastic_1_2_1",),
                   interleave=("fixed_sp2", "sp2_sana_interleave"))


def normalize_config(config):
    value = overlap_config(config)
    defaults = dict(devices=["cuda:0", "cuda:1"], frames=[17, 33],
                    seeds=[20261021, 20261022], timeout_seconds=5400,
                    group_timeout_seconds=60, threshold=0.05,
                    stage="all", sana_model_path="/models/sana-sprint-0.6b",
                    sana_prompt="A red bicycle leaning against a blue wall.",
                    interleave_steps=[0, 5, 10, 15],
                    schedule=[{"step": 5, "degree": 2}, {"step": 10, "degree": 1}],
                    pause_blocks=[23, 37], sp_linear_policy="native")
    for key, default in defaults.items():
        value.setdefault(key, default)
    if (not isinstance(value["devices"], list) or len(value["devices"]) != 2 or
            len(set(value["devices"])) != 2 or any(
                not str(device).startswith("cuda:") or not str(device)[5:].isdigit()
                for device in value["devices"])):
        raise ValueError("elastic validation requires two distinct cuda:N devices")
    if (not isinstance(value["frames"], list) or not value["frames"] or
            any(type(frame) is not int or frame < 1 or (frame-1) % 4 for frame in value["frames"])):
        raise ValueError("frames must contain frames=4*k+1")
    if (not isinstance(value["seeds"], list) or not value["seeds"] or
            len(set(value["seeds"])) != len(value["seeds"]) or
            any(type(seed) is not int or seed < 0 for seed in value["seeds"])):
        raise ValueError("seeds must contain distinct nonnegative integers")
    if value["threshold"] != 0.05:
        raise ValueError("elastic numerical threshold is fixed at 0.05")
    if value["schedule"] != defaults["schedule"]:
        raise ValueError("qualification fixes transitions at completed steps 5 and 10")
    if (not isinstance(value["pause_blocks"], list) or len(value["pause_blocks"]) < 2 or
            any(type(count) is not int or count < 1 for count in value["pause_blocks"]) or
            sum(value["pause_blocks"]) != 60):
        raise ValueError("pause_blocks must partition the 60 CFG transformer blocks")
    if type(value["timeout_seconds"]) is not int or value["timeout_seconds"] < 1:
        raise ValueError("timeout_seconds must be positive")
    if (type(value["group_timeout_seconds"]) not in (int, float) or
            not 0 < value["group_timeout_seconds"] <= 60):
        raise ValueError("group_timeout_seconds must be in (0, 60]")
    if value["stage"] not in STAGE_GATES:
        raise ValueError("unknown elastic validation stage")
    if value["sp_linear_policy"] not in ("native", "full_rows_except_ffn1"):
        raise ValueError("elastic validation supports native or full_rows_except_ffn1 linear policy")
    if (not isinstance(value["interleave_steps"], list) or not value["interleave_steps"] or
            len(set(value["interleave_steps"])) != len(value["interleave_steps"]) or
            any(type(step) is not int or not 0 <= step < 20 for step in value["interleave_steps"])):
        raise ValueError("interleave_steps must select distinct steps in the 20-step schedule")
    return value


def gate_order(previous, gates=GATES):
    """Return the next gate only after a complete passed predecessor chain."""
    for index, row in enumerate(previous):
        if index >= len(gates) or row.get("gate") != gates[index] or row.get("passed") is not True:
            return None
    return gates[len(previous)] if len(previous) < len(gates) else None


def _sana_image(backend, prompt, seed, key):
    """A real eager Sana request; its output is retained for exact checking."""
    backend.prepare(dict(state_id=key, spec=dict(prompt=prompt, seed=seed)))
    try:
        backend.advance(dict(state_id=key))
        output = backend.states[key]["output"]
        receipt = backend.finish(dict(state_id=key, output_mode="digest", id=0))
        return output, receipt
    finally:
        backend.release(dict(state_id=key))


def _world_pass(torch, device, passed):
    value = torch.tensor(int(bool(passed)), device=device, dtype=torch.int32)
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN)
    return bool(value.item())


def _spec(config, frames, seed, case):
    value = request_spec(dict(config, seed=seed), frames, "A", case)
    return value


def _bind_cpu(config, device):
    from .workflow_overlap import bind_cpu_to_gpu
    mask = config.get("cpu_masks", {}).get(device)
    if mask is None:
        return bind_cpu_to_gpu(device, config["cpu_threads"])
    allowed = sorted(os.sched_getaffinity(0))
    if (not isinstance(mask, list) or len(mask) != config["cpu_threads"] or
            len(set(mask)) != len(mask) or any(type(cpu) is not int or cpu < 0 for cpu in mask)
            or not set(mask).issubset(allowed)):
        raise ValueError("cpu_masks[device] must select available distinct inference CPUs")
    for task in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(task.name), mask)
        except ProcessLookupError:
            pass
    return dict(policy="explicit", device=device, bound_cpus=sorted(mask), allowed_before=allowed)


def _ordinary_reference(runtime, spec, destination, *, save, save_video=False, video_path=None):
    import numpy as np
    destination.mkdir(parents=True, exist_ok=True)
    state = runtime.prepare(spec)
    result = dict(spec=spec, steps=[], performance_evidence=False)
    try:
        result["initial"] = _hash_array(_cpu_array(state.latents))
        result["timesteps"] = [float(v) for v in state.scheduler.timesteps.detach().cpu().tolist()]
        if len(result["timesteps"]) != 20:
            raise RuntimeError("ordinary reference scheduler must retain 20 steps")
        for step in range(20):
            runtime.step(state)
            latent = _cpu_array(state.latents)
            result["steps"].append(_hash_array(latent))
            if save:
                np.save(destination / f"step-{step:02d}.npy", latent, allow_pickle=False)
        video = runtime.finish(state)
        result["video"] = _hash_array(video)
        if save and (save_video or video_path is not None):
            path = Path(video_path) if video_path is not None else destination / "video.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, video, allow_pickle=False)
        result["passed"] = (result["initial"]["finite"] and result["video"]["finite"] and
                            all(item["finite"] for item in result["steps"]))
        atomic_json(destination / "reference.json", result)
        return result
    finally:
        runtime.release(state)


def _transition(backend, state_id, source, target, epoch, destination, transitions,
                *, rank, completed_steps, audit=False):
    request = dict(action="transition", state_id=state_id,
                   transfer_id=destination.name, source_group=list(source),
                   target_group=list(target), expected_epoch=epoch, path=str(destination))
    started = time.perf_counter_ns()
    response = backend.distributed(request)
    row = dict(request=request, response=response, completed_steps=completed_steps,
               diagnostic_wall_ms=(time.perf_counter_ns()-started)/1e6)
    transitions.append(row)
    if audit:
        progress = response.get("progress")
        owner = rank in target
        digest = response.get("state_sha256", "")
        row["audit"] = dict(
            finalized=response.get("phase") == "finalized" and response.get("committed") is True,
            identity=response.get("state_id") == state_id and response.get("transfer_id") == destination.name,
            epoch_and_group=response.get("ownership_epoch") == epoch + 1 and response.get("group") == list(target),
            payload_digest=isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            transfer_ready=response.get("transfer_ready") is True,
            execution_contract=response.get("execution_contract") == dict(
                schema="ditserve.wan-sp-execution.v1", sp_linear_policy="full_rows_except_ffn1"),
            owner_state=(state_id in backend.states) == owner,
            owner_continuation=(state_id in backend.continuations) == owner,
            owner_record=(state_id in backend.elastic.records) == owner,
            progress=(isinstance(progress, dict) and progress.get("step") == completed_steps
                      and progress.get("ownership_epoch") == epoch + 1
                      and progress.get("group") == list(target)
                      and progress.get("transfer_ready") is True) if owner else progress is None)
        row["passed"] = all(row["audit"].values())
        if not row["passed"]:
            raise RuntimeError("elastic transition did not prove committed ownership and source release")
    return epoch + 1


def _quantum_audit(response, config, *, step, group, epoch, sequence, blocks, completed,
                   request_id=None, frames=None):
    """Require real block-path evidence; successful output hashes alone are insufficient."""
    events, progress = response.get("events", []), response.get("progress", {})
    event = events[0] if len(events) == 1 else {}
    repair = event.get("linear_repair", {})
    calls = repair.get("applied_calls")
    global_sequence, local_sequence = repair.get("global_sequence"), repair.get("local_sequence")
    checks = dict(
        block_path=len(events) == 1 and event.get("kind") == "block_quantum",
        step=event.get("step_index") == step and event.get("step_completed") is completed,
        block_count=event.get("blocks") == blocks,
        progress=progress.get("step") == step + int(completed),
        ownership=progress.get("group") == list(group) and progress.get("ownership_epoch") == epoch,
        sequence=progress.get("quantum_seq") == sequence + 1,
        policy=repair.get("policy") == config["sp_linear_policy"],
        degree=repair.get("degree") == len(group),
        repair_identity=repair.get("step_index") == step and repair.get("quantum_seq") == sequence
                        and (request_id is None or repair.get("request_id") == request_id),
        geometry=type(global_sequence) is int and type(local_sequence) is int
                 and global_sequence == local_sequence * len(group) and local_sequence > 0,
        sp1_passthrough=len(group) == 2 or repair.get("unchanged_sp1_calls", 0) > 0,
        applied=(type(calls) is int and (calls > 0 if len(group) == 2 else calls == 0)))
    if frames is not None:
        checks["request_geometry"] = global_sequence == ((frames-1)//4+1) * (config["height"]//16) * (config["width"]//16)
    return dict(passed=all(checks.values()), checks=checks)


def _replica_audit(torch, rank, group, local_hash):
    peers = [None, None]
    torch.distributed.all_gather_object(peers, dict(rank=rank, owner=rank in group, latent=local_hash))
    valid = (len(peers) == 2 and all(isinstance(row, dict) for row in peers)
             and {row.get("rank") for row in peers} == {0, 1}
             and all(row.get("owner") == (row.get("rank") in group) for row in peers))
    owners = [row for row in peers if isinstance(row, dict) and row.get("owner")]
    exact = (valid and len(owners) == len(group) and owners[0].get("latent") is not None
             and all(row.get("latent") == owners[0]["latent"] for row in owners))
    return dict(passed=bool(exact), replicas=peers)


def _validate_repaired_result(result):
    """Recheck the complete repaired-stage matrix from durable evidence on CPU.

    This validates one stage only. Cross-job transfer -> pause -> resize ordering
    and input/source bindings remain the study runner's responsibility.
    """
    config = result.get("configuration", {})
    errors = []
    def require(condition, label):
        if not condition:
            errors.append(label)
    policy, stage = config.get("sp_linear_policy"), config.get("stage")
    require(policy == "full_rows_except_ffn1", "explicit_conservative_policy")
    require(stage in STAGE_GATES, "known_stage")
    require(config.get("steps") == 20 and config.get("threshold") == .05, "fixed_numerical_gate")
    require(config.get("frames") == [17, 33] and len(set(config.get("seeds", []))) == 2,
            "two_seed_17_33_frame_matrix")
    gates = STAGE_GATES.get(stage, ())
    ranks = result.get("ranks", [])
    require(len(ranks) == 2 and {rank.get("rank") for rank in ranks} == {0, 1}, "both_ranks")
    expected_cases = [(seed, frames) for seed in config.get("seeds", []) for frames in config.get("frames", [])]
    for rank in ranks:
        prefix = f"rank{rank.get('rank')}"
        require(rank.get("status") == "completed" and rank.get("qualification", {}).get("passed") is True,
                prefix + ":completed")
        require(rank.get("backend_closed") is True and rank.get("process_lifecycle") == dict(exitcode=0, reaped=True),
                prefix + ":clean_process_exit")
        require(rank.get("ordinary_reference_before_distributed") is True, prefix + ":ordinary_reference")
        require(rank.get("backend_init", {}).get("sp_linear_policy") == policy, prefix + ":backend_policy")
        cases = rank.get("cases", [])
        require([(case.get("seed"), case.get("frames")) for case in cases] == expected_cases,
                prefix + ":case_matrix")
        for case in cases:
            label = prefix + ":" + str(case.get("case"))
            ordinary = case.get("ordinary", {})
            require(case.get("cross_card_sp1_exact") is True and case.get("passed") is True,
                    label + ":ordinary_cross_card_exact")
            require(ordinary.get("passed") is True and len(ordinary.get("steps", [])) == 20
                    and len(ordinary.get("timesteps", [])) == 20, label + ":ordinary_complete")
            rows = case.get("gates", [])
            require([row.get("gate") for row in rows] == list(gates), label + ":gate_chain")
            for row in rows:
                gate = row.get("gate")
                item = label + ":" + str(gate)
                require(row.get("passed") is True and row.get("first_failure") is None,
                        item + ":passed")
                require(row.get("sp_linear_policy") == policy and row.get("backend_evidence_required") is True,
                        item + ":policy")
                identity = row.get("initial_identity", {})
                initial_owner = rank["rank"] == 0 or gate in ("sp1_resume", "fixed_sp2", "fixed_sp2_pause", "sp2_sana_interleave")
                require(row.get("initial_and_schedule_exact") is True and identity.get("checked") is initial_owner,
                        item + ":initial_identity")
                require(identity.get("reference_initial") == ordinary.get("initial") and
                        identity.get("reference_timesteps") == ordinary.get("timesteps"), item + ":reference_identity")
                if initial_owner:
                    require(identity.get("actual_initial") == identity.get("reference_initial") and
                            identity.get("actual_timesteps") == identity.get("reference_timesteps"), item + ":initial_exact")
                release = row.get("release_audit", {})
                require(set(release) == {"states_cleared", "continuations_cleared", "ownership_cleared"}
                        and all(value is True for value in release.values()), item + ":release")
                transitions = row.get("transitions", [])
                expected_transitions = ([] if gate == "sp1_resume" else [5, 10]
                                        if gate in ("sp1_transfer", "elastic_1_2_1") else [0])
                require([entry.get("completed_steps") for entry in transitions] == expected_transitions,
                        item + ":transition_steps")
                require(all(entry.get("passed") is True and entry.get("audit")
                            and all(value is True for value in entry["audit"].values())
                            for entry in transitions), item + ":transitions")
                for index, transition in enumerate(transitions):
                    middle = [1] if gate == "sp1_transfer" else [0, 1]
                    source, target = ([0], middle) if index == 0 else (middle, [0])
                    request, response = transition.get("request", {}), transition.get("response", {})
                    digest = response.get("state_sha256", "")
                    require(request.get("source_group") == source and request.get("target_group") == target
                            and request.get("expected_epoch") == index, item + f":transition{index}:request")
                    require(response.get("phase") == "finalized" and response.get("committed") is True
                            and response.get("ownership_epoch") == index+1 and response.get("group") == target
                            and response.get("state_id") == request.get("state_id")
                            and response.get("transfer_id") == request.get("transfer_id")
                            and response.get("execution_contract") == dict(
                                schema="ditserve.wan-sp-execution.v1", sp_linear_policy=policy)
                            and isinstance(digest, str) and len(digest) == 64
                            and all(char in "0123456789abcdef" for char in digest),
                            item + f":transition{index}:response")
                steps = row.get("steps", [])
                require([entry.get("step") for entry in steps] == list(range(20)), item + ":20_steps")
                for entry in steps:
                    step = entry.get("step")
                    step_label = item + f":step{step}"
                    group = ([rank["rank"]] if gate == "sp1_resume" else [0, 1]
                             if gate in ("fixed_sp2", "fixed_sp2_pause", "sp2_sana_interleave") else
                             ([0, 1] if gate == "elastic_1_2_1" else [1]) if 5 <= step < 10 else [0])
                    epoch = (0 if gate == "sp1_resume" else 1 if gate in (
                        "fixed_sp2", "fixed_sp2_pause", "sp2_sana_interleave") else int(step >= 5) + int(step >= 10))
                    owner = rank["rank"] in group
                    require(entry.get("group") == group and entry.get("ownership_epoch") == epoch
                            and entry.get("owner") is owner, step_label + ":ownership")
                    require(entry.get("passed") is True and entry.get("global_passed") is True,
                            step_label + ":passed")
                    replica = entry.get("replica_audit", {})
                    peers = replica.get("replicas", [])
                    owners = [peer for peer in peers if peer.get("owner")]
                    replica_group = [0, 1] if gate == "sp1_resume" else group
                    require(replica.get("passed") is True and len(peers) == 2
                            and {peer.get("rank") for peer in peers} == {0, 1}
                            and {peer.get("rank") for peer in owners} == set(replica_group)
                            and all(peer.get("latent") == owners[0].get("latent") for peer in owners)
                            and bool(owners) and owners[0].get("latent") is not None,
                            step_label + ":replicas")
                    if not owner:
                        require(not entry.get("execution") and not entry.get("quanta"), step_label + ":nonowner_idle")
                        continue
                    exact = gate not in ("fixed_sp2", "elastic_1_2_1")
                    comparison = entry.get("comparison", {})
                    require(bool(comparison) and comparison_passes(comparison, exact=exact), step_label + ":numerical")
                    chunks = config["pause_blocks"] if gate in (
                        "sp1_resume", "fixed_sp2_pause", "sp2_sana_interleave") else [60]
                    execution, quanta = entry.get("execution", []), entry.get("quanta", [])
                    require(len(execution) == len(chunks) == len(quanta), step_label + ":quanta")
                    for index, (event, progress, count) in enumerate(zip(execution, quanta, chunks)):
                        audit = _quantum_audit(dict(events=event.get("events", []), progress=progress), config,
                            step=step, group=group, epoch=epoch, sequence=step*len(chunks)+index,
                            blocks=count, completed=index == len(chunks)-1, request_id=row.get("state_id"),
                            frames=case.get("frames"))
                        require(audit["passed"] and event.get("audit") == audit, step_label + f":quantum{index}")
                if rank["rank"] == 0 or gate == "sp1_resume":
                    comparison = row.get("video_comparison", {})
                    require(bool(comparison) and comparison_passes(comparison, exact=gate not in (
                        "fixed_sp2", "elastic_1_2_1")), item + ":video")
    return dict(passed=not errors, errors=errors, scope="one_stage_full_matrix_not_performance",
                stage=stage, policy=policy)


def validate_repaired_result(result):
    """Fail closed for incomplete or malformed durable evidence, without CUDA."""
    try:
        return _validate_repaired_result(result)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as error:
        return dict(passed=False, errors=["malformed_evidence: " + str(error)],
                    scope="one_stage_full_matrix_not_performance")


def _video_reference_path(reference, video_root):
    """Keep shared cross-rank comparison arrays outside permanent run artifacts."""
    reference = Path(reference)
    return (Path(video_root) / reference.parent.name / reference.name / "video.npy"
            if video_root is not None else reference / "video.npy")


def _candidate(backend, config, rank, spec, gate, reference, out, *, sana=None, sana_reference=None,
               video_root=None):
    """All ranks execute control flow; only current request owners compute."""
    import numpy as np
    from .workflow_encoder import compare_arrays
    torch, device = backend.runtime.torch, backend.runtime.device
    out.mkdir(parents=True, exist_ok=True)
    state_id = spec["request_id"] + "-" + gate
    group, epoch = [0], 0
    audit = config["sp_linear_policy"] != "native"
    transfer_root = out.parents[2] / "transfers"
    row = dict(gate=gate, steps=[], transitions=[], passed=False,
               performance_evidence=False, reference=str(reference),
               numerical_threshold=0.05, pause_blocks=config["pause_blocks"],
               first_failure=None, semantic="unverified", sana_interruptions=[],
               sp_linear_policy=config["sp_linear_policy"], backend_evidence_required=audit, state_id=state_id)
    if gate == "sp1_resume":
        group = [rank]
    try:
        if rank in group:
            backend.prepare(dict(state_id=state_id, spec=dict(spec, request_id=state_id)))
        torch.distributed.barrier()
        if gate in ("fixed_sp2", "fixed_sp2_pause", "sp2_sana_interleave"):
            epoch = _transition(backend, state_id, [0], [0, 1], epoch,
                                transfer_root / (state_id + "-initial.pt"), row["transitions"],
                                rank=rank, completed_steps=0, audit=audit)
            group = [0, 1]
        reference_metadata = json.loads((reference / "reference.json").read_text())
        initial_ok = True
        row["initial_identity"] = dict(checked=rank in group,
                                       reference_initial=reference_metadata["initial"],
                                       reference_timesteps=reference_metadata["timesteps"])
        if rank in group:
            state = backend.states[state_id]
            row["initial_identity"].update(actual_initial=_hash_array(_cpu_array(state.latents)),
                actual_timesteps=[float(v) for v in state.scheduler.timesteps.detach().cpu().tolist()])
            initial_ok = (row["initial_identity"]["actual_initial"] == reference_metadata["initial"] and
                          row["initial_identity"]["actual_timesteps"] == reference_metadata["timesteps"])
        row["initial_and_schedule_exact"] = _world_pass(torch, device, initial_ok)
        if not row["initial_and_schedule_exact"]:
            row["first_failure"] = dict(stage="initial_or_scheduler_identity")
            return row
        for step in range(20):
            if gate in ("elastic_1_2_1", "sp1_transfer") and step in (5, 10):
                target = ([0, 1] if gate == "elastic_1_2_1" else [1]) if step == 5 else [0]
                epoch = _transition(backend, state_id, group, target, epoch,
                                    transfer_root / (state_id + f"-step{step:02d}.pt"), row["transitions"],
                                    rank=rank, completed_steps=step, audit=audit)
                group = target
            local = dict(step=step, group=list(group), ownership_epoch=epoch,
                         owner=rank in group, passed=True)
            if rank in group:
                chunks = config["pause_blocks"] if gate in (
                    "sp1_resume", "fixed_sp2_pause", "sp2_sana_interleave") else [60]
                before_step = backend.states[state_id].step_index
                local["quanta"] = []
                local["execution"] = []
                for chunk_index, count in enumerate(chunks):
                    sequence = step*len(chunks)+chunk_index
                    response = backend.advance(dict(state_id=state_id, blocks=count,
                                                    expected_epoch=epoch,
                                                    quantum_seq=sequence))
                    local["quanta"].append(response.get("progress"))
                    execution = dict(events=response.get("events", []))
                    if audit:
                        execution["audit"] = _quantum_audit(
                            response, config, step=step, group=group, epoch=epoch,
                            sequence=sequence, blocks=count, completed=chunk_index == len(chunks)-1,
                            request_id=state_id, frames=spec.get("frames"))
                        if not execution["audit"]["passed"]:
                            local["passed"] = False
                    local["execution"].append(execution)
                    if (gate == "sp2_sana_interleave" and chunk_index == 0
                            and step in config["interleave_steps"]):
                        image_passed = True
                        if rank == 0:
                            if sana is None or sana_reference is None:
                                raise RuntimeError("real Sana backend/reference missing for interleave gate")
                            before = _hash_array(_cpu_array(backend.states[state_id].latents))
                            image, receipt = _sana_image(
                                sana, config["sana_prompt"], spec["seed"] + 17,
                                state_id + f"-sana-step{step:02d}")
                            actual = _hash_array(image)
                            unchanged = before == _hash_array(_cpu_array(backend.states[state_id].latents))
                            image_passed = actual == sana_reference and unchanged
                            row["sana_interruptions"].append(dict(
                                step=step, after_blocks=count, output=actual, reference=sana_reference,
                                passed=image_passed, wan_latent_unchanged=unchanged, receipt=receipt))
                        if not _world_pass(torch, device, image_passed):
                            row["first_failure"] = dict(stage="sana_interruption", step=step)
                            return row
                state = backend.states[state_id]
                if state.step_index != before_step + 1:
                    raise RuntimeError("block pause did not commit exactly one solver step")
                latent = _cpu_array(state.latents)
                expected = np.load(reference / f"step-{step:02d}.npy", allow_pickle=False, mmap_mode="r")
                comparison = compare_arrays(latent, expected)
                exact = gate in ("sp1_resume", "sp1_transfer", "fixed_sp2_pause", "sp2_sana_interleave")
                local.update(comparison=comparison, latent=_hash_array(latent),
                             passed=local["passed"] and comparison_passes(comparison, exact=exact))
                if rank == 0 and gate == "fixed_sp2":
                    np.save(out / f"step-{step:02d}.npy", latent, allow_pickle=False)
                if not local["passed"]:
                    np.save(out / f"failed-step-{step:02d}.npy", latent, allow_pickle=False)
            if audit:
                local["replica_audit"] = _replica_audit(
                    torch, rank, [0, 1] if gate == "sp1_resume" else group, local.get("latent"))
                local["passed"] = local["passed"] and local["replica_audit"]["passed"]
            passed = _world_pass(torch, device, local["passed"])
            local["global_passed"] = passed
            row["steps"].append(local)
            atomic_json(out / "progress.json", row)
            if not passed:
                failure = "backend_execution_or_replica" if audit and (
                    not local["replica_audit"]["passed"] or any(
                        not item["audit"]["passed"] for item in local.get("execution", []))) else "latent"
                row["first_failure"] = dict(stage=failure, step=step)
                row["not_run_steps"] = list(range(step+1, 20))
                return row
        local_passed = True
        if rank == min(group) or gate == "sp1_resume":
            video = backend.runtime.finish(backend.states[state_id])
            expected = np.load(_video_reference_path(reference, video_root), allow_pickle=False, mmap_mode="r")
            comparison = compare_arrays(video, expected)
            local_passed = comparison_passes(comparison, exact=gate in (
                "sp1_resume", "sp1_transfer", "fixed_sp2_pause", "sp2_sana_interleave"))
            row["video_comparison"] = comparison
            row["video"] = _hash_array(video)
            if rank == 0 and gate == "fixed_sp2":
                if video_root is not None or config["save_videos"]:
                    video_path = _video_reference_path(out, video_root)
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(video_path, video, allow_pickle=False)
                atomic_json(out / "reference.json", reference_metadata)
            if not local_passed and config["save_videos"]:
                np.save(out / "failed-video.npy", video, allow_pickle=False)
        row["passed"] = _world_pass(torch, device, local_passed)
        if not row["passed"]:
            row["first_failure"] = dict(stage="video")
        return row
    finally:
        if state_id in backend.states:
            backend.release(dict(state_id=state_id))
        if audit:
            row["release_audit"] = dict(
                states_cleared=state_id not in backend.states,
                continuations_cleared=state_id not in backend.continuations,
                ownership_cleared=state_id not in backend.elastic.records)
            if not all(row["release_audit"].values()):
                row["passed"] = False
                row["first_failure"] = dict(stage="release")
        atomic_json(out / "result.json", row)


def _child(config, rank, rendezvous, out, video_root=None):
    directory = Path(out) / f"rank{rank}"
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(rank=rank, status="running", cases=[], performance_evidence=False)
    atomic_json(directory / "run.json", result)
    backend = sana = None
    gates = STAGE_GATES[config["stage"]]
    try:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        if os.environ["NCCL_P2P_DISABLE"] != "1":
            raise ValueError("4090 validation retains NCCL_P2P_DISABLE=1")
        # Match other workflow runners' CPU affinity and arithmetic setup
        # before constructing the backend's one model instance.
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = str(config["cpu_threads"])
        result["cpu_affinity"] = _bind_cpu(config, config["devices"][rank])
        import torch
        torch.set_num_threads(config["cpu_threads"])
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(config["devices"][rank])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        from ditserve.models.wan.backend import WanBackend
        backend = WanBackend(dict(device=config["devices"][rank], model_path=config["model_path"],
                                  attention="flash", options=dict(execution="block",
                                      sp_linear_policy=config["sp_linear_policy"])))
        result["ordinary_reference_before_distributed"] = (
            backend.parallel is None and backend.elastic is None and not torch.distributed.is_initialized())
        if not result["ordinary_reference_before_distributed"]:
            raise RuntimeError("ordinary SP1 reference requires an unattached independent runtime")
        if "sp2_sana_interleave" in gates and rank == 0:
            from ditserve.models.sana.backend import SanaBackend
            sana = SanaBackend(dict(device=config["devices"][rank], model_path=config["sana_model_path"]))
        for seed in config["seeds"]:
            for frames in config["frames"]:
                case = f"seed{seed}-f{frames}"
                spec = _spec(config, frames, seed, case)
                reference = directory / case / "ordinary"
                ordinary = _ordinary_reference(backend.runtime, spec, reference, save=rank == 0,
                    save_video=config["save_videos"],
                    video_path=_video_reference_path(reference, video_root) if video_root is not None else None)
                sana_reference = None
                if sana is not None:
                    image, receipt = _sana_image(sana, config["sana_prompt"], seed + 17, case + "-sana-reference")
                    sana_reference = _hash_array(image)
                    atomic_json(directory / case / "sana-reference.json",
                                dict(output=sana_reference, receipt=receipt))
                result["cases"].append(dict(case=case, frames=frames, seed=seed, spec=spec,
                                            ordinary=ordinary, sana_reference=sana_reference,
                                            gates=[], passed=False))
                atomic_json(directory / "run.json", result)
        result["backend_init"] = backend.distributed(dict(
            action="init", elastic=True, rank=rank, world_size=2, ranks=[rank], init_method=rendezvous,
            timeout_seconds=config["group_timeout_seconds"]))
        if (config["sp_linear_policy"] != "native" and
                result["backend_init"].get("sp_linear_policy") != config["sp_linear_policy"]):
            raise RuntimeError("backend did not acknowledge the requested SP linear policy")
        for case in result["cases"]:
            peers = [None, None]
            torch.distributed.all_gather_object(peers, case["ordinary"])
            case["cross_card_sp1_exact"] = peers[0] == peers[1] and peers[0]["passed"]
            if not case["cross_card_sp1_exact"]:
                case["blocked_gates"] = list(gates)
                result["status"] = "qualification_failed"
                break
            while (gate := gate_order(case["gates"], gates)) is not None:
                reference_name = "fixed_sp2" if gate in ("fixed_sp2_pause", "sp2_sana_interleave") else "ordinary"
                reference = Path(out) / "rank0" / case["case"] / reference_name
                candidate = _candidate(backend, config, rank, case["spec"], gate, reference,
                                       directory / case["case"] / gate,
                                       sana=sana, sana_reference=case["sana_reference"], video_root=video_root)
                case["gates"].append(candidate)
                atomic_json(directory / "run.json", result)
                if not candidate["passed"]:
                    case["blocked_gates"] = list(gates[len(case["gates"]):])
                    break
            case["passed"] = len(case["gates"]) == len(gates) and all(gate["passed"] for gate in case["gates"])
            if not case["passed"]:
                result["status"] = "qualification_failed"
                break
        else:
            result["status"] = "completed"
        result["qualification"] = dict(passed=result["status"] == "completed")
        result["unrun_cases"] = [case["case"] for case in result["cases"]
                                 if "cross_card_sp1_exact" not in case]
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    finally:
        try:
            if backend is not None:
                backend.close()
            if sana is not None:
                sana.close()
            result["backend_closed"] = True
        except Exception:
            result.update(status="failed", cleanup_error=traceback.format_exc(),
                          qualification=dict(passed=False), backend_closed=False)
        result["finished_unix_ns"] = time.time_ns()
        atomic_json(directory / "run.json", result)


def _run_ranks(config, out, video_root=None):
    rendezvous = out / "elastic-rendezvous"
    if rendezvous.exists():
        raise FileExistsError(rendezvous)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_child, args=(config, rank, "file://"+str(rendezvous.resolve()), str(out), video_root))
                 for rank in (0, 1)]
    deadline = time.monotonic() + config["timeout_seconds"]
    try:
        for process in processes:
            process.start()
        while any(process.is_alive() for process in processes):
            if time.monotonic() >= deadline:
                raise TimeoutError("elastic validation exceeded its cumulative job allowance")
            for rank, process in enumerate(processes):
                path = out / f"rank{rank}" / "run.json"
                if path.exists() and json.loads(path.read_text()).get("status") == "failed":
                    raise RuntimeError(f"elastic rank {rank} failed; see {path}")
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"elastic rank {rank} exited {process.exitcode}")
                process.join(timeout=0.2)
        ranks = []
        for rank, process in enumerate(processes):
            process.join(timeout=0)
            if process.exitcode != 0:
                raise RuntimeError(f"elastic rank {rank} exited {process.exitcode} during finalization")
            result = json.loads((out / f"rank{rank}" / "run.json").read_text())
            result["process_lifecycle"] = dict(exitcode=process.exitcode, reaped=not process.is_alive())
            ranks.append(result)
        return ranks
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)


def run_validation(config, out):
    config, out = normalize_config(config), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(out / "run.json")
    result = dict(schema=SCHEMA, kind="elastic_validation", status="running",
                  configuration=config, started_unix_ns=time.time_ns(),
                  performance_evidence=False, semantic="unverified",
                  qualification=dict(passed=False), stage=config["stage"],
                  gates=list(STAGE_GATES[config["stage"]]))
    atomic_json(out / "run.json", result)
    try:
        # All readers finish (or are terminated and reaped by _run_ranks)
        # before temporary videos are removed, including failure/timeout paths.
        videos = (contextlib.nullcontext(None) if config["save_videos"] else
                  tempfile.TemporaryDirectory(prefix=".video-references-", dir=out))
        with videos as video_root:
            ranks = _run_ranks(config, out, video_root)
        passed = (len(ranks) == 2 and all(
            rank.get("status") == "completed" and rank.get("qualification", {}).get("passed")
            for rank in ranks))
        if config["sp_linear_policy"] != "native":
            result["ranks"] = ranks
            result["structural_qualification"] = validate_repaired_result(result)
            passed = passed and result["structural_qualification"]["passed"]
        result.update(ranks=ranks, status="completed" if passed else "qualification_failed",
                      qualification=dict(passed=passed))
    except Exception:
        result.update(status="failed", error=traceback.format_exc(),
                      rank_manifests=[str(path) for path in sorted(out.glob("rank*/run.json"))])
    finally:
        result["finished_unix_ns"] = time.time_ns()
        atomic_json(out / "run.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_validation(json.loads(args.config.read_text()), args.out)
    print(json.dumps(dict(status=result["status"], manifest=str(args.out / "run.json"))))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
