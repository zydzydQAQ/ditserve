"""Opt-in paired SP2 scratch repair qualification and normal completion timing.

The old repaired SP2 is the baseline. This does not claim speedup over SP1,
validate an online scheduler, or relax the earlier conservative SP2 gates.
"""

import argparse
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import time
import traceback

from .workflow_overlap import _decode_gpu, _hash_array, atomic_json
from .workflow_elastic_validation import (_bind_cpu, _ordinary_reference, _quantum_audit,
                                         _replica_audit, _spec, _world_pass)
from .workflow_serving_overlap import normalize_config as overlap_config
from .workflow_serving_sp import _cpu_array, comparison_passes
from ..models.wan.linear import SCRATCH_LIMIT_BYTES

SCHEMA = "ditserve.workflow.sp2_performance.v1"
POLICIES = ("full_rows_except_ffn1", "full_rows_except_ffn1_scratch")
TIMING_SCOPE = ("Prepared fixed-SP2 request, both ranks synchronized: 20 actual block-backend "
                "solver steps plus owner GPU VAE/postprocessing and a final world barrier. "
                "Excludes model/CPU encoding/input preparation/initial transfer, reference reads, "
                "per-step D2H, numerical/profile checks, final D2H/hashing, release and JSON. "
                "Includes production consensus and block synchronization. Not online latency.")


def _ordered_sum(values):
    """Preserve pinned Python 3.10 profile arithmetic on newer report hosts.

    Python 3.12 changed float ``sum`` accumulation. Explicit left-to-right
    addition reproduces existing raw totals exactly without numeric tolerance.
    """
    result = 0
    for value in values:
        result += value
    return result


def normalize_config(config):
    value = overlap_config(config)
    defaults = dict(devices=["cuda:0", "cuda:1"], frames=[17, 33], seeds=[3501, 3502],
                    policies=list(POLICIES), warmup=2, repeats=3, pause_blocks=[23, 14, 23],
                    timeout_seconds=1400, group_timeout_seconds=60, threshold=0.0,
                    schedule=[dict(step=5, degree=2), dict(step=10, degree=1)])
    for key, default in defaults.items():
        value.setdefault(key, default)
    for key in ("frames", "seeds", "policies", "warmup", "repeats", "pause_blocks", "threshold", "schedule"):
        if value[key] != defaults[key]:
            raise ValueError("SP2 paired study fixes " + key)
    if value["height"] != 480 or value["width"] != 832 or value["guidance_scale"] != 5.0:
        raise ValueError("SP2 paired study fixes the 480x832 CFG5 recipe")
    if (len(value["devices"]) != 2 or len(set(value["devices"])) != 2 or any(
            not str(device).startswith("cuda:") or not str(device)[5:].isdigit() for device in value["devices"])):
        raise ValueError("SP2 paired study requires two distinct CUDA devices")
    if type(value["timeout_seconds"]) is not int or value["timeout_seconds"] <= 0:
        raise ValueError("positive timeout required")
    if type(value["group_timeout_seconds"]) not in (int, float) or not 0 < value["group_timeout_seconds"] <= 60:
        raise ValueError("group timeout must be in (0,60]")
    return value


class _RepairProfile:
    """CUDA-event envelopes in one diagnostic warmup step, never timed trials."""
    def __init__(self, torch):
        self.torch, self.records = torch, []

    def begin(self, name):
        row = dict(module=name, events=[self.torch.cuda.Event(enable_timing=True) for _ in range(4)],
                   started_ns=time.perf_counter_ns())
        row["events"][0].record()
        self.records.append(row)
        return row

    def prepared(self, row):
        row["events"][1].record()
        row["padding_host_submit_ms"] = (time.perf_counter_ns() - row["started_ns"]) / 1e6

    def computed(self, row):
        row["events"][2].record()

    def cropped(self, row):
        row["events"][3].record()

    def finish(self):
        rows = [dict(module=row["module"], padding_host_submit_ms=row["padding_host_submit_ms"],
                     padding_cuda_ms=row["events"][0].elapsed_time(row["events"][1]),
                     linear_cuda_ms=row["events"][1].elapsed_time(row["events"][2]),
                     crop_cuda_ms=row["events"][2].elapsed_time(row["events"][3])) for row in self.records]
        return dict(scope="instrumented warmup step 0; CUDA event envelopes, not normal request timing",
                    calls=len(rows), totals={key: _ordered_sum(row[key] for row in rows) for key in
                    ("padding_host_submit_ms", "padding_cuda_ms", "linear_cuda_ms", "crop_cuda_ms")}, rows=rows)


def _select_policy(backend, policy):
    repair = backend.parallel._linear_repair
    if (backend.states or backend.continuations or backend.elastic.records or repair.geometry is not None
            or repair.scratch is not None or any(repair.pending.values())):
        raise RuntimeError("policy switch requires all prior requests and scratch released")
    peers = [None, None]
    backend.runtime.torch.distributed.all_gather_object(peers, policy)
    if peers != [policy, policy] or policy not in POLICIES:
        raise RuntimeError("policy switch world consensus failed")
    backend.sp_linear_policy = backend.parallel.linear_policy = policy


def _transition(backend, key, source, target, epoch, path, step):
    request = dict(action="transition", state_id=key, transfer_id=path.name,
                   source_group=source, target_group=target, expected_epoch=epoch, path=str(path))
    response = backend.distributed(request)
    owner = backend.parallel.handle.rank in target
    progress = response.get("progress")
    digest = response.get("state_sha256", "")
    checks = dict(committed=response.get("committed") is True and response.get("phase") == "finalized",
                  identity=response.get("state_id") == key and response.get("transfer_id") == path.name,
                  ownership=response.get("group") == target and response.get("ownership_epoch") == epoch + 1,
                  ready=response.get("transfer_ready") is True,
                  digest=isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                  contract=response.get("execution_contract") == dict(schema="ditserve.wan-sp-execution.v1",
                                                        sp_linear_policy=backend.parallel.linear_policy),
                  state=(key in backend.states) == owner,
                  continuation=(key in backend.continuations) == owner,
                  record=(key in backend.elastic.records) == owner,
                  progress=(isinstance(progress, dict) and progress.get("step") == step and
                            progress.get("group") == target and progress.get("ownership_epoch") == epoch+1
                            and progress.get("transfer_ready") is True) if owner else progress is None)
    if not all(checks.values()):
        raise RuntimeError("scratch transition contract failed: " + repr(checks))
    return dict(request=request, response=response, checks=checks, passed=True, completed_steps=step)


def _execution_check(response, config, policy, step, group, epoch, seq, blocks, completed, key, frames):
    check = _quantum_audit(response, dict(config, sp_linear_policy=policy), step=step, group=group,
                           epoch=epoch, sequence=seq, blocks=blocks, completed=completed,
                           request_id=key, frames=frames)
    repair = response.get("events", [{}])[0].get("linear_repair", {})
    expected = {60: 422, 23: 161 if seq % 3 == 0 else 162, 14: 99}[blocks] if len(group) == 2 else 0
    check["checks"]["exact_call_count"] = repair.get("applied_calls") == expected
    check["checks"]["bounded_scratch"] = 0 <= repair.get("scratch_peak_bytes", -1) <= SCRATCH_LIMIT_BYTES
    if policy == POLICIES[1] and len(group) == 2:
        check["checks"]["scratch_accounting"] = (repair.get("scratch_hits", 0) > 0 and
            repair.get("scratch_hits", 0) + repair.get("scratch_allocations", 0) + repair.get("scratch_fallbacks", 0) == expected)
    else:
        check["checks"]["no_scratch"] = all(repair.get(name) == 0 for name in
                                                  ("scratch_hits", "scratch_allocations", "scratch_fallbacks", "scratch_peak_bytes"))
    check["passed"] = all(check["checks"].values())
    return check


def _trial(backend, config, rank, spec, policy, mode, index, reference, out):
    """Qualification carries per-step arrays; normal timing never does."""
    import numpy as np
    from .workflow_encoder import compare_arrays
    torch, device = backend.runtime.torch, backend.runtime.device
    key = f"{spec['request_id']}-{policy}-{mode}-{index}"
    row = dict(state_id=key, policy=policy, mode=mode, index=index, frames=spec["frames"], seed=spec["seed"],
               passed=False, transitions=[], steps=[], performance_evidence=mode == "measured")
    group, epoch, seq = [0], 0, 0
    ordinary = json.loads((reference / "reference.json").read_text())
    is_diagnostic = mode != "measured"
    response_rows, profile, video_gpu = [], None, None
    try:
        if rank == 0:
            backend.prepare(dict(state_id=key, spec=dict(spec, request_id=key)))
        torch.distributed.barrier()
        if mode != "resize_pause":
            row["transitions"].append(_transition(backend, key, group, [0, 1], epoch,
                                                 out.parent / "transfers" / (key + "-initial.pt"), 0))
            group, epoch = [0, 1], 1
        initial_ok = True
        if rank in group:
            state = backend.states[key]
            row["initial"] = _hash_array(_cpu_array(state.latents))
            row["timesteps"] = [float(v) for v in state.scheduler.timesteps.detach().cpu().tolist()]
            initial_ok = row["initial"] == ordinary["initial"] and row["timesteps"] == ordinary["timesteps"]
        row["initial_and_schedule_exact"] = _world_pass(torch, device, initial_ok)
        if not row["initial_and_schedule_exact"]:
            raise RuntimeError("initial latent or schedule differs from ordinary SP1")
        backend.runtime._sync()
        torch.distributed.barrier()
        started = time.perf_counter_ns()
        for step in range(20):
            if mode == "resize_pause" and step in (5, 10):
                target = [0, 1] if step == 5 else [0]
                row["transitions"].append(_transition(backend, key, group, target, epoch,
                                                     out.parent / "transfers" / (key + f"-step{step}.pt"), step))
                group, epoch = target, epoch + 1
            step_row = dict(step=step, group=list(group), epoch=epoch, owner=rank in group, execution=[], passed=True)
            if rank in group:
                if mode == "warmup" and index == 0 and step == 0:
                    profile = _RepairProfile(torch)
                    backend.parallel._linear_repair.profiler = profile
                chunks = config["pause_blocks"] if mode == "resize_pause" else [60]
                for chunk_index, count in enumerate(chunks):
                    seq = step * len(chunks) + chunk_index
                    response = backend.advance(dict(state_id=key, blocks=count, expected_epoch=epoch, quantum_seq=seq))
                    # Retain production responses in RAM; validate outside the
                    # measured envelope to avoid diagnostic work in timing.
                    response_rows.append((response, step_row, step, list(group), epoch, seq, count,
                                          chunk_index == len(chunks)-1))
                if profile is not None and step == 0:
                    backend.parallel._linear_repair.profiler = None
                    row["repair_profile"] = profile.finish()
                if is_diagnostic:
                    actual = _cpu_array(backend.states[key].latents)
                    expected = np.load(reference / f"step-{step:02d}.npy", allow_pickle=False, mmap_mode="r")
                    step_row["comparison"] = compare_arrays(actual, expected)
                    step_row["latent"] = _hash_array(actual)
                    step_row["passed"] = comparison_passes(step_row["comparison"], exact=True)
            if is_diagnostic:
                step_row["replica_audit"] = _replica_audit(torch, rank, group, step_row.get("latent"))
                step_row["passed"] &= step_row["replica_audit"]["passed"]
                if not _world_pass(torch, device, step_row["passed"]):
                    raise RuntimeError("per-step exact SP1 comparison failed")
            row["steps"].append(step_row)
        backend.runtime._sync()
        dit_end = time.perf_counter_ns()
        if rank == 0:
            video_gpu = _decode_gpu(backend.runtime, backend.states[key])
        backend.runtime._sync()
        torch.distributed.barrier()
        finished = time.perf_counter_ns()
        if mode == "measured":
            row.update(wall_ms=(finished-started)/1e6, dit_ms=(dit_end-started)/1e6,
                       started_monotonic_ns=started, dit_end_monotonic_ns=dit_end, finished_monotonic_ns=finished,
                       timing_scope=TIMING_SCOPE, diagnostic_in_timing=False)
        for response, step_row, step, owners, step_epoch, sequence, count, completed in response_rows:
            audit = _execution_check(response, config, policy, step, owners, step_epoch, sequence,
                                     count, completed, key, spec["frames"])
            step_row["execution"].append(dict(response=response, audit=audit))
            step_row["passed"] &= audit["passed"]
        local_passed = all(step_row["passed"] for step_row in row["steps"])
        if rank in group:
            row["final_latent"] = _hash_array(_cpu_array(backend.states[key].latents))
            local_passed &= row["final_latent"] == ordinary["steps"][-1]
        if rank == 0:
            video = video_gpu.float().cpu().numpy()
            row["video"] = _hash_array(video)
            local_passed &= row["video"] == ordinary["video"]
        row["final_replica_audit"] = _replica_audit(torch, rank, group, row.get("final_latent"))
        local_passed &= row["final_replica_audit"]["passed"]
        row["passed"] = _world_pass(torch, device, local_passed)
        if not row["passed"]:
            raise RuntimeError("final output or real backend execution audit failed")
    finally:
        backend.parallel._linear_repair.profiler = None
        if key in backend.states:
            backend.release(dict(state_id=key))
        repair = backend.parallel._linear_repair
        row["release"] = dict(states=key not in backend.states, continuations=key not in backend.continuations,
                              ownership=key not in backend.elastic.records, scratch=repair.scratch is None,
                              geometry=repair.geometry is None, pending=not any(repair.pending.values()))
        row["passed"] &= all(row["release"].values())
        video_gpu = None
        atomic_json(out / "trials" / (key + ".json"), row)
    return row


def _child(config, rank, rendezvous, out):
    out = Path(out)
    directory = out / f"rank{rank}"
    directory.mkdir(parents=True, exist_ok=True)
    (out / "transfers").mkdir(exist_ok=True)
    result = dict(rank=rank, status="running", references=[], trials=[], qualification=dict(passed=False))
    backend = None
    try:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        if os.environ["NCCL_P2P_DISABLE"] != "1":
            raise ValueError("4090 SP2 keeps NCCL_P2P_DISABLE=1")
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
        backend = WanBackend(dict(device=config["devices"][rank], model_path=config["model_path"], attention="flash",
                                  options=dict(execution="block", sp_linear_policy=POLICIES[1])))
        result["ordinary_reference_before_distributed"] = (backend.parallel is None and backend.elastic is None
                                                           and not torch.distributed.is_initialized())
        if not result["ordinary_reference_before_distributed"]:
            raise RuntimeError("ordinary SP1 reference must precede all distributed attachment")
        for seed in config["seeds"]:
            for frames in config["frames"]:
                case = f"seed{seed}-f{frames}"
                spec = _spec(config, frames, seed, case)
                ordinary = _ordinary_reference(backend.runtime, spec, directory / case / "ordinary", save=rank == 0,
                                               save_video=config["save_videos"])
                result["references"].append(dict(case=case, frames=frames, seed=seed, spec=spec, ordinary=ordinary))
                atomic_json(directory / "run.json", result)
        result["backend_init"] = backend.distributed(dict(action="init", elastic=True, rank=rank, world_size=2,
                            ranks=[rank], init_method=rendezvous, timeout_seconds=config["group_timeout_seconds"]))
        for case in result["references"]:
            peers = [None, None]
            torch.distributed.all_gather_object(peers, case["ordinary"])
            case["cross_card_sp1_exact"] = peers[0] == peers[1] and peers[0]["passed"]
            if not case["cross_card_sp1_exact"]:
                raise RuntimeError("ordinary SP1 differs across GPUs")
        for case in result["references"]:
            reference = out / "rank0" / case["case"] / "ordinary"
            row = _trial(backend, config, rank, case["spec"], POLICIES[1], "resize_pause", 0, reference, directory)
            result["trials"].append(row)
            atomic_json(directory / "run.json", result)
        # Each arm/frame receives two genuine warmups. The two qualification
        # seeds occupy these warmups; the first also profiles repair envelopes.
        for warmup, seed in enumerate(config["seeds"]):
            for frames in config["frames"]:
                case = next(c for c in result["references"] if c["seed"] == seed and c["frames"] == frames)
                for policy in (POLICIES if warmup % 2 == 0 else POLICIES[::-1]):
                    _select_policy(backend, policy)
                    row = _trial(backend, config, rank, case["spec"], policy, "warmup", warmup,
                                 out / "rank0" / case["case"] / "ordinary", directory)
                    result["trials"].append(row)
                    atomic_json(directory / "run.json", result)
        result["qualification"]["passed"] = True
        for frames in config["frames"]:
            case = next(c for c in result["references"] if c["seed"] == config["seeds"][0] and c["frames"] == frames)
            for repeat in range(config["repeats"]):
                for policy in (POLICIES if repeat % 2 == 0 else POLICIES[::-1]):
                    _select_policy(backend, policy)
                    row = _trial(backend, config, rank, case["spec"], policy, "measured", repeat,
                                 out / "rank0" / case["case"] / "ordinary", directory)
                    result["trials"].append(row)
                    atomic_json(directory / "run.json", result)
        result["status"] = "completed"
    except Exception:
        result.update(status="failed", error=traceback.format_exc(), qualification=dict(passed=False))
    finally:
        try:
            if backend is not None:
                backend.close()
            result["backend_closed"] = True
        except Exception:
            result.update(status="failed", cleanup_error=traceback.format_exc(), backend_closed=False)
        atomic_json(directory / "run.json", result)


def _run_ranks(config, out):
    context = multiprocessing.get_context("spawn")
    rendezvous = out / "sp2-performance-rendezvous"
    if rendezvous.exists():
        raise FileExistsError(rendezvous)
    children = [context.Process(target=_child, args=(config, rank, "file://"+str(rendezvous.resolve()), str(out)))
                for rank in (0, 1)]
    deadline = time.monotonic() + config["timeout_seconds"]
    try:
        for process in children:
            process.start()
        while any(process.is_alive() for process in children):
            if time.monotonic() >= deadline:
                raise TimeoutError("SP2 paired repair exceeded cumulative allowance")
            for rank, process in enumerate(children):
                path = out / f"rank{rank}" / "run.json"
                if path.exists() and json.loads(path.read_text()).get("status") == "failed":
                    raise RuntimeError(f"SP2 rank {rank} failed; see {path}")
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"SP2 rank {rank} exited {process.exitcode}")
                process.join(timeout=.2)
        ranks = []
        for rank, process in enumerate(children):
            process.join(timeout=0)
            if process.exitcode != 0:
                raise RuntimeError(f"SP2 rank {rank} final exit {process.exitcode}")
            result = json.loads((out / f"rank{rank}" / "run.json").read_text())
            result["process_lifecycle"] = dict(exitcode=0, reaped=not process.is_alive())
            ranks.append(result)
        return ranks
    finally:
        for process in children:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)


def validate_result(result):
    """Fail closed from durable evidence without importing CUDA or trusting summaries."""
    errors = []
    def require(condition, label):
        if not condition:
            errors.append(label)
    try:
        config = normalize_config(result.get("configuration", {}))
        require(result.get("schema") == SCHEMA and result.get("status") == "completed", "completed_schema")
        require(result.get("performance_evidence") is not True or result.get("qualification", {}).get("passed") is True,
                "performance_requires_qualification")
        ranks = result.get("ranks", [])
        require(len(ranks) == 2 and {r.get("rank") for r in ranks} == {0, 1}, "two_ranks")
        expected_cases = {(seed, frames) for seed in config["seeds"] for frames in config["frames"]}
        expected_trials = {(POLICIES[1], "resize_pause", 0, seed, frames) for seed, frames in expected_cases}
        expected_trials |= {(policy, "warmup", i, seed, frames) for i, seed in enumerate(config["seeds"])
                            for frames in config["frames"] for policy in POLICIES}
        expected_trials |= {(policy, "measured", i, config["seeds"][0], frames)
                            for i in range(3) for frames in config["frames"] for policy in POLICIES}
        for rank in ranks:
            prefix = f"rank{rank.get('rank')}"
            require(rank.get("status") == "completed" and rank.get("backend_closed") is True
                    and rank.get("ordinary_reference_before_distributed") is True, prefix + ":lifecycle")
            require(rank.get("process_lifecycle") == dict(exitcode=0, reaped=True), prefix + ":reaped")
            references = rank.get("references", [])
            require(len(references) == 4 and {(r.get("seed"), r.get("frames")) for r in references} == expected_cases,
                    prefix + ":reference_matrix")
            lookup = {(r["seed"], r["frames"]): r["ordinary"] for r in references}
            for reference in references:
                ordinary = reference["ordinary"]
                require(reference.get("cross_card_sp1_exact") is True and ordinary.get("passed") is True
                        and len(ordinary.get("steps", [])) == 20 and len(ordinary.get("timesteps", [])) == 20,
                        prefix + ":ordinary")
            trials = rank.get("trials", [])
            require(len(trials) == 24 and {(r.get("policy"), r.get("mode"), r.get("index"), r.get("seed"),
                                          r.get("frames")) for r in trials} == expected_trials, prefix + ":trial_matrix")
            require([r.get("mode") for r in trials] == ["resize_pause"]*4 + ["warmup"]*8 + ["measured"]*12,
                    prefix + ":qualification_precedes_measurement")
            for trial in trials:
                label = prefix + ":" + trial.get("state_id", "missing")
                reference = lookup[(trial["seed"], trial["frames"])]
                policy, mode = trial["policy"], trial["mode"]
                require(trial.get("passed") is True and trial.get("initial_and_schedule_exact") is True
                        and all(trial.get("release", {}).get(k) is True for k in
                                ("states", "continuations", "ownership", "scratch", "geometry", "pending")), label+":passed_release")
                if rank["rank"] == 0 or mode != "resize_pause":
                    require(trial.get("initial") == reference["initial"] and trial.get("timesteps") == reference["timesteps"],
                            label+":initial_schedule_recomputed")
                steps = trial.get("steps", [])
                require(len(steps) == 20, label+":20steps")
                for step, row in enumerate(steps):
                    owners = [0, 1] if mode != "resize_pause" or 5 <= step < 10 else [0]
                    epoch = 1 if mode != "resize_pause" else 0 if step < 5 else 1 if step < 10 else 2
                    owner = rank["rank"] in owners
                    require(row.get("step") == step and row.get("group") == owners and row.get("epoch") == epoch
                            and row.get("owner") is owner and row.get("passed") is True, label+f":step{step}")
                    chunks = [23, 14, 23] if mode == "resize_pause" else [60]
                    execution = row.get("execution", [])
                    require(len(execution) == (len(chunks) if owner else 0), label+":quantum_count")
                    for j, event in enumerate(execution):
                        # Replica rank 1 joins at step 5: transferred quantum_seq
                        # continues the source sequence rather than restarting.
                        seq = step*len(chunks)+j
                        audit = _execution_check(event["response"], config, policy, step, owners, epoch, seq,
                                                 chunks[j], j == len(chunks)-1, trial["state_id"], trial["frames"])
                        require(audit["passed"], label+f":quantum{seq}")
                    if mode != "measured":
                        require(row.get("replica_audit", {}).get("passed") is True, label+":replicas")
                        if owner:
                            require(comparison_passes(row.get("comparison", {}), exact=True)
                                    and row.get("latent") == reference["steps"][step], label+":latent_exact")
                if rank["rank"] == 0 or mode != "resize_pause":
                    require(trial.get("final_latent") == reference["steps"][-1], label+":final_latent")
                if rank["rank"] == 0:
                    require(trial.get("video") == reference["video"], label+":video")
                require(trial.get("final_replica_audit", {}).get("passed") is True, label+":final_replicas")
                transitions = trial.get("transitions", [])
                require(len(transitions) == (2 if mode == "resize_pause" else 1), label+":transitions")
                for j, tx in enumerate(transitions):
                    require(tx.get("passed") is True and set(tx.get("checks", {})) == {
                                "committed", "identity", "ownership", "ready", "digest", "contract", "state",
                                "continuation", "record", "progress"} and all(tx["checks"].values()),
                            label+":transition_checks")
                    response, request = tx["response"], tx["request"]
                    source, target, epoch = ([0], [0, 1], 0) if j == 0 else ([0, 1], [0], 1)
                    step = (5 if j == 0 else 10) if mode == "resize_pause" else 0
                    require(request.get("state_id") == trial["state_id"] and request.get("source_group") == source
                            and request.get("target_group") == target and request.get("expected_epoch") == epoch
                            and tx.get("completed_steps") == step and response.get("phase") == "finalized"
                            and response.get("committed") is True and response.get("group") == target
                            and response.get("ownership_epoch") == epoch+1 and response.get("transfer_ready") is True
                            and response.get("state_id") == trial["state_id"]
                            and response.get("transfer_id") == request.get("transfer_id"), label+":transition_recomputed")
                    require(response.get("execution_contract") == dict(schema="ditserve.wan-sp-execution.v1",
                                            sp_linear_policy=policy), label+":transition_policy")
                if mode == "measured":
                    require(trial.get("diagnostic_in_timing") is False and trial.get("timing_scope") == TIMING_SCOPE
                            and all(type(trial.get(k)) in (int, float) and math.isfinite(trial[k]) and trial[k] > 0
                                    for k in ("wall_ms", "dit_ms")) and trial["wall_ms"] >= trial["dit_ms"], label+":normal_timing")
                    start, dit_end, end = (trial.get(k) for k in
                        ("started_monotonic_ns", "dit_end_monotonic_ns", "finished_monotonic_ns"))
                    require(all(type(v) is int for v in (start, dit_end, end)) and 0 <= start < dit_end <= end
                            and trial["wall_ms"] == (end-start)/1e6 and trial["dit_ms"] == (dit_end-start)/1e6,
                            label+":normal_timing_recomputed")
                if mode == "warmup" and trial["index"] == 0:
                    profile = trial.get("repair_profile", {})
                    require(profile.get("calls") == 422 and len(profile.get("rows", [])) == 422, label+":profile")
                    keys = ("padding_host_submit_ms", "padding_cuda_ms", "linear_cuda_ms", "crop_cuda_ms")
                    for key in keys:
                        values = [row.get(key) for row in profile.get("rows", [])]
                        require(len(values) == 422 and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                                                          for v in values)
                                and profile.get("totals", {}).get(key) == _ordered_sum(values), label+":profile_totals:"+key)
        if len(ranks) == 2:
            require([r["ordinary"] for r in ranks[0]["references"]] == [r["ordinary"] for r in ranks[1]["references"]],
                    "cross_rank_reference_recomputed")
        if "summary" in result or result.get("performance_evidence") is True:
            require(result.get("summary") == summarize(ranks), "summary_recomputed")
    except (TypeError, ValueError, KeyError, IndexError, AttributeError) as error:
        errors.append("malformed:" + repr(error))
    return dict(passed=not errors, errors=errors, scope="SP2 old-vs-scratch exactness and normal timing; capacity is external guard")


def summarize(ranks):
    pairs = []
    for frames in (17, 33):
        for repeat in range(3):
            values = {}
            for policy in POLICIES:
                rows = [trial for rank in ranks for trial in rank["trials"] if trial["mode"] == "measured"
                        and trial["frames"] == frames and trial["index"] == repeat and trial["policy"] == policy]
                values[policy] = dict(wall_ms=max(row["wall_ms"] for row in rows),
                                      dit_ms=max(row["dit_ms"] for row in rows))
            old, scratch = (values[p] for p in POLICIES)
            pairs.append(dict(frames=frames, repeat=repeat, arms=values,
                              wall_reduction_percent=100*(1-scratch["wall_ms"]/old["wall_ms"]),
                              dit_reduction_percent=100*(1-scratch["dit_ms"]/old["dit_ms"])))
    return dict(pairs=pairs, by_frames=[dict(frames=frames, pairs=3,
                  median_wall_reduction_percent=statistics.median(p["wall_reduction_percent"] for p in pairs if p["frames"] == frames),
                  median_dit_reduction_percent=statistics.median(p["dit_reduction_percent"] for p in pairs if p["frames"] == frames))
                  for frames in (17, 33)], timing_scope=TIMING_SCOPE,
                caveat="Three paired repeats per frame, one timed seed; no SP1 speedup or online/A100 inference.")


def run_study(config, out):
    config, out = normalize_config(config), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(out / "run.json")
    result = dict(schema=SCHEMA, kind="sp2_performance", status="running", configuration=config,
                  started_unix_ns=time.time_ns(), qualification=dict(passed=False), performance_evidence=False)
    atomic_json(out / "run.json", result)
    try:
        result["ranks"] = _run_ranks(config, out)
        result["status"] = "completed"
        result["qualification"] = validate_result(result)
        if result["qualification"]["passed"]:
            result["summary"] = summarize(result["ranks"])
            result["performance_evidence"] = True
        else:
            result["status"] = "qualification_failed"
    except Exception:
        result.update(status="failed", error=traceback.format_exc(),
                      rank_manifests=[str(path) for path in sorted(out.glob("rank*/run.json"))])
    finally:
        result["finished_unix_ns"] = time.time_ns()
        atomic_json(out / "run.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_study(json.loads(args.config.read_text()), args.out)
    print(json.dumps(dict(status=result["status"], manifest=str(args.out / "run.json"))))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
