"""Four-arm Wan request completion comparison on one fixed two-GPU envelope.

Each seed/arm owns fresh processes. Numerical qualification is separate from
normal timing; the native arm never attaches an SP processor. No model download
or CUDA import occurs at module import time.
"""

import argparse
from datetime import timedelta
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import statistics
import time
import traceback

from .workflow_elastic_validation import (_bind_cpu, _ordinary_reference, _quantum_audit,
                                         _replica_audit, _spec, _world_pass)
from .workflow_metrics import NvmlSampler, torch_snapshot
from .workflow_overlap import _hash_array, atomic_json
from .workflow_serving_overlap import normalize_config as overlap_config
from .workflow_serving_sp import _cpu_array
from .workflow_sp2_performance import _transition


SCHEMA = "ditserve.workflow.sp_comparison.v1"
ROLE = "sp_comparison"
ARMS = ("native_sp1", "repaired_sp1", "fixed_sp2", "resize5to10")
SEEDS = (5201, 5202, 5203)
SCOPES = ("preencoded", "full_request")
POLICY = "full_rows_except_ffn1"
TIMING_SCOPES = {
    "preencoded": "Host request envelope after a fresh single-use CPU encoding ticket is ready: "
        "GPU prepare, any initial/step-5/step-10 host-file ownership transfer, 20 real block-backend "
        "steps, owner VAE, final D2H/finite/output and latent digest checks, device synchronization "
        "and final world barrier. Excludes CPU encoding, model load, per-step diagnostic copies, "
        "reference I/O, sampler queries, release and JSON.",
    "full_request": "The same host request envelope, starting before synchronous CPU prompt encoding. "
        "Includes CPU encoding, GPU prepare, any initial/step-5/step-10 host-file ownership transfer, "
        "20 real block-backend steps, owner VAE, final D2H/finite/output and latent digest checks, "
        "device synchronization and final world barrier. Excludes model load, per-step diagnostic "
        "copies, reference I/O, sampler queries, release and JSON.",
}
_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def normalize_config(config):
    value = overlap_config(config)
    defaults = dict(devices=["cuda:0", "cuda:1"], gpu_uuids=[], frames=[17, 33],
                    seeds=list(SEEDS), arms=list(ARMS), scopes=list(SCOPES), warmup=2,
                    repeats=3, sp_linear_policy=POLICY, exact_required=True,
                    timeout_seconds=21600, arm_timeout_seconds=5400,
                    group_timeout_seconds=60, sample_interval_seconds=.1)
    for key, default in defaults.items():
        value.setdefault(key, default)
    for key in ("frames", "arms", "scopes", "warmup", "repeats", "sp_linear_policy", "exact_required"):
        if type(value[key]) is not type(defaults[key]) or value[key] != defaults[key]:
            raise ValueError("SP comparison fixes " + key)
    if (not isinstance(value["seeds"], list) or not value["seeds"] or
            value["seeds"] != [seed for seed in SEEDS if seed in value["seeds"]] or
            any(type(seed) is not int for seed in value["seeds"])):
        raise ValueError("seeds must be a nonempty ordered subset of 5201/5202/5203")
    if (value["devices"] != ["cuda:0", "cuda:1"] or value["height"] != 480 or
            value["width"] != 832 or value["guidance_scale"] != 5.0 or value["cpu_threads"] != 8):
        raise ValueError("comparison fixes two visible devices, 480x832 CFG5 and eight CPU threads")
    if (not isinstance(value["gpu_uuids"], list) or len(value["gpu_uuids"]) != 2 or
            len(set(value["gpu_uuids"])) != 2 or
            any(not isinstance(uuid, str) or not _UUID.fullmatch(uuid) for uuid in value["gpu_uuids"])):
        raise ValueError("two distinct physical gpu_uuids are required")
    masks = value["cpu_masks"]
    if (set(masks) != set(value["devices"]) or any(
            not isinstance(mask, list) or len(mask) != 8 or len(set(mask)) != 8 or
            any(type(cpu) is not int or cpu < 0 for cpu in mask) for mask in masks.values()) or
            set(masks["cuda:0"]) & set(masks["cuda:1"])):
        raise ValueError("two disjoint explicit eight-core CPU masks are required")
    for key in ("timeout_seconds", "arm_timeout_seconds"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError(key + " must be a finite positive integer")
    if value["arm_timeout_seconds"] > value["timeout_seconds"]:
        raise ValueError("arm timeout exceeds job timeout")
    if value["group_timeout_seconds"] != 60 or value["sample_interval_seconds"] != .1:
        raise ValueError("collective timeout and NVML cadence are fixed")
    if value["save_videos"] is not False:
        raise ValueError("comparison retains output digests, not decoded video files")
    return value


def arm_order(seed):
    # A four-arm cyclic order across the three independent seed blocks; this is
    # balanced rotation, not a complete four-period Latin square.
    offset = SEEDS.index(seed)
    return list(ARMS[offset:] + ARMS[:offset])


def trial_matrix(config, seed):
    rows = [(frames, "diagnostic", "preencoded", 0) for frames in config["frames"]]
    for frames in config["frames"]:
        for scope in (SCOPES if SEEDS.index(seed) % 2 == 0 else SCOPES[::-1]):
            rows.extend((frames, mode, scope, index) for mode, count in
                        (("warmup", 2), ("measured", 3)) for index in range(count))
    return rows


def _layout(arm, step):
    if arm == "fixed_sp2":
        return [0, 1], 1
    if arm == "resize5to10":
        return ([0, 1] if 5 <= step < 10 else [0]), int(step >= 5) + int(step >= 10)
    return [0], 0


def _digest(artifact):
    return dict(sha256=artifact.get("array_sha256"), shape=artifact.get("shape"),
                dtype=artifact.get("dtype"), finite=artifact.get("finite"))


def _valid_digest(value):
    return (isinstance(value, dict) and value.get("finite") is True and
            isinstance(value.get("shape"), list) and bool(value["shape"]) and
            all(type(n) is int and n > 0 for n in value["shape"]) and
            isinstance(value.get("dtype"), str) and isinstance(value.get("sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None)


def _execution(response, config, arm, step, key, frames):
    group, epoch = _layout(arm, step)
    if arm != "native_sp1":
        return _quantum_audit(response, dict(config, sp_linear_policy=POLICY), step=step,
            group=group, epoch=epoch, sequence=step, blocks=60, completed=True,
            request_id=key, frames=frames)["passed"]
    events = response.get("events", [])
    event = events[0] if len(events) == 1 else {}
    return (event.get("kind") == "block_quantum" and event.get("step_index") == step and
            event.get("step_completed") is True and event.get("blocks") == 60 and
            response.get("progress", {}).get("step") == step + 1 and
            not event.get("linear_repair"))


def _memory_rows(samples):
    keys = ("timestamp_ns", "uuid", "memory_total_bytes", "memory_free_bytes", "memory_used_bytes")
    return [{key: row.get(key) for key in keys} for row in samples]


def capacity_gate(memory, config, start=None, end=None):
    errors, rows = [], memory.get("samples", [])
    if memory.get("errors") or not rows:
        errors.append("NVML samples missing or query failed")
    if {row.get("uuid") for row in rows} != set(config["gpu_uuids"]):
        errors.append("GPU UUID set mismatch")
    for uuid in config["gpu_uuids"]:
        selected = [row for row in rows if row.get("uuid") == uuid]
        times = [row.get("timestamp_ns") for row in selected]
        if (not times or any(type(t) is not int for t in times) or
                (start is not None and (min(times) > start or max(times) < end))):
            errors.append(uuid + ": incomplete request coverage")
        for row in selected:
            total, free, used = (row.get(k) for k in
                                ("memory_total_bytes", "memory_free_bytes", "memory_used_bytes"))
            if (any(type(n) is not int for n in (total, free, used)) or total <= 0 or
                    not 0 <= free <= total or used != total-free or
                    free < max(1024**3, math.ceil(total*.05))):
                errors.append(uuid + ": capacity margin or malformed sample")
                break
    return dict(passed=not errors, errors=errors, scope="sampled NVML free >= max(1 GiB,5% total)")


def _release_audit(backend, key):
    repair = getattr(getattr(backend, "parallel", None), "_linear_repair", None)
    return dict(states=key not in backend.states, continuations=key not in backend.continuations,
                ownership=backend.elastic is None or key not in backend.elastic.records,
                tickets=not backend.tickets, runtime_states=not backend.runtime._states,
                scratch=repair is None or repair.scratch is None,
                geometry=repair is None or repair.geometry is None,
                pending=repair is None or not any(repair.pending.values()))


def _trial(backend, config, rank, arm, spec, mode, scope, index, reference, out, sampler=None):
    """Only diagnostic mode copies per-step tensors; final checks are request work."""
    import numpy as np
    torch, device = backend.runtime.torch, backend.runtime.device
    key = f"{spec['request_id']}-{arm}-{mode}-{scope}-{index}"
    request_spec = dict(spec, request_id=key)
    ordinary = json.loads((reference / "reference.json").read_text())
    row = dict(state_id=key, arm=arm, mode=mode, scope=scope, index=index, seed=spec["seed"],
               frames=spec["frames"], steps=[], transitions=[], passed=False,
               diagnostic_in_timing=mode == "diagnostic", backend_native=backend.parallel is None,
               cpu_encoding_in_timing=scope == "full_request", timing_scope=TIMING_SCOPES[scope])
    ticket, group, epoch, sample_index = None, [0], 0, 0
    try:
        if scope == "preencoded" and rank == 0:
            before = time.perf_counter_ns()
            ticket = backend.encode_start(dict(state_id=key, spec=request_spec))["ticket"]
            backend.tickets[ticket][1].result()
            receipt = backend.encode_poll(dict(ticket=ticket))
            if receipt.get("ready") is not True:
                raise RuntimeError("preencoded ticket is not ready")
            row["preencoding"] = dict(start_ns=before, end_ns=time.perf_counter_ns(), receipt=receipt)
        if sampler is not None:
            sample_index = len(sampler.samples)
            sampler.sample_once()
        torch.cuda.reset_peak_memory_stats(device)
        backend.runtime._sync()
        torch.distributed.barrier()
        started = time.perf_counter_ns()
        if rank == 0:
            row["prepare"] = backend.prepare(dict(state_id=key, spec=request_spec,
                                                    **({"encode_ticket": ticket} if ticket else {})))
        prepared = time.perf_counter_ns()

        def transition(target, step):
            nonlocal group, epoch
            before = time.perf_counter_ns()
            result = _transition(backend, key, group, target, epoch,
                                 out.parent / "transfers" / (key+f"-step{step}.pt"), step)
            result.update(start_ns=before, end_ns=time.perf_counter_ns())
            row["transitions"].append(result)
            group, epoch = target, epoch+1

        if arm == "fixed_sp2":
            transition([0, 1], 0)
        if mode == "diagnostic":
            if rank in group:
                state = backend.states[key]
                row["initial"] = _hash_array(_cpu_array(state.latents))
                row["timesteps"] = [float(v) for v in state.scheduler.timesteps.detach().cpu().tolist()]
                if row["initial"] != ordinary["initial"] or row["timesteps"] != ordinary["timesteps"]:
                    raise RuntimeError("initial latent or schedule mismatch")
        for step in range(20):
            if arm == "resize5to10" and step in (5, 10):
                transition([0, 1] if step == 5 else [0], step)
            item = dict(step=step, group=list(group), epoch=epoch, owner=rank in group)
            before = time.perf_counter_ns()
            if rank in group:
                item["response"] = backend.advance(dict(state_id=key, blocks=60,
                                                        expected_epoch=epoch, quantum_seq=step))
            item.update(start_ns=before, end_ns=time.perf_counter_ns())
            if mode == "diagnostic":
                if rank in group:
                    actual = _cpu_array(backend.states[key].latents)
                    expected = np.load(reference / f"step-{step:02d}.npy", allow_pickle=False, mmap_mode="r")
                    item["latent"] = _hash_array(actual)
                    item["exact"] = (actual.dtype == expected.dtype and actual.shape == expected.shape and
                                     np.array_equal(actual, expected) and item["latent"] == ordinary["steps"][step])
                item["replicas"] = _replica_audit(torch, rank, group, item.get("latent"))
                if not _world_pass(torch, device, item.get("exact", True) and item["replicas"]["passed"]):
                    raise RuntimeError("diagnostic step or replica is not exact")
            row["steps"].append(item)
        backend.runtime._sync()
        dit_end = time.perf_counter_ns()
        if rank in group:
            row["final_latent"] = _hash_array(_cpu_array(backend.states[key].latents))
            if row["final_latent"] != ordinary["steps"][-1]:
                raise RuntimeError("final latent is not exact")
        if rank == 0:
            receipt = backend.finish(dict(state_id=key, output_mode="digest", id=0))
            row["video"] = _digest(receipt["output"])
            if row["video"] != ordinary["video"]:
                raise RuntimeError("final video is not exact")
        backend.runtime._sync()
        torch.distributed.barrier()
        finished = time.perf_counter_ns()
        row.update(start_ns=started, prepared_ns=prepared, dit_end_ns=dit_end, end_ns=finished,
                   wall_ms=(finished-started)/1e6, prepare_ms=(prepared-started)/1e6,
                   dit_and_transfer_ms=(dit_end-prepared)/1e6)
        row["allocator"] = torch_snapshot(device)
        if sampler is not None:
            sampler.sample_once()
            row["memory"] = dict(samples=_memory_rows(sampler.samples[sample_index:]), errors=list(sampler.errors))
            row["capacity"] = capacity_gate(row["memory"], config, started, finished)
        execution_ok = all(not item["owner"] or _execution(item["response"], config, arm,
            item["step"], key, spec["frames"]) for item in row["steps"])
        row["final_replicas"] = _replica_audit(torch, rank, group, row.get("final_latent"))
        row["passed"] = _world_pass(torch, device, execution_ok and row["final_replicas"]["passed"]
                                    and (rank != 0 or row["capacity"]["passed"]))
        if not row["passed"]:
            raise RuntimeError("execution, capacity or final replica audit failed")
    finally:
        try:
            if key in backend.states:
                backend.release(dict(state_id=key))
            if ticket in backend.tickets:
                backend.encode_cancel(dict(ticket=ticket))
            row["release"] = _release_audit(backend, key)
            row["passed"] &= all(row["release"].values())
            row["after_release_allocator"] = torch_snapshot(device)
            if sampler is not None:
                row["after_release_memory"] = dict(samples=_memory_rows(sampler.sample_once()), errors=list(sampler.errors))
                row["after_release_capacity"] = capacity_gate(row["after_release_memory"], config)
                row["passed"] &= row["after_release_capacity"]["passed"]
        except Exception:
            row.update(passed=False, release_error=traceback.format_exc())
            raise
        finally:
            atomic_json(out / "trials" / (key+".json"), row)
    return row


def _child(config, seed, arm, rank, rendezvous, out):
    directory = Path(out) / f"rank{rank}"
    directory.mkdir(parents=True, exist_ok=True)
    (Path(out) / "transfers").mkdir(exist_ok=True)
    result = dict(rank=rank, seed=seed, arm=arm, status="running", references=[], trials=[])
    atomic_json(directory / "status.json", dict(rank=rank, status="running"))
    backend = sampler = None
    try:
        visible = ",".join(config["gpu_uuids"])
        if os.environ.get("CUDA_VISIBLE_DEVICES", visible) != visible:
            raise ValueError("CUDA_VISIBLE_DEVICES must match the two guarded UUIDs in order")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
        result["cuda_visible_devices"] = visible
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        if os.environ["NCCL_P2P_DISABLE"] != "1":
            raise ValueError("SP comparison retains NCCL_P2P_DISABLE=1")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = str(config["cpu_threads"])
        result["cpu_affinity"] = _bind_cpu(config, config["devices"][rank])
        import torch
        if torch.cuda.device_count() != 2:
            raise RuntimeError("both and only the two bound GPUs must be visible")
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(config["devices"][rank])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        from ditserve.models.wan.backend import WanBackend
        backend = WanBackend(dict(device=config["devices"][rank], model_path=config["model_path"],
            attention="flash", options=dict(execution="block", sp_linear_policy="native" if arm == "native_sp1" else POLICY)))
        result["ordinary_reference_before_distributed"] = (
            backend.parallel is None and backend.elastic is None and not torch.distributed.is_initialized())
        if not result["ordinary_reference_before_distributed"]:
            raise RuntimeError("reference runtime already has a distributed attachment")
        result["arithmetic"] = dict(torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
            attention=backend.runtime.attention,
            negative_prompt_cache=getattr(backend.runtime, "_negative_prompt_cache", None) is not None)
        for frames in config["frames"]:
            case = f"seed{seed}-f{frames}"
            spec = _spec(config, frames, seed, case)
            ordinary = _ordinary_reference(backend.runtime, spec, directory / case / "ordinary", save=rank == 0)
            result["references"].append(dict(seed=seed, frames=frames, case=case, spec=spec, ordinary=ordinary))
        if arm == "native_sp1":
            torch.distributed.init_process_group(backend="nccl", init_method=rendezvous, rank=rank,
                world_size=2, timeout=timedelta(seconds=60))
            backend.owns_world = True
            result["backend_init"] = dict(native_unattached=True, sp_linear_policy="native", transport=None)
        else:
            result["backend_init"] = backend.distributed(dict(action="init", elastic=True, rank=rank,
                world_size=2, ranks=[rank], init_method=rendezvous, timeout_seconds=60))
        for case in result["references"]:
            peers = [None, None]
            torch.distributed.all_gather_object(peers, case["ordinary"])
            case["cross_card_sp1_exact"] = peers[0] == peers[1] and peers[0]["passed"]
            if not case["cross_card_sp1_exact"]:
                raise RuntimeError("ordinary SP1 reference differs across physical GPUs")
        if rank == 0:
            sampler = NvmlSampler(devices=config["gpu_uuids"], interval_s=.1).start()
            if sampler.errors:
                raise RuntimeError("NVML sampler could not initialize: " + repr(sampler.errors))
        for frames, mode, scope, index in trial_matrix(config, seed):
            case = next(case for case in result["references"] if case["frames"] == frames)
            trial = _trial(backend, config, rank, arm, case["spec"], mode, scope, index,
                Path(out) / "rank0" / case["case"] / "ordinary", directory, sampler)
            result["trials"].append(trial)
            atomic_json(directory / "run.json", result)
            if not trial["passed"]:
                raise RuntimeError("trial release failed; dependent timing is blocked")
        result["status"] = "completed"
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    finally:
        cleanup_errors = []
        if sampler is not None:
            try:
                sampler.stop()
                result["nvml_errors"] = list(sampler.errors)
            except Exception:
                cleanup_errors.append(traceback.format_exc())
        try:
            if backend is not None:
                backend.close()
            result["backend_closed"] = True
        except Exception:
            result["backend_closed"] = False
            cleanup_errors.append(traceback.format_exc())
        if cleanup_errors:
            result.update(status="failed", cleanup_errors=cleanup_errors)
        atomic_json(directory / "run.json", result)
        atomic_json(directory / "status.json", dict(rank=rank, status=result["status"]))


def _run_block(config, seed, arm, out, deadline):
    out.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    rendezvous = out / "rendezvous"
    if rendezvous.exists():
        raise FileExistsError(rendezvous)
    children = [context.Process(target=_child, args=(config, seed, arm, rank,
                "file://"+str(rendezvous.resolve()), str(out))) for rank in (0, 1)]
    deadline = min(deadline, time.monotonic()+config["arm_timeout_seconds"])
    started = []
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("job timeout expired before starting another arm")
        for process in children:
            process.start()
            started.append(process)
        while any(process.is_alive() for process in started):
            if time.monotonic() >= deadline:
                raise TimeoutError("SP comparison exceeded its finite job/arm timeout")
            for rank, process in enumerate(started):
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"SP comparison rank {rank} exited {process.exitcode}")
                # Poll a tiny status record, not the growing numerical result,
                # while measured host-file transfers share the storage device.
                path = out / f"rank{rank}" / "status.json"
                if path.exists() and json.loads(path.read_text()).get("status") == "failed":
                    raise RuntimeError(f"SP comparison rank {rank} failed; see {path}")
                process.join(timeout=.2)
        ranks = []
        for rank, process in enumerate(started):
            process.join(timeout=0)
            if process.exitcode != 0:
                raise RuntimeError(f"SP comparison rank {rank} final exit {process.exitcode}")
            result = json.loads((out / f"rank{rank}" / "run.json").read_text())
            if result.get("status") != "completed":
                raise RuntimeError(f"SP comparison rank {rank} did not complete; see {out}")
            result["process_lifecycle"] = dict(exitcode=0, reaped=not process.is_alive())
            ranks.append(result)
        return dict(seed=seed, arm=arm, ranks=ranks)
    finally:
        for process in started:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)


def _replicas_valid(audit, group, expected):
    peers = audit.get("replicas", [])
    return (audit.get("passed") is True and len(peers) == 2 and
            {peer.get("rank") for peer in peers} == {0, 1} and all(
                peer.get("owner") is (peer["rank"] in group) and
                peer.get("latent") == (expected if peer["rank"] in group else None) for peer in peers))


def _transition_valid(tx, arm, key, index, rank):
    step = 0 if arm == "fixed_sp2" else (5 if index == 0 else 10)
    source, target = ([0], [0, 1]) if index == 0 else ([0, 1], [0])
    request, response = tx.get("request", {}), tx.get("response", {})
    progress = response.get("progress")
    owner = rank in target
    return (tx.get("passed") is True and tx.get("completed_steps") == step and
            request.get("state_id") == key and request.get("source_group") == source and
            request.get("target_group") == target and request.get("expected_epoch") == index and
            response.get("phase") == "finalized" and response.get("committed") is True and
            response.get("state_id") == key and response.get("transfer_id") == request.get("transfer_id") and
            isinstance(request.get("transfer_id"), str) and bool(request["transfer_id"]) and
            response.get("group") == target and response.get("ownership_epoch") == index+1 and
            response.get("transfer_ready") is True and
            isinstance(response.get("state_sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", response["state_sha256"]) is not None and
            response.get("execution_contract") == dict(schema="ditserve.wan-sp-execution.v1", sp_linear_policy=POLICY) and
            ((isinstance(progress, dict) and progress.get("step") == step and progress.get("group") == target and
              progress.get("ownership_epoch") == index+1 and progress.get("transfer_ready") is True) if owner else progress is None) and
            set(tx.get("checks", {})) == {"committed", "identity", "ownership", "ready", "digest", "contract",
                                         "state", "continuation", "record", "progress"} and all(tx["checks"].values()))


def validate_result(result):
    """Recompute gates from both rank records. A seed subset is not three-seed evidence."""
    errors = []
    def require(condition, label):
        if not condition:
            errors.append(label)
    try:
        config = normalize_config(result.get("configuration", {}))
        require(result.get("schema") == SCHEMA and result.get("role") == ROLE and
                result.get("status") == "completed", "completed_schema")
        blocks = result.get("blocks", [])
        require([(b.get("seed"), b.get("arm")) for b in blocks] ==
                [(seed, arm) for seed in config["seeds"] for arm in arm_order(seed)], "complete_ordered_arm_matrix")
        reference_anchor = {}
        for block in blocks:
            seed, arm, ranks = block["seed"], block["arm"], block["ranks"]
            require(len(ranks) == 2 and [r.get("rank") for r in ranks] == [0, 1], "both_ranks")
            for data in ranks:
                rank = data["rank"]
                prefix = f"{seed}:{arm}:rank{rank}"
                require(data.get("seed") == seed and data.get("arm") == arm and data.get("status") == "completed"
                        and data.get("backend_closed") is True and data.get("ordinary_reference_before_distributed") is True
                        and data.get("process_lifecycle") == dict(exitcode=0, reaped=True), prefix+":lifecycle")
                require(data.get("arithmetic") == dict(torch_threads=8, interop_threads=1, matmul_tf32=False,
                        cudnn_tf32=True, attention="flash", negative_prompt_cache=False), prefix+":arithmetic")
                require(data.get("cpu_affinity", {}).get("bound_cpus") == sorted(config["cpu_masks"][config["devices"][rank]]),
                        prefix+":affinity")
                require(data.get("cuda_visible_devices") == ",".join(config["gpu_uuids"]), prefix+":physical_binding")
                init = data.get("backend_init", {})
                require((init == dict(native_unattached=True, sp_linear_policy="native", transport=None)) if arm == "native_sp1"
                        else (init.get("initialized") is True and init.get("elastic") is True and
                              init.get("sp_linear_policy") == POLICY and init.get("transport") == "host_file_v1"), prefix+":backend")
                require(not data.get("nvml_errors"), prefix+":NVML")
                references = data.get("references", [])
                require([(c.get("seed"), c.get("frames")) for c in references] == [(seed, frame) for frame in config["frames"]],
                        prefix+":reference_matrix")
                lookup = {}
                for case in references:
                    ordinary, frames = case["ordinary"], case["frames"]
                    case_key = f"seed{seed}-f{frames}"
                    expected_spec = _spec(config, frames, seed, case_key)
                    require(case.get("case") == case_key and case.get("spec") == expected_spec
                            and ordinary.get("spec") == expected_spec, prefix+":reference_recipe")
                    require(case.get("cross_card_sp1_exact") is True and ordinary.get("passed") is True and
                            len(ordinary.get("steps", [])) == 20 and len(ordinary.get("timesteps", [])) == 20 and
                            all(_valid_digest(d) for d in [ordinary.get("initial"), ordinary.get("video")]+ordinary.get("steps", [])),
                            prefix+":reference")
                    identity = {key: ordinary[key] for key in ("initial", "steps", "timesteps", "video")}
                    reference_anchor.setdefault((seed, frames), identity)
                    require(identity == reference_anchor[(seed, frames)], prefix+":cross_arm_rank_reference_exact")
                    lookup[frames] = ordinary
                trials = data.get("trials", [])
                require(len({t.get("state_id") for t in trials}) == len(trials), prefix+":unique_request_states")
                require([(t.get("frames"), t.get("mode"), t.get("scope"), t.get("index")) for t in trials] ==
                        trial_matrix(config, seed), prefix+":diagnostics_warmups_measurements")
                for trial in trials:
                    key, frames, mode, scope = trial["state_id"], trial["frames"], trial["mode"], trial["scope"]
                    label, reference = prefix+":"+key, lookup[frames]
                    require(key == f"seed{seed}-f{frames}-{arm}-{mode}-{scope}-{trial['index']}", label+":request_namespace")
                    require(trial.get("seed") == seed and trial.get("arm") == arm and trial.get("passed") is True and
                            trial.get("backend_native") is (arm == "native_sp1") and
                            trial.get("diagnostic_in_timing") is (mode == "diagnostic") and
                            trial.get("cpu_encoding_in_timing") is (scope == "full_request") and
                            trial.get("timing_scope") == TIMING_SCOPES[scope], label+":identity")
                    require(set(trial.get("release", {})) == {"states", "continuations", "ownership", "tickets",
                            "runtime_states", "scratch", "geometry", "pending"} and
                            all(value is True for value in trial["release"].values()), label+":release")
                    start, prepared, dit_end, end = (trial.get(k) for k in ("start_ns", "prepared_ns", "dit_end_ns", "end_ns"))
                    require(all(type(t) is int for t in (start, prepared, dit_end, end)) and
                            0 < start <= prepared < dit_end < end and trial.get("wall_ms") == (end-start)/1e6 and
                            trial.get("prepare_ms") == (prepared-start)/1e6 and
                            trial.get("dit_and_transfer_ms") == (dit_end-prepared)/1e6, label+":host_timing")
                    if rank == 0:
                        preparation = trial.get("prepare", {})
                        require(preparation.get("state_id") == key and all(type(preparation.get(k)) in (int, float)
                                and math.isfinite(preparation[k]) and preparation[k] >= 0 for k in
                                ("cpu_prompt_encode_ms", "gpu_prepare_ms", "model_ms")), label+":prepare_receipt")
                        require(trial.get("capacity") == capacity_gate(trial.get("memory", {}), config, start, end)
                                and trial["capacity"]["passed"], label+":capacity")
                        require(trial.get("after_release_capacity") == capacity_gate(trial.get("after_release_memory", {}), config)
                                and trial["after_release_capacity"]["passed"] and all(
                                    sample["timestamp_ns"] >= end for sample in trial["after_release_memory"]["samples"]),
                                label+":released_capacity")
                        if scope == "preencoded":
                            encoding = trial.get("preencoding", {})
                            require(type(encoding.get("start_ns")) is int and type(encoding.get("end_ns")) is int and
                                    0 < encoding["start_ns"] <= encoding["end_ns"] <= start and
                                    encoding.get("receipt", {}).get("ready") is True, label+":preencoding_outside")
                        else:
                            require("preencoding" not in trial, label+":full_encoding_inside")
                    steps = trial.get("steps", [])
                    require(len(steps) == 20, label+":20steps")
                    for step, item in enumerate(steps):
                        group, epoch = _layout(arm, step)
                        require(item.get("step") == step and item.get("group") == group and item.get("epoch") == epoch
                                and item.get("owner") is (rank in group), label+f":step{step}:ownership")
                        require(type(item.get("start_ns")) is int and type(item.get("end_ns")) is int and
                                prepared <= item["start_ns"] <= item["end_ns"] <= dit_end and
                                (step == 0 or steps[step-1]["end_ns"] <= item["start_ns"]), label+f":step{step}:timing")
                        if rank in group:
                            require(_execution(item.get("response", {}), config, arm, step, key, frames), label+f":step{step}:backend")
                            if mode == "diagnostic":
                                require(item.get("exact") is True and item.get("latent") == reference["steps"][step],
                                        label+f":step{step}:exact")
                        else:
                            require("response" not in item, label+f":step{step}:idle")
                        if mode == "diagnostic":
                            require(_replicas_valid(item.get("replicas", {}), group, reference["steps"][step]), label+":step_replicas")
                        else:
                            require(not any(k in item for k in ("latent", "exact", "replicas")), label+":normal_no_step_diagnostic")
                    initial_group = [0, 1] if arm == "fixed_sp2" else [0]
                    if mode == "diagnostic" and rank in initial_group:
                        require(trial.get("initial") == reference["initial"] and trial.get("timesteps") == reference["timesteps"],
                                label+":initial_exact")
                    final_group, unused = _layout(arm, 19)
                    if rank in final_group:
                        require(trial.get("final_latent") == reference["steps"][-1], label+":final_exact")
                    if rank == 0:
                        require(trial.get("video") == reference["video"], label+":video_exact")
                    require(_replicas_valid(trial.get("final_replicas", {}), final_group, reference["steps"][-1]), label+":final_replicas")
                    transitions = trial.get("transitions", [])
                    require(len(transitions) == (1 if arm == "fixed_sp2" else 2 if arm == "resize5to10" else 0), label+":transition_count")
                    for index, tx in enumerate(transitions):
                        require(_transition_valid(tx, arm, key, index, rank), label+":transfer_contract")
                        step = tx["completed_steps"]
                        left = prepared if step == 0 else steps[step-1]["end_ns"]
                        require(type(tx.get("start_ns")) is int and type(tx.get("end_ns")) is int and
                                left <= tx["start_ns"] <= tx["end_ns"] <= steps[step]["start_ns"], label+":transfer_in_window")
            require([r["ordinary"] for r in ranks[0]["references"]] == [r["ordinary"] for r in ranks[1]["references"]],
                    f"{seed}:{arm}:rank_references")
            require([t["state_id"] for t in ranks[0]["trials"]] == [t["state_id"] for t in ranks[1]["trials"]],
                    f"{seed}:{arm}:rank_trial_identity")
        if "summary" in result:
            require(result["summary"] == summarize(result), "summary_recomputed")
        if result.get("performance_evidence") is True:
            require(result.get("qualification", {}).get("passed") is True, "qualified_performance")
    except (TypeError, ValueError, KeyError, IndexError, AttributeError, ZeroDivisionError) as error:
        errors.append("malformed:"+repr(error))
    return dict(passed=not errors, errors=errors, exact_required=True,
                scope="frozen seed subset, four arms, both timing scopes; not online or dual-SP1 throughput")


def _assigned_gpu_seconds(trial):
    # Reservation includes the union of source/target ranks during transfer.
    total = trial["end_ns"]-trial["start_ns"]
    tx = trial["transitions"]
    if trial["arm"] == "fixed_sp2":
        total += trial["end_ns"]-tx[0]["start_ns"]
    elif trial["arm"] == "resize5to10":
        total += tx[1]["end_ns"]-tx[0]["start_ns"]
    return total/1e9


def summarize(result):
    config, blocks = result["configuration"], result["blocks"]
    pairs = []
    for seed in config["seeds"]:
        for frames in config["frames"]:
            for scope in SCOPES:
                for repeat in range(3):
                    arms = {}
                    for arm in ARMS:
                        block = next(b for b in blocks if b["seed"] == seed and b["arm"] == arm)
                        trials = [next(t for t in rank["trials"] if t["frames"] == frames and t["scope"] == scope
                                       and t["mode"] == "measured" and t["index"] == repeat) for rank in block["ranks"]]
                        wall = (max(t["end_ns"] for t in trials)-min(t["start_ns"] for t in trials))/1e6
                        steps = [[trial["steps"][step] for trial in trials if trial["steps"][step]["owner"]]
                                 for step in range(20)]
                        arms[arm] = dict(wall_ms=wall, max_rank_wall_ms=max(t["wall_ms"] for t in trials),
                                        two_gpu_envelope_seconds=2*wall/1000,
                                        assigned_group_gpu_seconds=_assigned_gpu_seconds(trials[0]),
                                        prepare_ms=trials[0]["prepare_ms"],
                                        step_ms=[(max(row["end_ns"] for row in owners)-min(row["start_ns"] for row in owners))/1e6
                                                 for owners in steps],
                                        transfer_ms=sum((tx["end_ns"]-tx["start_ns"])/1e6 for tx in trials[0]["transitions"]))
                    baseline = arms["native_sp1"]["wall_ms"]
                    pairs.append(dict(seed=seed, frames=frames, scope=scope, repeat=repeat, arms=arms,
                                      wall_reduction_percent={arm:100*(1-arms[arm]["wall_ms"]/baseline) for arm in ARMS[1:]}))
    by_case = [dict(seed=seed, frames=frames, scope=scope,
                   median_wall_ms={arm: statistics.median(p["arms"][arm]["wall_ms"] for p in pairs
                       if (p["seed"], p["frames"], p["scope"]) == (seed, frames, scope)) for arm in ARMS},
                   median_wall_reduction_percent={arm: statistics.median(p["wall_reduction_percent"][arm] for p in pairs
                       if (p["seed"], p["frames"], p["scope"]) == (seed, frames, scope)) for arm in ARMS[1:]})
               for seed in config["seeds"] for frames in config["frames"] for scope in SCOPES]
    return dict(pairs=pairs, by_case=by_case, seeds=list(config["seeds"]), three_seed_complete=config["seeds"] == list(SEEDS),
                timing_scopes=TIMING_SCOPES,
                caveats=["Fresh processes per seed/arm, cyclic arm rotation; three seeds are not a complete four-period Latin square.",
                         "Two physical GPUs are reserved for every arm; assigned-group GPU seconds are a reservation integral, not GPU active time.",
                         "No independent dual-SP1 throughput, online arrivals, queue latency or A100 claim."])


def run_study(config, out):
    config, out = normalize_config(config), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(out / "run.json")
    result = dict(schema=SCHEMA, role=ROLE, kind=ROLE, status="running", configuration=config,
                  blocks=[], qualification=dict(passed=False), performance_evidence=False, started_unix_ns=time.time_ns())
    atomic_json(out / "run.json", result)
    deadline = time.monotonic()+config["timeout_seconds"]
    try:
        for seed in config["seeds"]:
            for arm in arm_order(seed):
                result["blocks"].append(_run_block(config, seed, arm, out / f"seed{seed}" / arm, deadline))
                atomic_json(out / "run.json", result)
        result["status"] = "completed"
        result["qualification"] = validate_result(result)
        if result["qualification"]["passed"]:
            result["summary"] = summarize(result)
            result["performance_evidence"] = True
        else:
            result["status"] = "qualification_failed"
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
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
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
