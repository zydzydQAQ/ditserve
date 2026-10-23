"""Fresh-process DiT/VAE lifecycle controls; no performance or overlap claim.

The outer guarded suite alone owns GPU admission, wall budget and container
cleanup. This wrapper starts one child at a time, never imports torch in the
parent, and reaps each child before starting the next arm.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from . import workflow_serving_overlap as overlap
from .workflow_budget import atomic_json


SCHEMA = "ditserve.workflow.overlap-isolated.v1"
ARMS = [
    dict(id="serial", mode="serial", decode_a_on_b_stream=False),
    dict(id="streams-current-decode", mode="streams", decode_a_on_b_stream=False),
    dict(id="streams-shared-decode", mode="streams", decode_a_on_b_stream=True),
]
REQUIRED_ARMS = ("serial", "streams-shared-decode")
OUTPUTS = ("a_latent", "a_video", "b_latent", "b_video")


def normalize_config(config):
    value = overlap.normalize_config(config)
    for name, default in dict(
            lifecycle_pairs=3, warmup_pairs=0, arm_specs=ARMS, arm_timeout_seconds=240,
            emergency_free_bytes=512 * 1024**2,
            steady_allocated_growth_bytes=16 * 1024**2,
            steady_reserved_growth_bytes=64 * 1024**2,
            steady_nvml_used_growth_bytes=64 * 1024**2).items():
        value.setdefault(name, default)
    if not isinstance(value.get("gpu_uuid"), str) or not re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value["gpu_uuid"]):
        raise ValueError("gpu_uuid must identify the one guarded physical GPU")
    if type(value["lifecycle_pairs"]) is not int or not 3 <= value["lifecycle_pairs"] <= 16:
        raise ValueError("lifecycle_pairs must be a bounded 3..16 consecutive pairs")
    if type(value["warmup_pairs"]) is not int or not 0 <= value["warmup_pairs"] <= 16:
        raise ValueError("warmup_pairs must be a fixed, bounded 0..16 pairs")
    for name in ("arm_timeout_seconds", "emergency_free_bytes", "steady_allocated_growth_bytes",
                 "steady_reserved_growth_bytes", "steady_nvml_used_growth_bytes"):
        if type(value[name]) is not int or value[name] < (1 if name in (
                "arm_timeout_seconds", "emergency_free_bytes") else 0):
            raise ValueError("invalid lifecycle limit " + name)
    if value["arm_specs"] != ARMS:
        raise ValueError("freeze all three ordered causal controls with their exact stream policies")
    mask = value["cpu_masks"].get(value["device"])
    if (not isinstance(mask, list) or len(mask) != value["cpu_threads"]
            or any(type(cpu) is not int or cpu < 0 for cpu in mask) or len(set(mask)) != len(mask)):
        raise ValueError("an explicit CPU mask is required before child model loading")
    for name in ("reuse_streams", "trim_after_references", "release_b_before_a_decode"):
        if value.get(name) is not True:
            raise ValueError(name + " must remain enabled in every isolated arm")
    value["memory_diagnostics"] = True
    return value


def child_config(config, arm, pair):
    value = json.loads(json.dumps(config))
    mask = value["cpu_masks"][value["device"]]
    value.update(device="cuda:0", cpu_masks={"cuda:0": mask}, frame_pairs=[list(pair)],
                 decode_a_on_b_stream=arm["decode_a_on_b_stream"], selected_arm=dict(arm))
    return value


def _identity():
    return dict(pid=os.getpid(), start_ticks=int(Path("/proc/self/stat").read_text().rsplit(
        ")", 1)[1].split()[19]))


def _stream_identity(worker, config, mode):
    streams = overlap._arm_streams(worker, config, mode)
    return [int(stream.cuda_stream) for stream in streams]


def _boundary(worker, config, label):
    worker.torch.cuda.synchronize(worker.device)
    result = overlap._memory_boundary(worker, config, label)
    # A synchronous NVML sample after state release is separate from the hot
    # interval's sampler. No cache flush or new CUDA allocation occurs here.
    sampler = overlap._start_sampler(config)
    sampler.stop()
    valid = [row for row in sampler.samples if type(row.get("memory_free_bytes")) is int
             and type(row.get("memory_used_bytes")) is int
             and type(row.get("memory_total_bytes")) is int and row["memory_total_bytes"] > 0]
    result.update(nvml=valid[-1] if valid else None, nvml_errors=list(sampler.errors),
                  state_count=len(worker._states),
                  scope="synchronized post-release allocator and actual NVML boundary; no cache trim")
    return result


def boundary_stop_reason(boundary, config):
    sample = boundary.get("nvml")
    if boundary.get("nvml_errors") or not sample:
        return "missing_or_failed_nvml_boundary"
    if sample["memory_free_bytes"] < config["emergency_free_bytes"]:
        return "emergency_memory_floor"
    if boundary.get("state_count") != 0:
        return "request_states_not_released"
    return None


def growth_gate(boundaries, config):
    """First measured pair establishes the steady-state comparison baseline."""
    if len(boundaries) < config["lifecycle_pairs"]:
        return dict(status="unverified", passed=False, observed_pairs=len(boundaries),
                    required_pairs=config["lifecycle_pairs"], reason="incomplete consecutive lifecycle")
    fields = (("allocated_bytes", "steady_allocated_growth_bytes"),
              ("reserved_bytes", "steady_reserved_growth_bytes"),
              ("nvml_used_bytes", "steady_nvml_used_growth_bytes"))
    observed = []
    for boundary in boundaries:
        if boundary.get("nvml_errors") or not boundary.get("nvml"):
            return dict(status="unverified", passed=False, reason="missing post-release NVML evidence")
        observed.append(dict(allocated_bytes=boundary.get("allocated_bytes"),
                             reserved_bytes=boundary.get("reserved_bytes"),
                             nvml_used_bytes=boundary["nvml"].get("memory_used_bytes")))
    if any(type(row.get(field)) is not int or row[field] < 0 for row in observed for field, _ in fields):
        return dict(status="unverified", passed=False, reason="missing allocator or NVML values")
    metrics = {}
    for field, limit_name in fields:
        values = [row[field] for row in observed]
        growth = max(0, max(values[1:]) - values[0])
        metrics[field] = dict(values=values, baseline_bytes=values[0],
                              maximum_growth_bytes=growth, final_delta_bytes=values[-1]-values[0],
                              limit_bytes=config[limit_name], passed=growth <= config[limit_name])
    passed = all(row["passed"] for row in metrics.values())
    return dict(status="passed" if passed else "failed", passed=passed, metrics=metrics,
                observed_pairs=len(boundaries), baseline="after first measured pair and state release",
                caveat="bounded observation of these consecutive pairs; not a proof of leak freedom")


def _hashes(row):
    return {key: row[key] for key in OUTPUTS}


def _finish_child_result(result, config):
    rounds = result["rounds"]
    warmups = config["warmup_pairs"]
    total = warmups + config["lifecycle_pairs"]
    complete = len(rounds) == total
    numeric = complete and all(row["measurement"]["correctness"]["passed"] and
                               row["repeat_exact"] for row in rounds)
    def boundary_capacity(boundary):
        nvml = boundary.get("nvml") or {}
        return (not boundary.get("nvml_errors") and type(nvml.get("memory_free_bytes")) is int
                and type(nvml.get("memory_total_bytes")) is int and nvml["memory_total_bytes"] > 0
                and nvml["memory_free_bytes"] >= max(1024**3, nvml["memory_total_bytes"] * .05))
    capacity = complete and all(row["measurement"]["memory_margin_gate"]["eligible"]
                                and boundary_capacity(row["after_release"]) for row in rounds)
    stable = growth_gate([row["after_release"] for row in rounds[warmups:]], config)
    stable.update(excluded_warmup_pairs=warmups,
                  measured_pair_indices=[row["index"] for row in rounds[warmups:]])
    cleanup = complete and all(row["after_release"]["state_count"] == 0 for row in rounds)
    stream_reused = complete and all(row["stream_ids"] == result["stream_ids"] for row in rounds)
    passed = numeric and capacity and stable["passed"] and cleanup and stream_reused
    result.update(steady_state=stable, completed_pairs=len(rounds),
                  expected_total_pairs=total,
                  completed_warmup_pairs=min(len(rounds), warmups),
                  completed_measured_pairs=max(0, len(rounds)-warmups),
                  unrun_pairs=list(range(len(rounds), total)),
                  unrun_warmup_pairs=list(range(min(len(rounds), warmups), warmups)),
                  unrun_measured_pairs=list(range(max(0, len(rounds)-warmups), config["lifecycle_pairs"])),
                  qualification=dict(passed=passed, scopes={
                      "numeric": dict(status="passed" if numeric else "unverified" if not complete else "failed",
                                      exact_required=True, scope="all warmup and measured pairs"),
                      "capacity": dict(status="passed" if capacity else "unverified" if not rounds else "failed",
                                       scope="all warmup and measured hot intervals and post-release boundaries"),
                      "steady_state": dict(status=stable["status"]),
                      "lifecycle": dict(status="passed" if cleanup and stream_reused else "unverified",
                                        states_released=cleanup, fixed_streams=stream_reused),
                      "performance": dict(status="unverified", reason="lifecycle diagnostics only"),
                      "semantic": dict(status="unverified")}))
    if result.get("error"):
        result["status"] = "failed"
    elif passed:
        result["status"] = "completed"
    elif not complete:
        result["status"] = "incomplete"
    elif not numeric:
        result["status"] = "qualification_failed"
    elif not capacity:
        result["status"] = "capacity_failed"
    else:
        result["status"] = "lifecycle_failed"


def run_child(config, out):
    config, out = normalize_config(config), Path(out)
    arm = config.get("selected_arm")
    if arm not in ARMS or len(config["frame_pairs"]) != 1 or config["device"] != "cuda:0":
        raise ValueError("child needs one selected arm, one frame pair and visible cuda:0")
    if "torch" in sys.modules or os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"]:
        raise RuntimeError("child UUID visibility must be established before importing torch")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(out / "run.json")
    result = dict(schema=SCHEMA, role="isolated_lifecycle_arm", status="running",
                  configuration=config, process=_identity(), started_unix_ns=time.time_ns(),
                  rounds=[], performance_evidence=False, overlap_requires_cuda_trace=True)
    atomic_json(out / "run.json", result)
    try:
        worker = overlap.make_worker(config, "cuda:0", out)
        if worker.torch.cuda.device_count() != 1:
            raise RuntimeError("child must see exactly the one bound GPU")
        result["cpu_affinity"] = worker.workflow_cpu_affinity
        timeline = overlap.Timeline("cuda:0")
        references = overlap._references(worker, config, out, timeline)
        result["references"] = [dict(role=role, frames=frames, path=ref["directory"], **ref["hashes"])
                                for (role, frames), ref in sorted(references.items())]
        worker.torch.cuda.synchronize(worker.device)
        result["before_startup_trim"] = overlap._memory_boundary(worker, config, "after_reference_warmup")
        worker.torch.cuda.empty_cache()
        result["startup_cache_trims"] = 1
        result["baseline"] = _boundary(worker, config, "after_reference_trim")
        result["stream_ids"] = _stream_identity(worker, config, arm["mode"])
        reason = boundary_stop_reason(result["baseline"], config)
        if reason:
            result["stop_reason"] = reason
        else:
            a, b = config["frame_pairs"][0]
            first_hashes = None
            for index in range(config["warmup_pairs"] + config["lifecycle_pairs"]):
                warmup = index < config["warmup_pairs"]
                measured = overlap._run_arm(worker, config, references, a, b, arm["mode"],
                                            index, "lifecycle_warmup" if warmup else "lifecycle", timeline)
                after = _boundary(worker, config, f"after_pair_{index}_release")
                hashes = _hashes(measured)
                first_hashes = hashes if first_hashes is None else first_hashes
                row = dict(index=index, phase="warmup" if warmup else "measured",
                           measured_index=None if warmup else index-config["warmup_pairs"],
                           measurement=measured, after_release=after,
                           repeat_exact=hashes == first_hashes,
                           stream_ids=_stream_identity(worker, config, arm["mode"]))
                result["rounds"].append(row)
                atomic_json(out / f"pair-{index}.json", row)
                atomic_json(out / "run.json", result)
                reason = boundary_stop_reason(after, config)
                gate = measured["memory_margin_gate"]
                if gate.get("errors") or gate.get("minimum_free_bytes") is None:
                    reason = reason or "missing_or_failed_hot_memory_sampling"
                elif gate["minimum_free_bytes"] < config["emergency_free_bytes"]:
                    reason = reason or "emergency_hot_memory_floor"
                if not measured["correctness"]["passed"] or not row["repeat_exact"]:
                    reason = reason or "numeric_mismatch"
                if row["stream_ids"] != result["stream_ids"]:
                    reason = reason or "stream_pool_recreated"
                if reason:
                    result["stop_reason"] = reason
                    break
        result["phases"] = timeline.records
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        _finish_child_result(result, config)
        result["finished_unix_ns"] = time.time_ns()
        atomic_json(out / "run.json", result)
    return result


def _launch_child(config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / "config.json"
    if input_path.exists() or (directory / "run.json").exists():
        raise FileExistsError(directory)
    atomic_json(input_path, config)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=config["gpu_uuid"],
                       OMP_NUM_THREADS=str(config["cpu_threads"]),
                       MKL_NUM_THREADS=str(config["cpu_threads"]),
                       OPENBLAS_NUM_THREADS=str(config["cpu_threads"]))
    command = [sys.executable, "-m", "ditserve.evaluation.workflow_overlap_isolated",
               "--child", "--config", str(input_path), "--out", str(directory)]
    started = time.time_ns()
    timed_out = False
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr)
        try:
            process.wait(timeout=config["arm_timeout_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    # Every child is reaped here before the caller can start the next arm.
    path = directory / "run.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    launch = dict(pid=process.pid, returncode=process.returncode, timed_out=timed_out,
                  reaped=process.poll() is not None, started_unix_ns=started,
                  finished_unix_ns=time.time_ns(), result_path=str(path),
                  result_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None)
    atomic_json(directory / "child-launch.json", launch)
    return result, launch


def aggregate(arms, config):
    required = [row for row in arms if row["id"] in REQUIRED_ARMS]
    expected = len(config["frame_pairs"]) * len(REQUIRED_ARMS)
    total = config["warmup_pairs"] + config["lifecycle_pairs"]
    required_complete = len(required) == expected and all(
        row["launch"]["reaped"] and row["launch"]["returncode"] == 0 and not row["launch"]["timed_out"]
        and row["result"].get("completed_pairs") == total for row in required)
    cross_arm, control_comparisons = [], []
    for pair in config["frame_pairs"]:
        relevant = [row for row in required if row["frame_pair"] == pair]
        signatures = [row["result"].get("output_hashes", []) for row in relevant]
        exact = (len(signatures) == 2 and all(len(rows) == total for rows in signatures)
                 and all(value == signatures[0][0] for rows in signatures for value in rows))
        cross_arm.append(dict(frames_a=pair[0], frames_b=pair[1], exact=exact,
                              expected_pairs_per_arm=total,
                              scope="required serial and shared-decode arms, all warmup and measured pairs"))
        serial = next((row for row in relevant if row["id"] == "serial"), None)
        reference = serial["result"].get("output_hashes", []) if serial else []
        controls = [row for row in arms if row["frame_pair"] == pair and row["id"] not in REQUIRED_ARMS]
        for row in controls:
            values = row["result"].get("output_hashes", [])
            comparable = bool(reference) and bool(values)
            control_comparisons.append(dict(id=row["id"], frames_a=pair[0], frames_b=pair[1],
                observed_pairs=len(values), expected_pairs=total,
                complete=len(values) == total,
                exact=(all(value == reference[0] for value in values) if comparable else None),
                scope="every available control output versus the independent serial arm; absent pairs do not pass"))
    numeric = required_complete and all(row["result"].get("qualification", {}).get("scopes", {}).get(
        "numeric", {}).get("status") == "passed" for row in required) and all(row["exact"] for row in cross_arm)
    capacity = required_complete and all(row["result"].get("qualification", {}).get("scopes", {}).get(
        "capacity", {}).get("status") == "passed" for row in required)
    stable = required_complete and all(row["result"].get("steady_state", {}).get("passed") for row in required)
    passed = required_complete and all(row["result"].get("qualification", {}).get("passed")
                                      for row in required) and all(row["exact"] for row in cross_arm)
    return dict(passed=passed, required_arms=list(REQUIRED_ARMS), cross_arm=cross_arm,
                control_cross_arm=control_comparisons, scopes={
        "numeric": dict(status="passed" if numeric else "failed" if required_complete else "unverified",
                        exact_required=True),
        "capacity": dict(status="passed" if capacity else "failed" if required_complete else "unverified"),
        "steady_state": dict(status="passed" if stable else "failed" if required_complete else "unverified"),
        "performance": dict(status="unverified", reason="fresh-process lifecycle diagnostics only"),
        "semantic": dict(status="unverified")},
        control_scope="streams-current-decode is retained diagnostic evidence; its capacity failure cannot qualify or disqualify the independently tested repair")


def _compact_child(result):
    keys = ("status", "process", "completed_pairs", "expected_total_pairs",
            "completed_warmup_pairs", "completed_measured_pairs",
            "unrun_pairs", "unrun_warmup_pairs", "unrun_measured_pairs", "qualification", "steady_state",
            "stop_reason", "error", "baseline", "stream_ids", "cpu_affinity", "references",
            "startup_cache_trims", "before_startup_trim")
    value = {key: result[key] for key in keys if key in result}
    value["output_hashes"] = [_hashes(row["measurement"]) for row in result.get("rounds", [])]
    value["memory_series"] = [dict(index=row["index"], phase=row.get("phase", "measured"),
        measured_index=row.get("measured_index", row["index"]), after_release=row["after_release"],
        peak_allocated_bytes=row["measurement"]["peak_allocated_bytes"],
        peak_reserved_bytes=row["measurement"]["peak_reserved_bytes"],
        memory_margin_gate=row["measurement"]["memory_margin_gate"]) for row in result.get("rounds", [])]
    return value


def run_isolated(config, out):
    config, out = normalize_config(config), Path(out)
    if "torch" in sys.modules:
        raise RuntimeError("isolated wrapper parent must never initialize torch/CUDA")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(out / "run.json")
    result = dict(schema=SCHEMA, role="isolated_lifecycle_controls", status="running",
                  configuration=config, started_unix_ns=time.time_ns(), arms=[],
                  qualification=dict(passed=False), performance_evidence=False,
                  overlap_requires_cuda_trace=True,
                  budget_owner="outer guarded suite; no independent child GPU budget",
                  scope="capacity, output exactness and bounded post-release memory growth; no speedup claim")
    atomic_json(out / "run.json", result)
    try:
        for pair in config["frame_pairs"]:
            for arm in config["arm_specs"]:
                directory = out / f"{arm['id']}-a{pair[0]}-b{pair[1]}"
                child, launch = _launch_child(child_config(config, arm, pair), directory)
                result["arms"].append(dict(id=arm["id"], frame_pair=list(pair), launch=launch,
                                           result=_compact_child(child)))
                atomic_json(out / "run.json", result)
                if not launch["reaped"]:
                    raise RuntimeError("previous child was not reaped; refusing another arm")
        result["qualification"] = aggregate(result["arms"], config)
        result["status"] = "completed" if result["qualification"]["passed"] else "qualification_failed"
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
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    result = (run_child if args.child else run_isolated)(config, args.out)
    print(json.dumps(dict(status=result["status"], manifest=str(args.out / "run.json"))))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
