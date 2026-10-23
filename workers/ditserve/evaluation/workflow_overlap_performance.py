"""Fresh-process serial/shared-VAE timing after lifecycle qualification.

Two independent process blocks use AB/BA order. Each arm has two fixed warmups
and three technical timing repeats; these are not six independent workloads.
The outer guarded suite owns GPU admission, wall budget, and container cleanup.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

from . import workflow_overlap_isolated as lifecycle
from . import workflow_serving_overlap as overlap
from .workflow_budget import atomic_json

SCHEMA = "ditserve.workflow.overlap-performance.v1"
ARMS = [dict(id="serial", mode="serial", decode_a_on_b_stream=False),
        dict(id="streams-shared-decode", mode="streams", decode_a_on_b_stream=True)]
OUTPUTS = lifecycle.OUTPUTS


def normalize_config(config):
    value = overlap.normalize_config(config)
    defaults = dict(process_blocks=2, warmup_pairs=2, lifecycle_pairs=3,
                    arm_specs=ARMS, arm_timeout_seconds=270,
                    emergency_free_bytes=512 * 1024**2,
                    steady_allocated_growth_bytes=16 * 1024**2,
                    steady_reserved_growth_bytes=64 * 1024**2,
                    steady_nvml_used_growth_bytes=64 * 1024**2)
    for key, default in defaults.items():
        value.setdefault(key, default)
    for key, expected in dict(process_blocks=2, warmup_pairs=2, repeats=3,
                              lifecycle_pairs=3, height=480, width=832, steps=20).items():
        if type(value[key]) is not int or value[key] != expected:
            raise ValueError(f"{key} is fixed at {expected} for this bounded comparison")
    scope = value.get("experiment_scope", "qualified_17_17")
    allowed_pairs = ([[17, 17]],) if scope == "qualified_17_17" else (
        [[17, 33]], [[33, 17]], [[33, 33]]) if scope == "long_frame_pairs_v1" else ()
    if value["frame_pairs"] not in allowed_pairs or value["arm_specs"] != ARMS:
        raise ValueError("frame pair requires its explicitly registered lifecycle-qualified scope")
    if not isinstance(value.get("gpu_uuid"), str) or not re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value["gpu_uuid"]):
        raise ValueError("gpu_uuid must bind the one guarded physical GPU")
    mask = value["cpu_masks"].get(value["device"])
    if (not isinstance(mask, list) or len(mask) != value["cpu_threads"] or
            any(type(cpu) is not int or cpu < 0 for cpu in mask) or len(set(mask)) != len(mask)):
        raise ValueError("an explicit CPU mask is required before child model loading")
    for key in ("reuse_streams", "trim_after_references", "release_b_before_a_decode"):
        if value[key] is not True:
            raise ValueError(key + " must be enabled equally in all arms")
    if value["memory_diagnostics"] is not False:
        raise ValueError("allocator diagnostics must stay outside normal request timing")
    for key in ("arm_timeout_seconds", "emergency_free_bytes", "steady_allocated_growth_bytes",
                "steady_reserved_growth_bytes", "steady_nvml_used_growth_bytes"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError("invalid positive bound " + key)
    # Retain the previously qualified lifecycle limits; changing them needs a
    # separately registered experiment, rather than an easier performance gate.
    for key in ("steady_allocated_growth_bytes", "steady_reserved_growth_bytes",
                "steady_nvml_used_growth_bytes", "emergency_free_bytes"):
        if value[key] != defaults[key]:
            raise ValueError("lifecycle bound must remain unchanged: " + key)
    return value


def arm_schedule(config):
    return [(block, arm) for block in range(config["process_blocks"])
            for arm in (ARMS if block % 2 == 0 else list(reversed(ARMS)))]


def child_config(config, arm, block):
    value = json.loads(json.dumps(config))
    value.update(device="cuda:0", cpu_masks={"cuda:0": config["cpu_masks"][config["device"]]},
                 selected_arm=dict(arm), process_block=block,
                 decode_a_on_b_stream=arm["decode_a_on_b_stream"])
    return value


def _boundary(worker, config, label):
    # This synchronizes only after _run_arm has stopped its full-request timer,
    # released both states, and stopped its hot-interval NVML sampler.
    return lifecycle._boundary(worker, dict(config, memory_diagnostics=True), label)


def _boundary_eligible(boundary):
    nvml = boundary.get("nvml") or {}
    total, free = nvml.get("memory_total_bytes"), nvml.get("memory_free_bytes")
    return (not boundary.get("nvml_errors") and boundary.get("state_count") == 0 and
            type(total) is int and total > 0 and type(free) is int and
            free >= max(1024**3, math.ceil(total * .05)))


def _finish_child(result, config):
    rows = result["rounds"]
    complete = len(rows) == config["warmup_pairs"] + config["repeats"]
    numeric = complete and all(row["measurement"]["correctness"]["passed"] and
                               row["repeat_exact"] for row in rows)
    capacity = (complete and _boundary_eligible(result.get("baseline", {})) and
                all(row["measurement"]["memory_margin_gate"]["eligible"] and
                    _boundary_eligible(row["after_release"]) for row in rows))
    stable = lifecycle.growth_gate([row["after_release"] for row in rows[config["warmup_pairs"]:]], config)
    cleanup = complete and all(row["after_release"]["state_count"] == 0 for row in rows)
    streams = complete and all(row["stream_ids"] == result["stream_ids"] for row in rows)
    normal = complete and all(row["measurement"].get("memory_diagnostics") is False and
                              row["measurement"].get("performance_evidence") is (row["phase"] == "measured")
                              for row in rows)
    passed = numeric and capacity and stable["passed"] and cleanup and streams and normal
    result.update(completed_pairs=len(rows), completed_warmup_pairs=min(2, len(rows)),
                  completed_measured_pairs=max(0, len(rows)-2), steady_state=stable,
                  qualification=dict(passed=passed, numeric=numeric, capacity=capacity,
                                     fixed_streams=streams, states_released=cleanup,
                                     normal_timing=normal), performance_evidence=passed,
                  status="failed" if result.get("error") else "completed" if passed else "qualification_failed")


def run_child(config, out):
    config, out = normalize_config(config), Path(out)
    arm = config.get("selected_arm")
    if arm not in ARMS or config["device"] != "cuda:0" or config.get("process_block") not in (0, 1):
        raise ValueError("child requires one registered arm, process block and visible cuda:0")
    if "torch" in sys.modules or os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"]:
        raise RuntimeError("bind the child UUID before importing torch")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(out / "run.json")
    result = dict(schema=SCHEMA, role="isolated_performance_arm", status="running",
                  configuration=config, process=lifecycle._identity(), started_unix_ns=time.time_ns(),
                  rounds=[], performance_evidence=False, timing_scope=overlap.TIMING_SCOPE,
                  overlap_requires_cuda_trace=True)
    atomic_json(out / "run.json", result)
    try:
        worker = overlap.make_worker(config, "cuda:0", out)
        if worker.torch.cuda.device_count() != 1:
            raise RuntimeError("child must see exactly the bound GPU")
        result["cpu_affinity"] = worker.workflow_cpu_affinity
        timeline = overlap.Timeline("cuda:0")
        references = overlap._references(worker, config, out, timeline)
        result["references"] = [dict(role=role, frames=frames, path=ref["directory"], **ref["hashes"])
                                for (role, frames), ref in sorted(references.items())]
        worker.torch.cuda.synchronize(worker.device)
        worker.torch.cuda.empty_cache()
        result["startup_cache_trims"] = 1
        result["baseline"] = _boundary(worker, config, "after_reference_trim")
        result["stream_ids"] = lifecycle._stream_identity(worker, config, arm["mode"])
        reason = lifecycle.boundary_stop_reason(result["baseline"], config)
        if reason:
            result["stop_reason"] = reason
        else:
            first_hashes = None
            for index in range(5):
                warmup = index < 2
                frames_a, frames_b = config["frame_pairs"][0]
                measured = overlap._run_arm(worker, config, references, frames_a, frames_b, arm["mode"], index,
                                            "performance_warmup" if warmup else "timed", timeline)
                after = _boundary(worker, config, f"after_pair_{index}_release")
                hashes = {key: measured[key] for key in OUTPUTS}
                first_hashes = hashes if first_hashes is None else first_hashes
                row = dict(index=index, phase="warmup" if warmup else "measured",
                           measured_index=None if warmup else index-2, measurement=measured,
                           after_release=after, repeat_exact=hashes == first_hashes,
                           stream_ids=lifecycle._stream_identity(worker, config, arm["mode"]))
                result["rounds"].append(row)
                atomic_json(out / f"pair-{index}.json", row)
                atomic_json(out / "run.json", result)
                reason = lifecycle.boundary_stop_reason(after, config)
                if not measured["memory_margin_gate"].get("eligible"):
                    reason = reason or "hot_memory_margin_not_qualified"
                if not measured["correctness"]["passed"] or not row["repeat_exact"]:
                    reason = reason or "numeric_mismatch"
                if row["stream_ids"] != result["stream_ids"]:
                    reason = reason or "stream_pool_recreated"
                if not _boundary_eligible(after):
                    reason = reason or "post_release_memory_margin_not_qualified"
                if reason:
                    result["stop_reason"] = reason
                    break
        result["phases"] = timeline.records
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        _finish_child(result, config)
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
                       OMP_NUM_THREADS=str(config["cpu_threads"]), MKL_NUM_THREADS=str(config["cpu_threads"]),
                       OPENBLAS_NUM_THREADS=str(config["cpu_threads"]))
    command = [sys.executable, "-m", "ditserve.evaluation.workflow_overlap_performance", "--child",
               "--config", str(input_path), "--out", str(directory)]
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
    path = directory / "run.json"
    raw = path.read_bytes() if path.exists() else None
    launch = dict(pid=process.pid, returncode=process.returncode, timed_out=timed_out,
                  reaped=process.poll() is not None, started_unix_ns=started,
                  finished_unix_ns=time.time_ns(), result_path=str(path),
                  result_sha256=hashlib.sha256(raw).hexdigest() if raw is not None else None)
    atomic_json(directory / "child-launch.json", launch)
    result = json.loads(raw) if raw is not None else {}
    return result, launch


def _valid_hash(row):
    return (isinstance(row, dict) and row.get("finite") is True and
            isinstance(row.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) and
            isinstance(row.get("shape"), list) and bool(row["shape"]) and
            all(type(v) is int and v > 0 for v in row["shape"]) and row.get("dtype") == "float32")


def _statistics(result):
    blocks = []
    for block in range(2):
        arms = {row["id"]: row["result"] for row in result["arms"] if row["block"] == block}
        summaries = {}
        for arm in ARMS:
            child = arms[arm["id"]]
            measured = [row["measurement"] for row in child["rounds"][2:]]
            summaries[arm["id"]] = dict(
                wall_ms=[row["wall_ms"] for row in measured],
                median_wall_ms=statistics.median(row["wall_ms"] for row in measured),
                stage_pair_wall_ms=[row["stage_pair_wall_ms"] for row in measured],
                median_stage_pair_wall_ms=statistics.median(row["stage_pair_wall_ms"] for row in measured))
        serial, shared = (summaries[arm["id"]] for arm in ARMS)
        blocks.append(dict(block=block, arms=summaries,
                           full_request_speedup=serial["median_wall_ms"] / shared["median_wall_ms"],
                           stage_pair_speedup=serial["median_stage_pair_wall_ms"] /
                                              shared["median_stage_pair_wall_ms"]))
    return dict(blocks=blocks,
                median_block_speedup=statistics.median(row["full_request_speedup"] for row in blocks),
                median_block_stage_pair_speedup=statistics.median(row["stage_pair_speedup"] for row in blocks),
                independent_process_blocks=2, technical_repeats_per_arm_per_block=3,
                inference="descriptive only; one seed and frame pair, two process blocks; no significance claim",
                timing_scope=overlap.TIMING_SCOPE, online_latency=False, cuda_overlap_proven=False)


def _qualification(passed):
    return dict(passed=passed, scopes={
        "numeric": dict(status="passed" if passed else "unverified", exact_required=True),
        "capacity": dict(status="passed" if passed else "unverified"),
        "lifecycle": dict(status="passed" if passed else "unverified"),
        "performance": dict(status="measured" if passed else "unverified"),
        "semantic": dict(status="unverified")})


def validate_result(result):
    """Fail closed using raw per-round evidence, not self-reported pass booleans."""
    errors, checks = [], 0
    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)
    try:
        config = normalize_config(result["configuration"])
        check(result.get("schema") == SCHEMA and result.get("role") == "isolated_performance_comparison",
              "wrong result schema or role")
        check(result.get("status") == "completed", "comparison did not complete")
        check(result.get("performance_evidence") is True, "performance gate absent")
        check(result.get("qualification") == _qualification(True), "parent qualification scopes")
        check(result.get("timing_scope") == overlap.TIMING_SCOPE and result.get("overlap_requires_cuda_trace") is True,
              "timing/overlap scope changed")
        rows = result["arms"]
        schedule = arm_schedule(config)
        check(len(rows) == len(schedule), "four independent process arms required")
        identities, all_hashes, previous_end = [], [], None
        for position, (row, (block, arm)) in enumerate(zip(rows, schedule)):
            prefix = f"arm {position}"
            check(row.get("block") == block and row.get("id") == arm["id"], prefix + " AB/BA order")
            child, launch = row["result"], row["launch"]
            check(child.get("configuration") == child_config(config, arm, block), prefix + " config binding")
            check(child.get("schema") == SCHEMA and child.get("role") == "isolated_performance_arm",
                  prefix + " child schema")
            check(child.get("status") == "completed" and not child.get("error") and not child.get("stop_reason"),
                  prefix + " child did not complete cleanly")
            check(launch.get("returncode") == 0 and launch.get("reaped") is True and
                  launch.get("timed_out") is False, prefix + " child exit/reap")
            check(isinstance(launch.get("result_sha256"), str) and
                  re.fullmatch(r"[0-9a-f]{64}", launch["result_sha256"]) is not None,
                  prefix + " child artifact digest")
            serialized = (json.dumps(child, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
            check(launch.get("result_sha256") == hashlib.sha256(serialized).hexdigest(),
                  prefix + " child artifact content binding")
            process = child["process"]
            check(type(process.get("pid")) is int and process["pid"] > 0 and
                  type(process.get("start_ticks")) is int and process["start_ticks"] > 0 and
                  process["pid"] == launch.get("pid"), prefix + " process identity")
            identities.append((process["pid"], process["start_ticks"]))
            times = [launch["started_unix_ns"], child["started_unix_ns"], child["finished_unix_ns"],
                     launch["finished_unix_ns"]]
            check(all(type(t) is int and t > 0 for t in times) and times == sorted(times),
                  prefix + " process time bounds")
            check(previous_end is None or previous_end <= times[0], prefix + " child overlap")
            previous_end = times[-1]
            check(child.get("startup_cache_trims") == 1, prefix + " one startup-only cache trim")
            check(child.get("cpu_affinity", {}).get("bound_cpus") == sorted(config["cpu_masks"][config["device"]]),
                  prefix + " CPU binding")
            check(child.get("completed_pairs") == 5 and child.get("completed_warmup_pairs") == 2 and
                  child.get("completed_measured_pairs") == 3, prefix + " fixed repeat counts")
            check(_boundary_eligible(child.get("baseline", {})), prefix + " startup memory/state boundary")
            check(child.get("baseline", {}).get("nvml", {}).get("uuid") == config["gpu_uuid"],
                  prefix + " startup NVML UUID binding")
            streams = child["stream_ids"]
            check(isinstance(streams, list) and len(streams) == 2 and
                  all(type(stream) is int and stream >= 0 for stream in streams) and
                  ((streams[0] == streams[1]) if arm["mode"] == "serial" else (streams[0] != streams[1])),
                  prefix + " explicit serial/shared stream roles")
            refs = child["references"]
            frames_a, frames_b = config["frame_pairs"][0]
            check(len(refs) == 2 and {(ref.get("role"), ref.get("frames")) for ref in refs} == {("A", frames_a), ("B", frames_b)},
                  prefix + " ordinary reference coverage")
            references = {ref["role"].lower() + "_" + output: ref[output]
                          for ref in refs for output in ("latent", "video")}
            check(all(_valid_hash(value) for value in references.values()), prefix + " finite reference hashes")
            rounds = child["rounds"]
            check(len(rounds) == 5, prefix + " missing warmup/timed outputs")
            last_finished = None
            for index, entry in enumerate(rounds):
                label = prefix + f" round {index}"
                warmup = index < 2
                measured = entry["measurement"]
                check(entry.get("index") == index and entry.get("phase") == ("warmup" if warmup else "measured") and
                      entry.get("measured_index") == (None if warmup else index-2), label + " phase/index")
                check(measured.get("mode") == arm["mode"] and measured.get("repeat") == index and
                      measured.get("purpose") == ("performance_warmup" if warmup else "timed"), label + " timing arm")
                check(measured.get("memory_diagnostics") is False and measured.get("memory_boundaries") == [] and
                      measured.get("performance_evidence") is (not warmup), label + " diagnostic timing contamination")
                check(measured.get("a_decode_stream") == ("b_stream" if arm["decode_a_on_b_stream"] else "current_stream"),
                      label + " A decode stream")
                check(entry.get("stream_ids") == streams and _boundary_eligible(entry["after_release"]),
                      label + " stream reuse/release/memory boundary")
                check(entry["after_release"].get("nvml", {}).get("uuid") == config["gpu_uuid"],
                      label + " post-release NVML UUID binding")
                hashes = {key: measured[key] for key in OUTPUTS}
                check(all(_valid_hash(value) for value in hashes.values()) and hashes == references,
                      label + " outputs differ from ordinary runtime")
                check(entry.get("repeat_exact") is True and measured.get("correctness") ==
                      dict(passed=True, exact={key: True for key in OUTPUTS}), label + " exactness record")
                all_hashes.append(hashes)
                start, finish = measured["started_monotonic_ns"], measured["finished_monotonic_ns"]
                stage_start, stage_finish = measured["stage_started_monotonic_ns"], measured["stage_finished_monotonic_ns"]
                check(all(type(t) is int and t > 0 for t in (start, stage_start, stage_finish, finish)) and
                      start <= stage_start < stage_finish <= finish and
                      (last_finished is None or start > last_finished), label + " timing nesting")
                last_finished = finish
                for key, duration in (("wall_ms", (finish-start)/1e6),
                                      ("stage_pair_wall_ms", (stage_finish-stage_start)/1e6)):
                    observed = measured.get(key)
                    check(type(observed) in (int, float) and math.isfinite(observed) and observed > 0 and
                          math.isclose(observed, duration, rel_tol=1e-12, abs_tol=1e-6), label + " " + key)
                errors_nvml = measured.get("nvml", {}).get("errors", [])
                gate = overlap.memory_margin_gate(measured["nvml_samples"], errors_nvml, start, finish)
                check(gate["eligible"] and measured.get("memory_margin_gate") == gate, label + " raw NVML margin")
                check(all(sample.get("uuid") == config["gpu_uuid"] for sample in measured["nvml_samples"]),
                      label + " hot NVML UUID binding")
                check(measured.get("sample_interval_seconds") == config["sample_interval_seconds"],
                      label + " NVML sample interval")
            growth = lifecycle.growth_gate([entry["after_release"] for entry in rounds[2:]], config)
            check(growth["passed"] and child.get("steady_state") == growth, prefix + " raw post-release growth")
            expected = json.loads(json.dumps(child))
            _finish_child(expected, config)
            check(child.get("qualification") == expected["qualification"] and expected["qualification"]["passed"] and
                  child.get("performance_evidence") is True, prefix + " child qualification")
        check(len(identities) == 4 and len(set(identities)) == 4, "fresh process identity reused or missing")
        check(len(all_hashes) == 20 and all(value == all_hashes[0] for value in all_hashes),
              "cross-arm/process outputs are not exact")
        check(result.get("summary") == _statistics(result), "summary does not match raw full/stage durations")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, ZeroDivisionError) as exc:
        errors.append("malformed evidence: " + str(exc))
    return dict(status="passed" if not errors else "failed", passed=not errors, checks=checks, errors=errors)


def run_performance(config, out):
    config, out = normalize_config(config), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(out / "run.json")
    result = dict(schema=SCHEMA, role="isolated_performance_comparison", status="running",
                  configuration=config, started_unix_ns=time.time_ns(), arms=[], performance_evidence=False,
                  qualification=_qualification(False),
                  overlap_requires_cuda_trace=True, timing_scope=overlap.TIMING_SCOPE,
                  budget_owner="outer guarded suite; no independent child GPU budget",
                  replication="AB/BA independent process blocks; 2 warmups and 3 technical repeats per arm")
    atomic_json(out / "run.json", result)
    try:
        for block, arm in arm_schedule(config):
            child, launch = _launch_child(child_config(config, arm, block), out / f"block-{block}-{arm['id']}")
            result["arms"].append(dict(block=block, id=arm["id"], result=child, launch=launch))
            atomic_json(out / "run.json", result)
            if not launch.get("reaped") or launch.get("returncode") != 0 or launch.get("timed_out"):
                raise RuntimeError("child did not exit successfully and reap; refusing next arm")
            if child.get("status") != "completed" or not child.get("qualification", {}).get("passed"):
                raise RuntimeError("child failed qualification; refusing further performance work")
        result.update(status="completed", performance_evidence=True, summary=_statistics(result),
                      qualification=_qualification(True))
        result["validation"] = validate_result(result)
        if not result["validation"]["passed"]:
            result.update(status="qualification_failed", performance_evidence=False,
                          qualification=_qualification(False))
    except Exception:
        result.update(status="failed", performance_evidence=False, qualification=_qualification(False),
                      error=traceback.format_exc())
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
    result = (run_child if args.child else run_performance)(json.loads(args.config.read_text()), args.out)
    print(json.dumps(dict(status=result["status"], manifest=str(args.out / "run.json"))))
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
