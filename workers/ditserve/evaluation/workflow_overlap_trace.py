"""One separately profiled shared-stream pair after lifecycle/performance gates.

The outer suite owns admission, the GPU budget, profiler, and cleanup. Timings
from this capture are diagnostics, never normal request performance evidence.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from . import workflow_overlap_isolated as lifecycle
from . import workflow_overlap_performance as performance
from . import workflow_serving_overlap as overlap
from .workflow_budget import atomic_json

SCHEMA = "ditserve.workflow.overlap_trace_capture.v1"


def normalize_config(config):
    value = performance.normalize_config(config)
    if value["frame_pairs"][0] not in ([17, 17], [17, 33], [33, 17], [33, 33]):
        raise ValueError("unregistered trace frame pair")
    if value["device"] != "cuda:0":
        raise ValueError("capture sees exactly the bound physical GPU as cuda:0")
    value["decode_a_on_b_stream"] = True
    return value


def capture_range(config):
    a, b = config["frame_pairs"][0]
    meta = dict(frames=a, mode="streams", peer_frames=b, purpose="trace", repeat=0,
                request_id=f"P4-trace-a{a}-b{b}-r0-streams", variant="P4")
    return "ditserve.workflow/P4/stage_pair|" + json.dumps(meta, sort_keys=True, separators=(",", ":"))


def validate_result(result):
    errors = []
    def require(ok, message):
        if not ok:
            errors.append(message)
    try:
        config = normalize_config(result["configuration"])
        require(result.get("schema") == SCHEMA and result.get("status") == "completed", "incomplete capture")
        require(result.get("performance_evidence") is False, "profile must not grant timing qualification")
        require(result.get("cuda_overlap_proven") is False, "capture cannot prove CUDA kernel overlap")
        require(result.get("capture_range") == capture_range(config), "NVTX capture selection differs")
        require(result.get("cpu_affinity", {}).get("bound_cpus") == sorted(config["cpu_masks"]["cuda:0"]), "CPU binding")
        require(result.get("startup_cache_trims") == 1, "one startup-only trim required")
        baseline = result["baseline"]
        require(performance._boundary_eligible(baseline), "startup capacity/lifetime")
        require(baseline.get("nvml", {}).get("uuid") == config["gpu_uuid"], "GPU binding")
        streams = result["stream_ids"]
        require(len(streams) == 2 and all(type(s) is int and s >= 0 for s in streams) and
                streams[0] != streams[1], "two explicitly owned streams")
        a, b = config["frame_pairs"][0]
        refs = result["references"]
        require(len(refs) == 2 and {(r["role"], r["frames"]) for r in refs} == {("A", a), ("B", b)}, "reference coverage")
        hashes = {r["role"].lower() + "_" + k: r[k] for r in refs for k in ("latent", "video")}
        require(all(performance._valid_hash(v) for v in hashes.values()), "finite ordinary references")
        rows = result["rounds"]
        require(len(rows) == 3, "two warmups and one captured pair required")
        stages = [phase for phase in result["phases"] if phase.get("name") == "P4/stage_pair"]
        require(len(stages) == 3, "one recorded stage per warmup/captured pair")
        require(sum(phase.get("nvtx_label") == capture_range(config) for phase in result["phases"]) == 1,
                "exactly one actual NVTX capture range")
        last_finished = None
        for index, row in enumerate(rows):
            measured = row["measurement"]
            purpose = "trace_warmup" if index < 2 else "trace"
            require(row.get("index") == index and measured.get("purpose") == purpose, "trace/warmup order")
            require(measured.get("repeat") == (index if index < 2 else 0), "repeat index")
            require(measured.get("mode") == "streams" and measured.get("a_decode_stream") == "b_stream", "shared decode policy")
            require(measured.get("memory_diagnostics") is False and measured.get("memory_boundaries") == [] and
                    measured.get("performance_evidence") is False, "diagnostic scope")
            require({k: measured[k] for k in performance.OUTPUTS} == hashes, "ordinary runtime output equality")
            require(measured.get("correctness") == dict(passed=True, exact={k: True for k in performance.OUTPUTS}), "exactness receipt")
            start, finish = measured["started_monotonic_ns"], measured["finished_monotonic_ns"]
            stage_start, stage_finish = measured["stage_started_monotonic_ns"], measured["stage_finished_monotonic_ns"]
            require(all(type(t) is int and t > 0 for t in (start, stage_start, stage_finish, finish)) and
                    start <= stage_start < stage_finish <= finish and
                    (last_finished is None or start > last_finished), "positive, nested, ordered capture times")
            last_finished = finish
            for key, duration in (("wall_ms", (finish-start)/1e6),
                                  ("stage_pair_wall_ms", (stage_finish-stage_start)/1e6)):
                value = measured.get(key)
                require(type(value) in (int, float) and math.isfinite(value) and value > 0 and
                        math.isclose(value, duration, rel_tol=1e-12, abs_tol=1e-6), "capture " + key)
            repeat = index if index < 2 else 0
            meta = dict(frames=a, mode="streams", peer_frames=b, purpose=purpose, repeat=repeat,
                        request_id=f"P4-{purpose}-a{a}-b{b}-r{repeat}-streams", variant="P4")
            stage = stages[index]
            require(stage.get("metadata") == meta and stage.get("nvtx_label") ==
                    "ditserve.workflow/P4/stage_pair|" + json.dumps(meta, sort_keys=True, separators=(",", ":")),
                    "actual stage metadata and NVTX selection")
            require(stage.get("status") == "completed" and stage.get("errors") == [] and
                    stage.get("timing_scope") == "host_interval" and
                    type(stage.get("start_ns")) is int and type(stage.get("end_ns")) is int and
                    stage_start <= stage["start_ns"] < stage["end_ns"] <= stage_finish,
                    "actual stage completed within measured window")
            gate = overlap.memory_margin_gate(measured["nvml_samples"], measured.get("nvml", {}).get("errors", []), start, finish)
            require(gate["eligible"] and gate == measured.get("memory_margin_gate"), "hot capacity")
            require(all(s.get("uuid") == config["gpu_uuid"] for s in measured["nvml_samples"]), "sample GPU UUID")
            timestamps = [s.get("timestamp_ns") for s in measured["nvml_samples"]]
            require(all(type(t) is int and t > 0 for t in timestamps) and
                    all(left < right for left, right in zip(timestamps, timestamps[1:])), "ordered NVML timestamps")
            require(measured.get("sample_interval_seconds") == config["sample_interval_seconds"], "NVML sample cadence")
            require(performance._boundary_eligible(row["after_release"]), "post-release state/capacity")
            require(row["after_release"].get("nvml", {}).get("uuid") == config["gpu_uuid"], "release GPU UUID")
            require(row.get("stream_ids") == streams, "bounded stream reuse")
        require(result.get("qualification", {}).get("passed") is True, "capture qualification")
    except (KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
        errors.append("malformed capture: " + str(exc))
    return dict(passed=not errors, errors=errors, scope="capture correctness and capacity; kernel overlap requires parsed CUDA events")


def run_capture(config, out):
    config, out = normalize_config(config), Path(out)
    if "torch" in sys.modules:
        raise RuntimeError("bind UUID before importing CUDA")
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_uuid"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(out / "run.json")
    result = dict(schema=SCHEMA, configuration=config, status="running", started_unix_ns=time.time_ns(),
                  performance_evidence=False, capture_range=capture_range(config), rounds=[],
                  cuda_overlap_proven=False, qualification=dict(passed=False))
    try:
        worker = overlap.make_worker(config, "cuda:0", out)
        if worker.torch.cuda.device_count() != 1:
            raise RuntimeError("capture must see one physical GPU")
        result["cpu_affinity"] = worker.workflow_cpu_affinity
        timeline = overlap.Timeline("cuda:0")
        refs = overlap._references(worker, config, out, timeline)
        result["references"] = [dict(role=role, frames=frames, path=ref["directory"], **ref["hashes"])
                                for (role, frames), ref in sorted(refs.items())]
        worker.torch.cuda.synchronize(worker.device)
        worker.torch.cuda.empty_cache()
        result["startup_cache_trims"] = 1
        result["baseline"] = performance._boundary(worker, config, "after_reference_trim")
        result["stream_ids"] = lifecycle._stream_identity(worker, config, "streams")
        if not performance._boundary_eligible(result["baseline"]):
            raise RuntimeError("startup capacity/lifetime failed")
        a, b = config["frame_pairs"][0]
        for i in range(3):
            measured = overlap._run_arm(worker, config, refs, a, b, "streams", i if i < 2 else 0,
                                        "trace_warmup" if i < 2 else "trace", timeline)
            boundary = performance._boundary(worker, config, f"after_pair_{i}_release")
            result["rounds"].append(dict(index=i, measurement=measured, after_release=boundary,
                                         stream_ids=lifecycle._stream_identity(worker, config, "streams")))
            atomic_json(out / "run.json", result)
            if not measured["correctness"]["passed"] or not measured["memory_margin_gate"]["eligible"] or not performance._boundary_eligible(boundary):
                raise RuntimeError("trace pair numeric/capacity/release failed")
        result.update(status="completed", qualification=dict(passed=True), phases=timeline.records)
        check = validate_result(result)
        result["audit"] = check
        if not check["passed"]:
            result.update(status="qualification_failed", qualification=dict(passed=False))
    except Exception:
        result.update(status="failed", error=traceback.format_exc(), qualification=dict(passed=False))
    finally:
        result["finished_unix_ns"] = time.time_ns()
        atomic_json(out / "run.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_capture(json.loads(args.config.read_text()), args.out)
    print(json.dumps(dict(status=result["status"], manifest=str(args.out / "run.json"))))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
