#!/usr/bin/env python3
"""Summarize native run records without inventing missing measurements."""

import argparse
import json
import math
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    if not 0 <= fraction <= 1:
        raise ValueError("quantile fraction must be in [0, 1]")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def ontime(row):
    return (row.get("status") == "completed" and row.get("finish_ms") is not None
            and row["finish_ms"] <= row["deadline_ms"])


def distribution(values):
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values), "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95), "p99": percentile(values, 0.99), "max": max(values),
    }


def kind_summary(rows, elapsed_ms):
    latencies = [row["finish_ms"] - row["arrival_ms"] for row in rows
                 if row.get("status") == "completed" and row.get("finish_ms") is not None]
    timely = sum(ontime(row) for row in rows)
    completed = sum(row.get("status") == "completed" for row in rows)
    return {
        "requests": len(rows), "completed": completed, "ontime": timely,
        "deadline_attainment": timely / len(rows) if rows else None,
        "goodput_rps": timely / (elapsed_ms / 1000) if elapsed_ms > 0 else None,
        "latency_ms": distribution(latencies),
    }


def merged_intervals(intervals):
    result = []
    for start, finish in sorted(intervals):
        if (type(start) not in (int, float) or type(finish) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(finish) or start < 0 or finish < start):
            raise ValueError("invalid timed execution interval")
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], finish)
        else:
            result.append([start, finish])
    return result


def approximation_summary(rows):
    videos = [row for row in rows if row["kind"] == "video" and row.get("cfg_fold")]
    if not videos:
        return None
    folded_quanta = fission_quanta = 0
    release_windows, image_users = {}, set()
    unattributed_quanta = 0
    gate_ms = []
    state_bytes = []
    for row in videos:
        stats = row["cfg_fold"]
        folded = stats.get("folded_cfg_steps")
        full = stats.get("full_cfg_steps")
        indices = stats.get("folded_step_indices")
        if type(folded) is not int or type(full) is not int or full + folded != row["spec"]["steps"]:
            raise ValueError("full and folded step counts do not cover the completed request")
        if type(indices) is not list or len(indices) != folded or len(set(indices)) != folded:
            raise ValueError("actual folded-step evidence is missing or ambiguous")
        if stats.get("final_step_exact_anchor") and row["spec"]["steps"] - 1 in indices:
            raise ValueError("final exact-anchor contract was violated")
        if stats.get("curvature_gate_ms") is not None:
            gate_ms.append(stats["curvature_gate_ms"])
        if stats.get("cfg_state_bytes") is not None:
            state_bytes.append(stats["cfg_state_bytes"])
        samples = row.get("advance_samples", [])
        full_group = None
        full_reference_seen = False
        for sample in samples:
            if not sample.get("cfg_folded"):
                if sample.get("form", "").startswith("cfg2"):
                    full_group = set(sample["group"])
                    full_reference_seen = True
                else:
                    # A previous contraction in exact SP execution is not a
                    # release caused by the later folded quantum.
                    full_group = None
                continue
            folded_quanta += 1
            if full_group is None:
                if full_reference_seen:
                    unattributed_quanta += 1
                    continue
                raise ValueError("folded execution lacks a prior full-CFG placement reference")
            if sample.get("start_ms") is None or sample.get("finish_ms") is None:
                raise ValueError("fission accounting requires timed execution intervals")
            interval = merged_intervals([(sample["start_ms"], sample["finish_ms"])])[0]
            # A folded group moved outside the latest full group cannot be
            # interpreted as contraction of that group without a layout epoch.
            if not set(sample["group"]).issubset(full_group):
                unattributed_quanta += 1
                continue
            released = sorted(full_group - set(sample["group"]))
            if released:
                fission_quanta += 1
                for gpu in released:
                    release_windows.setdefault(gpu, []).append(interval)
    merged = {gpu: merged_intervals(intervals) for gpu, intervals in release_windows.items()}
    released_gpu_ms = sum(finish - start for intervals in merged.values() for start, finish in intervals)
    image_windows = {}
    for row in rows:
        if row["kind"] != "image" or row.get("status") != "completed" or not row.get("gpu_group"):
            continue
        interval = merged_intervals([(row["start_ms"], row["finish_ms"])])[0]
        for gpu in set(row["gpu_group"]):
            for start, finish in merged.get(gpu, []):
                left, right = max(start, interval[0]), min(finish, interval[1])
                if right > left:
                    image_windows.setdefault(gpu, []).append((left, right))
                    image_users.add(row["request_id"])
    # Batch members share an execution interval. Count GPU time once, while
    # retaining the number of distinct requests that overlap that interval.
    image_overlap_ms = sum(finish - start for intervals in image_windows.values()
                           for start, finish in merged_intervals(intervals))
    missing_image_placement = sum(row['kind'] == 'image' and row.get('status') == 'completed'
                                  and not row.get('gpu_group') for row in rows)
    return {
        "video_count": len(videos), "folded_quanta": folded_quanta, "fission_quanta": fission_quanta,
        "folded_steps": sum(row["cfg_fold"]["folded_cfg_steps"] for row in videos),
        "full_cfg_steps": sum(row["cfg_fold"]["full_cfg_steps"] for row in videos),
        "released_gpu_ms_capacity": released_gpu_ms,
        "image_overlap_on_released_gpu_ms": image_overlap_ms,
        "image_requests_on_released_gpus": len(image_users),
        "image_requests_missing_placement": missing_image_placement,
        "overlap_attribution_complete": not missing_image_placement and not unattributed_quanta,
        "unattributed_folded_quanta": unattributed_quanta,
        "reuse_qualified": False,
        "curvature_gate_ms": distribution(gate_ms), "cfg_state_bytes": distribution(state_bytes),
        "accounting_scope": "per-GPU union of inferred contraction windows and image start-to-finish intervals; placement/overlap proxy only, without release ACK, utilization or causal reuse qualification",
    }


def summarize_run(record):
    cells = [(phase.get("id", "cluster"), phase.get("attempt", 1), cell)
             for phase in record["phases"] for cell in phase["cells"]]
    selectors = [(phase, attempt, cell['cell'], cell.get('order')) for phase, attempt, cell in cells]
    if len(selectors) != len(set(selectors)):
        raise ValueError('duplicate native cell identity')
    for row in record['requests']:
        possible = [key for key in selectors if key[0] == row.get('phase', 'cluster') and key[2] == row['cell']
                    and ('attempt' not in row or key[1] == row['attempt'])
                    and ('order' not in row or key[3] == row['order'])]
        if len(possible) > 1:
            raise ValueError('legacy request lacks an unambiguous attempt/order cell identity')
    summaries = []
    for phase, attempt, cell in cells:
        rows = [row for row in record["requests"]
                if row.get("phase", "cluster") == phase and row["cell"] == cell["cell"]
                and row.get("attempt", attempt) == attempt and row.get("order", cell.get("order")) == cell.get("order")]
        elapsed_ms = cell["elapsed_ms"]
        gpus = cell["gpus"]
        allocated_gpu_seconds = gpus * elapsed_ms / 1000
        completed = sum(row.get("status") == "completed" for row in rows)
        timely = sum(ontime(row) for row in rows)
        block_samples = [
            sample["elapsed_ms"] for row in rows
            for sample in row.get("advance_samples", []) if sample.get("steady") is True
        ]
        summaries.append({
            "phase": phase, "attempt": attempt, "cell": cell["cell"], "policy": cell["policy"], "order": cell.get("order"),
            "status": cell["status"], "elapsed_ms": elapsed_ms,
            "by_kind": {
                kind: kind_summary([row for row in rows if row["kind"] == kind], elapsed_ms)
                for kind in ("video", "image")
            },
            "allocated_gpu_seconds": allocated_gpu_seconds,
            "gpu_seconds_per_completed_request": allocated_gpu_seconds / completed if completed else None,
            "gpu_seconds_per_ontime_request": allocated_gpu_seconds / timely if timely else None,
            "image_batch": cell.get("image_batch"),
            "doorbell": cell.get("doorbell"),
            "observed_transfers": sum(row.get("transfers", 0) for row in rows),
            "approximation": approximation_summary(rows),
            "block_quantum_elapsed_ms": distribution(block_samples),
            "advance_sample_count": len(block_samples),
            "preemption_response_ms": distribution([
                row["start_ms"] - row["arrival_ms"] for row in rows
                if row.get("preemption_reason") == "block_preempt" and row.get("start_ms") is not None
            ]),
            "preempting_request_count": sum(row.get("preemption_reason") == "block_preempt" for row in rows),
            "preemption_stop_ms": distribution([
                sample["stop_ms"] for row in rows
                for sample in row.get("preemption_stop_samples", [])
            ]),
            "failures": [row for row in rows if row.get("status") not in ("completed",)],
        })
    return {
        "schema": "ditserve.native-metrics.v1",
        "experiment": record["experiment"], "run_id": record["run_id"],
        "status": record["status"], "run_metrics": record["metrics"],
        "used_gpu_seconds": record["budget"].get("used_gpu_seconds"),
        "cells": summaries,
        "notes": {
            "quantiles": "linear interpolation at (n-1)*p, matching the native controller",
            "goodput_rps": "completed and on-time requests divided by the finite cell wall time; not quality-qualified or sustained capacity",
            "allocated_gpu_seconds": "sum over native cell elapsed time times registered GPUs; budget.used_gpu_seconds additionally includes process startup and cleanup",
            "block_quantum_elapsed_ms": "steady quantum wall time, not a direct preemption-wait CDF",
            "preemption_response_ms": "arrival-to-first-start response of requests marked block_preempt; includes scheduling and queueing, not only the boundary stop",
            "preemption_stop_ms": "doorbell ring to a video worker-confirmed block-boundary stop on the preempted request; not the preempting request queueing delay",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.run.read_text(encoding="utf-8"))
    result = summarize_run(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "cells": len(result["cells"])}))


if __name__ == "__main__":
    main()
