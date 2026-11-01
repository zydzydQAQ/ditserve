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
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def distribution(values):
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values), "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95), "p99": percentile(values, 0.99), "max": max(values),
    }


def kind_summary(rows, elapsed_ms):
    latencies = [row["finish_ms"] - row["arrival_ms"] for row in rows if row.get("finish_ms") is not None]
    ontime = sum(row.get("finish_ms") is not None and row["finish_ms"] <= row["deadline_ms"] for row in rows)
    completed = sum(row.get("status") == "completed" for row in rows)
    return {
        "requests": len(rows), "completed": completed, "ontime": ontime,
        "deadline_attainment": ontime / len(rows) if rows else None,
        "goodput_rps": ontime / (elapsed_ms / 1000) if elapsed_ms > 0 else None,
        "latency_ms": distribution(latencies),
    }


def summarize_run(record):
    cells = record["phases"][0]["cells"]
    summaries = []
    for cell in cells:
        rows = [row for row in record["requests"] if row["cell"] == cell["cell"]]
        elapsed_ms = cell["elapsed_ms"]
        gpus = cell["gpus"]
        allocated_gpu_seconds = gpus * elapsed_ms / 1000
        completed = sum(row.get("status") == "completed" for row in rows)
        ontime = sum(row.get("finish_ms") is not None and row["finish_ms"] <= row["deadline_ms"] for row in rows)
        block_samples = [
            sample["elapsed_ms"] for row in rows
            for sample in row.get("advance_samples", []) if sample.get("steady") is True
        ]
        summaries.append({
            "cell": cell["cell"], "policy": cell["policy"], "order": cell.get("order"),
            "status": cell["status"], "elapsed_ms": elapsed_ms,
            "by_kind": {
                kind: kind_summary([row for row in rows if row["kind"] == kind], elapsed_ms)
                for kind in ("video", "image")
            },
            "allocated_gpu_seconds": allocated_gpu_seconds,
            "gpu_seconds_per_completed_request": allocated_gpu_seconds / completed if completed else None,
            "gpu_seconds_per_ontime_request": allocated_gpu_seconds / ontime if ontime else None,
            "block_quantum_elapsed_ms": distribution(block_samples),
            "advance_sample_count": len(block_samples),
            "preemption_response_ms": distribution([
                row["start_ms"] - row["arrival_ms"] for row in rows
                if row.get("preemption_reason") == "block_preempt" and row.get("start_ms") is not None
            ]),
            "preempting_request_count": sum(row.get("preemption_reason") == "block_preempt" for row in rows),
            "failures": [row for row in rows if row.get("status") not in ("completed",)],
        })
    return {
        "schema": "ditserve.native-metrics.v1",
        "experiment": record["experiment"], "run_id": record["run_id"],
        "status": record["status"], "run_metrics": record["metrics"],
        "used_gpu_seconds": record["budget"].get("used_gpu_seconds"),
        "cells": summaries,
        "notes": {
            "allocated_gpu_seconds": "sum over native cell elapsed time times registered GPUs; budget.used_gpu_seconds additionally includes process startup and cleanup",
            "block_quantum_elapsed_ms": "steady quantum wall time, not a direct preemption-wait CDF",
            "preemption_response_ms": "arrival-to-first-start response of requests marked block_preempt; includes scheduling and queueing, not only the boundary stop",
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
