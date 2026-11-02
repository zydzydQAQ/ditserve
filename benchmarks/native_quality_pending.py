#!/usr/bin/env python3
"""Recognize only completed native-quality runs whose archive awaits evaluation."""

import argparse
import json
from pathlib import Path

from native_exact import validate_run


PENDING = [{"consumer": "native_quality", "closed": False}]


def quality_deferral(record, journal):
    validate_run(record)
    benchmark = record["config"]["benchmark"]
    if benchmark.get("quality_arm") not in ("exact", "fixed", "curvature"):
        raise ValueError("not a controlled native-quality arm")
    quality = benchmark.get("trace", {}).get("manifest", {}).get("quality_requests", {})
    if quality.get("video", 0) <= 0 or benchmark.get("output_mode") != "check":
        raise ValueError("quality outputs were not requested")
    if record.get("status") != "completed" or record.get("failures") not in (None, []):
        raise ValueError("execution did not complete without failures")
    requests = record.get("requests", [])
    if not requests or any(row.get("status") != "completed" or not row.get("outputs") for row in requests):
        raise ValueError("a quality request is incomplete or has no output")
    metrics = record.get("metrics", {})
    if (metrics.get("requests") != len(requests) or metrics.get("completed") != len(requests)
            or metrics.get("failed") != 0 or metrics.get("incomplete") != 0):
        raise ValueError("request metrics do not confirm complete execution")
    phases = record.get("phases", [])
    config_phases = record["config"].get("phases", [])
    if (len(phases) != 1 or len(config_phases) != 1
            or config_phases[0].get("artifact_dependencies") != PENDING
            or phases[0].get("status") != "completed"
            or not phases[0].get("cells") or any(cell.get("status") != "completed" for cell in phases[0]["cells"])):
        raise ValueError("phase or native-quality dependencies are incomplete")
    artifacts = record.get("artifacts", [])
    if not artifacts or any(item.get("dependencies") != PENDING for item in artifacts):
        raise ValueError("an artifact has unexpected retention dependencies")
    if (journal.get("status") != "deferred" or journal.get("errors") not in (None, [])
            or journal.get("deleted") not in (None, [])):
        raise ValueError("archive is not solely awaiting native-quality evaluation")
    return {
        "status": "quality_evaluation_pending",
        "run_id": record["run_id"],
        "completed_outputs": len(requests),
        "quality_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    result = quality_deferral(json.loads(args.run.read_text()), json.loads(args.journal.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
