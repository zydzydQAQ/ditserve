"""Refresh the CPU-only Wan figure and its verified Excel/report archive.

This is an artifact publisher, not an experiment controller. It never starts
workers, queries GPUs, mutates measurement manifests, or closes raw-data
dependencies. --wait waits for the currently active software pass to settle.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from ...runtime import ROOT
from ...results import finalize, verify
from .figure import generate_figure


def refresh():
    work = ROOT / ".cache/results-work/wan-profile-corrected/run01"
    source = work / "run.json"
    hardware = work / "nsys/summary.json"
    destination = ROOT / "results/wan-profile-corrected"
    analysis = generate_figure(ROOT / "results/wan-breakdown/run01",
        ROOT / "results/wan-breakdown/analysis01",
        destination / "figures01/wan_profile_overview",
        study=source, hardware=hardware if hardware.is_file() else None)
    measured = {name: sum(row["value"] is not None for row in analysis["metrics"] if row["panel"] == name)
                for name in analysis["panel_definitions"]}
    pending = [name for name in ("activity", "shares", "sm", "tensor", "dram")
               if any(row["value"] is None for row in analysis["metrics"] if row["panel"] == name)]
    analysis_work = ROOT / ".cache/results-work/wan-profile-corrected/analysis01"
    analysis_work.mkdir(parents=True, exist_ok=True)
    archive_status = "completed" if not pending else "incomplete"
    captured = datetime.now(timezone.utc).isoformat()
    record = dict(schema="ditserve.run.v1", experiment="wan-profile-corrected", run_id="analysis01",
        status=archive_status, config=dict(scope="CPU-only chart and scalar evidence publication",
            source_work=str(work), original_archives_read_only=True,
            no_gpu_execution=True, no_raw_cleanup=True),
        metrics={"01_snapshot_utc": captured, "02_measurement_status": analysis["status"],
            "03_study_status": analysis["study_status"], "04_missing_panels": pending,
            "05_figure_png": analysis["figures"]["png"]["path"],
            "06_figure_svg": analysis["figures"]["svg"]["path"],
            "07_scope": "A-D are verified historical measurements; E-I require new validated measurements. Missing is not zero.",
            "08_raw_retention": "Measurement trace/Nsight inputs remain owned by run01; preserve until numerical, attribution and hardware coverage review closes dependencies.",
            "09_hardware_auth": "The capture script invokes sudo Nsight only; model uid must stay 1002. Interactive sudo authentication may be required.",
            "10_counts": measured, "figure_analysis": analysis},
        phases=[dict(id="figure_assembly", status="completed"),
                dict(id="new_measurements", status="completed" if not pending else "pending")],
        requests=[], failures=[], artifacts=[],
        budget=dict(scope="This analysis process is CPU-only; GPU accounting stays in the unchanged measurement ledger.",
                    gpu_seconds=0, active=None),
        retention=dict(pending_dependencies=["run01_measurements_and_validation"] if pending else []))
    # These files are small scalar/control records. Raw measurement sources are
    # never put on this analysis run's deletion list.
    implementation = Path(__file__)
    record["publisher_source"] = dict(path=str(implementation), bytes=implementation.stat().st_size,
                                     sha256=hashlib.sha256(implementation.read_bytes()).hexdigest())
    temporary = analysis_work / "run.json.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(analysis_work / "run.json")
    published = finalize(analysis_work, destination / "analysis01", prune=False)
    if published.get("status") != "completed":
        raise RuntimeError("analysis archive failed: " + json.dumps(published))
    verified = verify(destination / "analysis01")
    return dict(status=archive_status, study_status=analysis["study_status"], missing_panels=pending,
                figures=analysis["figures"], archive=verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="CPU-only wait for the current controller pass to settle")
    args = parser.parse_args()
    source = ROOT / ".cache/results-work/wan-profile-corrected/run01/run.json"
    if args.wait:
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            current = json.loads(source.read_text())
            if current.get("status") not in ("running", "waiting_resource", "admitted", "stopping", "prepared"):
                break
            time.sleep(15)
    print(json.dumps(refresh(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
