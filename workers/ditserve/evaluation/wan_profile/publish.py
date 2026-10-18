"""Refresh the CPU-only Wan figure and its verified Excel/report archive.

This is an artifact publisher, not an experiment controller. It never starts
workers, queries GPUs, mutates measurement manifests, or closes raw-data
dependencies. --wait waits for the currently active software pass to settle.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from ...runtime import ROOT
from ...results import finalize, verify
from .figure import generate_figure


MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_EVIDENCE_NODES = 200000


def _bounded(value):
    """Accept complete scalar summaries, rejecting raw event/tensor bodies."""
    nodes = 0
    def check(item, depth=0):
        nonlocal nodes
        nodes += 1
        if nodes > MAX_EVIDENCE_NODES or depth > 64:
            raise ValueError("scalar evidence exceeds node/depth limit")
        if isinstance(item, dict):
            if "traceEvents" in item:
                raise ValueError("raw profiler event bodies cannot enter the scalar archive")
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("evidence keys must be strings")
                check(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                check(child, depth + 1)
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("nonfinite scalar evidence")
        elif not (item is None or type(item) in (str, bool, int, float)):
            raise ValueError("unsupported scalar evidence type")
    check(value)
    return nodes


def _snapshot(path, *, schemas=None):
    """Read once, bind the exact bytes and preserve failures without promotion."""
    path = Path(path)
    result = dict(status="failed", source={"path": str(path), "bytes": None, "sha256": None}, payload=None)
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_EVIDENCE_BYTES:
                raise ValueError("source is not a bounded ordinary scalar metadata file")
            raw = stream.read(MAX_EVIDENCE_BYTES + 1)
            after = os.fstat(stream.fileno())
        result["source"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if len(raw) > MAX_EVIDENCE_BYTES:
            raise ValueError("scalar evidence exceeds byte limit")
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("scalar evidence changed during read")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("scalar evidence root must be an object")
        nodes = _bounded(value)
        if schemas and value.get("schema", value.get("schema_version")) not in schemas:
            raise ValueError("unexpected scalar evidence schema")
        result.update(status="loaded", payload=value, scalar_node_count=nodes)
    except FileNotFoundError:
        result.update(status="missing", error="source metadata has not been produced")
    except (OSError, ValueError, TypeError) as error:
        result.update(status="failed", error=type(error).__name__ + ": " + str(error))
    return result


def _failed_hardware(snapshot):
    return dict(schema="ditserve.wan_hardware_capture.v1", status="failed", measured=False,
                hardware_phase_qualified=False,
                failure=snapshot.get("error", "hardware summary unavailable"))


def _hardware_snapshot(path, study_snapshot):
    result = _snapshot(path, schemas=("ditserve.wan_hardware_capture.v1", "ditserve.nsys_gpu_metrics.v1"))
    if result["status"] != "missing":
        return result
    study = study_snapshot.get("payload") or {}
    embedded = study.get("hardware_capture")
    if embedded is None:
        return result
    source = dict(study_snapshot["source"], json_pointer="/hardware_capture",
                  storage="embedded_in_measurement_run", original_summary_source=study.get("hardware_capture_source"))
    try:
        if not isinstance(embedded, dict) or embedded.get("schema") != "ditserve.wan_hardware_capture.v1":
            raise ValueError("embedded hardware_capture is not a validated capture-wrapper schema")
        nodes = _bounded(embedded)
        return dict(status="loaded", source=source, payload=copy.deepcopy(embedded), scalar_node_count=nodes)
    except (ValueError, TypeError) as error:
        return dict(status="failed", source=source, payload=None, error=type(error).__name__ + ": " + str(error))


def _profile_evidence(study):
    evidence = []
    for p, phase in enumerate(study.get("phases", [])):
        for c, cell in enumerate(phase.get("cells", [])):
            for o, operation in enumerate(cell.get("operations", [])):
                for r, rank in enumerate(operation.get("ranks", [])):
                    if rank.get("mode") not in ("clean", "trace", "hardware"):
                        continue
                    evidence.append(dict(source_pointer="/phases/%d/cells/%d/operations/%d/ranks/%d" % (p, c, o, r),
                        phase=phase.get("id"), phase_status=phase.get("status"),
                        phase_qualification=copy.deepcopy(phase.get("qualification")),
                        final_occupancy=copy.deepcopy(phase.get("final_occupancy")),
                        result=copy.deepcopy(rank)))
    _bounded(evidence)
    return evidence


def build_record(analysis, study_snapshot, hardware_snapshot, ledger_snapshot, *, work, captured,
                 hardware_study_snapshot=None, prior_capture_snapshot=None):
    """Build a self-contained scalar publication; never finalize or clean here."""
    study = study_snapshot.get("payload") or {}
    hardware = hardware_snapshot.get("payload") or _failed_hardware(hardware_snapshot)
    ledger = ledger_snapshot.get("payload")
    measured = {name: sum(row["value"] is not None for row in analysis["metrics"] if row["panel"] == name)
                for name in analysis["panel_definitions"]}
    pending = [name for name in ("activity", "shares", "sm", "tensor", "dram")
               if any(row["value"] is None for row in analysis["metrics"] if row["panel"] == name)]
    failures = [dict(source="measurement_run", detail=copy.deepcopy(item)) for item in study.get("failures", [])]
    snapshots = [("measurement_run", study_snapshot), ("hardware_summary", hardware_snapshot),
                 ("measurement_ledger", ledger_snapshot)]
    if hardware_study_snapshot is not None:
        snapshots.append(("hardware_measurement_run", hardware_study_snapshot))
    for label, snapshot in snapshots:
        if snapshot["status"] == "failed":
            failures.append(dict(source=label, error=snapshot.get("error"), source_binding=snapshot["source"]))
    if hardware.get("failure") and hardware_snapshot["status"] == "loaded":
        failures.append(dict(source="hardware_capture", error=hardware["failure"],
                             status=hardware.get("status"), measured=hardware.get("measured")))
    cumulative = ledger if ledger is not None else study.get("budget", {})
    used = cumulative.get("used_gpu_seconds")
    if type(used) not in (int, float) or not math.isfinite(used):
        used = None
    status = "completed" if not pending and not failures and not cumulative.get("active") and used is not None else "incomplete"

    def number(panel, series, frames):
        matches = [row["value"] for row in analysis["metrics"]
                   if row["panel"] == panel and row["series"] == series and row["frames"] == frames]
        value = matches[0] if len(matches) == 1 else None
        return "unmeasured" if value is None else "%.2f" % value

    # results._report sorts keys and shows the first 30 short scalars. Keep
    # readable conclusions here and the exact, complete evidence outside metrics.
    metrics = {
        "01_status": status, "02_study_status": study.get("status", "unverified"),
        "03_hardware_status": hardware.get("status", "unverified"),
        "04_settled_GPU_seconds": used, "05_unresolved_failures": len(failures),
        "06_kernel_scope": "kernel interval union / request or GPU window; not SM utilization",
        "07_DiT_share_scope": "sum of real DiT kernel durations; exclusive module attribution",
        "08_hardware_scope": "sampled device activity; DRAM interface cycles, not peak bandwidth",
        "09_sample_scope": "new n=1 per shape; historical clean n=3 different prompt/seed inputs",
    }
    for index, frames in enumerate((17, 33, 65), 10):
        metrics["%02d_%df_kernel_active_pct" % (index, frames)] = "request %s; GPU window %s" % (
            number("activity", "request", frames), number("activity", "gpu_window", frames))
    for index, frames in enumerate((17, 33, 65), 13):
        metrics["%02d_%df_DiT_share_pct" % (index, frames)] = "self %s; cross %s; FFN %s; other %s" % tuple(
            number("shares", name, frames) for name in ("self_attention", "cross_attention", "ffn", "other"))
    for start, panel, key, label in ((16, "sm", "sm_active_pct", "SM"),
            (19, "tensor", "tensor_active_pct", "Tensor"),
            (22, "dram", "dram_read_active_pct", "DRAM_read"), (25, "dram", "dram_write_active_pct", "DRAM_write")):
        for index, frames in enumerate((17, 33, 65), start):
            metrics["%02d_%df_%s_pct" % (index, frames, label)] = "request %s; denoise NVTX %s" % (
                number(panel, "request/" + key, frames), number(panel, "denoise/" + key, frames))
    shares = [row for row in analysis["metrics"] if row["panel"] == "shares" and row["frames"] == 65
              and row["value"] is not None]
    largest = max(shares, key=lambda row: row["value"]) if len(shares) == 4 else None
    conclusion = ("65f DiT largest kernel share: %s %.2f%%; n=%d." %
                  (largest["series"], largest["value"], largest["n"]) if largest else
                  "Insufficient validated attribution; missing metrics remain unmeasured.")
    metrics.update({"28_conclusion": conclusion,
        "29_NVTX_scope": "denoise CPU range may exclude GPU tail; complete request synchronizes",
        "30_missing_panels": ",".join(pending) or "none",
        "31_snapshot_utc": captured, "32_counts": measured,
        "33_figure_png": analysis["figures"]["png"]["path"], "34_figure_svg": analysis["figures"]["svg"]["path"]})
    phase_evidence = [{key: copy.deepcopy(phase[key]) for key in
        ("id", "attempt", "status", "qualification", "charge", "error", "cleanup_error", "final_occupancy")
        if key in phase} for phase in study.get("phases", [])]
    record = dict(schema="ditserve.run.v1", experiment="wan-profile-corrected", run_id="analysis01", status=status,
        config=dict(scope="CPU-only chart and self-contained scalar evidence publication", source_work=str(work),
            original_archives_read_only=True, no_gpu_execution=True, no_raw_cleanup=True),
        metrics=metrics, figure_analysis=copy.deepcopy(analysis),
        measurement_evidence=dict(
            hardware_capture=copy.deepcopy(hardware_snapshot),
            software_and_hardware_profiles=_profile_evidence(study),
            measurement_run_source=copy.deepcopy(study_snapshot["source"]),
            measurement_run_status=study.get("status", "unverified"),
            measurement_failures=copy.deepcopy(study.get("failures", [])), phase_evidence=phase_evidence,
            accounting_source=copy.deepcopy(ledger_snapshot),
            retention_scope="All bounded summary fields retained exactly; raw timelines/samples are not embedded or deleted.",
            raw_re_evaluation="requires retained trace/SQLite/Nsight inputs; historical scalar inspection is self-contained"),
        phases=[dict(id="figure_assembly", status="completed"),
                dict(id="new_measurements", status="completed" if not pending else "pending")],
        requests=[], failures=failures, artifacts=[],
        budget=dict(scope="Publisher is CPU-only; cumulative measurement ledger is preserved separately, never reset.",
            publisher_gpu_seconds=0, active=None, settled_measurement_gpu_seconds=used,
            measurement_run_snapshot=copy.deepcopy(study.get("budget")),
            measurement_ledger_snapshot=copy.deepcopy(ledger),
            accounting_status=("active_or_unsettled" if cumulative.get("active") else
                               "settled_ledger_snapshot" if ledger is not None else "measurement_run_snapshot_only")),
        retention=dict(pending_dependencies=["run01_measurements_and_validation"] if status != "completed" else [],
            measurement_raw_cleanup="not_performed; source-run dependencies remain authoritative"))
    implementation = Path(__file__)
    record["publisher_source"] = dict(path=str(implementation), bytes=implementation.stat().st_size,
                                     sha256=hashlib.sha256(implementation.read_bytes()).hexdigest())
    if hardware_study_snapshot is not None:
        recovery = hardware_study_snapshot.get("payload") or {}
        record["measurement_evidence"]["hardware_measurement_run"] = copy.deepcopy(hardware_study_snapshot)
        record["budget"]["hardware_measurement_run_snapshot"] = copy.deepcopy(recovery.get("budget"))
        record["metrics"]["35_hardware_run_status"] = recovery.get("status", "unverified")
        record["metrics"]["36_hardware_phase_scope"] = (
            "hardware02 executes hardware_metrics only; software_trace evidence is reused from run01")
    if prior_capture_snapshot is not None:
        record["measurement_evidence"]["prior_capture_attempt"] = copy.deepcopy(prior_capture_snapshot)
        record["metrics"]["37_prior_capture_status"] = (
            prior_capture_snapshot.get("payload") or {}).get("status", "unverified")
        record["metrics"]["38_recovery_status"] = (
            "validated replacement; original failure retained" if status == "completed" else
            "replacement not yet validated; original failure retained")
    _bounded(record)
    return record, pending


def refresh():
    work = ROOT / ".cache/results-work/wan-profile-corrected/run01"
    source = work / "run.json"
    hardware = work / "nsys/summary.json"
    destination = ROOT / "results/wan-profile-corrected"
    study_snapshot = _snapshot(source)
    hardware_snapshot = _hardware_snapshot(hardware, study_snapshot)
    study = study_snapshot.get("payload") or {"status": "unverified", "phases": []}
    hardware_study_snapshot = prior_capture_snapshot = None
    recovery_work = ROOT / ".cache/results-work/wan-profile-corrected/hardware02"
    recovery_capture = ROOT / ".cache/results-work/wan-profile-corrected/capture02/summary.json"
    reanalyzed_capture = recovery_capture.with_name("summary.reanalyzed.json")
    if reanalyzed_capture.exists():
        recovery_capture = reanalyzed_capture
    if recovery_work.exists() or recovery_capture.exists():
        # Once a replacement attempt exists, a missing/failed replacement must
        # stay unmeasured, never borrow the previous run's numeric qualification.
        prior_capture_snapshot = hardware_snapshot
        hardware_study_snapshot = _snapshot(recovery_work / "run.json")
        hardware_snapshot = _hardware_snapshot(recovery_capture, hardware_study_snapshot)
    ledger_path = Path(study.get("budget_ledger", ROOT / ".state/wan-profile-corrected/budget.json"))
    ledger_snapshot = _snapshot(ledger_path, schemas=("ditserve.budget.v1",))
    extra = {} if hardware_study_snapshot is None else {
        "hardware_study": hardware_study_snapshot.get("payload") or {}}
    analysis = generate_figure(ROOT / "results/wan-breakdown/run01",
        ROOT / "results/wan-breakdown/analysis01",
        destination / "figures01/wan_profile_overview",
        study=study, hardware=hardware_snapshot.get("payload") or _failed_hardware(hardware_snapshot), **extra)
    for item in analysis["sources"]:
        snapshot = {"study": study_snapshot, "hardware": hardware_snapshot,
                    "hardware_study": hardware_study_snapshot}.get(item["id"])
        if snapshot:
            item.update(snapshot["source"], snapshot_status=snapshot["status"])
    analysis_work = ROOT / ".cache/results-work/wan-profile-corrected/analysis01"
    analysis_work.mkdir(parents=True, exist_ok=True)
    captured = datetime.now(timezone.utc).isoformat()
    record, pending = build_record(analysis, study_snapshot, hardware_snapshot, ledger_snapshot,
                                   work=work, captured=captured,
                                   hardware_study_snapshot=hardware_study_snapshot,
                                   prior_capture_snapshot=prior_capture_snapshot)
    # These files are small scalar/control records. Raw measurement sources are
    # never put on this analysis run's deletion list.
    temporary = analysis_work / "run.json.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(analysis_work / "run.json")
    published = finalize(analysis_work, destination / "analysis01", prune=False)
    if published.get("status") != "completed":
        raise RuntimeError("analysis archive failed: " + json.dumps(published))
    verified = verify(destination / "analysis01")
    return dict(status=record["status"], study_status=analysis["study_status"], missing_panels=pending,
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
