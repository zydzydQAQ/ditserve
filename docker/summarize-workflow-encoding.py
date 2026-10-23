#!/usr/bin/env python3
"""Refresh a CPU-only, derived encoding/replay evidence summary.

    taskset -c 9 nice -n 19 python3 docker/summarize-workflow-encoding.py \
        --root /mnt/ssd0/zyd/ditserve-data/workflow/run01

Reads JSON and array headers only; never imports torch or rewrites raw evidence.
Stage aggregation and launch vetoes use the canonical report implementation.
"""

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
from ditserve.evaluation import workflow_report as report

SCHEMA = "ditserve.workflow.encoding_summary.v1"
VARIANTS = tuple("T%d" % index for index in range(6))


def array_header(stream):
    """Inspect the bounded NPY header; this does not validate array values."""
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    version = stream.read(2)
    if version not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        raise ValueError("unsupported NPY version")
    size_bytes = 2 if version[0] == 1 else 4
    length = struct.unpack("<H" if size_bytes == 2 else "<I", stream.read(size_bytes))[0]
    if length > 1024 * 1024:
        raise ValueError("NPY header exceeds 1 MiB")
    header = ast.literal_eval(stream.read(length).decode("utf-8" if version[0] == 3 else "latin1"))
    return dict(shape=list(header["shape"]), storage_dtype=header["descr"],
                fortran_order=header["fortran_order"], values_recomputed=False)


def artifact_headers(value, root):
    path = report._output_path(value, root)
    if path is None:
        return dict(status="unavailable", reason="artifact path outside study root")
    path = path.resolve()
    if root not in path.parents:
        return dict(status="unavailable", reason="artifact resolves outside study root")
    try:
        if path.suffix == ".npz":
            with zipfile.ZipFile(path) as archive:
                arrays = {}
                for name in archive.namelist():
                    if name not in ("prompt.npy", "negative.npy"):
                        raise ValueError("unexpected embedding archive member")
                    with archive.open(name) as stream:
                        arrays[name[:-4]] = array_header(stream)
        else:
            with path.open("rb") as stream:
                arrays = array_header(stream)
        return dict(status="header_read", path=str(path), arrays=arrays, values_recomputed=False)
    except (OSError, ValueError, KeyError, SyntaxError, struct.error, zipfile.BadZipFile) as error:
        return dict(status="unavailable", path=str(path), reason=str(error))


def latest_launch(ledgers, output, variant=None, module="workflow_encoder"):
    candidates = [row for row in ledgers if row["module"] == "ditserve.evaluation." + module
                  and row["output"] == output and not row.get("profiler")
                  and (variant is None or variant in row["variants"])]
    return max(candidates, key=lambda row: row["data"].get("started_unix_ns", 0), default=None)


def launch_view(ledger):
    if ledger is None:
        return None
    value = ledger["data"]
    return dict(source={key: ledger[key] for key in ("path", "sha256", "bytes")},
                id=value.get("id"), status=value.get("status"), returncode=value.get("returncode"),
                error=value.get("error"), gpu_monitor=value.get("gpu_monitor"))


def outcome(source, latest):
    if source is None:
        status = latest["data"].get("status") if latest else None
        return dict(status="running" if status == "running" else "unmeasured",
                    raw_status=None, performance_eligible=False, validity=None)
    value, validity = source["data"], source.get("validity", {})
    raw_status = value.get("status", "unknown")
    status = "oom" if raw_status == "oom" or value.get("error", {}).get("type") == "OutOfMemoryError" else raw_status
    if source["role"].startswith("superseded"):
        status = source["role"]
    elif validity.get("eligible") is False and status not in ("oom", "running", "failed", "interrupted"):
        status = "invalidated"
    # Running source snapshots can survive an interrupted launcher.
    if status == "running" and validity.get("launch_status") not in (None, "running"):
        status = "interrupted" if validity["launch_status"] == "interrupted" else "incomplete"
    return dict(status=status, raw_status=raw_status, validity=validity or None,
                performance_eligible=raw_status == "completed" and source["role"] == "performance"
                and validity.get("eligible", True), error=value.get("error"))


def reserved_increments(records):
    """Mirror report's per-request peak grouping for the additional reserved delta."""
    groups = defaultdict(lambda: defaultdict(list))
    for source in records:
        for row in source["data"].get("phase_records", []):
            meta = row.get("metadata", {})
            if row.get("status") != "completed" or meta.get("purpose") not in report.PURPOSES:
                continue
            if report._number(meta.get("repeat")) and meta["repeat"] < 0:
                continue
            device = row.get("torch_after", {}).get("device", row.get("torch_before", {}).get("device"))
            key = (source["data"].get("variant"), meta["purpose"], report._frames(meta),
                   report.STAGES.get(row.get("name"), row.get("name")), device)
            request = (source["source"]["path"], meta.get("request_id"), meta.get("repeat"))
            if report._number(row.get("peak_reserved_increment_bytes")):
                groups[key][request].append(row["peak_reserved_increment_bytes"])
    return {key: report.distribution(max(values) for values in requests.values()) for key, requests in groups.items()}


def replay_summary(root, replay_root, source, ledgers):
    latest = latest_launch(ledgers, replay_root, module="workflow_embedding_replay")
    if source is None:
        return dict(outcome(None, latest), include_in_performance=False, launch=launch_view(latest), results=[])
    # Reuse the report's timestamp-aware launcher veto for numerical-only replay.
    # Adapt only the selection metadata; source timestamps/evidence remain intact.
    adapted = dict(source, role="performance", schema="ditserve.workflow.encoder.v1",
                   data=dict(source["data"], variant=replay_root.name))
    adapted_ledgers = [dict(row, module="ditserve.evaluation.workflow_encoder", variants=[replay_root.name],
                           output=replay_root.parent) for row in ledgers
                       if row["module"] == "ditserve.evaluation.workflow_embedding_replay" and row["output"] == replay_root]
    report.apply_launch_validity([adapted], root, adapted_ledgers)
    adapted_records = [adapted]
    report.apply_job_supersessions(root, adapted_records, adapted_ledgers)
    info = outcome(adapted, latest)
    value = source["data"]
    usable = value.get("status") in ("completed", "completed_with_unavailable_variants") and adapted["role"] == "performance"
    bridges = {row["frames"]: row.get("baseline_bridge_exact") is True for row in value.get("results", [])
               if row.get("variant") == "T0" and row.get("status") == "compared"}
    rows = [dict(row, evidence_eligible=usable and bridges.get(row.get("frames"), False))
            for row in value.get("results", [])]
    return dict(info, performance_eligible=False, include_in_performance=False, source=source["source"],
                launch=launch_view(latest), results=rows, baseline_bridge_by_frame=bridges,
                unavailable=value.get("sources", {}).get("unavailable", []),
                precision_path=value.get("precision_path"), validate_only=value.get("validate_only"))


def summarize(root, encoder_root=None, replay_root=None):
    root = Path(root).resolve()
    default_encoder = root / ("encoder-v2" if (root / "encoder-v2").is_dir() else "encoder")
    encoder_root = Path(encoder_root or default_encoder).resolve()
    replay_root = Path(replay_root or root / "embedding-replay").resolve()
    paths = []
    for variant in VARIANTS:
        directory = encoder_root / variant
        path = next((directory / name for name in ("result.json", "progress.json") if (directory / name).is_file()), None)
        if path:
            paths.append(path)
    if (replay_root / "run.json").is_file():
        paths.append(replay_root / "run.json")
    records, warnings, _ = report.discover(root, encoder_root, include=paths) if paths else ([], [], encoder_root)
    ledgers = report.load_launch_ledgers(root)
    report.apply_job_supersessions(root, records, ledgers)
    report.apply_launch_validity(records, root, ledgers)
    variants, eligible = {}, []
    for variant in VARIANTS:
        source = next((row for row in records if row["schema"] in report.ENCODER_SCHEMAS
                       and row["data"].get("variant") == variant), None)
        latest = latest_launch(ledgers, encoder_root, variant)
        info = outcome(source, latest)
        value = source["data"] if source else {}
        if info["performance_eligible"]:
            eligible.append(source)
        numeric = [dict(row, artifact_headers=artifact_headers(row.get("artifact"), root),
                        evidence_eligible=info["performance_eligible"], values_recomputed=False)
                   for row in value.get("numeric", [])]
        frames = {}
        for count in sorted(set((17, 33, *value.get("frames", [])))):
            requests = [row for row in value.get("requests", []) if row.get("kind") == "video" and row.get("frames") == count
                        and report._number(row.get("repeat")) and row["repeat"] >= 0 and row.get("purpose") != "warmup"]
            first = next((row for row in requests if row.get("repeat") == 0), {})
            frames[str(count)] = dict(status="observed" if requests else "not_applicable" if variant in ("T1", "T2") else "missing",
                performance_eligible=info["performance_eligible"] and bool(requests), requests=len(requests),
                inference_wall_ms=report.distribution(row.get("inference_wall_ms") for row in requests) if info["performance_eligible"] else None,
                finite_all_recorded=all(row.get("finite") is True for row in requests) if requests else None,
                shape=first.get("shape"), sha256=first.get("sha256"), comparison=first.get("comparison"),
                artifact_headers={name: artifact_headers(path, root) for name, path in first.get("artifacts", {}).items()})
        variants[variant] = dict(info, source=source["source"] if source else None, launch=launch_view(latest),
            configuration=value.get("configuration"), cpu_affinity=value.get("cpu_affinity"),
            matmul_tf32=value.get("matmul_tf32"), cudnn_tf32=value.get("cudnn_tf32"),
            numeric=numeric, frames=frames, sampling_errors=value.get("sampling_errors", []))
    stages, increments = report.encoder_stage_rows(eligible), reserved_increments(eligible)
    for row in stages:
        key = tuple(row[name] for name in ("variant", "purpose", "frames", "stage", "allocator_device"))
        row["peak_reserved_increment_bytes"] = increments.get(key)
    replay_source = next((row for row in records if row["schema"] in report.REPLAY_SCHEMAS), None)
    replay = replay_summary(root, replay_root, replay_source, ledgers)
    queue_context = dict(basis="operator notes and saved resume plan; no live process check",
        caveat="a historical failed launch is not a current blocker; absent results remain unmeasured, including experiments awaiting retry")
    for name in ("operator-notes.json", "jobs-resume.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text())
            if name == "operator-notes.json":
                queue_context["operator_notes"] = dict(source=report._binding(path), **{key: value.get(key) for key in
                    ("updated_utc", "active_job", "current_batch", "paused_for_reboot")})
            else:
                planned = [row.get("id") for row in value.get("jobs", []) if
                           "ditserve.evaluation.workflow_encoder" in row.get("command", [])]
                queue_context["resume_plan"] = dict(source=report._binding(path), encoding_job_ids=planned)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            warnings.append(dict(path=str(path), error=str(error)))
    return dict(schema=SCHEMA, generated_utc=datetime.now(timezone.utc).isoformat(),
        root=str(root), encoder_root=str(encoder_root), replay_root=str(replay_root),
        analysis_process=dict(cpu_affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0)),
        variants=variants, stage_resources=stages, dedicated_queue=report.encoder_queue_rows(report._encoder_requests(eligible)),
        replay=replay, current_queue_context=queue_context, warnings=warnings,
        availability={variant: dict(status=info["status"], performance=info["performance_eligible"],
            embedding_cases=[row.get("case") for row in info["numeric"] if row["evidence_eligible"]],
            video_frames=[int(count) for count, row in info["frames"].items() if row["performance_eligible"]],
            replay_frames=[row["frames"] for row in replay["results"] if row.get("variant") == variant and row["evidence_eligible"]])
            for variant, info in variants.items()},
        semantics=dict(cpu="process CPU milliseconds / summed stage wall milliseconds; average occupied CPU threads, not whole-machine percent",
            allocator="allocated/reserved peaks are process allocator bytes; increment is the maximum individual phase peak minus its phase start, not the whole-stage start",
            nvml_capacity="device-wide sampled used bytes / physical memory capacity; includes other device allocations; sampling may miss peaks",
            bandwidth="NVML memory_io_busy_percent is memory activity time, not DRAM bytes/s or achieved bandwidth; no bandwidth or SM occupancy inferred here",
            numeric="recorded finite/shape/max_abs/RMSE/cosine and FP16-input comparisons; headers only are reread, array values/hashes are not recomputed; no quality-equivalence claim",
            missing="missing/running/OOM/interrupted evidence is never an automatic numerical or performance pass",
            replay="numerical-only, eligible only after matching-launch veto and exact T0 latent+video bridge for that frame count",
            sources="source JSON SHA256 identifies the snapshot read; a running source may change on the next refresh"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--encoder-root", type=Path)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    value = summarize(args.root, args.encoder_root, args.replay_root)
    output = (args.out or args.root / "reviews/parallel-analysis/encoding-summary.json").resolve()
    study = args.root.resolve()
    if study in output.parents and study / "reviews" not in output.parents:
        parser.error("derived outputs inside --root must be placed under reviews/")
    for raw_directory in (Path(value["encoder_root"]), Path(value["replay_root"]), args.root.resolve() / "launches"):
        if output == raw_directory or raw_directory in output.parents:
            parser.error("--out must be a derived path outside encoder, replay and launch evidence directories")
    if output.exists():
        try:
            existing = json.loads(output.read_text())
        except (OSError, ValueError):
            parser.error("refusing to replace an unreadable or non-JSON output")
        if not isinstance(existing, dict) or existing.get("schema") != SCHEMA:
            parser.error("refusing to replace evidence with a different schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=output.parent, prefix=".encoding-summary-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps(dict(output=str(output), availability=value["availability"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
