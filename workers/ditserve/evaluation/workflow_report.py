"""Build a retained-evidence workflow report and the canonical two-file archive.

    python -m ditserve.evaluation.workflow_report --root /data/workflow/run01 \
        --encoder-root /data/workflow/run01/encoder-v2

Explicit --include files/directories replace discovery. Finalization retains key
evidence and prunes verified redundant data unless --keep-raw is requested.
Profiler timings are not mixed with normal performance runs.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
from urllib.parse import quote

from .workflow_retention import POLICY, archived_run, finalize_workflow


ENCODER_SCHEMAS = {"ditserve.workflow.encoder.v1", "workflow.encoder.v1"}
OVERLAP_SCHEMAS = {"ditserve.workflow-overlap.v1", "workflow-overlap.v1"}
TRACE_SCHEMAS = {"ditserve.workflow_trace.v1", "ditserve.workflow.trace.v1", "workflow.trace.v1"}
NCU_SCHEMAS = {"ditserve.workflow.ncu.v1"}
REPLAY_SCHEMAS = {"ditserve.workflow.embedding_replay.v1"}
EXCLUDED_PARTS = {"sources", "source-snapshots", "probes", "preflight", "excluded", "excluded-v1", ".git"}
PURPOSES = {"video_measurement", "encoder_measurement"}
STAGES = {"tokenize": "text_encoding", "text_token_cpu": "text_encoding", "text_token_h2d": "text_encoding",
          "text_encoder": "text_encoding", "text_embedding_pad": "text_encoding",
          "wan_prepare_input_h2d_latent": "prepare", "dit_step": "dit_denoising",
          "vae_prepare": "vae_decode", "vae_decode": "vae_decode", "video_postprocess_d2h": "output_d2h_postprocess"}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def preserve_dependencies(root, run):
    """A refreshed report cannot silently close registered downstream uses."""
    path = Path(root) / "run.json"
    if not path.exists():
        return run
    previous = json.loads(path.read_text())
    if previous.get("schema") != "ditserve.run.v1":
        raise ValueError("refusing to overwrite noncanonical run.json")
    run["retention"]["pending_dependencies"] = previous.get("retention", {}).get("pending_dependencies", [])
    artifacts = {row["path"]: row for row in run["artifacts"]}
    for row in previous.get("artifacts", []):
        if row.get("dependencies"):
            if row["path"] in artifacts:
                artifacts[row["path"]]["dependencies"] = row["dependencies"]
            else:
                run["artifacts"].append(row)
    return run


def distribution(values):
    values = sorted(value for value in values if _number(value))
    if not values:
        return None
    position = (len(values) - 1) * .95
    lower, upper = int(position), min(len(values) - 1, int(position) + 1)
    return dict(n=len(values), min=values[0], median=statistics.median(values), mean=statistics.mean(values),
                max=values[-1], p95=values[lower] + (values[upper] - values[lower]) * (position - lower),
                stdev=statistics.stdev(values) if len(values) > 1 else None)


def _binding(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return dict(path=str(path), bytes=size, sha256=digest.hexdigest())


def _frames(metadata):
    if isinstance(metadata.get("frames"), int):
        return metadata["frames"]
    match = re.search(r"(?:^|-)f(\d+)(?:-|$)", str(metadata.get("request_id", "")))
    return int(match.group(1)) if match else None


def _role(path, root, schema, encoder_root):
    parts = set(path.relative_to(root).parts)
    if parts & EXCLUDED_PARTS or any(part.startswith("excluded-") for part in parts):
        return "excluded_context"
    if schema in TRACE_SCHEMAS:
        return "trace_counters"
    if schema in NCU_SCHEMAS:
        return "kernel_diagnostics"
    if schema in REPLAY_SCHEMAS:
        return "numerical_replay"
    if parts & {"traces", "profiles", "profiling", "ncu", "nsys"}:
        return "profiled_workload"
    if schema in ENCODER_SCHEMAS and encoder_root and encoder_root not in path.parents:
        return "superseded_encoder"
    if path.name == "progress.json" and (path.parent / "result.json").exists():
        return "progress_snapshot"
    return "performance"


def discover(root, encoder_root=None, include=None):
    """Select schemas, retain excluded context separately, reject outside inputs."""
    root = Path(root).resolve()
    if encoder_root:
        encoder_root = Path(encoder_root).resolve()
    else:
        encoder_root = next((root / name for name in ("encoder-v2", "encoder") if (root / name).is_dir()), None)
    selected = [Path(value) if Path(value).is_absolute() else root / value for value in include or [root]]
    paths = set()
    for selected_path in selected:
        selected_path = selected_path.resolve()
        if selected_path != root and root not in selected_path.parents:
            raise ValueError("selected evidence must be inside --root: " + str(selected_path))
        if selected_path.is_file():
            paths.add(selected_path)
        elif selected_path.is_dir():
            for current, directories, names in os.walk(selected_path):
                # Frozen source JSONs are not experiment outputs and need not be parsed.
                directories[:] = [name for name in directories if name not in {"sources", "source-snapshots", ".git"}]
                paths.update(Path(current) / name for name in names if name.endswith(".json"))
        else:
            raise FileNotFoundError(selected_path)
    records, warnings, hashes = [], [], {}
    for path in sorted(paths):
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, ValueError) as error:
            warnings.append(dict(path=str(path), error=str(error)))
            continue
        if not isinstance(value, dict):
            continue
        schema = value.get("schema") or value.get("schema_version")
        if schema not in ENCODER_SCHEMAS | OVERLAP_SCHEMAS | TRACE_SCHEMAS | NCU_SCHEMAS | REPLAY_SCHEMAS:
            continue
        binding = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        role = _role(path, root, schema, encoder_root)
        if value.get("status") == "superseded_before_execution" and value.get("performance_evidence") is False:
            role = "superseded_before_execution"
        if schema in ENCODER_SCHEMAS and not value.get("variant") and role == "performance":
            role = "aggregate_manifest"
        if role in {"performance", "trace_counters", "kernel_diagnostics", "numerical_replay"}:
            if binding["sha256"] in hashes:
                role = "duplicate_content"
            else:
                hashes[binding["sha256"]] = str(path)
        records.append(dict(source=binding, role=role, schema=schema, data=value))
    return records, warnings, encoder_root


def _command_option(command, name):
    result = None
    for index, item in enumerate(command):
        if item == name and index + 1 < len(command):
            result = command[index + 1]
        elif item.startswith(name + "="):
            result = item[len(name) + 1:]
    return result


def _output_path(value, root):
    if not isinstance(value, str):
        return None
    path = Path(value)
    if ".." in path.parts:
        return None
    if path == root or root in path.parents:
        return path
    container_root = Path("/data/workflow") / root.name
    if path == container_root or container_root in path.parents:
        return root / path.relative_to(container_root)
    if not path.is_absolute():
        return root / path
    return None


def load_launch_ledgers(root):
    ledgers = []
    directory = root / "launches"
    if not directory.exists():
        return ledgers
    for path in sorted(directory.rglob("*.json")):
        if path.name != "launch.json" and not re.fullmatch(r"attempt-\d+\.json", path.name):
            continue
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, ValueError):
            continue
        command = value.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            continue
        module_index = next((index + 1 for index, item in enumerate(command[:-1])
                             if item == "-m" and command[index + 1].startswith("ditserve.evaluation.workflow_")), None)
        options = command[module_index + 1:] if module_index is not None else []
        output = _output_path(_command_option(options, "--out") or _command_option(options, "--output"), root)
        profiler_index = next((index for index, item in enumerate(command)
                               if Path(item).name in {"nsys", "ncu"}), None)
        profiler, capture = None, None
        if profiler_index is not None:
            profiler = Path(command[profiler_index]).name
            profiler_options = command[profiler_index + 1:module_index - 1] if module_index is not None else command[profiler_index + 1:]
            capture = _output_path(_command_option(profiler_options, "--output" if profiler == "nsys" else "--export")
                                   or _command_option(profiler_options, "-o"), root)
        if output is None and capture is None:
            continue
        variants = (_command_option(options, "--child") or _command_option(options, "--phases") or "").split(",")
        ledgers.append(dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                            data=value, module=command[module_index] if module_index is not None else None,
                            output=output, profiler=profiler, capture=capture, variants=[variant for variant in variants if variant]))
    return ledgers


def apply_job_supersessions(root, records, ledgers):
    """Honor explicit interrupted-job replacements without rewriting raw results."""
    path = root / "job-supersessions.json"
    if not path.is_file():
        return {}
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("schema") != "ditserve.workflow.job_supersessions.v1" or not isinstance(value.get("jobs"), list):
        raise ValueError("invalid job-supersessions.json schema")
    mapping = {}
    for entry in value["jobs"]:
        if not all(isinstance(entry.get(key), str) and entry[key] for key in ("id", "replacement", "reason")) or entry["id"] == entry["replacement"]:
            raise ValueError("job supersession requires distinct IDs and a reason")
        if entry["id"] in mapping and mapping[entry["id"]] != entry:
            raise ValueError("conflicting supersession for " + entry["id"])
        mapping[entry["id"]] = entry
    for source in records:
        if source["role"] != "performance":
            continue
        source_path = Path(source["source"]["path"])
        for ledger in ledgers:
            identifier = ledger["data"].get("id") or Path(ledger["path"]).parent.name
            if identifier not in mapping or ledger.get("output") is None:
                continue
            output = ledger["output"]
            if source["schema"] in ENCODER_SCHEMAS:
                variant = source["data"].get("variant")
                if variant not in ledger.get("variants", []):
                    continue
                output = output / variant
            if source_path.parent == output or output in source_path.parents:
                source["role"] = "superseded_interrupted"
                source["supersession"] = dict(mapping[identifier], manifest=str(path))
                break
    records.append(dict(source=dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                        role="job_supersession_manifest", schema=value["schema"], data=value))
    return mapping


def _source_unix_window(source):
    value = source["data"]
    if _number(value.get("started_unix_ns")):
        return value["started_unix_ns"], value.get("finished_unix_ns"), "explicit Unix timestamps"
    # Model runs record host monotonic timestamps; PhaseRecorder supplies a
    # simultaneous Unix timestamp that allows matching the launch interval.
    stack = [value]
    while stack:
        row = stack.pop()
        if isinstance(row, dict):
            if _number(row.get("start_unix_ns")) and _number(row.get("start_ns")) and _number(value.get("started_ns")):
                offset = row["start_unix_ns"] - row["start_ns"]
                return value["started_ns"] + offset, value["finished_ns"] + offset if _number(value.get("finished_ns")) else None, "phase clock mapping"
            stack.extend(child for child in row.values() if isinstance(child, (dict, list)))
        elif isinstance(row, list):
            stack.extend(child for child in row if isinstance(child, (dict, list)))
    return Path(source["source"]["path"]).stat().st_mtime_ns, None, "result file modification time fallback"


def apply_launch_validity(records, root, ledgers):
    """Fail closed for a matching bad launch, without poisoning another variant/retry."""
    for source in records:
        if source["role"] != "performance":
            continue
        path, value = Path(source["source"]["path"]), source["data"]
        candidates = []
        for ledger in ledgers:
            output = ledger["output"]
            if output is None:
                continue
            if source["schema"] in ENCODER_SCHEMAS:
                variant = value.get("variant")
                if ledger["module"] != "ditserve.evaluation.workflow_encoder" or variant not in ledger["variants"]:
                    continue
                output = output / variant
            elif ledger["module"] != "ditserve.evaluation.workflow_overlap":
                continue
            if output != path.parent and output not in path.parents:
                continue
            candidates.append(ledger)
        if not candidates:
            source["validity"] = dict(status="unmonitored_no_matching_ledger", eligible=True)
            continue
        started, finished, clock = _source_unix_window(source)
        compatible = []
        for ledger in candidates:
            launch = ledger["data"]
            lower, upper = launch.get("started_unix_ns"), launch.get("finished_unix_ns")
            if not _number(lower):
                continue
            if lower <= started and (not _number(upper) or started <= upper):
                compatible.append(ledger)
        if not compatible:
            source["validity"] = dict(status="unmonitored_no_temporally_matching_ledger", eligible=True,
                                       clock_basis=clock, output_matching_ledgers=[row["path"] for row in candidates])
            continue
        ledger = max(compatible, key=lambda row: row["data"].get("started_unix_ns", 0))
        launch, reasons = ledger["data"], []
        monitor = launch.get("gpu_monitor")
        if launch.get("status") != "completed":
            reasons.append("launch status: " + str(launch.get("status")))
        if launch.get("returncode") not in (None, 0):
            reasons.append("nonzero launcher return code")
        if isinstance(monitor, dict):
            if monitor.get("conflicts"):
                reasons.append("foreign GPU work observed")
            if monitor.get("errors"):
                reasons.append("GPU monitor could not verify the interval")
            if monitor.get("admission") == "rejected":
                reasons.append("GPU admission rejected")
        source["validity"] = dict(status="invalidated" if reasons else "monitored_valid" if monitor is not None else "legacy_completed_unmonitored",
                                   eligible=not reasons, reasons=reasons, ledger=ledger["path"], ledger_sha256=ledger["sha256"],
                                   launch_status=launch.get("status"), clock_basis=clock, source_started_unix_ns=started,
                                   source_finished_unix_ns=finished)
        if reasons:
            source["role"] = "invalidated_performance"


def _capture_stem(path):
    return path.with_suffix("") if path.suffix in {".sqlite", ".nsys-rep", ".ncu-rep"} else path


def apply_profiler_launch_validity(records, root, ledgers):
    """Counter provenance follows the capture output, not the later import time."""
    for source in records:
        if source["role"] not in {"trace_counters", "kernel_diagnostics"}:
            continue
        raw = source["data"].get("source")
        path = _output_path(raw if isinstance(raw, str) else raw.get("report") if isinstance(raw, dict) else None, root)
        profiler = "nsys" if source["role"] == "trace_counters" else "ncu"
        candidates = [row for row in ledgers if path is not None and row.get("profiler") == profiler and row.get("capture") is not None
                      and _capture_stem(row["capture"]) == _capture_stem(path)]
        if not candidates:
            source["validity"] = dict(status="unmonitored_no_matching_ledger", eligible=True)
            continue
        # A SQLite or JSON import may happen much later. Prefer the original
        # profiler capture's mtime, which is written during the guarded launch.
        capture = Path(str(_capture_stem(path)) + (".nsys-rep" if profiler == "nsys" else ".ncu-rep"))
        if not capture.is_file() and path.suffix in {".nsys-rep", ".ncu-rep"}:
            capture = path
        timestamp = capture.stat().st_mtime_ns if capture.is_file() else None
        compatible = candidates
        if timestamp is not None:
            compatible = [row for row in candidates if _number(row["data"].get("started_unix_ns"))
                          and row["data"]["started_unix_ns"] <= timestamp
                          and (not _number(row["data"].get("finished_unix_ns")) or timestamp <= row["data"]["finished_unix_ns"])]
        if not compatible:
            source["validity"] = dict(status="unmonitored_no_temporally_matching_ledger", eligible=True,
                                       clock_basis="original profiler capture modification time", capture_unix_ns=timestamp,
                                       output_matching_ledgers=[row["path"] for row in candidates])
            continue
        ledger = max(compatible, key=lambda row: row["data"].get("started_unix_ns", 0))
        launch, reasons = ledger["data"], []
        monitor = launch.get("gpu_monitor")
        if launch.get("status") != "completed":
            reasons.append("launch status: " + str(launch.get("status")))
        if launch.get("returncode") not in (None, 0):
            reasons.append("nonzero launcher return code")
        if isinstance(monitor, dict):
            if monitor.get("conflicts"):
                reasons.append("foreign GPU work observed")
            if monitor.get("errors"):
                reasons.append("GPU monitor could not verify the interval")
            if monitor.get("admission") == "rejected":
                reasons.append("GPU admission rejected")
        source["validity"] = dict(status="invalidated" if reasons else "monitored_valid" if monitor is not None else "legacy_completed_unmonitored",
                                   eligible=not reasons, reasons=reasons, ledger=ledger["path"], ledger_sha256=ledger["sha256"],
                                   launch_status=launch.get("status"), capture_path=str(capture), capture_unix_ns=timestamp,
                                   clock_basis="original profiler capture modification time" if timestamp is not None else "exact capture output prefix; latest matching launch")
        if reasons:
            source["role"] = "invalidated_" + source["role"]


def _nvml_device_rows(value, device):
    """Resolve recorded CUDA ordinals through the run's visible-device mapping."""
    match = re.fullmatch(r"(?:cuda:)?(\d+)", str(device))
    if not match:
        return []
    ordinal = int(match.group(1))
    affinity = value.get("cpu_affinity") or {}
    identifier = str(ordinal)
    visible = affinity.get("cuda_visible_devices")
    if visible:
        visible = [item.strip() for item in str(visible).split(",")]
        if ordinal >= len(visible):
            return []
        identifier = visible[ordinal]
    if affinity.get("device") == str(device) and affinity.get("gpu_uuid"):
        identifier = affinity["gpu_uuid"]
    rows = [row for row in value.get("samples", []) if _number(row.get("timestamp_ns"))
            and _number(row.get("memory_used_bytes"))
            and (row.get("uuid") == identifier if identifier.startswith("GPU-") else str(row.get("device")) == identifier)]
    return sorted(rows, key=lambda row: row["timestamp_ns"])


def _window_samples(rows, timestamps, phases):
    indices, observed_windows, periods = set(), 0, []
    for phase in phases:
        start, end = phase.get("start_ns"), phase.get("end_ns")
        if not _number(start) or not _number(end) or end < start:
            continue
        left, right = bisect_left(timestamps, start), bisect_right(timestamps, end)
        observed_windows += right > left
        indices.update(range(left, right))
        periods.extend((timestamps[index + 1] - timestamps[index]) / 1e6 for index in range(left, right - 1)
                       if timestamps[index + 1] > timestamps[index])
    return [rows[index] for index in sorted(indices)], observed_windows, periods


def encoder_stage_rows(records):
    groups, request_denominators, sample_cache = defaultdict(list), {}, {}
    for source in records:
        value = source["data"]
        if source["schema"] not in ENCODER_SCHEMAS or source["role"] != "performance":
            continue
        for request in value.get("requests", []):
            repeat = request.get("repeat")
            if request.get("warmup") or request.get("purpose") == "warmup" or (_number(repeat) and repeat < 0):
                continue
            request_id = request.get("request_id")
            field = "inference_wall_ms" if request.get("kind") == "video" else "encode_ms" if request.get("kind") == "encoder" else None
            if not request_id and request.get("kind") == "encoder":
                request_id = "encoder-measured-%s" % repeat
            elif not request_id and request.get("kind") == "video" and request.get("frames") is not None:
                request_id = "%s-f%s-r%s" % (value.get("variant"), request["frames"], repeat)
            if request_id and field and _number(request.get(field)) and request[field] > 0:
                key = (source["source"]["path"], request_id, repeat)
                request_denominators[key] = (field, request[field])
        for row in value.get("phase_records", []):
            meta = row.get("metadata", {})
            if meta.get("purpose") not in PURPOSES or row.get("status") != "completed":
                continue
            repeat = meta.get("repeat")
            if _number(repeat) and repeat < 0:
                continue
            variant = value.get("variant") or meta.get("variant")
            stage = STAGES.get(row.get("name"), row.get("name"))
            device = row.get("torch_after", {}).get("device", row.get("torch_before", {}).get("device"))
            sample_key = (source["source"]["path"], device)
            if sample_key not in sample_cache:
                samples = _nvml_device_rows(value, device)
                sample_cache[sample_key] = (samples, [sample["timestamp_ns"] for sample in samples])
            key = (variant, meta["purpose"], _frames(meta), stage, device)
            groups[key].append((row, source["source"]["path"]))
    output = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        per_request = defaultdict(list)
        for row, source in group:
            meta = row["metadata"]
            per_request[(source, meta.get("request_id"), meta.get("repeat"))].append(row)
        wall, cpu, peak, reserved, increments, capacity, rss = [], [], [], [], [], [], []
        shares, denominators, denominator_fields = [], [], set()
        allocated_starts, sampled_peaks, sampled_capacity, boundary_maxima, sample_counts = [], [], [], [], []
        sampled_windows, sample_periods = 0, []
        elapsed_total, cpu_total = 0., 0.
        for request_key, rows in per_request.items():
            durations = [row["wall_ms"] for row in rows if _number(row.get("wall_ms"))]
            duration = sum(durations) if len(durations) == len(rows) else None
            cpu_values = [row["process_cpu_ms"] for row in rows if _number(row.get("process_cpu_ms"))]
            if duration is not None:
                wall.append(duration)
                if request_key in request_denominators:
                    field, denominator = request_denominators[request_key]
                    shares.append(duration / denominator * 100)
                    denominators.append(denominator)
                    denominator_fields.add(field)
            if len(cpu_values) == len(rows):
                cpu.append(sum(cpu_values))
                if duration is not None:
                    cpu_total += sum(cpu_values)
                    elapsed_total += duration
            memory = [row.get("torch_after", {}) for row in rows]
            first = min(enumerate(rows), key=lambda pair: (pair[1].get("start_ns", pair[0]), pair[0]))[1]
            baseline = first.get("torch_before", {}).get("allocated_bytes")
            if _number(baseline):
                allocated_starts.append(baseline)
            boundaries = [row.get(snapshot, {}).get("device_used_bytes") for row in rows for snapshot in ("torch_before", "torch_after")]
            boundaries = [value for value in boundaries if _number(value)]
            if boundaries:
                boundary_maxima.append(max(boundaries))
            samples, timestamps = sample_cache[(request_key[0], key[4])]
            observed, windows, periods = _window_samples(samples, timestamps, rows)
            sampled_windows += windows
            sample_periods.extend(periods)
            sample_counts.append(len(observed))
            if observed:
                sampled_peaks.append(max(sample["memory_used_bytes"] for sample in observed))
                fractions = [sample["memory_used_bytes"] / sample["memory_total_bytes"] for sample in observed
                             if _number(sample.get("memory_total_bytes")) and sample["memory_total_bytes"] > 0]
                if fractions:
                    sampled_capacity.append(max(fractions))
            for field, target in (("peak_allocated_bytes", peak), ("peak_reserved_bytes", reserved)):
                values = [row[field] for row in memory if _number(row.get(field))]
                if values:
                    target.append(max(values))
            values = [row["peak_allocated_bytes"] / row["device_total_bytes"] for row in memory
                      if _number(row.get("peak_allocated_bytes")) and row.get("device_total_bytes")]
            if values:
                capacity.append(max(values))
            values = [row["peak_allocated_increment_bytes"] for row in rows if _number(row.get("peak_allocated_increment_bytes"))]
            if values:
                increments.append(max(values))
            values = [row.get("process_after", {}).get("rss_bytes") for row in rows]
            values = [value for value in values if _number(value)]
            if values:
                rss.append(max(values))
        output.append(dict(variant=key[0], purpose=key[1], frames=key[2], stage=key[3], allocator_device=key[4],
                           requests=len(per_request), phase_windows=len(group), wall_ms_per_request=distribution(wall),
                           request_wall_share_percent=distribution(shares), matched_denominator_requests=len(shares),
                           request_wall_denominator_ms=distribution(denominators), request_wall_denominator_fields=sorted(denominator_fields),
                           request_wall_share_basis="per-request sum of stage wall windows divided by that same request's recorded total wall time",
                           process_cpu_ms_per_request=distribution(cpu), process_average_cores=cpu_total / elapsed_total if cpu and elapsed_total else None,
                           peak_allocated_bytes=distribution(peak), peak_reserved_bytes=distribution(reserved),
                           stage_start_allocated_bytes=distribution(allocated_starts),
                           device_used_sampled_peak_bytes=distribution(sampled_peaks),
                           device_used_sampled_capacity_fraction=distribution(sampled_capacity),
                           device_used_boundary_max_bytes=distribution(boundary_maxima),
                           nvml_samples_per_request=distribution(sample_counts), nvml_observed_phase_windows=sampled_windows,
                           nvml_observed_sampling_period_ms=distribution(sample_periods),
                           device_memory_basis="NVML device-wide samples with timestamps inside same-device phase windows; boundary snapshots are separate and cannot establish a full phase peak",
                           peak_allocated_increment_bytes=distribution(increments), peak_capacity_fraction=distribution(capacity),
                           process_rss_bytes_observed_peak=distribution(rss),
                           peak_scope=sorted({row.get("peak_scope", "unknown") for row, _ in group}),
                           sources=sorted({source for _, source in group})))
    return output


def _arm_envelope(arm):
    execution = arm.get("execution_span_ms")
    if _number(execution) and execution > 0:
        return execution, "host NVTX and all launch-correlated GPU operations execution envelope"
    host = arm.get("host_span_ms")
    return (host if _number(host) and host > 0 else None), "host NVTX envelope fallback; asynchronous GPU tails may lie outside"


def trace_overlap_rows(traces):
    """Preserve genuine zero intervals while masking unavailable CUDA evidence."""
    result = []
    for trace in traces:
        availability = trace.get("availability") or {}
        kernels, copies = availability.get("cuda_kernels") is True, availability.get("cuda_memcpy") is True
        for arm in trace.get("arms", []):
            denominator, basis = _arm_envelope(arm)
            for device, values in arm.get("devices", {}).items():
                compute = values.get("compute_union_ms") if kernels else None
                communication = {}
                for name in ("nccl", "h2d", "d2h", "d2d", "p2p", "other_copy"):
                    evidence = values.get("communication", {}).get(name, {})
                    available = kernels if name == "nccl" else copies
                    duration = evidence.get("communication_ms") if available else None
                    communication[name] = dict(available=available, union_ms=duration,
                        envelope_percent=duration / denominator * 100 if _number(duration) and denominator else None,
                        compute_overlap_ms=evidence.get("overlap_ms") if available and kernels else None,
                        communication_coverage=evidence.get("communication_coverage") if available and kernels else None,
                        bytes=evidence.get("bytes") if available else None)
                result.append(dict(source=trace["source"], trace_source=trace.get("trace_source"), arm=arm["name"], device=device,
                    status="available" if kernels and copies else "partial" if kernels or copies else "unavailable",
                    cuda_kernels_available=kernels, cuda_memcpy_available=copies, envelope_ms=denominator,
                    host_envelope_ms=arm.get("host_span_ms"), execution_envelope_ms=arm.get("execution_span_ms"),
                    denominator_basis=basis + "; not the normal-run full request timer",
                    compute_union_ms=compute, compute_envelope_percent=compute / denominator * 100 if _number(compute) and denominator else None,
                    cpu_encode_gpu_compute_overlap_ms=values.get("cpu_encode_gpu_compute_overlap_ms") if kernels else None,
                    wan_sana_kernel_overlap_ms=values.get("wan_sana_kernel_overlap_ms") if kernels else None,
                    communication=communication))
    return result


def trace_stage_rows(records):
    groups = defaultdict(list)
    for source in records:
        if source["role"] != "trace_counters":
            continue
        for row in source["data"].get("phases", []):
            meta = row.get("metadata", {})
            if not row.get("included", True) or meta.get("purpose") not in PURPOSES:
                continue
            for device, values in row.get("devices", {}).items():
                key = (meta.get("variant"), meta["purpose"], _frames(meta), STAGES.get(row["name"], row["name"]), str(device))
                groups[key].append((values.get("host_window_metrics", {}), source["source"]["path"]))
    output = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        metrics = {}
        for name in sorted({name for values, _ in rows for name in values}):
            values = [value[name] for value, _ in rows if name in value and _number(value[name].get("mean"))]
            covered = sum(value.get("covered_ns", 0) for value in values)
            window = sum(value.get("window_ns", 0) for value in values)
            metrics[name] = dict(mean=sum(value["mean"] * value.get("covered_ns", 0) for value in values) / covered if covered else None,
                                 covered_ns=covered, window_ns=window, coverage_fraction=covered / window if window else None,
                                 phase_windows=len(values), scope="device-wide counters over host phase windows")
        output.append(dict(variant=key[0], purpose=key[1], frames=key[2], stage=key[3], physical_device_id=key[4],
                           metrics=metrics, sources=sorted({source for _, source in rows})))
    return output


def _correctness(value):
    flags = []
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, (bool, dict, list)):
                result = _correctness(child)
                if result is not None:
                    flags.append(result)
    elif isinstance(value, list):
        for child in value:
            result = _correctness(child)
            if result is not None:
                flags.append(result)
    return all(flags) if flags else None


def overlap_rows(records):
    rows = []
    for source in records:
        if source["schema"] not in OVERLAP_SCHEMAS or source["role"] != "performance":
            continue
        value, path = source["data"], source["source"]["path"]
        experiment, config = value.get("experiment"), value.get("config", {})
        common = dict(experiment=experiment, frames=config.get("frames"), steps=config.get("steps"),
                      height=config.get("height"), width=config.get("width"), source=path, execution_status=value.get("status"))
        if experiment == "E6":
            variants = defaultdict(list)
            for rank in value.get("ranks", []):
                for variant in rank.get("variants", []):
                    variants[variant["mode"]].append((rank.get("rank"), variant))
            timings, checks = {}, {}
            for mode, ranks in variants.items():
                by_repeat = defaultdict(list)
                checks[mode] = all(item.get("qualification", {}).get("passed") is True for _, item in ranks) if len(ranks) == 2 else None
                for rank, item in ranks:
                    for run in item.get("runs", []):
                        if not run.get("warmup") and run.get("repeat", -1) >= 0:
                            by_repeat[run["repeat"]].append((rank, run))
                timings[mode] = [dict(repeat=repeat, wall_ms=max(row["wall_ms"] for _, row in runs),
                                     dit_ms=max(row["dit_ms"] for _, row in runs))
                                 for repeat, runs in sorted(by_repeat.items()) if len({rank for rank, _ in runs}) == 2]
            baseline = {row["repeat"]: row for row in timings.get("baseline", [])}
            for mode, samples in timings.items():
                speedup = [baseline[row["repeat"]]["wall_ms"] / row["wall_ms"] for row in samples
                           if row["repeat"] in baseline and row["wall_ms"] > 0]
                rows.append(dict(common, mode=mode, baseline="baseline", condition="SP2 two-rank maximum, CPU encoding excluded",
                                 wall_ms=distribution([row["wall_ms"] for row in samples]), dit_ms=distribution([row["dit_ms"] for row in samples]),
                                 paired_speedup=distribution(speedup), numeric_passed=checks[mode],
                                 qualification=[dict(rank=rank, **item.get("qualification", {})) for rank, item in variants[mode]],
                                 trace_overlap_proven=None))
            continue
        grouped = defaultdict(list)
        for pair in value.get("pairs", []):
            for mode, measured in pair.get("modes", {}).items():
                condition = (pair.get("stage"), pair.get("preencoded", measured.get("preencoded")), mode)
                grouped[condition].append((pair, measured))
        for (stage, preencoded, mode), measured in grouped.items():
            baseline = {"E1": "serial", "E2": "single", "E3": "sync", "E4": "sync", "E5": "interleaved"}.get(experiment)
            wall, ratios, checks, throughput, requests, wan_completion, image_completion, peak = [], [], [], [], [], [], [], []
            devices, baseline_devices, other_devices = set(), set(), set()
            other_ratios = []
            input_setup, including_setup, setup_ratios = [], [], []
            for pair, row in measured:
                devices.update(row.get("devices", []))
                devices.update(worker["device"] for worker in row.get("workers", []) if worker.get("device"))
                if experiment == "E2":
                    baseline_devices.update(pair.get("modes", {}).get("single", {}).get("devices", []))
                    other_devices.update(pair.get("modes", {}).get("single_other", {}).get("devices", []))
                    if pair.get("single_device"):
                        baseline_devices.add(pair["single_device"])
                    if pair.get("other_single_device"):
                        other_devices.add(pair["other_single_device"])
                if _number(row.get("wall_ms")):
                    wall.append(row["wall_ms"])
                    reference = pair.get("modes", {}).get(baseline, {}).get("wall_ms")
                    if _number(reference) and row["wall_ms"] > 0 and not (experiment == "E5" and mode in {"wan_only", "sana_only"}):
                        ratios.append(reference / row["wall_ms"])
                    if row.get("requests") and row["wall_ms"] > 0:
                        throughput.append(len(row["requests"]) * 1000 / row["wall_ms"])
                    other_reference = pair.get("modes", {}).get("single_other", {}).get("wall_ms")
                    if experiment == "E2" and mode == "dual" and _number(other_reference) and row["wall_ms"] > 0:
                        other_ratios.append(other_reference / row["wall_ms"])
                check = (pair.get("other_single_correctness") if experiment == "E2" and mode == "single_other" else
                         None if experiment == "E2" and mode == "single" else pair.get("correctness"))
                input_setup.append(row.get("input_setup_ms"))
                including_setup.append(row.get("including_input_setup_ms"))
                reference_with_setup = pair.get("modes", {}).get(baseline, {}).get("including_input_setup_ms")
                if _number(reference_with_setup) and _number(row.get("including_input_setup_ms")) and row["including_input_setup_ms"] > 0:
                    setup_ratios.append(reference_with_setup / row["including_input_setup_ms"])
                if isinstance(check, dict) and mode in check:
                    check = check[mode]
                checks.append(_correctness(check))
                requests.extend(request.get("queue_response_ms") for request in row.get("requests", []))
                wan_completion.append(row.get("wan_stream_complete_ms"))
                image_completion.extend(row.get("image_complete_ms", []))
                peak.append(row.get("peak_allocated_bytes"))
            known = [check for check in checks if check is not None]
            condition = dict(stage=stage, preencoded=preencoded, output_mode=config.get("output_mode"))
            if experiment == "E2":
                condition["per_worker_schedule"] = "prefetch"
                condition["cpu_encoding_scope"] = "embeddings precomputed" if preencoded else "CPU encoding prefetched per worker and included in pipeline wall time"
            rows.append(dict(common, mode=mode, baseline=baseline, condition=condition,
                             devices=sorted(devices), baseline_devices=sorted(baseline_devices),
                             other_single_baseline_devices=sorted(other_devices),
                             paired_speedup_vs_other_single=distribution(other_ratios),
                             numeric_reference_mode="single" if experiment == "E2" and mode != "single" else None,
                             numeric_role="reference" if experiment == "E2" and mode == "single" else "comparison",
                             input_setup_ms=distribution(input_setup), including_input_setup_ms=distribution(including_setup),
                             paired_speedup_including_input_setup=distribution(setup_ratios),
                             timing_scope="CPU encoding excluded; wall_ms excludes CPU input preparation/pinning; including_input_setup_ms adds those costs" if experiment == "E3" else None,
                             wall_ms=distribution(wall), paired_speedup=distribution(ratios), throughput_per_second=distribution(throughput),
                             request_queue_response_ms=distribution(requests), wan_complete_ms=distribution(wan_completion),
                             image_complete_ms=distribution(image_completion), peak_allocated_bytes=distribution(peak),
                             numeric_passed=all(known) if known and len(known) == len(checks) else None, trace_overlap_proven=None))
        if not grouped:
            rows.append(dict(common, mode=None, condition=None, wall_ms=None, paired_speedup=None, numeric_passed=None,
                             trace_overlap_proven=None, error=value.get("error")))
    return rows


def _encoder_requests(records):
    rows = []
    for source in records:
        if source["schema"] not in ENCODER_SCHEMAS or source["role"] != "performance":
            continue
        value = source["data"]
        for request in value.get("requests", []):
            if request.get("warmup") or request.get("purpose") == "warmup" or (_number(request.get("repeat")) and request["repeat"] < 0):
                continue
            # Request metadata includes hashes/paths and numeric comparisons, never arrays.
            rows.append(dict(request, variant=value.get("variant"), source=source["source"]["path"]))
    return rows


def e1_cpu_encoding_rows(records):
    """Observe individual encode spans without assuming fixed stage costs in overlap."""
    groups = defaultdict(list)
    pattern = re.compile(r"^E1/r(-?\d+)/(serial|prefetch)/request\.(\d+)/cpu_encode$")
    for source in records:
        if source["role"] != "performance" or source["schema"] not in OVERLAP_SCHEMAS or source["data"].get("experiment") != "E1":
            continue
        config = source["data"].get("config", {})
        for phase in source["data"].get("phases", []):
            match = pattern.fullmatch(str(phase.get("name", "")))
            if not match or int(match.group(1)) < 0 or phase.get("status") != "completed" or not _number(phase.get("wall_ms")):
                continue
            repeat, mode, index = int(match.group(1)), match.group(2), int(match.group(3))
            key = (source["source"]["path"], config.get("frames"), config.get("steps"), config.get("height"), config.get("width"),
                   mode, "first_request" if index == 0 else "remaining_requests", config.get("cpu_threads"), config.get("cpu_affinity"))
            groups[key].append((repeat, index, phase["wall_ms"]))
    return [dict(source=key[0], frames=key[1], steps=key[2], height=key[3], width=key[4], mode=key[5], request_group=key[6],
                 cpu_threads=key[7], cpu_affinity_policy=key[8], windows=len(rows), repeats=len({row[0] for row in rows}),
                 request_indices=sorted({row[1] for row in rows}), cpu_encode_wall_ms=distribution([row[2] for row in rows]),
                 scope="individual normal-run CPU encode host windows; may overlap GPU work, never summed into request wall shares",
                 interpretation="Observed encoding duration under each schedule; these observations alone do not establish CPU contention or its cause.")
            for key, rows in sorted(groups.items(), key=lambda item: str(item[0]))]


def encoder_queue_rows(requests):
    groups = defaultdict(list)
    for row in requests:
        if row.get("kind") == "dedicated_queue":
            groups[(row.get("variant"), row.get("frames"), row.get("source"))].append(row)
    output = []
    for (variant, frames, source), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        baseline = {row.get("repeat"): row for row in rows if row.get("mode") == "serial"}
        for mode in sorted({row.get("mode") for row in rows}):
            measured = [row for row in rows if row.get("mode") == mode]
            ratios, checks = [], []
            for row in measured:
                reference = baseline.get(row.get("repeat"))
                if reference is None or reference.get("requests") != row.get("requests"):
                    continue
                if _number(reference.get("wall_ms")) and _number(row.get("wall_ms")) and row["wall_ms"] > 0:
                    ratios.append(reference["wall_ms"] / row["wall_ms"])
                if mode == "serial":
                    continue
                hashes = [(reference.get(field), row.get(field)) for field in ("video_sha256", "latent_sha256")]
                complete = all(isinstance(left, list) and isinstance(right, list) and len(left) == len(right) == row.get("requests")
                               and bool(left) for left, right in hashes)
                checks.append(all(left == right for left, right in hashes) if complete else None)
            known = [value for value in checks if value is not None]
            output.append(dict(variant=variant, frames=frames, mode=mode, baseline="serial", source=source,
                repeats=len(measured), queue_size=sorted({row.get("requests") for row in measured if isinstance(row.get("requests"), int)}),
                wall_ms=distribution([row.get("wall_ms") for row in measured]),
                videos_per_second=distribution([row.get("videos_per_second") for row in measured]),
                paired_speedup=distribution(ratios), numeric_role="reference" if mode == "serial" else "comparison",
                numeric_passed=all(known) if known and len(known) == len(measured) else None,
                numeric_basis="paired same-repeat video and latent SHA256 lists; absent latent evidence remains unverified"))
    return output


def planned_coverage(root, records, ledgers, include=None, supersessions=None):
    """Check declared jobs and parsed captures, not only already discovered results."""
    scope = "selected_evidence" if include else "discovered_evidence_only"
    result = dict(scope=scope, enforced=False, all_complete=False, jobs=[], counts={}, warnings=[])
    if include:
        result["reason"] = "Explicit --include is a partial evidence selection, not whole-study completion."
        return result
    planned, plans = {}, []
    modules = {"workflow_encoder": "normal", "workflow_overlap": "normal", "workflow_embedding_replay": "replay"}
    for path in sorted(root.glob("jobs*.json")):
        if "probe" in path.name.lower() or ".bak" in path.name:
            continue
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
            jobs = value.get("jobs", [])
            if not isinstance(jobs, list):
                raise ValueError("jobs must be a list")
        except (OSError, ValueError, AttributeError) as error:
            result["warnings"].append(dict(path=str(path), error=str(error)))
            continue
        plans.append(dict(source=dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                          role="job_plan", schema="ditserve.workflow.plan.v1", data={}))
        for job in jobs:
            command, identifier = job.get("command"), job.get("id")
            if not isinstance(identifier, str) or not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                result["warnings"].append(dict(path=str(path), error="job lacks string ID/list argv"))
                continue
            index = next((i + 1 for i, item in enumerate(command[:-1]) if item == "-m" and command[i + 1].rsplit(".", 1)[-1] in modules), None)
            if index is None:  # Probe modules and unrelated maintenance commands are outside the study.
                continue
            if identifier in planned:
                planned[identifier]["plans"].append(str(path))
                if planned[identifier]["command"] != command:
                    planned[identifier]["plan_conflict"] = True
                continue
            options, module = command[index + 1:], command[index].rsplit(".", 1)[-1]
            profiler = next((i for i, item in enumerate(command[:index]) if Path(item).name in {"nsys", "ncu"}), None)
            category = ("trace" if Path(command[profiler]).name == "nsys" else "ncu") if profiler is not None else modules[module]
            output = _output_path(_command_option(options, "--out") or _command_option(options, "--output"), root)
            capture = None
            if profiler is not None:
                flags = command[profiler + 1:index - 1]
                capture = _output_path(_command_option(flags, "--output" if category == "trace" else "--export") or _command_option(flags, "-o"), root)
            identities = ((_command_option(options, "--child") or _command_option(options, "--phases") or "T0,T1,T2,T3,T4,T5").split(",")
                          if module == "workflow_encoder" else [_command_option(options, "--experiment")] if module == "workflow_overlap" else [])
            frames = [int(frame) for frame in (_command_option(options, "--frames") or "17,33").split(",") if frame.isdigit()]
            planned[identifier] = dict(id=identifier, category=category, module=module, identities=identities, frames=frames,
                plans=[str(path)], command=command, plan_conflict=False, output=output, capture=capture,
                variants=(_command_option(options, "--variants") or "T1,T2").split(","))
    records.extend(plans)
    if not planned:
        result["reason"] = "No recognized job plans; discovered evidence cannot establish whole-study completion."
        return result
    result.update(scope="planned_study", enforced=True)
    for job in planned.values():
        output, capture, category = job["output"], job["capture"], job["category"]
        selected = []
        for record in records:
            if category in {"trace", "ncu"}:
                raw = record["data"].get("source")
                path = _output_path(raw if isinstance(raw, str) else raw.get("report") if isinstance(raw, dict) else None, root)
                schema = TRACE_SCHEMAS if category == "trace" else NCU_SCHEMAS
                if record["schema"] in schema and path is not None and capture is not None and _capture_stem(path) == _capture_stem(capture):
                    selected.append(record)
            elif output is not None:
                path = Path(record["source"]["path"])
                allowed = {"performance", "invalidated_performance", "superseded_before_execution", "superseded_interrupted"} if category == "normal" else {"numerical_replay"}
                if record["role"] in allowed and (path.parent == output or output in path.parents):
                    identity = record["data"].get("variant", record["data"].get("experiment"))
                    if not job["identities"] or identity in job["identities"]:
                        selected.append(record)
        launch_rows = [row for row in ledgers if row["data"].get("id") == job["id"] or Path(row["path"]).parent.name == job["id"]]
        launch_row = max(launch_rows, key=lambda row: (Path(row["path"]).name == "launch.json", row["data"].get("started_unix_ns", 0))) if launch_rows else None
        launch = launch_row["data"] if launch_row else {}
        execution = launch.get("status", "unmonitored_result" if selected else "not_started")
        monitor = launch.get("gpu_monitor") or {}
        invalid_guard = bool(monitor.get("errors") or monitor.get("conflicts") or monitor.get("admission") == "rejected")
        statuses = [row["data"].get("status") for row in selected]
        markers = [row for row in selected if row["role"] == "superseded_before_execution"]
        row = dict(id=job["id"], category=category, identities=job["identities"], frames=job["frames"], plans=job["plans"],
                   status="pending", execution_status=execution, evidence_status="missing", sources=[item["source"]["path"] for item in selected],
                   recorded_statuses=statuses, plan_conflict=job["plan_conflict"], capture_path=None, capture_exists=None, parse_availability=None)
        if category in {"trace", "ncu"}:
            capture_path = Path(str(_capture_stem(capture)) + (".nsys-rep" if category == "trace" else ".ncu-rep")) if capture else None
            row.update(capture_path=str(capture_path) if capture_path else None, capture_exists=capture_path.is_file() if capture_path else False)
            if selected:
                row["parse_availability"] = (selected[-1]["data"].get("availability") or {}) if category == "trace" else dict(kernels=bool(selected[-1]["data"].get("summaries")))
        if job["id"] in (supersessions or {}):
            declaration = supersessions[job["id"]]
            row.update(status="superseded", evidence_status="superseded_interrupted", planned_replacement=declaration["replacement"], reason=declaration["reason"])
        elif markers:
            row.update(status="superseded", evidence_status="historical_marker", planned_replacement=markers[0]["data"].get("planned_replacement"))
        elif job["plan_conflict"] or invalid_guard:
            row.update(status="failed", evidence_status="conflicting_plan" if job["plan_conflict"] else "guard_invalidated")
        elif "oom" in statuses and execution in {"completed", "failed", "unmonitored_result"}:
            row.update(status="attempted_negative", evidence_status="gpu_oom", interpretation="Valid negative capacity outcome; not passed performance.")
        elif execution not in {"completed", "running", "not_started", "unmonitored_result"}:
            row.update(status="failed", evidence_status="launch_failed")
        elif any(item["role"].startswith("invalidated_") for item in selected) and execution != "running":
            row.update(status="failed", evidence_status="invalidated")
        elif category in {"trace", "ncu"}:
            valid = [item for item in selected if item["role"] == ("trace_counters" if category == "trace" else "kernel_diagnostics")]
            if valid:
                data = valid[-1]["data"]
                availability = (data.get("availability") or {}) if category == "trace" else dict(kernels=bool(data.get("summaries")))
                row["parse_availability"] = availability
                available = (all(availability.get(key) is True for key in ("cuda_kernels", "cuda_memcpy", "gpu_metrics", "workflow_nvtx"))
                             if category == "trace" else availability["kernels"])
                row["evidence_status"] = "parsed_available" if available and data.get("status") in {"completed", "complete"} else "parsed_partial"
                if row["evidence_status"] == "parsed_available" and row["capture_exists"] and execution in {"completed", "unmonitored_result"}:
                    row["status"] = "completed"
            elif row["capture_exists"]:
                row["evidence_status"] = "capture_unparsed"
        elif selected and all(value in {"completed", "complete"} for value in statuses) and execution in {"completed", "unmonitored_result"}:
            missing = []
            if category == "replay":
                compared = {(item.get("variant"), item.get("frames")) for record in selected for item in record["data"].get("results", []) if item.get("status") == "compared"}
                missing = ["%s/f%s" % (variant, frames) for variant in ["T0"] + job["variants"] for frames in job["frames"] if (variant, frames) not in compared]
            else:
                present = {item["data"].get("variant", item["data"].get("experiment")) for item in selected}
                missing = [identity for identity in job["identities"] if identity not in present]
                for item in selected:
                    if item["schema"] in OVERLAP_SCHEMAS and item["data"].get("config", {}).get("frames") not in job["frames"]:
                        missing.append("planned overlap frames " + str(job["frames"]))
                    if item["schema"] in ENCODER_SCHEMAS and item["data"].get("variant") in {"T0", "T3", "T4", "T5"}:
                        measured_frames = {request.get("frames") for request in item["data"].get("requests", []) if request.get("kind") == "video" and request.get("purpose") == "video_measurement"}
                        missing.extend("%s/f%s" % (item["data"]["variant"], frame) for frame in job["frames"] if frame not in measured_frames)
            row.update(status="pending" if missing else "completed", evidence_status="incomplete_expected_results" if missing else "complete", missing_results=missing)
        elif any(value in {"failed", "interrupted", "baseline_bridge_failed"} for value in statuses):
            row.update(status="failed", evidence_status="result_failed")
        elif selected:
            row["evidence_status"] = "incomplete_results"
        result["jobs"].append(row)
    for category in ("normal", "trace", "ncu", "replay"):
        rows = [row for row in result["jobs"] if row["category"] == category]
        counts = {state: sum(row["status"] == state for row in rows) for state in ("completed", "pending", "failed", "attempted_negative", "superseded")}
        result["counts"][category] = dict(expected=len(rows) - counts["superseded"], **counts,
            capture_available=sum(row["capture_exists"] is True for row in rows), parsed_available=sum(row["evidence_status"] == "parsed_available" for row in rows))
    result["unresolved_replacements"] = [row.get("planned_replacement") for row in result["jobs"] if row["status"] == "superseded" and row.get("planned_replacement") not in planned]
    result["all_complete"] = all(row["status"] in {"completed", "superseded"} for row in result["jobs"]) and not result["warnings"] and not result["unresolved_replacements"]
    return result


def ncu_diagnostic_rows(records, ledgers):
    """Keep sampled-kernel scope, NVTX metadata and capture selection visible."""
    ledger_by_path = {row["path"]: row for row in ledgers}
    output = []
    for source in records:
        if source["role"] != "kernel_diagnostics":
            continue
        data = source["data"]
        ledger = ledger_by_path.get((source.get("validity") or {}).get("ledger"), {})
        command = ledger.get("data", {}).get("command", [])
        limit = _command_option(command, "--launch-count")
        selection = dict(launch_count_limit=int(limit) if limit and limit.isdigit() else None,
                         range_filter=_command_option(command, "--range-filter"),
                         nvtx_include=_command_option(command, "--nvtx-include"),
                         replay_mode=_command_option(command, "--replay-mode"),
                         ledger=ledger.get("path"))
        summaries = []
        for summary in data.get("summaries", []):
            selected = [row for row in data.get("kernels", []) if row.get("included")
                        and row.get("stage") == summary.get("stage")
                        and row.get("metadata", {}).get("variant") == summary.get("variant")
                        and str(row.get("device")) == str(summary.get("device"))]
            frames = sorted({_frames(row.get("metadata", {})) for row in selected
                             if _frames(row.get("metadata", {})) is not None})
            steps = sorted({row.get("metadata", {}).get("step") for row in selected
                            if isinstance(row.get("metadata", {}).get("step"), int)})
            summaries.append(dict(summary, frames=frames or None, dit_steps=steps or None))
        output.append(dict(source=source["source"]["path"], capture_source=data.get("source"),
                           coverage=data.get("coverage"), status=data.get("status"),
                           summaries=summaries, selection=selection, warnings=data.get("warnings", [])))
    return output


def build_report(root, encoder_root=None, include=None, *, final=True):
    root = Path(root).resolve()
    if final:
        saved = archived_run(root)
        if saved is not None:
            return saved
    records, warnings, chosen_encoder = discover(root, encoder_root, include)
    ledgers = load_launch_ledgers(root)
    supersessions = apply_job_supersessions(root, records, ledgers)
    apply_launch_validity(records, root, ledgers)
    apply_profiler_launch_validity(records, root, ledgers)
    coverage = planned_coverage(root, records, ledgers, include, supersessions)
    records.extend(dict(source={key: ledger[key] for key in ("path", "bytes", "sha256")},
                        role="launch_ledger", schema="ditserve.workflow.launch.v1", data=ledger["data"]) for ledger in ledgers)
    all_performance = [row for row in records if row["role"] in {"performance", "invalidated_performance"}]
    stages, counters, overlaps = encoder_stage_rows(records), trace_stage_rows(records), overlap_rows(records)
    traces = [dict(source=row["source"]["path"], trace_source=row["data"].get("source"),
                   status=row["data"].get("status"), availability=row["data"].get("availability"),
                   warnings=row["data"].get("warnings", []), arms=row["data"].get("arms", []))
              for row in records if row["role"] == "trace_counters"]
    diagnostics = ncu_diagnostic_rows(records, ledgers)
    replay = [dict(source=row["source"]["path"], status=row["data"].get("status"),
                   purpose="numerical_replay", include_in_performance=False,
                   baseline_bridge_exact=row["data"].get("baseline_bridge_exact"),
                   results=row["data"].get("results", []), sources=row["data"].get("sources", {}),
                   error=row["data"].get("error")) for row in records if row["role"] == "numerical_replay"]
    requests = _encoder_requests(records)
    superseded = [dict(source=row["source"]["path"], history_role=row["role"], performance_evidence=False,
        reason=(row.get("supersession") or {}).get("reason",row["data"].get("reason")),
        planned_replacement=(row.get("supersession") or {}).get("replacement",row["data"].get("planned_replacement")),
        supersession=row.get("supersession"), **{key: row["data"].get(key) for key in ("experiment", "variant", "status", "config", "evidence")})
        for row in records if row["role"] in {"superseded_before_execution", "superseded_interrupted"}]
    phases, failures = [], []
    for identity in ["T%d" % i for i in range(6)] + ["E%d" % i for i in range(1, 7)]:
        evidence = [row for row in all_performance if row["data"].get("variant", row["data"].get("experiment")) == identity]
        statuses = [row["data"].get("status") for row in evidence]
        status = ("not_run" if not evidence else "invalidated" if any(row["role"] == "invalidated_performance" for row in evidence)
                  else "completed" if all(value in {"completed", "complete"} for value in statuses)
                  else "incomplete")
        expected = [row for row in coverage["jobs"] if row["category"] == "normal" and identity in row["identities"] and row["status"] != "superseded"]
        if expected and not all(row["status"] == "completed" for row in expected):
            status = ("attempted" if all(row["status"] == "attempted_negative" for row in expected)
                      else "incomplete" if evidence or any(row["execution_status"] != "not_started" for row in expected) else "not_run")
        numeric = [row["numeric_passed"] for row in overlaps if row["experiment"] == identity and row["numeric_passed"] is not None]
        measured_scopes = set()
        if identity.startswith("T"):
            for request in requests:
                if request.get("variant") != identity:
                    continue
                for kind, field, scope in (("encoder", "encode_ms", "encoder_only"),
                                           ("video", "inference_wall_ms", "full_video"),
                                           ("dedicated_queue", "wall_ms", "dedicated_encoder_queue")):
                    if request.get("kind") == kind and _number(request.get(field)) and request[field] > 0:
                        measured_scopes.add(scope)
        elif any(row["experiment"] == identity and (row.get("wall_ms") or {}).get("n", 0) > 0 for row in overlaps):
            measured_scopes.add("normal_experiment_workload")
        performance = dict(status="measured" if status == "completed" and measured_scopes else "unverified",
                           measured_scopes=sorted(measured_scopes))
        if identity == "E6" and numeric and not any(numeric) and not measured_scopes:
            performance["reason"] = "Numerical qualification failed; full-step performance runs were not executed."
        qualification = dict(scopes=dict(numeric=dict(status="passed" if numeric and all(numeric) else "failed" if numeric else "unverified"),
                                        semantic=dict(status="unverified"), performance=performance))
        metrics = dict(recorded_statuses=statuses, source_count=len(evidence), expected_normal_jobs=[row["id"] for row in expected],
                       completed_normal_jobs=[row["id"] for row in expected if row["status"] == "completed"])
        if identity.startswith("T"):
            metrics["numeric_comparisons"] = [dict(source=row["source"]["path"], values=row["data"].get("numeric", [])) for row in evidence]
        phases.append(dict(id=identity, status=status, attempt=1 if evidence else 0, qualification=qualification, metrics=metrics))
        for row in evidence:
            if row["data"].get("status") not in {"complete", "completed"} or row["role"] == "invalidated_performance":
                failures.append(dict(phase=identity, source=row["source"]["path"], recorded_status=row["data"].get("status"),
                                     error=row.get("validity") if row["role"] == "invalidated_performance" else row["data"].get("error", "incomplete evidence")))
    sources = [dict(row["source"], role=row["role"], schema=row["schema"], recorded_status=row["data"].get("status"),
                    validity=row.get("validity"), supersession=row.get("supersession")) for row in records]
    identity = root.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identity):
        raise ValueError("root directory name must be a safe run identity")
    run = dict(schema="ditserve.run.v1", experiment="workflow", run_id=identity,
                status="completed" if coverage["all_complete"] and phases and all(row["status"] == "completed" for row in phases) else "partial",
                created_utc=datetime.now(timezone.utc).isoformat(),
                config=dict(root=str(root), encoder_root=str(chosen_encoder) if chosen_encoder else None,
                            include=[str(value) for value in include] if include else None,
                            completion_scope=coverage["scope"],
                            completion_meaning="Completed means planned evidence collection and parsing are complete; numerical acceptance and deployment suitability are separate.",
                            evidence_policy="normal-run timers; profiler counters separately; warmup, qualification and superseded data excluded",
                            units=dict(time="ms", memory="bytes", hardware_counters="percentage points, 0-100", capacity="fraction, 0-1")),
                metrics=dict(stage_resources=stages, stage_hardware_counters=counters, overlap_experiments=overlaps,
                             e1_cpu_encoding=e1_cpu_encoding_rows(records),
                             trace_evidence=traces, trace_overlap_summary=trace_overlap_rows(traces), kernel_diagnostics=diagnostics, embedding_replay=replay,
                             dedicated_encoder_queues=encoder_queue_rows(requests),
                             superseded_executions=superseded,
                             planned_coverage=coverage,
                             sources=sources, discovery_warnings=warnings,
                             limitations=["Missing values are null, never assumed zero.",
                                          "CPU usage is occupied cores over a phase window, not a percentage of all 240 host CPUs.",
                                          "GPU counters are device-wide; SM Active, elapsed warp slots and NCU achieved occupancy are different metrics.",
                                          "Allocator peaks include resident models; stage increments are separate and cannot be summed across stages.",
                                          "Phase durations sum sequential windows per request; text branches and all DiT steps are included.",
                                          "Stage wall shares use each matched request's measured wall time, not CPU/GPU utilization or a ratio of aggregate medians.",
                                          "Trace interval percentages use the host-plus-GPU execution envelope when available, otherwise the labeled host-only fallback; overlapping compute/communication percentages need not sum to 100.",
                                          "NVML device memory samples can miss short peaks; observed sampling periods include query overhead. Boundary snapshots are not full-stage peaks.",
                                          "Profiler counter runs do not supply performance speedups. Original captures and key quality arrays are retained; verified redundant data may be pruned.",
                                          "NCU launch limits select a prefix of kernels in a filtered NVTX range; these are not representative whole-stage occupancy measurements.",
                                          "Observed paired speedups are descriptive; small-sample p95 is not a production tail-latency guarantee.",
                                          "Numerical comparisons do not establish perceptual quality or automatic deployment suitability."]),
                phases=phases, requests=requests, failures=failures, budget={},
                retention=dict(pending_dependencies=[], policy=POLICY),
                artifacts=[dict(path=str(Path(source["path"]).relative_to(root)), bytes=source["bytes"], sha256=source["sha256"],
                                kind="control", dependencies=[]) for source in sources])
    return preserve_dependencies(root, run) if final else run


def _display(value, digits=3):
    if value is None:
        return "未测 / 不适用"
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    if isinstance(value, float):
        return format(value, ".%df" % digits)
    return str(value)


def _median(value):
    return value.get("median") if isinstance(value, dict) else None


def render_html(run):
    root = Path(run["config"]["root"])
    def link(path, label=None):
        if not path:
            return "未提供"
        path = _output_path(str(path), root) or Path(path)
        if path.suffix == ".sqlite" and path.with_suffix(".nsys-rep").is_file():
            path = path.with_suffix(".nsys-rep")
            if label == "SQLite":
                label = "原始捕获"
        target = os.path.relpath(path, root)
        return '<a href="%s">%s</a>' % (html.escape(quote(target), quote=True), html.escape(label or target))
    def table(headers, rows):
        return '<div class="table"><table><thead><tr>' + ''.join('<th>'+html.escape(header)+'</th>' for header in headers) + \
               '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+cell+'</td>' for cell in row)+'</tr>' for row in rows) + '</tbody></table></div>'
    def cell(value):
        return html.escape(_display(value))
    metrics = run["metrics"]
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>双 4090 视频 DiT 工作流实验</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 22px;color:#17212b;background:#f5f7fa}h1,h2{line-height:1.3}section{background:white;padding:22px;margin:22px 0;border-radius:10px}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #d9e1ea;text-align:left;vertical-align:top}th{background:#edf2f7;white-space:nowrap}code{font-size:13px}a{color:#075ca8}.note{color:#526272}details{margin:14px 0}pre{white-space:pre-wrap;word-break:break-word}</style>',
             '<h1>双 4090 视频 DiT 工作流实验</h1><p>运行 '+cell(run["run_id"])+ ' · 汇总状态 '+cell(run["status"])+
             ' · '+cell(run["created_utc"])+ '</p><p class="note">正常运行提供耗时与吞吐；分析工具提供硬件计数器和真实时间线。缺测保留为空，关键质量数组、原始捕获和解析证据长期保留；冗余副本可经验证后清理。completed 表示计划证据收集与解析完成，不代表所有方案通过数值门槛或适合部署。</p>']
    coverage = metrics.get("planned_coverage", {})
    parts.append('<section><h2>计划覆盖与报告范围</h2><p>范围：'+cell(coverage.get("scope"))+ '。正常实验、时间线、NCU 和数值回放分别验收；任务执行结束不等于 capture 已解析或计数器完整。OOM 可作为容量不足的负面结果，但不算性能通过。</p>')
    if coverage.get("enforced"):
        parts.append(table(["类别", "计划任务", "完成", "待完成/证据待补", "失败", "容量负面结果", "执行前撤换", "capture 存在", "有效解析"],
            [[cell(category)]+[cell(values.get(name)) for name in ("expected","completed","pending","failed","attempted_negative","superseded","capture_available","parsed_available")]
             for category,values in coverage.get("counts",{}).items()]))
        parts.append('<details><summary>逐任务执行与证据状态</summary>'+table(["任务", "类别", "帧数", "覆盖状态", "执行状态", "证据状态", "来源"],
            [[cell(row["id"]),cell(row["category"]),cell(row["frames"]),cell(row["status"]),cell(row["execution_status"]),cell(row["evidence_status"]),
              '<br>'.join(link(path) for path in row["sources"]) or "未提供"] for row in coverage.get("jobs",[])])+'</details>')
        if coverage.get("warnings") or coverage.get("unresolved_replacements"):
            parts.append('<pre>'+html.escape(json.dumps(dict(warnings=coverage.get("warnings"), unresolved_replacements=coverage.get("unresolved_replacements")),ensure_ascii=False,indent=2))+'</pre>')
    else:
        parts.append('<p>'+cell(coverage.get("reason"))+ '</p>')
    parts.append('</section>')
    parts.append('<section><h2>执行与验收</h2><p>执行完成、数值门槛和实际性能测量分别列示。资格检查失败而未进入完整性能运行时，性能保持 unverified；encoder_only 仅表示单独文本编码计时。</p>'+table(["实验", "执行状态", "源记录数", "数值门槛", "性能测量", "实测范围", "性能说明", "语义质量"],
        [[cell(row["id"]), cell(row["status"]), cell(row["metrics"]["source_count"]),
          cell(row["qualification"]["scopes"]["numeric"]["status"]),
          cell(row["qualification"]["scopes"]["performance"]["status"]),
          cell(", ".join(row["qualification"]["scopes"]["performance"].get("measured_scopes", [])) or None),
          cell(row["qualification"]["scopes"]["performance"].get("reason")),
          cell(row["qualification"]["scopes"]["semantic"]["status"])] for row in run["phases"]])+'</section>')
    parts.append('<section><h2>文本编码、DiT 和 VAE：每请求资源</h2><p>文本编码合并正负提示词及分词/填充；DiT 合并全部步骤。阶段耗时占比先逐请求配对计算，再取中位数，不是 CPU 或 GPU 利用率。视频分母为 inference_wall_ms（编码至视频输出）；单独编码分母为 encode_ms。显存峰值包含常驻模型，增量表示相对于阶段开始的分配增长。</p>')
    stage_rows, memory_rows = [], []
    for row in metrics["stage_resources"]:
        capacity = _median(row["peak_capacity_fraction"])
        stage_rows.append([cell(value) for value in (row["variant"], row["purpose"], row["frames"], row["stage"], row["allocator_device"], row["requests"],
                           _median(row["wall_ms_per_request"]), _median(row.get("request_wall_share_percent")),
                           _median(row.get("request_wall_denominator_ms")), ", ".join(row.get("request_wall_denominator_fields", [])) or None,
                           row.get("matched_denominator_requests"), row["process_average_cores"])])
        memory_values = [_median(row.get(field)) for field in ("stage_start_allocated_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
                         "peak_allocated_increment_bytes", "device_used_sampled_peak_bytes", "device_used_boundary_max_bytes")]
        sampled_capacity = _median(row.get("device_used_sampled_capacity_fraction"))
        memory_rows.append([cell(value) for value in (row["variant"], row["purpose"], row["frames"], row["stage"], row["allocator_device"],
                           *(value / 2**30 if value is not None else None for value in memory_values),
                           capacity * 100 if capacity is not None else None, sampled_capacity * 100 if sampled_capacity is not None else None,
                           _median(row.get("nvml_samples_per_request")), _median(row.get("nvml_observed_sampling_period_ms")))])
    parts.append(table(["方案", "场景", "帧数", "阶段", "分配器设备", "请求数", "耗时中位 ms", "请求耗时占比 %", "分母中位 ms", "分母字段", "配对数", "CPU 平均核数"],stage_rows)+
        '<h3>显存：常驻分配、缓存与整卡占用</h3><p>以下均先逐请求统计，再取中位数。起点 allocated 是该阶段首个窗口开始时的活跃张量分配；reserved 包含 PyTorch 缓存。NVML 整卡已用包含分配器以外的占用，并按设备和阶段窗口匹配；轮询等待间隔不含查询开销，实际采样周期另列，可能漏掉短峰值。无窗口内样本时采样峰值为空，边界最大值单独列出，不能替代完整阶段峰值。</p>'+table(
        ["方案", "场景", "帧数", "阶段", "分配器设备", "起点 allocated GiB", "allocated 峰值 GiB", "reserved 峰值 GiB", "allocated 增量 GiB", "整卡采样峰值 GiB", "整卡边界最大值 GiB", "allocated / CUDA 容量 %", "采样已用 / NVML 总量 %", "窗口内样本数", "实测采样周期 ms"],memory_rows)+'</section>')
    counters = []
    for row in metrics["stage_hardware_counters"]:
        values = row["metrics"]
        counters.append([cell(value) for value in (row["variant"], row["purpose"], row["frames"], row["stage"], row["physical_device_id"],
            values.get("sm_active_percent",{}).get("mean"), values.get("compute_warp_slots_percent_elapsed",{}).get("mean"),
            values.get("tensor_active_percent",{}).get("mean"), values.get("dram_total_throughput_percent",{}).get("mean"),
            values.get("sm_active_percent",{}).get("coverage_fraction"))])
    parts.append('<section><h2>阶段硬件计数器</h2><p>百分比均为时间加权；Warp 槽占用使用 elapsed cycles 分母，不等同于 Nsight Compute 的 achieved occupancy。计数器反映整张卡在该阶段窗口内的活动。</p>'+table(
        ["方案", "场景", "帧数", "阶段", "物理 GPU", "SM Active %", "Warp 槽占用 %", "Tensor Active %", "DRAM 总吞吐 %", "采样覆盖比例"],counters)+'</section>')
    diagnostic_rows = []
    for source in metrics.get("kernel_diagnostics", []):
        selection = source.get("selection", {})
        limit = selection.get("launch_count_limit")
        prefix = "首 %d 个 kernel 上限" % limit if limit is not None else "采样选择未提供"
        for row in source["summaries"]:
            values = row.get("metrics", {})
            duration = row.get("summed_profiled_kernel_duration_ns")
            diagnostic_rows.append([cell(value) for value in (
                row.get("variant"), row.get("frames"), row.get("stage"), row.get("dit_steps"), row.get("device"),
                row.get("scope"), prefix, selection.get("range_filter"), row.get("profiled_kernel_count"),
                duration / 1e6 if _number(duration) else None,
                values.get("achieved_occupancy_percent", {}).get("duration_weighted_mean"),
                values.get("sm_active_percent", {}).get("duration_weighted_mean"),
                values.get("dram_throughput_percent", {}).get("duration_weighted_mean"),
                (source.get("coverage") or {}).get("selected_kernel_count_fraction"))] + [link(source["source"], "诊断证据")])
    parts.append('<section><h2>Nsight Compute：所选 kernel 的 occupancy 诊断</h2><p>按所选 kernel 的 gpu__time_duration.sum 加权，不能代替完整阶段的 occupancy。当前捕获的 launch-count 上限和 range-filter 直接来自匹配的 launch 记录；首 128 个 kernel 只是被选中 NVTX 范围的前缀，不代表全部文本编码、全部 DiT 步骤或完整 VAE。帧数及 DiT 步编号取自内核 NVTX 标签，步编号沿用程序的零起点。重放会改变并发与缓存；这里的时间只用于诊断权重，不能计算服务加速比。完整范围的 kernel 总数未知时，覆盖率保持为空。</p>'+table(
        ["方案", "帧数", "阶段", "DiT 步", "GPU", "选择范围", "捕获上限", "range-filter", "kernel 数", "诊断时间合计 ms", "Achieved occupancy %", "SM Active %", "DRAM 吞吐 %", "kernel 数覆盖率", "来源"],diagnostic_rows)+'</section>')
    replay_rows, embedding_rows = [], []
    for source in metrics.get("embedding_replay", []):
        for row in source["results"]:
            for name, comparison in row.get("comparison", {}).items():
                replay_rows.append([cell(value) for value in (row.get("variant"), row.get("frames"), row.get("steps"),
                    row.get("purpose"), name, comparison.get("max_abs"), comparison.get("rmse"), comparison.get("cosine"),
                    comparison.get("exact"), comparison.get("finite"), comparison.get("shape_match"))] + [link(source["source"],"数值证据")])
            for representation, tensors in row.get("embedding_comparison", {}).items():
                for name, comparison in tensors.items():
                    embedding_rows.append([cell(value) for value in (row.get("variant"), representation, name,
                        comparison.get("max_abs"), comparison.get("rmse"), comparison.get("cosine"), comparison.get("exact"))])
    parts.append('<section><h2>GPU 文本编码：完整去噪与视频数值比较</h2><p>使用已保存的 T1/T2 embedding 继续执行完整视频流水，与 T0 输出比较；这些重放仅用于数值验证，不纳入速度、吞吐或常驻显存结论。T0 bridge 检查验证重放路径本身。</p>'+table(
        ["方案","帧数","步数","用途","输出","最大绝对误差","RMSE","Cosine","精确一致","有限值","形状一致","来源"],replay_rows)+
        '<h3>编码器输出与 DiT FP16 输入</h3>'+table(["方案","表示","条件","最大绝对误差","RMSE","Cosine","精确一致"],embedding_rows)+'</section>')
    comparisons = []
    for row in metrics["overlap_experiments"]:
        ratio = _median(row.get("paired_speedup"))
        mode = row.get("mode")
        if row.get("experiment") == "E2":
            mode = {"single":"单卡基线 single", "single_other":"另一单卡基线 single_other", "dual":"双卡请求并行 dual"}.get(mode,mode)
        numeric = "参考输出" if row.get("numeric_role") == "reference" else row.get("numeric_passed")
        comparisons.append([cell(row.get("experiment")),cell(row.get("frames")),cell(row.get("steps")),cell(mode),
                            cell(", ".join(row.get("devices", [])) or None),cell(row.get("condition")),cell(_median(row.get("wall_ms"))),cell(ratio),
                            cell(_median(row.get("paired_speedup_vs_other_single"))),cell((ratio-1)*100 if ratio is not None else None),
                            cell(numeric),link(row["source"],"原始结果")])
    parts.append('<section><h2>E1–E6 对照实验</h2><p>加速比为配对的基线耗时 ÷ 当前方案耗时；大于 1 表示观测到加速。E2 主比值使用 single（选定卡）基线，single_other 的比值为选定卡单卡 ÷ 另一卡单卡；dual 额外列出另一卡单卡 ÷ 双卡。E2 三种方案均在每个 worker 内使用 prefetch：完整流水线包含后台预取的 CPU 编码，预编码方案提前缓存 embedding。两张卡分别作为完整单卡基线，设备和数值比较不合并。E5 单模型基线不能与混合工作量直接比较；E6 使用两 rank 的最大耗时。正确性和真实重叠分别判断。</p>'+table(
        ["实验", "帧数", "步数", "方案", "设备", "条件", "耗时中位 ms", "主基线加速比", "E2 另一单卡 / 双卡", "吞吐相对变化 %", "数值检查", "来源"],comparisons)+'</section>')
    encoding_rows = metrics.get("e1_cpu_encoding", [])
    if encoding_rows:
        parts.append('<section><h2>E1：调度过程中实际观察到的 CPU 编码耗时</h2><p>这些是正常队列实验中的单次 CPU 编码窗口，按首请求和后续请求分开；并行时阶段自身的耗时可能变化，不能假定等于单独测量值。窗口可能与 GPU 工作重叠，不累加成请求耗时占比。这里描述观测，不证明 CPU 竞争或其来源。</p>'+table(
            ["帧数", "步数", "方案", "请求位置", "窗口数", "轮次", "中位 ms", "p95 ms", "范围 ms", "来源"],
            [[cell(row["frames"]),cell(row["steps"]),cell(row["mode"]),cell("首请求" if row["request_group"] == "first_request" else "后续请求"),
              cell(row["windows"]),cell(row["repeats"]),cell(row["cpu_encode_wall_ms"]["median"]),cell(row["cpu_encode_wall_ms"]["p95"]),
              cell(_display(row["cpu_encode_wall_ms"]["min"])+" – "+_display(row["cpu_encode_wall_ms"]["max"])),link(row["source"],"正常运行记录")]
             for row in encoding_rows])+'</section>')
    setup_rows = [row for row in metrics["overlap_experiments"] if row.get("experiment") == "E3"]
    if setup_rows:
        parts.append('<section><h2>E3：输入准备与 pinning 成本</h2><p>两种方案均预先完成 CPU 文本编码。主表 wall_ms 不包含 CPU latent 创建、输入准备及 pinning；下面单列 input_setup_ms 与包含该成本的总耗时，加速比也按同轮次完整范围重新配对。</p>'+table(
            ["帧数", "方案", "预编码", "wall_ms 中位", "输入准备 ms", "包含准备的总 ms", "包含准备的加速比"],
            [[cell(row["frames"]),cell(row["mode"]),cell(row["condition"].get("preencoded")),cell(_median(row["wall_ms"])),
              cell(_median(row.get("input_setup_ms"))),cell(_median(row.get("including_input_setup_ms"))),
              cell(_median(row.get("paired_speedup_including_input_setup")))] for row in setup_rows])+'</section>')
    queues = []
    for row in metrics.get("dedicated_encoder_queues", []):
        queues.append([cell(value) for value in (row["variant"],row["frames"],row["mode"],row["queue_size"],row["repeats"],
            _median(row["wall_ms"]),_median(row["videos_per_second"]),_median(row["paired_speedup"]),
            "参考输出" if row["numeric_role"] == "reference" else row["numeric_passed"])] + [link(row["source"],"原始结果")])
    parts.append('<section><h2>T5 专卡编码：串行与预取队列</h2><p>GPU 0 编码、GPU 1 去噪/解码，在相同硬件上按帧数和轮次配对 serial 与 prefetch。队列耗时含输出哈希检查；latent 和 video 均精确一致才通过数值检查。该组提示词/种子与 E2 不同，不能据此直接比较两种整机部署的加速比。</p>'+table(
        ["方案", "帧数", "调度", "队列请求数", "轮次", "耗时中位 ms", "video/s 中位", "serial / 当前", "latent + video 一致", "来源"],queues)+'</section>')
    overlap_evidence = []
    communication_rows = []
    for row in metrics.get("trace_overlap_summary", []):
        overlap_evidence.append([cell(row["arm"]),cell(row["device"]),cell(row["status"]),cell(row["envelope_ms"]),
            cell("执行包络" if row.get("execution_envelope_ms") else "host 包络（旧记录）"),
            cell(row["compute_union_ms"]),cell(row["compute_envelope_percent"]),
            cell(row["cpu_encode_gpu_compute_overlap_ms"]),cell(row["wan_sana_kernel_overlap_ms"]),
            link(row["source"],"解析结果"),link(row.get("trace_source"),"SQLite")])
        for name in ("h2d", "d2h", "nccl", "d2d", "p2p", "other_copy"):
            value = row["communication"][name]
            # H2D/D2H/NCCL are requested comparisons; show other copy classes
            # only when observed to keep the table compact without hiding work.
            if name not in {"h2d", "d2h", "nccl"} and not value.get("union_ms"):
                continue
            coverage = value["communication_coverage"]
            communication_rows.append([cell(item) for item in (row["arm"],row["device"],name.upper(),value["union_ms"],
                value["envelope_percent"],value["compute_overlap_ms"],coverage * 100 if coverage is not None else None,value["bytes"])])
    dual_rows = []
    for trace in metrics["trace_evidence"]:
        for arm in trace["arms"]:
            for pair in arm.get("dual_gpu_overlap", []):
                available = (trace.get("availability") or {}).get("cuda_kernels") is True
                duration = pair.get("simultaneous_compute_ms") if available else None
                denominator, _ = _arm_envelope(arm)
                share = duration / denominator * 100 if _number(duration) and _number(denominator) and denominator > 0 else None
                dual_rows.append([cell(arm["name"]),cell(pair.get("device_ids")),cell(duration),cell(share)])
    parts.append('<section><h2>设备时间线：实际计算与通信占比</h2><p>占比分母为该实验臂已纳入的 host NVTX 阶段及其关联 GPU 操作的执行包络，包含异步 GPU 尾部；旧记录回退到明确标注的 host 包络。它不替代正常运行的完整请求时间。计算与通信可以重叠，占比相加可超过 100%。通信覆盖率 = 通信与计算区间的交集 ÷ 通信区间并集。CUDA kernel 或 memcpy 表缺失时，相应数据保持未测。原始时间线优先链接保留的 .nsys-rep 捕获；SQLite 是可重新导出的中间文件，精简后可能不再保留。</p>'+table(
        ["实验臂", "GPU", "证据状态", "分母 ms", "分母口径", "计算区间 ms", "计算占比 %", "CPU 编码/GPU 交集 ms", "Wan/Sana kernel 交集 ms", "汇总", "原始时间线"],overlap_evidence)+
        '<h3>通信类型与计算重叠</h3>'+table(["实验臂", "GPU", "通信类型", "通信并集 ms", "包络占比 %", "计算交集 ms", "通信覆盖率 %", "传输字节"],communication_rows)+
        '<h3>双 GPU 同时计算</h3>'+table(["实验臂", "物理 GPU", "交集 ms", "包络占比 %"],dual_rows)+'</section>')
    parts.append('<section><h2>来源与缺测</h2><p>匹配到超时、外部 GPU 工作或监控失败的 launch 时，即使子程序写下 completed，也排除该轮性能与资源统计；profiler 的 capture 输出同样匹配 launch，排除受影响的硬件计数器和时间线。较早无监控的已完成记录及无匹配 launch 的证据单独标明。</p>'+table(["来源", "用途", "执行状态", "有效性", "SHA256"],
        [[link(source["path"]),cell(source["role"]),cell(source["recorded_status"]),cell((source.get("validity") or {}).get("status")),
          '<code>'+cell(source["sha256"])+'</code>'] for source in metrics["sources"]]))
    if metrics.get("superseded_executions"):
        parts.append('<h3>撤换的历史任务</h3><p>执行前撤换标记及显式 job-supersessions.json 声明的中断任务保留原始状态、原因与替代任务，排除有效性能和完成度汇总；部分已运行数据仍保留，后续完整任务独立验收。</p>'+table(
            ["实验", "帧数", "历史类别", "原始状态", "原因", "计划替代任务", "保留记录"],
            [[cell(row.get("experiment") or row.get("variant")),cell((row.get("config") or {}).get("frames")),
              cell(row.get("history_role")),cell(row["status"]),cell(row.get("reason")),cell(row.get("planned_replacement")),link(row["source"],"历史记录")]
             for row in metrics["superseded_executions"]]))
    for failure in run["failures"] + metrics["discovery_warnings"]:
        parts.append('<details><summary>未完成或读取问题</summary><pre>'+html.escape(json.dumps(failure,ensure_ascii=False,indent=2))+'</pre></details>')
    parts.append('<ul>'+''.join('<li>'+html.escape(note)+'</li>' for note in metrics["limitations"])+ '</ul></section></html>')
    return ''.join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--encoder-root", type=Path)
    parser.add_argument("--include", type=Path, nargs="+", help="Explicit files/directories; replaces automatic discovery")
    parser.add_argument("--destination", type=Path, help="Separate canonical archive directory")
    parser.add_argument("--preview", action="store_true", help="Write ANALYSIS-LIVE.html only; leave canonical run and archive unchanged")
    parser.add_argument("--keep-raw", action="store_true", help="Archive without pruning redundant raw artifacts")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    saved = archived_run(root) if not args.preview else None
    run = saved if saved is not None else build_report(root, args.encoder_root, args.include, final=not args.preview)
    analysis = root / ("ANALYSIS-LIVE.html" if args.preview else "ANALYSIS.html")
    if saved is None:
        analysis.write_text(render_html(run), encoding="utf-8")
    if args.preview:
        print(json.dumps(dict(status="preview", run_status=run["status"], analysis=str(analysis))))
        return
    target = root / "run.json"
    if saved is None:
        binding = _binding(analysis)
        artifact = next((row for row in run["artifacts"] if row["path"] == analysis.name), None)
        if artifact is None:
            artifact = dict(path=analysis.name, dependencies=[])
            run["artifacts"].append(artifact)
        artifact.update(bytes=binding["bytes"], sha256=binding["sha256"], kind="control")
        if target.exists() and json.loads(target.read_text()).get("schema") != "ditserve.run.v1":
            raise ValueError("refusing to overwrite noncanonical run.json")
        temporary = root / ".workflow-report-run.json.tmp"
        temporary.write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)
    result = finalize_workflow(root, args.destination, keep_raw=args.keep_raw)
    print(json.dumps(dict(status=run["status"], analysis=str(analysis), canonical_run=str(target), finalization=result)))
    if result.get("status") != "completed":
        raise RuntimeError("canonical archive finalization did not complete")


if __name__ == "__main__":
    main()
