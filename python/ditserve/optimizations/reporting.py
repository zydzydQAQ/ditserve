"""Source-bound finite-screen reports and non-destructive compact archives.

Reports never upgrade partial/component evidence into online qualification.
Archive builds a deliberately small staging tree, not an export of compiler
caches. It never deletes original files; its retention plan is review-only.
"""

import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from ..directions.direction1 import quantile
from ..research.common import atomic_json
from .protocol import PHASES, TOTAL, LIMITS


SCOPE = "finite paired screening; descriptive p95, no stable p99 or semantic-quality claim"
MAX_METADATA_BYTES = 32 * 1024 * 1024
OMIT = {"rows", "requests", "events", "quanta", "stages", "traceback", "inputs",
        "artifact", "retention", "first_calls", "active_trace", "active_unit", "active_partial"}
LOCAL_NAMES = {"sana-compile.json", "sana-batch.json", "wan-compile.json", "cache-k4.json",
               "text-encoder.json", "vae.json", "text_encoder.json"}


def _inside(path, root):
    return path == root or root in path.parents


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path):
    before = path.stat()
    value = dict(path=str(path), bytes=before.st_size, sha256=_hash(path))
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("source changed while reading: " + str(path))
    return value


def _read(path):
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError("metadata symlinks are not followed")
    before = path.stat()
    if before.st_size > MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds bounded reporting limit")
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("metadata changed while reading")
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _small(value, depth=0):
    """Keep scalar evidence; explicitly summarize long numeric arrays/lists."""
    if depth > 10:
        return {"omitted": "nesting limit; original source retained"}
    if isinstance(value, dict):
        return {key: _small(child, depth + 1) for key, child in value.items() if key not in OMIT}
    if isinstance(value, list):
        if len(value) <= 16:
            return [_small(child, depth + 1) for child in value]
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value):
            return dict(aggregation="numeric_array", **_stats(value), original_count=len(value))
        return dict(omitted="long list; original source retained", original_count=len(value))
    if isinstance(value, str) and len(value) > 2000:
        return dict(omitted="long text; original source retained", characters=len(value),
                    sha256=hashlib.sha256(value.encode()).hexdigest())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _stats(values):
    values = [float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                p95=quantile(values, .95), minimum=min(values) if values else None,
                maximum=max(values) if values else None)


def _reuse_origin(data):
    """Label copied measurements without treating replay admission as a rerun."""
    if "local_evidence_reuse" not in data:
        return None
    reuse = data.get("local_evidence_reuse")
    reuse = reuse if isinstance(reuse, dict) else {}
    binding = reuse.get("binding") if isinstance(reuse.get("binding"), dict) else {}
    original = dict(path=reuse.get("original_local_path"), sha256=reuse.get("original_local_sha256"))
    return dict(mode="reused_not_remeasured" if reuse.get("remeasured") is False else "reuse_declared_unverified",
                executed_in_current_attempt=False, original_source=original,
                original_attempt=binding.get("source_attempt"), original_session_id=binding.get("session_id"),
                scope="copied local measurements and compilation evidence; not new samples or current-process cold-start cost",
                binding_scope="producer replay admission; reporting does not repeat artifact or mathematical-source validation")


def _setup_windows(flushes):
    """Only subtract same-process perf-counter endpoints, never GPU anchors."""
    windows, preceding = [], None
    for index, row in enumerate(flushes):
        if row.get("kind") == "cumulative_record":
            preceding = (index, row) if row.get("status") == "completed" else None
        elif row.get("kind") == "prepare_partial" and preceding is not None:
            previous_index, previous = preceding
            start, end = previous.get("finished_perf_counter_s"), row.get("started_perf_counter_s")
            if (row.get("status") == "completed" and all(isinstance(x, (float, int)) and
                    not isinstance(x, bool) and math.isfinite(x) for x in (start, end)) and end >= start):
                windows.append(dict(start_pointer="/flushes/%d/finished_perf_counter_s" % previous_index,
                    end_pointer="/flushes/%d/started_perf_counter_s" % index, host_ms=(end - start) * 1000))
            preceding = None
    return dict(windows=windows, host_ms=_stats([row["host_ms"] for row in windows]),
                scope="mixed host window from cumulative-save completion to next trace preparation; may include capacity, backend preparation/compilation, warmup, bookkeeping and sidecar IO; excludes model loading before the first cumulative save; not compile-only or GPU time; already within allocated cost")


def _authorized(row):
    return bool(row and row.get("status") in ("passed", "completed", "stopped")
                and row.get("returncode") == 0 and row.get("charge")
                and not row.get("supervisor_failure")
                and row.get("final_occupancy", {}).get("status") == "exclusive_observed")


def _walk(value, pointer=""):
    yield pointer, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, pointer + "/" + str(index))


def _quality_failures(row, source, pointer):
    found = []
    for suffix, value in _walk(row):
        if not isinstance(value, dict):
            continue
        if "/quality" in suffix and "/numerical" in suffix:
            # Cross-backend LPIPS gates deliberately tolerate numerical changes.
            # Their diagnostic allclose result is not a second quality gate.
            continue
        relevant = any(word in (str(row.get("kind", "")) + suffix).lower()
                       for word in ("quality", "numerical", "comparison", "resume"))
        failed = ((relevant and value.get("passed") is False)
                  or value.get("repeat_exact") is False or value.get("cache_paths_equal") is False
                  or value.get("quality_status") in ("failed", "interrupted"))
        if failed:
            found.append(dict(source=source, pointer=pointer + suffix, evidence=_small(value)))
    return found


def _artifacts(row, source, pointer, phase, attempt):
    found = []
    for suffix, value in _walk(row):
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            continue
        if Path(value["path"]).suffix.lower() not in (".npy", ".npz", ".pt", ".pth"):
            continue
        path = Path(value["path"])
        path = path if path.is_absolute() else attempt / path
        input_id, arm = row.get("input_id"), row.get("arm", row.get("backend"))
        image_quality = None
        failed = bool(_quality_failures(row, source, pointer))
        if phase in ("sana_compile", "sana_batch"):
            quality = row.get("quality")
            if isinstance(quality, dict) and row.get("artifact") is value:
                image_quality = dict(source=source, pointer=pointer + "/quality", evidence=_small(quality))
            elif isinstance(quality, list):
                matches = [(index, check) for index, check in enumerate(quality)
                           if isinstance(check, dict) and check.get("artifact") is value]
                if len(matches) == 1:
                    index, check = matches[0]
                    input_id = check.get("input_id") if check.get("input_id") in row.get("input_ids", []) else None
                    image_quality = dict(source=source, pointer=pointer + "/quality/%d" % index,
                                         evidence=_small(check))
                    failed = check.get("passed") is False or row.get("repeat_exact") is False
            if phase == "sana_batch":
                if row.get("kind") == "eager_reference":
                    arm = "eager_reference"
                elif row.get("batch_size") in (1, 2):
                    arm = "batch-%d" % row["batch_size"]
        record = dict(phase=phase, path=str(path.absolute()), expected_sha256=value.get("sha256"),
                          source=source, pointer=pointer + suffix, shape=value.get("shape"),
                          failed=failed, quality_failed=failed,
                          input_id=input_id, kind=row.get("kind"), split=row.get("split"),
                          order=row.get("round", row.get("order")), repeat=row.get("repeat"),
                          native=row.get("native"), arm=arm)
        if image_quality is not None:
            record["image_quality"] = image_quality
        found.append(record)
    return found


def _bind_cache_pair_quality(artifacts, rows, source):
    """Attach the separate candidate-relative gate only to an unambiguous pair."""
    def identity(row):
        fields = (row.get("split"), row.get("order"), row.get("input_id"))
        return fields if all(value is not None for value in fields) else None
    pairs, outputs = collections.defaultdict(list), collections.defaultdict(list)
    for index, row in enumerate(rows):
        if isinstance(row, dict) and row.get("kind") == "cache_qualification_pair":
            key = identity(row)
            if key is not None:
                pairs[key].append((index, row))
    for artifact in artifacts:
        if artifact.get("source") == source and artifact.get("kind") == "cache_qualification":
            key = identity(artifact)
            if key is not None:
                outputs[key].append(artifact)
    for key, matches in pairs.items():
        generated = outputs.get(key, [])
        reference = [item for item in generated if item.get("arm") == "off"]
        candidate = [item for item in generated if item.get("arm") == "paired"]
        if len(matches) != 1 or len(generated) != 2 or len(reference) != 1 or len(candidate) != 1:
            continue
        index, pair = matches[0]
        quality = pair.get("quality")
        if pair.get("complete") is not True or not isinstance(quality, dict) or type(quality.get("passed")) is not bool:
            continue
        candidate[0]["pair_quality"] = dict(source=source, pointer="/rows/%d/quality" % index,
            split=key[0], order=key[1], input_id=key[2], candidate_arm="paired",
            reference_artifact=dict(path=reference[0]["path"], pointer=reference[0]["pointer"]),
            evidence=_small(quality), scope="candidate relative to the unique same-source off reference")
        candidate[0]["failed"] = candidate[0]["failed"] or quality["passed"] is False
        candidate[0]["quality_failed"] = candidate[0]["quality_failed"] or quality["passed"] is False


def _event_summary(events):
    groups = collections.defaultdict(list)
    for event in events:
        if not isinstance(event, dict):
            continue
        start, end = event.get("start_ms"), event.get("end_ms")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            groups[event.get("kind", "unknown")].append(end - start)
    return dict(event_count=len(events), intervals={kind: _stats(values) for kind, values in groups.items()},
                scope="nested/concurrent host intervals; sums must not be interpreted as GPU utilization",
                original_records_retained=True)


def _trace_identity(row):
    """Identity of one ordered arm, shared by cumulative and active records."""
    if not isinstance(row, dict):
        return None
    trace, order = row.get("trace_sha256"), row.get("order_index")
    arm = row.get("factorial_arm", row.get("arm"))
    if (not isinstance(trace, str) or not trace or type(order) is not int or order < 0
            or not isinstance(arm, str) or not arm):
        return None
    return trace, order, arm


def _attempt_metadata(reference, attempt, output, basename):
    if not isinstance(reference, str) or not reference:
        raise ValueError("metadata reference must name same-attempt " + basename)
    path = Path(reference)
    path = path if path.is_absolute() else attempt / path
    if path.absolute() != attempt.absolute() / basename or not _inside(path.resolve(), output):
        raise ValueError("metadata reference must be same-attempt " + basename + " inside output")
    # _read rejects file/parent symlinks and oversized metadata before IO.
    return _read(path)


def _online_records(data, source, attempt, output, errors):
    """Read one bounded active snapshot without attributing it to online.json."""
    rows = data.get("rows", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("online rows must be a list of trace records")
    records = [(row, source, "/rows/" + str(index), bool(row.get("partial")))
               for index, row in enumerate(rows)]
    merged = {_trace_identity(row) for row in rows} - {None}
    active_path = data.get("active_trace_path")
    if active_path is not None:
        try:
            active, active_source = _attempt_metadata(active_path, attempt, output, "partial.json")
            identity = _trace_identity(active)
            if identity is None:
                raise ValueError("active trace lacks trace_sha256/order_index/arm identity")
            if identity not in merged:
                records.append((active, active_source, "", True))
        except (OSError, ValueError, TypeError) as error:
            # Keep cumulative evidence visible while rejecting the unsafe
            # snapshot; phase-level errors already withhold qualification.
            errors.append("active trace " + str(active_path) + ": " + str(error))
    elif isinstance(data.get("active_trace"), dict):
        active = data["active_trace"]
        # Legacy records without identity remain descriptive partial evidence.
        if _trace_identity(active) not in merged:
            records.append((active, source, "/active_trace", True))
    return records


def _phase(output, phase, *, current_attempt=None, historical=False, include_history=True):
    root, errors = output / phase, []
    terminal_path = Path(current_attempt) / "supervisor.json" if current_attempt is not None else root / "terminal.json"
    terminal, terminal_source = ({}, None)
    if terminal_path.exists():
        try:
            terminal, terminal_source = _read(terminal_path)
        except (OSError, ValueError) as error:
            errors.append(str(error))
    attempt = Path(current_attempt) if current_attempt is not None else Path(terminal["output"]) if terminal.get("output") else None
    if attempt is not None and not _inside(attempt.resolve(), output):
        errors.append("terminal attempt points outside output; not read")
        attempt = None
    if attempt is None:
        candidates = sorted(path for path in root.glob("attempt[0-9][0-9]") if path.is_dir() and not path.is_symlink())
        attempt = candidates[-1] if candidates else None
    authority = _authorized(terminal) and not historical
    result = dict(phase=phase, status=terminal.get("status", "not_run"), terminal_source=terminal_source,
                  supervisor_valid=authority, terminal_present=bool(terminal),
                  decision=terminal.get("decision", "未验证") if authority or terminal.get("gpu_work_started") is False else "未验证",
                  reason=terminal.get("reason"), local_qualified=authority and terminal.get("local_qualified") is True,
                  online_qualified=authority and terminal.get("online_qualified") is True,
                  selected_backend=terminal.get("selected_backend"), charge=_small(terminal.get("charge")),
                  local=[], local_samples=[], online=[], request_samples=[], event_summaries=[],
                  quality_failures=[], artifacts=[], errors=errors, scope=SCOPE,
                  attempt_path=str(attempt) if attempt else None, historical=historical,
                  audit_notes=[], previous_attempts=[], previous_samples=[])
    if terminal.get("online_unverified_reason"):
        # Preserve the producer's original reason and surface the separately
        # recorded online admission limit without changing qualification.
        result["reported_reason"] = terminal.get("reason")
        result["online_unverified_reason"] = terminal["online_unverified_reason"]
        result["reason"] = "; ".join(filter(None, (terminal.get("reason"),
            "在线未验证原因：" + str(terminal["online_unverified_reason"]))))
    if attempt is None:
        return result
    for path in sorted(attempt.glob("*audit*.json")):
        try:
            note, source = _read(path)
            result["audit_notes"].append(dict(source=source, evidence=_small(note),
                invalidates_qualification=path.name.endswith("-audit-stop.json") or
                                         (isinstance(note, dict) and note.get("invalidates_qualification") is True),
                invalidates_online_qualification=isinstance(note, dict) and
                                                 note.get("invalidates_online_qualification") is True))
        except (OSError, ValueError) as error:
            errors.append(str(path) + ": " + str(error))
    if any(note["invalidates_qualification"] for note in result["audit_notes"]):
        # An explicitly audited/stopped attempt cannot regain qualification
        # through stale success fields in its supervisor or component file.
        authority = False
        result.update(supervisor_valid=False, local_qualified=False, online_qualified=False, decision="未验证")
    online_invalidated = any(note["invalidates_online_qualification"] for note in result["audit_notes"])
    online_authority = authority and not online_invalidated
    if online_invalidated:
        result.update(online_qualified=False, decision="未验证", online_audit_invalidated=True,
            online_qualification_scope="online measurement audit invalidates online conclusions only; local evidence retained")
    candidates = sorted(path for path in attempt.glob("*.json") if path.name in LOCAL_NAMES)
    # Future stage implementations may use a different unambiguous evidence name.
    candidates += sorted(path for path in attempt.glob("*.json") if path not in candidates and path.name not in {
        "config.json", "worker.json", "supervisor.json", "environment.json", "online.json", "partial.json",
        "preflight.json", "compiler-warmup.json"})
    for path in candidates:
        try:
            data, source = _read(path)
            if not isinstance(data, dict) or not str(data.get("schema_version", "")).startswith("ditserve.optimizations."):
                continue
            if data.get("schema_version") in ("ditserve.optimizations.online.v1", "ditserve.optimizations.cache_factorial.v1"):
                continue
            rows = data.get("rows", [])
            if not isinstance(rows, list):
                continue
            summary = dict(source=source, status=data.get("status"), rows_observed=len(rows),
                           conclusion_scope="supervisor-qualified" if authority else "descriptive partial; no qualification",
                           reported_summary=_small(data), measured_groups=[],
                           continuation_pending=bool(data.get("active_unit") or data.get("budget_stopped") or
                               (data.get("required_full_qualification") and
                                (data.get("full_qualification", {}).get("complete") is not True or
                                 data.get("full_qualification", {}).get("qualified") is not True))))
            origin = _reuse_origin(data)
            if origin is not None:
                summary.update(measurement_origin=origin, new_local_execution_rows=0,
                               reused_local_rows=len(rows))
                summary["reported_summary"]["compilation_measurement_origin"] = origin
                summary["compilation_evidence"] = dict(measurement_origin=origin,
                    values=_small(data.get("compilation", {})), current_process_measurement=False)
                summary["current_process_cost"] = dict(included_in_allocation_ledger=True,
                    separate_components_measured=False, model_loading_ms=None, capacity_ms=None,
                    compilation_ms=None, warmup_ms=None,
                    scope="new process loading, capacity checks, compilation/warmup and online work are charged together; copied compilation costs belong to original attempt and must not be added again")
            groups = collections.defaultdict(list)
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                pointer = "/rows/" + str(index)
                sample = dict(phase=phase, source=source, pointer=pointer, evidence=_small(row))
                if origin is not None:
                    sample["measurement_origin"] = origin
                result["local_samples"].append(sample)
                result["quality_failures"].extend(_quality_failures(row, source, pointer))
                result["artifacts"].extend(_artifacts(row, source, pointer, phase, attempt))
                service = row.get("service", {})
                if (row.get("kind") == "measurement" or str(row.get("kind", "")).endswith("performance")) and isinstance(service.get("ms"), (float, int)):
                    identity = (row.get("kind"), row.get("backend", row.get("arm")), row.get("batch_size"),
                                row.get("round", row.get("order")))
                    groups[identity].append(service["ms"])
            _bind_cache_pair_quality(result["artifacts"], rows, source)
            for index, pair in enumerate(data.get("pairs", [])):
                if isinstance(pair, dict):
                    pointer = "/pairs/" + str(index)
                    sample = dict(phase=phase, source=source, pointer=pointer,
                                  evidence=dict(_small(pair), record_kind="paired_comparison"))
                    if origin is not None:
                        sample["measurement_origin"] = origin
                    result["local_samples"].append(sample)
                    result["quality_failures"].extend(_quality_failures(pair, source, pointer))
                    if _quality_failures(pair, source, pointer):
                        for artifact in result["artifacts"]:
                            if artifact["source"] == source and artifact.get("input_id") == pair.get("input_id"):
                                artifact["failed"] = True
            if isinstance(data.get("active_unit"), dict):
                unit = data["active_unit"]
                result["local_samples"].append(dict(phase=phase, source=source, pointer="/active_unit",
                    evidence=dict(_small(unit), record_kind="partial_active_unit")))
                active_artifacts = _artifacts(unit, source, "/active_unit", phase, attempt)
                for artifact in active_artifacts:
                    artifact["failed"] = True
                result["artifacts"].extend(active_artifacts)
            summary["measured_groups"] = [dict(kind=key[0], arm=key[1], batch_size=key[2], order=key[3],
                                                service_ms=_stats(values)) for key, values in groups.items()]
            if origin is not None:
                for group in summary["measured_groups"]:
                    group["measurement_origin"] = origin
            result["local"].append(summary)
        except (OSError, ValueError, TypeError) as error:
            errors.append(str(path) + ": " + str(error))
    online_path = Path(terminal["online_path"]) if terminal.get("online_path") else attempt / (
        "cache-online.json" if phase == "cache_k4" and (attempt / "cache-online.json").exists() else "online.json")
    if online_path.exists() and _inside(online_path.resolve(), output):
        try:
            data, source = _read(online_path)
            for row, row_source, pointer, partial in _online_records(data, source, attempt, output, errors):
                complete = (row.get("status") == "completed" and row.get("complete") is True and not partial
                            and bool(row.get("requests")) and all(r.get("status") == "completed" and r.get("arrived", True)
                                                                  for r in row["requests"]))
                result["online"].append(dict(source=row_source, pointer=pointer, complete=complete,
                    authoritative=online_authority and data.get("status") == "completed" and complete,
                    evidence=dict(_small(row), partial=True) if partial else _small(row),
                    request_count=len(row.get("requests", []))))
                result["event_summaries"].append(dict(source=row_source, pointer=pointer + "/events",
                    trace_sha256=row.get("trace_sha256"), order_index=row.get("order_index"), arm=row.get("arm"),
                    **_event_summary(row.get("events", []))))
                for request_index, request in enumerate(row.get("requests", [])):
                    result["request_samples"].append(dict(phase=phase, source=row_source,
                        pointer=pointer + "/requests/" + str(request_index), trace_sha256=row.get("trace_sha256"),
                        order_index=row.get("order_index"), arm=row.get("arm"), trace_complete=complete,
                        evidence=_small(request)))
            qualification = data.get("qualification", data.get("comparisons", {}).get("cache_on_k4", {}))
            result["online_qualification"] = _small(qualification)
            if "comparisons" in data:
                result["factorial_comparisons"] = _small(data["comparisons"])
            result["online_source"] = source
            if data.get("flush_log_path") is not None:
                try:
                    name = "cache-online-flushes.json" if online_path.name == "cache-online.json" else "online-flushes.json"
                    flush_log, flush_source = _attempt_metadata(data["flush_log_path"], attempt, output, name)
                    flushes = flush_log.get("flushes")
                    if (flush_log.get("schema_version") != "ditserve.host_persistence.v1" or
                            not isinstance(flushes, list) or any(not isinstance(row, dict) for row in flushes)):
                        raise ValueError("invalid host persistence log")
                    result["persistence_summary"] = dict(source=flush_source, pointer="/flushes",
                        flush_count=len(flushes), status_counts=dict(collections.Counter(row.get("status", "unknown") for row in flushes)),
                        host_ms=_stats([row.get("host_ms") for row in flushes]),
                        scope="host metadata persistence only; descriptive, not GPU time or qualification")
                    result["persistence_summary"]["pre_trace_mixed_setup"] = _setup_windows(flushes)
                except (OSError, ValueError, TypeError, AttributeError) as error:
                    errors.append("flush log " + str(data["flush_log_path"]) + ": " + str(error))
            result["online_qualified"] = bool(result["online_qualified"] and data.get("status") == "completed"
                and qualification.get("passed") is True and result["online"]
                and all(row["complete"] for row in result["online"]))
            result["online_evaluation_complete"] = bool(online_authority and data.get("status") == "completed"
                and not qualification.get("failures") and qualification.get("pooled")
                and result["online"] and all(row["complete"] for row in result["online"]))
            if (result["local_qualified"] and result["online_evaluation_complete"]
                    and qualification.get("passed") is False):
                result["reported_decision"] = result["decision"]
                result["decision"] = "当前配置无收益"
                result["decision_scope"] = "complete system comparison failed thresholds; local speed gain retained separately"
        except (OSError, ValueError, TypeError) as error:
            errors.append(str(online_path) + ": " + str(error))
    if "online_source" not in result:
        result["online_qualified"] = False
    if errors:
        result["online_qualified"] = False
    if result["decision"] == "值得继续" and not result["online_qualified"]:
        result["decision"] = "未验证"
    result["sample_counts"] = dict(local_rows=len(result["local_samples"]),
        reused_local_rows=sum("measurement_origin" in row for row in result["local_samples"]),
        online_traces=len(result["online"]), complete_online_traces=sum(row["complete"] for row in result["online"]),
        online_requests=len(result["request_samples"]), quality_failures=len(result["quality_failures"]))
    if include_history:
        for prior in sorted(path for path in root.glob("attempt[0-9][0-9]")
                            if path.is_dir() and not path.is_symlink() and path.name < attempt.name):
            old = _phase(output, phase, current_attempt=prior, historical=True, include_history=False)
            summary = {key: value for key, value in old.items() if key not in (
                "local_samples", "request_samples", "event_summaries", "quality_failures", "artifacts",
                "previous_attempts", "previous_samples", "online")}
            summary.update(local_qualified=False, online_qualified=False, decision="未验证",
                           qualification_eligible=False, accounting_scope="already included in the authoritative ledger; never add twice")
            result["previous_attempts"].append(summary)
            reasons = [note["evidence"].get("reason") for note in old["audit_notes"] if isinstance(note["evidence"], dict)]
            for category in ("local_samples", "request_samples", "event_summaries", "quality_failures"):
                for sample in old[category]:
                    result["previous_samples"].append(dict(sample, historical=True, attempt=str(prior),
                        category=category, qualification_eligible=False, stop_reason=old.get("reason"), audit_reasons=reasons))
            for artifact in old["artifacts"]:
                result["artifacts"].append(dict(artifact, historical=True, qualification_eligible=False,
                    failed=artifact["failed"] or old["status"] in ("failed", "incomplete", "stopped")
                           or any(note["invalidates_qualification"] for note in old["audit_notes"])))
    return result


def collect(output, ledger):
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    budget, budget_source = _read(ledger) if ledger.exists() else ({}, None)
    active = budget.get("active")
    active_seconds = max(0., time.monotonic() - active["started_monotonic"]) * active["gpu_count"] if active else 0.
    phases = {}
    for phase in PHASES:
        identity = active.get("identity", {}) if active else {}
        active_path = identity.get("output") if identity.get("phase") == phase else None
        phases[phase] = _phase(output, phase, current_attempt=active_path)
    for phase, value in phases.items():
        if not value["terminal_present"] and (value["local"] or value["online"]):
            value["status"] = "running" if active and active.get("identity", {}).get("phase") == phase else "partial"
    return dict(schema_version="ditserve.optimizations.report.v2", output=str(output), budget_path=str(ledger),
        budget_source=budget_source, budget=_small(budget), active_gpu_seconds_estimate=active_seconds,
        phase_budget_gpu_seconds=LIMITS, total_budget_gpu_seconds=TOTAL,
        all_terminal=all(row["terminal_present"] and row["status"] not in ("running", "not_run", "blocked") for row in phases.values()),
        phases=phases, scope=SCOPE, archive_ready=not active and bool(budget) and all(
            row["terminal_present"] and row["status"] not in ("running", "not_run", "blocked") for row in phases.values()))


def _fmt(value):
    return "—" if value is None else "%.3f" % value if isinstance(value, float) else str(value)


def _audit_description(note, *, include_source=False):
    evidence = note.get("evidence", {})
    texts = [str(evidence[key]) for key in ("reason", "failure_interpretation") if evidence.get(key)]
    text = "; ".join(dict.fromkeys(texts)) or "查看原始审计记录"
    if include_source:
        source = note["source"]
        text = "[%s](%s)：%s（SHA256 `%s`）" % (
            Path(source["path"]).name, source["path"], text, source["sha256"])
    return text


def render_markdown(report):
    lines = ["# DiT serving 优化有限初筛", "", "局部性能、在线收益与质量验收分别报告；描述性 p95，不声称稳定 p99。", "",
             "| 阶段 | 状态 | 结论 | 局部/在线资格 | 局部样本行 | 完整在线轨迹/请求 | 质量失败 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for phase, value in report["phases"].items():
        n = value.get("sample_counts", {})
        lines.append("| %s | %s | %s | %s / %s | %s | %s / %s | %s |" % (
            phase, value["status"], value["decision"], value["local_qualified"], value["online_qualified"],
            n.get("local_rows", 0), n.get("complete_online_traces", 0), n.get("online_requests", 0), n.get("quality_failures", 0)))
    used = report["budget"].get("used_gpu_seconds", 0.)
    lines += ["", "已结算 %.4f / 8 GPU 小时；活动会话估计 %.4f GPU 小时。" % (used / 3600, report["active_gpu_seconds_estimate"] / 3600),
              "阶段额度可向后结转，1 GPU 小时修复备用额度单独记账；双卡占用按卡数计费。", ""]
    for phase, value in report["phases"].items():
        lines += ["## " + phase, "", str(value.get("reason") or "尚无终局原因。"), ""]
        source = value.get("terminal_source")
        if source:
            lines.append("监督器来源：[terminal.json](%s)，SHA256 `%s`。" % (source["path"], source["sha256"]))
        for note in value["audit_notes"]:
            lines += ["", "本次审计记录：[%s](%s)：%s。" % (Path(note["source"]["path"]).name,
                note["source"]["path"], _audit_description(note))]
        if value["previous_attempts"]:
            lines += ["", "历史尝试（不授予本次资格；费用已包含在总账中）：", "",
                      "| 尝试 | 状态 | GPU 秒 | 局部样本行 | 停止/审计原因 |",
                      "| --- | --- | ---: | ---: | --- |"]
            for old in value["previous_attempts"]:
                reason = "; ".join([str(old.get("reason") or "")] +
                                   [_audit_description(note, include_source=True) for note in old["audit_notes"]])
                lines.append("| [%s](%s) | %s | %s | %s | %s |" % (Path(old["attempt_path"]).name,
                    old["terminal_source"]["path"] if old["terminal_source"] else old["attempt_path"], old["status"],
                    _fmt((old.get("charge") or {}).get("gpu_seconds")), old.get("sample_counts", {}).get("local_rows", 0),
                    reason.replace("|", "/").replace("\n", " ")))
        for local in value["local"]:
            lines += ["", "局部来源：[%s](%s)，观察到 %d 行；%s。" % (
                Path(local["source"]["path"]).name, local["source"]["path"], local["rows_observed"], local["conclusion_scope"])]
            origin = local.get("measurement_origin")
            if origin:
                original = origin["original_source"]
                lines += ["", "本次 %d 行局部证据为复用，新增局部执行 0 行，质量/恢复和性能均未重测。原始来源：[%s](%s)，SHA256 `%s`。历史与本次引用是同一组样本，不能合并增加样本量。" % (
                    local["reused_local_rows"], Path(original.get("path") or "unknown").name,
                    original.get("path") or "", original.get("sha256") or "未提供"),
                    "局部 compilation/cold-start 字段属于原始尝试 %s，不是本进程编译耗时。本进程加载、容量检查、编译预热及在线执行均计入本次持卡账单；各组件未单独结构化测量，不能用旧编译耗时替代或重复加账。" %
                    (Path(origin.get("original_attempt") or original.get("path") or "unknown").name)]
            for group in local["measured_groups"]:
                stats = group["service_ms"]
                lines.append("- %s / %s / batch=%s / order=%s：service ms mean=%s，p95=%s，n=%s。" % (
                    group["kind"], group["arm"], group["batch_size"], group["order"],
                    _fmt(stats["mean"]), _fmt(stats["p95"]), stats["n"]))
        setup = value.get("persistence_summary", {}).get("pre_trace_mixed_setup", {})
        if setup.get("host_ms", {}).get("n"):
            stats = setup["host_ms"]
            lines += ["", "轨迹前混合 host 准备间隔：mean=%s ms，p95=%s ms，n=%s。起止为累计结果保存结束至下一轨迹准备开始；可包含容量检查、后端准备/编译、预热和日志/主机工作，不是精确 compile-only 耗时，也不含首次累计保存前的模型加载。此间隔已在持卡费用内，不另行相加。" % (
                _fmt(stats["mean"]), _fmt(stats["p95"]), stats["n"])]
        if value["online"]:
            lines += ["", "| trace / order / arm | 完整 | goodput req/s | 图像 E2E p95 ms / n | 视频按时完成 / 到达 |",
                      "| --- | --- | --- | --- | --- |"]
            for row in value["online"]:
                data = row["evidence"]
                metrics, images, videos = (data.get(key, {}) for key in ("metrics", "image_metrics", "video_metrics"))
                lines.append("| %s / %s / %s | %s | %s | %s / %s | %s / %s |" % (
                    str(data.get("trace_sha256", ""))[:12], data.get("order_index"), data.get("arm", data.get("factorial_arm")), row["complete"],
                    _fmt(metrics.get("goodput_rps")), _fmt(images.get("latency_p95_ms")), images.get("sample_count"),
                    videos.get("on_time"), videos.get("arrived")))
            source = value.get("online_source")
            lines += ["", "在线来源：[online.json](%s)。失败与未完成请求仍在到达分母中；p95 的样本数仅为已完成请求。" % source["path"]]
        if value["quality_failures"]:
            lines += ["", "质量/恢复失败来源："]
            for failure in value["quality_failures"]:
                lines.append("- [%s](%s) `%s`。" % (Path(failure["source"]["path"]).name,
                                                        failure["source"]["path"], failure["pointer"]))
        if value["errors"]:
            lines += ["", "未能读取的证据："] + ["- " + error for error in value["errors"]]
        lines.append("")
    lines += ["原始逐事件记录、模型输出数组及编译缓存不直接嵌入 Excel。归档只在全部阶段终止且账本无活动会话时开放；清理清单仍需独立审阅。", ""]
    return "\n".join(lines)


def write_report(output, ledger):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = collect(output, ledger)
    atomic_json(output / "report.json", report)
    (output / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _preview(raw, destination):
    """Bounded contact sheet: at most eight frames and 4 MiB sampled pixels."""
    import numpy as np
    from PIL import Image
    array = np.load(raw, mmap_mode="r", allow_pickle=False)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4 or array.shape[-1] not in (3, 4) or not array.shape[0]:
        raise ValueError("preview requires [N/T,H,W,3/4] or [1,T,H,W,3/4]")
    indices = sorted(set(int(x) for x in np.linspace(0, array.shape[0] - 1, min(8, array.shape[0]))))
    # One spatial stride preserves geometry; independent axis strides stretch
    # square images and change the aspect ratio of landscape video frames.
    stride = max(1, math.ceil(array.shape[1] / 240), math.ceil(array.shape[2] / 384))
    images = []
    for index in indices:
        frame = np.asarray(array[index, ::stride, ::stride, :3])
        if not np.isfinite(frame).all():
            raise ValueError("nonfinite preview pixels")
        if frame.dtype != np.uint8:
            if frame.min() < 0 or frame.max() > 1:
                raise ValueError("float preview outside documented [0,1]")
            frame = np.rint(frame * 255).astype(np.uint8)
        images.append(Image.fromarray(frame))
    width, height = max(image.width for image in images), max(image.height for image in images)
    columns = min(4, len(images))
    sheet = Image.new("RGB", (width * columns, height * math.ceil(len(images) / columns)))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    sheet.save(destination)
    return dict(path=str(destination), sha256=_hash(destination), frame_indices=indices,
                sampling_stride=stride, sample_frame_size=[width, height],
                sampled_rgb_bytes=len(images) * width * height * 3,
                geometry_scope="same integer stride on both spatial axes; ceiling edge rounding only",
                format="PNG contact sheet", scope="sparse visual preview, not full video or semantic evaluation")


def _video_pair(artifacts):
    """Prefer failed quality, current evidence, then actual candidate comparisons."""
    groups = collections.defaultdict(list)
    for artifact in artifacts:
        shape = artifact.get("shape")
        if shape != [1, 81, 480, 832, 3]:
            continue
        kind = str(artifact.get("kind", ""))
        key = (artifact["source"]["path"], artifact.get("input_id"), kind,
               str(artifact.get("order")), artifact.get("repeat"), artifact.get("arm") if "resume" in kind else None,
               artifact.get("split"), artifact["source"].get("sha256"))
        groups[key].append(artifact)
    complete, missing = [], []
    for key, rows in groups.items():
        baseline, candidate = [], []
        for row in rows:
            role = row.get("arm")
            if "resume" in str(row.get("kind")):
                role = "reference" if row["pointer"].endswith("/0") else "candidate" if row["pointer"].endswith("/1") else role
            elif row.get("native") is not None:
                role = "reference" if row["native"] else "candidate"
            if role in ("eager", "off", "cpu", "baseline", "reference"):
                baseline.append(row)
            elif role in ("compiled", "candidate", "paired", "gpu"):
                candidate.append(row)
        if len(baseline) == 1 and len(candidate) == 1:
            complete.append([baseline[0], candidate[0]])
        else:
            missing.append(dict(source=key[0], input_id=key[1], kind=key[2], split=key[6], source_sha256=key[7],
                                reason="no unique reported reference/candidate pair", paths=[r["path"] for r in rows]))
    def quality_failed(row):
        # Historical ``failed`` also describes a stopped supervisor; it is not
        # evidence that this artifact failed a numerical or quality comparison.
        if "quality_failed" in row:
            return row["quality_failed"] is True
        bound_quality = row.get("pair_quality", {}).get("evidence", {})
        return (bound_quality.get("passed") is False
                or bool(row.get("failed") and not row.get("historical")))

    def comparison_rank(pair):
        reference, candidate = pair
        kind = str(reference.get("kind", ""))
        explicit_comparison = (
            "resume" not in kind and all(row.get("native") is None for row in pair)
            and reference.get("arm") in ("eager", "off", "cpu", "baseline", "reference")
            and candidate.get("arm") in ("compiled", "candidate", "paired", "gpu"))
        if not explicit_comparison:
            return 3  # Same-backend resume and native/split equivalence diagnostics.
        if kind in ("performance", "measurement", "cache_qualification"):
            return 0
        return 1 if kind == "cache_screen" else 2

    complete.sort(key=lambda pair: (
        not any(quality_failed(row) for row in pair),
        any(bool(row.get("historical")) for row in pair),
        comparison_rank(pair), pair[0]["path"], pair[1]["path"],
        pair[0]["source"]["path"], pair[0]["pointer"], pair[1]["pointer"]))
    return (complete[0] if complete else []), missing


def _failed_image_pairs(artifacts):
    """Use the producer's exact per-input quality anchor; never infer from names."""
    pairs = []
    for candidate in artifacts:
        phase, arm = candidate.get("phase"), candidate.get("arm")
        source = candidate.get("source", {})
        if (not candidate.get("failed") or candidate.get("input_id") is None
                or not source.get("path") or not source.get("sha256")
                or candidate.get("shape") != [1, 1024, 1024, 3]
                or (phase, arm) not in (("sana_compile", "compiled"), ("sana_batch", "batch-2"))):
            continue
        same = [row for row in artifacts if row.get("source") == candidate.get("source")
                and row.get("phase") == phase and row.get("input_id") == candidate["input_id"]
                and row.get("shape") == candidate.get("shape")]
        if phase == "sana_compile":
            same = [row for row in same if row.get("order") == candidate.get("order")
                    and row.get("repeat") == candidate.get("repeat")]
            references = [row for row in same if row.get("kind") == "measurement" and row.get("arm") == "eager"]
        else:
            references = [row for row in same if row.get("kind") == "eager_reference" and row.get("arm") == "eager_reference"]
        candidates = [row for row in same if row.get("arm") == arm and row.get("kind") == candidate.get("kind")
                      and row.get("order") == candidate.get("order") and row.get("repeat") == candidate.get("repeat")]
        if len(references) == 1 and len(candidates) == 1:
            pairs.append([references[0], candidate])
    return sorted(pairs, key=lambda pair: (pair[1]["input_id"], pair[1]["path"]))


def _write_video(raw, destination):
    """Stream the original 81 frames at 16 fps, validating one frame at a time."""
    import numpy as np
    import imageio.v2 as imageio
    from ..gpu.media import video_frames
    array = np.load(raw, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != (1, 81, 480, 832, 3):
        raise ValueError("fixed video preview requires the recorded [1,81,480,832,3] output")
    with imageio.get_writer(str(destination), fps=16, codec="libx264", quality=8,
                            macro_block_size=1, ffmpeg_params=["-threads", "1"]) as writer:
        for index in range(81):
            writer.append_data(video_frames(array[:, index:index + 1], np, allow_uint8=True)[0])
    with imageio.get_reader(str(destination)) as reader:
        metadata, count = reader.get_meta_data(), reader.count_frames()
        fps = metadata.get("fps")
    if fps is None or abs(fps - 16.) > 1e-6 or count != 81 or tuple(metadata.get("size", ())) != (832, 480):
        raise ValueError("encoded preview shape/frame count/fps failed readback")
    return dict(path=str(destination), sha256=_hash(destination), frame_count=count, fps=fps,
                shape=list(array.shape), duration_seconds=81 / 16,
                scope="CPU offline video export; playback rate is not measured serving throughput")


def archive(output, ledger, *, destination=None, preview_limit=4):
    """Export small metadata/samples after terminal accounting; NEVER prune.

    Original artifacts outside output are hashed and may be previewed but are
    always marked non-deletable. Original files are not sources of the compact
    workbook; only this staging tree is. Thus this workbook cannot authorize
    compact_results.prune on any original artifact/event file.
    """
    if type(preview_limit) is not int or not 0 <= preview_limit <= 4:
        raise ValueError("preview_limit must be between zero and four per phase")
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    report = collect(output, ledger)
    if not report["archive_ready"]:
        return dict(status="protected", reason="all phases must be terminal and GPU ledger inactive", scope=SCOPE)
    # Hold the exact Allocation lock during archive work, preventing a resumed
    # supervisor from beginning between the admission check and final verify.
    import fcntl
    from .. import compact_results
    with ledger.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = collect(output, ledger)
        if not report["archive_ready"]:
            return dict(status="protected", reason="accounting changed before archive lock", scope=SCOPE)
        destination = Path(destination).absolute() if destination else output / "excel"
        stage = output / "archive-metadata"
        if stage.exists() or destination.exists():
            raise FileExistsError("archive is immutable; choose a new run/archive after reviewing prior output")
        stage.mkdir()
        media = output / "archive-previews"
        media.mkdir(exist_ok=True)
        inventory, previews, videos, video_groups = [], [], [], []
        for phase, data in report["phases"].items():
            seen, used = set(), 0
            pair, missing = _video_pair(data["artifacts"])
            video_groups.append(dict(phase=phase, selected_sources=[row["path"] for row in pair],
                                     missing_counterparts=missing, complete=False))
            pending = (data["status"] in ("failed", "incomplete") or bool(data["errors"])
                       or any(local["continuation_pending"] for local in data["local"])
                       or any(local["reported_summary"].get("cache_quality_qualified") is False for local in data["local"]))
            image_pairs = _failed_image_pairs(data["artifacts"])
            preferred = pair + [artifact for group in image_pairs for artifact in group]
            priority = {artifact["path"]: index for index, artifact in reversed(list(enumerate(preferred)))}
            for artifact in sorted(data["artifacts"], key=lambda item: (priority.get(item["path"], len(preferred)), not item["failed"], item["path"])):
                if artifact["path"] in seen:
                    continue
                seen.add(artifact["path"])
                path = Path(artifact["path"])
                record = dict(artifact, raw_available=path.is_file(), deletable=False,
                              continuation_or_qualification_pending=pending or artifact.get("historical", False),
                              retention_reason="review-only archive; original raw not in compact source tree")
                if not _inside(path.resolve(), output):
                    record["retention_reason"] = "outside output; never prune across roots"
                elif pending or artifact.get("historical"):
                    record["retention_reason"] = "qualification, repair or continuation dependency unresolved"
                try:
                    if path.is_symlink() or path.resolve() != path.absolute():
                        raise ValueError("raw symlink not followed")
                    record["observed"] = _source(path)
                    expected = artifact.get("expected_sha256")
                    record["hash_matches"] = expected is None or expected == record["observed"]["sha256"]
                    if not record["hash_matches"]:
                        record["retention_reason"] = "artifact hash mismatch; preserve for audit"
                    elif used < preview_limit and path.suffix == ".npy":
                        preview_path = media / (phase + "-%02d.png" % used)
                        used += 1
                        preview = _preview(path, preview_path)
                        preview.update(phase=phase, source_path=str(path), source_sha256=record["observed"]["sha256"],
                                       failed_sample=artifact["failed"])
                        previews.append(preview)
                        record["preview"] = preview
                except (OSError, ValueError, ImportError) as error:
                    record["preview_or_hash_error"] = str(error)
                inventory.append(record)
            if pair:
                records = {item["path"]: item for item in inventory if item["phase"] == phase}
                group_videos = []
                try:
                    if not all(records[row["path"]].get("hash_matches") for row in pair):
                        raise ValueError("paired video sources missing or hash mismatch")
                    for index, artifact in enumerate(pair):
                        result = _write_video(Path(artifact["path"]), media / (phase + "-%s.mp4" % ("reference" if index == 0 else "candidate")))
                        result.update(phase=phase, role="reference" if index == 0 else "candidate", input_id=artifact.get("input_id"),
                            source_path=artifact["path"], source_sha256=records[artifact["path"]]["observed"]["sha256"],
                            failed_sample=artifact["failed"], pair_complete=False)
                        group_videos.append(result)
                    for result in group_videos:
                        result["pair_complete"] = True
                    video_groups[-1]["complete"] = True
                except (OSError, ValueError, ImportError, RuntimeError) as error:
                    video_groups[-1]["error"] = str(error)
                videos.extend(group_videos)
            # Sample JSON files remain small and preserve every selected row;
            # events are represented by explicit counts/distributions + hash.
            for key in ("local_samples", "request_samples", "event_summaries", "quality_failures", "previous_samples"):
                with (stage / (phase + "-" + key + ".jsonl")).open("w", encoding="utf-8") as stream:
                    for row in data[key]:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        summary = dict(report)
        summary["phases"] = {phase: {key: value for key, value in data.items()
                                    if key not in ("local_samples", "request_samples", "event_summaries", "artifacts", "quality_failures", "previous_samples")}
                             for phase, data in report["phases"].items()}
        atomic_json(stage / "report-summary.json", summary)
        atomic_json(stage / "raw-retention-plan.json", dict(status="review_only", original_files_deleted=0,
            original_artifacts=inventory, previews=previews, videos=videos, video_groups=video_groups, scope=SCOPE,
            compiler_cache="excluded; retained; never walked or embedded",
            events="original records retained; source-bound interval summaries only",
            prune_authorization=False,
            reason="staged workbook does not authorize pruning original paths; separate explicit source-root archive required"))
        (stage / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
        # Give every scalar in the bounded selected sample files an Excel row.
        # The staging tree excludes raw records and compiler caches completely.
        metric_limit = 3000
        for path in stage.iterdir():
            if path.suffix not in (".json", ".jsonl"):
                continue
            if path.stat().st_size > compact_results.MAX_TEXT_BYTES:
                raise ValueError("selected sample metadata exceeds compact collection limit: " + str(path))
            with path.open(encoding="utf-8") as stream:
                documents = [json.load(stream)] if path.suffix == ".json" else [json.loads(line) for line in stream]
            leaves = sum(1 for document in documents for _, value in _walk(document)
                         if not isinstance(value, (dict, list)))
            metric_limit = max(metric_limit, leaves + 1)
        # compact_results acquires this same Allocation lock via ancestor ledger
        # discovery. Release our handle before asking it to lock (flock on a
        # second descriptor would correctly reject our own existing lock).
        fcntl.flock(lock, fcntl.LOCK_UN)
        exported = compact_results.export(stage, destination, metrics_limit=metric_limit)
        if exported.get("status") != "verified":
            return dict(status="protected", reason="compact export could not acquire settled source locks",
                        export=exported, originals_deleted=0, scope=SCOPE)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verified = compact_results.verify(destination, check_sources=True)
        if _read(ledger)[1]["sha256"] != report["budget_source"]["sha256"]:
            raise ValueError("ledger changed during archive despite lock")
        result = dict(status="verified", workbook=exported["workbook"], manifest=exported["manifest"],
                      verification=verified, staging_root=str(stage), preview_count=len(previews),
                      video_count=len(videos), complete_video_pairs=sum(row["complete"] for row in video_groups),
                      retention_plan=str(stage / "raw-retention-plan.json"), originals_deleted=0,
                      original_prune_authorized=False, scope=SCOPE)
        atomic_json(output / "archive.json", result)
        return result
