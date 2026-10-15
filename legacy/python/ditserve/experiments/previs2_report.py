"""Offline, evidence-preserving reports for the separate PreViS run02 suite.

Importing this module does not import torch, inspect devices, or read arrays.
Raw verification is explicit and only runs after the supervisor has stopped.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from ..util import atomic_json, utc_now


POLICIES = ("fixed_state_qc", "joint_state_qc")
KINDS = ("correctness", "shape_probe", "quality_reference", "quality", "mechanism_case",
         "gate", "online_summary", "online_request", "confirmation", "environment", "failure",
         "initial_policy", "decision", "action_execution", "shape_probe_case", "stage_skip", "unfinished")
FAILED = {"failed", "error", "oom", "interrupted", "budget_exhausted", "timed_out", "aborted", "resource_rejected"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _json(value):
    return json.dumps(_safe(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def _inside(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _identity(row):
    identity = row.get("source_identity")
    return row.get("source_identity_sha256") or (identity.get("sha256") if isinstance(identity, dict) else None)


def _read(path, issues, required=False):
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    except (OSError, ValueError) as error:
        if required or path.exists():
            issues.append({"source": str(path), "reason": "unreadable_metadata", "detail": str(error)})
        return {}, None


def collect(suite_root):
    """Read only this suite's JSON; preserve unlisted, failed and malformed evidence."""
    root = Path(suite_root).resolve()
    issues, sources, records, attempts = [], [], [], []
    manifest, source = _read(root / "manifest.json", issues, required=True)
    if source:
        sources.append(source)
    owners = defaultdict(list)
    raw_attempts = manifest.get("records", [])
    if not isinstance(raw_attempts, list):
        issues.append({"reason": "manifest_records_not_list"})
        raw_attempts = []
    for index, raw_attempt in enumerate(raw_attempts):
        if not isinstance(raw_attempt, dict):
            issues.append({"reason": "invalid_attempt", "index": index})
            continue
        attempt = dict(raw_attempt, manifest_index=index)
        output = attempt.get("output")
        if isinstance(output, str):
            path = Path(output)
            path = path if path.is_absolute() else root / path
            if not _inside(path, root):
                issues.append({"reason": "attempt_output_outside_suite", "index": index})
            else:
                owners[path.resolve() / "records.jsonl"].append(attempt)
                outcome, origin = _read(path / "outcome.json", issues)
                if outcome:
                    attempt["outcome"] = outcome
                if origin:
                    sources.append(origin)
        attempts.append(attempt)
    paths = set(owners) | set(root.rglob("records.jsonl"))
    for path in sorted(paths):
        if not _inside(path, root):
            issues.append({"reason": "record_stream_outside_suite", "source": str(path)})
            continue
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("record stream changed while reading")
            lines = raw.decode("utf-8").splitlines()
        except (OSError, ValueError, UnicodeError) as error:
            issues.append({"source": str(path), "reason": "unreadable_records", "detail": str(error)})
            continue
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        owned = owners.get(path.resolve(), [])
        attempt = owned[0] if len(owned) == 1 else {}
        source_rows = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record is not an object")
                row = dict(row)
            except ValueError as error:
                issues.append({"source": str(path), "line": number,
                               "reason": "invalid_record", "detail": str(error)})
                continue
            row["declared_source_identity_sha256"] = _identity(row)
            row.setdefault("device", attempt.get("device"))
            row.setdefault("stage", attempt.get("stage"))
            row.setdefault("source_identity_sha256", _identity(attempt))
            row.update(source=str(path), source_line=number,
                       source_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                       attempt_status=attempt.get("status", "unlisted"))
            source_rows.append(row)
        environments = [r for r in source_rows if r.get("type") == "environment"]
        expected = _identity(attempt)
        environment = environments[0] if len(environments) == 1 else {}
        attempt_identity = attempt.get("source_identity", {})
        detailed_source_match = (isinstance(attempt_identity, dict) and bool(attempt_identity.get("source_hashes"))
                                 and environment.get("source_hashes") == attempt_identity["source_hashes"]
                                 and environment.get("versions") == attempt_identity.get("packages"))
        environment_source_match = (environment.get("declared_source_identity_sha256") == expected
                                    or detailed_source_match)
        if environment.get("declared_source_identity_sha256") not in (None, expected):
            environment_source_match = False
        if isinstance(attempt_identity, dict):
            for observed, expected_value in ((environment.get("source_hashes"), attempt_identity.get("source_hashes")),
                                             (environment.get("versions"), attempt_identity.get("packages"))):
                if observed is not None and observed != expected_value:
                    environment_source_match = False
        if environment.get("protocol") is not None and environment["protocol"] != manifest.get("protocol"):
            environment_source_match = False
        provenance = (len(owned) == 1 and isinstance(expected, str) and bool(expected)
                      and len(environments) == 1 and environment_source_match
                      and environment.get("status") == "completed")
        for row in source_rows:
            row["source_verified"] = bool(provenance and _identity(row) == expected
                                          and row.get("device") == attempt.get("device"))
            if _identity(attempt) and _identity(row) != _identity(attempt):
                issues.append({"source": str(path), "line": row["source_line"],
                               "reason": "record_source_mismatch"})
        records.extend(source_rows)
    gate, origin = _read(root / "gate.json", issues)
    if origin:
        sources.append(origin)
    return {"root": str(root), "manifest": manifest, "attempts": attempts,
            "records": records, "gate_document": gate, "sources": sources, "issues": issues}


def budget_audit(manifest):
    """A fresh 12h ledger is independent of run01 and must reconcile exactly."""
    problems = []
    used, limit = manifest.get("used_seconds"), manifest.get("limit_seconds")
    rows = manifest.get("records")
    if not isinstance(rows, list):
        rows = []
        problems.append("missing_attempt_ledger")
    charges, running = [], 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append("invalid_attempt_%d" % index)
            continue
        if row.get("status") == "running":
            running += 1
        if row.get("charge_closed") is not True:
            problems.append("unclosed_or_unknown_charge_%d" % index)
        charge = row.get("elapsed_seconds")
        if not finite(charge) or charge < 0:
            problems.append("unknown_or_invalid_charge_%d" % index)
        else:
            charges.append(charge)
        start, end = row.get("started_monotonic_s"), row.get("finished_monotonic_s")
        if start is not None or end is not None:
            if not finite(start) or not finite(end) or end < start:
                problems.append("invalid_monotonic_interval_%d" % index)
            elif finite(charge) and not math.isclose(end - start, charge, rel_tol=0, abs_tol=.01):
                problems.append("charge_interval_mismatch_%d" % index)
    if not finite(used) or used < 0:
        problems.append("invalid_used_seconds")
    if not finite(limit) or not 0 < limit <= 43200:
        problems.append("invalid_limit_or_more_than_12h")
    total = math.fsum(charges)
    if finite(used) and not math.isclose(used, total, rel_tol=0, abs_tol=1e-6):
        problems.append("used_does_not_equal_attempt_charges")
    if finite(used) and (used > 43200 + 1e-6 or (finite(limit) and used > limit + 1e-6)):
        problems.append("budget_exceeded")
    if running or manifest.get("status") == "running":
        problems.append("running_attempt_or_suite")
    return {"status": "passed" if not problems else "failed", "used_seconds": used,
            "limit_seconds": limit, "sum_charges_seconds": total, "running_attempts": running,
            "remaining_seconds": limit - used if finite(limit) and finite(used) else None,
            "issues": problems, "scope": "accounting_not_scientific_gate_or_complete_plan"}


def _quality(row):
    quality = row.get("quality")
    if not isinstance(quality, dict):
        return None
    value = quality.get("lpips_mean")
    if (quality.get("status") not in ("ok", "completed") or quality.get("finite") is not True
            or not finite(value) or value < 0):
        return None
    return value


def _shape(row):
    shape = row.get("shape")
    if isinstance(shape, dict):
        return {k: shape.get(k) for k in ("height", "width", "frames", "steps")}
    return {k: row.get(k) for k in ("height", "width", "frames", "steps")}


def _scope(row):
    if row.get("logicalcap") is True or row.get("logical_capacity") is True or row.get("logical_budget") is True:
        return "controlled"
    value = row.get("scope", row.get("resource_scope"))
    if value is None and row.get("logical_budget") is False and isinstance(row.get("natural_capacity"), dict):
        value = "natural"
    return value if value in ("natural", "controlled", "synthetic") else "unknown"


def coverage(records, attempts, config):
    groups = defaultdict(Counter)
    for row in records:
        groups[(row.get("type", "unknown"), row.get("device", "unknown"))][row.get("status", "unknown")] += 1
    expected = config.get("expected_cases", []) if isinstance(config, dict) else []
    matrix = []
    if isinstance(expected, list):
        for case in expected:
            if not isinstance(case, dict):
                continue
            matched = [r for r in records if r.get("case_id") == case.get("case_id")
                       and r.get("device") == case.get("device")
                       and (case.get("type") is None or r.get("type") == case["type"])]
            status = "not_run" if not matched else "duplicate" if len(matched) > 1 else matched[0].get("status", "unknown")
            matrix.append(dict(case, observed_status=status, record_count=len(matched)))
    return {"record_groups": [dict(type=k[0], device=k[1], statuses=dict(v)) for k, v in sorted(groups.items(), key=str)],
            "attempt_statuses": dict(Counter(r.get("status", "unknown") for r in attempts)),
            "planned_case_matrix": matrix, "planned_coverage_status": "known" if matrix else "unknown_no_explicit_case_plan",
            "unknown_record_types": sorted({r.get("type", "unknown") for r in records} - set(KINDS))}


def gates(records, document, manifest=None):
    manifest = manifest or {}
    expected_source = _identity(manifest.get("current_source_identity", {}))
    if expected_source is None and isinstance(manifest.get("current_source_identity"), dict):
        expected_source = manifest["current_source_identity"].get("sha256")
    document_valid = bool(document.get("schema_version", document.get("schema")) == "previs2-gate.v1"
                          and expected_source and document.get("source_identity_sha256") == expected_source
                          and manifest.get("protocol_sha256")
                          and document.get("protocol_sha256") == manifest["protocol_sha256"]
                          and document.get("status") in ("passed", "failed", "not_assessed"))
    rows = []
    for row in records:
        if row.get("type") != "gate":
            continue
        passed = row.get("passed")
        status = ("passed" if passed is True else "failed" if passed is False else "unknown")
        if row.get("status") not in ("completed", "passed", "failed") or not row.get("source_verified"):
            status = "unknown"
        rows.append({"gate_id": row.get("gate_id"), "scope": _scope(row), "status": status,
                     "passed": passed if type(passed) is bool else None, "reasons": row.get("reasons", row.get("reason")),
                     "source": row.get("source"), "source_line": row.get("source_line")})
    return {"recorded_checks": rows, "document": document,
            "document_identity_status": "matched" if document_valid else "unknown_or_mismatched",
            "natural_resource_gate": document.get("natural_resource_gate") if document_valid and type(document.get("natural_resource_gate")) is bool else None,
            "mechanism_gate": document.get("mechanism_gate") if document_valid and type(document.get("mechanism_gate")) is bool else None,
            "interpretation": "reported_gate_evidence_not_recomputed_or_inferred_from_export_success; controlled_capacity_is_not_natural_benefit"}


def _origin(row):
    return {k: row.get(k) for k in ("case_id", "device", "source_identity_sha256", "source", "source_line", "status")}


def measurement_points(records):
    points, capacity, funnel = [], [], []
    for row in records:
        if row.get("type") in ("quality", "mechanism_case", "confirmation", "shape_probe", "shape_probe_case"):
            events = row.get("events")
            event_sums = {}
            if isinstance(events, list):
                for event_type in ("step", "snapshot", "restore", "prepare", "decode"):
                    values = [e.get("elapsed_ms", e.get("ms")) for e in events
                              if isinstance(e, dict) and e.get("type", e.get("event", e.get("kind"))) == event_type]
                    event_sums[event_type] = sum(values) if values and all(finite(v) and v >= 0 for v in values) else None
            snapshot = row.get("snapshot_ms", event_sums.get("snapshot"))
            restore = row.get("restore_ms", event_sums.get("restore"))
            points.append(dict(_origin(row), type=row.get("type"), scope=_scope(row), shape=_shape(row),
                               measurement_scope=row.get("measurement_scope", "unknown"),
                               quality_scope=row.get("quality_scope", "unknown"),
                               cache_config=row.get("cache_config"), config=row.get("config"),
                               state_mode=row.get("state_mode", row.get("mode")),
                               pause_completed_steps=row.get("pause_completed_steps", row.get("pause_step")),
                               elapsed_ms=row.get("elapsed_ms"), lpips_mean=_quality(row),
                               snapshot_ms=snapshot, restore_ms=restore,
                               snapshot_plus_restore_ms=snapshot + restore if finite(snapshot) and finite(restore) else None,
                               state_bytes=row.get("state_bytes"), observed_event_sums_ms=event_sums))
            points[-1].update({k: row.get(k) for k in ("repetition", "prompt_id", "seed", "resume_critical_ms",
                              "estimated_resume_critical_ms", "cost_error_ms", "suffix_wall_ms", "urgent_e2e_ms",
                              "urgent_deadline_ms", "urgent_deadline_met", "capacity_reusable_verified",
                              "copy_complete_before_release", "next_request_allocation_and_completion_verified",
                              "selected_action", "actual_action")})
        for key in ("memory", "memory_before", "memory_after", "memory_baseline", "capacity", "natural_capacity", "snapshot_memory", "restore_memory"):
            value = row.get(key)
            if isinstance(value, dict):
                capacity.append(dict(_origin(row), scope=_scope(row), location=key, measurements=value,
                                     semantics="allocated_reserved_driver_free_are_distinct; no_capacity_benefit_inferred"))
        peaks = {k: row[k] for k in ("peak_allocated_bytes", "peak_reserved_bytes", "next_peak_bytes") if k in row}
        if peaks:
            capacity.append(dict(_origin(row), scope=_scope(row), location="peak_fields", measurements=peaks,
                                 semantics="recorded_peak_fields_not_driver_free_or_available_capacity"))
        if isinstance(row.get("funnel"), dict):
            funnel.append(dict(_origin(row), policy=row.get("policy"), pair_id=row.get("pair_id", row.get("trace_id")),
                               scope=_scope(row), counts=row["funnel"]))
        elif row.get("type") == "decision":
            candidates, excluded = row.get("candidates"), row.get("excluded")
            funnel.append(dict(_origin(row), policy=row.get("policy"), pair_id=row.get("pair_id", row.get("trace_id")),
                               scope=_scope(row), counts={"eligible_candidates": len(candidates) if isinstance(candidates, list) else None,
                                   "excluded": excluded, "fixed_selection": row.get("fixed_selection"),
                                   "joint_selection": row.get("joint_selection"), "selection_scope": row.get("selection_scope"),
                                   "executed_nonkeep": None,
                                   "note": "candidate_or_suggestion_is_not_executed_action"}))
    return points, capacity, funnel


def _inputs(requests):
    identities, seen = [], set()
    for request in requests:
        if not isinstance(request, dict):
            return None
        request_id = request.get("request_id")
        shape = _shape(request)
        if (not isinstance(request_id, str) or request_id in seen or not request.get("prompt_id")
                or type(request.get("seed")) is not int or not finite(request.get("arrival_ms"))
                or not finite(request.get("deadline_ms"))
                or any(type(v) is not int or v <= 0 for v in shape.values())):
            return None
        seen.add(request_id)
        identities.append({"request_id": request_id, "prompt_id": request["prompt_id"],
                           "prompt": request.get("prompt"), "seed": request["seed"], "shape": shape,
                           "arrival_ms": request["arrival_ms"], "deadline_ms": request["deadline_ms"]})
    return hashlib.sha256(_json(sorted(identities, key=lambda r: r["request_id"])).encode()).hexdigest() if identities else None


def online_metrics(row):
    """Recompute deadline+quality counts; unknowns remain in offered denominator."""
    requests = row.get("requests")
    requests = requests if isinstance(requests, list) else []
    nested = row.get("summary") if isinstance(row.get("summary"), dict) else {}
    offered = row.get("offered", nested.get("offered"))
    epsilon = row.get("epsilon", row.get("quality_budget"))
    horizon = row.get("horizon_ms", nested.get("horizon_ms"))
    reasons = []
    not_arrived = sum(isinstance(r, dict) and r.get("status") == "not_arrived" for r in requests)
    arrived_records = len(requests) - not_arrived
    if type(offered) is not int or offered < arrived_records or offered < 0:
        reasons.append("offered_missing_or_inconsistent")
        offered = None
    if not finite(epsilon) or epsilon < 0:
        reasons.append("unknown_epsilon")
    if not finite(horizon) or horizon <= 0:
        reasons.append("unknown_horizon")
    counts = Counter(completed=0, failed=0, not_arrived=0, status_unknown=0, time_unknown=0,
                     quality_unknown=0, quality_failed=0, deadline_on_time=0,
                     quality_qualified_on_time=0, under_30s=0)
    latencies = []
    for request in requests:
        if not isinstance(request, dict):
            counts["status_unknown"] += 1
            continue
        if request.get("status") == "not_arrived":
            counts["not_arrived"] += 1
            continue
        if request.get("status") != "completed":
            counts["failed" if request.get("status") in FAILED else "status_unknown"] += 1
            continue
        counts["completed"] += 1
        arrival, deadline, finish = (request.get(k) for k in ("arrival_ms", "deadline_ms", "finish_ms"))
        timing = (all(finite(v) for v in (arrival, deadline, finish)) and 0 <= arrival <= finish
                  and deadline >= arrival and finite(horizon) and finish <= horizon + 1e-6)
        if not timing:
            counts["time_unknown"] += 1
        else:
            latencies.append(finish - arrival)
            counts["under_30s"] += int(finish - arrival < 30000)
            counts["deadline_on_time"] += int(finish <= deadline)
        loss = _quality(request)
        if loss is None:
            counts["quality_unknown"] += 1
        elif finite(epsilon):
            counts["quality_failed"] += int(loss > epsilon)
            counts["quality_qualified_on_time"] += int(timing and finish <= deadline and loss <= epsilon)
    missing = offered - arrived_records if offered is not None else None
    if missing:
        reasons.append("missing_request_records")
    if counts["status_unknown"] or counts["time_unknown"] or counts["quality_unknown"]:
        reasons.append("unknown_request_status_timing_or_quality")
    identity = _inputs(requests)
    if identity is None:
        reasons.append("invalid_or_duplicate_request_inputs")
    if row.get("status") != "completed":
        reasons.append("trace_not_completed")
    rate = counts["quality_qualified_on_time"] * 1000 / horizon if finite(horizon) and horizon > 0 else None
    return dict(_origin(row), pair_id=row.get("pair_id", row.get("trace_id")), policy=row.get("policy"),
                load=row.get("load"), pattern=row.get("pattern"), trace_seed=row.get("trace_seed"),
                epsilon=epsilon, scope=_scope(row), source_verified=row.get("source_verified", False),
                offered=offered, described_scheduled_requests=len(requests), missing_request_records=missing,
                horizon_ms=horizon, **dict(counts),
                goodput_rps=rate if not reasons else None, goodput_lower_bound_rps=rate,
                metrics_status="complete" if not reasons else "unknown", reasons=reasons,
                request_identity_sha256=identity,
                completed_e2e_ms={"n": len(latencies), "median": statistics.median(latencies) if latencies else None})


def paired_goodput(rows):
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("policy") not in POLICIES:
            continue
        key = tuple(row.get(k) for k in ("device", "source_identity_sha256", "pair_id", "load", "pattern", "trace_seed", "epsilon", "scope"))
        buckets[key][row["policy"]].append(row)
    pairs = []
    for key, arms in sorted(buckets.items(), key=str):
        result = dict(zip(("device", "source_identity_sha256", "pair_id", "load", "pattern", "trace_seed", "epsilon", "scope"), key))
        reasons = []
        if any(len(arms.get(policy, [])) != 1 for policy in POLICIES):
            reasons.append("missing_or_duplicate_arm")
        values = [arms[p][0] for p in POLICIES] if not reasons else []
        if values:
            if any(r["metrics_status"] != "complete" for r in values):
                reasons.append("incomplete_request_metrics")
            if any(not r["source_verified"] for r in values) or not key[1]:
                reasons.append("unverified_source")
            if values[0]["request_identity_sha256"] != values[1]["request_identity_sha256"]:
                reasons.append("request_input_mismatch")
            if not key[2] or key[5] is None or key[7] == "unknown":
                reasons.append("missing_pair_context_or_resource_scope")
        result.update(status="matched" if not reasons else "unavailable", reasons=reasons)
        if not reasons:
            baseline, candidate = [r["goodput_rps"] for r in values]
            result.update(baseline_rps=baseline, candidate_rps=candidate,
                          difference_rps=candidate - baseline,
                          relative_change=candidate / baseline - 1 if baseline > 0 else None,
                          relative_status="observed" if baseline > 0 else "unavailable_zero_baseline",
                          baseline_horizon_ms=values[0]["horizon_ms"], candidate_horizon_ms=values[1]["horizon_ms"])
        pairs.append(result)
    groups = defaultdict(list)
    for pair in pairs:
        if pair["status"] == "matched":
            groups[tuple(pair[k] for k in ("device", "source_identity_sha256", "load", "pattern", "epsilon", "scope"))].append(pair)
    summaries = []
    for key, items in sorted(groups.items(), key=str):
        baseline, candidate = (statistics.mean(r[k] for r in items) for k in ("baseline_rps", "candidate_rps"))
        summaries.append(dict(zip(("device", "source_identity_sha256", "load", "pattern", "epsilon", "scope"), key),
                              paired_traces=len(items), baseline_mean_rps=baseline, candidate_mean_rps=candidate,
                              relative_change=candidate / baseline - 1 if baseline > 0 else None,
                              estimator="ratio_of_paired_trace_mean_goodputs_minus_one", interval=None,
                              uncertainty="descriptive_only; no_request_or_frame_pseudoreplication"))
    return {"baseline": POLICIES[0], "candidate": POLICIES[1], "pairs": pairs, "groups": summaries}


def summarize(evidence):
    manifest, records = evidence["manifest"], evidence["records"]
    config = manifest.get("protocol", manifest.get("config", {}))
    points, capacity, funnel = measurement_points(records)
    online = [online_metrics(row) for row in records if row.get("type") == "online_summary"]
    return {"schema_version": "previs2.report.v1", "created_at": utc_now(), "suite_root": evidence["root"],
            "suite_status": manifest.get("status", "unknown"), "configuration": config,
            "budget": budget_audit(manifest), "coverage": coverage(records, evidence["attempts"], config),
            "gates": gates(records, evidence["gate_document"], manifest), "measurement_points": points,
            "capacity_observations": capacity, "trigger_funnel": funnel, "online": online,
            "paired_goodput": paired_goodput(online), "attempts": evidence["attempts"],
            "sources": evidence["sources"], "issues": evidence["issues"],
            "failures": [dict(_origin(r), type=r.get("type"), reason=r.get("reason"), error=r.get("error"))
                         for r in records if r.get("type") == "failure" or r.get("status") in FAILED],
            "raw_verification": {"status": "not_requested", "scope": "no_arrays_read"},
            "limitations": ["自然资源gate失败或未评估时，受控logical capacity实验不能算自然服务收益。",
                            "缺少合法候选不是keep击败hot/cold；未知计数不补成零。",
                            "质量门槛仅使用有限样本mean LPIPS；开发最大值不是未见输入保证。",
                            "有限trace drain goodput不是稳态吞吐，排队E2E不是单请求服务耗时。",
                            "缺失、失败、未运行及重复尝试保留；导出成功不代表科学gate通过。"]}


def _fmt(value):
    return "未知" if value is None else str(value)


def markdown(report):
    b, g = report["budget"], report["gates"]
    lines = ["# PreViS run02 实验报告", "", "本报告从本轮原始记录、outcome与manifest离线生成；不改写原始记录或run01。", "",
             f"实验状态：`{report['suite_status']}`。预算：`{b['status']}`，已收费{_fmt(b['used_seconds'])}秒 / 上限{_fmt(b['limit_seconds'])}秒。",
             f"记录的机制gate：{_fmt(g['mechanism_gate'])}；自然资源gate：{_fmt(g['natural_resource_gate'])}。这些是实验记录声明，不能由导出成功推定通过。", "",
             "## 配置与覆盖", "", "```json", _json(report["configuration"]), "```", "",
             "| 记录类型 | 设备 | 状态计数 |", "| --- | --- | --- |"]
    for row in report["coverage"]["record_groups"]:
        lines.append(f"| {row['type']} | {_fmt(row['device'])} | {_json(row['statuses'])} |")
    lines += ["", "| attempt阶段 | 设备 | 状态 | 记账秒数 | 未测/失败原因 |", "| --- | --- | --- | ---: | --- |"]
    for row in report["attempts"]:
        reason = str(row.get("reason", row.get("error", ""))).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {_fmt(row.get('stage'))} | {_fmt(row.get('device'))} | {_fmt(row.get('status'))} | {_fmt(row.get('elapsed_seconds'))} | {reason} |")
    lines += ["", "计划覆盖：" + report["coverage"]["planned_coverage_status"] + "。明确计划case及未运行项见JSON/工作簿；无计划时不推算完成率。", "",
              "## 同trace配对goodput", "", "仅比较唯一、同来源/设备/输入/质量阈值/资源范围的fixed_state_qc与joint_state_qc；失败请求保留在offered中，未知质量不算通过。", "",
              "| 设备 | 范围 | 负载 / 模式 | 配对trace数 | fixed / joint（请求/s） | 均值比减1 |", "| --- | --- | --- | ---: | --- | --- |"]
    for row in report["paired_goodput"]["groups"]:
        lines.append(f"| {row['device']} | {row['scope']} | {row['load']} / {row['pattern']} | {row['paired_traces']} | {row['baseline_mean_rps']:.6f} / {row['candidate_mean_rps']:.6f} | {_fmt(row['relative_change'])} |")
    if not report["paired_goodput"]["groups"]:
        lines += ["", "没有满足配对条件的已知结果；这不表示两策略效果相同。"]
    lines += ["", "## 触发、容量与恢复成本", "", f"记录了{len(report['trigger_funnel'])}项触发漏斗、{len(report['capacity_observations'])}项容量观测；未知计数保持未知。", "",
              "| case / trace | 策略 | 资源范围 | 漏斗计数与未触发原因 |", "| --- | --- | --- | --- |"]
    for row in report["trigger_funnel"]:
        counts = _json(row["counts"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {_fmt(row['pair_id'] or row['case_id'])} | {_fmt(row['policy'])} | {row['scope']} | {counts} |")
    if not report["trigger_funnel"]:
        lines += ["", "没有触发漏斗记录，不能报告零触发率或自然资源收益。"]
    lines += ["", "allocated、reserved、driver free分别报告；同步snapshot＋restore成本不等于首步、整段后缀或E2E。fixed/resume质量点及natural/controlled范围不混合。", "",
              "## 失败、原始验收与预算", "", f"失败记录{len(report['failures'])}项；读取/结构问题{len(report['issues'])}项。Raw验收：`{report['raw_verification']['status']}`。",
              "预算逐attempt累计并检查收费总和、12小时上限与未结算running；raw验收只验证本轮已保存数组，缺失配对保持不可用。", "",
              "## 解释限制", ""]
    lines += ["- " + item for item in report["limitations"]]
    lines += ["", "完整配置、原始来源哈希、计划缺测、未知分母、gate原因、容量与计时点见 [report_metadata.json](report_metadata.json)。", ""]
    return "\n".join(lines)


def _workbook(root, report):
    from openpyxl import Workbook
    book = Workbook()
    book.remove(book.active)
    tables = {"coverage": report["coverage"]["record_groups"], "planned_cases": report["coverage"]["planned_case_matrix"],
              "attempts": report["attempts"], "online": report["online"], "pairs": report["paired_goodput"]["pairs"],
              "quality_state_cost": report["measurement_points"], "capacity": report["capacity_observations"],
              "trigger_funnel": report["trigger_funnel"], "gates": report["gates"]["recorded_checks"],
              "failures": report["failures"], "budget": [report["budget"]]}
    for name, rows in tables.items():
        sheet = book.create_sheet(name)
        keys = list(dict.fromkeys(k for row in rows for k in row))
        sheet.append(keys or ["no_observations"])
        for row in rows:
            cells = []
            for key in keys:
                value = _safe(row.get(key))
                value = _json(value) if isinstance(value, (dict, list)) else value
                if isinstance(value, str):
                    value = value[:32000]
                    if value.startswith(("=", "+", "-", "@")):
                        value = "'" + value
                cells.append(value)
            sheet.append(cells)
        sheet.freeze_panes = "A2"
        if rows:
            sheet.auto_filter.ref = sheet.dimensions
    temporary = root / "results.tmp.xlsx"
    book.save(temporary)
    temporary.replace(root / "results.xlsx")
    return {"status": "completed", "path": str(root / "results.xlsx"), "sheets": list(tables)}


def _figures(root, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = root / "figures"
    target.mkdir(exist_ok=True)
    artifacts = []
    specifications = (("quality_time", "elapsed_ms", "lpips_mean", "Generation elapsed (ms)", "Mean LPIPS"),
                      ("state_cost_progress", "pause_completed_steps", "snapshot_plus_restore_ms", "Completed solver updates", "Snapshot + restore (ms)"))
    for name, xkey, ykey, xlabel, ylabel in specifications:
        grouped = defaultdict(list)
        for point in report["measurement_points"]:
            if point["status"] == "completed" and finite(point.get(xkey)) and finite(point.get(ykey)):
                key = _json({k: point.get(k) for k in ("device", "source_identity_sha256", "shape", "measurement_scope", "scope", "cache_config")})
                grouped[key].append(point)
        for index, (key, points) in enumerate(sorted(grouped.items())):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for mode in sorted({str(p.get("state_mode") or "fixed") for p in points}):
                selected = [p for p in points if str(p.get("state_mode") or "fixed") == mode]
                ax.scatter([p[xkey] for p in selected], [p[ykey] for p in selected], label=mode)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.grid(alpha=.25)
            ax.legend()
            fig.tight_layout()
            path = target / (name + "_%02d.png" % (index + 1))
            fig.savefig(path, dpi=160)
            plt.close(fig)
            artifacts.append({"path": str(path), "kind": name, "group": json.loads(key), "n": len(points)})
    groups = report["paired_goodput"]["groups"]
    if groups:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        valid = [(i, row) for i, row in enumerate(groups) if finite(row["relative_change"])]
        ax.scatter([i for i, _ in valid], [100 * r["relative_change"] for _, r in valid])
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(range(len(groups)), [f"{r['device']} {r['scope']}\n{r['load']} {r['pattern']} n={r['paired_traces']}" for r in groups])
        ax.set_ylabel("Joint / fixed paired mean goodput - 1 (%)")
        fig.tight_layout()
        path = target / "paired_goodput.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        artifacts.append({"path": str(path), "kind": "paired_goodput", "scope": "descriptive_no_inferred_intervals"})
    return {"status": "completed", "artifacts": artifacts, "missing_data_not_plotted_as_zero": True}


def _previews(root, evidence, verification):
    """Fixed first-two held-out prompts/first seed, never rank by measured quality."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .previs_verify import _resolve

    config = evidence["manifest"].get("protocol", {})
    prompts = [p for p in config.get("prompts", []) if p.get("split") == "test"][:2]
    seeds = config.get("test_seeds", [])
    target = root / "samples"
    target.mkdir(exist_ok=True)
    result = {"status": "completed", "selection": "first_two_test_prompts_in_protocol_order_first_test_seed_selected_long_policy",
              "quality_ranking": False, "source": "quality fixed/test rows with exact preregistered case_id",
              "groups": [], "artifacts": [], "issues": [],
              "semantics": "clipped_RGB_preview_only; metrics_use_original_arrays; no_fps_or_duration_claim"}
    verified = {str(Path(a["path"]).resolve()): a for a in verification.get("arrays", []) if a.get("status") == "verified"}
    if len(prompts) != 2 or not seeds:
        result.update(status="partial")
        result["issues"].append("protocol_missing_two_test_prompts_or_test_seed")
    for prompt in prompts:
        seed = seeds[0] if seeds else None
        case_id = "test-long-%s-%s-selected" % (prompt.get("id"), seed)
        matches = [r for r in evidence["records"] if r.get("type") == "quality" and r.get("case_id") == case_id
                   and r.get("measurement_scope") == "fixed" and r.get("split") == "test"
                   and r.get("prompt_id") == prompt.get("id") and r.get("seed") == seed
                   and r.get("status") == "completed" and r.get("source_verified") is True]
        entry = {"case_id": case_id, "prompt_id": prompt.get("id"), "seed": seed, "status": "missing"}
        result["groups"].append(entry)
        if len(matches) != 1:
            entry.update(status="missing" if not matches else "ambiguous", reason="no_unique_completed_fixed_selected_row")
            result["status"] = "partial"
            continue
        row, arrays = matches[0], []
        try:
            for key in ("reference", "output"):
                descriptor = row.get(key)
                if not isinstance(descriptor, dict):
                    raise ValueError("missing_" + key + "_descriptor")
                path = _resolve(descriptor.get("path"), root, Path(row["source"]))
                integrity = verified.get(str(path.resolve()))
                if integrity is None or any(integrity["actual"].get(k) != descriptor.get(k) for k in ("shape", "dtype", "finite", "sha256")):
                    raise ValueError("preview_array_not_independently_verified")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.ndim != 5 or array.shape[0] != 1 or array.shape[-1] != 3:
                    raise ValueError("unsupported_video_array_shape")
                arrays.append(array)
            fig, axes = plt.subplots(2, 3, figsize=(9, 5))
            try:
                for y, array in enumerate(arrays):
                    frames = (0, (array.shape[1] - 1) // 2, array.shape[1] - 1)
                    for x, frame in enumerate(frames):
                        axes[y, x].imshow(np.clip(array[0, frame], 0, 1))
                        axes[y, x].set_title(("reference" if y == 0 else "selected") + " frame %d" % frame)
                        axes[y, x].axis("off")
                fig.suptitle(case_id)
                fig.tight_layout()
                path = target / (case_id + ".png")
                fig.savefig(path, dpi=130)
            finally:
                plt.close(fig)
            entry.update(status="completed", path=str(path), source=row["source"], source_line=row["source_line"],
                         source_identity_sha256=row["source_identity_sha256"], quality=row.get("quality"))
            result["artifacts"].append({"path": str(path), "case_id": case_id, "format": "png"})
        except (OSError, ValueError, KeyError) as error:
            entry.update(status="failed", reason=str(error))
            result["status"] = "partial"
            result["issues"].append({"case_id": case_id, "reason": str(error)})
        finally:
            arrays.clear()
    atomic_json(target / "manifest.json", _safe(result))
    return result


def build_report(suite_root, *, verify_raw=False, make_figures=True, make_workbook=True, make_previews=False):
    root = Path(suite_root).resolve()
    if not root.is_dir():
        raise ValueError("suite directory does not exist")
    evidence = collect(root)
    if evidence["manifest"].get("schema_version") != "ditserve.previs2.suite.v1":
        raise ValueError("only a separate ditserve.previs2.suite.v1 suite may be exported; run01 is never rewritten")
    report = summarize(evidence)
    verification = None
    if make_previews and not verify_raw:
        raise ValueError("previews require --verify-raw in the same final export")
    if verify_raw:
        if report["budget"]["running_attempts"] or evidence["manifest"].get("status") == "running":
            raise ValueError("raw verification requires all supervised GPU attempts to finish")
        from .previs_verify import verify_suite
        verification = verify_suite(root)
        report["raw_verification"] = {k: verification.get(k) for k in ("status", "summary", "issues", "scope", "sources")}
    report["exports"] = {}
    for name, enabled, export in (("workbook", make_workbook, _workbook), ("figures", make_figures, _figures)):
        if not enabled:
            report["exports"][name] = {"status": "not_requested"}
            continue
        try:
            report["exports"][name] = export(root, report)
        except Exception as error:
            report["exports"][name] = {"status": "failed", "error": str(error)}
    if make_previews:
        if verification["status"] not in ("verified", "verified_with_unavailable_pairs"):
            report["exports"]["previews"] = {"status": "failed", "reason": "raw_verification_not_passed"}
        else:
            try:
                report["exports"]["previews"] = _previews(root, evidence, verification)
            except Exception as error:
                report["exports"]["previews"] = {"status": "failed", "error": str(error)}
    else:
        report["exports"]["previews"] = {"status": "not_requested"}
    report["export_status"] = "failed" if any(v["status"] == "failed" for v in report["exports"].values()) else "completed"
    temporary = root / "REPORT.md.tmp"
    temporary.write_text(markdown(report), encoding="utf-8")
    temporary.replace(root / "REPORT.md")
    atomic_json(root / "report_metadata.json", _safe(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_root", type=Path)
    parser.add_argument("--verify-raw", action="store_true", help="after GPU exit only; verify this suite's arrays")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-workbook", action="store_true")
    parser.add_argument("--previews", action="store_true", help="fixed held-out examples; requires --verify-raw")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.suite_root, verify_raw=args.verify_raw,
                              make_figures=not args.no_figures, make_workbook=not args.no_workbook,
                              make_previews=args.previews)
    except (OSError, ValueError) as error:
        print(_json({"export_status": "failed", "error": str(error)}))
        return 1
    print(_json({"export_status": report["export_status"], "suite_status": report["suite_status"],
                 "budget": report["budget"]["status"], "raw_verification": report["raw_verification"]["status"],
                 "report": str(args.suite_root / "REPORT.md")}))
    return int(report["export_status"] != "completed" or report["budget"]["status"] != "passed"
               or report["raw_verification"]["status"] == "failed" or bool(report["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
