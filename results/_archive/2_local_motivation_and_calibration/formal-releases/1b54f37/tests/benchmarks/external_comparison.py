"""Export an existing frozen study and account external results without resampling.

This module neither submits requests nor fabricates measurements. The result
schema uses one client monotonic origin and an explicit decoded-output boundary.
All frozen offered requests remain in every SLO denominator, including missing
or invalid results. These checks alone never qualify a full-system comparison.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import re

WORKLOAD_SCHEMA = "ditserve.external-workload.v1"
RESULT_SCHEMA = "ditserve.external-results.v1"
BOUNDARY = "decoded_output_ready_observed_by_common_client"
TERMINAL = {"completed", "failed", "rejected", "cancelled", "timed_out", "not_executed", "incomplete"}


def canonical(value, *, ascii=False):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=ascii,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(raw):
    return hashlib.sha256(raw).hexdigest()


def hex64(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def read_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def load_bound(path, expected):
    raw = Path(path).read_bytes()
    if not hex64(expected) or file_sha(raw) != expected:
        raise ValueError("input raw SHA256 differs: " + str(path))
    return read_json(raw), {"path": str(path), "raw_sha256": expected}


def finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    if value < 0 or positive and value == 0:
        raise ValueError(name + " must be " + ("positive" if positive else "nonnegative"))
    return value


def seal(value):
    value = copy.deepcopy(value)
    value.pop("sha256", None)
    value["sha256"] = canonical(value)
    return value


def verify_seal(value):
    if not hex64(value.get("sha256")) or canonical({k:v for k,v in value.items() if k != "sha256"}) != value["sha256"]:
        raise ValueError("sealed artifact SHA256 differs")


def validate_trace(workload, provenance):
    manifest = provenance.get("manifest", {})
    if manifest.get("schema") != "ditserve.benchmark.mixed-workload.v1":
        raise ValueError("unsupported frozen trace manifest")
    # Historical trace hashing uses ensure_ascii=True, unlike other project seals.
    digest = canonical(workload, ascii=True)
    if provenance.get("sha256") != digest or manifest.get("workload_sha256") != digest:
        raise ValueError("frozen workload hash differs")
    if not isinstance(workload, list) or not workload:
        raise ValueError("empty frozen workload")
    seen = set()
    previous = -1
    for row in workload:
        rid = row.get("request_id")
        spec = row.get("spec", {})
        if not isinstance(rid, str) or not rid or rid in seen or spec.get("request_id") != rid:
            raise ValueError("duplicate or inconsistent request identity")
        seen.add(rid)
        arrival = finite(row.get("arrival_ms"), "arrival_ms")
        deadline = finite(row.get("deadline_ms"), "deadline_ms")
        slo = finite(row.get("slo_ms"), "slo_ms", positive=True)
        finite(row.get("service_ms"), "service_ms", positive=True)
        if arrival < previous or deadline < arrival or not math.isclose(deadline, arrival + slo, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("arrival order or absolute deadline differs")
        previous = arrival
        if not isinstance(spec.get("prompt"), str) or not spec["prompt"].strip() or row.get("dataset_sample_id") is None:
            raise ValueError("missing prompt or dataset sample provenance")
        if type(spec.get("seed")) is not int or not 0 <= spec["seed"] < 2**63:
            raise ValueError("invalid frozen generation seed")
        if row.get("trace_seed") != manifest.get("seed"):
            raise ValueError("request trace seed differs from manifest")
    if dict(Counter(r["kind"] for r in workload)) != manifest.get("counts"):
        raise ValueError("manifest modality counts differ")
    if dict(Counter(r["class"] for r in workload)) != manifest.get("class_counts"):
        raise ValueError("manifest shape counts differ")
    finite(manifest.get("total_duration_ms"), "trace total_duration_ms")
    if type(manifest.get("slots")) is not int or manifest["slots"] < 1:
        raise ValueError("invalid trace allocation")


def export_study(study, protocol, contract, models, *, bindings, phase_id=None):
    if (protocol.get("schema") != "ditserve.formal-protocol.v1" or protocol.get("id") != "formal-mixed-v1"
            or contract.get("schema") != "ditserve.external-reproduction-contract.v1"
            or contract.get("id") != "external-reproduction-v1"
            or contract.get("formal_protocol_sha256") != canonical(protocol)):
        raise ValueError("formal/external protocol identity differs")
    if set(bindings) != {"study", "protocol", "contract", "models"} or any(
            not hex64(entry.get("raw_sha256")) for entry in bindings.values()):
        raise ValueError("externally pinned raw input bindings are required")
    phases = study.get("phases", [])
    chosen = [p for p in phases if phase_id is None or p.get("id") == phase_id]
    if len(chosen) != 1:
        raise ValueError("select exactly one frozen study phase explicitly")
    phase = chosen[0]
    matrix = phase.get("matrix", {})
    if (set(matrix) - {"/policy", "/workload", "/trace_provenance"}
            or len(matrix.get("/workload", [])) != 1 or len(matrix.get("/trace_provenance", [])) != 1):
        raise ValueError("phase must contain exactly one workload/provenance and no request-mutating matrix")
    workload, provenance = matrix["/workload"][0], matrix["/trace_provenance"][0]
    validate_trace(workload, provenance)
    meta = study.get("benchmark", {})
    identity = meta.get("formal_protocol", {})
    if (identity.get("id") != protocol["id"] or identity.get("sha256") != canonical(protocol)
            or identity.get("schema") != protocol["schema"] or meta.get("trace") != provenance):
        raise ValueError("study formal protocol or trace binding differs")
    if meta.get("model_manifests") != models:
        raise ValueError("study frozen model manifest mapping differs")
    shape_ids = identity.get("shape_ids")
    if not isinstance(shape_ids, list) or len(shape_ids) != len(set(shape_ids)):
        raise ValueError("invalid declared shape coverage")
    shape_models = {protocol["shapes"][sid]["model"] for sid in shape_ids}
    cells = [c for c in contract["comparison_cells"] if set(c["models"]) == shape_models]
    if len(cells) != 1 or len(shape_models) != 2 or "sd35" not in shape_models:
        raise ValueError("requires one pairwise Wan/SD3.5 cell; three-model mixture is not frozen")
    for model in shape_models:
        binding = models.get(model, {})
        if not isinstance(binding.get("path"), str) or not hex64(binding.get("sha256")):
            raise ValueError("invalid model manifest binding: " + model)
    requests = []
    for index, row in enumerate(workload):
        sid, spec = row["class"], row["spec"]
        if sid not in shape_ids:
            raise ValueError("request shape outside declared study coverage")
        shape = protocol["shapes"][sid]
        if row["kind"] != shape["kind"] or any(spec.get(k) != shape[k] for k in ("height", "width", "frames", "steps", "guidance_scale")):
            raise ValueError("request shape differs from formal protocol")
        if not isinstance(spec.get("negative_prompt"), str):
            raise ValueError("negative prompt must be explicit; no external default may replace it")
        expected = {k: shape[k] for k in ("height", "width", "frames", "steps", "guidance_scale")}
        if shape["kind"] == "video":
            expected.update(cache=False, threshold=0.0, max_continuous=1, negative_prompt="")
        else:
            expected.update(batch_size=1, max_sequence_length=256, negative_prompt="blurry, low quality")
        if {k:v for k,v in spec.items() if k not in ("prompt", "seed", "request_id")} != expected:
            raise ValueError("frozen generation settings differ from the formal shape contract")
        model = shape["model"]
        requests.append({"request_id": row["request_id"], "source_index": index,
            "source_request_sha256": canonical(row), "model": model, "shape_id": sid,
            "kind": row["kind"], "arrival_ms": row["arrival_ms"], "deadline_ms": row["deadline_ms"],
            "slo_ms": row["slo_ms"], "spec": copy.deepcopy(spec), "spec_sha256": canonical(spec),
            "prompt_utf8_sha256": file_sha(spec["prompt"].encode()),
            "negative_prompt_utf8_sha256": file_sha(spec["negative_prompt"].encode()),
            "model_manifest": copy.deepcopy(models[model])})
    manifest = provenance["manifest"]
    # One shared finite horizon, chosen from frozen inputs, includes planned drain.
    horizon = max(manifest["total_duration_ms"], max(r["deadline_ms"] for r in workload))
    finite(horizon, "common measurement horizon", positive=True)
    patterns = {item["name"] for item in protocol["patterns"]}
    windows = manifest.get("windows", [])
    window_controls_match = bool(windows) and all(
        isinstance(window, dict) and window.get("pattern") in patterns
        and type(window.get("load")) in (int, float) and window["load"] in protocol["loads"]
        and type(window.get("video_fraction")) in (int, float)
        and window["video_fraction"] in protocol["video_fractions"]
        for window in windows)
    eligible = (window_controls_match and identity.get("split") == "test" and identity.get("role") == "serving"
                and len(workload) == protocol["requests_per_cell"]["test"]
                and manifest["seed"] in protocol["seeds"]["test"]
                and all(protocol["shapes"][sid]["scope"] == "formal" for sid in shape_ids)
                and set(manifest["class_counts"]) == set(shape_ids))
    return seal({"schema": WORKLOAD_SCHEMA, "source_bindings": copy.deepcopy(bindings),
        "source_study_canonical_sha256": canonical(study), "experiment": study.get("experiment"),
        "selected_phase": phase["id"], "study_phase_count": len(phases), "comparison_cell": copy.deepcopy(cells[0]),
        "formal_protocol": copy.deepcopy(identity), "formal_protocol_document": copy.deepcopy(protocol),
        "external_contract_sha256": canonical(contract), "model_manifests": copy.deepcopy(models),
        "reference_worker_configs": copy.deepcopy(study.get("workers", {})),
        "workload": copy.deepcopy(workload), "trace_provenance": copy.deepcopy(provenance),
        "requests": requests, "offered_requests": len(workload), "declared_formal_trace_candidate": eligible,
        "formal_candidate_scope": "declared study identity/count/seed/shape and window pattern/load/video-fraction checks only; calibration, dataset split/contamination, hardware and source execution are not independently audited here",
        "clock_contract": {"reference": "client_origin_ms", "completion_boundary": BOUNDARY,
            "arrival_rule": "open-loop absolute offsets from one client monotonic origin; never reset deadline on dispatch",
            "measurement_horizon_ms": horizon,
            "horizon_rule": "max(frozen trace duration, last frozen absolute deadline); late outputs retain latency but do not extend the throughput denominator",
            "wire_serialization_rule": "record client_received_ms separately; do not subtract it from arrival-to-ready latency"},
        "batching": {"outputs_per_request": 1, "native_dynamic_batching": "allowed; calibration-only tuning",
                     "legacy_image_batch_limit_is_not_an_external_scheduler_limit": True},
        "qualification_eligible": False, "allows": [], "execution_performed": False})


def verify_export(export):
    verify_seal(export)
    if export.get("schema") != WORKLOAD_SCHEMA:
        raise ValueError("unsupported exported workload")
    validate_trace(export["workload"], export["trace_provenance"])
    if export["offered_requests"] != len(export["workload"]) or len(export["requests"]) != len(export["workload"]):
        raise ValueError("export lost offered requests")
    for index, (row, request) in enumerate(zip(export["workload"], export["requests"])):
        if (request["source_index"] != index or request["source_request_sha256"] != canonical(row)
                or request["spec"] != row["spec"] or request["spec_sha256"] != canonical(row["spec"])
                or any(request[k] != row[k] for k in ("request_id", "arrival_ms", "deadline_ms", "slo_ms", "kind"))):
            raise ValueError("exported request changed frozen input")
        sid = request["shape_id"]
        if sid != row["class"] or request["model"] != export["formal_protocol_document"]["shapes"][sid]["model"]:
            raise ValueError("export changed model or shape identity")
        if request["model_manifest"] != export["model_manifests"][request["model"]]:
            raise ValueError("export changed model manifest")
    expected_horizon = max(export["trace_provenance"]["manifest"]["total_duration_ms"],
                           max(r["deadline_ms"] for r in export["workload"]))
    if (export["clock_contract"]["measurement_horizon_ms"] != expected_horizon
            or export["clock_contract"]["completion_boundary"] != BOUNDARY):
        raise ValueError("export clock boundary/horizon differs")


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered)-1)*q
    lo, hi = math.floor(location), math.ceil(location)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(location-lo)


def group_metrics(rows, horizon):
    counts = Counter(r["status"] for r in rows)
    completed = [r for r in rows if r["status"] == "completed"]
    on_time = sum(r["slo_met"] for r in completed)
    completed_window = sum(r["output_ready_ms"] <= horizon for r in completed)
    latency = [r["latency_ms"] for r in completed]
    return {"offered": len(rows), "status_counts": dict(counts), "completed": len(completed),
        "slo_met": on_time, "slo_missed": len(rows)-on_time,
        "slo_attainment": on_time/len(rows) if rows else None,
        "completed_in_window": completed_window, "measurement_horizon_ms": horizon,
        "throughput_rps": completed_window*1000/horizon, "goodput_rps": on_time*1000/horizon,
        "latency_completed_only_ms": {"count": len(latency), "p50": percentile(latency,.5),
            "p95": percentile(latency,.95), "p99": percentile(latency,.99)},
        "dispatch_lag_ms": {"count": len(completed), "p50": percentile([r["dispatch_lag_ms"] for r in completed], .5)},
        "wire_after_ready_ms": {"count": sum("wire_after_ready_ms" in r for r in completed),
            "p50": percentile([r["wire_after_ready_ms"] for r in completed if "wire_after_ready_ms" in r], .5)}}


def checked_result(row, request, clock_id, observed_until):
    if row.get("source_request_sha256") != request["source_request_sha256"]:
        raise ValueError("result request identity differs")
    if row.get("clock_id") != clock_id or row.get("time_reference") != "client_origin_ms":
        raise ValueError("result is not bound to the common client clock")
    status = row.get("status")
    if status not in TERMINAL:
        raise ValueError("unknown terminal result status")
    result = {"request_id": request["request_id"], "model": request["model"], "shape_id": request["shape_id"],
              "status": status, "slo_met": False}
    for field in ("arrival_ms", "deadline_ms"):
        if field in row and row[field] != request[field]:
            raise ValueError("result attempted to alter frozen " + field)
    if status != "completed":
        return result
    if (row.get("executed_spec_sha256") != request["spec_sha256"]
            or row.get("model_manifest_sha256") != request["model_manifest"]["sha256"]):
        raise ValueError("executed model or generation parameters differ")
    dispatch = finite(row.get("dispatch_ms"), "dispatch_ms")
    ready = finite(row.get("output_ready_ms"), "output_ready_ms")
    if not request["arrival_ms"] <= dispatch <= ready <= observed_until:
        raise ValueError("completion/dispatch timestamp is outside the client observation interval")
    output = row.get("output", {})
    spec = request["spec"]
    if (output.get("decoded") is not True or output.get("finite") is not True or type(output.get("count")) is not int
            or output["count"] != 1 or not hex64(output.get("sha256"))
            or any(output.get(k) != spec[k] for k in ("height", "width", "frames"))):
        raise ValueError("completion requires one finite fully decoded output of the requested shape")
    result.update(output_ready_ms=ready, latency_ms=ready-request["arrival_ms"],
        dispatch_lag_ms=dispatch-request["arrival_ms"], slo_met=ready <= request["deadline_ms"])
    if "client_received_ms" in row:
        received = finite(row["client_received_ms"], "client_received_ms")
        if not ready <= received <= observed_until:
            raise ValueError("wire completion precedes decoded-ready or exceeds observation")
        result["wire_after_ready_ms"] = received-ready
    return result


def account_results(export, results):
    verify_export(export)
    expected = {r["request_id"]:r for r in export["requests"]}
    issues, global_valid = [], True
    try:
        if (results.get("schema") != RESULT_SCHEMA or results.get("workload_sha256") != export["sha256"]
                or results.get("completion_boundary") != BOUNDARY):
            raise ValueError("result envelope workload/completion binding differs")
        clock = results.get("client_clock", {})
        if (not isinstance(clock.get("id"), str) or not clock["id"]
                or type(clock.get("origin_monotonic_ns")) is not int or clock["origin_monotonic_ns"] < 0
                or clock.get("time_reference") != "client_origin_ms"):
            raise ValueError("missing common client monotonic origin")
        observed = finite(results.get("observed_until_ms"), "observed_until_ms")
        system = results.get("system", {})
        if not isinstance(system.get("id"), str) or not system["id"] or not hex64(system.get("source_manifest_sha256")):
            raise ValueError("system source identity is missing")
        records = results.get("requests")
        if not isinstance(records, list):
            raise ValueError("result requests must be a list")
    except (ValueError, TypeError) as exc:
        issues.append(str(exc)); global_valid = False
        clock, observed, records = {}, 0, []
    by_id = defaultdict(list)
    for row in records:
        if not isinstance(row, dict) or row.get("request_id") not in expected:
            issues.append("unknown or malformed result request")
            continue
        by_id[row["request_id"]].append(row)
    accounted = []
    for rid, request in expected.items():
        rows = by_id[rid]
        entry = {"request_id": rid, "model": request["model"], "shape_id": request["shape_id"],
                 "status": "missing", "slo_met": False}
        if not global_valid:
            entry["status"] = "invalid"
        elif len(rows) > 1:
            entry["status"] = "invalid"; issues.append(rid + ": duplicate terminal result")
        elif rows:
            try:
                entry = checked_result(rows[0], request, clock["id"], observed)
            except (ValueError, KeyError, TypeError) as exc:
                entry["status"] = "invalid"; issues.append(rid + ": " + str(exc))
        accounted.append(entry)
    horizon = export["clock_contract"]["measurement_horizon_ms"]
    by_model, by_shape = {}, {}
    for model in export["comparison_cell"]["models"]:
        by_model[model] = group_metrics([r for r in accounted if r["model"] == model], horizon)
    for shape in export["formal_protocol"]["shape_ids"]:
        by_shape[shape] = group_metrics([r for r in accounted if r["shape_id"] == shape], horizon)
    incomplete = any(r["status"] in ("missing", "incomplete", "not_executed") for r in accounted) or observed < horizon
    status = "invalid_evidence" if issues else "incomplete" if incomplete else "accounted"
    return seal({"schema": "ditserve.external-accounting.v1", "status": status,
        "workload_sha256": export["sha256"], "result_document_sha256": canonical(results),
        "comparison_cell": copy.deepcopy(export["comparison_cell"]), "system": copy.deepcopy(results.get("system", {})),
        "client_clock": copy.deepcopy(results.get("client_clock", {})), "observed_until_ms": observed,
        "completion_boundary": BOUNDARY, "measurement_horizon_ms": horizon,
        "denominator_rule": "all frozen offered requests, including missing, failed, rejected, cancelled, unexecuted and timed-out",
        "latency_scope": "completed only; no fabricated latency for missing or failed requests; linear-interpolated empirical quantiles",
        "overall": group_metrics(accounted, horizon), "by_model": by_model, "by_shape": by_shape,
        "accounted_requests": accounted, "issues": issues,
        "declared_formal_trace_candidate": export["declared_formal_trace_candidate"],
        "formal_candidate_scope": export["formal_candidate_scope"],
        "qualification_eligible": False, "allows": [], "quality_validated": False,
        "scope": "source-bound input preservation and accounting only; execution, numerical quality and resource equivalence need independent evidence"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    for name in ("study", "protocol", "contract", "models"):
        export.add_argument("--"+name, type=Path, required=True)
        export.add_argument("--"+name+"-sha256", required=True)
    export.add_argument("--phase")
    export.add_argument("--output", type=Path, required=True)
    account = sub.add_parser("account")
    for name in ("workload", "results"):
        account.add_argument("--"+name, type=Path, required=True)
        account.add_argument("--"+name+"-sha256", required=True)
    account.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        values, bindings = {}, {}
        for name in ("study", "protocol", "contract", "models"):
            values[name], bindings[name] = load_bound(getattr(args, name), getattr(args, name+"_sha256"))
        result = export_study(**values, bindings=bindings, phase_id=args.phase)
    else:
        workload, wb = load_bound(args.workload, args.workload_sha256)
        results, rb = load_bound(args.results, args.results_sha256)
        result = account_results(workload, results)
        result.pop("sha256")
        result["raw_input_bindings"] = {"workload": wb, "results": rb}
        result = seal(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": result["sha256"],
        "status": result.get("status", "exported"), "qualification_eligible": False}))
    return 1 if result.get("status") == "invalid_evidence" else 0


if __name__ == "__main__":
    raise SystemExit(main())
