#!/usr/bin/env python3
"""Recompute staged experiment eligibility from hash-bound source evidence.

This gate does not submit jobs. CONTRACT permits calibration and controlled
P0 runs; complete P0 permits P1 diagnostics. Later stages remain pending until
their scientific evaluators exist and produce the required evidence.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

from native_exact import compare, same_policy


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(raw):
    def reject(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(raw, parse_constant=reject)


def source_file(base, reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("source reference requires path and sha256 only")
    path = (base / reference["path"]).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("source file hash differs: " + str(path))
    return raw


def contract_check(base, evidence, build):
    smoke = read_json(source_file(base, evidence["runtime_smoke"]))
    log = source_file(base, evidence["log"]).decode("utf-8")
    job_raw = source_file(base, evidence["job_spec"])
    job = read_json(job_raw)
    if smoke.get("schema") != "ditserve.runtime-smoke.v1" or smoke.get("status") != "passed":
        raise ValueError("runtime CUDA smoke did not pass")
    for field in ("commit", "core_sha256", "image_digest"):
        if smoke.get(field) != build[field]:
            raise ValueError("runtime smoke build identity differs: " + field)
    if smoke.get("test_revision") != build["commit"]:
        raise ValueError("contract test revision differs from build")
    python_sources = smoke.get("python_sources")
    if not isinstance(python_sources, dict) or not python_sources:
        raise ValueError("contract requires the complete frozen Python source inventory")
    for name, digest in python_sources.items():
        path = PurePosixPath(name)
        if (str(path) != name or path.parts[0] not in ("ditserve", "xfuser")
                or ".." in path.parts or "\\" in name or path.suffix != ".py"
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("invalid contract Python source identity")
    env = {item["name"]: item.get("value") for item in job.get("env", [])}
    job_commit = env.get("DITSERVE_BUILD", "")
    if (not re.fullmatch(r"[0-9a-f]{7,40}", job_commit) or not build["commit"].startswith(job_commit)
            or job.get("name") != smoke.get("run_id")
            or not job.get("image", "").endswith("@" + build["image_digest"])
            or smoke.get("execution_image") != job["image"]
            or smoke.get("image_identity_source") != "submitted_job_spec_digest_pin"
            or smoke.get("image_job_spec_sha256") != hashlib.sha256(job_raw).hexdigest()):
        raise ValueError("contract image/build is not bound to the submitted digest-pinned job spec")
    if smoke.get("devices") != 1 or smoke.get("sum") != 524800 or smoke.get("matmul_value") != 32:
        raise ValueError("runtime smoke lacks expected allocated-GPU arithmetic")
    # Both the named contract modules and the dated regression suite must finish.
    # Partial reruns, missing second suite, or optional dependency skips cannot
    # substitute for a complete fixed-runtime contract.
    runs = re.findall(r"^Ran (\d+) tests? in [^\n]+\n\s*\nOK\s*$", log, re.MULTILINE)
    if len(runs) < 2 or any(int(count) <= 0 for count in runs):
        raise ValueError("two complete nonempty unittest success summaries required")
    if re.search(r"^(FAILED\b|ERROR:|FAIL:)|\bskipped\b|^recheck=", log, re.MULTILINE):
        raise ValueError("contract contains failures, skips or a partial recheck")
    return {"status": "passed", "tests": sum(map(int, runs)), "suites": len(runs),
            "python_sources_sha256": canonical_hash(python_sources),
            "image_identity_scope": "submitted digest-pinned job spec; not independent pod imageID inspection",
            "allows": ["shape_calibration", "shape_qualification", "preemption", "no_preemption"]}


def p0_controls(record, item, protocol):
    """Check executed inputs, not just the requested protocol label."""
    config = record["config"]
    benchmark = config["benchmark"]
    diagnostic = "contention" if item["role"] == "preemption" else "no_preemption"
    controls = protocol["p0_diagnostics"][diagnostic]
    if (benchmark.get("p0_diagnostic_controls") != controls
            or controls.get("require_observed_stops") is not (item["role"] == "preemption")):
        raise ValueError("P0 diagnostic controls differ from frozen protocol")
    for field in ("count", "video_fraction", "load", "pattern", "video_slo_scale", "image_slo_scale"):
        if benchmark.get(field) != controls[field]:
            raise ValueError("P0 benchmark control differs: " + field)
    count = controls["count"]
    fraction = controls["video_fraction"]
    if (type(count) is not int or count <= 0 or type(fraction) not in (int, float)
            or not 0 < fraction <= 1 or count * fraction != round(count * fraction)):
        raise ValueError("P0 controls require an exact positive video count")
    if len(config.get("phases", [])) != 1 or len(record.get("phases", [])) != 1:
        raise ValueError("P0 requires one controlled phase")
    phase = config["phases"][0]
    matrix = phase.get("matrix", {})
    if (set(matrix) != {"/policy", "/workload", "/trace_provenance"}
            or sorted(matrix["/policy"]) != ["block-cfg2", "edf-cfg2"]
            or len(matrix["/workload"]) != 1 or len(matrix["/trace_provenance"]) != 1
            or phase.get("options", {}).get("forms") != ["cfg2"]
            or phase.get("options", {}).get("lazy_preemption") is not controls["lazy_preemption"]):
        raise ValueError("P0 policy/form/lazy-preemption configuration differs")
    trace = matrix["/workload"][0]
    expected = {row["request_id"]: row for row in trace}
    if (len(trace) != count or len(expected) != count
            or sum(row["kind"] == "video" for row in trace) != round(count * fraction)
            or any(row["kind"] not in ("video", "image") for row in trace)):
        raise ValueError("P0 workload count or modality mix differs")
    provenance = benchmark["trace"]
    # build_trace hashes escaped JSON; this differs from the protocol's UTF-8
    # canonical encoding for non-ASCII prompts.
    trace_hash = hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
    if (matrix["/trace_provenance"][0] != provenance or provenance.get("sha256") != trace_hash
            or provenance.get("manifest", {}).get("workload_sha256") != trace_hash):
        raise ValueError("P0 workload provenance differs from actual trace")
    windows = provenance["manifest"].get("windows", [])
    if len(windows) != 1 or any(windows[0].get(field) != controls[field]
            for field in ("count", "video_fraction", "load", "pattern")):
        raise ValueError("P0 arrival controls differ from trace window")
    for row in trace:
        shape = protocol["shapes"].get(row["class"], {})
        if (shape.get("kind") != row["kind"] or shape.get("model") != (item["model"] if row["kind"] == "video" else "sd35")
                or any(row["spec"].get(field) != shape.get(field)
                       for field in ("height", "width", "frames", "steps", "guidance_scale"))):
            raise ValueError("P0 trace shape differs from protocol")
        cost = phase["profile"]["classes"][row["class"]]
        service = (cost.get("prepare_ms", 0) + cost.get("encode_ms", 0)
                   + row["spec"]["steps"] * cost["step_compute_ms"] + cost["finish_ms"]
                   if row["kind"] == "video" else cost["service_ms"])
        scale = controls[row["kind"] + "_slo_scale"]
        if (not math.isfinite(service) or service <= 0 or row.get("effective_slo_scale") != scale
                or not math.isclose(row["service_ms"], service, rel_tol=1e-10, abs_tol=1e-6)
                or not math.isclose(row["slo_ms"], service * scale, rel_tol=1e-10, abs_tol=1e-6)
                or not math.isclose(row["deadline_ms"] - row["arrival_ms"], row["slo_ms"], rel_tol=1e-10, abs_tol=1e-6)):
            raise ValueError("P0 deadline/SLO differs from frozen scale and measured profile")
    cells = record["phases"][0].get("cells", [])
    phase_id = phase.get("id", "cluster")
    if record["phases"][0].get("id", "cluster") != phase_id or len(cells) != 2 or any(
            sum(same_policy(cell["policy"], policy) for cell in cells) != 1
            for policy in ("edf-cfg2", "block-cfg2")):
        raise ValueError("P0 requires both complete controlled policy cells")
    if len(record["requests"]) != 2 * count:
        raise ValueError("P0 completed request count differs from frozen protocol")
    for cell in cells:
        actual = [row for row in record["requests"] if row.get("phase", "cluster") == phase_id and row["cell"] == cell["cell"]]
        if len(actual) != count or {row["request_id"] for row in actual} != set(expected):
            raise ValueError("P0 policy cell request coverage differs")
        for row in actual:
            wanted = expected[row["request_id"]]
            for field in ("kind", "class", "arrival_ms", "deadline_ms", "slo_ms", "spec", "dataset_sample_id"):
                if row.get(field) != wanted.get(field):
                    raise ValueError("P0 executed request differs from frozen trace: " + field)
    manifests = benchmark["model_manifests"]
    workers = {}
    for worker in config.get("workers", {}).values():
        model = item["model"] if worker["backend"] in ("wan", "wan_pipeline") else "sd35"
        binding = manifests[model]
        value = worker["config"]
        if (not re.fullmatch(r"[0-9a-f]{64}", binding.get("sha256", ""))
                or value.get("model_manifest_sha256") != binding["sha256"]
                or value.get("model_manifest_path") != binding["path"]):
            raise ValueError("P0 worker model manifest differs from frozen binding")
        workers.setdefault(model, set()).add(value["model_path"])
    for row in record["requests"]:
        model = item["model"] if row["kind"] == "video" else "sd35"
        generation = row["outputs"][-1].get("generation", {})
        if (generation.get("model_manifest_sha256") != manifests[model]["sha256"]
                or generation.get("model_integrity") != "manifest_verified"
                or generation.get("model_path") not in workers.get(model, set())):
            raise ValueError("P0 output weight identity differs from expected worker model manifest")


def p0_run(base, item, build, protocol_sha256, protocol, expected_python_sources):
    record = read_json(source_file(base, {key: item[key] for key in ("path", "sha256")}))
    if record.get("source", {}).get("core_sha256") != build["core_sha256"]:
        raise ValueError("run core identity differs from contract")
    if not expected_python_sources or record.get("source", {}).get("files") != expected_python_sources:
        raise ValueError("run frozen Python source inventory differs from verified contract")
    if record.get("failures"):
        raise ValueError("run contains failures")
    binding = record.get("config", {}).get("benchmark", {}).get("formal_protocol", {})
    shape = protocol.get("shapes", {}).get(item["shape_id"], {})
    if shape.get("model") != item["model"] or shape.get("kind") != "video":
        raise ValueError("P0 model/shape is not defined by the protocol")
    if (binding.get("schema") != "ditserve.formal-protocol.v1"
            or binding.get("sha256") != protocol_sha256
            or item["shape_id"] not in binding.get("shape_ids", [])
            or binding.get("role") != item["role"]
            or binding.get("split") != "development"):
        raise ValueError("run does not bind the required protocol, shape, split and role")
    for row in record.get("requests", []):
        if row.get("status") != "completed":
            raise ValueError("P0 contains a failed or incomplete request")
        for field in ("arrival_ms", "finish_ms", "deadline_ms"):
            if type(row.get(field)) not in (int, float):
                raise ValueError("P0 request lacks measured timing: " + field)
        if row["finish_ms"] < row["arrival_ms"]:
            raise ValueError("request finishes before arrival")
        if row.get("kind") == "video" and any(row.get("spec", {}).get(field) != shape.get(field)
                for field in ("height", "width", "frames", "steps", "guidance_scale")):
            raise ValueError("executed video shape differs from frozen protocol")
    p0_controls(record, item, protocol)
    result = compare(record, require_preemption=item["role"] == "preemption")
    if not result["qualification_eligible"]:
        raise ValueError("digest/mechanism/runtime/weight qualification did not pass")
    if result["paired_latency_delta_ms"]["count"] != result["matched_videos"]:
        raise ValueError("P0 latency evidence does not cover every paired video")
    return {"model": item["model"], "shape_id": item["shape_id"], "role": item["role"],
            "status": "passed", "matched_videos": result["matched_videos"],
            "preemption_stop_ms": result["preemption_stop_ms"],
            "paired_latency_delta_ms": result["paired_latency_delta_ms"],
            "source_sha256": item["sha256"]}


def evaluate(manifest, base):
    if manifest.get("schema") != "ditserve.stage-evidence.v1":
        raise ValueError("expected ditserve.stage-evidence.v1")
    protocol = read_json(source_file(base, manifest["protocol"]))
    if protocol.get("schema") != "ditserve.formal-protocol.v1":
        raise ValueError("expected frozen formal protocol")
    protocol_sha256 = canonical_hash(protocol)
    build = manifest["build"]
    if (not re.fullmatch(r"[0-9a-f]{40}", build.get("commit", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", build.get("core_sha256", ""))
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", build.get("image_digest", ""))):
        raise ValueError("build requires full commit, core SHA256 and image manifest digest")
    try:
        contract = contract_check(base, manifest["contract"], build)
    except (KeyError, ValueError, OSError) as error:
        contract = {"status": "blocked", "reason": str(error), "allows": []}
    expected_python_sources = (read_json(source_file(base, manifest["contract"]["runtime_smoke"]))["python_sources"]
                               if contract["status"] == "passed" else None)
    expected = protocol.get("p0_required_cases", [])
    if not expected or any(set(case) != {"model", "shape_id"} for case in expected):
        raise ValueError("protocol requires explicit nonempty p0_required_cases")
    required = {(case["model"], case["shape_id"], role) for case in expected
                for role in ("preemption", "no_preemption")}
    supplied = {}
    for item in manifest.get("runs", []):
        key = (item["model"], item["shape_id"], item["role"])
        if key not in required or key in supplied:
            raise ValueError("unexpected or duplicate P0 evidence case: " + repr(key))
        supplied[key] = item
    checks = []
    for key in sorted(required):
        if key not in supplied:
            checks.append(dict(zip(("model", "shape_id", "role"), key), status="blocked", reason="missing evidence"))
            continue
        try:
            checks.append(p0_run(base, supplied[key], build, protocol_sha256, protocol, expected_python_sources))
        except (KeyError, ValueError, OSError) as error:
            checks.append(dict(zip(("model", "shape_id", "role"), key), status="blocked", reason=str(error)))
    passed = contract["status"] == "passed" and all(check["status"] == "passed" for check in checks)
    stages = {
        "CONTRACT": contract,
        "P0": {"status": "passed" if passed else "blocked", "checks": checks,
               "allows": ["p1_diagnostics"] if passed else [],
               "scope": "specified model/shape fixed-form correctness with C++ and frozen Python identities matching CONTRACT; "
                        "not formal performance or quality"},
    }
    remaining = {
        "P1": "matched-shape stop/step response and no-preemption overhead; formal shape qualification",
        "P2": "paired strong-baseline and mechanism-ablation benefit with per-modality constraints",
        "P3": "frozen mixed-shape multi-seed serving matrix and sustained capacity",
        "P4": "held-out numeric and semantic quality plus matched-compute benefit and resource reuse",
        "P5": "external-system, model-family and hardware/scale validity",
    }
    for stage, requirements in remaining.items():
        stages[stage] = {"status": "pending", "allows": [], "required_evidence": requirements,
                         "reason": "no validated stage evaluator/evidence; job completion cannot promote this stage"}
    return {"schema": "ditserve.stage-gates.v1", "build": build,
            "protocol_sha256": protocol_sha256, "stages": stages}


def verify_receipt(path, *, build_commit=None, protocol_sha256=None, required_stage="P0", action=None):
    path = Path(path).resolve()
    receipt = read_json(path.read_bytes())
    if receipt.get("schema") != "ditserve.stage-gates.v1":
        raise ValueError("invalid stage receipt")
    reference = receipt["source_manifest"]
    manifest_path = (path.parent / reference["path"]).resolve()
    manifest = read_json(source_file(path.parent, reference))
    fresh = evaluate(manifest, manifest_path.parent)
    if {key: receipt.get(key) for key in fresh} != fresh:
        raise ValueError("receipt differs from recomputed evidence")
    if build_commit is not None and not fresh["build"]["commit"].startswith(build_commit):
        raise ValueError("receipt build does not match submitted build")
    if protocol_sha256 is not None and fresh["protocol_sha256"] != protocol_sha256:
        raise ValueError("receipt protocol does not match submitted protocol")
    stage = fresh["stages"].get(required_stage, {})
    if stage.get("status") != "passed":
        raise ValueError("required stage has not passed: " + required_stage)
    if action is not None and action not in stage.get("allows", []):
        raise ValueError("stage does not authorize action: " + action)
    return fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-stage", choices=["CONTRACT", "P0", "P1", "P2", "P3", "P4", "P5"], default="P0")
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    result = evaluate(read_json(raw), args.manifest.resolve().parent)
    result["source_manifest"] = {"path": str(args.manifest.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"stages": {key: value["status"] for key, value in result["stages"].items()}}))
    if result["stages"][args.require_stage]["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
