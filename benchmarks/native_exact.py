#!/usr/bin/env python3
"""Verify fixed-form block preemption against digest-bound native outputs."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from native_metrics import distribution


# Records written before the policy registry use the legacy names.
EQUIVALENT_POLICIES = (
    {"edf-cfg2", "static-cfg2", "static_cfg2"},
    {"block-cfg2", "block_cfg2"},
)


def same_policy(name, wanted):
    return name == wanted or any(name in group and wanted in group for group in EQUIVALENT_POLICIES)


def rows_for(record, policy):
    cells = [cell for cell in record["phases"][0]["cells"] if same_policy(cell["policy"], policy)]
    if len(cells) != 1:
        raise ValueError("expected exactly one cell for policy " + policy)
    cell = cells[0]
    if cell["status"] != "completed":
        raise ValueError("policy cell did not complete: " + policy)
    rows = [row for row in record["requests"] if row["cell"] == cell["cell"] and row["kind"] == "video"]
    indexed = {row["request_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate video request identity in " + policy)
    return cell, indexed


def validate_pair_identity(reference, candidate, *, allow_approximate=False, allow_legacy_runtime=False):
    """Fail closed on missing identity; approximation is the only permitted change."""
    required_spec = {"prompt", "negative_prompt", "seed", "steps", "frames", "height", "width", "guidance_scale"}
    required_generation = {"model_path", "revision", "dtype", "vae_dtype", "text_encoder_dtype",
                           "attention", "engine", "xdit_revision"}
    for row in (reference, candidate):
        if row.get("status") != "completed" or not row.get("outputs"):
            raise ValueError("comparison requires completed outputs")
        if not required_spec <= row.get("spec", {}).keys():
            raise ValueError("missing generation input identity")
        generation = row["outputs"][-1].get("generation", {})
        if not required_generation <= generation.keys():
            raise ValueError("missing model, dtype or execution identity")
        if not generation.get("runtime_identity") and not allow_legacy_runtime:
            raise ValueError("missing runtime_identity; legacy records require explicit diagnostic mode")
        for key in required_spec:
            if generation.get(key) != row["spec"][key]:
                raise ValueError("produced generation differs from request spec: " + key)
    # request_id is a run-local label; all actual inputs must match, including optional settings.
    normalized = lambda spec: {k: v for k, v in spec.items() if k != "request_id"}
    if normalized(reference["spec"]) != normalized(candidate["spec"]):
        raise ValueError("generation input identity differs")
    ignored = {"request_id"}
    if allow_approximate:
        ignored |= {"approximate", "cfg_folded_steps"}
    generations = [{k: v for k, v in row["outputs"][-1]["generation"].items() if k not in ignored}
                   for row in (reference, candidate)]
    if generations[0] != generations[1]:
        raise ValueError("model or execution identity differs")
    ref_artifact, candidate_artifact = (row["outputs"][-1]["output"] for row in (reference, candidate))
    for key in ("shape", "dtype"):
        if key not in ref_artifact or key not in candidate_artifact or ref_artifact[key] != candidate_artifact[key]:
            raise ValueError("output identity differs or is missing: " + key)
    generation = reference["outputs"][-1]["generation"]
    manifest_bound = bool(re.fullmatch(r"[0-9a-f]{64}", generation.get("model_manifest_sha256", "")))
    return {"runtime_verified": bool(generation.get("runtime_identity")),
            "weights_manifest_bound": manifest_bound,
            "weights_verified": manifest_bound and generation.get("model_integrity") == "manifest_verified"}


def checked_digest(row):
    if row["status"] != "completed" or not row.get("outputs"):
        raise ValueError("missing completed video output: " + row["request_id"])
    produced = row["outputs"][-1]
    artifact = produced["output"]
    if artifact.get("output_mode") != "digest" or not re.fullmatch(r"[0-9a-f]{64}", artifact.get("array_sha256", "")):
        raise ValueError("output is not digest-bound: " + row["request_id"])
    if artifact.get("finite") is not True:
        raise ValueError("digest output is not verified finite: " + row["request_id"])
    if produced["generation"].get("approximate") is not False:
        raise ValueError("exact comparison received approximate output: " + row["request_id"])
    return artifact["array_sha256"]


def compare(record, reference_policy="edf-cfg2", candidate_policy="block-cfg2", form="cfg2", *,
            require_preemption=True, allow_legacy_runtime=False):
    reference_cell, reference = rows_for(record, reference_policy)
    candidate_cell, candidate = rows_for(record, candidate_policy)
    missing = sorted(set(reference) ^ set(candidate))
    mismatches, pairs = [], []
    for request_id in sorted(set(reference) & set(candidate)):
        ref, other = reference[request_id], candidate[request_id]
        if ref["forms"] != [form] or other["forms"] != [form]:
            raise ValueError("parallel form differs from the controlled form: " + request_id)
        identity = validate_pair_identity(ref, other, allow_legacy_runtime=allow_legacy_runtime)
        ref_hash, other_hash = checked_digest(ref), checked_digest(other)
        pairs.append({"request_id": request_id, "reference_sha256": ref_hash, "candidate_sha256": other_hash,
                      **identity})
        if ref_hash != other_hash:
            mismatches.append(request_id)
    stops = [sample["stop_ms"] for row in candidate.values() for sample in row.get("preemption_stop_samples", [])]
    mechanism_observed = bool(stops) if require_preemption else not stops
    status = "completed" if pairs and not missing and not mismatches and mechanism_observed else "incomplete"
    deltas = [(candidate[key]["finish_ms"] - candidate[key]["arrival_ms"])
              - (reference[key]["finish_ms"] - reference[key]["arrival_ms"])
              for key in reference.keys() & candidate.keys()
              if all("finish_ms" in row and "arrival_ms" in row for row in (reference[key], candidate[key]))]
    return {
        "schema": "ditserve.native-exact.v1",
        "status": status,
        "scope": "same fixed form digest comparison; runtime and weight identity reported separately; not an FP32 or cross-SP claim",
        "identity_qualification": "verified" if pairs and all(row["runtime_verified"] for row in pairs) else "unverified",
        "weights_verified": bool(pairs) and all(row["weights_verified"] for row in pairs),
        "run_id": record["run_id"],
        "reference": {"policy": reference_policy, "cell": reference_cell["cell"]},
        "candidate": {"policy": candidate_policy, "cell": candidate_cell["cell"]},
        "form": form,
        "matched_videos": len(pairs),
        "missing_identities": missing,
        "mismatches": mismatches,
        "preemption_stop_ms": distribution(stops),
        "diagnostic": "contention" if require_preemption else "no_preemption_overhead",
        "mechanism_observed": mechanism_observed,
        "paired_latency_delta_ms": distribution(deltas),
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-policy", default="edf-cfg2")
    parser.add_argument("--candidate-policy", default="block-cfg2")
    parser.add_argument("--form", default="cfg2")
    parser.add_argument("--no-preemption", action="store_true", help="require zero observed stops for the overhead diagnostic")
    parser.add_argument("--allow-legacy-runtime", action="store_true", help="diagnostic only when historical runtime metadata is absent")
    parser.add_argument("--keep-pairs", action="store_true", help="retain per-video digests; default keeps aggregate and mismatch evidence")
    args = parser.parse_args()
    raw = args.run.read_bytes()
    record = json.loads(raw)
    result = compare(record,
                     args.reference_policy, args.candidate_policy, args.form, require_preemption=not args.no_preemption,
                     allow_legacy_runtime=args.allow_legacy_runtime)
    result["source"] = {"run_sha256": hashlib.sha256(raw).hexdigest(), "runtime_source": record.get("source")}
    if not args.keep_pairs:
        result["mismatch_details"] = [row for row in result.pop("pairs") if row["request_id"] in result["mismatches"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "matched_videos": result["matched_videos"],
                      "mismatches": result["mismatches"], "preemption_stop_ms": result["preemption_stop_ms"]}, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
