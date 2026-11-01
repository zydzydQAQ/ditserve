#!/usr/bin/env python3
"""Verify fixed-form block preemption against digest-bound native outputs."""

import argparse
import json
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
    return cell, {row["request_id"]: row for row in record["requests"]
                  if row["cell"] == cell["cell"] and row["kind"] == "video"}


def checked_digest(row):
    if row["status"] != "completed" or not row.get("outputs"):
        raise ValueError("missing completed video output: " + row["request_id"])
    produced = row["outputs"][-1]
    artifact = produced["output"]
    if artifact.get("output_mode") != "digest" or len(artifact.get("array_sha256", "")) != 64:
        raise ValueError("output is not digest-bound: " + row["request_id"])
    if produced["generation"].get("approximate") is not False:
        raise ValueError("exact comparison received approximate output: " + row["request_id"])
    return artifact["array_sha256"]


def compare(record, reference_policy="edf-cfg2", candidate_policy="block-cfg2", form="cfg2"):
    reference_cell, reference = rows_for(record, reference_policy)
    candidate_cell, candidate = rows_for(record, candidate_policy)
    missing = sorted(set(reference) ^ set(candidate))
    mismatches, pairs = [], []
    for request_id in sorted(set(reference) & set(candidate)):
        ref, other = reference[request_id], candidate[request_id]
        if ref["forms"] != [form] or other["forms"] != [form]:
            raise ValueError("parallel form differs from the controlled form: " + request_id)
        if ref["spec"]["prompt"] != other["spec"]["prompt"] or ref["spec"]["seed"] != other["spec"]["seed"]:
            raise ValueError("input identity differs: " + request_id)
        ref_hash, other_hash = checked_digest(ref), checked_digest(other)
        pairs.append({"request_id": request_id, "reference_sha256": ref_hash, "candidate_sha256": other_hash})
        if ref_hash != other_hash:
            mismatches.append(request_id)
    stops = [sample["stop_ms"] for row in candidate.values() for sample in row.get("preemption_stop_samples", [])]
    status = "completed" if pairs and not missing and not mismatches and stops else "incomplete"
    return {
        "schema": "ditserve.native-exact.v1",
        "status": status,
        "scope": "same-runtime, same fixed form, digest equality; not an FP32 or cross-SP claim",
        "run_id": record["run_id"],
        "reference": {"policy": reference_policy, "cell": reference_cell["cell"]},
        "candidate": {"policy": candidate_policy, "cell": candidate_cell["cell"]},
        "form": form,
        "matched_videos": len(pairs),
        "missing_identities": missing,
        "mismatches": mismatches,
        "preemption_stop_ms": distribution(stops),
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-policy", default="edf-cfg2")
    parser.add_argument("--candidate-policy", default="block-cfg2")
    parser.add_argument("--form", default="cfg2")
    args = parser.parse_args()
    result = compare(json.loads(args.run.read_text(encoding="utf-8")),
                     args.reference_policy, args.candidate_policy, args.form)
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
