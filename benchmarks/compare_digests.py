#!/usr/bin/env python3
"""Compare digest-bound video outputs of the same policy across two native runs.

Used for the collective-backend exactness check: one run with collective_backend=nccl and one with
collective_backend=gfc over the same frozen trace. Group-free collectives only move bytes, so every
video digest of a fixed-form policy must match; a mismatch is a correctness failure. Elastic
policies may legitimately differ only if their placement differs, so they are reported separately.
"""

import argparse
import hashlib
import json
from pathlib import Path

from native_exact import checked_digest, rows_for, validate_pair_identity
from native_metrics import distribution


def digests(record, policy):
    _, rows = rows_for(record, policy)
    out = {}
    for request_id, row in rows.items():
        out[request_id] = {"sha256": checked_digest(row), "forms": row.get("forms", []), "row": row}
    return out


def compare(reference, candidate, policies, *, allow_legacy_runtime=False):
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("policies must be nonempty and unique")
    if any(run.get("status") != "completed" for run in (reference, candidate)):
        raise ValueError("digest comparison requires completed runs")
    if not reference.get("source") or reference.get("source") != candidate.get("source"):
        raise ValueError("runtime source identity differs or is missing")
    results = {}
    for policy in policies:
        ref, cand = digests(reference, policy), digests(candidate, policy)
        common = sorted(set(ref) & set(cand))
        identities = [validate_pair_identity(ref[rid]["row"], cand[rid]["row"],
                                             allow_legacy_runtime=allow_legacy_runtime) for rid in common]
        same_forms = [rid for rid in common if ref[rid]["forms"] == cand[rid]["forms"]]
        mismatched = [rid for rid in common if ref[rid]["sha256"] != cand[rid]["sha256"]]
        results[policy] = {
            "matched_videos": len(common), "same_observed_forms": len(same_forms),
            "digest_mismatches_with_same_forms": [rid for rid in mismatched if rid in same_forms],
            "digest_mismatches_with_different_forms": [rid for rid in mismatched if rid not in same_forms],
            "identity_qualification": "verified" if identities and all(row["runtime_verified"] for row in identities) else "unverified",
            "weights_verified": bool(identities) and all(row["weights_verified"] for row in identities),
            "missing": sorted(set(ref) ^ set(cand)),
            "preemption_stop_ms": {name: distribution([sample["stop_ms"] for item in rows.values()
                for sample in item["row"].get("preemption_stop_samples", [])])
                for name, rows in (("reference", ref), ("candidate", cand))},
        }
    ok = all(not r["digest_mismatches_with_same_forms"] and not r["digest_mismatches_with_different_forms"]
             and not r["missing"] and r["matched_videos"]
             for r in results.values())
    return {"schema": "ditserve.digest-compare.v1", "status": "completed" if ok else "incomplete",
            "reference_run": reference["run_id"], "candidate_run": candidate["run_id"],
            "scope": "all paired videos must have equal digests; changed forms explain inconclusive mismatches, not a successful check",
            "policies": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--policies", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-legacy-runtime", action="store_true")
    args = parser.parse_args()
    reference_raw, candidate_raw = args.reference.read_bytes(), args.candidate.read_bytes()
    reference, candidate = json.loads(reference_raw), json.loads(candidate_raw)
    result = compare(reference, candidate, args.policies.split(","),
                     allow_legacy_runtime=args.allow_legacy_runtime)
    result["source"] = {"reference_run_sha256": hashlib.sha256(reference_raw).hexdigest(),
                        "candidate_run_sha256": hashlib.sha256(candidate_raw).hexdigest(),
                        "runtime_source": reference.get("source")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
