#!/usr/bin/env python3
"""Compare digest-bound video outputs of the same policy across two native runs.

Used for the collective-backend exactness check: one run with collective_backend=nccl and one with
collective_backend=gfc over the same frozen trace. Group-free collectives only move bytes, so every
video digest of a fixed-form policy must match; a mismatch is a correctness failure. Elastic
policies may legitimately differ only if their placement differs, so they are reported separately.
"""

import argparse
import json
from pathlib import Path


def digests(record, policy):
    cells = [cell for cell in record["phases"][0]["cells"] if cell["policy"] == policy]
    if len(cells) != 1:
        raise ValueError(f"expected one cell for {policy} in {record['run_id']}")
    out = {}
    for row in record["requests"]:
        if row["cell"] != cells[0]["cell"] or row["kind"] != "video":
            continue
        if row["status"] != "completed" or not row.get("outputs"):
            raise ValueError("missing completed video output: " + row["request_id"])
        artifact = row["outputs"][-1]["output"]
        if artifact.get("output_mode") != "digest":
            raise ValueError("output is not digest-bound: " + row["request_id"])
        out[row["request_id"]] = {"sha256": artifact["array_sha256"], "forms": row.get("forms", [])}
    return out


def compare(reference, candidate, policies):
    results = {}
    for policy in policies:
        ref, cand = digests(reference, policy), digests(candidate, policy)
        common = sorted(set(ref) & set(cand))
        same_forms = [rid for rid in common if ref[rid]["forms"] == cand[rid]["forms"]]
        mismatched = [rid for rid in same_forms if ref[rid]["sha256"] != cand[rid]["sha256"]]
        results[policy] = {
            "matched_videos": len(common), "same_form_sequence": len(same_forms),
            "digest_mismatches_with_same_forms": mismatched,
            "missing": sorted(set(ref) ^ set(cand)),
        }
    ok = all(not r["digest_mismatches_with_same_forms"] and not r["missing"] and r["same_form_sequence"]
             for r in results.values())
    return {"schema": "ditserve.digest-compare.v1", "status": "completed" if ok else "incomplete",
            "reference_run": reference["run_id"], "candidate_run": candidate["run_id"],
            "scope": "same trace, same runtime; videos with identical form sequences must have identical digests",
            "policies": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--policies", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.reference.read_text(encoding="utf-8")),
                     json.loads(args.candidate.read_text(encoding="utf-8")), args.policies.split(","))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
