#!/usr/bin/env python3
"""Collect explicit P0 source records against a previously verified contract.

No discovery by newest timestamp, job-success inference, or stage promotion is
performed. Keep every referenced source until downstream consumers close.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from stage_gates import evaluate, read_json, verify_receipt


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def collect(contract_receipt, runs):
    receipt_path = Path(contract_receipt).resolve()
    verified = verify_receipt(receipt_path, required_stage="CONTRACT")
    receipt = read_json(receipt_path.read_bytes())
    source = (receipt_path.parent / receipt["source_manifest"]["path"]).resolve()
    manifest = read_json(source.read_bytes())
    # Bind precisely the source bytes that were just verified, even if files
    # change between reading the receipt and assembling the new evidence.
    if reference(source)["sha256"] != receipt["source_manifest"]["sha256"]:
        raise ValueError("contract manifest changed during collection")
    manifest = copy.deepcopy(manifest)
    for item in [manifest["protocol"], *manifest["contract"].values()]:
        item["path"] = str((source.parent / item["path"]).resolve())
    protocol = read_json(Path(manifest["protocol"]["path"]).read_bytes())
    manifest["runs"] = []
    for path in runs:
        bound = reference(path)
        record = read_json(Path(bound["path"]).read_bytes())
        binding = record.get("config", {}).get("benchmark", {}).get("formal_protocol", {})
        role = binding.get("role")
        candidates = [case for case in protocol["p0_required_cases"]
                      if case["shape_id"] in binding.get("shape_ids", [])]
        if role not in ("preemption", "no_preemption") or len(candidates) != 1:
            raise ValueError("source must identify exactly one prescribed P0 model/shape and role")
        manifest["runs"].append(dict(candidates[0], role=role, **bound))
    result = evaluate(manifest, source.parent)
    if result["build"] != verified["build"] or result["protocol_sha256"] != verified["protocol_sha256"]:
        raise ValueError("contract identity changed during collection")
    return manifest, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-receipt", required=True, type=Path)
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest, result = collect(args.contract_receipt, args.runs)
    # One collection attempt has its own identity; never replace an old result.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "p0-evidence.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    result["source_manifest"] = reference(manifest_path)
    receipt_path = args.output_dir / "p0-receipt.json"
    receipt_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"receipt": str(receipt_path),
                      "stages": {name: row["status"] for name, row in result["stages"].items()}}))
    if result["stages"]["P0"]["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
