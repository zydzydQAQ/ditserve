#!/usr/bin/env python3
"""Build paper rows from whole native policy cells, never field-selected sources."""

import argparse
import hashlib
import json
from pathlib import Path

from native_exact import validate_run
from native_metrics import summarize_run


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def selected_cell(record, policy):
    benchmark = validate_run(record)
    matches = []
    for phase in record.get("phases", []):
        for cell in phase.get("cells", []):
            if cell.get("policy") == policy:
                matches.append((phase.get("id", "cluster"), cell))
    if len(matches) != 1:
        raise ValueError("expected exactly one native policy cell for " + policy)
    phase, cell = matches[0]
    if cell.get("status") != "completed":
        raise ValueError("policy cell did not complete: " + policy)
    return phase, cell, benchmark


def tex_escape(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def number(value, digits=2):
    return "---" if value is None else f"{value:.{digits}f}"


def seconds(value):
    return number(None if value is None else value / 1000)


def sar_text(metrics):
    attainment = metrics["deadline_attainment"]
    percent = "---" if attainment is None else f"{100 * attainment:.1f}\\%"
    return f'{metrics["ontime"]}/{metrics["requests"]} ({percent})'


def tex_row(row):
    cell, label = row["metrics"], tex_escape(row["label"])
    video, image = cell["by_kind"]["video"], cell["by_kind"]["image"]
    stop = cell["preemption_stop_ms"]
    preempt = f'{cell["preempting_request_count"]} / {stop["count"]}'
    values = [
        label, sar_text(video), sar_text(image),
        f'{seconds(video["latency_ms"]["p50"])} / {seconds(video["latency_ms"]["p95"])}',
        f'{seconds(image["latency_ms"]["p50"])} / {seconds(image["latency_ms"]["p99"])}',
        preempt, f'{number(stop["p50"])} / {number(stop["p95"])}',
        number(cell["gpu_seconds_per_ontime_request"]),
    ]
    return " & ".join(values) + r" \\"


def paper_row(manifest_dir, specification):
    if set(specification) != {"label", "run", "policy"}:
        raise ValueError("each paper row must contain only label, run and policy")
    label, policy = specification["label"], specification["policy"]
    if not isinstance(label, str) or not label or not isinstance(policy, str) or not policy:
        raise ValueError("paper row label and policy must be nonempty strings")
    path = (manifest_dir / specification["run"]).resolve()
    raw = path.read_bytes()
    record = json.loads(raw)
    phase, identity, benchmark = selected_cell(record, policy)
    summary = summarize_run(record)
    matches = [cell for cell in summary["cells"] if cell["phase"] == phase and cell["cell"] == identity["cell"]]
    if len(matches) != 1:
        raise ValueError("native summary does not identify the selected policy cell")
    metrics = {key: value for key, value in matches[0].items() if key != "failures"}
    metrics["failure_count"] = len(matches[0]["failures"])
    row = {
        "label": label,
        "metrics": metrics,
        "source": {
            "run": str(path), "run_sha256": hashlib.sha256(raw).hexdigest(),
            "experiment": record["experiment"], "run_id": record["run_id"],
            "config_sha256": record.get("config_sha256"),
            "core_sha256": record["source"]["core_sha256"],
            "source_sha256": canonical_hash(record["source"]),
            "phase": phase, "cell": identity["cell"], "policy": policy,
            "benchmark": benchmark,
        },
    }
    row["tex"] = tex_row(row)
    return row


def build(manifest, manifest_path):
    if not isinstance(manifest, dict) or manifest.get("schema") != "ditserve.paper-input.v1":
        raise ValueError("paper input must use schema ditserve.paper-input.v1")
    specifications = manifest.get("rows")
    if not isinstance(specifications, list) or not specifications:
        raise ValueError("paper input requires nonempty rows")
    labels = [row.get("label") for row in specifications if isinstance(row, dict)]
    if len(labels) != len(specifications) or len(set(labels)) != len(labels):
        raise ValueError("paper row labels must be present and unique")
    rows = [paper_row(manifest_path.parent, row) for row in specifications]
    return {
        "schema": "ditserve.paper-results.v1",
        "scope": "one complete native policy cell per row; no field-level source composition; not scientific qualification",
        "manifest_sha256": file_hash(manifest_path),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path)
    args = parser.parse_args()
    result = build(json.loads(args.manifest.read_text(encoding="utf-8")), args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    if args.tex_output:
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        with args.tex_output.open("x", encoding="utf-8") as stream:
            stream.write("% Generated from ditserve.paper-results.v1; one run/policy cell per row.\n")
            for row in result["rows"]:
                stream.write(row["tex"] + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(result["rows"]),
                      "tex_output": str(args.tex_output) if args.tex_output else None}, indent=2))


if __name__ == "__main__":
    main()
