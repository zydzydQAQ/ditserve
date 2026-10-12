"""Render C++ summaries without recalculating core metrics."""

import csv
from pathlib import Path
from typing import Dict, Mapping

from .util import atomic_json


def _summary(result):
    for key in ("summary", "metrics"):
        if isinstance(result.get(key), dict):
            return result[key]
    return result


def _cell(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return "%.6g" % value
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_report(outdir, runs: Mapping[str, dict]) -> Dict[str, str]:
    """Write reports from policy -> summary, or policy -> {summary/metrics: ...}."""
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    summaries = {policy: _summary(result) for policy, result in runs.items()}
    paths = {name: destination / filename for name, filename in (
        ("json", "summary.json"), ("csv", "summary.csv"), ("markdown", "REPORT.md")
    )}
    metadata = {
        "notice": "Synthetic/reference results are not measured GPU performance or perceptual quality evidence. No research improvement threshold is established.",
        "runs": summaries,
    }
    atomic_json(paths["json"], metadata)
    fields = sorted({field for summary in summaries.values() for field in summary if field != "policy"})
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["policy"] + fields)
        writer.writeheader()
        for policy, summary in summaries.items():
            writer.writerow({**summary, "policy": policy})
    columns = (
        ("Policy", None), ("Provenance", "provenance"),
        ("Arrivals", "total_arrivals"), ("Completed", "completed"),
        ("Rejected", "rejected"), ("Timed out", "timed_out"),
        ("SLO attainment", "slo_attainment"), ("Goodput/s", "goodput_per_sec"),
        ("P99 latency ms", "p99_latency_ms"), ("Latency samples", "latency_sample_count"),
        ("GPU s/completed", "gpu_seconds_per_completed_request"),
    )
    lines = [
        "# Reference serving experiment report", "",
        "**Synthetic/reference results are not measured GPU performance.** Synthetic costs and arrivals demonstrate the experiment harness; they do not establish a 20% speedup, publication-level novelty, or any other research conclusion.", "",
        "`quality_ok` checks reference numerical parity only. It does not measure perceptual image/video quality. Unknown quality never counts as an SLO success.", "",
        "SLO attainment uses every arrival, including rejected and timed-out requests. Success requires completion by the absolute deadline and known passing reference parity. `deadline_misses` counts all requests without on-time completion; `completed_deadline_misses` counts only late completions.", "",
        "Latency percentiles use completed requests only (nearest rank), with the sample count shown. GPU seconds per completion include work spent on rejected/timed-out requests. Resource utilization uses aggregate GPU busy time divided by observation duration times GPU count. Undefined empty-population percentiles/costs are null/N/A, never NaN.", "",
        "| " + " | ".join(label for label, key in columns) + " |",
        "| " + " | ".join("---" for column in columns) + " |",
    ]
    for policy, summary in summaries.items():
        lines.append("| " + " | ".join(_cell(policy if key is None else summary.get(key)) for label, key in columns) + " |")
    if not summaries:
        lines.extend(["", "No experiment runs supplied."])
    lines.extend(["", "The JSON and CSV files contain the full metric definitions and denominators for each policy.", ""])
    paths["markdown"].write_text("\n".join(lines), encoding="utf-8")
    return {name: str(path.resolve()) for name, path in paths.items()}
