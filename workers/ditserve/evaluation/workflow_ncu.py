"""Import existing Nsight Compute reports; never launch a GPU workload.

    python -m ditserve.evaluation.workflow_ncu --report stage.ncu-rep --out stage.json
    python -m ditserve.evaluation.workflow_ncu --csv raw.csv --nvtx-csv renamed.csv --out stage.json

The output describes sampled kernels, not whole-stage occupancy. Kernel replay
changes execution, caches and concurrency, so these timings are diagnostic only.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess

from .workflow_trace import parse_label


METRICS = {
    "gpu__time_duration.sum": ("duration_ns", "time"),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("achieved_occupancy_percent", "percent"),
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed": ("sm_active_percent", "percent"),
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": ("sm_throughput_percent", "percent"),
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": ("dram_throughput_percent", "percent"),
    "dram__bytes_read.sum": ("dram_read_bytes", "bytes"),
    "dram__bytes_write.sum": ("dram_write_bytes", "bytes"),
}
DEFAULT_NCU = "/opt/nvidia/nsight-compute/2024.1.0/ncu"
WORKFLOW_JSON = re.compile(r"(?:ditserve[.]workflow|workflow)/[^|\r\n\"]+\|(?=\{)")


def _value(text):
    text = str(text).strip().replace(",", "")
    try:
        value = float(text)
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def normalize_value(value, unit, kind):
    value = _value(value)
    if value is None:
        return None
    unit = unit.strip()
    if kind == "time":
        factor = {"nsecond": 1, "ns": 1, "usecond": 1000, "us": 1000, "µsecond": 1000, "µs": 1000,
                  "msecond": 1e6, "ms": 1e6, "second": 1e9, "s": 1e9}.get(unit)
    elif kind == "bytes":
        factor = {"byte": 1, "bytes": 1, "B": 1, "Kbyte": 1000, "Mbyte": 1e6, "Gbyte": 1e9}.get(unit)
    else:
        factor = 1 if unit in {"%", "percent"} else None
    if factor is None:
        raise ValueError("unsupported %s unit %r" % (kind, unit))
    return value * factor


def _repair_workflow_json(text):
    """Repair NCU 2024.1's unescaped JSON inside otherwise quoted CSV fields.

    NVIDIA escapes the surrounding NVTX stack quotes, but leaves the quotes in
    our JSON metadata bare, including in renamed kernel names. Only complete
    JSON objects immediately after a workflow label are repaired. Already valid
    CSV has doubled JSON quotes and does not match JSONDecoder's grammar.
    """
    decoder, repairs = json.JSONDecoder(), []
    for match in WORKFLOW_JSON.finditer(text):
        try:
            value, end = decoder.raw_decode(text, match.end())
        except ValueError:
            continue
        original = text[match.end():end]
        if isinstance(value, dict) and '"' in original:
            repairs.append((match.end(), end, original.replace('"', '""')))
    for start, end, replacement in reversed(repairs):
        text = text[:start] + replacement + text[end:]
    return text, len(repairs)


def _nvtx_column(name):
    return "nvtx" in name.lower() or "Push/Pop_Range" in name or "Start/Stop_Range" in name


def _csv_rows(text):
    text, repaired = _repair_workflow_json(text)
    reader, rows, warnings = csv.reader(io.StringIO(text), strict=True), [], []
    if repaired:
        warnings.append("Repaired %d unescaped workflow JSON objects in NVIDIA CSV fields." % repaired)
    while True:
        try:
            rows.append(next(reader))
        except StopIteration:
            break
        except csv.Error as error:
            warnings.append("Skipped malformed CSV record near line %d: %s" % (reader.line_num, error))
    index = next((index for index, row in enumerate(rows) if "ID" in row and "Kernel Name" in row), None)
    if index is None:
        return [], warnings + ["No raw kernel CSV header found; capture may contain no matching kernels."]
    header, remainder = rows[index], rows[index + 1:]
    if len(set(header)) != len(header):
        raise ValueError("duplicate raw CSV column names")
    warnings.extend(" ".join(row) for row in rows[:index] if row)
    parsed = []

    def kernel_row(fields):
        if len(fields) != len(header):
            if fields:
                warnings.append("Skipped non-kernel CSV row with %d fields; expected %d: %s" %
                                (len(fields), len(header), " ".join(fields)[:180]))
            return None
        row = dict(zip(header, fields))
        if not row.get("ID", "").isdigit() or not row.get("Kernel Name"):
            return None
        if any(name in row and not row[name].isdigit() for name in ("Process ID", "Device", "Context", "Stream")):
            warnings.append("Skipped kernel row with invalid numeric identity: " + str(_identity(row)))
            return None
        return row

    if "Metric Name" in header:
        grouped = {}
        for fields in remainder:
            row = kernel_row(fields)
            if row is None:
                continue
            key = _identity(row)
            record = grouped.setdefault(key, dict(identity=row, raw_metrics={}, nvtx={}))
            if record["identity"]["Kernel Name"] != row["Kernel Name"]:
                raise ValueError("conflicting kernel names for identity " + str(key))
            metric = dict(value=row.get("Metric Value"), unit=row.get("Metric Unit", ""))
            previous = record["raw_metrics"].get(row["Metric Name"])
            if previous is not None and previous != metric:
                raise ValueError("conflicting metric for kernel identity " + str(key))
            record["raw_metrics"][row["Metric Name"]] = metric
            record["nvtx"].update({key: value for key, value in row.items() if _nvtx_column(key) and value})
        parsed = list(grouped.values())
    else:
        units = {}
        if remainder and len(remainder[0]) == len(header):
            candidate = dict(zip(header, remainder[0]))
            if candidate.get("ID", "") == "":
                units = candidate
                remainder = remainder[1:]
        for fields in remainder:
            row = kernel_row(fields)
            if row is None:
                continue
            parsed.append(dict(identity=row, raw_metrics={name: dict(value=row[name], unit=units.get(name, ""))
                                                          for name in METRICS if name in row},
                               nvtx={key: value for key, value in row.items() if _nvtx_column(key) and value}))
    identities = [_identity(record["identity"]) for record in parsed]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate kernel identities in raw CSV")
    return parsed, warnings


def _identity(row):
    return tuple(row.get(name, "") for name in ("ID", "Process ID", "Device", "Context", "Stream"))


def _workflow_label(text):
    """Extract one full label from an NVTX stack or label/original-kernel name."""
    labels = set()
    for match in WORKFLOW_JSON.finditer(text):
        try:
            metadata, end = json.JSONDecoder().raw_decode(text, match.end())
        except ValueError:
            continue
        if isinstance(metadata, dict):
            labels.add(text[match.start():end])
    return next(iter(labels)) if len(labels) == 1 else None


def parse_csv(text, nvtx_csv=None, *, declared_stage=None, declared_variant=None, scope="sampled_kernels", expected_kernels=None):
    records, warnings = _csv_rows(text)
    renamed = {}
    if nvtx_csv:
        names, name_warnings = _csv_rows(nvtx_csv)
        warnings.extend(name_warnings)
        renamed = {_identity(row["identity"]): row for row in names}
    kernels = []
    for record in records:
        row = record["identity"]
        labels = list(record["nvtx"].values())
        name_record = renamed.get(_identity(row))
        if name_record:
            # A separate export must describe exactly the same kernel metrics;
            # an ID alone can collide across devices, processes or captures.
            common = record["raw_metrics"].keys() & name_record["raw_metrics"].keys()
            name = name_record["identity"].get("Kernel Name", "")
            renamed_label = _workflow_label(name)
            suffix = name[len(renamed_label):] if renamed_label and name.startswith(renamed_label) else None
            if (any(record["raw_metrics"][key] != name_record["raw_metrics"][key] for key in common)
                    or (suffix not in (None, "", "/" + row["Kernel Name"]))):
                warnings.append("Ignored mismatched NVTX rename export for kernel " + str(_identity(row)))
            else:
                labels.append(name)
        found_labels = {label for value in labels if (label := _workflow_label(value))}
        if len(found_labels) > 1:
            warnings.append("Conflicting NVTX labels for kernel " + str(_identity(row)))
        label = next(iter(found_labels)) if len(found_labels) == 1 else None
        parsed = parse_label(label) if label else None
        metrics, errors = {}, []
        for name, (key, kind) in METRICS.items():
            if name not in record["raw_metrics"]:
                metrics[key] = None
                continue
            raw = record["raw_metrics"][name]
            try:
                metrics[key] = normalize_value(raw["value"], raw["unit"], kind)
            except ValueError as error:
                metrics[key] = None
                errors.append(str(error))
        kernels.append(dict(id=row.get("ID"), process_id=row.get("Process ID"), device=row.get("Device"),
                            context=row.get("Context"), stream=row.get("Stream"), kernel_name=row.get("Kernel Name"),
                            nvtx_label=label, nvtx_columns=record["nvtx"],
                            stage=parsed["name"] if parsed else declared_stage,
                            metadata=parsed["metadata"] if parsed else dict(variant=declared_variant) if declared_variant else {},
                            association="NVTX rename/export" if parsed else "declared capture filter" if declared_stage else "unattributed",
                            included=parsed["included"] if parsed else True,
                            metrics=metrics, metric_errors=errors))
    groups = defaultdict(list)
    for kernel in kernels:
        if kernel["included"]:
            groups[(kernel["stage"], kernel["metadata"].get("variant"), kernel["device"])].append(kernel)
    summaries = []
    for (stage, variant, device), selected in sorted(groups.items(), key=lambda item: str(item[0])):
        timed = [row for row in selected if row["metrics"].get("duration_ns") is not None and row["metrics"]["duration_ns"] > 0]
        duration = sum(row["metrics"]["duration_ns"] for row in timed)
        metrics = {}
        for key, kind in METRICS.values():
            if kind == "time":
                continue
            measured = [row for row in timed if row["metrics"].get(key) is not None]
            covered = sum(row["metrics"]["duration_ns"] for row in measured)
            if kind == "percent":
                metrics[key] = dict(duration_weighted_mean=sum(row["metrics"][key] * row["metrics"]["duration_ns"] for row in measured) / covered if covered else None,
                                    measured_kernel_count=len(measured), measured_duration_ns=covered,
                                    fraction_of_profiled_kernel_duration=covered / duration if duration else None)
            else:
                total = sum(row["metrics"][key] for row in measured) if measured else None
                metrics[key] = dict(sum=total, measured_kernel_count=len(measured), measured_duration_ns=covered,
                                    effective_gb_s=total / covered if total is not None and covered else None)
        summaries.append(dict(stage=stage, variant=variant, device=device, profiled_kernel_count=len(selected),
                              timed_kernel_count=len(timed), summed_profiled_kernel_duration_ns=duration,
                              metrics=metrics, stage_time_coverage_fraction=None,
                              association=sorted({row["association"] for row in selected}), scope=scope))
    included = sum(row["included"] for row in kernels)
    if expected_kernels is not None and (expected_kernels <= 0 or included > expected_kernels):
        raise ValueError("expected_kernels must be positive and at least the included profiled count")
    if not kernels:
        warnings.append("No profiled kernels were imported.")
    return dict(schema="ditserve.workflow.ncu.v1", status="completed" if kernels else "empty",
                scope=scope, kernels=kernels, summaries=summaries, warnings=warnings,
                coverage=dict(profiled_kernel_count=len(kernels), included_profiled_kernel_count=included,
                              expected_selected_kernel_count=expected_kernels,
                              selected_kernel_count_fraction=included / expected_kernels if expected_kernels else None,
                              full_stage_time_fraction=None),
                interpretation=["Achieved occupancy uses active cycles, distinct from elapsed-cycle warp slots in Nsight Systems.",
                                "Aggregates weight per-kernel percentages by gpu__time_duration.sum, not active SM cycles; they are descriptive sampled-kernel averages.",
                                "NCU kernel replay may serialize work and modify caches/clocks. Summed diagnostic kernel durations are not stage latency or throughput.",
                                "Missing metrics are null; sampled kernels do not establish whole-stage occupancy or duration coverage.",
                                "selected_nvtx_range scope describes the declared capture selection, not coverage of an entire multi-step request.",
                                "DRAM effective GB/s uses measured bytes divided by measured kernel time; GB is decimal 1e9 bytes."])


def nvtx_filter(stage, *, variant, purpose, repeat=0, step=None):
    """Installed NCU docs: regex prefix plus '/' suffix denotes a push/pop range.

    '.' deliberately matches the literal slash in our known single range label,
    avoiding confusion with NCU's slash-separated NVTX stack syntax.
    """
    if stage not in {"text_encoder", "dit_step", "vae_decode"}:
        raise ValueError("unsupported stage")
    if not re.fullmatch(r"T[0-5]", variant) or purpose not in {"encoder_measurement", "video_measurement"}:
        raise ValueError("invalid workflow variant or purpose")
    fields = dict(purpose=purpose, repeat=repeat, variant=variant)
    if step is not None:
        fields["step"] = step
    pattern = "ditserve[.]workflow." + stage
    for key, value in sorted(fields.items()):
        pattern += '.*"' + key + '":' + json.dumps(value)
        if isinstance(value, int):
            pattern += "[^0-9]"
    return "regex:" + pattern + ".*/"


def profile_options(stage, *, variant, purpose, repeat=0, step=None, launch_count=None):
    """Return arguments only; caller explicitly chooses and launches the workload."""
    args = ["--nvtx", "--nvtx-include", nvtx_filter(stage, variant=variant, purpose=purpose, repeat=repeat, step=step),
            "--range-filter", "::1", "--replay-mode", "kernel", "--clock-control", "none", "--cache-control", "none",
            "--metrics", ",".join(METRICS), "--disable-extra-suffixes", "--target-processes", "all"]
    if launch_count is not None:
        if launch_count < 1:
            raise ValueError("launch_count must be positive")
        args.extend(["--launch-count", str(launch_count)])
    return args


def _export(ncu, report, rename=False):
    # NCU 2024.1's explicit base-unit formatting rounds percentages to 2 decimals;
    # auto preserves 6 decimals. The parser reads and normalizes each CSV unit.
    command = [str(ncu), "--import", str(report), "--csv", "--page", "raw", "--print-units", "auto", "--print-fp"]
    if rename:
        command += ["--print-nvtx-rename", "kernel"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout, result.stderr, command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path)
    source.add_argument("--csv", type=Path)
    parser.add_argument("--nvtx-csv", type=Path)
    parser.add_argument("--ncu", type=Path, default=Path(DEFAULT_NCU))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage")
    parser.add_argument("--variant")
    parser.add_argument("--scope", choices=("sampled_kernels", "selected_nvtx_range"), default="sampled_kernels")
    parser.add_argument("--expected-kernels", type=int)
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    commands, stderr = [], []
    if args.report:
        raw, errors, command = _export(args.ncu, args.report)
        renamed, rename_errors, rename_command = _export(args.ncu, args.report, True)
        commands, stderr = [command, rename_command], [errors, rename_errors]
        raw_path, rename_path = args.out.with_suffix(".raw.csv"), args.out.with_suffix(".nvtx.csv")
        for path, text in ((raw_path, raw), (rename_path, renamed)):
            with path.open("x") as stream:
                stream.write(text)
    else:
        raw, raw_path = args.csv.read_text(), args.csv
        renamed, rename_path = (args.nvtx_csv.read_text(), args.nvtx_csv) if args.nvtx_csv else (None, None)
    value = parse_csv(raw, renamed, declared_stage=args.stage, declared_variant=args.variant, scope=args.scope,
                      expected_kernels=args.expected_kernels)
    value["source"] = dict(report=str(args.report) if args.report else None, csv=str(raw_path),
                           nvtx_csv=str(rename_path) if rename_path else None,
                           csv_sha256=hashlib.sha256(raw.encode()).hexdigest(), export_commands=commands, stderr=stderr)
    with args.out.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(out=str(args.out), status=value["status"], coverage=value["coverage"])))


if __name__ == "__main__":
    main()
