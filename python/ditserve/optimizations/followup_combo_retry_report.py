"""Readback-verified, two-file report for the independent Sana combo retry.

Historical failures are provenance, never extra retry samples. This module
starts no GPU work, changes no evidence, and grants no serving qualification.
"""

import hashlib
import json
from pathlib import Path
import tempfile

from . import followup_report as reporting


SCHEMA = "ditserve.followup_combo_retry.report.v1"
MARKER = "<!-- " + SCHEMA + " -->"
MAX_HISTORY_BYTES = 16 * 1024 * 1024


def _source(path):
    path = reporting._safe(path)
    before = path.stat()
    if not path.is_file():
        raise ValueError("bound retry evidence must be an ordinary file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("bound retry evidence changed while hashing")
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=before.st_size)


def _bindings(value, pointer=""):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield pointer, value
        else:
            for key, child in value.items():
                yield from _bindings(child, pointer + "/" + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _bindings(child, pointer + "/%d" % index)


def _append_source(report, source):
    existing = next((row for row in report["sources"] if row["path"] == source["path"]), None)
    if existing is not None:
        if existing != source:
            report["errors"].append("retry source changed between reads: " + source["path"])
        return
    report["sources"].append(source)
    report["tables"]["Sources"].append([source[key] for key in ("path", "sha256", "bytes")])


def _metadata(report, table, scope, value, source, phase="study", attempt=""):
    reporting._metadata(report["tables"][table], phase, attempt, scope, value, str(source), report["omissions"])


def _provenance(report, root, queue, study):
    for name in ("original_failure", "source_manifest", "controller"):
        if not isinstance(queue.get(name), dict):
            report["errors"].append("retry queue is missing " + name + " binding")
    if not isinstance(study.get("retry_of"), dict):
        report["errors"].append("retry study is missing retry_of binding")
    for label, record in (("queue", queue), ("study", study)):
        for pointer, bound in _bindings(record):
            try:
                path = Path(bound["path"])
                if not path.is_absolute():
                    raise ValueError("bound retry evidence must use absolute paths")
                actual = _source(path)
                _append_source(report, actual)
                if actual["sha256"] != bound["sha256"] or ("bytes" in bound and actual["bytes"] != bound["bytes"]):
                    raise ValueError("bound retry evidence hash or size changed")
                _metadata(report, "Config", "verified retry provenance " + label + pointer,
                    dict(binding=bound, verified=True), root / (label + ".json"))
            except (OSError, ValueError, TypeError, KeyError) as error:
                report["errors"].append(label + pointer + ": " + str(error))
    # Keep the failure itself in the archive, but never import its local or
    # online rows into this run. Source/ledger bindings preserve full history.
    failure = queue.get("original_failure", {})
    if isinstance(failure, dict) and failure.get("path"):
        try:
            path = reporting._safe(failure["path"])
            if path.stat().st_size > MAX_HISTORY_BYTES:
                raise ValueError("original failure metadata exceeds explicit 16 MiB bound")
            old = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(old, dict):
                raise ValueError("original failure must be a JSON object")
            selected = {key: old[key] for key in ("phase", "status", "decision", "verdict", "reason", "error",
                "supervisor_failure", "traceback", "returncode", "cleanup_confirmed") if key in old}
            charge = old.get("charge")
            if isinstance(charge, dict):
                selected["historical_charge"] = {key: charge[key] for key in ("session_id", "stage", "status",
                    "gpu_seconds", "elapsed_seconds", "gpu_count") if key in charge}
            selected["sample_policy"] = "historical failure retained separately; zero historical samples pooled into retry"
            _metadata(report, "Failures", "original failed attempt; provenance only", selected, path)
            _metadata(report, "Config", "historical failure identity; not retry measurements", selected, path)
        except (OSError, ValueError, TypeError) as error:
            report["errors"].append("original failure metadata: " + str(error))


def _authoritative_status(report, root, budget, reader):
    for phase in report["phases"]:
        name, attempt = phase["phase"], phase["attempt"]
        original = dict(worker_status=phase["status"], worker_verdict=phase.get("recorded_verdict"),
            worker_reported_gate_passed=phase.get("reported_gate_passed"))
        directory = root / name / attempt
        supervisor, supervisor_source = reader.read(directory / "supervisor.json", phase["errors"])
        terminal, terminal_source = reader.read(root / name / "terminal.json", phase["errors"])
        controls = [(value, source) for value, source in ((supervisor, supervisor_source), (terminal, terminal_source))
                    if isinstance(value, dict) and reporting._settled(value, budget)]
        if controls:
            authoritative, source = controls[-1]
            # A phase-level terminal may describe a different retry. It is only
            # authoritative for this attempt when its charge agrees.
            if len(controls) == 2 and controls[0][0].get("charge") != authoritative.get("charge"):
                phase["errors"].append("settled supervisor and terminal charges disagree")
                controls = []
            elif len(controls) == 2 and any(controls[0][0].get(key) != authoritative.get(key)
                    for key in ("status", "returncode", "supervisor_failure")):
                phase["errors"].append("settled supervisor and terminal outcomes disagree")
                controls = []
        if controls:
            original.update(authoritative_source=source["path"], authoritative_status=authoritative.get("status"))
            phase["accounting_settled"] = True
            phase["supervisor_status"] = authoritative.get("status")
            complete = (authoritative.get("status") in ("completed", "passed", "stopped") and
                authoritative.get("returncode") == 0 and not authoritative.get("supervisor_failure") and
                authoritative.get("charge", {}).get("status") in ("completed", "passed"))
            if not complete or original["worker_status"] not in reporting.TERMINAL:
                phase.update(status="incomplete", recorded_verdict="未验证", dependencies_closed=False)
                original["display_reason"] = "settled supervisor ended the attempt without a complete worker result"
            elif original["worker_verdict"] == "值得继续" and (phase.get("online_status") != "completed"
                    or phase.get("recomputed_gate_passed") is not True or phase["errors"]):
                phase.update(recorded_verdict="未验证", dependencies_closed=False)
                original["display_reason"] = "recorded positive verdict lacks complete, consistent online evidence"
        else:
            phase["accounting_settled"] = False
            phase["dependencies_closed"] = False
            phase["recorded_verdict"] = "未验证"
            if original["worker_status"] in reporting.TERMINAL:
                phase["status"] = "unsettled"
            original["display_reason"] = "no matching settled supervisor or terminal; no scientific promotion"
        phase["worker_metadata"] = original
        _metadata(report, "Metrics", "worker status retained; display follows settled supervision", original,
            directory / "phase-result.json", name, attempt)
        for value, source in ((supervisor, supervisor_source), (terminal, terminal_source)):
            if source:
                _append_source(report, source)
                _metadata(report, "Budget", "retry outcome authority", value, source["path"], name, attempt)
    columns = report["tables"]["Summary"][0]
    report["tables"]["Summary"] = [columns] + [[phase.get(key) if key != "errors" else " | ".join(phase["errors"])
        for key in columns] for phase in report["phases"]]
    if not all(phase["dependencies_closed"] for phase in report["phases"]):
        report["dependencies_closed"] = False
    for phase in report["phases"]:
        if not phase["dependencies_closed"]:
            attempt_path = root / phase["phase"] / phase["attempt"]
            for item in report["retention"]:
                if attempt_path in Path(item["path"]).parents:
                    item["dependencies_closed"] = False
    columns = report["tables"]["Retention"][0]
    report["tables"]["Retention"] = [columns] + [[item.get(key) for key in columns] for item in report["retention"]]


def collect(work_root, *, ledger):
    root = reporting._safe(work_root)
    report = reporting.collect(root, ledger=ledger)
    reader = reporting._Reader()
    queue, _ = reader.read(root / "queue.json", report["errors"], required=True)
    study, _ = reader.read(root / "study.json", report["errors"], required=True)
    budget, _ = reader.read(reporting._safe(ledger), report["errors"], required=True)
    queue = reporting._obj(queue, report["errors"], "retry queue")
    study = reporting._obj(study, report["errors"], "retry study")
    budget = reporting._obj(budget, report["errors"], "retry ledger")
    _provenance(report, root, queue, study)
    _authoritative_status(report, root, budget, reader)
    for source in reader.sources.values():
        _append_source(report, source)
    report["schema_version"] = SCHEMA
    if report["status"] in ("completed", "finalizing") and (report["active_unsettled"] or report["errors"] or
            any(phase["status"] in ("incomplete", "unsettled") or not phase["accounting_settled"] or phase["errors"]
                for phase in report["phases"]) or not report["phases"]):
        report["status"] = "incomplete"
    if report["errors"]:
        report["dependencies_closed"] = False
    _metadata(report, "Failures", "retry report validation errors", report["errors"], root)
    for name, rows in report["tables"].items():
        if len(rows) > reporting.MAX_ROWS:
            raise ValueError("Excel sheet exceeds explicit row bound: " + name)
        report["tables"][name] = [[None if value == "" else value for value in row] for row in rows]
    return report


def write(work_root, output, *, ledger):
    """Publish only the workbook and report, preserving every source artifact."""
    root, destination = reporting._safe(work_root), reporting._safe(output)
    if root == destination or root in destination.parents or destination in root.parents:
        raise ValueError("work and final output directories must be disjoint")
    destination.mkdir(parents=True, exist_ok=True)
    markdown, excel = destination / "REPORT.md", destination / "results.xlsx"
    owner = "<!-- work-root-sha256: " + hashlib.sha256(str(root).encode("utf-8")).hexdigest() + " -->"
    if markdown.is_symlink() or excel.is_symlink():
        raise ValueError("retry artifacts must not be symlinks")
    if {path.name for path in destination.iterdir()} - {"results.xlsx", "REPORT.md"}:
        raise ValueError("retry output contains unrelated files; refusing overwrite")
    if markdown.exists() and not markdown.read_text(encoding="utf-8").startswith(MARKER + "\n" + owner + "\n"):
        raise ValueError("existing report belongs to another experiment; refusing overwrite")
    if excel.exists() and not markdown.exists():
        raise ValueError("existing workbook has no matching retry report")
    report = collect(root, ledger=ledger)
    with tempfile.TemporaryDirectory(prefix=".combo-retry-report-", dir=str(destination)) as temporary:
        temporary = Path(temporary)
        verification = reporting._write_excel(temporary / "results.xlsx", report["tables"])
        text = reporting.render(report).replace(reporting.MARKER, MARKER + "\n" + owner, 1)
        text += ("\n本次是独立的 Sana VAE 编译＋Transformer CUDA Graph 补跑：从头执行局部质量与性能检查、"
            "真实 Sana 插入后的 Wan 恢复检查，以及计划的 8 次在线配对重放；实际完成数以上表为准。"
            "历史失败及其费用独立保留，不与本次样本合并。retry_of、父账本、源码快照和仅存储路径变更"
            "的来源绑定见 Config／Sources 表。\n\n阶段显示状态以已结算的监督记录为准；被终止后遗留的 running "
            "不会表示仍在执行。原 worker 状态与判定保留在 Metrics 表，在线门槛值保持原样；"
            "未结算、恢复或在线未完成时仍为未验证，不升级为默认基线。\n")
        (temporary / "REPORT.md").write_text(text, encoding="utf-8")
        if (temporary / "REPORT.md").read_text(encoding="utf-8") != text:
            raise ValueError("retry report readback differs")
        (temporary / "results.xlsx").replace(excel)
        (temporary / "REPORT.md").replace(markdown)
    return dict(status=report["status"], excel=dict(path=str(excel), sha256=hashlib.sha256(excel.read_bytes()).hexdigest(), **verification),
        markdown=str(markdown), errors=report["errors"] + [error for phase in report["phases"] for error in phase["errors"]],
        sources=report["sources"], retention=report["retention"], phases=len(report["phases"]),
        requests=len(report["tables"]["Requests"]) - 1, paired_requests=len(report["tables"]["PairedRequests"]) - 1,
        work_bytes=report["work_bytes"], output_bytes=excel.stat().st_size + markdown.stat().st_size,
        dependencies_closed=report["dependencies_closed"], omissions=report["omissions"], raw_retained=report["raw_retained"],
        prior_cleanup_deleted=report["prior_cleanup_deleted"], prior_cleanup_rewrites=report["prior_cleanup_rewrites"],
        originals_deleted=0, baseline_promotion=False, historical_samples_pooled=0)
