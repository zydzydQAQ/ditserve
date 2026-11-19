"""Verify DITSERVE reports, retain key summaries, and retire explicit output directories."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone

MANIFEST_SCHEMA = "ditserve.output-retention-reports.v1"
PLAN_SCHEMA = "ditserve.output-retention-plan.v1"
INDEX_SCHEMA = "ditserve.output-retention-index.v1"
SUMMARY_SCHEMA = "ditserve.output-retention-summary.v1"
RESULT_SCHEMA = "ditserve.results.v1"
RESULT_SUMMARY_SCHEMA = "ditserve.results-summary.v1"
DATED_REPORT = re.compile(r"[0-9]{8}-[A-Za-z0-9][A-Za-z0-9_.-]*\.md")
IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
SENSITIVE = ("PASSWORD", "TOKEN", "SECRET", "CREDENTIAL", "PRIVATE_KEY", "ACCESS_KEY", "AUTH_TOKEN")
LOG_ERROR = re.compile(r"error|failed|failure|exception|traceback|out of memory|\boom\b|killed|abort|timeout|permission denied", re.I)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def real(path, exists=True):
    value = Path(os.path.abspath(os.fspath(path)))
    if value.resolve(strict=exists) != value:
        raise ValueError("symbolic links are not allowed: " + str(value))
    return value


def output_root(path):
    value = real(path)
    if value.name != "DITSERVE" or value.parent.name != "output" or not value.is_dir():
        raise ValueError("root must be a real output/DITSERVE directory")
    return value


def report_directory(root, path):
    value = real(path)
    if value.parent != root / "reports" or not re.fullmatch(r"(?:\d{8}-)?output-retention-.+", value.name):
        raise ValueError("report directory must be output/DITSERVE/reports/[YYYYMMDD-]output-retention-*")
    return value


def relative(root, path):
    value = str(path.relative_to(root))
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("unsafe relative path: " + value)
    return value


def inside(root, value):
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("unsafe retained path: " + value)
    return real(root / pure)


def write_json(path, value, replace=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(str(path))
    descriptor, temporary = tempfile.mkstemp(prefix=".retention-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and not replace:
            raise FileExistsError(str(path))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_bytes(path, limit=512 * 1024 * 1024):
    path = real(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("not a bounded regular file: " + str(path))
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > limit or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("file changed or exceeds limit: " + str(path))
    return raw


def file_hash(path):
    path = real(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("not a regular file: " + str(path))
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("file changed while hashing: " + str(path))
    return digest.hexdigest()


def read_json(path, limit=512 * 1024 * 1024):
    raw = read_bytes(path, limit)
    value = json.loads(raw)
    canonical(value)
    return value, {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def redact(value, key=""):
    if any(word in key.upper() for word in SENSITIVE):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def inventory(root):
    files, errors, links = [], [], []
    logical = allocated = 0
    def onerror(error):
        errors.append(str(error))
    for directory, dirnames, filenames in os.walk(root, followlinks=False, onerror=onerror):
        parent = Path(directory)
        for name in list(dirnames):
            if (parent / name).is_symlink():
                links.append(name)
                dirnames.remove(name)
        for name in filenames:
            path = parent / name
            if path.is_symlink():
                links.append(str(path.relative_to(root)))
                continue
            try:
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    errors.append("non-regular file: " + str(path))
                    continue
                files.append([str(path.relative_to(root)), info.st_size, info.st_mtime_ns, info.st_blocks])
                logical += info.st_size
                allocated += info.st_blocks * 512
            except OSError as error:
                errors.append(str(error))
    return {"files": sorted(files), "bytes": logical, "allocated_bytes": allocated,
            "errors": sorted(errors), "symlinks": sorted(links)}


def children(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted((entry for entry in os.scandir(directory) if entry.is_dir(follow_symlinks=False)), key=lambda e: e.name)


def closed(dependencies):
    return isinstance(dependencies, list) and all(isinstance(item, dict) and item.get("closed") is True
        and isinstance(item.get("reason"), str) and item["reason"].strip() for item in dependencies)


def formal_document(data):
    """Recognize ongoing scientific consumers independently of run completion."""
    if not isinstance(data, dict):
        return False
    if data.get("schema") in {"ditserve.stage-evidence.v1", "ditserve.stage-gates.v1",
                              "ditserve.formal-protocol.v1", "ditserve.shape-profiles.v1"}:
        return True
    # Acceptance directories hold generated configurations directly, whereas
    # results-work/run.json nests the same configuration under config.
    for config in (data, data.get("config", {})):
        if isinstance(config, dict) and isinstance(config.get("benchmark"), dict):
            if "formal_protocol" in config["benchmark"]:
                return True
    # Retain even an interrupted contract that has not published a receipt yet.
    return data.get("schema") == "ditserve.runtime-smoke.v1" and any(
        key in data for key in ("image_digest", "execution_image", "image_job_spec_sha256"))


def protected_formal_source(path, observed=None):
    """Protect original sources until a verified archive migration is implemented.

    stage_gates.verify_receipt reopens these exact sources. A completed run,
    an empty pending_dependencies list, or a manually closed consumer therefore
    cannot authorize their deletion. Scan nested acceptance paths as well.
    """
    observed = inventory(path) if observed is None else observed
    for name, unused, unused_mtime, unused_blocks in observed["files"]:
        source = Path(name)
        if source.name in {"contract-evidence.json", "contract-receipt.json"}:
            return "formal stage-gate source remains a live verification dependency: " + name
        if source.suffix.lower() == ".json":
            data, unused_binding = read_json(path / source)
            if formal_document(data):
                return "formal benchmark/stage-gate source remains a live verification dependency: " + name
    return None


def report_files(item):
    return ("report", "summary") if item.get("format") == RESULT_SUMMARY_SCHEMA else ("report", "workbook")


def verified_report(root, directory, identity):
    from ditserve import results
    data = results.load(directory)
    if data.get("schema") not in (RESULT_SCHEMA, RESULT_SUMMARY_SCHEMA):
        raise ValueError("only verified native results can authorize work-directory retirement")
    metadata, finalization = data["run"], data["finalization"]
    if (metadata.get("experiment"), metadata.get("run_id")) != identity:
        raise ValueError("report identity differs from its directory")
    work = Path(finalization["work_dir"])
    if work.parts[-4:-2] != (".cache", "results-work") or work.parts[-2:] != identity or ".." in work.parts:
        raise ValueError("report is not bound to the matching results-work directory")
    binding = finalization["input"]
    if binding.get("path") != str(work / "run.json") or not SHA256.fullmatch(binding.get("sha256", "")) or type(binding.get("bytes")) is not int:
        raise ValueError("report lacks an exact run.json source binding")
    result = {"experiment": identity[0], "run_id": identity[1], "format": data["schema"],
              "verification": "verified", "run_status": metadata.get("status"),
              "archive_status": finalization.get("status"),
              "artifact_dependencies_closed": all(closed(item.get("dependencies")) for item in metadata.get("artifacts", []))
                  and not any(str(item.get("reason", "")).startswith("dependencies remain open")
                              for item in finalization.get("retained", [])),
              "pending_dependencies": finalization.get("pending_dependencies", metadata.get("retention", {}).get("pending_dependencies", [])),
              "run_json": {key: binding[key] for key in ("bytes", "sha256")}}
    if data["schema"] == RESULT_SUMMARY_SCHEMA:
        name = data.get("report_name", "")
        if not DATED_REPORT.fullmatch(name):
            raise ValueError("unsafe dated report name")
        files = {"report": name, "summary": "summary.json"}
    else:
        files = {"report": "REPORT.md", "workbook": "results.xlsx"}
    for field, name in files.items():
        path = real(directory / name)
        result.update({field: relative(root, path), field + "_bytes": path.stat().st_size,
                       field + "_sha256": file_hash(path)})
    return result


def checked_work(path, identity, archived):
    data, binding = read_json(path / "run.json")
    if data.get("schema") != "ditserve.run.v1" or (data.get("experiment"), data.get("run_id")) != identity:
        raise ValueError("work run identity differs from its report")
    if binding != archived["run_json"]:
        raise ValueError("run.json differs from the latest verified finalization input; finalize and regenerate manifest")
    reason = protected_formal_source(path)
    if reason:
        raise ValueError(reason + "; verified archive migration is required before retirement")
    if archived["archive_status"] != "completed" or not closed(archived["pending_dependencies"]) or not archived["artifact_dependencies_closed"]:
        raise ValueError("pending dependencies or deferred archive")
    if not closed(data.get("retention", {}).get("pending_dependencies", [])) or any(
            not closed(item.get("dependencies")) for item in data.get("artifacts", [])):
        raise ValueError("raw work still has open consumers")
    from ditserve import results
    results._active(data)
    ledger = data.get("budget", {}).get("ledger") or data.get("budget_ledger")
    if ledger:
        ledger_path = path / ledger
        if not ledger_path.exists() and path.parts[-4:-2] == (".cache", "results-work"):
            ledger_name = PurePosixPath(str(ledger)).name
            if ledger_name and ledger_name not in (".", ".."):
                fallback = path.parents[3] / "acceptance" / identity[1] / "ledgers" / ledger_name
                if fallback.exists():
                    ledger_path = fallback
        budget, unused = read_json(ledger_path)
        results._active({"budget": budget})
    return binding


def command_manifest(args):
    root = output_root(args.root)
    source_root = args.source_root or str(root)
    if PurePosixPath(source_root).parts[-2:] != ("output", "DITSERVE"):
        raise ValueError("source root must end in output/DITSERVE")
    if args.experiment and not IDENTITY.fullmatch(args.experiment):
        raise ValueError("unsafe experiment identity")
    reports, skipped = [], []
    seen = set()
    for experiment in children(root / "results"):
        if args.experiment and experiment.name != args.experiment:
            continue
        for run in children(experiment.path):
            if args.run_id and run.name != args.run_id:
                continue
            directory = real(run.path)
            identity = (experiment.name, run.name)
            try:
                if not all(IDENTITY.fullmatch(value) for value in identity) or identity in seen:
                    raise ValueError("unsafe or duplicate report identity")
                seen.add(identity)
                reports.append(verified_report(root, directory, identity))
            except Exception as error:
                skipped.append({"experiment": experiment.name, "run_id": run.name,
                                "reason": type(error).__name__ + ": " + str(error)})
            completed = len(reports) + len(skipped)
            if completed % 10 == 0:
                print(f"manifest_progress={completed} verified={len(reports)} skipped={len(skipped)}", flush=True)
    value = {"schema": MANIFEST_SCHEMA, "created_at": timestamp(), "source_root": source_root,
             "reports": reports, "skipped": skipped,
             "scope": "results.load validation, exact report plus summary/workbook hashes, and finalization input identity; no scientific qualification"}
    write_json(args.output, value, replace=False)
    print(json.dumps({"verified_reports": len(reports), "skipped": skipped, "output": str(args.output)}, ensure_ascii=False, indent=2))


def load_manifest(root, path):
    manifest, binding = read_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("source_root") != str(root):
        raise ValueError("manifest schema or output root mismatch")
    reports = {}
    for item in manifest["reports"]:
        identity = (item["experiment"], item["run_id"])
        if not all(IDENTITY.fullmatch(value) for value in identity) or identity in reports:
            raise ValueError("unsafe or duplicate manifest identity")
        if item.get("verification") != "verified":
            raise ValueError("unverified report in manifest")
        directory = root / "results" / identity[0] / identity[1]
        for field in report_files(item):
            name = PurePosixPath(item[field]).name
            if field == "report" and item.get("format") == RESULT_SUMMARY_SCHEMA:
                if not DATED_REPORT.fullmatch(name):
                    raise ValueError("unsafe dated report name")
                suffix = name
            else:
                suffix = {"report": "REPORT.md", "workbook": "results.xlsx", "summary": "summary.json"}[field]
            expected = str(PurePosixPath("results") / identity[0] / identity[1] / suffix)
            if item[field] != expected or not SHA256.fullmatch(item[field + "_sha256"]):
                raise ValueError("report path or digest mismatch")
            source = inside(root, item[field])
            if source.stat().st_size != item[field + "_bytes"] or file_hash(source) != item[field + "_sha256"]:
                raise ValueError("retained report changed: " + str(source))
        trusted = verified_report(root, directory, identity)
        for field in report_files(trusted):
            if item.get(field) != trusted[field] or item.get(field + "_sha256") != trusted[field + "_sha256"]:
                raise ValueError("manifest does not bind the verified result files")
        reports[identity] = trusted
    return manifest, reports, binding["sha256"]


def build_plan(root, manifest_path, discard_pending, run_id):
    if discard_pending:
        raise ValueError("--discard-pending is no longer supported; close consumers and re-finalize first")
    manifest, reports, manifest_sha256 = load_manifest(root, manifest_path)
    candidates, protected = [], []
    work = root / ".cache" / "results-work"
    for experiment in children(work):
        for run in children(experiment.path):
            if run_id and run.name != run_id:
                continue
            path = real(run.path)
            rel = relative(root, path)
            observed = inventory(path)
            base = {"experiment": experiment.name, "run_id": run.name, "path": rel, "inventory": observed}
            if observed["errors"] or observed["symlinks"] or (path / "ACTIVE").exists():
                protected.append(dict(base, reason="unreadable, symlinked, non-regular, or active work directory"))
                continue
            identity = (experiment.name, run.name)
            archived = reports.get(identity)
            final_directory = root / "results" / identity[0] / identity[1]
            if archived is None and final_directory.exists():
                protected.append(dict(base, reason="final report exists but was not verified"))
                continue
            if archived is not None:
                try:
                    binding = checked_work(path, identity, archived)
                except Exception as error:
                    protected.append(dict(base, reason="cannot retire work: " + str(error)))
                    continue
                candidates.append(dict(base, kind="verified_report", run_json=binding))
                continue
            protected.append(dict(base, reason="no verified final report; finalize the run before retirement"))
    acceptance = []
    for entry in children(root / "acceptance"):
        if run_id and entry.name != run_id:
            continue
        path = real(entry.path)
        observed = inventory(path)
        item = {"run_id": entry.name, "path": relative(root, path), "inventory": observed}
        if observed["errors"] or observed["symlinks"] or (path / "ACTIVE").exists():
            protected.append(dict(item, reason="unreadable, symlinked, non-regular, or active acceptance directory"))
            continue
        try:
            reason = protected_formal_source(path, observed)
        except Exception as error:
            reason = "cannot inspect possible formal sources: " + str(error)
        if reason:
            protected.append(dict(item, reason=reason))
            continue
        matches = [report for identity, report in reports.items() if identity[1] == entry.name]
        if (len(matches) != 1 or matches[0]["archive_status"] != "completed"
                or not matches[0]["artifact_dependencies_closed"] or not closed(matches[0]["pending_dependencies"])):
            protected.append(dict(item, reason="acceptance evidence has no unique closed verified run"))
            continue
        matched = matches[0]
        corresponding_work = work / matched["experiment"] / matched["run_id"]
        if corresponding_work.exists() and not any(candidate["path"] == relative(root, corresponding_work) for candidate in candidates):
            protected.append(dict(item, reason="corresponding work is protected"))
            continue
        hashes = {}
        try:
            for name, unused, unused_mtime, unused_blocks in observed["files"]:
                hashes[name] = file_hash(path / name)
        except Exception as error:
            protected.append(dict(item, reason="cannot hash acceptance evidence: " + str(error)))
            continue
        item["hashes"] = hashes
        acceptance.append(item)
    if run_id and not candidates and not acceptance and not protected:
        raise ValueError("no matching work or acceptance directory for run_id: " + run_id)
    return {"schema": PLAN_SCHEMA, "root": str(root), "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256, "discard_pending": bool(discard_pending), "run_id": run_id,
            "reports": list(reports.values()), "manifest_skipped": manifest.get("skipped", []),
            "work": candidates, "acceptance": acceptance, "protected": protected,
            "candidate_bytes": sum(item["inventory"]["bytes"] for item in candidates + acceptance),
            "original_recoverable": False,
            "notice": "Only verified reports and compact summaries remain; original arrays, sources and full logs cannot be restored."}


def phase_summary(phase):
    return {key: phase[key] for key in ("id", "status", "attempt", "error", "cleanup_error", "metrics", "qualification")
            if key in phase}


def failure_document(data, binding, source):
    requests = data.get("requests", [])
    summary = {"schema": SUMMARY_SCHEMA, "created_at": timestamp(), "source": source,
               "source_run_json": binding, "original_recoverable": False,
               "experiment": data["experiment"], "run_id": data["run_id"], "status": data.get("status"),
               "config_sha256": data.get("config_sha256"), "config": redact(data.get("config")),
               "source_provenance": data.get("source"), "metrics": data.get("metrics"),
               "phases": [phase_summary(item) for item in data.get("phases", [])],
               "failures": data.get("failures", []), "request_count": len(requests),
               "request_errors": [{key: item.get(key) for key in ("request_id", "status", "error") if key in item}
                                  for item in requests if item.get("error") or item.get("status") == "failed"],
               "artifacts": data.get("artifacts", []), "budget": data.get("budget"),
               "budget_ledger": data.get("budget_ledger"), "retention": data.get("retention"),
               "notice": "Key configuration, metrics, errors and artifact hashes only; original work directory is not recoverable."}
    canonical(summary)
    return summary


def numeric_summary(values):
    ordered = sorted(values)
    def percentile(fraction):
        if not ordered:
            return None
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)
    return {"count": len(values), "min": ordered[0], "max": ordered[-1],
            "mean": math.fsum(values) / len(values), "p50": percentile(.5), "p99": percentile(.99), "last": values[-1]}


def numeric_leaves(value, pointer=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numeric_leaves(item, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from numeric_leaves(item, pointer + "/" + str(index))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield pointer or "/", value


def text_excerpt(text):
    lines = text.splitlines()
    matches = [{"line": index, "text": line[:2000]} for index, line in enumerate(lines, 1) if LOG_ERROR.search(line)]
    return {"bytes": len(text.encode("utf-8")), "lines": len(lines),
            "head": lines[:16], "tail": lines[-32:], "error_excerpts": matches[:32]}


def jsonl_summary(text):
    values, errors = {}, []
    records = 0
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            for pointer, value in numeric_leaves(item):
                values.setdefault(pointer, []).append(value)
            records += 1
        except (ValueError, TypeError, RecursionError) as error:
            errors.append({"line": number, "error": str(error)})
    return {"records": records, "parse_errors": errors[:32],
            "numeric_metrics": {key: numeric_summary(items) for key, items in sorted(values.items())},
            "log_excerpt": text_excerpt(text)}


def csv_summary(text):
    rows = [row for row in csv.reader(io.StringIO(text)) if row]
    result = {"rows": len(rows)}
    if not rows:
        return result
    header = rows[0] if any("memory" in cell.lower() or "timestamp" in cell.lower() for cell in rows[0]) else None
    data = rows[1:] if header else rows
    names = header or ["timestamp", "uuid", "memory.used", "memory.free", "memory.total", "utilization.gpu"]
    for index, name in enumerate(names):
        values = []
        for row in data:
            if index >= len(row):
                continue
            try:
                value = float(row[index].strip().replace("MiB", "").replace("GiB", "").replace("%", "").strip())
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
        if values:
            result[name.strip() or str(index)] = numeric_summary(values)
    result["log_excerpt"] = text_excerpt(text)
    return result


def acceptance_document(path, item):
    documents, logs = {}, {}
    for name, expected in sorted(item["hashes"].items()):
        raw = read_bytes(path / name)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("acceptance source changed: " + name)
        suffix = Path(name).suffix.lower()
        if suffix == ".json":
            documents[name] = redact(json.loads(raw))
        elif suffix in (".log", ".txt"):
            logs[name] = text_excerpt(raw.decode("utf-8", "replace"))
        elif suffix == ".jsonl":
            logs[name] = jsonl_summary(raw.decode("utf-8", "replace"))
        elif suffix == ".csv":
            logs[name] = csv_summary(raw.decode("utf-8", "replace"))
    return {"schema": SUMMARY_SCHEMA, "created_at": timestamp(), "kind": "acceptance",
            "run_id": item["run_id"], "source": item["path"], "source_files": item["hashes"],
            "original_recoverable": False, "key_documents": documents, "log_and_metric_summaries": logs,
            "notice": "Configuration/build documents, bounded log excerpts and numeric summaries only; full logs are not retained."}


def save_summary(report_dir, category, identity, document):
    path = report_dir / category / (identity + ".json")
    write_json(path, document, replace=False)
    checked, unused = read_json(path)
    if checked != document:
        raise ValueError("summary readback mismatch: " + str(path))
    return path, file_hash(path)


def command_plan(args):
    root = output_root(args.root)
    destination = report_directory(root, args.report_dir)
    manifest = real(args.manifest)
    if manifest.parent != destination or manifest.name != "verified-reports.json":
        raise ValueError("verified-reports.json must be inside the new report directory")
    if (destination / "plan.json").exists() or (destination / "retention-index.json").exists() or list(destination.glob("*.md")):
        raise ValueError("retention plan already exists; inspect it instead of overwriting")
    plan = build_plan(root, manifest, args.discard_pending, args.run_id)
    plan["created_at"] = timestamp()
    plan["report_name"] = datetime.now(timezone.utc).strftime("%Y%m%d-output-retention.md")
    write_json(destination / "plan.json", plan, replace=False)
    print(json.dumps({"plan": str(destination / "plan.json"), "verified_reports": len(plan["reports"]),
                      "work_candidates": len(plan["work"]), "acceptance_candidates": len(plan["acceptance"]),
                      "candidate_bytes": plan["candidate_bytes"], "protected": len(plan["protected"])}, ensure_ascii=False, indent=2))


def command_apply(args):
    root = output_root(args.root)
    destination = report_directory(root, args.report_dir)
    plan, unused = read_json(destination / "plan.json")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("root") != str(root):
        raise ValueError("invalid retention plan")
    current = build_plan(root, real(plan["manifest"]), plan["discard_pending"], plan.get("run_id"))
    expected = {key: value for key, value in plan.items() if key not in ("created_at", "report_name")}
    if current != expected:
        raise RuntimeError("output changed after planning; no deletion was attempted")
    if (destination / "retention-index.json").exists():
        raise RuntimeError("an apply receipt already exists; inspect the partial result instead of reapplying")
    report_name = plan.get("report_name", datetime.now(timezone.utc).strftime("%Y%m%d-output-retention.md"))
    if not DATED_REPORT.fullmatch(report_name) or list(destination.glob("*.md")):
        raise RuntimeError("unsafe report name or existing report; no historical report will be overwritten")
    index = {"schema": INDEX_SCHEMA, "created_at": timestamp(), "status": "in_progress",
             "plan_sha256": file_hash(destination / "plan.json"), "root": str(root),
             "report_name": report_name,
             "retained_reports": plan["reports"], "failure_summaries": [], "acceptance_summaries": [],
             "deleted_work": [], "deleted_acceptance": [], "protected": plan["protected"], "errors": [],
             "reclaimed_bytes": 0, "reclaimed_allocated_bytes": 0, "original_recoverable": False}
    def save():
        write_json(destination / "retention-index.json", index)
    save()
    for item in plan["work"]:
        path = inside(root, item["path"])
        try:
            if inventory(path) != item["inventory"] or (path / "ACTIVE").exists():
                raise RuntimeError("work directory changed or became active")
            if item["kind"] != "verified_report":
                raise RuntimeError("only verified closed runs may be retired")
            identity = (item["experiment"], item["run_id"])
            archived = verified_report(root, root / "results" / identity[0] / identity[1], identity)
            if archived not in plan["reports"] or checked_work(path, identity, archived) != item["run_json"]:
                raise RuntimeError("verified result or source binding changed")
            index["delete_intent"] = {"kind": "work", "path": item["path"]}
            save()
            shutil.rmtree(path)
            index.pop("delete_intent", None)
            index["deleted_work"].append({"path": item["path"], "bytes": item["inventory"]["bytes"],
                                          "allocated_bytes": item["inventory"]["allocated_bytes"]})
            index["reclaimed_bytes"] += item["inventory"]["bytes"]
            index["reclaimed_allocated_bytes"] += item["inventory"]["allocated_bytes"]
            save()
        except Exception as error:
            index.pop("delete_intent", None)
            index["errors"].append({"path": item["path"], "error": type(error).__name__ + ": " + str(error),
                                    "partial_directory_possible": True})
            save()
    for item in plan["acceptance"]:
        path = inside(root, item["path"])
        try:
            if any(error["path"] in {candidate["path"] for candidate in plan["work"]
                                     if candidate["run_id"] == item["run_id"]} for error in index["errors"]):
                raise RuntimeError("corresponding work retirement failed; acceptance evidence retained")
            if inventory(path) != item["inventory"] or (path / "ACTIVE").exists():
                raise RuntimeError("acceptance directory changed or became active")
            reason = protected_formal_source(path)
            if reason:
                raise RuntimeError(reason)
            document = acceptance_document(path, item)
            summary, digest = save_summary(destination, "acceptance-summaries", item["run_id"], document)
            index["acceptance_summaries"].append({"path": relative(root, summary), "sha256": digest, "run_id": item["run_id"]})
            index["delete_intent"] = {"kind": "acceptance", "path": item["path"]}
            save()
            if inventory(path) != item["inventory"]:
                raise RuntimeError("acceptance directory changed while summarizing")
            reason = protected_formal_source(path)
            if reason:
                raise RuntimeError(reason)
            shutil.rmtree(path)
            index.pop("delete_intent", None)
            index["deleted_acceptance"].append({"path": item["path"], "bytes": item["inventory"]["bytes"],
                                                "allocated_bytes": item["inventory"]["allocated_bytes"]})
            index["reclaimed_bytes"] += item["inventory"]["bytes"]
            index["reclaimed_allocated_bytes"] += item["inventory"]["allocated_bytes"]
            save()
        except Exception as error:
            index.pop("delete_intent", None)
            index["errors"].append({"path": item["path"], "error": type(error).__name__ + ": " + str(error),
                                    "partial_directory_possible": True})
            save()
    index["status"] = "completed" if not index["errors"] else "completed_with_errors"
    index["completed_at"] = timestamp()
    lines = ["# DITSERVE output 留存报告", "",
             "已删除目录仅保留已核验报告、关键配置、指标和错误摘要，不能从这些原工作目录续跑或重新评测；未删除项及错误见下列回执。", "",
             "- 已核验并保留的最终报告：%d" % len(index["retained_reports"]),
             "- 新增失败运行摘要：%d" % len(index["failure_summaries"]),
             "- 新增验收日志/指标摘要：%d" % len(index["acceptance_summaries"]),
             "- 已删除工作目录：%d；验收目录：%d" % (len(index["deleted_work"]), len(index["deleted_acceptance"])),
             "- 删除前逻辑字节：%d；分配字节：%d" % (index["reclaimed_bytes"], index["reclaimed_allocated_bytes"]),
             "- 保护未删记录：%d；删除/摘要错误：%d" % (len(index["protected"]), len(index["errors"])), "",
             "逐项路径、SHA256、删除事实和保留原因见 `retention-index.json`；计划与源文件绑定见 `plan.json`。", "",
             "## 保护未删记录", ""]
    lines += ["- `%s`：%s" % (item["path"], item["reason"]) for item in index["protected"]]
    if index["errors"]:
        lines += ["", "## 错误", ""] + ["- `%s`：%s" % (item["path"], item["error"]) for item in index["errors"]]
    with (destination / report_name).open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    index["report_sha256"] = file_hash(destination / report_name)
    save()
    print(json.dumps(verify_retention(root, destination), ensure_ascii=False, indent=2))


def verify_retention(root, destination):
    index, unused = read_json(destination / "retention-index.json")
    if index.get("schema") != INDEX_SCHEMA or index.get("status") not in ("completed", "completed_with_errors"):
        raise ValueError("retention did not complete")
    if index.get("root") != str(root):
        raise ValueError("retention root mismatch")
    if index.get("plan_sha256") != file_hash(destination / "plan.json"):
        raise ValueError("retention plan hash mismatch")
    report_name = index.get("report_name", "REPORT.md")
    if report_name != "REPORT.md" and not DATED_REPORT.fullmatch(report_name):
        raise ValueError("unsafe retention report name")
    report_path = real(destination / report_name)
    if "report_name" in index and (file_hash(report_path) != index.get("report_sha256")
                                   or list(destination.glob("*.md")) != [report_path]):
        raise ValueError("retention report changed or multiple reports exist")
    for item in index["retained_reports"]:
        for field in report_files(item):
            if file_hash(inside(root, item[field])) != item[field + "_sha256"]:
                raise ValueError("retained report hash mismatch: " + item[field])
    for category in ("failure_summaries", "acceptance_summaries"):
        for item in index[category]:
            if file_hash(inside(root, item["path"])) != item["sha256"]:
                raise ValueError("summary hash mismatch: " + item["path"])
    for category in ("deleted_work", "deleted_acceptance"):
        for item in index[category]:
            if (root / item["path"]).exists():
                raise ValueError("deleted path still exists: " + item["path"])
    for item in index["protected"]:
        if not (root / item["path"]).exists():
            raise ValueError("protected path disappeared: " + item["path"])
    return {"status": index["status"], "verified_reports": len(index["retained_reports"]),
            "failure_summaries": len(index["failure_summaries"]), "acceptance_summaries": len(index["acceptance_summaries"]),
            "deleted_work": len(index["deleted_work"]), "deleted_acceptance": len(index["deleted_acceptance"]),
            "reclaimed_bytes": index["reclaimed_bytes"], "reclaimed_allocated_bytes": index["reclaimed_allocated_bytes"],
            "protected": len(index["protected"]), "errors": index["errors"],
            "report": str(report_path)}


def command_verify(args):
    root = output_root(args.root)
    destination = report_directory(root, args.report_dir)
    print(json.dumps(verify_retention(root, destination), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--source-root")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--run-id")
    manifest.add_argument("--experiment")
    manifest.set_defaults(function=command_manifest)
    plan = commands.add_parser("plan")
    plan.add_argument("--root", required=True)
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--report-dir", required=True)
    plan.add_argument("--run-id")
    plan.add_argument("--discard-pending", action="store_true", help="unsupported: open consumers must be closed and re-finalized")
    plan.set_defaults(function=command_plan)
    apply = commands.add_parser("apply")
    apply.add_argument("--root", required=True)
    apply.add_argument("--report-dir", required=True)
    apply.set_defaults(function=command_apply)
    verify = commands.add_parser("verify")
    verify.add_argument("--root", required=True)
    verify.add_argument("--report-dir", required=True)
    verify.set_defaults(function=command_verify)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
