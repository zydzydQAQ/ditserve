"""Read a finalized cache report after explicitly receipted log aggregation.

This is a verified historical view, never a fresh evaluation or GPU admission.
No export, deletion, reconstruction of raw events, or mutation occurs here.
"""

import copy
import gzip
import hashlib
import json
from pathlib import Path

from .. import compact_results
from ..research.common import digest, file_digest
from .reporting import _read


SCHEMA = "ditserve.optimizations.cache_log_finalization.v1"
LOG_NAMES = frozenset(("cache-online.json", "partial.json", "isolation.jsonl"))
MAX_PAYLOAD_BYTES = 32 * 1024 * 1024


class ArchiveIntegrityError(ValueError):
    """A missing or changed source cannot become a valid archived conclusion."""


def _require(condition, message):
    if not condition:
        raise ArchiveIntegrityError("cache archive view refused: " + message)


def _inside(path, root):
    return path == root or root in path.parents


def _path(value):
    _require(isinstance(value, str) and value, "explicit absolute path missing")
    path = Path(value)
    _require(path.is_absolute() and path.resolve() == path, "relative/symlink path refused")
    return path


def _binding(value, label, expected=None):
    _require(isinstance(value, dict), label + " binding missing")
    path = _path(value.get("path"))
    _require(expected is None or path == expected, label + " path differs")
    try:
        data, observed = _read(path)
    except (OSError, ValueError, TypeError) as error:
        raise ArchiveIntegrityError(label + ": " + str(error)) from error
    _require(observed["sha256"] == value.get("sha256"), label + " SHA256 differs")
    _require("bytes" not in value or observed["bytes"] == value["bytes"], label + " size differs")
    return data, observed


def _payload(directory):
    with gzip.open(directory / "data.json.gz", "rb") as stream:
        raw = stream.read(MAX_PAYLOAD_BYTES + 1)
    _require(len(raw) <= MAX_PAYLOAD_BYTES, "archive payload exceeds view bound")
    return json.loads(raw)


def _json_entry(payload, name):
    entries = [entry for entry in payload.get("files", []) if entry.get("path") == name]
    _require(len(entries) == 1 and isinstance(entries[0].get("text"), str),
             "frozen metadata missing: " + name)
    entry = entries[0]
    _require(hashlib.sha256(entry["text"].encode()).hexdigest() == entry.get("sha256"),
             "frozen metadata hash differs: " + name)
    return json.loads(entry["text"]), entry


def _source_snapshot(output, attempt, terminal, study):
    manifest_path = output / "sources" / ("cache_k4-" + attempt.name) / "manifest.json"
    manifest, unused = _read(manifest_path)
    hashes = manifest.get("source_files", {})
    _require(isinstance(hashes, dict) and hashes and digest(hashes) == manifest.get("source_sha256")
             and manifest.get("source_sha256") == terminal.get("source_sha256")
             and manifest.get("config_sha256") == study.get("inherited_config_sha256")
             and manifest.get("execution_root") == str(manifest_path.parent), "snapshot identity differs")
    for name, expected in hashes.items():
        relative = Path(name)
        path = manifest_path.parent / relative
        _require(not relative.is_absolute() and ".." not in relative.parts and path.resolve() == path
                 and path.suffix in (".py", ".json") and path.is_file()
                 and path.stat().st_size <= MAX_PAYLOAD_BYTES and file_digest(path) == expected,
                 "snapshot file differs: " + name)


def _load_archive_report(output, ledger):
    output, ledger = _path(str(Path(output).absolute())), _path(str(Path(ledger).absolute()))
    _require(ledger.name == "gpu-budget.json" and output != ledger.parent and _inside(output, ledger.parent),
             "independent ledger must contain this run")
    finalization_path = output / "log-finalization.json"
    finalization, finalization_source = _read(finalization_path)
    _require(finalization.get("schema_version") == SCHEMA and finalization.get("status") == "completed"
             and finalization.get("dependencies_closed") is True and finalization.get("output") == str(output),
             "completed explicit finalization missing")
    budget, unused = _read(ledger)
    _require(budget.get("active") is None, "GPU ledger is active")

    main_record, main_record_source = _binding(finalization.get("main_archive"), "main archive", output / "archive.json")
    main_manifest, main_manifest_source = _binding(finalization.get("main_manifest"), "main manifest",
                                                 output / "excel/manifest.json")
    main = main_manifest_source["path"]
    main_dir = Path(main).parent
    stage = output / "archive-metadata"
    _require(main_record.get("status") == "verified" and main_record.get("manifest") == main
             and main_record.get("staging_root") == str(stage) and main_manifest.get("root") == str(stage),
             "main archive is not this run's verified metadata archive")
    compact_results.verify(main_dir, check_sources=True)
    payload = _payload(main_dir)
    frozen, entry = _json_entry(payload, "report-summary.json")
    _require(frozen.get("output") == str(output) and frozen.get("budget_path") == str(ledger)
             and frozen.get("matrix_complete") is True and frozen.get("online_evaluation_complete") is True
             and not frozen.get("errors") and frozen.get("archive_ready") is True,
             "frozen report was not a completed verified final evaluation")
    current = frozen.get("current", {})
    _require(len(current.get("rows", [])) == 16 and current.get("raw_request_count") == 384,
             "frozen full factorial counts differ")
    attempt = _path(frozen.get("selected_attempt"))
    _require(attempt.parent == output / "cache_k4", "selected attempt is outside new run")
    _require(main_record.get("source_bindings") == frozen.get("source_bindings"),
             "main archive and frozen report source bindings differ")

    deleted = finalization.get("deleted_sources")
    _require(isinstance(deleted, list) and deleted, "explicit deleted-source list missing")
    selected = {}
    for source in deleted:
        path = _path(source.get("path"))
        _require(path.parent == attempt and path.name in LOG_NAMES and str(path) not in selected
                 and source.get("kind") == "summarized_text" and source.get("dependencies_closed") is True
                 and isinstance(source.get("closure_reason"), str) and source["closure_reason"].strip()
                 and type(source.get("bytes")) is int and source["bytes"] >= 0
                 and not path.exists(), "deleted source is present, duplicated, out of scope or unclosed")
        selected[str(path)] = source
    _require(str(attempt / "cache-online.json") in selected, "online source was not explicitly finalized")

    audit, audit_source = _binding(finalization.get("final_audit"), "final independent audit")
    _require(Path(audit_source["path"]) == output / "audits/final-integrity.json"
             and audit.get("status") == "passed" and audit.get("output") == str(output)
             and audit.get("matrix_complete") is True
             and audit.get("cache_online_sha256") == selected[str(attempt / "cache-online.json")]["sha256"],
             "audit is not bound to this completed online matrix")
    log_manifest, log_manifest_source = _binding(finalization.get("log_manifest"), "log manifest")
    log_dir = Path(log_manifest_source["path"]).parent
    _require(_inside(log_dir, output) and log_dir not in (output, main_dir, stage)
             and Path(log_manifest_source["path"]).name == "manifest.json"
             and log_manifest.get("root") == str(attempt), "log archive is not bound to the original attempt")
    receipt, receipt_source = _binding(finalization.get("prune_receipt"), "prune receipt", log_dir / "pruned.json")
    _require(receipt.get("status") == "completed" and receipt.get("root") == str(attempt)
             and set(receipt.get("deleted", {})) == {Path(path).name for path in selected},
             "receipt does not contain exactly the reviewed cleanup list")
    compact_results.verify(log_dir)
    log_payload = _payload(log_dir)
    log_entries = {item["path"]: item for item in log_payload.get("files", [])}
    for path, source in selected.items():
        reference = compact_results.verify_compacted_reference(attempt, path, source["sha256"])
        _require(reference.get("archive") == str(log_dir) and reference.get("evidence_status") == "aggregated_only",
                 "missing source points to an unbound log archive")
        item = log_entries.get(Path(path).name, {})
        summary = item.get("summary", {})
        _require(item.get("kind") == "summarized_text" and item.get("bytes") == source["bytes"]
                 and item.get("sha256") == source["sha256"]
                 and summary.get("frozen_report_sha256") == entry["sha256"]
                 and summary.get("final_audit_sha256") == audit_source["sha256"],
                 "log summary is not bound to the frozen report and audit")

    retained = finalization.get("retained_sources", [])
    _require(isinstance(retained, list), "retained-source list must be explicit")
    retained_by_path = {item.get("path"): item for item in retained}
    _require(len(retained_by_path) == len(retained), "duplicate retained source")
    for path in (attempt / "config.json", attempt / "study.json"):
        _require(str(path) in retained_by_path, "current config/study raw binding missing")
    verified_sources = {}
    for source in list(frozen.get("source_bindings", [])) + retained:
        path = source.get("path")
        if path in selected:
            _require(source.get("sha256") == selected[path]["sha256"]
                     and source.get("bytes") == selected[path]["bytes"], "deleted binding differs from final report")
            continue
        value, observed = _binding(source, "retained source")
        if path in verified_sources:
            _require(verified_sources[path][1] == observed, "duplicate retained hashes disagree")
        verified_sources[path] = (value, observed)
    for path in (ledger, output / "study.json", output / "cache_k4/terminal.json", attempt / "supervisor.json"):
        _require(str(path) in verified_sources, "essential final source is not bound")
    terminal = verified_sources[str(output / "cache_k4/terminal.json")][0]
    supervisor = verified_sources[str(attempt / "supervisor.json")][0]
    study = verified_sources[str(output / "study.json")][0]
    _require(terminal == supervisor and terminal.get("output") == str(attempt)
             and terminal.get("status") in ("passed", "completed", "stopped") and terminal.get("returncode") == 0
             and not terminal.get("supervisor_failure") and terminal.get("online_evaluation_complete") is True
             and terminal.get("online_qualified") == frozen.get("online_qualified"), "final supervisor differs")
    final = terminal.get("final_occupancy", {})
    _require(final.get("status") == "exclusive_observed" and final.get("owned") == [] and final.get("foreign") == [],
             "exclusive final cleanup is not proven")
    charge = terminal.get("charge", {})
    settled = [item for item in budget.get("sessions", []) if item.get("session_id") == charge.get("session_id")]
    _require(len(settled) == 1 and settled[0] == charge and charge.get("status") == "completed"
             and charge.get("gpu_count") == 1 and charge.get("stage") == "cache_k4"
             and charge.get("identity", {}).get("output") == str(attempt)
             and charge.get("identity", {}).get("run_root") == str(output), "charge is not identically settled")
    _require(verified_sources[str(attempt / "study.json")][0] == study
             and digest(verified_sources[str(attempt / "config.json")][0]) == study.get("inherited_config_sha256"),
             "current config/study identity differs")
    auth_binding = study.get("authorization", {})
    _require(auth_binding.get("path") in verified_sources
             and verified_sources[auth_binding["path"]][1]["sha256"] == auth_binding.get("sha256"),
             "authorization is not retained with its frozen hash")
    _source_snapshot(output, attempt, terminal, study)

    view = copy.deepcopy(frozen)
    # These are the already verified bounded metric tables, not raw events.
    # Restore the public report structure without re-running any metric/gate.
    for index, item in enumerate(view.get("previous_attempts", []) + [view["current"]]):
        for suffix, key in (("requests", "request_samples"), ("event-summary", "event_summaries")):
            value, unused = _json_entry(payload, "attempt%02d-%s.json" % (index, suffix))
            _require(isinstance(value, list), "archived metric table must be a list")
            item[key] = value
    view.update(evidence_mode="archive_only", archive_only=True, archive_verified=True, archive_ready=False,
        qualification_recomputed=False, new_qualification_granted=False, raw_event_re_evaluable=False,
        archival_scope="frozen independently audited conclusion; no raw events reconstructed, no new evaluation or qualification",
        archive_view=dict(finalization_source=finalization_source, main_manifest=main_manifest_source,
            log_manifest=log_manifest_source, prune_receipt=receipt_source, final_audit=audit_source,
            frozen_report_sha256=entry["sha256"], deleted_sources=deleted, retained_sources_verified=len(verified_sources)),
        retention=dict(frozen_before_cleanup=frozen.get("retention", {}), original_log_files_deleted=len(deleted),
            original_events_recoverable=False, original_arrays_deleted=0, prune_authorized=False))
    return view


def load_archive_report(output, ledger):
    """Fail closed unless archives, receipt and all remaining evidence agree."""
    try:
        return _load_archive_report(output, ledger)
    except ArchiveIntegrityError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ArchiveIntegrityError("cache archive view refused: " + str(error)) from error
