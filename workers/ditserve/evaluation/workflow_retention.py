"""Keep workflow evidence and prune only verified, redundant experiment payloads."""

import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path
import re
import stat

from ditserve import results


POLICY = "workflow-key-evidence-v1: retain reports, metadata, sources, logs, captures and quality arrays; prune verified redundant payloads"
_LOCKS = ("driver.lock", ".postprocess.lock", ".watch-postprocess.lock", "budget.lock")
_E4 = re.compile(r"E4-r([0-9]+)-(sync|cpu_async|d2h_async)-request([0-9]+)\.npy\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _run(root):
    value, binding = results._document(root / "run.json")
    if value.get("schema") != "ditserve.run.v1" or value.get("experiment") != "workflow":
        raise ValueError("expected canonical workflow run metadata")
    return value, binding


def _journal(root):
    path = results._path(root / results.JOURNAL)
    if not path.exists():
        return None
    value, unused = results._document(path)
    if (value.get("schema") != results.SCHEMA or value.get("work_dir") != str(root)
            or value.get("sha256") != results._digest({k: v for k, v in value.items() if k != "sha256"})):
        raise ValueError("invalid workflow finalization journal")
    return value


def archived_run(root):
    """Reuse published metadata; never rebuild a finalized experiment in place."""
    root = results._path(root)
    journal = _journal(root)
    if journal and (journal.get("status") == "completed" or journal.get("deleted") or journal.get("pending_delete")):
        return _run(root)[0]
    return None


def _inactive(root, run):
    results._active(run)
    documents = [("run.json", run)]
    ledger = run.get("budget", {}).get("ledger") or run.get("budget_ledger")
    if ledger:
        data, unused = results._document(root / ledger)
        results._active({"budget": data})
    for name in ("study.json", "budget.json", "operator-notes.json"):
        path = results._path(root / name)
        if path.exists():
            value, unused = results._document(path)
            documents.append((name, value))
    for name, value in documents:
        results._active(value)
        if name == "budget.json":
            results._active({"budget": value})
        for key, item in value.items():
            if (key == "active" or key.startswith("active_") or key in {"current_job", "current_batch"}) and item:
                raise RuntimeError("active workflow field: " + name + ":" + key)
        for key in ("postprocess", "parallel_postprocess"):
            if isinstance(value.get(key), dict):
                results._active(value[key])
        dependencies = value.get("retention", {}).get("pending_dependencies", value.get("pending_dependencies", []))
        if not results._closed(dependencies):
            raise RuntimeError("active workflow dependencies remain open: " + name)


@contextmanager
def _guard(root, run):
    handles = []
    ledger = run.get("budget", {}).get("ledger") or run.get("budget_ledger")
    delegated = set()
    if ledger:
        ledger = results._path(root / ledger)
        delegated = {ledger.with_suffix(".lock"), Path(str(ledger) + ".lock")}
    try:
        # Acquire the retention lock first; existing producer locks stay held
        # through inventory hashing, archive publication and finalizer cleanup.
        for name in (".retention.lock", *_LOCKS):
            path = results._path(root / name)
            if path in delegated or name != ".retention.lock" and not path.exists():
                continue
            handle = path.open("a+b" if name == ".retention.lock" else "rb")
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("active workflow lock: " + name) from None
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def _extension(previous, current):
    """Allow retry after inventory was saved but its archive was not published."""
    old, new = copy.deepcopy(previous), copy.deepcopy(current)
    old_artifacts, new_artifacts = old.pop("artifacts", []), new.pop("artifacts", [])
    if new_artifacts[:len(old_artifacts)] != old_artifacts:
        return False
    extra = new_artifacts[len(old_artifacts):]
    if any(item.get("cleanup_evidence", {}).get("policy") != POLICY for item in extra):
        return False
    for value in (old, new):
        value.setdefault("retention", {}).pop("policy", None)
    return old == new


def _archive(root, run, destination, journal):
    if journal and journal.get("destination") != str(destination):
        raise ValueError("finalization destination changed")
    if not destination.exists() or not any(destination.iterdir()):
        if journal and journal.get("status") == "completed":
            raise ValueError("completed workflow archive is missing")
        return
    if not journal:
        raise ValueError("existing results require this run's finalization journal")
    previous = results.load(destination)  # Verifies workbook digest and report consistency.
    if previous.get("schema") != results.SCHEMA:
        raise ValueError("expected canonical workflow archive")
    if previous["run"] != run and (journal.get("deleted") or journal.get("pending_delete")
                                  or not _extension(previous["run"], run)):
        raise ValueError("run differs from verified archive beyond retention inventory")


def _ordinary(path):
    path = results._path(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("expected ordinary workflow artifact: " + str(path))
    return path


def _e4_evidence(path):
    match = _E4.fullmatch(path.name)
    if not match:
        return None
    mode_path = results._path(path.with_name("E4-r%s-%s.json" % (match[1], match[2])))
    parent_path = results._path(path.parent / "run.json")
    if not mode_path.is_file() or not parent_path.is_file():
        return None
    parent, parent_binding = results._document(parent_path)
    mode, mode_binding = results._document(mode_path)
    # Historical mode JSON has no status: its enclosing completed run and
    # bounded per-request completion timestamps establish completion.
    if parent.get("status") not in {"complete", "completed"} or mode.get("status", "completed") not in {"complete", "completed"}:
        return None
    start, end = mode.get("start_ns"), mode.get("end_ns")
    if type(start) is not int or type(end) is not int or end <= start:
        return None
    records = [row for row in mode.get("requests", []) if row.get("index") == int(match[3])]
    if len(records) != 1:
        return None
    record = records[0]
    completed = record.get("complete_ns")
    if (record.get("finite") is not True or not _SHA.fullmatch(str(record.get("sha256", "")))
            or not isinstance(record.get("shape"), list) or not record["shape"] or not record.get("dtype")
            or type(completed) is not int or not start <= completed <= end):
        return None
    return dict(retained=[parent_binding, mode_binding], request=record, mode_json=mode)


def _profile_evidence(root, path, bindings):
    def binding(file):
        if file not in bindings:
            bindings[file] = results._read(file)[0]
        return bindings[file]
    parts = path.relative_to(root / "profiling").parts
    if len(parts) == 3 and parts[0] == "overlap" and re.fullmatch(r"E4-f[0-9]+", parts[1]):
        profile = _e4_evidence(path)
        normal = _e4_evidence(root.joinpath(*parts))
        if profile and normal and all(profile["request"].get(key) == normal["request"].get(key)
                                       for key in ("sha256", "shape", "dtype", "finite", "latent")):
            return dict(retained=profile["retained"] + normal["retained"],
                        numeric_match=normal["request"], mode_json=profile["mode_json"])
        return None
    # Encoder outputs have a stable layout in both Nsight Systems and NCU
    # captures. Require a completed owner and an exact ordinary-run duplicate.
    owner = next((parent for parent in path.parents if parent != root / "profiling"
                  and root / "profiling" in parent.parents and (parent / "result.json").exists()), None)
    if owner is None or owner.parent.name != "encoder" or not re.fullmatch(r"T[0-9]+", owner.name):
        return None
    relative = path.relative_to(owner)
    if not (len(relative.parts) == 2 and relative.parts[0] == "numeric" and path.suffix == ".npz"
            or len(relative.parts) == 3 and relative.parts[0] == "outputs" and relative.parts[1].isdigit()
            and path.name in {"video.npy", "latent.npy"}):
        return None
    normal = results._path(root / "encoder-v2" / owner.name)
    normal_result = normal / "result.json"
    duplicate = results._path(normal / relative)
    results._path(normal_result)
    if not normal_result.is_file() or not duplicate.is_file():
        return None
    profile, profile_binding = results._document(owner / "result.json")
    baseline, baseline_binding = results._document(normal_result)
    if any(value.get("schema") != "ditserve.workflow.encoder.v1" or value.get("status") != "completed"
           or value.get("variant") != owner.name for value in (profile, baseline)):
        return None
    source, target = binding(path), binding(duplicate)
    if any(source[key] != target[key] for key in ("bytes", "sha256")):
        return None
    return dict(retained=[profile_binding, baseline_binding, target], candidate_binding=source,
                match="exact file bytes and SHA256")


def _candidates(root, run):
    """Yield narrowly scoped candidates with the evidence that permits pruning."""
    for path in sorted(root.glob("traces/*.sqlite")):
        path = _ordinary(path)
        capture, parsed = path.with_suffix(".nsys-rep"), path.with_suffix(".parsed.json")
        results._path(capture)
        results._path(parsed)
        if not capture.is_file() or not parsed.is_file():
            continue
        value, parsed_binding = results._document(parsed)
        if value.get("schema_version", value.get("schema")) != "ditserve.workflow_trace.v1" or value.get("status") != "completed":
            continue
        capture_binding, unused = results._read(capture)
        recorded = value.get("postprocess", {}).get("capture", {})
        if any(recorded.get(key) != capture_binding[key] for key in ("bytes", "sha256")):
            continue
        yield path, "trace_sqlite", dict(retained=[capture_binding, parsed_binding],
            parser_schema="ditserve.workflow_trace.v1", parser_status="completed")
    analysis = results._path(root / "ANALYSIS.html")
    if analysis.is_file():
        bindings = {}
        for path in sorted(root.glob("profiling/**/*")):
            if path.suffix in {".npy", ".npz"}:
                path = _ordinary(path)
                evidence = _profile_evidence(root, path, bindings)
                if evidence:
                    yield path, "profiling_array", evidence
    for path in sorted(root.glob("overlap/E4-f*/E4-r*-request*.npy")):
        path = _ordinary(path)
        evidence = _e4_evidence(path)
        if evidence:
            yield path, "e4_output", evidence
    for path in sorted(root.glob("encoder/T0/numeric/**/*.npz")):
        path = _ordinary(path)
        duplicate = results._path(root / "encoder-v2/T0/numeric" / path.relative_to(root / "encoder/T0/numeric"))
        if not duplicate.is_file():
            continue
        source, unused = results._read(path)
        target, unused = results._read(duplicate)
        if all(source[key] == target[key] for key in ("bytes", "sha256")):
            yield path, "duplicate_encoder_array", dict(retained=[target], candidate_binding=source)


def _inventory(root, run, *, dry_run):
    updated = copy.deepcopy(run)
    artifacts = updated.setdefault("artifacts", [])
    indexed = {}
    for artifact in artifacts:
        path = results._path(root / artifact["path"])
        if root not in path.parents or path in indexed:
            raise ValueError("invalid existing artifact inventory")
        if artifact.get("kind") in results.RAW_KINDS | results.LOG_KINDS and artifact.get("cleanup_evidence", {}).get("policy") != POLICY:
            raise ValueError("existing prunable artifact is outside workflow retention policy: " + str(path))
        indexed[path] = artifact
    counts, candidates = {}, []
    implementation = results._read(Path(__file__))[0]
    for path, category, evidence in _candidates(root, run):
        existing = indexed.get(path)
        if existing and existing.get("cleanup_evidence", {}).get("policy") != POLICY:
            # Existing control/source/quality bindings always win. Never turn an
            # existing artifact into a deletion candidate by changing its kind.
            continue
        binding = {"path": str(path), "bytes": path.stat().st_size} if dry_run else results._read(path)[0]
        compared = evidence.get("candidate_binding")
        if compared and not dry_run and compared != binding:
            raise ValueError("duplicate changed after comparison: " + str(path))
        evidence.update(policy=POLICY, category=category, implementation=implementation)
        if not dry_run:
            for retained in evidence.get("retained", []):
                evidence_path = results._path(retained["path"])
                if root not in evidence_path.parents:
                    raise ValueError("cleanup evidence is outside workflow directory")
                if evidence_path not in indexed:
                    artifact = dict(retained, path=str(evidence_path.relative_to(root)), kind="control", dependencies=[],
                                    cleanup_evidence=dict(policy=POLICY, role="retained evidence"))
                    artifacts.append(artifact)
                    indexed[evidence_path] = artifact
        if existing:
            if not dry_run and any(existing.get(key) != binding[key] for key in ("bytes", "sha256")):
                raise ValueError("cleanup candidate changed since inventory was saved")
        elif not dry_run:
            artifacts.append(dict(binding, path=str(path.relative_to(root)), kind="raw", dependencies=[], cleanup_evidence=evidence))
        candidates.append(dict(binding, category=category))
        item = counts.setdefault(category, dict(files=0, bytes=0))
        item["files"] += 1
        item["bytes"] += binding["bytes"]
    updated.setdefault("retention", {})["policy"] = POLICY
    return updated, dict(candidates=candidates, files=len(candidates), bytes=sum(row["bytes"] for row in candidates), categories=counts)


def finalize_workflow(root, destination=None, *, keep_raw=False, dry_run=False):
    """Validate evidence, append an exact inventory, and use the shared finalizer."""
    root = results._path(root)
    try:
        initial, unused = _run(root)
        with _guard(root, initial):
            run, source = _run(root)
            if (run.get("budget", {}).get("ledger"), run.get("budget_ledger")) != (initial.get("budget", {}).get("ledger"), initial.get("budget_ledger")):
                raise ValueError("budget ledger changed while acquiring workflow locks")
            journal = _journal(root)
            destination = results._path(destination or (journal or {}).get("destination")
                                        or root.parent.parent / "results" / "workflow" / root.name)
            if destination == root or root in destination.parents or destination in root.parents:
                raise ValueError("results and work directories must be separate")
            _inactive(root, run)
            _archive(root, run, destination, journal)
            if journal and (journal.get("deleted") or journal.get("pending_delete")):
                if dry_run:
                    return dict(status="dry_run", policy=POLICY, files=0, bytes=0, categories={}, candidates=[],
                                already_deleted=len(journal.get("deleted", [])))
                return results.finalize(root, destination, prune=not keep_raw)
            if keep_raw:
                if dry_run:
                    return dict(status="dry_run", policy="keep_raw", files=0, bytes=0, categories={}, candidates=[])
                return results.finalize(root, destination, prune=False)
            updated, plan = _inventory(root, run, dry_run=dry_run)
            if dry_run:
                return dict(status="dry_run", policy=POLICY, **plan)
            # Recheck controls and immutable source after potentially long hashes.
            _inactive(root, run)
            if results._read(root / "run.json")[0] != source:
                raise ValueError("run changed while preparing retention inventory")
            if updated != run:
                results._atomic_json(root / "run.json", updated)
            value = results.finalize(root, destination, prune=True)
            return dict(value, policy=POLICY, inventory=plan)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        return dict(status="protected" if isinstance(error, RuntimeError) else "deferred",
                    reason=type(error).__name__ + ": " + str(error), work_dir=str(root))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = finalize_workflow(args.root, args.destination, keep_raw=args.keep_raw, dry_run=args.dry_run)
    print(json.dumps(value, ensure_ascii=False))
    if value["status"] not in {"completed", "dry_run"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
