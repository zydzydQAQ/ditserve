"""Auditable, resumable raw-evidence retention; historical records stay immutable.

The planner fails closed on active ledgers, unsealed evidence, shared storage and
unknown output formats. Plans are file-level manifests, not glob deletion rules.
Only a verified plan may remove bytes. Key samples remain full-precision files in
an external content-addressed store; a sidecar records every declared deletion.
"""

import argparse
from contextlib import contextmanager
from decimal import Decimal
import fcntl
import json
import math
import os
import re
from pathlib import Path
import shutil
import stat
import statistics
import tempfile
import time

from .common import atomic_json, digest, file_digest, read_json

DEFAULT_ARTIFACT_ROOT = "/mnt/data/zyd-preempt-phase16-01a1296d/dit-research-artifacts"
SCHEMA = "ditserve.research.retention.v1"
RAW_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".safetensors", ".mp4", ".webm", ".gif", ".pkl"}
TERMINAL = {"completed", "failed", "aborted", "blocked_contention", "budget_stopped", "passed", "unqualified"}


def _seal(value):
    value = dict(value)
    value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    return value


def _check_seal(value):
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("retention or evidence manifest seal mismatch")
    return value


def _inside(path, root):
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def _logical(root, path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("parent traversal in artifact reference")
    return Path(os.path.abspath(path if path.is_absolute() else Path(root) / path))


def _ledger_paths(root, supplied):
    found = {str(Path(p).absolute()) for p in supplied}
    # Read metadata only. No raw-array scan or hashing occurs before this guard.
    for name in ("gpu-budget.json", "budget.json"):
        found.update(str(p.absolute()) for p in Path(root).rglob(name))
    return sorted(found)


@contextmanager
def idle_guard(root, ledger_paths=()):
    """Hold the same nonblocking locks as GPU accounting for the whole operation."""
    root = Path(root).absolute()
    handles = []
    try:
        for ledger in _ledger_paths(root, ledger_paths):
            lock = Path(ledger + ".lock")
            handle = lock.open("a+")
            handles.append(handle)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("active or locked GPU ledger prohibits retention: " + ledger) from error
            value = read_json(ledger)
            if value.get("active") or value.get("active_session"):
                raise RuntimeError("unsettled GPU session prohibits retention: " + ledger)
        root.mkdir(parents=True, exist_ok=True)
        handle = (root / ".retention.lock").open("a+")
        handles.append(handle)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another retention operation is active") from error
        # An explicit continuation marker is deliberately independent of process
        # visibility: a hidden PID is never interpreted as proof of completion.
        for name in ("ACTIVE", "RESUME_PENDING", "retention-blocked.json"):
            if next(root.rglob(name), None) is not None:
                raise RuntimeError("active or pending continuation marker prohibits retention")
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def _artifacts(value, run_root):
    """Extract known file references; do not confuse tensor hashes with file hashes."""
    found = []
    if isinstance(value, dict):
        for prefix in ("output", "latent", "state", "checkpoint"):
            path = value.get(prefix + "_path")
            if isinstance(path, str):
                found.append(dict(path=str(_logical(run_root, path)), sha256=value.get(prefix + "_file_sha256", value.get(prefix + "_sha256")), kind=prefix))
        if value.get("kind") in ("latent", "output", "state") and isinstance(value.get("path"), str):
            found.append(dict(path=str(_logical(run_root, value["path"])), sha256=value.get("sha256"), kind=value["kind"]))
        for child in value.values():
            found.extend(_artifacts(child, run_root))
    elif isinstance(value, list):
        for child in value:
            found.extend(_artifacts(child, run_root))
    unique = {}
    for item in found:
        previous = unique.get(item["path"])
        if previous and previous.get("sha256") and item.get("sha256") and previous["sha256"] != item["sha256"]:
            raise ValueError("conflicting sealed artifact hashes: " + item["path"])
        if previous is None or item.get("sha256"):
            unique[item["path"]] = item
    return list(unique.values())



def _evidence_output_hashes(value):
    """Legacy mechanism records sometimes preserve hashes but omit file paths."""
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("candidate_output_sha256", "reference_output_sha256") and isinstance(child, str):
                found.add(child)
            elif isinstance(child, (dict, list)):
                found.update(_evidence_output_hashes(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_evidence_output_hashes(child))
    return found


def _metrics(row):
    evidence = row.get("evidence", {})
    metrics = evidence.get("metrics", row.get("metrics", row.get("quality", {})))
    loss = metrics.get("lpips_mean")
    return loss if isinstance(loss, (int, float)) and not isinstance(loss, bool) and math.isfinite(loss) else None


def discover_catalog(root):
    """Normalize supported ended research/benchmark runs using small metadata.

    Unsupported legacy profile outputs are listed as exceptions and remain in
    place. No quality values or selected policies are invented for those runs.
    Call under ``idle_guard`` before using this catalog for cleanup.
    """
    root = Path(root).absolute()
    cases, sources, exceptions, protected = [], set(), [], set()
    for records in sorted(root.rglob("records.jsonl")):
        run = records.parent
        report_path = run / "report.json"
        protocol_path = run / "protocol.json"
        protocol = read_json(protocol_path) if protocol_path.exists() else {}
        cfg = protocol.get("schema_version") == "ditserve.research.cfg.frozen.v1"
        dual = cfg or protocol.get("schema_version") == "ditserve.research.dual.frozen.v1"
        if dual:
            active_path = run / "active-unit.json"
            active = read_json(active_path) if active_path.exists() else {}
            attempts = list(run.glob("*-attempt.json"))
            unresolved = [p for p in attempts if read_json(p).get("status") not in TERMINAL]
            pending_path = run / "pending-references.json"
            pending = read_json(pending_path) if pending_path.exists() else {}
            if active.get("status") == "running" or unresolved or pending.get("paths") or pending.get("pending"):
                protected.add(str(run))
                exceptions.append(dict(path=str(run), reason="dual unit/stage/continuation remains active"))
                continue
        if not report_path.exists():
            protected.add(str(run))
            exceptions.append(dict(path=str(run), reason="run has no terminal report; may resume"))
            continue
        report = read_json(report_path)
        if report.get("status") not in TERMINAL:
            protected.add(str(run))
            exceptions.append(dict(path=str(run), reason="run is not terminal"))
            continue
        rows = []
        for line in records.read_text().splitlines():
            row = _check_seal(json.loads(line))
            if (report.get("protocol_sha256") and row.get("protocol_sha256") != report["protocol_sha256"]):
                raise ValueError("record/report protocol mismatch")
            rows.append(row)
        sources.update((str(records), str(report_path)))
        recorded_times = [r["recorded_at"] for r in rows if isinstance(r.get("recorded_at"), (int, float))]
        ended_at = max(recorded_times) * 1e9 if recorded_times else records.stat().st_mtime_ns
        selected = None
        joint_selected = []
        action_selection_path = run / "action-selection.json"
        if not dual and action_selection_path.exists():
            from ..benchmark.common import verify_seal
            action_selection = verify_seal(read_json(action_selection_path))
            if action_selection.get("protocol_sha256") != report.get("protocol_sha256"):
                raise ValueError("joint action selection/report protocol mismatch")
            joint_selected = [c["candidate_id"] for c in action_selection.get("selected", [])]
            sources.add(str(action_selection_path))
        migration_selected = None
        if dual:
            sources.add(str(protocol_path))
            migration_selection = run / "migration-selection.json"
            if migration_selection.exists():
                migration_selected = _check_seal(read_json(migration_selection)).get("selected")
                sources.add(str(migration_selection))
        selection_path = run / "calibration.json"
        if selection_path.exists():
            selection = read_json(selection_path)
            if selection.get("sha256"):
                _check_seal(selection)
            selected = (selection.get("selected") or {}).get("recipe_id")
            sources.add(str(selection_path))
        if cfg and (run / "selection.json").exists():
            selection_path = run / "selection.json"
            selection = _check_seal(read_json(selection_path))
            if (selection.get("protocol_sha256") != report.get("protocol_sha256")
                    or selection.get("source_sha256") != protocol.get("source_sha256")):
                raise ValueError("CFG selection is not bound to the reported protocol/source")
            selected = selection.get("selected") if selection.get("status") == "selected" else None
            if selected not in (None, "independent_004", "paired_008"):
                raise ValueError("unknown selected CFG policy")
            sources.add(str(selection_path))
        for row in rows:
            artifacts = _artifacts(row, run)
            if not artifacts and not _evidence_output_hashes(row):
                continue
            unit_complete = True
            if dual:
                unit_path = run / "units" / (str(row.get("unit_id", "")) + ".json")
                if not unit_path.exists():
                    unit_complete = False
                else:
                    unit = _check_seal(read_json(unit_path))
                    if (row["sha256"] not in unit.get("records", [])
                            or unit.get("protocol_sha256") != row.get("protocol_sha256")
                            or unit.get("source_sha256") != row.get("source_sha256")):
                        raise ValueError("dual record is not bound to its sealed unit")
                    unit_complete = unit.get("status") == "completed"
                    sources.add(str(unit_path))
            evidence = row.get("evidence", {})
            reference_hashes = sorted(_evidence_output_hashes(row))
            reference_hashes += row.get("reference_output_sha256s", [])
            case_id = str(row.get("case_id", ""))
            kind = row.get("type")
            correctness = kind == "correctness" and row.get("status") == "passed" and (
                ("baseline" in row and "candidate" in row
                 and row.get("action") in ("hot", "exact_refresh", "full_migrate", "certified_omit"))
                or ("continuous" in row and "full" in row))
            policy = row.get("recipe_id", row.get("candidate_id"))
            selected_policy, loss = selected, _metrics(row)
            series_name = str(run.parent.relative_to(root))
            split = row.get("split", "dev" if kind == "calibration" else "unknown")
            evaluated_kind = kind in ("correctness", "probe", "calibration", "joint_quality", "state_trial", "memory_trial", "sp_trial", "migration_quality", "backend_pair", "backend_control", "migration_screen", "queue_probe", "mechanism", "capacity")
            extra = {}
            if cfg:
                # Keep the original case ID. Opposite orders can deliberately
                # share it; their immutable row seals distinguish the evidence.
                extra["evidence_id"] = row["sha256"]
                candidate = row.get("candidate", row.get("continuous", row.get("clean", row.get("run", {}))))
                policy = policy or candidate.get("recipe_id")
                role = str(row.get("device_role", candidate.get("device_role", "unknown")))
                implementation = candidate.get("identity", {}).get("implementation", "unknown")
                correctness = row.get("equal") is True and row.get("status") == "completed" and (
                    kind in ("cfg_correctness", "cfg_implementation_equivalence") and "continuous" in row and "restored" in row
                    or kind == "cfg_diagnostic_neutrality" and "clean" in row and "instrumented" in row)
                if kind in ("cfg_correctness", "cfg_screen", "cfg_calibration"):
                    series_name += "/cache-" + role + "/" + str(policy) + "/" + implementation
                    # An oracle screen is never a selected production policy
                    # sample, even after its split-prefix implementation wins.
                    selected_policy = selected if kind == "cfg_calibration" else None
                    if kind == "cfg_screen":
                        split = "stage_a"
                elif kind == "cfg_implementation_equivalence":
                    series_name += "/implementation-equivalence-" + role + "/" + str(policy)
                    selected_policy = None
                elif kind == "cfg_diagnostic_neutrality":
                    series_name += "/diagnostic-neutrality-" + role + "/" + str(policy)
                    selected_policy = None
                elif kind == "cfg_diagnostic":
                    series_name += "/diagnostic-" + role + "/" + str(policy)
                    selected_policy = None
                elif kind in ("cfg_migration_quality", "cfg_migration_screen", "cfg_migration_reference"):
                    series_name += "/migration/" + str(policy) + "/" + str(row.get("action", "path-reference"))
                    selected_policy = selected if kind == "cfg_migration_quality" else None
                    losses = [_metrics(dict(metrics=row.get(key, {}))) for key in ("canonical_quality", "path_quality")]
                    loss = max(losses) if all(x is not None for x in losses) else _metrics(dict(metrics=row.get("path_reference_quality", {})))
                else:
                    series_name += "/mechanism/" + str(policy)
                    selected_policy = None
                evaluated_kind = kind in ("cfg_correctness", "cfg_implementation_equivalence", "cfg_diagnostic_neutrality", "cfg_screen",
                    "cfg_calibration", "cfg_diagnostic", "cfg_migration_quality", "cfg_migration_screen",
                    "cfg_migration_reference", "cfg_queue_probe", "cfg_mechanism")
                if kind in ("cfg_correctness", "cfg_implementation_equivalence", "cfg_diagnostic_neutrality") and not correctness:
                    evaluated_kind = False  # Preserve failed or missing state-equivalence evidence.
                if kind in ("cfg_screen", "cfg_calibration", "cfg_diagnostic", "cfg_migration_quality") and loss is None:
                    evaluated_kind = False
            elif dual:
                if kind == "calibration":
                    series_name += "/cache-" + str(row.get("device_role", "unknown"))
                    selected_policy = "t008"
                elif kind in ("correctness", "migration_quality", "migration_screen"):
                    series_name += "/migration"
                    selected_policy, policy = migration_selected, row.get("action")
                    losses = [_metrics(dict(metrics=row.get(key, {}))) for key in ("canonical_quality", "path_quality")]
                    loss = max(losses) if all(x is not None for x in losses) else None
                else:
                    series_name += "/mechanism"
            else:
                # Initial cache qualification and joint recovery are different
                # policies. Never let their quality distributions replace one
                # another's development median/worst samples.
                if kind == "calibration":
                    series_name += "/cache"
                elif kind == "joint_quality":
                    policy = row.get("candidate_id")
                    selected_policy = policy if policy in joint_selected else None
                    series_name += "/state" + ("/" + str(policy) if len(joint_selected) > 1 else "")
                else:
                    series_name += "/state"
            cases.append(dict(series=series_name, run=str(run),
                ended_at=ended_at, case_id=case_id, kind="correctness" if correctness else ("correctness_other" if kind == "correctness" else kind),
                split=split,
                policy=policy, selected_policy=selected_policy,
                quality=loss, quality_order="max_of_two_complete_reference_means" if kind in ("migration_quality", "cfg_migration_quality") else "lpips_mean",
                failed=row.get("status") in ("failed", "unqualified") or (loss is not None and loss > .05),
                artifacts=artifacts, reference_sha256s=[h for h in reference_hashes if h], verified=unit_complete,
                evaluated=unit_complete and evaluated_kind, **extra))
    # Benchmark evaluation files bind exact run bytes and enumerate all candidate
    # and reference arrays. Cross-backend quality adds its necessary references.
    for evaluation in sorted(root.rglob("evaluation.json")):
        if "report/evidence" in str(evaluation.relative_to(root)):
            continue
        run_path = evaluation.parent / "run.json"
        if not run_path.exists():
            continue
        value, run = read_json(evaluation), read_json(run_path)
        if value.get("schema_version") != "ditserve.benchmark.evaluation.v1":
            continue
        from ..benchmark.common import verify_seal
        verify_seal(value)
        verify_seal(run)
        if run.get("status") != "completed":
            protected.add(str(evaluation.parent))
            continue
        if value.get("run_file_sha256") != file_digest(run_path) or value.get("run_sha256") != run.get("sha256"):
            raise ValueError("benchmark evaluation no longer matches run")
        sources.update((str(evaluation), str(run_path)))
        quality_path = evaluation.parent / "backend-quality.json"
        quality = read_json(quality_path) if quality_path.exists() else {}
        if quality:
            verify_seal(quality)
            sources.add(str(quality_path))
        reference_run = None
        if quality.get("reference_run_path"):
            ref_path = _logical(evaluation.parent, quality["reference_run_path"])
            if file_digest(ref_path) != quality.get("reference_run_file_sha256"):
                raise ValueError("backend quality reference run changed")
            reference_run = verify_seal(read_json(ref_path))
            sources.add(str(ref_path))
        for policy, values in value.get("policies", {}).items():
            for request_id, row in values.get("requests", {}).items():
                q = quality.get("requests", {}).get(request_id, {})
                refs = []
                if reference_run:
                    refs = _artifacts(reference_run.get("references", {}).get(request_id, {}), ref_path.parent)
                artifacts = _artifacts(row, evaluation.parent) + refs
                if not artifacts:
                    continue
                cases.append(dict(series=str(evaluation.parent.parent.relative_to(root)), run=str(evaluation.parent),
                    ended_at=evaluation.stat().st_mtime_ns, case_id=policy + "/" + request_id,
                    kind="correctness", split="holdout" if q else "unknown", policy=policy, selected_policy=None,
                    quality=_metrics(q), failed=row.get("qualified") is False or q.get("status") == "failed",
                    artifacts=artifacts, reference_sha256s=[], verified=True, evaluated=True))
    return dict(cases=cases, sources=sorted(sources), exceptions=exceptions, protected_roots=sorted(protected))


def select_key_samples(cases):
    """Latest verified terminal run per series, deterministic ties by case ID.

    Choose the observed sample closest to the numeric median; ties use case ID.
    A sample is selected once with all reasons; its full reference closure is added by
    ``build_plan``. Only the median requires a selected policy; the worst
    development sample remains necessary when every candidate is rejected.
    """
    result = []
    series = sorted({case["series"] for case in cases if case.get("verified")})
    for name in series:
        group = [c for c in cases if c["series"] == name and c.get("verified")]
        latest = max(group, key=lambda c: (c["ended_at"], c["run"]))["run"]
        group = [c for c in group if c["run"] == latest]
        selected = {}
        def tie(case):
            return (case["case_id"], case.get("evidence_id", ""))
        def keep(case, reason):
            key = (case["run"], case.get("evidence_id", case["case_id"]))
            selected.setdefault(key, dict(case, reasons=[]))["reasons"].append(reason)
        correct = sorted((c for c in group if c["kind"] == "correctness"), key=tie)
        if correct:
            keep(correct[0], "continuous_restore_or_migration_correctness")
        dev = [c for c in group if c["split"] == "dev" and c.get("quality") is not None]
        selected_dev = [c for c in dev if c.get("selected_policy") is not None
                        and c.get("policy") == c["selected_policy"]]
        if selected_dev:
            median = statistics.median(Decimal(str(c["quality"])) for c in selected_dev)
            keep(min(selected_dev, key=lambda c: (abs(Decimal(str(c["quality"])) - median), tie(c))),
                 "selected_policy_development_median")
        if dev:
            keep(min(dev, key=lambda c: (-c["quality"], tie(c))), "development_worst")
        for split, reason in (("stage_a", "prototype_screen_worst"),
                              ("holdout", "holdout_worst"), (None, "quality_failure_worst")):
            subset = [c for c in group if c.get("quality") is not None
                      and (c["split"] == split if split else c.get("failed"))]
            if subset:
                keep(min(subset, key=lambda c: (-c["quality"], tie(c))), reason)
        result.extend(selected.values())
    return sorted(result, key=lambda c: (c["series"], c["run"], c["case_id"], c.get("evidence_id", "")))


def _raw_files(root, external_roots):
    """Follow only explicitly project-owned external directories, without cycles."""
    seen = set()
    allowed = [root] + [Path(p).resolve() for p in external_roots]
    for directory, subdirs, files in os.walk(root, followlinks=True):
        physical = Path(directory).resolve()
        if physical in seen or not any(_inside(physical, p) for p in allowed):
            subdirs[:] = []
            continue
        seen.add(physical)
        subdirs[:] = sorted(d for d in subdirs if d not in (".retention", ".git"))
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() in RAW_SUFFIXES:
                yield path


def _snapshot(path):
    st = Path(path).stat()
    return dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                blocks_bytes=st.st_blocks * 512, links=st.st_nlink)



def _pinned_legacy_pairing(run):
    """Prove the legacy video/final-latent filename pair from archived code."""
    protocol_path = Path(run) / "protocol.json"
    if not protocol_path.is_file():
        return None
    protocol = read_json(protocol_path)
    relative = "python/ditserve/research/experiment.py"
    expected = protocol.get("source_files", {}).get(relative)
    source = Path(run) / "sources" / relative
    if not expected:
        return None
    _check_seal(protocol)
    if not source.is_file() or file_digest(source) != expected:
        return None
    text = source.read_text()
    required = ('artifact = self.root / "arrays" / spec["request_id"]',
                'np.save(str(artifact) + "-video.npy", output, allow_pickle=False)',
                'np.save(str(artifact) + "-latent.npy", latent, allow_pickle=False)',
                'request_id="research-%s-%d" % (self.job["stage"], self.counter)')
    if not all(fragment in text for fragment in required):
        return None
    return dict(source_path=str(source), source_sha256=expected,
                rule="same project-generated request_id produces video and final latent")


def build_plan(results_root, artifact_root=DEFAULT_ARTIFACT_ROOT, ledger_paths=(),
               project_external_roots=(), pending_references=(), catalog=None):
    """Make a sealed file-level plan; content hashing is deferred to verification."""
    root = Path(results_root).absolute()
    artifact_root = Path(artifact_root).absolute()
    if _inside(artifact_root, root) or _inside(root, artifact_root):
        raise ValueError("key artifact root must be separate from results")
    with idle_guard(root, ledger_paths):
        catalog = catalog if catalog is not None else discover_catalog(root)
        cases = catalog["cases"]
        selected = select_key_samples(cases)
        by_hash, known = {}, {}
        for case in cases:
            for item in case["artifacts"]:
                path = str(_logical(case["run"], item["path"]))
                previous = known.get(path)
                if previous and previous.get("sha256") and item.get("sha256") and previous["sha256"] != item["sha256"]:
                    raise ValueError("conflicting artifact hashes across cases")
                known[path] = dict(item, path=path, evaluated=case.get("evaluated", False) and (previous or {}).get("evaluated", True))
                if item.get("sha256"):
                    by_hash.setdefault(item["sha256"], set()).add(path)
        # Previously removed source paths remain meaningful declarations.
        # Their raw key copies and deletion journals are checked below; do not
        # require deleted sources to reappear on subsequent cleanup passes.
        prior_index = _load_index(root)
        for name, entry in prior_index.get("entries", {}).items():
            if _inside(Path(name), root):
                previous = known.get(name)
                if previous is None or not previous.get("sha256"):
                    known[name] = dict(path=name, sha256=entry["sha256"], kind="prior_retention", evaluated=True)
                by_hash.setdefault(entry["sha256"], set()).add(name)
        # Recover all evaluated implicit outputs, not just selected key samples.
        # A file's bytes must match an output hash in a sealed row of this run.
        bindings, unresolved_roots = [], set()
        allowed = [root.resolve()] + [Path(p).resolve() for p in project_external_roots]
        declared_by_run = {}
        for case in cases:
            if not case.get("evaluated"):
                continue
            hashes = set(case.get("reference_sha256s", []))
            hashes.update(a["sha256"] for a in case["artifacts"] if a.get("kind") == "output" and a.get("sha256"))
            declared_by_run.setdefault(case["run"], set()).update(hashes)
        for run_name, declared in sorted(declared_by_run.items()):
            array_root = Path(run_name) / "arrays"
            if (not _inside(array_root.absolute(), root)
                    or not any(_inside(array_root.resolve(), p) for p in allowed)):
                if any(h not in by_hash for h in declared):
                    unresolved_roots.add(run_name)
                continue
            for path in sorted(array_root.glob("research-*-video.npy")):
                if (str(path) in known or path.is_symlink() or not path.is_file()
                        or path.stat().st_nlink != 1 or not any(_inside(path.resolve(), p) for p in allowed)):
                    continue
                actual = file_digest(path)
                if actual in declared:
                    name = str(path.absolute())
                    known[name] = dict(path=name, sha256=actual, kind="output", evaluated=True)
                    by_hash.setdefault(actual, set()).add(name)
                    bindings.append(dict(path=name, sha256=actual, binding="sealed_output_hash_without_historical_path"))
            pairing = _pinned_legacy_pairing(run_name)
            if pairing:
                for name, item in list(known.items()):
                    video = Path(name)
                    if (video.parent != array_root or not item.get("evaluated")
                            or not re.fullmatch(r"research-(?:correctness|probe|calibrate|mechanism)-[1-9][0-9]*-video\.npy", video.name)):
                        continue
                    latent = video.with_name(video.name.replace("-video.npy", "-latent.npy"))
                    if str(latent) not in known and latent.is_file() and not latent.is_symlink():
                        known[str(latent)] = dict(path=str(latent), sha256=None, kind="latent", evaluated=True)
                        bindings.append(dict(path=str(latent), binding="pinned_generator_final_latent_pair",
                                             paired_video=name, **pairing))
        # Only a missing key reference blocks the run. Unknown unevaluated
        # files remain exceptions; a declared explicit missing path still fails.
        for case in selected:
            if any(h not in by_hash for h in case.get("reference_sha256s", [])):
                unresolved_roots.add(case["run"])
        keep = set()
        for case in selected:
            keep.update(str(_logical(case["run"], a["path"])) for a in case["artifacts"])
            for expected in case.get("reference_sha256s", []):
                if expected not in by_hash:
                    # No pathname was declared for this old hash. Keep this
                    # run intact; a declared pathname missing below still fails.
                    continue
                # Prefer one deterministic copy, exact bytes are deduplicated later.
                keep.add(sorted(by_hash[expected])[0])
        # Report export duplicates are named by the exact file SHA and tied to
        # a sealed evaluated original. Unknown hash-named files are not deleted.
        for copy_path in root.rglob("*.npy"):
            if ("artifacts" in copy_path.parts and copy_path.stem in by_hash
                    and len(copy_path.stem) == 64):
                known.setdefault(str(copy_path.absolute()), dict(path=str(copy_path.absolute()),
                    sha256=copy_path.stem, kind="report_copy", evaluated=True))
        pending = {str(_logical(root, p)) for p in pending_references}
        protected = [Path(p).absolute() for p in list(catalog.get("protected_roots", [])) + sorted(unresolved_roots)]
        allowed = [root.resolve()] + [Path(p).resolve() for p in project_external_roots]
        files, exceptions = [], list(catalog.get("exceptions", []))
        exceptions.extend(dict(path=name, reason="declared output hash has no identifiable owned raw reference; run preserved")
                          for name in sorted(unresolved_roots))
        discovered = {str(p.absolute()): p for p in _raw_files(root, project_external_roots)}
        for path in known:
            discovered.setdefault(path, Path(path))
        for name, path in sorted(discovered.items()):
            record = known.get(name)
            resolved = path.resolve()
            reason = None
            if name in pending or any(_inside(path, p) for p in protected):
                reason = "active or pending continuation reference"
            elif not _inside(path.absolute(), root):
                reason = "reference outside this results tree; ownership unclear"
            elif not any(_inside(resolved, p) for p in allowed):
                reason = "external target is not explicitly project-owned"
            elif path.is_symlink():
                reason = "individual symlink may be shared"
            elif not path.exists():
                # Declared pruning can be repeated; unexpected missing evidence fails.
                if record and record.get("sha256"):
                    status = verify_reference(root, path, record["sha256"])
                    if name in keep and status["status"] != "raw_available":
                        raise ValueError("selected key sample was already pruned without a retained raw copy: " + name)
                    if status["status"] in ("retained_metrics_only", "raw_available"):
                        continue
                raise FileNotFoundError("undeclared missing raw artifact: " + name)
            elif not stat.S_ISREG(path.stat().st_mode) or path.stat().st_nlink != 1:
                reason = "nonregular or hard-linked shared file"
            elif record is None or not record.get("evaluated"):
                reason = "unrecognized or unevaluated artifact; retained conservatively"
            if reason:
                exceptions.append(dict(path=name, reason=reason))
                continue
            files.append(dict(path=name, resolved_path=str(resolved), expected_sha256=record.get("sha256"),
                              action="retain_key" if name in keep else "prune", snapshot=_snapshot(path)))
        # All original metadata bytes are checked before and after application.
        metadata_paths = {Path(p).absolute() for p in catalog.get("sources", [])}
        for metadata_path in root.rglob("*"):
            if (metadata_path.is_file() and metadata_path.suffix in (".json", ".jsonl")
                    and ".retention" not in metadata_path.parts
                    and not metadata_path.name.startswith("retention-")):
                metadata_paths.add(metadata_path.absolute())
        metadata = {str(p): file_digest(p) for p in sorted(metadata_paths)}
        return _seal(dict(schema_version=SCHEMA, kind="plan", policy="summary_and_key_samples",
            results_root=str(root), artifact_root=str(artifact_root), ledger_paths=_ledger_paths(root, ledger_paths),
            project_external_roots=[str(Path(p).resolve()) for p in project_external_roots],
            created_at=time.time(), key_samples=selected, files=files, exceptions=exceptions,
            implicit_reference_bindings=bindings,
            metadata_sha256=metadata, statistics_sha256=digest(metadata),
            estimated_logical_prune_bytes=sum(f["snapshot"]["bytes"] for f in files if f["action"] == "prune")))


def _validate_plan(plan):
    _check_seal(plan)
    if plan.get("schema_version") != SCHEMA or plan.get("kind") != "plan":
        raise ValueError("unsupported retention plan")
    root = Path(plan["results_root"])
    for item in plan["files"]:
        if item["action"] not in ("retain_key", "prune") or not _inside(Path(item["path"]), root):
            raise ValueError("unsafe retention manifest action/path")
    return root


def _check_metadata(plan):
    for path, expected in plan["metadata_sha256"].items():
        if file_digest(path) != expected:
            raise ValueError("historical structured metrics changed: " + path)


def _unchanged(item):
    path = Path(item["path"])
    if path.is_symlink() or str(path.resolve()) != item["resolved_path"] or _snapshot(path) != item["snapshot"]:
        raise ValueError("artifact changed or shared since retention plan: " + str(path))


def verify_plan(plan):
    """Hash all planned source files before allowing the first deletion."""
    root = _validate_plan(plan)
    with idle_guard(root, plan["ledger_paths"]):
        _check_metadata(plan)
        hashes = {}
        for item in plan["files"]:
            _unchanged(item)
            actual = file_digest(item["path"])
            if item.get("expected_sha256") and actual != item["expected_sha256"]:
                raise ValueError("raw evidence hash mismatch: " + item["path"])
            hashes[item["path"]] = actual
        return _seal(dict(schema_version=SCHEMA, kind="predelete_verification", plan_sha256=plan["sha256"],
                          verified_at=time.time(), file_sha256=hashes, statistics_sha256=plan["statistics_sha256"]))


def _save_index(root, index):
    atomic_json(Path(root) / "retention-index.json", _seal(index))


def _load_index(root):
    path = Path(root) / "retention-index.json"
    if path.exists():
        return _check_seal(read_json(path))
    return dict(schema_version=SCHEMA, kind="index", entries={}, applications=[])


def _copy_verified(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_digest(destination) != expected:
            raise ValueError("content-addressed key artifact corrupted")
        return 0
    fd, name = tempfile.mkstemp(prefix=".retention-copy-", dir=str(destination.parent))
    temporary = Path(name)
    try:
        with Path(source).open("rb") as src, os.fdopen(fd, "wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if file_digest(temporary) != expected:
            raise ValueError("key artifact copy verification failed")
        copied = temporary.stat().st_blocks * 512
        os.replace(temporary, destination)
        return copied
    finally:
        temporary.unlink(missing_ok=True)


def apply_plan(plan, verification):
    """Copy/verify every key file, journal removals, then unlink exact planned files.

    A deletion intent is durably written before unlink. On interruption the same
    plan/verification safely resumes; other missing files remain errors. Numeric
    metrics are never rewritten. The returned byte count uses allocated blocks,
    with separately reported key-store growth and logical bytes.
    """
    root = _validate_plan(plan)
    _check_seal(verification)
    if verification.get("kind") != "predelete_verification" or verification.get("plan_sha256") != plan["sha256"]:
        raise ValueError("predelete verification does not match plan")
    with idle_guard(root, plan["ledger_paths"]):
        _check_metadata(plan)
        store = Path(plan["artifact_root"])
        index = _load_index(root)
        journal_dir = root / ".retention" / plan["sha256"]
        journal_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(journal_dir / "plan.json", plan)
        atomic_json(journal_dir / "verification.json", verification)
        journal_path = journal_dir / "journal.json"
        journal = _check_seal(read_json(journal_path)) if journal_path.exists() else dict(
            schema_version=SCHEMA, plan_sha256=plan["sha256"], state="copying", entries={}, copied_blocks_bytes=0)
        if journal.get("plan_sha256") != plan["sha256"]:
            raise ValueError("retention journal plan mismatch")
        def save():
            atomic_json(journal_path, _seal(journal))
        # Recheck all files before first destructive action, even after a delay.
        for item in plan["files"]:
            entry = journal["entries"].get(item["path"], {})
            if entry.get("state") in ("delete_intent", "deleted"):
                continue
            _unchanged(item)
            if file_digest(item["path"]) != verification["file_sha256"].get(item["path"]):
                raise ValueError("artifact content changed since verification")
        for item in plan["files"]:
            if item["action"] != "retain_key":
                continue
            expected = verification["file_sha256"][item["path"]]
            destination = store / "sha256" / expected[:2] / expected
            if destination.is_symlink() or not _inside(destination.resolve(), store.resolve()):
                raise ValueError("unsafe key artifact destination")
            if not destination.exists():
                journal["copied_blocks_bytes"] += _copy_verified(item["path"], destination, expected)
                save()
            elif file_digest(destination) != expected:
                raise ValueError("retained key artifact no longer matches content hash")
            item_entry = journal["entries"].setdefault(item["path"], {})
            item_entry["retained_path"] = str(destination)
            save()
        journal["state"] = "deleting"
        save()
        retained_by_hash = {entry["sha256"]: entry["retained_path"] for entry in index["entries"].values()
                            if entry.get("status") == "raw_available" and entry.get("retained_path")}
        retained_by_hash.update({verification["file_sha256"][name]: entry["retained_path"]
                                for name, entry in journal["entries"].items() if entry.get("retained_path")})
        for item in plan["files"]:
            path = Path(item["path"])
            entry = journal["entries"].setdefault(str(path), {})
            expected = verification["file_sha256"][str(path)]
            if entry.get("state") != "deleted":
                if path.exists():
                    _unchanged(item)
                    if file_digest(path) != expected:
                        raise ValueError("artifact changed immediately before deletion")
                    entry["state"] = "delete_intent"
                    save()
                    path.unlink()
                elif entry.get("state") != "delete_intent":
                    raise FileNotFoundError("unexpected missing artifact during cleanup: " + str(path))
                entry.update(state="deleted", blocks_bytes=item["snapshot"]["blocks_bytes"], bytes=item["snapshot"]["bytes"])
                save()
            index["entries"][str(path)] = dict(sha256=expected,
                status="raw_available" if expected in retained_by_hash else "retained_metrics_only",
                retained_path=retained_by_hash.get(expected), plan_sha256=plan["sha256"],
                source_removed=True, original_bytes=item["snapshot"]["bytes"], declared_at=time.time())
            _save_index(root, index)
        _check_metadata(plan)
        journal["state"] = "completed"
        save()
        result = _seal(dict(schema_version=SCHEMA, kind="completion", status="completed", plan_sha256=plan["sha256"],
            removed_blocks_bytes=sum(e["blocks_bytes"] for e in journal["entries"].values()),
            removed_logical_bytes=sum(e["bytes"] for e in journal["entries"].values()),
            key_store_added_blocks_bytes=journal["copied_blocks_bytes"],
            net_released_blocks_bytes=sum(e["blocks_bytes"] for e in journal["entries"].values()) - journal["copied_blocks_bytes"],
            key_sample_groups=len(plan["key_samples"]),
            unique_key_files=len({e["retained_path"] for e in index["entries"].values()
                                  if e.get("status") == "raw_available" and e.get("retained_path")}),
            key_files_processed_this_pass=len({e.get("retained_path") for e in journal["entries"].values() if e.get("retained_path")}),
            exceptions=len(plan["exceptions"]), statistics_unchanged=True, statistics_sha256=plan["statistics_sha256"]))
        atomic_json(journal_dir / "completion.json", result)
        if plan["sha256"] not in index["applications"]:
            index["applications"].append(plan["sha256"])
        _save_index(root, index)
        return result


def verify_reference(root, path, expected_hash):
    """Verify raw bytes or an explicit deletion declaration; never silently skip.

    Callers must label ``retained_metrics_only`` as recorded historical metrics,
    not as a new raw-data re-evaluation. The historical record hash stays intact.
    """
    root = Path(root).absolute()
    logical = _logical(root, path)
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("artifact file SHA256 is required")
    if logical.exists():
        if file_digest(logical) != expected_hash:
            raise ValueError("raw artifact changed: " + str(logical))
        return dict(status="raw_available", path=str(logical), re_evaluable=True, sha256=expected_hash)
    # The report root may be a run below the project results root.
    for directory in (root,) + tuple(root.parents):
        index_path = directory / "retention-index.json"
        if not index_path.exists():
            continue
        index = _load_index(directory)
        entry = index.get("entries", {}).get(str(logical))
        if not entry:
            continue
        if entry.get("sha256") != expected_hash:
            raise ValueError("retention declaration hash differs from sealed evidence")
        application = directory / ".retention" / entry.get("plan_sha256", "")
        try:
            saved_plan = _check_seal(read_json(application / "plan.json"))
            saved_verification = _check_seal(read_json(application / "verification.json"))
            saved_journal = _check_seal(read_json(application / "journal.json"))
        except (OSError, ValueError) as error:
            raise ValueError("retention declaration has no valid deletion journal") from error
        if (saved_plan["sha256"] != entry["plan_sha256"]
                or saved_plan["results_root"] != str(directory)
                or saved_verification.get("plan_sha256") != saved_plan["sha256"]
                or saved_verification.get("file_sha256", {}).get(str(logical)) != expected_hash
                or saved_journal.get("plan_sha256") != saved_plan["sha256"]
                or saved_journal.get("entries", {}).get(str(logical), {}).get("state") != "deleted"):
            raise ValueError("retention declaration is not backed by completed file deletion")
        if entry.get("status") == "raw_available":
            retained = Path(entry.get("retained_path") or "")
            if not retained.is_file() or file_digest(retained) != expected_hash:
                raise ValueError("retained key raw artifact changed or missing")
            return dict(status="raw_available", path=str(retained), re_evaluable=True, sha256=expected_hash)
        if entry.get("status") == "retained_metrics_only" and entry.get("source_removed") is True:
            return dict(status="retained_metrics_only", path=None, re_evaluable=False, sha256=expected_hash,
                        plan_sha256=entry["plan_sha256"])
        raise ValueError("invalid retention declaration")
    from ..compact_results import verify_compacted_reference
    try:
        return verify_compacted_reference(root, logical, expected_hash)
    except FileNotFoundError as error:
        raise FileNotFoundError("undeclared missing raw artifact: " + str(logical)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "verify", "apply"))
    parser.add_argument("--results", default="results")
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--ledger", action="append", default=[])
    parser.add_argument("--project-external-root", action="append", default=[])
    parser.add_argument("--pending-reference", action="append", default=[])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--verification")
    args = parser.parse_args(argv)
    if args.action == "plan":
        value = build_plan(args.results, args.artifact_root, args.ledger, args.project_external_root, args.pending_reference)
        atomic_json(args.plan, value)
        print(json.dumps(dict(plan=args.plan, files=len(value["files"]), key_sample_groups=len(value["key_samples"]),
                              exceptions=len(value["exceptions"]), estimated_logical_prune_bytes=value["estimated_logical_prune_bytes"])))
    elif args.action == "verify":
        if not args.verification:
            parser.error("verify requires --verification")
        value = verify_plan(read_json(args.plan))
        atomic_json(args.verification, value)
        print(json.dumps(dict(status="verified", files=len(value["file_sha256"]))))
    else:
        if not args.verification:
            parser.error("apply requires --verification")
        print(json.dumps(apply_plan(read_json(args.plan), read_json(args.verification)), sort_keys=True))


if __name__ == "__main__":
    main()
