"""Explicit, source-bound cleanup for the completed optimization run only.

This is a finalization API, never a reporting side effect. It uses real compact
archives of raw leaf directories; the main report's staging archive alone does
not authorize deletion. Imports and tiny fixtures require only the stdlib.
"""

from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
from pathlib import Path
import threading

from .. import compact_results as compact
from .protocol import ROOT, PHASES


OUTPUT = ROOT / "results/optimization-screen/run01"
CACHE = ROOT / ".cache/optimization-screen/run01"
LEDGER = ROOT / "results/optimization-screen/gpu-budget.json"
SCHEMA = "ditserve.optimizations.retention.v1"
MAX_JSON = 32 * 1024 * 1024
_MUTEX = threading.Lock()
_FORBIDDEN = {"sources", "source-snapshots", "compiler-cache", ".git", ".retention"}


def _inside(path, root):
    return path == root or root in path.parents


def _real(path):
    # compact._real rejects symlinks in every path component, including leaves.
    return compact._real(Path(path))


def _digest(path):
    size, digest, unused = compact._read(_real(path))
    return dict(path=str(_real(path)), bytes=size, sha256=digest)


def _json(path):
    path = _real(path)
    size, digest, raw = compact._read(path, collect=True, limit=MAX_JSON)
    return json.loads(raw), dict(path=str(path), bytes=size, sha256=digest)


def _pointer(value, pointer):
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("source pointer must be a JSON pointer")
    for part in pointer.split("/")[1:] if pointer else []:
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("an explicit SHA256 is required")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _terminal_bindings(output, ledger):
    budget, binding = _json(ledger)
    if budget.get("active") or budget.get("active_session"):
        raise ValueError("GPU accounting is active")
    bindings = [binding]
    for phase in PHASES:
        terminal, bound = _json(output / phase / "terminal.json")
        if terminal.get("status") not in ("passed", "stopped", "failed", "incomplete"):
            raise ValueError("every phase must have an explicit terminal record")
        if terminal.get("gpu_work_started") is not False:
            occupancy = terminal.get("final_occupancy", {})
            if (occupancy.get("status") != "exclusive_observed"
                    or occupancy.get("owned") != [] or occupancy.get("foreign") != []):
                raise ValueError("terminal GPU cleanup is not verified")
        bindings.append(bound)
    return bindings


def _main_archive(output, primary_archive, required_media):
    if isinstance(primary_archive, dict):
        primary, extra = primary_archive, []
    else:
        primary, source = _json(primary_archive)
        extra = [source]
    if primary.get("status") != "verified" or primary.get("verification", {}).get("status") != "verified":
        raise ValueError("main archive must already have passed readback")
    workbook = _real(primary["workbook"])
    if workbook.name != "results.xlsx" or not _inside(workbook, output):
        raise ValueError("main archive must belong to this run")
    destination = workbook.parent
    compact.verify(destination, check_sources=True)
    data = compact._load(destination)
    stage = _real(data["root"])
    if not _inside(stage, output) or stage == output:
        raise ValueError("main archive must have a bounded staging source")
    bindings = extra + [_digest(destination / name) for name in compact.FILES]
    bindings += [dict(path=str(stage / row["path"]), bytes=row["bytes"], sha256=row["sha256"])
                 for row in data["files"]]
    plan_path = _real(primary["retention_plan"])
    if not _inside(plan_path, stage):
        raise ValueError("main retention plan must be in verified staging")
    plan, plan_binding = _json(plan_path)
    if not any(row["path"] == str(plan_path) and row["sha256"] == plan_binding["sha256"] for row in bindings):
        raise ValueError("retention plan is absent from verified main archive")
    available = {row["path"]: row for row in plan.get("previews", []) + plan.get("videos", [])}
    if not isinstance(required_media, list) or not required_media:
        raise ValueError("explicit retained key PNG/MP4 media are required")
    media = []
    for item in required_media:
        path, expected = _real(item["path"]), _sha(item["sha256"])
        if not _inside(path, output) or path.suffix.lower() not in (".png", ".mp4"):
            raise ValueError("retained media must be run-local PNG/MP4")
        original = available.get(str(path))
        if original is None or original.get("sha256") != expected:
            raise ValueError("key media are not bound to the main archive")
        if path.suffix.lower() == ".mp4" and (not original.get("pair_complete") or original.get("frame_count") != 81):
            raise ValueError("key video pair has not completed frame readback")
        observed = _digest(path)
        if observed["sha256"] != expected:
            raise ValueError("key media hash changed")
        bindings.append(observed)
        media.append(observed)
    return bindings, media, dict(workbook=str(workbook), manifest=str(destination / "manifest.json"))


def _artifacts(output, artifacts):
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("a nonempty explicit artifact whitelist is required")
    selected, bindings, seen = [], [], set()
    for item in artifacts:
        path = _real(item["path"])
        if (path.suffix.lower() != ".npy" or not any(_inside(path, root) for root in (OUTPUT, CACHE))
                or any(part in _FORBIDDEN for part in path.parts)):
            raise ValueError("only this run's ordinary NPY artifacts are eligible")
        if str(path) in seen:
            raise ValueError("duplicate artifact path")
        seen.add(str(path))
        reason = item.get("closure_reason")
        if item.get("dependencies_closed") is not True or not isinstance(reason, str) or not reason.strip():
            raise ValueError("each artifact needs an explicit dependency closure reason")
        source = item["source"]
        source_path = _real(source["path"])
        if (not _inside(source_path, output) or source_path.suffix != ".json"
                or any(part in _FORBIDDEN for part in source_path.parts)):
            raise ValueError("artifact source must be this run's original JSON evidence")
        value, binding = _json(source_path)
        if binding["sha256"] != _sha(source["sha256"]):
            raise ValueError("artifact source JSON hash differs")
        recorded = _pointer(value, source["pointer"])
        recorded_path = Path(recorded["path"])
        recorded_path = recorded_path if recorded_path.is_absolute() else source_path.parent / recorded_path
        expected = _sha(item["sha256"])
        if _real(recorded_path) != path or recorded.get("sha256") != expected:
            raise ValueError("artifact pointer does not bind the selected path and hash")
        bindings.append(binding)
        selected.append(dict(path=str(path), sha256=expected, source=dict(binding, pointer=source["pointer"]),
                             dependencies_closed=True, closure_reason=reason.strip()))
    return sorted(selected, key=lambda row: row["path"]), bindings


def _validate(bindings, ledger):
    budget, unused = _json(ledger)
    if budget.get("active") or budget.get("active_session"):
        raise ValueError("GPU accounting became active")
    for row in bindings:
        if _digest(row["path"]) != row:
            raise ValueError("bound source changed: " + row["path"])


@contextmanager
def _cache_lock(ledger):
    with _real(ledger.with_suffix(".lock")).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def _validated_guard(bindings, ledger, leaves):
    """Scoped single-thread wrapper; restore the original even on interruption."""
    original = compact._guard

    @contextmanager
    def guarded(root, excluded=(), lock=False):
        if _real(root) not in leaves:
            raise ValueError("unexpected compact source during finalization")
        with original(root, excluded, lock=lock) as protected:
            _validate(bindings, ledger)
            yield protected
    compact._guard = guarded
    try:
        yield
    finally:
        compact._guard = original


def finalize(output, ledger, artifacts, *, primary_archive, required_media, destination=None):
    """Delete only explicitly closed, source-bound NPY files after real readback.

    Each artifact requires path, sha256, source={path, sha256, pointer},
    dependencies_closed=True, and closure_reason. No status implies closure.
    The caller must use a dedicated single-threaded finalization process.
    Failure preserves a partial receipt; replaying the same plan is idempotent.
    """
    output, ledger = _real(output), _real(ledger)
    if output != _real(OUTPUT) or ledger != _real(LEDGER):
        raise ValueError("finalization is restricted to optimization-screen/run01")
    if threading.active_count() != 1 or not _MUTEX.acquire(blocking=False):
        raise RuntimeError("finalization requires one dedicated single-threaded process")
    result = None
    try:
        bindings = _terminal_bindings(output, ledger)
        main_bindings, media, main = _main_archive(output, primary_archive, required_media)
        selected, source_bindings = _artifacts(output, artifacts)
        bindings += main_bindings + source_bindings
        bindings = sorted({row["path"]: row for row in bindings}.values(), key=lambda row: row["path"])
        _validate(bindings, ledger)
        destination = _real(destination or output / "raw-finalization")
        leaves = sorted({Path(row["path"]).parent for row in selected})
        if not _inside(destination, output) or any(_inside(destination, leaf) for leaf in leaves):
            raise ValueError("cleanup archive destination must be run-local and outside raw leaves")
        # Actual artifact leaves are flat. Refuse an accidentally broad attempt
        # root rather than recursively hash compiler caches or unrelated trees.
        if any(child.is_dir() and not child.is_symlink() for leaf in leaves for child in leaf.iterdir()):
            raise ValueError("raw source must be a flat artifact leaf, not a parent tree")
        plan = dict(schema_version=SCHEMA, output=str(output), ledger=str(ledger), bindings=bindings,
                    artifacts=selected, main_archive=main, retained_media=media,
                    scope="explicit NPY only; historical metrics preserved, raw quality reevaluation unavailable after deletion")
        destination.mkdir(parents=True, exist_ok=True)
        plan_path, receipt_path = destination / "plan.json", destination / "receipt.json"
        if plan_path.exists():
            if _json(plan_path)[0] != plan:
                raise ValueError("immutable cleanup plan differs")
        else:
            compact._write_json(plan_path, plan)
        result = dict(schema_version=SCHEMA, status="in_progress", plan=str(plan_path),
                      plan_sha256=hashlib.sha256(_canonical(plan).encode()).hexdigest(), groups=[],
                      deleted_files=0, deleted_bytes=0, retained_media=media, deleted_artifacts_re_evaluable=False,
                      accounting_scope="counts cover verified completed leaf groups; incomplete leaves require their durable receipts")
        compact._write_json(receipt_path, result)
        with _validated_guard(bindings, ledger, set(leaves)):
            for index, leaf in enumerate(leaves):
                chosen = [row for row in selected if Path(row["path"]).parent == leaf]
                archive = destination / ("raw-%03d" % index)
                group = dict(source_root=str(leaf), archive=str(archive), status="pending",
                             deletion_receipt=str(archive / compact.RECEIPT_NAME),
                             selected=[Path(row["path"]).name for row in chosen])
                result["groups"].append(group)
                compact._write_json(receipt_path, result)
                # Cache files have no results-ledger ancestor. Keep that exact
                # allocation lock through their export, readback and deletion.
                with _cache_lock(ledger) if _inside(leaf, CACHE) else nullcontext():
                    _validate(bindings, ledger)
                    if not archive.exists():
                        exported = compact.export(leaf, archive)
                        if exported.get("status") != "verified":
                            raise ValueError("raw leaf export is protected or unverified")
                    compact.verify(archive)
                    payload = compact._load(archive)
                    if payload["root"] != str(leaf):
                        raise ValueError("raw archive root differs")
                    entries = {row["path"]: row for row in payload["files"]}
                    for row in chosen:
                        entry = entries.get(Path(row["path"]).name, {})
                        if entry.get("kind") != "raw" or entry.get("sha256") != row["sha256"]:
                            raise ValueError("raw archive does not bind the expected artifact")
                    if not (archive / compact.RECEIPT_NAME).exists():
                        compact.verify(archive, check_sources=True)
                    pruned = compact.prune(archive, group["selected"])
                    for row in chosen:
                        compact.verify_compacted_reference(leaf, row["path"], row["sha256"])
                    _validate(bindings, ledger)
                    group.update(status="completed", workbook=str(archive / "results.xlsx"),
                                 deletion_receipt=str(archive / compact.RECEIPT_NAME),
                                 newly_deleted=pruned["deleted"], newly_deleted_bytes=pruned["bytes"],
                                 files=len(chosen), bytes=sum(entries[name]["bytes"] for name in group["selected"]))
                    result["deleted_files"] += group["files"]
                    result["deleted_bytes"] += group["bytes"]
                    compact._write_json(receipt_path, result)
        result["status"] = "completed"
        compact._write_json(receipt_path, result)
        return result
    except BaseException as error:
        if result is not None:
            result.update(status="incomplete", error=f"{type(error).__name__}: {error}",
                          unsettled_groups=[group["deletion_receipt"] for group in result["groups"]
                                            if group["status"] != "completed"])
            compact._write_json(receipt_path, result)
        raise
    finally:
        _MUTEX.release()
