"""Read-only verification of saved PreViS arrays and correctness pairs.

Array hashes cover NumPy C-order data bytes, not .npy file headers. Only the
derived verification.json is written. Run after GPU experiments have stopped;
scanning large arrays consumes host memory bandwidth even though it uses no GPU.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from ..util import atomic_json, utc_now
from . import previs_storage


FIELDS = ("path", "shape", "dtype", "sha256", "finite")
CHUNK_BYTES = 8 * 1024 * 1024


def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve(raw, root, source):
    if not isinstance(raw, str) or not raw:
        raise ValueError("artifact_path_must_be_nonempty_string")
    path = Path(raw)
    choices = [path] if path.is_absolute() else [root / path, source.parent / path, Path.cwd() / path]
    return previs_storage.resolve_candidates(choices, root).path


def _walk(value, location="$"):
    if isinstance(value, dict):
        path = value.get("path")
        # Step events also have path='compute'/'reuse'/'mixed' and finite=True;
        # neither describes an array file. A .npy path is sufficient to retain
        # malformed/incomplete references for validation. Otherwise require the
        # array-specific shape+dtype+hash signature, including missing-path cases.
        if ((isinstance(path, str) and path.lower().endswith(".npy"))
                or all(key in value for key in ("shape", "dtype", "sha256"))):
            yield location, value
        for key, child in value.items():
            yield from _walk(child, location + "." + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, location + "[" + str(index) + "]")


def _metadata(value, np):
    if any(key not in value for key in FIELDS):
        raise ValueError("incomplete_array_metadata")
    if (not isinstance(value["shape"], list)
            or any(type(dimension) is not int or dimension < 0 for dimension in value["shape"])
            or not isinstance(value["dtype"], str) or type(value["finite"]) is not bool):
        raise ValueError("invalid_shape_dtype_or_finite_metadata")
    digest = value["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise ValueError("invalid_array_sha256_metadata")
    try:
        dtype = str(np.dtype(value["dtype"]))
    except TypeError as error:
        raise ValueError("invalid_dtype_metadata") from error
    return {"shape": list(value["shape"]), "dtype": dtype, "finite": value["finite"], "sha256": digest.lower()}


def _chunks(array):
    # flat slices follow logical C order, including Fortran-order .npy arrays,
    # without flattening/copying the complete memory-mapped array at once.
    elements = max(1, CHUNK_BYTES // max(1, array.dtype.itemsize))
    for start in range(0, array.size, elements):
        yield array.flat[start:start + elements]


def _inspect(path, np):
    before = path.stat()
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise ValueError("artifact_is_not_a_nonobject_numpy_array")
        digest, finite = hashlib.sha256(), True
        for chunk in _chunks(array):
            digest.update(chunk.tobytes(order="C"))
            finite = bool(np.isfinite(chunk).all()) and finite
        actual = {"shape": list(array.shape), "dtype": str(array.dtype), "finite": finite,
                  "sha256": digest.hexdigest(), "array_bytes": int(array.nbytes)}
    finally:
        del array
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("artifact_changed_during_verification")
    return actual


def _equal(left, right, np):
    first, second = np.load(left, mmap_mode="r", allow_pickle=False), np.load(right, mmap_mode="r", allow_pickle=False)
    try:
        if first.shape != second.shape or first.dtype != second.dtype:
            return False
        return all(np.array_equal(a, b) for a, b in zip(_chunks(first), _chunks(second)))
    finally:
        del first, second


def verify_suite(suite_root):
    """Verify all records recursively; return and save an auditable report."""
    import numpy as np

    root = Path(suite_root).resolve()
    if not root.is_dir():
        raise ValueError("suite directory does not exist: " + str(root))
    records, sources, issues = [], [], []
    references, physical = {}, {}

    def issue(reason, **context):
        issues.append(dict(reason=reason, **context))

    try:
        storage = previs_storage.index_metadata(root)
    except (OSError, ValueError, TypeError) as error:
        storage = {"enabled": False, "invalid": True, "error": str(error)}
        issue("invalid_artifact_storage_index", error=str(error))

    for source in sorted(root.rglob("records.jsonl")):
        if not _within(source.resolve(), root):
            issue("record_stream_outside_suite", source=str(source))
            continue
        try:
            before = source.stat()
            raw = source.read_bytes()
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("record_stream_changed_during_verification")
            sources.append({"path": str(source), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
            lines = raw.decode("utf-8").splitlines()
        except (OSError, UnicodeError, ValueError) as error:
            issue("unreadable_record_stream", source=str(source), error=str(error))
            continue
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            context = {"source": str(source), "source_line": line_number}
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record is not an object")
            except ValueError as error:
                issue("invalid_record_json", **context, error=str(error))
                continue
            records.append((source, line_number, row))
            for location, descriptor in _walk(row):
                ref_key = (str(source), line_number, location)
                reference = dict(context, location=location, recorded_path=descriptor.get("path"))
                references[ref_key] = reference
                try:
                    metadata = _metadata(descriptor, np)
                    path = _resolve(descriptor["path"], root, source)
                    if path.suffix.lower() != ".npy" or not path.is_file():
                        raise ValueError("missing_or_non_npy_artifact")
                    stat = path.stat()
                    key = (stat.st_dev, stat.st_ino)
                    entry = physical.setdefault(key, {"path": str(path), "aliases": [], "metadata": [], "references": []})
                    if str(path) not in entry["aliases"]:
                        entry["aliases"].append(str(path))
                    if metadata not in entry["metadata"]:
                        entry["metadata"].append(metadata)
                    entry["references"].append(reference)
                    reference["physical_key"] = key
                except (OSError, TypeError, ValueError) as error:
                    reference["error"] = str(error)
                    issue("invalid_artifact_reference", **reference)

    for entry in physical.values():
        errors = []
        if len(entry["metadata"]) != 1:
            errors.append("conflicting_metadata_for_same_physical_file")
        try:
            actual = entry["actual"] = _inspect(Path(entry["path"]), np)
            for expected in entry["metadata"]:
                errors.extend("metadata_mismatch_" + field for field in FIELDS[1:] if expected[field] != actual[field])
            if actual["finite"] is not True:
                errors.append("nonfinite_array")
        except (OSError, TypeError, ValueError, EOFError) as error:
            errors.append("array_read_failed: " + str(error))
        entry["errors"] = sorted(set(errors))
        entry["status"] = "failed" if errors else "verified"
        for reason in entry["errors"]:
            issue(reason, path=entry["path"])

    by_source = defaultdict(list)
    for source, line_number, row in records:
        by_source[str(source)].append((line_number, row))

    def compare(source, line_number, row, kind, left, right):
        result = {"source": str(source), "source_line": line_number, "case_id": row.get("case_id"),
                  "device": row.get("device"), "recorded_status": row.get("status"), "array_kind": kind,
                  "recorded_exact_match": row.get(kind + "_exact_match"), "method": "chunked_numpy_array_equal"}
        if left is None or right is None:
            return dict(result, status="unavailable", reason="missing_saved_pair_no_cross_attempt_inference")
        left_ref, right_ref = references.get(left), references.get(right)
        entries = [physical.get(ref.get("physical_key")) if ref else None for ref in (left_ref, right_ref)]
        if any(entry is None or entry.get("status") != "verified" for entry in entries):
            return dict(result, status="failed", reason="pair_artifact_integrity_not_verified")
        left_path, right_path = [entry["path"] for entry in entries]
        try:
            equal = _equal(left_path, right_path, np)
        except (OSError, TypeError, ValueError, EOFError) as error:
            return dict(result, status="failed", reason="pair_comparison_error", error=str(error))
        return dict(result, status="passed" if equal else "failed", exact_match=equal,
                    reference_path=left_path, candidate_path=right_path)

    pairs = []
    for source, line_number, row in records:
        if row.get("type") != "correctness":
            continue
        def ref(line, field):
            key = (str(source), line, "$." + field)
            return key if key in references else None
        if row.get("case_id") in ("official-none", "official-cache"):
            output_pair = (ref(line_number, "output"), ref(line_number, "candidate"))
            latent_pair = (ref(line_number, "latent_reference"), ref(line_number, "latent_candidate"))
        elif row.get("case_id") == "cold-policy-interleave-16":
            # Cold reconstruction + a changed suffix policy is a different
            # numerical trajectory from continuous caching. Only its explicitly
            # saved same-trajectory reference is a valid interleaving oracle.
            valid_scope = (row.get("baseline_scope") == "same_cold_prefix_suffix_trajectory"
                           and row.get("cache") is True and row.get("mode") == "cold")
            output_pair = ((ref(line_number, "reference"), ref(line_number, "candidate"))
                           if valid_scope else (None, None))
            latent_pair = ((ref(line_number, "latent_reference"), ref(line_number, "latent_candidate"))
                           if valid_scope else (None, None))
        else:
            baseline = []
            if type(row.get("cache")) is bool:
                name = "official-cache" if row["cache"] else "official-none"
                baseline = [(line, candidate) for line, candidate in by_source[str(source)]
                            if candidate.get("type") == "correctness" and candidate.get("case_id") == name]
            baseline_line = baseline[0][0] if len(baseline) == 1 else None
            output_pair = (ref(baseline_line, "candidate"), ref(line_number, "output"))
            latent_pair = (ref(baseline_line, "latent_candidate"), ref(line_number, "latent"))
        pairs.append(compare(source, line_number, row, "output", *output_pair))
        pairs.append(compare(source, line_number, row, "latent", *latent_pair))

    try:
        if previs_storage.index_metadata(root) != storage:
            issue("artifact_storage_index_changed_during_verification")
    except (OSError, ValueError, TypeError) as error:
        issue("invalid_artifact_storage_index", error=str(error))
    failed_pairs = [pair for pair in pairs if pair["status"] == "failed"]
    unavailable = [pair for pair in pairs if pair["status"] == "unavailable"]
    status = ("failed" if issues or failed_pairs else "unavailable" if not physical else
              "verified_with_unavailable_pairs" if unavailable else "verified")
    latent_pairs = [pair for pair in pairs if pair["array_kind"] == "latent"]
    report = {"schema_version": "previs-artifact-verification.v1", "verified_at": utc_now(),
              "suite_root": str(root), "status": status, "numpy_version": np.__version__,
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "saved_array_integrity_and_available_correctness_pairs_not_runtime_or_quality_gate",
              "hash_semantics": "sha256(np_array.tobytes(order='C')); excludes npy header",
              "path_policy": "suite-contained paths or individually registered external NPY links; record streams remain within suite",
              "artifact_storage": storage,
              "storage_helper_sha256": hashlib.sha256(Path(previs_storage.__file__).read_bytes()).hexdigest(),
              "summary": {"record_streams": len(sources), "records": len(records), "artifact_references": len(references),
                          "physical_arrays": len(physical), "array_statuses": dict(Counter(entry["status"] for entry in physical.values())),
                          "correctness_pair_statuses": dict(Counter(pair["status"] for pair in pairs)),
                          "output_pair_statuses": dict(Counter(pair["status"] for pair in pairs if pair["array_kind"] == "output")),
                          "latent_pair_statuses": dict(Counter(pair["status"] for pair in latent_pairs)),
                          "all_historical_latents_independently_verified": bool(latent_pairs) and all(pair["status"] == "passed" for pair in latent_pairs),
                          "issues": len(issues)},
              "sources": sources, "arrays": list(physical.values()), "correctness_pairs": pairs, "issues": issues,
              "limitations": ["Missing saved latent/reference arrays are unavailable, including early historical rounds; recorded exact-match flags do not replace arrays.",
                              "Only explicitly associated pairs within the same records.jsonl are inferred; no baseline is borrowed across attempts.",
                              "Raw artifacts are never deleted or rewritten; integrity does not establish measured performance, quality, or GPU-memory release."]}
    atomic_json(root / "verification.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_root", type=Path, help="completed suite directory, e.g. results/previs/run01")
    args = parser.parse_args(argv)
    report = verify_suite(args.suite_root)
    print(json.dumps({"status": report["status"], **report["summary"],
                      "report": str(Path(report["suite_root"]) / "verification.json")}, ensure_ascii=False))
    return 1 if report["status"] == "failed" else 2 if report["status"] == "unavailable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
