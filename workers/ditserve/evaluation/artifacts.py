"""Hash-bound paired quality evidence, independent of execution control."""
import shutil
from pathlib import Path
from .common import atomic_json, canonical_hash, file_hash, positive, read_json, seal, verify_seal

IDENTITY_FIELDS = ("gpu", "precision", "text_encoder_device", "scheduler", "model_id", "revision", "torch_version", "cuda_runtime_version", "driver_version")

def generation_identity(generation, include_cache=True):
    """Generation inputs, independent of request IDs or arrival scheduling."""
    required = ("model", "revision", "prompt", "height", "width", "frames", "steps", "seed", "dtype")
    if not isinstance(generation, dict) or any(generation.get(key) is None for key in required):
        return None
    fields = ("model", "revision", "prompt", "negative_prompt", "height", "width", "frames", "steps",
              "seed", "guidance_scale", "dtype", "fps", "image_path", "image_sha256")
    if include_cache:
        fields += ("cache_enabled", "cache_threshold", "max_continuous", "approximate")
    return {key: generation.get(key) for key in fields}

def _run_path(path):
    path = Path(path).resolve()
    if path.is_dir():
        primary = path / "run.json"
        return primary if primary.is_file() else path / "evidence" / "run.json"
    return path

def resolve_artifact_path(value, digest, base):
    """Recover missing original paths from verified, content-addressed evidence.

    An existing but changed original is an integrity error, not a reason to
    silently substitute a historical copy. The sealed run remains untouched.
    """
    if not value or not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise LookupError("missing artifact path or valid sha256")
    path = Path(value)
    base = Path(base).resolve()
    if not path.is_absolute():
        path = base / path
    if path.is_file():
        if file_hash(path) != digest:
            raise ValueError("artifact hash mismatch: " + str(path))
        return path
    candidates = [base / "artifacts", base / "evidence" / "artifacts",
                  base / "report" / "evidence" / "artifacts"]
    for directory in candidates:
        copy = directory / (digest + path.suffix)
        if copy.is_file():
            if file_hash(copy) != digest:
                raise ValueError("evidence artifact hash mismatch: " + str(copy))
            return copy
    raise LookupError("missing artifact and hash-verified evidence: " + str(path))

def _write_overlay(path, value):
    """Keep old evaluator evidence when an evaluation is rerun."""
    path = Path(path)
    if path.is_file():
        previous = read_json(path)
        if previous != value:
            history = path.parent / "evaluation_history" / (path.stem + "-" + file_hash(path) + path.suffix)
            history.parent.mkdir(parents=True, exist_ok=True)
            if not history.exists():
                shutil.copy2(str(path), str(history))
    atomic_json(path, value)

def _checked_array(artifact, kind, base):
    """Read only hash-bound non-pickle arrays; don't silently trust filenames."""
    import numpy as np

    value = artifact.get(kind + "_path")
    digest = artifact.get(kind + "_sha256")
    if not value or not digest:
        raise LookupError("missing %s path or sha256" % kind)
    path = resolve_artifact_path(value, digest, base)
    array = np.load(str(path), allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.dtype.kind not in "fiu" or array.size == 0:
        if hasattr(array, "close"):
            array.close()
        raise ValueError("artifact must be a nonempty numeric NPY array")
    if not np.isfinite(array).all():
        raise ValueError("nonfinite %s artifact" % kind)
    return array

def freeze_outputs(*, backend, measured, records=None, run_path=None, output_path=None,
                   base_dir=".", execution_identity=None, phase=None, cell=None, observation_id=None):
    """Freeze actual worker output records for paired or official evaluation.

    Explicit records and C++ run manifests use the same scientific record form.
    Missing generation inputs are never reconstructed from model defaults.
    Hashing is outside the model's timed operation; this function never runs it.
    """
    import numpy as np

    if not isinstance(backend, str) or not backend or type(measured) is not bool:
        raise ValueError("outputs require an explicit backend and boolean measured")
    if measured and backend in ("mock", "core", "cpu-test"):
        raise ValueError("simulated outputs cannot be marked measured")
    if (records is None) == (run_path is None):
        raise ValueError("provide exactly one of records or run_path")
    source = None
    if run_path is not None:
        path = _run_path(run_path)
        source_hash = file_hash(path)
        original = read_json(path)
        if file_hash(path) != source_hash:
            raise ValueError("run manifest changed while reading")
        if original.get("schema") != "ditserve.run.v1":
            raise ValueError("outputs extraction requires a unified run manifest")
        records = original.get("requests", [])
        records = [row for row in records if (phase is None or row.get("phase") == phase)
                   and (cell is None or row.get("cell") == cell)
                   and (observation_id is None or row.get("observation_id") == observation_id)]
        base_dir = path.parent
        source = {"path": str(path), "sha256": source_hash, "phase": phase, "cell": cell, "observation_id": observation_id}
    if not isinstance(records, list) or not records:
        raise ValueError("outputs require a nonempty record list")
    rows, references, seen = [], {}, set()
    for row in records:
        identity = str(row.get("request_id", row.get("id", "")))
        if not identity or identity in seen:
            raise ValueError("missing or duplicate output identity; select one observation_id")
        seen.add(identity)
        status = row.get("status")
        if status != "completed":
            rows.append({"request_id": identity, "status": status or "unknown"})
            continue
        outputs = row.get("outputs", [row])
        if not isinstance(outputs, list) or not outputs or not isinstance(outputs[-1], dict):
            raise ValueError("completed request has no output record")
        produced = outputs[-1]
        generation = produced.get("generation", row.get("generation"))
        if generation_identity(generation) is None:
            raise ValueError("worker did not record actual generation inputs: " + identity)
        if produced.get("backend", backend) != backend or produced.get("measured", measured) is not measured:
            raise ValueError("worker output provenance differs from explicit evidence identity")
        artifact = {"generation": generation}
        if produced.get("path_history") is not None:
            artifact["path_history"] = produced["path_history"]
        for kind in ("output", "latent"):
            value = produced.get(kind)
            if value is None and kind == "latent":
                continue
            if not isinstance(value, dict) or not value.get("path"):
                raise ValueError("missing persisted " + kind + " artifact")
            actual = Path(value["path"])
            if not actual.is_absolute():
                actual = Path(base_dir) / actual
            actual = actual.resolve()
            size, digest = actual.stat().st_size, file_hash(actual)
            if value.get("bytes", size) != size or value.get("sha256", digest) != digest:
                raise ValueError("worker artifact size or hash changed")
            bound = {kind + "_path": str(actual), kind + "_sha256": digest}
            array = _checked_array(bound, kind, base_dir)
            if value.get("shape", list(array.shape)) != list(array.shape) or value.get("dtype", str(array.dtype)) != str(array.dtype):
                raise ValueError("worker artifact shape or dtype changed")
            if value.get("array_sha256") is not None:
                import hashlib
                if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != value["array_sha256"]:
                    raise ValueError("worker array content hash changed")
            if file_hash(actual) != digest:
                raise ValueError("worker artifact changed while freezing")
            artifact.update(bound)
            artifact[kind + "_bytes"] = size
        rows.append({"request_id": identity, "status": status, "generation": generation, "artifact": artifact})
        references[identity] = artifact
    result = seal({"schema_version": "ditserve.evaluation.outputs.v1", "backend": backend,
                   "measured": measured, "status": "completed" if all(row["status"] == "completed" for row in rows) else "incomplete",
                   "execution_identity": execution_identity or {}, "source": source,
                   "requests": rows, "references": references})
    if output_path is not None:
        target = Path(output_path)
        if target.exists() and read_json(target) != result:
            raise ValueError("cannot overwrite an immutable output evidence collection")
        atomic_json(target, result)
    return result

def _cache_check(candidate, reference, generation):
    left, right = candidate.get("path_history"), reference.get("path_history")
    if left is not None and right is not None:
        if not isinstance(left, list) or not isinstance(right, list) or not left or not right:
            return {"status": "unknown", "reason": "empty_or_invalid_branch_history"}
        return {"status": "pass" if left == right else "fail", "scope": "branch_path_parity_not_full_cache_state",
                "candidate_steps": len(left), "reference_steps": len(right)}
    if generation.get("cache_enabled") is False:
        return {"status": "not_applicable", "reason": "cache_explicitly_disabled"}
    return {"status": "unknown", "reason": "missing_cache_branch_history"}

def evaluate_artifacts(candidate, reference, generation=None, base_dir=".", reference_dir=None,
                       rtol=1e-5, atol=1e-6, with_lpips=False, epsilon=0.05, allow_uncached_reference=False):
    """Check final latent/video and branch trajectory against isolated reference."""
    positive(rtol, "rtol", allow_zero=True)
    positive(atol, "atol", allow_zero=True)
    positive(epsilon, "epsilon", allow_zero=True)
    result = {"status": "unknown", "qualified": False, "rtol": rtol, "atol": atol,
              "epsilon": epsilon, "scope": "reference_parity_not_absolute_video_quality"}
    if not isinstance(candidate, dict) or not isinstance(reference, dict):
        return dict(result, reason="missing_candidate_or_reference")
    generation = generation or candidate.get("generation")
    reference_generation = reference.get("generation")
    if generation_identity(generation) is None or generation_identity(reference_generation) is None:
        return dict(result, reason="missing_generation_identity")
    approximate = generation.get("cache_enabled") is True or generation.get("approximate") is True
    perceptual_scope = bool(with_lpips and approximate and allow_uncached_reference and reference_generation.get("cache_enabled") is False)
    if generation_identity(generation, not perceptual_scope) != generation_identity(reference_generation, not perceptual_scope):
        return dict(result, status="fail", reason="generation_identity_mismatch")
    result["generation_sha256"] = canonical_hash(generation_identity(generation))
    try:
        import numpy as np

        actual_latent = _checked_array(candidate, "latent", base_dir)
        expected_latent = _checked_array(reference, "latent", reference_dir or base_dir)
        actual_video = _checked_array(candidate, "output", base_dir)
        expected_video = _checked_array(reference, "output", reference_dir or base_dir)
        result["artifact_evidence"] = []
        for role, artifact, directory in (("candidate", candidate, base_dir), ("reference", reference, reference_dir or base_dir)):
            for kind in ("latent", "output"):
                source = resolve_artifact_path(artifact[kind + "_path"], artifact[kind + "_sha256"], directory)
                result["artifact_evidence"].append({"role": role, "kind": kind, "path": str(source.resolve()),
                                                    "sha256": artifact[kind + "_sha256"]})
        if actual_latent.shape != expected_latent.shape:
            raise ValueError("latent shape mismatch")
        if actual_video.shape != expected_video.shape:
            raise ValueError("video shape mismatch")
        if actual_video.ndim != 5 or actual_video.shape[0] != 1 or actual_video.shape[-1] != 3:
            raise ValueError("video must have shape [1,T,H,W,3]")
        if actual_video.dtype.kind != "f" or expected_video.dtype.kind != "f":
            raise ValueError("video must be floating point")
        if min(float(actual_video.min()), float(expected_video.min())) < 0 or max(float(actual_video.max()), float(expected_video.max())) > 1:
            raise ValueError("video outside [0,1]")
        latent_pass = bool(np.allclose(actual_latent, expected_latent, rtol=rtol, atol=atol))
        result["latent"] = {"status": "pass" if latent_pass else "fail", "shape": list(actual_latent.shape),
                            "max_abs_error": float(np.max(np.abs(actual_latent.astype(np.float64) - expected_latent.astype(np.float64))))}
        video_pass = bool(np.allclose(actual_video, expected_video, rtol=rtol, atol=atol))
        result["video"] = {"status": "pass" if video_pass else "fail", "finite": True, "range_valid": True,
                           "shape": list(actual_video.shape), "sha256": candidate["output_sha256"]}
        result["cache"] = _cache_check(candidate, reference, generation)
        cache_pass = result["cache"]["status"] in ("pass", "not_applicable")
        if with_lpips and approximate and not perceptual_scope:
            return dict(result, reason="approximate_lpips_requires_explicit_uncached_reference")
        if perceptual_scope:
            from .quality import evaluate_quality, quality_pass

            perceptual = evaluate_quality(expected_video, actual_video, with_lpips=True)
            result["perceptual"] = perceptual
            result["scope"] = "reference_relative_lpips_and_finite_output_not_absolute_video_quality"
            if perceptual.get("status") != "ok":
                return dict(result, reason="lpips_unavailable_or_invalid")
            perceptual_pass = quality_pass(perceptual, epsilon)
            result["status"] = "pass" if perceptual_pass else "fail"
            result["qualified"] = perceptual_pass
            return result
        if result["cache"]["status"] == "unknown":
            return dict(result, reason="cache_parity_unverified")
        result["status"] = "pass" if latent_pass and video_pass and cache_pass else "fail"
        result["qualified"] = result["status"] == "pass"
    except (ImportError, LookupError, OSError) as error:
        result.update(status="unknown", reason=str(error))
    except (ValueError, TypeError, KeyError) as error:
        result.update(status="fail", reason=str(error))
    return result

def evaluate_run(run_path, reference_run_path=None, references=None, with_lpips=False,
                 epsilon=0.05, output_path=None, rtol=1e-5, atol=1e-6):
    """Write evaluation.json bound to the immutable run and artifact hashes."""
    path = _run_path(run_path)
    run = verify_seal(read_json(path))
    if run.get("schema_version") not in ("ditserve.benchmark.run.v1", "ditserve.evaluation.outputs.v1"):
        raise ValueError("unsupported run schema")
    reference_dir = path.parent
    reference_source = None
    if reference_run_path is not None:
        reference_path = _run_path(reference_run_path)
        reference_run = verify_seal(read_json(reference_path))
        if run.get("schema_version") == "ditserve.evaluation.outputs.v1" and reference_run["sha256"] == run["sha256"]:
            raise ValueError("output collection cannot be its own independent reference")
        if reference_run.get("trace_sha256") != run.get("trace_sha256") and not with_lpips:
            raise ValueError("reference trace hash differs")
        references = reference_run.get("references", {})
        reference_dir = reference_path.parent
        reference_source = {"path": str(reference_path), "sha256": file_hash(reference_path)}
    if references is None:
        # A frozen output collection can serve as another run's baseline. Its
        # own outputs cannot qualify themselves as an independent reference.
        references = {} if run.get("schema_version") == "ditserve.evaluation.outputs.v1" else run.get("references", {})
    overlay = {"schema_version": "ditserve.benchmark.evaluation.v1", "run_sha256": run["sha256"],
               "run_file_sha256": file_hash(path), "trace_sha256": run.get("trace_sha256"),
               "reference_source": reference_source, "with_lpips": bool(with_lpips),
               "epsilon": epsilon, "policies": {}, "vbench": {"status": "not_run", "reason": "separate_official_protocol_required"}}
    policies = run.get("policies", {"outputs": {"requests": run.get("requests", [])}})
    for name, policy in policies.items():
        quality = {}
        for request in policy.get("requests", []):
            identity = str(request.get("request_id", request.get("id", "")))
            if not identity or identity in quality:
                raise ValueError("missing or duplicate request identity")
            if request.get("status") != "completed":
                quality[identity] = {"status": "unknown", "qualified": False, "reason": "request_not_completed"}
                continue
            quality[identity] = evaluate_artifacts(request.get("artifact"), references.get(identity),
                                                  generation=request.get("generation"), base_dir=path.parent,
                                                  reference_dir=reference_dir, rtol=rtol, atol=atol,
                                                  with_lpips=with_lpips, epsilon=epsilon,
                                                  allow_uncached_reference=reference_run_path is not None)
        overlay["policies"][name] = {"requests": quality}
    overlay = seal(overlay)
    target = Path(output_path) if output_path else path.parent / "evaluation.json"
    if target.resolve() == path or (reference_source and target.resolve() == Path(reference_source["path"])):
        raise ValueError("evaluation cannot overwrite sealed run")
    _write_overlay(target, overlay)
    return overlay

def check_backend_quality(references, reference_run_path, output_dir, epsilon=.05,
                          candidate_identity=None):
    import numpy as np
    from .quality import evaluate_quality

    positive(epsilon, "epsilon", allow_zero=True)
    path = _run_path(reference_run_path)
    baseline = verify_seal(read_json(path))
    if (baseline.get("backend") != "wan" or baseline.get("measured") is not True
            or baseline.get("status") != "completed"):
        raise ValueError("cross-attention gate requires a completed measured Wan reference run")
    identity = baseline.get("execution_identity", {})
    if identity.get("attention", {}).get("sdpa_backend") != "MATH":
        raise ValueError("cross-attention reference must use the frozen MATH backend")
    if not candidate_identity or any(identity.get(key) is None or candidate_identity.get(key) != identity.get(key) for key in IDENTITY_FIELDS):
        raise ValueError("cross-attention reference must match the candidate GPU and numerical recipe")
    rows = {}
    for rid, candidate in references.items():
        reference = baseline.get("references", {}).get(rid)
        if reference is None or generation_identity(candidate["generation"]) != generation_identity(reference["generation"]):
            raise ValueError("cross-attention reference request mismatch: " + rid)
        actual = resolve_artifact_path(candidate["output_path"], candidate["output_sha256"], Path(output_dir))
        expected = resolve_artifact_path(reference["output_path"], reference["output_sha256"], path.parent)
        metrics = evaluate_quality(np.load(expected, allow_pickle=False), np.load(actual, allow_pickle=False),
                                   with_lpips=True, lpips_device="cpu")
        lpips = metrics.get("lpips_mean")
        passed = (metrics.get("status") == "ok" and metrics.get("lpips_status") == "ok"
                  and metrics.get("finite") is True and metrics.get("range_valid") is True
                  and isinstance(lpips, (int, float)) and lpips <= epsilon)
        rows[rid] = dict(status="passed" if passed else "failed", metrics=metrics,
            candidate_sha256=candidate["output_sha256"], reference_sha256=reference["output_sha256"])
    return seal(dict(schema_version="ditserve.benchmark.backend_quality.v1",
        status="passed" if rows and all(row["status"] == "passed" for row in rows.values()) else "failed",
        reference_run_path=str(path), reference_run_sha256=baseline["sha256"],
        reference_run_file_sha256=file_hash(path), epsilon=epsilon, requests=rows,
        scope="held-out same-input cross-attention mean LPIPS; not absolute video quality or VBench"))
