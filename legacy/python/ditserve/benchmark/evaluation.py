"""Artifact-bound parity checks and an optional pinned official VBench adapter.

Evaluation is an overlay: a sealed run is never rewritten. Missing evidence is
unknown, including missing optional perceptual models. No proxy is a VBench score.
"""

import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .common import atomic_json, canonical_hash, file_hash, positive, read_json, seal, verify_seal


VBENCH_REPOSITORY = "https://github.com/Vchitect/VBench"
VBENCH_REVISION = "1039fe66d7aa2f4b214003ca2d35f5011c48bfee"
VBENCH_VERSION = "v0.1.5"
VBENCH_DIMENSIONS = (
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "object_class", "multiple_objects", "human_action", "color",
    "spatial_relationship", "scene", "temporal_style", "appearance_style",
    "overall_consistency",
)


def generation_identity(generation, include_cache=True):
    """Generation inputs, independent of request IDs or arrival scheduling."""
    if not isinstance(generation, dict) or not generation:
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
    if not isinstance(generation, dict) or not generation or not isinstance(reference_generation, dict):
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
            from ..gpu.quality import evaluate_quality, quality_pass

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
    if run.get("schema_version") != "ditserve.benchmark.run.v1":
        raise ValueError("unsupported run schema")
    reference_dir = path.parent
    reference_source = None
    if reference_run_path is not None:
        reference_path = _run_path(reference_run_path)
        reference_run = verify_seal(read_json(reference_path))
        if reference_run.get("trace_sha256") != run.get("trace_sha256") and not with_lpips:
            raise ValueError("reference trace hash differs")
        references = reference_run.get("references", {})
        reference_dir = reference_path.parent
        reference_source = {"path": str(reference_path), "sha256": file_hash(reference_path)}
    if references is None:
        references = run.get("references", {})
    overlay = {"schema_version": "ditserve.benchmark.evaluation.v1", "run_sha256": run["sha256"],
               "run_file_sha256": file_hash(path), "trace_sha256": run.get("trace_sha256"),
               "reference_source": reference_source, "with_lpips": bool(with_lpips),
               "epsilon": epsilon, "policies": {}, "vbench": {"status": "not_run", "reason": "separate_official_protocol_required"}}
    for name, policy in run.get("policies", {}).items():
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
    if target.resolve() == path:
        raise ValueError("evaluation cannot overwrite sealed run")
    _write_overlay(target, overlay)
    return overlay


def export_vbench_manifest(full_info_path, output_path, dimensions=None, generation_settings=None):
    """Export a separate generation protocol; no videos or weights are fetched.

    Five seeds are required for each prompt/dimension, 25 for flickering.
    The source file must be VBench_full_info.json from the pinned checkout.
    """
    source = Path(full_info_path).resolve()
    info = read_json(source)
    dimensions = list(dimensions or VBENCH_DIMENSIONS)
    if not dimensions or len(dimensions) != len(set(dimensions)) or set(dimensions) - set(VBENCH_DIMENSIONS):
        raise ValueError("invalid VBench dimensions")
    if not isinstance(info, list) or not info:
        raise ValueError("full_info must be a nonempty official metadata list")
    records, seen = [], set()
    for entry in info:
        prompt = entry.get("prompt_en")
        if not isinstance(prompt, str) or not prompt or any(char in prompt for char in ("/", "\\", "\0")):
            raise ValueError("unsafe or missing official prompt filename")
        for dimension in entry.get("dimension", []):
            if dimension not in dimensions:
                continue
            key = (dimension, prompt)
            if key in seen:
                raise ValueError("duplicate dimension prompt")
            seen.add(key)
            for seed in range(25 if dimension == "temporal_flickering" else 5):
                records.append({"dimension": dimension, "prompt": prompt, "seed": seed,
                                "video_path": "%s/%s-%d.mp4" % (dimension, prompt, seed)})
    if set(dimensions) != {row["dimension"] for row in records}:
        raise ValueError("official metadata does not cover requested dimensions")
    result = seal({"schema_version": "ditserve.benchmark.vbench_generation.v1", "repository": VBENCH_REPOSITORY,
                   "revision": VBENCH_REVISION, "version": VBENCH_VERSION, "full_info_path": str(source),
                   "full_info_sha256": file_hash(source), "dimensions": dimensions,
                   "scope": "full_dimension_suite" if set(dimensions) == set(VBENCH_DIMENSIONS) else "dimension_subset",
                   "generation_settings": generation_settings or {}, "seeds_per_prompt": 5,
                   "flickering_seeds_per_prompt": 25, "requests": records,
                   "total_score": None, "note": "generation protocol, not measured serving requests or a VBench score"})
    atomic_json(output_path, result)
    return result


def export_vbench_videos(run_path, manifest_path, videos_dir, policy="fcfs"):
    """Encode hash-checked raw outputs into protocol filenames without rerunning.

    Only completed, uniquely matching prompt/seed/config records are eligible.
    Existing video exports are preserved and must match their previous manifest.
    """
    import numpy as np

    path = _run_path(run_path)
    run = verify_seal(read_json(path))
    manifest = verify_seal(read_json(manifest_path))
    if run.get("backend") in ("cpu-test", "core") or run.get("measured") is not True:
        raise ValueError("official VBench video export requires measured model outputs, not a simulated backend")
    if manifest.get("schema_version") != "ditserve.benchmark.vbench_generation.v1":
        raise ValueError("invalid VBench generation manifest")
    if policy not in run.get("policies", {}):
        raise ValueError("unknown policy")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for VBench video export")
    destination = Path(videos_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    export_path = destination / "export.json"
    previous = verify_seal(read_json(export_path)) if export_path.is_file() else None
    if previous and (previous.get("run_sha256") != run["sha256"] or previous.get("manifest_sha256") != manifest["sha256"] or previous.get("policy") != policy):
        raise ValueError("video directory already belongs to another export")
    records = run["policies"][policy].get("requests", [])
    exported = []
    settings = manifest.get("generation_settings", {})
    for expected in manifest["requests"]:
        matches = []
        for row in records:
            gen = row.get("generation") or {}
            if row.get("status") == "completed" and gen.get("prompt") == expected["prompt"] and gen.get("seed") == expected["seed"]:
                if all(gen.get(key) == value for key, value in settings.items() if key not in ("fps",)):
                    matches.append(row)
        if not matches:
            raise ValueError("missing completed protocol prompt/seed/config: " + expected["video_path"])
        identities = {canonical_hash(generation_identity(row["generation"])) for row in matches}
        outputs = {(row.get("artifact") or {}).get("output_sha256") for row in matches}
        if len(identities) != 1 or len(outputs) != 1 or None in outputs:
            raise ValueError("ambiguous generation outputs for protocol request")
        row = matches[0]
        frames = _checked_array(row["artifact"], "output", path.parent)
        if frames.ndim != 5 or frames.shape[0] != 1 or frames.shape[-1] != 3 or frames.min() < 0 or frames.max() > 1:
            raise ValueError("invalid output video shape or pixel range")
        expected_shape = tuple(row["generation"].get(key) for key in ("frames", "height", "width"))
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in expected_shape) or tuple(frames.shape[1:4]) != expected_shape:
            raise ValueError("actual video frames/height/width do not match generation settings")
        relative = Path(expected["video_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe output filename")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            old = next((item for item in (previous or {}).get("videos", []) if item["video_path"] == str(relative)), None)
            if not old or old["sha256"] != file_hash(target) or old["source_output_sha256"] != row["artifact"]["output_sha256"]:
                raise ValueError("existing output cannot be verified: " + str(target))
        else:
            fps = settings.get("fps", row["generation"].get("fps", 16))
            positive(fps, "fps")
            height, width = frames.shape[2:4]
            fd, temp_name = tempfile.mkstemp(prefix=".video-", suffix=".mp4", dir=str(target.parent))
            os.close(fd)
            command = [ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
                       "%dx%d" % (width, height), "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                       "-crf", "18", "-pix_fmt", "yuv444p", temp_name]
            try:
                with tempfile.TemporaryFile() as log:
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                    try:
                        for frame in frames[0]:
                            process.stdin.write(np.rint(frame * 255).astype(np.uint8).tobytes())
                        process.stdin.close()
                        code = process.wait(timeout=120)
                    except BaseException:
                        process.kill()
                        process.wait()
                        raise
                    if code:
                        log.seek(0)
                        raise RuntimeError("ffmpeg failed: " + log.read(4096).decode("utf-8", errors="replace"))
                os.replace(temp_name, str(target))
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
        exported.append({"video_path": str(relative), "sha256": file_hash(target),
                         "source_output_sha256": row["artifact"]["output_sha256"],
                         "generation_sha256": canonical_hash(generation_identity(row["generation"])),
                         "frames": int(frames.shape[1]), "height": int(frames.shape[2]), "width": int(frames.shape[3])})
    result = seal({"schema_version": "ditserve.benchmark.vbench_export.v1", "run_sha256": run["sha256"],
                   "manifest_sha256": manifest["sha256"], "policy": policy, "videos": exported,
                   "encoding": {"codec": "libx264", "crf": 18, "pixel_format": "yuv444p", "raw_range": "[0,1]"}})
    atomic_json(export_path, result)
    return result


def run_vbench(manifest_path, videos_dir, repo_dir, output_dir=None, python_executable=None, timeout=3600):
    """Invoke the pinned official CLI with local checkpoints, validate coverage.

    Flickering uses official custom_input mode to include all 25 videos; the
    pinned standard loader only enumerates five. Results are dimension results,
    never an invented subset mean or leaderboard total score.
    """
    positive(timeout, "timeout")
    deadline = time.monotonic() + timeout

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("VBench total evaluation deadline exhausted")
        return seconds

    manifest_path = Path(manifest_path).resolve()
    manifest = verify_seal(read_json(manifest_path))
    if manifest.get("schema_version") != "ditserve.benchmark.vbench_generation.v1" or manifest.get("revision") != VBENCH_REVISION:
        raise ValueError("unrecognized VBench manifest or revision")
    output = Path(output_dir).resolve() if output_dir else manifest_path.parent / "vbench_results"
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": "ditserve.benchmark.vbench_evaluation.v1", "status": "not_run", "verified": False,
              "manifest_sha256": manifest["sha256"], "repository": VBENCH_REPOSITORY, "revision": VBENCH_REVISION,
              "scope": manifest["scope"], "total_score": None, "dimensions": {}, "reason": None}
    repo, videos = Path(repo_dir).resolve(), Path(videos_dir).resolve()
    try:
        if not (repo / "evaluate.py").is_file():
            raise LookupError("pinned official VBench checkout unavailable")
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=remaining()).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], check=True, capture_output=True, text=True, timeout=remaining()).stdout.strip()
        if commit != VBENCH_REVISION or dirty:
            raise ValueError("VBench checkout is modified or not the pinned revision")
        official_info = repo / "vbench" / "VBench_full_info.json"
        if file_hash(official_info) != manifest["full_info_sha256"]:
            raise ValueError("metadata hash does not match pinned official metadata")
        # Reconstruct the expected protocol instead of trusting edited counts.
        with tempfile.TemporaryDirectory(prefix="ditserve-vbench-") as temporary:
            expected = export_vbench_manifest(official_info, Path(temporary) / "manifest.json", manifest["dimensions"], manifest["generation_settings"])
        if expected["requests"] != manifest["requests"]:
            raise ValueError("generation manifest protocol was altered")
        artifacts = {}
        for entry in manifest["requests"]:
            remaining()
            path = videos / entry["video_path"]
            if not path.is_file() or path.stat().st_size == 0:
                raise LookupError("required VBench video missing: " + str(path))
            artifacts[entry["video_path"]] = file_hash(path)
        result["video_sha256"] = artifacts
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        for dimension in manifest["dimensions"]:
            remaining()
            required = {str((videos / row["video_path"]).resolve()) for row in manifest["requests"] if row["dimension"] == dimension}
            present = {str(p.resolve()) for p in (videos / dimension).iterdir() if p.suffix.lower() in (".mp4", ".gif", ".png", ".jpg")}
            if required != present:
                raise ValueError("dimension video set differs from generation manifest")
            attempt = Path(tempfile.mkdtemp(prefix=dimension + "-", dir=str(output)))
            mode = "custom_input" if dimension == "temporal_flickering" else "vbench_standard"
            command = [python_executable or sys.executable, str(repo / "evaluate.py"), "--videos_path", str(videos / dimension),
                       "--dimension", dimension, "--output_path", str(attempt), "--full_json_dir", str(official_info),
                       "--mode", mode, "--load_ckpt_from_local", "True"]
            with (attempt / "stdout.log").open("w") as stdout, (attempt / "stderr.log").open("w") as stderr:
                process = subprocess.run(command, cwd=str(repo), env=env, stdout=stdout, stderr=stderr, timeout=remaining(), check=False)
            results = list(attempt.glob("*_eval_results.json"))
            metadata = list(attempt.glob("*_full_info.json"))
            if process.returncode or len(results) != 1 or len(metadata) != 1:
                raise RuntimeError("official evaluator failed or outputs incomplete: " + str(attempt))
            actual = {str(Path(value).resolve()) for row in read_json(metadata[0]) for value in row.get("video_list", [])}
            if actual != required:
                raise ValueError("official evaluator did not evaluate the required video set")
            raw = read_json(results[0])
            values = raw.get(dimension)
            if not isinstance(values, list) or len(values) != 2 or not isinstance(values[0], (int, float)) or not math.isfinite(values[0]):
                raise ValueError("invalid official dimension result")
            if not isinstance(values[1], list) or len(values[1]) != len(required):
                raise ValueError("official per-video result count mismatch")
            result["dimensions"][dimension] = {"score": values[0], "video_count": len(required), "mode": mode,
                                                "result_path": str(results[0]), "result_sha256": file_hash(results[0]),
                                                "metadata_sha256": file_hash(metadata[0]), "command": command}
        if any(file_hash(videos / name) != digest for name, digest in artifacts.items()):
            raise ValueError("video changed during evaluation")
        remaining()
        result.update(status="completed", verified=True, reason=None)
    except (LookupError, FileNotFoundError) as error:
        result["reason"] = str(error)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        result.update(status="failed", reason=str(error))
    result = seal(result)
    _write_overlay(output / "vbench_evaluation.json", result)
    return result
