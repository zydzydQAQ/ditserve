"""Pinned official VBench generation, video encoding and local evaluation."""
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from .common import atomic_json, canonical_hash, file_hash, positive, read_json, seal, verify_seal
from .artifacts import generation_identity, _checked_array, _run_path, _write_overlay

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


def export_vbench_manifest(full_info_path, output_path=None, dimensions=None, generation_settings=None):
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
    if output_path is not None:
        target = Path(output_path)
        if target.resolve() == source or (target.exists() and read_json(target) != result):
            raise ValueError("cannot overwrite immutable metadata or generation protocol")
        atomic_json(target, result)
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
    if run.get("backend") in ("cpu-test", "core", "mock") or run.get("measured") is not True:
        raise ValueError("official VBench video export requires measured model outputs, not a simulated backend")
    if manifest.get("schema_version") != "ditserve.benchmark.vbench_generation.v1":
        raise ValueError("invalid VBench generation manifest")
    policies = run.get("policies", {"fcfs": {"requests": run.get("requests", [])}})
    if policy not in policies:
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
    records = policies[policy].get("requests", [])
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
