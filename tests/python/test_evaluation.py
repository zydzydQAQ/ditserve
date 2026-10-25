"""Local CPU fixtures for frozen datasets, paired evidence and VBench coverage."""

import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve import evaluation as ev
from ditserve.evaluation import artifacts, datasets, vbench
from ditserve.evaluation.common import atomic_json, file_hash, seal, verify_seal

try:
    import numpy as np
except ImportError:
    np = None


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_import_does_not_load_models_or_numerical_packages(self):
        check = subprocess.run([sys.executable, "-c", "import sys; import ditserve.evaluation; "
            "assert not {'torch','numpy','lpips','diffusers'} & set(sys.modules)"],
            check=False, capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)

    def test_vbench_full_metadata_and_disjoint_service_splits(self):
        path = self.root / "official.json"
        rows = [{"prompt_en": "  cat\n", "dimension": ["scene"]}, {"prompt_en": "dog"},
                {"prompt_en": "fish"}, {"prompt_en": "  cat\n", "extra": "preserved"}]
        atomic_json(path, rows)
        request = dict(version=1, id=1, op="evaluate", evaluator="dataset", kind="vbench",
                       source=str(path), dev_count=1, seed=5, sample_count=1, rank=0, world_size=1, device="cpu")
        with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            result = ev.evaluate(request)
            from ditserve.runtime.worker import Worker
            worker = Worker("mock", {})
            self.assertEqual(worker.dispatch(request), result)
            self.assertIsNone(worker.backend)
        self.assertEqual(verify_seal(result), result)
        self.assertEqual(len(result["samples"]), 4)
        self.assertTrue(all(r["split"] == "evaluation" for r in result["samples"]))
        self.assertEqual(result["samples"][0]["prompt"], "  cat\n")
        self.assertEqual(result["samples"][0]["service_split"], result["samples"][3]["service_split"])
        self.assertEqual(result["samples"][3]["metadata"], rows[3])
        self.assertEqual(ev.evaluate(request), result)
        with self.assertRaisesRegex(ValueError, "source metadata"):
            ev.evaluate(dict(request, output_path=str(path)))
        self.assertEqual(json.loads(path.read_text()), rows)

    def test_vidprom_streaming_sample_is_bounded_and_preserves_timestamps(self):
        path = self.root / "prompts.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["uuid", "prompt", "timestamp", "video"])
            writer.writeheader()
            writer.writerows(dict(uuid=i, prompt="scene,%d\n" % i, timestamp="2024-01-01", video="never-read.mp4") for i in range(50))
        with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            first = ev.prepare_dataset("vidprom", source=path, dev_count=2, seed=7, sample_count=8)
            second = ev.prepare_dataset("vidprom", source=path, dev_count=2, seed=7, sample_count=8)
        self.assertEqual(first, second)
        self.assertEqual(first["selection"]["rows_seen"], 50)
        self.assertEqual(len(first["samples"]), 8)
        self.assertEqual(sum(s["split"] == "dev" for s in first["samples"]), 2)
        self.assertTrue(all(s["timestamp"] == "2024-01-01" and "video" not in s for s in first["samples"]))

    def test_vidprom_skips_empty_and_nsfw_prompts(self):
        path = self.root / "filtered.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["uuid", "prompt", "time", "toxicity"])
            writer.writeheader()
            writer.writerows([
                dict(uuid="a", prompt="safe a", time="2024-01-01", toxicity=0.1),
                dict(uuid="empty", prompt="  ", time="2024-01-01", toxicity=0.0),
                dict(uuid="b", prompt="safe b", time="2024-01-01", toxicity=0.2),
                dict(uuid="bad", prompt="unsafe", time="2024-01-01", toxicity=0.9),
                dict(uuid="c", prompt="safe c", time="2024-01-01", toxicity=0.3),
            ])
        result = ev.prepare_dataset("vidprom", source=path, dev_count=0, seed=7, sample_count=10)
        self.assertEqual(result["selection"]["rows_seen"], 5)
        self.assertEqual(result["selection"]["rows_skipped"], 2)
        self.assertEqual({sample["prompt"] for sample in result["samples"]}, {"safe a", "safe b", "safe c"})

    def test_download_requires_opt_in_and_validates_revision_hash_and_length(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaises(ValueError):
                ev.prepare_dataset("vbench")
            with self.assertRaises(ValueError):
                ev.prepare_dataset("vbench", download="false")
        blob = b"uuid,prompt\na,cat\nb,dog\n"
        metadata = [{"type": "file", "path": "VidProM_unique.csv", "size": len(blob),
                     "lfs": {"oid": hashlib.sha256(blob).hexdigest(), "size": len(blob)}}]
        response = io.BytesIO(blob)
        response.headers = {"Content-Length": str(len(blob))}
        with patch.object(datasets, "_json_url", side_effect=[{"sha": "a" * 40}, metadata]), \
                patch("urllib.request.urlopen", return_value=response) as network:
            result = ev.prepare_dataset("vidprom", download=True, cache_dir=self.root, dev_count=1)
        self.assertIn("/resolve/" + "a" * 40, network.call_args[0][0].full_url)
        self.assertEqual(result["source"]["integrity"]["status"], "verified")
        # A fresh truncated fixture is rejected before any CSV sampling.
        response = io.BytesIO(blob[:-4])
        response.headers = {"Content-Length": str(len(blob))}
        with patch.object(datasets, "_json_url", return_value=metadata), \
                patch("urllib.request.urlopen", return_value=response), self.assertRaises(ValueError):
            ev.prepare_dataset("vidprom", revision="b" * 40, download=True, cache_dir=self.root, dev_count=1)
        self.assertTrue(list(self.root.rglob("*.partial.error.json")))

    def manifest(self, dimensions):
        repo = self.root / "repo"
        (repo / "vbench").mkdir(parents=True, exist_ok=True)
        (repo / "evaluate.py").write_text("# test fixture, not an evaluator\n")
        info = repo / "vbench/VBench_full_info.json"
        atomic_json(info, [{"prompt_en": "a bird", "dimension": dimensions}])
        path = self.root / "generation.json"
        return repo, path, ev.export_vbench_manifest(info, path, dimensions)

    def test_official_manifest_retains_seed_protocol_and_missing_evaluator_is_unverified(self):
        repo, path, manifest = self.manifest(["scene", "temporal_flickering"])
        self.assertEqual(len(manifest["requests"]), 30)
        self.assertEqual(sum(row["dimension"] == "temporal_flickering" for row in manifest["requests"]), 25)
        self.assertIsNone(manifest["total_score"])
        result = ev.run_vbench(path, self.root / "videos", self.root / "missing")
        self.assertEqual(result["status"], "not_run")
        self.assertFalse(result["verified"])
        self.assertIsNone(result["total_score"])
        with self.assertRaisesRegex(ValueError, "immutable metadata"):
            ev.export_vbench_manifest(repo / "vbench/VBench_full_info.json", repo / "vbench/VBench_full_info.json", ["scene"])
        atomic_json(repo / "vbench/VBench_full_info.json", [{"prompt_en": "../bad", "dimension": ["scene"]}])
        with self.assertRaises(ValueError):
            ev.export_vbench_manifest(repo / "vbench/VBench_full_info.json", dimensions=["scene"])

    def test_official_evaluation_checks_coverage_and_uses_one_total_deadline(self):
        repo, path, manifest = self.manifest(["scene", "color"])
        videos = self.root / "videos"
        for row in manifest["requests"]:
            video = videos / row["video_path"]
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"test-only marker")
        elapsed, timeouts = [0.], []
        def invoke(command, **kwargs):
            if command[0] == "git":
                return SimpleNamespace(stdout=vbench.VBENCH_REVISION if "rev-parse" in command else "", returncode=0)
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            timeouts.append(kwargs["timeout"])
            if kwargs["timeout"] < 6:
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            elapsed[0] += 6
            dimension = command[command.index("--dimension") + 1]
            output = Path(command[command.index("--output_path") + 1])
            required = [str((videos / r["video_path"]).resolve()) for r in manifest["requests"] if r["dimension"] == dimension]
            atomic_json(output / "fixture_full_info.json", [{"video_list": required}])
            atomic_json(output / "fixture_eval_results.json", {dimension: [0.25, [{} for p in required]]})
            return SimpleNamespace(returncode=0)
        with patch.object(vbench.shutil, "which", return_value="/usr/bin/git"), \
                patch.object(vbench.time, "monotonic", side_effect=lambda: elapsed[0]), \
                patch.object(vbench.subprocess, "run", side_effect=invoke):
            result = ev.run_vbench(path, videos, repo, timeout=10)
        self.assertEqual(timeouts, [10., 4.])
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["verified"])
        self.assertEqual(list(result["dimensions"]), ["scene"])

    def test_vbench_metadata_relocates_to_pinned_checkout(self):
        repo, path, manifest = self.manifest(["scene"])
        videos = self.root / "videos"
        for row in manifest["requests"]:
            video = videos / row["video_path"]
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"test-only marker")
        manifest["full_info_path"] = str(self.root / "missing" / "VBench_full_info.json")
        def invoke(command, **kwargs):
            if command[0] == "git":
                return SimpleNamespace(stdout=vbench.VBENCH_REVISION if "rev-parse" in command else "", returncode=0)
            dimension = command[command.index("--dimension") + 1]
            output = Path(command[command.index("--output_path") + 1])
            required = [str((videos / row["video_path"]).resolve()) for row in manifest["requests"]]
            atomic_json(output / "fixture_full_info.json", [{"video_list": required}])
            atomic_json(output / "fixture_eval_results.json", {dimension: [0.25, [{} for p in required]]})
            return SimpleNamespace(returncode=0)
        with patch.object(vbench.shutil, "which", return_value="/usr/bin/git"), \
                patch.object(vbench.subprocess, "run", side_effect=invoke):
            result = ev.run_vbench(path, videos, repo)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["verified"])

    def artifact(self, stem, *, value=.5, cache=False, seed=0):
        generation = dict(model="fixture", revision="one", prompt="a bird", negative_prompt="", height=2,
            width=2, frames=2, steps=2, seed=seed, guidance_scale=1., cache_enabled=cache, cache_threshold=.08,
            dtype="float32", max_continuous=1)
        row = dict(generation=generation, path_history=["compute", "compute"])
        for kind, array in (("latent", np.zeros((2, 2), np.float32)), ("output", np.full((1, 2, 2, 2, 3), value, np.float32))):
            path = self.root / (stem + "-" + kind + ".npy")
            np.save(path, array)
            row[kind + "_path"], row[kind + "_sha256"] = str(path), file_hash(path)
        return row

    @unittest.skipIf(np is None, "numpy optional")
    def test_freeze_worker_outputs_binds_actual_files_without_rewriting_run(self):
        evidence = self.artifact("worker")
        produced = dict(generation=evidence["generation"], path_history=evidence["path_history"],
                        backend="wan", measured=True)
        for kind in ("output", "latent"):
            path = Path(evidence[kind + "_path"])
            array = np.load(path, allow_pickle=False)
            produced[kind] = dict(path=str(path), bytes=path.stat().st_size, shape=list(array.shape),
                                  dtype=str(array.dtype), array_sha256=hashlib.sha256(array.tobytes()).hexdigest())
        run = dict(schema="ditserve.run.v1", status="running", requests=[
            dict(request_id="r", phase="p", cell=0, status="completed", outputs=[produced]),
            dict(request_id="f", phase="p", cell=0, status="failed"),
            dict(request_id="r", phase="other", cell=0, status="completed", outputs=[produced])])
        path, target = self.root / "run.json", self.root / "frozen.json"
        atomic_json(path, run)
        before = path.read_bytes()
        result = ev.freeze_outputs(backend="wan", measured=True, run_path=path, phase="p", cell=0, output_path=target)
        self.assertEqual(verify_seal(result), result)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(result["requests"][0]["artifact"]["output_sha256"], evidence["output_sha256"])
        self.assertEqual(result["status"], "incomplete")
        overlay = ev.evaluate_run(target)
        self.assertFalse(overlay["policies"]["outputs"]["requests"]["r"]["qualified"])
        with self.assertRaisesRegex(ValueError, "own independent"):
            ev.evaluate_run(target, reference_run_path=target)
        reference = self.artifact("independent")
        overlay = ev.evaluate_run(target, references={"r": reference})
        self.assertTrue(overlay["policies"]["outputs"]["requests"]["r"]["qualified"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ev.freeze_outputs(backend="wan", measured=True, run_path=path)
        with self.assertRaisesRegex(ValueError, "immutable"):
            ev.freeze_outputs(backend="wan", measured=True, run_path=path, phase="other", output_path=target)
        produced["output"]["array_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "content hash"):
            ev.freeze_outputs(backend="wan", measured=True,
                records=[dict(request_id="r", status="completed", **produced)])
        with self.assertRaisesRegex(ValueError, "simulated"):
            ev.freeze_outputs(backend="mock", measured=True, run_path=path)

    @unittest.skipIf(np is None, "numpy optional")
    def test_pair_checks_decoded_video_identity_hash_and_missing_evidence(self):
        actual, reference = self.artifact("actual"), self.artifact("reference")
        self.assertTrue(ev.evaluate_artifacts(actual, reference)["qualified"])
        self.assertEqual(ev.evaluate_artifacts(actual, None)["status"], "unknown")
        actual["generation"]["seed"] = 1
        self.assertEqual(ev.evaluate_artifacts(actual, reference)["reason"], "generation_identity_mismatch")
        actual = self.artifact("actual", value=.4)
        result = ev.evaluate_artifacts(actual, reference)
        self.assertEqual(result["latent"]["status"], "pass")
        self.assertEqual(result["video"]["status"], "fail")
        Path(actual["output_path"]).write_bytes(b"tampered")
        self.assertIn("hash mismatch", ev.evaluate_artifacts(actual, reference)["reason"])

    @unittest.skipIf(np is None, "numpy optional")
    def test_approximate_lpips_missing_model_never_becomes_qualification(self):
        actual, reference = self.artifact("cache", cache=True), self.artifact("reference")
        with patch("ditserve.evaluation.quality._lpips_frames", side_effect=ImportError("no weights")):
            result = ev.evaluate_artifacts(actual, reference, with_lpips=True, allow_uncached_reference=True)
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["qualified"])
        self.assertIn("lpips_unavailable", result["reason"])

    @unittest.skipIf(np is None, "numpy optional")
    def test_backend_gate_requires_math_and_matching_device_and_binds_evidence(self):
        actual = self.artifact("output")
        identity = dict(gpu={"uuid": "GPU-test"}, precision={"dit": "float32"}, attention={"sdpa_backend": "MATH"},
                        text_encoder_device="cpu", scheduler="test", model_id="fixture", revision="one",
                        torch_version="fixture", cuda_runtime_version="fixture", driver_version="fixture")
        baseline = seal(dict(backend="wan", measured=True, status="completed", execution_identity=identity,
                             references={"r": actual}))
        atomic_json(self.root / "run.json", baseline)
        with self.assertRaisesRegex(ValueError, "numerical recipe"):
            ev.check_backend_quality({"r": actual}, self.root, self.root, candidate_identity=dict(identity, gpu="other"))
        metrics = dict(status="ok", lpips_status="ok", finite=True, range_valid=True, lpips_mean=.049)
        with patch("ditserve.evaluation.quality.evaluate_quality", return_value=metrics):
            result = ev.check_backend_quality({"r": actual}, self.root, self.root, candidate_identity=identity)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["requests"]["r"]["candidate_sha256"], actual["output_sha256"])
        baseline["execution_identity"]["attention"]["sdpa_backend"] = "FLASH"
        atomic_json(self.root / "run.json", seal(baseline))
        with self.assertRaisesRegex(ValueError, "MATH"):
            ev.check_backend_quality({"r": actual}, self.root, self.root, candidate_identity=identity)

    @unittest.skipIf(np is None, "numpy optional")
    def test_evaluation_overlay_preserves_input_and_incomplete_requests(self):
        candidate, reference = self.artifact("candidate"), self.artifact("reference")
        run = seal(dict(schema_version="ditserve.evaluation.outputs.v1", references={"r": reference},
            requests=[dict(request_id="r", status="completed", artifact=candidate, generation=candidate["generation"]),
                      dict(request_id="failed", status="failed")]))
        path = self.root / "run.json"
        atomic_json(path, run)
        before = path.read_bytes()
        result = ev.evaluate_run(path, references={"r": reference})
        self.assertTrue(result["policies"]["outputs"]["requests"]["r"]["qualified"])
        self.assertEqual(result["policies"]["outputs"]["requests"]["failed"]["status"], "unknown")
        self.assertEqual(path.read_bytes(), before)

    @unittest.skipIf(np is None or shutil.which("ffmpeg") is None, "numpy and ffmpeg optional")
    def test_vbench_video_export_uses_five_hash_bound_outputs(self):
        repo, plan, manifest = self.manifest(["scene"])
        rows = []
        for seed in range(5):
            artifact = self.artifact("seed-%d" % seed, seed=seed)
            rows.append(dict(status="completed", artifact=artifact, generation=artifact["generation"]))
        run = seal(dict(schema_version="ditserve.evaluation.outputs.v1", backend="wan", measured=True, requests=rows))
        path = self.root / "outputs.json"
        atomic_json(path, run)
        result = ev.export_vbench_videos(path, plan, self.root / "videos")
        self.assertEqual(len(result["videos"]), 5)
        self.assertEqual(ev.export_vbench_videos(path, plan, self.root / "videos"), result)
        target = self.root / "videos" / result["videos"][0]["video_path"]
        target.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            ev.export_vbench_videos(path, plan, self.root / "videos")


if __name__ == "__main__":
    unittest.main()
