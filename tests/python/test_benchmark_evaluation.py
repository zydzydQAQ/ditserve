"""Real arrays and failed evidence never silently turn into quality passes."""

import copy
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from ditserve.benchmark.common import atomic_json, file_hash, read_json, seal
from ditserve.benchmark import evaluation as ev

try:
    import numpy as np
except ImportError:
    np = None


def generation(**updates):
    result = dict(model="fixture", revision="one", prompt="a bird", negative_prompt="", height=2, width=2,
                  frames=2, steps=2, seed=0, guidance_scale=1., cache_enabled=False,
                  cache_threshold=.08, max_continuous=1)
    result.update(updates)
    return result


@unittest.skipIf(np is None, "numpy optional")
class ArtifactEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def artifact(self, stem, latent=None, video=None, gen=None):
        latent = np.zeros((2, 2), dtype=np.float32) if latent is None else latent
        video = np.full((1, 2, 2, 2, 3), .5, dtype=np.float32) if video is None else video
        result = {"generation": gen or generation(), "path_history": ["compute", "compute"]}
        for kind, value in (("latent", latent), ("output", video)):
            path = self.root / (stem + "_" + kind + ".npy")
            np.save(str(path), value)
            result[kind + "_path"] = str(path)
            result[kind + "_sha256"] = file_hash(path)
        return result

    def test_exact_arrays_pass_but_tampered_video_does_not(self):
        left, right = self.artifact("candidate"), self.artifact("reference")
        self.assertTrue(ev.evaluate_artifacts(left, right)["qualified"])
        np.save(left["output_path"], np.zeros((1, 2, 2, 2, 3), np.float32))
        result = ev.evaluate_artifacts(left, right)
        self.assertEqual(result["status"], "fail")
        self.assertIn("hash mismatch", result["reason"])

    def test_latent_match_does_not_hide_decoded_video_mismatch(self):
        left = self.artifact("candidate", video=np.zeros((1, 2, 2, 2, 3), np.float32))
        right = self.artifact("reference")
        result = ev.evaluate_artifacts(left, right)
        self.assertEqual(result["latent"]["status"], "pass")
        self.assertEqual(result["video"]["status"], "fail")
        self.assertFalse(result["qualified"])

    def test_missing_reference_and_unhashed_output_are_unknown(self):
        left, right = self.artifact("candidate"), self.artifact("reference")
        self.assertEqual(ev.evaluate_artifacts(left, None)["status"], "unknown")
        del left["latent_sha256"]
        self.assertEqual(ev.evaluate_artifacts(left, right)["status"], "unknown")

    def test_generation_and_cache_branch_parity_are_checked(self):
        left, right = self.artifact("candidate"), self.artifact("reference")
        left["generation"]["arrival_ms"] = 100
        right["generation"]["arrival_ms"] = 0
        self.assertTrue(ev.evaluate_artifacts(left, right)["qualified"])
        left["path_history"] = ["compute", "reuse"]
        self.assertEqual(ev.evaluate_artifacts(left, right)["cache"]["status"], "fail")
        left["generation"]["seed"] = 1
        self.assertEqual(ev.evaluate_artifacts(left, right)["reason"], "generation_identity_mismatch")

    def test_nonfinite_latent_rejected_even_when_hash_is_correct(self):
        left = self.artifact("candidate", latent=np.full((2, 2), np.nan, np.float32))
        result = ev.evaluate_artifacts(left, self.artifact("reference"))
        self.assertEqual(result["status"], "fail")
        self.assertIn("nonfinite", result["reason"])

    def test_lpips_requires_explicit_uncached_reference(self):
        gen = generation(cache_enabled=True)
        left, right = self.artifact("candidate", gen=gen), self.artifact("reference", gen=gen)
        result = ev.evaluate_artifacts(left, right, with_lpips=True)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("explicit_uncached", result["reason"])
        reference = self.artifact("uncached")
        with mock.patch("ditserve.gpu.quality._lpips_frames", side_effect=ImportError("no local model")):
            result = ev.evaluate_artifacts(left, reference, with_lpips=True, allow_uncached_reference=True)
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["qualified"])

    def test_run_remains_immutable_and_overlay_binds_hash(self):
        candidate, reference = self.artifact("candidate"), self.artifact("reference")
        run = seal({"schema_version": "ditserve.benchmark.run.v1", "trace_sha256": "trace", "references": {"r": reference},
                    "policies": {"fcfs": {"requests": [{"request_id": "r", "status": "completed", "artifact": candidate,
                                                           "generation": generation()}]}}})
        path = self.root / "run.json"
        atomic_json(path, run)
        digest = file_hash(path)
        result = ev.evaluate_run(path)
        self.assertEqual(file_hash(path), digest)
        self.assertEqual(result["run_file_sha256"], digest)
        self.assertTrue(result["policies"]["fcfs"]["requests"]["r"]["qualified"])
        self.assertEqual(read_json(self.root / "evaluation.json"), result)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg optional")
    def test_actual_video_export_obeys_seeds_and_preserves_verified_outputs(self):
        info = self.root / "official.json"
        plan = self.root / "generation.json"
        atomic_json(info, [{"prompt_en": "a bird", "dimension": ["subject_consistency"]}])
        ev.export_vbench_manifest(info, plan, dimensions=["subject_consistency"])
        rows = []
        for seed in range(5):
            gen = generation(seed=seed)
            rows.append({"request_id": str(seed), "status": "completed", "generation": gen,
                         "artifact": self.artifact("video-%d" % seed, gen=gen)})
        run = seal({"schema_version": "ditserve.benchmark.run.v1", "backend": "wan", "measured": True,
                    "policies": {"fcfs": {"requests": rows}}})
        atomic_json(self.root / "run.json", run)
        result = ev.export_vbench_videos(self.root / "run.json", plan, self.root / "videos")
        self.assertEqual(len(result["videos"]), 5)
        for row in result["videos"]:
            self.assertEqual(file_hash(self.root / "videos" / row["video_path"]), row["sha256"])
        self.assertEqual(ev.export_vbench_videos(self.root / "run.json", plan, self.root / "videos"), result)
        target = self.root / "videos" / result["videos"][0]["video_path"]
        target.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            ev.export_vbench_videos(self.root / "run.json", plan, self.root / "videos")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg optional")
    def test_video_export_rejects_simulation_and_generation_shape_mismatch(self):
        info, plan = self.root / "official.json", self.root / "generation.json"
        atomic_json(info, [{"prompt_en": "a bird", "dimension": ["subject_consistency"]}])
        ev.export_vbench_manifest(info, plan, dimensions=["subject_consistency"])
        gen = generation(height=4)
        row = {"request_id": "r", "status": "completed", "generation": gen, "artifact": self.artifact("wrong-shape", gen=gen)}
        for backend in ("cpu-test", "core", "wan"):
            run = seal({"schema_version": "ditserve.benchmark.run.v1", "backend": backend, "measured": True,
                        "policies": {"fcfs": {"requests": [row]}}})
            atomic_json(self.root / "run.json", run)
            with self.subTest(backend=backend), self.assertRaises(ValueError) as failure:
                ev.export_vbench_videos(self.root / "run.json", plan, self.root / "videos")
            self.assertIn("simulated" if backend != "wan" else "frames/height/width", str(failure.exception))

    def test_missing_original_arrays_recover_from_relocated_report_evidence(self):
        from ditserve.benchmark.reporting import write_report

        candidate, reference = self.artifact("candidate"), self.artifact("reference")
        row = {"request_id": "r", "status": "completed", "artifact": candidate, "generation": generation(),
               "shape_id": "small", "arrival_ms": 0, "submit_ms": 0, "start_ms": 0, "finish_ms": 1, "deadline_ms": 2}
        trace = seal({"schema_version": "ditserve.benchmark.trace.v1", "requests": [row]})
        run = seal({"schema_version": "ditserve.benchmark.run.v1", "trace_sha256": trace["sha256"], "references": {"r": reference},
                    "status": "completed", "policies": {"fcfs": {"status": "completed", "horizon_ms": 2, "requests": [row]}}})
        atomic_json(self.root / "run.json", run)
        atomic_json(self.root / "trace.json", trace)
        ev.evaluate_run(self.root)
        write_report(self.root)
        original_run_hash = file_hash(self.root / "run.json")
        for artifact in (candidate, reference):
            for kind in ("latent", "output"):
                Path(artifact[kind + "_path"]).unlink()
        self.assertTrue(ev.evaluate_run(self.root)["policies"]["fcfs"]["requests"]["r"]["qualified"])
        relocated = self.root / "relocated"
        shutil.copytree(str(self.root / "report"), str(relocated))
        shutil.rmtree(str(self.root / "report"))
        result = ev.evaluate_run(relocated)
        self.assertTrue(result["policies"]["fcfs"]["requests"]["r"]["qualified"])
        report = write_report(relocated, output_dir=self.root / "rebuilt")
        self.assertEqual(report["summary"]["policies"]["fcfs"]["quality_qualified"], 1)
        self.assertEqual(file_hash(relocated / "evidence" / "run.json"), original_run_hash)
        self.assertFalse(report["summary"]["evidence"]["missing"])


class VBenchProtocolTests(unittest.TestCase):
    def test_official_dimensions_share_one_monotonic_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            (repo / "vbench").mkdir(parents=True)
            (repo / "evaluate.py").write_text("# orchestration test fixture, not an evaluator\n")
            info = repo / "vbench" / "VBench_full_info.json"
            dimensions = ["subject_consistency", "background_consistency"]
            atomic_json(info, [{"prompt_en": "a bird", "dimension": dimensions}])
            plan = root / "generation.json"
            manifest = ev.export_vbench_manifest(info, plan, dimensions=dimensions)
            videos = root / "videos"
            for row in manifest["requests"]:
                path = videos / row["video_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test-only-video-marker")
            elapsed, timeouts = [0.], []

            def invoke(command, **kwargs):
                self.assertNotIn("shell", kwargs)
                if command[0] == "git":
                    return SimpleNamespace(stdout=ev.VBENCH_REVISION if "rev-parse" in command else "", returncode=0)
                timeouts.append(kwargs["timeout"])
                if kwargs["timeout"] < 6:
                    elapsed[0] += kwargs["timeout"]
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                elapsed[0] += 6
                dimension = command[command.index("--dimension") + 1]
                output = Path(command[command.index("--output_path") + 1])
                paths = [str((videos / row["video_path"]).resolve()) for row in manifest["requests"] if row["dimension"] == dimension]
                atomic_json(output / "fixture_full_info.json", [{"video_list": paths}])
                atomic_json(output / "fixture_eval_results.json", {dimension: [0., [{"fixture_only": True} for _ in paths]]})
                return SimpleNamespace(returncode=0)

            with mock.patch.object(ev.time, "monotonic", side_effect=lambda: elapsed[0]), mock.patch.object(ev.subprocess, "run", side_effect=invoke):
                result = ev.run_vbench(plan, videos, repo, timeout=10)
            self.assertEqual(timeouts, [10., 4.])
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["verified"])
            self.assertEqual(list(result["dimensions"]), ["subject_consistency"])

    def test_manifest_separates_seed_protocol_and_never_fakes_total(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root / "official.json", [{"prompt_en": "a bird", "dimension": ["subject_consistency", "temporal_flickering"]}])
            result = ev.export_vbench_manifest(root / "official.json", root / "generation.json",
                                               dimensions=["subject_consistency", "temporal_flickering"])
            self.assertEqual(len(result["requests"]), 30)
            self.assertEqual(len([r for r in result["requests"] if r["dimension"] == "temporal_flickering"]), 25)
            self.assertEqual(result["scope"], "dimension_subset")
            self.assertIsNone(result["total_score"])
            measured = ev.run_vbench(root / "generation.json", root / "videos", root / "missing-repo")
            self.assertEqual(measured["status"], "not_run")
            self.assertFalse(measured["verified"])
            self.assertIsNone(measured["total_score"])

    def test_manifest_rejects_unsafe_official_filenames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root / "bad.json", [{"prompt_en": "../../bad", "dimension": ["scene"]}])
            with self.assertRaises(ValueError):
                ev.export_vbench_manifest(root / "bad.json", root / "generation.json", dimensions=["scene"])


if __name__ == "__main__":
    unittest.main()
