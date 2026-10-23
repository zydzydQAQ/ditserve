"""CPU-only retention and exact-reference bridges; no model or CUDA load."""

import contextlib
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ditserve.evaluation import workflow_encoder as encoder
from ditserve.evaluation import workflow_embedding_replay as replay
from ditserve.evaluation import workflow_serving as serving
from ditserve.models.wan import runtime as wan_runtime


class VideoRetentionTests(unittest.TestCase):
    def setUp(self):
        self.video = np.zeros((1, 1, 480, 832, 3), dtype=np.float32)
        self.latent = np.zeros((1, 16, 1, 60, 104), dtype=np.float32)
        self.row = dict(kind="video", frames=1, repeat=0, finite=True, shape=list(self.video.shape),
                        sha256=dict(video=replay.array_sha256(self.video), latent=replay.array_sha256(self.latent)))

    def test_cli_retention_is_opt_in(self):
        arguments = ["--model-path", "/models/wan", "--output", "/tmp/not-created"]
        self.assertFalse(encoder._parse_args(arguments).save_videos)
        self.assertTrue(encoder._parse_args(arguments + ["--save-videos"]).save_videos)
        arguments = ["--encoder-root", "/tmp/source", "--out", "/tmp/not-created"]
        self.assertFalse(replay.parse_args(arguments).save_videos)
        self.assertTrue(replay.parse_args(arguments + ["--save-videos"]).save_videos)

    def test_replay_retains_latent_and_video_numeric_evidence_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latent = replay._save_replay_array(root, "latent", self.latent)
            video = replay._save_replay_array(root, "video", self.video)
            self.assertTrue(Path(latent["path"]).is_file())
            self.assertFalse((root / "video.npy").exists())
            self.assertNotIn("path", video)
            self.assertNotIn("file_sha256", video)
            self.assertFalse(video["retained"])
            self.assertEqual(video["statistics"], dict(min=0., max=0., mean=0.))
            saved = replay._save_replay_array(root, "video", self.video, save_videos=True)
            self.assertTrue(Path(saved["path"]).is_file())
            self.assertEqual(saved["array_sha256"], video["array_sha256"])

    def test_encoder_output_defaults_and_opt_in_preserve_identical_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, evidence = encoder._save_encoder_outputs(root, self.latent, self.video)
            self.assertEqual(set(artifacts), {"latent"})
            self.assertFalse((root / "video.npy").exists())
            self.assertEqual(evidence["array_sha256"], self.row["sha256"]["video"])
            retained, saved = encoder._save_encoder_outputs(root, self.latent, self.video, save_videos=True)
            self.assertEqual(set(retained), {"latent", "video"})
            self.assertEqual(dict(evidence, retained=True), saved)
            self.assertTrue(Path(retained["video"]).is_file())

    def test_serving_save_output_alone_preserves_latent_without_video(self):
        with tempfile.TemporaryDirectory() as directory:
            request = dict(request_id="r0", observation_id="o0", save_output=True)
            output = serving._save_request_arrays(directory, request, self.video, self.latent)
            self.assertEqual(set(output), {"latent"})
            self.assertEqual(list(Path(directory).rglob("*.video.npy")), [])
            output = serving._save_request_arrays(directory, request, self.video, self.latent, save_videos=True)
            self.assertEqual(set(output), {"latent", "video"})
            self.assertTrue(Path(output["video"]).is_file())

    def test_missing_legacy_t0_video_uses_recorded_hash_and_existing_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in ("T0", "T1"):
                record = dict(schema="ditserve.workflow.encoder.v1", variant=variant, status="completed",
                              configuration=dict(encoder_dtype="float32", video=variant == "T0"),
                              matmul_tf32=False, cudnn_tf32=True, numeric=[dict(case="english", finite=True)],
                              steps=20, frames=[1], requests=[self.row] if variant == "T0" else [])
                output = root / variant
                (output / "numeric").mkdir(parents=True)
                (output / "result.json").write_text(json.dumps(record))
                (output / "numeric/english.npz").write_bytes(b"audited embedding fixture")
            output = root / "T0/outputs/1"
            output.mkdir(parents=True)
            np.save(output / "latent.npy", self.latent, allow_pickle=False)
            args = SimpleNamespace(encoder_root=root, variants=["T1"], frames=[1], steps=20)
            with patch.object(replay, "load_embeddings", return_value={"prompt": np.zeros(1, dtype=np.float32)}):
                _, _, references, provenance = replay.audit_sources(args)
            expected = references[1]["video"]
            self.assertFalse(provenance["normal_t0_outputs"]["1"]["video"]["retained"])
            reference = replay._replay_video_reference(self.video, expected, "T0")
            candidate = self.video.copy()
            candidate.flat[0] = .1
            comparison = encoder.compare_arrays(candidate, reference)
            self.assertFalse(comparison["exact"])
            self.assertGreater(comparison["rmse"], 0)
            self.assertGreater(comparison["max_abs"], 0)
            self.assertIs(replay._replay_video_reference(candidate, reference, "T1"), reference)
            self.assertFalse((output / "video.npy").exists())

    def test_replay_bridge_rejects_mismatched_shape_dtype_hash_and_skipped_t0(self):
        expected = encoder._video_reference_metadata(self.row)
        changed = self.video.copy()
        changed.flat[0] = 1
        for array in (self.video.reshape(-1), self.video.astype(np.float64), changed):
            with self.subTest(shape=array.shape, dtype=array.dtype), self.assertRaisesRegex(ValueError, "shape/dtype/hash"):
                replay._replay_video_reference(array, expected, "T0")
        with self.assertRaisesRegex(ValueError, "before candidate"):
            replay._replay_video_reference(self.video, expected, "T1")

    def test_complete_replay_compares_in_memory_without_additional_video_inferences(self):
        generated, released = [], []
        latent = self.latent
        class Tensor:
            def detach(self):
                return self
            float = cpu = detach
            def numpy(self):
                return latent
        class Worker:
            def __init__(self, *_):
                self._owner, self.pipe = object(), SimpleNamespace()
            def prepare_encoded(self, spec, payload):
                return SimpleNamespace(spec=spec, latents=Tensor())
            def step(self, state):
                pass
            def release(self, state):
                released.append(state.spec["request_id"])
        def decode(worker, state, recorder, metadata):
            generated.append(metadata["variant"])
            result = self.video.copy()
            if metadata["variant"] == "T1":
                result.flat[0] = .1
            return result
        fake_torch = SimpleNamespace(__version__="fixture", version=SimpleNamespace(cuda="fixture"),
            set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None, from_numpy=lambda value: value,
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                     cudnn=SimpleNamespace(allow_tf32=True)),
            cuda=SimpleNamespace(set_device=lambda _: None, synchronize=lambda _: None, get_device_name=lambda _: "fixture"))
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", torch=fake_torch), \
                patch.object(wan_runtime, "WanStepWorker", Worker), \
                patch.object(replay, "NumericalRanges", return_value=SimpleNamespace(phase=lambda *_args, **_kwargs: contextlib.nullcontext())), \
                patch.object(replay, "_decode_separate", side_effect=decode), patch.dict("os.environ", {}, clear=False):
            args = SimpleNamespace(out=Path(directory), frames=[1], steps=1, variants=["T1"], save_videos=False,
                                   cpu_affinity="none", cpu_threads=1, model_path=Path("/models/wan"),
                                   attention="flash", device="cuda:1")
            arrays = {variant: {name: np.zeros(1, dtype=np.float32) for name in ("prompt", "negative")}
                      for variant in ("T0", "T1")}
            references = {1: dict(latent=self.latent, video=encoder._video_reference_metadata(self.row))}
            record = dict(results=[])
            replay.run_replay(args, {"T0": dict(matmul_tf32=False, cudnn_tf32=True)}, arrays,
                              references, record, Path(directory) / "run.json")
            self.assertEqual(generated, ["T0", "T1"])
            self.assertEqual(len(released), 2)
            self.assertTrue(record["results"][0]["baseline_bridge_exact"])
            self.assertGreater(record["results"][1]["comparison"]["video"]["rmse"], 0)
            self.assertEqual(list(Path(directory).rglob("video.npy")), [])
            self.assertEqual(len(list(Path(directory).rglob("latent.npy"))), 2)
            self.assertNotIn("video", references[1])

    def reconstruct(self, path, *, row=None, latent=None, returned=None):
        phases = []
        @contextlib.contextmanager
        def phase(name, **metadata):
            phases.append(dict(name=name, **metadata))
            yield
        worker = SimpleNamespace(device="fake-device", torch=SimpleNamespace(
            from_numpy=lambda value: SimpleNamespace(to=lambda device: value)))
        def decode(worker, state, recorder, metadata):
            self.assertEqual(metadata["purpose"], "numerical_reference_reconstruction")
            self.assertFalse(metadata["include_in_performance"])
            with recorder.phase("vae_decode", **metadata):
                return self.video.copy() if returned is None else returned
        with patch.object(encoder, "_decode_separate", side_effect=decode) as mocked:
            result = encoder._reference_video(worker, self.row if row is None else row,
                self.latent if latent is None else latent, path, SimpleNamespace(phase=phase),
                dict(purpose="video_measurement", repeat=0, frames=1))
        return result, phases, mocked.call_count

    def test_encoder_reconstruction_is_memory_only_and_separate_from_measured_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.npy"
            result, phases, count = self.reconstruct(path)
            self.assertEqual(count, 1)
            self.assertEqual(replay.array_sha256(result), self.row["sha256"]["video"])
            self.assertEqual([row["name"] for row in phases], ["reference_latent_h2d", "numerical_reference_vae_decode"])
            self.assertTrue(all(row["include_in_performance"] is False for row in phases))
            self.assertFalse(path.exists())
            np.save(path, self.video, allow_pickle=False)
            _, phases, count = self.reconstruct(path)
            self.assertEqual((phases, count), ([], 0))

    def test_encoder_reconstruction_fails_closed_on_changed_source_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.npy"
            row = copy.deepcopy(self.row)
            row["sha256"]["latent"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "latent reference"):
                self.reconstruct(path, row=row)
            with self.assertRaisesRegex(ValueError, "latent reference"):
                self.reconstruct(path, latent=self.latent.astype(np.float64))
            changed = self.video.copy()
            changed.flat[0] = 1
            with self.assertRaisesRegex(ValueError, "shape/dtype/hash"):
                self.reconstruct(path, returned=changed)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
