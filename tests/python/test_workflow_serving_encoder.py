"""CPU-only contracts for bounded prefetch, masks and quality attribution."""

import contextlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ditserve.evaluation import workflow_serving_encoder as study


class ConfigurationTests(unittest.TestCase):
    def test_import_does_not_import_torch_or_create_cuda_context(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import ditserve.evaluation.workflow_serving_encoder; assert 'torch' not in sys.modules"],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mask_must_use_distinct_allowed_physical_cores(self):
        topology = lambda cpu: (0, cpu % 4)
        self.assertEqual(study.validate_physical_mask([0, 1], range(8), topology), [(0, 0), (0, 1)])
        with self.assertRaisesRegex(ValueError, "SMT"):
            study.validate_physical_mask([0, 4], range(8), topology)
        with self.assertRaisesRegex(ValueError, "allowed"):
            study.validate_physical_mask([8], range(8), topology)
        with self.assertRaisesRegex(ValueError, "cpu_mask"):
            study.validate_config("cpu_micro", dict(cpu_threads=4, cpu_mask=[0, 1]))

    def test_quality_recipes_cover_each_case_seed_frame_without_aliasing(self):
        config = study.validate_config("quality", dict(cpu_threads=2, cpu_mask=[0, 1]))
        recipes = list(study.quality_recipes(config))
        self.assertEqual(len(recipes), 8)
        self.assertEqual(len({row["spec_sha256"] for row in recipes}), 8)
        self.assertEqual({(row["case"], row["seed"], row["frames"]) for row in recipes},
                         {(case, seed, frames) for case in ("chinese", "truncation")
                          for seed in (20261019, 20261020) for frames in (17, 33)})
        for row in recipes:
            self.assertEqual(row["spec"]["seed"], row["seed"])
            self.assertEqual(row["spec"]["prompt"], study.prompt_case(row["case"])["prompt"])
        self.assertFalse(config["save_videos"])
        for invalid in (1, 0, "true", None):
            with self.subTest(save_videos=invalid), self.assertRaisesRegex(ValueError, "save_videos"):
                study.validate_config("quality", dict(cpu_threads=2, cpu_mask=[0, 1], save_videos=invalid))

    def test_nonempty_output_is_immutable_and_oom_is_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(cpu_threads=2, cpu_mask=[0, 1])
            out = Path(directory) / "new"
            result = study._run("cpu_micro", config, out, lambda *args: None)
            self.assertEqual(result["status"], "completed")
            old = (out / "run.json").read_bytes()
            with self.assertRaises(FileExistsError):
                study._run("cpu_micro", config, out, lambda *args: self.fail("must not run"))
            self.assertEqual((out / "run.json").read_bytes(), old)
            exception = type("OutOfMemoryError", (RuntimeError,), {})
            def fail(*args):
                raise exception("fixture capacity exhaustion")
            result = study._run("quality", config, Path(directory) / "oom", fail)
            self.assertEqual(result["status"], "oom")
            self.assertEqual(result["error"]["type"], "OutOfMemoryError")


class BoundedPrefetchTests(unittest.TestCase):
    def test_depth_bounds_inflight_plus_ready_and_uses_one_producer(self):
        for depth in (0, 1, 2):
            with self.subTest(depth=depth):
                produced, disposed, consumed = [], [], []
                enough = threading.Event()
                def encode(item):
                    produced.append(item["index"])
                    if len(produced) >= depth + 1:
                        enough.set()
                    return dict(index=item["index"])
                def consume(item, payload):
                    self.assertEqual(item["index"], payload["index"])
                    if depth and item["index"] == 0:
                        self.assertTrue(enough.wait(1), "producer failed to fill the bounded pipeline")
                    consumed.append(item["index"])
                    return item["index"]
                result = study.bounded_prefetch([dict(index=index) for index in range(6)], encode, consume, depth,
                    dispose=lambda payload: disposed.append(payload["index"]), join_timeout_s=1)
                self.assertEqual(produced, list(range(6)))
                self.assertEqual(consumed, produced)
                self.assertEqual(disposed, produced)
                self.assertEqual(len(result["encoder_thread_ids"]), 1)
                self.assertLessEqual(result["max_prefetch_outstanding"], depth)
                self.assertTrue(all(event["outstanding"] <= depth for event in result["events"]))
                self.assertFalse(any(thread.name == "serving-cpu-encoder" for thread in threading.enumerate()))

    def test_no_encoding_before_arrival_and_completed_callback_preserves_results(self):
        completed = []
        result = study.bounded_prefetch([dict(index=0, arrival_offset_ms=15)], lambda item: 0,
            lambda item, payload: "output", 2, on_complete=completed.append, join_timeout_s=1)
        self.assertGreaterEqual(result["requests"][0]["encode_start_ns"] - result["started_ns"], 15_000_000)
        self.assertEqual(completed[0]["result"], "output")

    def test_encode_failure_terminates_without_deadlock_and_disposes_ready_payloads(self):
        produced, disposed = [], []
        def encode(item):
            if item["index"] == 2:
                raise ValueError("encode failed")
            produced.append(item["index"])
            return item["index"]
        with self.assertRaisesRegex(ValueError, "encode failed"):
            study.bounded_prefetch([dict(index=index) for index in range(6)], encode,
                lambda item, payload: time.sleep(.005), 2, dispose=disposed.append, join_timeout_s=1)
        self.assertEqual(sorted(disposed), sorted(produced))
        self.assertFalse(any(thread.name == "serving-cpu-encoder" for thread in threading.enumerate()))

    def test_consumer_failure_preserves_completed_items_and_releases_each_payload_once(self):
        produced, disposed, completed = [], [], []
        def encode(item):
            produced.append(item["index"])
            return item["index"]
        def consume(item, payload):
            if item["index"] == 1:
                raise RuntimeError("GPU consumer fixture failed")
            return payload
        with self.assertRaisesRegex(RuntimeError, "consumer fixture"):
            study.bounded_prefetch([dict(index=index) for index in range(6)], encode, consume, 2,
                dispose=disposed.append, on_complete=completed.append, join_timeout_s=1)
        self.assertEqual([row["index"] for row in completed], [0])
        self.assertEqual(sorted(disposed), sorted(produced))
        self.assertEqual(len(disposed), len(set(disposed)))


class FakeTensor:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class RuntimeContractTests(unittest.TestCase):
    def test_cpu_micro_never_constructs_wan_and_retains_complete_encoding_time(self):
        tensor = FakeTensor(np.zeros((1, 512, 4096), dtype=np.float32))
        calls = []
        def encode(**kwargs):
            calls.append(kwargs)
            return tensor, tensor
        fake_torch = SimpleNamespace(float32="float32", inference_mode=contextlib.nullcontext,
                                    cuda=SimpleNamespace(is_initialized=lambda: False))
        def runtime(config, record, gpu):
            self.assertFalse(gpu)
            record["runtime"] = dict(cuda_initialized=False)
            return fake_torch
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(study, "_runtime", side_effect=runtime), \
                patch.object(study, "_standalone_encoder", return_value=SimpleNamespace(encode_prompt=encode)) as loader, \
                patch("ditserve.models.wan.runtime.WanStepWorker", side_effect=AssertionError("CPU micro must not construct Wan")):
            result = study.run_cpu_micro(dict(cpu_threads=2, cpu_mask=[0, 1], prompt_cases=["shortenglish"],
                                             repeats=2, warmups=1), directory)
            self.assertEqual(result["status"], "completed", result.get("error"))
            self.assertEqual(len(result["samples"]), 2)
            self.assertEqual(len(calls), 3)
            loader.assert_called_once_with("/models/wan2.1-t2v-1.3b", "float32", "cpu")
            self.assertTrue(all(row["device"] == "cpu" for row in calls))
            self.assertFalse(result["qualification"]["cuda_initialized"])
            self.assertTrue(result["qualification"]["repeated_exact"])

    def quality_fixture(self, directory, *, corrupt_bridge=False, save_videos=False):
        transitions, loads = [], []
        def embeddings(config, out, record, variant, torch, recorder):
            transitions.append("encoder-" + variant)
        def factory(*args):
            self.assertEqual(transitions, ["encoder-T0", "encoder-T1", "encoder-T2"])
            transitions.append("wan")
            return SimpleNamespace(_owner=object())
        def load(path, dtype):
            loads.append((str(path), dtype))
            return {name: np.ones(2, dtype=np.float32) for name in ("prompt", "negative")}
        def video(worker, spec, payload, recorder, metadata):
            array = np.array([spec["seed"] % 100, spec["frames"]], dtype=np.float32)
            if metadata["variant"] in ("T1", "T2") or (corrupt_bridge and metadata["variant"] == "T0"):
                array = array + .01
            arrays = dict(latent=array, video=array.copy())
            return arrays, {name: study._array_info(value) for name, value in arrays.items()}
        fake_torch = SimpleNamespace(from_numpy=lambda value: value)
        with patch.object(study, "_runtime", return_value=fake_torch), \
                patch.object(study, "_quality_embeddings", side_effect=embeddings), \
                patch("ditserve.models.wan.runtime.WanStepWorker", side_effect=factory), \
                patch.object(study, "load_embeddings", side_effect=load), \
                patch.object(study, "_video", side_effect=video):
            result = study.run_quality(dict(cpu_threads=2, cpu_mask=[0, 1], save_videos=save_videos), directory)
        return result, loads

    def test_quality_has_32_paths_case_specific_artifacts_and_independent_cpu_bridges(self):
        with tempfile.TemporaryDirectory() as directory:
            result, loads = self.quality_fixture(directory)
            self.assertEqual(result["status"], "completed", result.get("error"))
            self.assertEqual(len(result["requests"]), 32)
            self.assertEqual(sum(row.get("baseline_bridge_exact") is True for row in result["requests"]), 8)
            self.assertEqual(sum(row["variant"] == "cpu_direct" for row in result["requests"]), 8)
            self.assertEqual(result["qualification"]["semantic_quality"], "unverified")
            self.assertFalse(result["include_in_performance"])
            for row in result["requests"]:
                artifact = row["artifacts"]["latent"]
                self.assertTrue(Path(artifact["path"]).is_file())
                self.assertIn(row["id"], artifact["path"])
                video = row["artifacts"]["video"]
                self.assertNotIn("path", video)
                self.assertNotIn("file_sha256", video)
                self.assertFalse(video["retained"])
                self.assertTrue(video["finite"])
                self.assertEqual(len(video["sha256"]), 64)
                if row["variant"] in ("T1", "T2"):
                    self.assertFalse(row["comparison"]["video"]["exact"])
            self.assertEqual(len(loads), 24)
            self.assertEqual({Path(path).stem for path, _ in loads}, {"chinese", "truncation"})
            self.assertEqual(list(Path(directory).rglob("video.npy")), [])
            json.dumps(result, allow_nan=False)

    def test_quality_video_opt_in_does_not_change_numeric_comparisons(self):
        with tempfile.TemporaryDirectory() as default, tempfile.TemporaryDirectory() as retained:
            baseline, _ = self.quality_fixture(default)
            result, _ = self.quality_fixture(retained, save_videos=True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(list(Path(retained).rglob("video.npy"))), 32)
            for before, after in zip(baseline["requests"], result["requests"]):
                self.assertEqual(before.get("comparison"), after.get("comparison"))
                self.assertEqual(before["artifacts"]["video"]["sha256"], after["artifacts"]["video"]["sha256"])
                self.assertTrue(after["artifacts"]["video"]["retained"])
                self.assertTrue(Path(after["artifacts"]["video"]["path"]).is_file())

    def test_failed_cpu_bridge_stops_before_precision_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            result, loads = self.quality_fixture(directory, corrupt_bridge=True)
            self.assertEqual(result["status"], "failed")
            self.assertEqual([row["variant"] for row in result["requests"]], ["cpu_direct", "T0"])
            self.assertEqual(len(loads), 1)
            self.assertEqual(result["requests"][-1]["status"], "baseline_bridge_failed")
            self.assertNotIn("cpu_bridge_exact", result["qualification"])


if __name__ == "__main__":
    unittest.main()
