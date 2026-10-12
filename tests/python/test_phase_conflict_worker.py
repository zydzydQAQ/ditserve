"""CPU-only contracts for the real phase worker; no CUDA/model imports."""

import concurrent.futures
import contextlib
import gc
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

from ditserve.gpu import phase_conflict as pc


def job(directory, **changes):
    result = {"id": "test", "mode": "prepare", "height": 480, "width": 832,
              "frames": 17, "steps": 50, "seed": 0, "cache": "none",
              "model_id": "/mock/checkpoint", "revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
              "output_dir": str(directory)}
    result.update(changes)
    return result


class FakeStorage:
    def __init__(self, pointer=123, size=64):
        self.pointer, self.size = pointer, size

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.size


class FakeTensor:
    shape = (4,)
    dtype = "torch.float32"
    device = "cuda:0"

    def __init__(self, storage=None):
        self.storage = storage or FakeStorage()
        self.recorded_streams = []

    def untyped_storage(self):
        return self.storage

    def storage_offset(self):
        return 0

    def numel(self):
        return 4

    def element_size(self):
        return 4

    def record_stream(self, stream):
        self.recorded_streams.append(stream)

    def to(self, *args, **kwargs):
        return self


class FakeEvent:
    def __init__(self, trace, **kwargs):
        self.trace = trace

    def record(self, stream=None):
        self.trace.append(("event_record", stream))

    def synchronize(self):
        self.trace.append(("event_synchronize", self))

    def elapsed_time(self, other):
        return 1.0


class FakeStream:
    cuda_stream = 17

    def __init__(self, trace):
        self.trace = trace

    def wait_event(self, event):
        self.trace.append(("wait_event", event))


class FakeCuda:
    def __init__(self, trace):
        self.trace = trace
        self.main = FakeStream(trace)

    def Stream(self, **kwargs):
        return FakeStream(self.trace)

    def Event(self, **kwargs):
        return FakeEvent(self.trace, **kwargs)

    def current_stream(self, device=None):
        return self.main

    def stream(self, stream):
        return contextlib.nullcontext()

    def device(self, device):
        return contextlib.nullcontext()

    def synchronize(self, device):
        self.trace.append(("global_synchronize", device))

    def reset_peak_memory_stats(self, device):
        self.trace.append(("reset_peak", device))

    def memory_allocated(self, device):
        return 10

    def memory_reserved(self, device):
        return 20


def fake_torch(trace):
    local = threading.local()

    @contextlib.contextmanager
    def inference_mode():
        old = getattr(local, "inference", False)
        local.inference = True
        try:
            yield
        finally:
            local.inference = old

    backends = SimpleNamespace(math_sdp_enabled=lambda: True, flash_sdp_enabled=lambda: False,
                               mem_efficient_sdp_enabled=lambda: False, cudnn_sdp_enabled=lambda: False)
    return SimpleNamespace(cuda=FakeCuda(trace), inference_mode=inference_mode,
                           is_inference_mode_enabled=lambda: getattr(local, "inference", False),
                           backends=SimpleNamespace(cuda=backends),
                           isfinite=lambda value: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)),
                           is_tensor=lambda value: isinstance(value, FakeTensor), float32="torch.float32",
                           save=Mock())


class NormalizeTests(unittest.TestCase):
    def test_fixed_protocol_and_extra_job_fields_are_preserved(self):
        normalized = pc.normalize_job(job("/tmp/test-phase", extra_field="retained"))
        self.assertEqual(normalized["dtype"], "float16")
        self.assertEqual(normalized["vae_dtype"], "float32")
        self.assertEqual(normalized["text_encoder_device"], "cpu")
        self.assertEqual(normalized["extra_field"], "retained")
        self.assertEqual(normalized["prompt_id"], "water")
        self.assertEqual(normalized["cache_parameters"]["threshold"], .08)

    def test_unsupported_input_fails_without_importing_torch(self):
        for change in ({"dtype": "float32"}, {"frames": 18}, {"height": 481}, {"seed": True},
                       {"revision": "wrong"}, {"model_id": "relative"}, {"mode": "wrong"},
                       {"mode": "case"}, {"mode": "case", "b_latent_path": "/tmp/b.pt",
                                           "strategy": "overlap", "steps": 25, "insert_after": 25}):
            with self.subTest(change=change), patch.object(pc.importlib, "import_module") as imports:
                with self.assertRaises(ValueError):
                    pc.normalize_job(job("/tmp/test-phase", **change))
                imports.assert_not_called()


class InventoryTests(unittest.TestCase):
    def test_views_share_one_storage_and_records_do_not_retain_tensors(self):
        storage = FakeStorage(size=128)
        tensor = FakeTensor(storage)
        view = FakeTensor(storage)
        reference = weakref.ref(tensor)
        view_reference = weakref.ref(view)
        data = pc.storage_inventory({"required": [tensor, tensor], "stack": [view]},
                                    lambda value: isinstance(value, FakeTensor))
        self.assertEqual(data["unique_storage_bytes"], 128)
        self.assertEqual(len(data["storages"]), 1)
        self.assertEqual(len(data["tensors"]), 2)
        self.assertEqual(data["storages"][0]["first_owner"], "required")
        json.dumps(data)
        del tensor, view
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(view_reference())


class CleanupTests(unittest.TestCase):
    def make_pipe(self, active=False, context=None):
        class Scheduler:
            config = {"solver_order": 2}

            @classmethod
            def from_config(cls, config):
                assert config == cls.config
                return cls()

        manager = SimpleNamespace(get_context=Mock(return_value=context),
                                  _cached_context_manager={"blocks": object()}, clear_contexts=Mock())
        return SimpleNamespace(_phase_conflict_active=active, _context_manager=manager, scheduler=Scheduler())

    def test_refuses_active_request_even_without_current_cache_context(self):
        for active, context in ((True, None), (False, object())):
            pipe = self.make_pipe(active, context)
            scheduler = pipe.scheduler
            with self.assertRaises(pc.p.ProfilingError):
                pc.release_completed_denoiser_state(pipe)
            self.assertIs(pipe.scheduler, scheduler)
            pipe._context_manager.clear_contexts.assert_not_called()

    def test_completed_request_drops_cache_and_recreates_identical_scheduler(self):
        pipe = self.make_pipe()
        scheduler = pipe.scheduler
        report = pc.release_completed_denoiser_state(pipe)
        pipe._context_manager.clear_contexts.assert_called_once_with()
        self.assertIsNot(pipe.scheduler, scheduler)
        self.assertEqual(pipe.scheduler.config, scheduler.config)
        self.assertFalse(report["weights_reloaded"])


class VaeThreadTests(unittest.TestCase):
    def test_worker_owns_inference_context_waits_boundary_and_never_global_syncs(self):
        trace = []
        torch = fake_torch(trace)
        progress = SimpleNamespace(mark=Mock(), path=Path("/tmp/progress.json"))
        pipe = object()
        boundary = FakeEvent(trace)
        latent = FakeTensor()
        record = {}
        array = object()

        def decode(actual_torch, actual_pipe, actual_latent):
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertTrue(threading.current_thread().name.startswith("phase-vae"))
            self.assertIs(actual_latent, latent)
            self.assertIn(("wait_event", boundary), trace)
            return array

        numpy = SimpleNamespace(isfinite=lambda value: SimpleNamespace(all=lambda: True))
        with patch.object(pc, "decode_final_latent", side_effect=decode), \
                patch.object(pc.importlib, "import_module", return_value=numpy):
            worker = pc.VaeWorker(torch, pipe, progress)
            try:
                future = worker.submit(latent, boundary, record, "test")
                self.assertIs(future.result(timeout=5), array)
            finally:
                worker.close()
        self.assertEqual(latent.recorded_streams, [worker.stream])
        self.assertEqual(record["status"], "completed")
        self.assertTrue(any(event[0] == "event_synchronize" for event in trace))
        self.assertFalse(any(event[0] == "global_synchronize" for event in trace))
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_bad_attention_policy_fails_before_calling_vae(self):
        torch = fake_torch([])
        torch.backends.cuda.flash_sdp_enabled = lambda: True
        with patch.object(pc, "decode_final_latent") as decode:
            worker = pc.VaeWorker(torch, object(), SimpleNamespace(mark=Mock()))
            try:
                future = worker.submit(FakeTensor(), FakeEvent([]), {}, "test")
                with self.assertRaises(pc.p.ProfilingError):
                    future.result(timeout=5)
            finally:
                worker.close()
            decode.assert_not_called()

    def test_pending_decode_transfers_ownership_and_prevents_double_launch(self):
        future = concurrent.futures.Future()
        future.set_result("cpu-output")
        worker = SimpleNamespace(submit=Mock(return_value=future))
        pending = pc.PendingDecode(FakeTensor(), worker, {})
        pending.launch(object())
        self.assertIsNone(pending.latent)
        self.assertEqual(pending.join(), "cpu-output")
        with self.assertRaises(pc.p.ProfilingError):
            pending.launch(object())
        self.assertEqual(worker.submit.call_count, 1)


class CallbackTests(unittest.TestCase):
    def test_residency_observer_does_not_pin_callers_old_latent_across_steps(self):
        torch = fake_torch([])
        pipe = SimpleNamespace(scheduler=SimpleNamespace(config={}))
        pending = SimpleNamespace(check_failure=Mock(), launch=Mock(), join=Mock(), record={})
        callback = pc.InsertDecodeCallback(torch,
            {"strategy": "deferred", "steps": 3, "insert_after": 1, "diagnostic": False},
            pipe, pending, SimpleNamespace(events=[None, None]), {"steps": []}, SimpleNamespace(mark=Mock()))

        def parent_trajectory():
            state = FakeTensor()
            previous = weakref.ref(state)
            kwargs = {"latents": state, "prompt_embeds": None, "negative_prompt_embeds": None}
            result = callback(pipe, 0, 1, kwargs)
            # Model the official pipeline consuming callback results before the
            # next step overwrites its fast local. f_locals observation would
            # leave the previous tensor alive in the parent frame's locals map.
            state = result.pop("latents", state)
            result.pop("prompt_embeds", None)
            result.pop("negative_prompt_embeds", None)
            state = FakeTensor()
            gc.collect()
            return previous()

        with patch.object(pc, "_memory", return_value={}), \
                patch("sys._getframe", side_effect=AssertionError("frame introspection pins tensors")):
            retained = parent_trajectory()
        self.assertIsNone(retained)

    def test_four_policies_launch_once_and_join_before_a_vae(self):
        for strategy, launches, joins in (("immediate", 0, 1), ("deferred", 0, 0),
                                           ("blocking", 1, 2), ("overlap", 1, 1)):
            with self.subTest(strategy=strategy):
                trace = []
                torch = fake_torch(trace)
                pending = SimpleNamespace(check_failure=Mock(), launch=Mock(), join=Mock(), record={})
                forwards = SimpleNamespace(events=[])
                events = {"steps": []}
                config = {"strategy": strategy, "steps": 4, "insert_after": 2, "diagnostic": False}
                callback = pc.InsertDecodeCallback(torch, config, object(), pending, forwards, events,
                                                 SimpleNamespace(mark=Mock()))
                with patch.object(pc, "_active_snapshot", return_value={}), \
                        patch.object(pc, "_memory", return_value={"peak_allocated_bytes": 20}):
                    for index in range(4):
                        forwards.events.extend([None, None])
                        kwargs = {"latents": object()}
                        self.assertIs(callback(callback.pipe, index, 1, kwargs), kwargs)
                self.assertEqual(pending.launch.call_count, launches)
                self.assertEqual(pending.join.call_count, joins)
                self.assertEqual(events["steps"][-1]["forward_count"], 8)
                self.assertIn("before_a_vae", events)
                if launches:
                    boundary = pending.launch.call_args.args[0]
                    self.assertTrue(any(item[0] == "event_record" for item in trace))
                    self.assertIs(callback.boundaries[0][1], boundary)

    def test_b_failure_stops_at_next_callback_without_advancing_step_record(self):
        pending = SimpleNamespace(check_failure=Mock(side_effect=RuntimeError("B failed")))
        events = {"steps": []}
        callback = pc.InsertDecodeCallback(fake_torch([]), {}, object(), pending, object(), events,
                                         SimpleNamespace(mark=Mock()))
        with self.assertRaisesRegex(RuntimeError, "B failed"):
            callback(callback.pipe, 0, 1, {"latents": object()})
        self.assertEqual(events["steps"], [])


class ProvenanceTests(unittest.TestCase):
    def test_b_identity_defaults_to_a_but_explicit_b_seed_allows_fixture_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            config = pc.normalize_job(job(directory, mode="case", seed=1, b_latent_path="/tmp/b.pt"))
            config["b_metadata_path"] = str(Path(directory) / "provenance.json")
            meta_config = dict(config, seed=0)
            latent_record = {"sha256": "payload", "shape": [1, 16, 5, 60, 104], "dtype": "float32", "finite": True}
            metadata = {"latent": latent_record, "config": meta_config, "output_b": {"sha256": "pixels"}}
            Path(config["b_metadata_path"]).write_text(json.dumps(metadata))
            tensor = FakeTensor()
            torch = SimpleNamespace(load=Mock(return_value=tensor), float32="torch.float32")
            with patch.object(pc, "_latent_fingerprint", return_value=(latent_record, tensor)):
                with self.assertRaisesRegex(pc.p.ProfilingError, "seed"):
                    pc._load_b(torch, config, {})
                config["b_seed"] = 0
                report = {}
                self.assertIs(pc._load_b(torch, config, report), tensor)
            self.assertEqual(report["b_source"]["config"]["seed"], 0)
            self.assertTrue(torch.load.call_args.kwargs["weights_only"])

    def test_bad_b_payload_hash_and_shape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = pc.normalize_job(job(directory, mode="case", b_latent_path="/tmp/b.pt"))
            config["b_metadata_path"] = str(Path(directory) / "provenance.json")
            tensor = FakeTensor()
            torch = SimpleNamespace(load=Mock(return_value=tensor), float32="torch.float32")
            for sha, shape in (("wrong", [1, 16, 5, 60, 104]), ("payload", [1, 16, 9, 60, 104])):
                latent = {"sha256": "payload", "shape": shape, "dtype": "float32", "finite": True}
                metadata = {"latent": dict(latent, sha256=sha), "config": config, "output_b": {}}
                Path(config["b_metadata_path"]).write_text(json.dumps(metadata))
                with patch.object(pc, "_latent_fingerprint", return_value=(latent, tensor)):
                    with self.assertRaises(pc.p.ProfilingError):
                        pc._load_b(torch, config, {})


class PersistenceTests(unittest.TestCase):
    def test_preflight_failure_is_written_and_existing_profile_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            request = job(directory)
            uuid = pc.normalize_job(request)["gpu_uuid"]
            env = {"CUDA_VISIBLE_DEVICES": uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
            with patch.dict(pc.os.environ, env), patch.object(pc.p, "preflight", return_value={"ready": False, "errors": ["mock failure"]}):
                result = pc.run_worker(request)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure"]["category"], "preflight")
            self.assertFalse(result["measured"])
            saved = json.loads((Path(directory) / "profile.json").read_text())
            self.assertEqual(saved["failure"], result["failure"])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                pc.run_worker(request)

    def test_nonfinite_output_is_saved_before_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            failure = pc.p.ProfilingError("NONFINITE_OUTPUT", category="nonfinite_output",
                                          details={"output": {"sha256": "bad", "finite": False}})
            numpy = SimpleNamespace(save=Mock())
            record = {}
            with patch.object(pc.p, "_fingerprint", side_effect=failure), \
                    patch.object(pc.importlib, "import_module", return_value=numpy):
                with self.assertRaises(pc.p.ProfilingError):
                    pc._save_output(object(), directory, "failure", record, save=False)
            self.assertFalse(record["finite"])
            self.assertTrue(record["artifact"].endswith("failure.npy"))
            numpy.save.assert_called_once()

    def test_thread_joins_before_global_math_restoration_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = []
            request = job(directory)
            config = pc.normalize_job(request)
            env = {"CUDA_VISIBLE_DEVICES": config["gpu_uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
            environment = {"ready": True, "packages": pc.EXPECTED_PACKAGES,
                           "gpu": {"uuid": config["gpu_uuid"]}}
            torch = fake_torch(trace)
            torch.cuda.set_device = Mock()
            pipe = SimpleNamespace(device="cuda:0", _execution_device="cuda:0",
                                   vae=SimpleNamespace(use_slicing=False, use_tiling=False),
                                   scheduler=SimpleNamespace(config={}))

            @contextlib.contextmanager
            def math_guard(*args):
                trace.append("math_enter")
                try:
                    yield
                finally:
                    trace.append("math_exit")

            decoder = SimpleNamespace(close=lambda: trace.append("decoder_close"))
            with patch.dict(pc.os.environ, env), \
                    patch.object(pc.p, "preflight", return_value=environment), \
                    patch.object(pc.importlib, "import_module", side_effect=lambda name: torch if name == "torch" else object()), \
                    patch.object(pc.p, "_pipeline", return_value=pipe), \
                    patch.object(pc.p, "CacheBackend", return_value=SimpleNamespace(config=None)), \
                    patch.object(pc.p, "_component_metadata", return_value={}), \
                    patch.object(pc.p, "_revision_metadata", return_value={}), \
                    patch.object(pc.p, "_sdpa_math_context", side_effect=math_guard), \
                    patch.object(pc, "VaeWorker", return_value=decoder), \
                    patch.object(pc, "_warmup", side_effect=RuntimeError("mock warmup failed")):
                report = pc.run_worker(request)
            self.assertEqual(report["status"], "failed")
            self.assertLess(trace.index("decoder_close"), trace.index("math_exit"))


if __name__ == "__main__":
    unittest.main()
