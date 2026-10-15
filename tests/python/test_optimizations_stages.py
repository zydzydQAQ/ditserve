"""Lightweight standard-library fixtures; no torch import and no GPU work."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.directions.budget import BudgetStop
from ditserve.optimizations.stage_phases import (
    FP32Encoder, FunctionalDecoder, run_text_encoder, run_vae, retain, install_encoder_defaults,
    generate, _record_full_warmup,
)


class OOM(RuntimeError):
    pass


class Tensor:
    def __init__(self, device="cuda:0", dtype="float32"):
        self.device, self.dtype = SimpleNamespace(type=device), dtype
    def detach(self):
        return self
    def to(self, *, device, dtype):
        return Tensor(device, dtype)


class Model:
    def __init__(self):
        self.device, self.moves, self.oom = "cpu", [], False
    def to(self, *, device, dtype):
        self.moves.append((device, dtype))
        if self.oom and device == "cuda:0":
            # Simulate a partially completed module transfer.
            self.device = "partial_cuda"
            raise OOM("no memory")
        self.device = device
        return self


class Guard:
    def __init__(self):
        self.units = []
        self.fail_at = None
    def __call__(self):
        pass
    def ensure_unit(self, seconds):
        self.units.append(seconds)
        if len(self.units) == self.fail_at:
            raise BudgetStop("between units")


class Array:
    shape = (1, 81, 2, 2, 3)
    def copy(self):
        return Array()


class FullArray(Array):
    shape = (1, 81, 480, 832, 3)
    def copy(self):
        return FullArray()


def request():
    return dict(request_id="fixture", prompt="a", negative_prompt="", seed=101,
                height=480, width=832, frames=81, steps=32, guidance_scale=5.,
                cache=False, threshold=0., max_continuous=1)


def worker():
    model, encode_calls = Model(), []
    def encode_prompt(**kwargs):
        encode_calls.append(kwargs)
        return Tensor(kwargs["device"]), Tensor(kwargs["device"])
    def validate(prompt, negative, spec):
        assert prompt.device.type == negative.device.type == "cpu"
        assert prompt.dtype == negative.dtype == "float32"
    value = SimpleNamespace(
        device="cuda:0", _owner=object(), _encoder_lock=threading.RLock(),
        torch=SimpleNamespace(float32="float32", inference_mode=nullcontext,
                              cuda=SimpleNamespace(synchronize=lambda device: None, OutOfMemoryError=OOM)),
        pipe=SimpleNamespace(text_encoder=model, encode_prompt=encode_prompt,
                             vae=SimpleNamespace(dtype="float32")),
        _validate_cpu_embeddings=validate, encode_prompt=lambda spec: None)
    value.encode_calls = encode_calls
    return value


def context(directory):
    return SimpleNamespace(output=Path(directory), run_root=Path(directory), config={},
                           metadata={}, opts={"prompts": ["a", "b"], "steps": 32},
                           profile={"selected_shape": {"height": 480, "width": 832, "frames": 81}},
                           worker=worker(), check_budget=Guard())


class WarmEvidenceTests(unittest.TestCase):
    def local_generate(self, directory, failure=None):
        value = context(directory)
        value.metadata['phase'] = 'text_encoder'
        state = SimpleNamespace(step_index=0)
        def step(current):
            if failure == 'step':
                raise OOM('incomplete trajectory')
            current.step_index += 1
        def finish(current):
            if failure == 'decode':
                raise OOM('decode failed')
            return FullArray()
        def release(current):
            if failure == 'release':
                raise RuntimeError('release failed')
            current.released = True
        value.worker.prepare_encoded = lambda spec, payload: state
        value.worker.step, value.worker.finish, value.worker.release = step, finish, release
        value.worker._sync, value.worker._memory = lambda: None, lambda: {}
        evidence = SimpleNamespace(data={'active_unit': {}})
        return value, state, evidence

    def test_complete_generation_marks_warm_only_after_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            value, state, evidence = self.local_generate(directory)
            generate(value, evidence, request(), lambda spec: SimpleNamespace(cpu_prompt_encode_ms=1.))
            self.assertTrue(value.metadata['wan_warmed'])
            self.assertTrue(state.released)
            self.assertEqual(value.metadata['wan_warm_evidence']['completed_steps'], 32)
            self.assertEqual(value.metadata['wan_warm_evidence']['shape'], list(FullArray.shape))

    def test_failed_step_decode_or_cleanup_does_not_mark_warm(self):
        for failure in ('step', 'decode', 'release'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                value, state, evidence = self.local_generate(directory, failure)
                with self.assertRaises(RuntimeError):
                    generate(value, evidence, request(), lambda spec: SimpleNamespace(cpu_prompt_encode_ms=1.))
                self.assertNotIn('wan_warmed', value.metadata)

    def test_partial_shape_or_step_count_cannot_supply_capacity_evidence(self):
        value = context('/unused')
        _record_full_warmup(value, request(), Array(), 32, 'small output')
        _record_full_warmup(value, request(), FullArray(), 31, 'partial steps')
        self.assertNotIn('wan_warmed', value.metadata)

    def test_real_online_orchestrator_reuses_completed_warm_evidence(self):
        from ditserve.optimizations.worker import _online
        with tempfile.TemporaryDirectory() as directory:
            value, state, evidence = self.local_generate(directory)
            generate(value, evidence, request(), lambda spec: SimpleNamespace(cpu_prompt_encode_ms=1.))
            value.config['old_online'] = 'fake-frozen-source'
            value.check_budget.remaining = lambda: 10000.
            value.executor = object()
            value.sana = SimpleNamespace(set_backend=lambda backend: None,
                generate_batch=lambda *args, **kwargs: None, steady_context=nullcontext)
            trace = dict(seed=101, load=.5, pattern='ordinary', trace_sha256='fixture',
                         requests=[dict(arrival_ms=0., service_ms=1., kind='image')])
            with patch('ditserve.optimizations.online.load_frozen_traces', return_value=[trace]), \
                 patch('ditserve.optimizations.online.run_trace', return_value={'complete': True}), \
                 patch('ditserve.optimizations.worker.capacity') as capacity:
                _online(value, {}, baseline_only=True)
            capacity.assert_not_called()


class EncoderTests(unittest.TestCase):
    def test_gpu_fp32_encoding_returns_cpu_fp32_owned_payload(self):
        value = worker()
        encoder = FP32Encoder(value, "cuda:1")
        encoder.move("cuda:1")
        payload = encoder.encode(request())
        self.assertIs(payload.owner, value._owner)
        self.assertFalse(payload.consumed)
        self.assertEqual(payload.prompt_embeds.device.type, "cpu")
        self.assertEqual(payload.encoding_device, "cuda:1")
        self.assertEqual(value.encode_calls[0]["dtype"], "float32")
        self.assertEqual(value.encode_calls[0]["max_sequence_length"], 512)
        encoder.restore_cpu()
        self.assertEqual(value.pipe.text_encoder.device, "cpu")

    def test_partial_transfer_oom_forces_restore_despite_cpu_bookkeeping(self):
        value = worker()
        value.pipe.text_encoder.oom = True
        encoder = FP32Encoder(value, "cuda:0")
        with self.assertRaises(OOM):
            encoder.move("cuda:0")
        self.assertEqual(encoder.current_device, "cpu")
        self.assertEqual(value.pipe.text_encoder.device, "partial_cuda")
        encoder.restore_cpu()
        self.assertEqual(value.pipe.text_encoder.device, "cpu")

    def test_unadmitted_device_and_unplaced_encoder_rejected(self):
        with self.assertRaises(ValueError):
            FP32Encoder(worker(), "cuda:2")
        value = FP32Encoder(worker(), "cuda:1")
        with self.assertRaises(RuntimeError):
            value.encode(request())
        with self.assertRaises(ValueError):
            value.move("cuda:0")

    def test_four_full_pairs_before_online_and_arm_placement_is_outside_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            value, observed = context(directory), []
            def generate(ctx, evidence, spec, encode):
                gpu = ctx.worker.pipe.text_encoder.device.startswith("cuda")
                return Array(), dict(encode_ms=1000. if gpu else 20000.,
                                     generation_ms=141000. if gpu else 160000., completed=True,
                                     frames=81, completed_steps=32, memory={})
            def online(ctx, result, options):
                self.assertTrue(result["local_qualified"])
                self.assertEqual(result["complete_pairs"], 4)
                self.assertTrue(options["encode_on_scheduler"])
                ctx.prepare_online_arm(options)
                observed.append(ctx.worker.pipe.text_encoder.device)
                ctx.prepare_online_arm({})
                observed.append(ctx.worker.pipe.text_encoder.device)
                result["online_qualified"] = False
            value.run_online = online
            with patch("ditserve.optimizations.stage_phases.generate", generate), \
                 patch("ditserve.optimizations.stage_phases.retain", return_value={"path": "fixture"}), \
                 patch("ditserve.optimizations.stage_phases.quality", return_value={"passed": True}):
                result = run_text_encoder(value)
            self.assertEqual(len(result["rows"]), 8)
            self.assertEqual(result["summaries"]["gpu"]["generation_ms"]["n"], 4)
            self.assertEqual(observed, ["cuda:0", "cpu"])
            self.assertFalse(result["online_qualified"])
            self.assertFalse(hasattr(value, "prepare_online_arm"))

    def test_a100_oom_requests_fresh_v100_attempt_and_restores_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            value = context(directory)
            value.worker.pipe.text_encoder.oom = True
            with patch("ditserve.optimizations.stage_phases.generate", return_value=(Array(), {"encode_ms": 20., "generation_ms": 160.})), \
                 patch("ditserve.optimizations.stage_phases.retain", return_value={}):
                result = run_text_encoder(value)
            self.assertTrue(result["needs_v100"])
            self.assertFalse(result["local_qualified"])
            self.assertEqual(len(result["rows"]), 1)
            self.assertEqual(value.worker.pipe.text_encoder.device, "cpu")

    def test_budget_between_pairs_preserves_completed_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            value = context(directory)
            value.check_budget.fail_at = 2
            with patch("ditserve.optimizations.stage_phases.generate", return_value=(Array(), {"encode_ms": 20., "generation_ms": 160.})), \
                 patch("ditserve.optimizations.stage_phases.retain", return_value={}), \
                 patch("ditserve.optimizations.stage_phases.quality", return_value={"passed": True}):
                with self.assertRaises(BudgetStop):
                    run_text_encoder(value)
            result = json.loads((Path(directory) / "text_encoder.json").read_text())
            self.assertEqual(len(result["rows"]), 2)
            self.assertEqual(len(result["pairs"]), 1)
            self.assertFalse(result["active_unit"]["started"])
            self.assertFalse(result["local_qualified"])

    def test_cpu_baseline_oom_does_not_request_encoder_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            value = context(directory)
            with patch("ditserve.optimizations.stage_phases.generate", side_effect=OOM("baseline OOM")):
                result = run_text_encoder(value)
            self.assertFalse(result["needs_v100"])
            self.assertFalse(result["local_qualified"])

    def test_online_budget_keeps_local_measurement_but_never_online_qualifies(self):
        with tempfile.TemporaryDirectory() as directory:
            value = context(directory)
            def generate(ctx, evidence, spec, encode):
                gpu = ctx.worker.pipe.text_encoder.device.startswith("cuda")
                return Array(), {"encode_ms": 1. if gpu else 20., "generation_ms": 140. if gpu else 160.}
            def online(*args):
                raise BudgetStop("online incomplete")
            value.run_online = online
            with patch("ditserve.optimizations.stage_phases.generate", generate), \
                 patch("ditserve.optimizations.stage_phases.retain", return_value={}), \
                 patch("ditserve.optimizations.stage_phases.quality", return_value={"passed": True}):
                with self.assertRaises(BudgetStop):
                    run_text_encoder(value)
            result = json.loads((Path(directory) / "text_encoder.json").read_text())
            self.assertEqual(result["status"], "stopped")
            self.assertTrue(result["local_gate_passed_before_failure"])
            self.assertTrue(result["local_qualified"])
            self.assertFalse(result["online_qualified"])

    def test_inherited_v100_requires_admission_and_moves_only_before_online(self):
        with tempfile.TemporaryDirectory() as directory:
            value = context(directory)
            value.metadata["phase"] = "vae"
            with self.assertRaisesRegex(ValueError, "admitted"):
                install_encoder_defaults(value, {"encode_device": "cuda:1"})
            value.metadata["encoder_device"] = "cuda:1"
            encoder = install_encoder_defaults(value, {"encode_device": "cuda:1"})
            self.assertEqual(encoder.current_device, "cpu")
            self.assertFalse(value.online_defaults["encode_on_scheduler"])
            value.prepare_online_arm(value.online_defaults)
            self.assertEqual(encoder.current_device, "cuda:1")
            value.encoder_cleanup()
            value.encoder_cleanup()
            self.assertEqual(value.worker.pipe.text_encoder.device, "cpu")
            self.assertFalse(hasattr(value, "prepare_online_arm"))

    def test_inherited_a100_policy_chains_prior_prepare_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            value = context(directory)
            value.metadata["phase"] = "cache_k4"
            calls = []
            prior = lambda options: calls.append(options["policy"])
            value.prepare_online_arm = prior
            install_encoder_defaults(value, {"encode_device": "cuda:0"})
            self.assertTrue(value.online_defaults["encode_on_scheduler"])
            value.prepare_online_arm(dict(value.online_defaults, policy="block-4"))
            self.assertEqual(calls, ["block-4"])
            self.assertEqual(value.worker.pipe.text_encoder.device, "cuda:0")
            value.encoder_cleanup()
            self.assertIs(value.prepare_online_arm, prior)


class DecoderTests(unittest.TestCase):
    def decoder(self):
        def original(value, feat_cache=None, feat_idx=None, first_chunk=False):
            if feat_cache is not None:
                value += feat_cache[0] or 0
                feat_cache[0] = value
                feat_idx[0] += 1
            return value * 2
        module = SimpleNamespace(forward=original)
        value = SimpleNamespace(pipe=SimpleNamespace(vae=SimpleNamespace(decoder=module, clear_cache=lambda: None)))
        return FunctionalDecoder(value)

    def test_functional_call_keeps_input_cache_container_unchanged(self):
        decoder = self.decoder()
        cache = (3,)
        output, after, index = decoder.pure(4, cache, 0, False)
        self.assertEqual((output, after, index), (14, (7,), 1))
        self.assertEqual(cache, (3,))

    def test_external_cache_updates_only_after_pure_call_returns(self):
        decoder = self.decoder()
        decoder.compiled = decoder.pure
        cache, index = [None], [0]
        self.assertEqual(decoder.forward(2, cache, index, True), 4)
        self.assertEqual((cache, index), ([2], [1]))
        index[0] = 0
        self.assertEqual(decoder.forward(3, cache, index, False), 10)
        self.assertEqual(cache, [5])

    def test_failed_pure_call_does_not_install_partial_cache(self):
        decoder = self.decoder()
        def failed(*args):
            raise RuntimeError("compiler failed")
        decoder.compiled = failed
        cache, index = [None], [0]
        with self.assertRaises(RuntimeError):
            decoder.forward(2, cache, index, True)
        self.assertEqual((cache, index), ([None], [0]))

    @unittest.skipUnless(os.environ.get("DITSERVE_STAGE_COMPILER_CPU_SMOKE") == "1",
                         "opt in only while no GPU experiment is timing CPU encoding")
    def test_real_cpu_inductor_cache_transition_and_backend_switch(self):
        import torch
        class Decoder(torch.nn.Module):
            def forward(self, value, feat_cache=None, feat_idx=None, first_chunk=False):
                prior = 0 if feat_cache[0] is None else feat_cache[0]
                output = torch.sin(value) + prior
                feat_cache[0] = value.square()
                feat_idx[0] += 1
                return output
        module = Decoder()
        fake = SimpleNamespace(torch=torch, pipe=SimpleNamespace(
            vae=SimpleNamespace(decoder=module, clear_cache=lambda: None)))
        backend = FunctionalDecoder(fake)
        tensor = torch.arange(4, dtype=torch.float32)
        with torch.inference_mode():
            backend.select(True)
            cache = [None]
            module(tensor, feat_cache=cache, feat_idx=[0], first_chunk=True)
            module(tensor, feat_cache=cache, feat_idx=[0], first_chunk=False)
            module(tensor, feat_cache=cache, feat_idx=[0], first_chunk=False)
            compiled_count = len(backend.records)
            backend.select(False)
            reference_cache = [None]
            first = module(tensor, feat_cache=reference_cache, feat_idx=[0], first_chunk=True)
            second = module(tensor, feat_cache=reference_cache, feat_idx=[0], first_chunk=False)
            backend.select(True)
            with torch.compiler.set_stance("fail_on_recompile"):
                cache = [None]
                actual_first = module(tensor, feat_cache=cache, feat_idx=[0], first_chunk=True)
                actual_second = module(tensor, feat_cache=cache, feat_idx=[0], first_chunk=False)
        torch.testing.assert_close(first, actual_first)
        torch.testing.assert_close(second, actual_second)
        torch.testing.assert_close(cache[0], reference_cache[0], rtol=0, atol=0)
        self.assertEqual(len(backend.records), compiled_count)
        self.assertGreater(compiled_count, 0)
        backend.close()

    def test_four_input_decode_matrix_and_complete_online_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            value = context(directory)
            class Backend:
                def __init__(self, worker):
                    self.records = []
                def select(self, compiled):
                    value.is_compiled = compiled
                def close(self):
                    value.closed = True
            def decode(ctx, latent, compiled=False):
                return FullArray(), {"ms": 80. if value.is_compiled else 100., "memory": {}}
            def online(ctx, result, options):
                self.assertTrue(ctx.metadata['wan_warmed'])
                self.assertEqual(ctx.metadata['wan_warm_evidence']['source'], 'vae_local_full_decode')
                self.assertEqual(options["vae_backend"], "compiled")
                self.assertTrue(callable(options["finish_video"]))
                ctx.prepare_online_arm(options)
                self.assertTrue(value.is_compiled)
                ctx.prepare_online_arm({})
                self.assertFalse(value.is_compiled)
                result["online_qualified"] = False
            value.run_online = online
            with patch("ditserve.optimizations.stage_phases.FunctionalDecoder", Backend), \
                 patch("ditserve.optimizations.stage_phases.final_latent", return_value=object()), \
                 patch("ditserve.optimizations.stage_phases.decode", decode), \
                 patch("ditserve.optimizations.stage_phases.quality", return_value={"passed": True}), \
                 patch("ditserve.optimizations.stage_phases.retain", return_value={}):
                result = run_vae(value)
            self.assertTrue(result["local_qualified"])
            self.assertEqual(result["complete_inputs"], 4)
            self.assertEqual(result["summaries"]["compiled"]["n"], 8)
            self.assertEqual(result["summaries"]["eager"]["n"], 8)
            self.assertEqual(result["measurement_matrix"]["expected_decodes"], 16)
            self.assertFalse(result["online_qualified"])
            self.assertTrue(value.closed)

    def test_failed_vae_trajectory_or_first_decode_cannot_mark_warm(self):
        for failure in ('latent', 'decode'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                value = context(directory)
                class Backend:
                    def __init__(self, worker): self.records = []
                    def select(self, compiled): pass
                    def close(self): pass
                with patch('ditserve.optimizations.stage_phases.FunctionalDecoder', Backend), \
                     patch('ditserve.optimizations.stage_phases.final_latent',
                           side_effect=OOM('trajectory OOM') if failure == 'latent' else None,
                           return_value=object()), \
                     patch('ditserve.optimizations.stage_phases.decode', side_effect=OOM('decode OOM')):
                    result = run_vae(value)
                self.assertEqual(result['status'], 'failed')
                self.assertNotIn('wan_warmed', value.metadata)

    def test_lpips_failure_keeps_completed_decode_artifact_and_service(self):
        with tempfile.TemporaryDirectory() as directory:
            value = context(directory)
            class Backend:
                def __init__(self, worker):
                    self.records = []
                def select(self, compiled):
                    pass
                def close(self):
                    pass
            with patch("ditserve.optimizations.stage_phases.FunctionalDecoder", Backend), \
                 patch("ditserve.optimizations.stage_phases.final_latent", return_value=object()), \
                 patch("ditserve.optimizations.stage_phases.decode", return_value=(Array(), {"ms": 123., "memory": {}})), \
                 patch("ditserve.optimizations.stage_phases.quality", side_effect=RuntimeError("LPIPS unavailable")), \
                 patch("ditserve.optimizations.stage_phases.retain", return_value={"path": "completed.npy"}):
                result = run_vae(value)
            row = next(row for row in result["rows"] if row["kind"] == "measurement")
            self.assertTrue(row["generation_completed"])
            self.assertEqual(row["service"]["ms"], 123.)
            self.assertEqual(row["artifact"]["path"], "completed.npy")
            self.assertEqual(row["quality_status"], "failed")
            self.assertFalse(result["local_qualified"])

    def test_artifact_path_is_unique_per_phase_and_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            value = context(directory)
            value.run_root = Path(directory) / "run01"
            value.output = value.run_root / "vae" / "attempt02"
            value.metadata["phase"] = "vae"
            value.config["artifact_root"] = str(Path(directory) / "artifacts")
            fake_numpy = SimpleNamespace(save=lambda stream, array, allow_pickle: stream.write(b"fixture"))
            with patch.dict("sys.modules", {"numpy": fake_numpy}):
                artifact = retain(value, "vae-example", Array())
            self.assertEqual(Path(artifact["path"]), Path(directory) / "artifacts/run01/vae/attempt02/vae-example.npy")


if __name__ == "__main__":
    unittest.main()
