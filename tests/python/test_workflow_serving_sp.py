"""CPU-only contract tests for schedule identity and fail-closed SP gates."""

import ast
import contextlib
import dataclasses
import inspect
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_serving_sp as workflow


class GateTests(unittest.TestCase):
    def comparison(self, **changes):
        result = dict(shape_match=True, finite=True, reference_finite=True, max_abs=0.0, exact=True)
        result.update(changes)
        return result

    def test_threshold_is_inclusive_and_never_accepts_missing_or_nonfinite(self):
        self.assertTrue(workflow.comparison_passes(self.comparison(max_abs=0.05, exact=False)))
        for value in ({}, self.comparison(max_abs=0.0500001), self.comparison(max_abs=float("nan")),
                      self.comparison(max_abs=None), self.comparison(shape_match=False),
                      self.comparison(finite=False), self.comparison(reference_finite=False)):
            self.assertFalse(workflow.comparison_passes(value), value)
        self.assertFalse(workflow.comparison_passes(self.comparison(max_abs=1e-9, exact=False), exact=True))

    def test_failed_prefix_never_advances_and_success_requires_immediate_predecessor(self):
        rows = [{"mode": "bad", "prefixes": [{"prefix": 1, "passed": False}]},
                {"mode": "good", "prefixes": [{"prefix": 1, "passed": True}]},
                {"mode": "unrun", "prefixes": []}]
        self.assertEqual([row["mode"] for row in workflow.active_variants(rows, 4)], ["good"])
        self.assertFalse(workflow.active_variants(rows, 20))
        rows[1]["prefixes"].append({"prefix": 4, "passed": True})
        self.assertEqual([row["mode"] for row in workflow.active_variants(rows, 20)], ["good"])

    def test_config_rejects_short_scheduler_and_relaxed_threshold(self):
        for value in ({"steps": 4}, {"threshold": 0.1}, {"prefixes": [4, 20]},
                      {"devices": ["cuda:1", "cuda:1"]}):
            with self.assertRaises(ValueError):
                workflow.normalize_config(value)
        value = workflow.normalize_config({"frames": 17, "custom_provenance": "kept"})
        self.assertEqual(value["frames"], [17])
        self.assertEqual(value["steps"], 20)
        self.assertEqual(value["custom_provenance"], "kept")
        self.assertEqual(value["seeds"], [value["seed"]])
        multi = workflow.normalize_config({"seed": 3501, "seeds": [3501, 3502]})
        self.assertEqual(multi["seeds"], [3501, 3502])
        for seeds in ([], [3501, 3501], [True], [-1]):
            with self.assertRaises(ValueError):
                workflow.normalize_config({"seeds": seeds})

    def test_group_timeout_matches_production_contract_and_is_separate_from_job_budget(self):
        self.assertEqual(workflow.normalize_config({})["group_timeout_seconds"], 60)
        value = workflow.normalize_config({"timeout_seconds": 5400, "group_timeout_seconds": 15})
        self.assertEqual(value["timeout_seconds"], 5400)
        for timeout in (0, -1, 61, 120, float("nan"), True):
            with self.assertRaises(ValueError):
                workflow.normalize_config({"group_timeout_seconds": timeout})
        self.assertEqual(workflow.normalize_config({"variants": ["baseline"], "localize_modes": []})["localize_modes"], [])
        with self.assertRaises(ValueError):
            workflow.normalize_config({"variants": ["baseline"], "localize_modes": ["cache_dit"]})

    def test_prefix_loop_does_not_replace_scheduler_steps(self):
        # This prevents the previous experiment's 4-step-grid vs production
        # 20-step-grid mismatch: preparation receives unmodified spec.
        source = inspect.getsource(workflow._prefix_run)
        tree = ast.parse(source)
        self.assertIn("range(prefix)", source)
        self.assertIn('spec["steps"] != 20', source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                self.assertFalse(any(isinstance(target, ast.Subscript) and
                                     ast.unparse(target) in ("spec['steps']", 'spec["steps"]') for target in targets))

    def test_causal_variants_are_opt_in_and_keep_original_default_matrix(self):
        self.assertEqual(workflow.normalize_config({})["variants"], list(workflow.DEFAULT_VARIANTS))
        modes = ["cache_dit", "full_rows_linear", "no_reduced_precision", "full_rows_ffn2",
                 "full_rows_ffn2_qkv", "full_rows_except_ffn1", "global_tile_1024", "global_tile_2048"]
        self.assertEqual(workflow.normalize_config(dict(variants=modes))["variants"], modes)

    def test_video_retention_requires_explicit_boolean_opt_in(self):
        self.assertFalse(workflow.normalize_config({})["save_videos"])
        self.assertTrue(workflow.normalize_config({"save_videos": True})["save_videos"])
        for invalid in (1, "false", None):
            with self.assertRaisesRegex(ValueError, "save_videos"):
                workflow.normalize_config({"save_videos": invalid})


class ManifestTests(unittest.TestCase):
    def test_diagnostic_success_cannot_qualify_unmodified_deployment(self):
        for mode in ("full_rows_linear", "no_reduced_precision", "full_rows_ffn2",
                     "full_rows_ffn2_qkv", "full_rows_except_ffn1", "global_tile_1024", "global_tile_2048"):
            ranks = [dict(status="completed", variants=[dict(mode=mode, seed=3501, frames=17,
                         status="qualified")], qualification=[dict(sp1_exact=True)]) for _ in range(2)]
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp, \
                    patch.object(workflow, "_run_ranks", return_value=ranks):
                result = workflow.run_sp(dict(variants=[mode], seeds=[3501], frames=[17]), temp)
                self.assertTrue(result["qualification"]["numerical_variants_passed"])
                self.assertTrue(result["qualification"]["passed"])
                self.assertEqual(result["qualification"]["scope"], "diagnostic_numerical_only")
                self.assertEqual(result["role"], "diagnostic_only")
                self.assertFalse(result["deployment_qualification"]["passed"])
                self.assertFalse(result["performance_evidence"])

    def test_numerical_failure_is_result_and_not_performance(self):
        ranks = [dict(status="qualification_failed", variants=[{"mode": "baseline", "status": "numerical_failed"}],
                      qualification=[{"sp1_exact": True}]) for _ in range(2)]
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_run_ranks", return_value=ranks):
            result = workflow.run_sp({"frames": [17]}, temp)
            stored = json.loads((Path(temp) / "run.json").read_text())
            self.assertEqual(stored["status"], "qualification_failed")
            self.assertFalse(result["qualification"]["passed"])
            self.assertFalse(result["performance_evidence"])
            self.assertEqual(result["threshold"], 0.05)
            self.assertEqual(result["scheduler_steps"], 20)

    def test_timeout_keeps_manifest_and_partial_rank_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "rank0"
            directory.mkdir()
            (directory / "run.json").write_text('{"status":"running"}')
            with patch.object(workflow, "_run_ranks", side_effect=TimeoutError("budget")):
                result = workflow.run_sp({}, temp)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["rank_manifests"], [str(directory / "run.json")])
            self.assertIn("budget", result["error"])
            self.assertIn("finished_unix_ns", result)


class PrefixExecutionTests(unittest.TestCase):
    """Exercise the actual diagnostic loop, files and hook cleanup using NumPy."""

    def setUp(self):
        import numpy as np
        self.np = np
        self.prepared_steps, self.released = [], []
        self.hooks = []
        self.video_scope = workflow._reference_videos()
        self.video_cache = self.video_scope.__enter__()
        self.addCleanup(self.video_scope.__exit__, None, None, None)

        class Tensor:
            def __init__(self, data):
                self.data = np.asarray(data, dtype=np.float32)

            def detach(self):
                return self

            cpu = float = detach

            def numpy(self):
                return self.data

            def tolist(self):
                return self.data.tolist()

        self.Tensor = Tensor
        def register(hook):
            self.hooks.append(hook)
            return types.SimpleNamespace(remove=lambda: self.hooks.remove(hook))

        def prepare(spec, payload):
            self.prepared_steps.append(spec["steps"])
            return types.SimpleNamespace(spec=spec, step_index=0, latents=Tensor([[0., 1.]]),
                                         scheduler=types.SimpleNamespace(timesteps=Tensor(np.arange(spec["steps"])[::-1])))

        def step(state):
            for offset in (1., 2.):
                prediction = Tensor(state.latents.data + offset)
                for hook in self.hooks:
                    hook(None, None, (prediction,))
            state.latents = Tensor(state.latents.data + 1.)
            state.step_index += 1

        self.worker = types.SimpleNamespace(prepare_encoded=prepare, step=step,
                                           release=lambda state: self.released.append(state),
                                           pipe=types.SimpleNamespace(transformer=types.SimpleNamespace(
                                               register_forward_hook=register)))
        self.timeline = types.SimpleNamespace(phase=lambda *args, **kwargs: contextlib.nullcontext())
        self.spec = dict(steps=20, frames=17, guidance_scale=5.0)

        @dataclasses.dataclass
        class Payload:
            spec: dict
            consumed: bool = False
        self.payload = Payload(self.spec)

    def run_prefix(self, path, prefix, **kwargs):
        kwargs.setdefault("video_cache", self.video_cache)
        with patch.object(workflow, "_decode_gpu", lambda worker, state: state.latents):
            return workflow._prefix_run(self.worker, self.spec, self.payload, prefix, path,
                                        self.timeline, **kwargs)

    def test_short_prefixes_execute_full_grid_and_exact_self_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            for prefix in (1, 4):
                reference = Path(temp) / str(prefix) / "reference"
                first = self.run_prefix(reference, prefix)
                replay = self.run_prefix(Path(temp) / str(prefix) / "replay", prefix,
                                         reference=reference, exact=True)
                self.assertTrue(replay["passed"])
                self.assertEqual(len(replay["steps"]), prefix)
                self.assertEqual(len(first["timestep_grid"]), 20)
                self.assertEqual(first["timestep_grid"][:prefix], [19., 18., 17., 16.][:prefix])
                self.assertFalse(list(Path(temp).rglob("*video*.npy")))
        self.assertEqual(self.prepared_steps, [20, 20, 20, 20])
        self.assertEqual(len(self.released), 4)
        self.assertFalse(self.hooks)

    def test_first_divergence_saves_previous_and_current_values_and_fails_gate(self):
        ordinary = self.worker.step
        def altered(state):
            ordinary(state)
            if state.step_index == 2:
                state.latents = self.Tensor(state.latents.data + 0.2)
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp) / "reference"
            self.run_prefix(reference, 4)
            candidate = self.run_prefix(Path(temp) / "candidate", 4, reference=reference,
                                         backend=types.SimpleNamespace(step=altered), mode="baseline")
            self.assertFalse(candidate["passed"])
            self.assertGreater(candidate["video_comparison"]["max_abs"], .05)
            self.assertNotIn("failed_video", candidate["artifacts"])
            self.assertFalse(list(Path(temp).rglob("*video*.npy")))
            self.assertEqual(candidate["first_nonzero_step"], 1)
            self.assertEqual(candidate["first_threshold_failure_step"], 1)
            artifacts = candidate["artifacts"]["first_threshold_failure"]
            self.assertEqual(len(artifacts), 2)
            with self.np.load(artifacts[1], allow_pickle=False) as arrays:
                self.assertIn("candidate_cond", arrays.files)
                delta = arrays["candidate_latent_after"] - arrays["reference_latent_after"]
                self.assertAlmostEqual(float(delta.max()), 0.2, places=6)
                self.assertTrue(self.np.array_equal(arrays["candidate_latent_before"], arrays["reference_latent_before"]))
        self.assertFalse(self.hooks)

    def test_hook_and_state_are_released_when_model_step_fails(self):
        def broken(state):
            raise RuntimeError("bad collective")
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(RuntimeError, "bad collective"):
            self.run_prefix(temp, 1, backend=types.SimpleNamespace(step=broken))
        self.assertFalse(self.hooks)
        self.assertEqual(len(self.released), 1)

    def test_candidate_sp1_bridge_arrays_and_original_reference_gate_remain_independent(self):
        ordinary = self.worker.step
        for drift in (0.01, 0.2):
            def altered(state):
                ordinary(state)
                state.latents = self.Tensor(state.latents.data + drift)
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                self.run_prefix(base / "original", 1)
                bridge = self.run_prefix(base / "tile-sp1", 1, reference=base / "original",
                    backend=types.SimpleNamespace(step=altered), save_arrays=True)
                self.assertTrue((base / "tile-sp1/step-00.npz").is_file())
                self.assertFalse((base / "tile-sp1/video.npy").exists())
                repeat = self.run_prefix(base / "tile-replay", 1, reference=base / "tile-sp1",
                    backend=types.SimpleNamespace(step=altered), exact=True)
                self.assertTrue(repeat["passed"])
                candidate = self.run_prefix(base / "tile-sp2", 1, reference=base / "original",
                    backend=types.SimpleNamespace(step=altered), secondary_reference=base / "tile-sp1")
                self.assertEqual(bridge["passed"], drift <= .05)
                self.assertEqual(candidate["passed"], drift <= .05)
                self.assertTrue(candidate["comparison_to_candidate_sp1"]["passed"])
                self.assertTrue(candidate["comparison_to_candidate_sp1"]["video_comparison"]["exact"])
                self.assertTrue(candidate["steps"][0]["candidate_sp1_comparison"]["latent_after"]["exact"])
                self.assertFalse(candidate["video_comparison"]["exact"])
                self.assertFalse(list(base.rglob("*video*.npy")))

    def test_video_comparison_keeps_full_errors_for_passing_and_failing_drift(self):
        for drift in (.01, .2):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                self.run_prefix(base / "reference", 1)
                with patch.object(workflow, "_decode_gpu", lambda worker, state:
                                  self.Tensor(state.latents.data + drift)):
                    candidate = workflow._prefix_run(self.worker, self.spec, self.payload, 1,
                        base / "candidate", self.timeline, reference=base / "reference",
                        video_cache=self.video_cache)
                comparison = candidate["video_comparison"]
                self.assertFalse(comparison["exact"])
                self.assertAlmostEqual(comparison["max_abs"], drift, places=6)
                self.assertAlmostEqual(comparison["rmse"], drift, places=6)
                self.assertIn("cosine", comparison)
                self.assertEqual(candidate["passed"], drift <= .05)
                self.assertTrue(candidate["steps"][0]["comparison"]["latent_after"]["exact"])
                self.assertFalse(list(base.rglob("*video*.npy")))

    def test_reference_cache_cleans_on_success_and_decode_failure_without_video_files(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                try:
                    with workflow._reference_videos() as cache:
                        self.run_prefix(base / "reference", 1, video_cache=cache)
                        self.assertEqual(len(cache), 1)
                        if fail:
                            with patch.object(workflow, "_decode_gpu", side_effect=RuntimeError("decode failed")):
                                workflow._prefix_run(self.worker, self.spec, self.payload, 1,
                                    base / "candidate", self.timeline, reference=base / "reference",
                                    video_cache=cache)
                        else:
                            self.run_prefix(base / "candidate", 1, reference=base / "reference", video_cache=cache)
                except RuntimeError as error:
                    self.assertTrue(fail)
                    self.assertEqual(str(error), "decode failed")
                self.assertFalse(cache)
                self.assertFalse(self.hooks)
                self.assertFalse(list(base.rglob("*video*.npy")))
        self.assertEqual(len(self.released), 4)

    def test_explicit_save_videos_keeps_reference_and_failed_video(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            self.run_prefix(base / "reference", 1, video_cache=None, save_videos=True)
            with patch.object(workflow, "_decode_gpu", lambda worker, state:
                              self.Tensor(state.latents.data + .2)):
                candidate = workflow._prefix_run(self.worker, self.spec, self.payload, 1,
                    base / "candidate", self.timeline, reference=base / "reference", save_videos=True)
            self.assertFalse(candidate["passed"])
            self.assertEqual(candidate["artifacts"]["failed_video"], str(base / "candidate/video.npy"))
            self.assertTrue((base / "reference/video.npy").is_file())
            self.assertTrue((base / "candidate/video.npy").is_file())


class LayerLocalizationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np

        class Tensor:
            def __init__(self, data):
                self.data = np.asarray(data, dtype=np.float16)

            @property
            def ndim(self):
                return self.data.ndim

            @property
            def shape(self):
                return self.data.shape

            def detach(self):
                return self

            cpu = detach

            def numpy(self):
                return self.data

            def chunk(self, count, dim):
                return [Tensor(part) for part in np.split(self.data, count, axis=dim)]

        self.Tensor = Tensor

        class Module:
            def __init__(self, offset=0.):
                self.hooks, self.prehooks, self.offset = [], [], offset

            def register_forward_hook(self, hook):
                self.hooks.append(hook)
                return types.SimpleNamespace(remove=lambda: self.hooks.remove(hook))

            def register_forward_pre_hook(self, hook, with_kwargs, prepend=False):
                self.prehooks.insert(0, hook) if prepend else self.prehooks.append(hook)
                return types.SimpleNamespace(remove=lambda: self.prehooks.remove(hook))

            def __call__(self, value):
                for hook in self.prehooks:
                    changed = hook(self, (value,), {})
                    if changed is not None:
                        value = changed[0][0]
                result = Tensor(value.data + self.offset)
                for hook in self.hooks:
                    changed = hook(self, (value,), result)
                    if changed is not None:
                        result = changed
                return result

        class Activation(Module):
            def __init__(self):
                super().__init__()
                self.proj = Module()

            def __call__(self, value):
                return super().__call__(self.proj(value))

        class FeedForward(Module):
            def __init__(self):
                super().__init__()
                self.net = [Activation(), Module(), Module()]

            def __call__(self, value):
                for hook in self.prehooks:
                    hook(self, (value,), {})
                result = value
                for module in self.net:
                    result = module(result)
                for hook in self.hooks:
                    hook(self, (value,), result)
                return result

        class Block(Module):
            def __init__(self, injected=0.):
                super().__init__()
                self.attn1 = types.SimpleNamespace(to_q=Module(injected), to_k=Module(), to_v=Module(),
                                                   norm_q=Module(), norm_k=Module(), to_out=[Module()])
                self.ffn = FeedForward()

            def __call__(self, value):
                for hook in self.prehooks:
                    hook(self, (), dict(hidden_states=value))
                q = self.attn1.norm_q(self.attn1.to_q(value))
                k = self.attn1.norm_k(self.attn1.to_k(value))
                v = self.attn1.to_v(value)
                result = self.attn1.to_out[0](Tensor((q.data + k.data + v.data) / 3))
                result = self.ffn(result)
                for hook in self.hooks:
                    hook(self, (), result)
                return result

        self.Block = Block
        self.spec = dict(steps=20, guidance_scale=5.)

        @dataclasses.dataclass
        class Payload:
            spec: dict
            consumed: bool = False
        self.payload = Payload(self.spec)

    def worker(self, *, sharded=False, inject=False):
        np, Tensor = self.np, self.Tensor
        blocks = [self.Block(), self.Block(0.75 if inject else 0.)]
        full = np.arange(24, dtype=np.float16).reshape(1, 8, 3)
        value = full[:, 4:, :] if sharded else full
        def prepare(spec, payload):
            return types.SimpleNamespace(latents=Tensor(value))
        def step(state):
            for branch in (0., 1.):
                current = Tensor(state.latents.data + branch)
                for block in blocks:
                    current = block(current)
        worker = types.SimpleNamespace(prepare_encoded=prepare, step=step, release=lambda *_: None,
                                       pipe=types.SimpleNamespace(transformer=types.SimpleNamespace(blocks=blocks)))
        return worker

    def assert_no_hooks(self, worker):
        for block in worker.pipe.transformer.blocks:
            modules = [block] + [getattr(block.attn1, name) for name in
                                 ("to_q", "to_k", "to_v", "norm_q", "norm_k")] + block.attn1.to_out + [
                                 block.ffn, *block.ffn.net, block.ffn.net[0].proj]
            self.assertTrue(all(not module.hooks and not module.prehooks for module in modules))

    def test_sp1_shard_alignment_and_first_block_tensor(self):
        reference, candidate = self.worker(), self.worker(sharded=True, inject=True)
        with tempfile.TemporaryDirectory() as temp:
            base, output = Path(temp) / "reference", Path(temp) / "candidate"
            workflow._capture_layers(reference, self.spec, self.payload, 0, base, 1)
            result = workflow._capture_layers(candidate, self.spec, self.payload, 0, output, 1,
                                               backend=candidate, reference=base)
            self.assertEqual(len(result["layers"]), 4)
            self.assertTrue(result["layers"][0]["comparison"]["exact"])
            self.assertEqual(result["first_divergence"]["block"], 1)
            self.assertEqual(result["first_divergence"]["branch"], "cond")
            self.assertEqual(len(list(output.glob("*.npy"))), 1)
            stored = self.np.load(result["first_divergence"]["reference"], allow_pickle=False)
            self.assertEqual(stored.shape, (1, 4, 3))
            self.assertEqual(stored.dtype, self.np.float16)
        self.assert_no_hooks(reference)
        self.assert_no_hooks(candidate)

    def test_detailed_replay_localizes_q_projection_and_keeps_cfg_tensors(self):
        reference, candidate = self.worker(), self.worker(sharded=True, inject=True)
        with tempfile.TemporaryDirectory() as temp:
            base, output = Path(temp) / "reference", Path(temp) / "candidate"
            workflow._capture_layers(reference, self.spec, self.payload, 0, base, 1, target_block=1)
            result = workflow._capture_layers(candidate, self.spec, self.payload, 0, output, 1,
                                               backend=candidate, reference=base, target_block=1)
            self.assertEqual(result["first_divergence"]["tensor"], "to_q")
            self.assertEqual(len(result["layers"]), 16)
            self.assertEqual(len(list(output.glob("*.npy"))), 16)
            self.assertTrue(result["layers"][0]["comparison"]["exact"])
        self.assert_no_hooks(reference)
        self.assert_no_hooks(candidate)

    def test_failed_layer_capture_removes_all_registered_hooks(self):
        worker = self.worker()
        def broken(state):
            raise RuntimeError("diagnostic failure")
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
            workflow._capture_layers(worker, self.spec, self.payload, 0, temp, 1,
                                      backend=types.SimpleNamespace(step=broken), target_block=1)
        self.assert_no_hooks(worker)

    def test_extended_capture_compares_pre_projection_attention_and_removes_hooks(self):
        reference, candidate = self.worker(), self.worker(sharded=True)
        with tempfile.TemporaryDirectory() as temp:
            base, output = Path(temp) / "reference", Path(temp) / "candidate"
            workflow._capture_layers(reference, self.spec, self.payload, 0, base, 1,
                                     target_block=1, extended=True)
            result = workflow._capture_layers(candidate, self.spec, self.payload, 0, output, 1,
                                             backend=candidate, reference=base,
                                             target_block=1, extended=True)
            self.assertTrue(result["all_exact"])
            self.assertIn("attention_result", result["expected_tensors"])
            for name in ("to_q_input", "feed_forward_input", "feed_forward_first_projection",
                         "feed_forward_activation", "feed_forward_second_projection"):
                self.assertIn(name, result["expected_tensors"])
            self.assertEqual(len(result["layers"]), 30)
        self.assert_no_hooks(reference)
        self.assert_no_hooks(candidate)

    def test_extended_input_capture_precedes_padding_and_projection_capture_follows_crop(self):
        reference, candidate = self.worker(), self.worker(sharded=True)
        attention = candidate.pipe.transformer.blocks[1].attn1
        def pad(module, args, kwargs):
            array = args[0].data
            return ((self.Tensor(self.np.concatenate([self.np.zeros_like(array), array], axis=1)),), kwargs)
        def crop(module, args, output):
            return self.Tensor(output.data[:, 4:, :])
        interventions = []
        for projection in (attention.to_q, attention.to_out[0]):
            interventions.append(projection.register_forward_pre_hook(pad, with_kwargs=True))
            interventions.append(projection.register_forward_hook(crop))
        try:
            with tempfile.TemporaryDirectory() as temp:
                base, output = Path(temp) / "reference", Path(temp) / "candidate"
                workflow._capture_layers(reference, self.spec, self.payload, 0, base, 1,
                                         target_block=1, extended=True)
                result = workflow._capture_layers(candidate, self.spec, self.payload, 0, output, 1,
                                                 backend=candidate, reference=base,
                                                 target_block=1, extended=True)
                self.assertTrue(result["all_exact"])
                for layer in result["layers"]:
                    if layer["tensor"] in ("to_q_input", "to_q", "attention_result",
                                           "attention_output_projection"):
                        self.assertEqual(layer["statistics"]["shape"], [1, 4, 3])
        finally:
            for intervention in interventions:
                intervention.remove()
        self.assert_no_hooks(reference)
        self.assert_no_hooks(candidate)


if __name__ == "__main__":
    unittest.main()
