"""Exercise the combined adapter with CPU fakes, without torch/model imports."""

import unittest
from unittest.mock import patch

import test_optimizations_next_sana as fakes
from ditserve.optimizations import followup_sana, next_sana
from ditserve.optimizations.followup_sana import FollowupSanaAdapter
from ditserve.optimizations.sana import OptimizedSanaAdapter


def adapter(**kwargs):
    obj = fakes.adapter(**kwargs)
    obj.__class__ = FollowupSanaAdapter
    return obj


class FollowupSanaTests(unittest.TestCase):
    def setUp(self):
        def inductor(original, inputs, **kwargs):
            return original
        backend = patch.object(next_sana, "_inductor_backend", return_value=inductor)
        parent = patch.object(OptimizedSanaAdapter, "generate_batch", fakes.parent_generate)
        backend.start()
        parent.start()
        self.addCleanup(backend.stop)
        self.addCleanup(parent.stop)

    def test_combination_executes_both_paths_and_preserves_output_and_recipe(self):
        obj = adapter(extra=True, post_quant=True)
        vae, transformer = obj.pipe.vae, obj.pipe.transformer
        vae.use_tiling = True
        reference, _ = obj.generate("abc", seed=201)
        summary = obj.set_candidate("vae_graph")
        self.assertFalse(summary["qualified"])
        actual, service = obj.generate("abc", seed=201)
        self.assertEqual(actual.values, reference.values)
        self.assertIs(obj.pipe.vae, vae)
        self.assertIs(obj.pipe.transformer, transformer)
        self.assertTrue(vae.use_tiling)
        self.assertEqual(service["candidate"], "vae_graph")
        self.assertTrue(service["candidate_summary"]["combination"])
        self.assertEqual([region["graph_invocations"] for region in service["candidate_summary"]["vae_regions"]], [1, 1])
        self.assertEqual(service["candidate_summary"]["transformer_graph"]["graph_replays"], 2)
        self.assertEqual(service["candidate_summary"]["transformer_graph"]["capture_count"], 1)
        self.assertTrue(all(item["fullgraph"] and not item["dynamic"] for item in obj.torch.compile_options))

    def test_legacy_backend_notification_preserves_combination(self):
        obj = adapter()
        obj.set_candidate("vae_graph")
        decoder, transformer = obj.pipe.vae.decoder.forward, obj.pipe.transformer.forward
        obj.set_backend("eager")
        self.assertIs(obj.pipe.vae.decoder.forward, decoder)
        self.assertIs(obj.pipe.transformer.forward, transformer)
        self.assertEqual(obj.candidate, "vae_graph")

    def test_switch_to_compile_baseline_frees_graph_then_recaptures_combination(self):
        obj = adapter()
        original_transformer = obj.pipe.transformer.forward
        obj.set_candidate("vae_graph")
        obj.generate("a", seed=101)
        compiled_decoder = obj.pipe.vae.decoder.forward
        old_graph = obj.torch.cuda.graphs[0]
        obj.set_candidate("vae_compile")
        self.assertEqual(obj.pipe.transformer.forward, original_transformer)
        self.assertIs(obj.pipe.vae.decoder.forward, compiled_decoder)
        self.assertEqual(old_graph.resets, 1)
        self.assertIsNone(obj._transformer_graph.static_inputs)
        self.assertIsNone(obj._transformer_graph.static_output)
        obj.generate("b", seed=102)
        self.assertEqual(obj._transformer_graph.replays, 2)
        obj.set_candidate("vae_graph")
        obj.generate("c", seed=201)
        self.assertEqual(obj._transformer_graph.capture_count, 2)
        self.assertEqual(len(obj._vae_regions["decoder"].records), 1)

    def test_graph_only_switch_removes_vae_hook_and_keeps_valid_capture(self):
        obj = adapter()
        original_decoder = obj.pipe.vae.decoder.forward
        obj.set_candidate("vae_graph")
        obj.generate("a", seed=101)
        graph = obj.torch.cuda.graphs[0]
        obj.set_candidate("transformer_graph")
        self.assertEqual(obj.pipe.vae.decoder.forward, original_decoder)
        with obj.steady_context():
            obj.generate("b", seed=102)
        self.assertEqual(graph.resets, 0)
        self.assertEqual(obj._transformer_graph.capture_count, 1)

    def test_eager_switch_restores_all_hooks_and_releases_graph_storage(self):
        obj = adapter(post_quant=True)
        originals = [obj.pipe.transformer.forward, obj.pipe.vae.decoder.forward, obj.pipe.vae.post_quant_conv.forward]
        obj.set_candidate("vae_graph")
        obj.generate("a", seed=101)
        obj.set_candidate("eager")
        self.assertEqual([obj.pipe.transformer.forward, obj.pipe.vae.decoder.forward, obj.pipe.vae.post_quant_conv.forward], originals)
        self.assertIsNone(obj._transformer_graph.graph)
        self.assertFalse(obj._next_originals)

    def test_steady_requires_warmup_and_checks_recompile_for_combination(self):
        obj = adapter()
        obj.set_candidate("vae_graph")
        with obj.steady_context(), self.assertRaisesRegex(RuntimeError, "not captured"):
            obj.generate("a", seed=101)
        self.assertEqual(obj._steady_depth, 0)
        obj.set_candidate("vae_graph")
        obj.generate("a", seed=101)
        with obj.steady_context():
            obj.generate("b", seed=102)
        self.assertIn("fail_on_recompile", obj.torch.stances)
        obj.torch.recompile = True
        with obj.steady_context(), self.assertRaisesRegex(RuntimeError, "compiled during steady"):
            obj.generate("c", seed=201)
        self.assertEqual(obj._steady_depth, 0)
        self.assertEqual(obj.candidate_summary()["status"], "failed")
        self.assertIsNone(obj._transformer_graph.graph)

    def test_stance_creation_error_restores_steady_depth(self):
        obj = adapter()
        obj.set_candidate("vae_graph")
        with patch.object(obj.torch.compiler, "set_stance", side_effect=RuntimeError("stance failed")):
            with self.assertRaisesRegex(RuntimeError, "stance failed"), obj.steady_context():
                self.fail("stance failure must not enter measurement")
        self.assertEqual(obj._steady_depth, 0)

    def test_warm_graph_does_not_hide_first_vae_compile_in_steady_measurement(self):
        obj = adapter()
        obj.set_candidate("transformer_graph")
        obj.generate("a", seed=101)
        obj.set_candidate("vae_graph")
        with obj.steady_context(), self.assertRaisesRegex(RuntimeError, "compiled during steady"):
            obj.generate("b", seed=102)
        self.assertEqual(obj._steady_depth, 0)
        self.assertFalse(obj._vae_regions)
        self.assertIsNone(obj._transformer_graph.graph)

    def test_compile_or_capture_failure_restores_both_and_requires_explicit_retry(self):
        for reason in ("compile", "capture", "silent"):
            obj = adapter(extra=True)
            originals = (obj.pipe.transformer.forward, obj.pipe.vae.decoder.forward,
                         obj.pipe.image_processor.postprocess)
            obj.set_candidate("vae_graph")
            obj.torch.fail_compile = reason == "compile"
            obj.torch.silent_compile = reason == "silent"
            obj.torch.cuda.fail_capture = reason == "capture"
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                obj.generate("a", seed=101)
            self.assertEqual((obj.pipe.transformer.forward, obj.pipe.vae.decoder.forward,
                              obj.pipe.image_processor.postprocess), originals)
            self.assertIsNone(obj._transformer_graph.static_output)
            self.assertIsNone(obj._transformer_graph.graph)
            self.assertFalse(obj._vae_regions)
            self.assertFalse(obj._next_lock.locked())
            self.assertEqual(obj.candidate_summary()["failures"][0]["candidate"], "vae_graph")
            with self.assertRaisesRegex(RuntimeError, "explicit set_candidate"):
                obj.generate("a", seed=101)
            obj.set_candidate("eager")
            obj.generate("a", seed=101)

    def test_partial_install_failure_unwinds_compiled_hook(self):
        obj = adapter()
        original = obj.pipe.vae.decoder.forward
        with patch.object(followup_sana, "_TransformerGraph", side_effect=RuntimeError("allocation failed")):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                obj.set_candidate("vae_graph")
        self.assertEqual(obj.pipe.vae.decoder.forward, original)
        self.assertFalse(obj._next_originals)
        self.assertFalse(obj._vae_regions)
        self.assertEqual(obj.candidate_summary()["status"], "failed")

    def test_cancel_after_graph_replay_frees_private_buffers_without_rng_rollback(self):
        obj = adapter()
        original = obj.pipe.transformer.forward
        obj.set_candidate("vae_graph")
        def cancel():
            if obj._transformer_graph.replays:
                raise RuntimeError("budget interrupted")
        with self.assertRaisesRegex(RuntimeError, "budget interrupted"):
            obj.generate("a", seed=101, check_budget=cancel)
        self.assertEqual(obj.pipe.transformer.forward, original)
        self.assertIsNone(obj._transformer_graph.static_inputs)
        self.assertIsNone(obj._transformer_graph.static_output)
        self.assertFalse(obj._vae_regions)

    def test_switch_teardown_failure_blocks_reuse_and_keeps_evidence(self):
        obj = adapter()
        original = obj.pipe.transformer.forward
        obj.set_candidate("vae_graph")
        obj.generate("a", seed=101)
        graph = obj.torch.cuda.graphs[0]
        graph.reset = lambda: (_ for _ in ()).throw(RuntimeError("teardown failed"))
        with self.assertRaisesRegex(RuntimeError, "teardown failed"):
            obj.set_candidate("vae_compile")
        self.assertEqual(obj.pipe.transformer.forward, original)
        self.assertTrue(obj._cleanup_failed)
        self.assertFalse(obj._vae_regions)
        self.assertEqual(obj.candidate_summary()["failures"][-1]["phase"], "candidate_graph_teardown")
        with self.assertRaisesRegex(RuntimeError, "fresh process"):
            obj.set_candidate("eager")

    def test_close_releases_both_candidates_and_restores_external_module_references(self):
        obj = adapter()
        transformer, vae = obj.pipe.transformer, obj.pipe.vae
        originals = (transformer.forward, vae.decoder.forward)
        obj.set_candidate("vae_graph")
        obj.generate("a", seed=101)
        graph = obj.torch.cuda.graphs[0]
        obj.close()
        self.assertEqual((transformer.forward, vae.decoder.forward), originals)
        self.assertEqual(graph.resets, 1)
        self.assertIsNone(obj.pipe)
        self.assertIsNone(obj._transformer_graph)
        self.assertFalse(obj._vae_regions)

    def test_unknown_or_busy_selection_does_not_change_current_hooks(self):
        obj = adapter()
        obj.set_candidate("vae_graph")
        obj.generate("a", seed=101)
        hooks = (obj.pipe.transformer.forward, obj.pipe.vae.decoder.forward)
        with self.assertRaises(ValueError):
            obj.set_candidate("invalid")
        obj._next_lock.acquire()
        try:
            with self.assertRaises(RuntimeError):
                obj.set_candidate("eager")
        finally:
            obj._next_lock.release()
        self.assertEqual((obj.pipe.transformer.forward, obj.pipe.vae.decoder.forward), hooks)
        self.assertTrue(obj._transformer_graph.summary()["captured"])


if __name__ == "__main__":
    unittest.main()
