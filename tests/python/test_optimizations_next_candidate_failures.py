"""CPU fakes exercise real candidate wrappers and fail-closed phase handling."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_optimizations_next_sana as fakes
from ditserve.optimizations import next_candidate_failures as failures
from ditserve.optimizations import next_phases as phases
from ditserve.optimizations.budget import BudgetStop
from ditserve.optimizations.next_chain import _success
from ditserve.optimizations.next_sana import NextSanaAdapter
from ditserve.optimizations.sana import OptimizedSanaAdapter
from ditserve.research.common import read_json


Unsupported = type("Unsupported", (RuntimeError,), {"__module__": "torch._dynamo.exc"})
RecompileError = type("RecompileError", (RuntimeError,), {"__module__": "torch._dynamo.exc"})
BackendCompilerFailed = type("BackendCompilerFailed", (RuntimeError,), {"__module__": "torch._dynamo.exc"})
OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch"})


class CandidateFailures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.adapter = fakes.adapter()
        self.root = Path(self.tmp.name)
        self.context = SimpleNamespace(output=self.root, run_root=self.root, sana=self.adapter,
            opts=dict(prompts=["one", "two"]), config=dict(seeds=[101,102,201,202], quality={}),
            check_budget=lambda: None)
        self.parent = patch.object(OptimizedSanaAdapter, "generate_batch", fakes.parent_generate)
        self.trigger = patch.object(phases, "graph_trigger", return_value=dict(triggered=True))
        self.parent.start()
        self.trigger.start()
        self.addCleanup(self.parent.stop)
        self.addCleanup(self.trigger.stop)

    def compile_error(self, error):
        def compile(original, **options):
            def invoke(*args, **kwargs):
                raise error
            return invoke
        self.adapter.torch.compile = compile

    def run_phase(self, candidate="vae_compile"):
        with patch.object(phases, "_candidate_online") as online:
            result = phases.run_candidate(self.context, candidate)
        online.assert_not_called()
        return result

    def assert_stopped(self, result, classification):
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["decision"], "未验证")
        self.assertFalse(result["local_qualified"])
        self.assertFalse(result["online_qualified"])
        self.assertFalse(result["online_evaluation_complete"])
        failure = result["expected_failure"]
        self.assertEqual(failure["classification"], classification)
        self.assertTrue(failure["cleanup"]["passed"])
        self.assertTrue(failure["cleanup"]["graph_state_released"])
        self.assertEqual(failure["failed_unit"]["arm"], result["candidate"])
        self.assertEqual(len(result["rows"]), 2)  # Both eager warmups retained.
        self.assertIn("Traceback", result["traceback"])
        self.assertTrue(Path(failure["operation"]["source_path"]).exists())
        self.assertEqual(len(failure["operation"]["source_sha256"]), 64)
        self.assertEqual(read_json(self.root / "sana-candidate.json"), result)
        self.assertEqual(self.adapter.candidate, "eager")
        self.assertFalse(self.adapter._next_originals)
        self.assertTrue(_success(dict(result, returncode=0, charge=dict(status="completed"))))

    def test_unsupported_compile_concludes_and_preserves_failed_invocation(self):
        self.compile_error(Unsupported("operator not supported by fullgraph"))
        result = self.run_phase()
        self.assert_stopped(result, "compiler_unsupported")
        self.assertTrue(result["mechanics"]["failures"])
        self.assertTrue(result["mechanics"]["failures"][0]["vae_regions"][0]["execution_failures"])

    def test_backend_wrapper_requires_recognized_inner_exception(self):
        error = BackendCompilerFailed("backend raised unsupported")
        error.inner_exception = Unsupported("unsupported operator")
        self.compile_error(error)
        self.assert_stopped(self.run_phase(), "compiler_unsupported")

    def test_silent_eager_fallback_is_negative_not_speed_evidence(self):
        self.adapter.torch.silent_compile = True
        self.assert_stopped(self.run_phase(), "compiled_graph_not_executed")

    def test_canonical_capture_unsupported_releases_graph_then_advances(self):
        @contextmanager
        def graph(*args, **kwargs):
            raise RuntimeError("CUDA error: operation not permitted when stream is capturing")
            yield
        self.adapter.torch.cuda.graph = graph
        result = self.run_phase("transformer_graph")
        self.assert_stopped(result, "cuda_capture_unsupported")
        summary = self.adapter._transformer_graph.summary()
        self.assertFalse(summary["captured"])
        self.assertGreaterEqual(result["mechanics"]["transformer_graph"]["records"][0]["cold_host_ms"], 0.)

    def test_empty_capture_runtime_error_is_preserved_as_unknown(self):
        @contextmanager
        def graph(*args, **kwargs):
            raise RuntimeError("")
            yield
        self.adapter.torch.cuda.graph = graph
        with self.assertRaises(RuntimeError) as raised:
            self.run_phase("transformer_graph")
        self.assertEqual(str(raised.exception), "")
        self.assertNotIn("expected_failure", read_json(self.root / "sana-candidate.json"))

    def test_unknown_compile_error_remains_fatal(self):
        self.compile_error(RuntimeError("unexpected compiler bug"))
        with self.assertRaisesRegex(RuntimeError, "unexpected compiler bug"):
            self.run_phase()
        record = read_json(self.root / "sana-candidate.json")
        self.assertEqual(record["status"], "failed")
        self.assertNotIn("expected_failure", record)

    def test_unknown_backend_inner_error_remains_fatal(self):
        error = BackendCompilerFailed("backend raised arbitrary runtime error")
        error.inner_exception = RuntimeError("compiler internal invariant broken")
        self.compile_error(error)
        with self.assertRaises(BackendCompilerFailed):
            self.run_phase()

    def test_budget_cancel_and_value_errors_propagate(self):
        for error in (BudgetStop("stop"), KeyboardInterrupt(), SystemExit(2),
                      ValueError("bound source identity differs")):
            with self.subTest(kind=type(error).__name__):
                self.compile_error(error)
                with self.assertRaises(type(error)):
                    self.run_phase()
                self.assertNotIn("expected_failure", read_json(self.root / "sana-candidate.json"))

    def test_nested_oom_and_kernel_failures_cannot_be_wrapped_as_unsupported(self):
        for inner in (OutOfMemoryError("allocation exhausted"), RuntimeError("CUDA illegal memory access"),
                      ValueError("scientific source differs"), BudgetStop("parent stopped")):
            with self.subTest(kind=type(inner).__name__):
                error = Unsupported("unsupported operation")
                error.__cause__ = inner
                self.compile_error(error)
                with self.assertRaises(Unsupported):
                    self.run_phase()

    def test_cleanup_failure_is_persisted_and_fatal(self):
        self.compile_error(Unsupported("unsupported operator"))
        with patch.object(failures, "restore_and_verify", side_effect=RuntimeError("cleanup failed")):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                self.run_phase()
        record = read_json(self.root / "sana-candidate.json")
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["expected_failure"]["cleanup"]["passed"])
        self.assertIn("unsupported operator", record["reason"])

    def test_unsupported_from_outside_candidate_frame_is_not_converted(self):
        original = self.adapter.generate
        def generate(*args, **kwargs):
            if self.adapter.candidate != "eager":
                raise Unsupported("encoder failed before candidate operation")
            return original(*args, **kwargs)
        with patch.object(self.adapter, "generate", side_effect=generate):
            with self.assertRaises(Unsupported):
                self.run_phase()
        self.assertNotIn("expected_failure", read_json(self.root / "sana-candidate.json"))

    def test_recompile_allowed_only_in_selected_measurement(self):
        self.compile_error(RecompileError("recompile detected"))
        self.adapter.set_candidate("vae_compile")
        try:
            self.adapter.generate("one", seed=101)
        except RecompileError as error:
            summary = self.adapter.candidate_summary()
            warm = dict(kind="warmup", arm="vae_compile")
            measured = dict(kind="measurement", arm="vae_compile")
            self.assertIsNone(failures.classify(error, "vae_compile", warm, summary))
            self.assertEqual(failures.classify(error, "vae_compile", measured, summary)["classification"],
                             "measurement_recompile")
            for unit in (None, dict(kind="capacity", arm="vae_compile"),
                         dict(kind="measurement", arm="eager")):
                self.assertIsNone(failures.classify(error, "vae_compile", unit, summary))
        else:
            self.fail("fake compile did not fail")

    def test_restore_checks_actual_hook_and_graph_ownership(self):
        self.adapter.set_candidate("vae_compile")
        summary = failures.restore_and_verify(self.adapter)
        self.assertTrue(summary["passed"])
        self.adapter._cleanup_failed = True
        with self.assertRaisesRegex(RuntimeError, "teardown failed"):
            failures.restore_and_verify(self.adapter)


if __name__ == "__main__":
    unittest.main()
