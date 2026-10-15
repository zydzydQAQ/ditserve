"""Lightweight orchestration fixtures; no model or CUDA execution."""

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.optimizations import next_resume as module
from ditserve.directions.budget import BudgetStop


def complete_result(candidate="vae_compile"):
    paths = dict(valid=True, required_steps=32, observed_steps=32,
                 counts=dict(full=23, reuse=9), full_reuse_observed=True)
    return dict(passed=True, steps=[dict(step=i+1, latent=True, solver=True,
        cache_state=True, rng=True, passed=True, cache_paths_equal=True) for i in range(32)],
        interruptions=[dict(step=i, image_service=dict(candidate=candidate, finite=True, ms=1.))
                       for i in (0, 8, 16)],
        cache_paths=dict(valid=True, paths_equal=True, continuous=copy.deepcopy(paths), k4=copy.deepcopy(paths)),
        final=dict(passed=True, finite=True, exact=True, max_abs=0., rtol=1e-5, atol=1e-6,
                   shape=module.VIDEO_SHAPE),
        artifacts=[dict(path="/retained/"+name+".npy", sha256="a"*64,
                        raw_available=True, shape=module.VIDEO_SHAPE) for name in ("continuous", "k4")])


class Guard:
    def __init__(self):
        self.checks, self.units = 0, []
    def __call__(self):
        self.checks += 1
    def ensure_unit(self, seconds):
        self()
        self.units.append(seconds)


class Adapter:
    candidate = "vae_compile"
    def __init__(self):
        self.depth, self.generations = 0, []
    def candidate_summary(self):
        return dict(candidate=self.candidate, qualified=False)
    @contextmanager
    def steady_context(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
    def generate(self, prompt, *, seed, check_budget):
        assert self.depth == 1
        check_budget()
        self.generations.append((prompt, seed, self.candidate))
        return None, dict(candidate=self.candidate, finite=True, ms=1.)
    def set_candidate(self, unused):
        raise AssertionError("recovery must preserve the selected candidate")


class Bank:
    def __init__(self, worker):
        self.entries = {"private": "cpu-encoding"}
    def manifest(self):
        return [{"mode": "frozen_cpu_embeddings_mechanism_only"}]


class Executor:
    def __init__(self, worker, backend):
        self.worker, self.backend, self._states = worker, backend, {}
    def close(self, state):
        state.closed = True
        self._states.pop(id(state), None)


class CandidateResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.worker = SimpleNamespace(attention="flash", _states={})
        self.released = []
        def release(state):
            state.residency = "released"
            self.released.append(state)
            self.worker._states.pop(id(state), None)
        self.worker.release = release
        self.context = SimpleNamespace(output=root/"sana_vae/attempt00", run_root=root,
            config={}, metadata={}, worker=self.worker, sana=Adapter(), check_budget=Guard(),
            executor=object(), executors={"existing": object()})
        self.item = dict(input_id="kettle-101", seed=101, prompt="frozen kettle", split="dev")
        self.calls, self.banks, self.executors = [], [], []
        def bank(worker):
            value = Bank(worker)
            self.banks.append(value)
            return value
        def executor(worker, backend):
            value = Executor(worker, backend)
            self.executors.append(value)
            return value
        def resume(context, evidence, bank, item, executor, *, cache):
            self.calls.append((item, cache, executor.backend))
            result = complete_result(context.sana.candidate)
            for interruption in result["interruptions"]:
                _, interruption["image_service"] = context.sana.generate(item["prompt"], seed=item["seed"],
                    check_budget=context.check_budget)
            return result
        for name, value in dict(FrozenPromptBank=bank, PairedCacheExecutor=executor,
                BlockBackend=lambda worker, compiled: SimpleNamespace(compiled=compiled),
                _cache_inputs=lambda: [dict(self.item)], _resume_check=resume).items():
            p = patch.object(module.phases, name, value)
            p.start()
            self.addCleanup(p.stop)

    def saved(self):
        return json.loads((self.context.output / "sana-wan-resume.json").read_text())

    def test_fixed_candidate_uses_existing_strict_helper_and_retains_evidence(self):
        old_executor, old_executors = self.context.executor, dict(self.context.executors)
        result = module.run_candidate_resume(self.context)
        self.assertTrue(result["resume_qualified"])
        self.assertTrue(result["eligible"])
        self.assertEqual(result["gate"]["step_count"], 32)
        self.assertEqual(result["gate"]["real_sana_insertions"], 3)
        self.assertEqual(self.calls[0][0], self.item)
        self.assertTrue(self.calls[0][1])
        self.assertFalse(self.calls[0][2].compiled)
        self.assertEqual(self.context.sana.generations, [("frozen kettle", 101, "vae_compile")]*3)
        self.assertEqual(self.context.check_budget.units, [600.])
        self.assertGreaterEqual(self.context.check_budget.checks, 5)
        self.assertFalse(result["local_qualified"])
        self.assertFalse(result["online_qualified"])
        self.assertFalse(result["cache_quality_qualified"])
        self.assertFalse(result["retention_policy"]["may_delete_now"])
        self.assertEqual(len(result["retention"]), 2)
        self.assertEqual(result, self.saved())
        self.assertFalse(self.banks[0].entries)
        self.assertIs(self.context.executor, old_executor)
        self.assertEqual(self.context.executors, old_executors)

    def test_graph_candidate_is_preserved(self):
        self.context.sana.candidate = "transformer_graph"
        result = module.run_candidate_resume(self.context)
        self.assertTrue(result["passed"])
        self.assertEqual(self.context.sana.candidate, "transformer_graph")

    def test_denominator_or_numerical_mismatch_never_qualifies(self):
        changes = [lambda r:r["steps"].pop(),
            lambda r:r["steps"][4].update(rng=False),
            lambda r:r["interruptions"].pop(),
            lambda r:r["interruptions"][1]["image_service"].update(candidate="eager"),
            lambda r:r["final"].update(passed=False),
            lambda r:r["final"].update(atol=.001),
            lambda r:r["cache_paths"].update(paths_equal=False),
            lambda r:r["artifacts"].pop()]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.context.output = Path(self.temp.name)/("mismatch%d" % index)
                value = complete_result()
                change(value)
                with patch.object(module.phases, "_resume_check", return_value=value):
                    result = module.run_candidate_resume(self.context)
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["eligible"])
                self.assertFalse(result["resume_qualified"])
                self.assertTrue(result["cleanup"]["passed"])

    def test_rejects_eager_before_work(self):
        self.context.sana.candidate = "eager"
        with self.assertRaisesRegex(ValueError, "selected"):
            module.run_candidate_resume(self.context)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.saved()["eligible"])

    def test_existing_peer_state_is_never_touched(self):
        peer = SimpleNamespace(residency="gpu")
        self.worker._states[id(peer)] = peer
        with self.assertRaisesRegex(ValueError, "live request"):
            module.run_candidate_resume(self.context)
        self.assertIs(self.worker._states[id(peer)], peer)
        self.assertEqual(self.released, [])
        self.assertTrue(self.saved()["cleanup"]["skipped"])

    def test_prepare_failure_cleans_orphan_and_preserves_partial_and_error(self):
        orphan = SimpleNamespace(residency="gpu")
        def fail(context, evidence, bank, item, executor, **unused):
            self.worker._states[id(orphan)] = orphan
            evidence.data["active_unit"]["completed_steps"] = 0
            raise RuntimeError("second prepare failed")
        with patch.object(module.phases, "_resume_check", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "second prepare"):
                module.run_candidate_resume(self.context)
        self.assertFalse(self.worker._states)
        self.assertEqual(self.released, [orphan])
        self.assertFalse(self.banks[0].entries)
        self.assertEqual(self.saved()["active_unit"]["completed_steps"], 0)
        self.assertTrue(self.saved()["cleanup"]["passed"])

    def test_budget_or_cancel_mid_run_keeps_committed_steps_and_cleans(self):
        for index, exception in enumerate((BudgetStop("parent gone"), KeyboardInterrupt())):
            self.context.output = Path(self.temp.name)/("interrupt%d" % index)
            continuation = SimpleNamespace(closed=False)
            base = SimpleNamespace(residency="gpu")
            def fail(context, evidence, bank, item, executor, **unused):
                self.worker._states[id(base)] = base
                executor._states[id(continuation)] = continuation
                evidence.data["active_unit"].update(completed_steps=8, step_checks=[{"step":i+1} for i in range(8)])
                raise exception
            with patch.object(module.phases, "_resume_check", side_effect=fail):
                with self.assertRaises(type(exception)):
                    module.run_candidate_resume(self.context)
            saved = self.saved()
            self.assertEqual(saved["active_unit"]["completed_steps"], 8)
            self.assertFalse(saved["eligible"])
            self.assertTrue(saved["cleanup"]["passed"])
            self.assertTrue(continuation.closed)
            self.assertFalse(self.worker._states)
            self.assertEqual(self.context.sana.candidate, "vae_compile")

    def test_cleanup_failure_cannot_report_success(self):
        base = SimpleNamespace(residency="gpu")
        def leak(*args, **kwargs):
            self.worker._states[id(base)] = base
            return complete_result()
        self.worker.release = lambda unused: (_ for _ in ()).throw(RuntimeError("release failed"))
        with patch.object(module.phases, "_resume_check", side_effect=leak):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                module.run_candidate_resume(self.context)
        self.assertFalse(self.saved()["resume_qualified"])
        self.assertFalse(self.saved()["cleanup"]["passed"])

    def test_manifest_error_still_cleans_private_states(self):
        base = SimpleNamespace(residency="gpu")
        def leak(*args, **kwargs):
            self.worker._states[id(base)] = base
            return complete_result()
        with patch.object(module.phases, "_resume_check", side_effect=leak), \
             patch.object(Bank, "manifest", side_effect=RuntimeError("manifest failed")):
            with self.assertRaisesRegex(RuntimeError, "manifest failed"):
                module.run_candidate_resume(self.context)
        self.assertFalse(self.worker._states)
        self.assertFalse(self.banks[0].entries)
        self.assertTrue(self.saved()["cleanup"]["passed"])
        self.assertFalse(self.saved()["resume_qualified"])

    def test_candidate_change_inside_helper_is_rejected(self):
        def change(*args, **kwargs):
            self.context.sana.candidate = "eager"
            return complete_result()
        with patch.object(module.phases, "_resume_check", side_effect=change):
            result = module.run_candidate_resume(self.context)
        self.assertFalse(result["eligible"])
        self.assertIn("selected_candidate_unchanged", result["gate"]["failed_checks"])

    def test_original_evidence_is_not_overwritten(self):
        module.run_candidate_resume(self.context)
        before = (self.context.output / "sana-wan-resume.json").read_bytes()
        with self.assertRaises(FileExistsError):
            module.run_candidate_resume(self.context)
        self.assertEqual(before, (self.context.output / "sana-wan-resume.json").read_bytes())

    def test_fresh_import_does_not_load_frameworks(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(Path(module.__file__).resolve().parents[2]))
        result = subprocess.run([sys.executable, "-c",
            "import sys; import ditserve.optimizations.next_resume; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
            env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
