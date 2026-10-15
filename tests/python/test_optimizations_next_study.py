import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.optimizations.next_phases import profile_summary, graph_trigger
from ditserve.optimizations.next_protocol import safe_paths, PHASES
from ditserve.optimizations import next_study as study


class NextStudyTests(unittest.TestCase):
    def test_profile_does_not_add_overlapping_cuda_host_spans(self):
        rows = [dict(service=dict(ms=100., stages=[
            dict(stage="transformer", host_ms=3., cuda_elapsed_ms=35.),
            dict(stage="transformer", host_ms=3., cuda_elapsed_ms=35.),
            dict(stage="vae_decode", host_ms=10., cuda_elapsed_ms=40.),
            dict(stage="output_postprocess", host_ms=70., cuda_elapsed_ms=5.)]))]
        result = profile_summary(rows)
        self.assertEqual(result["transformer_host_enqueue_fraction"], .06)
        self.assertTrue(result["graph_probe_trigger"])
        self.assertEqual(result["full_service"]["mean"], 100.)

    def test_missing_profile_does_not_trigger(self):
        self.assertFalse(profile_summary([])["graph_probe_trigger"])

    def test_profile_from_failed_isolation_cannot_trigger_candidate(self):
        from ditserve.research.common import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root / "diagnosis/attempt00"
            atomic_json(attempt / "sana-profile.json", dict(status="completed", summary=dict(graph_probe_trigger=True)))
            atomic_json(root / "diagnosis/terminal.json", dict(status="incomplete", returncode=0,
                charge={"gpu_seconds": 1}, output=str(attempt), supervisor_failure={"reason": "foreign GPU"}))
            self.assertFalse(graph_trigger(root)["triggered"])

    def test_no_reuse_of_historical_output_tree(self):
        from ditserve.optimizations.next_protocol import PARENTS
        for parent in PARENTS:
            with self.assertRaises(ValueError):
                safe_paths(parent.parent / "new-run", parent.parent / "gpu-budget.json")

    def test_phase_order_precedes_gpu_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(study, "authorization", return_value=({}, {})), patch.object(study, "inspect_study") as inspect:
                with self.assertRaises(ValueError):
                    study.execute({}, root / "run01", root / "gpu-budget.json", "sana_vae", root / "auth.json")
                inspect.assert_not_called()

    def test_existing_terminal_is_read_without_gpu_or_history_mutation(self):
        from ditserve.research.common import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = dict(status="completed", decision="finite conclusion")
            atomic_json(root / "run01/diagnosis/terminal.json", expected)
            with patch.object(study, "authorization", return_value=({}, {})), patch.object(study, "inspect_study") as inspect:
                self.assertEqual(study.execute({}, root / "run01", root / "gpu-budget.json", "diagnosis", root / "auth.json"), expected)
                inspect.assert_not_called()

    def test_cleanup_releases_private_states(self):
        from types import SimpleNamespace
        class Executor:
            def __init__(self):
                self._states = {1: SimpleNamespace(closed=False)}
            def close(self, state):
                state.closed = True
                self._states.clear()
        executor = Executor()
        worker = SimpleNamespace(_states={1: "private"}, release=lambda state: worker._states.clear(),
            _memory=lambda: {}, torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda device: 1)), device="cuda:0")
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            context = SimpleNamespace(worker=worker, executor=executor, executors={"same": executor},
                sana=SimpleNamespace(close=lambda: calls.append("close")), output=Path(tmp))
            study._cleanup(context)
        self.assertFalse(executor._states)
        self.assertFalse(worker._states)
        self.assertEqual(calls, ["close"])


if __name__ == "__main__":
    unittest.main()
