import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.optimizations import next_encoder
from ditserve.research.common import atomic_json


class Worker:
    def __init__(self):
        self._states = {}
        self.released = []
        self.fail_prepare = False
    def prepare_encoded(self, request, payload):
        self.request = request
        state = SimpleNamespace(step_index=0, spec=request)
        self._states[id(state)] = state
        if self.fail_prepare:
            raise RuntimeError("prepare OOM")
        return state
    def finish(self, state):
        return SimpleNamespace(shape=(1, 81, 480, 832, 3))
    def _sync(self): pass
    def _memory(self): return {}
    def release(self, state):
        self.released.append(state)
        del self._states[id(state)]


class Executor:
    calls = []
    def __init__(self, worker): self.worker = worker
    def attach(self, state): return SimpleNamespace(base=state, closed=False)
    def advance(self, continuation, blocks, check_budget):
        check_budget()
        self.calls.append(blocks)
        continuation.base.step_index += 1
    def close(self, continuation): continuation.closed = True


def fake_run(context):
    result = dict(status="passed", local_qualified=True, online_qualified=False,
                  rows=[{}]*8, pairs=[{}]*4)
    context.run_online(context, result, {"encode_device": "cuda:1"})
    atomic_json(context.output / "text_encoder.json", result)
    return result


class NextEncoderTests(unittest.TestCase):
    def test_paired_local_generation_and_release(self):
        worker = Worker()
        context = SimpleNamespace(worker=worker, check_budget=lambda: None)
        evidence = SimpleNamespace(data={"active_unit": {}})
        Executor.calls = []
        with patch("ditserve.optimizations.cache_executor.PairedCacheExecutor", Executor), \
             patch("ditserve.optimizations.stage_phases._record_full_warmup") as warmed:
            array, row = next_encoder.paired_generate(context, evidence, {"steps":32},
                lambda request: SimpleNamespace(cpu_prompt_encode_ms=7))
        self.assertEqual(Executor.calls, [4]*32)
        self.assertTrue(worker.request["cache"])
        self.assertEqual(worker.request["threshold"], .08)
        self.assertEqual(row["completed_steps"], 32)
        self.assertEqual(row["encode_ms"], 7)
        self.assertEqual(worker._states, {})
        warmed.assert_called_once()

    def test_prepare_failure_releases_its_partial_state(self):
        worker = Worker(); worker.fail_prepare = True
        context = SimpleNamespace(worker=worker, check_budget=lambda: None)
        with patch("ditserve.optimizations.cache_executor.PairedCacheExecutor", Executor):
            with self.assertRaisesRegex(RuntimeError, "prepare OOM"):
                next_encoder.paired_generate(context, SimpleNamespace(data={"active_unit":{}}),
                    {"steps":32}, lambda request:SimpleNamespace(cpu_prompt_encode_ms=0))
        self.assertFalse(worker._states)
        self.assertEqual(len(worker.released), 1)

    def test_never_releases_another_requests_state(self):
        worker = Worker(); worker._states[1] = object()
        context = SimpleNamespace(worker=worker, check_budget=lambda: None)
        with self.assertRaisesRegex(ValueError, "ownership"):
            next_encoder.paired_generate(context, None, {}, None)
        self.assertIn(1, worker._states)
        self.assertEqual(worker.released, [])

    def test_callback_bound_and_restored_without_promoting_combination(self):
        from ditserve.optimizations import stage_phases
        with tempfile.TemporaryDirectory() as tmp:
            previous = object()
            candidate_calls = []
            def online(context, result, options, **kwargs):
                candidate_calls.append((options, kwargs))
                result["online_qualified"] = True
            context = SimpleNamespace(output=Path(tmp), worker=Worker(),
                metadata={"encoder_device":"cuda:1"}, admitted_devices=["GPU-a", "GPU-v"],
                config={"gpu_uuid":"GPU-a", "v100_uuid":"GPU-v"},
                sana=SimpleNamespace(set_candidate=lambda value: None), run_online=previous)
            with patch.object(stage_phases, "run_text_encoder", fake_run), \
                 patch.dict(sys.modules, {"ditserve.optimizations.next_conditional":SimpleNamespace(online_callback=online)}):
                result = next_encoder.run_encoder(context)
            self.assertIs(context.run_online, previous)
            self.assertEqual(candidate_calls[0][0], {"encode_device":"cuda:1"})
            self.assertEqual(result["recorded_rows"], 8)
            self.assertEqual(result["decision"], "值得继续")
            self.assertNotIn("selected_defaults", result)
            saved = json.loads((Path(tmp)/"conditional-text-encoder.json").read_text())
            self.assertEqual(saved, result)

    def test_unadmitted_second_gpu_rejected(self):
        context = SimpleNamespace(metadata={"encoder_device":"cuda:1"},
            admitted_devices=["GPU-a"], config={"gpu_uuid":"GPU-a", "v100_uuid":"GPU-v"})
        with patch.dict(sys.modules, {"ditserve.optimizations.next_conditional":SimpleNamespace(online_callback=None)}):
            with self.assertRaisesRegex(ValueError, "charged admission"):
                next_encoder.run_encoder(context)


if __name__ == "__main__":
    unittest.main()
