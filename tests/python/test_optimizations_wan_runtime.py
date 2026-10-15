"""Real tiny CPU Wan trajectories exercise production phase/executor interfaces."""

import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.directions.block_executor import WanBlockExecutor
from ditserve.directions.budget import BudgetStop
from ditserve.optimizations.wan import NativeWanExecutor, WanCompiledExecutor
from ditserve.optimizations import wan_phases as phases
from test_directions_block_executor import fixture
from test_optimizations_cache_executor import clone_request, prepared
from test_optimizations_wan_phases import Budget


class Event:
    def __init__(self, **unused):
        pass
    def record(self):
        pass
    def elapsed_time(self, other):
        return 0.  # CUDA clock intentionally unavailable in this CPU fixture.


class RuntimeInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def make(self, cache=False, native=False):
        ids = itertools.count()
        if cache:
            worker, template, executor, old = prepared(boundary=0)
            executor.close(old)
            prepare = lambda spec: clone_request(worker, template, "actual-%d" % next(ids))
        else:
            worker, unused, native_prepare = fixture(steps=32)
            with patch.object(WanBlockExecutor,"_validate_model"):
                executor = (NativeWanExecutor if native else WanCompiledExecutor)(worker)
            def prepare(spec):
                state = native_prepare(spec["request_id"])
                state.generator = self.torch.Generator().manual_seed(spec["seed"])
                return state
            def release(state):
                state.residency = "released"
                del worker._states[state.spec["request_id"]]
            worker.release = release
        bank = SimpleNamespace(prepare=prepare,warm=lambda spec: None)
        def finish(state):
            state.finished = True
            return state.latents.detach().permute(0,2,3,4,1).numpy().copy()
        worker.finish = finish
        interrupts = []
        def image(prompt, **kwargs):
            interrupts.append(kwargs["seed"])
            self.torch.rand(3)  # Other request's RNG must not disturb Wan state.
            return None,dict(ms=1.)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        context = SimpleNamespace(worker=worker,sana=SimpleNamespace(generate=image),
            config={},output=Path(temp.name)/"phase",run_root=Path(temp.name),
            check_budget=Budget(),metadata={})
        evidence = phases._Evidence(context,"runtime")
        item = dict(input_id="tiny101",prompt="tiny",seed=101)
        return context,evidence,bank,item,executor,interrupts

    def test_real_phase_resume_interfaces_cache_off_and_paired(self):
        for cache,native in ((False,False),(False,True),(True,False)):
            with self.subTest(cache=cache,native=native):
                context,evidence,bank,item,executor,interrupts=self.make(cache,native)
                evidence.begin(kind="correctness")
                with patch.object(self.torch.cuda,"Event",Event):
                    result=phases._resume_check(context,evidence,bank,item,executor,cache=cache)
                self.assertTrue(result["passed"])
                self.assertEqual(len(result["steps"]),32)
                self.assertTrue(all(row["solver"] and row["rng"] and row["cache_state"] for row in result["steps"]))
                self.assertEqual(interrupts,[101,101,101])
                self.assertIsNone(result["dispatch_observation"]["host_dispatch_fraction"])
                if native:
                    self.assertIsNone(result["dispatch_observation"]["pure_host_control_ms"])
                self.assertEqual(len(result["artifacts"]),2)
                self.assertTrue(all(Path(row["path"]).is_file() for row in result["artifacts"]))
                self.assertEqual(executor._states,{})

    def test_real_generation_failure_releases_continuation_and_worker_state(self):
        context,evidence,bank,item,executor,interrupts=self.make(False)
        evidence.begin(kind="generation")
        with patch.object(executor.backend,"run",side_effect=BudgetStop("in real block")):
            with self.assertRaises(BudgetStop):
                phases._generate(context,evidence,bank,item,executor)
        self.assertEqual(executor._states,{})
        self.assertEqual(context.worker._states,{})

    def test_interrupted_correctness_has_last_committed_step_on_disk(self):
        import json
        context,evidence,bank,item,executor,interrupts=self.make(native=True)
        evidence.begin(kind="correctness")
        calls=[0]
        def image(*args,**kwargs):
            calls[0]+=1
            if calls[0]==2:
                raise BudgetStop("during step9 image interruption")
            return None,dict(ms=1.)
        context.sana.generate=image
        with patch.object(self.torch.cuda,"Event",Event):
            with self.assertRaises(BudgetStop):
                phases._resume_check(context,evidence,bank,item,executor)
        saved=json.loads(evidence.path.read_text())
        self.assertTrue(saved["active_unit"]["started"])
        self.assertEqual(saved["active_unit"]["completed_steps"],8)
        self.assertEqual(len(saved["active_unit"]["step_checks"]),8)
        self.assertEqual(context.worker._states,{})
        self.assertEqual(executor._states,{})


if __name__=="__main__":
    unittest.main()
