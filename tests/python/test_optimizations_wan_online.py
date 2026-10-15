"""Native baseline identity survives the actual worker online orchestrator."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from ditserve.directions.block_executor import WanBlockExecutor
from ditserve.optimizations.wan import NativeWanExecutor
from ditserve.optimizations.worker import _online
from test_optimizations_wan_phases import Budget


class OnlineNativeBaselineTests(unittest.TestCase):
    def test_online_keeps_native_eager_in_existing_map_and_default_map(self):
        for existing in (False,True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as directory:
                with patch.object(WanBlockExecutor,"__init__",return_value=None):
                    native=NativeWanExecutor(object())
                context=SimpleNamespace(
                    output=Path(directory),run_root=Path(directory),config=dict(old_online="unused"),
                    opts=dict(prompts=["kettle","pendulum"]),metadata=dict(wan_warmed=True),
                    executor=object() if existing else native,check_budget=Budget(),
                    sana=SimpleNamespace(set_backend=lambda name:None,
                        generate_batch=lambda *args,**kwargs:([],{}),steady_context=nullcontext))
                if existing:
                    context.executors={"eager":native}
                trace=dict(seed=101,load=.5,pattern="ordinary",trace_sha256="fixed",
                    requests=[dict(arrival_ms=0.,service_ms=1.,kind="video")])
                seen=[]
                def measured(ctx,requests,options):
                    seen.append(ctx.executor)
                    self.assertEqual(options["wan_backend"],"eager")
                    self.assertIs(ctx.executor.advance.__func__,WanBlockExecutor.advance)
                    return dict(complete=True)
                defaults=dict(sana_backend="eager",image_batch_size=1,wan_backend="eager")
                with patch("ditserve.optimizations.worker.selected",return_value=defaults), \
                     patch("ditserve.optimizations.online.load_frozen_traces",return_value=[trace]), \
                     patch("ditserve.optimizations.online.run_trace",side_effect=measured), \
                     patch("ditserve.optimizations.wan.WanCompiledExecutor",
                           side_effect=AssertionError("online eager cannot create a wrapped baseline")):
                    result={}
                    _online(context,result,baseline_only=True)
                self.assertEqual(seen,[native])
                self.assertIs(context.executors["eager"],native)
                self.assertFalse(result["online_qualified"])


if __name__=="__main__":
    unittest.main()
