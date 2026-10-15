import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ditserve.directions.direction1 import frozen_requests
from ditserve.optimizations import next_workloads as module
from ditserve.optimizations.online import request_identity


class Fixture:
    def __init__(self, root):
        self.root = root
        self.source = root / "old.json"
        self.costs = dict(video=1000., image=100.)
        self.rows = []
        self.ids = []
        for seed, pattern in ((102, "burst"), (101, "ordinary")):
            requests = frozen_requests(count=24, seed=seed, load=.5, pattern=pattern,
                video_ms=self.costs["video"], image_ms=self.costs["image"])
            identity = request_identity(requests)
            self.ids.append(identity)
            for request in requests:
                request.update(status="completed", start_ms=request["arrival_ms"],
                               finish_ms=request["arrival_ms"]+request["service_ms"])
            self.rows.append(dict(seed=seed, load=.5, pattern=pattern, policy="block-4",
                                 trace_sha256=identity, requests=requests))
        self.write()
        self.config = dict(old_online=str(self.source))

    def write(self):
        self.source.write_text(json.dumps(dict(frozen_service_ms=self.costs, rows=self.rows)))

    def build(self):
        with patch.object(module, "FROZEN_TRACE_IDS", tuple(self.ids)):
            return module.build_workloads(self.config)

    def context(self):
        worker = SimpleNamespace(_states={}, encode_prompt=Mock(side_effect=lambda spec:dict(spec)),
            prepare_encoded=Mock(side_effect=lambda spec,payload:(dict(spec),payload)))
        off, paired = SimpleNamespace(_states={}), SimpleNamespace(_states={})
        ctx = SimpleNamespace(output=self.root, worker=worker, executor=off,
            executors=dict(cache_factorial_off=off, cache_factorial_paired=paired),
            check_budget=Mock(), metadata=dict(wan_warmed=True, cache_factorial_warmed=True,
                                               sana_warmed=True),
            config=self.config, opts=dict(prompts=["private warmup"]),
            sana=SimpleNamespace(set_backend=Mock(), steady_context=contextlib.nullcontext),
            save_partial=Mock(), prepare_online_arm=Mock())
        return ctx


def measured(context, requests, options):
    rows=[]; events=[]
    for request in requests:
        row=dict(request, arrived=True, status="completed", start_ms=request["arrival_ms"],
                 finish_ms=request["arrival_ms"]+request["service_ms"])
        rows.append(row)
        if row["kind"] == "image":
            events.append(dict(kind="image_stage",stage="vae_decode",start_ms=row["start_ms"],
                               end_ms=row["start_ms"]+2.,cuda_elapsed_ms=1.5))
    result=dict(status="completed",complete=True,requests=rows,events=events,
                trace_sha256=request_identity(rows),elapsed_ms=max(r["finish_ms"] for r in rows)+1.)
    context.save_partial(result)
    return result


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture=Fixture(Path(self.directory.name))

    def test_diagnosis_retains_all_image_fields_and_original_arrival_clock(self):
        original=copy.deepcopy(self.fixture.rows)
        result=self.fixture.build()
        self.assertEqual(len(result["diagnosis"]),2)
        for pair in result["diagnosis"]:
            mixed,images=pair["mixed"],pair["image_only"]
            self.assertEqual(len(mixed["requests"]),24)
            self.assertEqual(len(images["requests"]),22)
            self.assertEqual(module._inputs(images["requests"]),module._inputs([
                r for r in mixed["requests"] if r["kind"]=="image"]))
            self.assertNotEqual(images["trace_sha256"],mixed["trace_sha256"])
            self.assertEqual(images["trace_sha256"],request_identity(images["requests"]))
            self.assertEqual(images["parent_trace_sha256"],mixed["trace_sha256"])
            self.assertTrue(all("prompt" not in r for r in images["requests"]))
        self.assertEqual(original,self.fixture.rows)
        self.assertGreater(result["diagnosis"][1]["image_only"]["requests"][0]["arrival_ms"],0.)

    def test_all_18_traces_use_original_reference_costs_not_candidate_speed(self):
        self.fixture.config["candidate_speed"] = dict(video=1.,image=.001)
        result=self.fixture.build(); identities=set()
        for load,traces in result["highload"].items():
            self.assertEqual(len(traces),6)
            self.assertEqual({(t["seed"],t["pattern"]) for t in traces},
                             {(s,p) for s in (101,102,103) for p in ("ordinary","burst")})
            for trace in traces:
                self.assertEqual(trace["load"],float(load)); identities.add(trace["trace_sha256"])
                for request in trace["requests"]:
                    cost=self.fixture.costs[request["kind"]]
                    self.assertEqual(request["service_ms"],cost)
                    self.assertAlmostEqual(request["deadline_ms"]-request["arrival_ms"],cost*2.5)
        self.assertEqual(len(identities),18)
        self.assertEqual(result,self.fixture.build())
        for pair in result["diagnosis"]:
            trace=next(t for t in result["highload"]["0.5"] if
                       (t["seed"],t["pattern"])==(pair["seed"],pair["pattern"]))
            self.assertEqual(trace["trace_sha256"],pair["parent_trace_sha256"])

    def test_source_and_calibration_bindings(self):
        result=self.fixture.build()
        self.assertEqual(result["source"]["sha256"],hashlib.sha256(self.fixture.source.read_bytes()).hexdigest())
        self.assertEqual(result["calibration"]["frozen_service_ms"],self.fixture.costs)
        self.assertEqual(result["generator"]["deadline_multiplier"],2.5)
        self.assertEqual(result["generator"]["sha256"],hashlib.sha256(Path(result["generator"]["path"]).read_bytes()).hexdigest())

    def test_rejects_source_hash_and_explicit_calibration_mismatch(self):
        for options,message in ((dict(expected_source_sha256="wrong"),"SHA256"),
                                (dict(expected_calibration=dict(video=1.,image=1.)),"calibration")):
            self.fixture.config["next_workloads"]=options
            with self.assertRaisesRegex(ValueError,message): self.fixture.build()

    def test_rejects_matrix_changes(self):
        for key,value in (("seeds",[101]),("patterns",["burst"]),("loads",[.9]),("count",12)):
            self.fixture.config["next_workloads"]={key:value}
            with self.assertRaisesRegex(ValueError,"matrix"): self.fixture.build()

    def test_rejects_nonpositive_or_nonfinite_calibration(self):
        for cost in (0.,-1.,float('nan'),float('inf'),True):
            self.fixture.costs["image"]=cost;self.fixture.write()
            with self.assertRaisesRegex(ValueError,"calibration"):self.fixture.build()

    def test_rejects_source_request_hash_mismatch(self):
        self.fixture.rows[0]["requests"][1]["deadline_ms"]+=1.;self.fixture.write()
        with self.assertRaisesRegex(ValueError,"digest"):self.fixture.build()

    def test_rejects_consistent_but_recalibrated_historical_deadline(self):
        row=self.fixture.rows[0];row["requests"][1]["deadline_ms"]+=1.
        row["trace_sha256"]=request_identity(row["requests"])
        self.fixture.ids[0]=row["trace_sha256"];self.fixture.write()
        with self.assertRaisesRegex(ValueError,"generator/calibration"):self.fixture.build()

    def test_import_does_not_load_model_frameworks(self):
        # Other CPU fixtures may legitimately have imported Torch already.
        # Check this module's import contract in a fresh process instead.
        result = subprocess.run([sys.executable, "-c",
            "import sys; import ditserve.optimizations.next_workloads; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
                     PYTHONPATH=str(Path(module.__file__).resolve().parents[2])),
            capture_output=True, text=True, timeout=30.)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.fixture=Fixture(Path(self.directory.name));self.workloads=self.fixture.build()
        self.ctx=self.fixture.context()
        self.identities=patch.object(module,"FROZEN_TRACE_IDS",tuple(self.fixture.ids))
        self.identities.start();self.addCleanup(self.identities.stop)

    def test_diagnosis_eight_replays_same_images_no_overall_goodput_gate(self):
        original_executor=self.ctx.executor;original_partial=self.ctx.save_partial;seen=[]
        def run(context,requests,options):
            seen.append((copy.deepcopy(requests),dict(options)))
            return measured(context,requests,options)
        with patch.object(module,"run_trace",side_effect=run),patch.object(module,"paired_gate",side_effect=AssertionError("diagnosis is not promotion")):
            result=module.run_diagnosis(self.ctx,self.workloads)
        self.assertEqual(result["status"],"completed");self.assertTrue(result["evaluation_complete"])
        self.assertFalse(result["eligible"]);self.assertEqual(len(result["rows"]),8)
        self.assertEqual(sum(len(x[0]) for x in seen),184)
        self.assertEqual([(r["order_index"],r["arm"]) for r in result["rows"][:4]],
                         [(0,"image_only"),(0,"mixed"),(1,"mixed"),(1,"image_only")])
        self.assertIs(self.ctx.executor,original_executor);self.assertIs(self.ctx.save_partial,original_partial)
        for summary in result["summaries"]:
            self.assertEqual(summary["arms"]["image_only"]["arrived"],44)
            self.assertEqual(summary["arms"]["mixed"]["arrived"],44)
            self.assertNotIn("goodput_rps",summary["arms"]["image_only"])
            self.assertEqual(summary["arms"]["image_only"]["stages"]["vae_decode"]["cuda_ms"]["n"],44)
        self.assertEqual(len(list((self.ctx.output/"partials"/"diagnosis").glob("*/partial.json"))),8)

    def test_diagnosis_rejects_image_subset_mutation_before_work(self):
        pair=self.workloads["diagnosis"][0]
        pair["image_only"]["requests"][0]["arrival_ms"]+=1
        pair["image_only"]["trace_sha256"]=request_identity(pair["image_only"]["requests"])
        with patch.object(module,"run_trace") as run:
            with self.assertRaisesRegex(ValueError,"subset"):module.run_diagnosis(self.ctx,self.workloads)
            run.assert_not_called()

    def test_diagnosis_partials_do_not_rewrite_history(self):
        def run(context,requests,options):
            source=context.output/"diagnosis.json";before=source.read_bytes()
            context.save_partial(dict(status="running",requests=[],events=[dict(kind="fixture")]))
            context.save_partial(dict(status="running",requests=[],events=[]))
            self.assertEqual(before,source.read_bytes())
            return measured(context,requests,options)
        with patch.object(module,"run_trace",side_effect=run):module.run_diagnosis(self.ctx,self.workloads)

    def test_failure_retains_active_arrived_denominator_and_completed_history(self):
        calls=[]
        def run(context,requests,options):
            calls.append(1)
            if len(calls)==1:return measured(context,requests,options)
            partial=[dict(r,status="failed" if i==0 else "interrupted",arrived=True) for i,r in enumerate(requests)]
            context.save_partial(dict(status="interrupted",complete=False,requests=partial,events=[]))
            raise RuntimeError("injected cancellation")
        with patch.object(module,"run_trace",side_effect=run):
            with self.assertRaisesRegex(RuntimeError,"cancellation"):module.run_diagnosis(self.ctx,self.workloads)
        record=json.loads((self.ctx.output/"diagnosis.json").read_text())
        self.assertEqual(record["status"],"incomplete");self.assertEqual(len(record["rows"]),1)
        self.assertEqual(record["incomplete_trace"]["arrived"],24)
        self.assertEqual(record["incomplete_trace"]["completed"],0)
        self.assertEqual(len(json.loads(Path(record["incomplete_trace"]["path"]).read_text())["requests"]),24)
        self.assertFalse(record["evaluation_complete"])

    def test_budget_interrupt_keeps_partial_and_does_not_promote(self):
        def run(context,requests,options):
            context.save_partial(dict(status="interrupted",complete=False,requests=[dict(requests[0],arrived=True,status="interrupted")],events=[]))
            raise KeyboardInterrupt("budget/cancel")
        with patch.object(module,"run_trace",side_effect=run):
            with self.assertRaises(KeyboardInterrupt):module.run_diagnosis(self.ctx,self.workloads)
        record=json.loads((self.ctx.output/"diagnosis.json").read_text())
        self.assertEqual(record["incomplete_trace"]["arrived"],1);self.assertFalse(record["eligible"])

    def test_cache_six_traces_abba_options_and_matching_cached_specs(self):
        seen=[]
        def run(context,requests,options):
            info=json.loads(next((context.output/"partials"/"cache_07").glob("replay-%03d/partial.json" % len(seen))).read_text())
            cached=info["arm"]=="candidate"
            self.assertIs(context.executor,self.ctx.executors["cache_factorial_paired" if cached else "cache_factorial_off"])
            self.assertEqual(options["policy"],"block-4");self.assertEqual(options["encode_device"],"cpu")
            self.assertFalse(options["encode_on_scheduler"]);self.assertEqual(options["image_batch_size"],1)
            spec=dict(request_id="already-arrived",cache=False,prompt="payload",threshold=0.,max_continuous=1)
            payload=options["encode_video"](spec);prepared,payload2=options["prepare_video"](spec,payload)
            self.assertEqual(payload,prepared);self.assertEqual(payload,payload2);self.assertEqual(payload["cache"],cached)
            seen.append(info)
            return measured(context,requests,options)
        traces=self.workloads["highload"]["0.7"]
        with patch.object(module,"run_trace",side_effect=run),patch.object(module,"paired_gate",wraps=module.paired_gate) as gate:
            result=module.run_cache_load(self.ctx,traces,"cache_07")
        self.assertEqual(len(result["rows"]),24);self.assertEqual(sum(len(r["requests"]) for r in result["rows"]),576)
        self.assertTrue(result["evaluation_complete"]);self.assertFalse(result["eligible"])
        self.assertEqual(gate.call_args.kwargs["expected_trace_ids"],[t["trace_sha256"] for t in traces])
        self.assertEqual([(r["order_index"],r["arm"]) for r in seen[:4]],[(0,"baseline"),(0,"candidate"),(1,"candidate"),(1,"baseline")])

    def test_cache_rejects_missing_seed_mixed_load_and_bad_identity(self):
        valid=self.workloads["highload"]["0.5"]
        bad=copy.deepcopy(valid);bad[0]["load"]=.7
        mutated=copy.deepcopy(valid);mutated[0]["requests"][0]["seed"]+=1
        for traces in (valid[:-1],bad,mutated):
            with patch.object(module,"run_trace") as run:
                with self.assertRaises(ValueError):module.run_cache_load(self.ctx,traces,"invalid")
                run.assert_not_called()

    def test_cache_rejects_rehashed_deadline_and_candidate_recalibration(self):
        deadline=copy.deepcopy(self.workloads["highload"]["0.5"])
        deadline[0]["requests"][1]["deadline_ms"]+=10.
        deadline[0]["trace_sha256"]=request_identity(deadline[0]["requests"])
        calibration=copy.deepcopy(self.workloads["highload"]["0.5"])
        calibration[0]["calibration"]["image"]=1.
        for traces in (deadline,calibration):
            with patch.object(module,"run_trace") as run:
                with self.assertRaisesRegex(ValueError,"calibration"):module.run_cache_load(self.ctx,traces,"invalid")
                run.assert_not_called()

    def test_requires_residency_warmup_and_no_live_request_leak(self):
        self.ctx.metadata["wan_warmed"]=False
        with self.assertRaisesRegex(ValueError,"warmup"):module.run_diagnosis(self.ctx,self.workloads)
        self.ctx.metadata["wan_warmed"]=True;self.ctx.metadata["cache_factorial_warmed"]=False
        with self.assertRaisesRegex(ValueError,"warmup"):module.run_cache_load(self.ctx,self.workloads["highload"]["0.5"],"load")
        self.ctx.worker._states["leaked"]=object()
        with self.assertRaisesRegex(RuntimeError,"leaked"):module.run_diagnosis(self.ctx,self.workloads)

    def test_no_encoding_during_workload_build_or_option_construction(self):
        self.ctx.online_defaults=dict(encode_device="cuda:1",wan_backend="compiled",vae_backend="compiled")
        for cached in (False,True):
            options=module._fixed_options(self.ctx,cached=cached)
            self.assertEqual(options["encode_device"],"cpu")
            self.assertEqual(options["wan_backend"],"eager")
            self.assertEqual(options["vae_backend"],"eager")
        self.ctx.worker.encode_prompt.assert_not_called();self.ctx.worker.prepare_encoded.assert_not_called()

    def test_evidence_is_not_overwritten(self):
        (self.ctx.output/"diagnosis.json").write_text("existing")
        with self.assertRaises(FileExistsError):module.run_diagnosis(self.ctx,self.workloads)
        (self.ctx.output/"cache_05.json").write_text("existing")
        with self.assertRaises(FileExistsError):module.run_cache_load(self.ctx,self.workloads["highload"]["0.5"],"cache_05")


class BoundsTests(unittest.TestCase):
    def test_optimistic_equal_deadline_bound_and_stage_specific_statistics(self):
        rows=[dict(request_id=str(i),kind="image",arrival_ms=10.,deadline_ms=810.,start_ms=10.+i*300,
                   finish_ms=310.+i*300,status="completed",arrived=True) for i in range(5)]
        row=dict(requests=rows,elapsed_ms=1510.,events=[])
        bound=module.burst_bounds(row)[0]
        self.assertEqual(bound["max_completions_given_observed_costs"],2)
        self.assertEqual(bound["required_average_service_ms"],160.)
        self.assertFalse(bound["all_feasible_given_observed_costs"])
        self.assertEqual(bound["measured_on_time"],2)
        rows[-1].update(status="failed",finish_ms=None)
        bound=module.burst_bounds(row)[0]
        self.assertIsNone(bound["max_completions_given_observed_costs"])
        summary=module._image_summary([row])
        self.assertEqual(summary["arrived"],5);self.assertEqual(summary["failed_or_incomplete"],1)
        self.assertEqual(summary["full_service_ms"]["n"],4)


if __name__ == "__main__":
    unittest.main()
