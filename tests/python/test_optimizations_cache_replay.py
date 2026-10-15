"""Portable metadata/source fixtures; deliberately no arrays, models or GPU."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.optimizations import cache_replay as replay
from ditserve.optimizations.wan_phases import _cache_paths, _cache_resume_paths, _cache_coverage
from ditserve.research.common import digest, file_digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CacheReplayTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        change = patch.object(replay, "ROOT", self.root)
        change.start()
        self.addCleanup(change.stop)
        self.run = self.root / "results/optimization-screen/run01"
        self.attempt = self.run / "cache_k4/attempt00"
        self.path = self.attempt / "supervisor.json"
        self.local_path = self.attempt / "cache-k4.json"
        self.ledger_path = self.run.parent / "gpu-budget.json"
        self.snapshot = self.run / "sources/cache_k4-attempt00"
        self.config = dict(gpu_uuid="a100", quality=dict(image_lpips=.01, video_lpips=.05, rtol=1e-5, atol=1e-6),
                           metrics=dict(minimum_local_gain=.05))
        self.config_sha = digest(self.config)
        self.catalog = dict(prompts=[dict(id=name, text=name+" prompt", split=split)
            for split, names in (("dev", ("kettle", "pendulum", "origami", "ball")),
                                 ("holdout", ("garden", "train"))) for name in names])
        hashes = {}
        for name in replay._REQUIRED | {"python/ditserve/optimizations/runner.py"}:
            for base in (self.root, self.snapshot):
                path = base / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == "configs/research_cfg.json":
                    write(path, self.catalog)
                elif name == "configs/serving_optimizations.json":
                    write(path, self.config)
                else:
                    path.write_text("# pinned tiny source fixture\n")
            hashes[name] = file_digest(self.root / name)
        self.manifest = dict(phase="cache_k4", attempt="attempt00", execution_root=str(self.snapshot),
                             source_files=hashes, source_sha256=digest(hashes), config_sha256=self.config_sha)
        write(self.snapshot / "manifest.json", self.manifest)
        write(self.attempt / "config.json", self.config)
        write(self.run / "protocol.json", dict(config=self.config, sha256=self.config_sha))
        self.inputs = replay._catalog(self.snapshot)
        self.counter = 0
        self.events = [dict(kind="step", step_index=index, completed_steps=index+1,
            path="reuse" if index in (8, 10) else "compute", reused_forwards=2 if index in (8, 10) else 0)
            for index in range(32)]
        self.local = self.make_local()
        self.charge = dict(session_id="settled-cache", status="completed", stage="cache_k4", gpu_count=1,
            gpu_seconds=1000., elapsed_seconds=1000., ended_at=5000.,
            identity=dict(phase="cache_k4", run_root=str(self.run), output=str(self.attempt),
                config_sha256=self.config_sha, source_sha256=self.manifest["source_sha256"],
                gpu_uuids=["a100"], auxiliary_encoder=False))
        self.ledger = dict(active=None, sessions=[self.charge], used_gpu_seconds=1000.)
        self.save()

    def save(self):
        write(self.local_path, self.local)
        self.supervisor = dict(copy.deepcopy(self.local), phase="cache_k4", output=str(self.attempt),
            config_sha256=self.config_sha, source_sha256=self.manifest["source_sha256"],
            returncode=0, charge=self.charge,
            final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[]))
        write(self.path, self.supervisor)
        write(self.ledger_path, self.ledger)

    def artifact(self):
        self.counter += 1
        return dict(path=str(self.root / "absent-arrays" / (str(self.counter)+".npy")),
                    sha256="a"*64, array_sha256="b"*64, shape=replay._SHAPE, raw_available=True)

    def numerical(self):
        return dict(passed=True, finite=True, exact=True, max_abs=0., rtol=1e-5, atol=1e-6, shape=replay._SHAPE)

    def quality(self):
        return dict(passed=True, metrics=dict(status="ok", finite=True, range_valid=True, lpips_status="ok",
            shape=replay._SHAPE, frame_count=81, lpips_per_frame=[.01]*81, lpips_mean=.01))

    def generation(self, cache, native=False):
        service = dict(ms=80. if cache else 100., completed_steps=32, generation_completed=True,
            output_sha256="b"*64, cache=cache, native_cache=native,
            backend="paired_008_original" if native else "eager", reuse_steps=[9,11] if cache else [])
        if cache:
            service["cache_paths"] = _cache_paths(self.events)
        return dict(service=service, artifact=self.artifact())

    def make_local(self):
        rows=[]
        for item in self.inputs["screen"]:
            name=item["input_id"]
            for native in (True,False):
                rows.append(dict(kind="original_split_equivalence",input_id=name,native=native,
                                 **self.generation(True,native)))
            rows.append(dict(kind="original_split_comparison",input_id=name,numerical=self.numerical(),
                passed=True,cache_paths_equal=True,path_records_valid=True,full_paths_equal=True))
            interruptions=[dict(step=index,phase="prefix",branch="cond",next_block=4,
                                image_service=dict(ms=1.,finite=True)) for index in (0,8,16)]
            paths=_cache_resume_paths(self.events,self.events,interruptions)
            steps=[dict(step=index+1,latent=True,solver=True,cache_state=True,rng=True,passed=True,
                cache_paths_equal=True,continuous_path=paths["continuous"]["path_sequence"][index],
                k4_path=paths["k4"]["path_sequence"][index]) for index in range(32)]
            rows.append(dict(kind="same_backend_resume",input_id=name,result=dict(passed=True,steps=steps,
                interruptions=interruptions,cache_paths=paths,final=self.numerical(),
                artifacts=[self.artifact(),self.artifact()])))
        coverage=_cache_coverage(rows,[item["input_id"] for item in self.inputs["screen"]])
        for order,arms in enumerate((("off","paired"),("paired","off"))):
            for item in self.inputs["screen"]:
                for arm in arms:
                    rows.append(dict(kind="cache_screen",order=order,input_id=item["input_id"],arm=arm,
                        complete=True,quality=self.quality(),**self.generation(arm=="paired")))
        for split,order in (("dev",0),("dev",1),("holdout",0)):
            for item in self.inputs[split]:
                for arm in (("off","paired") if order==0 else ("paired","off")):
                    rows.append(dict(kind="cache_qualification",split=split,order=order,
                        input_id=item["input_id"],arm=arm,complete=True,**self.generation(arm=="paired")))
                rows.append(dict(kind="cache_qualification_pair",split=split,order=order,
                    input_id=item["input_id"],complete=True,quality=self.quality(),off_ms=100.,paired_ms=80.))
        return dict(schema_version="ditserve.optimizations.wan.v1",phase="cache-k4",status="passed",rows=rows,
            active_unit=None,backend="eager",compiled=False,selected_backend="eager-paired-k4",
            local_qualified=True,screen_local_qualified=True,cache_quality_qualified=True,
            full_qualification_completed=True,inputs=self.inputs["screen"],metadata=dict(original=True),
            quality_policy=dict(video_lpips=.05,same_backend_resume_rtol=1e-5,same_backend_resume_atol=1e-6),
            eager_baseline=dict(original_advance=True),cache_path_coverage=coverage,
            screen=dict(complete=True,distinct_development_inputs=3,heldout_inputs=0,quality_passed=True,
                        quality_checked=12,n_per_arm=6,performance_passed=True,gains=[.2,.2]),
            full_qualification=dict(status="passed",complete=True,qualified=True,quality_passed=True,
                performance_passed=True,pairs=20,required_pairs=20,dev_pairs=16,heldout_pairs=4,
                remaining_pairs=0,remaining_units=[],heldout_selection=False,gains=[.2,.2]))

    def inspect(self):
        return replay.inspect_cache_replay(self.path,self.config,self.run)

    def test_complete_source_validates_without_any_array_existing(self):
        result=self.inspect()
        self.assertEqual(result["local_validation"]["qualification_pairs"],20)
        self.assertEqual(result["local_validation"]["recovery_steps"],96)
        self.assertEqual(result["local_validation"]["real_sana_interruptions"],9)
        self.assertFalse(result["local_validation"]["arrays_read"])
        self.assertEqual(result["mathematical_source_sha256"],digest(result["mathematical_source_files"]))
        self.assertFalse((self.root/"absent-arrays").exists())

    def test_missing_duplicate_reordered_or_unevaluated_qualification_is_rejected(self):
        original=copy.deepcopy(self.local)
        for mutation in ("missing","duplicate","reorder","unevaluated"):
            with self.subTest(mutation=mutation):
                self.local=copy.deepcopy(original)
                if mutation=="missing":self.local["rows"].pop()
                elif mutation=="duplicate":self.local["rows"][-1]=copy.deepcopy(self.local["rows"][-4])
                elif mutation=="reorder":self.local["rows"][-3],self.local["rows"][-2]=self.local["rows"][-2],self.local["rows"][-3]
                else:self.local["rows"][-1]["complete"]=False
                self.save()
                with self.assertRaises(ValueError):self.inspect()

    def test_quality_flag_cannot_hide_failed_frame_mean_or_quality_failure(self):
        original=copy.deepcopy(self.local)
        for mutation in ("flag","mean","frames"):
            with self.subTest(mutation=mutation):
                self.local=copy.deepcopy(original)
                quality=self.local["rows"][-1]["quality"]
                if mutation=="flag":quality["passed"]=False
                elif mutation=="mean":quality["metrics"].update(lpips_mean=.06,lpips_per_frame=[.06]*81)
                else:quality["metrics"]["lpips_per_frame"]=[.01]*80
                self.save()
                with self.assertRaisesRegex(ValueError,"quality"):self.inspect()

    def test_both_gain_orders_are_recomputed_from_bound_generation_times(self):
        self.local["full_qualification"]["gains"][1]=.3
        self.save()
        with self.assertRaisesRegex(ValueError,"speed"):self.inspect()

    def test_recovery_requires_all_steps_and_real_fixed_sana_calls(self):
        original=copy.deepcopy(self.local)
        for mutation in ("step","solver","sana","resolved"):
            with self.subTest(mutation=mutation):
                self.local=copy.deepcopy(original)
                result=self.local["rows"][3]["result"]
                if mutation=="step":result["steps"].pop()
                elif mutation=="solver":result["steps"][9]["solver"]=False
                elif mutation=="sana":result["interruptions"][1]["image_service"]["finite"]=False
                else:result["interruptions"][1]["resolved_step_path"]="full"
                self.save()
                with self.assertRaises(ValueError):self.inspect()

    def test_path_flags_cannot_hide_missing_or_mixed_commit(self):
        self.local["rows"][3]["result"]["cache_paths"]["k4"]["path_sequence"][8]["path"]="mixed"
        self.save()
        with self.assertRaisesRegex(ValueError,"cache path"):self.inspect()

    def test_aggregate_cannot_invent_reuse_coverage(self):
        self.local["cache_path_coverage"]["counts"]["resumed_k4"]["reuse"]+=1
        self.save()
        with self.assertRaisesRegex(ValueError,"aggregate"):self.inspect()

    def test_failed_or_contended_supervisor_is_never_reusable(self):
        original=copy.deepcopy(self.supervisor)
        for mutation in ("failed","nonzero","foreign"):
            with self.subTest(mutation=mutation):
                value=copy.deepcopy(original)
                if mutation=="failed":value["status"]="failed"
                elif mutation=="nonzero":value["returncode"]=1
                else:value["final_occupancy"]["foreign"]=[dict(pid=42)]
                write(self.path,value)
                with self.assertRaises(ValueError):self.inspect()

    def test_active_unsettled_or_duplicate_old_charge_is_rejected(self):
        original=copy.deepcopy(self.ledger)
        for mutation in ("active","unsettled","duplicate"):
            with self.subTest(mutation=mutation):
                value=copy.deepcopy(original)
                if mutation=="active":value["active"]=dict(session_id="other")
                elif mutation=="unsettled":value["sessions"]=[]
                else:value["sessions"]*=2
                write(self.ledger_path,value)
                with self.assertRaisesRegex(ValueError,"settled"):self.inspect()

    def test_source_charge_must_bind_config_gpu_and_original_attempt(self):
        self.charge["identity"]["gpu_uuids"]=["v100"]
        self.save()
        with self.assertRaisesRegex(ValueError,"identity"):self.inspect()

    def test_local_and_supervisor_cannot_disagree(self):
        value=copy.deepcopy(self.local)
        value["rows"][-1]["paired_ms"]=81.
        write(self.local_path,value)
        with self.assertRaisesRegex(ValueError,"disagree"):self.inspect()

    def test_changed_math_refused_but_current_controller_may_change(self):
        (self.root/"python/ditserve/optimizations/runner.py").write_text("# new controller\n")
        self.inspect()
        (self.root/"python/ditserve/optimizations/cache_executor.py").write_text("# altered math\n")
        with self.assertRaisesRegex(ValueError,"mathematical source changed"):self.inspect()

    def test_snapshot_tampering_and_changed_config_are_refused(self):
        changed=copy.deepcopy(self.config);changed["quality"]["video_lpips"]=.1
        with self.assertRaisesRegex(ValueError,"config differs"):
            replay.inspect_cache_replay(self.path,changed,self.run)
        (self.snapshot/"python/ditserve/optimizations/runner.py").write_text("# tampered historical controller\n")
        with self.assertRaisesRegex(ValueError,"frozen source changed"):self.inspect()

    def test_loader_writes_reference_only_in_new_run_without_charge_defaults_or_rows(self):
        binding=self.inspect()
        new_run=self.root/"results/optimization-cache-online/run01"
        context=SimpleNamespace(config=self.config,run_root=new_run,output=new_run/"cache_online/attempt00",
                                cache_replay_binding=binding,metadata=dict(wan_warmed=True))
        old=self.local_path.read_bytes()
        result=replay.load_cache_evidence(context,self.path)
        self.assertEqual(result["new_local_executions"],0)
        self.assertEqual(result["rows"],[])
        self.assertEqual(result["local_evidence_reuse"]["original_rows"],84)
        self.assertFalse(result["online_qualified"])
        self.assertTrue(result["cache_quality_qualified"])
        self.assertTrue(result["full_qualification_completed"])
        for key in ("charge","selected_backend","defaults","metadata"):
            self.assertNotIn(key,result)
        self.assertTrue((context.output/"cache-replay.json").is_file())
        self.assertFalse((context.output/"cache-k4.json").exists())
        self.assertEqual(self.local_path.read_bytes(),old)
        with self.assertRaisesRegex(ValueError,"already contains"):
            replay.load_cache_evidence(context,self.path)

    def test_loader_refuses_changed_admission_binding_and_source_output_overlap(self):
        binding=self.inspect()
        new_run=self.root/"results/new/run01"
        context=SimpleNamespace(config=self.config,run_root=new_run,output=new_run/"attempt00",
                                cache_replay_binding=copy.deepcopy(binding))
        context.cache_replay_binding["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"changed after admission"):
            replay.load_cache_evidence(context,self.path)
        context.cache_replay_binding=binding
        context.run_root=self.run;context.output=self.run/"cache_k4/attempt01"
        with self.assertRaisesRegex(ValueError,"overlaps archived"):
            replay.load_cache_evidence(context,self.path)


if __name__=="__main__":
    unittest.main()
